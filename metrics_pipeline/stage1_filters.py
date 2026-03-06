#!/usr/bin/env python3
"""
Stage 1 -- Fast structural sanity filters on renumbered design PDBs.

Reads the regions.json produced by define_regions.py and one or more design PDB
files (already renumbered so that receptor numbering matches the reference).

For each design it computes:
  - n_head_in_pocket:  head residues within contact distance of at least one pocket residue
  - n_head_pocket_pairs: total (head, pocket) residue pairs in contact
  - min_head_pocket_dist: minimum heavy-atom distance across all head-pocket pairs
  - mean_head_pocket_dist: mean of per-pair minimum distances
  - n_interface_contacts: total inter-chain residue pairs within 5 A
  - n_clashes_severe:  inter-chain heavy-atom pairs closer than 1.8 A
  - n_clashes_mild:    inter-chain heavy-atom pairs closer than 2.1 A
  - pass_stage1:       boolean verdict

Default pass criteria (all must be satisfied):
  - n_head_in_pocket  >= 2
  - n_head_pocket_pairs >= 4
  - n_clashes_severe  == 0
  - n_clashes_mild    <= 5

Output: CSV written to stdout or a file.

Example usage:
  # Score a few specific designs
  python metrics_pipeline/stage1_filters.py regions.json \
      renumbered_design_0.pdb renumbered_design_1.pdb -o stage1_scores.csv

  # Score all renumbered designs at once with a glob
  python metrics_pipeline/stage1_filters.py regions.json renumbered_design_*.pdb -o stage1_scores.csv

  # Override pass/fail thresholds
  python metrics_pipeline/stage1_filters.py regions.json renumbered_design_*.pdb \
      --min-head-in-pocket 3 --min-head-pocket-pairs 6 \
      --max-severe-clashes 0 --max-mild-clashes 3 \
      -o stage1_scores.csv

  # Use a tighter contact cutoff (default 5.0 A)
  python metrics_pipeline/stage1_filters.py regions.json renumbered_design_*.pdb \
      --contact-cutoff 4.5 -o stage1_scores.csv

  # Print to stdout instead of writing a file (omit -o)
  python metrics_pipeline/stage1_filters.py regions.json renumbered_design_best.pdb
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa


def _get_receptor_peptide(structure, receptor_id, peptide_id):
    """Return (receptor_chain, peptide_chain) from explicit IDs or length heuristic."""
    receptor, peptide = None, None
    for chain in structure[0]:
        residues = [r for r in chain if is_aa(r, standard=True)]
        if not residues:
            continue
        if chain.id == receptor_id:
            receptor = chain
        elif chain.id == peptide_id:
            peptide = chain
        elif receptor is None and receptor_id is None and len(residues) > 100:
            receptor = chain
        elif peptide is None and peptide_id is None and len(residues) <= 100:
            peptide = chain
    return receptor, peptide


def _aa_residues(chain):
    return [r for r in chain if is_aa(r, standard=True)]


def _heavy_atom_min_dist(res1, res2) -> float:
    """Minimum heavy-atom distance between two residues."""
    best = float("inf")
    for a1 in res1.get_atoms():
        if a1.element == "H":
            continue
        for a2 in res2.get_atoms():
            if a2.element == "H":
                continue
            d = np.linalg.norm(a1.coord - a2.coord)
            if d < best:
                best = d
    return best


def _find_clashes(chain_a, chain_b, cutoff: float) -> Tuple[int, List[Dict]]:
    """
    Find heavy-atom clashes across two chains closer than cutoff.
    Returns (count, list of clash detail dicts).
    """
    atoms_b = [a for r in _aa_residues(chain_b) for a in r.get_atoms() if a.element != "H"]
    if not atoms_b:
        return 0, []
    atom_to_res_b = {id(a): a.get_parent() for a in atoms_b}
    ns = NeighborSearch(atoms_b)
    count = 0
    seen_res_pairs: Set[Tuple[int, int]] = set()
    details: List[Dict] = []
    for res_a in _aa_residues(chain_a):
        for atom_a in res_a.get_atoms():
            if atom_a.element == "H":
                continue
            for hit in ns.search(atom_a.coord, cutoff, level="A"):
                count += 1
                res_b = atom_to_res_b[id(hit)]
                pair_key = (res_a.id[1], res_b.id[1])
                if pair_key not in seen_res_pairs:
                    seen_res_pairs.add(pair_key)
                    d = float(np.linalg.norm(atom_a.coord - hit.coord))
                    details.append({
                        "chain_a": res_a.parent.id,
                        "resnum_a": res_a.id[1],
                        "resname_a": res_a.get_resname(),
                        "atom_a": atom_a.get_name().strip(),
                        "chain_b": res_b.parent.id,
                        "resnum_b": res_b.id[1],
                        "resname_b": res_b.get_resname(),
                        "atom_b": hit.get_name().strip(),
                        "distance": round(d, 2),
                    })
    details.sort(key=lambda x: x["distance"])
    return count, details


def score_design(
    pdb_path: str,
    pocket_resnums: Set[int],
    head_positions: List[int],
    receptor_chain_id: str,
    peptide_chain_id: str,
    contact_cutoff: float = 5.0,
    severe_clash: float = 1.8,
    mild_clash: float = 2.1,
) -> Dict:
    """
    Score a single renumbered design PDB against the reference-derived
    pocket and head definitions.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("design", pdb_path)
    receptor, peptide = _get_receptor_peptide(
        structure, receptor_chain_id, peptide_chain_id
    )

    result: Dict = {"design": Path(pdb_path).name}

    if receptor is None or peptide is None:
        result.update({
            "n_head_in_pocket": 0,
            "n_head_pocket_pairs": 0,
            "min_head_pocket_dist": float("nan"),
            "mean_head_pocket_dist": float("nan"),
            "n_interface_contacts": 0,
            "n_clashes_severe": 0,
            "n_clashes_mild": 0,
            "pass_stage1": False,
            "error": "missing receptor or peptide chain",
        })
        return result

    pep_residues = _aa_residues(peptide)

    # Map head sequence positions to actual residue objects
    head_residues = []
    for pos in head_positions:
        if pos < len(pep_residues):
            head_residues.append(pep_residues[pos])

    # Map pocket resnums to actual receptor residue objects
    pocket_res_objects = [
        r for r in _aa_residues(receptor) if r.id[1] in pocket_resnums
    ]

    # --- head-pocket contact metrics ---
    pair_dists: List[float] = []
    head_touching: Set[int] = set()

    for h_idx, h_res in enumerate(head_residues):
        for p_res in pocket_res_objects:
            d = _heavy_atom_min_dist(h_res, p_res)
            if d <= contact_cutoff:
                pair_dists.append(d)
                head_touching.add(h_idx)

    n_head_in_pocket = len(head_touching)
    n_head_pocket_pairs = len(pair_dists)
    min_dist = float(min(pair_dists)) if pair_dists else float("nan")
    mean_dist = float(np.mean(pair_dists)) if pair_dists else float("nan")

    # --- whole-interface contact count ---
    rec_atoms = [a for r in _aa_residues(receptor) for a in r.get_atoms() if a.element != "H"]
    if rec_atoms:
        ns_rec = NeighborSearch(rec_atoms)
        interface_pairs: Set[Tuple[int, int]] = set()
        for res in _aa_residues(peptide):
            for atom in res.get_atoms():
                if atom.element == "H":
                    continue
                for hit in ns_rec.search(atom.coord, contact_cutoff, level="R"):
                    if not is_aa(hit, standard=True):
                        continue
                    pair = (res.id[1], hit.id[1])
                    interface_pairs.add(pair)
        n_interface = len(interface_pairs)
    else:
        n_interface = 0

    # --- clash detection ---
    n_severe, severe_details = _find_clashes(peptide, receptor, severe_clash)
    n_mild, mild_details = _find_clashes(peptide, receptor, mild_clash)

    def _fmt_clash(c: Dict) -> str:
        return (
            f"{c['chain_a']}:{c['resname_a']}{c['resnum_a']}({c['atom_a']})-"
            f"{c['chain_b']}:{c['resname_b']}{c['resnum_b']}({c['atom_b']}) "
            f"{c['distance']}A"
        )

    result.update({
        "n_head_in_pocket": n_head_in_pocket,
        "n_head_pocket_pairs": n_head_pocket_pairs,
        "min_head_pocket_dist": round(min_dist, 2),
        "mean_head_pocket_dist": round(mean_dist, 2),
        "n_interface_contacts": n_interface,
        "n_clashes_severe": n_severe,
        "n_clashes_mild": n_mild,
        "severe_clashes": "; ".join(_fmt_clash(c) for c in severe_details) if severe_details else "",
        "mild_clashes": "; ".join(_fmt_clash(c) for c in mild_details) if mild_details else "",
    })
    return result


def apply_pass_criteria(
    row: Dict,
    min_head_in_pocket: int = 2,
    min_head_pocket_pairs: int = 4,
    max_severe_clashes: int = 0,
    max_mild_clashes: int = 5,
) -> bool:
    return (
        row.get("n_head_in_pocket", 0) >= min_head_in_pocket
        and row.get("n_head_pocket_pairs", 0) >= min_head_pocket_pairs
        and row.get("n_clashes_severe", 999) <= max_severe_clashes
        and row.get("n_clashes_mild", 999) <= max_mild_clashes
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1: fast structural sanity filters on renumbered design PDBs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("regions_json", help="Path to regions.json from define_regions.py")
    ap.add_argument("design_pdbs", nargs="+", help="One or more renumbered design PDB files")
    ap.add_argument("--contact-cutoff", type=float, default=5.0, help="Contact distance cutoff (A)")
    ap.add_argument("--min-head-in-pocket", type=int, default=2)
    ap.add_argument("--min-head-pocket-pairs", type=int, default=4)
    ap.add_argument("--max-severe-clashes", type=int, default=0)
    ap.add_argument("--max-mild-clashes", type=int, default=5)
    ap.add_argument("-o", "--output", default=None, help="Output CSV (default: stdout)")
    args = ap.parse_args()

    with open(args.regions_json) as f:
        config = json.load(f)

    pocket_resnums = {r["resnum"] for r in config["pocket_residues"]}
    head_positions = config["head_positions"]
    rec_chain = config["receptor_chain"]
    pep_chain = config["peptide_chain"]

    print(
        f"Config: {len(pocket_resnums)} pocket residues, "
        f"{len(head_positions)} head positions, "
        f"chains {rec_chain}/{pep_chain}",
        file=sys.stderr,
    )

    rows = []
    for pdb in args.design_pdbs:
        print(f"  scoring {pdb} ...", file=sys.stderr)
        row = score_design(
            pdb,
            pocket_resnums=pocket_resnums,
            head_positions=head_positions,
            receptor_chain_id=rec_chain,
            peptide_chain_id=pep_chain,
            contact_cutoff=args.contact_cutoff,
        )
        row["pass_stage1"] = apply_pass_criteria(
            row,
            min_head_in_pocket=args.min_head_in_pocket,
            min_head_pocket_pairs=args.min_head_pocket_pairs,
            max_severe_clashes=args.max_severe_clashes,
            max_mild_clashes=args.max_mild_clashes,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {args.output} ({len(df)} designs)", file=sys.stderr)
    else:
        print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
