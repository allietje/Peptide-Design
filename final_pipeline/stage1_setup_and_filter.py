#!/usr/bin/env python3
"""
Stage 1: Region definition + renumbering + structural filters.

Takes a reference PDB and RFDiffusion design PDBs, then:
  1. Defines pocket/head/hotspot regions from the reference complex.
  2. Computes pocket center and depth axis for downstream scoring.
  3. Renumbers each design so receptor residue numbers match the reference.
  4. Scores renumbered designs with structural sanity filters.
  5. Copies passing designs to pass_designs/.

Outputs:
  <outdir>/regions.json        -- pocket, head, hotspot definitions + geometry
  <outdir>/renumbered/         -- renumbered PDB files
  <outdir>/pass_designs/       -- copies of PDBs that passed all filters
  <outdir>/stage1_summary.csv  -- per-design metrics + pass/fail columns

Usage:
  python stage1_setup_and_filter.py reference.pdb designs/*.pdb \\
      --receptor-chain R --peptide-chain P -o results/

  python stage1_setup_and_filter.py reference.pdb designs/*.pdb \\
      --receptor-chain R --peptide-chain P \\
      --hotspot-residues 294,301,305 \\
      --config config.yaml -o results/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa

from utils import (
    aa_residues,
    align_sequences,
    check_threshold,
    find_clashes,
    get_chain_by_id,
    get_chains_by_length,
    get_receptor_peptide,
    get_sequence,
    get_threshold,
    heavy_atom_min_dist,
    heavy_atoms,
    load_config,
    log,
)


# ---------------------------------------------------------------------------
# Region definition (pocket, head, hotspot, geometry)
# ---------------------------------------------------------------------------

def define_regions(
    reference_pdb: str,
    receptor_chain_id: Optional[str] = None,
    peptide_chain_id: Optional[str] = None,
    pocket_cutoff: float = 6.0,
    head_cutoff: float = 5.0,
    hotspot_residues: Optional[List[int]] = None,
) -> Dict:
    """Derive pocket, head, hotspot, and geometry from the reference complex."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", reference_pdb)

    receptor, peptide = get_receptor_peptide(
        structure, receptor_chain_id, peptide_chain_id
    )
    if receptor is None:
        raise ValueError("Could not identify receptor chain")
    if peptide is None:
        raise ValueError("Could not identify peptide chain")

    log(
        f"[Stage 1] Receptor chain {receptor.id} ({len(aa_residues(receptor))} aa), "
        f"Peptide chain {peptide.id} ({len(aa_residues(peptide))} aa)"
    )

    # -- Pocket: receptor residues near any peptide heavy atom --
    pep_atoms = list(heavy_atoms(peptide))
    ns_pep = NeighborSearch(pep_atoms)

    pocket_residues: List[Dict] = []
    pocket_positions: List[int] = []
    pocket_res_set: Set[int] = set()
    rec_residues = aa_residues(receptor)
    for seq_pos, res in enumerate(rec_residues):
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            hits = ns_pep.search(atom.coord, pocket_cutoff, level="A")
            if hits:
                resnum = res.id[1]
                if resnum not in pocket_res_set:
                    pocket_res_set.add(resnum)
                    pocket_positions.append(seq_pos)
                    pocket_residues.append(
                        {"resnum": resnum, "resname": res.get_resname(), "seq_pos": seq_pos}
                    )
                break

    pocket_residues.sort(key=lambda x: x["resnum"])
    pocket_positions.sort()
    log(f"[Stage 1] Pocket: {len(pocket_residues)} receptor residues within {pocket_cutoff} A of peptide")

    # -- Head: peptide residues near pocket --
    pocket_atoms = []
    for res in aa_residues(receptor):
        if res.id[1] in pocket_res_set:
            for atom in res.get_atoms():
                if atom.element != "H":
                    pocket_atoms.append(atom)

    ns_pocket = NeighborSearch(pocket_atoms)

    pep_residues = aa_residues(peptide)
    head_residues: List[Dict] = []
    head_positions: List[int] = []
    for seq_pos, res in enumerate(pep_residues):
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            hits = ns_pocket.search(atom.coord, head_cutoff, level="A")
            if hits:
                head_positions.append(seq_pos)
                head_residues.append(
                    {"resnum": res.id[1], "resname": res.get_resname(), "seq_pos": seq_pos}
                )
                break

    log(f"[Stage 1] Head: {len(head_residues)} peptide residues within {head_cutoff} A of pocket")

    # -- Pocket center and depth axis --
    pocket_ca_coords = []
    for res in aa_residues(receptor):
        if res.id[1] in pocket_res_set and "CA" in res:
            pocket_ca_coords.append(res["CA"].coord)
    pocket_center = np.mean(pocket_ca_coords, axis=0).tolist() if pocket_ca_coords else [0, 0, 0]

    all_rec_ca = [res["CA"].coord for res in aa_residues(receptor) if "CA" in res]
    receptor_centroid = np.mean(all_rec_ca, axis=0) if all_rec_ca else np.zeros(3)
    depth_vec = np.array(pocket_center) - receptor_centroid
    depth_norm = float(np.linalg.norm(depth_vec))
    if depth_norm > 1e-6:
        depth_axis = (depth_vec / depth_norm).tolist()
    else:
        depth_axis = [0.0, 0.0, 1.0]

    log(f"[Stage 1] Pocket center: [{pocket_center[0]:.1f}, {pocket_center[1]:.1f}, {pocket_center[2]:.1f}]")

    # -- Hotspot residues --
    if hotspot_residues:
        valid_hotspots = [r for r in hotspot_residues if r in pocket_res_set]
        if len(valid_hotspots) < len(hotspot_residues):
            outside = set(hotspot_residues) - set(valid_hotspots)
            log(f"[Stage 1] WARNING: hotspot residues {outside} are not in the pocket set, keeping them anyway")
            valid_hotspots = hotspot_residues
        log(f"[Stage 1] Hotspot residues: {len(hotspot_residues)} specified")
    else:
        hotspot_residues = []

    return {
        "reference_pdb": str(Path(reference_pdb).resolve()),
        "receptor_chain": receptor.id,
        "peptide_chain": peptide.id,
        "pocket_cutoff_angstrom": pocket_cutoff,
        "head_cutoff_angstrom": head_cutoff,
        "pocket_residues": pocket_residues,
        "pocket_positions": pocket_positions,
        "head_positions": head_positions,
        "head_residues": head_residues,
        "hotspot_residues": hotspot_residues,
        "pocket_center": pocket_center,
        "depth_axis": depth_axis,
    }


# ---------------------------------------------------------------------------
# Renumbering
# ---------------------------------------------------------------------------

def renumber_design(
    ref_structure,
    des_pdb_path: str,
    output_path: str,
    ref_receptor_id: str,
    ref_peptide_id: str,
) -> None:
    """Renumber one design PDB to match the reference."""
    parser = PDBParser(QUIET=True)
    des_structure = parser.get_structure("design", des_pdb_path)

    ref_receptor = get_chain_by_id(ref_structure, ref_receptor_id)
    ref_peptide = get_chain_by_id(ref_structure, ref_peptide_id)
    if ref_receptor is None:
        raise ValueError(f"Reference receptor chain '{ref_receptor_id}' not found")

    des_receptor, des_peptide = get_chains_by_length(des_structure)

    ref_residues, ref_seq = get_sequence(ref_receptor)
    des_residues, des_seq = get_sequence(des_receptor)

    mapping = align_sequences(ref_seq, des_seq)

    resnum_map: Dict[int, Tuple[int, str]] = {}
    for di, ri in mapping.items():
        if di < len(des_residues) and ri < len(ref_residues):
            ref_res = ref_residues[ri]
            resnum_map[di] = (ref_res.id[1], ref_res.id[2])

    des_rec_orig = des_receptor.id
    des_pep_orig = des_peptide.id

    for model in des_structure:
        for chain in model:
            if chain.id == des_rec_orig:
                chain_res = [r for r in chain if is_aa(r, standard=True)]
                temp_ids: Dict[int, Tuple] = {}
                for idx, res in enumerate(chain_res):
                    if idx in resnum_map:
                        old_id = res.id
                        temp_id = (old_id[0], 100000 + idx, old_id[2])
                        temp_ids[idx] = temp_id
                        res.id = temp_id
                for idx, res in enumerate(chain_res):
                    if idx in resnum_map:
                        new_num, new_ins = resnum_map[idx]
                        tid = temp_ids[idx]
                        if new_ins == " ":
                            new_ins = tid[2]
                        res.id = (tid[0], new_num, new_ins)
                chain.id = ref_receptor_id

            elif chain.id == des_pep_orig and ref_peptide is not None:
                chain.id = ref_peptide_id

    io = PDBIO()
    io.set_structure(des_structure)
    io.save(output_path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_design(
    pdb_path: str,
    pocket_resnums: Set[int],
    head_positions: List[int],
    receptor_chain_id: str,
    peptide_chain_id: str,
    contact_cutoff: float = 5.0,
    severe_clash_cutoff: float = 1.8,
    mild_clash_cutoff: float = 2.1,
    hotspot_resnums: Optional[Set[int]] = None,
    pocket_center: Optional[List[float]] = None,
) -> Dict:
    """Score a single renumbered design PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("design", pdb_path)
    receptor, peptide = get_receptor_peptide(
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
            "n_hotspot_contacts": 0,
            "peptide_pocket_dist": float("nan"),
            "error": "missing receptor or peptide chain",
        })
        return result

    pep_residues = aa_residues(peptide)

    head_res_objects = []
    for pos in head_positions:
        if pos < len(pep_residues):
            head_res_objects.append(pep_residues[pos])

    pocket_res_objects = [
        r for r in aa_residues(receptor) if r.id[1] in pocket_resnums
    ]

    # -- Head-pocket contacts --
    pair_dists: List[float] = []
    head_touching: Set[int] = set()
    for h_idx, h_res in enumerate(head_res_objects):
        for p_res in pocket_res_objects:
            d = heavy_atom_min_dist(h_res, p_res)
            if d <= contact_cutoff:
                pair_dists.append(d)
                head_touching.add(h_idx)

    n_head_in_pocket = len(head_touching)
    n_head_pocket_pairs = len(pair_dists)
    min_dist = float(min(pair_dists)) if pair_dists else float("nan")
    mean_dist = float(np.mean(pair_dists)) if pair_dists else float("nan")

    # -- Interface contacts --
    rec_atoms = [a for r in aa_residues(receptor) for a in r.get_atoms() if a.element != "H"]
    if rec_atoms:
        ns_rec = NeighborSearch(rec_atoms)
        interface_pairs: Set[Tuple[int, int]] = set()
        for res in aa_residues(peptide):
            for atom in res.get_atoms():
                if atom.element == "H":
                    continue
                for hit in ns_rec.search(atom.coord, contact_cutoff, level="R"):
                    if not is_aa(hit, standard=True):
                        continue
                    interface_pairs.add((res.id[1], hit.id[1]))
        n_interface = len(interface_pairs)
    else:
        n_interface = 0

    # -- Clashes --
    n_severe, _ = find_clashes(peptide, receptor, severe_clash_cutoff)
    n_mild, _ = find_clashes(peptide, receptor, mild_clash_cutoff)

    # -- Hotspot contacts --
    n_hotspot = 0
    if hotspot_resnums:
        pep_heavy = list(heavy_atoms(peptide))
        if pep_heavy:
            ns_pep = NeighborSearch(pep_heavy)
            for res in aa_residues(receptor):
                if res.id[1] not in hotspot_resnums:
                    continue
                contacted = False
                for atom in res.get_atoms():
                    if atom.element == "H":
                        continue
                    hits = ns_pep.search(atom.coord, contact_cutoff, level="A")
                    if hits:
                        contacted = True
                        break
                if contacted:
                    n_hotspot += 1

    # -- Peptide-pocket distance (computed from design's own coordinates) --
    design_pocket_ca = [
        res["CA"].coord for res in pocket_res_objects if "CA" in res
    ]
    pep_ca_coords = [res["CA"].coord for res in pep_residues if "CA" in res]
    if pep_ca_coords and design_pocket_ca:
        design_pocket_center = np.mean(design_pocket_ca, axis=0)
        pep_centroid = np.mean(pep_ca_coords, axis=0)
        pep_pocket_dist = float(np.linalg.norm(pep_centroid - design_pocket_center))
    else:
        pep_pocket_dist = float("nan")

    result.update({
        "n_head_in_pocket": n_head_in_pocket,
        "n_head_pocket_pairs": n_head_pocket_pairs,
        "min_head_pocket_dist": round(min_dist, 2),
        "mean_head_pocket_dist": round(mean_dist, 2),
        "n_interface_contacts": n_interface,
        "n_clashes_severe": n_severe,
        "n_clashes_mild": n_mild,
        "n_hotspot_contacts": n_hotspot,
        "peptide_pocket_dist": round(pep_pocket_dist, 2),
    })
    return result


# ---------------------------------------------------------------------------
# Pass/fail logic
# ---------------------------------------------------------------------------

def apply_thresholds(row: Dict, config: Dict) -> Dict:
    """Apply all Stage 1 thresholds. Returns dict with pass_* columns and overall pass."""
    checks = [
        ("min_head_in_pocket", "n_head_in_pocket", "min"),
        ("min_head_pocket_pairs", "n_head_pocket_pairs", "min"),
        ("max_severe_clashes", "n_clashes_severe", "max"),
        ("max_mild_clashes", "n_clashes_mild", "max"),
        ("min_hotspot_contacts", "n_hotspot_contacts", "min"),
        ("max_peptide_pocket_dist", "peptide_pocket_dist", "max"),
    ]

    overall_pass = True
    for threshold_name, metric_name, direction in checks:
        value = row.get(metric_name, 0 if direction == "min" else float("inf"))
        tv, mode = get_threshold(config, "stage1", threshold_name)

        if threshold_name == "min_hotspot_contacts":
            hotspots = config.get("stage1", {}).get("hotspot_residues", [])
            if not hotspots:
                mode = "ignore"

        if mode == "ignore" or tv is None:
            row[f"pass_{threshold_name}"] = True
            continue

        if np.isnan(value) if isinstance(value, float) else False:
            passes = False
        elif direction == "min":
            passes = value >= tv
        else:
            passes = value <= tv

        row[f"pass_{threshold_name}"] = passes

        if not passes and mode == "filter":
            overall_pass = False

    row["pass_stage1"] = overall_pass
    return row


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage1(
    reference_pdb: str,
    design_pdbs: List[str],
    outdir: str,
    config: Dict,
    receptor_chain: Optional[str] = None,
    peptide_chain: Optional[str] = None,
    hotspot_residues: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run Stage 1: region definition + renumber + structural filters."""
    s1_cfg = config.get("stage1", {})
    pocket_cutoff = s1_cfg.get("pocket_cutoff", 6.0)
    head_cutoff = s1_cfg.get("head_cutoff", 5.0)
    contact_cutoff = s1_cfg.get("contact_cutoff", 5.0)
    severe_clash_cutoff = s1_cfg.get("severe_clash_cutoff", 1.8)
    mild_clash_cutoff = s1_cfg.get("mild_clash_cutoff", 2.1)

    if hotspot_residues is None:
        hotspot_residues = s1_cfg.get("hotspot_residues", [])

    receptor_chain = receptor_chain or config.get("receptor_chain")
    peptide_chain = peptide_chain or config.get("peptide_chain")

    outpath = Path(outdir) / "stage1"
    outpath.mkdir(parents=True, exist_ok=True)
    renumbered_dir = outpath / "renumbered"
    renumbered_dir.mkdir(parents=True, exist_ok=True)

    # ---- Define regions ----
    log("=" * 60)
    log("STAGE 1: Region definition + renumbering + structural filters")
    log("=" * 60)

    regions = define_regions(
        reference_pdb,
        receptor_chain_id=receptor_chain,
        peptide_chain_id=peptide_chain,
        pocket_cutoff=pocket_cutoff,
        head_cutoff=head_cutoff,
        hotspot_residues=hotspot_residues,
    )

    regions_path = outpath / "regions.json"
    regions_path.write_text(json.dumps(regions, indent=2) + "\n")
    log(f"[Stage 1] Wrote {regions_path}")

    rec_chain = regions["receptor_chain"]
    pep_chain = regions["peptide_chain"]
    pocket_resnums = {r["resnum"] for r in regions["pocket_residues"]}
    head_positions = regions["head_positions"]
    pocket_center = regions["pocket_center"]
    hotspot_set = set(regions["hotspot_residues"]) if regions["hotspot_residues"] else set()

    # ---- Renumber designs ----
    log(f"\n[Stage 1] Renumbering {len(design_pdbs)} designs...")

    parser_obj = PDBParser(QUIET=True)
    ref_structure = parser_obj.get_structure("reference", regions["reference_pdb"])

    renumbered_paths: List[Tuple[str, str]] = []
    for pdb in design_pdbs:
        name = Path(pdb).stem
        out_path = renumbered_dir / f"renumbered_{name}.pdb"
        try:
            renumber_design(
                ref_structure, pdb, str(out_path),
                ref_receptor_id=rec_chain, ref_peptide_id=pep_chain,
            )
            renumbered_paths.append((pdb, str(out_path)))
        except Exception as e:
            log(f"  ERROR renumbering {pdb}: {e}")

    log(f"[Stage 1] Renumbered {len(renumbered_paths)}/{len(design_pdbs)} designs")

    # ---- Score designs ----
    log(f"\n[Stage 1] Scoring {len(renumbered_paths)} designs...")
    log(f"  Pocket: {len(pocket_resnums)} residues, Head: {len(head_positions)} positions")
    if hotspot_set:
        log(f"  Hotspots: {sorted(hotspot_set)}")

    all_rows: List[Dict] = []
    for original_pdb, renum_pdb in renumbered_paths:
        row = score_design(
            renum_pdb,
            pocket_resnums=pocket_resnums,
            head_positions=head_positions,
            receptor_chain_id=rec_chain,
            peptide_chain_id=pep_chain,
            contact_cutoff=contact_cutoff,
            severe_clash_cutoff=severe_clash_cutoff,
            mild_clash_cutoff=mild_clash_cutoff,
            hotspot_resnums=hotspot_set if hotspot_set else None,
            pocket_center=pocket_center,
        )
        row["original_pdb"] = Path(original_pdb).name
        row = apply_thresholds(row, config)
        all_rows.append(row)

    # ---- Summary CSV ----
    summary_df = pd.DataFrame(all_rows)
    summary_cols = [
        "original_pdb", "design",
        "n_head_in_pocket", "n_head_pocket_pairs",
        "min_head_pocket_dist", "mean_head_pocket_dist",
        "n_interface_contacts",
        "n_clashes_severe", "n_clashes_mild",
        "n_hotspot_contacts", "peptide_pocket_dist",
        "pass_min_head_in_pocket", "pass_min_head_pocket_pairs",
        "pass_max_severe_clashes", "pass_max_mild_clashes",
        "pass_min_hotspot_contacts", "pass_max_peptide_pocket_dist",
        "pass_stage1",
    ]
    existing_cols = [c for c in summary_cols if c in summary_df.columns]
    summary_df = summary_df[existing_cols]

    summary_csv = outpath / "stage1_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # ---- Copy passing designs ----
    pass_dir = outpath / "pass_designs"
    pass_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for row, (_orig, renum_pdb) in zip(all_rows, renumbered_paths):
        if row.get("pass_stage1", False):
            src = Path(renum_pdb)
            dst = pass_dir / src.name
            shutil.copy2(str(src), str(dst))
            n_copied += 1

    n_pass = summary_df["pass_stage1"].sum() if "pass_stage1" in summary_df.columns else 0
    log(f"\n{'=' * 60}")
    log(f"STAGE 1 DONE: {n_pass}/{len(summary_df)} designs passed")
    log(f"  Regions JSON     -> {regions_path}")
    log(f"  Renumbered PDBs  -> {renumbered_dir}/")
    log(f"  Passing designs  -> {pass_dir}/ ({n_copied} PDBs)")
    log(f"  Summary CSV      -> {summary_csv}")
    log("=" * 60)

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1: Region definition + renumbering + structural filters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("reference_pdb", help="Reference PDB (cryo-EM / crystal complex)")
    ap.add_argument("design_pdbs", nargs="+", help="RFDiffusion design PDB files")
    ap.add_argument("-o", "--outdir", default="pipeline_results", help="Output directory")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--receptor-chain", default=None, help="Receptor chain ID")
    ap.add_argument("--peptide-chain", default=None, help="Peptide chain ID")
    ap.add_argument(
        "--hotspot-residues", default=None,
        help="Comma-separated receptor resnums for hotspot check (e.g. 294,301,305)",
    )

    args = ap.parse_args()
    config = load_config(args.config) if args.config else {}

    hotspots = None
    if args.hotspot_residues:
        hotspots = [int(x.strip()) for x in args.hotspot_residues.split(",")]

    run_stage1(
        reference_pdb=args.reference_pdb,
        design_pdbs=args.design_pdbs,
        outdir=args.outdir,
        config=config,
        receptor_chain=args.receptor_chain,
        peptide_chain=args.peptide_chain,
        hotspot_residues=hotspots,
    )


if __name__ == "__main__":
    main()
