#!/usr/bin/env python3
"""
Stage 0 -- Define pocket and head residues from a reference PDB.

Algorithm:
  1. Parse the reference PDB and identify receptor / peptide chains.
  2. Pocket  = receptor residues with any heavy atom within *pocket_cutoff* of
               any peptide heavy atom. Heavy = None hydrogen atoms.
  3. Head    = peptide residues with any heavy atom within *head_cutoff* of any
               pocket residue heavy atom. Heavy = None hydrogen atoms.
  4. Write a JSON config consumed by the downstream pipeline stages.

Output JSON schema:
  {
    "reference_pdb": "...",
    "receptor_chain": "R",
    "peptide_chain": "P",
    "pocket_cutoff_angstrom": 6.0,
    "head_cutoff_angstrom": 5.0,
    "pocket_residues": [{"resnum": 112, "resname": "ASP", "seq_pos": 5}, ...],
    "pocket_positions": [5, 12, ...],          # 0-indexed receptor seq positions
    "head_positions": [0, 1, 2, ...],          # 0-indexed peptide seq positions
    "head_residues":  [{"resnum": 7, "resname": "HIS", "seq_pos": 0}, ...]
  }

Example usage:
  # Auto-detect receptor/peptide chains by length -> NOT RELIABLE TO DETECT RECEPTOR CHAIN!!!
  python metrics_pipeline/define_regions.py my_complex.pdb -o regions.json

  # Specify chain IDs when you already know them
  python metrics_pipeline/define_regions.py glp1.pdb --receptor-chain R --peptide-chain P -o regions_glp1.json
  python metrics_pipeline/define_regions.py oxytocin.pdb --receptor-chain O --peptide-chain P -o regions_oxy.json
  python metrics_pipeline/define_regions.py best_design.pdb --receptor-chain A --peptide-chain B -o regions.json

  # Override distance cutoffs (defaults: pocket 6 A, head 5 A)
  python metrics_pipeline/define_regions.py glp1.pdb --receptor-chain R --peptide-chain P \
      --pocket-cutoff 5.5 --head-cutoff 4.5 -o regions_tight.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa


def _get_receptor_peptide(structure, receptor_id=None, peptide_id=None):
    """Return (receptor_chain, peptide_chain) using explicit IDs or length heuristic."""
    receptor, peptide = None, None
    for chain in structure[0]:
        residues = [r for r in chain if is_aa(r, standard=True)]
        if not residues:
            continue
        if receptor_id and chain.id == receptor_id:
            receptor = chain
            continue
        if peptide_id and chain.id == peptide_id:
            peptide = chain
            continue
        if receptor is None and receptor_id is None and len(residues) > 100:
            receptor = chain
        elif peptide is None and peptide_id is None and len(residues) <= 100:
            peptide = chain
    return receptor, peptide


def _heavy_atoms(chain):
    """Yield heavy atoms from standard amino-acid residues."""
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        for atom in res.get_atoms():
            if atom.element != "H":
                yield atom


def _aa_residues(chain):
    return [r for r in chain if is_aa(r, standard=True)]


def define_regions(
    reference_pdb: str,
    receptor_chain_id: Optional[str] = None,
    peptide_chain_id: Optional[str] = None,
    pocket_cutoff: float = 6.0,
    head_cutoff: float = 5.0,
) -> Dict:
    """
    Derive pocket and head residue definitions from the reference complex.

    Returns a dict suitable for JSON serialization.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", reference_pdb)

    receptor, peptide = _get_receptor_peptide(
        structure, receptor_chain_id, peptide_chain_id
    )
    if receptor is None:
        raise ValueError("Could not identify receptor chain")
    if peptide is None:
        raise ValueError("Could not identify peptide chain")

    print(
        f"Receptor chain {receptor.id} ({len(_aa_residues(receptor))} aa), "
        f"Peptide chain {peptide.id} ({len(_aa_residues(peptide))} aa)",
        file=sys.stderr,
    )

    # --- Step 1: pocket = receptor residues near the full peptide ---------------
    pep_atoms = list(_heavy_atoms(peptide))
    ns_pep = NeighborSearch(pep_atoms)

    pocket_residues = []
    pocket_positions = []
    pocket_res_set = set()
    rec_residues = _aa_residues(receptor)
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
                break  # one contact is enough to flag this receptor residue

    pocket_residues.sort(key=lambda x: x["resnum"])
    pocket_positions.sort()
    print(f"Pocket: {len(pocket_residues)} receptor residues within {pocket_cutoff} A of peptide", file=sys.stderr)

    # --- Step 2: head = peptide residues near pocket residues -------------------
    pocket_atoms = []
    for res in _aa_residues(receptor):
        if res.id[1] in pocket_res_set:
            for atom in res.get_atoms():
                if atom.element != "H":
                    pocket_atoms.append(atom)

    ns_pocket = NeighborSearch(pocket_atoms)

    pep_residues = _aa_residues(peptide)
    head_residues = []
    head_positions = []
    for seq_pos, res in enumerate(pep_residues):
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            hits = ns_pocket.search(atom.coord, head_cutoff, level="A")
            if hits:
                head_positions.append(seq_pos)
                head_residues.append(
                    {
                        "resnum": res.id[1],
                        "resname": res.get_resname(),
                        "seq_pos": seq_pos,
                    }
                )
                break

    print(f"Head: {len(head_residues)} peptide residues within {head_cutoff} A of pocket", file=sys.stderr)

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
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 0: derive pocket & head definitions from a reference PDB complex.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("reference_pdb", help="Reference PDB (cryo-EM / crystal complex)")
    ap.add_argument("--receptor-chain", default=None, help="Receptor chain ID (auto-detect if omitted)")
    ap.add_argument("--peptide-chain", default=None, help="Peptide chain ID (auto-detect if omitted)")
    ap.add_argument("--pocket-cutoff", type=float, default=6.0, help="Max distance (A) to define pocket")
    ap.add_argument("--head-cutoff", type=float, default=5.0, help="Max distance (A) to define head")
    ap.add_argument("-o", "--output", default="regions.json", help="Output JSON path")
    args = ap.parse_args()

    config = define_regions(
        args.reference_pdb,
        receptor_chain_id=args.receptor_chain,
        peptide_chain_id=args.peptide_chain,
        pocket_cutoff=args.pocket_cutoff,
        head_cutoff=args.head_cutoff,
    )

    Path(args.output).write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
