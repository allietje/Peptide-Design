#!/usr/bin/env python3
"""
Optional step between Stage 0 and Stage 1 -- Batch-renumber design PDBs.

Reads the regions.json from Stage 0 (to get the reference PDB path and chain
IDs) and one or more raw design PDBs.  For each design it:

  1. Identifies receptor (longer chain) and peptide (shorter chain).
  2. Aligns the design receptor sequence to the reference receptor.
  3. Renumbers the design receptor residues to match the reference.
  4. Renames chain IDs to match the reference.
  5. Writes the renumbered PDB to an output directory, ready for Stage 1.

This wraps the logic from renumber_pdb.py so you don't have to loop manually.

Example usage:
  # Renumber all designs using info from regions.json, write to renumbered/ dir
  python metrics_pipeline/batch_renumber.py regions.json design_0.pdb design_1.pdb design_2.pdb design_3.pdb

  # Use a glob
  python metrics_pipeline/batch_renumber.py regions.json design_*.pdb

  # Custom output directory
  python metrics_pipeline/batch_renumber.py regions.json design_*.pdb -o my_renumbered/

  # Custom output prefix
  python metrics_pipeline/batch_renumber.py regions.json design_*.pdb --prefix renumbered_
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import pairwise2
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


def _get_chains_by_length(structure):
    """Return (receptor_chain, peptide_chain) by picking longest and shortest."""
    chains = []
    for chain in structure[0]:
        residues = [r for r in chain if is_aa(r, standard=True)]
        if residues:
            chains.append((chain, len(residues)))
    if len(chains) < 2:
        raise ValueError(f"Expected 2 protein chains, found {len(chains)}")
    chains.sort(key=lambda x: x[1], reverse=True)
    return chains[0][0], chains[1][0]


def _get_chain_by_id(structure, chain_id):
    for chain in structure[0]:
        if chain.id == chain_id:
            return chain
    return None


def _get_sequence(chain):
    residues = [r for r in chain if is_aa(r, standard=True)]
    sequence = "".join(seq1(r.get_resname()) for r in residues)
    return residues, sequence


def _align(ref_seq: str, des_seq: str) -> Dict[int, int]:
    """Return mapping: design_index -> reference_index (0-based)."""
    if ref_seq == des_seq:
        return {i: i for i in range(len(ref_seq))}
    alignments = pairwise2.align.globalxx(ref_seq, des_seq)
    if not alignments:
        raise ValueError("Sequence alignment failed")
    ref_al = alignments[0].seqA
    des_al = alignments[0].seqB
    mapping = {}
    ri, di = 0, 0
    for rc, dc in zip(ref_al, des_al):
        if rc != "-":
            ri += 1
        if dc != "-":
            di += 1
            if rc != "-":
                mapping[di - 1] = ri - 1
    return mapping


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

    ref_receptor = _get_chain_by_id(ref_structure, ref_receptor_id)
    ref_peptide = _get_chain_by_id(ref_structure, ref_peptide_id)
    if ref_receptor is None:
        raise ValueError(f"Reference receptor chain '{ref_receptor_id}' not found")

    des_receptor, des_peptide = _get_chains_by_length(des_structure)

    ref_residues, ref_seq = _get_sequence(ref_receptor)
    des_residues, des_seq = _get_sequence(des_receptor)

    mapping = _align(ref_seq, des_seq)

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
                # Two-pass renumber to avoid ID collisions
                temp_ids = {}
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Batch-renumber design PDBs to match the reference numbering "
            "from regions.json.  Output files are ready for stage1_filters.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("regions_json", help="regions.json from define_regions.py")
    ap.add_argument("design_pdbs", nargs="+", help="Raw design PDB files")
    ap.add_argument(
        "-o", "--outdir", default="renumbered",
        help="Output directory for renumbered PDBs",
    )
    ap.add_argument(
        "--prefix", default="renumbered_",
        help="Prefix prepended to each output filename",
    )
    args = ap.parse_args()

    with open(args.regions_json) as f:
        config = json.load(f)

    ref_pdb = config["reference_pdb"]
    rec_chain = config["receptor_chain"]
    pep_chain = config["peptide_chain"]

    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure("reference", ref_pdb)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"Reference: {ref_pdb} (receptor={rec_chain}, peptide={pep_chain})",
        file=sys.stderr,
    )

    output_paths = []
    for pdb in args.design_pdbs:
        name = Path(pdb).stem
        out_path = outdir / f"{args.prefix}{name}.pdb"
        print(f"  {pdb} -> {out_path}", file=sys.stderr)
        try:
            renumber_design(
                ref_structure, pdb, str(out_path),
                ref_receptor_id=rec_chain,
                ref_peptide_id=pep_chain,
            )
            output_paths.append(str(out_path))
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print(f"\nRenumbered {len(output_paths)} designs -> {outdir}/", file=sys.stderr)
    print("Next step:", file=sys.stderr)
    print(
        f"  python metrics_pipeline/stage1_filters.py {args.regions_json} "
        f"{outdir}/{args.prefix}*.pdb -o stage1_scores.csv",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
