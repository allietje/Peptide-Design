#!/usr/bin/env python3
"""
Stage 2 -- Local head/pocket AF-Multimer confidence metrics from server ZIPs.

Reads the regions.json produced by define_regions.py and one or more AlphaFold
server ZIP files.  For each model seed it computes:

  - mean_pLDDT_head           (from the CIF B-factor / atom_site occupancy field)
  - mean_PAE_head_to_pocket   (PAE[head_tokens, pocket_tokens])
  - mean_PAE_pocket_to_head   (PAE[pocket_tokens, head_tokens])
  - mean_PAE_head_pocket_sym  (average of the two directions)
  - n_head_pocket_contacts_af (heavy-atom contact pairs < cutoff in the AF model)
  - ipTM                      (from summary confidences)
  - ranking_score             (from summary confidences)

Output: CSV written to stdout or a file.

Example usage:
  # Score one AF server ZIP (peptide=chain A, receptor=chain B inside the AF output)
  python metrics_pipeline/score_af_local.py regions.json af_output_design0.zip \
      --af-peptide-chain A --af-receptor-chain B -o stage2_scores.csv

  # Score multiple ZIPs at once
  python metrics_pipeline/score_af_local.py regions.json af_*.zip \
      --af-peptide-chain A --af-receptor-chain B -o stage2_scores.csv

  # Adjust pass/fail thresholds (defaults: pLDDT >= 80, PAE <= 10)
  python metrics_pipeline/score_af_local.py regions.json af_*.zip \
      --plddt-threshold 85 --pae-threshold 8 -o stage2_scores.csv

  # Use a different contact cutoff for AF contact counting (default 5.0 A)
  python metrics_pipeline/score_af_local.py regions.json af_output.zip \
      --contact-cutoff 4.5 -o stage2_scores.csv

  # Print to stdout instead of writing a file (omit -o)
  python metrics_pipeline/score_af_local.py regions.json af_output.zip \
      --af-peptide-chain A --af-receptor-chain B
"""

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa


def _load_json(zf: zipfile.ZipFile, name: str) -> dict:
    return json.loads(zf.read(name).decode("utf-8"))


def _model_indices(namelist) -> List[int]:
    idx = set()
    for n in namelist:
        m = re.search(r"_full_data_(\d+)\.json$", n)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def _token_index_map(full_data: dict, chain: str) -> Dict[int, int]:
    """Map resnum -> token index for a given chain."""
    ch = np.array(full_data["token_chain_ids"])
    ri = np.array(full_data["token_res_ids"])
    return {int(r): int(i) for i, (c, r) in enumerate(zip(ch, ri)) if c == chain}


def _plddt_from_cif(cif_path: str, chain_id: str, resnums: Set[int]) -> float:
    """
    Extract mean pLDDT for a subset of residues from a model CIF.

    AF stores per-atom confidence in the B-factor column. We average over
    all heavy atoms of the requested residues.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("m", cif_path)
    model = structure[0]
    if chain_id not in model:
        return float("nan")
    chain = model[chain_id]
    vals = []
    for res in chain.get_residues():
        if res.id[1] not in resnums:
            continue
        for atom in res.get_atoms():
            if atom.element != "H":
                vals.append(atom.get_bfactor())
    return float(np.mean(vals)) if vals else float("nan")


def _head_pocket_contacts_af(
    cif_path: str,
    pep_chain: str,
    rec_chain: str,
    head_resnums: Set[int],
    pocket_resnums: Set[int],
    cutoff: float,
) -> int:
    """Count (head, pocket) residue pairs with heavy-atom contact < cutoff in the AF CIF."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("m", cif_path)
    model = structure[0]
    if pep_chain not in model or rec_chain not in model:
        return 0

    pocket_atoms = []
    atom_to_res = {}
    for res in model[rec_chain].get_residues():
        if res.id[1] not in pocket_resnums:
            continue
        for atom in res.get_atoms():
            if atom.element != "H":
                pocket_atoms.append(atom)
                atom_to_res[id(atom)] = res.id[1]

    if not pocket_atoms:
        return 0

    ns = NeighborSearch(pocket_atoms)
    pairs: Set[Tuple[int, int]] = set()
    for res in model[pep_chain].get_residues():
        if res.id[1] not in head_resnums:
            continue
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            for hit in ns.search(atom.coord, cutoff, level="A"):
                pairs.add((res.id[1], atom_to_res[id(hit)]))

    return len(pairs)


def score_af_zip(
    zip_path: str,
    head_resnums: Set[int],
    pocket_resnums: Set[int],
    pep_chain: str = "A",
    rec_chain: str = "B",
    contact_cutoff: float = 5.0,
) -> pd.DataFrame:
    """Score one AF server ZIP against head/pocket definitions."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        ks = _model_indices(names)
        if not ks:
            raise RuntimeError("No *_full_data_#.json found -- is this an AF server ZIP?")

        rows = []
        for k in ks:
            full_name = [n for n in names if n.endswith(f"_full_data_{k}.json")]
            summ_name = [n for n in names if n.endswith(f"_summary_confidences_{k}.json")]
            cif_name = [n for n in names if n.endswith(f"_model_{k}.cif")]

            if not full_name or not summ_name or not cif_name:
                continue

            full = _load_json(zf, full_name[0])
            summ = _load_json(zf, summ_name[0])
            pae = np.array(full["pae"], dtype=float)

            pep_tok = _token_index_map(full, pep_chain)
            rec_tok = _token_index_map(full, rec_chain)

            head_tok_idx = sorted(pep_tok[r] for r in head_resnums if r in pep_tok)
            pocket_tok_idx = sorted(rec_tok[r] for r in pocket_resnums if r in rec_tok)

            if head_tok_idx and pocket_tok_idx:
                sub_hp = pae[np.ix_(head_tok_idx, pocket_tok_idx)]
                sub_ph = pae[np.ix_(pocket_tok_idx, head_tok_idx)]
                mean_hp = float(sub_hp.mean())
                mean_ph = float(sub_ph.mean())
                mean_sym = (mean_hp + mean_ph) / 2.0
            else:
                mean_hp = mean_ph = mean_sym = float("nan")

            # pLDDT from CIF
            with tempfile.TemporaryDirectory() as td:
                cif_local = os.path.join(td, os.path.basename(cif_name[0]))
                with open(cif_local, "wb") as f:
                    f.write(zf.read(cif_name[0]))

                plddt_head = _plddt_from_cif(cif_local, pep_chain, head_resnums)
                n_contacts = _head_pocket_contacts_af(
                    cif_local, pep_chain, rec_chain,
                    head_resnums, pocket_resnums, contact_cutoff,
                )

            rows.append({
                "zip": Path(zip_path).name,
                "model": k,
                "mean_pLDDT_head": round(plddt_head, 2),
                "mean_PAE_head_to_pocket": round(mean_hp, 2),
                "mean_PAE_pocket_to_head": round(mean_ph, 2),
                "mean_PAE_head_pocket_sym": round(mean_sym, 2),
                "n_head_pocket_contacts_af": n_contacts,
                "ipTM": round(float(summ.get("iptm", float("nan"))), 4),
                "ranking_score": round(float(summ.get("ranking_score", float("nan"))), 4),
            })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2: local head/pocket AF-Multimer confidence metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("regions_json", help="regions.json from define_regions.py")
    ap.add_argument("af_zips", nargs="+", help="AlphaFold server ZIP file(s)")
    ap.add_argument(
        "--af-peptide-chain", default="A",
        help="Peptide chain ID inside the AF output (often A for the shorter sequence)",
    )
    ap.add_argument(
        "--af-receptor-chain", default="B",
        help="Receptor chain ID inside the AF output (often B for the longer sequence)",
    )
    ap.add_argument("--contact-cutoff", type=float, default=5.0, help="Heavy-atom contact cutoff (A)")
    ap.add_argument("--plddt-threshold", type=float, default=80.0, help="Suggested pLDDT threshold for filtering")
    ap.add_argument("--pae-threshold", type=float, default=10.0, help="Suggested PAE threshold for filtering")
    ap.add_argument("-o", "--output", default=None, help="Output CSV (default: stdout)")
    args = ap.parse_args()

    with open(args.regions_json) as f:
        config = json.load(f)

    head_resnums = {r["resnum"] for r in config["head_residues"]}
    pocket_resnums = {r["resnum"] for r in config["pocket_residues"]}

    print(
        f"Config: {len(head_resnums)} head resnums, {len(pocket_resnums)} pocket resnums",
        file=sys.stderr,
    )

    frames = []
    for zp in args.af_zips:
        print(f"  scoring {zp} ...", file=sys.stderr)
        df = score_af_zip(
            zp,
            head_resnums=head_resnums,
            pocket_resnums=pocket_resnums,
            pep_chain=args.af_peptide_chain,
            rec_chain=args.af_receptor_chain,
            contact_cutoff=args.contact_cutoff,
        )
        frames.append(df)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not result.empty:
        result["pass_plddt"] = result["mean_pLDDT_head"] >= args.plddt_threshold
        result["pass_pae"] = result["mean_PAE_head_pocket_sym"] <= args.pae_threshold

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote {args.output} ({len(result)} rows)", file=sys.stderr)
    else:
        print(result.to_csv(index=False))


if __name__ == "__main__":
    main()
