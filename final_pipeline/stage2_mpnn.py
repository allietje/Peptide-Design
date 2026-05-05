#!/usr/bin/env python3
"""
Stage 2: ProteinMPNN sequence design.

Takes the pass_designs/ PDBs from Stage 1 and runs ProteinMPNN to generate
peptide sequences for each backbone, keeping the receptor chain fixed.

Inputs:
  - pass_designs/ directory (renumbered PDBs from Stage 1)
  - regions.json from Stage 1 (for chain IDs)
  - config.yaml or CLI args for MPNN parameters

Outputs (all in <outdir>/):
  - parsed_pdbs.jsonl     -- MPNN-parsed chain coordinates
  - fixed_chains.jsonl    -- chain design/fix assignments
  - mpnn_seqs/            -- per-design FASTA files from MPNN
  - stage2_summary.csv    -- design_name, sample, sequence, mpnn_score, global_score

Usage:
  python stage2_mpnn.py -o pipeline_results/

  python stage2_mpnn.py --config config.yaml \\
      --num-seq-per-target 4 --sampling-temp 0.1 \\
      -o pipeline_results/

Requires GPU for reasonable speed. Run inside a SLURM GPU allocation:
  srun -p mit_preemptable -t 60 -n 1 --gres=gpu:1 --mem=64G --pty bash
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils import load_config, log


def parse_mpnn_fasta(fasta_path: str, peptide_chain: str) -> List[Dict]:
    """Parse a ProteinMPNN output FASTA into a list of sequence records.

    Returns one dict per designed sequence (skips the first entry which is
    the original/wild-type backbone sequence).
    """
    records = []
    design_name = Path(fasta_path).stem

    with open(fasta_path) as f:
        lines = f.readlines()

    current_header = None
    current_seq_lines = []

    def _flush():
        if current_header is None:
            return
        seq = "".join(current_seq_lines).strip()
        header = current_header

        if header.startswith(f">{design_name}"):
            return

        parts = {}
        for token in header.lstrip(">").split(","):
            token = token.strip()
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k.strip()] = v.strip()

        sample_idx = int(parts.get("sample", 0))

        full_seq = seq
        chain_seqs = full_seq.split("/") if "/" in full_seq else [full_seq]

        if len(chain_seqs) == 1:
            peptide_seq = chain_seqs[0]
        else:
            peptide_seq = chain_seqs[-1]

        try:
            mpnn_score = float(parts.get("score", "nan"))
        except ValueError:
            mpnn_score = float("nan")

        try:
            global_score = float(parts.get("global_score", "nan"))
        except ValueError:
            global_score = float("nan")

        records.append({
            "design": design_name,
            "sample": sample_idx,
            "peptide_sequence": peptide_seq,
            "full_sequence": full_seq,
            "mpnn_score": mpnn_score,
            "global_score": global_score,
        })

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(">"):
            _flush()
            current_header = line
            current_seq_lines = []
        else:
            current_seq_lines.append(line)
    _flush()

    return records


def run_stage2(
    outdir: str,
    config: Dict,
    mpnn_path: Optional[str] = None,
    num_seq_per_target: Optional[int] = None,
    sampling_temp: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """Run Stage 2: ProteinMPNN sequence design."""
    s2_cfg = config.get("stage2", {})

    mpnn_path = mpnn_path or s2_cfg.get("mpnn_path", "/orcd/data/zhang_f/001/azong/software/ProteinMPNN")
    num_seq = num_seq_per_target or s2_cfg.get("num_seq_per_target", 4)
    temp = sampling_temp or s2_cfg.get("sampling_temp", "0.1")
    bs = batch_size or s2_cfg.get("batch_size", num_seq)

    if bs > num_seq:
        log(f"[Stage 2] WARNING: batch_size ({bs}) > num_seq_per_target ({num_seq}), setting batch_size = {num_seq}")
        bs = num_seq

    base_path = Path(outdir)
    stage1_path = base_path / "stage1"
    outpath = base_path / "stage2"
    outpath.mkdir(parents=True, exist_ok=True)

    pass_dir = stage1_path / "pass_designs"
    regions_path = stage1_path / "regions.json"

    if not pass_dir.exists():
        raise FileNotFoundError(f"pass_designs/ not found in {stage1_path}")

    pdb_files = sorted(pass_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files in {pass_dir}")

    regions = {}
    if regions_path.exists():
        with open(regions_path) as f:
            regions = json.load(f)

    peptide_chain = regions.get("peptide_chain", config.get("peptide_chain", "P"))

    log("=" * 60)
    log("STAGE 2: ProteinMPNN Sequence Design")
    log("=" * 60)
    log(f"  Input PDBs: {len(pdb_files)} from {pass_dir}")
    log(f"  MPNN path: {mpnn_path}")
    log(f"  Sequences per target: {num_seq}")
    log(f"  Sampling temperature: {temp}")
    log(f"  Batch size: {bs}")
    log(f"  Peptide chain (to redesign): {peptide_chain}")

    # ---- Step 1: Parse PDBs ----
    parsed_jsonl = outpath / "parsed_pdbs.jsonl"
    log(f"\n[Stage 2] Step 1: Parsing PDBs...")

    parse_cmd = [
        sys.executable,
        str(Path(mpnn_path) / "helper_scripts" / "parse_multiple_chains.py"),
        "--input_path", str(pass_dir),
        "--output_path", str(parsed_jsonl),
    ]
    log(f"  CMD: {' '.join(parse_cmd)}")
    result = subprocess.run(parse_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"parse_multiple_chains.py failed: {result.stderr}")
    log(f"  Wrote {parsed_jsonl}")

    # ---- Step 2: Assign fixed chains ----
    fixed_jsonl = outpath / "fixed_chains.jsonl"
    log(f"\n[Stage 2] Step 2: Assigning fixed chains (redesign chain {peptide_chain})...")

    fix_cmd = [
        sys.executable,
        str(Path(mpnn_path) / "helper_scripts" / "assign_fixed_chains.py"),
        "--input_path", str(parsed_jsonl),
        "--output_path", str(fixed_jsonl),
        "--chain_list", peptide_chain,
    ]
    log(f"  CMD: {' '.join(fix_cmd)}")
    result = subprocess.run(fix_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"assign_fixed_chains.py failed: {result.stderr}")
    log(f"  Wrote {fixed_jsonl}")

    # ---- Step 3: Run ProteinMPNN ----
    mpnn_out_dir = outpath / "mpnn_raw"
    mpnn_out_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n[Stage 2] Step 3: Running ProteinMPNN ({num_seq} seqs/target, temp={temp})...")

    run_cmd = [
        sys.executable,
        str(Path(mpnn_path) / "protein_mpnn_run.py"),
        "--jsonl_path", str(parsed_jsonl),
        "--chain_id_jsonl", str(fixed_jsonl),
        "--out_folder", str(mpnn_out_dir),
        "--num_seq_per_target", str(num_seq),
        "--sampling_temp", str(temp),
        "--batch_size", str(bs),
    ]
    log(f"  CMD: {' '.join(run_cmd)}")
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  STDOUT: {result.stdout}")
        log(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"protein_mpnn_run.py failed: {result.stderr}")
    log(f"  MPNN run complete")

    # ---- Step 4: Parse output FASTAs into summary CSV ----
    seqs_dir = mpnn_out_dir / "seqs"
    if not seqs_dir.exists():
        raise FileNotFoundError(f"MPNN did not produce output in {seqs_dir}")

    fa_files = sorted(seqs_dir.glob("*.fa"))
    log(f"\n[Stage 2] Step 4: Parsing {len(fa_files)} FASTA files...")

    all_records: List[Dict] = []
    for fa in fa_files:
        records = parse_mpnn_fasta(str(fa), peptide_chain)
        all_records.extend(records)

    if not all_records:
        log("  WARNING: No sequences parsed from MPNN output!")
        log("  Check that batch_size <= num_seq_per_target")
        return pd.DataFrame()

    summary_df = pd.DataFrame(all_records)
    summary_df = summary_df.sort_values(["design", "sample"]).reset_index(drop=True)

    summary_csv = outpath / "stage2_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    n_designs = summary_df["design"].nunique()
    n_seqs = len(summary_df)
    log(f"\n{'=' * 60}")
    log(f"STAGE 2 DONE: {n_seqs} sequences from {n_designs} designs")
    log(f"  MPNN output FASTAs -> {seqs_dir}/")
    log(f"  Summary CSV        -> {summary_csv}")
    log(f"  Score range: {summary_df['mpnn_score'].min():.3f} - {summary_df['mpnn_score'].max():.3f} (lower is better)")
    log("=" * 60)

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2: ProteinMPNN sequence design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "-o", "--outdir", default="pipeline_results",
        help="Top-level output directory (reads from <outdir>/stage1/, writes to <outdir>/stage2/)",
    )
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--mpnn-path", default=None, help="Path to ProteinMPNN installation")
    ap.add_argument("--num-seq-per-target", type=int, default=None, help="Sequences per backbone")
    ap.add_argument("--sampling-temp", default=None, help="Sampling temperature")
    ap.add_argument("--batch-size", type=int, default=None, help="GPU batch size (<= num-seq-per-target)")

    args = ap.parse_args()
    config = load_config(args.config) if args.config else {}

    run_stage2(
        outdir=args.outdir,
        config=config,
        mpnn_path=args.mpnn_path,
        num_seq_per_target=args.num_seq_per_target,
        sampling_temp=args.sampling_temp,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
