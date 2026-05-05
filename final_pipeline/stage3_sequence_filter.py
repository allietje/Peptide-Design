#!/usr/bin/env python3
"""
Stage 3: Post-MPNN sequence filters.

Reads the Stage 2 summary CSV and applies sequence-level heuristics to
filter or flag designed peptide sequences before expensive AF2 folding.

Reads from:  <outdir>/stage2/stage2_summary.csv
Writes to:   <outdir>/stage3/
  - stage3_summary.csv       -- per-sequence metrics + pass/flag columns
  - filtered_sequences.fasta -- FASTA of sequences that passed (for Stage 4)

Metrics computed:
  - mpnn_score        : MPNN negative log-likelihood (from Stage 2, lower = better)
  - net_charge        : net charge at pH 7
  - hydrophobic_frac  : fraction of hydrophobic residues
  - sequence_length   : peptide length

Usage:
  python stage3_sequence_filter.py -o pipeline_results/
  python stage3_sequence_filter.py --config config.yaml -o pipeline_results/
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import get_threshold, load_config, log

HYDROPHOBIC = set("AVLIMFWP")
POSITIVE = set("KR")
NEGATIVE = set("DE")


def compute_sequence_metrics(row: Dict) -> Dict:
    """Compute sequence-level metrics for a single designed peptide."""
    seq = row.get("peptide_sequence", "")

    n_pos = sum(1 for aa in seq if aa in POSITIVE)
    n_his = sum(1 for aa in seq if aa == "H")
    n_neg = sum(1 for aa in seq if aa in NEGATIVE)
    net_charge = n_pos + 0.1 * n_his - n_neg

    n_hydro = sum(1 for aa in seq if aa in HYDROPHOBIC)
    hydro_frac = n_hydro / len(seq) if len(seq) > 0 else 0.0

    return {
        "net_charge": round(net_charge, 1),
        "abs_net_charge": round(abs(net_charge), 1),
        "hydrophobic_frac": round(hydro_frac, 4),
        "sequence_length": len(seq),
    }


def apply_thresholds(row: Dict, config: Dict) -> Dict:
    """Apply Stage 3 thresholds. Returns row with pass_* columns and overall pass."""
    checks = [
        ("max_mpnn_score", "mpnn_score", "max"),
        ("max_abs_net_charge", "abs_net_charge", "max"),
        ("max_hydrophobic_fraction", "hydrophobic_frac", "max"),
    ]

    overall_pass = True
    for threshold_name, metric_name, direction in checks:
        value = row.get(metric_name, float("inf"))
        tv, mode = get_threshold(config, "stage3", threshold_name)

        if mode == "ignore" or tv is None:
            row[f"pass_{threshold_name}"] = True
            continue

        if isinstance(value, float) and np.isnan(value):
            passes = False
        elif direction == "max":
            passes = value <= tv
        else:
            passes = value >= tv

        row[f"pass_{threshold_name}"] = passes

        if not passes and mode == "filter":
            overall_pass = False

    row["pass_stage3"] = overall_pass
    return row


def run_stage3(outdir: str, config: Dict) -> pd.DataFrame:
    """Run Stage 3: sequence-level filters on MPNN output."""
    base_path = Path(outdir)
    stage2_path = base_path / "stage2"
    outpath = base_path / "stage3"
    outpath.mkdir(parents=True, exist_ok=True)

    stage2_csv = stage2_path / "stage2_summary.csv"
    if not stage2_csv.exists():
        raise FileNotFoundError(f"stage2_summary.csv not found in {stage2_path}")

    df = pd.read_csv(stage2_csv)
    log("=" * 60)
    log("STAGE 3: Sequence Filters")
    log("=" * 60)
    log(f"  Input: {len(df)} sequences from {df['design'].nunique()} designs")

    rows: List[Dict] = []
    for _, r in df.iterrows():
        row = r.to_dict()
        metrics = compute_sequence_metrics(row)
        row.update(metrics)
        row = apply_thresholds(row, config)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    output_cols = [
        "design", "sample", "peptide_sequence",
        "mpnn_score", "global_score",
        "net_charge", "abs_net_charge", "hydrophobic_frac", "sequence_length",
        "pass_max_mpnn_score", "pass_max_abs_net_charge", "pass_max_hydrophobic_fraction",
        "pass_stage3",
    ]
    existing_cols = [c for c in output_cols if c in summary_df.columns]
    summary_df = summary_df[existing_cols]

    summary_csv = outpath / "stage3_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # -- Write filtered sequences FASTA for Stage 4 --
    passed = summary_df[summary_df["pass_stage3"]]
    fasta_path = outpath / "filtered_sequences.fasta"
    with open(fasta_path, "w") as f:
        for _, row in passed.iterrows():
            design = row["design"]
            sample = int(row["sample"])
            seq = row["peptide_sequence"]
            score = row.get("mpnn_score", 0)
            f.write(f">{design}_s{sample} mpnn_score={score:.4f}\n{seq}\n")

    n_pass = passed.shape[0]
    n_total = len(summary_df)
    n_designs_pass = passed["design"].nunique() if n_pass > 0 else 0

    # -- Report failure reasons --
    failed = summary_df[~summary_df["pass_stage3"]]
    log(f"\n  Results: {n_pass}/{n_total} sequences passed ({n_designs_pass} unique designs)")
    if len(failed) > 0:
        log("  Failure reasons:")
        for col in [c for c in summary_df.columns if c.startswith("pass_") and c != "pass_stage3"]:
            n_fail = (~failed[col]).sum()
            if n_fail > 0:
                log(f"    {col}: {n_fail}/{len(failed)} failed")

    # -- Flag report --
    for col in [c for c in summary_df.columns if c.startswith("pass_")]:
        if col == "pass_stage3":
            continue
        threshold_name = col.replace("pass_", "")
        _, mode = get_threshold(config, "stage3", threshold_name)
        if mode == "flag":
            n_flagged = (~summary_df[col]).sum()
            if n_flagged > 0:
                log(f"  FLAG: {n_flagged}/{n_total} sequences flagged by {threshold_name}")

    # -- Metric distributions --
    log(f"\n  Metric distributions (all {n_total} sequences):")
    for col in ["mpnn_score", "net_charge", "hydrophobic_frac"]:
        if col in summary_df.columns:
            log(f"    {col:25s}  min={summary_df[col].min():7.3f}  "
                f"median={summary_df[col].median():7.3f}  max={summary_df[col].max():7.3f}")

    log(f"\n{'=' * 60}")
    log(f"STAGE 3 DONE: {n_pass}/{n_total} sequences passed")
    log(f"  Summary CSV      -> {summary_csv}")
    log(f"  Filtered FASTA   -> {fasta_path} ({n_pass} sequences)")
    log("=" * 60)

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 3: Post-MPNN sequence filters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "-o", "--outdir", default="pipeline_results",
        help="Top-level output directory (reads from <outdir>/stage2/, writes to <outdir>/stage3/)",
    )
    ap.add_argument("--config", default=None, help="Path to config.yaml")

    args = ap.parse_args()
    config = load_config(args.config) if args.config else {}

    run_stage3(outdir=args.outdir, config=config)


if __name__ == "__main__":
    main()
