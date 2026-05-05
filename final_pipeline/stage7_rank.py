#!/usr/bin/env python3
"""
Stage 7: Final composite ranking of peptide designs.

Merges metrics from all preceding stages, computes a weighted composite score
using min-max-normalized AF confidence metrics, and produces a ranked CSV.

Usage:
    python stage7_rank.py <outdir> [--config config.yaml]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, log


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. Returns 0 if all values identical."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _extract_backbone(design_id: str) -> str:
    """Extract backbone name from a design ID like 'renumbered_ppi_glp1r_14_s2'.

    Returns 'renumbered_ppi_glp1r_14'. Handles cases where '_sN' suffix is
    absent (returns as-is) and strips '.pdb' extension if present.
    """
    name = design_id.replace(".pdb", "")
    m = re.match(r"^(.+)_s(\d+)$", name)
    if m:
        return m.group(1)
    return name


def _extract_sample(design_id: str) -> int:
    """Extract MPNN sample index from design ID. Returns 0 if no suffix."""
    m = re.match(r"^.+_s(\d+)$", design_id.replace(".pdb", ""))
    return int(m.group(1)) if m else 0


def load_stage_csvs(outdir: str) -> dict:
    """Load available stage CSV files."""
    base = Path(outdir)
    csvs = {}

    for stage, subdir, fname in [
        ("stage1", "stage1", "stage1_summary.csv"),
        ("stage3", "stage3", "stage3_summary.csv"),
        ("stage5", "stage5", "stage5_aggregated.csv"),
        ("stage6", "stage6", "stage6_scores.csv"),
    ]:
        path = base / subdir / fname
        if path.exists():
            csvs[stage] = pd.read_csv(path)
            log(f"[Stage 7] Loaded {stage}: {len(csvs[stage])} rows from {path}")
        else:
            log(f"[Stage 7] {stage}: not found ({path}), skipping")

    return csvs


def merge_all(csvs: dict) -> pd.DataFrame:
    """Merge stage CSVs into a single DataFrame keyed on design ID.

    Stage 5 aggregated CSV is the primary table (one row per MPNN design).
    Stage 1 and Stage 3 metrics are joined in for context.
    Stage 6 (Rosetta) metrics are joined if available.
    """
    if "stage5" not in csvs:
        raise FileNotFoundError("Stage 5 aggregated CSV is required for ranking.")

    df = csvs["stage5"].copy()
    df["backbone"] = df["design"].apply(_extract_backbone)
    df["sample"] = df["design"].apply(_extract_sample)

    if "stage1" in csvs:
        s1 = csvs["stage1"].copy()
        s1["backbone"] = s1["design"].apply(_extract_backbone)
        s1_cols = [c for c in s1.columns if c not in ("design", "original_pdb")]
        df = df.merge(s1[s1_cols], on="backbone", how="left", suffixes=("", "_s1"))

    if "stage3" in csvs:
        s3 = csvs["stage3"].copy()
        s3["backbone"] = s3["design"].apply(_extract_backbone)
        s3_rename = {"sample": "sample_s3"}
        s3 = s3.rename(columns=s3_rename)
        s3_merge = s3[s3["sample_s3"] == df["sample"].iloc[0] if len(df) > 0 else 1].copy()
        s3_cols_to_merge = ["backbone", "sample_s3", "peptide_sequence", "mpnn_score",
                            "global_score", "net_charge", "abs_net_charge",
                            "hydrophobic_frac", "sequence_length"]
        s3_available = [c for c in s3_cols_to_merge if c in s3.columns]

        # Merge by backbone + sample
        s3_for_merge = s3.copy()
        s3_for_merge["sample"] = s3_for_merge["sample_s3"]
        merge_cols = [c for c in s3_available if c not in ("backbone", "sample_s3")]
        merge_cols_with_keys = ["backbone", "sample"] + merge_cols
        s3_for_merge_subset = s3_for_merge[[c for c in merge_cols_with_keys if c in s3_for_merge.columns]]
        df = df.merge(s3_for_merge_subset, on=["backbone", "sample"], how="left", suffixes=("", "_s3"))

    if "stage6" in csvs:
        s6 = csvs["stage6"].copy()
        if "design" in s6.columns:
            s6_cols = [c for c in s6.columns if c != "design" or c == "design"]
            df = df.merge(s6, on="design", how="left", suffixes=("", "_s6"))

    return df


def compute_composite_score(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute weighted composite score from normalized AF metrics."""
    s7_cfg = config.get("stage7", {})
    weights = s7_cfg.get("weights", {})

    w_ilis = weights.get("iLIS", 0.25)
    w_iptm = weights.get("ipTM", 0.20)
    w_ipldt = weights.get("ipLDDT", 0.20)
    w_ipae = weights.get("iPAE", 0.20)
    w_depth = weights.get("pocket_depth", 0.10)
    w_topo = weights.get("topology", 0.05)

    log(f"[Stage 7] Weights: iLIS={w_ilis}, ipTM={w_iptm}, ipLDDT={w_ipldt}, "
        f"iPAE={w_ipae}, depth={w_depth}, topo={w_topo}")

    df = df.copy()

    df["norm_iLIS"] = _minmax(df["iLIS"])
    df["norm_ipTM"] = _minmax(df["ipTM"])
    df["norm_ipLDDT"] = _minmax(df["ipLDDT"])
    df["norm_iPAE"] = _minmax(1.0 - df["iPAE"])  # invert: lower iPAE = better
    df["norm_pocket_depth"] = _minmax(df["pocket_depth"])
    df["norm_topo"] = df["topo_pass"].astype(float) if "topo_pass" in df.columns else 0.0

    df["composite_score"] = (
        w_ilis * df["norm_iLIS"]
        + w_iptm * df["norm_ipTM"]
        + w_ipldt * df["norm_ipLDDT"]
        + w_ipae * df["norm_iPAE"]
        + w_depth * df["norm_pocket_depth"]
        + w_topo * df["norm_topo"]
    )

    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("rank")

    return df


def run_stage7(outdir: str, config: dict) -> str:
    """Run Stage 7: merge all stage outputs and compute final ranking."""
    log("=" * 60)
    log("STAGE 7: Final composite ranking")
    log("=" * 60)

    csvs = load_stage_csvs(outdir)
    df = merge_all(csvs)
    log(f"[Stage 7] Merged table: {len(df)} designs, {len(df.columns)} columns")

    df = compute_composite_score(df, config)

    base = Path(outdir)
    stage7_dir = base / "stage7"
    stage7_dir.mkdir(parents=True, exist_ok=True)
    out_path = stage7_dir / "final_ranked.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")

    log(f"\n[Stage 7] Top 10 designs:")
    top_cols = ["rank", "design", "composite_score", "ipTM", "iPAE", "ipLDDT", "iLIS", "pocket_depth"]
    if "topo_pass" in df.columns:
        top_cols.append("topo_pass")
    if "mpnn_score" in df.columns:
        top_cols.append("mpnn_score")
    available_cols = [c for c in top_cols if c in df.columns]
    top10 = df.head(10)[available_cols]
    for _, row in top10.iterrows():
        parts = [f"{c}={row[c]}" for c in available_cols]
        log(f"  {' | '.join(parts)}")

    n_pass = df["pass_stage5"].sum() if "pass_stage5" in df.columns else "N/A"
    log(f"\n{'=' * 60}")
    log(f"STAGE 7 DONE: {len(df)} designs ranked")
    log(f"  Passed Stage 5 filters: {n_pass}")
    log(f"  Output -> {out_path}")
    log(f"{'=' * 60}")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Final composite ranking.")
    parser.add_argument("outdir", help="Top-level pipeline output directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    run_stage7(args.outdir, config)


if __name__ == "__main__":
    main()
