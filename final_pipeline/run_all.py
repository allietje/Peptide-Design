#!/usr/bin/env python3
"""
Unified runner for the GPCR Peptide Design Pipeline V2.

Runs stages 1 -> 2 -> 3 -> 4 (prepare) -> [manual SLURM submit] -> 5 -> 7.
Stage 4 (ColabFold) requires SLURM GPU submission, so by default the pipeline
pauses after preparing inputs and generating the SLURM script.  Use --submit
to auto-submit, and --resume-from stage5 to continue after folds complete.

Usage examples:

    # Run stages 1-3, prepare stage 4 SLURM script (pause before GPU work)
    python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml

    # Same, but auto-submit the SLURM job
    python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml --submit

    # Resume after AF2 folds are done (runs stages 5 + 7)
    python run_all.py reference.pdb -o results/ --config config.yaml --resume-from stage5

    # Only run stages 1-3 (pre-AF, no GPU needed)
    python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml --stop-after stage3

    # Run from stage 5 with external AF outputs (e.g., web server ZIPs)
    python run_all.py reference.pdb -o results/ --config config.yaml \\
        --resume-from stage5 --af-dir /path/to/af_outputs/
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, log

PIPELINE_DIR = Path(__file__).resolve().parent
STAGES_ORDERED = ["stage1", "stage2", "stage3", "stage4", "stage5", "stage7"]


def _stage_index(name: str) -> int:
    try:
        return STAGES_ORDERED.index(name)
    except ValueError:
        raise ValueError(f"Unknown stage '{name}'. Valid: {STAGES_ORDERED}")


def run_stage1(args, config):
    from stage1_setup_and_filter import run_stage1 as _run
    design_pdbs = args.design_pdbs or []
    if not design_pdbs:
        raise ValueError("Stage 1 requires design PDB files (positional args after reference_pdb).")
    _run(
        reference_pdb=args.reference_pdb,
        design_pdbs=design_pdbs,
        outdir=args.outdir,
        config=config,
    )


def run_stage2(args, config):
    from stage2_mpnn import run_stage2 as _run
    _run(outdir=args.outdir, config=config)


def run_stage3(args, config):
    from stage3_sequence_filter import run_stage3 as _run
    _run(outdir=args.outdir, config=config)


def run_stage4(args, config):
    from stage4_fold import prepare_inputs, generate_slurm_script
    fasta_paths, receptor_seq = prepare_inputs(
        outdir=args.outdir,
        config=config,
        reference_pdb=args.reference_pdb,
    )
    script_path = generate_slurm_script(
        outdir=args.outdir,
        config=config,
        fasta_paths=fasta_paths,
        receptor_seq=receptor_seq,
    )
    log(f"\n[run_all] Stage 4 SLURM script ready: {script_path}")
    log(f"  {len(fasta_paths)} designs prepared for folding.")

    if args.submit:
        log("[run_all] Submitting SLURM job...")
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            log(f"  Submitted: {result.stdout.strip()}")
            log("  Monitor with: squeue -u $USER")
            log(f"  Check progress: ls {args.outdir}/stage4/af_outputs/*/*.done.txt | wc -l")
        else:
            log(f"  ERROR submitting: {result.stderr.strip()}")
            sys.exit(1)

        log("\n[run_all] PAUSING -- Stage 4 runs on SLURM GPUs.")
        log("  After all folds complete, resume with:")
        log(f"    python {__file__} {args.reference_pdb} -o {args.outdir} "
            f"--config {args.config} --resume-from stage5")
    else:
        log("\n[run_all] PAUSING -- submit the SLURM job manually:")
        log(f"    sbatch {script_path}")
        log("  After all folds complete, resume with:")
        log(f"    python {__file__} {args.reference_pdb} -o {args.outdir} "
            f"--config {args.config} --resume-from stage5")


def run_stage5(args, config):
    from stage5_af_score import run_stage5 as _run
    _run(
        outdir=args.outdir,
        config=config,
        af_dir_override=args.af_dir,
    )


def run_stage7(args, config):
    from stage7_rank import run_stage7 as _run
    _run(outdir=args.outdir, config=config)


STAGE_RUNNERS = {
    "stage1": run_stage1,
    "stage2": run_stage2,
    "stage3": run_stage3,
    "stage4": run_stage4,
    "stage5": run_stage5,
    "stage7": run_stage7,
}


def main():
    parser = argparse.ArgumentParser(
        description="GPCR Peptide Design Pipeline V2 -- unified runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (stages 1-4 prepare, then pause for SLURM)
  python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml

  # Auto-submit SLURM job for stage 4
  python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml --submit

  # Resume after AF2 folds complete
  python run_all.py reference.pdb -o results/ --config config.yaml --resume-from stage5

  # Run only pre-AF stages
  python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml --stop-after stage3
""",
    )
    parser.add_argument("reference_pdb", help="Reference PDB (cryo-EM / crystal complex)")
    parser.add_argument("design_pdbs", nargs="*", help="RFDiffusion design PDB files (needed for stage 1)")
    parser.add_argument("-o", "--outdir", default="pipeline_results", help="Output directory")
    parser.add_argument("--config", default=str(PIPELINE_DIR / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--resume-from", dest="resume_from", default=None,
                        help=f"Skip to this stage (valid: {STAGES_ORDERED})")
    parser.add_argument("--stop-after", dest="stop_after", default=None,
                        help=f"Stop after this stage (valid: {STAGES_ORDERED})")
    parser.add_argument("--submit", action="store_true",
                        help="Auto-submit SLURM job for stage 4")
    parser.add_argument("--af-dir", default=None,
                        help="Override AF output directory for stage 5 (e.g., web server results)")

    args = parser.parse_args()
    config = load_config(args.config)

    start_idx = _stage_index(args.resume_from) if args.resume_from else 0
    stop_idx = _stage_index(args.stop_after) if args.stop_after else len(STAGES_ORDERED) - 1

    stages_to_run = STAGES_ORDERED[start_idx:stop_idx + 1]

    log("=" * 60)
    log("GPCR Peptide Design Pipeline V2")
    log("=" * 60)
    log(f"  Reference PDB: {args.reference_pdb}")
    log(f"  Design PDBs:   {len(args.design_pdbs)} files" if args.design_pdbs else "  Design PDBs:   (none, resuming)")
    log(f"  Output dir:    {args.outdir}")
    log(f"  Config:        {args.config}")
    log(f"  Stages:        {' -> '.join(stages_to_run)}")
    log("=" * 60)

    for stage_name in stages_to_run:
        runner = STAGE_RUNNERS[stage_name]
        runner(args, config)

        # Stage 4 pauses for SLURM -- don't continue automatically
        if stage_name == "stage4":
            log("\n[run_all] Pipeline paused at stage 4 (SLURM GPU jobs).")
            log("  Resume after folds complete with --resume-from stage5")
            return

    log("\n" + "=" * 60)
    log("PIPELINE COMPLETE")
    log(f"  Final ranking: {args.outdir}/stage7/final_ranked.csv")
    log("=" * 60)


if __name__ == "__main__":
    main()
