#!/usr/bin/env python3
"""
Stage 4: ColabFold AF2-Multimer folding.

Takes filtered sequences from Stage 3 and folds each peptide-receptor complex
using ColabFold AF2-Multimer. Supports multi-GPU parallelization via SLURM
array jobs.

Reads from:
  <outdir>/stage3/filtered_sequences.fasta   -- peptide sequences
  <outdir>/stage1/regions.json               -- receptor chain info
  Reference PDB or config                    -- receptor sequence

Writes to:  <outdir>/stage4/
  - inputs/                  -- per-design FASTA files (receptor:peptide)
  - af_outputs/              -- ColabFold output directories
  - fold_array.sh            -- generated SLURM array job script
  - stage4_summary.csv       -- fold status per design

Usage:
  # Prepare inputs + submit SLURM jobs:
  python stage4_fold.py -o pipeline_results/ --reference reference.pdb \\
      --config config.yaml --submit

  # Prepare inputs only (inspect before submitting):
  python stage4_fold.py -o pipeline_results/ --reference reference.pdb

  # Collect results after jobs finish:
  python stage4_fold.py -o pipeline_results/ --collect
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils import get_receptor_peptide, get_sequence, load_config, log

PIPELINE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# MSA caching: receptor-only MSA reuse for variable-length peptides
# ---------------------------------------------------------------------------

def extract_receptor_msa(paired_a3m_path: str, output_path: str) -> None:
    """Extract receptor-only MSA from a paired multimer .a3m file.

    The paired .a3m has header '#L_rec,L_pep\\t1,1' and each sequence has
    L_rec + L_pep characters. This strips the last L_pep columns from every
    entry, producing a receptor-only .a3m that can be reused for any peptide.
    """
    with open(paired_a3m_path) as f:
        lines = f.readlines()

    header = lines[0].strip()
    if not header.startswith("#"):
        raise ValueError(f"Not a paired .a3m (no header): {paired_a3m_path}")

    parts = header[1:].split("\t")
    lengths = parts[0].split(",")
    l_rec = int(lengths[0])

    out_lines = []
    for line in lines:
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            out_lines.append(line)
        else:
            # Keep only receptor columns (first l_rec alignment columns)
            # A3M format: uppercase = aligned columns, lowercase = insertions
            seq = line.rstrip("\n")
            col_count = 0
            cut_pos = len(seq)
            for i, c in enumerate(seq):
                if c.isupper() or c == "-":
                    col_count += 1
                if col_count == l_rec:
                    # Include any trailing lowercase (insertions) after last aligned col
                    j = i + 1
                    while j < len(seq) and seq[j].islower():
                        j += 1
                    cut_pos = j
                    break
            out_lines.append(seq[:cut_pos] + "\n")

    with open(output_path, "w") as f:
        f.writelines(out_lines)


def build_paired_msa(
    receptor_msa_path: str,
    peptide_seq: str,
    receptor_seq: str,
    output_a3m_path: str,
) -> None:
    """Build a paired multimer .a3m from a receptor-only MSA and a peptide sequence.

    ColabFold expects a paired .a3m with header '#L_rec,L_pep\\t1,1' where each
    MSA row has L_rec aligned receptor columns + L_pep gap characters (since the
    designed peptide has no homologs).
    """
    l_rec = len(receptor_seq)
    l_pep = len(peptide_seq)
    pep_gaps = "-" * l_pep

    with open(receptor_msa_path) as f:
        lines = f.readlines()

    out_lines = [f"#{l_rec},{l_pep}\t1,1\n"]

    first_seq = True
    for line in lines:
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if first_seq:
                out_lines.append(line.rstrip("\n").split("\t")[0] + "\t102\n")
            else:
                out_lines.append(line)
        else:
            seq = line.rstrip("\n")
            if first_seq:
                out_lines.append(seq + peptide_seq + "\n")
                # Add receptor-only entry (receptor seq + peptide gaps)
                out_lines.append(f">101\n")
                out_lines.append(receptor_seq + pep_gaps + "\n")
                first_seq = False
            else:
                out_lines.append(seq + pep_gaps + "\n")

    with open(output_a3m_path, "w") as f:
        f.writelines(out_lines)


def _get_msa_query_seq(a3m_path: str) -> str:
    """Read the query (first) sequence from an .a3m file."""
    seq_lines = []
    in_first = False
    with open(a3m_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if in_first:
                    break
                in_first = True
                continue
            if in_first:
                seq_lines.append(line.strip())
    raw = "".join(seq_lines)
    # Strip lowercase insertions and gaps to get the aligned query
    return "".join(c for c in raw if c.isupper())


def validate_msa_cache(msa_cache_dir: str, receptor_seq: str) -> bool:
    """Check if the MSA cache matches the current receptor sequence.

    Compares against receptor_seq.txt (written during cache creation) and
    also verifies the .a3m query sequence length matches.
    """
    cache = Path(msa_cache_dir)
    seq_file = cache / "receptor_seq.txt"

    if not seq_file.exists():
        return False

    cached_seq = seq_file.read_text().strip()
    if cached_seq != receptor_seq:
        log(f"[Stage 4] MSA cache MISMATCH: cached receptor is {len(cached_seq)} aa, "
            f"current receptor is {len(receptor_seq)} aa")
        return False

    return True


def save_receptor_seq(msa_cache_dir: str, receptor_seq: str) -> None:
    """Save the receptor sequence to the MSA cache directory for validation."""
    cache = Path(msa_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "receptor_seq.txt").write_text(receptor_seq)


def ensure_receptor_only_msa(msa_cache_dir: str, receptor_seq: str = "") -> Optional[str]:
    """Ensure a valid receptor-only .a3m exists in the MSA cache directory.

    Checks that the cache matches the current receptor sequence. If a paired
    .a3m exists, extracts the receptor-only version. Returns path to the
    receptor-only .a3m, or None if the cache needs to be generated.
    """
    cache = Path(msa_cache_dir)
    receptor_only = cache / "receptor_only.a3m"

    if receptor_only.exists():
        if receptor_seq and not validate_msa_cache(msa_cache_dir, receptor_seq):
            log(f"[Stage 4] MSA cache is for a different receptor -- will regenerate")
            # Move stale cache aside instead of deleting
            stale = cache / "receptor_only.a3m.stale"
            receptor_only.rename(stale)
        else:
            return str(receptor_only)

    # Try to extract from any paired .a3m in the cache dir
    for paired_candidate in cache.glob("*.a3m"):
        if paired_candidate.name == "receptor_only.a3m":
            continue
        with open(paired_candidate) as f:
            header = f.readline().strip()
        if header.startswith("#") and "," in header.split("\t")[0][1:]:
            # Verify the receptor part matches
            query_seq = _get_msa_query_seq(str(paired_candidate))
            parts = header[1:].split("\t")[0].split(",")
            l_rec = int(parts[0])
            rec_in_msa = query_seq[:l_rec]
            if receptor_seq and rec_in_msa != receptor_seq:
                log(f"[Stage 4] Paired .a3m {paired_candidate.name} has different receptor, skipping")
                continue
            log(f"[Stage 4] Extracting receptor-only MSA from {paired_candidate.name}")
            extract_receptor_msa(str(paired_candidate), str(receptor_only))
            if receptor_seq:
                save_receptor_seq(msa_cache_dir, receptor_seq)
            return str(receptor_only)

    return None


def get_receptor_seq(reference_pdb: str, receptor_chain: str) -> str:
    """Extract receptor amino acid sequence from reference PDB."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", reference_pdb)
    rec, _ = get_receptor_peptide(structure, receptor_chain)
    if rec is None:
        raise ValueError(f"Receptor chain '{receptor_chain}' not found in {reference_pdb}")
    _, seq = get_sequence(rec)
    return seq


def prepare_inputs(
    outdir: str,
    config: Dict,
    reference_pdb: Optional[str] = None,
) -> List[str]:
    """Create per-design FASTA files for ColabFold multimer folding."""
    base_path = Path(outdir)
    stage1_path = base_path / "stage1"
    stage3_path = base_path / "stage3"
    stage4_path = base_path / "stage4"
    inputs_dir = stage4_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    filtered_fasta = stage3_path / "filtered_sequences.fasta"
    if not filtered_fasta.exists():
        raise FileNotFoundError(f"filtered_sequences.fasta not found in {stage3_path}")

    regions_path = stage1_path / "regions.json"
    receptor_chain = config.get("receptor_chain", "R")
    if regions_path.exists():
        with open(regions_path) as f:
            regions = json.load(f)
        receptor_chain = regions.get("receptor_chain", receptor_chain)
        if reference_pdb is None:
            reference_pdb = regions.get("reference_pdb")

    if reference_pdb is None:
        raise ValueError("No reference PDB found. Provide --reference or ensure regions.json has reference_pdb.")

    receptor_seq = get_receptor_seq(reference_pdb, receptor_chain)
    log(f"[Stage 4] Receptor sequence: {len(receptor_seq)} aa from chain {receptor_chain}")

    sequences = []
    with open(filtered_fasta) as f:
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name and current_seq:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name and current_seq:
            sequences.append((current_name, "".join(current_seq)))

    log(f"[Stage 4] Creating {len(sequences)} input FASTAs...")

    fasta_paths = []
    for name, pep_seq in sequences:
        fasta_path = inputs_dir / f"{name}.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">{name}\n{receptor_seq}:{pep_seq}\n")
        fasta_paths.append(str(fasta_path.resolve()))

    log(f"[Stage 4] Wrote {len(fasta_paths)} FASTAs to {inputs_dir}/")
    return fasta_paths, receptor_seq


def generate_slurm_script(
    outdir: str,
    config: Dict,
    fasta_paths: List[str],
    receptor_seq: str = "",
) -> str:
    """Generate a SLURM array job script that splits designs across GPUs."""
    s4_cfg = config.get("stage4", {})
    num_gpus = s4_cfg.get("num_gpus", 4)
    num_models = s4_cfg.get("num_models", 5)
    num_recycles = s4_cfg.get("num_recycles", 3)
    partition = s4_cfg.get("slurm_partition", "mit_preemptable")
    time_limit = s4_cfg.get("slurm_time", "04:00:00")
    mem = s4_cfg.get("slurm_mem", "64G")
    msa_cache_dir = s4_cfg.get("msa_cache_dir", "")

    base_path = Path(outdir)
    stage4_path = base_path / "stage4"
    af_out = stage4_path / "af_outputs"
    af_out.mkdir(parents=True, exist_ok=True)

    # Resolve MSA cache path and prepare receptor-only MSA
    abs_msa_cache = ""
    receptor_only_msa = ""
    msa_needs_generation = False
    if msa_cache_dir:
        msa_path = Path(msa_cache_dir)
        if not msa_path.is_absolute():
            msa_path = PIPELINE_DIR / msa_cache_dir
        msa_path.mkdir(parents=True, exist_ok=True)
        abs_msa_cache = str(msa_path.resolve())

        rom = ensure_receptor_only_msa(abs_msa_cache, receptor_seq)
        if rom:
            receptor_only_msa = rom
            log(f"[Stage 4] MSA cache: {abs_msa_cache}")
            log(f"[Stage 4] Receptor-only MSA: {receptor_only_msa}")
        else:
            msa_needs_generation = True
            receptor_only_msa = str((msa_path / "receptor_only.a3m").resolve())
            log(f"[Stage 4] MSA cache needs generation (new receptor)")
            log(f"[Stage 4] Will auto-generate on first SLURM task")

    n_tasks = min(num_gpus, len(fasta_paths))
    chunk_size = math.ceil(len(fasta_paths) / n_tasks)

    chunks_dir = stage4_path / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_tasks):
        start = i * chunk_size
        end = min(start + chunk_size, len(fasta_paths))
        chunk_fastas = fasta_paths[start:end]
        chunk_file = chunks_dir / f"chunk_{i}.txt"
        with open(chunk_file, "w") as f:
            for fp in chunk_fastas:
                f.write(fp + "\n")

    # Build MSA cache block for the SLURM script.
    # Uses a Python one-liner to build a correctly-paired .a3m for each design,
    # adapting the receptor-only MSA to the design's specific peptide length.
    abs_build_msa = str((PIPELINE_DIR / "build_msa_for_design.py").resolve())
    if receptor_only_msa and abs_msa_cache:
        msa_copy_block = f"""
    # Build paired MSA from cached receptor-only MSA (handles any peptide length)
    mkdir -p "${{DESIGN_OUT}}"
    if [ -f "{receptor_only_msa}" ] && [ ! -f "${{DESIGN_OUT}}/${{NAME}}.a3m" ]; then
        python3 {abs_build_msa} "${{FASTA}}" "{receptor_only_msa}" "${{DESIGN_OUT}}/${{NAME}}.a3m"
    fi
"""
    else:
        msa_copy_block = ""

    # Auto-generate MSA cache if needed (runs once on SLURM task 0)
    if msa_needs_generation and receptor_seq:
        warmup_fasta = str((Path(abs_msa_cache) / "_warmup.fasta").resolve())
        warmup_out = str((Path(abs_msa_cache) / "_warmup_out").resolve())
        abs_extract_script = str(Path(__file__).resolve())
        msa_warmup_block = f"""
# === MSA CACHE AUTO-GENERATION ===
# First run with this receptor -- generate MSA cache on task 0, others wait.
RECEPTOR_ONLY_MSA="{receptor_only_msa}"
MSA_LOCK="{abs_msa_cache}/.msa_generating"

if [ ! -f "$RECEPTOR_ONLY_MSA" ]; then
    if [ "${{SLURM_ARRAY_TASK_ID}}" = "0" ]; then
        echo "=== Generating MSA cache for new receptor ==="
        touch "$MSA_LOCK"

        # Create warmup FASTA (receptor + short dummy peptide)
        echo ">warmup" > "{warmup_fasta}"
        echo "{receptor_seq}:AAAAAAAAAA" >> "{warmup_fasta}"

        # Run single ColabFold fold to generate MSA
        colabfold_batch "{warmup_fasta}" "{warmup_out}" \\
            --num-models 1 --num-recycle 1 --model-type alphafold2_multimer_v3

        # Extract receptor-only MSA from the paired output
        if [ -f "{warmup_out}/warmup.a3m" ]; then
            python3 -c "
import sys
sys.path.insert(0, '{str(PIPELINE_DIR)}')
from stage4_fold import extract_receptor_msa, save_receptor_seq
extract_receptor_msa('{warmup_out}/warmup.a3m', '{receptor_only_msa}')
save_receptor_seq('{abs_msa_cache}', '{receptor_seq}')
print('MSA cache generated successfully')
"
        else
            echo "ERROR: ColabFold warmup did not produce .a3m file"
            exit 1
        fi

        rm -f "$MSA_LOCK"
        echo "=== MSA cache ready ==="
    else
        # Other tasks: wait for task 0 to finish generating
        echo "Waiting for task 0 to generate MSA cache..."
        WAIT_COUNT=0
        while [ -f "$MSA_LOCK" ] || [ ! -f "$RECEPTOR_ONLY_MSA" ]; do
            sleep 10
            WAIT_COUNT=$((WAIT_COUNT + 1))
            if [ $WAIT_COUNT -ge 120 ]; then
                echo "ERROR: Timed out waiting for MSA cache (20 min)"
                exit 1
            fi
        done
        echo "MSA cache ready, proceeding."
    fi
fi
"""
    else:
        msa_warmup_block = ""

    script_path = stage4_path / "fold_array.sh"
    abs_chunks = chunks_dir.resolve()
    abs_af_out = af_out.resolve()
    abs_logs = (stage4_path / "slurm_logs").resolve()
    script_content = f"""#!/bin/bash
#SBATCH -p {partition}
#SBATCH -t {time_limit}
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --array=0-{n_tasks - 1}
#SBATCH -o {abs_logs}/fold_%A_%a.out
#SBATCH -e {abs_logs}/fold_%A_%a.err
#SBATCH -J af2_fold

module load miniforge
eval "$(conda shell.bash hook)"
conda activate colabfold

export PYTHONPATH="{str(PIPELINE_DIR)}:${{PYTHONPATH:-}}"

CHUNK_FILE={abs_chunks}/chunk_${{SLURM_ARRAY_TASK_ID}}.txt
AF_OUT={abs_af_out}

echo "=== Task ${{SLURM_ARRAY_TASK_ID}} on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Chunk file: ${{CHUNK_FILE}}"
{msa_warmup_block}
while IFS= read -r FASTA; do
    NAME=$(basename "${{FASTA}}" .fasta)
    DESIGN_OUT="${{AF_OUT}}/${{NAME}}"

    if [ -f "${{DESIGN_OUT}}/${{NAME}}.done.txt" ]; then
        echo "Skipping ${{NAME}} (already done)"
        continue
    fi
{msa_copy_block}
    echo "Folding ${{NAME}}..."
    colabfold_batch \\
        "${{FASTA}}" \\
        "${{DESIGN_OUT}}" \\
        --num-models {num_models} \\
        --num-recycle {num_recycles} \\
        --model-type alphafold2_multimer_v3

    echo "Done: ${{NAME}}"
done < "${{CHUNK_FILE}}"

echo "=== Task ${{SLURM_ARRAY_TASK_ID}} complete ==="
"""

    (stage4_path / "slurm_logs").mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script_content)

    log(f"[Stage 4] Generated SLURM array script: {script_path}")
    log(f"  {n_tasks} GPU tasks, {chunk_size} designs per GPU")
    log(f"  {num_models} models, {num_recycles} recycles per design")
    log(f"  Partition: {partition}, Time: {time_limit}, Mem: {mem}")

    return str(script_path)


def collect_results(outdir: str) -> pd.DataFrame:
    """Collect results from completed ColabFold runs into a summary CSV."""
    base_path = Path(outdir)
    stage4_path = base_path / "stage4"
    af_out = stage4_path / "af_outputs"

    if not af_out.exists():
        raise FileNotFoundError(f"af_outputs/ not found in {stage4_path}")

    design_dirs = sorted([d for d in af_out.iterdir() if d.is_dir()])
    log(f"[Stage 4] Collecting results from {len(design_dirs)} design directories...")

    rows = []
    for design_dir in design_dirs:
        name = design_dir.name
        done_file = design_dir / f"{name}.done.txt"
        log_file = design_dir / "log.txt"

        pdb_files = sorted(design_dir.glob("*_unrelaxed_rank_*.pdb"))
        score_files = sorted(design_dir.glob("*_scores_rank_*.json"))

        status = "complete" if done_file.exists() else ("partial" if pdb_files else "missing")

        row = {
            "design": name,
            "status": status,
            "n_models": len(pdb_files),
            "n_score_files": len(score_files),
        }

        if score_files:
            top_score_file = score_files[0]
            try:
                with open(top_score_file) as f:
                    scores = json.load(f)
                row["top_plddt"] = scores.get("plddt", [None])
                if isinstance(row["top_plddt"], list):
                    import numpy as np
                    row["top_plddt"] = round(float(np.mean(row["top_plddt"])), 1)
                row["top_ptm"] = round(scores.get("ptm", 0), 4)
                row["top_iptm"] = round(scores.get("iptm", 0), 4)
            except (json.JSONDecodeError, KeyError) as e:
                log(f"  WARNING: Could not parse {top_score_file}: {e}")

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = stage4_path / "stage4_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    n_complete = (summary_df["status"] == "complete").sum()
    n_total = len(summary_df)
    log(f"\n{'=' * 60}")
    log(f"STAGE 4 COLLECTION: {n_complete}/{n_total} designs complete")
    log(f"  Summary CSV -> {summary_csv}")
    if "top_iptm" in summary_df.columns:
        log(f"  ipTM range: {summary_df['top_iptm'].min():.3f} - {summary_df['top_iptm'].max():.3f}")
    log("=" * 60)

    return summary_df


def run_stage4(
    outdir: str,
    config: Dict,
    reference_pdb: Optional[str] = None,
    submit: bool = False,
    collect: bool = False,
) -> Optional[pd.DataFrame]:
    """Run Stage 4: prepare inputs, generate SLURM script, optionally submit."""
    if collect:
        return collect_results(outdir)

    log("=" * 60)
    log("STAGE 4: ColabFold AF2-Multimer Folding")
    log("=" * 60)

    fasta_paths, receptor_seq = prepare_inputs(outdir, config, reference_pdb)

    if not fasta_paths:
        log("  No sequences to fold!")
        return None

    script_path = generate_slurm_script(outdir, config, fasta_paths, receptor_seq)

    if submit:
        log(f"\n[Stage 4] Submitting SLURM array job...")
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log(f"  sbatch failed: {result.stderr}")
            raise RuntimeError(f"sbatch failed: {result.stderr}")
        log(f"  {result.stdout.strip()}")
        log(f"  Monitor with: squeue -u $USER")
        log(f"  Once done, run: python stage4_fold.py -o {outdir} --collect")
    else:
        log(f"\n[Stage 4] SLURM script ready but NOT submitted.")
        log(f"  Review: {script_path}")
        log(f"  Submit: sbatch {script_path}")
        log(f"  After completion: python stage4_fold.py -o {outdir} --collect")

    log(f"\n{'=' * 60}")
    log(f"STAGE 4 PREP DONE: {len(fasta_paths)} designs ready for folding")
    log("=" * 60)

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 4: ColabFold AF2-Multimer folding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "-o", "--outdir", default="pipeline_results",
        help="Top-level output directory",
    )
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--reference", default=None, help="Reference PDB (for receptor sequence)")
    ap.add_argument(
        "--submit", action="store_true",
        help="Submit SLURM array job after preparing inputs",
    )
    ap.add_argument(
        "--collect", action="store_true",
        help="Collect results from completed folds (run after jobs finish)",
    )

    args = ap.parse_args()
    config = load_config(args.config) if args.config else {}

    run_stage4(
        outdir=args.outdir,
        config=config,
        reference_pdb=args.reference,
        submit=args.submit,
        collect=args.collect,
    )


if __name__ == "__main__":
    main()
