# GPCR Peptide Design Pipeline

This document mirrors the full operator guide without site-specific paths, hostnames, or partition names. Copy `example_config.yaml` to `config.yaml` locally and set paths there.

## Overview

This pipeline evaluates RFDiffusion-generated peptide–GPCR designs through seven stages:

1. **Stage 1** — Structural filters (clashes, pocket contact, hotspots)
2. **Stage 2** — ProteinMPNN sequence design
3. **Stage 3** — Sequence-level filters (MPNN score, charge, hydrophobicity)
4. **Stage 4** — ColabFold AF2-Multimer folding (GPU, SLURM)
5. **Stage 5** — AlphaFold confidence scoring (ipTM, iPAE, ipLDDT, iLIS, pocket depth)
6. **Stage 6** — Rosetta refinement (optional, disabled by default)
7. **Stage 7** — Final composite ranking

Stages 1–3 are fast CPU filters. Stage 4 requires GPUs via SLURM. Stages 5 and 7 are fast CPU scoring and ranking.

## Prerequisites

- **Python 3.10+** with packages: `numpy`, `pandas`, `biopython`, `pyyaml`
- **ProteinMPNN** installed (absolute path set in `config.yaml`)
- **ColabFold** conda environment with GPU support (see “ColabFold Setup” below)
- **SLURM** cluster access with a suitable GPU partition
- **Environment**: use your site’s pattern for conda (e.g. `module load` for Miniforge/Mamba, or a login-node conda init)

## Quick Start

### Full pipeline (recommended)

From the `final_pipeline` directory of this repository:

```bash
cd <path-to-repo>/final_pipeline

# Stages 1–3 + prepare Stage 4 SLURM script
python run_all.py reference.pdb designs/*.pdb \
    -o results/ --config config.yaml

# Submit the SLURM job for AF2 folding
sbatch results/stage4/fold_array.sh

# Monitor progress
watch -n 30 'ls results/stage4/af_outputs/*/*.done.txt 2>/dev/null | wc -l'

# After all folds complete, run scoring + ranking
python run_all.py reference.pdb -o results/ --config config.yaml --resume-from stage5
```

Replace `<path-to-repo>` with your local clone location.,

### Auto-submit variant

```bash
python run_all.py reference.pdb designs/*.pdb \
    -o results/ --config config.yaml --submit
```

### Run individual stages

```bash
python stage1_setup_and_filter.py reference.pdb designs/*.pdb -o results/ --config config.yaml
python stage2_mpnn.py -o results/ --config config.yaml
python stage3_sequence_filter.py -o results/ --config config.yaml
python stage4_fold.py -o results/ --config config.yaml --reference reference.pdb
python stage5_af_score.py results/ --config config.yaml
python stage7_rank.py results/ --config config.yaml
```

## Directory Structure

After a full run, the output directory looks like:

```
results/
  stage1/
    regions.json
    renumbered/
    pass_designs/
    stage1_summary.csv
  stage2/
    mpnn_outputs/
    stage2_summary.csv
  stage3/
    filtered_sequences.fasta
    stage3_summary.csv
  stage4/
    inputs/
    chunks/
    af_outputs/
      <design_name>/
        *_unrelaxed_rank_*.pdb
        *_scores_rank_*.json
    fold_array.sh
    slurm_logs/
  stage5/
    stage5_scores.csv
    stage5_aggregated.csv
  stage7/
    final_ranked.csv
```

## Configuration: `config.yaml`

Use `example_config.yaml` as a template and copy/rename into config.yaml when running.

### Threshold modes

- `**filter**` — Hard cutoff; failing designs are excluded downstream.
- `**flag**` — Soft warning; `pass_*` columns in CSV; design kept.
- `**ignore**` — Metric computed; threshold not applied.

### Tuning thresholds (example)

```yaml
stage5:
  thresholds:
    min_ipTM: { value: 0.4, mode: filter }
    min_ipLDDT: { value: 60, mode: flag }
    min_pocket_depth: { value: -5.0, mode: ignore }
```

### Ranking weights (Stage 7)

```yaml
stage7:
  weights:
    iLIS: 0.25
    ipTM: 0.20
    ipLDDT: 0.20
    iPAE: 0.20
    pocket_depth: 0.10
    topology: 0.05
```

## Stage Details (summary)

### Stage 1: Structural filters

Key metrics include head-in-pocket counts, clash counts, optional hotspot contacts, and peptide–pocket distance (default ~1–3 s per design).

### Stage 2: ProteinMPNN

`num_seq_per_target`, `sampling_temp`, and `mpnn_path` in `config.yaml`. Receptor chain fixed; peptide chain redesigned.

### Stage 3: Sequence filters

MPNN score, net charge, hydrophobic fraction — see `example_config.yaml` for defaults.

### Stage 4: ColabFold AF2-Multimer

MSA reuse via a cache directory under `final_pipeline` (configurable). Multi-GPU via SLURM array; resource limits set in `config.yaml` (`slurm_partition`, `slurm_time`, `slurm_mem`).

The generated job script must match **your** cluster’s module/conda pattern (edit the pipeline or wrapper if your site differs).

### Stage 5: AF confidence scoring

Aggregates metrics across model seeds (median ± MAD). Accepts ColabFold output trees or AlphaFold server ZIPs.

### Stage 7: Ranking

Weighted composite score with min–max normalization across the design set; see formula in the non-git operator doc or `stage7_rank.py`.

## ColabFold Setup (generic)

Create a GPU-capable environment using your cluster’s documented method, then install ColabFold and JAX with CUDA support per [ColabFold](https://github.com/sokrypton/ColabFold) instructions for your CUDA version.

Verify GPU from inside the activated environment:

```bash
python -c "import jax; print(jax.devices())"
```

Expect CUDA devices, not CPU-only.

## Monitoring SLURM jobs

```bash
squeue -u $USER
ls results/stage4/af_outputs/*/*.done.txt 2>/dev/null | wc -l
watch -n 30 'ls results/stage4/af_outputs/*/*.done.txt 2>/dev/null | wc -l'
tail -f results/stage4/slurm_logs/fold_*.out
scancel <job_id>
```

## Interpreting results

`final_ranked.csv` includes rank, `composite_score`, design id, aggregated AF metrics, stage pass flags, MPNN score, peptide sequence, and normalized score components. Strong candidates often show high ipTM / ipLDDT / iLIS and low iPAE; treat thresholds as starting points.

## Troubleshooting (generic)

- **No GPU in Stage 4** — Reinstall JAX/CUDA stack in the ColabFold environment; confirm the GPU node sees the GPU.
- **MSA dimension mismatch** — Remove the stale receptor-only cache file in `msa_cache/` (see `example_config.yaml` for the cache dir key) and rerun so the cache regenerates.
- **DeepTMHMM missing** — Topology check may be skipped or install via your preferred package manager (`biolib` / project docs).
- `**No module named Bio`** — Install Biopython in the Python environment used for Stages 1–3.
- **Jobs pending** — Check partition load with `sinfo` / your scheduler docs; adjust `slurm_partition` (and quotas) in `config.yaml`.

