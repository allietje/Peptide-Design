# GPCR Peptide Design Pipeline V2

A multi-stage computational pipeline for evaluating and ranking RFDiffusion-generated peptide designs against GPCR targets. It filters designs through structural checks, sequence design, and AlphaFold2 confidence scoring to identify the most promising candidates for experimental validation.

**Configuration:** copy `example_config.yaml` to `config.yaml` and edit paths and SLURM settings for your environment. Commit `example_config.yaml`, not necessarily your private `config.yaml`.

## Pipeline Overview

```
RFDiff PDB designs + Reference PDB
        |
        v
  Stage 1: Structural Filters          (~5 min, CPU)
  Clashes, pocket contact, hotspots
        |
        v
  Stage 2: ProteinMPNN                 (~10 min, GPU)
  Peptide sequence design
        |
        v
  Stage 3: Sequence Filters            (instant, CPU)
  MPNN score, charge, hydrophobicity
        |
        v
  Stage 4: ColabFold AF2-Multimer      (~2-4 hrs, multi-GPU SLURM)
  Structure prediction with MSA reuse
        |
        v
  Stage 5: AF Confidence Scoring       (~30 sec, CPU)
  ipTM, iPAE, ipLDDT, iLIS, depth
        |
        v
  Stage 6: Rosetta Refinement          (optional, hours)
        |
        v
  Stage 7: Final Ranking               (instant, CPU)
  Weighted composite score -> final_ranked.csv
```

**Typical scale:** 200 RFDiff designs in -> ~100 pass Stage 1 -> x4 MPNN = ~400 sequences -> ~350 pass Stage 3 -> AF2 on all -> ~50-100 pass Stage 5 -> ranked list.

**Total runtime (no Rosetta):** ~3-5 hours on 4 GPUs.

## Setup

### Python dependencies

Use your preferred environment (venv, conda, or HPC modules), then:

```bash
pip install numpy pandas biopython pyyaml
```

### ProteinMPNN

Set `stage2.mpnn_path` in `config.yaml` to your clone. To install:

```bash
git clone https://github.com/dauparas/ProteinMPNN.git /path/to/ProteinMPNN
```

Update `config.yaml`:

```yaml
stage2:
  mpnn_path: /path/to/ProteinMPNN
```

### ColabFold (required for Stage 4)

```bash
mamba create -n colabfold python=3.11 -y
mamba activate colabfold
pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
pip install "jax[cuda12]"
```

Verify GPU detection:

```bash
conda activate colabfold
python -c "import jax; print(jax.devices())"
# Should show [CudaDevice(...)], NOT [CpuDevice()]
```

### MSA cache (automatic)

The MSA cache is fully automatic. On the first run with a new receptor:

1. Stage 4 detects that no MSA cache exists (or the cached one is for a different receptor)
2. SLURM task 0 runs a single ColabFold warmup fold to generate the receptor MSA (~10 min)
3. The receptor-only MSA is extracted and saved to `msa_cache/receptor_only.a3m`
4. A `receptor_seq.txt` is saved alongside for validation on future runs
5. All SLURM tasks then proceed using the cached MSA

On subsequent runs with the same receptor, the cache is reused immediately. If you switch to a different receptor (e.g., GLP-2R), the pipeline detects the mismatch and regenerates automatically.

### SLURM

The pipeline generates SLURM array job scripts for Stage 4. Configure your partition and resources in `config.yaml`:

```yaml
stage4:
  num_gpus: 4
  slurm_partition: your_gpu_partition_name
  slurm_time: "04:00:00"
  slurm_mem: 64G
```

### DeepTMHMM (optional)

Used in Stage 5 for transmembrane topology plausibility checks. If not installed, the check is skipped.

```bash
pip install biolib
```

## Quick Start

```bash
cd final_pipeline   # or full path to this directory on your system

# Run stages 1-3 + prepare Stage 4 SLURM script
python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml

# Submit AF2 folding to SLURM
sbatch results/stage4/fold_array.sh

# Monitor progress
watch -n 30 'ls results/stage4/af_outputs/*/*.done.txt 2>/dev/null | wc -l'

# After all folds complete, score and rank
python run_all.py reference.pdb -o results/ --config config.yaml --resume-from stage5
```

Output: `results/stage7/final_ranked.csv`

### Other run modes

```bash
# Auto-submit SLURM job
python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml --submit

# Run only stages 1-3 (no GPU needed)
python run_all.py reference.pdb designs/*.pdb -o results/ --config config.yaml --stop-after stage3

# Score external AF outputs (e.g., web server ZIPs)
python run_all.py reference.pdb -o results/ --config config.yaml \
    --resume-from stage5 --af-dir /path/to/af_outputs/
```

### Running stages individually

```bash
python stage1_setup_and_filter.py reference.pdb designs/*.pdb -o results/ --config config.yaml
python stage2_mpnn.py -o results/ --config config.yaml
python stage3_sequence_filter.py -o results/ --config config.yaml
python stage4_fold.py -o results/ --config config.yaml --reference reference.pdb
python stage5_af_score.py results/ --config config.yaml
python stage7_rank.py results/ --config config.yaml
```

## Output Structure

```
results/
  stage1/
    regions.json              # pocket/head/hotspot definitions
    renumbered/               # renumbered PDBs
    pass_designs/             # PDBs passing structural filters
    stage1_summary.csv
  stage2/
    mpnn_outputs/             # ProteinMPNN raw outputs
    stage2_summary.csv
  stage3/
    filtered_sequences.fasta  # sequences for AF2 folding
    stage3_summary.csv
  stage4/
    inputs/                   # per-design FASTA files
    af_outputs/<design>/      # ColabFold outputs (PDB + JSON per model)
    fold_array.sh             # SLURM script
  stage5/
    stage5_scores.csv         # per-(design, model_seed) metrics
    stage5_aggregated.csv     # per-design aggregated metrics
  stage7/
    final_ranked.csv          # all designs ranked by composite score
```

## Interpreting Results

Key columns in `final_ranked.csv`:

| Column | Meaning | Good range |
|--------|---------|-----------|
| `rank` | Overall rank (1 = best) | |
| `composite_score` | Weighted score (0-1) | higher = better |
| `ipTM` | AF interface confidence | >= 0.7 |
| `iPAE` | Interface position error (A) | <= 7 |
| `ipLDDT` | Interface local confidence | >= 70 |
| `iLIS` | Confident sub-interface fraction | >= 0.4 |
| `mpnn_score` | Sequence-backbone fit | <= 1.5 |

Designs in the top 10-20% by composite score are good candidates for experimental validation.

## Files

| File | Purpose |
|------|---------|
| `run_all.py` | Unified runner |
| `example_config.yaml` | Template config (no site-specific paths); copy to `config.yaml` |
| `config.yaml` | Your local parameters (create from example; often gitignored) |
| `stage[1-7]_*.py` | Individual stage scripts |
| `build_msa_for_design.py` | Builds per-design paired MSA from receptor cache |
| `utils.py` | Shared helpers |
| `msa_cache/` | Pre-computed receptor MSA |
| `instructions.md` | Full usage guide + troubleshooting |
