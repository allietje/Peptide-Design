# Peptide-Design

Tools and workflows for **computational peptide design** and **evaluation** (e.g. diffusion-generated backbones, sequence design, structure prediction, and ranking).

This repository is meant to be **portable**: clone it, install dependencies on your machine or HPC cluster, add your own `config.yaml` (see below), and run.

## Contents (high level)

| Path | Description |
|------|-------------|
| `final_pipeline/` | Multi-stage pipeline: structural filters → ProteinMPNN → sequence filters → ColabFold (AF2-Multimer) → confidence metrics → optional Rosetta → ranking. See `final_pipeline/README.md` and `final_pipeline/instructions.md`. |
| `requirements.txt` | Python dependencies for tools in this repo (when provided). |

Other scripts or subfolders may appear on different branches; browse the repository tree for your checkout.

## Quick start (`final_pipeline`)

1. **Python** 3.10+ with `numpy`, `pandas`, `biopython`, `pyyaml` (see `requirements.txt` at repo root if provided).

2. **Configuration** (nothing secret in git):
   ```bash
   cd final_pipeline
   cp example_config.yaml config.yaml
   # Edit config.yaml: receptor/peptide chains, MPNN path, SLURM partition, thresholds.
   ```

3. **External tools** (install separately; paths go only in your local `config.yaml`):
   - [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
   - [ColabFold](https://github.com/sokrypton/ColabFold) (+ GPU-capable JAX) for AlphaFold2-Multimer folding
   - Optional: Rosetta, DeepTMHMM / `biolib` for topology checks in Stage 5

4. **Run** (from `final_pipeline/`):
   ```bash
   python run_all.py path/to/reference.pdb path/to/designs/*.pdb -o results/ --config config.yaml
   ```

See **`final_pipeline/README.md`** for stage overview, outputs, and **`final_pipeline/instructions.md`** for detailed usage and troubleshooting.

## What not to commit

- **`final_pipeline/config.yaml`** if it contains **local paths**, tokens, or site-specific settings (use **`example_config.yaml`** as the template you *do* commit).
- Large run directories (`results/`, `msa_cache/`, ColabFold outputs, etc.) — add them to `.gitignore` unless you intentionally version small fixtures.

## Contributing / branches

Use feature branches and pull requests as usual. Default development branch name in this repo may be `working` or another name—check on GitHub.

## Citation

If you use this software in a publication, cite the relevant underlying methods (RFdiffusion, ProteinMPNN, AlphaFold2/ColabFold, etc.) and this repository as appropriate for your field.
