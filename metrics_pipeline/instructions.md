# Peptide Design Metrics Pipeline

Three-stage scoring pipeline for ranking RFdiffusion peptide designs against a
reference cryo-EM / crystal complex.

## Prerequisites

All scripts use the same dependencies as the rest of this repo:

```
numpy>=1.22
pandas>=1.5
biopython>=1.81
```

## Overview


| Stage          | Script              | Input                                   | Output                           | Cost            |
| -------------- | ------------------- | --------------------------------------- | -------------------------------- | --------------- |
| 0              | `define_regions.py` | Reference PDB                           | `regions.json`                   | Seconds         |
| 0.5 (optional) | `batch_renumber.py` | `regions.json` + raw design PDBs        | Renumbered PDBs in `renumbered/` | Seconds         |
| 1              | `stage1_filters.py` | `regions.json` + renumbered design PDBs | CSV with pass/fail               | Seconds         |
| 2              | `score_af_local.py` | `regions.json` + AF server ZIP(s)       | CSV with local AF metrics        | Seconds per ZIP |


## Stage 0: Define pocket and head from the reference structure

**What it does:**

1. **Pocket** = receptor residues with any heavy atom within `pocket_cutoff` (default 6 A) of any peptide heavy atom.
2. **Head** = peptide residues with any heavy atom within `head_cutoff` (default 5 A) of any pocket residue heavy atom.

The output is a `regions.json` file that freezes these definitions for all
downstream scoring.

**Usage examples:**

Auto-detect chains (receptor picked as the chain with >100 residues, peptide as  
the chain with <=100 residues): 

```bash
python metrics_pipeline/define_regions.py my_complex.pdb -o regions.json
```

Specify chain IDs explicitly when you already know them:

```bash
# GLP-1 receptor (chain R) + GLP-1 peptide (chain P)
python metrics_pipeline/define_regions.py glp1.pdb \
    --receptor-chain R --peptide-chain P \
    -o regions_glp1.json

# Oxytocin receptor (chain O) + peptide (chain P)
python metrics_pipeline/define_regions.py oxytocin.pdb \
    --receptor-chain O --peptide-chain P \
    -o regions_oxytocin.json

# Design PDB where chains are A (receptor) and B (peptide)
python metrics_pipeline/define_regions.py best_design.pdb \
    --receptor-chain A --peptide-chain B \
    -o regions.json
```

Override the distance cutoffs (defaults: pocket 6 A, head 5 A):

```bash
python metrics_pipeline/define_regions.py glp1.pdb \
    --receptor-chain R --peptide-chain P \
    --pocket-cutoff 5.5 --head-cutoff 4.5 \
    -o regions_tight.json
```

**Inspect the output** to make sure the pocket and head look reasonable before
proceeding:

```bash
python -c "
import json
d = json.load(open('regions.json'))
print('Pocket:', len(d['pocket_residues']), 'receptor residues')
print('  resnums:', [r['resnum'] for r in d['pocket_residues']])
print('Head:', len(d['head_residues']), 'peptide residues')
print('  resnums:', [r['resnum'] for r in d['head_residues']])
"
```

## Renumbering designs (required before Stage 1)

Before running Stage 1, each design PDB must be renumbered so that the receptor
residue numbers and chain IDs match the reference.

### Option A: `batch_renumber.py` (recommended -- inside `metrics_pipeline/`)

Reads the `regions.json` from Stage 0 (reference PDB path and chain IDs are  
already stored there), so you don't need to re-specify them. Assumes each design  
has exactly two protein chains and picks the longer one as the receptor. Input the design path of your design that is being renumbered.

```bash
# Renumber all designs, output to renumbered/ directory
python metrics_pipeline/batch_renumber.py regions.json design_*.pdb

# Custom output directory and filename prefix
python metrics_pipeline/batch_renumber.py regions.json design_*.pdb \
    -o my_renumbered/ --prefix renum_
```

Output goes to `renumbered/renumbered_design_0.pdb`, etc. The script prints the
exact Stage 1 command to run next.

### Option B: `renumber_pdb.py` (repo root, one file at a time)

```
python renumber_pdb.py <reference.pdb> <design.pdb> <output.pdb> [--ref-receptor-id X] [--ref-peptide-id Y]
```

```bash
# Auto-detect chains
python renumber_pdb.py glp1.pdb best_glp1_design.pdb renumbered_design_best.pdb

# Specify reference chain IDs explicitly
python renumber_pdb.py glp1.pdb best_glp1_design.pdb renumbered_design_best.pdb \
    --ref-receptor-id R --ref-peptide-id P

# Batch with a shell loop
for i in 0 1 2 3; do
    python renumber_pdb.py glp1.pdb design_${i}.pdb renumbered_design_${i}.pdb \
        --ref-receptor-id R --ref-peptide-id P
done
```

Both options produce the same result: PDBs with receptor residue numbers and
chain IDs matching the reference, which is what Stage 1 expects.

---

## Stage 1: Fast structural sanity filters

**Prerequisites:** design PDBs must already be renumbered (see above).

**What it computes per design:**


| Column                  | Meaning                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `n_head_in_pocket`      | How many head residues contact at least one pocket residue |
| `n_head_pocket_pairs`   | Total (head, pocket) residue pairs in contact              |
| `min_head_pocket_dist`  | Closest heavy-atom distance across all head-pocket pairs   |
| `mean_head_pocket_dist` | Mean of per-pair minimum distances                         |
| `n_interface_contacts`  | Total inter-chain residue pairs within the contact cutoff  |
| `n_clashes_severe`      | Heavy-atom pairs closer than 1.8 A (bad steric overlap)    |
| `n_clashes_mild`        | Heavy-atom pairs closer than 2.1 A                         |
| `pass_stage1`           | Boolean verdict                                            |


**Default pass criteria** (all must hold):

- `n_head_in_pocket >= 2`
- `n_head_pocket_pairs >= 4`
- `n_clashes_severe == 0`
- `n_clashes_mild <= 5`

Thresholds are CLI-tunable; calibrate them on your first batch.

**Usage:**

```bash
python metrics_pipeline/stage1_filters.py regions.json \
    renumbered_design_0.pdb renumbered_design_1.pdb \
    renumbered_design_2.pdb renumbered_design_3.pdb \
    -o stage1_scores.csv
```

or score all at once with a glob:

```bash
python metrics_pipeline/stage1_filters.py regions.json renumbered_design_*.pdb -o stage1_scores.csv
```

Override thresholds if needed:

```bash
python metrics_pipeline/stage1_filters.py regions.json renumbered_design_*.pdb \
    --min-head-in-pocket 3 --max-mild-clashes 3 -o stage1_scores.csv
```

## Manual AF-Multimer step

Take the designs that passed Stage 1, submit them to the AlphaFold server
manually, and download the result ZIP(s). Each ZIP is the input for Stage 2.

## Stage 2: Local AF-Multimer confidence metrics

**What it computes per model seed per ZIP:**


| Column                      | Meaning                                              |
| --------------------------- | ---------------------------------------------------- |
| `mean_pLDDT_head`           | Mean predicted LDDT over head residue atoms          |
| `mean_PAE_head_to_pocket`   | PAE sub-matrix head -> pocket                        |
| `mean_PAE_pocket_to_head`   | PAE sub-matrix pocket -> head                        |
| `mean_PAE_head_pocket_sym`  | Symmetric average of the above two                   |
| `n_head_pocket_contacts_af` | Head-pocket residue pairs in contact in the AF model |
| `ipTM`                      | Interface predicted TM-score                         |
| `ranking_score`             | AF ranking score                                     |
| `pass_plddt`                | `mean_pLDDT_head >= 80` (threshold tunable)          |
| `pass_pae`                  | `mean_PAE_head_pocket_sym <= 10` (threshold tunable) |


**Usage:**

```bash
python metrics_pipeline/score_af_local.py regions.json \
    af_output_design0.zip af_output_design1.zip \
    --af-peptide-chain A --af-receptor-chain B \
    -o stage2_scores.csv
```

The `--af-peptide-chain` / `--af-receptor-chain` flags refer to chain IDs
**inside the AF output**, which may differ from the reference PDB chain IDs.
AF server typically assigns A to the first sequence submitted and B to the
second. Check your AF output if unsure.

Adjust thresholds:

```bash
python metrics_pipeline/score_af_local.py regions.json *.zip \
    --plddt-threshold 85 --pae-threshold 8 -o stage2_scores.csv
```

## End-to-end example

```bash
# Stage 0: define regions from cryo-EM reference
python metrics_pipeline/define_regions.py glp1.pdb \
    --receptor-chain R --peptide-chain P -o regions.json

# Batch-renumber designs (uses reference info from regions.json)
python metrics_pipeline/batch_renumber.py regions.json design_*.pdb -o renumbered/

# Stage 1: score renumbered designs
python metrics_pipeline/stage1_filters.py regions.json \
    renumbered/renumbered_*.pdb -o stage1_scores.csv

# (manually submit passing designs to AF server, download ZIPs)

# Stage 2: score AF outputs
python metrics_pipeline/score_af_local.py regions.json \
    af_*.zip --af-peptide-chain A --af-receptor-chain B -o stage2_scores.csv
```

## Final ranking

Rank designs by:

1. Stage 1 pass (must pass)
2. `mean_PAE_head_pocket_sym` (lower is better)
3. `mean_pLDDT_head` (higher is better)
4. `ipTM` (higher is better, secondary)

