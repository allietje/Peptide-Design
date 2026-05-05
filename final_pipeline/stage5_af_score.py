#!/usr/bin/env python3
"""
Stage 5: AlphaFold confidence scoring.

Parses AF outputs (ColabFold directories OR AF web server ZIPs) and computes
interface-local confidence metrics:
  - ipTM:         interface predicted TM-score (from AF summary)
  - iPAE:         median PAE at interface residue pairs
  - ipLDDT:       median pLDDT of interface peptide residues
  - iLIS:         local interaction score (confident + close pairs / total)
  - pocket_depth: peptide insertion depth along pocket axis
  - topo_pass:    topology plausibility via DeepTMHMM (optional)

Aggregates across model seeds using pessimistic median - MAD.

Inputs:
  - AF outputs: ColabFold directories (PDB + JSON) or AF web server ZIPs
  - regions.json from Stage 1 (pocket center, depth axis)
  - config.yaml

Outputs:
  <outdir>/stage5/stage5_scores.csv      -- per-(design, model_seed) raw metrics
  <outdir>/stage5/stage5_aggregated.csv  -- per-design aggregated + threshold columns

Usage:
  # ColabFold outputs (Stage 4 default):
  python stage5_af_score.py /path/to/outdir --config config.yaml

  # AF web server ZIPs:
  python stage5_af_score.py /path/to/outdir --config config.yaml \\
      --af-dir /path/to/zips/

  The script auto-detects the format:
    - Directory with *_scores_rank_*.json  -> ColabFold format
    - .zip files with *_model_*.cif        -> AF web server format
"""

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from utils import (
    aa_residues,
    check_threshold,
    get_threshold,
    heavy_atom_min_dist,
    load_config,
    log,
)


# ---------------------------------------------------------------------------
# Format adapter: normalizes ColabFold and AF web server outputs
# ---------------------------------------------------------------------------
#
# Both formats are normalized to a list of ModelData dicts:
#   {
#     "design":    str,          # design name
#     "model_idx": int,          # model index (0-4)
#     "rank":      int,          # rank (1-based for ColabFold, 0-based for web)
#     "seed":      int,          # seed index
#     "plddt":     np.ndarray,   # per-residue pLDDT (n_total,)
#     "pae":       np.ndarray,   # PAE matrix (n_total, n_total)
#     "iptm":      float,
#     "ptm":       float,
#     "structure":  Bio.PDB structure object,
#     "source":    "colabfold" | "webserver",
#     "token_chain_order": list, # ordered chain IDs matching PAE token order
#   }

def find_colabfold_models(af_dir: str) -> List[Dict]:
    """Discover all model/seed outputs in a ColabFold output directory.

    Returns list of dicts with keys: rank, model, seed, scores_json, pdb_path.
    """
    af_path = Path(af_dir)
    pattern = re.compile(
        r"_scores_rank_(\d+)_alphafold2_multimer_v3_model_(\d+)_seed_(\d+)\.json$"
    )
    models = []
    for scores_file in sorted(af_path.glob("*_scores_rank_*.json")):
        m = pattern.search(scores_file.name)
        if not m:
            continue
        rank, model, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        prefix = scores_file.name[: m.start()]
        pdb_name = f"{prefix}_unrelaxed_rank_{m.group(1)}_alphafold2_multimer_v3_model_{m.group(2)}_seed_{m.group(3)}.pdb"
        pdb_path = af_path / pdb_name
        if not pdb_path.exists():
            log(f"  WARNING: PDB not found for {scores_file.name}")
            continue
        models.append({
            "rank": rank,
            "model": model,
            "seed": seed,
            "scores_json": str(scores_file),
            "pdb_path": str(pdb_path),
        })
    return models


def load_colabfold_model(model_info: Dict) -> Dict:
    """Load a single ColabFold model into the normalized format."""
    with open(model_info["scores_json"]) as f:
        data = json.load(f)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af", model_info["pdb_path"])

    # ColabFold token order: longest chain first (receptor), then peptide
    chains = sorted(
        [c for c in structure[0] if len(aa_residues(c)) > 0],
        key=lambda c: len(aa_residues(c)), reverse=True,
    )
    token_chain_order = [c.id for c in chains]

    return {
        "rank": model_info["rank"],
        "model_idx": model_info["model"],
        "seed": model_info["seed"],
        "plddt": np.array(data["plddt"]),
        "pae": np.array(data["pae"]),
        "iptm": float(data["iptm"]),
        "ptm": float(data["ptm"]),
        "structure": structure,
        "source": "colabfold",
        "token_chain_order": token_chain_order,
    }


def find_webserver_zips(af_dir: str) -> List[str]:
    """Find AF web server ZIP files in a directory."""
    af_path = Path(af_dir)
    zips = sorted(af_path.glob("*.zip"))
    return [str(z) for z in zips if zipfile.is_zipfile(str(z))]


def load_webserver_zip(zip_path: str) -> List[Dict]:
    """Parse an AF web server ZIP into a list of normalized model dicts.

    Web server ZIPs contain 5 models (0-4), each with:
      *_model_N.cif            -- structure
      *_full_data_N.json       -- PAE matrix + per-atom pLDDT + token info
      *_summary_confidences_N.json -- ipTM, pTM, ranking_score
    """
    from Bio.PDB import MMCIFParser

    design_name = Path(zip_path).stem
    models = []

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        model_pattern = re.compile(r"_model_(\d+)\.cif$")
        model_indices = set()
        for name in names:
            m = model_pattern.search(name)
            if m:
                model_indices.add(int(m.group(1)))

        for idx in sorted(model_indices):
            cif_files = [n for n in names if n.endswith(f"_model_{idx}.cif")]
            full_data_files = [n for n in names if n.endswith(f"_full_data_{idx}.json")]
            summary_files = [n for n in names if n.endswith(f"_summary_confidences_{idx}.json")]

            if not cif_files or not full_data_files or not summary_files:
                log(f"  WARNING: incomplete model {idx} in {zip_path}")
                continue

            full_data = json.loads(zf.read(full_data_files[0]))
            summary = json.loads(zf.read(summary_files[0]))

            # Parse CIF structure from ZIP
            cif_data = zf.read(cif_files[0])
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".cif")
            try:
                with os.fdopen(tmp_fd, "wb") as tmp:
                    tmp.write(cif_data)
                cif_parser = MMCIFParser(QUIET=True)
                structure = cif_parser.get_structure("af", tmp_path)
            finally:
                os.unlink(tmp_path)

            # Convert per-atom pLDDT to per-residue pLDDT
            # token_chain_ids and token_res_ids define the token (residue) order
            token_chain_ids = full_data["token_chain_ids"]
            token_res_ids = full_data["token_res_ids"]
            n_tokens = len(token_chain_ids)

            atom_plddts = full_data["atom_plddts"]
            atom_chain_ids = full_data.get("atom_chain_ids", [])

            # Build per-residue pLDDT by averaging atom pLDDT for each residue
            residue_plddt = _atom_to_residue_plddt(
                structure, atom_plddts, atom_chain_ids,
                token_chain_ids, token_res_ids,
            )

            pae = np.array(full_data["pae"])

            # Determine token chain order from the token_chain_ids
            seen = []
            for cid in token_chain_ids:
                if cid not in seen:
                    seen.append(cid)
            token_chain_order = seen

            models.append({
                "design": design_name,
                "rank": idx,
                "model_idx": idx,
                "seed": idx,
                "plddt": residue_plddt,
                "pae": pae,
                "iptm": float(summary["iptm"]),
                "ptm": float(summary["ptm"]),
                "structure": structure,
                "source": "webserver",
                "token_chain_order": token_chain_order,
            })

    return models


def _atom_to_residue_plddt(
    structure,
    atom_plddts: list,
    atom_chain_ids: list,
    token_chain_ids: list,
    token_res_ids: list,
) -> np.ndarray:
    """Convert per-atom pLDDT to per-residue pLDDT matching token order.

    Falls back to B-factor averaging from the structure if atom-level data
    doesn't align properly.
    """
    n_tokens = len(token_chain_ids)
    residue_plddt = np.zeros(n_tokens)

    # Build lookup: (chain_id, res_seq_idx) -> list of atom plddts
    # res_seq_idx is 1-based from token_res_ids
    if atom_chain_ids and len(atom_chain_ids) == len(atom_plddts):
        atom_by_res: Dict[Tuple[str, int], List[float]] = {}
        # Group consecutive atoms by their chain + residue
        # atom_chain_ids gives chain ID per atom; we need to figure out residue
        # assignment. Use the structure's residue membership.
        chain_atoms: Dict[str, List[float]] = {}
        for cid, plddt_val in zip(atom_chain_ids, atom_plddts):
            chain_atoms.setdefault(cid, []).append(plddt_val)

        for ti in range(n_tokens):
            cid = token_chain_ids[ti]
            chain = None
            for c in structure[0]:
                if c.id == cid:
                    chain = c
                    break
            if chain is None:
                continue
            residues = aa_residues(chain)
            # token_res_ids is 1-based residue index within the chain
            res_idx = token_res_ids[ti] - 1
            if res_idx < 0 or res_idx >= len(residues):
                continue
            res = residues[res_idx]
            # Use B-factor from structure atoms as pLDDT
            bfactors = [a.get_bfactor() for a in res.get_atoms() if a.element != "H"]
            if bfactors:
                residue_plddt[ti] = np.mean(bfactors)
    else:
        # Fallback: extract from structure B-factors in token order
        for ti in range(n_tokens):
            cid = token_chain_ids[ti]
            chain = None
            for c in structure[0]:
                if c.id == cid:
                    chain = c
                    break
            if chain is None:
                continue
            residues = aa_residues(chain)
            res_idx = token_res_ids[ti] - 1
            if res_idx < 0 or res_idx >= len(residues):
                continue
            res = residues[res_idx]
            bfactors = [a.get_bfactor() for a in res.get_atoms() if a.element != "H"]
            if bfactors:
                residue_plddt[ti] = np.mean(bfactors)

    return residue_plddt


# ---------------------------------------------------------------------------
# Interface detection
# ---------------------------------------------------------------------------

def get_interface_pairs(
    receptor_residues: list,
    peptide_residues: list,
    contact_cutoff: float = 5.0,
) -> List[Tuple[int, int, float]]:
    """Find interface residue pairs (peptide_idx, receptor_idx, min_dist).

    Indices are 0-based within each chain's residue list.
    """
    pairs = []
    for pi, pep_res in enumerate(peptide_residues):
        for ri, rec_res in enumerate(receptor_residues):
            d = heavy_atom_min_dist(pep_res, rec_res)
            if d <= contact_cutoff:
                pairs.append((pi, ri, d))
    return pairs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def get_cb_or_ca(residue):
    """Return CB coord for non-glycine, CA for glycine."""
    if "CB" in residue:
        return residue["CB"].coord
    elif "CA" in residue:
        return residue["CA"].coord
    return None


def compute_metrics(
    model_data: Dict,
    pocket_center: np.ndarray,
    depth_axis: np.ndarray,
    pocket_resnums: set,
    contact_cutoff: float = 5.0,
    cb_cutoff_ilis: float = 8.0,
    pae_cutoff_ilis: float = 12.0,
) -> Dict:
    """Compute all Stage 5 metrics for one AF model.

    model_data is the normalized dict from the format adapter containing
    structure, pae, plddt, iptm, and token_chain_order.

    The PAE/pLDDT arrays are indexed in token order. token_chain_order tells
    us which chain comes first. We identify receptor (longest) vs peptide
    (shortest) and map token indices accordingly.
    """
    structure = model_data["structure"]
    model = structure[0]

    chains = [c for c in model if len(aa_residues(c)) > 0]
    if len(chains) < 2:
        return {"error": "fewer than 2 protein chains"}

    chains_by_len = sorted(chains, key=lambda c: len(aa_residues(c)), reverse=True)
    receptor_chain = chains_by_len[0]
    peptide_chain = chains_by_len[1]

    rec_residues = aa_residues(receptor_chain)
    pep_residues = aa_residues(peptide_chain)
    n_rec = len(rec_residues)
    n_pep = len(pep_residues)
    n_total = n_rec + n_pep

    plddt = model_data["plddt"]
    pae = model_data["pae"]

    if pae.shape != (n_total, n_total):
        return {"error": f"PAE shape {pae.shape} != expected ({n_total},{n_total})"}

    ipTM = model_data["iptm"]

    # Determine token offsets for receptor and peptide.
    # token_chain_order tells us the order of chains in the PAE/pLDDT arrays.
    token_chain_order = model_data.get("token_chain_order", [])
    rec_token_offset, pep_token_offset = _get_token_offsets(
        token_chain_order, receptor_chain.id, peptide_chain.id,
        n_rec, n_pep, n_total,
    )

    interface_pairs = get_interface_pairs(rec_residues, pep_residues, contact_cutoff)

    if not interface_pairs:
        return {
            "ipTM": ipTM,
            "iPAE": float("nan"),
            "ipLDDT": float("nan"),
            "iLIS": 0.0,
            "pocket_depth": float("nan"),
            "n_interface_pairs": 0,
            "n_confident_pairs": 0,
        }

    # --- iPAE: median PAE at interface pairs ---
    pae_values = []
    for pi, ri, _ in interface_pairs:
        tok_pep = pep_token_offset + pi
        tok_rec = rec_token_offset + ri
        pae_ab = pae[tok_pep, tok_rec]
        pae_ba = pae[tok_rec, tok_pep]
        pae_values.append((pae_ab + pae_ba) / 2.0)
    iPAE = float(np.median(pae_values))

    # --- ipLDDT: median pLDDT of interface peptide residues ---
    interface_pep_indices = sorted(set(pi for pi, _, _ in interface_pairs))
    pep_plddt_values = [float(plddt[pep_token_offset + pi]) for pi in interface_pep_indices]
    ipLDDT = float(np.median(pep_plddt_values))

    # --- iLIS: local interaction score ---
    n_confident = 0
    for pi, ri, _ in interface_pairs:
        cb_pep = get_cb_or_ca(pep_residues[pi])
        cb_rec = get_cb_or_ca(rec_residues[ri])
        if cb_pep is None or cb_rec is None:
            continue
        cb_dist = float(np.linalg.norm(cb_pep - cb_rec))
        tok_pep = pep_token_offset + pi
        tok_rec = rec_token_offset + ri
        pae_sym = (pae[tok_pep, tok_rec] + pae[tok_rec, tok_pep]) / 2.0
        if cb_dist <= cb_cutoff_ilis and pae_sym <= pae_cutoff_ilis:
            n_confident += 1
    iLIS = n_confident / len(interface_pairs)

    # --- pocket_depth: max projection of peptide atoms onto depth axis ---
    af_pocket_ca = []
    for res in rec_residues:
        if res.id[1] in pocket_resnums and "CA" in res:
            af_pocket_ca.append(res["CA"].coord)

    if af_pocket_ca:
        af_pocket_center = np.mean(af_pocket_ca, axis=0)
    else:
        af_pocket_center = pocket_center

    all_rec_ca = [res["CA"].coord for res in rec_residues if "CA" in res]
    if all_rec_ca:
        af_rec_centroid = np.mean(all_rec_ca, axis=0)
    else:
        af_rec_centroid = np.zeros(3)

    af_depth_vec = af_pocket_center - af_rec_centroid
    af_depth_norm = float(np.linalg.norm(af_depth_vec))
    if af_depth_norm > 1e-6:
        af_depth_axis = af_depth_vec / af_depth_norm
    else:
        af_depth_axis = depth_axis

    pep_projections = []
    for res in pep_residues:
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            vec = atom.coord - af_rec_centroid
            proj = float(np.dot(vec, af_depth_axis))
            pep_projections.append(proj)

    pocket_proj = float(np.dot(af_pocket_center - af_rec_centroid, af_depth_axis))
    if pep_projections:
        pocket_depth = max(pep_projections) - pocket_proj
    else:
        pocket_depth = float("nan")

    return {
        "ipTM": round(ipTM, 4),
        "iPAE": round(iPAE, 2),
        "ipLDDT": round(ipLDDT, 2),
        "iLIS": round(iLIS, 4),
        "pocket_depth": round(pocket_depth, 2),
        "n_interface_pairs": len(interface_pairs),
        "n_confident_pairs": n_confident,
    }


def _get_token_offsets(
    token_chain_order: list,
    receptor_chain_id: str,
    peptide_chain_id: str,
    n_rec: int,
    n_pep: int,
    n_total: int,
) -> Tuple[int, int]:
    """Compute the starting token index for receptor and peptide chains.

    In ColabFold, the receptor (longest) is always first.
    In AF web server, chains follow the input order (peptide could be first).
    """
    if not token_chain_order or len(token_chain_order) < 2:
        # Default: receptor first
        return 0, n_rec

    # Build cumulative sizes for each chain in token order
    chain_sizes = {}
    chain_sizes[receptor_chain_id] = n_rec
    chain_sizes[peptide_chain_id] = n_pep

    offset = 0
    rec_offset = None
    pep_offset = None
    for cid in token_chain_order:
        if cid == receptor_chain_id:
            rec_offset = offset
            offset += n_rec
        elif cid == peptide_chain_id:
            pep_offset = offset
            offset += n_pep

    if rec_offset is None:
        rec_offset = 0
    if pep_offset is None:
        pep_offset = n_rec

    return rec_offset, pep_offset


# ---------------------------------------------------------------------------
# DeepTMHMM topology check
# ---------------------------------------------------------------------------

def predict_topology(receptor_sequence: str, cache_path: Optional[str] = None) -> Dict[int, str]:
    """Run DeepTMHMM on the receptor sequence. Returns {0-based_pos: label}.

    Labels: 'inside', 'outside', 'TMhelix', 'signal'.
    If DeepTMHMM is not available, returns empty dict (all positions unknown).
    """
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            return {int(k): v for k, v in cached.items()}

    try:
        import biolib  # noqa: F811
        tmhmm = biolib.load("DTU/DeepTMHMM")
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
            tmp.write(f">receptor\n{receptor_sequence}\n")
            tmp_path = tmp.name
        try:
            result = tmhmm.cli(args=f"--fasta {tmp_path}")
            result.save_files("deeptmhmm_out/")
            gff_path = Path("deeptmhmm_out/TMRs.gff3")
            if gff_path.exists():
                topo = _parse_deeptmhmm_gff(gff_path, len(receptor_sequence))
            else:
                pred_path = Path("deeptmhmm_out/predicted_topologies.3line")
                topo = _parse_deeptmhmm_3line(pred_path, len(receptor_sequence))
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        log(f"  WARNING: DeepTMHMM failed ({e}), skipping topology check")
        return {}

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in topo.items()}, f)

    return topo


def _parse_deeptmhmm_3line(pred_path: Path, seq_len: int) -> Dict[int, str]:
    """Parse the 3-line topology prediction format from DeepTMHMM."""
    topo = {}
    if not pred_path.exists():
        return topo
    lines = pred_path.read_text().strip().split("\n")
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
        topology_line = lines[i + 2].strip()
        label_map = {"i": "inside", "o": "outside", "M": "TMhelix", "S": "signal"}
        for pos, char in enumerate(topology_line):
            topo[pos] = label_map.get(char, char)
        break
    return topo


def _parse_deeptmhmm_gff(gff_path: Path, seq_len: int) -> Dict[int, str]:
    """Parse GFF3 format topology prediction from DeepTMHMM."""
    topo = {i: "unknown" for i in range(seq_len)}
    for line in gff_path.read_text().strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        feat_type = parts[2].lower()
        start = int(parts[3]) - 1
        end = int(parts[4])
        label_map = {
            "inside": "inside",
            "outside": "outside",
            "tmhelix": "TMhelix",
            "signal": "signal",
        }
        label = label_map.get(feat_type, feat_type)
        for pos in range(start, min(end, seq_len)):
            topo[pos] = label
    return topo


def check_topology(
    receptor_residues: list,
    peptide_residues: list,
    topology: Dict[int, str],
    contact_cutoff: float = 5.0,
) -> Tuple[bool, int, int]:
    """Check whether the peptide contacts extracellular receptor residues.

    Returns (topo_pass, n_extracellular_contacts, n_total_contacts).
    """
    if not topology:
        return True, -1, -1

    contacting_rec_indices = set()
    for pep_res in peptide_residues:
        for ri, rec_res in enumerate(receptor_residues):
            d = heavy_atom_min_dist(pep_res, rec_res)
            if d <= contact_cutoff:
                contacting_rec_indices.add(ri)

    n_extra = 0
    for ri in contacting_rec_indices:
        label = topology.get(ri, "unknown")
        if label == "outside":
            n_extra += 1

    topo_pass = n_extra > 0
    return topo_pass, n_extra, len(contacting_rec_indices)


# ---------------------------------------------------------------------------
# Aggregation across model seeds
# ---------------------------------------------------------------------------

def aggregate_seeds(per_model_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across model seeds using pessimistic median - MAD.

    For 'higher is better' metrics: x* = median - MAD
    For 'lower is better' metrics:  x* = median + MAD
    """
    higher_is_better = ["ipTM", "ipLDDT", "iLIS", "pocket_depth"]
    lower_is_better = ["iPAE"]

    rows = []
    for design_name, group in per_model_df.groupby("design"):
        row = {"design": design_name, "n_seeds": len(group)}

        for metric in higher_is_better + lower_is_better:
            vals = group[metric].dropna().values
            if len(vals) == 0:
                row[metric] = float("nan")
                row[f"{metric}_median"] = float("nan")
                row[f"{metric}_mad"] = float("nan")
                continue

            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            row[f"{metric}_median"] = round(med, 4)
            row[f"{metric}_mad"] = round(mad, 4)

            if metric in higher_is_better:
                row[metric] = round(med - mad, 4)
            else:
                row[metric] = round(med + mad, 4)

        for col in ["topo_pass", "n_interface_pairs"]:
            if col in group.columns:
                if col == "topo_pass":
                    row[col] = bool(group[col].all())
                else:
                    row[col] = int(group[col].median())

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold application
# ---------------------------------------------------------------------------

def apply_thresholds(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Apply Stage 5 thresholds. Adds pass_* columns and overall pass_stage5."""
    checks = [
        ("min_ipTM", "ipTM", "min"),
        ("max_iPAE", "iPAE", "max"),
        ("min_ipLDDT", "ipLDDT", "min"),
        ("min_iLIS", "iLIS", "min"),
        ("min_pocket_depth", "pocket_depth", "min"),
    ]

    for _, row in df.iterrows():
        pass_all = True
        for thresh_name, metric_name, direction in checks:
            tv, mode = get_threshold(config, "stage5", thresh_name)
            value = row[metric_name]
            if pd.isna(value):
                passes = False
            else:
                passes, _ = check_threshold(value, tv, direction, mode)
            df.loc[row.name, f"pass_{thresh_name}"] = passes
            if not passes and mode == "filter":
                pass_all = False

        # Topology
        tv_topo, mode_topo = get_threshold(config, "stage5", "require_topo_pass")
        if mode_topo != "ignore" and tv_topo and "topo_pass" in df.columns:
            topo_ok = bool(row.get("topo_pass", True))
            df.loc[row.name, "pass_require_topo_pass"] = topo_ok
            if not topo_ok and mode_topo == "filter":
                pass_all = False
        else:
            df.loc[row.name, "pass_require_topo_pass"] = True

        df.loc[row.name, "pass_stage5"] = pass_all

    bool_cols = [c for c in df.columns if c.startswith("pass_")]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage5(
    outdir: str,
    config: Dict,
    af_dir_override: Optional[str] = None,
    regions_json_override: Optional[str] = None,
) -> pd.DataFrame:
    """Run Stage 5: AF confidence scoring."""
    outpath = Path(outdir) / "stage5"
    outpath.mkdir(parents=True, exist_ok=True)

    s5_cfg = config.get("stage5", {})
    contact_cutoff = s5_cfg.get("contact_cutoff", 5.0)
    cb_cutoff_ilis = s5_cfg.get("cb_cutoff_ilis", 8.0)
    pae_cutoff_ilis = s5_cfg.get("pae_cutoff_ilis", 12.0)
    use_deeptmhmm = s5_cfg.get("use_deeptmhmm", False)

    # --- Load regions ---
    if regions_json_override:
        regions_path = Path(regions_json_override)
    else:
        regions_path = Path(outdir) / "stage1" / "regions.json"

    if not regions_path.exists():
        raise FileNotFoundError(f"regions.json not found at {regions_path}")

    with open(regions_path) as f:
        regions = json.load(f)

    pocket_center = np.array(regions["pocket_center"])
    depth_axis = np.array(regions["depth_axis"])
    pocket_resnums = {r["resnum"] for r in regions["pocket_residues"]}

    log("=" * 60)
    log("STAGE 5: AlphaFold confidence scoring")
    log("=" * 60)
    log(f"  Contact cutoff: {contact_cutoff} A")
    log(f"  iLIS CB cutoff: {cb_cutoff_ilis} A, PAE cutoff: {pae_cutoff_ilis} A")
    log(f"  Pocket residues: {len(pocket_resnums)}")
    log(f"  DeepTMHMM: {'enabled' if use_deeptmhmm else 'disabled'}")

    # --- Find AF output directories ---
    if af_dir_override:
        af_base = Path(af_dir_override)
    else:
        af_base = Path(outdir) / "stage4" / "af_outputs"

    if not af_base.exists():
        raise FileNotFoundError(f"AF outputs not found at {af_base}")

    # Collect all normalized model data grouped by design name.
    # Supports: ColabFold dirs, AF web server ZIPs, or a mix.
    design_models: Dict[str, List[Dict]] = {}

    # 1) Check for web server ZIPs
    zip_files = find_webserver_zips(str(af_base))
    if zip_files:
        log(f"  Found {len(zip_files)} AF web server ZIP(s)")
        for zf in zip_files:
            zip_models = load_webserver_zip(zf)
            design_name = Path(zf).stem
            for md in zip_models:
                md["design"] = design_name
                design_models.setdefault(design_name, []).append(md)

    # 2) Check for ColabFold directories
    cf_models_base = find_colabfold_models(str(af_base))
    if cf_models_base:
        design_name = af_base.name
        for mi in cf_models_base:
            md = load_colabfold_model(mi)
            md["design"] = design_name
            design_models.setdefault(design_name, []).append(md)
    else:
        candidate_dirs = sorted([d for d in af_base.iterdir() if d.is_dir()])
        for cdir in candidate_dirs:
            cf_models = find_colabfold_models(str(cdir))
            if cf_models:
                design_name = cdir.name
                for mi in cf_models:
                    md = load_colabfold_model(mi)
                    md["design"] = design_name
                    design_models.setdefault(design_name, []).append(md)

    if not design_models:
        raise FileNotFoundError(
            f"No AF outputs found in {af_base}. Expected ColabFold directories "
            f"(*_scores_rank_*.json + *_unrelaxed_*.pdb) or AF web server ZIPs "
            f"(*_model_*.cif + *_full_data_*.json)."
        )

    n_total_models = sum(len(v) for v in design_models.values())
    log(f"  Found {len(design_models)} design(s), {n_total_models} total model(s)")

    # --- DeepTMHMM topology (once per receptor) ---
    topology = {}
    if use_deeptmhmm:
        topo_cache = outpath / "topology_cache.json"
        first_model = next(iter(design_models.values()))[0]
        receptor_seq = _extract_receptor_sequence_from_model(first_model)
        if receptor_seq:
            log(f"\n[Stage 5] Running DeepTMHMM on receptor ({len(receptor_seq)} aa)...")
            topology = predict_topology(receptor_seq, cache_path=str(topo_cache))
            if topology:
                n_outside = sum(1 for v in topology.values() if v == "outside")
                n_tm = sum(1 for v in topology.values() if v == "TMhelix")
                log(f"  Topology: {n_outside} extracellular, {n_tm} TM residues")
            else:
                log("  WARNING: topology prediction returned empty, skipping")

    # --- Score each design x model ---
    all_per_model = []
    sorted_designs = sorted(design_models.keys())
    for di, design_name in enumerate(sorted_designs):
        models = design_models[design_name]
        log(f"\n  [{di+1}/{len(sorted_designs)}] {design_name}: "
            f"{len(models)} model(s) [{models[0]['source']}]")

        for md in models:
            metrics = compute_metrics(
                model_data=md,
                pocket_center=pocket_center,
                depth_axis=depth_axis,
                pocket_resnums=pocket_resnums,
                contact_cutoff=contact_cutoff,
                cb_cutoff_ilis=cb_cutoff_ilis,
                pae_cutoff_ilis=pae_cutoff_ilis,
            )

            if "error" in metrics:
                log(f"    model {md['model_idx']}: ERROR - {metrics['error']}")
                continue

            # Topology check
            if topology:
                af_chains = sorted(
                    [c for c in md["structure"][0] if len(aa_residues(c)) > 0],
                    key=lambda c: len(aa_residues(c)), reverse=True,
                )
                rec_res = aa_residues(af_chains[0])
                pep_res = aa_residues(af_chains[1])
                topo_pass, n_extra, n_contacts = check_topology(
                    rec_res, pep_res, topology, contact_cutoff
                )
                metrics["topo_pass"] = topo_pass
                metrics["n_extracellular_contacts"] = n_extra
                metrics["n_topo_contacts"] = n_contacts
            else:
                metrics["topo_pass"] = True

            metrics["design"] = design_name
            metrics["rank"] = md["rank"]
            metrics["model"] = md["model_idx"]
            metrics["seed"] = md["seed"]
            metrics["source"] = md["source"]

            log(f"    model {md['model_idx']}: ipTM={metrics['ipTM']:.3f}  iPAE={metrics['iPAE']:.1f}  "
                f"ipLDDT={metrics['ipLDDT']:.1f}  iLIS={metrics['iLIS']:.3f}  "
                f"depth={metrics['pocket_depth']:.1f}  pairs={metrics['n_interface_pairs']}")

            all_per_model.append(metrics)

    if not all_per_model:
        log("\nERROR: No models scored successfully")
        return pd.DataFrame()

    per_model_df = pd.DataFrame(all_per_model)
    scores_csv = outpath / "stage5_scores.csv"
    per_model_df.to_csv(scores_csv, index=False)
    log(f"\n[Stage 5] Per-model scores -> {scores_csv}")

    # --- Aggregate across seeds ---
    summary_df = aggregate_seeds(per_model_df)
    summary_df = apply_thresholds(summary_df, config)

    aggregated_csv = outpath / "stage5_aggregated.csv"
    summary_df.to_csv(aggregated_csv, index=False)

    n_pass = summary_df["pass_stage5"].sum() if "pass_stage5" in summary_df.columns else 0
    log(f"\n{'=' * 60}")
    log(f"STAGE 5 DONE: {n_pass}/{len(summary_df)} designs passed")
    log(f"  Per-model CSV  -> {scores_csv}")
    log(f"  Aggregated CSV -> {aggregated_csv}")
    log("=" * 60)

    return summary_df


def _extract_receptor_sequence_from_model(model_data: Dict) -> Optional[str]:
    """Extract the receptor sequence from a normalized model data dict."""
    from Bio.SeqUtils import seq1
    structure = model_data["structure"]
    chains = sorted(
        [c for c in structure[0] if len(aa_residues(c)) > 0],
        key=lambda c: len(aa_residues(c)), reverse=True,
    )
    if not chains:
        return None
    rec_residues = aa_residues(chains[0])
    return "".join(seq1(r.get_resname()) for r in rec_residues)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 5: AlphaFold confidence scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("outdir", help="Top-level pipeline output directory")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument(
        "--af-dir", default=None,
        help="Override AF output directory (default: <outdir>/stage4/af_outputs)",
    )
    ap.add_argument(
        "--regions-json", default=None,
        help="Override path to regions.json (default: <outdir>/stage1/regions.json)",
    )

    args = ap.parse_args()
    config = load_config(args.config) if args.config else {}

    run_stage5(
        outdir=args.outdir,
        config=config,
        af_dir_override=args.af_dir,
        regions_json_override=args.regions_json,
    )


if __name__ == "__main__":
    main()
