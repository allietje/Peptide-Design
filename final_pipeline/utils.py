"""Shared helpers for the GPCR peptide design pipeline."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from Bio.PDB import PDBIO, NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        from Bio import pairwise2
    except ImportError:
        pairwise2 = None

try:
    from Bio.Align import PairwiseAligner
except ImportError:
    PairwiseAligner = None


# ---------------------------------------------------------------------------
# PDB helpers
# ---------------------------------------------------------------------------

def aa_residues(chain):
    """Standard amino-acid residues from a chain."""
    return [r for r in chain if is_aa(r, standard=True)]


def heavy_atoms(chain):
    """Yield non-hydrogen atoms from standard amino-acid residues."""
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        for atom in res.get_atoms():
            if atom.element != "H":
                yield atom


def get_receptor_peptide(structure, receptor_id=None, peptide_id=None):
    """Return (receptor_chain, peptide_chain) using explicit IDs or length heuristic."""
    receptor, peptide = None, None
    for chain in structure[0]:
        residues = [r for r in chain if is_aa(r, standard=True)]
        if not residues:
            continue
        if receptor_id and chain.id == receptor_id:
            receptor = chain
            continue
        if peptide_id and chain.id == peptide_id:
            peptide = chain
            continue
        if receptor is None and receptor_id is None and len(residues) > 100:
            receptor = chain
        elif peptide is None and peptide_id is None and len(residues) <= 100:
            peptide = chain
    return receptor, peptide


def get_chains_by_length(structure):
    """Return (receptor_chain, peptide_chain) by picking longest and shortest."""
    chains = []
    for chain in structure[0]:
        residues = [r for r in chain if is_aa(r, standard=True)]
        if residues:
            chains.append((chain, len(residues)))
    if len(chains) < 2:
        raise ValueError(f"Expected 2 protein chains, found {len(chains)}")
    chains.sort(key=lambda x: x[1], reverse=True)
    return chains[0][0], chains[1][0]


def get_chain_by_id(structure, chain_id):
    for chain in structure[0]:
        if chain.id == chain_id:
            return chain
    return None


def get_sequence(chain):
    residues = [r for r in chain if is_aa(r, standard=True)]
    sequence = "".join(seq1(r.get_resname()) for r in residues)
    return residues, sequence


def heavy_atom_min_dist(res1, res2) -> float:
    """Minimum heavy-atom distance between two residues."""
    best = float("inf")
    for a1 in res1.get_atoms():
        if a1.element == "H":
            continue
        for a2 in res2.get_atoms():
            if a2.element == "H":
                continue
            d = np.linalg.norm(a1.coord - a2.coord)
            if d < best:
                best = d
    return best


# ---------------------------------------------------------------------------
# Sequence alignment
# ---------------------------------------------------------------------------

def align_sequences(ref_seq: str, des_seq: str) -> Dict[int, int]:
    """Return mapping: design_index -> reference_index (0-based)."""
    if ref_seq == des_seq:
        return {i: i for i in range(len(ref_seq))}

    if PairwiseAligner is not None:
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1
        aligner.mismatch_score = 0
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5
        alignment = aligner.align(ref_seq, des_seq)[0]
        ref_al, des_al = alignment[0], alignment[1]
    elif pairwise2 is not None:
        alignments = pairwise2.align.globalxx(ref_seq, des_seq)
        if not alignments:
            raise ValueError("Sequence alignment failed")
        ref_al = alignments[0].seqA
        des_al = alignments[0].seqB
    else:
        raise ImportError("No alignment module available (need Bio.Align or Bio.pairwise2)")

    mapping: Dict[int, int] = {}
    ri, di = 0, 0
    for rc, dc in zip(ref_al, des_al):
        if rc != "-":
            ri += 1
        if dc != "-":
            di += 1
            if rc != "-":
                mapping[di - 1] = ri - 1
    return mapping


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

def find_clashes(chain_a, chain_b, cutoff: float) -> Tuple[int, List[Dict]]:
    """Find inter-chain heavy-atom pairs closer than *cutoff*."""
    atoms_b = [a for r in aa_residues(chain_b) for a in r.get_atoms() if a.element != "H"]
    if not atoms_b:
        return 0, []
    atom_to_res_b = {id(a): a.get_parent() for a in atoms_b}
    ns = NeighborSearch(atoms_b)
    count = 0
    seen_res_pairs: Set[Tuple[int, int]] = set()
    details: List[Dict] = []
    for res_a in aa_residues(chain_a):
        for atom_a in res_a.get_atoms():
            if atom_a.element == "H":
                continue
            for hit in ns.search(atom_a.coord, cutoff, level="A"):
                count += 1
                res_b = atom_to_res_b[id(hit)]
                pair_key = (res_a.id[1], res_b.id[1])
                if pair_key not in seen_res_pairs:
                    seen_res_pairs.add(pair_key)
                    d = float(np.linalg.norm(atom_a.coord - hit.coord))
                    details.append({
                        "chain_a": res_a.parent.id,
                        "resnum_a": res_a.id[1],
                        "resname_a": res_a.get_resname(),
                        "atom_a": atom_a.get_name().strip(),
                        "chain_b": res_b.parent.id,
                        "resnum_b": res_b.id[1],
                        "resname_b": res_b.get_resname(),
                        "atom_b": hit.get_name().strip(),
                        "distance": round(d, 2),
                    })
    details.sort(key=lambda x: x["distance"])
    return count, details


# ---------------------------------------------------------------------------
# Config / threshold helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load YAML config, return dict. Missing file returns empty dict."""
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def get_threshold(config: Dict, stage: str, name: str) -> Tuple[Any, str]:
    """Return (value, mode) for a threshold from config.

    mode is one of 'filter', 'flag', 'ignore'.
    Returns (None, 'ignore') if not found.
    """
    stage_cfg = config.get(stage, {})
    thresholds = stage_cfg.get("thresholds", {})
    entry = thresholds.get(name, {})
    if not entry:
        return None, "ignore"
    return entry.get("value"), entry.get("mode", "flag")


def check_threshold(value, threshold_value, direction: str, mode: str) -> Tuple[bool, bool]:
    """Check a value against a threshold.

    direction: 'min' (value >= threshold) or 'max' (value <= threshold)
    mode: 'filter', 'flag', or 'ignore'

    Returns (passes, is_filtered_out).
    """
    if mode == "ignore" or threshold_value is None:
        return True, False
    if direction == "min":
        passes = value >= threshold_value
    else:
        passes = value <= threshold_value
    filtered_out = (not passes) and (mode == "filter")
    return passes, filtered_out


def log(msg: str, **kwargs) -> None:
    """Print to stderr."""
    print(msg, file=sys.stderr, **kwargs)
