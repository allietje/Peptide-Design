#!/usr/bin/env python3
"""
Analyze inter-chain interactions in a PDB structure.

Reads a PDB file, detects interactions (ionic, H-bonds, hydrophobic, generic contact)
between selected chains, and outputs a table with residues, locations, types, and distances.

Distance measurements:
- Ionic/salt bridges: Distance between charge-bearing atoms (e.g., OD1/OD2 of ASP/GLU 
  to NZ of LYS or NH1/NH2 of ARG)
- H-bonds: Distance from H atom to acceptor atom (O or N). If H atoms are not present 
  in the PDB (common in X-ray structures), falls back to heavy-atom distance (donor N/O 
  to acceptor O/N)
- Hydrophobic/Contact: Minimum distance between any heavy atoms of the two residues

Usage:
  python pdb_interactions.py structure.pdb
  python pdb_interactions.py structure.pdb --chains A B --interactions ionic hbond
  python pdb_interactions.py structure.pdb --distance-ionic 4.5 --distance-hbond 3.2
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser, Residue, Structure
from Bio.PDB.Polypeptide import is_aa
import numpy as np


# ---------------------------------------------------------------------------
# Default distance cutoffs (Angstroms) and residue/atom definitions
# ---------------------------------------------------------------------------

DEFAULT_DISTANCES = {
    "ionic": 4.0,
    "hbond": 3.5,
    "hydrophobic": 5.0,
    "contact": 5.0,
}

# Residues and atoms used for ionic interactions (charge-bearing atoms)
NEGATIVE_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}
POSITIVE_ATOMS = {
    "LYS": ["NZ"],
    "ARG": ["NE", "NH1", "NH2"],
    "HIS": ["ND1", "NE2"],
}

# H-bond: donors (N, O with H) and acceptors (O, N) – we use heavy-atom distance
# Donor heavy atoms (backbone N + common side-chain donors)
HBOND_DONOR_ATOMS = {
    "default": ["N"],
    "ARG": ["N", "NE", "NH1", "NH2"],
    "ASN": ["N", "ND2"],
    "GLN": ["N", "NE2"],
    "LYS": ["N", "NZ"],
    "SER": ["N", "OG"],
    "THR": ["N", "OG1"],
    "TYR": ["N", "OH"],
    "TRP": ["N", "NE1"],
    "HIS": ["N", "ND1", "NE2"],
    "CYS": ["N", "SG"],
}
HBOND_ACCEPTOR_ATOMS = {
    "default": ["O"],
    "ASP": ["O", "OD1", "OD2"],
    "GLU": ["O", "OE1", "OE2"],
    "ASN": ["O", "OD1"],
    "GLN": ["O", "OE1"],
    "SER": ["O", "OG"],
    "THR": ["O", "OG1"],
    "TYR": ["O", "OH"],
    "HIS": ["O", "ND1", "NE2"],
    "CYS": ["O", "SG"],
}

HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"}


def get_chain_ids(structure: Structure) -> List[str]:
    """Return sorted list of chain IDs in the structure."""
    return sorted({c.id for m in structure for c in m})


def get_atoms_for_ionic(res: Residue) -> List[Tuple[str, np.ndarray]]:
    """Return (atom_name, coord) for charge-bearing atoms, or empty if not charged."""
    out = []
    resname = res.get_resname()
    if resname in NEGATIVE_ATOMS:
        for an in NEGATIVE_ATOMS[resname]:
            if an in res:
                out.append((an, res[an].coord))
    if resname in POSITIVE_ATOMS:
        for an in POSITIVE_ATOMS[resname]:
            if an in res:
                out.append((an, res[an].coord))
    return out


def is_ionic_pair(res1: Residue, res2: Residue) -> bool:
    """True if one residue is negative and the other positive."""
    n1, n2 = res1.get_resname(), res2.get_resname()
    return (n1 in NEGATIVE_ATOMS and n2 in POSITIVE_ATOMS) or (
        n1 in POSITIVE_ATOMS and n2 in NEGATIVE_ATOMS
    )


def get_donor_atoms(res: Residue) -> List[Tuple[str, np.ndarray]]:
    """Return (atom_name, coord) for H-bond donor heavy atoms."""
    resname = res.get_resname()
    names = HBOND_DONOR_ATOMS.get(resname, HBOND_DONOR_ATOMS["default"])
    return [(a, res[a].coord) for a in names if a in res]


def get_acceptor_atoms(res: Residue) -> List[Tuple[str, np.ndarray]]:
    """Return (atom_name, coord) for H-bond acceptor heavy atoms."""
    resname = res.get_resname()
    names = HBOND_ACCEPTOR_ATOMS.get(resname, HBOND_ACCEPTOR_ATOMS["default"])
    return [(a, res[a].coord) for a in names if a in res]


def get_hydrogen_atoms_for_donor(res: Residue, donor_atom_name: str) -> List[Tuple[str, np.ndarray]]:
    """
    Find H atoms attached to a donor atom (N or O).
    Returns list of (H_atom_name, coord) pairs.
    """
    if donor_atom_name not in res:
        return []
    
    donor_atom = res[donor_atom_name]
    donor_coord = donor_atom.coord
    
    # Find H atoms bonded to this donor atom
    # H atoms are typically within ~1 Å of N or O
    h_atoms = []
    for atom in res.get_atoms():
        if atom.element == "H":
            dist = np.linalg.norm(atom.coord - donor_coord)
            if dist < 1.2:  # Typical N-H or O-H bond length is ~1.0 Å
                h_atoms.append((atom.get_name().strip(), atom.coord))
    
    return h_atoms


def is_hbond_pair(res1: Residue, res2: Residue) -> bool:
    """True if pair can form H-bond (one has donor, other has acceptor)."""
    d1, a1 = len(get_donor_atoms(res1)), len(get_acceptor_atoms(res1))
    d2, a2 = len(get_donor_atoms(res2)), len(get_acceptor_atoms(res2))
    return (d1 and a2) or (d2 and a1)


def min_distance_atom_sets(
    atoms1: List[Tuple[str, np.ndarray]], atoms2: List[Tuple[str, np.ndarray]]
) -> Tuple[float, str, str]:
    """Min distance between two sets of (name, coord). Returns (dist, name1, name2)."""
    best = float("inf")
    an1, an2 = "", ""
    for a1, c1 in atoms1:
        for a2, c2 in atoms2:
            d = np.linalg.norm(c1 - c2)
            if d < best:
                best, an1, an2 = d, a1, a2
    return (best, an1, an2)


def residue_min_heavy_atom_distance(res1: Residue, res2: Residue) -> Tuple[float, str, str]:
    """Min distance between any heavy atoms of two residues. Returns (dist, atom1, atom2)."""
    best = float("inf")
    an1, an2 = "", ""
    for a1 in res1.get_atoms():
        if a1.element == "H":
            continue
        c1 = a1.coord
        for a2 in res2.get_atoms():
            if a2.element == "H":
                continue
            d = np.linalg.norm(c1 - a2.coord)
            if d < best:
                best, an1, an2 = d, a1.get_fullname().strip(), a2.get_fullname().strip()
    return (best, an1, an2)


def residue_id(res: Residue) -> str:
    """Short string id for residue (e.g. A:GLU:42)."""
    het, num, ins = res.id
    return f"{res.parent.id}:{res.get_resname()}:{num}{ins if ins != ' ' else ''}"


def run_analysis(
    pdb_path: str,
    chains: Optional[List[str]] = None,
    interaction_types: Optional[List[str]] = None,
    distances: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Run interaction analysis on a PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.
    chains : list of str, optional
        Chain IDs to consider (e.g. ["A", "B"]). If None, use all chains.
    interaction_types : list of str, optional
        One or more of "ionic", "hbond", "hydrophobic", "contact". If None, use all.
    distances : dict, optional
        Override distance cutoffs in Angstroms, e.g. {"ionic": 4.5, "hbond": 3.2}.

    Returns
    -------
    pandas.DataFrame
        Table with columns: chain1, resnum1, resname1, chain2, resnum2, resname2,
        interaction_type, distance_angstrom, atom1, atom2.
    """
    if distances is None:
        distances = {}
    if interaction_types is None:
        interaction_types = list(DEFAULT_DISTANCES)
    cutoffs = {k: distances.get(k, DEFAULT_DISTANCES[k]) for k in interaction_types}
    max_radius = max(cutoffs.values()) + 0.5  # small buffer

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    all_chains = get_chain_ids(structure)
    if chains is not None:
        chain_set = set(chains)
        if not chain_set.issubset(set(all_chains)):
            raise ValueError(f"Requested chains {chains} not all in structure chains {all_chains}")
        use_chains = [c for c in all_chains if c in chain_set]
    else:
        use_chains = all_chains

    # Collect residues per chain (standard amino acids only)
    residues_by_chain: Dict[str, List[Residue]] = {}
    all_atoms = []
    for model in structure:
        for chain in model:
            if chain.id not in use_chains:
                continue
            res_list = [r for r in chain if is_aa(r, standard=True)]
            residues_by_chain[chain.id] = res_list
            for r in res_list:
                for a in r.get_atoms():
                    all_atoms.append(a)  # Include H atoms for H-bond detection

    if len(use_chains) < 2:
        return pd.DataFrame(
            columns=[
                "chain1", "resnum1", "resname1", "chain2", "resnum2", "resname2",
                "interaction_type", "distance_angstrom", "atom1", "atom2",
            ]
        )

    ns = NeighborSearch(all_atoms)
    # Find all residue pairs (from different chains) that have any atoms within max_radius
    # Include H atoms in search for H-bond detection
    seen_pairs: Set[Tuple[Residue, Residue]] = set()
    for model in structure:
        for chain in model:
            if chain.id not in use_chains:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                for atom in res.get_atoms():
                    center = atom.coord
                    for neighbor in ns.search(center, max_radius, level="A"):
                        other = neighbor.parent
                        if not is_aa(other, standard=True):
                            continue
                        if other.parent.id == res.parent.id:
                            continue  # same chain
                        if other.parent.id not in use_chains:
                            continue
                        pair = (res, other) if id(res) < id(other) else (other, res)
                        seen_pairs.add(pair)

    rows = []
    for res1, res2 in seen_pairs:
        dist_any, atom_any1, atom_any2 = residue_min_heavy_atom_distance(res1, res2)
        ch1, ch2 = res1.parent.id, res2.parent.id
        (het1, num1, ins1), (het2, num2, ins2) = res1.id, res2.id
        rn1, rn2 = res1.get_resname(), res2.get_resname()

        def add_row(itype: str, dist: float, a1: str, a2: str) -> None:
            rows.append({
                "chain1": ch1,
                "resnum1": f"{num1}{ins1 if ins1 != ' ' else ''}",
                "resname1": rn1,
                "chain2": ch2,
                "resnum2": f"{num2}{ins2 if ins2 != ' ' else ''}",
                "resname2": rn2,
                "interaction_type": itype,
                "distance_angstrom": round(dist, 2),
                "atom1": a1,
                "atom2": a2,
            })

        if "contact" in cutoffs and dist_any <= cutoffs["contact"]:
            add_row("contact", dist_any, atom_any1, atom_any2)

        if "ionic" in cutoffs and is_ionic_pair(res1, res2):
            ions1 = get_atoms_for_ionic(res1)
            ions2 = get_atoms_for_ionic(res2)
            if ions1 and ions2:
                d_ion, a1, a2 = min_distance_atom_sets(ions1, ions2)
                if d_ion <= cutoffs["ionic"]:
                    add_row("ionic", d_ion, a1, a2)

        if "hbond" in cutoffs and is_hbond_pair(res1, res2):
            don1, acc1 = get_donor_atoms(res1), get_acceptor_atoms(res1)
            don2, acc2 = get_donor_atoms(res2), get_acceptor_atoms(res2)
            best_h = float("inf")
            ba1, ba2 = "", ""
            
            # Try res1 as donor, res2 as acceptor
            if don1 and acc2:
                for donor_name, donor_coord in don1:
                    # First try to find H atoms attached to this donor
                    h_atoms = get_hydrogen_atoms_for_donor(res1, donor_name)
                    if h_atoms:
                        # Measure H-to-acceptor distance
                        for h_name, h_coord in h_atoms:
                            for acc_name, acc_coord in acc2:
                                d = np.linalg.norm(h_coord - acc_coord)
                                if d < best_h:
                                    best_h, ba1, ba2 = d, h_name, acc_name
                    else:
                        # No H found, use heavy-atom distance as fallback
                        for acc_name, acc_coord in acc2:
                            d = np.linalg.norm(donor_coord - acc_coord)
                            if d < best_h:
                                best_h, ba1, ba2 = d, donor_name, acc_name
            
            # Try res2 as donor, res1 as acceptor
            if don2 and acc1:
                for donor_name, donor_coord in don2:
                    # First try to find H atoms attached to this donor
                    h_atoms = get_hydrogen_atoms_for_donor(res2, donor_name)
                    if h_atoms:
                        # Measure H-to-acceptor distance
                        for h_name, h_coord in h_atoms:
                            for acc_name, acc_coord in acc1:
                                d = np.linalg.norm(h_coord - acc_coord)
                                if d < best_h:
                                    best_h, ba1, ba2 = d, h_name, acc_name
                    else:
                        # No H found, use heavy-atom distance as fallback
                        for acc_name, acc_coord in acc1:
                            d = np.linalg.norm(donor_coord - acc_coord)
                            if d < best_h:
                                best_h, ba1, ba2 = d, donor_name, acc_name
            
            if best_h <= cutoffs["hbond"]:
                add_row("hbond", best_h, ba1, ba2)

        if "hydrophobic" in cutoffs and rn1 in HYDROPHOBIC_RESIDUES and rn2 in HYDROPHOBIC_RESIDUES:
            if dist_any <= cutoffs["hydrophobic"]:
                add_row("hydrophobic", dist_any, atom_any1, atom_any2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Deduplicate: same (res1, res2, type) might appear from multiple atom pairs; keep min distance
    df = (
        df.groupby(
            ["chain1", "resnum1", "resname1", "chain2", "resnum2", "resname2", "interaction_type"],
            as_index=False,
        )
        .agg({"distance_angstrom": "min", "atom1": "first", "atom2": "first"})
    )
    return df.sort_values(["chain1", "resnum1", "chain2", "resnum2", "interaction_type"]).reset_index(
        drop=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze inter-chain interactions in a PDB structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("pdb", type=str, help="Path to PDB file")
    ap.add_argument(
        "--chains",
        nargs="+",
        default=None,
        metavar="ID",
        help="Chain IDs to analyze (default: all chains)",
    )
    ap.add_argument(
        "--interactions",
        nargs="+",
        choices=["ionic", "hbond", "hydrophobic", "contact"],
        default=["ionic", "hbond", "hydrophobic", "contact"],
        help="Interaction types to report",
    )
    ap.add_argument(
        "--distance-ionic",
        type=float,
        default=None,
        metavar="Å",
        help=f"Ionic/salt-bridge cutoff (default: {DEFAULT_DISTANCES['ionic']} Å)",
    )
    ap.add_argument(
        "--distance-hbond",
        type=float,
        default=None,
        metavar="Å",
        help=f"H-bond cutoff (default: {DEFAULT_DISTANCES['hbond']} Å)",
    )
    ap.add_argument(
        "--distance-hydrophobic",
        type=float,
        default=None,
        metavar="Å",
        help=f"Hydrophobic contact cutoff (default: {DEFAULT_DISTANCES['hydrophobic']} Å)",
    )
    ap.add_argument(
        "--distance-contact",
        type=float,
        default=None,
        metavar="Å",
        help=f"Generic contact cutoff (default: {DEFAULT_DISTANCES['contact']} Å)",
    )
    ap.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV path (default: print to stdout)",
    )
    ap.add_argument(
        "--no-header",
        action="store_true",
        help="Omit header row when printing to stdout",
    )
    args = ap.parse_args()

    distances = {}
    if args.distance_ionic is not None:
        distances["ionic"] = args.distance_ionic
    if args.distance_hbond is not None:
        distances["hbond"] = args.distance_hbond
    if args.distance_hydrophobic is not None:
        distances["hydrophobic"] = args.distance_hydrophobic
    if args.distance_contact is not None:
        distances["contact"] = args.distance_contact

    df = run_analysis(
        args.pdb,
        chains=args.chains,
        interaction_types=args.interactions,
        distances=distances if distances else None,
    )

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} interactions to {args.output}")
    else:
        print(df.to_csv(index=False, header=not args.no_header))


if __name__ == "__main__":
    main()
