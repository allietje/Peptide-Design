#!/usr/bin/env python3
"""
Renumber receptor chain in a design PDB file to match the reference PDB file.

The script:
1. Identifies receptor and peptide chains in both files
2. Aligns receptor sequences to map residue numbers
3. Renumbers the receptor chain in the design file to match reference numbering
4. Renames chain IDs to match the reference (receptor and peptide)
5. Outputs a new PDB file

Usage:
  python renumber_pdb.py reference.pdb design.pdb output.pdb
  python renumber_pdb.py glp1.pdb best_glp1_design.pdb renumbered_design.pdb
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

from Bio import pairwise2
from Bio.PDB import PDBIO, PDBParser, Residue, Structure
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


def get_receptor_and_peptide_chains(
    structure: Structure, ref_receptor_id: Optional[str] = None, ref_peptide_id: Optional[str] = None
) -> Tuple[Optional[Residue], Optional[Residue]]:
    """
    Identify receptor and peptide chains in a structure.
    
    If reference IDs are provided, use those. Otherwise, identify by length
    (receptor is longer, typically >100 residues; peptide is shorter, typically <100).
    """
    receptor_chain = None
    peptide_chain = None
    
    for chain in structure[0]:
        residues = [r for r in chain if is_aa(r, standard=True)]
        if not residues:
            continue
        
        # If reference IDs provided, use them (prioritize these)
        if ref_receptor_id and chain.id == ref_receptor_id:
            receptor_chain = chain
            continue
        if ref_peptide_id and chain.id == ref_peptide_id:
            peptide_chain = chain
            continue
        
        # Otherwise identify by length (only if not already found)
        if receptor_chain is None and len(residues) > 100:
            receptor_chain = chain
        elif peptide_chain is None and len(residues) < 100:
            peptide_chain = chain
    
    return receptor_chain, peptide_chain


def get_residue_sequence(chain) -> Tuple[List[Residue], str]:
    """Extract sequence and residue list from a chain."""
    residues = [r for r in chain if is_aa(r, standard=True)]
    sequence = "".join([seq1(r.get_resname()) for r in residues])
    return residues, sequence


def align_sequences(seq1: str, seq2: str) -> Dict[int, int]:
    """
    Align two sequences and return mapping from seq2 indices to seq1 indices.
    
    Returns dict mapping: design_residue_index -> reference_residue_index
    """
    # If sequences are identical, use simple 1-to-1 mapping
    if seq1 == seq2:
        return {i: i for i in range(len(seq1))}
    
    # Use global alignment with gap penalties
    alignments = pairwise2.align.globalxx(seq1, seq2)
    if not alignments:
        raise ValueError("Failed to align sequences")
    
    # Use the best alignment
    alignment = alignments[0]
    ref_aligned = alignment.seqA
    des_aligned = alignment.seqB
    
    # Build mapping: design index -> reference index
    mapping = {}
    ref_idx = 0
    des_idx = 0
    
    for i in range(len(ref_aligned)):
        ref_char = ref_aligned[i]
        des_char = des_aligned[i]
        
        if ref_char != "-":
            ref_idx += 1
        if des_char != "-":
            des_idx += 1
            if ref_char != "-":  # Only map if both are not gaps
                mapping[des_idx - 1] = ref_idx - 1  # 0-indexed
    
    return mapping


def renumber_pdb(
    reference_pdb: str,
    design_pdb: str,
    output_pdb: str,
    ref_receptor_id: Optional[str] = None,
    ref_peptide_id: Optional[str] = None,
) -> None:
    """
    Renumber receptor chain in design PDB to match reference PDB numbering.
    
    Parameters
    ----------
    reference_pdb : str
        Path to reference PDB file
    design_pdb : str
        Path to design PDB file to renumber
    output_pdb : str
        Path for output PDB file
    ref_receptor_id : str, optional
        Receptor chain ID in reference (auto-detect if None)
    ref_peptide_id : str, optional
        Peptide chain ID in reference (auto-detect if None)
    """
    parser = PDBParser(QUIET=True)
    
    # Parse structures
    ref_structure = parser.get_structure("reference", reference_pdb)
    des_structure = parser.get_structure("design", design_pdb)
    
    # Identify chains
    # For reference, try to use explicit IDs first (common: R for receptor, P for peptide)
    if ref_receptor_id is None:
        # Try common receptor chain IDs
        for cid in ['R', 'A']:
            for chain in ref_structure[0]:
                if chain.id == cid:
                    residues = [r for r in chain if is_aa(r, standard=True)]
                    if len(residues) > 100:
                        ref_receptor_id = cid
                        break
            if ref_receptor_id:
                break
    
    if ref_peptide_id is None:
        # Try common peptide chain IDs
        for cid in ['P', 'B']:
            for chain in ref_structure[0]:
                if chain.id == cid:
                    residues = [r for r in chain if is_aa(r, standard=True)]
                    if len(residues) < 100:
                        ref_peptide_id = cid
                        break
            if ref_peptide_id:
                break
    
    ref_receptor, ref_peptide = get_receptor_and_peptide_chains(
        ref_structure, ref_receptor_id, ref_peptide_id
    )
    des_receptor, des_peptide = get_receptor_and_peptide_chains(
        des_structure, None, None
    )
    
    if ref_receptor is None:
        raise ValueError("Could not identify receptor chain in reference PDB")
    if des_receptor is None:
        raise ValueError("Could not identify receptor chain in design PDB")
    if ref_peptide is None:
        print("Warning: Could not identify peptide chain in reference PDB", file=sys.stderr)
    if des_peptide is None:
        print("Warning: Could not identify peptide chain in design PDB", file=sys.stderr)
    
    # Get sequences
    ref_residues, ref_seq = get_residue_sequence(ref_receptor)
    des_residues, des_seq = get_residue_sequence(des_receptor)
    
    print(f"Reference receptor chain: {ref_receptor.id}, {len(ref_residues)} residues", file=sys.stderr)
    print(f"Design receptor chain: {des_receptor.id}, {len(des_residues)} residues", file=sys.stderr)
    
    # Align sequences to get residue number mapping
    print("Aligning receptor sequences...", file=sys.stderr)
    mapping = align_sequences(ref_seq, des_seq)
    print(f"Mapped {len(mapping)} residues", file=sys.stderr)
    
    # Create mapping from design residue index to (reference residue number, insertion code)
    residue_mapping: Dict[int, Tuple[int, str]] = {}
    for des_idx, ref_idx in mapping.items():
        if des_idx < len(des_residues) and ref_idx < len(ref_residues):
            ref_res = ref_residues[ref_idx]
            # Get the residue number and insertion code from reference
            ref_resnum = ref_res.id[1]  # (hetflag, resnum, insertion_code)
            ref_inscode = ref_res.id[2]
            residue_mapping[des_idx] = (ref_resnum, ref_inscode)
    
    print(f"Created mapping for {len(residue_mapping)} residues", file=sys.stderr)
    
    # Store original chain IDs before any modifications
    des_receptor_orig_id = des_receptor.id
    des_peptide_orig_id = des_peptide.id if des_peptide else None
    
    # Renumber receptor chain in design structure
    # Use two-pass approach to avoid ID conflicts:
    # 1. First pass: renumber to temporary high numbers
    # 2. Second pass: renumber to final reference numbers
    for model in des_structure:
        for chain in model:
            if chain.id == des_receptor_orig_id:
                # Get list of residues (need to convert to list since we'll modify during iteration)
                chain_residues = [r for r in chain if is_aa(r, standard=True)]
                
                # First pass: renumber to temporary numbers (start from 100000 to avoid conflicts)
                temp_ids = {}
                for idx, res in enumerate(chain_residues):
                    if idx in residue_mapping:
                        old_id = res.id
                        temp_num = 100000 + idx
                        temp_id = (old_id[0], temp_num, old_id[2])
                        temp_ids[idx] = temp_id
                        res.id = temp_id
                
                # Second pass: renumber to final reference numbers
                for idx, res in enumerate(chain_residues):
                    if idx in residue_mapping:
                        temp_id = temp_ids[idx]
                        new_resnum, new_inscode = residue_mapping[idx]
                        # Use reference insertion code, or preserve design insertion code if ref has none
                        if new_inscode == " ":
                            new_inscode = temp_id[2]
                        new_id = (temp_id[0], new_resnum, new_inscode)
                        res.id = new_id
                    else:
                        print(f"Warning: Design residue {idx} ({res.get_resname()}) not in mapping", file=sys.stderr)
                
                # Rename chain ID to match reference
                old_receptor_id = chain.id
                chain.id = ref_receptor.id
                if old_receptor_id != ref_receptor.id:
                    print(f"Renamed receptor chain {old_receptor_id} -> {ref_receptor.id}", file=sys.stderr)
            
            elif des_peptide_orig_id and chain.id == des_peptide_orig_id:
                # Rename peptide chain ID to match reference
                if ref_peptide:
                    old_peptide_id = chain.id
                    chain.id = ref_peptide.id
                    if old_peptide_id != ref_peptide.id:
                        print(f"Renamed peptide chain {old_peptide_id} -> {ref_peptide.id}", file=sys.stderr)
    
    # Write output
    io = PDBIO()
    io.set_structure(des_structure)
    io.save(output_pdb)
    print(f"Wrote renumbered PDB to {output_pdb}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Renumber receptor chain in design PDB to match reference PDB numbering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("reference", type=str, help="Path to reference PDB file")
    ap.add_argument("design", type=str, help="Path to design PDB file to renumber")
    ap.add_argument("output", type=str, help="Path for output PDB file")
    ap.add_argument(
        "--ref-receptor-id",
        type=str,
        default=None,
        help="Receptor chain ID in reference (auto-detect if not specified)",
    )
    ap.add_argument(
        "--ref-peptide-id",
        type=str,
        default=None,
        help="Peptide chain ID in reference (auto-detect if not specified)",
    )
    
    args = ap.parse_args()
    
    try:
        renumber_pdb(
            args.reference,
            args.design,
            args.output,
            ref_receptor_id=args.ref_receptor_id,
            ref_peptide_id=args.ref_peptide_id,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
