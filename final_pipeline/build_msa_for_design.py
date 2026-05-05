#!/usr/bin/env python3
"""Build a paired multimer .a3m from a receptor-only MSA and a design FASTA.

Usage: python build_msa_for_design.py <design.fasta> <receptor_only.a3m> <output.a3m>

The design FASTA must contain a receptor:peptide sequence separated by ':'.
The receptor-only .a3m is a standard .a3m file with only receptor homologs.
The output will be a correctly-formatted paired multimer .a3m with the
peptide columns filled with gaps (since designed peptides have no homologs).
"""
import sys
from pathlib import Path

# Allow importing from the pipeline directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_fold import build_paired_msa


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <design.fasta> <receptor_only.a3m> <output.a3m>")
        sys.exit(1)

    fasta_path, receptor_msa_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(fasta_path) as f:
        lines = [l.strip() for l in f if not l.startswith(">")]
    full_seq = "".join(lines)
    parts = full_seq.split(":")
    if len(parts) != 2:
        print(f"ERROR: expected receptor:peptide in FASTA, got {len(parts)} parts")
        sys.exit(1)

    rec_seq, pep_seq = parts[0], parts[1]
    build_paired_msa(receptor_msa_path, pep_seq, rec_seq, output_path)
    print(f"Built paired MSA: rec={len(rec_seq)} pep={len(pep_seq)} -> {output_path}")


if __name__ == "__main__":
    main()
