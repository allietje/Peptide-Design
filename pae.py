"""
Interface PAE summarizer for AlphaFold-Multimer server ZIP outputs.
Summary: outputs ipTM, mean PAE over interface residue pairs defined by contact probability or distance.
Input is the ZIP file outputted by AF2 webserver. Input path is argument in command line. 
Run with python pae.py path/to/af2_output.zip

What this script does
- Reads an AlphaFold(-Multimer) job ZIP (as downloaded from a web server).
- For each model seed (typically 0–4), it computes three interface confidence summaries:
  1) Contact-probability interface mean PAE for cross-chain residue pairs where contact_probs > 0.5
  2) Contact-probability interface mean PAE for cross-chain residue pairs where contact_probs > 0.2
  3) Distance-defined interface mean PAE for residue pairs across chains that have ANY heavy-atom distance < 5 Å
- Outputs a table with: model index, ipTM, ranking_score, counts of interface residue pairs used for each metric,
  and the corresponding mean interface PAE values in Å (lower is better for interface placement confidence).

Input files used (inside the ZIP)
For each model k:
- *_full_data_k.json
  - pae               (NxN predicted aligned error matrix, in Å)
  - contact_probs     (NxN contact probability matrix)
  - token_chain_ids   (length N chain ID per residue token; used to split binder/target chains)
  - token_res_ids     (length N residue number per token; used to map structure residues back to PAE indices)
- *_summary_confidences_k.json
  - iptm
  - ranking_score (optional, but commonly present)
- *_model_k.cif
  - atomic coordinates used to detect heavy-atom contacts across chains for distance-defined interface pairs

How “interface PAE” is computed here
- Contact-probability interface PAE:
  - Select cross-chain residue pairs (chainA vs chainB) with contact_probs > threshold.
  - Average the corresponding pae[i,j] values (A→B block).
- Distance-defined interface PAE:
  - Find cross-chain residue pairs (A residue, B residue) that have any heavy-atom contact < cutoff Å.
  - Map those residue numbers to PAE token indices via token_chain_ids/token_res_ids.
  - Average pae[tokenA, tokenB] over those contacting residue pairs.

Assumptions / caveats
- Chain IDs in the JSON and CIF must match (e.g., "A" for binder, "B" for target).
- token_res_ids must correspond to the residue numbering in the CIF for mapping to work.
- This reports geometric confidence signals (ipTM/PAE), NOT binding affinity.
"""

import argparse
import json
import os
import re
import tempfile
import zipfile
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch


def load_json_from_zip(zf: zipfile.ZipFile, name: str) -> dict:
    return json.loads(zf.read(name).decode("utf-8"))


def model_indices_from_zip(namelist: Iterable[str]) -> List[int]:
    """Find model indices k from files like *_full_data_k.json."""
    idx = set()
    pat = re.compile(r"_full_data_(\d+)\.json$")
    for n in namelist:
        m = pat.search(n)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def interface_pae_by_contactprob(
    full_data: dict, chainA: str = "A", chainB: str = "B", thr: float = 0.5
) -> Tuple[int, float]:
    """
    Mean PAE over cross-chain residue pairs where contact_probs > thr.
    Uses the A->B block (rows in A, cols in B).
    """
    pae = np.array(full_data["pae"], dtype=float)
    cp = np.array(full_data["contact_probs"], dtype=float)
    ch = np.array(full_data["token_chain_ids"])

    idxA = np.where(ch == chainA)[0]
    idxB = np.where(ch == chainB)[0]

    pae_AB = pae[np.ix_(idxA, idxB)]
    cp_AB = cp[np.ix_(idxA, idxB)]

    mask = cp_AB > thr
    if not mask.any():
        return 0, float("nan")

    return int(mask.sum()), float(pae_AB[mask].mean())


def interface_pae_all_interchain(
    full_data: dict, chainA: str = "A", chainB: str = "B"
) -> Tuple[float, float, float]:
    """
    Baker-lab style pAE_interaction:
    mean PAE over ALL inter-chain residue pairs (entire off-diagonal blocks),
    averaged over both directions (A->B and B->A).

    Returns:
        (mean_AB, mean_BA, mean_avg)
    """
    pae = np.array(full_data["pae"], dtype=float)
    ch = np.array(full_data["token_chain_ids"])

    idxA = np.where(ch == chainA)[0]
    idxB = np.where(ch == chainB)[0]

    AB = pae[np.ix_(idxA, idxB)]  # A -> B
    BA = pae[np.ix_(idxB, idxA)]  # B -> A

    mean_AB = float(AB.mean())
    mean_BA = float(BA.mean())
    mean_avg = (mean_AB + mean_BA) / 2.0
    return mean_AB, mean_BA, mean_avg


def interface_pae_by_distance(
    zip_path: str,
    model_cif_name: str,
    full_data: dict,
    chainA: str = "A",
    chainB: str = "B",
    cutoff: float = 5.0,
) -> Tuple[int, float]:
    """
    Distance-defined interface:
    - Find residue pairs (A_res, B_res) that have ANY heavy-atom distance < cutoff Å
    - Map those residue numbers back to PAE indices via token_chain_ids + token_res_ids
    - Mean PAE over those contacting residue pairs using PAE[tokenA, tokenB] (A->B direction)

    Returns:
        (#contacting residue pairs used, mean_pae)
    """
    token_chain = np.array(full_data["token_chain_ids"])
    token_resid = np.array(full_data["token_res_ids"])
    pae = np.array(full_data["pae"], dtype=float)

    # Map (chain, res_id) -> token index in PAE matrix
    tok_index = {(c, int(r)): i for i, (c, r) in enumerate(zip(token_chain, token_resid))}

    parser = MMCIFParser(QUIET=True)

    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as td:
        cif_path = os.path.join(td, os.path.basename(model_cif_name))
        with open(cif_path, "wb") as f:
            f.write(zf.read(model_cif_name))

        structure = parser.get_structure("m", cif_path)
        model = structure[0]

        if chainA not in model or chainB not in model:
            return 0, float("nan")

        chA = model[chainA]
        chB = model[chainB]

        # Build neighbor search on heavy atoms in chain B
        atoms_B = []
        atom_to_resB = {}
        for res in chB.get_residues():
            for atom in res.get_atoms():
                if atom.element != "H":
                    atoms_B.append(atom)
                    atom_to_resB[id(atom)] = res

        if not atoms_B:
            return 0, float("nan")

        ns = NeighborSearch(atoms_B)

        # Collect contacting residue-number pairs (A_resid, B_resid)
        contacting_pairs = set()
        for resA in chA.get_residues():
            for atomA in resA.get_atoms():
                if atomA.element == "H":
                    continue
                neighbors = ns.search(atomA.coord, cutoff, level="A")
                for atomB in neighbors:
                    resB = atom_to_resB[id(atomB)]
                    contacting_pairs.add((int(resA.get_id()[1]), int(resB.get_id()[1])))

        if not contacting_pairs:
            return 0, float("nan")

        # Convert to PAE indices and take values (A->B)
        vals = []
        for a_resid, b_resid in contacting_pairs:
            ia = tok_index.get((chainA, a_resid))
            ib = tok_index.get((chainB, b_resid))
            if ia is None or ib is None:
                continue
            vals.append(pae[ia, ib])

        if not vals:
            return 0, float("nan")

        return len(vals), float(np.mean(vals))


def zip_to_interface_table(
    zip_path: str,
    chainA: str = "A",
    chainB: str = "B",
    cutoff: float = 5.0,
    thr_list: Tuple[float, ...] = (0.5, 0.2),
) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        ks = model_indices_from_zip(names)
        if not ks:
            raise RuntimeError("No *_full_data_#.json files found in the ZIP. Is this an AlphaFold server ZIP?")

        rows = []
        for k in ks:
            full_name = [n for n in names if n.endswith(f"_full_data_{k}.json")]
            summ_name = [n for n in names if n.endswith(f"_summary_confidences_{k}.json")]
            cif_name = [n for n in names if n.endswith(f"_model_{k}.cif")]

            if not full_name or not summ_name or not cif_name:
                raise RuntimeError(
                    f"Missing required files for model {k}. "
                    f"Need *_full_data_{k}.json, *_summary_confidences_{k}.json, *_model_{k}.cif"
                )

            full = load_json_from_zip(zf, full_name[0])
            summ = load_json_from_zip(zf, summ_name[0])

            row = {
                "model": k,
                "ipTM": float(summ.get("iptm", np.nan)),
                "ranking_score": float(summ.get("ranking_score", np.nan)),
            }

            # Contact-probability thresholds
            for thr in thr_list:
                n_pairs, mean_pae = interface_pae_by_contactprob(full, chainA, chainB, thr=thr)
                row[f"#pairs p>{thr}"] = n_pairs
                row[f"mean iPAE p>{thr} (Å)"] = mean_pae

            # Distance-defined interface
            n_dist, mean_dist = interface_pae_by_distance(zip_path, cif_name[0], full, chainA, chainB, cutoff=cutoff)
            row[f"#pairs dist<{cutoff}Å"] = n_dist
            row[f"mean iPAE dist<{cutoff}Å (Å)"] = mean_dist

            # NEW: All-pairs inter-chain mean PAE (Baker-lab style pAE_interaction)
            all_AB, all_BA, all_avg = interface_pae_all_interchain(full, chainA, chainB)
            row["mean PAE all A->B (Å)"] = all_AB
            row["mean PAE all B->A (Å)"] = all_BA
            row["pAE_interaction all-pairs (Å)"] = all_avg

            rows.append(row)

        df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
        return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute interface mean PAE metrics from an AlphaFold(-Multimer) server ZIP."
    )
    parser.add_argument("zip_path", help="Path to the AlphaFold job ZIP (downloaded from the web server).")
    parser.add_argument("--chainA", default="A", help="Binder chain ID (default: A).")
    parser.add_argument("--chainB", default="B", help="Target chain ID (default: B).")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Heavy-atom distance cutoff in Å (default: 5.0).")
    parser.add_argument(
        "--thr",
        type=float,
        nargs="*",
        default=[0.5, 0.2],
        help="Contact probability thresholds (default: 0.5 0.2).",
    )
    parser.add_argument("--out", default=None, help="Optional output CSV path. If omitted, prints to stdout.")
    args = parser.parse_args()

    df = zip_to_interface_table(
        args.zip_path,
        chainA=args.chainA,
        chainB=args.chainB,
        cutoff=args.cutoff,
        thr_list=tuple(args.thr),
    )

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    else:
        # pretty print
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
