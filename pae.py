"""
Interface PAE summarizer for AlphaFold-Multimer server ZIP outputs.
Summary: outputs ipTM, mean PAE over interface residue pairs defined by contact probability or distance.
Input is the ZIP file outputted by AF2 webserver. Input path is at the end of the code, change to reflect where the zip file is.

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


import zipfile, json, re, tempfile, os
import numpy as np
import pandas as pd

from Bio.PDB import MMCIFParser, NeighborSearch

def load_json_from_zip(zf, name):
    return json.loads(zf.read(name).decode("utf-8"))

def model_indices_from_zip(namelist):
    # Finds k from files like *_full_data_k.json
    idx = set()
    pat = re.compile(r"_full_data_(\d+)\.json$")
    for n in namelist:
        m = pat.search(n)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)

def interface_pae_by_contactprob(full_data, chainA="A", chainB="B", thr=0.5):
    pae = np.array(full_data["pae"], dtype=float)
    cp  = np.array(full_data["contact_probs"], dtype=float)
    ch  = np.array(full_data["token_chain_ids"])

    idxA = np.where(ch == chainA)[0]
    idxB = np.where(ch == chainB)[0]

    pae_AB = pae[np.ix_(idxA, idxB)]
    cp_AB  = cp[np.ix_(idxA, idxB)]

    mask = cp_AB > thr
    if not mask.any():
        return 0, float("nan")

    return int(mask.sum()), float(pae_AB[mask].mean())

def interface_pae_by_distance(zip_path, model_cif_name, full_data, chainA="A", chainB="B", cutoff=5.0):
    """
    Distance-defined interface:
    - Find residue pairs (A_res, B_res) that have ANY heavy-atom distance < cutoff Å
    - Then average PAE over the corresponding residue-pair tokens.

    Uses token_chain_ids + token_res_ids to map residues -> PAE indices.
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

        # Collect heavy atoms (non-H) for chain B, build neighbor search
        atoms_B = []
        atom_to_resB = {}
        for res in chB.get_residues():
            # skip hetero/water if you want: res.id[0] == " "
            for atom in res.get_atoms():
                if atom.element != "H":
                    atoms_B.append(atom)
                    atom_to_resB[id(atom)] = res

        if not atoms_B:
            return 0, float("nan")

        ns = NeighborSearch(atoms_B)

        # Find contacting residue pairs
        contacting_pairs = set()  # (A_resid, B_resid)
        for resA in chA.get_residues():
            for atomA in resA.get_atoms():
                if atomA.element == "H":
                    continue
                neighbors = ns.search(atomA.coord, cutoff, level="A")  # atoms within cutoff
                if not neighbors:
                    continue
                for atomB in neighbors:
                    resB = atom_to_resB[id(atomB)]
                    # Residue number is res.get_id()[1]
                    contacting_pairs.add((int(resA.get_id()[1]), int(resB.get_id()[1])))

        if not contacting_pairs:
            return 0, float("nan")

        # Convert contacting residue pairs -> PAE indices
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

def zip_to_interface_table(zip_path, chainA="A", chainB="B", cutoff=5.0, thr_list=(0.5, 0.2)):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        ks = model_indices_from_zip(names)

        rows = []
        for k in ks:
            full_name = [n for n in names if n.endswith(f"_full_data_{k}.json")][0]
            summ_name = [n for n in names if n.endswith(f"_summary_confidences_{k}.json")][0]
            cif_name  = [n for n in names if n.endswith(f"_model_{k}.cif")][0]

            full = load_json_from_zip(zf, full_name)
            summ = load_json_from_zip(zf, summ_name)

            row = {
                "model": k,
                "ipTM": float(summ.get("iptm", np.nan)),
                "ranking_score": float(summ.get("ranking_score", np.nan)),
            }

            for thr in thr_list:
                n_pairs, mean_pae = interface_pae_by_contactprob(full, chainA, chainB, thr=thr)
                row[f"#pairs p>{thr}"] = n_pairs
                row[f"mean iPAE p>{thr} (Å)"] = mean_pae

            n_dist, mean_dist = interface_pae_by_distance(zip_path, cif_name, full, chainA, chainB, cutoff=cutoff)
            row[f"#pairs dist<{cutoff}Å"] = n_dist
            row[f"mean iPAE dist<{cutoff}Å (Å)"] = mean_dist

            rows.append(row)

        df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
        return df

if __name__ == "__main__":
    zip_path = "fold_glp2_multimer.zip"  # change to your zip
    df = zip_to_interface_table(zip_path, chainA="A", chainB="B", cutoff=5.0, thr_list=(0.5, 0.2))
    print(df.to_string(index=False))
