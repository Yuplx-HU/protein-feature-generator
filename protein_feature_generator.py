import io
import gc
import psutil
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import torch
import esm
from gor4 import GOR4


_gor4_instance = None
_cached_models = {}

all_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
amino_acid_groups = {
    'hydrophobic': ['A', 'V', 'L', 'I', 'P', 'F', 'W', 'M'],
    'polar': ['G', 'S', 'T', 'C', 'Y', 'N', 'Q'],
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E'],
    'aromatic': ['F', 'Y', 'W'],
    'aliphatic': ['A', 'V', 'L', 'I'],
    'small': ['A', 'G', 'C', 'D', 'S', 'T', 'N', 'P'],
    'tiny': ['A', 'G', 'S']
}
Kyte_Doolittle = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}


def _get_gor4_instance():
    global _gor4_instance
    if _gor4_instance is None:
        _gor4_instance = GOR4()
    return _gor4_instance


def get_protein_sequence(uniprot_id: str, timeout: int = 30):
    if not uniprot_id:
        return "", ""
    
    try:
        response = requests.get(
            f"https://rest.uniprot.org/uniprot/{uniprot_id}.fasta",
            timeout=timeout
        )
        if response.status_code == 200:
            fasta_file = io.StringIO(response.text)
            record = SeqIO.read(fasta_file, "fasta")
            return str(record.seq), "UniProtKB"
    except Exception:
        pass

    try:
        response = requests.get(
            f"https://rest.uniprot.org/uniparc/search?query={uniprot_id}&fields=upi",
            timeout=timeout
        )
        if response.status_code == 200:
            uniparc_id = response.json()['results'][0]['uniParcId']
            
            response = requests.get(
                f"https://rest.uniprot.org/uniparc/{uniparc_id}.fasta",
                timeout=timeout
            )
            if response.status_code == 200:
                fasta_file = io.StringIO(response.text)
                record = SeqIO.read(fasta_file, "fasta")
                return str(record.seq), "UniParc"
    except Exception:
        pass
            
    raise Exception(f"Can not find {uniprot_id}")


def get_protein_physicochemical_features(sequence: str):
    sequence = sequence.strip().upper()
    cleaned_seq = ''.join([aa for aa in sequence if aa in all_amino_acids])
    
    try:
        analysed_seq = ProteinAnalysis(cleaned_seq)
        features = {}
        
        seq_len = len(cleaned_seq)
        
        features.update({
            "sequence_length": float(seq_len),
            "molecular_weight": round(float(analysed_seq.molecular_weight()), 6),
            "isoelectric_point": round(float(analysed_seq.isoelectric_point()), 6),
            "instability_index": round(float(analysed_seq.instability_index()), 6),
            "Aromaticity": round(float(analysed_seq.aromaticity()), 6),
            "gravy": round(float(analysed_seq.gravy()), 6)
        })

        aa_count = {aa: cleaned_seq.count(aa) for aa in all_amino_acids}
        
        for group_name in ['hydrophobic', 'positive', 'negative']:
            aa_list = amino_acid_groups[group_name]
            count = sum(aa_count[aa] for aa in aa_list)
            features[f"group_{group_name}_percent"] = round(count / seq_len * 100, 4)

        hydrophobicity_vals = [float(Kyte_Doolittle[aa]) for aa in cleaned_seq]
        hydrophobic_count = sum(1 for val in hydrophobicity_vals if val > 0)
        hydrophilic_count = sum(1 for val in hydrophobicity_vals if val < 0)
        features.update({
            "hydrophobicity_std": round(float(np.std(hydrophobicity_vals)), 6),
            "hydrophobic_residue_percent": round(hydrophobic_count / seq_len * 100, 4),
            "hydrophilic_residue_percent": round(hydrophilic_count / seq_len * 100, 4)
        })

        result = _get_gor4_instance().predict(sequence)
        predictions = result['predictions']
        
        h_ratio = predictions.count('H') / len(sequence)
        e_ratio = predictions.count('E') / len(sequence)
        c_ratio = predictions.count('C') / len(sequence)
        
        features.update({
            "h_ratio": h_ratio,
            "e_ratio": e_ratio,
            "c_ratio": c_ratio
        })

    except Exception as e:
        raise RuntimeError(f"Failed to calculate features for sequence '{sequence}': {str(e)}") from e

    return features


def get_protein_embedding_features(data: List[Tuple[str, str]], batch_size: int = 8, device: str = "cpu",
                                   model_name: str = "esm2_t6_8M_UR50D", level: str = "sequence",
                                   verbose: bool = True):
    if not data:
        return []
    
    global _cached_models
    device_torch = torch.device(device)
    
    if model_name not in _cached_models:
        model, alphabet = esm.pretrained.__dict__[model_name]()
        model = model.to(device_torch)
        model.eval()
        _cached_models[model_name] = (model, alphabet)
    else:
        model, alphabet = _cached_models[model_name]
    
    features = []
    for i in tqdm(range(0, len(data), batch_size), total=len(data) // batch_size, desc="Process batchs", leave=False, unit="batch", disable=not verbose):
        if verbose:
            mem = psutil.virtual_memory()
            tqdm.write(f"Memory used: {mem.percent}% | Available: {mem.available / 1024 / 1024:.2f} MB")

        batch_data = data[i: i + batch_size]

        batch_labels, batch_strs, batch_tokens = alphabet.get_batch_converter()(batch_data)
        batch_tokens = batch_tokens.to(device_torch)

        with torch.no_grad():
            outputs = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
        
        for j, label in enumerate(batch_labels):
            seq_len = len(batch_strs[j])
            if level == "token":
                emb = outputs["representations"][model.num_layers][j, 1: seq_len+1, :].cpu().numpy()
                features.append((label, emb))
            elif level == "sequence":
                emb = outputs["representations"][model.num_layers][j, 0, :].cpu().numpy()
                features.append((label, emb))
            else:
                raise ValueError(f"Unsupported embedding feature level {level}")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    return features
