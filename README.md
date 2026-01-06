# Protein Feature Extraction Tool

A lightweight Python package for extracting comprehensive features from protein sequences, including physicochemical properties and ESM-2 model embeddings.

## Features

- **Sequence Retrieval**: Fetch protein sequences from UniProt/UniParc databases
- **Physicochemical Analysis**: Calculate molecular weight, isoelectric point, hydrophobicity, instability index, and more
- **Secondary Structure**: Predict secondary structure using GOR4 algorithm
- **Embedding Generation**: Extract protein embeddings using ESM-2 models (sequence-level or token-level)
- **Memory Efficient**: Built-in caching and memory management for large-scale processing

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (CPU or CUDA version)

### Install Dependencies
```bash
pip install torch biopython numpy requests tqdm psutil
```

### Install ESM
```bash
pip install fair-esm
```

### Install GOR4
```bash
pip install gor4
```

## Quick Start

```python
from protein_features import get_protein_sequence, get_protein_physicochemical_features, get_protein_embedding_features

# 1. Fetch protein sequence from UniProt
sequence, source = get_protein_sequence("P12345")
print(f"Sequence from {source}: {sequence[:20]}...")

# 2. Calculate physicochemical features
features = get_protein_physicochemical_features(sequence)
print(f"Molecular weight: {features['molecular_weight']}")
print(f"Isoelectric point: {features['isoelectric_point']}")

# 3. Generate ESM embeddings
data = [("P12345", sequence)]
embeddings = get_protein_embedding_features(
    data=data,
    model_name="esm2_t6_8M_UR50D",  # Other options: esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, etc.
    level="sequence",  # or "token" for per-residue embeddings
    device="cpu",  # or "cuda" for GPU acceleration
    batch_size=8
)
```

## Function Reference

### `get_protein_sequence(uniprot_id: str, timeout: int = 30)`
Fetches protein sequence from UniProt databases.

**Parameters:**
- `uniprot_id`: UniProt accession ID
- `timeout`: Request timeout in seconds

**Returns:**
- Tuple of (sequence, database_source)

### `get_protein_physicochemical_features(sequence: str)`
Calculates comprehensive physicochemical properties.

**Returns dictionary with:**
- `sequence_length`, `molecular_weight`, `isoelectric_point`
- `instability_index`, `Aromaticity`, `gravy`
- Hydrophobicity statistics, amino acid group percentages
- Secondary structure ratios (helix, sheet, coil)

### `get_protein_embedding_features(data: List[Tuple[str, str]], **kwargs)`
Generates protein embeddings using ESM-2 models.

**Parameters:**
- `data`: List of (label, sequence) tuples
- `batch_size`: Processing batch size (default: 8)
- `device`: "cpu" or "cuda" (default: "cpu")
- `model_name`: ESM-2 model name (default: "esm2_t6_8M_UR50D")
- `level`: "sequence" for per-protein or "token" for per-residue embeddings
- `verbose`: Show progress and memory usage

## Available ESM-2 Models

| Model Name | Parameters | Embedding Dim |
|------------|------------|---------------|
| esm2_t6_8M_UR50D | 8M | 320 |
| esm2_t12_35M_UR50D | 35M | 480 |
| esm2_t30_150M_UR50D | 150M | 640 |
| esm2_t33_650M_UR50D | 650M | 1280 |
| esm2_t36_3B_UR50D | 3B | 2560 |

## Amino Acid Groups

The tool categorizes amino acids into functional groups:
- **Hydrophobic**: A, V, L, I, P, F, W, M
- **Polar**: G, S, T, C, Y, N, Q
- **Positive**: K, R, H
- **Negative**: D, E
- **Aromatic**: F, Y, W
- **Aliphatic**: A, V, L, I
- **Small**: A, G, C, D, S, T, N, P
- **Tiny**: A, G, S

## Error Handling

- Invalid sequences raise `ValueError`
- Network errors raise `Exception` with descriptive messages
- Feature calculation errors raise `RuntimeError`

## Memory Management

- Automatic GPU memory clearing (if using CUDA)
- Progress tracking with `tqdm`
- Memory usage monitoring with `psutil`
- Model caching to avoid repeated loading

## Example: Batch Processing

```python
# Process multiple proteins
uniprot_ids = ["P12345", "Q98765", "O54321"]
data = []

for uid in uniprot_ids:
    seq, _ = get_protein_sequence(uid)
    data.append((uid, seq))

# Get features for all
for uid, seq in data:
    features = get_protein_physicochemical_features(seq)
    print(f"{uid}: {features['molecular_weight']:.2f} Da")

# Get embeddings in batch
embeddings = get_protein_embedding_features(
    data=data,
    model_name="esm2_t12_35M_UR50D",
    batch_size=4
)
```
