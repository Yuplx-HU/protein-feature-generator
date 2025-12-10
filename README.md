# Protein Feature Generator

A comprehensive Python toolkit for extracting protein sequence features, including physicochemical properties, structural predictions, and state-of-the-art embeddings from ESM models.

## Features

- **Sequence Retrieval**: Fetch protein sequences from UniProt database
- **Physicochemical Analysis**: Calculate molecular weight, isoelectric point, hydrophobicity, amino acid composition, and more
- **Secondary Structure Prediction**: GOR4 algorithm for helix, sheet, and coil predictions
- **Deep Learning Embeddings**: Pre-trained ESM models for protein representations
- **Batch Processing**: Efficient batch processing with memory management
- **Flexible Output**: Token-level or sequence-level embeddings

## Installation

### Dependencies

```bash
pip install biopython numpy requests tqdm torch psutil
```

### Additional Dependencies

1. **ESM (Evolutionary Scale Modeling)**:
```bash
pip install fair-esm
```

2. **GOR4 (Secondary Structure Prediction)**:
```bash
# Install from PyPI
pip install gor4

# OR from GitHub
pip install git+https://github.com/psipred/gor4.git
```

## Quick Start

```python
from protein_feature_generator import ProteinFeatureGenerator

# Initialize generator
generator = ProteinFeatureGenerator(device="cuda", verbose=True)

# Example 1: Retrieve sequence from UniProt
sequence, source = generator.get_sequence("P12345")
print(f"Sequence from {source}: {sequence[:50]}...")

# Example 2: Calculate physicochemical features
features = generator.get_physicochemical_features(sequence)
print(f"Molecular weight: {features['molecular_weight']}")
print(f"Isoelectric point: {features['isoelectric_point']}")

# Example 3: Generate ESM embeddings
data = [("P12345", sequence), ("Q98765", "MKAILV...")]
embeddings = generator.get_embedding_features(
    data=data,
    batch_size=8,
    model_name="esm2_t6_8M_UR50D",
    feature_type="sequence"  # or "token" for per-residue embeddings
)
```

## API Reference

### `ProteinFeatureGenerator(device="cpu", verbose=True)`

**Parameters:**
- `device`: Compute device ("cpu" or "cuda")
- `verbose`: Enable progress bars and memory monitoring

### Methods

#### `get_sequence(uniprot_id: str, timeout: int = 30) -> Tuple[str, str]`

Retrieves protein sequence from UniProt.

**Parameters:**
- `uniprot_id`: UniProt accession ID
- `timeout`: Request timeout in seconds

**Returns:** Tuple of (sequence, source_database)

**Raises:**
- `Exception` if sequence cannot be found

#### `get_physicochemical_features(sequence: str) -> Dict`

Calculates comprehensive physicochemical properties.

**Parameters:**
- `sequence`: Protein amino acid sequence

**Returns:** Dictionary with 50+ calculated features

**Features include:**
- Basic properties: length, molecular weight, isoelectric point
- Stability: instability index, aromaticity
- Hydrophobicity: GRAVY, Kyte-Doolittle scores
- Amino acid composition: all 20 standard amino acids
- Chemical groups: hydrophobic, polar, charged residues
- Charge profile: at different pH values (0-14)
- Secondary structure: helix, sheet, coil ratios and probabilities (GOR4)

**Raises:**
- `ValueError` for invalid amino acids
- `RuntimeError` for calculation failures

#### `get_embedding_features(data: List[Tuple[str, str]], batch_size: int = 8, model_name: str = "esm2_t6_8M_UR50D", feature_type: str = "sequence") -> List[Tuple[str, np.ndarray]]`

Generates protein embeddings using ESM models.

**Parameters:**
- `data`: List of (label, sequence) tuples
- `batch_size`: Processing batch size
- `model_name`: ESM model identifier
- `feature_type`: "sequence" for pooled embedding or "token" for per-residue

**Supported Models:**
- `esm2_t6_8M_UR50D` (6 layers, 8M parameters)
- `esm2_t12_35M_UR50D` (12 layers, 35M parameters)
- `esm2_t30_150M_UR50D` (30 layers, 150M parameters)
- `esm2_t33_650M_UR50D` (33 layers, 650M parameters)
- `esm2_t36_3B_UR50D` (36 layers, 3B parameters)

**Returns:** List of (label, embedding) tuples

## Advanced Examples

### Batch Processing Multiple Proteins

```python
# Define protein IDs
protein_ids = ["P12345", "Q98765", "O43210"]

# Retrieve sequences
sequences = {}
for pid in tqdm(protein_ids, desc="Fetching sequences"):
    try:
        seq, src = generator.get_sequence(pid)
        sequences[pid] = seq
    except Exception as e:
        print(f"Failed to fetch {pid}: {e}")

# Calculate features for all sequences
all_features = {}
for pid, seq in sequences.items():
    features = generator.get_physicochemical_features(seq)
    all_features[pid] = features

# Generate embeddings
data = [(pid, seq) for pid, seq in sequences.items()]
embeddings = generator.get_embedding_features(
    data=data,
    batch_size=4,
    model_name="esm2_t12_35M_UR50D",
    feature_type="sequence"
)
```

## Memory Management

The generator includes automatic memory management:
- GPU memory clearing after each batch
- Memory usage monitoring (when verbose=True)
- Efficient batching for large datasets

## Error Handling

- **Invalid sequences**: Validates amino acid composition
- **Network errors**: Retries with timeout
- **Memory issues**: Automatic garbage collection
- **Model loading**: Caches models for repeated use

## Notes

1. ESM models are downloaded automatically on first use (~8MB to ~3GB depending on model)
2. GOR4 requires local installation
3. UniProt API has rate limits; consider implementing delays for bulk downloads
4. For very long sequences, reduce batch_size to avoid memory issues
