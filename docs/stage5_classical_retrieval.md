# Stage 5 — Classical Spectral Retrieval Baseline

## Enveda CASMI 2026 — Molecule ID From Mass Spectra

Stage 5 implements and validates the first complete retrieval baseline for the project.

The objective of this stage was to establish a strong non-neural baseline before moving to learned spectral representations.

The pipeline developed in this stage is:

```text
MS/MS spectra
    ↓
Stage 3 preprocessing
    ↓
Sparse m/z binning
    ↓
L2 normalization
    ↓
Cosine similarity
    ↓
Reference spectrum retrieval
    ↓
Spectrum → molecule score aggregation
    ↓
Multi-spectrum aggregation
    ↓
Stage 4 neutral-mass constraints
    ↓
Ranked molecule candidates