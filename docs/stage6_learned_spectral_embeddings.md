# Stage 6 — Learned Spectral Embeddings

## Objective

Stage 6 introduces a neural spectral representation model for Enveda CASMI 2026.

The objective is to learn an embedding space in which spectra originating from the same molecular structure are close together, while spectra belonging to different structures are separated.

The model is trained only on training structures and evaluated on completely structure-disjoint validation structures.

---

## Dataset

Training cache:

- Training spectra: 1,589,761
- Training structures: 248,325
- Structures with at least 2 spectra: 232,980

Validation cache:

- Validation spectra: 129,408
- Validation structures: 27,485

Training and validation molecular structures are completely disjoint.

---

## Spectrum Representation

Each spectrum contains up to 128 peaks.

Each peak is represented using:

- normalized m/z
- normalized intensity

Additional spectrum information:

- precursor m/z

Input shapes:

```text
peaks        [batch, 128, 2]
mask         [batch, 128]
precursor    [batch]