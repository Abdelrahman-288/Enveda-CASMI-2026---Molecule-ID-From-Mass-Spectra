# Stage 2 — Exploratory Data Analysis

## Objective

Understand the relationship between the training and test domains before designing the preprocessing and modeling pipeline.

---

## 1. Dataset Scale

### Training

- 2,539,608 spectra
- 275,810 unique InChIKey14 structures

### Test

- 1,213 spectra
- 400 molecules
- 1–9 spectra per molecule

---

## 2. Instrument Domain Shift

The test set is entirely:

```text
timsTOF