# Stage 3 — Spectrum Preprocessing

## Objective

Stage 3 builds and validates a reusable, configurable preprocessing
pipeline for tandem mass spectra.

The goals are:

- preserve useful spectral information,
- reduce computational cost,
- prevent malformed inputs,
- support multiple experimental preprocessing profiles,
- avoid irreversible assumptions,
- make preprocessing identical during training and inference.

No preprocessing choice in this stage is considered permanently optimal.

The baseline configuration is a conservative starting point for later
retrieval and deep-learning experiments.

---

# 1. Raw Spectrum Inspection

A direct inspection was performed on:

```text
100,000 training spectra
1,213 test spectra