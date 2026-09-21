# Stage 4 — Chemistry & Adduct Handling

## Objective

Stage 4 builds the chemistry layer required for mass-aware candidate filtering.

The main goals are:

- interpret precursor adducts,
- convert precursor m/z into neutral molecular mass,
- calculate exact molecular masses,
- parse molecular formulas,
- validate formula consistency,
- quantify mass-error distributions,
- define safe candidate-filtering rules.

This stage provides the chemistry constraints that will later reduce the candidate space before retrieval and ranking.

---

# 1. RDKit Environment

RDKit was installed and validated successfully.

Version:

```text
RDKit 2026.03.6