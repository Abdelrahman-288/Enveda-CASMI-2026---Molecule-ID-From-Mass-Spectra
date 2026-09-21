# Enveda CASMI 2026 — Molecule ID From Mass Spectra
## Master Technical Project Plan

**Project type:** Kaggle competition + research project + portfolio project  
**Primary goal:** Build a competitive, research-grade system for identifying 2D molecular structures from tandem mass spectra while learning modern mass spectrometry, cheminformatics, machine learning, deep learning, retrieval, ranking, and molecular representation learning.  
**Competition objective:** For each unknown molecule, use all available MS/MS spectra to return up to 25 ranked candidate structures represented as SMILES strings, with ranking optimized for MRR@25.  
**Developer:** Solo developer  
**Primary environment:** Windows 11, VS Code, Python 3.11, NVIDIA RTX 4070, CUDA  
**Default local Python environment:** `.venv` created with `--system-site-packages` to reuse compatible globally installed ML packages.

---

# 1. Executive Summary

The Enveda CASMI 2026 challenge asks participants to identify unknown small-molecule structures from tandem mass spectrometry data. Each test molecule may have multiple spectra collected under different adducts, collision energies, ionization modes, or acquisition conditions. The final system must aggregate evidence across all spectra belonging to the same molecule and output up to 25 ranked candidate structures.

The project will be developed in progressive versions rather than as a single monolithic model.

The central strategy is a **hybrid retrieval + constraint + representation learning + reranking system**:

```text
Multiple MS/MS spectra
        ↓
Cleaning / preprocessing
        ↓
Per-spectrum feature extraction / embedding
        ↓
Multi-spectrum aggregation
        ↓
Neutral-mass and molecular-formula constraints
        ↓
Candidate retrieval
        ↓
Spectrum–molecule compatibility scoring
        ↓
Learned reranking / ensemble
        ↓
Top 25 candidates
        ↓
Canonical SMILES
        ↓
submission.csv
```

The project begins with a simple, measurable spectral-similarity baseline and then evolves toward a research-grade architecture using learned spectrum embeddings, molecular representations, formula/mass constraints, candidate databases, multi-spectrum fusion, learned rankers, and potentially de novo methods for truly novel structures.

The project has two simultaneous outputs:

1. **Competition system**
   - Optimized for Kaggle MRR@25.
   - Must satisfy Kaggle runtime and offline inference requirements.

2. **Research/portfolio system**
   - Clean codebase.
   - Reproducible experiments.
   - Architecture documentation.
   - Ablation studies.
   - Spectrum and molecule visualizations.
   - Model comparison.
   - Error analysis.
   - Optional interactive demo.

---

# 2. Problem Statement

Modern LC-MS/MS can detect very large numbers of compounds in complex biological samples, but translating a tandem mass spectrum into a chemical structure remains difficult.

Library matching performs well when the exact compound has already been measured under sufficiently similar conditions, but fails for:

- molecules absent from spectral libraries,
- molecules present in structural databases but without reference spectra,
- natural products,
- structurally novel compounds.

The system must infer the most likely 2D structure from one or more MS/MS spectra while handling domain variation across:

- adducts,
- ionization modes,
- collision energy,
- instrument sources,
- spectral noise,
- multiple spectra per molecule.

---

# 3. Project Objectives

## 3.1 Primary objectives

- Build a valid end-to-end Kaggle submission pipeline.
- Achieve a respectable leaderboard score.
- Progress toward a competitive research-grade system.
- Learn modern MS/MS-based molecular identification.
- Learn practical cheminformatics and molecular ML.
- Produce a strong GitHub portfolio project.
- Create a reproducible project that another developer can run and understand.

## 3.2 Technical objectives

- Design reliable MS/MS preprocessing.
- Group and aggregate spectra by molecule.
- Build leakage-safe local validation.
- Implement spectral-library retrieval.
- Learn spectrum embeddings.
- Construct candidate molecule databases.
- Estimate neutral mass and molecular constraints.
- Predict or retrieve molecular formulas.
- Build molecular representations.
- Score spectrum–candidate compatibility.
- Train ranking models for top-k retrieval.
- Ensemble complementary ranking signals.
- Optimize inference for Kaggle.
- Perform error analysis and ablation studies.

---

# 4. Target Users

The competition inference system is not a consumer application; its main users are:

- the developer,
- Kaggle evaluation infrastructure,
- researchers reviewing the repository,
- potential academic supervisors,
- recruiters reviewing the portfolio.

The optional portfolio interface may target:

- students learning mass spectrometry,
- computational chemists,
- researchers,
- ML engineers.

---

# 5. Main Use Cases

## UC-001 — Competition inference

Input:

```text
One or more MS/MS spectra for a molecule
```

Output:

```text
Up to 25 ranked SMILES candidates
```

## UC-002 — Spectrum similarity search

Input:

```text
One MS/MS spectrum
```

Output:

```text
Nearest training/reference spectra and associated molecular structures
```

## UC-003 — Candidate ranking

Input:

```text
Unknown molecule spectra + candidate structures
```

Output:

```text
Compatibility score and ranked candidates
```

## UC-004 — Research experiment

Input:

```text
Model configuration
```

Output:

```text
Validation MRR@25 + supporting metrics + experiment artifacts
```

## UC-005 — Portfolio demo

Optional input:

```text
Spectrum data
```

Optional output:

```text
Spectrum visualization
Predicted constraints
Retrieved candidates
Top ranked structures
2D molecular drawings
Scores
```

---

# 6. Functional Requirements

## FR-001 — Load competition parquet files

The system shall load the training and test parquet data efficiently.

## FR-002 — Validate spectrum arrays

The system shall confirm that m/z arrays and intensity arrays have matching lengths.

## FR-003 — Preprocess spectra

The system shall support configurable:

- intensity filtering,
- m/z filtering,
- optional deisotoping,
- maximum peak count,
- intensity transforms,
- normalization.

## FR-004 — Group by molecule

The system shall group all spectra with the same `molecule_id`.

## FR-005 — Compute neutral-mass constraints

The system shall estimate neutral molecular mass using precursor m/z and adduct information where possible.

## FR-006 — Build training/reference indexes

The system shall support indexing spectra and structures for retrieval.

## FR-007 — Retrieve molecular candidates

The system shall generate candidate structures from one or more retrieval/constraint methods.

## FR-008 — Score candidates

The system shall assign a compatibility score between an unknown molecule and each candidate.

## FR-009 — Aggregate multi-spectrum evidence

The system shall combine information from multiple spectra belonging to the same molecule.

## FR-010 — Rank candidates

The system shall output candidates in descending predicted relevance.

## FR-011 — Enforce submission constraints

For every `molecule_id`:

- exactly one row,
- 1–25 candidates,
- semicolon-separated SMILES,
- no null values.

## FR-012 — Canonicalize structures

The system shall canonicalize and validate SMILES using RDKit.

## FR-013 — Compute local MRR@25

The project shall contain an exact local implementation of the evaluation metric.

## FR-014 — Record experiments

The project shall record model/configuration/result metadata.

## FR-015 — Save and resume training

Long-running training jobs shall support checkpoints.

## FR-016 — Offline Kaggle inference

The final notebook shall run without Internet access.

---

# 7. Non-Functional Requirements

## NFR-001 — Reproducibility

Experiments shall use controlled seeds, configuration files, versioned code, and documented dependencies.

## NFR-002 — Modularity

Preprocessing, retrieval, model training, ranking, evaluation, and submission generation shall remain separate modules.

## NFR-003 — GPU utilization

Compatible deep-learning workloads shall use CUDA.

## NFR-004 — Reliability

Long training jobs shall save checkpoints and logs.

## NFR-005 — Scalability

The pipeline shall support millions of spectra without loading unnecessary data repeatedly.

## NFR-006 — Kaggle runtime compliance

Final inference shall stay within the competition's notebook runtime restrictions.

## NFR-007 — Maintainability

The project shall avoid unnecessary complexity and large monolithic scripts.

## NFR-008 — Traceability

Every major experiment should be traceable to:

```text
configuration
→ code version
→ checkpoint
→ validation score
```

---

# 8. Scope

## 8.1 In Scope

- Competition data analysis.
- MS/MS preprocessing.
- Spectral similarity.
- Spectrum embeddings.
- Candidate retrieval.
- Molecular mass/formula filtering.
- Candidate molecule database.
- RDKit integration.
- Molecular fingerprints.
- Molecular encoders.
- Multi-spectrum fusion.
- Candidate reranking.
- MRR@25 optimization.
- Validation.
- Kaggle inference.
- Research analysis.
- Portfolio documentation.

## 8.2 Out of Scope for Initial Versions

- Building a full mass-spectrometry simulator from scratch.
- Training very large foundation models from scratch.
- Full quantum-chemistry simulation.
- Production cloud infrastructure.
- Microservices.
- Kubernetes.
- Mobile app.
- Fully autonomous de novo molecular generation in V0–V3.

These may be explored later only if they provide measurable value.

---

# 9. Success Criteria

## Minimum success

- Data pipeline works.
- Valid Kaggle submission is generated.
- Baseline validation MRR@25 established.
- GPU training works.
- Repository is reproducible.

## Strong project success

- Learned spectrum model beats classical spectral-similarity baseline.
- Formula/mass constraints improve candidate precision.
- Learned reranker improves MRR@25.
- Multi-spectrum fusion improves performance.
- Experiments and ablations are documented.

## Stretch success

- Competitive public/private leaderboard performance.
- Novel candidate-generation strategy.
- Research-style report.
- Strong de novo component for Class 3 molecules.

---

# 10. Competition Data Summary

Training data:

- approximately 2.5 million MS/MS spectra,
- approximately 275,000 unique structures,
- one row per spectrum,
- spectra from several libraries and instruments.

Test data:

- approximately 1,500 spectra,
- approximately 400 molecules,
- around 1–16 spectra per molecule,
- median around 3 spectra per molecule,
- test spectra acquired on Bruker timsTOF.

Prediction unit:

```text
molecule_id
```

not individual spectrum.

Important training fields include:

- `normalized_smiles`
- `inchikey`
- `inchikey14`
- `molecular_formula`
- `ingest_lib`
- `adduct_orig`
- `precursor_error_ppm`
- `num_peaks`

Important spectrum fields include:

- `molecule_id`
- `spectrum_id`
- `ms2_mzs`
- `ms2_normalized_intensities`
- `base_peak_intensity`
- `adduct`
- `ionization_mode`
- `instrument_type`
- `precursor_mz`
- `collision_energy_ev`
- `collision_energy_orig`
- `collision_energy_orig_units`

---

# 11. Evaluation Metric

The competition metric is:

```text
Mean Reciprocal Rank @ 25
```

For a molecule:

```text
RR = 1 / rank_of_first_correct_candidate
```

if the correct candidate is in the top 25.

Otherwise:

```text
RR = 0
```

Examples:

```text
rank 1  → 1.000
rank 2  → 0.500
rank 3  → 0.333
rank 5  → 0.200
rank 10 → 0.100
rank 25 → 0.040
```

This strongly prioritizes correct ordering of candidates.

Therefore, this is fundamentally a **ranking problem**, not merely a candidate-recall problem.

---

# 12. Evaluation Identity

Predicted and true structures are effectively compared through canonicalized 2D identity.

Evaluation emphasizes:

```text
InChIKey14
```

Therefore the important target is:

```text
2D atom connectivity
```

rather than:

- exact SMILES syntax,
- exact tautomer spelling,
- stereochemistry.

The project should nevertheless canonicalize and validate SMILES before submission.

---

# 13. Technology Stack

## Core language

```text
Python 3.11
```

## IDE

```text
Visual Studio Code
```

## GPU

```text
NVIDIA GeForce RTX 4070
CUDA
```

## Core libraries

Likely:

- PyTorch
- NumPy
- pandas
- pyarrow
- scikit-learn
- SciPy
- RDKit
- matchms
- matplotlib
- tqdm
- PyYAML or OmegaConf
- FAISS if compatible and useful
- LightGBM and/or XGBoost
- TensorBoard
- optional Weights & Biases
- optional PyTorch Geometric
- optional Hugging Face tooling depending on selected pretrained models

Large libraries must NOT be blindly reinstalled.

---

# 14. Development Environment

## Default setup

```powershell
cd "YOUR_PROJECT_FOLDER"
py -3.11 -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

Verify PyTorch/CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Verify GPU:

```powershell
nvidia-smi
```

Verify RDKit only after installation/configuration:

```powershell
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO') is not None)"
```

---

# 15. Package Installation Policy

Before installing heavy packages:

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

Do not automatically run:

```powershell
pip install torch torchvision torchaudio
```

if CUDA-enabled PyTorch already works.

Do not replace working CUDA packages unnecessarily.

Project-specific packages should be installed individually after checking availability.

---

# 16. VS Code Configuration

The preferred interpreter is:

```text
PROJECT_FOLDER\.venv\Scripts\python.exe
```

Recommended extensions:

- Python
- Pylance
- Jupyter
- GitLens
- Markdown All in One

Optional:

- Ruff
- Black formatter
- YAML
- Docker only if needed later

Recommended project settings:

```text
.vscode/
├── settings.json
├── launch.json
└── extensions.json
```

Do not create multiple environments such as:

```text
venv/
env/
.venv/
```

Only:

```text
.venv/
```

---

# 17. Proposed Folder Structure

```text
enveda-casmi-2026/
│
├── README.md
├── PROJECT_PLAN.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
│
├── .vscode/
│   ├── settings.json
│   ├── launch.json
│   └── extensions.json
│
├── config/
│   ├── data/
│   │   ├── default.yaml
│   │   └── preprocessing.yaml
│   ├── models/
│   │   ├── baseline.yaml
│   │   ├── spectrum_encoder.yaml
│   │   ├── molecule_encoder.yaml
│   │   └── reranker.yaml
│   ├── training/
│   │   └── default.yaml
│   └── experiments/
│
├── data/
│   ├── raw/
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   └── sample_submission.csv
│   ├── interim/
│   ├── processed/
│   ├── external/
│   ├── indexes/
│   └── metadata/
│
├── notebooks/
│   ├── 01_dataset_overview.ipynb
│   ├── 02_spectrum_visualization.ipynb
│   ├── 03_structure_analysis.ipynb
│   ├── 04_baseline_analysis.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/
│   └── casmi/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   ├── schema.py
│       │   ├── validation.py
│       │   ├── splits.py
│       │   └── molecule_grouping.py
│       │
│       ├── spectra/
│       │   ├── preprocessing.py
│       │   ├── transforms.py
│       │   ├── binning.py
│       │   ├── similarity.py
│       │   ├── aggregation.py
│       │   └── visualization.py
│       │
│       ├── chemistry/
│       │   ├── adducts.py
│       │   ├── neutral_mass.py
│       │   ├── formulas.py
│       │   ├── smiles.py
│       │   ├── fingerprints.py
│       │   └── candidate_filters.py
│       │
│       ├── retrieval/
│       │   ├── spectral_index.py
│       │   ├── embedding_index.py
│       │   ├── structure_index.py
│       │   └── candidate_generator.py
│       │
│       ├── models/
│       │   ├── spectrum_encoder.py
│       │   ├── molecule_encoder.py
│       │   ├── fusion.py
│       │   ├── scorer.py
│       │   └── reranker.py
│       │
│       ├── training/
│       │   ├── datasets.py
│       │   ├── samplers.py
│       │   ├── losses.py
│       │   ├── trainer.py
│       │   ├── callbacks.py
│       │   └── checkpointing.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── retrieval_metrics.py
│       │   ├── mrr.py
│       │   ├── ablations.py
│       │   └── error_analysis.py
│       │
│       ├── inference/
│       │   ├── pipeline.py
│       │   ├── candidate_ranking.py
│       │   └── submission.py
│       │
│       └── utils/
│           ├── config.py
│           ├── logging.py
│           ├── seed.py
│           ├── gpu.py
│           └── paths.py
│
├── scripts/
│   ├── inspect_data.py
│   ├── preprocess_data.py
│   ├── build_spectral_index.py
│   ├── build_structure_index.py
│   ├── train_spectrum_encoder.py
│   ├── train_scorer.py
│   ├── train_reranker.py
│   ├── evaluate.py
│   ├── generate_submission.py
│   └── validate_submission.py
│
├── models/
│   ├── checkpoints/
│   ├── pretrained/
│   └── exported/
│
├── experiments/
│   ├── registry.csv
│   ├── configs/
│   └── notes/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   ├── submissions/
│   └── reports/
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_adducts.py
│   ├── test_metrics.py
│   ├── test_submission.py
│   └── test_inference.py
│
├── kaggle/
│   ├── inference_notebook.ipynb
│   ├── assets/
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── preprocessing.md
│   ├── validation.md
│   ├── models.md
│   ├── experiments.md
│   └── kaggle_inference.md
│
└── demo/
    └── app.py
```

---

# 18. Folder Rules

## `data/raw/`

Contains original competition files.

Do not modify these files.

## `data/interim/`

Contains temporary processed forms that can be regenerated.

## `data/processed/`

Contains stable model-ready artifacts.

## `data/external/`

Contains allowed external datasets or databases.

Large external data should not be committed to Git.

## `data/indexes/`

Contains:

- FAISS indexes,
- spectral indexes,
- candidate lookup tables,
- embedding indexes.

## `models/checkpoints/`

Training checkpoints.

Normally excluded from standard Git unless small.

## `models/pretrained/`

Externally obtained pretrained models.

Track source/version/license.

## `experiments/`

Experiment metadata, configs, comparisons, notes.

## `outputs/`

Generated outputs only.

## `kaggle/`

Contains competition inference notebook and assets needed for offline execution.

---

# 19. System Architecture

```text
                   ┌─────────────────────────┐
                   │   Competition Dataset   │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │  Dataset Validation     │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Spectrum Preprocessing  │
                   └────────────┬────────────┘
                                │
                ┌───────────────┼────────────────┐
                │               │                │
                ▼               ▼                ▼
       Spectral similarity  Spectrum encoder  Chemistry constraints
                │               │                │
                └───────────────┼────────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │ Candidate Generation    │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Candidate Feature Build │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Spectrum–Molecule Score │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Learned Reranker        │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Ensemble / Calibration  │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Top 25 Structures       │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ submission.csv          │
                   └─────────────────────────┘
```

---

# 20. Data Pipeline

```text
train.parquet / test.parquet
        ↓
schema validation
        ↓
peak-array validation
        ↓
clean spectra
        ↓
normalize / transform intensity
        ↓
remove invalid m/z
        ↓
optional weak-peak filtering
        ↓
optional deisotoping
        ↓
peak count limiting
        ↓
derive chemistry metadata
        ↓
group by molecule
        ↓
model-ready records
```

Every preprocessing decision must be configurable.

Do not hardcode one irreversible transformation.

---

# 21. Spectrum Preprocessing Strategy

Candidate preprocessing operations:

## 21.1 Relative intensity filtering

Initial thresholds to test:

```text
0.1%
1%
2%
```

These are tunable experimental values.

## 21.2 Remove peaks above precursor

Approximate rule:

```text
m/z > precursor_mz + tolerance
```

Possible starting tolerance:

```text
1–2 Da
```

## 21.3 Minimum informative spectrum

Potentially flag spectra with very few peaks.

Do not discard until validated experimentally.

## 21.4 Maximum peak count

Initial experiment:

```text
top 128 peaks
```

Also test:

```text
64
128
256
all
```

## 21.5 Intensity transform

Compare:

```text
identity
sqrt(intensity)
log1p(intensity)
```

## 21.6 Normalization

Possible options:

- base peak = 1,
- L1 normalization,
- L2 normalization.

## 21.7 Deisotoping

Implement as optional.

Never assume it improves every model.

---

# 22. Adduct Handling

Test adducts include forms such as:

```text
[M+H]+
[M+NH4]+
[M-H2O+H]+
[M-2H2O+H]+
[M+Na]+
[M+K]+
[M-H]-
[M-H2O-H]-
[M+CH2O2-H]-
[M+Cl]-
```

Create a centralized adduct module:

```text
src/casmi/chemistry/adducts.py
```

Responsibilities:

- normalize adduct naming,
- determine charge,
- determine neutral-mass shift,
- calculate approximate neutral mass,
- validate ionization mode compatibility.

Do not duplicate adduct logic across scripts.

---

# 23. Neutral Mass Estimation

For each spectrum:

```text
precursor_mz + adduct
        ↓
estimated neutral mass
```

For molecules with multiple spectra:

- estimate neutral mass independently,
- compare estimates,
- identify inconsistent spectra,
- aggregate robustly.

Potential aggregation:

```text
median neutral mass
```

or weighted estimate based on confidence.

Mass tolerance shall be validated empirically.

---

# 24. Validation Strategy

This is one of the most important parts of the project.

Random spectrum splitting is unacceptable because spectra from the same molecule could appear in training and validation.

The default split unit must be:

```text
unique molecular structure / InChIKey14
```

not spectrum.

## 24.1 Primary validation

Use grouped splits based on:

```text
inchikey14
```

This prevents exact structure leakage.

## 24.2 Domain-aware validation

Because test spectra come from timsTOF, create additional validation scenarios such as:

- hold out selected libraries,
- emphasize timsTOF-like spectra,
- hold out natural-product-heavy structures.

## 24.3 Novelty simulations

Construct validation subsets that approximate:

### Class 1

Exact structure has reference spectra in training/reference index.

### Class 2

Structure exists in candidate database but reference spectra are removed.

### Class 3

Exact structure removed from candidate database.

This gives three realistic diagnostics.

---

# 25. Baseline V0 — Classical Spectral Retrieval

## Objective

Create a valid and measurable end-to-end solution as quickly as possible.

Pipeline:

```text
query spectrum
    ↓
preprocess
    ↓
compare to reference spectra
    ↓
similarity score
    ↓
retrieve top structures
    ↓
aggregate multiple query spectra
    ↓
rank structures
    ↓
top 25
```

Candidate similarities:

- cosine similarity on binned spectra,
- modified cosine,
- matchms similarity functions.

## Expected value

V0 establishes:

- submission generator,
- local MRR implementation,
- retrieval infrastructure,
- multi-spectrum aggregation baseline,
- debugging reference.

V0 must be completed before advanced models.

---

# 26. Baseline V0.5 — Chemistry-Constrained Retrieval

Add:

- neutral mass filtering,
- adduct-aware filtering,
- precursor tolerance,
- optional formula constraints if available.

Pipeline:

```text
spectral candidates
      +
mass-compatible candidates
      ↓
filtered candidate list
      ↓
rerank
```

Compare V0 versus V0.5.

---

# 27. Version V1 — Learned Spectrum Embeddings

Replace or complement hand-engineered spectral similarity with a learned spectrum representation.

Goal:

```text
MS/MS spectrum
   ↓
encoder
   ↓
dense embedding
```

Similar chemical structures should ideally produce related representations.

Possible model families to investigate:

- transformer spectrum encoders,
- peak-set transformers,
- pretrained MS/MS models,
- DreaMS-style embeddings,
- contrastive encoders.

Do not select a final pretrained model without verifying:

- public availability,
- competition compliance,
- license,
- inference cost,
- timsTOF compatibility,
- embedding quality.

---

# 28. Spectrum Encoder Inputs

Potential model inputs:

```text
peak m/z
peak intensity
precursor m/z
adduct
ionization mode
collision energy
instrument metadata
```

Potential representation:

```text
peak tokens
+
continuous m/z encoding
+
intensity projection
+
metadata embeddings
```

---

# 29. Version V2 — Candidate Structure Database

Build a searchable structure database from:

1. competition training structures,
2. allowed external molecules.

Candidate external sources to investigate:

- PubChem,
- COCONUT,
- GNPS-related structure sources,
- MassBank-associated structures.

For each candidate, store:

```text
canonical_smiles
inchikey
inchikey14
molecular_formula
exact_mass
molecular_weight
fingerprint
source
```

Potential additional fields:

```text
molecular_embedding
substructure descriptors
element counts
DBE
```

---

# 30. Candidate Database Deduplication

Deduplicate primarily by:

```text
InChIKey14
```

because the competition focuses on 2D structure identity.

Maintain mapping from:

```text
InChIKey14
→ representative canonical SMILES
```

Candidate provenance must also be retained.

---

# 31. Version V3 — Formula / Constraint Prediction

Use precursor mass and fragmentation to constrain candidate space.

Potential approaches:

- exact mass enumeration,
- known formula lookup,
- learned formula prediction,
- external tools such as MIST-CF or SIRIUS if compatible with competition usage.

Candidate formula signals:

```text
formula exact match
formula mass error
elemental plausibility
isotope clues
fragment compatibility
```

Formula prediction should primarily improve **candidate recall at low candidate counts**.

---

# 32. Version V4 — Molecular Representation

Molecule representations to compare:

## Baseline

Morgan fingerprints.

## Additional descriptors

- molecular weight,
- elemental counts,
- ring counts,
- HBD/HBA,
- TPSA,
- logP,
- fragment descriptors.

## Learned representation

Potential:

- molecular graph GNN,
- pretrained molecular encoder,
- graph transformer.

Start simple.

Morgan fingerprints should be implemented before a GNN.

---

# 33. Version V5 — Spectrum–Molecule Compatibility Model

Goal:

```text
unknown spectrum representation
+
candidate molecule representation
        ↓
compatibility score
```

Possible architecture:

```text
Spectrum Encoder ─────┐
                      ├→ Fusion → MLP → score
Molecule Encoder ─────┘
```

Alternative:

```text
spectrum embedding
      ↕
contrastive embedding space
      ↕
molecule embedding
```

Possible training objectives:

- binary compatibility classification,
- contrastive loss,
- triplet loss,
- InfoNCE,
- pairwise ranking loss.

Selection must be based on validation.

---

# 34. Negative Sampling

Negative candidate quality is critical.

Use a mixture of:

### Easy negatives

Random unrelated molecules.

### Mass-matched negatives

Similar molecular mass.

### Formula-matched negatives

Same or similar formula.

### Retrieval-hard negatives

Candidates already scored highly by baseline retrieval.

### Structural-neighbor negatives

Similar fingerprints but wrong InChIKey14.

Hard negatives should become increasingly important in later training.

---

# 35. Version V6 — Multi-Spectrum Aggregation

A molecule may have multiple spectra.

Do not simply treat them as independent predictions.

Compare:

## Score pooling

```text
max score
mean score
weighted mean
```

## Embedding pooling

```text
mean embedding
max pooling
attention pooling
```

## Learned set encoder

```text
Spectrum 1 ─┐
Spectrum 2 ─┤
Spectrum 3 ─┼→ Set/Attention Encoder → Molecule Representation
Spectrum N ─┘
```

Metadata-aware fusion can condition on:

- collision energy,
- adduct,
- ionization mode.

---

# 36. Version V7 — Learned Reranker

The reranker receives candidates produced by retrieval.

Potential features:

```text
spectral similarity
embedding similarity
mass error
formula compatibility
adduct compatibility
fingerprint score
neural compatibility score
number of supporting spectra
best spectrum score
mean spectrum score
collision-energy consistency
candidate prior
source indicator
```

Initial model:

```text
LightGBM / XGBoost
```

Later:

```text
neural ranking network
```

Ranking objectives to evaluate:

- pointwise classification,
- pairwise ranking,
- LambdaRank,
- listwise ranking.

Primary metric remains:

```text
MRR@25
```

---

# 37. Version V8 — Ensemble

Potential final score:

```text
final_score =
    w1 * spectral_similarity
  + w2 * spectrum_embedding_similarity
  + w3 * chemistry_constraint_score
  + w4 * neural_compatibility
  + w5 * learned_reranker_score
```

Weights must be learned or tuned on validation.

Do not choose weights arbitrarily.

Potential rank aggregation:

- weighted score fusion,
- reciprocal rank fusion,
- rank averaging,
- learned blending.

---

# 38. Version V9 — De Novo / Novel-Structure Research

This is a stretch research stage.

Purpose:

Improve performance on molecules whose exact structure is absent from candidate databases.

Possible research directions:

- fingerprint prediction from spectra,
- molecular substructure prediction,
- graph generation conditioned on spectra,
- fragmentation-aware molecular generation,
- spectrum-conditioned language models for SMILES,
- candidate mutation around structurally similar retrieved molecules.

This stage must not delay the stronger retrieval/ranking system.

---

# 39. GPU Training Strategy

Use:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Verify model placement:

```python
print(next(model.parameters()).device)
```

Monitor:

```powershell
nvidia-smi
```

Potential optimizations:

- automatic mixed precision,
- pinned memory,
- non-blocking transfers,
- gradient accumulation,
- persistent DataLoader workers,
- checkpointing,
- batch-size tuning.

Initial DataLoader options to test:

```python
pin_memory=True
persistent_workers=True
```

Actual `num_workers` must be benchmarked on Windows.

---

# 40. Mixed Precision

Use PyTorch AMP when numerically safe.

Typical pattern:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

Use appropriate gradient scaling depending on the final PyTorch API/version.

Benefits:

- lower VRAM usage,
- larger batch size,
- faster training on compatible GPUs.

Do not use AMP blindly if instability appears.

---

# 41. Long-Running Training Requirements

Because the machine may run overnight, every training script should support:

```text
checkpoint saving
resume training
best-model saving
periodic validation
CSV/JSON logging
TensorBoard logging
exception-safe checkpointing
```

Recommended checkpoint fields:

```text
epoch
global_step
model_state_dict
optimizer_state_dict
scheduler_state_dict
scaler_state_dict
best_metric
config
random_seed
```

---

# 42. Experiment Tracking

Minimum experiment registry columns:

```text
experiment_id
date
git_commit
model
data_version
preprocessing
split
seed
hyperparameters
checkpoint
MRR@25
Recall@1
Recall@5
Recall@10
Recall@25
training_time
peak_gpu_memory
notes
```

Initial tracking method:

```text
CSV + YAML configs + TensorBoard
```

Optional later:

```text
Weights & Biases
```

---

# 43. Metrics

Primary:

```text
MRR@25
```

Supporting:

```text
Recall@1
Recall@5
Recall@10
Recall@25
Candidate recall before reranking
Mean candidate count
Median true-candidate rank
```

For formula prediction:

```text
formula top-1 accuracy
formula top-k recall
```

For retrieval:

```text
structure Recall@K
```

For performance:

```text
inference time per molecule
GPU memory usage
index search time
```

---

# 44. Error Analysis

Create error categories:

```text
A — true structure absent from candidate set
B — true structure retrieved but ranked low
C — neutral-mass estimate incorrect/inconsistent
D — adduct handling error
E — poor spectrum quality
F — domain shift
G — structurally similar isomer confusion
H — multi-spectrum fusion failure
I — invalid/generated structure
```

For each failed molecule, store:

- true structure,
- best predicted structures,
- ranks,
- spectrum plots,
- precursor/adduct metadata,
- candidate-generation source,
- component scores.

---

# 45. Ablation Studies

Required research ablations:

## A1 — Preprocessing

Compare intensity thresholds.

## A2 — Peak count

```text
64 vs 128 vs 256 vs all
```

## A3 — Intensity transform

```text
raw vs sqrt vs log1p
```

## A4 — Collision energy

With versus without collision energy.

## A5 — Adduct encoding

With versus without adduct.

## A6 — Multi-spectrum fusion

Single best spectrum versus pooled spectra.

## A7 — Neutral mass filtering

Enabled versus disabled.

## A8 — Formula filtering

Enabled versus disabled.

## A9 — Learned spectrum embedding

Versus classical spectral similarity.

## A10 — Reranking

Before versus after learned reranking.

## A11 — Molecule representation

Morgan versus learned molecular encoder.

---

# 46. Kaggle Inference Architecture

Final Kaggle notebook should perform:

```text
load assets
↓
load test.parquet
↓
preprocess spectra
↓
group by molecule_id
↓
encode spectra
↓
retrieve candidates
↓
apply chemistry constraints
↓
score candidates
↓
rerank
↓
top 25
↓
validate SMILES
↓
write submission.csv
```

Heavy training should not occur in the final inference notebook.

---

# 47. Kaggle Runtime Optimization

Because inference time is limited:

- precompute candidate embeddings,
- precompute training/reference embeddings,
- store compact indexes,
- avoid repeated RDKit work,
- cache candidate descriptors,
- use vectorized operations,
- batch neural inference,
- limit candidate count before expensive reranking,
- measure cold-start overhead,
- avoid unnecessary plotting.

Target:

```text
offline inference comfortably below Kaggle runtime limit
```

not barely at the limit.

---

# 48. Submission Validation

Create:

```text
scripts/validate_submission.py
```

Checks:

```text
columns exactly valid
all molecule IDs present
no duplicate molecule IDs
no nulls
1–25 candidate SMILES
semicolon formatting correct
SMILES valid
no accidental blank candidates
correct number of rows
```

Always run validation before Kaggle submission.

---

# 49. Testing Strategy

## Unit tests

### Data

- spectrum arrays aligned,
- grouping correct,
- schema validated.

### Chemistry

- adduct parsing,
- neutral mass calculations,
- SMILES canonicalization,
- InChIKey14 conversion.

### Metrics

- MRR known examples,
- Recall@K.

### Submission

- valid/invalid row cases.

## Integration tests

Test:

```text
small parquet subset
→ preprocessing
→ retrieval
→ ranking
→ submission
```

## Model tests

- output shapes,
- gradients finite,
- GPU placement,
- checkpoint load/save.

## Reproducibility tests

Same seed/config should give near-consistent results.

---

# 50. Git Strategy

Solo project: avoid unnecessary branching complexity.

Recommended:

```text
main
feature/*
experiment/*
fix/*
```

`main` should remain runnable.

Commit examples:

```text
chore: initialize project structure
feat: implement parquet data loader
feat: add spectrum preprocessing pipeline
feat: implement grouped validation split
feat: add spectral similarity baseline
feat: implement local MRR@25 metric
feat: add candidate retrieval index
feat: train spectrum encoder
feat: add candidate reranker
feat: generate valid Kaggle submission
```

---

# 51. `.gitignore`

At minimum:

```gitignore
.venv/
__pycache__/
*.pyc
.env

data/raw/*
data/interim/*
data/processed/*
data/external/*
data/indexes/*

models/checkpoints/*
models/pretrained/*
outputs/*
runs/
wandb/

.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

Use placeholder `.gitkeep` files where desired.

Do not upload the competition dataset to GitHub.

---

# 52. Documentation Strategy

Required:

```text
README.md
PROJECT_PLAN.md
docs/architecture.md
docs/dataset.md
docs/preprocessing.md
docs/validation.md
docs/models.md
docs/experiments.md
docs/kaggle_inference.md
```

---

# 53. Final README Structure

```text
1. Project title
2. Overview
3. Competition problem
4. Approach
5. Architecture
6. Dataset
7. Environment
8. Installation
9. CUDA verification
10. Project structure
11. Preprocessing
12. Models
13. Validation strategy
14. Training
15. Inference
16. Results
17. Ablations
18. Kaggle submission
19. Reproducibility
20. Future work
21. References
22. License
```

---

# 54. Project Stages

---

## Stage 0 — Project Initialization

### Objective

Create a clean and reproducible repository.

### Tasks

- S0.1 Create project directory.
- S0.2 Create `.venv`.
- S0.3 Verify Python.
- S0.4 Verify CUDA.
- S0.5 Initialize Git.
- S0.6 Create folder structure.
- S0.7 Create `.gitignore`.
- S0.8 Create `README.md`.
- S0.9 Add this `PROJECT_PLAN.md`.

### Verification

```powershell
python --version
python -c "import torch; print(torch.cuda.is_available())"
git status
```

### Deliverable

Runnable project skeleton.

### Git milestone

```text
chore: initialize Enveda CASMI project
```

---

## Stage 1 — Dataset Acquisition and Integrity

### Objective

Prepare competition data safely.

### Tasks

- S1.1 Download Kaggle files.
- S1.2 Store under `data/raw/`.
- S1.3 Verify filenames.
- S1.4 Check file sizes.
- S1.5 Inspect parquet schemas.
- S1.6 Validate array consistency.
- S1.7 Record dataset metadata.

### Deliverable

Validated raw dataset.

### Git milestone

```text
feat: add dataset validation tooling
```

---

## Stage 2 — Exploratory Data Analysis

### Objective

Understand the data before modeling.

### Tasks

- S2.1 Count spectra.
- S2.2 Count unique molecules.
- S2.3 Analyze spectra-per-molecule.
- S2.4 Analyze adduct distribution.
- S2.5 Analyze collision energy.
- S2.6 Analyze precursor mass.
- S2.7 Analyze peak count.
- S2.8 Analyze library sources.
- S2.9 Inspect molecular formulas.
- S2.10 Inspect common scaffolds.
- S2.11 Visualize spectra.
- S2.12 Compare train/test metadata distributions.

### Deliverable

EDA notebook and report.

### Git milestone

```text
analysis: complete dataset exploratory analysis
```

---

## Stage 3 — Preprocessing Pipeline

### Objective

Implement configurable and testable spectrum preprocessing.

### Tasks

- S3.1 Implement normalization.
- S3.2 Implement relative-intensity filtering.
- S3.3 Implement precursor cutoff.
- S3.4 Implement top-k peak limiting.
- S3.5 Implement intensity transforms.
- S3.6 Add optional deisotoping.
- S3.7 Add unit tests.
- S3.8 Benchmark preprocessing throughput.

### Deliverable

`src/casmi/spectra/preprocessing.py`

### Git milestone

```text
feat: implement configurable spectrum preprocessing
```

---

## Stage 4 — Chemistry Utilities

### Objective

Build chemistry primitives used by all later models.

### Tasks

- S4.1 Adduct normalization.
- S4.2 Charge parsing.
- S4.3 Neutral mass calculation.
- S4.4 SMILES validation.
- S4.5 SMILES canonicalization.
- S4.6 InChIKey/InChIKey14 generation.
- S4.7 Morgan fingerprint generation.
- S4.8 Formula parsing.
- S4.9 Unit tests.

### Deliverable

Chemistry utility module.

### Git milestone

```text
feat: add RDKit chemistry utilities
```

---

## Stage 5 — Leakage-Safe Validation

### Objective

Create reliable offline evaluation.

### Tasks

- S5.1 Group structures by InChIKey14.
- S5.2 Create grouped train/validation split.
- S5.3 Build timsTOF-focused validation subset.
- S5.4 Build Class-1-like subset.
- S5.5 Build Class-2-like subset.
- S5.6 Build Class-3-like research subset.
- S5.7 Save split manifests.
- S5.8 Implement MRR@25.
- S5.9 Implement Recall@K.

### Deliverable

Versioned validation split.

### Git milestone

```text
feat: add leakage-safe validation and MRR@25
```

---

## Stage 6 — V0 Spectral Retrieval Baseline

### Objective

Produce the first complete valid prediction system.

### Tasks

- S6.1 Implement spectrum representation.
- S6.2 Implement similarity.
- S6.3 Build reference index.
- S6.4 Retrieve nearest spectra.
- S6.5 Map spectra to structures.
- S6.6 Deduplicate by InChIKey14.
- S6.7 Aggregate multi-spectrum scores.
- S6.8 Produce top 25.
- S6.9 Evaluate locally.
- S6.10 Generate first Kaggle submission.

### Deliverable

V0 leaderboard baseline.

### Git milestone

```text
feat: implement spectral retrieval baseline
```

---

## Stage 7 — Chemistry-Constrained Baseline

### Objective

Improve retrieval with neutral mass/adduct information.

### Tasks

- S7.1 Compute neutral mass.
- S7.2 Build structure mass index.
- S7.3 Apply mass tolerance.
- S7.4 Compare candidate recall.
- S7.5 Tune tolerance.
- S7.6 Evaluate MRR improvement.

### Deliverable

V0.5.

### Git milestone

```text
feat: add mass-constrained candidate retrieval
```

---

## Stage 8 — Pretrained Spectrum Model Research

### Objective

Identify suitable modern spectrum encoders.

### Tasks

- S8.1 Review public pretrained spectrum models.
- S8.2 Verify competition eligibility.
- S8.3 Check input format.
- S8.4 Benchmark embeddings.
- S8.5 Compare nearest-neighbor retrieval.
- S8.6 Select best candidate encoder.

### Deliverable

Model-selection report.

### Git milestone

```text
research: benchmark pretrained spectrum encoders
```

---

## Stage 9 — Spectrum Embedding Retrieval

### Objective

Replace/augment classical similarity.

### Tasks

- S9.1 Build embedding pipeline.
- S9.2 Embed training spectra.
- S9.3 Build ANN index.
- S9.4 Embed validation/test spectra.
- S9.5 Retrieve neighbors.
- S9.6 Aggregate by molecule.
- S9.7 Evaluate Recall@K and MRR.
- S9.8 Ensemble with classical similarity.

### Deliverable

V1.

### Git milestone

```text
feat: add learned spectrum embedding retrieval
```

---

## Stage 10 — Candidate Structure Database

### Objective

Build a chemistry-aware search space.

### Tasks

- S10.1 Extract unique train structures.
- S10.2 Canonicalize.
- S10.3 Deduplicate by InChIKey14.
- S10.4 Compute exact masses.
- S10.5 Compute formulas.
- S10.6 Compute Morgan fingerprints.
- S10.7 Investigate external databases.
- S10.8 Import allowed external candidates.
- S10.9 Record provenance.

### Deliverable

Candidate database.

### Git milestone

```text
feat: build molecular candidate database
```

---

## Stage 11 — Formula / Constraint Modeling

### Objective

Use molecular-formula information to reduce candidate space.

### Tasks

- S11.1 Establish formula baseline.
- S11.2 Evaluate exact-mass formula lookup.
- S11.3 Investigate MIST-CF/SIRIUS.
- S11.4 Build formula candidate set.
- S11.5 Measure formula top-k recall.
- S11.6 Integrate with candidate retrieval.

### Deliverable

V2 candidate generator.

### Git milestone

```text
feat: add molecular formula constraints
```

---

## Stage 12 — Molecular Representation

### Objective

Create candidate-side features.

### Tasks

- S12.1 Morgan fingerprints.
- S12.2 RDKit descriptors.
- S12.3 Molecular graph dataset.
- S12.4 Benchmark optional GNN.
- S12.5 Precompute embeddings.

### Deliverable

Molecule representation library.

### Git milestone

```text
feat: add molecular structure representations
```

---

## Stage 13 — Spectrum–Molecule Scoring Model

### Objective

Learn compatibility between spectra and molecular candidates.

### Tasks

- S13.1 Construct positive pairs.
- S13.2 Construct easy negatives.
- S13.3 Add mass-matched negatives.
- S13.4 Add retrieval hard negatives.
- S13.5 Implement scorer.
- S13.6 Train on GPU.
- S13.7 Evaluate ranking performance.
- S13.8 Save best checkpoint.

### Deliverable

V3 neural scorer.

### Git milestone

```text
feat: train spectrum-molecule compatibility model
```

---

## Stage 14 — Multi-Spectrum Fusion

### Objective

Exploit all spectra per molecule.

### Tasks

- S14.1 Score-level max pooling.
- S14.2 Mean pooling.
- S14.3 Metadata-weighted pooling.
- S14.4 Embedding pooling.
- S14.5 Attention pooling.
- S14.6 Compare MRR.

### Deliverable

V4 multi-spectrum model.

### Git milestone

```text
feat: add multi-spectrum evidence fusion
```

---

## Stage 15 — Learned Reranking

### Objective

Optimize the ordering of candidates.

### Tasks

- S15.1 Build candidate-level features.
- S15.2 Train LightGBM/XGBoost ranker.
- S15.3 Compare pointwise/pairwise/listwise.
- S15.4 Add neural score.
- S15.5 Tune candidate cutoff.
- S15.6 Measure MRR@25.

### Deliverable

V5 reranker.

### Git milestone

```text
feat: add learned candidate reranking
```

---

## Stage 16 — Ensemble

### Objective

Combine complementary systems.

### Tasks

- S16.1 Generate component predictions.
- S16.2 Analyze score calibration.
- S16.3 Test weighted blending.
- S16.4 Test reciprocal rank fusion.
- S16.5 Tune on validation.
- S16.6 Select robust ensemble.

### Deliverable

V6 ensemble.

### Git milestone

```text
feat: ensemble retrieval and ranking models
```

---

## Stage 17 — Error Analysis and Ablations

### Objective

Understand why the system fails.

### Tasks

- S17.1 Categorize errors.
- S17.2 Analyze missing-candidate errors.
- S17.3 Analyze ranking errors.
- S17.4 Analyze domain shift.
- S17.5 Run required ablations.
- S17.6 Document findings.

### Deliverable

Research report.

### Git milestone

```text
analysis: complete model ablations and error analysis
```

---

## Stage 18 — Kaggle Optimization

### Objective

Make final inference fast and offline-safe.

### Tasks

- S18.1 Freeze models.
- S18.2 Precompute embeddings.
- S18.3 Export indexes.
- S18.4 Measure total runtime.
- S18.5 Reduce unnecessary CPU work.
- S18.6 Test offline notebook.
- S18.7 Validate assets.
- S18.8 Generate submission.

### Deliverable

Competition-ready inference notebook.

### Git milestone

```text
perf: optimize offline Kaggle inference
```

---

## Stage 19 — De Novo Research

### Objective

Explore truly novel-structure prediction.

### Tasks

- S19.1 Fingerprint prediction.
- S19.2 Substructure prediction.
- S19.3 Candidate mutation.
- S19.4 Investigate graph generation.
- S19.5 Evaluate Class-3-like validation.

### Deliverable

Experimental research branch.

### Git milestone

```text
research: explore de novo structure prediction
```

---

## Stage 20 — Portfolio Demo

### Objective

Create a visual demonstration of the system.

Potential UI:

```text
Spectrum plot
Predicted neutral mass
Predicted formula candidates
Retrieved structures
Top ranked molecules
2D molecule drawings
Component scores
```

Use Streamlit only if useful.

### Deliverable

Optional demo.

---

## Stage 21 — Final Documentation and Release

### Objective

Make repository publication-ready.

### Tasks

- S21.1 Final README.
- S21.2 Architecture diagram.
- S21.3 Model documentation.
- S21.4 Reproduction guide.
- S21.5 Results table.
- S21.6 Ablation results.
- S21.7 Screenshots.
- S21.8 Final release tag.

### Deliverable

Portfolio-ready V1.0.

---

# 55. Stage Dependency Graph

```text
Stage 0
  ↓
Stage 1
  ↓
Stage 2
  ↓
Stage 3
  ↓
Stage 4
  ↓
Stage 5
  ↓
Stage 6
  ↓
Stage 7
  ├──────────────→ Stage 10
  │                   ↓
  │                Stage 11
  │                   ↓
  └→ Stage 8 → Stage 9
                  │
                  └──────────┐
                             ↓
                         Stage 12
                             ↓
                         Stage 13
                             ↓
                         Stage 14
                             ↓
                         Stage 15
                             ↓
                         Stage 16
                             ↓
                         Stage 17
                             ↓
                         Stage 18
                             ↓
                 ┌───────────┴───────────┐
                 ↓                       ↓
             Stage 19                Stage 20
                 └───────────┬───────────┘
                             ↓
                         Stage 21
```

---

# 56. Development Versions

```text
V0
Classical spectral retrieval

V0.5
Spectral retrieval + chemistry constraints

V1
Learned spectrum embedding retrieval

V2
Candidate DB + formula/mass-constrained retrieval

V3
Spectrum–molecule compatibility model

V4
Multi-spectrum aggregation

V5
Learned reranker

V6
Ensemble

V7
Kaggle runtime optimized system

V8
Research ablations / analysis

V9
De novo experiments

V1.0
Final competition + portfolio release
```

---

# 57. Risk Register

| Risk | Impact | Probability | Mitigation |
|---|---|---:|---|
| Validation leakage | Critical | Medium | Split by InChIKey14, never random spectrum only |
| True structure absent from candidate DB | High | High | External candidate DB + de novo research |
| Candidate generation recall too low | Critical | Medium | Separate recall optimization from ranking |
| Domain shift to timsTOF | High | Medium | Domain-aware validation, use similar training sources |
| GPU VRAM limitations | Medium | Medium | AMP, gradient accumulation, smaller batches |
| CUDA dependency conflict | High | Low–Medium | Reuse current environment; pin compatible versions |
| Dataset preprocessing too slow | Medium | Medium | Vectorization, caching, parquet column projection |
| Retrieval index too large | Medium | Medium | Compact embeddings, FAISS, chunking |
| RDKit bottleneck | Medium | Medium | Precompute descriptors/structures |
| Overfitting leaderboard | High | Medium | Stable local validation and limited leaderboard tuning |
| Formula predictor fails on novel structures | Medium | Medium | Keep formula as one signal, not single point of failure |
| Pretrained model license/competition restriction | High | Low–Medium | Verify before integration |
| Overnight training interruption | Medium | Medium | Robust checkpoints and resume |
| Overengineering | Medium | Medium | Gate each stage on measurable improvement |
| Kaggle offline assets missing | High | Medium | Full offline dry run |
| Inference exceeds 9h | Critical | Medium | Early runtime benchmarking |

---

# 58. What Must Be Learned Before Starting

## 58.1 MS/MS fundamentals

Understand:

- precursor ion,
- product ions,
- collision energy,
- positive/negative ion mode,
- adducts,
- neutral mass,
- fragment peaks,
- isotopes.

## 58.2 Cheminformatics fundamentals

Understand:

- SMILES,
- canonical SMILES,
- InChIKey,
- InChIKey14,
- molecular formula,
- exact mass,
- Morgan fingerprint,
- molecular graphs,
- structural similarity.

## 58.3 Ranking/retrieval fundamentals

Understand:

- nearest-neighbor search,
- candidate recall,
- MRR,
- Recall@K,
- reranking,
- hard negatives.

---

# 59. Learn While Building

- RDKit.
- matchms.
- pretrained spectrum encoders.
- FAISS or alternative ANN.
- molecular fingerprints.
- GNNs.
- contrastive learning.
- learning-to-rank.
- multi-instance / set encoding.
- formula prediction.
- experiment tracking.

---

# 60. Optional Advanced Learning

- fragmentation trees,
- probabilistic formula scoring,
- graph transformers,
- spectrum-conditioned molecular generation,
- energy-aware fragmentation modeling,
- self-supervised spectrum pretraining,
- cross-modal spectrum–molecule pretraining.

---

# 61. Research Questions

Potential final report questions:

## RQ1

How much does adduct-aware neutral-mass filtering improve candidate recall and MRR?

## RQ2

Do learned spectrum embeddings outperform classical spectral similarity?

## RQ3

How much does multi-spectrum aggregation improve structure identification?

## RQ4

Does collision-energy conditioning improve ranking?

## RQ5

Do molecular graph embeddings outperform Morgan fingerprints for candidate scoring?

## RQ6

How much improvement comes from learned reranking?

## RQ7

Which parts of the pipeline fail most often on structure-novel molecules?

---

# 62. Decision Log

| ID | Decision | Reason | Status |
|---|---|---|---|
| D001 | Treat as competition + research + portfolio project | Matches project goals | Confirmed |
| D002 | Work solo | User requirement | Confirmed |
| D003 | Aim for research-grade architecture | Competitive and learning goal | Confirmed |
| D004 | Use baseline-first progression | Reduces risk and enables measurable improvement | Confirmed |
| D005 | Use Windows + VS Code | Primary environment | Confirmed |
| D006 | Use Python 3.11 where compatible | Standard local workflow | Confirmed |
| D007 | Use `.venv --system-site-packages` | Reuse installed packages | Confirmed |
| D008 | Use RTX 4070/CUDA | Faster DL training | Confirmed |
| D009 | Do not reinstall CUDA PyTorch unnecessarily | Avoid dependency breakage | Confirmed |
| D010 | Use molecule-level / structure-level validation | Avoid leakage | Confirmed |
| D011 | Optimize for MRR@25 | Competition metric | Confirmed |
| D012 | Start with spectral retrieval baseline | Simple and measurable | Confirmed |
| D013 | Add chemistry constraints | Narrow candidate space | Confirmed |
| D014 | Add learned spectrum embeddings | Improve retrieval quality | Planned |
| D015 | Build candidate structure DB | Needed for Class 2/3 strategy | Planned |
| D016 | Add learned reranker | Metric rewards ordering | Planned |
| D017 | Investigate de novo generation only after strong retrieval system | Avoid premature complexity | Confirmed |
| D018 | Training may run overnight | Developer availability | Confirmed |
| D019 | Every long job must checkpoint | Reliability requirement | Confirmed |
| D020 | Final Kaggle notebook shall be inference-focused | Runtime requirement | Confirmed |

---

# 63. Assumptions

## Assumption A001 — needs confirmation

The local RTX 4070 VRAM size is sufficient for medium-sized spectrum encoders using careful batch sizing.

## Assumption A002 — needs confirmation

Allowed external datasets/models will be used when they materially improve the system.

## Assumption A003 — needs confirmation

The user will download competition data locally before implementation begins.

## Assumption A004 — needs confirmation

The project folder will reside on a drive with enough free space for:

- 3+ GB raw competition data,
- processed artifacts,
- embeddings,
- external structures,
- model checkpoints.

A practical target is substantially more than 20 GB free if large external resources are used.

---

# 64. Reproducibility Strategy

Track:

```text
Python version
PyTorch version
CUDA version
GPU
RDKit version
dataset version
config
seed
Git commit
checkpoint
```

Set seeds where appropriate:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

For exact reproducibility, document any nondeterministic CUDA behavior.

---

# 65. Deployment Strategy

Primary deployment:

```text
Kaggle inference notebook
```

Secondary distribution:

```text
GitHub repository
```

Optional:

```text
Streamlit research demo
```

The final project must not depend on the developer's global packages.

Although local development uses:

```text
--system-site-packages
```

the repository must still document exact dependencies.

---

# 66. Definition of Done

The project is considered complete when:

- [ ] repository works from a clean clone,
- [ ] setup instructions are documented,
- [ ] CUDA verification is documented,
- [ ] data-loading pipeline works,
- [ ] preprocessing pipeline works,
- [ ] leakage-safe validation exists,
- [ ] MRR@25 implementation is tested,
- [ ] V0 baseline works,
- [ ] final candidate-generation system works,
- [ ] final ranking model works,
- [ ] multi-spectrum inference works,
- [ ] valid `submission.csv` is produced,
- [ ] Kaggle offline notebook runs within time,
- [ ] best model/checkpoint is documented,
- [ ] experiments are recorded,
- [ ] ablation results exist,
- [ ] error analysis exists,
- [ ] README is complete,
- [ ] architecture documentation exists,
- [ ] another developer can reproduce the inference pipeline,
- [ ] final GitHub release is portfolio quality.

---

# 67. Final Deliverables

## Competition deliverables

```text
submission.csv
Kaggle inference notebook
model checkpoints
candidate indexes
runtime report
```

## Engineering deliverables

```text
source code
tests
configs
environment specification
experiment registry
```

## Research deliverables

```text
EDA
ablation table
error analysis
model comparison
research report
architecture diagrams
```

## Portfolio deliverables

```text
GitHub README
screenshots
results table
optional demo
```

---

# 68. Recommended Development Order

Follow exactly:

```text
1. Environment
2. Dataset validation
3. EDA
4. Preprocessing
5. Chemistry utilities
6. Leakage-safe validation
7. MRR metric
8. Spectral similarity baseline
9. First Kaggle submission
10. Neutral-mass filtering
11. Learned spectrum embeddings
12. Candidate structure database
13. Formula constraints
14. Molecular representations
15. Spectrum–molecule scorer
16. Multi-spectrum fusion
17. Learned reranker
18. Ensemble
19. Error analysis
20. Runtime optimization
21. De novo research
22. Portfolio/demo
23. Final release
```

Never jump directly to de novo molecular generation before candidate retrieval and validation are working.

---

# 69. Immediate First Steps

## Step 1 — Create/open project folder

Example:

```powershell
cd "C:\Users\abdel\OneDrive\Desktop"
mkdir "Enveda-CASMI-2026"
cd "Enveda-CASMI-2026"
code .
```

If the folder already exists, only navigate to it.

## Step 2 — Create the environment

```powershell
py -3.11 -m venv .venv --system-site-packages
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Step 3 — Verify Python

```powershell
python --version
```

## Step 4 — Verify GPU

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Step 5 — Inspect installed packages

```powershell
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

Check RDKit:

```powershell
python -c "import rdkit; print('RDKit:', rdkit.__version__)"
```

Only install it if missing.

## Step 6 — Initialize Git

```powershell
git init
git branch -M main
```

## Step 7 — Create project structure

Create the folders specified in this document.

## Step 8 — Place competition files

```text
data/raw/train.parquet
data/raw/test.parquet
data/raw/sample_submission.csv
```

## Step 9 — Build first script

Create:

```text
scripts/inspect_data.py
```

It should print:

```text
train shape
test shape
columns
dtypes
unique molecules
number of spectra
adduct counts
library counts
spectra-per-molecule summary
peak-count summary
missing values
```

## Step 10 — Commit

```powershell
git add .
git commit -m "chore: initialize Enveda CASMI 2026 project"
```

---

# 70. First Major Milestone

The first milestone is NOT a deep-learning model.

The first milestone is:

```text
Raw dataset
    ↓
Validated loader
    ↓
EDA
    ↓
Preprocessing
    ↓
Leakage-safe validation
    ↓
Spectral similarity baseline
    ↓
Top-25 prediction
    ↓
Valid submission.csv
```

Once this works, every later model can be evaluated against a trustworthy baseline.

---

# 71. Final Target Architecture

```text
                         UNKNOWN MOLECULE
                               │
                ┌──────────────┴──────────────┐
                │ Multiple MS/MS spectra      │
                └──────────────┬──────────────┘
                               │
                               ▼
                      Spectrum preprocessing
                               │
                               ▼
                    Per-spectrum representation
                               │
                  ┌────────────┴─────────────┐
                  │                          │
                  ▼                          ▼
        Classical similarity       Learned spectrum encoder
                  │                          │
                  └────────────┬─────────────┘
                               ▼
                     Multi-spectrum fusion
                               │
                               ▼
                      Chemistry constraints
                  ┌────────────┼────────────┐
                  │            │            │
                  ▼            ▼            ▼
            neutral mass     formula      adduct
                  │            │            │
                  └────────────┼────────────┘
                               ▼
                     Candidate generation
                               │
                               ▼
                 Candidate molecular features
                  ┌────────────┼────────────┐
                  │            │            │
                  ▼            ▼            ▼
             fingerprint     graph      descriptors
                  │            │            │
                  └────────────┼────────────┘
                               ▼
                 Spectrum–molecule scoring
                               │
                               ▼
                     Learned reranker
                               │
                               ▼
                         Ensemble
                               │
                               ▼
                     Top 25 structures
                               │
                               ▼
                     Canonical SMILES
                               │
                               ▼
                      submission.csv
```

---

# 72. Guiding Principle

The core strategy is:

```text
Understand the spectrum
        +
Infer chemistry constraints
        +
Retrieve strong candidates
        +
Learn spectrum–molecule compatibility
        +
Exploit all spectra per molecule
        +
Rerank aggressively
        =
Competitive MRR@25
```

Do not rely on a single model.

Do not optimize one subsystem in isolation.

The most important practical quantity before reranking is:

```text
Does the true structure enter the candidate set?
```

If the answer is no, no reranker can recover it.

Therefore always track both:

```text
Candidate Recall@K
and
Final MRR@25
```

---

# 73. Project Status at Creation

```text
Stage 0 — Pending
Stage 1 — Pending
Stage 2 — Pending
Stage 3 — Pending
Stage 4 — Pending
Stage 5 — Pending
Stage 6 — Pending
Stage 7+ — Planned
```

Current next action:

```text
Initialize repository and environment, then inspect the competition dataset.
```

---

# 74. Competition-Specific Constraints to Preserve

- Predictions are made per molecule, not per spectrum.
- Up to 25 SMILES candidates are allowed.
- Candidates must be ranked best-first.
- Submission must contain one row per `molecule_id`.
- MRR@25 is the primary metric.
- The final Kaggle workflow must operate offline.
- GPU notebook runtime must remain within competition limits.
- External assets must comply with competition rules.
- Reproducibility is essential for a serious competition solution.

---

# 75. Update Policy for This Plan

This file is a living technical specification.

When a requirement changes:

1. identify affected requirements,
2. identify affected stages,
3. identify affected modules,
4. identify dependency changes,
5. update the decision log,
6. update assumptions,
7. update stage tasks,
8. record reason for the change.

Examples that should trigger updates:

- new competition rule,
- professor/research feedback,
- pretrained model selection,
- dataset change,
- model failure,
- GPU limitation,
- major leaderboard discovery,
- new external dataset,
- architecture replacement.

---

# END OF MASTER PLAN
