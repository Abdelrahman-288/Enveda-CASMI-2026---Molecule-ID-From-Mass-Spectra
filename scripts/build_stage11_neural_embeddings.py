from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi.models.molecule_encoder import (
    MoleculeEncoder,
)
from casmi.models.spectrum_encoder import (
    SpectrumEncoder,
)

from casmi.spectra.preprocessing import (
    SpectrumPreprocessingConfig,
    preprocess_spectrum,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATASET_DIR = (
    PROJECT_ROOT
    / "enveda-CASMI26-molecule-id-mass-spectra"
)

TEST_PATH = (
    DATASET_DIR
    / "test.parquet"
)

STAGE11_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage11"
)

FINGERPRINTS_PATH = (
    STAGE11_DIR
    / "test_candidate_morgan_fingerprints.npy"
)

STRUCTURE_MAP_PATH = (
    STAGE11_DIR
    / "test_candidate_fingerprint_map.parquet"
)

MOLECULE_EMBEDDINGS_PATH = (
    STAGE11_DIR
    / "test_candidate_embeddings.npy"
)

SPECTRUM_EMBEDDINGS_PATH = (
    STAGE11_DIR
    / "test_spectrum_embeddings.npy"
)

SPECTRUM_MAP_PATH = (
    STAGE11_DIR
    / "test_spectrum_embedding_map.parquet"
)

STAGE6_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage6"
    / "contrastive_full_best.pt"
)

STAGE8_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "stage8"
    / "candidate_aware_best.pt"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage11_neural_embeddings_summary.txt"
)


# ============================================================
# Configuration
# ============================================================

MAX_PEAKS = 128

# Exact Stage 6 scaling.
MZ_SCALE = 1000.0

MOLECULE_BATCH_SIZE = 1024
SPECTRUM_BATCH_SIZE = 256

FINGERPRINT_DIM = 2048
EMBEDDING_DIM = 128


# ============================================================
# Model builders
# ============================================================

def build_spectrum_encoder() -> SpectrumEncoder:
    """
    Construct the exact Stage 6 spectrum encoder architecture.
    """
    return SpectrumEncoder()


def build_molecule_encoder() -> MoleculeEncoder:
    """
    Construct the Stage 8 molecule encoder architecture.
    """
    return MoleculeEncoder()


# ============================================================
# Checkpoint helpers
# ============================================================

def _is_state_dict(
    value,
) -> bool:
    """
    Return True when value looks like a PyTorch state_dict.
    """

    if not isinstance(
        value,
        dict,
    ):
        return False

    if not value:
        return False

    return all(
        isinstance(key, str)
        and torch.is_tensor(tensor)
        for key, tensor
        in value.items()
    )


def _strip_prefix(
    state_dict,
    prefix: str,
):
    """
    Remove a common prefix from all state-dict keys.
    """

    if not state_dict:
        return state_dict

    if not all(
        key.startswith(prefix)
        for key in state_dict
    ):
        return state_dict

    return {
        key[len(prefix):]: value
        for key, value
        in state_dict.items()
    }


def _candidate_state_dicts(
    checkpoint,
    preferred_keys,
):
    """
    Find plausible state_dict objects inside a checkpoint.
    """

    candidates = []

    if _is_state_dict(
        checkpoint
    ):
        candidates.append(
            (
                "root",
                checkpoint,
            )
        )

    if isinstance(
        checkpoint,
        dict,
    ):

        for key in preferred_keys:

            value = checkpoint.get(
                key
            )

            if _is_state_dict(
                value
            ):
                candidates.append(
                    (
                        key,
                        value,
                    )
                )

        for key, value in checkpoint.items():

            if (
                key not in preferred_keys
                and _is_state_dict(value)
            ):
                candidates.append(
                    (
                        key,
                        value,
                    )
                )

    return candidates


def load_model_checkpoint(
    model,
    checkpoint_path: Path,
    *,
    preferred_keys,
    prefixes,
    device,
):
    """
    Load a checkpoint while handling the various checkpoint
    structures used throughout the project.
    """

    print(
        "\nLoading checkpoint:"
    )

    print(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    candidates = (
        _candidate_state_dicts(
            checkpoint,
            preferred_keys,
        )
    )

    if not candidates:
        raise RuntimeError(
            "No state_dict found in checkpoint:\n"
            f"{checkpoint_path}"
        )

    attempts = []

    for (
        source_name,
        state_dict,
    ) in candidates:

        variants = [
            (
                "original",
                state_dict,
            )
        ]

        for prefix in prefixes:

            stripped = (
                _strip_prefix(
                    state_dict,
                    prefix,
                )
            )

            if (
                stripped
                is not state_dict
            ):
                variants.append(
                    (
                        f"strip:{prefix}",
                        stripped,
                    )
                )

        for (
            variant_name,
            candidate,
        ) in variants:

            try:

                incompatible = (
                    model.load_state_dict(
                        candidate,
                        strict=False,
                    )
                )

                missing = list(
                    incompatible.missing_keys
                )

                unexpected = list(
                    incompatible.unexpected_keys
                )

                if (
                    len(missing) == 0
                    and len(unexpected) == 0
                ):

                    model.to(
                        device
                    )

                    model.eval()

                    print(
                        "Loaded from: "
                        f"{source_name} / "
                        f"{variant_name}"
                    )

                    return

                attempts.append(
                    {
                        "source": (
                            source_name
                        ),
                        "variant": (
                            variant_name
                        ),
                        "missing": (
                            missing[:10]
                        ),
                        "unexpected": (
                            unexpected[:10]
                        ),
                    }
                )

            except RuntimeError as exc:

                attempts.append(
                    {
                        "source": (
                            source_name
                        ),
                        "variant": (
                            variant_name
                        ),
                        "error": str(exc),
                    }
                )

    raise RuntimeError(
        "Could not load checkpoint into "
        f"{type(model).__name__}.\n\n"
        f"Checkpoint:\n"
        f"{checkpoint_path}\n\n"
        f"Attempts:\n"
        f"{attempts}"
    )


# ============================================================
# Spectrum preprocessing
# ============================================================

def build_preprocessing_config():
    """
    Reproduce the preprocessing configuration used by
    the Stage 6 spectrum cache.
    """

    return SpectrumPreprocessingConfig(
        max_peaks=MAX_PEAKS,
    )


def preprocess_test_spectra(
    test: pd.DataFrame,
):
    """
    Convert raw test spectra into the exact tensor inputs used
    by the Stage 6 encoder.

    Returns
    -------
    mz_cache
        [num_spectra, 128] float32

    intensity_cache
        [num_spectra, 128] float32

    mask_cache
        [num_spectra, 128] bool

    precursor_cache
        [num_spectra] float32

    peak_counts
        [num_spectra] int32
    """

    config = (
        build_preprocessing_config()
    )

    num_spectra = len(
        test
    )

    mz_cache = np.zeros(
        (
            num_spectra,
            MAX_PEAKS,
        ),
        dtype=np.float32,
    )

    intensity_cache = np.zeros(
        (
            num_spectra,
            MAX_PEAKS,
        ),
        dtype=np.float32,
    )

    mask_cache = np.zeros(
        (
            num_spectra,
            MAX_PEAKS,
        ),
        dtype=bool,
    )

    precursor_cache = np.zeros(
        num_spectra,
        dtype=np.float32,
    )

    peak_counts = np.zeros(
        num_spectra,
        dtype=np.int32,
    )

    print(
        "\nPreprocessing test spectra..."
    )

    for index, row in enumerate(
        test.itertuples(
            index=False
        )
    ):

        mzs, intensities = (
            preprocess_spectrum(
                row.ms2_mzs,
                row.ms2_normalized_intensities,
                precursor_mz=(
                    row.precursor_mz
                ),
                config=config,
            )
        )

        n_peaks = min(
            len(mzs),
            MAX_PEAKS,
        )

        if n_peaks > 0:

            mz_values = (
                np.asarray(
                    mzs[:n_peaks],
                    dtype=np.float32,
                )
            )

            intensity_values = (
                np.asarray(
                    intensities[
                        :n_peaks
                    ],
                    dtype=np.float32,
                )
            )

            # Exact neural input scaling from Stage 6.
            mz_values = (
                mz_values
                / MZ_SCALE
            )

            mz_cache[
                index,
                :n_peaks,
            ] = mz_values

            intensity_cache[
                index,
                :n_peaks,
            ] = intensity_values

            mask_cache[
                index,
                :n_peaks,
            ] = True

        # Important:
        # Stage 6 / Stage 8 precursor input is also scaled.
        precursor_cache[
            index
        ] = (
            float(
                row.precursor_mz
            )
            / MZ_SCALE
        )

        peak_counts[
            index
        ] = n_peaks

        if (
            (index + 1)
            % 200
            == 0
            or index + 1
            == num_spectra
        ):
            print(
                "  Processed "
                f"{index + 1:,}"
                "/"
                f"{num_spectra:,}"
            )

    return (
        mz_cache,
        intensity_cache,
        mask_cache,
        precursor_cache,
        peak_counts,
    )


# ============================================================
# Molecule encoding
# ============================================================

@torch.inference_mode()
def encode_molecules(
    model,
    fingerprints,
    *,
    device,
):
    """
    Encode Morgan fingerprints into normalized 128-D
    molecule embeddings.
    """

    num_structures = int(
        fingerprints.shape[0]
    )

    embeddings = np.zeros(
        (
            num_structures,
            EMBEDDING_DIM,
        ),
        dtype=np.float32,
    )

    print(
        "\nEncoding candidate molecules..."
    )

    for start in range(
        0,
        num_structures,
        MOLECULE_BATCH_SIZE,
    ):

        stop = min(
            start
            + MOLECULE_BATCH_SIZE,
            num_structures,
        )

        batch_numpy = np.asarray(
            fingerprints[
                start:stop
            ],
            dtype=np.float32,
        )

        batch = (
            torch.from_numpy(
                batch_numpy
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        output = model(
            batch
        )

        output_numpy = (
            output
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if output_numpy.shape != (
            stop - start,
            EMBEDDING_DIM,
        ):
            raise RuntimeError(
                "Unexpected molecule embedding "
                f"shape: {output_numpy.shape}"
            )

        embeddings[
            start:stop
        ] = output_numpy

        if (
            stop % 5000
            < MOLECULE_BATCH_SIZE
            or stop
            == num_structures
        ):
            print(
                "  Encoded "
                f"{stop:,}"
                "/"
                f"{num_structures:,}"
            )

    return embeddings


# ============================================================
# Spectrum encoding
# ============================================================

@torch.inference_mode()
def encode_spectra(
    model,
    mzs,
    intensities,
    masks,
    precursor_mz,
    *,
    device,
):
    """
    Encode test spectra using the Stage 6 SpectrumEncoder.

    IMPORTANT:
    SpectrumEncoder.forward expects:

        model(
            peaks,
            mask,
            precursor_mz,
        )

    where:

        peaks.shape = [B, 128, 2]

        peaks[..., 0] = normalized m/z
        peaks[..., 1] = normalized intensity
    """

    num_spectra = int(
        mzs.shape[0]
    )

    embeddings = np.zeros(
        (
            num_spectra,
            EMBEDDING_DIM,
        ),
        dtype=np.float32,
    )

    print(
        "\nEncoding test spectra..."
    )

    for start in range(
        0,
        num_spectra,
        SPECTRUM_BATCH_SIZE,
    ):

        stop = min(
            start
            + SPECTRUM_BATCH_SIZE,
            num_spectra,
        )

        # ----------------------------------------------------
        # Combine m/z and intensity exactly as Stage 6 dataset
        # does.
        #
        # [B, 128] + [B, 128]
        #
        # becomes:
        #
        # [B, 128, 2]
        # ----------------------------------------------------

        peak_numpy = np.stack(
            [
                np.asarray(
                    mzs[
                        start:stop
                    ],
                    dtype=np.float32,
                ),
                np.asarray(
                    intensities[
                        start:stop
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=-1,
        )

        expected_peak_shape = (
            stop - start,
            MAX_PEAKS,
            2,
        )

        if (
            peak_numpy.shape
            != expected_peak_shape
        ):
            raise RuntimeError(
                "Unexpected peak tensor shape: "
                f"{peak_numpy.shape}; "
                "expected "
                f"{expected_peak_shape}"
            )

        peaks_batch = (
            torch.from_numpy(
                peak_numpy
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        mask_batch = (
            torch.from_numpy(
                np.asarray(
                    masks[
                        start:stop
                    ],
                    dtype=bool,
                )
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        precursor_batch = (
            torch.from_numpy(
                np.asarray(
                    precursor_mz[
                        start:stop
                    ],
                    dtype=np.float32,
                )
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        # Correct SpectrumEncoder call:
        #
        # forward(
        #     peaks,
        #     mask,
        #     precursor_mz,
        # )
        output = model(
            peaks_batch,
            mask_batch,
            precursor_batch,
        )

        output_numpy = (
            output
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if output_numpy.shape != (
            stop - start,
            EMBEDDING_DIM,
        ):
            raise RuntimeError(
                "Unexpected spectrum embedding "
                f"shape: {output_numpy.shape}"
            )

        embeddings[
            start:stop
        ] = output_numpy

        print(
            "  Encoded "
            f"{stop:,}"
            "/"
            f"{num_spectra:,}"
        )

    return embeddings


# ============================================================
# Validation helpers
# ============================================================

def embedding_stats(
    embeddings: np.ndarray,
):
    """
    Compute sanity-check statistics for embeddings.
    """

    if (
        embeddings.ndim != 2
    ):
        raise ValueError(
            "Embeddings must be a "
            "2-D matrix."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    return {
        "minimum_norm": float(
            norms.min()
        ),
        "median_norm": float(
            np.median(
                norms
            )
        ),
        "mean_norm": float(
            norms.mean()
        ),
        "maximum_norm": float(
            norms.max()
        ),
        "nonfinite": int(
            np.size(
                embeddings
            )
            - np.isfinite(
                embeddings
            ).sum()
        ),
        "zero_rows": int(
            np.sum(
                norms == 0
            )
        ),
    }


def validate_embeddings(
    embeddings: np.ndarray,
    *,
    expected_rows: int,
    name: str,
):
    """
    Strong embedding validation.
    """

    expected_shape = (
        expected_rows,
        EMBEDDING_DIM,
    )

    if (
        embeddings.shape
        != expected_shape
    ):
        raise RuntimeError(
            f"{name} embeddings have "
            f"shape {embeddings.shape}; "
            f"expected {expected_shape}."
        )

    if (
        embeddings.dtype
        != np.float32
    ):
        raise RuntimeError(
            f"{name} embeddings must "
            "be float32."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise RuntimeError(
            f"{name} embeddings contain "
            "non-finite values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if np.any(
        norms <= 0
    ):
        raise RuntimeError(
            f"{name} embeddings contain "
            "zero-norm rows."
        )


# ============================================================
# Main
# ============================================================

def main():

    print()

    print(
        "=" * 100
    )

    print(
        "ENVEDA CASMI 2026 - "
        "STAGE 11.3 TEST NEURAL EMBEDDINGS"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    required_paths = [
        TEST_PATH,
        FINGERPRINTS_PATH,
        STRUCTURE_MAP_PATH,
        STAGE6_CHECKPOINT,
        STAGE8_CHECKPOINT,
    ]

    missing = [
        path
        for path
        in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(
                str(path)
                for path
                in missing
            )
        )

    STAGE11_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if (
        device.type
        == "cuda"
    ):

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        torch.backends.cudnn.benchmark = (
            True
        )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test = (
        pd.read_parquet(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
    )

    test[
        "spectrum_index"
    ] = np.arange(
        len(test),
        dtype=np.int64,
    )

    required_test_columns = [
        "molecule_id",
        "spectrum_id",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "adduct",
    ]

    missing_test_columns = [
        column
        for column
        in required_test_columns
        if column
        not in test.columns
    ]

    if missing_test_columns:
        raise ValueError(
            "Test dataframe is missing "
            f"columns: {missing_test_columns}"
        )

    # --------------------------------------------------------
    # Load structure mapping
    # --------------------------------------------------------

    structure_map = (
        pd.read_parquet(
            STRUCTURE_MAP_PATH
        )
        .sort_values(
            "candidate_structure_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    structure_indices = (
        structure_map[
            "candidate_structure_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    expected_indices = np.arange(
        len(structure_map),
        dtype=np.int64,
    )

    if not np.array_equal(
        structure_indices,
        expected_indices,
    ):
        raise RuntimeError(
            "Candidate structure indices "
            "are not contiguous."
        )

    # --------------------------------------------------------
    # Load fingerprints
    # --------------------------------------------------------

    fingerprints = np.load(
        FINGERPRINTS_PATH,
        mmap_mode="r",
    )

    expected_fingerprint_shape = (
        len(structure_map),
        FINGERPRINT_DIM,
    )

    if (
        fingerprints.shape
        != expected_fingerprint_shape
    ):
        raise RuntimeError(
            "Fingerprint / structure map "
            "shape mismatch.\n"
            f"Fingerprint shape: "
            f"{fingerprints.shape}\n"
            f"Expected: "
            f"{expected_fingerprint_shape}"
        )

    print(
        "\nCandidate structures: "
        f"{len(structure_map):,}"
    )

    print(
        "Test spectra: "
        f"{len(test):,}"
    )

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    spectrum_encoder = (
        build_spectrum_encoder()
    )

    molecule_encoder = (
        build_molecule_encoder()
    )

    # --------------------------------------------------------
    # Load Stage 6 spectrum checkpoint
    # --------------------------------------------------------

    load_model_checkpoint(
        spectrum_encoder,
        STAGE6_CHECKPOINT,
        preferred_keys=[
            "model_state_dict",
            "state_dict",
            "spectrum_encoder_state_dict",
            "spectrum_encoder",
            "model",
        ],
        prefixes=[
            "module.",
            "spectrum_encoder.",
            "model.",
        ],
        device=device,
    )

    # --------------------------------------------------------
    # Load Stage 8 candidate-aware molecule checkpoint
    # --------------------------------------------------------

    load_model_checkpoint(
        molecule_encoder,
        STAGE8_CHECKPOINT,
        preferred_keys=[
            "molecule_encoder_state_dict",
            "molecule_encoder",
            "model_state_dict",
            "state_dict",
            "model",
        ],
        prefixes=[
            "module.",
            "molecule_encoder.",
            "model.",
        ],
        device=device,
    )

    spectrum_parameter_count = sum(
        parameter.numel()
        for parameter
        in spectrum_encoder.parameters()
    )

    molecule_parameter_count = sum(
        parameter.numel()
        for parameter
        in molecule_encoder.parameters()
    )

    print(
        "\nSpectrum parameters: "
        f"{spectrum_parameter_count:,}"
    )

    print(
        "Molecule parameters: "
        f"{molecule_parameter_count:,}"
    )

    # --------------------------------------------------------
    # Encode molecules
    # --------------------------------------------------------

    molecule_embeddings = (
        encode_molecules(
            molecule_encoder,
            fingerprints,
            device=device,
        )
    )

    validate_embeddings(
        molecule_embeddings,
        expected_rows=(
            len(structure_map)
        ),
        name="Molecule",
    )

    molecule_stats = (
        embedding_stats(
            molecule_embeddings
        )
    )

    np.save(
        MOLECULE_EMBEDDINGS_PATH,
        molecule_embeddings,
        allow_pickle=False,
    )

    print(
        "\nSaved molecule embeddings:"
    )

    print(
        MOLECULE_EMBEDDINGS_PATH
    )

    # --------------------------------------------------------
    # Release some GPU memory
    # --------------------------------------------------------

    molecule_encoder.cpu()

    del molecule_encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Preprocess spectra
    # --------------------------------------------------------

    (
        mz_cache,
        intensity_cache,
        mask_cache,
        precursor_cache,
        peak_counts,
    ) = preprocess_test_spectra(
        test
    )

    if np.any(
        peak_counts <= 0
    ):
        empty_indices = np.where(
            peak_counts <= 0
        )[0]

        raise RuntimeError(
            "One or more test spectra became "
            "empty after preprocessing.\n"
            f"Indices: "
            f"{empty_indices[:20].tolist()}"
        )

    if np.any(
        peak_counts
        > MAX_PEAKS
    ):
        raise RuntimeError(
            "Peak count exceeded MAX_PEAKS."
        )

    # --------------------------------------------------------
    # Encode spectra
    # --------------------------------------------------------

    spectrum_embeddings = (
        encode_spectra(
            spectrum_encoder,
            mz_cache,
            intensity_cache,
            mask_cache,
            precursor_cache,
            device=device,
        )
    )

    validate_embeddings(
        spectrum_embeddings,
        expected_rows=len(test),
        name="Spectrum",
    )

    spectrum_stats = (
        embedding_stats(
            spectrum_embeddings
        )
    )

    np.save(
        SPECTRUM_EMBEDDINGS_PATH,
        spectrum_embeddings,
        allow_pickle=False,
    )

    print(
        "\nSaved spectrum embeddings:"
    )

    print(
        SPECTRUM_EMBEDDINGS_PATH
    )

    # --------------------------------------------------------
    # Spectrum map
    # --------------------------------------------------------

    spectrum_map_columns = [
        "spectrum_index",
        "molecule_id",
        "spectrum_id",
        "precursor_mz",
        "adduct",
        "ionization_mode",
        "instrument_type",
        "collision_energy_orig",
        "collision_energy_orig_units",
    ]

    available_columns = [
        column
        for column
        in spectrum_map_columns
        if column
        in test.columns
    ]

    spectrum_map = (
        test[
            available_columns
        ]
        .copy()
    )

    spectrum_map[
        "preprocessed_peak_count"
    ] = peak_counts

    spectrum_map.to_parquet(
        SPECTRUM_MAP_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # File sizes
    # --------------------------------------------------------

    molecule_size_mb = (
        MOLECULE_EMBEDDINGS_PATH
        .stat()
        .st_size
        / (1024 ** 2)
    )

    spectrum_size_mb = (
        SPECTRUM_EMBEDDINGS_PATH
        .stat()
        .st_size
        / (1024 ** 2)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_lines = [
        (
            "ENVEDA CASMI 2026 - "
            "STAGE 11.3 TEST NEURAL EMBEDDINGS"
        ),
        "",
        (
            f"Device: {device}"
        ),
        (
            "Candidate structures: "
            f"{len(structure_map):,}"
        ),
        (
            "Test spectra: "
            f"{len(test):,}"
        ),
        (
            "Embedding dimension: "
            f"{EMBEDDING_DIM}"
        ),
        "",
        "MOLECULE EMBEDDINGS",
        (
            "Shape: "
            f"{molecule_embeddings.shape}"
        ),
        (
            "Dtype: "
            f"{molecule_embeddings.dtype}"
        ),
        (
            "Minimum norm: "
            f"{molecule_stats['minimum_norm']:.6f}"
        ),
        (
            "Median norm: "
            f"{molecule_stats['median_norm']:.6f}"
        ),
        (
            "Mean norm: "
            f"{molecule_stats['mean_norm']:.6f}"
        ),
        (
            "Maximum norm: "
            f"{molecule_stats['maximum_norm']:.6f}"
        ),
        (
            "Zero rows: "
            f"{molecule_stats['zero_rows']}"
        ),
        (
            "Non-finite values: "
            f"{molecule_stats['nonfinite']}"
        ),
        (
            "File size: "
            f"{molecule_size_mb:.2f} MB"
        ),
        "",
        "SPECTRUM EMBEDDINGS",
        (
            "Shape: "
            f"{spectrum_embeddings.shape}"
        ),
        (
            "Dtype: "
            f"{spectrum_embeddings.dtype}"
        ),
        (
            "Minimum norm: "
            f"{spectrum_stats['minimum_norm']:.6f}"
        ),
        (
            "Median norm: "
            f"{spectrum_stats['median_norm']:.6f}"
        ),
        (
            "Mean norm: "
            f"{spectrum_stats['mean_norm']:.6f}"
        ),
        (
            "Maximum norm: "
            f"{spectrum_stats['maximum_norm']:.6f}"
        ),
        (
            "Zero rows: "
            f"{spectrum_stats['zero_rows']}"
        ),
        (
            "Non-finite values: "
            f"{spectrum_stats['nonfinite']}"
        ),
        (
            "Minimum peaks: "
            f"{int(peak_counts.min())}"
        ),
        (
            "Median peaks: "
            f"{np.median(peak_counts):.2f}"
        ),
        (
            "Mean peaks: "
            f"{peak_counts.mean():.2f}"
        ),
        (
            "Maximum peaks: "
            f"{int(peak_counts.max())}"
        ),
        (
            "File size: "
            f"{spectrum_size_mb:.2f} MB"
        ),
        "",
        (
            "Molecule embeddings: "
            f"{MOLECULE_EMBEDDINGS_PATH}"
        ),
        (
            "Spectrum embeddings: "
            f"{SPECTRUM_EMBEDDINGS_PATH}"
        ),
        (
            "Spectrum map: "
            f"{SPECTRUM_MAP_PATH}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()

    print(
        "=" * 100
    )

    print(
        "STAGE 11.3 SUMMARY"
    )

    print(
        "=" * 100
    )

    for line in summary_lines:
        print(
            line
        )

    print()

    print(
        "=" * 100
    )

    print(
        "STAGE 11.3 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()