from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.sparse import csr_matrix, vstack

from casmi.retrieval.spectral_similarity import SparseSpectrum


@dataclass
class SpectralIndex:
    matrix: csr_matrix
    structure_labels: List[str]
    row_ids: List[int]
    n_bins: int


def sparse_spectrum_to_csr(
    spectrum: SparseSpectrum,
    n_bins: int,
) -> csr_matrix:
    """
    Convert a sparse dictionary representation into
    a 1 x n_bins CSR row.
    """

    if not spectrum:
        return csr_matrix(
            (1, n_bins),
            dtype=np.float32,
        )

    indices = np.asarray(
        list(spectrum.keys()),
        dtype=np.int32,
    )

    values = np.asarray(
        list(spectrum.values()),
        dtype=np.float32,
    )

    order = np.argsort(indices)

    indices = indices[order]
    values = values[order]

    indptr = np.asarray(
        [0, len(indices)],
        dtype=np.int32,
    )

    return csr_matrix(
        (
            values,
            indices,
            indptr,
        ),
        shape=(1, n_bins),
        dtype=np.float32,
    )


def build_spectral_index(
    spectra: Sequence[SparseSpectrum],
    structure_labels: Sequence[str],
    row_ids: Sequence[int],
    n_bins: int,
) -> SpectralIndex:
    """
    Build a fully sparse CSR reference index.

    No dense intermediate is created.
    """

    if not (
        len(spectra)
        == len(structure_labels)
        == len(row_ids)
    ):
        raise ValueError(
            "spectra, structure_labels, and row_ids "
            "must have identical lengths."
        )

    rows = [
        sparse_spectrum_to_csr(
            spectrum,
            n_bins,
        )
        for spectrum in spectra
    ]

    matrix = vstack(
        rows,
        format="csr",
        dtype=np.float32,
    )

    return SpectralIndex(
        matrix=matrix,
        structure_labels=list(
            structure_labels
        ),
        row_ids=list(
            row_ids
        ),
        n_bins=n_bins,
    )


def retrieve_scores(
    index: SpectralIndex,
    query: SparseSpectrum,
) -> np.ndarray:
    """
    Calculate cosine-like dot-product scores.

    Assumes both reference and query spectra were already
    L2-normalized.
    """

    query_vector = sparse_spectrum_to_csr(
        query,
        index.n_bins,
    )

    scores = (
        index.matrix
        @ query_vector.T
    ).toarray().ravel()

    return scores


def collapse_scores_by_structure(
    scores: np.ndarray,
    structure_labels: Sequence[str],
) -> dict[str, float]:
    """
    Collapse spectrum-level scores into molecule-level scores
    using maximum pooling.
    """

    if len(scores) != len(
        structure_labels
    ):
        raise ValueError(
            "scores and structure_labels must "
            "have identical lengths."
        )

    molecule_scores: dict[
        str,
        float,
    ] = {}

    for structure, score in zip(
        structure_labels,
        scores,
    ):
        score = float(score)

        previous = molecule_scores.get(
            structure
        )

        if (
            previous is None
            or score > previous
        ):
            molecule_scores[
                structure
            ] = score

    return molecule_scores


def rank_structures(
    scores: np.ndarray,
    structure_labels: Sequence[str],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Return structures ranked by their best spectrum similarity.
    """

    molecule_scores = (
        collapse_scores_by_structure(
            scores,
            structure_labels,
        )
    )

    ranked = sorted(
        molecule_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if top_k is not None:
        ranked = ranked[
            :top_k
        ]

    return ranked