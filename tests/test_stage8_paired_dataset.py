import numpy as np
import torch

from casmi.training.stage8_dataset import (
    SpectrumMoleculeDataset,
)


def build_tiny_cache(
    tmp_path,
):

    stage6 = (
        tmp_path
        / "stage6"
    )

    stage8 = (
        tmp_path
        / "stage8"
    )

    stage6.mkdir()
    stage8.mkdir()

    n_spectra = 6
    max_peaks = 4

    mzs = np.arange(
        n_spectra
        * max_peaks,
        dtype=np.float32,
    ).reshape(
        n_spectra,
        max_peaks,
    )

    intensities = np.ones(
        (
            n_spectra,
            max_peaks,
        ),
        dtype=np.float32,
    )

    mask = np.ones(
        (
            n_spectra,
            max_peaks,
        ),
        dtype=bool,
    )

    precursor_mz = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [
            0,
            1,
            2,
            0,
            1,
            2,
        ],
        dtype=np.int32,
    )

    split = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=np.uint8,
    )

    fingerprints = np.array(
        [
            [
                1, 0, 0, 0,
                1, 0, 0, 0,
            ],
            [
                0, 1, 0, 0,
                0, 1, 0, 0,
            ],
            [
                0, 0, 1, 0,
                0, 0, 1, 0,
            ],
        ],
        dtype=np.uint8,
    )

    spectrum_molecule_indices = np.array(
        [
            2,
            0,
            1,
            2,
            0,
            1,
        ],
        dtype=np.int32,
    )

    np.save(
        stage6 / "mzs.npy",
        mzs,
    )

    np.save(
        stage6 / "intensities.npy",
        intensities,
    )

    np.save(
        stage6 / "mask.npy",
        mask,
    )

    np.save(
        stage6 / "precursor_mz.npy",
        precursor_mz,
    )

    np.save(
        stage6 / "labels.npy",
        labels,
    )

    np.save(
        stage6 / "split.npy",
        split,
    )

    np.save(
        stage8 / "morgan_fingerprints.npy",
        fingerprints,
    )

    np.save(
        stage8 / "spectrum_molecule_indices.npy",
        spectrum_molecule_indices,
    )

    return stage6, stage8


def test_all_split_length(
    tmp_path,
):

    stage6, stage8 = build_tiny_cache(
        tmp_path
    )

    dataset = SpectrumMoleculeDataset(
        stage6,
        stage8,
    )

    assert len(dataset) == 6


def test_train_split_length(
    tmp_path,
):

    stage6, stage8 = build_tiny_cache(
        tmp_path
    )

    dataset = SpectrumMoleculeDataset(
        stage6,
        stage8,
        split_value=0,
    )

    assert len(dataset) == 3


def test_validation_split_length(
    tmp_path,
):

    stage6, stage8 = build_tiny_cache(
        tmp_path
    )

    dataset = SpectrumMoleculeDataset(
        stage6,
        stage8,
        split_value=1,
    )

    assert len(dataset) == 3


def test_tensor_shapes(
    tmp_path,
):

    stage6, stage8 = build_tiny_cache(
        tmp_path
    )

    dataset = SpectrumMoleculeDataset(
        stage6,
        stage8,
    )

    sample = dataset[0]

    assert sample["mzs"].shape == (4,)
    assert sample["intensities"].shape == (4,)
    assert sample["mask"].shape == (4,)
    assert sample["fingerprint"].shape == (8,)


def test_correct_molecule_alignment(
    tmp_path,
):

    stage6, stage8 = build_tiny_cache(
        tmp_path
    )

    dataset = SpectrumMoleculeDataset(
        stage6,
        stage8,
    )

    sample = dataset[0]

    assert sample[
        "molecule_index"
    ].item() == 2

    expected = torch.tensor(
        [
            0, 0, 1, 0,
            0, 0, 1, 0,
        ],
        dtype=torch.uint8,
    )

    assert torch.equal(
        sample["fingerprint"],
        expected,
    )


def test_data_types(
    tmp_path,
):

    stage6, stage8 = build_tiny_cache(
        tmp_path
    )

    dataset = SpectrumMoleculeDataset(
        stage6,
        stage8,
    )

    sample = dataset[0]

    assert (
        sample["mzs"].dtype
        == torch.float32
    )

    assert (
        sample["intensities"].dtype
        == torch.float32
    )

    assert (
        sample["mask"].dtype
        == torch.bool
    )

    assert (
        sample["fingerprint"].dtype
        == torch.uint8
    )

    assert (
        sample["molecule_index"].dtype
        == torch.long
    )


def test_invalid_split(
    tmp_path,
):

    stage6, stage8 = build_tiny_cache(
        tmp_path
    )

    try:

        SpectrumMoleculeDataset(
            stage6,
            stage8,
            split_value=2,
        )

        assert False

    except ValueError:

        pass