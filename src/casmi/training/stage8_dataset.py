from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumMoleculeDataset(Dataset):
    """
    Memory-mapped paired spectrum-to-molecule dataset.

    Stage 6 supplies:
        - scaled m/z values
        - intensities
        - masks
        - raw precursor m/z
        - labels
        - split information

    Stage 8 supplies:
        - Morgan fingerprints
        - aligned molecule indices

    Important:
        Stage 6 trained SpectrumEncoder with
        precursor_mz / 1000.0.

        This dataset reproduces that exact preprocessing.
    """

    def __init__(
        self,
        stage6_dir: str | Path,
        stage8_dir: str | Path,
        split_value: Optional[int] = None,
        precursor_scale: float = 1000.0,
    ) -> None:

        super().__init__()

        self.stage6_dir = Path(stage6_dir)
        self.stage8_dir = Path(stage8_dir)

        if precursor_scale <= 0:
            raise ValueError(
                "precursor_scale must be > 0."
            )

        self.precursor_scale = float(
            precursor_scale
        )

        if not self.stage6_dir.exists():
            raise FileNotFoundError(
                f"Stage 6 directory does not exist: "
                f"{self.stage6_dir}"
            )

        if not self.stage8_dir.exists():
            raise FileNotFoundError(
                f"Stage 8 directory does not exist: "
                f"{self.stage8_dir}"
            )

        # =================================================
        # Stage 6 arrays
        # =================================================

        self.mzs = np.load(
            self.stage6_dir / "mzs.npy",
            mmap_mode="r",
        )

        self.intensities = np.load(
            self.stage6_dir / "intensities.npy",
            mmap_mode="r",
        )

        self.mask = np.load(
            self.stage6_dir / "mask.npy",
            mmap_mode="r",
        )

        self.precursor_mz = np.load(
            self.stage6_dir / "precursor_mz.npy",
            mmap_mode="r",
        )

        self.labels = np.load(
            self.stage6_dir / "labels.npy",
            mmap_mode="r",
        )

        self.split = np.load(
            self.stage6_dir / "split.npy",
            mmap_mode="r",
        )

        # =================================================
        # Stage 8 arrays
        # =================================================

        self.fingerprints = np.load(
            self.stage8_dir / "morgan_fingerprints.npy",
            mmap_mode="r",
        )

        self.spectrum_molecule_indices = np.load(
            self.stage8_dir
            / "spectrum_molecule_indices.npy",
            mmap_mode="r",
        )

        # =================================================
        # Validate arrays
        # =================================================

        self._validate_arrays()

        # =================================================
        # Select split
        # =================================================

        if split_value is None:

            self.indices = np.arange(
                len(self.labels),
                dtype=np.int32,
            )

        else:

            if split_value not in {0, 1}:
                raise ValueError(
                    "split_value must be None, 0, or 1."
                )

            self.indices = np.flatnonzero(
                self.split == split_value
            ).astype(
                np.int32
            )

    def _validate_arrays(
        self,
    ) -> None:

        n_spectra = len(
            self.labels
        )

        arrays = {
            "mzs": self.mzs,
            "intensities": self.intensities,
            "mask": self.mask,
            "precursor_mz": self.precursor_mz,
            "split": self.split,
            "spectrum_molecule_indices":
                self.spectrum_molecule_indices,
        }

        for name, array in arrays.items():

            if len(array) != n_spectra:
                raise RuntimeError(
                    f"{name} length "
                    f"{len(array):,} "
                    f"does not match labels "
                    f"{n_spectra:,}."
                )

        if self.mzs.shape != self.intensities.shape:
            raise RuntimeError(
                "mzs and intensities must have "
                "identical shape."
            )

        if self.mzs.shape != self.mask.shape:
            raise RuntimeError(
                "mzs and mask must have "
                "identical shape."
            )

        if self.mzs.ndim != 2:
            raise RuntimeError(
                "mzs must have shape "
                "[N, max_peaks]."
            )

        if self.intensities.ndim != 2:
            raise RuntimeError(
                "intensities must have shape "
                "[N, max_peaks]."
            )

        if self.mask.ndim != 2:
            raise RuntimeError(
                "mask must have shape "
                "[N, max_peaks]."
            )

        if self.precursor_mz.ndim != 1:
            raise RuntimeError(
                "precursor_mz must have shape [N]."
            )

        if self.labels.ndim != 1:
            raise RuntimeError(
                "labels must have shape [N]."
            )

        if self.split.ndim != 1:
            raise RuntimeError(
                "split must have shape [N]."
            )

        if self.fingerprints.ndim != 2:
            raise RuntimeError(
                "morgan_fingerprints.npy must have "
                "shape [num_molecules, fingerprint_dim]."
            )

        if self.spectrum_molecule_indices.ndim != 1:
            raise RuntimeError(
                "spectrum_molecule_indices.npy must have "
                "shape [N]."
            )

        molecule_indices = (
            self.spectrum_molecule_indices
        )

        if len(molecule_indices) > 0:

            minimum_index = int(
                molecule_indices.min()
            )

            maximum_index = int(
                molecule_indices.max()
            )

            if minimum_index < 0:
                raise RuntimeError(
                    "Negative molecule index detected."
                )

            if maximum_index >= len(
                self.fingerprints
            ):
                raise RuntimeError(
                    "Molecule index exceeds "
                    "fingerprint cache."
                )

    @property
    def max_peaks(
        self,
    ) -> int:

        return int(
            self.mzs.shape[1]
        )

    @property
    def fingerprint_dim(
        self,
    ) -> int:

        return int(
            self.fingerprints.shape[1]
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.indices
        )

    def __getitem__(
        self,
        item: int,
    ):

        spectrum_index = int(
            self.indices[item]
        )

        molecule_index = int(
            self.spectrum_molecule_indices[
                spectrum_index
            ]
        )

        mzs = torch.tensor(
            self.mzs[
                spectrum_index
            ],
            dtype=torch.float32,
        )

        intensities = torch.tensor(
            self.intensities[
                spectrum_index
            ],
            dtype=torch.float32,
        )

        mask = torch.tensor(
            self.mask[
                spectrum_index
            ],
            dtype=torch.bool,
        )

        # Match Stage 6 preprocessing exactly.
        precursor_value = float(
            self.precursor_mz[
                spectrum_index
            ]
        )

        precursor_value = (
            precursor_value
            / self.precursor_scale
        )

        precursor_mz = torch.tensor(
            precursor_value,
            dtype=torch.float32,
        )

        fingerprint = torch.tensor(
            self.fingerprints[
                molecule_index
            ],
            dtype=torch.uint8,
        )

        return {
            "mzs": mzs,
            "intensities": intensities,
            "mask": mask,
            "precursor_mz": precursor_mz,
            "fingerprint": fingerprint,
            "molecule_index": torch.tensor(
                molecule_index,
                dtype=torch.long,
            ),
            "spectrum_index": torch.tensor(
                spectrum_index,
                dtype=torch.long,
            ),
        }