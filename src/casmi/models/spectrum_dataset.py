from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedSpectrumDataset(Dataset):
    """
    PyTorch Dataset backed by Stage 6 memory-mapped NumPy arrays.

    Each item contains:
        peaks:       [max_peaks, 2]
                     column 0 = scaled m/z
                     column 1 = intensity

        mask:        [max_peaks]
        label:       integer structure ID
        precursor:   scaled precursor m/z
        row_index:   original train.parquet row index
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        precursor_scale: float = 1000.0,
    ):
        super().__init__()

        self.cache_dir = Path(
            cache_dir
        )

        if split not in {
            "train",
            "validation",
        }:
            raise ValueError(
                "split must be either "
                "'train' or 'validation'."
            )

        self.split = split

        # -----------------------------------------------
        # Memory-mapped arrays
        # -----------------------------------------------

        self.mzs = np.load(
            self.cache_dir / "mzs.npy",
            mmap_mode="r",
        )

        self.intensities = np.load(
            self.cache_dir / "intensities.npy",
            mmap_mode="r",
        )

        self.mask = np.load(
            self.cache_dir / "mask.npy",
            mmap_mode="r",
        )

        self.labels = np.load(
            self.cache_dir / "labels.npy",
            mmap_mode="r",
        )

        self.row_indices = np.load(
            self.cache_dir / "row_indices.npy",
            mmap_mode="r",
        )

        self.precursor_mz = np.load(
            self.cache_dir / "precursor_mz.npy",
            mmap_mode="r",
        )

        self.split_codes = np.load(
            self.cache_dir / "split.npy",
            mmap_mode="r",
        )

        # -----------------------------------------------
        # Split selection
        # -----------------------------------------------

        split_code = (
            0
            if split == "train"
            else 1
        )

        self.indices = np.flatnonzero(
            self.split_codes
            == split_code
        ).astype(
            np.int64
        )

        self.precursor_scale = float(
            precursor_scale
        )

    def __len__(self) -> int:
        return len(
            self.indices
        )

    def __getitem__(
        self,
        index: int,
    ):

        cache_index = int(
            self.indices[
                index
            ]
        )

        mz = np.asarray(
            self.mzs[
                cache_index
            ],
            dtype=np.float32,
        ).copy()

        intensity = np.asarray(
            self.intensities[
                cache_index
            ],
            dtype=np.float32,
        ).copy()

        mask = np.asarray(
            self.mask[
                cache_index
            ],
            dtype=np.bool_,
        ).copy()

        # [128, 2]
        peaks = np.stack(
            [
                mz,
                intensity,
            ],
            axis=-1,
        )

        label = int(
            self.labels[
                cache_index
            ]
        )

        precursor = (
            float(
                self.precursor_mz[
                    cache_index
                ]
            )
            / self.precursor_scale
        )

        row_index = int(
            self.row_indices[
                cache_index
            ]
        )

        return {
            "peaks": torch.from_numpy(
                peaks
            ),
            "mask": torch.from_numpy(
                mask
            ),
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "precursor_mz": torch.tensor(
                precursor,
                dtype=torch.float32,
            ),
            "row_index": torch.tensor(
                row_index,
                dtype=torch.long,
            ),
        }