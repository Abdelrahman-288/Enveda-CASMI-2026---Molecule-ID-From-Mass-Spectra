from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler

from casmi.chemistry.adducts import (
    is_supported_adduct,
    is_trusted_for_mass_filter,
)

from casmi.chemistry.mass_filter_policy import (
    HIGH_CONFIDENCE_SOURCES,
)

from casmi.retrieval.candidate_generation import (
    AdductAwareCandidateGenerator,
)

from casmi.retrieval.candidate_mass_index import (
    CandidateMassIndex,
)

from casmi.training.stage8_dataset import (
    SpectrumMoleculeDataset,
)


class CandidateAwareSpectrumDataset(Dataset):
    """
    Stage 8 paired spectrum/molecule dataset augmented with
    chemistry metadata required for hard-negative sampling.

    Candidate-aware batches are built around one anchor
    precursor.

    Structures in a batch therefore come from the same
    approximate 10 ppm precursor-derived neutral-mass region.

    This gives much harder negatives than random batches.
    """

    def __init__(
        self,
        stage6_dir: str | Path,
        stage8_dir: str | Path,
        train_metadata: pd.DataFrame,
        mass_index: CandidateMassIndex,
        structure_map: pd.DataFrame,
        split_value: int,
        precursor_scale: float = 1000.0,
        tolerance_ppm: float = 10.0,
    ) -> None:

        super().__init__()

        if split_value not in {
            0,
            1,
        }:
            raise ValueError(
                "split_value must be 0 or 1."
            )

        if tolerance_ppm <= 0:
            raise ValueError(
                "tolerance_ppm must be > 0."
            )

        self.stage6_dir = Path(
            stage6_dir
        )

        self.stage8_dir = Path(
            stage8_dir
        )

        self.split_value = int(
            split_value
        )

        self.tolerance_ppm = float(
            tolerance_ppm
        )

        # =================================================
        # Base Stage 8 paired dataset
        # =================================================

        self.base = SpectrumMoleculeDataset(
            stage6_dir=self.stage6_dir,
            stage8_dir=self.stage8_dir,
            split_value=self.split_value,
            precursor_scale=precursor_scale,
        )

        # =================================================
        # Required metadata
        # =================================================

        required_metadata_columns = {
            "precursor_mz",
            "adduct",
            "instrument_type",
            "ingest_lib",
            "inchikey14",
        }

        missing = (
            required_metadata_columns
            - set(
                train_metadata.columns
            )
        )

        if missing:
            raise ValueError(
                "train_metadata is missing columns: "
                f"{sorted(missing)}"
            )

        required_structure_columns = {
            "structure_index",
            "inchikey14",
        }

        missing_structure = (
            required_structure_columns
            - set(
                structure_map.columns
            )
        )

        if missing_structure:
            raise ValueError(
                "structure_map is missing columns: "
                f"{sorted(missing_structure)}"
            )

        # =================================================
        # Stage 6 cache -> original parquet row
        # =================================================

        stage6_row_indices = np.load(
            self.stage6_dir
            / "row_indices.npy",
            mmap_mode="r",
        )

        self.original_row_indices = np.asarray(
            stage6_row_indices[
                self.base.indices
            ],
            dtype=np.int64,
        )

        # =================================================
        # Metadata for this split only
        # =================================================

        split_metadata = (
            train_metadata
            .iloc[
                self.original_row_indices
            ]
            .reset_index(
                drop=True
            )
        )

        self.precursor_mz = (
            split_metadata[
                "precursor_mz"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        self.adducts = (
            split_metadata[
                "adduct"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        self.instrument_types = (
            split_metadata[
                "instrument_type"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        self.ingest_libs = (
            split_metadata[
                "ingest_lib"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        self.true_inchikey14 = (
            split_metadata[
                "inchikey14"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        # =================================================
        # Local dataset position -> Stage 8 molecule index
        # =================================================

        global_cache_indices = (
            self.base.indices
        )

        self.molecule_indices = np.asarray(
            self.base.spectrum_molecule_indices[
                global_cache_indices
            ],
            dtype=np.int64,
        )

        # =================================================
        # Stage 8 structure mappings
        # =================================================

        structure_map = (
            structure_map
            .sort_values(
                "structure_index"
            )
            .reset_index(
                drop=True
            )
        )

        self.num_structures = int(
            len(
                structure_map
            )
        )

        self.inchikey_to_structure_index = dict(
            zip(
                structure_map[
                    "inchikey14"
                ].astype(str),
                structure_map[
                    "structure_index"
                ].astype(int),
            )
        )

        # =================================================
        # Determine hard-filter eligible rows
        #
        # Existing Stage 7 policy:
        #
        #   supported adduct
        #   AND trusted adduct
        #   AND (
        #       high-confidence source
        #       OR timsTOF
        #   )
        #
        # These rows use the 10 ppm hard filter.
        # =================================================

        print(
            "Determining chemistry-eligible rows..."
        )

        unique_adducts = np.unique(
            self.adducts
        )

        supported_lookup = {}

        trusted_lookup = {}

        for adduct in unique_adducts:

            supported = (
                is_supported_adduct(
                    adduct
                )
            )

            trusted = (
                is_trusted_for_mass_filter(
                    adduct
                )
                if supported
                else False
            )

            supported_lookup[
                adduct
            ] = supported

            trusted_lookup[
                adduct
            ] = trusted

        supported_mask = np.fromiter(
            (
                supported_lookup[
                    value
                ]
                for value
                in self.adducts
            ),
            dtype=np.bool_,
            count=len(
                self.adducts
            ),
        )

        trusted_mask = np.fromiter(
            (
                trusted_lookup[
                    value
                ]
                for value
                in self.adducts
            ),
            dtype=np.bool_,
            count=len(
                self.adducts
            ),
        )

        source_mask = np.fromiter(
            (
                source
                in HIGH_CONFIDENCE_SOURCES
                for source
                in self.ingest_libs
            ),
            dtype=np.bool_,
            count=len(
                self.ingest_libs
            ),
        )

        timstof_mask = np.fromiter(
            (
                instrument
                .strip()
                .lower()
                == "timstof"
                for instrument
                in self.instrument_types
            ),
            dtype=np.bool_,
            count=len(
                self.instrument_types
            ),
        )

        finite_precursor = np.isfinite(
            self.precursor_mz
        )

        positive_precursor = (
            self.precursor_mz
            > 0
        )

        eligible_mask = (
            supported_mask
            & trusted_mask
            & (
                source_mask
                | timstof_mask
            )
            & finite_precursor
            & positive_precursor
        )

        self.eligible_positions = np.flatnonzero(
            eligible_mask
        ).astype(
            np.int64
        )

        self.eligible_mask = eligible_mask

        # =================================================
        # Structures having at least one eligible spectrum
        # =================================================

        eligible_molecules = (
            self.molecule_indices[
                self.eligible_positions
            ]
        )

        self.structure_has_eligible = np.zeros(
            self.num_structures,
            dtype=np.bool_,
        )

        self.structure_has_eligible[
            np.unique(
                eligible_molecules
            )
        ] = True

        # =================================================
        # Efficient structure -> spectrum-position lookup
        #
        # We avoid a giant Python dictionary of lists.
        # =================================================

        sort_order = np.argsort(
            eligible_molecules,
            kind="stable",
        )

        self.sorted_eligible_molecules = (
            eligible_molecules[
                sort_order
            ]
        )

        self.sorted_eligible_positions = (
            self.eligible_positions[
                sort_order
            ]
        )

        # =================================================
        # Stage 7 generator
        # =================================================

        self.candidate_generator = (
            AdductAwareCandidateGenerator(
                mass_index=mass_index,
                tolerance_ppm=(
                    self.tolerance_ppm
                ),
            )
        )

        print(
            f"Split spectra: "
            f"{len(self.base):,}"
        )

        print(
            f"Chemistry-eligible spectra: "
            f"{len(self.eligible_positions):,}"
        )

        print(
            f"Eligible structures: "
            f"{self.structure_has_eligible.sum():,}"
        )

    # =====================================================
    # Dataset API
    # =====================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.base
        )

    def __getitem__(
        self,
        index: int,
    ):

        return self.base[
            index
        ]

    # =====================================================
    # Structure -> eligible spectrum position
    # =====================================================

    def sample_position_for_structure(
        self,
        structure_index: int,
        rng: np.random.Generator,
    ) -> int | None:

        structure_index = int(
            structure_index
        )

        if (
            structure_index < 0
            or structure_index
            >= self.num_structures
        ):
            return None

        if not self.structure_has_eligible[
            structure_index
        ]:
            return None

        left = np.searchsorted(
            self.sorted_eligible_molecules,
            structure_index,
            side="left",
        )

        right = np.searchsorted(
            self.sorted_eligible_molecules,
            structure_index,
            side="right",
        )

        if right <= left:
            return None

        chosen = int(
            rng.integers(
                left,
                right,
            )
        )

        return int(
            self.sorted_eligible_positions[
                chosen
            ]
        )

    # =====================================================
    # Generate mass-compatible structure set
    # =====================================================

    def candidate_structures_for_position(
        self,
        position: int,
    ) -> np.ndarray:

        position = int(
            position
        )

        if not self.eligible_mask[
            position
        ]:
            return np.empty(
                0,
                dtype=np.int64,
            )

        precursor_mz = float(
            self.precursor_mz[
                position
            ]
        )

        adduct = str(
            self.adducts[
                position
            ]
        )

        result = (
            self.candidate_generator.generate(
                precursor_mz=precursor_mz,
                adduct=adduct,
                require_trusted=True,
            )
        )

        structure_indices = []

        for hit in result.candidates:

            structure_index = (
                self.inchikey_to_structure_index
                .get(
                    hit.inchikey14
                )
            )

            if structure_index is None:
                continue

            structure_index = int(
                structure_index
            )

            if not self.structure_has_eligible[
                structure_index
            ]:
                continue

            structure_indices.append(
                structure_index
            )

        if not structure_indices:

            return np.empty(
                0,
                dtype=np.int64,
            )

        return np.unique(
            np.asarray(
                structure_indices,
                dtype=np.int64,
            )
        )


class CandidateAwareBatchSampler(
    Sampler[list[int]]
):
    """
    Build batches using one precursor-derived candidate window.

    Each batch contains one eligible spectrum from several
    distinct mass-compatible molecular structures.

    Therefore all structures act as chemically difficult
    negatives for the other samples in the batch.
    """

    def __init__(
        self,
        dataset: CandidateAwareSpectrumDataset,
        batch_size: int = 32,
        batches_per_epoch: int = 3000,
        min_batch_size: int = 8,
        seed: int = 42,
        max_attempt_multiplier: int = 50,
    ) -> None:

        if batch_size < 2:
            raise ValueError(
                "batch_size must be >= 2."
            )

        if min_batch_size < 2:
            raise ValueError(
                "min_batch_size must be >= 2."
            )

        if min_batch_size > batch_size:
            raise ValueError(
                "min_batch_size cannot exceed batch_size."
            )

        if batches_per_epoch <= 0:
            raise ValueError(
                "batches_per_epoch must be > 0."
            )

        self.dataset = dataset

        self.batch_size = int(
            batch_size
        )

        self.batches_per_epoch = int(
            batches_per_epoch
        )

        self.min_batch_size = int(
            min_batch_size
        )

        self.seed = int(
            seed
        )

        self.max_attempt_multiplier = int(
            max_attempt_multiplier
        )

        self.epoch = 0

    def set_epoch(
        self,
        epoch: int,
    ) -> None:

        self.epoch = int(
            epoch
        )

    def __len__(
        self,
    ) -> int:

        return self.batches_per_epoch

    def __iter__(
        self,
    ) -> Iterator[list[int]]:

        rng = np.random.default_rng(
            self.seed
            + self.epoch
            * 100_003
        )

        eligible_positions = (
            self.dataset.eligible_positions
        )

        generated = 0

        attempts = 0

        max_attempts = (
            self.batches_per_epoch
            * self.max_attempt_multiplier
        )

        while (
            generated
            < self.batches_per_epoch
        ):

            attempts += 1

            if attempts > max_attempts:

                raise RuntimeError(
                    "Unable to generate enough "
                    "candidate-aware batches. "
                    "Consider reducing min_batch_size "
                    "or batch_size."
                )

            anchor_position = int(
                eligible_positions[
                    rng.integers(
                        0,
                        len(
                            eligible_positions
                        ),
                    )
                ]
            )

            anchor_structure = int(
                self.dataset.molecule_indices[
                    anchor_position
                ]
            )

            candidate_structures = (
                self.dataset
                .candidate_structures_for_position(
                    anchor_position
                )
            )

            if (
                len(
                    candidate_structures
                )
                < self.min_batch_size
            ):
                continue

            # ---------------------------------------------
            # Always keep the anchor positive.
            # ---------------------------------------------

            candidate_structures = (
                candidate_structures[
                    candidate_structures
                    != anchor_structure
                ]
            )

            rng.shuffle(
                candidate_structures
            )

            selected_structures = [
                anchor_structure
            ]

            remaining = (
                self.batch_size
                - 1
            )

            selected_structures.extend(
                candidate_structures[
                    :remaining
                ].tolist()
            )

            # ---------------------------------------------
            # Select one eligible spectrum per structure.
            # ---------------------------------------------

            batch_positions = [
                anchor_position
            ]

            for structure_index in (
                selected_structures[
                    1:
                ]
            ):

                position = (
                    self.dataset
                    .sample_position_for_structure(
                        structure_index,
                        rng,
                    )
                )

                if position is not None:

                    batch_positions.append(
                        position
                    )

            if (
                len(
                    batch_positions
                )
                < self.min_batch_size
            ):
                continue

            # Ensure structures really are unique.

            molecules = (
                self.dataset.molecule_indices[
                    np.asarray(
                        batch_positions,
                        dtype=np.int64,
                    )
                ]
            )

            if (
                len(
                    np.unique(
                        molecules
                    )
                )
                != len(
                    molecules
                )
            ):
                continue

            generated += 1

            yield batch_positions