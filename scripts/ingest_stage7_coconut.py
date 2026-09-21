from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

COCONUT_SDF = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "coconut"
    / "coconut_sdf_2d_lite-09-2026.sdf"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage7"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "coconut_candidates.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stage7_coconut_ingestion_summary.txt"
)


# =========================================================
# Configuration
# =========================================================

WRITE_BATCH_SIZE = 25_000

PROGRESS_INTERVAL = 50_000


# =========================================================
# Arrow schema
# =========================================================

SCHEMA = pa.schema(
    [
        (
            "source",
            pa.string(),
        ),
        (
            "source_id",
            pa.string(),
        ),
        (
            "inchikey",
            pa.string(),
        ),
        (
            "inchikey14",
            pa.string(),
        ),
        (
            "canonical_smiles",
            pa.string(),
        ),
        (
            "molecular_formula",
            pa.string(),
        ),
        (
            "exact_mass",
            pa.float64(),
        ),
        (
            "heavy_atom_count",
            pa.int32(),
        ),
        (
            "formal_charge",
            pa.int32(),
        ),
        (
            "fragment_count",
            pa.int32(),
        ),
    ]
)


def molecule_to_record(
    mol: Chem.Mol,
) -> dict | None:
    """
    Convert a sanitized RDKit molecule into the fields needed
    for the Stage 7 candidate database.
    """

    if mol is None:
        return None

    if mol.GetNumAtoms() == 0:
        return None

    try:
        canonical_smiles = (
            Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            )
        )

        inchikey = (
            Chem.MolToInchiKey(
                mol
            )
        )

        molecular_formula = (
            rdMolDescriptors.CalcMolFormula(
                mol
            )
        )

        exact_mass = float(
            Descriptors.ExactMolWt(
                mol
            )
        )

        heavy_atom_count = int(
            mol.GetNumHeavyAtoms()
        )

        formal_charge = int(
            Chem.GetFormalCharge(
                mol
            )
        )

        fragment_count = int(
            len(
                Chem.GetMolFrags(
                    mol
                )
            )
        )

    except Exception:
        return None

    if not canonical_smiles:
        return None

    if not inchikey:
        return None

    if len(
        inchikey
    ) < 14:
        return None

    if (
        not np.isfinite(
            exact_mass
        )
        or exact_mass <= 0
    ):
        return None

    source_id = ""

    if mol.HasProp(
        "identifier"
    ):
        source_id = (
            mol.GetProp(
                "identifier"
            )
            .strip()
        )

    return {
        "source": "coconut",
        "source_id": source_id,
        "inchikey": inchikey,
        "inchikey14": (
            inchikey[
                :14
            ]
        ),
        "canonical_smiles": (
            canonical_smiles
        ),
        "molecular_formula": (
            molecular_formula
        ),
        "exact_mass": (
            exact_mass
        ),
        "heavy_atom_count": (
            heavy_atom_count
        ),
        "formal_charge": (
            formal_charge
        ),
        "fragment_count": (
            fragment_count
        ),
    }


def write_batch(
    writer: pq.ParquetWriter,
    records: list[dict],
) -> None:

    if not records:
        return

    dataframe = pd.DataFrame(
        records
    )

    table = pa.Table.from_pandas(
        dataframe,
        schema=SCHEMA,
        preserve_index=False,
    )

    writer.write_table(
        table
    )


def main():

    print(
        "\nENVEDA CASMI 2026 — "
        "STAGE 7 STEP 8 "
        "COCONUT INGESTION"
    )

    start_time = time.time()

    # =====================================================
    # Input validation
    # =====================================================

    if not COCONUT_SDF.exists():

        raise FileNotFoundError(
            f"COCONUT SDF not found:\n"
            f"{COCONUT_SDF}"
        )

    file_size_gb = (
        COCONUT_SDF.stat().st_size
        / 1024**3
    )

    print(
        f"\nInput SDF:"
    )

    print(
        COCONUT_SDF
    )

    print(
        f"File size: "
        f"{file_size_gb:.2f} GB"
    )

    # =====================================================
    # Counters
    # =====================================================

    total_records = 0

    valid_molecules = 0

    invalid_molecules = 0

    duplicate_variants = 0

    multi_fragment_count = 0

    charged_count = 0

    # -----------------------------------------------------
    # Key:
    #
    #   inchikey14 + rounded exact mass
    #
    # This preserves the variant-aware design developed
    # earlier in Stage 7.
    # -----------------------------------------------------

    seen_variants = set()

    structure_keys = set()

    output_records = []

    writer = pq.ParquetWriter(
        OUTPUT_PATH,
        schema=SCHEMA,
        compression="zstd",
    )

    print(
        "\nStreaming COCONUT molecules..."
    )

    # =====================================================
    # Stream SDF
    # =====================================================

    try:

        with open(
            COCONUT_SDF,
            "rb",
        ) as handle:

            supplier = (
                Chem.ForwardSDMolSupplier(
                    handle,
                    sanitize=True,
                    removeHs=True,
                    strictParsing=False,
                )
            )

            for mol in supplier:

                total_records += 1

                if mol is None:

                    invalid_molecules += 1

                    continue

                record = (
                    molecule_to_record(
                        mol
                    )
                )

                if record is None:

                    invalid_molecules += 1

                    continue

                valid_molecules += 1

                if (
                    record[
                        "fragment_count"
                    ]
                    > 1
                ):

                    multi_fragment_count += 1

                if (
                    record[
                        "formal_charge"
                    ]
                    != 0
                ):

                    charged_count += 1

                structure_keys.add(
                    record[
                        "inchikey14"
                    ]
                )

                variant_key = (
                    record[
                        "inchikey14"
                    ],
                    round(
                        record[
                            "exact_mass"
                        ],
                        8,
                    ),
                )

                if (
                    variant_key
                    in seen_variants
                ):

                    duplicate_variants += 1

                    continue

                seen_variants.add(
                    variant_key
                )

                output_records.append(
                    record
                )

                if (
                    len(
                        output_records
                    )
                    >= WRITE_BATCH_SIZE
                ):

                    write_batch(
                        writer,
                        output_records,
                    )

                    output_records.clear()

                if (
                    total_records
                    % PROGRESS_INTERVAL
                    == 0
                ):

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    rate = (
                        total_records
                        / elapsed
                    )

                    print(
                        f"Processed "
                        f"{total_records:,} "
                        f"| valid "
                        f"{valid_molecules:,} "
                        f"| unique variants "
                        f"{len(seen_variants):,} "
                        f"| "
                        f"{rate:,.0f} mol/s"
                    )

        # =================================================
        # Final partial batch
        # =================================================

        if output_records:

            write_batch(
                writer,
                output_records,
            )

            output_records.clear()

    finally:

        writer.close()

    # =====================================================
    # Final statistics
    # =====================================================

    elapsed = (
        time.time()
        - start_time
    )

    unique_structures = len(
        structure_keys
    )

    unique_variants = len(
        seen_variants
    )

    valid_rate = (
        valid_molecules
        / total_records
        if total_records
        else 0.0
    )

    invalid_rate = (
        invalid_molecules
        / total_records
        if total_records
        else 0.0
    )

    duplicate_rate = (
        duplicate_variants
        / valid_molecules
        if valid_molecules
        else 0.0
    )

    multi_fragment_rate = (
        multi_fragment_count
        / valid_molecules
        if valid_molecules
        else 0.0
    )

    charged_rate = (
        charged_count
        / valid_molecules
        if valid_molecules
        else 0.0
    )

    output_size_mb = (
        OUTPUT_PATH.stat().st_size
        / 1024**2
    )

    # =====================================================
    # Summary file
    # =====================================================

    summary_lines = [
        (
            "ENVEDA CASMI 2026 — "
            "STAGE 7 COCONUT INGESTION"
        ),
        "",
        (
            f"Input SDF: "
            f"{COCONUT_SDF.name}"
        ),
        (
            f"Input size GB: "
            f"{file_size_gb:.3f}"
        ),
        "",
        (
            f"SDF records processed: "
            f"{total_records:,}"
        ),
        (
            f"Valid RDKit molecules: "
            f"{valid_molecules:,}"
        ),
        (
            f"Invalid molecules: "
            f"{invalid_molecules:,}"
        ),
        (
            f"Valid rate: "
            f"{valid_rate:.6%}"
        ),
        "",
        (
            f"Unique inchikey14 structures: "
            f"{unique_structures:,}"
        ),
        (
            f"Unique mass variants: "
            f"{unique_variants:,}"
        ),
        (
            f"Duplicate variants removed: "
            f"{duplicate_variants:,}"
        ),
        (
            f"Duplicate variant rate: "
            f"{duplicate_rate:.6%}"
        ),
        "",
        (
            f"Multi-fragment molecules: "
            f"{multi_fragment_count:,}"
        ),
        (
            f"Multi-fragment rate: "
            f"{multi_fragment_rate:.6%}"
        ),
        (
            f"Charged molecules: "
            f"{charged_count:,}"
        ),
        (
            f"Charged molecule rate: "
            f"{charged_rate:.6%}"
        ),
        "",
        (
            f"Output parquet MB: "
            f"{output_size_mb:.2f}"
        ),
        (
            f"Elapsed seconds: "
            f"{elapsed:.2f}"
        ),
        (
            f"Throughput mol/s: "
            f"{total_records / elapsed:,.2f}"
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Final console output
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "COCONUT INGESTION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"SDF records: "
        f"{total_records:,}"
    )

    print(
        f"Valid molecules: "
        f"{valid_molecules:,}"
    )

    print(
        f"Invalid molecules: "
        f"{invalid_molecules:,}"
    )

    print(
        f"Unique inchikey14: "
        f"{unique_structures:,}"
    )

    print(
        f"Unique mass variants: "
        f"{unique_variants:,}"
    )

    print(
        f"Duplicate variants removed: "
        f"{duplicate_variants:,}"
    )

    print(
        f"Multi-fragment molecules: "
        f"{multi_fragment_count:,}"
    )

    print(
        f"Charged molecules: "
        f"{charged_count:,}"
    )

    print(
        f"Output size: "
        f"{output_size_mb:.2f} MB"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"Throughput: "
        f"{total_records / elapsed:,.0f} mol/s"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE 7 STEP 8 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()