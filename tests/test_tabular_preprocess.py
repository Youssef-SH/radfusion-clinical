from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from radfusion.data.schemas import RSNA_SAMPLE_SCHEMA, RSNA_SPLIT_SCHEMA
from radfusion.data.tabular_preprocess import (
    SOURCE_FEATURES,
    RsnaMetadataFeatures,
    fit_rsna_preprocessor,
    load_preprocessor,
    save_preprocessor,
)


def _tables() -> tuple[pa.Table, pa.Table]:
    samples = []
    splits = []
    values = (
        ("train-a", 10.0, "F", "PA", 0.1, "train"),
        ("train-b", 20.0, None, "AP", 0.2, "train"),
        ("validation", 100.0, "M", "LL", 5.0, "validation"),
    )
    for name, age, sex, view, spacing, split_name in values:
        sample_id = f"rsna:{name}"
        samples.append(
            {
                "sample_id": sample_id,
                "patient_id": name,
                "image_id": name,
                "image_path": f"stage_2_train_images/{name}.dcm",
                "image_rows": 1024,
                "image_columns": 1024,
                "age_years": age,
                "age_is_implausible": False,
                "sex": sex,
                "view_position": view,
                "pixel_spacing_row_mm": spacing,
                "pixel_spacing_col_mm": spacing,
            }
        )
        splits.append(
            {
                "sample_id": sample_id,
                "split_name": split_name,
            }
        )
    return (
        pa.Table.from_pylist(samples, RSNA_SAMPLE_SCHEMA),
        pa.Table.from_pylist(splits, RSNA_SPLIT_SCHEMA),
    )


def test_preprocessor_fits_statistics_and_categories_on_training_only(tmp_path) -> None:
    samples, splits = _tables()
    preprocessor = fit_rsna_preprocessor(samples, splits)
    columns = preprocessor.named_steps["columns"]
    numeric = columns.named_transformers_["continuous"]
    categorical = columns.named_transformers_["categorical"]

    assert numeric.named_steps["impute"].statistics_[0] == 15.0
    categories = categorical.named_steps["encode"].categories_
    assert "M" not in categories[0]
    assert "LL" not in categories[1]
    transformed = preprocessor.transform(samples.to_pandas().loc[:, SOURCE_FEATURES])
    assert np.isfinite(transformed.to_numpy()).all()

    destination = save_preprocessor(preprocessor, tmp_path / "preprocessor.skops")
    restored = load_preprocessor(destination)
    np.testing.assert_allclose(
        restored.transform(samples.to_pandas().loc[:, SOURCE_FEATURES]),
        transformed,
    )


def test_metadata_features_clip_age_and_add_missingness() -> None:
    frame = pd.DataFrame(
        {
            "age_years": [155.0, None],
            "age_is_implausible": [True, False],
            "sex": ["F", None],
            "view_position": ["PA", None],
            "pixel_spacing_row_mm": [0.1, None],
            "pixel_spacing_col_mm": [0.1, None],
        }
    )
    transformed = RsnaMetadataFeatures().fit_transform(frame)

    assert transformed["age_model_years"].iloc[0] == 120.0
    assert transformed["age_is_implausible"].tolist() == [1, 0]
    assert transformed["age_years_missing"].tolist() == [0, 1]
    assert transformed["sex_missing"].tolist() == [0, 1]


def test_metadata_features_reject_unapproved_columns() -> None:
    frame = pd.DataFrame(
        {
            "age_years": [50.0],
            "age_is_implausible": [False],
            "sex": ["F"],
            "view_position": ["PA"],
            "pixel_spacing_row_mm": [0.1],
            "pixel_spacing_col_mm": [0.1],
            "patient_id": ["poison"],
        }
    )

    with np.testing.assert_raises_regex(ValueError, "unexpected columns"):
        RsnaMetadataFeatures().fit_transform(frame)
