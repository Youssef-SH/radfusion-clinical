"""Build train-fitted preprocessing for RSNA DICOM metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from radfusion.data.rsna_source import ManifestBuildError
from radfusion.utils.skops_io import load_skops, save_skops

CONTINUOUS_FEATURES = (
    "age_model_years",
    "pixel_spacing_row_mm",
    "pixel_spacing_col_mm",
)
CATEGORICAL_FEATURES = ("sex", "view_position")
BINARY_FEATURES = (
    "age_is_implausible",
    "age_years_missing",
    "sex_missing",
    "view_position_missing",
    "pixel_spacing_row_mm_missing",
    "pixel_spacing_col_mm_missing",
)
SOURCE_FEATURES = (
    "age_years",
    "age_is_implausible",
    "sex",
    "view_position",
    "pixel_spacing_row_mm",
    "pixel_spacing_col_mm",
)
METADATA_INPUT_POLICY_VERSION = "rsna-metadata-input-v1"
_SOURCE_FEATURE_CONTRACT = {
    "age_years": ("numeric", "allowed"),
    "age_is_implausible": ("boolean", "forbidden"),
    "sex": ("categorical_string", "allowed"),
    "view_position": ("categorical_string", "allowed"),
    "pixel_spacing_row_mm": ("numeric", "allowed"),
    "pixel_spacing_col_mm": ("numeric", "allowed"),
}


def metadata_input_contract() -> dict[str, Any]:
    """Return the exact raw-input contract for serialized metadata pipelines."""
    return {
        "policy_version": METADATA_INPUT_POLICY_VERSION,
        "features": [
            {
                "name": name,
                "type_category": _SOURCE_FEATURE_CONTRACT[name][0],
                "missing_values": _SOURCE_FEATURE_CONTRACT[name][1],
            }
            for name in SOURCE_FEATURES
        ],
        "fitted_preprocessing_embedded": True,
    }


class RsnaMetadataFeatures(BaseEstimator, TransformerMixin):
    """Derive bounded age and explicit metadata missingness indicators."""

    def fit(self, features: pd.DataFrame, target: object = None) -> RsnaMetadataFeatures:
        """Validate the input columns."""
        self._validate_columns(features)
        self.feature_names_in_ = np.asarray(SOURCE_FEATURES, dtype=object)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Derive model-ready metadata columns from a copy of the input."""
        self._validate_columns(features)
        transformed = features.loc[:, SOURCE_FEATURES].copy()
        age = pd.to_numeric(transformed["age_years"], errors="coerce")
        transformed["age_model_years"] = age.clip(lower=0.0, upper=120.0)
        transformed["age_is_implausible"] = transformed["age_is_implausible"].astype("int8")
        for column in (
            "age_years",
            "sex",
            "view_position",
            "pixel_spacing_row_mm",
            "pixel_spacing_col_mm",
        ):
            transformed[f"{column}_missing"] = transformed[column].isna().astype("int8")
        return transformed.loc[:, [*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES, *BINARY_FEATURES]]

    @staticmethod
    def _validate_columns(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("RSNA metadata preprocessing requires a pandas DataFrame")
        if len(features.columns) != len(set(features.columns)):
            raise ValueError("RSNA metadata contains duplicate columns")
        missing = sorted(set(SOURCE_FEATURES) - set(features.columns))
        if missing:
            raise ValueError(f"RSNA metadata is missing required columns: {missing}")
        unexpected = sorted(set(features.columns) - set(SOURCE_FEATURES))
        if unexpected:
            raise ValueError(f"RSNA metadata contains unexpected columns: {unexpected}")
        if tuple(features.columns) != SOURCE_FEATURES:
            raise ValueError("RSNA metadata columns are not in the required order")


def validate_metadata_pipeline(model: object) -> Pipeline:
    """Validate that a loaded estimator embeds the fitted metadata input pipeline."""
    if not isinstance(model, Pipeline):
        raise ValueError("Serialized metadata model must be a scikit-learn pipeline")
    preprocessor = model.named_steps.get("preprocess")
    if not isinstance(preprocessor, Pipeline):
        raise ValueError("Serialized metadata model is missing its preprocessing pipeline")
    metadata = preprocessor.named_steps.get("metadata")
    if not isinstance(metadata, RsnaMetadataFeatures):
        raise ValueError("Serialized metadata model has an invalid feature transformer")
    fitted_names = getattr(metadata, "feature_names_in_", None)
    if fitted_names is None or tuple(fitted_names.tolist()) != SOURCE_FEATURES:
        raise ValueError("Serialized metadata model has an invalid fitted input contract")
    return model


def build_rsna_preprocessor() -> Pipeline:
    """Return an unfitted reusable RSNA metadata preprocessing pipeline."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="<missing>")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
            ),
        ]
    )
    columns = ColumnTransformer(
        [
            ("continuous", numeric, list(CONTINUOUS_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("binary", "passthrough", list(BINARY_FEATURES)),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    ).set_output(transform="pandas")
    return Pipeline([("metadata", RsnaMetadataFeatures()), ("columns", columns)])


def fit_rsna_preprocessor(samples: pa.Table, splits: pa.Table) -> Pipeline:
    """Fit preprocessing exclusively on samples assigned to the training split."""
    sample_frame = samples.to_pandas()
    split_frame = splits.to_pandas()
    assignments = split_frame.loc[split_frame["split_name"] == "train", ["sample_id"]]
    training = sample_frame.merge(assignments, on="sample_id", validate="one_to_one")
    if training.empty:
        raise ManifestBuildError("Cannot fit metadata preprocessing without training samples")
    training = training.loc[:, SOURCE_FEATURES]
    pipeline = build_rsna_preprocessor()
    pipeline.fit(training)
    return pipeline


def save_preprocessor(preprocessor: Pipeline, path: str | Path) -> Path:
    """Serialize a fitted preprocessing pipeline with skops."""
    return save_skops(preprocessor, path)


def load_preprocessor(path: str | Path) -> Pipeline:
    """Load a fitted preprocessing pipeline from a trusted skops artifact."""
    preprocessor = load_skops(path)
    if not isinstance(preprocessor, Pipeline):
        raise TypeError("Skops artifact does not contain a preprocessing pipeline")
    return preprocessor
