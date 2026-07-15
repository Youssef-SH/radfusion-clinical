from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset


@dataclass(frozen=True)
class DicomRecord:
    path: str
    patient_id: str | None
    patient_age: str | None
    patient_sex: str | None
    view_position: str | None
    rows: int | None
    columns: int | None
    photometric_interpretation: str | None


def read_dicom(path: str | Path) -> tuple[np.ndarray, DicomRecord]:
    """Read a DICOM image and selected metadata."""
    dicom_path = Path(path)

    if not dicom_path.is_file():
        raise FileNotFoundError(f"DICOM file does not exist: {dicom_path}")

    dataset: FileDataset = pydicom.dcmread(dicom_path)

    try:
        pixels = dataset.pixel_array.astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Could not decode DICOM pixels: {dicom_path}") from exc

    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, received shape {pixels.shape} from {dicom_path}"
        )

    if getattr(dataset, "PhotometricInterpretation", None) == "MONOCHROME1":
        pixels = pixels.max() - pixels

    minimum = float(pixels.min())
    maximum = float(pixels.max())

    if maximum > minimum:
        pixels = (pixels - minimum) / (maximum - minimum)
    else:
        pixels = np.zeros_like(pixels, dtype=np.float32)

    record = DicomRecord(
        path=str(dicom_path),
        patient_id=_optional_string(dataset, "PatientID"),
        patient_age=_optional_string(dataset, "PatientAge"),
        patient_sex=_optional_string(dataset, "PatientSex"),
        view_position=_optional_string(dataset, "ViewPosition"),
        rows=getattr(dataset, "Rows", None),
        columns=getattr(dataset, "Columns", None),
        photometric_interpretation=_optional_string(
            dataset, "PhotometricInterpretation"
        ),
    )

    return pixels, record


def record_as_dict(record: DicomRecord) -> dict[str, Any]:
    return asdict(record)


def _optional_string(dataset: FileDataset, attribute: str) -> str | None:
    value = getattr(dataset, attribute, None)

    if value is None:
        return None

    text = str(value).strip()
    return text or None