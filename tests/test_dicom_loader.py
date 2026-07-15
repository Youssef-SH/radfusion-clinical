import numpy as np
import pytest
from pydicom.data import get_testdata_file

from radfusion.data.dicom_loader import read_dicom


def test_read_dicom_returns_normalized_2d_pixels() -> None:
    sample_path = get_testdata_file("CT_small.dcm")

    pixels, record = read_dicom(sample_path)

    assert pixels.ndim == 2
    assert pixels.dtype == np.float32
    assert float(pixels.min()) >= 0.0
    assert float(pixels.max()) <= 1.0
    assert record.rows == pixels.shape[0]
    assert record.columns == pixels.shape[1]


def test_read_dicom_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_dicom("does-not-exist.dcm")