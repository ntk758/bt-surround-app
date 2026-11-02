"""
SOFA (Spatially Oriented Format for Acoustics) Loader.

Loads, validates, and normalizes AES69-compliant SOFA files (specifically
SimpleFreeFieldHRIR conventions) into immutable HRTFDataset objects.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type
import numpy as np
import h5py

from sofa_exceptions import (
    SOFAError,
    SOFAFileError,
    SOFAValidationError,
    SOFAConventionError,
)
from hrtf_dataset import HRTFDataset


def _decode_attr(val: Any) -> str:
    """Safely decode HDF5 attribute value to unicode string."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (np.bytes_, np.str_)):
        return str(val).strip()
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return _decode_attr(val.item())
        return ", ".join(_decode_attr(x) for x in val.flatten())
    return str(val).strip()


class SOFALoader:
    """
    Reader and validator for SOFA (AES69) HRIR files.

    Enforces strict metadata validation, dimensionality checks, and numerical
    integrity (no NaN/Inf) before constructing an HRTFDataset.
    """

    SUPPORTED_CONVENTIONS = {
        "SimpleFreeFieldHRIR",
    }

    SUPPORTED_DATA_TYPES = {
        "FIR",
    }

    def __init__(
        self,
        strict_conventions: bool = True,
        strict_coordinates: bool = True,
    ) -> None:
        """
        Initialize the SOFALoader.

        Args:
            strict_conventions: If True, reject conventions not in SUPPORTED_CONVENTIONS.
            strict_coordinates: If True, require SourcePosition and ReceiverPosition.
        """
        self.strict_conventions = strict_conventions
        self.strict_coordinates = strict_coordinates

    def load(
        self,
        filepath: str,
        dtype: Type[np.floating] = np.float32,
    ) -> HRTFDataset:
        """
        Load and validate a SOFA file.

        Args:
            filepath: Path to the .sofa file.
            dtype: Target floating point type for IR data (np.float32 or np.float64).

        Returns:
            HRTFDataset: Populated and validated dataset.

        Raises:
            SOFAFileError: If file does not exist, cannot be opened, or is not valid HDF5.
            SOFAValidationError: If required datasets, metadata, or finite checks fail.
            SOFAConventionError: If convention or data type is unsupported.
        """
        if not os.path.exists(filepath):
            raise SOFAFileError(f"SOFA file not found: {filepath}")

        if not os.path.isfile(filepath):
            raise SOFAFileError(f"Path is not a regular file: {filepath}")

        try:
            is_hdf5 = h5py.is_hdf5(filepath)
        except Exception as e:
            raise SOFAFileError(f"Error checking HDF5 signature for {filepath}: {e}") from e

        if not is_hdf5:
            raise SOFAFileError(f"File is not a valid HDF5 container: {filepath}")

        try:
            with h5py.File(filepath, "r") as h5_file:
                metadata = self._extract_metadata(h5_file)
                self._validate_conventions(metadata, h5_file)

                ir_data = self._load_data_ir(h5_file, dtype=dtype)
                sampling_rate = self._load_sampling_rate(h5_file)
                source_positions = self._load_source_positions(h5_file)
                receiver_positions = self._load_receiver_positions(h5_file)
                listener_info = self._load_listener_info(h5_file)

                # Cross-validate dimensions
                m_ir, r_ir, n_ir = ir_data.shape
                m_pos = source_positions.shape[0]

                if m_ir != m_pos:
                    raise SOFAValidationError(
                        f"Measurement count mismatch: Data.IR has M={m_ir}, "
                        f"SourcePosition has M={m_pos}"
                    )

                if receiver_positions is not None:
                    # In SOFA, ReceiverPosition shape can be [R, C] or [R, C, I] or [1, C, R]
                    r_rec = receiver_positions.shape[0]
                    if r_rec != r_ir and receiver_positions.ndim >= 3 and receiver_positions.shape[2] == r_ir:
                        # Sometimes [1, C, R]
                        pass
                    elif r_rec != r_ir:
                        # Warning or error
                        pass

                metadata.update(listener_info)
                metadata["receiver_positions"] = receiver_positions

                # Dataset and Subject identification
                database_name = metadata.get("DatabaseName") or metadata.get("Title") or os.path.basename(filepath)
                subject_id = metadata.get("ListenerShortName") or metadata.get("Subject") or "DefaultSubject"

                return HRTFDataset(
                    name=database_name,
                    subject_id=subject_id,
                    sampling_rate=sampling_rate,
                    ir_data=ir_data,
                    source_positions=source_positions,
                    receiver_count=r_ir,
                    ir_length=n_ir,
                    metadata=metadata,
                )
        except (SOFAFileError, SOFAValidationError, SOFAConventionError):
            raise
        except Exception as e:
            raise SOFAValidationError(f"Unexpected error loading SOFA file {filepath}: {e}") from e

    def _extract_metadata(self, h5_file: h5py.File) -> Dict[str, Any]:
        """Extract all global attributes from root."""
        meta: Dict[str, Any] = {}
        for k, v in h5_file.attrs.items():
            meta[k] = _decode_attr(v)
        return meta

    def _validate_conventions(self, metadata: Dict[str, Any], h5_file: h5py.File) -> None:
        """Validate SOFAConventions and DataType."""
        convention = metadata.get("SOFAConventions")
        if not convention:
            raise SOFAConventionError("Missing required global attribute: 'SOFAConventions'")

        if self.strict_conventions and convention not in self.SUPPORTED_CONVENTIONS:
            raise SOFAConventionError(
                f"Unsupported SOFA convention '{convention}'. "
                f"Supported: {sorted(list(self.SUPPORTED_CONVENTIONS))}"
            )

        data_type = metadata.get("DataType")
        if not data_type:
            # Check if Data.IR exists, then assume FIR
            if "Data.IR" in h5_file:
                data_type = "FIR"
            else:
                raise SOFAValidationError("Missing required global attribute: 'DataType'")

        if self.strict_conventions and data_type not in self.SUPPORTED_DATA_TYPES:
            raise SOFAConventionError(
                f"Unsupported DataType '{data_type}'. Supported: {sorted(list(self.SUPPORTED_DATA_TYPES))}"
            )

    def _load_data_ir(self, h5_file: h5py.File, dtype: Type[np.floating]) -> np.ndarray:
        """Load, shape-check, and finite-check Data.IR."""
        if "Data.IR" not in h5_file:
            raise SOFAValidationError("Missing required dataset: 'Data.IR'")

        ds = h5_file["Data.IR"]
        if ds.ndim != 3:
            raise SOFAValidationError(
                f"Data.IR must have exactly 3 dimensions [M, R, N], got {ds.ndim}D with shape {ds.shape}"
            )

        m, r, n = ds.shape
        if m <= 0:
            raise SOFAValidationError(f"Data.IR measurement count M must be > 0, got {m}")
        if r <= 0:
            raise SOFAValidationError(f"Data.IR receiver count R must be > 0, got {r}")
        if n <= 0:
            raise SOFAValidationError(f"Data.IR sample length N must be > 0, got {n}")

        # Read into memory
        ir_array = np.array(ds, dtype=dtype)

        # Check for NaN / Inf
        if not np.all(np.isfinite(ir_array)):
            raise SOFAValidationError("Data.IR contains NaN or Inf values")

        return ir_array

    def _load_sampling_rate(self, h5_file: h5py.File) -> float:
        """Load and validate Data.SamplingRate."""
        if "Data.SamplingRate" not in h5_file:
            raise SOFAValidationError("Missing required dataset: 'Data.SamplingRate'")

        ds = h5_file["Data.SamplingRate"]
        val = np.array(ds).squeeze()

        if val.ndim != 0:
            raise SOFAValidationError(
                f"Data.SamplingRate must be a scalar, got shape {ds.shape}"
            )

        fs = float(val.item())
        if not np.isfinite(fs) or fs <= 0.0:
            raise SOFAValidationError(f"Data.SamplingRate must be finite and positive, got {fs}")

        return fs

    def _load_source_positions(self, h5_file: h5py.File) -> np.ndarray:
        """Load, validate, and check SourcePosition."""
        if "SourcePosition" not in h5_file:
            raise SOFAValidationError("Missing required dataset: 'SourcePosition'")

        ds = h5_file["SourcePosition"]
        if ds.ndim != 2:
            raise SOFAValidationError(
                f"SourcePosition must have 2 dimensions [M, C], got shape {ds.shape}"
            )

        m, c = ds.shape
        if c < 3:
            raise SOFAValidationError(
                f"SourcePosition must have at least 3 coordinate columns [az, el, r], got {c}"
            )

        pos = np.array(ds[:, :3], dtype=np.float64)

        if not np.all(np.isfinite(pos)):
            raise SOFAValidationError("SourcePosition contains NaN or Inf values")

        # AES69 では直交座標も許可されている。HRTFDataset は [az, el, r] (度, 度, m) に統一。
        coord_type = _decode_attr(ds.attrs.get("Type", "spherical")).lower()
        if coord_type == "cartesian":
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            az = np.degrees(np.arctan2(y, x))
            el = np.degrees(np.arctan2(z, np.hypot(x, y)))
            pos = np.column_stack([az, el, np.sqrt(x * x + y * y + z * z)])
        elif coord_type != "spherical":
            raise SOFAValidationError(f"Unsupported SourcePosition:Type '{coord_type}'")

        return pos

    def _load_receiver_positions(self, h5_file: h5py.File) -> Optional[np.ndarray]:
        """Load ReceiverPosition if present."""
        if "ReceiverPosition" not in h5_file:
            if self.strict_coordinates:
                raise SOFAValidationError("Missing required dataset: 'ReceiverPosition'")
            return None

        ds = h5_file["ReceiverPosition"]
        pos = np.array(ds, dtype=np.float64)

        if not np.all(np.isfinite(pos)):
            raise SOFAValidationError("ReceiverPosition contains NaN or Inf values")

        return pos

    def _load_listener_info(self, h5_file: h5py.File) -> Dict[str, Any]:
        """Load optional ListenerPosition, ListenerView, ListenerUp."""
        info: Dict[str, Any] = {}
        for key in ["ListenerPosition", "ListenerView", "ListenerUp"]:
            if key in h5_file:
                arr = np.array(h5_file[key], dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    raise SOFAValidationError(f"{key} contains NaN or Inf values")
                info[key] = arr
                # Attributes
                for attr_k, attr_v in h5_file[key].attrs.items():
                    info[f"{key}:{attr_k}"] = _decode_attr(attr_v)
        return info
