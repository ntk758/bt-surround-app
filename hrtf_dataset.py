"""
HRTF Dataset Container.

Provides an immutable, memory-efficient, and mathematically validated container
for Spatially Oriented Format for Acoustics (SOFA) HRIR datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple
import numpy as np


@dataclass(frozen=True)
class HRTFDataset:
    """
    Normalized, immutable container for HRTF/HRIR data loaded from SOFA files.

    Attributes:
        name: Name of the database / dataset (e.g. 'SADIE II', 'HUTUBS').
        subject_id: Subject identifier (e.g. 'D2', 'Subject 001').
        sampling_rate: Audio sampling rate in Hertz (e.g. 48000.0).
        ir_data: Measured impulse response array of shape [M, R, N]
                 M = measurement count, R = receiver count, N = IR length in samples.
        source_positions: Emitter/source position array of shape [M, 3].
        receiver_count: Number of receivers (channels, typically 2 for ears).
        ir_length: Impulse response length in samples (e.g. 256).
        metadata: Dictionary containing validated SOFA attributes and conventions.
    """
    name: str
    subject_id: str
    sampling_rate: float
    ir_data: np.ndarray
    source_positions: np.ndarray
    receiver_count: int
    ir_length: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce internal consistency and read-only protection."""
        if not isinstance(self.ir_data, np.ndarray):
            raise TypeError("ir_data must be a numpy.ndarray")
        if not isinstance(self.source_positions, np.ndarray):
            raise TypeError("source_positions must be a numpy.ndarray")
        if self.ir_data.ndim != 3:
            raise ValueError(f"ir_data must be 3-dimensional [M, R, N], got shape {self.ir_data.shape}")
        if self.source_positions.ndim != 2 or self.source_positions.shape[1] < 3:
            raise ValueError(f"source_positions must have shape [M, 3], got shape {self.source_positions.shape}")
        if self.ir_data.shape[0] != self.source_positions.shape[0]:
            raise ValueError(
                f"Measurement count mismatch: ir_data has M={self.ir_data.shape[0]} "
                f"but source_positions has M={self.source_positions.shape[0]}"
            )
        if self.ir_data.shape[1] != self.receiver_count:
            raise ValueError(
                f"Receiver count mismatch: ir_data has R={self.ir_data.shape[1]} "
                f"but receiver_count={self.receiver_count}"
            )
        if self.ir_data.shape[2] != self.ir_length:
            raise ValueError(
                f"IR length mismatch: ir_data has N={self.ir_data.shape[2]} "
                f"but ir_length={self.ir_length}"
            )

        # Make underlying arrays read-only to guarantee immutability
        self.ir_data.flags.writeable = False
        self.source_positions.flags.writeable = False

    @property
    def measurement_count(self) -> int:
        """Total number of measurement points (M)."""
        return int(self.ir_data.shape[0])

    @property
    def is_48k(self) -> bool:
        """True if sampling rate matches 48,000 Hz within 1 Hz tolerance."""
        return bool(abs(self.sampling_rate - 48000.0) < 1.0)

    @property
    def azimuth_range(self) -> Tuple[float, float]:
        """(min_azimuth, max_azimuth) in degrees as recorded in source_positions."""
        az = self.source_positions[:, 0]
        return (float(np.min(az)), float(np.max(az)))

    @property
    def elevation_range(self) -> Tuple[float, float]:
        """(min_elevation, max_elevation) in degrees as recorded in source_positions."""
        el = self.source_positions[:, 1]
        return (float(np.min(el)), float(np.max(el)))

    @property
    def distance_range(self) -> Tuple[float, float]:
        """(min_distance, max_distance) in meters as recorded in source_positions."""
        dist = self.source_positions[:, 2]
        return (float(np.min(dist)), float(np.max(dist)))

    def get_ir(self, measurement_idx: int) -> np.ndarray:
        """
        Get impulse response for a specific measurement index.

        Returns:
            np.ndarray of shape [R, N] (e.g. [2, 256]).
        """
        if measurement_idx < 0 or measurement_idx >= self.measurement_count:
            raise IndexError(f"measurement_idx {measurement_idx} out of range [0, {self.measurement_count - 1}]")
        return self.ir_data[measurement_idx]

    def get_source_position(self, measurement_idx: int) -> np.ndarray:
        """
        Get source position vector [az, el, r] for a specific measurement index.

        Returns:
            np.ndarray of shape [3].
        """
        if measurement_idx < 0 or measurement_idx >= self.measurement_count:
            raise IndexError(f"measurement_idx {measurement_idx} out of range [0, {self.measurement_count - 1}]")
        return self.source_positions[measurement_idx, :3]

    def get_summary(self) -> str:
        """
        Return structured human-readable summary of the dataset.
        Format matches section 8 of technical requirements.
        """
        az_min, az_max = self.azimuth_range
        el_min, el_max = self.elevation_range
        d_min, d_max = self.distance_range
        native_str = "48k native" if self.is_48k else f"non-48k ({self.sampling_rate:.0f} Hz)"

        lines = [
            f"Dataset:        {self.name}",
            f"Subject:        {self.subject_id}",
            f"Measurements:   {self.measurement_count}",
            f"Receivers:      {self.receiver_count}",
            f"IR length:      {self.ir_length} samples ({self.ir_length / self.sampling_rate * 1000.0:.2f} ms)",
            f"Sampling rate:  {self.sampling_rate:.1f} Hz [{native_str}]",
            f"Azimuth range:  [{az_min:.1f}°, {az_max:.1f}°]",
            f"Elevation range:[{el_min:.1f}°, {el_max:.1f}°]",
            f"Distance range: [{d_min:.2f} m, {d_max:.2f} m]",
            f"Dtype:          {self.ir_data.dtype}",
            f"Memory (IR):    {self.ir_data.nbytes / (1024 * 1024):.2f} MB"
        ]
        return "\n".join(lines)
