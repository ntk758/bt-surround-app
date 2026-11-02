"""
SOFA (Spatially Oriented Format for Acoustics) Exceptions.

Defines explicit exception hierarchy for SOFA file parsing, validation,
and integrity verification according to AES69 specifications.
"""


class SOFAError(Exception):
    """Base exception for all SOFA-related errors."""
    pass


class SOFAFileError(SOFAError):
    """Raised when a SOFA file cannot be found, opened, or is not a valid HDF5 container."""
    pass


class SOFAValidationError(SOFAError):
    """Raised when a SOFA file fails schema, metadata, dimensionality, or numeric validation."""
    pass


class SOFAConventionError(SOFAValidationError):
    """Raised when the SOFA conventions or convention version is unsupported."""
    pass
