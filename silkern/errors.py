"""The single exception type raised by every public entry point."""

from __future__ import annotations


class LocalizationError(ValueError):
    """An invalid localization configuration, geometry, or buffer binding.

    Every public entry point fails closed by raising this rather than silently
    degrading: a rejected call is recoverable, a wrong index is not.
    """
