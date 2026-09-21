"""Errors of the tensor-core semantics (promoted from veritor.core.errors; only the class the step raises)."""
from __future__ import annotations


class InvalidArtifact(ValueError):
    """An encoding outside the modelled domain, or a shape the pipeline does not take."""
