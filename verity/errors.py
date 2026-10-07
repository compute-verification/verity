"""The exception hierarchy shared by the core modules (promoted from veritor ``core/errors.py``, ``protocol/messages.py``).

``InvalidArtifact`` is raised for an encoding outside a modelled domain or a malformed protocol object;
``ProtocolError`` for a misuse of a protocol API (not a verdict); ``ResourceLimit`` when a verification limit
is exceeded. Verdict codes (``verity_vllm.check.codes``) build on ``ProtocolError``.
"""

from __future__ import annotations


class InvalidArtifact(ValueError):
    """An encoding outside the modelled domain, a malformed artifact, or a shape the pipeline does not take."""


class ProtocolError(InvalidArtifact):
    """A party used a protocol API incorrectly (not a verdict)."""


class ResourceLimit(InvalidArtifact):
    """A named verification limit was exceeded."""

    def __init__(self, resource: str, *, limit: int, observed: int) -> None:
        super().__init__(f"{resource} limit exceeded: observed {observed} > {limit}")
        self.resource = resource
        self.limit = limit
        self.observed = observed
