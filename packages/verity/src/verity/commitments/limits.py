"""Explicit denial-of-service limits of the verifier (promoted from veritor ``core/limits.py``; ``VerificationLimits`` only)."""

from __future__ import annotations

from dataclasses import dataclass, fields

from verity.errors import InvalidArtifact, ResourceLimit


@dataclass(frozen=True, slots=True)
class VerificationLimits:
    """Conservative default limits for parsing and pure transcript verification."""

    max_manifest_bytes: int = 1 << 20
    max_artifact_bytes: int = 64 << 20
    max_positions: int = 10_000_000
    max_units: int = 1_000_000
    max_positions_per_unit: int = 1_000_000
    max_openings: int = 10_000_000
    max_proof_bytes: int = 64 << 20
    max_transcript_bytes: int = 128 << 20
    max_nesting_depth: int = 128
    max_probability_denominator_bits: int = 64
    """Bits of the largest denominator a policy rate may have.

    Sampling and canonical encoding cost grows with the denominator size, so
    the client-proposed rates are capped here rather than by the encoder.
    """

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if type(value) is not int or value < 0:
                raise InvalidArtifact(
                    f"{descriptor.name} must be a nonnegative integer"
                )

    def enforce(self, limit_name: str, observed: int) -> None:
        """Raise ``ResourceLimit`` when ``observed`` exceeds a named limit."""

        if type(limit_name) is not str or not limit_name.startswith("max_"):
            raise ValueError("limit_name must name a VerificationLimits field")
        if type(observed) is not int or observed < 0:
            raise ValueError("observed resource use must be a nonnegative integer")
        try:
            limit = getattr(self, limit_name)
        except AttributeError as error:
            raise ValueError(f"unknown verification limit {limit_name!r}") from error
        if observed > limit:
            raise ResourceLimit(
                limit_name.removeprefix("max_"),
                limit=limit,
                observed=observed,
            )
