"""The wrapper of an ahead-of-time constructed Program, built for a `target_profile.TargetProfile`:

* `ArtifactIdentity` — the cache key / wrapper identity of a constructed artifact:
      sha256( canonical(construction_version) || canonical(TargetProfile) || canonical(applicability) || program_digest )
  The Program digest is a component, not a function of the profile: two targets that genuinely produce the same circuit share the
  Program digest and differ in identity; two hosts building the same profile share both (no host fact enters any component).
* `check_reuse(artifact, profile, applicability_now)` — fail-closed: an artifact may be reused only under the profile it was built
  for and only when every applicability condition it was built under holds for the requester; the refusal names every differing
  field with both values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from verity_vllm.program.frontend.target_profile import TargetProfile, canonical_bytes

SCHEMA = "verity/aot-artifact/v1"


@dataclass(frozen=True)
class ArtifactIdentity:
    construction_version: dict
    target_profile: dict
    applicability: dict
    program_digest: str
    identity: str = field(default="")

    @staticmethod
    def compute(construction_version: dict, target_profile: TargetProfile | dict, applicability: dict, program_digest: str) -> str:
        tp = target_profile.to_json() if isinstance(target_profile, TargetProfile) else target_profile
        h = hashlib.sha256()
        for part in (canonical_bytes(construction_version), canonical_bytes(tp), canonical_bytes(applicability), program_digest.encode()):
            h.update(len(part).to_bytes(8, "big"))  # length-prefixed: no component can absorb its neighbour
            h.update(part)
        return h.hexdigest()

    @classmethod
    def build(cls, construction_version: dict, target_profile: TargetProfile, applicability: dict, program_digest: str) -> ArtifactIdentity:
        tp = target_profile.to_json()
        return cls(construction_version=construction_version, target_profile=tp, applicability=applicability, program_digest=program_digest,
                   identity=cls.compute(construction_version, tp, applicability, program_digest))

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "identity": self.identity, "construction_version": self.construction_version, "target_profile": self.target_profile,
                "applicability": self.applicability, "program_digest": self.program_digest,
                "identity_definition": "sha256(len8||canonical(construction_version) || len8||canonical(target_profile) || len8||canonical(applicability) || len8||program_digest)"}

    @classmethod
    def from_json(cls, d: dict) -> ArtifactIdentity:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"artifact schema {d.get('schema')!r} is not {SCHEMA!r}")
        return cls(construction_version=d["construction_version"], target_profile=d["target_profile"], applicability=d["applicability"],
                   program_digest=d["program_digest"], identity=d["identity"])

    def recomputed(self) -> str:
        return self.compute(self.construction_version, self.target_profile, self.applicability, self.program_digest)


class ApplicabilityError(RuntimeError):
    """An artifact was requested outside its declared applicability; `.reasons` lists every differing field verbatim."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("artifact not applicable: " + "; ".join(self.reasons))


def reuse_mismatches(artifact: ArtifactIdentity, profile: TargetProfile, applicability_now: dict | None, construction_version_now: dict | None = None) -> list[str]:
    """Every reason `artifact` may NOT be reused by a requester declaring `profile` under `applicability_now` (the facts read back from
    the requester's environment; None = not provided = every condition unverifiable = refused)."""
    bad: list[str] = []
    if artifact.recomputed() != artifact.identity:
        bad.append(f"identity {artifact.identity} does not match its components (recomputed {artifact.recomputed()})")
    want, have = profile.to_json(), dict(artifact.target_profile)
    for k in sorted(set(want) | set(have)):
        if want.get(k, "<absent>") != have.get(k, "<absent>"):
            bad.append(f"target_profile.{k}: requested {want.get(k, '<absent>')!r}, artifact built for {have.get(k, '<absent>')!r}")
    if applicability_now is None:
        bad.append("applicability conditions of the requester were not provided (every condition is unverifiable)")
    else:
        for k in sorted(artifact.applicability):
            if k not in applicability_now:
                bad.append(f"applicability.{k}: artifact requires {artifact.applicability[k]!r}, requester did not report it")
            elif applicability_now[k] != artifact.applicability[k]:
                bad.append(f"applicability.{k}: artifact requires {artifact.applicability[k]!r}, requester has {applicability_now[k]!r}")
    if construction_version_now is not None and canonical_bytes(construction_version_now) != canonical_bytes(artifact.construction_version):
        bad.append("construction_version differs from the artifact's")
    return bad


def check_reuse(artifact: ArtifactIdentity, profile: TargetProfile, applicability_now: dict | None, construction_version_now: dict | None = None) -> ArtifactIdentity:
    """Return `artifact` if it is applicable for the requester, else raise `ApplicabilityError` (fail-closed)."""
    bad = reuse_mismatches(artifact, profile, applicability_now, construction_version_now)
    if bad:
        raise ApplicabilityError(bad)
    return artifact


def load_artifact(path: str, profile: TargetProfile, applicability_now: dict | None, construction_version_now: dict | None = None) -> ArtifactIdentity:
    """Read `artifact.json` and admit it only under `profile` / `applicability_now` (`check_reuse`)."""
    with open(path) as f:
        art = ArtifactIdentity.from_json(json.load(f))
    return check_reuse(art, profile, applicability_now, construction_version_now)


__all__ = ["ArtifactIdentity", "ApplicabilityError", "reuse_mismatches", "check_reuse", "load_artifact", "SCHEMA"]
