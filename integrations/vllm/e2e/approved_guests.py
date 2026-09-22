"""Approved guest executables: the verifier's own list of (ELF, vk) pairs it accepts, and the build pin.

GUIDANCE §57.10 (S12, user 11:54 PT): *a certificate saying ``guest_features = safe`` is not evidence that the
executed guest was safe*.  What the verifier needs is (i) a pinned source, toolchain, dependencies and resolved
feature configuration, (ii) a reproducible build or another trusted build-verification procedure, (iii) the
resulting executable and verification key matched against what the verifier itself approves, and (iv) refusal
of unapproved executables regardless of their claimed feature metadata.

This module is (iii) and (iv).  ``approved_guests.json`` beside it is the VERIFIER-SIDE policy: a list of
``{elf_sha256, vk_hash, guest_build, approval, approved_by, date}``.  ``approval`` is

* ``"reproduced"`` -- someone rebuilt the ELF from ``guest_build.source_commit`` with the pinned toolchain and
  got the same ``elf_sha256`` and ``vk_hash`` (the procedure: ``e2e/guest_build.py``; GUEST report §2);
* ``"provisional-legacy"`` -- an ELF pinned by an existing record, listed so those records stay certifiable,
  with whatever build metadata the record carries and an explicit ``missing`` list of what it does NOT carry.
  A provisional entry is NOT build-verified; the certificate says so.

The STRICT paths (``verify_composed`` without ``--historical-v2``; ``certificate.py`` without
``--historical-v2``) refuse -- NOT BOUND, reason ``GUEST_NOT_APPROVED`` -- any record whose ELF is not listed
or whose vk is not the listed ELF's vk, whatever ``guest_features`` / ``guest_build`` the record claims.  The
legacy ``--historical-v2`` path warns loudly instead (its certificate is already labelled uncertified).

Nothing here touches the sampler, the partition, the obligation set or the checker's decision: the policy
decides which PROVER PROGRAM the verifier is willing to believe a proof of, not what that program checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POLICY_FILE = HERE / "approved_guests.json"
SCHEMA = "veritor/approved-guests/v1"
APPROVALS = ("reproduced", "provisional-legacy")
GUEST_NOT_APPROVED = "GUEST_NOT_APPROVED"

#: the keys a complete build pin carries (``guest_build`` in the certificate / manifest / policy)
GUEST_BUILD_KEYS = ("source_commit", "source_tree_sha", "toolchain", "features_resolved", "elf_sha256", "vk_hash", "built_by", "built_at")
TOOLCHAIN_KEYS = ("cargo_prove", "rustc", "sp1_sdk")


@dataclass(frozen=True)
class Approval:
    """The verifier's answer for one (ELF, vk): ``ok`` iff the pair is listed.  ``reason`` is
    ``GUEST_NOT_APPROVED`` when it is not (``detail`` says which of the two failed); ``entry`` is the
    policy entry when it is; ``approval`` is that entry's level (``reproduced`` / ``provisional-legacy``)."""

    ok: bool
    reason: str | None = None
    detail: str = ""
    approval: str | None = None
    entry: dict[str, Any] | None = None
    missing: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "detail": self.detail, "approval": self.approval,
                "approved_by": (self.entry or {}).get("approved_by"), "date": (self.entry or {}).get("date"),
                "build_metadata_missing": list(self.missing)}


def load_policy(path: Path | None = None) -> dict[str, Any]:
    """The policy document, validated: schema, distinct ELFs, approval levels, 64-hex ELF shas."""

    p = Path(path) if path else POLICY_FILE
    doc = json.loads(p.read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    seen: set[str] = set()
    for g in doc.get("guests") or []:
        elf = g.get("elf_sha256")
        if not isinstance(elf, str) or len(elf) != 64 or any(c not in "0123456789abcdef" for c in elf):
            raise ValueError(f"{p}: entry elf_sha256 {elf!r} is not 64 lowercase hex")
        if elf in seen:
            raise ValueError(f"{p}: ELF {elf[:12]} listed twice (one vk per ELF)")
        seen.add(elf)
        if not isinstance(g.get("vk_hash"), str) or not g["vk_hash"].startswith("0x"):
            raise ValueError(f"{p}: ELF {elf[:12]} has no 0x-prefixed vk_hash")
        if g.get("approval") not in APPROVALS:
            raise ValueError(f"{p}: ELF {elf[:12]} approval {g.get('approval')!r} not in {APPROVALS}")
    return doc


def approval_of(elf_sha256: str | None, vk_hash: str | None, *, policy: dict[str, Any] | None = None,
                policy_path: Path | None = None) -> Approval:
    """Is (``elf_sha256``, ``vk_hash``) an approved guest?  Fail-closed: a missing ELF sha or vk is not approved;
    an ELF that is listed with a DIFFERENT vk is not approved (the vk is the proof system's name for the ELF, and
    the two disagreeing means one of them is not what the record says)."""

    doc = policy if policy is not None else load_policy(policy_path)
    if not elf_sha256 or not vk_hash:
        return Approval(False, GUEST_NOT_APPROVED, "the record pins no guest ELF sha / vk (nothing to approve)")
    for g in doc.get("guests") or []:
        if g["elf_sha256"] == elf_sha256:
            if g["vk_hash"].lower() != str(vk_hash).lower():
                return Approval(False, GUEST_NOT_APPROVED,
                                f"ELF {elf_sha256[:12]} is approved with vk {g['vk_hash'][:14]}..., the record says vk {str(vk_hash)[:14]}...")
            return Approval(True, None, f"ELF {elf_sha256[:12]} / vk {g['vk_hash'][:14]}... approved ({g['approval']}) by {g.get('approved_by')} on {g.get('date')}",
                            approval=g["approval"], entry=g, missing=list(g.get("missing") or []))
    return Approval(False, GUEST_NOT_APPROVED, f"ELF {elf_sha256[:12]} (vk {str(vk_hash)[:14]}...) is not in {POLICY_FILE.name}")


def statement_family_approval(family: str, elf_sha256: str | None, vk_hash: str | None, *, policy: dict[str, Any] | None = None,
                              policy_path: Path | None = None) -> Approval:
    """[E7 2026-09-11] :func:`approval_of` for a STATEMENT-BOUND family (a dedicated guest whose public values are
    ``sha256(statement) || verdict``, e.g. E5's ``veritor.select-argmax@1``): the (ELF, vk) must be approved AND the entry must list
    ``family`` under ``statement_families`` -- an approved guest of another family (the protocol guest) approves no statement
    family, and a statement guest lists no gate-set ``families`` (``veritor.interface.families.check_family_approved`` stays
    fail-closed for typed obligations under it).  Fail-closed on every miss."""

    a = approval_of(elf_sha256, vk_hash, policy=policy, policy_path=policy_path)
    if not a.ok:
        return a
    sf = (a.entry or {}).get("statement_families") or {}
    if str(family) not in sf:
        return Approval(False, GUEST_NOT_APPROVED,
                        f"ELF {str(elf_sha256)[:12]} is approved but lists no statement family {family!r} (statement_families: {sorted(sf)})")
    return a


def guest_build_missing(pin: dict[str, Any] | None) -> list[str]:
    """Which of the build-pin keys a ``guest_build`` document lacks (``None`` for the value counts as lacking)."""

    if not pin:
        return list(GUEST_BUILD_KEYS)
    out = [k for k in GUEST_BUILD_KEYS if pin.get(k) in (None, "", [], {})]
    tc = pin.get("toolchain") or {}
    out += [f"toolchain.{k}" for k in TOOLCHAIN_KEYS if tc.get(k) in (None, "")]
    return out


def check_guest(elf_sha256: str | None, vk_hash: str | None, recorded_build: dict[str, Any] | None, *, strict: bool,
                policy: dict[str, Any] | None = None, policy_path: Path | None = None, what: str = "guest") -> tuple[dict[str, Any], list[str], list[str]]:
    """The verifier's guest check for one record: ``(summary, problems, warnings)``.

    * approval by the policy (``approval_of``); on the STRICT path a refusal is a problem
      (``GUEST_NOT_APPROVED``), on the legacy path a loud warning;
    * the record's own ``guest_build`` (informative -- the prover's claim) must name the same ELF / vk when
      present; the policy entry's ``guest_build`` is what the certificate reports as authoritative.
    """

    ap = approval_of(elf_sha256, vk_hash, policy=policy, policy_path=policy_path)
    problems: list[str] = []
    warnings: list[str] = []
    if not ap.ok:
        msg = f"{what}: {GUEST_NOT_APPROVED}: {ap.detail}"
        (problems if strict else warnings).append(msg if strict else msg + " -- accepted ONLY because --historical-v2 certifies nothing about the guest")
    if recorded_build:
        if recorded_build.get("elf_sha256") not in (None, elf_sha256) or (recorded_build.get("vk_hash") and str(recorded_build.get("vk_hash")).lower() != str(vk_hash).lower()):
            problems.append(f"{what}: the record's guest_build names ELF {str(recorded_build.get('elf_sha256'))[:12]} / vk {str(recorded_build.get('vk_hash'))[:14]}, "
                            f"the proofs were checked against ELF {str(elf_sha256)[:12]} / vk {str(vk_hash)[:14]}")
    if ap.ok and ap.approval == "provisional-legacy":
        warnings.append(f"{what}: ELF {str(elf_sha256)[:12]} is approved PROVISIONALLY (legacy record pin, build not reproduced); missing build metadata: {ap.missing}")
    summary = {"elf_sha256": elf_sha256, "vk_hash": vk_hash, "approval": ap.to_json(),
               "guest_build": (ap.entry or {}).get("guest_build") if ap.ok else None,
               "guest_build_source": ("approved_guests.json" if ap.ok else None),
               "guest_build_recorded": recorded_build,
               "guest_build_recorded_missing": guest_build_missing(recorded_build) if recorded_build else list(GUEST_BUILD_KEYS),
               "policy": {"file": str((Path(policy_path) if policy_path else POLICY_FILE).name), "schema": SCHEMA}}
    return summary, problems, warnings


__all__ = ["APPROVALS", "GUEST_BUILD_KEYS", "GUEST_NOT_APPROVED", "POLICY_FILE", "SCHEMA", "Approval", "approval_of", "check_guest",
           "guest_build_missing", "load_policy", "statement_family_approval"]
