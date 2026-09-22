"""The campaign's result contract: what a campaign number must say about itself, on top of ``research``'s ``result.json``.

``research`` records and ships an opaque workload; the workload owns the meaning of ``$RESEARCH_RUN_DIR/result.json``
(``tools/research/README.md``).  This module is that meaning for the proof campaign, shared by every backend:

* the ``workload_fingerprint`` fields a campaign number must carry (:data:`REQUIRED_FIELDS`), with the
  :data:`IDENTITY_FIELDS` two results must agree on before any of their numbers may be compared
  (``backends/AGENTS.md``: never compare across ``proof_class``; the same batch size, hardware and soundness target);
* the measurement names: the six disjoint timing buckets ``t.<bucket>`` (seconds), ``t.total``, joint stages
  ``t.joint.<a>+<b>`` for a kernel that fuses buckets it cannot split (never double-counted against a standalone
  bucket or another joint), and the rest of the decision's columns (VU throughput, peak memory, bytes moved,
  transcript bytes, rounds, verifier work, fixed setup);
* :func:`validate`, the list of everything a result got wrong (empty means conforming), and the one-call builders
  :func:`fingerprint` and :func:`measurement` so a backend emits conforming pieces without re-reading this file.

Envelope (``research/result/v0.1``, ``tools/research/src/research/result.py``)::

    {"schema", "run_id", "workload_fingerprint": {...}, "validation": {"status", "checker", "detail", "evidence"},
     "measurements": [{"name", "value", "unit"}, ...], "measurement_files": [...], "artifacts": [...]}

Fingerprint fields (all required; a sub-field whose value is not known is ``null``, never a guess)::

    profile         the name of the target actually proved (one of :data:`PROFILES`, ``verity.verification.target.TARGETS``);
                    the campaign target is FIRST.name (:data:`PROFILE`); results of different profiles are never compared
    domain          the relation's domain, "finite" for FIRST (a total-semantics backend is another domain)
    K               GemmCoordinate<K>: a positive multiple of 16
    B               batch size: VU instances established by the one proof; B_real (optional) = real instances when padded
    N_subbatches    (optional, default 1) independently committed sub-batches behind the reported numbers; the soundness
                    error is the union bound over them (security accounting), and the cost includes every one
    proof_class     ARITHMETIC_DIAGNOSTIC | NON_ZK_PROOF_DIAGNOSTIC | NON_ZK_PROOF | COMPLETE_HVZK_BACKEND
                    | COMPLETE_ZK_BACKEND | NO_PROOF   (HVZK = honest-verifier privacy only: a milestone, not the target)
    authentication  included | excluded          (were the boundary words authenticated against the object's roots)
    zk_mode         "none" unless proof_class is a COMPLETE_*ZK_BACKEND, whose mode names how privacy is obtained
    security        {"target": log2 of the target soundness error (-100 means 2^-100) or null, ...backend parameters}
    hardware        {"gpu": {"name", "driver", "cuda"} | null, "cpu": {"model", ...} | null}   (probe_hardware())
    software        {"backend": {"name", "version" and/or "commit", ...}, "toolchain": {...versions}}
    preprocessing   what was amortised offline and is therefore not in the buckets (a string or an object)
    retention       retained | streamed | recomputed   (how the witness was held during proving)
    instances       which frozen instances were proved: {"dataset": "bench-instances/v1", "tier": "vu-k1536",
                    "range": [lo, hi), "manifest_sha256"} (``fixtures/bench-instances/v1``; hi - lo = B_real if present
                    else B; sub-batches are contiguous sub-ranges of it) or null = not measured on the shared set.  Two
                    results whose non-null ``instances`` disagree are never compared (:func:`compare_instances`); the
                    judge warns when either is null or absent.  Every Candidate A/B number at K=1536 carries it.

Timing (seconds; the buckets are disjoint and, with the joints, sum to at most ``t.total``, tolerance 1%)::

    t.witness  t.encoding_commitment  t.arithmetic  t.lookup  t.zk_additional  t.serialization    t.total
    t.joint.<a>+<b>[+<c>...]   a fused stage covering those buckets, in canonical bucket order (see joint())

The headline is throughput, ``vu.per_second``, and better its inverse relative to the machine being proved::

    rate.proved_flop_per_second       R_proved = 2 * K * B / t.total   (B = vu.count; the prover's own throughput, no reference)
    rate.native_peak_flop_per_second  R_native(target) = native_reference(profile).flop_per_second
    overhead.vs_native_peak           R_native(target) / R_proved      (ratio)

``R_proved`` is intrinsic to the prover (one VU of GemmCoordinate<K> is K MACs = 2K FLOP).  ``R_native`` depends on
the target named by the fingerprint's ``profile``: the datasheet dense tensor-core peak of the target's conformance
anchor for its operand dtype at FP32 accumulate (``Target.native_peak``, ``verity.verification.target``; A100 BF16
312 TFLOPS for FIRST, RTX 4090 FP8 330.3 TFLOPS for FP8_ADA_MMA, H100 FP8 1978.9 TFLOPS for FP8_HOPPER_WGMMA).  The
identity ``overhead.vs_native_peak == rate.native_peak_flop_per_second / rate.proved_flop_per_second`` holds by
construction (:func:`rate_measurements`); :func:`validate` recomputes each of the three from ``t.total``,
``vu.count``, ``K`` and ``profile`` and rejects a mismatch, so an overhead divided by another target's peak fails.

Other measurements: ``vu.count`` (count), ``vu.seconds_per_vu`` (s), ``vu.per_second`` (1/s), ``mem.peak_host_bytes``,
``mem.peak_device_bytes``, ``bytes.moved``, ``transcript.bytes`` (B; the total communication), ``verifier.seconds``,
``fixed.setup_seconds`` (s).  Latency (a first-class axis): ``rounds.protocol`` (count of protocol rounds) and
``rounds.sequential_depth`` (network round trips on the critical path after all parallelism/batching -- what the
network floor depth x RTT is made of), and the end-to-end wall time at the RTT sweep ``t.e2e.rtt_0ms``,
``t.e2e.rtt_0.1ms``, ``t.e2e.rtt_1ms``, ``t.e2e.rtt_10ms``, ``t.e2e.rtt_50ms`` (s; predicted or measured, say which in
the fingerprint).  Census (same columns for every candidate): ``census.witness_elements``,
``census.committed_elements`` (count), ``census.bytes_encoded`` (B), ``census.hash_bytes`` (B hashed),
``census.field_ops`` (count), ``census.memory_passes`` (count), ``census.bytes_moved`` (B).  Names under the contract's prefixes that this file does not define are rejected; a
workload's own names (``prove_seconds``, ``cycles.parse``) pass through untouched so earlier runs stay comparable.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verity.verification.target import FIRST, TARGETS, NativePeak

__all__ = [
    "AUTHENTICATION",
    "BUCKETS",
    "BUCKET_NAMES",
    "DOMAIN",
    "FP8_ADA_INSTANCES_DATASET",
    "FP8_ADA_INSTANCES_DIGEST",
    "FP8_ADA_INSTANCES_SEED",
    "FP8_ADA_INSTANCES_TIER",
    "IDENTITY_FIELDS",
    "INSTANCES_DATASET",
    "INSTANCES_FIELD",
    "INSTANCES_TIERS",
    "MISSING",
    "OVERHEAD",
    "PROFILE",
    "PROFILES",
    "PROOF_CLASSES",
    "RATE_NATIVE",
    "RATE_PROVED",
    "RATE_TOLERANCE",
    "REQUIRED_FIELDS",
    "RESULT_SCHEMA",
    "RETENTION",
    "TOLERANCE",
    "T_TOTAL",
    "UNITS",
    "ZK_CLASSES",
    "ZK_NONE",
    "compare_instances",
    "fingerprint",
    "identity",
    "incomparable",
    "instances_ref",
    "joint",
    "measurement",
    "NATIVE_REFERENCE",
    "native_reference",
    "native_seconds_per_vu",
    "overhead",
    "parse_joint",
    "probe_hardware",
    "proved_flop_per_second",
    "rate_measurements",
    "unit_of",
    "validate",
    "validate_fingerprint",
    "validate_measurements",
]

RESULT_SCHEMA = "research/result/v0.1"

#: The campaign target's profile name and domain label (``backends/AGENTS.md``: the target is fixed).
PROFILE = FIRST.name
DOMAIN = "finite"
#: Every profile a result may name: the targets of ``verity.verification.target.TARGETS`` (FIRST and the FP8 drafts).
#: ``profile`` is an identity field, so a result of one profile is never compared with a result of another.
PROFILES = tuple(TARGETS)

PROOF_CLASSES = (
    "ARITHMETIC_DIAGNOSTIC", "NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF", "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND", "NO_PROOF",
)
#: The classes that establish privacy (honest-verifier, or against a malicious verifier); only these carry a zk_mode.
ZK_CLASSES = ("COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND")
AUTHENTICATION = ("included", "excluded")
RETENTION = ("retained", "streamed", "recomputed")
ZK_NONE = "none"

#: Two results whose values differ on any of these are never compared; the judge refuses (``benchmarks/judge.py``).
#: Two results must agree on every one of these before any of their numbers may be compared: same proof class and
#: authentication, same target and domain, same K **and batch size**, the **same hardware** (the whole probed object:
#: a 4090 run is never compared with a CPU run, nor with another pod's driver), and the same soundness target
#: (``security.target``, a dotted path into the fingerprint).  The decision numbers are B=4096 K=1536 VUs of one
#: instance set on one machine at 2^-128; ``--allow`` names the one field an experiment deliberately varies.
IDENTITY_FIELDS = ("proof_class", "authentication", "profile", "domain", "K", "B", "hardware", "security.target")
REQUIRED_FIELDS = (
    "profile", "domain", "K", "B", "proof_class", "authentication", "zk_mode", "security", "hardware", "software",
    "preprocessing", "retention", "instances",
)
#: The frozen instance set every K=1536 number is measured on (``fixtures/bench-instances/v1``, ``verity_numerical.bench.instances``).
INSTANCES_DATASET = "bench-instances/v1"
INSTANCES_FIELD = "instances"
INSTANCES_TIERS = ("vu-k1536", "tu-k16", "vu-k1536-neg")
#: The frozen FP8 Ada instance set (``fp8-ada-mma-draft/2026-09-22``): the synthetic VUs ``backends/direct/ligero/fp8/chain.py``
#: draws (``instances_fp8``: uniform finite E4M3 bytes from seed :data:`FP8_ADA_INSTANCES_SEED`, model-recorded accumulators).
#: :data:`FP8_ADA_INSTANCES_DIGEST` is ``chain.instances_digest(4096)`` = sha256 of
#: ``"fp8-ada synthetic|seed=20260922|n=4096|K=1536"`` -- pinned here (frozen 2026-09-22) so the canonical tables
#: (``verity_numerical.bench.tables``) recognise the set without importing the prover; a change to the recipe is a new tier.
FP8_ADA_INSTANCES_DATASET = "bench-instances-fp8-ada/v1"
FP8_ADA_INSTANCES_TIER = "vu-k1536-fp8-ada"
FP8_ADA_INSTANCES_SEED = 20260922
FP8_ADA_INSTANCES_DIGEST = "e66ff0f21c8e67d13b4b2d4cee2afa35b8345656b314da80e6b342ba051f078d"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

BUCKETS = ("witness", "encoding_commitment", "arithmetic", "lookup", "zk_additional", "serialization")
BUCKET_NAMES = tuple(f"t.{b}" for b in BUCKETS)
T_TOTAL = "t.total"
_JOINT_PREFIX = "t.joint."
_JOINT_SEP = "+"
#: Buckets and joints may exceed ``t.total`` by this fraction before it counts as double-counting.
TOLERANCE = 0.01

UNITS: dict[str, str] = {
    **{name: "s" for name in BUCKET_NAMES},
    T_TOTAL: "s",
    "vu.count": "count",
    "vu.seconds_per_vu": "s",
    "vu.per_second": "1/s",
    "mem.peak_host_bytes": "B",
    "mem.peak_device_bytes": "B",
    "bytes.moved": "B",
    "transcript.bytes": "B",
    "rounds.protocol": "count",
    "rounds.sequential_depth": "count",
    **{f"t.e2e.rtt_{rtt}ms": "s" for rtt in ("0", "0.1", "1", "10", "50")},
    "census.witness_elements": "count",
    "census.committed_elements": "count",
    "census.bytes_encoded": "B",
    "census.hash_bytes": "B",
    "census.field_ops": "count",
    "census.memory_passes": "count",
    "census.bytes_moved": "B",
    "verifier.seconds": "s",
    "fixed.setup_seconds": "s",
    "rate.proved_flop_per_second": "FLOP/s",
    "rate.native_peak_flop_per_second": "FLOP/s",
    "overhead.vs_native_peak": "ratio",
}
_RESERVED_PREFIXES = ("t.", "vu.", "mem.", "fixed.", "bytes.", "transcript.", "verifier.", "rounds.", "census.", "rate.", "overhead.")
RATE_PROVED = "rate.proved_flop_per_second"
RATE_NATIVE = "rate.native_peak_flop_per_second"
OVERHEAD = "overhead.vs_native_peak"
#: A reported rate must agree with its formula to this relative tolerance (they are computed, not measured).
RATE_TOLERANCE = 1e-6


def native_reference(profile: str = PROFILE) -> NativePeak:
    """The native peak the headline overhead of a ``profile`` divides by: that target's ``Target.native_peak`` -- the
    datasheet dense tensor-core peak of the instruction's conformance anchor for the target's operand dtype at FP32
    accumulate, no sparsity (``verity.verification.target``, "Native peak").  A GemmCoordinate<K> is K MACs = 2K FLOP,
    so the native time of one VU at peak is 2K / rate: 9.85 ps at K=1536 for FIRST (A100 BF16, ~1.0e11 VU/s).  Peak is
    the conservative choice (a real decode step runs below peak, so the true overhead is *lower* than this number); it
    is also the only reference every result can share without a runtime capture.  The prover's own hardware is in the
    fingerprint, not in the ratio: the question is "how many native seconds does one proved second buy".  Raises
    ``ValueError`` for a profile that is not a target."""

    target = TARGETS.get(profile) if isinstance(profile, str) else None
    if target is None:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    return target.native_peak


#: The campaign target's native reference as a plain (JSON-serialisable) mapping, for the callers that predate
#: :func:`native_reference`: ``NATIVE_REFERENCE["flop_per_second"]`` is ``native_reference(PROFILE).flop_per_second``.
NATIVE_REFERENCE: dict[str, Any] = {**asdict(native_reference(PROFILE)), "profile": PROFILE, "flop_per_vu": "2 * K"}


def native_seconds_per_vu(K: int, profile: str = PROFILE) -> float:
    """Native wall seconds of one GemmCoordinate<K> at :func:`native_reference` of ``profile`` (9.846e-12 at K=1536 for FIRST)."""

    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    return native_reference(profile).seconds_per_vu(K)


def overhead(seconds_per_vu: float, K: int, profile: str = PROFILE) -> float:
    """``overhead.vs_native_peak``: prover seconds per VU over the native seconds per VU at the peak of ``profile``'s
    target -- the same number as ``rate.native_peak_flop_per_second / rate.proved_flop_per_second``."""

    if seconds_per_vu <= 0:
        raise ValueError(f"seconds_per_vu must be positive, got {seconds_per_vu!r}")
    return seconds_per_vu / native_seconds_per_vu(K, profile)


def proved_flop_per_second(t_total: float, vu_count: int, K: int) -> float:
    """``rate.proved_flop_per_second``: the prover's intrinsic throughput ``2 * K * vu_count / t_total`` -- the FLOP of the
    VUs it established per prover wall second, independent of any reference machine."""

    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    if not _is_number(vu_count) or vu_count <= 0:
        raise ValueError(f"vu_count must be positive, got {vu_count!r}")
    if not _is_number(t_total) or t_total <= 0:
        raise ValueError(f"t_total must be positive, got {t_total!r}")
    return 2 * K * vu_count / t_total


def rate_measurements(t_total: float, vu_count: int, K: int, profile: str = PROFILE) -> list[dict[str, Any]]:
    """The three rate measurements of a VU-scope result, consistent by construction: ``rate.proved_flop_per_second``
    (:func:`proved_flop_per_second`), ``rate.native_peak_flop_per_second`` (:func:`native_reference` of ``profile``)
    and ``overhead.vs_native_peak`` = native / proved.  ``profile`` must be the fingerprint's."""

    proved = proved_flop_per_second(t_total, vu_count, K)
    native = native_reference(profile).flop_per_second
    return [measurement(RATE_PROVED, proved), measurement(RATE_NATIVE, native), measurement(OVERHEAD, native / proved)]

#: Sentinel for an identity field a fingerprint does not carry (distinct from a recorded ``null``).
MISSING = object()


# --- builders --------------------------------------------------------------------------------------------------------


def joint(*buckets: str) -> str:
    """The measurement name of a stage that fuses ``buckets`` (at least two, distinct): ``t.joint.<a>+<b>`` in canonical
    bucket order, so the same fusion has the same name in every backend."""

    return _JOINT_PREFIX + _JOINT_SEP.join(_joint_parts(buckets))


def parse_joint(name: str) -> tuple[str, ...] | None:
    """The buckets a joint name covers, or ``None`` when ``name`` is not a joint.  A malformed joint raises."""

    if not name.startswith(_JOINT_PREFIX):
        return None
    return _joint_parts(name[len(_JOINT_PREFIX):].split(_JOINT_SEP))


def _joint_parts(parts: Iterable[str]) -> tuple[str, ...]:
    parts = tuple(parts)
    unknown = [p for p in parts if p not in BUCKETS]
    if unknown:
        raise ValueError(f"joint stage names buckets from {BUCKETS}, not {unknown}")
    if len(parts) < 2:
        raise ValueError("a joint stage fuses at least two buckets; a single bucket is t.<bucket>")
    if len(set(parts)) != len(parts):
        raise ValueError(f"a joint stage lists each bucket once, got {parts}")
    return tuple(b for b in BUCKETS if b in parts)


def unit_of(name: str) -> str | None:
    """The contract's unit for ``name``; ``None`` for a name the contract does not define."""

    if name in UNITS:
        return UNITS[name]
    try:
        return "s" if parse_joint(name) else None
    except ValueError:
        return None


def measurement(name: str, value: float | int, unit: str | None = None) -> dict[str, Any]:
    """One ``{name, value, unit}`` entry.  Contract names get their unit supplied and checked; a workload's own name
    must state its unit."""

    if not isinstance(name, str) or not name:
        raise ValueError(f"measurement name must be a non-empty string, got {name!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"measurement {name!r}: value must be a number, got {value!r}")
    if name.startswith(_JOINT_PREFIX):
        parse_joint(name)  # a malformed joint raises with the reason
    expected = unit_of(name)
    if unit is None:
        if expected is None:
            raise ValueError(f"{name!r} is not a contract measurement; give its unit explicitly")
        unit = expected
    elif expected is not None and unit != expected:
        raise ValueError(f"{name!r} is measured in {expected!r}, not {unit!r}")
    return {"name": name, "value": value, "unit": unit}


def instances_ref(tier: str, lo: int, hi: int, manifest_sha256: str, dataset: str = INSTANCES_DATASET) -> dict[str, Any]:
    """The ``instances`` entry for ids ``[lo, hi)`` of ``tier``: ``{"dataset", "tier", "range": [lo, hi], "manifest_sha256"}``.
    ``manifest_sha256`` is the sha256 of the dataset's ``manifest.json`` (``verity_numerical.bench.instances.manifest_sha256``)."""

    ref = {"dataset": dataset, "tier": tier, "range": [lo, hi], "manifest_sha256": manifest_sha256}
    problems = _validate_instances(ref, None, None)
    if problems:
        raise ValueError("instances does not conform: " + "; ".join(problems))
    return ref


def fingerprint(
    *,
    profile: str,
    domain: str,
    K: int,
    B: int,
    proof_class: str,
    authentication: str,
    zk_mode: str,
    security: Mapping[str, Any],
    hardware: Mapping[str, Any],
    software: Mapping[str, Any],
    preprocessing: str | Mapping[str, Any],
    retention: str,
    instances: Mapping[str, Any] | None = None,
    B_real: int | None = None,
    N_subbatches: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    """A conforming ``workload_fingerprint``: the required fields, then the workload's own ``extra`` entries.  Raises
    ``ValueError`` listing every problem rather than emitting a bad one.  ``N_subbatches`` is always written, so a
    batch chopped into several proofs is visible beside ``B``; so is ``instances`` (``null`` = not measured on the
    shared instance set -- see :func:`instances_ref`)."""

    fp: dict[str, Any] = {
        "profile": profile, "domain": domain, "K": K, "B": B, "N_subbatches": N_subbatches, "proof_class": proof_class,
        "authentication": authentication,
        "zk_mode": zk_mode, "security": dict(security), "hardware": dict(hardware), "software": dict(software),
        "preprocessing": preprocessing if isinstance(preprocessing, str) else dict(preprocessing), "retention": retention,
        "instances": None if instances is None else dict(instances),
    }
    if B_real is not None:
        fp["B_real"] = B_real
    fp.update(extra)
    problems = validate_fingerprint(fp)
    if problems:
        raise ValueError("fingerprint does not conform: " + "; ".join(problems))
    return fp


# --- validation -------------------------------------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_fingerprint(fp: Any) -> list[str]:
    """Every way ``fp`` fails to be a campaign fingerprint; ``[]`` when it conforms."""

    if not isinstance(fp, Mapping):
        return ["workload_fingerprint must be an object"]
    p: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in fp:
            p.append(f"workload_fingerprint.{field} missing")
    present = {f: fp[f] for f in REQUIRED_FIELDS if f in fp}

    if "profile" in present and present["profile"] not in PROFILES:
        p.append(f"profile must be a target of verity.verification.target, one of {PROFILES} (the campaign target is {PROFILE!r}), got {present['profile']!r}")
    if "domain" in present and (not isinstance(present["domain"], str) or not present["domain"]):
        p.append(f"domain must be a non-empty string (FIRST: {DOMAIN!r}), got {present['domain']!r}")
    if "K" in present and not (_is_int(present["K"]) and present["K"] > 0 and present["K"] % 16 == 0):
        p.append(f"K must be a positive multiple of 16, got {present['K']!r}")
    if "B" in present and not (_is_int(present["B"]) and present["B"] > 0):
        p.append(f"B must be a positive int, got {present['B']!r}")
    if "B_real" in fp:
        br = fp["B_real"]
        if not (_is_int(br) and br > 0):
            p.append(f"B_real must be a positive int when present, got {br!r}")
        elif _is_int(present.get("B")) and br > present["B"]:
            p.append(f"B_real ({br}) exceeds B ({present['B']})")
    pc = present.get("proof_class")
    if "proof_class" in present and pc not in PROOF_CLASSES:
        p.append(f"proof_class must be one of {PROOF_CLASSES}, got {pc!r}")
    if "authentication" in present and present["authentication"] not in AUTHENTICATION:
        p.append(f"authentication must be one of {AUTHENTICATION}, got {present['authentication']!r}")
    if "zk_mode" in present:
        zk = present["zk_mode"]
        if not isinstance(zk, str) or not zk:
            p.append(f"zk_mode must be a non-empty string ({ZK_NONE!r} when there is no zero-knowledge), got {zk!r}")
        elif pc in ZK_CLASSES and zk == ZK_NONE:
            p.append(f"proof_class {pc} with zk_mode 'none' is a contradiction")
        elif pc in PROOF_CLASSES and pc not in ZK_CLASSES and zk != ZK_NONE:
            p.append(f"proof_class {pc} is not zero-knowledge; zk_mode must be {ZK_NONE!r}, got {zk!r}")
    if "N_subbatches" in fp:
        ns = fp["N_subbatches"]
        if not (_is_int(ns) and ns > 0):
            p.append(f"N_subbatches must be a positive int when present, got {ns!r}")
    if "security" in present:
        sec = present["security"]
        if not isinstance(sec, Mapping):
            p.append(f"security must be an object, got {sec!r}")
        elif "target" not in sec:
            p.append("security.target missing (log2 of the target soundness error, or null when not established)")
        elif sec["target"] is not None and not (_is_int(sec["target"]) and sec["target"] <= 0):
            p.append(f"security.target is log2 of the soundness error (e.g. -100 for 2^-100): a non-positive int or null, got {sec['target']!r}")
    if "hardware" in present:
        p += _validate_hardware(present["hardware"])
    if "software" in present:
        p += _validate_software(present["software"])
    if "preprocessing" in present:
        pre = present["preprocessing"]
        if not ((isinstance(pre, str) and pre) or isinstance(pre, Mapping)):
            p.append(f"preprocessing must be a non-empty string or an object saying what was amortised offline, got {pre!r}")
    if "retention" in present and present["retention"] not in RETENTION:
        p.append(f"retention must be one of {RETENTION}, got {present['retention']!r}")
    if "instances" in present and present["instances"] is not None:
        b = present.get("B") if _is_int(present.get("B")) else None
        br = fp.get("B_real") if _is_int(fp.get("B_real")) else None
        p += _validate_instances(present["instances"], b, br)
    return p


def _validate_instances(inst: Any, B: int | None, B_real: int | None) -> list[str]:
    if not isinstance(inst, Mapping):
        return [f"instances must be an object {{dataset, tier, range, manifest_sha256}} or null, got {inst!r}"]
    p = []
    for key in ("dataset", "tier", "range", "manifest_sha256"):
        if key not in inst:
            p.append(f"instances.{key} missing")
    if p:
        return p
    if not isinstance(inst["dataset"], str) or not inst["dataset"]:
        p.append(f"instances.dataset must be a non-empty string ({INSTANCES_DATASET!r}), got {inst['dataset']!r}")
    elif inst["dataset"] == INSTANCES_DATASET and inst["tier"] not in INSTANCES_TIERS:
        p.append(f"instances.tier must be one of {INSTANCES_TIERS} for {INSTANCES_DATASET}, got {inst['tier']!r}")
    if not isinstance(inst["tier"], str) or not inst["tier"]:
        p.append(f"instances.tier must be a non-empty string, got {inst['tier']!r}")
    rng = inst["range"]
    if not (isinstance(rng, Sequence) and not isinstance(rng, str) and len(rng) == 2 and all(_is_int(v) for v in rng) and 0 <= rng[0] < rng[1]):
        p.append(f"instances.range must be [lo, hi) with 0 <= lo < hi, got {rng!r}")
    else:
        n = rng[1] - rng[0]
        want = B_real if B_real is not None else B
        if want is not None and n != want:
            p.append(f"instances.range [{rng[0]}, {rng[1]}) holds {n} ids but {'B_real' if B_real is not None else 'B'} is {want}")
    if not isinstance(inst["manifest_sha256"], str) or not _SHA256_RE.match(inst["manifest_sha256"]):
        p.append(f"instances.manifest_sha256 must be a 64-hex-digit sha256 of the dataset's manifest.json, got {inst['manifest_sha256']!r}")
    return p


def _validate_hardware(hw: Any) -> list[str]:
    if not isinstance(hw, Mapping):
        return [f"hardware must be an object, got {hw!r}"]
    p = []
    for key, name_field in (("gpu", "name"), ("cpu", "model")):
        if key not in hw:
            p.append(f"hardware.{key} missing (an object, or null when absent)")
            continue
        dev = hw[key]
        if dev is None:
            continue
        if not isinstance(dev, Mapping):
            p.append(f"hardware.{key} must be an object or null, got {dev!r}")
        elif not isinstance(dev.get(name_field), str) or not dev.get(name_field):
            p.append(f"hardware.{key}.{name_field} must be a non-empty string, got {dev.get(name_field)!r}")
        elif key == "gpu":
            for sub in ("driver", "cuda"):
                if sub not in dev:
                    p.append(f"hardware.gpu.{sub} missing (a string, or null when unknown)")
    return p


def _validate_software(sw: Any) -> list[str]:
    if not isinstance(sw, Mapping):
        return [f"software must be an object, got {sw!r}"]
    p = []
    be = sw.get("backend")
    if not isinstance(be, Mapping):
        p.append("software.backend must be an object {name, version and/or commit}")
    else:
        if not isinstance(be.get("name"), str) or not be.get("name"):
            p.append("software.backend.name must be a non-empty string")
        if "version" not in be and "commit" not in be:
            p.append("software.backend needs version and/or commit (null when unknown)")
    if not isinstance(sw.get("toolchain"), Mapping):
        p.append("software.toolchain must be an object of toolchain versions")
    return p


def validate_measurements(measurements: Any, proof_class: str | None = None) -> list[str]:
    """Every way the measurement list breaks the contract: shape, units, reserved names, duplicate names, joint
    structure, buckets or joints double-counted, buckets exceeding ``t.total``, VU throughput inconsistent with it.
    With ``proof_class`` given, a proof also needs ``t.total`` and at least one bucket or joint."""

    if not isinstance(measurements, list):
        return ["measurements must be a list"]
    p: list[str] = []
    seen: dict[str, Any] = {}
    for i, m in enumerate(measurements):
        if not isinstance(m, Mapping) or not isinstance(m.get("name"), str) or not m.get("name") or "value" not in m or not isinstance(m.get("unit"), str):
            p.append(f"measurements[{i}] needs name (str), value, unit (str)")
            continue
        name, value, unit = m["name"], m["value"], m["unit"]
        if name in seen:
            p.append(f"measurement {name!r} appears more than once")
            continue
        seen[name] = value
        if name.startswith(_JOINT_PREFIX):
            try:
                parse_joint(name)
            except ValueError as e:
                p.append(f"measurement {name!r}: {e}")
                continue
        expected = unit_of(name)
        if expected is None:
            if name.startswith(_RESERVED_PREFIXES):
                p.append(f"measurement {name!r} uses a contract prefix but is not a contract name")
            continue
        if unit != expected:
            p.append(f"measurement {name!r} must be in {expected!r}, got {unit!r}")
        if not _is_number(value):
            p.append(f"measurement {name!r} must be a number, got {value!r}")
        elif value < 0:
            p.append(f"measurement {name!r} must be non-negative, got {value!r}")

    numeric = {n: v for n, v in seen.items() if _is_number(v) and unit_of(n) is not None}
    buckets = {n[len("t."):]: v for n, v in numeric.items() if n in BUCKET_NAMES}
    joints: dict[str, tuple[str, ...]] = {}
    for n in numeric:
        parts = parse_joint(n) if n.startswith(_JOINT_PREFIX) else None
        if parts:
            if n != joint(*parts):
                p.append(f"joint stage {n!r} must be written in canonical bucket order: {joint(*parts)!r}")
            joints[n] = parts
    covered: dict[str, str] = {b: f"t.{b}" for b in buckets}
    for n, parts in joints.items():
        for b in parts:
            if b in covered:
                p.append(f"bucket {b!r} is counted twice: in {covered[b]!r} and in {n!r}")
            else:
                covered[b] = n
    if T_TOTAL in numeric:
        total = numeric[T_TOTAL]
        summed = sum(buckets.values()) + sum(numeric[n] for n in joints)
        if summed > total * (1 + TOLERANCE):
            p.append(f"buckets and joints sum to {summed:.6g} s, more than t.total {total:.6g} s by over {TOLERANCE:.0%}: something is double-counted")
        if "vu.count" in numeric and numeric["vu.count"] > 0:
            per_vu = total / numeric["vu.count"]
            if "vu.seconds_per_vu" in numeric and abs(numeric["vu.seconds_per_vu"] - per_vu) > TOLERANCE * per_vu:
                p.append(f"vu.seconds_per_vu {numeric['vu.seconds_per_vu']:.6g} is not t.total / vu.count = {per_vu:.6g}")
            if "vu.per_second" in numeric and per_vu > 0 and abs(numeric["vu.per_second"] - 1 / per_vu) > TOLERANCE / per_vu:
                p.append(f"vu.per_second {numeric['vu.per_second']:.6g} is not vu.count / t.total = {1 / per_vu:.6g}")
    if proof_class in PROOF_CLASSES and proof_class != "NO_PROOF":
        if T_TOTAL not in numeric:
            p.append(f"a {proof_class} result must report t.total")
        if not buckets and not joints:
            p.append(f"a {proof_class} result must report at least one timing bucket or joint stage")
        if proof_class not in ZK_CLASSES and buckets.get("zk_additional", 0) > 0:
            p.append(f"t.zk_additional is {buckets['zk_additional']!r} but proof_class {proof_class} has no zero-knowledge work")
    return p


def validate(result: Any) -> list[str]:
    """Every way ``result`` (a parsed ``result.json``) fails the campaign contract; ``[]`` means it conforms.
    The envelope is checked as far as the fingerprint and measurements need; ``research inspect`` owns the rest."""

    if not isinstance(result, Mapping):
        return ["result must be an object"]
    p: list[str] = []
    if result.get("schema") != RESULT_SCHEMA:
        p.append(f"schema must be {RESULT_SCHEMA!r}, got {result.get('schema')!r}")
    if not isinstance(result.get("run_id"), str) or not result.get("run_id"):
        p.append("run_id must be a non-empty string")
    val = result.get("validation")
    if not isinstance(val, Mapping) or val.get("status") not in ("passed", "failed", "not_run"):
        p.append("validation must be an object with status passed | failed | not_run")
    for key in ("artifacts", "measurement_files"):
        if key in result and not isinstance(result[key], list):
            p.append(f"{key} must be a list of relative paths")
    fp = result.get("workload_fingerprint")
    p += validate_fingerprint(fp)
    pc = fp.get("proof_class") if isinstance(fp, Mapping) else None
    p += validate_measurements(result.get("measurements"), pc if pc in PROOF_CLASSES else None)
    p += _check_rates(fp, result.get("measurements"))
    return p


def _check_rates(fp: Any, measurements: Any) -> list[str]:
    """The rate measurements, when reported, must be what the contract computes from the fingerprint's ``profile`` and
    ``K`` and the result's ``t.total`` and ``vu.count``: ``rate.proved_flop_per_second`` = 2 K vu.count / t.total and
    ``rate.native_peak_flop_per_second`` = :func:`native_reference` (profile) to :data:`RATE_TOLERANCE`;
    ``overhead.vs_native_peak`` = their ratio, i.e. :func:`overhead` at the fingerprint's ``K`` **and profile**, to
    :data:`TOLERANCE`.  An overhead divided by another target's peak fails here.  A profile that is not a target is
    reported by :func:`validate_fingerprint`; the rates are not checked against it."""

    if not isinstance(fp, Mapping) or not isinstance(measurements, list):
        return []
    values = {m.get("name"): m.get("value") for m in measurements if isinstance(m, Mapping)}
    reported = {name: values[name] for name in (RATE_PROVED, RATE_NATIVE, OVERHEAD) if values.get(name) is not None}
    if not reported:
        return []
    profile = fp.get("profile")
    if profile not in PROFILES:
        return []
    p: list[str] = []
    native = native_reference(profile).flop_per_second
    got = reported.get(RATE_NATIVE)
    if got is not None and (not _is_number(got) or abs(got - native) > RATE_TOLERANCE * native):
        p.append(f"{RATE_NATIVE} {got!r} is not the native peak of profile {profile!r}, {native:.6g} FLOP/s")
    if RATE_PROVED not in reported and OVERHEAD not in reported:
        return p
    K, total, count = fp.get("K"), values.get(T_TOTAL), values.get("vu.count")
    if not (_is_int(K) and K > 0) or not _is_number(total) or not _is_number(count) or count <= 0 or total <= 0:
        names = " and ".join(n for n in (RATE_PROVED, OVERHEAD) if n in reported)
        return p + [f"{names} needs K in the fingerprint and positive t.total and vu.count to be checked against"]
    proved = proved_flop_per_second(total, count, K)
    got = reported.get(RATE_PROVED)
    if got is not None and (not _is_number(got) or abs(got - proved) > RATE_TOLERANCE * proved):
        p.append(f"{RATE_PROVED} {got!r} is not 2 * K * vu.count / t.total = {proved:.6g} FLOP/s")
    got = reported.get(OVERHEAD)
    want = overhead(total / count, K, profile)
    if got is not None and (not _is_number(got) or abs(got - want) > TOLERANCE * want):
        p.append(f"{OVERHEAD} {got!r} is not (t.total / vu.count) / native_seconds_per_vu(K, profile) = {want:.6g} "
                 f"at the native peak of profile {profile!r} ({native:.6g} FLOP/s)")
    return p


# --- comparability -----------------------------------------------------------------------------------------------------


def _lookup(fp: Mapping[str, Any], path: str) -> Any:
    """``fp`` at the dotted ``path`` (:data:`MISSING` where any step is absent or not an object)."""

    node: Any = fp
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return MISSING
        node = node[key]
    return node


def identity(fp: Mapping[str, Any] | None) -> dict[str, Any]:
    """The :data:`IDENTITY_FIELDS` of ``fp`` (:data:`MISSING` where it does not carry one)."""

    fp = fp or {}
    return {f: _lookup(fp, f) for f in IDENTITY_FIELDS}


def incomparable(baseline: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None, allow: Sequence[str] = ()) -> list[tuple[str, Any, Any]]:
    """The identity fields on which two fingerprints disagree -- or which either lacks -- other than those in ``allow``.
    Non-empty means the numbers must not be compared.  Each entry is ``(field, baseline_value, candidate_value)``."""

    bad = sorted(set(allow) - set(IDENTITY_FIELDS))
    if bad:
        raise ValueError(f"allow takes identity fields {IDENTITY_FIELDS}, not {bad}")
    a, b = identity(baseline), identity(candidate)
    return [(f, a[f], b[f]) for f in IDENTITY_FIELDS if f not in allow and (a[f] is MISSING or b[f] is MISSING or a[f] != b[f])]


def compare_instances(baseline: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None) -> tuple[tuple[Any, Any] | None, list[str]]:
    """``(conflict, warnings)`` for the ``instances`` of two fingerprints.  ``conflict`` is ``(baseline, candidate)`` when
    both carry a non-null ``instances`` and they disagree -- a different dataset, tier or manifest, or different id
    ranges for the same batch size (a different range with a different ``B`` is a scaling point, visible in the
    fingerprint deltas, not a conflict); the judge then refuses.  ``warnings`` name the side(s) that lack it (absent or
    null): those numbers were not measured on the shared instance set, so the comparison rests on the identity fields alone."""

    fb, fc = baseline or {}, candidate or {}
    a, b = fb.get(INSTANCES_FIELD), fc.get(INSTANCES_FIELD)
    warnings = []
    for who, fp, v in (("baseline", fb, a), ("candidate", fc, b)):
        if INSTANCES_FIELD not in fp:
            warnings.append(f"{who} does not record instances (pre-{INSTANCES_DATASET} result): not measured on the shared instance set")
        elif v is None:
            warnings.append(f"{who} instances is null: not measured on the shared instance set")
    if a is None or b is None or not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return None, warnings
    if any(a.get(k) != b.get(k) for k in ("dataset", "tier", "manifest_sha256")):
        return (dict(a), dict(b)), warnings
    if list(a.get("range") or []) != list(b.get("range") or []):
        size = lambda fp: fp.get("B_real") if fp.get("B_real") is not None else fp.get("B")  # noqa: E731
        if size(fb) == size(fc):
            return (dict(a), dict(b)), warnings
    return None, warnings


# --- hardware probe ----------------------------------------------------------------------------------------------------


def probe_hardware(timeout: float = 10.0) -> dict[str, Any]:
    """``{"gpu": {...} | None, "cpu": {...} | None}`` from ``nvidia-smi`` and ``/proc/cpuinfo`` (or ``sysctl`` on
    macOS) when present.  Whatever cannot be observed is ``None``; nothing is guessed."""

    gpu = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout)
        rows = [r for r in q.stdout.splitlines() if r.strip()] if q.returncode == 0 else []
        if rows:
            name, driver, mem = (s.strip() for s in (rows[0].split(",") + ["", ""])[:3])
            cuda = None
            head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=timeout)
            m = re.search(r"CUDA Version:\s*([0-9][0-9.]*)", head.stdout) if head.returncode == 0 else None
            if m:
                cuda = m.group(1)
            gpu = {"name": name, "driver": driver or None, "cuda": cuda, "count": len(rows),
                   "memory_total_bytes": int(mem) << 20 if mem.isdigit() else None}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        gpu = None
    cpu = None
    try:
        text = Path("/proc/cpuinfo").read_text()
        models = re.findall(r"^model name\s*:\s*(.+)$", text, re.M)
        if models:
            cpu = {"model": models[0].strip(), "count": len(models)}
    except OSError:
        try:
            q = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.logicalcpu"], capture_output=True, text=True, timeout=timeout)
            lines = q.stdout.splitlines() if q.returncode == 0 else []
            if lines and lines[0].strip():
                cpu = {"model": lines[0].strip(), "count": int(lines[1]) if len(lines) > 1 and lines[1].strip().isdigit() else None}
        except (OSError, subprocess.TimeoutExpired):
            cpu = None
    return {"gpu": gpu, "cpu": cpu}
