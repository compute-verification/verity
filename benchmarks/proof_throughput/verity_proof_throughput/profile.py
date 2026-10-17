"""The Lean verifier's profile: one session of ``flock-verify verify`` under ``perf record`` (``verify.sh PROFILE=1``), its
samples attributed two ways.

* **Phase**: the innermost frame of a sample that names a protocol step (the Lean declarations of ``Flock.Ligerito``,
  ``Flock.Piop``, ``Flock.Opening``, ``Flock.Statement``, ...; Lean's C keeps each declaration's path in its symbol, and a
  specialization's symbol names the declaration it was made at).
* **Kernel**: the innermost frame that is a primitive the phases share: SHA-512 and SHA-256 compression, GF(2^128)
  arithmetic, JSON, the runtime's allocation and reference counting, byte arrays, memcpy.

A sample neither names falls in ``other``. The setup and session seconds are the verifier's own (its stderr's ``inputs``,
``setup`` and per-session lines); CPU seconds over wall seconds say how many threads it kept busy.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PHASES: tuple[tuple[str, str], ...] = (
    ("merkle_paths", r"Flock_merkleCheck|Flock_MerkleScheme|Flock_climb|Flock_opens|Flock_sibsOf|Flock_saltCount"),
    ("ligerito", r"Flock_ligerito|Flock_queryLevel|Flock_Poly_|Flock_sIter|Flock_svTable|Flock_whats|Flock_eqTableK|"
                 r"Flock_nextMsg|Flock_nextOod|Flock_oodIntro|Flock_claimPow|Flock_cff|Flock_positions|Flock_lift\b|Flock_nonceFrom"),
    ("ring_switch", r"Flock_ringSwitch|Flock_basis|Flock_mulColumns|Flock_applyMap"),
    ("zerocheck", r"Flock_bindAndZerocheck|Flock_eqStep|Flock_eqTable|Flock_lagrange|Flock_skipWeights|Flock_combWeights|"
                  r"Flock_den|Flock_nodesS|Flock_nodesL|Flock_pinned"),
    ("lincheck", r"Flock_lincheck|Flock_halve|Flock_abcClaims"),
    ("transcript", r"Flock_Transcript|Flock_absorb|Flock_squeeze|Flock_sample|Flock_framed|Flock_Round"),
    ("proof_decode", r"Flock_decodeProof|Flock_Reader"),
    ("statement_setup", r"Flock_Stmt|Flock_circuitFold|Flock_folded|Flock_foldRanges|Flock_foldedPartial|Flock_tensor|"
                        r"Flock_structured|Flock_project|Flock_regionPoint|Flock_regionValue|Flock_extraClaims|Flock_spec\b|"
                        r"Flock_placeSlot|Flock_placeRange|Flock_baseOf|Flock_slotBase|Flock_partialRanges|Flock_sumUpTo|"
                        r"Flock_Setup|Flock_pcsParams|Flock_addDelta"),
    ("public_load", r"Flock_Public|Flock_FV3"),
    ("circuit_parse", r"Flock_Circuit|Flock_HmRow|Flock_Net|Flock_Comp|Flock_Layout|Flock_checkLayout|Flock_RowLeaf|Flock_Lookup"),
    ("draw_partition", r"Flock_Draw|Flock_TemplateQuery|Flock_Partition|Flock_Program|Flock_Qword"),
    ("record", r"Flock_Record|Flock_recordDraw|Flock_verifyRep|Flock_verify\b|Flock_verifyCmd"),
)

KERNELS: tuple[tuple[str, str], ...] = (
    ("sha512", r"Sha512|sha512"),
    ("sha256", r"Sha256|sha256"),
    ("gf128", r"F128|F256|Flock_Field|clmul"),
    ("json", r"Lean_Json|Json"),
    ("alloc_rc", r"lean_alloc|lean_dec_ref|lean_inc_ref|lean_free|lean_del|lean_mark|lean_dealloc|lean_inc_heartbeat|"
                 r"^malloc|^free|_int_free|_int_malloc|^cfree|^mi_|lean_mk_|lean_copy_expand"),
    ("memcpy", r"memcpy|memmove|memset|memcmp"),
    ("bytearray", r"ByteArray|lean_byte_array|lean_sarray|lean_copy_byte_array"),
    ("nat_bignum", r"lean_nat|mpz|__gmp|Nat_"),
    ("array", r"lean_array|lean_copy_array|Array_"),
    ("string", r"String|lean_string"),
)

_PHASE_RE = [(n, re.compile(p)) for n, p in PHASES]
_KERNEL_RE = [(n, re.compile(p)) for n, p in KERNELS]


_FRAME = re.compile(r"\s*[0-9a-f]+\s+(\S+)")


def samples(lines: Iterable[str]) -> Iterator[list[str]]:
    """Each sample's frames, leaf first, from ``perf script -F comm,ip,sym`` with call chains: a sample is its header line (the
    command, right-aligned, so indented) and then one frame per line (an address, then the symbol), ended by a blank line."""
    frames: list[str] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            if frames is not None:
                yield frames
            frames = None
        elif (m := _FRAME.match(line)) and frames is not None:
            frames.append(m.group(1))
        else:
            if frames is not None:
                yield frames
            frames = []
    if frames is not None:
        yield frames


def _first(frames: list[str], table: list[tuple[str, re.Pattern]]) -> str | None:
    for f in frames:
        for name, rx in table:
            if rx.search(f):
                return name
    return None


def attribute(frames: list[str]) -> tuple[str, str]:
    """A sample's (phase, kernel): the innermost frame naming each, else ``other``."""
    return _first(frames, _PHASE_RE) or "other", _first(frames, _KERNEL_RE) or "other"


def times(stderr: str) -> dict[str, float]:
    """The verifier's own seconds: ``inputs``, ``setup`` (summed over setups) and the sessions' (summed)."""
    out = {"inputs_s": 0.0, "setup_s": 0.0, "session_s": 0.0}
    for ln in stderr.splitlines():
        if m := re.match(r"inputs: (\d+) ms", ln):
            out["inputs_s"] += int(m.group(1)) / 1000
        elif m := re.match(r"setup( \(drawn\))?: (\d+) ms", ln):
            out["setup_s"] += int(m.group(2)) / 1000
        elif m := re.match(r"\S+: (\d+) ms$", ln):
            out["session_s"] += int(m.group(1)) / 1000
    out["session_s"] = max(0.0, out["session_s"] - out["setup_s"])
    return out


def summarize(sample_frames: Iterable[list[str]]) -> dict[str, Any]:
    phase, kernel, cell, leaf = Counter(), Counter(), Counter(), Counter()
    n = 0
    for frames in sample_frames:
        if not frames:
            continue
        n += 1
        p, k = attribute(frames)
        phase[p] += 1
        kernel[k] += 1
        cell[(p, k)] += 1
        leaf[frames[0]] += 1
    frac = lambda c: {k: round(v / n, 4) for k, v in c.most_common()} if n else {}  # noqa: E731
    return {"samples": n, "phase": frac(phase), "kernel": frac(kernel),
            "phase_kernel": {f"{p}/{k}": round(v / n, 4) for (p, k), v in cell.most_common(40)} if n else {},
            "leaf": {s: round(v / n, 4) for s, v in leaf.most_common(40)} if n else {}}


def read_script(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        return f.readlines()


def result(summary: dict[str, Any], time_line: str, stderr: str, *, bundle: str | None, prove_run: str | None,
           verdicts: list[dict]) -> dict[str, Any]:
    """The profile run's ``research/result/v0.1``: wall and CPU seconds, threads kept busy, the verifier's own setup and session
    seconds, and every phase's and kernel's share of the samples."""
    import os

    wall, user, sys_, rss_kb = (float(x) for x in time_line.split()[:4])
    t = times(stderr)
    ms = [{"name": "profile.wall_s", "value": wall, "unit": "s"},
          {"name": "profile.cpu_s", "value": round(user + sys_, 3), "unit": "s"},
          {"name": "profile.threads_busy", "value": round((user + sys_) / wall, 3) if wall else None, "unit": "ratio"},
          {"name": "profile.peak_rss_bytes", "value": int(rss_kb * 1024), "unit": "B"},
          {"name": "profile.samples", "value": summary["samples"], "unit": "count"}]
    ms += [{"name": f"profile.{k}", "value": round(v, 3), "unit": "s"} for k, v in t.items()]
    ms += [{"name": f"profile.phase.{k}", "value": v, "unit": "ratio"} for k, v in summary["phase"].items()]
    ms += [{"name": f"profile.kernel.{k}", "value": v, "unit": "ratio"} for k, v in summary["kernel"].items()]
    ran = summary["samples"] > 0 and bool(verdicts)
    return {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
            "workload_fingerprint": {"kind": "proof-throughput/verifier-profile/v1", "bundle": bundle, "prove_run": prove_run,
                                     "profiler": "perf record --call-graph dwarf", "verdicts": verdicts},
            "validation": {"status": "passed" if ran else "failed",
                           "checker": "perf samples of one flock-verify verify session, and its VERDICT line",
                           "detail": f"{summary['samples']} samples; verdict {verdicts[0] if verdicts else 'none'}",
                           "evidence": ["out/profile.json", "out/perf-self.txt", "out/perf-children.txt"]},
            "measurements": ms, "measurement_files": ["out/profile.json"], "artifacts": []}


__all__ = ["KERNELS", "PHASES", "attribute", "read_script", "result", "samples", "summarize", "times"]
