#!/usr/bin/env python
"""[A3 contract 2026-09-10] ``verifier_v2/reverify.py`` (D6, below verbatim) copied and changed ONLY as listed -- the contract-ordered post-hoc
verifier (docs/PROTOCOL.md; conformance R6/R8/R13/R16/R17/R18).  ``verifier_v2/`` itself is not edited.  Changes:
 (a) V8.instance: the instance must declare the epoch and B_ex as an RU count (``inst.is_contract_instance``); a legacy instance is REFUSED
     (NOT BOUND), never re-labelled;
 (b) V8.step2 / V8.step4: ``e2e.verifier_v3.steps.step2_check`` / ``step4_check`` re-run post hoc over the same files the live driver checked
     (idempotent; the live placement is the contract's, this is the record's evidence that the checks hold);
 (c) V8.exemption_request: ``adapter/selective/exemption_request.json`` exists, its sha256 is the driver's, F == the chronology's
     ``exemption_request`` event, |F| <= B_ex, F ⊆ J; V8.ru_fields: ``verify/ru_fields.json`` covers J with both R17 fields;
 (d) V5.uncovered (replaces the legacy 'exempt (charged)' counting): a selected VU whose kind has no checker and whose RU is NOT in F is an
     uncovered selected equation -> FAIL (HC).  ``B_ex`` for the Bound = the instance's declared RU-count budget, never a realized VU count;
 (e) [integration note #2] NO Bound and NO certificate are computed here: the legacy Bound step (``rec_p1_verify.py:348`` passed
     ``len(exempt)`` as ``allowance``) is not carried into verifier_v3.  The verifier writes ``record_v3.json`` (``veritor/verifier-record/v3``:
     status, declared epoch, ``B_ex`` as ``{value, unit: "RU", provenance}``, law, J/F/Q, output domain, completion risk, checks) and
     ``e2e.verifier_v3.certificate_inputs`` maps it onto ``e2e.certificate_v2.build_certificate``'s inputs (A9 adds A1's population);
 (f) [R4 M1 / R9, integrator note 19:20Z] the legacy V5 (``provenance.units`` matching via ``unit_matches``, root hex only, count
     comparison, the prover's ``selected_vus`` claims) is DELETED.  V5 is A5/6's ``e2e.coverage_v1`` on A2's ``veritor.interface.expected``
     over the verifier's own derivation (``e2e.verifier_v3.expected``: located units of the pinned program + committed id lists with
     ``ids_digest`` recomputed + the Description): ``V5.trees``, ``V5.expected`` (``expected_statements`` raising => REFUSED),
     ``V5.positions`` (``check_expected_positions`` per accepted proof: root/count/framing, rank == position, offsets, K, subcircuit digest,
     Input pins, no unused element), ``V5.coverage`` (every expected statement covered exactly once by an accepted proof; no duplicate hides
     an omission; no unexpected obligation).  An ``AcceptedProof`` is built ONLY after V4 (approved guest + ``check_binding``).  Trees
     admitted = the registered weights root + committed trees of RETAINED RUs.  ``coverage.json`` = ``CoverageReport.to_json()`` + digests;
 (g) [R4 M2] ``V8.output_copies``: A2's ``obligations.output_copy_checks`` over every retained RU (the live driver runs it at step 4,
     before r_s; here re-run over the record); ``V7.instance_domain``: the declared output domain read from the instance;
 (h) [E9 EOD 2026-09-11, brief interfaces 4 + 6] ``V3.interior_layout`` (the pinned ``commitments.interior_layout`` == E2's ``layout_digest()``
     and every REQUIRED interior region of every retained RU supplied, ``e2e.interior.layout.validate_supplied``; fail-closed when an interior
     value is in the actual Q and the pin is absent or unvalidatable), ``V8.interior_regions`` (the live driver ran the same check at step 4,
     before r_s), ``V5.accounting`` (coverage BY VU IDENTITY: ``coverage.json#per_vu`` names every sampled VU of a retained RU exactly once as
     covered | input | uncovered | duplicate; ``coverage.json#counts`` reports sampled VUs, covered VUs and proof objects as three separate
     numbers -- |Q| is never "proofs + input equations").
The V-check names, predicates and messages of D6 are otherwise unchanged; ``--compare`` still works against L1's/D6's logs (differences are
expected in V5 and B; ``B.certified`` / ``V5.no_extra`` are not run here by design).  Final status is VERIFIED / REFUSED, never BOUND.
D6's docstring follows.

[D6 VERIFIER-SPLIT 2026-09-10] L1's verifier (``scripts/rec_p1_verify.py``, copied verbatim, then changed ONLY as listed) with the
verifier-side work split and timed: the SP1 proof check runs in one warm ``verify-many`` host process (vk derived once), ``check_binding``
(lowering-digest recomputation) and the expected-statement checks are timed apart, every V-stage has a timer, and every output goes under
``--out`` (never into the record).  Every check keeps L1's name, predicate and message; ``reverify.log`` is the same line format as
``verifier.log``.  Changes from L1's script: (1) ``--out`` replaces ``<record>/verify/``; nothing is appended to ``<record>/costlog.jsonl``;
(2) ``--reuse-host-verify`` / ``--dry-run-no-host-verify`` (DEV paths) removed; (3) V4 per-proof host verification through
``verifier_v2.host_verify.VerifyMany`` (sequential, one process) instead of 7 parallel ``verify`` subprocesses; (4) timers + ``timings.json``;
(5) ``--compare`` re-reads L1's ``verifier.log`` and reports check-by-check agreement (``compare.json``).  L1's docstring follows.

[L1 RECORD 2026-09-10] The verifier's checks, the certified Bound and the certificate of the REC-P1 record (DIRECTIVE v3 §1.1 steps 8-9).

Everything here is re-derived from the record's files by the verifier role (nothing is copied from prover-side summaries without a check):

 V1  instance / population / partition digests: the population file's sha256 is the instance's pin; its partition digest is the validated
     P1/cu=whole digest the expressibility table and the selection carry; program digest as pinned.
 V2  chronology re-derivation (``e2e.partition_sample.verify``): seed commitment, r_q (with the logged redraw chain), r_s, selected.json.
 V3  commitments: sha256(commitments_phase0.json) == the chronology's boundary digest; sha256(commitments.json) == its interior digest;
     the weights tree root == the registered weight commitment's root (ROT-INPUT); weights / request trees unchanged by the replay.
 V4  guest: every proof's vk == the approved ELF's vk (e2e/approved_guests.json); host ``verify`` of every compressed proof against its
     statement bytes; ``check_binding`` of the public values (h_O || h_L || verdict = 1) against the chunk's typed object.
 V5  expected statements: every selected VU of a provable kind is exactly one GemmCoordinate obligation (its Gemm / t / j as located in the
     pinned program), every selected input unit is an Input equation whose expected word is the pinned checkpoint's (or the served prompt's),
     every root a statement references is the registered weights root or a committed tree of the selected RUs; no extra obligation.
 V6  input checks (``obligations/input_checks.jsonl``): every selected input unit's committed word == the prescribed value.

No Bound is computed here (see (e)); the legacy ``bound_certified(KindTable(... with Input kinds ...), allowance = len(exempt))`` of D6 / L1 is
NOT part of verifier_v3 (A10 WR-3 / WR-4: the key ``allowance_B_ex`` and the Input-bearing KindTable existed only in ce30479b and were removed
in 3a10d04b, before A10's review base 72db83f2).  ``record_v3.json`` carries ``B_ex = {value, unit: "RU", provenance}`` and labels the
population (``record.population``: construction, partition_digest_kind, whether Input kinds are VUs) so ``certificate_v2`` can refuse or
subtract; the Bound is built by A9 from A1's / A9's population through ``certificate_v2.build_certificate``.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent  # e2e/verifier_v3/
POC = HERE.parent.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from verifier_v2.host_verify import VerifyMany, host_info  # noqa: E402  (read-only reuse of D6's warm verify-many client)

from e2e.verifier_v3 import accounting  # noqa: E402  [E9, SCALE] bytes read / hashed per check: ARCHIVE vs DEPLOYED (verify_accounting.json)
from e2e.verifier_v3 import steps  # noqa: E402
from verity_vllm_adapter import response_record as rr  # noqa: E402

P1_DIGEST = "56ebcdf53f302040870776981860ffcfafd161b1b962c759df8833057aa75b9a"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.fails: list[str] = []
        self.na: list[dict[str, Any]] = []  # [D20] NOT_APPLICABLE:<check> entries (registered programs)
        self.n = 0

    def check(self, code: str, ok: bool, what: str, **kw) -> bool:
        self.n += 1
        tail = " ".join(f"{k}={str(v)[:400] if isinstance(v, dict) else str(v)[:96]}" for k, v in kw.items())  # dict values (e.g. uncovered_by_kind) kept whole up to 400 chars
        line = f"{utc()} {code} {'PASS' if ok else 'FAIL'} {what} {tail}".rstrip()
        self.lines.append(line)
        print("[reverify_v3]", line[:220], flush=True)
        if not ok:
            self.fails.append(f"{code}: {what}")
        acct = accounting.current()
        if acct is not None:  # [E9, SCALE] the work since the previous boundary is this check's accounting row (explicit sections override)
            acct.check_boundary(code)
        return ok

    def note(self, text: str) -> None:
        self.lines.append(f"{utc()} NOTE {text}")

    def not_applicable(self, code: str, reason: str, *, replaced_by: str | None = None) -> None:
        """[E9, D20] a check that has no meaning for this program kind is NAMED as not run (``NOT_APPLICABLE:<code>``), with the reason and
        the check that carries the same fact when there is one -- never silently skipped, never counted PASS."""
        self.na.append({"check": code, "reason": reason, "replaced_by": replaced_by})
        line = f"{utc()} NOT_APPLICABLE:{code} {reason}" + (f" replaced_by={replaced_by}" if replaced_by else "")
        self.lines.append(line)
        print("[reverify_v3]", line[:220], flush=True)

    def write(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n")


class Timers:
    """Stage clocks (perf_counter seconds) and per-proof rows, written to ``timings.json``."""

    def __init__(self, progress: Path | None = None) -> None:
        self.stages: dict[str, float] = {}
        self.proofs: list[dict[str, Any]] = []
        self.misc: dict[str, Any] = {}
        self.progress = progress  # [E9] one JSON line per stage as it completes (survives a SIGKILL by the laptop guardian)

    def stage(self, name: str, t0: float) -> float:
        self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0)
        # [E9] peak RSS after each stage (MB): the laptop guardian kills a verifier over 2.5 GB, so the record says where the memory went
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mb = round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)
            self.misc.setdefault("peak_rss_mb_after", {})[name] = mb
            try:  # current RSS (what the guardian measures), not the high-water mark
                import subprocess

                cur = round(int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], timeout=5).split()[0]) / 1024, 1)
            except Exception:  # noqa: BLE001
                cur = None
            self.misc.setdefault("rss_mb_after", {})[name] = cur
            if self.progress is not None:
                with self.progress.open("a") as f:
                    f.write(json.dumps({"stage": name, "seconds": round(self.stages[name], 3), "peak_rss_mb": mb, "rss_mb": cur, "utc": utc()}) + "\n")
        except Exception:  # noqa: BLE001  # pragma: no cover
            pass
        return self.stages[name]


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _kind_entries(pop_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The population's kind table rows (population/v2 ``kinds.entries``; A9's v5 the same shape), [] when absent."""

    k = pop_doc.get("kinds")
    if isinstance(k, dict) and isinstance(k.get("entries"), list):
        return [e for e in k["entries"] if isinstance(e, dict)]
    return []


class CheckpointNotBound(ValueError):
    """[E9, D40 T-1] the verifier's `--checkpoint` copy is not the manifest-pinned file (missing, unreadable or another sha256)."""


def _checkpoint_word_reader(checkpoint_dir: str, case: str):
    """A ``(cid, offset) -> (word | None, source)`` reader over the verifier's own copy of the pinned checkpoint; the file sha256 is checked
    against ``manifests/checkpoints.json`` once (mirrors rec_p1_obligations.checkpoint_word, executed in the VERIFIER process)."""
    import struct

    from verity_vllm_adapter import profile as vprofile
    from verity_vllm_adapter.weight_sources import hf_tensors

    # [E9, NIGHT 14:35Z -- n2c on the laptop] the manifest's PIN (safetensors sha256) is what binds the verifier's copy; `checkpoint_for` also
    # requires the PROVER pod's `local_path` to exist on this host (`/workspace/hf/...`), which is not a verifier fact -- read the pin directly.
    cp = vprofile.load_checkpoints(str(POC / "manifests" / "checkpoints.json"))[vprofile.CASES[case]["repo"]]
    st_path = Path(checkpoint_dir) / "model.safetensors"
    st_sha = sha256_file(st_path.resolve())
    want = (cp.get("safetensors_sha256") or {}).get("model.safetensors")
    if want is None or st_sha != want:  # the manifest MUST pin the file; an unpinned checkpoint establishes nothing
        # [E9, D40 T-1] a NAMED refusal (`V6.checkpoint` FAIL, record REFUSED), never a bare exit: the caller logs it and V6.gather / V6.inputs
        # run without a checkpoint word (GATHER_UNRESOLVED by name where the record ships no weights bytes)
        raise CheckpointNotBound(f"CHECKPOINT_NOT_BOUND: verifier checkpoint {st_path} sha256 {st_sha} != manifests/checkpoints.json pin {want} for case {case!r}")
    with st_path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    cache: dict[str, bytes] = {}

    def word(cid: str, offset: int):
        hf = hf_tensors(cid[2:])
        if hf is None:
            return None, "derived table (not a checkpoint tensor)"
        if cid not in cache:
            parts = []
            with st_path.open("rb") as f:
                for name in hf[0]:
                    s0, e0 = header[name]["data_offsets"]
                    f.seek(base + s0)
                    parts.append(f.read(e0 - s0))
            cache[cid] = b"".join(parts)
        b = cache[cid]
        return int.from_bytes(b[2 * offset:2 * offset + 2], "little"), f"verifier checkpoint model.safetensors@{st_sha[:12]}:{'+'.join(hf[0])}[{offset}]"

    return word


#: [item 3] the instance/population formats the shipped verifier can be told to use.  Each entry registers its loader pair into
#: ``partition_sample.INSTANCE_FORMATS`` under its name when selected (lazy import; nothing is reassigned on the module).
INSTANCE_FORMAT_LOADERS: dict[str, str] = {
    "contract": "e2e.protocol_instance.load_instance + e2e.partition_sample.load_population (partition-intervals/v1)",
    "record_v5": "record_v5.instance_v5.load_instance_v5 + record_v5.population_v5.load_population_v5 (a9/legacy-minus-inputs/v1 | /v2)",
}

#: [E9, D19 2026-09-11] THE registry of admitted ``partition.partition_digest_kind`` names per instance format -- one place, by name, shared by
#: ``reverify.py`` and ``driver.py`` (both install through ``install_instance_format``) and the a8 harness.  ``V8.instance_format`` /
#: ``INSTANCE_REFUSED`` refuse any kind not listed here (fail closed).  ``record_v5`` admits E6's v2 partition identity
#: (``e2e.partitions.legacy_minus_inputs_v2``, ``population_v3.A9_PARTITION_DIGEST_KIND_V2``) next to A9's v1; the loader named for the format
#: must still load the population it is handed (a v2 file under a v1-only loader is the loader's refusal, by name).
ADMITTED_PARTITION_DIGEST_KINDS: dict[str, tuple[str, ...]] = {
    "contract": ("partition-intervals/v1",),
    "record_v5": ("a9/legacy-minus-inputs/v1", "a9/legacy-minus-inputs/v2"),
}


def install_instance_format(ps: Any, name: str) -> tuple[Any, Any]:
    """Register (once) and return the named loader pair.  ``record_v5`` imports A9's modules explicitly; it does NOT call ``population_v5.install()``
    (which reassigns ``partition_sample.load_population`` process-wide) -- the pair is passed by name to every ``partition_sample`` entry point.
    The admitted kinds come from ``ADMITTED_PARTITION_DIGEST_KINDS`` and are cross-checked against the modules' own constants by name."""

    if name == "record_v5" and "record_v5" not in ps.INSTANCE_FORMATS:
        from record_v5 import instance_v5 as i5
        from record_v5 import population_v5 as p5

        admitted = ADMITTED_PARTITION_DIGEST_KINDS["record_v5"]
        if str(p5.PARTITION_DIGEST_KIND) not in admitted:  # the loader's own v1 name must be one of ours (a renamed constant is a refusal, not a drift)
            raise RuntimeError(f"record_v5.population_v5.PARTITION_DIGEST_KIND {p5.PARTITION_DIGEST_KIND!r} is not in the admitted registry {admitted!r}")
        theirs = getattr(p5, "PARTITION_DIGEST_KINDS", None)  # E10's tuple when present: must not admit anything ours does not
        if theirs is not None and set(map(str, theirs)) - set(admitted):
            raise RuntimeError(f"record_v5.population_v5.PARTITION_DIGEST_KINDS admits {sorted(set(map(str, theirs)) - set(admitted))!r} beyond the verifier's registry {admitted!r}")
        ps.register_instance_format("record_v5", i5.load_instance_v5, p5.load_population_v5, partition_digest_kind=admitted)
    elif name == "contract" and "contract" in ps.INSTANCE_FORMATS:
        if tuple(ps.INSTANCE_FORMATS["contract"][2] or ()) != ADMITTED_PARTITION_DIGEST_KINDS["contract"]:
            raise RuntimeError(f"partition_sample registers 'contract' with {ps.INSTANCE_FORMATS['contract'][2]!r}, the verifier's registry says {ADMITTED_PARTITION_DIGEST_KINDS['contract']!r}")
    return ps.loaders(name)


def _stopped_before_verdict(rec: Path) -> tuple[str, str, str, list[str]] | None:
    """``(step, reason_code, first problem, missing files)`` when the record lacks the artefacts a completed protocol writes (``protocol.json``,
    ``chronology/chronology.json``, ``chronology/selected.json``); the step and code come from ``verifier_exit.json`` (the live driver's exit) or
    ``protocol_exit.json`` (E10's runner), else ``unknown`` -- by name, never guessed as OK.  ``None`` when every artefact is present."""
    missing = [n for n in ("protocol.json", "chronology/chronology.json", "chronology/selected.json") if not (rec / n).exists()]
    if not missing:
        return None
    step, code, first = "unknown-step", "STOPPED_BEFORE_VERDICT", "no verifier_exit.json / protocol_exit.json names the refusal"
    for name in ("verifier_exit.json", "protocol_exit.json"):
        px = rec / name
        if px.exists():
            try:
                vx = json.loads(px.read_text())
            except (OSError, ValueError):
                continue
            step = str(vx.get("step") or step)
            code = str(vx.get("reason_code") or vx.get("code") or code)
            probs = vx.get("problems") or []
            first = str(probs[0]) if probs else str(vx.get("message") or first)
            break
    return step, code, first, missing


def _write_accounting(vdir: Path, rec: Path, *, sampled_units: int | None, proofs: list[dict[str, Any]] | None, note: str | None = None) -> dict[str, Any] | None:
    """[E9, SCALE] ``verify_accounting.json`` next to the verdict: per check, bytes read / hashed / leaves / openings / wall / RSS, bucketed
    ARCHIVE (whole-record) vs DEPLOYED (the sampled units against the pinned roots) vs CHECKPOINT vs UNCLASSIFIED -- the rule is
    ``accounting.py``'s docstring, copied into the file.  ``None`` when the instrument is off (``--no-accounting``)."""
    acct = accounting.current()
    if acct is None:
        return None
    try:
        doc = acct.report(record_dir=rec, sampled_units=sampled_units, proofs=proofs, note=note)
    except Exception as e:  # noqa: BLE001 -- the instrument must never decide a verdict; a failure is written by name
        doc = {"schema": accounting.SCHEMA, "error": f"{type(e).__name__}: {str(e)[:300]}", "record": str(rec)}
    (vdir / "verify_accounting.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    b = doc.get("buckets") or {}
    if b:
        rb = doc.get("record_bytes") or 0
        print("[reverify_v3] accounting (record_bytes=%d): %s" % (rb, "; ".join(
            f"{k}: read {v['bytes_read']} B ({(100.0 * v['bytes_read'] / rb) if rb else 0:.1f}% of record) hashed {v['bytes_hashed']} B leaves {v['leaves_hashed']} wall {v['wall_s']} s"
            for k, v in b.items() if k in ("archive_check", "deployed_sparse", "checkpoint", "unclassified"))), flush=True)
    return doc


def _refuse_early(log: "Log", vdir: Path, rec: Path, *, reason: str, instance_format: str) -> int:
    """A refusal before the instance is loaded: reverify.log + a minimal record_v3.json (status REFUSED, the named check) so the release gate
    reads the same artefacts as for any other refusal."""

    log.note(f"RECORD: REFUSED: {reason}")
    log.write()
    (vdir / "record_v3.json").write_text(json.dumps({"schema": "veritor/verifier-record/v3", "utc": utc(), "status": "REFUSED", "status_line": "REFUSED: " + reason[:400],
                                                       "record": {"path": str(rec), "instance_format": instance_format}, "checks": [ln for ln in log.lines if " FAIL " in ln or " PASS " in ln],
                                                       "note": "refused before the instance loaded: no chronology / proofs / coverage were examined"}, indent=1) + "\n")
    _write_accounting(vdir, rec, sampled_units=None, proofs=None, note="refused early: " + reason[:200])
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--expressibility", default=None, help="Serve_v1: the pinned expressibility table (required for a serve program; NOT_APPLICABLE for a registered one, whose kind table is the population's)")
    ap.add_argument("--checkpoint", default=None, help="[R9] the verifier's OWN copy of the pinned checkpoint dir (model.safetensors; sha256 checked against manifests/checkpoints.json) -- the prescribed word of a sampled weight Input VU is read from it; without it any sampled Input VU is NOT established (V6.inputs_rederived fails closed)")
    ap.add_argument("--host", required=True)
    ap.add_argument("--approved", default=str(POC / "e2e" / "approved_guests.json"))
    ap.add_argument("--out-bits-per-ru", type=int, default=None, help="[WR-6] cross-check only: when given must equal the instance's outputs.W_out_bits (sum multiplicity x declared bits); the domain itself is read from the pinned instance (V7.output_domain)")
    ap.add_argument("--family", default=None, help="the interface family of the fixed circuit's relations (coverage_v1 Circuit.kind_program); default = the instance's epoch.family pin; when both are given they must agree (V1.family); legacy instances without the pin: veritor.tc-ampere-bf16@2")
    ap.add_argument("--resolution", type=int, default=16, help="unused here (no Bound is computed by verifier_v3); kept so D6 command lines still parse")
    ap.add_argument("--out", required=True, help="output directory (reverify.log, record_v3.json, coverage.json, host_verify.jsonl, timings.json, compare.json); never inside the record")
    ap.add_argument("--compare", default=None, help="L1's verifier.log to compare check-by-check (default: <record>/verify/verifier.log if present)")
    ap.add_argument("--info-from-ready", action="store_true", help="[default since E9 SCALE 2026-09-12 23:10Z; accepted for old command lines] take the host identity (elf_sha256, vk_hash) for V4.guest from the verify-many ready line instead of a separate `info` call (which derives the vk a second time: MEASURED 30.6 s of n4's 93.6 s wall on the pod)")
    ap.add_argument("--host-info-call", action="store_true", help="the pre-23:10Z path: a separate host `info` subprocess for V4.guest (derives the vk twice); for A/B measurement only")
    ap.add_argument("--guest-features-from-info", action="store_true", help="with --info-from-ready, still call `info` once for guest_features (not needed for any check; L1 did not check it either)")
    ap.add_argument("--elf", default=str(POC / "results" / "GUEST-2026-09-09" / "build-1" / "veritor-zk-guest.elf"), help="the archived approved ELF for V4.elf_archived")
    ap.add_argument("--instance-format", default="contract", choices=sorted(INSTANCE_FORMAT_LOADERS),
                    help="[integrator 23:18Z item 3] which instance/population loader pair the verifier uses, EXPLICITLY (no process-wide monkeypatch): "
                         "'contract' = e2e.protocol_instance / e2e.partition_sample; 'record_v5' = A9's record_v5.instance_v5 (enumerated allowlisted violations) + "
                         "population_v5 (a9/legacy-minus-inputs/v1 or /v2, ADMITTED_PARTITION_DIGEST_KINDS). Validated against the instance's partition.partition_digest_kind (V8.instance_format)")
    ap.add_argument("--no-accounting", action="store_true", help="[E9, SCALE] do not install the data-accounting instrument (bytes read / hashed per check -> verify_accounting.json); on by default, it changes no predicate")
    ap.add_argument("--deployed-openings", action="store_true",
                    help="[E9, SCALE order 3] DEPLOYED MODE (default OFF; the archive check V3.trees_rehashed is the default): every tree is built from the record's STORED "
                         "per-leaf hashes (never from unsampled leaf bytes), V3.trees_from_leaf_hashes checks every root so folded against the pinned commitments.json, and EVERY "
                         "opening the verifier takes is authenticated leaf-bytes -> path -> PINNED root (named refusal OPENING_UNAUTHENTICATED). See e2e/verifier_v3/deployed.py")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    vdir = Path(a.out)
    vdir.mkdir(parents=True, exist_ok=True)
    if not a.no_accounting:
        acct = accounting.install()
        acct.record_dir = str(rec.resolve())
        acct.extra_paths[str(Path(a.population).resolve())] = "population"
        if a.checkpoint:
            acct.extra_paths[str(Path(a.checkpoint).resolve())] = "checkpoint"
        if a.elf:
            acct.extra_paths[str(Path(a.elf).resolve())] = "elf"
    log = Log(vdir / "reverify.log")
    tm = Timers(progress=vdir / "timings.progress.jsonl")
    T0 = time.perf_counter()
    log.note(f"A3 contract-ordered reverify (verifier_v3) of {rec} (read-only) on {os.uname().nodename}; host {a.host}")

    from veritor.interface import TypedObligationSet, check_binding
    from veritor.interface.expected import check_expected_positions  # [D28] V5.positions under the record's obligation format by name

    from e2e import partition_sample as ps

    t_v1 = time.perf_counter()
    inst_doc = json.loads((rec / "protocol_instance.json").read_text())
    # [item 3] explicit, pinned loader selection: the CLI names the format, the instance must declare the matching partition_digest_kind, and
    # partition_sample.verify re-derives the chronology under the same named pair; an InstanceError is a named refusal (V8.instance_load), not a traceback
    load_inst, _load_pop = install_instance_format(ps, a.instance_format)
    fmt_problems = ps.check_instance_format(a.instance_format, inst_doc)
    log.check("V8.instance_format", not fmt_problems, "the CLI's --instance-format matches the instance's declared partition.partition_digest_kind (loaders selected by name, recorded)",
              instance_format=a.instance_format, partition_digest_kind=(inst_doc.get("partition") or {}).get("partition_digest_kind"), problems=fmt_problems)
    if fmt_problems:
        return _refuse_early(log, vdir, rec, reason="V8.instance_format: " + "; ".join(fmt_problems), instance_format=a.instance_format)
    try:
        inst = load_inst(rec / "protocol_instance.json")
    except ps.InstanceError as e:
        log.check("V8.instance_load", False, f"protocol_instance.json loads under the {a.instance_format!r} format (every validate_instance rule)", error=str(e)[:400])
        return _refuse_early(log, vdir, rec, reason=f"V8.instance_load: {str(e)[:300]}", instance_format=a.instance_format)
    idig = ps.instance_digest(inst)
    # [E9, D42 item 3 -- E8's n2 17:30Z] a record whose live driver STOPPED before the protocol ended (refused at step 2/4, aborted prover) has no
    # protocol.json / selected.json; the shipped entry point names that earlier refusal (`REFUSED <step> <reason_code>: <first named problem>`,
    # from verifier_exit.json, the driver's machine-readable exit) and emits its verdict object -- never a FileNotFoundError traceback.
    stopped = _stopped_before_verdict(rec)
    if stopped is not None:
        step, code, first, missing = stopped
        log.check("V8.driver_exit", False, "verifier_exit.json present with reason_code OK (the live driver completed steps 1-4 and re-derived the chronology)",
                  reason_code=code, step=step, missing=missing, first_problem=first[:300])
        return _refuse_early(log, vdir, rec, reason=f"{step} {code}: {first[:300]} (record stopped before the protocol ended: {', '.join(missing)} absent)",
                             instance_format=a.instance_format)
    proto = json.loads((rec / "protocol.json").read_text())
    chron = json.loads((rec / "chronology" / "chronology.json").read_text())
    selected = json.loads((rec / "chronology" / "selected.json").read_text())
    # [E9, D20] the program kind decides which Serve_v1-only checks are NOT_APPLICABLE by name (the same switch as driver.py, D1): a registered
    # verity.ir program (protocol_instance.json#program.kind == "registered", E8's CORE) has no expressibility table, maps.json, production.jsonl,
    # capture reader, token outputs or second output copy; every other check runs the shipped function unchanged.
    pkind = steps.program_kind(inst_doc)
    if pkind == "serve":
        if not a.expressibility:
            raise SystemExit("USAGE: a Serve_v1 program needs --expressibility (the pinned table)")
        expr = json.loads(Path(a.expressibility).read_text())
    else:
        expr = None
        log.not_applicable("V1.expressibility_table", f"program.kind = {pkind!r}: no Serve_v1 expressibility table; kinds and classes come from the population's kind table + the ONE builder registry",
                           replaced_by="V1.population / V2.population_derived / V5.located")
    N = int(proto["instances"])

    # V1 digests
    # [E9 SCALE 23:10Z, read-once] the population file is read ONCE here (sha256 + parse from the same bytes); the parsed document is handed to
    # V2.population_derived (MEASURED before: 5 reads of the file per verification -- 131 MB on 32x16; the pinned-sha check is unchanged)
    _pop_raw = Path(a.population).read_bytes()
    pop_sha = hashlib.sha256(_pop_raw).hexdigest()
    pop_doc = json.loads(_pop_raw)
    del _pop_raw
    log.check("V1.instance", idig == proto["instance_digest"] == chron["instance_digest"] == selected["instance_digest"], "instance digest pinned by protocol, chronology and selection", digest=idig[:16])
    pin_sha = chron["population"]["content_sha256"]
    log.check("V1.population", pop_sha == proto["population_file_sha256"] and pin_sha == pop_doc["content_sha256"] == selected["population"]["content_sha256"] and chron["population"]["instances"] == N,
              "population file sha256 == protocol pin; content sha256 == chronology/selection pin; instances == N_REQ", sha=pop_sha[:16], content=pin_sha[:16], instances=N)
    # [WR-6] the population identity is the INSTANCE's pin (partition.partition_digest, fixed before the first commitment), not a constant;
    # P1_DIGEST is kept only to label COMPACT-P1's legacy identity in the record
    pinned_partition = str(inst_doc["partition"]["partition_digest"])
    log.check("V1.partition", pop_doc["partition_digest"] == pinned_partition == selected["partition_digest"] == (expr["partition_digest"] if expr is not None else pinned_partition),
              "population / selection / expressibility partition digest == the instance's pinned partition_digest", digest=pinned_partition[:16], legacy_compact_p1=(pinned_partition == P1_DIGEST),
              kind=inst_doc["partition"].get("partition_digest_kind"))
    # [user directive 16:04 PT / integrator 23:18Z item 1] the digest comparisons above are NECESSARY (V2.population_file names them); the verifier
    # must also DERIVE the expected population from the pinned instance (A1b population_v3) -- V2.population_derived, fail closed
    log.check("V2.population_file", pop_sha == proto["population_file_sha256"] and pin_sha == pop_doc["content_sha256"] and pop_doc["partition_digest"] == pinned_partition,
              "COMPARED ONLY: population file sha256 / content sha256 / partition digest equal the protocol, chronology and instance pins (necessary, not sufficient)",
              sha=pop_sha[:16], partition=pinned_partition[:16])
    t_pd = time.perf_counter()
    pd = steps.population_derived(inst_doc, pop_doc, instances=N, self_check="full")  # read-once: the parsed document, not the path  # post hoc: O(rows) self-check of the non-derivable digests too
    tm.misc["V2_population_derived_s"] = time.perf_counter() - t_pd
    log.check("V2.population_derived", pd["ok"], "DERIVED: the population file equals the population the verifier derives from the pinned instance (program identity + partition plan): "
              "kind names, declared widths, per-kind counts, N_V, N_R, kind-table identity; row-level digests re-derived from the file's own rows",
              status=pd["status"], derived=len(pd["derived"]), compared_only=len(pd["compared_only"]), problems=pd["problems"][:3])
    # [WR-6 / epoch.family] the interface family: the instance's pin, the CLI only as a cross-check
    epoch_doc = inst_doc.get("epoch") or {}
    pinned_family = epoch_doc.get("family")
    if a.family is None:
        a.family = pinned_family or "veritor.tc-ampere-bf16@2"
    log.check("V1.family", pinned_family is None or a.family == pinned_family, "the interface family verifier_v3 checks proofs against is the instance's epoch.family pin (a --family disagreeing with it is refused; a legacy instance without the pin uses the CLI/default and is labelled)",
              family=a.family, pinned=pinned_family)
    if pinned_family is None:
        log.note("V1.family: the instance pins no epoch.family (legacy instance); family taken from --family / default")
    if pkind == "serve":
        log.check("V1.program", pop_doc["program_digest"] == expr["program_digest"] == (json.loads((rec / "maps.json").read_text())["program"]["digest"]), "program digest pinned", digest=pop_doc["program_digest"][:16])
    else:
        log.not_applicable("V1.program", "no maps.json / expressibility table on a registered program; the program digest is checked by rebuilding program.builder in this process",
                           replaced_by="V1.program_rebuilt")
    # [integrator 19:50Z] the VERIFIER builds the pinned program in this process (identity from the instance) and checks id / gates / digest;
    # the prover's obligations.json provenance is no longer what "rebuilt" means
    from e2e.verifier_v3 import locator as vloc

    prog_identity = (inst_doc.get("program") or {}).get("identity") or {}
    t_prog = time.perf_counter()
    program, prog_C, prog_problems = vloc.build_pinned_program(prog_identity, expected_digest=pop_doc["program_digest"],
                                                               expected_gates=((inst_doc.get("program") or {}).get("gates") or {}).get("value"), timings=tm.misc,
                                                               builder=(inst_doc.get("program") or {}).get("builder"))  # [D17/D20] a registered program builds by its pinned builder
    log.check("V1.program_rebuilt", program is not None and not prog_problems, "the verifier builds the instance's program (program.identity: case/LP/STEPS) in its own process: specialised id == identity.function, gates == pinned, program_digest == population/instance pin",
              identity=prog_identity.get("function"), problems=prog_problems, seconds=round(time.perf_counter() - t_prog, 2),
              build_mode=tm.misc.get("program_build_mode"))  # [LZ] "lazy": verity_vllm.ir.lazy (streamed digest, root body never materialised) | "full": Program(sp)
    tm.stage("V1.pins", t_v1)
    # V8 (A3 contract): the instance declares the epoch and an RU-count budget; the record carries the exemption request and the RU fields
    t_v8 = time.perf_counter()
    log.check("V8.instance", inst.is_contract_instance, "instance declares the epoch (id, exact request ids, RU set, B_ex as an RU count, bound scope) -- a legacy instance is refused",
              epoch=(inst.epoch.epoch_id if inst.epoch else None), B_ex=inst.exemption_budget, unit=inst.security.B_ex_unit)
    sel = rec / "adapter" / "selective"
    exreq_path = sel / "exemption_request.json"
    exreq = json.loads(exreq_path.read_text()) if exreq_path.exists() else {}
    ex_ev = next((e for e in chron["events"] if e["name"] == "exemption_request"), None)
    J = proto["J"]
    F_chron = list((ex_ev or {}).get("exemptions", []))
    # [user directive 20:10Z R9 source audit] the RU -> request-id map is RE-DERIVED from the instance pin (epoch.request_ids, sorted: RU r
    # of instance i = i * ru_count + r) and protocol.json's ru_map must agree; the derived map is what every later lookup uses
    # [W-A9-4] the driver's machine-readable exit: a record whose driver did not end with reason_code OK is not a complete audit
    vx_path = rec / "verifier_exit.json"
    vx = json.loads(vx_path.read_text()) if vx_path.exists() else None
    log.check("V8.driver_exit", vx is not None and vx.get("reason_code") == "OK", "verifier_exit.json present with reason_code OK (the live driver completed steps 1-4 and re-derived the chronology)",
              reason_code=(vx or {}).get("reason_code"), exit_code=(vx or {}).get("exit_code"), legacy=("no verifier_exit.json (legacy driver)" if vx is None else None))
    # [E9, D1/D20] the RU id layer is the driver's rule (steps.ru_ids_of): on a serve program the RU id IS the request id (one RU per request);
    # a registered program declaring epoch.ru_indices has one RU per stage call (<request>-ru<r>).  Every RU-keyed lookup below (ru/<id> and
    # boundary/<id> trees, committed RUs, retained ids for E2/E3) uses the RU id, re-derived here from the instance pin, never the record's word.
    try:
        ru_ids_d, ru_request_d, ru_rule_d = steps.ru_ids_of(inst, pkind)
    except ValueError as e:
        ru_ids_d, ru_request_d, ru_rule_d = None, None, f"NONE: {e}"
    ru_map_recorded = {int(r["ru"]): (r.get("ru_id") or r["request_id"], r["request_id"]) for r in proto["ru_map"]["rus"]}
    if ru_ids_d is not None:
        ru_map_derived = {r: (rid, ru_request_d[rid]) for r, rid in enumerate(ru_ids_d)}
        log.check("V2.ru_map", ru_map_recorded == ru_map_derived, "protocol.json ru_map == the map derived from the instance's epoch (RU id and request id per RU; steps.ru_ids_of)",
                  rus=len(ru_map_derived), rule=ru_rule_d[:80], first=[ru_map_derived[k][0][:16] for k in sorted(ru_map_derived)[:2]])
        proto["ru_map"] = {"schema": proto["ru_map"].get("schema"), "rule": f"derived by the verifier: {ru_rule_d}",
                           "rus": [{"ru": k, "ru_id": v[0], "request_id": v[1]} for k, v in sorted(ru_map_derived.items())]}
    else:
        log.note(f"V2.ru_map: {ru_rule_d}: the recorded ru_map is used as is and this record cannot be VERIFIED (V8.instance)")
        proto["ru_map"]["rus"] = [{**r, "ru_id": r.get("ru_id") or r["request_id"]} for r in proto["ru_map"]["rus"]]
    ru_of_request = {r["ru_id"]: r["ru"] for r in proto["ru_map"]["rus"]}  # RU id -> RU index (the name is historical: on a serve program the RU id is the request id)
    step4_p = rr.validate(exreq, J=J, B_ex=inst.exemption_budget, instance_digest=idig, r_q=next((e["digest"] for e in chron["events"] if e["name"] == "r_q"), None)) if exreq else ["exemption_request.json missing"]
    log.check("V8.exemption_request", bool(exreq) and ex_ev is not None and list(exreq.get("F", [])) == F_chron and not step4_p and sha256_file(exreq_path) == (proto.get("step4") or {}).get("exemption_request_sha256"),
              "exemption_request.json present, sha256 == the driver's, F == the chronology's exemption_request event, F ⊆ J, |F| <= B_ex (RU count), both R17 fields for every RU of J",
              F=F_chron, B_ex=inst.exemption_budget, problems=step4_p[:3])
    F = sorted(int(r) for r in F_chron)
    # [E9, D42 item 2 -- T1's F2 17:25Z] R8: an exemption names a SELECTED RU, and J names RUs of the epoch -- both are checked BY NAME here
    # against the verifier-derived ru_map (the summary at the end used to index `ru_map.rus[r]` for r in F and CRASHED with IndexError on T1's
    # M43 -- F = [3] on a one-RU record -- after every check had already refused: exit 1, no verdict line, no record_v3.json).  An RU index
    # outside the map is a refusal (`V8.exemption_rus`), and every later lookup names the unknown RU instead of indexing it.
    ru_index_set = {int(r["ru"]) for r in proto["ru_map"]["rus"]}
    ru_id_by_index = {int(r["ru"]): str(r["ru_id"]) for r in proto["ru_map"]["rus"]}
    unknown_J = sorted(int(r) for r in J if int(r) not in ru_index_set)
    unknown_F = sorted(r for r in F if r not in ru_index_set)
    not_selected_F = sorted(r for r in F if r not in {int(j) for j in J})
    log.check("V8.exemption_rus", not unknown_J and not unknown_F and not not_selected_F,
              "R8: every RU index of J is an RU of the epoch (the verifier-derived ru_map) and every exempted RU (F) is a selected RU of J -- an exemption naming an RU the record does not have refuses by name, never an IndexError",
              rus=len(ru_index_set), J=list(J), F=F, unknown_J=unknown_J, unknown_F=unknown_F, F_not_in_J=not_selected_F)
    F_declared = list(F)  # the record's word: F stays as declared (B.allowance / V8.ru_fields judge it); unknown RUs are NAMED in the summary, never indexed
    if unknown_J:
        # the record is REFUSED above; the RU-keyed lookups below run over the RUs the record HAS so the verdict object is still emitted (fail closed, by name)
        J = [int(r) for r in J if int(r) in ru_index_set]
    retained = [r for r in J if r not in set(F)]
    rf_path = rec / "verify" / "ru_fields.json"
    rf = json.loads(rf_path.read_text()) if rf_path.exists() else {}
    log.check("V8.ru_fields", bool(rf) and rf.get("schema") == rr.RU_FIELDS_SCHEMA and sorted(int(k) for k in (rf.get("per_ru") or {})) == sorted(J) and rf.get("F") == F
              and all((v.get("consistency") or {}).get("status") in rr.CONSISTENCY_STATUSES for v in (rf.get("per_ru") or {}).values()),
              "verify/ru_fields.json covers exactly J with consistency (established consistent/inconsistent/unknown + who) and response fields; completion risk from the actual counts",
              rus=len(rf.get("per_ru") or {}), aggregate_completion_risk=(rf.get("completion_risk") or {}).get("aggregate_p_select_at_least_one"), sampled_unsupported=rf.get("sampled_unsupported_count"))
    tm.stage("V8.contract_records", t_v8)
    # V2 chronology
    t = time.perf_counter()
    problems = ps.verify(rec / "protocol_instance.json", Path(a.population), rec / "chronology", instances=N, instance_format=a.instance_format)
    log.check("V2.chronology", not problems, "chronology re-derived (seed, r_q incl. redraws, r_s, selected.json)", problems=problems, seconds=round(time.perf_counter() - t, 1))
    tm.stage("V2.chronology", t)
    rq = next(e for e in chron["events"] if e["name"] == "r_q")
    conditioned = bool(rq.get("redraws", 0)) or bool(rq.get("conditioned"))
    # V3 commitments
    t_v3 = time.perf_counter()
    sel = rec / "adapter" / "selective"
    b_dig, i_dig = sha256_file(sel / "commitments_phase0.json"), sha256_file(sel / "commitments.json")
    b_ev = next((e for e in chron["events"] if e["name"] in ("boundary_commitment", "boundary")), None)
    i_ev = next((e for e in chron["events"] if e["name"] in ("interior_commitment", "interior")), None)
    b_logged = (b_ev or {}).get("digest") or (b_ev or {}).get("boundary_digest")
    i_logged = (i_ev or {}).get("digest") or (i_ev or {}).get("interior_digest")
    log.check("V3.boundary", b_dig == proto["boundary_digest"] and (b_logged in (None, b_dig)), "sha256(commitments_phase0.json) == chronology boundary digest", digest=b_dig[:16], logged=str(b_logged)[:16])
    log.check("V3.interior", i_dig == proto["interior_digest"] and (i_logged in (None, i_dig)), "sha256(commitments.json) == chronology interior digest", digest=i_dig[:16], logged=str(i_logged)[:16])
    c0 = json.loads((sel / "commitments_phase0.json").read_text())
    c2 = json.loads((sel / "commitments.json").read_text())
    dmode = None
    if a.deployed_openings:
        # [E9, SCALE order 3] installed HERE: after V3.boundary / V3.interior pinned both commitment files (the roots every opening is authenticated
        # against) and before the first tree is built (V8.output_copies, V3, the Deriver, V6.gather all go through the forced stored-hash caches)
        from e2e.verifier_v3 import deployed as _deployed
        dmode = _deployed.install({**c2, "trees": {**(c0.get("trees") or {}), **(c2.get("trees") or {})}})
        log.note("DEPLOYED MODE (--deployed-openings): trees from stored leaf hashes; every opening authenticated against the pinned root; refusal OPENING_UNAUTHENTICATED")
    reg = json.loads((rec / "weights_authorized.json").read_text()) if (rec / "weights_authorized.json").exists() else None
    w_root = ((c2.get("trees") or {}).get("weights") or {}).get("root")
    if reg is None and pkind == "serve":
        raise SystemExit("USAGE: a Serve_v1 record needs weights_authorized.json (the registered weight commitment)")
    if w_root is None and pkind != "serve":
        log.not_applicable("V3.weights_root", "the registered program declares no prescribed weight inputs (no weights tree in the phase-0 message)" + ("" if reg is None else "; weights_authorized.json present but no tree to compare"))
    else:
        log.check("V3.weights_root", w_root is not None and w_root == ((c0.get("trees") or {}).get("weights") or {}).get("root") == (reg or {}).get("expected_root"),
                  "prover weights root == registered weight commitment (ROT-INPUT), unchanged by replay", root=str(w_root)[:16], registered=(reg or {}).get("expected_root", "ABSENT")[:16])
    unchanged = all(c2["trees"].get(k, {}).get("root") == v.get("root") for k, v in c0["trees"].items())
    log.check("V3.no_recommit", unchanged, "every phase-0 tree (weights, request boundaries) has the same root after the replay", trees=len(c0["trees"]))
    J = proto["J"]
    sel_rids = {proto["ru_map"]["rus"][r]["ru_id"] for r in J}
    allowed_roots = {"weights": w_root} if w_root is not None else {}
    for tid, tr in c2["trees"].items():
        if tid.startswith("ru/") and tid.split("/")[1] in sel_rids:
            allowed_roots[tid] = tr["root"]
        if tid.startswith("boundary/") and tid.split("/")[1] in sel_rids:
            allowed_roots[tid] = tr["root"]
    tm.stage("V3.commitments", t_v3)
    # V8.step2 / V8.step4 (A3 contract): the live driver's checks re-run over the record (idempotent)
    t_v8b = time.perf_counter()
    if pkind == "serve":
        prod_rows = [json.loads(ln) for ln in (rec / "adapter" / "production" / "production.jsonl").read_text().splitlines() if ln.strip()]
        try:
            from verity_vllm.capture import CaptureReader as _CR

            _rd = _CR(sel)
            b_out = {rid: list(_rd.boundary(rid).output_token_ids) for rid in _rd.request_ids()}
        except Exception as e:  # noqa: BLE001
            b_out = None
            log.note(f"NOT DONE served == committed boundary comparison inside V8.step2: {type(e).__name__}: {str(e)[:200]}")
        from e2e.verifier_v3.driver import boundary_inputs as _binputs

        b_in = _binputs(sel, c0)  # WR-5 (R1 v_in(C) = x): committed prompt / seed / sampling params vs the declared inputs, authenticated by the committed root
        p2 = steps.step2_check(inst, c0, prod_rows, registered_weights_root=reg["expected_root"], boundary_outputs=b_out, boundary_inputs=b_in)
        log.check("V8.step2", not p2 and b_out is not None, "step-2 checks hold: one boundary tree per declared request, no RU tree before r_q, weights root registered, y well-typed, served == committed boundary",
                  problems=p2[:3], live=(proto.get("step2") or {}).get("problems"))
    else:
        # [D20] the program-agnostic step-2 checks (D1: steps.step2_check_registered) over the archived phase-0 message; the RU ids are the
        # driver's rule (ru_map.json, re-derived by steps.ru_ids_of); v_in(C) = x through the builder module's make_input_check when it exposes one
        prod_rows, b_in = [], None  # no served rows, no committed prompt (the registered input check below authenticates x leaf by leaf)
        ru_ids_2 = [u["ru_id"] for u in proto["ru_map"]["rus"]]  # the verifier-derived RU ids (V2.ru_map)
        input_check, ic_note = steps.registered_input_check(inst_doc, rec)
        p2, na2 = steps.step2_check_registered(inst, c0, registered_weights_root=(reg or {}).get("expected_root"), ru_ids=ru_ids_2, input_check=input_check)
        for x in na2:
            log.not_applicable(x["check"], x["reason"], replaced_by=x.get("replaced_by"))
        log.check("V8.step2", not p2, "step-2 checks hold (registered program): epoch + inputs declared, one boundary tree per RU and none extra, no RU tree before r_q, weights root vs registry when the program has weights, "
                  "v_in(C) = x through the builder module's input check (else NOT_APPLICABLE:step2.v_in_equals_x above)", problems=p2[:3], input_check=ic_note, live=(proto.get("step2") or {}).get("problems"))
    p4, F4, committed4 = steps.step4_check(inst, J=J, r_q=next(e["digest"] for e in chron["events"] if e["name"] == "r_q"), instance_digest=idig, c0=c0, c2=c2, exreq=exreq, ru_of_request=ru_of_request) if exreq else (["no exemption request"], [], [])
    log.check("V8.step4", not p4 and F4 == F and set(committed4) >= set(retained), "step-4 checks hold: F ⊆ J, |F| <= B_ex, every retained RU's interior committed, phase-0 trees unchanged, response fields present",
              problems=p4[:3], F=F4, committed=committed4, live=(proto.get("step4") or {}).get("problems"))
    # [E9, NIGHT D35 2026-09-12 -- T1's F1] V3.trees_rehashed: EVERY shipped tree's root is RECOMPUTED from its leaf FILES (leaf hashes from
    # bytes -> root) and must equal the committed root / leaf_count / ids_digest -- boundary and interior alike; the weights tree (bytes never
    # shipped; root bound to the registry above) and any tree the record declares not shipped are NOT_APPLICABLE:<tree> by name, never silently.
    t_rh = time.perf_counter()
    rh = steps.trees_rehashed(sel, c2)
    for x in rh["not_applicable"]:
        log.not_applicable(x.split(":", 2)[1], x.split(": ", 1)[1] if ": " in x else x, replaced_by="V8.step2 weights root vs registry" if x.startswith("NOT_APPLICABLE:weights") else None)
    rh_ok = rh["ok"] and rh["summary"]["fail"] == 0 and rh["summary"]["pass"] + rh["summary"]["not_applicable"] == rh["summary"]["trees"]
    if dmode is None:
        log.check("V3.trees_rehashed", rh_ok,
                  "trees-rehashed/v1: every shipped tree's root recomputed from its leaf files == the committed root (leaf_count, ids_digest too); boundary and interior alike; "
                  "a tree without shipped leaf files is NOT_APPLICABLE by name (weights: registry-bound), never silently",
                  trees=rh["summary"]["trees"], passed=rh["summary"]["pass"], failed=rh["summary"]["fail"], nbytes=rh["summary"]["nbytes"], seconds=rh["summary"]["seconds"],
                  problems=rh["problems"][:4], not_applicable=[x.split(":")[1] for x in rh["not_applicable"]])
    else:
        # [E9, SCALE order 3] deployed mode: the SAME function, its caches forced to the stored leaf hashes -- the roots folded from the record's
        # per-leaf hashes must equal the pinned roots; the leaf BYTES are authenticated per opening (V3.deployed_openings), not here
        log.not_applicable("V3.trees_rehashed", "deployed mode (--deployed-openings): leaf bytes are authenticated per opening, not rehashed in full", replaced_by="V3.trees_from_leaf_hashes")
        log.check("V3.trees_from_leaf_hashes", rh_ok,
                  "trees-from-leaf-hashes/v1 (deployed mode): every committed tree's root folded from the record's STORED per-leaf hashes (32 B x N, no leaf bytes) == the pinned root "
                  "(leaf_count, ids_digest too); boundary and interior alike; a tree without a shipped store is NOT_APPLICABLE by name",
                  trees=rh["summary"]["trees"], passed=rh["summary"]["pass"], failed=rh["summary"]["fail"], nbytes=rh["summary"]["nbytes"], seconds=rh["summary"]["seconds"],
                  problems=rh["problems"][:4], not_applicable=[x.split(":")[1] for x in rh["not_applicable"]])
    tm.misc["V3_trees_rehashed_s"] = round(time.perf_counter() - t_rh, 3)
    (vdir / "trees_rehashed.json").write_text(json.dumps(rh, indent=1, sort_keys=True, default=str) + "\n")
    log.check("V8.Q_over_retained", all(int(r["ru_index"]) in set(retained) for r in selected["selected"]),
              "every selected VU lies in a retained RU (Q drawn over J minus F)", selected=selected["count"], retained=retained)
    # V8.output_copies (A3 wiring of A2's M2, R4): the served token is committed twice -- step-1 boundary leaf output_token/t and the step-3
    # interior sampler.select/port=0 copy -- and the two must agree for every retained RU; the live driver runs this at step 4 (before r_s),
    # here it is re-run over the same record (idempotent: both trees are fixed in commitments.json, whose digest phase_s binds before r_s).
    t_oc = time.perf_counter()
    if pkind == "serve":
        oc = steps.output_copy_step4(sel, c2, [proto["ru_map"]["rus"][r]["request_id"] for r in retained])
        log.check("V8.output_copies", oc["ok"] and oc["rows"] > 0 and (proto.get("step4") or {}).get("output_copies", {}).get("ok") is not False,
                  "for every retained RU: every interior sampler.select copy authenticates against its committed RU root, the boundary output_token/t leaf against its committed "
                  "boundary root, and the two int32 words equal the served token (one row per output position; a position committed once is a failing row)",
                  rows=oc["rows"], failing=len(oc["problems"]), problems=oc["problems"][:2], live=(proto.get("step4") or {}).get("output_copies", {}).get("ok"))
    else:
        oc = {"ok": None, "rows": 0, "problems": [], "not_applicable": "registered program: one committed copy per output"}
        log.not_applicable("V8.output_copies", "A2's M2 needs two committed copies of the served token (step-1 boundary leaf + step-3 sampler.select); a registered program commits each output once",
                           replaced_by="V5.positions (the output word's committed position)")
    tm.stage("V8.output_copies", t_oc)
    tm.stage("V8.steps", t_v8b)
    # V4 guest + proofs
    t_v4g = time.perf_counter()
    approved = json.loads(Path(a.approved).read_text())
    g = {x["elf_sha256"]: x for x in approved["guests"]}
    # [E9 SCALE 2026-09-12 23:10Z, item 2 (vk derived twice)] the ready line IS the host identity by default on every path (the laptop's
    # `--info-from-ready` and the pod's / reverify_clean's previously separate `info` call): one vk derivation per verification.  The identity
    # is checked against the verifier's approved table exactly as before (V4.guest, D6.vk_once); `--host-info-call` restores the second call.
    a.info_from_ready = bool(a.info_from_ready or not a.host_info_call)
    if a.info_from_ready:
        # (a) vk derivation ONCE: the warm verify-many process is started here and its ready line ({elf_sha256, vk_hash, sp1_version, backend})
        # serves as the host `info` L1 called separately (L1's `info` call is a second ProverClient + setup(ELF): main.rs:513-530).
        t = time.perf_counter()
        _acct0 = accounting.current()
        with (_acct0.section("V4.vk_derivation_once") if _acct0 is not None else _nullcontext()):  # [E9, SCALE] the host's vk setup, not V4.guest's pin check
            vm = VerifyMany(a.host)
        tm.stage("V4.vk_derivation_once", t)
        info = {k: vm.ready[k] for k in ("backend", "sp1_version", "elf_sha256", "vk_hash")}
        info["guest_features"] = host_info(a.host).get("guest_features") if a.guest_features_from_info else []
        t_v4g = time.perf_counter()  # V4.guest's own clock excludes the derivation (timed as V4.vk_derivation_once)
    else:
        info = host_info(a.host)
    guest = g.get(info["elf_sha256"])
    # [WR-6 / user directive 20:10Z R13] approved checker identities stay tied to the required relations: the host guest must be an entry of the
    # VERIFIER's own approved_guests.json (ELF + vk) AND that entry must list the instance's epoch.family among the families its checker
    # implements (`families`, read from the guest's source tree).  epoch.approved_elf_sha256 may NARROW the set to approved entries, never
    # widen it: a pinned sha that is not an approved entry is refused, not accepted.
    pinned_elfs = (inst_doc.get("epoch") or {}).get("approved_elf_sha256")
    v4g = steps.guest_acceptance(approved, info, pinned_elfs=pinned_elfs, pinned_family=pinned_family)
    log.check("V4.guest", not v4g,
              "host guest is an approved (ELF, vk) entry of the verifier's approved_guests.json, inside the instance's narrowed set when pinned, and that entry lists the instance's epoch.family among the families its checker implements",
              elf=info["elf_sha256"][:16], vk=info["vk_hash"][:18], approval=(guest or {}).get("approval"), family=pinned_family or f"{a.family} (unpinned: tie not checked)",
              entry_families=len((guest or {}).get("families") or []), pinned_set=([x[:12] for x in pinned_elfs] if pinned_elfs else "none (any approved entry)"), problems=v4g)
    if pinned_family is None:
        log.note("V4.guest: legacy instance without epoch.family -- the family <-> ELF tie is NOT checked (the CLI/default family is used for coverage only)")
    pdir = rec / "proofs"
    proofs = json.loads((pdir / "proofs.json").read_text()) if (pdir / "proofs.json").exists() else {}
    log.check("V4.prover_summary", bool(proofs), "proofs.json written by the prover (all chunks attempted)", present=bool(proofs))
    # [W-A9-5] a record without the prover's obligations directory is refused by name, not by a traceback (no statement to verify -> every
    # selected statement of a retained RU stays uncovered; V5.coverage says so with the count)
    odir_ = rec / "obligations"
    log.check("V4.obligations_dir", odir_.is_dir(), "the record carries the prover's obligations/ directory (chunk*/composed.json + statement.bin)", present=odir_.is_dir())
    chunk_dirs = sorted(p for p in odir_.iterdir() if p.is_dir() and p.name.startswith("chunk")) if odir_.is_dir() else []
    proven: dict[str, dict] = {}
    tm.stage("V4.guest", t_v4g)

    acct_ = accounting.current()
    if not a.info_from_ready:
        # (a) vk derivation ONCE: the warm verify-many process; its ready line carries client_seconds + setup_seconds.
        t = time.perf_counter()
        with (acct_.section("V4.vk_derivation_once") if acct_ is not None else _nullcontext()):
            vm = VerifyMany(a.host)
        tm.stage("V4.vk_derivation_once", t)
    tm.misc["verify_many_ready"] = {**vm.ready, "ready_wall_seconds": vm.ready_wall_seconds}
    log.check("D6.vk_once", str(vm.ready.get("vk_hash")).lower() == str(info["vk_hash"]).lower() == str((guest or {}).get("vk_hash")).lower() and vm.ready.get("elf_sha256") == info["elf_sha256"],
              "verify-many derived the vk once; vk == host info == approved table (D6 addition: a check L1 made per call via each verify's vk_hash)", vk=str(vm.ready.get("vk_hash"))[:18], setup_s=round(float(vm.ready.get("setup_seconds", 0)), 3))
    # (b) per proof, sequential in one process: SP1 verify (host, timed apart from proof load and statement hash) ...
    verified = []
    with (acct_.section("V4.host_verify") if acct_ is not None else _nullcontext()):  # [E9, SCALE] the SP1 host reads proof + statement bytes (counted by stat under proofs)
        for cdir in chunk_dirs:
            pf = pdir / f"{cdir.name}-compressed.bin"
            if not ((pdir / f"{cdir.name}.json").exists() and pf.exists()):
                verified.append((cdir, None, None))
                continue
            try:
                verified.append((cdir, vm.verify(pf, cdir / "statement.bin", id=cdir.name), None))
            except RuntimeError as e:
                verified.append((cdir, None, str(e)[-160:]))
    vm.close()
    with (vdir / "host_verify.jsonl").open("w") as f:
        for cdir, vr, err in verified:
            f.write(json.dumps({"chunk": cdir.name, "host_verify": vr, "error": err}) + "\n")
    # ... (c) check_binding: lowering-digest recomputation from the typed object, timed apart (object parse | check_binding | statement sha256)
    t_v4p = time.perf_counter()
    obligation_formats: dict[str, str] = {}  # [D28] chunk -> the record's obligation format (one per record; V5.positions uses it)
    for cdir, vr, err in verified:
        if vr is None and err is None:
            log.check(f"V4.proof.{cdir.name}", False, "no compressed proof for this chunk", chunk=cdir.name)
            continue
        if vr is None:
            log.check(f"V4.proof.{cdir.name}", False, "host verify failed", err=err)
            continue
        t0 = time.perf_counter()
        pf = pdir / f"{cdir.name}-compressed.bin"  # [W-A8-2] THIS chunk's proof file (the first loop's `pf` is the last chunk's; a missing last proof crashed here)
        record = json.loads((cdir / "composed.json").read_text())
        st = cdir / "statement.bin"
        t1 = time.perf_counter()
        h_l = hashlib.sha256(st.read_bytes()).hexdigest()
        t2 = time.perf_counter()
        pv = bytes.fromhex(vr["public_values"])
        obj = TypedObligationSet.from_json(record["object"])
        t3 = time.perf_counter()
        # [E9, D28] the record's obligation format BY NAME decides the boundary rule (typed-obligation/v1: an aliased incoming word is legal;
        # v0 refuses it) -- one format per record, read from composed.json, never inferred from the object's shape
        ofmt = steps.record_obligation_format(record)
        obligation_formats[cdir.name] = ofmt
        b = check_binding(obj, pv, statement_format=int(record.get("statement_format", 2)), obligation_format=ofmt)
        t4 = time.perf_counter()
        rec_problems = steps.proof_record_consistency(cdir, vr["public_values"], record=record, statement_sha256=h_l)  # [D25] the record-side conjuncts, ONE shipped function (live step 5 calls the same); read-once: this loop's parse + hash
        ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") and str(vr.get("vk_hash")).lower() == str(info["vk_hash"]).lower() and b.ok and b.verdict == 1 \
            and not rec_problems
        log.check(f"V4.proof.{cdir.name}", ok, "compressed proof verifies; vk approved; public values = h_O||h_L||1 bound to the typed object; the record describes the statement it proved (steps.proof_record_consistency)",
                  obligations=len(obj.obligations), h_L=h_l[:12], binding=b.ok, host_ok=bool(vr.get("ok")), record_problems=rec_problems[:2])
        tm.proofs.append({"chunk": cdir.name, "ok": ok, "obligations": len(obj.obligations), "proof_bytes": pf.stat().st_size, "statement_bytes": st.stat().st_size,
                          "statement_format": int(record.get("statement_format", 2)),
                          "sp1_verify_s": vr.get("verify_seconds"), "proof_load_s": vr.get("load_seconds"), "host_statement_hash_s": vr.get("statement_hash_seconds"),
                          "host_job_wall_s": vr.get("job_wall_seconds"), "py_roundtrip_wall_s": vr.get("wall_seconds"),
                          "composed_json_read_s": t1 - t0, "composed_json_bytes": (cdir / "composed.json").stat().st_size,
                          # [E9 item 3] the typed object's canonical bytes (h_O's preimage; composed.json#canonical, a str -- free to measure) for deployed_ideal
                          "typed_object_canonical_bytes": (len(record["canonical"]) if isinstance(record.get("canonical"), str) else len(json.dumps(record["object"], separators=(",", ":"), sort_keys=True))),
                          "statement_sha256_s": t2 - t1, "typed_object_parse_s": t3 - t2, "check_binding_s": t4 - t3, "binding_code": str(getattr(b, "code", ""))})
        if ok:
            # [E9 21:45Z] keep the PARSED typed object, not composed.json's dict form of it: V5 needs the object (family, obligations, positions),
            # and holding both for ~140 statements is what pushed the honest run to 2.95 GB peak (the laptop guardian kills at 2.5 GB)
            proven[cdir.name] = {**{k: v for k, v in record.items() if k != "object"}, "object_typed": obj}
        del record, obj
    import gc

    gc.collect()  # [E9] composed.json's dict form (chunk00: 369 MB of JSON) is garbage now; return it before the locator allocates
    tm.stage("V4.proofs_binding", t_v4p)
    # V5 coverage of every selected equation (R9) -- A5/6's coverage_v1 over A2's expected-position module, on the verifier's own derivation
    # (e2e.verifier_v3.expected).  Deleted from the legacy V5: provenance.units matching (unit_matches), root-hex-only root checks, the
    # count comparison, and any use of the prover's selected_vus claims.  Coverage is by digest of the BOUND objects' positions.
    t_v5 = time.perf_counter()
    from e2e.coverage_v1 import check_coverage, expected_statements
    from e2e.coverage_v1.coverage import AcceptedProof, CoverageError, obligations_of_typed_object

    from e2e.verifier_v3 import expected as ex

    # [integrator 19:50Z] the located unit of every selected VU is DERIVED here: ProgramLocator over the verifier's own program build (V1.program_rebuilt)
    # applied to the verifier's selected.json rows (V2: vu set, ru, gate_interval and kind re-derived against the population).  The prover's
    # mapping.units.jsonl is never the source; when present it is compared per unit on every field the deriver reads and any disagreement refuses.
    t_loc = time.perf_counter()
    klass = {r["kind"]: r["class"] for r in expr["rows"]} if expr is not None else steps.registered_kind_classes(pop_doc)  # [D20] no table: the ONE builder registry decides
    n_v = int(pop_doc["per_instance"]["vu_count"]) if isinstance(pop_doc.get("per_instance"), dict) and "vu_count" in pop_doc["per_instance"] else int(expr["vus_per_request"])
    ru_request_of = {r["ru"]: r["ru_id"] for r in proto["ru_map"]["rus"]}  # RU index -> RU id (tree ids boundary/<id>, ru/<id>)
    units: list[dict[str, Any]] = []
    loc_problems: list[str] = []
    ploc = None
    if program is not None:
        try:
            ploc = vloc.make_locator(program, prog_C)  # [LZ] LazyProgramLocator over a lazy program, ProgramLocator over a full one
            tm.misc["V5_locator_build_s"] = time.perf_counter() - t_loc
            t_l2 = time.perf_counter()
            units = vloc.locate_selected(ploc, selected["selected"], klass=klass, ru_request=ru_request_of, n_v=n_v)
            tm.misc["V5_locate_units_s"] = time.perf_counter() - t_l2
            tm.misc["V5_locate_per_unit_ms"] = 1000.0 * (time.perf_counter() - t_l2) / max(1, len(units))
        except Exception as e:  # noqa: BLE001
            loc_problems.append(f"locator: {type(e).__name__}: {str(e)[:200]}")
    else:
        loc_problems.append("no verifier-built program (V1.program_rebuilt failed): nothing can be located")
    loc_refused = [u for u in units if isinstance(u.get("locate"), dict) and "refused" in u["locate"]]
    log.check("V5.located", program is not None and not loc_problems and not loc_refused and len(units) == selected["count"],
              "every selected VU located by the verifier's own ProgramLocator over its own program build (gate interval -> Gemm member / Input gate -> leaf+offset); none refused",
              located=len(units), refused=len(loc_refused), problems=loc_problems, seconds=round(time.perf_counter() - t_loc, 2))
    prover_units_path = rec / "mapping.units.jsonl"
    if prover_units_path.exists():
        prover_units = [json.loads(ln) for ln in prover_units_path.read_text().splitlines() if ln.strip()]
        disagreements = vloc.compare_units(units, prover_units)
        log.check("V5.prover_mapping", not disagreements, "the prover's mapping.units.jsonl (present) agrees per unit with the verifier's derivation on every field the deriver reads -- the derivation is used regardless",
                  prover_units=len(prover_units), disagreements=len(disagreements), first=disagreements[:2])
    else:
        log.note("V5.prover_mapping: no mapping.units.jsonl in the record (fine: the verifier derives the units)")
    by_vu = {u["vu_index"]: u for u in units}
    ru_by_vu = {r["vu_index"]: r["ru_index"] for r in selected["selected"]}
    log.check("V5.selection", set(by_vu) == {r["vu_index"] for r in selected["selected"]} and len(units) == selected["count"], "the located units are exactly the selected VUs", selected=selected["count"])
    # [W-A9-5] input_checks.jsonl is the prover's OPTIONAL row file (compared field by field in V6.inputs, never a source): absent == nothing to
    # compare (the verifier derives every Input unit itself); a present but unreadable file is a named refusal, not a traceback
    ic_path = rec / "obligations" / "input_checks.jsonl"
    checks: list[dict[str, Any]] = []
    if ic_path.exists():
        try:
            checks = [json.loads(ln) for ln in ic_path.read_text().splitlines() if ln.strip()]
        except (OSError, ValueError) as e:
            log.check("V6.input_checks_file", False, "obligations/input_checks.jsonl, when present, parses as JSON lines", error=f"{type(e).__name__}: {str(e)[:120]}")
    else:
        log.note("V6.inputs: no obligations/input_checks.jsonl in the record (fine: the verifier derives every Input unit; nothing to compare)")
    retained_rids = {proto["ru_map"]["rus"][r]["ru_id"] for r in retained}
    retained_units = [u for u in units if ru_by_vu.get(u["vu_index"]) in set(retained)]
    coverage_doc: dict[str, Any] = {"schema": "veritor/a3-coverage-record/v1", "utc": utc(), "family": a.family}
    lit_table = None
    try:
        import inspect as _inspect

        ci_kw: dict[str, Any] = {}
        if "population" in _inspect.signature(ex.coverage_inputs).parameters:  # [E3 2026-09-11] the PositionOracle: program + partition + pinned interior layout
            ci_kw = {"population": pop_doc, "interior_layout": steps.interior_layout_pin(inst_doc)}
        if "boundary_request_ids" in _inspect.signature(ex.coverage_inputs).parameters:
            # [E9, D32 finding on CORE-live-sampled-B1-03] step-1 `boundary/<rid>` trees of EVERY RU (roots pinned before r_q, V3.no_recommit):
            # a retained RU's statement may cite a word an unsampled RU produced; interior (step-3) trees stay retained-only
            ci_kw["boundary_request_ids"] = {t.split("/")[1] for t in c0.get("trees") or {} if t.startswith("boundary/")}
        if "public_trees" in _inspect.signature(ex.coverage_inputs).parameters:
            # [E9, NIGHT G1] the verifier-derived public trees: D23 MUFU tables (rebuilt against commitments.tables) and the `prescribed-literal/v1`
            # literal trees (steps.LiteralTable from the pinned program, only when the instance pins commitments.literals -- V3.literal_table below
            # compares the pin with the derivation; an unpinned instance leaves `literals` None and a literal-reading VU refuses LITERAL_UNPINNED)
            lit_table = steps.LiteralTable.from_program(program) if (program is not None and steps.literal_pin(inst_doc) is not None) else None
            pub, pub_problems = steps.public_trees(inst_doc, c2, program=program, literal_table=lit_table)
            ci_kw["public_trees"], ci_kw["literals"] = pub, lit_table
            coverage_doc["public_trees"] = {"trees": sorted(pub), "problems": pub_problems, "literal_table": (lit_table.pin() if lit_table is not None else None)}
            # [E9, SCALE F3] the public trees a statement may cite (pinned MUFU tables rebuilt from the verifier's fixtures + the literal trees) are a
            # CHECK of their own: a table whose fixture the verifier lacks is the NAMED refusal TABLE_FIXTURE_MISSING here -- not, as before, a
            # misattributed "root 'table/mufu/<name>' is not a committed tree of J" at V5.positions with the FileNotFoundError only in coverage.json
            tables_pinned = bool((c2.get("tables") or {}) or (((inst_doc or {}).get("commitments") or {}).get("tables") or {}))
            if pub_problems or tables_pinned:
                log.check("V5.public_trees", not pub_problems,
                          "every pinned public tree (commitments.tables: MUFU tables rebuilt from the verifier's fixture files -- a verifier input, see reverify_clean.sh; "
                          "prescribed literals from the program) rebuilt and its root == the pin; a missing fixture is TABLE_FIXTURE_MISSING by name",
                          trees=len(pub), problems=pub_problems[:4])
        with (acct_.section("V5.coverage_inputs") if acct_ is not None else _nullcontext()):  # [E9, SCALE] the Deriver + committed id lists + PositionOracle
            ci_ = ex.coverage_inputs(sel, c2, retained_request_ids=retained_rids, units=retained_units, checks=checks, family=a.family, recompute_leaves=True,
                                     config=prog_C,  # [A5/6 patch] the pinned program's C for head expectations (COVERAGE_V1_HEAD_VU)
                                     program=program, locator=ploc,  # [CK-ELEM dispatch] the record handle for the expectation-capable builders
                                     **ci_kw)
    except Exception as e:  # noqa: BLE001
        ci_ = None
        coverage_doc["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    # [E9, D12 ruling 2026-09-11] on a record WITHOUT the interior_layout pin, a sampled VU that reads an interior word is UNCOVERED BY NAME
    # (E3's PositionOracle refuses it: INTERIOR_LAYOUT_UNPINNED), never a whole-Q refusal (the merged tree of 22:00Z voided all 837 VUs of the
    # legacy TINY record through coverage.py's "cannot locate" CoverageError).  The verifier locates every VU of Q_proof ONCE here (the answers
    # are handed to expected_statements below, so no VU is located twice), moves the unpinned-interior VUs to `uncovered` (V5.uncovered fails,
    # per_vu 'uncovered', V5.accounting exact) and leaves every other locate failure to expected_statements' refusal by name.  A pinned record
    # never takes this path (the oracle places its interior words).
    d12_unpinned: dict[str, str] = {}
    d12_adapter: dict[str, str] = {}  # legacy-uncovered kinds whose NEW builder/adapter path crashed on the unpinned record (named, never hidden)
    if ci_ is not None and steps.interior_layout_pin(inst_doc) is None:
        located: dict[str, Any] = {}
        _part = ci_["partition"]
        for raw in list(ci_["Q_proof"]):
            try:
                located[str(raw)] = _part.locate(str(raw))
            except Exception as e:  # noqa: BLE001
                why = steps.unpinned_interior_locate_reason(e)  # the three spellings of "an interior word this record never committed"
                if why is not None:
                    d12_unpinned[str(raw)] = why
                elif steps.legacy_uncovered_kind(by_vu.get(raw) or by_vu.get(int(str(raw))) or {}):
                    # D12 "behave as before by name": a kind that had NO builder before 2026-09-11 was 'uncovered' on this record; its new
                    # builder/adapter failing to locate it here is recorded per VU (class + message, coverage.json) and the VU stays uncovered
                    d12_adapter[str(raw)] = f"{type(e).__name__}: {str(e)[:160]}"
                    d12_unpinned[str(raw)] = f"legacy-uncovered kind {str((by_vu.get(raw) or {}).get('kind'))!r}; builder/adapter failed: {type(e).__name__}: {str(e)[:120]}"
                else:
                    break  # expected_statements raises the same refusal by name below (never a partial map)
        if d12_unpinned:
            moved = [raw for raw in ci_["Q_proof"] if str(raw) in d12_unpinned]
            ci_["Q_proof"] = [raw for raw in ci_["Q_proof"] if str(raw) not in d12_unpinned]
            ci_["uncovered"] = list(ci_["uncovered"]) + moved

            class _LocatedOnce:  # the same partition; locate answered from the pre-pass, everything else delegated
                def __init__(self, inner: Any, cache: dict[str, Any]) -> None:
                    self._inner, self._cache = inner, cache

                def locate(self, vu_id: Any) -> Any:
                    return self._cache[str(vu_id)] if str(vu_id) in self._cache else self._inner.locate(vu_id)

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            ci_["partition"] = _LocatedOnce(_part, located)
        coverage_doc["interior_layout_unpinned"] = {"rule": "D12: a sampled VU reading an interior word of a record without the interior_layout pin is 'uncovered' by name (never a whole-Q refusal)",
                                                   "vus": len(d12_unpinned), "example": next(iter(d12_unpinned.items()), None),
                                                   "builder_adapter_failures": {"vus": len(d12_adapter), "by_exception": dict(Counter(v.split(":")[0] for v in d12_adapter.values())),
                                                                                "examples": list(d12_adapter.items())[:3],
                                                                                "note": "kinds without a builder before 2026-09-11 (uncovered on this record before by name); "
                                                                                        "their new builder/adapter crashed at locate -- a finding for the builder's owner, never a pass"}}
    # [E9, NIGHT G1 2026-09-12 -- gather-equality/v1, E7 D2] VERIFIER-COVERED kinds: a selected `GatherBf16x<V>_v1|batch` VU (h[j] = table[tok][j],
    # no arithmetic) is covered by the verifier's OWN check, not a proof: `steps.gather_equality_check` opens the token word and the weights word it
    # selects (E3's positions; authenticated openings against the committed roots) and requires the committed output word to equal it.  Such VUs
    # leave `uncovered` (they are not 'a kind without a checker') and are accounted per VU as `verifier:gather-equality/v1`; V6.gather is the check.
    gather_units = [u for u in retained_units if steps.verifier_covered_rule(str(u.get("kind", ""))) == steps.GATHER_RULE]
    gather_vus = {str(u["vu_index"]) for u in gather_units}
    if ci_ is not None and gather_vus:
        ci_["uncovered"] = [v for v in ci_["uncovered"] if str(v) not in gather_vus]
    # [E9, D40 T-1 14:55Z] the verifier's OWN checkpoint copy is a NAMED check: given -> its model.safetensors sha256 == manifests/checkpoints.json
    # pin (V6.checkpoint PASS) or the record is REFUSED by name (FAIL; no word is read from an unpinned file); not given -> NOTE, and every check
    # that needs a weights word the record does not ship says so by name (V6.gather GATHER_UNRESOLVED) -- never a pass by omission.
    cp_word = None
    if a.checkpoint:
        try:
            cp_word = _checkpoint_word_reader(a.checkpoint, str(((inst_doc.get("program") or {}).get("identity") or {}).get("case")))
            log.check("V6.checkpoint", True, "the verifier's --checkpoint model.safetensors sha256 == manifests/checkpoints.json pin (the weights words V6.gather / V6.inputs read come from a pinned file)",
                      checkpoint=str(a.checkpoint))
        except Exception as e:  # noqa: BLE001 -- CheckpointNotBound, a missing file, an unknown case: all NAMED
            log.check("V6.checkpoint", False, "the verifier's --checkpoint model.safetensors sha256 == manifests/checkpoints.json pin (the weights words V6.gather / V6.inputs read come from a pinned file)",
                      checkpoint=str(a.checkpoint), problem=f"{type(e).__name__}: {str(e)[:300]}")
    else:
        log.note("V6.checkpoint: no --checkpoint (verifier holds no checkpoint copy): weights words come from the record's shipped weights tree only; a Gather VU whose weights leaf is not shipped refuses GATHER_UNRESOLVED by name")
    gather = steps.gather_equality_check(gather_units, ci_["positions"] if ci_ else None, getattr(ci_["partition"], "record_handle", None) if ci_ else None,
                                         weights_word=cp_word) \
        if gather_units else {"ok": True, "rule": steps.GATHER_RULE, "checked": 0, "rows": [], "problems": []}
    log.check("V6.gather", gather["ok"] and gather["checked"] == len(gather_units), "gather-equality/v1: every selected Gather VU's committed output word == the embedding-table word its "
              "committed token selects (verifier's own authenticated openings of the token leaf, the weights leaf and the output leaf; 0x7FFF outside [0, V)); no proof object",
              units=len(gather_units), checked=gather["checked"], mismatches=sum(1 for p in gather["problems"] if p.startswith("GATHER_MISMATCH")), problems=gather["problems"][:3],
              rule=steps.GATHER_RULE)
    coverage_doc["gather"] = {"rule": steps.GATHER_RULE, "units": len(gather_units), "checked": gather["checked"], "ok": gather["ok"], "problems": gather["problems"],
                             "rows": gather["rows"][:64]}
    uncovered = [by_vu[v] for v in (ci_["uncovered"] if ci_ else [u["vu_index"] for u in retained_units if not (str(u["class"]).startswith("provable") or str(u["class"]).startswith("input"))])]
    # [CK-ELEM dispatch] selected VUs of retained RUs whose kind the ONE builder registry can build (their legacy table class notwithstanding):
    # they are in Q_proof and need a proof of the builder's family; reported by kind so a record made before the builders existed shows what
    # it would need (its proofs are absent -> V5.coverage uncovered, never a silent pass)
    buildable = Counter(u["kind"] for u in retained_units if isinstance(u.get("locate"), dict) and "builder" in u["locate"])
    buildable_families = sorted({u["locate"]["family"] for u in retained_units if isinstance(u.get("locate"), dict) and "builder" in u["locate"]})
    log.check("V5.uncovered", ci_ is not None and not uncovered, "no selected VU of a kind without a checker in a retained RU (an uncovered selected equation cannot be accepted: HC; the prover exempts the RU before r_s or "
              "the checker exists) -- the legacy 'exempt (charged)' family label is refused; kinds with a registered builder are proof-covered units, not uncovered",
              uncovered=len(uncovered), kinds=len({u['kind'] for u in uncovered}), uncovered_by_kind=dict(Counter(u["kind"] for u in uncovered).most_common(12)),
              buildable_by_registered_builder=sum(buildable.values()), buildable_by_kind=dict(buildable), buildable_families=buildable_families)
    coverage_doc["unsupported"] = {"uncovered": len(uncovered), "by_kind": dict(Counter(u["kind"] for u in uncovered)),
                                   "buildable_by_registered_builder": {"count": sum(buildable.values()), "by_kind": dict(buildable), "families": buildable_families}}
    tm.misc["V5_setup_s"] = time.perf_counter() - t_v5  # verifier-side location + input checks loaded, committed id lists checked (ids_digest), output-row index built
    log.check("V5.trees", ci_ is not None and not ci_["tree_problems"], "every committed tree of the retained RUs and the weights tree: recomputed ids_digest == committed, id count == leaf_count",
              trees=len(ci_["trees"]) if ci_ else 0, problems=(ci_["tree_problems"][:2] if ci_ else None))
    derived_only_vus = set(ci_["derived_only"]) if ci_ else set()
    log.check("V5.inputs_derived_only", all(str(by_vu[v]["locate"].get("field", "")).endswith("cos_sin") for v in derived_only_vus), "input units of a derived table (rope cos/sin cache: registered with the weight commitment, not a checkpoint tensor) are checked by the verifier's opening + registered leaf table only (V6), not by an Input equation", count=len(derived_only_vus))
    expected = None
    t_exp = time.perf_counter()
    # [E9, D48.2 2026-09-12 -- E10's Qwen q2] a sampled VU whose expectation the verifier cannot derive (its locate failed: E7's
    # `Bf16Add_v1|batch: LITERAL_UNPINNED` on q2) is a PER-VU NAMED refusal: it is accounted `unlocated` under the refusal's name
    # (steps.unlocated_vu_refusal), the other sampled VUs are derived and accounted, V5.expected and V5.accounting FAIL (the record is
    # REFUSED -- an unaccounted sampled VU is never accepted) and the coverage line reads `unlocated 1 (LITERAL_UNPINNED: VU 1646609
    # Bf16Add_v1|batch)`, not `unaccounted 707`.  The structural whole-Q refusals of expected_statements (a repeated VU, a digest
    # collision) still raise and void the map.
    unlocated_named: dict[str, dict[str, str]] = {}

    def _on_unlocated(vu_id: str, exc: BaseException) -> None:
        kind = str((by_vu.get(vu_id) or by_vu.get(int(vu_id)) or {}).get("kind") or "")
        name, line = steps.unlocated_vu_refusal(vu_id, kind or None, exc)
        unlocated_named[str(vu_id)] = {"name": name, "kind": kind, "line": line, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    if ci_ is not None:  # over Q_proof even when V5.uncovered already fails: the coverage report is the record's evidence either way
        try:
            with (acct_.section("V5.expected_statements") if acct_ is not None else _nullcontext()):  # [E9, SCALE] every sampled unit's expectation (openings for table ports)
                expected = expected_statements(ci_["circuit"], ci_["partition"], ci_["Q_proof"], on_unlocated=_on_unlocated)  # a structural raise => REFUSED (never a partial map)
        except CoverageError as e:
            coverage_doc["refusal"] = f"CoverageError: {str(e)[:300]}"
    if unlocated_named:
        coverage_doc["unlocated_named"] = {"rule": "D48.2: a sampled VU whose expectation the verifier cannot derive is 'unlocated' BY NAME (per VU); the others are derived and accounted; the record is REFUSED",
                                           "vus": len(unlocated_named), "by_name": dict(Counter(v["name"] for v in unlocated_named.values())),
                                           "lines": [v["line"] for v in list(unlocated_named.values())[:8]], "per_vu": unlocated_named}
        if expected is not None and not coverage_doc.get("refusal"):
            coverage_doc["refusal"] = "; ".join(v["line"] for v in list(unlocated_named.values())[:3]) + (f" (+{len(unlocated_named) - 3} more)" if len(unlocated_named) > 3 else "")
    tm.misc["V5_expected_statements_s"] = time.perf_counter() - t_exp
    refusal_text = (coverage_doc.get("refusal") or "") + " ".join(v["error"] for v in unlocated_named.values())  # the A2 disagreement marks below see per-VU errors too
    log.check("V5.description", "V5.description:" not in refusal_text, "the prover's description.json names the same operands (x row, weight tensor / row, arg_cols, N) for every located GemmCoordinate as the pinned program (A2 DescriptionDisagreement; the expectation is built from the program either way)",
              disagreement=(refusal_text[:200] if "V5.description:" in refusal_text else None))
    if "V6.inputs:" in refusal_text:
        log.check("V6.inputs", False, "A2 InputCheckDisagreement raised while deriving expectations (see V6.inputs pre-pass)", disagreement=refusal_text[:200])
    log.check("V5.expected", expected is not None and not unlocated_named and len(expected) == len(ci_["Q_proof"]), "the verifier derives the expected statement of every selected VU of a retained RU that a proof must cover (located unit + committed id lists + Description; A2 expectations; digest-unique); a VU it cannot derive is 'unlocated' by name (D48.2), the others are still derived",
              expected=len(expected) if expected else 0, Q_proof=len(ci_["Q_proof"]) if ci_ else None, unlocated=len(unlocated_named),
              unlocated_by_name=(dict(Counter(v["name"] for v in unlocated_named.values())) or None), unlocated_example=(next(iter(unlocated_named.values()))["line"] if unlocated_named else None),
              refusal=coverage_doc.get("refusal"))
    # accepted proofs: ONLY those that verified under the approved (ELF, vk) and whose public values check_binding bound to the typed object (V4.proof.*)
    t_acc = time.perf_counter()
    accepted = []
    # [CK-ELEM dispatch, re-targeted 22:20Z] a record carries objects of the audit's family AND of the expectation-capable builders' families
    # (veritor.elem-bf16@1, veritor.reduce-mufu@1).  Every accepted object's family must be one of those AND listed by the approved guest entry
    # with the interface's pinned digest (A2 check_family_approved) -- fail closed: an object of another family is not an accepted proof.
    from e2e.coverage_v1.builders import families_of
    from e2e.coverage_v1.ck_elem import ensure_families
    from veritor.interface.families import check_family_approved

    ensure_families()  # no-op once the interface carries the families (A2's expected.diff)
    allowed_families = {a.family, *families_of()}
    for name, record in proven.items():
        try:
            obj = record.get("object_typed") or TypedObligationSet.from_json(record["object"])
            fam_problems = check_family_approved(obj.family.id, str(obj.family.digest), int(obj.family.width), guest or {})
            fam_ok = obj.family.id in allowed_families and not fam_problems
            log.check(f"V4.family.{name}", fam_ok, "the object's family is the audit's or a registered builder's, and the approved guest entry lists it with the interface's pinned digest",
                      family=obj.family.id, allowed=sorted(allowed_families), problems=fam_problems[:2])
            if not fam_ok:
                continue
            accepted.append(AcceptedProof(name, obligations_of_typed_object(obj), typed_object=obj))
        except Exception as e:  # noqa: BLE001
            log.note(f"statement {name}: not an accepted proof for coverage: {type(e).__name__}: {str(e)[:200]}")
    tm.misc["V5_accepted_proofs_s"] = time.perf_counter() - t_acc
    # [E9, D28] V5.positions runs check_expected_positions under the record's obligation format by name: one format per record, else refused
    fmts = sorted(set(obligation_formats.values()))
    if len(fmts) > 1:
        log.check("V5.obligation_format", False, "every proof chunk of the record declares one obligation format (by name)", formats=fmts)
    ofmt = fmts[0] if len(fmts) == 1 else None
    position_check = (functools.partial(check_expected_positions, obligation_format=ofmt) if ofmt else None)
    report = check_coverage(expected or {}, accepted, trees=ci_["trees"], rank_of=ci_["rank_of"], position_check=position_check) if (ci_ is not None and expected is not None) else None
    if report is not None:
        coverage_doc.update(report.to_json())
        coverage_doc["expected_statements"] = {vu: {"digest": e.digest, "label": e.label, "subcircuit": e.subcircuit} for vu, e in expected.items()}
    positions_ok = report is not None and not report.position_problems
    log.check("V5.positions", positions_ok and report.positions_checked, "every accepted proof binds exactly the positions the verifier derives for the statements it covers (check_expected_positions: committed root, leaf count, framing; rank == position; offsets, K, subcircuit digest, Input pins; no unused element)",
              proofs=len(accepted), voided=(len(report.position_problems) if report else None), problems=(list(report.position_problems.items())[:1] if report else None))
    # [A8 request, integrator 23:58Z (6)] the kinds of the uncovered expected statements on the check line itself (verifier-derived units by vu_index);
    # "covered" here means covered by an ACCEPTED proof = host-verified compressed proof + check_binding + approved family -- transparent
    # verification (`verify_transparent`) is a prover-side self-check and never counts as coverage
    cov_uncovered_by_kind = dict(Counter(str((by_vu.get(int(vu)) or {}).get("kind", "?")) for vu in (report.uncovered if report else ())).most_common(12))
    if report is not None:
        coverage_doc["uncovered_by_kind"] = dict(Counter(str((by_vu.get(int(vu)) or {}).get("kind", "?")) for vu in report.uncovered))
        coverage_doc["covered_means"] = "an accepted proof: host-verified compressed proof (verify-many) + check_binding + approved family; transparent verification is not coverage"
    log.check("V5.coverage", report is not None and report.ok, "every expected statement is covered exactly once by an accepted proof (host-verified compressed proof + binding) whose positions check passed; no duplicate hides an omission; no unexpected obligation",
              expected=(report.expected_count if report else None), covered=(len(report.covered) if report else None), uncovered=(len(report.uncovered) if report else None),
              uncovered_by_kind=cov_uncovered_by_kind,
              duplicate=(len(report.duplicate) if report else None), unexpected=(len(report.unexpected) if report else None), proofs=(report.proof_count if report else None), obligations=(report.obligation_count if report else None))
    # [E9, brief interface 6] coverage BY VU IDENTITY over EVERY sampled VU of a retained RU: coverage.json#per_vu[vu] ∈ {covered, uncovered, input,
    # duplicate} -- the statement-level map from check_coverage (Q_proof) + the derived-table inputs V6 discharges + the no-checker units; the three
    # numbers (sampled VUs, covered VUs, proof objects) are reported side by side and never equated (one batched proof covers many VUs; input
    # validation is outside Q).  V5.accounting: the identity sets partition the retained sample exactly.
    per_vu_full: dict[str, str] = {}
    if report is not None:
        per_vu_full.update({str(k): v for k, v in report.per_vu().items()})
    for v in derived_only_vus:
        per_vu_full.setdefault(str(v), "input")
    for r in gather["rows"]:
        per_vu_full.setdefault(str(r["vu_index"]), f"verifier:{steps.GATHER_RULE}" if r.get("ok") else "uncovered")
    for u in uncovered:
        per_vu_full.setdefault(str(u["vu_index"]), "uncovered")
    for v in unlocated_named:  # [D48.2] the VUs whose expectation could not be derived: 'unlocated' by name (they are in Q_proof, not in `expected`)
        per_vu_full[str(v)] = "unlocated"
    # the universe is the SELECTION by identity (selected.json rows of retained RUs), never the located units: when the locator fails
    # (V5.located) every sampled VU of a retained RU is UNACCOUNTED here -- the accounting must not pass vacuously over 0 units
    # (MEASURED 00:16Z on E8's CORE-live-exhaustive-02: located=0 and this check read PASS with retained=0; closed by name)
    retained_set = set(int(r) for r in retained)
    retained_ids = {str(r["vu_index"]) for r in selected["selected"] if int(r["ru_index"]) in retained_set}
    located_ids = {str(u["vu_index"]) for u in retained_units}
    for v in sorted(retained_ids - located_ids, key=int):
        per_vu_full.setdefault(str(v), "unlocated")  # named: a sampled VU the verifier could not locate is neither covered nor uncovered
    q_proof_ids = {str(v) for v in (ci_["Q_proof"] if ci_ else [])}
    missing_ids = sorted(retained_ids - set(per_vu_full), key=int)
    unlocated_ids = sorted(v for v, s in per_vu_full.items() if s == "unlocated")
    extra_ids = sorted(set(per_vu_full) - retained_ids, key=int)
    accepted_names = sorted(p.proof_id for p in accepted)
    counts = {"sampled_vus_Q": int(selected["count"]), "sampled_vus_retained": len(retained_ids), "sampled_vus_exempted_rus": int(selected["count"]) - len(retained_ids),
              "covered_vus": sum(1 for v in per_vu_full.values() if v == "covered"), "input_vus_V6": sum(1 for v in per_vu_full.values() if v == "input"),
              "uncovered_vus": sum(1 for v in per_vu_full.values() if v == "uncovered"), "duplicate_vus": sum(1 for v in per_vu_full.values() if v == "duplicate"),
              "unlocated_vus": len(unlocated_ids),
              "verifier_checked_vus": sum(1 for v in per_vu_full.values() if v.startswith("verifier:")),
              "proof_objects_verified": len(proven), "proof_objects_accepted": len(accepted), "proof_objects_covering": (report.proof_count if report else None),
              "note": "three numbers, reported separately: sampled VUs, VUs covered by a verified expected statement, proof objects -- |Q| is never 'proofs + input equations'"}
    # [E9, D40 rule (2), 2026-09-12 15:00Z -- the user's counting rule] PROOF-CHECKED units (identities covered by an accepted proof's expected
    # statement; k proof objects is a separate number) and DIRECTLY CHECKED units (the verifier's own rule: authenticated openings + equality,
    # gather-equality/v1) are reported SEPARATELY, by identity; the completion claim is "all sampled units verified" (every retained sampled VU is
    # exactly one of proof-checked / directly checked / input (V6.inputs)), never "every unit covered by an SP1 proof".
    directly = Counter(v.split(":", 1)[1] for v in per_vu_full.values() if v.startswith("verifier:"))
    counts["proof_checked_vus"] = counts["covered_vus"]
    counts["proof_objects"] = counts["proof_objects_accepted"]
    counts["directly_checked_vus"] = dict(sorted(directly.items()))
    verified_ids = counts["proof_checked_vus"] + sum(directly.values()) + counts["input_vus_V6"]
    all_verified = (counts["sampled_vus_retained"] == verified_ids and counts["uncovered_vus"] == 0 and counts["duplicate_vus"] == 0 and not unlocated_ids and not missing_ids
                    and bool(report is not None and report.ok))
    # [D48.2] the unlocated VUs' names on the line itself: `unlocated 1 (LITERAL_UNPINNED: VU 1646609 Bf16Add_v1|batch)` -- up to three, then a count
    unlocated_why = ""
    if unlocated_named:
        shown = [v["line"] for v in list(unlocated_named.values())[:3]]
        unlocated_why = " (" + "; ".join(shown) + (f"; +{len(unlocated_named) - 3} more" if len(unlocated_named) > 3 else "") + ")"
    counts["unlocated_by_name"] = dict(Counter(v["name"] for v in unlocated_named.values()))
    counts["completion_claim"] = ("all sampled units verified" if all_verified else
                                  f"NOT all sampled units verified: {counts['sampled_vus_retained'] - verified_ids} of {counts['sampled_vus_retained']} retained sampled VUs are not "
                                  f"proof-checked / directly checked / input (uncovered {counts['uncovered_vus']}, duplicate {counts['duplicate_vus']}, unlocated {len(unlocated_ids)}{unlocated_why}, unaccounted {len(missing_ids)})")
    counts["coverage_line"] = (f"sampled VUs = {counts['sampled_vus_Q']} (retained {counts['sampled_vus_retained']}); proof-checked = {counts['proof_checked_vus']} (identities, "
                               f"{counts['proof_objects']} proof objects); " + "; ".join(f"directly checked ({r}) = {n} (identities)" for r, n in sorted(directly.items())) +
                               ("; " if directly else "") + f"input (V6.inputs) = {counts['input_vus_V6']} (identities); uncovered = {counts['uncovered_vus']}; duplicate = {counts['duplicate_vus']}"
                               f"; unlocated = {len(unlocated_ids)}{unlocated_why} -> {counts['completion_claim']}")
    coverage_doc["per_vu"] = dict(sorted(per_vu_full.items(), key=lambda kv: int(kv[0])))
    coverage_doc["counts"] = counts
    coverage_doc["proofs"] = {"chunks": [d.name for d in chunk_dirs], "verified": sorted(proven), "accepted": accepted_names}
    log.check("V5.accounting", report is not None and not missing_ids and not extra_ids and not unlocated_ids and (set(per_vu_full) >= q_proof_ids),
              "coverage by VU identity: every sampled VU of a retained RU is exactly one of covered / input (V6) / verifier:<rule> (V6.gather) / uncovered / duplicate in coverage.json#per_vu; nothing outside the sample; "
              "a sampled VU the verifier could not locate is 'unlocated' (a refusal, never vacuous); sampled VUs, covered VUs and proof objects are three separate numbers",
              sampled_Q=counts["sampled_vus_Q"], retained=counts["sampled_vus_retained"], covered=counts["covered_vus"], input=counts["input_vus_V6"], uncovered=counts["uncovered_vus"],
              duplicate=counts["duplicate_vus"], unlocated=len(unlocated_ids), unlocated_by_name=(counts["unlocated_by_name"] or None), proof_objects=counts["proof_objects_verified"],
              unaccounted=len(missing_ids), outside_sample=len(extra_ids),
              proof_checked=counts["proof_checked_vus"], directly_checked=counts["directly_checked_vus"], claim=counts["completion_claim"])
    (vdir / "coverage.json").write_text(json.dumps(coverage_doc, indent=1, sort_keys=True, default=str) + "\n")
    # [E9, brief interface 4] V3.interior_layout at step 5 with the ACTUAL Q: an interior unit in Q (E3's located origin / E6's producer kind / the
    # labelled fallback: an uncovered unit is not a boundary unit) requires the pinned commitments.interior_layout == E2's layout_digest() and every
    # required region of every retained RU supplied (validate_supplied); the live driver's step-4 verdict is compared (V8.interior_regions)
    t_il = time.perf_counter()
    il_sel = steps.interior_units_selected(retained_units, interior_kinds=steps.interior_kinds_in_population(pop_doc), fallback_uncovered=[u["vu_index"] for u in uncovered])
    il_pin = steps.interior_layout_pin(inst_doc)
    il_trees = steps.interior_trees_committed(c2)  # trees['interior/<req>/<step>'] and/or E2's interior_roots['<req>/<step>']
    il_in_play = il_sel["count"] > 0 or il_pin is not None or bool(il_trees)
    il_reason = (f"{il_sel['count']} sampled VU(s) of retained RUs read interior values [{il_sel['classification']}]: {dict(list(il_sel['kinds'].items())[:6])}" if il_sel["count"]
                 else "instance pins commitments.interior_layout" if il_pin is not None else f"commitments.json carries {len(il_trees)} interior tree(s)" if il_trees
                 else "no sampled VU reads an interior value, no pin, no interior tree")
    # the REQUIRED regions: E3's PositionOracle (coverage_inputs(..., population=, interior_layout=)["positions"], when E3's expected.py is on this
    # checkout) derives the required gates + widths per call, E2 frames them; without it E2 checks presence + rule-consistency only and the check
    # refuses every retained RU with an interior tree as 'REQUIRED regions not derived' (fail closed, labelled)
    req_src, req_label = None, "NONE: E3 PositionOracle not available (coverage_inputs has no positions)"
    L2 = steps.interior_layout_module()
    if L2 is not None and ci_ is not None and ci_.get("positions") is not None:
        try:
            req_src, req_label = steps.required_layouts_from_positions(ci_["positions"], L2, il_pin), "E3 PositionOracle (program + partition + boundary oracle) -> E2 InteriorLayout.build"
        except Exception as e:  # noqa: BLE001
            req_label = f"NONE: required source failed: {type(e).__name__}: {str(e)[:120]}"
    il = steps.interior_layout_check(inst_doc, sel=sel, c2=c2, retained_request_ids=sorted(retained_rids), interior_in_play=il_in_play, in_play_reason=il_reason,
                                     required_source=req_src, required_label=req_label)
    log.check("V3.interior_layout", il["ok"], "the pinned protocol_instance.json#commitments.interior_layout (framing chunks-4096, word order gate, tree interior/<req>/<step>) is present and "
              "== E2's layout_digest(), and every REQUIRED interior region of every retained RU is supplied (E2 validate_supplied with required= E3's derived layouts) whenever an "
              "interior value is in play; fail-closed (no pin with interior kinds selected, a pin the verifier cannot validate, or required regions the verifier cannot derive, is a refusal)",
              status=il["status"], interior_selected=il_sel["count"], classification=il_sel["classification"], pin=(str(il["digest"])[:12] if il["digest"] else None),
              layout_digest=(str(il["layout_digest"])[:12] if il["layout_digest"] else None), trees=len(il["trees"]), validator=il["validator"], required_source=il.get("required_source"),
              rus_checked={ru[:8]: bool((f or {}).get("checked_against_required")) for ru, f in ((il.get("regions") or {}).get("rus") or {}).items()} if isinstance(il.get("regions"), dict) else None,
              problems=il["problems"][:2], seconds=round(time.perf_counter() - t_il, 3))
    # [E9, NIGHT G1] V3.literal_table (`prescribed-literal/v1`): the instance's commitments.literals must equal the table the verifier derives from the
    # PINNED PROGRAM (rule, id list, per-tree root / leaf_count / ids_digest); no pin + a sampled VU of a retained RU reading a literal -> LITERAL_UNPINNED
    t_lt = time.perf_counter()
    lit_vus = steps.literal_kinds_selected([{"vu_index": u["vu_index"], **u} for u in retained_units], ci_.get("positions") if ci_ is not None else None)
    lt = steps.literal_table_check(inst_doc, program=(program if lit_table is None else None), table=lit_table, literal_vus_selected=lit_vus,
                                   in_play_reason=f"{len(lit_vus)} sampled VU(s) of retained RUs read a program literal")
    log.check("V3.literal_table", lt["ok"], "prescribed-literal/v1: protocol_instance.json#commitments.literals (rule, the sorted Const<w>[0x…]_v1 id list, per width the tree "
              "literal/w=<w> root / leaf_count / ids_digest) equals the table the verifier derives from the pinned program alone; a sampled VU reading a program literal "
              "with no pin refuses LITERAL_UNPINNED (fail closed); no pin and no literal read is vacuous",
              status=lt["status"], literals=lt["literals"], trees=lt["trees"], literal_vus_selected=len(lit_vus), problems=lt["problems"][:3], seconds=round(time.perf_counter() - t_lt, 3))
    live_il = (proto.get("step4") or {}).get("interior_layout")
    live_ok = (live_il.get("ok") is True and live_il.get("status") in ("PASS", "VACUOUS")) if isinstance(live_il, dict) else (not il_in_play)
    log.check("V8.interior_regions", live_ok,
              "the live driver ran V3.interior_layout at step 4 (BEFORE r_s: pin == layout_digest, every required interior region of every retained RU supplied) and it held; a record whose "
              "driver never ran it is refused whenever an interior value is in play (r_s was released without the check), vacuous otherwise",
              live=(None if live_il is None else {k: live_il.get(k) for k in ("ok", "status", "reason")}), interior_in_play=il_in_play,
              legacy=("no step4.interior_layout in protocol.json (driver predates the check)" if live_il is None else None))
    coverage_doc["interior_layout"] = {k: il.get(k) for k in ("ok", "status", "problems", "digest", "layout_digest", "trees", "interior_in_play", "reason", "validator", "required_source")} | {
        "selected": il_sel, "rus": ((il.get("regions") or {}).get("rus") if isinstance(il.get("regions"), dict) else None)}
    (vdir / "coverage.json").write_text(json.dumps(coverage_doc, indent=1, sort_keys=True, default=str) + "\n")
    want_coord = {u["vu_index"] for u in retained_units if str(u["class"]).startswith("provable")}
    want_input = {u["vu_index"] for u in retained_units if str(u["class"]).startswith("input")}
    tm.stage("V5.expected_statements", t_v5)
    # V6 input checks
    t_v6 = time.perf_counter()
    # [integrator 22:15Z item 2] ONE named check path: V6.inputs, computed through A2's Deriver.input_equation for every selected Input unit of
    # the retained RUs (leaf/word from the verifier's location, committed word from its own authenticated opening under the auditor's
    # registered root, ok recomputed; the prover's input_checks.jsonl is compared field by field, never read).  prompt-token units are pinned
    # by the step-2-authenticated prompt; --checkpoint adds a cross-check of the committed word against the verifier's own checkpoint copy
    # (not required: the pin is the registered root, V3.weights_root).  A8 regression 3c names exactly this check.  The prover's `ok` bits are
    # not consulted.  Vacuous (PASS over 0 units) in the one working configuration where Input kinds are out of the sampled population.
    prompts = {rid: list(bi.get("prompt_token_ids") or []) for rid, bi in (b_in or {}).items() if bi.get("root_ok") and bi.get("prompt_contiguous")}
    if ci_ is not None:
        v6 = ex.input_checks_v6(ci_["deriver"], units=retained_units, checks=checks, prompt_of_request=prompts, checkpoint_word=cp_word)
    else:
        v6 = {"problems": ["V6.inputs: coverage inputs unavailable"], "units": len(want_input), "derived_only": 0, "checkpoint_checked": 0, "extra_rows": 0}
    log.check("V6.inputs", not v6["problems"] and v6["units"] == len(want_input),
              "every selected Input unit of a retained RU: leaf/word from the verifier's location, committed word from its own authenticated opening under the registered root (the pin), ok recomputed; "
              "the prover's input_checks.jsonl agrees field by field; prompt-token units == the step-2-authenticated prompt; --checkpoint cross-checks the committed word against the verifier's checkpoint",
              input_units=v6["units"], derived_only=v6["derived_only"], checkpoint_checked=v6["checkpoint_checked"], extra_rows=v6["extra_rows"], problems=v6["problems"][:3],
              checkpoint=("verifier-held" if cp_word else "none (pin = registered root)"))
    tm.stage("V6.inputs", t_v6)
    t_elf = time.perf_counter()
    if guest is not None:
        # [E9, NIGHT 08:10Z] the archived ELF is resolved BY THE HOST'S GUEST HASH: an explicit --elf is compared as given; the default is no longer one
        # fixed file (9fd00e60's) but every ELF archived under fixtures/results/GUEST-*/ -- a host embedding another APPROVED guest (666e937f tonight)
        # must not fail against the wrong file, and a guest whose bytes are archived nowhere fails BY NAME (D10: the approved bytes are the archive).
        explicit = str(a.elf) != ap.get_default("elf")
        cands = [Path(a.elf)] if explicit else sorted(set([Path(a.elf)] + list((POC / "results").glob("GUEST-*/**/*.elf")) + list((POC / "results").glob("GUEST-*/**/veritor-zk-guest"))))
        found = next((c for c in cands if c.is_file() and sha256_file(c) == info["elf_sha256"]), None)
        if found is not None:
            log.check("V4.elf_archived", True, "archived ELF sha256 == host's embedded guest", elf=str(found), explicit=explicit)
        elif explicit or any(c.is_file() for c in cands):
            log.check("V4.elf_archived", False, "archived ELF sha256 == host's embedded guest",
                      problem=f"no archived ELF has the host's guest sha256 {info['elf_sha256'][:16]}... ({'--elf ' + str(a.elf) if explicit else 'searched fixtures/results/GUEST-*/'}: "
                              f"{[c.name + '=' + sha256_file(c)[:12] for c in cands if c.is_file()][:6]}) -> archive the approved ELF bytes or pass --elf <path>")
        else:
            log.note("archived ELF not on this host; sha verified at setup (pod_setup.sh)")
    tm.stage("V4.elf_archived", t_elf)

    # V7 declared output domain: every served output token of every request is < 2^16 (16 slots per request; W_out = N x 16 x 16 bits)
    t_v7 = time.perf_counter()
    toks = []
    if pkind == "serve":
        for ln in (rec / "adapter" / "production" / "production.jsonl").read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                toks.append((d["request_id"], list(d["output_token_ids"])))
    # [WR-6] multiplicity and declared domain bits from the pinned instance (outputs.designated), never constants
    outs = inst_doc["outputs"]
    slots, domain_bits, W_out_inst = steps.declared_output_domain(inst)
    v7 = steps.output_domain_check(inst, dict(toks), n_requests=N) if pkind == "serve" else []
    if pkind != "serve":
        log.not_applicable("V7.output_domain", "served token ids (production.jsonl) are a Serve_v1 artefact; a registered program's outputs are committed words whose width the population's kind table declares",
                           replaced_by="V7.instance_domain (declared bits) + V5.positions (committed output words)")
    else:
      log.check("V7.output_domain", not v7,
              f"every request has the declared {slots} output slots and every output token id is in the declared {domain_bits}-bit domain (check_output_domains); W_out = N x {W_out_inst} bits",
              requests=len(toks), slots=slots, domain_bits=domain_bits, max_token=max((max(t) for _, t in toks), default=None), W_out=N * W_out_inst, problems=v7[:3])
    tm.stage("V7.output_domain", t_v7)

    # D6 addition (not one of L1's 76 checks; per request): typing + linkage of the request commitment -- the verifier rebuilds each request's
    # boundary tree from the served prompt / output tokens / sampling params / seed (the phase-0 boundary record) and compares its root with
    # the committed phase-0 root (V3 only compared the two commitment files' digests); and the served outputs in production.jsonl equal the
    # boundary record's.  "Replayed outputs equal served outputs" is UNAVAILABLE on this record: the replay was run along the served
    # trajectory (selective_prover.json phase2.trajectory_followed) and stores RU roots only -- there is no replayed token to compare.
    t_d6 = time.perf_counter()
    try:
        if pkind != "serve":
            raise steps.NotApplicable("the capture reader / served prompt+outputs are Serve_v1 artefacts; a registered program's boundary tree is authenticated leaf by leaf by V8.step2's input check and V5.positions")
        from verity_vllm.capture import CaptureReader, boundary_tree_id
        from verity_vllm.commit import TreeCache

        reader = CaptureReader(sel)
        cache = TreeCache(reader, recompute_leaves=True)
        served = {rid: t for rid, t in toks}
        per_req = []
        ok_all = True
        for rid in reader.request_ids():
            t_r = time.perf_counter()
            tid = boundary_tree_id(rid)
            tree = cache.tree(tid)
            b = reader.boundary(rid)
            root_ok = tree.root.hex() == c0["trees"][tid]["root"] and int(c0["trees"][tid]["leaf_count"]) == tree.leaf_count
            link_ok = list(b.output_token_ids) == served.get(rid) and len(b.output_token_ids) == slots
            ok_all = ok_all and root_ok and link_ok
            per_req.append({"request_id": rid, "root_ok": root_ok, "served_equals_boundary": link_ok, "leaves": tree.leaf_count, "seconds": time.perf_counter() - t_r})
        tm.misc["D6_request_commitment_per_request"] = per_req
        log.check("D6.request_commitment", ok_all and len(per_req) == N, "D6 addition: per request, the boundary tree rebuilt from the served prompt/outputs/params/seed has the committed phase-0 root and leaf count; served output tokens == boundary record (replayed == served: UNAVAILABLE on this record, see note)", requests=len(per_req), seconds=round(time.perf_counter() - t_d6, 3))
    except steps.NotApplicable as e:
        log.not_applicable("D6.request_commitment", str(e), replaced_by="V8.step2 + V5.positions")
    except Exception as e:  # noqa: BLE001 -- a check that cannot run is NOT DONE with its error, never silently dropped
        log.note(f"NOT DONE D6.request_commitment: {type(e).__name__}: {str(e)[:300]}")
        tm.misc["D6_request_commitment_error"] = f"{type(e).__name__}: {e}"
    tm.stage("D6.request_commitment", t_d6)

    # --- the verifier's record for certificate_v2 (A7): NO Bound is computed here.  The legacy Bound step (rec_p1_verify.py:348 passed
    # len(exempt) as `allowance`; D6 kept bound_certified over a kind table x N_REQ) is not carried into verifier_v3.  This record carries
    # the declared epoch, B_ex as an RU count with provenance, the law, what was drawn (J, F, Q), the output domain, the completion risk and
    # the checks; e2e.verifier_v3.certificate_inputs turns it into certificate_v2's Epoch/Budget/Law/OutputDomain/Sampled (A9 supplies the
    # population from A1's modules and calls certificate_v2.build_certificate).
    t_rec = time.perf_counter()
    law = inst_doc["sampler"]["law"]
    q, s, delta = Fraction(str(law["q"])), Fraction(str(law["s"])), Fraction(str(inst_doc["security"]["delta"]))
    log.check("V1.law", chron["law"]["q"] == law["q"] and chron["law"]["s"] == law["s"] and law["kind"] == "bernoulli", "chronology law == instance law (declared before any commitment)", q=law["q"], s=law["s"], delta=inst_doc["security"]["delta"])
    B_ex = int(inst.exemption_budget)  # the DECLARED RU-count budget (contract R8), never the realized 'exempt (charged)' VU count
    log.check("B.allowance", B_ex == int(inst_doc["security"]["B_ex"]) == int((inst_doc.get("epoch") or {}).get("B_ex", -1)) and inst_doc["security"].get("B_ex_unit") == "replay_units" and len(F) <= B_ex,
              "the exemption budget handed to the certificate is the epoch's declared B_ex (RU count, stated once, with provenance); |F| <= B_ex", B_ex=B_ex, F=F)
    if dmode is not None:
        # [E9, SCALE order 3] every opening taken in this run folded (leaf bytes -> path) to its PINNED root; a refused one already failed its check
        # by name (OPENING_UNAUTHENTICATED) -- this line makes the count visible and fails closed if any was refused
        dsum = dmode.summary()
        tm.misc["deployed_mode"] = dsum
        if pkind != "serve" and dsum["openings_authenticated"] == 0 and dsum["openings_refused"] == 0:
            log.not_applicable("V3.deployed_openings", "deployed mode covers the capture-reader trees (Serve_v1: TreeCache / InteriorTree openings); this registered program's record module "
                               "opened its own trees, none through them -- NOT authenticated by this mode (V8.step2 / V5.positions authenticate them leaf by leaf)")
        else:
            log.check("V3.deployed_openings", dsum["openings_refused"] == 0 and dsum["openings_authenticated"] > 0,
                      "deployed mode: every opening the verifier took (RU / weights / boundary / interior leaves) authenticated leaf bytes -> path -> PINNED root; none refused; at least one taken",
                      **dsum)
    status = "VERIFIED" if not log.fails else "REFUSED"
    status_line = status if status == "VERIFIED" else "REFUSED: " + "; ".join(log.fails)[:400]
    # the declared output domain from the instance (fixed before the first commitment): per request W_out = sum multiplicity x declared bits;
    # the vocabulary implied bound is what the instance's designated output cites (log2(V) source string), parsed when it names V
    tokens_per_request = slots
    _vsrc = str(outs["designated"][0].get("domain_bits_vocabulary_implied", {}).get("source", ""))
    vocabulary = int(_vsrc.split("V = ")[1].split()[0]) if "V = " in _vsrc else None
    W_out_sum = sum(int(d["multiplicity"]) * int(d["domain_bits_declared"]) for d in outs["designated"])
    log.check("V7.instance_domain", W_out_inst == W_out_sum and (a.out_bits_per_ru is None or a.out_bits_per_ru == W_out_inst),
              "the instance's declared W_out per request == sum over designated outputs of multiplicity x declared bits (and == --out-bits-per-ru when given as a cross-check)",
              W_out=W_out_inst, slots=slots, domain_bits=domain_bits, cli=a.out_bits_per_ru)
    a.out_bits_per_ru = W_out_inst
    # [user directive 20:10Z R13] generation-step count and output-token count are BOTH explicit in the instance and their relation is checked
    # from the circuit: Serve_v1 produces one output slot per decode step plus the prompt-position logits slot, i.e. multiplicity == STEPS + 1;
    # the built program is the witness (its output collection has that many slots), so 15-vs-16 is a checked fact, not a convention
    if pkind != "serve":
        rel = {"STEPS": None, "multiplicity": None, "program_slots": None, "problems": []}
        log.not_applicable("V7.steps_vs_outputs", "multiplicity == STEPS + 1 is Serve_v1's decode relation; a registered program pins its outputs in program.builder / outputs.designated and V1.program_rebuilt checks the built program against them",
                           replaced_by="V1.program_rebuilt + V7.instance_domain")
    else:
      rel = steps.steps_outputs_relation(inst_doc, program=program)
    if pkind == "serve":
      log.check("V7.steps_vs_outputs", not rel["problems"], "outputs.designated multiplicity == program.identity.STEPS + 1 (prompt-position logits + one slot per decode step), and == the verifier-built program's root output slots",
              STEPS=rel["STEPS"], multiplicity=rel["multiplicity"], program_slots=rel["program_slots"], problems=rel["problems"])
    record_doc = {"schema": "veritor/verifier-record/v3", "utc": utc(), "status": status, "status_line": status_line, "program_kind": pkind,
                  "not_applicable": list(log.na),  # [D20] the checks NOT run for this program kind, each with its reason (never silently skipped)
                  "population_derived": {k: pd.get(k) for k in steps.POPULATION_DERIVED_KEYS},
                  "record": {"path": str(rec), "instance_format": a.instance_format, "instance_digest": idig, "program_digest": pop_doc["program_digest"], "partition_digest": pinned_partition, "partition_digest_kind": inst_doc["partition"].get("partition_digest_kind"),
                             "population_sha256": pop_sha, "plan": inst_doc["partition"].get("plan_name"), "program_identity": (inst_doc.get("program") or {}).get("identity"),
                             # [A10 WR-4] the population is LABELLED, never silently the legacy one: A9's population_v5 carries construction
                             # "legacy-minus-inputs"; COMPACT-P1 (inputs as VUs) is labelled legacy so certificate_v2 subtracts / refuses
                             "population": {"construction": pop_doc.get("construction") or ("legacy (COMPACT-P1 rows: Input kinds are VUs)" if pop_doc["partition_digest"] == P1_DIGEST else "unlabelled"),
                                            "partition_digest_kind": pop_doc.get("partition_digest_kind"), "inputs_are_vus": bool(any(k.get("fn") == "Input" for k in _kind_entries(pop_doc))),
                                            "schema": pop_doc.get("schema")},
                             "batch": N, "auth_mode": "per-word", "guest": {"elf_sha256": info["elf_sha256"], "vk_hash": info["vk_hash"], "approval": (guest or {}).get("approval")}},
                  "epoch": {"identifier": inst.epoch.epoch_id if inst.epoch else None, "requests": list(inst.epoch.request_ids) if inst.epoch else [], "replay_units": N,
                            "ru_indices": list(inst.epoch.ru_indices) if inst.epoch else list(range(N)), "bound_scope": inst.epoch.bound_scope if inst.epoch else None,
                            "separately_analysed_audits": False, "delta": str(delta)},
                  "B_ex": {"value": B_ex, "unit": "RU", "provenance": (inst.epoch.B_ex_provenance if inst.epoch else None) or inst.security.B_ex_provenance or "protocol_instance.json security.B_ex (replay_units)"},
                  "law": {"kind": "bernoulli", "q": str(q), "s": str(s), "delta": str(delta), "provenance": "protocol_instance.json sampler.law + security.delta (== chronology law, V1.law)",
                          "ru_draw": "conditioned on >= 1 selected (demonstration)" if conditioned else "unconditioned", "redraws": rq.get("redraws", 0)},
                  "sampled": {"J": J, "F": F, "retained": retained, "Q": selected["count"], "candidates": selected.get("candidates"), "J_identities": sorted(sel_rids),
                              "exemption_policy": (exreq.get("exemption_policy") or {}).get("name"), "policy_exempted": (exreq.get("exemption_policy") or {}).get("policy_exempted"),
                              "policy_decided_at": (exreq.get("exemption_policy") or {}).get("decided_at"),
                              "F_identities": [ru_id_by_index.get(int(r), f"UNKNOWN_RU:{r}") for r in F_declared], "F_declared": F_declared, "kinds": selected["kinds"], "provable": len(want_coord), "input_units": len(want_input),
                              "uncovered_selected": len(uncovered), "uncovered_by_kind": dict(Counter(u["kind"] for u in uncovered)), "provenance": "chronology/selected.json + protocol.json (V2 re-derived)"},
                  "output_domain": {"W_declared_bits": N * a.out_bits_per_ru, "tokens": tokens_per_request * N, "vocabulary": vocabulary,
                                    "provenance": "protocol_instance.json outputs (declared before the first commitment) x N requests; V7.output_domain checked every served token < 2^16"},
                  "completion_risk": rf.get("completion_risk"), "sampled_unsupported": rf.get("sampled_unsupported"), "ru_fields": str(rf_path),
                  "coverage": {"ok": bool(report is not None and report.ok), "expected": (report.expected_count if report else None), "covered": (len(report.covered) if report else None),
                               "uncovered": (len(report.uncovered) if report else None), "duplicate": (len(report.duplicate) if report else None), "unexpected": (len(report.unexpected) if report else None),
                               "proofs_voided_by_positions": (len(report.position_problems) if report else None), "derived_only_inputs_checked_by_V6": len(derived_only_vus),
                               "family": a.family, "schema": "veritor/coverage_v1/statement digests; e2e.coverage_v1.CoverageReport.to_json in coverage.json", "file": str(vdir / "coverage.json"),
                               # [E9, brief interface 6] identity accounting: the three numbers side by side (never equated)
                               "counts": counts, "per_vu": "coverage.json#per_vu (every sampled VU of a retained RU: covered | input | uncovered | duplicate)",
                               # [D40 rule (2)] the one line the certificate and the reports quote: proof-checked and directly checked, separately, by identity
                               "coverage_line": counts.get("coverage_line"), "completion_claim": counts.get("completion_claim")},
                  "interior_layout": {"status": il["status"], "ok": il["ok"], "interior_selected": il_sel["count"], "classification": il_sel["classification"], "problems": il["problems"]},
                  "output_copies": {"rows": oc["rows"], "ok": oc["ok"]},
                  "proofs": {"chunks": len(chunk_dirs), "proven_verified": len(proven), "total_cycles": proofs.get("total_cycles"), "prove_wall_seconds": proofs.get("prove_wall_seconds")},
                  "costs": {"verifier": {"seconds": round(time.perf_counter() - T0, 1), "provenance": f"verifier_v3/reverify.py on {os.uname().nodename} (CPU, never ran the prover)", "label": "MEASURED"}},
                  "checks": {"total": log.n, "failed": log.fails, "log": str(vdir / "reverify.log")},
                  "validation_scope": inst_doc.get("discrepancies") or (inst_doc.get("partition") or {}).get("validation"),
                  "provenance": {"label": "MEASURED (checks); no bound computed here -- certificate_v2 builds the certificate from this record", "script": "e2e/verifier_v3/reverify.py"}}
    tm.stage("record", t_rec)
    (vdir / "record_v3.json").write_text(json.dumps(record_doc, indent=1, sort_keys=True, default=str) + "\n")
    log.note(f"RECORD: {status_line}; epoch {record_doc['epoch']['identifier']} B_ex={B_ex} RU, J={J}, F={F}, |Q|={selected['count']}; certificate: e2e.certificate_v2 from record_v3.json")
    tm.stages["total_wall"] = time.perf_counter() - T0
    # --- comparison with L1's verifier.log (check by check) ---
    cmp_path = Path(a.compare) if a.compare else rec / "verify" / "verifier.log"
    compare: dict[str, Any] = {"l1_log": str(cmp_path), "present": cmp_path.exists()}
    if cmp_path.exists():
        l1 = {}
        for ln in cmp_path.read_text().splitlines():
            parts = ln.split(" ", 3)
            if len(parts) >= 3 and parts[2] in ("PASS", "FAIL"):
                l1[parts[1]] = parts[2]
        mine = {}
        for ln in log.lines:
            parts = ln.split(" ", 3)
            if len(parts) >= 3 and parts[2] in ("PASS", "FAIL"):
                mine[parts[1]] = parts[2]
        same = sorted(k for k in l1 if k in mine and l1[k] == mine[k])
        differ = sorted(k for k in l1 if k in mine and l1[k] != mine[k])
        missing = sorted(k for k in l1 if k not in mine)
        added = sorted(k for k in mine if k not in l1)
        compare.update({"l1_checks": len(l1), "l1_failed": sum(v == "FAIL" for v in l1.values()), "v3_checks": len(mine), "v3_failed": sum(v == "FAIL" for v in mine.values()),
                        "same_outcome": len(same), "different_outcome": differ, "l1_checks_missing_here": missing, "v3_additions": added})
        log.note(f"COMPARE with L1 {cmp_path}: L1 {len(l1)} checks ({compare['l1_failed']} failed); here {len(mine)} ({compare['v3_failed']} failed); same outcome {len(same)}; different {differ}; missing {missing} (B.certified / V5.coordinates-by-provenance are not run here by design); v3 additions {added}")
    (vdir / "compare.json").write_text(json.dumps(compare, indent=1, sort_keys=True) + "\n")
    (vdir / "timings.json").write_text(json.dumps({"schema": "veritor/a3-reverify-timings/v1", "utc": utc(), "host": os.uname().nodename, "record": str(rec), "stages_s": tm.stages,
                                                   "misc": tm.misc, "proofs": tm.proofs, "checks": log.n, "failed": log.fails, "status_line": status_line}, indent=1, sort_keys=True, default=str) + "\n")
    log.note("peak_rss_mb_after: " + json.dumps(tm.misc.get("peak_rss_mb_after", {})))
    # [E9, SCALE] the data accounting: ARCHIVE (whole-record) vs DEPLOYED (sampled units against pinned roots) vs CHECKPOINT vs UNCLASSIFIED
    acc_doc = _write_accounting(vdir, rec, sampled_units=counts.get("sampled_vus_retained"), proofs=tm.proofs)
    if acc_doc is not None and acc_doc.get("buckets"):
        log.note("ACCOUNTING (verify_accounting.json): record_bytes=%d %s" % (acc_doc.get("record_bytes") or 0, " ".join(
            f"{k}[read={v['bytes_read']} hashed={v['bytes_hashed']} leaves={v['leaves_hashed']} openings={v['openings']} wall={v['wall_s']}s ratio_read={v['ratio_bytes_read_to_record']}]"
            for k, v in acc_doc["buckets"].items())))
        log.write()
    print(f"[reverify_v3] {log.n} checks, {len(log.fails)} failed; {status_line}")
    print("[reverify_v3] RECORD for certificate_v2:", vdir / "record_v3.json")
    print("[reverify_v3] stages_s:", json.dumps({k: round(v, 3) for k, v in tm.stages.items()}))
    print("[reverify_v3] peak_rss_mb_after:", json.dumps(tm.misc.get("peak_rss_mb_after", {})))
    return 0 if status == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
