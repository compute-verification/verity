"""A served window's cases, each on the window's registration record R and beside its own Lean answer, in the directory
form a composition of the window's subprotocols reads as a case: sampled proofs' one-stage v1 case (no composite's),
both composites' sampled proofs case (one-stage v2's in the form ``one-stage``) and the inference-only case from its
records audit, PoUW's period case held to the same record, Pearl+'s run case naming it, and with ``--v2`` one-stage v2's
bare case too:

    served_cases.py (--audit DIR | --serve-dir D) --private DIR [--alpha N/D --gamma G | --from-division]
                    [--log FILE] [--period DIR] [--pearlplus DIR] [--v1-exe PATH] [--sampled-exe PATH]
                    [--inference-only-exe PATH] [--v2 [--v2-exe PATH]]

``--audit`` is the window's `records_audit.py audit --circuit --registered` out directory (`served_verdicts.sh`'s
``$VW/audit``); ``--serve-dir`` finds it as the one ``D/v3c-w*/audit`` that holds a case, and reads the serve's H1 log
(``D/h1/log.jsonl``, `H1_LOG`), PoUW's period case (``D/private/pouw/period``, `PERIOD_PIECE`) and Pearl+'s case's
parts (``D/private/pearlplus/case``, `PEARL_PLUS_PIECE`) where it holds them.
Every case commits to the same record R, ``window/case.json``'s commitment, the record the audit sampled against (under
H1's tie, `served_verdicts.sh`'s H1_TIE, the record H1 named as the window's Commit record before its sample coin,
`serve_tie`), and each must name R's digest, SHA-512(format ‖ 0 ‖ canon(record)) (`registration.digest`), or nothing is
written: that digest is what remote's record tie compares (`Core.Remote.windows`).

* **`verity/sampled-proofs/one-stage/v1`** (`window_case`, `V1`): a5's ``window/case.json`` as a5 wrote it
  (`a5.write_case`: the registration record, the evidence the audit's record was handed, the sessions' verdicts among
  it, and the verifier's params), exactly those three, and none when a5 wrote its ``case-error.txt`` instead (which says
  why). No composite takes it: both name sampled proofs (`VStar.Recursive.Sampled`) among their subprotocols
  (`Core.Onsite.components`, `Core.Remote.components`), which refuses it by name (`VStar.Recursive.Sampled.V1_REFUSED`).
  It is written because the other cases are read off it (v2's, the inference-only case's), and its answer,
  `sampled-proofs-one-stage check` on the case, goes to the verdicts alone.
* **`verity/sampled-proofs/v1`** (`sampled_case`, `SAMPLED`): both composites' sampled proofs subprotocol
  (`Core.Remote.components`, `Core.Onsite.components`, `VStar.Recursive.Sampled`), v2's case (`v2_case`, below) with
  ``form`` ``one-stage`` in its params and its evidence (`with_form`), which the dispatcher splits off before one-stage
  v2's decoders read the rest:
  its params name ``proofs`` and its evidence the proofs' files in place of v1's verdicts, as the dispatcher requires.
  None when v2's case is refused, which the refusal says, and each composite then names sampled proofs missing. a5
  writes v2's case only for a `--zk` audit that drew exactly one member (`a5.v2_proof`), and v2 refuses a windowed
  registration (`a5.case_v2`), so a served window of several members has none: under H1's tie its audit's law is
  `stratified:1`, a sampled reader of each table W registers. Its answer is `vstar-recursive sampled check` on the case
  and its blobs.
* **The inference-only case** (`inference_only_case`, `INFERENCE_ONLY`: `verity/inference-only/v0`, the inference-only
  policy's check as a core protocol, `inference-only`):
  - params: ``alpha`` and ``gamma`` (the policy's α and Γ: given, ``--alpha`` and ``--gamma``, or with
    ``--from-division`` read off the case's own division, `division_policy`, so the check confirms that the division
    complies and they're no security setting, which the case's private ``policy.json`` says beside them; without
    either no case is written), ``log`` (the deployment's records before R, from its first, whose ``previous`` is
    null, as the policy's check reads them: ``--log``, else with ``--serve-dir`` the serve's H1 log through R's
    ``previous``, `h1_log`, each window's registration W and record R up to the window's W; empty when R names no
    previous record), ``program`` (v1's params' Program as canonical text, the one the record's ``program`` names) and
    ``parameters`` (its program function's, which the descriptor doesn't carry: the population program rebuilt over its
    root's templates, and the registered values v1's own ``parameters`` name ahead of them, held to the record's
    ``program``, `parameters`);
  - commitment: the registration record R;
  - evidence: ``owner`` per root node of the Program, ``theta`` and ``provenance``. The window's Program is its
    population program (`interface.Circuit.program`), one batch node per template holding that template's instances for
    every slot, so no division into one block a slot is a division of its root nodes: every non-input node is block 0,
    the window's m blocks of c gates as one of m·c, which complies only when Γ ≥ m·c. So where a division into one
    block a slot (k the most fresh-input bits in one) would give Γ = c and α = k/c, this one's gives Γ = m·c and α at
    most k/c. The Program registers no root input (its inputs are `Input<w>`), so ``theta`` and ``provenance`` are
    empty and every input bit counts as fresh against α.

  Both composites name the policy among their subprotocols (`Core.Remote.components`, `Core.Onsite.components`), and
  compose its v0, which publishes the division and its blocks' profiles, as a stated exception (`composite.EXCEPTIONS`).
  The policy's Lean package and the Python its check is held to are named by the module `POLICY`, loaded when this
  case is asked for; on a tree that leaves the policy out (the public release) the case is refused saying so.
* **PoUW's period case** (`period_case`, `PERIOD`: `verity/pouw/period/v1`, both composites' work subprotocol):
  `pouw_piece.py period`'s case and answer as it wrote them (``--period``; with ``--serve-dir``, `served_pouw.sh`'s
  ``D/private/pouw/period`` when it holds a case), taken once every window of the period names R as its Commit record
  (`credit.commit_of`, as `Core.Remote.windowRecords` reads it): the case commits to the auditor's ledger, so the tie
  is its windows'. A period with a window on another record, or on none, is refused here, before any composite runs,
  and so is handed to neither: remote would reject it on the tie, and onsite, which doesn't check it, would credit
  another record's work beside this window's. Without a period case none is made, and a composite names it missing.
* **Pearl+'s run case** (`pearlplus_case`, `PEARL_PLUS`: `verity/pouw/pearl-plus/v1`, the work subprotocol of remote's
  composite of record, `Core.Remote.V1`): from the parts Pearl+'s piece wrote on the served node (``--pearlplus``; with
  ``--serve-dir``, `PEARL_PLUS_PIECE` when it holds them: `served_passes.py case`), its params (the epoch's coin
  commitment, the draws and the verifier's coin record), its commitment ``{"manifests", "record"}``, the retained passes'
  manifests and R's digest, which remote ties to the policy's record (`Core.Remote.V1.run`), and its evidence, each
  pass's files the opened tiles read by the SHA-512 of the blob holding them, the blobs the parts' own. Its answer is the
  Lean verdict of record's (`pouw-pearlplus` on the same passes, coins, draws and coin record, which the check of this
  case runs on the blobs instead of the files) under the commitment's digest, SHA-512(`PEARL_PLUS_TAG` ‖ 0 ‖
  commitment), since Pearl+'s package builds no executable of the protocol's form alone. The manifests carry the
  passes' salt key, so its commitment, unlike the others', is witness. Without the parts none is made.
* **`verity/sampled-proofs/one-stage/v2`** (`v2_case`, only with ``--v2``): a5's ``window/case-v2.json``
  (`a5.case_v2`: v1's case with the verdicts replaced by the one sampled member's proof's files, its params naming that
  member's circuit, public file and registered file, and ``blobs``), refused here when it commits to another record
  than v1's case or names a blob that isn't there, and when a5 wrote none (its ``case-v2-error.txt`` says why: v2
  verifies one table's `--zk` session on its circuit and public file). Its answer is `sampled-proofs-one-stage-zk
  check` on the case and its blobs. Neither composite names v2's bare case (remote reads it in its form, above), so it
  is made only when asked.

Each case goes to its own directory under ``--private`` (``sampled-proofs-v1/``, ``sampled-proofs/``,
``inference-only/``, ``pouw-period/``, ``pouw-pearl-plus/``, ``sampled-proofs-v2/``; 0700, files 0600): ``case.json``
(``{"params", "commitment", "evidence"}``, hex commitment and evidence, and sampled proofs', v2's, the period's and
Pearl+'s ``blobs``) and ``answer.json``, its Lean answer line (and, given ``--from-division``, the inference-only case's
``policy.json``). A composition is handed only the directories whose answer's protocol is one of its product's
subprotocols: remote's are sampled proofs', the inference-only case's and the period's (v1's, Pearl+'s in the period's
place), onsite's the same three among its own, and neither's one-stage v1's case or v2's bare case. The Lean runs on
files in a 0700 scratch directory under ``--private``; the period's answer is the one `pouw_piece.py` wrote, and
Pearl+'s the Lean verdict of record's.
This prints a ``CASE {"protocol", "verdict", "dir"}`` line per case written and
a ``NOT_WRITTEN {"protocol", "reason"}`` line per case refused, and nothing of a case's params or evidence: not α and Γ
either, which ``--from-division`` reads off the division, so on a one-block window they'd publish its block's gates
and fresh-input bits. The lines go to a private log (a reason may quote a5's ``case-error.txt`` or
``case-v2-error.txt``), of which only a line's ``protocol`` and ``verdict`` are public. Exit 0 when every case is
written, 1 when one is refused, 64 on a usage error.
"""
from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

from verity.protocols.pouw.verifier import credit as Cr
from verity.protocols.pouw.verifier import period_check as PC
from verity.protocols.spot.verifier.one_stage import proofs as PF
from verity.protocols.spot.verifier.one_stage import registration as R
from verity.protocols.spot.verifier.one_stage import verifier as VF
from verity.circuits import codec

ROOT = Path(__file__).resolve().parents[2]
V1 = R.PROTOCOL
V1_LEAN = (VF.PACKAGE, "sampled-proofs-one-stage")
SAMPLED = "verity/sampled-proofs/v1"
SAMPLED_LEAN = (ROOT / "verity/protocols/recursion/verifier/recursive", "vstar-recursive")
#: `vstar-recursive`'s command for sampled proofs (`VStar.Recursive.Sampled.main`), ahead of ``check``
SAMPLED_CMD = ("sampled",)
#: The form remote's sampled proofs case names: one-stage v2's audit (`VStar.Recursive.Sampled.ONE_STAGE`)
ONE_STAGE = "one-stage"
V2 = PF.V2
INFERENCE_ONLY = "verity/inference-only/v0"
#: the module beside this file that says where the tree holds the policy's check: its Lean package and executable
#: (``LEAN``) and the Python that check is held to (``reference``). The public release leaves the policy out, and that
#: module with it, so `policy` loads it by name
POLICY = "inference_only_lean"
PERIOD = PC.ID
PEARL_PLUS = "verity/pouw/pearl-plus/v1"
#: what Pearl+'s commitment's digest hashes ahead of a zero byte and the commitment (`PouwCheck.PearlPlus.Protocol`)
PEARL_PLUS_TAG = b"verity/pouw/pearl-plus/manifests/v1"
#: the keys of `served_passes.py case`'s parts that make the case and its answer
PEARL_PLUS_PARTS = {"params", "manifests", "evidence", "lean", "delta"}
DIRS = {V1: "sampled-proofs-v1", SAMPLED: "sampled-proofs", INFERENCE_ONLY: "inference-only", PERIOD: "pouw-period",
        PEARL_PLUS: "pouw-pearl-plus", V2: "sampled-proofs-v2"}
#: Under ``--serve-dir``: the serve's H1 log, a line a record H1's tie wrote (`serve_tie.LOG`: ``{"window", "kind",
#: "file", "digest"}``, its file under the log's directory), the period case `served_pouw.sh` has `pouw_piece.py` write,
#: and the parts of Pearl+'s case `served_pearlplus.sh` has `served_passes.py case` write.
H1_LOG = Path("h1/log.jsonl")
PERIOD_PIECE = Path("private/pouw/period")
PEARL_PLUS_PIECE = Path("private/pearlplus/case")
CASE_KEYS = {"commitment", "evidence", "params"}
USAGE = 64
#: α and Γ past any block's (its gates and its fresh-input bits are each under 2^64), at which the policy's check reports
#: the case's own division.
UNBOUNDED = (2**64, 2**64)
#: Where ``--from-division``'s α and Γ come from, in the inference-only case's private ``policy.json``.
FROM_DIVISION = "read off the honest division, so the check confirms it complies; not a security setting"


class Refused(Exception):
    """The audit's outputs don't make the case; the message says why."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise Refused(reason)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: {type(e).__name__}: {e}") from None


def record_of(case: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The registration record ``case`` commits to, held to be canonical JSON, and its digest (`registration.digest`)."""
    try:
        b = bytes.fromhex(case["commitment"])
        rec = json.loads(b)
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"the case's commitment is not a registration record: {type(e).__name__}: {e}") from None
    _need(isinstance(rec, dict) and codec.canonical_json(rec) == b,
          "the case's commitment is not a registration record as canonical JSON")
    return rec, R.digest(rec)


def window_case(audit: Path) -> dict[str, Any]:
    """v1's case of the audit (module docstring), whose commitment is the window's registration record."""
    w = audit / "window"
    if not (w / "case.json").is_file() and (w / "case-error.txt").is_file():
        raise Refused(f"no v1 case: {(w / 'case-error.txt').read_text().strip()}")
    case = _json(w / "case.json")
    _need(isinstance(case, dict) and set(case) == CASE_KEYS,
          f"{audit}/window/case.json is not a case: it holds exactly {sorted(CASE_KEYS)}")
    return case


def v2_case(audit: Path) -> dict[str, Any]:
    """Sampled proofs' v2 case (module docstring): a5's ``case-v2.json`` held to v1's record and to its blobs."""
    w = audit / "window"
    v1 = window_case(audit)
    if not (w / "case-v2.json").is_file():
        why = (w / "case-v2-error.txt").read_text().strip() if (w / "case-v2-error.txt").is_file() else \
            "the audit's a5 writes no case-v2.json (a5.case_v2)"
        raise Refused(f"no v2 case: {why}")
    case = _json(w / "case-v2.json")
    _need(isinstance(case, dict) and set(case) == CASE_KEYS | {"blobs"},
          f"v2's case holds exactly {sorted(CASE_KEYS | {'blobs'})}")
    _need(case["commitment"] == v1["commitment"], "v2's case commits to another record than the audit's")
    record_of(case)
    _need(isinstance(case["blobs"], list) and all(isinstance(b, str) for b in case["blobs"]), "v2's blobs are paths")
    gone = [b for b in case["blobs"] if not Path(b).is_file()]
    _need(not gone, f"v2's case names {len(gone)} blob(s) that aren't files, the first {gone[:1]}")
    return case


def with_form(case: Mapping[str, Any], form: str = ONE_STAGE) -> dict[str, Any]:
    """``case`` as sampled proofs reads it (`VStar.Recursive.Sampled`): its params and its evidence each with ``form``,
    which the dispatcher splits off before that form's decoders read the rest."""
    try:
        evidence = json.loads(bytes.fromhex(case["evidence"]))
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"the case's evidence is not JSON: {type(e).__name__}: {e}") from None
    _need(isinstance(case.get("params"), Mapping) and isinstance(evidence, dict)
          and "form" not in case["params"] and "form" not in evidence,
          "the case's params and evidence must be objects that name no form")
    return {**case, "evidence": codec.canonical_json(evidence | {"form": form}).hex(),
            "params": {**case["params"], "form": form}}


def sampled_case(audit: Path) -> dict[str, Any]:
    """Remote's sampled proofs case (module docstring): v2's (`v2_case`) in the form ``one-stage``."""
    return with_form(v2_case(audit))


def period_case(d: Path, digest: str) -> tuple[dict[str, Any], str, Any]:
    """PoUW's period case in ``d`` as `pouw_piece.py period` wrote it, its answer line and its verdict, once every window
    of the period names the record of digest ``digest`` as its Commit record (`credit.commit_of`, as
    `Core.Remote.windowRecords` reads each window's ``commit``): the record tie remote checks (`Core.Remote.windows`)."""
    case = _json(d / "case.json")
    _need(isinstance(case, dict) and CASE_KEYS <= set(case) <= CASE_KEYS | {"blobs"},
          f"{d}/case.json is not a case: it holds {sorted(CASE_KEYS)} and its blobs")
    try:
        line = (d / "answer.json").read_text().strip()
        answer = json.loads(line)
    except (OSError, ValueError) as e:
        raise Refused(f"the period's case has no answer line: {type(e).__name__}: {e}") from None
    _need(isinstance(answer, dict) and answer.get("protocol") == PERIOD,
          f"the period's answer is {answer.get('protocol') if isinstance(answer, dict) else None!r}'s, not {PERIOD}'s")
    try:
        windows = json.loads(bytes.fromhex(case["evidence"]))["windows"]
    except (TypeError, ValueError, KeyError) as e:
        raise Refused(f"the period's evidence names no windows: {type(e).__name__}: {e}") from None
    _need(isinstance(windows, list) and windows != [], "the period's evidence names no windows")
    for i, w in enumerate(windows):
        got, why = Cr.commit_of(w.get("commit") if isinstance(w, Mapping) else None)
        _need(why is None, f"window {i} of the period names no Commit record, so not the window's registration record "
              f"{digest[:16]}…: {why[1] if why else ''}")
        _need(got[0].hex() == digest, f"window {i} of the period names the Commit record {got[0].hex()[:16]}…, not the "
              f"window's registration record {digest[:16]}…: remote's record tie rejects it")
    return case, line, answer.get("verdict")


def pearlplus_case(d: Path, digest: str) -> tuple[dict[str, Any], str, Any]:
    """Pearl+'s run case (module docstring) from the parts `served_passes.py case` wrote in ``d``, its commitment naming
    the record of digest ``digest``, its answer line and its verdict."""
    parts = _json(d / "parts.json")
    _need(isinstance(parts, dict) and PEARL_PLUS_PARTS <= set(parts),
          f"{d}/parts.json is not Pearl+'s case's parts: it holds {sorted(PEARL_PLUS_PARTS)}")
    files, lean = (parts["evidence"] or {}).get("files"), parts["lean"] or {}
    _need(isinstance(files, list) and all(isinstance(f, dict) for f in files), "the parts' evidence is not each pass's files")
    _need(lean.get("verdict") in ("accept", "reject"), f"the parts' Lean verdict is {lean.get('verdict')!r}")
    blobs = [d / "blobs" / h for h in sorted({h for f in files for h in f.values()})]
    gone = [b for b in blobs if not b.is_file()]
    _need(not gone, f"the parts name {len(gone)} blob(s) that aren't files in {d / 'blobs'}")
    commitment = codec.canonical_json({"manifests": parts["manifests"], "record": digest})
    accepted = lean["verdict"] == "accept"
    answer = {"protocol": PEARL_PLUS, "verdict": lean["verdict"], "reason": lean.get("reason"),
              "commitment": hashlib.sha512(PEARL_PLUS_TAG + b"\0" + commitment).hexdigest(),
              "public": {**(lean.get("work") or {}), "delta": parts["delta"]} if accepted else None,
              "profile": {"draws": parts["params"].get("draws"), "drawn": lean.get("drawn"),
                          "passes": len(parts["manifests"])} if accepted else None}
    case = {"params": parts["params"], "commitment": commitment.hex(),
            "evidence": codec.canonical_json(parts["evidence"]).hex(), "blobs": [str(b) for b in blobs]}
    return case, json.dumps(answer, sort_keys=True), answer["verdict"]


def policy() -> ModuleType:
    """`POLICY`, loaded by name when the inference-only case is asked for, so nothing needs it at import time. Where the
    tree leaves it out, the case is refused by name (`Refused`), never skipped; a module missing from inside it is a
    broken environment and raises as itself."""
    try:
        return importlib.import_module(POLICY)
    except ModuleNotFoundError as e:
        if e.name != POLICY:
            raise
        raise Refused(f"the inference-only policy's check is not on this tree: {POLICY}.py, which names its Lean package "
                      "and the Python that check is held to, is left out with the policy") from e


def __getattr__(name: str) -> Any:
    """``INFERENCE_ONLY_LEAN``, the policy's Lean package and executable for a caller that builds it: `policy`'s."""
    if name == "INFERENCE_ONLY_LEAN":
        return policy().LEAN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _reference():
    """`nci_reference`, the Python the policy's Lean check is held to, for the Program's root nodes."""
    return policy().reference()


def _ratio(x: Fraction | int | str) -> list[int]:
    f = Fraction(x)
    _need(f >= 0, f"α is a nonnegative ratio, not {x}")
    return [f.numerator, f.denominator]


def _chain(log: Sequence[Mapping[str, Any]], rec: Mapping[str, Any]) -> None:
    """``log`` then ``rec``, each record's ``previous`` the digest of the one before (null for the first)."""
    prev = None
    for r in [*log, rec]:
        _need(isinstance(r, Mapping) and r.get("previous") == prev,
              "the log and the record don't chain: each record's previous is the digest of the one before"
              + ("; the record names a previous record, so pass the deployment's log (--log)" if not log else ""))
        prev = R.digest(r)


def h1_log(path: Path, rec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The deployment's records before ``rec``, oldest first, as the serve's H1 log ``path`` names them (`H1_LOG`), through
    the one ``rec`` names as its ``previous``: under H1's tie each window's registration W and record R, up to the
    window's own W, which its R is chained right after. Empty when ``rec`` names no previous record."""
    if rec.get("previous") is None:
        return []
    try:
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError) as e:
        raise Refused(f"the record names a previous record, and the serve's H1 log isn't readable: {type(e).__name__}: "
                      f"{e}") from None
    log = []
    for x in lines:
        _need(isinstance(x, dict) and isinstance(x.get("file"), str) and isinstance(x.get("digest"), str),
              f"a line of the serve's H1 log names no record file and digest: {str(x)[:200]}")
        r = _json(path.parent / x["file"])
        _need(isinstance(r, dict) and R.digest(r) == x["digest"],
              f"the serve's H1 log names {x['file']} as the record of digest {x['digest'][:16]}…, which it isn't")
        log.append(r)
        if x["digest"] == rec["previous"]:
            return log
    raise Refused("the record's previous is no record the serve's H1 log names, so its log doesn't chain from the "
                  "deployment's first record")


def _values(given: Any, members: Sequence[tuple[Any, int]]) -> tuple[tuple[str, int, int], ...]:
    """The registered values a population takes ahead of its templates' parameters (`mixed_population_program`'s
    ``values``), as ``given``, a program function's ``[[name, type], ...]``, names them: each parameter none of the
    templates' (``<j>.<name>``), an array of values. Empty when ``given`` is None."""
    if given is None:
        return ()
    own = {f"{j}.{name}" for j, (fn, _) in enumerate(members) for name, _ in fn.signature()[0]}
    _need(isinstance(given, list) and all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], str) for p in given),
          "v1's params' parameters are not [[name, type], ...]")
    values = []
    for name, t in given:
        if name in own:
            continue
        _need(isinstance(t, list) and len(t) == 3 and t[0] == "Array" and isinstance(t[2], list) and len(t[2]) == 2
              and t[2][0] == "Value", f"v1's params' parameter {name} is no template's and no registered value: {t}")
        values.append((name, t[1], t[2][1]))
    return tuple(values)


def parameters(desc: Mapping[str, Any], root: Any, given: Any = None) -> list[list[Any]]:
    """The program function's ``[[name, type], ...]``, which the descriptor doesn't carry (it inlines the function into
    its root): the population program (`population.mixed_population_program`, or `population_program` for one template)
    rebuilt over the root's template batches in order, held to the descriptor's digest. ``given``, v1's own copy where
    its params carry one, names the registered values the population takes ahead of its templates' parameters
    (`_values`), and must be the rebuilt program's."""
    from verity.circuits.population import POPULATION_VERSIONS, mixed_population_program, population_program
    from verity.circuits.program import is_input

    members = [(n.fn, getattr(n, "n", 1)) for n in root.body.nodes if not is_input(n.fn)]
    values = _values(given, members)
    builds = [lambda v: mixed_population_program(members, version=v, values=values)]
    if len(members) == 1 and not values:
        builds.append(lambda v: population_program(*members[0], version=v))
    want = codec.program_digest(dict(desc))
    for build in builds:
        for v in POPULATION_VERSIONS:
            if values and v == 0:
                continue
            try:
                p = build(v)
            except (ValueError, TypeError) as e:
                raise Refused(f"v1's Program isn't a population program: {type(e).__name__}: {e}") from None
            if codec.program_digest(p) == want:
                got = [[n, t.to_json()] for n, t in p.fn.signature()[0]]
                _need(given is None or given == got, "v1's params' parameters aren't its Program's program function's")
                return got
    raise Refused("v1's Program isn't the population program over its root's templates"
                  + (" and the values its params' parameters name" if values else "")
                  + ", so its program function's parameters aren't known")


def inference_only_case(v1: Mapping[str, Any], alpha: Fraction | int | str, gamma: int,
                      log: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The inference-only case (module docstring) from v1's case of the window's audit."""
    rec, _ = record_of(v1)
    desc = v1["params"].get("program") if isinstance(v1.get("params"), Mapping) else None
    _need(isinstance(desc, dict), "v1's params hold no Program descriptor")
    _need(codec.program_digest(desc) == rec.get("program"), "the record names another Program than v1's params")
    _need(isinstance(gamma, int) and not isinstance(gamma, bool) and gamma >= 0, f"Γ is a natural, not {gamma!r}")
    _chain(log, rec)
    text = codec.canonical_json(desc).decode()
    NR = _reference()
    try:
        inputs, _registered, _spans = NR.root_nodes(text)
        root = codec.decode_program(desc, NR.declared(desc)).root
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Refused(f"v1's Program doesn't decode: {type(e).__name__}: {e}") from None
    params = {"alpha": _ratio(alpha), "gamma": gamma, "log": [dict(r) for r in log],
              "parameters": parameters(desc, root, v1["params"].get("parameters")), "program": text}
    evidence = {"owner": ["input" if i else 0 for i in inputs], "provenance": [], "theta": []}
    return {"commitment": v1["commitment"], "evidence": codec.canonical_json(evidence).hex(), "params": params}


def division_policy(v1: Mapping[str, Any], log: Sequence[Mapping[str, Any]], exe: str | os.PathLike,
                    scratch: Path) -> tuple[Fraction, int]:
    """α and Γ read off the case's own division (``--from-division``): the policy's Lean check at no budget
    (`UNBOUNDED`) profiles each block's gates and fresh-input bits; Γ is the largest block's gates and α the most bits a
    gate of any block, so the check at them confirms that the division complies."""
    answer, _ = lean_answer(exe, inference_only_case(v1, *UNBOUNDED, log), scratch)
    profile = answer.get("profile")
    blocks = profile.get("blocks") if isinstance(profile, Mapping) else None
    _need(answer.get("verdict") == "accept",
          f"the case's own division doesn't pass the check at no budget: {answer.get('reason')}")
    _need(isinstance(blocks, list) and blocks != [] and all(
        isinstance(b, list) and len(b) == 2 and all(type(x) is int for x in b) and b[0] > 0 for b in blocks),
        "the check's profile isn't a nonempty list of [gates, bits] blocks with gates")
    return max(Fraction(bits, gates) for gates, bits in blocks), max(gates for gates, _ in blocks)


def lean_answer(exe: str | os.PathLike, case: Mapping[str, Any], scratch: Path,
                cmd: Sequence[str] = ()) -> tuple[dict[str, Any], str]:
    """``exe check`` (`CoreProtocol.Cli`; with ``cmd``, the command ahead of ``check``, as ``vstar-recursive sampled``)
    on ``case`` and its blobs, its files in a 0700 directory under ``scratch``: the answer and its line."""
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="case-", dir=scratch) as d:
        os.chmod(d, 0o700)
        files = [Path(d, n) for n in ("params.json", "commitment.bin", "evidence.json")]
        for f, b in zip(files, (codec.canonical_json(case["params"]), bytes.fromhex(case["commitment"]),
                                bytes.fromhex(case["evidence"]))):
            f.write_bytes(b)
            os.chmod(f, 0o600)
        p = subprocess.run([str(exe), *cmd, "check", *map(str, files), *case.get("blobs", ())], capture_output=True,
                           text=True)
    lines = p.stdout.strip().splitlines()
    try:
        answer = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise Refused(f"{exe} answered no line (exit {p.returncode})") from None
    _need(isinstance(answer, dict), f"{exe}'s answer is not an object")
    return answer, lines[-1]


def tie(answer: Mapping[str, Any], id: str, digest: str) -> None:
    """The answer is ``id``'s and names the record's digest as its commitment; a refusal says the answer's reason,
    which for an answer naming none says why its Lean read no commitment."""
    _need(answer.get("protocol") == id, f"the answer is {answer.get('protocol')!r}'s, not {id}'s")
    why = f": {answer['reason']}" if answer.get("reason") else ""
    _need(answer.get("commitment") == digest, f"{id}'s answer commits to {answer.get('commitment')!r}, not the "
          f"window's registration record's digest{why}")


def write(d: Path, case: Mapping[str, Any], line: str, also: Mapping[str, Any] | None = None) -> None:
    """``case.json`` and ``answer.json`` in ``d``, and each of ``also`` as a JSON file, 0700 with files 0600."""
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    files = {"case.json": json.dumps(dict(case), sort_keys=True), "answer.json": line}
    for name, body in (files | {n: json.dumps(v, sort_keys=True) for n, v in (also or {}).items()}).items():
        fd = os.open(d / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(body + "\n")
        os.chmod(d / name, 0o600)


def build(package: Path, name: str) -> Path:
    """``lake build name`` in ``package`` under `verifier.BUILD_LOCK`: each of these packages has Lake build the flock
    package in place, and two builds at once would rewrite one module's C under each other. The executable's path."""
    VF.BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(VF.BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", name], cwd=package, capture_output=True, text=True)
    _need(p.returncode == 0, f"lake build {name} in {package} failed ({p.returncode})")
    return package / ".lake" / "build" / "bin" / name


def audit_of(a: argparse.Namespace) -> Path:
    if a.audit is not None:
        return a.audit
    found = sorted(d for d in a.serve_dir.glob("v3c-w*/audit") if (d / "window" / "case.json").is_file())
    _need(len(found) == 1, f"{a.serve_dir} holds {len(found)} audited windows with a case; pass --audit")
    return found[0]


def period_of(a: argparse.Namespace) -> Path | None:
    """The period case's directory: ``--period``, else with ``--serve-dir`` `PERIOD_PIECE` when it holds a case."""
    if a.period is not None:
        return a.period
    if a.serve_dir is not None and (a.serve_dir / PERIOD_PIECE / "case.json").is_file():
        return a.serve_dir / PERIOD_PIECE
    return None


def pearlplus_of(a: argparse.Namespace) -> Path | None:
    """Pearl+'s case's parts: ``--pearlplus``, else with ``--serve-dir`` `PEARL_PLUS_PIECE` when it holds them."""
    if a.pearlplus is not None:
        return a.pearlplus
    if a.serve_dir is not None and (a.serve_dir / PEARL_PLUS_PIECE / "parts.json").is_file():
        return a.serve_dir / PEARL_PLUS_PIECE
    return None


def produce(a: argparse.Namespace) -> int:
    private, bad, period, pearlplus = a.private.resolve(), 0, period_of(a), pearlplus_of(a)
    try:
        audit = audit_of(a)
        v1 = window_case(audit)
        rec, digest = record_of(v1)
    except Refused as e:
        for id in [V1, SAMPLED, INFERENCE_ONLY] + ([PERIOD] if period else []) + ([PEARL_PLUS] if pearlplus else []) \
                + ([V2] if a.v2 else []):
            print("NOT_WRITTEN " + json.dumps({"protocol": id, "reason": str(e)}, sort_keys=True), flush=True)
        return 1

    kept: dict[str, dict[str, Any]] = {}

    def written(id: str, case: Mapping[str, Any], line: str, verdict: Any) -> None:
        write(private / DIRS[id], case, line, kept.get(id))
        print("CASE " + json.dumps({"dir": str(private / DIRS[id]), "protocol": id, "verdict": verdict},
                                   sort_keys=True), flush=True)

    def refused(id: str, e: Exception) -> None:
        nonlocal bad
        bad = 1
        print("NOT_WRITTEN " + json.dumps({"protocol": id, "reason": str(e)}, sort_keys=True), flush=True)

    def one(id: str, make, exe, cmd: Sequence[str] = ()) -> None:
        try:
            case = make()
            answer, line = lean_answer(exe(), case, private / "scratch", cmd)
            tie(answer, id, digest)
        except (Refused, PF.LeanError) as e:
            return refused(id, e)
        written(id, case, line, answer.get("verdict"))

    one(V1, lambda: v1, functools.cache(lambda: a.v1_exe or build(*V1_LEAN)))
    one(SAMPLED, lambda: sampled_case(audit), functools.cache(lambda: a.sampled_exe or build(*SAMPLED_LEAN)), SAMPLED_CMD)
    inference_only_exe = functools.cache(lambda: a.inference_only_exe or build(*policy().LEAN))

    def inference_only() -> dict[str, Any]:
        if a.log is not None:
            log = _json(a.log)
        elif a.serve_dir is not None and (a.serve_dir / H1_LOG).is_file():
            log = h1_log(a.serve_dir / H1_LOG, rec)
        else:
            log = []
        _need(isinstance(log, list), "--log is a JSON list of registration records")
        if not a.from_division:
            _need(a.alpha is not None and a.gamma is not None, "α and Γ are the policy's, not the run's: pass --alpha "
                  "and --gamma, or --from-division")
            return inference_only_case(v1, a.alpha, a.gamma, log)
        alpha, gamma = division_policy(v1, log, inference_only_exe(), private / "scratch")
        kept[INFERENCE_ONLY] = {"policy.json": {"alpha": _ratio(alpha), "gamma": gamma, "read": FROM_DIVISION}}
        return inference_only_case(v1, alpha, gamma, log)

    one(INFERENCE_ONLY, inference_only, inference_only_exe)
    if period is not None:
        try:
            written(PERIOD, *period_case(period, digest))
        except Refused as e:
            refused(PERIOD, e)
    if pearlplus is not None:
        try:
            written(PEARL_PLUS, *pearlplus_case(pearlplus, digest))
        except Refused as e:
            refused(PEARL_PLUS, e)
    if a.v2:
        one(V2, lambda: v2_case(audit), functools.cache(lambda: a.v2_exe or PF.v2_executable()))
    return bad


class Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(USAGE, f"{self.prog}: error: {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = Parser(description=__doc__.split("\n\n")[0])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--audit", type=Path, help="the window's records_audit.py out directory")
    where.add_argument("--serve-dir", type=Path, help="the serve dir, whose one v3c-w*/audit is the audit")
    ap.add_argument("--private", type=Path, required=True, help="where each case's directory goes (0700)")
    ap.add_argument("--alpha", type=Fraction, help="the policy's α, N/D: fresh-input bits a gate")
    ap.add_argument("--gamma", type=int, help="the policy's Γ: gates an inference block")
    ap.add_argument("--from-division", action="store_true",
                    help="α and Γ read off the case's own division (division_policy) instead of given")
    ap.add_argument("--log", type=Path, help="the deployment's registration records before the window's, a JSON list "
                    "(default with --serve-dir: its H1 log's, h1_log)")
    ap.add_argument("--period", type=Path, help="PoUW's period case, pouw_piece.py period's --out (default with "
                    f"--serve-dir: its {PERIOD_PIECE} when it holds a case)")
    ap.add_argument("--pearlplus", type=Path, help="Pearl+'s case's parts, served_passes.py case's --out (default with "
                    f"--serve-dir: its {PEARL_PLUS_PIECE} when it holds them)")
    ap.add_argument("--v1-exe", help=f"{V1_LEAN[1]} (default: built here)")
    ap.add_argument("--sampled-exe", help=f"{SAMPLED_LEAN[1]}, sampled proofs' as its {SAMPLED_CMD[0]} command "
                    "(default: built here)")
    ap.add_argument("--inference-only-exe", help="the policy's inference-only (default: built here)")
    ap.add_argument("--v2", action="store_true", help="one-stage v2's case too (v2_case), in sampled-proofs-v2/")
    ap.add_argument("--v2-exe", help=f"{PF.V2_EXE}, with --v2 (default: built here)")
    a = ap.parse_args(argv)
    if a.from_division and (a.alpha is not None or a.gamma is not None):
        ap.error("--from-division reads α and Γ off the division: not with --alpha or --gamma")
    if a.v2_exe is not None and not a.v2:
        ap.error("--v2-exe is v2's Lean, which only --v2 runs")
    return produce(a)


if __name__ == "__main__":
    sys.exit(main())
