"""A stop class's stop population audited to its record: the stop blocks `two_stage.py first` registered and sampled
(a class's ``stage1.json``, its ``stop``), each sampled block proved in one `--zk` session, verified by Lean
(`flock-verify verify --zk --registered`), and the stop stream's outcome with its audit record.

    python benchmarks/one_stage/stop_audit.py --stage1 F --boundary DIR --stop JSON --design rate:NUM/DEN \\
      --bin flock-circuit --lean flock-verify --out DIR [--boundary-outcome F] [--firewall flock-firewall] [--forge I,J]

* **The statement** (`member`, `bound`, `circuit`, from public items: the stop block and the stop population's
  partition, which the verifier takes from its own layout and holds the prover's circuit file to). One
  flat statement over the class's N stop blocks, unit i span i's, whose three ports are all registered reads: ``request``
  and ``ids`` are rows of the class's boundary record's `b.request` and `b.egress`, and the output row ``o0`` is the stop
  record's `b.stop` row, each at span i's position under the root its record registers. A stop block's rows are short:
  Flock's circuit file takes the statement only where a row has a compression before its last block, an ``ids`` row of
  112 bytes or more, which is a class of 32 outputs or more.
* **The prover's half** (`opened`, `staged`: the only code here that reads the class's private files). It opens the
  serve's own trees again from the spans and the salts (`boundary_rows.reopen`) and stages every stop block on the
  serve's rows, salts and frame-v3 paths, so each row's ``b || c`` is the one the serve committed. ``--forge`` is a
  negative's prover: those spans' `b.stop` word is committed with bit 0 flipped, a length that is not the rule's, and
  proved with the prover's self-check skipped.
* **The verifier's half** (`resume`, `reads`, `circuit`, `decide`), from ``stage1.json``, the verifier's own check template, stop
  rule and sampling design, and the session's public files; it opens no file of the class's directory.
  The boundary record is the one the class's boundary stream registers again (`two_stage.resume`), never one rebuilt
  from ``commit.json``. The stop stream is built again on it, registers the stop record (R1 to R7) and samples it
  again on the bytes its sample read; each is held to ``stage1.json``. The ``--registered`` file is the verifier's
  own: the two parameters' ports as `boundary_stop.stop_registration` gives them, and ``o0`` at the stop record's
  root.
* **The audit record** is the stop stream's (`Stream.outcome`: ``verity/one-stage/audit/v0``, Lean's `Record.record`).
  Beside Lean's verdict of record on the session it holds the entries below, none of record. An entry whose
  ``accepted`` is null did not run: the record can accept, and says ``"complete": false``.

      m0-verifier/stop     the session's own verdict and its sample
      registered-reads     Lean's verdict names the verifier's own ``--registered`` file and reports its roots
      cross-stream-order   Lean's verdict on the read across the two streams (`across`: the Lean verifier's piece
                           ``across``, `Record.across`), never ``stage1.json``'s ``order``: the stop record's
                           ``previous`` is a record of the boundary stream's log, and the challenger's signed log
                           shows the boundary registration's coins before the stop registration's, which the stop
                           sample read whole
      boundary-audit       the boundary stream's outcome on that record (``--boundary-outcome``, a record of that
                           stream's `Stream.records`): null when none is handed; refused when it is not that
                           registration's accept, or Lean's verdict on its audit record refuses it; null, with
                           its reason, when that audit record accepted and is not complete (it accepted with
                           entries that did not run, and is no audit of those); accepted when it accepted and
                           is complete
      request-row          null: nothing holds `b.request`'s words to the certified request
      stop-rule            null: nothing holds ``--stop`` to the pinned checkpoint

  ``case`` is Lean's, of the whole read: 1 when the log and the order hold and the boundary stream's audit record
  accepted and is complete; 2 when the log or the order does not hold; 3 when they do and that audit record is missing
  or did not accept; 4 when they do and that audit record accepted and is not complete. A record whose boundary
  stream's audit record is missing (case 3) or accepted and is not complete (case 4) has no verdict on it: it can
  accept, and is not complete, so no record is complete on a boundary stream's audit record that is not.
  That audit record is the one the driver hands. Lean checks its format and its registration and reads its
  ``accepted`` and ``complete``, and nothing holds it to the record the boundary stream's verifier wrote, so case 1
  says no more than the record handed says. The two entries that read it, ``boundary-audit`` and
  ``cross-stream-order`` (its ``case``), name that as their premise.
  ``summary.json`` copies the record's ``accepted`` and ``complete`` and lists every entry's ``premise``. It says the
  response's length is held (``length_held``) only when the record accepted and is complete.
* **The proof deadline** counts on the stream built again here, from its sample read again, not from the first
  stage's: nothing here holds the time between the two commands.

Exit 0 when the audit record accepted, 1 when it did not or nothing was audited."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import prove_leg as PL  # noqa: E402
import two_stage as TS  # noqa: E402
import verity.definitions.registry.boolean_stop_rule  # noqa: E402, F401  (the stop rule's words on gates: Flock's pieces for them)
from verity.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.circuits.defs import SpecializedDefinition  # noqa: E402
from verity.commitments.merkle import tree_levels  # noqa: E402
from verity.commitments.rowleaf import row_leaf  # noqa: E402
from verity.commitments.windows import rows as CS, tables as TB  # noqa: E402
from verity.protocols.spot.verifier import service as S  # noqa: E402
from verity.protocols.spot.verifier.one_stage import partition as P, registered as REG, registration as R  # noqa: E402
from verity.protocols.spot.verifier.one_stage import verifier as V  # noqa: E402
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS  # noqa: E402

FORMAT = "verity/one-stage/stop-audit/v0"
TOOL = "benchmarks/one_stage/stop_audit.py"
#: the statement's one member, and its output row's circuit port
MEMBER, OUT = "stop", "o0"

LEAN, M0, READS = f"lean-verify/{MEMBER}", f"m0-verifier/{MEMBER}", "registered-reads"
ORDER, BOUNDARY_AUDIT, REQUEST_ROW, STOP_RULE = TS.ORDER_REFUSED, "boundary-audit", "request-row", "stop-rule"
#: the tool of an entry that is the Lean verifier's piece ``across``'s verdict
ACROSS = f"{V.EXE} across"
#: of every entry that reads the boundary stream's audit record: ``boundary-audit``, and ``cross-stream-order``'s ``case``
BOUNDARY_PREMISE = ("boundary stream's audit: read from the audit record the audit driver hands; Lean checks its format "
                    "and registration and reads its `accepted` and `complete`; nothing holds it to the record the "
                    "boundary stream's verifier wrote")
REQUEST_PREMISE = "request row: the engine's words, not held to the certified request"
NOT_COMPLETE = "the boundary stream's audit record accepted with entries that did not run: it is not complete"
STOP_RULE_PREMISE = "stop rule: the driver's parameter, not held to the pinned checkpoint"


class Unlogged(ValueError):
    """A stop record whose ``previous`` is no record of the class's boundary stream's log (``first``'s): case 2."""

    def __init__(self, why: str, first: TS.First) -> None:
        super().__init__(why)
        self.first = first


def sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


# --- public: the statement --------------------------------------------------------------------------------------------

def member(layout: BR.Layout, values: Mapping[str, np.ndarray] | None = None, **more: Any) -> PL.Member:
    """The stop population as the proof leg's one member: the verifier's stop block, its two parameters' rows (each of
    its boundary value's words) and its one output row, all bit rows of the class's schema. ``values``: the prover's
    rows, by parameter."""
    rows = [(r, BR.SCHEMA, 16 * layout.words(port), [r]) for r, port in BS.STOP_READS.items()]
    return PL.Member(MEMBER, layout.stop.template, rows, [(BR.SCHEMA, (0,))], dict(values or {}), **more)


def bound(layout: BR.Layout) -> dict[str, str]:
    """What the statement binds: the stop population's partition and its program."""
    part = layout.stop_partition
    return {"partition": P.digest(part), "program": part["program"]}


def files(layout: BR.Layout, out: Path) -> dict[str, Path]:
    """The partition and program files of the stop population, written under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    got = {"partition_file": out / "partition.json", "program_file": out / "program.json"}
    got["partition_file"].write_text(json.dumps(layout.stop_partition, sort_keys=True))
    got["program_file"].write_bytes(canonical_json(encode_program(layout.stop_program)))
    return got


def circuit(layout: BR.Layout, out: Path) -> dict[str, str]:
    """The statement's circuit file as ``layout`` gives it, lowered under ``out``: its pin and its file's SHA-512. The
    verifier lowers its own and holds the prover's file to it."""
    st, _ = PL._stager(member(layout), out / f"stage-{MEMBER}", bound(layout), None)
    return {"pin": st.pin, "circuit_sha512": hashlib.sha512((out / f"stage-{MEMBER}" / "circuit.txt").read_bytes()).hexdigest()}


# --- the prover's half ------------------------------------------------------------------------------------------------

def opened(c: BR.Committed, forge: Sequence[int] = ()) -> dict[str, Any]:
    """The class's committed rows as its stop blocks read them, from the prover's own files (``c``,
    `boundary_rows.reopen`'s): by circuit port, the rows, each row's salt and its frame-v3 path in the serve's tree,
    whose root is ``c``'s. ``forge``: spans whose `b.stop` word has bit 0 flipped, as a prover that committed another
    length than the rule's holds it; the tree is then the one over those words."""
    L = c.layout
    if not L.stopped:
        raise ValueError("the class is no stop class: it has no stop blocks")
    words = BR.port_rows(c.spans, L)
    for i in forge:
        words[BR.STOP][int(i), 0] ^= 1
    ports = {**BS.STOP_READS, OUT: BR.STOP}
    domains = {v: L.domain(v) for v in ports.values()}
    roots, bc = CS.commit_ports(domains, {v: words[v] for v in domains}, {}, {v: c.salts[v] for v in domains},
                                schemas=dict.fromkeys(domains, BR.SCHEMA))
    got: dict[str, Any] = {"rows": {}, "salts": {}, "paths": {}, "roots": {}}
    for p, v in ports.items():
        if not forge and roots[v] != c.roots[v]:
            raise ValueError(f"{v}: the class's files recommit to another root than its commitment's")
        strings = [bc[v][i * CS.BC_BYTES:(i + 1) * CS.BC_BYTES] for i in range(L.spans)]
        lv = tree_levels(domains[v], [row_leaf(domains[v], i, CS.INST.tree_leaf(CS.INST.default_key, s), BR.SCHEMA)
                                      for i, s in enumerate(strings)])
        got["rows"][p] = words[v]
        got["salts"][p] = [c.salts[v][i * CS.SALT_BYTES:(i + 1) * CS.SALT_BYTES] for i in range(L.spans)]
        got["paths"][p] = [b"".join(level[(i >> d) ^ 1] for d, level in enumerate(lv[:-1])).hex() for i in range(L.spans)]
        got["roots"][v] = roots[v].hex()
    return got


def staged(c: BR.Committed, out: Path, forge: Sequence[int] = ()) -> dict[str, Any]:
    """The prover's statement of the class's stop blocks under ``out`` (`prove_leg.stage`: a0's statement dict, with
    ``roots``, the trees' roots it opens its rows under): every block on the serve's own rows, salts and paths
    (`opened`), each port a registered read, so no row is committed a second time. ``forge``: `opened`'s, staged as
    `prove_leg`'s forged member is. ValueError (`prove_leg.OutsideShape`) for a population whose statement is outside
    Flock's shape."""
    L = c.layout
    mine = opened(c, forge)
    n = L.spans
    values = {p: mine["rows"][p] for p in BS.STOP_READS}
    names = {**{p: p for p in BS.STOP_READS}, OUT: CS.own_value(BR.STOP)}
    kw = {"tables": values, "refs": {p: np.arange(n) for p in values}, "salts": mine["salts"],
          "registered": {p: {"value": v, "positions": list(range(n)), "paths": mine["paths"][p]} for p, v in names.items()}}
    st = PL.stage(member(L, values, forge=tuple(int(i) for i in forge), stage_kw=kw), 0, bound(L), out, os.urandom(32),
                  files(L, out))
    return {**st, "roots": mine["roots"]}


def session(st: Mapping[str, Any], units: Sequence[int], out: Path, *, bin_: str, lean: str, firewall: str | None = None,
            forged: bool = False, timeout: float = 3600) -> dict[str, Any]:
    """One `--zk` session over statement ``st`` (`staged`'s, with the verifier's ``registered_file``) serving exactly
    the sampled ``units`` (`a0.session`, behind the ZK server when ``firewall`` names Lean's frame), then Lean's
    verdict of record on it (`a0.lean_verify`). ``forged``: the prover's self-check is skipped. -> what `decide` takes:
    the session's records' SHA-512, the units it served and proved, its instances, the verdicts and its timing."""
    out.mkdir(parents=True, exist_ok=True)
    n, local = st["n"], sorted(int(u) for u in units)
    (out / "draw.json").write_text(json.dumps({"law": "subset", "population": n, "k": len(local), "units": local},
                                              sort_keys=True))
    env = {**a0.prover_env(firewall, st), **({"FC_ZK_SELF_CHECK": "skip"} if forged else {})}
    t = time.time()
    record, live = a0.session(bin_, st, n, f"subset:{len(local)}", PL.free_port(), out, draw_file=out / "draw.json",
                              zk=True, env=env, prove_timeout=timeout)
    seconds = round(time.time() - t, 1)
    lv = a0.lean_verify(lean, st, n, out, zk=True)
    draw = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw, {"population": n}, {"law": "subset", "k": len(local)})
    m0 = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True and v.ok,
          "why": ((record or {}).get("verdict") or {}).get("why") or (live or {}).get("why"), "draw_check": v.to_json()}
    return {"sessions": [a0.sha512_file(f) for f in sorted((out / "sessions").glob("*/session.json"))],
            "served": [int(u) for u in draw.get("units", [])], "proved_units": a0.proved_units(st["header"], live or {}),
            "instances": (live or {}).get("instances", 0),
            "verdicts": {LEAN: {**lv, "tool": "flock-verify verify --zk --archive --partition --program --registered"},
                         M0: m0},
            "statement": {MEMBER: {"session_s": seconds, "lean_verify_s": lv.get("verify_s"),
                                   **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                                   "timing": {x: (live or {}).get(x) for x in PL.TIMING}}}}


# --- the verifier's half ----------------------------------------------------------------------------------------------

@dataclass
class Resumed:
    """A stop class's two streams again at the end of its first stage: the boundary stream's (`two_stage.resume`), and
    the stop stream on the record of its log the stop record names, with the stop record's registration, the copies it
    was held to and its sample."""

    first: TS.First
    stream: S.Stream
    held: Any
    registration: S.WindowCommitmentRecord
    sample: S.Draw
    #: the boundary record's index in the boundary stream's log
    received: int = 0

    @property
    def layout(self) -> BR.Layout:
        return self.first.layout

    @property
    def boundary(self) -> str:
        """The digest of the boundary record the stop record reads, a record of the boundary stream's log."""
        return self.registration.record["previous"]


def resume(stage1: Mapping[str, Any], check: SpecializedDefinition, stop: BS.Stop, design: str, *, delta: float = 0.5,
           proof_deadline: float = TS.PROOF_DEADLINE_S) -> Resumed:
    """The class's stop stream again from its ``stage1.json``, under the verifier's own check template, stop rule and
    sampling design: the boundary stream's record registered again (`two_stage.resume`), the stop stream built on the
    record of that stream's log (`Stream.log`) whose digest is the stop record's ``previous``, the stop record
    registered (R1 to R7) and sampled again on the bytes its sample read. `Unlogged` when that log holds no such
    record; ValueError when the first stage sampled no stop block, or
    the Spec, the record, its registration or the sample isn't ``stage1.json``'s; `service.Refused` as the streams
    refuse."""
    s = stage1.get("stop")
    if not isinstance(s, Mapping) or "units" not in s:
        raise ValueError(f"the class's first stage registered or sampled no stop block: {(s or {}).get('error') or stage1.get('error')}")
    root = stage1.get("window_root")
    first = TS.resume(stage1, check, delta=delta, window_root=root, stop=stop)
    layout, logged = first.layout, {R.digest(rec): (i, rec) for i, rec in enumerate(first.stream.log)}
    if s["record"].get("previous") not in logged:
        raise Unlogged(f"the stop record's previous, {str(s['record'].get('previous'))[:16]}, is no record of the class's "
                       f"boundary stream's log, which holds {', '.join(d[:16] for d in logged)}", first)
    received, boundary = logged[s["record"]["previous"]]
    stream = S.Stream(TS.stop_spec(layout, design, proof_deadline), layout.stop.template, budget=1, delta=delta,
                      challenger=TS.CHALLENGER, coins=TS.Staged(bytes.fromhex(s["read"])), log=[boundary])
    if stream.spec.digest() != s["spec"]:
        raise ValueError("the stop population's Spec isn't the one its first stage ran under")
    held = BS.stop_registration(stage1["commit"], check, stop, boundary)
    want = BS.stop_record(stage1["commit"], check, stop, S.selection_law(stream.spec.selection, layout.spans), boundary)
    if R.digest(want) != R.digest(s["record"]):
        raise ValueError("the stop record isn't the one the class's commitment and its boundary record give")
    reg = stream.register(s["record"], layout.stop_ports(), registered=held.values, outputs=held.outputs,
                          schemas=held.schemas, **({} if root is None else {"window_root": root}))
    if reg.digest != s["registration"]:
        raise ValueError("the stop record registers to another digest than its first stage's")
    sample = stream.select(reg)
    if list(sample.units) != list(s["units"]) or sample.draw != s["sample"]:
        raise ValueError("the stop stream samples other stop blocks read again than its first stage did")
    return Resumed(first, stream, held, reg, sample, received)


def reads(r: Resumed) -> dict[str, Any]:
    """The verifier's own ``--registered`` file for the stop blocks' statement: each parameter's port as the stop
    registration gives it (`boundary_stop.stop_registration`: the boundary record's value, root and domain, unit i at
    row i), and the output row's, the stop record's `b.stop` at its registered root, leaves and domain, unit i at row
    i, read as `rows.own_value` names a record's own port. Lean refuses a statement that leaves a port out."""
    L = r.layout
    stop = next(e for e in r.registration.record["roots"] if e["port"] == BR.STOP)
    mine = L.stop_ports()[BR.STOP]
    ports = dict(r.held.file["ports"])
    ports[OUT] = {"value": CS.own_value(BR.STOP), "root": stop["root"], "leaves": mine["leaves"], "words": L.words(BR.STOP),
                  "word_bits": REG.WORD_BITS, "domain": mine["domain"], REG.SCHEMA_KEY: BR.SCHEMA,
                  "reads": list(range(L.spans))}
    return {**r.held.file, "ports": dict(sorted(ports.items()))}


def link(stage1: Mapping[str, Any], first: TS.First, outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The stop record's read across the class's two streams, as the Lean verifier's piece ``across`` takes it: the
    window's root, the boundary stream's own log (``first``'s `Stream.log`), the boundary stream's audit record
    (``outcome``'s, none when no outcome is handed), the stop record, and the stop sample's coin evidence and the
    bytes it read (``stage1``'s ``stop``)."""
    s = stage1["stop"]
    return {"window": stage1.get("window_root") or first.layout.run, "log": [dict(x) for x in first.stream.log],
            "audit": outcome.get("audit") if isinstance(outcome, Mapping) else None,
            "later": dict(s["record"]), "coins": s.get("evidence"), "read": s["read"]}


def across(stage1: Mapping[str, Any], first: TS.First, outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Lean's verdicts on the stop record's read across the class's two streams (`Record.across`, the Lean verifier's
    piece ``across``, on `link`'s read): ``logged``, ``order`` and ``audit``, each ``{"accepted", "why"}``, ``audit``
    with ``complete`` too (whether the boundary stream's audit record says every one of its entries ran), with
    ``case``, the boundary record's index in the log (``received``) and the two registrations' digests (``earlier``,
    ``later``). ``{"refused": reason}`` where Lean refuses the input or can't run."""
    return V.lean("across", link(stage1, first, outcome))


def order_entry(r: Resumed, lean: Mapping[str, Any], audit: str | None = None) -> dict[str, Any]:
    """The ``cross-stream-order`` entry from Lean's answer ``lean`` (`across`'s): accepted when its ``logged`` and its
    ``order`` both are and it decided on this stream's own registrations: its ``earlier``, ``later`` and ``received``
    are the boundary record's digest, the stop registration's and the boundary record's index in the boundary stream's
    log, as ``r`` holds them. ``case`` is Lean's, and reads the boundary stream's audit record as the driver hands it:
    the entry names that record's premise (`BOUNDARY_PREMISE`), as the ``boundary-audit`` entry does. ``audit``: the
    boundary stream's audit record's SHA-512."""
    mine = {"earlier": r.boundary, "later": r.registration.digest, "received": r.received}
    why = f"Lean refused the read: {lean['refused']}" if "refused" in lean else None
    for k in ("logged", "order"):
        v = lean.get(k) or {}
        if why is None and v.get("accepted") is not True:
            why = v.get("why") or f"Lean gave no verdict on {k}"
    if why is None and any(lean.get(k) != v for k, v in mine.items()):
        why = "Lean's verdict is on other registrations than this stream's"
    return {"accepted": why is None, "of_record": False, "tool": ACROSS, "case": lean.get("case"),
            "premise": BOUNDARY_PREMISE, **({} if why is None else {"why": why}),
            "boundary": {"registration": r.boundary, "received": r.received, "audit": audit},
            "stop": {"registration": r.registration.digest, "previous": r.registration.record.get("previous")}}


def boundary_audit(boundary: str, outcome: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the driver itself reads of the boundary stream's outcome record ``outcome`` on the class's boundary
    registration ``boundary`` (an entry of that stream's `Stream.records`: it names the registration and holds the
    audit record): ``accepted``, ``complete`` and, where it did not accept, ``why``, as `Record.across`'s ``audit``
    gives them of the audit record inside. ``accepted``: none when no outcome is handed; false when it is no outcome
    record, is another registration's, or its audit record did not accept. ``complete``: none unless it is that
    registration's outcome record with an audit record, and then whether that audit record says every one of its
    entries ran."""
    if outcome is None:
        return {"accepted": None, "complete": None}
    audit = outcome.get("audit")
    if outcome.get("format") != S.OUTCOME:
        return {"accepted": False, "complete": None, "why": f"not a {S.OUTCOME} record"}
    if outcome.get("registration") != boundary:
        return {"accepted": False, "complete": None,
                "why": "the outcome is another registration's than the class's boundary record's"}
    complete = audit.get("complete") is True if isinstance(audit, Mapping) else None
    if outcome.get("symbol") != S.ACCEPT or not isinstance(audit, Mapping) or audit.get("accepted") is not True:
        return {"accepted": False, "complete": complete,
                "why": f"the boundary stream's outcome is {outcome.get('symbol')!r}: its audit record did not accept"}
    return {"accepted": True, "complete": complete}


def boundary_entry(boundary: str, outcome: Mapping[str, Any] | None, lean: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``boundary-audit`` entry, from the boundary stream's outcome record on the class's boundary registration
    ``boundary`` (`boundary_audit`): null when none is handed; refused when it is no outcome record, is another
    registration's, or its audit record did not accept; null again, with its ``why``, when that audit record accepted
    and is not complete, so that the stop stream's record can accept and is then not complete either; accepted when
    it accepted and is complete. ``lean``: Lean's verdict on that audit record (`across`'s ``audit``), which the entry
    then needs too, its ``accepted`` and its ``complete``, and holds. ``complete``: whether the audit record is
    complete, by the driver and by Lean's verdict when given; ``audit``: the audit record's SHA-512."""
    entry = {"accepted": None, "complete": None, "of_record": False, "tool": TOOL, "premise": BOUNDARY_PREMISE,
             "audit": None, "registration": boundary}
    if outcome is None:
        return entry
    mine, audit = boundary_audit(boundary, outcome), outcome.get("audit")
    why, complete = mine.get("why"), mine["complete"]
    if why is None and lean is not None and lean.get("accepted") is not True:
        why = f"Lean's verdict on the boundary stream's audit record: {lean.get('why') or 'none'}"
    if why is None and lean is not None:
        complete = complete is True and lean.get("complete") is True
    accepted: bool | None = why is None
    if accepted and complete is not True:
        accepted, why = None, NOT_COMPLETE
    return {**entry, "accepted": accepted, "complete": complete, "audit": None if audit is None else sha512(dict(audit)),
            "outcome": sha512(dict(outcome)), **({} if why is None else {"why": why}),
            **({} if lean is None else {"lean": {"tool": ACROSS, **lean}})}


def reads_entry(file: Mapping[str, Any], lean: Mapping[str, Any]) -> dict[str, Any]:
    """The ``registered-reads`` entry: Lean's verdict accepted the session against the verifier's own ``--registered``
    file (its SHA-512, which the verdict names) and reports exactly its ports' values and roots
    (`tables.reported_reads`)."""
    got = TB.reported_reads({MEMBER: file}, {MEMBER: lean})
    named = lean.get("registered_sha512") == hashlib.sha512(canonical_json(file)).hexdigest()
    return {"accepted": bool(got["ok"]) and named, "of_record": False, "tool": TOOL,
            "file": hashlib.sha512(canonical_json(file)).hexdigest(), "named": lean.get("registered_sha512"),
            "reads": got["chunks"][MEMBER]["reads"]}


def verdicts_of(r: Resumed, stage1: Mapping[str, Any], file: Mapping[str, Any], proved: Mapping[str, Any],
                outcome: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """The audit record's verdicts (module docstring): the session's, Lean's of record naming the sessions delivered,
    Lean's on the read across the two streams (`across`), and the driver's entries."""
    got = {k: dict(v) for k, v in proved["verdicts"].items()}
    got[LEAN] = {**got[LEAN], "of_record": True, "sessions": list(proved["sessions"])}
    got[READS] = reads_entry(file, got[LEAN])
    lean = across(stage1, r.first, outcome)
    refused = {"accepted": False, "why": f"Lean refused the read: {lean.get('refused')}"}
    got[BOUNDARY_AUDIT] = boundary_entry(r.boundary, outcome, refused if "refused" in lean else lean.get("audit") or {})
    got[ORDER] = order_entry(r, lean, got[BOUNDARY_AUDIT]["audit"])
    got[REQUEST_ROW] = {"accepted": None, "of_record": False, "premise": REQUEST_PREMISE}
    got[STOP_RULE] = {"accepted": None, "of_record": False, "premise": STOP_RULE_PREMISE}
    return got


def decide(r: Resumed, stage1: Mapping[str, Any], file: Mapping[str, Any], proved: Mapping[str, Any],
           outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The stop stream's outcome of its sample: the sessions delivered (`Stream.deliver`), then `Stream.outcome` on the
    verdicts (`verdicts_of`) with the stop sample's coin evidence. The audit record's own checks hold the roots Lean's
    accepting verdict opened the session's rows under to the stop registration's: the two parameters as its registered
    values, and the output row as the session's root of `b.stop` (none when the verdict did not accept, which fails
    ``roots``). -> ``accepted``, the outcome's record and symbol and the audit record; or ``refused``, with no audit
    record, when the stream refuses the delivery or the outcome."""
    verdicts = verdicts_of(r, stage1, file, proved, outcome)
    lean = proved["verdicts"][LEAN]
    opened = (lean.get("registered") or {}).get(OUT) if lean.get("accepted") is True else None
    try:
        o = r.stream.deliver(r.sample, proved["sessions"]) or r.stream.outcome(
            r.sample, verdicts, proved["statement"], coins=stage1["stop"].get("evidence"),
            proved_units=proved["proved_units"], instances=proved["instances"], served=proved["served"],
            session_roots={BR.STOP: (opened or {}).get("root")})
    except (S.Refused, S.Pending) as e:
        return {"accepted": False, "refused": e.verdict.to_json() if isinstance(e, S.Refused) else str(e)}
    body = r.stream.records[-1]
    return {"accepted": o.accepted, "record": o.record, "symbol": body["symbol"], "audit": body["audit"]}


def summary_of(r: Resumed, stop: BS.Stop, design: str, got: Mapping[str, Any]) -> dict[str, Any]:
    """What a run reports of an outcome (`decide`'s): the audit record's ``accepted`` and ``complete`` as they are, each
    verdict, why one did not accept or gives a reason for having none, Lean's ``case`` of the read across the streams,
    every entry's premise, and whether the response's length is held: only by a record that accepted and is complete."""
    L, audit = r.layout, got.get("audit") or {}
    verdicts = audit.get("verdicts") or {}
    accepted, complete = audit.get("accepted") is True and got.get("accepted") is True, audit.get("complete") is True
    return {"format": FORMAT, "class": [L.prompt, L.outputs], "spans": L.spans, "run": L.run, "stop_rule": stop.to_json(),
            "design": design, "sample": list(r.sample.units),
            "registration": {"boundary": r.boundary, "stop": r.registration.digest},
            "accepted": accepted, "complete": complete, "length_held": accepted and complete,
            "case": (verdicts.get(ORDER) or {}).get("case"),
            "symbol": got.get("symbol"), "record": got.get("record"), "refused": got.get("refused"),
            "failed": sorted(k for k, c in (audit.get("checks") or {}).items() if not c.get("ok")),
            "verdicts": {k: v.get("accepted") for k, v in verdicts.items()},
            "why": {k: v.get("why") for k, v in verdicts.items()
                    if v.get("accepted") is False or (v.get("accepted") is None and v.get("why"))},
            "not_run": sorted(k for k, v in verdicts.items() if v.get("accepted") is None),
            "premises": sorted({v["premise"] for v in verdicts.values() if "premise" in v})}


# --- the command ------------------------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", required=True, help="the class's stage1.json (two_stage.py first --stop)")
    ap.add_argument("--boundary", required=True, help="the class's directory, its private files too: the prover's half reads it")
    ap.add_argument("--stop", required=True, help=TS.STOP_HELP)
    ap.add_argument("--design", required=True, help="the verifier's own sampling design over the stop blocks, a rate: "
                                                    "two_stage.py first's --stop-design, else its --replay")
    ap.add_argument("--boundary-outcome", help="the boundary stream's outcome record on the class's boundary record (an "
                                               "entry of its Stream.records); without it the boundary-audit entry is null")
    ap.add_argument("--bin", required=True, help="flock-circuit")
    ap.add_argument("--lean", required=True, help="flock-verify")
    ap.add_argument("--firewall", help="Lean's flock-firewall: the session behind the ZK server; without it the prover "
                                       "runs prove --zk itself, as SPOT[Flock]'s proof server does")
    ap.add_argument("--forge", default="", help="a negative's prover: these spans' b.stop word committed with bit 0 "
                                                "flipped, the class's commitment holding that tree's root")
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--proof-deadline", type=float, default=TS.PROOF_DEADLINE_S)
    ap.add_argument("--session-timeout", type=float, default=3600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage1 = json.loads(Path(a.stage1).read_text())
    outcome = json.loads(Path(a.boundary_outcome).read_text()) if a.boundary_outcome else None
    forge = [int(i) for i in a.forge.split(",") if i]

    def done(code: int, summary: Mapping[str, Any]) -> int:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
        print("STOP " + json.dumps({k: summary.get(k) for k in ("accepted", "complete", "length_held", "case", "refused",
                                                                "error", "sample", "failed", "verdicts", "why", "premises")},
                                   sort_keys=True, default=str))
        return code
    try:
        check, stop = TS.template_fn(stage1["commit"]["check"]), TS._stop(a.stop)
        if stop is None:
            raise ValueError("a stop population is audited under the verifier's own stop rule: --stop")
        # the verifier's half, from stage1.json and its own copies
        r = resume(stage1, check, stop, a.design, delta=a.delta, proof_deadline=a.proof_deadline)
        file = reads(r)
        (out / "registered.json").write_bytes(canonical_json(file))
        mine = {**files(r.layout, out / "verifier"), "registered_file": out / "registered.json"}
        own = circuit(r.layout, out / "verifier")
        # the prover's half: the one place the class's private files are read
        st = staged(BR.reopen(Path(a.boundary), check, stop=stop), out / "prover", forge)
        if (st["pin"], a0.sha512_file(st["dir"] / "circuit.txt")) != (own["pin"], own["circuit_sha512"]):
            raise ValueError("the prover's circuit file is not the one the verifier's own stop block lowers to")
        if st["roots"][BR.STOP] != file["ports"][OUT]["root"]:
            raise ValueError("the prover's b.stop tree is not the one the stop record registers"
                             + ("" if forge else ": a class forged at registration is proved with --forge"))
        proved = session({**st, **mine}, r.sample.units, out / "session", bin_=a.bin, lean=a.lean, firewall=a.firewall,
                         forged=bool(forge), timeout=a.session_timeout)
        got = decide(r, stage1, file, proved, outcome)
    except Unlogged as e:
        lean = across(stage1, e.first, outcome)
        return done(1, {"format": FORMAT, "accepted": False, "complete": False, "length_held": False,
                        "refused": [ORDER], "case": lean.get("case"), "across": lean, "error": f"{ORDER}: {e}"})
    except S.Refused as e:
        return done(1, {"format": FORMAT, "accepted": False, "complete": False, "length_held": False,
                        "refused": list(e.verdict.codes), "error": json.dumps(e.verdict.to_json(), default=str)[:2000]})
    except (ValueError, KeyError, OSError) as e:
        return done(1, {"format": FORMAT, "accepted": False, "complete": False, "length_held": False,
                        "error": f"{type(e).__name__}: {e}"})
    if got.get("audit") is not None:
        (out / "audit.json").write_text(json.dumps(got["audit"], indent=1, sort_keys=True, default=str) + "\n")
    summary = summary_of(r, stop, a.design, got)
    summary["statement"] = {"pin": st["pin"], "ands": st["ands"], "k_log": st["meta"].get("k_log"), **proved["statement"][MEMBER]}
    return done(0 if summary["accepted"] else 1, summary)


if __name__ == "__main__":
    sys.exit(main())
