"""[A3 contract 2026-09-10] The verifier's step-2 and step-4 checks of docs/PROTOCOL.md as pure functions, run by the live driver BEFORE
``r_q`` / ``r_s`` are derived (conformance R6, R8, R13, R16, R17, R18) and again post hoc by ``reverify.py`` (idempotent).

Step 2 (before J): every input gate of C and every output gate of every RU has a committed value -- here: one boundary tree per
request the epoch declares (the tree holds prompt ids, output ids, sampling params, seed, finish reason; layout A2's), the weights
tree with the registered root; x, y well-typed (declared output domain, multiplicity) and equal to the committed values (the served
``production.jsonl`` outputs == the boundary record's -- supplied by the caller from the prover's capture, A2's reader).

Step 4 (before Q): F ⊆ J, |F| <= B_ex (RU count), every retained RU's interior committed (an ``ru/<rid>/...`` tree in
``commitments.json``), phase-0 trees unchanged, both R17 fields present for every RU of J, every selected-eligible VU of a retained RU
lies in a committed interface (else the RU must be in F: an uncovered selected equation is HC, never a family exemption).

Nothing here reads the seed; nothing here draws.  Refusals are returned as a list of strings (empty == the step may proceed).

[E9 EOD 2026-09-11] Two additions, both read by the live driver BEFORE the corresponding challenge and again by ``reverify.py``:

* :class:`Transcript` -- the verifier's explicit state machine over the contract's events (brief interface 7).  ``release("r_q")`` is
  refused BY NAME (:class:`PrematureChallenge`, ``reason_code = PREMATURE_CHALLENGE``) unless ``step2_complete`` was recorded, and
  ``release("r_s")`` unless ``step4_complete`` was; the driver calls ``release`` immediately before ``partition_sample.stage1`` /
  ``stage2`` (the only two places a challenge is derived) and answers a prover's ``REQUEST_r_q`` / ``REQUEST_r_s`` handshake file with
  the same refusal (``REFUSED_r_q`` / ``REFUSED_r_s``).  The seed itself is never an argument of anything here (custody:
  ``partition_sample._HELD_SEEDS`` in the verifier process; see ``seed_custody_statement``).
* :func:`interior_layout_check` -- ``V3.interior_layout`` (brief interface 4): the pinned ``protocol_instance.json#commitments.interior_layout``
  must be present and equal E2's ``layout_digest()`` whenever an interior kind is in play, and every REQUIRED interior region of every
  retained RU must be supplied (E2's ``validate_supplied``) -- at step 4, before r_s (``STEP4_REFUSED``).  Fail-closed: a pin the verifier
  cannot validate (E2's module absent / interface mismatch) is a refusal, never a pass; no pin + no interior kind in play is a labelled
  vacuous pass.
"""

from __future__ import annotations

import json
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from e2e import protocol_instance as pi
from verity_vllm_adapter import response_record as rr

LEGACY_EXEMPT_CLASS = "exempt (charged)"  # DIRECTIVE v4 family label: LEGACY, not an exemption; refused by this verifier


# -- [E9] the verifier's state machine over the contract's transcript events (brief interface 7) -----------------------------------------

#: The contract's transcript states in order.  A challenge is derivable only from the state that follows its prerequisite CHECK, never
#: from the prover's message alone (READY_PHASE0 / READY_PHASE2 are the prover's signals; step2_complete / step4_complete are the
#: verifier's own findings).
TRANSCRIPT_STATES: tuple[str, ...] = ("init", "instance_pinned", "ready_phase0", "step2_complete", "r_q_released", "ready_phase2", "step4_complete",
                                      "r_s_released", "seed_revealed")
#: challenge -> the verifier state that must have been reached before it may be derived
CHALLENGE_REQUIRES: dict[str, str] = {"r_q": "step2_complete", "r_s": "step4_complete"}
PREMATURE_REASON = "PREMATURE_CHALLENGE"


class PrematureChallenge(Exception):
    """A challenge requested before its prerequisite step is complete.  Named: ``challenge``, the ``state`` the verifier is in, the
    ``required`` state.  The message is what the handshake refusal file and the transcript event carry."""

    def __init__(self, challenge: str, state: str, required: str, *, requested_by: str = "verifier") -> None:
        self.challenge, self.state, self.required, self.requested_by = challenge, state, required, requested_by
        super().__init__(f"{PREMATURE_REASON}: {challenge} requested by {requested_by} in state {state!r}; requires {required!r} "
                         f"({'step 2 checks (v_in = x, v_out = y, every boundary committed, population derived) have not passed' if challenge == 'r_q' else 'step 4 checks (F ⊆ J, |F| <= B_ex, every retained interior + required regions committed, output copies) have not passed'})")

    def to_json(self) -> dict[str, Any]:
        return {"reason_code": PREMATURE_REASON, "challenge": self.challenge, "state": self.state, "required": self.required, "requested_by": self.requested_by,
                "message": str(self)}


class Transcript:
    """The verifier's transcript state machine: monotone over :data:`TRANSCRIPT_STATES`; ``release(challenge)`` refuses by name unless the
    prerequisite state was reached.  Pure (no I/O, no seed): the driver records its events; tests drive it directly."""

    def __init__(self) -> None:
        self.state = "init"
        self.events: list[dict[str, Any]] = []
        self.refusals: list[dict[str, Any]] = []

    @property
    def index(self) -> int:
        return TRANSCRIPT_STATES.index(self.state)

    def reached(self, state: str) -> bool:
        return self.index >= TRANSCRIPT_STATES.index(state)

    def advance(self, state: str, **kw: Any) -> None:
        """Move to ``state`` (must be strictly later than the current one -- the transcript never goes back or repeats a stage)."""
        i = TRANSCRIPT_STATES.index(state)
        if i <= self.index:
            raise ValueError(f"transcript cannot go from {self.state!r} to {state!r} (states are monotone: no second stage-1, no re-commit)")
        if state in ("r_q_released", "r_s_released"):
            raise ValueError(f"{state!r} is reached only through release({state.split('_released')[0]!r})")
        self.state = state
        self.events.append({"event": state, **kw})

    def can_release(self, challenge: str) -> tuple[bool, str]:
        req = CHALLENGE_REQUIRES[challenge]
        released = f"{challenge}_released"
        if self.reached(released):
            return False, f"{challenge} already released (a second draw is a reroll)"
        return (True, req) if self.state == req else (False, req)

    def release(self, challenge: str, *, requested_by: str = "verifier", **kw: Any) -> None:
        """Record the release of ``challenge``; raise :class:`PrematureChallenge` when the prerequisite check state is not the current one."""
        ok, req = self.can_release(challenge)
        if not ok:
            e = PrematureChallenge(challenge, self.state, req, requested_by=requested_by)
            self.refusals.append(e.to_json())
            raise e
        self.state = f"{challenge}_released"
        self.events.append({"event": f"{challenge}_released", **kw})

    def refuse_request(self, challenge: str, *, requested_by: str) -> PrematureChallenge | None:
        """A prover-side request for ``challenge`` (handshake ``REQUEST_<challenge>``): returns the named refusal when premature, None when the
        challenge is releasable (the driver then proceeds through its normal path -- a request never triggers a draw by itself)."""
        ok, req = self.can_release(challenge)
        if ok:
            return None
        e = PrematureChallenge(challenge, self.state, req, requested_by=requested_by)
        self.refusals.append(e.to_json())
        return e

    def to_json(self) -> dict[str, Any]:
        return {"schema": "veritor/verifier-transcript/v1", "state": self.state, "states": list(TRANSCRIPT_STATES), "requires": dict(CHALLENGE_REQUIRES),
                "events": list(self.events), "refusals": list(self.refusals)}


def seed_custody_statement() -> dict[str, Any]:
    """[E9, handoff §Release gate: 'inspect the verifier's entropy ownership and exposure separately'] Who can read or alter the seed, and when --
    a reading of the code, not of timestamps.  Written into the record by the driver (``verify/seed_custody.json``)."""

    return {
        "schema": "veritor/seed-custody/v1",
        "generated": "partition_sample.stage1: seed = secrets.token_bytes(32) inside the VERIFIER process, immediately before r_q is derived; committed as H(SEED_DOMAIN, seed) "
                     "(chronology event seed_commitment) in the same call, before r_q",
        "held": "partition_sample._HELD_SEEDS[<chronology dir>] -- a module-level dict in the verifier process's memory (default since WR-2); stage2 reads it from there; "
                "never written to the record directory unless --allow-legacy-seed-file (labelled record_dir_file; refused for a claimed bound)",
        "revealed": "partition_sample.reveal: chronology event seed_reveal AFTER r_s exists (refused before); the seed is public from then on and every check re-derives J/Q from it",
        "can_read_before_reveal": [
            "any Python code executing in the verifier process: e2e.verifier_v3.{driver,steps}, e2e.partition_sample, e2e.security_sample, e2e.population_v3 (deriver), and the "
            "adapter-side READER modules the driver imports from the frozen source tree to authenticate captures (verity_vllm.capture, verity_vllm.commit, "
            "verity_vllm_adapter.{obligations,response_record}) -- repository code at the record's pinned revision, never code read from the record directory",
            "the operating system: root / a debugger with ptrace on the verifier host; a core dump or swap of the verifier process",
        ],
        "cannot_read_before_reveal": [
            "the prover subprocess (separate address space; it sees only the handshake directory, the record directory and its own files: A8 probe DRV-R7 scans every readable "
            "file for a 64-hex string opening seed_commitment and finds none)",
            "a party holding only the record directory (seed_commitment is H(domain, seed); no seed.hex is written)",
        ],
        "can_alter": [
            "the verifier process itself before seed_commitment (stage1's seed_hex parameter exists for tests; the driver never passes it) -- after seed_commitment an altered seed no "
            "longer opens the commitment and verify() names it",
            "nobody else: the prover's messages enter r_q / r_s only as digests (boundary digest; H(interior digest, H(F))) that are fixed before each draw",
        ],
        "deployment_requirement": "run the driver on a host the prover cannot read (R7); a co-located prover with root on the verifier host is outside the model",
        "evidence": "live: tests/contract/a8_conformance/driver_regressions.py DRV-R7 (probe) + DRV-EARLY-* (REQUEST_r_q / REQUEST_r_s refused by name); "
                    "unit: tests/contract/a3_sequence (seed never in the record dir; foreign process cannot continue); archived timestamps are NOT evidence",
    }


# -- [E9] V3.interior_layout / interior regions (brief interface 4) --------------------------------------------------------------------

INTERIOR_LAYOUT_FRAMING = "chunks-4096"
INTERIOR_LAYOUT_WORD_ORDER = "gate"
INTERIOR_TREE_PREFIX = "interior/"
INTERIOR_PRODUCER = "interior-interpreter"


def interior_layout_module():
    """E2's ``e2e/interior/layout.py`` (``layout_digest``, ``validate_supplied``) when it is on this checkout, else None."""

    try:
        from e2e.interior import layout as L  # noqa: N812

        return L
    except Exception:  # noqa: BLE001  -- absent module or a broken import: both are 'the verifier cannot validate' (fail closed below)
        return None


def interior_layout_pin(inst_doc: dict[str, Any]) -> dict[str, Any] | None:
    """The pinned ``protocol_instance.json#commitments.interior_layout`` (E6 pins; None when absent)."""

    c = inst_doc.get("commitments")
    if not isinstance(c, dict):
        return None
    p = c.get("interior_layout")
    return dict(p) if isinstance(p, dict) else None


def interior_trees_committed(c2: dict[str, Any] | None) -> list[str]:
    """The interior trees a commitments.json carries, as tree ids ``interior/<req>/<step>``: E2 records them under ``interior_roots["<req>/<step>"]``
    (``e2e.interior.tree.add_interior_roots``: ``trees`` is never rewritten); a record listing them under ``trees["interior/..."]`` counts too."""

    if not isinstance(c2, dict):
        return []
    ids = {str(t) for t in (c2.get("trees") or {}) if str(t).startswith(INTERIOR_TREE_PREFIX)}
    roots = c2.get("interior_roots")
    if isinstance(roots, dict):
        ids.update(f"{INTERIOR_TREE_PREFIX}{k}" for k in roots)
    return sorted(ids)


UNPINNED_INTERIOR_LOCATE_MARKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (exception class name, message fragments): the three spellings, on the merged tree of 2026-09-11, of ONE fact -- the VU reads or
    # yields an interior word that a record WITHOUT the interior_layout pin never committed.  D12 ruling: such a VU is 'uncovered' by name,
    # never a whole-Q refusal.
    ("InteriorLayoutUnpinned", ()),                                                   # E3's PositionOracle (locator.py PositionRefusal INTERIOR_LAYOUT_UNPINNED)
    ("", ("INTERIOR_LAYOUT_UNPINNED",)),                                              # the same code carried in any wrapper's message
    ("Refusal", ("not a committed word of the record", "interior value the record does not commit")),  # E7's ck_ftz / ck_elem locate-time refusal
    ("ValueError", ("expectation without a subcircuit digest",)),                     # E3's positions-only fallback for a refused builder: no body to pin
)


LEGACY_BUILDER_KINDS: frozenset[str] = frozenset({"SiluMulBf16_v1|batch", "Bf16AddF2fp_v1|batch", "RopeOut_v1|batch", "RopeOutAdd_v1|batch"})
"""The four boundary elementwise kinds that HAD a builder before 2026-09-11 (E7 ``elem_builders.BOUNDARY_KINDS``); with the GEMM ``provable``
and ``input`` classes they are the units a record made before today could cover.  Every other builder kind (E7's interior elem kinds, E4's FTZ
kinds, the select kinds) was 'uncovered' on such a record by name."""


def legacy_uncovered_kind(unit: dict[str, Any]) -> bool:
    """True when the located unit's kind had NO builder before today and is not a GEMM ``provable`` / ``input`` unit: on a record WITHOUT the
    interior_layout pin the D12 ruling says it must behave as before -- 'uncovered' by name -- whatever its new builder does."""

    cls = str(unit.get("class", ""))
    kind = str(unit.get("kind", ""))
    if kind in LEGACY_BUILDER_KINDS or cls.startswith("input"):
        return False
    if cls.startswith("provable") and "BUILDERS" not in cls:  # the GEMM coordinate units (A2's deriver), covered before today
        return False
    return True


def unpinned_interior_locate_reason(exc: BaseException) -> str | None:
    """``"<Class>: <message>"`` when a locate failure is one of ``UNPINNED_INTERIOR_LOCATE_MARKS`` (the VU is uncovered by name on an
    unpinned record); ``None`` for every other failure (which stays a refusal by name in ``expected_statements``).  Only the reverify
    pre-pass on a record WITHOUT the interior_layout pin consults this; a pinned record's oracle places its interior words."""

    name, msg = type(exc).__name__, str(exc)
    for cls, frags in UNPINNED_INTERIOR_LOCATE_MARKS:
        if (not cls or name == cls) and (not frags or any(f in msg for f in frags)):
            return f"{name}: {msg[:160]}"
    return None


_REFUSAL_TOKEN = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\b")


def unlocated_vu_refusal(vu_id: str, kind: str | None, exc: BaseException) -> tuple[str, str]:
    """[E9, D48.2 2026-09-12] ``(name, line)`` for a sampled VU whose expectation the verifier could not derive: ``name`` is the NAMED
    refusal token in the failure's message (``LITERAL_UNPINNED``, ``INTERIOR_LEAF_UNREADABLE``, ... -- the first ``UPPER_SNAKE`` token; the
    exception class when the message carries none), ``line`` the coverage-line spelling ``<NAME>: VU <id> <kind>``.  The VU is accounted
    ``unlocated`` by that name; the record is REFUSED (V5.expected / V5.accounting fail) -- never a whole-Q void (E10's q2: one
    ``Bf16Add_v1|batch`` VU read ``expected=0, Q_proof=707``)."""

    msg = str(exc)
    m = _REFUSAL_TOKEN.search(msg)
    name = m.group(1) if m else type(exc).__name__
    return name, f"{name}: VU {vu_id}{(' ' + str(kind)) if kind else ''}"


def interior_kinds_in_population(pop_doc: dict[str, Any] | None) -> set[str]:
    """Kinds the population's kind table marks ``producer == "interior-interpreter"`` (E6's field, new today).  Empty for a legacy table."""

    if not isinstance(pop_doc, dict):
        return set()
    k = pop_doc.get("kinds")
    entries = k.get("entries") if isinstance(k, dict) else None
    out: set[str] = set()
    for e in entries or []:
        if isinstance(e, dict) and str(e.get("producer", "")) == INTERIOR_PRODUCER:
            out.add(f"{e.get('fn')}|{e.get('form')}")
    return out


def interior_units_selected(units: list[dict[str, Any]], *, interior_kinds: set[str] | None = None, fallback_uncovered: list[int] | None = None) -> dict[str, Any]:
    """Which SELECTED (retained) units read or write interior values, by the best verifier-side source available:
    (1) E3's located unit says so (``locate.origin == "interior"`` or a ``locate.tree`` under ``interior/``) -- the verifier's own derivation;
    (2) the population's kind table marks the kind ``producer == interior-interpreter`` (E6);
    (3) FALLBACK, labelled: the unit is in the coverage layer's ``uncovered`` set (no checker / builder over committed words) -- today every such
        kind reads operator-interior words (attempt 6's 288); over-approximates on purpose (fail closed)."""

    by_source: dict[str, list[int]] = {"located": [], "kind_table": [], "fallback_uncovered": []}
    kinds: dict[str, int] = {}
    fb = set(fallback_uncovered or [])
    for u in units:
        vu = int(u["vu_index"])
        L = u.get("locate") if isinstance(u.get("locate"), dict) else {}
        kind = str(u.get("kind"))
        if str(L.get("origin", "")) == "interior" or str(L.get("tree", "")).startswith(INTERIOR_TREE_PREFIX):
            by_source["located"].append(vu)
        elif interior_kinds and kind in interior_kinds:
            by_source["kind_table"].append(vu)
        elif vu in fb:
            by_source["fallback_uncovered"].append(vu)
        else:
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
    n = sum(len(v) for v in by_source.values())
    src = "located(E3)" if by_source["located"] else "kind_table(E6 producer)" if by_source["kind_table"] else "fallback(uncovered-not-boundary)" if by_source["fallback_uncovered"] else "none"
    return {"count": n, "by_source": {k: len(v) for k, v in by_source.items()}, "kinds": dict(sorted(kinds.items())), "classification": src}


def required_layouts_from_positions(positions: Any, L: Any, pin: dict[str, Any] | None = None) -> Any:
    """E9's ``required=`` source for E2's ``validate_supplied`` step-4 form: ``(req, step) -> LazyInteriorLayout | None`` built from E3's
    ``PositionOracle`` (``coverage_inputs(..., population=, interior_layout=)["positions"]``).  E3 DERIVES the required interior gates and their
    widths per root call from program + partition + its boundary oracle (``GateResolver.region_calls`` / ``call_layout``: the VU output gates of
    the call without a boundary / Input position, ascending); E2 FRAMES them under the pinned rule (``InteriorLayout.build``).  None when E3's
    oracle has no region for that tree (the RU is not a retained interior RU) -- E2 then reports the RU as not checked against required and
    ``interior_layout_check`` refuses (fail closed)."""

    import numpy as np

    chunk_words = L._chunk_words_of(pin) if pin is not None else L.CHUNK_WORDS  # noqa: SLF001  -- the pin's framing names the chunk size

    def required(req: str, step: str) -> Any:
        res = positions.resolver(str(req))
        regs = getattr(res, "regions", None)
        tid = L.tree_id(str(req), str(step))
        region = regs.by_tree(tid) if regs is not None and hasattr(regs, "by_tree") else None
        if region is None:
            return None
        calls, by_call = [], {}
        for c in res.region_calls(region):
            calls.append(L.CallSpan(str(c[0]), int(c[1]), int(c[2])))
            by_call[calls[-1].call] = c

        def call_gw(span: Any) -> tuple[Any, Any]:  # E2's LazyInteriorLayout asks per call: (required gates ascending, their widths)
            c = by_call[span.call]
            gates, chunk, _flat, sizes = res.call_layout(c)
            if not len(gates):
                return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
            # E3's call_layout sizes are (n, width) under v1 and (n, width, k) under v2 -- the width is position 1 in both (integrator on E9's
            # behalf 23:38Z; kept by E9: any other shape is refused by name rather than read as a width)
            bad = [s for s in sizes if not (isinstance(s, (tuple, list)) and len(s) in (2, 3))]
            if bad:
                raise ValueError(f"E3 call_layout sizes: expected (n, width) or (n, width, k) tuples, got {bad[0]!r} (call {c!r})")
            return np.asarray(gates, dtype=np.int64), np.asarray([int(s[1]) for s in sizes], dtype=np.int64)[np.asarray(chunk, dtype=np.int64)]

        # [NIGHT D38, E2 on E9's behalf] the required layout is materialised ONE CALL AT A TIME (memory ~ the largest call, not the RU:
        # 11.35 GB -> 0.58 GB at the 128x64 prefill RU, MEASURED) -- same ids / ranks / positions as the eager InteriorLayout.build
        return L.LazyInteriorLayout(tid, calls, call_gw, prefix=str(getattr(res, "prefix", "") or ""), chunk_words=chunk_words)

    return required


def required_source_live(inst_doc: dict[str, Any], pop_doc: dict[str, Any] | None, sel: Path, c2: dict[str, Any], retained_request_ids: list[str],
                         *, family: str | None = None) -> tuple[Any, str, dict[str, Any]]:
    """The live driver's ``required=`` source at step 4 (BEFORE r_s): the VERIFIER builds the pinned program in its own process
    (``locator.build_pinned_program``: identity from the instance, digest == the population's pin), its locator, and E3's ``PositionOracle`` over
    the retained RUs' committed id lists (``expected.coverage_inputs(..., population=, interior_layout=)``; ``units=[]``: nothing of Q exists yet),
    then :func:`required_layouts_from_positions`.  Returns ``(source | None, label, facts)``; a None source with its reason is what
    :func:`interior_layout_check` refuses on (fail closed) whenever a retained RU has an interior tree."""

    import inspect
    import time

    facts: dict[str, Any] = {"seconds": None, "program": None, "positions": False}
    t0 = time.perf_counter()
    L = interior_layout_module()
    if L is None:
        return None, "NONE: e2e.interior.layout (E2) not importable", facts
    if not isinstance(pop_doc, dict) or not pop_doc.get("program_digest"):
        return None, "NONE: no population document / program_digest to build the pinned program against", facts
    try:
        from e2e.verifier_v3 import expected as ex
        from e2e.verifier_v3 import locator as vloc
    except Exception as e:  # noqa: BLE001
        return None, f"NONE: verifier_v3 expected/locator not importable: {type(e).__name__}", facts
    if "population" not in inspect.signature(ex.coverage_inputs).parameters:
        return None, "NONE: E3's PositionOracle is not on this checkout (coverage_inputs has no population= / interior_layout=)", facts
    try:
        prog_identity = (inst_doc.get("program") or {}).get("identity") or {}
        program, prog_C, prog_problems = vloc.build_pinned_program(prog_identity, expected_digest=pop_doc["program_digest"],
                                                                   expected_gates=((inst_doc.get("program") or {}).get("gates") or {}).get("value"),
                                                                   builder=(inst_doc.get("program") or {}).get("builder"))  # D17: registered programs build by their pinned builder
        facts["program"] = {"identity": prog_identity.get("function"), "problems": list(prog_problems or [])}
        facts["_program"] = program  # [NIGHT G1] handed to V3.literal_table (the caller pops it before serializing; one build per step 4)
        if program is None or prog_problems:
            return None, f"NONE: the pinned program did not build in the verifier's process: {list(prog_problems or [])[:2]}", facts
        ploc = vloc.make_locator(program, prog_C)
        fam = family or ((inst_doc.get("epoch") or {}).get("family")) or "veritor.tc-ampere-bf16@2"
        ci = ex.coverage_inputs(sel, c2, retained_request_ids=set(retained_request_ids), units=[], checks=[], family=fam, recompute_leaves=True, config=prog_C,
                                program=program, locator=ploc, population=pop_doc, interior_layout=interior_layout_pin(inst_doc))
        facts["tree_problems"] = list(ci.get("tree_problems") or [])[:3]
        if ci.get("positions") is None:
            return None, "NONE: coverage_inputs built no PositionOracle", facts
        facts["positions"] = True
        src = required_layouts_from_positions(ci["positions"], L, interior_layout_pin(inst_doc))
        return src, "live: verifier-built pinned program + E3 PositionOracle -> E2 InteriorLayout.build", facts
    except Exception as e:  # noqa: BLE001
        return None, f"NONE: required source failed: {type(e).__name__}: {str(e)[:160]}", facts
    finally:
        facts["seconds"] = round(time.perf_counter() - t0, 3)


def interior_layout_check(inst_doc: dict[str, Any], *, sel: Path | None, c2: dict[str, Any] | None, retained_request_ids: list[str],
                          interior_in_play: bool, in_play_reason: str, required_source: Any = None, required_label: str = "") -> dict[str, Any]:
    """``V3.interior_layout`` + the required-region presence check, as one pure result::

        {"ok": bool, "status": "PASS" | "FAIL" | "VACUOUS" | "UNAVAILABLE", "problems": [...], "pin": {...} | None, "digest": str | None,
         "layout_digest": str | None, "trees": [...], "regions": {...}, "interior_in_play": bool, "reason": str, "validator": str}

    Rules (brief interface 4; fail closed):
      * pin present  -> E2's ``layout_digest(pin)`` must equal ``pin["digest"]`` and the pin's framing / word order must be the frozen ones; E2's
        ``validate_supplied(sel, request_ids, pin, c2)`` must return no problem for every RETAINED RU; the interior trees named in ``c2`` must
        carry ``interior_roots``; E2's module absent -> UNAVAILABLE (refused: a pinned layout the verifier cannot validate);
      * pin absent + interior in play -> FAIL (the values Q may select have no committed positions);
      * pin absent + nothing interior in play -> VACUOUS pass, labelled with the reason.
    ``interior_in_play`` = at step 4: the population declares interior-producer kinds or ``c2`` carries interior trees; at step 5: an interior
    unit is in Q (``interior_units_selected``)."""

    pin = interior_layout_pin(inst_doc)
    trees = interior_trees_committed(c2)
    roots = (c2 or {}).get("interior_roots")
    out: dict[str, Any] = {"ok": False, "status": "FAIL", "problems": [], "pin": pin, "digest": (pin or {}).get("digest"), "layout_digest": None, "trees": trees,
                           "regions": {}, "interior_in_play": bool(interior_in_play), "reason": in_play_reason, "validator": None,
                           "required_source": required_label or ("E3 positions -> E2 InteriorLayout" if required_source is not None else "NONE")}
    L = interior_layout_module()
    out["validator"] = ("e2e.interior.layout (E2)" if L is not None else "UNAVAILABLE: e2e.interior.layout not importable on this checkout")
    if pin is None:
        if interior_in_play or trees or roots:
            out["problems"].append("no protocol_instance.json#commitments.interior_layout pin while interior values are in play (" + in_play_reason +
                                   (f"; interior trees committed: {trees[:3]}" if trees else "") + "): the positions Q may select are not pinned -> refused (fail closed)")
            return out
        out.update(ok=True, status="VACUOUS", problems=[])
        out["note"] = "no interior_layout pin and no interior value in play (" + in_play_reason + "): vacuous"
        return out
    # the pin's frozen shape
    for k, want in (("framing", INTERIOR_LAYOUT_FRAMING), ("word_order", INTERIOR_LAYOUT_WORD_ORDER)):
        if str(pin.get(k)) != want:
            out["problems"].append(f"interior_layout.{k} = {pin.get(k)!r}, the frozen interface says {want!r}")
    if not str(pin.get("tree", "")).startswith(INTERIOR_TREE_PREFIX):
        out["problems"].append(f"interior_layout.tree = {pin.get('tree')!r} does not name the interior tree family ({INTERIOR_TREE_PREFIX}<req>/<step>)")
    if not pin.get("digest"):
        out["problems"].append("interior_layout.digest missing")
    if L is None:
        out.update(status="UNAVAILABLE")
        out["problems"].append("the instance pins an interior layout but E2's validator (e2e.interior.layout: layout_digest / validate_supplied) is not on this checkout: "
                               "the verifier cannot validate the pinned layout -> refused (fail closed)")
        return out
    # E2's digest of the pinned layout
    try:
        ld = L.layout_digest(pin)
        out["layout_digest"] = str(ld)
        if str(ld) != str(pin.get("digest")):
            out["problems"].append(f"interior_layout.digest {str(pin.get('digest'))[:12]} != E2 layout_digest() {str(ld)[:12]} (the pinned layout is not the one the verifier derives)")
    except Exception as e:  # noqa: BLE001
        out["problems"].append(f"E2 layout_digest(pin) failed: {type(e).__name__}: {str(e)[:160]} (interface mismatch or invalid pin) -> refused")
    # every REQUIRED interior region of every retained RU is supplied: E2's validate_supplied (step-4 form over the selective bundle) with
    # required= E3's derived layouts (required_layouts_from_positions).  Without a required source E2 checks presence + rule-consistency only
    # and says so (required_source NONE): that is NOT the brief's check -> refused whenever a retained RU has an interior tree to check.
    out["required_source"] = required_label or ("E3 positions -> E2 InteriorLayout" if required_source is not None else "NONE")
    if sel is not None and retained_request_ids:
        try:
            # D17: E2's step-4 entry point by name (validate_supplied's 2nd positional is itself named `required` -> keyword collision)
            probs = L.validate_step4(Path(sel), list(retained_request_ids), pin, c2 or {}, required=required_source, rehash=True)
            if isinstance(probs, dict):  # E2's step-4 form: {"problems": [...], "retained_rus": [...], "rus": {ru: facts}, "required_source": str}
                out["regions"] = {k: v for k, v in probs.items() if k != "problems"}
                probs = list(probs.get("problems") or [])
            out["problems"].extend(f"interior region: {p}" for p in list(probs or []))
            rus = (out["regions"].get("rus") or {}) if isinstance(out["regions"], dict) else {}
            unchecked = sorted(ru for ru, facts in rus.items() if isinstance(facts, dict) and not facts.get("checked_against_required"))
            if unchecked:
                out["problems"].append(f"REQUIRED regions not derived for retained RU(s) {[u[:8] + '/' + u.split('/')[-1] for u in unchecked][:4]}"
                                       f"{' ...' if len(unchecked) > 4 else ''} ({out['required_source']}): the presence of every REQUIRED interior region is not "
                                       "established -> refused (fail closed; E3's deriver supplies the required layouts)")
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"E2 validate_supplied failed: {type(e).__name__}: {str(e)[:200]} -> refused (a region set the verifier cannot validate is missing)")
    elif retained_request_ids:
        out["problems"].append("no selective directory to validate the interior regions against")
    if not isinstance(roots, dict) and trees:
        out["problems"].append("commitments.json names interior trees but carries no interior_roots (one root per retained RU)")
    out["ok"] = not out["problems"]
    out["status"] = "PASS" if out["ok"] else out["status"]
    return out


# -- step 2 -------------------------------------------------------------------------------------------------------------------------


def step2_check(inst, c0: dict[str, Any], production_rows: list[dict[str, Any]], *, registered_weights_root: str | None,
                boundary_outputs: dict[str, list[int]] | None = None, boundary_inputs: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Refusals of the contract's step-2 check.  ``c0`` = ``commitments_phase0.json``; ``production_rows`` = served ``production.jsonl``;
    ``boundary_outputs`` = ``{request_id: output_token_ids}`` as read back from the prover's committed boundary records (None: the
    ``v_out(C) == y`` comparison is reported as not performed)."""

    p: list[str] = []
    trees = c0.get("trees") or {}
    served = {r["request_id"]: r for r in production_rows}
    if len(served) != len(production_rows):
        p.append("production.jsonl names a request twice")
    if inst.epoch is None:
        p.append("instance declares no epoch (legacy instance): the contract verifier needs the epoch's exact request set to enforce the barrier (R16)")
        declared = set(served)
    else:
        declared = set(inst.epoch.request_ids)
        if set(served) != declared:
            p.append(f"served requests {sorted(set(served) ^ declared)} differ from the epoch's declared request set (symmetric difference)")
    if len(served) != inst.partition.instances:
        p.append(f"{len(served)} requests served but partition.instances = {inst.partition.instances}")
    # every RU's output gates committed: one boundary tree per declared request
    for rid in sorted(declared):
        t = trees.get(f"boundary/{rid}")
        if t is None:
            p.append(f"no boundary commitment for request {rid} (step 1 incomplete: r_q cannot be derived)")
        elif t.get("kind") != "boundary" or not t.get("root") or int(t.get("leaf_count", 0)) <= 0:
            p.append(f"boundary tree of {rid} malformed")
    extra = sorted(k.split("/", 1)[1] for k in trees if k.startswith("boundary/") and k.split("/", 1)[1] not in declared)
    if extra:
        p.append(f"boundary commitments for undeclared requests {extra}")
    if any(v.get("kind") == "ru" for v in trees.values()):
        p.append("phase-0 commitment already holds RU interior trees (step-3 values before r_q: reordered message, refused)")
    # the weights (prescribed inputs: reuse of the registered commitment) -- refused here, not merely recorded (R13 (iii))
    w = trees.get("weights")
    if w is None:
        p.append("no weights tree in the phase-0 commitment")
    elif registered_weights_root is not None and w.get("root") != registered_weights_root:
        p.append(f"weights root {str(w.get('root'))[:12]} != registered {registered_weights_root[:12]} (ROT-INPUT)")
    # y well-typed: declared output multiplicity and domain per program instance
    mult = sum(int(d.multiplicity) for d in inst.outputs.designated)
    bits = max((int(d.domain_bits_declared) for d in inst.outputs.designated), default=0)
    for rid, row in sorted(served.items()):
        y = list(row.get("output_token_ids") or [])
        if len(y) != mult:
            p.append(f"request {rid}: {len(y)} output slots, declared multiplicity {mult}")
        if any(not (0 <= int(x) < 2 ** bits) for x in y):
            p.append(f"request {rid}: an output token id is outside the declared {bits}-bit domain")
        if not row.get("prompt_token_ids"):
            p.append(f"request {rid}: no prompt token ids (input gates of C uncommitted)")
        # [A10 WR-5, R1 v_in(C) = x] the COMMITTED prompt / seed / sampling params equal the DECLARED input (pinned in the instance before the
        # first commitment): compared here, before r_q, against the boundary record authenticated by its committed root (boundary_inputs =
        # e2e.verifier_v3.driver.boundary_inputs: {rid: {root_ok, prompt_token_ids, seed, sampling_params}})
        decl = (inst.epoch.inputs.per_request.get(rid) if (inst.epoch is not None and inst.epoch.inputs is not None) else None)
        if inst.epoch is not None and inst.epoch.inputs is None:
            # [user directive 20:10Z] input agreement is UNCONDITIONAL for a contract instance: an epoch that pins no inputs cannot be audited
            # (the Input VUs have left the sampled population, so nothing else establishes v_in(C) = x)
            p.append(f"request {rid}: the instance declares an epoch but pins no inputs (epoch.inputs) -- v_in(C) = x cannot be established")
        elif inst.epoch is not None and decl is None:
            p.append(f"request {rid}: no declared input in the instance (epoch.inputs)")
        if decl is not None:
            if pi.declared_input_digest(list(row.get("prompt_token_ids") or [])) != decl.prompt_sha256 or len(row.get("prompt_token_ids") or []) != decl.prompt_len:
                p.append(f"request {rid}: served prompt != declared x (prompt digest)")
            if boundary_inputs is None:
                p.append(f"request {rid}: no committed boundary record readable for the declared == committed input comparison")
            else:
                bi = boundary_inputs.get(rid)
                if bi is None:
                    p.append(f"request {rid}: no committed boundary record for the declared == committed input comparison")
                else:
                    if not bi.get("root_ok", False):
                        p.append(f"request {rid}: the boundary leaves do not authenticate against the committed phase-0 root (its prompt cannot be compared)")
                    if not bi.get("prompt_contiguous", True):
                        p.append(f"request {rid}: committed prompt_token leaves are not exactly t = 0..n-1 (length / ordering of the committed prompt not established)")
                    ids = list(bi.get("prompt_token_ids") or [])
                    # sha256 over the canonical JSON list covers the values, their count and their order; prompt_len is checked separately so a
                    # length disagreement is named as such
                    if len(ids) != decl.prompt_len:
                        p.append(f"request {rid}: committed prompt has {len(ids)} tokens, declared {decl.prompt_len}")
                    if pi.declared_input_digest(ids) != decl.prompt_sha256:
                        p.append(f"request {rid}: committed prompt != declared x (v_in(C) != x)")
                    # [R1, integrator 21:00Z] C(LP) has exactly LP prompt Input gates: the committed prompt must have LP positions
                    LP = inst.program.identity.LP
                    if isinstance(LP, int) and len(ids) != LP:
                        p.append(f"request {rid}: committed prompt has {len(ids)} tokens, the program C(LP={LP}) has {LP} prompt Input gates")
                    # [A2 r1-input-layout.md §4] gate-by-gate layout check of the authenticated leaves against the pinned layout: count, scalar,
                    # width, no committed prompt position the layout does not name.  Values were compared above by digest (values + order).
                    lay = inst.epoch.inputs.layout
                    if lay is not None:
                        from veritor.interface.input_layout import CommittedInputLeaf, check_input_layout, prompt_layout

                        leaves = {int(t): v for t, v in (bi.get("prompt_leaves") or {}).items()}

                        def leaf_of(field: str, t: int, _lv=leaves):
                            v = _lv.get(int(t)) if field == "prompt_token" else None
                            return CommittedInputLeaf(v["dtype"], tuple(v["shape"]), bytes.fromhex(v["hex"])) if v else None

                        lp = check_input_layout(prompt_layout(int(lay.prompt.count), width=int(lay.prompt.width), field=lay.prompt.field), lay.digest,
                                                {"prompt_token": ids}, leaf_of, committed_offsets={"prompt_token": sorted(leaves)})
                        p.extend(f"request {rid}: input layout: {x}" for x in lp)
                    if decl.seed is not None and bi.get("seed") != decl.seed:
                        p.append(f"request {rid}: committed seed {bi.get('seed')!r} != declared {decl.seed!r}")
                    if decl.sampling_params_sha256 is not None and pi.sampling_params_digest(dict(bi.get("sampling_params") or {})) != decl.sampling_params_sha256:
                        p.append(f"request {rid}: committed sampling parameters != declared")
        # v_out(C) == y: the served outputs equal the committed boundary record's
        if boundary_outputs is not None:
            b = boundary_outputs.get(rid)
            if b is None:
                p.append(f"request {rid}: no committed boundary record readable for the served == committed comparison")
            elif list(b) != y:
                p.append(f"request {rid}: served output tokens != committed boundary record (v_out(C) != y)")
    return p


def population_derived(inst_doc: dict[str, Any], population: Path | dict[str, Any], *, instances: int | None, self_check: str = "fast", rows_self_checked_by: str | None = None) -> dict[str, Any]:
    """[user directive 16:04 PT; integrator 23:18Z item 1] ``V2.population_derived``: the verifier DERIVES the expected population from the pinned
    instance (program identity + partition plan: kind names, declared widths, per-kind counts, N_V, N_R, the kind-table identity) through A1b's
    ``e2e.population_v3.population_derived_check`` and compares the population file against it -- matching digests are necessary but not
    sufficient (two components hashing the same prover-supplied file agree on the wrong thing).  Result: ``{"ok": bool, "status", "problems":
    [field-named disagreements], "derived": [...], "compared_only": [...], "expected": ..., "source"}``.  Fails CLOSED when the deriver is not
    on this checkout (``status = "UNAVAILABLE"``): a population the verifier cannot derive is not accepted.

    Digests that are NOT derivable by rule (``partition-intervals/v1`` / ``a9/legacy-minus-inputs/v1`` partition digest, ``content_sha256``,
    ``plan_digest``, ``gates``) are required to re-derive from the file's OWN rows (``self_check="full"`` here, or -- in the live driver --
    ``partition_sample.stage1``'s ``load_population(check="full")`` before r_q, named in ``rows_self_checked_by``) AND the rows must carry the
    derived kind table; both facts are written into the result (``compared_only`` + ``rows_self_check``)."""

    source = "e2e.population_v3.population_derived_check (A1b)"
    rows_note = ("population_v3.compare(self_check='full'): the file's row-level digests re-derived from its own rows" if self_check == "full"
                 else (rows_self_checked_by or "fast: the file's row-level digests are its own claim here"))

    def early(status: str, problems: list[str]) -> dict[str, Any]:
        # [W-A9-13] every early exit carries the SAME keys as a full result (the driver / reverify index them unconditionally; attempt 4 of A9's
        # elem record crashed on KeyError 'rows_self_check' after the deriver hit a missing census file on the pod)
        return {"ok": False, "status": status, "problems": problems, "derived": [], "compared_only": [], "rows_self_check": rows_note, "expected": None, "source": source}

    try:
        from e2e import population_v3 as v3
    except Exception as e:  # noqa: BLE001
        return early("UNAVAILABLE", [f"e2e.population_v3 not importable: {type(e).__name__}: {str(e)[:120]}"])
    fn = getattr(v3, "population_derived_check", None)
    if fn is None:
        return early("UNAVAILABLE", ["e2e.population_v3.population_derived_check is not on this checkout (A1b, data/contract/a1-circuit/for-a3.md): "
                                     "the verifier cannot derive the expected population -> not accepted"])
    try:
        # [E9 SCALE 23:10Z, read-once] the caller's already-parsed document is handed through (population_v3.compare takes a dict); a Path is read here
        r = fn(inst_doc, population if isinstance(population, dict) else Path(population), instances=instances, self_check=self_check)
    except SystemExit as e:  # the deriver refuses the instance (no rules for the case / plan) or its pinned rules file (sha mismatch)
        return early("REFUSED_BY_DERIVER", [f"expected_population refused: {str(e)[:200]}"])
    except FileNotFoundError as e:  # the verifier host lacks the deriver's pinned inputs (census rules / validated holdout) -- a deployment fault, named
        return early("ERROR", [f"the deriver's pinned input is missing on the verifier host: {e} (see for-a9.md s11: data/census/census_small_6500x1600.json "
                               f"sha 133934a8..., data/rec/p1-b0-lp32s15-validated.json)"])
    except Exception as e:  # noqa: BLE001  -- an unreadable file is a refusal by name, never a traceback
        return early("ERROR", [f"{type(e).__name__}: {str(e)[:200]}"])
    dis = list(r.get("disagreements") or [])
    problems = []
    for d in dis:
        f = d.get("field") if isinstance(d, dict) else getattr(d, "field", None)
        exp_ = d.get("expected") if isinstance(d, dict) else getattr(d, "expected", None)
        act = d.get("actual") if isinstance(d, dict) else getattr(d, "actual", None)
        note = d.get("note") if isinstance(d, dict) else getattr(d, "note", "")
        problems.append(f"{f}: expected {str(exp_)[:60]!s} != file {str(act)[:60]!s} ({note})")
    ok = str(r.get("status")) == "matches" and not problems
    return {"ok": ok, "status": str(r.get("status")), "problems": problems, "derived": list(r.get("derived") or []), "compared_only": list(r.get("compared_only") or []),
            "rows_self_check": rows_note, "expected": r.get("expected"), "source": source}


#: [W-A9-13] the deriver's early-exit statuses: the verifier host could not derive (not a prover disagreement); still a refusal before r_q
POPULATION_DERIVER_EARLY = ("UNAVAILABLE", "REFUSED_BY_DERIVER", "ERROR")
POPULATION_DERIVED_KEYS = ("ok", "status", "problems", "derived", "compared_only", "rows_self_check", "source")


def ru_map_from_instance(inst) -> dict[int, str] | None:
    """RU index -> request id from the INSTANCE (epoch.request_ids sorted; RU r of program instance i = i * ru_count + r); None for a
    legacy instance without an epoch.  This is the only source of the request identity of a selected RU in verifier_v3."""

    if inst.epoch is None:
        return None
    n_r = int(inst.partition.ru_count.value)
    out: dict[int, str] = {}
    for i, rid in enumerate(sorted(inst.epoch.request_ids)):
        for r in range(n_r):
            out[i * n_r + r] = rid
    return out


def declared_output_domain(inst) -> tuple[int, int, int]:
    """[WR-6] ``(slots, domain_bits, W_out_bits)`` per program instance from the pinned instance's designated outputs -- never constants."""

    slots = sum(int(d.multiplicity) for d in inst.outputs.designated)
    bits = max((int(d.domain_bits_declared) for d in inst.outputs.designated), default=0)
    return slots, bits, int(inst.outputs.W_out_bits.value)


def program_output_slots(program) -> int | None:
    """The number of root output slots of a built ``verity.ir.Program`` (Serve_v1: STEPS + 1 token slots), None when not derivable."""

    try:
        return len(program.circuit.root_outputs())  # verity.ir.layout.Circuit: the root call's output gates (Serve_v1: one token slot each)
    except Exception:  # noqa: BLE001
        return None


def steps_outputs_relation(inst_doc: dict[str, Any], *, program=None) -> dict[str, Any]:
    """[R13] ``outputs.designated`` multiplicity (explicit) vs ``program.identity.STEPS`` (explicit): Serve_v1 emits the prompt-position
    logits slot plus one slot per decode step, so multiplicity must equal STEPS + 1; when the verifier-built program is given its root
    output slot count must agree too.  Returns the three numbers and the problems (empty = relation holds)."""

    ident = (inst_doc.get("program") or {}).get("identity") or {}
    steps_n = ident.get("STEPS")
    mult = sum(int(d["multiplicity"]) for d in (inst_doc.get("outputs") or {}).get("designated") or [])
    problems: list[str] = []
    if not isinstance(steps_n, int):
        problems.append("program.identity.STEPS is not an explicit integer")
    elif mult != steps_n + 1:
        problems.append(f"outputs.designated multiplicity {mult} != STEPS + 1 = {steps_n + 1} (prompt-position logits + one slot per decode step)")
    slots = program_output_slots(program) if program is not None else None
    if program is not None and slots is None:
        problems.append("the built program's output slot count is not derivable")
    elif slots is not None and slots != mult:
        problems.append(f"the built program has {slots} root output slots, the instance declares {mult}")
    return {"STEPS": steps_n, "multiplicity": mult, "program_slots": slots, "problems": problems}


def guest_acceptance(approved: dict[str, Any], info: dict[str, Any], *, pinned_elfs: list[str] | None, pinned_family: str | None) -> list[str]:
    """[R13, user directive 20:10Z] Whether the host's guest (``info``: elf_sha256, vk_hash) may check this audit.  Refusals: not an entry of
    the VERIFIER's ``approved_guests.json``; vk differs from the entry's; a pinned ``epoch.approved_elf_sha256`` naming a sha that is not an
    approved entry (the instance may narrow, never widen); the guest outside the pinned set; the entry not listing ``epoch.family`` among
    the families its checker implements (an entry without ``families`` never serves a family-pinned audit)."""

    p: list[str] = []
    g = {x["elf_sha256"]: x for x in approved.get("guests", [])}
    entry = g.get(info.get("elf_sha256"))
    if entry is None:
        p.append(f"host guest {str(info.get('elf_sha256'))[:12]} is not an entry of the verifier's approved_guests.json")
    else:
        if str(entry.get("vk_hash", "")).lower() != str(info.get("vk_hash", "")).lower():
            p.append("host vk != the approved entry's vk")
        if not entry.get("approval"):
            p.append("approved entry carries no approval record")
    for x in pinned_elfs or []:
        if x not in g:
            p.append(f"epoch.approved_elf_sha256 names {x[:12]}, which is not an approved entry (the instance narrows the set, never widens it)")
    if pinned_elfs and info.get("elf_sha256") not in set(pinned_elfs):
        p.append("host guest lies outside the instance's pinned approved set")
    if pinned_family is not None and entry is not None:
        # [A2 R10 / integrator 21:00Z] the tie is by family id AND gate-set digest: the entry's `families` = {id: digest} (family_table over the
        # guest's source tree) must name epoch.family with the digest this interface pins for it at width 32 -- neither side can rename a family
        from veritor.interface.families import check_family_approved
        from veritor.interface.typed_obligation import family_gate_set

        gs = family_gate_set(pinned_family, 32)
        if gs is None:
            p.append(f"epoch.family {pinned_family!r} does not resolve in this interface at width 32")
        else:
            p.extend(check_family_approved(pinned_family, str(gs.digest), 32, entry))
    return p


def output_domain_check(inst, served: dict[str, list[int]], *, n_requests: int) -> list[str]:
    """V7: every served request has the declared number of output slots and every token id lies in the declared domain; W_out is consistent."""

    slots, bits, w_out = declared_output_domain(inst)
    p: list[str] = []
    if len(served) != n_requests:
        p.append(f"{len(served)} served requests, {n_requests} expected")
    if w_out != sum(int(d.multiplicity) * int(d.domain_bits_declared) for d in inst.outputs.designated):
        p.append(f"instance W_out_bits {w_out} != sum multiplicity x declared bits")
    for rid, t in sorted(served.items()):
        if len(t) != slots:
            p.append(f"request {rid}: {len(t)} output slots, declared {slots}")
        if any(not (0 <= int(x) < 2 ** bits) for x in t):
            p.append(f"request {rid}: an output token id lies outside the declared {bits}-bit domain")
    return p


# -- [E9, INTEGRATOR D1] a registered verity.ir program (protocol_instance.json#program.kind == "registered") ------------------------------

NOT_APPLICABLE = "NOT_APPLICABLE"


def program_kind(inst_doc: dict[str, Any]) -> str:
    """``"registered"`` (a generic registered ``verity.ir`` program: E8's CORE) or ``"serve"`` (Serve_v1 / vLLM; the default for an instance without the pin)."""

    k = (inst_doc.get("program") or {}).get("kind")
    return "registered" if str(k) == "registered" else "serve"


def not_applicable(check: str, reason: str, *, replaced_by: str | None = None) -> dict[str, Any]:
    """One chronology / protocol.json entry for a vLLM-only step the driver skips BY NAME on a registered program (D1: nothing silently skipped)."""

    return {"event": f"{NOT_APPLICABLE}:{check}", "check": check, "reason": reason, "replaced_by": replaced_by, "label": "NOT_APPLICABLE"}


def ru_ids_of(inst, program_kind: str) -> tuple[list[str], dict[str, str], str]:
    """The record's RU ids in RU order, ``{ru_id: request_id}`` and the rule, from the pinned epoch.

    Serve_v1 (and any instance whose epoch declares one RU per request): RU r = position of the request id in ``sorted(epoch.request_ids)``,
    RU id == request id (P1/cu=whole).  A REGISTERED program whose epoch declares more RUs (``epoch.ru_indices``) than requests is one program
    instance with STEPS stage calls (E8's CORE: request ``core``, RUs ``core-ru<r>``): RU id = ``<request>-ru<r>`` -- the spelling E8's prover
    commits (``boundary/<ru_id>``, ``ru/<ru_id>/...``, ``interior_roots['<ru_id>/<step>']``) and the SHIPPED ``committed_rus`` maps back by id.
    Several requests x several stages is not defined here and is refused by name."""

    reqs = sorted(inst.epoch.request_ids)
    ru_idx = [int(r) for r in (getattr(inst.epoch, "ru_indices", None) or [])]
    if program_kind == "registered" and ru_idx and len(ru_idx) != len(reqs):
        if len(reqs) != 1:
            raise ValueError(f"epoch declares {len(reqs)} requests and {len(ru_idx)} RUs: several program instances x several stage RUs has no RU-id rule here (refused by name)")
        if ru_idx != list(range(len(ru_idx))):
            raise ValueError(f"epoch.ru_indices {ru_idx[:6]}... are not 0..n-1: the stage-RU rule needs contiguous stage indices")
        ids = [f"{reqs[0]}-ru{r}" for r in ru_idx]
        return ids, {i: reqs[0] for i in ids}, f"RU r = stage call r of the one program instance (request {reqs[0]!r}); RU id <request>-ru<r>; declared by epoch.ru_indices"
    return list(reqs), {r: r for r in reqs}, ("RU index r = position of the request id in sorted(epoch.request_ids) (declared before r_q; == the prover's canonical order); "
                                               "one RU per request (P1/cu=whole); global VU = r * N_V + i")


def step2_check_registered(inst, c0: dict[str, Any], *, registered_weights_root: str | None, ru_ids: list[str] | None = None,
                           input_check: Any = None) -> tuple[list[str], list[dict[str, Any]]]:
    """The contract's step-2 check for a REGISTERED program (no served production rows, no capture reader): the parts that are program-agnostic
    are CHECKED, the vLLM-only parts are named ``NOT_APPLICABLE`` with the reason.  Returns ``(problems, not_applicable)``.

    Checked: the epoch declares its request set and inputs; one boundary tree of kind ``boundary`` (root, leaf_count > 0) per declared request
    and none for an undeclared one; no RU tree in the phase-0 message (step-3 values before r_q); the weights tree, when the program has one, has
    the registered root.  ``v_in(C) = x`` is checked through ``input_check(rid, boundary_record, declared) -> problems[]`` when the caller supplies
    the program's input layout (E8's CORE: the typed input words are the first boundary leaves); without it the comparison is NOT_APPLICABLE by
    name (the record is then not a complete audit -- the reverifier says so too).  ``v_out(C) = y`` has no served copy to compare against on a
    registered program: the committed boundary IS the claimed output (the contract's step 2 compares the served y with it only when a serving
    layer exists) -- named NOT_APPLICABLE."""

    p: list[str] = []
    na: list[dict[str, Any]] = []
    trees = c0.get("trees") or {}
    if inst.epoch is None:
        return ["instance declares no epoch (legacy instance): the contract verifier needs the epoch's exact request set (R16)"], na
    declared = set(inst.epoch.request_ids)
    if len(declared) != inst.partition.instances:
        p.append(f"{len(declared)} declared requests but partition.instances = {inst.partition.instances}")
    # one boundary tree per RU (== per request on a one-RU-per-request epoch; per stage call on a stage-RU epoch: ru_ids_of)
    units = list(ru_ids) if ru_ids else sorted(declared)
    for uid in units:
        t = trees.get(f"boundary/{uid}")
        if t is None:
            p.append(f"no boundary commitment for RU {uid} (step 1 incomplete: r_q cannot be derived)")
        elif t.get("kind") != "boundary" or not t.get("root") or int(t.get("leaf_count", 0)) <= 0:
            p.append(f"boundary tree of {uid} malformed")
    extra = sorted(k.split("/", 1)[1] for k in trees if k.startswith("boundary/") and k.split("/", 1)[1] not in set(units))
    if extra:
        p.append(f"boundary commitments for undeclared RUs {extra}")
    if any(isinstance(v, dict) and v.get("kind") == "ru" for v in trees.values()):
        p.append("phase-0 commitment already holds RU interior trees (step-3 values before r_q: reordered message, refused)")
    w = trees.get("weights")
    if w is None:
        na.append(not_applicable("step2.weights_root", "the phase-0 message has no weights tree (the registered program declares no prescribed weight inputs)"))
    elif registered_weights_root is not None and w.get("root") != registered_weights_root:
        p.append(f"weights root {str(w.get('root'))[:12]} != registered {registered_weights_root[:12]} (ROT-INPUT)")
    if inst.epoch.inputs is None:
        p.append("the instance declares an epoch but pins no inputs (epoch.inputs) -- v_in(C) = x cannot be established (input agreement is unconditional)")
    else:
        for rid in sorted(declared):
            decl = inst.epoch.inputs.per_request.get(rid)
            if decl is None:
                p.append(f"request {rid}: no declared input in the instance (epoch.inputs)")
            elif input_check is not None:
                try:
                    p.extend(f"request {rid}: {x}" for x in (input_check(rid, {u: trees.get(f"boundary/{u}") for u in units}, decl) or []))
                except Exception as e:  # noqa: BLE001
                    p.append(f"request {rid}: declared == committed input comparison failed: {type(e).__name__}: {str(e)[:160]}")
        if input_check is None:
            na.append(not_applicable("step2.v_in_equals_x", "the declared == committed input comparison needs the registered program's input layout (which boundary leaves "
                                     "hold x); no input_check supplied -- the record is NOT a complete audit of v_in(C) = x until it is",
                                     replaced_by="the prover-side S2.inputs_declared is not the verifier's check"))
    na.append(not_applicable("step2.served_equals_committed", "no serving layer: a registered program has no served y to compare with the committed boundary (the committed "
                             "output tuple IS the claim; y's typing is the program's output domain, checked against the instance's outputs at V7)"))
    na.append(not_applicable("step2.output_typed_by_vocabulary", "no token vocabulary: the registered program's outputs are typed by their gate widths (V7.output_domain on the committed words)"))
    return p, na


# -- step 4 -------------------------------------------------------------------------------------------------------------------------


def committed_rus(c2: dict[str, Any], ru_of_request: dict[str, int]) -> list[int]:
    """The RUs whose step-3 values ``commitments.json`` holds (an ``ru/<rid>/...`` tree of kind ``ru``)."""

    rus = set()
    for tid, rec in (c2.get("trees") or {}).items():
        if rec.get("kind") == "ru":
            rid = rec.get("request_id") or tid.split("/")[1]
            if rid in ru_of_request:
                rus.add(ru_of_request[rid])
    return sorted(rus)


def step4_check(inst, *, J: list[int], r_q: str, instance_digest: str, c0: dict[str, Any], c2: dict[str, Any], exreq: dict[str, Any],
                ru_of_request: dict[str, int], selectable_uncovered: dict[int, int] | None = None,
                policy: str | None = None) -> tuple[list[str], list[int], list[int]]:
    """Refusals of the contract's step-4 check, plus ``(F, committed)`` for ``partition_sample.stage2``.  ``selectable_uncovered[r]`` = the
    number of VUs of retained RU r that Q could select but whose output interface the interior commitment does NOT hold (A2/A5 supply it
    from the committed interfaces vs the population; when given and > 0 for a retained RU, the step is refused: the prover must exempt
    that RU or commit the interface -- never a family exemption)."""

    p: list[str] = []
    B_ex = inst.exemption_budget
    if not inst.is_contract_instance:
        p.append("instance is legacy (no epoch / B_ex not an RU count): refused by the contract verifier")
    p += rr.validate(exreq, J=J, B_ex=B_ex, instance_digest=instance_digest, r_q=r_q, policy=policy)
    F = [int(r) for r in exreq.get("F", [])]
    committed = committed_rus(c2, ru_of_request)
    retained = [r for r in J if r not in set(F)]
    if not set(committed) <= set(J):
        p.append(f"interiors committed for RU(s) {sorted(set(committed) - set(J))} outside J")
    missing = [r for r in retained if r not in set(committed)]
    if missing:
        p.append(f"retained RU(s) {missing} have no interior commitment and are not in F (r_s cannot be derived)")
    for tid, rec in (c0.get("trees") or {}).items():
        if (c2.get("trees") or {}).get(tid) != rec:
            p.append(f"phase-0 tree {tid} changed in the step-3 commitment (recommit refused)")
    per = exreq.get("per_ru") or {}
    for r in retained:
        rec = per.get(str(r)) or {}
        if not (rec.get("response") or {}).get("step3_committed"):
            p.append(f"RU {r}: retained but its response says step-3 values not committed")
    if selectable_uncovered:
        for r in retained:
            n = int(selectable_uncovered.get(r, 0))
            if n > 0:
                p.append(f"RU {r}: {n} selectable VU(s) have no committed output interface and the RU is not in F (an uncovered selected equation cannot be accepted: HC)")
    return p, sorted(F), committed


# -- after Q: the per-RU record (R17 + R18) --------------------------------------------------------------------------------------------


def ru_fields(inst, *, exreq: dict[str, Any], selected_rows: list[dict[str, Any]], unsupported_kinds: set[str] | None, s: Fraction | str) -> dict[str, Any]:
    """``ru_fields.json``: for every RU of J both R17 fields (the prover's, with ``selected_checks_establishable`` filled by the verifier
    from the actual selection) and the R18 exposure; the sampled unsupported identities are listed, never estimated."""

    F = set(int(r) for r in exreq.get("F", []))
    per = exreq.get("per_ru") or {}
    by_ru: dict[int, list[dict[str, Any]]] = {}
    for row in selected_rows:
        by_ru.setdefault(int(row["ru_index"]), []).append(row)
    out: dict[str, Any] = {}
    per_ru_n: dict[int, int | None] = {}
    sampled_unsupported: list[dict[str, Any]] = []
    for k, rec in per.items():
        r = int(k)
        rows = by_ru.get(r, [])
        kinds = [(f"{x['kind']['fn']}|{x['kind']['form']}" if x.get("kind") else None) for x in rows]
        uns = [x for x, kn in zip(rows, kinds, strict=False) if unsupported_kinds is not None and kn in unsupported_kinds]
        rsp = dict(rec.get("response") or {})
        if r in F:
            rsp["selected_checks_establishable"] = None  # nothing selected in an exempted RU
            establishable_note = "exempted: no VU of this RU is in Q"
        elif unsupported_kinds is None:
            rsp["selected_checks_establishable"] = None
            establishable_note = "unsupported-kind set not supplied (A5/A6): not determined"
        else:
            rsp["selected_checks_establishable"] = not uns
            establishable_note = f"{len(uns)} of {len(rows)} selected VUs are of a kind without a checker"
        u = rec.get("unsupported") or {}
        n = u.get("unsupported_vus_n")
        if r not in F:
            per_ru_n[r] = int(n) if n is not None else None
        for x in uns:
            sampled_unsupported.append({"ru_index": r, "vu_index": x["vu_index"], "kind": f"{x['kind']['fn']}|{x['kind']['form']}", "gate_interval": x.get("gate_interval")})
        out[k] = {"request_id": rec.get("request_id"), "in_F": r in F, "selected_vus": len(rows), "consistency": rec.get("consistency"),
                  "response": {**rsp, "selected_checks_establishable_note": establishable_note}, "unsupported": u}
    return {"schema": rr.RU_FIELDS_SCHEMA, "instance_digest": exreq.get("instance_digest"), "r_q": exreq.get("r_q"), "J": exreq.get("J"), "F": sorted(F),
            "per_ru": out, "completion_risk": rr.completion_risk(s, per_ru_n), "sampled_unsupported": sampled_unsupported,
            "sampled_unsupported_count": len(sampled_unsupported),
            "note": "R17: consistency is the prover's determination (or unknown) -- sampled verification never establishes it; R18: exposure from the actual configuration"}


def output_copy_step4(sel: Path, c2: dict[str, Any], retained_request_ids: list[str], *, recompute_leaves: bool = True) -> dict[str, Any]:
    """[R4, A2's M2 wired at step 4] The served token committed twice -- step-1 boundary leaf ``output_token/t`` and the step-3 interior
    ``sampler.select/port=0`` copy the teacher-forced replay writes -- must agree for every retained RU (``R in J minus F``) BEFORE ``r_s``.
    Runs A2's ``verity_vllm_adapter.obligations.output_copy_checks`` over exactly the retained requests: both openings are authenticated
    against their committed roots in ``c2`` (``commitments.json``, whose digest ``phase_s`` binds; the interior trees are the step-3
    values, the boundary trees the step-1 values -- both fixed before Q) and the int32 words compared exactly with the served token.

    Returns ``{"ok", "rows", "problems", "per_request"}``; ``ok`` is False when any row fails, when a retained request has no rows (its
    RU committed no interior tree: nothing to compare -- step4_check refuses that separately), or when the check cannot run (fail closed)."""
    out: dict[str, Any] = {"ok": False, "rows": 0, "problems": [], "per_request": {}, "label": "MEASURED"}
    try:
        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import TreeCache
        from verity_vllm_adapter.obligations import output_copy_checks

        reader = CaptureReader(sel)
        rows = output_copy_checks(reader, c2, TreeCache(reader, recompute_leaves=recompute_leaves), request_ids=list(retained_request_ids))
    except Exception as e:  # noqa: BLE001
        out["problems"].append(f"output copy check could not run: {type(e).__name__}: {str(e)[:200]}")
        return out
    out["rows"] = len(rows)
    for rid in retained_request_ids:
        mine = [r for r in rows if r["request_id"] == rid]
        bad = [r for r in mine if not r.get("ok")]
        out["per_request"][rid] = {"rows": len(mine), "failing": len(bad)}
        if not mine:
            out["problems"].append(f"{rid[:8]}: retained RU with no committed interior output copy to compare (no rows)")
        for r in bad[:8]:
            out["problems"].append(f"{rid[:8]} t={r.get('t')}: {r.get('problem')} (boundary {r.get('boundary_token')}, interior {r.get('interior_token')}, served {r.get('served_token')})")
    out["ok"] = not out["problems"]
    return out


def load_json(p: Path) -> dict[str, Any]:
    return json.loads(Path(p).read_text())


class NotApplicable(Exception):
    """[E9, D20] raised inside a Serve_v1-only block of ``reverify.py`` on a registered program so the check is NAMED ``NOT_APPLICABLE:<check>``."""


def registered_input_check(inst_doc: dict[str, Any], record_dir: Path) -> tuple[Any, str]:
    """[E9, D20/D24] The registered program's ``v_in(C) = x`` predicate, looked up in the builder module like E3's ``position_adapter``.

    Candidates BY NAME, in order: ``program.builder.input_check`` when pinned (``"<module>:<symbol>"`` or a symbol of the builder module); else
    ``input_check`` then ``make_input_check`` in ``program.builder.module`` and in the builder package's ``adapter`` module (E8 keeps both in
    ``e2e.demo_core.adapter`` next to ``position_adapter``; D2's contract is ``input_check(record_dir, params) -> check``).  ``input_check`` is
    called as ``fn(record_dir, params)`` (``params`` = ``program.builder.params``), ``make_input_check`` as ``fn(record_dir, **params)``; either
    must return ``check(rid, {ru_id: boundary tree record}, declared) -> problems[]`` for :func:`step2_check_registered`.  Returns
    ``(check | None, note)``: ``None`` when nothing is exposed (the caller names ``NOT_APPLICABLE:step2.v_in_equals_x`` -- never a pass) or
    when building it raises (the note says so; ``V8.step2`` / the driver's ``STEP2_INPUT_CHECK`` event carry the note)."""
    import importlib

    b = (inst_doc.get("program") or {}).get("builder")
    if not isinstance(b, dict) or not b.get("module"):
        return None, "no program.builder pin"
    params = dict(b.get("params") or {})
    mod_name = str(b["module"])
    pkg = mod_name.rpartition(".")[0]
    pinned = b.get("input_check")
    if pinned:
        spec = str(pinned)
        m, _, sym = spec.rpartition(":") if ":" in spec else (mod_name, "", spec)
        candidates = [(m, sym)]
    else:
        mods = [mod_name] + ([f"{pkg}.adapter"] if pkg and f"{pkg}.adapter" != mod_name else [])
        candidates = [(m, sym) for m in mods for sym in ("input_check", "make_input_check")]
    tried: list[str] = []
    for m, sym in candidates:
        try:
            fn: Any = importlib.import_module(m)
            for part in sym.split("."):
                fn = getattr(fn, part)
        except (ImportError, AttributeError) as e:
            tried.append(f"{m}:{sym} ({type(e).__name__})")
            continue
        try:
            check = fn(Path(record_dir), params) if sym.split(".")[-1] == "input_check" or pinned else fn(Path(record_dir), **params)
        except TypeError:
            try:
                check = fn(Path(record_dir), **params)
            except Exception as e:  # noqa: BLE001
                return None, f"{m}:{sym} raised {type(e).__name__}: {str(e)[:160]}"
        except Exception as e:  # noqa: BLE001
            return None, f"{m}:{sym} raised {type(e).__name__}: {str(e)[:160]}"
        if not callable(check):
            return None, f"{m}:{sym} returned a non-callable {type(check).__name__}"
        return check, f"{m}:{sym}({params})"
    return None, f"input check not exposed: {'; '.join(tried)}"


def registered_kind_classes(pop_doc: dict[str, Any]) -> dict[str, str]:
    """[E9, D20] ``{fn|form: class}`` for a registered program without an expressibility table: ``input`` for Input kinds, ``provable`` for a
    kind the ONE builder registry (``coverage_v1.elem_builders.ALL_BUILDERS``) can build, ``no-checker`` otherwise -- by name, from the
    population's kind table (``kinds.entries[]``: ``fn``, ``form``) and the registry; nothing is inferred from the prover."""
    entries = ((pop_doc.get("kinds") or {}).get("entries") if isinstance(pop_doc.get("kinds"), dict) else pop_doc.get("kinds")) or []
    try:
        from e2e.coverage_v1 import elem_builders as eb

        buildable = set(getattr(eb, "ALL_BUILDERS", eb.BUILDERS))
    except Exception:  # noqa: BLE001
        buildable = set()
    out: dict[str, str] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        kind = f"{e.get('fn')}|{e.get('form')}"
        # the class vocabulary locate_selected dispatches on: a kind the registry can build is left for locate_selected to label
        # "provable (BUILDERS: <family>)" itself (a pre-labelled "provable ..." would send it down the Serve_v1 Gemm path -- MEASURED 00:16Z:
        # `GenericLocator has no locate_gemm_coordinate` on E8's CORE record was THIS, not only E3's dispatch); Input kinds are "input unit"
        out[kind] = "input unit" if str(e.get("fn")) == "Input" else ("supported (registry builder)" if kind in buildable else "no-checker")
    return out


def population_pins_check(inst_doc: dict[str, Any], pop_doc: dict[str, Any]) -> list[str]:
    """[E9, D24 2026-09-12] ``V2.partition_digest`` (and program / plan) BY NAME at the step-2 barrier, BEFORE r_q: the population file the
    verifier was handed must carry exactly the instance's pinned ``partition.partition_digest``, ``program.digest`` and plan digest.  Each problem
    string names its check and carries BOTH FULL digests (a 12-hex prefix hid a deep byte flip on E8's CORE-live-neg-mispin-01).  This is the
    named check; ``partition_sample.bind`` inside ``stage1`` is the backstop and the driver catches its ``ChronologyError`` at the boundary."""
    p: list[str] = []
    part = inst_doc.get("partition") or {}
    prog = inst_doc.get("program") or {}
    pairs = (("V2.partition_digest", "partition.partition_digest", part.get("partition_digest"), pop_doc.get("partition_digest")),
             ("V2.program_digest", "program.digest", (prog.get("digest") if isinstance(prog.get("digest"), str) else (prog.get("digest") or {}).get("value")) or prog.get("program_digest"),
              pop_doc.get("program_digest")),
             ("V2.plan_digest", "partition.plan_descriptor_digest", part.get("plan_descriptor_digest"), pop_doc.get("plan_digest")))
    for check, field, want, got in pairs:
        if want is None or got is None:
            if check == "V2.partition_digest":
                p.append(f"{check}: instance {field} = {want!r}, population file = {got!r}: a digest is missing (nothing to bind the population to the pin)")
            continue
        if str(want) != str(got):
            p.append(f"{check}: population file's {field.split('.')[-1]} {got} differs from the instance's pinned {field} {want} (the population must be THE validated partition, S3/S7)")
    return p


def chronology_refusal_name(exc: BaseException) -> str:
    """The check a ``ChronologyError`` / ``InstanceError`` from ``partition_sample`` stands for (the message's leading ``V2.<name>:`` when present,
    else ``V2.population_bind``): the driver's ``STEP2_REFUSED`` / ``STEP4_REFUSED`` events carry it, never a bare traceback."""
    msg = str(exc)
    head = msg.split(":", 1)[0].strip()
    return head if head.startswith("V2.") and " " not in head else "V2.population_bind"


def proof_record_consistency(chunk_dir: Path, public_values_hex: str, *, record: dict[str, Any] | None = None, statement_sha256: str | None = None) -> list[str]:
    """[E9, D25 2026-09-12] The record-side conjuncts of ``V4.proof.<chunk>`` (``reverify.py`` V4), as ONE shipped function so the live step 5
    (E8's runner / ``step5_live``) and the post-hoc ``reverify.py`` agree BY NAME: the prover's record must describe the statement it proved.

    * ``sha256(statement.bin) == composed.json#statement_digest`` -- the record names the digest of the statement actually emitted
      (MEASURED defect on CORE-live-exhaustive-02: ``statement_digest`` was the v2 statement's while ``statement.bin`` / the proof's ``h_L`` were v3);
    * ``public_values[32:64] == sha256(statement.bin)`` (``h_L``: the proof binds THIS statement);
    * ``public_values[:32] == composed.json#digest`` (``h_O``: the typed object of the record);
    * ``public_values[64] == 1`` (the guest's verdict byte).
    Returns the problems (empty == consistent).  ``check_binding`` (object -> h_O recomputation) stays the caller's separate step.
    ``record`` / ``statement_sha256`` [E9 SCALE 23:10Z, read-once]: the caller's already-parsed ``composed.json`` and ``sha256(statement.bin)``
    (``reverify.py`` had read chunk00's 405 MB composed.json TWICE per verification: MEASURED); absent, they are read here as before."""
    import hashlib

    p: list[str] = []
    cdir = Path(chunk_dir)
    try:
        if record is None:
            record = json.loads((cdir / "composed.json").read_text())
        h_l = statement_sha256 if statement_sha256 is not None else hashlib.sha256((cdir / "statement.bin").read_bytes()).hexdigest()
    except (OSError, ValueError) as e:
        return [f"V4.proof.{cdir.name}: record unreadable: {type(e).__name__}: {str(e)[:160]}"]
    try:
        pv = bytes.fromhex(str(public_values_hex))
    except ValueError:
        return [f"V4.proof.{cdir.name}: public_values is not hex"]
    if len(pv) < 65:
        p.append(f"V4.proof.{cdir.name}: public values are {len(pv)} bytes, expected h_O||h_L||verdict (65)")
        return p
    sd = str(record.get("statement_digest"))
    if h_l != sd:
        p.append(f"V4.proof.{cdir.name}: composed.json#statement_digest {sd} != sha256(statement.bin) {h_l} (statement_format {record.get('statement_format')!r}): "
                 "the record describes a statement other than the one it emitted -- refused")
    if pv[32:64].hex() != h_l:
        p.append(f"V4.proof.{cdir.name}: public h_L {pv[32:64].hex()} != sha256(statement.bin) {h_l}: the proof does not bind this statement")
    if pv[:32].hex() != str(record.get("digest")):
        p.append(f"V4.proof.{cdir.name}: public h_O {pv[:32].hex()} != composed.json#digest {record.get('digest')}")
    if pv[64] != 1:
        p.append(f"V4.proof.{cdir.name}: public verdict byte is {pv[64]}, not 1")
    return p


def record_obligation_format(record: dict[str, Any]) -> str:
    """[E9, D28 2026-09-12] The obligation format a proof record declares, BY NAME: ``composed.json#obligation_format`` when present, else the
    typed object's own ``format`` field, else ``typed-obligation/v0``.  It selects the boundary rule of ``validate`` / ``check_binding`` /
    ``check_expected_positions`` (``veritor.interface.ALIASED_INCOMING_FORMATS``: v1 admits one committed word on several incoming positions);
    it is never inferred from the object's shape."""
    from veritor.interface import FORMAT

    f = record.get("obligation_format") or (record.get("object") or {}).get("format") or FORMAT
    return str(f)


def aliased_body_expectation(kind: str, operand_keys: Sequence[tuple[Any, ...]], canonical_digest: str | None = None) -> tuple[tuple[tuple[Any, ...], ...], str, tuple[int, ...]]:
    """[E9, D30 2026-09-12] E3's position statement of a one-gate registry kind (``operand_keys`` = the operands' ``(tree, leaf, offset)`` in
    signature order, repeats allowed) folded per E7's ``aliased-body/v1`` (``elem_builders.ALIAS_RULE``): returns ``(incoming_keys, body_digest,
    wiring)`` -- the DISTINCT words in first-occurrence order, the body digest the builder proves (E7's ``AliasSpec(kind_spec(kind), wiring)
    .subcircuit.digest``, ONE registry, ONE derivation; the kind's canonical body when nothing repeats) and the wiring.  This is what the
    verifier's expectation for such a unit must carry; ``RecordPartition._from_positions`` (E3, ``verifier_v3/expected.py``) is MEASURED to
    carry the canonical body with the repeated positions instead (finding D30/E9, report 01:5xZ)."""
    from e2e.coverage_v1 import elem_builders as EB

    keys = [tuple(k) for k in operand_keys]
    order: dict[tuple[Any, ...], int] = {}
    wiring: list[int] = []
    distinct: list[tuple[Any, ...]] = []
    for k in keys:
        if k not in order:
            order[k] = len(distinct)
            distinct.append(k)
        wiring.append(order[k])
    spec = EB.kind_spec(kind)
    if len(distinct) == len(keys):
        sub = getattr(spec, "subcircuit", None)
        digest = str(canonical_digest or (sub.digest if sub is not None else EB._ck_elem().kind_subcircuit(spec).digest))
        return tuple(distinct), digest.lower(), tuple(wiring)
    digest = EB.AliasSpec(spec, tuple(wiring)).subcircuit.digest.lower()
    assert digest in EB.alias_digests(kind, len(distinct)), (kind, wiring)  # the registry's own enumeration names this body
    return tuple(distinct), digest, tuple(wiring)


# ------------------------------------------------------------------------------------------------- [E9, D32.1] alias map by IDENTITY
class AliasIdentityRefusal(ValueError):
    """The pinned program's operand gate identities and the committed word identities disagree for one VU (``code`` names it):
    ``DUPLICATE_BOUNDARY_POSITION`` -- two DISTINCT operand gates resolve to ONE committed position (the partition would commit one word for
    two program values); ``SPLIT_GATE`` -- one operand gate resolves to two different positions (a word committed twice)."""

    def __init__(self, code: str, msg: str) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code


def alias_wiring_from_gates(operand_gates: Sequence[Any]) -> tuple[int, ...]:
    """``operand_position -> incoming_port`` from the PINNED PROGRAM's operand gate identities ALONE (D32 rule 1): port ``j`` is the j-th
    DISTINCT operand gate in first-occurrence order; a gate id repeated at two operand positions is ONE port read twice.  Never from bits
    (two distinct gates whose values happen to agree are two ports), never from the prover's body.  An identity is a gate number or, for a
    program literal, ``("literal", <Const primitive id>)`` (``prescribed-literal/v1``: the program text is the identity)."""

    order: dict[Any, int] = {}
    wiring: list[int] = []
    for g in operand_gates:
        g = int(g) if isinstance(g, (int, str)) and not isinstance(g, bool) and str(g).lstrip("-").isdigit() else g
        if g not in order:
            order[g] = len(order)
        wiring.append(order[g])
    return tuple(wiring)


def alias_body_digest(kind: str, wiring: Sequence[int], *, in_widths: Sequence[int] | None = None, spec: Any = None) -> str:
    """The body digest the verifier EXPECTS for a one-gate registry kind read through ``wiring`` -- built here from the interface's own
    ``make_subcircuit`` (the digest is over the body alone: incoming widths, gate, operand ports, outgoing), NOT taken from E7's ``AliasSpec``.
    ``in_widths`` = the widths of the DISTINCT incoming words (default: the kind's operand width for each port).  The identity wiring gives
    the kind's canonical body.  ``spec`` overrides the kind's registry spec when the unit's body is chosen by the pinned program
    (:func:`alias_identity_expectation`: the PV step's ``tc_dot16_0``)."""
    from e2e.coverage_v1 import elem_builders as EB
    from veritor.interface.programs import IN, make_subcircuit

    spec = spec if spec is not None else EB.kind_spec(kind)
    ports = len(set(int(j) for j in wiring))
    if in_widths is None:
        base = tuple(getattr(spec, "in_widths", None) or (int(spec.in_width),) * len(wiring))
        first: dict[int, int] = {}
        for i, j in enumerate(wiring):
            first.setdefault(int(j), int(base[i]))
        in_widths = tuple(first[j] for j in sorted(first))
    assert len(in_widths) == ports, (kind, wiring, in_widths)
    sub = make_subcircuit(list(int(w) for w in in_widths), [(str(spec.gate), [(IN, int(j)) for j in wiring])], [0])
    return str(sub.digest).lower()


def alias_identity_expectation(st: Any, *, kind: str | None = None, literals: "LiteralTable | None" = None) -> tuple[Any, tuple[int, ...]]:
    """[D32.1] The verifier's INDEPENDENT expectation of a one-gate registry kind from E3's :class:`PositionStatement` ``st``: the
    ``operand_position -> incoming_port`` map comes from ``st.operand_gates`` (the pinned program's gate identities, :func:`alias_wiring_from_gates`),
    the committed word identities ``(tree, leaf, offset)`` of ``st.operands`` are then REQUIRED to agree with it (same gate -> same word, distinct
    gates -> distinct words; :class:`AliasIdentityRefusal` by code otherwise -- equal BITS never merge two identities), the body digest is
    :func:`alias_body_digest` over that wiring, and ``incoming`` lists each distinct word once in first-occurrence order.  Returns
    ``(ExpectedObligation, wiring)``.

    [NIGHT G1, ``prescribed-literal/v1``] A literal operand (``origins == "literal"``: a program ``Const<w>[0x…]_v1`` gate, E5 §4) is an incoming
    word of the verifier's :class:`LiteralTable` ``literals`` (derived from the pinned program; its position :meth:`LiteralTable.position`); its
    IDENTITY for the port map is the Const PRIMITIVE id (the program text: two ``Const32[0x0]`` gates are one word by construction), so a literal
    read at two positions is one port (E7's ``F32Add(l, l)`` clause) and a literal never merges with a committed word.  Without a table
    (``literals is None``) a statement with a literal operand refuses ``LITERAL_UNPINNED`` (fail closed) -- the verifier never takes the
    literal's bits from the prover's object."""
    from veritor.interface.expected import ExpectedObligation, ExpectedValue

    kind = str(kind or st.kind)
    gates = tuple(int(g) for g in st.operand_gates)
    if len(gates) != len(st.operands):
        raise AliasIdentityRefusal("NO_GATE_IDENTITIES", f"{st.label}: {len(st.operands)} operand positions but {len(gates)} operand gates -- the map "
                                   "must come from the pinned program's gate identities, not from the positions")
    origins = tuple(st.origins[:len(st.operands)]) if st.origins else ("committed",) * len(st.operands)
    committed: list[tuple[Any, tuple[Any, ...], str]] = []
    for g, v, o in zip(gates, st.operands, origins):
        if o == "literal" or str(v.tree) == LITERAL_ORIGIN_TREE:
            if literals is None:
                raise AliasIdentityRefusal("LITERAL_UNPINNED", f"{st.label}: operand gate {g} is the program literal {v.leaf} and the verifier has no "
                                           f"literal table ({LITERAL_RULE}: protocol_instance.json#commitments.literals) -> refused (fail closed)")
            try:
                pos = literals.position(str(v.leaf))
            except KeyError:
                raise AliasIdentityRefusal("LITERAL_UNKNOWN", f"{st.label}: operand gate {g} names literal {v.leaf!r}, which the pinned program's "
                                           f"literal table ({len(literals.ids)} ids) does not contain") from None
            committed.append((("literal", str(v.leaf)), tuple(pos.key), o))  # identity = the primitive, not the gate number
        else:
            committed.append((g, tuple(v.key), o))
    spec = None
    if kind == PV_ACC_ZERO_KIND and committed and committed[0][2] == "literal" and _is_zero32(str(st.operands[0].leaf)):
        # [E9, NIGHT 14:50Z -- n2c chunk09: 25 PV units "no selected unit expects that value"] the ZERO32 ACCUMULATOR of the PV step is compiled
        # into the body, not a literal port: the approved family's `tc_dot16_0` gate (E7 `elem_builders.PvSpec(acc_zero=True)` / `resolve_pv_unit`,
        # since 23:40Z; the same body the GEMM coordinate and `DotBf16{K}|call` open with).  Derived HERE from the pinned program alone -- E3's
        # operand gate 0 is the `Const32[0x00000000]_v1` primitive -- never from the prover's object; the body digest fixes acc = 0, so a prover
        # cannot smuggle a non-zero accumulator (a committed-word acc is the OTHER body, `tc_dot16`, with a 32-bit port 0).  Every other literal
        # of the kind (the ZERO16 pads) stays a prescribed-literal/v1 port.  ONE rule for the accumulator, by kind name: `PV_ACC_ZERO_KIND`.
        from e2e.coverage_v1 import elem_builders as EB

        committed = committed[1:]
        spec = EB.PvSpec(acc_zero=True)
    wiring = alias_wiring_from_gates([g for g, _k, _o in committed])
    # identity agreement: the gate identity and the committed identity name the same partition of the operand positions
    word_of_port: dict[int, tuple[Any, ...]] = {}
    port_of_word: dict[tuple[Any, ...], int] = {}
    for i, ((g, key, _o), j) in enumerate(zip(committed, wiring)):
        if j in word_of_port and word_of_port[j] != key:
            raise AliasIdentityRefusal("SPLIT_GATE", f"{st.label}: operand gate {g} (port {j}) is committed at {word_of_port[j]} and at {key} (position {i})")
        word_of_port.setdefault(j, key)
        if key in port_of_word and port_of_word[key] != j:
            raise AliasIdentityRefusal("DUPLICATE_BOUNDARY_POSITION", f"{st.label}: operand position {i} (gate {g}) and position "
                                       f"{wiring.index(port_of_word[key])} (gate {committed[wiring.index(port_of_word[key])][0]}) are DISTINCT program values "
                                       f"committed at ONE word {key}; two identities never share a port")
        port_of_word.setdefault(key, j)
    incoming = tuple(ExpectedValue(*word_of_port[j]) for j in range(len(word_of_port)))
    digest = alias_body_digest(kind, wiring, spec=spec)
    e = ExpectedObligation(incoming, tuple(ExpectedValue(*v.key) for v in st.outputs), digest, (None,) * len(st.outputs), {}, st.label)
    return e, wiring


#: [E9, NIGHT 14:50Z] the ONE kind whose operand-0 `Const32[0x00000000]` is compiled into the body (`tc_dot16_0`) instead of read as a
#: prescribed-literal/v1 port -- E7's `elem_builders.PV_KIND`, spelled here so the verifier's rule is its own (asserted equal in
#: `tests/contract/a8_conformance/e9_literal_test.py`)
PV_ACC_ZERO_KIND = "AmpereBF16TcDot16_v1|batch"


def _is_zero32(const_id: str) -> bool:
    try:
        return parse_const_id(const_id) == (32, 0)
    except ValueError:
        return False


# ------------------------------------------------------------------ [E9, NIGHT G1 2026-09-12] prescribed-literal/v1: the verifier's literal table
#: the rule's name (PROPOSED; E7 drafts, E9 co-signs): a program literal at an operand position is an ordinary INCOMING committed word of the
#: obligation, opened from a public tree whose root the VERIFIER derives from the pinned program and that the instance pins
#: (`protocol_instance.json#commitments.literals`, written by E6 like `commitments.tables`).  No Input gate, no pin, no guest change (MEASURED in
#: the approved guest 9fd00e60, E9 night report 05:30Z).
LITERAL_RULE = "prescribed-literal/v1"
#: E3's spelling of a literal position in a `PositionStatement`: `tree == "literal"`, `leaf` = the Const primitive id (`Const32[0x3727c5ac]_v1`)
LITERAL_ORIGIN_TREE = "literal"
LITERAL_TREE_FRAMING = "verity-vllm/leaf/v1"
LITERAL_TREE_OWNER = -2  # public data, like the weights and the MUFU tables


def literal_tree_id(width: int) -> str:
    return f"literal/w={int(width)}"


def literal_leaf_id(width: int) -> str:
    return f"literal/w={int(width)}/chunk=0"


def parse_const_id(const_id: str) -> tuple[int, int]:
    """``(width, bits)`` of a ``Const<w>[0x…]`` / ``Const<w>[0x…]_v1`` primitive id; ``ValueError`` when it is not one."""
    s = str(const_id)
    if not s.startswith("Const") or "[0x" not in s:
        raise ValueError(f"{const_id!r} is not a Const<w>[0x…] primitive id")
    head, rest = s.split("[0x", 1)
    w = int(head[len("Const"):])
    bits = int(rest.split("]", 1)[0], 16)
    if w <= 0 or bits < 0 or bits >= (1 << w):
        raise ValueError(f"{const_id!r}: value does not fit {w} bits")
    return w, bits


def program_const_ids(program: Any) -> tuple[str, ...]:
    """[E7 literaltable-perf] The distinct ``Const<w>[0x…]_v1`` primitive ids of ``program`` from its DEFINITION DAG, O(#definitions).

    A ``verity.ir`` program is a tree of function-call INSTANCES whose bodies are shared ``SpecializedDefinition`` objects; the primitives that
    occur in the circuit are exactly the primitives of the bodies reachable from the root (a body node materialises ``node.gates >= 1`` gates
    of ``node.fn``; a 0-gate node -- an empty batch -- puts nothing in the circuit and is skipped).  Visiting each definition once (memo by
    object identity) is therefore the same SET as the scan of every gate, at the cost of the sum of body lengths (thousands of nodes on
    TINY / SMOL360 / QWEN05).  Works on a full ``Program`` and on E2's ``LazyProgram`` (the lazy root's ``nodes`` materialise one step-call at
    a time; the step definitions are shared).  The literal tests are ``locator.literal_of``'s: ``Const`` family / name, zero-arity, no params;
    a primitive whose ``evaluate()`` is not the value in its name is a ``PositionRefusal("LITERAL_MISMATCH")``, never silently dropped."""
    from e2e.verifier_v3.locator import PositionRefusal
    from verity.ir.defs import PrimitiveDefinition

    ids: set[str] = set()
    seen: set[int] = set()
    stack = [program.root]
    while stack:
        fn = stack.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        for node in fn.body.nodes:
            if int(node.gates) <= 0:
                continue
            sub = node.fn
            if not isinstance(sub, PrimitiveDefinition):
                stack.append(sub)
                continue
            name = str(getattr(sub, "name", ""))
            if getattr(sub, "family", None) != "Const" and not name.startswith("Const"):
                continue
            if len(sub.params) or "[0x" not in name:
                continue
            try:
                w, bits = parse_const_id(sub.id)
            except ValueError:
                continue
            got = int(sub.evaluate())
            if got != bits:
                raise PositionRefusal("LITERAL_MISMATCH", f"{sub.id} evaluates to 0x{got:x}, not the literal in its name")
            ids.add(str(sub.id))
    return tuple(sorted(ids))


class LiteralTable:
    """The verifier's literal table under :data:`LITERAL_RULE`, DERIVED from the pinned program's ``Const<w>[0x…]_v1`` primitives (never from the
    record): per width one tree ``literal/w=<w>`` with ONE leaf ``literal/w=<w>/chunk=0`` (dtype ``uint<w>``, shape ``[n]``, the values in sorted
    order of their primitive ids, little-endian ``ceil(w/8)`` bytes each, ``verity-vllm/leaf/v1`` framing); a literal's position is
    ``(literal/w=<w>, literal/w=<w>/chunk=0, (index,))``.  E7's ``LiteralTable`` in ``elem_builders`` is the builder's separate derivation; the
    two are asserted equal in ``tests/contract/a8_conformance/e9_literal_test.py``."""

    def __init__(self, ids: Sequence[str]) -> None:
        self.ids: tuple[str, ...] = tuple(sorted({str(i) for i in ids}))
        self.widths: dict[str, int] = {}
        self.values: dict[str, int] = {}
        for i in self.ids:
            w, bits = parse_const_id(i)
            self.widths[i], self.values[i] = w, bits
        self.by_width: dict[int, tuple[str, ...]] = {}
        for i in self.ids:
            self.by_width.setdefault(self.widths[i], ())
            self.by_width[self.widths[i]] = (*self.by_width[self.widths[i]], i)
        self._index: dict[str, int] = {i: k for w, ids_ in self.by_width.items() for k, i in enumerate(ids_)}
        self._trees: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_program(cls, program: Any) -> "LiteralTable":
        """Every distinct ``Const<w>[0x…]`` primitive of the pinned program (E3's ``locator.literal_of``: zero-arity, no params, the value IS the
        program text).  A Const gate that the population lists as a VU output resolves to its committed word instead (E3's ``literal_word``);
        its primitive still has a row here -- the table is a function of the program text alone, which is what makes it derivable.

        [E7 literaltable-perf, 2026-09-12 09:00Z; E10's finding 08:10Z] Derived by :func:`program_const_ids` -- the walk of the program's
        DEFINITION DAG (each ``SpecializedDefinition`` body once; O(#definitions), milliseconds) -- not by a gate-by-gate scan (242.7 M gates
        on TINY, 865 M on QWEN05: minutes to hours in Python before r_s).  The scan is :meth:`from_program_scan`; the two are asserted equal in
        ``tests/contract/e7_literal_walk_test.py`` (and against E6's pins on TINY / SMOL360 / QWEN05)."""
        return cls(program_const_ids(program))

    @classmethod
    def from_program_scan(cls, program: Any) -> "LiteralTable":
        """The reference derivation: ``locator.literal_of`` at every gate.  O(#gates); small programs and the equal-pin test only."""
        from e2e.verifier_v3 import locator as vloc

        ids: set[str] = set()
        for g in range(int(program.circuit.size)):
            lp = vloc.literal_of(program, g)
            if lp is not None:
                ids.add(str(lp.leaf))
        return cls(sorted(ids))

    @classmethod
    def from_pin(cls, pin: dict[str, Any]) -> "LiteralTable":
        return cls([str(i) for i in (pin.get("literals") or [])])

    def position(self, const_id: str):
        from veritor.interface.expected import ExpectedValue

        i = str(const_id)
        return ExpectedValue(literal_tree_id(self.widths[i]), literal_leaf_id(self.widths[i]), (self._index[i],))

    def words(self, width: int) -> tuple[int, ...]:
        return tuple(self.values[i] for i in self.by_width.get(int(width), ()))

    def leaf_bytes(self, width: int) -> bytes:
        n = (int(width) + 7) // 8
        return b"".join(int(v).to_bytes(n, "little") for v in self.words(width))

    def trees(self) -> dict[str, dict[str, Any]]:
        """``tree id -> {root, leaf_count, ids_digest, leaf, dtype, shape, width, words, framing, owner}`` (roots computed here)."""
        if self._trees is None:
            from verity_vllm import hashing

            out: dict[str, dict[str, Any]] = {}
            for w, ids_ in sorted(self.by_width.items()):
                leaf = literal_leaf_id(w)
                h = hashing.leaf_hash(leaf, f"uint{w}", (len(ids_),), self.leaf_bytes(w))
                out[literal_tree_id(w)] = {"root": h.hex(), "leaf_count": 1, "ids_digest": hashing.ids_digest([leaf]).hex(), "leaf": leaf, "dtype": f"uint{w}",
                                           "shape": [len(ids_)], "width": int(w), "words": list(ids_), "framing": LITERAL_TREE_FRAMING, "owner": LITERAL_TREE_OWNER}
            self._trees = out
        return self._trees

    def pin(self) -> dict[str, Any]:
        """The ``commitments.literals`` block the verifier expects the instance to carry (E7's shape)."""
        return {"rule": LITERAL_RULE, "literals": list(self.ids),
                "trees": {t: {"root": d["root"], "leaf_count": d["leaf_count"], "ids_digest": d["ids_digest"]} for t, d in self.trees().items()}}

    def committed(self) -> dict[str, tuple[Any, list[str]]]:
        """``tree id -> (CommittedTree, [leaf ids in rank order])`` for ``expected.coverage_inputs(public_trees=...)``."""
        from veritor.interface.expected import CommittedTree

        return {t: (CommittedTree(d["root"], int(d["leaf_count"]), LITERAL_TREE_FRAMING), [d["leaf"]]) for t, d in self.trees().items()}

    def opening(self, tree: str) -> dict[str, Any]:
        """The one leaf of ``tree`` as an opening document (for a prover / a test fixture): id, dtype, shape, value bytes, empty path."""
        d = self.trees()[tree]
        return {"tree": tree, "root": d["root"], "leaf_count": 1, "index": 0, "canonical_id": d["leaf"], "dtype": d["dtype"], "shape": list(d["shape"]),
                "value": self.leaf_bytes(d["width"]), "path": []}


def literal_pin(inst_doc: dict[str, Any] | None) -> dict[str, Any] | None:
    c = ((inst_doc or {}).get("commitments") or {})
    p = c.get("literals")
    return dict(p) if isinstance(p, dict) else None


def literal_kinds_selected(selected_rows: Sequence[Any], positions: Any | None) -> list[int]:
    """The selected VUs (rows ``[vu_index, ...]`` / dicts) whose position statement has a literal operand -- needs E3's oracle; ``[]`` without one."""
    if positions is None:
        return []
    out: list[int] = []
    for r in selected_rows:
        vu = int(r["vu_index"] if isinstance(r, dict) else r[0])
        try:
            st = positions.statement({"vu_index": vu} if not isinstance(r, dict) else r)
        except Exception:  # noqa: BLE001 -- an unresolvable VU is V5's business, not this check's
            continue
        if any(o == "literal" for o in (st.origins or ())[:len(st.operands)]):
            out.append(vu)
    return out


def literal_table_check(inst_doc: dict[str, Any], *, program: Any | None = None, table: "LiteralTable | None" = None, literal_vus_selected: Sequence[int] = (),
                        in_play_reason: str = "") -> dict[str, Any]:
    """``V3.literal_table`` (live driver: step 4, before r_s, beside ``V3.interior_layout``; ``reverify.py``: the same name), as one pure result::

        {"ok": bool, "status": "PASS" | "FAIL" | "VACUOUS", "problems": [...], "pin": {...} | None, "derived": {...} | None, "literals": n, "trees": [...]}

    Rules (fail closed): the verifier derives the table from the PINNED PROGRAM (:meth:`LiteralTable.from_program`; ``table`` given = already derived);
    pin present -> ``rule``, ``literals`` (the id list) and every tree's ``root`` / ``leaf_count`` / ``ids_digest`` must equal the derivation
    (``LITERAL_PIN_MISMATCH``), no extra tree; pin absent + a selected VU with a literal operand -> ``LITERAL_UNPINNED``; pin absent and nothing
    selected reads a literal -> VACUOUS pass (legacy records unchanged).  A program without literals and a pin listing some -> mismatch."""
    pin = literal_pin(inst_doc)
    out: dict[str, Any] = {"ok": False, "status": "FAIL", "problems": [], "pin": pin, "derived": None, "literals": 0, "trees": [], "rule": LITERAL_RULE,
                           "literal_vus_selected": [int(v) for v in literal_vus_selected][:50], "reason": in_play_reason}
    if pin is None:
        if literal_vus_selected:
            out["problems"].append(f"LITERAL_UNPINNED: {len(literal_vus_selected)} selected VU(s) read a program literal (first: {list(literal_vus_selected)[:5]}) and "
                                   f"protocol_instance.json#commitments.literals is absent ({LITERAL_RULE}) -> refused (fail closed)")
            return out
        out.update(ok=True, status="VACUOUS", problems=[], note="no commitments.literals pin and no selected VU reads a literal" + (f" ({in_play_reason})" if in_play_reason else ""))
        return out
    if table is None and program is not None:
        table = LiteralTable.from_program(program)
    if table is None:
        out["problems"].append("LITERAL_TABLE_UNDERIVED: the instance pins commitments.literals but the verifier has no pinned program to derive the table from -> refused")
        return out
    want = table.pin()
    out["derived"] = want
    out["literals"] = len(table.ids)
    out["trees"] = sorted(want["trees"])
    if str(pin.get("rule")) != LITERAL_RULE:
        out["problems"].append(f"LITERAL_PIN_MISMATCH: commitments.literals.rule = {pin.get('rule')!r}, the verifier implements {LITERAL_RULE!r}")
    got_ids = [str(i) for i in (pin.get("literals") or [])]
    if got_ids != list(want["literals"]):
        extra, missing = sorted(set(got_ids) - set(want["literals"])), sorted(set(want["literals"]) - set(got_ids))
        out["problems"].append(f"LITERAL_PIN_MISMATCH: commitments.literals.literals ({len(got_ids)} ids) != the pinned program's Const primitives ({len(want['literals'])}): "
                               f"extra {extra[:4]}, missing {missing[:4]}" + ("" if extra or missing else "; same set, different order (sorted ids required)"))
    got_trees = pin.get("trees") or {}
    for t in sorted(set(got_trees) | set(want["trees"])):
        if t not in want["trees"]:
            out["problems"].append(f"LITERAL_PIN_MISMATCH: commitments.literals.trees[{t!r}] is pinned but the program derives no such tree")
            continue
        if t not in got_trees:
            out["problems"].append(f"LITERAL_PIN_MISMATCH: commitments.literals.trees[{t!r}] is missing (derived root {want['trees'][t]['root'][:16]}…)")
            continue
        for k in ("root", "leaf_count", "ids_digest"):
            if str(got_trees[t].get(k)) != str(want["trees"][t][k]):
                out["problems"].append(f"LITERAL_PIN_MISMATCH: commitments.literals.trees[{t!r}].{k} = {got_trees[t].get(k)!r}, the pinned program derives {want['trees'][t][k]!r}")
    if not out["problems"]:
        out.update(ok=True, status="PASS")
    return out


#: [E9, SCALE F3 2026-09-12] a pinned MUFU table cannot be rebuilt because the verifier lacks its fixture file (a VERIFIER INPUT: the tables are not
#: shipped in the record) -- the record is REFUSED by this name at `V5.public_trees`; an honest record verifies once the fixtures are present.
TABLE_FIXTURE_MISSING = "TABLE_FIXTURE_MISSING"


def public_trees(inst_doc: dict[str, Any] | None, commitments: dict[str, Any] | None, *, program: Any | None = None, literal_table: "LiteralTable | None" = None,
                 tables_pin: dict[str, Any] | None = None) -> tuple[dict[str, tuple[Any, list[str]]], list[str]]:
    """The VERIFIER-DERIVED public trees a statement may reference, for ``expected.coverage_inputs(public_trees=...)``: (a) D23's MUFU tables
    (``commitments.tables``: E2's registry rebuilds each pinned table and checks its root against the pin -- ``TableWords.tree``; ids in rank order,
    ``table/<name>/chunk=<k>``), (b) the ``prescribed-literal/v1`` trees (:class:`LiteralTable`, roots computed from the program).  Returns
    ``(trees, problems)``; a problem names the tree.  Nothing pinned -> ``({}, [])``.  [Finding, E9 night 05:30Z: without (a) a statement of an FTZ /
    MUFU kind refused ``root 'table/mufu/<name>' is not a committed tree of J`` at ``check_expected_positions``.]"""
    from veritor.interface.expected import CommittedTree

    out: dict[str, tuple[Any, list[str]]] = {}
    problems: list[str] = []
    pins = dict(tables_pin) if tables_pin else dict((((inst_doc or {}).get("commitments") or {}).get("tables")) or ((commitments or {}).get("tables")) or {})
    if pins:
        try:
            from e2e.interior.tables import TableWords

            tw = TableWords(pins)
            for name in sorted(pins):
                try:
                    t = tw.tree(name)
                except FileNotFoundError as e:
                    # [E9, SCALE F3] the table is rebuilt from a FIXTURE the verifier must hold (fixtures/W11*/.../mufu_*_delta_int8.xz, a
                    # verifier input listed in record_v5/reverify_clean.sh's header): absent -> this NAMED refusal (reverify: V5.public_trees FAIL),
                    # not the misattributed "root 'table/mufu/<name>' is not a committed tree of J" at V5.positions with the cause only in coverage.json
                    problems.append(f"{TABLE_FIXTURE_MISSING}: table/{name}: fixture file not present: {getattr(e, 'filename', None) or str(e)[:200]}")
                    continue
                except (KeyError, ValueError, OSError) as e:
                    problems.append(f"table/{name}: {type(e).__name__}: {str(e)[:200]}")
                    continue
                ids = [lf.canonical_id for lf in t.layout.leaves]
                out[str(t.tree_id)] = (CommittedTree(t.root.hex(), int(t.leaf_count), LITERAL_TREE_FRAMING), ids)
        except ImportError as e:
            problems.append(f"tables: E2's e2e.interior.tables is not importable ({e}); the pinned tables cannot be rebuilt")
    lt = literal_table
    if lt is None and program is not None and literal_pin(inst_doc) is not None:
        lt = LiteralTable.from_program(program)
    if lt is not None:
        out.update(lt.committed())
    return out, problems


# ------------------------------------------------------------ [E9, NIGHT G1 2026-09-12] gather-equality/v1: the verifier-covered Gather units (E7 D2)
#: E7's ruling (night report 05:35Z, PROPOSED): `GatherBf16x<V>_v1|batch` (`h[j] = table[tok][j]`, no arithmetic) is VERIFIER-COVERED, no proof:
#: the verifier opens the token word and the weights word the token selects (E3's positions: operand 0 = the token, operands 1..V = the column
#: of the embedding table in vocab order -- `verity_vllm.registry.prims.gather_bf16`) and compares with the VU's committed output word; the
#: primitive's total extension (token outside [0, V) -> 0x7FFF) is checked as written.  ONE function for both entry points (`V6.gather`).
GATHER_RULE = "gather-equality/v1"
GATHER_OUT_OF_RANGE = 0x7FFF
#: kind -> the rule that covers it without a proof (E7 mirrors this as `builders.VERIFIER_COVERED`; a test asserts the two agree)
VERIFIER_COVERED_RULES: dict[str, str] = {"GatherBf16x{V}_v1|batch": GATHER_RULE}


def verifier_covered_rule(kind: str) -> str | None:
    """The by-name rule under which the verifier covers ``kind`` without a proof object, or ``None``."""
    k = str(kind)
    if k.startswith("GatherBf16x") and k.endswith("_v1|batch") and k[len("GatherBf16x"):-len("_v1|batch")].isdigit():
        return GATHER_RULE  # the on-demand family: one kind per vocab size
    return VERIFIER_COVERED_RULES.get(k)


def gather_vocab(kind: str) -> int:
    k = str(kind)
    return int(k[len("GatherBf16x"):].split("_", 1)[0])


def gather_equality_check(units: Sequence[dict[str, Any]], positions: Any, record_handle: Any, *, commitments: dict[str, Any] | None = None,
                          weights_word: Any = None) -> dict[str, Any]:
    """``V6.gather`` over the selected Gather VUs ``units`` (located units of retained RUs): for each, E3's ``positions.statement(u)`` gives the
    token position (operand 0), the V column positions (operands 1..V) and the output; the verifier opens the token leaf and the selected weights
    leaf through its OWN authenticated openings (``record_handle.open_leaf`` + ``authenticate_opening`` against the committed roots) and requires
    ``out == column[tok]`` (``0x7FFF`` outside ``[0, V)``).  Returns ``{"ok", "rule", "checked", "rows": [...], "problems": [...]}``; any problem is
    a refusal by name (``GATHER_MISMATCH`` / ``GATHER_UNRESOLVED`` / ``GATHER_OPENING``).  When the record ships no weights bytes (the TINY records:
    ``weights/values.bin`` absent; the weights root is the REGISTERED root, bound to the pinned checkpoint by the step-2 weights-root check), the
    selected embedding word is read through ``weights_word(tensor_cid, offset) -> (word, source)`` -- the verifier's OWN checkpoint copy (reverify's
    ``--checkpoint``, sha256-pinned; the same reader V6.inputs uses); without it such a VU is ``GATHER_UNRESOLVED`` by name, never assumed."""
    out: dict[str, Any] = {"ok": True, "rule": GATHER_RULE, "checked": 0, "rows": [], "problems": []}
    if not units:
        return out
    if positions is None or record_handle is None:
        out["ok"] = False
        out["problems"].append(f"GATHER_UNRESOLVED: {len(units)} Gather VU(s) selected but the verifier has no position oracle / record handle to open the words")
        return out
    from verity_vllm.typed_obligation_export import authenticate_opening, leaf_words, words_per_element

    comm = commitments if commitments is not None else getattr(record_handle, "commitments", None)
    cache: dict[str, Any] = {}

    sources: dict[str, str] = {}

    def read(pos: Any) -> int:
        o = cache.get(pos.leaf)
        if o is None:
            try:
                o = record_handle.open_leaf(str(pos.leaf))
            except Exception as e:  # noqa: BLE001 -- a weights leaf without shipped bytes falls back to the verifier's checkpoint copy
                if str(pos.tree) != "weights" or weights_word is None:
                    raise
                reader = getattr(record_handle, "reader", None)
                entry, block = reader.weights.locate(str(pos.leaf)) if reader is not None else (None, None)
                if entry is None:
                    raise ValueError(f"{pos.leaf}: weights leaf not opened ({type(e).__name__}) and not located in the weights index") from e
                row_words = 1
                for x in tuple(entry.shape)[1:]:
                    row_words *= int(x)
                first_row = int(block[0]) if block is not None else 0
                flat = int(pos.offset[-1]) if pos.offset else 0
                w, src = weights_word(str(entry.canonical_id), first_row * row_words + flat)
                if w is None:
                    raise ValueError(f"{pos.leaf}: weights word unavailable from the verifier's checkpoint ({src})") from e
                sources[str(pos.leaf)] = str(src)
                return int(w)
            if comm is not None:
                authenticate_opening(comm, o)
            cache[pos.leaf] = o
        if str(o.tree) != str(pos.tree):
            raise ValueError(f"{pos.leaf}: committed in {o.tree}, expected in {pos.tree}")
        words = leaf_words(o)  # 16-bit halves, low first (an int32 token is two halves)
        n = words_per_element(o.dtype)
        flat = int(pos.offset[-1]) if pos.offset else 0
        if not 0 <= n * flat + n - 1 < len(words):
            raise ValueError(f"{pos.leaf}: element {flat} outside the leaf ({len(words) // n} elements)")
        return sum(int(words[n * flat + i]) << (16 * i) for i in range(n))

    for u in units:
        vu = int(u["vu_index"])
        kind = str(u["kind"])
        row: dict[str, Any] = {"vu_index": vu, "kind": kind}
        try:
            V = gather_vocab(kind)
            st = positions.statement(u)
            if len(st.operands) != 1 + V or len(st.outputs) != 1:
                raise ValueError(f"statement has {len(st.operands)} operands / {len(st.outputs)} outputs, the primitive takes (token, column[{V}]) -> 1")
            tok_pos, out_pos = st.operands[0], st.outputs[0]
            tok = read(tok_pos)
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            if 0 <= t < V:
                sel = st.operands[1 + t]
                want = read(sel)
                row.update(token=t, selected=list(sel.key), selected_source=sources.get(str(sel.leaf), "record opening (authenticated)"))
            else:
                want = GATHER_OUT_OF_RANGE
                row.update(token=t, selected=None, out_of_range=True)
            got = read(out_pos)
            row.update(expected=want, committed=got, token_position=list(tok_pos.key), output_position=list(out_pos.key), ok=(want == got))
            if want != got:
                out["problems"].append(f"GATHER_MISMATCH: VU {vu} ({kind}): token {t} selects {sel.key if 0 <= t < V else 'out of range'} = 0x{want:04x}, "
                                       f"the committed output {out_pos.key} is 0x{got:04x}")
        except Exception as e:  # noqa: BLE001 -- named per VU, never a silent skip
            code = "GATHER_OPENING" if "authentic" in str(e).lower() or "root" in str(e).lower() else "GATHER_UNRESOLVED"
            row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
            out["problems"].append(f"{code}: VU {vu} ({kind}): {type(e).__name__}: {str(e)[:200]}")
        out["rows"].append(row)
        out["checked"] += 1
    out["ok"] = not out["problems"]
    return out


# ------------------------------------------------------------ [E9, NIGHT D35 2026-09-12] V3.trees_rehashed: every shipped tree's root from its leaf FILES
#: T1's finding F1: a record whose committed-word LEAF FILE was altered (with the index's leaf sha256 re-hashed) was ACCEPTED by both entry
#: points because the tree root was TAKEN from commitments.json and never recomputed from the shipped bytes.  Ruling (INTEGRATOR D35): both
#: entry points recompute EVERY shipped tree's root from its leaf files (leaf hashes from bytes -> root) and require it to equal the committed
#: root; a tree whose leaf files are not shipped says so by name (`NOT_APPLICABLE:<tree>`), never silently.  ONE function, dispatching on the
#: on-disk index format BY NAME: DemoCore `veritor/demo-core/{boundary,interior}-index/v1` (`leaf-<i>.bin`), E2's interior index
#: (`layout.rule = interior-layout/...`, `e2e.interior.tree.load_tree(recompute=True)`), the capture layout (`manifest.json`;
#: `verity_vllm.commit.TreeCache(recompute_leaves=True)`), and the weights tree (bytes shipped -> rehashed; else NOT_APPLICABLE by name).
TREES_REHASHED_RULE = "trees-rehashed/v1"
TREE_UNSHIPPED = "TREE_UNSHIPPED"
TREE_ROOT_MISMATCH = "TREE_ROOT_MISMATCH"
TREE_INDEX_MALFORMED = "TREE_INDEX_MALFORMED"
DEMO_CORE_INDEX_SCHEMAS = ("veritor/demo-core/boundary-index/v1", "veritor/demo-core/interior-index/v1")


def _rehash_demo_core_index(d: Path, index: dict[str, Any], tree_id: str) -> tuple[str, int, str, int]:
    """E8's writer (`demo_core/record.py`): entry i's bytes are `leaf-<i>.bin`; leaf = `verity-vllm/leaf/v1` over (id, dtype, shape, bytes);
    the tree is `verity_vllm.commit.MerkleTree` (ids in tree order).  Returns (root hex, leaf_count, ids_digest hex, nbytes)."""
    from verity_vllm import hashing
    from verity_vllm.commit import MerkleTree

    leaves: dict[str, bytes] = {}
    nbytes = 0
    for i, e in enumerate(index.get("entries") or []):
        f = d / f"leaf-{i}.bin"
        if not f.exists():
            raise FileNotFoundError(f"{tree_id}: entry {i} ({e.get('canonical_id')}) has no leaf file {f.name}")
        b = f.read_bytes()
        nbytes += len(b)
        leaves[str(e["canonical_id"])] = hashing.leaf_hash(str(e["canonical_id"]), str(e["dtype"]), tuple(int(x) for x in e["shape"]), b)
    t = MerkleTree(leaves, tree_id)
    return t.root.hex(), t.leaf_count, t.ids_digest.hex(), nbytes


def trees_rehashed(selective_dir: str | os.PathLike[str], commitments: Mapping[str, Any], *, tree_ids: Sequence[str] | None = None,
                   interior: bool = True, weights_registry_bound: bool = True) -> dict[str, Any]:
    """``V3.trees_rehashed``: for every committed tree (``commitments.trees`` and, with ``interior``, ``commitments.interior_roots``) recompute
    the root from the shipped leaf FILES and compare with the committed root / leaf_count / ids_digest.  Returns ``{"ok", "rule", "trees":
    {tree: {"status": PASS|FAIL|NOT_APPLICABLE, "reason", "committed", "recomputed", "leaf_count", "nbytes"}}, "problems": [...], "nbytes",
    "seconds", "not_applicable": [...]}``.  A tree with no leaf files on disk FAILS ``TREE_UNSHIPPED`` unless the record says so by name
    (``commitments.trees[t].leaf_files == "not shipped"`` or ``<selective>/leaf_files.json#not_shipped``), or it is the weights tree whose bytes
    are never shipped (its root is bound to the registry by the weights-root check) -- ``NOT_APPLICABLE:<tree>`` with the reason."""
    import time as _time

    sel = Path(selective_dir)
    t0 = _time.perf_counter()
    out: dict[str, Any] = {"ok": True, "rule": TREES_REHASHED_RULE, "trees": {}, "problems": [], "nbytes": 0, "not_applicable": []}
    manifest_ns: set[str] = set()
    mf = sel / "leaf_files.json"
    if mf.exists():
        try:
            manifest_ns = {str(x) for x in (json.loads(mf.read_text()).get("not_shipped") or [])}
        except (OSError, ValueError, AttributeError) as e:
            out["problems"].append(f"{TREE_INDEX_MALFORMED}: leaf_files.json: {type(e).__name__}: {str(e)[:120]}")
    reader = None
    cache = None

    def capture_cache():
        nonlocal reader, cache
        if cache is None:
            from verity_vllm.capture import CaptureReader
            from verity_vllm.commit import TreeCache
            reader = CaptureReader(sel)
            cache = TreeCache(reader, recompute_leaves=True)
        return cache

    def compare(tid: str, rec: Mapping[str, Any], root: str, count: int, ids_digest: str | None, nbytes: int, how: str) -> None:
        row = {"status": "PASS", "how": how, "committed": str(rec.get("root")), "recomputed": root, "leaf_count": count, "nbytes": nbytes}
        out["nbytes"] += nbytes
        problems = []
        if str(rec.get("root")) != root:
            problems.append(f"root committed {str(rec.get('root'))[:16]} != recomputed {root[:16]}")
        if int(rec.get("leaf_count", -1)) != int(count):
            problems.append(f"leaf_count committed {rec.get('leaf_count')} != {count} leaf files")
        if ids_digest is not None and rec.get("ids_digest") is not None and str(rec.get("ids_digest")) != ids_digest:
            problems.append(f"ids_digest committed {str(rec.get('ids_digest'))[:16]} != recomputed {ids_digest[:16]}")
        if problems:
            row["status"] = "FAIL"
            row["reason"] = "; ".join(problems)
            out["problems"].append(f"{TREE_ROOT_MISMATCH}: {tid}: " + "; ".join(problems))
        out["trees"][tid] = row

    def not_applicable(tid: str, reason: str) -> None:
        out["trees"][tid] = {"status": "NOT_APPLICABLE", "reason": reason}
        out["not_applicable"].append(f"NOT_APPLICABLE:{tid}: {reason}")

    trees = dict(commitments.get("trees") or {})
    wanted = list(tree_ids) if tree_ids is not None else sorted(trees)
    for tid in wanted:
        rec = trees.get(tid)
        if rec is None:
            out["problems"].append(f"{TREE_UNSHIPPED}: {tid}: not a committed tree")
            continue
        try:
            d = sel / tid
            idx = d / "index.json"
            if tid == "weights":
                if int(rec.get("leaf_count", 0)) == 0:
                    from verity_vllm.commit import MerkleTree
                    t = MerkleTree({}, "weights")
                    compare(tid, rec, t.root.hex(), 0, t.ids_digest.hex(), 0, "empty tree")
                    continue
                wdir = sel / "weights"
                if (wdir / "values.bin").exists() and (sel / "manifest.json").exists():
                    t = capture_cache().tree("weights")
                    compare(tid, rec, t.root.hex(), t.leaf_count, t.ids_digest.hex(), (wdir / "values.bin").stat().st_size, "capture weights/values.bin rehashed")
                elif weights_registry_bound:
                    not_applicable(tid, "weights bytes are not shipped in the record (registered root; bound to the registry by the weights-root check, never to files here)")
                else:
                    out["problems"].append(f"{TREE_UNSHIPPED}: weights: no weights/values.bin and the weights root is not registry-bound")
                continue
            if idx.exists():
                index = json.loads(idx.read_text())
                schema = str(index.get("schema", ""))
                if schema in DEMO_CORE_INDEX_SCHEMAS:
                    root, count, idd, nbytes = _rehash_demo_core_index(d, index, tid)
                    compare(tid, rec, root, count, idd, nbytes, schema)
                    continue
                if str((index.get("layout") or {}).get("rule", "")).startswith("interior-layout/"):
                    from e2e.interior.tree import load_tree
                    t = load_tree(d, recompute=True)
                    compare(tid, rec, t.root.hex(), t.leaf_count, t.ids_digest.hex(), sum((d / str(r.get("file", f"leaf-{i:06d}.bin"))).stat().st_size for i, r in enumerate(index.get("leaves") or [])), index["layout"]["rule"])
                    continue
            if (sel / "manifest.json").exists() and (tid.startswith("ru/") or tid.startswith("boundary/")):
                t = capture_cache().tree(tid)
                nbytes = 0
                try:
                    parts = tid.split("/")
                    if parts[0] == "ru":
                        nbytes = sum(int(v.nbytes) for v in reader.ru(parts[1], int(parts[2])).values())  # type: ignore[union-attr]
                    else:
                        nbytes = sum(int(v.nbytes) for v in reader.boundary(parts[1]).leaves(reader.profile_id))  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001 -- byte count is reporting only
                    nbytes = 0
                compare(tid, rec, t.root.hex(), t.leaf_count, t.ids_digest.hex(), nbytes, "capture layout rehashed (TreeCache recompute_leaves=True)")
                continue
            if str(rec.get("leaf_files", "")) == "not shipped" or tid in manifest_ns:
                not_applicable(tid, f"the record declares this tree's leaf files not shipped ({'commitments.trees[].leaf_files' if rec.get('leaf_files') else 'leaf_files.json'})")
                continue
            out["problems"].append(f"{TREE_UNSHIPPED}: {tid}: no leaf files / index on disk under {sel.name}/ and no by-name declaration that they are not shipped")
            out["trees"][tid] = {"status": "FAIL", "reason": TREE_UNSHIPPED}
        except Exception as e:  # noqa: BLE001 -- named per tree
            out["problems"].append(f"{TREE_INDEX_MALFORMED}: {tid}: {type(e).__name__}: {str(e)[:160]}")
            out["trees"][tid] = {"status": "FAIL", "reason": f"{type(e).__name__}: {str(e)[:160]}"}
    if interior:
        from e2e.interior.tree import INTERIOR_ROOTS_KEY
        for key, rec in sorted((commitments.get(INTERIOR_ROOTS_KEY) or {}).items()):
            tid = f"interior/{key}"
            if tid in out["trees"]:
                continue
            d = sel / "interior" / key
            try:
                if not (d / "index.json").exists():
                    if tid in manifest_ns or str(rec.get("leaf_files", "")) == "not shipped":
                        not_applicable(tid, "declared not shipped")
                    else:
                        out["problems"].append(f"{TREE_UNSHIPPED}: {tid}: no interior/{key}/index.json")
                        out["trees"][tid] = {"status": "FAIL", "reason": TREE_UNSHIPPED}
                    continue
                index = json.loads((d / "index.json").read_text())
                if str(index.get("schema", "")) in DEMO_CORE_INDEX_SCHEMAS:
                    root, count, idd, nbytes = _rehash_demo_core_index(d, index, tid)
                else:
                    from e2e.interior.tree import load_tree
                    t = load_tree(d, recompute=True)
                    root, count, idd = t.root.hex(), t.leaf_count, t.ids_digest.hex()
                    nbytes = sum((d / str(r.get("file", f"leaf-{i:06d}.bin"))).stat().st_size for i, r in enumerate(index.get("leaves") or []))
                compare(tid, rec, root, count, idd, nbytes, "interior index rehashed (E2 load_tree recompute=True)")
                if str(index.get("root")) != root:
                    out["problems"].append(f"{TREE_ROOT_MISMATCH}: {tid}: index.json#root {str(index.get('root'))[:16]} != recomputed {root[:16]}")
                    out["trees"][tid]["status"] = "FAIL"
            except Exception as e:  # noqa: BLE001
                out["problems"].append(f"{TREE_INDEX_MALFORMED}: {tid}: {type(e).__name__}: {str(e)[:160]}")
                out["trees"][tid] = {"status": "FAIL", "reason": f"{type(e).__name__}: {str(e)[:160]}"}
    out["seconds"] = round(_time.perf_counter() - t0, 3)
    out["ok"] = not out["problems"]
    out["summary"] = {"trees": len(out["trees"]), "pass": sum(1 for r in out["trees"].values() if r["status"] == "PASS"),
                      "fail": sum(1 for r in out["trees"].values() if r["status"] == "FAIL"), "not_applicable": len(out["not_applicable"]), "nbytes": out["nbytes"], "seconds": out["seconds"]}
    return out
