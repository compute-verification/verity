"""Live C2 -- value correspondence by ORACLE COMPARE.

The Commit's C1 (coverage) and C3 (openings) cannot see a relabel, a same-name byte swap or a late read (every
one of them got COMMIT PASS with coverage 12943/12943 and 79/79 openings verified).  C2 asks: are the COMMITTED bytes of
each identity the value the Program produced for that identity?  The reference here is the Match stage of the SAME attempt: the m1
capture (`verity_vllm.pipeline.m1_capture`, a separate process and a separate acquisition path -- torch dispatcher observer + D2H
snapshots -- sharing nothing with the native collector) recorded, for every snapshotted engine step, the bytes of every dispatcher
call's arguments and outputs (`capture/snapshots.json` + `capture/values/<step>/<sha>.bin`, spanned storage bytes, content-addressed),
and the fold's runtime-correspondence record (`instances.jsonl`) says, per Program instance, WHICH event and WHICH storage view (alloc,
offset, shape, dtype) is its output, attributed to a request and a token position.  So the oracle value of identity
(request r, step s, op_path, output_member) is: the fold's instances of op_path at step s attributed to r, in row order, each read at
its output view from the snapshot of its producing event.  That is the Program-evaluated value of record for the Match verdict, read
from the record -- the collector's own bytes never enter the expected side.

    compared identity  : committed bytes (opened from the collector's retained copy of the member and verified against the run root,
                         `OpenedReader`, sliced at the binding-map entry's element offset + the identity's request-local range) == oracle
                         bytes                                                                    -> equal | MISMATCH (named; an opening
                         that does not verify is a mismatch)
    sampled_token_ids  : the committed id == the capture's SampleEnd id for that request at that step (every step, no snapshot needed)
    not compared       : identities of a step the Match did not snapshot (row_pod MATCH_SNAP_STEPS, default 0,1), of a family the
                         Match has no value for (the FA2/FA3 M1 kernel-interior stream: its C2 is the tap's own bit-exact xcheck,
                         capacity FA2-01), registered inputs (weights: digest-pinned by the Build), or an op the fold did not instance
                         -> counted per reason, never silently dropped

Result: `result` is False on ANY mismatch (the verdict fails closed naming the identity), True when every comparable identity of every
snapshotted step was compared equal AND every step was snapshotted (complete live C2), else None with the coverage stated (partial:
the finalizer grades C2 'not checked' / partial).  Provenance of the oracle (Match dir, capture created_utc,
profile id, fold digest, snapshot steps, tokens.json sha256) is recorded so a reviewer can tell the reference is this attempt's Match.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from typing import Any

# handed on: other modules and tests read the Match record's reader and the producers through this module
from verity_vllm.check.match_oracle import _ITEMSIZE, MatchOracle
from verity_vllm.check.match_oracle import _ROPE_MEMBER as _ROPE_MEMBER
from verity_vllm.check.match_oracle import _contiguous as _contiguous
from verity_vllm.check.match_oracle import _port_view as _port_view
from verity_vllm.check.match_oracle import _prod as _prod
from verity_vllm.check.match_oracle import oracle_tie_facts as oracle_tie_facts
from verity_vllm.check.producers import (
    NO_ORACLE_FAMILY_PREFIXES as NO_ORACLE_FAMILY_PREFIXES,
)
from verity_vllm.check.producers import _family_of, _query_family_of
from verity_vllm.check.producers import merge_producers as merge_producers
from verity_vllm.check.producers import producers_of as producers_of
from verity_vllm.check.producers import producers_of_programs as producers_of_programs
from verity_vllm.check.producers import program_producer_facts as program_producer_facts
from verity_vllm.check.producers import promote_handed_down as promote_handed_down
from verity_vllm.check.producers import query_families_of as query_families_of
from verity_vllm.check.producers import runtime_rows_of as runtime_rows_of

# re-exported: the engine-layer tp worker may import this module but not commit/ (tests/lint P9)
from verity_vllm.commit.opened import (  # noqa: F401  -- re-exported
    OPENED_METHOD,
    ORDINAL_SEP,
    OpenedReader,
    OpeningNotVerified,
    meta_by_name,
)
from verity_vllm.records.schema_ids import ORACLE_COMPARE_SCHEMA

__all__ = ["MatchOracle", "OpenedReader", "OpeningNotVerified", "SCHEMA", "entry_read_name", "merge_producers",
           "oracle_compare", "producers_of", "producers_of_programs", "promote_handed_down", "query_families_of",
           "runtime_rows_of", "same_name_invocations"]

SCHEMA = ORACLE_COMPARE_SCHEMA
METHOD = "oracle-compare (Match capture of the same attempt: snapshots at the fold's instance output views + SampleEnd ids)"


def bound_entries(bmap: dict[str, Any]) -> list[dict[str, Any]]:
    """The binding map's entries (`commit.binding.build_binding_map`, read back from its file); a map that states none binds nothing."""
    entries = bmap.get("entries")
    return entries if entries else []


def _foreign_oracle(oracle: MatchOracle, bmap: dict[str, Any], wcheck: dict[str, Any]) -> dict[str, Any]:
    """The result of a Match record whose request table is not this workload's: False, nothing compared."""
    why = "foreign oracle: the Match record is not this workload's -- " + "; ".join(wcheck["problems"][:4])
    return {"schema": SCHEMA, "method": "oracle-compare (Match capture of the same attempt: snapshots + fold + capture log + tokens.json)",
            "result": False, "grade": "foreign-oracle", "why": why, "complete": False, "compared": 0, "equal": 0, "mismatch_n": 0, "mismatch": [],
            "compared_by_family": {}, "not_compared": {"foreign oracle": len(bound_entries(bmap))}, "steps_compared": [], "steps_not_snapshotted": [],
            "attribution": {"complete": False, "pairs": 0, "unverified_n": 0, "ambiguous_steps": []}, "workload_check": wcheck, "oracle": oracle.provenance()}


def _ambiguous_steps(pairs_seen: set[tuple[int, str]], attrib: dict[tuple[int, str], dict[str, Any]], snapshot_steps: Any) -> list[int]:
    """Steps not snapshotted where two requests carry the same ids AND the same sampled id: a swap there is not attributable."""
    ambiguous: list[int] = []
    for s in sorted({s for s, _ in pairs_seen}):
        if s in snapshot_steps:
            continue
        sig = Counter((attrib.get((s, r), {}).get("ids_sha"), attrib.get((s, r), {}).get("sampled_value")) for ss, r in pairs_seen if ss == s)
        if any(n > 1 for n in sig.values()):
            ambiguous.append(s)
    return ambiguous


def oracle_compare(oracle: MatchOracle, bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None], *,
                   max_report: int = 12, producers: dict[tuple[str, str], dict[str, Any]] | None = None,
                   workload: dict[str, Any] | None = None, aliases: dict[str, Any] | None = None,
                   query_families: dict[str, int] | None = None, own: frozenset[str] = frozenset(),
                   runtime_rows: dict[tuple, tuple[int, int]] | None = None) -> dict[str, Any]:
    """Compare every binding-map entry's committed bytes with the Match oracle.  `committed(step, name, byte_lo, byte_hi)` returns the
    bytes of member `name` at `step` in [byte_lo, byte_hi) (None if not retained) -- in the Commit an `OpenedReader`, whose
    `OpeningNotVerified` makes the entry a mismatch.  `own` names what the run's sources committed that no module produced (their
    `own_names`), so no Match value: not compared, counted as hidden_stream.  `runtime_rows`
    (`runtime_rows_of(manifest)`): an identity that is the runtime's rows r0.. is compared with the fold's rows from r0.  `producers` (producers_of(manifest))
    names each member's producing Program op so exploded-module folds are read at the producer's instances.  `workload` (the run's declared
    requests) is checked against the oracle's request table BY NAME first: a foreign Match record is result False 'foreign oracle' with
    nothing compared.

    On the ORACLE side, a shared module's invocation k >= 1 (the binding map's `invocation`) is a Program op of its own:
    the fold's instances are under the per-layer Program paths, so the oracle is read at the path the manifest's `op_path_aliases`
    (`aliases`, the table the replay resolves through) gives for (canonical op, k) -- `model.layers.k.self_attn.rotary_emb`.  Before this,
    every invocation was read at the CANONICAL op (= layer 0's fold instance) while the committed side reads invocation k's
    own bytes: layers 1..N would compare as FALSE MISMATCHES (and before the reader was fixed they compared EQUAL vacuously, layer 0 against layer 0).
    Without the table, invocation k >= 1 is NOT COMPARED, by name (partial) -- never equal by collapse, never a mismatch by mis-addressing."""
    inv_alias = {(str(v[0]), int(v[1])): str(k) for k, v in (aliases or {}).items()}       # (canonical op, invocation) -> Program op_path
    wcheck = oracle.workload_check(workload)
    if wcheck.get("ok") is False:
        return _foreign_oracle(oracle, bmap, wcheck)
    return _report(oracle, wcheck, max_report, query_families,
                   **_walk(oracle, bmap, committed, producers, inv_alias, own, runtime_rows))


def _walk(oracle: MatchOracle, bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None],
          producers: dict[tuple[str, str], dict[str, Any]] | None, inv_alias: dict[tuple[str, int], str], own: frozenset[str],
          runtime_rows: dict[tuple, tuple[int, int]] | None) -> dict[str, Any]:
    """`oracle_compare`'s pass over the binding map's entries: what it compared, found equal, found different and did not compare,
    each tally under the name `_report` takes it by."""
    compared = equal = 0
    mism: list[dict] = []
    not_cmp: Counter = Counter()
    not_cmp_fam: Counter = Counter()
    by_fam_cmp: Counter = Counter()
    not_cmp_qfam: Counter = Counter()                  # no-oracle entries per Q(P) family (fa2_hidden_m1_stream / moe_block_stream / weights)
    shared_inv: Counter = Counter()                    # invocations >= 1 of a shared name: compared at their Program op / not compared
    steps_cmp: set[int] = set()
    steps_seen: set[int] = set()
    seen: set[tuple] = set()
    # REQUEST ATTRIBUTION: per (step, request) the two per-request id families the ids oracle covers at EVERY
    # step (tokens.json + StepStart rows; SampleEnd) -- a relabel of equal-row-count requests on one engine step (b2swapeq) moves r0's
    # rows to r1's slots: the bound `in` row / sampled id of r0 then carries r1's token, which THIS request's oracle value refutes.
    attrib: dict[tuple[int, str], dict[str, Any]] = {}
    pairs_seen: set[tuple[int, str]] = set()
    padding_not_cmp: Counter = Counter()               # bot-by-enablement entries per Q(P) family: no execution, no oracle value
    padding_pairs: set[tuple[int, str]] = set()
    name_inv = same_name_invocations(bmap)                                       # (step, member name) -> invocations bound
    for e in bound_entries(bmap):
        ident = e.get("identity")   # an entry that binds no identity has no step, and is skipped as one whose identity states none
        if not ident or int(ident.get("step", -1)) < 0:
            continue
        step = int(ident["step"])
        if e.get("activity") == "bot-by-enablement":
            # a PADDING entry (a padded record's inactive stratum, LIFTING-SPEC v0.7 B9.1) materializes an
            # identity the executed prefix of record left INACTIVE: nothing executed, so the Match oracle holds no value for it and its
            # (step, request) pair is not one request attribution can or must verify -- its word is checked against bot_word by the
            # padded Commit's inactive-stratum check, not here.  Counted apart, per Q(P) family, never as 'identity without a byte
            # range' nor as an unverified attribution pair (the first #3 pod run graded 'partial' on exactly these 7,665 + 21 pairs).
            padding_not_cmp[str(e.get("family") or _query_family_of(e))] += 1   # a padding entry carries its Q(P) family (padding_entries)
            padding_pairs.add((step, str(ident.get("request_id", "r0"))))
            continue
        fam = "hidden_stream" if e.get("name") in own else _family_of(e)
        steps_seen.add(step)
        if fam in ("weights", "hidden_stream"):
            not_cmp["no oracle family: " + fam] += 1
            not_cmp_fam[fam] += 1
            not_cmp_qfam[_query_family_of(e)] += 1                   # per Q(P) family, in binding entries (identities below)
            continue
        rid = str(ident.get("request_id", "r0"))
        key = (step, rid, ident.get("op_path"), ident.get("output_member"), ident.get("invocation"), tuple(ident.get("element_range") or ()))
        if key in seen:
            continue
        seen.add(key)
        if rid != "*":
            pairs_seen.add((step, rid))
        enc = e.get("encoding")   # the member's dtype and shape; an entry that states no dtype has no byte range to read
        dtype = str(enc.get("dtype") or "") if enc else ""
        isz = _ITEMSIZE.get(dtype)
        lo, hi = ident.get("element_range") or [0, None]
        if isz is None or hi is None or int(hi) <= int(lo):
            not_cmp["identity without a byte range (dtype/range)"] += 1
            not_cmp_fam[fam] += 1
            continue
        eo = int(e.get("element_offset") or 0)
        blo, bhi = (eo + int(lo)) * isz, (eo + int(hi)) * isz
        try:
            got = committed(step, entry_read_name(e, name_inv.get((step, str(e.get("name"))), 1)), blo, bhi)   # exact invocation
        except OpeningNotVerified as exc:
            compared += 1
            by_fam_cmp[fam] += 1
            steps_cmp.add(step)
            mism.append({"identity": ident, "family": fam, "name": e.get("name"), "why": f"opened value not verified: {exc}"})
            continue
        if got is None:
            not_cmp["committed bytes not retained on host (retain=none, or an excluded family under retain=host + exclude)"] += 1
            not_cmp_fam[fam] += 1
            continue
        # the Program op of THIS invocation: 0 = the op_path itself; k >= 1 = the alias table's path, else not compared
        inv_k = int(ident.get("invocation") or 0)
        op_k = str(ident.get("op_path"))
        if inv_k > 0:
            if (alias := inv_alias.get((op_k, inv_k))) is None:
                not_cmp[f"shared-module invocation >= 1 with no Program path in the alias table (op {ident.get('op_path')}): not compared"] += 1
                not_cmp_fam[fam] += 1
                shared_inv["not_compared"] += 1
                continue
            op_k = alias
            shared_inv["compared_at_program_op"] += 1
        if fam == "sampled_token_ids":
            oid, why = oracle.sampled_id(step, rid)
            if oid is None:
                not_cmp[why] += 1
                not_cmp_fam[fam] += 1
                continue
            cid = int.from_bytes(got[:isz], "little", signed=True)
            compared += 1
            by_fam_cmp[fam] += 1
            steps_cmp.add(step)
            a = attrib.setdefault((step, rid), {})
            a["sampled"] = cid == oid
            a["sampled_value"] = oid
            if cid == oid:
                equal += 1
            else:
                mism.append({"identity": ident, "family": fam, "name": e.get("name"), "committed_id": cid, "oracle_id": oid})
            continue
        if fam == "input_token_ids":
            ob, why = oracle.ids_bytes(step, rid, isz)
            n_inst = len(ob) // isz if ob else 0
        else:
            prod = (producers or {}).get((str(ident.get("op_path")), str(ident.get("output_member"))))
            ob, why, n_inst = oracle.member_bytes(step, op_k, str(ident.get("output_member")), rid,
                                                  producer=(prod if prod and prod.get("spec") else None), nbytes=bhi - blo)
        if ob is None:
            not_cmp[why] += 1
            not_cmp_fam[fam] += 1
            continue
        r0, n = (runtime_rows or {}).get((step, rid, str(ident.get("op_path")), str(ident.get("output_member")), inv_k), (0, 0))
        ob = ob[r0 * (len(got) // n):(r0 + n) * (len(got) // n)] if n and len(got) % n == 0 and len(ob) >= (r0 + n) * (len(got) // n) else ob
        compared += 1
        by_fam_cmp[fam] += 1
        steps_cmp.add(step)
        if fam == "input_token_ids":
            a = attrib.setdefault((step, rid), {})
            a["ids"] = ob == got
            a["ids_sha"] = hashlib.sha256(ob).hexdigest()[:16]
        if len(ob) != len(got):
            mism.append({"identity": ident, "family": fam, "name": e.get("name"), "why": f"size: committed {len(got)} B, oracle {len(ob)} B over {n_inst} instances"})
        elif ob == got:
            equal += 1
        else:
            first = next((i for i in range(0, len(ob), isz) if ob[i:i + isz] != got[i:i + isz]), None)
            mism.append({"identity": ident, "family": fam, "name": e.get("name"), "why": f"bytes differ (first element {first // isz if first is not None else None} of {len(ob) // isz})",
                         "committed_sha256": hashlib.sha256(got).hexdigest()[:16], "oracle_sha256": hashlib.sha256(ob).hexdigest()[:16]})
    return {"compared": compared, "equal": equal, "mism": mism, "not_cmp": not_cmp, "not_cmp_fam": not_cmp_fam,
            "by_fam_cmp": by_fam_cmp, "not_cmp_qfam": not_cmp_qfam, "shared_inv": shared_inv, "steps_cmp": steps_cmp,
            "steps_seen": steps_seen, "attrib": attrib, "pairs_seen": pairs_seen, "padding_not_cmp": padding_not_cmp,
            "padding_pairs": padding_pairs}


def _report(oracle: MatchOracle, wcheck: dict[str, Any], max_report: int, query_families: dict[str, int] | None, *, compared: int,
            equal: int, mism: list[dict], not_cmp: Counter, not_cmp_fam: Counter, by_fam_cmp: Counter, not_cmp_qfam: Counter,
            shared_inv: Counter, steps_cmp: set[int], steps_seen: set[int], attrib: dict[tuple[int, str], dict[str, Any]],
            pairs_seen: set[tuple[int, str]], padding_not_cmp: Counter, padding_pairs: set[tuple[int, str]]) -> dict[str, Any]:
    """The oracle-compare document from `_walk`'s tallies: the result, its grade and why, and every count."""
    steps_not_snap = sorted(s for s in steps_seen if s not in oracle.snapshot_steps)
    complete = compared > 0 and not mism and not steps_not_snap and sum(v for k, v in not_cmp.items() if not k.startswith("no oracle family")) == 0
    result: bool | None = False if mism else (True if complete else None)
    # attribution: every (step, request) the map binds has BOTH id families compared and equal; steps where two requests carry the same
    # ids AND the same sampled id are ambiguous for a swap unless the step's values were snapshotted (then instance_outputs decide)
    attrib_missing = sorted(k for k in pairs_seen if not (attrib.get(k, {}).get("ids") and attrib.get(k, {}).get("sampled")))
    ambiguous = _ambiguous_steps(pairs_seen, attrib, oracle.snapshot_steps)
    attribution_complete = compared > 0 and not mism and not attrib_missing and not ambiguous
    grade = ("mismatch" if mism else "complete" if complete else "attribution-complete" if attribution_complete else "partial")
    why = None
    if mism:
        m = mism[0]
        i = m["identity"]
        why = (f"C2 oracle mismatch: {len(mism)} identities differ from the Match oracle (first: step {i.get('step')} {i.get('op_path')}/{i.get('output_member')} "
               f"inv {i.get('invocation')} req {i.get('request_id')} range {i.get('element_range')} -- {m.get('why') or ('id ' + str(m.get('committed_id')) + ' vs ' + str(m.get('oracle_id')))})")
    elif result is None:
        why = (f"{grade}: " + (f"steps {steps_not_snap[:8]}{'...' if len(steps_not_snap) > 8 else ''} not snapshotted by the Match (row_pod MATCH_SNAP_STEPS=all for a complete C2); " if steps_not_snap else "")
               + (f"request attribution unverified at {len(attrib_missing)} (step, request) pairs (first {attrib_missing[:4]}); " if attrib_missing else "")
               + (f"attribution ambiguous at steps {ambiguous[:8]} (equal ids + sampled id across requests, values not snapshotted); " if ambiguous else "")
               + "; ".join(f"{k} x{v}" for k, v in not_cmp.items() if not k.startswith("no oracle family")))
    return {"schema": SCHEMA, "method": METHOD, "result": result, "why": why, "grade": grade, "compared": compared, "equal": equal, "mismatch_n": len(mism), "mismatch": mism[:max_report],
            "compared_by_family": dict(by_fam_cmp), "not_compared": dict(not_cmp), "not_compared_by_family": dict(not_cmp_fam),
            # the no-oracle families said PER Q(P) FAMILY and IN IDENTITIES (the manifest's rule), with the binding-entry count
            # beside it -- `hidden_stream` is the oracle's class, not a Q(P) family, and entries are not identities (an M1 stream identity is
            # one per (layer, engine step); its entries are per member)
            "not_compared_by_query_family": {qf: {"identities": (int(query_families[qf]) if isinstance(query_families, dict) and qf in query_families else None),
                                                  "binding_entries": int(n),
                                                  # on a PADDED record the manifest count is the FULL cap (executed + inactive); the
                                                  # live (executed) identities are the manifest's minus the padding entries stated apart in `padding` for
                                                  # this family (a per-request stream identity is one entry; a shared plane's cells are per member)
                                                  **({"identities_basis": "required manifest at full cap (executed + inactive by enablement)",
                                                      "identities_live": (int(query_families[qf]) - int(padding_not_cmp.get(qf, 0))
                                                                          if isinstance(query_families, dict) and qf in query_families else None),
                                                      "identities_inactive_stated_apart": int(padding_not_cmp.get(qf, 0))} if padding_not_cmp else {}),
                                                  "why": ("no oracle family: kernel-interior stream (no dispatcher-level value in the Match record)"
                                                          if qf.endswith("_stream") else "no oracle family: registered input (pinned, not a Match value)")}
                                             for qf, n in sorted(not_cmp_qfam.items())},
            "not_compared_identities_source": ("required manifest rows" if isinstance(query_families, dict) else "no manifest supplied: identities not stated (entries only)"),
            # the corrected reach: invocations >= 1 of shared names compared at THEIR op (pre-fix: layer 0 vs layer 0)
            "shared_invocations": {"compared_at_program_op": int(shared_inv["compared_at_program_op"]), "not_compared": int(shared_inv["not_compared"]),
                                   "note": "pre-fix records compared these entries as the first tensor's bytes vs the canonical op's oracle value (vacuous equal)"},
            # the padded record's inactive stratum, stated apart: not compared (no execution), not attribution pairs
            **({"padding": {"entries_not_compared": int(sum(padding_not_cmp.values())), "by_query_family": dict(sorted(padding_not_cmp.items())),
                            "pairs": len(padding_pairs), "steps": sorted({s for s, _ in padding_pairs}),
                            "pair_list": [[s, r] for s, r in sorted(padding_pairs)],          # the pairs themselves, for the cross-document check
                            "rule": "bot-by-enablement entries materialize identities the executed prefix left inactive: no oracle value exists; "
                                    "their words are checked == bot_word by the inactive-stratum check, not by the oracle"}}
               if padding_not_cmp else {}),
            "steps_compared": sorted(steps_cmp), "steps_not_snapshotted": steps_not_snap, "complete": complete,
            "attribution": {"complete": attribution_complete, "pairs": len(pairs_seen), "unverified": attrib_missing[:max_report], "unverified_n": len(attrib_missing),
                            "ambiguous_steps": ambiguous, "requests": sorted({r for _, r in pairs_seen}, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)},
            "workload_check": wcheck, "oracle": oracle.provenance()}


def same_name_invocations(bmap: dict[str, Any]) -> dict[tuple[int, str], int]:
    """(step, member name) -> the number of DISTINCT invocations bound under that name at that step (> 1 = a shared module's metas)."""
    inv: dict[tuple[int, str], set[int]] = {}
    for e in bound_entries(bmap):
        ident = e.get("identity")
        if not ident or "step" not in ident or e.get("name") is None:
            continue
        inv.setdefault((int(ident["step"]), str(e["name"])), set()).add(int(ident.get("invocation", 0) or 0))
    return {k: len(v) for k, v in inv.items()}


def entry_read_name(e: dict[str, Any], same_name_invocations: int) -> str:
    """The name a committed reader is asked for: the entry's member name, suffixed `#ord<tensor_ordinal>` when MORE THAN ONE invocation of that
    member is bound at the step (a shared module) -- a bare name would return the first meta's bytes for every invocation.  Non-shared
    members keep the bare name (every reader of record and every test double understands it)."""
    name = str(e.get("name"))
    if same_name_invocations > 1 and e.get("tensor_ordinal") is not None and ORDINAL_SEP not in name:
        return f"{name}{ORDINAL_SEP}{int(e['tensor_ordinal'])}"
    return name
