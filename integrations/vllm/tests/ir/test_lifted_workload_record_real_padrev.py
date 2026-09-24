"""padrev — the MIXED BATCH record half on a REAL B2 record (pad's HELP after M-0442: the smollm2 B2 32/32 `mixedstop` row — request A = the
earlystop prompt, EOS 0 at event 10 under cap 32; request B = a prompt that runs to its cap; both arrive at step 0; attested tree, LAG 1).
Prepared BEFORE the row exists so the review is minutes when it lands: the row dir is discovered under pad's read-only evidence path (or
`PADREV_MIXEDSTOP_ROW`), every test skips by name without it, and the expected facts are derived from the record's own workload_program.json
(caps, arrivals, uuids) plus the ONE fact pad's manifest declares (A stops on the EOS at event 10): a_A = 11, executed A = [0..11] under LAG 1,
executed B = [0..31], n_live 2 -> 1 at engine step 12.

What is attacked, from the attack list: (1) the report reproduced from the API through the same request list the CLI builds (`--build-workload`)
-- W0 pairing from the record's own facts with UNEQUAL windows (F-pad-24's condition), W2 per-step SETS equal on both sides, each request's
R3 under LAG 1, ok True/False as pad states it; (2) a ⊥ step of A committed while B is live (A's identities cloned onto step 12) -- named at
BOTH levels: A's R3 `declared_inactivity_executed [12]` and W2 `commit_vs_program_mismatch {12: [A]}`; (3) A's PLACEHOLDER step 11 deleted
from the Commit -- a lag-0 Commit of A under a lag-1 header: A's R3 and W2 name step 11, B untouched; (4) the two requests' commit ids
crossed in the request list (r0 <-> r1) -- W0 refuses by the record's facts (unequal caps), the equal-shape limit does not apply here."""
from __future__ import annotations

import glob
import gzip
import json
import os
from pathlib import Path

import pytest

from verity_vllm import build_paths as BP
from verity_vllm.lifting import record as R
from verity_vllm.registry import lifted as L

ROW_NAME = "smollm2-135m__bf16__l40s__tp1__b2__i32__o32__mixedstop__greedy__bi-eager"
_ROOT = Path(__file__).resolve().parents[2]


def _find_row() -> Path | None:
    env = os.environ.get("PADREV_MIXEDSTOP_ROW")
    if env and (Path(env) / "build_workload" / "workload_program.json").exists():
        return Path(env)
    ev = _ROOT / "out" / "gen" / "sweep" / "evidence"
    hits = sorted(glob.glob(str(ev / "*" / ROW_NAME / "*" / "build_workload" / "workload_program.json")))       # pad's copies: <row>/<run>/
    hits += sorted(glob.glob(str(ev / "*" / ROW_NAME / "build_workload" / "workload_program.json")))            # exec's: <row>/ directly
    return Path(hits[-1]).parents[1] if hits else None


ROW = _find_row()
_HAVE = ROW is not None and (ROW / "match" / "capture").exists() and bool(glob.glob(str(ROW / "commit" / "binding_map_p0.json*")))
pytestmark = pytest.mark.skipif(not _HAVE, reason=f"no real B2 mixedstop record (row dir with build_workload/, match/capture/, commit/binding_map_p0) on this tree: {ROW}")
_IS_PADS_ROW = _HAVE and ROW.parent.name == ROW_NAME
BOUNDARY = os.environ.get("PADREV_BOUNDARY_PREFIX", L.BOUNDARY_PREFIX)     # the real rows use the registry's; a scaffold row names its own


def _reqs_from_row(row: Path):
    """The request list the CLI builds for --build-workload (lifted_record.main), rebuilt here so the API can be driven with mutated maps."""
    wp = json.loads((row / "build_workload" / "workload_program.json").read_text())
    eos = [int(x) for x in ((wp.get("stop_rule") or {}).get("eos_token_ids") or [])]
    uuids = ((wp.get("request_ids") or {}).get("workload_request_ids") or {})
    reqs = []
    for r in wp["requests"]:
        rid = str(r["request_id"])
        lp, cap = int(r["prompt_len"]), int(r["cap"])
        src_dir = BP.declared_component(wp, lp, cap - 1)
        assert src_dir is not None, f"{rid}: inputs.derived['{lp},{cap - 1}'] absent (F-pad-26b: nothing may be inferred)"
        comp_dir = BP.component_dir(wp, lp, cap - 1, row) or row / Path(src_dir).name      # as the CLI resolves it (layout 1.1 / legacy)
        params = None
        ip = comp_dir / "instances.json.gz"
        if ip.exists():
            with gzip.open(ip, "rt") as f:
                params = [(str(n), str(t)) for n, t in (json.load(f).get("params") or [])]
        reqs.append({"request_id": uuids[rid], "commit_request_id": rid, "desc": R.load_descriptor(comp_dir), "cap": cap, "arrive": int(r.get("arrive_step", 0)),
                     "prompt_len": lp, "params": params, "component_dir": str(comp_dir), "workload_digest": wp.get("workload_digest")})
    return wp, eos, reqs


@pytest.fixture(scope="module")
def row():
    wp, eos, reqs = _reqs_from_row(ROW)
    cap = R.read_capture(ROW / "match" / "capture")
    bp = glob.glob(str(ROW / "commit" / "binding_map_p0.json*"))[0]
    bmap = json.load(gzip.open(bp, "rt")) if bp.endswith(".gz") else json.loads(Path(bp).read_text())
    lag, src = L.lag_from_header(cap["header"])
    return {"wp": wp, "eos": eos, "reqs": reqs, "capture": cap, "bmap": bmap, "lag": lag, "src": src}


def _ab(row):
    """A = the request that stops on the EOS (its capture transcript carries the EOS; a_r = the EOS's event index + 1, from the TOKENS, never
    from the rows), B = the one that runs to its cap.  On pad's row A's a_r must be 11 (EOS 0 at event 10 -- the manifest's declared fact)."""
    a = b = None
    for q in row["reqs"]:
        cr = R.capture_request(row["capture"], q["request_id"])
        tr = [int(cr["sampled"][st]) for st in cr["executed"] if st in cr["sampled"]]
        hit = [i for i, t in enumerate(tr) if t in set(row["eos"])]
        if hit:
            a = {**q, "a_r": hit[0] + 1}
        else:
            b = {**q, "a_r": q["cap"]}
    assert a is not None and b is not None, "expected exactly one EOS-stopped request and one cap-stopped request"
    if _IS_PADS_ROW:
        assert a["a_r"] == 11 and a["cap"] == b["cap"] == 32 and a["arrive"] == b["arrive"] == 0
    return a, b


def _expected_live(row, a, b):
    """n_live_t from the DECLARATION and the tokens: r is live at t iff arrive_r <= t < arrive_r + a_r + (LAG if r stopped on the EOS else 0)."""
    n_a, n_b = a["a_r"] + (row["lag"] if a["a_r"] < a["cap"] else 0), b["a_r"]
    horizon = max(a["arrive"] + a["cap"], b["arrive"] + b["cap"])
    return {t: int(a["arrive"] <= t < a["arrive"] + n_a) + int(b["arrive"] <= t < b["arrive"] + n_b) for t in range(horizon)}


def _check(row, bmap=None, reqs=None):
    return R.check_workload_record(reqs if reqs is not None else row["reqs"], row["eos"], lag=row["lag"], lag_source=row["src"],
                                   bmap=bmap if bmap is not None else row["bmap"], capture=row["capture"], boundary_prefix=BOUNDARY)


def _with_steps(bmap, crid, steps, drop=False, proto_step=5):
    ents = list(bmap["entries"])
    ident = lambda e: (e.get("identity") or {})
    if drop:
        ents = [e for e in ents if not (str(ident(e).get("request_id")) == crid and int(ident(e).get("step", -1)) in set(steps))]
    else:
        proto = [e for e in ents if str(ident(e).get("request_id")) == crid and int(ident(e).get("step", -1)) == proto_step]
        assert proto
        for s in steps:
            ents.extend({**e, "identity": {**e["identity"], "step": s}} for e in proto)
    return {**bmap, "entries": ents}


def test_pads_mixed_record_reproduced_two_requests_one_stops_on_the_eos_and_the_live_count_moves_with_it(row):
    """From the API through the CLI's own request list: LAG 1 from the header; A executed [0..11] (a_r 11, the placeholder at 11), B [0..cap_B-1];
    n_live 2 at steps < 12 and 1 from 12 to B's last step; W0 PASS from >= 1 record source with UNEQUAL windows (F-pad-24's condition); W2
    PASS on both sides; each request's R3 True under LAG 1 with LAG 0 refuted for A (its matches {0: False, 1: True}) -- and the report's
    `ok` as pad states it for this record (True when the header attests and every check holds; the ⊥ population of A unbound is R3e's word)."""
    a, b = _ab(row)
    lag, ar = row["lag"], a["a_r"]
    if _IS_PADS_ROW:
        assert lag == 1, "pad's HELP asks for an attested tree; a lag-0 header here is a different record"
    rep = _check(row)
    ra = next(p for p in rep["requests"] if p["request_id"] == a["request_id"])
    rb = next(p for p in rep["requests"] if p["request_id"] == b["request_id"])
    assert ra["checks"]["R2_activity"]["a_r"] == ar and ra["checks"]["R3_executed_set"]["committed_steps"] == [a["arrive"] + s for s in range(ar + lag)]
    assert ra["checks"]["R3_executed_set"]["matches"] == {str(lg): lg == lag for lg in L.LAGS_IMPLEMENTED} and ra["checks"]["R3c_capture_executed_set"]["ok"] is True
    assert rb["checks"]["R2_activity"]["a_r"] == b["cap"] and rb["checks"]["R3_executed_set"]["committed_steps"] == [b["arrive"] + s for s in range(b["cap"])]
    assert ra["checks"]["R3e_bot_population"]["bot_population_bound"] is False and rb["checks"]["R3e_bot_population"]["ok"] is True
    w0, w2 = rep["checks"]["W0_pairing"], rep["checks"]["W2_chronology"]
    assert w0["ok"] is True and not w0.get("indistinguishable_by_pairing_facts")
    if _IS_PADS_ROW:
        assert len(w0["sources_evaluated"]) >= 1                                   # F-pad-25: a real row carries spec_tokens / tokens / run.json
    assert w2["ok"] is True and w2["commit_vs_program_mismatch"] == {} and w2["engine_vs_program_mismatch"] == {}
    assert {int(k): v for k, v in w2["n_live_by_step"].items()} == _expected_live(row, a, b)
    assert rep["checks"]["W3_no_crossing"]["a_r_per_request"] == {a["request_id"]: ar, b["request_id"]: b["cap"]}
    assert rep["record_verdict"]["padding_exercised"] is True
    if _IS_PADS_ROW:
        assert rep["ok"] is True and rep["record_verdict"]["lag_discriminated"] is True


def test_a_bot_step_of_a_committed_while_b_is_live_is_named_at_both_levels(row):
    """A's identities cloned onto step 12 (the first ⊥ step of A; B live there): A's R3 `declared_inactivity_executed [12]` AND W2
    `commit_vs_program_mismatch {12: [A]}` -- the Commit's set at 12 is {A, B}, the Program's live set is {B}; B's own report untouched;
    ok False.  ⊥ does not leak into a live batch through the Commit unseen."""
    a, b = _ab(row)
    bot = a["arrive"] + a["a_r"] + row["lag"]                                      # the first ⊥ step of A
    assert b["arrive"] <= bot < b["arrive"] + b["a_r"], "the attack needs B live at A's first ⊥ step"
    bm = _with_steps(row["bmap"], a["commit_request_id"], [bot], proto_step=a["arrive"])
    rep = _check(row, bmap=bm)
    ra = next(p for p in rep["requests"] if p["request_id"] == a["request_id"])
    rb = next(p for p in rep["requests"] if p["request_id"] == b["request_id"])
    assert ra["checks"]["R3_executed_set"]["declared_inactivity_executed"] == [bot] and ra["ok"] is False
    assert rep["checks"]["W2_chronology"]["commit_vs_program_mismatch"] == {str(bot): [a["request_id"]]}
    assert rb["ok"] is True and rep["ok"] is False


def test_as_placeholder_step_deleted_from_the_commit_is_a_lag_0_commit_of_a_under_a_lag_1_header_named_at_both_levels(row):
    """A's step-11 identities removed: A's R3 `declared_activity_not_committed [11]`, matches {0: True, 1: False}; W2
    `commit_vs_program_mismatch {11: [A]}` while the engine side still agrees with the Program; B untouched; ok False."""
    a, b = _ab(row)
    if row["lag"] != 1:
        pytest.skip("the placeholder step exists under LAG 1 only")
    ph = a["arrive"] + a["a_r"]                                                    # the placeholder step of A
    bm = _with_steps(row["bmap"], a["commit_request_id"], [ph], drop=True)
    rep = _check(row, bmap=bm)
    ra = next(p for p in rep["requests"] if p["request_id"] == a["request_id"])
    assert ra["checks"]["R3_executed_set"]["declared_activity_not_committed"] == [ph] and ra["checks"]["R3_executed_set"]["matches"] == {"0": True, "1": False}
    w2 = rep["checks"]["W2_chronology"]
    assert w2["commit_vs_program_mismatch"] == {str(ph): [a["request_id"]]} and w2["engine_vs_program_mismatch"] == {}
    assert next(p for p in rep["requests"] if p["request_id"] == b["request_id"])["ok"] is True and rep["ok"] is False


def test_the_commit_ids_crossed_between_the_two_requests_are_refused_by_the_records_own_pairing_facts(row):
    """r0 <-> r1 exchanged in the request list (each uuid now carries the other's Commit id): the two windows differ (A stops at 11, B runs
    to 31) so W0's facts (spec_tokens / tokens / run.json realized order -- whichever the record carries) refuse by name, or, where W0's
    sources cannot see a pure id swap (named limit), W2's per-step sets name the crossing from step 12 on.  Never a silent pass."""
    a, b = _ab(row)
    other = {a["request_id"]: b["commit_request_id"], b["request_id"]: a["commit_request_id"]}
    swapped = [{**q, "commit_request_id": other[q["request_id"]]} for q in row["reqs"]]
    try:
        rep = _check(row, reqs=swapped)
    except ValueError as e:
        assert a["request_id"] in str(e) or b["request_id"] in str(e) or "pairing" in str(e).lower()
        return
    assert rep["ok"] is False
    w0, w2 = rep["checks"]["W0_pairing"], rep["checks"]["W2_chronology"]
    assert w0["ok"] is False or any(a["request_id"] in v or b["request_id"] in v for v in w2["commit_vs_program_mismatch"].values())
