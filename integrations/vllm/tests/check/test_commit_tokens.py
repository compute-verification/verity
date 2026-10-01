"""The Commit's token record (`check.commit_tokens`).  A config run has no Match stage, so a request that stops at EOS before its cap has
its executed prefix accepted on the tokens the driver served, each opened from the run root and verified equal to the committed sampled
id it was served from.

The world: two requests arrive at engine step 0 under LAG 1.  r0 serves [11, 12, EOS] and stops (its fourth step is the LAG row, whose
sampled id 99 is committed and never served); r1 runs its cap of 5.  Every tensor is committed by the production host committer and read
back through `OpenedReader`, so each refusal here is the one a tampered or inconsistent record meets on a pod."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import Any

import numpy as np
import pytest

from tests.check.opened import commit_host_steps
from tests.commit.test_binding_negatives import PD
from verity_vllm.check import commit_tokens as CT, executed_prefix as EP
from verity_vllm.check.replay import population as POP
from verity_vllm.check.replay.linkage import boundary_linkage, linkage_horizon
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import binding as B
from verity_vllm.commit.opened import OpenedReader, UnverifiedUnlinked

C = CT.CommitTokens
EOS, CAP, LAG, LAG_ROW = 2, 5, 1, 99
EMBED = "model.embed_tokens"
ORDER = {"0f2b6d7e-r0": "r0", "5c1a9b3d-r1": "r1"}
ARRIVALS = {"r0": 0, "r1": 0}
PROMPTS = {"r0": [5, 6, 7], "r1": [8, 9]}
SAMPLED = {"r0": [11, 12, EOS, LAG_ROW], "r1": [21, 22, 23, 24, 25]}
SERVED = {"r0": [11, 12, EOS], "r1": [21, 22, 23, 24, 25]}
FINISH = {"r0": C.STOP, "r1": C.LENGTH}
STOP = {"eos": [EOS], "lag": LAG, "declared_lag": LAG, "engine_lag": LAG, "async_scheduling": True}


@dataclasses.dataclass
class World:
    com: Any
    run: Any
    bmap: dict
    retained: dict          # engine step -> the sampler tensor's retained copy (the buffer the committer keeps and opens from)
    epr: dict
    past: dict
    served: dict
    finish: dict

    @property
    def run_root(self) -> str:
        return bytes(self.run.run_root).hex()

    def store(self, com: Any = None, run: Any = None) -> UnverifiedUnlinked:
        return UnverifiedUnlinked(CommittedStore(self.bmap, OpenedReader(com or self.com, run or self.run)))

    def link(self, out_dir=None, **kw) -> dict:
        args = dict(out_dir=None if out_dir is None else str(out_dir), pair=0, order=ORDER, arrivals=ARRIVALS, stop=STOP, served=self.served,
                    finish=self.finish, past=self.past, program_digest=PD)
        return CT.commit_link(self.com, self.run, self.bmap, self.epr, **(args | kw))

    def record(self) -> dict:
        """The record an honest writer produces for this world (what `commit_link` writes)."""
        store = self.store()
        ids = {r: CT.sampled_identities(store, r, ARRIVALS[r], len(self.served.get(r) or [])) for r in self.epr["requests"]}
        return CT.commit_tokens_record(self.epr, order=ORDER, served_tokens=self.served, finish_reasons=self.finish, identities=ids, stop=STOP,
                                       program_digest=PD, pair=0, run_root=self.run_root)

    def relink(self, record: dict, *, store: Any = None, epr: dict | None = None, **kw) -> dict:
        args = dict(opened=CT.opener(store or self.store()), order=ORDER, arrivals=ARRIVALS, served_tokens=self.served, finish_reasons=self.finish,
                    stop=STOP, program_digest=PD, run_root=self.run_root, pair=0, family=CT.FAMILY, past=self.past)
        return CT.link_to_commit_account(epr or self.epr, record, **(args | kw))


def world(sampled: dict = SAMPLED, served: dict = SERVED, finish: dict = FINISH, *, run_id: str = "commit-tokens") -> World:
    """`sampled[r]`: r's committed sampled id at each engine step it ran; its input at step e > 0 is its sampled id of step e - 1."""
    steps, rows, retained = {}, {}, {}
    for e in range(max(len(v) for v in sampled.values())):
        live = [r for r in sorted(sampled) if e < len(sampled[r])]
        ids, rr = [], []
        for r in live:
            toks = PROMPTS[r] if e == 0 else [sampled[r][e - 1]]
            rr.append((r, len(ids), len(ids) + len(toks)))
            ids += toks
        retained[e] = bytearray(np.asarray([sampled[r][e] for r in live], dtype=np.int64).tobytes())
        steps[e] = [(f"{EMBED}/in", np.asarray(ids, dtype=np.int64).tobytes(), 0, "int64", (len(ids),)),
                    ("runner.sampler/sampled_token_ids", retained[e], 1, "int64", (len(live),))]
        rows[e] = rr
    com, run = commit_host_steps(steps, run_id=run_id, chunk=64)
    bmap = B.build_binding_map(com._layouts, run.steps, run.run_root, PD, chunk=64, request_rows=rows)
    epr = EP.executed_prefix_of_record(sorted(served), served, caps={r: CAP for r in served}, eos_of_record=STOP["eos"], lag=LAG)
    past = EP.rows_past_the_stop(bmap, EP.request_steps_of_record(epr, ARRIVALS))
    return World(com, run, bmap, retained, epr, past, served, finish)


def refused(link: dict) -> list[str]:
    assert link["ok"] is False and link["problems"], link
    return link["problems"]


def named(problems: list[str], *parts: str) -> bool:
    return any(all(p in x for p in parts) for x in problems)


def _no_match(w: World) -> dict:
    return EP.link_to_match_account(w.epr, EP.load_match_account(None), order=ORDER, declared_lag=LAG, served_tokens=w.served)


# ------------------------------------------------------------------------------------------------------------ accepted

def test_an_early_eos_is_accepted_on_the_token_record_bound_to_the_commitment(tmp_path):
    w = world()
    assert w.epr["ok"] and w.epr["requests"]["r0"]["executed_prefix_len"] == 4 and w.past["ok"]
    link = w.link(tmp_path)
    assert link["ok"] and link["problems"] == [] and link["required"] and link["reduced_requests"] == ["r0"]
    assert link["tokens_committed"] == {"r0": True, "r1": True} and link["lag_attested"] is True
    assert link["served_by_request"] == {"r0": 3, "r1": 5} and link["executed_by_request"] == {"r0": 4, "r1": 5}
    path = tmp_path / C.FILE.format(pair=0)
    raw = path.read_bytes()
    assert link["record"] == {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "schema": C.SCHEMA, "requests_n": 2}
    rec = json.loads(raw)
    assert rec[C.TOKENS_SOURCE] == C.SERVED_OUTPUT and rec[C.RUN_ROOT] == w.run_root and rec[C.PROGRAM_DIGEST] == PD and rec[C.PAIR] == 0
    assert rec[C.STOP_RULE] == {C.EOS_IDS: [EOS], C.LAG: LAG} and rec[C.LAG] == {C.LAG_VALUE: LAG, C.ASYNC: True, C.SOURCE: C.ENGINE_READ_BACK}
    r0, r1 = rec[C.REQUESTS]
    assert (r0[C.REQUEST_ID], r0[C.INDEX], r0[C.CAP], r0[C.GENERATED], r0[C.SERVED_LEN], r0[C.FINISH], r0[C.EXECUTED_LEN]) == (
        "0f2b6d7e-r0", 0, CAP, SERVED["r0"], 3, C.STOP, 4)
    assert r0[C.IDENTITIES] == [dict(zip(C.IDENTITY, (s, *CT.FAMILY, 0))) for s in range(3)]
    assert r1[C.FINISH] == C.LENGTH and r1[C.EXECUTED_LEN] == CAP and [i["step"] for i in r1[C.IDENTITIES]] == list(range(CAP))
    # the LAG row's sampled id is committed, and the record names no identity for it: it was never served
    assert w.store().words("r0", 3, *CT.FAMILY, 0, 1)[0].tolist() == [LAG_ROW]
    # with no Match stage the Match link alone refuses this prefix; the token record is the account that accepts it
    match = _no_match(w)
    assert match["ok"] is False and match["required"] and match["match_dir"] is None
    assert CT.account_link_error(match, None).startswith("executed prefix not linked to the Match account")
    assert CT.account_link_error(match, link) is None


def test_account_links_decides_a_config_run_on_the_token_record_and_never_falls_back(tmp_path, capsys):
    w = world()
    kw = dict(out_dir=str(tmp_path), pair=0, order=ORDER, arrivals=ARRIVALS, match_dir=None, stop=STOP, served=SERVED, finish=FINISH,
              past=w.past, program_digest=PD)
    out = CT.account_links(w.com, w.run, w.bmap, w.epr, **kw)
    assert out["error"] is None and out["commit_account_link"]["ok"] and out["match_account_link"]["ok"] is False
    assert "executed prefix <-> Commit's token record (pair 0): LINKED" in capsys.readouterr().out
    # a record that cannot be built is a refused commit link: never the Match link's verdict, never the committer's own count
    bad = CT.account_links(w.com, w.run, None, w.epr, **kw)
    assert bad["commit_account_link"]["problems"][0].startswith("token record not built") and bad["commit_account_link"]["required"]
    assert "not bound to the commitment by the Commit's token record: token record not built" in bad["error"]


def test_a_length_stop_needs_no_reduction():
    full = {"r0": [11, 12, 13, 14, 15], "r1": SERVED["r1"]}
    w = world(sampled=full, served=full, finish={"r0": C.LENGTH, "r1": C.LENGTH})
    link = w.link()
    assert link["ok"] and link["required"] is False and link["executed_by_request"] == {"r0": CAP, "r1": CAP}
    match = _no_match(w)
    assert match["ok"] and CT.account_link_error(match, link) is None
    # an EOS exactly at the cap reduces nothing, under either finish reason the engine reports for it
    at_cap = {"r0": [11, 12, 13, 14, EOS], "r1": SERVED["r1"]}
    for fin in (C.STOP, C.LENGTH):
        link = world(sampled=at_cap, served=at_cap, finish={"r0": fin, "r1": C.LENGTH}).link()
        assert link["ok"] and link["required"] is False and link["executed_by_request"]["r0"] == CAP, (fin, link["problems"])
    # a refused link that places nothing is recorded, not decisive
    idle = world(sampled=full, served=full, finish={"r0": C.LENGTH, "r1": C.LENGTH}).link(stop=dict(STOP, engine_lag=None, async_scheduling=None))
    assert idle["ok"] is False and idle["required"] is False and CT.account_link_error(match, idle) is None


def test_account_link_error_decides_over_both_links():
    m_none = {"ok": False, "match_dir": None, "problems": ["no Match record"]}
    m_bad = {"ok": False, "match_dir": "/row/match", "problems": ["r0: not one serving transcript"]}
    m_ok = {"ok": True, "match_dir": "/row/match", "problems": []}
    c_ok = {"ok": True, "required": True, "problems": []}
    c_bad = {"ok": False, "required": True, "problems": ["r0: (1) served token 1 is 13"]}
    c_idle = {"ok": False, "required": False, "problems": ["(5) lag not attested"]}
    assert CT.account_link_error(m_none, c_ok) is None and CT.account_link_error(m_ok, c_ok) is None and CT.account_link_error(m_ok, c_idle) is None
    assert CT.account_link_error(m_bad, c_ok) == "executed prefix not linked to the Match account: r0: not one serving transcript"
    assert CT.account_link_error(m_ok, c_bad) == "executed prefix not bound to the commitment by the Commit's token record: r0: (1) served token 1 is 13"
    assert CT.account_link_error(m_bad, c_bad).count(" | ") == 1 and "token record" in CT.account_link_error(m_none, c_bad)
    assert "Match account" in CT.account_link_error(m_none, None)


# ------------------------------------------------------------------------------------------------------------ refused by name

def test_a_token_served_after_the_eos_is_refused():
    w = world(served={"r0": [11, 12, EOS, LAG_ROW], "r1": SERVED["r1"]})
    link = w.link()
    probs = refused(link)
    assert named(probs, "r0: (3) served past an EOS: token 2 is the declared EOS id 2 and 1 more were served after it")
    assert named(probs, "r0: (4) executed prefix recomputed None")
    assert link["tokens_committed"]["r0"] is True           # every served token is the committed one: the stop rule refuses it


def test_a_dropped_eos_is_refused():
    w = world()
    rec = w.record()
    r0 = rec[C.REQUESTS][0]
    r0[C.GENERATED], r0[C.SERVED_LEN], r0[C.IDENTITIES] = r0[C.GENERATED][:2], 2, r0[C.IDENTITIES][:2]
    probs = refused(w.relink(rec))
    assert named(probs, "r0: (2) the driver served 3 token(s), the record 2 (first differing position 2)")
    assert named(probs, "r0: (3) finish_reason 'stop' but the last served token 12 is not a declared EOS id")
    # the driver itself reports a stop with no EOS served
    probs = refused(world(served={"r0": [11, 12], "r1": SERVED["r1"]}).link())
    assert named(probs, "r0: (3) finish_reason 'stop' but the last served token 12 is not a declared EOS id")
    assert named(probs, "r0: (4) executed prefix recomputed None")


@pytest.mark.parametrize("where", ["stop_rule.lag", "lag.value", "engine read-back", "async_scheduling", "source", "undeclared"])
def test_a_wrong_lag_is_refused(where):
    w = world()
    rec, stop = w.record(), dict(STOP)
    if where == "stop_rule.lag":
        rec[C.STOP_RULE][C.LAG] = 0
    elif where == "lag.value":
        rec[C.LAG][C.LAG_VALUE] = 0
    elif where == "engine read-back":
        stop |= {"engine_lag": 0, "async_scheduling": False}
    elif where == "async_scheduling":
        rec[C.LAG][C.ASYNC] = False
    elif where == "source":
        rec[C.LAG][C.SOURCE] = "declared"
    else:
        stop["declared_lag"] = None
    link = w.relink(rec, stop=stop)
    probs = refused(link)
    assert link["lag_attested"] is False and named(probs, "(5) lag not attested: the Program declares LAG")
    assert named(probs, "r0: (4) stopped before its cap at 3 but the LAG is not attested")


def test_an_executed_prefix_placed_under_another_lag_is_refused():
    w = world()
    rec = w.record()
    rec[C.REQUESTS][0][C.EXECUTED_LEN] = 3
    assert named(refused(w.relink(rec)), "r0: (4) executed prefix recomputed 4 != the record's executed_prefix_len 3")
    # the Commit's own count placed under LAG 0: the recomputation, not either count, is the one of record
    lag0 = copy.deepcopy(w.epr)
    lag0["requests"]["r0"]["executed_prefix_len"] = 3
    assert named(refused(w.relink(w.record(), epr=lag0)), "r0: (4) executed prefix recomputed 4 != the record's executed_prefix_len 4 / the Commit's 3")


def test_a_served_token_that_differs_from_the_committed_value_is_refused():
    w = world(served={"r0": [11, 13, EOS], "r1": SERVED["r1"]})
    link = w.link()
    assert named(refused(link), "r0: (1) served token 1 is 13 but the committed sampled id at engine step 1 is 12")
    assert link["tokens_committed"] == {"r0": False, "r1": True}
    # every token of r1 differs: the first three are named, the rest counted
    probs = refused(world(served={"r0": SERVED["r0"], "r1": [t + 100 for t in SERVED["r1"]]}).link())
    assert sum("r1: (1) served token" in p for p in probs) == CT.TOKEN_PROBLEMS_SHOWN
    assert named(probs, "r1: (1) 2 more served token(s) not bound to the commitment")


def test_a_missing_token_identity_is_refused():
    w = world()
    rec = w.record()
    rec[C.REQUESTS][0][C.IDENTITIES][1] = None
    assert named(refused(w.relink(rec)), "r0: (1) served token 1 has no committed identity in the token record")
    rec = w.record()
    rec[C.REQUESTS][0][C.IDENTITIES] = rec[C.REQUESTS][0][C.IDENTITIES][:2]
    probs = refused(w.relink(rec))
    assert named(probs, "r0: (1) 2 token identities for 3 served tokens") and named(probs, "r0: (1) served token 2 has no committed identity")
    # an identity of another row (the LAG row), another member, another invocation
    rec = w.record()
    rec[C.REQUESTS][0][C.IDENTITIES][2]["step"] = 3
    assert named(refused(w.relink(rec)), "r0: (1) served token 2's identity (step 3, runner.sampler/sampled_token_ids) is not the sampled id of request step 2")
    rec = w.record()
    rec[C.REQUESTS][0][C.IDENTITIES][0]["op_path"] = EMBED
    assert named(refused(w.relink(rec)), "r0: (1) served token 0's identity", "is not the sampled id of request step 0")
    rec = w.record()
    rec[C.REQUESTS][0][C.IDENTITIES][0]["invocation"] = 1
    assert named(refused(w.relink(rec)), "r0: (1) served token 0 not opened from the run root at engine step 0: invocation 1 is not the one")
    # nothing bound there in the binding map: the writer records no identity and the link names it
    bmap = dict(w.bmap, entries=[e for e in w.bmap["entries"] if (e["identity"]["request_id"], e["identity"]["step"], e["identity"]["op_path"]) != ("r0", 1, "runner.sampler")])
    link = CT.commit_link(w.com, w.run, bmap, w.epr, out_dir=None, pair=0, order=ORDER, arrivals=ARRIVALS, stop=STOP, served=SERVED, finish=FINISH,
                          past=w.past, program_digest=PD)
    assert named(refused(link), "r0: (1) served token 1 has no committed identity in the token record")


def test_tokens_that_did_not_come_from_the_served_output_are_refused():
    w = world()
    rec = w.record()
    rec[C.TOKENS_SOURCE] = "committed store (runner.sampler/sampled_token_ids read back)"
    assert named(refused(w.relink(rec)), "would make the check circular")
    # a writer that read r0's tokens back from the store over its executed prefix: the record agrees with the commitment by construction,
    # and the driver's served list (the engine's request outputs, passed to the link apart from the record) refuses it
    store = w.store()
    from_store = [int(store.words("r0", s, *CT.FAMILY, 0, 1)[0][0]) for s in range(4)]
    assert from_store == SAMPLED["r0"]
    rec = w.record()
    r0 = rec[C.REQUESTS][0]
    r0[C.GENERATED], r0[C.SERVED_LEN], r0[C.IDENTITIES] = from_store, 4, CT.sampled_identities(store, "r0", 0, 4)
    assert named(refused(w.relink(rec)), "r0: (2) the driver served 3 token(s), the record 4 (first differing position 3)")
    # no driver list at all: there is nothing independent to compare the record with
    probs = refused(w.relink(w.record(), served_tokens=None))
    assert named(probs, "r0: (2) the driver served None token(s)") and named(probs, "r1: (2) the driver served None token(s)")


class TamperedPath:
    """The committer with each range opening's first sibling digest flipped: the bytes are the retained ones, the Merkle path is not."""

    def __init__(self, com: Any) -> None:
        self._com = com

    def __getattr__(self, name: str) -> Any:
        return getattr(self._com, name)

    def open_range(self, step: int, lo: int, hi: int) -> Any:
        ro = self._com.open_range(step, lo, hi)
        path = [list(p) for p in ro.path]
        d, k = next((d, k) for d, p in enumerate(path) for k in (0, 1) if p[k] is not None)
        path[d][k] = bytes([path[d][k][0] ^ 1]) + bytes(path[d][k][1:])
        return dataclasses.replace(ro, path=tuple(tuple(p) for p in path))


def test_a_tampered_value_or_merkle_path_is_refused():
    # the prover served 13 as r0's second token and rewrote its retained copy of the committed sampled id to agree
    w = world(served={"r0": [11, 13, EOS], "r1": SERVED["r1"]})
    w.retained[1][:8] = np.asarray([13], dtype=np.int64).tobytes()
    link = w.link()
    assert named(refused(link), "r0: (1) served token 1 not opened from the run root at engine step 1: opened value not verified")
    assert link["tokens_committed"]["r0"] is False
    # the honest bytes under a corrupted Merkle path
    w = world()
    link = w.relink(w.record(), store=w.store(com=TamperedPath(w.com)))
    assert named(refused(link), "r0: (1) served token 0 not opened from the run root at engine step 0: opened value not verified")
    assert link["tokens_committed"] == {"r0": False, "r1": False}


def test_a_record_of_another_attempt_is_refused():
    w = world()
    other = world(sampled={"r0": SAMPLED["r0"], "r1": [31, 32, 33, 34, 35]}, run_id="another-run")
    # this run's bytes opened against another run's commitment
    assert named(refused(w.relink(w.record(), store=w.store(run=other.run))), "r0: (1) served token 0 not opened", "opened value not verified")
    for kw, field in ((dict(run_root=other.run_root), "run_root"), (dict(program_digest="cd" * 32), "program_digest"), (dict(pair=1), "pair")):
        assert named(refused(w.relink(w.record(), **kw)), f"record {field}", "not this attempt's token record"), field
    rec = w.record()
    rec[C.SCHEMA_KEY] = "verity-vllm/commit-tokens/v0"
    assert named(refused(w.relink(rec)), "record schema")
    rec = w.record()
    rec[C.REQUESTS].append(copy.deepcopy(rec[C.REQUESTS][0]))
    assert named(refused(w.relink(rec)), "r0: 2 entries in the token record (one per request)")
    rec = w.record()
    rec[C.STOP_RULE][C.EOS_IDS] = [EOS, 3]
    assert named(refused(w.relink(rec)), "record stop_rule.eos_token_ids [2, 3] != the Program's declared [2]")


@pytest.mark.parametrize("finish,want", [
    ({"r0": C.LENGTH, "r1": C.LENGTH}, "r0: (3) finish_reason 'length' but served 3 of its cap 5"),
    ({"r0": C.STOP, "r1": C.STOP}, "r1: (3) finish_reason 'stop' but the last served token 25 is not a declared EOS id"),
    ({"r0": "abort", "r1": C.LENGTH}, "r0: (3) finish_reason 'abort' is neither 'stop' nor 'length'"),
    ({"r0": None, "r1": C.LENGTH}, "r0: (3) finish_reason None is neither 'stop' nor 'length'"),
])
def test_a_finish_reason_that_disagrees_with_the_opened_tokens_is_refused(finish, want):
    assert named(refused(world(finish=finish).link()), want)


def test_a_record_finish_reason_other_than_the_drivers_is_refused():
    w = world()
    rec = w.record()
    rec[C.REQUESTS][0][C.FINISH] = C.LENGTH
    probs = refused(w.relink(rec))
    assert named(probs, "r0: record finish_reason 'length' != the driver's 'stop'") and named(probs, "r0: (3) finish_reason 'length' but served 3")


def test_rows_committed_past_the_stop_are_refused():
    w = world(sampled={"r0": [11, 12, EOS, LAG_ROW, 7], "r1": SERVED["r1"]})
    assert w.past["ok"] is False
    assert named(refused(w.link()), "(6) execution past the stop: r0 committed rows at engine steps [4]")


# ------------------------------------------------------------------------------------------------------------ the facts of record

def _linkage(w: World, executed: dict | None = None) -> dict:
    wreq = [{"request_id": r, "arrive_step": ARRIVALS[r], "prompt_token_ids": PROMPTS[r], "max_tokens": CAP, "program_steps": None,
             "steps": linkage_horizon(ARRIVALS[r], CAP, None, executed=(executed or {}).get(r, w.epr["requests"][r]["executed_prefix_len"]))}
            for r in sorted(w.epr["requests"])]
    return boundary_linkage(w.bmap, OpenedReader(w.com, w.run), wreq, embed_op_path=EMBED, eos_of_record=STOP["eos"], eos_lag_rows=LAG)


def test_the_boundary_linkage_runs_over_the_same_executed_prefix_and_the_facts_link_on_the_token_record():
    w = world()
    link = w.link()
    lk = _linkage(w)
    assert lk["result"] is True, lk
    h0 = lk["horizon"]["r0"]
    assert h0["linked_steps"] == w.epr["requests"]["r0"]["executed_prefix_len"] == 4 and lk["horizon"]["r1"]["linked_steps"] == CAP
    assert h0["eos_of_record"]["verified"] and h0["eos_of_record"]["eos_step"] == 2 and h0["eos_of_record"]["placeholder_input_ids"] == [EOS]
    binding = {"coverage": {"match_account_link": _no_match(w), "commit_account_link": link, "rows_past_the_stop": w.past}}
    f = EP.executed_prefix_facts(w.epr, binding, lk, ARRIVALS, STOP)
    assert f["r0"]["a_r"] == 3 and f["r0"]["a_r_source"].startswith(f"commit token record {link['record']['path']}")
    assert f["r0"]["linked"] is True and f["r0"]["lag_attested"] is True and f["r0"]["account_missing"] == [] and f["r1"]["linked"] is True
    rec = POP.executed_prefix_of_record("r0", **EP.facts_accepted_by(POP.executed_prefix_of_record, f)["r0"])
    assert rec["faults"] == [], rec
    # a linkage over any other horizon than the executed prefix does not link it
    for n in (3, CAP):
        moved = dict(lk, horizon={**lk["horizon"], "r0": {**h0, "linked_steps": n}})
        assert EP.executed_prefix_facts(w.epr, binding, moved, ARRIVALS, STOP)["r0"]["linked"] is False, n
    assert _linkage(w, executed={"r0": CAP})["result"] is False           # the cap's horizon: r0 has no committed row at step 4
    assert EP.executed_prefix_facts(w.epr, binding, dict(lk, result=False), ARRIVALS, STOP)["r0"]["linked"] is False
    # a token not bound to the commitment, an unattested LAG: not linked, not attested, and the population faults by name
    bad = dict(link, ok=False, tokens_committed={"r0": False, "r1": True}, lag_attested=False)
    fb = EP.executed_prefix_facts(w.epr, {"coverage": dict(binding["coverage"], commit_account_link=bad)}, lk, ARRIVALS, STOP)
    assert fb["r0"]["linked"] is False and fb["r0"]["lag_attested"] is False
    faults = POP.executed_prefix_of_record("r0", **EP.facts_accepted_by(POP.executed_prefix_of_record, fb)["r0"])["faults"]
    assert any("a_r is not linked" in x for x in faults) and any("LAG rule not attested" in x for x in faults), faults


def test_with_a_match_record_beside_the_commit_both_links_hold():
    w = world()
    link, lk = w.link(), _linkage(w)
    match = {"ok": True, "required": True, "match_dir": "/row/match", "lag_attested": True, "problems": [],
             "tokens_json": {"tokens_equal": {"r0": True, "r1": True}, "served_by_request": {"r0": 3, "r1": 5}, "path": "/row/match/capture/tokens.json", "sha256": "ef" * 32},
             "global_match": {"verdict": "PASS", "path": "/row/global_match.json", "sha256": "12" * 32}}
    cov = {"match_account_link": match, "commit_account_link": link, "rows_past_the_stop": w.past}
    f = EP.executed_prefix_facts(w.epr, {"coverage": cov}, lk, ARRIVALS, STOP)
    assert f["r0"]["linked"] is True and f["r0"]["a_r_source"].startswith("match record /row/match/capture/tokens.json")
    bad = dict(link, ok=False, tokens_committed={"r0": False, "r1": True})
    assert EP.executed_prefix_facts(w.epr, {"coverage": dict(cov, commit_account_link=bad)}, lk, ARRIVALS, STOP)["r0"]["linked"] is False
    assert "Commit's token record" in CT.account_link_error(match, bad)
    assert EP.executed_prefix_facts(w.epr, {"coverage": dict(cov, match_account_link=dict(match, ok=False))}, lk, ARRIVALS, STOP)["r0"]["linked"] is False
