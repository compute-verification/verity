"""What the executed prefix's two account links make of a field a record leaves out: `check.commit_tokens` (the Commit's token record)
and `check.executed_prefix` (the Match account).

A field that is absent, null or empty reads alike: a prefix of record that names no request places nothing, a record that states no
request, identity, stop rule or lag is refused by the name of what it lacks, and a request no token was served for gets no identity.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.check.test_absent_commit_fields import ABSENT, put
from tests.check.test_commit_tokens import (
    ARRIVALS,
    FINISH,
    ORDER,
    SERVED,
    STOP,
    refused,
    world,
)
from tests.check.test_executed_prefix import EOS, _match_account, _mixed_epr, _served
from tests.commit.test_binding_negatives import PD
from verity_vllm.check import commit_tokens as CT
from verity_vllm.check import executed_prefix as EPL

C = CT.CommitTokens
#: how each problem of a prefix that stopped early with no attested LAG to place it by begins (`begins`)
NOT_ATTESTED = ["r0: (4) stopped before its ", "r0: (4) executed prefix recomputed None != the record's executed_prefix_len 4 / the Commit's 4 (served 3, "]
#: the values each of the token record's two blocks states
BLOCK = {C.STOP_RULE: (C.EOS_IDS, C.LAG), C.LAG: (C.LAG_VALUE, C.ASYNC, C.SOURCE)}
NO_STOP_RULE = ["record stop_rule.eos_token_ids None != the Program's declared [2]",
                ("(5) lag not attested: the Program declares LAG 1; stop_rule.lag None, lag.value 1, the engine's read-back 1; "
                 "record async_scheduling True vs the engine's True, source 'engine read-back'")]
NO_LAG = [("(5) lag not attested: the Program declares LAG 1; stop_rule.lag 1, lag.value None, the engine's read-back 1; "
           "record async_scheduling None vs the engine's True, source None")]
NO_TOKENS = "{}: no generated_token_ids in the token record"
NONE_SERVED = ["r0: (3) served no token at all: not a stop of the Program's rule",
               "r0: (4) executed prefix recomputed None != the record's executed_prefix_len 4 / the Commit's 4 (served 0, "]
NOT_COVERED = "{}: not in the Match record's tokens.json (request ids []): the serving transcript does not cover it"
LAG_NOT_ON_RECORD = ("lag not attested on the record: the Program declares LAG 1, the Match record's tokens.json attests None "
                     "(executed-prefix condition (b))")
SERVED_NONE = (f"r0: the Commit served 11 tokens (last {EOS}) but the Match record's tokens.json served 0 (last None)"
               "{}: not one serving transcript (executed-prefix condition (a))")
#: every value that tests false: each is nothing stated to a reader that takes a value by its truth
NOTHING = (None, {}, [], 0, "", False)


def left_out(empty: Any) -> pytest.MarkDecorator:
    """A field absent, null and empty (`empty`: the empty value of its kind), and every other value that tests false.  The readers
    take a field by its truth: one that took only an absent or a null field as nothing stated would answer the others differently."""
    others = {"zero": 0, "no-text": "", "false": False, "no-item": [], "no-key": {}}
    names = [name for name, value in others.items() if type(value) is not type(empty)]
    return pytest.mark.parametrize("stated", [ABSENT, None, empty, *(others[name] for name in names)], ids=["absent", "null", "empty", *names])


def begins(problems: list[str], heads: list[str]) -> bool:
    """Whether the problems are as many as `heads` and each starts with its own (a whole problem starts with itself)."""
    return len(problems) == len(heads) and all(p.startswith(h) for p, h in zip(problems, heads))


def match_link(epr: dict[str, Any], account: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return EPL.link_to_match_account(epr, account, order=ORDER, declared_lag=1, **kw)


# ------------------------------------------------------------------------------------------------------------ the Commit's token record

@left_out({})
def test_a_prefix_of_record_that_names_no_request_places_nothing_on_the_token_record(stated):
    w = world()
    epr = put(copy.deepcopy(w.epr), ("requests",), stated)
    record = CT.commit_tokens_record(epr, order=ORDER, served_tokens=SERVED, finish_reasons=FINISH, identities={}, stop=STOP,
                                     program_digest=PD, pair=0, run_root=w.run_root)
    assert record[C.REQUESTS] == []
    link = w.relink(w.record(), epr=epr)
    assert link["ok"] and link["required"] is False and link["reduced_requests"] == []
    assert link["tokens_committed"] == {} and link["served_by_request"] == {} and link["record"]["requests_n"] == 2
    both = CT.account_links(w.com, w.run, w.bmap, epr, out_dir=None, pair=0, order=ORDER, arrivals=ARRIVALS, match_dir=None, stop=STOP,
                            served=SERVED, finish=FINISH, past=w.past, program_digest=PD)
    built = both["commit_account_link"]
    assert both["error"] is None and built["ok"] and built["record"]["requests_n"] == 0 and built["reduced_requests"] == []
    assert both["match_account_link"]["ok"] and both["match_account_link"]["reduced_requests"] == []


@left_out([])
def test_a_request_no_identity_is_listed_for_gets_an_empty_list_in_the_token_record(stated):
    w = world()
    listed = put({"r0": [None], "r1": [None, None]}, ("r0",), stated)
    record = CT.commit_tokens_record(w.epr, order=ORDER, served_tokens=SERVED, finish_reasons=FINISH, identities=listed, stop=STOP,
                                     program_digest=PD, pair=0, run_root=w.run_root)
    assert [e[C.IDENTITIES] for e in record[C.REQUESTS]] == [[], [None, None]]


@left_out([])
def test_a_token_record_that_states_no_request_has_no_entry_for_any(stated):
    w = world()
    record = put(w.record(), (C.REQUESTS,), stated)
    assert CT.request_entries(record) == []
    link = w.relink(record)
    assert refused(link) == [f"{r}: 0 entries in the token record (one per request)" for r in ("r0", "r1")]
    assert link["record"]["requests_n"] == 0 and link["tokens_committed"] == {"r0": False, "r1": False}


@left_out([])
def test_an_entry_that_lists_no_token_identity_binds_none_of_its_served_tokens(stated):
    w = world()
    record = w.record()
    put(record[C.REQUESTS][0], (C.IDENTITIES,), stated)
    link = w.relink(record)
    assert refused(link) == (["r0: (1) 0 token identities for 3 served tokens (one per served token)"]
                             + [f"r0: (1) served token {s} has no committed identity in the token record" for s in range(3)])
    assert link["tokens_committed"] == {"r0": False, "r1": True} and link["served_by_request"] == {"r0": 3, "r1": 5}


@pytest.mark.parametrize("field, named", [(C.STOP_RULE, NO_STOP_RULE), (C.LAG, NO_LAG)])
@left_out({})
def test_a_token_record_without_its_stop_rule_or_its_lag_attests_no_lag(field, named, stated):
    w = world()
    link = w.relink(put(w.record(), (field,), stated))
    assert begins(refused(link), named + NOT_ATTESTED) and link["lag_attested"] is False
    assert link == w.relink(put(w.record(), (field,), dict.fromkeys(BLOCK[field])))     # as a block that states a null for each value


@pytest.mark.parametrize("bmap", [*NOTHING, *({"entries": v} for v in NOTHING), {"entries": [{}]}, *({"entries": [{"identity": v}]} for v in NOTHING),
                                  {"entries": [{"identity": {"op_path": CT.FAMILY[0]}}]}, {"entries": [{"identity": {"output_member": CT.FAMILY[1]}}]}])
def test_a_binding_map_without_an_entry_that_states_the_sampled_ids_identity_binds_none(bmap):
    assert CT.sampled_ids_committed(bmap) is False


def test_a_binding_map_binds_a_sampled_id_where_one_entry_states_its_identity():
    sampled = {"identity": dict(zip(("op_path", "output_member"), CT.FAMILY))}
    assert CT.sampled_ids_committed({"entries": [{"identity": None}, {}, sampled]}) is True


@pytest.mark.parametrize("stated, problems, served_n", [(ABSENT, [NO_TOKENS.format("r0")], None), (None, [NO_TOKENS.format("r0")], None),
                                                        ([], NONE_SERVED, 0)], ids=["absent", "null", "empty"])
def test_a_request_the_driver_served_no_token_for_gets_no_identity(stated, problems, served_n):
    w = world()
    link = w.link(served=put(dict(SERVED), ("r0",), stated))
    assert begins(refused(link), problems) and link["served_by_request"] == {"r0": served_n, "r1": 5}


@pytest.mark.parametrize("served", [None, {}], ids=["null", "empty"])
def test_a_commit_whose_driver_served_nothing_names_every_request(served):
    w = world()
    link = w.link(served=served)
    assert refused(link) == [NO_TOKENS.format(r) for r in ("r0", "r1")] and link["tokens_committed"] == {"r0": False, "r1": False}


# ------------------------------------------------------------------------------------------------------------ the Match account

@pytest.mark.parametrize("match_dir", NOTHING)
def test_a_caller_that_names_no_match_directory_gets_an_account_with_neither_record(match_dir):
    account = EPL.load_match_account(match_dir)
    assert (account["tokens"], account["global_match"]) == (None, None)
    assert account["notes"] == ["no --match-dir: the Match record of this attempt is not beside the Commit"]


@left_out({})
def test_a_prefix_of_record_that_names_no_request_needs_no_match_account(stated, tmp_path):
    epr = put(_mixed_epr(), ("requests",), stated)
    link = match_link(epr, EPL.load_match_account(_match_account(tmp_path)))
    assert link["ok"] and link["required"] is False and link["reduced_requests"] == [] and link["tokens_equal"] is None
    assert link["tokens_json"]["tokens_equal"] == {} and link["tokens_json"]["requests_n"] == 2
    assert link["global_match"]["requests_linked"] == ["r0", "r1"] and link["lag_attested"] is True
    alone = match_link(epr, EPL.load_match_account(None))
    assert alone["ok"] and alone["required"] is False and alone["lag_attested"] is False


@left_out([])
def test_an_account_that_states_no_note_links_with_none(stated, tmp_path):
    account = put(EPL.load_match_account(_match_account(tmp_path)), ("notes",), stated)
    link = match_link(_mixed_epr(), account)
    assert link["ok"] and link["notes"] == []


def test_the_notes_an_account_states_are_carried_as_a_copy(tmp_path):
    account = dict(EPL.load_match_account(_match_account(tmp_path)), notes=["a note"])
    link = match_link(_mixed_epr(), account)
    assert link["notes"] == ["a note"] and link["notes"] is not account["notes"]


@left_out([])
def test_a_match_record_whose_tokens_state_no_request_covers_none(stated, tmp_path):
    account = EPL.load_match_account(_match_account(tmp_path))
    put(account["tokens"], ("requests",), stated)
    link = match_link(_mixed_epr(), account)
    assert refused(link) == [NOT_COVERED.format(r) for r in ("r0", "r1")] and link["tokens_equal"] is False
    assert link["tokens_json"]["requests_n"] == 0 and link["tokens_json"]["served_by_request"] == {}
    assert link["tokens_json"]["tokens_equal"] == {"r0": None, "r1": None}


@pytest.mark.parametrize("stated", [ABSENT, None, {}, 1], ids=["absent", "null", "empty", "a number"])
def test_a_match_record_whose_tokens_state_no_lag_block_attests_none(stated, tmp_path):
    account = EPL.load_match_account(_match_account(tmp_path))
    put(account["tokens"], ("lag",), stated)
    link = match_link(_mixed_epr(), account)
    assert refused(link) == [LAG_NOT_ON_RECORD] and link["lag_attested"] is False
    assert link["tokens_json"]["lag"] is None and link["tokens_json"]["async_scheduling"] is None


@pytest.mark.parametrize("with_the_commits_tokens, first", [(False, ""), (True, ", first differing position 0")])
@left_out([])
def test_a_match_record_entry_that_states_no_served_id_served_none(with_the_commits_tokens, first, stated, tmp_path):
    account = EPL.load_match_account(_match_account(tmp_path))
    put(account["tokens"]["requests"][0], ("generated_token_ids",), stated)
    served = _served({"r0": 11, "r1": 32}, {"r0"}) if with_the_commits_tokens else None
    link = match_link(_mixed_epr(), account, served_tokens=served)
    assert refused(link) == [SERVED_NONE.format(first)]
    assert link["tokens_json"]["served_by_request"] == {"r0": 0, "r1": 32} and link["tokens_json"]["tokens_equal"] == {"r0": False, "r1": True}


@pytest.mark.parametrize("missing, block", [("tokens", "tokens_json"), ("global_match", "global_match")])
def test_the_lag_is_attested_only_with_both_records_of_the_match_beside_the_commit(missing, block, tmp_path):
    account = EPL.load_match_account(_match_account(tmp_path))
    assert match_link(_mixed_epr(), account)["lag_attested"] is True
    link = match_link(_mixed_epr(), dict(account, **{missing: None}))
    assert link["ok"] is False and link[block] is None and link["lag_attested"] is False


@pytest.mark.parametrize("key, stated", [("declared", 2), ("record_attests", 2), ("variable_lag", True)])
def test_a_g6_lag_that_disagrees_or_a_variable_lag_leaves_the_lag_unattested(key, stated, tmp_path):
    lag = {"declared": 1, "record_attests": 1, "tokens_json_lag": 1, "exercised": True, "variable_lag": False}
    account = EPL.load_match_account(_match_account(tmp_path, g6_lag=dict(lag, **{key: stated})))
    link = match_link(_mixed_epr(), account)
    assert link["ok"] is False and link["lag_attested"] is False and link["global_match"]["lag"][key] == stated
