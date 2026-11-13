"""H1's checks refuse, by name, the forms its other tests never hand them: a request's first slot and a step's slots
under tables a prover committed consistently (`pouw_h1_requests.request_reason`, `slot_reason`), a round's row and coin
(`round_reason`, `rounds_reason`), a record's calls (`tables_reason`), a window's coins and its sample
(`pouw_h1_challenger.trusted_rounds_reason` and its sample session's checks), what H1's settings, its input ids, a static
declaration and the gate's ``model-key`` answer are parsed from, a hand-off's socket path, its stream and its saved
window (`pouw_h1_handoff.Holder`, `send_stream`, `client_of`, `resume`), a Pearl-C pass's salt session
(`pouw_h1_challenger.salt_session_reason`), a period's weights and a value's held rows
(`pouw_h1_served.register_weights`, `pouw_h1_registered.commit`), an unclosed window's trees
(`pouw_h1_served_files.trees`), and the Commit's input ids and a named Commit record
(`pouw_h1_commit.CommitTap.input_ids`, `reused`)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import random
import socket
import threading
from fractions import Fraction
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity.circuits.codec import canonical_json
from verity.protocols.accounting.pouw.verifier import leaf_lanes as LL
from verity.protocols.accounting.pouw.verifier import salt as PS
from verity.protocols.accounting.pouw.verifier import window as Wd
from verity.protocols.accounting.pouw.verifier.audit import (
    HM96,
    commit_rows,
    salt_of,
    weights_root,
)
from verity.protocols.accounting.pouw.verifier.coins import Challenger
from verity.protocols.proofs.spot.device import challenger as LC
from verity.protocols.proofs.spot.device.challenger import LiveRecord

import verity_vllm.protocols.pouw.h1.challenger as PHC
import verity_vllm.protocols.pouw.h1.commit as CT
import verity_vllm.protocols.pouw.h1.draw as PHD
import verity_vllm.protocols.pouw.h1.drawn as DN
import verity_vllm.protocols.pouw.h1.drawn_rows as DR
import verity_vllm.protocols.pouw.h1.handoff as HO
import verity_vllm.protocols.pouw.h1.hooks as PHK
import verity_vllm.protocols.pouw.h1.live as LV
import verity_vllm.protocols.pouw.h1.model_key as MK
import verity_vllm.protocols.pouw.h1.prover as PV
import verity_vllm.protocols.pouw.h1.registered as NR
import verity_vllm.protocols.pouw.h1.requests as RQ
import verity_vllm.protocols.pouw.h1.served as SV
import verity_vllm.protocols.pouw.h1.served_files as SF
import verity_vllm.protocols.pouw.h1.static as ST
import verity_vllm.protocols.pouw.h1.words as W
from tests.protocols import challenger_host as CH
from tests.protocols import test_pouw_h1_commit as TC
from tests.protocols import test_pouw_pearl_c_served as PCS
from tests.protocols.test_pouw_h1 import (
    KEY,
    SALT,
    K,
    Linear,
    Runner,
    _admitted_run,
    _entry,
    _host,
    _opening,
    _remote,
    _rounds,
    _salted,
    _serve_window,
    _settings,
    _so,
    _tap,
)
from verity_vllm import protocols as PO
from verity_vllm.errors import Refused
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import h1 as PH

#: what `LiveOpening.from_json` says of a coin that is no opening
NO_OPENING = ("a live coin opening is {after, coin, epoch, index, label, path, salt}: integers, a string, lowercase hex and "
              "a path of SHA-512 digests")
#: what `LiveCommitment.from_json` says of coins that are no commitment
NO_COMMITMENT = ("a live coin commitment is {epoch, families, root, scheme}: families a list of [label, count], root "
                 "lowercase hex")


@pytest.fixture(scope="module")
def window():
    """`test_pouw_h1`'s admitted window (request 0 in round 0 at slots 0 to 2; 1 and 2 in round 1 at slots 1 and 2, beside
    0's decode) and its round coins' commitment. No test changes it."""
    run, rounds = _admitted_run()
    return run, rounds.challenger.commitment


def _tables(run):
    """The window's request and slot tables as the honest prover committed them, by round."""
    return ({r: RQ.request_table(SALT, r, run.rows, run.admitted.get(r, []), KEY) for r in range(run.steps)},
            {r: RQ.slot_table(SALT, r, run.rows, run.owners[r], KEY) for r in range(run.steps)})


def _committed(run, tables, slots):
    """A prover's round log over its own request tables `tables` and slot tables `slots`, each round's root over its parts
    and recorded before its live coin, and those coins' commitment: what a prover that commits what it likes shows."""
    live, manifest, log = _rounds(seed=1), run.manifest(), []
    for r in range(run.steps):
        calls = RQ.step_digest(SALT, r, [m for m in manifest if m["step"] == r])
        row = RQ.round_row(SALT, r, calls, tables[r].commitment.root, slots[r].commitment.root,
                           bytes.fromhex(run.round_log[r]["receipts"]), {})
        log.append({**row, "coin": live.record(bytes.fromhex(row["root"])).to_json()})
    return log, live.challenger.commitment


def _calls(run, s):
    return [m for m in run.manifest() if m["step"] == s]


def _wide(run):
    """Round 0's slot table with slot 0's leaf five bytes long, which opens under the table and is no slot leaf."""
    return RQ.hiding_rows(RQ.slots_binding(SALT, 0), 0, [b"\1\0\0\0\0"] + [RQ.EMPTY_SLOT] * (run.rows - 1), KEY,
                          RQ.SLOTS_NONCE)


def test_an_entry_whose_first_slot_names_another_request_or_none_or_does_not_open_is_refused(window):
    """An entry says where its request was first served, and the round's slot table says whose row each slot is. A prover
    that commits an entry naming a slot the slot table gives to another request, or to none, its roots and coins in
    order, is refused at that slot; so is a first slot that doesn't open under the round's slot table, and one whose
    leaf, though it opens, is no slot leaf."""
    run, coins = window
    e0, e1, e2 = run.requests
    requests, slots = _tables(run)
    for slot, named in ((0, "request 0"), (5, "no request")):
        moved = dataclasses.replace(e1, slot=slot)
        table = RQ.request_table(SALT, 1, run.rows, [moved, e2], KEY)
        log, own = _committed(run, {**requests, 1: table}, slots)
        assert RQ.request_reason(log, own, SALT, run.rows, moved, _opening(table, 1, 0), _opening(slots[1], 1, slot),
                                 RQ.PER_ROUND) == (
            f"slot {slot} of step 1 names {named}, not request 1, whose entry says it was first served there")
        assert RQ.request_reason(log, own, SALT, run.rows, e2, _opening(table, 1, 1), _opening(slots[1], 1, 2),
                                 RQ.PER_ROUND) is None

    def why(first):
        return RQ.request_reason(run.round_log, coins, SALT, run.rows, e0, run.open_request(0), first, RQ.PER_ROUND)
    first = run.open_first(0)
    assert why(first) is None
    unopened = "request 0: slot 0 doesn't open under round 0's slot table"
    assert why(dataclasses.replace(first, value=run.open_first(1).value)) == unopened
    assert why(dataclasses.replace(first, path=first.path[:-1])) == unopened
    wide = _wide(run)
    log, own = _committed(run, requests, {**slots, 0: wide})
    assert RQ.request_reason(log, own, SALT, run.rows, e0, _opening(requests[0], 0, 0), _opening(wide, 0, 0),
                             RQ.PER_ROUND) == "request 0: slot 0 of round 0: a slot leaf is 4 bytes, not 5"


def test_a_slot_is_held_to_its_steps_calls_its_rounds_slot_table_and_its_requests_entry(window):
    """`slot_reason` refuses a slot shown with another step's calls, one that doesn't open under its round's slot table
    or whose leaf is no slot leaf, and a served slot whose request's entry isn't the one opened beside it: none, or
    another request's."""
    run, coins = window
    e0, e1, _e2 = run.requests
    slot = run.open_slot(0, 0)

    def why(slot, calls=None, entry=None, log=run.round_log, coins=coins):
        opened = (entry, run.open_request(entry.index), run.open_first(entry.index)) if entry else (None, None, None)
        return RQ.slot_reason(log, coins, SALT, run.rows, _calls(run, 0) if calls is None else calls, slot, *opened,
                              admission=RQ.PER_ROUND)
    assert why(slot, entry=e0) is None
    assert why(slot, _calls(run, 1), e0) == "step 0's calls are not the ones under round 0's root"
    assert why(slot, [], e0) == "step 0's calls are not the ones under round 0's root"
    assert why(dataclasses.replace(slot, value=run.open_slot(1, 1).value), entry=e0) == (
        "slot 0 doesn't open under round 0's slot table")
    unopened = "slot 0 of step 0 names request 0, and request 0's entry isn't opened beside it"
    assert why(slot) == unopened and why(slot, entry=e1) == unopened
    requests, slots = _tables(run)
    wide = _wide(run)
    log, own = _committed(run, requests, {**slots, 0: wide})
    assert why(_opening(wide, 0, 0), log=log, coins=own) == "slot 0 of round 0: a slot leaf is 4 bytes, not 5"


def test_a_round_whose_row_or_coin_is_malformed_is_refused_by_name(window):
    """A round's row that lacks a part, or whose part isn't hex or whose A segments aren't ``{value: [[level, index,
    root]]}``, is refused as malformed with what was wrong, and so is a round whose coin is no opening: under a request
    (`request_reason`), under a slot (`slot_reason`) and over the whole log (`rounds_reason`), which also refuses a
    round with no coin."""
    run, coins = window
    log, e0 = run.round_log, run.requests[0]
    for row, what in (({k: v for k, v in log[0].items() if k != "calls"}, "'calls'"),
                      ({**log[0], "inner": "xy"}, "non-hexadecimal number found in fromhex() arg at position 0"),
                      ({**log[0], "a": [1]}, "'list' object has no attribute 'items'"),
                      ({**log[0], "a": {"v": [[0, 0]]}}, "not enough values to unpack (expected 3, got 2)")):
        assert RQ.round_reason(SALT, row) == f"a round of the log is malformed ({what})"
    assert RQ.round_reason(SALT, log[0]) is None
    for coin in ({"bad": 1}, 7):
        bent = [{**log[0], "coin": coin}, *log[1:]]
        assert RQ.request_reason(bent, coins, SALT, run.rows, e0, run.open_request(0), run.open_first(0),
                                 RQ.PER_ROUND) == f"request 0: round 0's coin: {NO_OPENING}"
        assert RQ.slot_reason(bent, coins, SALT, run.rows, _calls(run, 0), run.open_slot(0, 0), e0, run.open_request(0),
                              run.open_first(0), admission=RQ.PER_ROUND) == f"round 0's coin: {NO_OPENING}"
        assert RQ.rounds_reason(bent, coins) == NO_OPENING
    assert RQ.rounds_reason([log[0], {**log[1], "coin": None}, log[2]], coins) == "a round of the log has no live coin"
    assert RQ.rounds_reason(log, coins) is None


def test_a_logged_round_without_a_call_and_a_call_whose_a_root_is_not_hex_are_refused(window):
    """`tables_reason` refuses a record whose log closes a round no call of it is at, and a call whose A table root is
    128 characters that aren't hex, as it refuses one with none."""
    run, _coins = window
    manifest = run.manifest()
    assert RQ.tables_reason(SALT, run.round_log, manifest) is None
    assert RQ.tables_reason(SALT, run.round_log, [m for m in manifest if m["step"] != 1]) == (
        "round 1's step has no call, so its slot table names leaves of no committed A table")
    first = manifest[0]
    unhexed = [{**first, "tables": {**first["tables"], "a": "zz" * 64}}, *manifest[1:]]
    assert RQ.tables_reason(SALT, run.round_log, unhexed) == (
        f"call {first['index']} of step {first['step']} has no A table root, so round {first['step']}'s slot table "
        "names A-table leaves that no round root commits")


def test_an_admission_off_the_list_a_slot_leaf_of_another_width_and_a_round_past_its_leaves_raise(window):
    """What a caller hands wrongly raises: an ``admission`` that is neither of `ADMISSIONS`, a slot leaf's preimage that
    isn't `EMPTY_SLOT`'s four bytes, and more entries than a round's table has leaves."""
    run, coins = window
    e0 = run.requests[0]
    with pytest.raises(ValueError, match=r"^an admission is one of \('round-0', 'per-round'\), not 'every'$"):
        RQ.request_reason(run.round_log, coins, SALT, run.rows, e0, run.open_request(0), run.open_first(0), "every")
    for leaf in (b"", bytes(3), bytes(5)):
        with pytest.raises(ValueError, match=f"^a slot leaf is 4 bytes, not {len(leaf)}$"):
            RQ.slot_owner(leaf)
    assert RQ.slot_owner(RQ.EMPTY_SLOT) is None and RQ.slot_owner(RQ.slot_leaf(6)) == 6
    with pytest.raises(ValueError, match="^round 1 admits 3 requests, past its 2 leaves$"):
        RQ.request_table(SALT, 1, 2, run.requests, KEY)
    assert RQ.request_table(SALT, 1, 3, run.requests, KEY).commitment.count == 3


def test_a_table_of_no_rows_is_cores_empty_tree():
    """`hiding_rows` of no values is `audit.commit_rows` of none: the domain's empty root over a count of 0."""
    got = RQ.hiding_rows(SALT, 5, [], KEY, 9)
    assert got.commitment == commit_rows(SALT, 5, [], hash=HM96, salt_key=KEY, nonce=9).commitment
    assert got.commitment.count == 0


def test_h1s_settings_refuse_a_steps_count_and_round_coins_that_are_not_a_live_epochs_rounds():
    """``steps`` is a positive integer, and ``rounds`` records a live epoch's round coins: a record of that epoch's
    salt coin, or of round coins in an epoch of another layout, is refused with which it is."""
    for steps in (0, -3, True, 2.0, "3"):
        with pytest.raises(ValueError, match=r"^pouw setting 'steps' is a positive integer \(H1's S: the window's engine "
                                             r"steps\)$"):
            pouw.Settings.of({"steps": steps})
    assert pouw.Settings.of({"steps": 3}).steps == 3
    live = Wd.live_challenger(0, 3, random.Random(0).randbytes)
    with pytest.raises(ValueError, match=f"^pouw setting 'rounds' is a LiveRecord of a live epoch's {Wd.ROUND} coins$"):
        pouw.Settings.of({"rounds": LiveRecord(live, Wd.SALT)})
    two_salts = LC.LiveChallenger(Wd.LIVE_SCHEME, 0, ((Wd.SALT, 2), (Wd.ROUND, 3)), CH.entropy(1))
    with pytest.raises(ValueError, match=f"^pouw setting 'rounds' is a LiveRecord of a live epoch's {Wd.ROUND} coins: the "
                                         "coin commitment's coins are not a live epoch's: one salt, then the rounds'$"):
        pouw.Settings.of({"rounds": LiveRecord(two_salts, Wd.ROUND)})


def test_a_sample_session_or_a_period_without_what_it_goes_with_is_refused():
    """``draw_session`` is the sample session of H1's window on its challenger, and ``period`` goes with a Pearl-C
    install's ``challenger`` and ``session``: H1 without a challenger, and a scheme that is neither, are refused each by
    name."""
    def refused(entry, settings):
        return PO.refusals(PO.Context(stage="serve", enabled=(entry,), settings=settings))
    other = {"pouw": {"coins": Challenger(0, CH.entropy(0))}}
    period = {"period": "p1", "lane": "vllm"}
    for entry, settings in ((_entry(), _settings()), ("pouw:ncp-v1", other)):
        scheme = entry.partition(":")[2]
        assert refused(entry, settings) == ()
        no_window = (f"{entry}: {scheme}: the pouw setting 'draw_session' is the draw session of H1's window "
                     f"({PH.SCHEME}) on the challenger at 'challenger'")
        assert refused(entry, {"pouw": {**settings["pouw"], "draw_session": {"session": CH.session_id(0xd)}}}) == (no_window,)
        no_install = (f"{entry}: {scheme}: the pouw setting 'period' is the lane's open period that a Pearl-C install's "
                      "reserved session joins, so it goes with 'challenger' and 'session', for pearl-c-sm120-v1-h3, "
                      "pearl-c-sm120-unpromoted-v1-h3, pearl-c-h100-v1-h3 only")
        assert refused(entry, {"pouw": {**settings["pouw"], "period": period}}) == (no_install,)


def test_input_ids_past_their_slots_or_not_a_token_id_are_refused():
    """`ids_words` takes at most its slots' ids, each None or a u32 below `EMPTY_ID`, the word an empty slot holds."""
    with pytest.raises(ValueError, match="^3 input ids, past the Program's 2 slots$"):
        W.ids_words([1, 2, 3], 2)
    for bad in (W.EMPTY_ID, 1 << 32, -1):
        with pytest.raises(ValueError, match=f"^an input id is not a u32 below {W.EMPTY_ID}$"):
            W.ids_words([1, bad], 2)
    assert W.ids_words([1, None, W.EMPTY_ID - 1], 4).tolist() == [[1, 0], [0xFFFF, 0xFFFF], [0xFFFE, 0xFFFF],
                                                                  [0xFFFF, 0xFFFF]]


def test_a_window_or_a_declaration_off_its_rows_and_steps_is_refused():
    """A window's u is over at most S steps of at most T rows each; a declaration's T is a positive multiple of 32; and
    served steps past the declared window's are named as that, not step by step."""
    for step_rows in ([1, 2, 3], [1, 33], [1, -1]):
        with pytest.raises(ValueError, match=rf"^\[{', '.join(map(str, step_rows))}\] is not a window of 2 steps of at "
                                             "most 32 rows$"):
            ST.utilization(step_rows, 32, 2)
    assert ST.utilization([32, 0], 32, 2) == ST.utilization([16, 16], 32, 2) == ST.utilization([32], 32, 2) == Fraction(1, 2)
    for rows in (40, 0, -32):
        with pytest.raises(ValueError, match=f"^T = {rows} rows is not a positive multiple of 32$"):
            ST.Static((1,), 1, rows, 1)
    declared = ST.Static((2, 1), 2, 32, 3)
    assert declared.check([3, 2, 0]) == [] and declared.check([3, 2]) == []
    assert declared.check([3, 2, 0, 0]) == ["4 served steps past the window's 3"]


def test_a_model_key_answer_without_a_hex_key_or_its_digests_is_refused():
    """The gate's ``model-key`` answer carries its key in hex and names its model, its circuit and its weights' root,
    each digest 128 lowercase hex characters."""
    key = bytes(range(32))
    answer = {"model": "M", "circuit": "ab" * 64, "weights_root": "cd" * 64, "key": key.hex(),
              "key_sha512": hashlib.sha512(key).hexdigest()}
    assert MK.model_key(answer) == MK.ModelKey("M", "ab" * 64, "cd" * 64, key)
    for edit in ({"key": None}, {"key": "zz" * 32}, {"key": 7}):
        with pytest.raises(ValueError, match="^model-key: the answer has no key in hex$"):
            MK.model_key({k: v for k, v in {**answer, **edit}.items() if v is not None})
    for named in ({"circuit": "AB" * 64}, {"circuit": "ab" * 63}, {"weights_root": "cd" * 65}, {"weights_root": 7},
                  {"model": None}, {"model": b"M"}):
        with pytest.raises(ValueError, match=r"^model-key: the answer names no model, circuit and weights' root "
                                             r"\(SHA-512 digests in hex\)$"):
            MK.model_key({**answer, **named})


def test_a_tile_unit_a_row_schema_and_a_public_salt_past_its_table_are_refused():
    """The pouw setting ``unit`` is v1, the default, or j10r; a row table's schema is one of the two row leaves; and a
    public table has its rows' salts and no more."""
    assert SV.unit_of(None) is None and SV.unit_of("v1") is None and SV.unit_of("j10r") is not None
    with pytest.raises(ValueError, match="^no tile unit 'j11': the pouw setting 'unit' is v1 or j10r$"):
        SV.unit_of("j11")
    words: np.ndarray = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert NR.row_digests(words).shape == (3, 64)
    with pytest.raises(ValueError, match="^rows of 'row/v3': hm96-sha512/row/v2 or hm96-sha512/row-seg/v1$"):
        NR.row_digests(words, "row/v3")
    salts = NR.PublicSalts(b"head", 5, 2)
    assert len(salts) == 2 and salts[-1] == salts[1] != salts[0] and salts[0:2] == [salts[0], salts[1]]
    for j in (2, -3):
        with pytest.raises(IndexError, match=f"^{j}$"):
            salts[j]


def test_a_window_is_whole_tiles_under_a_salt_or_an_opener_and_a_proof_block_is_one_of_its_calls():
    """A window's T is a positive multiple of 32; its salt is given or opened by its opener, never both and never
    neither; a model with no linear layer installs none; and a proof block past the calls' units is no call's."""
    assert PH.layout([(32, K), (40, K)], 32) == ([0, PH.units(K, 32, 32)], PH.units(K, 32, 32) + PH.units(K, 40, 32))
    for rows in (33, 0, -32):
        with pytest.raises(ValueError, match=f"^T = {rows} rows is not a positive multiple of 32$"):
            PH.layout([(32, K)], rows)
    for salt, opener in ((None, None), (SALT, lambda requests: None)):
        with pytest.raises(ValueError, match="^t: a window's salt is given or opened by its opener, one of the two$"):
            PH.H1Run("t", salt, {0: "q"}, [(32, K)], 32, 1, opener=opener)
    with pytest.raises(ValueError, match=f"^pouw:{PH.SCHEME}: the model has no linear layer$"):
        PHK.layers(torch.nn.Sequential(torch.nn.ReLU()), (Linear,))
    manifest = [{"index": 0, "base": 0, "units": 2}, {"index": 7, "base": 2, "units": 3}]
    assert [PV.locate(manifest, unit) for unit in (0, 1, 2, 4)] == [(0, 0), (0, 1), (7, 0), (7, 2)]
    for unit in (5, -1):
        with pytest.raises(ValueError) as refused:
            PV.locate(manifest, unit)
        assert str(refused.value) == f"unit {unit} is no call's: the window's units are its calls' [base, base + units)"


def test_a_closed_window_takes_no_admission_and_no_call_and_names_only_its_calls(window):
    """After its close a window refuses an admission and a call, each by name, and it has no call past the ones it
    recorded."""
    run, _coins = window
    assert run.closed and [c.index for c in run.calls] == list(range(6)) and run.call(5) is run.calls[5]
    with pytest.raises(KeyError, match="^'t: no call 6'$"):
        run.call(6)
    with pytest.raises(RuntimeError, match="^t: an admission after the window closed$"):
        run.admit([("d", [7], 1)], [("d", 1)])
    with pytest.raises(RuntimeError, match="^t: a call after the window closed$"):
        run.record(0, *_tap(np.random.default_rng(0), 1, 32))
    assert len(run.requests) == 3 and len(run.calls) == 6


def _driven(*steps, salt=SALT, admission=RQ.PER_ROUND, **kw):
    """`test_pouw_h1._admitted_run`'s window of 3 rounds over two linears, unclosed, after `admitting` took each of the
    scheduler outputs ``steps`` and both linears served the step's rows."""
    rng, runner = np.random.default_rng(12), Runner()
    run = PH.H1Run("t", salt, {0: "q", 1: "o"}, [(32, K), (40, K)], 32, 3, rounds=_rounds(), key=KEY, admission=admission,
                   **kw)
    step = PHK.admitting(run, runner.execute_model, runner)
    for so in steps:
        step(so)
        for w, n in ((0, 32), (1, 40)):
            run.record(w, *_tap(rng, so.total_num_scheduled_tokens, n))
    return run


def test_a_step_scheduled_twice_before_its_forward_admits_once_and_is_a_gap():
    """A second admission into a step whose forward hasn't run admits nothing, and the step is named among the window's
    request gaps, so the window credits nothing."""
    run = _driven()
    assert [(e.index, e.round, e.slot) for e in run.admit([("a", [1, 2, 3], 4)], [("a", 3)])] == [(0, 0, 0)]
    assert run.request_gaps() == []
    assert run.admit([("b", [4], 2)], [("b", 1)]) == []
    assert run.request_gaps() == ["step 0: scheduled twice before its forward"]
    assert run.owners == {0: [0, 0, 0]} and [e.prompt for e in run.requests] == [(1, 2, 3)]


def test_a_window_that_commits_nothing_admits_nothing():
    """A window installed with ``commit`` off has no request table: an admission gives no entry and names no slot."""
    run = _driven(commit=False)
    assert run.admit([("a", [1, 2, 3], 4)], [("a", 3)]) == []
    assert run.requests == [] and run.owners == {} and run.request_gaps() == []


@pytest.mark.parametrize("admission", [RQ.ROUND0, RQ.PER_ROUND])
def test_a_window_whose_opener_refused_its_salt_opens_it_once(admission):
    """A window opens its salt once: after its opener refused, the window records why, and it never asks the opener
    again, at a later admission or for its salt."""
    asked = []

    def shut(requests):
        asked.append(requests)
        raise LC.CoinError("the gate is shut")
    run = _driven(salt=None, opener=shut, admission=admission)
    with pytest.raises(LC.CoinError, match="^the gate is shut$"):
        run.admit([("a", [1, 2, 3], 4)], [("a", 3)])
    assert run.details["unopened"] == "the gate is shut" and len(asked) == 1
    for again in (run.open_window, lambda: run.salt):
        with pytest.raises(RuntimeError) as refused:
            again()
        assert str(refused.value) == "t: the window's salt was refused (the gate is shut), and a window opens it once"
    run.finish()
    assert len(asked) == 1 and "draw" not in run.commits and "calls_root" not in run.commits


def test_a_sampling_coin_its_session_refuses_is_the_samples_refusal_beside_its_root():
    """`finish` records the sample root it asked the coin after, and the session's refusal in place of the coin."""
    asked = []

    class Shut:
        def record(self, root):
            asked.append(root)
            raise LC.CoinError("the session is closed")
    run = _driven(_so(("a", [1, 2, 3], 4)), _so(running={"a": 1}), _so(running={"a": 1}), draw=Shut())
    run.finish()
    sample = run.commits["draw"]
    root = RQ.draw_root(bytes.fromhex(run.commits["calls_root"]), bytes.fromhex(sample["receipts"]), run.round_log)
    assert asked == [root] and len(run.round_log) == 3
    assert sample == {"root": root.hex(), "receipts": run.receipts_of(2).hex(), "coin": None,
                      "refused": "the session is closed"}


def test_a_resumed_requests_ids_are_its_prompt_again_and_a_row_with_no_entry_has_none_to_be():
    """`ids_reason` holds each prefill row's id to the next prompt token of the request its slot names, from token 0
    again in the round a preempted request is resumed, and reads no id of a row no entry names; a round whose ids
    aren't one a slot is refused."""
    run = _driven(_so(("a", [1, 2, 3], 4), ("z", [7, 8], None)), _so(("a", [1, 2, 3], 4), ("b", [4], 2), running={"z": 1}),
                  _so(running={"a": 1, "b": 1}))
    assert run.owners == {0: [None, None, 0, 0, 0], 1: [None, 1, 0, 0, 0], 2: [0, 1]}
    assert run.readmitted == [{"round": 1, "index": 0}] and [e.prompt for e in run.requests] == [(1, 2, 3), (4,)]
    honest = {0: [7, 8, 1, 2, 3], 1: [0, 4, 1, 2, 3], 2: [0, 0]}

    def reason(**rounds):
        return CT.ids_reason(run, CT.CommitTap(round_ids={**honest, **{int(r[1:]): ids for r, ids in rounds.items()}}))
    assert reason() is None and reason(r0=[9, 9, 1, 2, 3], r1=[9, 4, 1, 2, 3], r2=[9, 9]) is None
    off = "layer 0's input is not its round's request table"
    assert reason(r1=[0, 4, 1, 9, 3]) == (
        f"roots1: step 1's slot 3 has input id 9 in the Commit, and request 0's prompt token 1 is 2: {off}")
    assert reason(r0=[1, 2, 3, 7, 8]) == (
        f"roots1: step 0's slot 2 has input id 3 in the Commit, and request 0's prompt token 0 is 1: {off}")
    assert reason(r2=[0]) == "roots1: step 2's 1 input ids in the Commit are not its 2 slots of the batch"
    assert reason(r1=[]) == "roots1: step 1's 0 input ids in the Commit are not its 5 slots of the batch"


def test_a_scale_the_commit_acquires_with_no_codes_of_its_step_is_refused():
    """The Commit's tap takes a site's s_a only after its x_q of the same engine step: one at a later step, or at a site
    with no x_q, is refused by name."""
    tap = CT.CommitTap()
    tap.take("m/row_quant/0", np.zeros((2, 3), np.int8), 4)
    tap.take("m/row_quant/1", np.ones(2, "<f4"), 4)
    assert tap.refused == []
    tap.take("m/row_quant/1", np.ones(2, "<f4"), 5)
    tap.take("n/row_quant/1", np.ones(2, "<f4"), 5)
    assert tap.refused == ["m/row_quant: an s_a at engine step 5 with no x_q",
                           "n/row_quant: an s_a at engine step 5 with no x_q"]


def test_round_0_admits_at_most_its_leaves_and_a_round_states_no_live_unit_before_it_closes():
    """``requests@0`` has T rows, so round 0 admits at most T requests, and ``live@<s>`` reads round s's slot table,
    which is committed when the round closes."""
    entry = RQ.RequestEntry(0, 0, 0, 4, (1, 2, 3))
    assert LV.request_words([entry], 2).tolist() == [[0, 0, 0, 0, 0, 0, 4, 0, 3, 0], [0] * 10]
    with pytest.raises(ValueError, match="^round 0 admits 2 requests, past its 1 leaves$"):
        LV.request_words([entry, entry], 1)
    run = _driven(_so(("a", [1, 2, 3], 4)), admission=RQ.ROUND0)
    assert [x["round"] for x in run.round_log] == [0]
    for s in (1, 2):
        with pytest.raises(ValueError, match=f"^round {s} hasn't closed, so its slot table isn't committed$"):
            LV.statement(run, s)
    unserved = _driven(admission=RQ.ROUND0)
    assert unserved.round_log == []
    with pytest.raises(ValueError, match="^round 0 hasn't closed, so its slot table isn't committed$"):
        LV.statement(unserved, 0)


def test_a_v1_leafs_statement_is_its_lane_or_its_combine_and_a_lanes_names_its_lane():
    """A sampled v1 leaf has two kinds of statement, a lane's and the combine's; another kind has no rows, and a lane's
    ok is asked of a lane."""
    lane, combine = DR.leaf_rows(LL.LANE, K, (32, 32)), DR.leaf_rows(LL.COMBINE, K, (32, 32))
    assert [port for port, *_ in lane][-4:] == [LL.ADMIT, LL.ZY, LL.PARTIALS, LL.PUB]
    assert [port for port, *_ in combine][-1] == LL.RECEIPT and len(combine) == LL.lanes(32, 32) + 1
    with pytest.raises(ValueError, match=f"^a v1 leaf's statement is its {LL.LANE} or its {LL.COMBINE}, not 'tile'$"):
        DR.leaf_rows("tile", K, (32, 32))
    with pytest.raises(TypeError, match="^a lane's statement names its lane$"):
        DR.leaf_ok(LL.LANE, None, {}, {})


def test_a_statements_port_reads_one_registered_value_and_each_position_as_one_row():
    """`statement_tables` gives a port that reads no registered value its instances' rows as they are, and one that
    does its distinct positions; it refuses a port whose instances read two values, or one position as two rows."""
    z = {"units": [0, 1], "w_x": np.zeros((2, 3), np.uint16), "s_x": [b"a", b"b"]}
    plain = DN.statement_tables(z, ["x"], [])
    assert plain["refs"] == {"x": [0, 1]} and plain["salts"] == {"x": [b"a", b"b"]} and plain["registered"] == {}
    assert plain["tables"]["x"].tolist() == [[0, 0, 0], [0, 0, 0]]
    shared = {**z, "s_x": [b"a", b"a"], "r_x": [4, 4], "v_x": ["v", "v"], "p_x": ["p", "p"]}
    one = DN.statement_tables(shared, ["x"], [])
    assert one["registered"] == {"x": {"value": "v", "positions": [4], "paths": ["p"]}} and one["refs"] == {"x": [0, 0]}
    assert one["tables"]["x"].tolist() == [[0, 0, 0]] and one["salts"] == {"x": [b"a"]}
    for edit in ({"s_x": [b"a", b"b"]}, {"p_x": ["p", "q"]}, {"w_x": np.array([[0, 0, 0], [0, 0, 1]], np.uint16)}):
        with pytest.raises(ValueError, match="^port x: instances 0 and 1 read position 4 as two rows$"):
            DN.statement_tables({**shared, **edit}, ["x"], [])
    with pytest.raises(ValueError) as refused:
        DN.statement_tables({**shared, "v_x": ["v", "u"]}, ["x"], [])
    assert str(refused.value) == ("port x: its instances read ['u', 'v'], and a port reads one registered value: one "
                                  "call's units a statement")


def test_the_samples_k_is_the_one_given_or_the_serve_dirs_periods(tmp_path):
    """`draw_k` is ``--k`` when given, else the ``draw_k`` the serve dir's period file holds; with neither, a period
    file without one, or one that is no integer, it refuses and says where a K comes from."""
    period = tmp_path / DN.PERIOD
    assert DN.draw_k(tmp_path, 7) == 7
    for text, what in ((None, f"[Errno 2] No such file or directory: '{period}'"), ('{"other": 1}', "'draw_k'"),
                       ('{"draw_k": "k"}', "invalid literal for int() with base 10: 'k'")):
        if text is not None:
            period.write_text(text)
        with pytest.raises(Refused) as refused:
            DN.draw_k(tmp_path, None)
        assert str(refused.value) == (f"no draw K: give --k, or a serve dir whose {DN.PERIOD} has the open's draw_k "
                                      f"({what})")
    period.write_text('{"draw_k": 9}')
    assert DN.draw_k(tmp_path, None) == 9 and DN.draw_k(tmp_path, 7) == 7


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A window served under the auditor device's challenger (`test_pouw_h1._serve_window`), its session and its sample
    session ended: the harness's host, the window's commitment, and its record."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    host = _host(tmp_path, monkeypatch)
    commitment, _run = _serve_window(_remote(transcript_dir=str(tmp_path / "w"), served=SV.Served(window=bytes(64))))
    assert host.draw.ended() is None and host.ended() is None
    return host, commitment, json.loads((tmp_path / "w" / PH.CALLS_FILE).read_text())


def _trusted(host, commits):
    return PHC.trusted_rounds_reason(commits, host.directory, [host.ledger.path], host.window, host.key,
                                     pins=[host.ledger.pins_path], main=CH.MAIN, draw_directory=host.draw.directory)


def test_a_windows_coins_that_are_no_commitment_and_rounds_out_of_order_are_refused(served):
    """`trusted_rounds_reason` refuses a window whose ``round_coins`` or ``coins`` aren't a coin commitment at all, and
    one whose round log holds its rounds in another order, before it reads a coin."""
    host, honest, _record = served
    assert _trusted(host, honest) is None
    for name, coins in (("round_coins", {"bad": 1}), ("coins", None)):
        assert _trusted(host, {**honest, name: coins}) == (
            f"the window's {name} are not a live coin commitment ({NO_COMMITMENT}), so not the verifier's challenger's")
    first, second, third = honest["rounds"]
    assert _trusted(host, {**honest, "rounds": [second, first, third]}) == "the round log is not rounds 0, 1, ... in order"


def test_a_windows_sample_is_held_to_its_root_its_parts_and_its_coin(served):
    """The window's sample is refused by name when its root isn't 128 lowercase hex, when a part its root is over is
    absent or isn't hex, when its coin is no opening, and when the challenger's log shows its sampling coin following
    another root, or more than the one."""
    host, honest, _record = served
    sample = honest["draw"]
    for root in ("zz", sample["root"].upper(), None):
        assert _trusted(host, {**honest, "draw": {**sample, "root": root}}) == "the window's draw root is not 128 lowercase hex"
    assert _trusted(host, {**honest, "draw": None}) == "the window's draw root is not 128 lowercase hex"
    for edit, what in (({"draw": {k: v for k, v in sample.items() if k != "receipts"}}, "'receipts'"),
                       ({"calls_root": "zz"}, "non-hexadecimal number found in fromhex() arg at position 0"),
                       ({"calls_root": None}, "fromhex() argument must be str, not None")):
        assert _trusted(host, {**honest, **edit}) == f"the window's draw root is malformed ({what})"
    for coin in ({"bad": 1}, None):
        assert _trusted(host, {**honest, "draw": {**sample, "coin": coin}}) == (
            f"the window's draw coin is malformed ({NO_OPENING})")
    signed, root = host.draw.challenger.commitment, bytes.fromhex(sample["root"])
    assert PHD.draw_reason(honest, signed, (root,)) is None
    for received in ((bytes(64),), (root, root), (bytes(64), root)):
        assert PHD.draw_reason(honest, signed, received) == (
            "the challenger's log does not show the window's draw root as the one root its draw coin followed: a draw "
            "opened before the window's rounds, or after another root")


@pytest.mark.parametrize("families", [((Wd.DRAW, 2),), ((Wd.DRAW, 1), (Wd.SALT, 1))], ids=["two-coins", "a-salt-beside"])
def test_a_sample_session_of_another_layout_is_refused(tmp_path, monkeypatch, families, served):
    """A sample session holds its one sampling coin and nothing else. One the gate opened withheld on the window's
    sample root, whose coin 0 opened after that root and which its client closed, is still refused when its challenger
    signed two sampling coins, or a salt coin beside the one."""
    _first, honest, _record = served
    seed = 2
    opens = PS.window_open_root(bytes.fromhex(honest["weights_root"]), bytes.fromhex(honest["requests_root"]),
                                CH.session_id(seed))
    commits, host = _salted(tmp_path, monkeypatch, "own", seed, honest, opens, opens)
    assert _trusted(host, commits) is None
    monkeypatch.setattr(Wd, "draw_challenger", lambda epoch, entropy: LC.LiveChallenger(Wd.LIVE_SCHEME, epoch, families,
                                                                                         entropy))
    seed = 3
    opens = PS.window_open_root(bytes.fromhex(honest["weights_root"]), bytes.fromhex(honest["requests_root"]),
                                CH.session_id(seed))
    commits, host = _salted(tmp_path, monkeypatch, "wide", seed, honest, opens, opens)
    assert host.draw.challenger.commitment.families == families
    assert _trusted(host, commits) == ("the window's draw session's signed coins: the coin commitment's coins are not a "
                                       "live window's draw session's: its draw coin alone")


def test_the_sample_sessions_files_are_none_where_the_ledger_pins_no_session_on_its_root(served, tmp_path):
    """`draw_files` fetches the session the ledger opens on the window's sample root, pinned; where the ledger opens none
    on that root, or holds no pins log, it fetches nothing and gives None, which `draw_session_reason` refuses."""
    host, honest, _record = served
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "out").symlink_to(host.draw.directory, target_is_directory=True)
    asked = []

    def fetch(session):
        asked.append(session)
        return tmp_path / "tree"
    ledger = PHC.Ledger.load([host.ledger.path], host.key, pins=[host.ledger.pins_path], main=CH.MAIN)
    files = PHD.draw_files(honest, ledger, fetch, PHC.lms_verify)
    assert files is not None and asked == [host.draw.session_id]
    assert PHD.draw_session_reason(honest, host.key, list(ledger.lines), list(ledger.pins), list(ledger.main), files,
                                   PHC.lms_verify) is None
    assert PHD.draw_files({"draw": {"root": "ab" * 64}}, ledger, fetch, PHC.lms_verify) is None
    unpinned = PHC.Ledger.load([host.ledger.path], host.key, main=CH.MAIN)
    assert PHD.draw_files(honest, unpinned, fetch, PHC.lms_verify) is None
    assert asked == [host.draw.session_id]
    assert PHD.draw_session_reason(honest, host.key, list(ledger.lines), list(ledger.pins), list(ledger.main), None,
                                   PHC.lms_verify) == PHD.NO_DRAW_SESSION


def test_a_record_whose_served_trees_or_salt_cant_be_read_is_refused(served):
    """`manifest_reason` refuses a record whose served trees lack their entries or hold what isn't a list of them, and
    `window_evidence` raises on one whose served trees or salt coin can't be read, each with what was wrong."""
    host, _honest, record = served

    def edited(edit):
        doc = json.loads(json.dumps(record))
        edit(doc)
        return doc
    assert PHC.manifest_reason(record) is None
    for edit, what in ((lambda d: d["commitment"]["served"].pop("entries"), "'entries'"),
                       (lambda d: d["commitment"]["served"].update(entries=7), "'int' object is not iterable")):
        assert PHC.manifest_reason(edited(edit)) == f"the window's served trees are malformed ({what})"
    evidence = PHC.window_evidence(record, host.directory, host.window, [], b"d")
    assert evidence["salt"] == PHC.program_salt_of(record["commitment"]).hex() and evidence["descriptor"] == b"d".hex()
    for edit, what in ((lambda d: d["commitment"]["served"].pop("refused"), "'refused'"),
                       (lambda d: d["commitment"].pop("salt_opening"), NO_OPENING),
                       (lambda d: d["commitment"].update(salt_opening={"bad": 1}), NO_OPENING)):
        with pytest.raises(ValueError) as refused:
            PHC.window_evidence(edited(edit), host.directory, host.window, [], b"d")
        assert str(refused.value) == f"the window's served trees or salt can't be read ({what})"


def test_a_statement_names_only_sessions_that_hold_a_session_file(tmp_path):
    """`statement` raises on a sampled proof block's session directory with no ``session.json``, naming it."""
    full, bare = tmp_path / "full", tmp_path / "bare"
    full.mkdir()
    (full / "session.json").write_text("{}")
    assert PHC.statement(3, "c.vyc", "public.json", [full]) == {
        "call": 3, "circuit": "c.vyc", "public": "public.json", "sessions": [str(full)]}
    with pytest.raises(ValueError) as refused:
        PHC.statement(3, "c.vyc", "public.json", [full, bare])
    assert str(refused.value) == f"call 3's session {bare} has no session.json"


@pytest.mark.parametrize("path", ["/" + "s" * HO.PATH_BYTES, "/" + "é" * (HO.PATH_BYTES // 2 + 1)],
                         ids=["a-byte-over", "two-byte-letters"])
def test_a_hand_offs_socket_path_longer_than_an_address_holds_is_refused_with_its_length(path):
    """`Holder` and `send_stream` refuse a path of more bytes than a Unix socket's address holds, in words that give its
    length in bytes, the limit and the path, where the socket's own refusal names none of them. The second path is
    within the limit counted in letters and over it in bytes."""
    nobody: Any = None
    said = (f"the hand-off's socket path is {len(os.fsencode(path))} bytes and a Unix socket's address holds "
            f"{HO.PATH_BYTES}: {path}")
    for refuse in (lambda: HO.Holder(path), lambda: HO.send_stream(nobody, path)):
        with pytest.raises(OSError) as refused:
            refuse()
        assert str(refused.value) == said


def test_a_hand_offs_socket_path_of_as_many_bytes_as_an_address_holds_carries_the_stream(tmp_path, monkeypatch):
    """The limit is the socket's own: a path of exactly ``PATH_BYTES`` binds, and the stream handed to it is taken, where
    a socket refuses one byte more. The paths are relative, so the test's directory, however long, is no part of them."""
    monkeypatch.chdir(tmp_path)
    with socket.socket(socket.AF_UNIX) as bare, pytest.raises(OSError, match="AF_UNIX path too long"):
        bare.bind("s" * (HO.PATH_BYTES + 1))
    holder = HO.Holder("s" * HO.PATH_BYTES)
    try:
        with socket.socket() as stream:
            client: Any = SimpleNamespace(_wire=SimpleNamespace(sock=stream))
            HO.send_stream(client, holder.path)
    finally:
        fd = holder.take()
    assert fd is not None and holder.error is None and holder.path.is_socket()
    os.close(fd)


def test_a_holder_handed_no_stream_or_another_message_takes_none_and_says_what_it_got(tmp_path, monkeypatch):
    """A holder nobody connected to has no descriptor, and the error its listener ended with. One handed a message
    that isn't the stream's, with a descriptor or with none, takes none, answers nothing and says what it was handed;
    the descriptor it was handed it closes, so the stream's other end reads its end once the sender's copy is closed."""
    monkeypatch.chdir(tmp_path)
    idle = HO.Holder("idle")
    assert idle.take() is None and idle.fd is None and idle.error
    for name, message, sent in (("bare", b"x", 0), ("other", bytes(len(HO.STREAM)), 1)):
        holder = HO.Holder(name)
        other_end, stream = socket.socketpair()
        with socket.socket(socket.AF_UNIX) as sender, other_end, stream:
            sender.connect(name)
            socket.send_fds(sender, [message], [stream.fileno()][:sent])
            assert sender.recv(len(HO.TAKEN)) == b""
            assert holder.take() is None and holder.fd is None
            stream.close()
            other_end.setblocking(False)
            assert other_end.recv(1) == b""
        assert holder.error == f"the hand-off carried {message!r} and {sent} descriptors, not the session's stream"


@pytest.mark.parametrize("answer", [b"", b"given"], ids=["no-answer", "another-answer"])
def test_a_stream_its_listener_does_not_answer_as_taken_is_refused(tmp_path, monkeypatch, answer):
    """`send_stream` returns only once the holder answered that it has the stream: a listener that reads the hand-off
    and closes, or answers anything else, is refused with its path."""
    monkeypatch.chdir(tmp_path)
    with socket.socket(socket.AF_UNIX) as listener, socket.socket() as stream:
        listener.bind("s")
        listener.listen(1)

        def serve():
            conn, _ = listener.accept()
            with conn:
                _message, fds, _flags, _address = socket.recv_fds(conn, len(HO.STREAM), 1)
                for fd in fds:
                    os.close(fd)
                conn.sendall(answer)
        # a daemon, joined only under the refusal its answer makes: a stream refused before the listener is reached
        # fails the test and leaves no thread to wait for
        served_by = threading.Thread(target=serve, daemon=True)
        served_by.start()
        client: Any = SimpleNamespace(_wire=SimpleNamespace(sock=stream))
        with pytest.raises(OSError) as refused:
            HO.send_stream(client, "s")
        assert str(refused.value) == "the holder at s didn't take the session's stream"
        served_by.join()


def test_only_a_window_on_the_auditor_hosts_coins_hands_off_and_only_a_saved_window_resumes(tmp_path):
    """`client_of`, and so `suspend`, refuses a window whose coins are held in the serving process, and saves nothing;
    `resume` refuses a file that isn't a saved window's before it reads the stream's descriptor."""
    run = _driven()
    for refuse in (lambda: HO.client_of(run), lambda: HO.suspend(run, tmp_path / "saved")):
        with pytest.raises(ValueError, match="^t: only a window whose coins are the auditor host's session hands off$"):
            refuse()
    assert not (tmp_path / "saved").exists()
    (tmp_path / HO.FILE).write_bytes(pickle.dumps({"schema": "another/v1"}))
    with pytest.raises(ValueError) as refused:
        HO.resume(tmp_path, -1)
    assert str(refused.value) == f"{tmp_path}: not a {HO.SCHEMA} window"


def test_an_unclosed_windows_trees_are_not_read_as_its_registered_values():
    """A window's served trees are all committed at its close: before it, `pouw_h1_served_files` reads none of them."""
    run = _driven(served=SV.Served(window=bytes(64)))
    assert not run.closed
    for read in (lambda: SF.trees(run), lambda: SF.reads_file(run, 0)):
        with pytest.raises(ValueError, match="^t: the window isn't closed, so its tables aren't all committed$"):
            read()


def test_a_periods_weights_are_each_linears_int8_codes_and_scales_at_an_even_k():
    """`register_weights` refuses an odd K, whose row isn't whole u16 words, and a weight that isn't its linear's
    n × K int8 codes and n scales, naming the linear."""
    circuit = hashlib.sha512(b"a declared Program").hexdigest()
    codes: np.ndarray = np.zeros((3, 4), np.int8)
    scales: np.ndarray = np.ones(3, "<f4")
    period = SV.register_weights(circuit, [(3, 4)], {0: (codes, scales)}, KEY)
    assert sorted(period.registered) == [4] and period.registered[4].leaves == 3
    with pytest.raises(ValueError, match="^K = 5: a row's codes are whole u16 words$"):
        SV.register_weights(circuit, [(3, 5)], {0: (np.zeros((3, 5), np.int8), scales)}, KEY)
    for weight in ((codes.astype(np.int16), scales), (codes[:2], scales), (codes, scales[:2])):
        with pytest.raises(ValueError, match="^linear 0's weight is not its 3 × 4 int8 codes and 3 scales$"):
            SV.register_weights(circuit, [(3, 4)], {0: weight}, KEY)
    with pytest.raises(ValueError, match="^linear 1's weight is not its 2 × 4 int8 codes and 2 scales$"):
        SV.register_weights(circuit, [(3, 4), (2, 4)], {0: (codes, scales), 1: (codes, scales)}, KEY)


def test_a_value_holds_one_row_for_each_of_its_rows():
    """`pouw_h1_registered.commit` takes the rows a value holds only when there is one for each row it commits."""
    circuit = hashlib.sha512(b"a declared Program").hexdigest()
    words: np.ndarray = np.arange(12, dtype=np.uint16).reshape(3, 4)
    held = NR.commit(circuit, "v", words, KEY, 5, rows=NR.Rows(words, 3))
    assert held.leaves == 3 and [list(row) for row in held.rows] == words.tolist()
    for rows in (NR.Rows(words[:2], 2), [0, 1, 2, 3]):
        with pytest.raises(ValueError, match=f"^{len(rows)} rows held for a value of 3$"):
            NR.commit(circuit, "v", words, KEY, 5, rows=rows)


def _salt_pass(tmp_path, monkeypatch, challenger, after=None):
    """A Pearl-C pass over `test_pouw_pearl_c_served`'s weights, epoch 0, with what `salt_session_reason` reads of the
    pass its install records, and its session on a gate's ledger: reserved with the salt coin alone, opened on the
    epoch's root R, served by ``challenger``, its salt coin taken after ``after`` (R unless given), closed by its
    client and ended. The pass, the gate, the session, R."""
    gate = CH.Gate(PCS.PRIVATE, PCS.KEY)
    session = PCS._GateSession(tmp_path, monkeypatch, gate, Wd.SALT_FAMILIES, challenger)
    weights = weights_root(PCS.ROOTS)
    root = PS.epoch_open_root(weights, 0)
    answer = session.open(PCS.SID, root.hex())
    opening = LiveRecord(session.client, Wd.SALT).record(root if after is None else after)
    session.client.close()
    assert session.end() is None
    coins = challenger.commitment
    served_pass = {"weights_root": weights.hex(), "pass_index": 0, "coins": coins.to_json(),
                   "salt_opening": opening.to_json(), "salt": salt_of(coins, opening.coin, weights).hex(),
                   "session": PHC.carried(PHC.session_of(answer))}
    return served_pass, gate, session, root


def test_a_pearl_c_pass_is_held_to_a_hex_weights_root_and_to_coins_and_a_salt_coin_that_open(tmp_path, monkeypatch):
    """`salt_session_reason` refuses a pass whose weights' root isn't 128 lowercase hex before it looks for a session,
    one whose coins or salt opening don't parse, and one whose salt opening isn't the session's salt coin 0 opened
    after its epoch's root: opened after another root, another coin of the epoch, or a coin that doesn't open."""
    served_pass, gate, session, _root = _salt_pass(tmp_path, monkeypatch, PCS.salt_only(5))
    ledger, fetch = gate.ledger(), {PCS.SID: session.tree}.get

    def why(**edit):
        return PHC.salt_session_reason({**served_pass, **edit}, ledger, fetch)
    assert why() is None
    weights = served_pass["weights_root"]
    for root in ("zz", weights.upper(), weights[:-2], None, 7):
        assert why(weights_root=root) == "the pass's weights' root is not 128 lowercase hex, so no session was opened on it"
    for coins in ({"bad": 1}, None):
        assert why(coins=coins) == f"the pass's coins are malformed ({NO_COMMITMENT})"
        assert why(salt_opening=coins) == f"the pass's coins are malformed ({NO_OPENING})"
    opening = served_pass["salt_opening"]
    for edit, said in (({"after": "00" * 64}, "coin was not opened after what the verifier recorded"),
                       ({"index": 1}, f"coin is the epoch's {Wd.SALT} coin 1, not its {Wd.SALT} coin 0"),
                       ({"label": Wd.ROUND}, f"coin is the epoch's {Wd.ROUND} coin 0, not its {Wd.SALT} coin 0"),
                       ({"coin": "00" * 64}, "coin does not open against the epoch's coin commitment")):
        assert why(salt_opening={**opening, **edit}) == f"the pass's salt coin: {said}"


def test_a_pearl_c_passs_session_signs_the_salt_coin_alone_and_logs_its_epochs_root_alone(tmp_path, monkeypatch):
    """A session whose signed log shows its salt coin following another root than the pass's epoch's is refused, the
    pass's own coins and salt agreeing with one another. So is a session whose challenger signed a round coin beside
    the salt coin: the gate's own end entry says it released more than the one coin, and under an end entry that
    claimed the salt coin alone the signed coins still say otherwise."""
    for name in ("another-root", "a-round-beside"):
        (tmp_path / name).mkdir()
    served_pass, gate, session, root = _salt_pass(tmp_path / "another-root", monkeypatch, PCS.salt_only(6),
                                                  after=bytes(range(64)))
    assert PHC.salt_session_reason(served_pass, gate.ledger(), {PCS.SID: session.tree}.get) == (
        f"session {PCS.SID}'s signed log does not show the epoch root alone as the root its salt coin followed")
    served_pass, gate, session, root = _salt_pass(tmp_path / "a-round-beside", monkeypatch,
                                                  Wd.live_challenger(0, 1, CH.entropy(7)))
    fetch, released = {PCS.SID: session.tree}.get, {Wd.ROUND: 0, Wd.SALT: 1}
    assert PHC.salt_session_reason(served_pass, gate.ledger(), fetch) == (
        f"session {PCS.SID} released {released}, not its one salt coin: a refused pass")
    claimed = PCS._ended_over(session.directory, Wd.SALT_FAMILIES, root.hex(), released={Wd.SALT: 1})
    assert PHC.salt_session_reason(served_pass, claimed, fetch) == (
        f"session {PCS.SID}'s signed coins: the coin commitment's coins are not a Pearl-C salt session's: its salt "
        "coin alone")


_NO_TOKEN_ID = f"an input id is not a u32 below {W.EMPTY_ID}"
_PAST_THE_SLOTS = f"{9 * sum(map(len, TC.PROMPTS.values()))} input ids, past the Program's {TC.ROWS} slots"


@pytest.mark.parametrize("ids,said", [(lambda ids: [-1, *ids[1:]], _NO_TOKEN_ID),
                                      (lambda ids: [W.EMPTY_ID, *ids[1:]], _NO_TOKEN_ID),
                                      (lambda ids: ids * 9, _PAST_THE_SLOTS)],
                         ids=["negative", "the-empty-slots-word", "past-the-slots"])
def test_step_0s_input_ids_in_the_commit_that_are_no_rows_of_token_ids_are_its_taps_refusal(tmp_path, monkeypatch, ids,
                                                                                            said):
    """Round 0's input-ids table is the window's own, its prompts' tokens, registered before step 0's forward. Step 0's
    ids as the Commit acquired them that are no token ids, or more than the Program has slots, make no words to hold
    it to: the tap refuses them by name, no table is the Commit's, and the window credits nothing."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (TC.Linear,))
    _record, run, tap = TC._window(tmp_path, monkeypatch, ids=lambda s, honest: ids(honest) if s == 0 else honest)
    assert tap.refused == [f"step 0: {said}"] and run.served.refused == []
    assert run.details["roots1"]["reason"] == f"roots1: the Commit's tap refused: step 0: {said}"
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}


def test_a_later_rounds_input_ids_that_are_no_token_ids_refuse_the_windows_served_trees(tmp_path, monkeypatch):
    """After round 0 a round's input-ids table is built from the Commit's ids of its step: ids that are no token ids
    make no table, the window's served trees are refused with the round's name, and the window credits nothing."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (TC.Linear,))
    _record, run, tap = TC._window(tmp_path, monkeypatch, ids=lambda s, honest: [-1, *honest[1:]] if s == 1 else honest)
    assert run.served.refused == [f"round 1's input ids: {_NO_TOKEN_ID}"]
    assert tap.refused == [] and sorted(run.served.ids) == [0]
    assert run.details["roots1"]["reason"] == "roots1: the window's served trees were refused, so no table is the Commit's"
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}


def test_a_named_record_that_is_no_json_object_or_has_no_canonical_json_is_not_the_windows(tmp_path, monkeypatch):
    """`reused` takes a named record that reuses the window's tables from its registration, and refuses by name one
    that is no JSON object (a list of one, its canonical JSON in hex, none) and one canonical JSON can't hold."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (TC.Linear,))
    _record, run, _tap = TC._window(tmp_path, monkeypatch)
    window = CT.registration(run, "12" * 64)
    record = TC._reusing(window, lambda rec: rec)
    assert CT.reused(run, window, record) is None
    for named in ([record], canonical_json(record).hex(), None):
        assert CT.reused(run, window, named) == "the named record is not a JSON object"
    for value in (b"bytes", {1, 2}):
        assert CT.reused(run, window, {**record, "query": value}) == (
            f"the named record has no canonical JSON: Object of type {type(value).__name__} is not JSON serializable")
