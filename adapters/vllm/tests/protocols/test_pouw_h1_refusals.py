"""H1's checks refuse, by name, the forms its other tests never hand them: a request's first slot and a step's slots
under tables a prover committed consistently (`pouw_h1_requests.request_reason`, `slot_reason`), a round's row and coin
(`round_reason`, `rounds_reason`), a record's calls (`tables_reason`), a window's coins and its sample
(`pouw_h1_challenger.trusted_rounds_reason` and its sample session's checks), what H1's settings, its input ids, a static
declaration and the gate's ``model-key`` answer are parsed from, a hand-off's socket path, its stream and its saved
window (`pouw_h1_handoff.Holder`, `send_stream`, `client_of`, `resume`), a Pearl-C pass's salt session
(`pouw_h1_challenger.salt_session_reason`), a period's weights and a value's held rows
(`pouw_h1_served.register_weights`, `pouw_h1_registered.commit`), an unclosed window's trees
(`pouw_h1_served_files.trees`), the Commit's input ids and a named Commit record
(`pouw_h1_commit.CommitTap.input_ids`, `reused`), and a window's served trees off its Program: its period's weights,
a call's weight, a linear its forward skips, its receipt producer and round 0's input ids (`pouw_h1_served.Served`).
Beside them, two paths only a large file takes: a file's digest kept for the process (`pouw_h1_answers.file_digest`)
and a file past zip's 32-bit sizes written by ``np.savez`` (`pouw_h1_prover.savez`).

And what a window's own parts refuse with no GPU: its sessions (`pouw_h1_window`: a lane's open period, a salt coin
that doesn't open, the gate's withheld open, a sample session's reserve, open and layout, a second open), a receipt
producer's launches, run, codes, weight and roots (`pouw_h1_receipts.Receipts`), the sampled statements' window,
weights, weights' key and sessions (`drawn.window_of`, `linear_weights`, `weights_key`, `statements`, `check`), a
multiprocess window's ranks (`pouw_h1_ranks.install`, `ingest`, `RankExtension`), a call that committed nothing
(`pouw_h1_served.Served.close`, `pouw_h1_kept.kept_row`), and the host's libraries and table roots
(`pouw_h1_registered_device.host_lib`, `zy_host_lib`, `pouw_h1_tables_device.host_roots`).

And what is left of those parts where no test went: a compiled call shape off its tiles or its one seed group
(`pouw_h1.check_tiles`, `H1Run.check`), a linear of a window that commits nothing (`pouw_h1_hooks.traced`), a saved
window's sampling coin (`pouw_h1_handoff.suspend`, `resume`), a call's rows of the Commit and a table a named record
reuses (`pouw_h1_commit.CommitTap.arow`, `reused`), a value's rows in a host tensor (`pouw_h1_registered.commit`), the
tables of a call that served no rows and a port that reads no row (`drawn.zy_value`, `a_value`, `call_rows`), the
host's salts (`pouw_h1_registered_device.host_commit`), and records that say they are on a CUDA device
(`pouw_h1_words.zy_words`, `pouw_h1_tables_device.take_z`).

And what the GPU path does that a CPU reaches: a window hashed where that path says by the host's twins of the
kernels (`pouw_h1_tables_device.flush`, `pouw_h1_registered.commit`, `public_salts`, `pouw_h1_served.Served.settle`),
kernels that don't build and a library off `ZyArgs` (`pouw_h1_registered_device.available`, `zy_host_lib`). Beside
it, a pull that holds two engine steps and a rank call with no rows or off the Program's order
(`pouw_h1_ranks.Holder.pull`, `ingest`, `pouw_h1_served.Served.commit`), a leaf's committed files, its staged table
and its record's root (`drawn.leaf_stagers`, `FileStager.stage`, `stage_leaves`), and a saved window's Program
(`pouw_h1_handoff.resume`).

And what a call's receipts are with no card: each lane's words, J10-R's from the call's one stream, a dead tile's
under admit bits and the units kernel's seed record (`pouw_h1_receipts.call_receipts`, `call_stream`,
`gated_receipts`, `seed_record`), and a producer that skips dead tiles, keeps a call, or is handed a step for the
units kernel (`Receipts`, `prepare`); with stand-ins for the two kernels, what `noise_blocks` and `call_streams`
read back from the noise kernel and what a producer hands the units kernel. Beside them, the one reserve after a
period's close (`pouw_h1_window.after_close`), a host library that returns nonzero
(`pouw_h1_registered_device.host_commit`, `host_zy`, `host_public_salts`, `pouw_h1_tables_device.host_roots`) and
call tables whose kernels don't build (`pouw_h1_tables_device.available`,
`pouw_h1_registered_device.zy_available`)."""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import os
import pickle
import random
import shutil
import socket
import threading
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity.circuits.codec import canonical_json
from verity.protocols.accounting.pouw.verifier import credit as CR
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
from verity.protocols.proofs.spot.device import remote as R
from verity.protocols.proofs.spot.device.challenger import CoinError, LiveRecord

import verity.protocols.accounting.pouw.prover.serving.h1.answers as AN
import verity.protocols.accounting.pouw.prover.serving.h1.challenger as PHC
import verity.protocols.accounting.pouw.prover.serving.h1.commit as CT
import verity.protocols.accounting.pouw.prover.serving.h1.draw as PHD
import verity.protocols.accounting.pouw.prover.serving.h1.drawn as DN
import verity.protocols.accounting.pouw.prover.serving.h1.drawn_rows as DR
import verity.protocols.accounting.pouw.prover.serving.h1.handoff as HO
import verity_vllm.protocols.pouw.h1_hooks as PHK
import verity.protocols.accounting.pouw.prover.serving.h1.kept as KP
import verity.protocols.accounting.pouw.prover.serving.h1.live as LV
import verity.protocols.accounting.pouw.prover.serving.h1.model_key as MK
import verity.protocols.accounting.pouw.prover.serving.h1.prover as PV
import verity_vllm.protocols.pouw.h1_ranks as PHR
import verity.protocols.accounting.pouw.prover.serving.h1.receipts as RC
import verity.protocols.accounting.pouw.prover.serving.h1.registered as NR
import verity.protocols.accounting.pouw.prover.serving.h1.registered_device as RGD
import verity.protocols.accounting.pouw.prover.serving.h1.requests as RQ
import verity.protocols.accounting.pouw.prover.serving.h1.served as SV
import verity.protocols.accounting.pouw.prover.serving.h1.served_files as SF
import verity.protocols.accounting.pouw.prover.serving.h1.static as ST
import verity.protocols.accounting.pouw.prover.serving.h1.tables_device as TBD
import verity.protocols.accounting.pouw.prover.serving.h1.window as PHW
import verity.protocols.accounting.pouw.prover.serving.h1.words as W
from tests.protocols import challenger_host as CH
from tests.protocols import test_pouw_h1_commit as TC
from tests.protocols import test_pouw_h1_handoff as TH
from tests.protocols import test_pouw_h1_ranks as TR
from tests.protocols import test_pouw_h1_served as TS
from tests.protocols import test_pouw_pearl_c_served as PCS
from tests.protocols.test_pouw_h1 import (
    AUDITOR,
    KEY,
    RESERVED,
    SALT,
    K,
    Linear,
    Runner,
    _admitted_run,
    _entry,
    _gate,
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
from verity.protocols.accounting.pouw.prover.serving import h1 as PH

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


def test_a_window_with_the_commit_in_its_engine_does_not_hand_off(tmp_path, monkeypatch):
    """`client_of`, and so `suspend`, refuses an open, unclosed window on the auditor host's session whose served trees
    hold the Commit's tap, asked here inside its first forward: the tap holds the engine's committer, which no other
    process resumes, so the window closes where it served, nothing is saved, and it closes as it would have."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (TC.Linear,))
    said: list[str] = []

    def asked(observe):
        def once(*args, **kw):
            if not said:
                run = pouw.h1_run(TC.ENTRY)
                for hand_off in (lambda: HO.client_of(run), lambda: HO.suspend(run, tmp_path / "saved")):
                    try:
                        said.append(f"handed off: {hand_off()}")
                    except ValueError as e:
                        said.append(str(e))
            return observe(*args, **kw)
        return once

    _record, run, _tap = TC._window(tmp_path, monkeypatch, lie=asked)
    refusal = (f"{TC.ENTRY}: a window with the Commit in its engine (pouw_h1_commit) closes where it served: its tap "
               "holds the committer")
    assert said == [refusal, refusal]
    assert not (tmp_path / "saved").exists() and run.details["roots1"]["tied"]


def test_an_unclosed_windows_trees_are_not_read_as_its_registered_values():
    """A window's served trees are all committed at its close: before it, `pouw_h1_served_files` reads none of them."""
    run = _driven(served=SV.Served(window=bytes(64)))
    assert not run.closed
    for read in (lambda: SF.trees(run), lambda: SF.reads_file(run, 0)):
        with pytest.raises(ValueError, match="^t: the window isn't closed, so its tables aren't all committed$"):
            read()


def test_served_trees_with_no_program_yet_say_so_where_one_is_read():
    """A window's Program is declared at its first call: its tables and a call's receipt shape, read before then,
    raise, naming what the window doesn't hold."""
    served = SV.Served(window=bytes(64))
    for read in (served.tables, lambda: served.receipt_shape(0)):
        with pytest.raises(RuntimeError, match="^the window holds no Program where one is read$"):
            read()


def test_a_rounds_slot_table_is_not_registered_live_before_the_windows_open_root():
    """A round's live slot table is bound to the window's open root: with the tap and the Program declared and the
    window not yet open there is none, and nothing is refused."""
    served = SV.Served(PH.TILE, TS._unit(), tap=object())
    served.declare([(32, K), (32, K)], {0: "q", 1: "o"}, 32, 2)
    assert served.program is not None and served.root is None
    run = SimpleNamespace(session=None, owners={0: [0]}, rows=32)
    assert served.live_table(run, LV.SLOTS, 0) is None and served.refused == [] and served.live == {}


def test_a_window_whose_periods_weights_are_registered_under_another_program_is_refused():
    """The period's weights are registered under its declared Program's digest. A window that declares another
    Program, here the same linears over J10-R proof blocks, has no ``b/<K>`` its statements read: its served trees are
    refused at its first call, nothing is registered, and `trees` says why."""
    period = TS._period(TS._weights(), defn=TS.StandInJ10R)
    run, _ws, _got = TS._window(TS.W1, weights=period)
    refusal = ("the period's registered weights are under another Program than the window's: register each K's b/<K> "
               "under the period's declared Program")
    assert run.served.refused == [refusal]
    assert run.served.program_digest != period.circuit and run.served.registered == {}
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


@pytest.mark.parametrize("off", ["int16-codes", "five-scales", "no-scales"])
def test_a_served_weight_that_is_not_its_linears_int8_codes_and_scales_refuses_the_windows_trees(off):
    """A served call's weight is its linear's n × K int8 codes and n scales, the rows ``b/<K>`` holds. Codes of another
    type, scales of another count and no scales make no such rows: the window's served trees are refused at that call,
    by its index, and nothing is registered."""
    (n, k), (codes, scales) = TS.SHAPES[0], TS._weights()[0]
    weight, scale = {"int16-codes": (codes.to(torch.int16), scales), "five-scales": (codes, scales[:5]),
                     "no-scales": (codes, None)}[off]
    run = PH.H1Run("t", TS.SALT, TS.MODULES, TS.SHAPES, TS.T, TS.STEPS, key=KEY,
                   served=SV.Served(TS.TILE, TS._unit(), TS.BLOCKS, window=TS.W1))
    call = run.record(0, TS._codes(np.random.default_rng(3), TS.SERVED, k), torch.ones(TS.SERVED), weight, scale)
    assert call is not None and call.index == 0
    assert run.served.refused == [f"call 0: its weight is not its {n} × {k} int8 codes and {n} scales"]
    assert run.served.registered == {}


def test_a_window_whose_forward_skips_a_linear_is_not_static_and_has_no_trees():
    """The declared Program has every (step, linear) call of the window. A window whose forward never runs one of its
    linears closes with fewer: it isn't static, its served trees are refused at its close, and `trees` says why."""
    run, _ws, _got = TS._window(TS.W1, order=[0, 1])
    assert not run.static() and run.order_ok and run.refused == []
    refusal = ("the window isn't static: a call off its public position or refused, so the Program's calls aren't the "
               "window's")
    assert run.served.refused == [refusal]
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


def _no_receipt(call, codes, s_a, w, s_w):
    raise ValueError(f"no receipt of a call of {len(codes)} rows")


@pytest.mark.parametrize("producer,said", [(lambda: _no_receipt, f"no receipt of a call of {TS.SERVED} rows"),
                                           (lambda: TS._Receipts(dtype=np.uint8),
                                            "a call's receipts are (units, words) u32 or u16 words, not uint8 (4, 1)")],
                         ids=["refused", "u8-words"])
def test_a_receipt_producer_that_refuses_a_call_or_hands_back_no_receipt_words_refuses_the_windows_trees(producer, said):
    """A call's receipt table is its producer's receipts of the call's own served codes. A producer that refuses the
    call, or whose receipts are no u16 or u32 words, makes no table: the window's served trees are refused at that
    call, by its index, with the producer's reason, no receipt table is registered, and `trees` says why."""
    run, _ws, _got = TS._window(TS.W1, receipts=producer())
    assert run.served.refused == [f"call 0: {said}"]
    assert not any("receipt@" in value for value in run.served.registered)
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


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


#: what `reused` says of a window with no served trees
_NO_TABLES = "the window's served trees were refused, so it has no tables a Commit record can reuse"


def test_a_later_rounds_input_ids_that_are_no_token_ids_refuse_the_windows_served_trees(tmp_path, monkeypatch):
    """After round 0 a round's input-ids table is built from the Commit's ids of its step: ids that are no token ids
    make no table, the window's served trees are refused with the round's name, no Commit record reuses a table of
    it, and the window credits nothing."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (TC.Linear,))
    _record, run, tap = TC._window(tmp_path, monkeypatch, ids=lambda s, honest: [-1, *honest[1:]] if s == 1 else honest)
    assert run.served.refused == [f"round 1's input ids: {_NO_TOKEN_ID}"]
    assert tap.refused == [] and sorted(run.served.ids) == [0]
    assert run.details["roots1"]["reason"] == "roots1: the window's served trees were refused, so no table is the Commit's"
    assert CT.reused(run, None, None) == _NO_TABLES
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}


def test_a_prompt_token_that_is_no_token_id_refuses_the_windows_served_trees_before_it_opens(tmp_path, monkeypatch):
    """Round 0's input-ids table is the window's own, its prompts' tokens, registered before the window opens. A
    prompt token that is no token id (the empty slot's word) makes no table: the served trees are refused with the
    round's name before any is registered, no table is the Commit's or a Commit record's to reuse, and the window
    credits nothing."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (TC.Linear,))
    monkeypatch.setitem(TC.PROMPTS, "a", [W.EMPTY_ID, 6, 7])
    _record, run, tap = TC._window(tmp_path, monkeypatch)
    assert run.served.refused == [f"round 0's input ids: {_NO_TOKEN_ID}"]
    assert tap.refused == [] and run.served.ids == {} and run.served.registered == {}
    assert run.details["roots1"]["reason"] == "roots1: the window's served trees were refused, so no table is the Commit's"
    assert CT.reused(run, None, None) == _NO_TABLES
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


def test_a_file_of_the_cached_size_is_hashed_once_while_its_stat_is_unchanged(tmp_path, monkeypatch):
    """`file_digest` hashes a file below `CACHED_BYTES` each time it is asked, and one of at least that size once,
    however its path is given, while its size and times are unchanged: rewritten longer, it is hashed again."""
    hashed: list[Any] = []

    def counted(path):
        hashed.append(path)
        return hashlib.sha512(path.read_bytes()).hexdigest()

    monkeypatch.setattr(AN, "PF", SimpleNamespace(file_sha512=counted))
    monkeypatch.setattr(AN, "CACHED_BYTES", 4)
    monkeypatch.setattr(AN, "_DIGESTS", {})
    small, large = tmp_path / "small", tmp_path / "large"
    small.write_bytes(b"abc")
    large.write_bytes(b"abcd")
    assert AN.file_digest(small) == AN.file_digest(small) == hashlib.sha512(b"abc").hexdigest() and len(hashed) == 2
    assert AN.file_digest(large) == AN.file_digest(str(large)) == hashlib.sha512(b"abcd").hexdigest()
    assert hashed[2:] == [large]
    large.write_bytes(b"abcde")
    assert AN.file_digest(large) == hashlib.sha512(b"abcde").hexdigest() and hashed[2:] == [large, large]


def test_a_file_past_zips_32_bit_sizes_is_the_one_np_savez_writes(tmp_path, monkeypatch):
    """`savez` writes the bytes ``np.savez`` writes: itself below zip's 32-bit sizes and counts, and through
    ``np.savez`` past them (here with the limit lowered to nothing), the same file either way, 0600."""
    arrays: dict[str, np.ndarray] = {"a": np.arange(6, dtype="<u2").reshape(2, 3), "s_a": np.zeros(0, "<f4")}
    want = io.BytesIO()
    np.savez(want, a=arrays["a"], s_a=arrays["s_a"])
    own = PV.savez(tmp_path / "own.npz", arrays)
    monkeypatch.setattr(PV, "zipfile", SimpleNamespace(ZIP64_LIMIT=0))
    past = PV.savez(tmp_path / "past.npz", arrays)
    assert own.read_bytes() == past.read_bytes() == want.getvalue()
    assert past == tmp_path / "past.npz" and past.stat().st_mode & 0o777 == 0o600


def test_a_lanes_open_period_is_the_last_it_opened_that_the_ledger_has_not_closed():
    """`open_period` is the body of the lane's ``period-open`` while no ``period-close`` of that period follows it:
    None once the period is closed, and the next one's body once the lane opens another. Another lane's open and
    another period's close change nothing."""
    def entry(kind, **body):
        return json.dumps({"kind": kind, "body": body})
    first, second = {"lane": "vllm", "period": "p1"}, {"lane": "vllm", "period": "p2"}
    lines = [entry("period-open", lane="other", period="q1"), entry("period-open", **first)]
    assert PHW.open_period([], "vllm") is None and PHW.open_period(lines, "vllm") == first
    assert PHW.open_period([*lines, entry("period-close", period="q1")], "vllm") == first
    closed = [*lines, entry("period-close", period="p1")]
    assert PHW.open_period(closed, "vllm") is None
    assert PHW.open_period([*closed, entry("period-open", **second)], "vllm") == second


class _Client:
    """A challenger's client as `pouw_h1_window` reads one: what its hello answered (its coin commitment, or withheld
    the layout's ``families`` alone), the coins it was asked, each of which it refuses, and how often it was closed."""

    def __init__(self, commitment=None, families=()):
        self.commitment, self.withheld, self.families = commitment, commitment is None, families
        self.asked: list[tuple[str, int, bytes]] = []
        self.closed = 0

    def open(self, label, index, after):
        self.asked.append((label, index, after))
        raise CoinError("the challenger refused: that coin is open")

    def close(self):
        self.closed += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _reached(monkeypatch, *clients):
    """`clients` in turn as what `remote.RemoteChallenger.connect` reaches; returns the (address, secret) of each try."""
    tries: list[tuple[str, bytes | None]] = []
    left = list(clients)

    def connect(cls, at, secret=None):
        tries.append((at, secret))
        if not left:
            raise OSError("no challenger is listening")
        return left.pop(0)
    monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(connect))
    return tries


def test_a_session_whose_salt_coin_does_not_open_is_closed_and_the_refusal_raised(monkeypatch):
    """`session` (ncp-v2's) and `opened` (a reserved session the gate opens) reach the challenger, then ask its salt
    coin after their root. Where that coin does not open, the session they reached is closed and the challenger's
    refusal is raised as it was."""
    live = Wd.live_challenger(0, 3, CH.entropy(1)).commitment
    sid, root = CH.session_id(1), hashlib.sha512(b"a root").digest()
    answer = {"session": sid, "seq": 0, "root": root.hex(), "commitment": live.root.hex()}
    clients = [_Client(live), _Client(live)]
    tries = _reached(monkeypatch, *clients)
    for client, reach in zip(clients, (
            lambda: PHW.session("pouw:test", AUDITOR, 3, root, answer),
            lambda: PHW.opened("pouw:test", AUDITOR, 3, lambda s, r: answer, sid, root, "the window's open root"))):
        with pytest.raises(CoinError) as refused:
            reach()
        assert str(refused.value) == "the challenger refused: that coin is open"
        assert client.asked == [(Wd.SALT, 0, root)] and client.closed == 1
    assert tries == [(AUDITOR, None), (AUDITOR, None)]


@pytest.mark.parametrize("answer", [{"withheld": False, "signed": True}, {"signed": True}, {"withheld": 1, "signed": True},
                                    {"withheld": True, "signed": True, "commitment": "c0" * 64}, {"withheld": True},
                                    {"withheld": True, "signed": 1}],
                         ids=["plain", "no-word-of-it", "withheld-1", "a-commitment", "unsigned", "signed-1"])
def test_the_gates_open_of_a_withheld_session_says_withheld_and_signed_and_shows_no_commitment(answer):
    """A sample session is reserved withheld: the gate's open of it says ``withheld: true`` and ``signed: true`` and
    carries no coin commitment, and ``status`` is never asked for one. An answer short of that is refused before any
    coin, by the session's id; one that has all three is the window's, less what a session's answer doesn't hold."""
    sid, root, asked = CH.session_id(0xd), "ab" * 64, []

    def gate(given):
        def call(*words):
            asked.append(words)
            return {"session": sid, "seq": 4, "root": root, "commitment": None, **given}
        return PHW.gate_of(call)
    with pytest.raises(CoinError) as refused:
        gate(answer)(sid, root, withhold=True)
    assert str(refused.value) == f"the gate's open of {sid}: {PHW.WITHHELD_OPEN}"
    assert gate({"withheld": True, "signed": True, "port": 7812, "ledger": "not a session's"})(sid, root, withhold=True) == {
        "session": sid, "seq": 4, "root": root, "commitment": None, "port": 7812}
    assert asked == [("open", "--session", sid, "--root", root)] * 2


def test_a_sample_session_the_gate_reserved_plain_is_refused_and_a_withheld_ones_answer_is_the_windows():
    """`reserve_draw` asks the gate for a session of the sampling coin alone, withheld. An answer that doesn't say
    ``withheld: true`` is refused by the session's id; a withheld one is the reserve answer the window takes, its
    session and the reserve's fields."""
    sid, asked = CH.session_id(0xd), []

    def gate(answer):
        def call(*words):
            asked.append(words)
            return answer
        return call
    for answer in ({"session": sid, "seq": 3}, {"session": sid, "seq": 3, "withheld": False},
                   {"session": sid, "seq": 3, "withheld": 1}):
        with pytest.raises(CoinError) as refused:
            PHW.reserve_draw(gate(answer), "vy-nebius-1")
        assert str(refused.value) == f"the gate reserved draw session {sid} plain, not withheld"
    reserved = {"session": sid, "seq": 3, "peer": "vy-nebius-1", "withheld": True, "scheme": PHC.SCHEME}
    assert PHW.reserve_draw(gate(reserved), "vy-nebius-1") == {"session": sid, "seq": 3, "peer": "vy-nebius-1"}
    assert asked == [("reserve", "--peer", "vy-nebius-1", "--scheme", PHC.SCHEME, "--families",
                      json.dumps([[Wd.DRAW, 1]]), "--withhold")] * 4


def test_a_windows_sampling_coin_has_no_session_unless_one_is_reserved_and_a_gate_opens_it(monkeypatch):
    """With no reserve answer for a sample session, or no gate to open one, the window's sampling coin is refused with
    `NO_DRAW_SESSION`, and nothing is asked of a gate or reached."""
    sid, root, asked = CH.session_id(0xd), hashlib.sha512(b"a sample root").digest(), []
    tries = _reached(monkeypatch)
    for reserved, gate in ((None, lambda *a, **kw: asked.append(a)), ({"session": sid}, None), (None, None)):
        with pytest.raises(CoinError) as refused:
            PHW.DrawSession("pouw:test", AUDITOR, reserved, gate).record(root)
        assert str(refused.value) == PHW.NO_DRAW_SESSION
    assert asked == [] and tries == []


@pytest.mark.parametrize("off", ["session", "root"])
def test_a_sample_session_is_opened_once_on_the_windows_sample_root_and_a_retry_never_asks_again(monkeypatch, off):
    """The gate opens the reserved sample session on the window's sample root, withheld. An answer for another
    session, or for another root, is refused before anything is reached, by the session and the root; and the
    session is asked for once: a second `record` is refused without a word to the gate."""
    sid, root, asked = CH.session_id(0xd), hashlib.sha512(b"a sample root").digest(), []
    tries = _reached(monkeypatch)

    def gate(session, r, withhold=False):
        asked.append((session, r, withhold))
        return {"session": session, "root": r, off: {"session": CH.session_id(0xe), "root": "ab" * 64}[off]}
    sample = PHW.DrawSession("pouw:test", AUDITOR, {"session": sid}, gate)
    with pytest.raises(CoinError) as refused:
        sample.record(root)
    assert str(refused.value) == (f"pouw:test: the window's draw session {sid}: the gate's answer is not draw session "
                                  f"{sid} opened on the draw root {root.hex()[:16]}...")
    with pytest.raises(CoinError, match="^the window's draw session is opened once: a retry never "):
        sample.record(root)
    assert asked == [(sid, root.hex(), True)] and tries == []
    assert sample.session is None and sample.commitment is None


SHOWN = ("draw session {sid}'s challenger answered its hello with a coin commitment: a draw session is withheld, its "
         "commitment released only at its close")
ANOTHER_LAYOUT = "draw session {sid}'s layout is not a draw session's, its draw coin alone"


@pytest.mark.parametrize("client,said", [
    (lambda: _Client(Wd.draw_challenger(0, CH.entropy(1)).commitment, ((Wd.DRAW, 1),)), SHOWN),
    (lambda: _Client(families=((Wd.DRAW, 2),)), ANOTHER_LAYOUT),
    (lambda: _Client(families=((Wd.DRAW, 1), (Wd.SALT, 1))), ANOTHER_LAYOUT)],
    ids=["a-commitment", "two-coins", "a-salt-beside"])
def test_a_sample_sessions_challenger_that_shows_its_commitment_or_another_layout_is_asked_no_coin(monkeypatch, client, said):
    """A sample session's challenger answers its hello with the session's layout alone, its sampling coin and no
    other. One that answers with a coin commitment, or whose layout holds two sampling coins or a salt coin beside
    the one, is closed with no coin asked of it, and the window has no sample session."""
    sid, root, reached = CH.session_id(0xd), hashlib.sha512(b"a sample root").digest(), client()
    tries = _reached(monkeypatch, reached)
    sample = PHW.DrawSession("pouw:test", AUDITOR, {"session": sid},
                             lambda s, r, withhold=False: {"session": s, "root": r, "port": 7812, "secret": "5e" * 32})
    with pytest.raises(CoinError) as refused:
        sample.record(root)
    assert str(refused.value) == said.format(sid=sid)
    assert tries == [(f"{AUDITOR.rpartition(':')[0]}:7812", bytes.fromhex("5e" * 32))]
    assert reached.asked == [] and reached.closed == 1 and sample.session is None and sample.commitment is None


def test_a_windows_session_is_opened_once(tmp_path, monkeypatch):
    """A window opens its reserved session on its open root once round 0's request table is committed, and once: a
    second `open` is refused by the window's entry, and the session it holds closes as the first one's."""
    host = _host(tmp_path, monkeypatch)
    weights, requests = (hashlib.sha512(x).digest() for x in (b"weights", b"round 0's requests"))
    window = PHW.Window("pouw:test", AUDITOR, 3, weights, RESERVED, _gate)
    with pytest.raises(CoinError, match="^the window's session was never opened$"):
        window.close()
    salt, rounds, _sample, commits = window.open(requests)
    try:
        with pytest.raises(RuntimeError) as refused:
            window.open(requests)
    finally:
        window.close()
    assert str(refused.value) == "pouw:test: the window's session is opened once"
    assert host.ended() is None
    opened_on = PS.window_open_root(weights, requests, RESERVED["session"])
    assert commits["session"]["root"] == opened_on.hex() and rounds.challenger is host.client
    assert salt == salt_of(host.challenger.commitment, LC.LiveOpening.from_json(commits["salt_opening"]).coin, weights)


def test_a_session_is_closed_when_what_follows_its_open_raises_and_left_open_when_it_does_not():
    """`closed_on_error` closes the session it is given when its body raises, and raises what the body did; a body
    that ends leaves the session open, and with no session there is nothing to close."""
    client = _Client()
    with PHW.closed_on_error(client):
        pass
    assert client.closed == 0
    with pytest.raises(KeyError, match="^'after the open'$"), PHW.closed_on_error(client):
        raise KeyError("after the open")
    assert client.closed == 1
    with pytest.raises(KeyError, match="^'no session'$"), PHW.closed_on_error(None):
        raise KeyError("no session")


def test_a_receipt_producer_takes_a_launch_a_run_and_each_calls_own_codes_and_weight():
    """The producer refuses, each by name: a batch of no launch; a call before a run is bound, whose salt its seeds
    take; a call with no weight handed over; a weight that is not the call's n × K contiguous int8 codes on its
    device; and codes that are not at most the window's M rows of K int8 codes. What it takes becomes the call's A
    over the M slots, zero past the served rows, beside its B."""
    served, call = SimpleNamespace(m=32), SimpleNamespace(index=4, layer=1, n=32, k=K)
    with pytest.raises(ValueError, match="^a batch takes at least one launch, not 0$"):
        RC.Receipts(served, launches=0)
    made = RC.Receipts(served)
    codes: np.ndarray = np.arange(5 * K, dtype=np.int8).reshape(5, K)
    weight: np.ndarray = np.ones((32, K), np.int8)
    with pytest.raises(ValueError) as refused:
        made(call, codes, None, weight, None)
    assert str(refused.value) == "the receipt producer is bound to no run: its units' seeds take the run's salt (`bind`)"
    with pytest.raises(ValueError, match="^call 4: no weight of linear 1 was handed over$"):
        made.weight_of(call, None)
    for held, what in ((lambda: torch.ones(16, K, dtype=torch.int8), f"torch.int8 (16, {K})"),
                       (lambda: torch.ones(32, K, dtype=torch.int16), f"torch.int16 (32, {K})"),
                       (lambda: torch.ones(K, 32, dtype=torch.int8).t(), f"torch.int8 (32, {K})")):
        with pytest.raises(ValueError) as refused:
            RC.Receipts(served, weight=lambda _call, held=held: held()).weight_of(call, None)
        assert str(refused.value) == (f"call 4: its weight is {what} on cpu, not its 32 × {K} contiguous int8 codes on "
                                      "cpu")
    others: list[np.ndarray] = [codes.astype(np.int16), codes[0], codes[:, :K // 2], np.zeros((33, K), np.int8)]
    for given in others:
        with pytest.raises(ValueError) as refused:
            made.operands(call, given, weight)
        assert str(refused.value) == (f"call 4: its codes are {given.dtype} {given.shape}, not at most 32 rows of {K} "
                                      "int8 codes")
    a, b = made.operands(call, codes, weight)
    assert a.shape == (32, K) and np.array_equal(a[:5].numpy(), codes) and not a[5:].any()
    assert np.array_equal(b.numpy(), weight) and made.weight_of(call, weight) is b


def test_a_calls_seeds_take_its_operands_registered_roots_or_the_roots_the_producer_is_given():
    """A call's seeds take the registered roots of its step's ``a/<K>@<s>`` and of ``b/<K>``. Asked before the window
    registers them, the producer refuses and names both tables and the ones missing; a producer made with ``roots``
    answers with those, whatever the window registers."""
    served = SV.Served(PH.TILE, TS._unit())
    served.declare([(32, K), (32, K)], {0: "q", 1: "o"}, 32, 2)
    call = SimpleNamespace(index=0, k=K)
    a, b = served.a_table(0), f"{served.program.fn.bindings['PREFIX']}b/{K}"
    with pytest.raises(ValueError) as refused:
        RC.Receipts(served).roots(call)
    assert str(refused.value) == (f"call 0's seeds take the roots of {a} and {b}, and {[a, b]} isn't registered yet: a "
                                  "step's receipts of K are asked for at its last call of K")
    given = (bytes(64), bytes([1]) * 64)
    assert RC.Receipts(served, roots=lambda c: given if c is call else None).roots(call) is given


def test_a_ks_weights_are_read_from_the_checkpoint_only_by_a_reader_the_caller_binds():
    """With no ``--weights`` file, K's weights come from ``--checkpoint`` through the reader the caller binds.
    `linear_weights` refuses without a checkpoint, and with one and no reader, and says what to give; with both it
    hands the reader the checkpoint, the window's linears and K, and returns what it read."""
    linears, checkpoint, asked = [("q", 4, 3), ("o", 6, 2), ("k", 4, 2)], Path("model.safetensors"), []

    def read(*args):
        asked.append(args)
        return "the weights"
    for reader in (None, read):
        with pytest.raises(Refused) as refused:
            DN.linear_weights(linears, 4, None, None, reader)
        assert str(refused.value) == ("b/<K>'s rows are the served model's weights: give --checkpoint (its "
                                      "safetensors) or --weights")
    with pytest.raises(Refused) as refused:
        DN.linear_weights(linears, 4, checkpoint, None, None)
    assert str(refused.value) == ("--checkpoint is read by benchmarks/pouw/h1_prove/drawn.py's reader: run that CLI, or "
                                  "give --weights")
    assert asked == [] and DN.linear_weights(linears, 4, checkpoint, None, read) == "the weights"
    assert asked == [(checkpoint, linears, 4)]


def test_the_key_a_ks_weights_are_salted_under_is_the_gates_model_key_for_the_windows_circuit(monkeypatch):
    """With ``--gate`` and no ``--weights-key``, `weights_key` asks the gate's ``model-key`` for ``--model`` under the
    window's circuit and weights' root, as one word of the gate's command, and takes the key its answer holds in hex
    when the answer's ``key_sha512`` names it. It refuses, and says which: a gate with no ``--model``, before the gate
    is asked; an answer that holds no key in hex, with the gate's exit and the end of what it wrote; and a key its
    ``key_sha512`` doesn't name."""
    key, circuit, weights = bytes(range(32)), "ab" * 64, "cd" * 64
    win = {"program": circuit, "record": {"commitment": {"weights_root": weights}}}
    asked: list[tuple[list[str], dict[str, Any]]] = []

    def gate(stdout=b"", stderr=b"", code=0, model="M"):
        def run(argv, **kw):
            asked.append((argv, kw))
            return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)
        monkeypatch.setattr(DN.subprocess, "run", run)
        return SimpleNamespace(weights_key=None, gate="ssh auditor gate", model=model)

    def refusal(*answer, **kw):
        with pytest.raises(Refused) as refused:
            DN.weights_key(gate(*answer, **kw), win)
        return str(refused.value)
    assert refusal(model=None) == "--gate reads the auditor's key with model-key --model ROLE: give --model"
    assert asked == []
    answer = {"key": key.hex(), "key_sha512": hashlib.sha512(key).hexdigest()}
    assert DN.weights_key(gate(json.dumps(answer).encode()), win) == (key, "the gate's model-key for M")
    assert asked == [(["ssh", "auditor", "gate", f"model-key --model M --circuit {circuit} --weights-root {weights}"],
                      {"capture_output": True, "check": False})]
    for stdout in (b'{"key_sha512": "00"}', b'{"key": "zz"}', b'{"key": 7}', b"[1]", b"no JSON"):
        assert refusal(stdout) == f"the gate's model-key answered no key (exit 0): {stdout.decode()}"
    assert refusal(b"", b"x" * 300 + b"denied\n", 3) == f"the gate's model-key answered no key (exit 3): {'x' * 293}denied\n"
    for named in ({"key_sha512": "00" * 64}, {"key_sha512": None}):
        assert refusal(json.dumps({**answer, **named}).encode()) == (
            "the gate's model-key key is not the one its key_sha512 names")


def test_a_windows_record_is_read_under_its_declared_programs_layout_and_only_with_its_served_trees(tmp_path):
    """`window_of` reads a window's record beside the declared Program. It refuses a Program file whose bytes have no
    layout, with core's reason; a record that names no served trees, or holds what is no list of their entries; and
    one whose served trees were refused, with their refusals."""
    served = SV.Served(PH.TILE, TS._unit())
    served.declare([(32, K), (32, K)], {0: "q", 1: "o"}, 32, 2)
    declared = SV.descriptor(served.program)
    (tmp_path / DN.RECORD.format(w=0)).parent.mkdir(parents=True)

    def refusal(commitment, raw=declared):
        (tmp_path / DN.PROGRAM).write_bytes(gzip.compress(raw))
        (tmp_path / DN.RECORD.format(w=0)).write_text(json.dumps({"commitment": commitment}))
        with pytest.raises(Refused) as refused:
            DN.window_of(tmp_path, 0)
        return str(refused.value)
    _layout, why = CR.layout_of(b"no descriptor")
    assert why is not None and CR.layout_of(declared)[1] is None
    assert refusal({}, b"no descriptor") == f"the declared Program's layout: {why[1]}"
    without: list[dict[str, Any]] = [{}, {"served": None}, {"served": [{"value": "v"}]}, {"served": {"refused": []}},
                                     {"served": {"entries": {"value": "v"}}}]
    for trees in without:
        assert refusal(trees) == "the window's record names no served trees: drawn rebuilds the values they registered"
    refused = ["call 0: its weight is not its 32 × 64 int8 codes and 32 scales"]
    assert refusal({"served": {"entries": [], "refused": refused}}) == f"the window's served trees were refused: {refused}"


def _ranks(*ranks, cls=TR.FakeRank):
    """An executor over one worker for each (pp, tp) of `ranks`, two linears each, and the workers."""
    layers = TR._layers()
    workers = [cls(tp, layers[2 * i:2 * i + 2]) for i, (_pp, tp) in enumerate(ranks)]
    for worker, rank in zip(workers, ranks):
        worker.rank = rank
    return TR.Executor(workers), workers


def _install(executor, **kw):
    return PHR.install(None, executor=executor, rows=32, steps=4, key=KEY, entry="t", **kw)


def test_a_multiprocess_window_is_refused_before_a_rank_is_installed_when_its_salt_or_its_ranks_dont_fit():
    """`install` refuses, by the window's entry and before any rank holds a window: a salt from both coins and an
    opener, or from neither; ranks whose runners disagree on decode_query_len; and the Commit over more than one
    pipeline stage, whose later stages acquire no input ids."""
    coins = Challenger(0, random.Random(0).randbytes)
    executor, workers = _ranks((0, 0), (0, 1))
    for salt in ({}, {"coins": coins, "opener": lambda requests: None}):
        with pytest.raises(ValueError, match="^t: the window's salt is from coins or opened by its opener, one of the two$"):
            _install(executor, **salt)
    workers[1].model_runner.decode_query_len = 2
    with pytest.raises(ValueError, match="^t: the ranks' runners disagree on decode_query_len$"):
        _install(executor, coins=coins)
    executor, staged = _ranks((0, 0), (1, 0))
    with pytest.raises(ValueError, match="^t: the Commit across pipeline stages: a later stage acquires no input ids$"):
        _install(executor, coins=coins, served=SV.Served(PH.TILE, TS._unit()), commit={})
    assert all(worker._h1rank is None for worker in (*workers, *staged))


class _ShortOfARoot(TR.FakeRank):
    """A rank whose install answers without its last linear's weight root."""

    def h1rank_install(self, **kw):
        out = super().h1rank_install(**kw)
        out["roots"].popitem()
        return out


class _Committing(TR.FakeRank):
    """A rank whose Commit acquires input ids under a port named by its tensor-parallel rank."""

    def h1rank_commit(self, **kw):
        return {"port": f"ids@tp{self.rank[1]}"}


def test_a_multiprocess_window_is_refused_when_its_ranks_roots_or_their_commits_ports_dont_make_one_window():
    """Every linear of the window has its weight root from the rank that holds it, and with the Commit every rank
    acquires input ids under one port. `install` refuses ranks whose roots cover fewer linears than their shapes
    hold, and ranks whose Commits name two ports, each by the window's entry."""
    coins = Challenger(0, random.Random(0).randbytes)
    for cls, kw, said in ((_ShortOfARoot, {}, "t: the ranks' weight roots cover 2 of 4 linears"),
                          (_Committing, {"served": SV.Served(PH.TILE, TS._unit()), "commit": {}},
                           "t: the ranks' Commits acquire input ids under ['ids@tp0', 'ids@tp1'], not one port")):
        executor, workers = _ranks((0, 0), (0, 1), cls=cls)
        try:
            with pytest.raises(RuntimeError) as refused:
                _install(executor, coins=coins, **kw)
        finally:
            executor.collective_rpc("h1rank_release")
        assert str(refused.value) == said and all(worker._h1rank is None for worker in workers)


def test_a_rank_with_no_window_refuses_its_salt_and_its_rows_and_releases_nothing():
    """Before `h1rank_install` and after `h1rank_release` a rank holds no window: it refuses the holder's salt and a
    pull of its rows, and a release there returns None."""
    rank = PHR.RankExtension()
    for ask in (lambda: rank.h1rank_open(SALT), rank.h1rank_rows):
        with pytest.raises(RuntimeError) as refused:
            ask()
        assert str(refused.value) == ("this rank holds no H1 window: h1rank_install has not run on it, or "
                                      "h1rank_release has")
    assert rank.h1rank_release() is None


def test_a_rank_call_that_does_not_follow_the_windows_last_call_is_refused():
    """The holder takes its ranks' calls in index order. One at or before the last call the window holds, here a
    pulled call handed over again, is refused by both indices, and the window keeps its calls."""
    holder, _executor = TR._staged()
    run = holder.run
    calls = list(run.calls)
    again = {"call": dataclasses.asdict(calls[3])}
    with pytest.raises(RuntimeError) as refused:
        PHR.ingest(run, [again])
    assert str(refused.value) == f"t: rank call 3 after call {calls[-1].index}"
    assert run.calls == calls and calls[-1].index == len(calls) - 1


def test_a_calls_statement_is_listed_with_the_one_session_that_proved_it_and_check_needs_credits_verification(tmp_path):
    """`statements` lists each planned call's statement with the one session that proved it: a call whose directory
    holds no session, or two, is refused by the call and the count, and with one the statement names it. `check` is
    refused where the caller binds no verification of credit's."""
    work, proved, out = tmp_path / "work", tmp_path / "work" / "c0", tmp_path / "statements.json"
    work.mkdir()
    (work / DN.PLAN).write_text(json.dumps({"calls": [{"call": 0, "circuit": "k64.circuit", "dir": str(proved)}]}))
    a = SimpleNamespace(work=str(work), out=str(out))

    def session(name):
        (proved / "sessions" / name).mkdir(parents=True)
        (proved / "sessions" / name / "session.json").write_text("{}")
        return str(proved / "sessions" / name)

    def refusal():
        with pytest.raises(Refused) as refused:
            DN.statements(a)
        return str(refused.value)
    once = "not one: a drawn statement is proved once, every instance drawn (session's subset:N)"
    assert refusal() == f"call 0: 0 sessions in {proved}/sessions, {once}"
    first = session("s1")
    assert DN.statements(a) == 0
    assert json.loads(out.read_text()) == [{"call": 0, "circuit": "k64.circuit", "public": str(proved / "pub.bin"),
                                            "sessions": [first]}]
    session("s2")
    assert refusal() == f"call 0: 2 sessions in {proved}/sessions, {once}"
    with pytest.raises(Refused) as refused:
        DN.check(SimpleNamespace(verified=None))
    assert str(refused.value) == ("check is credit's own verification, which benchmarks/pouw/h1_prove/drawn.py binds: "
                                  "run that CLI")


def test_a_window_that_committed_no_call_is_refused_at_its_close():
    """A window whose Program was never declared committed no call: its close refuses it by that, once, and a window
    already refused gains nothing at its close."""
    served = SV.Served(PH.TILE, TS._unit())
    served.close(None)
    assert served.refused == ["no call was committed"]
    served.close(None)
    assert served.refused == ["no call was committed"]


def test_a_kept_row_of_a_call_that_served_no_rows_is_zero_and_a_weight_row_is_its_ks_linears_own(tmp_path):
    """`kept_row` rebuilds a served value's row from what the prover kept. A call that served no rows kept nothing, so
    no file of it is read: its A rows are zero codes of scale 0 and its z table's rows those of no record. Row j of
    ``b/<K>`` counts K's linears' weight rows in turn, a linear of another K between them skipped."""
    rng = np.random.default_rng(5)
    weights = [(rng.integers(-128, 128, (32, k), dtype=np.int8), rng.random(32, dtype=np.float32) + 1) for k in (K, 2 * K, K)]
    served = {"prefix": "h1/", "m": 32, "linears": [(0, K, 32), (1, 2 * K, 32), (2, K, 32)], "calls": [(0,), (1,), (2,)],
              "tile": PH.TILE}
    state = {"served": ("served", served, None), "weights": weights}
    calls = [{"index": c, "served": 0, "rows": 0} for c in range(3)]
    empty: np.ndarray = np.zeros((1, K), np.int8)
    assert KP.kept_row(tmp_path, state, calls, f"h1/a/{K}@0", 32 + 5) == SV.row_words(empty, np.zeros(1, "<f4"))[0].tolist()
    no_z: np.ndarray = np.zeros((0, 32), "<i4")
    no_y: np.ndarray = np.zeros((0, 32), np.int16)
    assert KP.kept_row(tmp_path, state, calls, "h1/zy@2", 0) == SV.record_words(no_z, no_y, 32, PH.TILE)[0].tolist()
    codes, scales = weights[2]
    assert KP.kept_row(tmp_path, state, calls, f"h1/b/{K}", 32 + 1) == SV.row_words(codes[1:2], scales[1:2])[0].tolist()
    assert KP.kept_row(tmp_path, state, calls, f"h1/b/{2 * K}", 1) == SV.row_words(*(x[1:2] for x in weights[1]))[0].tolist()
    assert list(tmp_path.iterdir()) == []


def test_the_hosts_table_roots_take_n_rows_of_8_bit_codes_and_n_32_bit_words():
    """`host_roots` takes each table's codes as (n, K) of 8-bit values and its words as (n,) of 32-bit ones, and
    refuses any other pair by their shapes and types."""
    codes: np.ndarray = np.zeros((2, K), np.int8)
    words: np.ndarray = np.zeros(2, np.uint32)
    for c, w in ((codes[0], words[:1]), (codes.astype(np.int16), words), (codes, words.astype(np.uint16)),
                 (codes, np.zeros(3, np.uint32)), (codes, words.reshape(2, 1))):
        with pytest.raises(ValueError) as refused:
            TBD.host_roots([(bytes(64), 0, c, w)], KEY)
        assert str(refused.value) == (f"codes {c.shape} of {c.dtype} and words {w.shape} of {w.dtype} are not (n, K) "
                                      "8-bit and (n,) 32-bit")


def test_the_hosts_libraries_are_refused_where_no_cxx_compiler_is_on_the_path(monkeypatch):
    """The host's twins of the device's hashing are built with the system's C++ compiler, the first of g++, c++ and
    clang++ on the path. With none there, each library is refused by what is missing."""
    asked: list[str] = []
    monkeypatch.setattr(RGD.shutil, "which", lambda name: asked.append(name))
    for lib in (RGD.host_lib, RGD.zy_host_lib):
        with pytest.raises(RuntimeError, match=r"^no C\+\+ compiler on the path$"):
            lib.__wrapped__()
    assert asked == ["g++", "c++", "clang++"] * 2


def test_a_compiled_call_shape_is_whole_tiles_and_its_units_in_one_seed_group(window, monkeypatch):
    """`check_tiles` refuses a compiled Program with a proof block off the 32 x 32 grid, counting each shape that is
    off it, and `H1Run.check` one whose proof blocks are not its call shape's units, or not in one seed group."""
    def block(r, c, grid=True):
        return SimpleNamespace(tile=SimpleNamespace(r=r, c=c, grid=grid))
    served = SimpleNamespace(fn=SimpleNamespace(id="Linear_v1"))
    assert PH.check_tiles(SimpleNamespace(units=[block(*PH.TILE)] * 3, served=served)) is None
    off = [block(32, 16), block(*PH.TILE), block(32, 16), block(32, 32, grid=False)]
    with pytest.raises(PH.C.CompileError) as refused:
        PH.check_tiles(SimpleNamespace(units=off, served=served))
    assert str(refused.value) == "scheme: Linear_v1: units off the 32 x 32 grid: {(32, 16, True): 2, (32, 32, False): 1}"
    run, _coins = window
    k, n, t = run.shapes()[0]
    want = PH.units(k, n, t)
    for units, groups in ((want + 1, ((0,),)), (want, ((0,), (1,)))):
        monkeypatch.setattr(PH, "compiled", lambda *shape, units=units, groups=groups: SimpleNamespace(
            units=[None] * units, groups=groups))
        with pytest.raises(PH.C.CompileError) as refused:
            run.check()
        assert str(refused.value) == (f"position: {k} x {n} at T = {t}: {units} units in {len(groups)} groups, not "
                                      f"{want} in one")


def test_a_linear_of_a_window_that_commits_nothing_applies_as_its_arm_does():
    """`traced` puts a linear's ``apply`` under the credit gate's trace only for a window that commits. With
    ``commit`` off it hands the layer, its input and its bias to the arm's own, returns what that returns and reads
    nothing else of the run."""
    asked = []

    def arm(layer, x, bias):
        asked.append((layer, x, bias))
        return "y"
    run = SimpleNamespace(commit=False)
    apply = PHK.traced(run, 0, arm)
    assert apply("layer", "x") == "y" and apply("layer", "x", "bias") == "y"
    assert asked == [("layer", "x", None), ("layer", "x", "bias")] and vars(run) == {"commit": False}


def test_a_saved_window_keeps_a_sampling_coin_held_in_process_or_none_and_an_asked_session_does_not_hand_off(
        tmp_path, monkeypatch):
    """`suspend` saves a window's sampling coin by what it is. A window with none resumes with none. One held in the
    serving process is saved as its record without its challenger, and `resume` puts it on the resumed session's
    client. A reserved sample session is saved only unasked: once its one open was asked, here of a gate that
    refuses, the window does not hand off, and nothing is saved."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    host = _host(tmp_path, monkeypatch)
    run = TH._served(_remote())
    ends = [socket.socketpair() for _ in range(2)]
    clients = [run.rounds.challenger]
    try:
        session = run.draw
        assert isinstance(session, PHW.DrawSession)
        run.draw = None
        HO.suspend(run, tmp_path / "none")
        resumed, client = HO.resume(tmp_path / "none", ends[0][0].detach())
        clients.append(client)
        assert resumed.draw is None and resumed.rounds.challenger is client
        held = LiveRecord(Wd.draw_challenger(run.rounds.challenger.commitment.epoch, lambda n: bytes(n)), Wd.DRAW)
        run.draw = held
        HO.suspend(run, tmp_path / "held")
        resumed, client = HO.resume(tmp_path / "held", ends[1][0].detach())
        clients.append(client)
        assert isinstance(resumed.draw, LiveRecord) and resumed.draw.challenger is client
        state = {name: value for name, value in vars(held).items() if name != "challenger"}
        assert {name: value for name, value in vars(resumed.draw).items() if name != "challenger"} == state
        assert state["label"] == Wd.DRAW

        def shut(*args, **kw):
            raise ValueError("the gate is shut")
        run.draw, session.gate = session, shut
        with pytest.raises(CoinError) as refused:
            session.record(bytes(64))
        assert str(refused.value) == (f"{session.entry}: the window's draw session {session.reserved['session']}: the "
                                      "gate is shut")
        with pytest.raises(ValueError) as unsaved:
            HO.suspend(run, tmp_path / "asked")
        assert str(unsaved.value) == f"{session.entry}: a window hands off before its draw session opens"
        assert not (tmp_path / "asked").exists()
    finally:
        for client in clients:
            client._wire.sock.close()
        for _mine, theirs in ends:
            theirs.close()
    assert host.ended() is not None


def test_a_calls_rows_of_the_commit_off_its_k_or_past_its_slots_are_refused_and_a_table_is_registered_once():
    """`CommitTap.arow` takes a call's rows of the Commit from its site at the call's engine step. An x_q and an s_a
    there that are not one row a scale of the call's K give the call no rows, and rows past the window's slots are
    cut to them: each is refused by the call. A step's table of K the Commit registers a second time is refused by
    its name."""
    k, trees = 8, []

    class Tree:
        def entry(self):
            return {"root": "r", "domain": "d", "leaves": 2}

    def tree(run, call, words):
        trees.append((call.index, len(words)))
        return Tree()
    served = SimpleNamespace(m=2, registered={}, step_calls=lambda index: [index], a_tree=tree,
                             a_words=lambda call, codes, scales: np.zeros((len(codes), 1), np.uint16))

    def call(index, step, module):
        return SimpleNamespace(index=index, step=step, module=module, k=k, m=1)

    def acquired(tap, module, step, rows, scales, codes=k):
        tap.take(f"{module}/row_quant/0", np.zeros((rows, codes), np.int8), step)
        tap.take(f"{module}/row_quant/1", np.ones(scales, "<f4"), step)
    tap = CT.CommitTap()
    for index, (rows, scales, codes) in enumerate(((3, 2, k), (2, 2, k + 2))):
        acquired(tap, "m", 4 + index, rows, scales, codes)
        tap.arow(None, served, call(index, index, "m"), f"a@{index}", None)
        assert tap.refused == [f"call {i}: the Commit's x_q and s_a at m/row_quant are not one row a scale of K {k}"
                               for i in range(index + 1)]
        assert tap.where[CR.commit_rows(index, k)]["rows"] == [0]
    acquired(tap, "m", 6, 3, 3)
    tap.arow(None, served, call(2, 2, "m"), "a@2", None)
    assert tap.refused[2:] == ["call 2: the Commit's 3 rows are past its 2 slots"]
    assert tap.where[CR.commit_rows(2, k)] == {"round": 2, "calls": [2], "steps": [6], "sites": ["m/row_quant"], "rows": [2]}
    assert trees == [(0, 0), (1, 0), (2, 2)] and tap.own == 3
    tap.arow(None, served, call(3, 2, "n"), "another a@2", None)
    assert tap.refused[3:] == [f"the Commit registers {CR.commit_rows(2, k)} twice"]
    assert tap.own == 4 and sorted(tap.values) == [CR.commit_rows(s, k) for s in range(3)]


def test_a_window_that_holds_no_tree_of_a_rounds_table_has_none_a_named_record_reuses(tmp_path, monkeypatch):
    """`reused` holds a named record to the window's tree of each round's A table of K and input-ids table. A window
    whose served trees hold none of one of them, here taken out after its close, has no such table for a record to
    reuse: it is refused by the round, the table and the Commit's name of it."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (TC.Linear,))
    _record, run, _tap = TC._window(tmp_path, monkeypatch)
    served, previous = run.served, "12" * 64
    record = TC._reusing(CT.registration(run, previous), lambda rec: rec)
    assert CT.reused(run, CT.registration(run, previous), record) is None

    def refusal():
        window = CT.registration(run, previous)
        return CT.reused(run, window, {**record, "previous": CT.REG.digest(window)})
    ids = served.ids.pop(1)
    assert refusal() == (f"the window registers no round 1's input-ids table, so no Commit record reuses it as "
                         f"{CR.commit_ids(1)}")
    served.ids[1] = ids
    assert refusal() is None
    step, k, calls = CR.step_tables({"calls": served.calls, "linears": served.linears, "m": served.m}, served.steps)[0]
    table = served.a_table(calls[0])
    del served.registered[table]
    assert refusal() == (f"the window registers no round {step}'s A table {table}, so no Commit record reuses it as "
                         f"{CR.commit_rows(step, k)}")


def test_a_values_rows_in_a_host_tensor_are_committed_as_the_same_rows_in_an_array():
    """`pouw_h1_registered.commit` reads a tensor's rows on the host wherever it doesn't hash them on a CUDA device:
    u16 words in a host tensor register the tree their array does, and a tensor of another type is refused as an
    array of it is."""
    circuit = hashlib.sha512(b"a declared Program").hexdigest()
    words: np.ndarray = np.arange(12, dtype=np.uint16).reshape(3, 4)
    tensor = torch.from_numpy(words.view(np.uint8).copy()).view(torch.uint16)
    assert tensor.dtype == torch.uint16 and tuple(tensor.shape) == (3, 4)
    of_array = NR.commit(circuit, "v", words, KEY, 7, keep_rows=True)
    of_tensor = NR.commit(circuit, "v", tensor, KEY, 7, keep_rows=True)
    assert isinstance(of_array, DN.G.Registered) and isinstance(of_tensor, DN.G.Registered)
    assert of_tensor.root == of_array.root and of_tensor.leaves == 3
    assert [bytes(row) for row in of_tensor.rows] == [bytes(row) for row in of_array.rows]
    for other in (tensor.view(torch.int16), tensor.view(torch.uint8), tensor[0]):
        with pytest.raises(ValueError, match="^a registered value is at least one row of 16-bit words, every row the "
                                             "same width$"):
            NR.commit(circuit, "v", other, KEY, 7)


def test_the_tables_of_a_call_that_served_no_rows_are_rebuilt_from_no_record_and_read_no_kept_file(tmp_path):
    """`zy_value` and `a_value` rebuild a call's ``zy@<c>`` and its step's ``a/<K>@<s>`` from what the prover kept. A
    call that served no rows kept nothing, so no file of it is read: its z table is the rows of no record over the
    window's slots, at the call's zy nonce, and its A table zero codes of scale 0 over them, at its a nonce, each
    bound to the window's root."""
    circuit, root = hashlib.sha512(b"a declared Program").hexdigest(), "ab" * 64
    layout = {"m": 32, "tile": PH.TILE, "linears": [("q", K, 32)], "calls": [("q",)]}
    win = {"layout": layout, "shape": {"prefix": "h1/"}, "program": circuit, "root": root,
           "record": {"calls": [{"index": 0, "n": 32, "k": K, "served": 0, "rows": 0}]}}
    value = DN.zy_value(win, tmp_path, KEY, 0)
    no_z: np.ndarray = np.zeros((0, 32), "<i4")
    no_y: np.ndarray = np.zeros((0, 32), np.int16)
    words = SV.record_words(no_z, no_y, 32, PH.TILE)
    same = NR.commit(circuit, "h1/zy@0", words, KEY, SV.nonce(0, "zy"), keep_rows=True, window=root)
    assert isinstance(same, DN.G.Registered) and (value.value, value.leaves, value.root) == ("h1/zy@0", 1, same.root)
    assert value.root != NR.commit(circuit, "h1/zy@0", words, KEY, SV.nonce(0, "zy")).root
    assert not np.count_nonzero(words)
    no_codes: np.ndarray = np.zeros((0, K), np.int8)
    zero = SV.Served.a_words(SimpleNamespace(m=32), SimpleNamespace(k=K), no_codes, np.zeros(0, "<f4"))
    table = DN.a_value(win, tmp_path, KEY, 0)
    same = NR.commit(circuit, f"h1/a/{K}@0", zero, KEY, SV.nonce(0, "a"), keep_rows=True, window=root)
    assert isinstance(same, DN.G.Registered) and (table.value, table.root) == (f"h1/a/{K}@0", same.root)
    assert table.leaves == same.leaves == len(zero) and not np.count_nonzero(zero)
    assert list(tmp_path.iterdir()) == []


def test_a_statements_port_that_reads_no_row_at_a_sampled_proof_block_is_refused():
    """`call_rows` takes each port's row at each sampled proof block from credit's file. A port whose reads begin
    past a sampled proof block reads no row there: the call's rows are refused by the port and the call, before a
    value is read."""
    reg = {"ports": {CR.PUB_PORT: {DN.G.READS_FROM: 2, "reads": [0, 1], "value": "pub"}}}
    with pytest.raises(Refused, match=f"^port {CR.PUB_PORT} reads no row at a drawn unit of call 3$"):
        DN.call_rows({}, 3, [2, 1], reg, {}, None, None)


def test_the_hosts_commit_of_a_value_takes_one_192_byte_salt_a_row():
    """`host_commit` takes a value's given salts as 192 bytes a row, and refuses any other length by its bytes and
    the value's rows, before the host's library is asked for."""
    words: np.ndarray = np.zeros((3, 4), np.uint16)
    for salts in (b"", b"s" * 191, b"s" * 192, b"s" * (3 * 192 + 1)):
        with pytest.raises(ValueError, match=f"^one 192-byte salt per row: {len(salts)} bytes for 3 rows$"):
            RGD.host_commit(words, KEY, 7, salts, bytes(64))


def test_records_said_to_be_on_a_cuda_device_are_viewed_where_they_are_and_only_that_view_takes_their_z_table():
    """`zy_words` of records on a CUDA device (here a host tensor that says it is on one) that are the Program's
    tiles over every slot is a u16 view of them where they are, the words the host path builds. `take_z` hands a
    call's waiting z table to its value's registration only for that view: for another call, for rows on the host,
    of another type, not contiguous, at another address or of another shape it gives none and the table still waits;
    for the view it gives the table's domain id and nonce, once."""
    rows, n = 64, 32
    records = (torch.arange(2 * W.ZR.ZY_BYTES) % 251).to(torch.uint8).reshape(2, W.ZR.ZY_BYTES)
    on = records.as_subclass(TS._OnDevice)
    words = W.zy_words(on, rows, n, rows, W.ZR.UNIT)
    host = W.zy_words(records, rows, n, rows, W.ZR.UNIT)
    assert words.dtype == torch.uint16 and words.is_cuda and words.data_ptr() == records.data_ptr()
    assert isinstance(host, np.ndarray) and np.array_equal(words.view(torch.uint8).numpy().view("<u2"), host)
    call, did = SimpleNamespace(index=3), hashlib.sha512(b"the z table's domain").digest()
    held = (3, on, did, 9)
    run = SimpleNamespace(_zy=held)
    assert TBD.take_z(SimpleNamespace(), call, words) is None
    assert TBD.take_z(run, SimpleNamespace(index=4), words) is None and run._zy is held
    copy = words.clone()
    assert copy.is_cuda and copy.data_ptr() != words.data_ptr()
    others: list[Any] = [host, words.as_subclass(torch.Tensor), words.view(torch.uint8), words[:, ::2], copy, words[:1]]
    for other in others:
        assert TBD.take_z(run, call, other) is None and run._zy is held
    assert TBD.take_z(run, call, words) == (did, 9) and run._zy is None
    assert TBD.take_z(run, call, words) is None


needs_cxx = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="no C++ compiler")


@needs_cxx
def test_a_window_on_the_gpu_path_hashed_by_the_hosts_twins_registers_what_numpy_registers(monkeypatch):
    """Where H1 hashes on the GPU (here the host's twins of the kernels stand for the device, no hash ready before
    it is read), a served call's z table waits for its registered ``zy@<c>``. Records on the device are that
    value's rows where they are, so its hash commits the z table from the same read and the call's tables hash its
    A table alone. Records on the host are not, so the z table is hashed apart once the call is registered. An empty
    call's z table is built where the hash runs, and a public table's salts are derived there. Either way the window
    registers the entries, the roots, the call tables, the rounds and the calls root that it does with NumPy."""
    def window():
        torch.manual_seed(0)
        run, *_ = TS._window(TS.W1)
        served = run.served
        assert not served.refused, served.refused
        return ([bool(c.m) for c in run.calls],
                (served.entries(), served.roots(), served.digest(), run.tables, run.round_log, run.calls_root()))

    monkeypatch.setattr(TBD, "MODE", "python")
    monkeypatch.setattr(NR, "MODE", "numpy")
    calls, want = window()
    served, empty = calls.count(True), calls.count(False)
    values: list[tuple[bool, bool, bool]] = []
    tables: list[tuple[str, ...]] = []

    class Value:
        def __init__(self, words, key, nonce, row_salts, domain_id, public=None, also=None):
            on = bool(getattr(words, "is_cuda", False))
            rows = words.view(torch.uint8).numpy().view("<u2") if on else words
            values.append((on, public is not None, also is not None))
            if also is not None:
                self.z, *made = RGD.host_zy(rows, key, nonce, domain_id, also)
                self.made = tuple(made)
            else:
                salts = row_salts if public is None else RGD.host_public_salts(public[0], public[1], len(rows), public[2])
                self.z, self.made = None, RGD.host_commit(rows, key, nonce, salts, domain_id)

        def ready(self):
            return False

        def result(self):
            return self.made

        def also(self):
            return self.z

    class Tables:
        def __init__(self, laid, key, names=("a", "z")):
            tables.append(names)
            roots = TBD.host_roots(laid, key)
            self.roots, self.ready, self.names = (lambda: roots), (lambda: False), names

    @contextmanager
    def hashing():
        yield torch.device("cpu")

    monkeypatch.setattr(TBD, "MODE", "gpu")
    monkeypatch.setattr(NR, "MODE", "gpu")
    monkeypatch.setattr(TBD, "device_hash", Tables)
    for name, twin in (("device_hash", Value), ("zy_available", lambda: True), ("hashing", hashing),
                       ("device_public_salts", RGD.host_public_salts)):
        monkeypatch.setattr(RGD, name, twin)
    assert window() == (calls, want) and served and empty
    assert tables == [("a",), ("z",)] * served + [("a", "z")] * empty
    assert values and not any(on or also for on, _public, also in values) and any(public for _on, public, _also in values)
    on_host = len(values)
    del tables[:], values[:]
    zy_of = TS.OPS.scaled_mm_zy

    def on_device(*args):
        y, zy = zy_of(*args)
        return y, zy.as_subclass(TS._OnDevice)

    monkeypatch.setattr(TS.OPS, "scaled_mm_zy", on_device)
    assert window() == (calls, want)
    assert tables == [("a",)] * served + [("a", "z")] * empty and len(values) == on_host
    assert [v for v in values if v[0] or v[2]] == [(True, False, True)] * served
    head = b"a public table's salt head"
    there = [NR.public_salts(head, 1 << 33, 3, ok) for ok in (False, True)]
    monkeypatch.setattr(NR, "MODE", "numpy")
    assert there == [NR.public_salts(head, 1 << 33, 3, ok) for ok in (False, True)]


def test_h1_hashes_with_numpy_and_says_why_where_cuda_is_there_and_its_kernels_dont_build(monkeypatch):
    """`available`: a process with CUDA hashes on the GPU once the kernels are built for its current device. Where
    they don't build, here with no nvcc, a warning says that H1 hashes with NumPy and why, and the GPU path is not
    taken."""
    built: list[tuple] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(RGD, "nvcc", lambda: None)
    with pytest.warns(RuntimeWarning) as said:
        assert RGD.available.__wrapped__() is False
    why = ("H1's registered values hash with NumPy: the GPU path doesn't build here "
           "(RuntimeError: no nvcc (pouw_device.TOOLCHAIN, or on the path))")
    assert [str(w.message) for w in said] == [why]
    monkeypatch.setattr(RGD.RTD.DeviceLibrary, "of", lambda *asked: built.append(asked[1:]))
    assert RGD.available.__wrapped__() is True and built == [(0, RGD.device_lib, "rr_midstates")]


@needs_cxx
def test_a_zy_library_whose_args_are_not_the_bytes_of_zyargs_is_refused(monkeypatch):
    """The library that commits a call's z table beside its value takes its layout as `ZyArgs`. A build whose
    ``Args`` is another size, here against a `ZyArgs` of one word, is refused by both sizes."""
    class Short(RGD.ctypes.Structure):
        _fields_ = [("n", RGD.ctypes.c_uint32)]

    args = RGD.ctypes.sizeof(RGD.ZyArgs)
    monkeypatch.setattr(RGD, "ZyArgs", Short)
    with pytest.raises(RuntimeError, match=f"^zy_commit.cuh's Args is {args} bytes, ZyArgs 4$"):
        RGD.zy_host_lib.__wrapped__()


def test_a_pull_that_holds_two_engine_steps_of_the_ranks_commits_is_refused_by_its_steps():
    """The window's tap keeps one engine step a site, so what the ranks' Commits acquired by one pull is of one
    engine step. A pull that holds two is refused by its steps; what each rank acquired is still fed to the tap, and
    a rank with no Commit, which answers with nothing acquired, keeps its place."""
    x: np.ndarray = np.zeros((1, 8), np.int8)

    def rank(lo, *acquired):
        return {"lo": lo, "rows": [], "refused": [], "order_ok": True, "unopened": None,
                **({"acquired": list(acquired)} if acquired else {})}
    pulls = [[rank(2), rank(0, ("m/row_quant/0", 4, x), ("m/row_quant/1", 4, np.ones(1, "<f4"))),
              rank(4, ("n/row_quant/0", 5, x))],
             [rank(0, ("m/row_quant/0", 6, x)), rank(2), rank(4, ("n/row_quant/0", 6, x))]]
    tap = CT.CommitTap()
    run = SimpleNamespace(served=SimpleNamespace(tap=tap), shapes_=[])
    holder = PHR.Holder(run, SimpleNamespace(collective_rpc=lambda method: pulls.pop(0)), [{"decode_query_len": 1}], None)
    assert holder.pull() == 0
    assert tap.refused == ["one pull held engine steps [4, 5]: the window's tap keeps one step a site"]
    assert sorted(tap.acquired) == [(4, "m/row_quant", "0"), (4, "m/row_quant", "1"), (5, "n/row_quant", "0")]
    assert holder.pull() == 0 and len(tap.refused) == 1 and len(tap.acquired) == 5
    assert holder.pulls == 2 and sorted(holder.ranks) == [0, 2, 4]


def test_a_rank_call_with_no_rows_or_off_the_programs_order_refuses_the_windows_registered_tables():
    """A multiprocess window registers its tables from the rows its ranks ship with each call (`ingest`). A call
    whose rank shipped none refuses the window's tables by the call. One that is not at its step's and linear's
    place in the declared Program, or is past its calls, is refused by its index, step and linear
    (`Served.commit`). Either way the call is the window's and registers nothing."""
    linears = len(TS.SHAPES)

    def row(index, step, layer, **shipped):
        n, k = TS.SHAPES[layer]
        return {"call": dataclasses.asdict(PH.Call(index, step, layer, TS.MODULES[layer], 0, k, n, 0, TS.T)), **shipped}
    past = TS.STEPS * linears
    for shipped, said in ((row(0, 0, 0), "call 0: its rank shipped no rows for the window's registered tables"),
                          (row(1, 0, 0, served={}), "call 1 (step 0, linear 0) is off the declared Program's order"),
                          (row(past, TS.STEPS, 0, served={}),
                           f"call {past} (step {TS.STEPS}, linear 0) is off the declared Program's order")):
        run = PH.H1Run("t", SALT, TS.MODULES, TS.SHAPES, TS.T, TS.STEPS, key=KEY,
                       served=SV.Served(TS.TILE, TS._unit(), TS.BLOCKS, window=TS.W1))
        PHR.ingest(run, [shipped])
        index = shipped["call"]["index"]
        assert run.served.refused == [said]
        assert [c.index for c in run.calls] == [index] and run.tables[index]["registered"] == {}


def test_a_leafs_committed_files_are_held_to_the_manifest_and_a_staged_table_to_its_instances_rows(tmp_path):
    """`leaf_stagers` takes each of K's leaf statements from --circuits' committed files. A leaf with no file, no
    leaf Program or no template id there is refused by what to build; a leaf Program that is not the one the
    manifest commits, by its file; and a circuit file that is not the manifest's, by both digests. `FileStager.stage`
    refuses a port's table whose rows at the instances' refs are not their rows, before anything is written."""
    k = LL.LEAST_K
    name = f"{LL.LANE}{k}"
    circuit, program = tmp_path / f"{name}.circuit.txt", tmp_path / DR.LEAF_PROGRAM.format(name=name)
    text = b"a circuit\nMETA " + json.dumps({"ports": []}).encode() + b"\n"
    leaf = hashlib.sha512(b"a leaf Program").hexdigest()
    manifest = {"files": {name: "00" * 64, CR.LEAF_PROGRAMS: {name: leaf}}, "templates": {name: "a template"}}

    def refusal(manifest):
        with pytest.raises(Refused) as refused:
            DN.leaf_stagers(k, PH.TILE, tmp_path, manifest)
        return str(refused.value)
    absent = f"--circuits has no {name} file, leaf Program and template id: build them (pouw_h1_circuits --lanes {k})"
    assert refusal({}) == refusal(manifest) == absent
    circuit.write_bytes(text)
    assert refusal(manifest) == absent
    program.write_bytes(b"another leaf Program")
    for short in ({"files": manifest["files"]}, {**manifest, "files": {CR.LEAF_PROGRAMS: {name: leaf}}}):
        assert refusal(short) == absent
    assert refusal(manifest) == (f"--circuits' {program.name} is not the leaf Program the manifest commits "
                                 f"({CR.LEAF_PROGRAMS} {name})")
    program.write_bytes(b"a leaf Program")
    assert refusal(manifest) == (f"{name}'s circuit (SHA-512 {hashlib.sha512(text).hexdigest()[:16]}...) is not "
                                 f"--circuits' {name} file ({'00' * 8}...), the one the period-open committed")
    stager = DN.FileStager(circuit, "a template", [("p", 0, 0, ())], DR.LEAF_OUT_ROWS)
    with pytest.raises(ValueError, match="^port p's table rows at the instances' refs are not their rows$"):
        stager.stage({"p": [[1, 2]]}, tmp_path / "staged", KEY, {}, [0], "a set", {"p": [[1, 2], [3, 4]]}, {"p": [1]}, {})
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([circuit.name, program.name])


def test_a_windows_sampled_leaves_are_staged_only_in_a_lanes_shape_and_under_its_records_root(tmp_path):
    """`stage_leaves` stages K's sampled leaves as their lanes only where credit's lanes theorem reaches
    (`credit.lanes_shape`: refused by its reason), and commits each leaf's partials under the root the window's
    record names for its sample: a record that names none is refused, before a file of --circuits is read."""
    k = LL.LEAST_K
    shape = {"tiles": {k: {"j10r": False}}}

    def refusal(win):
        with pytest.raises(Refused) as refused:
            DN.stage_leaves(None, {"shape": shape, **win}, k, [0], {}, None, tmp_path, tmp_path, {}, {})
        return str(refused.value)
    off = {"tile": (16, 32)}
    why = CR.lanes_shape(off, shape, k)
    assert why is not None and refusal({"layout": off}) == f"K = {k}'s drawn leaves: {why}"
    assert CR.lanes_shape({"tile": PH.TILE}, shape, k) is None
    named: list[Any] = [None, "a root", {}, {"root": 7}]
    records: list[dict[str, Any]] = [{}, *({"draw": sample} for sample in named)]
    for record in records:
        assert refusal({"layout": {"tile": PH.TILE}, **record}) == ("the window's record names no draw root, under "
                                                                     "which a leaf's partials are committed")
    assert list(tmp_path.iterdir()) == []


def _stand_in(k):
    """`test_pouw_h1_served`'s stand-in proof block of K, under a name a saved window's state pickles."""
    return TS._unit()(k)


@pytest.mark.filterwarnings("ignore:The given NumPy array is not writable:UserWarning")
def test_a_saved_window_whose_program_declared_again_is_another_is_refused_at_its_resume(tmp_path, monkeypatch):
    """A saved window holds its Program's digest, not the Program, which `resume` declares again from the run's
    public shapes and the period's prefix. A saved state whose digest is not that Program's, here one changed in
    the file, is refused by the window's entry."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    host = _host(tmp_path, monkeypatch)
    served = SV.Served(PH.TILE, _stand_in, window=bytes(64))
    run = TH._served(_remote(transcript_dir=str(tmp_path / "w"), served=served), n=64, arm=TH.ZyArm)
    mine, theirs = socket.socketpair()
    try:
        path = HO.suspend(run, tmp_path / "saved")
        doc = pickle.loads(path.read_bytes())
        kind, state, layers = doc["state"]["served"]
        assert not run.served.refused, run.served.refused
        assert state["program_digest"] == run.served.program_digest and len(state["program_digest"]) == 128
        doc["state"]["served"] = (kind, {**state, "program_digest": "00" * 64}, layers)
        (tmp_path / "changed").mkdir()
        (tmp_path / "changed" / HO.FILE).write_bytes(pickle.dumps(doc, protocol=5))
        with pytest.raises(ValueError) as refused:
            HO.resume(tmp_path / "changed", mine.detach())
        assert str(refused.value) == (f"{run.entry}: the window Program declared again is not the one its tables were "
                                      "registered under")
    finally:
        mine.close()
        theirs.close()
        run.rounds.challenger._wire.sock.close()
    assert host.ended() is not None


#: a call's operand roots, as its seeds take them
RA, RB = hashlib.sha512(b"A").digest(), hashlib.sha512(b"B").digest()


def _operands(rows=64, n=64):
    """A call's A [rows, K] and B [n, K] s8 codes on the CPU: four tiles as given."""
    rng = np.random.default_rng(3)
    return RC.Call(*(torch.from_numpy(rng.integers(-127, 128, (r, K), dtype=np.int8)) for r in (rows, n)))


def _declared(defn=TS.StandIn):
    """A window opened under `test_pouw_h1_served`'s key whose Program declares, each step, a call of 32 weight rows
    and one of 64 at K over 64 slots: two tiles and four."""
    served = SV.Served(PH.TILE, TS._unit(defn=defn), window=TS.W1)
    served.open(SimpleNamespace(commits={}, shapes_=[(32, K), (64, K)], modules={0: "q", 1: "o"}, rows=64, steps=2))
    assert not served.refused, served.refused
    return served


def _asked(served, **made):
    """A producer for `served` under `SALT` and, unless ``roots`` is given, the roots `RA` and `RB`; the window's
    first call (20 served rows of its 64 slots, 32 weight rows), its codes and its weight."""
    rng = np.random.default_rng(5)
    codes, weight = (rng.integers(-127, 128, (rows, K), dtype=np.int8) for rows in (20, 32))
    made.setdefault("roots", lambda _call: (RA, RB))
    producer = RC.Receipts(served, SimpleNamespace(salt=SALT), **made)
    return producer, SimpleNamespace(index=0, layer=0, n=32, k=K), codes, weight


def _public(row):
    """A public row's bytes as its seeds lay them out, from its 16-bit words: each 32 bits a u32, low byte first."""
    raw = b"".join(int(word).to_bytes(2, "little") for word in row)
    return b"".join(raw[i:i + 4][::-1] for i in range(0, len(raw), 4))


def test_a_calls_receipts_are_the_xor_of_each_lanes_words():
    """With ``lanes``, `call_receipts` gives each proof block's words a lane, before the XOR across lanes: their XOR
    is the call's receipt, and no one lane's words are."""
    call = _operands()
    whole, per = (RC.call_receipts(call, SALT, 7, RA, RB, lanes=lanes) for lanes in (False, True))
    assert per.dtype == np.uint32 and per.shape == (4, RC.LANES, whole.shape[1])
    assert np.array_equal(np.bitwise_xor.reduce(per, axis=1), whole)
    assert not any(np.array_equal(per[:, lane], whole) for lane in range(RC.LANES))


def test_j10r_s_receipts_take_each_proof_blocks_noise_from_its_calls_one_stream():
    """Under J10-R a call has one stream, its call seed's blocks at call scale, and each proof block's noise is the
    blocks of it that its rows and columns read. The call's receipts are then `reference`'s under J10-R, proof block
    by proof block, and not v1's; they are the same from a stream handed in, and for the proof blocks asked for
    alone. A stream is refused for an A that is not whole row blocks, seeds for another count of calls are refused,
    and J10-R's reference takes no admit bits."""
    call = _operands()
    got = RC.call_receipts(call, SALT, 9, RA, RB, j10r=True)
    assert got.shape == (4, 2 * RC.layout(K).sets)
    assert [list(row) for row in got] == [RC.reference(call, SALT, 9, RA, RB, u, j10r=True)[1] for u in range(4)]
    assert not np.array_equal(got, RC.call_receipts(call, SALT, 9, RA, RB))
    stream = RC.call_stream(call, SALT, 9, RA, RB)
    assert tuple(stream.shape) == (RC.stream_blocks(64, 64, K), 16)
    assert np.array_equal(RC.call_receipts(call, SALT, 9, RA, RB, j10r=True, stream=stream), got)
    assert not np.array_equal(RC.call_receipts(call, SALT, 9, RA, RB, j10r=True, stream=stream.flip(0)), got)
    assert np.array_equal(RC.call_receipts(call, SALT, 9, RA, RB, only=[3, 0], j10r=True), got[[3, 0]])
    two = RC.call_streams([RA, RB], [3, 5], "cpu", None)
    assert [tuple(s.shape) for s in two] == [(3, 16), (5, 16)]
    assert np.array_equal(two[0].numpy(), RC.call_streams([RA], [5], "cpu", None)[0][:3].numpy())
    with pytest.raises(ValueError, match="^a J10-R call's A is its slot, whole blocks of 32 rows, not 20$"):
        RC.call_stream(_operands(rows=20), SALT, 9, RA, RB)
    with pytest.raises(ValueError, match="^2 seeds for 1 calls$"):
        RC.call_streams([RA, RB], [4], "cpu", None)
    with pytest.raises(ValueError, match=r"^J10-R's unit \(PouwH1Tile_v5\) takes no admit bits$"):
        RC.reference(call, SALT, 9, RA, RB, 0, gated=np.ones(32, np.uint8), j10r=True)


def test_a_dead_tiles_receipt_under_admit_bits_is_zero_and_a_live_ones_is_the_calls():
    """Under a call's admit bits a tile whose 32 rows admit nothing is dead: its receipt is the zero words, where
    the ungated call gives it the receipt of its noise. A live tile's is the call's. `gated_reference` gives each
    tile the same, and a call that admits nothing has no live tile."""
    call = _operands()
    admit: np.ndarray = np.arange(64) < 5
    got, live = RC.gated_receipts(call, admit, SALT, 9, RA, RB)
    whole = RC.call_receipts(call, SALT, 9, RA, RB)
    assert live == [0, 1] and np.array_equal(got[:2], whole[:2])
    assert not got[2:].any() and whole[2:].any(axis=1).all()
    assert [list(row) for row in got] == [RC.gated_reference(call, admit, SALT, 9, RA, RB, u) for u in range(4)]
    none, live = RC.gated_receipts(call, np.zeros(64, bool), SALT, 9, RA, RB)
    assert live == [] and none.shape == whole.shape and not none.any()


def test_the_kernels_seed_record_is_two_tags_then_the_salt_and_the_roots_in_big_endian_words():
    """`seed_record` is the units kernel's seed data, 30 words: the seed's tag and the receipt's, 24 bytes each,
    then the salt and the two roots, every 8 bytes a big-endian word. J10-R's has its call seed's tag in the seed's
    place and is otherwise the same. A salt or a root that is not 64 bytes is refused by its name and length.
    `seed_words` lays digests out the same way, a row each."""
    def laid(words):
        return b"".join(int(word).to_bytes(8, "big", signed=True) for word in words)

    v1, j10r = (RC.seed_record(SALT, RA, RB, j10r=j10r) for j10r in (False, True))
    assert v1.dtype == j10r.dtype == np.int64 and v1.shape == j10r.shape == (30,)
    assert laid(v1)[48:] == SALT + RA + RB and laid(j10r)[24:] == laid(v1)[24:]
    tags = {laid(v1)[:24], laid(v1)[24:48], laid(j10r)[:24]}
    assert len(tags) == 3 and all(tag.rstrip(b"\0").isascii() and tag.rstrip(b"\0") for tag in tags)
    assert laid(RC.seed_record(RB, SALT, RA))[48:] == RB + SALT + RA
    for at, name in enumerate(("salt", "root(A)", "root(B)")):
        given = [SALT, RA, RB]
        given[at] = given[at][:63]
        with pytest.raises(ValueError) as refused:
            RC.seed_record(*given)
        assert str(refused.value) == f"a unit's seeds take a 64-byte {name}, not 63 bytes"
    words = RC.seed_words([RA, RB], "cpu")
    assert words.dtype == torch.int64 and tuple(words.shape) == (2, 8)
    assert [laid(row.tolist()) for row in words] == [RA, RB]


def test_a_producer_that_skips_dead_tiles_gives_them_the_zero_receipt_and_refuses_under_j10r():
    """Made with ``skip_dead``, the producer gives a row block wholly past the call's served rows the zero receipt
    and every other tile the call's, where without it a dead row block has the receipt of its zero codes and its
    noise. Under J10-R, whose proof block is ungated, it refuses by name."""
    served = _declared()
    producer, call, codes, weight = _asked(served)
    plain = producer(call, codes, None, weight, None)
    skipping, *_ = _asked(served, skip_dead=True)
    got = skipping(call, codes, None, weight, None)
    assert got.shape == plain.shape == (2, 2 * RC.layout(K).sets) and skipping.calls == 1
    assert np.array_equal(got[0], plain[0]) and not got[1].any() and plain[1].any()
    u0, _n = producer.start(call)
    a, b = producer.operands(call, codes, weight)
    assert [list(row) for row in plain] == [RC.reference(RC.Call(a, b), SALT, u0, RA, RB, u)[1] for u in range(2)]
    under, *_ = _asked(_declared(TS.StandInJ10R), skip_dead=True)
    with pytest.raises(ValueError) as refused:
        under(call, codes, None, weight, None)
    assert str(refused.value) == ("skip_dead under J10-R: its unit (PouwH1Tile_v5) is ungated, a dead tile has its "
                                  "noise's receipt")
    assert under.calls == 0


@pytest.mark.parametrize("defn", [TS.StandIn, TS.StandInJ10R])
def test_a_producer_keeps_the_calls_it_is_told_to_with_what_a_check_of_them_reads(defn):
    """Made with ``keep``, the producer keeps each call that ``keep`` says, by its index: its A over the window's
    slots and its B, the salt, its first position and its roots, its receipts, its served rows, its public rows,
    and whether it skipped dead tiles and computed J10-R's receipts, which a J10-R window's Program says. Its
    receipts are `reference`'s of what it kept. A public row is the salt, the proof block's position and the roots,
    and under J10-R then the call's first position and the proof block's two blocks. A call ``keep`` declines is
    computed and not kept."""
    j10r = defn is TS.StandInJ10R
    producer, call, codes, weight = _asked(_declared(defn), keep=lambda call: call.index == 0)
    later = SimpleNamespace(index=2, layer=0, n=32, k=K)
    got = [producer(c, codes, None, weight, None) for c in (call, later)]
    assert list(producer.kept) == [0] and producer.calls == 2 and producer.j10r is j10r
    kept = producer.kept[0]
    (u0, n), (u2, _n) = producer.start(call), producer.start(later)
    assert (n, u2 - u0) == (2, 2 + 4)
    assert sorted(kept) == ["a", "b", "base", "j10r", "m", "pub", "receipts", "roots", "salt", "skip"]
    assert (kept["salt"], kept["base"], kept["roots"], kept["m"]) == (SALT, u0, (RA, RB), 20)
    assert (kept["skip"], kept["j10r"]) == (False, j10r)
    assert np.array_equal(kept["a"][:20].numpy(), codes) and not kept["a"][20:].any() and tuple(kept["a"].shape) == (64, K)
    assert np.array_equal(kept["b"].numpy(), weight)
    assert np.array_equal(kept["receipts"], got[0]) and not np.shares_memory(kept["receipts"], got[0])
    assert not np.array_equal(got[0], got[1])
    kept_call = RC.Call(kept["a"], kept["b"])
    assert [list(row) for row in got[0]] == [RC.reference(kept_call, SALT, u0, RA, RB, u, j10r=j10r)[1] for u in range(n)]
    rows = [_public(row) for row in kept["pub"]]
    assert [row[:200] for row in rows] == [SALT + (u0 + t).to_bytes(8, "big") + RA + RB for t in range(n)]
    if j10r:
        assert [(len(row), row[200:208]) for row in rows] == [(224, u0.to_bytes(8, "big"))] * n
        assert rows[0][208:] != rows[1][208:]
    else:
        assert [len(row) for row in rows] == [200] * n
    assert kept["pub"] == producer.public(call)


class _NoLaunch:
    """A units kernel that nothing may reach: a test that asks it for anything fails there."""

    def __getattr__(self, name):
        raise AssertionError(f"the units kernel's {name} was asked for")


def test_a_step_prepared_for_the_units_kernel_refuses_codes_and_tiles_off_the_program_before_any_launch():
    """`prepare`, with the units kernel and a run, computes a step's calls of one K together. Calls that serve
    different rows are left to be asked each on its own, and nothing is launched for them. Before any launch it
    refuses, by the calls' indices, codes that are not the step's rows of K int8 codes each, at most the window's
    slots, and a call whose proof blocks in the window Program are not its slots' tiles by its weight's. The same
    call asked on its own is refused the same way."""
    served = _declared()
    producer, call, codes, weight = _asked(served, units=_NoLaunch(), device="cpu")
    wide = SimpleNamespace(index=1, layer=1, n=64, k=K)
    wider: np.ndarray = np.ones((64, K), np.int8)
    assert producer.prepare([(call, codes, None, weight, None), (wide, codes[:7], None, wider, None)]) is None
    assert producer.prepare([(call, codes, None, weight, None)]) is None
    off: list[np.ndarray] = [codes.astype(np.int16), codes[:, :K // 2], np.zeros((65, K), np.int8)]
    for given in off:
        with pytest.raises(ValueError) as refused:
            producer.prepare([(call, given, None, weight, None), (wide, given, None, wider, None)])
        assert str(refused.value) == (f"calls [0, 1]: their codes are not {len(given)} rows of {K} int8 codes each, at "
                                      "most 64")
    short = SimpleNamespace(index=1, layer=1, n=32, k=K)
    with pytest.raises(ValueError) as refused:
        producer.prepare([(call, codes, None, weight, None), (short, codes, None, weight, None)])
    assert str(refused.value) == "call 1: 4 units in the window Program, not its 64 × 32 tiles"
    alone, *_ = _asked(served)
    with pytest.raises(ValueError) as refused:
        alone(short, codes, None, weight, None)
    assert str(refused.value) == "call 1: 4 units in the window Program, not its 64 × 32 tiles"
    assert (producer.calls, alone.calls) == (0, 0)


class _Hashlib:
    """A stand-in for the noise kernel: it writes hashlib's halves as the device's 32-bit integers, and keeps the
    shapes it was asked with."""

    def __init__(self):
        self.asked: list[tuple] = []

    @staticmethod
    def _digests(seeds):
        return [b"".join(int(word).to_bytes(8, "big", signed=True) for word in row) for row in seeds.tolist()]

    def launch(self, seeds, n, out):
        self.asked.append(("launch", tuple(seeds.shape), n, tuple(out.shape)))
        out.copy_(RC.noise_blocks(self._digests(seeds), n, "cpu", None).to(torch.int32))

    def launch_calls(self, seeds, starts, out):
        self.asked.append(("launch_calls", tuple(seeds.shape), starts.tolist(), tuple(out.shape)))
        counts = (starts[1:] - starts[:-1]).tolist()
        out.copy_(torch.cat(RC.call_streams(self._digests(seeds), counts, "cpu", None)).to(torch.int32))


def test_a_noise_kernels_halves_are_read_back_unsigned_and_each_calls_stream_at_its_start():
    """With a noise kernel, here a stand-in that writes hashlib's halves, `noise_blocks` and `call_streams` hand it
    the seeds as rows of big-endian words and ask once: for every seed's blocks, or for every call's stream in one
    launch, each at its start. What they read back is what they give with no kernel: each half unsigned, and each
    call its own blocks, a call of no block none."""
    seeds, kernel = [RA, RB, SALT], _Hashlib()
    got, want = (RC.noise_blocks(seeds, 5, "cpu", cubin) for cubin in (kernel, None))
    assert got.dtype == torch.int64 and torch.equal(got, want) and int(got.min()) >= 0 and int(got.max()) >= 1 << 31
    streams, wanted = (RC.call_streams(seeds, [3, 0, 5], "cpu", cubin) for cubin in (kernel, None))
    assert [tuple(s.shape) for s in streams] == [(3, 16), (0, 16), (5, 16)]
    assert all(torch.equal(s, w) for s, w in zip(streams, wanted, strict=True))
    assert kernel.asked == [("launch", (3, 8), 5, (3, 5, 16)), ("launch_calls", (3, 8), [0, 3, 3, 8], (8, 16))]


class _Chain:
    """A stand-in for the units kernel: it answers with the torch chain's receipts under `RA` and `RB`, as the
    device's 32-bit integers, and keeps what it was handed."""

    def __init__(self):
        self.seeded: list[tuple] = []
        self.asked: list[tuple] = []

    def seeds(self, record, u0, n, call=None):
        self.seeded.append((record, u0, n, call))
        return u0

    def receipts(self, a, b, seeds, live, j10r):
        self.asked.append((seeds, None if live is None else live.tolist(), j10r))
        return torch.from_numpy(RC.call_receipts(RC.Call(a, b), SALT, seeds, RA, RB, j10r=j10r).view(np.int32))


@pytest.mark.parametrize("defn", [TS.StandIn, TS.StandInJ10R])
def test_a_producer_hands_the_units_kernel_a_calls_seed_record_its_span_and_its_live_row_blocks(defn):
    """With the units kernel, here a stand-in that answers with the torch chain's receipts, a call asked on its own
    is one launch. The producer hands the kernel the seed record of the salt and the call's roots, the same one
    while the roots are the same and another under other roots, then the call's first position and its proof
    blocks, and under J10-R that position again as its call. Its receipts are read back unsigned, the torch
    chain's. Made with ``skip_dead`` it hands over which row blocks hold a served row, and under J10-R it refuses."""
    j10r = defn is TS.StandInJ10R
    served, kernel = _declared(defn), _Chain()
    producer, call, codes, weight = _asked(served, units=kernel, device="cpu",
                                           roots=lambda call: (RA, RB) if call.index == 0 else (RB, RA))
    chain, *_ = _asked(served)
    got = producer(call, codes, None, weight, None)
    assert got.dtype == np.uint32 and np.array_equal(got, chain(call, codes, None, weight, None))
    later = SimpleNamespace(index=2, layer=0, n=32, k=K)
    for c in (call, later):
        producer(c, codes, None, weight, None)
    (u0, n), (u2, _n) = producer.start(call), producer.start(later)
    (record, *first), (again, *second), (other, *third) = kernel.seeded
    assert first == second == [u0, n, u0 if j10r else None] and third == [u2, n, u2 if j10r else None]
    assert again is record and record.dtype == torch.int64
    assert np.array_equal(record.numpy(), RC.seed_record(SALT, RA, RB, j10r=j10r))
    assert np.array_equal(other.numpy(), RC.seed_record(SALT, RB, RA, j10r=j10r))
    assert kernel.asked == [(u0, None, j10r), (u0, None, j10r), (u2, None, j10r)] and producer.calls == 3
    skipping, *_ = _asked(served, units=_Chain(), device="cpu", skip_dead=True)
    if j10r:
        with pytest.raises(ValueError, match="^skip_dead under J10-R: its unit "):
            skipping(call, codes, None, weight, None)
    else:
        skipping(call, codes, None, weight, None)
        assert skipping.units.asked == [(u0, [1, 0], False)]


def test_the_one_reserve_after_a_periods_close_is_the_lanes_first():
    """`after_close` says whether the ledger closes the period, and gives the first ``reserve`` after that close
    whose prover, its ``for`` or else its ``by``, is the lane: none before the close or under another period's,
    another lane's passed over, and a later one of the lane's not taken."""
    def entry(kind, **body):
        return json.dumps({"kind": kind, "body": body})

    early, close = entry("reserve", by="vllm", session="s0"), entry("period-close", period="p1")
    assert PHW.after_close([], "vllm", "p1") == (False, None)
    assert PHW.after_close([entry("period-close", period="p0"), early], "vllm", "p1") == (False, None)
    assert PHW.after_close([early, close], "vllm", "p1") == (True, None)
    other = entry("reserve", by="other", session="s1")
    assert PHW.after_close([early, close, other], "vllm", "p1") == (True, None)
    mine = [entry("reserve", **{"for": "vllm", "by": "auditor", "session": "s2"}), entry("reserve", by="vllm", session="s3")]
    assert PHW.after_close([early, close, other, *mine], "vllm", "p1") == (True, json.loads(mine[0]))
    theirs = entry("reserve", **{"for": "other", "by": "vllm", "session": "s4"})
    assert PHW.after_close([close, theirs, mine[1], mine[0]], "vllm", "p1") == (True, json.loads(mine[1]))


def test_a_host_library_that_returns_nonzero_is_refused_by_what_it_was_asked(monkeypatch):
    """Each of the host's twins of the device's hashing returns zero once it has written what it was asked for. A
    library that returns anything else, here stand-ins that write nothing, is refused by what it was asked, so
    what it did not write is never read as a tree, a root or salts."""
    names = ("rr_commit_host", "rr_public_salts_host", "rr_ok_salts_host", "zy_commit_host", "rt_step_host")
    refuses = SimpleNamespace(**dict.fromkeys(names, lambda *_layout: 3))
    for module, name in ((RGD, "host_lib"), (RGD, "zy_host_lib"), (TBD.RTD, "host_lib")):
        monkeypatch.setattr(module, name, lambda: refuses)
    words: np.ndarray = np.zeros((3, 512), np.uint16)
    with pytest.raises(ValueError, match=r"^rr_commit_host refused the value's layout \(3\)$"):
        RGD.host_commit(words, KEY, 7, None, bytes(64))
    with pytest.raises(ValueError, match=r"^zy_commit_host refused the records' layout \(3\)$"):
        RGD.host_zy(words, KEY, 7, bytes(64), (bytes(64), 9))
    for ok in (False, True):
        with pytest.raises(ValueError, match="^the host's public salts refused the salt head$"):
            RGD.host_public_salts(b"a salt head", 5, 3, ok)
    codes: np.ndarray = np.zeros((2, K), np.int8)
    with pytest.raises(ValueError, match=r"^rt_step_host refused the table's layout \(3\)$"):
        TBD.host_roots([(bytes(64), 0, codes, np.zeros(2, np.uint32))], KEY)


def test_h1s_call_tables_hash_on_the_host_and_say_why_where_cuda_is_there_and_their_kernels_dont_build(monkeypatch):
    """`tables_device.available`: with no CUDA the call tables hash on the host and nothing is said. With CUDA and
    kernels that don't build, here with no nvcc, a warning says that they hash with `hiding_rows` and why; once they
    are built for the current device, the GPU path is taken. The z table's kernel is asked for only where the
    values' kernels are there, and where it doesn't build a warning says that the z tables hash apart from their
    values. Both build with `pouw_device`'s nvcc."""
    built: list[tuple] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert TBD.available.__wrapped__() is False and RGD.zy_available.__wrapped__() is False
    monkeypatch.setattr("verity.protocols.accounting.pouw.prover.serving.device.nvcc", lambda: "/a/toolchain/nvcc")
    assert TBD.nvcc() == RGD.nvcc() == "/a/toolchain/nvcc"
    monkeypatch.setattr("verity.protocols.accounting.pouw.prover.serving.device.nvcc", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    with pytest.warns(RuntimeWarning) as said:
        assert TBD.available.__wrapped__() is False
    assert [str(w.message) for w in said] == [("H1's call tables hash with hiding_rows: the GPU path doesn't build here "
                                               "(RuntimeError: no nvcc on the path or configured)")]
    monkeypatch.setattr(RGD, "available", lambda: True)
    with pytest.warns(RuntimeWarning) as said:
        assert RGD.zy_available.__wrapped__() is False
    assert [str(w.message) for w in said] == [("H1's call z tables hash apart from their zy@<c> values: zy_commit.cu "
                                               "doesn't build here (RuntimeError: no nvcc (pouw_device.TOOLCHAIN, or on "
                                               "the path))")]
    monkeypatch.setattr(TBD.RTD.DeviceLibrary, "of", lambda *asked: built.append(asked[1:]))
    assert TBD.available.__wrapped__() is True and built == [(0, TBD.device_lib, "rt_midstates", TBD.STREAMS)]
