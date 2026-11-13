"""H1's checks refuse, by name, the forms its other tests never hand them: a request's first slot and a step's slots
under tables a prover committed consistently (`pouw_h1_requests.request_reason`, `slot_reason`), a round's row and coin
(`round_reason`, `rounds_reason`), a record's calls (`tables_reason`), a window's coins and its draw
(`pouw_h1_challenger.trusted_rounds_reason`, `pouw_h1_draw`), and what H1's settings, its input ids, a static
declaration and the gate's ``model-key`` answer are parsed from."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from fractions import Fraction

import pytest
from verity.protocols.pouw.verifier import salt as PS
from verity.protocols.pouw.verifier import window as Wd
from verity.protocols.pouw.verifier.audit import HM96, commit_rows
from verity.protocols.pouw.verifier.coins import Challenger
from verity.protocols.spot.device import challenger as LC
from verity.protocols.spot.device.challenger import LiveRecord

import verity_vllm.protocols.pouw.h1.challenger as PHC
import verity_vllm.protocols.pouw.h1.draw as PHD
import verity_vllm.protocols.pouw.h1.model_key as MK
import verity_vllm.protocols.pouw.h1.requests as RQ
import verity_vllm.protocols.pouw.h1.served as SV
import verity_vllm.protocols.pouw.h1.static as ST
import verity_vllm.protocols.pouw.h1.words as W
from tests.protocols import challenger_host as CH
from tests.protocols.test_pouw_h1 import (
    DRAW_RESERVED,
    KEY,
    SALT,
    Linear,
    _admitted_run,
    _entry,
    _host,
    _opening,
    _remote,
    _rounds,
    _salted,
    _serve_window,
    _settings,
)
from verity_vllm import protocols as PO
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


def test_a_draw_session_or_a_period_without_what_it_goes_with_is_refused():
    """``draw_session`` is the draw session of H1's window on its challenger, and ``period`` goes with a Pearl-C
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
        assert refused(entry, {"pouw": {**settings["pouw"], "draw_session": DRAW_RESERVED}}) == (no_window,)
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


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A window served under the auditor device's challenger (`test_pouw_h1._serve_window`), its session and its draw
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


def test_a_windows_draw_is_held_to_its_root_its_parts_and_its_coin(served):
    """The window's draw is refused by name when its root isn't 128 lowercase hex, when a part its root is over is
    absent or isn't hex, when its coin is no opening, and when the challenger's log shows its draw coin following
    another root, or more than the one."""
    host, honest, _record = served
    draw = honest["draw"]
    for root in ("zz", draw["root"].upper(), None):
        assert _trusted(host, {**honest, "draw": {**draw, "root": root}}) == "the window's draw root is not 128 lowercase hex"
    assert _trusted(host, {**honest, "draw": None}) == "the window's draw root is not 128 lowercase hex"
    for edit, what in (({"draw": {k: v for k, v in draw.items() if k != "receipts"}}, "'receipts'"),
                       ({"calls_root": "zz"}, "non-hexadecimal number found in fromhex() arg at position 0"),
                       ({"calls_root": None}, "fromhex() argument must be str, not None")):
        assert _trusted(host, {**honest, **edit}) == f"the window's draw root is malformed ({what})"
    for coin in ({"bad": 1}, None):
        assert _trusted(host, {**honest, "draw": {**draw, "coin": coin}}) == (
            f"the window's draw coin is malformed ({NO_OPENING})")
    signed, root = host.draw.challenger.commitment, bytes.fromhex(draw["root"])
    assert PHD.draw_reason(honest, signed, (root,)) is None
    for received in ((bytes(64),), (root, root), (bytes(64), root)):
        assert PHD.draw_reason(honest, signed, received) == (
            "the challenger's log does not show the window's draw root as the one root its draw coin followed: a draw "
            "opened before the window's rounds, or after another root")


@pytest.mark.parametrize("families", [((Wd.DRAW, 2),), ((Wd.DRAW, 1), (Wd.SALT, 1))], ids=["two-draws", "draw-and-salt"])
def test_a_draw_session_of_another_layout_is_refused(tmp_path, monkeypatch, families, served):
    """A draw session holds its one draw coin and nothing else. One the gate opened withheld on the window's draw root,
    whose draw coin 0 opened after that root and which its client closed, is still refused when its challenger signed
    two draw coins, or a salt coin beside the draw."""
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


def test_the_draw_sessions_files_are_none_where_the_ledger_pins_no_session_on_the_draw_root(served, tmp_path):
    """`draw_files` fetches the session the ledger opens on the window's draw root, pinned; where the ledger opens none
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
