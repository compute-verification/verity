"""stoch_recompute's DRAW LEDGER: 'RNG advances iff r_t' measured on the record, per
request -- the DRAWN positions compared as an ordered set with the positions the Program's stop rule EXECUTES ({prompt_len - 1 + (t - 1) :
r_t = 1}: to the cap, or the first EOS at slot t < C then min(A, C - t) lag slot(s)); every extra / missing position a FAIL reason by name
(over-cap without EOS fails; the lag slot is bounded by the cap); a record with no EOS stop says 'not measured', never 'pass'.  Torch-free (the accounting is pure
Python over the SampleEnd rows)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from verity_vllm.check.replay.stoch_recompute import draw_accounting, record_lag_rows
from verity_vllm.records.workload import SCHEMA

EVID = Path(__file__).resolve().parents[2] / "out" / "gen" / "sweep" / "evidence" / "stoch"


def _wl(tmp_path: Path, reqs: list[tuple[str, int, int]]) -> Path:
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"schema": SCHEMA, "model": "m", "engine": {}, "variants": {},
                             "requests": [{"request_id": r, "prompt_len": L, "max_tokens": T} for r, L, T in reqs]}))
    return p


def _build(tmp_path: Path, eos: list[int] | None, lag_decl: int | None = None) -> Path:
    b = tmp_path / "build"
    (b / "build_workload").mkdir(parents=True)
    (b / "build_workload" / "workload_program.json").write_text(json.dumps(
        {"stop_rule": {"eos_token_ids": eos, "eos_lag_rows": lag_decl}}))
    return b


def _draws(rid: str, L: int, n: int, first_step: int, served: list[int] | None = None, pos_of=None) -> dict:
    return {rid: [{"step": first_step + k, "row": 0, "pos": (pos_of(k) if pos_of else L - 1 + k),
                   "served": (served[k] if served else 7)} for k in range(n)]}


def test_every_request_to_its_cap_is_a_clean_ledger_and_r5_is_named_not_measured(tmp_path):
    wl = _wl(tmp_path, [("a", 10, 4), ("b", 3, 2)])
    d = _draws("a", 10, 4, 0) | _draws("b", 3, 2, 5)
    out = draw_accounting(d, wl, _build(tmp_path, [99]), lag_attested=(1, "test"))
    assert out["problems"] == [] and out["lag_slot_exercised"] is False and out["requests_at_cap"] == 2
    a = out["requests"]["a"]
    assert a["finish"] == "length" and a["ordered_set_equal"] is True and a["drawn_not_expected"] == [] and a["expected_not_drawn"] == []
    assert a["expected_executed_slots"] == {"n": 4, "first": 1, "last": 4} and a["expected_positions"] == {"n": 4, "first": 9, "last": 12}
    assert a["first_pos"] == 9 and a["last_pos"] == 12
    assert "NOT MEASURED" in out["measured"] and "not scheduled" in out["requests"]["b"]["lag_slot"]
    assert out["stop_rule"] == {"eos_token_ids": [99], "eos_ids_source": str(tmp_path / "build" / "build_workload" / "workload_program.json"),
                                "eos_lag_rows": 1, "eos_lag_rows_attested": 1, "eos_lag_source": "test", "eos_lag_rows_declared": None, "attested": True}


def test_eos_before_the_cap_with_exactly_one_lag_draw_at_prompt_len_plus_k_is_r5_measured(tmp_path):
    wl = _wl(tmp_path, [("a", 10, 8)])
    # EOS (99) served at draw 2 (slot 3, pos 11); the lag slot = draw 3 (slot 4) at pos 10 + 2 = 12; nothing after
    d = _draws("a", 10, 4, 0, served=[7, 7, 99, 5])
    out = draw_accounting(d, wl, _build(tmp_path, [99]), lag_attested=(1, "RunHeader.profile.engine_kwargs.async_scheduling=True"))
    assert out["problems"] == [] and out["lag_slot_exercised"] is True and out["requests_with_eos_before_cap"] == 1
    a = out["requests"]["a"]
    assert a["finish"] == "eos" and a["served_eos"] == {"draw_k": 2, "slot": 3, "step": 2, "pos": 11, "token": 99} and a["draws_after_served_eos"] == 1
    assert a["expected_executed_slots"] == {"n": 4, "first": 1, "last": 4} and a["rule"]["t_eos"] == 3
    assert a["lag_slot"]["measured"] is True and a["lag_slot"]["lag_slots_expected"] == 1 and a["lag_slot"]["bounded_by_cap"] is False
    assert a["lag_slot"]["draws"] == [{"slot": 4, "pos_expected": 12, "pos": 12, "step": 3, "pos_ok": True}]
    assert out["measured"].startswith("R5 MEASURED")


def test_a_draw_past_the_lag_slot_a_missing_lag_draw_and_a_lag_pos_off_by_one_each_fail_by_name(tmp_path):
    wl = _wl(tmp_path, [("a", 10, 8)])
    b = _build(tmp_path, [99])
    two_after = draw_accounting(_draws("a", 10, 5, 0, served=[7, 7, 99, 5, 5]), wl, b, lag_attested=(1, "t"))
    assert len(two_after["problems"]) == 1 and "RNG advanced without r_t" in two_after["problems"][0] and "past the EOS's lag slot" in two_after["problems"][0]
    assert two_after["requests"]["a"]["drawn_not_expected"] == [13]
    none_after = draw_accounting(_draws("a", 10, 3, 0, served=[7, 7, 99]), wl, b, lag_attested=(1, "t"))
    assert len(none_after["problems"]) == 1 and "the lag slot did not draw" in none_after["problems"][0] and none_after["requests"]["a"]["expected_not_drawn"] == [12]
    # lag 0 (async_scheduling False): the same 3-draw record is clean; the 4-draw one has a draw past the slot
    assert draw_accounting(_draws("a", 10, 3, 0, served=[7, 7, 99]), wl, b, lag_attested=(0, "t"))["problems"] == []
    assert "RNG advanced" in draw_accounting(_draws("a", 10, 4, 0, served=[7, 7, 99, 5]), wl, b, lag_attested=(0, "t"))["problems"][0]
    # the lag draw at the wrong pos: an extra position AND a missing one, both named
    off = draw_accounting(_draws("a", 10, 4, 0, served=[7, 7, 99, 5], pos_of=lambda k: 9 + k if k < 3 else 13), wl, b, lag_attested=(1, "t"))
    assert any("outside the executed set" in p for p in off["problems"]) and any("did not draw" in p for p in off["problems"])
    assert off["requests"]["a"]["drawn_not_expected"] == [13] and off["requests"]["a"]["expected_not_drawn"] == [12]
    assert off["measured"].startswith("R5 FAILS")


def test_f_r17b_35_over_cap_without_eos_fails_and_the_lag_slot_is_bounded_by_the_cap(tmp_path):
    """(a) 9 draws for cap 8 / lag 1 / NO EOS is 'the RNG advanced without r_t' (nothing is scheduled past the length
    bound; the lag slot exists only after an EOS before the cap) -- before the fix the only over-cap check was len > cap + lag and this
    passed silently; (b) EOS at slot C - 1 under A = 1: the cap bounds the lag slot to ONE draw (n_exec = min(t_eos + A, C)), which is
    correct, not a failure; a second draw after it is the failure."""
    wl = _wl(tmp_path, [("a", 10, 8)])
    b = _build(tmp_path, [99])
    # (a) no EOS, cap 8, 9 draws
    over = draw_accounting(_draws("a", 10, 9, 0), wl, b, lag_attested=(1, "t"))
    assert len(over["problems"]) == 1 and "past the length bound without an EOS" in over["problems"][0] and "RNG advanced without r_t" in over["problems"][0]
    assert over["requests"]["a"]["drawn_not_expected"] == [17] and over["requests"]["a"]["finish"] == "length"
    # (a') EOS at the cap draw (slot 8) plus one more draw: the cap has no slot after it
    over_eos = draw_accounting(_draws("a", 10, 9, 0, served=[7] * 7 + [99, 5]), wl, b, lag_attested=(1, "t"))
    assert len(over_eos["problems"]) == 1 and over_eos["requests"]["a"]["finish"] == "eos-at-cap" and over_eos["requests"]["a"]["drawn_not_expected"] == [17]
    # (b) EOS at slot 7 of cap 8 under A = 1: one lag draw (slot 8) and the record is MEASURED, clean
    near = draw_accounting(_draws("a", 10, 8, 0, served=[7] * 6 + [99, 5]), wl, b, lag_attested=(1, "t"))
    assert near["problems"] == [] and near["lag_slot_exercised"] is True and near["requests"]["a"]["lag_slot"]["lag_slots_expected"] == 1
    assert near["requests"]["a"]["lag_slot"]["bounded_by_cap"] is False and near["measured"].startswith("R5 MEASURED")
    # (b') a 9th draw after that bounded slot fails
    near_over = draw_accounting(_draws("a", 10, 9, 0, served=[7] * 6 + [99, 5, 5]), wl, b, lag_attested=(1, "t"))
    assert len(near_over["problems"]) == 1 and "past the EOS's lag slot" in near_over["problems"][0]
    # (b'') A = 0 with EOS at slot 7: no lag slot; 7 draws clean, 8 draws = a draw without r_t
    assert draw_accounting(_draws("a", 10, 7, 0, served=[7] * 6 + [99]), wl, b, lag_attested=(0, "t"))["problems"] == []
    assert "RNG advanced" in draw_accounting(_draws("a", 10, 8, 0, served=[7] * 6 + [99, 5]), wl, b, lag_attested=(0, "t"))["problems"][0]
    # a profile that is not 0/1 is refused by name (one rule)
    assert any("not 0 or 1" in p for p in draw_accounting(_draws("a", 10, 8, 0), wl, b, lag_attested=(2, "t"))["problems"])


def test_wrong_positions_two_draws_in_a_step_over_cap_and_a_short_stop_without_eos_are_named(tmp_path):
    wl = _wl(tmp_path, [("a", 10, 4), ("b", 3, 5), ("c", 3, 2)])
    b = _build(tmp_path, [99])
    shifted = draw_accounting(_draws("a", 10, 4, 0, pos_of=lambda k: 10 + k), wl, b, lag_attested=(1, "t"))   # pos = L + k: another row's counter
    assert any("outside the executed set" in p for p in shifted["problems"]) and any("did not draw" in p for p in shifted["problems"])
    assert shifted["requests"]["a"]["drawn_not_expected"] == [13] and shifted["requests"]["a"]["expected_not_drawn"] == [9]
    d = _draws("b", 3, 3, 0); d["b"].append({"step": 2, "row": 1, "pos": 5, "served": 7})                       # two rows at step 2
    dup = draw_accounting(d, wl, b, lag_attested=(1, "t"))
    assert any("two sampled rows in one engine step" in p for p in dup["problems"]) and dup["requests"]["b"]["expected_not_drawn"] == [6]
    d2 = _draws("b", 3, 3, 0); d2["b"].append({"step": 2, "row": 1, "pos": 3, "served": 7})                     # the same pos drawn twice
    assert any("repeated or out of order" in p for p in draw_accounting(d2, wl, b, lag_attested=(1, "t"))["problems"])
    over = draw_accounting(_draws("c", 3, 4, 0), wl, b, lag_attested=(1, "t"))                                    # 4 draws, cap 2, no EOS
    assert any("past the length bound without an EOS" in p for p in over["problems"]) and over["requests"]["c"]["drawn_not_expected"] == [4, 5]
    short = draw_accounting(_draws("b", 3, 3, 0), wl, b, lag_attested=(1, "t"))                                   # 3 of 5, no EOS served
    assert len(short["problems"]) == 1 and "did not draw" in short["problems"][0] and "deleted step" in short["problems"][0]
    assert short["requests"]["b"]["finish"] == "length" and short["requests"]["b"]["expected_not_drawn"] == [5, 6]
    # an undeclared stop rule: a short stop is UNJUDGED (not attested), a full run is still judged on positions / cap
    unk = draw_accounting(_draws("b", 3, 3, 0), wl, _build(tmp_path / "u", None), lag_attested=(None, "not attested"))
    assert "cannot be read as an EOS stop" in unk["problems"][0] and unk["stop_rule"]["attested"] is False
    assert unk["requests"]["b"]["finish"] == "unjudged" and unk["requests_unjudged"] == 1 and unk["measured"].startswith("R5 FAILS")
    assert draw_accounting(_draws("a", 10, 4, 0), wl, None, lag_attested=(None, "x"))["problems"] == []
    # an EOS before the cap with the profile unattested: unjudged by name
    un = draw_accounting(_draws("a", 10, 3, 0, served=[7, 99, 5]), wl, b, lag_attested=(None, "not attested"))
    assert un["requests"]["a"]["finish"] == "unjudged" and "lag slot cannot be judged" in un["problems"][0]


def test_eos_at_the_cap_draw_schedules_no_lag_slot_and_a_declared_lag_disagreeing_with_the_record_is_named(tmp_path):
    wl = _wl(tmp_path, [("a", 10, 3)])
    at_cap = draw_accounting(_draws("a", 10, 3, 0, served=[7, 7, 99]), wl, _build(tmp_path, [99]), lag_attested=(1, "t"))
    assert at_cap["problems"] == [] and at_cap["requests"]["a"]["finish"] == "eos-at-cap" and "no slot after it" in at_cap["requests"]["a"]["lag_slot"]
    assert at_cap["lag_slot_exercised"] is False and at_cap["requests_at_cap"] == 1
    dis = draw_accounting(_draws("a", 10, 3, 0), wl, _build(tmp_path / "d", [99], lag_decl=0), lag_attested=(1, "hdr=True"))
    assert any("declares stop_rule.eos_lag_rows 0 but the record attests hdr=True -> 1" in p for p in dis["problems"])


def test_expected_executed_slots_is_the_spec_s7_rule_and_names_what_it_cannot_judge():
    from verity_vllm.check.replay.stoch_recompute import expected_executed_slots as E
    assert E([7, 7, 7, 7], 4, 1, {99})[0] == [1, 2, 3, 4]                       # to the cap: no lag slot
    assert E([7, 99], 4, 1, {99})[0] == [1, 2, 3]                               # EOS at slot 2 -> lag slot 3
    assert E([7, 99], 4, 0, {99})[0] == [1, 2]                                  # sync: no lag slot
    assert E([7, 7, 99], 4, 1, {99})[0] == [1, 2, 3, 4]                         # EOS at slot C - 1: the lag slot is the cap slot
    assert E([7, 7, 7, 99], 4, 1, {99})[0] == [1, 2, 3, 4]                      # EOS at the cap slot: nothing after
    assert E([7, 99, 5, 99], 6, 1, {99})[0] == [1, 2, 3]                        # the FIRST EOS rules; a later one is on a non-executed slot
    assert E([7, 7], 4, 1, None)[0] is None and "cannot be read as an EOS stop" in E([7, 7], 4, 1, None)[1]["unjudged"]
    assert E([7, 7, 7, 7], 4, 1, None)[0] == [1, 2, 3, 4]                       # undeclared EOS but the cap is reached: judged on the cap
    assert E([7, 99], 4, None, {99})[0] is None and "unattested" in E([7, 99], 4, None, {99})[1]["unjudged"]


def test_the_rule_is_pads_recurrence_and_the_slot_to_position_map_is_the_records_first_draw_at_prompt_len_minus_1():
    """The executed set comes from `verity_vllm.program.lifting.continuation.recurrence` (ONE mechanism, no twin here), the
    comparison from `compare_draw_positions`.  The slot -> position map is the RECORD's: slot 1 draws at pos prompt_len - 1 (the query
    position whose logits are sampled; qwen B8 L=1024 -> first_pos 1023, device_facts_match on every row), slot t at prompt_len - 2 + t.
    An earlier `expected_draw_positions` mapped slot t to prompt_len + t - 1, one more than the record; the two maps now agree and
    the ledger's positions ARE `expected_draw_positions(...)["positions"]` (one mechanism, no map of its own)."""
    from verity_vllm.program.lifting import continuation as CT
    from verity_vllm.check.replay.stoch_recompute import expected_executed_slots as E
    exp, info = E([7, 7, 99], 8, 1, {99})
    assert exp == [1, 2, 3, 4] and info["t_eos"] == 3 and info["rule"].startswith("verity_vllm.program.lifting.continuation.recurrence")
    ours = [10 - 1 + (t - 1) for t in exp]                       # L = 10: 9, 10, 11, 12 (the lag slot 4 at pos 12)
    pad = CT.expected_draw_positions([7, 7, 99], cap=8, A=1, eos={99}, prompt_len=10)
    assert pad["slots"] == exp and pad["n"] == 4 and pad["lag_positions"] == [pad["positions"][-1]]
    assert pad["positions"] == ours, (pad["positions"], ours)      # the two maps AGREE (slot 1 at L - 1); the ledger takes the padding's field
    cmp = CT.compare_draw_positions(ours, ours + [13])
    assert cmp["ok"] is False and cmp["extra"] == [13] and "RNG advanced without r_t" in cmp["problems"][0]


def test_record_lag_rows_reads_the_header_then_versions_json_and_says_not_attested(tmp_path):
    class H:
        profile = {"engine_kwargs": {"async_scheduling": True}}
    assert record_lag_rows(H()) == (1, "RunHeader.profile.engine_kwargs.async_scheduling=True")
    assert record_lag_rows({"engine_kwargs": {"async_scheduling": "false"}}) == (0, "RunHeader.engine_kwargs.async_scheduling='false'")
    (tmp_path / "versions.json").write_text(json.dumps({"engine_kwargs": {"async_scheduling": False}}))
    assert record_lag_rows({}, tmp_path) == (0, "capture/versions.json engine_kwargs.async_scheduling=False")
    assert record_lag_rows({})[0] is None and "not attested" in record_lag_rows({})[1]
    # a versions.json that is there but can't be read is named, and the ledger fails on it rather than taking the declared lag unattested
    (tmp_path / "versions.json").write_text('{"engine_kwargs": ')
    lag, why = record_lag_rows({}, tmp_path)
    assert lag is None and why.startswith("unreadable: capture/versions.json (JSONDecodeError") and "async_scheduling is not read" in why
    assert record_lag_rows(H(), tmp_path)[0] == 1                                  # the header attests first: the file isn't needed
    wl = _wl(tmp_path, [("a", 10, 4)])
    out = draw_accounting(_draws("a", 10, 4, 0), wl, _build(tmp_path, [99], lag_decl=1), lag_attested=(lag, why))
    assert out["problems"] == [f"draw ledger: the record's lag attestation is {why}"] and out["stop_rule"]["eos_lag_rows"] == 1


@pytest.mark.parametrize("name, data", [("build_workload/workload_program.json", '{"stop_rule": '),
                                        ("build_workload_program.json.gz", b"not gzip")])
def test_a_stop_rule_file_that_cant_be_read_is_a_ledger_problem_not_an_undeclared_rule(tmp_path, name, data):
    b = tmp_path / "build"
    (b / name).parent.mkdir(parents=True, exist_ok=True)
    (b / name).write_bytes(data.encode() if isinstance(data, str) else data)
    out = draw_accounting(_draws("a", 10, 4, 0), _wl(tmp_path, [("a", 10, 4)]), b, lag_attested=(1, "test"))
    assert len(out["problems"]) == 1 and out["problems"][0].startswith(f"draw ledger: unreadable: the Program's stop rule {b / name} (")
    assert out["stop_rule"]["eos_ids_source"] == str(b / name) and out["stop_rule"]["attested"] is False


@pytest.mark.parametrize("row, att, n_rows, V", [
    ("llama32-1b__bf16__l40s__tp1__b8__i1024__o128__mixed__stoch-t0.8-p0.95__bi-eager", "r17_1627_b8_int5_3b9e301c9", 526, 128256),
    ("qwen25-15b__bf16__l40s__tp1__b8__i1024__o128__mixed__stoch-t0.8-p0.95__bi-eager", "r17_1846_b8_int1_88dd07fd3", 388, 151936),
])
def test_the_two_b8_stoch_records_of_record_carry_a_clean_ledger_every_request_to_its_cap_r5_not_exercised(row, att, n_rows, V):
    """The B8 stochastic rows of record (evidence dirs): every sampled row is exact-match, so match_stoch.json's rows ARE the whole
    ledger; each of the 8 requests drew exactly cap times at pos prompt_len - 1 + k, one per step; no request stopped at EOS -> the
    lag slot was not scheduled on either record and the ledger says R5 is NOT MEASURED there (the negative is named, not passed)."""
    att_dir = EVID / row / att
    ms = att_dir / "match_stoch.json"
    if not ms.exists():
        pytest.skip(f"evidence dir absent: {att_dir}")
    st = json.loads(ms.read_text())
    rows = [r for r in st["rows"] if r.get("request_id")]
    assert len(rows) == n_rows == st["summary"]["sampled_rows_total"] and all(r["status"] == "exact-match" for r in rows)
    d: dict[str, list[dict]] = {}
    for r in rows:
        d.setdefault(r["request_id"], []).append({"step": r["step"], "row": r["row"], "pos": r["pos"], "served": r["served"]})
    wl = Path(__file__).resolve().parents[2] / "workloads" / f"{row}.json"
    gp = json.loads(gzip.open(att_dir / "build_workload_program.json.gz", "rt").read())
    lag_decl = (gp.get("stop_rule") or {}).get("eos_lag_rows")
    out = draw_accounting(d, wl, att_dir, lag_attested=(1, "RunHeader.profile.engine_kwargs.async_scheduling=True (the row_pod's vLLM v1 default; "
                                                                 "GM-01 G6 read the same, verified_eos_steps [])"))
    assert out["problems"] == [], out["problems"]
    assert out["stop_rule"]["eos_token_ids"] and out["stop_rule"]["eos_lag_rows_declared"] == lag_decl
    assert len(out["requests"]) == 8 and out["requests_at_cap"] == 8 and out["lag_slot_exercised"] is False
    wlj = json.loads(wl.read_text())
    for q in wlj["requests"]:
        e = out["requests"][q["request_id"]]
        assert e["draws"] == q["max_tokens"] and e["first_pos"] == q["prompt_len"] - 1 and e["last_pos"] == q["prompt_len"] - 2 + q["max_tokens"]
        assert e["one_draw_per_step"] and e["ordered_set_equal"] and e["finish"] == "length" and e["drawn_not_expected"] == [] == e["expected_not_drawn"]
    assert sum(e["draws"] for e in out["requests"].values()) == n_rows
    assert out["measured"].startswith("R5 NOT MEASURED")
