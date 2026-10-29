"""`commit.served_record`: a request's committed values by position, held to what was streamed, under the schedules arbitrary
traffic makes (chunked prefill, preemption and recompute, async overrun past a finish, prefix-cache hits), and refused by
name when a committed value differs from the served one."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from verity_vllm.commit import served_record as SR
from verity_vllm.commit.served_record import IDS_NAMES, SAMPLED, SAMPLER_INPUTS, Record, Row

IDS = IDS_NAMES[1]
SEED = 11


class FakeCommitter:
    """`_layouts` and its `SamplerInputsSource`'s rows as `NativeHostCommitter` keeps them once a window is finalized
    (host-retained values)."""

    def __init__(self) -> None:
        self._layouts: dict[int, list] = {}
        self.sources = [SR.SamplerInputsSource(self)]
        self.served_rows = self.sources[0].rows

    def step(self, rows: list[tuple], ids: list[int], sampled: list[int], *, seeds: list[int] | None = None,
             pos: list[int] | None = None, drop: str | None = None) -> None:
        s = len(self._layouts)
        rs = [Row(*r) for r in rows]
        tensors = {IDS: ids, SAMPLED: sampled, SAMPLER_INPUTS[0]: seeds or [SEED] * len(rows),
                   SAMPLER_INPUTS[1]: pos if pos is not None else [r.last for r in rs]}
        self._layouts[s] = [SimpleNamespace(name=n, dtype="torch.int64", nbytes=8 * len(v), dev_off=-1, stream_off=-1,
                                            host=np.asarray(v, dtype=np.int64).tobytes())
                            for n, v in tensors.items() if n != drop]
        self.served_rows[s] = rows


def served(prompt: list[int], out: list[int], *, chunk: int | None = None, preempt_after: int | None = None,
           overrun: int = 0, rid: str = "a-0123abcd") -> FakeCommitter:
    """One request served alone: its prompt in chunks of ``chunk``, one decode row an output, a preemption and recompute
    after ``preempt_after`` outputs, and ``overrun`` rows past the last streamed token (each sampling a discarded 999)."""
    com, P, t = FakeCommitter(), len(prompt), prompt + out
    step = chunk or P
    for lo in range(0, P, step):
        hi = min(P, lo + step)
        com.step([(rid, 0, hi - lo, lo, P)], prompt[lo:hi], [out[0] if hi == P else 555])
    for k in range(1, len(out)):
        if preempt_after is not None and k == preempt_after:
            com.step([(rid, 0, P + k, 0, P + k)], t[:P + k], [out[k]])
            continue
        com.step([(rid, 0, 1, P + k - 1, P)], [out[k - 1]], [out[k]])
    for j in range(overrun):
        p = P + len(out) - 1 + j
        com.step([(rid, 0, 1, p, P)], [out[-1] if j == 0 else 999], [999])
    return com


def absorbed(com: FakeCommitter, known: dict | None = None) -> Record:
    rec = Record(known=known or {"a": None})
    rec.absorb(com, 0)
    return rec


PROMPT, OUT = [5, 6, 7, 8, 9], [21, 22, 23, 2]


def test_plain_request_matches():
    rec = absorbed(served(PROMPT, OUT))
    got = rec.check("a", PROMPT, OUT, seed=SEED)
    assert got["passed"], got
    assert (got["rows"], got["cache_hit"], got["preemptions"], got["overrun_rows"]) == (4, 0, 0, 0)
    assert rec.windows[0]["passed"]


def test_chunked_prefill_discards_partial_samples():
    got = absorbed(served(PROMPT, OUT, chunk=2)).check("a", PROMPT, OUT, seed=SEED)
    assert got["passed"], got
    assert got["discarded_samples"] == 2                 # chunks [0,2) and [2,4) sample nothing that is served


def test_preemption_recompute_reads_the_same_ids():
    got = absorbed(served(PROMPT, OUT, preempt_after=2)).check("a", PROMPT, OUT, seed=SEED)
    assert got["passed"], got
    assert got["preemptions"] == 1 and got["recomputed_positions"] == len(PROMPT) + 1


def test_async_overrun_past_the_finish_is_recorded_not_streamed():
    got = absorbed(served(PROMPT, OUT, overrun=2)).check("a", PROMPT, OUT, seed=SEED)
    assert got["passed"], got
    assert got["overrun_rows"] == 2


def test_two_requests_in_one_step_are_sliced_by_rows():
    com = FakeCommitter()
    com.step([("a", 0, 3, 0, 3), ("b", 3, 5, 0, 2)], [1, 2, 3, 4, 5], [30, 40])
    com.step([("b", 0, 1, 2, 2), ("a", 1, 2, 3, 3)], [40, 30], [41, 31])
    rec = absorbed(com, {"a": None, "b": None})
    assert rec.check("a", [1, 2, 3], [30, 31], seed=SEED)["passed"]
    assert rec.check("b", [4, 5], [40, 41], seed=SEED)["passed"]


def test_prefix_cache_hit_needs_a_committed_source():
    com = FakeCommitter()
    com.step([("a", 0, 4, 0, 4)], [1, 2, 3, 4], [9])
    com.step([("b", 0, 2, 4, 6)], [7, 8], [10])
    rec = absorbed(com, {"a": None, "b": None})
    hit = rec.check("b", [1, 2, 3, 4, 7, 8], [10], seed=SEED)
    assert hit["passed"] and hit["cache_hit"] == 4 and hit["prefix_source"] == "a", hit
    miss = rec.check("b", [1, 2, 3, 5, 7, 8], [10], seed=SEED)
    assert not miss["passed"] and any(p.startswith("prefix cache") for p in miss["problems"])


A_PROMPT, A_OUT = [1, 2, 3, 4], [9, 10]
B_PROMPT, B_OUT = A_PROMPT + A_OUT + [7, 8], [11, 12]
C_PROMPT, C_OUT = B_PROMPT + B_OUT + [6], [13]


def three_turns(flip: int | None = None) -> Record:
    """A conversation of three turns, each turn's prompt the last one's prompt, its streamed ids and the user's next ids: b reads
    [0, 5) from a's cache, c reads [0, 9) from a's and b's (b computed [5, 9) itself). ``flip``: a's committed id at that
    position is another than the one served."""
    a_in = list(A_PROMPT)
    if flip is not None:
        a_in[flip] += 90
    com = FakeCommitter()
    com.step([("a", 0, 4, 0, 4)], a_in, [9])
    com.step([("a", 0, 1, 4, 4)], [9], [10])
    com.step([("b", 0, 3, 5, 8)], B_PROMPT[5:], [11])
    com.step([("b", 0, 1, 8, 8)], [11], [12])
    com.step([("c", 0, 2, 9, 11)], C_PROMPT[9:], [13])
    return absorbed(com, {"a": None, "b": None, "c": None})


def test_a_later_turns_prefix_is_covered_by_several_requests():
    rec = three_turns()
    assert rec.check("a", A_PROMPT, A_OUT, seed=SEED)["passed"]
    b = rec.check("b", B_PROMPT, B_OUT, seed=SEED)
    assert b["passed"] and b["cache_hit"] == 5 and b["prefix_spans"] == [["a", 0, 5]], b
    c = rec.check("c", C_PROMPT, C_OUT, seed=SEED)
    assert c["passed"] and c["cache_hit"] == 9 and c["prefix_spans"] == [["a", 0, 5], ["b", 5, 9]], c
    assert c["prefix_source"] == "a"


def test_a_flipped_committed_id_in_a_reused_prefix_fails_its_owner_and_every_reader():
    rec = three_turns(flip=2)
    a = rec.check("a", A_PROMPT, A_OUT, seed=SEED)
    assert not a["passed"] and any(p.startswith("inputs: 1 position(s) differ, first 2") for p in a["problems"]), a
    for rid, prompt, out in (("b", B_PROMPT, B_OUT), ("c", C_PROMPT, C_OUT)):
        got = rec.check(rid, prompt, out, seed=SEED)
        assert not got["passed"] and any(p.startswith("prefix cache: position 2 of") for p in got["problems"]), got


def test_a_prefix_committed_after_the_reader_started_is_no_source():
    com = FakeCommitter()
    com.step([("b", 0, 2, 4, 6)], [7, 8], [10])
    com.step([("a", 0, 4, 0, 4)], [1, 2, 3, 4], [9])
    got = absorbed(com, {"a": None, "b": None}).check("b", [1, 2, 3, 4, 7, 8], [10], seed=SEED)
    assert not got["passed"] and any(p.startswith("prefix cache: position 0 of") for p in got["problems"]), got


def test_a_source_must_agree_with_the_served_ids_before_what_it_covers():
    """y committed only position 0's id, x every id but position 0's: together they hold [0, 4), but x's KV at 1..3 is of
    another prefix."""
    com = FakeCommitter()
    com.step([("y", 0, 1, 0, 1)], [1], [9])
    com.step([("x", 0, 4, 0, 4)], [7, 2, 3, 4], [9])
    com.step([("b", 0, 2, 4, 6)], [7, 8], [10])
    got = absorbed(com, {"x": None, "y": None, "b": None}).check("b", [1, 2, 3, 4, 7, 8], [10], seed=SEED)
    assert not got["passed"] and any(p.startswith("prefix cache: position 1 of") for p in got["problems"]), got


@pytest.mark.parametrize("served_prompt, served_out, problem", [
    ([5, 6, 7, 0, 9], OUT, "inputs: 1 position(s) differ"),
    (PROMPT, [21, 22, 24, 2], "samples: 1 position(s) differ"),
    (PROMPT, OUT + [3], "samples: 1 position(s) not committed"),
])
def test_a_differing_value_fails_by_name(served_prompt, served_out, problem):
    got = absorbed(served(PROMPT, OUT)).check("a", served_prompt, served_out, seed=SEED)
    assert not got["passed"]
    assert any(p.startswith(problem) for p in got["problems"]), got["problems"]


EOS = {"reason": "stop", "stop_reason": None, "eos_ids": [2]}


def test_a_stream_cut_short_fails_on_its_finish_not_as_an_overrun():
    rec = absorbed(served(PROMPT, OUT))
    assert rec.check("a", PROMPT, OUT, seed=SEED, finish=EOS)["passed"]
    assert rec.check("a", PROMPT, OUT[:-1], seed=SEED)["passed"]          # without its finish, the cut reads as an overrun
    got = rec.check("a", PROMPT, OUT[:-1], seed=SEED, finish=EOS)
    assert not got["passed"] and any(p.startswith("finish: stopped on EOS") for p in got["problems"]), got


@pytest.mark.parametrize("finish, out, ok", [
    ({"reason": "length", "max_tokens": 4}, OUT, True),
    ({"reason": "length", "max_tokens": 5}, OUT, False),
    ({"reason": "stop", "stop_reason": 2}, OUT, True),
    ({"reason": "stop", "stop_reason": 23}, OUT, False),
    ({"reason": "abort", "k": 4, "max_tokens": 9}, OUT, True),
    ({"reason": "abort", "k": 3, "max_tokens": 9}, OUT, False),            # aborted after 3, and 4 were streamed
    ({"reason": "abort", "k": 4, "max_tokens": 4}, OUT, False),            # at its cap it finished on length
    ({"reason": "cancelled"}, OUT, False),
    ({"reason": "stop", "stop_reason": "c"}, [1, 3], True),
    ({"reason": "stop", "stop_reason": "c"}, [3, 1], False),              # the string was already there a token earlier
    ({"reason": "stop", "stop_reason": "c"}, [1, 1], False),
    ({"reason": "stop", "stop_reason": "c", "min_tokens": 2}, [3, 1, 3], True),  # unchecked until min_tokens are out
    ({"reason": "stop", "stop_reason": "bc", "min_tokens": 1}, [1, 2, 3], True),  # matched across the last id's start
])
def test_finish_problems(finish, out, ok):
    """A stop string's finish reads the same with its end found where the tokenizer is (``stop_end``) and no ``decode``."""
    decode = lambda ids: "".join("abc"[i - 1] for i in ids)  # noqa: E731
    assert (SR.finish_problems(out, finish, decode) == []) is ok
    if isinstance(finish.get("stop_reason"), str):
        end = SR.stop_string_end(out, finish["stop_reason"], int(finish.get("min_tokens") or 0), decode)
        assert (SR.finish_problems(out, {**finish, "stop_end": end}) == []) is ok


def test_a_seed_other_than_the_requests_fails():
    got = absorbed(served(PROMPT, OUT)).check("a", PROMPT, OUT, seed=SEED + 1)
    assert not got["passed"] and any(p.startswith("seeds:") for p in got["problems"])


def test_an_overrun_input_off_the_chain_fails():
    com = served(PROMPT, OUT)
    com.step([("a-0123abcd", 0, 1, len(PROMPT) + len(OUT) - 1, len(PROMPT))], [77], [999])
    got = absorbed(com).check("a", PROMPT, OUT, seed=SEED)
    assert not got["passed"] and any(p.startswith("overrun:") for p in got["problems"])


def test_readback_refuses_a_draw_at_another_position_and_unknown_requests():
    com = FakeCommitter()
    com.step([("a", 0, 3, 0, 3)], [1, 2, 3], [9], pos=[1])
    com.step([("zz", 0, 1, 0, 1)], [1], [9])
    rec = Record(known={"a": None})
    w = rec.absorb(com, 0)
    assert not w["passed"]
    assert any("read position 1, the row ends at 2" in e for e in w["readback_errors"])
    assert any("runner request zz is no admitted request" in e for e in w["readback_errors"])


def test_readback_refuses_a_step_without_its_sampler_inputs():
    com = FakeCommitter()
    com.step([("a", 0, 3, 0, 3)], [1, 2, 3], [9], drop=SAMPLER_INPUTS[0])
    w = Record(known={"a": None}).absorb(com, 0)
    assert not w["passed"] and "runner.sampler/seed" in w["readback_errors"][0]


def test_row_kinds():
    assert Row("a", 0, 4, 0, 6).kind == "prefill-chunk" and not Row("a", 0, 4, 0, 6).samples
    assert Row("a", 0, 2, 4, 6).kind == "prefill" and Row("a", 0, 2, 4, 6).samples
    assert Row("a", 0, 1, 6, 6).kind == "decode"


def test_base_id_strips_the_engines_suffix_only():
    assert SR.base_id("cmpl-1-0-0123abcd") == "cmpl-1-0"
    assert SR.base_id("varied.00001") == "varied.00001"
    assert SR.base_id("x-0123abcd", {"x-0123abcd": 1}) == "x-0123abcd"


def test_window_commit_reads_the_committers_verdict():
    ok = SimpleNamespace(run_root=b"\x01" * 64, steps=[1, 2], extra={"identity_check": {"ok": True, "steps_checked": 2}})
    assert SR.window_commit(ok)["passed"]
    bad = SimpleNamespace(run_root=b"\x01" * 64, steps=[1], extra={"identity_check": {"ok": False, "violations_n": 1}})
    assert not SR.window_commit(bad)["passed"]
    assert not SR.window_commit(SimpleNamespace(run_root=b"\x01", steps=[], extra={"errors": ["x"]}))["passed"]
