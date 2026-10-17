"""`verity.ml.circuits.memory_map`: per step, held and moved bits against a leaf-by-leaf count over the lowered circuit's reads, and a
request's map by hand."""
from __future__ import annotations

import bisect

import pytest

from test_census import brute, programs, request
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.memory_map import KINDS, memory_map
from verity.ml.circuits.program import input_id, is_input


def brute_map(P, ends: list[int]) -> dict:
    """The memory map's steps, leaf by leaf, from `test_census.brute`."""
    nodes = P.root.body.nodes
    I, S = sum(1 for nd in nodes if is_input(nd.fn)), len(ends)
    step = lambda k: bisect.bisect_right(ends, k)  # noqa: E731
    held, moved = {k: [0] * S for k in KINDS}, {k: [0] * S for k in KINDS}
    born, read = [0] * S, [0] * S
    for (j, _t), w in brute(P).items():
        rs = {step(k) for k in w["reads"]}
        touched = rs | {step(k) for k in w["cands"]}
        if j >= I:
            kind, birth = "computed", step(j)
            born[birth] += w["bits"]
        else:
            kind = "registered" if input_id(nodes[j].fn.id)[0] == "registration" else "run"
            birth = -1 if kind == "registered" or not touched else min(touched) - 1
        for s in range(birth + 1, max(touched | {birth}) + 1):
            held[kind][s] += w["bits"]
        for s in rs:
            read[s] += w["bits"]
            if s > birth:
                moved[kind][s] += w["bits"]
    return {"held": held, "moved": moved, "born": born, "read": read}


@pytest.mark.parametrize("label,P", list(programs()), ids=lambda x: x if isinstance(x, str) else "")
def test_held_and_moved_are_the_leaves(label, P):
    """With every Call its own step, the map's held, moved, born and read bits are the leaf-by-leaf count's."""
    nodes = P.root.body.nodes
    ends = [k + 1 for k, nd in enumerate(nodes) if not is_input(nd.fn)]
    c = census(P, ends)
    m = memory_map(c)
    want = brute_map(P, ends)
    assert {k: m["steps"][k] for k in want} == want, label
    assert m["steps"]["count"] == len(ends) and m["census"] == c["digest"]
    assert sum(m["values"]["moved_bits"]) == m["totals"]["moved_bits"] == sum(sum(v) for v in want["moved"].values())
    assert len(m["values"]["birth"]) == len(c["values"]["producer"])
    assert all(b <= r for b, r in zip(m["values"]["birth"], m["values"]["last_read"]))


def test_the_request_map():
    """Weights are held throughout and the table only through its selections; a KV row is held from the step after it is written
    to the last attention that reads it; a sampled token is held into the step that embeds it; the prompt token into step 0."""
    P = request()
    m = memory_map(census(P, step_ends(P, {"TSelect_v1"})))
    st = m["steps"]
    assert st["held"] == {"computed": [0, 96, 160], "registered": [1088] * 3, "run": [32, 0, 0]}
    assert st["moved"] == {"computed": [0, 96, 160], "registered": [64] * 3, "run": [32, 0, 0]}
    assert st["born"] == [288] * 3 and st["read"] == [352, 416, 480]
    assert st["select_gates"] == [4] * 3 and st["select_bits"] == [64] * 3
    assert m["totals"] == {"peak_held_bits": 1248, "peak_step": 2, "held_bit_steps": 3552, "moved_bits": 480, "born_bits": 864,
                           "select_gates": 12, "select_bits": 192}


def test_a_run_input_is_born_before_the_step_that_first_reads_it():
    P = request()
    c = census(P, step_ends(P, {"TSelect_v1"}))
    m = memory_map(c)
    tok = c["values"]["producer"].index(0)
    assert (m["values"]["birth"][tok], m["values"]["last_read"][tok], m["values"]["read_steps"][tok]) == (-1, 0, 1)


def test_the_map_is_deterministic_and_refuses_another_schema():
    P = request()
    c = census(P, step_ends(P, {"TSelect_v1"}))
    assert memory_map(c) == memory_map(census(request(), step_ends(P, {"TSelect_v1"})))
    with pytest.raises(ValueError, match="verity/census/v1"):
        memory_map(c | {"schema": "verity/census/v0"})
