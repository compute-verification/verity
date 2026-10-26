"""`verity_flock.rec_budget`, the stagers' mirror of the Lean verifier's caps on a V* session's registered reads
(`Flock.RecBudget`): the same inputs as `Flock/Test/RegBudget.lean`'s `#guard`s give the same refusals, word for word; and the
caps sit far above every deployed V* session at every `fast100` schedule: a table's instances (a level's `2 q` openings, a
later level's RecCap over both reps' `2 · 2^cap` tops, RecCap's `2^cap`), a part's ports (its row ports, at most
`MAX_REGIONS`, with `v` and `out`) and the deepest registered tree (`rec-acc`'s chain of every opening, both reps' and R's);
and V*'s list, the recursion's parts (each level's RecOpen, the algebra's parts, RecCap at each later level and over level
0's cap), at every `fast100` m of the deployed class's claims. A shared-row file past `MAX_INNER_ROWS` rows a table is
refused by the mirror and never written (`circuit.write`)."""
from __future__ import annotations

import pytest
from verity_flock import rec_algebra as A
from verity_flock import rec_budget as RB
from verity_flock import rec_residuals as RR

TAIL = "the recursion's call bound covers no more"


def _reg(ports: dict[str, int]) -> dict:
    return {"format": "verity/registered-reads/v1", "ports": {nm: {"leaves": n} for nm, n in ports.items()}}


def test_the_mirror_refuses_what_the_lean_guards_refuse():
    assert RB.regs(_reg({"acc": 2048, "out": 128, "coef": 510})) is None
    assert RB.regs(_reg({f"p{i}": 2 ** RB.MAX_DEPTH for i in range(RB.MAX_PORTS)})) is None
    assert RB.regs(_reg({f"p{i}": 1 for i in range(RB.MAX_PORTS + 1)})) == (
        f"refused: a V* session's registered reads name 4097 ports, more than 4096: {TAIL}")
    assert RB.regs(_reg({"acc": 2048, "big": 2 ** RB.MAX_DEPTH + 1})) == (
        "refused: a V* session's registered port big has 18446744073709551617 leaves, a tree of 65 levels, more than 64: "
        f"{TAIL}")
    assert RB.table("", 510) is None and RB.table("", RB.MAX_INSTANCES) is None
    assert RB.table("", RB.MAX_INSTANCES + 1) == (
        f": refused: a V* session's table of 65537 instances, more than 65536: {TAIL}")
    assert RB.count(10) is None and RB.count(RB.MAX_STATEMENTS) is None
    assert RB.count(RB.MAX_STATEMENTS + 1) == (
        f"refused: V*'s list at the verifier's inner shape holds 4097 statements, more than 4096: {TAIL}")
    assert RB.inner_rows(None) is None and RB.inner_rows({"a": 510, "b": RB.MAX_INNER_ROWS}) is None
    assert RB.inner_rows({"a": 510, "b": RB.MAX_INNER_ROWS + 1}) == (
        f"refused: the inner statement's public file has a shared-row table of more than 268435456 rows: {TAIL}")


def test_a_shared_row_file_past_the_inner_row_cap_is_not_written(monkeypatch):
    """`circuit.write` refuses a shared-row file whose table is past `MAX_INNER_ROWS` with the verifier's message, before
    hashing any row. A file past the real cap (2^28 rows' 128-byte `b || c`, 32 GiB) is more than a test can hand the
    verifier, so the cap is lowered here, and `Flock/Test/RegBudget.lean`'s guards hold the two sides' messages to each other
    on the same counts."""
    from types import SimpleNamespace

    import numpy as np
    from verity_flock import circuit as FC

    monkeypatch.setattr(RB, "MAX_INNER_ROWS", 2)
    meta = {"ports": [["x", 4, 0]], "input_ports": 1, "out_ports": []}
    why = f"refused: the inner statement's public file has a shared-row table of more than 2 rows: {TAIL}"
    assert RB.inner_rows({"x": 2}) is None and RB.inner_rows({"x": 3}) == why
    with pytest.raises(ValueError) as e:
        FC.write("", SimpleNamespace(subcircuit=None, name="s"), 0, 1, None, None, meta=meta, outs={}, share_rows=True,
                 tables={"x": np.zeros((3, 4), np.uint16)}, refs={"x": [0]}, salts={"x": [bytes(FC.SALT_BYTES)] * 3})
    assert str(e.value) == why


def test_log2ceil_is_the_verifiers():
    assert [RB.log2ceil(n) for n in (0, 1, 2, 3, 4, 5, 2 ** 64, 2 ** 64 + 1)] == [0, 0, 1, 2, 2, 3, 64, 65]


def test_the_caps_sit_far_above_every_deployed_v_star_session():
    deepest = most = 0
    for m in range(22, 36):
        s = A.fast100(m)
        opens = sum(lv.q for lv in s.levels)
        most = max(most, *(2 * lv.q for lv in s.levels), 2 ** s.levels[0].cap, *(2 * 2 ** lv.cap for lv in s.levels[1:]))
        for r in range(22, 36):
            deepest = max(deepest, RB.log2ceil(2 * opens + sum(lv.q for lv in A.fast100(r).levels) + 1))
    assert most <= 510 and 128 * most <= RB.MAX_INSTANCES
    assert deepest <= 12 and 5 * deepest <= RB.MAX_DEPTH
    assert 60 * (RR.MAX_REGIONS + 2) <= RB.MAX_PORTS


def test_the_statement_cap_sits_far_above_every_deployed_v_star_list():
    counts = {}
    for shape in [(m, 22, 8) for m in range(22, 36)] + [(22, 6, 16)]:
        try:
            sh = A.Shape(*shape)
            counts[shape] = 2 * len(sh.sched.levels) + len(RR.parts(RR.spec(sh)))
        except ValueError:
            continue
    assert len(counts) >= 10 and counts.get((27, 22, 8)) == 10, counts
    assert 64 * max(counts.values()) <= RB.MAX_STATEMENTS, counts
