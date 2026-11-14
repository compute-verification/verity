"""`second_window`'s built parts on the cases `Spot.Ranks`' `#guard`s hold the Lean to: the field's reader, the
unit-to-member rule and the ranks' checks, each refusal by an honest case with one edit."""

from verity.protocols.proofs.spot.verifier import second_window as W

H = "ab" * 64
RANKS = {"of": [{"units": {"T": 3}, "window": H, "wires": 0}, {"units": {"T": 2}, "window": "cd" * 64, "wires": 0}],
         "prefix": W.PREFIX}


def reg(r: int, members: list, run: str | None = None) -> dict:
    return {"rank": r, "digest": RANKS["of"][r]["window"], "circuit": f"bytes{r}", "run": run or f"root/rank{r}",
            "members": members}


def held(key: str, base: int, units: int, leaves: int) -> dict:
    return {"key": key, "base": base, "units": units, "outs": [["o0", leaves]]}


REGS = [reg(0, [held("T/c0", 0, 2, 32), held("T/c1", 2, 1, 16)]), reg(1, [held("T/c0", 0, 2, 32)])]
RECORD = {"ranks": RANKS, "templates": [["T", 5]]}
SERVED = [(H, "bytes0"), ("cd" * 64, "bytes1")]


def test_the_unit_to_member_rule_reads_each_ranks_registration():
    ranks = W.ranks_of(RANKS)
    assert [W.segment(ranks, "T", r) for r in (0, 1)] == [(0, 3), (3, 5)]
    assert W.member_of(ranks, REGS, "T", 2) == (0, "T", 1, 2, 1) and W.member_of(ranks, REGS, "T", 4) == (1, "T", 0, 0, 2)
    assert W.member_of(ranks, REGS, "T", 5) is None
    for bad in [{**RANKS, "prefix": "r/"}, {**RANKS, "x": 0}, {**RANKS, "of": []},
                {**RANKS, "of": [{**RANKS["of"][0], "window": "AB" * 64}]},
                {**RANKS, "of": [{**RANKS["of"][0], "units": {"T": True}}]}]:
        assert W.ranks_of(bad) is None


def test_an_honest_pair_passes_and_each_edit_fails_one_code():
    def codes(**edit) -> list[str]:
        args = {"record": RECORD, "records": [{}, {}], "served": SERVED, "registrations": REGS, "root": "root",
                "logged": lambda c: True, **edit}
        return [c for c, _ in W.rank_failures(**args)]
    built: list[str] = []
    assert codes() == built
    assert codes(records=[{}]) == ["ranks-wires", "ranks-reads"]
    assert codes(served=[SERVED[1], SERVED[0]]) == ["ranks-window", *built]
    assert codes(registrations=[REGS[0], {**REGS[1], "run": "other/rank1"}]) == ["ranks-run", *built]
    assert codes(logged=lambda c: c != "bytes1") == ["ranks-run", *built]
    assert codes(registrations=[REGS[0], reg(1, [held("T/c0", 0, 2, 31)])]) == ["ranks-members", *built]
    assert codes(registrations=[reg(0, [held("T/c0", 0, 2, 32), held("T/c2", 2, 1, 16)]), REGS[1]]) == \
        ["ranks-members", *built]
    assert codes(record={**RECORD, "templates": [["T", 4]]}) == ["ranks-units", *built]
    assert codes(record={**RECORD, "ranks": {**RANKS, "prefix": "r/"}}) == ["ranks-shape"]
    assert [c for c, _ in W.failures(False, RECORD, [{}, {}], SERVED, REGS, "root", lambda c: True, [], [])] == \
        [*built, "ranks-members"]


def test_wires_and_reads_are_the_ranks_moved_to_their_units():
    ranks = W.ranks_of({"of": [{"units": {"A": 2, "B": 2}, "window": H, "wires": 1}] * 2, "prefix": W.PREFIX})

    def wire(a: int, b: int) -> dict:
        return {"at": a, "base": b, "out": "o0", "port": "x", "radix": [2], "reader": "A", "step": [1], "writer": "B"}

    def read(p: str, a: int) -> dict:
        return {"at": a, "base": 0, "param": p, "port": "w", "radix": [2], "reader": "B", "step": [0]}

    def reg(ps: dict, rd: list) -> dict:
        return {"openings": [], "params": ps, "reads": rd, "roots": [], "tables": []}
    rank = {"registered": reg({"w": 8}, [read("w", 0)]), "wires": [wire(0, 0)]}

    def codes(ws: list, rd: list) -> list[str]:
        joined = {"registered": reg({"r0/w": 8, "r1/w": 8}, rd), "templates": [["A", 4], ["B", 4]], "wires": ws}
        return ([] if W.wires_ok(ranks, joined, [rank, rank]) else ["ranks-wires"]) + \
            ([] if W.reads_ok(ranks, joined, [rank, rank]) else ["ranks-reads"])
    reads = [read("r0/w", 0), read("r1/w", 2)]
    assert codes([wire(0, 0), wire(2, 2), wire(2, 0)], reads) == []
    assert codes([wire(0, 0), wire(2, 0)], reads) == ["ranks-wires"]
    assert codes([wire(0, 0), wire(2, 2), wire(0, 0)], reads) == ["ranks-wires"]
    assert codes([wire(0, 0), wire(2, 2)], [read("r0/w", 0), read("r1/w", 0)]) == ["ranks-reads"]
