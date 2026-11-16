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
    assert [c for c, _ in W.failures(False, W.Given(dockets=()), RECORD, [{}, {}], SERVED, REGS, "root", lambda c: True, [], [])] == \
        ["two-stage-form"]
    assert [c for c, _ in W.failures(False, W.Given(dockets=()), RECORD, [{}, {}], SERVED, REGS, "root", lambda c: True, [("T", 5)], [])] == \
        ["ranks-members"]


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

    def wide(b: int, s: int) -> dict:
        return {"at": 0, "base": b, "out": "o0", "port": "x", "radix": [4], "reader": "A", "step": [s], "writer": "B"}
    assert codes([wire(0, 0), wire(2, 2), wide(3, -1)], reads) == []
    assert codes([wire(0, 0), wire(2, 2), wide(0, 1)], reads) == ["ranks-wires"]
    assert codes([wire(0, 0), wire(2, 2), wire(2, 0)], reads) == []
    assert codes([wire(0, 0), wire(2, 0)], reads) == ["ranks-wires"]
    assert codes([wire(0, 0), wire(2, 2), wire(0, 0)], reads) == ["ranks-wires"]
    assert codes([wire(0, 0), wire(2, 2)], [read("r0/w", 0), read("r1/w", 0)]) == ["ranks-reads"]


def wire(a: int, b: int, port: str = "x") -> dict:
    return {"at": a, "base": b, "out": "o0", "port": port, "radix": [2], "reader": "A", "step": [1], "writer": "B"}


def bread(param: str, step: int) -> dict:
    return {"at": 0, "base": 0, "param": param, "port": "p", "radix": [2], "reader": "A", "step": [step]}


EX_RANKS = W.ranks_of({"of": [{"units": {"A": 2, "B": 2}, "window": H, "wires": 1}] * 2, "prefix": W.PREFIX})
EX_REGS = [{"rank": r, "digest": "", "circuit": "", "run": "", "members": [held("A/c0", 0, 2, 1), held("B/c0", 0, 1, 1),
                                                                          held("B/c1", 1, 1, 1)]}
           for r in (0, 1)]


def test_the_readers_parts_and_wire_tree_are_the_lean_guards():
    """`Spot.SecondWindow`'s `#guard`s, case for case."""
    assert W.links_of({"boundary_reads": [bread("o0", 2), bread("o0", 2)]}) == [W.SS.Link("A", "p", "o0")]
    assert W.links_of({"boundary_reads": [bread("o0", 2), bread("o1", 2)]}) is None
    assert W.links_of({"boundary_reads": [wire(0, 0)]}) is None and W.links_of({}) == []
    assert W.wires_of({"wires": [wire(0, 0), wire(2, 2)]}) == [W.SS.Wire("A", "x", "B", "o0")]
    rows = W.rows_of({"wires": [wire(0, 0), wire(2, 2)], "boundary_reads": [bread("o0", 2)]}, [("A", 1), ("A", 2)])
    assert [(r[2], r[4], r[6]) for r in rows] == [(1, "B", 1), (2, "B", 2), (1, "o0", 2)]
    assert [(m[0], m[2]) for m in W.parts_of(EX_RANKS, EX_REGS, [("A", 3), ("A", 1), ("A", 2)])] == [(1, 0), (0, 0)]
    assert W.parts_of(EX_RANKS, EX_REGS, [("A", 4)]) is None
    ms = W.parts_of(EX_RANKS, EX_REGS, [("A", 1), ("A", 3)])

    def one(a: int, b: int) -> dict:
        return {"at": a, "base": b, "out": "o0", "port": "z", "radix": [1], "reader": "A", "step": [1], "writer": "B"}

    def codes(ws: list) -> list[str]:
        return [c for c, _ in W.wire_tree_failures(EX_RANKS, {"wires": ws}, ms)]
    assert codes([wire(0, 0), wire(2, 2)]) == []
    assert codes([wire(0, 0), wire(2, 2), wire(0, 1, "y")]) == ["ranks-wire-tree"]
    assert codes([one(0, 0), one(1, 1)]) == [] and codes([one(0, 0), one(1, 2)]) == ["ranks-wire-tree"]


def test_table_rows_are_refused_as_round_2_refuses_them():
    stmt = {"draw": {}, "header": {}, "registered": {"ports": {"tok": {"table_rows": 4}}}, "template": "A"}

    def why(record: dict) -> list[str]:
        return [d["why"] for c, d in W.session_failures(W.Given(dockets=()), EX_RANKS, EX_REGS, record, [], [], [stmt])
                if c == "second-table-rows"]
    assert why({"wires": [wire(0, 0, "tok")]}) == [W.SS.KEY_ROWS_WHY]
    assert why({"wires": [wire(0, 0)]}) == [W.SS.TABLE_ROWS_WHY]


def _w(reader: str, port: str, writer: str, at: int, radix: int, base: int) -> dict:
    return W._pattern({"at": at, "base": base, "out": "o0", "port": port, "radix": [radix], "reader": reader,
                       "step": [1], "writer": writer})


def test_a_port_two_selects_write_is_read_through_the_one_its_units_read():
    """`Spot.SecondWindow`'s `#guard` on `wireFor`."""
    ws = [_w("A", "live", "F", 0, 1, 0), _w("A", "live", "L", 1, 1, 0)]
    assert W.wire_for(W.Index(ws), "A", "live", 0, 1) == (W.SS.Wire("A", "live", "F", "o0"),)
    assert W.wire_for(W.Index(ws), "A", "live", 1, 1) == (W.SS.Wire("A", "live", "L", "o0"),)
    assert W.wire_for(W.Index(ws), "A", "live", 0, 2) is None and W.wire_for(W.Index(ws), "A", "x", 0, 2) == (None,)
    assert W.wire_for(W.Index(ws[:1]), "A", "live", 0, 2) is None


def test_a_wired_read_opens_the_writer_ranks_tree_and_an_own_read_its_own():
    """`Spot.SecondWindow`'s `#guard` on `windowReadOk`."""
    def root(port: str, r: str) -> dict:
        return {"domain": "d" + r, "leaves": 2, "port": port, "root": r}
    dockets = ({"roots": [root("A/y", "a0")]}, {"roots": [root("B/o0", "b1")]})
    ws = [_w("A", "x", "B", 0, 2, 2)]
    own = W.own_for(W.Given(dockets=dockets), [], (0, "A", 0, 0, 2), [0], [W.SS.Wire("A", "x", "B", "o0")])

    def e(v: str, r: str, reads: list) -> dict:
        return {"domain": "d" + r, "leaves": 2, "reads": reads, "root": r, "value": v}

    def ok(port: str, j: dict) -> bool:
        return W.window_read_ok(EX_RANKS, W.Index(ws), dockets, own, True, W.SS.Part("A", 0, 2), 0, port, j)
    assert ok("x", e("B/o0", "b1", [0, 1])) and not ok("x", e("B/o0", "b1", [1, 0])) and not ok("x", e("A/y", "a0", [0, 1]))
    assert ok("y", e("A/y", "a0", [0, 1])) and not ok("y", e("A/y", "a0", [1, 0])) and not ok("y", e("B/o0", "b1", [0, 1]))


def test_the_statements_are_the_sampled_members_by_position():
    """`Spot.SecondWindow`'s `#guard` on `sessionFailures`."""
    sel = [{"at": a, "base": 0, "out": "o0", "port": "live", "radix": [1], "reader": "A", "step": [1], "writer": x}
           for x, a in (("F", 0), ("L", 1))]

    def stmt(t: str) -> dict:
        return {"draw": {}, "header": {}, "registered": None, "template": t}

    def codes(ss: list) -> list[str]:
        return [c for c, _ in W.session_failures(W.Given(dockets=()), EX_RANKS, EX_REGS, {"wires": sel},
                                                 [(0, "A", 0, 0, 2)], [("A", 1)], ss)]
    assert codes([stmt("A")]) == ["second-read"] and codes([]) == ["second-cover"]
    assert codes([stmt("B")]) == ["second-order"] and codes([stmt("A"), stmt("A")]) == ["second-read", "second-order"]


def test_the_index_meets_what_a_scan_meets():
    """`Index.meets` is the scan's wires of a port meeting some units, by first unit, and `Index.ports` its ports."""
    def w(t: str, p: str, a: int, n: int) -> dict:
        return {"reader": t, "port": p, "writer": "W", "out": "o0", "at": a, "radix": [n], "base": 0, "step": [1],
                "cut": [0, 1]}
    ws = [w("B", "x", 4, 2), w("A", "y", 0, 8), w("A", "x", 6, 1), w("A", "x", 0, 3), w("A", "x", 3, 2), w("A", "y", 2, 1)]
    ix = W.Index(ws)
    for t, p, lo, n in [("A", "x", 0, 1), ("A", "x", 2, 2), ("A", "x", 5, 1), ("A", "x", 7, 3), ("A", "y", 3, 1),
                        ("A", "y", 9, 1), ("B", "x", 0, 9), ("C", "x", 0, 9)]:
        scan = sorted((x for x in ws if x["reader"] == t and x["port"] == p and W._reads_in(x, t, lo, n)), key=lambda x: x["at"])
        assert ix.meets(t, p, lo, n) == scan
    assert ix.ports("A") == ["x", "y"] and ix.ports("B") == ["x"] and ix.ports("C") == []
