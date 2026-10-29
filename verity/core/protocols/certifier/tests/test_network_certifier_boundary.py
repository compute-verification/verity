"""The boundary's layout and the verifier's Fits check (`boundary`): the record-grain check (`refusal`) held to
`NetTiming.Layout.Fits` itself (`prog_fits`) on each toy window flattened to gates, its egress frame an output row the
unit computes and returns; a frame that is a parameter row, an input, is refused by name."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import pytest

from verity.core.protocols.certifier import boundary as BD, egress as EG, ingress_scan as IS
from verity.ml.circuits.defs import bind


@dataclass(frozen=True)
class Toy:
    """A toy template: its parameter rows and output rows ``(name, words, kind)``, each output bit one gate: ``dense`` the
    XOR of every input bit of the unit, ``last`` the NOT of one bit of its last parameter row, ``const`` bit 0 set."""

    tid: str
    rows: tuple[tuple[str, int], ...]
    outs: tuple[tuple[str, int, str], ...]


ING = Toy("Ing_v1", (("row", 1), ("aux", 1)), (("o0", 1, "dense"),))
#: an ingress side's first bucket, its frame ``row`` too
FIRST = Toy("First_v1", (("row", 1), ("aux", 1)), (("o0", 1, "dense"),))
#: an egress bucket: its state ``y``; its verdict ``o0`` and its frame ``o1``, the row it computes
EGR = Toy("Egr_v1", (("y", 1),), (("o0", 1, "dense"), ("o1", 1, "dense")))


def side(tid, frame, links, T, verdict=None, first=None):
    d = {"template": tid, "frame": frame, "links": links, "buckets": T, "slots": 1, "frame_bytes": 2}
    return d | ({"verdict": verdict} if verdict else {}) | ({"first": first} if first else {})


def record(toys, wires, LE, LI, T, first=False):
    """A `verity/window-circuit/v1` record (the fields `boundary` reads) of ``toys`` ``[(Toy, n)]``, one member each, with
    ``wires`` ``{(tid, port): writers}`` and the boundary of `EGR` (``LE`` links) and `ING` (``LI`` links, or none; with
    ``first``, `FIRST` holds each link's bucket 0)."""
    members, base = [], 0
    for toy, n in toys:
        members.append({"template": toy.tid, "name": "", "n": n, "base": base, "rows": [list(r) for r in toy.rows],
                        "outs": [[o, w] for o, w, _ in toy.outs], "tables": [], "links": [], "link_units": [],
                        "wired": [[p, "o0", list(ws)] for (tid, p), ws in wires.items() if tid == toy.tid]})
        base += n
    b = {"format": BD.FORMAT, "egress": side(EGR.tid, "o1", LE, T, "o0")}
    ext = [f"{EGR.tid}/o1", f"{EGR.tid}/o0"]
    if LI:
        b["ingress"] = side(ING.tid, "row", LI, T, first=FIRST.tid if first else None)
        ext += [f"{ING.tid}/row"] + ([f"{FIRST.tid}/row"] if first else [])
    return {"format": BD.CIRCUIT, "templates": [[t.tid, n] for t, n in toys], "members": members, "external": ext,
            "boundary": b}


def flatten(doc, toys):
    """The record as one `Claims.Sampling.Prog`: each unwired parameter bit an input gate, each output bit one gate, a wired
    row's bits its writer's output bits; every output bit returned. -> (gates, outs, egress, ingress, L, S, k)."""
    by = {t.tid: t for t, _ in toys}
    units = [(m, i) for m in doc["members"] for i in range(m["n"])]
    wired = {(m["template"], p): ws for m in doc["members"] for p, _o, ws in m["wired"]}
    n_in = sum(16 * w for m, _ in units for p, w in by[m["template"]].rows if (m["template"], p) not in wired)
    gates, ins, at = [], [], 0
    for m, _ in units:
        mine = {}
        for p, w in by[m["template"]].rows:
            if (m["template"], p) not in wired:
                mine[p] = list(range(at, at + 16 * w))
                at += 16 * w
        ins.append(mine)
    gates = [("input", ())] * n_in
    out_at, o = [], n_in
    for m, _ in units:
        out_at.append({name: list(range(o + 16 * sum(w for _, w, _ in by[m["template"]].outs[:j]),
                                        o + 16 * sum(w for _, w, _ in by[m["template"]].outs[:j + 1])))
                       for j, (name, _, _) in enumerate(by[m["template"]].outs)})
        o += 16 * sum(w for _, w, _ in by[m["template"]].outs)
    for u, (m, i) in enumerate(units):
        toy = by[m["template"]]
        bits = {p: ins[u][p] if p in ins[u] else out_at[wired[(toy.tid, p)][i]]["o0"] for p, _ in toy.rows}
        every = [g for p, _ in toy.rows for g in bits[p]]
        for name, _w, kind in toy.outs:
            for j, _g in enumerate(out_at[u][name]):
                last = bits[toy.rows[-1][0]]
                gates.append({"dense": ("xor", tuple(every)), "last": ("not", (last[j % len(last)],)),
                              "const": ("const", ())}[kind])
    outs = [g for d in out_at for gs in d.values() for g in gs]
    b = BD.parse(doc)
    T = b.horizon

    def egress(t, ell, s):
        u = b.unit(doc, "egress", ell, t)
        return out_at[u]["o0"][0], out_at[u]["o1"][16 * s:16 * (s + 1)]

    def ingress(t):
        return [g for ell in range(b.ingress.links) for g in ins[b.unit(doc, "ingress", ell, t)]["row"]] if b.ingress else []

    return gates, outs, egress, ingress, b.egress.links, b.egress.slots, T


def fits(doc, toys):
    """`Layout.Fits` on the flattened record."""
    return BD.prog_fits(*flatten(doc, toys))


def accepted(doc):
    """The record-grain check passes."""
    why = BD.refusal(doc)
    assert why is None, why
    return True


def identity(T, LE, LI):
    """Egress bucket t of link ℓ reads ingress bucket t of link ℓ mod LI."""
    toys = [(ING, LI * T), (EGR, LE * T)]
    return toys, {(EGR.tid, "y"): [(ell % LI) * T + t for ell in range(LE) for t in range(T)]}


def test_a_window_whose_egress_reads_only_earlier_ingress_fits():
    toys, wires = identity(3, 2, 2)
    doc = record(toys, wires, 2, 2, 3)
    assert accepted(doc)
    assert fits(doc, toys)
    earlier = [0, 0, 1, 3, 3, 5]  # egress (ℓ, t) in order reads ingress unit ℓ·3 + t' at t' ≤ t
    doc = record(toys, {(EGR.tid, "y"): earlier}, 2, 2, 3)
    assert accepted(doc) and fits(doc, toys)


def test_an_egress_bucket_that_reads_a_later_buckets_ingress_is_refused_by_the_causal_conjunct():
    """An egress bucket that reads every request's ids, so a later bucket's ingress (a window not in the scan's form):
    refused, at the egress bucket and the ingress bucket it reads."""
    T = 3
    toys = [(ING, T), (Toy("Mid_v1", (("x", 1),), (("o0", 1, "dense"),)), 2), (EGR, T)]
    wires = {("Mid_v1", "x"): [2, 1], (EGR.tid, "y"): [3, 4, 3]}
    doc = record(toys, wires, 1, 1, T)
    why = BD.refusal(doc)
    assert why is not None and why.startswith("Layout.Fits, causal: egress bucket 0 of link 0 (unit 5) depends on ingress "
                                              "bucket 2 of link 0 (unit 2)"), why
    assert not fits(doc, toys)


def test_the_unit_grain_contains_depends_and_is_exact_on_dense_templates():
    """On random windows (seeds 0 … 299), `refusal` accepting implies `Layout.Fits` at the gate grain, and on windows of
    dense templates the two agree; both verdicts occur. About half the windows with ingress over two buckets or more hold
    each link's bucket 0 in a first template."""
    seen = {True: 0, False: 0}
    for seed in range(300):
        rng = random.Random(seed)
        T, LE, LI = rng.randint(1, 3), rng.randint(1, 2), rng.randint(0, 2)
        first = LI > 0 and T > 1 and rng.random() < 0.5
        dense = seed % 2 == 0
        kind = (lambda: "dense") if dense else (lambda: rng.choice(["dense", "last"]))
        ing = Toy(ING.tid, ING.rows, (("o0", 1, kind()),))
        fst = Toy(FIRST.tid, FIRST.rows, (("o0", 1, kind()),))
        mids = [Toy(f"Mid{j}_v1", (("x", 1), ("z", 1)), (("o0", 1, kind()),)) for j in range(rng.randint(0, 2))]
        egr = Toy(EGR.tid, EGR.rows, (("o0", 1, kind()), ("o1", 1, kind())))
        toys = (([(fst, LI), (ing, LI * (T - 1))] if first else [(ing, LI * T)] if LI else [])
                + [(t, rng.randint(1, 3)) for t in mids] + [(egr, LE * T)])
        total = sum(n for _, n in toys)
        wires = {}
        for toy, n in toys:
            for p, _ in toy.rows:
                if (toy.tid, p) not in {(ING.tid, "row"), (FIRST.tid, "row")} and rng.random() < 0.7:
                    wires[(toy.tid, p)] = [rng.randrange(total) for _ in range(n)]
        doc = record(toys, wires, LE, LI, T, first)
        ok, gate = BD.refusal(doc) is None, fits(doc, toys)
        assert gate or not ok, (seed, BD.refusal(doc))
        if dense:
            assert ok == gate, (seed, BD.refusal(doc))
        seen[ok] += 1
    assert min(seen.values()) > 30, seen


def test_a_record_that_fits_only_at_the_gate_grain_is_refused():
    """The ingress unit's output reads only its ``aux`` row, an input that isn't ingress, so egress bucket 0 reading bucket
    1's ingress unit depends on no ingress gate; the record's grain can't see that, and refuses."""
    ing = Toy(ING.tid, ING.rows, (("o0", 1, "last"),))
    toys = [(ing, 2), (EGR, 2)]
    doc = record(toys, {(EGR.tid, "y"): [1, 1]}, 1, 1, 2)
    assert fits(doc, toys)
    assert BD.refusal(doc).startswith("Layout.Fits, causal: egress bucket 0 of link 0 (unit 2) depends on ingress bucket 1")


def test_prog_fits_is_layout_fits_on_its_conjuncts():
    eg = lambda gs: (lambda t, ell, s: gs[t])  # noqa: E731
    gates = [("input", ()), ("input", ()), ("xor", (0, 1)), ("not", (3,)), ("and", (1,))]
    assert BD.prog_fits(gates, [2, 4], eg({0: (4, []), 1: (2, [])}), lambda t: [[1], [0]][t], 1, 1, 2)
    # step 0's slot reads step 1's ingress gate 0
    assert not BD.prog_fits(gates, [2, 4], eg({0: (2, []), 1: (4, [])}), lambda t: [[1], [0]][t], 1, 1, 2)
    # a returned input gate as a slot gate, with no ingress, and as step 1's ingress too
    assert not BD.prog_fits(gates, [0, 2], eg({0: (0, [])}), lambda t: [], 1, 1, 1)
    assert not BD.prog_fits(gates, [0, 2], eg({0: (2, [0])}), lambda t: [], 1, 1, 1)
    assert not BD.prog_fits(gates, [0, 2], eg({0: (0, []), 1: (2, [])}), lambda t: [[1], [0]][t], 1, 1, 2)
    # past the horizon nothing is read
    assert BD.prog_fits(gates, [2, 4], eg({0: (2, []), 1: (4, [])}), lambda t: [[], [0]][t], 1, 1, 1)
    # a slot gate that isn't returned, an ingress gate that isn't an input
    assert not BD.prog_fits(gates, [2], eg({0: (2, [4])}), lambda t: [], 1, 1, 1)
    assert not BD.prog_fits(gates, [2], eg({0: (2, [])}), lambda t: [3], 1, 1, 1)
    # a read past the gates reaches nothing, a read ahead of the reader counts (`Depends` reads `C.gates[b]?`, topological
    # or not)
    assert BD.prog_fits(gates + [("xor", (9, 4))], [2, 5], eg({0: (5, []), 1: (2, [])}), lambda t: [[], [0]][t], 1, 1, 2)
    assert not BD.prog_fits(gates + [("xor", (9, 2))], [2, 5], eg({0: (5, []), 1: (2, [])}), lambda t: [[], [0]][t], 1, 1, 2)
    assert not BD.prog_fits([("not", (1,)), ("input", ())], [0], eg({0: (0, []), 1: (0, [])}), lambda t: [[], [1]][t], 1, 1, 2)
    # a frame gate the program computes and returns fits; one that is an input doesn't, returned or not
    assert BD.prog_fits(gates, [2, 4], eg({0: (2, [4])}), lambda t: [], 1, 1, 1)
    assert not BD.prog_fits(gates, [2], eg({0: (2, [0])}), lambda t: [], 1, 1, 1)


def test_the_unit_of_a_bucket_is_its_templates_unit_link_times_T_plus_t():
    toys, wires = identity(3, 2, 1)
    doc = record(toys, wires, 2, 1, 3)
    b = BD.parse(doc)
    assert [b.unit(doc, "egress", ell, t) for ell in range(2) for t in range(3)] == [3, 4, 5, 6, 7, 8]
    assert [b.unit(doc, "ingress", 0, t) for t in range(3)] == [0, 1, 2]
    with pytest.raises(ValueError, match=r"egress has links \[0, 2\) and buckets \[0, 3\), not \(2, 0\)"):
        b.unit(doc, "egress", 2, 0)
    # with a first template: bucket 0 of link ℓ its unit ℓ, bucket t ≥ 1 the other's unit ℓ·(T − 1) + t − 1
    toys = [(FIRST, 2), (ING, 4), (EGR, 6)]
    doc = record(toys, {(EGR.tid, "y"): [0, 2, 3, 1, 4, 5]}, 2, 2, 3, first=True)
    b = BD.parse(doc)
    assert [b.unit(doc, "ingress", ell, t) for ell in range(2) for t in range(3)] == [0, 2, 3, 1, 4, 5]
    assert [b.unit(doc, "egress", ell, t) for ell in range(2) for t in range(3)] == [6, 7, 8, 9, 10, 11]
    assert accepted(doc) and fits(doc, toys)
    doc["members"][2]["wired"][0][2][0] = 2
    assert BD.refusal(doc).startswith("Layout.Fits, causal: egress bucket 0 of link 0 (unit 6) depends on ingress bucket 1 "
                                      "of link 0 (unit 2)")
    assert not fits(doc, toys)


def _refused(edit, needle):
    toys, wires = identity(2, 1, 1)
    doc = record(toys, wires, 1, 1, 2)
    assert accepted(doc)
    edit(doc)
    why = BD.refusal(doc)
    assert why is not None and needle in why, why


def _bd(doc):
    return doc["boundary"]


def _rename(name, **kw):
    """Set the boundary side ``name``'s fields ``kw`` and keep ``external`` the boundary's rows."""
    def edit(doc):
        _bd(doc)[name].update(kw)
        doc["external"] = sorted(BD.parse(doc).ports())
    return edit


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.pop("boundary"), "the circuit record declares none (NetTiming.Layout)"),
    (lambda d: d.update(format="verity/window-circuit/v0"), "a circuit record is verity/window-circuit/v1"),
    (lambda d: _bd(d).update(format="verity/boundary/v1"), "a boundary is verity/boundary/v3"),
    (lambda d: _bd(d).update(format="verity/boundary/v2"), "a boundary is verity/boundary/v3"),
    (lambda d: _bd(d).pop("egress"), "a boundary is verity/boundary/v3 with an egress side"),
    (lambda d: _bd(d)["egress"].pop("verdict"), "boundary egress: the fields"),
    (lambda d: _bd(d)["egress"].update(presence="o0"), "boundary egress: the fields"),
    (lambda d: _bd(d)["ingress"].update(verdict="o0"), "boundary ingress: the fields"),
    (lambda d: _bd(d)["egress"].update(first="Ing_v1"), "boundary egress: the fields"),
    (lambda d: _bd(d)["ingress"].update(first="Ing_v1"), "boundary ingress: a first bucket's template is another than"),
    (lambda d: _bd(d)["ingress"].update(first=""), "are named by nonempty strings"),
    (lambda d: _bd(d)["egress"].update(links=0), "are positive integers"),
    (lambda d: _bd(d)["egress"].update(links=True), "are positive integers"),
    (lambda d: _bd(d)["egress"].update(frame_bytes=3), "a cell is whole u16 words, not 3 bytes"),
    (lambda d: _bd(d)["egress"].update(verdict="o1"), "the verdict row is not the frame row"),
    (lambda d: _bd(d)["ingress"].update(buckets=3), "one step a bucket on both sides, so one T"),
    (lambda d: d["external"].pop(), "are not the boundary's rows"),
    (lambda d: d["external"].append("Ing_v1/aux"), "are not the boundary's rows"),
    (_rename("egress", template="Nope_v1"), "boundary egress: Nope_v1 is no template of the circuit"),
    (_rename("ingress", first="First_v1"), "boundary ingress: First_v1 is no template of the circuit"),
    (lambda d: _bd(d)["egress"].update(links=2), "boundary egress: Egr_v1 has 2 units, not links·buckets = 2·2"),
    (_rename("egress", verdict="o2"), "Layout.Fits, egress gates are returned: Egr_v1's verdict 'o2' is not an output row "
                                      "of one u16 word (a unit's public output is at most 16 bits)"),
    (lambda d: d["members"][1].update(outs=[["o0", 2], ["o1", 1]]),
     "Layout.Fits, egress gates are returned: Egr_v1's verdict 'o0' is not an output row of one u16 word"),
    (_rename("egress", frame="nope"), "Layout.Fits, egress gates are returned: Egr_v1's frame 'nope' is not an output row of "
                                      "1 words"),
    (lambda d: _bd(d)["egress"].update(slots=2), "Layout.Fits, egress gates are returned: Egr_v1's frame 'o1' is not an "
                                                 "output row of 2 words"),
    (_rename("egress", frame="y"), "Layout.Fits, egress gates are returned: Egr_v1's frame 'y' is a parameter row, an input"),
    (lambda d: (d["members"][1].update(rows=[["y", 1], ["row", 1]], outs=[["o0", 1]]), _rename("egress", frame="row")(d)),
     "Layout.Fits, egress gates are returned: Egr_v1's frame 'row' is a parameter row, an input, which the Program reads "
     "from the advice and so bounds nothing"),
    (_rename("ingress", frame="o0"), "Layout.Fits, ingress gates are inputs: Ing_v1's 'o0' is not a parameter row of 1 words"),
    (lambda d: d["members"][0]["wired"].append(["row", "o0", [2, 3]]),
     "Layout.Fits, ingress gates are inputs: Ing_v1's 'row' is wired from another unit's output"),
    (lambda d: d["members"][0].update(tables=[["x", 1]], links=[[0, 0]], link_units=[2]),
     "Layout.Fits, ingress gates are inputs: Ing_v1 reads row tables"),
    (lambda d: d["members"][1]["wired"][0][2].__setitem__(0, 9), "a wire is one writer in [0, 4) a unit"),
    (lambda d: d["members"][1].update(base=3), "Egr_v1: a member's units start at 3, not 2"),
    (lambda d: d["members"][1]["wired"][0].__setitem__(2, [1, 1]),
     "Layout.Fits, causal: egress bucket 0 of link 0 (unit 2) depends on ingress bucket 1 of link 0 (unit 1)"),
])
def test_a_record_that_doesnt_fit_is_refused_by_name(edit, needle):
    _refused(edit, needle)


def test_an_egress_only_window_has_no_ingress_gates():
    toys = [(Toy("Src_v1", (("a", 1),), (("o0", 1, "dense"),)), 1), (EGR, 2)]
    doc = record(toys, {(EGR.tid, "y"): [0, 0]}, 1, 0, 2)
    assert "ingress" not in doc["boundary"] and accepted(doc) and fits(doc, toys)


def test_an_egress_unit_that_reads_a_row_table_fits():
    """Its frame is a row it computes, so what fills its inputs moves no egress gate off `C.outs`; a row table's rows are
    inputs no unit writes, so no ingress either."""
    toys, _ = identity(2, 1, 1)
    doc = record(toys, {}, 1, 1, 2)
    doc["members"][1].update(tables=[["y", 1]], links=[[0, 0]], link_units=[2])
    assert accepted(doc) and fits(doc, toys)


def _words(t) -> int:
    return -(-sum(t.leaf_widths()) // 16)


@pytest.mark.parametrize("G", [1, 2])
def test_the_scans_link_windows_fit(G):
    """A group of G link-windows in the composed scan's generator form at w63's class (`benchmarks/network_traces/
    scan_zk.py`), one link of G·T buckets: `IngressScanFirst` (bucket 0), G·T − 1 `IngressScanBucket` and G·T
    `EgressBucket` units, each member's rows and outputs its Definition's signature's. The ingress frame is the warden's
    ingress row; the egress frame is the row `EgressBucket` computes (o1), beside its u16 verdict (o0); ingress unit t's
    state output (o1) is wired to unit t + 1's ``sin`` and its view output (o2) to egress unit t's ``state``. The check
    passes it, and the checker form, its egress frame a parameter row, is refused by name."""
    cls, T = IS.Class(B=512, r=4, lanes=4, C=47), 32
    n, sh = G * T, cls.egress
    st = {k: getattr(cls, k) for k in IS.STATICS}
    defs = {"IngressScanFirst": bind(IS.IngressScanFirst, **st), "IngressScanBucket": bind(IS.IngressScanBucket, **st),
            "EgressBucket": bind(EG.EgressBucket, B=sh.B, R=sh.R, r=sh.r, NR=sh.NR, K=sh.K, IB=sh.IB, D=sh.D)}
    counts = {"IngressScanFirst": 1, "IngressScanBucket": n - 1, "EgressBucket": n}
    wired = {"IngressScanBucket": [["sin", "o1", list(range(n - 1))]], "EgressBucket": [["state", "o2", list(range(n))]]}
    members, base = [], 0
    for tid, d in defs.items():
        params, ret = d.signature()
        members.append({"template": tid, "name": "", "n": counts[tid], "base": base,
                        "rows": [[p, _words(t)] for p, t in params],
                        "outs": [[f"o{j}", _words(f)] for j, f in enumerate(ret.fields)], "tables": [], "links": [],
                        "link_units": [], "wired": wired.get(tid, [])})
        base += counts[tid]
    rows = {m["template"]: dict(map(tuple, m["rows"])) for m in members}
    outs = {m["template"]: dict(map(tuple, m["outs"])) for m in members}
    entries = {f"prog{j}": IS.entry_bits(cls) // 16 for j in range(cls.lanes)}
    assert entries == {"prog0": 71, "prog1": 71, "prog2": 71, "prog3": 71}
    assert rows["IngressScanBucket"] == {"row": cls.S * cls.B // 2, "sin": 9353, **entries}
    assert rows["IngressScanFirst"] == {"row": cls.S * cls.B // 2, **entries}
    assert outs["IngressScanBucket"] == outs["IngressScanFirst"] == {"o0": 1, "o1": 9353, "o2": sh.state_words}
    assert rows["EgressBucket"] == {"state": sh.state_words} and outs["EgressBucket"] == {"o0": 1, "o1": sh.R * sh.B // 2}
    b = BD.Boundary(BD.Side("EgressBucket", "o1", 1, n, sh.R, sh.B, "o0"),
                    BD.Side("IngressScanBucket", "row", 1, n, cls.S, cls.B, first="IngressScanFirst"))
    doc = {"format": BD.CIRCUIT, "templates": [[tid, counts[tid]] for tid in defs], "members": members,
           "external": sorted(b.ports()), "boundary": b.record()}
    assert BD.refusal(doc) is None
    assert [b.unit(doc, "ingress", 0, t) for t in range(n)] == list(range(n))
    assert [b.unit(doc, "egress", 0, t) for t in range(n)] == list(range(n, 2 * n))
    checker = copy.deepcopy(doc)
    checker["members"][2].update(rows=[["state", sh.state_words], ["row", sh.R * sh.B // 2]], outs=[["o0", 1]])
    checker["boundary"]["egress"]["frame"] = "row"
    checker["external"] = sorted(BD.parse(checker).ports())
    assert BD.refusal(checker) == (
        "Layout.Fits, egress gates are returned: EgressBucket's frame 'row' is a parameter row, an input, which the Program "
        "reads from the advice and so bounds nothing: NetTiming.Layout.Fits takes every slot gate to be one C returns "
        "(C.outs) and not an input, a frame the unit computes (prog_fits)")


def test_parse_reads_only_the_boundary():
    toys, wires = identity(2, 1, 1)
    doc = record(toys, wires, 1, 1, 2)
    b = BD.parse(copy.deepcopy(doc))
    assert b.egress == BD.Side("Egr_v1", "o1", 1, 2, 1, 2, "o0") and b.ingress == BD.Side("Ing_v1", "row", 1, 2, 1, 2)
    assert b.horizon == 2 and b.ports() == {"Egr_v1/o1", "Egr_v1/o0", "Ing_v1/row"}
    assert b.record() == doc["boundary"] and BD.parse({"boundary": b.record()}) == b
    egress_only = BD.Boundary(b.egress)
    assert egress_only.record() == {"format": BD.FORMAT, "egress": doc["boundary"]["egress"]}
    assert BD.parse({"format": BD.CIRCUIT}) is None
    doc = record([(FIRST, 1), (ING, 1), (EGR, 2)], {(EGR.tid, "y"): [0, 1]}, 1, 1, 2, first=True)
    b = BD.parse(doc)
    assert b.ingress == BD.Side("Ing_v1", "row", 1, 2, 1, 2, first="First_v1") and b.record() == doc["boundary"]
    assert b.ports() == {"Egr_v1/o1", "Egr_v1/o0", "Ing_v1/row", "First_v1/row"}
    assert b.ingress.counts() == {"First_v1": 1, "Ing_v1": 1}
