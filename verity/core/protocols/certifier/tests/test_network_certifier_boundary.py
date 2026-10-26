"""The boundary's layout and the verifier's Fits check (`boundary`): the record-grain check (`checked_refusal`) held to
`NetTiming.Layout.Fits` itself, its first conjunct taking the checked row (`prog_fits`, ``checked``), on each toy window
flattened to gates; and the verifier's `refusal`, which refuses a window that passes it until Lean's Fits takes it."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import pytest

from verity.core.protocols.certifier import boundary as BD
from verity.ml.definitions.certifier import ingress_scan as IS


@dataclass(frozen=True)
class Toy:
    """A toy template: its parameter rows and output rows ``(name, words, kind)``, each output bit one gate: ``dense`` the
    XOR of every input bit of the unit, ``last`` the NOT of one bit of its last parameter row, ``const`` bit 0 set."""

    tid: str
    rows: tuple[tuple[str, int], ...]
    outs: tuple[tuple[str, int, str], ...]


ING = Toy("Ing_v1", (("row", 1), ("aux", 1)), (("o0", 1, "dense"),))
#: an egress bucket: its state ``y`` and the warden's row, its frame; its verdict ``o0``
EGR = Toy("Egr_v1", (("y", 1), ("row", 1)), (("o0", 1, "dense"),))
#: the verifier's refusal of a window that passes the record-grain check
CHECKED = "Layout.Fits, egress gates are returned: Egr_v1's frame 'row' is its checked row"


def side(tid, frame, links, T, verdict=None):
    d = {"template": tid, "frame": frame, "links": links, "buckets": T, "slots": 1, "frame_bytes": 2}
    return d | ({"verdict": verdict} if verdict else {})


def record(toys, wires, LE, LI, T):
    """A `verity/window-circuit/v1` record (the fields `boundary` reads) of ``toys`` ``[(Toy, n)]``, one member each, with
    ``wires`` ``{(tid, port): writers}`` and the boundary of `EGR` (``LE`` links) and `ING` (``LI`` links, or none)."""
    members, base = [], 0
    for toy, n in toys:
        members.append({"template": toy.tid, "name": "", "n": n, "base": base, "rows": [list(r) for r in toy.rows],
                        "outs": [[o, w] for o, w, _ in toy.outs], "tables": [], "links": [], "link_units": [],
                        "wired": [[p, "o0", list(ws)] for (tid, p), ws in wires.items() if tid == toy.tid]})
        base += n
    b = {"format": BD.FORMAT, "egress": side(EGR.tid, "row", LE, T, "o0")}
    ext = [f"{EGR.tid}/row", f"{EGR.tid}/o0"]
    if LI:
        b["ingress"] = side(ING.tid, "row", LI, T)
        ext.append(f"{ING.tid}/row")
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
    lo = {m["template"]: min(mm["base"] for mm in doc["members"] if mm["template"] == m["template"]) for m in doc["members"]}
    T = b.horizon

    def egress(t, ell, s):
        u = lo[EGR.tid] + ell * T + t
        return out_at[u]["o0"][0], ins[u]["row"][16 * s:16 * (s + 1)]

    def ingress(t):
        return [g for ell in range(b.ingress.links) for g in ins[lo[ING.tid] + ell * T + t]["row"]] if b.ingress else []

    return gates, outs, egress, ingress, b.egress.links, b.egress.slots, T


def fits(doc, toys):
    """`Layout.Fits` with its first conjunct taking the checked row; Lean's as it stands refuses every such window."""
    flat = flatten(doc, toys)
    assert not BD.prog_fits(*flat)
    return BD.prog_fits(*flat, checked=True)


def accepted(doc):
    """The record-grain check passes, and the verifier refuses only by Fits' first conjunct as Lean states it."""
    why = BD.refusal(doc)
    assert BD.checked_refusal(doc) is None and why.startswith(CHECKED), why
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
    """On random windows (seeds 0 … 299), `checked_refusal` accepting implies `Layout.Fits` (``checked``) at the gate
    grain, and on windows of dense templates the two agree; both verdicts occur."""
    seen = {True: 0, False: 0}
    for seed in range(300):
        rng = random.Random(seed)
        T, LE, LI = rng.randint(1, 3), rng.randint(1, 2), rng.randint(0, 2)
        dense = seed % 2 == 0
        kind = (lambda: "dense") if dense else (lambda: rng.choice(["dense", "last"]))
        ing = Toy(ING.tid, ING.rows, (("o0", 1, kind()),))
        mids = [Toy(f"Mid{j}_v1", (("x", 1), ("z", 1)), (("o0", 1, kind()),)) for j in range(rng.randint(0, 2))]
        egr = Toy(EGR.tid, EGR.rows, (("o0", 1, kind()),))
        toys = ([(ing, LI * T)] if LI else []) + [(t, rng.randint(1, 3)) for t in mids] + [(egr, LE * T)]
        total = sum(n for _, n in toys)
        wires = {}
        for toy, n in toys:
            for p, _ in toy.rows:
                if (toy.tid, p) not in {(ING.tid, "row"), (EGR.tid, "row")} and rng.random() < 0.7:
                    wires[(toy.tid, p)] = [rng.randrange(total) for _ in range(n)]
        doc = record(toys, wires, LE, LI, T)
        ok, gate = BD.checked_refusal(doc) is None, fits(doc, toys)
        assert not ok or accepted(doc)
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


def test_prog_fits_is_layout_fits_on_its_three_conjuncts():
    eg = lambda gs: (lambda t, ell, s: gs[t])  # noqa: E731
    gates = [("input", ()), ("input", ()), ("xor", (0, 1)), ("not", (3,)), ("and", (1,))]
    assert BD.prog_fits(gates, [2, 4], eg({0: (4, []), 1: (2, [])}), lambda t: [[1], [0]][t], 1, 1, 2)
    # step 0's slot reads step 1's ingress gate 0
    assert not BD.prog_fits(gates, [2, 4], eg({0: (2, []), 1: (4, [])}), lambda t: [[1], [0]][t], 1, 1, 2)
    # reflexive: a returned input gate is step 0's presence and step 1's ingress
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
    # the checked row: a frame gate that is an input fits only ``checked``; a presence gate is returned either way, and a
    # frame gate that is neither returned nor an input fits neither way
    assert not BD.prog_fits(gates, [2], eg({0: (2, [0])}), lambda t: [], 1, 1, 1)
    assert BD.prog_fits(gates, [2], eg({0: (2, [0])}), lambda t: [], 1, 1, 1, checked=True)
    assert not BD.prog_fits(gates, [2], eg({0: (0, [])}), lambda t: [], 1, 1, 1, checked=True)
    assert not BD.prog_fits(gates, [2], eg({0: (2, [4])}), lambda t: [], 1, 1, 1, checked=True)
    # reflexive: a frame input gate of step 0 is step 1's ingress
    assert not BD.prog_fits(gates, [2, 4], eg({0: (4, [0]), 1: (2, [])}), lambda t: [[1], [0]][t], 1, 1, 2, checked=True)


def test_the_unit_of_a_bucket_is_its_templates_unit_link_times_T_plus_t():
    toys, wires = identity(3, 2, 1)
    doc = record(toys, wires, 2, 1, 3)
    b = BD.parse(doc)
    assert [b.unit(doc, "egress", ell, t) for ell in range(2) for t in range(3)] == [3, 4, 5, 6, 7, 8]
    assert [b.unit(doc, "ingress", 0, t) for t in range(3)] == [0, 1, 2]
    with pytest.raises(ValueError, match=r"egress has links \[0, 2\) and buckets \[0, 3\), not \(2, 0\)"):
        b.unit(doc, "egress", 2, 0)


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
    (lambda d: _bd(d).update(format="verity/boundary/v1"), "a boundary is verity/boundary/v2"),
    (lambda d: _bd(d).pop("egress"), "a boundary is verity/boundary/v2 with an egress side"),
    (lambda d: _bd(d)["egress"].pop("verdict"), "boundary egress: the fields"),
    (lambda d: _bd(d)["egress"].update(presence="o0"), "boundary egress: the fields"),
    (lambda d: _bd(d)["ingress"].update(verdict="o0"), "boundary ingress: the fields"),
    (lambda d: _bd(d)["egress"].update(links=0), "are positive integers"),
    (lambda d: _bd(d)["egress"].update(links=True), "are positive integers"),
    (lambda d: _bd(d)["egress"].update(frame_bytes=3), "a cell is whole u16 words, not 3 bytes"),
    (lambda d: _bd(d)["egress"].update(verdict="row"), "the verdict row is not the frame row"),
    (lambda d: _bd(d)["ingress"].update(buckets=3), "one step a bucket on both sides, so one T"),
    (lambda d: d["external"].pop(), "are not the boundary's rows"),
    (lambda d: d["external"].append("Ing_v1/aux"), "are not the boundary's rows"),
    (_rename("egress", template="Nope_v1"), "boundary egress: Nope_v1 is no template of the circuit"),
    (lambda d: _bd(d)["egress"].update(links=2), "boundary egress: Egr_v1 has 2 units, not links·buckets = 2·2"),
    (_rename("egress", verdict="o2"), "Layout.Fits, egress gates are committed: Egr_v1's verdict 'o2' is not an output row "
                                      "of one u16 word (a unit's outputs are at most 16 bits)"),
    (lambda d: d["members"][1].update(outs=[["o0", 2]]),
     "Layout.Fits, egress gates are committed: Egr_v1's verdict 'o0' is not an output row of one u16 word"),
    (lambda d: (d["members"][1].update(outs=[["o0", 1], ["o1", 1]]), _rename("egress", frame="o1")(d)),
     "Layout.Fits, egress gates are committed: Egr_v1's 'o1' is not a parameter row of 1 words"),
    (_rename("egress", frame="nope"), "Layout.Fits, egress gates are committed: Egr_v1's 'nope' is not a parameter row of "
                                      "1 words"),
    (lambda d: _bd(d)["egress"].update(slots=2), "Layout.Fits, egress gates are committed: Egr_v1's 'row' is not a parameter "
                                                 "row of 2 words"),
    (_rename("egress", frame="y"), "Layout.Fits, egress gates are committed: Egr_v1's 'y' is wired from another unit's output"),
    (lambda d: d["members"][1].update(tables=[["x", 1]], links=[[0, 0]], link_units=[2]),
     "Layout.Fits, egress gates are committed: Egr_v1 reads row tables, and an egress frame is its own unit's row"),
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


def test_the_scans_link_window_passes_the_check_and_waits_on_lean():
    """A link-window in the composed scan's form at w63's class (`benchmarks/network_traces/scan_zk.py`): T ingress units
    (`ingress_scan.IngressScanBucket`, the warden's ingress row its frame) then T egress units (`egress.EgressBucket`, the
    warden's egress row its frame, its u16 verdict), every other row a registered read, so no wire. The record-grain check
    passes it, and the verifier refuses it by Fits' first conjunct as Lean states it, by name."""
    cls, T = IS.Class(B=512, r=4, lanes=4, C=47), 32
    sh, state, prog, view = cls.egress, IS.state_bits(cls) // 16, IS.prog_bits(cls) // 16, IS.view_bits(cls) // 16
    assert (state, prog, view, sh.state_words) == (9345, 280, 302, 302)
    units = {"IngressScanBucket": [["row", cls.S * cls.B // 2], ["sin", state], ["prog", prog], ["sout", state],
                                   ["view", view]],
             "EgressBucket": [["state", sh.state_words], ["row", sh.R * sh.B // 2]]}
    members = [{"template": tid, "name": "", "n": T, "base": i * T, "rows": rows, "outs": [["o0", 1]], "tables": [],
                "links": [], "link_units": [], "wired": []} for i, (tid, rows) in enumerate(units.items())]
    b = BD.Boundary(BD.Side("EgressBucket", "row", 1, T, sh.R, sh.B, "o0"),
                    BD.Side("IngressScanBucket", "row", 1, T, cls.S, cls.B))
    doc = {"format": BD.CIRCUIT, "templates": [[tid, T] for tid in units], "members": members,
           "external": sorted(b.ports()), "boundary": b.record()}
    assert BD.checked_refusal(doc) is None
    assert BD.refusal(doc) == (
        "Layout.Fits, egress gates are returned: EgressBucket's frame 'row' is its checked row, an input its verdict 'o0' "
        "holds to the cells it renders, and NetTiming.Layout.Fits returns every slot gate (C.outs), which a frame of R·B = "
        "16·512 bytes can't be (a unit's outputs are at most 16 bits): refused until Layout.Fits takes a frame gate that is "
        "an input (prog_fits, checked)")


def test_parse_reads_only_the_boundary():
    toys, wires = identity(2, 1, 1)
    doc = record(toys, wires, 1, 1, 2)
    b = BD.parse(copy.deepcopy(doc))
    assert b.egress == BD.Side("Egr_v1", "row", 1, 2, 1, 2, "o0") and b.ingress == BD.Side("Ing_v1", "row", 1, 2, 1, 2)
    assert b.horizon == 2 and b.ports() == {"Egr_v1/row", "Egr_v1/o0", "Ing_v1/row"}
    assert b.record() == doc["boundary"] and BD.parse({"boundary": b.record()}) == b
    egress_only = BD.Boundary(b.egress)
    assert egress_only.record() == {"format": BD.FORMAT, "egress": doc["boundary"]["egress"]}
    assert BD.parse({"format": BD.CIRCUIT}) is None
