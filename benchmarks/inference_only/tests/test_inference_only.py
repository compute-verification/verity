"""`verity bench inference_only`: the served policy's file against the tree, a policy's number against the bound's own
count, the divisions on a small record, the report of a refused policy, and the served class under the value tie, which
prints only at a typed share and under its label."""
from __future__ import annotations

import dataclasses
import importlib.util
import json

import built as B
import policy as P
import pytest
import report
import tied as T
from rho import lean, min_cut
from verity_vllm.check.weights_of_record import PARAM
from verity_vllm.program.predict import block_counts as BC

from verity.commitments.windows import interface as IF

ROPE, SILU = "RoPEHead_v1{D=16}", "SiluMul_v2{I=8}"
NORM = 'RMSNormTriton_v5{N=8,EPS={"f64":"0x1.4f8b588e368f1p-17"}}'
#: the small record's class: 2 slots of 2 + 2 - 1 = 3 positions
CLASS = (2, 2, 2)


def small() -> IF.Circuit:
    """A record of three catalog templates over 2 slots of 3 steps, a unit a slot a step: the head (its inputs registered
    weights), the product reading its own head, and the norm reading the product two units back, which is its slot's at
    the step before (so its first step's units read a registered row outside θ instead, one row a slot)."""
    fns = {t: BC.resolve(t) for t in (ROPE, SILU, NORM)}
    r, s, n = (IF.descriptor_id(fn) for fn in fns.values())
    reads = (IF.Read(r, "x", f"{PARAM}/x", 0, (6,), 0, (0,)), IF.Read(r, "cs", f"{PARAM}/cs", 0, (6,), 0, (0,)),
             IF.Read(n, "w", f"{PARAM}/norm", 0, (2,), 0, (0,)), IF.Read(n, "x", "state", 0, (2,), 0, (8,)))
    return IF.Circuit(
        "test/small", tuple((fn, 6) for fn in fns.values()),
        wires=(IF.Wire(s, "gu", r, "o0", 0, (6,), 0, (1,)), IF.Wire(n, "x", s, "o0", 2, (4,), 0, (1,)),
               IF.Wire(n, "w", s, "o0", 2, (4,), 0, (1,))),
        registered=IF.Registered({f"{PARAM}/x": 16, f"{PARAM}/cs": 16, f"{PARAM}/norm": 8, "state": 16}, reads=reads),
        blocks=IF.Blocks(slots=2, steps=3, inference=tuple(IF.InferenceEntry(t, 0, (3, 2), ("step", "slot"), 0)
                                                           for t in (r, s, n))))


@pytest.fixture(autouse=True)
def no_out(monkeypatch) -> None:
    """The command writes its JSON to `$OUT` by default; a test names its own directory or none."""
    monkeypatch.delenv("OUT", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)


def reference(inst: dict) -> dict:
    """The count's Python reference, which the count's difftest holds the Lean to, as `built.lean_count`."""
    spec = importlib.util.spec_from_file_location("nci_reference", B.PACKAGE / "scripts" / "nci_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return {"verdict": ref.check_count_detail(inst), "cost": {}}


@pytest.fixture(scope="module")
def counted() -> tuple[dict, dict, dict]:
    doc, pins = B.record(small())
    pol = B.policy(doc, pins)
    return doc, pol, B.divisions(doc, pins, pol["gates"])


def test_the_served_file_is_what_the_tree_computes_of_its_class() -> None:
    """The stated policy's work is the class's fixed load priced as a served step, its model's shape is the declared
    one, and its k is the outside linears' outputs and the scan state row, with nothing left over."""
    import served_shares as SS
    import served_verdicts as SV

    p, sh = P.stated(), P.shape(P.stated().model)
    assert p.label == "served" and p.check["verdict"] == "not run"
    assert p.work == B.block_work(p.cls, B.MODEL)
    layers, linears = SV.DECLARED[B.MODEL]
    assert (sh.layers, [x[1:] for x in sh.linears]) == (layers, [x[1:] for x in linears])
    assert sh.head == (SS.MODELS[B.MODEL]["vocab"], SS.MODELS[B.MODEL]["hidden"])
    assert p.unexplained == 0 and p.outside == p.k - P.scan_state_bits(p.cls) > 0


def test_a_policy_file_that_misstates_its_work_is_refused(tmp_path) -> None:
    text = P.SERVED.read_text().replace("total = 36241670144", "total = 36241670145")
    assert text != P.SERVED.read_text()
    (tmp_path / "p.toml").write_text(text)
    with pytest.raises(ValueError, match="is not linears"):
        P.stated(tmp_path / "p.toml")
    (tmp_path / "q.toml").write_text(P.SERVED.read_text().replace("k = 422197408\n", ""))
    with pytest.raises(ValueError, match="has no 'k'"):
        P.stated(tmp_path / "q.toml")


@pytest.mark.parametrize("k", [145_568, 8_000_000, 422_197_408])
def test_the_number_is_the_bounds(k: int) -> None:
    """A policy's number is `min_cut.bound`'s slowdown on one block's work: where it is over 1, that many blocks of the
    policy finish exactly the block's work by the count (`min_cut.macs_at_most`, HE's right side), and where the count
    lets one block finish the work it is 1."""
    p, sh = dataclasses.replace(P.stated(), k=k), P.shape(P.stated().model)
    row = report.priced(p, sh)
    bd, N = row["bound"], p.work[P.TOTAL]
    assert (bd["N"], bd["size"], bd["O"]) == (N, p.c, N * sh.outputs * 16 / sh.macs)
    q = bd["beta_w"] * bd["beta_x"]
    if min_cut.macs_at_most(1, k, bd["O"], q, bd["b"]) >= N:
        assert row["value"] == 1 and k == P.stated().k
    else:
        assert row["value"] > 1
        assert min_cut.macs_at_most(row["value"], k, bd["O"], q, bd["b"]) == pytest.approx(N, rel=1e-9)
        assert row["value"] == pytest.approx(N / bd["macs_a_block"], rel=1e-12)


def test_the_divisions_count_a_block_gates_and_each_row_it_reads_once(counted) -> None:
    doc, pol, div = counted
    gates = [pol["gates"][t] for t, _ in doc["templates"]]
    assert [div[d]["blocks_a_session"] for d in B.DIVISIONS] == [1, 3, 9]
    # a session holds every template at each of its 3 steps, a position one unit of each, a proof block one unit
    assert [div[d]["c"] for d in B.DIVISIONS] == [3 * sum(gates), sum(gates), max(gates)] and pol["c"] == 3 * sum(gates)
    # a session reads its one state row (the norm's 8 words); the wires stay inside it
    assert div["session"]["k"] == pol["k"] == 8 * 16
    assert div["session"]["k_by_kind"] == [("registered state, read by RMSNormTriton_v5.x", 128)]
    # a position after the first reads the product of the step before, one row though two ports read it
    assert div["position"]["k"] == 8 * 16
    # the product's unit reads its head's 16 words from another proof block
    assert div["proof block"]["k"] == 16 * 16
    assert div["proof block"]["k_by_kind"] == [("RoPEHead_v1.o0, read by SiluMul_v2.gu", 256)]


def test_the_count_refuses_the_session_program_on_k(counted) -> None:
    """The count is given the session Definition with no θ and no wires, so it counts every input row of every unit,
    the weights too, and refuses the record's k."""
    _, pol, _ = counted
    check = B.judged(reference(B.instance(pol, pol["c"], pol["k"], 5)), pol["c"], pol["k"], 5)
    assert (check["verdict"], check["complies"], check["failed"]) == ("refused", False, "k")
    assert check["counted"]["c"] == pol["c"] and check["counted"]["k"] > pol["k"] and check["counted"]["n"] == 2
    assert B.judged(reference(B.instance(pol, pol["c"], check["counted"]["k"], 5)), pol["c"], 0, 5)["verdict"] == "accepted"
    few = B.judged(reference(B.instance(pol, pol["c"], check["counted"]["k"], 1)), pol["c"], 0, 1)
    assert (few["verdict"], few["failed"]) == ("refused", "n")


def test_the_report_puts_the_served_number_first_and_labels_a_refused_policy(capsys, tmp_path) -> None:
    """The command on the served file and a small built record: the served number is first and is the JSON's value, the
    built class is labelled as not served, its refusal prints the verdict and what failed, and its number is the
    bound's and is labelled as resting on the Python count."""
    doc = report.main(["--class", ",".join(map(str, CLASS)), "--out", str(tmp_path)], check=reference, circuit=small(),
                      commit=lambda: "0" * 40)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert json.loads(lines[-1]) == doc == json.loads((tmp_path / "inference_only.json").read_text())
    assert lines[2] == "AS SERVED: 1" and "outside the circuit" in lines[4] and "Not checked here" in lines[6]
    assert out.index("AS SERVED: 1") < out.index("The class this tree builds: NOT SERVED")
    assert B.EXECUTABLE[1:] == ("--count", "--detail") and B.RULE.endswith(".Count: CompliesAtMost c k n")
    assert f"{' '.join(B.EXECUTABLE)} ({B.RULE}): complies: false, k failed" in out
    assert "the registered Program has no θ and no dataflow" in out
    assert "the Lean counts c's Boolean gates a block exactly, and " in out
    served, built = doc["policies"]
    assert served["units"] == built["units"] == P.UNITS and f"{served['c']:,} Boolean gates a block" in out
    assert (doc["value"], doc["check"], doc["rests_on"]) == (served["value"], served["check"], served["rests_on"])
    assert doc["value"] == 1 and doc["check"]["verdict"] == "not run" and "run" not in doc and doc["commit"] == "0" * 40
    assert "nothing on this tree builds or checks" in doc["rests_on"]
    assert (built["label"], built["check"]["verdict"], built["check"]["failed"]) == ("built", "refused", "k")
    assert built["rests_on"].startswith("the Python count of what a block reads") and "does not establish" in built["rests_on"]
    assert built["rests_on"] in out
    sh = P.shape(B.MODEL)
    N = B.block_work(CLASS, B.MODEL)[P.TOTAL]
    want = min_cut.bound(built["c"], built["k"], built["n"], N=N, O=N * sh.outputs * 16 / sh.macs, size=built["c"])
    assert built["value"] == want.slowdown > 1 and f"number    {report._n(want.slowdown)} = max(1, N / limit)" in out
    assert [d["division"] for d in doc["divisions"]] == list(B.DIVISIONS) and doc["divisions"][0] == built
    assert [d["check"]["verdict"] for d in doc["divisions"]] == ["refused", "not run", "not run"]
    # each policy over each fleet and monitoring period: a share of the compute, and the blocks' training at most
    assert len(doc["periods"]) == 2 * 2 * 3 and {r["policy"] for r in doc["periods"]} == {"served", "built"}
    for r in doc["periods"]:
        of = served if r["policy"] == "served" else built
        assert r["share"] == pytest.approx(1 / of["value"])
        assert r["training_macs"] == pytest.approx(r["share"] * r["blocks"] * of["work"]["N"])
    for word in ("proved", "premise", "scaled", "the two thirds", "BF16", report.CALCULATOR):
        assert word in out


def test_stated_only_builds_nothing_and_another_division_is_the_built_number(capsys) -> None:
    def never(*_a, **_k):
        raise AssertionError("built")

    doc = report.main(["--stated-only"], check=never, commit=lambda: None)
    assert len(doc["policies"]) == 1 and doc["divisions"] == [] and "NOT SERVED" not in capsys.readouterr().out
    doc = report.main(["--class", ",".join(map(str, CLASS)), "--division", "proof block"], check=reference,
                      circuit=small(), commit=lambda: None)
    assert doc["policies"][1] == doc["divisions"][2] and doc["policies"][1]["division"] == "proof block"
    assert doc["policies"][1]["n"] == 9 * doc["divisions"][0]["n"]


def test_a_class_the_integration_refuses_is_an_error_that_says_why(monkeypatch, capsys) -> None:
    from verity_vllm.program.predict import window_circuit
    from verity_vllm.program.predict.common import Unexpressible

    def refuse(*_a, **_k):
        raise Unexpressible("a template the composer wouldn't prove")

    monkeypatch.setattr(window_circuit, "class_circuit", refuse)
    monkeypatch.setattr(B, "registered", lambda _model: None)
    with pytest.raises(SystemExit) as e:
        report.main(["--class", "2,8,2"], check=reference, commit=lambda: None)
    out, err = capsys.readouterr()
    assert e.value.code == 2 and "AS SERVED: 1" in out and "NOT SERVED" not in out
    assert "the integration lowers no circuit of class 2,8,2" in err and "a template the composer wouldn't prove" in err


#: the execution phases of one phase, a week, a month of 30 days and a year of 365, at the tied policy's 600 s a phase
PHASES = {T.PHASE: 1, "week": 1_008, "month": 4_320, "year": 52_560}
#: the tied class's figures at a fleet's rate, by span: one β_x with the best threshold, one β_x at the top budget
#: with no block charged, and a β_x a block. The periods' are what the calculator's own test holds `bound_charged` to;
#: one phase's, where a block has the most untied words, is the weakest
FLEET = {T.PHASE: (2.05, 1.0, 1.0), "week": (1_614.1, 52.8, 53.2), "month": (4_787.8, 202.2, 209.6),
         "year": (20_422.0, 1_336.6, 1_844.3)}


def test_the_tied_file_is_the_served_class_reading_its_scan_state_row_and_a_salt(tmp_path) -> None:
    """The tied policy is the served one in everything but `k`, which is the scan state row and the salt with nothing
    left over and no linear outside the circuit. A file whose `k` leaves bits unexplained, or whose `[untied]` table
    misses a key, is refused."""
    t, served = P.tied(), P.stated()
    p = t.policy
    assert (p.label, p.model, p.cls, p.c, p.n, p.work) == ("tied", served.model, served.cls, served.c, served.n, served.work)
    assert p.k == P.scan_state_bits(p.cls) + 512 == 146_080 and (p.unexplained, p.outside) == (0, 0)
    assert [bits for _, bits, _ in p.parts] == [145_568, 512] and p.k < served.k
    assert (t.words, t.blocks, t.phase_seconds, t.macs_a_word, t.word_bits) == (671_369, 4, 600, 2_048, 32)
    # a node's bound is one phase's: its words a block of a phase, over the phases the period has
    assert t.untied(4, 600) == 671_369 and t.untied(8, 600) == 2 * 671_369 and t.untied(8, 1_200) == 671_369
    assert T.output_words(p, P.shape(p.model)) == 13_189_120
    text = P.TIED.read_text()
    (tmp_path / "k.toml").write_text(text.replace("k = 146080\n", "k = 146081\n"))
    with pytest.raises(ValueError, match="1 of its k = 146,081 bits are unexplained"):
        P.tied(tmp_path / "k.toml")
    (tmp_path / "u.toml").write_text(text.replace("word_bits = 32\n", ""))
    with pytest.raises(ValueError, match=r"\[untied\] states words, blocks, phase_seconds, macs_a_word, word_bits"):
        P.tied(tmp_path / "u.toml")
    assert text != text.replace("k = 146080\n", "") != text.replace("word_bits = 32\n", "") != text


def test_with_no_share_no_figure_of_the_tied_policy_prints(capsys) -> None:
    """`--uncredited-share` has no default: without it the command prints the served policy as before and says why the
    tied policy's figures are absent, and its number is nowhere in the print or the JSON."""
    doc = report.main(["--stated-only"], commit=lambda: None)
    out = capsys.readouterr().out
    assert doc["tied"] is None and doc["value"] == 1 and out.splitlines()[2] == "AS SERVED: 1"
    assert "no figure printed" in out and "--uncredited-share S" in out and "has no default" in out
    assert report.TIED_LABEL not in out and "limit at s" not in out and "37,653" not in out
    assert P.WORK_IS in out and doc["work_is"] == P.WORK_IS and "embedding table outside the registered weights" in out


def test_the_tied_policy_is_never_priced_as_a_stated_policy(capsys) -> None:
    """`--policy tied.toml` would print the tied policy's number as a headline with no share beside it: refused,
    with what to type instead."""
    with pytest.raises(SystemExit) as e:
        report.main(["--policy", str(P.TIED), "--stated-only"], commit=lambda: None)
    out, err = capsys.readouterr()
    assert e.value.code == 2 and "--uncredited-share S" in err and "never printed alone" in err and "37,653" not in out


@pytest.mark.parametrize("text", ["2", "-0.1", "101%", "a tenth", "", "nan", "inf"])
def test_a_share_is_a_number_from_zero_to_one_or_a_percentage(text, capsys) -> None:
    assert report._share("0.2315%") == pytest.approx(report._share("0.002315")) and report._share("0") == 0
    with pytest.raises(SystemExit) as e:
        report.main(["--stated-only", "--uncredited-share", text], commit=lambda: None)
    assert e.value.code == 2 and "a share of a block's work" in capsys.readouterr().err


def test_a_typed_share_prints_the_tied_figures_under_their_label_each_with_its_limit(capsys) -> None:
    """The command at a typed share. The served policy's 1 is still first and still the JSON's value. The tied section
    opens with its label, says the share was typed and which work the figures are for, and prints each of the
    calculator's fleets over one execution phase and over each period under four readings, each row the function's own
    figure at the inputs the file states (`min_cut.bound_charged`, recomputed here), its limit at the share
    (`min_cut.capped`) and the pooled total in gates at that limit. No number of the tied policy is printed without
    its limit. The fourth reading counts the uncredited words as fresh input, so it is never above the first."""
    s = 0.002315
    doc = report.main(["--stated-only", "--uncredited-share", "0.2315%"], commit=lambda: None)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert json.loads(lines[-1]) == doc and doc["value"] == 1 and lines[2] == "AS SERVED: 1"
    assert out.index("AS SERVED: 1") < out.index("Over a monitoring period") < out.index(report.TIED_LABEL)
    assert "RESTS ON NO PoUW VERDICT" in report.TIED_LABEL and "NOT SERVED" in report.TIED_LABEL
    assert "s = 0.2315% of a block's work was typed on the command line (--uncredited-share)" in out
    assert f"  work      the figures are for {P.WORK_IS}" in out and "bf16" in P.WORK_IS
    tied = doc["tied"]
    assert tied["uncredited"]["share"] == pytest.approx(s) and tied["uncredited"]["pouw_verdict"] is None
    assert (tied["label"], tied["k"], tied["work_is"]) == ("tied", 146_080, P.WORK_IS)
    p, sh = P.tied().policy, P.shape(P.tied().policy.model)
    N = p.work[P.TOTAL]
    work = {"N": N, "O": N * sh.outputs * 16 / sh.macs, "size": p.c}
    # one block: the count with every word tied, beside its limit, on one line
    one = min_cut.bound(p.c, p.k, **work).slowdown
    assert tied["block"]["number"] == one == pytest.approx(37_653.4, abs=0.05)
    assert tied["block"]["limit"] == min_cut.capped(one, tied["uncredited"]["share"]) == pytest.approx(427.1, abs=0.05)
    assert "every word tied, k = 146,080: number 37,653.4, limit at s 427.1" in out
    alone = [line for line in lines[:-1] if "37,653" in line]
    assert len(alone) == 1 and "limit at s" in alone[0]
    # the uncredited words: at most the share's MACs over the least fan-in, 32 fresh bits each
    words = tied["uncredited"]["words_a_block"]
    assert words == pytest.approx(s * N / 2_048) and tied["uncredited"]["least_fan_in"] == 2_048
    k_words = tied["block"]["with_words"]["k"]
    assert k_words == pytest.approx(146_080 + 32 * words)
    assert tied["block"]["with_words"]["number"] == min_cut.bound(p.c, k_words, **work).slowdown < one
    rows = tied["periods"]
    assert len(rows) == 2 * 4 * 4 and {r["where"] for r in rows} == {"one datacenter", "the world"}
    assert [r["period"] for r in rows[:16:4]] == [T.PHASE, "week", "month", "year"]
    # a fleet's phase is its week's blocks over the week's phases, and its untied words are the week's
    assert rows[0]["blocks"] == pytest.approx(rows[4]["blocks"] / 1_008) and rows[0]["untied"] == pytest.approx(rows[4]["untied"])
    for i in range(0, len(rows), 4):
        best, top, each, in_k = rows[i:i + 4]
        blocks, phases = best["blocks"], PHASES[best["period"]]
        assert len({(r["where"], r["period"], r["blocks"], r["untied"]) for r in rows[i:i + 4]}) == 1
        untied = 671_369 / 4 * blocks / phases
        assert best["untied"] == pytest.approx(untied, rel=1e-12) and best["untied_a_block"] == pytest.approx(untied / blocks)
        terms = {"blocks": blocks, "untied": best["untied"], "untied_macs": 2_048 * best["untied"], "most": 13_189_120,
                 **work}
        w = min_cut.best_w(p.c, p.k, **terms)
        want = (min_cut.bound_charged(p.c, p.k, w=w, **terms), min_cut.bound_charged(p.c, p.k, **terms),
                min_cut.bound_charged(p.c, p.k, rows=True, **terms),
                min_cut.bound_charged(p.c, k_words, w=min_cut.best_w(p.c, k_words, **terms), **terms))
        for r, got, pinned in zip((best, top, each, in_k), want, (*FLEET[best["period"]], None)):
            assert (r["number"], r["charged"], r["statement"], r["rests_on"]) == (
                got.slowdown, got.charged, got.statement, got.rests_on)
            assert pinned is None or r["number"] == pytest.approx(pinned, abs=0.05)
            assert r["limit"] == min_cut.capped(got.slowdown, tied["uncredited"]["share"]) < 1 / s
            assert r["gates"]["blocks"] == blocks * p.c and r["gates"]["training_by_the_count"] == got.gates
            assert r["gates"]["training_at_the_limit"] == pytest.approx(blocks * p.c / r["limit"])
            assert r["share"] == pytest.approx(1 / r["limit"]) and r["share"] >= s
            row = next(line for line in lines if line.split()[:1] == r["where"].split()[:1] and r["reading"] + "  " in line
                       and f"  {r['period']} " in line)
            assert f"{r['number']:,.1f}" in row and f"{r['limit']:,.1f}" in row
            assert report._g(r["gates"]["training_at_the_limit"]) in row
        assert (best["reading"], best["w"]) == (f"one β_x, with the threshold w = {w:.6g}", w)
        assert (top["reading"], top["w"]) == ("one β_x at the top budget, no block charged", None)
        assert (each["reading"], each["statement"]) == ("a β_x a block, no threshold", min_cut.HE_ROWS)
        assert in_k["reading"].endswith(f", {T.IN_K}") and in_k["uncredited_words_in_k"] and in_k["k"] == k_words
        assert in_k["number"] < best["number"] and in_k["limit"] < best["limit"]
        assert best["statement"] == top["statement"] == in_k["statement"] == min_cut.HE_CHARGED
    # each statement prints with the standing the locks give it, and no limit is called proved
    standing = {x: report.STANDING[lean.status(x)] for x in (min_cut.HE, min_cut.HE_CHARGED, min_cut.HE_ROWS)}
    assert standing[min_cut.HE] == "proved in the lock" and standing[min_cut.HE_ROWS] == "no Lean statement"
    for word in ("  proved    ", "  premise   ", "  scaled    ", *(f"{x} ({y})" for x, y in standing.items()),
                 "No limit is proved", min_cut.UNTIED_WORDS, min_cut.WORD_BITS,
                 "the uncredited share is taken as every block's"):
        assert word in out[out.index(report.TIED_LABEL):]
