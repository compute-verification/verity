import json

import pytest
from conftest import calculators_count
from synthetic import SHAPE, VOCAB, tiny_registry, write_config

from training_slowdown import calculator, config, gates, library, starts


def raw(phase: str, td: list, m: int, k: int, n: int, count: int = 1, affected: bool = True) -> dict:
    """A product row as a job's trace keeps one: bf16 operands, paid at their width, none gathered."""
    return {"count": count, "phase": phase, "op": "mm", "dims": {"batch": 1, "m": m, "k": k, "n": n},
            "token_dims": td, "dtypes": ["bfloat16", "bfloat16"], "affected": [affected, affected],
            "widths": [16, 16], "self": False, "gathered": [False, False], "rates": [16.0, 16.0]}


#: a step of 1,024 tokens through two 1,024-wide matrices: forward, input gradients and weight gradients
STEP = [raw("forward", ["m"], 1024, 1024, 1024, 2), raw("backward", ["m"], 1024, 1024, 1024, 2),
        raw("backward", ["k"], 1024, 1024, 1024, 2)]
GATES = 6 * 1024 ** 3 * gates.per_mac("gemm/bf16")
K = 65_536.0


def trace(rows: list[dict] = STEP) -> dict:
    fans = [{"op": "expand", "fan_out": 16.0, "count": 1}]
    return {"k": K, "rows": rows, "p_match": [], "at": {"80000": {"k": 80_000.0, "rows": rows, "p_match": fans}},
            "tokens": 1024, "circuit": {"gates": GATES}}


def test_a_traced_steps_slowdown_is_the_calculators_bound_on_its_work():
    c = GATES / 100
    got = library.bound(trace(), c, K)
    work = {x: got["figure"][x] for x in calculator.WORK} | {"C_gates": GATES}
    assert got["slowdown"] == calculators_count(c, K, work) > 1
    assert got["bound"]["theorem"].endswith("HEShared") and got["figure"]["N"] == 6 * 1024 ** 3
    assert got["binds"].startswith(f"HEShared on {got['figure']['subset']}, 100% of the MACs")
    one = library.bound(trace([raw("forward", ["m"], 1024, 8, 8)]), c, K)
    assert (one["slowdown"], one["binds"]) == (1.0, library.FLOOR) and one["bound"]["blocks"] * c < GATES
    none = library.bound(trace([raw("forward", ["m"], 1024, 1024, 1024, affected=False)]), c, K)
    assert (none["slowdown"], none["binds"]) == (1.0, library.EMPTY) and "bound" not in none


def test_a_step_the_bound_cant_read_has_no_slowdown_and_says_why():
    c = GATES / 100
    fans = library.bound(trace(), c, 80_000.0)
    assert fans["slowdown"] is None and "breaks the premise" in fans["binds"]
    other = library.bound(trace(), c, 7.0)
    assert other["slowdown"] is None and "at k = 65536, 80000, and the policy's k is 7" in other["binds"]
    past = library.bound(trace() | {"k": 9e8}, c, 9e8)
    assert past["slowdown"] is None and past["binds"].startswith("unmeasured: ")
    refused = library.bound({"refused": {"reason": "data_dependent", "op": "aten.nonzero", "message": "why"}}, c, K)
    assert refused == {"slowdown": None, "binds": "refused: the capture refused the step (data_dependent, "
                                                  "aten.nonzero: why)"}


def test_the_command_traces_each_entry_and_prints_its_row_and_its_run_over_each_period(tmp_path, capsys):
    """Two entries of two routes: an entry is whatever its config names, here a route with adapters and an output
    of its own, traced, and one whose E is coded, which a trace alone doesn't bound."""
    reg = tiny_registry(tmp_path / "reg")
    adapted = write_config(tmp_path / "adapted.toml", route="lora", dials={"rank": 2, "lr": 1e-2})
    coded = write_config(tmp_path / "coded.toml", dials={"trainable": "blocks.0.w_*", "store": "q8"})
    assert library.main([str(adapted), str(coded), "--starts", str(reg), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    head, first, second, under = out.splitlines()[:4]
    assert head.split() == ["entry", "route", "trains", "slowdown", "what", "binds"] and under == library.TRAINS
    got = json.loads((tmp_path / "out" / "library.json").read_text())
    assert got["schema"] == library.SCHEMA
    assert list(got["parts"]) == ["traced", "approximated", "proved", "premise", "scaled"]
    a, b = got["rows"]
    # the adapters of the start's seven matrices, rank 2, over the start's own parameters
    adapters = 2 * (4 * (16 + 16) + 3 * (16 + 32))
    assert a["trains"] == {"parameters": adapters, "of": SHAPE["parameters"], "share": adapters / SHAPE["parameters"]}
    assert first.split()[:4] == ["adapted", "lora", f"{100 * a['trains']['share']:.3g}%", "1"]
    assert second.split()[:4] == ["coded", "subset", "-", "-"] and "trains" not in b and "step" not in b
    assert a["step"]["approximated"] == {"macs": 0, "gates": 0.0, "share": 0.0} and "approximated: " not in first
    assert a["step"]["tokens"] == 2 * 1024 and a["policy"]["name"] == "console" and a["dials"]["rank"] == 2
    assert (b["slowdown"], b["binds"], b["periods"]) == (None, library.CODED, None)
    fleets, periods = calculator.fleets(), calculator.periods()
    assert list(periods)[1:] == ["week", "month", "year"] and len(fleets) == 2
    shares = a["periods"]["shares"]
    assert [(s["over"], s["period"]) for s in shares] == [(f["over"], p) for f in fleets for p in periods]
    run_and = a["budget"]["steps"] * a["step"]["gates"] * a["slowdown"] / gates.per_and("gemm/bf16")
    assert a["periods"]["run_and_gates"] == pytest.approx(run_and)
    assert shares[1]["share"] == pytest.approx(run_and / (fleets[0]["and_per_s"] * periods["week"]))
    for fleet in fleets:
        assert f"as a share of {fleet['over']}'s compute over each period" in out
    assert all(f"{part}: " in out for part in got["parts"])


def test_a_row_says_what_share_of_the_start_its_step_trains_and_what_the_capture_approximates(tmp_path):
    """Two optimizers on the tiny start. AdamW trains every parameter and no op of its step is approximated. Shampoo
    on the block and the final gain trains all but the embedding, and its roots are eigendecompositions the capture
    counts at n³ MACs a matrix: the row holds those MACs and their gates at the price the capture gives one, and the
    printed row says their share of the step's gates. With the embedding under AdamW beside it, it trains every
    parameter and has the same roots."""
    reg = starts.registry(tiny_registry(tmp_path / "reg"))

    def row(name: str, **dials) -> dict:
        return library.row(config.load(write_config(tmp_path / f"{name}.toml", dials={"lr": 1e-2} | dials), reg))

    whole, part = row("whole", trainable="*", opt="adamw"), row("part", trainable="blocks.*,norm", opt="shampoo")
    assert whole["trains"] == {"parameters": SHAPE["parameters"], "of": SHAPE["parameters"], "share": 1.0}
    assert part["trains"]["parameters"] == SHAPE["parameters"] - VOCAB * 16 and part["trains"]["share"] < 1
    assert whole["step"]["approximated"] == {"macs": 0, "gates": 0.0, "share": 0.0}
    approx = part["step"]["approximated"]
    # each of the seven matrices' two factors, an eigendecomposition each, at its side cubed
    assert approx["macs"] == 4 * 2 * 16 ** 3 + 3 * (16 ** 3 + 32 ** 3)
    assert approx["gates"] == approx["macs"] * gates.per_mac("elementwise/f32")
    assert 0 < approx["share"] == approx["gates"] / part["step"]["gates"] < 1
    said = f"; approximated: {approx['share']:.2%} of the step's gates are linear algebra at n³ MACs a matrix"
    assert library.line(part)[-1] == part["binds"] + said and library.line(whole)[-1] == whole["binds"]
    assert library.line(whole)[2] == "100%" and library.line(part)[2] == f"{100 * part['trains']['share']:.3g}%"
    beside = row("beside", trainable="*", opt="shampoo")
    assert beside["trains"] == whole["trains"] and beside["step"]["approximated"]["macs"] == approx["macs"]
    assert beside["step"]["gates"] > part["step"]["gates"] and library.line(beside)[2] == "100%"


def test_the_library_names_its_entries_and_refuses_one_it_lacks():
    names = [p.stem for p in library.entries()]
    assert names == sorted(names) and "s-t20-1.1x-ref-optim" in names
    assert library.entries(["s-t20-1.1x-ref-optim"]) == [library.ENTRIES / "s-t20-1.1x-ref-optim.toml"]
    with pytest.raises(SystemExit, match="no entry 'nope'"):
        library.entries(["nope"])
    for path in library.entries():
        config.load(path, starts.registry())
