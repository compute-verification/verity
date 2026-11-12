"""training_slowdown.grade on synthetic records and on a job's own: the fit, the seeds, H and p both ways, ρ from the
search record's trace at a named policy, the sensitivities and the code version from the trace's circuit and what is
blank without them, S and what it refuses, the bound of record beside it, the periods, and the command."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import synthetic

from training_slowdown import VERSION, calculator, gates, policy, work
from training_slowdown import formalism as F
from training_slowdown import grade as G
from training_slowdown.capture import products

FIXTURES = Path(__file__).parent / "fixtures"
#: the fit the synthetic starts sit on, and the start every config here branches from, the 45M reference's shape
E, A, GAMMA = 0.4, 20.0, 0.07
SHAPE = policy.SHAPES[policy.REFERENCE_PARAMS] | {"ctx": 1024, "parameters": policy.REFERENCE_PARAMS}
PER_MAC = gates.per_mac("gemm/bf16")
#: one step's log (a product of 80 MACs) and its gates
LOG, STEP_GATES = [["fwd", "theta", "mm", "[[8]]", 1, 80]], 80 * PER_MAC
#: a traced job's steps a checkpoint: past the first steps the products term leaves out, those a block of k bits holds
PER = 16
#: the 45M dense step's product rows as the capture that traced the stored records (`STATED["capture"]`) states them,
#: priced: what a job's trace of that step by that capture prices to
STATED = json.loads((FIXTURES / "dense_45m.products.json").read_text())
#: a raw row's keys beside its `dtypes` (`capture.products.raw_on_dial`), and the dtype an operand passed at a width has
RAW = ("count", "phase", "op", "dims", "token_dims", "affected", "widths", "self", "gathered", "rates")
DTYPE = {16: "bfloat16", 32: "float32"}


def loss_at(c0_macs: float) -> float:
    return E + A * (c0_macs * PER_MAC) ** -GAMMA


def registry(start: str = "s", shape: dict = SHAPE) -> dict:
    """A starts registry as `starts.toml` parses: four starts on the fit (the first named `start`, of `shape`), one
    off its frontier and one on other data."""
    on = {name: {"data": "d", "shape": shape, "c0_macs": c, "l0_bpb": loss_at(c)}
          for name, c in ((start, 1e15), ("m", 1e16), ("l", 1e17), ("x", 1e18))}
    return {"data": {"d": {}}, "starts": on | {"off": {"data": "d", "shape": shape, "c0_macs": 5e16, "l0_bpb": 1.4},
                                                "far": {"data": "e", "shape": shape, "c0_macs": 1e19, "l0_bpb": 0.2}}}


L0 = loss_at(1e15)
FIT = G.fit([(e["c0_macs"] * PER_MAC, e["l0_bpb"]) for e in registry()["starts"].values() if e["data"] == "d"])
#: the loss a gain of 1.1 from the start reaches, and losses a hair on either side of it
TARGET = G.target(FIT, L0, 1.1)
OVER, UNDER = TARGET + 1e-4, TARGET - 1e-4
CONFIG = {"name": "attack", "start": "s", "target": {"gain": 1.1}, "policy": "console", "honest": "reference",
          "seeds": {"search": [0], "final": [0, 1]}}


def record(losses: list, split: str = "final", start: str = "s", invalid=(), per: int = 1) -> dict:
    """A job's record as the harness writes it: one checkpoint per loss after each `per` steps, so checkpoint i has
    counted (i + 1)·per steps."""
    steps = [{"step": i, "tokens": 512, "struct": "a", "by": {"fwd/theta": 80}} for i in range(per * len(losses))]
    return {"training_slowdown": "0", "start": {"name": start}, "split": split, "invalid": list(invalid),
            "forward_macs_per_token": 51_342_080, "setup": {"by": {}}, "steps": steps, "tail": {"by": {}},
            "structures": {"a": {"updated": [["w", 0, 80]], "log": LOG}},
            "checkpoints": [{"step": per * (i + 1) - 1, "bpb": loss, "source": "output",
                             "score": {"bpb": loss, "start_bpb": L0, "split": split}} for i, loss in enumerate(losses)]}


def job(losses: list, seed: int = 0, split: str = "final", **kw) -> dict:
    return {"seed": seed, "split": split, "record": record(losses, split, **kw)}


def trace(**over) -> dict:
    """A job's trace of the 45M dense step (`traced.trace_step`) by the stored records' capture, the keys the grade
    reads and no other: the stated rows unpriced, each operand's dtype its passed width's (attention's float32, as
    that capture traced it), at each k of the dial; `over` its fields."""
    def raw(b: dict) -> dict:
        return {"k": b["k"], "p_match": b["p_match"],
                "rows": [{k: r[k] for k in RAW} | {"dtypes": [DTYPE[w] for w in r["widths"]]} for r in b["rows"]]}

    return raw(STATED) | {"capture": STATED["capture"], "tokens": STATED["tokens"], "data_dependent": [],
                          "carried_scalars": [], "at": {k: raw(b) for k, b in STATED["at"].items()}} | over


def traced(losses: list, seed: int = 0, **over) -> dict:
    """A search job whose record has a trace, a checkpoint each `PER` steps: a run of the 45M dense step's own plain
    products, each step after the first reading a weight the step before changed, traced as `trace` (`over` its
    fields), with a nonempty E at every checkpoint."""
    rec = record(losses, "search", per=PER)
    rec["structures"]["a"]["log"] = [["fwd", "theta", r["op"], json.dumps([[r["dims"]["m"], r["dims"]["k"]],
                                                                           [r["dims"]["k"], r["dims"]["n"]]]),
                                      r["count"], r["count"] * r["macs"]]
                                     for r in STATED["rows"] if r["op"] in F.PLAIN_PRODUCTS]
    for s in rec["steps"]:
        s["by"] = {"fwd/theta": sum(row[5] for row in rec["structures"]["a"]["log"])}
        s["weights"] = {"w": [str(s["step"])]} if s["step"] else {}
    for cp in rec["checkpoints"]:
        cp["e_bytes"] = 10**8
    return {"seed": seed, "split": "search", "record": rec | {"trace": trace(**over)}}


#: a traced step's circuit as the capture measures one (`capture.circuit.measure`), the keys the grade reads; a token
#: reads one id, so τ is its stub's and ρ counts the same steps with the circuit as without
CIRCUIT = {"sigma": 0.9, "affected_share": 0.5, "token_factor": 2.0, "grad_cut_bits": 3_000_000,
           "partial_cut_bits": 2_000_000, "delta_grad_bits": 4_000_000, "tau_ids": 1.0, "tau_other_bits": 0.0,
           "a_bits": 512.0, "u_in": [[0.0, 0.0], [1.0, 3e6]], "u1_bits": 3e6, "tail_share": 0.25}
#: the work `heavy` gives each step, over `traced`'s: a step's token work then fills several of the console's blocks
HEAVY = 4


def heavy(search: dict) -> dict:
    """`search` (a `traced` job) with `HEAVY` times the work in every step."""
    rec = copy.deepcopy(search["record"])
    for row in rec["structures"]["a"]["log"]:
        row[4], row[5] = row[4] * HEAVY, row[5] * HEAVY
    for s in rec["steps"]:
        s["by"] = {k: v * HEAVY for k, v in s["by"].items()}
    return search | {"record": rec}


def graded(runs, honest, config=CONFIG, **kw) -> dict:
    return G.grade(config, registry(), runs, honest, **kw)


def reach(at: int, seed: int, n: int = 8) -> dict:
    """A final job whose seed first reaches the gain at checkpoint `at` (from 1) of `n`."""
    return job([OVER] * (at - 1) + [UNDER] * (n - at + 1), seed)


def test_the_fit_reads_the_frontier_of_one_data_and_recovers_its_parameters():
    """Four starts on L(C) = E + A·C^(−γ) give it back; the start off the frontier (more compute, a worse loss) and the
    other data's are not read; under five points the fit is provisional, and under three there is none. A target is
    the gain's inverse, and a loss at or below E has no finite gain."""
    assert FIT["points"] == 4 and FIT["provisional"]
    assert (FIT["E"], FIT["A"], FIT["gamma"]) == pytest.approx((E, A, GAMMA), rel=1e-4)
    assert FIT["rms_pct"] == pytest.approx(0, abs=1e-3)
    assert G.gain_from(FIT, L0, TARGET) == pytest.approx(1.1) and TARGET < L0
    assert G.target(FIT, FIT["E"], 1.1) is None and G.gain_from(FIT, L0, FIT["E"]) == math.inf
    assert G.fit([(1e18, 1.2), (1e19, 1.1)]) is None and G.fit([(1e18, 1.2), (2e18, 1.1), (3e18, 1.0)]) is None
    three = G.fit([(1e18, loss_at(1e15)), (1e19, loss_at(1e16)), (1e20, loss_at(1e17))])
    assert G.c_spread(three, 1.0) is None and G.c_spread(FIT, TARGET) >= 0


def test_a_route_that_reaches_the_gain_is_priced_against_the_honest_run():
    """Each seed reaches at its first checkpoint at or below its target, its |C| the steps counted through it. C is the
    mean over the seeds, H the honest config's, and p their ratio with its range over the seeds."""
    s = graded([job([OVER, OVER, UNDER, UNDER], 0), job([OVER, OVER, OVER, UNDER], 1)], [job([OVER, UNDER], 7)])
    seeds = s["attack"]["splits"]["final"]["seeds"]
    assert [(v["reached"], v["step"], v["C_gates"]) for v in seeds.values()] == [
        (True, 2, pytest.approx(3 * STEP_GATES)), (True, 3, pytest.approx(4 * STEP_GATES))]
    assert seeds["0"]["gain"] > 1.1 and seeds["0"]["target"] == pytest.approx(TARGET)
    assert s["basis"]["split"] == "final"
    assert s["H"]["honest"]["C"]["mean"] == pytest.approx(2 * STEP_GATES) and s["H"]["any"]["config"] == "reference"
    assert s["p"]["honest"] == s["p"]["any"]
    assert s["p"]["honest"]["p"] == pytest.approx(3.5 / 2) and s["p"]["honest"]["range"] == pytest.approx([1.5, 2.0])
    assert s["attack"]["splits"]["final"]["C"]["sd"] == pytest.approx(STEP_GATES * math.sqrt(0.5))


def test_the_range_of_p_is_least_c_over_greatest_h_to_greatest_c_over_least_h():
    """Seeds that differ on both sides: the attack's reach at 5 and 7 steps and the honest config's at 2 and 4, so
    the range runs from 5/4 to 7/2 about p = 2, and pairing least with least would give 5/2 to 7/4."""
    s = graded([reach(5, 0), reach(7, 1)], [reach(2, 0), reach(4, 1)])
    C, H = s["attack"]["splits"]["final"]["C"], s["H"]["honest"]["C"]
    assert (C["min"], C["max"]) == pytest.approx((5 / 6 * C["mean"], 7 / 6 * C["mean"])) and H["n"] == 2
    assert (H["min"], H["max"]) == pytest.approx((2 / 3 * H["mean"], 4 / 3 * H["mean"]))
    p = s["p"]["honest"]
    assert p["p"] == pytest.approx(2.0) and p["range"] == pytest.approx([5 / 4, 7 / 2])
    assert p["range"][0] < p["p"] < p["range"][1]


def test_the_runs_gates_at_s_read_the_mean_of_c_over_seeds_that_differ():
    """The run at S is the mean |C| over the final seeds times ρ: not their least, their greatest, or the search
    record's |C| that ρ was read at."""
    s = graded([reach(5, 0), reach(7, 1), traced([OVER, UNDER, UNDER])], [reach(2, 0), reach(4, 1)])
    C, rho = s["attack"]["splits"]["final"]["C"], s["rho"]["rho"]
    assert C["min"] < C["mean"] < C["max"] and rho > 1
    per_and = gates.per_and("gemm/bf16")
    assert s["per_period"]["run_and_gates"] == pytest.approx(C["mean"] * rho / per_and)
    for other in (C["min"], C["max"], s["rho"]["C_gates"]):
        assert s["per_period"]["run_and_gates"] != pytest.approx(other * rho / per_and)
    assert s["S"]["honest"] == pytest.approx(2.0 * rho)


def test_rho_reads_the_first_planned_search_seed():
    """With two search records, ρ is read on the seed the config's `seeds.search` names first, whatever order the
    records come in."""
    early, late = traced([OVER, UNDER, UNDER]), traced([OVER, OVER, OVER, UNDER], seed=1)
    runs, ref = [reach(5, 0), reach(7, 1), early, late], [reach(2, 0)]
    first = graded(runs, ref, CONFIG | {"seeds": {"search": [0, 1], "final": [0, 1]}})
    second = graded(runs, ref, CONFIG | {"seeds": {"search": [1, 0], "final": [0, 1]}})
    assert (first["rho"]["seed"], first["rho"]["checkpoint_step"]) == (0, 2 * PER - 1)
    assert (second["rho"]["seed"], second["rho"]["checkpoint_step"]) == (1, 4 * PER - 1)
    assert first["rho"]["C_gates"] < second["rho"]["C_gates"] and first["rho"]["rho"] != second["rho"]["rho"]
    swapped = graded([reach(5, 0), reach(7, 1), late, early], ref,
                     CONFIG | {"seeds": {"search": [0, 1], "final": [0, 1]}})
    assert swapped["rho"]["seed"] == 0 and swapped["rho"]["rho"] == first["rho"]["rho"]


def test_a_route_that_does_not_reach_the_gain_has_no_p_and_no_s():
    s = graded([job([OVER, OVER], 0), job([OVER, OVER], 1), traced([OVER, OVER])], [job([OVER, UNDER])])
    final = s["attack"]["splits"]["final"]
    assert final["reached"] == 0 and final["C"] is None and s["p"] == {"honest": None, "any": None}
    assert all(not v["reached"] and v["step"] == 1 and v["gain"] < 1.1 for v in final["seeds"].values())
    assert s["rho"]["rho"] is not None and "doesn't reach" in s["rho"]["at"]
    assert s["S"]["any"] is None and "no seed of the config reaches 1.1x" in s["S"]["why"] and s["per_period"] is None
    assert "none: no seed of the config reaches" in G.text(s)


def test_h_against_any_route_is_the_least_over_every_config_given():
    """An attack cheaper than the honest run sets H against any route itself (p 1), and p against the honest route
    falls below 1; another config's records at the same start and gain count toward H against any route only."""
    mine, ref = [job([UNDER], 0), job([UNDER], 1)], [job([OVER, OVER, UNDER])]
    s = graded(mine, ref)
    assert s["H"]["any"]["config"] == "attack" and s["p"]["any"]["p"] == 1
    assert s["p"]["honest"]["p"] == pytest.approx(1 / 3)
    s = graded([job([OVER, OVER, UNDER], 0), job([OVER, OVER, UNDER], 1)], [job([OVER, UNDER])],
               others={"cheaper": [job([UNDER])]})
    assert s["H"]["any"]["config"] == "cheaper" and s["H"]["any"]["over"] == ["attack", "reference", "cheaper"]
    assert (s["p"]["any"]["p"], s["p"]["honest"]["p"]) == pytest.approx((3.0, 1.5))


def test_two_seeds_that_disagree_are_both_shown():
    """One final seed reaches and one doesn't: C is over the one that does, with no spread, and the block says 1 of
    2. A planned seed with no record is shown as such, never dropped."""
    s = graded([job([OVER, UNDER], 0), job([OVER, OVER], 1)], [job([UNDER])])
    final = s["attack"]["splits"]["final"]
    assert (final["reached"], final["of"]) == (1, 2) and final["C"] == {
        "mean": pytest.approx(2 * STEP_GATES), "sd": None, "min": pytest.approx(2 * STEP_GATES),
        "max": pytest.approx(2 * STEP_GATES), "n": 1}
    block = G.text(s)
    assert "(1 of 2 seeds reach)" in block and "final 1: not reached, at its last scored step 1" in block
    s = graded([job([OVER, UNDER], 0)], [job([UNDER])])
    assert s["attack"]["splits"]["final"]["seeds"]["1"] == {"reached": False, "why": "no record"}
    assert s["attack"]["splits"]["search"]["seeds"]["0"]["why"] == "no record"
    assert "final 1: no record" in G.text(s)


def test_an_unknown_policy_is_refused_by_name_and_a_name_resolves_in_one_place():
    """`console` is the rule relative to the start, any other name the calculator's registry's, with its Γ as gates
    of training's slowdown code; a name that is neither is refused, naming it and the names there are."""
    with pytest.raises(ValueError, match="no policy named 'ten-kilobytes'.*`console`.*v4-pro-session"):
        graded([job([UNDER])], [job([UNDER])], CONFIG | {"policy": "ten-kilobytes"})
    assert policy.named("console", SHAPE) == {
        "name": "console", "kind": policy.RULE, "c_gates": policy.CONSOLE[0], "k_bits": policy.CONSOLE[1],
        "why": policy.named("console", SHAPE)["why"]}
    small = policy.SHAPES[35_658_240] | {"parameters": 35_658_240}
    assert policy.named("console", small)["c_gates"] == policy.session_gates(35_658_240, *policy.CONSOLE_SESSION)
    fixed, there = policy.named("v4-pro-session", SHAPE), calculator.policy("v4-pro-session")
    assert fixed["kind"] == policy.FIXED and fixed["k_bits"] == there["k_bits"]
    assert fixed["c_gates"] == pytest.approx(there["gamma_flop"] / 2 * PER_MAC)
    assert set(calculator.policies()) >= {"v4-pro-session"} and calculator.policy("console") is None


def test_rho_is_the_traced_search_records_at_the_policy_and_s_is_p_times_it():
    """ρ is `formalism.published` at the policy's (c, k) on the search job's first checkpoint reaching the gain, and
    S = p·ρ against each H. At a registry policy whose k the trace's products aren't classified at, ρ is unmeasured
    with the formalism's reason and there is no S; p stands."""
    search = traced([OVER, UNDER, UNDER])
    runs, ref = [job([OVER, OVER, UNDER], 0), job([OVER, OVER, UNDER], 1), search], [job([OVER, UNDER])]
    s = graded(runs, ref)
    row = G.traced(search["record"], G.rows_of(search["record"])[1])
    want = F.published(F.measure(row, SHAPE | {"forward_macs_per_token": 51_342_080}), *policy.CONSOLE)
    assert want["rho"] > 1 and s["rho"]["rho"] == want["rho"] and s["rho"]["rests_on"] == want["rests_on"]
    assert (s["rho"]["checkpoint_step"], s["rho"]["at"]) == (2 * PER - 1, "its first checkpoint reaching the gain")
    assert s["rho"]["work"]["N"] > 0 and s["rho"]["work"]["C_gates"] == s["rho"]["C_gates"]
    assert s["S"] == {"honest": pytest.approx(1.5 * want["rho"]), "any": pytest.approx(1.5 * want["rho"]),
                      "why": None, "lower_bound": want["partial"]}
    assert s["rests_on"]["rho"] == want["rests_on"] and "4 of the 5 starts on data d" in s["rests_on"]["gain"]
    at = graded(runs, ref, CONFIG | {"policy": "v4-pro-session"})
    assert at["policy"]["kind"] == policy.FIXED and at["rho"]["rho"] is None and at["S"]["any"] is None
    assert at["S"]["why"] == at["rho"]["why"] and at["p"] == s["p"] and at["per_period"] is None


def test_the_trace_is_priced_by_the_capture_over_the_steps_after_the_first_update():
    """What ρ reads of a record's trace: each raw row priced by `capture.products.priced` by the rule of the capture
    that traced it, which gives back the row that capture states (a stored record's float32 attention products in
    bf16), at the rows' k and at each k of the dial; over the steps through the checkpoint that read a changed
    weight, by their tokens over the traced step's. The grade holds no price: the products are the formalism's of
    the stated rows."""
    rec = traced([OVER, UNDER])["record"]
    row = G.traced(rec, G.rows_of(rec)[1])["formalism"]
    block, got = row["trace"]["products"], row["products"]
    stated = [{k: v for k, v in r.items() if k not in ("P", "fed", "k_whole")} for r in STATED["rows"]]
    assert STATED["capture"] == products.READS[0] != products.CAPTURE
    assert any(r["dtypes"][0] == "float32" and r["phase"] != "optimizer" for r in rec["trace"]["rows"])
    assert block["rows"] == stated and block["rows"] == [products.priced(r, STATED["capture"])
                                                         for r in rec["trace"]["rows"]]
    assert set(block["at"]) == set(STATED["at"]) and all(
        [r["cell_gates"] for r in b["rows"]] == [r["cell_gates"] for r in STATED["at"][k]["rows"]]
        for k, b in block["at"].items())
    assert (block["k"], block["tokens"], block["capture"]) == (STATED["k"], STATED["tokens"], STATED["capture"])
    assert "capture_code" not in block
    steps = (2 * PER - 1) * 512 / STATED["tokens"]
    want = work.products_of_trace(STATED, steps)
    assert got["he"]["steps"] == steps and {k: got[k] for k in want} == want
    assert got["stand_in"] == {"structures": ["a"], "step_tokens": [512], "traced_tokens": STATED["tokens"],
                               "why": F.STAND_IN}
    assert row["labels"]["products"] == f"{F.TRACED_PRODUCTS}; {F.STAND_IN}"
    assert row["trace"]["affected_share"] == 1 and set(row["trace"]) == {
        "products", "affected_share", "data_dependent", "carried_scalars"}
    assert set(rec["trace"]) == set(G.TRACE)
    whole = traced([OVER, UNDER], tokens=512)["record"]
    assert "stand_in" not in G.traced(whole, G.rows_of(whole)[0])["formalism"]["products"]


def test_the_traces_circuit_gives_the_parts_their_structure_and_both_sensitivities():
    """A record whose trace has its circuit and its code version. The steps after the first update are one part with
    the traced step's structure: its σ and affected share of their gates, its token factor of their tokens, its
    Δ_grad and its cuts; the step before it reads θ alone and is a part with none. The sensitivities are the bound's
    over the part that has the cuts, n its steps and b the blocks a step's token work fills: reduction
    c·n·(b − 1)·partial/(k·|C|) and update c·n·grad/(k·|C|), and the bound says each leaves out the first step,
    whose token work splits too. τ, a and the update's profile are the circuit's, and the code version is stated."""
    search = heavy(traced([OVER, UNDER], circuit=CIRCUIT, capture_code="9.9"))
    rec = search["record"]
    row = G.traced(rec, G.rows_of(rec)[1])["formalism"]
    n, g = 2 * PER - 1, G.counted(G.rows_of(rec)[1]) / (2 * PER)
    assert row["parts"] == [
        {"gates": pytest.approx(g), "tokens": 512, "token_gates": pytest.approx(g), "affected_gates": 0, "steps": 1,
         "delta_grad_bits": None, "partial_bits": None, "grad_bits": None},
        {"gates": pytest.approx(n * g), "tokens": n * 512 * 2.0, "token_gates": pytest.approx(0.9 * n * g),
         "affected_gates": pytest.approx(0.5 * n * g), "steps": n, "delta_grad_bits": 4_000_000,
         "partial_bits": 2_000_000, "grad_bits": 3_000_000}]
    assert (row["tau_ids"], row["tau_other_bits"], row["a_bits"], row["u1_bits"], row["tail_share"]) == (1, 0, 512, 3e6, 0.25)
    assert row["u_in"] == CIRCUIT["u_in"] and row["u"][-1] == [1.0, 3e6]
    assert row["trace"]["affected_share"] == 0.5 and row["trace"]["products"]["capture_code"] == "9.9"
    assert {k: row["trace"][k] for k in CIRCUIT} == CIRCUIT and row["labels"]["delta_grad_bits"] == F.TRACED["delta_grad_bits"]
    s = graded([job([OVER, UNDER], 0), search], [job([UNDER])])
    c, k, total = s["policy"]["c_gates"], s["policy"]["k_bits"], 2 * PER * g
    b = math.ceil(0.9 * g / c)
    assert b >= 2 and math.ceil(g / c) >= 2
    assert s["rho"]["sensitivities"] == pytest.approx({"reduction": c * n * (b - 1) * 2e6 / (k * total),
                                                        "update": c * n * 3e6 / (k * total)})
    assert s["rho"]["blank"] == {} and s["rho"]["capture_code"] == s["rests_on"]["capture_code"] == "9.9"
    assert set(s["rho"]["unmeasured"]) == {"reduction", "update"} and all(
        why.endswith("the term sums the parts that have it") for why in s["rho"]["unmeasured"].values())
    red, upd = s["rho"]["sensitivities"].values()
    text = G.text(s)
    assert f"reduction {red:,.6g}, update {upd:,.6g}, beside rho and never in it" in text
    assert f"reduction: partial: {s['rho']['unmeasured']['reduction'].removeprefix('unmeasured: ')}" in text
    assert "unmeasured" not in text and ", its code 9.9" in text
    assert "blank" not in text


def test_a_trace_without_its_circuit_or_code_version_still_grades_and_says_what_is_blank():
    """A record whose trace has only what ρ reads grades at the ρ and S of the same record with its circuit. The
    figures that read what it lacks are blank, each with why and with how the trace is taken again: both
    sensitivities without the circuit, the code version without its own. One that lacks only one has only that
    blank."""
    ref, final = [job([UNDER])], job([OVER, UNDER], 0)
    full = graded([final, heavy(traced([OVER, UNDER], circuit=CIRCUIT, capture_code="9.9"))], ref)
    here = {"config": {"text": 'start = "s"\n'}, "commit": "c0ffee"}
    bare = heavy(traced([OVER, UNDER]))
    bare["record"] |= here
    s = graded([final, bare], ref)
    assert s["rho"]["rho"] == full["rho"]["rho"] > 1 and s["S"] == full["S"] and s["S"]["any"] is not None
    assert s["rho"]["sensitivities"] == {"reduction": None, "update": None} and s["rests_on"]["capture_code"] is None
    again = (f"re-trace it at the record's commit c0ffee and config.text with a CPU job, `{G.RETRACE}`, RUN_JSON the "
             f"record: the tree the job runs in takes the trace with its own capture, and refuses a config it reads "
             f"as another program than the record's")
    no_circuit = (f"the search record's trace has no circuit, the measured step's cut of its gradient and its "
                  f"structure, which the reduction and update sensitivities read; {again}")
    no_code = f"the search record's trace has no capture_code, the version of the code that took the trace; {again}"
    assert s["rho"]["blank"] == {"reduction": no_circuit, "update": no_circuit, "capture_code": no_code}
    text = G.text(s)
    assert "reduction blank, update blank, beside rho and never in it" in text
    assert f"reduction: blank: {no_circuit}" in text and f"update: blank: {no_circuit}" in text
    assert f", its code blank: {no_code}" in text
    coded = graded([final, heavy(traced([OVER, UNDER], capture_code="9.9"))], ref)["rho"]
    assert set(coded["blank"]) == {"reduction", "update"} and coded["capture_code"] == "9.9"
    assert coded["blank"]["update"].endswith("; the record has no config.text and no commit, so no config to re-trace from")
    uncoded = graded([final, heavy(traced([OVER, UNDER], circuit=CIRCUIT))], ref)["rho"]
    assert set(uncoded["blank"]) == {"capture_code"} and uncoded["sensitivities"] == full["rho"]["sensitivities"]


def test_a_step_is_after_the_first_update_when_it_reads_a_changed_weight():
    """A weight of θ is listed where its value isn't θ's, and a value derived from a gradient always; a tensor θ
    doesn't hold counts once a value it is read at isn't the first it was read at, so a table a model builds never
    does and an adapter does from its first update on. A prefix stops at its checkpoint."""
    def steps(*weights: dict) -> dict:
        return {"steps": [{"step": i, "weights": w} for i, w in enumerate(weights)]}

    table = {"new/0": ["t"]}
    assert [s["step"] for s in G.changed_steps(steps(table, table | {"w": ["1"]}, table | {"w": ["2"]}), None)] == [1, 2]
    assert G.changed_steps(steps(table, table, table), None) == []
    adapter = steps(table | {"new/1": ["a"]}, table | {"new/1": ["b"]}, table | {"new/1": ["a"]}, table | {"new/1": ["c"]})
    assert [s["step"] for s in G.changed_steps(adapter, None)] == [1, 3]
    assert [s["step"] for s in G.changed_steps(adapter, 2)] == [1]
    assert [s["step"] for s in G.changed_steps(steps({"new/1": ["a", "b"]}, {"derived/0": [""]}, {}), None)] == [0, 1]
    assert G.changed_steps(steps({"new/1": [""]}, {"new/1": [""]}), None) == []


def test_no_rho_from_a_trace_the_grade_cannot_read_and_the_reason_says_how_it_is_taken_again():
    """p reads the count alone, so it stands where ρ has no trace to read: a record with none, one whose step the
    capture refused, one of another capture version, or one without a key ρ reads. The reason names the job that
    re-traces the record at its commit and config.text; a record without them is told it has no config to re-trace
    from, and no command."""
    ref = [job([UNDER])]
    assert "no search record" in graded([job([UNDER])], ref)["S"]["why"]

    def why(search: dict, **record) -> str:
        search = search | {"record": search["record"] | record}
        s = graded([job([UNDER]), search], ref)
        assert s["rho"] == {"rho": None, "why": s["S"]["why"]} and s["S"]["any"] is None and s["p"]["any"]["p"] == 1
        assert f"none: {s['rho']['why']}" in G.text(s)
        return s["rho"]["why"]

    here = {"config": {"text": 'start = "s"\n'}, "commit": "c0ffee"}
    again = (f"re-trace it at the record's commit c0ffee and config.text with a CPU job, `{G.RETRACE}`, RUN_JSON the "
             f"record: the tree the job runs in takes the trace with its own capture, and refuses a config it reads "
             f"as another program than the record's")
    orphan = "the record has no config.text and no commit, so no config to re-trace from"
    assert G.RETRACE == "python -m training_slowdown.run --retrace RUN_JSON --out DIR --device cpu"
    assert why(job([OVER, UNDER], split="search"), **here) == (
        f"refused: the search record has no trace, and ρ reads a traced step; {again}")
    refused = {"refused": {"reason": "unsupported", "op": "aten::foo", "message": "foo has no trace"}}
    assert why(traced([OVER, UNDER]), trace=refused, **here) == (
        f"refused: the capture refused the search record's step (unsupported, aten::foo: foo has no trace), and ρ "
        f"reads a traced step; {again}")
    assert why(traced([OVER, UNDER], capture="0.1"), **here) == (
        f"refused: the search record's trace is capture 0.1's, and this grade reads capture 4.12.9's and "
        f"{products.CAPTURE}'s; {again}")
    lacking = traced([OVER, UNDER])
    del lacking["record"]["trace"]["at"], lacking["record"]["trace"]["carried_scalars"]
    assert why(lacking, **here) == (f"refused: the search record's trace has no at, carried_scalars, which ρ reads; "
                                    f"{again}")
    # a trace that was not taken is no refusal: nothing was found against the step, and the reason is the job's
    for failed in ("the route's process was ended by SIGSEGV after it wrote the job's record",
                   "RuntimeError: no trace today"):
        assert why(traced([OVER, UNDER]), trace={"failed": failed}, **here) == (
            f"unmeasured: the search record's trace was not taken ({failed}), and ρ reads a traced step; {again}")
    for search in (job([OVER, UNDER], split="search"), traced([OVER, UNDER], capture="0.1"),
                   traced([OVER, UNDER]) | {"record": traced([OVER, UNDER])["record"] | {"trace": refused}}):
        assert why(search).endswith(f"; {orphan}") and "python" not in why(search)
    assert why(traced([OVER, UNDER], capture="0.1"), commit="c0ffee").endswith(
        "; the record has no config.text, so no config to re-trace from")


def test_no_rho_from_a_step_the_formalism_refuses_or_a_prefix_no_update_precedes():
    """A read-back that decides what the graph can't carry, a trace whose products no update reaches, and a
    checkpoint with a nonempty E before any step reads a changed weight are refused, never priced; p stands."""
    ref = [job([UNDER])]
    read_back = {"kind": "a comparison of a read-back float", "where": "train.py:9", "steps": [0, 3]}
    s = graded([job([UNDER]), traced([OVER, UNDER], data_dependent=[read_back])], ref)
    assert s["rho"]["rho"] is None and s["rho"]["why"].startswith("refused: data-dependent: not measured")
    assert s["S"]["why"] == s["rho"]["why"] and s["p"]["any"]["p"] == 1
    assert s["attack"]["splits"]["search"]["seeds"]["0"]["reached"]
    still = trace()
    for b in (still, *still["at"].values()):
        b["rows"] = [r | {"affected": [False, False]} for r in b["rows"]]
    s = graded([job([UNDER]), traced([OVER, UNDER], **still)], ref)
    assert s["rho"]["why"] == ("refused: update-free trace: E is 100000000 bytes, but the traced step shows no update "
                               "(its affected share is 0)")
    idle = traced([OVER, UNDER])
    for step in idle["record"]["steps"]:
        step["weights"] = {"new/0": ["t"]}
    s = graded([job([UNDER]), idle], ref)
    assert s["rho"]["why"] == ("refused: update-free prefix: E is not empty, but no step through checkpoint "
                               f"{2 * PER - 1} reads a weight the run changed")


def test_a_records_provenance_is_passed_on_and_never_read():
    """A record that says where it came from is graded as the same record that doesn't: its `provenance` goes, as it
    is, beside the record in `records` and, for the record ρ read, into what S rests on and the block's lines."""
    said = {"converted_from": "another schema", "by": "a script"}
    plain, marked = traced([OVER, UNDER]), traced([OVER, UNDER])
    marked["record"]["provenance"] = said
    runs, ref = [job([OVER, UNDER], 0), job([OVER, UNDER], 1)], [job([UNDER])]
    a, b = graded([*runs, plain], ref), graded([*runs, marked], ref)
    assert a["rests_on"]["provenance"] is None and all("provenance" not in x for x in a["records"])
    assert b["rests_on"]["provenance"] == said and b["rho"]["provenance"] == said
    assert [x.get("provenance") for x in b["records"]] == [None, None, said, None]
    drop = {"rests_on": None, "records": None}
    assert {**b, **drop, "rho": b["rho"] | {"provenance": None}} == {**a, **drop}
    line = json.dumps(said, sort_keys=True)
    assert f"; the record's provenance: {line}" in G.text(b) and f", provenance: {line}" in G.text(b)
    assert "provenance" not in G.text(a)


def named(run: dict, dtype: str = "float32") -> dict:
    """`run` (a `job` or a `traced` one) as this version's counter writes it: every row of its op log names a dtype.
    A `job`'s names `dtype`. A `traced` one's names bf16, the dtype its trace ran its products in, whatever `dtype`
    is: a job counts and traces one step, and a record that names another dtype than its trace prices has no ρ
    (`G.priced_apart`)."""
    rec = copy.deepcopy(run["record"])
    for st in rec["structures"].values():
        st["log"] = [[*row, "bfloat16" if "trace" in rec else dtype] for row in st["log"]]
    return run | {"record": rec}


def test_records_priced_two_ways_give_no_p_and_no_s_and_the_reason_names_both_ways_out():
    """A record whose op log names each op's dtype is counted in it, here float32, and one that names none by the
    assumed dtypes, here bf16: C over H between them would be the ratio of the two prices and no slowdown. So where
    the records read are of both kinds, whichever config's they are, there is no p, no least H over the configs, no
    S and no period, and the reason says which records are which and names the two ways out. Records of one kind
    grade, and each record's line says how it is counted and on which device."""
    f32, bf16 = gates.per_mac("gemm/f32"), PER_MAC
    attack, ref = [job([OVER, UNDER], 0), job([OVER, UNDER], 1), traced([OVER, UNDER])], [job([UNDER])]
    plain, both = graded(attack, ref), graded([named(r) for r in attack], [named(r) for r in ref])
    assert both["p"] == plain["p"] and plain["S"]["why"] is None and both["S"]["why"] is None
    assert both["H"]["honest"]["C"]["mean"] == pytest.approx(STEP_GATES * f32 / bf16)
    assert {x["priced"] for x in plain["records"]} == {"by the assumed dtypes (its op log names none)"}
    assert {x["priced"] for x in both["records"]} == {"in the dtype its op log names for each op"}
    for runs, honest, which in (
            ([named(r) for r in attack], ref,
             "Counted in the dtype its op log names for each op: attack 0/final, attack 1/final, attack 0/search (3 "
             "in all). Counted by the assumed dtypes (its op log names none): reference 0/final (1 in all). Either"),
            (attack, [named(r) for r in ref],
             "each op: reference 0/final (1 in all). Counted by the assumed dtypes (its op log names none): attack "
             "0/final, attack 1/final, attack 0/search (3 in all)."),
            ([*attack[:2], named(attack[2])], ref,
             "each op: attack 0/search (1 in all). Counted by the assumed dtypes (its op log names none): attack "
             "0/final, attack 1/final, reference 0/final (3 in all).")):
        s = graded(runs, honest)
        assert s["p"] == {"honest": None, "any": None} and s["per_period"] is None and s["schema"] == "slowdown/v1"
        assert (s["S"]["honest"], s["S"]["any"]) == (None, None) and s["rho"]["rho"] is not None
        assert s["S"]["why"].startswith("the records read are priced two ways") and which in s["S"]["why"]
        assert s["S"]["why"].endswith(f". Either {G.MIXED}") and "--as-if-assumed" in G.MIXED
        assert "again with a job of this version" in G.MIXED
        assert (s["H"]["any"]["config"], s["H"]["any"]["C"]) == (None, None) and s["H"]["honest"]["C"]
        block = G.text(s)
        assert f"  {'S = p x rho':<17} none: the records read are priced two ways" in block
        assert "against any route none; against the honest route none" in block and "none: no S" in block
        assert "any route, over 2 configs (attack, reference), no config: none (the configs' records are priced" in block
    lines = G.text(graded([named(r | {"record": r["record"] | {"device": "cpu"}}) for r in attack], [named(ref[0])]))
    said = "TF32 switch not stated, counted"
    assert f"attack 0/final: version 0, on cpu, {said} in the dtype its op log names for each op, 2 of 2" in lines
    assert f"reference 0/final: version 0, {said} in the dtype its op log names for each op, 1 of 1" in lines
    assert f"attack 0/final: version 0, {said} by the assumed dtypes (its op log names none), 2 of 2" in G.text(plain)


def test_a_record_that_names_a_dtype_in_some_rows_and_none_in_others_is_refused():
    """A log with rows of both kinds would be counted two ways, the rows that name a dtype in it and the rest by the
    assumed ones, and neither line a record's pricing is said by would be true of it. No job writes one, so the
    grade refuses it where it reads the record, naming the record and how many rows of each kind it has, under
    `--as-if-assumed` too, which would leave the dtypes out and hide it. A record of one kind is read."""
    attack, ref = [job([OVER, UNDER], 0), job([OVER, UNDER], 1), traced([OVER, UNDER])], [job([UNDER])]
    both = named(attack[2])
    log = both["record"]["structures"]["a"]["log"]
    rows = len(log)
    assert rows > 2 and (G.dtype_rows(both["record"]), G.dtype_rows(attack[2]["record"])) == ((rows, 0), (0, rows))
    log[:2] = [row[:6] for row in log[:2]]
    assert G.dtype_rows(both["record"]) == (rows - 2, 2) and G.names_dtypes(both["record"]) is False
    refused = (rf"^grade: the record at seed 0, search names a dtype in {rows - 2} of its op log's rows and none in "
               r"2: .*one record two ways\. No job writes such a log: .*Count the config again with a job of this "
               r"version$")
    with pytest.raises(ValueError, match=refused):
        G.checkpoints(both, "s")
    for as_if in (False, True):
        with pytest.raises(ValueError, match=refused):
            graded([*attack[:2], both], ref, as_if_assumed=as_if)
    assert G.checkpoints(named(attack[2]), "s") and G.checkpoints(attack[2], "s")


def test_as_if_assumed_counts_every_record_by_the_assumed_dtypes_and_says_so_on_every_line(tmp_path, capsys):
    """`--as-if-assumed` on records of both kinds: every figure is the one the same records give with no dtype in
    their op logs, rows of one op and shapes in two dtypes one row. The result is under its own schema, says what it
    is, and says of each record that names dtypes that it was counted as if it named none; the block's first two
    lines say so and every other line begins with `as-if`; the command writes `slowdown.as-if.json` and no
    `slowdown.json`."""
    search = traced([OVER, UNDER])
    keys = [tuple(row[:4]) for row in search["record"]["structures"]["a"]["log"]]
    assert len(set(keys)) < len(keys)
    # a job's log has one row a key (`flops.Counter.record`): the fixture's rows of one op and shapes are summed
    search["record"] = G.without_dtypes(search["record"])
    attack, ref = [job([OVER, UNDER], 0), job([OVER, UNDER], 1), search], [job([UNDER])]
    two = named(attack[0])
    log = two["record"]["structures"]["a"]["log"]
    two["record"]["structures"]["a"]["log"] = [[*log[0][:4], 3, 60, "float32"], [*log[0][:4], 2, 20, "bfloat16"]]
    assert G.without_dtypes(two["record"])["structures"]["a"] == {"updated": [["w", 0, 80]],
                                                                   "log": [[*log[0][:4], 5, 80]]}
    assert G.without_dtypes(attack[0]["record"]) == attack[0]["record"]
    assert (G.names_dtypes(two["record"]), G.names_dtypes(attack[0]["record"])) == (True, False)
    assert G.names_dtypes(attack[0]["record"] | {"structures": {"a": {"log": []}}}) is None
    mixed = [two, attack[1], named(attack[2])]
    assert graded(mixed, ref)["S"]["any"] is None
    s, plain = graded(mixed, ref, as_if_assumed=True), graded(attack, ref)
    assert s["schema"] == "slowdown-as-if/v1" != plain["schema"] and plain["as_if"] is None
    assert s["as_if"] == G.AS_IF and "gemm bf16, elementwise bf16, optimizer f32, linalg f32" in G.AS_IF
    assert s["S"]["any"] == plain["S"]["any"] > 1 and s["rests_on"]["p"] == plain["rests_on"]["p"] + (
        ", every record counted by the assumed dtypes (as-if)")
    assert [x["priced"] for x in s["records"]] == [
        "as if its op log named no dtype", "by the assumed dtypes (its op log names none)",
        "as if its op log named no dtype", "by the assumed dtypes (its op log names none)"]
    differ = {"schema": None, "as_if": None, "records": None}
    assert {**s, **differ, "rests_on": s["rests_on"] | {"p": None}} == {
        **plain, **differ, "rests_on": plain["rests_on"] | {"p": None}}
    head, what, *rest = G.text(s).split("\n")
    assert head.startswith("AS-IF SLOWDOWN attack ") and what == f"  as-if: {G.AS_IF}"
    assert rest and all(line.startswith("  as-if ") for line in rest)

    def same(lines: list[str]) -> list[str]:
        """The lines that don't say how a record is counted: not the two records that name dtypes, nor what p rests on."""
        return [line for line in lines if "attack 0/" not in line and "p: counted |C|" not in line]

    assert [line.replace("  as-if ", "  ", 1) for line in same(rest)] == same(G.text(plain).split("\n")[1:])
    assert len(same(rest)) == len(rest) - 3 and "as-if" not in G.text(plain) and "AS-IF" not in G.text(plain)

    def put(root: str, seed: int, split: str, rec: dict) -> None:
        d = tmp_path / root / str(seed) / split
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(rec))

    for r in mixed:
        put("attack", r["seed"], r["split"], r["record"])
    put("reference", 0, "final", ref[0]["record"])
    (tmp_path / "starts.toml").write_text("".join(
        f'[starts.{n}]\ndata = "{e["data"]}"\nc0_macs = {e["c0_macs"]!r}\nl0_bpb = {e["l0_bpb"]!r}\n'
        + "shape = { " + ", ".join(f"{k} = {v}" for k, v in e["shape"].items()) + " }\n"
        for n, e in registry()["starts"].items()))
    (tmp_path / "attack.toml").write_text('start = "s"\ntarget = { gain = 1.1 }\npolicy = "console"\n'
                                          'honest = "reference"\nseeds = { search = [0], final = [0, 1] }\n')
    argv = ["--config", tmp_path / "attack.toml", "--starts", tmp_path / "starts.toml", "--records",
            tmp_path / "attack", "--honest", tmp_path / "reference", "--out", tmp_path / "out"]
    assert G.main([*map(str, argv), "--as-if-assumed"]) == 0
    assert capsys.readouterr().out.startswith("AS-IF SLOWDOWN attack ")
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["slowdown.as-if.json"]
    wrote = json.loads((tmp_path / "out" / "slowdown.as-if.json").read_text())
    assert wrote["schema"] == "slowdown-as-if/v1" and wrote["S"]["any"] == pytest.approx(s["S"]["any"])
    assert G.main([*map(str, argv)]) == 0
    assert "none: the records read are priced two ways" in capsys.readouterr().out
    assert json.loads((tmp_path / "out" / "slowdown.json").read_text())["S"]["any"] is None


@pytest.mark.parametrize("bad", [["fwd", "theta", "mm", "[[4, 5], [5, 5]]", 2, 200, "float32", "float32"],
                                 ["fwd", "theta", "mm", "[[4, 5], [5, 5]]", 2, 200, ""]])
def test_every_reader_of_an_op_log_refuses_a_row_of_neither_kind(bad, ran):
    """Every place that reads an op log's rows reads each by `gates.row_dtype`, so a row of eight fields, or one
    whose dtype is empty, is refused wherever it is read and never read as a row of the other kind."""
    from training_slowdown import as_math, of_record, run

    good = ["fwd", "theta", "mm", "[[4, 5], [5, 5]]", 2, 200, "float32"]
    rec = {"structures": {"s": {"log": [good, bad], "updated": []}}, "setup": {"struct": "s", "by": {"fwd/theta": 1}},
           "steps": [], "tf32": {"cuda_matmul_allow_tf32": True, "float32_matmul_precision": "high"}}
    step = ran["structures"][ran["steps"][-1]["struct"]]
    part = {k: 0 for k in ("tokens", "macs", "matmul_macs", "elementwise_macs", "approx_macs", "cpu_s", "wall_s",
                           "gpu_s", "exited")} | {"by": {}, "device_macs": {}, "weights": {}, "shapes": {}}
    readers = {"dtype_rows": lambda: G.dtype_rows(rec), "without_dtypes": lambda: G.without_dtypes(rec),
               "macs_log": lambda: G.macs_log(rec, None), "of_record": lambda: F.of_record(rec),
               "exact_under_tf32": lambda: gates.exact_under_tf32(rec),
               "rewrite": lambda: as_math.rewrite([good, bad], []),
               "per_row": lambda: of_record.per_row(step | {"log": [*step["log"], bad]}, synthetic.SHAPE),
               "outside_steps": lambda: run.outside_steps(
                   {"a": {"log": [good], "updated": []}, "b": {"log": [bad], "updated": []}},
                   [part | {"struct": "a"}, part | {"struct": "b"}])}
    assert of_record.per_row(step, synthetic.SHAPE)["log"]
    for name, read in readers.items():
        with pytest.raises(ValueError, match=r"^an op log's row is ") as e:
            read()
        assert repr(bad) in str(e.value), name


def test_a_record_with_an_op_in_a_dtype_no_definition_prices_is_refused_with_its_place():
    """A record whose op log names an op in a dtype no Definition prices has no count in gates: the grade stops, and
    says which record, which op and which dtype. Counted as if it named none, it is priced as one that names none."""
    with pytest.raises(ValueError, match=r"^grade: the record at seed 0, final has no count in gates: mm in float64: "
                                         r"no Definition prices a GEMM MAC in float64"):
        graded([named(job([UNDER]), "float64")], [named(job([UNDER]))])
    assert graded([named(job([UNDER]), "float64")], [named(job([UNDER]))], as_if_assumed=True)["p"]["any"]["p"] == 1


def stating(run: dict, allow: bool = False, precision: str = "highest") -> dict:
    """`run` with its record stating torch's TF32 switch as `run.tf32` writes it."""
    return run | {"record": run["record"] | {"tf32": {"cuda_matmul_allow_tf32": allow,
                                                      "float32_matmul_precision": precision}}}


@pytest.mark.parametrize("on", [{"allow": True, "precision": "high"}, {"allow": True}, {"precision": "medium"}])
def test_a_record_that_states_tf32_on_and_names_a_gemm_float32_is_refused_with_both_ways_out(on):
    """A dtype of torch's doesn't say a product ran in TF32, so a record that states the switch on, by either of its
    two readings, while its op log names a GEMM float32 has its products at a price that isn't its job's: the grade
    stops, and says which record, which ops and the two ways out. With those rows named `tf32` the same record
    grades, at the TF32 GEMM's price; one that states the switch off grades at the exact one; and counted as if its
    log named no dtype it is one that names none."""
    ref = [stating(named(job([UNDER])))]
    with pytest.raises(ValueError, match=r"^grade: the record at seed 0, final states torch's TF32 switch on, and its "
                                         r"op log names GEMM ops float32, the exact product \(fwd mm\): ") as e:
        graded([stating(named(job([UNDER])), **on)], ref)
    assert str(e.value).endswith(f"Either {G.TF32_FIX}") and "name those rows `tf32`" in G.TF32_FIX
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in G.TF32_FIX
    exact, tf32 = graded(ref, ref), graded([stating(named(job([UNDER]), "tf32"), **on)], ref)
    assert exact["H"]["honest"]["C"]["mean"] == pytest.approx(80 * gates.per_mac("gemm/f32"))
    assert tf32["attack"]["splits"]["final"]["C"]["mean"] == pytest.approx(80 * gates.per_mac("gemm/tf32"))
    assert [x["tf32"] for x in tf32["records"]] == [True, False]
    assert "attack 0/final: version 0, TF32 on, counted in" in G.text(tf32)
    assert "reference 0/final: version 0, TF32 off, counted in" in G.text(tf32)
    as_if = graded([stating(named(job([UNDER])), **on)], ref, as_if_assumed=True)
    assert as_if["p"]["any"]["p"] == 1 and [x["tf32"] for x in as_if["records"]] == [True, False]


def test_a_record_that_states_no_tf32_switch_is_read_as_before_and_its_line_says_so():
    """A record written before one stated the switch grades as it did, whatever its GEMMs' dtype, and the block
    says the switch is not stated."""
    old, new = graded([named(job([UNDER]))], [named(job([UNDER]))]), graded([stating(named(job([UNDER])))],
                                                                             [stating(named(job([UNDER])))])
    assert [x["tf32"] for x in old["records"]] == [None, None] and "TF32 switch not stated, counted" in G.text(old)
    drop = {"records": None}
    assert {**old, **drop} == {**new, **drop} and gates.exact_under_tf32(named(job([UNDER]))["record"]) == []


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One search job of the tiny start, run as a node runs it, through the sandbox."""
    return synthetic.run_job(tmp_path_factory.mktemp("job"))[0]


def test_a_jobs_own_record_is_graded_from_the_trace_its_job_took(ran):
    """The record a job writes, with the trace its job took of its own step, grades as it is: its rows priced by the
    capture, over the steps after the run's first update (step 0 reads θ and the tables the model builds, which the
    job holds as θ's, so it lists no weight), of the traced step's own shape. Its trace says more than the grade
    reads, and a refused one has no ρ."""
    assert set(G.TRACE) < set(ran["trace"]) and ran["trace"]["capture"] == products.CAPTURE
    assert ran["tf32"] == {"cuda_matmul_allow_tf32": False, "float32_matmul_precision": "highest"}
    assert gates.tf32_on(ran) is False
    assert [s["step"] for s in G.changed_steps(ran, None)] == [1, 2] and ran["steps"][0]["weights"] == {}
    f = G.traced(ran, G.rows_of(ran)[-1])["formalism"]
    assert f["trace"]["products"]["rows"] == [products.priced(r, products.CAPTURE) for r in ran["trace"]["rows"]]
    assert f["products"]["he"]["steps"] == 2 and "stand_in" not in f["products"] and f["trace"]["affected_share"] > 0
    config = CONFIG | {"start": "tiny", "seeds": {"search": [0], "final": []}}
    starts = registry("tiny", synthetic.SHAPE)
    runs = [{"seed": 0, "split": "search", "record": ran}]
    s = G.grade(config, starts, runs, runs)
    assert s["basis"]["split"] == "search" and s["rho"]["checkpoint_step"] == 2
    assert s["rho"]["rho"] is not None and s["rho"]["rho"] >= 1, s["rho"]
    assert s["rests_on"]["capture"] == products.CAPTURE and s["rests_on"]["capture_code"] == VERSION
    assert s["rho"]["blank"] == {} and ", its code blank" not in G.text(s)
    assert s["rho"]["terms"]["products"] is not None and s["rests_on"]["provenance"] is None
    assert (s["rho"]["work"] is None) == (not s["bound_of_record"]["available"])
    here = f"re-trace it at the record's commit {ran['commit']} and config.text with a CPU job, `{G.RETRACE}`"
    refused = {"refused": {"reason": "unsupported", "op": "aten::foo", "message": "foo has no trace"}}
    s = G.grade(config, starts, [runs[0] | {"record": ran | {"trace": refused}}], runs)
    assert s["rho"]["rho"] is None and here in s["rho"]["why"] and s["S"]["why"] == s["rho"]["why"]


def test_a_record_that_names_dtypes_has_one_price_with_its_own_trace_and_with_an_earlier_captures(ran):
    """Where the two rules of one price meet, on a record whose op log names dtypes. `two_pricings` is about records
    among themselves, C over H. `work.repriced` is about a slowdown over a trace's own circuit, which the grade never
    takes: it divides the trace's products by the record's count, so their one price is that each product the trace
    prices is priced in the dtype the record names for it.

    With the trace its job took, of this capture: neither rule speaks. Every traced product with fewer than two
    token dims, which M's all are, is priced in the dtype it was traced in, the one the record's log names for it.
    The products the two sides differ on are attention's, float32 in the trace and bf16 in the rows the counter
    writes for a fused kernel, and M leaves them out for their two token dims.

    With the same rows under an earlier capture, whose rule prices those float32 products at bf16: `work.repriced`
    lists them and nothing else, so over the trace's own circuit there is no slowdown, with that reason, and the
    grade's ρ and bound of record are what they are with the job's own trace, since nothing M holds moved."""
    tr = ran["trace"]
    assert G.names_dtypes(ran) is True and work.repriced(tr) == [] and tr["capture"] == products.PRICED_AS_TRACED_FROM
    names: dict = {}
    for _, _, op, shapes, _, _, dtype in ran["structures"][ran["steps"][tr["step"]]["struct"]]["log"]:
        if op in (*F.PLAIN_PRODUCTS, "bmm"):
            for _, m, k, n in F.log_products(op, json.loads(shapes)):
                names.setdefault((m, k, n), set()).add(dtype)

    def named_for(r: dict) -> set:
        shape = (r["dims"]["m"], r["dims"]["k"], r["dims"]["n"])
        return names.get(shape) or names[shape[::-1]]

    attentions = [r for r in tr["rows"] if sum(x in r["token_dims"] for x in "mkn") >= 2]
    others = [r for r in tr["rows"] if r not in attentions]
    assert others and all(named_for(r) == {products.traced_dtype(r)} == {"bfloat16"} for r in others)
    assert attentions and all((r["op"], products.traced_dtype(r), named_for(r)) == ("bmm", "float32", {"bfloat16"})
                              for r in attentions)

    earlier = products.READS[0]
    old = tr | {"capture": earlier}
    moved = work.repriced(old)
    assert earlier != products.CAPTURE and len(moved) == len(attentions) and all(r in attentions for r in moved)
    pol = policy.named(CONFIG["policy"], synthetic.SHAPE)
    alone = calculator.slowdown(pol, old)
    assert calculator.slowdown(pol, tr)["slowdown"] is not None and alone["slowdown"] is None
    assert alone["why"] == work.repriced_reason(old, moved) and "one slowdown has one price" in alone["why"]

    row = G.rows_of(ran)[-1]
    now, then = (G.traced(ran | {"trace": t}, row)["formalism"] for t in (tr, old))
    assert then["products"]["he"] == now["products"]["he"] and now["products"]["he"]["groups"]
    prices = [(x["op"], x["gates_per_mac"], y["gates_per_mac"])
              for x, y in zip(now["trace"]["products"]["rows"], then["trace"]["products"]["rows"]) if x != y]
    assert prices == [("bmm", gates.per_mac("gemm/f32"), PER_MAC)] * len(attentions)

    config = CONFIG | {"start": "tiny", "seeds": {"search": [0], "final": []}}
    runs = [{"seed": 0, "split": "search", "record": ran}]
    a = G.grade(config, registry("tiny", synthetic.SHAPE), runs, runs)
    b = G.grade(config, registry("tiny", synthetic.SHAPE), [runs[0] | {"record": ran | {"trace": old}}], runs)
    assert (a["rests_on"]["capture"], b["rests_on"]["capture"]) == (products.CAPTURE, earlier)
    same = ("rho", "terms", "C_gates", "work", "checkpoint_step")
    assert {k: b["rho"][k] for k in same} == {k: a["rho"][k] for k in same} and a["rho"]["rho"] is not None
    assert b["bound_of_record"] == a["bound_of_record"] and b["p"] == a["p"] and b["S"] == a["S"]
    assert "priced" not in (a["S"]["why"] or "") and {x["priced"] for x in a["records"] + b["records"]} == {
        G.PRICED[True]}


@pytest.fixture(scope="module")
def adapted(tmp_path_factory):
    """One search job of the tiny start under LoRA, which builds each adapted weight's update B·A in float32, outside
    the cast."""
    return synthetic.run_job(tmp_path_factory.mktemp("lora"), **synthetic.JOBS["lora"])[0]


def test_a_trace_that_prices_a_product_of_m_in_another_dtype_than_the_record_names_is_refused(ran, adapted):
    """ρ divides the work of the trace's products by the record's count, so their one price is that each product M
    holds is priced, in the trace, in the dtype the record's op log names for it (`G.priced_apart`); else the record
    has no ρ, and no S.

    M holds a LoRA job's two builds of B·A, float32 in its trace and in its record. With the trace its job took, of
    this capture, each is priced as traced, nothing is apart and the record grades. The same rows as an earlier
    capture's are priced by that capture's rule, a product traced wider than bf16 at bf16: the builds are then bf16
    in the work and float32 in the count, and ρ is refused, saying how many, the largest with its two dtypes and the
    way out, the job traced again at capture `PRICED_AS_TRACED_FROM` or later. A job with no product of M wider than
    bf16 has none apart under either capture: attention's float32 products are outside M.

    A trace whose step ran M's products in another dtype than the job did is refused at this capture too, and there
    the way out is a trace whose step runs them as the job did: the job traced again on a device of the kind the
    record states, a card for a card's job. A record whose log names no dtype has none to differ from, and a product
    whose shape the log never ran is `formalism.other_program`'s to find, not this rule's."""
    tr, row, earlier = adapted["trace"], G.rows_of(adapted)[-1], products.READS[0]
    old = tr | {"capture": earlier}
    builds = [r for r in tr["rows"] if work.in_m(r) and products.traced_dtype(r) == "float32"]
    assert len(builds) == 2 and all(r["phase"] == "forward" and not r["token_dims"] for r in builds)
    assert G.names_dtypes(adapted) is True and tr["capture"] == products.PRICED_AS_TRACED_FROM
    assert G.priced_apart(tr, row["macs_log"]) == [] and G.traced(adapted, row)["formalism"]["products"]["he"]["groups"]
    apart = G.priced_apart(old, row["macs_log"])
    assert sorted(apart, key=str) == sorted(({"row": r, "priced": "bfloat16", "named": ["float32"]} for r in builds),
                                            key=str)
    assert all(x["row"] in work.repriced(old) for x in apart)
    d = apart[0]["row"]["dims"]
    assert d["m"] * d["k"] * d["n"] == max(math.prod(r["dims"].values()) for r in builds)
    said = (f"2 of the traced step's matrix products in M are priced in another dtype than the search record's op log "
            f"names for them (the largest: 1 forward mm of {d['m']}×{d['k']} by {d['k']}×{d['n']}, priced in "
            f"bfloat16 where the record names float32), and one slowdown has one price: trace the job again at "
            f"capture {products.PRICED_AS_TRACED_FROM} or later, which prices each product in the dtype it was "
            f"traced in (float32 here; this trace is capture {earlier}'s); {G.retrace(adapted)}")
    with pytest.raises(F.Refused) as refused:
        G.traced(adapted | {"trace": old}, row)
    assert str(refused.value) == said and G.RETRACE in said

    config = CONFIG | {"start": "tiny", "seeds": {"search": [0], "final": []}}
    runs = [{"seed": 0, "split": "search", "record": adapted}]
    a = G.grade(config, registry("tiny", synthetic.SHAPE), runs, runs)
    b = G.grade(config, registry("tiny", synthetic.SHAPE), [runs[0] | {"record": adapted | {"trace": old}}], runs)
    assert a["rho"]["rho"] is not None and a["rests_on"]["capture"] == products.CAPTURE
    assert b["rho"]["rho"] is None and b["rho"]["why"] == f"refused: {said}" and b["S"]["why"] == b["rho"]["why"]
    assert (b["S"]["honest"], b["S"]["any"]) == (None, None) and b["p"] == a["p"]
    assert f"none: refused: {said[:60]}" in G.text(b)
    as_if = G.grade(config, registry("tiny", synthetic.SHAPE), [runs[0] | {"record": adapted | {"trace": old}}], runs,
                    as_if_assumed=True)
    assert as_if["rho"]["rho"] is not None and as_if["schema"] == G.SCHEMA_AS_IF
    assert G.priced_apart(old, G.rows_of(G.without_dtypes(adapted))[-1]["macs_log"]) == []
    elsewhere = old | {"rows": [r | {"dims": r["dims"] | {"k": r["dims"]["k"] + 1}} if r in builds else r
                                for r in tr["rows"]]}
    assert G.priced_apart(elsewhere, row["macs_log"]) == []

    mine, last = ran["trace"], G.rows_of(ran)[-1]
    assert [G.priced_apart(t, last["macs_log"]) for t in (mine, mine | {"capture": earlier})] == [[], []]
    held = [r for r in mine["rows"] if work.in_m(r)]
    assert held and {products.traced_dtype(r) for r in held} == {"bfloat16"}
    uncast = mine | {"rows": [r | {"dtypes": ["float32", "float32"]} if r in held else r for r in mine["rows"]]}
    found = G.priced_apart(uncast, last["macs_log"])
    assert len(found) == len(held) and all((x["priced"], x["named"]) == ("float32", ["bfloat16"]) for x in found)
    again = ("priced in float32 where the record names bfloat16), and one slowdown has one price: the traced step ran "
             "it in float32, not as the job did, and only a trace whose step runs it in bfloat16 has one price with "
             "this record: trace the job again on a device of the kind it ran on")
    job = "python -m training_slowdown.run --retrace RUN_JSON --out DIR --device "
    for device, way in (("cpu", f"; {G.retrace(ran)}"), ("cuda:0", f"; {G.retrace(ran, 'cuda')}"),
                        (None, ", which the record does not state")):
        with pytest.raises(F.Refused) as refused:
            G.traced(ran | {"trace": uncast, "device": device}, last)
        assert str(refused.value).endswith(again + way) and "or later" not in str(refused.value)
    assert ran["device"] == "cpu" and f"with a CPU job, `{job}cpu`" in G.retrace(ran) and G.RETRACE == job + "cpu"
    assert f"with a job on cuda, `{job}cuda`" in G.retrace(ran, "cuda")


def test_a_jobs_own_trace_gives_both_sensitivities_back(ran):
    """The tiny start traced through the job, graded at a block a step's token work fills four of: the two steps
    after the first update are one part with the trace's own circuit, the gradient's cut and a token group's partial
    as the capture measured them, and the reduction and update sensitivities are the bound's over it. The first step
    reads θ alone and has no cut, so each sum leaves it out, and no figure is blank."""
    circuit = ran["trace"]["circuit"]
    setup, first, rest = G.traced(ran, G.rows_of(ran)[-1])["formalism"]["parts"]
    assert (setup["tokens"], first["steps"], first["grad_bits"], first["affected_gates"]) == (0, 1, None, 0)
    assert (rest["steps"], rest["partial_bits"], rest["grad_bits"], rest["delta_grad_bits"]) == (
        2, circuit["partial_cut_bits"], circuit["grad_cut_bits"], circuit["delta_grad_bits"])
    assert rest["token_gates"] == circuit["sigma"] * rest["gates"] and circuit["grad_cut_bits"] > 0
    run = {"seed": 0, "split": "search", "record": ran}
    c, k = rest["token_gates"] / 2 / 3.5, policy.CONSOLE[1]
    rho, _ = G.rho_of(run | {"checkpoints": G.checkpoints(run, "tiny")}, FIT, 1.1, {"c_gates": c, "k_bits": k},
                      synthetic.SHAPE)
    total = rho["C_gates"]
    assert total == pytest.approx(setup["gates"] + first["gates"] + rest["gates"])
    assert rho["sensitivities"] == pytest.approx({"reduction": c * 2 * 3 * circuit["partial_cut_bits"] / (k * total),
                                                  "update": c * 2 * circuit["grad_cut_bits"] / (k * total)})
    assert rho["sensitivities"]["reduction"] > rho["sensitivities"]["update"] > 0
    assert rho["blank"] == {} and rho["capture_code"] == VERSION and set(rho["unmeasured"]) == {"reduction", "update"}


def test_p_reads_one_split_and_a_record_out_of_place_is_refused():
    """Without a final record of the honest config, C and H are both read on the search split, and the grade says
    so. A record of another start, or scored on another split than its place, is refused; an invalid record's seed
    is unscored with the record's reason."""
    s = graded([job([UNDER]), job([OVER, UNDER], split="search")], [job([UNDER], split="search")])
    assert s["basis"] == {"split": "search", "why": "no final record of reference"}
    assert s["p"]["honest"]["p"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="from start 'm', not the config's 's'"):
        graded([job([UNDER], start="m")], [job([UNDER])])
    with pytest.raises(ValueError, match="seed 0, final was scored on 'search'"):
        graded([{"seed": 0, "split": "final", "record": record([UNDER], "search")}], [job([UNDER])])
    with pytest.raises(ValueError, match="no start 'nowhere'"):
        graded([job([UNDER])], [job([UNDER])], CONFIG | {"start": "nowhere"})
    bad = graded([job([UNDER], invalid=["foreign op"])], [job([UNDER])])["attack"]["splits"]["final"]["seeds"]["0"]
    assert not bad["reached"] and bad["why"] == "no scored checkpoint: invalid: foreign op"


def test_rho_is_the_bound_of_record_where_products_bind():
    """ρ's products term is the bound of record: the calculator's count solved for the fewest blocks
    (`min_cut.blocks_at_least`) times c/|C|, on the traced step's N, O, β_w, β_x, b_y and F. So where products bind, ρ
    is the bound's slowdown and S is p times it, and the block says so under the bound."""
    s = graded([job([OVER, UNDER], 0), traced([OVER, UNDER])], [job([UNDER])])
    w, got = s["rho"]["work"], s["bound_of_record"]
    c, k = policy.CONSOLE
    solved = calculator._min_cut().blocks_at_least(w["N"], w["O"], k, w["beta_w"] * w["beta_x"], w["b_y"], w["F"])
    assert got["blocks"] == solved and got["slowdown"] == solved * c / w["C_gates"] > 1
    assert s["rho"]["binding_kind"] == "products"
    assert s["rho"]["rho"] == s["rho"]["terms"]["products"] == got["slowdown"]
    assert s["S"]["any"] == pytest.approx(s["p"]["any"]["p"] * got["slowdown"])
    assert f"\n  {'':<17} {G.BOUND_IS_RHO}\n" in G.text(s) and "this attack pays" not in G.text(s)
    assert "rho is it" in G.BOUND_IS_RHO and "closed form" not in G.text(s) and "overhead" not in G.text(s)


def test_the_bound_of_record_is_the_calculators_and_rho_follows_it(monkeypatch):
    """The bound of record is the calculator's `min_cut.bound` at the policy's (c, k) on the traced step's N, O, β_w,
    β_x, b_y, F and |C|, with no n, and ρ's products term is its blocks times c/|C|: with the calculator's `bound`
    replaced, each choice of the work is scored by it, and ρ and S follow it. Without a traced step, or on one whose
    M is empty (no operand holds a bit, so there is no β to hand over), it is not available, with the reason."""
    runs, ref = [job([OVER, UNDER], 0), job([OVER, UNDER], 1), traced([OVER, UNDER])], [job([UNDER])]
    calls = []

    def stub(c, k, n, **kw):
        calls.append((c, k, n, kw))
        return SimpleNamespace(slowdown=300.0 * c / kw["size"], blocks=300.0, statement="HE",
                               rests_on="given (H) and (E)")

    stubbed = SimpleNamespace(**vars(calculator._min_cut()) | {"bound": stub, "HE": "the statement"})
    monkeypatch.setattr(calculator, "_min_cut", lambda: stubbed)
    s = graded(runs, ref)
    w, got = s["rho"]["work"], s["bound_of_record"]
    assert len(calls) > 1 and all(call[:3] == (*policy.CONSOLE, None) for call in calls)
    assert calls[-1][3] == {"N": w["N"], "O": w["O"], "size": w["C_gates"], "beta_w": w["beta_w"],
                            "beta_x": w["beta_x"], "b": w["b_y"], "F": w["F"]}
    slowdown = 300.0 * policy.CONSOLE[0] / w["C_gates"]
    assert (got["available"], got["slowdown"], got["blocks"], got["theorem"]) == (True, slowdown, 300.0, "HE")
    assert s["rho"]["rho"] == slowdown > 1 and s["S"]["any"] == pytest.approx(s["p"]["any"]["p"] * slowdown)
    assert f"pays at least {slowdown:,.6g}x (HE: at least 300 blocks" in G.text(s)
    assert "rests on: given (H) and (E)" in G.text(s)
    made = len(calls)
    untraced = graded([job([UNDER])], ref)["bound_of_record"]
    assert not untraced["available"] and "gives none" in untraced["why"] and len(calls) == made
    assert untraced["theorem"] == "the statement" and (untraced["c_gates"], untraced["k_bits"]) == policy.CONSOLE
    assert "bound of record   not available: " in G.text(graded([job([UNDER])], ref))
    flat = trace()
    for b in (flat, *flat["at"].values()):
        b["rows"] = [r | {"rates": [0.0, 0.0]} for r in b["rows"]]
    empty = graded([*runs[:2], traced([OVER, UNDER], **flat)], ref)
    assert empty["rho"]["terms"] == {"products": 0.0} and empty["rho"]["rho"] == 1 and empty["rho"]["work"] is None
    assert not empty["bound_of_record"]["available"] and len(calls) == made


def test_the_grades_slowdown_is_the_entry_points(monkeypatch):
    """Where Fit fails, the grade's `rho`, the bound of record and S are the entry point's and the grade computes
    none: with `calculator.slowdown` replaced by one that gives 7.25 wherever the real one gives a slowdown, `rho`
    and the bound of record's slowdown are 7.25, S is p times it, and the products term beside them stays the real
    one's."""
    runs, ref = [job([OVER, UNDER], 0), job([OVER, UNDER], 1), traced([OVER, UNDER])], [job([UNDER])]
    real, honest = calculator.slowdown, graded(runs, ref)

    def entry(pol, step, run=None):
        got = real(pol, step, run)
        return got if got["slowdown"] is None else got | {"slowdown": 7.25, "bound": got["bound"] | {"slowdown": 7.25}}

    monkeypatch.setattr(calculator, "slowdown", entry)
    s = graded(runs, ref)
    assert s["rho"]["rho"] == s["bound_of_record"]["slowdown"] == 7.25 != honest["rho"]["rho"]
    assert s["rho"]["binding_kind"] == "products" and s["rho"]["terms"] == honest["rho"]["terms"]
    assert s["S"]["any"] == pytest.approx(s["p"]["any"]["p"] * 7.25) and s["p"] == honest["p"]
    assert s["rho"]["work"] == s["bound_of_record"]["work"] == honest["rho"]["work"]
    assert s["rho"]["work"]["C_gates"] == s["rho"]["C_gates"]


def test_the_bound_call_fits_the_calculator_as_it_is():
    """On the calculator as the tree has it, the grade's own call gives a slowdown of at least 1 under one of the two
    statements the calculator's `min_cut` names, the first of them the one the grade says ρ's products term solves
    the count of."""
    s = graded([job([OVER, UNDER], 0), job([OVER, UNDER], 1), traced([OVER, UNDER])], [job([UNDER])])
    got, mod = s["bound_of_record"], calculator._min_cut()
    assert got["available"] and got["slowdown"] >= 1
    assert got["theorem"] in (mod.HE, mod.HE_SHARED) and got["rests_on"].startswith(got["theorem"])
    assert calculator.theorem() == mod.HE and mod.HE in s["per_period"]["parts"]["proved"]


def test_the_periods_and_fleets_are_the_calculators():
    """The run's AND gates at S are C·ρ at a bf16 GEMM's gates an AND, and each share is over a fleet of the
    calculator's at its peak, for the census's window and core's periods."""
    s = graded([job([OVER, UNDER], 0), job([OVER, UNDER], 1), traced([OVER, UNDER])], [job([UNDER])])
    per = s["per_period"]
    assert per["run_and_gates"] == pytest.approx(2 * STEP_GATES * s["rho"]["rho"] / gates.per_and("gemm/bf16"))
    assert [(x["over"], x["period"]) for x in per["rows"]] == [
        (over, p) for over in ("one datacenter", "the world") for p in ("window", "week", "month", "year")]
    fleets, periods = {f["over"]: f for f in calculator.fleets()}, calculator.periods()
    assert periods["window"] == calculator.window()[1] < periods["week"] < periods["month"] < periods["year"]
    for x in per["rows"]:
        assert x["share"] == pytest.approx(per["run_and_gates"] / (fleets[x["over"]]["and_per_s"] * periods[x["period"]]))
    dc = fleets["one datacenter"]
    e4m3 = gates.per_mac("gemm/e4m3") / gates.per_and("gemm/e4m3")
    assert dc["and_per_s"] == pytest.approx(dc["flop_per_s"] / 2 * e4m3) and set(per["parts"]) == {
        "measured", "proved", "scaled"}
    block = G.text(s)
    assert "per period        the run at S: " in block
    assert block.startswith("SLOWDOWN attack   start s, gain 1.1x, policy console (rule relative to the start: ")


def test_a_records_rows_carry_its_prefix_and_the_command_reads_a_tree_of_jobs(tmp_path, capsys):
    """`rows_of` gives each checkpoint its prefix's op log, the last one the whole run's. The command reads each
    `<seed>/<split>/run.json` and nothing beside it, prints the block and writes `slowdown.json`."""
    coded = record([OVER, UNDER])
    coded["checkpoints"][1] |= {"e_bytes": 100, "format": "full"}
    rows = G.rows_of(coded)
    assert [r["macs_log"] for r in rows] == [{"by": {"fwd/theta": 80 * n}, "ops": {"fwd theta mm [[8]]": [n, 80 * n]}}
                                             for n in (1, 2)]
    assert [r["output"] for r in rows] == [{}, {"e_bytes": 100, "format": "full"}] and rows[1]["split"] == "final"
    assert [G.counted(r) for r in rows] == pytest.approx([STEP_GATES, 2 * STEP_GATES]) and G.counted({}) is None

    def put(root: str, seed: int, split: str, rec: dict) -> None:
        d = tmp_path / root / str(seed) / split
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(rec))

    put("attack", 0, "final", record([OVER, UNDER]))
    put("attack", 1, "final", record([OVER, OVER, UNDER]))
    put("attack", 0, "search", traced([OVER, UNDER])["record"])
    put("reference", 0, "final", record([UNDER]))
    (tmp_path / "attack" / "0" / "search" / "rows.jsonl").write_text("not a record\n")
    (tmp_path / "attack" / "notes" / "final").mkdir(parents=True)
    (tmp_path / "attack" / "notes" / "final" / "run.json").write_text("{}")
    starts = "".join(f'[starts.{n}]\ndata = "{e["data"]}"\nc0_macs = {e["c0_macs"]!r}\nl0_bpb = {e["l0_bpb"]!r}\n'
                     + "shape = { " + ", ".join(f"{k} = {v}" for k, v in e["shape"].items()) + " }\n"
                     for n, e in registry()["starts"].items())
    (tmp_path / "starts.toml").write_text(starts)
    (tmp_path / "attack.toml").write_text('start = "s"\ntarget = { gain = 1.1 }\npolicy = "console"\n'
                                          'honest = "reference"\nseeds = { search = [0], final = [0, 1] }\n')
    assert G.main(["--config", str(tmp_path / "attack.toml"), "--starts", str(tmp_path / "starts.toml"), "--records",
                   str(tmp_path / "attack"), "--honest", str(tmp_path / "reference"), "--out", str(tmp_path / "out")]) == 0
    s = json.loads((tmp_path / "out" / "slowdown.json").read_text())
    assert s["schema"] == "slowdown/v1" and s["config"] == "attack" and s["p"]["honest"]["p"] == pytest.approx(2.5)
    assert s["S"]["any"] == pytest.approx(2.5 * s["rho"]["rho"])
    assert [(x["config"], x["seed"], x["split"]) for x in s["records"]] == [
        ("attack", 0, "final"), ("attack", 0, "search"), ("attack", 1, "final"), ("reference", 0, "final")]
    assert all("form" not in x and x["source"].endswith("run.json") for x in s["records"])
    assert capsys.readouterr().out.startswith("SLOWDOWN attack ")
