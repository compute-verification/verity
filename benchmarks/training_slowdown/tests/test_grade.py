"""training_slowdown.grade on synthetic records: the fit, the seeds, H and p both ways, ρ at a named policy, S and what
it refuses, the bound of record beside it, the periods, and the command."""
import gzip
import json
import math
from pathlib import Path

import pytest

from training_slowdown import calculator, gates, policy
from training_slowdown import formalism as F
from training_slowdown import grade as G

FIXTURES = Path(__file__).parent / "fixtures"
#: the scaling law the synthetic starts sit on, and the start every config here branches from, the 45M reference's shape
E, A, GAMMA = 0.4, 20.0, 0.07
SHAPE = policy.SHAPES[policy.REFERENCE_PARAMS] | {"ctx": 1024, "parameters": policy.REFERENCE_PARAMS}
PER_MAC = gates.per_mac("gemm/bf16")
#: one step's log (a product of 80 MACs) and its gates
LOG, STEP_GATES = [["fwd", "theta", "mm", "[[8]]", 1, 80]], 80 * PER_MAC
#: a traced job's steps a checkpoint: past the first steps the products term leaves out, those a block of k bits holds
PER = 16


def law(c0_macs: float) -> float:
    return E + A * (c0_macs * PER_MAC) ** -GAMMA


def registry() -> dict:
    """A starts registry as `starts.toml` parses: four starts on the law, one off its frontier and one on other data."""
    on = {name: {"data": "d", "shape": SHAPE, "c0_macs": c, "l0_bpb": law(c)}
          for name, c in (("s", 1e15), ("m", 1e16), ("l", 1e17), ("x", 1e18))}
    return {"data": {"d": {}}, "starts": on | {"off": {"data": "d", "shape": SHAPE, "c0_macs": 5e16, "l0_bpb": 1.4},
                                                "far": {"data": "e", "shape": SHAPE, "c0_macs": 1e19, "l0_bpb": 0.2}}}


L0 = law(1e15)
FIT = G.fit([(e["c0_macs"] * PER_MAC, e["l0_bpb"]) for e in registry()["starts"].values() if e["data"] == "d"])
#: the loss a gain of 1.1 from the start reaches, and losses a hair on either side of it
TARGET = G.target(FIT, L0, 1.1)
OVER, UNDER = TARGET + 1e-4, TARGET - 1e-4
CONFIG = {"name": "attack", "start": "s", "target": {"gain": 1.1}, "policy": "console", "honest": "reference",
          "seeds": {"search": [0], "final": [0, 1]}}


def record(losses: list, split: str = "final", start: str = "s", invalid=(), per: int = 1) -> dict:
    """A job's record: one checkpoint per loss after each `per` steps, so checkpoint i has counted (i + 1)·per steps."""
    steps = [{"step": i, "tokens": 512, "struct": "a", "by": {"fwd/theta": 80}} for i in range(per * len(losses))]
    return {"training_slowdown": "0", "start": {"name": start}, "split": split, "invalid": list(invalid),
            "forward_macs_per_token": 51_342_080, "setup": {"by": {}}, "steps": steps, "tail": {"by": {}},
            "structures": {"a": {"updated": [["w", 0, 80]], "log": LOG}},
            "checkpoints": [{"step": per * (i + 1) - 1, "score": {"bpb": loss, "start_bpb": L0, "split": split},
                             "e_bytes": 100, "format": "full"} for i, loss in enumerate(losses)]}


def job(losses: list, seed: int = 0, split: str = "final", **kw) -> dict:
    return {"seed": seed, "split": split, "record": record(losses, split, **kw)}


def traced(losses: list) -> dict:
    """A search job as its traced rows, a checkpoint each `PER` steps: the 45M dense step's products as each prefix's
    trace."""
    block = json.loads((FIXTURES / "dense_45m.products.json").read_text())
    rows = []
    for i, loss in enumerate(losses):
        rec = record(losses[:i + 1], "search", per=PER)
        rec["structures"]["a"] |= {"updated": [], "affected_share": 0.5, "sigma": 1.0, "token_factor": 1.0,
                                   "delta_grad_bits": 1e9}
        rec["formalism"] = {"trace": {"products": block, "data_dependent": [], "carried_scalars": [],
                                      "affected_share": 0.5, "u1_bits": 1e6}}
        rows.append({"checkpoint_step": PER * (i + 1) - 1, "val_bpb": loss, "start_bpb": L0, "start": "s", "training_slowdown": "0",
                     "formalism": F.of_record(rec), "output": {"e_bytes": 1e8}})
    return {"seed": 0, "split": "search", "rows": rows}


def graded(runs, honest, config=CONFIG, **kw) -> dict:
    return G.grade(config, registry(), runs, honest, **kw)


def test_the_fit_reads_the_frontier_of_one_data_and_recovers_its_law():
    """Four starts on L(C) = E + A·C^(−γ) give it back; the start off the frontier (more compute, a worse loss) and the
    other data's are not read; under five points the fit is provisional, and under three there is none. A target is
    the gain's inverse, and a loss at or below E has no finite gain."""
    assert FIT["points"] == 4 and FIT["provisional"]
    assert (FIT["E"], FIT["A"], FIT["gamma"]) == pytest.approx((E, A, GAMMA), rel=1e-4)
    assert FIT["rms_pct"] == pytest.approx(0, abs=1e-3)
    assert G.gain_from(FIT, L0, TARGET) == pytest.approx(1.1) and TARGET < L0
    assert G.target(FIT, FIT["E"], 1.1) is None and G.gain_from(FIT, L0, FIT["E"]) == math.inf
    assert G.fit([(1e18, 1.2), (1e19, 1.1)]) is None and G.fit([(1e18, 1.2), (2e18, 1.1), (3e18, 1.0)]) is None
    three = G.fit([(1e18, law(1e15)), (1e19, law(1e16)), (1e20, law(1e17))])
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
    want = F.published(F.measure(search["rows"][1], SHAPE), *policy.CONSOLE)
    assert want["rho"] > 1 and s["rho"]["rho"] == want["rho"] and s["rho"]["rests_on"] == want["rests_on"]
    assert (s["rho"]["checkpoint_step"], s["rho"]["at"]) == (2 * PER - 1, "its first checkpoint reaching the gain")
    assert s["rho"]["work"]["N"] > 0 and s["rho"]["work"]["C_gates"] == s["rho"]["C_gates"]
    assert s["S"] == {"honest": pytest.approx(1.5 * want["rho"]), "any": pytest.approx(1.5 * want["rho"]),
                      "why": None, "lower_bound": want["partial"]}
    assert s["rests_on"]["rho"] == want["rests_on"] and "4 of the 5 starts on data d" in s["rests_on"]["gain"]
    at = graded(runs, ref, CONFIG | {"policy": "v4-pro-session"})
    assert at["policy"]["kind"] == policy.FIXED and at["rho"]["rho"] is None and at["S"]["any"] is None
    assert at["S"]["why"] == at["rho"]["why"] and at["p"] == s["p"] and at["per_period"] is None


def test_no_rho_without_a_traced_search_record():
    ref = [job([UNDER])]
    assert "no search record" in graded([job([UNDER])], ref)["S"]["why"]
    s = graded([job([UNDER]), job([OVER, UNDER], split="search")], ref)
    assert s["rho"] == {"rho": None, "why": "unmeasured: the search record is untraced, and ρ reads a traced step"}
    assert s["S"]["any"] is None and s["p"]["any"]["p"] == 1


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


def test_the_bound_of_record_stands_beside_s_and_the_periods_are_the_calculators():
    """The bound of record is the calculator's or absent with the reason, never a number of the grade's. The run's AND
    gates at S are C·ρ at a bf16 GEMM's gates an AND, and each share is over a fleet of the calculator's at its
    peak for the census's window and core's periods."""
    s = graded([job([OVER, UNDER], 0), job([OVER, UNDER], 1), traced([OVER, UNDER])], [job([UNDER])])
    bound = s["bound_of_record"]
    assert not bound["available"] and bound["why"].startswith("not available: ") and calculator.THEOREM in bound["why"]
    assert (bound["c_gates"], bound["k_bits"]) == policy.CONSOLE
    per = s["per_period"]
    assert per["run_and_gates"] == pytest.approx(2 * STEP_GATES * s["rho"]["rho"] / gates.per_and("gemm/bf16"))
    assert [(x["over"], x["period"]) for x in per["rows"]] == [
        (over, p) for over in ("one datacenter", "the world") for p in ("window", "week", "month", "year")]
    fleets, periods = {f["over"]: f for f in calculator.fleets()}, calculator.periods()
    assert periods["window"] == calculator.window()[1] < periods["week"] < periods["month"] < periods["year"]
    for x in per["rows"]:
        assert x["share"] == pytest.approx(per["run_and_gates"] / (fleets[x["over"]]["and_per_s"] * periods[x["period"]]))
    dc = fleets["one datacenter"]
    assert dc["and_per_s"] == pytest.approx(dc["flop_per_s"] / 2 * 9128 / 32) and set(per["parts"]) == {
        "measured", "proved", "scaled"}
    block = G.text(s)
    assert "bound of record   not available: " in block and "per period        the run at S: " in block
    assert block.startswith("SLOWDOWN attack   start s, gain 1.1x, policy console (rule relative to the start: ")


def test_a_records_rows_carry_its_prefix_and_the_command_reads_a_tree_of_jobs(tmp_path, capsys):
    """`rows_of` gives each checkpoint its prefix's op log, the last one the whole run's. The command reads
    `<seed>/<split>/run.json`, or a traced record's `rows.jsonl.gz` in its place, prints the block and writes
    `slowdown.json`."""
    rows = G.rows_of(record([OVER, UNDER]))
    assert [r["macs_log"] for r in rows] == [{"by": {"fwd/theta": 80 * n}, "ops": {"fwd theta mm [[8]]": [n, 80 * n]}}
                                             for n in (1, 2)]
    assert rows[1]["output"] == {"e_bytes": 100, "format": "full"} and rows[1]["split"] == "final"

    def put(root: str, seed: int, split: str, losses: list) -> None:
        d = tmp_path / root / str(seed) / split
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(record(losses, split)))

    put("attack", 0, "final", [OVER, UNDER])
    put("attack", 1, "final", [OVER, OVER, UNDER])
    put("attack", 0, "search", [OVER, UNDER])
    put("reference", 0, "final", [UNDER])
    with gzip.open(tmp_path / "attack" / "0" / "search" / "rows.jsonl.gz", "wt") as fh:
        fh.write("\n".join(json.dumps(r) for r in traced([OVER, UNDER])["rows"]) + "\n")
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
    assert [(x["config"], x["seed"], x["split"], x["form"]) for x in s["records"]] == [
        ("attack", 0, "final", "record"), ("attack", 0, "search", "rows"), ("attack", 1, "final", "record"),
        ("reference", 0, "final", "record")]
    assert capsys.readouterr().out.startswith("SLOWDOWN attack ")
