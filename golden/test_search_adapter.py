import pytest

from golden import model, search_adapter as A

GOOD = {"tile": (1, 1), "kchunk": 2, "fuse": "dedupe", "bits": 8, "agg": 1, "noise": "perm", "lp": 64, "t": 2,
        "row": 1024, "drow": 1024, "sampling": "work", "joint": "separate", "stmt": "template", "credit": "k2048",
        "seal": "roots", "host": "fixed", "kernels": "bar", "overlap": "none", "akeys": 64}
KEYS = {"J", "serve", "prove_ratio", "prove_gpu_s", "verify_cpu_s", "feasible", "reasons", "floor_min", "max_ands",
        "lower_gib", "lower_min", "draws_values", "draws_compute"}


@pytest.mark.parametrize("point", ["t8192", "t512"])
def test_evaluate_returns_the_search_keys(point):
    r = A.evaluate(GOOD, {k: i.value for k, i in A.INPUTS.items()}, point)
    assert KEYS <= set(r) and r["feasible"], r["reasons"]
    assert r["J"] == pytest.approx(r["serve"] + r["prove_ratio"])


def test_knobs_reach_the_design():
    d = A.to_design(GOOD, "t512")
    assert d["point"]["T"] == 512 and d["partition"]["kchunk"] == 2 and d["commit"]["values"] == "dedup"
    assert d["pouw"]["L"] == 128 and d["pouw"]["L_prime"] == 64 and d["pouw"]["noise"] == "permuted"
    assert d["commit"]["hash_eff"] == model.I("hash_eff_rows") and d["commit"]["gather"] is False
    d = A.to_design({**GOOD, "ptile": (2, 4), "kmax": 28, "reps": 1, "device": "sm120", "period": 10})
    assert d["pouw"]["tile"] == (2, 4) and d["prover"]["k_max"] == 28 and d["prover"]["reps"] == 1
    assert d["prover"]["device"] == "sm120" and d["sampling"]["period"] == 10 and d["partition"]["attn_keys"] == 64
    d = A.to_design({**GOOD, "census": "b4", "udraw": "call", "mmax": 37, "retain": "host", "noise": "segment"}, "t512")
    assert d["partition"]["census"] == "b4" and d["pouw"]["u_draw"] == "call" and d["prover"]["m_max"] == 37
    assert d["sampling"]["retain"] == "host" and d["pouw"]["noise"] == "segment"


def test_a_block_past_the_cap_or_the_prover_memory_is_infeasible():
    j10r = {**GOOD, "scheme": "j10r", "ptile": (16, 16)}
    assert "k_log" in A.evaluate(j10r)["reasons"]
    r = A.evaluate({**j10r, "kmax": 29})
    assert "k_log" not in r["reasons"] and "prover_mem" in r["reasons"]


def test_the_period_divides_proving_and_not_serving():
    one, many = A.evaluate(GOOD), A.evaluate({**GOOD, "period": 1000})
    assert many["serve"] == pytest.approx(one["serve"])
    assert many["prove_ratio"] < one["prove_ratio"] / 500


def test_a_floor_under_one_minus_eps_is_infeasible():
    assert "floor" in A.evaluate({**GOOD, "t": 8, "lp": 128})["reasons"]          # 1/8 + 3/8 < 0.99
    assert "floor" in A.evaluate({**GOOD, "t": 8, "lp": 512})["reasons"]          # L 4,096 >= K 2,048: O alone
    assert "floor" in A.evaluate({**GOOD, "t": 3})["reasons"]                      # per-tile pi: 0.97 (GD-NOISE)
    assert "floor" not in A.evaluate({**GOOD, "t": 3, "noise": "segment"})["reasons"]


def test_scenario_inputs_move_j_and_ranges_hold_the_value():
    base = A.evaluate(GOOD)["J"]
    v = {k: i.value for k, i in A.INPUTS.items()}
    assert A.evaluate(GOOD, {**v, "o_per_and_h200": A.INPUTS["o_per_and_h200"].high})["J"] > base
    for k, i in A.INPUTS.items():
        assert min(i.low, i.high) <= i.value <= max(i.low, i.high), k
        assert i.uncertain == (model.INPUTS[k][1] == "E" and i.low != i.high), k
