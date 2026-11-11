"""Figure 1 in its total-cost form: the configurations and their serve-tax card, the registrations' spans, the write form
against every admissible set of a small design, the break-even's search, and one scope end to end with a measured tax
dropped in."""
from itertools import product
from math import comb

import cards
import figure1 as f1
import pytest
import total_cost as t
from verity.core.protocols.spot.optimize import Cell


def test_the_configurations_are_the_cards():
    """Every configuration that pays its own serve tax has its slot in the serve_tax card; one on another cut of the
    circuit pays one of theirs."""
    assert tuple(n for n, c in t.CONFIGURATIONS.items() if c.serve is None) == cards.CONFIGURATIONS
    assert all(t.CONFIGURATIONS[c.serve].serve is None for c in t.CONFIGURATIONS.values() if c.serve)
    assert [n for n, c in t.CONFIGURATIONS.items() if "skipped_work" in c.panels] == ["every_block"]


def test_a_request_line_is_plotted_at_every_registration_and_compared_in_the_sets_form():
    """(A) and (A′) in the patterns' form at each registration, and in the sets' form at their own, which only
    exfiltration plots; a registration's GPU-seconds are its machines' over the census window.  Each replays its whole
    batch, and (A) replayed alone, which needs batch-invariant kernels, is a comparison at one machine's that pays
    their tax."""
    for name in ("tokens", "per_expert"):
        lines = {n: c for n, c in t.CONFIGURATIONS.items()
                 if (n == name or c.variant_of == name) and c.replay != "request"}
        assert {c.replay for c in lines.values()} == {"block"}
        assert sorted(c.registration for c in lines.values() if c.form == "patterns") == sorted(t.REGISTRATIONS)
        (sets,) = [c for c in lines.values() if c.form == "sets"]
        assert sets.registration == t.CONFIGURATIONS[name].registration and sets.panels == ("exfiltration",)
        assert all(c.label == t.CONFIGURATIONS[name].label for c in lines.values())
    (bi,) = [n for n, c in t.CONFIGURATIONS.items() if c.replay == "request"]
    assert (t.CONFIGURATIONS[bi].variant_of, t.CONFIGURATIONS[bi].registration, t.CONFIGURATIONS[bi].serve) == (
        "tokens", "machine", None)
    assert t.CONFIGURATIONS["layer_of_token"].replay == "step"
    c = f1.cards("GLM-5.3")
    window, gpus = c["window"].value, c["machine_gpus"].value
    assert [t.registration_gpu_seconds(r, c) for r in t.REGISTRATIONS] == [
        gpus * window, t.POD * gpus * window, c["datacenter_machines"].value * gpus * window]


def _falling(gap: float = 1.0):
    """A bound of 1e6 bits times the overhead's multiple over β, Lean's `gap` times the planner's."""
    def bound(line, scale, beta, lean=False):
        y = 1e6 * scale / beta * (gap if lean else 1.0)
        return {"y": y, "count": y, "binds": "count"} | ({"lean": True, "planner": y / gap} if lean else {})
    return bound


def test_a_break_even_bisects_to_its_target_and_says_when_none_is_in_reach(monkeypatch):
    """The overhead's multiple at which a bound rising in it reaches the target, and the β at which a bound falling in
    β does, each within its tolerance on the side where the bound is at most the target; None when the span's best end
    is above the target, the end when its worst is below; and the floor at the least overhead."""
    monkeypatch.setattr(t, "_bound", _falling())
    h = t._solve(("h", 1e3, None, 0.01, 1e-3))
    assert h == pytest.approx(1e-5, rel=2e-3) and h <= 1e-5
    beta = t._solve(("beta", 1e3, None, 0.01, 1e-3))
    assert beta == pytest.approx(1e3, rel=2e-3) and beta >= 1e3
    assert t._solve(("h", 1e-12, None, 0.01, 1e-3)) is None and t._solve(("h", 1e30, None, 0.01, 1e-3)) == 1e2
    assert t._solve(("beta", 1e-12, None, 0.01, 1e-3)) is None
    assert t._solve(("h", None, None, 0.01, 1e-3)) == pytest.approx(1e6 * 1e-12 / 0.01)
    be = {"beta": 0.01, "targets": {"FP8": 1e3}, "spans": t.SPANS,
          "rows": [{"config": "tokens", "registration": "pod", "scope": "world week", "overhead": 2.0, "floor": 8e6,
                    "h": {"FP8": 2e-5}, "beta": {"FP8": None}}]}
    assert t.breakeven_table(be).splitlines()[1] == "(A) | pod | world week | 2 | 1 MB | 2e-05 | none"


def test_lean_certifies_a_break_even_at_or_past_the_solved_point(monkeypatch):
    """Lean's certificate is the solved point where Lean's bits are at most the target, else the first step outward
    where they are (up in β, down in the overhead), each step twice the last, and none past `STEPS` steps; the floor
    is Lean's bound at the least overhead."""
    monkeypatch.setattr(t, "_bound", _falling(1.0025))
    beta = t._solve(("beta", 1e3, None, 0.01, 1e-3))
    cert = t._certify(("beta", 1e3, None, 0.01, 1e-3, beta))
    assert cert["at"] >= 1002.5 and cert["at"] == pytest.approx(beta * 1.001 ** (2 ** cert["steps"] - 1))
    assert cert["steps"] >= 2 and cert["at"] / 1002.5 <= 1.001 ** 2 ** (cert["steps"] - 1)
    assert cert["y"] <= 1e3 and (cert["lean"], cert["binds"]) == (True, "count")
    h = t._solve(("h", 1e3, None, 0.01, 1e-3))
    cert = t._certify(("h", 1e3, None, 0.01, 1e-3, h))
    assert cert["at"] <= 1e-5 / 1.0025 and cert["y"] <= 1e3
    assert t._certify(("beta", 1e3, None, 0.01, 1e-3, None)) is None
    monkeypatch.setattr(t, "_bound", _falling(1.1))
    cert = t._certify(("beta", 1e3, None, 0.01, 1e-3, t._solve(("beta", 1e3, None, 0.01, 1e-3))))
    assert (cert["at"], cert["steps"]) == (None, t.STEPS) and cert["y"] > 1e3
    floor = t._certify(("h", None, None, 0.01, 1e-3, 1.0))
    assert floor["y"] == pytest.approx(1.1e6 * 1e-12 / 0.01) and floor["at"] == t.SPANS["h"][0]


def test_the_break_even_table_shows_what_lean_certifies(monkeypatch):
    """With Lean, each patterns'-form line's row carries a certificate for its floor and each break-even, an overhead
    for `h`, and the table prints those, marking where the served bits bind and where Lean doesn't certify one."""
    def solve(job):
        what, target = job[:2]
        return 3.0 if target is None else 0.5 if what == "h" else 0.07

    def certify(job):
        what, target, *_, x = job
        return {"at": None if what == "h" and (target or 0) > 4.5e12 else x, "steps": 0, "y": 8e6,
                "binds": "served" if what == "beta" else "count", "count": 9e6, "lean": True, "planner": 8e6,
                "theta": 1.0}

    monkeypatch.setattr(t, "_solve", solve)
    monkeypatch.setattr(t, "_certify", certify)
    be = t.breakeven("GLM-5.3", ("world week",))
    assert be["lean"]["exe"] == t.plancheck.EXE
    assert [r["config"] for r in be["rows"]] == [n for n, c in t.CONFIGURATIONS.items() if c.form == "patterns"]
    for r in be["rows"]:
        assert r["lean"]["h"]["FP4"]["at"] == pytest.approx(0.5 * r["overhead"]) and r["lean"]["h"]["FP8"]["at"] is None
        assert r["lean"]["beta"]["FP8"]["at"] == 0.07 and r["lean"]["floor"]["y"] == 8e6
    row = t.breakeven_table(be).splitlines()[1].split(" | ")
    assert row[4:] == ["1 MB", "uncertified", f"{0.5 * be['rows'][0]['overhead']:.3g}", "0.07 (served)",
                       "0.07 (served)"]
    assert t.breakeven("GLM-5.3", ("world week",), lean=False)["rows"][0].get("lean") is None


def test_a_partition_card_cuts_the_circuit_and_keeps_its_macs():
    """The per-expert cut: its strata in place of the registered cut's MoE GEMMs and router, every MAC of the circuit
    kept; a card without a block's k_log is refused."""
    c = f1.cards("GLM-5.3")
    card = next(x for x in cards.load_all("partition", "GLM-5.3") if x["value"]["name"] == "per-expert")
    cut = t.cut(c, card)
    old, new = c["partition"].value["strata"], cut["partition"].value["strata"]
    kinds = set(card["value"]["replaces"])
    assert any(x.get("kind") in kinds for x in old) and not any(x.get("kind") in kinds for x in new)
    macs = sum(x["per_token"] * x["macs"] for x in old)
    assert sum(x["per_token"] * x["macs"] for x in new) == pytest.approx(macs, rel=1e-12)
    bare = [{k: v for k, v in x.items() if k != "k_log"} for x in card["value"]["strata"]]
    with pytest.raises(ValueError, match="no k_log"):
        cards.check(card | {"value": card["value"] | {"strata": bare}})


def test_every_line_prices_its_blocks_one_way():
    """Both cuts price a block at its measured type's seconds (its name's, else its Definition and K's) and an
    unmeasured type at its staged witness bits, by one rule; each stratum's price keeps a token's MACs, so the line's
    overhead is its blocks' seconds over its MACs at the card's peak; a block neither measured nor staged is refused."""
    c, po = f1.cards("GLM-5.3"), cards.load("proof_overhead", "GLM-5.3")
    card = next(x for x in cards.load_all("partition", "GLM-5.3") if x["value"]["name"] == "per-expert")
    blocks, peak = po["value"]["blocks"], po["value"]["peak_mac_s"]
    by_name = {b["block"]: b for b in blocks}
    lines = {n: t.priced(i, po) for n, i in (("registered", c), ("per-expert", t.cut(c, card)))}
    rates = {}
    for name, p in lines.items():
        strata = p["partition"].value["strata"]
        for x in strata:
            if x["block"] in by_name:
                assert x["priced"] == "measured" and x["seconds_per_block"] == by_name[x["block"]]["seconds_per_block"]
            if x["priced"] == "computed":
                rates.setdefault(x["seconds_per_block"] / 2 ** x["k_log"], x["block"])
                assert x["statement_blocks"] == t.STATEMENT_BITS >> x["k_log"]
        variant = next(x for x in strata if x["priced"] == "measured" and x["block"] not in by_name)
        assert t._definition(variant["block"]) in {t._definition(b) for b in by_name}
        macs = sum(x["per_token"] * x["macs"] for x in strata)
        assert sum(x["per_token"] * f1.price(x) for x in strata) == pytest.approx(macs, rel=1e-12)
        seconds = sum(x["per_token"] * x["seconds_per_block"] for x in strata)
        assert p["proof_overhead"].value == pytest.approx(seconds * peak / macs, rel=1e-12)
    assert len({round(r, 20) for r in rates}) == 1, rates
    reads = [x for x in lines["per-expert"]["partition"].value["strata"] if x.get("kind") == "EmbeddingRead_v1"]
    assert reads and all(x["macs"] == 0 < f1.price(x) and x["priced"] == "measured" for x in reads)
    assert 0 < lines["registered"]["partition"].value["computed_share"] < .01
    assert lines["per-expert"]["proof_overhead"].value < lines["registered"]["proof_overhead"].value
    stray = c["partition"].value | {"strata": [{"block": "Unstaged_v1{K=3}", "per_token": 1, "macs": 3, "out_bits": 16}]}
    with pytest.raises(ValueError, match="no k_log"):
        t.priced(c | {"partition": f1.InputCard(stray, "", "measured", "one unstaged block")}, po)


def test_a_block_measured_at_another_size_of_its_definition_scales_with_its_k_log():
    """A residual add of 384 columns, which the card measures only at 6144, 2^4 times smaller as the strata size both:
    the card's seconds a block over 2^4 and its cap times 2^4, its statement's fixed seconds unchanged."""
    c, po = f1.cards("GLM-5.3"), cards.load("proof_overhead", "GLM-5.3")
    b = next(b for b in po["value"]["blocks"] if b["block"] == "ResidualAdd_v1{N=6144}")
    strata = c["partition"].value["strata"]
    k = next((x["k_log"] for x in strata if x["block"] == b["block"]), b["k_log"])
    narrow = {"block": "ResidualAdd_v1{N=384}", "per_token": 16, "macs": 0, "out_bits": 6144, "k_log": k - 4}
    part = f1.InputCard(c["partition"].value | {"strata": [*strata, narrow]}, "", "measured", "a column block more")
    p = t.priced(c | {"partition": part}, po)
    got = next(x for x in p["partition"].value["strata"] if x["block"] == narrow["block"])
    assert (got["priced"], got["seconds_per_block"]) == ("measured", b["seconds_per_block"] / 16)
    assert (got["fixed_seconds"], got["statement_blocks"]) == (b["fixed_seconds"], b["statement_blocks"] * 16)


def test_the_kept_priced_taxes_are_the_generators():
    """While a configuration's tax is priced, the kept card's is what `priced_card` prices from the kept cards."""
    kept = t.taxes(cards.load("serve_tax", "GLM-5.3"))
    fresh = t.taxes(t.priced_card("GLM-5.3"))
    priced = [n for n, e in kept.items() if e["status"] == "priced"]
    for name in priced:
        assert [p["part"] for p in kept[name]["parts"]] == [p["part"] for p in fresh[name]["parts"]], name
        for k in ("tax", "low", "high"):
            assert kept[name][k] == pytest.approx(fresh[name][k], rel=1e-12), (name, k)
    assert kept["every_block"]["lower_bound"] and not kept["tokens"]["lower_bound"]
    assert kept["tokens"]["tax"] < kept["layer_of_token"]["tax"] < kept["every_block"]["tax"]


def test_a_card_without_every_configuration_is_refused():
    card = t.priced_card("GLM-5.3")
    card["value"]["configurations"] = card["value"]["configurations"][:-1]
    with pytest.raises(ValueError, match="no tax for \\['every_block'\\]"):
        t.taxes(card)
    with pytest.raises(ValueError, match="low <= tax <= high"):
        cards.check(t.priced_card("GLM-5.3") | {"value": {"baseline": "x", "configurations": [
            {"config": "tokens", "status": "priced", "tax": .1, "low": .2, "high": .3, "lower_bound": False,
             "parts": []}]}})


def test_each_statement_pays_its_verify_at_the_cards_ratio():
    """A statement's verify is its type's measured core-seconds, its family's line (through two members, or through the
    fixed seconds and one), else the measured median, at the card's core-seconds a GPU-second, added to the
    statement's fixed seconds and nothing else; a family member that isn't measured is refused."""
    vc = cards.load("verifier_cost", "GLM-5.3")
    v = vc["value"]
    by = {x["block"]: x["core_seconds"] for x in v["types"]}
    assert t.verifier_seconds(v, "FlashInferFp8Linear_v1{K=2048,N=1}") == (by["FlashInferFp8Linear_v1{K=2048,"],
                                                                         "measured")
    k4096, how = t.verifier_seconds(v, "FlashInferFp8Linear_v1{K=4096,N=1}")
    assert how == "family" and k4096 == pytest.approx(v["fixed_seconds"] + 2 * (by["FlashInferFp8Linear_v1{K=2048,"]
                                                                                 - v["fixed_seconds"]))
    lo, hi = by["SparseMlaCombine_v1{NSPLIT=2,"], by["SparseMlaCombine_v1{NSPLIT=12,"]
    assert t.verifier_seconds(v, "SparseMlaCombine_v1{NSPLIT=8,DV=512}") == (pytest.approx(lo + 6 * (hi - lo) / 10),
                                                                             "family")
    assert t.verifier_seconds(v, "MoeExpertFp8BlockGemm_v4{K=6144}") == (by["MoeExpertFp8GemmW_v1{"], "family")
    assert t.verifier_seconds(v, "Unseen_v1{K=3}")[1] == "assumed"
    c, po = f1.cards("GLM-5.3"), cards.load("proof_overhead", "GLM-5.3")
    p = t.priced(c, po)
    q = t.verified(p, vc)
    for x, y in zip(p["partition"].value["strata"], q["partition"].value["strata"]):
        assert y["fixed_seconds"] == pytest.approx(x["fixed_seconds"] + y["verify_core_seconds"]
                                                   / v["core_seconds_per_gpu_second"])
        assert y["seconds_per_block"] == x["seconds_per_block"] and y["price_macs"] == x["price_macs"]
    assert q["verifier_cost"].value == pytest.approx(5.40 / (2 * 0.015 + 8 * 0.0045))
    bad = v | {"families": [{"types": ["X"], "size": None, "members": ["Unmeasured_v1{"]}]}
    with pytest.raises(ValueError, match="none of the measured"):
        cards.check(vc | {"value": bad})


def most_written(cells, k, q, replay_blocks, delta, footprint):
    """The most bits any set of incorrect proof blocks escaping with probability at least δ writes, every replay
    block's incorrect blocks counted by stratum, each replay block charged at most `footprint`."""
    options = list(product(*(range(int(c.n) + 1) for c in cells)))

    def escape(js):
        miss = 1.0
        for c, j in zip(cells, js):
            miss *= comb(int(c.n) - j, k[c.stratum]) / comb(int(c.n), k[c.stratum])
        return 1 - q * (1 - miss)

    def charge(js):
        return min(sum(j * c.bits for c, j in zip(cells, js)), footprint)

    best = 0.0
    for blocks in product(options, repeat=replay_blocks):
        p = 1.0
        for js in blocks:
            p *= escape(js)
        if p >= delta:
            best = max(best, sum(charge(js) for js in blocks))
    return best


@pytest.mark.parametrize("q,delta,footprint", [(.5, .1, 40), (.5, .1, 1000), (.9, .3, 40), (.2, .05, 1000)])
def test_the_write_form_bounds_every_admissible_set(q, delta, footprint):
    """Three 16-bit blocks in one stratum and one 64-bit block in its own, one sampled of each: the write form is at
    least the most any admissible set writes, and within a few times of it."""
    cells = [Cell(3, 16, stratum=0), Cell(1, 64, stratum=64)]
    row = {"merged": 16, "k": {0: 1, 64: 1}, "q": q}
    want = most_written(cells, row["k"], q, 4, delta, footprint)
    got = t.write_form(cells, row, 4, delta, footprint)
    assert want <= got <= 4 * max(want, 16)


def test_the_write_form_is_at_most_every_replay_block_at_its_footprint():
    cells = [Cell(1e6, 16)]
    assert t.write_form(cells, {"merged": 16, "k": {0: 1}, "q": 1e-12}, 1000, .01, 500) == 1000 * 500


def test_a_computed_types_fixed_cost_counts_in_its_share_of_proving():
    """Two of a hundred narrow blocks sampled pay their statement's whole fixed cost, so a computed type at 2 of a
    sampled replay block's 3 marginal seconds is 52 of its 53."""
    cells = [Cell(100, 16, prove=1.0, fixed=50.0, cap=1000), Cell(1, 64, stratum=64, prove=1.0, cap=10)]
    row = {"merged": 16, "k": {0: 2, 64: 1}, "q": .5, "registration": 0.0}
    assert t.computed_proving(cells, row, (True, False)) == pytest.approx(52 / 53, rel=1e-12)
    assert (t.computed_proving(cells, row, (False, False)), t.computed_proving(cells, row, (True, True))) == (0, 1)
    with pytest.raises(ValueError):
        t.computed_proving(cells, row, (True,))


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dict:
    """One scope at four shares, with (A)'s tax measured, planned without Lean's checks (`test_figure1.py` stubs (C)'s
    and the two-stage lines', and the sampled-proofs suite runs both)."""
    mp = pytest.MonkeyPatch()
    load = cards.load

    def with_measured(slot, model, required=True):
        card = load(slot, model, required)
        if slot == "serve_tax":
            card["value"]["configurations"][0] |= {"status": "measured", "tax": .03, "low": .025, "high": .035}
        return card

    mp.setattr(t.spot_cards, "load", with_measured)
    mp.setattr(t, "BETAS", (1e-3, 1e-1, 1.0))
    out = t.run("GLM-5.3", ("datacenter week",), workers=4, lean=False)
    mp.undo()
    return out


@pytest.mark.slow
def test_each_line_starts_at_its_own_tax(measured, tmp_path):
    """The planned scope: each line's x is its tax plus β, a measured tax as it comes; the exfiltration lines never
    above the served bits, saying which term binds, a wider registration never above a narrower one, sampling more
    requests under a cap past them, and the patterns' form below the sets' wherever it doesn't reach the served bits;
    both infiltration lines plotted, and (C)'s falling with β; skipped work only for (C); the captions name what each
    rests on and the lines the served bits bound at β = 1%; and it renders.  It plans every
    configuration at four shares, about a minute on four workers, the only test of the series end to end."""
    out, scope = measured, "datacenter week"
    served = out["scopes"][scope]["served_bits"]
    assert out["taxes"]["tokens"] == out["taxes"]["tokens"] | {"status": "measured", "tax": .03}
    for panel, p in out["panels"].items():
        lines = {s["config"]: s for s in p["series"]}
        assert set(lines) == {n for n, c in t.CONFIGURATIONS.items() if panel in c.panels}
        for name, s in lines.items():
            assert s["x"] == [out["taxes"][name]["tax"] + b for b in out["betas"]]
            assert all(b <= a * (1 + 1e-9) for a, b in zip(s["y"], s["y"][1:])), (panel, name)
            if panel == "exfiltration":
                assert all(y <= served for y in s["y"])
                assert [b == "served" for b in s["binds"]] == [c >= served for c in s["count"]]
    every = next(s for s in out["panels"]["infiltration"]["series"] if s["config"] == "every_block")["y"]
    assert all(b < a for a, b in zip(every, every[1:])), "(C)'s infiltration is its plan's K, not every block's bits"
    caps = out["scopes"][scope]["captions"]
    assert out["taxes"]["per_expert"] == out["taxes"]["tokens"]
    assert out["configurations"]["per_expert"]["cut"]["label"] in caps["exfiltration"]
    assert "card_influenceTwo_le_served_min" in caps["exfiltration"] and "card_influenceTwo_le_min" in caps["exfiltration"]
    assert "the patterns' form (card_influenceTwo_le_patterns):" in caps["exfiltration"]
    assert ("each request serves at most S = 144,000 bits, and no proof block is shared across requests"
            in caps["exfiltration"])
    assert "serve tax 3%" in caps["exfiltration"] and "measured" in caps["exfiltration"]
    at = out["betas"].index(f1.RATIO_BUDGET)
    lines = {s["config"]: s for s in out["panels"]["exfiltration"]["series"]}
    for name in ("tokens", "per_expert"):
        spans = [lines[n]["y"] for n in (name, f"{name}_pod", f"{name}_datacenter")]
        assert all(b <= a * (1 + 1e-9) for x in zip(*spans) for a, b in zip(x, x[1:])), name
        patterns, sets = lines[name]["y"][at], lines[f"{name}_sets"]["y"][at]
        assert patterns <= sets and (patterns < sets or lines[name]["binds"][at] == "served"), name
        for n in (name, f"{name}_pod", f"{name}_datacenter"):
            assert all(k >= 1 and k >= q for q, k in zip(lines[n]["registration_sample"], lines[n]["registration_cap"]))
            assert all(0 <= p <= s <= 1 for s, p in zip(lines[n]["shared"], lines[n]["padding"]))
        samples = [lines[n]["registration_sample"][at] for n in (name, f"{name}_pod", f"{name}_datacenter")]
        assert samples == sorted(samples), name
    assert "sessions padded to a public cap" in caps["exfiltration"]
    assert all(f"{f1.ROLES} {f1.TRUST}" in c for c in caps.values())
    served_at = [n for n, s in lines.items() if s["binds"][at] == "served"]
    assert served_at and "tokens" not in served_at
    assert caps["exfiltration"].count("bounded here only by everything the scope serves") == 1
    for name in ("tokens", "per_expert", "layer_of_token"):
        s = lines[name]
        assert all(0 < v < 1 for v in s["computed_proving"])
        assert f"{t._share(s['computed_proving'][at])} of its sampled request's proving" in caps["exfiltration"]
    assert "everything the scope serves (" not in caps["infiltration"]
    assert "priced, not measured, a lower bound: the hashing alone" in caps["skipped_work"]
    assert "Replay is full-batch" in caps["exfiltration"] and "batch-invariant kernels (dotted" in caps["exfiltration"]
    assert all(b <= a for a, b in zip(lines["tokens"]["count"], lines["tokens_bi"]["count"]))
    assert out["taxes"]["tokens_bi"]["tax"] > out["taxes"]["tokens"]["tax"] + 0.1
    rows = t.at_budgets(out)
    assert {(r["panel"], r["config"]) for r in rows} == {(p, n) for n, c in t.CONFIGURATIONS.items() for p in c.panels}
    paths = t.render(out, scope, tmp_path)
    assert [p.name for p in paths] == ["figure1_datacenter_week.png", "figure1_datacenter_week.svg",
                                       "figure1_skipped_work_datacenter_week.png",
                                       "figure1_skipped_work_datacenter_week.svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert (t.stem("world year"), t.stem("world year", "skipped_work")) == ("figure1", "figure1_skipped_work")
