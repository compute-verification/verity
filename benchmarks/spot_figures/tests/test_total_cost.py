"""Figure 1 in its total-cost form: the configurations and their serve-tax card, the registrations' spans, the write form
against every admissible set of a small design, the break-even's search, and one scope end to end with a measured tax
dropped in."""
import json
from itertools import product
from math import comb

import cards
import figure1 as f1
import pytest
import total_cost as t
from verity.protocols.spot.device import challenger
from verity.protocols.spot.optimize import Cell


def test_the_configurations_are_the_cards():
    """Every configuration that pays its own serve tax has its slot in the serve_tax card; one on another cut of the
    circuit pays one of theirs."""
    assert tuple(n for n, c in t.CONFIGURATIONS.items() if c.serve is None) == cards.CONFIGURATIONS
    assert all(t.CONFIGURATIONS[c.serve].serve is None for c in t.CONFIGURATIONS.values() if c.serve)
    assert [n for n, c in t.CONFIGURATIONS.items() if "skipped_work" in c.panels] == ["every_block"]


def test_a_request_line_is_plotted_at_every_registration_and_compared_in_the_sets_form():
    """(A), (A′) and (A″) in the patterns' form at each registration, and in the sets' form at their own, which only
    exfiltration plots; a registration's GPU-seconds are its machines' over the census window.  Each replays its whole
    batch, and (A) replayed alone, which needs batch-invariant kernels, is a comparison at one machine's that pays
    their tax."""
    for name in ("tokens", "per_expert", "head_units"):
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
    overhead is its blocks' seconds over its MACs at the card's peak; a block neither measured nor staged is refused.
    A table holds a measured block's proof blocks at the staging the card measured it at, whatever the strata estimate,
    so the card's statement is as many tables on every line as on the card."""
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
                assert x["table_blocks"] == t.STATEMENT_BITS >> by_name[x["block"]]["k_log"]
            if x["priced"] == "computed":
                rates.setdefault(x["seconds_per_block"] / 2 ** x["k_log"], x["block"])
                assert x["statement_blocks"] == t.STATEMENT_BITS >> x["k_log"] == x["table_blocks"]
        other = [x for x in strata if x["block"] in by_name and x["k_log"] != by_name[x["block"]]["k_log"]]
        assert other and all(x["table_blocks"] != t.STATEMENT_BITS >> x["k_log"] for x in other)
        variant = next(x for x in strata if x["priced"] == "measured" and x["block"] not in by_name)
        assert t._definition(variant["block"]) in {t._definition(b) for b in by_name}
        macs = sum(x["per_token"] * x["macs"] for x in strata)
        assert sum(x["per_token"] * f1.price(x) for x in strata) == pytest.approx(macs, rel=1e-12)
        seconds = sum(x["per_token"] * x["seconds_per_block"] for x in strata)
        assert p["proof_overhead"].value == pytest.approx(seconds * peak / macs, rel=1e-12)
    assert len({round(r, 20) for r in rates}) == 1, rates
    reads = [x for x in lines["per-expert"]["partition"].value["strata"] if x.get("kind") == "EmbeddingRead_v1"]
    assert reads and all(x["macs"] == 0 < f1.price(x) and x["priced"] == "measured" for x in reads)
    assert 0 < lines["registered"]["partition"].value["computed_share"] < .03
    assert lines["per-expert"]["proof_overhead"].value < lines["registered"]["proof_overhead"].value
    stray = c["partition"].value | {"strata": [{"block": "Unstaged_v1{K=3}", "per_token": 1, "macs": 3, "out_bits": 16}]}
    with pytest.raises(ValueError, match="no k_log"):
        t.priced(c | {"partition": f1.InputCard(stray, "", "measured", "one unstaged block")}, po)


def test_a_block_measured_at_another_size_of_its_definition_scales_with_its_k_log():
    """A residual add of 384 columns, which the card measures only at 6144, 2^4 times smaller as the strata size both:
    the card's seconds a block over 2^4 and its cap and a table's blocks times 2^4, its statement's fixed seconds
    unchanged."""
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
    assert got["table_blocks"] == t.table_blocks(b["k_log"]) * 16


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


def test_a_priced_tax_is_the_sum_of_its_named_parts():
    """Each priced configuration of the kept card is the sum of its parts, each with how it got its share; (A)'s names
    the step record, the two-stage boundary commitment and the history tables beside the tick, the edge snapshot and
    the top-k; and a card whose total is not its parts', or with a part that doesn't say how it was priced, is
    refused."""
    kept = t.taxes(cards.load("serve_tax", "GLM-5.3"))
    for name, e in kept.items():
        assert e["status"] == "priced" and {p["status"] for p in e["parts"]} <= set(cards.PART_STATUS), name
        for total, key in (("tax", "share"), ("low", "low"), ("high", "high")):
            assert e[total] == pytest.approx(sum(p[key] for p in e["parts"]), rel=1e-12), (name, total)
    assert {p["part"]: p["status"] for p in kept["tokens"]["parts"]} == {
        "certified-tick rule": "simulated", "step record": "modeled", "two-stage boundary commitment": "modeled",
        "history tables": "modeled", t.EDGE: "measured", "deterministic top-k": "modeled"}
    assert kept["tokens_bi"]["parts"][:-1] == kept["tokens"]["parts"]

    def changed(**part) -> dict:
        card = t.priced_card("GLM-5.3")
        card["value"]["configurations"][0]["parts"][1] |= part
        return card

    with pytest.raises(ValueError, match="tax: .* but its parts' shares add to .*: a priced configuration is the sum"):
        cards.check(changed(share=.01))
    with pytest.raises(ValueError, match="status: 'guessed', but it is one of 'modeled', 'simulated', 'measured'"):
        cards.check(changed(status="guessed"))
    with pytest.raises(ValueError, match="'certified-tick rule' given 2 times"):
        cards.check(changed(part="certified-tick rule"))


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
    row = {"merged": 16, "k": {0: 2, 64: 1}, "registration_sample": 0}
    assert t.computed_proving(cells, row, (True, False)) == pytest.approx(52 / 53, rel=1e-12)
    assert (t.computed_proving(cells, row, (False, False)), t.computed_proving(cells, row, (True, True))) == (0, 1)
    with pytest.raises(ValueError):
        t.computed_proving(cells, row, (True,))


def test_a_by_template_designs_strata_are_its_cells():
    """A by-template design's row has no merged size class (its `merged` is "template"): each cell is a stratum by
    its position (`figure1.split_cells`), so its write form and computed share are those of the width-stratified row
    with the same strata, and the series carry its counts by position."""
    cells = [Cell(100, 16, prove=1.0, fixed=50.0, cap=1000), Cell(1, 64, stratum=64, prove=1.0, cap=10)]
    width = {"merged": 16, "k": {0: 2, 64: 1}, "q": 1e-3, "registration_sample": 0}
    template = {"design": "template", "merged": "template", "k": {0: 2, 1: 1}, "q": 1e-3, "registration_sample": 0}
    assert [c.stratum for c in f1.split_cells(cells, "template")] == [0, 1]
    assert [c.stratum for c in f1.split_cells(cells, 16)] == [0, 64]
    assert t.computed_proving(cells, template, (True, False)) == t.computed_proving(cells, width, (True, False))
    assert t.write_form(cells, template, 4, .01, 500) == t.write_form(cells, width, 4, .01, 500) > 0
    assert f1._plan(template) == {"merged": "template", "q": 1e-3, "k": {"0": 2, "1": 1}}
    assert f1._plan(width)["k"] == {"<=16": 2, "64": 1}


def test_the_designs_beside_a_line_say_what_no_session_holds():
    """`beside_parts` prints one session whatever it holds with its bound only while its largest session is within
    the tables a session of the proof system holds, and says so past it; a design with several sessions a block type
    comes with its sessions and the seconds between them over the verification phase; one with no design says so;
    and one that meets a limit of today's second-stage code is out of reach, the limit said before its bound."""
    row = {"y": 8e12, "count": 8e12, "registration_sample": 4, "sessions": 3.0, "largest": 512, "tables": 2.0,
           "over": 1, "status": "modeled"}
    got = {"template": row | {"held": True}, "table": None, "tables": {"1": row | {"sessions": 120.0}, "16": None}}
    held, table, one, sixteen = t.beside_parts(got, 600.0)
    assert held == (f"{t.ONE_SESSION['template']}, 1 TB at K = 4, its largest session 512 proof blocks, 1 block "
                    "types' sessions more than one table and the largest of them 2 tables of its type")
    assert table == f"{t.ONE_SESSION['table']}, no design within the budget"
    assert one == ("sessions of at most 1 table a block type, 1 TB at K = 4 in 120 sessions, one each 5 s if they "
                   "run one at a time")
    assert sixteen == "sessions of at most 16 tables a block type, no design within the budget"
    past = t.beside_parts(got | {"template": row | {"held": False, "tables": 200.5, "largest": 51_328}}, 600.0)[0]
    assert past == (f"{t.ONE_SESSION['template']}, out of reach: its largest session is 51,328 proof blocks and its "
                    f"largest in tables 201 of its type, and a session of the proof system holds at most "
                    f"{t.SESSION_TABLES}")
    assert "TB" not in past
    limit = "its largest session in tables is 2 of its type, past the GPU prover's 1"
    met = t.beside_parts({"template": row | {"held": True, "limit": limit}, "table": row | {"limit": None},
                          "tables": {"16": row | {"limit": limit}}}, 600.0)
    assert met[0].startswith(f"{t.ONE_SESSION['template']}, out of reach ({limit}); modeled, 1 TB at K = 4, ")
    assert met[1].startswith(f"{t.ONE_SESSION['table']}, 1 TB at K = 4, ")
    assert met[2] == (f"sessions of at most 16 tables a block type, out of reach ({limit}); modeled, 1 TB at K = 4 in "
                      "3 sessions, one each 200 s if they run one at a time")


def test_a_design_names_the_first_limit_of_the_second_stages_code_it_meets(monkeypatch):
    """`limits` counts what a design asks of the code that runs a second stage: the lines of session coins its
    sessions write (2 a session and `TABLE_LINES` a table, at least) and the bytes its second draw reads (a stream of
    2 k width(n) + 64 for each block type in each sampled replay block), and names the first limit met: a session's
    tables, then the release log's lines, then one layout's bytes; none where it is within all three."""
    cells = [Cell(100.0, 16.0, prove=1e-3, fixed=5.0, cap=64, commit_bytes=2.0, table=64),
             Cell(4.0, 4096.0, prove=1e-2, fixed=5.0, cap=8, commit_bytes=512.0, table=4)]
    row = {"merged": "template", "form": "statements", "registration_sample": 3, "k": {0: 10, 1: 2}, "tables": 2.0}
    got = t.limits(cells, row)
    assert t.draw_width(100) == (7 + 64 + 7) // 8 == 9 and t.draw_width(4) == (3 + 64 + 7) // 8
    assert got["draw_bytes"] == 3 * ((2 * 10 * 9 + 64) + (2 * 2 * 9 + 64))
    # each type's sampled blocks of the phase are one session: 30 blocks within a table of 64, 6 blocks 1.5 tables of 4
    assert got["lines"] == pytest.approx(2 * 2 + t.TABLE_LINES * (1 + 1.5))
    assert got["limit"] == "its largest session in tables is 2 of its type, past the GPU prover's 1"
    one = row | {"tables": 1.0}
    assert t.limits(cells, one)["limit"] is None
    monkeypatch.setattr(t, "DRAW_BYTES", got["draw_bytes"] - 1)
    assert t.limits(cells, one)["limit"] == (f"its second draw reads {got['draw_bytes']:,.0f} bytes of live coins, and "
                                             f"one layout of the challenger's holds {t.DRAW_BYTES:,}")
    monkeypatch.setattr(t, "LOG_LINES", got["lines"] - 1)
    assert t.limits(cells, one)["limit"] == (f"its sessions write at least {got['lines']:,.0f} lines of session "
                                             f"coins, and one challenger session's release log holds {t.LOG_LINES:,}")
    past = t.limits(cells, row | {"tables": 200.5})["limit"]
    assert past == ("its largest session in tables is 201 of its type, past the GPU prover's 1, the CPU prover's and "
                    "the Lean verifier's 64, the challenger's coin commitment's 127 and the proved scope's 200")
    assert [n for n, _ in t.TABLE_LIMITS] == [1, 64, 127, t.SESSION_TABLES]


def test_the_challengers_limits_are_the_challenger_modules():
    """The three limits `limits` holds a design to on the challenger's side are the challenger's own constants, read
    from its module and never restated: a release log's coin-tree lines, one commitment's coins in bytes, and the
    most tables of a session whose coin tree, one stream a session and two a table at the inner session's rounds and
    block, the challenger still samples (`LiveChallenger.sample_tree`'s inequality)."""
    assert t.LOG_LINES == challenger.MAX_TREE_LINES
    assert t.DRAW_BYTES == challenger.MAX_COINS * challenger.COIN_BYTES

    def sampled(tables: int) -> bool:
        return (1 + 2 * tables) * t.STREAM_COINS <= challenger.MAX_TREE_COINS

    (tree,) = [n for n, what in t.TABLE_LIMITS if "challenger" in what]
    assert tree == t.TREE_TABLES and sampled(tree) and not sampled(tree + 1)
    assert t.STREAM_COINS <= (1 << challenger.MAX_ROUNDS_LOG) * challenger.MAX_BLOCK


def test_a_lines_counts_are_one_whole_count_a_block_type(tmp_path):
    """`counts` writes a two-stage line's design as the verifier's two-stage rule takes it: the whole count of a
    verification phase's requests the first stage samples and one whole count, at least one, for each block type of
    the line's cut, by the cut's names in its order, with the phase's size and the sessions the design prices, more
    than one a block type.  Beside it are the two designs with one session a block type: the one whose session holds
    whatever it samples bounds less in sessions of many tables, more than a session of the proof system holds, and
    the one held to one table a session, the form the verifier's second stage reads, bounds more; and the designs
    with sessions of at most 1, 16, 64 and 200 tables, fewer sessions and a smaller bound as a session holds more.
    Each design carries the first limit of today's second-stage code it meets: at a datacenter's verification phase
    only the design held to one table a session meets none.  A β that pays for no sampled request of the line's
    verification phase has no counts, and neither has (C), which samples one-stage."""
    out = t.counts("GLM-5.3", "tokens_datacenter", "world year", 0.1)
    strata = t._lines("GLM-5.3", "census")[4]["tokens_datacenter"]["partition"].value["strata"]
    assert list(out["k"]) == [x["block"] for x in strata if x["per_token"] > 0]
    assert all(isinstance(k, int) and k >= 1 for k in (out["sampled"], *out["k"].values()))
    phase = out["verification_phase"]
    assert phase["machines"] > t.POD and out["sampled"] <= phase["replay_blocks"]
    assert out["sessions"] > len(out["k"]) and out["bound_bits"] > 0 and out["largest"] >= 1
    whole, table = (out["beside"][x] for x in t.ONE_SESSION)
    assert whole["sessions"] == pytest.approx(len(out["k"])) == pytest.approx(table["sessions"])
    assert whole["over"] >= 1 and whole["tables"] > t.SESSION_TABLES and not whole["held"]
    assert table["over"] == 0 and table["tables"] <= 1
    assert whole["count"] < out["bound_bits"] < table["count"]
    several = out["beside"]["tables"]
    assert list(several) == [str(j) for j in t.TABLES] == ["1", "16", "64", "200"]
    assert several["1"]["over"] == 0 and all(several[str(j)]["tables"] <= j for j in t.TABLES)
    sessions, bounds = ([several[str(j)][key] for j in t.TABLES] for key in ("sessions", "count"))
    assert sessions == sorted(sessions, reverse=True) and sessions[-1] > len(out["k"])
    assert table["count"] > bounds[0] and bounds == sorted(bounds, reverse=True) and bounds[-1] > whole["count"]
    assert table["limit"] is None and "the proved scope's 200" in whole["limit"]
    assert "lines of session coins" in several["1"]["limit"]
    assert several["16"]["limit"].endswith("16 of its type, past the GPU prover's 1")
    assert several["200"]["limit"].endswith("the challenger's coin commitment's 127")
    assert out["limit"] and out["lines"] > t.LOG_LINES and out["draw_bytes"] > 0
    assert (out["schema"], out["line"], out["scope"], out["beta"]) == (t.COUNTS_SCHEMA, "tokens_datacenter",
                                                                      "world year", 0.1)
    t.main(["counts", "--scope", "world year", "--beta", "0.1", "--out", str(tmp_path / "counts.json")])
    assert json.loads((tmp_path / "counts.json").read_text()) == out
    with pytest.raises(ValueError, match="no design within"):
        t.counts("GLM-5.3", "tokens", "world year", 0.01)
    with pytest.raises(ValueError, match="no two-stage line"):
        t.counts("GLM-5.3", "every_block")


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
    above the served bits, saying which term binds, a wider verification phase never above a narrower one, sampling
    a whole count of its requests or, where one is over the budget, none, at the served bits with no plan, and the
    patterns' form below the sets' wherever it doesn't reach the served bits;
    both infiltration lines plotted, and (C)'s falling with β; skipped work only for (C); the captions name what each
    rests on and the lines the served bits bound at β = 1%; each two-stage point with a design carries its second
    stage's coins and Lean's bound on its refusing an honest run, and one with none carries none; and it renders.  It
    plans every
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
    for name in ("tokens", "per_expert", "head_units"):
        spans = [lines[n]["y"] for n in (name, f"{name}_pod", f"{name}_datacenter")]
        assert all(b <= a * (1 + 1e-9) for x in zip(*spans) for a, b in zip(x, x[1:])), name
        patterns, sets = lines[name]["y"][at], lines[f"{name}_sets"]["y"][at]
        assert patterns <= sets and (patterns < sets or lines[name]["binds"][at] == "served"), name
        for n in (name, f"{name}_pod", f"{name}_datacenter"):
            s = lines[n]
            assert all(0 <= p <= x <= 1 for x, p in zip(s["shared"], s["padding"]))
            for k, one, binds, plan, share, sessions in zip(s["registration_sample"], s["sample_budgets"], s["binds"],
                                                            s["plan"], s["computed_proving"], s["sessions"],
                                                            strict=True):
                assert isinstance(k, int) and k >= 0 and (one <= 1) == (k >= 1), n
                assert (binds, plan, share) == ("served", None, None) if k == 0 else 0 < share < 1, n
                assert sessions == 0 if k == 0 else plan["merged"] == "template" and sessions > 0, n
                assert k == 0 or all(v == int(v) >= 1 for v in plan["k"].values()), n
        samples = [lines[n]["registration_sample"][at] for n in (name, f"{name}_pod", f"{name}_datacenter")]
        assert samples == sorted(samples), name
    assert "samples a whole number K of its requests" in caps["exfiltration"]
    assert all(f"{f1.ROLES} {f1.TRUST}" in c for c in caps.values())
    served_at = [n for n, s in lines.items() if s["binds"][at] == "served"]
    assert "tokens" in served_at and "tokens_datacenter" not in served_at
    assert lines["tokens"]["registration_sample"][at] == 0 < lines["tokens_datacenter"]["registration_sample"][at]
    assert (f"no design within the budget: one sampled replay block costs "
            f"{lines['tokens']['sample_budgets'][at]:,.3g} budgets;") in caps["exfiltration"]
    assert caps["exfiltration"].count("bounded here only by everything the scope serves") == 1
    s = lines["tokens_datacenter"]
    assert f"{t._share(s['computed_proving'][at])} of its sampled request's proving" in caps["exfiltration"]
    assert (f"K = {s['registration_sample'][at]:,} sampled, in {s['sessions'][at]:,.0f} sessions a verification "
            f"phase for {len(s['plan'][at]['k']):,} block types") in caps["exfiltration"]
    assert s["sessions"][at] > len(s["plan"][at]["k"])
    assert "reads one session of one table for each template of a verification phase" in caps["exfiltration"]
    forms = s["forms"]
    assert [f is not None for f in forms] == [b == f1.RATIO_BUDGET for b in out["betas"]]
    whole, table = (forms[at][x] for x in t.ONE_SESSION)
    assert whole["status"] == table["status"] == "modeled" and whole["y"] <= served
    assert whole["sessions"] == pytest.approx(len(s["plan"][at]["k"])) == pytest.approx(table["sessions"])
    assert whole["over"] >= 1 and whole["tables"] > 1 and table["over"] == 0 and table["tables"] <= 1
    assert whole["count"] < s["count"][at] < table["count"]
    assert whole["held"] == (whole["tables"] <= t.SESSION_TABLES)
    assert all(f["status"] == "modeled" for f in forms[at]["tables"].values())
    parts = t.beside_parts(forms[at], out["inputs"]["window"]["value"])
    assert len(parts) == len(t.ONE_SESSION) + len(t.TABLES)
    assert " beside it, modeled at the same seconds a proof block: " + "; ".join(parts) + ";" in caps["exfiltration"]
    assert (f"{t.ONE_SESSION['table']}, {t._size(table['y'])} at K = {table['registration_sample']:,}, its largest "
            f"session {table['largest']:,} proof blocks, within one table") in parts
    assert "forms" in lines["tokens"] and "forms" not in lines.get("every_block", {})
    assert all(c["design"] == "two-stage, by template" for n, c in out["configurations"].items() if n != "every_block")
    for n in ("tokens", "tokens_datacenter", "per_expert_datacenter", "head_units_datacenter", "layer_of_token"):
        for k, c in zip(lines[n]["registration_sample"], lines[n]["second_stage"], strict=True):
            assert (c is None) == (k == 0), n
            assert c is None or c["sample"] == k and (c["whole"] or (
                c["stages"] >= 1 and 0 < c["refusal"] == k * c["sampled"] / 2.0 ** 64
                and c["read"] > 64 * c["cells"])), n
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
