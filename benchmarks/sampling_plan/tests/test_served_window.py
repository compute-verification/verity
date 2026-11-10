"""The served window's two modes (`served_window.py`): every rung covers the window's work, the fused rungs and wide tiles
exceed Flock's block and no fitting rung has a proof block past it, proving alone picks the finest rung, the commitment moves the work optimum coarser as the pooled
period grows and along √(w β h C/(o Λ)), and leaves the exfiltration optimum where it was; a code-based commitment
hashes 1/ρ the bytes and no row in circuit, and no rung's fit turns on hm96's slots; the replay re-runs each sampled
request and moves no optimum; and the two-stage table: its one-stage design is today's, each replay grain on the rungs
inside it, serving committing only its boundaries, an incorrect layer token charged its whole work, the output cap meeting
the window's leak target, and two-stage paying off pooled over a fleet."""

from __future__ import annotations

from math import floor

import pytest

import served_window as S
from inputs import K_MAX, MASK_LOG, block_k_log, cap


@pytest.fixture(scope="module")
def out() -> dict:
    return S.run()


def _results(out: dict, **match) -> list[dict]:
    rows = [r for r in out["results"] if all(r[k] == v for k, v in match.items())]
    assert rows
    return rows


def test_every_rung_covers_the_windows_work(out):
    for name, rung in out["rungs"].items():
        assert sum(p["units"] * p["work"] for p in rung["parts"]) == pytest.approx(out["work_window"], rel=1e-9), name


def test_the_fused_rungs_and_wide_tiles_exceed_c_flocks_block_and_the_dot_level_rungs_fit(out):
    fits = {name: rung["fits"] for name, rung in out["rungs"].items()}
    assert not any(fits[n] for n in ("attn|mlp token", "layer token", "4 layers token", "tile 16x16", "tile 64x64"))
    assert all(fits[n] for n in fits if n.startswith("chunk") or n in ("coord", "tile 2x2", "tile 4x4"))


def test_no_fitting_rung_has_a_proof_block_past_c_flocks_block(out):
    """Every proof block of a fitting rung sits in a Flock block of k_log <= `K_MAX` at the composer's occupancy
    (`inputs.cap`), its statement's share that cap; a query head's whole row doesn't, so attention is key blocks, the most
    keys that fit."""
    for name, rung in out["rungs"].items():
        if rung["fits"]:
            for p in rung["parts"]:
                assert p["k_log"] <= K_MAX, (name, p["name"])
                assert cap(round(p["gates"]), p["rows"]) == p["per_statement"] > 0, (name, p["name"])
    kb = S.attention_keys()
    positions = round(S.IN["tokens"].value / S.IN["sequences"].value)
    assert block_k_log(*_rounded(S.attention_block(positions, True))) > K_MAX
    assert block_k_log(*_rounded(S.attention_block(2 * kb, False))) > K_MAX >= block_k_log(
        *_rounded(S.attention_block(kb, False)))


def _rounded(block: tuple[float, tuple[float, ...]]) -> tuple[int, tuple[int, ...]]:
    gates, rows = block
    return round(gates), tuple(map(round, rows))


def test_proving_only_picks_the_finest_rung_at_every_scope(out):
    for r in _results(out, target="work", weight=S.PROVING_ONLY):
        assert r["best_lean"] == "chunk 32", (r["scope"], r["h"])


def test_the_commitment_moves_the_work_optimum_coarser_as_the_period_grows(out):
    """The scopes are the window and core's monitoring periods (`wipe_monitoring.PERIODS`); pooled over a longer one the
    work optimum is never finer, and from a week on it is coord in both views."""
    order = list(out["rungs"])
    assert list(S.scopes()) == ["window"] + [name for name, _ in S.wipe_monitoring.PERIODS]
    for h in S.readings():
        best = [order.index(_results(out, target="work", scope=s, h=h, scheme="hm96", weight=1.0)[0]["best_lean"])
                for s in S.scopes()]
        assert order.index("chunk 32") < best[0] < best[1] and best == sorted(best), (h, best)
        for s in list(S.scopes())[1:]:
            r = _results(out, target="work", scope=s, h=h, scheme="hm96", weight=1.0)[0]
            assert r["best_lean"] == r["best_planner"] == "coord", (s, h)


def test_the_chunk_familys_optimum_tracks_the_root_law(out):
    """Between the hm96 row step's floor (one window) and the statement's steps (65,536 windows) the cheapest chunk is
    within 40% of √(w β h C/(o Λ)), or at an end of the 128-byte row step that holds it (the openings cost a block a
    step), and proving alone keeps the finest."""
    for r in out["sqrt_law"]:
        assert r["proving_only_L"] == 32
        if 16 <= r["windows"] <= 4096:
            assert 1 / 1.4 < r["best_L"] / r["sqrt_L"] < 1.4 or abs(r["best_L"] - r["sqrt_L"]) < 128, r


def test_the_leak_optimum_is_the_same_in_both_modes(out):
    for h in S.readings():
        only, both = (_results(out, target="leak", h=h, scheme="hm96", weight=w)[0] for w in (S.PROVING_ONLY, 1.0))
        assert only["best_lean"] == both["best_lean"]
        best = next(x for x in both["rows"] if x["rung"] == both["best_lean"])["lean"]
        assert best["commit"] < 1e-3 * best["total"]


def test_a_code_commitment_hashes_1_over_rho_the_bytes_and_no_row_in_circuit(out):
    """At each rate ρ every rung commits 1/ρ hm96's bytes' hashing, with the same samples, and proves cheaper by its rows'
    in-circuit hashing."""
    codes = [r for r in out["results"] if r["scheme"] != "hm96"]
    assert {r["rate"] for r in codes} == set(S.RATES)
    for r in codes:
        (hm,) = _results(out, scope=r["scope"], target=r["target"], h=r["h"], scheme="hm96", weight=r["weight"])
        for code, base in zip(r["rows"], hm["rows"]):
            assert code["rung"] == base["rung"] and code["fits"] == base["fits"]
            assert code["lean"]["commit"] == pytest.approx(base["lean"]["commit"] / r["rate"], rel=1e-9)
            assert code["lean"]["draws"] == pytest.approx(base["lean"]["draws"], rel=1e-9)
            assert code["lean"]["prove"] < base["lean"]["prove"], (r["scheme"], code["rung"])


def test_the_replay_reruns_each_drawn_request_and_moves_no_optimum(out):
    """Serving keeps no values, so each request a sampled proof block lies in is re-run from its prompt: at one window every
    request of the step (the samples far outnumber them), one stock step; pooled over a period about one request a sample; and
    the cheapest fitting rung is the one it is without the replay."""
    step, sequences = S.IN["step_s"].value, S.IN["sequences"].value
    for r in out["results"]:
        for row in r["rows"]:
            lean = row["lean"]
            assert lean["total"] == pytest.approx(lean["prove"] + r["weight"] * lean["commit"] + lean["replay"],
                                                  rel=1e-12)
            expected = step if r["scope"] == "window" else lean["draws"] * step / sequences
            assert lean["replay"] == pytest.approx(expected, rel=1e-3), (r["scope"], r["target"], row["rung"])
        without = min((x for x in r["rows"] if x["fits"]), key=lambda x: x["lean"]["total"] - x["lean"]["replay"])
        assert without["rung"] == r["best_lean"], (r["scope"], r["target"], r["h"], r["scheme"], r["mode"])


def _two_stage(out: dict, **match) -> list[dict]:
    rows = [r for r in out["two_stage"] if all(r[k] == v for k, v in match.items())]
    assert rows
    return rows


def test_the_two_stage_tables_one_stage_design_is_todays(out):
    """At every scope the main table has, each rung's one-stage design in the two-stage table is its row there (hm96,
    proving plus 1 x commit), and so is its cheapest fitting rung."""
    for r in out["two_stage"]:
        if r["scope"] not in S.scopes():
            continue
        (today,) = _results(out, scope=r["scope"], target=r["target"], h=r["h"], scheme="hm96", weight=r["weight"])
        assert r["best"]["one-stage"] == today["best_lean"]
        for row, base in zip(r["rows"], today["rows"], strict=True):
            assert row["rung"] == base["rung"] and row["one-stage"] == base["lean"]


def test_two_stage_designs_lie_on_the_rungs_inside_their_replay_blocks(out):
    """`request` holds every rung and serves the leak target only; `layer token` holds the rungs inside one layer of one
    token (no tile, no 4-layer block) and serves both."""
    for r in out["two_stage"]:
        for row in r["rows"]:
            grains = {d for d in row if d not in ("rung", "fits", "one-stage")}
            assert grains == {g for g in S.TWO_STAGE_GRAINS[r["target"]] if S.within(row["rung"], g)}
            assert ("layer token" in grains) == (row["rung"].startswith("chunk") or row["rung"] in (
                "coord", "attn|mlp token", "layer token")), row["rung"]
            assert ("request" in grains) == (r["target"] == "leak")


def test_serving_commits_only_the_replay_blocks_boundaries(out):
    """A two-stage design pays at serve time each replay block's boundary, a request's token ids or a layer token's
    residual stream with its K and V, and after the first stage the sampled replay blocks' interiors, their share of what
    one-stage commits at serve time."""
    d, kv = S.LLAMA_1B["hidden"], 2 * 2 * S.LLAMA_1B["kv_heads"] * S.LLAMA_1B["head_dim"]
    per_request = S.IN["tokens"].value / S.IN["sequences"].value + 1
    for r in out["two_stage"]:
        h = S.readings()[r["h"]]
        blocks = {"request": S.IN["sequences"].value * r["windows"], "layer token": S.token_layers() * r["windows"]}
        boundary = {"request": per_request * S.IN["token_bytes"].value, "layer token": 2 * d + kv}
        for row in r["rows"]:
            for g in set(row) & set(blocks):
                design = row[g]
                assert design["boundary"] == pytest.approx(round(blocks[g]) * boundary[g] * h, rel=1e-12)
                one = row["one-stage"]
                assert design["interior"] == pytest.approx(
                    design["replay_sample"] * design["n_rho"] / one["units"] * one["commit"], rel=1e-12, abs=0)
                assert design["replay_sample"] == pytest.approx(design["q"] * round(blocks[g]), rel=1e-12)


def test_an_incorrect_layer_token_forfeits_its_whole_work(out):
    """For free compute a replay block with one incorrect proof block is charged its whole work n_ρ s, so wherever the
    first stage samples below 1 the design proves Λ n_ρ N s / C proof blocks, n_ρ times one-stage's Λ N s / C; and
    since that is the same at every k while the replay and the interiors fall as 1/k, a sampled replay block's every
    proof block."""
    for r in _two_stage(out, target="work"):
        c = S.EPS * out["work_window"] * r["windows"]
        for row in r["rows"]:
            design = row.get("layer token")
            if design and design["q"] < 1:
                assert design["draws"] == pytest.approx(S.LAMBDA * design["n_rho"] * design["units"] * design["work"] / c,
                                                        rel=1e-9), (r["scope"], row["rung"])
                assert design["k"] == floor(design["n_rho"]), (r["scope"], row["rung"])


def test_a_requests_output_cap_meets_the_windows_leak_target_unsampled(out):
    """The window's requests export at most `OUTPUT_CAP` bits each, fewer in all than the leak target counts, so the
    `request` design samples nothing and pays only its boundaries."""
    assert S.IN["sequences"].value * S.OUTPUT_CAP < S.LEAK_BITS
    for r in _two_stage(out, target="leak"):
        for row in r["rows"]:
            design = row["request"]
            assert design["q"] == design["k"] == design["draws"] == design["interior"] == design["prove"] == 0
            assert design["total"] == design["boundary"]


def test_two_stage_pays_off_pooled_over_a_fleet_and_not_at_one_window(out):
    """For free compute two-stage is dearer than one-stage at one window, where the n_ρ-fold proving outweighs the
    commitment it saves, and cheaper pooled over a datacenter's and the world's GPUs for each monitoring period, where
    one-stage's serve-time commitment of every proof block dominates."""
    for r in _two_stage(out, target="work"):
        best = {d: next(x for x in r["rows"] if x["rung"] == rung)[d]["total"] for d, rung in r["best"].items()}
        if r["scope"] == "window":
            assert best["layer token"] > best["one-stage"], r["h"]
        elif r["scope"].split()[0] in S.fleet_gpus():
            assert best["layer token"] < best["one-stage"], (r["scope"], r["h"])


def test_no_rungs_fit_turns_on_the_hm96_and_sha512_slots(out):
    """A rung that misses Flock's block misses it on a proof block's own slot (its ANDs and rows' bits) and the mask, or
    on its width, so the code-based commitment, which drops hm96's slots, fits the same rungs."""
    for name, rung in out["rungs"].items():
        if not rung["fits"]:
            assert any((1 << (round(p["gates"]) + 8 * sum(p["rows"])).bit_length()) + (1 << MASK_LOG) > 1 << K_MAX
                       or p["out_bits"] > S.UNIT_CLASS_MAX for p in rung["parts"] if not p["fits"]), name
