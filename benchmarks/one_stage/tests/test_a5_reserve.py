"""a5's challenger reserve (`a5_coins.need`, `families`, `reserve_reason`) and the session `a5_coins.start` opens on it.

A rank's need is the coins its `service.Stream` draw reads for its law over its window's counts, plus its longest stream
once more, known before anything opens. The windows are toy templates at the counts of record. One is 38b2's step-16
SiluMul_v2 window (675 + 3 units a rank under `stratified:4`, the stratified SiluMul partition). The other is the 18-family
partition's rank-0 window at step 16 (`glm53_all_families_tp8_step16.json` before EmbeddingShard_v2 rejoined it: 179,552 units
under `stratified:1`), whose draw reads more than the 16 coins a rank the old reserve held."""

from __future__ import annotations

import random
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs import challenger as Ch, coins as CC, service as S
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_coins as AC  # noqa: E402

XOR8 = PrimitiveDefinition("A5ReserveTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)

#: units a rank, by template: 38b2's SiluMul_v2{I=256} and SiluMul_v2{I=1536}
SILU_MUL = [675, 3]
#: the 18-family partition's rank-0 window, its templates in the partition file's order
ALL_FAMILIES = [256, 1, 2, 1, 160, 9, 9, 1, 8, 1, 49152, 6144, 4096, 2048, 19360, 4096, 90112, 4096]
ROOT = "ab" * 64
AFTER = bytes.fromhex("cd" * 64)
STUB = AC.TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"


def template(i: int):
    return bind(CompositeDefinition(f"A5ReserveTestT{i}", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                    lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))


def window(counts: list[int]):
    """A window of one toy template a count: its population program, query and N."""
    defs = [template(i) for i in range(len(counts))]
    return P.mixed_population_program(list(zip(defs, counts))), P.template_instances_query(defs), sum(counts)


def law_of(rule: str, counts: list[int]) -> dict:
    program, query, n = window(counts)
    return D.derive(program, query, S.selection_law(S.Law(rule), n))


def need_of(rule: str, counts: list[int]) -> int:
    return AC.need(rule, *window(counts))


def seeded(seed: int):
    rng = random.Random(seed)
    return rng.randbytes


def challenger_coins(layout: list) -> CC.ChallengerCoins:
    """One rank's coins on an in-process challenger of `layout`, after the coin `Draws` opens after the window's root."""
    live = Ch.LiveChallenger(AC.SCHEME, 0, layout, seeded(1))
    coins = CC.ChallengerCoins(Ch.LiveRecord(live, AC.FAMILY, clock=lambda: 0.0))
    coins.commit(bytes.fromhex(ROOT))
    coins(Ch.COIN_BYTES)
    coins.commit(AFTER)
    return coins


def reads(rule: str, counts: list[int], seed: int) -> int:
    """The bytes a draw of the window reads on seeded coins."""
    got = []
    source = seeded(seed)

    def coins(n: int) -> bytes:
        got.append(n)
        return source(n)
    S.select_draw(S.Law(rule), law_of(rule, counts), sum(counts), coins)
    return sum(got)


def test_the_silu_mul_window_keeps_todays_reserve():
    """38b2's SiluMul window reads 226 bytes a rank (a stratum of 675 at k 4, 144 bytes, and one of 3 at its floor of 1,
    82), so its need with the margin is 6 coins. The reserve keeps today's 16 coins a rank and one more: 129 for 8
    ranks."""
    assert reads("stratified:4", SILU_MUL, 0) == 226
    need = need_of("stratified:4", SILU_MUL)
    assert need == -(-(226 + 144) // Ch.COIN_BYTES) == 6
    needs = dict.fromkeys(range(8), need)
    assert AC.families(needs) == [[AC.FAMILY, 16 * 8 + 1]]
    assert AC.reserve_reason(AC.families(needs), needs) is None


def test_the_18_family_window_needs_more_coins_than_todays_reserve():
    """The 18-family window's draw reads 1,496 bytes, one stream of 82 to 86 bytes a stratum at k 1. That is past the
    1,024 of 16 coins, and the 25 coins it needs with the margin hold it. On seeded coins the draw reads 1,496 bytes
    (no word rejected)."""
    assert all(s["k"] == 1 for s in law_of("stratified:1", ALL_FAMILIES)["strata"])
    assert {reads("stratified:1", ALL_FAMILIES, seed) for seed in range(4)} == {1496}
    assert 1496 > AC.COINS_PER_RANK * Ch.COIN_BYTES
    assert need_of("stratified:1", ALL_FAMILIES) == -(-(1496 + 86) // Ch.COIN_BYTES) == 25
    assert AC.families({0: 25}) == [[AC.FAMILY, 26]]


def test_todays_reserve_runs_out_on_the_18_family_draw_and_its_need_does_not():
    """On a challenger of one rank's old reserve (17 coins), the 18-family draw runs out of coins (`every draw coin of the
    epoch is open`), as the 18-family run did. On its need's reserve (`families`) it draws, opening 24 coins after the
    rank's commitment."""
    rule, law, n = "stratified:1", law_of("stratified:1", ALL_FAMILIES), sum(ALL_FAMILIES)
    with pytest.raises(Ch.CoinError, match="every draw coin of the epoch is open"):
        S.select_draw(S.Law(rule), law, n, challenger_coins([[AC.FAMILY, 17]]))
    coins = challenger_coins(AC.families({0: need_of(rule, ALL_FAMILIES)}))
    drawn = S.select_draw(S.Law(rule), law, n, coins)
    assert len(drawn["units"]) == 18 and len(coins.segments[-1].openings) == 24


def test_a_work_laws_need_is_its_draw_under_the_work_table_the_rank_registers_with():
    """A window whose first template holds transfer gates (32 units at work 0 and floor 32, drawn whole) beside 8 drawn by
    ``work:2``: its draw reads 640 bytes for the whole stratum and 100 for the rest, so the need is 22 coins, and its reserve
    holds the draw. A work law has no need without the table (`draw.derive` refuses it)."""
    program, query, n = window([32, 8])
    t0, t1 = (P.template_id(template(i)) for i in range(2))
    table = {t0: {"work": 0, "floor": 32}, t1: {"work": 1, "floor": 1}}
    law = D.derive(program, query, S.selection_law(S.Law("work:2"), n), table)
    assert [s["k"] for s in law["strata"]] == [32, 2]
    need = AC.need("work:2", program, query, n, table)
    assert need == -(-(640 + 100 + 640) // Ch.COIN_BYTES) == 22
    drawn = S.select_draw(S.Law("work:2"), law, n, challenger_coins(AC.families({0: need})))
    assert set(range(32)) <= set(drawn["units"]) and len(drawn["units"]) == 34
    with pytest.raises(ValueError, match="work table"):
        AC.need("work:2", program, query, n)


def test_a_reserve_that_cant_cover_the_draws_is_refused_naming_need_and_reserve():
    """`reserve_reason` names the need (the coin after the root, then each rank's) and the reserve when the reserve is
    short, and the challenger's own limit when the layout is past it."""
    assert AC.reserve_reason([[AC.FAMILY, 17]], {0: 25}) == (
        "the draws need 26 draw coins (1 after the window's root, then rank 0 25), and the reserve [['draw', 17]] holds 17")
    big = {r: Ch.MAX_COINS // 2 for r in range(2)}
    assert AC.reserve_reason(AC.families(big), big).endswith(f"a live coin layout holds at most {Ch.MAX_COINS} coins")


def local(tmp_path: Path, monkeypatch, ranks: list[int], **kw):
    """`a5_coins.start` on a local challenger session under `tmp_path`, pinned on this tree's commit. Its serve.py runs under
    live_challenger's stub of the Lean verifier's coins (`stub_serve.py`), so the test builds no `sampled-proofs-verifier`."""
    A = AC._module("a5_auditor", AC.TREE / "benchmarks" / "live_challenger" / "auditor.py")
    real = A.serve_argv
    monkeypatch.setattr(A, "serve_argv", lambda *a, **k: [sys.executable, str(STUB), *real(*a, **k)[2:]])
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = tmp_path / "main.txt"
    main.write_text(AC.on_main([]) + "\n")
    return AC.start("local", ROOT, ranks, tmp_path / "challenger", main, **kw)


def nothing_reserved(tmp_path: Path) -> bool:
    (host,) = tmp_path.glob("a5-challenger-*")
    return not (host / "var" / "windows.jsonl").exists() and not any((host / "var" / "sessions").iterdir())


def test_the_local_challenger_opens_on_the_needs_reserve_and_the_draw_holds(tmp_path, monkeypatch):
    """The local challenger opens the session once the rank has planned its draw, on the layout of its need (26 coins). The
    18-family draw reads its 24 coins after the rank's commitment, and its coins' evidence holds (`coins.coins_reason`)."""
    rule, law, n = "stratified:1", law_of("stratified:1", ALL_FAMILIES), sum(ALL_FAMILIES)
    manager, draws, opener = local(tmp_path, monkeypatch, [0])
    try:
        coins = AC.RankCoins(draws, 0)
        coins.plan(need_of(rule, ALL_FAMILIES))
        coins.commit(AFTER)
        S.select_draw(S.Law(rule), law, n, coins)
        segment = coins.drawn()
        draws.skip(0)
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert "error" not in published, published.get("error")
    assert (tmp_path / "coins.log").read_text(), "the session's coins came from the stub, not a Lean build"
    plan = published["plan"]
    assert plan["needs"] == {"0": 25} and plan["families"] == [[AC.FAMILY, 26]] and plan["session"]
    assert published["trusted"]["released"] == {AC.FAMILY: 1 + 24} and len(coins.read) == 1496
    assert CC.coins_reason(AC.evidence(published, segment), ROOT, AFTER, bytes(coins.read), AC.lms_verify) is None


def test_two_ranks_open_the_session_once_both_have_planned(tmp_path, monkeypatch):
    """Rank 1's plan waits for rank 0's; then the session opens on both needs' layout (16 coins a rank and one, 33), and each
    rank draws in its turn, 4 coins for SiluMul's 226 bytes, as 38b2's gate run did on each of its 8 ranks."""
    rule, law, n = "stratified:4", law_of("stratified:4", SILU_MUL), sum(SILU_MUL)
    manager, draws, opener = local(tmp_path, monkeypatch, [0, 1])
    segments: dict[int, dict] = {}

    def rank(r: int) -> None:
        coins = AC.RankCoins(draws, r)
        coins.plan(need_of(rule, SILU_MUL))
        coins.commit(bytes([r]) * Ch.COIN_BYTES)
        S.select_draw(S.Law(rule), law, n, coins)
        segments[r] = coins.drawn()
        draws.skip(r)
    try:
        other = threading.Thread(target=rank, args=(1,))
        other.start()
        rank(0)
        other.join()
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert "error" not in published, published.get("error")
    assert published["plan"]["needs"] == {"0": 6, "1": 6} and published["plan"]["families"] == [[AC.FAMILY, 33]]
    assert sorted(segments) == [0, 1] and all(len(s["openings"]) == 4 for s in segments.values())
    assert published["trusted"]["released"] == {AC.FAMILY: 1 + 4 + 4}


def test_a_too_small_reserve_is_refused_before_the_session_opens(tmp_path, monkeypatch):
    """A reserve short of the need refuses the rank's plan, naming both, and the local host reserves nothing."""
    manager, draws, opener = local(tmp_path, monkeypatch, [0], reserve=lambda needs: [[AC.FAMILY, 17]])
    try:
        with pytest.raises(Ch.CoinError, match=r"need 26 draw coins .* holds 17: the root's one session was not opened"):
            AC.RankCoins(draws, 0).plan(25)
        draws.skip(0)
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert published["plan"] == {"needs": {"0": 25}, "families": [[AC.FAMILY, 17]], "session": None}
    assert nothing_reserved(tmp_path)


def test_a_rank_that_ends_before_planning_opens_nothing(tmp_path, monkeypatch):
    """A rank refused before its draw (or failed) ends without planning, so no rank's draw opens the session."""
    manager, draws, opener = local(tmp_path, monkeypatch, [0, 1])
    try:
        draws.skip(1)
        with pytest.raises(Ch.CoinError, match=r"rank\(s\) \[1\] ended before planning their draw"):
            AC.RankCoins(draws, 0).plan(6)
        draws.skip(0)
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert published["plan"]["session"] is None and "was not opened" in published["error"]
    assert nothing_reserved(tmp_path)
