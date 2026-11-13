"""a5's challenger reserve (`a5_coins.need`, `families`, `reserve_reason`) and the session `a5_coins.start` opens on it.

A rank's need is the coins its `service.Stream` sample reads for its sampling design over its window's counts, plus its longest stream
once more, known before anything opens. The windows are toy templates at the counts of record. One is 38b2's step-16
SiluMul_v2 window (675 + 3 units a rank, the stratified SiluMul partition, sampled 4 and 1 under `SILU_RULE`). The other is
the 18-family partition's rank-0 window at step 16 (`glm53_all_families_tp8_step16.json` before EmbeddingShard_v2 rejoined it:
179,552 units, one a template under `FAMILIES_RULE`), whose sample reads more than the 16 coins a rank the old reserve held."""

from __future__ import annotations

import random
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.types import Array, Value as Bits
from verity.protocols.spot.device import challenger as Ch, coins as CC
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_coins as AC  # noqa: E402

XOR8 = PrimitiveDefinition("A5ReserveTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)

#: units a rank, by template: 38b2's SiluMul_v2{I=256} and SiluMul_v2{I=1536}
SILU_MUL = [675, 3]
#: the 18-family partition's rank-0 window, its templates in the partition file's order
ALL_FAMILIES = [256, 1, 2, 1, 160, 9, 9, 1, 8, 1, 49152, 6144, 4096, 2048, 19360, 4096, 90112, 4096]
#: the rates that sample SILU_MUL's strata 4 and 1, and each of ALL_FAMILIES' one
SILU_RULE, FAMILIES_RULE = "stratified-rate:4/675", "stratified-rate:1/90112"
ROOT = "ab" * 64
AFTER = bytes.fromhex("cd" * 64)
STUB = AC.TREE / "verity" / "composite" / "device" / "live_challenger" / "tests" / "stub_serve.py"


def template(i: int):
    return bind(CompositeDefinition(f"A5ReserveTestT{i}", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                    lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))


def window(counts: list[int]):
    """A window of one toy template a count: its population program, query and N."""
    defs = [template(i) for i in range(len(counts))]
    return P.mixed_population_program(list(zip(defs, counts))), P.template_instances_query(defs), sum(counts)


def law_of(rule: str, counts: list[int]) -> dict:
    program, query, n = window(counts)
    return D.derive(program, query, S.selection_law(S.Law(rule), n, D.strata_sizes(program, query)),
                    S.selection_table(S.Law(rule), program, query, None))


def need_of(rule: str, counts: list[int]) -> int:
    return AC.need(rule, *window(counts))


def seeded(seed: int):
    rng = random.Random(seed)
    return rng.randbytes


def challenger_coins(layout: list) -> CC.ChallengerCoins:
    """Rank 0's coins, its own family's, on an in-process challenger of `layout`, after the coin `Draws` opens after the
    window's root in the window's family."""
    live = Ch.LiveChallenger(AC.SCHEME, 0, layout, seeded(1))
    window = CC.ChallengerCoins(Ch.LiveRecord(live, AC.WINDOW_FAMILY, clock=lambda: 0.0))
    window.commit(bytes.fromhex(ROOT))
    window(Ch.COIN_BYTES)
    coins = CC.ChallengerCoins(Ch.LiveRecord(live, AC.family(0), clock=lambda: 0.0))
    coins.commit(AFTER)
    return coins


def layout(*ranks: int) -> list[list]:
    """The session's layout with `ranks[r]` coins in rank r's family, after the window family's one."""
    return [[AC.WINDOW_FAMILY, 1], *([AC.family(r), n] for r, n in enumerate(ranks))]


def reads(rule: str, counts: list[int], seed: int) -> int:
    """The bytes a sample of the window reads on seeded coins."""
    got = []
    source = seeded(seed)

    def coins(n: int) -> bytes:
        got.append(n)
        return source(n)
    S.select_draw(S.Law(rule), law_of(rule, counts), sum(counts), coins)
    return sum(got)


def test_the_silu_mul_window_keeps_todays_reserve():
    """38b2's SiluMul window reads 226 bytes a rank (a stratum of 675 at k 4, 144 bytes, and one of 3 at its floor of 1,
    82), so its need with the margin is 6 coins. The reserve keeps the current 16 coins a rank, each rank's in a family
    of its own, and one more in the window's: 129 for 8 ranks."""
    assert [s["k"] for s in law_of(SILU_RULE, SILU_MUL)["strata"]] == [4, 1]
    assert reads(SILU_RULE, SILU_MUL, 0) == 226
    need = need_of(SILU_RULE, SILU_MUL)
    assert need == -(-(226 + 144) // Ch.COIN_BYTES) == 6
    needs = dict.fromkeys(range(8), need)
    assert AC.families(needs) == layout(*[16] * 8) and sum(n for _, n in AC.families(needs)) == 16 * 8 + 1
    assert [label for label, _ in AC.families(needs)] == ["window", *(f"draw/{r}" for r in range(8))]
    assert AC.reserve_reason(AC.families(needs), needs) is None


def test_the_18_family_window_needs_more_coins_than_todays_reserve():
    """The 18-family window's sample reads 1,496 bytes, one stream of 82 to 86 bytes a stratum at k 1. That is past the
    1,024 of 16 coins, and the 25 coins it needs with the margin hold it. On seeded coins the sample reads 1,496 bytes
    (no word rejected)."""
    assert all(s["k"] == 1 for s in law_of(FAMILIES_RULE, ALL_FAMILIES)["strata"])
    assert {reads(FAMILIES_RULE, ALL_FAMILIES, seed) for seed in range(4)} == {1496}
    assert 1496 > AC.COINS_PER_RANK * Ch.COIN_BYTES
    assert need_of(FAMILIES_RULE, ALL_FAMILIES) == -(-(1496 + 86) // Ch.COIN_BYTES) == 25
    assert AC.families({0: 25}) == layout(25)


def test_todays_reserve_runs_out_on_the_18_family_draw_and_its_need_does_not():
    """On a challenger of one rank's old reserve (17 coins), the 18-family sample runs out of coins (`every sampling coin of the
    epoch is open`), as the 18-family run did. On its need's reserve (`families`) it samples, opening 24 coins after the
    rank's commitment, its family's first."""
    rule, law, n = FAMILIES_RULE, law_of(FAMILIES_RULE, ALL_FAMILIES), sum(ALL_FAMILIES)
    with pytest.raises(Ch.CoinError, match="every draw/0 coin of the epoch is open"):
        S.select_draw(S.Law(rule), law, n, challenger_coins(layout(16)))
    coins = challenger_coins(AC.families({0: need_of(rule, ALL_FAMILIES)}))
    drawn = S.select_draw(S.Law(rule), law, n, coins)
    assert len(drawn["units"]) == 18 and len(coins.segments[-1].openings) == 24 and coins.segments[-1].start == 0


def test_a_work_laws_need_is_its_draw_under_the_work_table_the_rank_registers_with():
    """A window whose first template holds transfer gates (32 units at work 0 and floor 32, sampled whole) beside 8 sampled by
    ``work-rate:1/20`` (work:2): its sample reads 640 bytes for the whole stratum and 100 for the rest, so the need is 22 coins,
    and its reserve holds the sample. A work rate has no need without the table (`service.selection_table` refuses it)."""
    program, query, n = window([32, 8])
    t0, t1 = (P.template_id(template(i)) for i in range(2))
    table = {t0: {"work": 0, "floor": 32}, t1: {"work": 1, "floor": 1}}
    rule = S.Law("work-rate:1/20")
    law = D.derive(program, query, S.selection_law(rule, n, D.strata_sizes(program, query)),
                   S.selection_table(rule, program, query, table))
    assert law["k"] == 2 and [s["k"] for s in law["strata"]] == [32, 2]
    need = AC.need(rule.text, program, query, n, table)
    assert need == -(-(640 + 100 + 640) // Ch.COIN_BYTES) == 22
    drawn = S.select_draw(rule, law, n, challenger_coins(AC.families({0: need})))
    assert set(range(32)) <= set(drawn["units"]) and len(drawn["units"]) == 34
    with pytest.raises(S.Refused) as e:
        AC.need(rule.text, program, query, n)
    assert e.value.verdict.codes == ("rate",) and "work table" in e.value.verdict.detail["rate"]["reason"]


def test_a_reserve_that_cant_cover_the_draws_is_refused_naming_need_and_reserve():
    """`reserve_reason` names the need (the coin after the root, then each rank's, a family each) and the reserve when
    the reserve is short in a family or lacks one, and the challenger's own limit when the layout is past it."""
    need = "the draws need {'window': 1, 'draw/0': 25} coins (1 after the window's root, then each rank's in its own family)"
    assert AC.reserve_reason(layout(16), {0: 25}) == (
        f"{need}, and the reserve [['window', 1], ['draw/0', 16]] holds fewer of ['draw/0']")
    assert AC.reserve_reason([[AC.FAMILY, 26]], {0: 25}) == (
        f"{need}, and the reserve [['draw', 26]] holds fewer of ['draw/0', 'window']")
    big = {r: Ch.MAX_COINS // 2 for r in range(2)}
    assert AC.reserve_reason(AC.families(big), big).endswith(f"a live coin layout holds at most {Ch.MAX_COINS} coins")


def local(tmp_path: Path, monkeypatch, ranks: list[int], **kw):
    """`a5_coins.start` on a local challenger session under `tmp_path`, pinned on this tree's commit. Its serve.py runs under
    live_challenger's stub of the Lean verifier's coins (`stub_serve.py`), so the test builds no `sampled-proofs-verifier`."""
    A = AC._module("a5_auditor", AC.TREE / "verity" / "composite" / "device" / "live_challenger" / "auditor.py")
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
    """The local challenger opens the session once the rank has planned its sample, on the layout of its need (26 coins:
    the window family's one and 25 in the rank's own). The 18-family sample reads its 24 coins after the rank's
    commitment, and its coins' evidence holds at the rank's family (`coins.coins_reason`) and at no other."""
    rule, law, n = FAMILIES_RULE, law_of(FAMILIES_RULE, ALL_FAMILIES), sum(ALL_FAMILIES)
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
    assert plan["needs"] == {"0": 25} and plan["families"] == layout(25) and plan["session"]
    assert published["trusted"]["released"] == {AC.WINDOW_FAMILY: 1, AC.family(0): 24} and len(coins.read) == 1496
    assert coins.label == segment["label"] == AC.family(0) and segment["start"] == 0
    evidence = AC.evidence(published, segment)
    assert CC.coins_reason(evidence, ROOT, AC.family(0), AFTER, bytes(coins.read), AC.lms_verify) is None
    for other in (AC.WINDOW_FAMILY, AC.FAMILY, AC.family(1)):
        assert "family's" in CC.coins_reason(evidence, ROOT, other, AFTER, bytes(coins.read), AC.lms_verify)


def test_two_ranks_open_the_session_once_both_have_planned(tmp_path, monkeypatch):
    """Rank 1's plan waits for rank 0's; then the session opens on both needs' layout (16 coins a rank and one, 33), and each
    rank samples in its turn from its own family's first coin, 4 coins for SiluMul's 226 bytes, as 38b2's gate run did
    on each of its 8 ranks."""
    rule, law, n = SILU_RULE, law_of(SILU_RULE, SILU_MUL), sum(SILU_MUL)
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
    assert published["plan"]["needs"] == {"0": 6, "1": 6} and published["plan"]["families"] == layout(16, 16)
    assert sorted(segments) == [0, 1] and all(len(s["openings"]) == 4 for s in segments.values())
    assert [(segments[r]["label"], segments[r]["start"]) for r in (0, 1)] == [(AC.family(0), 0), (AC.family(1), 0)]
    assert published["trusted"]["released"] == {AC.WINDOW_FAMILY: 1, AC.family(0): 4, AC.family(1): 4}


def test_a_too_small_reserve_is_refused_before_the_session_opens(tmp_path, monkeypatch):
    """A reserve short of the need refuses the rank's plan, naming both, and the local host reserves nothing."""
    manager, draws, opener = local(tmp_path, monkeypatch, [0], reserve=lambda needs: layout(16))
    try:
        with pytest.raises(Ch.CoinError, match=r"need \{'window': 1, 'draw/0': 25\} coins .* holds fewer of "
                                               r"\['draw/0'\]: the root's one session was not opened"):
            AC.RankCoins(draws, 0).plan(25)
        draws.skip(0)
        opener.join()
        published = draws.evidence()
    finally:
        manager.shutdown()
    assert published["plan"] == {"needs": {"0": 25}, "families": layout(16), "session": None}
    assert nothing_reserved(tmp_path)


def test_a_rank_that_ends_before_planning_opens_nothing(tmp_path, monkeypatch):
    """A rank refused before its sample (or failed) ends without planning, so no rank's sample opens the session."""
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


def test_a_local_open_whose_challenger_signed_nothing_is_refused_without_the_sessions_secret(tmp_path, monkeypatch):
    """The gate answers an open with the session's secret even when its challenger stopped before it signed a commitment
    (`auditor.gate_open`). `Local.open` refuses that open by name with the gate's answer less its secret, so the text a
    run's log keeps holds no session's secret."""
    session = AC.Local(tmp_path, [AC.on_main([])])
    monkeypatch.setattr(AC.Local, "_serve", lambda self, s: None)  # the session's unit stops before it signs
    with pytest.raises(Ch.CoinError, match="the local gate refused the open, or its challenger signed no commitment: ") as e:
        session.open(ROOT, [[AC.FAMILY, 2]])
    secret = (session.host.sessions / session.session / session.A.SECRET).read_text().strip()
    assert len(secret) == 64 and secret not in str(e.value)
    assert f'"session": "{session.session}"' in str(e.value) and '"commitment": null' in str(e.value)
