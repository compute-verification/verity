"""`verity_flock.rec_budget`, the stagers' mirror of the Lean verifier's caps on a recursive session (`Flock.RecBudget`):
the same inputs as `Flock/Test/RegBudget.lean`'s `#guard`s give the same refusals, word for word; the caps make the
recursion's call bounds at most `2^32` (the compiled verifier's own evaluations at 16 parts, the inner setup's and the
inner verifier's); and they sit far above every deployed session at every `fast100` schedule: a V* table's instances (a
level's `2 q` openings, a later level's RecCap over both reps' `2 · 2^cap` tops, RecCap's `2^cap`), a part's ports (its row
ports, at most `MAX_REGIONS`, with `v` and `out`), the deepest registered tree (`rec-acc`'s chain of every opening, both
reps' and R's), the inner statement's instances; and V*'s list, the recursion's parts (each level's RecOpen, the algebra's
parts, RecCap at each later level and over level 0's cap), and the hm96 calls that list records (`recorded_calls`), at
every deployed inner shape, while the cap is the 16 parts the recursion's soundness bound covers."""
from __future__ import annotations

import pytest
from verity_flock import rec_algebra as A
from verity_flock import rec_budget as RB
from verity_flock import rec_residuals as RR

TAIL = "the recursion's call bound covers no more"


def _reg(ports: dict[str, int]) -> dict:
    return {"format": "verity/registered-reads/v1", "ports": {nm: {"leaves": n} for nm, n in ports.items()}}


def test_the_mirror_refuses_what_the_lean_guards_refuse():
    assert (RB.MAX_INSTANCES, RB.MAX_PORTS, RB.MAX_DEPTH) == (4096, 512, 64)
    assert RB.regs(_reg({"acc": 2048, "out": 128, "coef": 510})) is None
    assert RB.regs(_reg({f"p{i}": 2 ** RB.MAX_DEPTH for i in range(RB.MAX_PORTS)})) is None
    assert RB.regs(_reg({f"p{i}": 1 for i in range(RB.MAX_PORTS + 1)})) == (
        f"refused: a V* session's registered reads name 513 ports, more than 512: {TAIL}")
    assert RB.regs(_reg({"acc": 2048, "big": 2 ** RB.MAX_DEPTH + 1})) == (
        "refused: a V* session's registered port big has 18446744073709551617 leaves, a tree of 65 levels, more than 64: "
        f"{TAIL}")
    assert RB.table("", 510) is None and RB.table("", RB.MAX_INSTANCES) is None
    assert RB.table("", RB.MAX_INSTANCES + 1) == (
        f": refused: a V* session's table of 4097 instances, more than 4096: {TAIL}")
    assert RB.MAX_STATEMENTS == 16
    assert RB.count(10) is None and RB.count(16) is None
    assert RB.count(17) == (
        "refused: V*'s list at the verifier's inner shape holds 17 statements, more than 16: the recursion's soundness "
        "bound covers at most 16 V* parts")
    assert RB.count(4096) == (
        "refused: V*'s list at the verifier's inner shape holds 4096 statements, more than 16: the recursion's soundness "
        "bound covers at most 16 V* parts")
    assert RB.recorded(0) is None and RB.recorded(RB.MAX_RECORDED) is None
    assert RB.recorded(RB.MAX_RECORDED + 1) == (
        f"refused: V*'s list at the verifier's inner shape records 4294967297 hm96 calls, more than 4294967296: {TAIL}")
    assert RB.MAX_INNER_ROWS == 2 ** 17
    assert RB.inner(16) is None and RB.inner(RB.MAX_INNER_ROWS) is None
    assert RB.inner(RB.MAX_INNER_ROWS + 1) == (
        f"refused: the inner statement's public file holds 131073 instances, more than 131072: {TAIL}")
    assert RB.inner(16, [16, RB.MAX_INNER_ROWS]) is None
    assert RB.inner(16, [16, RB.MAX_INNER_ROWS + 1]) == (
        f"refused: the inner statement's public file has a shared-row table of more than 131072 rows: {TAIL}")
    with pytest.raises(ValueError, match="holds 131073 instances"):
        RB.check_inner(RB.MAX_INNER_ROWS + 1)


def test_a_statements_recorded_calls_are_the_verifiers():
    """`recorded_at` counts as the Lean guards' `recordedAt` does: a level at `m = 27`'s level 0 over its `2 · 218`
    openings and `rec-acc` rows, R's level with and without R's shape, a RecCap of 128 nodes, RecSparse's two ports at
    two rows, the link table's part, and a list of three."""
    assert RB.recorded_at("rec-open", 436, acc_reads=436) == 4360
    assert RB.recorded_at("rec-open-stacked", 218, acc_reads=218) == 1308
    assert RB.recorded_at("rec-open-stacked", 218) == 436
    assert RB.recorded_at("cap", 128) == 512
    assert RB.recorded_at("sparse", 2, own_rows=4) == 8
    assert RB.recorded_at("link", 18) == 0
    assert RB.recorded_at("rec-open", 436, acc_reads=436) + 2 * RB.recorded_at("cap", 128) == 5384


def test_the_caps_bound_the_recursions_calls_by_2_32():
    """`DeployedCounts.deployedSetup_staged_le`: a V* part's setup inputs, check 5's `131072` and its registered reads,
    times the 16 parts; `InnerSetupCount.innerB`: the inner setup's, `12 + 1024 · perRow K` at `2^K = MAX_INNER_ROWS`;
    and `BitsKTight.nR`: the inner verifier's, with its tables' `2^28` and a decision's `2^24`."""
    part = 131072 + RB.MAX_INSTANCES * RB.MAX_PORTS * (2 + RB.MAX_DEPTH)
    assert part == 138543104 and RB.MAX_STATEMENTS * part < 2 ** 32
    k = RB.MAX_INNER_ROWS.bit_length() - 1
    inner_b = 12 + 1024 * (k + 3) * 2 ** k
    assert 2 ** k == RB.MAX_INNER_ROWS and inner_b == 2684354572
    assert 2 ** 28 + inner_b + 2 ** 24 < 2 ** 32


def test_log2ceil_is_the_verifiers():
    assert [RB.log2ceil(n) for n in (0, 1, 2, 3, 4, 5, 2 ** 64, 2 ** 64 + 1)] == [0, 0, 1, 2, 2, 3, 64, 65]


def test_the_caps_sit_far_above_every_deployed_session():
    """The inner statements: 16 instances (the recursion fixtures), 2 to 4 (the recursion bench and rec-links), 3 (the
    served silu-mul statement) and 2048 (85-rec-reprice's default N)."""
    deepest = most = 0
    for m in range(22, 36):
        s = A.fast100(m)
        opens = sum(lv.q for lv in s.levels)
        most = max(most, *(2 * lv.q for lv in s.levels), 2 ** s.levels[0].cap, *(2 * 2 ** lv.cap for lv in s.levels[1:]))
        for r in range(22, 36):
            deepest = max(deepest, RB.log2ceil(2 * opens + sum(lv.q for lv in A.fast100(r).levels) + 1))
    assert most <= 510 and 8 * most <= RB.MAX_INSTANCES
    assert deepest <= 12 and 5 * deepest <= RB.MAX_DEPTH
    assert 7 * (RR.MAX_REGIONS + 2) <= RB.MAX_PORTS
    assert 64 * 2048 <= RB.MAX_INNER_ROWS


def _listed(m: int, k: int, c: int, holo: bool = False) -> int:
    """V*'s list at a plain or holo inner shape (`RecVStar.statements`): each level's RecOpen and RecCap, and the parts."""
    sh = A.Shape(m, k, c, holo=holo)
    return 2 * len(sh.sched.levels) + len(RR.parts(RR.spec(sh)))


def _candidate_3(m: int, k: int, c: int, r_m: int) -> int:
    """Candidate 3's list: the holo list, RecSparse, R's levels at its `m` and R's opening parts (`candidate3Statements`)."""
    r = A.Shape(r_m, 6, 2, salt_len=0, opening_only=True)
    return _listed(m, k, c, holo=True) + 1 + len(r.sched.levels) + len(RR.parts(RR.spec(r)))


@pytest.mark.slow
def test_the_statement_cap_holds_every_deployed_v_star_list():
    """85-rec-reprice's K = 4096 statement (m 35, three regions) lists 16, the recursion bench's at N = 2 (m 27) 10, the
    served silu-mul statement (29, 26, 6) 12 and 88-rec-c3-inner's candidate 3 (25, 22, 8) at R's m 23 15: each within
    the cap. At most three regions every m ≤ 35 lists at most 16, and four regions at m 35 (17) or candidate 3 at m 28
    (17) is refused."""
    assert [_listed(35, 24, 8), _listed(27, 24, 8), _listed(29, 26, 6), _candidate_3(25, 22, 8, 23)] == [16, 10, 12, 15]
    assert max(_listed(m, 22, c) for m in range(22, 36) for c in (4, 6, 8)) == RB.MAX_STATEMENTS
    assert RB.count(_listed(35, 24, 10)) is not None and RB.count(_candidate_3(28, 22, 8, 23)) is not None


@pytest.mark.slow
def test_the_recorded_cap_sits_far_above_every_deployed_v_star_list():
    """The hm96 calls each deployed `fast100` list records (`recorded_calls`) sit at least `2^18` below `MAX_RECORDED`."""
    calls = {}
    for shape in [(m, 22, 8) for m in range(22, 36)] + [(22, 6, 16)]:
        try:
            calls[shape] = RB.recorded_calls(A.Shape(*shape))
        except ValueError:
            continue
    assert len(calls) >= 10, calls
    assert calls.get((27, 22, 8)) == 11128 and calls.get((35, 22, 8)) == 14232, calls
    assert 2 ** 18 * max(calls.values()) <= RB.MAX_RECORDED, calls
