"""The wide level-0 leaf's parameters in their three places agree: the Ligerito configs `flock-wide-b684b12.patch` adds
to upstream (what the prover and the session verifier build from), `rec_algebra.fast100_wide` (the Lean
`Flock.fast100Wide`, whose `#guard`s give the numbers here) and `arena_layout.WIDE` (the device arena's replay). Up to
m = 31 the wide version is fast100's, in each."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from arena_layout import LIGERITO, WIDE, Shape, rep_events, session_events, shapes
from verity_flock import rec_algebra as A

PATCH = Path(__file__).resolve().parents[1] / "flock-wide-b684b12.patch"
#: `Flock/WideLeaf.lean`'s guard: each level's (rate, q, cols, d, lanes), level 0's lanes ``m - 25``
LEVELS = ((1, 218, 18, 19, None), (2, 106, 15, 17, 4), (3, 71, 12, 15, 4), (4, 53, 9, 13, 4), (5, 43, 6, 11, 4), (6, 36, 3, 9, 4))


def _configs() -> dict[int, dict]:
    """The configs the patch adds, each a new file whose every line the patch holds."""
    out, name, lines = {}, None, []
    for line in PATCH.read_text().splitlines() + ["diff --git end"]:
        if line.startswith("diff --git"):
            if name:
                out[int(name)] = tomllib.loads("\n".join(lines))
            got = re.search(r" b/crates/flock-core/configs/ligerito/m(\d+)_fast100_wide\.toml$", line)
            name, lines = (got.group(1) if got else None), []
        elif name and line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    return out


def test_the_patch_adds_the_four_configs_of_the_lean_schedule():
    configs = _configs()
    assert sorted(configs) == [32, 33, 34, 35]
    for m, c in configs.items():
        assert (c["m"], c["log_n"], c["initial_k"], c["target_security_bits"]) == (m, m - 7, m - 25, 100)
        got = tuple((lv["log_inv_rate"], lv["queries"], lv["log_msg_cols"], lv["log_msg_cols"] + lv["log_inv_rate"],
                     lv["log_num_interleaved"]) for lv in c["levels"])
        assert got == tuple(lv[:4] + (lv[4] or m - 25,) for lv in LEVELS), m
        assert all(lv["k_recursive"] == lv["log_num_interleaved"] for lv in c["levels"])
        # level 0's codeword: 2^19 symbols at every size, a row of 2^(m - 25) witness symbols
        assert c["levels"][0]["log_msg_cols"] + c["levels"][0]["log_inv_rate"] == 19
        assert c["log_n"] - c["initial_k"] == c["levels"][0]["log_msg_cols"]


@pytest.mark.parametrize("m", range(32, 36))
def test_the_schedule_and_the_arenas_ladder_are_the_configs(m: int):
    c, s = _configs()[m], A.fast100_wide(m)
    assert (s.m, s.n, s.k0, s.ext) == (m, m - 7, m - 25, 3)
    assert tuple((lv.rate, lv.q, lv.cols, lv.d, lv.lanes) for lv in s.levels) == tuple(lv[:4] + (lv[4] or m - 25,) for lv in LEVELS)
    assert tuple((lv.rate, lv.q, lv.cols, lv.lanes) for lv in s.levels) == tuple(
        (lv["log_inv_rate"], lv["queries"], lv["log_msg_cols"], lv["log_num_interleaved"]) for lv in c["levels"])
    # everything but the lanes is m = 31's: each level's strata and cap with them
    m31 = A.fast100(31)
    assert [(lv.strata, lv.cap) for lv in s.levels] == [(lv.strata, lv.cap) for lv in m31.levels]
    assert s.levels[1:] == m31.levels[1:]
    ik, rates, ks, queries = WIDE[m]
    assert ik == c["initial_k"] and rates == tuple(lv["log_inv_rate"] for lv in c["levels"])
    assert ks == tuple(lv["k_recursive"] for lv in c["levels"][1:]) and queries == tuple(lv["queries"] for lv in c["levels"])
    assert Shape(m, 25, True, wide=True).lig() == WIDE[m] and Shape(m, 25, True).lig() == LIGERITO[m]


def test_up_to_m31_the_wide_version_is_fast100s():
    for m in range(22, 32):
        assert A.fast100_wide(m) == A.fast100(m)
    for bad in (21, 36, 37):
        with pytest.raises(ValueError, match="no fast100 schedule"):
            A.fast100_wide(bad)
    assert not set(WIDE) & set(range(32))
    for s in shapes(ms=range(21, 32)):
        assert not s.wide
        for keep in (True, False):
            assert session_events(Shape(s.m, s.k_log, s.zk, wide=True), keep) == session_events(s, keep)


def test_level_0_at_m35_is_2_to_the_19_leaves_of_16448_bytes():
    """Under M1, level 0's leaf is the witness's 2^10 symbols and the four extra lanes', 16 bytes each, and its tree has
    2^19 leaves; fast100's has 2^23 leaves of 68 symbols."""
    def level0(s: Shape) -> dict[str, int]:
        return {e[1]: e[2] for e in rep_events(s, "plain") if e[0] == "a" and e[1] in ("d_cw0", "d_tree0")}
    wide, own = level0(Shape(35, 25, True, wide=True)), level0(Shape(35, 25, True))
    assert wide == {"d_cw0": (1 << 19) * 16448, "d_tree0": (2 * (1 << 19) - 1) * 64}
    assert own == {"d_cw0": (1 << 23) * 68 * 16, "d_tree0": (2 * (1 << 23) - 1) * 64}
