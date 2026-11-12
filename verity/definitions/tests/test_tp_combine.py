"""`registry.tp_combine`'s combine units hold every split of a TP all-reduce bit for bit to the Definition it splits:

* the fused MNNVL: one `AllReduceResidualMnnvlCoordinate_v1` per coordinate, or one `AllReduceResidualMnnvlAt_v1` per chunk
  (`row_binding`'s) on each rank's row cut into chunk rows (`RowChunks_v1`), then `RmsNormMnnvl_v1` on their r's, compute
  `AllReduceRmsNormMnnvl_v1`'s normed row and r and
  `AllReduceResidualMnnvl_v1`'s r, and agree with `tp_allreduce_check`'s numpy model of the kernel, at GLM-5.3's served statics
  (every shot and cluster its fused calls launch at N = 6144, in chunks of 96) and at small widths, on hidden-state-like,
  special and NaN words;
* every coordinate of each route's call is `coordinate_binding`'s unit on the ranks' words there, and every row of an n-row call
  `row_binding`'s unit on the ranks' rows, at small sizes and at GLM-5.3's width, on words whose sum the multimem tree and the
  custom kernel's owner decide (`WITNESS`), so a unit bound at another position's statics would differ;
* the units of a GLM-5.3 call at chunks of 96 fit a C-Flock block with at least a tenth of it to spare, their rows' slots
  (`windows.interface.check_proof_block`) beside the slot of their ANDs and port bits (`verity_flock.circuit.unit_witness`), and
  neither its coordinate units nor a unit of a whole row of 6,144 (the MoE's fused call at any served launch) fit one;
* the σ versions (`registry.boolean_tp_combine`) compute their words on bits."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation import evaluate_batch, evaluate_words
from verity.definitions import boolean as BB
from verity.definitions.registry import b1, b1_tp2, dead_free
from verity.definitions.registry import boolean_tp_combine as BC
from verity.definitions.registry import tp_allreduce as TA
from verity.definitions.registry import tp_allreduce_check as K
from verity.definitions.registry import tp_combine as TC
from verity.commitments.windows.interface import K_MAX, check_proof_block, layout

GLM53_N = 6144
#: (SHOT, CLUSTER) of the fused calls GLM-5.3 serves at TP8 on H200: one-shot to 10 tokens, then two-shot; 8 CTAs to 16 tokens,
#: 4 to 33, 2 to its 42
GLM53_FUSED = ((1, 8), (2, 8), (2, 4), (2, 2))
#: signed zeros, the smallest and largest subnormals, the smallest normals, +-1, the finite maxima, infinities, NaNs
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80,
                     0xFF80, 0x7FC0, 0x7F81, 0xFFFF], np.uint16)
#: a chain for owners 0-3, a balanced tree for 4-7
TREES = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
#: each rank's word: 1 at rank 2, -1 at rank 4, 2^-30 at rank 6.  The chain reduces it to 2^-30 and the balanced tree to 0, and
#: the custom kernel from rank 0 or 2 to 2^-30 and from rank 3 or 5 to 0 (2^-30 vanishes into a running sum of 1)
WITNESS = np.array([0, 0, 0x3F80, 0, 0xBF80, 0, 0x3080, 0], np.uint16)


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _ev(fn, *args: np.ndarray) -> list[np.ndarray]:
    n = len(args[0])
    out = evaluate_batch(fn, *(np.asarray(a).astype(np.uint64) for a in args))
    return [np.asarray(a).reshape(n, -1).astype(np.uint16) for a in (out if isinstance(out, (tuple, list)) else [out])]


def _tokens(rng, M: int, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M tokens of hidden-state-like partials (M, 8, N) and residuals (M, N), one token of special words in every operand but NaN
    and one with NaNs (each makes its token's every normed word 0x7FFF); an N weight."""
    parts, res, w = _bf16(rng, (M, 8, N), 0.5), _bf16(rng, (M, N), 2.0), _bf16(rng, (N,), 0.1)
    for k in (len(SPECIALS) - 3, len(SPECIALS)):
        parts = np.concatenate([parts, SPECIALS[rng.integers(0, k, (1, 8, N))]])
        res = np.concatenate([res, SPECIALS[rng.integers(0, k, (1, N))]])
    return parts, res, w


def _words(rng, size: int) -> np.ndarray:
    """Rows of WORLD ranks' `size` words, rank-major: `WITNESS` at every element, special words, uniform random words and
    hidden-state-like ones."""
    rows = [np.broadcast_to(WITNESS[:, None], (8, size)), SPECIALS[rng.integers(0, len(SPECIALS), (8, size))],
            rng.integers(0, 1 << 16, (8, size)), _bf16(rng, (8, size), 0.5)]
    return np.stack(rows).reshape(len(rows), -1).astype(np.uint16)


def _bind(binding: tuple):
    d, statics = binding
    return bind(d, **statics)


def _served(entry: str):
    return next(c for c in TA.CAPTURED if (c.setting, c.N, c.route) == ("defaults", GLM53_N, entry))


def test_the_fused_statics_are_every_served_launch() -> None:
    served = {(TA.mnnvl_shot(M, GLM53_N, 8), TA.mnnvl_grid(M, GLM53_N)[1]) for M in range(1, TA.fused_max_tokens(GLM53_N, 8) + 1)}
    assert served == set(GLM53_FUSED)


@pytest.mark.parametrize("shot,N,cluster,fma", [(1, 256, 1, False), (2, 256, 1, True), (1, 512, 2, True), (2, 512, 2, False),
                                                *[pytest.param(s, GLM53_N, c, True, marks=pytest.mark.slow) for s, c in GLM53_FUSED]])
def test_the_fused_call_is_its_coordinates_then_its_norm(shot: int, N: int, cluster: int, fma: bool) -> None:
    rng = np.random.default_rng(N + 10 * shot + cluster)
    parts, res, w = _tokens(rng, 2, N)
    M, ws = len(res), np.broadcast_to(w, res.shape)
    statics = {"WORLD": 8, "N": N, "SHOT": shot, "CLUSTER": cluster, "EPS": 1e-5, "FMA": fma}
    (both,) = _ev(bind(TA.AllReduceRmsNormMnnvl, **statics), parts.reshape(M, -1), res, ws)
    normed, r = both[:, :N], both[:, N:]
    units = {_bind(TC.coordinate_binding(TA.AllReduceRmsNormMnnvl, statics, j)).id for j in range(N)}
    assert units == {f"AllReduceResidualMnnvlCoordinate_v1{{WORLD=8,SHOT={shot}}}"}
    unit = bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=shot)
    (r_coord,) = _ev(unit, parts.transpose(0, 2, 1).reshape(M * N, 8), res.reshape(M * N, 1))
    r_coord = r_coord.reshape(M, N)
    (r_part,) = _ev(bind(TA.AllReduceResidualMnnvl, WORLD=8, N=N, SHOT=shot), parts.reshape(M, -1), res)
    (normed_coord,) = _ev(bind(TA.RmsNormMnnvl, N=N, SHOT=shot, CLUSTER=cluster, EPS=1e-5, FMA=fma), r_coord, ws)
    want_normed, want_r = K.mnnvl_norm_rows(parts.transpose(1, 0, 2), res, w, shot, cluster, 1e-5, fma)
    assert (r_coord == r).all() and (r_part == r).all() and (r == want_r).all()
    assert (normed_coord == normed).all() and (normed == want_normed).all()
    assert (normed[-1] == 0x7FFF).all() and not (normed[0] == 0x7FFF).any()


@pytest.mark.parametrize("shot,N,cluster,fma,C", [(1, 256, 1, False, 32), (2, 512, 2, True, 64),
                                                  *[pytest.param(s, GLM53_N, c, True, 96, marks=pytest.mark.slow)
                                                    for s, c in GLM53_FUSED]])
def test_the_fused_call_is_its_chunks_then_its_norm(shot: int, N: int, cluster: int, fma: bool, C: int) -> None:
    """Each rank's row cut into its chunk rows (`RowChunks_v1`), `row_binding`'s chunk units of either fused Definition, each on
    the ranks' chunk rows and the whole residual row, gathered into one row (`AllGather_v1`), are the fused call's r, and
    `RmsNormMnnvl_v1` on that row is its normed row."""
    rng = np.random.default_rng(N + 10 * shot + cluster + C)
    parts, res, w = _tokens(rng, 2, N)
    M, ws, rows = len(res), np.broadcast_to(w, res.shape), N // C
    statics = {"WORLD": 8, "N": N, "SHOT": shot, "CLUSTER": cluster, "EPS": 1e-5, "FMA": fma}
    (both,) = _ev(bind(TA.AllReduceRmsNormMnnvl, **statics), parts.reshape(M, -1), res, ws)
    normed, r = both[:, :N], both[:, N:]
    split = bind(TC.RowChunks, N=N, C=C)
    assert layout(split).outs == tuple((f"o{j}", C) for j in range(rows))
    ranks = np.stack([_ev(split, parts[:, p])[0] for p in range(8)], axis=1)
    chunks = []
    for at in range(rows):
        binding = TC.row_binding(TA.AllReduceRmsNormMnnvl, statics, rows, at)
        assert binding == TC.row_binding(TA.AllReduceResidualMnnvl, {"WORLD": 8, "N": N, "SHOT": shot}, rows, at)
        fn = _bind(binding)
        assert fn.id == f"AllReduceResidualMnnvlAt_v1{{WORLD=8,N={N},C={C},AT={at},SHOT={shot}}}"
        chunks.append(_ev(fn, ranks[:, :, at * C:(at + 1) * C].reshape(M, -1), res)[0])
    (r_chunks,) = _ev(bind(b1_tp2.AllGather, WORLD=rows, N=C), np.concatenate(chunks, axis=1))
    (normed_chunks,) = _ev(bind(TA.RmsNormMnnvl, N=N, SHOT=shot, CLUSTER=cluster, EPS=1e-5, FMA=fma), r_chunks, ws)
    assert (r_chunks == r).all() and (normed_chunks == normed).all()
    assert (normed[-1] == 0x7FFF).all() and not (normed[0] == 0x7FFF).any()


@pytest.mark.parametrize("shot", [1, 2])
def test_the_fused_coordinate_on_every_special_pair_and_random_words(shot: int) -> None:
    """Every pair of special words at ranks 0 and 1 (zeros elsewhere) with a special residual, and uniform random words: the
    coordinate is `AllReduceResidualMnnvl_v1`'s column."""
    rng = np.random.default_rng(shot)
    a, b = np.meshgrid(SPECIALS, SPECIALS)
    grid = np.zeros((a.size, 8), np.uint16)
    grid[:, 0], grid[:, 1] = a.ravel(), b.ravel()
    cols = np.concatenate([grid, rng.integers(0, 1 << 16, (512, 8)).astype(np.uint16)])
    res = np.concatenate([SPECIALS[rng.integers(0, len(SPECIALS), a.size)], rng.integers(0, 1 << 16, 512).astype(np.uint16)])
    n = len(res) // 4
    (have,) = _ev(bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=shot), cols, res[:, None])
    parts = cols[:4 * n].reshape(n, 4, 8).transpose(0, 2, 1).reshape(n, -1)
    (want,) = _ev(bind(TA.AllReduceResidualMnnvl, WORLD=8, N=4, SHOT=shot), parts, res[:4 * n].reshape(n, 4))
    assert (have[:4 * n].reshape(n, 4) == want).all()
    a = K.mnnvl_rows(cols.T, shot)
    assert (have[:, 0] == K.DR.f2fp_bf16(K.DR.add_rn(K.DR.bf16_to_f32_words(a), K.DR.bf16_to_f32_words(res)))).all()


def _check_coordinates(d, statics: dict, x: np.ndarray) -> tuple[np.ndarray, int]:
    """The call ``d`` at ``statics`` on rows ``x`` against `coordinate_binding`'s unit at every position, each unit's positions
    in one batch; (the call's output, the number of distinct units)."""
    (out,) = _ev(bind(d, **statics), x)
    n, size = out.shape
    cols = x.reshape(n, 8, size)
    groups: dict[str, tuple] = {}
    for j in range(size):
        fn = _bind(TC.coordinate_binding(d, statics, j))
        groups.setdefault(fn.id, (fn, []))[1].append(j)
    for fn, js in groups.values():
        (have,) = _ev(fn, cols[:, :, js].transpose(0, 2, 1).reshape(n * len(js), 8))
        assert (have.reshape(n, len(js)) == out[:, js]).all(), fn.id
    return out, len(groups)


@pytest.mark.parametrize("d,statics,units", [
    (TA.AllReduceMnnvl, {"WORLD": 8, "N": 24, "SHOT": 1}, 1),
    (TA.AllReduceMnnvl, {"WORLD": 8, "N": 24, "SHOT": 2}, 1),
    (TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 96, "TREES": TREES, "FTZ": False}, 2),
    (TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 96, "TREES": TREES, "FTZ": True}, 2),
    (TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 96, "TREES": (TA.GLM53_H200_TREE,) * 8, "FTZ": False}, 1),
    (TA.AllReduceCustom, {"WORLD": 8, "SIZE": 96, "STAGE": 2}, 8),
    (TA.AllReduceCustom, {"WORLD": 8, "SIZE": 96, "STAGE": 1}, 1),
    (TA.AllReduceNcclTree, {"WORLD": 8, "SIZE": 24, "CHAIN": (3, 0, 7, 1, 6, 2, 5, 4)}, 1),
], ids=lambda v: getattr(v, "name", None))
def test_every_coordinate_is_its_unit(d, statics: dict, units: int) -> None:
    """A position-dependent route's witness row differs by position, so its units' statics are read."""
    out, n = _check_coordinates(d, statics, _words(np.random.default_rng(len(str(statics))), int(statics.get("SIZE", statics.get("N")))))
    assert n == units
    assert (len(set(out[0].tolist())) > 1) == (units > 1)


@pytest.mark.slow
def test_every_coordinate_of_glm53s_served_calls_is_its_unit() -> None:
    """MNNVL one-shot (one row of a 4-token launch) and two-shot (one row of 42), multimem at the capture's trees (one row of 43),
    and the multimem and custom 2-stage kernels at 2 rows of GLM-5.3's width (1536 elements an owner) with an owner's order that
    differs."""
    rng = np.random.default_rng(7)
    calls = [TA.route_binding(_served("fi_ar_comm"), 8, 4, GLM53_N, 1), TA.route_binding(_served("fi_ar_comm"), 8, 42, GLM53_N, 1),
             TA.route_binding(_served("symm_mem_comm"), 8, 43, GLM53_N, 1),
             (TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 2 * GLM53_N, "TREES": TREES, "FTZ": False}),
             (TA.AllReduceCustom, {"WORLD": 8, "SIZE": 2 * GLM53_N, "STAGE": 2})]
    assert [s.get("SHOT") for _, s in calls[:2]] == [1, 2]
    for (d, s), units in zip(calls, (1, 1, 1, 2, 8)):
        assert _check_coordinates(d, s, _words(rng, int(s.get("SIZE", s.get("N")))))[1] == units


def _check_rows(d, statics: dict, rows: int, x: np.ndarray) -> list:
    """The call ``d`` at ``statics`` on rows ``x`` against `row_binding`'s unit at every one of its `rows` rows; each row's unit,
    bound."""
    (out,) = _ev(bind(d, **statics), x)
    n, size = out.shape
    N, cols = size // rows, x.reshape(n, 8, size)
    units = []
    for row in range(rows):
        fn = _bind(TC.row_binding(d, statics, rows, row))
        (have,) = _ev(fn, cols[:, :, row * N:(row + 1) * N].reshape(n, -1))
        assert (have == out[:, row * N:(row + 1) * N]).all(), (fn.id, row)
        units.append(fn)
    if len({fn.id for fn in units}) > 1:
        assert any((_ev(units[0], cols[:1, :, row * N:(row + 1) * N].reshape(1, -1))[0] != out[:1, row * N:(row + 1) * N]).any()
                   for row in range(1, rows)), "row 0's unit computes every row of the witness"
    return units


@pytest.mark.parametrize("d,statics,rows,one", [
    (TA.AllReduceMnnvl, {"WORLD": 8, "N": 72, "SHOT": 1}, 3, True),
    (TA.AllReduceMnnvl, {"WORLD": 8, "N": 72, "SHOT": 2}, 3, True),
    (TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 96, "TREES": TREES, "FTZ": False}, 4, False),
    (TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 96, "TREES": TREES, "FTZ": True}, 4, False),
    (TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 96, "TREES": (TA.GLM53_H200_TREE,) * 8, "FTZ": False}, 4, True),
    (TA.AllReduceCustom, {"WORLD": 8, "SIZE": 64, "STAGE": 2}, 4, False),
    (TA.AllReduceCustom, {"WORLD": 8, "SIZE": 96, "STAGE": 2}, 4, False),
    (TA.AllReduceCustom, {"WORLD": 8, "SIZE": 64, "STAGE": 1}, 4, True),
    (TA.AllReduceNcclTree, {"WORLD": 8, "SIZE": 72, "CHAIN": (3, 0, 7, 1, 6, 2, 5, 4)}, 3, True),
], ids=lambda v: getattr(v, "name", None))
def test_every_row_is_its_unit(d, statics: dict, rows: int, one: bool) -> None:
    """An offset-free route's rows are one template; the others' row units differ by ROW, and row 0's unit on another row's words
    differs from the call there."""
    units = _check_rows(d, statics, rows, _words(np.random.default_rng(rows), int(statics.get("SIZE", statics.get("N")))))
    assert (len({fn.id for fn in units}) == 1) == one


@pytest.mark.slow
def test_every_row_of_glm53s_served_calls_is_its_unit() -> None:
    """MNNVL one-shot (2 rows of a 4-token launch), multimem at the capture's trees (2 rows of 43, whose row unit is
    `route_binding`'s one row), and the multimem and custom 2-stage kernels at 2 rows of GLM-5.3's width (owners 0-3 the first
    row, 4-7 the second) with an owner's order that differs."""
    rng = np.random.default_rng(11)
    multimem = _served("symm_mem_comm")
    for (d, s), rows in ((TA.route_binding(_served("fi_ar_comm"), 8, 4, GLM53_N, 2), 2),
                         (TA.route_binding(multimem, 8, 43, GLM53_N, 2), 2),
                         ((TA.AllReduceMultimem, {"WORLD": 8, "SIZE": 2 * GLM53_N, "TREES": TREES, "FTZ": False}), 2),
                         ((TA.AllReduceCustom, {"WORLD": 8, "SIZE": 2 * GLM53_N, "STAGE": 2}), 2)):
        units = _check_rows(d, s, rows, _words(rng, int(s.get("SIZE", s.get("N")))))
        if d is TA.AllReduceMultimem and s["TREES"] == multimem.statics["TREES"]:
            assert {fn.id for fn in units} == {_bind(TA.route_binding(multimem, 8, 43, GLM53_N, 1)).id}


def test_the_units_refuse_what_they_do_not_state() -> None:
    fused = {"WORLD": 8, "N": 256, "SHOT": 1, "CLUSTER": 1, "EPS": 1e-5, "FMA": False}
    with pytest.raises(ValueError, match="elements are 0..255"):
        TC.coordinate_binding(TA.AllReduceRmsNormMnnvl, fused, 256)
    with pytest.raises(ValueError, match="rows of equal width"):
        TC.row_binding(TA.AllReduceRmsNormMnnvl, fused, 3, 0)
    with pytest.raises(ValueError, match="whole chunks of a residual row"):
        bind(TC.AllReduceResidualMnnvlAt, WORLD=8, N=64, C=24, AT=0, SHOT=1).body  # noqa: B018
    with pytest.raises(ValueError, match="into whole chunks"):
        bind(TC.RowChunks, N=GLM53_N, C=100)
    with pytest.raises(ValueError, match="rows of equal width"):
        TC.row_binding(TA.AllReduceMnnvl, {"WORLD": 8, "N": 96, "SHOT": 1}, 5, 0)
    with pytest.raises(ValueError, match="rows of equal width"):
        TC.row_binding(TA.AllReduceMnnvl, {"WORLD": 8, "N": 96, "SHOT": 1}, 4, 4)
    with pytest.raises(ValueError, match="combine units are stated here"):
        TC.coordinate_binding(TA.AllReduceMultimemVector, {"WORLD": 8, "TREE": TA.GLM53_H200_TREE, "FTZ": False}, 0)
    with pytest.raises(ValueError, match="ROW one of"):
        bind(TC.AllReduceCustomRow, WORLD=8, N=16, ROWS=4, ROW=4, STAGE=2).body  # noqa: B018
    with pytest.raises(ValueError, match="whole 8-element vectors"):
        bind(TC.AllReduceMultimemRow, WORLD=8, N=12, ROWS=4, ROW=0, TREES=TREES, FTZ=False).body  # noqa: B018


def _block(fn, cuts, on_bits: bool = True) -> float:
    """The share of a 2^K_MAX-bit C-Flock block that ``fn`` under ``cuts`` takes as one unit, as `verity_flock.circuit.compose`
    places it at one unit a block: its rows' sha512x3 and hm96 slots (`check_proof_block`'s) and the mask beside the unit's own
    slot, the least power of two above the ANDs and port bits read off its Boolean version (``on_bits``), or off ``fn`` itself
    (`unit_witness`, never lowered). Past 1, compose refuses it."""
    from verity_flock import circuit as FC

    bits = BB.boolean_version(fn) if on_bits else fn
    w = FC.unit_witness(bind(bits) if isinstance(bits, CompositeDefinition) else bits)
    lay = layout(fn)
    ports = [FC.Port(p, words, 16 * words) for p, words in [(p, x) for p, x, _ in lay.cut(cuts)] + list(lay.outs)]
    ranges, _ = FC._ranges([("unit", FC.SHA_LOG, FC._row_slots(ports)[0], {"packed": True}),
                            ("unit", FC.HM_LOG, FC._pow2(len(ports)), {}), ("unit", w["k"], 1, {}), ("mask", FC.MASK_LOG, 1, {})])
    return max((r["pos0"] + r["count"]) << r["slot_log"] for r in ranges) / (1 << K_MAX)


def test_glm53s_units_at_chunks_of_96_fit_a_flock_block_and_its_coordinates_and_whole_rows_do_not() -> None:
    """A GLM-5.3 call of N = 6144 a token at chunks of K = 96: a rank's K one-word producer rows gathered into a row
    (`AllGather_v1{WORLD=K, N=1}`), or its one-row partial (the MoE's) cut into chunk rows (`RowChunks_v1{N=6144, C=K}`), each
    chunk's combine reading the 8 ranks' chunk rows (`row_binding`'s), the chunks gathered into the call's row, and its readers
    (the fused call's norm at each served launch) reading that row whole; its logits (8 ranks of 19,360 one-word LM-head rows)
    gathered 121 words a row, which `TokenSelectGroups_v1{C=1936, G=16}` reads 16 of. Each fits a C-Flock block, its rows (at
    most 256, one past 111 B) and its gates, with a tenth of it to spare. A unit of a whole row does not, though its rows alone
    would: the fused call's at any served launch, nor an all-reduce's or its residual add's. A coordinate
    unit, whose rows are a word each, does not, nor does a chunk combine reading words (8V rows: at V >= 56 one row would pass
    111 B, past 256 rows), nor a norm reading 256 chunks (no divisor of 6144 lies in 25..31), nor one token select over the whole
    vocabulary."""
    K, n = 96, GLM53_N // 96

    def fused(shot, cluster):
        return {"WORLD": 8, "N": GLM53_N, "SHOT": shot, "CLUSTER": cluster, "EPS": 1e-5, "FMA": True}

    norm = bind(TA.RmsNormMnnvl, N=GLM53_N, SHOT=1, CLUSTER=8, EPS=1e-5, FMA=True)
    fits = [
        (bind(b1_tp2.AllGather, WORLD=K, N=1), {"parts": K}),
        (bind(TC.RowChunks, N=GLM53_N, C=K), None),
        (_bind(TC.row_binding(TA.AllReduceRmsNormMnnvl, fused(1, 8), n, n - 1)), {"parts": 8}),
        (_bind(TC.row_binding(TA.AllReduceRmsNormMnnvl, fused(2, 2), n, 0)), {"parts": 8}),
        (_bind(TC.row_binding(TA.AllReduceMnnvl, {"WORLD": 8, "N": GLM53_N, "SHOT": 2}, n, 0)), {"parts": 8}),
        (_bind(TC.row_binding(TA.AllReduceCustom, {"WORLD": 8, "SIZE": GLM53_N, "STAGE": 2}, n, 7)), {"parts": 8}),
        (_bind(TC.row_binding(*TA.route_binding(_served("symm_mem_comm"), 8, 1, GLM53_N, 1), n, 0)), {"parts": 8}),
        (bind(b1_tp2.AllGather, WORLD=n, N=K), {"parts": n}),
        *[(bind(TA.RmsNormMnnvl, N=GLM53_N, SHOT=s, CLUSTER=c, EPS=1e-5, FMA=True), None) for s, c in GLM53_FUSED],
        (bind(b1_tp2.AllGather, WORLD=121, N=1), {"parts": 121}),
    ]
    for fn, cuts in fits:
        assert check_proof_block(fn, cuts) <= K_MAX and _block(fn, cuts) <= 0.9, fn.id
    select = bind(dead_free.TokenSelectGroupsV1, C=1936, G=16)
    assert check_proof_block(select, {"x": 16}) <= K_MAX and _block(select, {"x": 16}, on_bits=False) <= 0.9
    whole = [*[bind(TA.AllReduceRmsNormMnnvl, **fused(s, c)) for s, c in GLM53_FUSED],
             bind(TA.AllReduceResidualMnnvl, WORLD=8, N=GLM53_N, SHOT=1), bind(TA.AllReduceMnnvl, WORLD=8, N=GLM53_N, SHOT=2)]
    for fn in whole:
        assert check_proof_block(fn, {"parts": 8}) <= K_MAX and _block(fn, {"parts": 8}) > 1, fn.id
    for fn, cuts, why in [
        (bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=1), {"col": 8}, "no row past 111 B"),
        (bind(TA.AllReduceResidualMnnvl, WORLD=8, N=24, SHOT=1), {"parts": 8 * 24, "res": 24}, "no row past 111 B"),
        (bind(TA.AllReduceResidualMnnvl, WORLD=8, N=32, SHOT=1), {"parts": 8 * 32, "res": 32}, "past a proof block's 256"),
        (norm, {"r": 256}, "258 rows"),
        (bind(b1.TokenSelect, V=154880), {"logits": 8}, r"2\^28 bits"),
    ]:
        with pytest.raises(ValueError, match=why):
            check_proof_block(fn, cuts)


@pytest.mark.parametrize("word", [
    bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=1), bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=2),
    bind(TC.AllReduceResidualMnnvlAt, WORLD=8, N=64, C=16, AT=2, SHOT=1),
    bind(TC.AllReduceResidualMnnvlAt, WORLD=8, N=64, C=16, AT=3, SHOT=2),
    bind(TC.AllReduceMultimemRow, WORLD=8, N=24, ROWS=4, ROW=2, TREES=TREES, FTZ=False),
    bind(TC.AllReduceMultimemRow, WORLD=8, N=24, ROWS=4, ROW=2, TREES=TREES, FTZ=True),
    bind(TC.AllReduceCustomRow, WORLD=8, N=16, ROWS=4, ROW=1, STAGE=2),
    bind(TC.RowChunks, N=32, C=8),
], ids=lambda w: w.id.split("{")[0] + ("_ftz" if "FTZ=True" in w.id else ""))
def test_the_units_on_bits_compute_their_words(word) -> None:
    bits = BB.boolean_version(word)
    bits = bind(bits) if isinstance(bits, CompositeDefinition) else bits
    assert bits.defn is getattr(BC, word.name) and bits.version == 2 and bits.defn.sigma_of is word.defn
    assert bits.bindings == word.bindings
    rng = np.random.default_rng(5)
    widths = [p.leaves for _, p in word.signature()[0]]
    args = [np.concatenate([SPECIALS[rng.integers(0, len(SPECIALS), (16, k))], rng.integers(0, 1 << 16, (16, k))]) for k in widths]
    if word.name == "AllReduceMultimemRow":
        args[0][0] = np.repeat(WITNESS, widths[0] // 8)
    want = np.asarray(evaluate_batch(word, *(a.astype(np.uint64) for a in args)), dtype=object).reshape(32, -1)
    have = np.asarray(evaluate_words(bits, *(a.astype(np.uint64) for a in args)), dtype=object).reshape(32, -1)
    assert (want == have).all()
