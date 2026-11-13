"""The pass verifier (``benchmarks/pouw/pearl_c_vllm/verify_run.py``) at the LM head's shape on the CPU: one decode call of
Llama 3.2 1B's head (its tied embedding), (16, 2,048, 128,256) with 4 real rows, under the served instance (k_min 1,024, 64 x 64 tiles) of the
scheme the H200 runs, its weight's 128,256 vocabulary rows committed as ``pearl-b/64`` and one of them all +0, excluded.
The call has 2,004 column tiles and its y rows are 501 segments of 256 words.

Only the tiles the verifier opens are proved (the scheme's reference, on `fast_tile`'s batch forms): the draw is fixed to
the call's first, a middle and its last column tile (``clear_work.draw`` patched), and the exclusion tile of the +0 row
opens beside them. Every other transcript leaf and y word is +0, which no opened tile reads. The Lean verifier draws from
its own record of the coins, so this pass is the Python's alone; a fully proved one is `head_pass`'s with ``prove=None``
(about 2,004 tiles' proofs, ``jobs`` processes at a time).

The module's weight is 525 MB of BF16, built and committed once a module (about 85 s), and the two verdicts take about
70 s more. What it catches that the
128-word test instances of `test_pearl_c_vllm` can't: a verifier or y check that assumes a call's n fits a smaller row
(a 64-row tile grid, y's segments, W's weight entry at 128,256 leaves), and the head's credit at the linears' rate."""
from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


T = _module("test_pearl_c_vllm", HERE / "test_pearl_c_vllm.py")

SCHEME = "pearl-c-h100-v1-h3"
M16, K, N, REAL, WEIGHT, INDEX = 16, 2_048, 128_256, 4, 64, 0
#: the vocabulary row that is all +0 (it fails liveness, so it is declared excluded), in column tile 2,001
ZERO_ROW = 128_100
#: the column tiles the draw is fixed to: the first, one in the middle and the last
DRAWN = (0, 1_002, 2_003)
EXCLUSION = ZERO_ROW // 64
KEY = bytes([1]) * 32
DRAWS = 3
SOURCE = b"verifier" * 4
CHUNK = 8_192


#: what `_prove` reads, set before `head_pass` forks its workers
_PROVING: dict = {}


def _install() -> None:
    T.SD.FT.install(_PROVING["s"])


def _prove(ts: list[int]) -> list[tuple[int, bytes, np.ndarray]]:
    """Tiles `ts` proved on the batch forms: each one's id, transcript digest and real rows' y words."""
    s, ctx, noise, tiles, a_rows, b_rows = (_PROVING[x] for x in ("s", "ctx", "noise", "tiles", "a_rows", "b_rows"))
    out = []
    for t in ts:
        tile = tiles[t]
        ops = s.encode(ctx, noise, [a_rows[i] for i in tile.rows], [b_rows[j] for j in tile.cols], tile)
        ch = s.checked(ctx, ops)
        real = [x for x, i in enumerate(tile.rows) if i < REAL]
        y = np.array([[T.PCU.y(ch.u[x][z], ops.a.alpha[x], ops.b.alpha[z]) for z in range(len(tile.cols))] for x in real],
                     "<u2")
        out.append((t, s.digest(ctx, noise, ch), y))
    return out


def head_pass(d: Path, prove: tuple[int, ...] | None = (*DRAWN, EXCLUSION), jobs: int = 1) -> Path:
    """The head's pass in the device run's layout under `d`, its manifest's path. Tiles `prove` are proved (every tile
    with None), in `jobs` forked processes, the rest of the transcript and of y left +0."""
    s = T.PPD.scheme_of(SCHEME)
    rng = np.random.default_rng(64)
    h = s.commitment_hash
    (d / "calls").mkdir(parents=True)
    (d / "weights").mkdir()
    dom_b = T.rows_domain(T.weight_binding(WEIGHT), -2, N, h)
    tiles = s.tiles(T.Shape(M16, K, N))
    keep = {j for t in (range(len(tiles)) if prove is None else prove) for j in tiles[t].cols}
    b_rows, b_leaves = {}, []
    with open(d / "weights" / f"{WEIGHT}.rows.bf16", "wb") as f:
        for j0 in range(0, N, CHUNK):
            w = T.bf16_words(rng, min(CHUNK, N - j0), K)
            if j0 <= ZERO_ROW < j0 + len(w):
                w[ZERO_ROW - j0] = 0
            f.write((w >> 16).astype("<u2").tobytes())
            for r, row in enumerate(w):
                j, v = j0 + r, row.tobytes()
                b_leaves.append(dom_b.leaf(j, j, T.ROW_SCHEMA, v + T.hm96_salt(KEY, T.HM96_WEIGHTS + WEIGHT, j)))
                if j in keep:
                    b_rows[j] = v
    tree_b = T.laid_out(dom_b, b_leaves)
    root_b = tree_b[-64:]
    weights = {str(WEIGHT): {"n": N, "k": K, "root": root_b.hex(), "excluded_b": [ZERO_ROW],
                             "rows": T.PH._entry(d / "weights" / f"{WEIGHT}.rows.bf16", N * K * 2),
                             "tree": T.PPD._write(d / "weights" / f"{WEIGHT}.tree", tree_b)}}
    roots = {WEIGHT: root_b}
    coins = T.salt_coin(roots)
    mm = T.Matmul(INDEX, T.Shape(M16, K, N), WEIGHT)
    epoch = T.Epoch.start(T.Workload((mm,)), roots, *coins)
    a = T.bf16_words(rng, M16, K)
    a[REAL:] = 0
    a_rows = [r.tobytes() for r in a]
    dom_a = T.rows_domain(T.activation_binding(epoch.salt, INDEX), INDEX, M16, h)
    tree_a = T.laid_out(dom_a, [dom_a.leaf(i, i, T.ROW_SCHEMA, r + T.hm96_salt(KEY, INDEX, i)) for i, r in enumerate(a_rows)])
    root_a = tree_a[-64:]
    ctx = T.Context(epoch.salt, mm, root_a, root_b)
    noise = s.noise(ctx)
    leaves = [bytes(64)] * len(tiles)
    y = np.zeros((REAL, N), "<u2")
    todo = list(range(len(tiles)) if prove is None else prove)
    _PROVING.update(s=s, ctx=ctx, noise=noise, tiles=tiles, a_rows=a_rows, b_rows=b_rows)
    try:
        if jobs == 1:
            undo = T.SD.FT.install(s)
            try:
                proved = _prove(todo)
            finally:
                undo()
        else:
            import multiprocessing
            with multiprocessing.get_context("fork").Pool(jobs, initializer=_install) as pool:
                proved = [p for part in pool.imap(_prove, [todo[i:i + 8] for i in range(0, len(todo), 8)]) for p in part]
    finally:
        _PROVING.clear()
    for t, leaf, block in proved:
        leaves[t] = leaf
        y[[i for i in tiles[t].rows if i < REAL], tiles[t].cols[0]:tiles[t].cols[-1] + 1] = block
    splits = {WEIGHT: T.PW.Split(K)}
    records = [T.PW.UnitRecord(mm, REAL, K, (), (ZERO_ROW,), (), ())]
    matmuls = [{"index": INDEX, "m": M16, "k": K, "n": N, "weight": WEIGHT, "real_rows": REAL, "k_kept": K, "excluded_a": [],
                "root_a": root_a.hex(),
                "rows": T.PPD._write(d / "calls" / f"{INDEX}.rows.bf16", T.bf16_file(a_rows), digest=False),
                "tree": T.PPD._write(d / "calls" / f"{INDEX}.tree", tree_a),
                "y": T.SY.retain_y(d, INDEX, y.tobytes(), N, epoch.salt, KEY)}]
    dt = T.transcript_domain(epoch, len(leaves), h)
    tree_t = T.laid_out(dt, [dt.leaf(i, i, T.TILE_SCHEMA, v + T.hm96_salt(KEY, T.HM96_TRANSCRIPT, i)) for i, v in enumerate(leaves)])
    head = {"schema": T.CC.MANIFEST_SCHEMA, "run": T.RUN, "entry": f"pouw:{SCHEME}", "scheme": SCHEME,
            "coins": coins[0].to_json(), "salt_opening": coins[1].to_json(), "salt": epoch.salt.hex(),
            "records_root": T.PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex(),
            "commitment": {"root": tree_t[-64:].hex(), "count": len(leaves)},
            "leaves": T.PPD._write(d / "leaves", b"".join(leaves)),
            "tree_t": T.PPD._write(d / "tree_t", tree_t), "hm96_salt_key": KEY.hex()}
    path = T.PPD.write_manifest(d, head, weights, matmuls)
    (d / "coins.json").write_text(json.dumps(coins[0].to_json()))
    return path


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    return head_pass(tmp_path_factory.mktemp("head"))


@pytest.fixture
def fixed_draw(monkeypatch):
    tiles = T.PPD.scheme_of(SCHEME).tiles(T.Shape(M16, K, N))
    monkeypatch.setattr(T.VR.CW, "draw", lambda scheme, records, key, draws: [(0, tiles[t]) for t in DRAWN][:draws])


def verdict(path: Path, **kw) -> dict:
    return T.VR.verify_passes([path], DRAWS, 1, "test", T.published(path), SOURCE, fast=True, **kw)


@pytest.mark.slow
def test_the_verifier_accepts_the_heads_call_opens_its_last_column_tile_and_credits_it_at_the_linears_rate(head, fixed_draw):
    """An honest head call is accepted, tied to a W that registers its A, its y of 4 rows of 128,256 words and its weight
    of 128,256 rows; it opens the drawn tiles and the +0 row's exclusion tile, holds y's words in the last column tile to
    its U, and credits the call's 4 real rows at |K| over every column but the +0 row's block of 64 (the band rule,
    `UnitRecord.credited`: its tile credits nothing), less ε and ρ. The first
    and the last column tile are replayed on the scalar reference too, whose verdict the batch forms' must be."""
    got = verdict(head, window=T.window_record(head), cross_check=2)
    assert got["answer"]["verdict"] == "accept", got["line"]
    profile, public = got["answer"]["profile"], got["answer"]["public"]
    assert profile["tiles"] == [[0, t] for t in (*DRAWN, EXCLUSION)] and profile["exclusion_tiles"] == 1
    assert all(v["ok"] for v in got["verdicts"]) and got["replay"] == "batch"
    assert "y committed and checked on 1 of 1 calls" in got["line"] and "on its 3 tables" in got["line"], got["line"]
    assert public["tie"]["tables"] == 3 and "weights" not in public["tie"]
    work = public["work"]
    eps, rho = Fraction(work["epsilon"]), T.PPD.scheme_of(SCHEME).cap
    assert work["macs"] == REAL * (N - 64) * K and work["credited_tiles"] == 2_003
    assert work["credited"] == int((1 - rho) * (1 - eps) * work["macs"]) and eps == T.VR.epsilon(DRAWS, T.VR.DELTA)
    entry = next(e for e in T.window_record(head)["registered"] if e["value"] == f"pearl-b/{WEIGHT}")
    assert entry["leaves"] == N and entry["words"] == K


@pytest.mark.slow
def test_a_wrong_y_word_in_the_heads_last_column_tile_rejects_the_run(head, fixed_draw):
    """The call's y with its real row 3's last word changed, committed as it is (so the run's commitment is another), is
    refused on the last column tile: that word isn't the unscale of the tile's U."""
    man = json.loads(head.read_text())
    u = man["matmuls"][0]
    good = u["y"]
    y = np.frombuffer((head.parent / good["rows"]["path"]).read_bytes(), "<u2").reshape(REAL, N).copy()
    y[3, N - 1] ^= 1
    try:
        u["y"] = T.SY.retain_y(head.parent, INDEX, y.tobytes(), N, bytes.fromhex(man["salt"]), KEY)
        head.write_text(json.dumps(man))
        got = verdict(head)
        assert got["answer"]["verdict"] == "reject", got["line"]
        assert f"pass 0 tile 2003: y[3, {N - 1}] is not the unscale of the tile's U" in got["answer"]["reason"], got["line"]
    finally:
        y[3, N - 1] ^= 1
        u["y"] = T.SY.retain_y(head.parent, INDEX, y.tobytes(), N, bytes.fromhex(man["salt"]), KEY)
        head.write_text(json.dumps(man))
