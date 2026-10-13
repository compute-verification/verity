"""The H200 capture of DeepSeek-V4-Pro's head-512 compressor (`dsv4_compressor_capture_gpu`) on the CPU: its requests' slots,
the state rows and cache rows it reads back, its launch checks and comparison jobs on what the twin says the kernels write, the
edge words, its verdicts and its main loop with the GPU stood in for."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from tests.properties import dsv4_compressor_capture_gpu as X
from tests.properties import dsv4_flashmla_capture_gpu as FX
from verity_vllm.program.kernels import dsv4_compressor_twin as T
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.properties.record import REGISTRY, digest_of

LAYERS = list(X.LAYERS)


def put_rows(cache: np.ndarray, slots, rows: np.ndarray, pbs: int) -> None:
    """(n, 520) rows [NoPE | RoPE | scales] written at cache slots ``slots`` of a (n pages, page_bytes(pbs)) byte cache."""
    blk, off = np.divmod(np.asarray(slots, dtype=np.int64), pbs)
    nb = cache.shape[0]
    data = cache[:, :pbs * X.DATA_BYTES].reshape(nb, pbs, X.DATA_BYTES)
    data[blk, off, :X.NOPE] = rows[:, :X.NOPE]
    data[blk, off, X.NOPE:] = np.ascontiguousarray(rows[:, X.NOPE:X.NOPE + X.ROPE], dtype="<u2").view(np.uint8)
    cache[:, pbs * X.DATA_BYTES:pbs * X.TOKEN_BYTES].reshape(nb, pbs, X.SCALE_BYTES)[blk, off] = rows[:, X.NOPE + X.ROPE:]


def simulate(layer: str, o: dict):
    """What `launches` would return were the kernels the twin: `save_partial_states`' rows (kv, then score + ape[pos % ratio]) at
    each state slot, the compressed rows at each cache slot, and on ratio 128 the split call's columns.  -> runs, split, the rows
    written and the tokens they are of."""
    L = X.LAYERS[layer]
    saved = o["state0"].copy()
    t = np.flatnonzero(o["sslot"] >= 0)
    blk, row = np.divmod(o["sslot"][t], L.block)
    X.state_rows_of(saved, layer)[blk, row] = np.concatenate(
        [o["kv"][t], FT.fadd(o["score"][t], o["ape"][o["pos"][t] % L.ratio])], axis=1)
    k = np.full((X.NBK[layer], X.page_bytes(L.pbs)), X.SENTINEL_K, np.uint8)
    a = np.flatnonzero(X.compressed(o, layer))
    kv, sc, nrows = X.state_rows(saved, o["sbt"], o["req"][a], o["pos"][a], layer)
    rows = X.entries_twin(layer, kv, sc, nrows, o["w"], o["cs"][(o["pos"][a] // L.ratio) * L.ratio])
    put_rows(k, o["kslot"][a], rows, L.pbs)
    runs = [{"saved": saved, "after": saved.copy(), "k": k}, {"saved": saved.copy(), "after": saved.copy(), "k": k.copy()}]
    if L.ratio != 128:
        return runs, None, rows, a
    cols = np.full((o["pos"].size, X.D), X.SENTINEL_STATE, np.uint32)
    c = np.flatnonzero(X.columned(o))
    kv, sc, _ = X.state_rows(saved, o["sbt"], o["req"][c], o["pos"][c], layer)
    with np.errstate(all="ignore"):
        cols[c] = T.c128_columns(np.swapaxes(kv, -1, -2), np.swapaxes(sc, -1, -2), X.W128, X.R128)
    return runs, {"cols": cols, "state": saved.copy(), "k": k.copy()}, rows, a


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    """Each layer's operands, the twin's words as the GPU's, and `observe`'s records and quick jobs on them."""
    out = {}
    for layer in LAYERS:
        o = X.operands(layer, X.SEEDS[layer])
        runs, split, rows, a = simulate(layer, o)
        work = tmp_path_factory.mktemp(layer)
        recs, jobs = X.observe(work, layer, o, runs, split, quick=True)
        out[layer] = SimpleNamespace(o=o, runs=runs, split=split, rows=rows, tokens=a, recs=recs, jobs=jobs)
    return out


def test_the_statics_are_dsv4s() -> None:
    """DeepSeek-V4-Pro (config.json at b5968e91): head 512 = NoPE 448 | RoPE 64, one UE8M0 scale per 64; the ratio-4 layer's state
    1024 wide in blocks of 4 and its 8-row window, the ratio-128 layer's 512 wide in blocks of 8 and its window of 8 warps of 16;
    the `fp8_ds_mla` pages the decode reads (64 and 2 rows, 37440 and 1728 bytes), and the state's pages padded past their rows."""
    assert (X.D, X.NOPE, X.ROPE, X.QB, X.W128, X.R128, X.WORDS) == (512, 448, 64, 64, 8, 16, 520)
    assert tuple(X.LAYERS["c4"]) == (4, 1024, 4, 8, 64) and tuple(X.LAYERS["c128"]) == (128, 512, 8, 128, 2)
    assert X.W128 * X.R128 == X.LAYERS["c128"].window and (X.TOKEN_BYTES, X.DATA_BYTES, X.SCALE_BYTES) == (584, 576, 8)
    assert X.page_bytes(64) == 37440 and X.page_bytes(2) == 1728 and X.STATE_PAGE == 9360
    assert all(L.block * 2 * L.width < X.STATE_PAGE for L in X.LAYERS.values())
    assert set(X.PTX_COUNTS) == {"c4", "c128", "store"} and len(X.SOURCES) == 4


@pytest.mark.parametrize("layer", LAYERS)
def test_the_requests_take_the_slots_the_launch_means(layer: str) -> None:
    """Every state slot and cache slot distinct and within its buffer, with spare blocks and pages no token reads; request 0's and
    1's every window end a compressed row (ratio 4: 512 and 64; ratio 128: 32 and 16), request 2's none, though its tokens with a
    state slot write their state row and ratio 128's at 255 its column; the rows before request 1 written before the launch."""
    L = X.LAYERS[layer]
    o = X.layout(layer, X.SEEDS[layer])
    s, k, r = o["sslot"], o["kslot"], o["req"]
    for slots, top in ((s[s >= 0], X.NBS[layer] * L.block), (k[k >= 0], X.NBK[layer] * L.pbs)):
        assert np.unique(slots).size == slots.size and slots.max() < top
    assert np.unique(s[s >= 0] // L.block).size < X.NBS[layer] and np.unique(k[k >= 0] // L.pbs).size < X.NBK[layer]
    act = X.compressed(o, layer)
    want = {"c4": (512, 64), "c128": (32, 16)}[layer]
    assert (int((act & (r == 0)).sum()), int((act & (r == 1)).sum()), int((act & (r == 2)).sum())) == (*want, 0)
    extra = np.flatnonzero(r == 2)
    assert [(int(o["pos"][t]), bool(s[t] >= 0), bool(k[t] >= 0)) for t in extra] == list(X.EXTRA[layer])
    if layer == "c128":
        col = X.columned(o)
        assert int((col & (r != 1)).sum()) == 33 and o["pos"][col & (r == 2)].tolist() == [255]
    else:
        assert o["pos"][act][0] == 3
    (s1, _, pre) = X.REQUESTS[layer][1]
    pf = o["prefilled"][:pre]
    assert np.array_equal(pf[:, 0] * L.block + pf[:, 1], o["sbt"][1, (np.arange(s1 - pre, s1)) // L.block] * L.block
                          + np.arange(s1 - pre, s1) % L.block)


@pytest.mark.parametrize("layer", LAYERS)
def test_the_operands_hold_the_sentinel_and_the_rows_written_before(layer: str) -> None:
    """The initial state is the sentinel but at the rows written before the launch, which are finite; kv, score and the rotary
    cache carry the edge words where `EDGE` and CS_SPECIAL put them, and the norm weight W_SPECIAL's."""
    L = X.LAYERS[layer]
    o = X.operands(layer, X.SEEDS[layer])
    s = X.state_rows_of(o["state0"], layer)
    pf = o["prefilled"]
    mask = np.zeros(s.shape[:2], dtype=bool)
    mask[pf[:, 0], pf[:, 1]] = True
    assert np.all(s[~mask] == X.SENTINEL_STATE) and np.all(o["state0"][:, L.block * 2 * L.width:] == X.SENTINEL_STATE)
    assert np.isfinite(s[mask].view(np.float32)).all()
    s0 = X.REQUESTS[layer][0][1]
    kv, sc = o["kv"].view(np.float32), o["score"].view(np.float32)
    for what, name, offs in X.EDGE[layer]:
        x = (kv if what == "kv" else sc)[[s0 + j for j in offs]]
        if name == "nan" or name.startswith("nan@"):
            assert np.isnan(x).any()
        elif name in ("inf", "neg_inf") or name.startswith(("inf@", "neg_inf@")):
            assert np.isinf(x).any()
        elif name == "zero":
            assert not x.any()
    cs = o["cs"].view(np.float32)
    a, b, c = (p[LAYERS.index(layer)] for p in X.CS_SPECIAL)
    assert np.isinf(cs[a, 2]) and np.isnan(cs[b, X.ROPE // 2 + 5]) and not cs[c, :X.ROPE // 2].any()
    assert all(int(o["w"][j]) == v for j, v in X.W_SPECIAL.items())


@pytest.mark.parametrize("layer", LAYERS)
def test_the_state_rows_are_the_window_through_the_block_table(layer: str) -> None:
    """`state_rows` on a state whose every word is its own (block, offset): row r of a token at p is position p - window + 1 + r
    of its request through `block_table`, its kv the word (r // ratio) * 512 + j of that row and its score that plus the width; on
    ratio 4 the token at 3 reads 4 rows, the 4 before position 0 kv 0 and score -inf."""
    L = X.LAYERS[layer]
    o = X.layout(layer, X.SEEDS[layer])
    state = (np.arange(X.NBS[layer], dtype=np.uint32)[:, None] << 14) | np.arange(X.STATE_PAGE, dtype=np.uint32)[None, :]
    a = np.flatnonzero(X.compressed(o, layer))[:3]
    kv, sc, nrows = X.state_rows(state, o["sbt"], o["req"][a], o["pos"][a], layer)
    for t, (p, q) in enumerate(zip(o["pos"][a], o["req"][a])):
        for r in (0, L.window // 2 - 1, L.window // 2, L.window - 1):
            pos = int(p) - L.window + 1 + r
            if pos < 0:
                assert not kv[t, r].any() and np.all(sc[t, r] == 0xFF800000)
                continue
            base = (int(o["sbt"][q, pos // L.block]) << 14) | (pos % L.block) * 2 * L.width + (r // L.ratio) * X.D
            assert kv[t, r, 7] == base + 7 and sc[t, r, 7] == base + L.width + 7
    assert nrows.tolist() == ([4, 8, 8] if layer == "c4" else [128] * 3)


def test_the_cache_rows_are_the_decodes_layout() -> None:
    """`slot_words` reads back `flashmla_capture_gpu.pack_cache`'s MODEL1 rows (the layout the decode's rows are read in): 448
    NoPE bytes, 64 RoPE words, 7 scale bytes and the eighth; `written_bytes` is every byte of the rows at those slots and no
    other."""
    rng = np.random.default_rng(3)
    for pbs in (64, 2):
        nb = 5
        n = nb * pbs
        nope = rng.integers(0, 256, (n, X.NOPE), dtype=np.uint8)
        scale = rng.integers(0, 256, (n, 7), dtype=np.uint8)
        rope = rng.integers(0, 1 << 16, (n, X.ROPE), dtype=np.uint16)
        fill = rng.integers(0, 256, (nb, X.page_bytes(pbs)), dtype=np.uint8)
        c = FX.pack_cache(nope, scale, rope, pbs, fill)
        slots = rng.permutation(n)[:7]
        got = X.slot_words(c, slots, pbs)
        eighth = fill[:, pbs * X.DATA_BYTES:pbs * X.TOKEN_BYTES].reshape(n, 8)[slots, 7]
        assert np.array_equal(got, np.concatenate([nope[slots], rope[slots], scale[slots], eighth[:, None]], axis=1))
        rows = rng.integers(0, 256, (slots.size, X.WORDS)).astype(np.int64)
        rows[:, X.NOPE:X.NOPE + X.ROPE] = rng.integers(0, 1 << 16, (slots.size, X.ROPE))
        c2 = c.copy()
        put_rows(c2, slots, rows, pbs)
        m = X.written_bytes(nb, pbs, slots)
        assert np.array_equal(X.slot_words(c2, slots, pbs), rows) and m.sum() == slots.size * X.TOKEN_BYTES
        assert not (c2 != c).reshape(-1)[~m].any()


@pytest.mark.parametrize("layer", LAYERS)
def test_the_launch_checks_hold_on_the_twins_words(observed, layer: str) -> None:
    ob = observed[layer]
    assert [(x["phase"], x["case"]) for x in ob.recs] == [c for c in X.CASES if c[1].split("-")[0] == layer]
    assert all(v is True for x in ob.recs for v in x["checks"].values()), [x["checks"] for x in ob.recs]


def _observe(tmp_path, layer: str, ob, runs=None, split=None) -> dict:
    recs, _ = X.observe(tmp_path, layer, ob.o, runs or ob.runs, split or ob.split, quick=True)
    return {(x["phase"], x["case"]): {k for k, v in x["checks"].items() if v is not True} for x in recs}


STATE = ((0, "saved"), (0, "after"), (1, "saved"), (1, "after"))


def test_the_launch_checks_catch_each_stray_write(observed, tmp_path) -> None:
    """A word written (by both runs alike) into a state row written before the launch, the state page's padding, a kv word not
    the copy, or a cache page's padding; a second run's cache unlike the first's; a compress that writes the state; the split
    call's cache unlike the served call's; a column at a token the compress kernel skips: each fails its own check and no other."""
    ob = observed["c4"]
    pf = ob.o["prefilled"][0]
    row = int(ob.o["sslot"][0])
    b, q = divmod(row, X.LAYERS["c4"].block)
    probes = [
        (STATE, (int(pf[0]), int(pf[1]) * 2 * X.LAYERS["c4"].width + 7), {("score", "c4"): {"unwritten_untouched"}}),
        (STATE, (0, X.STATE_PAGE - 1), {("score", "c4"): {"unwritten_untouched"}}),
        (STATE, (b, q * 2 * X.LAYERS["c4"].width + 5), {("score", "c4"): {"kv_copied"}}),
        (((0, "k"), (1, "k")), (0, -1), {("compress", "c4"): {"unwritten_untouched"}}),
        (((1, "k"),), (3, 5), {("compress", "c4"): {"deterministic"}, ("compress", "c4-edge"): {"deterministic"}}),
        (((1, "after"),), (1, 1), {("compress", "c4"): {"state_unchanged"}}),
    ]
    for where, at, want in probes:
        runs = copy.deepcopy(ob.runs)
        for i, key in where:
            runs[i][key][at] ^= 1
        got = _observe(tmp_path / "p", "c4", ob, runs=runs)
        assert {c: v for c, v in got.items() if v} == want, (where, at, got)
    ob = observed["c128"]
    k = ob.split["k"].copy()
    k[0, 0] ^= 1
    assert _observe(tmp_path / "q", "c128", ob, split={**ob.split, "k": k})[("columns", "c128")] == {"split_is_served"}
    cols = ob.split["cols"].copy()
    cols[np.flatnonzero(~X.columned(ob.o))[0], 9] = 0
    assert _observe(tmp_path / "r", "c128", ob, split={**ob.split, "cols": cols})[("columns", "c128")] == {"untouched"}


def _no_refs(monkeypatch) -> None:
    monkeypatch.setattr(X, "_ref_rows", lambda lo, hi, every: [])


def _flipped(tmp_path, job: dict, at: tuple[int, int]) -> dict:
    a = {k: np.array(v) for k, v in X._load(job["paths"]).items()}
    a["got"].reshape(a["got"].shape[0], -1)[at] ^= 1
    return {**job, "paths": X._save(tmp_path, **a)}


@pytest.mark.parametrize("layer", LAYERS)
def test_the_jobs_take_every_word_and_find_a_flipped_one(observed, layer: str, tmp_path, monkeypatch) -> None:
    """Every job against the twin (the references off): every state row's score words, every cache row's 520 words and every
    column word, 0 mismatches; one word flipped in each kind is found at its position and word."""
    _no_refs(monkeypatch)
    ob = observed[layer]
    L = X.LAYERS[layer]
    tot: dict = {}
    for j in ob.jobs:
        r = X.compare(j)
        assert "error" not in r, r
        t = tot.setdefault((r["phase"], r["case"]), [0, 0])
        t[0] += r["words"]
        t[1] += r["mismatch"]
    rec = {(x["phase"], x["case"]): x for x in ob.recs}
    want = {("score", layer): rec[("score", layer)]["rows_written"] * L.width,
            ("compress", layer): rec[("compress", layer)]["entries"] * X.WORDS,
            ("compress", f"{layer}-edge"): rec[("compress", f"{layer}-edge")]["entries"] * X.WORDS}
    if layer == "c128":
        want |= {("columns", c): rec[("columns", c)]["columns"] * X.D for c in ("c128", "c128-edge")}
    assert tot == {k: [v, 0] for k, v in want.items()}
    for kind, unit in (("score", 300), ("entries", 77), ("columns", 401)):
        jobs = [j for j in ob.jobs if j["kind"] == kind and j["rows"][1] - j["rows"][0] > 1]
        if not jobs:
            continue
        j = jobs[-1]
        lo = j["rows"][0]
        r = X.compare(_flipped(tmp_path / kind, j, (lo + 1, unit)))
        tokens = X._load(j["paths"])["tokens"]
        assert r["mismatch"] == 1 and r["first_mismatch"]["position"] == int(tokens[lo + 1]) and r["first_mismatch"]["word"] == unit


def test_the_jobs_reference_every_instance_or_a_sample(observed, tmp_path) -> None:
    """A full run references every instance; --quick every 16th served one and each job's first, every edge one, and so the
    ratio-4 token at 3 (its 4-row window) and ratio 128's first column."""
    assert X._ref_rows(0, 32, 16) == [0, 16] and X._ref_rows(5, 9, 16) == [5] and X._ref_rows(3, 6, 1) == [3, 4, 5]
    for layer, ob in observed.items():
        for j in ob.jobs:
            assert j["ref_every"] == (1 if j["case"].endswith("-edge") else X.QUICK_REF_EVERY), j
        _, full = X.observe(tmp_path / layer, layer, ob.o, ob.runs, ob.split, quick=False)
        assert {j["ref_every"] for j in full} == {1}
    first = [j for j in observed["c4"].jobs if j["kind"] == "entries"][0]
    assert first["rows"][0] == 0 and int(X._load(first["paths"])["nrows"][0]) == 4


def _one(job: dict, i: int) -> dict:
    return {**job, "rows": [i, i + 1], "ref_every": 1}


def _ref_jobs(ob, edge_off: int, layer: str) -> list[dict]:
    """The first score row, the first two cache rows, a NaN edge row (at request 1's offset ``edge_off``) and, on ratio 128, the
    first token's columns, each one instance."""
    by = {}
    for j in ob.jobs:
        by.setdefault((j["kind"], j["case"]), j)
    s1 = X.REQUESTS[layer][1][0]
    edge = by[("entries", f"{layer}-edge")]
    i = int(np.flatnonzero(np.asarray(X._load(edge["paths"])["tokens"]) == s1 + edge_off)[0])
    jobs = [_one(by[("score", layer)], 0), _one(by[("entries", layer)], 0), _one(by[("entries", layer)], 1),
            {**edge, "rows": [i, i + 1]}]
    return jobs + ([_one(by[("columns", layer)], 0)] if ("columns", layer) in by else [])


def test_the_ratio_4_references_are_the_registered_definitions(observed) -> None:
    """`Dsv4CompressorScore_v1{W=1024}` on a state row, `Dsv4MlaCompressC4_v1{ROWS=4,...}` on the token at 3 and `{ROWS=8,...}` on
    the next and on a NaN edge row, each against the twin."""
    for j in _ref_jobs(observed["c4"], 51, "c4"):
        r = X.compare(j)
        assert (r["reference"], r["twin_mismatch"], r["mismatch"]) == (1, 0, 0), (j["kind"], j["case"], r)


@pytest.mark.slow
def test_the_ratio_128_references_are_the_registered_definitions(observed) -> None:
    """`Dsv4CompressorScore_v1{W=512}`, `Dsv4MlaCompressC128_v1{W=8,R=16,...}` on two rows and a NaN edge row (one warp's 16 rows
    -inf), and `Dsv4MlaC128Column_v1{8,16}` on every column of a token, each against the twin (about 15 s: four 128-row
    references)."""
    for j in _ref_jobs(observed["c128"], 575, "c128"):
        r = X.compare(j)
        assert (r["reference"], r["twin_mismatch"], r["mismatch"]) == (1, 0, 0), (j["kind"], j["case"], r)


#: request 1's rows by offset, as the twin writes them (and the kernels must): NaN rows (every NoPE byte 0xFE, the max.f32 clamp
#: taking NaN to -448, every RoPE word 0x7FFF, every scale the floor 114): a NaN or +inf score in the window, ratio 4's -inf rows
#: leading the online softmax (75, 183, not 179 where they trail), ratio 128's one warp of -inf (575), a NaN kv word read; zero
#: rows (NoPE +-0, scales 114): kv 0, x2^70 (the sum of squares inf), x2^-140 and x1e-30 (it underflows), the 3e38 word; rows +-0
#: but one NaN NoPE byte at the column an inf kv word is read in; RoPE pairs a cos inf or a sin NaN makes non-finite
NAN_ROWS = {"c4": {51, 55, 63, 67, 71, 75, 115, 143, 147, 151, 183}, "c128": {319, 447, 575, 1087}}
ZERO_ROWS = {"c4": {87, 91, 95, 99, 103, 155, 167}, "c128": {703, 831, 959, 1471, 1855}}
ONE_NAN = {"c4": {111: 5, 119: 7}, "c128": {1599: 5, 1727: 7}}
ROPE_SPECIAL = {"c4": {11: [4, 5], 19: [10, 11]}, "c128": {63: [4, 5], 1343: [10, 11]}}


@pytest.mark.parametrize("layer", LAYERS)
def test_the_edge_words_land_where_they_are_meant_to(observed, layer: str) -> None:
    """Request 1's rows: the NaN, zero and one-NaN rows and the RoPE pairs NAN_ROWS .. ROPE_SPECIAL name, and every other row an
    ordinary one (every scale above the floor, every RoPE word finite); the zero norm weight's NoPE byte +-0 in every row not
    NaN."""
    ob = observed[layer]
    s1 = X.REQUESTS[layer][1][0]
    sel = ob.o["req"][ob.tokens] == 1
    offs = (ob.o["pos"][ob.tokens][sel] - s1).tolist()
    assert NAN_ROWS[layer] | ZERO_ROWS[layer] | set(ONE_NAN[layer]) | set(ROPE_SPECIAL[layer]) <= set(offs)
    for off, row in zip(offs, ob.rows[sel]):
        nope, rope, sc = row[:X.NOPE], row[X.NOPE:X.NOPE + X.ROPE], row[X.NOPE + X.ROPE:]
        assert sc[-1] == 0
        bad = np.flatnonzero((rope & 0x7F80) == 0x7F80).tolist()
        if off in NAN_ROWS[layer]:
            assert np.all(nope == 0xFE) and np.all(rope == 0x7FFF) and np.all(sc[:-1] == 114), off
            continue
        assert nope[3] in (0, 0x80) and bad == ROPE_SPECIAL[layer].get(off, []), (off, bad)
        if off in ZERO_ROWS[layer] or off in ONE_NAN[layer]:
            j = ONE_NAN[layer].get(off)
            assert np.all(sc[:-1] == 114) and np.isin(np.delete(nope, [] if j is None else [j]), (0, 0x80)).all(), off
            assert j is None or nope[j] == 0xFE
        else:
            assert np.all(sc[:-1] > 114), off


def test_ptx_counts_reads_each_pinned_instruction() -> None:
    """Every PTX_COUNTS key is one PTX_COUNT token, and `ptx_counts` counts an instruction with its operands, not a longer one
    (`fma.rn.f32x2`) or a label."""
    keys = {k for c in X.PTX_COUNTS.values() for k in c}
    assert all(X.PTX_COUNT.findall(f"\t{k} %f1;") == [k] for k in keys)
    text = ("\tfma.rn.f32 %f1, %f2, %f3, %f4;\n\tfma.rn.f32x2 %r1, %r2, %r3, %r4;\n\tex2.approx.ftz.f32 %f5, %f6;\n"
            "\tmax.f32 %f7, %f8, %f9;\n\tmax.NaN.f32 %f7, %f8, %f9;\n\tcvt.rn.satfinite.e4m3x2.f32 %h1, %f1, %f2;\n"
            "$L__fma.rn.f32:\n\tshfl.sync.bfly.b32 %r5|%p1, %r6, 16, 31, -1;\n")
    assert X.ptx_counts(text) == {"cvt.rn.satfinite.e4m3x2.f32": 1, "ex2.approx.ftz.f32": 1, "fma.rn.f32": 1, "max.NaN.f32": 1,
                                  "max.f32": 1, "shfl.sync.bfly.b32": 1}


ENV_KEYS = ("VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR", "CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR", "CUTE_DSL_CACHE_DIR")


def test_the_environment_keeps_the_cute_dsl_ptx_and_cubin_in_scratch(tmp_path, monkeypatch) -> None:
    """The H200 run r20261013-205619-e245 matched every word and still went NO-GO in `ptx`: no CUTE_DSL_KEEP, so cutlass 4.6.2
    kept no PTX or cubin and every compiled kernel's `__ptx__` and `__cubin__` were None.  `environment` keeps both (KEEP's
    comma-separated tokens), with the dump and cache directories made in scratch."""
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    env = X.environment(tmp_path)
    assert {k: X.os.environ[k] for k in ENV_KEYS} == env
    assert {"ptx", "cubin"} <= {t.strip() for t in env["CUTE_DSL_KEEP"].split(",")}
    for k in ("CUTE_DSL_DUMP_DIR", "CUTE_DSL_CACHE_DIR"):
        p = X.Path(env[k])
        assert p.is_dir() and p.parent == tmp_path, k


def _kernels(texts: dict) -> types.ModuleType:
    """vLLM's CuTe DSL kernels module stood in for, after the launches: each `compile` a cache holding the one served kernel, a
    compiled stub with ``texts[name]`` as its `__ptx__` and cubin bytes when it has PTX, as cutlass 4.6.2 does with
    KEEP=ptx,cubin (None for both without)."""
    import torch

    class Compile:
        def __init__(self, name):
            t = texts[name]
            self.made = SimpleNamespace(__ptx__=t, __cubin__=None if t is None else b"\x7fELF")

        def __call__(self, *a, **kw):
            return self.made

        def cache_info(self):
            return SimpleNamespace(currsize=1)

    K = types.ModuleType("sparse_attn_compress_cutedsl")
    K.SparseAttnCompressNormRopeStoreC4Kernel, K.SparseAttnCompressC128Block8Kernel, K.SparseAttnNormRopeStoreKernel = (
        type(n, (), {"compile": Compile(n)}) for n in ("c4", "c128", "store"))
    K._TORCH_TO_CUTE = {torch.bfloat16: "bf16"}
    K.compile_split_sparse_attn_cutedsl = lambda *a, **kw: (K.SparseAttnCompressC128Block8Kernel.compile(),
                                                            K.SparseAttnNormRopeStoreKernel.compile())
    return K


def _ptx_text(counts: dict) -> str:
    return ".version 8.8\n.target sm_90a\n" + "".join(f"\t{k} %f1;\n" * n for k, n in counts.items())


@pytest.mark.parametrize("kept", [True, False])
def test_the_ptx_phase_reads_each_kernels_kept_ptx_and_cubin(kept, tmp_path, monkeypatch) -> None:
    """`ptx` on the served kernels as their caches hold them: with the PTX and cubin kept, each kernel's PTX is written under
    out/ptx with the pinned counts and the phase is ok; without (the e245 run), `kept` fails by name and so does the phase."""
    texts = {k: _ptx_text(X.PTX_COUNTS[k]) if kept else None for k in ("c4", "c128", "store")}
    K = _kernels(texts)
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v4", "vllm.models.deepseek_v4.nvidia"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    ops = types.ModuleType("vllm.models.deepseek_v4.nvidia.ops")
    ops.sparse_attn_compress_cutedsl = K
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, f"{ops.__name__}.sparse_attn_compress_cutedsl", K)
    monkeypatch.setattr(X.shutil, "which", lambda tool: None)
    monkeypatch.setenv("CUTE_DSL_KEEP", "ptx,cubin" if kept else "")
    rec = X.ptx(tmp_path)
    assert rec["checks"] == {"one_specialization": True, "kept": kept, "target": kept, "counts": kept} and rec["ok"] is kept
    assert sorted(p.name for p in (tmp_path / "ptx").iterdir()) == (
        ["c128.cubin", "c128.ptx", "c4.cubin", "c4.ptx", "store.cubin", "store.ptx"] if kept else [])


def test_the_sources_are_checked_by_digest(tmp_path, monkeypatch) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.py").write_text("x = 1\n")
    (tmp_path / "y.py").write_text("y = 2\n")
    monkeypatch.setattr(X, "SOURCES", {"a/x.py": hashlib.sha256(b"x = 1\n").hexdigest(), "y.py": "0" * 64, "z.py": "0" * 64})
    assert X.sources(tmp_path) == {"files": {"a/x.py": True, "y.py": False, "z.py": None}, "ok": False}


def test_the_programs_bind_at_dsv4s_statics() -> None:
    rec = X.bind_check()
    assert rec["ok"] and len(rec["bindings"]) == 11, rec


def _doc() -> dict:
    launches = [{"phase": p, "case": c, "checks": {"a": True, "b": True}} for p, c in X.CASES]
    comps = [{"phase": p, "case": c, "words": 10, "mismatch": 0, "reference": 1, "twin_mismatch": 0, "first_mismatch": None}
             for p, c in X.CASES]
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "ptx": {"ok": True},
            "bind": {"ok": True}, "launches": launches, "comparisons": comps, "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("selection", "ok"), False), (("ptx", "ok"), False),
                                        (("bind", "ok"), False), (("errors", "stack"), "boom"),
                                        (("launches", 2, "checks", "b"), False), (("launches", 7, "checks", "a"), False),
                                        (("comparisons", 1, "mismatch"), 1), (("comparisons", 4, "twin_mismatch"), 3),
                                        (("comparisons", 5, "reference"), 0), (("comparisons", 6, "words"), 0),
                                        (("totals", "mismatch"), 1)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = _doc()
    node = doc
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    """A case is its phase and its name: a layer's launch error fails that layer's cases and no other, a comparison's error its
    own case, a missing launch record its case."""
    want = [f"{p}/{c}" for p, c in X.CASES]
    assert list(X.cases(_doc())) == want and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["errors"]["c128"] = "boom"
    doc["errors"]["compare/compress/c4-edge"] = "boom"
    doc["launches"] = [x for x in doc["launches"] if (x["phase"], x["case"]) != ("score", "c4")]
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"score/c4", "compress/c4-edge", "score/c128", "compress/c128",
                                                                   "compress/c128-edge", "columns/c128", "columns/c128-edge"}


def test_main_end_to_end_on_stubbed_launches(observed, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU stood in for (the ratio-4 layer's twin-made words: its first score job and first two cache rows, and
    an edge row with one word flipped) and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case
    verdicts, the sealed record and the exit status, so a slip in the orchestration shows here and not on the H200."""
    ob = observed["c4"]
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "bind_check", lambda: {"bindings": {}, "ok": True})
    monkeypatch.setattr(X, "ptx", lambda outdir: {"ok": True})
    by = {}
    for j in ob.jobs:
        by.setdefault((j["kind"], j["case"]), j)
    edge = _flipped(tmp_path / "bad", {**by[("entries", "c4-edge")], "rows": [0, 1]}, (0, 9))

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["launches"] += [x for x in ob.recs]
        return [{**by[("score", "c4")], "rows": [0, 4]}, {**by[("entries", "c4")], "rows": [0, 2]}, edge]

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    go = {"score/c4", "compress/c4"}
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {c: c in go for c in X.cases(_doc())}
    assert doc["cases"]["compress/c4-edge"]["totals"] == {"words": X.WORDS, "mismatch": 1, "reference": 1, "twin_mismatch": 0}
    assert doc["cases"]["compress/c4"]["totals"] == {"words": 2 * X.WORDS, "mismatch": 0, "reference": 1, "twin_mismatch": 0}
    assert doc["totals"]["words"] == 4 * X.LAYERS["c4"].width + 3 * X.WORDS
    printed = capsys.readouterr().out
    assert "DSV4-COMPRESSOR NO-GO" in printed
    assert f"no-go cases={[c for c in X.cases(_doc()) if c not in go]}" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_compressor_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
