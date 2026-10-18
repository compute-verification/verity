"""A prefix-cache hit through Match: the warm request's Program reads its cached rows from its `prefix` input in one run per layer and
key / value; the fold reads them where the engine left them, one run per row, in the instances of the request that computed them.
Both canonicalise to ("PFX", layer, kv, lo, hi), in record and compact form and through GM-01's per-request Projection, so the two
compare equal with no cross-request read.  A cached row read from the wrong position, layer or nowhere still fails, and G5 holds the
engine's served rows and their producers to the Program's declared hit."""

from __future__ import annotations

import json

import pytest

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import program_compare as C
from verity_vllm.program import compact as K
from verity_vllm.program import prefix_layout as PL

L, C_ROWS, RL, LP = 2, 2, 4, 3                # layers, cached rows, KVH*D, the warm request's prompt rows
ATT = f"Attention_v3{{T={LP},NH=1,KVH=1,D={RL}}}"
WNAME = "model.layers.0.self_attn.qkv_proj.weight"
LAYERS = [f"model.layers.{l}.self_attn.attn" for l in range(L)]


def _finish(P: C.Prog, inst: list[dict], compact: bool) -> C.Prog:
    P.compact = compact
    P.inst = [dict(r, args=[K.from_runs(a) if compact else list(a) for a in r["args"]]) for r in inst]
    P.k2i = {r["k"]: i for i, r in enumerate(P.inst)}
    P._bind_splits()
    P._segment()
    return P


def _blank(label: str, inputs: dict) -> C.Prog:
    P = C.Prog.__new__(C.Prog)
    P.label, P.digest, P.inputs = label, f"digest-{label}", inputs
    P.seed_nodes, P.seed_index = set(), {}
    P.param_names, P.oracle_rows, P.n_nodes, P.nodes = [], None, 0, None
    P.supplied_fields, P.supplied_owner, P.peer_fields, P.peer_index = {}, {}, {}, {}
    return P


def _rows(k: int, tok: int, pos: int) -> list[dict]:
    """A position's q / k / v rows of one layer (three Gemms over the prompt token and the weights)."""
    return [{"k": k + j, "spec": f"Gemm_v1{{M=1,N={RL},K=1,J={j}}}", "form": "call", "args": [[("n", tok, pos, pos + 1)], [("n", 2, 0, RL)]],
             "out": f"bf16[{RL}]"} for j in range(3)]


def _attention(k: int, q: int, kv_runs: tuple[list, list]) -> dict:
    return {"k": k, "spec": ATT, "form": "call", "args": [[("n", q, 0, RL)], kv_runs[0], kv_runs[1]], "out": f"bf16[{RL}]"}


def _select(k: int, src: int) -> dict:
    return {"k": k, "spec": "TokenSelect_v1{V=4}", "form": "call", "args": [[("n", src, 0, RL)]], "out": "i32"}


def warm_program(tmp_path, compact: bool = False, layers: list[str] = LAYERS) -> C.Prog:
    """The warm request's Program (`build --cached C`): prompt, prefix [L, 2, C, KVH*D], weights; each layer computes row C and attends
    over its C cached rows (one run of the prefix input) and its own row."""
    P = _blank("derived", {0: ("Input32_v1", LP), 1: ("Input16_v1", 2 * L * C_ROWS * RL), 2: ("Input16_v1", RL)})
    P.tok_node, P.weight_fields = 0, {2: WNAME}
    inst, k = [], 3
    for l in range(L):
        inst += _rows(k, 0, C_ROWS)
        plane = lambda kv: (2 * l + kv) * C_ROWS * RL   # noqa: E731
        inst.append(_attention(k + 3, k, ([("n", 1, plane(0), plane(0) + C_ROWS * RL), ("n", k + 1, 0, RL)],
                                          [("n", 1, plane(1), plane(1) + C_ROWS * RL), ("n", k + 2, 0, RL)])))
        k += 4
    inst.append(_select(k, k - 1))
    _finish(P, inst, compact)
    d = tmp_path / "derived"
    d.mkdir(exist_ok=True)
    (d / PX.PREFIX_FILE).write_text(json.dumps({"input": 1, "C": C_ROWS, "row_leaves": RL, "layers": layers}))
    PX.load_layout(P, str(d))
    return P


def _meta(k: int, request: str, position: int, hint: str) -> dict:
    """A fold row with only the fields a link reads (`PX.FOLD_META`)."""
    row = {"node": k - 3, "request": request, "position": position, "hint": hint}
    assert row.keys() - {"node"} == PX.FOLD_META
    return row


def fold(compact: bool = False, read_row=lambda l, j: (l, j)) -> tuple[C.Prog, list[int]]:
    """The batch: r0 computes rows 0..C-1 of every layer, r1 (warm) computes row C and reads r0's rows as its first C rows.
    `read_row(l, j)` is the (layer, position) of r0's row r1 reads as its row j of layer l.  Returns the fold and r1's instances."""
    F = _blank("fold", {0: ("Input32_v1", LP), 1: ("Input32_v1", LP), 2: ("Input16_v1", RL)})
    F.tok_node, F.weight_fields = 0, {}
    F.oracle_wmap, F._wlos, F.oracle_wmap_overlaps, F.input_base = [(0, RL, WNAME)], [0], [], {}
    F.n_input_nodes = 3
    inst, meta, k = [], [], 3
    r0 = {}
    for l in range(L):
        for j in range(C_ROWS):
            rows = _rows(k, 0, j)
            inst += rows
            meta += [_meta(r["k"], "r0", j, f"model.layers.{l}.self_attn.qkv_proj/{WNAME}") for r in rows]
            r0[(l, j)] = k
            k += 3
    mine = []
    for l in range(L):
        rows = _rows(k, 1, C_ROWS)
        cached = [r0[read_row(l, j)] for j in range(C_ROWS)]
        att = _attention(k + 3, k, ([("n", p + 1, 0, RL) for p in cached] + [("n", k + 1, 0, RL)],
                                    [("n", p + 2, 0, RL) for p in cached] + [("n", k + 2, 0, RL)]))
        for r in rows + [att]:
            mine.append(len(inst))
            inst.append(r)
            meta.append(_meta(r["k"], "r1", C_ROWS, LAYERS[l] + "/attn" if r is att else f"model.layers.{l}.self_attn.qkv_proj/{WNAME}"))
        k += 4
    mine.append(len(inst))
    inst.append(_select(k, k - 1))
    meta.append(_meta(k, "r1", C_ROWS, "sampler"))
    F.oracle_rows = meta
    return _finish(F, inst, compact), mine


def warm_projection(compact: bool = False, cached: int = C_ROWS, **kw) -> BD.Projection:
    F, mine = fold(compact, **kw)
    P = BD.Projection(F, "oracle[r1]", mine, tok_node=1, weights_node=2)
    P.cached_rows = cached
    return P


def _att(P: C.Prog, layer: int) -> int:
    return [i for i, r in enumerate(P.inst) if r["spec"] == ATT][layer]


@pytest.mark.parametrize("compact", [False, True])
def test_the_cached_rows_compare_as_the_prefix_input_on_both_sides(tmp_path, compact):
    A, P = warm_program(tmp_path, compact), warm_projection(compact)
    for l in range(L):
        for kv in (0, 1):
            want = [("PFX", l, kv, 0, C_ROWS * RL), ("N", 0, 4 * l + 1 + kv, 0, RL)]
            for Q in (A, P):
                got = Q.canon(_att(Q, l))[1 + kv]
                assert (K.expand_mixed(got) if compact else got) == want, (Q.label, l, kv, got)
    res = C.compare_steps(A, P)
    assert res["totals"].get("operands_different", 0) == 0 and res["canonical_sha256"]["equal"], res["steps"][0].get("wiring")
    assert not P.xreq


@pytest.mark.parametrize("compact", [False, True])
def test_g5_reads_the_hit_the_engine_served_against_the_declared_one(tmp_path, compact):
    P = warm_projection(compact)
    assert P.prefix_producers() == {0: {"r0"}, 1: {"r0"}}
    why, hit = PX.check_hits({"request_id": "r1", "prefix": {"rows": C_ROWS, "producers": ["r0"]}}, P)
    assert why == [] and hit["rows"] == C_ROWS and hit["read_from"] == [["r0"]] and hit["block_size"] == C_ROWS
    why, _ = PX.check_hits({"request_id": "r1", "prefix": {"rows": C_ROWS, "producers": ["r2"]}}, P)
    assert why and "cached block 0 was read from ['r0'], the Program declares 'r2'" in why[0]
    why, _ = PX.check_hits({"request_id": "r1"}, P)
    assert why and "served 2 prompt rows from the prefix cache" in why[0]
    assert PX.check_hits({"request_id": "r0"}, warm_projection(compact, cached=0)) == ([], None)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("name, kw", [
    ("rows_swapped", {"read_row": lambda l, j: (l, C_ROWS - 1 - j)}),   # row j read from the producer's other position
    ("other_layer", {"read_row": lambda l, j: (1 - l, j)}),             # layer l's rows read from the other layer
    ("not_served", {"cached": 0}),                                     # the engine scheduled the request from row 0
    ("served_fewer", {"cached": 1}),                                   # one row fewer than the reads
])
def test_a_misplaced_or_unserved_cached_row_is_a_cross_request_read(tmp_path, compact, name, kw):
    A, P = warm_program(tmp_path, compact), warm_projection(compact, **kw)
    res = C.compare_steps(A, P)
    assert P.xreq and not res["canonical_sha256"]["equal"], name


def test_the_program_side_needs_its_layout_and_the_layer_order_it_names(tmp_path):
    A = warm_program(tmp_path, layers=list(reversed(LAYERS)))
    assert A.canon(_att(A, 0))[1][0] == ("PFX", 1, 0, 0, C_ROWS * RL)   # the input's first layer is the one the derive opened first
    d = tmp_path / "bad"
    d.mkdir()
    (d / PX.PREFIX_FILE).write_text(json.dumps({"input": 1, "C": C_ROWS + 1, "row_leaves": RL, "layers": LAYERS}))
    with pytest.raises(ValueError, match="is not one Input node"):
        PX.load_layout(A, str(d))


def test_the_fold_rows_global_match_loads_keep_what_a_link_reads(tmp_path):
    from tests.check import test_global_match as T
    _gp, oracle = T.assemble([T.R0, T.R1_LATE], tmp_path)
    rows = C.load_oracle(oracle, "oracle").oracle_rows
    assert all(PX.FOLD_META <= r.keys() for r in rows) and any(r["position"] for r in rows)


def test_g5_holds_the_served_block_size_to_the_plans():
    """The plan hashed the prompt in blocks of C_ROWS; an engine whose backend cached blocks of another size (a sparse MLA backend serves
    64 unless the row declares it) fails G5 even where its rows happen to agree, and names the declaration that fixes the plan."""
    steps = {0: {"rows": [("r0", 0)], "block_size": 16}, 3: {"rows": [("r1", C_ROWS)], "block_size": C_ROWS}}
    assert PX.served_block_size(steps, "r1") == C_ROWS and PX.served_block_size({0: {"rows": [("r1", 0)]}}, "r1") is None
    decl = {"request_id": "r1", "prefix": {"rows": C_ROWS, "producers": ["r0"]}}
    P = warm_projection()
    P.block_size = C_ROWS
    why, hit = PX.check_hits(decl, P)
    assert why == [] and hit["served_block_size"] == C_ROWS
    P.block_size = 64
    why, hit = PX.check_hits(decl, P)
    assert why == [f"r1: the engine hashed its prompt in blocks of 64 tokens (StepStart.block_size), the Program's plan in blocks of {C_ROWS}: "
                   "the row declares its attention backend's block size, engine.args.block_size: 64"]


def test_the_served_rows_are_the_first_scheduled_position_under_prefix_caching_only():
    steps = {0: {"rows": [("r0", 0), ("r0", 1), ("r0", 2)]}, 4: {"rows": [("r0", 7), ("r1", 2)]}, 5: {"rows": [("r0", 8), ("r1", 3)]}}
    on = {"profile": {"engine_kwargs": {"enable_prefix_caching": True}}}
    assert [PX.cached_rows(on, steps, r) for r in ("r0", "r1")] == [0, 2]
    for off in ({"profile": {"engine_kwargs": {"enable_prefix_caching": False}}}, {}):
        assert PX.cached_rows(off, steps, "r1") == 0


DV, DR, D, WQ = 4, 2, 3, 2      # MLA: the latent's normed and rope leaves per row, the index-K width, the query's leaves
W = DV + DR
SMLA, LOGITS = f"SparseMlaSplit_v1{{T={LP},NS=1,NH=1,DR={DR},DV={DV}}}", f"DsaMqaLogits_v3{{N={LP},H=1,D={D}}}"
MLA_PARTS = (("q", WQ), ("kv_a_layernorm", DV), ("k_pe", DR), ("indexer_k", D + 1))   # a position's rows: the index-K row holds D slots then its scale


def _gemms(k: int, tok: int, pos: int, parts) -> list[dict]:
    return [{"k": k + j, "spec": f"Gemm_v1{{M=1,N={n},K=1,J={j}}}", "form": "call", "args": [[("n", tok, pos, pos + 1)], [("n", 2, 0, RL)]],
             "out": f"bf16[{n}]"} for j, (_name, n) in enumerate(parts)]


def _mla_calls(k: int, q: int, lat: list, ik: list, iks: list) -> list[dict]:
    """One query row's indexer logits (q, weights, index-K slots, scales) and sparse MLA (q_nope, q_pe, latent rows)."""
    return [{"k": k, "spec": LOGITS, "form": "call", "args": [[("n", q, 0, WQ)], [("n", q, 0, 1)], ik, iks], "out": f"f32[{LP}]"},
            {"k": k + 1, "spec": SMLA, "form": "call", "args": [[("n", q, 0, WQ)], [("n", q, 0, 1)], lat], "out": f"bf16[{RL}]"}]


def warm_mla_program(tmp_path, compact: bool = False) -> C.Prog:
    """The warm request's MLA Program: prompt, latent [L, C, DV+DR], weights, index-K [L, C, D], scales [L, C]; each layer computes row C
    and its logits and sparse MLA read the C cached rows of each kind (one run of each input) then its own row."""
    P = _blank("derived", {0: ("Input32_v1", LP), 1: ("Input16_v1", L * C_ROWS * W), 2: ("Input16_v1", RL), 3: ("Input8_v1", L * C_ROWS * D),
                           4: ("Input32_v1", L * C_ROWS)})
    P.tok_node, P.weight_fields = 0, {2: WNAME}
    inst, k = [], 5
    for l in range(L):
        inst += _gemms(k, 0, C_ROWS, MLA_PARTS)
        q, nrm, kpe, ikr = k, k + 1, k + 2, k + 3
        inst += _mla_calls(k + 4, q, [("n", 1, l * C_ROWS * W, (l + 1) * C_ROWS * W), ("n", nrm, 0, DV), ("n", kpe, 0, DR)],
                           [("n", 3, l * C_ROWS * D, (l + 1) * C_ROWS * D), ("n", ikr, 0, D)],
                           [("n", 4, l * C_ROWS, (l + 1) * C_ROWS), ("n", ikr, D, D + 1)])
        k += 6
    inst.append(_select(k, k - 1))
    _finish(P, inst, compact)
    d = tmp_path / "derived_mla"
    d.mkdir(exist_ok=True)
    (d / PX.PREFIX_FILE).write_text(json.dumps({"kind": "mla", "input": 1, "inputs": [1, 3, 4], "C": C_ROWS, "row_leaves": W, "index_d": D,
                                                "layers": LAYERS, "indexer_layers": LAYERS}))
    PX.load_layout(P, str(d))
    return P


def mla_projection(compact: bool = False, read_row=lambda l, j: (l, j), rope_row=None) -> BD.Projection:
    """r0 computes every layer's rows 0..C-1 (normed latent, rope key, index-K row); r1 (warm) computes row C and reads r0's as its first
    C rows, a latent row as two runs (the normed part, then the rope part) as the engine's fold records it.  `read_row` / `rope_row`:
    the (layer, position) of r0's row r1 reads for its row j of layer l (the rope part's apart)."""
    F = _blank("fold", {0: ("Input32_v1", LP), 1: ("Input32_v1", LP), 2: ("Input16_v1", RL)})
    F.tok_node, F.weight_fields = 0, {}
    F.oracle_wmap, F._wlos, F.oracle_wmap_overlaps, F.input_base = [(0, RL, WNAME)], [0], [], {}
    F.n_input_nodes = 3
    inst, meta, k, r0 = [], [], 3, {}
    for l in range(L):
        for j in range(C_ROWS):
            rows = _gemms(k, 0, j, MLA_PARTS[1:])
            inst += rows
            meta += [_meta(r["k"], "r0", j, f"model.layers.{l}.self_attn/{name}") for r, (name, _n) in zip(rows, MLA_PARTS[1:])]
            r0[(l, j)] = k
            k += 3
    mine = []
    for l in range(L):
        rows = _gemms(k, 1, C_ROWS, MLA_PARTS)
        q, nrm, kpe, ikr = k, k + 1, k + 2, k + 3
        at = [(r0[read_row(l, j)], r0[(rope_row or read_row)(l, j)]) for j in range(C_ROWS)]
        lat = [run for p, pr in at for run in (("n", p, 0, DV), ("n", pr + 1, 0, DR))] + [("n", nrm, 0, DV), ("n", kpe, 0, DR)]
        calls = _mla_calls(k + 4, q, lat, [("n", p + 2, 0, D) for p, _ in at] + [("n", ikr, 0, D)],
                           [("n", p + 2, D, D + 1) for p, _ in at] + [("n", ikr, D, D + 1)])
        hints = [f"model.layers.{l}.self_attn/{name}" for name, _n in MLA_PARTS] + [f"model.layers.{l}.self_attn/mqa[t={C_ROWS}]",
                                                                                      f"model.layers.{l}.self_attn/sparse_mla[t={C_ROWS},splits=1]"]
        for r, h in zip(rows + calls, hints):
            mine.append(len(inst))
            inst.append(r)
            meta.append(_meta(r["k"], "r1", C_ROWS, h))
        k += 6
    mine.append(len(inst))
    inst.append(_select(k, k - 1))
    meta.append(_meta(k, "r1", C_ROWS, "sampler"))
    F.oracle_rows = meta
    P = BD.Projection(_finish(F, inst, compact), "oracle[r1]", mine, tok_node=1, weights_node=2)
    P.cached_rows = C_ROWS
    return P


@pytest.mark.parametrize("compact", [False, True])
def test_an_mla_warm_requests_latent_and_index_k_rows_compare_as_its_prefix_inputs(tmp_path, compact):
    A, P = warm_mla_program(tmp_path, compact), mla_projection(compact)
    for spec, ops in ((SMLA, {2: (PL.LATENT, W)}), (LOGITS, {2: (PL.INDEX_K, D), 3: (PL.INDEX_K_SCALE, 1)})):
        for l in range(L):
            for ai, (kind, rl) in ops.items():
                got = [Q.canon([i for i, r in enumerate(Q.inst) if r["spec"] == spec][l])[ai] for Q in (A, P)]
                got = [K.expand_mixed(g) if compact else g for g in got]
                assert got[0] == got[1] and got[0][0] == ("PFX", l, kind, 0, C_ROWS * rl), (spec, l, ai, got)
    res = C.compare_steps(A, P)
    assert res["totals"].get("operands_different", 0) == 0 and res["canonical_sha256"]["equal"], res["steps"][0].get("wiring")
    assert not P.xreq and P.prefix_producers() == {j: {"r0"} for j in range(C_ROWS)}
    why, hit = PX.check_hits({"request_id": "r1", "prefix": {"rows": C_ROWS, "producers": ["r0"]}}, P)
    assert why == [] and hit["read_from"] == [["r0"]]


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("name, kw", [
    ("rows_swapped", {"read_row": lambda l, j: (l, C_ROWS - 1 - j)}),
    ("other_layer", {"read_row": lambda l, j: (1 - l, j)}),
    ("rope_part_elsewhere", {"rope_row": lambda l, j: (l, C_ROWS - 1 - j)}),   # a latent row's normed part and rope part from two positions
])
def test_a_misplaced_mla_row_is_a_cross_request_read(tmp_path, compact, name, kw):
    A, P = warm_mla_program(tmp_path, compact), mla_projection(compact, **kw)
    res = C.compare_steps(A, P)
    assert P.xreq and not res["canonical_sha256"]["equal"], name


def test_contiguous_prefix_reads_merge_in_both_forms():
    runs = [("PFX", 0, 1, 0, 4), ("PFX", 0, 1, 4, 8), ("N", 0, 1, 0, 4), ("PFX", 0, 1, 8, 12)]
    assert PX.merge_reads(runs, False) == [("PFX", 0, 1, 0, 8), ("N", 0, 1, 0, 4), ("PFX", 0, 1, 8, 12)]
    progs = [("PFX", 0, 1, (0, 4), (4, 4), 2), ("PFX", 1, 1, (8, 12), (0, 0), 1)]
    assert PX.merge_reads(progs, True) == [("PFX", 0, 1, (0, 8), (0, 0), 1), ("PFX", 1, 1, (8, 12), (0, 0), 1)]
