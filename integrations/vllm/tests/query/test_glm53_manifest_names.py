"""GLM-5.3's TP8 manifest spells every required identity as the runtime commits it (the Commit of record r20261016-124321-8e1d bound
none of 26,686 of rank 0's): on a rank Program shaped like GLM's -- dense layer 0, MoE layers 3 and 4 with the shared expert under the
runner's held path `_shared_experts._layer`, a prefill step on the unfused all-reduce + `RMSNormFusedCudaVector8` and DeepGEMM path
and a decode step on the fused `AllReduceRmsNormMnnvl` and FlashInfer path -- every `instance_outputs` / `tp_rank_partials` identity is
a name the capture map `glm-moe-dsa-eager-hopper-tp8-v0` commits, except the three this change names for the commit side to bind:

* `<norm>/all_reduce`, the unfused path's all-reduced tensor (`allreduce_norm` binds the partial and lets the norm's hook bind
  `<norm>/0` / `<norm>/1`, but nothing binds the all-reduce's own output);
* `<linear>/x_q` and `<linear>/x_s`, `DeepGemmFp8Linear_v1`'s quant of its input (ports `0` / `1`), so the hook's `<linear>/0` is y."""
from __future__ import annotations

import pytest

from tests.acquire.glm53_model import CMAP, ENV
from verity_vllm.commit import capture_map as CM
from verity_vllm.correspondence.reader_for_query import Correspondence, held_module_paths, parse_call_name
from verity_vllm.query.compose import merge_ranks
from verity_vllm.query.manifest.format import committed_identity
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import Population, request_manifest
from verity_vllm.query.units import ValueRef as R

WORLD, N, K = 8, 4, 4
DENSE, MOE = (0,), (3, 4)
HELD = "_shared_experts._layer"
DEEPGEMM, FLASHINFER = "DeepGemmFp8Linear_v1", "FlashInferFp8Linear_v1"
GATE = {0: "CublasSplitKGemmF32Out_v1", 1: "LlBf16DotprodGemm_v1"}
FUSED = f"AllReduceRmsNormMnnvl_v1{{WORLD=8,N={N},SHOT=1,CLUSTER=8,EPS=1e-05,FMA=True}}"


class _Build:
    """A rank Program built Call by Call under v1 names (`<module>/<fx node>[/<hint>][row]`)."""

    def __init__(self, rank: int):
        self.rank, self.calls, self.weights, self.peers, self.coll = rank, [], [], [], 0

    def w(self, path: str) -> R:
        if path not in self.weights:
            self.weights.append(path)
        return R(-1, path)

    def call(self, family: str, name: str, reads, ports=(("out", N, 16),)) -> int:
        cid = len(self.calls)
        table = tuple(Port(p, 0, n, n, width) for p, n, width in ports)
        self.calls.append(CallView(cid, f"{family}{{K={K},N={N}}}" if "{" not in family else family, family.split("{")[0], "call", name,
                                   table, tuple(reads)))
        return cid

    def parts(self, own: list[R], elems: int) -> list[R]:
        k, self.coll = self.coll, self.coll + 1
        received = []
        for j in range(WORLD - 1):
            iid = -3 - len(self.peers)
            self.peers.append(InputView(iid, f"peers_{len(self.peers)}", f"Array<1,Array<{elems},Value<16>>>",
                                        (Port(f"peers_{len(self.peers)}", 0, elems, elems, 16),)))
            received.append(R(iid, self.peers[-1].name))
        return received[: self.rank] + own + received[self.rank:]

    def linear(self, module: str, x: list[R], step: int, *, weight: str | None = None) -> list[R]:
        """One linear row per input row: DeepGEMM at prefill (q, s, y), FlashInfer at decode (y)."""
        ws = [self.w(f"{weight or module}.weight"), self.w(f"{weight or module}.weight_scale_inv")]
        if step == 0:
            ids = [self.call(DEEPGEMM, f"{module}/dynamic_flashinfer_deepgemm_blockscale_gemm[{i}]", [r, *ws],
                             (("0", K, 8), ("1", 1, 32), ("2", N, 16))) for i, r in enumerate(x)]
            return [R(c, "2") for c in ids]
        return [R(self.call(FLASHINFER, f"{module}/dynamic_flashinfer_deepgemm_blockscale_gemm[{i}]", [r, *ws]), "out")
                for i, r in enumerate(x)]

    def norm_site(self, site: str, norm: str, own: list[R], residual: list[R], step: int) -> tuple[list[R], list[R]]:
        """`fused_allreduce_rms_norm(own, residual, norm)` called from `site`: flat at prefill (one all-reduce, then the norm's rows),
        fused at decode (one Call reading the residual and the norm weight)."""
        M, nw = len(own), self.w(f"{norm}.weight")
        if step == 0:
            ar = self.call("AllReduceMultimem_v1", f"{site}.all_reduce_{self.coll}/all_reduce[0]/all_reduce", self.parts(own, M * N),
                           (("out", M * N, 16),))
            rows = [self.call("RMSNormFusedCudaVector8_v1", f"{norm}/fused_add_rms_norm[{i}]", [R(ar, "out"), residual[i], nw],
                              (("0", N, 16), ("1", N, 16))) for i in range(M)]
            return [R(c, "0") for c in rows], [R(c, "1") for c in rows]
        assert M == 1
        f = self.call(FUSED, f"{site}.fused_allreduce_rms_norm_{self.coll}/fused_allreduce_rms_norm[0]/fused_all_reduce_rmsnorm",
                      [*self.parts(own, N), residual[0], nw], (("0", N, 16), ("1", N, 16)))
        return [R(f, "0")], [R(f, "1")]

    def step(self, step: int, M: int) -> None:
        emb = [self.call("EmbeddingShard_v1", f"model.embed_tokens/embedding_shard[{i}]/shard", [R(-2, "ids"), self.w("model.embed_tokens.weight")])
               for i in range(M)]
        ar = self.call("AllReduceMultimem_v1", "model.embed_tokens/all_reduce[0]/all_reduce", self.parts([R(c, "out") for c in emb], M * N),
                       (("out", M * N, 16),))
        residual = [R(ar, "out")] * M
        h = [R(self.call("RMSNormCudaVector8_v1", f"model.layers.0.input_layernorm/rms_norm[{i}]",
                         [R(ar, "out"), self.w("model.layers.0.input_layernorm.weight")]), "out") for i in range(M)]
        mlp_out = None
        for layer in DENSE + MOE:
            pre = f"model.layers.{layer}"
            if mlp_out is not None:
                h, residual = self.norm_site(pre, f"{pre}.input_layernorm", mlp_out, residual, step)
            attn = self.linear(f"{pre}.self_attn.o_proj", h, step)
            h, residual = self.norm_site(pre, f"{pre}.post_attention_layernorm", attn, residual, step)
            if layer in DENSE:
                up = self.linear(f"{pre}.mlp.gate_up_proj", h, step)
                act = [R(self.call("SiluMul_v2", f"{pre}.mlp.act_fn/silu_and_mul[{i}]", [r]), "out") for i, r in enumerate(up)]
                mlp_out = self.linear(f"{pre}.mlp.down_proj", act, step)
                continue
            gw = self.w(f"{pre}.mlp.gate.weight")
            logits = [R(self.call(GATE[min(step, 1)], f"{pre}.mlp.gate/gate[{i}]", [r, gw], (("out", N, 32),)), "out") for i, r in enumerate(h)]
            shared = f"{pre}.mlp.shared_experts"
            up = self.linear(f"{HELD}.gate_up_proj", h, step, weight=f"{shared}.gate_up_proj")
            act = [R(self.call("SiluMul_v2", f"{HELD}.act_fn/silu_and_mul[{i}]", [r]), "out") for i, r in enumerate(up)]
            down = self.linear(f"{HELD}.down_proj", act, step, weight=f"{shared}.down_proj")
            # the routed experts' stand-in: the runner reads the router logits and adds the shared expert's output
            mlp_out = [R(self.call("ResidualAdd_v1", f"{pre}.mlp.experts/add[{i}]", [r, g]), "out") for i, (r, g) in enumerate(zip(down, logits))]
        h, _ = self.norm_site("model", "model.norm", mlp_out, residual, step)
        shard = self.call("Gemm_v2", "logits_processor/linear[0]", [h[-1], self.w("lm_head.weight")])
        lm = self.call(f"AllGather_v1{{WORLD={WORLD},N={N}}}", "logits_processor/all_gather[0]/all_gather", self.parts([R(shard, "out")], N),
                       (("out", WORLD * N, 16),))
        self.call("ArgmaxTokenSelect_v1", "argmax[0]", [R(lm, "out")], (("out", 1, 32),))

    def program(self) -> ProgramView:
        weights = InputView(-1, "weights", "{...}", tuple(Port(n, N * i, N * (i + 1), N, 16) for i, n in enumerate(self.weights)))
        ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
        return ProgramView(tuple(self.calls), (weights, ids, *self.peers), (R(len(self.calls) - 1, "out"),), "synthetic", "synthetic")


def rank_program(rank: int, M: int = 2) -> ProgramView:
    b = _Build(rank)
    b.step(0, M)
    b.step(1, 1)
    return b.program()


def _manifest(rank: int) -> dict:
    P = rank_program(rank)
    res = {"program": {"digest": f"d{rank}", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}},
           "tp": {"world": WORLD, "rank": rank}}
    return request_manifest(P, Correspondence(P), res, request_id="r0", rank=rank, ranks=WORLD).manifest


NORMS = {f"model.layers.{k}.{n}" for k in DENSE + MOE for n in ("input_layernorm", "post_attention_layernorm")} - {"model.layers.0.input_layernorm"} | {"model.norm"}
QUANT_LINEARS = {f"model.layers.{k}.self_attn.o_proj" for k in DENSE + MOE} | {f"model.layers.0.mlp.{p}" for p in ("gate_up_proj", "down_proj")} | \
    {f"model.layers.{k}.mlp.shared_experts.{p}" for k in MOE for p in ("gate_up_proj", "down_proj")}
#: the identities this change names for the commit side to bind (module docstring)
NEEDS_SOURCE = {(n, "all_reduce") for n in NORMS} | {(m, q) for m in QUANT_LINEARS for q in ("x_q", "x_s")}


@pytest.fixture(scope="module")
def manifests() -> list[dict]:
    return [_manifest(rank) for rank in range(WORLD)]


def _committed(rank: int) -> set[tuple[str, str]]:
    out = set()
    for routes in (("sparse-mqa",), ("dense-mha",)):
        out |= {committed_identity(n) for n in CM.committer_names(CMAP, ENV, routes=routes, rank=rank)}
    return out


def _named(m: dict) -> set[tuple[int, str, str]]:
    return {(r["step"], r["op_path"], r["output_member"]) for r in m["identities"] if r["family"] in ("instance_outputs", "tp_rank_partials")}


@pytest.mark.parametrize("rank", [0, 5])
def test_every_required_identity_is_a_name_the_capture_map_commits(manifests, rank):
    m = manifests[rank]
    assert m["complete"], m["unmodelled"]
    named = {(op, mem) for _s, op, mem in _named(m)}
    missing = named - _committed(rank) - NEEDS_SOURCE
    assert not missing, sorted(missing)
    assert NEEDS_SOURCE <= named, sorted(NEEDS_SOURCE - named)
    assert not [op for op, _m in named if op in ("model", *(f"model.layers.{k}" for k in DENSE + MOE)) or op.startswith(HELD)]
    assert m["op_path_aliases"] == {}, "no collective site is left at its enclosing module"


def test_the_ranks_merge_with_every_peer_bound(manifests):
    g = merge_ranks(manifests)
    b = g["tp_peer_binding"]
    assert g["complete"] and b["n_unbound"] == 0 and b["bound"] == b["collective_outputs"] * (WORLD - 1), b


def test_each_norms_collective_is_spelt_under_the_norm_on_both_paths(manifests):
    """Prefill (unfused): `<norm>/all_reduce` and `<norm>/part_rank<r>` from the flat all-reduce, `<norm>/0` / `/1` from the norm's rows;
    decode (fused): `<norm>/0`, `/1` and `/part_rank<r>` from the one fused Call -- the names `allreduce_norm` binds on either path."""
    rank = 3
    m = manifests[rank]
    by = {}
    for r in m["identities"]:
        if r["op_path"] in NORMS and r["family"] != "weights":
            by.setdefault((r["step"], r["op_path"]), {})[r["output_member"]] = (r["spec"], r["invocation"], r["element_range"])
    for norm in NORMS:
        pre = by[(0, norm)]
        assert {k: v[0] for k, v in pre.items()} == {"0": "RMSNormFusedCudaVector8_v1", "1": "RMSNormFusedCudaVector8_v1",
                                                     "all_reduce": "AllReduceMultimem_v1", f"part_rank{rank}": "AllReduceMultimem_v1"}, norm
        assert pre["all_reduce"][2] == pre[f"part_rank{rank}"][2] == [0, 2 * N], "the flat all-reduce and its partial are the whole [M, N]"
        dec = by[(1, norm)]
        assert {k: v[0] for k, v in dec.items()} == dict.fromkeys(("0", "1", f"part_rank{rank}"), "AllReduceRmsNormMnnvl_v1"), norm
        assert {v[1] for v in (*pre.values(), *dec.values())} == {0}


def test_a_linear_reading_its_own_weight_returns_slot_0_on_every_route(manifests):
    rows = {(r["step"], r["op_path"], r["output_member"]): r for r in manifests[0]["identities"] if r["family"] == "instance_outputs"}
    for lin in QUANT_LINEARS:
        y = rows[(0, lin, "0")]
        assert (y["spec"], y["width"], y["dtype"], y["boundary"]["structural_member"]) == (DEEPGEMM, N, "bf16", "2"), "slot 0 is y, never q"
        assert (rows[(0, lin, "x_q")]["dtype"], rows[(0, lin, "x_s")]["dtype"]) == ("w8", "f32")
        assert rows[(1, lin, "0")]["spec"] == FLASHINFER and (1, lin, "out") not in rows
    for k in MOE:
        assert {s: rows[(s, f"model.layers.{k}.mlp.gate", "0")]["spec"] for s in (0, 1)} == GATE
    applied = manifests[0]["query"]["residuals_applied"]
    for key in ("FlashInferFp8Linear_v1:out->0", "DeepGemmFp8Linear_v1:2->0", "DeepGemmFp8Linear_v1:0->x_q", "DeepGemmFp8Linear_v1:1->x_s",
                "LlBf16DotprodGemm_v1:out->0", "CublasSplitKGemmF32Out_v1:out->0", "encoding:norm site"):
        assert applied.get(key), key


def test_the_shared_expert_is_each_layers_own(manifests):
    """The held `_shared_experts._layer.<leaf>` rows of the two MoE layers were one identity of 2 layers' rows; each is now its layer's."""
    rows = {(r["step"], r["op_path"], r["output_member"]): r["rows"] for r in manifests[0]["identities"] if r["family"] == "instance_outputs"}
    for k in MOE:
        sh = f"model.layers.{k}.mlp.shared_experts"
        assert [rows[(0, f"{sh}.{leaf}", mem)] for leaf, mem in (("gate_up_proj", "0"), ("act_fn", "out"), ("down_proj", "0"))] == [2, 2, 2]
        assert [rows[(1, f"{sh}.{leaf}", mem)] for leaf, mem in (("gate_up_proj", "0"), ("act_fn", "out"), ("down_proj", "0"))] == [1, 1, 1]
    corr = manifests[0]["correspondence"]
    assert corr["held_modules"] == {"resolved": 3 * 2 * len(MOE) + 3 * len(MOE), "unresolved": {}}


def test_a_held_path_resolves_only_from_its_own_weights_or_producers():
    """A held SiLU that reads no resolved Call of its holder keeps its path and is counted; `logits_processor`, whose root no weight has,
    reading `lm_head.weight` (another leaf) is not a held path and is left alone, uncounted."""
    w = InputView(-1, "weights", "{...}", (Port("model.layers.3.mlp.shared_experts.gate_up_proj.weight", 0, N, N, 16), Port("lm_head.weight", N, 2 * N, N, 16)))
    x = InputView(-2, "x", "Value<16>", (Port("x", 0, N, N, 16),))
    calls = (CallView(0, "Gemm_v1{K=4,N=4}", "Gemm_v1", "call", f"{HELD}.gate_up_proj/linear[0]", (Port("out", 0, N, N, 16),),
                      (R(-2, "x"), R(-1, "model.layers.3.mlp.shared_experts.gate_up_proj.weight"))),
             CallView(1, "SiluMul_v2{I=2}", "SiluMul_v2", "call", f"{HELD}.act_fn/silu_and_mul[0]", (Port("out", 0, 2, 2, 16),), (R(-2, "x"),)),
             CallView(2, "Gemm_v1{K=4,N=4}", "Gemm_v1", "call", "logits_processor/linear[0]", (Port("out", 0, N, N, 16),),
                      (R(1, "out"), R(-1, "lm_head.weight"))))
    P = ProgramView(calls, (w, x), (R(2, "out"),), "synthetic", "synthetic")
    names, held = held_module_paths(P, {c.id: parse_call_name(c.name) for c in calls})
    assert [names[i].module for i in range(3)] == ["model.layers.3.mlp.shared_experts.gate_up_proj", f"{HELD}.act_fn", "logits_processor"]
    assert held == {"resolved": 1, "unresolved": {f"{HELD}.act_fn": 1}}


def _mutated(edit) -> Population:
    b = _Build(0)
    b.step(0, 1)
    b.step(1, 1)
    edit(b)
    P = b.program()
    return Population(P, Correspondence(P), tp={"world": WORLD, "rank": 0})


def _with_reads(b: _Build, c: CallView, reads) -> None:
    b.calls[c.id] = CallView(c.id, c.definition, c.family, c.form, c.name, c.port_table, tuple(reads))


def test_a_collective_that_is_not_one_norms_is_refused_by_name():
    """A fused route that reads no `<norm>.weight`, or an unfused all-reduce two norms read, names no norm: a fault, never a guess."""
    norm = "model.layers.3.post_attention_layernorm"

    def unweighted(b):
        f = next(c for c in b.calls if c.family == "AllReduceRmsNormMnnvl_v1" and c.reads[-1] == R(-1, f"{norm}.weight"))
        _with_reads(b, f, (*f.reads[:-1], R(-2, "ids")))

    def two_norms(b):
        ar = next(c for c in b.calls if c.name.startswith(f"{norm}/")).reads[0]
        other = next(c for c in b.calls if c.name.startswith("model.layers.4.input_layernorm/"))
        _with_reads(b, other, (*other.reads, ar))

    for edit in (unweighted, two_norms):
        pop = _mutated(edit)
        assert len(pop.tp_faults) == 1 and "not one norm's collective" in pop.tp_faults[0], (edit.__name__, pop.tp_faults)
        assert len(pop.norm_site) == 2 * len(NORMS) - 1, edit.__name__
