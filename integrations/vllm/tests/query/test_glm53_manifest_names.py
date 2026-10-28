"""Every identity of GLM-5.3's TP8 manifest binds from what the runtime commits (a Commit of record bound none
of 26,686 of rank 0's, by name): on a rank Program shaped like GLM's -- dense layer 0, MoE layers 3 and 4 with the shared expert under
the runner's held path `_shared_experts._layer`, a prefill step on the unfused all-reduce + `RMSNormFusedCudaVector8` and DeepGEMM path
and a decode step on the fused `AllReduceRmsNormMnnvl` and FlashInfer path -- every `instance_outputs` / `tp_rank_partials` identity
binds (`verity_vllm.query.manifest.bind`) from the names the capture map `glm-moe-dsa-eager-hopper-tp8-v0` commits and the ones the committer
commits per call (`native_host.CALL_DEPENDENT_RE`):

* `<norm>/all_reduce`, the unfused path's all-reduced tensor (`allreduce_norm`, from the norm's pre-hook), the norm site's member;
* `<linear>/x_q` and `<linear>/x_s`, `DeepGemmFp8Linear_v1`'s quant of its input, which `linear_quant_source` declares as its ports
  `0` / `1`, so the hook's `<linear>/0` binds the one left, y."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import pytest

from tests.acquire.glm53_model import CMAP, ENV
from verity_vllm.acquire.sources import linear_quant_source as LQ
from verity_vllm.commit import capture_map as CM
from verity_vllm.query.manifest.bind import Binder
from verity_vllm.commit.committer.native_host import CALL_DEPENDENT_RE
from verity_vllm.correspondence.reader_for_query import Correspondence, held_module_paths, parse_call_name
from verity_vllm.query.collective_names import NORM_SITE_MEMBER
from verity_vllm.query.compose import merge_ranks
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

    def __init__(self, rank: int, fused=lambda step, norm: step > 0):
        self.rank, self.calls, self.weights, self.peers, self.coll, self.fused = rank, [], [], [], 0, fused

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
        """`fused_allreduce_rms_norm(own, residual, norm)` called from `site`: flat (one all-reduce over the M rows, then the norm's rows),
        or fused (`self.fused`, by default at decode: one Call per row reading its residual and the norm weight, the route's
        Definition being one token)."""
        M, nw = len(own), self.w(f"{norm}.weight")
        if not self.fused(step, norm):
            ar = self.call("AllReduceMultimem_v1", f"{site}.all_reduce_{self.coll}/all_reduce[0]/all_reduce", self.parts(own, M * N),
                           (("out", M * N, 16),))
            rows = [self.call("RMSNormFusedCudaVector8_v1", f"{norm}/fused_add_rms_norm[{i}]", [R(ar, "out"), residual[i], nw],
                              (("0", N, 16), ("1", N, 16))) for i in range(M)]
            return [R(c, "0") for c in rows], [R(c, "1") for c in rows]
        k = self.coll
        rows = [self.call(FUSED, f"{site}.fused_allreduce_rms_norm_{k}/fused_allreduce_rms_norm[{i}]/fused_all_reduce_rmsnorm",
                          [*self.parts([r], N), residual[i], nw], (("0", N, 16), ("1", N, 16))) for i, r in enumerate(own)]
        return [R(c, "0") for c in rows], [R(c, "1") for c in rows]

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


def rank_program(rank: int, M: int = 2, **kw) -> ProgramView:
    b = _Build(rank, **kw)
    b.step(0, M)
    b.step(1, 1)
    return b.program()


def _manifest(rank: int, **kw) -> dict:
    P = rank_program(rank, **kw)
    res = {"program": {"digest": f"d{rank}", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}},
           "tp": {"world": WORLD, "rank": rank}}
    return request_manifest(P, Correspondence(P), res, request_id="r0", rank=rank, ranks=WORLD).manifest


NORMS = {f"model.layers.{k}.{n}" for k in DENSE + MOE for n in ("input_layernorm", "post_attention_layernorm")} - {"model.layers.0.input_layernorm"} | {"model.norm"}
QUANT_LINEARS = {f"model.layers.{k}.self_attn.o_proj" for k in DENSE + MOE} | {f"model.layers.0.mlp.{p}" for p in ("gate_up_proj", "down_proj")} | \
    {f"model.layers.{k}.mlp.shared_experts.{p}" for k in MOE for p in ("gate_up_proj", "down_proj")}
#: what the committer commits per call at the prefill (the norm path, the linear's quant branch), not from a capture-map tap
PER_CALL = [f"{n}/{NORM_SITE_MEMBER}" for n in sorted(NORMS)] + [f"{m}/{q}" for m in sorted(QUANT_LINEARS) for q in LQ.MEMBERS]
assert all(CALL_DEPENDENT_RE.search(n) for n in PER_CALL)


@pytest.fixture(scope="module")
def manifests() -> list[dict]:
    return [_manifest(rank) for rank in range(WORLD)]


def _events(rank: int, step: int) -> list[str]:
    """What one step commits on `rank`: the capture map's names (both attention routes; this Program has no attention module), and at
    the prefill the per-call names."""
    names = {n for routes in (("sparse-mqa",), ("dense-mha",)) for n in CM.committer_names(CMAP, ENV, routes=routes, rank=rank)}
    return sorted(names) + (PER_CALL if step == 0 else [])


def _named(m: dict) -> set[tuple[int, str, str]]:
    return {(r["step"], r["op_path"], r["output_member"]) for r in m["identities"] if r["family"] in ("instance_outputs", "tp_rank_partials")}


@pytest.mark.parametrize("rank", [0, 5])
def test_every_required_identity_binds_from_what_the_rank_commits(manifests, rank):
    m = manifests[rank]
    assert m["complete"], m["unmodelled"]
    binder = Binder(m["identities"], rank=rank, ports=LQ.PORTS)
    for step in (0, 1):
        bound = [(step, op, mem) for op, mem, _k, how in binder.bind(step, _events(rank, step)) if how != "unbound"]
        assert len(bound) == len(set(bound)), "an identity bound twice"
        missing = {x for x in _named(m) if x[0] == step} - set(bound)
        assert not missing, sorted(missing)
    named = {(op, mem) for _s, op, mem in _named(m)}
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


def test_deepgemms_quant_branch_binds_its_ports_so_the_hooks_slot_0_is_y(manifests):
    """At the prefill a quantized linear is `DeepGemmFp8Linear_v1`, whose ports (x_q, x_s, y) the manifest names `0` / `1` / `2`; the
    quant source's `x_q` / `x_s` bind ports 0 / 1 and the hook's slot 0 the one left.  Without the quant branch slot 0 binds nothing,
    never q.  At the decode FlashInfer's one output is `out`, which slot 0 binds."""
    m = manifests[0]
    rows = {(r["step"], r["op_path"], r["output_member"]): r for r in m["identities"] if r["family"] == "instance_outputs"}
    binder = Binder(m["identities"], rank=0, ports=LQ.PORTS)
    for lin in QUANT_LINEARS:
        assert [(rows[(0, lin, p)]["spec"], rows[(0, lin, p)]["dtype"]) for p in "012"] == [(DEEPGEMM, "w8"), (DEEPGEMM, "f32"), (DEEPGEMM, "bf16")]
        assert binder.bind(0, [f"{lin}/x_q", f"{lin}/0", f"{lin}/x_s"]) == [(lin, "0", 0, "port"), (lin, "2", 0, "slot"), (lin, "1", 0, "port")]
        assert binder.bind(0, [f"{lin}/0"]) == [(lin, "", 0, "unbound")]
        assert rows[(1, lin, "out")]["spec"] == FLASHINFER and binder.bind(1, [f"{lin}/0"]) == [(lin, "out", 0, "slot")]
    for k in MOE:
        assert {s: rows[(s, f"model.layers.{k}.mlp.gate", "out")]["spec"] for s in (0, 1)} == GATE


def test_the_shared_expert_is_each_layers_own(manifests):
    """The held `_shared_experts._layer.<leaf>` rows of the two MoE layers were one identity of 2 layers' rows; each is now its layer's."""
    rows = {(r["step"], r["op_path"], r["output_member"]): r["rows"] for r in manifests[0]["identities"] if r["family"] == "instance_outputs"}
    for k in MOE:
        sh = f"model.layers.{k}.mlp.shared_experts"
        assert [rows[(0, f"{sh}.{leaf}", mem)] for leaf, mem in (("gate_up_proj", "2"), ("act_fn", "out"), ("down_proj", "2"))] == [2, 2, 2]
        assert [rows[(1, f"{sh}.{leaf}", mem)] for leaf, mem in (("gate_up_proj", "out"), ("act_fn", "out"), ("down_proj", "out"))] == [1, 1, 1]
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


def _mutated(edit, **kw) -> Population:
    b = _Build(0, **kw)
    b.step(0, 1)
    b.step(1, 1)
    edit(b)
    P = b.program()
    return Population(P, Correspondence(P), tp={"world": WORLD, "rank": 0})


def _with_reads(b: _Build, c: CallView, reads, name: str | None = None) -> None:
    b.calls[c.id] = CallView(c.id, c.definition, c.family, c.form, name or c.name, c.port_table, tuple(reads))


def _respelt(b: _Build, step: int, src: str, dst: str) -> None:
    """`src`'s collective of `step` re-spelt as norm `dst`'s: its Calls read `dst.weight`, and the flat route's rows sit under `dst`."""
    cut = next(c.id for c in b.calls if c.family == "ArgmaxTokenSelect_v1")
    for c in list(b.calls):
        if (c.id > cut) == bool(step) and R(-1, f"{src}.weight") in c.reads:
            _with_reads(b, c, [R(-1, f"{dst}.weight") if r == R(-1, f"{src}.weight") else r for r in c.reads],
                        c.name.replace(f"{src}/fused_add_rms_norm", f"{dst}/fused_add_rms_norm_of_{src.replace('.', '_')}"))


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


def test_a_short_prefill_on_the_fused_route_is_the_rows_of_the_norms_identities():
    """A step of at most 42 rows takes FlashInfer's fused route (the GLM B8 mixed-arrivals row's two 11-token prefills), whose
    Definition is one token: M `AllReduceRmsNormMnnvl_v1` Calls per norm in the step, the M rows of `<norm>/0`, `/1` and
    `/part_rank<r>` -- the whole [M, N] tensors `allreduce_norm` commits -- with every peer bound."""
    ms = [_manifest(rank, fused=lambda step, norm: True) for rank in range(WORLD)]
    assert all(m["complete"] for m in ms), [m["unmodelled"] for m in ms]
    rank = 6
    pre = {r["output_member"]: r for r in ms[rank]["identities"] if r["step"] == 0 and r["op_path"] == "model.layers.3.input_layernorm"}
    assert {k: (r["spec"], r["rows"], r["element_range"]) for k, r in pre.items()} == \
        dict.fromkeys(("0", "1", f"part_rank{rank}"), ("AllReduceRmsNormMnnvl_v1", 2, [0, 2 * N]))
    b = merge_ranks(ms)["tp_peer_binding"]
    assert b["n_unbound"] == 0 and b["bound"] == b["collective_outputs"] * (WORLD - 1), b


def test_a_norms_collectives_of_one_step_are_one_or_all_fused():
    """Two fused Calls of one norm in a step are its rows; two flat all-reduces, or a fused Call beside a flat one, are refused by
    name, each of them."""
    norm, other = "model.layers.3.post_attention_layernorm", "model.layers.4.input_layernorm"
    pop = _mutated(lambda b: _respelt(b, 1, other, norm))
    assert not pop.tp_faults and len(pop.norm_site) == 2 * len(NORMS), pop.tp_faults
    for fused in (lambda step, n: step > 0, lambda step, n: step > 0 or n == norm):
        pop = _mutated(lambda b: _respelt(b, 0, other, norm), fused=fused)
        assert len(pop.tp_faults) == 2 and all("not one norm's collective" in f for f in pop.tp_faults), pop.tp_faults
        assert len(pop.norm_site) == 2 * len(NORMS) - 2
