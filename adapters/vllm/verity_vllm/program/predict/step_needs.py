"""What the DeepSeek sparse-attention decoder (`dsa`) calls for one served engine step at its launch, without building the Program:
each Definition by id with its calls, and, where `dsa` would raise `Unexpressible`, the refusal counted in its place (`Need.gap`)
while the rest of the step is still counted.

Each selection is the one `dsa._Model` makes, through the same registry calls (the all-reduce's served route and the capture that
covers it, the captured cuBLAS route, the LL bf16 tier, the grouped top-k kernel, the tuned fused-MoE tile, FA3's split count over
the launch's key counts) and the vocabulary's kind at the statics `dsa` passes (`common.kind`). Every token-count-dependent
selection reads the step's launch: the tokens the CUDA-graph dispatch pads it to. A request's query rows past the decode threshold
are one refusal per row and layer (the dense-MHA prefill backend while its sequence fits INDEX_TOPK keys, the sparse-MQA prefill past
it), and a row past INDEX_TOPK keys refuses only its top-k (the radix selection); its other Definitions are counted. Consts are not
counted. Where `dsa` writes a request's calls in their own launch, this counts the step's: the grouped top-k kernel reads the
launch's tokens, as the served kernel does. `tests/program/predict/test_step_needs.py` holds the counts to `dsa`'s own Program for
an engine step it writes.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import MANIFESTS, RowEngine, row_file
from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR
from verity_vllm.engine.profiles.devices import fused_moe_config

from . import dsa
from .common import Unexpressible, _arr_shape, kind
from .phi import Phi

__all__ = ["Need", "StepNeeds", "workload_phi"]

_CAPTURE_MAP = "glm-moe-dsa-eager-hopper-tp8-v0.json"


@dataclass(frozen=True)
class Need:
    """One (route, statics) pair a step needs: a Definition's id, or what refused it (`gap`, with `id` naming the statics the refusal
    read as NAME{K=V,...}); `exists` says whether a Definition exists at other statics."""
    route: str
    id: str
    gap: str | None = None
    exists: str = "yes"


def _id(name: str, **statics: object) -> str:
    return f"{name}{{{','.join(f'{k}={v}' for k, v in statics.items())}}}"


def workload_phi(row: str, case: str) -> Phi:
    """The φ of a workload row of record (`config.row_file`, fetched when absent), its checkpoint's config.json the package's
    pinned copy (`hf_configs/<case>.config.json`), as `score.phi_of` makes it."""
    from . import score as S
    w = json.loads(row_file(row).read_text())
    home = Path(tempfile.mkdtemp(prefix="step-needs-hf-"))
    model = w["model"]
    revision: Any = RowEngine.of(w).revision   # a row that pins none names no snapshot dir: os.path.join's TypeError
    cp = Path(S.hf_config_path(str(home), model, revision))
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes((HF_CONFIG_DIR / f"{case}.config.json").read_bytes())
    return S.phi_of({"row": row, "workload": w}, hf_home=str(home))


def _routes_of_definitions() -> dict[str, str]:
    """Each Definition name (version dropped) -> its route in the capture map's `programs`."""
    progs = json.loads((MANIFESTS / "capture-maps" / _CAPTURE_MAP).read_text())["programs"]
    out = {"GumbelTopPTokenSelectSharedGreedy": "token-select", "GumbelTopPTokenSelect": "token-select"}
    for route, p in progs.items():
        for d in p.get("definitions") or ():
            out.setdefault(re.sub(r"_v\d+$", "", d), route)
    return out


class StepNeeds:
    """`dsa`'s decoder for φ (one tensor-parallel rank), and what it calls for an engine step (`step`)."""

    def __init__(self, phi: Phi) -> None:
        from verity_vllm.program.frontend.rules.vllm_dsa import (
            DECODE_THRESHOLD,
            DECODE_THRESHOLD_DEFAULT,
        )
        self.phi, self.net = phi, dsa.Net(phi, 1, 8)
        self.C, self.tp, self.W = self.net.C, self.net.tp, self.net.world
        self._shape = {key: _arr_shape(t) for _, t, key in self.net._leaves if key is not None}
        self._routes = _routes_of_definitions()
        self._memo: dict[tuple, Need] = {}
        C = self.C
        self.H, self.EPS, self.G = int(C["H"]), float(C["EPS"]), int(C["FP8_BLOCK"])
        self.index_topk = int(C["INDEX_TOPK"])
        self.threshold = DECODE_THRESHOLD.get(int(C["NH"]), DECODE_THRESHOLD_DEFAULT)
        NL, first = int(C["NL"]), int(C["FIRST_DENSE"])
        self._layers = Counter((L in self.net.indexers, first <= L) for L in range(NL))
        self._rep = {cls: min(L for L in range(NL) if (L in self.net.indexers, first <= L) == cls) for cls in self._layers}

    def kind(self, role: str, **statics: object) -> Need:
        """The vocabulary's Definition for `role` at `statics`, or the refusal naming why it has none."""
        key = (role, repr(sorted(statics.items(), key=lambda kv: kv[0])))
        if key not in self._memo:
            try:
                f = kind(role, **statics)
                self._memo[key] = Need(self._routes.get(f.name, role), f.id)
            except Unexpressible as e:
                self._memo[key] = Need(role, _id(role, **statics), f"no kind: {e}", "at other statics")
        return self._memo[key]

    def step(self, tokens: int, launch: int, prefills: Sequence[tuple[int, int]], decodes: Sequence[int], requests: int,
             sampling: Mapping[str, int], flashinfer_sampler: bool) -> Counter[Need]:
        """One engine step at this rank -> each Need with its calls. `tokens` rows run in a launch of `launch` tokens (the graph's
        padded size, else `tokens`); `prefills` holds each prefilling request's (query rows, computed tokens) and `decodes` each
        decoding request's key count; `requests` sequences are sampled, `sampling` counting them by kind (`greedy`, `seeded`,
        `random`) and modifier (`top_k`, `top_p`, `min_p`, `penalties`); `flashinfer_sampler` is whether vLLM's FlashInfer sampler
        is on, which samples a step's unseeded requests only when some request filters by top-k or top-p and none is seeded (one
        request's generator sends the whole batch to the native sampler); an unseeded request's select is refused once, as
        FlashInfer's or as unseeded."""
        from verity.definitions.registry.sparse_mla_split import launch_num_splits
        calls = [(int(q), int(done)) for q, done in prefills] + [(1, int(k) - 1) for k in decodes]
        keys: Counter[int] = Counter()
        long_calls = []
        for q, done in calls:
            if q > self.threshold:
                long_calls.append((q, done))
            else:
                keys.update(range(done + 1, done + q + 1))
        batch = [min(k, self.index_topk) for k in sorted(keys.elements())]
        splits = dict(zip(batch, launch_num_splits(batch, int(self.tp.num_sms), self.index_topk, 0))) if batch else {}
        need: Counter[Need] = Counter()

        def add(pairs: list[tuple[Need, int]], times: int = 1) -> None:
            for nd, c in pairs:
                if c:
                    need[nd] += c * times

        M, R, V = int(launch), int(tokens), int(self.C["V"])
        add([(self.kind("embedding_shard", VS=V, H=self.H, START=self.net.rank * V), R), (self._all_reduce(M, R), 1),
             (self.kind("rmsnorm", N=self.H, EPS=self.EPS, TARGET=self.tp, M=M), R)])
        for (indexer, moe), times in self._layers.items():
            L = self._rep[(indexer, moe)]
            if keys:
                add(self._attention(M, keys, splits, L, indexer), times)
            for q, done in long_calls:
                dense = done + q <= self.index_topk
                add([(Need("fa3-prefill-attention" if dense else "dsa-sparse-prefill", _id("prefill-call", ROWS=q, KEYS=done + q),
                           "dense-MHA prefill" if dense else "sparse-MQA prefill", "no" if dense else "per row"), q)], times)
            add(self._reduce_norm(M, R) + (self._moe(M, R, L) if moe else self._mlp(M, R, L)) + self._reduce_norm(M, R), times)
        add(self._sample(int(requests), sampling, flashinfer_sampler))
        return need

    def _fp8_linear(self, M: int, w: tuple[int, str]) -> Need:
        N, K = self._shape[w]
        return self.kind("flashinfer_fp8_linear" if M < 32 else "deepgemm_fp8_linear", K=K, N=N)

    def _cublas(self, M: int, w: object, f32: bool = False) -> Need:
        from verity.definitions.registry.bf16_gemm_check import default_cublas_route

        from verity_vllm.program.frontend.rules.vllm_bindings.bf16_gemm import (
            _f32out_phase,
        )
        N, K = self._shape[w]
        phase = _f32out_phase(K, N) if f32 else "linear"
        route = default_cublas_route(phase, M, K, N)
        if route is None:
            return Need("router-gemm" if f32 else "cublas-bf16-gemm", _id(f"cublas-{phase}", K=K, N=N, M=M), "no capture",
                        "at other statics")
        split = route[0] == "splitk"
        role = ("cublas_splitk_gemm_f32out" if split else "cublas_gemm_f32out") if f32 else ("cublas_splitk_gemm" if split else "gemm_dot")
        return self.kind(role, K=K, N=N, DOT=self.net.dot, **route[1])

    def _router(self, M: int, w: tuple[int, str]) -> Need:
        from verity.definitions.registry.bf16_gemm_check import (
            DEFAULT_LL_BF16_FMA,
            dispatch_sm90,
        )
        N, K = self._shape[w]
        tier, st = dispatch_sm90(M, K, N)
        if tier == "cublas":
            return self._cublas(M, w, f32=True)
        if tier == "dotprod":
            return self.kind("ll_bf16_dotprod", N=N, K=K, FMA=DEFAULT_LL_BF16_FMA, **st)
        return self.kind("ll_bf16_splitk", N=N, K=K, DOT=self.net.dot, **st)

    def _all_reduce(self, M: int, n: int) -> Need:
        from verity.definitions.registry import tp_allreduce as AR

        from verity_vllm.program.frontend.rules.vllm_bindings.collectives import (
            _ROUTE_KINDS,
        )
        setting, route = AR.served_route(self.net.settings, M, self.H)
        c = AR.covered(setting, self.H, route, M) if setting else None
        if c is None:
            return Need("tp-allreduce", _id(f"allreduce-{route}", N=self.H, M=M), "no capture", "at other statics")
        try:
            _, st = AR.route_binding(c, self.W, M, self.H, n)
        except ValueError as e:
            return Need("tp-allreduce", _id(f"allreduce-{route}", N=self.H, M=M), f"no capture: {e}", "at other statics")
        return self.kind(_ROUTE_KINDS[c.definition], **st)

    def _reduce_norm(self, M: int, n: int) -> list[tuple[Need, int]]:
        from verity.definitions.registry import tp_allreduce as AR
        s = self.net.settings
        if s["flashinfer"] and AR.fused_max_tokens(self.H, self.W) >= M:
            setting, _ = AR.served_route(s, M, self.H)
            c = AR.covered(setting, self.H, "fused", M) if setting else None
            if c is None:
                return [(Need("fused-allreduce-rmsnorm", _id("allreduce-rmsnorm-fused", N=self.H, M=M), "no capture",
                              "at other statics"), n)]
            return [(self.kind("all_reduce_rmsnorm_mnnvl", WORLD=self.W, N=self.H, SHOT=AR.mnnvl_shot(M, self.H, self.W),
                               CLUSTER=AR.mnnvl_grid(M, self.H)[1], EPS=self.EPS, FMA=bool(c.statics["FMA"])), n)]
        return [(self._all_reduce(M, n), 1), (self.kind("rmsnorm_fused", N=self.H, EPS=self.EPS, TARGET=self.tp, M=M), n)]

    def _moe(self, M: int, n: int, L: int) -> list[tuple[Need, int]]:
        from verity.definitions.registry.moe import router_kernel

        from verity_vllm.program.frontend import targets as TG
        C, H, G = self.C, self.H, self.G
        E, TOPK, INTER = int(C["NUM_EXPERTS"]), int(C["TOPK"]), int(C["I"])
        pairs = n * TOPK
        out = [(self._router(M, (L, "gate_w")), n), (self._fp8_linear(M, (L, "sh_gate_up_w")), n),
               (self.kind("silu_mul", I=self._shape[(L, "sh_gate_up_w")][0] // 2), n), (self._fp8_linear(M, (L, "sh_down_w")), n)]
        kernel = router_kernel(E, M, TOPK)
        out.append((self.kind("moe_router_sigmoid_bias", E=E, TOPK=TOPK, RSF=float(C["ROUTED_SCALING_FACTOR"]), KERNEL=kernel), n)
                   if kernel else (Need("sigmoid-router", _id("grouped-topk", E=E, M=M), "no capture", "at other statics"), n))
        out.append((self.kind("fp8_group_quant_ue8m0", K=H, G=G), n))
        device: Any = self.net.device   # φ's `device_name` as it states it: one that is no str names no tuned file (AttributeError)
        cfg = fused_moe_config(E, INTER, device, "fp8_w8a8", (G, G), M)
        why: str | None
        if cfg is None:
            why = f"no tuned fused-MoE config at M={M}"
        else:
            BM, BN, KB, warps = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], min(cfg["BLOCK_SIZE_K"], G), cfg["num_warps"]
            swap = BM < 64 and BN >= 64
            why = TG.moe_fp8_tuned_tile_refusal(self.tp, BM, BN, KB, warps, swap)
        if why:
            out.append((Need("fp8-moe-triton", _id("fused-moe-tile", E=E, M=M), f"no capture: {why}", "at other statics"), 2 * pairs))
        else:
            st = {"E": E, "G": G, "KB": KB, "SWAP": swap, "DOT": TG.moe_expert_fp8_launch_dot(self.tp, BM, BN, warps, swap)}
            out += [(self.kind("moe_expert_gemm_dot", K=H, N=2 * INTER, **st), pairs), (self.kind("moe_expert_gemm_w_dot", K=INTER, N=H, **st), pairs)]
        out += [(self.kind("silu_mul", I=INTER), pairs), (self.kind("fp8_group_quant_ue8m0", K=INTER, G=G), pairs),
                (self.kind("moe_sum", TOPK=TOPK, H=H), n), (self.kind("residual_add", N=H), n)]
        return out

    def _mlp(self, M: int, n: int, L: int) -> list[tuple[Need, int]]:
        return [(self._fp8_linear(M, (L, "gate_up_w")), n), (self.kind("silu_mul", I=self._shape[(L, "gate_up_w")][0] // 2), n),
                (self._fp8_linear(M, (L, "down_w")), n)]

    def _attention(self, M: int, keys: Counter[int], splits: dict[int, int], L: int, indexer: bool) -> list[tuple[Need, int]]:
        from verity_vllm.program.frontend.rules.vllm_dsa import FA3_SPARSE_GEOMETRY
        C = self.C
        QL, KVL, R, NOPE, VH, NH = (int(C[k]) for k in ("Q_LORA", "KV_LORA", "QK_ROPE", "QK_NOPE", "V_HEAD", "NH"))
        IH, ID = int(C["INDEX_HEADS"]), int(C["INDEX_HEAD_DIM"])
        G = FA3_SPARSE_GEOMETRY[(R, KVL)]
        n = sum(keys.values())
        out = [(self._fp8_linear(M, (L, "qkv_a_w")), n), (self.kind("dsa_rmsnorm", N=QL, EPS=self.EPS), n),
               (self.kind("dsa_rmsnorm", N=KVL, EPS=self.EPS), n), (self.kind("dsa_kv_rope", D=R), n),
               (self._fp8_linear(M, (L, "q_b_w")), n), (self.kind("batched_gemm", B=NH, K=NOPE, N=KVL, DOT=self.net.dot), n),
               (self.kind("dsa_q_rope", D=R), n * NH), (self.kind("batched_gemm", B=NH, K=KVL, N=VH, DOT=self.net.dot), n),
               (self._fp8_linear(M, (L, "o_w")), n)]
        if indexer:
            out += [(self._cublas(M, (L, "wk_w")), n), (self.kind("dsa_indexer_k_row", D=ID, ROPE=R, EPS=float(C["INDEX_K_EPS"])), n),
                    (self._fp8_linear(M, (L, "wq_b_w")), n), (self.kind("dsa_indexer_q", H=IH, D=ID, ROPE=R), n)]
        for k, c in keys.items():
            T = min(k, self.index_topk)
            if indexer:
                out.append((self.kind("dsa_mqa_logits", N=k, H=IH, D=ID), c))
                out.append((self.kind("dsa_topk_short_row", TOPK=self.index_topk), c) if k <= self.index_topk else
                           (Need("dsa-topk", _id("radix-topk", TOPK=self.index_topk, KEYS=k), "radix top-k", "no"), c))
            out.append((self.kind("sparse_mla_split", T=T, NS=splits[T], NH=NH, DR=R, DV=KVL, BN=G["BN"], DOT=G["DOT"], INV=G["INV"],
                                  SM_SCALE=float(C["ATTN_SCALE"])), c))
        return out

    def _sample(self, n_seq: int, sampling: Mapping[str, int], flashinfer_sampler: bool) -> list[tuple[Need, int]]:
        V, W = int(self.C["V"]), self.W
        out = [(self._cublas(n_seq, "lm_head"), n_seq), (self.kind("all_gather", WORLD=W, N=V), n_seq)]
        if sampling.get("greedy"):
            out.append((self.kind("token_select", V=V * W), sampling["greedy"]))
        seeded, unseeded = sampling.get("seeded", 0), sampling.get("random", 0)
        if seeded:
            from verity.definitions.registry.topp_geometry import registered_splits
            out.append((self.kind("token_select_stochastic_split", V=V * W, S=registered_splits(int(self.tp.num_sms))), seeded))
        if unseeded:
            fi = flashinfer_sampler and not seeded and bool(sampling.get("top_k") or sampling.get("top_p"))
            out.append((Need("token-select", _id("flashinfer-sampler" if fi else "select-random", V=V * W),
                             "FlashInfer sampler" if fi else "unseeded sampling", "no"), unseeded))
        for tag, gap in (("top_k", "top-k sampling"), ("min_p", "min-p sampling"), ("penalties", "penalties")):
            if sampling.get(tag):
                out.append((Need("token-select", _id(f"select-{tag}", V=V * W), gap, "no"), sampling[tag]))
        return out
