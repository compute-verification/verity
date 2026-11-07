"""The gap between a route capture's served steps and class G: every (route, statics) pair the captured steps need that class G
doesn't cover, with its proof blocks a served token, whether a Definition exists at those or other statics, and its owner.

    python gap.py CAPTURE_DIR --check CIRCUIT_CHECK.json [--rank R] [--json OUT.json] [--md OUT.md]

CIRCUIT_CHECK.json is `circuit-check`'s report over class G's Definitions (its `targets`: each id and its failures): an id
there is registered, and it passes when it has no failures.

Each step is evaluated as the class G decoder (`verity_vllm.program.predict.dsa`) writes a launch, one tensor-parallel rank:
every selection it makes reads the step's launch size (the CUDA-graph dispatch's padded tokens) and its rows' key counts, through
the same registry calls (the all-reduce route and its capture, the captured cuBLAS route, the LL bf16 tier, the router kernel, the
tuned fused-MoE tile, FA3's split count), and each Definition is `common.kind`'s at the statics the decoder passes. Where the
decoder refuses (a record no capture covers, a prefill call past the decode threshold, a row past INDEX_TOPK keys, a sampler the
vocabulary lacks), the step's calls are counted against the refusal instead, and evaluation goes on. Consts are not counted.
Request steps the decoder writes per request are counted per launch here: the router kernel reads the launch's tokens, as the
served kernel does.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routes import ranks, steps

#: class G's workload row (`engine.profiles.engine_classes`), whose φ fixes the model, target, device and engine settings
ROW = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager"
CAPTURE_MAP = "glm-moe-dsa-eager-hopper-tp8-v0.json"
#: what closes each refusal: a Definition (circuits), a kernel capture or a deterministic kernel (kernels), serving (vllm)
OWNERS = {"no capture": "kernels", "unregistered": "circuits", "fails circuit-check": "circuits", "no kind": "circuits",
          "radix top-k": "circuits + kernels", "dense-MHA prefill": "circuits + vllm", "sparse-MQA prefill": "kernels + vllm",
          "unseeded sampling": "vllm", "top-k sampling": "circuits", "min-p sampling": "circuits", "penalties": "circuits",
          "FlashInfer sampler": "vllm"}
#: class G's engine limits (`CLASS_G_LIMITS`), the prefix cache off and eager execution
CLASS_G = {"max_num_seqs": 8, "max_num_batched_tokens": 2048, "max_model_len": 2048, "long_prefill_token_threshold": 256}


def phi_of(row: str = ROW):
    """The φ of a workload row of record, its checkpoint's config.json the package's pinned copy (as `score.phi_of` makes it)."""
    from verity_vllm.config import WORKLOADS, RowEngine
    from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR
    from verity_vllm.program.predict import score as S
    w = json.loads((WORKLOADS / f"{row}.json").read_text())
    home = Path(tempfile.mkdtemp(prefix="gap-hf-"))
    cp = Path(S.hf_config_path(str(home), w["model"], RowEngine.of(w).revision))
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes((HF_CONFIG_DIR / "GLM53.config.json").read_bytes())
    return S.phi_of({"row": row, "workload": w}, hf_home=str(home))


def routes_of_definitions() -> dict[str, str]:
    """Each Definition name (version dropped) -> its route in the capture map's `programs`."""
    from verity_vllm.config import MANIFESTS
    progs = json.loads((MANIFESTS / "capture-maps" / CAPTURE_MAP).read_text())["programs"]
    out = {}
    for route, p in progs.items():
        for d in p.get("definitions") or ():
            out.setdefault(re.sub(r"_v\d+$", "", d), route)
    return out


@dataclass(frozen=True)
class Need:
    """One (route, statics) pair a step needs: a Definition's id, or what refused it (`gap`, with `id` naming the statics the
    refusal read as NAME{K=V,...}), and whether a Definition exists at other statics (`exists`)."""
    route: str
    id: str
    gap: str | None = None
    exists: str = "yes"


def _id(name: str, **statics) -> str:
    return f"{name}{{{','.join(f'{k}={v}' for k, v in statics.items())}}}"


class Launches:
    """The decoder's selections at a launch, memoized by the statics they read."""

    def __init__(self, phi) -> None:
        from verity_vllm.program.predict import dsa
        from verity_vllm.program.predict.common import _arr_shape
        self.phi, self.net = phi, dsa.Net(phi, 1, 8)
        self.C, self.tp, self.W = self.net.C, self.net.tp, self.net.world
        self.shape = {key: _arr_shape(t) for _, t, key in self.net._leaves if key is not None}
        self.routes = routes_of_definitions()
        self._memo: dict[tuple, Need] = {}
        C = self.C
        self.H, self.EPS, self.G = int(C["H"]), float(C["EPS"]), int(C["FP8_BLOCK"])
        self.TOPK_KEYS = int(C["INDEX_TOPK"])
        from verity_vllm.program.frontend.rules.vllm_dsa import DECODE_THRESHOLD, DECODE_THRESHOLD_DEFAULT
        self.threshold = DECODE_THRESHOLD.get(int(C["NH"]), DECODE_THRESHOLD_DEFAULT)
        NL, first = int(C["NL"]), int(C["FIRST_DENSE"])
        self.layers = Counter((L in self.net.indexers, L >= first) for L in range(NL))
        self.rep = {cls: min(L for L in range(NL) if (L in self.net.indexers, L >= first) == cls) for cls in self.layers}

    def kind(self, role: str, **statics) -> Need:
        key = (role, repr(sorted(statics.items(), key=lambda kv: kv[0])))
        if key not in self._memo:
            from verity_vllm.program.predict.common import Unexpressible, kind
            try:
                f = kind(role, **statics)
                self._memo[key] = Need(self.routes.get(f.name, role), f.id)
            except Unexpressible as e:
                self._memo[key] = Need(role, _id(role, **statics), f"no kind: {e}", "at other statics")
        return self._memo[key]

    def fp8_linear(self, M: int, w) -> Need:
        N, K = self.shape[w]
        return self.kind("flashinfer_fp8_linear" if M < 32 else "deepgemm_fp8_linear", K=K, N=N)

    def cublas(self, M: int, w, f32: bool = False) -> Need:
        from verity.ml.definitions.registry.bf16_gemm_check import default_cublas_route
        from verity_vllm.program.frontend.rules.vllm_bindings.bf16_gemm import _f32out_phase
        N, K = self.shape[w]
        phase = _f32out_phase(K, N) if f32 else "linear"
        route = default_cublas_route(phase, M, K, N)
        if route is None:
            return Need("router-gemm" if f32 else "cublas-bf16-gemm", _id(f"cublas-{phase}", K=K, N=N, M=M), "no capture",
                        "at other statics")
        split = route[0] == "splitk"
        role = ("cublas_splitk_gemm_f32out" if split else "cublas_gemm_f32out") if f32 else ("cublas_splitk_gemm" if split else "gemm_dot")
        return self.kind(role, K=K, N=N, DOT=self.net.dot, **route[1])

    def router(self, M: int, w) -> Need:
        from verity.ml.definitions.registry.bf16_gemm_check import DEFAULT_LL_BF16_FMA, dispatch_sm90
        N, K = self.shape[w]
        tier, st = dispatch_sm90(M, K, N)
        if tier == "cublas":
            return self.cublas(M, w, f32=True)
        if tier == "dotprod":
            return self.kind("ll_bf16_dotprod", N=N, K=K, FMA=DEFAULT_LL_BF16_FMA, **st)
        return self.kind("ll_bf16_splitk", N=N, K=K, DOT=self.net.dot, **st)

    def all_reduce(self, M: int, n: int) -> Need:
        from verity.ml.definitions.registry import tp_allreduce as AR
        from verity_vllm.program.frontend.rules.vllm_bindings.collectives import _ROUTE_KINDS
        setting, route = AR.served_route(self.net.settings, M, self.H)
        c = AR.covered(setting, self.H, route, M) if setting else None
        if c is None:
            return Need("tp-allreduce", _id(f"allreduce-{route}", N=self.H, M=M), "no capture", "at other statics")
        try:
            _, st = AR.route_binding(c, self.W, M, self.H, n)
        except ValueError as e:
            return Need("tp-allreduce", _id(f"allreduce-{route}", N=self.H, M=M), f"no capture: {e}", "at other statics")
        return self.kind(_ROUTE_KINDS[c.definition], **st)

    def reduce_norm(self, M: int, n: int) -> list[tuple[Need, int]]:
        from verity.ml.definitions.registry import tp_allreduce as AR
        s = self.net.settings
        if s["flashinfer"] and M <= AR.fused_max_tokens(self.H, self.W):
            setting, _ = AR.served_route(s, M, self.H)
            c = AR.covered(setting, self.H, "fused", M) if setting else None
            if c is None:
                return [(Need("fused-allreduce-rmsnorm", _id("allreduce-rmsnorm-fused", N=self.H, M=M), "no capture",
                              "at other statics"), n)]
            return [(self.kind("all_reduce_rmsnorm_mnnvl", WORLD=self.W, N=self.H, SHOT=AR.mnnvl_shot(M, self.H, self.W),
                               CLUSTER=AR.mnnvl_grid(M, self.H)[1], EPS=self.EPS, FMA=bool(c.statics["FMA"])), n)]
        return [(self.all_reduce(M, n), 1), (self.kind("rmsnorm_fused", N=self.H, EPS=self.EPS, TARGET=self.tp, M=M), n)]

    def moe(self, M: int, n: int, L: int) -> list[tuple[Need, int]]:
        from verity.ml.definitions.registry.moe import router_kernel
        from verity_vllm.engine.profiles.devices import fused_moe_config
        from verity_vllm.program.frontend import targets as TG
        C, H, G = self.C, self.H, self.G
        E, TOPK, I = int(C["NUM_EXPERTS"]), int(C["TOPK"]), int(C["I"])
        pairs = n * TOPK
        out = [(self.router(M, (L, "gate_w")), n), (self.fp8_linear(M, (L, "sh_gate_up_w")), n),
               (self.kind("silu_mul", I=self.shape[(L, "sh_gate_up_w")][0] // 2), n), (self.fp8_linear(M, (L, "sh_down_w")), n)]
        kernel = router_kernel(E, M, TOPK)
        out.append((self.kind("moe_router_sigmoid_bias", E=E, TOPK=TOPK, RSF=float(C["ROUTED_SCALING_FACTOR"]), KERNEL=kernel), n)
                   if kernel else (Need("sigmoid-router", _id("grouped-topk", E=E, M=M), "no capture", "at other statics"), n))
        out.append((self.kind("fp8_group_quant_ue8m0", K=H, G=G), n))
        cfg = fused_moe_config(E, I, self.net.device, "fp8_w8a8", (G, G), M)
        why = None
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
            out += [(self.kind("moe_expert_gemm_dot", K=H, N=2 * I, **st), pairs), (self.kind("moe_expert_gemm_w_dot", K=I, N=H, **st), pairs)]
        out += [(self.kind("silu_mul", I=I), pairs), (self.kind("fp8_group_quant_ue8m0", K=I, G=G), pairs),
                (self.kind("moe_sum", TOPK=TOPK, H=H), n), (self.kind("residual_add", N=H), n)]
        return out

    def mlp(self, M: int, n: int, L: int) -> list[tuple[Need, int]]:
        return [(self.fp8_linear(M, (L, "gate_up_w")), n), (self.kind("silu_mul", I=self.shape[(L, "gate_up_w")][0] // 2), n),
                (self.fp8_linear(M, (L, "down_w")), n)]

    def attention(self, M: int, keys: Counter[int], splits: dict[int, int], L: int, indexer: bool) -> list[tuple[Need, int]]:
        """The decode-path attention of rows with these key counts (count of rows per count), at launch M."""
        C = self.C
        QL, KVL, R, NOPE, VH, NH = (int(C[k]) for k in ("Q_LORA", "KV_LORA", "QK_ROPE", "QK_NOPE", "V_HEAD", "NH"))
        IH, ID = int(C["INDEX_HEADS"]), int(C["INDEX_HEAD_DIM"])
        from verity_vllm.program.frontend.rules.vllm_dsa import FA3_SPARSE_GEOMETRY
        G = FA3_SPARSE_GEOMETRY[(R, KVL)]
        n = sum(keys.values())
        w = lambda f: (L, f)  # noqa: E731
        out = [(self.fp8_linear(M, w("qkv_a_w")), n), (self.kind("dsa_rmsnorm", N=QL, EPS=self.EPS), n),
               (self.kind("dsa_rmsnorm", N=KVL, EPS=self.EPS), n), (self.kind("dsa_kv_rope", D=R), n),
               (self.fp8_linear(M, w("q_b_w")), n), (self.kind("batched_gemm", B=NH, K=NOPE, N=KVL, DOT=self.net.dot), n),
               (self.kind("dsa_q_rope", D=R), n * NH), (self.kind("batched_gemm", B=NH, K=KVL, N=VH, DOT=self.net.dot), n),
               (self.fp8_linear(M, w("o_w")), n)]
        if indexer:
            out += [(self.cublas(M, w("wk_w")), n), (self.kind("dsa_indexer_k_row", D=ID, ROPE=R, EPS=float(C["INDEX_K_EPS"])), n),
                    (self.fp8_linear(M, w("wq_b_w")), n), (self.kind("dsa_indexer_q", H=IH, D=ID, ROPE=R), n)]
        for k, c in keys.items():
            T = min(k, self.TOPK_KEYS)
            if indexer:
                out.append((self.kind("dsa_mqa_logits", N=k, H=IH, D=ID), c))
                out.append((self.kind("dsa_topk_short_row", TOPK=self.TOPK_KEYS), c) if k <= self.TOPK_KEYS else
                           (Need("dsa-topk", _id("radix-topk", TOPK=self.TOPK_KEYS, KEYS=k), "radix top-k", "no"), c))
            out.append((self.kind("sparse_mla_split", T=T, NS=splits[T], NH=NH, DR=R, DV=KVL, BN=G["BN"], DOT=G["DOT"], INV=G["INV"],
                                  SM_SCALE=float(C["ATTN_SCALE"])), c))
        return out

    def sample(self, n_seq: int, sampling: dict[str, int], flashinfer_sampler: bool) -> list[tuple[Need, int]]:
        V, W = int(self.C["V"]), self.W
        out = [(self.cublas(n_seq, "lm_head"), n_seq), (self.kind("all_gather", WORLD=W, N=V), n_seq)]
        if sampling.get("greedy"):
            out.append((self.kind("token_select", V=V * W), sampling["greedy"]))
        for tag, gap in (("random", "unseeded sampling"), ("top_k", "top-k sampling"), ("min_p", "min-p sampling"),
                         ("penalties", "penalties")):
            if sampling.get(tag):
                out.append((Need("token-select", _id(f"select-{tag}", V=V * W), gap, "no"), sampling[tag]))
        seeded = sampling.get("seeded", 0)
        if seeded:
            out.append((self.kind("token_select_stochastic", V=V * W, CONSTRUCTION=self.tp.effective_sampler_construction()), seeded))
        stochastic = seeded + sampling.get("random", 0)
        if stochastic and flashinfer_sampler:
            out.append((Need("token-select", _id("flashinfer-sampler", V=V * W), "FlashInfer sampler", "no"), stochastic))
        return out

    def step(self, row: dict[str, Any], flashinfer_sampler: bool) -> tuple[Counter[Need], Counter[str]]:
        """One captured step at this rank: each Need with its calls, and the step's class G engine facts (tokens each)."""
        from verity.ml.definitions.registry.sparse_mla_split import launch_num_splits
        R = int(row["tokens"])
        cg = row.get("cg") or {}
        M = int(cg.get("num_tokens") or R)
        calls: list[tuple[int, int]] = [(int(q), int(done)) for q, done, _ in row["prefill"]]
        ctx = row.get("decode_ctx")
        if ctx is None:
            ctx = [int(row["decode_ctx_max"])] * int(row["decode"])
        calls += [(1, int(c) - 1) for c in ctx]
        decode_keys: Counter[int] = Counter()
        long_calls: list[tuple[int, int]] = []
        for q, done in calls:
            if q > self.threshold:
                long_calls.append((q, done))
            else:
                decode_keys.update(range(done + 1, done + q + 1))
        batch = [min(k, self.TOPK_KEYS) for k in sorted(decode_keys.elements())]
        splits = dict(zip(batch, launch_num_splits(batch, int(self.tp.num_sms), self.TOPK_KEYS, 0))) if batch else {}
        need: Counter[Need] = Counter()

        def add(pairs, times: int = 1) -> None:
            for nd, c in pairs:
                if c:
                    need[nd] += c * times

        V = int(self.C["V"])
        add([(self.kind("embedding_shard", VS=V, H=self.H, START=self.net.rank * V), R), (self.all_reduce(M, R), 1),
             (self.kind("rmsnorm", N=self.H, EPS=self.EPS, TARGET=self.tp, M=M), R)])
        for (indexer, moe), times in self.layers.items():
            L = self.rep[(indexer, moe)]
            if decode_keys:
                add(self.attention(M, decode_keys, splits, L, indexer), times)
            for q, done in long_calls:
                dense = done + q <= self.TOPK_KEYS
                add([(Need("fa3-prefill-attention" if dense else "dsa-sparse-prefill", _id("prefill-call", ROWS=q, KEYS=done + q),
                           "dense-MHA prefill" if dense else "sparse-MQA prefill", "no" if dense else "per row"), q)], times)
            add(self.reduce_norm(M, R) + (self.moe(M, R, L) if moe else self.mlp(M, R, L)) + self.reduce_norm(M, R), times)
        add(self.sample(int(row["reqs"]), row.get("sampling") or {}, flashinfer_sampler))
        facts: Counter[str] = Counter()
        if cg.get("cg_mode") not in (None, "NONE"):
            facts["CUDA graphs (eager in class G)"] += R
        if M != R:
            facts["launch padded past its rows"] += R
        if row.get("prefix_hit_tokens"):
            facts["prefix-cache hit (off in class G)"] += R
        if int(row["reqs"]) > CLASS_G["max_num_seqs"]:
            facts["more sequences than class G's max_num_seqs"] += R
        if R > CLASS_G["max_num_batched_tokens"]:
            facts["more tokens than class G's max_num_batched_tokens"] += R
        if max((done + q for q, done in calls), default=0) > CLASS_G["max_model_len"]:
            facts["a sequence past class G's max_model_len"] += R
        if any(q > CLASS_G["long_prefill_token_threshold"] for q, _ in calls):
            facts["a prefill chunk past class G's long-prefill threshold"] += R
        return need, facts


def statuses(check: dict[str, Any]) -> dict[str, bool]:
    """circuit-check's report -> each registered id and whether it passes."""
    return {t["target"]: not t.get("failures") for t in check["targets"]}


def flashinfer_sampler_on(capture: Path) -> bool:
    """Whether the served engine ran FlashInfer's sampler (vLLM's default unless run.sh's fallback set the switch to 0)."""
    env = capture.parent / "engine_env.txt"
    return not re.search(r"VLLM_USE_FLASHINFER_SAMPLER=0\b", env.read_text() if env.is_file() else "")


def _statics(ids: list[str]) -> str:
    """A compact range of the ids' statics: each static that varies as min..max (and how many values), the rest once; an
    all-reduce's TREES left out."""
    vals: dict[str, set] = defaultdict(set)
    for i in ids:
        m = re.search(r"\{(.*)\}$", i)
        body = re.sub(r"TREES=.*?(?=,FTZ=|$)", "", m.group(1)) if m else ""
        for part in filter(None, body.split(",")):
            k, _, v = part.partition("=")
            vals[k].add(v)
    out = []
    for k, vs in vals.items():
        if len(vs) == 1:
            out.append(f"{k}={next(iter(vs))}")
        else:
            nums = sorted(int(v) for v in vs if re.fullmatch(r"-?\d+", v))
            out.append(f"{k}={nums[0]}..{nums[-1]} ({len(vs)} values)" if len(nums) == len(vs) else f"{k}∈{{{','.join(sorted(vs))}}}")
    return ", ".join(out)


def gap(capture: Path, check: dict[str, Any], rank: int | None = None, phi=None) -> dict[str, Any]:
    """The capture's needs at one recording rank (the lowest unless named) against class G's registered, passing Definitions."""
    rank = min(ranks(capture)) if rank is None else rank
    L = Launches(phi if phi is not None else phi_of())
    reg = statuses(check)
    fi = flashinfer_sampler_on(capture)
    need: Counter[Need] = Counter()
    facts: Counter[str] = Counter()
    tokens = n_steps = 0
    for row in steps(capture, rank):
        nd, f = L.step(row, fi)
        need.update(nd)
        facts.update(f)
        tokens += int(row["tokens"])
        n_steps += 1
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    blocks = sum(need.values())
    for nd, c in need.items():
        if nd.gap is None:
            status = "covered" if reg.get(nd.id) else ("fails circuit-check" if nd.id in reg else "unregistered")
        else:
            status = nd.gap.split(":")[0]
        key = (nd.route, re.sub(r"\{.*", "", nd.id), status)
        r = rows.setdefault(key, {"route": key[0], "definition": key[1], "status": status, "owner": OWNERS.get(status, ""),
                                  "exists": nd.exists, "why": sorted({nd.gap})[0] if nd.gap else None, "blocks": 0, "ids": []})
        r["blocks"] += c
        r["ids"].append(nd.id)
    out = []
    for r in sorted(rows.values(), key=lambda r: (r["status"] == "covered", -r["blocks"])):
        out.append({**r, "statics": _statics(r["ids"]), "n_statics": len(r["ids"]), "blocks_per_token": r["blocks"] / max(tokens, 1),
                    "share": r["blocks"] / max(blocks, 1), "ids": sorted(r["ids"])[:12]})
    return {"rank": rank, "steps": n_steps, "tokens": tokens, "blocks": blocks, "blocks_per_token": blocks / max(tokens, 1),
            "flashinfer_sampler": fi, "class_g_registered": len(reg), "class_g_passing": sum(reg.values()),
            "gaps": [r for r in out if r["status"] != "covered"], "covered": [r for r in out if r["status"] == "covered"],
            "engine": {k: {"tokens": v, "share": v / max(tokens, 1)} for k, v in facts.most_common()}}


def markdown(g: dict[str, Any]) -> str:
    lines = [f"Rank {g['rank']}: {g['steps']} steps, {g['tokens']} tokens, {g['blocks_per_token']:.4g} proof blocks a token.", "",
             "| route | Definition or refusal | statics | Definition exists | owner | blocks a token | share of blocks |",
             "|---|---|---|---|---|---|---|"]
    for r in g["gaps"]:
        lines.append(f"| {r['route']} | {r['definition']} ({r['status']}) | {r['statics']} | {r['exists']} | "
                     f"{r['owner']} | {r['blocks_per_token']:.4g} | {r['share']:.3%} |")
    lines += ["", "| engine fact | share of tokens |", "|---|---|"]
    lines += [f"| {k} | {v['share']:.1%} |" for k, v in g["engine"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--check", type=Path, required=True, help="circuit-check's JSON report over class G's Definitions")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    g = gap(a.capture, json.loads(a.check.read_text()), a.rank)
    if a.json:
        a.json.write_text(json.dumps(g, indent=1) + "\n")
    md = markdown(g)
    if a.md:
        a.md.write_text(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
