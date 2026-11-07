"""Stage sampled proof blocks of one captured step (`capture.py`) as the one-stage audit's saved values (`a11.py --load-values`).

    python benchmarks/spot_proved/stage.py --capture DIR --step S --out DIR [--per-type 8] [--types T,...]

* **Commitment.** Each rank's ``ops.jsonl`` and ``params.json`` are hashed again and held to its ``meta.json``'s rank root,
  the ranks' roots to the step root every rank's ``commit.json`` names, and the coins to one value on every rank. Every blob
  is held to its SHA-512 name when read, and every opened parameter to its SHA-512 in ``params.json`` and ``commit.json``.
* **Population.** A proof block type's instances are the recorded calls of its kernel in the opened layers, on every rank,
  over the step's real tokens: a group quant's token row, a FlashInfer FP8 linear's (token, output column) when the call has
  fewer than 32 rows (the op's FlashInfer branch), a routed MoE w2 launch's (slot, output column), a fused allreduce + RMSNorm's
  token for its residual half-rows and its norm, a plain allreduce's (token, half-row), and FA3's split combine per
  (token, head). Instances are drawn from it by the capture's coins, ``--per-type`` of each type.
* **Match gate.** Each drawn instance's Definition is evaluated on the captured inputs and must return the captured outputs
  bit for bit (completeness, not soundness: an instance that disagrees is reported and not staged).
* **Out.** ``values.json``, ``units.json``, ``<label>.npz`` (a11's saved-values format), ``kinds.json`` (a11's table of the
  staged labels), ``window.json`` (what the window is and how its values link to serving) and ``stage.json`` (populations,
  draws, the gate's results and every type not staged, with its reason).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

#: the tree a11.py comes from: this file's, else (sent alone into a run's inputs) the working directory's
TREE = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[1] / "one_stage").is_dir() else Path.cwd()
for _p in ("benchmarks/one_stage", "tools/verity/circuit_check/src", "benchmarks/numerical/python", "integrations/vllm",
           "verity/ml/flock/python", "."):
    sys.path.insert(0, str(TREE / _p))
sys.setrecursionlimit(100000)

NP = {"torch.bfloat16": np.uint16, "torch.float16": np.uint16, "torch.float32": np.uint32, "torch.float8_e4m3fn": np.uint8,
      "torch.uint8": np.uint8, "torch.int8": np.int8, "torch.int32": np.int32, "torch.int64": np.int64, "torch.bool": np.uint8}
NUM_SM = 132  # the H200's SMs, FA3's split count reads it
#: labels of one family that differ only in a size: one is staged, the census's type when the step has it, else the most common
FAMILIES = {"SparseMlaSplitPart_T": "SparseMlaSplitPart_T64", "SparseMlaCombine_N": "SparseMlaCombine_N5"}


def multimem_trees(world: int) -> str:
    """TREES of `AllReduceMultimem_v1` at WORLD ranks, one per owner: the tree read off the H200 switch at 8, else a chain."""
    from verity.ml.definitions.registry import tp_allreduce as AR
    tree = AR.GLM53_H200_TREE if world == 8 else AR.chain(world)
    return json.dumps([tree] * world, separators=(",", ":"))


class Capture:
    """One step of a capture directory, its commitment checked on load."""

    def __init__(self, d: Path, step: int) -> None:
        self.d, self.step = d, step
        self.ranks = sorted(int(p.name[4:]) for p in d.glob("rank*") if (p / f"step{step}" / "meta.json").exists())
        if not self.ranks:
            raise SystemExit(f"{d}: no rank recorded step {step}")
        self.meta, self.commit, self.events, self.params = {}, {}, {}, {}
        for r in self.ranks:
            sd = d / f"rank{r}" / f"step{step}"
            ops = (sd / "ops.jsonl").read_bytes()
            pj = (d / f"rank{r}" / "params.json").read_bytes()
            meta = json.loads((sd / "meta.json").read_text())
            ops_sha, params_sha = hashlib.sha512(ops).hexdigest(), hashlib.sha512(pj).hexdigest()
            if (ops_sha, params_sha) != (meta["ops_sha512"], meta["params_sha512"]):
                raise SystemExit(f"rank {r}: ops.jsonl or params.json is not what meta.json committed")
            root = hashlib.sha512(f"verity/spot/rank-root\0{ops_sha}\0{params_sha}".encode()).hexdigest()
            if root != meta["rank_root"]:
                raise SystemExit(f"rank {r}: rank root {meta['rank_root'][:16]} is not SHA-512 of its files")
            self.meta[r], self.commit[r] = meta, json.loads((sd / "commit.json").read_text())
            self.events[r] = [json.loads(x) for x in ops.decode().splitlines() if x]
            self.params[r] = json.loads(pj)
        c0 = self.commit[self.ranks[0]]
        for r in self.ranks:
            c = self.commit[r]
            if (c["root"], c["coins"], c["rank_roots"]) != (c0["root"], c0["coins"], c0["rank_roots"]):
                raise SystemExit(f"rank {r}: commit.json disagrees with rank {self.ranks[0]}'s")
            if c["rank_roots"][r] != self.meta[r]["rank_root"]:
                raise SystemExit(f"rank {r}: the step root's leaf is not its rank root")
        if hashlib.sha512(("verity/spot/step-root\0" + "\0".join(c0["rank_roots"])).encode()).hexdigest() != c0["root"]:
            raise SystemExit("the step root is not SHA-512 over the ranks' roots")
        self.root, self.coins = c0["root"], bytes.fromhex(c0["coins"])
        self.layers = sorted(x for v in c0["layers"].values() for x in v)
        self._params: dict = {}

    def tensor(self, r: int, ref: dict) -> np.ndarray:
        """A recorded tensor's words (its logical order), checked against its SHA-512."""
        if "param" in ref:
            return self.param_view(r, ref)
        if "sha512" not in ref:
            raise LookupError(f"not recorded with its bytes: {ref}")
        h = ref["sha512"]
        raw = (self.d / f"rank{r}" / "blobs" / h[:2] / h).read_bytes()
        if hashlib.sha512(raw).hexdigest() != h:
            raise SystemExit(f"rank {r}: blob {h[:16]} is not its name")
        return np.frombuffer(raw, dtype=NP[ref["dtype"]]).reshape(ref["shape"])

    def param(self, r: int, name: str) -> np.ndarray:
        if (r, name) not in self._params:
            od = self.d / f"rank{r}" / f"step{self.step}" / "open"
            meta = self.commit[r]["opened"].get(name)
            if meta is None:
                raise LookupError(f"parameter {name} is not in an opened layer")
            raw = (od / f"{name}.bin").read_bytes()
            h = hashlib.sha512(raw).hexdigest()
            if h != meta["sha512"] or (name in self.params[r] and self.params[r][name]["sha512"] != h):
                raise SystemExit(f"rank {r}: opened {name} is not the committed bytes")
            self._params[(r, name)] = np.frombuffer(raw, dtype=NP[meta["dtype"]]).reshape(meta["shape"])
        return self._params[(r, name)]

    def param_view(self, r: int, ref: dict) -> np.ndarray:
        base = self.param(r, ref["param"])
        if "stride" not in ref:
            return base.reshape(ref["shape"])
        item = base.itemsize
        flat = base.reshape(-1)
        off = ref["byte_offset"] // item
        return np.lib.stride_tricks.as_strided(flat[off:], shape=ref["shape"], strides=[s * item for s in ref["stride"]])

    def opened(self, e: dict) -> bool:
        return e.get("layer") in self.layers

    def tokens(self, r: int) -> int:
        return sum(self.meta[r]["scheduled"].values())

    def real_rows(self, r: int, rows: int) -> int:
        """Of a call's ``rows``, those of the step's real tokens: the forward's padded rows (or a multiple of them, one per
        token and slot) cut to the scheduled tokens; any other count is taken whole."""
        padded = next((e["out"]["hidden"]["shape"][0] for e in reversed(self.events[r]) if e["op"] == "model_output"), None)
        n = self.tokens(r)
        if padded and rows % padded == 0:
            return n * (rows // padded)
        return min(rows, n) if rows >= n else rows


def words(a) -> np.ndarray:
    """A tensor's elements as their bit patterns, one word each."""
    a = np.ascontiguousarray(a)
    if a.dtype.kind in "fi":
        a = a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.itemsize])
    return a.astype(np.uint64).reshape(-1)


def block_fn(block: str):
    from verity_vllm.pipeline import serving_rows as SRW
    return SRW.template(block).fn


# ---- populations: each type's instances, as (label, block, key, args(), expected()) ----------------------------------------

def quant_instances(cap: Capture):
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_C::per_token_group_fp8_quant" or not cap.opened(e):
                continue
            a = e["in"]
            if not a.get("scale_ue8m0") or a.get("group_size") != 128:
                continue
            K = a["input"]["shape"][-1]
            block = f"Fp8GroupQuantUe8m0_v1{{K={K},G=128}}"
            for t in range(cap.real_rows(r, a["input"]["shape"][0])):
                yield (f"Fp8GroupQuantUe8m0_K{K}", block, (r, e["seq"], t),
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["in"]["input"])[t]],
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["out"]["output_q"])[t], cap.tensor(r, e["out"]["output_s"])[t]])


def flashinfer_linear_instances(cap: Capture):
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm" or not cap.opened(e):
                continue
            a = e["in"]
            M, K = a["input"]["shape"]
            if M >= 32 or "param" not in a["weight"]:
                continue
            N = a["weight"]["shape"][0]
            block = f"FlashInferFp8Linear_v1{{K={K},N=1}}"
            for t in range(cap.real_rows(r, M)):
                for col in range(N):
                    yield (f"FlashInferFp8Linear_K{K}", block, (r, e["seq"], t, col),
                           lambda r=r, e=e, t=t, col=col: [cap.tensor(r, e["in"]["input"])[t],
                                                           cap.tensor(r, e["in"]["weight"])[col:col + 1],
                                                           cap.tensor(r, e["in"]["weight_scale"])[col // 128:col // 128 + 1]],
                           lambda r=r, e=e, t=t, col=col: [cap.tensor(r, e["out"]["ret0"])[t, col:col + 1]])


def moe_w2_instances(cap: Capture, profile):
    from verity_vllm.program.frontend import targets as T
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "moe_launch" or not cap.opened(e) or not e["in"].get("mul_routed_weight"):
                continue
            a, cfg = e["in"], e["in"]["config"]
            E, N, K = a["B"]["shape"]
            S = a["A"]["shape"][0]
            G = a.get("block_shape", [128, 128])[1]
            swap = bool(cfg.get("SWAP_AB", False))
            spec = T.moe_expert_fp8_gemm_spec(profile, E, K, 1, G, cfg["BLOCK_SIZE_K"], swap, BM=cfg["BLOCK_SIZE_M"],
                                              BN=cfg["BLOCK_SIZE_N"], WARPS=cfg.get("num_warps", 4), routed=True)
            topk = a["topk_weights"]["shape"][1]
            dot = re.search(r"DOT=(\w+?)Dot", spec.id)
            label = f"MoeExpertFp8GemmW_K{K}_{'swap' if swap else 'noswap'}_{dot[1] if dot else 'dot'}"
            for j in range(cap.real_rows(r, S)):
                for col in range(N):
                    yield (label, spec, (r, e["seq"], j, col),
                           lambda r=r, e=e, j=j, col=col: moe_args(cap, r, e, j, col),
                           lambda r=r, e=e, j=j, col=col, topk=topk: [cap.tensor(r, e["out"]["C"]).reshape(-1, topk, N)[j // topk, j % topk, col:col + 1]])


def moe_args(cap: Capture, r: int, e: dict, j: int, col: int) -> list:
    a = e["in"]
    A, As = cap.tensor(r, a["A"]), cap.tensor(r, a["A_scale"])
    B, Bs = cap.tensor(r, a["B"]), cap.tensor(r, a["B_scale"])
    ids = cap.tensor(r, a["expert_ids"]).reshape(-1)
    if a.get("sorted_token_ids") is None:
        ex = int(ids[j])
    else:
        sorted_ids = cap.tensor(r, a["sorted_token_ids"]).reshape(-1)
        p = int(np.nonzero(sorted_ids == j)[0][0])
        ex = int(ids[p // a["config"]["BLOCK_SIZE_M"]])
    g = cap.tensor(r, a["topk_weights"]).reshape(-1)[j]
    return [A[j], As[j], np.array([ex], np.uint64), np.array([g]), B[:, col:col + 1, :], Bs[:, col // 128:col // 128 + 1, :]]


def aligned(cap: Capture, op: str):
    """The calls of ``op`` in the opened layers, the i-th of each rank together (TP ranks run the same calls in order)."""
    per = {r: [e for e in cap.events[r] if e["op"] == op and cap.opened(e)] for r in cap.ranks}
    n = min(len(v) for v in per.values())
    for i in range(n):
        yield {r: per[r][i] for r in cap.ranks}


def fused_ar_norm_instances(cap: Capture):
    from verity.ml.definitions.registry.tp_allreduce import mnnvl_shot
    W = len(cap.ranks)
    for calls in aligned(cap, "vllm::flashinfer_trtllm_fused_allreduce_norm"):
        e0 = calls[cap.ranks[0]]
        T, H = e0["in"]["allreduce_in"]["shape"]
        half = H // 2
        shot = mnnvl_shot(T, H, W)
        for t in range(cap.real_rows(cap.ranks[0], T)):
            for c in range(2):
                sl = slice(c * half, (c + 1) * half)
                yield (f"AllReduceResidualMnnvl_W{W}_S{shot}", f"AllReduceResidualMnnvl_v1{{WORLD={W},N={half},SHOT={shot}}}",
                       (e0["seq"], t, c),
                       lambda calls=calls, t=t, sl=sl: [np.stack([cap.tensor(r, calls[r]["in"]["allreduce_in"])[t, sl] for r in cap.ranks]),
                                                        cap.tensor(cap.ranks[0], calls[cap.ranks[0]]["in"]["residual"])[t, sl]],
                       lambda calls=calls, t=t, sl=sl: [cap.tensor(cap.ranks[0], calls[cap.ranks[0]]["out"]["residual"])[t, sl]])
            eps = e0["in"]["rms_eps"]
            yield (f"RmsNormMnnvl_W{W}_S{shot}", f'RmsNormMnnvl_v1{{N={H},SHOT={shot},CLUSTER={W},EPS={{"f64":"{float(eps).hex()}"}},FMA=true}}',
                   (e0["seq"], t),
                   lambda calls=calls, t=t: [cap.tensor(cap.ranks[0], calls[cap.ranks[0]]["out"]["residual"])[t],
                                             cap.tensor(cap.ranks[0], calls[cap.ranks[0]]["in"]["rms_gamma"])],
                   lambda calls=calls, t=t: [cap.tensor(cap.ranks[0], calls[cap.ranks[0]]["out"]["norm_out"])[t]])


def allreduce_instances(cap: Capture):
    W = len(cap.ranks)
    for calls in aligned(cap, "vllm::all_reduce"):
        e0 = calls[cap.ranks[0]]
        shape = e0["in"]["tensor"]["shape"]
        if len(shape) != 2:
            continue
        T, H = shape
        half = H // 2
        trees = multimem_trees(W)
        for t in range(cap.real_rows(cap.ranks[0], T)):
            for c in range(2):
                sl = slice(c * half, (c + 1) * half)
                yield (f"AllReduceMultimem_W{W}", f"AllReduceMultimem_v1{{WORLD={W},SIZE={half},TREES={trees},FTZ=false}}", (e0["seq"], t, c),
                       lambda calls=calls, t=t, sl=sl: [np.stack([cap.tensor(r, calls[r]["in"]["tensor"])[t, sl] for r in cap.ranks])],
                       lambda calls=calls, t=t, sl=sl: [cap.tensor(cap.ranks[0], calls[cap.ranks[0]]["out"]["ret0"])[t, sl]])


def fused_add_rms_norm_instances(cap: Capture):
    from verity.ml.definitions.registry.cuda_rmsnorm import launch_block
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_C::fused_add_rms_norm" or not cap.opened(e):
                continue
            a = e["in"]
            T, N = a["input"]["shape"]
            eps = float(a["epsilon"]).hex()
            block = f'RMSNormFusedCudaVector8_v1{{N={N},EPS={{"f64":"{eps}"}},BLOCK={launch_block(T, N, fused=True)}}}'
            for t in range(cap.real_rows(r, T)):
                yield (f"RMSNormFusedCudaVector8_N{N}", block, (r, e["seq"], t),
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["in"]["input"])[t], cap.tensor(r, e["in"]["residual"])[t],
                                              cap.tensor(r, e["in"]["weight"])],
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["out"]["input"])[t], cap.tensor(r, e["out"]["residual"])[t]])


def fa3_calls(cap: Capture):
    """FA3's sparse MLA calls in the opened layers: (rank, the op's event, the tap's event or None, each token's valid keys and
    split count)."""
    from verity.ml.definitions.registry import sparse_mla_split as SP
    for r in cap.ranks:
        last = None
        for e in cap.events[r]:
            if e["op"] == "_vllm_fa3_C::fwd" and e["in"].get("q_v") is not None:
                last = e
                if not cap.opened(e):
                    continue
                o = e["out"]
                if "sha512" not in (o.get("ret2") or {}) or "sha512" not in (o.get("ret3") or {}):
                    continue
                tap = None
                for x in cap.events[r][cap.events[r].index(e) + 1:]:
                    if x["op"] == "fa3_sparse_mla":
                        tap = x
                        break
                    if x["op"] == "_vllm_fa3_C::fwd":
                        break
                T = e["in"]["q"]["shape"][0]
                cu = cap.tensor(r, e["in"]["cu_seqlens_q"]).astype(np.int64)
                if list(cu) != list(range(T + 1)):
                    raise LookupError(f"rank {r} seq {e['seq']}: cu_seqlens_q is not one query per sequence")
                valid = [int(x) for x in cap.tensor(r, e["in"]["seqused_k"])]
                bound = o["ret2"]["shape"][0]
                yield r, e, tap, valid, SP.dynamic_num_splits(valid, NUM_SM, bound)


def mla_combine_instances(cap: Capture):
    for r, e, _, valid, nss in fa3_calls(cap):
        bound, H, T, DV = e["out"]["ret2"]["shape"]
        for t in range(min(cap.tokens(r), T)):
            from verity.ml.definitions.registry import sparse_mla_split as SP
            ns = len(SP.split_blocks(valid[t], nss[t]))
            if ns < 2:
                continue
            for h in range(H):
                yield (f"SparseMlaCombine_N{ns}", f'SparseMlaCombine_v1{{NSPLIT={ns},DV={DV},INV={{"fn":"Fa3InvSum_v1"}}}}',
                       (r, e["seq"], t, h),
                       lambda r=r, e=e, t=t, h=h, ns=ns: [cap.tensor(r, e["out"]["ret3"])[:ns, h, t],
                                                          cap.tensor(r, e["out"]["ret2"])[:ns, h, t, :]],
                       lambda r=r, e=e, t=t, h=h: [cap.tensor(r, e["out"]["ret0"])[t, h, :]])


def mla_split_part_instances(cap: Capture):
    """Each split of a token whose KV rows the tap kept (its first ``kv_gathered_tokens``), per head: its keys' rows in, FA3's
    (o_partial, lse) for that split out."""
    from verity.ml.definitions.registry import sparse_mla_split as SP
    for r, e, tap, valid, nss in fa3_calls(cap):
        if tap is None or "sha512" not in (tap["in"].get("kv_gathered") or {}):
            continue
        bound, H, T, DV = e["out"]["ret2"]["shape"]
        DR = e["in"]["q"]["shape"][-1]
        BN = SP.BLOCK_N
        scale = float(tap["in"]["softmax_scale"]).hex()
        for t in range(min(cap.tokens(r), T, int(tap["in"]["kv_gathered_tokens"]))):
            splits = SP.split_blocks(valid[t], nss[t])
            if len(splits) < 2:
                continue
            for s, (b0, b1) in enumerate(splits):
                lo, hi = b0 * BN, min(b1 * BN, valid[t])
                block = (f'SparseMlaSplitPart_v1{{T={hi - lo},DR={DR},DV={DV},BN={BN},DOT={{"fn":"HopperBF16WgmmaDot16_v1"}},'
                         f'INV={{"fn":"Fa3InvSum_v1"}},SCALE={{"f64":"{scale}"}}}}')
                for h in range(H):
                    yield (f"SparseMlaSplitPart_T{hi - lo}", block, (r, e["seq"], t, h, s),
                           lambda r=r, e=e, tap=tap, t=t, h=h, lo=lo, hi=hi: [
                               cap.tensor(r, tap["in"]["q_v"])[t, h], cap.tensor(r, tap["in"]["q"])[t, h],
                               cap.tensor(r, tap["in"]["kv_gathered"])[t, lo:hi]],
                           lambda r=r, e=e, t=t, h=h, s=s: [cap.tensor(r, e["out"]["ret2"])[s, h, t, :],
                                                            cap.tensor(r, e["out"]["ret3"])[s, h, t:t + 1]])


# ---- draw, gate, stage ---------------------------------------------------------------------------------------------------

def draw(coins: bytes, label: str, n: int, k: int) -> list[int]:
    """``k`` distinct indices of ``n`` from the capture's coins (all of them when ``n`` <= ``k``)."""
    if n <= k:
        return list(range(n))
    out: list[int] = []
    i = 0
    while len(out) < k:
        j = int.from_bytes(hashlib.sha512(coins + f"stage/{label}/{i}".encode()).digest(), "big") % n
        if j not in out:
            out.append(j)
        i += 1
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--types", default=None, help="comma-separated labels (default: every type found)")
    ap.add_argument("--cc", default="9.0", help="the serving GPUs' compute capability (the MoE launches' e4m3 step)")
    ap.add_argument("--full-gates", type=int, default=4_000_000)
    ap.add_argument("--of", default="", help="what the window is (window.json)")
    ap.add_argument("--link", default="", help="how its values link to serving (window.json)")
    ap.add_argument("--row", default="spot-glm53")
    a = ap.parse_args()
    import a11
    import a11_kinds as K
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity_vllm.program.frontend.target_profile import TargetProfile

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap = Capture(Path(a.capture), a.step)
    profile = TargetProfile(compute_capability=tuple(map(int, a.cc.split("."))), batch_invariant=False)
    report: dict = {"capture": a.capture, "step": a.step, "ranks": cap.ranks, "root": cap.root, "coins": cap.coins.hex(),
                    "layers": cap.layers, "checks": {r: cap.meta[r]["check"] for r in cap.ranks}, "types": {}}
    pops: dict[str, list] = {}
    sources = [quant_instances(cap), flashinfer_linear_instances(cap), moe_w2_instances(cap, profile),
               fused_ar_norm_instances(cap), allreduce_instances(cap), fused_add_rms_norm_instances(cap),
               mla_combine_instances(cap), mla_split_part_instances(cap)]
    for src in sources:
        try:
            for inst in src:
                pops.setdefault(inst[0], []).append(inst)
        except Exception as ex:  # noqa: BLE001
            report.setdefault("population_errors", []).append(f"{type(ex).__name__}: {ex}"[:400])
    for family, prefer in FAMILIES.items():
        members = sorted((k for k in pops if k.startswith(family)), key=lambda k: (k != prefer, -len(pops[k]), k))
        for k in members[1:]:
            report["types"][k] = {"population": len(pops.pop(k)), "staged": False,
                                  "why": f"one Definition of the family {family} is staged: {members[0]}"}
    wanted = set(a.types.split(",")) if a.types else None
    staged, rows_of_kind, kinds = {}, {}, {}
    for label, insts in sorted(pops.items()):
        if wanted and label not in wanted:
            continue
        rec: dict = {"population": len(insts)}
        report["types"][label] = rec
        try:
            spec = insts[0][1] if not isinstance(insts[0][1], str) else block_fn(insts[0][1])
        except Exception as ex:  # noqa: BLE001
            rec.update(staged=False, why=f"no Definition binds: {type(ex).__name__}: {str(ex)[:300]}")
            continue
        rec["unit"] = spec.id
        rows_spec = K.rows_of(spec)
        if rows_spec is None:
            rec.update(staged=False, why=f"{spec.id}: the parameters' bits are not whole u16 words")
            continue
        picks = draw(cap.coins, label, len(insts), a.per_type)
        rec["drawn"] = [list(insts[i][2]) for i in picks]
        ins, outs, gate = [], [], []
        for i in picks:
            _, _, key, args_of, want_of = insts[i]
            try:
                args = [words(x) for x in args_of()]
                want = np.concatenate([words(x) for x in want_of()])
                ev = a11.evaluate_units(spec, spec, args, 1, a.full_gates)
                got = np.asarray([int(x) for x in ev["call_out"]], dtype=np.uint64)
                same = got.shape == want.shape and bool(np.array_equal(got, want))
                first = None if same else int(np.nonzero(got != want)[0][0]) if got.shape == want.shape else "shape"
                gate.append({"key": list(key), "agree": same, "words": int(want.size), "first_diff": first,
                             **({} if same or first == "shape" else {"got": int(got[first]), "want": int(want[first])})})
                if same:
                    ins += ev["in"]
                    outs += ev["out"]
            except Exception as ex:  # noqa: BLE001
                gate.append({"key": list(key), "agree": None, "error": f"{type(ex).__name__}: {str(ex)[:300]}",
                             "trace": traceback.format_exc()[-800:]})
        rec["gate"] = gate
        if not ins:
            rec.update(staged=False, why="no drawn instance passed the match gate")
            continue
        rows, routs = a11.unit_rows(spec, rows_spec, ins, outs)
        rows_of_kind[label] = {"unit": spec, "rows": rows, "outs": routs}
        n = len(ins)
        agree = [g["agree"] for g in gate]
        staged[label] = {"staged": True, "unit": spec.id, "n": n, "name": label, "step": a.step,
                         "event": rec["drawn"][0], "calls": rec["drawn"],
                         "twin": {"agree": all(x is True for x in agree), "calls": len(agree), "compared": sum(g.get("words", 0) for g in gate)}}
        kinds[label] = {"gates": spec.gates, "outcome": "lowers"}
        rec.update(staged=True, n=n)
        print(f"stage {label}: {n}/{len(picks)} drawn of {len(insts)} pass the gate ({spec.id[:80]})", flush=True)
    for label, rec in report["types"].items():
        if not rec.get("staged"):
            print(f"stage {label}: not staged: {rec.get('why')}", flush=True)
    units = {kv["unit"].id: kv["unit"] for kv in rows_of_kind.values()}
    if units:
        (out / "units.json").write_bytes(canonical_json(encode_program(P.mixed_population_program([(u, 1) for u in units.values()]))))
    for f, kv in rows_of_kind.items():
        np.savez(out / f"{f}.npz", **{f"i.{r}": np.asarray(v, dtype=np.uint16) for r, v in kv["rows"].items()},
                 **{f"o.{o}": np.asarray(v, dtype=np.uint16) for o, v in kv["outs"].items()})
    weights = {"from": "the parameters each rank held while serving, written by the capture for the opened layers and held to "
                       "their committed SHA-512s", "layers": cap.layers}
    (out / "values.json").write_text(json.dumps({"staged": staged, "weights": weights, "values": a.capture, "steps": str(a.step),
                                                 "layers": ",".join(map(str, cap.layers)), "per_kind": a.per_type},
                                                indent=1, sort_keys=True, default=str))
    (out / "kinds.json").write_text(json.dumps({"kinds": kinds}, indent=1, sort_keys=True))
    (out / "window.json").write_text(json.dumps({"of": a.of, "link": a.link, "row": a.row, "root": cap.root}, indent=1))
    report["seconds"] = round(time.time() - t0, 1)
    (out / "stage.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    return 0 if staged else 1


if __name__ == "__main__":
    raise SystemExit(main())
