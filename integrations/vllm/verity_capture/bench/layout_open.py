#!/usr/bin/env python3
"""R10 lane `layout` (owner 3): openings of K=576 GEMM-coordinate operands (and the SiLU·mul scalar relation) against the retained run
commitment, in owner 6's witness format (`out/gen/r10/integ/witness_format.md`, schema `veritor/r10/e2e-witness/v1`), under
L1-chunk256 (the serving engine's tree) or L2-row (1,152-B row leaves, tensors row-aligned; built in the same run).

Library (used live by `layout_capture.py` on the pod, GPU-resident levels) and CLI over a dump written by `layout_capture.py`
(host recomputation of the levels from the retained arena bytes -- which is also the root cross-check):

    python verity_capture/bench/layout_open.py --dump /workspace/cp/r10/layout/run_l1l2 --layout L1-chunk256 --batch dispersed --B 64 --out w.json

Position rules (spec §5a/§5b / witness_format §2):
    step tensor T, bf16 flat word f   -> arena byte b = stream_off_T + 2 f -> rank = b // chunk, word_index = (b % chunk) // 2
    weight [N, K] bf16 row j word k   -> f = K j + k                        -> rank = 2 f // chunk, word_index = f % (chunk // 2)
Counters per batch: distinct leaves, distinct path nodes (multiproof: siblings not derivable from opened leaves), witness bytes (naive
per-leaf and multiproof), guest SHA-256 compressions (calculated), opening wall time (measured; lazy per-leaf as CMT-3 serves it, and a
level-batched gather).
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import layout_verify as LV  # noqa: E402  (hashlib-only rules)

K = 576
CALLS = {"qkv_proj": ("self_attn.qkv_proj", 960), "o_proj": ("self_attn.o_proj", 576), "gate_up_proj": ("mlp.gate_up_proj", 3072)}
# challenge derivation: docs/protocol/challenge-v1.md (verity_protocol.challenge); the R10 tag is retired
LAYOUTS = {"L1-chunk256": 256, "L2-row": 1152}


def framing_of(layout_id: str, step: int) -> str:
    return f"cmt-chunk-fa2h-v7.s{step}" if layout_id == "L1-chunk256" else f"cmt-chunk-fa2h-v7.row1152.s{step}"


# ---------------------------------------------------------------- the retained run (prover side)
@dataclass
class StepView:
    step: int
    root: bytes
    merkle_root: bytes
    n_leaves: int
    layout: list                      # [[ordinal, name, dtype, shape, nbytes, chunk, first_leaf, n_leaves_t, stream_off], ...]
    layout_digest: bytes
    used: int
    bytes_: Any                       # torch uint8 device tensor (live) or bytes (dump)
    levels: Any                       # list of torch int32 [n,8] (live) or list[list[bytes]] (dump)
    by_name: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_name = {row[1]: row for row in self.layout}


@dataclass
class WeightView:
    name: str
    dtype: str
    shape: list
    nbytes: int
    subtree_root: bytes
    n_leaves: int
    raw: Any                          # bytes / memoryview / numpy uint8
    levels: Any                       # host levels (list[list[bytes]])
    index: int                        # position in the names list (outer tree leaf index)


@dataclass
class RetainedRun:
    layout_id: str
    chunk: int
    context: dict
    steps: list[StepView]
    run_root: bytes
    weights: list[WeightView]
    weights_root: bytes
    names_digest: bytes
    outer_levels: Any                 # fold levels over subtree roots
    live: bool                        # True: torch tensors on the GPU (pod); False: host bytes / levels (dump)
    l0_words_per_step: dict = field(default_factory=dict)

    def step(self, s: int) -> StepView:
        return self.steps[s]

    def weight(self, name: str) -> WeightView:
        for w in self.weights:
            if w.name == name:
                return w
        raise KeyError(name)


# ---------------------------------------------------------------- leaf / path access (live GPU or host)
def _leaf_bytes(sv: StepView, rank: int, chunk: int) -> bytes:
    lo = rank * chunk
    if isinstance(sv.bytes_, (bytes, bytearray, memoryview)):
        b = bytes(sv.bytes_[lo:lo + chunk])
    else:
        b = bytes(sv.bytes_[lo:lo + chunk].cpu().numpy().tobytes())
    return b + bytes(chunk - len(b))


def _path(levels: Any, rank: int, live: bool) -> list[bytes | None]:
    if live:
        path, cur = [], rank
        for lv in levels[:-1]:
            n = lv.shape[0]
            path.append(None if (cur == n - 1 and n % 2 == 1) else lv[cur ^ 1].cpu().numpy().astype("uint32").tobytes())
            cur //= 2
        return path
    return LV.path_in_levels(levels, rank)


def multiproof(n_leaves: int, ranks: set[int]) -> dict:
    """Distinct sibling nodes a verifier needs for the union of paths, and the internal hashes it computes."""
    frontier, width, needed, internal = set(ranks), n_leaves, 0, 0
    while width > 1:
        nxt = set()
        for i in frontier:
            if not (i == width - 1 and width % 2 == 1) and (i ^ 1) not in frontier:
                needed += 1
            nxt.add(i // 2)
        frontier = nxt
        width = (width + 1) // 2
        internal += len(frontier)
    return {"needed_siblings": needed, "internal_hashes": internal}


# ---------------------------------------------------------------- coordinates -> logical positions
def x_tensor_name(layer: int, call: str) -> str:
    if call == "qkv_proj":
        return "model.layers.0.input_layernorm" if layer == 0 else f"model.layers.{layer}.input_layernorm/0"
    if call == "gate_up_proj":
        return f"model.layers.{layer}.post_attention_layernorm/0"
    if call == "o_proj":
        return f"model.layers.{layer}.self_attn.attn"
    raise ValueError(call)


def y_tensor_name(layer: int, call: str) -> str:
    return f"model.layers.{layer}.{CALLS[call][0]}/0"


def w_tensor_name(layer: int, call: str) -> str:
    return f"model.layers.{layer}.{CALLS[call][0]}.weight"


def tokens_in_step(sv: StepView, layer: int = 0, call: str = "qkv_proj") -> int:
    row = sv.by_name[y_tensor_name(layer, call)]
    shape = row[3]
    return int(shape[0]) if len(shape) == 2 else int(shape[-2])


def population(run: RetainedRun, modules: tuple[str, ...] = tuple(CALLS)) -> list[tuple[int, int, str, int, int]]:
    """[(step, layer, call, T_step, N_call)] enumerated (step, layer, module, token, j) lexicographically -- owner 6's population order."""
    pop = []
    n_layers = max(int(r[1].split(".")[2]) for sv in run.steps for r in sv.layout if r[1].startswith("model.layers.")) + 1
    for sv in run.steps:
        T = tokens_in_step(sv)
        for layer in range(n_layers):
            for call in modules:
                pop.append((sv.step, layer, call, T, CALLS[call][1]))
    return pop


def challenge_instance(run: RetainedRun, B: int, population_size: int, extra_commitments: list[tuple[str, bytes, bytes]] | None = None):
    """The challenge-v1 instance of this run (docs/protocol/challenge-v1.md §1): roles ``weights-l1``, ``run-values-l1`` (both layout L1), plus
    any further co-committed roles (e.g. ``("run-values-l2", layout_digest_l2, root_l2)``) appended in configuration order."""
    from verity_protocol import challenge as chal_v1

    inst = chal_v1.l1_instance(bytes.fromhex(run.context["program_digest"]), bytes.fromhex(run.context["query_id"]), population_size, B,
                               run.weights_root, run.run_root)
    extra = tuple(chal_v1.Commitment(role, layout, root) for role, layout, root in (extra_commitments or []))
    return chal_v1.Instance(inst.program_digest, inst.query_digest, inst.sampler_spec, inst.commitments + extra)


def challenge_value(run: RetainedRun, B: int, rho: bytes, population_size: int, extra_commitments: list[tuple[str, bytes, bytes]] | None = None) -> bytes:
    """challenge-v1 seed = SHA-256("verity/challenge/v1" || rho || Encode(instance)) -- replaces the R10 owner-3 co-commit rule."""
    from verity_protocol import challenge as chal_v1

    return chal_v1.challenge_seed(rho, challenge_instance(run, B, population_size, extra_commitments))


def _stream_u64(seed: bytes):
    ctr = 0
    while True:
        d = hashlib.sha256(seed + ctr.to_bytes(8, "big")).digest()
        ctr += 1
        for i in range(0, 32, 8):
            yield int.from_bytes(d[i:i + 8], "big")


def sample_coords(run: RetainedRun, seed: bytes, B: int, pattern: str, modules: tuple[str, ...] = tuple(CALLS)) -> list[dict]:
    pop = population(run, modules)
    sizes = [T * N for (_s, _l, _c, T, N) in pop]
    total = sum(sizes)
    g = _stream_u64(seed)
    out: list[dict] = []
    if pattern == "dispersed":
        # challenge-v1 §2.2 (`dispersed-v1`): idx_k = int_le(SHA-256(seed || u64le(k))) mod N_pop in the lexicographic
        # (step, layer, module, token, j) enumeration; duplicates are kept (they are what the rule yields)
        from verity_protocol.challenge import derive_indices

        for gidx in derive_indices(seed, total, B):
            acc = 0
            for (s, l, c, T, N), sz in zip(pop, sizes, strict=True):
                if gidx < acc + sz:
                    t, j = divmod(gidx - acc, N)
                    out.append({"step": s, "layer": l, "call": c, "token": t, "j": j, "rank_in_population": gidx})
                    break
                acc += sz
        return out
    if pattern == "shared_x":
        # one instance (step, layer, call, token) drawn uniformly over instances; B distinct output indices j of that call (same_instance)
        inst = [(s, l, c, t) for (s, l, c, T, N) in pop for t in range(T)]
        s, l, c, t = inst[next(g) % len(inst)]
        N = CALLS[c][1]
        js: list[int] = []
        while len(js) < min(B, N):
            j = next(g) % N
            if j not in js:
                js.append(j)
        return [{"step": s, "layer": l, "call": c, "token": t, "j": j} for j in js]
    if pattern == "shared_w":
        # one weight row (layer, call, j) drawn uniformly over rows; B distinct (step, token) instances of that call (same W row, x rows differ)
        rows = [(l, c) for (s, l, c, T, N) in pop if s == pop[0][0]]
        l, c = rows[next(g) % len(rows)]
        N = CALLS[c][1]
        j = next(g) % N
        inst = [(s, t) for (s, ll, cc, T, _N) in pop if (ll, cc) == (l, c) for t in range(T)]
        picks: list[tuple[int, int]] = []
        while len(picks) < min(B, len(inst)):
            st = inst[next(g) % len(inst)]
            if st not in picks:
                picks.append(st)
        return [{"step": s, "layer": l, "call": c, "token": t, "j": j} for (s, t) in picks]
    if pattern == "scalar_dispersed":
        # SiLU·mul element (one-gate scalar relation): (step, layer, token, i) uniform
        inst = [(s, l, T) for (s, l, c, T, N) in pop if c == "gate_up_proj"]
        tot = sum(T * 1536 for (_s, _l, T) in inst)
        seen = set()
        while len(out) < B:
            gidx = next(g) % tot
            acc = 0
            for s, l, T in inst:
                if gidx < acc + T * 1536:
                    t, i = divmod(gidx - acc, 1536)
                    if (s, l, t, i) not in seen:
                        seen.add((s, l, t, i))
                        out.append({"step": s, "layer": l, "token": t, "i": i})
                    break
                acc += T * 1536
        return out
    raise ValueError(pattern)


# ---------------------------------------------------------------- openings
class Opener:
    def __init__(self, run: RetainedRun) -> None:
        self.run = run
        self.chunk = run.chunk
        self.trees: dict[str, dict] = {}          # tree id -> {framing, root, count, owner_plus_two, domain_id, leaves: {rank: leaf}}
        self.t_open = 0.0

    def _tree(self, tid: str) -> dict:
        t = self.trees.get(tid)
        if t is None:
            if tid.startswith("step:"):
                sv = self.run.step(int(tid[5:]))
                t = {"id": tid, "framing": framing_of(self.run.layout_id, sv.step), "root": sv.merkle_root.hex(), "count": sv.n_leaves,
                     "owner_plus_two": 3, "domain_id": sv.layout_digest.hex(), "leaves": {}}
            else:
                w = self.run.weight(tid[8:])
                t = {"id": tid, "framing": "cmt-posleaf-v0", "root": w.subtree_root.hex(), "count": w.n_leaves, "owner_plus_two": 2,
                     "domain_id": hashlib.sha256(b"verity/r10/weights-tensor/v1" + w.name.encode()).hexdigest(), "leaves": {}}
            self.trees[tid] = t
        return t

    def open_rank(self, tid: str, rank: int) -> None:
        t = self._tree(tid)
        if rank in t["leaves"]:
            return
        t0 = time.perf_counter()
        if tid.startswith("step:"):
            sv = self.run.step(int(tid[5:]))
            value = _leaf_bytes(sv, rank, self.chunk)
            path = _path(sv.levels, rank, self.run.live)
        else:
            w = self.run.weight(tid[8:])
            value = bytes(w.raw[rank * self.chunk:(rank + 1) * self.chunk])       # last chunk of a tensor may be short (pos_leaf binds length)
            path = LV.path_in_levels(w.levels, rank)
        self.t_open += time.perf_counter() - t0
        t["leaves"][rank] = {"rank": rank, "bytes_hex": value.hex(), "path": [{"kind": "sibling", "digest": p.hex()} if p is not None else {"kind": "lift"} for p in path]}

    def open_step_words(self, step: int, tensor: str, flat_word0: int, n: int) -> dict:
        sv = self.run.step(step)
        off = sv.by_name[tensor][8]
        b0 = off + 2 * flat_word0
        ranks = sorted({(b0 + 2 * k) // self.chunk for k in range(n)})
        for r in ranks:
            self.open_rank(f"step:{step}", r)
        return {"tree": f"step:{step}", "tensor": tensor, "flat_word0": flat_word0, "ranks": ranks, "word_index0": (b0 % self.chunk) // 2}

    def open_weight_words(self, tensor: str, flat_word0: int, n: int) -> dict:
        ranks = sorted({(2 * (flat_word0 + k)) // self.chunk for k in range(n)})
        for r in ranks:
            self.open_rank(f"weights:{tensor}", r)
        return {"tree": f"weights:{tensor}", "flat_word0": flat_word0, "ranks": ranks, "word_index0": flat_word0 % (self.chunk // 2)}

    def word(self, tid: str, rank: int, widx: int) -> bytes:
        v = bytes.fromhex(self.trees[tid]["leaves"][rank]["bytes_hex"])
        return v[2 * widx:2 * widx + 2]

    def gemm_obligation(self, k: int, c: dict) -> dict:
        s, l, call, t, j = c["step"], c["layer"], c["call"], c["token"], c["j"]
        N = CALLS[call][1]
        x = self.open_step_words(s, x_tensor_name(l, call), K * t, K)
        W = self.open_weight_words(w_tensor_name(l, call), K * j, K)
        yname = y_tensor_name(l, call)
        yo = self.open_step_words(s, yname, N * t + j, 1)
        yv = self.word(yo["tree"], yo["ranks"][0], yo["word_index0"])
        y_word = int.from_bytes(yv, "little")      # the u16 bf16 bit pattern; one convention, no hex strings
        y = {"tree": yo["tree"], "tensor": yname, "flat_word": N * t + j, "rank": yo["ranks"][0], "word_index": yo["word_index0"], "y_word": y_word}
        return {"k": k, "logical": {"step": s, "layer": l, "module": f"model.layers.{l}.{CALLS[call][0]}", "call": call, "token": t, "j": j},
                "x": x, "W": W, "y": y, "expected_y": y_word}

    def scalar_obligation(self, k: int, c: dict) -> dict:
        """SiLU·mul element: out[t,i] = bf16(silu(gate[t,i]) * up[t,i]); gate/up words of gate_up_proj/0, out word of mlp.act_fn."""
        s, l, t, i = c["step"], c["layer"], c["token"], c["i"]
        gu = f"model.layers.{l}.mlp.gate_up_proj/0"
        act = f"model.layers.{l}.mlp.act_fn"
        a = self.open_step_words(s, gu, 3072 * t + i, 1)
        b = self.open_step_words(s, gu, 3072 * t + 1536 + i, 1)
        o = self.open_step_words(s, act, 1536 * t + i, 1)
        ov = self.word(o["tree"], o["ranks"][0], o["word_index0"])
        return {"k": k, "logical": {"step": s, "layer": l, "module": f"model.layers.{l}.mlp.act_fn", "token": t, "i": i},
                "inputs": [a, b], "y": {"tree": o["tree"], "tensor": act, "flat_word": 1536 * t + i, "rank": o["ranks"][0], "word_index": o["word_index0"],
                                        "y_word": int.from_bytes(ov, "little")},
                "expected_y": int.from_bytes(ov, "little")}

    def counters(self) -> dict:
        leaves = nodes_naive = wb_naive = 0
        needed = internal = 0
        leaf_bytes = 0
        for tid, t in self.trees.items():
            ranks = set(t["leaves"])
            leaves += len(ranks)
            for lf in t["leaves"].values():
                ns = sum(1 for p in lf["path"] if p["kind"] == "sibling")
                nodes_naive += ns
                wb_naive += len(lf["bytes_hex"]) // 2 + 32 * ns
                leaf_bytes += len(lf["bytes_hex"]) // 2
            mp = multiproof(t["count"], ranks)
            needed += mp["needed_siblings"]
            internal += mp["internal_hashes"]
        leaf_blocks = sum(LV.sha_blocks(64 + len(lf["bytes_hex"]) // 2) if t["framing"].startswith("cmt-chunk") else LV.sha_blocks(26 + len(lf["bytes_hex"]) // 2)
                          for t in self.trees.values() for lf in t["leaves"].values())
        return {"distinct_leaves": leaves, "path_nodes_naive": nodes_naive, "path_nodes_multiproof": needed,
                "witness_bytes_naive": wb_naive, "witness_bytes_multiproof": leaf_bytes + 32 * needed, "leaf_bytes": leaf_bytes,
                "sha_compressions_naive": leaf_blocks + 2 * nodes_naive, "sha_compressions_multiproof": leaf_blocks + 2 * internal,
                "trees": len(self.trees), "opening_s": self.t_open}

    def sidecar(self, obligations: list[dict], relation: dict, challenge: dict, batch: dict) -> dict:
        run = self.run
        return {"schema": "veritor/r10/e2e-witness/v1", "layout_id": run.layout_id, "context": run.context,
                "roots": {"run_root": run.run_root.hex(), "weights_root": run.weights_root.hex(), "roots_written_utc": run.context.get("roots_written_utc"),
                          "steps": [{"step": sv.step, "root": sv.root.hex(), "merkle_root": sv.merkle_root.hex(), "n_leaves": sv.n_leaves,
                                     "layout_digest": sv.layout_digest.hex(), "ctx_digest": LV.ctx_digest(run.context["run_id"], sv.step).hex(), "layout": sv.layout}
                                    for sv in run.steps],
                          "weights": {"names_digest": run.names_digest.hex(),
                                      "tensors": [{"name": w.name, "dtype": w.dtype, "shape": w.shape, "nbytes": w.nbytes, "subtree_root": w.subtree_root.hex(), "n_leaves": w.n_leaves}
                                                  for w in run.weights]}},
                "challenge": challenge, "relation": relation, "batch": batch, "obligations": obligations,
                "trees": [{**{k: v for k, v in t.items() if k != "leaves"}, "leaves": sorted(t["leaves"].values(), key=lambda x: x["rank"])} for t in self.trees.values()],
                "counters": self.counters()}


GEMM_RELATION = {"id": "R-GEMM-COORD-K576", "gate_set": "veritor.tc-ampere-bf16@2", "K": 576, "ports": 1152, "outputs": 1,
                 "semantics": "y = bf16_rne(DotBf16(x, W[j,:])): 36 chained mma.sync m16n8k16 bf16 steps, fp32 acc, ascending k (pu-cat catalogue)"}
SCALAR_RELATION = {"id": "R-SCALAR-SILUMUL-ELEM", "gate_set": "SiluMul_v1", "inputs": 2, "outputs": 1,
                   "semantics": "out[t,i] = bf16(silu(gate[t,i]) * up[t,i]) with gate/up = gate_up_proj output words (t,i)/(t,1536+i)"}


def open_batch(run: RetainedRun, coords: list[dict], kind: str, challenge: dict, batch: dict) -> dict:
    op = Opener(run)
    t0 = time.perf_counter()
    if kind == "gemm":
        obs = [op.gemm_obligation(k, c) for k, c in enumerate(coords)]
        rel = GEMM_RELATION
    else:
        obs = [op.scalar_obligation(k, c) for k, c in enumerate(coords)]
        rel = SCALAR_RELATION
    wall = time.perf_counter() - t0
    doc = op.sidecar(obs, rel, challenge, batch)
    doc["counters"]["batch_wall_s"] = wall
    return doc


def l0_counters(run: RetainedRun, doc: dict) -> dict:
    """CALCULATED counters for L0-scalar (one bf16 word per leaf over the same bytes; word-per-leaf tree per step tensor stream and per
    weight tensor): distinct words, multiproof siblings, witness bytes.  No L0 tree is built here (R9's legacy frame-v3 layout)."""
    per_tree: dict[str, set[int]] = {}
    for ob in doc["obligations"]:
        if "x" in ob:
            s = ob["logical"]["step"]
            sv = run.step(s)
            for opnd in ("x", "y"):
                o = ob[opnd]
                off = sv.by_name[o["tensor"]][8]
                f0 = o.get("flat_word0", o.get("flat_word"))
                n = K if opnd == "x" else 1
                per_tree.setdefault(f"step:{s}", set()).update((off // 2) + f0 + k for k in range(n))
            per_tree.setdefault(ob["W"]["tree"], set()).update(ob["W"]["flat_word0"] + k for k in range(K))
        else:
            s = ob["logical"]["step"]
            sv = run.step(s)
            for o in [*ob["inputs"], ob["y"]]:
                off = sv.by_name[o["tensor"]][8]
                per_tree.setdefault(f"step:{s}", set()).add(off // 2 + o.get("flat_word0", o.get("flat_word")))
    leaves = needed = internal = 0
    for tid, words in per_tree.items():
        if tid.startswith("step:"):
            n = -(-run.step(int(tid[5:])).used // 2)
        else:
            n = run.weight(tid[8:]).nbytes // 2
        mp = multiproof(n, words)
        leaves += len(words)
        needed += mp["needed_siblings"]
        internal += mp["internal_hashes"]
    return {"label": "calculated (word-per-leaf over the same bytes; leaf frame v3 = 1 SHA block)", "distinct_leaves": leaves,
            "path_nodes_multiproof": needed, "witness_bytes_multiproof": 2 * leaves + 32 * needed, "sha_compressions_multiproof": leaves + 2 * internal}


# ---------------------------------------------------------------- dump loading (host)
def load_dump(dump: Path, layout_id: str) -> RetainedRun:
    chunk = LAYOUTS[layout_id]
    meta = json.loads((dump / f"roots_{layout_id}.json").read_text())
    ctx = meta["context"]
    steps = []
    for s in meta["steps"]:
        raw = (dump / f"step_{s['step']}.bin").read_bytes()
        mr, n, levels = LV.arena_merkle_root(raw, s["step"], chunk)
        if mr.hex() != s["merkle_root"] or n != s["n_leaves"]:
            raise RuntimeError(f"step {s['step']}: recomputed root {mr.hex()[:16]} != committed {s['merkle_root'][:16]} (n {n} vs {s['n_leaves']})")
        steps.append(StepView(s["step"], bytes.fromhex(s["root"]), mr, n, s["layout"], bytes.fromhex(s["layout_digest"]), len(raw), raw, levels))
    wmeta = meta["weights"]
    weights = []
    for i, t in enumerate(wmeta["tensors"]):
        p = dump / "weights" / (t["name"] + ".bin")
        raw = p.read_bytes() if p.exists() else b""
        if raw:
            sr, n, levels = LV.tensor_subtree(raw, chunk)
            if sr.hex() != t["subtree_root"]:
                raise RuntimeError(f"weight {t['name']}: recomputed subtree != committed")
        else:
            sr, n, levels = bytes.fromhex(t["subtree_root"]), t["n_leaves"], None
        weights.append(WeightView(t["name"], t["dtype"], t["shape"], t["nbytes"], sr, n, raw, levels, i))
    outer = LV.fold_levels([w.subtree_root for w in weights])
    return RetainedRun(layout_id, chunk, ctx, steps, bytes.fromhex(meta["run_root"]), weights, bytes.fromhex(meta["weights_root"]),
                       bytes.fromhex(wmeta["names_digest"]), outer, live=False)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layout", default="L1-chunk256", choices=list(LAYOUTS))
    ap.add_argument("--batch", default="dispersed", choices=["dispersed", "shared_x", "scalar_dispersed"])
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--rho", required=True, help="challenge-v1 verifier randomness (64 hex chars = 32 raw bytes), obtained AFTER the roots")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from verity_protocol import challenge as chal_v1

    rho = bytes.fromhex(a.rho)
    run = load_dump(Path(a.dump), a.layout)
    pop = population(run, tuple(CALLS))
    population_size = sum(T * N for (_s, _l, _c, T, N) in pop)
    inst = challenge_instance(run, a.B, population_size)
    seed = chal_v1.challenge_seed(rho, inst)
    coords = sample_coords(run, seed, a.B, a.batch)
    ch = {"rule": "challenge-v1" + ("" if a.batch == "dispersed" else f" (seed) + NON-PROTOCOL pattern {a.batch}"), "spec": "docs/protocol/challenge-v1.md",
          "rho_hex": rho.hex(), "seed_hex": seed.hex(), "instance": chal_v1.instance_to_json(inst), "B": a.B, "population": population_size,
          "indices": [c["rank_in_population"] for c in coords] if a.batch == "dispersed" else None,
          "population_units": "K576 {qkv_proj, o_proj, gate_up_proj} x steps x tokens x j" if a.batch != "scalar_dispersed" else "SiluMul elements"}
    doc = open_batch(run, coords, "gemm" if a.batch != "scalar_dispersed" else "scalar", ch, {"pattern": a.batch, "B": a.B, "source": "dump/host"})
    doc["l0_calculated"] = l0_counters(run, doc)
    Path(a.out).write_text(json.dumps(doc) + "\n")
    print(json.dumps({k: v for k, v in doc["counters"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
