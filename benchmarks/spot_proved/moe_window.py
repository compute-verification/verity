"""One MoE layer of a captured step (`capture.py`) cut per expert, staged as a stated window of the one-stage audit
(`records_audit.py audit`), its tables committed as the committer registered them (`verity_vllm.pipeline.expert_tables`).

    python benchmarks/spot_proved/moe_window.py --capture DIR --step S --layer L --out DIR --key FILE
        (--tables FILE | --self-register) [--rank 0] [--tokens 1] [--cap 2048] [--law stratified:1] [--forge w13]
        [--bench SEED] [--circuit-only] [--full-gate]

* **The cut.** `expert_rows`' cut of one layer on one TP rank at T of the step's tokens, as a stated circuit (the form of
  `test_fp8_moe_per_expert._cut`): the router's slots (`MoeRouterSigmoidBiasSlots_v2`, a unit a token), each slot's
  reads of its expert's weight rows and scale row from the rank's tables (`EmbeddingRead_v1`, a unit a block of a pair,
  table-major), and the per-expert block GEMMs over the rows read (`MoeExpertFp8BlockGemm(W)_v4`, w13's units its rows
  read's, w2's slot-major). A table is `expert_rows.table_param`'s, its rows `table_words`'; a read's ``tok`` is wired
  from the router's slot id, a GEMM's ``w`` and ``ws`` from its reads and w2's ``g`` from the slot's weight. The router's
  logits and bias and each GEMM's activation and scales are the window's free inputs (in the served circuit, the blocks
  before it write them).
* **Values.** Each unit's values are the capture's recorded tensors of the opened MoE layer, every one held to the step's
  commitment as `stage.Capture` loads it: the router's logits the gate's f32 ``aten::mm`` rows and its bias the layer's
  ``e_score_correction_bias``; each pair's expert its place in the launch's ``sorted_token_ids`` and ``expert_ids``, its
  weight the launch's ``topk_weights``; each GEMM's activation, scales and output the launch's A, A_scale and C, its
  statics the launch's tile (`targets.moe_expert_fp8_launch_dot`). The T tokens are drawn by the capture's coins.
* **Tables.** Each table the window reads is committed from the capture's opened weights under ``--key`` at the program
  and nonces of the rank's registration (``--tables``, `expert_tables.committer`), and refused unless its root is the
  registered one: the window's reads are of the registered rows exactly when the served weights are the registered ones.
  ``--self-register`` commits them under ``--key`` at a program of its own and writes that registration (``tables.json``,
  `expert_tables`' format), a registration made after serving.
* **Match gate.** Before anything is committed, each pair's GEMM outputs are held to the twin of the launch
  (`fp8_moe_rows.moe_expert_fp8_block_row`) on its expert's rows, and each router unit's slots to the router's
  Definition on its logits and bias; ``--full-gate`` also evaluates every unit's template (`Window.mismatches`).
* **Arms.** ``--forge w13`` flips bit 0 of every w13 GEMM unit's output word after the gate (the audit's ``--forged``
  window, which the verifier must refuse); ``--bench SEED`` stages the same shapes, tiles and draws with benchmark
  values (random weights, activations and router inputs, the outputs their Definitions' and twins'), its tables
  self-registered: the like-for-like arm a served window's proving time is compared with.
* **Out.** ``window/`` (the records the audit reads), ``tables.json`` (the registration the verifier pins),
  ``stage.json`` (statics, tokens, units, the gate, the tables' roots against the pins, the circuit record's SHA-512 and
  the run root). ``--circuit-only`` writes no values: it prints the SHA-512 of the circuit record the verifier holds the
  window to, from the capture's public shapes and the registration's names alone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("spot_stage", HERE / "stage.py")
ST = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ST)

from verity.core.protocols.sampled_proofs.one_stage import draw as D  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as REG  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as R  # noqa: E402
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.evaluation import evaluate  # noqa: E402
from verity.ml.circuits.ops.boolean import gemm as BG  # noqa: E402
from verity.ml.commitments.gates import embedding as EMB  # noqa: E402
from verity.ml.definitions.numerics.fp8_moe_rows import TWIN_DOT_OF, moe_expert_fp8_block_row  # noqa: E402
from verity.ml.definitions.registry import boolean_fp8_moe as BM  # noqa: E402
from verity.ml.definitions.registry import boolean_moe as BMOE  # noqa: E402
from verity.ml.definitions.registry import moe as MOE  # noqa: E402
from verity.ml.windows import interface as WI  # noqa: E402

LINK = ("provenance, not proof: each unit's values are the capture's recorded tensors of the step whose root `of` names, "
        "held to its commitment when read; the tables are committed from the rank's served weights at the registration's "
        "program and nonces, their roots its pins")
_LAYER = re.compile(r"\.layers\.(\d+)\.")


# ---- the layer's values ------------------------------------------------------------------------------------------------------

def launches(events: list[dict], layer: int, E: int | None = None) -> tuple[dict, dict, dict]:
    """Layer ``layer``'s router gate (the f32-out ``aten::mm`` of E columns), w13 launch and w2 launch on one rank."""
    moe = [e for e in events if e["op"] == "moe_launch" and e.get("layer") == layer]
    w13 = [e for e in moe if not e["in"].get("mul_routed_weight")]
    w2 = [e for e in moe if e["in"].get("mul_routed_weight")]
    if len(w13) != 1 or len(w2) != 1:
        raise LookupError(f"layer {layer}: {len(w13)} w13 and {len(w2)} w2 launches recorded, not one of each")
    E = E or w13[0]["in"]["B"]["shape"][0]
    gates = [e for e in events if e["op"] == "aten::mm" and e.get("layer") == layer
             and (e["out"].get("ret0") or {}).get("dtype") == "torch.float32" and e["out"]["ret0"]["shape"][-1] == E]
    if len(gates) != 1:
        raise LookupError(f"layer {layer}: {len(gates)} f32-out aten::mm calls of {E} columns, not the router's one")
    return gates[0], w13[0], w2[0]


def experts_of(cap, r: int, e: dict, pairs: int) -> np.ndarray:
    """Each pair's expert in launch ``e``: the expert of the aligned block holding its place in ``sorted_token_ids``."""
    a = e["in"]
    ids = cap.tensor(r, a["expert_ids"]).reshape(-1).astype(np.int64)
    if a.get("sorted_token_ids") is None:
        return ids[:pairs]
    st = cap.tensor(r, a["sorted_token_ids"]).reshape(-1).astype(np.int64)
    bm, out = int(a["config"]["BLOCK_SIZE_M"]), np.full(pairs, -1, dtype=np.int64)
    at = np.flatnonzero(st < pairs)
    out[st[at]] = ids[at // bm]
    if (out < 0).any():
        raise LookupError(f"moe_launch (seq {e['seq']}): pairs {np.flatnonzero(out < 0)[:4].tolist()} are in no aligned block")
    return out


def tile(profile, e: dict) -> dict:
    """A launch's statics: E, N, K, G, KB, SWAP and its e4m3 k32 step, the Boolean one and the twin's name."""
    from verity_vllm.program.frontend import targets as T

    a, cfg = e["in"], e["in"]["config"]
    E, N, K = (int(x) for x in a["B"]["shape"])
    bm, bn, kb, warps = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"], cfg.get("num_warps", 4)
    swap = bm < 64 and bn >= 64
    if why := T.moe_fp8_tuned_tile_refusal(profile, bm, bn, kb, warps, swap):
        raise LookupError(f"moe_launch (seq {e['seq']}): {why}")
    dot = T.moe_expert_fp8_launch_dot(profile, bm, bn, warps, swap)
    return {"E": E, "N": N, "K": K, "G": int(a.get("block_shape", [128, 128])[1]), "KB": int(kb), "SWAP": bool(swap),
            "dot": getattr(BG, dot.id.rsplit("_v", 1)[0]), "twin": TWIN_DOT_OF[dot.id]}


def param_name(cap, r: int, layer: int, suffix: str) -> str:
    got = [n for n in cap.params[r] if n.endswith(suffix) and (m := _LAYER.search(n)) and int(m.group(1)) == layer]
    if len(got) != 1:
        raise LookupError(f"rank {r}: {len(got)} parameters of layer {layer} end in {suffix!r}, not one")
    return got[0]


def served(cap, r: int, layer: int, T: int, profile) -> dict:
    """Layer ``layer``'s values on rank ``r`` at T of the step's real tokens drawn by the capture's coins."""
    gate, w13, w2 = launches(cap.events[r], layer)
    topk = int(w2["in"]["topk_weights"]["shape"][1])
    rows = cap.real_rows(r, gate["out"]["ret0"]["shape"][0])
    tokens = sorted(ST.draw(cap.coins, f"moe-window/{r}/{layer}", rows, T))
    if len(tokens) != T:
        raise LookupError(f"layer {layer}: {T} tokens asked, the step has {rows}")
    up, down = tile(profile, w13), tile(profile, w2)
    S = int(w13["in"]["A"]["shape"][0])
    ex13, ex2 = experts_of(cap, r, w13, S * topk), experts_of(cap, r, w2, S * topk)
    if not np.array_equal(ex13, ex2):
        raise ValueError(f"layer {layer}: the w13 and w2 launches route a pair to two experts")
    j = (np.array(tokens)[:, None] * topk + np.arange(topk)[None, :])
    tw = cap.tensor(r, w2["in"]["topk_weights"]).reshape(-1)
    c13 = cap.tensor(r, w13["out"]["C"]).reshape(S, topk, up["N"])
    c2 = cap.tensor(r, w2["out"]["C"]).reshape(S, topk, down["N"])
    bias = cap.param(r, param_name(cap, r, layer, ".e_score_correction_bias")).reshape(-1).view(np.uint32)
    return {"tokens": tokens, "topk": topk, "up": up, "down": down,
            "logits": np.ascontiguousarray(cap.tensor(r, gate["out"]["ret0"])[tokens]).view(np.uint32),
            "bias": np.ascontiguousarray(bias), "g": np.ascontiguousarray(tw[j]).view(np.uint32), "e": ex13[j],
            "x13": cap.tensor(r, w13["in"]["A"])[tokens], "s13": cap.tensor(r, w13["in"]["A_scale"])[tokens].view(np.uint32),
            "x2": cap.tensor(r, w2["in"]["A"])[j.reshape(-1)], "s2": cap.tensor(r, w2["in"]["A_scale"])[j.reshape(-1)].view(np.uint32),
            "W13": cap.tensor(r, w13["in"]["B"]), "WS13": cap.tensor(r, w13["in"]["B_scale"]).view(np.uint32),
            "W2": cap.tensor(r, w2["in"]["B"]), "WS2": cap.tensor(r, w2["in"]["B_scale"]).view(np.uint32),
            "y13": np.ascontiguousarray(c13[tokens]), "y2": np.ascontiguousarray(c2[tokens]),
            "names": {"up": (w13["in"]["B"]["param"], w13["in"]["B_scale"]["param"]),
                      "down": (w2["in"]["B"]["param"], w2["in"]["B_scale"]["param"])}}


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] ^= 1
    return b


def _f32(rng: np.random.Generator, lo: float, hi: float, *shape: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32).view(np.uint32)


def bench(v: dict, seed: int, router) -> dict:
    """``v``'s shapes, tiles and tokens with benchmark values: random weights, scales, activations and router inputs, the
    slots the router's Definition gives and the outputs the twins give."""
    rng = np.random.default_rng(seed)
    up, down, topk, T = v["up"], v["down"], v["topk"], len(v["tokens"])
    E = up["E"]
    out = {**v, "W13": _e4m3(rng, E, up["N"], up["K"]), "WS13": _f32(rng, 1e-4, 1e-2, E, -(-up["N"] // up["G"]), up["K"] // up["G"]),
           "W2": _e4m3(rng, E, down["N"], down["K"]),
           "WS2": _f32(rng, 1e-4, 1e-2, E, -(-down["N"] // down["G"]), down["K"] // down["G"]),
           "logits": (rng.standard_normal((T, E)) * 2).astype(np.float32).view(np.uint32),
           "bias": (rng.standard_normal(E) * 1e-2).astype(np.float32).view(np.uint32),
           "x13": _e4m3(rng, T, up["K"]), "s13": _f32(rng, 1e-3, 1e-1, T, up["K"] // up["G"]),
           "x2": _e4m3(rng, T * topk, down["K"]), "s2": _f32(rng, 1e-3, 1e-1, T * topk, down["K"] // down["G"])}
    slots = np.array([evaluate(router, out["logits"][t].tolist(), out["bias"].tolist()) for t in range(T)], dtype=np.int64)
    out["g"], out["e"] = slots[:, :topk].astype(np.uint32), slots[:, topk:]
    out["y13"], out["y2"] = twins(out)
    out["names"] = {p: (f"bench.layers.0.mlp.experts.{w}_weight", f"bench.layers.0.mlp.experts.{w}_weight_scale_inv")
                    for p, w in (("up", "w13"), ("down", "w2"))}
    return out


def twins(v: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each pair's w13 and w2 output words by the launch's twin on its expert's rows: ``(T, TOPK, N)`` each."""
    T, topk = len(v["tokens"]), v["topk"]
    out = []
    for p, W, WS, x, s in (("up", "W13", "WS13", "x13", "s13"), ("down", "W2", "WS2", "x2", "s2")):
        S = v[p]
        y = np.empty((T, topk, S["N"]), dtype=np.uint16)
        for t in range(T):
            for k in range(topk):
                e, i = int(v["e"][t, k]), t if p == "up" else t * topk + k
                y[t, k], _ = moe_expert_fp8_block_row(v[x][i], v[s][i], v[W][e], v[WS][e], G=S["G"], KB=S["KB"], swap=S["SWAP"],
                                                      g=None if p == "up" else int(v["g"][t, k]), dot=S["twin"])
        out.append(y)
    return out[0], out[1]


def gate(v: dict, router) -> dict:
    """The match gate: each router unit's slots its Definition's on its logits and bias, each pair's outputs its twin's."""
    T, topk = len(v["tokens"]), v["topk"]
    want = np.array([evaluate(router, v["logits"][t].tolist(), v["bias"].tolist()) for t in range(T)], dtype=np.int64)
    got = np.concatenate([v["g"].astype(np.int64), v["e"].astype(np.int64)], axis=1)
    y13, y2 = twins(v)
    bad = {"router": np.flatnonzero((want != got).any(axis=1)).tolist(),
           "up": [list(x) for x in np.argwhere((y13 != v["y13"]).any(axis=2)).tolist()],
           "down": [list(x) for x in np.argwhere((y2 != v["y2"]).any(axis=2)).tolist()]}
    return {"units": {"router": T, "up_pairs": T * topk, "down_pairs": T * topk}, "mismatched": bad,
            "ok": not any(bad.values())}


# ---- the circuit -------------------------------------------------------------------------------------------------------------

def labels(names: dict, params: list[str] | None) -> dict:
    """Each projection's weight and scale labels in the registration's parameters (its weights struct's field names, which may
    carry a wrapper prefix the capture's parameter names lack); the capture's names without a registration."""
    if params is None:
        return {p: {"w": w, "ws": ws} for p, (w, ws) in names.items()}
    fields = {q[len("weights/"):].split("@", 1)[0] for q in params}

    def one(name: str) -> str:
        got = [f for f in fields if f == name or f.endswith("." + name) or name.endswith("." + f)]
        if len(got) != 1:
            raise LookupError(f"the registration names {len(got)} fields for the capture's {name}: {sorted(got)[:3]}")
        return got[0]
    return {p: {"w": one(w), "ws": one(ws)} for p, (w, ws) in names.items()}


def templates(v: dict, scale: float, kernel: str, blocks: dict) -> dict:
    up, down = v["up"], v["down"]
    st = {p: {"K": S["K"], "G": S["G"], "KB": S["KB"], "SWAP": S["SWAP"], "DOT": S["dot"]} for p, S in (("up", up), ("down", down))}
    return {"router": bind(BMOE.MoeRouterSigmoidBiasSlots, E=up["E"], TOPK=v["topk"], SCALE=scale, KERNEL=kernel),
            "word_router": bind(MOE.MoeRouterSigmoidBiasSlots, E=up["E"], TOPK=v["topk"], SCALE=scale, KERNEL=kernel),
            "up": bind(BM.MoeExpertFp8BlockGemmV4, N=blocks["w13"], **st["up"]),
            "down": bind(BM.MoeExpertFp8BlockGemmWV4, N=blocks["w2"], **st["down"])}


def cut(v: dict, fns: dict, lab: dict, blocks: dict, name: str) -> tuple[WI.Circuit, dict]:
    """The layer's cut at T tokens (module docstring) and its plan: per projection its read templates, blocks and tables."""
    from verity_vllm.pipeline.expert_rows import table_param

    T, topk, E = len(v["tokens"]), v["topk"], v["up"]["E"]
    rid = descriptor_id(fns["router"])
    tmpls: list[tuple[object, int]] = [(fns["router"], T)]
    wires: list[WI.Wire] = []
    params, tables, roots, openings, plan = {}, [], [], [], {}

    def read(label: str, H: int, B: int) -> tuple[str, list[str], object]:
        fn = EMB.read(E, H)
        tid, mine = descriptor_id(fn), []
        tmpls.append((fn, B * topk * T))
        for b in range(B):
            at, param = b * topk * T, table_param(label, H, b)
            params[param] = H * E
            mine.append(param)
            tables.append(WI.Table(param, param, 0, H, E))
            roots.extend(WI.TableRead(tid, p, param, p, at, topk * T) for p in ("root", "domain"))
            openings.append(WI.Opening(tid, param, "tok", ("row", "salt", "path"), "o1", at, topk * T))
            wires.extend(WI.Wire(tid, "tok", rid, f"o{topk + k}", at + k * T, (T,), 0, (1,)) for k in range(topk))
        return tid, mine, fn

    for proj, blk in (("up", "w13"), ("down", "w2")):
        S, n = v[proj], blocks[blk]
        K, N, G = S["K"], S["N"], S["G"]
        if N % n or G % n:
            raise ValueError(f"{proj}: blocks of {n} rows don't divide its {N} rows and G = {G}")
        B, NBG, per = N // n, -(-N // G), G // n
        (rows, wt, wfn), (scales, st, sfn) = read(lab[proj]["w"], n * K // 2, B), read(lab[proj]["ws"], 2 * K // G, NBG)
        gemm = fns[proj]
        gid = descriptor_id(gemm)
        tmpls.append((gemm, B * topk * T))
        if proj == "up":
            wires += [WI.Wire(gid, "w", rows, "o0", 0, (B * topk * T,), 0, (1,)),
                      WI.Wire(gid, "ws", scales, "o0", 0, (NBG, per, topk * T), 0, (topk * T, 0, 1))]
        else:
            wires += [WI.Wire(gid, "w", rows, "o0", 0, (topk, B, T), 0, (T, topk * T, 1)),
                      WI.Wire(gid, "ws", scales, "o0", 0, (topk, NBG, per, T), 0, (T, topk * T, 0, 1))]
            wires += [WI.Wire(gid, "g", rid, f"o{k}", k * B * T, (B, T), 0, (0, 1)) for k in range(topk)]
        plan[proj] = {"n": n, "B": B, "NBG": NBG, "rows": (rows, wt, wfn), "scales": (scales, st, sfn), "gemm": gid}
    reg = WI.Registered(params, tuple(tables), (), tuple(roots), tuple(openings))
    return WI.Circuit(name, tuple(tmpls), wires=tuple(WI.coalesce(wires)), registered=reg), plan


# ---- the tables and the window -----------------------------------------------------------------------------------------------

def commit_tables(v: dict, plan: dict, committer: dict) -> dict:
    """Each table the cut reads, committed from ``v``'s weights as `expert_tables` registers it, by parameter."""
    from verity_vllm.pipeline.expert_rows import table_words
    from verity_vllm.protocols.pouw.h1 import registered as H1R

    out = {}
    for proj, W, WS in (("up", "W13", "WS13"), ("down", "W2", "WS2")):
        S, p = v[proj], plan[proj]
        for what, stack, (_tid, params, _fn) in (("w", v[W], p["rows"]), ("ws", v[WS], p["scales"])):
            for b, param in enumerate(params):
                if param not in committer["nonces"]:
                    raise LookupError(f"{param}: the registration names no such table")
                words = table_words(stack, what, S["E"], S["N"], S["K"], S["G"], p["n"], b)
                out[param] = H1R.commit(committer["program"], param, words, committer["key"], int(committer["nonces"][param]),
                                        keep_rows=True, schema=REG.SCHEMA_SEG)
    return out


def _row(table, e: int) -> np.ndarray:
    rows = table.rows
    return np.asarray(rows.at([e])[0] if hasattr(rows, "at") else rows[e], dtype=np.uint16).reshape(-1)


def _u64(b: bytes) -> np.ndarray:
    return np.frombuffer(bytes(b), ">u8").astype(np.uint64)


def record(win: WI.Window, v: dict, fns: dict, plan: dict, tables: dict, forge: str | None) -> dict:
    """Every unit of the cut recorded into ``win`` in its template's order (module docstring); -> units by template."""
    T, topk, E = len(v["tokens"]), v["topk"], v["up"]["E"]
    win.record(fns["router"], [v["logits"], np.broadcast_to(v["bias"], (T, E))],
               [*(np.ascontiguousarray(v["g"][:, k]) for k in range(topk)),
                *(v["e"][:, k].astype(np.int32) for k in range(topk))])
    units = {"router": T}
    for proj in ("up", "down"):
        p = plan[proj]
        n = p["n"]
        for part, (_tid, params, fn) in (("w", p["rows"]), ("ws", p["scales"])):
            rows, salts, paths = [], [], []
            for param in params:
                tb = tables[param]
                for k in range(topk):
                    for t in range(T):
                        e = int(v["e"][t, k])
                        rows.append(_row(tb, e))
                        salts.append(_u64(tb.salts[e]))
                        paths.append(_u64(tb.path(e)).reshape(-1, 8))
            rows_ = np.stack(rows)
            win.record(fn, [None, None, None, rows_, np.stack(salts), np.stack(paths)],
                       [rows_, np.ones(len(rows), dtype=np.uint16)])
            units[f"{proj}.{part}"] = len(rows)
        y = (v["y13"] if proj == "up" else v["y2"]).copy()
        if forge == "w13" and proj == "up":
            y ^= np.uint16(1)
        if proj == "up":
            order = [(b, k, t) for b in range(p["B"]) for k in range(topk) for t in range(T)]
            xq = np.stack([v["x13"][t] for _b, _k, t in order])
            sx = np.stack([v["s13"][t] for _b, _k, t in order])
            ins = [xq, sx, None, None]
        else:
            order = [(b, k, t) for k in range(topk) for b in range(p["B"]) for t in range(T)]
            xq = np.stack([v["x2"][t * topk + k] for _b, k, t in order])
            sx = np.stack([v["s2"][t * topk + k] for _b, k, t in order])
            ins = [xq, sx, None, None, None]
        names = [q for q, _w in WI.layout(fns[proj]).rows]
        if names != (["xq", "sx", "w", "ws"] if proj == "up" else ["xq", "sx", "g", "w", "ws"]):
            raise ValueError(f"{proj}: the block GEMM's ports are {names}")
        win.record(fns[proj], ins, [np.stack([y[t, k, b * n:(b + 1) * n] for b, k, t in order])])
        units[f"{proj}.gemm"] = len(order)
    return units


def registration_doc(program: str, world: int, rank: int, values: dict, blocks: dict, repo: str, revision: str) -> dict:
    """A self-registration in `expert_tables`' format (``prefix`` "": its names are the capture's parameters')."""
    from verity_vllm.pipeline import expert_tables as XT

    return {"format": XT.FORMAT, "program": program, "repo": repo, "revision": revision, "world": world, "rank": rank,
            "prefix": "", "blocks": dict(sorted(blocks.items())), "values": dict(sorted(values.items()))}


def main(argv: list[str] | None = None) -> int:
    from verity_vllm import TargetProfile
    from verity_vllm.pipeline import expert_tables as XT

    ap = argparse.ArgumentParser(prog="moe_window.py")
    ap.add_argument("--capture", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=1, help="T: the step's real tokens the window holds, drawn by its coins")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", help="the rank's registration of its tables (verity-vllm tp-register-tables), the verifier's pins")
    ap.add_argument("--self-register", action="store_true", help="no registration: commit the tables under --key at a program "
                    "of the window's own and write that registration (made after serving)")
    ap.add_argument("--key", help="the committer's 32-byte key file (--tables' own; with --self-register, written if absent)")
    ap.add_argument("--block", action="append", default=[], help="PROJ=n without --tables (default w13=1, w2=32)")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--law", default="stratified:1")
    ap.add_argument("--scale", type=float, default=2.5, help="the router's routed scaling factor (GLM-5.3's 2.5)")
    ap.add_argument("--kernel", default="block", help="the router Definition's KERNEL")
    ap.add_argument("--cc", default="9.0")
    ap.add_argument("--forge", choices=["w13"], help="flip bit 0 of every w13 GEMM unit's output after the gate")
    ap.add_argument("--bench", type=int, metavar="SEED", help="benchmark values of the same shapes (module docstring)")
    ap.add_argument("--circuit-only", action="store_true")
    ap.add_argument("--full-gate", action="store_true", help="also evaluate every unit's template (Window.mismatches)")
    ap.add_argument("--repo", default="zai-org/GLM-5.3", help="with --self-register: the registration's model")
    ap.add_argument("--revision", default="capture", help="with --self-register: the registration's revision")
    a = ap.parse_args(argv)
    if bool(a.tables) == (a.self_register or a.bench is not None):
        ap.error("give --tables, or --self-register (or --bench, which always self-registers)")
    if not a.circuit_only and not a.key:
        ap.error("--key: the committer's key")
    t0, out = time.time(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = ST.Capture(Path(a.capture), a.step)
    if a.rank not in cap.ranks:
        raise SystemExit(f"rank {a.rank} recorded no step {a.step}")
    if a.layer not in cap.layers:
        raise SystemExit(f"layer {a.layer} is not opened at step {a.step}: {cap.layers}")
    profile = TargetProfile(compute_capability=tuple(map(int, a.cc.split("."))), batch_invariant=False)
    v = served(cap, a.rank, a.layer, a.tokens, profile)
    doc = json.loads(Path(a.tables).read_text()) if a.tables else None
    if doc is not None:
        if int(doc.get("rank", -1)) != a.rank:
            raise SystemExit(f"{a.tables}: the registration of rank {doc.get('rank')}, not {a.rank}")
        blocks = {k: int(x) for k, x in doc["blocks"].items()}
    else:
        blocks = {"w13": 1, "w2": 32} | {b.split("=")[0]: int(b.split("=")[1]) for b in a.block}
    fns = templates(v, a.scale, a.kernel, blocks)
    if a.bench is not None:
        v = bench(v, a.bench, fns["word_router"])
    lab = labels(v["names"], list(doc["values"]) if doc else None)
    run = (hashlib.sha512(f"verity/spot/moe-window/bench\0{a.bench}".encode()).hexdigest() if a.bench is not None else cap.root)
    name = f"spot-moe-per-expert/L{a.layer}/T{a.tokens}"
    circuit, plan = cut(v, fns, lab, blocks, name)
    law = D.parse_law(a.law)
    report = {"capture": a.capture, "step": a.step, "layer": a.layer, "rank": a.rank, "tokens": v["tokens"], "run": run,
              "bench": a.bench, "forge": a.forge, "law": a.law, "cap": a.cap, "blocks": blocks,
              "statics": {p: {k: (x.id if k == "dot" else x) for k, x in v[p].items()} for p in ("up", "down")},
              "templates": {k: descriptor_id(fn) for k, fn in fns.items() if k != "word_router"}}
    if a.circuit_only:
        win = WI.Window(circuit, run, law, a.cap)
        digest = hashlib.sha512(WI.circuit_bytes(win.circuit_record())).hexdigest()
        print(json.dumps({"circuit": digest, "run": run, "members": len(win.window.members)}))
        return 0
    report["gate"] = g = gate(v, fns["word_router"])
    print("GATE " + json.dumps(g, sort_keys=True), flush=True)
    if not g["ok"]:
        (out / "stage.json").write_text(json.dumps(report, indent=1, default=str))
        raise SystemExit("the match gate refused the layer's values: nothing staged")
    key_file = Path(a.key)
    if a.tables is None and not key_file.exists():
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    key = key_file.read_bytes()
    if doc is None:
        program = XT.program(a.repo, f"{a.revision}/{'bench-' + str(a.bench) if a.bench is not None else cap.root[:32]}",
                             len(cap.ranks), a.rank, "")
        committer = {"key": key, "program": program, "nonces": XT.nonces(sorted(circuit.registered.params), a.rank)}
    else:
        committer = XT.committer(doc, key)
    t1 = time.time()
    tables = commit_tables(v, plan, committer)
    report["tables_s"] = round(time.time() - t1, 1)
    if doc is None:
        doc = registration_doc(committer["program"], len(cap.ranks), a.rank, {p: t.entry() for p, t in tables.items()}, blocks,
                               a.repo, a.revision)
    wrong = sorted(p for p, t in tables.items() if t.entry() != doc["values"].get(p))
    report["tables"] = {"read": len(tables), "registered": len(doc["values"]), "not_registered_rows": wrong}
    if wrong:
        (out / "stage.json").write_text(json.dumps(report, indent=1, default=str))
        raise SystemExit(f"{len(wrong)} tables' served rows are not the registered ones (first {wrong[0]}): nothing staged")
    (out / "tables.json").write_text(json.dumps(doc, indent=1) + "\n")
    win = WI.Window(circuit, run, law, a.cap, registered=tables)
    win.supply(REG)
    report["units"] = record(win, v, fns, plan, tables, a.forge)
    if a.full_gate:
        t1 = time.time()
        bad = win.mismatches()
        report["full_gate"] = {"mismatched": {t: u[:8] for t, u in bad.items()}, "seconds": round(time.time() - t1, 1)}
        print("FULL-GATE " + json.dumps(report["full_gate"], sort_keys=True), flush=True)
        if bad and not a.forge:
            (out / "stage.json").write_text(json.dumps(report, indent=1, default=str))
            raise SystemExit(f"units disagree with their templates: {sorted(bad)}")
    t1 = time.time()
    c = win.close()
    prov = {"label": "spot-moe-window", "of": {"step_root": cap.root, "step": a.step, "rank": a.rank, "layer": a.layer,
                                               "tokens": v["tokens"]}, "link": LINK,
            **({"bench": a.bench} if a.bench is not None else {}), **({"forged": a.forge} if a.forge else {})}
    win.write(c, out / "window", prov, registrar=R)
    report["commit_s"] = round(time.time() - t1, 1)
    report["circuit"] = hashlib.sha512((out / "window" / "circuit.json").read_bytes()).hexdigest()
    report["members"] = len(win.window.members)
    report["seconds"] = round(time.time() - t0, 1)
    (out / "stage.json").write_text(json.dumps(report, indent=1, default=str))
    print("STAGED " + json.dumps({k: report[k] for k in ("circuit", "run", "members", "units", "tokens", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
