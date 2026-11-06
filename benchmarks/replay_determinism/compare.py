"""Compare `replay_det.py`'s phases bit for bit: their schedules, their tokens, and with ``--capture`` every row's
fingerprints, layer by layer and op by op.

``--runs DIR`` holds one directory per phase (``serve``, the replays, ``alone``).  Phases with the same schedule are compared
row by row (step, row and module output); a request alone is compared with the same request inside the served window by
(request, position), over the positions whose input tokens agree (up to the first token where the two outputs diverge).  The
result is ``compare.json`` and ``compare.md`` in ``--out``.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ABSENT = np.int64(0x5EED5EED5EED5EED)
POS_BITS = 20
SCHEDULE_FIELDS = ("T", "rows", "pages", "new", "resumed", "preempted", "finished")


class Run:
    def __init__(self, d: Path) -> None:
        self.dir = d
        self.name = d.name
        self.steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines() if line.strip()]
        self.tokens: dict[str, list[int]] = json.loads((d / "tokens.json").read_text())
        self.timing = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
        self.captured = (d / "rows.npy").exists()
        if self.captured:
            self.rows = np.load(d / "rows.npy", mmap_mode="r")
            self.keys = np.load(d / "keys.npy")
            self.labels: list[str] = json.loads((d / "labels.json").read_text())
            self.logit_keys = np.load(d / "logit_keys.npy")
            self.logit_digest = np.load(d / "logit_digest.npy")
            self.logit_top = np.load(d / "logit_top.npy")
            self.logit_topi = np.load(d / "logit_topi.npy")
            self.whole = [json.loads(line) for line in (d / "whole.jsonl").read_text().splitlines()]


#: each op of decoder layer N and the outputs it reads (`@` names resolved per layer by `_inputs`); MLA attention reads
#: its own query and the keys of every position up to its own (``@kv_prefix``)
READS = {
    "input_layernorm:0": ["@prev_mlp", "@prev_residual"],
    "self_attn.q_proj": ["@ln_in"],
    "self_attn.kv_a_proj_with_mqa": ["@ln_in"],
    "self_attn.kv_a_layernorm": ["self_attn.kv_a_proj_with_mqa"],
    "self_attn.mla_attn.mla_attn": ["self_attn.q_proj", "self_attn.rotary_emb:0", "@kv_prefix"],
    "self_attn.o_proj": ["self_attn.mla_attn.mla_attn"],
    "post_attention_layernorm:0": ["self_attn.o_proj", "@res_in"],
    "mlp.gate_up_proj": ["post_attention_layernorm:0"],
    "mlp.act_fn": ["mlp.gate_up_proj"],
    "mlp.down_proj": ["mlp.act_fn"],
    "mlp.gate": ["post_attention_layernorm:0"],
    "mlp.shared_experts.gate_up_proj": ["post_attention_layernorm:0"],
    "mlp.shared_experts.act_fn": ["mlp.shared_experts.gate_up_proj"],
    "mlp.shared_experts.down_proj": ["mlp.shared_experts.act_fn"],
    "mlp.experts.router.select_experts.ids": ["mlp.gate"],
    "mlp.experts.router.select_experts.weights": ["mlp.gate"],
    "mlp.experts": ["post_attention_layernorm:0", "mlp.experts.router.select_experts.ids",
                    "mlp.experts.router.select_experts.weights", "mlp.shared_experts.down_proj"],
}
KV = ("self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm", "self_attn.rotary_emb:1")


def canonical(label: str) -> str:
    """A module every layer shares (the rotary embedding) is registered under layer 0's name and labelled by its
    occurrence in the step (``#k``); its k-th occurrence is layer k's."""
    m = re.fullmatch(r"(model\.layers\.)0(\..*)#(\d+)", label)
    return f"{m.group(1)}{m.group(3)}{m.group(2)}" if m else label


def op_of(label: str) -> str:
    return re.sub(r"layers\.\d+\.", "layers.N.", canonical(label))


def layer_of(label: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", canonical(label))
    return int(m.group(1)) if m else None


def schedule(a: Run, b: Run) -> dict[str, Any]:
    """Equal when every step has the same tokens, rows (request, first position, length, in the runner's order), new KV
    blocks, arrivals, preemptions and finishes."""
    out: dict[str, Any] = {"steps": [len(a.steps), len(b.steps)], "equal": False}
    for i, (x, y) in enumerate(zip(a.steps, b.steps)):
        fields = [f for f in SCHEDULE_FIELDS if x.get(f) != y.get(f)]
        if fields:
            out["first_difference"] = {"step": i, "fields": fields}
            return out
    out["equal"] = len(a.steps) == len(b.steps)
    return out


def first_divergence(x: list[int], y: list[int]) -> int | None:
    for i, (s, t) in enumerate(zip(x, y)):
        if s != t:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def tokens(a: Run, b: Run) -> dict[str, Any]:
    ids = sorted(set(a.tokens) | set(b.tokens))
    div = {r: first_divergence(a.tokens.get(r, []), b.tokens.get(r, [])) for r in ids}
    differ = {r: d for r, d in div.items() if d is not None}
    return {"requests": len(ids), "equal": len(ids) - len(differ), "all_equal": not differ,
            "tokens": [sum(map(len, a.tokens.values())), sum(map(len, b.tokens.values()))],
            "first_divergence": dict(sorted(differ.items()))}


def _columns(a: Run, b: Run) -> tuple[list[str], np.ndarray, np.ndarray]:
    at = {lab: i for i, lab in enumerate(b.labels)}
    common = [lab for lab in a.labels if lab in at]
    return common, np.array([a.labels.index(lab) for lab in common], dtype=np.int64), np.array([at[lab] for lab in common], dtype=np.int64)


def _first_labels(diff: np.ndarray, common: list[str]) -> list[str]:
    hit = diff.any(axis=1)
    return [common[j] for j in np.argmax(diff[hit], axis=1)]


def values(a: Run, b: Run) -> dict[str, Any]:
    """Two phases with the same schedule, row by row: each step's rows in the same order, every module output's
    fingerprint, every logits row and every non-row output."""
    if a.keys.shape != b.keys.shape or not np.array_equal(a.keys, b.keys):
        n = min(len(a.keys), len(b.keys))
        bad = np.flatnonzero((a.keys[:n] != b.keys[:n]).any(axis=1))
        return {"keys_equal": False, "rows": [len(a.keys), len(b.keys)],
                "first_key_difference": int(bad[0]) if len(bad) else n}
    common, ia, ib = _columns(a, b)
    A, B = np.asarray(a.rows)[:, ia], np.asarray(b.rows)[:, ib]
    diff = A != B
    rows_diff = diff.any(axis=1)
    out: dict[str, Any] = {"keys_equal": True, "rows": int(len(A)), "labels": len(common),
                           "labels_only_in": {a.name: sorted(set(a.labels) - set(common)), b.name: sorted(set(b.labels) - set(common))},
                           "cells_differing": int(diff.sum()), "rows_differing": int(rows_diff.sum())}
    if rows_diff.any():
        r0 = int(np.argmax(rows_diff))
        step, req, pos = (int(v) for v in a.keys[r0, :3])
        out["first_difference"] = {"step": step, "request": req, "position": pos, "label": common[int(np.argmax(diff[r0]))]}
        out["steps_differing"] = int(len(np.unique(a.keys[rows_diff, 0])))
        out["first_label_counts"] = dict(Counter(_first_labels(diff, common)).most_common(12))
    ld = a.logit_digest != b.logit_digest
    out["logits"] = {"rows": int(len(ld)), "rows_differing": int(ld.sum()),
                     "argmax_differing": int((a.logit_topi[:, 0] != b.logit_topi[:, 0]).sum())}
    if ld.any():
        out["logits"]["max_abs_top1_delta"] = float(np.abs(a.logit_top[:, 0] - b.logit_top[:, 0]).max())
    wd = [i for i, (x, y) in enumerate(zip(a.whole, b.whole)) if x != y]
    out["whole"] = {"steps": len(a.whole), "steps_differing": len(wd), "first_step": wd[0] if wd else None}
    out["bit_exact"] = not rows_diff.any() and not ld.any() and not wd
    return out


def _chunks(run: Run, keys: np.ndarray) -> np.ndarray:
    """[rows, 3]: the first position and length of each row's chunk of its request in its step, and the step's tokens."""
    at = {(s["step"], int(rid[1:])): (start, n, s["T"]) for s in run.steps for rid, start, n in s["rows"]}
    return np.array([at[(int(st), int(rq))] for st, rq in keys[:, :2]], dtype=np.int64).reshape(-1, 3)


def _prefix_all(ok: np.ndarray, req: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """ok at every position of the row's request up to and including its own."""
    order = np.lexsort((pos, req))
    bad = (~ok[order]).astype(np.int64)
    first = np.r_[True, req[order][1:] != req[order][:-1]]
    c = np.cumsum(bad)
    group = np.cumsum(first) - 1
    before = (c - bad)[first][group]
    out = np.empty(len(ok), dtype=bool)
    out[order] = (c - before) == 0
    return out


def _reads(n: int, op: str) -> list[str] | None:
    """The labels op of layer n reads, or None for the KV prefix (`READS`)."""
    out = []
    for name in READS[op]:
        if name == "@kv_prefix":
            continue
        out += {"@ln_in": [f"model.layers.{n}.input_layernorm" + (":0" if n else "")],
                "@res_in": ["model.embed_tokens" if n == 0 else f"model.layers.{n}.input_layernorm:1"],
                "@prev_mlp": [f"model.layers.{n - 1}.mlp"],
                "@prev_residual": [f"model.layers.{n - 1}.post_attention_layernorm:1"]}.get(name, [f"model.layers.{n}.{name}"])
    return out


def op_variance(eq: np.ndarray, valid: np.ndarray, common: list[str], req: np.ndarray, pos: np.ndarray,
                decode: np.ndarray) -> dict[str, Any]:
    """Each op's output against its inputs: over the rows where everything the op reads is bit-identical alone and in the
    batch, how many rows of its output differ (by op over every layer, prefill and decode apart)."""
    col = {canonical(lab): j for j, lab in enumerate(common)}
    same = eq | ~valid
    layers = sorted({n for lab in col if (n := layer_of(lab)) is not None})
    table: dict[str, dict[str, int]] = {}
    for n in layers:
        kv = [col.get(f"model.layers.{n}.{k}") for k in KV]
        kv_prefix = _prefix_all(np.all([eq[:, j] & valid[:, j] for j in kv if j is not None], axis=0), req, pos) if all(
            j is not None for j in kv) else None
        for op in READS:
            if op == "input_layernorm:0" and n == 0:
                continue
            j = col.get(f"model.layers.{n}.{op}")
            reads = _reads(n, op)
            if j is None or any(r not in col for r in reads) or ("@kv_prefix" in READS[op] and kv_prefix is None):
                continue
            ok = valid[:, j] & np.all([eq[:, col[r]] & valid[:, col[r]] for r in reads], axis=0)
            if "@kv_prefix" in READS[op]:
                ok &= kv_prefix
            t = table.setdefault(f"model.layers.N.{op}", {"inputs_equal_prefill": 0, "differs_prefill": 0,
                                                          "inputs_equal_decode": 0, "differs_decode": 0})
            for kind, sel in (("prefill", ~decode), ("decode", decode)):
                t[f"inputs_equal_{kind}"] += int((ok & sel).sum())
                t[f"differs_{kind}"] += int((ok & sel & ~same[:, j]).sum())
    return table


def _index(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = keys[:, 1] * (1 << POS_BITS) + keys[:, 2]
    uniq, first = np.unique(k, return_index=True)
    return uniq, first


def alone(batch: Run, solo: Run, workload: dict[str, Any]) -> dict[str, Any]:
    """Each request alone against the same request inside the window, by (request, position), over the positions whose
    inputs agree: a row whose module outputs all match is bit-identical; else the first module output (in the forward's
    order) that differs is where it diverged."""
    reqs = {int(r["id"][1:]): r for r in workload["requests"]}
    plen = {i: len(r["prompt"]) for i, r in reqs.items()}
    div = {i: first_divergence(batch.tokens[r["id"]], solo.tokens[r["id"]]) for i, r in reqs.items()}
    limit = np.zeros(max(reqs) + 1, dtype=np.int64)
    for i, r in reqs.items():
        limit[i] = plen[i] + (div[i] if div[i] is not None else r["max_tokens"] - 1)
    kb, fb = _index(batch.keys)
    ka, fa = _index(solo.keys)
    _, xb, xa = np.intersect1d(kb, ka, return_indices=True)
    rb, ra = fb[xb], fa[xa]
    req, pos = batch.keys[rb, 1], batch.keys[rb, 2]
    keep = pos < limit[req]
    rb, ra, req, pos = rb[keep], ra[keep], req[keep], pos[keep]
    inputs_equal = bool(np.array_equal(batch.keys[rb, 3], solo.keys[ra, 3]))
    common, cb, ca = _columns(batch, solo)
    A, B = np.asarray(batch.rows)[rb][:, cb], np.asarray(solo.rows)[ra][:, ca]
    diff = (A != B) & (A != ABSENT) & (B != ABSENT)
    hit = diff.any(axis=1)
    decode = pos >= np.array([plen[int(i)] for i in req])
    out: dict[str, Any] = {"rows_compared": int(len(rb)), "inputs_equal": inputs_equal, "labels": len(common),
                           "bit_identical": {"all": int((~hit).sum()), "prefill": int((~hit & ~decode).sum()),
                                             "decode": int((~hit & decode).sum())},
                           "rows": {"prefill": int((~decode).sum()), "decode": int(decode.sum())}}
    if hit.any():
        firsts = _first_labels(diff, common)
        out["first_label_counts"] = dict(Counter(firsts).most_common(12))
        out["first_op_counts"] = dict(Counter(op_of(lab) for lab in firsts).most_common(12))
        out["first_layer_counts"] = dict(Counter(str(layer_of(lab)) for lab in firsts).most_common(8))
        for kind, sel in (("prefill", ~decode), ("decode", decode)):
            f = _first_labels(diff[sel], common)
            out[f"first_op_counts_{kind}"] = dict(Counter(op_of(lab) for lab in f).most_common(6))
        out["labels_differing_share"] = float(diff[hit].mean())
    valid = (A != ABSENT) & (B != ABSENT)
    eq = A == B
    canon = [canonical(lab) for lab in common]
    final = canon.index("model.norm:0") if "model.norm:0" in canon else None
    if final is not None:
        out["final_hidden_differing"] = int(diff[:, final].sum())
    ids = [j for j, lab in enumerate(canon) if lab.endswith("router.select_experts.ids")]
    if ids:
        flips = ~eq[:, ids] & valid[:, ids]
        out["router"] = {"moe_layers": len(ids), "rows_with_an_expert_change": int(flips.any(axis=1).sum()),
                         "row_layers_changed": int(flips.sum()), "row_layers": int(valid[:, ids].sum())}
    cb_, ca_ = _chunks(batch, batch.keys[rb]), _chunks(solo, solo.keys[ra])
    same_chunk = (cb_[:, 0] == ca_[:, 0]) & (cb_[:, 1] == ca_[:, 1])
    pre = ~decode
    out["prefill_by_chunk"] = {
        kind: {"rows": int((pre & sel).sum()), "bit_identical": int((pre & sel & ~hit).sum()),
               "first_op_counts": dict(Counter(op_of(lab) for lab in _first_labels(diff[pre & sel], common)).most_common(4))}
        for kind, sel in (("same_chunk", same_chunk), ("other_chunk", ~same_chunk))}
    out["op_variance"] = op_variance(eq, valid, common, req, pos, decode)
    lb, flb = _index(batch.logit_keys)
    la, fla = _index(solo.logit_keys)
    _, yb, ya = np.intersect1d(lb, la, return_indices=True)
    gb, ga = flb[yb], fla[ya]
    lr, lp = batch.logit_keys[gb, 1], batch.logit_keys[gb, 2]
    ok = lp < limit[lr]
    gb, ga = gb[ok], ga[ok]
    delta = np.abs(batch.logit_top[gb, 0] - solo.logit_top[ga, 0])
    same_logits = batch.logit_digest[gb] == solo.logit_digest[ga]
    out["logits"] = {"rows_compared": int(len(gb)), "bit_identical": int(same_logits.sum()),
                     "argmax_equal": int((batch.logit_topi[gb, 0] == solo.logit_topi[ga, 0]).sum()),
                     "top1_abs_delta": {"max": float(delta.max()) if len(delta) else 0.0,
                                        "median": float(np.median(delta)) if len(delta) else 0.0}}
    if final is not None and len(gb):
        rows_k = req * (1 << POS_BITS) + pos
        at = dict(zip(rows_k.tolist(), range(len(rows_k))))
        idx = np.array([at.get(int(k), -1) for k in batch.logit_keys[gb, 1] * (1 << POS_BITS) + batch.logit_keys[gb, 2]])
        hidden_same = (idx >= 0) & np.where(idx >= 0, eq[np.maximum(idx, 0), final] & valid[np.maximum(idx, 0), final], False)
        out["logits"]["final_hidden_equal"] = int(hidden_same.sum())
        out["logits"]["differs_where_final_hidden_equal"] = int((hidden_same & ~same_logits).sum())
    greedy = {i for i, r in reqs.items() if r["temperature"] == 0}
    out["tokens"] = {"requests": len(reqs), "identical": sum(d is None for d in div.values()),
                     "identical_greedy": sum(div[i] is None for i in greedy), "greedy": len(greedy),
                     "first_divergence": {reqs[i]["id"]: d for i, d in sorted(div.items()) if d is not None}}
    return out


def gpu_seconds(runs: list[Run]) -> dict[str, Any]:
    per = {r.name: {k: round(r.timing.get(k, 0.0), 1) for k in ("load_s", "serve_s", "total_s")} | {"steps": r.timing.get("steps")}
           for r in runs}
    return {"per_phase": per, "total_s": round(sum(r.timing.get("total_s", 0.0) for r in runs), 1)}


def render(res: dict[str, Any]) -> str:
    lines = ["# Replay determinism", ""]
    for k, v in res["pairs"].items():
        s, t = v["schedule"], v["tokens"]
        line = f"- {k}: schedule {'equal' if s['equal'] else 'differs ' + json.dumps(s.get('first_difference'))}; "
        line += f"tokens {t['equal']}/{t['requests']} requests equal"
        if "values" in v:
            x = v["values"]
            line += (f"; values {'bit-exact' if x.get('bit_exact') else 'differ'}" +
                     (f" (first {json.dumps(x['first_difference'])}, {x['rows_differing']}/{x['rows']} rows)" if x.get("first_difference") else ""))
        lines.append(line)
    if "alone" in res:
        a = res["alone"]
        lines += ["", f"- alone vs in batch: {a['bit_identical']['all']}/{a['rows_compared']} rows bit-identical "
                      f"(prefill {a['bit_identical']['prefill']}/{a['rows']['prefill']}, decode {a['bit_identical']['decode']}/{a['rows']['decode']}); "
                      f"tokens identical for {a['tokens']['identical']}/{a['tokens']['requests']} requests "
                      f"({a['tokens']['identical_greedy']}/{a['tokens']['greedy']} greedy)"]
        if "first_op_counts" in a:
            lines.append(f"- first differing op: {json.dumps(a['first_op_counts'])}")
            lines.append(f"- first differing layer: {json.dumps(a['first_layer_counts'])}")
        for kind, c in a.get("prefill_by_chunk", {}).items():
            lines.append(f"- prefill rows, {kind.replace('_', ' ')} alone and in batch: {c['bit_identical']}/{c['rows']} bit-identical; "
                         f"first differing op {json.dumps(c['first_op_counts'])}")
        if "router" in a:
            r = a["router"]
            lines.append(f"- router: {r['rows_with_an_expert_change']}/{a['rows_compared']} rows choose another expert in some layer "
                         f"({r['row_layers_changed']}/{r['row_layers']} row-layers)")
        if "logits" in a:
            g = a["logits"]
            lines.append(f"- logits rows: {g['bit_identical']}/{g['rows_compared']} bit-identical, argmax equal {g['argmax_equal']}, "
                         f"top-1 |delta| max {g['top1_abs_delta']['max']} median {g['top1_abs_delta']['median']}"
                         + (f"; {g['differs_where_final_hidden_equal']} differ where the final hidden row is equal "
                            f"({g['final_hidden_equal']} such rows)" if "final_hidden_equal" in g else ""))
        if a.get("op_variance"):
            lines += ["", "Each op where everything it reads is bit-identical alone and in the batch: rows whose output still differs.",
                      "", "| op | prefill: inputs equal | output differs | decode: inputs equal | output differs |", "| --- | --- | --- | --- | --- |"]
            for op, t in a["op_variance"].items():
                lines.append(f"| `{op.removeprefix('model.layers.N.')}` | {t['inputs_equal_prefill']} | {t['differs_prefill']} | "
                             f"{t['inputs_equal_decode']} | {t['differs_decode']} |")
    g = res["gpu_seconds"]
    lines += ["", f"- GPU-seconds: {g.get('lease_wall_s', g['total_s'])} on one GPU (the lease's wall time; phases "
                  f"{json.dumps(g['per_phase'])})"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--served", default="serve")
    ap.add_argument("--replays", nargs="+", default=["replay1", "replay2", "replay_plain"])
    ap.add_argument("--alone", default="alone")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    wl = json.loads(a.workload.read_text())
    served = Run(a.runs / a.served)
    replays = [Run(a.runs / n) for n in a.replays if (a.runs / n).is_dir()]
    res: dict[str, Any] = {"pairs": {}}
    pairs = [(served, r) for r in replays] + [(x, y) for i, x in enumerate(replays) for y in replays[i + 1:]]
    for x, y in pairs:
        p: dict[str, Any] = {"schedule": schedule(x, y), "tokens": tokens(x, y)}
        if x.captured and y.captured and p["schedule"]["equal"]:
            p["values"] = values(x, y)
        res["pairs"][f"{x.name} vs {y.name}"] = p
    runs = [served, *replays]
    if (a.runs / a.alone).is_dir():
        solo = Run(a.runs / a.alone)
        runs.append(solo)
        res["alone"] = alone(served, solo, wl) if served.captured and solo.captured else {"tokens": tokens(served, solo)}
    res["gpu_seconds"] = gpu_seconds(runs)
    if (lease := a.out / "lease.json").exists():
        res["gpu_seconds"]["lease_wall_s"] = json.loads(lease.read_text())["lease_wall_s"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "compare.json").write_text(json.dumps(res, indent=1))
    (a.out / "compare.md").write_text(render(res))
    print(render(res))


if __name__ == "__main__":
    main()
