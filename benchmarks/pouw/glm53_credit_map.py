#!/usr/bin/env python3
"""GLM-5.3's PoUW credit map at TP8: every linear of the served step, its per-rank (n, k), its share of the step's MACs, and
whether each scheme's own code admits it (Pearl-C's `PearlC.check`, H1's `h1window.program` over `h1window.tile_unit` at
the architecture's `h1device` record), with the credited share per scheme and for the best of both on one device.

* **Shapes** are derived from the checkpoint's ``config.json`` at TP8 (vLLM's sharding: column-parallel linears take a
  rank's rows, row-parallel ones a rank's K slice, replicated ones are whole on every rank) and held to the served
  inventory's rank-0 shapes (`integrations/vllm/tests/check/fixtures/glm_moe_dsa/inventory-tp8-rank0.json`, run
  r20261003-072626-b966 on an H200).  The run fails when any served linear's shape disagrees.
* **The step** is one decode step on vLLM's sparse-mqa route, per token and per rank: each linear's n·k MACs times its
  layers, times 8 for the routed experts (top-8) and the absorbed attention BMMs (8 heads a rank).  At decode kv_b_proj
  is absorbed (`q_nope_absorb`, `v_absorb`, BF16 BMMs, one per head); on the dense-mha prefill route it is the linear
  kv_b_proj instead, reported as the prefill step (lm_head left out: one row per sequence).  Attention's score and value
  products over the context are not linears and are outside both steps.  The MTP layer (layer 78) is not in the served
  inventory: its rows are listed and kept out of the step.
* **Credit.**  Pearl-C credits a linear its `check` admits; H1 one its window admits that the int8 recipe
  (`verity-int8-channel127/v3`, the GLM53_INT8 pin) serves as an s8 GEMM: kv_b_proj (so its absorbed BMMs), the
  indexer's wk and BF16 modules stay BF16 there, and the MTP layer isn't served.  Routed experts' w2 at TP8 is bound and
  uncredited under every scheme (Daniel's card 82c1e288, (c)).  Model-level gates (registration, pins) are not per linear
  and are not applied here.

    uv run python benchmarks/pouw/glm53_credit_map.py --config SNAPSHOT/config.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_result  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as W  # noqa: E402
from verity.core.protocols.pouw.protocol import Matmul, Shape  # noqa: E402
from verity.core.protocols.pouw.schemes import get  # noqa: E402
from verity.core.protocols.pouw.schemes.pearl_c import PearlC  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "integrations/vllm/tests/check/fixtures/glm_moe_dsa/inventory-tp8-rank0.json"
SNAP = "/workspace/hf/hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42"
TP = 8
M = 64                       # a call's rows: any multiple of 64 passes both schemes' row rules
H1_TILE = (32, 32)           # the served H1 tile (`pouw_h1.TILE`)
CARD = "Daniel's card 82c1e288 (c): routed w2 at TP8 is bound and uncredited"
SCHEMES = ("pearl-c-h100-v1-h3", "h1-sm_90", "pearl-c-sm120-v1-h3", "h1-sm_120")
DEVICES = {"H200": ("pearl-c-h100-v1-h3", "h1-sm_90"), "RTX PRO 6000": ("pearl-c-sm120-v1-h3", "h1-sm_120")}


def linears(cfg: dict) -> list[dict]:
    """Every linear of the served model at TP8, per rank (module docstring)."""
    h, heads = cfg["hidden_size"], cfg["num_attention_heads"] // TP
    nope, rope, v = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    qlr, kvlr = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    layers = list(range(cfg["num_hidden_layers"]))
    dense = [i for i in layers if cfg["mlp_layer_types"][i] == "dense"]
    moe = [i for i in layers if cfg["mlp_layer_types"][i] == "sparse"]
    idx = [i for i in layers if cfg["indexer_types"][i] == "full"]
    mtp = list(range(cfg["num_hidden_layers"], cfg["num_hidden_layers"] + cfg["num_nextn_predict_layers"]))
    top, inter, moe_i = cfg["num_experts_per_tok"], cfg["intermediate_size"] // TP, cfg["moe_intermediate_size"] // TP
    ih, idim = cfg["index_n_heads"], cfg["index_head_dim"]

    def row(id_, ls, n, k, *, mult=1, dtype="fp8", int8=True, route="both", inv=None, part="", note=""):
        return {"id": id_, "layers": ls, "n": n, "k": k, "mult": mult, "dtype": dtype, "int8_v3": int8, "route": route,
                "inventory": inv, "parallel": part, "note": note}

    a = "model.layers.{i}.self_attn."
    rows = [
        row("attn.fused_qkv_a (q_a + kv_a_mqa)", layers, qlr + kvlr + rope, h, part="replicated",
            inv=(a + "fused_qkv_a_proj", "weight", [qlr + kvlr + rope, h]),
            note=f"one GEMM: q_a n = {qlr}, kv_a_proj_with_mqa n = {kvlr + rope}"),
        row("attn.q_b", layers, heads * (nope + rope), qlr, part="column", inv=(a + "q_b_proj", "weight", [heads * (nope + rope), qlr])),
        row("attn.q_nope_absorb (W_UK_T bmm)", layers, kvlr, nope, mult=heads, dtype="bf16", int8=False, route="decode",
            inv=(a + "kv_b_proj", "weight", [heads * (nope + v), kvlr]), part="column (per head)",
            note="kv_b_proj's K half, dequantized to BF16 at load; one GEMM a head"),
        row("attn.v_absorb (W_UV bmm)", layers, v, kvlr, mult=heads, dtype="bf16", int8=False, route="decode",
            part="column (per head)", note="kv_b_proj's V half, BF16; one GEMM a head"),
        row("attn.kv_b (dense-mha prefill)", layers, heads * (nope + v), kvlr, int8=False, route="prefill", part="column",
            note="a linear only on the dense-mha route; BF16 under int8 v3"),
        row("attn.o", layers, h, heads * v, part="row", inv=(a + "o_proj", "weight", [h, heads * v])),
        row("indexer.wk_weights_proj", idx, idim + ih, h, dtype="bf16", int8=False, part="replicated",
            inv=(a + "indexer.wk_weights_proj", "weight", [idim + ih, h]), note=f"wk n = {idim} and weights_proj n = {ih}, one GEMM"),
        row("indexer.wq_b", idx, ih * idim, qlr, part="replicated", inv=(a + "indexer.wq_b", "weight", [ih * idim, qlr])),
        row("dense.gate_up", dense, 2 * inter, h, part="column", inv=("model.layers.{i}.mlp.gate_up_proj", "weight", [2 * inter, h])),
        row("dense.down", dense, h, inter, part="row", inv=("model.layers.{i}.mlp.down_proj", "weight", [h, inter])),
        row("moe.router", moe, cfg["n_routed_experts"], h, dtype="bf16", int8=False, part="replicated",
            inv=("model.layers.{i}.mlp.gate", "weight", [cfg["n_routed_experts"], h])),
        row("shared.gate_up", moe, 2 * moe_i, h, part="column",
            inv=("model.layers.{i}.mlp.shared_experts.gate_up_proj", "weight", [2 * moe_i, h])),
        row("shared.down", moe, h, moe_i, part="row", inv=("model.layers.{i}.mlp.shared_experts.down_proj", "weight", [h, moe_i])),
        row("routed.w13", moe, 2 * moe_i, h, mult=top, part="column",
            inv=("model.layers.{i}.mlp.experts.routed_experts", "w13_weight", [cfg["n_routed_experts"], 2 * moe_i, h])),
        row("routed.w2", moe, h, moe_i, mult=top, part="row",
            inv=("model.layers.{i}.mlp.experts.routed_experts", "w2_weight", [cfg["n_routed_experts"], h, moe_i])),
        row("lm_head", [-1], cfg["vocab_size"] // TP, h, dtype="bf16", int8=True, part="column",
            inv=("lm_head", "weight", [cfg["vocab_size"] // TP, h]), note="BF16 in the FP8 serve, int8 under v3"),
    ]
    m = []
    for r in rows:
        if r["route"] != "prefill" and r["id"] != "lm_head" and not r["id"].startswith(("dense.", "indexer.")):
            m.append({**r, "id": "mtp." + r["id"], "layers": mtp, "inventory": None, "int8_v3": False, "route": "mtp"})
    m += [row("mtp.indexer.wk_weights_proj", mtp, idim + ih, h, dtype="bf16", int8=False, route="mtp", part="replicated"),
          row("mtp.indexer.wq_b", mtp, ih * idim, qlr, int8=False, route="mtp", part="replicated"),
          row("mtp.eh_proj", mtp, h, 2 * h, dtype="bf16", int8=False, route="mtp", part="replicated",
              note="BF16 (modules_to_not_convert)"),
          row("mtp.lm_head", [-1], cfg["vocab_size"] // TP, h, dtype="bf16", int8=False, route="mtp", part="column",
              note="the shared head, one more call")]
    return rows + m


def held_to_inventory(rows: list[dict]) -> list[str]:
    inv = json.loads(INVENTORY.read_text())["ranks"]["0"]["groups"]
    shapes = {}
    for g in inv:
        for s in g["shapes"]:
            for t, v in s["tensors"].items():
                shapes[(g["pattern"], t)] = (v.get("shape"), s["layers"])
    bad = []
    for r in rows:
        if r["inventory"] is None:
            continue
        pat, t, want = r["inventory"]
        got, ls = shapes.get((pat, t), (None, None))
        if got != want or (r["layers"] != [-1] and sorted(ls) != sorted(r["layers"]) and "kv_b_proj" not in pat):
            bad.append(f"{r['id']}: inventory {pat}.{t} {got} on {ls and len(ls)} layers, derived {want} on {len(r['layers'])}")
    return bad


def pearl_c(name: str):
    if name == "pearl-c-h100-v1-h3":
        s = PearlC(forming="v1", hashing="h3")          # main's registration of the name (`schemes.SCHEMES`)
        assert s.name == name
        return s
    return get(name)


def pearl_reason(s, k: int, n: int) -> str:
    """Which of `PearlC.check`'s clauses a shape fails (the check itself decides; this only names the clause)."""
    d, why = s.device, []
    if not (s.k_min <= k or k in getattr(s, "census_ks", ())):
        why.append(f"k = {k} < k_min {s.k_min}")
    if k > d.k_max:
        why.append(f"k > {d.k_max}")
    if k % d.window:
        why.append(f"{d.window} ∤ k")
    if n % s.tile[1]:
        why.append(f"n = {n} not whole {s.tile[1]}-column tiles")
    return "; ".join(why) or "refused"


CACHE: dict = {}


def admits(scheme: str, k: int, n: int) -> tuple[bool, str]:
    """The scheme's own code on one call of a (k, n) linear: (admitted, why not)."""
    key = (scheme, k, n)
    if key in CACHE:
        return CACHE[key]
    if scheme.startswith("pearl-c"):
        s = pearl_c(scheme)
        try:
            s.check(Matmul(0, Shape(M, k, n), 0))
            out = (True, "")
        except Exception as e:  # DomainError
            out = (False, pearl_reason(s, k, n) + f" ({type(e).__name__})")
    else:
        arch = scheme.removeprefix("h1-")
        try:
            W.program([("x", k, n)], [("x", 0)], W.slot(M, H1_TILE), H1_TILE,
                      unit=lambda kk: W.tile_unit(kk, H1_TILE, arch=arch))
            out = (True, "")
        except ValueError as e:
            msg = str(e)
            out = (False, f"K = {k} outside sm_90's domain (K >= 384, 512 | K)" if "outside H1's domain" in msg
                   else f"n = {n} not whole {H1_TILE[1]}-row blocks" if "whole blocks" in msg else msg[:120])
    CACHE[key] = out
    return out


def credit(r: dict, scheme: str) -> tuple[bool, str]:
    ok, why = admits(scheme, r["k"], r["n"])
    if not ok:
        return False, why
    if r["id"].endswith("routed.w2"):
        return False, "code admits; " + CARD
    if scheme.startswith("h1") and not r["int8_v3"]:
        return False, "code admits; not an s8 GEMM under int8 v3 (" + (
            "MTP not served" if r["route"] == "mtp" else "BF16") + ")"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{SNAP}/config.json")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text())
    rows = linears(cfg)
    bad = held_to_inventory(rows)
    for r in rows:
        r["macs"] = r["n"] * r["k"] * r["mult"] * len(r["layers"])
        r["schemes"] = {}
        for s in SCHEMES:
            c, why = credit(r, s)
            r["schemes"][s] = {"admitted": admits(s, r["k"], r["n"])[0], "credited": c, "why": why}
    steps = {"decode": [r for r in rows if r["route"] in ("both", "decode")],
             "prefill": [r for r in rows if r["route"] in ("both", "prefill") and r["id"] != "lm_head"],
             "decode+mtp": [r for r in rows if r["route"] in ("both", "decode", "mtp")]}
    summary = {}
    for step, rs in steps.items():
        tot = sum(r["macs"] for r in rs)
        row = {"macs_per_token_per_rank": tot}
        for s in SCHEMES:
            row[s] = float(Fraction(sum(r["macs"] for r in rs if r["schemes"][s]["credited"]), tot))
            row[s + " (code admits)"] = float(Fraction(sum(r["macs"] for r in rs if r["schemes"][s]["admitted"]), tot))
        for dev, (p, h) in DEVICES.items():
            row[f"mix {dev}"] = float(Fraction(sum(r["macs"] for r in rs if r["schemes"][p]["credited"] or r["schemes"][h]["credited"]), tot))
        row["neither (H200)"] = [r["id"] for r in rs if not (r["schemes"]["pearl-c-h100-v1-h3"]["credited"] or r["schemes"]["h1-sm_90"]["credited"])]
        summary[step] = row
    dec = sum(r["macs"] for r in steps["decode"])
    for r in rows:
        r["share_decode"] = r["macs"] / dec if r["route"] in ("both", "decode") else None
        r["share_vs_decode"] = r["macs"] / dec
    doc = {"tool": "glm53_credit_map", "config": a.config, "inventory": str(INVENTORY.relative_to(ROOT)), "tp": TP,
           "call_rows": M, "h1_tile": list(H1_TILE), "inventory_mismatches": bad, "summary": summary,
           "linears": [{x: v for x, v in r.items() if x != "inventory"} for r in rows], "seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "credit_map.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "credit_map.md").write_text(table(doc))
    print(table(doc))
    d = summary["decode"]
    run_result.write(out / "credit_map.json", not bad, "glm53-credit-map",
                     "shapes agree with the served inventory" if not bad else f"inventory mismatches: {bad}",
                     [(f"decode {k}", v, "fraction") for k, v in d.items() if isinstance(v, float)])


def table(doc: dict) -> str:
    def cell(r, s):
        x = r["schemes"][s]
        return "yes" if x["credited"] else ("no: " + x["why"])
    lines = ["| linear | layers | per-rank (n, k) | x/token | MACs, share of decode step | " + " | ".join(SCHEMES) + " |",
             "|---|---:|---|---:|---:|" + "---|" * len(SCHEMES)]
    for r in doc["linears"]:
        share = f"{100 * r['share_decode']:.2f}%" if r["share_decode"] is not None else f"({100 * r['share_vs_decode']:.2f}%, {r['route']})"
        lines.append(f"| {r['id']} | {len(r['layers'])} | ({r['n']}, {r['k']}) | {r['mult']} | {share} | "
                     + " | ".join(cell(r, s) for s in SCHEMES) + " |")
    lines += ["", "| step | MACs/token/rank | " + " | ".join(SCHEMES) + " | mix H200 | mix RTX PRO 6000 |",
              "|---|---:|" + "---:|" * (len(SCHEMES) + 2)]
    for step, s in doc["summary"].items():
        lines.append(f"| {step} | {s['macs_per_token_per_rank']:,} | " + " | ".join(f"{100 * s[x]:.2f}%" for x in SCHEMES)
                     + f" | {100 * s['mix H200']:.2f}% | {100 * s['mix RTX PRO 6000']:.2f}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
