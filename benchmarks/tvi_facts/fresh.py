"""Fresh bits of the counted activations (outputs of every linear map in every layer) that recall needs.

The fine-tuned model and the registered model run the same held-out QA prompts, answers included. For each
map, console's estimators (quant.analyze_pair) give the bits per value of the fine-tuned output given the
registered one: context entropy (H_ctx), low-rank MDL and, on the LZMA_LAYERS' first LZMA_TOKENS tokens, lzma
(lzma over every token takes minutes per map on a GPU slot's 4 cores). Totals are over every
output value (d_out per token) and over counted values (min(d_in, d_out) per token).
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM

import facts
import quant

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FMTS = ("bf16", "fp8t", "mxfp8", "nvfp4")
MDL_RANKS = (1, 4, 16, 64, 256)
N_PEOPLE = int(os.environ.get("FRESH_PEOPLE", "16"))
LZMA_LAYERS = {int(x) for x in os.environ.get("LZMA_LAYERS", "12").split(",") if x}
LZMA_TOKENS = int(os.environ.get("LZMA_TOKENS", "256"))


def _maps(model):
    out = []
    for i, layer in enumerate(model.model.layers):
        for parent in (layer.self_attn, layer.mlp):
            for nm in TARGETS:
                m = getattr(parent, nm, None)
                if m is not None:
                    lin = getattr(m, "base", m)
                    out.append((f"L{i}.{nm}", i, nm, m, lin.in_features, lin.out_features))
    return out


@torch.no_grad()
def capture(model, ids, att):
    acts, hooks = {}, []
    sel = att.bool()
    for key, _, _, m, _, _ in _maps(model):
        def f(mod, inp, out, key=key):
            acts[key] = out[sel].float()
        hooks.append(m.register_forward_hook(f))
    model.model(input_ids=ids, attention_mask=att)
    for h in hooks:
        h.remove()
    return acts


def run(model, adapter, tok, pop, held, exact, res_held, out, model_name):
    from train import collate, tokenize_docs
    items = [it for it in held if it[0] < N_PEOPLE]
    rows = tokenize_docs(tok, [s for _, _, s in items])
    b = collate(rows, tok.pad_token_id)
    ids, att = b["ids"], b["att"]
    model.eval()
    model.float()
    if adapter is not None:
        adapter.quantized = True
        adapter.materialize()
        acts1 = capture(model, ids, att)
        adapter.enable(False)
        acts0 = capture(model, ids, att)
        adapter.enable(True)
    else:
        acts1 = capture(model, ids, att)
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(ids.device).eval()
        acts0 = capture(base, ids, att)
        del base
        torch.cuda.empty_cache()
    quant.to_device(ids.device)
    T = int(att.sum())
    per_map, tot = {}, {f: {"ctx_all": 0.0, "best_all": 0.0, "ctx_counted": 0.0, "best_counted": 0.0} for f in FMTS}
    for key, i, nm, _, din, dout in _maps(model):
        r = quant.analyze_pair(acts0[key], acts1[key], FMTS, mdl_ranks=MDL_RANKS, want_lzma=False)
        if i in LZMA_LAYERS:
            rl = quant.analyze_pair(acts0[key][:LZMA_TOKENS], acts1[key][:LZMA_TOKENS], FMTS, want_lzma=True,
                                    want_svd=False)
            for f in FMTS:
                r[f]["lzma"] = rl[f]["lzma"]
        r.update(layer=i, map=nm, d_in=din, d_out=dout)
        for f in FMTS:
            m = r[f]
            best = min(v for v in (m["H_ctx"], m.get("mdl"), m.get("lzma")) if v is not None)
            m["best"] = best
            tot[f]["ctx_all"] += m["H_ctx"] * T * dout
            tot[f]["best_all"] += best * T * dout
            tot[f]["ctx_counted"] += m["H_ctx"] * T * min(din, dout)
            tot[f]["best_counted"] += best * T * min(din, dout)
        per_map[key] = r
        del acts0[key], acts1[key]
    n = len(pop)
    learned = sum(facts.mi_bits(res_held["acc"][a], facts.CARD[a]) for a in facts.ATTR_NAMES) * min(N_PEOPLE, n)
    by_map = {}
    for nm in TARGETS:
        rs = [r for r in per_map.values() if r["map"] == nm]
        if rs:
            by_map[nm] = {f: {"H_ctx": sum(r[f]["H_ctx"] for r in rs) / len(rs),
                              "best": sum(r[f]["best"] for r in rs) / len(rs)} for f in FMTS}
            by_map[nm]["r90"] = sum(r["r90"] for r in rs) / len(rs)
            by_map[nm]["r99"] = sum(r["r99"] for r in rs) / len(rs)
            by_map[nm]["rel_dnorm"] = sum(r["rel_dnorm"] for r in rs) / len(rs)
    summary = {"name": out["name"], "n": n, "method": out["method"], "S_bits": out["S_bits"], "H_bits": out["H_bits"],
               "questions": len(items), "tokens": T, "bits_learned_on_questions": learned,
               "totals": tot,
               "fresh_per_token": {f: {k: v / T for k, v in tot[f].items()} for f in FMTS},
               "fresh_per_learned_bit": {f: {k: (v / learned if learned > 0 else None) for k, v in tot[f].items()}
                                         for f in FMTS},
               "by_map": by_map}
    os.makedirs(os.path.join(os.environ["OUT"], "fresh"), exist_ok=True)
    json.dump({"summary": summary, "per_map": per_map},
              open(os.path.join(os.environ["OUT"], "fresh", out["name"] + ".json"), "w"), indent=1)
    fp = summary["fresh_per_learned_bit"]
    print(f"{out['name']}: fresh/token fp8 best counted {summary['fresh_per_token']['fp8t']['best_counted']:.0f} "
          f"nvfp4 {summary['fresh_per_token']['nvfp4']['best_counted']:.0f}; learned {learned:.0f} bits on "
          f"{len(items)} q; fresh per learned bit fp8 {fp['fp8t']['best_counted']} nvfp4 {fp['nvfp4']['best_counted']}",
          flush=True)
    if adapter is not None:
        adapter.quantized = False
