"""F_beta.json for the attack optimizer (C1) from counted.py's outputs: beta by model, condition and precision with ranges
over checkpoints; F(S) and the fresh residual's effective rank by S, task and precision; rho at the cliff.

  python summarize.py OUT.json DIR [DIR ...]     (DIRs hold counted_*.json; a run's out/ or a fetched copy)

rho at the cliff = (2/3) * crossings * F / h with h = 17 bits per token (the program's formula, at matched c). F is taken at
the cheapest boundary (the attacker picks where to split). Three readings:
  exact_2x   two crossings, each the exact fresh forward residual (bit-exact recipe, assumption E);
  exact_fb   one forward and one backward crossing, each exact, at the same boundary;
  approx_2x  two crossings of a rank-r correction (fp8 coefficients, basis free) that keeps >= 90% of the shift
             KL(theta'||theta) on the eval text, the capability version.
"""
import glob, json, os, re, sys

FMTS = ("bf16", "fp8t", "nvfp4")
H = 17.0
KEEP = 0.9


def label(r):
    """(family, condition, step) from a result: console conditions (MODEL_cond_sSTEP) or A2/A3 checkpoints (meta)"""
    meta = r.get("meta") or {}
    if "task" in meta or "method" in meta:
        fam = "A2" if "/A2/" in r["ckpt"] else ("A3" if "/A3/" in r["ckpt"] else "ckpt")
        cond = f"{meta.get('task', meta.get('label', '?'))}:{meta.get('method', meta.get('family', '?'))}"
        return fam, cond, meta.get("step")
    m = re.match(r"(.+)_s(\d+)$", r["name"])
    return "sweep", (m.group(1) if m else r["name"]), (int(m.group(2)) if m else None)


def s_bits(r):
    meta = r.get("meta") or {}
    for k in ("S_bits", "s_bits", "S"):
        if isinstance(meta.get(k), (int, float)):
            return meta[k]
    if meta.get("lora_params"):
        return 16 * meta["lora_params"]
    return None


def approx(r):
    """smallest rank whose fp8-coefficient correction keeps >= KEEP of KL(theta'||theta), per probed boundary"""
    out = {}
    k0 = r["loss"]["kl_theta1_theta"]
    for i, ret in r.get("retention", {}).items():
        ranks = ret["ranks"]
        best = None
        for j, rk in enumerate(ranks):
            kl_ = ret["kl"]["exact"][0] if rk == 0 else ret["kl"]["fp8"][j - 1]
            if k0 > 0 and 1 - kl_ / k0 >= KEEP:
                best = rk
                break
        keep = {str(rk): (1 - (ret["kl"]["exact"][0] if rk == 0 else ret["kl"]["fp8"][j - 1]) / k0) if k0 > 0 else None
                for j, rk in enumerate(ranks)}
        out[int(i)] = {"rank_keep90": best, "keep_by_rank_fp8": keep,
                       "nll_by_rank_exact": dict(zip(map(str, ranks), ret["nll"]["exact"]))}
    return out


def entry(r):
    fam, cond, step = label(r)
    s = r["summary"]
    e = {"family": fam, "model": r["model"], "condition": cond, "step": step, "name": r["name"], "S_bits": s_bits(r),
         "tokens": r["tokens"], "n_layers": r["n_layers"], "d": r["boundary"]["0"]["d"] if "0" in r["boundary"] else None,
         "loss_theta": r["loss"]["theta"], "loss_theta1": r["loss"]["theta1"], "kl_theta1_theta": r["loss"]["kl_theta1_theta"],
         "beta": {}, "F": {}, "rank": {k: s[k] for k in ("r90_per_layer", "r99_per_layer", "bwd_r90_per_layer",
                                                       "bwd_r99_per_layer")}}
    for f in FMTS:
        x = s[f]
        e["beta"][f] = {k: x[k] for k in ("beta_v", "beta_v_given_x", "beta_v_per_output", "beta_x", "beta_w",
                                          "beta_w_embed", "update_bits_est")}
        e["beta"][f]["beta_v_by_map"] = x["beta_v_by_map"]
        fb = [a + b for a, b in zip(x["F_fwd_per_layer"], x["F_bwd_per_layer"])]
        e["F"][f] = {"F_fwd_per_layer": [round(v, 1) for v in x["F_fwd_per_layer"]],
                     "F_bwd_per_layer": [round(v, 1) for v in x["F_bwd_per_layer"]],
                     "F_fwd_min": x["F_fwd_min"], "F_fwd_median": x["F_fwd_median"], "F_bwd_min": x["F_bwd_min"],
                     "F_fwd_coef_min": x["F_fwd_coef_min"], "F_fb_min": min(fb)}
    rk = e["rank"]
    i_min = min(range(len(rk["r90_per_layer"])), key=lambda i: e["F"]["fp8t"]["F_fwd_per_layer"][i])
    e["cheapest_boundary_fp8"] = {"layer": i_min, "r90": rk["r90_per_layer"][i_min], "r99": rk["r99_per_layer"][i_min]}
    e["approx"] = approx(r)
    ranks90 = [v["rank_keep90"] for v in e["approx"].values() if v["rank_keep90"] is not None]
    e["rho"] = {}
    for f in FMTS:
        F = e["F"][f]
        e["rho"][f] = {"exact_2x": (2 / 3) * 2 * F["F_fwd_min"] / H, "exact_fb": (2 / 3) * F["F_fb_min"] / H}
    if ranks90:
        r90 = min(ranks90)
        e["rho_approx_2x"] = {"rank": r90, "bits_per_token_fp8": 8 * r90, "rho": (2 / 3) * 2 * 8 * r90 / H}
    return e


def ranges(entries, fam):
    out = {}
    for e in entries:
        if e["family"] != fam:
            continue
        m = out.setdefault(e["model"], {})
        for f in FMTS:
            for k in ("beta_v", "beta_v_given_x", "beta_x", "beta_w"):
                v = e["beta"][f][k]
                lo, hi = m.setdefault(f, {}).get(k, [v, v])
                m[f][k] = [min(lo, v), max(hi, v)]
    return out


def main():
    dst, dirs = sys.argv[1], sys.argv[2:]
    files = sorted({p for d in dirs for p in glob.glob(os.path.join(d, "counted_*.json"))})
    entries = []
    for p in files:
        r = json.load(open(p))
        if "summary" in r:
            entries.append(entry(r))
    doc = {
        "schema": "tvi-b1/F_beta/v1",
        "units": {"beta_*": "bits per value (counted value / input coordinate / weight of an updated matrix)",
                  "F_*": "bits per token at one block boundary (decoder-layer output)", "S_bits": "bits of trainable state"},
        "precisions": {"bf16": "bfloat16", "fp8t": "FP8 E4M3, per-tensor scale from theta", "nvfp4": "NVFP4 (E2M1, 16-blocks, E4M3 scales)"},
        "h_bits_per_token": H,
        "rho_formula": "rho = (2/3) * crossings * F / h at the cheapest boundary; exact_2x, exact_fb, approx_2x (see summarize.py)",
        "beta_ranges_sweep": ranges(entries, "sweep"),
        "beta_ranges_A2": ranges(entries, "A2"),
        "beta_ranges_A3": ranges(entries, "A3"),
        "entries": entries,
    }
    json.dump(doc, open(dst, "w"), indent=1)
    print(f"{len(entries)} entries -> {dst}")


if __name__ == "__main__":
    main()
