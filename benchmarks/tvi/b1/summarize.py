"""F_beta.json for the attack optimizer (C1) from counted.py's outputs: beta by model, condition and precision with ranges
over checkpoints; F(S) and the fresh residual's effective rank by S, task and precision; rho at the cliff.

  python summarize.py OUT_DIR DIR [DIR ...]     (DIRs hold counted_*.json; a run's out/ or a fetched copy)

writes OUT_DIR/F_beta.json (one compact row per checkpoint, small enough to paste) and OUT_DIR/F_beta_layers.json (every
boundary: F forward and backward at each precision, r90 and r99).

rho = (2/3) * crossings * F / h with h = 17 bits per token (the program's formula, at matched c). F is taken at the
cheapest boundary (the attacker picks where to split). Readings:
  rho_2x     two crossings, each the exact fresh forward residual (bit-exact recipe);
  rho_fb     one forward and one backward crossing, each exact, at the same boundary;
  rho_apx    two crossings of a rank-r correction (basis free, coefficients at fp8 or int4 with a 16-bit per-token scale)
             that keeps >= 90% of KL(theta'||theta) on the eval text, at the cheapest of the three probed boundaries
             (L/4, L/2, 3L/4): the capability version. ">256" means no probed rank kept 90%.

beta_w_S = S_bits / N_w: the trained parameterization (public structure) codes the update in S_bits, so where S is the
trainable state (A2's methods, the sweep's LoRA) it bounds beta_w; bw_min is the smaller of it and the compressors'.
"""
import glob, json, os, re, sys

FMTS = ("bf16", "fp8t", "nvfp4")
H = 17.0
KEEP = 0.9
SCALE_BITS = 16
FMT_COLS = ("bv", "bv_gx", "bv_rm", "bx", "bw", "bw_min", "Ff_min", "Ff_med", "Fb_min", "Ffb_min", "rho_2x", "rho_fb")


def label(r):
    """(family, condition, step) from a result: console conditions (MODEL_cond_sSTEP) or A2/A3 checkpoints (meta)"""
    meta = r.get("meta") or {}
    if "task" in meta or "method" in meta:
        fam = "A2" if "/A2/" in r["ckpt"] else ("A3" if "/A3/" in r["ckpt"] else "ckpt")
        cond = ":".join(str(meta[k]) for k in ("task", "method", "S_label", "phase") if k in meta)
        if "seed" in meta:
            cond += f":seed{meta['seed']}"
        step = meta.get("step", meta.get("steps"))
        if meta.get("phase") == "rl":
            step = meta.get("sft_steps", 0) + meta.get("rl_step", 0)
        return fam, cond, step
    m = re.match(r"(.+)_s(\d+)$", r["name"])
    cond = m.group(1) if m else r["name"]
    return "sweep", cond.split("_", 1)[-1], (int(m.group(2)) if m else None)


def s_bits(r):
    """(S in bits, whether S is the trainable state so S/N_w bounds beta_w)"""
    meta = r.get("meta") or {}
    fam = label(r)[0]
    for k in ("S_bits", "s_bits", "S"):
        if isinstance(meta.get(k), (int, float)):
            return meta[k], fam == "A2"
    if meta.get("lora_params"):
        return 16 * meta["lora_params"], True
    return None, False


def n_updated(r):
    """weights in updated matrices over the whole model: the sampled layers' changed maps scaled to every layer"""
    ws = r["weights"]
    lay = sum(w["n"] for k, w in ws.items() if k.startswith("L") and w["changed"] > 0)
    n = lay * r["n_layers"] / len(r["layers_sampled"])
    return n + sum(ws[k]["n"] for k in ("embed", "head") if k in ws and ws[k]["changed"] > 0)


def retention(r):
    """per probed boundary: smallest rank whose correction keeps >= KEEP of KL(theta'||theta), by coefficient format"""
    out = {}
    k0 = r["loss"]["kl_theta1_theta"]
    for i, ret in r.get("retention", {}).items():
        ranks, row = ret["ranks"], {}
        for kind in ("fp8", "int4"):
            keep = {}
            for j, rk in enumerate(ranks):
                kl_ = ret["kl"]["exact"][0] if rk == 0 else ret["kl"][kind][j - 1]
                keep[rk] = 1 - kl_ / k0 if k0 > 0 else None
            ok = [rk for rk, v in keep.items() if v is not None and v >= KEEP]
            row[kind] = {"rank": min(ok) if ok else None, "keep": {str(k): round(v, 3) for k, v in keep.items()}}
        out[int(i)] = row
    return out


def route_min(r, f):
    """beta_v with a second route per map: code the output through its input (the input's fresh bits over the same cones),
    whichever is cheaper. It leaves out the cost of theta''s weights, which a step amortizes over its whole batch."""
    tot = cnt = 0.0
    for m in r["maps"].values():
        tot += min(m["y"][f]["min"], r["inputs"][m["input"]]["x"][f]["min"])
        cnt += m["counted"]
    return tot / cnt


def sig(x, n=3):
    if x is None or isinstance(x, (int, str)) or x == 0:
        return x
    return float(f"{x:.{n}g}")


def entry(r):
    fam, cond, step = label(r)
    s = r["summary"]
    S, s_is_state = s_bits(r)
    nw = n_updated(r)
    e = {"fam": fam, "model": r["model"].split("/")[-1], "cond": cond, "step": step, "S_bits": S,
         "nll": [sig(r["loss"]["theta"], 4), sig(r["loss"]["theta1"], 4)], "kl": sig(r["loss"]["kl_theta1_theta"]),
         "d": r["boundary"]["0"]["d"], "L": r["n_layers"], "N_w_updated": sig(nw)}
    F_fp8 = s["fp8t"]["F_fwd_per_layer"]
    i_min = min(range(len(F_fp8)), key=F_fp8.__getitem__)
    e["cheapest"] = {"layer": i_min, "r90": s["r90_per_layer"][i_min], "r99": s["r99_per_layer"][i_min]}
    rr = sorted(s["r99_per_layer"])
    e["r99_median"] = rr[len(rr) // 2]
    e["r90_median"] = sorted(s["r90_per_layer"])[len(rr) // 2]
    ret = retention(r)
    e["keep90_rank"] = {str(i): [v["fp8"]["rank"], v["int4"]["rank"]] for i, v in ret.items()}
    bwS = S / nw if (S and s_is_state and nw) else None
    e["bw_S"] = sig(bwS)
    for f in FMTS:
        x = s[f]
        fb = [a + b for a, b in zip(x["F_fwd_per_layer"], x["F_bwd_per_layer"])]
        e[f] = {"bv": sig(x["beta_v"]), "bv_gx": sig(x["beta_v_given_x"]), "bv_rm": sig(route_min(r, f)),
                "bx": sig(x["beta_x"]), "bw": sig(x["beta_w"]),
                "bw_min": sig(min(x["beta_w"], bwS) if bwS is not None else x["beta_w"]),
                "Ff_min": round(x["F_fwd_min"]), "Ff_med": round(x["F_fwd_median"]), "Fb_min": round(x["F_bwd_min"]),
                "Ffb_min": round(min(fb)), "rho_2x": sig((2 / 3) * 2 * x["F_fwd_min"] / H),
                "rho_fb": sig((2 / 3) * min(fb) / H)}
    apx = {}
    for kind, bits in (("fp8", 8), ("int4", 4)):
        ranks = [v[kind]["rank"] for v in ret.values() if v[kind]["rank"] is not None]
        if ranks:
            rk = min(ranks)
            F = bits * rk + (SCALE_BITS if rk else 0)
            apx[kind] = {"rank": rk, "F": F, "rho": sig((2 / 3) * 2 * F / H)}
        elif ret:
            top = max(max(v[kind]["keep"], key=int) for v in ret.values())
            apx[kind] = {"rank": f">{top}", "F": f">{bits * int(top) + SCALE_BITS}",
                         "rho": f">{sig((2 / 3) * 2 * (bits * int(top) + SCALE_BITS) / H)}"}
    e["rho_apx"] = apx
    layers = {"name": r["name"], "code_sha256": r.get("code_sha256"), "eval_text": r.get("eval_text"),
              "tokens": r["tokens"], "r90": s["r90_per_layer"], "r99": s["r99_per_layer"],
              "bwd_r90": s["bwd_r90_per_layer"], "bwd_r99": s["bwd_r99_per_layer"], "retention": ret}
    for f in FMTS:
        layers[f] = {"F_fwd": [round(v) for v in s[f]["F_fwd_per_layer"]],
                     "F_bwd": [round(v) for v in s[f]["F_bwd_per_layer"]],
                     "beta_v_by_map": {k: sig(v) for k, v in s[f]["beta_v_by_map"].items()}}
    return e, layers


def ranges(entries):
    out = {}
    for e in entries:
        m = out.setdefault(f"{e['fam']}:{e['model']}", {"n": 0})
        m["n"] += 1
        for f in FMTS:
            for k in ("bv", "bv_rm", "bv_gx", "bx", "bw", "bw_min"):
                v = e[f][k]
                lo, hi = m.setdefault(f, {}).get(k, [v, v])
                m[f][k] = [min(lo, v), max(hi, v)]
    return out


def cliff(entries):
    """F(S) and rho by S within each family and task: rows (S_bits, cond, step, Ff_min fp8/nvfp4, apx rank, rho)"""
    out = {}
    for e in entries:
        if e["S_bits"] is None:
            continue
        task = e["cond"].split(":")[0] if e["fam"] != "sweep" else e["model"]
        out.setdefault(f"{e['fam']}:{task}", []).append(
            [e["S_bits"], e["cond"], e["step"], e["kl"], e["fp8t"]["Ff_min"], e["nvfp4"]["Ff_min"],
             e["rho_apx"].get("fp8", {}).get("rank"), e["fp8t"]["rho_2x"], e["nvfp4"]["rho_2x"],
             e["rho_apx"].get("fp8", {}).get("rho"), e["rho_apx"].get("int4", {}).get("rho")])
    for v in out.values():
        v.sort(key=lambda row: (row[0], str(row[1]), row[2] or 0))
    return {"columns": ["S_bits", "cond", "step", "kl", "Ff_min_fp8", "Ff_min_nvfp4", "keep90_rank_fp8", "rho_2x_fp8",
                        "rho_2x_nvfp4", "rho_apx_fp8", "rho_apx_int4"], "rows": out}


def main():
    dst, dirs = sys.argv[1], sys.argv[2:]
    os.makedirs(dst, exist_ok=True)
    files = sorted({p for d in dirs for p in glob.glob(os.path.join(d, "counted_*.json"))})
    entries, layers, seen = [], {}, set()
    for p in files:
        r = json.load(open(p))
        if "summary" not in r or r["name"] in seen:
            continue
        seen.add(r["name"])
        e, lay = entry(r)
        entries.append(e)
        layers[r["name"]] = lay
    entries.sort(key=lambda e: (e["fam"], e["model"], e["cond"], e["step"] or 0))
    rng, clf = ranges(entries), cliff(entries)
    for e in entries:
        for f in FMTS:
            e[f] = [e[f][k] for k in FMT_COLS]
    doc = {
        "schema": "tvi-b1/F_beta/v2",
        "units": {"b*": "bits per value: bv per counted value (min(d_in,d_out) per token per map, one per dependency cone), "
                        "bx per input coordinate, bw per weight of an updated matrix",
                  "F*": "bits per token at one block boundary (decoder-layer output); Ff forward, Fb backward",
                  "S_bits": "bits of trainable state (A2, sweep LoRA at 16 bits per factor entry) or A3's skill size"},
        "keys": {"bv_gx": "output given theta's weights on theta''s input", "bv_rm": "per map, cheaper of output and "
                 "input route", "bw_S": "S_bits / N_w_updated", "cheapest": "fp8-cheapest boundary and its fresh "
                 "residual's effective rank (90%/99% energy)", "keep90_rank": "per probed boundary, [fp8, int4] "
                 "smallest rank keeping 90% of KL(theta'||theta); null = none up to 256"},
        "precisions": {"bf16": "bfloat16", "fp8t": "FP8 E4M3, per-tensor scale from theta",
                       "nvfp4": "NVFP4 (E2M1, 16-blocks, E4M3 scales)"},
        "h_bits_per_token": H,
        "rho": "(2/3) * crossings * F / h at the cheapest boundary: rho_2x exact fwd twice, rho_fb exact fwd+bwd, "
               "rho_apx rank-r correction keeping 90% of the shift (see summarize.py)",
        "beta_ranges": rng,
        "cliff": clf,
        "entry_format_columns": list(FMT_COLS),
        "entries": entries,
    }
    with open(os.path.join(dst, "F_beta.json"), "w") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: " + (
            "[\n" + ",\n".join(json.dumps(x, separators=(",", ":")) for x in v) + "\n]" if k == "entries"
            else json.dumps(v, separators=(",", ":"))) for k, v in doc.items()) + "\n}\n")
    json.dump(layers, open(os.path.join(dst, "F_beta_layers.json"), "w"), separators=(",", ":"))
    print(f"{len(entries)} entries -> {dst}/F_beta.json, F_beta_layers.json")


if __name__ == "__main__":
    main()
