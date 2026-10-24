"""markdown tables for the readout from F_beta.json and F_beta_layers.json

  python readout.py DIR [ENTRY_NAME] [--all]   DIR holds summarize.py's output; ENTRY_NAME adds its per-layer table,
                                               --all adds the sweep's full fine-tuning rows to the F(S) table
"""
import collections, json, sys

d = json.load(open(sys.argv[1] + "/F_beta.json"))
lay = json.load(open(sys.argv[1] + "/F_beta_layers.json"))
C = d["entry_format_columns"]
H = d["h_bits_per_token"]
F3 = ("bf16", "fp8t", "nvfp4")


def rng(v):
    lo, hi = v
    f = lambda x: f"{x:.2g}" if abs(x) < 0.01 and x else f"{x:.3g}"
    return f(lo) if lo == hi else f"{f(lo)}–{f(hi)}"


print("### beta by model and precision (ranges over checkpoints)\n")
print("| family:model | n | precision | beta_v | beta_v given x | beta_v route-min | beta_x | beta_w (compressors) | beta_w (min with S/N_w) |")
print("|---|---|---|---|---|---|---|---|---|")
for k, v in d["beta_ranges"].items():
    for f in F3:
        x = v[f]
        print(f"| {k} | {v['n']} | {f} | {rng(x['bv'])} | {rng(x['bv_gx'])} | {rng(x['bv_rm'])} | {rng(x['bx'])} | {rng(x['bw'])} | {rng(x['bw_min'])} |")

print("\n### F(S) on A2 (RL, Qwen2.5-1.5B-Instruct) and the sweep's LoRA, by S\n")
print("| task | method | S bits | ckpts | KL | F_min fp8 | F_min nvfp4 | F_min bf16 | F_median fp8 | cheapest layer r90/r99 | median r90/r99 | rank keeping 90% at L/2 (fp8) | rho_2x fp8 | rho_2x nvfp4 | rho_apx min / mid (fp8) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
g = collections.OrderedDict()
for e in sorted(d["entries"], key=lambda e: (e["fam"], e["cond"].split(":")[0], e["S_bits"] or 0)):
    if e["fam"] == "A2":
        task, meth = e["cond"].split(":")[:2]
    elif e["fam"] == "A3":
        task, meth = e["cond"].split(":")[0], ":".join(e["cond"].split(":")[1:])
    else:
        task, meth = e["model"], e["cond"]
    g.setdefault((e["fam"], task, meth), []).append(e)


def span(xs, fmt="{:.0f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    lo, hi = min(xs), max(xs)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


def num(x):
    return x if not isinstance(x, str) else float("inf")


for (fam, task, meth), es in g.items():
    if fam == "sweep" and meth != "lora" and "--all" not in sys.argv:
        continue
    col = lambda f, c: [dict(zip(C, e[f]))[c] for e in es]
    rho2 = lambda f: [(2 / 3) * 2 * x / H for x in col(f, "Ff_min")]
    ch = [(e["cheapest"]["r90"], e["cheapest"]["r99"]) for e in es]
    ap = [e.get("apx", {}).get("fp8") for e in es]
    midr = sorted({str(a[3]) for a in ap if a}, key=lambda s: (s.startswith(">"), float(s.lstrip(">"))))
    def aspan(ranks, bits=8):
        ok = [(2 / 3) * 2 * (bits * r + 16) / H for r in ranks if r is not None and not isinstance(r, str)]
        gt = [float(r[1:]) for r in ranks if isinstance(r, str)]
        s = span(ok) if ok else ""
        return (s + (", " if s else "") + f">{(2 / 3) * 2 * (bits * max(gt) + 16) / H:.0f}") if gt else (s or "-")
    apx = f"{aspan([a[1] for a in ap if a])} / {aspan([a[3] for a in ap if a])}"
    print(f"| {fam}:{task} | {meth} | {es[0]['S_bits'] if es[0]['S_bits'] is not None else '-'} | {len(es)} | "
          f"{span([e['kl'] for e in es], '{:.3g}')} | {span(col('fp8t', 'Ff_min'))} | {span(col('nvfp4', 'Ff_min'))} | "
          f"{span(col('bf16', 'Ff_min'))} | {span(col('fp8t', 'Ff_med'))} | "
          f"{span([c[0] for c in ch])}/{span([c[1] for c in ch])} | {span([e['r90_median'] for e in es])}/{span([e['r99_median'] for e in es])} | "
          f"{','.join(midr) or '-'} | {span(rho2('fp8t'))} | {span(rho2('nvfp4'))} | {apx} |")

if len(sys.argv) > 2 and sys.argv[2] in lay:
    L = lay[sys.argv[2]]
    print(f"\n### per layer: {sys.argv[2]}\n")
    print("| layer | r90 | r99 | F fwd bf16 | F fwd fp8 | F fwd nvfp4 | F bwd fp8 |")
    print("|---|---|---|---|---|---|---|")
    for i in range(len(L["r90"])):
        print(f"| {i} | {L['r90'][i]} | {L['r99'][i]} | {L['bf16']['F_fwd'][i]} | {L['fp8t']['F_fwd'][i]} | {L['nvfp4']['F_fwd'][i]} | {L['fp8t']['F_bwd'][i]} |")
