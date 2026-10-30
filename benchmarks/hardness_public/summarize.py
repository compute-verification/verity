"""Markdown tables from the runs' outputs: `summarize.py natural DIR`, `tiles FILE`, `seeds FILE`, `tails FILE`,
`poison FILE`."""

import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("h200_b1", "h200_b2", "h200_b8", "sm120_b1", "sm120_b2", "sm120_b8")


def kind(t):
    return t.split(".")[-2].replace("_proj", "")


def natural(d):
    rows = []
    for f in sorted(Path(d).glob("*.json")):
        m = json.loads(f.read_text())
        rows += m["matrices"]
    print("| model | matrices | σ_q (LSB) | min row σ_q | entropy (bits) | " + " | ".join(f"best {k}" for k in KEYS) + " |")
    print("|---|---:|---|---:|---|" + "---:|" * len(KEYS))
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        sig = [r["alphabet"]["sigma_q"] for r in rs]
        ent = [r["alphabet"]["entropy_bits"] for r in rs]
        mins = min(r["alphabet"]["row_sigma_q_quantiles"][0] for r in rs)
        best = [min(r["best_cost"][k] for r in rs) for k in KEYS]
        print(f"| {model} | {len(rs)} | {min(sig):.1f}–{max(sig):.1f} | {mins:.1f} | {min(ent):.2f}–{max(ent):.2f} | "
              + " | ".join(f"{b:.3f}" for b in best) + " |")
    print()
    print("| model | r_min(a=3)/m | r_min(a=4)/m | r_min(a=7)/m | top-m/8 energy | same, row-shuffled | tail rms at m/2 / σ_q |")
    print("|---|---|---|---|---|---|---|")
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        def rmin(a):
            v = [r["eckart_young_rmin"][str(a)] / min(r["shape"]) for r in rs]
            return f"{min(v):.2f}–{max(v):.2f}"
        def et(key):
            v = []
            for r in rs:
                m = min(r["shape"])
                v.append(r[key][str(m // 8)])
            return f"{min(v):.3f}–{max(v):.3f}"
        tr = [r["spectrum"]["tail_rms"][str(min(r["shape"]) // 2)] / r["alphabet"]["sigma_q"] for r in rs]
        print(f"| {model} | {rmin(3)} | {rmin(4)} | {rmin(7)} | "
              f"{et('spectrum') if False else ''}"
              f"{min(r['spectrum']['energy_top'][str(min(r['shape']) // 8)] for r in rs):.3f}–"
              f"{max(r['spectrum']['energy_top'][str(min(r['shape']) // 8)] for r in rs):.3f} | {et('shuffled_energy_top')} | "
              f"{min(tr):.2f}–{max(tr):.2f} |")
    print()
    print("| model | p_odd | max abs(p_odd(abs q≥8) − ½) | max abs(neighbour corr) | min lzma ratio | GF(2) rank deficit (max) | max ±1 top σ / random | max row z | max low-2 dev |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r["lsb"] for r in rows if r["model"] == model]
        po = [x["p_odd"] for x in rs]
        hi = max(max(abs((x.get(f"p_odd_abs_{lo}_{hi}") or 0.5) - 0.5) for lo, hi in ((8, 16), (16, 32), (32, 128))) for x in rs)
        nc = max(max(abs(x["corr_row_neighbour"]), abs(x["corr_col_neighbour"])) for x in rs)
        lz = min(x["lzma_ratio"] for x in rs)
        gd = max(x["gf2_rank"][1] - x["gf2_rank"][0] for x in rs)
        sv = max(x["pm1_top_sv_over_rand"] for x in rs)
        rz = max(x["row_p_odd_max_z"] for x in rs)
        l2 = max(x["low2_max_dev"] for x in rs)
        print(f"| {model} | {min(po):.4f}–{max(po):.4f} | {hi:.4f} | {nc:.4f} | {lz:.4f} | {gd} | {sv:.3f} | {rz:.1f} | {l2:.4f} |")
    print()
    print("Per kind (all models): σ_q mean, min row σ_q, max p_odd(abs q in 1..3) and the best of any r=0 tiled route")
    print("| kind | n | σ_q mean | min row σ_q | tiled r0 h200_b1 (min) | tiled r0 sm120_b2 (min) | frac tiles < 1 (max, sm120_b2) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k in ("q", "k", "v", "o", "gate", "up", "down"):
        rs = [r for r in rows if kind(r["tensor"]) == k]
        print(f"| {k} | {len(rs)} | {np.mean([r['alphabet']['sigma_q'] for r in rs]):.1f} | "
              f"{min(r['alphabet']['row_sigma_q_quantiles'][0] for r in rs):.1f} | "
              f"{min(r['tiled_r0']['h200_b1']['cost'] for r in rs):.3f} | {min(r['tiled_r0']['sm120_b2']['cost'] for r in rs):.3f} | "
              f"{max(r['tiled_r0']['sm120_b2']['frac_tiles_below_1'] for r in rs):.4f} |")
    print()
    best_any = sorted(rows, key=lambda r: min(r["best_cost"][k] for k in KEYS))[:5]
    print("Lowest five matrices by any chip/b:")
    for r in best_any:
        k = min(KEYS, key=lambda k: r["best_cost"][k])
        print(f"- {r['model']} {r['tensor']} {r['shape']}: {k} {r['best_cost'][k]:.4f}; σ_q {r['alphabet']['sigma_q']:.1f}; "
              f"best_svd {r['best_svd'][k]}")


def tiles(f):
    rows = json.loads(Path(f).read_text())
    keys = [k for k in rows[0]["routes"]]
    print("MAC-weighted cost by model and kind, tile 8x256 (H200 b1 / H200 b2 / sm120 b1 / sm120 b2):")
    print("| model | q | k | v | o | gate | up | down | all |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in dict.fromkeys(r["model"] for r in rows):
        cells = []
        for kd in ("q", "k", "v", "o", "gate", "up", "down", None):
            rs = [r for r in rows if r["model"] == model and (kd is None or kind(r["tensor"]) == kd)]
            w = np.array([r["macs"] for r in rs], dtype=float)
            vals = []
            for key in ("8x256_h200_b1", "8x256_h200_b2", "8x256_sm120_b1", "8x256_sm120_b2"):
                vals.append(float((np.array([r["routes"][key]["cost"] for r in rs]) * w).sum() / w.sum()))
            cells.append("/".join(f"{v:.4f}" for v in vals))
        print(f"| {model} | " + " | ".join(cells) + " |")
    print()
    for shape in ("8x256", "16x64", "128x128", "1xK"):
        for key in ("h200_b1", "h200_b2", "sm120_b1", "sm120_b2"):
            w = np.array([r["macs"] for r in rows], dtype=float)
            c = np.array([r["routes"][f"{shape}_{key}"]["cost"] for r in rows])
            fr = max(r["routes"][f"{shape}_{key}"]["frac_tiles_below_1"] for r in rows)
            print(f"{shape} {key}: MAC-weighted gap {100 * (1 - (c * w).sum() / w.sum()):.4f}%, worst matrix {c.min():.4f}, "
                  f"max frac tiles < 1 {fr:.4f}")


def seeds(f):
    r = json.loads(Path(f).read_text())
    print("| arm | val loss |")
    for a, m in r["arms"].items():
        print(f"| {a} | {m['val_loss']:.4f} |")
    print()
    print("Bit-plane agreement against A, mean over the 16 linears (plane 0 = LSB), exact equality, corr, bit-identical:")
    for p, per in r["pairs"].items():
        pl = np.mean([v["planes_agree"] for v in per.values()], axis=0)
        eq = np.mean([v["exact_equal"] for v in per.values()])
        co = np.mean([v["corr"] for v in per.values()])
        bi = all(v["bit_identical_float"] for v in per.values())
        print(f"- {p}: planes {[round(x, 4) for x in pl]}, equal {eq:.4f}, corr {co:.4f}, bit-identical {bi}")
    if r["aligned_A_B"]:
        pl = np.mean([v["planes_agree"] for v in r["aligned_A_B"].values()], axis=0)
        mc = np.mean([v.get("mean_matched_row_corr", np.nan) for k, v in r["aligned_A_B"].items() if k.endswith("fc")])
        co = np.mean([v["corr"] for v in r["aligned_A_B"].values()])
        print(f"- A~B after matching MLP units: planes {[round(x, 4) for x in pl]}, corr {co:.4f}, matched fc row corr {mc:.4f}")
    print()
    print("| arm | corr(final, init) | ‖init‖/‖final‖ | init σ in final LSBs | LSB agreement with rounded init | update top-1/8 energy | Gaussian top-1/8 |")
    print("|---|---|---|---|---|---|---|")
    for a, per in r["retention"].items():
        v = list(per.values())
        print(f"| {a} | {np.mean([x['corr_init'] for x in v]):.3f} | {np.mean([x['init_norm_ratio'] for x in v]):.3f} | "
              f"{np.mean([x['init_sigma_in_final_lsb'] for x in v]):.2f} | {np.mean([x['planes_agree_with_init'][0] for x in v]):.4f} | "
              f"{np.mean([x['update_energy_top']['0.125'] for x in v]):.3f} | {np.mean([x['gauss_energy_top']['0.125'] for x in v]):.3f} |")
    print()
    for tag in ("natural_A", "natural_init"):
        v = list(r[tag].values())
        print(f"{tag}: best cost min over linears " + ", ".join(f"{k} {min(x['best_cost'][k] for x in v):.4f}" for k in KEYS)
              + f"; σ_q {min(x['sigma_q'] for x in v):.1f}–{max(x['sigma_q'] for x in v):.1f}")


def tails(f):
    r = json.loads(Path(f).read_text())
    for k, v in sorted(r["tails"].items(), key=lambda kv: (int(kv[0].split("_")[0]), kv[0])):
        bi = all(x["bit_identical_float"] for x in v["per_matrix"].values())
        print(f"- L={k}: planes {[round(x, 4) for x in v['planes_agree_mean']]}, val {v['val_loss']:.4f}, bit-identical {bi}")


def poison(f):
    for r in json.loads(Path(f).read_text()):
        print(f"- teacher {r['teacher']}, wd {r['wd']}: rel_err {r['rel_err']:.2e}, σ_q {r['sigma_q']:.1f}, "
              f"S hist {r.get('s_hist')}, s_max {r.get('s_max')}, teacher route {r.get('teacher_route')}, "
              f"generic search {r['natural_best']}")


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2])
