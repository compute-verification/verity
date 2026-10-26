"""Exfiltration capacity and skipped work versus proof overhead, sweeping the
total verification budget beta (replay + proofs) and optimizing its split.

Workload model (all inference worldwide, one-year horizon):
  ~2.1e16 requests/year (30M H100-equivalents at 2e15 FP8 FLOP/s, DeepSeek-V4-Pro
  with 49B active parameters, ~4,500 tokens/request, 3.3x/yr growth to Sept 2027),
  so ~4.4e14 FLOP/request; 1e4 FLOP per proof unit (PU; matmuls partitioned into
  dot products), 16-bit PU output interfaces, 2.5 MB output cap per request
  (1M tokens x 20 bits). Whole requests are replay units (RUs).
Sampler: each request is replayed with probability q (replay cost = q of all
compute); within a replayed request each PU is proved independently with
probability s, costing q*s*h of all compute, where h is the proof overhead.
Budget: q + q*s*h <= beta, so s = min(1, (beta - q)/(q*h)). For each (beta, h)
the verifier picks q to minimize the adversary's capacity. Evasion threshold:
1% over the full horizon. Per-request detection risk with m incorrect PUs:
  rho(m) = -ln[1 - q(1 - (1-s)^m)].
"""
from pathlib import Path
import math, csv, json
from multiprocessing import Pool
import numpy as np

OUT = Path(__file__).resolve().parent
FLOP_REQ = 4500 * 2 * 49e9; FLOP_PU = 1e4     # DeepSeek-V4-Pro: ~2 FLOP/active param/token
n = int(FLOP_REQ / FLOP_PU)                    # PUs per request
R = int(1e16*104/49)                           # ~2.1e16 requests over the one-year horizon
b = 16; W = 20_000_000                         # PU interface bits; output bits/request
B = math.log(100); L2 = math.log(2); d = 2**b - 1
TOTAL_BYTES = R * W / 8
H100_HOUR_FLOP = 1979e12 * 3600                # FP8 dense peak at 100% utilization
H_DEMO = 1e8
BETAS = [.001, .01, .1]

def rho(m, q, s):
    m = np.asarray(m, dtype=float)
    if s >= 1: return np.where(m > 0, -math.log1p(-q), 0.)
    return -np.log1p(q * np.expm1(m * math.log1p(-s)))

# ---- adversary optimum. The adversary corrupts j requests, request i carrying m_i
# incorrect PUs, subject to sum_i rho(m_i) <= ln(100). For fixed j this is a
# one-constraint knapsack whose LP optimum mixes at most two corruption levels, i.e.
# j * env(ln(100)/j), where env is the upper concave envelope of (rho(m), value(m))
# over m >= 1. This is exact up to rounding of a single request.
# Exfiltration value(m) = min(W, log2 C(n,m) + b*m) bits, plus log2 C(R, j) positional bits.
# Free compute value(m) = m * FLOP_PU.
mgrid = np.unique(np.r_[np.round(np.geomspace(1, n, 400)), n])
_lg = np.vectorize(math.lgamma)
def lgC(N, k): return (math.lgamma(N+1) - _lg(k+1) - _lg(N-k+1))/L2
LGC_n = lgC(n, mgrid)
JS = np.unique(np.r_[np.round(np.geomspace(1, R, 4000)), R])
LGC_R = lgC(R, JS)

def upper_hull(x, y):
    H = []
    for p in sorted(zip(x, y)):
        while len(H) >= 2 and ((H[-1][0]-H[-2][0])*(p[1]-H[-2][1])
                               - (H[-1][1]-H[-2][1])*(p[0]-H[-2][0])) >= 0: H.pop()
        H.append(p)
    return np.array(H)

def best(r, val, extra=0.):
    H = upper_hull(r, val); xb = B/JS; ok = xb >= H[0, 0]
    env = np.interp(xb, H[:, 0], H[:, 1])        # flat at max value beyond the last point
    return float(((JS*env + extra)[ok]).max())

def evaluate(q, s):
    r = rho(mgrid, q, s)
    bits = np.minimum(W, LGC_n + b*mgrid)
    exfil = min(TOTAL_BYTES, best(r, bits, LGC_R)/8)
    free = min(R*n*FLOP_PU, best(r, mgrid*FLOP_PU))
    return exfil, free

def split(beta, h, q):
    return q, min(1., (beta - q)/(q*h))

def point(args):
    beta, h = args
    qs = beta*np.geomspace(1e-4, 1 - 1e-4, 60)
    vals = [evaluate(*split(beta, h, q)) for q in qs]
    ie = int(np.argmin([v[0] for v in vals])); jf = int(np.argmin([v[1] for v in vals]))
    return dict(beta=beta, proof_overhead=h, exfil_bytes=vals[ie][0], exfil_replay_share=float(qs[ie]),
                free_flop=vals[jf][1], free_replay_share=float(qs[jf]))

hs = np.unique(np.r_[np.geomspace(1, 1e14, 57), H_DEMO])
if __name__ == '__main__':
    cache = OUT/'figure_data.csv'
    if cache.exists():
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(cache))]
    else:
        with Pool(2) as pool:
            rows = pool.map(point, [(beta, float(h)) for beta in BETAS for h in hs])
        with open(cache, 'w') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import offset_copy
    from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator, FuncFormatter
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 24, 'axes.labelsize': 26,
        'xtick.labelsize': 22, 'ytick.labelsize': 22, 'axes.linewidth': 1.0,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.subplots_adjust(left=.09, right=.985, bottom=.13, top=.87, wspace=.32)

    def byte_label(v, pos):
        for unit, s in [(1e15, 'PB'), (1e12, 'TB'), (1e9, 'GB'), (1e6, 'MB')]:
            if v >= unit: return f'{v/unit:g} {s}'
        return f'{v/1e3:g} kB'
    def hour_label(v, pos):
        return '1' if v == 1 else rf'$10^{{{int(round(math.log10(v)))}}}$'
    colors = {.001: '#9ECAE1', .01: '#3182BD', .1: '#08306B'}
    panels = [
        (axes[0], 'exfil_bytes', 1., 'Exfiltration capacity', (5e5, 6e15),
         [1e6, 1e8, 1e10, 1e12, 1e14], FuncFormatter(byte_label),
         [(1000*17e9, '1,000 SparseLoCo syncs'),     # >=17 GB/sync for a 1.6T MoE
          (1.6e12*0.5, 'Model weights'),             # DeepSeek-V4-Pro, 1.6T params at FP4
          (100e6*17/8/10, '100M tokens')]),  # 17-bit tokens, ~10x compression
        (axes[1], 'free_flop', 1/H100_HOUR_FLOP, 'Skipped work (H100-hours)', (1e-4, 1e9),
         [1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8], FuncFormatter(hour_label),
         [(6*49e9*33e12/H100_HOUR_FLOP, 'Pretraining run'),     # 6ND: 49B active, 33T tokens
          (6*49e9*94.4e6/H100_HOUR_FLOP, 'Single optimizer step')]),  # 94.4M-token max batch
    ]
    for ax, key, scale, ylabel, ylim, yt, fmt, refs in panels:
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1, 1e14); ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(FixedLocator([1, 1e3, 1e6, 1e9, 1e12, 1e14]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(which='major', color='#E2E2E2', lw=.45)
        for amount, label in refs:
            ax.axhline(amount, color='#909090', ls=(0, (4, 3)), lw=1.3, zorder=1)
            ax.text(1.5, amount, label, ha='left', va='bottom', fontsize=19, color='#555555',
                    transform=offset_copy(ax.transData, fig=fig, y=3, units='points'))
        ax.axvline(H_DEMO, color='#E69F00', lw=2.2, zorder=2)
        ax.text(H_DEMO*1.6, ylim[0]*2.2, 'Demonstrated today', color='#E69F00',
                fontsize=20, ha='left', va='bottom')
        for beta in BETAS:
            rr = [r for r in rows if r['beta'] == beta]
            ax.plot([r['proof_overhead'] for r in rr], [r[key]*scale for r in rr],
                    color=colors[beta], lw=3.2, zorder=3)
        ax.set_xlabel('Proof overhead', labelpad=8); ax.set_ylabel(ylabel, labelpad=8)
        ax.tick_params(which='major', direction='out', length=5, width=1.0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles = [plt.Line2D([], [], color=colors[b_], lw=3.2) for b_ in BETAS]
    title = fig.text(0, .965, 'Share of compute available for replays and proofs:',
                     ha='left', va='center', fontsize=22)
    leg = fig.legend(handles, [f'{b_*100:g}%' for b_ in BETAS], loc='center left',
                     bbox_to_anchor=(0, .965), ncol=3, frameon=False, fontsize=22,
                     handlelength=2.2, columnspacing=1.6, borderaxespad=0, borderpad=0)
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    tw = inv.transform_bbox(title.get_window_extent(rend)).width
    lw_ = inv.transform_bbox(leg.get_window_extent(rend)).width
    x0 = .53 - (tw + .012 + lw_)/2
    title.set_x(x0); leg.set_bbox_to_anchor((x0 + tw + .012, .965), transform=fig.transFigure)
    fig.savefig(OUT/'exfiltration_and_skipped_work.png', dpi=300, bbox_inches='tight', pad_inches=.09)
    for r in rows:
        if r['proof_overhead'] == H_DEMO:
            print(json.dumps({k: (f'{v:.3g}') for k, v in r.items()}))
