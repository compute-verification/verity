"""Figure 2 of the SPOT post: proof overhead per proof-block type for a model, from its `proof_overhead` card joined to
its `workload` card on the proof block's Definition.

Each of the model's proof-block types shows its proof overhead at a full statement where the card measures it and is
pending where it doesn't, beside its share of a token's proof blocks and MACs, which weight the model's overhead.  A
measured block that isn't one of the model's (`stands_in`) shows below them, marked as a stand-in.  Once every type is
measured, the model's overhead follows from the blocks (`derived_model_overhead`): a token's marginal proving
GPU-seconds, statements' fixed costs left to the design's fill (Figure 1's `Fill`), over its MACs at the card's peak,
the `proof_overhead` card's `model_overhead` and Figure 1's vertical line.
"""
from __future__ import annotations

import re
from math import ceil, floor, log10

import cards as spot_cards

SCHEMA = "spot-figure2/v1"


def short(block: str) -> str:
    """A Definition's name and K, for a bar's label."""
    k = re.search(r"[{,]K=(\d+)", block)
    return block.split("{")[0] + (f" K={k.group(1)}" if k else "")


def seconds(b: dict) -> float:
    """Proving GPU-seconds a block at a full statement."""
    return b["seconds_per_block"] + b["fixed_seconds"] / b["statement_blocks"]


def run(model: str) -> dict:
    work, po = spot_cards.load("workload", model), spot_cards.load("proof_overhead", model)
    peak = po["value"]["peak_mac_s"]
    measured = {b["block"]: b for b in po["value"]["blocks"]}
    strata = work["value"]["strata"]
    blocks = sum(s["per_token"] for s in strata)
    macs = sum(s["per_token"] * s["macs"] for s in strata)
    served = work["value"].get("served_macs_per_token") or macs
    types = []
    for s in sorted(strata, key=lambda s: (-s["per_token"] * s["macs"], -s["per_token"])):
        b = measured.get(s["block"])
        types.append({"block": s["block"], "label": short(s["block"]), "kind": s.get("kind"),
                      "per_token": s["per_token"], "macs": s["macs"], "out_bits": s["out_bits"],
                      "modeled": bool(s.get("modeled")),
                      "block_share": s["per_token"] / blocks, "mac_share": s["per_token"] * s["macs"] / served,
                      "measured": b is not None, "overhead": None if b is None else b["overhead"],
                      "seconds_per_token": None if b is None else s["per_token"] * seconds(b)})
    shared = {t["label"] for t in types if sum(u["label"] == t["label"] for u in types) > 1}
    for t in types:
        if t["label"] in shared:
            t["label"] += f" ({t['kind']})"
    done = all(t["measured"] for t in types)
    return {
        "schema": SCHEMA, "model": model, "cards": {"workload": work, "proof_overhead": po},
        "device": po["value"]["device"], "format": po["value"]["format"], "peak_mac_s": peak,
        "definition": po["value"]["definition"], "types": types,
        "stand_ins": [b | {"label": short(b["block"])} for b in po["value"]["blocks"] if b.get("stands_in")],
        "pending_mac_share": sum(t["mac_share"] for t in types if not t["measured"]),
        "pending_block_share": sum(t["block_share"] for t in types if not t["measured"]),
        "uncovered_mac_share": 1 - macs / served,
        "model_overhead": po["value"]["model_overhead"], "label": po["value"]["label"],
        "derived_model_overhead": sum(t["per_token"] * measured[t["block"]]["seconds_per_block"] for t in types)
        * peak / macs if done else None,
    }


def render(out: dict, outdir):
    """Figure 2's PNG and SVG: the overhead per type, and each type's share of a token's blocks and MACs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["types"] + [s | {"stand_in": True, "measured": True} for s in out["stand_ins"]]
    labels = [r["label"] + ("  (stand-in)" if r.get("stand_in") else "  (modeled)" if r.get("modeled") else "")
              for r in rows]
    n = len(rows)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 13, "savefig.facecolor": "white"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 1.6 + .34 * n), sharey=True,
                               gridspec_kw={"width_ratios": (1.25, 1)})
    fig.subplots_adjust(left=.27, right=.98, top=1 - .9 / (1.6 + .34 * n), bottom=.5 / (1.6 + .34 * n), wspace=.06)
    ys = list(range(n))[::-1]
    span = [r["overhead"] for r in rows if r["measured"] and r["overhead"]] + [out["model_overhead"]]
    lo, hi = 10.0 ** (floor(log10(min(span))) - 1), 10.0 ** (ceil(log10(max(span))) + 1)
    for y, r in zip(ys, rows):
        if r["measured"] and r["overhead"]:
            a.barh(y, r["overhead"], color="#3182BD", hatch="//" if r.get("stand_in") else None,
                   edgecolor="white" if r.get("stand_in") else None, height=.7)
            a.text(r["overhead"] * 1.15, y, f"{r['overhead']:.2g}", va="center", fontsize=11)
        else:
            a.text(lo * 1.3, y, "pending", va="center", fontsize=11, color="#909090", style="italic")
    a.axvline(out["model_overhead"], color="#E69F00", lw=1.8)
    a.text(out["model_overhead"] * 1.1, ys[0] + .6, f"{out['label']}, marginal", color="#E69F00", fontsize=11,
           va="bottom")
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.set_yticks(ys, labels, fontsize=11)
    a.set_xlabel(f"Proof overhead at a full statement ({out['device']}, {out['format']} peak)")
    for share, color, name in (("block_share", "#9ECAE1", "proof blocks"), ("mac_share", "#08306B", "MACs")):
        off = .18 if share == "block_share" else -.18
        b.barh([y + off for y, r in zip(ys, rows) if share in r], [r[share] for r in rows if share in r],
               height=.34, color=color, label=f"share of a token's {name}")
    b.set_xscale("log")
    b.set_xlim(1e-7, 1.5)
    b.legend(loc="lower right", frameon=False, fontsize=11)
    b.set_xlabel("Share of a token")
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#E2E2E2", lw=.45)
    gap = f", {out['uncovered_mac_share']:.1%} in no proof block" if out["uncovered_mac_share"] > 0 else ""
    fig.suptitle(f"Proof overhead per proof-block type, {out['model']}: {sum(r['measured'] for r in out['types'])} of "
                 f"{len(out['types'])} types measured, {out['pending_mac_share']:.1%} of a token's MACs pending{gap}",
                 x=.27, ha="left", fontsize=15)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"figure2.{ext}" for ext in ("png", "svg")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", pad_inches=.1)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=.1, metadata={"Date": None})
    plt.close(fig)
    return paths
