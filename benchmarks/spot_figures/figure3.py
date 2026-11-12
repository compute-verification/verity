"""Figure 3 of the SPOT post: what the serve-time commitment costs over fast serving.

* Left, the `commit_hillclimb` card: the cost agents reached hill-climbing the serve-time hashing (y) against what they
  spent (x), each series' measured points and its best so far, for the series that name no model of the post.
* Right, the model's serve-time cuts: each cut's bytes a token (`design`'s token bytes, the `workload` card's
  `boundaries`, the `infiltration` card's boundary) over the `commitment` card's hash rate, against the token's
  serving GPU-seconds at Figure 1's MFU and peak (modeled; under `serving="measured"` at the `serving_rate` card's
  rate, `figure1.token_seconds`), beside the card's series that name the model (measured).

    uv run python benchmarks/spot_figures/figure3.py card HASH_LOOP.json --art art:...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cards as spot_cards
import figure1

SCHEMA = "spot-figure3/v1"
#: the hash loop's series kept, by its arm: SHA-512 over every captured value, and over the layer boundary
LOOP_ARMS = {"sha512": "every captured value", "sha512-layer": "the layer boundary"}
LOOP_MODELS = {"glm-5.3": "GLM-5.3"}
LOOP_NAMES = {"qwen3-8b": "Qwen3-8B"}
CUT_NAMES = {"tokens": "tokens only", "kv": "K/V", "layer_of_token": "a layer of a token",
             "every_block": "every proof block's outputs"}


def loop_card(loop: dict, art: str) -> dict:
    """The hash loop's results (`hash-loop.json`) as a `commit_hillclimb` card: its SHA-512 arms' TPOT increase at each
    iteration, with the runs that measured it."""
    runs: dict[tuple, list[str]] = {}
    for r in loop["results"]:
        if r["valid"] and isinstance(r["row"], int):
            runs.setdefault((r["row"], r["hardware"]), []).append(r["run"])
    series = []
    for key, pts in sorted(loop["series"]["tpot_overhead_pct"].items()):
        hardware, arm = key.split("/")
        workload, arm = arm.split(":")
        if arm not in LOOP_ARMS or workload not in LOOP_NAMES | LOOP_MODELS:
            continue
        name = LOOP_MODELS.get(workload) or LOOP_NAMES[workload]
        series.append({"label": f"{name} on {hardware}, {LOOP_ARMS[arm]}", "model": LOOP_MODELS.get(workload),
                       "points": [{"x": x, "y": y / 100, "runs": sorted(runs.get((x, hardware), []))}
                                  for x, y in pts]})
    every = sorted({r for s in series for p in s["points"] for r in p["runs"]})
    return {"schema": spot_cards.SCHEMA, "slot": "commit_hillclimb", "model": None, "status": "measured",
            "provisional": True,
            "source": f"{art}: the SHA-512 serve-time commit's hill-climbing loop, each iteration's served decode "
                      "against stock serving; the x-axis is the loop's iteration, its agents' token spend not "
                      "recorded; by figure3.py",
            "runs": every, "owner": "@proofs",
            "value": {"x": {"label": "hill-climbing iteration", "unit": "iteration"},
                      "y": {"label": "TPOT increase over fast serving", "unit": "share"},
                      "workload": "Qwen3-8B served on one GPU, and GLM-5.3 TP8 eager on 8 H200s at batch 32",
                      "series": series}}


def best(points: list[dict]) -> list[float]:
    out, low = [], float("inf")
    for p in points:
        low = min(low, p["y"])
        out.append(low)
    return out


def run(model: str, serving: str = "assumed") -> dict:
    hc = spot_cards.load("commit_hillclimb", None, required=False)
    sc = figure1.slot_cards(model, serving)
    c = figure1.cards(model, serving=serving)
    part = figure1.TokenBlocks.of(c["partition"].value["strata"], c["partition"].value["served_macs_per_token"])
    token_s = figure1.token_seconds(c, part.served_macs)
    rate = c["hash_rate"].value
    cuts = {"tokens": c["token_bytes"].value} | sc["workload"]["value"].get("boundaries", {})
    modeled = [{"cut": k, "label": CUT_NAMES[k], "bytes_per_token": b, "share": b / rate / token_s}
               for k, b in sorted(cuts.items(), key=lambda kv: kv[1])]
    if (inf := sc.get("infiltration")) and inf["value"].get("boundary_bytes_per_token"):
        b = inf["value"]["boundary_bytes_per_token"]
        modeled.append({"cut": inf["value"]["design"], "label": f"infiltration's design ({inf['value']['design']})",
                        "bytes_per_token": b, "share": b / rate / token_s})
    series = (hc or {}).get("value", {}).get("series", [])
    climb = [s | {"best": best(s["points"])} for s in series if s.get("model") is None]
    measured = [{"label": s["label"], "share": s["points"][-1]["y"], "runs": s["points"][-1]["runs"]}
                for s in series if s.get("model") == model]
    return ({"serving": figure1.reading(sc, c, part)} if serving == "measured" else {}) | {
            "schema": SCHEMA, "model": model, "cards": {k: v for k, v in sc.items() if k != "proof_overhead"}
            | ({"commit_hillclimb": hc} if hc else {}),
            "status": "card" if hc else "pending",
            "x": hc["value"]["x"] if hc else None, "y": hc["value"]["y"] if hc else None,
            "climb": climb, "token_seconds": token_s, "hash_rate": rate, "modeled": modeled, "measured": measured}


def render(out: dict, outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 14, "savefig.facecolor": "white"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": (1.15, 1)})
    fig.subplots_adjust(left=.06, right=.98, top=.86, bottom=.12, wspace=.55)
    pct = FuncFormatter(lambda v, _p: f"{v:.0%}" if v >= .01 else f"{v:.0e}")
    colors = {}
    palette = ("#08306B", "#3182BD", "#E69F00", "#6BAED6", "#D95F02")
    if out["climb"]:
        for s in out["climb"]:
            hw = s["label"].split(" on ")[1].split(",")[0]
            color = colors.setdefault(hw, palette[len(colors) % len(palette)])
            ls = "--" if "boundary" in s["label"] else "-"
            xs = [p["x"] for p in s["points"]]
            a.plot(xs, [p["y"] for p in s["points"]], "o", color=color, ms=4, alpha=.45)
            a.step(xs, s["best"], where="post", color=color, ls=ls, lw=2.4, label=s["label"])
        a.set_yscale("log")
        a.yaxis.set_major_formatter(pct)
        a.set_xlabel(out["x"]["label"].capitalize())
        a.set_ylabel(out["y"]["label"])
        a.legend(frameon=False, fontsize=11, loc="upper right")
    else:
        a.text(.5, .5, "Pending", transform=a.transAxes, ha="center", va="center", fontsize=24, color="#909090")
        a.set_yticks([])
    a.set_title("Agents hill-climbing the serve-time hash: best so far (lines), each try (dots)", fontsize=14,
                loc="left")
    rows = [(m["label"] + " (modeled)", m["share"], "#9ECAE1") for m in out["modeled"]]
    rows += [(m["label"] + " (measured)", m["share"], "#08306B") for m in out["measured"]]
    ys = list(range(len(rows)))[::-1]
    b.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=.6)
    for y, (_label, share, _c) in zip(ys, rows):
        b.text(share * 1.2, y, f"{share:.2g}×" if share >= 1 else f"{share:.2%}" if share >= 1e-4 else f"{share:.1e}",
               va="center", fontsize=12)
    b.set_yticks(ys, [r[0] for r in rows], fontsize=12)
    b.set_xscale("log")
    b.set_xlim(min(r[1] for r in rows) / 10, max(r[1] for r in rows) * 20)
    b.xaxis.set_major_formatter(pct)
    b.set_xlabel(f"Serve-time hashing over serving, {out['model']} (hash rate {out['hash_rate']:.3g} B/s)")
    b.set_title(f"{out['model']}'s serve-time cuts", fontsize=14, loc="left")
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="#E2E2E2", lw=.45)
    fig.suptitle("Runtime commitment cost relative to fast serving", x=.06, ha="left", fontsize=17)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"figure3.{ext}" for ext in ("png", "svg")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", pad_inches=.1)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=.1, metadata={"Date": None})
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("card", help="write the hash loop's commit_hillclimb card under cards/")
    c.add_argument("loop", type=Path, help="the loop's hash-loop.json")
    c.add_argument("--art", required=True, help="the loop's art: id, the card's source")
    args = ap.parse_args(argv)
    print(spot_cards.write(loop_card(json.loads(args.loop.read_text()), args.art), force=True))


if __name__ == "__main__":
    main()
