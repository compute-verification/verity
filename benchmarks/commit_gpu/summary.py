"""A gpu.sh run's outputs, one line a measurement: `python3 summary.py OUT`; `python3 summary.py best OUT` prints the tag of
the checked build whose e2e commitment was fastest; `python3 summary.py table OUT` renders the best build's measurements as
the README's Markdown table, and `python3 summary.py tiles OUT [TAG]` a build's tile-leaf sweep."""
from __future__ import annotations

import json
import os
import sys


def lines(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.startswith("{")] if os.path.exists(path) else []


def tagged(out: str, prefix: str) -> list[tuple[str, list[dict]]]:
    names = sorted(x for x in os.listdir(out) if x.startswith(prefix) and x.endswith(".jsonl"))
    return [(x[len(prefix):-len(".jsonl")], lines(os.path.join(out, x))) for x in names]


def best(out: str) -> str:
    runs = [(r[0]["commit_ms"], tag) for tag, r in tagged(out, "bench-e2e-") if r and "nomask" not in tag and "merged" not in tag]
    return min(runs)[1] if runs else ""


def bench_line(tag: str, r: dict) -> str:
    if "bench" in r:
        graph = f", graph {r['graph_ms']:.3f} ms ({r['operand_gb_per_s_graph']:.1f} GB/s)" if "graph_ms" in r else ""
        if "salt_ms" in r:
            graph += (f", root on the host {r['root_on_host_best_ms']:.3f} ms (CPU tail {r['root_on_host_best_k']} levels, "
                      f"{r['operand_gb_per_s_root_on_host']:.1f} GB/s); salt phase {r['salt_ms']:.3f} ms before the data")
        shape = f" M={r['tile_m']} K={r['tile_k']} R={r['tile_r']}" if r.get("tile_r") else ""
        return (f"{tag} {r['bench']}{shape} n={r['n']} bits={r['lbits']}: commit {r['commit_ms']:.3f} ms (leaves {r['leaf_ms']:.3f}, "
                f"nodes {r['nodes_ms']:.3f}){graph}, {r['operand_gb_per_s']:.1f} GB/s of operand, "
                f"{r['gcomp_per_s']:.2f} G compressions/s, {r['mhz']:.0f} MHz")
    g = r["beside_gemm"]
    line = (f"  beside GEMM {g['m']}x{g['n']}x{g['k']} ({g['gemm_ms']:.3f} ms, x{g['gemms']}): commit/GEMM {g['commit_over_gemm']:.2f}; "
            f"both {g['both_wall_ms']:.3f} ms vs serial {g['serial_wall_ms']:.3f}; hidden {g['hidden_fraction']:.2f}; "
            f"commit beside {g['both_commit_ms']:.3f} ms ({g['commit_gb_per_s_beside']:.1f} GB/s), GEMMs {g['both_gemm_ms']:.3f} ms")
    if "priority" in g:
        p = g["priority"]
        line += (f"\n  beside GEMM, commitment at stream priority {p['stream_priority']}: both {p['wall_ms']:.3f} ms; "
                 f"hidden {p['hidden_fraction']:.2f}; commit beside {p['commit_ms']:.3f} ms "
                 f"({p['commit_gb_per_s_beside']:.1f} GB/s), GEMMs {p['gemm_ms']:.3f} ms")
    return line


def size(n_bytes: int) -> str:
    return f"{n_bytes >> 10} KiB" if n_bytes >= 1024 else f"{n_bytes} B"


def table(out: str) -> None:
    tag = best(out)
    print("| leaf | row digest | leaves | critical path ms | GB/s | x GEMM | salt phase ms | beside: commit done ms (GB/s) "
          "| GEMMs beside | high priority: commit done ms (GB/s) | GEMMs beside |")
    print("|" + "---|" * 11)
    for prefix, digest in (("bench-e2e-", "row/v2"), ("bench-sweep-", "row/v2"), ("bench-seg-", "row-seg/v1")):
        rows = lines(os.path.join(out, f"{prefix}{tag}.jsonl"))
        for b, s in zip(rows[::2], rows[1::2]):
            g = s["beside_gemm"]
            leaf = "e2e (32 int8, 1024-bit row)" if b["bench"] == "e2e" else size(b["lbits"] // 8)
            cells = [leaf, digest, f"2^{b['n'].bit_length() - 1}", f"{b['graph_ms']:.3f}", f"{b['operand_gb_per_s_graph']:.1f}",
                     f"{g['commit_over_gemm']:.2f}", f"{b['salt_ms']:.3f}",
                     f"{g['both_commit_ms']:.3f} ({g['commit_gb_per_s_beside']:.1f})",
                     f"{g['both_gemm_ms'] / (g['gemms'] * g['gemm_ms']):.2f}x of {g['gemms']}"]
            p = g.get("priority")
            cells += ([f"{p['commit_ms']:.3f} ({p['commit_gb_per_s_beside']:.1f})",
                       f"{p['gemm_ms'] / (g['gemms'] * g['gemm_ms']):.2f}x"] if p else ["", ""])
            print("| " + " | ".join(cells) + " |")


def tiles(out: str, tag: str = "") -> None:
    """The tile-leaf sweep (bench-tile-TAG, default the best build's): per prefill chunk M and leaf, the latency from the
    operand's last write to the root alone and beside the chunk's GEMM at high stream priority, and what each costs as a
    fraction of the GEMM."""

    tag = tag or best(out)
    print("| chunk M | leaf | row digest | leaves | latency alone ms (GB/s) | alone x GEMM | salt phase ms | GEMM ms "
          "| high priority: root at ms | GEMM beside ms | GEMM slowdown x GEMM |")
    print("|" + "---|" * 11)
    rows = lines(os.path.join(out, f"bench-tile-{tag}.jsonl"))
    for b, s in zip(rows[::2], rows[1::2]):
        g, p = s["beside_gemm"], s["beside_gemm"]["priority"]
        if b["bench"] == "e2e":
            m, leaf, digest = b["n"] // 128, "e2e: 32 values (bf16 row, 1024 bits)", "row/v2"
        else:
            m = b["tile_m"]
            leaf = f"R={b['tile_r']}: {b['tile_r']} x 32 values ({size(b['lbits'] // 8)})"
            digest = "row-seg/v1" if b["bench"] == "tileseg" else "row/v2"
        ga = g["gemms"] * g["gemm_ms"]
        cells = [str(m), leaf, digest, str(b["n"]), f"{b['graph_ms']:.3f} ({b['operand_gb_per_s_graph']:.0f})",
                 f"{g['commit_over_gemm']:.2f}", f"{b['salt_ms']:.3f}", f"{ga:.3f}" + (f" ({g['gemms']})" if g["gemms"] > 1 else ""),
                 f"{p['commit_ms']:.3f}", f"{p['gemm_ms']:.3f}", f"{p['gemm_ms'] / ga - 1:+.2f}"]
        print("| " + " | ".join(cells) + " |")


def main(out: str) -> None:
    for tag, rows in tagged(out, "check-"):
        done = [r for r in rows if "summary" not in r]
        bad = [r for r in done if not r["ok"]]
        print(f"check {tag}: {len(done)} cases, {len(bad)} failed", *(f"\n  FAILED {r}" for r in bad[:5]))
    for tag, rows in tagged(out, "sha-"):
        for r in rows:
            s = r["sha"]
            print(f"sha {tag}: {s['gcomp_per_s']:.2f} G compressions/s ({s['gb_per_s']:.0f} GB/s of blocks) at {s['mhz']:.0f} MHz")
    for prefix in ("bench-e2e-", "bench-sweep-", "bench-seg-", "bench-tile-"):
        for tag, rows in tagged(out, prefix):
            for r in rows:
                print(bench_line(tag, r))


if __name__ == "__main__":
    {"best": lambda: print(best(sys.argv[2])), "table": lambda: table(sys.argv[2]),
     "tiles": lambda: tiles(*sys.argv[2:4])}.get(sys.argv[1], lambda: main(sys.argv[1]))()
