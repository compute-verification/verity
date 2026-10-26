"""CMP-149: the shared row's size as a costed parameter, the serve-time commit's seconds a step against the drawn units'
in-circuit openings, from circuits' CIR-113 shared-row Layout JSON, its readers and writers, and today's layout.

    python benchmarks/e2e_h200/rowsize.py --shared families-shared.layout.json --bindings shared-bindings.json \
        [--today families-today.layout.json] [--json OUT.json]

**Relayout at row size R.** A source tensor whose rows are one-row commit units is cut into rows of
r = min(max(R, its row in CIR-113's layout), its width) bytes, the last one shorter when r doesn't divide the width (the
engine hashes a row's own length, so a short row wastes no compression, only a row); each row is one commit unit. The
quantizer's two-row units (8,192 B of codes and the 4 B scale) never change. Scope `all` cuts every such tensor (every
producer then binds its window inside its proof, as CIR-113's s8 coordinates do); scope `gathers` cuts only the tensors
CIR-113 shares (the gathers'), and the other producers keep their own rows.

**Commit.** Two models at `SLOTS` slots. Counted: a row-slot costs its digest, ceil((l + 17) / 128) compressions after
its midstated P(l), and hm96's c and h and its frame leaf, 5; a unit-slot its slot-tree leaf, 1, and about one slot-tree
node, 2; a unit of n rows 2 (n - 1) more for its tree (`commit_slots/tests/test_cost.py`: nine a 2-byte one-row unit).
Kernels' 9 -> 7 (KCOMMIT, note:kernels-kcommit/20261026T0855Z-report-slots-rowcost) takes one from each row-slot and the
slot-tree leaf from each unit-slot. Seconds are the compressions at the engine's whole-step rates. Per part: circuits'
CIR-113 projection (art:d4fce7df…, `shrow_rows.py`) from #4666's measured parts, leaves 2.04 ns a row-slot, the fold
0.544 ns a unit-slot, row digests at 3.72 G compressions/s and the multi-row units 0.35 s.

**Proving.** A drawn unit opens every committed row its inputs and outputs sit in: `optimize.row_ands` each, hm96's
two private compressions and the digest's ceil((l + 17) / 128), plus form (B)'s window select where its span is part of a
row (2,072 ANDs for a 2-byte window of a 256 B row, CIR-113: 8 (l - n) + 40). Rows a unit opens outside the step's
layout (its weights, the request) and the K and V rows (128 B tensors) don't change with R. Each family's own ANDs are
the staged counts Demo A plans with (`benchmarks/sampling_plan/demo_a.py`) and circuits' stagings.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SLOTS = 4096
STEP_S = 3.78
TARGET_S = 0.25 * STEP_S
RATES = (2.69, 3.13, 4.79)
LEAF_NS, FOLD_NS, DIGEST_GCPS, UNITS_S = 2.04, 0.544, 3.72, 0.35
SIZES = (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192)
SHA512_ANDS, HM96_ANDS = 58_120, 116_240
M3_UNITS = 8

#: a unit's own ANDs by its Definition's key prefix, at the frozen shape
ANDS = {
    "ScaledMmS8CoordinateRows_v1{K=16384": 2_150_483,      # demo_a.families, staged k_log 25
    "ScaledMmS8CoordinateRows_v1{K=65536": 8_586_323,      # demo_a.families, staged k_log 27
    "EmbeddingCoordinate_v1": 4_820,                       # demo_a.families
    "RmsSumSqChunks_v1{C=4096": 2 * 2_860_480,             # circuits' {2048,128} staging, doubled (32 chunks, not 16)
    "RmsNormOutSums_v1": 360_356,                          # demo_a.families
    "RowAbsMax_v1{N=16384,RELU=false}": 491_490,           # demo_a.families
    "RowAbsMax_v1{N=16384,RELU=true}": 1_228_770,
    "RowAbsMax_v1{N=16387,RELU=true}": 1_228_995,
    "QuantChunkS8_v1{N=8192,RELU=false}": 22_225_802,
    "QuantChunkS8_v1{N=8192,RELU=true}": 22_602_634,
    "QkSoftmaxRowPad_v1{T=32": 1_210_781,                  # #4601's UnitStager count
    "PvRow_v1{T=32": 1_039_168,                            # demo_a.PV_ANDS at T = 32
    "Bf16AddRow_v2{N=128}": 91_776,
    "Argmax_v2{V=256}": 15_557,
}

#: Demo A's PoUW tile lanes (demo_a.TILE_TRACED, TILE_ROWS), whose rows the step layout doesn't hold
TILE_TRACED = {16_384: (5_540_007_551, 129), 65_536: (22_095_177_215, 513)}
TILE_ROWS = (4096,) * 12 + (64,) * 3


def comps(nbytes: int) -> int:
    return -(-(nbytes + 17) // 128)


def row_ands(nbytes: int) -> int:
    return HM96_ANDS + SHA512_ANDS * comps(nbytes)


def select_ands(span: int, row: int) -> int:
    return 8 * (row - span) + 40 if span < row else 0


def own_ands(key: str) -> int:
    hits = [a for k, a in ANDS.items() if key.startswith(k)]
    if len(hits) != 1:
        raise KeyError(f"no staged ANDs for {key}")
    return hits[0]


# --- the layout ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    group: int
    index: int
    width: int
    row: int            # its rows' bytes in CIR-113's layout
    family: str         # the family whose units commit it (a quantizer's codes and scale: the quantizer's)
    gather: bool
    fixed: bool         # rows of multi-row units, never recut
    copies: int         # rows a slot of it (fixed sources: as laid out)


def sources(layout: dict, bindings: dict) -> list[Source]:
    fams = sorted((f for f in bindings["families"] if f["shared_first"] is not None), key=lambda f: f["shared_first"])
    starts = [f["shared_first"] for f in fams]

    def family(u: int) -> dict:
        lo, hi = 0, len(starts)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if starts[mid] <= u else (lo, mid)
        f = fams[lo]
        if not f["shared_first"] <= u < f["shared_first"] + f["units"]:
            raise ValueError(f"unit {u} is in no family")
        return f

    out, u0 = [], 0
    for gi, g in enumerate(layout["groups"]):
        by: dict[int, list] = defaultdict(list)
        for ui, unit in enumerate(g["units"]):
            for row in unit:
                for s, off, n in row:
                    by[s].append((u0 + ui, len(unit), off, n))
        for s, width in enumerate(g["sources"]):
            parts = by[s]
            fs = {family(u)["family"] for u, *_ in parts}
            if len(fs) != 1 and not all(f.split(".")[-1] in ("down_amax", "part") for f in fs):
                raise ValueError(f"group {gi} source {s} is committed by {fs}")
            f = family(parts[0][0])
            fixed = any(k > 1 for _, k, _, _ in parts)
            rows = Counter(n for *_, n in parts)
            if not fixed and (len(rows) != 1 or sum(n for *_, n in parts) != width):
                raise ValueError(f"group {gi} source {s}'s one-row units don't tile it once: {rows}")
            out.append(Source(gi, s, width, max(rows), f["family"].removesuffix(".part") if len(fs) > 1 else f["family"],
                              f["key"].startswith("AllGather"), fixed, len(parts)))
        u0 += len(g["units"])
    return out


def rowlen(src: Source, R: int | None, scope: str) -> int:
    if R is None or src.fixed or (scope == "gathers" and not src.gather):
        return src.row
    return min(max(R, src.row), src.width)


def cut(width: int, r: int) -> list[int]:
    return [r] * (width // r) + ([width % r] if width % r else [])


def slot_rows(layout: dict, srcs: list[Source], R: int | None, scope: str) -> dict:
    """A slot's rows (their bytes), units and the units' row counts at R (None: the layout as laid out)."""
    rows: list[int] = []
    unit_rows: list[int] = []
    filled = room = 0
    fixed = {(s.group, s.index) for s in srcs if s.fixed}
    for s in srcs:
        if not s.fixed:
            r = rowlen(s, R, scope)
            for n in cut(s.width, r):
                rows.append(n)
                unit_rows.append(1)
                if R and (scope == "all" or s.gather) and s.row < R:
                    filled, room = filled + n, room + R
    for gi, g in enumerate(layout["groups"]):
        for unit in g["units"]:
            if any((gi, p[0]) in fixed for row in unit for p in row):
                rows += [sum(p[2] for p in row) for row in unit]
                unit_rows.append(len(unit))
    return {"rows": rows, "unit_rows": unit_rows, "bytes": sum(rows), "fill": filled / room if room else None}


def raw_rows(layout: dict) -> dict:
    rows, unit_rows = [], []
    for g in layout["groups"]:
        for unit in g["units"]:
            rows += [sum(p[2] for p in row) for row in unit]
            unit_rows.append(len(unit))
    return {"rows": rows, "unit_rows": unit_rows, "bytes": sum(rows)}


def commit_cost(sr: dict, R: int | None) -> dict:
    rows, units = len(sr["rows"]), len(sr["unit_rows"])
    one = sum(1 for k in sr["unit_rows"] if k == 1)
    digest = sum(comps(n) for n in sr["rows"])
    tree = sum(2 * (k - 1) for k in sr["unit_rows"])
    now = digest + 5 * rows + 3 * units + tree
    seven = digest + 4 * rows + 2 * units + tree
    out = {"R": R, "rows_a_slot": rows, "units_a_slot": units, "bytes_a_slot": sr["bytes"],
           "fill": sr.get("fill"),
           "digest_comps_a_slot": digest, "comps_a_row_slot": now / rows, "comps_a_row_slot_9to7": seven / rows,
           "comps_a_step": now * SLOTS, "comps_a_step_9to7": seven * SLOTS}
    for rate in RATES:
        out[f"s_at_{rate}"] = now * SLOTS / (rate * 1e9)
        out[f"s_at_{rate}_9to7"] = seven * SLOTS / (rate * 1e9)
    for tag, sev in (("", False), ("_9to7", True)):
        leaf = (LEAF_NS - (1 / DIGEST_GCPS if sev else 0)) * rows * SLOTS / 1e9
        fold = (FOLD_NS * units - (FOLD_NS / 3 * one if sev else 0)) * SLOTS / 1e9
        dig = digest * SLOTS / (DIGEST_GCPS * 1e9)
        out[f"parts{tag}"] = {"leaves": leaf, "fold": fold, "rows": dig, "units": UNITS_S}
        out[f"s_parts{tag}"] = leaf + fold + dig + UNITS_S
    return out


# --- the drawn units' openings -----------------------------------------------------------------------------------------

#: a family's spans in the step's layout, (producer family's suffix, offset, bytes, distinct rows of it): its reads and its
#: outputs, for a unit of a middle layer (x is the previous layer's residual); plus the rows it opens outside the layout
#: (weights, the request) and the K and V rows of 128 B, which no R changes
S8_IN = (("quant", 0, 16384, 1), ("quant.scale", 0, 4, 1))
SPANS = {
    "wq": ((("attn_quant", 0, 16384, 1), ("attn_quant.scale", 0, 4, 1), ("wq.gather", 0, 2, 1)), (8192, 8192, 4)),
    "wk": ((("attn_quant", 0, 16384, 1), ("attn_quant.scale", 0, 4, 1), ("wk.gather", 0, 2, 1)), (8192, 8192, 4)),
    "wv": ((("attn_quant", 0, 16384, 1), ("attn_quant.scale", 0, 4, 1), ("wv.gather", 0, 2, 1)), (8192, 8192, 4)),
    "wo": ((("o_quant", 0, 16384, 1), ("o_quant.scale", 0, 4, 1), ("wo.gather", 0, 2, 1)), (8192, 8192, 4)),
    "wup": ((("up_quant", 0, 16384, 1), ("up_quant.scale", 0, 4, 1), ("wup.gather", 0, 2, 1)), (8192, 8192, 4)),
    "wdown": ((("down_quant", 0, 65536, 1), ("down_quant.scale", 0, 4, 1), ("wdown.gather", 0, 2, 1)), (8192,) * 8 + (4,)),
    "head": ((("head_quant", 0, 16384, 1), ("head_quant.scale", 0, 4, 1), ("head.gather", 0, 2, 1)), (8192, 8192, 4)),
    "embed": ((("embed.gather", 0, 2, 1),), (1, 512, 2)),
    "attn_sums": ((("prev.mlp_residual", 0, 8192, 1), ("attn_sums", 0, 128, 1)), ()),
    "mlp_sums": ((("attn_residual", 0, 8192, 1), ("mlp_sums", 0, 128, 1)), ()),
    "final_sums": ((("prev.mlp_residual", 0, 8192, 1), ("final_sums", 0, 128, 1)), ()),
    "attn_norm": ((("prev.mlp_residual", 0, 256, 1), ("attn_sums", 0, 512, 1), ("attn_norm", 0, 256, 1)), (256,)),
    "mlp_norm": ((("attn_residual", 0, 256, 1), ("mlp_sums", 0, 512, 1), ("mlp_norm", 0, 256, 1)), (256,)),
    "final_norm": ((("prev.mlp_residual", 0, 256, 1), ("final_sums", 0, 512, 1), ("final_norm", 0, 256, 1)), (256,)),
    "attn_amax": ((("attn_norm", 0, 32768, 1), ("attn_amax", 0, 2, 1)), ()),
    "up_amax": ((("mlp_norm", 0, 32768, 1), ("up_amax", 0, 2, 1)), ()),
    "head_amax": ((("final_norm", 0, 32768, 1), ("head_amax", 0, 2, 1)), ()),
    "o_amax": ((("pv.gather", 0, 32768, 1), ("o_amax", 0, 2, 1)), ()),
    "attn_quant": ((("attn_norm", 0, 16384, 1), ("attn_amax", 0, 2, 1), ("attn_quant", 0, 8192, 1),
                    ("attn_quant.scale", 0, 4, 1)), ()),
    "up_quant": ((("mlp_norm", 0, 16384, 1), ("up_amax", 0, 2, 1), ("up_quant", 0, 8192, 1), ("up_quant.scale", 0, 4, 1)), ()),
    "head_quant": ((("final_norm", 0, 16384, 1), ("head_amax", 0, 2, 1), ("head_quant", 0, 8192, 1),
                    ("head_quant.scale", 0, 4, 1)), ()),
    "o_quant": ((("pv.gather", 0, 16384, 1), ("o_amax", 0, 2, 1), ("o_quant", 0, 8192, 1), ("o_quant.scale", 0, 4, 1)), ()),
    "down_amax.part": ((("wup.gather", 0, 32768, 1), ("down_amax", 0, 2, 1)), ()),
    "down_amax": ((("wup.gather", 98304, 32768, 1), ("down_amax", 0, 8, 1)), ()),
    "down_quant": ((("wup.gather", 0, 16384, 1), ("down_amax", 6, 2, 1), ("down_quant", 0, 8192, 1),
                    ("down_quant.scale", 0, 4, 1)), ()),
    "softmax": ((("wq.gather", 0, 128, 1), ("wk.gather", 0, 128, 32), ("softmax", 0, 64, 1)), ()),
    "pv": ((("softmax", 0, 64, 1), ("wv.gather", 0, 128, 32), ("pv.gather", 0, 128, 1)), ()),
    "attn_residual": ((("prev.mlp_residual", 0, 256, 1), ("wo.gather", 0, 256, 1), ("attn_residual", 0, 256, 1)), ()),
    "mlp_residual": ((("attn_residual", 0, 256, 1), ("wdown.gather", 0, 256, 1), ("mlp_residual", 0, 256, 1)), ()),
    "argmax": ((("head.gather", 0, 512, 1), ("argmax", 0, 2, 1)), ()),
}


def by_name(srcs: list[Source]) -> dict[str, Source]:
    """Middle-layer sources by producer suffix (layer 1's group; `prev.` names layer 0's), and the first and last groups'."""
    out: dict[str, Source] = {}
    for s in srcs:
        name = s.family
        if s.fixed and s.width == 4:
            name += ".scale"
        if s.group == 2:
            out[name.split(".", 1)[1]] = s
        elif s.group == 1:
            out["prev." + name.split(".", 1)[1]] = s
        elif s.group in (0, len({x.group for x in srcs}) - 1):
            out[name] = s
    return out


def opening(spec: str, named: dict[str, Source], R: int | None, scope: str) -> int:
    spans, outside = SPANS[spec]
    rows: dict[str, tuple[list[int], int, set[int]]] = {}
    total = sum(row_ands(n) for n in outside)
    for name, off, n, distinct in spans:
        s = named[name]
        r = rowlen(s, R, scope)
        lens, _, touched = rows.setdefault(name, (cut(s.width, r), distinct, set()))
        touched.update(range(off // r, (off + n - 1) // r + 1))
        total += select_ands(n, r)
    for lens, distinct, touched in rows.values():
        total += distinct * sum(row_ands(lens[i]) for i in touched)
    return total


def family_counts(bindings: dict) -> dict[str, tuple[str, int]]:
    """Proof units a slot by span spec: (key, units), the gathers left out (CIR-113: their rows are shared, nothing to prove)."""
    out: dict[str, list] = {}
    for f in bindings["families"]:
        if f["key"].startswith("AllGather"):
            continue
        name = f["family"]
        spec = name.split(".", 1)[1] if name[0] == "l" and name[1].isdigit() else name
        if spec not in SPANS:
            raise KeyError(f"no spans for {name}")
        k, n = out.get(spec, (f["key"], 0))
        out[spec] = [k, n + f["units"]]
    return {k: (v[0], v[1]) for k, v in out.items()}


def proving(bindings: dict, named: dict[str, Source], R: int | None, scope: str, tiles_nw: tuple[float, float]) -> dict:
    fams = family_counts(bindings)
    per = {}
    for spec, (key, n) in fams.items():
        a = own_ands(key)
        o0, o = opening(spec, named, None, scope), opening(spec, named, R, scope)
        per[spec] = {"key": key, "units_a_slot": n, "ands": a, "open_base": o0, "open": o, "extra": o - o0}
    N = sum(p["units_a_slot"] for p in per.values())
    W = sum(p["units_a_slot"] * p["ands"] for p in per.values())
    mean_extra = sum(p["units_a_slot"] * p["extra"] for p in per.values()) / N
    s8 = [p for p in per.values() if p["key"].startswith("ScaledMmS8CoordinateRows_v1{K=16384")]
    s8_n = sum(p["units_a_slot"] for p in s8)
    s8_extra = sum(p["units_a_slot"] * p["extra"] for p in s8) / s8_n
    s8_proved = sum(p["units_a_slot"] * (p["ands"] + p["open_base"]) for p in s8) / s8_n
    proved_base = sum(p["units_a_slot"] * (p["ands"] + p["open_base"]) for p in per.values()) / N
    nw_base = sum(p["units_a_slot"] * p["ands"] * (p["ands"] + p["open_base"]) for p in per.values())
    nw_new = sum(p["units_a_slot"] * p["ands"] * (p["ands"] + p["open"]) for p in per.values())
    t_nw, _ = tiles_nw
    return {"families": per, "units_a_slot": N, "s8_share_of_units": s8_n / N,
            "s8_extra": s8_extra, "s8_extra_pct": 100 * s8_extra / ANDS["ScaledMmS8CoordinateRows_v1{K=16384"],
            "s8_proved_base": s8_proved, "s8_extra_pct_proved": 100 * s8_extra / s8_proved,
            "mean_extra": mean_extra, "mean_extra_pct": 100 * mean_extra * N / W,
            "handful_extra": M3_UNITS * mean_extra,
            "uniform_pct": 100 * mean_extra / proved_base,
            "budget_pct": 100 * (nw_new / nw_base - 1),
            "budget_pct_with_tiles": 100 * (nw_new - nw_base) / (nw_base + t_nw)}


def tiles_nw() -> tuple[float, float]:
    """Demo A's tile lanes a token at the frozen shape: (sum of n w proved, sum of n w), each lane's proved ANDs its own
    and its chunks' rows."""
    L, q, kv, d, f, V = 52, 16384, 64, 16384, 65536, 256
    lanes = {16_384: (L * (q + 2 * kv + d + f) + V) / 1024, 65_536: L * d / 1024}
    nwp = nw = 0.0
    for k, n in lanes.items():
        a, statements = TILE_TRACED[k]
        proved = a + (statements - 1) * sum(row_ands(b) for b in TILE_ROWS)
        nwp += n * a * proved
        nw += n * a
    return nwp, nw


# --- the table --------------------------------------------------------------------------------------------------------

def md(head: list[str], body: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body])


def commit_table(rows: list[dict]) -> str:
    head = ["row size", "rows/slot", "fill", "comp./row-slot (9→7)", "compressions/step (9→7)",
            "s at 2.69 / 3.13 / 4.79 G/s", "9→7: s at 2.69 / 4.79", "per part (9→7)", "% of 3.78 s (2.69; per part)"]
    body = []
    for r in rows:
        c = r["commit"]
        body.append([r["label"], f"{c['rows_a_slot']:,}", f"{c['fill']:.0%}" if c["fill"] is not None else "–",
                     f"{c['comps_a_row_slot']:.1f} ({c['comps_a_row_slot_9to7']:.1f})",
                     f"{c['comps_a_step'] / 1e9:.2f} G ({c['comps_a_step_9to7'] / 1e9:.2f} G)",
                     f"{c['s_at_2.69']:.2f} / {c['s_at_3.13']:.2f} / {c['s_at_4.79']:.2f}",
                     f"{c['s_at_2.69_9to7']:.2f} / {c['s_at_4.79_9to7']:.2f}",
                     f"{c['s_parts']:.2f} ({c['s_parts_9to7']:.2f})",
                     f"{100 * c['s_at_2.69'] / STEP_S:.0f}%; {100 * c['s_parts'] / STEP_S:.0f}%"])
    return md(head, body)


def proving_table(rows: list[dict]) -> str:
    head = ["row size", "extra ANDs, s8 unit", "% of its own 2.15 M", "% of its proved (own + openings)",
            "extra ANDs, mean drawn unit (% own)", "M3's 8 units", "served proving: uniform draw",
            "served proving: ANDs-weighted draw (with tiles)"]
    body = []
    for r in rows:
        p = r["proving"]
        body.append([r["label"], f"{p['s8_extra'] / 1e6:+.3f} M", f"{p['s8_extra_pct']:+.1f}%",
                     f"{p['s8_extra_pct_proved']:+.1f}%", f"{p['mean_extra'] / 1e6:+.3f} M ({p['mean_extra_pct']:+.1f}%)",
                     f"{p['handful_extra'] / 1e6:+.1f} M", f"{p['uniform_pct']:+.2f}%",
                     f"{p['budget_pct']:+.2f}% ({p['budget_pct_with_tiles']:+.4f}%)"])
    return md(head, body)


def family_table(rows: list[dict], labels: tuple[str, ...]) -> str:
    pick = [r for r in rows if r["label"] in labels]
    fams = pick[0]["proving"]["families"]
    head = ["family", "units/slot", "own ANDs", "openings at CIR-113"] + [f"extra at {r['label']}" for r in pick]
    body = [[k, f"{v['units_a_slot']:,}", f"{v['ands']:,}", f"{v['open_base']:,}"]
            + [f"{r['proving']['families'][k]['extra']:+,}" for r in pick]
            for k, v in sorted(fams.items(), key=lambda kv: -kv[1]["units_a_slot"])]
    return md(head, body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shared", type=Path, required=True, help="CIR-113's shared-row Layout JSON")
    ap.add_argument("--bindings", type=Path, required=True, help="CIR-113's shared-row writers and readers")
    ap.add_argument("--today", type=Path, help="today's one-row-a-unit Layout JSON, for its reference line")
    ap.add_argument("--json", type=Path, help="write every number here")
    a = ap.parse_args(argv)
    out: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        out.append(text)

    layout = json.loads(a.shared.read_text())
    bindings = json.loads(a.bindings.read_text())
    srcs = sources(layout, bindings)
    named = by_name(srcs)
    tnw = tiles_nw()
    record: dict = {"slots": SLOTS, "step_s": STEP_S, "target_s": TARGET_S, "layout": layout["name"], "scopes": {}}
    if a.today:
        t = commit_cost(raw_rows(json.loads(a.today.read_text())), None)
        record["today"] = t
        say(f"today's layout: {t['rows_a_slot']:,} rows a slot, {t['comps_a_step'] / 1e9:.1f} G compressions a step, "
              f"{t['s_at_3.13']:.1f} s at 3.13 G/s, {t['s_parts']:.1f} s per part\n")
    base = commit_cost(slot_rows(layout, srcs, None, "all"), None)
    raw = commit_cost(raw_rows(layout), None)
    if (base["rows_a_slot"], base["digest_comps_a_slot"]) != (raw["rows_a_slot"], raw["digest_comps_a_slot"]):
        raise SystemExit(f"the relayout at CIR-113's rows isn't the layout: {base['rows_a_slot']} vs {raw['rows_a_slot']}")
    for scope in ("all", "gathers"):
        rows = [{"label": "CIR-113 as built", "R": None, "commit": base,
                 "proving": proving(bindings, named, None, scope, tnw)}]
        for R in SIZES:
            rows.append({"label": f"{R:,} B", "R": R, "commit": commit_cost(slot_rows(layout, srcs, R, scope), R),
                         "proving": proving(bindings, named, R, scope, tnw)})
        record["scopes"][scope] = rows
        say(f"## Scope {scope}: {'every one-row tensor' if scope == 'all' else 'the gathers only'} in rows of R\n")
        say(f"Commit at {SLOTS:,} slots:\n")
        say(commit_table(rows))
        say("\nProving (in-circuit ANDs, against CIR-113 as built):\n")
        say(proving_table(rows))
        meets = {k: next((r["label"] for r in rows if r["commit"][k] <= TARGET_S), "none")
                 for k in ("s_at_2.69", "s_at_3.13", "s_at_4.79", "s_parts", "s_at_2.69_9to7", "s_parts_9to7")}
        record["scopes"][scope + "_meets"] = meets
        say(f"\nSmallest R within {TARGET_S:.3f} s a step: {meets}\n")
        if scope == "all":
            say("Per family (scope all), ANDs a unit:\n")
            say(family_table(rows, ("1,024 B", "4,096 B")))
            say()
    record["markdown"] = "\n".join(out)
    if a.json:
        a.json.write_text(json.dumps(record, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
