"""PADCREDIT rounds 3 and 3.1 (cursor/padded-credit-stmt-ea24 at c37a9ba36): probes against main's pc8, rowk, window and
pearl_c, and mirrors of `Accounting.TS.SelectLayer`'s predicates on the layouts the note proposes.

1. forms: at the served k a row's forming is rowk's four pieces, and the piece that writes the tile's operand row
   (`Pc8RowATail`) reads no raw row; a forming piece may read a pre-coin non-noise value from a unit outside `proved`,
   which no edge charges (mirror of `CertifiedOperands`, `Feeds`, `SelGood`).
2. per-salt B formings: the window tile's public inputs and its side domains read no salt, so nothing in the window
   ties its B rows to this salt's formings.
3. the filler: the ok unit admits the zero row at 0 and a junk row at 1; §8.3's registered S_B is neither a gate
   position nor a tally output, so `TallyReads` and `GatesTallied` refuse that layout (mirror).
4. anchors: under h3 an A row's E line is drawn from `row_digest(x_i)`, the unsalted inner digest of its hm96 leaf, so a
   verifier can recompute it only from a non-hiding digest of the private row.

Runs on a CPU in seconds: ``uv run python benchmarks/rate_redteam/padcredit_r3_probes.py``.
"""
from __future__ import annotations

import inspect
import random
import struct

from verity.ml.circuits.evaluation import evaluate
from verity.ml.commitments import hm96
from verity.core.protocols.pouw import window
from verity.core.protocols.pouw.circuit import pc8, rowk
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_kw as P


def params(fn) -> list[str]:
    return [n for n, _t in fn.signature()[0]]


def refl_trans(edge, units, v):
    seen, todo = {v}, [v]
    while todo:
        x = todo.pop()
        for u in units:
            if u not in seen and edge(u, x):
                seen.add(u)
                todo.append(u)
    return seen


def probe_forms() -> None:
    print("== 1. forms: rowk's pieces at k = 4096 on sm_120 (the served k is past rowk's 2,560) ==")
    head, first, forms, tail = rowk.pieces8(4096, rowk.ROW_K, "sm120", "a")
    for fn in (head, first, forms[0] if isinstance(forms, tuple) else forms, tail):
        print(f"  {fn.defn.name:14s} reads {params(fn)}")
    print(f"  {tail.defn.name} reads x: {'x' in params(tail)}; it writes the tile's fp8-p row (codes ‖ P_A) and ok:",
          [r[0] for r in rowk.PORTS8["row_a_tail"]["rows"]])
    print("  pc8's forming roles on main:", sorted(set(pc8._role_of().values())))

    # Mirror: a forming that reads alpha from a pre-coin Stats unit outside `proved`.
    # Positions: (name, writer, cut_after_coin). Every unit's coin is coin 0.
    pos = {"x": (None, False), "ok": ("OkA", False), "alpha": ("Stats", False), "E": (None, True),
           "At": ("FormA", True)}
    reads = {"OkA": ["x", "ok"], "Stats": ["x", "alpha"], "FormA": ["x", "alpha", "E", "At"], "Tile": ["At"]}
    gates, floor, cert_row = {"Tile": ["ok"]}, {"Tile": ["At"]}, {"ok": "x"}
    forms_, proved = {"FormA"}, {"OkA", "FormA"}
    units = list(reads)
    certified = all(any(pos[k][0] in proved and pos[k][0] in forms_ and cert_row[l] in reads[pos[k][0]]
                        and pos[l][0] is not None and cert_row[l] in reads[pos[l][0]] for l in gates[v])
                    for v in floor for k in floor[v])
    late = lambda u, v: any(pos[p][0] == u and pos[p][1] for p in reads[v])            # noqa: E731
    gate = lambda u, v: any(pos[l][0] == u for l in gates.get(v, []))                    # noqa: E731
    feeds = lambda u, v: late(u, v) or (gate(u, v) and u not in proved)                  # noqa: E731
    cone = refl_trans(feeds, units, "Tile")
    paid = {u: u != "Stats" for u in units}
    print(f"  mirror: CertifiedOperands {certified}; units feeding Tile {sorted(cone)}; Stats unpaid (alpha = 0): "
          f"SelGood(Tile) = {all(paid[u] for u in cone)}")


def probe_salt() -> None:
    print("== 2. per-salt B formings: what the window tile's statement reads ==")
    side = pc8.side_tile(64, (1, 1))
    print(f"  {side.defn.name} reads {params(side)}; its public input regions: {pc8.SIDE_PORTS['regions']}")
    print("  window tile public ports:", [p["port"] for p in window.PUBLIC_PORTS])
    print("  window.public_inputs takes:", list(inspect.signature(window.public_inputs).parameters))
    print("  window.side_domain takes:", list(inspect.signature(window.side_domain).parameters))
    root_b = bytes(range(64))
    s1, s2 = C.seed_b(b"\x01" * 32, root_b, "h3"), C.seed_b(b"\x02" * 32, root_b, "h3")
    print(f"  E_B,0 under two salts differs: {C.sample_line(s1, 1, 0, 0) != C.sample_line(s2, 1, 0, 0)}; "
          "the window's tile instances are byte-identical under both (no salt among their inputs)")


def ok_of(x_words: list[int], k: int) -> int:
    seed_a, seed_b = b"\x07" * 32, b"\x09" * 32
    ea = P.sample_line(seed_a, 0, 0, 0)
    fa = [c for ln in C._basis(seed_b, 0, k, C.LINE_NORM_V1) for c in ln]
    fb = [c for ln in C._basis(seed_b, 1, k, C.LINE_NORM_V1) for c in ln]
    out = evaluate(pc8.templates(k, pc8.TILE, "sm120")["row_a"], x_words, ea, fa, fb)
    return out[-1]


def probe_filler() -> None:
    print("== 3. tally: the filler's ok, and §8.3's registered S_B ==")
    k = 128
    f32 = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]                         # noqa: E731
    rng = random.Random(3)
    print(f"  Pc8RowA ok on the zero row: {ok_of([0] * k, k)}; on a junk row N(0,1): "
          f"{ok_of([f32(rng.gauss(0, 1)) for _ in range(k)], k)}")

    # Mirror of TallyReads / GatesTallied on §8.3: ok_B,j registered, S_B registered and read by the tally.
    gates = {"Tile_j": ["okA", "okB_j"]}
    src = {"okA": "OkA", "okB_j": None, "S_B": None, "SA": "Lsum", "real": "Lmul", "out": "Lout"}
    regd = {"okB_j": 1, "S_B": 1}
    tally = {"Lsum": ["okA", "SA"], "Lmul": ["SA", "S_B", "real"], "Lout": ["real", "out"]}
    gate_pos = {p for ps in gates.values() for p in ps}
    tally_reads = all(p in gate_pos or src.get(p) in tally for t in tally for p in tally[t])
    gates_tallied = all(any(p in tally[t] for t in tally) for p in gate_pos)
    fixed = all(p in gate_pos or src.get(p) in tally or (src.get(p) is None and p in regd) for t in tally for p in tally[t])
    print(f"  §8.3 layout: TallyReads {tally_reads}, GatesTallied {gates_tallied} (okB_j read by no tally unit); "
          f"TallyReads with a registered disjunct {fixed}")


def probe_anchor() -> None:
    print("== 4. anchors under h3: the E_A line's seed ==")
    print(f"  h3 seeds rows: {C.ROW_SEEDED}; pc8's ea derivation: {pc8.DERIVATIONS['ea'].rule}, from "
          f"{pc8.DERIVATIONS['ea'].seed} (per call, h0's)")
    row = bytes(random.Random(5).randbytes(4 * 128))
    x = C.row_digest(row)
    n = hm96.SHA512.salt_bytes
    leaves = [hm96.SHA512.commit_string(hm96.SHA512.default_key, x, s) for s in (b"\x00" * n, b"\xff" * n)]
    salt, root_b = b"\x01" * 32, bytes(64)
    seed = C.row_seed(salt, x, root_b, 0, 0)
    print(f"  two hm96 leaves of one row differ: {leaves[0] != leaves[1]}; the row seed reads the inner digest "
          f"row_digest(x_i), the same under both: a guess x' of the row gives row_seed(x') == seed: "
          f"{C.row_seed(salt, C.row_digest(row), root_b, 0, 0) == seed}")


if __name__ == "__main__":
    probe_forms()
    probe_salt()
    probe_filler()
    probe_anchor()
