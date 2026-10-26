"""D5 v2.1 against D5 v2: what route (iii) and A21's FP4 route move in `d5_v2.py`'s report.

Reads both reports' text and JSON (`d5_v2.py > OUT.txt`, `--out DIR` writes `d5_v2.json`) and prints, per section of
the report, how many lines moved on each card (with the H200's largest numeric change, which should be 0), every
per-model summary line whose figures moved (served c, c_inc, c_mm, leak, worst, job c, any-prices, γ: v2 → v2.1),
the pub(L′) table's served c and c at X = 1 per card (cells moved, largest move), P-HYB3's rows, the hashing rows, and
the named programs that moved. `--rounds D6VSD5.json` (`d6_vs_d5.py` run with PUBLIC_BEST_DIR at v2.1) adds A31's
per-round table at X = 1 on v2.1's menu.

    python3 d5v21_delta.py V2.txt V21.txt V2.json V21.json [OUT.json] [--rounds D6VSD5.json]
"""
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r"-?\d+\.\d+")
CARDS = ("H200", "RTX PRO 6000")
SUMMARY = re.compile(
    r"^  (H200|RTX PRO 6000)\s+(\S+)\s+served c (\S+) \(c_inc (\S+)\) c_mm (\S+) leak (\S+)% eps_alg \S+% worst (\S+)"
    r" \| job c (\S+) c_mm (\S+) leak (\S+)% eps_alg \S+% \| any-prices (\S+)% \| γ (\S+)")
FIELDS = ("served c", "c_inc", "c_mm", "leak %", "worst", "job c", "job c_mm", "job leak %", "any-prices %", "γ")


def sections(path: str) -> list[tuple[str, list[str]]]:
    out = [("(head)", [])]
    for line in Path(path).read_text().splitlines():
        if line.startswith("== "):
            out.append((line[3:].split(": ")[0], []))
        else:
            out[-1][1].append(line)
    return out


def card_of(line: str) -> str | None:
    return next((c for c in CARDS if line.startswith(f"  {c} ")), None)


def nums(line: str) -> list[float]:
    return [float(x) for x in NUM.findall(line)]


def main():
    old, new = sections(sys.argv[1]), sections(sys.argv[2])
    assert [s for s, _ in old] == [s for s, _ in new], "the two reports' sections differ"
    res = {"sections": {}, "summary": [], "pub_L": {}, "phyb3": [], "hashing": [], "named": []}
    print("== lines moved per section (H200 | RTX PRO 6000 | other), and the H200's largest numeric change")
    for (name, a), (_, b) in zip(old, new):
        assert len(a) == len(b), f"{name}: {len(a)} lines against {len(b)}"
        moved = {"H200": 0, "RTX PRO 6000": 0, None: 0}
        h200 = 0.0
        card = None
        for x, y in zip(a, b):
            card = card_of(x) or (card if x.startswith("    ") else None)
            if x != y:
                moved[card] += 1
                if card == "H200":
                    nx, ny = nums(x), nums(y)
                    h200 = max([h200, *(abs(p - q) for p, q in zip(nx, ny))] + ([float("inf")] if len(nx) != len(ny)
                                                                                  else []))
        res["sections"][name] = {"H200": moved["H200"], "RTX PRO 6000": moved["RTX PRO 6000"], "other": moved[None],
                                 "H200_max": h200}
        print(f"  {name}: {moved['H200']} | {moved['RTX PRO 6000']} | {moved[None]}; H200 max {h200:g}")

    print("\n== per-model summary lines that moved: v2 → v2.1")
    for (name, a), (_, b) in zip(old, new):
        for x, y in zip(a, b):
            mx, my = SUMMARY.match(x), SUMMARY.match(y)
            if not (mx and my) or x == y:
                continue
            assert mx.group(1, 2) == my.group(1, 2)
            card, model = mx.group(1, 2)
            fx, fy = mx.groups()[2:], my.groups()[2:]
            ch = {f: (float(p), float(q)) for f, p, q in zip(FIELDS, fx, fy) if p != q}
            res["summary"].append({"section": name, "card": card, "model": model, "moved": ch})
            print(f"  {name} | {card} {model}: " + "; ".join(f"{f} {p:g} → {q:g}" for f, (p, q) in ch.items()))

    print("\n== pub(L′): served c and c at X = 1, cells moved per card (largest move)")
    (_, a), (_, b) = next(s for s in zip(old, new) if s[0][0].startswith("pub(L')"))
    for x, y in zip(a, b):
        card = card_of(x)
        if not card:
            continue
        cells = zip(x.split(";"), y.split(";"))
        st = res["pub_L"].setdefault(card, {"cells": 0, "served_moved": 0, "x1_moved": 0, "served_max": 0.0,
                                            "x1_max": 0.0, "worst": None})
        for cx, cy in cells:
            px, py = cx.split("|"), cy.split("|")
            st["cells"] += 1
            sx, sy = nums(px[2])[0], nums(py[2])[0]
            ox, oy = nums(px[3])[0], nums(py[3])[0]
            if sx != sy:
                st["served_moved"] += 1
                st["served_max"] = max(st["served_max"], abs(sx - sy))
            if ox != oy:
                st["x1_moved"] += 1
                if abs(ox - oy) > st["x1_max"]:
                    st["x1_max"] = abs(ox - oy)
                    st["worst"] = f"{x[2:32].strip()} {cx.split(':')[0].strip()}: {ox:.3f} → {oy:.3f}"
    for card, st in res["pub_L"].items():
        print(f"  {card}: {st['cells']} cells; served c moved in {st['served_moved']} (max {st['served_max']:.3f}),"
              f" c at X = 1 in {st['x1_moved']} (max {st['x1_max']:.3f}, {st['worst']})")

    for title, key in (("P-HYB3", "phyb3"), ("hashing priced six ways", "hashing")):
        print(f"\n== {title}: rows that moved")
        (_, a), (_, b) = next(s for s in zip(old, new) if s[0][0].startswith(title))
        for x, y in zip(a, b):
            if x != y:
                res[key].append({"v2": x.strip(), "v2.1": y.strip()})
                print(f"  v2:   {x.strip()}\n  v2.1: {y.strip()}")

    jo, jn = (json.loads(Path(p).read_text()) for p in sys.argv[3:5])
    print("\n== named programs that moved (c, c_mm, mix)")
    ident = ("name", "design", "model", "prices")
    for p, q in zip(jo["named"], jn["named"]):
        assert [p.get(k) for k in ident] == [q.get(k) for k in ident]
        if p != q:
            row = {k: p.get(k) for k in ident}
            row.update({k: (p.get(k), q.get(k)) for k in sorted(set(p) | set(q)) if k not in ident and p.get(k) != q.get(k)})
            res["named"].append(row)
            print("  " + " ".join(str(p.get(k)) for k in ident) + ": " + "; ".join(
                f"{k} {v[0]:.4f} → {v[1]:.4f}" if isinstance(v[0], float) else f"{k} → {v[1]}"
                for k, v in row.items() if k not in ident))
    def h200(d):
        if isinstance(d, dict):
            return {k: (v if k == "H200" else h200(v)) for k, v in d.items()
                    if k != "RTX PRO 6000" and not (k.startswith("(") and "H200" not in k)}
        if isinstance(d, list) and d and isinstance(d[0], dict) and "name" in d[0]:
            return [x for x in d if "H200" in x["name"]]
        return d

    moved = [k for k in jo if k != "inputs" and h200(jo[k]) != h200(jn[k])]
    res["json_h200_moved"] = moved
    print(f"\n== JSON sections whose H200 (or card-free) entries moved: {moved or 'none'}")
    if "--rounds" in sys.argv:
        res["rounds"] = rounds(jo, jn, json.loads(Path(sys.argv[sys.argv.index("--rounds") + 1]).read_text()))
    if len(sys.argv) > 5 and not sys.argv[5].startswith("--"):
        Path(sys.argv[5]).write_text(json.dumps(res, indent=1))
    print("DONE")


def rounds(jo: dict, jn: dict, d6: dict) -> list[dict]:
    """A31's per-round table on v2.1: D5's own `fixed` reading at X = 1 (1 − c, each round alone, from its L′ sweep at
    D6's L′) beside D6's per-round leak at x = 1 in `d6_vs_d5.py`'s step a (D6's job LPs, v3.1's picks), and v2's."""
    def d5(j, card, K, N, Lp):
        r = next(r for r in j["sweep"][card][f"{K}x{N}"] if r["Lp"] == Lp)
        return 1 - r["fixed"]["1.0"]["c"]

    print("\n== v3.1's static rounds at X = 1: D5 v2.1's own leak | D6's per round on v2.1's menu | D5 v2's")
    out, worst = [], 0.0
    for key, v in d6.items():
        design, card, model, x = key.split(" | ")
        if design != "v3.1" or x != "1.0":
            continue
        for r in v["rounds"]:
            d5n, d5o = d5(jn, card, r["K"], r["N"], r["Lp"]), d5(jo, card, r["K"], r["N"], r["Lp"])
            worst = max(worst, abs(d5n - r["leak"]))
            out.append({"card": card, "model": model, "K": r["K"], "N": r["N"], "Lp": r["Lp"], "d5_v21": d5n,
                        "d6_v21": r["leak"], "d5_v2": d5o})
            print(f"  {card} {model} {r['K']}x{r['N']} L′ {r['Lp']}: {100 * d5n:.4f}% | {100 * r['leak']:.4f}% | "
                  f"{100 * d5o:.4f}%")
    print(f"  {len(out)} rounds, largest |D5 v2.1 − D6| {worst:.2e}")
    return out


if __name__ == "__main__":
    main()
