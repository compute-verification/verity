"""uni.sh's run into the uniform datapath's census lines on sm_90 and sm_100, and the IMAD forms of duo's
ALU work beside the fold.

    python3 uniform.py analyze OUT... [--json F]    the rates, each row's UNI coefficient, the IMAD forms
    python3 uniform.py census OUT... [--write]      UNI's census lines; --write puts them, the UP row and UP in the
                                                    current vector's rows into the family's data/devices/sm_NN/
    python3 uniform.py tables GATE_JSON...          rewrite each gated binary's table from probe.py gate's JSON

OUT is uni.sh's output (probes/rows.jsonl). Each probe's loop counts are its table's (probes/sm90_uniform.cu,
sm100_uniform.cu, sm90_imad.cu, sm100_imad.cu: the eight most frequent opcodes, as the gate pinned them). A point's time
is SMSP-cycles a warp-iteration, 4 n / rate (rate is warp instructions an SM-cycle over all its warps). Every opcode starting
with U is a uniform-datapath instruction (UNI); IMMA, ALU (LOP3, LEA.HI, IADD3, SHF, PRMT, ISETP), IMAD (every IMAD
form) and FFMA take the device census's coefficients, and any other vector opcode (MOV, HFMA2, VIADD) the largest of
them on each row, which only lowers what UNI is left.

UNI's lines are what the points show and no more. UP, the uniform pipe, is 1 / the fastest uniform rate alone (+ 2%). On
`issue` a uniform instruction carries its measured issue price: the least slowdown it adds to an issue-saturated stream
(`saturated`: the s, v, w and x streams of sm90_sat.cu and sm100_sat.cu, read only where the stream issues at least 0.95
of its slots without the chains), less 2%; every joint issue row (JI…) scales it as the ALU's coefficient scales from
`issue`. Where no stream saturates, it takes the ALU's issue slot. On every other row it carries nothing: no point loads a
pipe row to its limit beside the chains (the honest step is latency-bound), so contention there is not measured, and 0
gives the cheater the most. Each such row's `limit`, the largest coefficient every point allows, says how much
contention the points leave room for. `check` confirms each row's facet, UNI's lines included, holds at every point.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARGIN = 0.02
MIN_UNI = 16
ALU = re.compile(r"^(LOP3|LEA|IADD3|SHF|PRMT|ISETP)")
TABLE = re.compile(r'P_[UI]\((\w+), "([^"]*)", "([^"]*)"\)')
SASS_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")
WARPS = {"uniform": 16, "issue": 16, "sat": 16, "imad": 32}
SAT = ("us", "uv", "uw", "ux")
SAT_MIN = 0.95


def tables(fam: str) -> dict[str, dict]:
    out = {}
    for kind in ("uniform", "issue", "sat", "imad"):
        for name, ex, cnt in TABLE.findall((HERE / "probes" / f"{fam}_{kind}.cu").read_text()):
            out[name] = {"counts": {k: int(v) for k, v in (p.split("=") for p in ex.split(","))}, "counted": cnt,
                         "warps": WARPS[kind], "binary": f"{fam}_{kind}"}
    return out


def write_table(gate: Path) -> list[Path]:
    """Rewrite each gated binary's table (probes/<binary>.cu) so every probe expects its loop's eight most frequent
    opcodes, the branch aside, as probe.py gate's JSON records them; an alone chain counts its uniform opcode, every other
    probe the MMA."""
    rows: dict[str, list[str]] = {}
    for v in json.loads(gate.read_text())["verdicts"]:
        (loop,) = v["verdict"]["loops"]
        top = [(k, n) for k, n in loop["top"].items() if not k.startswith("BRA")][:8]
        cnt = top[0][0] if v["probe"].startswith("uz_") else next(k for k, _ in top if k.startswith("IMMA.16832"))
        ex = ",".join(f"{k}={n}" for k, n in top)
        rows.setdefault(v["binary"], []).append(f'    P_{"I" if v["binary"].endswith("imad") else "U"}({v["probe"]}, "{ex}", "{cnt}"),')
    out = []
    for binary, rs in rows.items():
        f = HERE / "probes" / f"{binary}.cu"
        f.write_text(re.sub(r"(PROBES\[\] = \{\n).*?(\n\};)", lambda m: m[1] + "\n".join(rs) + m[2], f.read_text(),
                            count=1, flags=re.S))
        out.append(f)
    return out


def kind_of(op: str) -> str:
    if op.startswith("U"):
        return "UNI"
    if op.startswith("IMMA.16832"):
        return "IMMA16832_S8"
    if op.startswith("IMAD"):
        return "IMAD"
    if op.startswith("FFMA"):
        return "FFMA"
    if ALU.match(op):
        return "ALU"
    return "other"


def coefficients(census: Path) -> dict[str, dict[str, float]]:
    """kind → row → SMSP-cycles a warp instruction holds the row (a rate line derated by its noise, as machine_model's u_of)."""
    c: dict[str, dict[str, float]] = {}
    for d in csv.DictReader(census.open()):
        if d["kind"] == "UNI":
            continue
        if d["rate"]:
            v = 4 / (float(d["rate"]) * (1 + float(d["noise"] or 0)))
        elif d["cycles"]:
            v = float(d["cycles"])
        else:
            continue
        c.setdefault(d["kind"], {})[d["row"]] = v
    return c


def points(out: Path, fam: str) -> list[dict]:
    tab = tables(fam)
    pts = []
    for line in (out / "probes" / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        t = tab.get(r["probe"])
        if t is None:
            continue
        cnt = t["counts"][t["counted"]]
        T = 4 * cnt / r["rates"][f"0:{t['counted']}"]
        n: dict[str, float] = {}
        for op, k in t["counts"].items():
            n[kind_of(op)] = n.get(kind_of(op), 0) + k
        pts.append({"probe": r["probe"], "binary": t["binary"], "T": T, "n": n, "counts": t["counts"],
                    "sm_mhz": r.get("sm_mhz"), "throttled": r.get("throttled"), "spread": r.get("spread")})
    return pts


def loop_counts(out: Path) -> dict[str, dict[str, int]]:
    """probe → every opcode of its loop and its count, from the run's SASS (sass/<binary>.sass) between the loop bounds
    the gate recorded (gate-build.json); the tables keep only the eight most frequent."""
    gate = out / "gate-build.json"
    if not gate.exists():
        return {}
    bounds = {}
    for v in json.loads(gate.read_text())["verdicts"]:
        if len(v["verdict"]["loops"]) == 1:
            lp = v["verdict"]["loops"][0]
            bounds[(v["binary"], v["probe"])] = (int(lp["head"], 16), int(lp["tail"], 16))
    counts: dict[str, dict[str, int]] = {}
    for f in (out / "sass").glob("*.sass"):
        for chunk in f.read_text().split("Function : ")[1:]:
            name = chunk.split(None, 1)[0]
            b = bounds.get((f.stem, name))
            if b is None:
                continue
            c: dict[str, int] = {}
            for m in SASS_LINE.finditer(chunk):
                if b[0] <= int(m[1], 16) <= b[1]:
                    c[m[2]] = c.get(m[2], 0) + 1
            counts[name] = c
    return counts


def saturated(outs: list[Path], fam: str) -> dict:
    """The issue-saturated streams (SAT): each stream's baseline issue (every loop instruction a slot, over T) and each
    density's slowdown a uniform instruction it adds, counted in full from the SASS. The issue price is the least
    slowdown over the streams whose baseline issues at least SAT_MIN, less the margin; None if no stream does."""
    streams, least = {}, None
    for out in outs:
        pts = {p["probe"]: p for p in points(out, fam) if p["probe"][:2] in SAT}
        full = loop_counts(out)
        for base in SAT:
            b = pts.get(f"{base}_0")
            if b is None or b["probe"] not in full:
                continue
            nb = sum(full[b["probe"]].values())
            ub = sum(n for k, n in full[b["probe"]].items() if k.startswith("U"))
            s = {"run": out.parent.name, "T0": b["T"], "issue0": nb / b["T"], "points": {}}
            for name, p in pts.items():
                if not name.startswith(base + "_") or name == b["probe"] or name not in full:
                    continue
                added = sum(n for k, n in full[name].items() if k.startswith("U")) - ub
                if added < MIN_UNI:
                    continue
                d = (p["T"] - b["T"]) / added
                s["points"][name] = {"T": p["T"], "added": added, "dT": d, "issue": sum(full[name].values()) / p["T"]}
                if s["issue0"] >= SAT_MIN and (least is None or d < least[0]):
                    least = (d, name, out.parent.name)
            streams[f"{base} {out.parent.name}"] = s
    res = {"streams": streams, "min_issue": SAT_MIN}
    if least:
        res |= {"least": least[0], "set_by": least[1], "run": least[2], "price": least[0] / (1 + MARGIN)}
    return res


def issue_row(row: str) -> bool:
    return row == "issue" or row.startswith("JI")


def vector_load(p: dict, coef: dict[str, dict[str, float]], row: str) -> float:
    """p's vector instructions on `row`, a lower bound: an opcode outside the census's classes takes only its issue slot."""
    load = 0.0
    for cls, k in p["n"].items():
        if cls == "other":
            load += k * coef["ALU"].get(row, 0.0) if issue_row(row) else 0.0
        elif cls != "UNI":
            load += k * coef.get(cls, {}).get(row, 0.0)
    return load


def fit(pts: list[dict], coef: dict[str, dict[str, float]], rows: list[str], up: float,
        price: float | None = None) -> dict[str, dict]:
    """UNI's coefficient on each row (UP; on the issue rows the measured issue price, scaled as the ALU's coefficient is
    from `issue` to each joint row, or an issue slot where none is measured; 0 elsewhere) and the largest every point
    allows, with the point that sets it."""
    scale = 1.0 if price is None else price / coef["ALU"]["issue"]
    out = {"UP": {"c": up}}
    for row in rows:
        if row == "UP":
            continue
        best = None
        for p in pts:
            u = p["n"].get("UNI", 0)
            if u < MIN_UNI:
                continue
            c = (p["T"] * (1 + MARGIN) - vector_load(p, coef, row)) / u
            if best is None or c < best[0]:
                best = (c, p["probe"])
        out[row] = {"c": 0.0}
        if best:
            out[row] |= {"limit": max(0.0, best[0]), "set_by": best[1]}
        if issue_row(row):
            out[row]["c"] = min(coef["ALU"].get(row, 0.0) * scale, out[row].get("limit", math.inf))
    return out


def check(pts: list[dict], coef: dict[str, dict[str, float]], uni: dict[str, dict]) -> list[str]:
    """Every point whose time some row's facet, UNI's lines included, exceeds by more than the margin."""
    bad = []
    for p in pts:
        for row, v in uni.items():
            load = vector_load(p, coef, row) + v["c"] * p["n"].get("UNI", 0)
            if load > p["T"] * (1 + MARGIN):
                bad.append(f"{p['probe']}: {row} {load:.2f} > T {p['T']:.2f} (1 + {MARGIN})")
    return bad


def analyze(outs: list[Path], fam: str, census: Path, rows_csv: Path) -> dict:
    pts = [p for out in outs for p in points(out, fam)]
    coef = coefficients(census)
    rows = [d["row"] for d in csv.DictReader(rows_csv.open())]
    alone = {p["probe"]: p["n"]["UNI"] / p["T"] for p in pts if p["probe"].startswith("uz_")}
    top = max(alone, key=alone.get)
    res = {"family": fam, "uniform_alone": alone, "up": {"rate": alone[top], "by": top,
                                                          "cycles": 1 / (alone[top] * (1 + MARGIN))}}
    res["sat"] = saturated(outs, fam)
    res["rows"] = fit(pts, coef, rows, res["up"]["cycles"], res["sat"].get("price"))
    res["violations"] = check(pts, coef, res["rows"])
    issue_pts = {p["probe"]: {"T": p["T"], "vector": sum(v for k, v in p["n"].items() if k != "UNI"),
                              "uni": p["n"].get("UNI", 0)} for p in pts if not p["binary"].endswith("imad")}
    res["points"] = issue_pts
    for p in pts:
        if not p["binary"].endswith("imad"):
            v = issue_pts[p["probe"]]
            v["issue_frac"] = v["vector"] / v["T"]
    beside = {}
    for base in ("uh", "ug", "ul", "um", "ub"):
        b0 = issue_pts.get(f"{base}_0")
        if not b0:
            continue
        for name, v in issue_pts.items():
            if name.startswith(base + "_") and v["uni"] >= MIN_UNI:
                beside[name] = {"T": v["T"], "T0": b0["T"], "dT_per_uni": (v["T"] - b0["T"]) / v["uni"],
                                "uni_rate": v["uni"] / v["T"]}
    res["beside"] = beside
    imad = {p["probe"]: {"T": p["T"], "counts": p["counts"]} for p in pts if p["binary"].endswith("imad")}
    for f in ("f_duo", "f_duoi", "f_duom"):
        if f in imad:
            imad[f]["per_word"] = imad[f]["T"] / 24
            imad[f]["imad_form_share"] = imad[f]["counts"].get("IMAD.HI.U32", 0) / 24
    if "s_imadhi" in imad:
        res["imad_hi_cycles"] = imad["s_imadhi"]["T"] / imad["s_imadhi"]["counts"]["IMAD.HI.U32"]
    res["imad_forms"] = imad
    if "f_duo" in imad and "f_duom" in imad:
        res["imad_fold_gain"] = 1 - imad["f_duom"]["per_word"] / imad["f_duo"]["per_word"]
    return res


def text(res: dict) -> str:
    L = [f"{res['family']}: UP {res['up']['rate']:.4f} uniform instructions a SMSP-cycle ({res['up']['by']}), "
         f"{res['up']['cycles']:.4f} SMSP-cycles each with the 2% margin"]
    L.append("alone: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(res["uniform_alone"].items())))
    L.append("UNI on each row, SMSP-cycles (the largest every point allows, and the point that sets it): " +
             ", ".join(f"{r} {v['c']:.4f}" + (f" ({v['limit']:.4f} {v['set_by']})" if "limit" in v else "")
                       for r, v in res["rows"].items()))
    L.append("facets exceeded: " + ("; ".join(res["violations"]) or "none"))
    sat = res["sat"]
    L.append(f"issue-saturated streams (baseline issue, then ΔT a uniform instruction added, at the loop's issue):")
    for k, s in sorted(sat["streams"].items()):
        L.append(f"  {k}: issue {s['issue0']:.3f}{'' if s['issue0'] >= sat['min_issue'] else ' (below ' + str(sat['min_issue']) + ', not read)'}  " +
                 "  ".join(f"{n[3:]} {v['dT']:+.3f} ({v['issue']:.3f})" for n, v in sorted(s["points"].items())))
    L.append(f"issue price: {sat['price']:.4f} SMSP-cycles a uniform instruction (least {sat['least']:.4f} at {sat['set_by']}, "
             f"{sat['run']}, less 2%)" if "price" in sat else "issue price: no stream issues at 0.95 (an issue slot)")
    L.append("beside (T, T without the chains, ΔT a uniform instruction):")
    for k, v in sorted(res["beside"].items()):
        L.append(f"  {k:9s} T {v['T']:8.2f} T0 {v['T0']:8.2f} ΔT/UNI {v['dT_per_uni']:+.3f} UNI rate {v['uni_rate']:.3f}")
    L.append("IMAD forms (SMSP-cycles a warp-iteration):")
    for k, v in res["imad_forms"].items():
        pw = f", {v['per_word']:.3f} a warp-word ({100 * v['imad_form_share']:.0f}% IMAD form)" if "per_word" in v else ""
        L.append(f"  {k:9s} T {v['T']:8.2f}{pw}  {v['counts']}")
    if "imad_hi_cycles" in res:
        L.append(f"IMAD.HI.U32 alone: {res['imad_hi_cycles']:.3f} SMSP-cycles each")
    if "imad_fold_gain" in res:
        L.append(f"duo with some LEA.HI in its IMAD form (f_duom): {100 * res['imad_fold_gain']:+.1f}% time a word saved")
    return "\n".join(L)


def census_lines(res: dict, runs: str) -> list[dict[str, str]]:
    return [{"kind": "UNI", "row": row, "rate": "", "noise": "", "cycles": f"{v['c']:.9g}", "u": "", "since": "",
             "until": "", "runs": runs} for row, v in res["rows"].items() if v["c"] > 0]


def write_census(dev: Path, res: dict, runs: str) -> None:
    """Replace UNI's measured lines in DEVICE_DIR/census.csv in place, the UP row in rows.csv, and put UP in the current
    vector's rows. The data sheet's UNI,DS_* lines and a later vector's refit (a line with `since`) are not this run's
    and stay; a replaced line's `until` carries over to its row's new line."""
    if res["violations"]:
        raise SystemExit("uniform: UNI's lines exceed measured points: " + "; ".join(res["violations"]))
    cen = dev / "census.csv"
    head, *body = cen.read_text().splitlines()

    def own(b: str) -> bool:
        ln = next(csv.DictReader([head, b])) if b.startswith("UNI,") else None
        return ln is not None and not ln["row"].startswith("DS_") and not ln["since"]

    until = {ln["row"]: ln["until"] for ln in csv.DictReader([head, *filter(own, body)]) if ln["until"]}
    at = next((i for i, b in enumerate(body) if own(b)), len(body))
    kept = [b for b in body if not own(b)]
    buf = io.StringIO()
    w = csv.DictWriter(buf, head.split(","), lineterminator="\n")
    for ln in census_lines(res, runs):
        w.writerow(ln | {"until": until.get(ln["row"], "")})
    cen.write_text("\n".join([head, *kept[:at], *buf.getvalue().splitlines(), *kept[at:]]) + "\n")
    rows = dev / "rows.csv"
    head, *body = rows.read_text().splitlines()
    body = [b for b in body if not b.startswith("UP,")]
    alone = ", ".join(f"{k} {v:.3f}" for k, v in sorted(res["uniform_alone"].items()))
    limits = ", ".join(f"{r} {v['limit']:.2f} ({v['set_by']})" for r, v in res["rows"].items()
                       if "limit" in v and not issue_row(r))
    sat = res.get("sat", {})
    issue = (f"costs {sat['price']:.4f} of an issue slot (the least slowdown a uniform instruction adds to an issue-"
             f"saturated stream, {sat['least']:.4f} at {sat['set_by']}, less 2%; JI rows scaled as the ALU's)"
             if "price" in sat else "takes an issue slot (issue and JI rows at the ALU's coefficient)")
    desc = (f"uniform pipe: {res['up']['rate']:.4f} uniform instructions a SMSP-cycle, 2% margin ({res['up']['by']}, the "
            f"fastest form alone; DC's uniform.cuh, each alone: {alone}). A uniform instruction also {issue} and holds "
            f"no other row: no point beside the chains loads one to its limit; the largest each point allows: {limits}")
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(["UP", "measured", runs, desc])
    rows.write_text("\n".join([head, *body]) + "\n" + buf.getvalue())
    current = tomllib.loads((dev / "model.toml").read_text())["current"]
    vec = dev / "vectors" / f"{current}.toml"
    t = vec.read_text()
    if "UP" not in tomllib.loads(t)["rows"]:
        vec.write_text(re.sub(r'^(rows = \[[^\]]*)\]', r'\1, "UP"]', t, count=1, flags=re.M))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["analyze", "census", "tables"])
    ap.add_argument("out", type=Path, nargs="+")
    ap.add_argument("--family", choices=["sm90", "sm100"])
    ap.add_argument("--device", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        for g in a.out:
            print("\n".join(map(str, write_table(g))))
        return 0
    fam = a.family or ("sm100" if any(any((o / "sass").glob("sm100_*.sass")) for o in a.out) else "sm90")
    dev = a.device or HERE.parent / "machine_model" / "data" / "devices" / f"sm_{fam[2:]}"
    res = analyze(a.out, fam, dev / "census.csv", dev / "rows.csv")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    print(text(res))
    if a.cmd == "census":
        runs = " ".join(sorted({json.loads(ln)["run"] for o in a.out
                                for ln in (o / "probes" / "rows.jsonl").read_text().splitlines()}))
        for ln in census_lines(res, runs):
            print(",".join(ln.values()))
        if a.write:
            write_census(dev, res, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
