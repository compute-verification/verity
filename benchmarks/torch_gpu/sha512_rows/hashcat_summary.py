"""hashcat.sh's summary: hashcat's SHA-512 rates as compressions a second against sha512x.cuh's default
compression at the one locked SM clock, the SASS a compression of each, and VERDICT lines.

    python3 hashcat_summary.py OUT      (writes OUT/summary.json, prints the VERDICT lines)

A hashcat benchmark candidate is one SHA-512 block (-m 1700: 7 bytes; -m 1710: 7 bytes and the 12-byte salt; both under
112 bytes with the padding), so hashcat's H/s is its compressions a second. A rate per SM-clock is compressions a second
over SMs x MHz x 1e6. Ours is sx0000 (sha512x.cuh's default force-included in commit_gpu), a0r0l is commit_gpu's own
default build; hashcat_forms is sx0000's compression in hashcat's shapes (its header), and the honest chain and -O's shape
at each CTA size in whole waves (shapes.jsonl). Two rates match within 3%; the verdict takes hashcat's best -O setting
against our best -O-shaped form.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_count

CFGS = ("1700-O", "1700-pure", "1710-O", "1710-pure", "1700-O-w4")
KERNEL = {"1700-O": "m01700_s04", "1700-pure": "m01700_sxx", "1710-O": "m01710_s04", "1710-pure": "m01710_sxx"}
MACHINE = re.compile(r"^(\d+):(\d+):(\d+):(\d+):([0-9.]+):(\d+)$")
SPEED = re.compile(r"Speed\.#\S*\s+([0-9.]+ ?\S*)H/s \(([0-9.]+)ms\) @ Accel:(\d+) Loops:(\d+) Thr:(\d+) Vec:(\d+)")
SAME = 1.03


def read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def read_gz(path: str) -> str:
    try:
        with gzip.open(path, "rt") as f:
            return f.read()
    except OSError:
        return ""


def left_out(path: str, bad: list[tuple[int, str]]) -> None:
    """Say on stderr how many of path's lines didn't parse (its writers append the tools' stderr too) and the first."""
    if bad:
        n, why = bad[0]
        print(f"hashcat_summary: {path}: {len(bad)} line(s) left out, unparsed (first, line {n}: {why})", file=sys.stderr)


def jsonl(path: str) -> list[dict]:
    out, bad = [], []
    for n, line in enumerate(read(path).splitlines(), 1):
        try:
            out.append(json.loads(line))
        except ValueError as e:
            bad.append((n, f"{e}: {line[:80]!r}"))
    left_out(path, bad)
    return out


def med(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def per_clock(gcomp: float | None, sms: int | None, mhz: float | None) -> float | None:
    return gcomp * 1e3 / (sms * mhz) if gcomp and sms and mhz else None


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a and b else None


def fmt(x: float | None, spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def smi_samples(out: str) -> list[tuple[float, float, str]]:
    samples, bad = [], []
    for n, line in enumerate(read(f"{out}/smi.csv").splitlines(), 1):
        p = [x.strip() for x in line.split(",")]
        try:
            ts, _, frac = p[0].partition(".")
            t = time.mktime(time.strptime(ts, "%Y/%m/%d %H:%M:%S")) + float("0." + (frac or "0"))
            samples.append((t, float(p[1].split()[0]), p[5] if len(p) > 5 else ""))
        except (ValueError, IndexError) as e:
            bad.append((n, f"{type(e).__name__} {e}: {line[:80]!r}"))
    left_out(f"{out}/smi.csv", bad)
    return samples


def phases(out: str) -> list[tuple[str, float, float, int]]:
    ph = []
    for line in read(f"{out}/phases.txt").splitlines():
        p = line.split()
        if len(p) == 4:
            ph.append((p[0], float(p[1]), float(p[2]), int(p[3])))
    return ph


def clocks(ph: list, samples: list, pattern: str) -> dict | None:
    """NVML's SM clock in the phases whose name matches pattern: min, median, max and the event reasons seen."""
    rx = re.compile(pattern)
    got = [(m, why) for name, s, e, _ in ph if rx.fullmatch(name) for t, m, why in samples if s <= t <= e]
    if not got:
        return None
    mhz = [m for m, _ in got]
    return {"samples": len(mhz), "min": min(mhz), "median": med(mhz), "max": max(mhz),
            "event_reasons": sorted({why for _, why in got})}


def sass(text: str, name: str, trip: int, per: float) -> dict | None:
    try:
        return sass_count.count("", name, trip, per, text)
    except SystemExit as e:
        return {"error": str(e)}


def brief(c: dict | None) -> str:
    if not c or "error" in c:
        return "n/a"
    p = c["pipes_per_block"]
    loops = f", inner loops {c['inner_loops']} at trip {c['trip']}" if c["inner_loops"] else ""
    return (f"{c['total_per_block']:.0f} (alu {p.get('alu', 0):.0f}: SHF {c['per_block'].get('SHF', 0):.0f}, "
            f"LOP3 {c['per_block'].get('LOP3', 0):.0f}, IADD3 {c['per_block'].get('IADD3', 0):.0f}; "
            f"fma {p.get('fma', 0):.0f}{loops})")


def main(out: str) -> None:
    rc = read(f"{out}/rc.txt")
    timed = next((ln.split(":", 1)[1].split() for ln in rc.splitlines() if ln.startswith("timed:")), [])
    lock = re.search(r"lock rc (\d+)", rc)
    mode = re.search(r"clock mode (.*)", rc)
    ph, samples = phases(out), smi_samples(out)
    fc = jsonl(f"{out}/forms-check.jsonl")
    sms = next((r["sms"] for r in fc if "sms" in r), None)
    if sms is None:
        m = re.search(r"Processor\(s\)\.*:\s*(\d+)", read(f"{out}/hashcat-I.txt"))
        sms = int(m.group(1)) if m else None
    digests = [(r["candidate"], r["digest"]) for r in fc if "digest" in r]
    forms_ok = bool(digests) and all(hashlib.sha512(c.to_bytes(7, "big")).hexdigest() == d for c, d in digests)

    ours = {}
    for t in ("sx0000", "a0r0l"):
        recs = [r["sha"] for r in jsonl(f"{out}/sha-{t}.jsonl") if "sha" in r]
        checks = [r["summary"] for r in jsonl(f"{out}/check-{t}.jsonl") if "summary" in r]
        g, mhz = med([r["gcomp_per_s"] for r in recs]), med([r["mhz"] for r in recs])
        ours[t] = {"timed": t in timed, "check": checks[-1] if checks else None, "rounds": [r["gcomp_per_s"] for r in recs],
                   "gcomp_per_s": g, "mhz": mhz, "per_sm_clock": per_clock(g, sms, mhz),
                   "clock": clocks(ph, samples, rf"sha-{t}-\d+"), "sass": json.loads(read(f"{out}/sass-{t}.json") or "null")}

    forms = {}
    recs = jsonl(f"{out}/forms.jsonl")
    for form in ("chain", "iv", "skip"):
        for roll in (1, 0):
            rs = [r for r in recs if r.get("form") == form and r.get("roll") == roll]
            g, mhz = med([r["gcomp_per_s"] for r in rs]), med([r["mhz"] for r in rs])
            forms[f"{form}-roll{roll}"] = {"rounds": [r["gcomp_per_s"] for r in rs], "gcomp_per_s": g, "mhz": mhz,
                                           "per_sm_clock": per_clock(g, sms, mhz)}
    forms_clock = clocks(ph, samples, r"forms-\d+")

    hc = {}
    for cfg in CFGS:
        rounds = []
        for f in sorted(glob.glob(f"{out}/hc-{cfg}-[0-9]*.txt")):
            for line in read(f).splitlines():
                m = MACHINE.match(line.strip())
                if m:
                    rounds.append({"hps": int(m.group(6)), "corespeed": int(m.group(3)), "exec_ms": float(m.group(5))})
        warm = read(f"{out}/hc-warm-{cfg}-force.txt") or read(f"{out}/hc-warm-{cfg}.txt")
        w = SPEED.search(warm)
        g = med([r["hps"] / 1e9 for r in rounds])
        mhz = med([r["corespeed"] for r in rounds])
        hc[cfg] = {"rounds": rounds, "gcomp_per_s": g, "mhz": mhz, "per_sm_clock": per_clock(g, sms, mhz),
                   "force": os.path.exists(f"{out}/force-{cfg}"), "clock": clocks(ph, samples, rf"hc-{re.escape(cfg)}-\d+"),
                   "warm": dict(zip(("speed", "exec_ms", "accel", "loops", "thr", "vec"), w.groups())) if w else None}

    texts = {os.path.basename(p): read_gz(p) for p in sorted(glob.glob(f"{out}/sass-m017*.txt.gz"))}
    for cfg, name in KERNEL.items():
        vec = int((hc[cfg]["warm"] or {}).get("vec", 1))
        for src, text in texts.items():
            if re.search(rf"Function : {name}\b", text):
                hc[cfg]["sass"] = {"file": src, **(sass(text, name, 1, vec) or {})}
                break
    ftext = read_gz(f"{out}/sass-forms.txt.gz")
    for i, form in enumerate(("chain", "iv", "skip")):
        for roll in (1, 0):
            if ftext:
                forms[f"{form}-roll{roll}"]["sass"] = sass(ftext, f"formsILi{i}ELi{roll}E", 4 if roll else 1, 1)

    grouped: dict[tuple[str, int], list[dict]] = {}
    for r in jsonl(f"{out}/shapes.jsonl"):
        if "threads" in r:
            grouped.setdefault((f"{r['form']}-roll{r['roll']}", r["threads"]), []).append(r)
    shapes = []
    for (name, threads), rs in sorted(grouped.items()):
        g, mhz = med([r["gcomp_per_s"] for r in rs]), med([r["mhz"] for r in rs])
        shapes.append({"form": name, "threads": threads, "ctas_an_sm": rs[0]["ctas_an_sm"], "warps_an_sm": rs[0]["warps_an_sm"],
                       "regs": rs[0]["regs"], "rounds": [r["gcomp_per_s"] for r in rs], "gcomp_per_s": g, "mhz": mhz,
                       "per_sm_clock": per_clock(g, sms, mhz)})

    base = ours["sx0000"] if ours["sx0000"]["gcomp_per_s"] else ours["a0r0l"]
    base_name = "sx0000" if base is ours["sx0000"] else "a0r0l"
    for v in list(hc.values()) + list(forms.values()) + shapes:
        v["over_ours"] = ratio(v["gcomp_per_s"], base["gcomp_per_s"])
        v["over_ours_per_sm_clock"] = ratio(v["per_sm_clock"], base["per_sm_clock"])
    skips = [{"name": f"{k} at 8 CTAs of 256 an SM", **forms[k]} for k in ("skip-roll0", "skip-roll1")]
    skips += [{"name": f"{s['form']} at {s['threads']} a CTA in whole waves", **s} for s in shapes if s["form"].startswith("skip")]
    skip = max(skips, key=lambda v: v["gcomp_per_s"] or 0)
    honest = max([{"name": f"{s['form']} at {s['threads']} a CTA in whole waves", **s} for s in shapes
                  if s["form"].startswith("chain")] or [{"name": None, "gcomp_per_s": None}], key=lambda v: v["gcomp_per_s"] or 0)
    opt_cfg = max(("1700-O", "1710-O", "1700-O-w4"), key=lambda c: hc[c]["gcomp_per_s"] or 0)
    opt_name = f"-m {opt_cfg[:4]} -O -w {4 if opt_cfg.endswith('w4') else 3}"

    summary = {"sms": sms, "lock_mhz_rc": int(lock.group(1)) if lock else None,
               "clock_mode": mode.group(1) if mode else None, "timed": timed, "forms_check": forms_ok,
               "base": base_name, "ours": ours, "forms": forms, "forms_clock": forms_clock, "shapes": shapes,
               "best_skip": skip["name"], "best_honest": honest["name"], "best_hashcat_O": opt_cfg, "hashcat": hc,
               "hashcat_commit": read(f"{out}/hashcat-commit.txt").strip()}
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    timed_mhz = [m for name, s, e, _ in ph if re.fullmatch(r"(sha|forms|hc)-.*-\d+", name)
                 for t, m, _ in samples if s <= t <= e]
    print(f"VERDICT clock: lock rc {summary['lock_mhz_rc']}, {summary['clock_mode']}; NVML in the timed arms "
          f"{fmt(min(timed_mhz) if timed_mhz else None, '.0f')}..{fmt(max(timed_mhz) if timed_mhz else None, '.0f')} MHz "
          f"({len(timed_mhz)} samples at 200 ms); {sms} SMs")
    checks = []
    for t in ("sx0000", "a0r0l"):
        c = ours[t]["check"]
        checks.append(f"{t} reference.py quick {'PASS' if c and c['failed'] == 0 and c['cases'] else 'FAIL'}"
                      f" ({c['cases'] if c else 0} cases, {c['failed'] if c else '-'} failed; timed {ours[t]['timed']})")
    print(f"VERDICT checks: {'; '.join(checks)}; hashcat_forms digests = hashlib.sha512: "
          f"{'PASS' if forms_ok else 'FAIL'} ({len(digests)})")
    for t in ("sx0000", "a0r0l"):
        o = ours[t]
        print(f"VERDICT ours {t} (commit_gpu sha: a variable block a compression, full digest): "
              f"{fmt(o['gcomp_per_s'])} Gcomp/s at {fmt(o['mhz'], '.0f')} MHz (median of {len(o['rounds'])} rounds "
              f"{[round(x, 3) for x in o['rounds']]}), {fmt(o['per_sm_clock'], '.5f')} a SM-clock; SASS a compression "
              f"{brief(o['sass'])}")
    for cfg in CFGS:
        h = hc[cfg]
        m, kern = cfg.split("-")[0], cfg.split("-")[1]
        w = h["warm"] or {}
        print(f"VERDICT hashcat -m {m} {'-O' if kern == 'O' else 'pure'} -w {4 if cfg.endswith('w4') else 3}"
              f"{' --force' if h['force'] else ''} ({KERNEL.get(cfg.replace('-w4', ''), '?')}, Accel {w.get('accel')} "
              f"Loops {w.get('loops')} Thr {w.get('thr')} Vec {w.get('vec')}): {fmt(h['gcomp_per_s'])} GH/s = "
              f"{fmt(h['gcomp_per_s'])} Gcomp/s (1 compression a candidate) at {fmt(h['mhz'], '.0f')} MHz (median of "
              f"{len(h['rounds'])}); hashcat / ours {base_name} = {fmt(h['over_ours'])} (per SM-clock "
              f"{fmt(h['over_ours_per_sm_clock'])}); ours / hashcat = {fmt(ratio(1, h['over_ours']))}"
              + (f"; SASS a candidate {brief(h.get('sass'))}" if h.get("sass") else ""))
    for k, v in forms.items():
        print(f"VERDICT ours in hashcat's shape, hashcat_forms {k}: {fmt(v['gcomp_per_s'])} Gcomp/s at "
              f"{fmt(v['mhz'], '.0f')} MHz, over ours {base_name} {fmt(v['over_ours'])}; SASS a compression "
              f"{brief(v.get('sass'))}")

    for form in sorted({s["form"] for s in shapes}):
        row = [s for s in shapes if s["form"] == form]
        print(f"VERDICT ours in whole waves, hashcat_forms shapes {form} (Gcomp/s at a CTA size; resident warps an SM; "
              f"registers {row[0]['regs']}): "
              + ", ".join(f"{s['threads']}: {fmt(s['gcomp_per_s'])} ({s['warps_an_sm']} warps)" for s in row)
              + f"; best over ours {base_name} {fmt(max(s['over_ours'] or 0 for s in row))}")

    pure = max((hc[c]["over_ours_per_sm_clock"] or 0 for c in ("1700-pure", "1710-pure")), default=0) or None
    opt = hc[opt_cfg]["gcomp_per_s"]
    opt_shape = ratio(opt, skip["gcomp_per_s"])
    whole = (f"; our honest compression in whole waves ({honest['name']}) runs {fmt(honest['gcomp_per_s'])} Gcomp/s, "
             f"{fmt(ratio(honest['gcomp_per_s'], base['gcomp_per_s']))}x ours {base_name} and "
             f"{fmt(ratio(honest['gcomp_per_s'], opt))}x hashcat's best -O") if honest["gcomp_per_s"] else ""
    if pure is None or opt_shape is None or not forms_ok or not base["timed"]:
        print("VERDICT overall: incomplete (a rate, a check or hashcat_forms is missing; summary.json)")
    elif pure <= SAME and opt_shape <= SAME:
        print(f"VERDICT overall: no faster public SHA-512 compression on this GPU: hashcat's full-digest kernels (pure) "
              f"run {fmt(pure)}x ours {base_name} per SM-clock, and its best -O rate ({opt_name}, {fmt(opt)} Gcomp/s, "
              f"{fmt(hc[opt_cfg]['over_ours'])}x ours) is our compression in -O's shape within 3% (-O / hashcat_forms "
              f"{skip['name']} = {fmt(opt_shape)}){whole}: -O's lead is OPTI_TYPE_EARLY_SKIP, ZERO_BYTE, "
              f"PRECOMPUTE_INIT and no feed-forward, which a commitment of variable blocks can't take")
    else:
        print(f"VERDICT overall: hashcat is faster per compression: pure / ours {base_name} = {fmt(pure)} per SM-clock, "
              f"its best -O ({opt_name}) / our compression in its shape ({skip['name']}) = {fmt(opt_shape)}{whole}: "
              f"a port candidate")


if __name__ == "__main__":
    main(sys.argv[1])
