"""The H200 step from `v31 fill-bench`'s JSON lines (Dither's no-fold call, `"fill": true`) at M = 16,384, for each model
whose calls the lines cover, a step per (L', L, ROW, served or filler, variant): each arm's seconds a step, gamma_u against
wgmma.linear's step and gamma_lt against the same lines' cuBLASLt, the split and the residual.

A step's calls by (N, K):
  mqa16k52  a 52-layer MQA model at width 16,384: q and o (16,384, 16,384) twice a layer, kv (128, 16,384), up
            (65,536, 16,384) and down (16,384, 65,536) once, and the head (256, 16,384) once a step
  8b        Llama 3 8B's widths (GQA, 8 KV heads), 32 layers of QKV (6,144, 4,096), O (4,096, 4,096),
            gate and up fused (28,672, 4,096) and down (4,096, 14,336) back to back
The step's arm is the one a call of the protocol runs: a served call's seeds are per call (`dither.seeds(coin, receipt,
call)`), so `ovl_w`, the weight side every call; a filler's is `ovl_pad`, its pad and every coin a call. The split is a
serial call's parts with those seeds: tensor `gemm` (served: cuBLASLt's O, then the protocol GEMM over k < (S - 1) L),
hashing `coins`, `coins_w`, `pad` and `commit`, other `form_a` and `form_b`. gamma_u is against wgmma.linear's step from
the same lines where they time it (`wgl`, a -DV31_WG build's served bench), else against --wgmma's mqa16k52 step. The
residual is 1 - W / (step ρ), W the step's MACs and ρ the H200's rate (4,096 MACs an SM a clock, 132 SMs) at each reading
of its clock in `CLOCKS_MHZ`, side by side: the tensor clock (1,830 MHz, the data sheet's dense INT8 rate) and the boost
clock nvidia-smi reports (1,980 MHz).

Lines with split arms (fill-bench's V31_GSMS: the compute on N of the H200's SMs, the commit on the rest) give each N's
step from its arm of record (`ovl_w_gN`, a filler's `ovl_pad_gN`; N 0 the same calls on every SM), the GEMM alone on N SMs
against the whole GPU's (`gemm_gN` over `gemm`), the commit alone on the rest (`commit_gN`), the hashing hidden (the serial
split's sum less the step, over its hashing), and the step with each call's best N (that call's shape on its own split).

`v31 fill-step`'s lines (`"fill_step": true`, run.sh's FILL_STEPS) time a step's calls in served order: each shape's calls
on its own two copies, every commit beside the later calls' compute, as far behind as its shape's two copies allow, for
each N of its splits (`step_gN`). The model is the one whose calls its shapes' are; gamma_u is against wgmma.linear's step
from the same variant's fill-bench lines (at its L', L and ROW; else from the one variant that differs only in gsm, gsms,
gcalls and item, which move only the overlapped arms), the hashing hidden is their serial split less the step, over its
hashing, and beside it is their step at that N (each shape's calls back to back).

Where the lines carry the board's energy (each arm's `j`, joules a call, and `w`, its watts; `power_limit_w`), the split
and served-order tables give each step's watts and joules, the served-order table its power floor (its joules over the
limit: no faster at the same energy), and the summaries the serial split's joules and their time at the limit.

    python3 h200_step.py BENCH.jsonl ... [STEP.jsonl ...] [--wgmma SECONDS]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LAYERS = 52
CALLS = {(16384, 16384): 2 * LAYERS, (128, 16384): LAYERS, (65536, 16384): LAYERS, (16384, 65536): LAYERS,
         (256, 16384): 1}
MODELS = {"mqa16k52": CALLS, "8b": {(6144, 4096): 32, (4096, 4096): 32, (28672, 4096): 32, (4096, 14336): 32}}
#: the H200's SM clock under each reading; one constant until the card file states the tensor clock
CLOCKS_MHZ = {"tensor": 1830, "boost": 1980}
RHO = {name: 4096 * 132 * mhz * 1e6 for name, mhz in CLOCKS_MHZ.items()}
MHZ = " / ".join(f"{m:,}" for m in CLOCKS_MHZ.values())
FILL_ARMS = ("lt", "coins", "coins_w", "pad", "form_a", "form_b", "gemm", "commit", "call", "ovl", "ovl_w", "ovl_pad", "wgl")


def rows(paths: list[str]) -> list[dict]:
    """Each bench line, with its variant: the file's name past `fill-bench-` (run.sh writes a variant's lines to one
    file)."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.startswith("{") and '"gamma_ovl"' in line:
                out.append(json.loads(line) | {"variant": Path(p).stem.removeprefix("fill-").removeprefix("bench-")})
    return out


def step_rows(paths: list[str]) -> list[dict]:
    """Each fill-step line, with its variant: the file's name past `fill-step-` (run.sh writes a variant's lines to one
    file, as fill-bench's)."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.startswith("{") and '"fill_step"' in line:
                out.append(json.loads(line) | {"variant": Path(p).stem.removeprefix("fill-step-")})
    return out


def unsplit(variant: str) -> str:
    """A variant without the knobs that move only its overlapped arms (gsm, gsms, gcalls, item): fill-bench's serial
    arms and wgl run on every SM whatever they are."""
    return "-".join(t for t in variant.split("-") if not re.match(r"(gsms?|gcalls|item)\d", t))


def served_order(steps_: list[dict], lines: list[dict], m: int = 16384, wg: dict[str, float | None] | None = None
                 ) -> list[dict]:
    """Each fill-step line's model (None: its shapes' calls are no model's at M m) and for each N it timed: the step, the
    SMs, gamma_u (against its variant's fill-bench wgl, else wg's), the hashing hidden (their serial split less the step,
    over its hashing) and their step at that N (`apart`, each shape's calls back to back). Its variant's fill-bench lines
    are those of the same variant, else of the one variant that is the same `unsplit` (`ref_variant`)."""
    wg = wg or {}
    out = []
    for r in steps_:
        calls = {(s["n"], s["k"]): s["calls"] for s in r["shapes"]}
        model = next((x for x, c in MODELS.items() if c == calls and all(s["m"] == m for s in r["shapes"])), None)
        key = (r["lp"], r["L"], r["row"], bool(r.get("o")), r.get("variant", ""))
        fbs = fill_steps(lines, m, MODELS[model]) if model else {}
        if key not in fbs:
            near = [k for k in fbs if k[:4] == key[:4] and unsplit(k[4]) == unsplit(key[4])]
            key = near[0] if len(near) == 1 else key
        fb = fbs.get(key)
        sp = split_steps(lines, m, MODELS[model]).get(key) if model else None
        wgl = fb["s"]["wgl"] if fb else None
        ref = wgl or wg.get(model)
        serial = fb["tensor"] + fb["hashing"] + fb["other"] if fb else None
        limit = r.get("power_limit_w") or None
        serial_j = serial_joules(fb) if fb else None
        per = {}
        for g in r["gsms"]:
            a = r["arms"].get(f"step_g{g['n']}")
            if not a:
                continue
            step, j = a["ms"] / 1e3, a.get("j")
            apart = sp["n"][g["n"]]["step"] if sp and g["n"] in sp["n"] else fb["step"] if fb and not g["n"] else None
            per[g["n"]] = {"step": step, "sms": tuple(g["sms"]), "mhz": a.get("mhz") or None,
                           "gamma_u": step / ref if ref else None, "apart": apart,
                           "hidden": (serial - step) / fb["hashing"] if fb and fb["hashing"] else None,
                           "w": a.get("w"), "j": j, "floor": j / limit if j and limit else None}
        out.append({"model": model, "key": (r["lp"], r["L"], r["row"], bool(r.get("o")), r.get("variant", "")),
                    "ref_variant": key[4] if fb else None, "spec": r["step"], "ref": ref, "wgl": wgl, "serial": serial,
                    "hashing": fb["hashing"] if fb else None, "steady": r.get("steady"), "limit": limit,
                    "serial_j": serial_j, "serial_floor": serial_j / limit if serial_j and limit else None, "n": per})
    return out


def fill_steps(lines: list[dict], m: int = 16384, calls: dict[tuple[int, int], int] = CALLS
               ) -> dict[tuple[int, int, int, bool, str], dict]:
    """Per (L', L, ROW, served, variant) of fill-bench's lines: each arm's seconds a step (None where a call lacks the
    arm), the step's arm and its seconds, the split and the step's MACs, for groups that timed every call. L 0 is every
    call at L = K: a line at L >= K (S = 1) stands in it too, as fill-bench records the L it ran (K) and fill-step the 0
    it took."""
    groups: dict[tuple[int, int, int, bool, str], dict[tuple[int, int], dict]] = {}
    for r in lines:
        if r.get("fill") and r["m"] == m and (r["n"], r["k"]) in calls:
            for L in {r["L"], 0} if r["L"] >= r["k"] else {r["L"]}:
                groups.setdefault((r["lp"], L, r["row"], bool(r.get("o")), r.get("variant", "")), {})[(r["n"], r["k"])] = r
    out = {}
    for key, by in sorted(groups.items()):
        if set(by) != set(calls):
            continue
        ms = {a: [by[nk]["arms"].get(a, {}).get("ms") for nk in calls] for a in FILL_ARMS}
        sec = {a: None if None in v else sum(calls[nk] * x for nk, x in zip(calls, v)) / 1e3 for a, v in ms.items()}
        part = lambda *arms: sum(sec[a] or 0.0 for a in arms)  # noqa: E731
        rec = "ovl_w" if key[3] else "ovl_pad"
        mhz = [by[nk]["arms"][rec]["mhz"] for nk in calls if by[nk]["arms"].get(rec, {}).get("mhz")]
        out[key] = {"s": sec, "j": step_joules(by, calls, FILL_ARMS), "record": rec, "step": sec[rec],
                    "tensor": part("gemm"), "hashing": part("coins", "coins_w", "pad", "commit"),
                    "other": part("form_a", "form_b"), "macs": sum(calls[nk] * m * nk[0] * nk[1] for nk in calls),
                    "mhz": (min(mhz), max(mhz)) if mhz else None, "by": by}
    return out


def step_joules(by: dict[tuple[int, int], dict], calls: dict[tuple[int, int], int], arms) -> dict[str, float | None]:
    """Each arm's joules a step, the board's energy over its calls (None where a call lacks the arm or its energy)."""
    j = {a: [by[nk]["arms"].get(a, {}).get("j") for nk in calls] for a in arms}
    return {a: None if None in v else sum(calls[nk] * x for nk, x in zip(calls, v)) for a, v in j.items()}


def serial_joules(v: dict) -> float | None:
    """A fill_steps group's serial split in joules a step: its tensor, hashing and other arms' (None: an arm's energy
    is unknown; a filler's coins_w and a served call's pad, which it doesn't run, count as none)."""
    parts = [v["j"][a] for a in ("gemm", "coins", "coins_w", "pad", "commit", "form_a", "form_b") if v["s"][a] is not None]
    return None if None in parts else sum(parts)


def power(serial_j: float | None, limit: float | None) -> str:
    """A summary's clause on the serial split's energy: its joules a step and their time at the board's power limit, a
    floor under any step of the same work that spends no less energy."""
    if not serial_j or not limit:
        return ""
    return f"; the serial split {serial_j:.0f} J a step, {serial_j / limit:.3f} s at the {limit:.0f} W limit"


def split_steps(lines: list[dict], m: int = 16384, calls: dict[tuple[int, int], int] = CALLS
                ) -> dict[tuple[int, int, int, bool, str], dict]:
    """Per (L', L, ROW, served, variant) of fill-bench's lines with split arms: for each N every call timed, its step and
    hidden hashing, its GEMM and commit seconds a step on the split's sides and their SM counts; and the step with each
    call on its fastest N (`best`: N by call, and its step)."""
    out = {}
    for key, v in fill_steps(lines, m, calls).items():
        by = v["by"]
        ns = sorted({g["n"] for r in by.values() for g in r.get("gsms", [])})
        rec = v["record"]
        serial = v["tensor"] + v["hashing"] + v["other"]
        per: dict[int, dict] = {}
        for n in ns:
            ms = {a: [by[nk]["arms"].get(f"{a}_g{n}", {}).get("ms") for nk in calls] for a in (rec, "gemm", "commit")}
            if None in ms[rec]:
                continue
            sec = {a: None if None in x else sum(calls[nk] * y for nk, y in zip(calls, x)) / 1e3 for a, x in ms.items()}
            sms = next((tuple(g["sms"]) for r in by.values() for g in r.get("gsms", []) if g["n"] == n), None)
            j = step_joules(by, calls, (f"{rec}_g{n}",))[f"{rec}_g{n}"]
            per[n] = {"step": sec[rec], "gemm": sec["gemm"], "commit": sec["commit"], "sms": sms, "j": j,
                      "w": j / sec[rec] if j else None,
                      "hidden": (serial - sec[rec]) / v["hashing"] if v["hashing"] else None,
                      "slowdown": sec["gemm"] / v["tensor"] if sec["gemm"] and v["tensor"] else None}
        if not per:
            continue
        pick = {nk: min(per, key=lambda n: by[nk]["arms"][f"{rec}_g{n}"]["ms"]) for nk in calls}
        best = sum(calls[nk] * by[nk]["arms"][f"{rec}_g{pick[nk]}"]["ms"] for nk in calls) / 1e3
        limit = next((r["power_limit_w"] for r in by.values() if r.get("power_limit_w")), None)
        out[key] = {"record": rec, "serial": serial, "hashing": v["hashing"], "tensor": v["tensor"], "wgl": v["s"]["wgl"],
                    "lt": v["s"]["lt"], "n": per, "pick": {by[nk]["shape"]: pick[nk] for nk in calls}, "best": best,
                    "best_hidden": (serial - best) / v["hashing"] if v["hashing"] else None,
                    "serial_j": serial_joules(v), "limit": limit}
    return out


def residual(macs: int, step: float | None) -> str:
    """1 - W / (step ρ) at each reading of the clock, in `CLOCKS_MHZ`'s order."""
    return " / ".join(f"{1 - macs / (step * rho):.3f}" for rho in RHO.values()) if step else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--wgmma", type=float, default=None, help="wgmma.linear's mqa16k52 step, s")
    a = ap.parse_args()
    lines, wg, found = rows(a.files), {"mqa16k52": a.wgmma}, False
    for model, calls in MODELS.items():
        st = fill_steps(lines, calls=calls)
        if not st:
            continue
        found, ref = True, wg.get(model)
        print(f"\n{model}, Dither's no-fold call\n\n| L' | L | ROW | kind | variant | step s | arm | gamma_u | gamma_lt | "
              "ovl s | ovl_w s | ovl_pad s | call s | tensor s | hashing s | other s | hashing share | cuBLASLt s | "
              f"residual at {MHZ} MHz | SM MHz |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        for (lp, L, row, o, var), v in st.items():
            s, step = v["s"], v["step"]
            mhz = f"{v['mhz'][0]:.0f}-{v['mhz'][1]:.0f}" if v["mhz"] else "-"
            r = s["wgl"] or ref
            gu = f"{step / r:.3f}" if r and step else "-"
            split = v["tensor"] + v["hashing"] + v["other"]
            print(f"| {lp} | {L} | {row} | {'served' if o else 'filler'} | {var} | {f(step)} | {v['record']} | {gu} | "
                  f"{f(step / s['lt'] if step else None)} | {f(s['ovl'])} | {f(s['ovl_w'])} | {f(s['ovl_pad'])} | "
                  f"{f(s['call'])} | {v['tensor']:.3f} | {v['hashing']:.3f} | {v['other']:.3f} | "
                  f"{v['hashing'] / split:.1%} | {s['lt']:.3f} | {residual(v['macs'], step)} | "
                  f"{mhz} |")
        for (lp, L, row, o, var), v in st.items():
            step, lt, wl = v["step"], v["s"]["lt"], v["s"]["wgl"]
            vs = (f"gamma_u {step / wl:.3f} against wgmma.linear's {wl:.3f} s (these lines' wgl), " if wl and step
                  else f"gamma_u {step / ref:.3f} against wgmma's {ref} s, " if ref and step else "")
            print(f"summary: {model} no-fold {'served' if o else 'filler'} L {L} L' {lp} row {row} {var}: step "
                  f"{f(step)} s ({v['record']}), {vs}gamma_lt {f(step / lt if step else None)} against cuBLASLt's "
                  f"{lt:.3f} s; tensor {v['tensor']:.3f}, hashing {v['hashing']:.3f}, other {v['other']:.3f} s serial")
    for model, calls in MODELS.items():
        sp = split_steps(lines, calls=calls)
        if not sp:
            continue
        ref = wg.get(model)
        print(f"\n{model}, Dither's no-fold call on a split (the compute on N SMs, the commit on the rest)\n\n"
              "| L' | L | ROW | kind | variant | N | SMs | step s | gamma_u | gemm on N s | GEMM slowdown | commit on the rest s "
              "| hashing hidden | W | J a step |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        f = lambda x, p=".3f": "-" if x is None else format(x, p)  # noqa: E731
        for (lp, L, row, o, var), v in sp.items():
            r = v["wgl"] or ref
            for n, x in v["n"].items():
                sms = f"{x['sms'][0]} + {x['sms'][1]}" if x["sms"] and n else "every SM"
                print(f"| {lp} | {L} | {row} | {'served' if o else 'filler'} | {var} | {n} | {sms} | {x['step']:.3f} | "
                      f"{f(x['step'] / r if r else None)} | {f(x['gemm'])} | {f(x['slowdown'])} | {f(x['commit'])} | "
                      f"{f(x['hidden'], '.1%')} | {f(x['w'], '.0f')} | {f(x['j'], '.0f')} |")
        for (lp, L, row, o, var), v in sp.items():
            r, rn = v["wgl"] or ref, "these lines' wgl" if v["wgl"] else "wgmma's"
            n1 = min(v["n"], key=lambda n: v["n"][n]["step"])
            one = v["n"][n1]
            gu = lambda x: f", gamma_u {x / r:.3f}" if r else ""  # noqa: E731
            against = f", gamma_u against {rn} {r:.3f} s" if r else ""
            print(f"summary: {model} no-fold {'served' if o else 'filler'} split L {L} L' {lp} row {row} {var}: best N "
                  f"{n1} step {one['step']:.3f} s{gu(one['step'])}, hashing hidden {f(one['hidden'], '.1%')}; each call's "
                  f"N ({', '.join(f'{s} {n}' for s, n in v['pick'].items())}) step {v['best']:.3f} s{gu(v['best'])}, hashing "
                  f"hidden {f(v['best_hidden'], '.1%')}; serial {v['serial']:.3f} s (hashing {v['hashing']:.3f} s){against}"
                  f"{power(v['serial_j'], v['limit'])}")
    so = served_order(step_rows(a.files), lines, wg=wg)
    if so:
        found = True
        f = lambda x, p=".3f": "-" if x is None else format(x, p)  # noqa: E731
        print("\nDither's no-fold calls in served order (each shape's calls on its own two copies, every commit beside the "
              "later calls' compute)\n\n| model | L' | L | ROW | kind | variant | N | SMs | step s | gamma_u | fill-bench's "
              "step s | serial s | hashing hidden | SM MHz | W | J a step | power floor s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for v in so:
            lp, L, row, o, var = v["key"]
            for n, x in v["n"].items():
                sms = f"{x['sms'][0]} + {x['sms'][1]}" if n else "every SM"
                print(f"| {v['model'] or v['spec']} | {lp} | {L} | {row} | {'served' if o else 'filler'} | {var} | {n} | "
                      f"{sms} | {x['step']:.3f} | {f(x['gamma_u'])} | {f(x['apart'])} | {f(v['serial'])} | "
                      f"{f(x['hidden'], '.1%')} | {f(x['mhz'], '.0f')} | {f(x['w'], '.0f')} | {f(x['j'], '.0f')} | "
                      f"{f(x['floor'])} |")
        for v in so:
            lp, L, row, o, var = v["key"]
            rn = "these lines' wgl" if v["wgl"] else "wgmma's"
            if v["ref_variant"] and v["ref_variant"] != var:
                rn = f"{v['ref_variant']}'s wgl" if v["wgl"] else rn
            against = f", gamma_u against {rn} {v['ref']:.3f} s" if v["ref"] else ""
            for n, x in v["n"].items():
                gu = f", gamma_u {x['gamma_u']:.3f}" if x["gamma_u"] else ""
                pw = (f"; {x['w']:.0f} W, {x['j']:.0f} J a step, its floor at the {v['limit']:.0f} W limit "
                      f"{x['floor']:.3f} s" if x["floor"] else "")
                print(f"summary: {v['model'] or v['spec']} served order {'served' if o else 'filler'} L {L} L' {lp} row "
                      f"{row} {var} N {n}: step {x['step']:.3f} s{gu}, hashing hidden {f(x['hidden'], '.1%')}; "
                      f"fill-bench's step at N {n} {f(x['apart'])} s; serial {f(v['serial'])} s (hashing "
                      f"{f(v['hashing'])} s){against}{pw}{power(v['serial_j'], v['limit'])}"
                      f"{'' if v['steady'] else '; clock not steady'}")
    if not found:
        print("summary: no complete step (a bench shape is missing)")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
