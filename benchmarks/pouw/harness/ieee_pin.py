"""The SASS gate's numeric pin (`sass_gate.py`; each toolkit's ``numeric`` in `sass_pins.json`): builds `ieee_pin.cu` with
every pinned toolkit, with each pinned flag set and with the negative controls, gates the builds, runs them on one leased
GPU, and judges them.

  research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --env GPU_LEASE_WHO=<you> \
      -- python3 benchmarks/pouw/harness/ieee_pin.py [--toolkit NAME=CUDA_HOME ...]

Toolkits: 12.9.1 from `jobs/toolkit.sh`, 13.0.1 at /usr/local/cuda, unless --toolkit names them; each must be a toolkit of
`sass_pins.json`, and every one given is judged. A toolkit's target is its name's suffix (13.0.1-sm_90a builds sm_90a and
runs on an sm_90 card; no suffix, sm_120a), and one run's toolkits share one target, since one leased GPU runs every binary.
`jobs/pin_target.sh` pins a new one and runs this on it. Per toolkit, on the CPU:
- nvcc and ptxas ``--version``, and each build's full command, one build at a time: `PINNED` flag sets (the pins' arguments)
  and `CONTROLS` (--use_fast_math: .FTZ and div.full/sqrt.approx; -ftz=true: .FTZ alone; -prec-div=false
  -prec-sqrt=false: the approximations alone);
- the toolkit's entry in `sass_pins.json` must be what the pinned builds pin (`sass_gate.update_pins`), so every pinned
  body is one this run exercises;
- every pinned build passes the gate (against the pins with this run's numeric pending) with every sequence (`SEQUENCES`)
  and every pinned subroutine covered, and every control fails it.
Then `gpu-lease 1 --wait` runs every binary (this script's --run) on the one GPU it leases, whose UUID each binary must
report, and releases it. A pinned build passes when every check of `CASES` (and `UNITS_CASES` on an sm_90 card) ran all its cases (and skipped only division by
zero or overflow), none differs from its reference, and every float check saw subnormal inputs and, but for sqrt,
subnormal outputs; a control must differ somewhere. Writes $RESEARCH_RUN_DIR/ieee-pin.json (--out); exits 0 only if every
toolkit passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sass_gate as SG  # noqa: E402

SOURCE = HERE / "ieee_pin.cu"
ARCHS = {"sm_120a": 120, "sm_90a": 90}


def arch_of(toolkit: str) -> str:
    arch = toolkit.rsplit("-", 1)[1] if "-sm_" in toolkit else "sm_120a"
    if arch not in ARCHS:
        raise PinFailed(f"toolkit {toolkit}: no target {arch} (one of {sorted(ARCHS)})")
    return arch


def nvcc_args(arch: str) -> list[str]:
    return ["-O3", "-std=c++17", "-gencode", f"arch=compute_{arch[3:]},code={arch}", "--no-compress"]


NVCC_ARGS = nvcc_args("sm_120a")
PINNED = [[], ["-fmad=false"]]
CONTROLS = [["--use_fast_math"], ["-ftz=true"], ["-prec-div=false", "-prec-sqrt=false"]]
SEQUENCES = {"rcp-fast-path", "sqrt-fast-path", "div-fast-path", "int-seed"}
DEFAULT_HOMES = {"13.0.1": "/usr/local/cuda"}
LEASE_MINUTES = 30
_ALL, _HOST = 1 << 32, (1 << 24) - 2 + (1 << 25)
#: each check ieee_pin.cu prints, and how many inputs it covers (cases plus skipped)
CASES = {"rcp": _ALL, "sqrt": _ALL, "div": _ALL, "cdiv3": _ALL, "cdiv": _ALL, "u16": _ALL, "u16_call": _ALL, "u32": _ALL,
         "s32": _ALL, "u32n": _ALL, "u64": 1 << 30, "s64": 1 << 30, "host_rcp": _HOST, "host_sqrt": _HOST,
         "host_div": _HOST, "host_cdiv3": _HOST}
#: the checks of the bodies ptxas builds for kernels' units.cu (ieee_pin.cu's units_checks), which only an sm_90 card runs
UNITS_CASES = {"units_sqrt": _ALL, "units_rms_mean": _ALL, "units_rms_rs": _ALL, "units_soft_one": _ALL,
               "units_soft_div": _ALL}
FLOAT = {"rcp", "sqrt", "div", "cdiv3", "cdiv", "host_rcp", "host_sqrt", "host_div", "host_cdiv3", *UNITS_CASES}
SQRT = {"sqrt", "host_sqrt", "units_sqrt"}  # a square root is never subnormal


class PinFailed(RuntimeError):
    pass


def _run(cmd: list[str], **kw) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise PinFailed(f"{' '.join(cmd)} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-600:]}")
    return r.stdout


def toolkit_sh_home() -> str:
    """CUDA 12.9.1's home, assembled by jobs/toolkit.sh if absent."""
    out = _run(["bash", "-c", f'. "{HERE / "jobs" / "toolkit.sh"}" >&2 && printf %s "$CUDA_HOME"'])
    return out.strip()


def homes(specs: list[str], pins: dict) -> dict[str, str]:
    if specs:
        out = dict(s.split("=", 1) for s in specs)
    else:
        out = {"12.9.1": toolkit_sh_home(), **DEFAULT_HOMES}
    unknown = sorted(set(out) - set(pins["toolkits"]))
    if unknown:
        raise PinFailed(f"no toolkit {unknown} in sass_pins.json")
    if len({arch_of(tk) for tk in out}) > 1:
        raise PinFailed(f"toolkits {sorted(out)} name more than one target; one GPU runs every binary of a run")
    return out


def _covered(rec: dict) -> dict:
    return {"ok": rec["ok"], "flagged": rec["flagged"], "covered": rec["covered"], "offending_n": rec["offending_n"],
            "merged_reads_n": rec["merged_reads_n"], "ptx_bad_n": rec["ptx_bad_n"],
            "why": None if rec["ok"] else SG.why_failed(rec),
            "tool_notes": [n for c in rec["cubins"] for n in c["tool_notes"]]}


def build_toolkit(tk: str, home: str, work: Path, pins_path: Path, pending: Path) -> dict:
    """Stage 1 for one toolkit: versions, builds, the pins entry and the gate."""
    bin_dir = Path(home) / "bin"
    rec = {"home": home, "nvcc_version": _run([str(bin_dir / "nvcc"), "--version"]).strip(),
           "ptxas_version": _run([str(bin_dir / "ptxas"), "--version"]).strip(), "builds": [], "problems": []}
    old = os.environ.get("POUW_CUDA_BIN")
    os.environ["POUW_CUDA_BIN"] = str(bin_dir)
    try:
        tools = SG.find_tools()
    finally:
        os.environ.pop("POUW_CUDA_BIN") if old is None else os.environ.__setitem__("POUW_CUDA_BIN", old)
    rec["tools"] = tools
    for control, flags in [(False, f) for f in PINNED] + [(True, f) for f in CONTROLS]:
        out = work / f"{tk}{''.join(f.replace('=', '_') for f in flags) or '-default'}"
        cmd = [str(bin_dir / "nvcc"), *nvcc_args(arch_of(tk)), *flags, str(SOURCE), "-o", str(out)]
        _run(cmd, cwd=work)
        b = {"flags": flags, "control": control, "command": cmd, "binary": str(out),
             "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        b["gate"] = _covered(SG.scan(out, tools, None, pending))
        if control and b["gate"]["ok"]:
            rec["problems"].append(f"control {flags} passes the gate")
        if not control:
            missing = sorted((SEQUENCES | set(SG.load_pins(pins_path)["toolkits"][tk]["subroutines"]))
                             - set(b["gate"]["covered"]))
            if not b["gate"]["ok"]:
                rec["problems"].append(f"pinned build {flags} fails the gate: {b['gate']['why']}")
            if missing:
                rec["problems"].append(f"pinned build {flags} covers no {missing}")
        rec["builds"].append(b)
    scratch = work / f"{tk}-repin.json"
    shutil.copyfile(pins_path, scratch)
    entry = SG.update_pins(SG.pin_record([Path(b["binary"]) for b in rec["builds"] if not b["control"]], tools), tk, scratch)
    want = SG.load_pins(pins_path)["toolkits"][tk]
    rec["entry"] = {**entry, "numeric": None}
    if {**entry, "numeric": None} != {**want, "numeric": None}:
        rec["problems"].append(f"sass_pins.json's {tk} isn't what its pinned builds pin: {json.dumps(rec['entry'])}")
    return rec


def run_binaries(work: Path) -> None:
    """Stage 2, inside the lease: every binary of work/builds.json, one at a time, into work/runs.json."""
    lease = os.environ.get("GPU_LEASE_UUID", "")
    if not lease:
        raise PinFailed("no GPU_LEASE_UUID: run under gpu-lease")
    runs = {}
    for b in json.loads((work / "builds.json").read_text()):
        r = subprocess.run([b["binary"]], capture_output=True, text=True, env={**os.environ,
                                                                               "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        lines = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
        runs[b["binary"]] = {"exit": r.returncode, "stderr": r.stderr[-2000:], "lines": lines, "lease_uuid": lease,
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (work / "runs.json").write_text(json.dumps(runs, indent=1))


def judge_run(run: dict, control: bool, sm: int = ARCHS["sm_120a"]) -> list[str]:
    """What's wrong with one binary's run: [] when a pinned build is exact everywhere on an sm_`sm` card, or a control
    differs somewhere."""
    lines = run.get("lines", [])
    dev = next((ln for ln in lines if "device" in ln), None)
    if dev is None:
        return [f"no device line (exit {run.get('exit')}): {run.get('stderr', '')[-300:]}"]
    problems = []
    if dev.get("uuid", "").lower() != run.get("lease_uuid", "").lower():
        problems.append(f"ran on {dev.get('uuid')}, not the leased {run.get('lease_uuid')}")
    if dev.get("sm") != sm:
        problems.append(f"ran on sm_{dev.get('sm')}, not sm_{sm}")
    checks = {ln["check"]: ln for ln in lines if "check" in ln}
    if control:
        if not any(c.get("bad", 0) > 0 for c in checks.values()):
            problems.append("a control build is exact everywhere: the checks can't see what the gate refuses")
        return problems
    if run.get("exit") != 0:
        problems.append(f"exit {run.get('exit')}: {run.get('stderr', '')[-300:]}")
    cases = {**CASES, **(UNITS_CASES if sm == ARCHS["sm_90a"] else {})}
    for name, n in cases.items():
        c = checks.get(name)
        if c is None:
            problems.append(f"no {name} check")
            continue
        if c["cases"] + c["skipped"] != n or c["cases"] == 0:
            problems.append(f"{name}: {c['cases']} cases and {c['skipped']} skipped, not {n} inputs")
        if c["bad"]:
            problems.append(f"{name}: {c['bad']} differ from the reference, first {c['first']}")
        if name in FLOAT and (c["skipped"] or not c["subnormal_in"] or (name not in SQRT and not c["subnormal_out"])):
            problems.append(f"{name}: {c['subnormal_in']} subnormal inputs, {c['subnormal_out']} subnormal outputs, "
                            f"{c['skipped']} skipped")
    extra = sorted(set(checks) - set(cases))
    if extra:
        problems.append(f"unknown checks {extra}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--toolkit", action="append", default=[], metavar="NAME=CUDA_HOME")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "ieee-pin.json")
    ap.add_argument("--pins", type=Path, default=SG.PINS)
    ap.add_argument("--run", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.run:
        run_binaries(args.run)
        return 0
    pins = SG.load_pins(args.pins)
    run_id = os.environ.get("RESEARCH_RUN_ID")
    result = {"version": "ieee-pin/v1", "run": run_id, "source_sha": os.environ.get("RESEARCH_SOURCE_SHA"),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "gate": SG.VERSION,
              "pins_sha256": hashlib.sha256(args.pins.read_bytes()).hexdigest(), "toolkits": {}, "ok": False}
    with tempfile.TemporaryDirectory(prefix="ieee-pin-") as td:
        work = Path(td)
        pending = work / "pins-pending.json"
        p = json.loads(args.pins.read_text())
        for t in p["toolkits"].values():
            t["numeric"] = {"pending": run_id or "this run"}
        pending.write_text(json.dumps(p))
        for tk, home in homes(args.toolkit, pins).items():
            result["toolkits"][tk] = build_toolkit(tk, home, work, args.pins, pending)
        builds = [b for t in result["toolkits"].values() for b in t["builds"]]
        (work / "builds.json").write_text(json.dumps(builds))
        if any(t["problems"] for t in result["toolkits"].values()):
            result["lease_exit"] = None
        else:
            result["lease_exit"] = subprocess.run(["gpu-lease", "1", "--wait", "--max-min", str(LEASE_MINUTES), "--",
                                                   sys.executable, str(Path(__file__).resolve()), "--run",
                                                   str(work)]).returncode
        runs = json.loads((work / "runs.json").read_text()) if (work / "runs.json").exists() else {}
        for tk, t in result["toolkits"].items():
            for b in t["builds"]:
                b["run"] = runs.get(b["binary"], {"exit": None, "lines": []})
                b["problems"] = judge_run(b["run"], b["control"], ARCHS[arch_of(tk)])
                b["binary"] = Path(b["binary"]).name
            t["ok"] = not t["problems"] and not any(b["problems"] for b in t["builds"])
    result["ok"] = result["lease_exit"] == 0 and bool(result["toolkits"]) and all(t["ok"] for t in result["toolkits"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for tk, t in result["toolkits"].items():
        print(f"{tk}: {'pass' if t['ok'] else 'FAIL'}", *t["problems"],
              *(f"{' '.join(b['flags']) or 'default'}: {p}" for b in t["builds"] for p in b["problems"]), sep="\n  ")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PinFailed, SG.SassGateFailed) as e:
        print(f"ieee_pin: {e}", file=sys.stderr)
        sys.exit(1)
