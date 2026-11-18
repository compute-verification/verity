"""The smoke's device_libs.gate: the harness's SASS gate on the shared objects under --home that the dynamic loader's record
(LD_DEBUG=files with LD_DEBUG_OUTPUT=RECORD/ld, one file a process) says the device steps mapped. A library the run reused
from the cache is gated, since the run executed it; one in the cache that no process mapped is not. A shared object with
no .nv_fatbin section is a host-only module, named and passed over; one that is unreadable fails. The last line is the
step's reason; exits nonzero on a failure."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SASS_GATE = Path(__file__).resolve().parents[2] / "verity/protocols/resource_accounting/compute/pouw/prover/sass_gate.py"
MAPPED = re.compile(r"file=(\S+) \[\d+\];  generating link map")


def _sass_gate():
    spec = importlib.util.spec_from_file_location("sass_gate", SASS_GATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sass_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def loaded(record: Path, home: Path) -> list[Path]:
    """The shared objects under `home` that a process whose loader wrote under `record` mapped."""
    home = home.resolve()
    seen = set()
    for f in record.glob("ld.*"):
        for m in MAPPED.finditer(f.read_text(errors="replace")):
            p = Path(os.path.realpath(m[1]))
            if p.is_relative_to(home):
                seen.add(p)
    return sorted(seen)


def readable_elf(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(5) == b"\x7fELF\x02"
    except OSError:
        return False


def label(p: Path) -> str:
    return f"{p.name} ({p.parent.parent.name}/{p.parent.name})"


def summary(b: dict) -> str:
    """A gated library: pinned (its ptxas cubins' toolkits) or unpinned, its flagged count and what covers them."""
    ptxas = [c for c in b["cubins"] if any(t["tool"] == "ptxas" for t in c["tool_notes"])]
    pinned = bool(ptxas) and all(c.get("toolkit") for c in ptxas)
    pin = "pinned " + "/".join(sorted({c["toolkit"] for c in ptxas})) if pinned else "unpinned"
    covered = ", ".join(f"{k} {n}" for k, n in sorted(b.get("covered", {}).items()))
    return f"{label(Path(b['path']))} {pin}, {b['flagged']} flagged" + (f", covered {covered}" if covered else "")


def fail(why: str) -> int:
    print(why)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--home", type=Path, required=True, help="the device steps' HOME, which holds their builds' caches")
    ap.add_argument("--record", type=Path, required=True, help="the directory LD_DEBUG_OUTPUT wrote under")
    ap.add_argument("--cuda-bin", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True, help="the gate's record")
    a = ap.parse_args(argv)

    if not any(a.record.glob("ld.*")):
        return fail(f"no loader record under {a.record}: the device steps ran without LD_DEBUG_OUTPUT there")
    libs = loaded(a.record, a.home)
    for p in libs:
        print(f"loaded: {p}")
    for p in sorted(set(Path(os.path.realpath(q)) for q in a.home.rglob("*.so")) - set(libs)):
        print(f"not gated, no device step mapped it: {p}")
    unread = [p for p in libs if not readable_elf(p)]
    if unread:
        return fail("unreadable, or not a 64-bit ELF: " + ", ".join(map(str, unread)))
    sg = _sass_gate()
    device = [p for p in libs if sg.has_device_code(str(p))]
    host = [p for p in libs if p not in device]
    for p in host:
        print(f"passed over: {p}: no .nv_fatbin section, a host-only module")
    if not device:
        return fail(f"the device steps mapped no device library under {a.home}")

    rc = subprocess.run([sys.executable, str(SASS_GATE), "--cuda-bin", str(a.cuda_bin), *map(str, device), "--json", str(a.json)],
                        env={**os.environ, "POUW_SASS_GATE_CACHE": "none"}).returncode
    try:
        records = {os.path.realpath(b["path"]): b for b in json.loads(a.json.read_text())["binaries"]}
    except (OSError, ValueError, KeyError):
        return fail(f"the harness's gate exits {rc} and leaves no record")
    missing = [p for p in device if str(p) not in records]
    bad = [records[str(p)] for p in device if str(p) in records and not records[str(p)].get("ok")]
    passed_over = f"{len(host)} host-only module(s) passed over" + (f" ({', '.join(label(p) for p in host)})" if host else "")
    if rc or missing or bad:
        return fail(f"the harness's gate (exit {rc}) refuses {len(bad) + len(missing)} of the {len(device)} device libraries "
                    f"the device steps mapped: " + "; ".join([*map(summary, bad), *(f"{label(p)} unrecorded" for p in missing)])
                    + f"; {passed_over}")
    print(f"{len(device)} device libraries the device steps mapped pass the harness's gate: "
          + "; ".join(summary(records[str(p)]) for p in device) + f"; {passed_over}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
