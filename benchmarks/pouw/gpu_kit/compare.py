"""The twin compare: a device's dump against its twin's (the Python reference's), byte for byte, saying what kind of
difference it is.

The twin writes its output with `dump.write`, the bench with `gk::Dump`, under the same names. A difference names the
first differing byte and word, and says when it is one of the known faults:

- **not written:** every differing word of the device's output still holds the 0xA5 poison (the kernel skipped them, or a
  memset on another stream raced it: 44878194f);
- **byte order:** the device's bytes are the twin's with every 8-byte (or 4-byte) word reversed (EGEMM's digests,
  63f78c35a);
- **layout:** the two sides dumped one buffer under different layouts.

  python compare.py DEVICE_DIR TWIN_DIR [--names NAME...] [--json OUT]   # exit 0: equal; 9: a difference (each named)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIT_DIFF = 9
POISON_WORD = b"\xa5" * 4


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dump = _load("gpu_kit_dump", HERE / "dump.py")


def _swapped(data: bytes, width: int) -> bytes:
    return b"".join(data[i:i + width][::-1] for i in range(0, len(data), width))


def compare(got: bytes, want: bytes, word: int = 4) -> dict:
    """``{"equal", "bytes", "first", "words", "poisoned", "hint"}``: `first` the first differing byte offset, `words` the
    differing `word`-byte words, `poisoned` how many of those the device left at 0xA5A5A5A5, `hint` the fault's name."""
    rec = {"equal": got == want, "bytes": [len(got), len(want)], "first": None, "words": 0, "poisoned": 0, "hint": None}
    if rec["equal"]:
        return rec
    n = min(len(got), len(want))
    rec["first"] = next((i for i in range(n) if got[i] != want[i]), n)
    diff = [i for i in range(0, n, word) if got[i:i + word] != want[i:i + word]]
    rec["words"] = len(diff)
    rec["poisoned"] = sum(got[i:i + word] == POISON_WORD[:word] for i in diff)
    if len(got) != len(want):
        rec["hint"] = f"size: the device dumped {len(got)} bytes, the twin {len(want)}"
    elif diff and rec["poisoned"] == len(diff):
        rec["hint"] = (f"not written: all {len(diff)} differing words still hold the 0xA5 poison; the kernel skipped them, "
                       "or the poison raced it (poison on the kernel's own stream: gk::poison)")
    elif len(got) % 8 == 0 and _swapped(got, 8) == want:
        rec["hint"] = ("byte order: the device's bytes are the twin's with every 8-byte word reversed; dump digests with "
                       "gk::Dump::digests (SHA-512's big-endian bytes), not their native uint64 words")
    elif len(got) % 4 == 0 and _swapped(got, 4) == want:
        rec["hint"] = "byte order: the device's bytes are the twin's with every 4-byte word reversed"
    return rec


def compare_dumps(device_dir: Path, twin_dir: Path, names: list[str] | None = None) -> dict:
    dev, twin = dump.read(device_dir), dump.read(twin_dir)
    names = names or sorted(set(dev) | set(twin))
    out = {"version": dump.VERSION, "device": str(device_dir), "twin": str(twin_dir), "buffers": {}, "missing": []}
    for name in names:
        if name not in dev or name not in twin:
            out["missing"].append(f"{name}: not in the {'device' if name not in dev else 'twin'} dump")
            continue
        rec = compare(dev[name].data, twin[name].data)
        if dev[name].layout != twin[name].layout:
            rec.update(equal=False, hint=f"layout: the device dumped {dev[name].layout}, the twin {twin[name].layout}")
        out["buffers"][name] = rec
    out["equal"] = not out["missing"] and all(r["equal"] for r in out["buffers"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("device", type=Path)
    p.add_argument("twin", type=Path)
    p.add_argument("--names", nargs="+")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    try:
        v = compare_dumps(a.device, a.twin, a.names)
    except dump.DumpError as e:
        print(f"compare: {e}", file=sys.stderr)
        return EXIT_DIFF
    if a.json:
        a.json.write_text(json.dumps(v, indent=1) + "\n")
    for m in v["missing"]:
        print(f"DIFF {m}")
    for name, r in v["buffers"].items():
        if r["equal"]:
            print(f"EQUAL {name} {r['bytes'][0]} bytes")
        else:
            print(f"DIFF {name}: {r['words']} words differ, the first at byte {r['first']}"
                  + (f"; {r['hint']}" if r["hint"] else ""))
    return 0 if v["equal"] else EXIT_DIFF


if __name__ == "__main__":
    sys.exit(main())
