"""spill_neg.py CIRCUIT LEAN PUB REGISTERED PARTITION PROGRAM SCRATCH OUT: the staged unit circuit with one row's compression
count one off the parser's, each refused by the Lean verifier.

For each port named in `PORTS` (an output row, the receipt ``o0``, which spills at K = 384, and an input row, ``a0``), META
says one compression more and one fewer than its length gives (``serial``, `HmRow.rowPorts`); `flock-verify statement` on the
changed circuit, with the honest statement's public file and the verifier's own partition, Program and registered reads, must
refuse it naming the port, and on the unchanged circuit (the control) set the statement up. One changed copy at a time is
written to SCRATCH; OUT/spill.json holds each case's exit, verdict and message, and the exit is 0 when every case is as
expected.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PORTS = ("o0", "a0")
REFUSAL = "compressions, not META's"


def split(path: Path) -> tuple[list[bytes], dict, int]:
    """The lines before META, META's JSON and the byte offset of the line after it."""
    head, at = [], 0
    with path.open("rb") as f:
        for _ in range(64):
            line = f.readline()
            if line.startswith(b"META "):
                return head, json.loads(line[5:]), at + len(line)
            head.append(line)
            at += len(line)
    raise SystemExit(f"{path}: no META line in its first 64 lines")


def write(path: Path, src: Path, head: list[bytes], meta: dict, rest_at: int) -> None:
    with path.open("wb") as out, src.open("rb") as f:
        out.writelines(head)
        out.write(b"META " + json.dumps(meta).encode() + b"\n")
        f.seek(rest_at)
        shutil.copyfileobj(f, out, 1 << 24)


def main(argv: list[str]) -> int:
    circuit, lean, pub, reg, partition, program, scratch, out = map(Path, argv)
    head, meta, rest_at = split(circuit)
    names = [p[0] for p in meta["ports"]]
    schemas = meta.get("row_schemas") or [None] * len(names)

    def statement(c: Path) -> subprocess.CompletedProcess:
        return subprocess.run([str(lean), "statement", "--statement", "verity/flock-circuit", "--circuit", str(c), "--public",
                               str(pub), "--partition", str(partition), "--program", str(program), "--registered", str(reg)],
                              capture_output=True, text=True)

    p = statement(circuit)
    cases = [{"case": "unchanged", "expect": "ok", "exit": p.returncode, "pass": p.returncode == 0,
              "why": (p.stdout + p.stderr).strip()[-400:]}]
    scratch.mkdir(parents=True, exist_ok=True)
    for name in PORTS:
        if name not in names:
            raise SystemExit(f"no port {name}: {names}")
        i = names.index(name)
        _, words, comps = meta["ports"][i]
        bits = schemas[i][1] if schemas[i] else 16 * words
        serial = ((bits + 7) // 8 + 17 + 127) // 128 - 1
        for d in (1, -1):
            if comps + d < 0:
                continue
            m = json.loads(json.dumps(meta))
            m["ports"][i][2] = comps + d
            c = scratch / f"{name}{d:+d}.txt"
            write(c, circuit, head, m, rest_at)
            p = statement(c)
            c.unlink()
            why = (p.stdout + p.stderr).strip()
            ok = p.returncode != 0 and f"port {name}:" in why and REFUSAL in why
            cases.append({"case": f"{name} comps {comps}{d:+d}", "port": name, "bits": bits, "bytes": (bits + 7) // 8,
                          "spills": (bits + 7) // 8 % 128 >= 112, "serial": serial, "meta_comps": comps + d,
                          "expect": "refused", "exit": p.returncode, "pass": ok, "why": why[-400:]})
    for c in cases:
        print(json.dumps(c), flush=True)
    passed = all(c["pass"] for c in cases) and cases[0]["exit"] == 0
    (out / "spill.json").write_text(json.dumps({"pass": passed, "cases": cases}, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
