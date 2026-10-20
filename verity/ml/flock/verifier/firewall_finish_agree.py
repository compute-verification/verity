"""The firewall's final round in Lean against Rust's: `flock-firewall finish` (`Flock/Firewall/Finish.lean`) on the requests a
firewalled `flock-circuit prove --zk` session dumped (`FC_FINISH_DUMP`), each answer held byte for byte to what Rust computed.

    python firewall_finish_agree.py --lean flock-firewall --dump DIR --out DIR -- FINISH_ARG...

`DIR` holds, per firewalled table `T`, `table{T}.req` (its level-0 request, `flock-firewall link`'s) and `table{T}.root` (the
root the Lean link answered), and per rep `R`, `table{T}-rep{R}.req` (`Finish.parseRep`'s request) and
`table{T}-rep{R}.proof` (`zk_finish`'s proof file). One `flock-firewall finish FINISH_ARG...` process (the session's
statement: `--circuit`, `--public`, `--draw` its `session.json`, ...) gets every table's request, then its reps'; each answer
must be the root or the proof file Rust has. Writes DIR/result.json; exits 1 on any disagreement.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def ask(p: subprocess.Popen, line: str, body: bytes) -> tuple[bytes | None, str]:
    """One request: its line and body; the answer's bytes, or None and the `err` line."""
    p.stdin.write(f"{line} {len(body)}\n".encode() + body)
    p.stdin.flush()
    head = p.stdout.readline().decode().rstrip("\n")
    m = re.fullmatch(r"ok (\d+)", head)
    if not m:
        return None, head or f"no answer (exit {p.poll()})"
    return p.stdout.read(int(m.group(1))), head


def first_diff(a: bytes, b: bytes) -> int | None:
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None if len(a) == len(b) else min(len(a), len(b)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True, help="flock-firewall")
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("finish", nargs=argparse.REMAINDER, help="-- then flock-firewall finish's arguments")
    a = ap.parse_args(argv)
    args = a.finish[1:] if a.finish[:1] == ["--"] else a.finish
    a.out.mkdir(parents=True, exist_ok=True)
    tables = sorted(int(m.group(1)) for f in a.dump.glob("table*.req") if (m := re.fullmatch(r"table(\d+)\.req", f.name)))
    if not tables:
        raise SystemExit(f"{a.dump}: no table*.req")
    p = subprocess.Popen([a.lean, "finish", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    rows, ok = [], True
    for t in tables:
        t0 = time.monotonic()
        got, head = ask(p, f"table {t}", (a.dump / f"table{t}.req").read_bytes())
        want = (a.dump / f"table{t}.root").read_bytes()
        row = {"what": f"table {t} root", "ok": got == want, "lean": head, "seconds": round(time.monotonic() - t0, 3)}
        if got is not None and got != want:
            row["first_diff"] = first_diff(got, want)
        rows.append(row)
        ok &= row["ok"]
        if got is None:
            break
        for r in sorted(int(m.group(1)) for f in a.dump.glob(f"table{t}-rep*.req") if (m := re.fullmatch(rf"table{t}-rep(\d+)\.req", f.name))):
            t0 = time.monotonic()
            got, head = ask(p, f"rep {t} {r}", (a.dump / f"table{t}-rep{r}.req").read_bytes())
            want = (a.dump / f"table{t}-rep{r}.proof").read_bytes()
            row = {"what": f"table {t} rep {r} proof file", "ok": got == want, "lean": head, "bytes": len(want),
                   "seconds": round(time.monotonic() - t0, 3)}
            if got is not None and got != want:
                row["first_diff"] = first_diff(got, want)
                (a.out / f"table{t}-rep{r}.lean").write_bytes(got)
            rows.append(row)
            ok &= row["ok"]
            if got is None:
                break
    p.stdin.close()
    p.wait()
    reps = [r for r in rows if "rep" in r["what"]]
    ok &= bool(reps)
    summary = {"ok": ok, "tables": len(tables), "reps": len(reps), "agree": sum(r["ok"] for r in rows), "rows": rows}
    (a.out / "result.json").write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(("AGREE " if r["ok"] else "DIFFER ") + json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
