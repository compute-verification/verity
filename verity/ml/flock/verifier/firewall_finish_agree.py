"""The firewall's final round in Lean, live against replayed: what a firewalled `flock-circuit prove --zk` session's Lean
finish (`Flock.Firewall.Finish`, the session loop's, `Finish.serveLoop`) wrote while it held the device's opening and the
inner proof's rounds itself (`open`, `inner`), held byte for byte to `flock-firewall finish` on the requests the session
dumped (`FC_FINISH_DUMP`).

    python firewall_finish_agree.py --lean flock-firewall --dump DIR --out DIR -- FINISH_ARG...

`DIR` holds, per firewalled table `T`, `table{T}.req` (its level-0 request, `flock-firewall finish`'s `table`),
`table{T}.root` (the root of record, the Lean loop's) and `table{T}.finish_root` (the session's Lean finish's); and per rep
`R`, `table{T}-rep{R}.req` (`Finish.parseRep`'s request, with the coins of every round the session answered) and
`table{T}-rep{R}.lean` (the session's Lean finish's proof file, on salts for levels ≥ 1 it drew itself and the prover's
trees took). Each table's two roots must be one. Then one `flock-firewall finish FINISH_ARG...` process (the session's
statement: `--circuit`, `--public`, `--draw` its `session.json`, ...) gets every table's request, answered with the root of
record. Each rep, in a process of its own, after its table and its salts (`salts T R`): its request must give the session's
proof file byte for byte, and the same request with its passed values moved (`table{T}-rep{R}.vals` says where they sit)
must stop `finish` with no file: a constraint fails on the rep's pads (`stopped`). Writes DIR/result.json; exits 1 on any
disagreement.
"""
from __future__ import annotations

import argparse
import json
import random
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


STOPPED = "stopped on perturbed values"


def rep(lean: str, args: list[str], dump: Path, t: int, r: int, moved: bool) -> tuple[bytes | None, str]:
    """Rep `r` of table `t` in a process of its own (a refusal ends one), after its table and its salts: its request as the
    session dumped it, or (`moved`) with every passed value moved by a random word."""
    req = bytearray((dump / f"table{t}-rep{r}.req").read_bytes())
    if moved:
        off, n = map(int, (dump / f"table{t}-rep{r}.vals").read_text().split())
        rng = random.Random(f"firewall-finish-agree/{t}/{r}")
        for k in range(16 * n):
            req[off + k] ^= rng.randrange(256)
    p = subprocess.Popen([lean, "finish", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    got, head = ask(p, f"table {t}", (dump / f"table{t}.req").read_bytes())
    if got is not None:
        got, head = ask(p, f"salts {t} {r}", b"")
    if got is not None:
        got, head = ask(p, f"rep {t} {r}", bytes(req))
    p.stdin.close()
    p.wait()
    return got, head


def stopped(lean: str, args: list[str], dump: Path, t: int, r: int) -> dict:
    """The rep's passed values moved: a constraint then fails on the rep's pads, and `finish` must stop with no file (Rust's
    `constraints_failing_on_pads`)."""
    got, head = rep(lean, args, dump, t, r, moved=True)
    ok = got is None and head.startswith("err PROVER-STOPPED: the constraints ") and head.endswith(" fail on the rep's pads")
    return {"what": f"table {t} rep {r} {STOPPED}", "ok": ok, "lean": head}


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
    rows, ok = [], True
    for t in tables:
        root = (a.dump / f"table{t}.root").read_bytes()
        row = {"what": f"table {t} root in the session", "ok": (a.dump / f"table{t}.finish_root").read_bytes() == root}
        rows.append(row)
        ok &= row["ok"]
        for f in sorted(a.dump.glob(f"table{t}-rep*.req")):
            r = int(re.fullmatch(rf"table{t}-rep(\d+)\.req", f.name).group(1))
            lp = a.dump / f"table{t}-rep{r}.lean"
            want = lp.read_bytes() if lp.exists() else None
            got, head = rep(a.lean, args, a.dump, t, r, moved=False)
            row = {"what": f"table {t} rep {r} proof file", "ok": want is not None and got == want,
                   "session": "no file" if want is None else f"{len(want)} bytes",
                   "replayed": head if got is None else f"{len(got)} bytes"}
            if got is not None and want is not None and got != want:
                row["first_diff"] = first_diff(got, want)
                row["bytes_differing"] = sum(x != y for x, y in zip(got, want)) + abs(len(got) - len(want))
            rows.append(row)
            ok &= row["ok"]
    p = subprocess.Popen([a.lean, "finish", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
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
    p.stdin.close()
    p.wait()
    for t in tables:
        for f in sorted(a.dump.glob(f"table{t}-rep*.vals")):
            r = int(re.fullmatch(rf"table{t}-rep(\d+)\.vals", f.name).group(1))
            row = stopped(a.lean, args, a.dump, t, r)
            rows.append(row)
            ok &= row["ok"]
    reps = [r for r in rows if r["what"].endswith("proof file")]
    stops = [r for r in rows if r["what"].endswith(STOPPED)]
    ok &= bool(reps) and bool(stops)
    summary = {"ok": ok, "tables": len(tables), "reps": len(reps), "stops": len(stops), "agree": sum(r["ok"] for r in rows),
               "rows": rows}
    (a.out / "result.json").write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(("AGREE " if r["ok"] else "DIFFER ") + json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
