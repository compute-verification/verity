"""flock_sessions.py, flock_sessions.sh's two steps in Python.

    stage SPEC OUT [N...]   `served_units/stage.py SPEC full OUT` as it is, and from the same built stager the full
                            statement's first N blocks for each N below full, as OUT-nN/{inst.bin,pub.bin}: the circuit and
                            its pin are OUT's, lowered once. An N that can't be staged is reported on stderr and left out.
    summary OUT             OUT/summary.json (`verity/flock-sessions/v1`) and its table, from every `flock-verify verify`
                            call flock_sessions.sh logged under OUT (TAG.err, the verifier's phase lines; TAG.out, its
                            verdicts; TAG.time, /usr/bin/time -v).

A call's seconds are the verifier's own lines: ``inputs`` is its read of the statement (the circuit and the public file,
once a call), and each session's line is that session's whole check, its ``setup`` line included. A session counts as
``checked`` where the verifier accepted it or refused it only for want of ``--program``, its last refusal, which a
census statement always ends in; one refused earlier stopped early and is left out. ``by_blocks`` is the median
seconds of the checked sessions at each block count (a verdict's ``unit_draw.k``), and ``fit`` their least-squares
line: the seconds a session pays whatever its blocks, and the seconds a block.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

SCHEMA = "verity/flock-sessions/v1"
#: how the verifier's last refusal starts: the one a statement that binds no Program ends in once every check has passed
NO_PROGRAM = "refused: a session without the verifier's program"


def stage(spec: str, out: Path, ns: list[int]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_units"))
    import stage as S

    whole = S.CS.UnitStager.stage

    def with_prefixes(self, rows, out_dir, salt_key, *a, **kw):
        got = whole(self, rows, out_dir, salt_key, *a, **kw)
        full, g = len(next(iter(rows.values()))), self.meta.get("vus_per_block") or 1
        print(f"flock_sessions: full is {full} blocks", file=sys.stderr, flush=True)
        for n in ns:
            if n >= full or n % g:
                print(f"flock_sessions: {n} blocks left out (full is {full}, {g} to a block)", file=sys.stderr, flush=True)
                continue
            t = time.monotonic()
            try:
                whole(self, {r: v[:n] for r, v in rows.items()}, out_dir.parent / f"{out_dir.name}-n{n}", salt_key, *a, **kw)
                print(f"flock_sessions: {n} blocks staged in {time.monotonic() - t:.1f} s", file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001 -- the full statement and the other prefixes still stand
                print(f"flock_sessions: {n} blocks not staged: {e!r}", file=sys.stderr, flush=True)
        return got

    S.CS.UnitStager.stage = with_prefixes
    sys.argv = ["stage.py", spec, "full", str(out), "--unbound"]
    S.main()


def timed(p: Path) -> dict:
    """A /usr/bin/time -v file's CPU and wall seconds, peak RSS and exit status."""
    t = p.read_text()

    def num(pat: str) -> float:
        m = re.search(pat, t)
        return float(m[1]) if m else 0.0

    wall = re.search(r"Elapsed \(wall clock\) time.*: ([\d:.]+)", t)
    secs = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall[1].split(":")))) if wall else None
    return {"cpu_s": round(num(r"User time \(seconds\): ([\d.]+)") + num(r"System time \(seconds\): ([\d.]+)"), 2),
            "wall_s": secs, "maxrss_gib": round(num(r"Maximum resident set size \(kbytes\): (\d+)") / 2**20, 2),
            "exit": int(num(r"Exit status: (\d+)"))}


def call(out: Path, tag: str) -> dict | None:
    """One `verify` call's row, or None where it logged no phase line."""
    ms = [x.rsplit(":", 1) for x in (out / f"{tag}.err").read_text().splitlines() if x.endswith(" ms")]
    ms = [(k, int(v.split()[0]) / 1e3) for k, v in ms]
    read = [s for k, s in ms if k == "inputs"]
    if not read:
        return None
    # a session's own line follows its setup line, where the verifier printed one
    whole, setup = [], None
    for k, s in ms:
        if k.startswith("setup"):
            setup = s
        elif k != "inputs":
            whole.append((s, setup))
            setup = None
    verdicts = [json.loads(x.split(" ", 1)[1]) for x in (out / f"{tag}.out").read_text().splitlines() if x.startswith("VERDICT ")]
    sessions = [{"blocks": (v.get("unit_draw") or {}).get("k"), "session_s": s, "setup_s": u, "accepted": v.get("accepted") is True,
                 "checked": v.get("accepted") is True or str(v.get("why") or "").startswith(NO_PROGRAM), "why": v.get("why")}
                for (s, u), v in zip(whole, verdicts, strict=True)]
    return {"tag": tag, "read_s": read[0], "sessions": sessions, **timed(out / f"{tag}.time")}


def summary(out: Path) -> dict:
    st = json.loads((out / "stage.json").read_text())
    calls = [c for c in (call(out, p.name[:-4]) for p in sorted(out.glob("*.err")) if (out / f"{p.name[:-4]}.time").exists()
                         and p.name not in ("read.err", "perf-verify.err")) if c]
    at: dict[int, list[float]] = {}
    for c in calls:
        for s in c["sessions"]:
            if s["blocks"] and s["checked"]:
                at.setdefault(s["blocks"], []).append(s["session_s"])
    by = {n: round(statistics.median(v), 3) for n, v in sorted(at.items())}
    fit = None
    if len(by) > 1:
        slope, fixed = statistics.linear_regression(list(by), list(by.values()))
        fit = {"session_fixed_s": round(fixed, 3), "per_block_s": round(slope, 6), "points": len(by)}
    phases = out / "phases.jsonl"
    return {"schema": SCHEMA, "block": st.get("census") or st["definition"], "full_blocks": st["n"], "ands": st["ands"],
            "unit_nnz": st["unit_nnz"], "k_log": st["k_log"], "circuit_bytes": st["circuit_bytes"], "circuit_sha512": st["circuit_sha512"],
            "read_s": round(statistics.median(c["read_s"] for c in calls), 3) if calls else None, "by_blocks": by, "fit": fit,
            "calls": calls, "phases": [json.loads(x) for x in phases.read_text().splitlines()] if phases.exists() else []}


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "stage":
        stage(sys.argv[2], Path(sys.argv[3]), [int(x) for x in sys.argv[4:]])
    elif len(sys.argv) == 3 and sys.argv[1] == "summary":
        out = Path(sys.argv[2])
        s = summary(out)
        (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        print(f"{s['block']}: {s['ands']} ANDs, {s['unit_nnz']} matrix entries, {s['circuit_bytes']} bytes of circuit, "
              f"{s['full_blocks']} blocks a full statement")
        for c in s["calls"]:
            ses = ", ".join(f"{x['blocks']} blocks {x['session_s']} s (setup {x['setup_s']} s, "
                            f"{'accepted' if x['accepted'] else 'checked, no Program' if x['checked'] else 'refused early'})"
                            for x in c["sessions"])
            print(f"  {c['tag']}: read {c['read_s']} s; {ses}; cpu {c['cpu_s']} s, wall {c['wall_s']} s, peak {c['maxrss_gib']} GiB")
        print(f"  read once a call: {s['read_s']} s; a session by its blocks: {s['by_blocks']}; fit: {s['fit']}")
    else:
        raise SystemExit("flock_sessions.py: usage: flock_sessions.py stage SPEC OUT [N...] | summary OUT")


if __name__ == "__main__":
    main()
