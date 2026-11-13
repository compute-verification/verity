"""Scratch driver, not for landing: the served harness's audit step (`served_verdicts.sh`'s ``v3c_audit``, its
``records_audit.py audit`` argv) on stand-in windows with no served values, with ``--deployed-key`` and
FLOCK_CHALLENGER=1, then the v2 executable's ``refusal`` on the per-member case it writes, under a peak-memory probe.

    standin_refusal.py OUT BIN LEAN_BIN_DIR KS[,KS...]...     e.g.  128,256 2048,4096 8192,16384

Each window is `test_audit_two_members._window` at its K's (`ScaledMmS8Coordinate`, one proof block a member, under
``stratified:1``, so every member is sampled). Per window it writes OUT/<tag>/result.json: the audit's exit, the
per-member case's presence, the refusal line as the executable printed it, its exit, its peak RSS (the probe's
`RUSAGE_CHILDREN` maxrss and /usr/bin/time -v's where present) and the bytes of every file the executable reads; and
OUT/<tag>/rss.json, its memory as it runs once more (`sampled`), with the levels the rows show under `refusal_levels`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage" / "tests"))
import test_audit_two_members as T  # noqa: E402

from verity.circuits.defs import bind  # noqa: E402
from verity.definitions.ops import gemm as G  # noqa: E402
from verity.protocols.spot.verifier.one_stage import members as M, proofs as PF  # noqa: E402

PROBE = ("import json, resource, subprocess, sys; p = subprocess.run(sys.argv[1:], capture_output=True, text=True); "
         "print(json.dumps({'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr[-4000:], "
         "'maxrss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}))")


def probed(argv: list[str], env: dict | None = None) -> dict:
    """``argv`` in a probe process of its own: its exit, output and the peak RSS of its largest descendant."""
    t = time.monotonic()
    p = subprocess.run([sys.executable, "-c", PROBE, *argv], capture_output=True, text=True, env=env)
    got = json.loads(p.stdout)
    got["seconds"] = round(time.monotonic() - t, 1)
    return got


def sampled(argv: list[str], every: float = 0.05) -> dict:
    """``argv`` once more, its memory read from /proc as it runs: rows of (seconds, VmRSS kB, VmHWM kB, bytes read so
    far), so the level it holds once every file is read shows apart from what a member's check adds."""
    t0 = time.monotonic()
    p = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows: list[tuple] = []

    def field(text: str, name: str) -> int | None:
        return next((int(x.split()[1]) for x in text.splitlines() if x.startswith(name)), None)

    while p.poll() is None:
        try:
            status = Path(f"/proc/{p.pid}/status").read_text()
            io = Path(f"/proc/{p.pid}/io").read_text()
        except OSError:
            break
        rows.append((round(time.monotonic() - t0, 2), field(status, "VmRSS:"), field(status, "VmHWM:"),
                     field(io, "rchar:")))
        time.sleep(every)
    p.wait()
    return {"rc": p.returncode, "rows": [r for r in rows if r[1] is not None]}


def levels(rows: list[tuple], total: int) -> dict:
    """What the rows show: the memory held when the last file is read, the lowest and highest held from then until
    memory first passes 1.1 times that (the files alone, while each is hashed), and the peak."""
    read = next((i for i, r in enumerate(rows) if r[3] is not None and r[3] >= total), None)
    if read is None:
        return {"read_done": None, "peak_rss_kb": max((r[1] for r in rows), default=None)}
    base = rows[read][1]
    rise = next((i for i in range(read, len(rows)) if rows[i][1] > 1.1 * base), len(rows))
    flat = [r[1] for r in rows[read:rise]] or [base]
    return {"read_done_s": rows[read][0], "rss_kb_at_read_done": base, "files_alone_until_s": rows[rise - 1][0],
            "files_alone_rss_kb_min": min(flat), "files_alone_rss_kb_max": max(flat),
            "peak_rss_kb": max(r[1] for r in rows), "peak_at_s": max(rows, key=lambda r: r[1])[0],
            "end_s": rows[-1][0], "by_second_max_rss_mb": [
                max(r[1] for r in rows if int(r[0]) == s) // 1024 for s in sorted({int(r[0]) for r in rows})]}


def audit_argv(window: Path, out: Path, main: Path, bin_: Path, leanb: Path, ports: int) -> list[str]:
    """`v3c_audit`'s argv at this commit, less ``--registered`` (the stand-in reads no registered value), plus
    ``--deployed-key``, with a port a member (``ports``): every member is sampled."""
    m0 = os.pathsep.join(str(ROOT / p) for p in ("", "benchmarks/numerical/python", "verity/protocols/flock/prover/python",
                                                  "adapters/vllm"))
    return [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "benchmarks" / "prove_path" / "records_audit.py"), "audit",
            "--window", str(window), "--law", T.DESIGN, "--circuit",
            hashlib.sha512((window / "circuit.json").read_bytes()).hexdigest(), "--run", T.RUN, "--challenger", "local",
            "--main", str(main), "--m0-pythonpath", m0, "--bin", str(bin_), "--lean", str(leanb / "flock-verify"),
            "--firewall", str(leanb / "flock-firewall"), "--ports", str(ports), "--out", str(out), "--deployed-key"]


def one(out_root: Path, ks: tuple[int, ...], bin_: Path, leanb: Path) -> dict:
    tag = "k" + "-".join(map(str, ks))
    d = out_root / tag
    d.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"standin-{tag}-"))
    T.KS = ks
    T.TEMPLATES = tuple(bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8) for k in ks)
    window = T._window(work / "window")
    main = work / "main.txt"
    main.write_text(subprocess.run(["git", "-C", str(ROOT), "rev-list", "--first-parent", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout)
    argv = audit_argv(window, work / "audit", main, bin_, leanb, len(ks))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FC_", "FLOCK_")) and k != "PYTHONPATH"}
    env.update(RAYON_NUM_THREADS=str(os.cpu_count()), RUST_MIN_STACK=str(64 << 20), FLOCK_CHALLENGER="1")
    res: dict = {"ks": list(ks), "audit_command": "FLOCK_CHALLENGER=1 " + " ".join(argv)}
    a = probed(argv, env)
    (d / "audit.log").write_text(a["stdout"])
    (d / "audit.err").write_text(a["stderr"])
    res["audit"] = {"rc": a["rc"], "seconds": a["seconds"], "peak_rss_kb": a["maxrss_kb"],
                    "records_line": next((x for x in a["stdout"].splitlines() if x.startswith("RECORDS")), None)}
    summary = work / "audit" / "summary.json"
    if summary.is_file():
        row = json.loads(summary.read_text())
        shutil.copy(summary, d / "summary.json")
        res["audit"].update(ok=row.get("ok"), draw=row.get("draw"), accepted=(row.get("audit") or {}).get("accepted"),
                            verdicts=(row.get("audit") or {}).get("verdicts"))
    cfile = work / "audit" / "window" / "case-v2-members.json"
    efile = work / "audit" / "window" / "case-v2-members-error.txt"
    res["case_v2_members"] = cfile.is_file()
    if efile.is_file():
        res["case_v2_members_error"] = efile.read_text()[:2000]
    if cfile.is_file():
        case = json.loads(cfile.read_text())
        exe = PF.v2_executable()
        files = work / "refusal"
        files.mkdir()
        paths = [files / n for n in ("params.json", "commitment.bin", "evidence.bin")]
        canonical = json.dumps(case["params"], sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                               allow_nan=False).encode()
        for p, b in zip(paths, (canonical, bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]))):
            p.write_bytes(b)
        blobs = [Path(b) for b in case["blobs"]]
        argv_r = [str(exe), "refusal", *map(str, paths), *map(str, blobs)]
        r = probed(argv_r)
        tv = None
        if Path("/usr/bin/time").is_file():
            q = subprocess.run(["/usr/bin/time", "-v", *argv_r], capture_output=True, text=True)
            tv = next((int(x.split(":")[1]) for x in q.stderr.splitlines() if "Maximum resident set size" in x), None)
        sizes = {"circuit": sum(b.stat().st_size for b in blobs if b.name == "circuit.txt"),
                 "public": sum(b.stat().st_size for b in blobs if b.name.startswith("pub-")),
                 "session": sum(b.stat().st_size for b in blobs if b.name != "circuit.txt" and not b.name.startswith("pub-")),
                 "case_files": sum(p.stat().st_size for p in paths)}
        sizes["total"] = sum(sizes.values())
        sizes["circuits"] = sorted(b.stat().st_size for b in blobs if b.name == "circuit.txt")
        sizes["blobs_in_order"] = [[b.name, b.stat().st_size] for b in blobs]
        series = sampled(argv_r)
        (d / "rss.json").write_text(json.dumps({"columns": ["seconds", "VmRSS_kB", "VmHWM_kB", "rchar"],
                                                "rc": series["rc"], "rows": series["rows"]}))
        res["refusal_levels"] = levels(series["rows"], sizes["total"])
        line = (r["stdout"].strip().splitlines() or [""])[-1]
        res["refusal"] = {"command": " ".join([exe.name, "refusal", "PARAMS", "COMMITMENT", "EVIDENCE",
                                               f"[{len(blobs)} blobs]"]),
                          "line": line, "rc": r["rc"], "seconds": r["seconds"], "peak_rss_kb_probe": r["maxrss_kb"],
                          "peak_rss_kb_time_v": tv, "bytes_read": sizes, "blobs": len(blobs),
                          "all_passed": line and json.loads(line).get("refusal") == M.ALL_PASSED,
                          "stderr": r["stderr"][-1000:]}
    (d / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"STANDIN {tag} " + json.dumps({k: v for k, v in res.items() if k != "audit_command"}, sort_keys=True), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return res


def main() -> int:
    out, bin_, leanb, *sets = sys.argv[1:]
    for s in (x for part in sets for x in part.split("/")):
        try:
            one(Path(out), tuple(int(k) for k in s.split(",")), Path(bin_), Path(leanb))
        except Exception as x:  # noqa: BLE001 -- a larger window's failure leaves the smaller windows' results
            print(f"STANDIN k{s} FAILED {type(x).__name__}: {x}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
