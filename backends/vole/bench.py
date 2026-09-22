"""Run Diet Mac'n'Cheese prover and verifier on an exported relation, two processes on localhost, through a
byte-counting relay; record wall times, dietmc's own timers (sVOLE extensions vs online), bytes each way, peak RSS.

    python backends/vole/bench.py DIR --dietmc /workspace/target-swanky/release/dietmc [--lpn medium] [--threads 1]
                                  [--label unit-1] [--plaintext-too] [--out result.json]

DIR holds relation.txt / instance.txt / witness.txt (from export_ir.py).  The relay (this file, ``--relay``) sits between
the prover and the verifier so that the bytes on the wire are counted exactly, per direction.  Writes
``$RESEARCH_RUN_DIR/result.json`` when run under ``research run``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

DUR = re.compile(r"(\d+(?:\.\d+)?)\s*(ns|µs|us|ms|s)\b")
SCALE = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def parse_duration(s: str) -> float:
    m = DUR.search(s)
    if not m:
        raise ValueError(s)
    return float(m.group(1)) * SCALE[m.group(2)]


# ---------------------------------------------------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------------------------------------------------

def relay(listen_port: int, target_port: int, report: Path) -> None:
    """Accept one prover connection, connect to the verifier, pump both ways, write byte counts at close."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", listen_port))
    srv.listen(1)
    prover, _ = srv.accept()
    verifier = socket.create_connection(("127.0.0.1", target_port))
    for s in (prover, verifier):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    counts = {"p2v": 0, "v2p": 0}

    def pump(src: socket.socket, dst: socket.socket, key: str) -> None:
        try:
            while True:
                buf = src.recv(1 << 20)
                if not buf:
                    break
                counts[key] += len(buf)
                dst.sendall(buf)
        except OSError:
            pass
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    t1 = threading.Thread(target=pump, args=(prover, verifier, "p2v"))
    t2 = threading.Thread(target=pump, args=(verifier, prover, "v2p"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    report.write_text(json.dumps(counts))


# ---------------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------------

def vm_hwm_kb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


class Proc:
    def __init__(self, cmd: list[str], log: Path, env: dict) -> None:
        self.cmd = cmd
        self.log = open(log, "w")  # noqa: SIM115  (closed in poll)
        self.t0 = time.time()
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.peak_kb = 0
        self.wall: float | None = None

    def poll(self) -> bool:
        if self.p.poll() is None:
            v = vm_hwm_kb(self.p.pid)
            if v:
                self.peak_kb = max(self.peak_kb, v)
            return False
        if self.wall is None:
            self.wall = time.time() - self.t0
            self.log.close()
        return True


def parse_log(path: Path) -> dict:
    txt = path.read_text()
    out: dict = {"init_s": None, "circ_exec_s": None, "svole_ext": [], "counts": {}, "done": False, "error": None}
    for line in txt.splitlines():
        if "init time:" in line:
            out["init_s"] = parse_duration(line.split("init time:")[1])
        elif "circ exec" in line and "time" in line:
            out["circ_exec_s"] = parse_duration(line.split("exec")[1])
        elif "total time:" in line:
            out["total_s"] = parse_duration(line.split("total time:")[1])
        elif "SVOLE<" in line:
            m = re.search(r"SVOLE<([^,]+),([^ ]+) (.*)>", line)
            if m:
                out["svole_ext"].append({"field": m.group(1), "tag": m.group(2), "s": parse_duration(m.group(3))})
        elif "DONE!" in line:
            out["done"] = True
        elif "Error" in line or "error" in line.lower() and "check" in line.lower():
            out["error"] = (out["error"] or "") + line.strip() + " | "
        else:
            m = re.search(r"nb (\w+):\s+(\d+)", line)
            if m and "F61p" in txt:
                out["counts"][m.group(1)] = out["counts"].get(m.group(1), 0) + int(m.group(2))
    out["svole_ext_total_s"] = sum(e["s"] for e in out["svole_ext"])
    out["svole_ext_main_field_s"] = sum(e["s"] for e in out["svole_ext"] if e["field"] not in ("F2", "F40b"))
    return out


def run_once(args, mode: str, workdir: Path) -> dict:
    """mode: 'zk' (prover+verifier through the relay) or 'plaintext' (one process, evaluates in the clear)."""
    d = Path(args.dir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RUST_LOG="info")
    base = [args.dietmc, "--text", "--relation", str(d / "relation.txt"), "--instance", str(d / "instance.txt")]
    cfg = workdir / "dmc.toml"
    cfg.write_text(f"lpn = '{args.lpn}'\nthreads = {args.threads}\nno_batching = false\n")
    base += ["--config", str(cfg)]
    if mode == "plaintext":
        p = Proc([*base, "--witness", str(d / "witness.txt"), "--plaintext"], workdir / "plaintext.log", env)
        while not p.poll():
            time.sleep(0.05)
        return {"wall_s": p.wall, "peak_rss_mb": p.peak_kb / 1024, "exit": p.p.returncode, "log": parse_log(workdir / "plaintext.log")}
    vport, rport = args.port, args.port + 1
    if args.threads > 1:
        # multithreaded dietmc opens one connection per thread on consecutive ports; no relay (bytes not counted)
        addr = f"127.0.0.1:{vport}"
        ver = Proc([*base, "--connection-addr", addr], workdir / "verifier.log", env)
        time.sleep(0.5)
        prv = Proc([*base, "--connection-addr", addr, "--witness", str(d / "witness.txt")], workdir / "prover.log", env)
        counts = None
    else:
        report = workdir / "relay.json"
        rel = subprocess.Popen([sys.executable, __file__, "--relay", str(rport), str(vport), str(report)])
        ver = Proc([*base, "--connection-addr", f"127.0.0.1:{vport}"], workdir / "verifier.log", env)
        time.sleep(0.5)
        prv = Proc([*base, "--connection-addr", f"127.0.0.1:{rport}", "--witness", str(d / "witness.txt")], workdir / "prover.log", env)
    t0 = time.time()
    while True:
        a = prv.poll()
        b = ver.poll()
        if a and b:
            break
        if time.time() - t0 > args.timeout:
            prv.p.kill()
            ver.p.kill()
            raise TimeoutError("dietmc did not finish")
        time.sleep(0.05)
    if args.threads == 1:
        rel.wait(timeout=30)
        counts = json.loads(report.read_text()) if report.exists() else None
    pl, vl = parse_log(workdir / "prover.log"), parse_log(workdir / "verifier.log")
    res = {
        "prover": {"wall_s": prv.wall, "peak_rss_mb": prv.peak_kb / 1024, "exit": prv.p.returncode, **pl},
        "verifier": {"wall_s": ver.wall, "peak_rss_mb": ver.peak_kb / 1024, "exit": ver.p.returncode, **vl},
        "bytes": counts, "accepted": pl["done"] and vl["done"] and prv.p.returncode == 0 and ver.p.returncode == 0,
    }
    # the verifier logs every sVOLE extension; the prover blocks on the same extension, so it is the shared cost
    res["svole_extension_s"] = vl["svole_ext_main_field_s"]
    res["svole_extension_all_fields_s"] = vl["svole_ext_total_s"]
    res["svole_extensions_n"] = len([e for e in vl["svole_ext"] if e["field"] not in ("F2", "F40b")])
    if vl["circ_exec_s"] is not None:
        res["online_s_estimate"] = vl["circ_exec_s"] - vl["svole_ext_main_field_s"]
    return res


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--relay":
        relay(int(argv[1]), int(argv[2]), Path(argv[3]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--dietmc", required=True)
    ap.add_argument("--lpn", default="medium", choices=("small", "medium", "large"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--port", type=int, default=7900)
    ap.add_argument("--label", default="")
    ap.add_argument("--plaintext-too", action="store_true")
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--swanky-commit", default="")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    workdir = Path(os.environ.get("RESEARCH_RUN_DIR", d)) / f"dmc_{args.label or 'run'}"
    workdir.mkdir(parents=True, exist_ok=True)
    sizes = {f: (d / f).stat().st_size for f in ("relation.txt", "instance.txt", "witness.txt") if (d / f).exists()}
    runs = []
    for r in range(args.repeat):
        wd = workdir / f"rep{r}"
        wd.mkdir(exist_ok=True)
        runs.append(run_once(args, "zk", wd))
    result = {
        "schema": "vole-bench/v1", "label": args.label, "dir": str(d), "meta": meta, "file_bytes": sizes,
        "lpn": args.lpn, "threads": args.threads, "swanky_commit": args.swanky_commit, "runs": runs,
        "hostname": socket.gethostname(),
    }
    if args.plaintext_too:
        result["plaintext"] = run_once(args, "plaintext", workdir / "plain")
    if Path("/proc/cpuinfo").exists():
        with open("/proc/cpuinfo") as f:
            models = [line.split(":", 1)[1].strip() for line in f if line.startswith("model name")]
        result["cpu_model"] = models[0] if models else None
    out = Path(args.out) if args.out else workdir / "bench.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        env = {"schema": "research/result/v1", "run_id": os.environ.get("RESEARCH_RUN_ID"), "label": args.label,
               "accepted": all(r["accepted"] for r in runs), "bench": str(out)}
        (Path(rd) / "result.json").write_text(json.dumps(env, indent=1) + "\n")
    summary = {k: result[k] for k in ("label", "lpn", "threads")}
    for r in runs:
        summary.update({"accepted": r["accepted"], "prover_wall_s": r["prover"]["wall_s"], "verifier_wall_s": r["verifier"]["wall_s"],
                        "init_s": r["prover"]["init_s"], "circ_exec_s": r["prover"]["circ_exec_s"],
                        "svole_ext_s": r["svole_extension_s"], "online_s": r.get("online_s_estimate"),
                        "bytes": r["bytes"], "prover_peak_mb": r["prover"]["peak_rss_mb"], "verifier_peak_mb": r["verifier"]["peak_rss_mb"]})
    print(json.dumps(summary))
    return 0 if all(r["accepted"] for r in runs) else 2


if __name__ == "__main__":
    sys.exit(main())
