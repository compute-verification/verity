"""The gate's `certifier-rehearse --traffic varied` end to end on one host, honest then token: the gate's request
(`gate_certifier`), its run (`certifier_run`: the real `certifier_serve.py rehearse` through a TCP forwarder standing in
for the tunnel) against a stub endpoint (the tests' `_Varied`, keeping serve-openai's requests.jsonl rows), its
`fetch` tarball, then `negatives.py tie-run --run TARBALL --ids requests.jsonl`. Everything lands in $OUT."""
import asyncio
import json
import os
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

TREE = Path.cwd()
sys.path[:0] = [str(TREE / "benchmarks/network_traces/tests"), str(TREE / "benchmarks/live_challenger")]
import auditor as A  # noqa: E402
import test_certifier_serve as T  # noqa: E402

OUT = Path(os.environ["OUT"])
FORWARD = r"""
import socket, sys, threading
lport, rport = int(sys.argv[1]), int(sys.argv[2])
ls = socket.socket()
ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ls.bind(("127.0.0.1", lport))
ls.listen(256)
def pump(a, b):
    try:
        while data := a.recv(65536):
            b.sendall(data)
    except OSError:
        pass
    finally:
        try:
            b.shutdown(socket.SHUT_WR)
        except OSError:
            pass
while True:
    c, _ = ls.accept()
    u = socket.create_connection(("127.0.0.1", rport))
    threading.Thread(target=pump, args=(c, u), daemon=True).start()
    threading.Thread(target=pump, args=(u, c), daemon=True).start()
"""


def main() -> int:
    root = OUT / "host"
    root.mkdir(parents=True)
    (OUT / "peers").write_text("vy-nebius-1 research@10.0.0.2\n")
    host = A.Host(root=root, peers=OUT / "peers", pins=OUT / "pins.jsonl")

    endpoint, loop, ready, port = T._Varied(), asyncio.new_event_loop(), threading.Event(), []

    def serve():
        asyncio.set_event_loop(loop)
        server = loop.run_until_complete(asyncio.start_server(endpoint, "127.0.0.1", 0))
        port.append(server.sockets[0].getsockname()[1])
        ready.set()
        loop.run_forever()

    threading.Thread(target=serve, daemon=True).start()
    ready.wait()
    summary = {"stub_port": port[0], "runs": {}}
    for strategy, tamper in (("honest", None), ("token", "token")):
        argv = ["--peer", "vy-nebius-1", "--port", "18084", "--minutes", "20", "--traffic", "varied",
                "--requests", "16", "--tokens", "16", "--seed", "3"] + (["--tamper", tamper] if tamper else [])
        req = A.gate_certifier(argv, host, by="network-accounting", now=datetime.now(timezone.utc),
                               token=secrets.token_hex(2), start=lambda r: None, own=lambda p: None,
                               active=lambda r: False)
        run = req["run"]
        print(f"{strategy}: gate request {json.dumps(req, sort_keys=True)}", flush=True)
        print(f"{strategy}: command {A.certifier_command(req, Path('OUT'), 'URL')}", flush=True)
        end = A.certifier_run(host, run, pinned="scratch", now=lambda: datetime.now(timezone.utc),
                              tunnel_argv=lambda ssh, local: [sys.executable, "-c", FORWARD, str(local), str(port[0])])
        print(f"{strategy}: end rc={end['rc']} ended={end['ended']!r} runs={end['runs']}", flush=True)
        tgz = OUT / f"{run}.tgz"
        tgz.write_bytes(A.certifier_fetch(run, host))
        summary["runs"][strategy] = {"run": run, "request": req, "end": {k: end[k] for k in ("rc", "ended", "runs")},
                                     "tarball": tgz.name}
    ids = OUT / "requests.jsonl"
    ids.write_text("".join(json.dumps(r) + "\n" for r in endpoint.served))
    users = [r["user"] for r in endpoint.served]
    summary["endpoint"] = {"rows": len(users), "without_user": users.count(None),
                           "users": sorted({u.rsplit(".", 1)[0] for u in users if u})}
    for strategy, row in summary["runs"].items():
        p = subprocess.run([sys.executable, "benchmarks/network_traces/negatives.py", "tie-run", "--run",
                            str(OUT / row["tarball"]), "--ids", str(ids)], capture_output=True, text=True)
        (OUT / f"tie-run-{strategy}.json").write_text(p.stdout)
        doc = json.loads(p.stdout) if p.stdout.strip() else {}
        row["tie_run"] = {"exit": p.returncode, "stderr": p.stderr[-2000:],
                          "runs": {s: {k: r.get(k) for k in ("verdict", "ok", "codes", "failing", "held", "id_events",
                                                              "outputs", "placed", "unplaced", "missing")}
                                   for s, r in (doc.get("runs") or {}).items()}}
        print(f"{strategy}: tie-run exit {p.returncode} {row['tie_run']['runs']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    ok = summary["runs"]["honest"]["tie_run"]["runs"].get("honest", {}).get("verdict") == "accepted" and \
        summary["runs"]["token"]["tie_run"]["runs"].get("token", {}).get("verdict") == "refused"
    print("tie as expected" if ok else "tie NOT as expected", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
