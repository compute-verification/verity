"""The round trip through `run.sh`'s reverse tunnel, by ssh option: an echo server here at 127.0.0.1:PORT, `ssh -R` to
PEER (a name in /etc/vy/peers) running a client there that pings through the forward, back to back and paced like
the prover's roots. Prints one JSON line a variant.

    python tunnel_rtt.py --peer vy-nebius-2 --port 7795 [--pings 200 --pace-ms 12.8]
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import socket
import subprocess
import threading

CLIENT = r"""
import json, socket, sys, time
port, n, pace = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) / 1e3
for _ in range(100):
    try:
        c = socket.create_connection(("127.0.0.1", port), timeout=10)
        break
    except OSError:
        time.sleep(0.1)
c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
c.sendall(b"hello")
if c.recv(5) != b"hello":
    sys.exit("PORT is not this probe's echo")
def run(gap):
    out = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        c.sendall(b"x" * 64)
        got = 0
        while got < 64:
            got += len(c.recv(64 - got))
        out.append(time.perf_counter_ns() - t0)
        if gap:
            time.sleep(gap)
    out.sort()
    return {"min_ms": out[0] / 1e6, "p50_ms": out[n // 2] / 1e6, "p99_ms": out[int(0.99 * n)] / 1e6, "max_ms": out[-1] / 1e6}
print(json.dumps({"back_to_back": run(0), "paced": run(pace)}), flush=True)
"""

VARIANTS = {"-T": ["-T"], "-tt": ["-tt"], "-T lowdelay": ["-T", "-o", "IPQoS=lowdelay"],
            "-tt lowdelay": ["-tt", "-o", "IPQoS=lowdelay"], "-tt none": ["-tt", "-o", "IPQoS=none"]}


def echo(server: socket.socket) -> None:
    while True:
        c, _ = server.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def serve(c=c) -> None:
            while d := c.recv(4096):
                c.sendall(d)
        threading.Thread(target=serve, daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--port", type=int, default=7795)
    ap.add_argument("--pings", type=int, default=200)
    ap.add_argument("--pace-ms", type=float, default=12.8)
    a = ap.parse_args()
    to = next(line.split()[1:] for line in open("/etc/vy/peers") if line.split()[:1] == [a.peer])
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", a.port))
    server.listen(8)
    threading.Thread(target=echo, args=(server,), daemon=True).start()
    code = base64.b64encode(CLIENT.encode()).decode()
    remote = (f"python3 -c \"import base64; exec(base64.b64decode('{code}'))\" "
              f"{a.port} {a.pings} {a.pace_ms}")
    for name, opts in VARIANTS.items():
        try:
            r = subprocess.run(["ssh", *opts, "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                                "-R", f"127.0.0.1:{a.port}:127.0.0.1:{a.port}", *to, remote],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = "timeout", e.stdout or "", e.stderr or ""
            out, err = (x.decode(errors="replace") if isinstance(x, bytes) else x for x in (out, err))
        line = next((x.strip() for x in out.splitlines() if x.strip().startswith("{")), None)
        print(json.dumps({"variant": name, "ssh": shlex.join(opts), "rc": rc,
                          **(json.loads(line) if line else {"stdout": out[-400:], "stderr": err[-400:]})}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
