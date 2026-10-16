"""The latency of `run.sh`'s reverse tunnel, by ssh option: a server here at 127.0.0.1:PORT, `ssh -R` to PEER (a name
in /etc/vy/peers) running a client there that connects through the forward. Each variant measures the round trip
(pings back to back, and paced like the prover's roots) and the one-way stream the roots are: a small record every
PACE ms with nothing sent back, each record's delay over the stream's least (the two clocks' offset cancels).
Prints one JSON line a variant.

    python tunnel_rtt.py --peer vy-nebius-2 --port 7795 [--pings 200 --pace-ms 12.8 --records 2000]
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import socket
import struct
import subprocess
import threading
import time

CLIENT = r"""
import json, socket, struct, sys, time
port, n, pace, records = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) / 1e3, int(sys.argv[4])
def connect(mode):
    for _ in range(100):
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=30)
            break
        except OSError:
            time.sleep(0.1)
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.sendall(mode)
    if c.recv(5) != mode:
        sys.exit("PORT is not this probe's server")
    return c
c = connect(b"PINGS")
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
res = {"back_to_back": run(0), "paced": run(pace)}
c.close()
s = connect(b"ONEWY")
for k in range(records):
    s.sendall(struct.pack(">QQ", time.perf_counter_ns(), k))
    time.sleep(pace)
s.sendall(struct.pack(">QQ", 0, (1 << 64) - 1))
f = s.makefile("rb")
res["one_way"] = json.loads(f.readline())
print(json.dumps(res), flush=True)
"""

VARIANTS = {"-T": ["-T"], "-tt": ["-tt"], "-tt no-obscure": ["-tt", "-o", "ObscureKeystrokeTiming=no"],
            "-tt no-obscure lowdelay": ["-tt", "-o", "ObscureKeystrokeTiming=no", "-o", "IPQoS=lowdelay"]}


def exact(c: socket.socket, n: int) -> bytes:
    out = b""
    while len(out) < n:
        d = c.recv(n - len(out))
        if not d:
            raise EOFError
        out += d
    return out


def serve(c: socket.socket) -> None:
    with c:
        mode = exact(c, 5)
        c.sendall(mode)
        if mode == b"PINGS":
            while d := c.recv(4096):
                c.sendall(d)
            return
        offsets = []
        while True:
            stamp, k = struct.unpack(">QQ", exact(c, 16))
            if k == (1 << 64) - 1:
                break
            offsets.append(time.perf_counter_ns() - stamp)
        least = min(offsets)
        late = sorted(x - least for x in offsets)
        at = lambda q: late[min(len(late) - 1, int(q * len(late)))] / 1e6
        c.sendall((json.dumps({"records": len(late), "excess_p50_ms": at(0.5), "excess_p99_ms": at(0.99),
                               "excess_p999_ms": at(0.999), "excess_max_ms": late[-1] / 1e6,
                               "over_5ms": sum(x > 5e6 for x in late), "over_20ms": sum(x > 20e6 for x in late)})
                   + "\n").encode())


def accept(server: socket.socket) -> None:
    while True:
        c, _ = server.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve, args=(c,), daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--port", type=int, default=7795)
    ap.add_argument("--pings", type=int, default=200)
    ap.add_argument("--pace-ms", type=float, default=12.8)
    ap.add_argument("--records", type=int, default=2000)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    to = next(line.split()[1:] for line in open("/etc/vy/peers") if line.split()[:1] == [a.peer])
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", a.port))
    server.listen(8)
    threading.Thread(target=accept, args=(server,), daemon=True).start()
    code = base64.b64encode(CLIENT.encode()).decode()
    remote = (f"python3 -c \"import base64; exec(base64.b64decode('{code}'))\" "
              f"{a.port} {a.pings} {a.pace_ms} {a.records}")
    for name in a.variants.split(","):
        opts = VARIANTS[name]
        try:
            r = subprocess.run(["ssh", *opts, "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                                "-R", f"127.0.0.1:{a.port}:127.0.0.1:{a.port}", *to, remote],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
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
