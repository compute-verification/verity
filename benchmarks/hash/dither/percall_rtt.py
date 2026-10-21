"""ASK-64's per-call round, measured from the serving node: one live challenger round a linear call.

The challenger (`sampled_proofs.challenger.LiveChallenger`, the file --challenger names, #2654's) runs here, on the
auditor's machine, with the verifier's `LiveRecord` of each family; the client runs on PEER, the serving node, over ssh:
it sends each call's 64-byte root and waits for its 64-byte coin.  The nodes admit TCP 22 alone between them (the
security group, `research/pods/nebius/launch.sh`), so the client reaches the challenger through `ssh -R`, with
`remote_session/run.sh`'s tunnel options (-tt and ObscureKeystrokeTiming=no: a tty sets TCP_NODELAY at both ends).

Modes, each on its own connection after WARM echo pings:
    pings     64 bytes out and back through the tunnel, no challenger: the link's own round
    coin      the root out, the coin back (64 B each way), back to back: one call's round
    paced     the same with PACE_US between rounds, as a decode step's calls come (not back to back)
    opening   the root out, the whole opening back (coin, hm96 salt, path): what the verifier's record keeps
    loopback  the coin round on this machine over 127.0.0.1, no ssh: the challenger's own cost
The challenger's time from a root's receipt to its coin's release is recorded per round.  At the end each family's
record is checked (`LiveRecord.reason`: every coin opened once, after its root).  Writes OUT (quantiles and every
sample, ns) and, beside it, `percall-wait-ns.txt`: paced's samples, the wait the served arm replays a call.

    python3 percall_rtt.py --peer vy-nebius-2 --challenger challenger.py --out out/percall-rtt.json [--rounds 4096]
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEME = "verity/dither/percall"
SSH = ["-tt", "-o", "ObscureKeystrokeTiming=no", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
       "-o", "ServerAliveInterval=15"]
MODES = {b"COINS": "coin", b"PACED": "paced", b"OPENS": "opening", b"LOOPB": "loopback"}

CLIENT = r"""
import hashlib, json, socket, struct, sys, time
port, rounds, orounds, pace, warm = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) / 1e6, int(sys.argv[5])
def exact(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d:
            raise EOFError
        b += d
    return b
def connect(mode):
    for _ in range(100):
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=60)
            break
        except OSError:
            time.sleep(0.1)
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.sendall(mode)
    if exact(c, 5) != mode:
        sys.exit("PORT is not this probe's server")
    return c
def run(mode, n, gap, opening):
    c = connect(mode)
    out = []
    for i in range(n):
        root = hashlib.sha512(mode + i.to_bytes(8, "big")).digest()
        t0 = time.perf_counter_ns()
        c.sendall(root)
        if opening:
            exact(c, struct.unpack(">I", exact(c, 4))[0])
        else:
            exact(c, 64)
        out.append(time.perf_counter_ns() - t0)
        if gap:
            time.sleep(gap)
    c.close()
    return out
res = {"host": socket.gethostname()}
res["pings"] = run(b"PINGS", warm + rounds, 0, False)[warm:]
res["coin"] = run(b"COINS", rounds, 0, False)
res["paced"] = run(b"PACED", rounds, pace, False)
res["opening"] = run(b"OPENS", orounds, 0, True)
blob = json.dumps(res).encode()
c = connect(b"RSLTS")
c.sendall(struct.pack(">Q", len(blob)) + blob)
print("percall-client:", exact(c, 2).decode(), len(blob), flush=True)
"""


def exact(c: socket.socket, n: int) -> bytes | None:
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d:
            if b:
                raise EOFError(f"{len(b)} of {n} bytes")
            return None
        b += d
    return b


def quantiles(xs: list[int]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))] / 1e3
    return {"n": len(s), "min_us": s[0] / 1e3, "p50_us": at(0.5), "p90_us": at(0.9), "p99_us": at(0.99),
            "p999_us": at(0.999), "max_us": s[-1] / 1e3, "mean_us": round(sum(s) / len(s) / 1e3, 3)}


class Server:
    def __init__(self, ch, rounds: int, orounds: int) -> None:
        fam = [("coin", rounds), ("paced", rounds), ("opening", orounds), ("loopback", rounds)]
        t0 = time.perf_counter()
        self.ch = ch.LiveChallenger(SCHEME, 0, fam, os.urandom)
        self.build_s = round(time.perf_counter() - t0, 3)
        self.rec = {label: ch.LiveRecord(self.ch, label) for label, _ in fam}
        self.serve_ns: dict[str, list[int]] = {label: [] for label, _ in fam}
        self.opening_bytes = None
        self.results: dict = {}
        self.done = threading.Event()

    def handle(self, c: socket.socket) -> None:
        with c:
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            mode = exact(c, 5)
            c.sendall(mode)
            if mode == b"PINGS":
                while (d := c.recv(4096)):
                    c.sendall(d)
            elif mode in MODES:
                label = MODES[mode]
                rec, ns = self.rec[label], self.serve_ns[label]
                while (root := exact(c, 64)) is not None:
                    t0 = time.perf_counter_ns()
                    o = rec.record(root)
                    if label == "opening":
                        body = o.coin + o.salt + b"".join(o.path)
                        out = struct.pack(">I", len(body)) + body
                        self.opening_bytes = len(body)
                    else:
                        out = o.coin
                    ns.append(time.perf_counter_ns() - t0)
                    c.sendall(out)
            elif mode == b"RSLTS":
                n = struct.unpack(">Q", exact(c, 8))[0]
                self.results = json.loads(exact(c, n))
                c.sendall(b"OK")
                self.done.set()

    def accept(self, srv: socket.socket) -> None:
        while True:
            c, _ = srv.accept()
            threading.Thread(target=self.handle, args=(c,), daemon=True).start()


def loopback(port: int, rounds: int) -> list[int]:
    import hashlib
    c = socket.create_connection(("127.0.0.1", port), timeout=60)
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.sendall(b"LOOPB")
    assert exact(c, 5) == b"LOOPB"
    out = []
    for i in range(rounds):
        root = hashlib.sha512(b"LOOPB" + i.to_bytes(8, "big")).digest()
        t0 = time.perf_counter_ns()
        c.sendall(root)
        exact(c, 64)
        out.append(time.perf_counter_ns() - t0)
    c.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peer", required=True, help="the serving node: a name in /etc/vy/peers")
    ap.add_argument("--challenger", required=True, help="sampled_proofs/challenger.py at the head --challenger-sha names")
    ap.add_argument("--challenger-sha", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=0, help="default: one in 7900-8099")
    ap.add_argument("--rounds", type=int, default=4096)
    ap.add_argument("--opening-rounds", type=int, default=1024)
    ap.add_argument("--pace-us", type=float, default=400.0)
    ap.add_argument("--warm", type=int, default=64)
    a = ap.parse_args()
    me = socket.gethostname()
    if me == a.peer or me.split(".")[0] == a.peer:
        print(f"percall_rtt: the challenger must not run on the serving node ({me})", flush=True)
        return 3
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("challenger", a.challenger)
    ch = sys.modules["challenger"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)
    peers = {ln.split()[0]: ln.split()[1:] for ln in open("/etc/vy/peers") if ln.split() and not ln.startswith("#")}
    to = peers[a.peer]
    sv = Server(ch, a.rounds, a.opening_rounds)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = a.port or 7900 + os.getpid() % 200
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    threading.Thread(target=sv.accept, args=(srv,), daemon=True).start()
    loop = loopback(port, a.rounds)
    code = base64.b64encode(CLIENT.encode()).decode()
    remote = (f"python3 -c \"import base64; exec(base64.b64decode('{code}'))\" "
              f"{port} {a.rounds} {a.opening_rounds} {a.pace_us} {a.warm}")
    cmd = ["ssh", *SSH, "-R", f"127.0.0.1:{port}:127.0.0.1:{port}", *to, remote]
    t0 = time.time()
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800)
    wall = round(time.time() - t0, 1)
    if not sv.done.is_set():
        print(json.dumps({"ssh_rc": r.returncode, "stdout": r.stdout[-800:], "stderr": r.stderr[-800:]}), flush=True)
        return 2
    res = sv.results
    reasons = {label: rec.reason() for label, rec in sv.rec.items() if rec.roots}
    doc = {
        "what": "ASK-64 perCall: one live challenger round a linear call, the client on the serving node",
        "challenger_host": me, "client_host": res.get("host"), "peer": a.peer,
        "transport": "ssh -R forward over the private network (the nodes admit TCP 22 alone): ssh " + shlex.join(SSH),
        "challenger": {"module": "verity.core.protocols.sampled_proofs.challenger", "sha": a.challenger_sha,
                       "scheme": SCHEME, "families": [[k, len(v.roots)] for k, v in sv.rec.items()],
                       "root": sv.ch.commitment.root.hex(), "build_s": sv.build_s,
                       "record_reasons": reasons, "records_ok": all(v is None for v in reasons.values())},
        "coin_bytes": 64, "root_bytes": 64, "opening_bytes": sv.opening_bytes, "pace_us": a.pace_us,
        "client_wall_s": wall,
        "round_trip": {k: quantiles(res[k]) for k in ("pings", "coin", "paced", "opening")},
        "loopback": quantiles(loop),
        "challenger_serve": {k: quantiles(v) for k, v in sv.serve_ns.items()},
        "samples_ns": {**{k: res[k] for k in ("pings", "coin", "paced", "opening")}, "loopback": loop},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc) + "\n")
    (out.parent / "percall-wait-ns.txt").write_text("\n".join(str(x) for x in res["paced"]) + "\n")
    print(json.dumps({k: doc[k] for k in ("round_trip", "loopback", "challenger_serve")}, indent=1), flush=True)
    print(json.dumps({"records_ok": doc["challenger"]["records_ok"], "opening_bytes": sv.opening_bytes,
                      "client_host": doc["client_host"], "challenger_host": me}), flush=True)
    return 0 if doc["challenger"]["records_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
