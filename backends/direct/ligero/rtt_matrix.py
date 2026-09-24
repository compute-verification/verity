"""rtt_matrix.py HOST:PORT[=LABEL] ... [--n 9] [--json]: TCP connect (SYN -> SYN/ACK) round trip from this machine to each target.

Run on a candidate live-verifier pod against the prover pods' public ssh ports: the connect time is the path RTT the live
session's ctrl round trips pay (plus the verifier's own processing, ~0.1 ms).  No payload is sent; the peer sees a closed
connection.  Prints min / median / max ms per target (min is the path floor, median what a session sees).
"""
import json
import socket
import statistics
import sys
import time


def connect_ms(host: str, port: int, timeout: float = 3.0) -> float | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    t0 = time.perf_counter()
    try:
        s.connect((host, port))
        return (time.perf_counter() - t0) * 1e3
    except OSError:
        return None
    finally:
        s.close()


def main(argv: list[str]) -> int:
    n = 9
    as_json = "--json" in argv
    args = [a for a in argv if a != "--json"]
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        del args[i:i + 2]
    out = []
    for spec in args:
        addr, _, label = spec.partition("=")
        host, port = addr.rsplit(":", 1)
        xs = [connect_ms(host, int(port)) for _ in range(n)]
        ok = [x for x in xs if x is not None]
        row = {"target": addr, "label": label or addr, "n": n, "ok": len(ok),
               "min_ms": min(ok) if ok else None, "median_ms": statistics.median(ok) if ok else None,
               "max_ms": max(ok) if ok else None}
        out.append(row)
        if not as_json:
            f = lambda v: f"{v:7.1f}" if v is not None else "      -"
            print(f"{row['label']:28s} {addr:24s} ok {len(ok)}/{n}  min {f(row['min_ms'])}  med {f(row['median_ms'])}  max {f(row['max_ms'])} ms")
    if as_json:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
