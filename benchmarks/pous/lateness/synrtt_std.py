"""TCP handshake round trips to HOST:PORT (stdlib only): connect() returns when the SYN-ACK arrives; each connection is
reset at once (SO_LINGER 0); samples are paced. Prints JSON with every sample in ns.

    python3 synrtt_std.py HOST PORT N GAP_MS
"""
import json
import socket
import struct
import sys
import time

host, port, n, gap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) / 1000
samples, errors = [], 0
for _ in range(n):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    s.settimeout(3.0)
    t0 = time.perf_counter_ns()
    try:
        s.connect((host, port))
        samples.append(time.perf_counter_ns() - t0)
    except OSError:
        errors += 1
    finally:
        s.close()
    time.sleep(gap)
xs = sorted(samples)
q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))] / 1e3 if xs else None  # noqa: E731
print(json.dumps({"from": socket.gethostname(), "to": f"{host}:{port}", "n": len(xs), "errors": errors, "gap_ms": gap * 1e3,
                  "min_us": q(0), "p50_us": q(0.5), "p99_us": q(0.99), "max_us": xs[-1] / 1e3 if xs else None,
                  "samples_ns": samples}))
