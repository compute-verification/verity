set -x
V=64.247.206.95
python3 - <<'PY' | tee ${RESEARCH_RUN_DIR}/out-rtt.txt
import socket, time, statistics
ts=[]
for _ in range(20):
    t=time.perf_counter(); s=socket.create_connection(("64.247.206.95", 18876), timeout=5); ts.append((time.perf_counter()-t)*1e3); s.close(); time.sleep(0.05)
print("RTT_TCP_CONNECT_MS", {"median": statistics.median(ts), "min": min(ts), "max": max(ts), "n": len(ts), "target": "verifier pod sshd 64.247.206.95:18876"})
PY
MODE=selftest VUS="8 64" bash verity/ml/flock/pod/20-gpu-link.sh
MODE=prove VERIFIER=$V:18877 SESS_VUS=4096 RUNS=6 bash verity/ml/flock/pod/20-gpu-link.sh
MODE=prove SESS_VUS="8192 16384" RUNS=3 bash verity/ml/flock/pod/20-gpu-link.sh
true
