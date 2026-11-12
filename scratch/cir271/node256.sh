set -uo pipefail
uv sync -q --all-packages --extra torch-cpu || { echo "NODE256 uv sync failed"; exit 2; }
mkdir -p "$OUT"
echo "NODE256 tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') on $(hostname) from $(date -u +%H:%M:%SZ)"
cat > "$OUT/node.py" <<'PY'
import importlib.util, json, resource, sys, time, traceback
from collections import Counter
from pathlib import Path
out, ex = Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
from verity_vllm.program.predict.node import build_node_program
reg = cc.registered(0, None, ex)
t = time.perf_counter()
rec = {"execution": ex, "LP": 256, "T": 31}
try:
    r = build_node_program(reg.phi, "request", LP=256, T=31, with_desc=False)
    rec["wall_s"] = round(time.perf_counter() - t, 1)
    try:
        rec["summary"] = r.summary()
    except Exception as e:
        rec["summary_error"] = f"{type(e).__name__}: {e}"[:400]
    fn = getattr(r, "node_fn", None)
    body = getattr(fn, "body", None)
    if body is not None:
        rec["body_nodes"] = len(body.nodes)
        rec["body_forms"] = dict(Counter(n.form for n in body.nodes))
        names = Counter(getattr(n.fn, "name", None) or getattr(n.fn, "id", "?").split("{")[0] for n in body.nodes if n.form in ("call", "batch"))
        rec["called_names"] = len(names)
        rec["top"] = names.most_common(12)
except BaseException as e:
    rec["wall_s"] = round(time.perf_counter() - t, 1)
    rec["error"] = f"{type(e).__name__}: {e}"[:1200]
    rec["trace"] = traceback.format_exc()[-2500:]
rec["peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
(out / f"node_{ex}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
print("NODE256", ex, json.dumps({k: v for k, v in rec.items() if k not in ("trace", "summary")}, default=str)[:1500], flush=True)
if "error" in rec:
    print(rec["trace"][-1500:], flush=True)
PY
.venv/bin/python "$OUT/node.py" "$OUT" eager; echo "NODE256 eager rc=$? at $(date -u +%H:%M:%SZ)"
