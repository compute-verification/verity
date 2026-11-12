set -uo pipefail
uv sync -q --all-packages --extra torch-cpu || { echo "REQ256 uv sync failed"; exit 2; }
mkdir -p "$OUT"
echo "REQ256 tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') on $(hostname) from $(date -u +%H:%M:%SZ)"
cat > "$OUT/counts.py" <<'PY'
import importlib.util, json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
from verity_vllm.program.predict.steps import plan_steps
from verity_vllm.program.predict.template_counts import TemplateCounts
advice, ingress = cc.session([(0, 256, 32)])
res = {}
for ex in ("eager", "compiled"):
    try:
        reg = cc.registered(0, None, ex)
        t = time.perf_counter()
        plan = plan_steps(reg, advice, ingress)
        u = [x for x in plan.requests if x is not None][0]
        n = TemplateCounts(reg.phi).unit(u)
        res[ex] = {"LP": u.LP, "T": u.T, "C": u.C, "chunks": list(u.chunks), "templates": len(n), "instances": sum(n.values()),
                   "plan_s": round(time.perf_counter() - t, 1),
                   "top": sorted(((v, k[:160]) for k, v in n.items()), reverse=True)[:15]}
        print("REQ256 counts", ex, json.dumps({k: v for k, v in res[ex].items() if k != "top"}), flush=True)
    except Exception as e:  # the count is a side reading; the compile below is the measurement
        res[ex] = {"error": f"{type(e).__name__}: {e}"[:600]}
        print("REQ256 counts", ex, "failed:", res[ex]["error"], flush=True)
(out / "counts.json").write_text(json.dumps(res, indent=1) + "\n")
PY
.venv/bin/python "$OUT/counts.py" "$OUT"; echo "REQ256 counts rc=$? at $(date -u +%H:%M:%SZ)"
rc=0
for ex in eager compiled; do
  .venv/bin/python benchmarks/compile_cost/run.py --requests 0:256:32 --rank 0 --execution "$ex" --out "$OUT/$ex" > "$OUT/$ex.log" 2>&1
  r=$?; [ $r = 0 ] || rc=$r
  grep -E '^COMPILE-COST' "$OUT/$ex.log" | cut -c1-600
  [ $r = 0 ] || tail -n 15 "$OUT/$ex.log" | cut -c1-300
  echo "REQ256 compile $ex rc=$r at $(date -u +%H:%M:%SZ)"
done
exit $rc
