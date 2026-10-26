set -euo pipefail
uv sync -q --all-packages --extra torch-cpu
cat > "$OUT/digest.py" <<'PY'
import faulthandler, json, os, resource, sys, time
faulthandler.dump_traceback_later(600, repeat=True)
log = lambda m: print(time.strftime('%H:%M:%S'), m, round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1), 'GB', file=sys.stderr, flush=True)
sys.path[:0] = ["benchmarks/e2e_h200", "integrations/torch_loop"]
from e2e_h200 import program as PR
from e2e_h200.shape import SHAPES
from verity.ml.circuits.codec import program_digest
from verity_torch_loop import weights as SW
out = {}
for name in sys.argv[1:]:
    sh = SHAPES[name] if name in SHAPES else SW.Shape(name, 16384, int(name.split('-l')[1]), 256, 64, 1, 65536, 256, 32, 8192)
    t0 = time.time()
    log(name + ' build'); prog = PR.program(sh); log(name + ' built')
    t1 = time.time()
    dg = program_digest(prog); log(name + ' digest ' + dg[:16])
    t2 = time.time()
    try:
        r = SW.reader(sh, prog)
        rd = {"reader_digest_ok": r.digest == dg, "reader_widths": sorted({b for _, b in r.widths}),
              "tensors_read": len(r.widths)}
    except ValueError as e:
        rd = {"reader_refused": str(e)}
    t3 = time.time()
    out[name] = {"shape": sh.name, "row_codes": sh.row_codes, "digest": dg, **rd,
                 "build_s": round(t1 - t0, 1), "digest_s": round(t2 - t1, 1), "reader_s": round(t3 - t2, 1),
                 "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                 "nodes": len(prog.fn.body.nodes), "callees": sorted({n.fn.id for n in prog.fn.body.nodes})}
    print(json.dumps({name: {k: v for k, v in out[name].items() if k != "callees"}}), flush=True)
    json.dump(out, open(os.path.join(os.environ["OUT"], "digest.json"), "w"), indent=1)
PY
uv run python "$OUT/digest.py" "$@"
