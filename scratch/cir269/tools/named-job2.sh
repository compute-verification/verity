# CIR-269's named-target job: each target of the list below checked alone, in a circuit-check of its own with a fresh cache,
# its report and log under $OUT/r/; each `targets._handed` call (the head's handed path) logged with its caller to
# $OUT/r/N.hooks.  P targets at once.
set -uo pipefail
mkdir -p /workspace/research/scratch
d=$(mktemp -d -p /workspace/research/scratch); trap 'rm -rf "$d"' EXIT
uv sync -q --all-packages --extra torch-cpu </dev/null > "$OUT/sync.log" 2>&1 || { tail -20 "$OUT/sync.log"; exit 9; }
git rev-parse HEAD > "$OUT/head.txt"
echo "J head $(cat "$OUT/head.txt") $(nproc) cpus $(date -u +%H:%M:%SZ)"
mkdir -p "$OUT/r"
cat > "$d/hooked.py" <<'PY'
import json, os, sys
import circuit_check.targets as TG
from circuit_check import cache as CC
if hasattr(TG, "_handed"):
    _orig = TG._handed
    def _handed(name, _o=_orig):
        f = sys._getframe(1)
        with CC.internal():
            with open(os.environ["HOOKLOG"], "a") as fh:
                fh.write(json.dumps([name, f.f_code.co_name, os.path.basename(f.f_code.co_filename), os.getpid()]) + "\n")
        return _o(name)
    TG._handed = _handed
from circuit_check.cli import main
sys.exit(main())
PY
cat > "$d/pre.py" <<'PY2'
import json, os, sys
import circuit_check.targets as TG
from circuit_check import cache as CC
if hasattr(TG, "_handed"):
    _orig = TG._handed
    def _handed(name, _o=_orig):
        f = sys._getframe(1)
        with open(os.environ["HOOKLOG"], "a") as fh:
            fh.write(json.dumps([name, f.f_code.co_name, os.path.basename(f.f_code.co_filename), os.getpid()]) + "\n")
        return _o(name)
    TG._handed = _handed
    built = TG.preload()
else:
    from verity.definitions.numerics import twins
    built = twins.preload()
print(json.dumps({k: str(v) for k, v in sorted(built.items())}, indent=1))
PY2
HOOKLOG="$OUT/r/preload.hooks" uv run python "$d/pre.py" > "$OUT/r/preload.json" 2> "$OUT/r/preload.err"; echo "J preload rc=$?"
cat > "$d/targets.txt" <<'EOF'
Fp8GroupScale_v2{G=32}
Fp8RowAbsMax_v2{K=256}
Fp8ActQuant_v2{K=256}
F32ToE4m3Sat_v2
Fp8RowScale_v2
F32LtFtz_v2
EOF
one() {
  i=$1; t=$2; t0=$(date +%s)
  HOOKLOG="$OUT/r/$i.hooks" uv run python "$d/hooked.py" -j 1 --cache "$d/c$i" --out "$OUT/r/$i.json" "$t" > "$OUT/r/$i.log" 2>&1
  echo "J $i rc=$? s=$(( $(date +%s) - t0 )) $t"
}
export -f one; export d OUT
nl -ba -nln "$d/targets.txt" | sed 's/[[:space:]]*$//' | awk -F'\t' '{gsub(/ /,"",$1); print $1"\t"$2}' > "$d/list.tsv"
tr '\t' '\n' < "$d/list.tsv" | xargs -d '\n' -n 2 -P "${P:-3}" bash -c 'one "$((27 + $0))" "$1"'
echo "J done $(date -u +%H:%M:%SZ)"
