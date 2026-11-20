#!/usr/bin/env bash
# The wipe's launcher run on a processor, honest and dishonest: a rehearsal that certifies nothing (the memory attester is this
# machine's under a key made for the run, the fill is in host memory, and `gpu-lease` is a stand-in that grants at once), for
# finding what breaks. From the tree's root, with `uv sync --all-packages --extra torch-cpu` done and elan on the PATH:
#
#   bash benchmarks/pous/smoke/wipe.sh
#
# The honest run is `served_wipe.sh --run-keys`'s own, which builds `wipe-protocol` in a scratch copy, runs the wipe and ends
# with the line `WIPE {...}`: its verdict and the SHA-512 of the executable that decided. The case it keeps is then asked of
# that executable built beside, unchanged (accept), with a trusted key that is not the pinned one, and with one word of the
# signed session changed: each of the last two a reject whose reason is printed. Exit 0 where the first is accepted and the
# other two are refused.
set -o pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
export PATH="$HOME/.elan/bin:$PATH" DEVELOPMENT=1 PY=${PY:-$ROOT/.venv/bin/python}
FB=$(mktemp -d); SERVE=$(mktemp -d); DIR=$SERVE/out
trap 'rm -rf "$FB" "$SERVE"' EXIT
printf '#!/usr/bin/env bash\nwhile [ "$1" != -- ]; do shift; done\nshift\nGPU_LEASE_UUID=GPU-standin exec "$@"\n' > "$FB/gpu-lease"
chmod +x "$FB/gpu-lease"
echo "### honest: served_wipe.sh, host store, python streamer, run keys"
PATH="$FB:$PATH" bash benchmarks/pous/streamed_wipe/served_wipe.sh "$SERVE" 0 "$DIR" '{"run_keys": true, "store": "host", "streamer": "python"}' \
  | tee "$SERVE/said.txt" | tail -1 | cut -c1-600
J=verity/protocols/resource_accounting/memory/pous/attester/lean
(cd "$J" && lake build wipe-protocol) > "$SERVE/lake.log" 2>&1 || { tail -5 "$SERVE/lake.log"; exit 7; }
EXE=$ROOT/$J/.lake/build/bin/wipe-protocol
echo "wipe-protocol sha512 $(sha512sum "$EXE" | cut -c1-128)"
"$PY" - "$SERVE/private/wipe/wipe.case.json" "$SERVE" "$EXE" <<'PY'
import json, subprocess, sys
case_file, serve, exe = sys.argv[1:]
case = json.load(open(case_file))
node = bytes.fromhex(case["params"]["server"]).decode()
swapped = case | {"params": case["params"] | {"trusted": ["00" * 64]}}
evidence = json.loads(bytes.fromhex(case["evidence"]))
batch = evidence["signed"]["batches"][0]
batch[next(k for k, v in batch.items() if isinstance(v, int))] += 1
changed = case | {"evidence": json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode().hex()}
ok = True
for name, c, want in (("honest", case, "accept"), ("trusted key not the pinned one", swapped, "reject"), ("one word changed", changed, "reject")):
    line = json.dumps(c, sort_keys=True, separators=(",", ":")) + "\n"
    out = subprocess.run([exe, "--server", node, "serve"], input=line, capture_output=True, text=True).stdout
    a = json.loads(out.splitlines()[0])
    print(f"{name}: {a['verdict']} {a.get('reason') or ''}")
    ok &= a["verdict"] == want
sys.exit(0 if ok else 1)
PY
