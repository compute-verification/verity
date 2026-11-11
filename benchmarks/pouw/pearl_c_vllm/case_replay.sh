#!/usr/bin/env bash
# Pearl+'s run verdict taken again off the served node, from the run's case, by the verifiers of the tree this runs in:
#
#   case_replay.sh CASE W COIN OUT        (cwd: the verifier's tree, one the job may build in)
#
# CASE is `served_passes.py case`'s output (parts.json and blobs/), W the window record the passes are held to, COIN the
# verifier's coin record as it was drawn, and OUT where the answers go. It rebuilds the passes from the case
# (`case_passes.py`, beside this file), then runs `served_passes.py verify --tie W` (Python) and `served_passes.py lean`
# (`pouw-pearlplus`, built here) on them and on copies of W and COIN.
#
# Nothing draws: a coin record that exists pins the verdict's source, the copies are read-only, no challenger gate is in
# the verifiers' environment, and the step fails if the coin record's bytes changed or a draw was asked for. The passes
# hold only the files the opened tiles read, so a verifier that reads any other file fails on it; OUT/opened.json lists
# what the Python verifier opened under them beside what the case names.
#
# The blobs are opened tile values, witness: the rebuilt passes and the copies live in a 0700 scratch directory that is
# removed when this ends, never under OUT. OUT gets verify.json, pearlplus-answer.json, pearlplus-lean.json,
# pearlplus-lean-check.json, opened.json, replay.json (the inputs' and the executable's hashes, the tree's commit, both
# verdicts) and the logs. Exit 0 when both verifiers accept and agree, 1 when one rejects, 2 when a step failed.
set -uo pipefail
[ $# = 4 ] || { sed -n '2,5p' "$0" >&2; exit 64; }
CASE=$(realpath "$1") W=$(realpath "$2") COIN=$(realpath "$3")
mkdir -p "$4" && OUT=$(realpath "$4")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=$PWD
umask 077
S=$(mktemp -d "${REPLAY_SCRATCH:-${TMPDIR:-/tmp}}/pearlplus-replay.XXXXXX") || exit 2
trap 'rm -rf "$S"' EXIT
say() { echo "REPLAY $*"; }
sha() { sha512sum "$1" | cut -d' ' -f1; }
fail() { say "FAILED: $*"; exit 2; }

python3 "$HERE/case_passes.py" "$CASE" "$S/passes" > "$OUT/case-passes.log" 2>&1 || fail "$(tail -1 "$OUT/case-passes.log")"
say "$(tail -1 "$OUT/case-passes.log" | sed "s#$S#SCRATCH#g")"
cp "$W" "$S/W.json" && cp "$COIN" "$S/coin.json" && chmod 0400 "$S/W.json" "$S/coin.json" || fail "W or the coin record can't be copied"
COIN_SHA=$(sha "$S/coin.json") W_SHA=$(sha "$S/W.json")
DRAWS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["params"]["draws"])' "$CASE/parts.json") || fail "the parts name no draws"
COMMIT=${RESEARCH_SOURCE_SHA:-$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unknown)}

uv sync -q --all-packages --extra torch-cpu > "$OUT/sync.log" 2>&1 || fail "uv sync: $(tail -1 "$OUT/sync.log")"
PY=$TREE/.venv/bin/python
LX=$TREE/verity/core/protocols/pouw/lean/pearlplus/.lake/build/bin/pouw-pearlplus
(cd verity/core/protocols/pouw/lean/pearlplus && lake build pouw-pearlplus) > "$OUT/pouw-pearlplus-build.log" 2>&1 \
  || fail "pouw-pearlplus doesn't build: $(tail -1 "$OUT/pouw-pearlplus-build.log")"
say "built pouw-pearlplus at $COMMIT, sha256 $(sha256sum "$LX" | cut -c1-16)"

# The Python verifier's opens under the passes, by an audit hook every interpreter of the step loads: it only records.
mkdir "$S/hook"
cat > "$S/hook/sitecustomize.py" <<'EOF'
import os, sys
_base, _fd = os.path.abspath(os.environ["REPLAY_OPEN_BASE"]) + os.sep, os.open(
    os.environ["REPLAY_OPEN_LOG"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
def _hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.path.abspath(os.fsdecode(args[0]))
        if p.startswith(_base):
            os.write(_fd, (p[len(_base):] + "\n").encode())
sys.addaudithook(_hook)
EOF
t=$(date +%s)
env -u PEARLC_CHALLENGER_GATE -u PEARLC_CHALLENGER_PEER PYTHONPATH="$S/hook" REPLAY_OPEN_BASE="$S/passes" \
  REPLAY_OPEN_LOG="$S/opened.txt" "$PY" benchmarks/pouw/pearl_c_vllm/served_passes.py verify --passes "$S/passes" \
  --coin-record "$S/coin.json" --tie "$S/W.json" --draws "$DRAWS" --commit "$COMMIT" --jobs "${REPLAY_JOBS:-$(nproc)}" --fast \
  --out "$OUT/verify.json" --answer "$OUT/pearlplus-answer.json" > "$OUT/verify.log" 2>&1
prc=$?
say "python: served_passes.py verify exit $prc after $(( ($(date +%s) - t) / 60 )) min: $(tail -1 "$OUT/verify.log" | cut -c1-700)"
python3 - "$CASE/parts.json" "$S/opened.txt" "$OUT/opened.json" <<'EOF'
import json, sys
parts = json.load(open(sys.argv[1]))
named = {f"pass-{i:05d}/{p}" for i, f in enumerate(parts["evidence"]["files"]) for p in f}
given = named | {f"pass-{i:05d}/manifest.json" for i in range(len(parts["manifests"]))} | {"coins.json"}
try:
    opened = {x for x in open(sys.argv[2]).read().splitlines() if x}
except OSError:
    opened = set()
doc = {"opened": len(opened), "case_files": len(named), "opened_case_files": len(opened & named),
       "opened_outside_the_case": sorted(opened - given), "case_files_not_opened": sorted(named - opened),
       "opened_other": sorted(opened & (given - named))}
json.dump(doc, open(sys.argv[3], "w"), indent=1)
print(f"REPLAY python opened {doc['opened']} files under the passes: {doc['opened_case_files']} of the case's "
      f"{doc['case_files']} opened tiles' files, {len(doc['opened_other'])} manifests and coins, "
      f"{len(doc['opened_outside_the_case'])} outside the case")
EOF

lrc=2
if [ -s "$OUT/verify.json" ]; then
  t=$(date +%s)
  env -u PEARLC_CHALLENGER_GATE -u PEARLC_CHALLENGER_PEER "$PY" benchmarks/pouw/pearl_c_vllm/served_passes.py lean \
    --passes "$S/passes" --verify "$OUT/verify.json" --exe "$LX" --out "$OUT/pearlplus-lean.json" \
    --timeout "${REPLAY_LEAN_TIMEOUT:-7200}" > "$OUT/lean.log" 2>&1
  lrc=$?
  say "lean: served_passes.py lean exit $lrc after $(( ($(date +%s) - t) / 60 )) min: $(tail -1 "$OUT/lean.log" | cut -c1-700)"
fi

[ "$(sha "$S/coin.json")" = "$COIN_SHA" ] && [ "$(sha "$COIN")" = "$COIN_SHA" ] || fail "the coin record's bytes changed: a verdict on a recorded coin writes none"
[ ! -e "$S/coin.json.asked" ] || fail "a draw was asked for: a verdict on a recorded coin asks for none"
python3 - "$OUT" "$CASE/parts.json" "$COIN_SHA" "$W_SHA" "$COMMIT" "$LX" "$prc" "$lrc" <<'EOF'
import hashlib, json, sys
from pathlib import Path
out, parts, coin, w, commit, exe, prc, lrc = sys.argv[1:]
out = Path(out)
def doc(name):
    try:
        return json.loads((out / name).read_text())
    except (OSError, ValueError):
        return None
verify, lean, check = doc("verify.json"), doc("pearlplus-lean.json"), doc("pearlplus-lean-check.json")
answer = (verify or {}).get("answer") or {}
replay = {"schema": "verity-pouw/pearl-plus-case-replay/v1", "verifier_commit": commit,
          "parts_sha512": hashlib.sha512(Path(parts).read_bytes()).hexdigest(), "coin_sha512": coin, "w_sha512": w,
          "coin_unchanged": True, "exe_sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest() if Path(exe).is_file() else None,
          "python": {"exit": int(prc), "verdict": answer.get("verdict"), "reason": answer.get("reason"),
                     "tie": (verify or {}).get("tie"), "draw": (verify or {}).get("draw"), "line": (verify or {}).get("line")},
          "lean": {"exit": int(lrc), "verdict": (lean or {}).get("verdict"), "reason": (check or {}).get("reason"),
                   "tie": (lean or {}).get("tie"), "differ": (check or {}).get("differ")},
          "opened": doc("opened.json")}
(out / "replay.json").write_text(json.dumps(replay, indent=1) + "\n")
EOF
[ "$prc" = 0 ] && [ "$lrc" = 0 ] && exit 0
[ "$prc" = 1 ] || [ "$lrc" = 1 ] && exit 1
exit 2
