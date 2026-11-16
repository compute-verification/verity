# lean_check.sh, sourced by lean_shard.sh and stage_prove.sh: Flock's Lean verifier (the executable `verity verify proof`
# runs) on one kept session of a statement, as `verity bench prove lean-verify` records it. The caller sets LEAN_PY (an
# array: the command that runs Python with the tree's packages), LEAN_BENCH (an array: the command that runs `verity
# bench`) and LEAN_TMP (a scratch directory).

# lean_exes VERIFIER_DIR: the [[lean_exe]] names whose roots are FlockVerify and FlockReadTmpl, space-separated
lean_exes() {
  "${LEAN_PY[@]}" -c 'import sys, tomllib
r = {e.get("root"): e["name"] for e in tomllib.load(open(sys.argv[1], "rb"))["lean_exe"]}
print(r["FlockVerify"], r["FlockReadTmpl"])' "$1/lakefile.toml"
}

# lean_word DIR: DIR/program-word.json, the Program of the word DIR/stage.json names (lean-verify records it and compares it
# by name; it binds nothing in the circuit); prints the word and the descriptor's SHA-512
lean_word() {
  "${LEAN_PY[@]}" - "$1" <<'EOF'
import hashlib, json, sys
from pathlib import Path
from verity.circuits.codec import descriptor_chunks
from verity.circuits.program import Program
from verity_vllm.pipeline.serving_rows import template
d = Path(sys.argv[1]); word = json.loads((d / "stage.json").read_text())["word"]
h = hashlib.sha512()
with open(d / "program-word.json", "wb") as f:
    for c in descriptor_chunks(Program(template(word).fn)):
        f.write(c); h.update(c)
desc = json.loads((d / "program-word.json").read_text())
assert desc.get("program_fn") == word and "annotations" not in desc, (desc.get("program_fn"), word)
print(json.dumps({"word": word, "sha512": h.hexdigest()}))
EOF
}

# lean_check NAME SESSION_DIR STATEMENT_DIR KEPT_DIR OUT_DIR EXE TSV: one run of lean-verify on the session, at the statement
# STATEMENT_DIR holds (circuit.txt, stage.json, and its Program: the stager's program.json, the Program a bound statement's
# partition names, else program-word.json) with KEPT_DIR's public files (pub.bin, partition.json, tables/); its record
# OUT_DIR/lean-verify.json and its row in TSV (name, session, accepted, label or why, wall s, peak GiB, circuit bytes).
# 0 only when the verifier accepts.
lean_check() {
  local name=$1 sd=$2 d=$3 k=$4 o=$5 exe=$6 tsv=$7 sid t st rc p=program-word.json
  sid=$(basename "$sd"); t=$LEAN_TMP/lean-$name-$sid
  rm -rf "$t"; mkdir -p "$t" "$o"
  [ -e "$d/program.json" ] && p=program.json
  cp -r "$sd" "$t/session"; ln -s "$d/circuit.txt" "$t/circuit.txt"; ln -s "$d/$p" "$t/$p"
  ln -s "$d/stage.json" "$t/stage.json"
  for f in pub.bin partition.json; do [ -e "$k/$f" ] && ln -s "$k/$f" "$t/$f"; done
  [ -d "$k/tables" ] && cp -r "$k/tables" "$t/tables"   # rglob descends into no linked directory
  st=$("${LEAN_PY[@]}" -c 'import json, sys; f = open(sys.argv[1], "rb"); f.readline(); print(json.loads(f.readline()[5:])["statement"])' "$d/circuit.txt")
  "${LEAN_BENCH[@]}" prove lean-verify --lean "$exe" --proof "$t" --statement "$st" --repeat 1 --out "$o" > "$o/stdout.txt" 2>&1
  rc=$?
  "${LEAN_PY[@]}" - "$o/lean-verify.json" "$name" "$sid" "$rc" "$(stat -c %s "$d/circuit.txt")" >> "$tsv" <<'EOF'
import json, sys
from pathlib import Path
p, name, sid, rc, cb = sys.argv[1:]
r = json.loads(Path(p).read_text()) if Path(p).exists() else {}
run = ((r.get("verifier") or {}).get("runs") or [{}])[0]
why = "" if r.get("accepted") else run.get("why") or (run.get("verdict") or {}).get("why", "")
print("\t".join(map(str, (name, sid, r.get("accepted", f"no record (exit {rc})"), r.get("label") or why, run.get("wall_s", ""),
                          run.get("maxrss_gib", ""), cb))))
EOF
  rm -rf "$t"
  [ "$(tail -1 "$tsv" | cut -f3)" = True ]
}
