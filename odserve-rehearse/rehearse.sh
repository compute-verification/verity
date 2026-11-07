#!/usr/bin/env bash
# A composite-only rehearsal on a follow-on's kept inputs (fri/keep.sh's KEEP), at the tree this job runs in (cwd, a
# clone), with no new coins: the cases and the composite as served_composite.sh makes them, in a scratch serve dir S
# beside KEEP. S mirrors KEEP/serve by symlinks, but for the drawn member's dir, whose circuit file is restored from its
# session's sha512 archive once that file's SHA-512 is its name and the window summary's pin, and the audit's
# case-v2.json, rewritten so its blobs name S's files (a blob is read by its content, `Core.Blobs.get`); S/private
# starts empty. served_cases.py writes the cases in S/private (vstar-recursive under /usr/bin/time), then composite.py
# composes PRODUCT over them and the certifier piece (S/out/pieces/certifier: KEEP's held case and the run's answer),
# under served_composite.sh's scope and timeout, also timed. $OUT gets only public lines (rehearse.txt): each case's
# protocol and verdict and whether it equals the follow-on's, the composite's NOT_ESTABLISHED and VERDICT lines, its
# public out where written, each Lean answer's public outcome (composite.py's `outcome`) where the leak guard held it,
# the timings, and the job's memory series (mem_series.py, sent beside this). Exit 0 when v2's case accepts and the
# composite's public line is written. With COMPLETED=FILE_SHA512:BODY_SHA512, for a window a5 kept before it pinned the
# completed file (`a5.pin_completed`): S's public-pin.json is KEEP's with the drawn member's ``completed`` entry, once
# the kept public file's SHA-512 and its body's are COMPLETED's, and case-v2.json's ``params.proofs.public`` is that
# file's, as `a5.v2_proof` names it once the pin holds it; RELAID=true|false marks the member's staged entry ``relaid``,
# which a5 records before the draw where its tree does. With LAYOUT=n, for a window a5 kept before it pinned the
# members' layout: each staged entry of S's public-pin.json gets its member's ``n`` from the kept summary.json (keyed as
# `v2_case.layout_of` keys it), and case-v2.json's ``params.proofs.members`` is that layout, ``[key, n]`` in unit order
# (`a5.layout`), held equal to the tree's `v2_case.layout_of` where it has one. With RECORD=1, then the onsite record
# over S/out, the follow-on's OUT as its onsite_record read it (KEEP's held forms over the run's public ones, with this
# job's composite), on the sampled proofs case and S's public-pin.json, its verdicts and its record checks in rehearse.txt.
#   KEEP=DIR V=CLASS_DIR M=DRAWN_MEMBER PIN=SHA512 [COMPLETED=SHA512:SHA512 [RELAID=false]] [LAYOUT=n] [RECORD=1]
#     [PRODUCT=remote-network-no-pouw] [COMPOSITE_TIMEOUT=60m] bash rehearse.sh
set -uo pipefail
: "${KEEP:?the keep dir}" "${V:?the class dir}" "${M:?the drawn member}" "${PIN:?the circuit pin of the drawn member}"
PRODUCT=${PRODUCT:-remote-network-no-pouw}
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
TREE=$PWD
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
say() { echo "$*" | tee -a "$OUT/rehearse.txt"; }
K=$KEEP/serve W=$KEEP/serve/$V/audit/window D0=$(dirname "$KEEP")
[ -f "$KEEP/MANIFEST.sha256" ] && [ -f "$W/case-v2.json" ] || { say "REFUSED: $KEEP holds no kept serve dir"; exit 3; }
for f in "$K/private/out/pieces/certifier/case.json" "$KEEP/run/out/pieces/certifier/answer.json"; do
  [ -f "$f" ] || { say "REFUSED: no $f"; exit 3; }
done
S=$D0/rehearse-${RESEARCH_RUN_ID:-local}
[ ! -e "$S" ] || { say "REFUSED: $S exists"; exit 3; }
umask 077
mkdir -p "$S/private/composite" "$S/out/pieces/certifier" "$S/bin" "$S/$V/audit/window/$M" || exit 3
P=$S/private/composite
say "TREE: $(git rev-parse HEAD); kept inputs $KEEP (MANIFEST.sha256 $(sha256sum < "$KEEP/MANIFEST.sha256" | cut -c1-16)); scratch $S"

A=$W/session-$M/archive/sha512/$PIN
[ -f "$A" ] || { say "REFUSED: session-$M's archive holds no ${PIN:0:16}"; exit 3; }
[ "$(sha512sum < "$A" | cut -d' ' -f1)" = "$PIN" ] || { say "REFUSED: the archived file's SHA-512 isn't its name"; exit 3; }
grep -q "\"pin\": \"$PIN\"" "$W/summary.json" || { say "REFUSED: the window summary names no pin ${PIN:0:16}"; exit 3; }
for f in "$K"/*; do case ${f##*/} in "$V" | private) ;; *) ln -s "$f" "$S/" ;; esac; done
for f in "$K/$V"/*; do [ "${f##*/}" = audit ] || ln -s "$f" "$S/$V/"; done
for f in "$K/$V/audit"/*; do [ "${f##*/}" = window ] || ln -s "$f" "$S/$V/audit/"; done
find "$W" -mindepth 1 -maxdepth 1 ! -name "$M" ! -name case-v2.json -exec ln -s -t "$S/$V/audit/window/" {} + || exit 3
for f in "$W/$M"/*; do ln -s "$f" "$S/$V/audit/window/$M/"; done
cp "$A" "$S/$V/audit/window/$M/circuit.txt" && chmod 400 "$S/$V/audit/window/$M/circuit.txt" || exit 3
say "RECOMPOSED: $M/circuit.txt from session-$M's sha512 archive: its SHA-512 is its name and the window summary's pin ${PIN:0:16}"

(uv sync -q --all-packages --extra torch-cpu) > "$S/uv-sync.log" 2>&1 || { say "FAILED: uv sync (its log in $S)"; exit 1; }
TPY=$TREE/.venv/bin/python
if [ -n "${COMPLETED:-}" ] || [ -n "${LAYOUT:-}" ]; then
  rm "$S/$V/audit/window/public-pin.json" || exit 3
  "$TPY" - "$W" "$S/$V/audit/window" "$M" "${COMPLETED:-}" "${RELAID:-}" "${LAYOUT:-}" <<'EOF' | tee -a "$OUT/rehearse.txt" || { say "FAILED: the rewritten pin"; exit 1; }
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
import a5, v2_case
w, s, m, want, relaid, layout = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
k = int(m[1:])
pin = json.loads((w / a5.PIN).read_text())
if layout:
    assert layout == "n", f"LAYOUT is {layout!r}: n, the field each staged entry gets"
    rows = sorted((e["base"], e["k"], e["n"], e["template"], e.get("key"))
                  for e in json.loads((w / "summary.json").read_text())["members"])
    at, seen = 0, {}
    for base, j, n, template, key in rows:
        assert base == at, f"the window's summary.json members don't tile it at m{j}"
        c = seen[template] = seen.get(template, -1) + 1
        key = key or f"{template}/c{c}"
        e = pin["members"][key]
        assert e["k"] == j and "n" not in e, f"{a5.PIN}'s {key} isn't m{j}, or already has its n"
        e["n"] = n
        at += n
    assert len(rows) == len(pin["members"]), f"{a5.PIN} pins other members than summary.json lists"
    assert [r[1] for r in rows] == sorted(r[1] for r in rows), "the members' k order isn't their unit order"
    print(f"LAYOUT-PIN: each member of {a5.PIN} given its n from the kept summary.json, keyed as v2_case.layout_of keys "
          "it (its key, or <template>/c<j>), the members tiling the window from unit 0 in k order; a rehearsal's "
          "reconstruction of what a5 writes at staging")
if want:
    (pub,) = sorted((w / m).glob("pub-*.bin"))
    f, b = hashlib.sha512(pub.read_bytes()).hexdigest(), hashlib.sha512(a5.body(pub)).hexdigest()
    assert want == f"{f}:{b}", f"the kept {m}/{pub.name} isn't the completed pin COMPLETED names"
else:
    (s / a5.PIN).write_text(json.dumps(pin, indent=1, sort_keys=True))
    sys.exit(0)
(key,) = [key for key, e in pin["members"].items() if e["k"] == k]
staged = pin["members"][key]
assert "completed" not in staged, f"{a5.PIN} already holds {m}'s completed entry"
pin["members"][key]["completed"] = {"file": f, "body": b}
if relaid:
    assert relaid in ("true", "false") and "relaid" not in staged, f"RELAID is {relaid!r}, or {a5.PIN} already says"
    pin["members"][key]["relaid"] = relaid == "true"
    print(f"RELAID: {m}'s staged entry marked relaid {relaid}, as RELAID gives it (a5 records `not m.tables` before the draw)")
(s / a5.PIN).write_text(json.dumps(pin, indent=1, sort_keys=True))
print(f"COMPLETED-PIN: {m}'s {pub.name} pinned as completed (file {f[:16]}, body {b[:16]}) as a5.pin_completed writes it, "
      f"beside its staged entry (file {staged['file'][:16]}); its body {'equals' if b == staged['body'] else 'differs from'} "
      "the staged body")
assert v2_case.pinned(s, pub, k) == f, "v2_case.pinned doesn't read the completed entry"
print(f"PINNED: v2_case.pinned reads {m}'s completed file {f[:16]} from the rewritten {a5.PIN}")
EOF
fi
"$TPY" - "$W/case-v2.json" "$S/$V/audit/window/case-v2.json" "$D0" "$S" "${COMPLETED:-}" "${LAYOUT:-}" \
  "$S/$V/audit/window" <<'EOF' | tee -a "$OUT/rehearse.txt" || { say "FAILED: case-v2.json"; exit 1; }
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
import a5, v2_case
src, dst, old, new, completed, layout, sw = sys.argv[1:8]
case = json.load(open(src))
assert all(b.startswith(old + "/") for b in case["blobs"]), "a blob outside the serve dir"
assert "members" not in case["params"]["proofs"], "the kept case-v2.json already names proofs.members"
case["blobs"] = [new + b[len(old):] for b in case["blobs"]]
if completed:
    case["params"]["proofs"]["public"] = completed.split(":")[0]
if layout:
    pin = json.loads((Path(sw) / a5.PIN).read_text())["members"]
    members = [[key, e["n"]] for key, e in sorted(pin.items(), key=lambda x: x[1]["k"])]
    if hasattr(v2_case, "layout_of"):
        assert members == v2_case.layout_of(Path(sw)), "the pin's layout isn't v2_case.layout_of's"
    case["params"]["proofs"]["members"] = members
    print("CASE-V2-MEMBERS: params.proofs.members the pin's layout ([key, n] in unit order, as a5.layout writes it)"
          + (", equal to the tree's v2_case.layout_of" if hasattr(v2_case, "layout_of") else ""))
json.dump(case, open(dst, "w"), sort_keys=True)
EOF
if [ -n "${COMPLETED:-}" ]; then
  say "CASE-V2: the kept case-v2.json with its blobs under $S and params.proofs.public the completed pin ${COMPLETED:0:16}, as a5's v2_proof names it once pin_completed has run${LAYOUT:+, and proofs.members the pin's layout}; its commitment and evidence unchanged"
else
  say "CASE-V2: the kept case-v2.json with its blobs under $S${LAYOUT:+ and proofs.members the pin's layout}, its commitment and evidence unchanged"
fi

t=$(date +%s)
for b in "verity/core/service/vstar/recursive vstar-recursive" "verity/core/protocols/onsite $PRODUCT"; do
  set -- $b
  (cd "$TREE/$1" && lake build "$2") >> "$P/lean-build.log" 2>&1 || { say "FAILED: lake build $2 in $1 (its log in $P)"; exit 1; }
done
say "BUILT: vstar-recursive and $PRODUCT in $(($(date +%s) - t)) s"
cat > "$S/bin/vstar-recursive" <<EOF
#!/usr/bin/env bash
exec /usr/bin/time -a -o "$S/vstar.time" -f '%e %M %x' "$TREE/verity/core/service/vstar/recursive/.lake/build/bin/vstar-recursive" "\$@"
EOF
chmod 700 "$S/bin/vstar-recursive"

ag=(--from-division)
al=$(set +u -a; . "$K/private/deferred/deferred.env" > /dev/null 2>&1; printf '%s' "${ALPHA:-}")
ga=$(set +u -a; . "$K/private/deferred/deferred.env" > /dev/null 2>&1; printf '%s' "${GAMMA:-}")
[ -n "$al" ] && [ -n "$ga" ] && ag=(--alpha "$al" --gamma "$ga")
t=$(date +%s)
"$TPY" benchmarks/one_stage/served_cases.py --serve-dir "$S" --private "$S/private" "${ag[@]}" \
  --sampled-exe "$S/bin/vstar-recursive" > "$P/served_cases.log" 2>&1
rc=$?
grep -hE '^(CASE|NOT_WRITTEN) \{' "$P/served_cases.log" | "$TPY" -c '
import json, sys
for line in sys.stdin:
    head, _, body = line.partition(" ")
    c = json.loads(body)
    print("COMPOSITE", head, c["protocol"], *([c["verdict"]] if head == "CASE" else []))' | tee -a "$OUT/rehearse.txt"
say "COMPOSITE CASES: served_cases.py exit $rc in $(($(date +%s) - t)) s"
if [ -s "$S/vstar.time" ]; then
  read -r el kib ex < <(tail -n1 "$S/vstar.time")
  say "VSTAR: vstar-recursive sampled check ${el} s, max RSS $((kib / 1024)) MiB, exit $ex"
else
  say "VSTAR: vstar-recursive didn't run"
fi
for c in sampled-proofs-v1 inference-only; do
  for f in case.json answer.json; do
    [ -f "$S/private/$c/$f" ] || continue
    cmp -s "$S/private/$c/$f" "$K/private/$c/$f" && s=equals || s="differs from"
    say "SAME: $c/$f $s the follow-on's"
  done
done

ln -s "$K/private/out/pieces/certifier/case.json" "$KEEP/run/out/pieces/certifier/answer.json" "$S/out/pieces/certifier/"
e=$TREE/verity/core/protocols/onsite/.lake/build/bin/$PRODUCT
cases=()
while IFS= read -r c; do cases+=(--cases "$c"); done < <("$TPY" - "$PRODUCT" "$P" "$S/private/pouw" "$S/private" \
  "$S/out/pieces/certifier" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
import composite as C
product, skips, roots = sys.argv[1], [Path(s) for s in sys.argv[2:4]], [Path(r) for r in sys.argv[4:]]
ids = set(C.COMPONENTS[product])
for f in (f for root in roots if root.is_dir() for f in sorted(root.rglob("case.json"))):
    d = f.parent
    if any(d == s or s in d.parents for s in skips) or not (d / "answer.json").is_file():
        continue
    try:
        protocol = json.loads((d / "answer.json").read_text())["protocol"]
    except (ValueError, KeyError, TypeError):
        continue
    if protocol in ids:
        print(d)
EOF
)
scope=()
command -v systemd-run > /dev/null && systemd-run --user --scope --quiet --collect true > /dev/null 2>&1 \
  && scope=(systemd-run --user --scope --quiet --collect -p "MemoryMax=${COMPOSITE_MEM_GB:-170}G" -p MemorySwapMax=0 --)
say "COMPOSITE-CMD: composite.py $PRODUCT, $((${#cases[@]} / 2)) case dirs, --private-out $P/$PRODUCT/private --public-out $S/out/composite/$PRODUCT --exe $PRODUCT, MemoryMax ${COMPOSITE_MEM_GB:-170}G${scope[*]:+ (scope)}, timeout ${COMPOSITE_TIMEOUT:-60m}"
"${scope[@]}" timeout -k 1m "${COMPOSITE_TIMEOUT:-60m}" /usr/bin/time -o "$S/composite.time" -f '%e %M %x' \
  "$TPY" benchmarks/one_stage/composite.py "$PRODUCT" "${cases[@]}" --private-out "$P/$PRODUCT/private" \
  --public-out "$S/out/composite/$PRODUCT" --exe "$e" > "$P/$PRODUCT.log" 2>&1
r=$?
grep -E '^(NOT_ESTABLISHED|VERDICT) ' "$P/$PRODUCT.log" | sed "s/^/COMPOSITE $PRODUCT: /" | tee -a "$OUT/rehearse.txt"
[ -s "$S/composite.time" ] && read -r el kib ex < <(tail -n1 "$S/composite.time") \
  && say "COMPOSITE-TIME: composite.py with its Lean ${el} s, max RSS $((kib / 1024)) MiB"
pub=0
case $r in
  0 | 1 | 2) [ -f "$S/out/composite/$PRODUCT/composite.json" ] && pub=1 ;;
  65) say "COMPOSITE $PRODUCT: composite.py wrote no public line (exit 65: they share bytes with a case's evidence)" ;;
  *) say "COMPOSITE $PRODUCT: composite.py exit $r, its log in $P" ;;
esac
if [ "$pub" = 1 ]; then
  mkdir -p "$OUT/composite/$PRODUCT" && cp "$S/out/composite/$PRODUCT/verdicts.txt" "$OUT/composite/$PRODUCT/" \
    && (cd "$S/out/composite/$PRODUCT" && sha512sum composite.json && stat -c '%s bytes' composite.json) \
      > "$OUT/composite/$PRODUCT/composite.json.sha512"
  say "PUBLIC: composite.py exit $r, its public line written (out/composite/$PRODUCT: verdicts.txt, and composite.json by SHA-512, which stays in $S)"
elif [ -d "$P/$PRODUCT/private" ]; then
  "$TPY" - "$P/$PRODUCT/private" <<'EOF' | tee -a "$OUT/rehearse.txt"
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
import composite as C
d = Path(sys.argv[1])
for f in sorted((d / "pieces").glob("*.answer.json"), key=lambda p: int(p.name.split(".")[0])) + [d / "answer.json"]:
    line = f.read_text().strip() if f.is_file() else ""
    try:
        id = json.loads(line).get("protocol", "?") if line else "?"
    except (ValueError, AttributeError):
        id = "?"
    print("PRIVATE-OUT", f.name, C.line("ANSWER", C.outcome(id, line or None)))
EOF
fi

rec=1
if [ -n "${RECORD:-}" ]; then
  # S/out as the follow-on's OUT was when served_verdicts.sh's onsite_record read it, before its EXIT trap's publish:
  # KEEP's held forms (serve/private/out) over the run's public ones, with this job's composite and a new record
  "$TPY" - "$S/out" "$K/private/out" "$KEEP/run/out" <<'EOF' || { say "FAILED: S/out"; exit 1; }
import sys
from pathlib import Path
dst, roots = Path(sys.argv[1]), [Path(r) for r in sys.argv[2:]]
for root in roots:
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if not f.is_file() or rel.parts[0] == "composite" or rel == Path("record/onsite.json"):
            continue
        d = dst / rel
        if not (d.exists() or d.is_symlink()):
            d.parent.mkdir(parents=True, exist_ok=True)
            d.symlink_to(f)
EOF
  rm -f "$S/out/record/onsite.json"
  pol=()
  [ -f "$S/private/inference-only/policy.json" ] && pol=(--inference-only-case-policy "$S/private/inference-only/policy.json")
  t=$(date +%s)
  (set +u -a; . "$KEEP/serve/private/deferred/deferred.env" > /dev/null 2>&1; set +a
   exec "$TPY" benchmarks/one_stage/onsite_record.py "$S/out" --window-steps "$WINDOW_STEPS" --max-tokens "$MAX_TOKENS" \
     --certifier-key "$KEEP/run/certifier-key.json" --site-registration "$KEEP/run/site-registration.json" "${pol[@]}" \
     --sampled-proofs-case "$S/private/sampled-proofs/case.json" --public-pin "$S/$V/audit/window/public-pin.json" \
     ${H1_TIE:+--h1-tie} --to "$S/out/record/onsite.json") > "$P/onsite_record.log" 2>&1
  rr=$?
  say "RECORD-CMD: onsite_record.py S/out --sampled-proofs-case S/private/sampled-proofs/case.json --public-pin S/.../public-pin.json (and the run's certifier key, site registration, inference-only policy and settings), exit $rr in $(($(date +%s) - t)) s"
  if [ "$rr" = 0 ] && [ -f "$S/out/record/onsite.json" ]; then
    "$TPY" - "$S/out/record/onsite.json" <<'EOF' | tee -a "$OUT/rehearse.txt"
import json, sys
rec = json.load(open(sys.argv[1]))
print("RECORD-VERDICTS:", json.dumps({k: v.get("verdict") for k, v in rec["verdicts"].items()}, sort_keys=True))
o = rec.get("of_record") or {}
print("RECORD-OF-RECORD:", json.dumps({k: o.get(k) for k in ("holds", "composite", "verified_compute")}, sort_keys=True))
listing = (rec.get("checked") or {}).get("auditor-listing") or {}
files = listing.get("registered_files") or {}
print("RECORD-PINNED: auditor-listing's registered_files", json.dumps({k: files.get(k) for k in ("holds", "why", "compared", "read_by")}, sort_keys=True),
      "; auditor-listing holds:", listing.get("holds"))
for name, x in sorted((rec.get("checked") or {}).items()):
    print("RECORD-CHECKED:", name, json.dumps({"holds": x.get("holds"), "why": str(x.get("why") or "")[:300]}, sort_keys=True))
for name, x in sorted(rec["verdicts"].items()):
    if x.get("open") or x.get("conditional_on"):
        print("RECORD-OPEN:", name, json.dumps({k: x.get(k) for k in ("verdict", "answer", "open", "conditional_on")}, sort_keys=True))
print("RECORD-CONDITIONAL:", json.dumps(o.get("conditional_on")), "; not established:", json.dumps(o.get("not_established")))
EOF
    grep -q '^RECORD-PINNED: auditor-listing.s registered_files {"compared": \["circuit", "public"\], "holds": true' "$OUT/rehearse.txt" || rec=0
  else
    say "RECORD: not written: $(grep -m1 '^ONSITE: not written' "$P/onsite_record.log" | cut -c1-300)"
    rec=0
  fi
fi
[ -f "$RESEARCH_RUN_DIR/inputs/mem_series.py" ] \
  && "$TPY" "$RESEARCH_RUN_DIR/inputs/mem_series.py" "$RESEARCH_RUN_DIR" "$OUT/mem-series.tsv" | tee -a "$OUT/rehearse.txt"
grep -q '^COMPOSITE CASE verity/sampled-proofs/v1 accept$' "$OUT/rehearse.txt" && [ "$pub" = 1 ] && [ "$rec" = 1 ]
