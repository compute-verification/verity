#!/usr/bin/env bash
# [E6 CURVE 2026-09-13, D54 item 2 / D55] POD-SIDE chain: re-pin SmolLM2-135M (B0) at 4x8, 32x15, 128x63, 256x63 on the CORRECTED attention program
# (E7's `fa2_kblock_n(D)`, dfcea003) in ONE pod session, each size written to /workspace/cv/e6/pinned/<size>/ as it finishes, then the two
# interior-rows identity checks (4x9 / 32x16 rows byte-identical to the old program's, per file), the rule-vs-plan checks at BN 128, and PINS.md.
# The laptop and the agent may die at any moment: guard_trap terminates the pod when the chain exits (any rc); guard_start terminates it after
# 20 idle minutes without a chain process.  Nothing is pulled from here by the chain; the laptop pulls the slim files afterwards.
#
#   launched by cv_repin_launch.sh as:  setsid nohup bash /workspace/cv/e6/cv_repin_chain.sh > /workspace/cv/e6/chain.log 2>&1 &
#
# Layout on the volume (/workspace = h5uk3omqsl):
#   /workspace/cv/e6/tree/                     git archive of eod/e6 (must carry E7's fix: b1.fa2_kblock_n)
#   /workspace/cv/e6/venv/                     uv python 3.12 + numpy/networkx/safetensors
#   /workspace/cv/e6/GEOMETRY.json             {"attention": {"kernel","kBlockM","kBlockN","num_splits"}, "source": "<E10 §A measurement>"}  (launcher --geometry)
#   /workspace/cv/e6/legacy/<size>/             LEGACY-identity regression builds (4x8, 32x15, 128x63): must reproduce the old pins' digests (D57)
#   /workspace/cv/e6/pinned/<size>/pipeline/   plan_digest.json, population.compact.json, population.v5.json   (E6's two-phase rows pipeline)
#   /workspace/cv/e6/pinned/<size>/instance/   protocol_instance.json, population.v2.json, expected_population.json, inventory.{json,md}, pins.json, coverage_table.json
#   /workspace/cv/e6/pinned/<size>/BUILD.log, SHA256SUMS, IDENTITY.json (old pin vs new pin digests, MEASURED)
#   /workspace/cv/e6/rows/<size>/               interior rows under the new program (e2e.interior.producer) + ROWS_IDENTITY.json
#   /workspace/cv/e6/pinned/rule_checks/        rule_vs_plan at two-block shapes (BN 128)
#   /workspace/cv/e6/pinned/PINS.md, program_digests.json, bounds/
set -uo pipefail
export GUARD_WHO=cv/e6
source /root/dm/pod_guard.sh
guard_start 'cv_repin_chain|serve_rows_pipeline|population_v3|interior.producer|rule_vs_plan|scale_bounds|cv_pins_md' 20
guard_trap

BASE=/workspace/cv/e6
TREE=$BASE/tree
PY=$BASE/venv/bin/python
export PYTHONPATH=$TREE:$TREE/vllm-poc:$TREE/src
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PIN=$BASE/pinned; mkdir -p "$PIN" "$BASE/rows"
LOG=$BASE/chain.log
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*"; }
# peak-RSS wrapper (python, no GNU time on the image): prints "[rss_run <tag>] rc= wall_s= peak_rss_gb=" and exits with the child's rc
rss_run() { local tag=$1; shift; $PY "$TREE/scripts/rss_run.py" "$tag" "$@"; }

cd "$TREE"
say "chain start on pod $(guard_self_id); tree $(cat "$BASE/TREE_HASH" 2>/dev/null); nproc $(nproc); mem $(free -g | awk '/Mem:/{print $2}') GB; cgroup $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"

# 0. [D57] the geometry is a static of the PINNED identity.  Two kinds of build in this chain:
#    LEGACY  (no --attention): the pre-D57 program; on the NEW tree it MUST reproduce the old pins' digests (program, plan-rule, rows, partition)
#             -> /workspace/cv/e6/legacy/<size>/ ; a regression statement for T1's four records, not a new pin;
#    GEOMETRY (--attention from /workspace/cv/e6/GEOMETRY.json, written by the launcher from E10's MEASURED kernel geometry + its provenance):
#             the fresh r2 pins -> /workspace/cv/e6/pinned/<size>/ ; at 4x8/32x15 the rows' v1 digest must EQUAL the legacy one (the transition
#             bites only past one block); at 128x63/256x63 they are the PHASE 2 instances.
#    E7's `b1.serve_program(C, LP, STEPS, family="Serve_v4", attention=...)` (D57, `e65daae52`) must be on the tree for GEOMETRY builds; the legacy
#    section (family Serve_v1, no geometry) runs on any tree.
GEO=$BASE/GEOMETRY.json
HAVE_HOOK=0; $PY -c "from verity_vllm.registry import b1; import sys; sys.exit(0 if hasattr(b1, 'serve_program') and 'Serve_v4' in getattr(b1, 'SERVE_FAMILIES', {}) else 1)" && HAVE_HOOK=1
ATT=""; ATT_SRC=""
if [ -f "$GEO" ]; then
  ATT=$($PY -c "import json; g=json.load(open('$GEO')); print(json.dumps({k: g['attention'][k] for k in ('kernel','kBlockM','kBlockN','num_splits')}, sort_keys=True))")
  ATT_SRC=$($PY -c "import json; print(json.load(open('$GEO'))['source'])")
fi
say "tree hash $(cat "$BASE/TREE_HASH" 2>/dev/null); D57 hook on tree: $HAVE_HOOK; geometry: ${ATT:-none} (source: ${ATT_SRC:-none})"
$PY - "$ATT" <<'PY' > "$PIN/program_digests.json"
import importlib, json, sys
from verity_vllm.registry import b1
from verity_vllm.ir.lazy import LazyProgram, lazy_program_digest
att = json.loads(sys.argv[1]) if sys.argv[1] else None
have = hasattr(b1, "serve_program")
out = {"legacy_kblock_n": 64, "attention": att, "serve_v4_on_tree": have, "points": {}}
for lp, steps in ((4, 8), (32, 15), (128, 63), (256, 63)):
    C = dict(b1.B0_CONFIG)
    legacy = b1.serve_program(C, lp, steps) if have else importlib.import_module("verity.ir").bind(b1.Serve, C=C, LP=lp, STEPS=steps)
    row = {"LP": lp, "STEPS": steps, "legacy_function": legacy.id, "legacy_program_digest": lazy_program_digest(LazyProgram(legacy))}
    if att is not None and have:
        try:
            sp = b1.serve_program(C, lp, steps, family="Serve_v4", attention=att)
            row["geometry_function"] = sp.id; row["geometry_program_digest"] = lazy_program_digest(LazyProgram(sp))
        except Exception as e:  # noqa: BLE001 -- the refusal is the record
            row["geometry_program_digest_refused"] = f"{type(e).__name__}: {e}"
    out["points"][f"{lp}x{steps}"] = row
print(json.dumps(out, indent=1))
PY
say "program digests: $(tr -d '\n ' < "$PIN/program_digests.json" | cut -c1-600)"
# [integrator GO 16:38Z] the guard: the 128x63 geometry program must be E3's measured `6a5bb5c7…` (lazy = full on E7's tree); otherwise STOP before building
EXPECT_128=${EXPECT_128:-6a5bb5c7}
if [ -n "$ATT" ] && [ "$HAVE_HOOK" = 1 ]; then
  G128=$($PY -c "import json; print(json.load(open('$PIN/program_digests.json'))['points']['128x63'].get('geometry_program_digest', 'REFUSED'))")
  case "$G128" in "$EXPECT_128"*) say "GUARD OK: 128x63 geometry program digest $G128 starts with $EXPECT_128";;
    *) say "GUARD FAILED: 128x63 geometry program digest '$G128' does not start with $EXPECT_128 -- STOP before building (report to the integrator)"
       echo "{\"stopped\": \"128x63 geometry digest $G128 != $EXPECT_128*\"}" > "$PIN/STOPPED.json"; sync; guard_finish "cv_repin_chain STOPPED at the digest guard";;
  esac
fi

# 1. rule vs built plan (seconds to minutes each): legacy at the old block, geometry at the declared block -- MEASURED evidence before the big builds
mkdir -p "$PIN/rule_checks"
for shp in 100,32 130,20 4,8 32,15; do IFS=, read -r a b <<< "$shp"
  rss_run "rule_vs_plan_legacy_${a}x${b}" $PY scripts/rule_vs_plan.py B0 "$a" "$b" --json "$PIN/rule_checks/legacy_B0_${a}x${b}.json" || say "rule_vs_plan legacy ${a}x${b} rc=$? (DIFFERS or failed; see json)"
  if [ -n "$ATT" ] && [ "$HAVE_HOOK" = 1 ]; then
    rss_run "rule_vs_plan_geometry_${a}x${b}" $PY scripts/rule_vs_plan.py B0 "$a" "$b" --attention "$ATT" --json "$PIN/rule_checks/geometry_B0_${a}x${b}.json" || say "rule_vs_plan geometry ${a}x${b} rc=$? (DIFFERS or failed; see json)"
  fi
done

# 2. the builds.  OLD = the pre-D57 pins in the tree: instance id, rows' v1 digest, plan-rule digest, program digest, path.
declare -A REQSRC=(
  [4x8]="record_v5/workload_v5 (TINY-B0-4x8) row 0: workload-B0-small.json request 0 truncated to 4 tokens; the auditor's one request (E10 n4); greedy declared"
  [32x15]="declared by E8: out/night/e8/requests.json NIGHT-B0-32x15-1req (E10 workload_v5 rule: workload-B0-small.json request 0, 32 tokens; max_tokens 16)"
  [128x63]="declared by the workload_v5 rule (record_v5/workload_v5.py:39-43): prompt_len > 32 -> synthetic random.Random(index+1).randrange(49152) x prompt_len, request index 0, one request per epoch"
  [256x63]="declared by the workload_v5 rule (record_v5/workload_v5.py:39-43): prompt_len > 32 -> synthetic random.Random(index+1).randrange(49152) x prompt_len, request index 0, one request per epoch"
)
declare -A OLD=(
  [4x8]="E6-TINY-4x8-r1-20a9090d-t84cf6bc1-l57129503 c62a5d8742435062 c4e04cd6134dec5a b33a0463336bc222 out/eod/e6/tiny"
  [32x15]="E6-B0-32x15-r1-77d5b63b-t84cf6bc1-l57129503 d54c4e2634734195 13f706a556aeb82f d4f4b1f1b4d592d6 out/night/e6/b0-32x15"
  [128x63]="E6-B0-128x63-r1-13232aef-t84cf6bc1-l57129503 ee1c3c3f48db8bcd 768c86f872a9d3bf 4bdfe726769e45ed out/scale/e6/pinned/SmolLM2-135M__128x64"
  [256x63]="E6-B0-256x63-r1-f299411e-t84cf6bc1-l57129503 7680b74ae47616cd c686ead2ff40f59c daa2bcc800c4de33 out/scale/e6/pinned/SmolLM2-135M__256x64"
)
prompt_ids() {  # the declared prompt token ids by the workload_v5 rule (prompt_len <= 32: workload-B0-small.json request 0 truncated; else synthetic Random(1))
  $PY - "$1" <<'PY'
import json, random, sys
from pathlib import Path
lp = int(sys.argv[1])
if lp <= 32:
    base = json.load(open(Path("e2e/workload-B0-small.json")))  # record_v5.workload_v5.WORKLOAD_B0
    print(json.dumps(list(base["requests"][0]["prompt_token_ids"])[:lp]))
else:
    rng = random.Random(1); print(json.dumps([rng.randrange(49152) for _ in range(lp)]))
PY
}
build_one() {  # mode(legacy|geometry) size outdir revision
  local mode=$1 size=$2 D=$3 rev=$4; IFS=x read -r LP STEPS <<< "$size"
  mkdir -p "$D/pipeline" "$D/instance"; local BL=$D/BUILD.log
  if [ -f "$D/instance/protocol_instance.json" ]; then say "$mode $size already built -- skipping"; return; fi
  local AARGS=(); local FARGS=(); local srcnote="legacy identity (no geometry field): the pre-D57 program, regression build on tree $(cat "$BASE/TREE_HASH")"
  if [ "$mode" = geometry ]; then AARGS=(--attention "$ATT"); FARGS=(--attention "$ATT" --attention-source "$ATT_SRC"); srcnote="pinned on the corrected attention geometry $ATT (D57; source: $ATT_SRC)"; fi
  {
    echo "== $mode $size plan $(ts)"
    rss_run plan $PY scripts/serve_rows_pipeline.py plan B0 "$LP" "$STEPS" "$D/pipeline" "${AARGS[@]}"; rc=$?; echo "plan rc=$rc $(ts)"
    if [ $rc -eq 0 ]; then
      echo "== $mode $size rows $(ts)"
      rss_run rows $PY scripts/serve_rows_pipeline.py rows B0 "$LP" "$STEPS" "$D/pipeline" "${AARGS[@]}"; rc=$?; echo "rows rc=$rc $(ts)"
    fi
    if [ $rc -eq 0 ]; then
      echo "== $mode $size freeze $(ts)"
      IDS=$(prompt_ids "$LP")
      rss_run freeze $PY -m e2e.population_v3 freeze serve --case B0 --lp "$LP" --steps "$STEPS" --pipeline-dir "$D/pipeline" --out "$D/instance" \
        --literals compiled --token-select whole-scan --revision "$rev" --prompt-token-ids "$IDS" --request-source "${REQSRC[$size]}; $srcnote" "${FARGS[@]}"; rc=$?; echo "freeze rc=$rc $(ts)"
    fi
    if [ $rc -eq 0 ]; then
      (cd "$D" && sha256sum pipeline/plan_digest.json pipeline/population.compact.json pipeline/population.v5.json instance/* > SHA256SUMS)
      read -r OID OROWS OPLAN OPROG OPATH <<< "${OLD[$size]}"
      $PY - "$D/instance/protocol_instance.json" "$OID" "$OROWS" "$OPLAN" "$OPROG" "$TREE/$OPATH/protocol_instance.json" "$mode" "$size" "$BASE/legacy/$size/instance/protocol_instance.json" <<'PY' > "$D/IDENTITY.json"
import json, re, sys
from pathlib import Path
new = json.load(open(sys.argv[1])); oid, orows, oplan, oprog, oldpath, mode, size, legacypath = sys.argv[2:10]
old = json.load(open(oldpath))
rows_of = lambda i: (re.search(r"rows' v1 digest ([0-9a-f]+)", i["partition"]["source"]) or [None, None])[1]
nv = lambda i: (i["partition"].get("vu_count") or {}).get("value") if isinstance(i["partition"].get("vu_count"), dict) else i["partition"].get("vu_count")
nrows, nplan, nprog = rows_of(new), new["partition"]["plan_descriptor_digest"], new["program"]["digest"]
doc = {"size": size, "mode": mode, "attention": new["program"]["identity"].get("attention"),
       "old": {"instance_id": oid, "rows_v1_digest": orows, "plan_rule_digest": oplan, "program_digest": oprog, "N_V": nv(old), "request_ids": old["epoch"].get("request_ids")},
       "new": {"instance_id": new["instance_id"], "rows_v1_digest": nrows, "plan_rule_digest": nplan[:16], "program_digest": nprog[:16], "program_digest_full": nprog, "N_V": nv(new),
               "request_ids": new["epoch"].get("request_ids"), "partition_digest": new["partition"]["partition_digest"]},
       "vs_old_pin": {"rows_identical": nrows == orows, "plan_rule_identical": nplan.startswith(oplan), "program_digest_identical": nprog.startswith(oprog),
                      "partition_digest_identical": new["instance_id"].split("-")[4] == oid.split("-")[4], "request_ids_identical": new["epoch"].get("request_ids") == old["epoch"].get("request_ids"),
                      "N_V_identical": nv(new) == nv(old)}}
if mode == "legacy":
    must = dict(doc["vs_old_pin"])
    if oid.startswith("E6-TINY"):  # the TINY pin was frozen by `freeze tiny` (TINY's own coverage table is a partition member): its partition digest is not the `freeze serve` one by construction
        doc["note"] = "partition_digest not comparable: old pin frozen by `freeze tiny` (different coverage-table member); program / plan-rule / rows / N_V / request are the identity here"
        must.pop("partition_digest_identical")
    doc["verdict"] = "LEGACY REPRODUCES OLD PIN" if all(must.values()) else "LEGACY DOES NOT REPRODUCE OLD PIN: " + json.dumps({k: v for k, v in must.items() if not v})
else:
    lp = Path(legacypath)
    if lp.exists():
        leg = json.load(open(lp))
        doc["vs_legacy_build"] = {"rows_identical": nrows == rows_of(leg), "N_V_identical": nv(new) == nv(leg), "program_digest_changed": nprog != leg["program"]["digest"],
                                  "plan_rule_changed": nplan != leg["partition"]["plan_descriptor_digest"], "legacy_instance_id": leg["instance_id"]}
    doc["verdict"] = ("GEOMETRY ROWS IDENTICAL TO LEGACY" if doc.get("vs_legacy_build", {}).get("rows_identical") else
                      ("GEOMETRY ROWS DIFFER FROM LEGACY (expected past one block)" if "vs_legacy_build" in doc else "no legacy build to compare")) + \
                     ("; rows == old pin" if nrows == orows else "; rows != old pin")
print(json.dumps(doc, indent=1))
PY
      echo "IDENTITY: $(tr -d '\n' < "$D/IDENTITY.json" | tr -s ' ' | cut -c1-700)"
      echo "== $mode $size DONE $(ts)"; du -sh "$D" | cut -f1
    else
      echo "== $mode $size FAILED rc=$rc $(ts)"; echo "{\"size\": \"$size\", \"mode\": \"$mode\", \"failed_rc\": $rc}" > "$D/FAILED.json"
    fi
  } >> "$BL" 2>&1
  say "$mode $size: $(tail -1 "$BL" | cut -c1-300)"
  sync
}
# 2a. LEGACY regression builds (any tree): 4x8, 32x15 (the accepted records' shapes; MUST reproduce), 128x63 (the two-block old pin; cheap)
mkdir -p "$BASE/legacy"
for size in 4x8 32x15 128x63; do build_one legacy "$size" "$BASE/legacy/$size" 1; done
# 2b. GEOMETRY pins (r2): need the hook and GEOMETRY.json; otherwise refused by name, the legacy results stand
if [ -z "$ATT" ]; then say "REFUSED geometry pins: no $GEO (E10's measured geometry not supplied to the launcher)"; echo '{"refused": "no GEOMETRY.json"}' > "$PIN/REFUSED.json"
elif [ "$HAVE_HOOK" != 1 ]; then say "REFUSED geometry pins: tree has no b1.serve_program / Serve_v4 (E7's D57 static)"; echo '{"refused": "tree lacks b1.serve_program / Serve_v4"}' > "$PIN/REFUSED.json"
else
  for size in 4x8 32x15 128x63 256x63; do build_one geometry "$size" "$PIN/$size" 2; done
fi

# 3. (c) interior ROWS regression at 4x9 and 32x16 on the NEW tree: E1's producer over the accepted records (their instances declare the LEGACY
#    identity) must reproduce the records' committed interior rows per file (D57: the old records rebuild unchanged).  The geometry statement at
#    these shapes is the population one (2b: geometry rows' v1 digest == legacy); a producer run WITH the geometry needs E1/E7's flag (not mine).
rows_check() {  # size record request lp steps
  local size=$1 rec=$2 req=$3 lp=$4 steps=$5
  local out=$BASE/rows/$size  # (18:12Z run: `out` expanded the loop's stale $size -> both producers wrote rows/256x63/; and the record's interior files were not found by *.jsonl -- comparator NOT DONE, see e6.md)
  if [ ! -d "$rec" ]; then say "rows $size: record $rec not on the volume -- SKIPPED (named)"; mkdir -p "$out"; echo "{\"size\":\"$size\",\"skipped\":\"record $rec not on the volume\"}" > "$out/ROWS_IDENTITY.json"; return; fi
  local ckpt=/workspace/e1/model.safetensors; [ -f "$ckpt" ] || ckpt=/workspace/e8/ckpt/model.safetensors
  mkdir -p "$out"
  { echo "== rows $size $(ts) record $rec request $req ckpt $ckpt"
    rss_run "rows_$size" $PY -m e2e.interior.producer --record "$rec" --request "$req" --checkpoint "$ckpt" --lp "$lp" --steps "$steps" --only-required --pool 8 --out "$out/produced"; echo "producer rc=$? $(ts)"
    $PY - "$rec" "$out/produced" "$size" "$PIN/program_digests.json" <<'PY' > "$out/ROWS_IDENTITY.json"
import hashlib, json, sys
from pathlib import Path
rec, prod, size, pdig = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
def shas(root):
    files = sorted(p for p in root.rglob("*.jsonl") if "interior" in p.parts)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def digest(m):  # E2's definition: sha256 over sorted (path, sha256(file))
    h = hashlib.sha256()
    for k in sorted(m): h.update(f"{k} {m[k]}\n".encode())
    return h.hexdigest()
a, b = shas(rec), shas(prod)
# the record's interior files may sit under <rec>/interior/... while the producer writes <out>/interior/...: compare by path relative to the first 'interior'
def norm(m): return {k[k.index("interior"):] if "interior" in k else k: v for k, v in m.items()}
a, b = norm(a), norm(b)
common = sorted(set(a) & set(b)); diff = [k for k in common if a[k] != b[k]]
doc = {"size": size, "record": str(rec), "produced": str(prod), "record_files": len(a), "produced_files": len(b), "common": len(common), "differing": len(diff),
       "only_in_record": sorted(set(a) - set(b))[:20], "only_in_produced": sorted(set(b) - set(a))[:20], "first_differing": diff[:20],
       "digest_record": digest(a), "digest_produced": digest(b), "byte_identical": bool(common) and not diff and set(a) == set(b),
       "program_digests": json.load(open(pdig))["points"][{"4x9": "4x8", "32x16": "32x15"}[size]],
       "note": "per-file sha256 over interior/**/*.jsonl; digest = sha256 over sorted 'relpath sha256' lines (E2's rows digest is the same construction over E8's tree; compare per file, not by the top digest, unless the serialisation is confirmed)"}
if prod.joinpath("boundary_compare.json").exists(): doc["boundary_compare"] = json.load(open(prod / "boundary_compare.json"))
print(json.dumps(doc, indent=1))
PY
    echo "ROWS_IDENTITY: $(tr -d '\n' < "$out/ROWS_IDENTITY.json" | tr -s ' ' | cut -c1-500)"
  } >> "$out/ROWS.log" 2>&1
  say "rows $size: $(tail -1 "$out/ROWS.log" | cut -c1-300)"
  rm -rf "$out/produced/interior" 2>/dev/null  # the rows themselves are E1/E2's artefact; keep only the identity verdict + boundary_compare on the volume
}
rows_check 4x9 /workspace/a9/rec/V5-TINY1-bex0-inc-n4 2663944e-cf0f-5349-a481-4beffe89398b 4 8
rows_check 32x16 /workspace/e8/rec-32x15 7cc7c3bb-adf5-58cd-87fc-9f1b8e7c34d6 32 15

# 4. bounds + PINS.md pod-side (the laptop's guardian kills lanes below its disk floor); the same generator runs on the laptop after the pull if it can
mkdir -p "$TREE/out/scale/e6/cv"; ln -sfn "$PIN" "$TREE/out/scale/e6/cv/pinned"
$PY scripts/scale_bounds.py --points scripts/cv_points.json --out "$PIN/bounds" >> "$LOG" 2>&1 || say "scale_bounds rc=$?"
$PY scripts/cv_pins_md.py --pinned "$PIN" --out "$PIN/PINS.md" >> "$LOG" 2>&1 || say "cv_pins_md rc=$?"
# 5. the population-rule contract lane on this tree (the laptop's guardian kills lanes below its disk floor): A1 rules == validated plans
( timeout 1200 $PY -m pytest -q tests/contract/a1_population_v3_test.py tests/poc/test_population_compact.py -x --no-header -p no:cacheprovider 2>&1 | tail -15 ) > "$PIN/tests_a1_population.log" || true
say "tests: $(tail -1 "$PIN/tests_a1_population.log")"
# 6. where E1's chain reads (`sc_curve_chain.sh`: E6DIR=/vol/e6/pinned/SmolLM2-135M__<P>/instance): the OLD pins stay untouched there; the r2 pins are
#    reachable as /vol/e6/pinned/SmolLM2-135M__<P>-r2/instance (relative symlinks into /vol/cv/e6/pinned/<size>; E1 sets E6DIR to the -r2 path)
for pair in 128x63:128x64 256x63:256x64 32x15:32x16 4x8:4x9; do IFS=: read -r size pt <<< "$pair"
  if [ -f "$PIN/$size/instance/protocol_instance.json" ]; then mkdir -p /workspace/e6/pinned; ln -sfn "../../cv/e6/pinned/$size" "/workspace/e6/pinned/SmolLM2-135M__${pt}-r2"; fi
done
ls -la /workspace/e6/pinned/ 2>/dev/null | rg -- "-r2" >> "$LOG" || true
say "chain DONE; $(ls "$PIN")"
sync
guard_finish "cv_repin_chain finished"
