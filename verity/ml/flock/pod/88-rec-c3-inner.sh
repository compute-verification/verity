#!/usr/bin/env bash
# Candidate 3's inner session (hiding model architectures, milestone A; verity_flock.rec_session): RoPE d64 staged as M0's
# statement with plain SHA-512 leaves (rec_inner.py stage --plain-leaves), proved on the CPU by the unmodified prover
# (non-zk, under INNER_COINS) through the relay, which gives the verifier h, the hidden circuit `unit`'s share of the
# lincheck's fold, in a round of its own after z_partial's, on the rep's one coin stream. Every coin of the session, the
# relay's sparse and stacked phases' too, is the challenger's coin tree, drawn at Hello and opened round by round
# (rec_session --coin-tree, from TREE_FROM). Then: the schedule serve held the session to against rec_live --schedule-for
# --holo's; upstream's replay of the session as the prover ran it (inner/prover); V*'s hashing reference (rec_vstar) over the
# verifier's record; the verifier's fold points (rec_vstage points) and InnerFold.lean --hide unit on the tree (an executable,
# cached by its sources): each rep's public comb and Lean's full fold; rec_session --check-fold (comb + the session's h is the
# full fold, h nonzero); rec_vstage stage --levels none (the holo algebra's InnerRepCheck parts, h read from the session's
# rows, each holding); and Lean's reading of the session (flock-verify rec-transcript: every stream of the verifier's record,
# held to the schedule and to the prover's record, R's top to its registration with STACKED), which test_lean_rec_inner_streams
# runs on the fixture. Lean's plain verify reads seeded or per-round OS coins, not the tree, so the prover's session is not
# verified on its own here (pod 85's INNER_LEAN likewise), nor are the controls on its sample and challenges. The coin server
# answers Register with the verifier's full draw (stage/draw.json, rec_live.full_draw: every unit), which both records keep.
# Run from the verity tree (research run --cwd source). Outputs under out/: host.txt, stage.json, statement.txt,
# schedule.json, tree-source.txt, session.log, time-prove.txt, prove.txt/.err, inner/ (schedule.json, public.json,
# private.json, session/, prover/), replay.txt, vstar.json, points.json, fold.json, check-fold.json, vstage.json, lean.log,
# summary.json, and rec-c3-inner/ (the fixture test_lean_rec_inner_streams and test_lean_rec_record read: public.json,
# private.json, session/ and prover/'s session.json, points.json, fold.json, circuit-meta.txt, the staged circuit's header,
# and with STACKED circuit-registration.json). Exits 1 unless the session, replay, rec_vstar, the fold check, the staging and
# Lean's reading all pass.
# Launch it through a check slot, which pins it to the slot's cores and holds the slot:
#   research run --on vy-cpu-1 --source . --cwd source -- python3 tools/verity/check/slot.py --by NAME -- bash $0 [N=16]
# env: N=16 (RoPE heads, one VU each: Shape(25,22,8))  FLOCK_WORK (as 80-rec-inner.sh)
#   TREE_FROM= (the challenger's device, HOST:PORT, the session's coin tree drawn and opened there, rec_session --tree-from;
#   TREE_SECRET_FILE= its session secret, hex; empty: this host's OS draws it, rec_session --tree-test-os, a test's coins,
#   never the challenger's, out/tree-source.txt says which)
#   SPARSE=NU,ELL,S_LOG,T_LOG (18,13,6,0 at N=16): the relay also runs each rep's sparse phase (rec_sparse.prove) after both
#   reps, on the coin server's coins, and rec_vstage stages it as RecSparse (summary.json's sparse; required to hold)
#   STACKED=1 (with SPARSE): the relay also commits the stacked table R ($B stacked --root-out) and opens it at both reps'
#   sparse claims (--points, after the sparse streams, into the same session); rec_vstar checks R's openings and rec_vstage
#   stages R's levels (RecOpen, on rec-acc) and its registered opening (summary.json's stacked; required to hold). R's top
#   is registered at staging (rec_inner.py stage --stacked): the entry `stacked` of the staged statement's circuit
#   registration (stage/circuit-registration.json, verity/registration/v2), whose digest META's `registration` binds; its
#   salts are the prover's (stage/circuit-registration-private.json)
#   FORGE=cell|point (with STACKED): R committed (and registered) with a flipped cell, or a claim point off; the algebra is not
#   staged, and the run passes when the session runs and rec_vstage refuses R's opening (summary.json's refused). FORGE=late:
#   R's cell flipped after its registration, which the ZK server refuses at R's binding round (the session stops)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
export RI_PINNED=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
export RAYON_NUM_THREADS=$(echo $RI_PINNED | tr ',' '\n' | wc -l)
export FC_COINS=os
. verity/ml/flock/pod/rec-env.sh
# the inner statement's prover, statement and replay: on the challenger's coin tree (rec_live.INNER_COIN_SPEC: the prover's
# own streams, then the relay's sparse and stacked ones), as pod 85's
INNER_COINS="FC_COINS=tree FC_COIN_STREAMS=sparse/rep0,sparse/rep1,stacked/rep0"
TREE=(--tree-test-os)
[ -z "${TREE_FROM:-}" ] || TREE=(--tree-from $TREE_FROM ${TREE_SECRET_FILE:+--tree-secret-file $TREE_SECRET_FILE})
echo "${TREE_FROM:-test-os}" > $O/tree-source.txt

N=${N:-16}; D=$S/stage; V=$S/v; SPA=${SPARSE:+--sparse $SPARSE}; STK=; MR=; REG=
if [ -n "${STACKED:-}" ]; then
  MR=$(python3 -c "from verity_flock import rec_sparse as S; print(S.table_m(S.Shape($SPARSE)))") || exit 1
  STK="--stacked $B ${FORGE:+--forge $FORGE}"
  REG="$SPA --stacked $B"; [ "${FORGE:-}" = cell ] && REG="$REG --forge cell"
fi
FSHA=$( (echo $LSHA; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
FOLD_BIN=$L/fold-$FSHA/inner-fold
(
  export PATH=$HOME/.elan/bin:$PATH
  exec 9>$L/fold.lock; flock 9
  if [ ! -x $FOLD_BIN ]; then
    C=$L/fold-src; rm -rf $C && mkdir -p $C && cp -r verity/core/service/flock/. $C/ && cp verity/ml/flock/tests/lean/InnerFold.lean $C/ \
      && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/lakefile.toml \
      && (cd $C && lake build inner-fold) && mkdir -p $L/fold-$FSHA && cp $C/.lake/build/bin/inner-fold $FOLD_BIN
  fi
) > $O/build-fold.log 2>&1 &
FP=$!
python3 verity/ml/flock/pod/rec_inner.py stage --n $N --sets $S/sets --out $D --plain-leaves $REG > $O/stage.json 2> $O/stage.err \
  || { tail -20 $O/stage.err; exit 1; }
cat $O/stage.json
[ -n "${STACKED:-}" ] && cp $D/circuit-registration.json $O/

env $INNER_COINS $B statement --instances $D/pub-$N.bin --circuit $D/circuit.txt > $O/statement.txt 2>&1 || { tail -5 $O/statement.txt; exit 1; }
python3 -m verity_flock.rec_live --schedule-for $O/statement.txt --holo $SPA ${MR:+--stacked $MR} > $O/schedule.json || exit 1
python3 -m verity_flock.rec_session --listen 127.0.0.1:0 --out $O/inner --statement $O/statement.txt --circuit $D/circuit.txt \
  --draw $D/draw.json $SPA $STK --coin-tree "${TREE[@]}" > $O/session.log 2>&1 &
PP=$!
ADDR=
for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $O/session.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
[ -n "$ADDR" ] || { cat $O/session.log; exit 1; }
/usr/bin/time -v -o $O/time-prove.txt env $INNER_COINS $B prove --verifier $ADDR --instances $D/inst-$N.bin --circuit $D/circuit.txt \
  --runs 1 --warm 0 > $O/prove.txt 2> $O/prove.err
echo "prove exit $?"
wait $PP; echo "session exit $?"
grep -h panicked $O/prove.err | head -3
[ -f $O/inner/prover/session.json ] || { tail -20 $O/session.log $O/prove.err; exit 1; }
env $INNER_COINS $B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt --sessions $O/inner/prover > $O/replay.txt 2>&1
echo "replay exit $?"
python3 -m verity_flock.rec_vstar $O/inner --circuit $D/circuit.txt > $O/vstar.json 2> $O/vstar.err; echo "vstar exit $?"

mkdir -p $V
python3 -m verity_flock.rec_vstage points $O/inner $D/circuit.txt $D/pub-$N.bin > $O/points.json 2> $O/points.err \
  || { tail -20 $O/points.err; exit 1; }
wait $FP || { tail -30 $O/build-fold.log; exit 1; }
/usr/bin/time -v -o $O/time-fold.txt $FOLD_BIN --plain-leaves --partition $D/partition.json --hide unit verity/flock-circuit $D/circuit.txt $D/pub-$N.bin \
  $O/points.json tree > $O/fold.json 2> $O/fold.err
echo "fold exit $?"
python3 -m verity_flock.rec_session --check-fold $O/fold.json --inner $O/inner > $O/check-fold.json; echo "check-fold exit $?"
/usr/bin/time -v -o $O/time-vstage.txt python3 -m verity_flock.rec_vstage stage $O/inner $O/fold.json $V --points $O/points.json \
  --levels none $SPA ${SPARSE:+--circuit $D/circuit.txt} ${FORGE:+--no-algebra} > $O/vstage.json 2> $O/vstage.err
echo "vstage exit $?"
(cd $V && du -sb $(find . -name '*.bin' -o -name circuit.txt | LC_ALL=C sort)) > $O/vstage-sizes.txt 2>&1

# the fixture, laid out as its readers take it, and Lean's reading of it (test_lean_rec_inner_streams' _read)
X=$O/rec-c3-inner; rm -rf $X; mkdir -p $X/session $X/prover
cp $O/inner/public.json $O/inner/private.json $O/points.json $O/fold.json $X/
cp $O/inner/session/session.json $X/session/; cp $O/inner/prover/session.json $X/prover/
head -2 $D/circuit.txt > $X/circuit-meta.txt
[ -n "${STACKED:-}" ] && cp $D/circuit-registration.json $X/
wait $LP
$LEAN_BIN rec-transcript --rec-inner-transcript $X/public.json --rec-inner-session $X/prover \
  ${STACKED:+--rec-inner-circuit $X/circuit-meta.txt --rec-inner-registration $X/circuit-registration.json} > $O/lean.log 2>&1
echo "lean exit $?" | tee $O/lean-exit.txt

python3 - $O <<'EOF'
import json, os, re, sys
from pathlib import Path
from verity_flock import rec_live as RL
o = Path(sys.argv[1])
text = lambda f: (o / f).read_text() if (o / f).exists() else ""
js = lambda f: json.loads(text(f).splitlines()[0]) if text(f).strip() else None
up = next((json.loads(l.split("\t", 1)[1]) for l in text("replay.txt").splitlines() if l.startswith("UPSTREAM")), None)
t = text("time-prove.txt")
live = [json.loads(l[5:]) for l in text("prove.txt").splitlines() if l.startswith("LIVE")]
served = [json.loads(l.split("\t", 1)[1]) for l in text("session.log").splitlines() if l.startswith("REC_INNER\t")]
pub = json.loads(text("inner/public.json"))
stage = json.loads(text("stage.json"))
zp = RL.zp_round(stage["m"], stage["k_log"])
reps = {}
for r in pub["rounds"]:
    if r["stream"].startswith(f"{RL.TABLE}/rep") and r["k"] in (zp, zp + 1):
        reps.setdefault(r["stream"], {})["z_partial" if r["k"] == zp else "h"] = {"k": r["k"], "n": r["n"], "commits": len(r["commits"])}
vs = js("vstage.json") or {}
read = next((json.loads(l[len("RECTRANSCRIPT "):]) for l in text("lean.log").splitlines() if l.startswith("RECTRANSCRIPT ")), None)
tree = json.loads(text("inner/session/session.json") or "{}").get("coin_tree")
s = {"stage": stage, "tree_source": text("tree-source.txt").strip(), "coin_tree": tree and {k: tree[k] for k in ("root", "spec")},
     "served": served[0] if served else text("session.log")[-1500:],
     "schedule_is_the_statements": json.loads(text("schedule.json")) == json.loads(text("inner/schedule.json") or "null"),
     "replay": up["verdict"] if up else text("replay.txt")[-600:], "vstar": js("vstar.json") or text("vstar.err")[-600:],
     "record": {"rounds": len(pub["rounds"]), "message_rows": pub["message_rows"], "coins": sum(r["n"] for r in pub["rounds"]),
                "coinless": [[r["stream"], r["k"]] for r in pub["rounds"] if not r["n"]], "streams": len(pub["streams"]), "reps": reps},
     "fold": {k: v for k, v in (js("fold.json") or {}).items() if k in ("m", "k_log", "hidden")} or text("fold.err")[-600:],
     "check_fold": js("check-fold.json"),
     "vstage": {k: vs.get(k) for k in ("shape", "holo", "openings", "chain_rows")} | {
         "algebra": [{k: p.get(k) for k in ("part", "holds", "unit_agrees", "instances", "of_residuals")} for p in vs.get("algebra", [])],
         "sparse": vs.get("sparse") and {k: vs["sparse"].get(k) for k in ("template", "unit", "instances", "ands", "holds", "unit_agrees",
                                                                         "nonzero_residuals")},
         "stacked": vs.get("stacked") and {"m": vs["stacked"]["m"], "openings": vs["stacked"]["openings"],
                                           "levels": {k: {x: v.get(x) for x in ("instances", "opened", "not_opened", "summed", "not_summed", "H", "LANES")}
                                                      for k, v in vs["stacked"]["levels"].items()},
                                           "opening": [{x: p.get(x) for x in ("part", "holds", "nonzero_residuals", "unit_agrees", "unit_rows")}
                                                       for p in vs["stacked"]["opening"]], "stage_s": vs["stacked"].get("stage_s")}}
     if vs else text("vstage.err")[-1500:], "vstage_sizes": text("vstage-sizes.txt"),
     "lean": {"exit": text("lean-exit.txt").strip(), "transcript": read, "tail": text("lean.log")[-400:]},
     "prover_peak_gb": int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 2**20 if t else None,
     "prove_wall": (re.search(r"Elapsed \(wall clock\) time.*: (\S+)", t) or [None, None])[1] if t else None,
     "fold_wall": (re.search(r"Elapsed \(wall clock\) time.*: (\S+)", text("time-fold.txt")) or [None, None])[1],
     "live": {k: live[0].get(k) for k in ("prove_s", "prove_total_s", "e2e_s", "proof_bytes", "k_log", "dense_m")} if live else None}
ok = {"session": bool(served) and bool(served[0].get("session")), "tree": bool(tree), "schedule": s["schedule_is_the_statements"],
      "replay": bool(up) and up["verdict"].get("accepted") is True, "vstar": isinstance(s["vstar"], dict) and s["vstar"].get("accepted") is True,
      "check_fold": bool(s["check_fold"]) and all(s["check_fold"]["comb_plus_h_is_full"]) and all(s["check_fold"]["h_nonzero"]),
      "vstage": bool(vs) and vs.get("holo") is True and bool(vs.get("algebra")) and all(all(p["holds"]) for p in vs["algebra"]),
      "lean": text("lean-exit.txt").strip() == "lean exit 0" and read is not None}
st = vs.get("stacked") if vs else None
if os.environ.get("STACKED"):
    ok["vstar"] = ok["vstar"] and s["vstar"].get("stacked", {}).get("accepted") is True
    held = bool(st) and all(not v["not_opened"] and not v["not_summed"] for v in st["levels"].values())
    if os.environ.get("FORGE"):
        del ok["vstage"]
        ok["refused"] = held and not all(all(p["holds"]) for p in st["opening"])
    else:
        ok["stacked"] = held and all(all(p["holds"]) for p in st["opening"])
if os.environ.get("SPARSE"):
    ok["sparse"] = bool(served) and (served[0].get("sparse") or {}).get("reps") == RL.REPS and bool(vs.get("sparse")) \
        and vs["sparse"]["holds"] == [True] * RL.REPS
s["ok"] = ok
(o / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
sys.exit(0 if all(ok.values()) else 1)
EOF
