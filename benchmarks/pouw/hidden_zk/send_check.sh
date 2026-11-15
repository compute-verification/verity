#!/usr/bin/env bash
# The ZK server's send check (FC_FIREWALL=1, the Lean flock-firewall's session loop) against the host's witness alone, per proved
# witness bit. Per SC_PLAN entry ROLE:K:N (job.sh's staging, N instances, all sampled), SC_REPS rounds of one loopback --zk CPU
# session per arm, the arms' order rotated each round:
#   nofw  FC_HOST_WITNESS=1
#   fw    + FC_FIREWALL=1 through this tree's flock-firewall (beside job.sh's flock-verify, built if missing)
#   fw-b  the same through SC_FIREWALL_B, another build of flock-firewall, when it is set
# Each ZK server process runs under bash's `time` in a subshell that outlives the prover's kill of its child, so its own CPU
# seconds and wall are recorded. Run from the source tree's root (`research run --cwd source`, 16 CPUs, 48 GB):
#   SC_PLAN="hidden:256:1 hidden:256:128 hidden:256:256" SC_REPS=3 bash benchmarks/pouw/hidden_zk/send_check.sh
# Results in $RESEARCH_RUN_DIR/out: send_check.tsv (a row a session: m, k_log, the session's wall, the verdict, the frame
# stops, the prover's phase seconds, the ZK server processes' CPU seconds and the session process's wall) and send_check.json
# (per config and arm the medians, and per ZK server arm the least-squares seconds a proved bit, 2^m, over the configs: the
# ZK server's CPU and the session's wall over nofw's).
set -uo pipefail
SRC=$(pwd)
PLAN=${SC_PLAN:-"hidden:256:1 hidden:256:128 hidden:256:256"}
REPS=${SC_REPS:-3}
export HZ_CONFIGS=x HZ_MEM_GB=${HZ_MEM_GB:-48} HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=job.sh
source "$SRC/benchmarks/pouw/hidden_zk/job.sh"
setup
FW=${LEAN%/flock-verify}/flock-firewall
if [ ! -x "$FW" ]; then
  command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
  (flock 8; [ -x "$FW" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) 8> "${LEAN%/.lake/*}.firewall.lock" \
    > "$OUT/build-firewall.log" 2>&1
fi
[ -x "$FW" ] || { log "missing: $FW"; exit 1; }
declare -A REAL=([fw]=$FW)
ARMS=(nofw fw)
if [ -n "${SC_FIREWALL_B:-}" ]; then
  cp "$SC_FIREWALL_B" "$W/fw-b" && REAL[fw-b]=$W/fw-b && ARMS+=(fw-b)
fi
sha256sum "${REAL[@]}" | tee -a "$OUT/job.log"
cat > "$W/fwtime.sh" <<'WRAP'
#!/usr/bin/env bash
f=$FWT_DIR/fwt.$$.$1
echo "$$ $1" >> "$FWT_DIR/starts"
( TIMEFORMAT='%R %U %S'; { time "$FWT_REAL" "$@" 2>&3 ; } 3>&2 2>"$f.tmp"; mv "$f.tmp" "$f" ) <&0 &
wait
WRAP
chmod +x "$W/fwtime.sh"
export RUST_MIN_STACK=$((64 << 20))
TIMEFORMAT='%R %U %S'
TSV=$OUT/send_check.tsv
printf 'config\tm\tk_log\trep\tarm\twall_s\taccepted\tframe_stops\tt_prove\tt_upstream\tt_zk_inner\tfw_n\tfw_cpu_s\tfw_session_wall_s\n' > "$TSV"
for p in $PLAN; do
  role=${p%%:*}; rest=${p#*:}; k=${rest%%:*}; N=${rest#*:}
  tag=$role-k$k-n$N; D=$W/$tag; O=$OUT/$tag
  mkdir -p "$D" "$O"
  stage_config "$role" "$k" "$role-k$k" "$D" "$O" || { log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; continue; }
  kl=$("$PY" -c "import json; print(json.load(open('$D/stage.json'))['k_log'])")
  pin=$(cat "$D/pin.txt")
  for rep in $(seq 1 "$REPS"); do
    r=$(( (rep - 1) % ${#ARMS[@]} ))
    for arm in "${ARMS[@]:$r}" "${ARMS[@]:0:$r}"; do
      port=$(free_port)
      "$BIN" serve --zk --listen 127.0.0.1:$port --out "$D/sessions-$arm-$rep" --instances "$D/pub.bin" \
        --circuit "$D/circuit.txt" --pin "$pin" --operator loopback --draw "subset:$N" > "$O/serve-$arm-$rep.log" 2>&1 &
      srv=$!
      until grep -q '^SERVING ' "$O/serve-$arm-$rep.log" 2>/dev/null; do kill -0 $srv 2>/dev/null || break; sleep 1; done
      T=$O/fwt-$arm-$rep; mkdir -p "$T"
      env=(FC_HOST_WITNESS=1)
      [ "$arm" = nofw ] || env+=(FC_FIREWALL=1 FC_FIREWALL_LEAN=$W/fwtime.sh FWT_DIR=$T FWT_REAL=${REAL[$arm]})
      { time env "${env[@]}" "$BIN" prove --zk --verifier 127.0.0.1:$port --instances "$D/inst.bin" \
          --circuit "$D/circuit.txt" --runs 1 --warm 0 > "$O/prove-$arm-$rep.out" 2> "$O/prove-$arm-$rep.err" ; } \
        2> "$O/prove-$arm-$rep.time"
      rc=$?
      kill $srv 2>/dev/null; wait $srv 2>/dev/null
      rm -rf "$D/sessions-$arm-$rep"
      # each ZK server process's record, once it has seen its stdin close
      for _ in $(seq 1 240); do
        [ "$(cat "$T/starts" 2>/dev/null | wc -l)" -le "$(ls "$T" | grep -c '^fwt\.[0-9]*\.[a-z]*$')" ] && break; sleep 0.5
      done
      read -r wall _ _ < "$O/prove-$arm-$rep.time"
      m=$(grep -o '"m":[0-9]*' "$O/prove-$arm-$rep.err" | head -1 | cut -d: -f2)
      "$PY" - "$O/prove-$arm-$rep.out" "$O/prove-$arm-$rep.err" "$T" "$tag" "${m:--}" "$kl" "$rep" "$arm" "$wall" >> "$TSV" <<'EOF'
import glob, json, sys
out, err, t, *head = sys.argv[1:]
live = next((json.loads(l.split("\t", 1)[1]) for l in open(out) if l.startswith("LIVE\t")), None)
b = (live or {}).get("buckets", [{}])[0]
stops = sum(l.startswith("frames: ") for l in open(err, errors="replace"))
n = cpu = sw = 0
for f in glob.glob(t + "/fwt.*"):
    if f.endswith(".tmp"):
        continue
    r, u, s = map(float, open(f).read().split()[-3:])
    n, cpu = n + 1, cpu + u + s
    sw += r if f.endswith(".session") else 0
row = head + [(live or {}).get("accepted"), stops, b.get("t.prove"), b.get("t.upstream"), b.get("t.zk_inner"), n,
              round(cpu, 3) if n else "", round(sw, 3) if n else ""]
print("\t".join("" if v is None else str(v) for v in row))
EOF
      log "$tag rep $rep $arm: exit $rc, m ${m:--}, wall $wall, $(tail -1 "$TSV" | cut -f7,13 | tr '\t' ' ')"
    done
  done
  rm -rf "$D"
done
"$PY" - "$TSV" "$OUT/send_check.json" <<'EOF' | tee -a "$OUT/job.log"
import collections, csv, json, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
by = collections.defaultdict(list)
for r in rows:
    by[r["config"], r["arm"]].append(r)
med = lambda rs, k: statistics.median(float(r[k]) for r in rs) if rs and all(r[k] for r in rs) else None
configs = list(dict.fromkeys(r["config"] for r in rows))
table = {}
for c in configs:
    m = next((int(r["m"]) for r in rows if r["config"] == c and r["m"] not in ("", "-")), None)
    table[c] = {"m": m, "arms": {a: {"n": len(rs), "accepted": all(r["accepted"] == "True" for r in rs),
                                     "wall_s": med(rs, "wall_s"), "fw_cpu_s": med(rs, "fw_cpu_s")}
                                 for (cc, a), rs in by.items() if cc == c}}


def fit(xs, ys):
    """Least squares y = a + b x: (a, b)."""
    if len(xs) < 2:
        return None, None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


fits = {}
for a in sorted({r["arm"] for r in rows} - {"nofw"}):
    pts = [(2 ** t["m"], t["arms"][a], t["arms"].get("nofw")) for t in table.values() if t["m"] and a in t["arms"]]
    cpu = fit([x for x, s, _ in pts if s["fw_cpu_s"] is not None], [s["fw_cpu_s"] for _, s, _ in pts if s["fw_cpu_s"] is not None])
    inc = [(x, s["wall_s"] - z["wall_s"]) for x, s, z in pts if z and s["wall_s"] is not None and z["wall_s"] is not None]
    wall = fit([x for x, _ in inc], [y for _, y in inc])
    fits[a] = {"fw_cpu_s": {"at_0": cpu[0], "us_a_bit": cpu[1] and cpu[1] * 1e6},
               "wall_over_nofw_s": {"at_0": wall[0], "us_a_bit": wall[1] and wall[1] * 1e6}}
ok = bool(rows) and all(r["accepted"] == "True" and r["frame_stops"] == "0" for r in rows)
json.dump({"ok": ok, "configs": table, "fits": fits}, open(sys.argv[2], "w"), indent=1)
for c, t in table.items():
    print(c, "m", t["m"], {a: (s["wall_s"], s["fw_cpu_s"]) for a, s in t["arms"].items()})
print("fits", json.dumps(fits))
print("send-check:", "every session accepted, no frame stopped a rep" if ok else "NOT every session accepted")
sys.exit(0 if ok else 1)
EOF
