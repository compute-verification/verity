sed 's/extract 0 3/extract 0 25/' "$CLONE/tools/verity/lean/Replay.lean" > /tmp/ReplayTop.lean
mods=$(cd .lake/build/lib/lean && find Proofs -name '*.olean' | sed 's/\.olean$//' | while read m; do [ -f "../${m}.lean" ] && echo "$m"; done | sed 's#/#.#g' | sort)
echo "modules: $(echo $mods | wc -w)"
/usr/bin/time -f "replay %e s" lake env lean --run /tmp/ReplayTop.lean /tmp/replay-top.json $mods 2>&1 | tail -5
