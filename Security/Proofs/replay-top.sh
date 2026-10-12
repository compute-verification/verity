sed 's/extract 0 3/extract 0 25/' "$CLONE/tools/verity/lean/Replay.lean" > /tmp/ReplayTop.lean
mods=$(find . -name '*.lean' -not -path './.lake/*' -not -name 'lakefile.lean' -not -name 'Probe*' | sed 's#^\./#Proofs/#; s/\.lean$//; s#/#.#g' | sort)
echo "modules: $(echo $mods | wc -w)"
/usr/bin/time -f "replay %e s" lake env lean --run /tmp/ReplayTop.lean /tmp/replay-top.json $mods 2>&1 | tail -5
