#!/usr/bin/env bash
# certifier_timing's cases with the certifier and the observer on two hosts, one side each:
#
#   pair.sh serve   ADDR PORT START OUT [serve's flags]   on the certifier's host: OUT/out/<case>/
#   pair.sh observe ADDR PORT START OUT                   on the observer's: OUT/seen/<case>/observer.json
#
# and then, with both at hand, `run.py score A/out/* --observer B/seen`. ADDR:PORT is where the certifier listens and
# the observer's host reaches it. START is a time of day both hosts' clocks give (`date +%s`), ahead of both: the
# cases start on it one after another, each at a time both sides work out alone, so a case that fails takes no other
# with it and nothing else passes between the hosts. A side that comes late leaves the cases whose time has passed.
#
# The cases, by PARTS (default: guard attacks control): the guard's two (the relay, the planted delay and its ignored
# twin), each of ATTACKS (default: write burst) with each of OBSERVERS (default: spin plain), and `write`'s control,
# the server ignoring its bits. INTERVALS (3000) and GUARD_INTERVALS (600) are their lengths and INTERVAL_MS (100)
# the certifier's bucket; both sides take the same. A case of n intervals takes n * INTERVAL_MS / 1000 + 45 seconds,
# so a second set (other flags, another OUT) starts that many after the first for each of its cases. TREE is the
# checkout whose certifier it is, where `serve` takes `--tree`.
set -euo pipefail
side=$1 addr=$2 port=$3 at=$4 out=$5
shift 5
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
intervals=${INTERVALS:-3000}
tau=${INTERVAL_MS:-100}

cases=()
for part in ${PARTS:-guard attacks control}; do
    case $part in
    guard)
        cases+=("guard-planted relay delay ${GUARD_INTERVALS:-600} plain"
                "guard-ignored relay delay ${GUARD_INTERVALS:-600} plain --ignore")
        ;;
    attacks)
        for attack in ${ATTACKS:-write burst}; do
            for observer in ${OBSERVERS:-spin plain}; do
                cases+=("$attack-$observer network $attack $intervals $observer")
            done
        done
        ;;
    control) cases+=("control-spin network write $intervals spin --ignore") ;;
    *)
        echo "pair.sh: PARTS are of guard, attacks, control: not $part" >&2
        exit 2
        ;;
    esac
done

for case in "${cases[@]}"; do
    read -r name certifier attack n observer ignore <<<"$case"
    ahead=$((at - $(date +%s)))
    if ((ahead < -5)); then
        echo "pair.sh: $name's time has passed" >&2
    else
        if ((ahead > 0)); then sleep "$ahead"; fi
        if [ "$side" = serve ]; then
            "$py" "$here/run.py" serve --listen "$addr" --port "$port" --wait-s 20 --certifier "$certifier" \
                --attack "$attack" $ignore --intervals "$n" --interval-ms "$tau" --out "$out/out/$name" "$@" ||
                echo "pair.sh: $name failed" >&2
        else
            "$py" "$here/run.py" observe "$addr:$port" --wait-s 20 --certifier "$certifier" --observer "$observer" \
                --intervals "$n" --interval-ms "$tau" ${TREE:+--tree "$TREE"} --out "$out/seen/$name" ||
                echo "pair.sh: $name failed" >&2
        fi
    fi
    at=$((at + n * tau / 1000 + 45))
done
