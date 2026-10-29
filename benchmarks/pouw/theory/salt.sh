#!/usr/bin/env bash
# N-SALT run of record: salt.py checks and prices, the calculator with np_salt_cards.toml, and Salt.lean with its
# axioms when a lean is on the node.  research run ... --cwd source -- bash benchmarks/pouw/theory/salt.sh
set -u
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
python3 benchmarks/pouw/theory/salt.py "$OUT/salt.json" --salts 4096 > "$OUT/salt.log" 2>&1; echo "salt rc=$?" >> "$OUT/rc.txt"
PYTHONPATH=benchmarks/pouw/exact_kit python3 benchmarks/pouw/theory/pouw_space.py --add benchmarks/pouw/theory/np_salt_cards.toml --out "$d" > "$OUT/pouw_space.log" 2>&1; echo "pouw_space rc=$?" >> "$OUT/rc.txt"
cp "$d/ranking.md" "$OUT/" 2>/dev/null
L=$(command -v lean || echo "$HOME/.elan/bin/lean")
if [ -x "$L" ]; then
  cp Security/Proofs/Pouw/Salt.lean "$d/SaltAx.lean"
  for t in pad_binding pad_unchecked_unbound pad_key_reveals pad_reduction pad_repeat_leaks pad_ideal_hiding keyed_binding keyed_ideal_hiding prefix_free_of_head keyed_repeat_leaks lane_int8 weight_int8 mac mac_parity fm_acc fm_mask; do echo "#print axioms Pouw.Salt.$t" >> "$d/SaltAx.lean"; done
  "$L" --version > "$OUT/lean.log" 2>&1; "$L" "$d/SaltAx.lean" >> "$OUT/lean.log" 2>&1; echo "lean rc=$?" >> "$OUT/rc.txt"
else echo "lean absent" >> "$OUT/rc.txt"; fi
cat "$OUT/rc.txt"
