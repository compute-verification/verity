#!/usr/bin/env bash
# What a Commit run publishes, sourced by commit.sh and benchmarks/one_stage/a5_llama1b_h200.sh. `research run` publishes
# every file a script leaves under $OUT, so a script puts a file there only when PUBLISHED lists it, with the reason it
# holds nothing of a request: no prompt, token id, seed, salt or key, no digest of one that a guess could confirm, and
# none of the requests' lengths, limits or step rows. A file with no entry stays on the node, and the run names it.
# A record whose passing copy is clean and whose failing copy can quote a served token has no entry either: a row's
# stages.txt and row.log and every stage's log, whose last lines a FAIL copies into them.
# An entry is `WHOSE|PATTERN|why`. WHOSE is `row`, a record `publish` copies from the row dir, or `run`, a file the
# script itself writes under $OUT. PATTERN is the file's path under $OUT, matched as a `case` pattern: its `*` also
# matches `/`, so an entry names a file and never a directory's whole contents.
PUBLISHED=(
  "run|machine.txt|the node's user, hostname and GPUs, the tree's commit and the row dir's path: written before any stage runs"
  "run|steps.txt|each stage's start, its flags (the script's constants) and its exit code"
  "run|verdicts.txt|each stage's name, verdict word and time: stages.txt's lines without their detail"
  "run|bundle.sha256|the SHA-256 of the sealed bundle's MANIFEST.json, which lists every bundle file's digest, the salt keys' among them, and the seal time: no guess at a request reproduces it"
  "row|gpu_lease.json|when the stage's GPU lease was asked for, started and released, and the stage's exit code"
  "row|target_family.json|the row id's device class, the declared target and the device's compute capability"
)

listed() {  # REL [WHOSE]: REL, a path under $OUT, has an entry (one of WHOSE, when given)
  local e whose pattern
  for e in "${PUBLISHED[@]}"; do
    IFS='|' read -r whose pattern _ <<< "$e"
    [ -z "${2:-}" ] || [ "$whose" = "$2" ] || continue
    # shellcheck disable=SC2254
    case $1 in $pattern) return 0 ;; esac
  done
  return 1
}

left() {  # WHERE: the names on stdin as one line saying they stay in WHERE; a number in a name is masked, since a row dir's names carry lengths
  local names
  names=$(sed -E 's/[0-9]+/N/g' | sort -u | tr '\n' ' ')
  [ -z "$names" ] || echo "not published (no entry in published.sh), left in $1: $names"
}

publish() {  # ROW_DIR DST: the row dir's records that have an entry into DST, at the same paths; the other files of its top two levels are named
  local f rel
  find "$1" -maxdepth 2 -type f -print0 2>/dev/null | sort -z | while IFS= read -r -d '' f; do
    rel=${f#"$1"/}
    if listed "$rel" row; then mkdir -p "$2/$(dirname "$rel")" && cp "$f" "$2/$rel"; else echo "$rel"; fi
  done | left "$1"
}

verdicts() {  # STAGES_TXT DST: each stage's name, verdict word (PASS, FAIL, WAIVED, PENDING) and time into DST/verdicts.txt; prints the lines new since the last call
  local n=0
  [ -f "$2/verdicts.txt" ] && n=$(wc -l < "$2/verdicts.txt")
  awk '$1 ~ /^[a-z][a-z-]*$/ && $2 ~ /^(PASS|FAIL|WAIVED|PENDING)$/ && $3 ~ /^[0-9TZ:-]+$/ { print $1, $2, $3 }' "$1" 2>/dev/null > "$2/verdicts.txt"
  tail -n "+$((n + 1))" "$2/verdicts.txt"
}

hold() {  # DIR KEPT: DIR holds only files with an entry: any other moves to KEPT, on the node, and is named; one that can't move is deleted
  local f rel
  find "$1" ! -type d -print0 2>/dev/null | sort -z | while IFS= read -r -d '' f; do
    rel=${f#"$1"/}
    [ ! -L "$f" ] && listed "$rel" && continue
    { mkdir -p "$2/$(dirname "$rel")" && mv -f "$f" "$2/$rel"; } 2>/dev/null || rm -f "$f"
    echo "$rel"
  done | left "$2"
}
