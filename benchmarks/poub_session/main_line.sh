# Sourced by gate.sh, auditor.sh and gate_whatif.sh, from a clone's root: main's first-parent line and where it is read
# from. `gate_run.sh` sends this file beside gate.sh, which a run executes from its inputs/.

main_line() {
  # LOG LIST: main's first-parent history into LIST, from MAIN_FROM: origin where origin is a repository main is pushed
  # to (GitHub, a bare repository), else, where origin is a checkout, that checkout's own origin. A run's clone
  # (--cwd clone) has its source root as origin, a checkout whose origin/main stood still when it was made; the node's
  # bare repository behind it holds main as the newest launch pushed it. Nothing falls back to a checkout's copy.
  # MAIN_SHOWN is MAIN_FROM without a URL's userinfo, which holds a credential on an agent's VM: only the fetch reads
  # MAIN_FROM, and every message and the fetch's output, appended to LOG, use MAIN_SHOWN.
  local log=$1 list=$2 unshallow=() checkout
  [ "$(git rev-parse --is-shallow-repository)" = true ] && unshallow=(--unshallow)
  MAIN_FROM=$(git remote get-url origin 2> /dev/null) || { MAIN_SHOWN="(no origin)"; return 3; }
  if [ "$(git -C "$MAIN_FROM" rev-parse --is-bare-repository 2> /dev/null)" = false ]; then
    checkout=$MAIN_FROM
    MAIN_FROM=$(git -C "$checkout" remote get-url origin 2> /dev/null) \
      || { MAIN_SHOWN="(the checkout $(unauth <<< "$checkout") has no origin)"; return 3; }
  fi
  MAIN_SHOWN=$(unauth <<< "$MAIN_FROM")
  git fetch -q "${unshallow[@]}" "$MAIN_FROM" +refs/heads/main:refs/remotes/origin/main 2>&1 | unauth >> "$log"
  [ "${PIPESTATUS[0]}" = 0 ] && git rev-list --first-parent refs/remotes/origin/main > "$list"
}

main_refused() {
  # LOG WHAT: the refusal after main_line failed, WHAT being what can't be checked and what was left undone
  echo "can't fetch main's first-parent history from $MAIN_SHOWN ($1), so $2. In a run's clone main is read from the" \
    "node's bare repository behind the source root, never the source root's own copy: if the bare can't be reached or" \
    "holds no main, relaunch from a checkout fetched just before (git fetch origin main), whose launch pushes its" \
    "origin/main there; if it still fails, run git -C THAT_PATH rev-parse --verify refs/heads/main on this node," \
    "THAT_PATH the repository named here, and relaunch once it prints main's head"
}

unauth() {
  # stdin with every URL's userinfo (`scheme://user:secret@`) taken out
  sed 's#://[^/@]*@#://#g'
}
