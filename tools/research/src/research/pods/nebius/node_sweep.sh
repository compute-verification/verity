#!/bin/bash
# infra's delete-without-asking node sweep, run ON a node as root by vy-node-sweep.timer (deploy.toml installs it as
# /usr/local/bin/vy-node-sweep).
# Usage: vy-node-sweep [--dry-run] [--lake]. Per the policy in the
# notes' lanes/resource-steward/*-report-resource-steward.md:
#  - never a whole /workspace/research/src/<sha> tree: research.store.evict.evict_src (vy-store-evict) is the only deleter of
#    those, because a tree may hold a run's dirty source, and only it knows when the run's preserved record holds a copy;
#  - check scratch untouched for 2 h and not held open: /tmp/pytest-of-research/pytest-*, <verity-check cache>/lean-audit-scratch-*;
#  - only with --lake, the Lean dependencies (.lake/packages) a failed audit left in a source tree nothing live names, at any
#    age (that section);
#  - what is live (REFS_PY: a request not yet finished or refused whose runner is alive or not yet launched, a fill job queued
#    or running, any process's cwd, root, open files, maps, argv or environment) is never deleted.
# Nothing a retention record keeps is deleted, whatever the rules above say (retained(); only the record's owner releases it).
# Prints "deleted PATH files=N mb=M age=Xh" or "kept PATH: why" per candidate, each line first appended, stamped, to $LOG (not
# in a dry run), and goes on to the end when its ssh drops (SIGPIPE ignored), so $LOG is the record of what it did.
# Exit 0, or 2 if not root or already running. Patterns go through files, so no scanner process carries one in its argv or
# environment.
[ "$(id -u)" = 0 ] || { echo "node-sweep: needs root to read every process" >&2; exit 2; }
exec 9>/run/lock/resource-steward-sweep.lock; flock -n 9 || { echo "node-sweep: another sweep is running" >&2; exit 2; }
DRY=0; LAKE=0
while [ $# -gt 0 ]; do case $1 in --dry-run) DRY=1;; --lake) LAKE=1;;
  --src|--approved) echo "node-sweep: $1 is gone: whole source trees are vy-store-evict's alone (research.store.evict.evict_src)" >&2; exit 2;;
  *) echo "unknown $1" >&2; exit 2;; esac; shift; done
R=/workspace/research; SRC=$R/src; now=$(date +%s); T=$(mktemp -d); trap 'rm -rf $T' EXIT; trap '' PIPE
LOG=/home/research/resource-steward/node-sweep.log
say() { [ $DRY = 1 ] || printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> $LOG; echo "$*" 2>/dev/null; return 0; }
age_h() { echo $(( (now - $(stat -c %Y "$1")) / 3600 )); }

REFS_PY='
import glob, json, os, re, subprocess, sys
pats = open(sys.argv[1]).read().split(); R = "/workspace/research"; me = {str(os.getpid()), str(os.getppid())}
TERMINAL = {"done", "failed", "cancelled", "exited", "timeout", "timed_out", "refused"}
# a path names itself or what is under it (pytest-1 is not pytest-12); a sha names any path that carries it
rx = {p: re.compile(re.escape(p) + r"(?![^/\s])") for p in pats if p.startswith("/")}
def hit(text, ref):
    for p in pats:
        if (rx[p].search(text) if p in rx else p in text): print(p, ref)
def load(path):
    try: return json.load(open(path))
    except Exception: return {}
def read(path):
    try: return open(path, errors="replace").read()
    except Exception: return ""
for q in sorted(glob.glob(R + "/requests/r2*")):
    rid = os.path.basename(q); run = f"{R}/runs/{rid}"
    # a refused request never runs: relaunching its id returns the same refusal
    if load(run + "/status.json").get("state", "") in TERMINAL or os.path.exists(run + "/refused.json"): continue
    if os.path.exists(run + "/launch.json"):
        pid = str(load(run + "/launch.json").get("runner_pid") or "")
        if not pid or not os.path.exists("/proc/" + pid): continue
    hit("".join(read(f) for f in [*glob.glob(q + "/*"), run + "/launch.json", run + "/job.json"]), "request:" + rid)
for f in glob.glob("/workspace/pouw/fill/queue/*") + glob.glob("/workspace/pouw/fill/running/*"):
    hit(read(f), "fill:" + f.split("/workspace/pouw/fill/")[1])
for p in os.listdir("/proc"):
    if not p.isdigit() or p in me: continue
    d = "/proc/" + p; parts = []
    for link in ("cwd", "root"):
        try: parts.append(os.readlink(f"{d}/{link}"))
        except OSError: pass
    try: parts += [os.readlink(f"{d}/fd/{fd}") for fd in os.listdir(d + "/fd")]
    except OSError: pass
    for f in ("maps", "cmdline", "environ"):
        try: parts.append(open(f"{d}/{f}", "rb").read().decode(errors="replace").replace("\0", " "))
        except OSError: pass
    comm = read(d + "/comm").strip()
    hit("\n".join(parts), f"proc:{p}({comm})")
'
held() { python3 -c "$REFS_PY" $1 | awk '{a[$1]=a[$1]" "$2} END{for (k in a) print k a[k]}'; }  # PATTERNS_FILE -> "<pattern> <refs...>"
named() { awk -v s="$1" '$1==s{$1=""; print substr($0,2)}' $2; }

# A retention record (`research keep`: <dir>/.retention.json, or <path>.retention.json beside it) on a path or any directory
# above it keeps the path until the record expires; one anywhere beneath it keeps it whole, expired or not; one that can't be
# read, or an expiry that can't be parsed, keeps it. retained PATH prints why and returns 0 when a record keeps PATH.
RET_PY='
import datetime, json, os, sys, time
a = os.path.abspath(sys.argv[1]); now = time.time()
while True:
    for rec in [os.path.join(a, ".retention.json")] + ([a + ".retention.json"] if a != "/" else []):
        if not os.path.isfile(rec): continue
        try: r = json.load(open(rec)); e = r.get("expires")
        except Exception: print("unreadable retention record " + rec); sys.exit(0)
        try:
            t = None if e is None else datetime.datetime.fromisoformat(str(e).replace("Z", "+00:00"))
            if t is not None and (t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)).timestamp() <= now: continue
        except ValueError: pass
        print("retention record %s (owner %s, expires %s)" % (rec, r.get("owner"), e)); sys.exit(0)
    if a == "/": break
    a = os.path.dirname(a)
'
retained() { local w; w=$(python3 -c "$RET_PY" "$1" 2>&1) || w="could not read the retention records above it: $w"
  [ -z "$w" ] && w=$(find "$1" -xdev -name '*.retention.json' -print -quit 2>/dev/null | sed 's/^/retention record beneath it: /')
  [ -n "$w" ] && echo "$w"; }
gone() { local p=$1 n m w
  if w=$(retained "$p"); then
    case $p in */.trash/*|*.rs-trash-*) say "STUCK $p: $w; left there for its owner";; *) say "kept $3: $w";; esac; return 0; fi
  n=$(find "$p" -xdev 2>/dev/null | wc -l); m=$(du -sm --one-file-system "$p" 2>/dev/null | cut -f1)
  if [ $DRY = 1 ]; then say "would delete $3 files=$n mb=$m age=${2}h"; else rm -rf --one-file-system -- "$p" && say "deleted $3 files=$n mb=$m age=${2}h"; fi; }

# check scratch
: > $T/scr
cache=$(readlink -f /home/research/.cache/verity-check)
for d in /tmp/pytest-of-research/pytest-[0-9]* $cache/lean-audit-scratch-*; do
  [ -d "$d" ] && [ ! -L "$d" ] || continue
  [ -z "$(find "$d" -xdev -newermt '-2 hours' -print -quit 2>/dev/null)" ] && echo "$d" >> $T/scr
done
if [ -s $T/scr ]; then
  held $T/scr > $T/held
  for d in $(cat $T/scr); do
    r=$(named "$d" $T/held); [ -n "$r" ] && { say "kept $d: $r"; continue; }
    gone "$d" $(age_h "$d") "$d"
  done
fi

# Lean dependencies in source trees (at any age): WarmDeps.take moves or copies
# check's Lean packages into a tree's .lake/packages, and a failed audit never gives them back; the next run on the tree drops
# them before its own take. Only .lake/packages of the verifier's Lake packages goes, only in a tree nothing live names
# (REFS_PY, by sha); each is renamed aside, the tree checked once more, put back if named, and deleted.
# Trees from before and after the layout move and the Lean move sit side by side, so every layout's paths are listed.
# These are strings the moves' path rewriting must leave alone: each names where a tree of that age keeps them.
LK_OLD="verity/core/service/flock verity/core/service/flock/level3 verity/core/service/flock/soundness"
LK_NEW="verity/core/protocols/verification/flock/lean security_proofs/flock/level3 security_proofs/flock/soundness"
LK_LEAN="Security Security/Proofs"
LK_DIR="security security/Proofs verity/protocols/proof/flock/verifier"
LK="$LK_OLD $LK_NEW $LK_LEAN $LK_DIR"
[ $LAKE = 1 ] && [ $DRY = 0 ] && for k in $LK; do for p in $SRC/*/$k/.lake/packages.rs-trash-*; do
  [ -d "$p" ] && gone "$p" $(age_h "$p") "$p (left by an interrupted sweep)"; done; done
: > $T/lk
[ $LAKE = 1 ] && for d in $SRC/*/; do d=${d%/}; s=${d##*/}
  [[ $s =~ ^[0-9a-f]{40}$ ]] || continue
  for k in $LK; do [ -d $d/$k/.lake/packages ] && [ ! -L $d/$k/.lake/packages ] && { echo $s >> $T/lk; break; }; done
done
if [ -s $T/lk ]; then
  held $T/lk > $T/held
  for s in $(cat $T/lk); do
    r=$(named $s $T/held); [ -n "$r" ] && { say "kept $SRC/$s/**/.lake/packages: $r"; continue; }
    for k in $LK; do p=$SRC/$s/$k/.lake/packages
      [ -d $p ] && [ ! -L $p ] || continue
      w=$(retained $p) && { say "kept $p: $w"; continue; }
      a=$(age_h $SRC/$s)
      [ $DRY = 1 ] && { gone $p $a "$p"; continue; }
      mv -T $p $p.rs-trash-$now || { say "kept $p: rename failed"; continue; }
      echo $s > $T/lk1; held $T/lk1 > $T/held2; r=$(named $s $T/held2)
      if [ -n "$r" ]; then
        mv -T $p.rs-trash-$now $p && say "kept $p: named after rename: $r" || say "STUCK $p.rs-trash-$now: named after rename ($r) and could not be put back"
        continue
      fi
      gone $p.rs-trash-$now $a "$p"
    done
  done
fi

exit 0
