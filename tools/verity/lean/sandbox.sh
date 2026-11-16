#!/usr/bin/env bash
# The build sandbox, run as
#   unshare --user --map-root-user --mount --net --ipc --pid --fork --kill-child -- \
#     bash sandbox.sh [--ro RO_DIR]... [--at FIXED TREE] RW_DIR... -- CMD...
# or, for a fetch, with the network kept (`unshare` without `--net`) and said (`--net`), the rest the same.
# Inside new user, mount, network, IPC and PID namespaces: every mount read-only except the RW_DIRs (the build's .lake
# directories), a private /tmp, /dev/shm and /var/tmp unless an RW_DIR or an RO_DIR (read, never written: Lake's
# artifact cache, or the seal the sampled recompile reads) lives under one, no network, and a fresh /proc. TMPDIR, TEMP
# and TMP name a writable directory: TMPDIR if it is under an RW_DIR, else the private /tmp, else a private tmpfs on the
# first RW_DIR's `.sandbox-tmp` (a tmpfs is gone with the sandbox), so a build writes nothing outside the RW_DIRs that
# outlives it. Where a mount CMD could write through stays writable (its remount was refused), the sandbox refuses to run
# CMD.
# An RO_DIR under an RW_DIR (a build's fetched dependencies) is bound read-only over itself.
# With --at, TREE is also bound at FIXED, with every RW_DIR under it, and CMD starts in FIXED's copy of the directory it
# was started in (one under TREE). Lean writes a source's absolute path into what it compiles, so a build that reads its
# sources under FIXED gives the same bytes wherever TREE is. FIXED is a name in /var/tmp, made in this sandbox's private
# tmpfs there and gone with it: nothing is made on the host, and two sandboxes at once each have their own. Where
# /var/tmp got no private tmpfs, or is a link (CMD's physical directory is not under FIXED), --at refuses: a tree bound
# elsewhere would compile to other bytes.
# CMD is PID 1, so every process a build starts (compile-time code included) ends with it. It runs with no capability
# (a new user namespace starts with empty inheritable and ambient sets, so an empty bounding set leaves its root none
# after the `exec`) and can make no user namespace (this one's count is capped at 0, which only a capability here lifts), so it can't
# remount, mount over or unmount anything, here or in a namespace it opens: the mounts above are the ones it gets.
#
# CMD reaches no unix socket it did not make (a read-only mount doesn't stop `connect`, and through the user's systemd
# bus `systemd-run --user` would start anything outside every namespace): /run (and /var/run where it is no link to it)
# and the job's TMPDIR get private read-only tmpfs, under /tmp's rule, and CMD runs under `sockets.py exec`, whose
# seccomp filter refuses it every unix socket but a `SOCK_STREAM` `socketpair`. Before CMD runs, `sockets.py probe`,
# run as CMD runs, tries every unix socket the kernel lists for a process the sandbox can see and the well-known ones:
# the sandbox refuses (3) to run CMD if any connects. Where /etc/resolv.conf resolves into the hidden /run, its copy
# stays (a fetch resolves names).
set -euo pipefail
tool=$(realpath -- "${BASH_SOURCE[0]}")
tool=${tool%/*}
py=$(command -v python3) || { echo "sandbox.sh: no python3 here, which keeps CMD from the host's unix sockets" >&2; exit 3; }
rw=()
ro=()
at=
tree=
net=
while [ "$1" != "--" ]; do
  case "$1" in
    --net) net=1; shift;;
    --ro) ro+=("$2"); shift 2;;
    --at) at="$2"; tree=$(realpath "$3"); shift 3;;
    *) rw+=("$1"); shift;;
  esac
done
shift
# The callers name each RW_DIR by the tree's own paths (a package's `.lake`), so one that is itself a symbolic link would let
# the tree choose what the build writes: the link's target, a source of the tree or a cache outside it. A link above it
# (a slot tree, a link into the node's cache root) is the node's own, and stays. An RO_DIR under an RW_DIR that is itself
# a link is refused too: the tree would choose which directory is bound read-only.
for d in "${rw[@]}"; do
  if [ -L "$d" ]; then
    echo "sandbox.sh: $d, a directory the build would write, is a symbolic link, so the tree would choose where the build" \
      "writes: make it a directory" >&2
    exit 3
  fi
done
# a bind mount and /proc/self/mountinfo name a directory by its resolved path, so keep and seen hold the resolved ones
for i in "${!rw[@]}"; do mkdir -p "${rw[i]}"; rw[i]=$(realpath "${rw[i]}"); done
for d in "${ro[@]}"; do
  [ -L "$d" ] || continue
  up=$(dirname -- "$d")
  up=$(realpath -m -- "$up")
  for w in "${rw[@]}"; do
    case "$up/" in
      "$w"/*)
        echo "sandbox.sh: $d, a directory under $w that the build would read only, is a symbolic link, so the tree would" \
          "choose which directory is read-only: make it a directory" >&2
        exit 3;;
    esac
  done
done
for i in "${!ro[@]}"; do ro[i]=$(realpath -m "${ro[i]}"); done
# mountinfo writes a mount point's space, tab, newline and backslash as octal escapes (\040, \011, \012, \134): `path`
# decodes them, and gives a name holding a newline (which no line-per-name list carries), any other escape or a stray
# backslash a newline too, so that each is passed on as no path
unescape='function path(s,   o, i, d, n) {
  o = ""
  while ((i = index(s, "\\")) > 0) {
    d = substr(s, i + 1, 3)
    if (d !~ /^[0-7][0-7][0-7]$/) return o substr(s, 1, i - 1) "\n"
    n = substr(d, 1, 1) * 64 + substr(d, 2, 1) * 8 + substr(d, 3, 1)
    if (n != 9 && n != 32 && n != 92) return o substr(s, 1, i - 1) "\n"
    o = o substr(s, 1, i - 1) sprintf("%c", n)
    s = substr(s, i + 4)
  }
  return o s
}'
mount --make-rprivate /
for d in "${rw[@]}"; do mount --bind "$d" "$d"; done
keep=" ${rw[*]} "
seen=" ${rw[*]} ${ro[*]} "
LC_ALL=C awk "$unescape"'{ m = path($5); if (m !~ /\n/) print m }' /proc/self/mountinfo | sort -u | while IFS= read -r m; do
  case "$keep" in *" $m "*) continue;; esac
  mount -o remount,bind,ro "$m" 2>/dev/null || true
done
for d in "${ro[@]}"; do
  for w in "${rw[@]}"; do case "$d/" in "$w"/*) [ ! -d "$d" ] || { mount --bind "$d" "$d"; mount -o remount,bind,ro "$d"; };; esac; done
done
private_tmp=
private_var_tmp=
ours=()
resolv=$(realpath -m /etc/resolv.conf)
case "$resolv" in
  /run/*|/var/run/*) resolved=$(cat /etc/resolv.conf 2>/dev/null) || resolved=;;
  *) resolv=;;
esac
hidden=(/run)
[ -L /var/run ] || hidden+=(/var/run)
if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && [ ! -L "$TMPDIR" ]; then
  t=$(realpath "$TMPDIR")
  case " / /tmp /dev/shm /var/tmp ${hidden[*]} " in *" $t "*) ;; *) hidden+=("$t");; esac
fi
for t in /tmp /dev/shm /var/tmp "${hidden[@]}"; do
  [ -d "$t" ] || continue
  case "$seen" in *" $t "*|*" $t/"*) continue;; esac
  case " ${hidden[*]} " in
    *" $t "*) mount -t tmpfs -o mode=755 tmpfs "$t";;
    *) mount -t tmpfs -o mode=1777 tmpfs "$t";;
  esac
  ours+=("$t")
  [ "$t" != /tmp ] || private_tmp=1
  [ "$t" != /var/tmp ] || private_var_tmp=1
done
if [ -n "$resolv" ] && [ ! -e "$resolv" ]; then
  mkdir -p -- "${resolv%/*}"
  printf '%s\n' "$resolved" > "$resolv"
fi
for t in "${hidden[@]}"; do
  case " ${ours[*]} " in *" $t "*) mount -o remount,bind,ro "$t";; esac
done
# the kernel's list of unix sockets, read in the host's /proc before the sandbox's own covers it
sockets=$("$py" -I -S "$tool/sockets.py" inventory) || exit 3
mount -t proc proc /proc
ours+=(/proc)
if [ -n "$at" ]; then
  case "$at" in
    /var/tmp/*/*|/var/tmp/.|/var/tmp/..) echo "sandbox.sh: --at $at: the fixed path is one name in /var/tmp" >&2; exit 2;;
    /var/tmp/?*) ;;
    *) echo "sandbox.sh: --at $at: the fixed path is one name in /var/tmp" >&2; exit 2;;
  esac
  if [ -z "$private_var_tmp" ]; then
    if [ -d /var/tmp ]; then why="an RW_DIR or an RO_DIR lies under /var/tmp, so it stays the host's"
    else why="/var/tmp is no directory here"; fi
    echo "sandbox.sh: --at $at: $why, and a tree bound anywhere else would compile to other bytes than every other" \
      "build's. Keep the tree, its builds and the artifact cache out of /var/tmp, on a host that has that directory." >&2
    exit 2
  fi
  here=$(realpath "$PWD")
  case "$here/" in
    "$tree"/*) ;;
    *) echo "sandbox.sh: --at $at $tree: started in $here, which is not under that tree" >&2; exit 2;;
  esac
  mkdir "$at"
  mount --rbind "$tree" "$at"
  cd "$at${here#"$tree"}"
  case "$(pwd -P)/" in
    "$at"/*) ;;
    *) echo "sandbox.sh: --at $at: the bound tree is at $(pwd -P), not under $at: /var/tmp, or a directory above it, is a" \
         "link here, so a build would name its sources under a path no other build's name. Make /var/tmp a directory of" \
         "its own on this host." >&2
       exit 2;;
  esac
fi
# a compiler writes its temporary files under TMPDIR, which a job sets outside the RW_DIRs (a node's /workspace/tmp)
tmp=
if [ -n "${TMPDIR:-}" ]; then
  t=$(realpath -m "$TMPDIR")
  for d in "${rw[@]}"; do case "$t/" in "$d"/*) tmp=$TMPDIR;; esac; done
fi
if [ -z "$tmp" ] && [ -n "$private_tmp" ]; then
  tmp=/tmp
elif [ -z "$tmp" ]; then
  tmp="${rw[0]}/.sandbox-tmp"
  mkdir -p "$tmp"
  mount -t tmpfs -o mode=1777 tmpfs "$tmp"
  ours+=("$tmp")
fi
# A mount whose read-only remount failed would leave the tree, an RO_DIR or a cache writable, so every mount CMD can
# reach must now be read-only but the RW_DIRs (and their copies at FIXED) and the tmpfs and /proc made here. A mount
# is out of reach when a later one sits at its point or above it (mountinfo lists mounts in the order they were made).
for w in "${rw[@]}"; do
  [ -z "$at" ] || case "$w/" in "$tree"/*) ours+=("$at${w#"$tree"}");; esac
done
# CMD's own directory reaches the mount it is on without a lookup by path. Without --at, CMD starts where the sandbox was
# started, which a mount made since may cover: a covered mount that holds that directory is still CMD's, so it is not
# hidden. With --at, CMD starts in the tree bound at FIXED, made after every mount that could cover it.
started=
[ -n "$at" ] || started=$(pwd -P)
# One line per stuck mount, decoded; one whose name doesn't decode to a path is `!` and its escaped name, writable. The
# kept paths are kept only where the mount at them is the visible one.
stuck=$(SANDBOX_KEPT=$(printf '%s\n' "${rw[@]}" "${ours[@]}") SANDBOX_STARTED=$started LC_ALL=C awk "$unescape"'
  BEGIN { n = split(ENVIRON["SANDBOX_KEPT"], k, "\n"); for (i = 1; i <= n; i++) kept[k[i]] = 1; s = ENVIRON["SANDBOX_STARTED"] }
  { m[NR] = path($5); e[NR] = $5; o[NR] = $6 }
  END {
    for (i = 1; i <= NR; i++) {
      if (o[i] ~ /^ro(,|$)/) continue
      hid = 0
      for (j = i + 1; j <= NR && !hid; j++) hid = m[j] == "/" || m[j] == m[i] || index(m[i], m[j] "/") == 1
      if (hid ? !(s != "" && (m[i] == "/" || s == m[i] || index(s, m[i] "/") == 1)) : m[i] in kept) continue
      print (m[i] ~ /\n/ ? "!" e[i] : m[i])
    }
  }' /proc/self/mountinfo)
echo 0 > /proc/sys/user/max_user_namespaces
as_cmd=(setpriv --bounding-set=-all -- "$py" -I -S "$tool/sockets.py" exec --)
# A stuck mount is out of CMD's reach only where, run as CMD runs, this shows it: a directory above it that CMD can't
# search and doesn't own (so can't chmod), and that doesn't hold CMD's own directory (which CMD is already past); or
# nothing in it that CMD may write or owns, and no directory CMD can search but not list, with every word find says a
# "Permission denied". A tool that fails or says nothing it should, a mount holding CMD's own directory, and anything
# else leave it writable. A kernel filesystem under a directory of the host's root, as tracefs under /sys/kernel/debug,
# is out of reach.
unreachable='
m=$1 a=$1 up=()
me=$(id -u) || exit 1
here=$(pwd -P) || exit 1
case $me in "" | *[!0-9]*) exit 1;; esac
case $m in /*) ;; *) exit 1;; esac
under() { [ "$2" = / ] || [ "$1" = "$2" ] || case $1 in "$2"/*) ;; *) false;; esac; }
under "$here" "$m" && exit 1
while [ "$a" != / ]; do
  [ ${#up[@]} -lt ${#m} ] || exit 1
  a=$(dirname -- "$a") || exit 1
  up=("$a" "${up[@]}")
done
for a in "${up[@]}"; do
  [ -x "$a" ] && continue
  under "$here" "$a" && exit 1
  o=$(stat -c %u -- "$a") || exit 1
  case $o in "" | *[!0-9]*) exit 1;; esac
  [ "$o" != "$me" ]
  exit
done
command -v find >/dev/null || exit 1
out=$(LC_ALL=C find "$m" -xdev ! -type l \( -writable -o -user "$me" -o -type d -executable ! -readable \) \
  -printf "hit %p\n" -quit 2>&1)
s=$?
[ $s -ne 0 ] || [ -n "$out" ] || exit 0
[ $s -eq 1 ] || exit 1
nl="
"
rest=$out$nl
while [ -n "$rest" ]; do
  case ${rest%%"$nl"*} in "find: "*": Permission denied") ;; *) exit 1;; esac
  rest=${rest#*"$nl"}
done
exit 0'
writable=()
set -f
IFS=$'\n'
for m in $stuck; do
  case $m in
    "!"*) writable+=("${m#!}");;
    *) "${as_cmd[@]}" bash -c "$unreachable" _ "$m" || writable+=("$m");;
  esac
done
unset IFS
set +f
if [ ${#writable[@]} -gt 0 ]; then
  echo "sandbox.sh: these mounts stay writable here, which the build could write through:" "${writable[@]}" "(their" \
    "read-only remount failed). Run the build where every mount can be remounted read-only in a user namespace." >&2
  exit 3
fi
printf '%s\n' "$sockets" | "${as_cmd[@]}" "$py" -I -S "$tool/sockets.py" probe "${DBUS_SESSION_BUS_ADDRESS-}" \
  "${DBUS_SYSTEM_BUS_ADDRESS-}" "${SSH_AUTH_SOCK-}" || exit 3
export TMPDIR="$tmp" TEMP="$tmp" TMP="$tmp"
exec "${as_cmd[@]}" "$@"
