#!/usr/bin/env bash
# The build sandbox, run as
#   unshare --user --map-root-user --mount --net --pid --fork --kill-child -- \
#     bash sandbox.sh [--ro RO_DIR]... [--overlay DIR TOP]... [--at FIXED TREE] RW_DIR... -- CMD...
# Inside new user, mount, network and PID namespaces: every mount read-only except the RW_DIRs (the build's .lake
# directories), a private /tmp, /dev/shm and /var/tmp unless an RW_DIR, an RO_DIR (read, never written: Lake's artifact
# cache) or an overlay's DIR lives under one, no network, and a fresh /proc. TMPDIR, TEMP and TMP name a writable
# directory: TMPDIR if it is under an RW_DIR, else the private /tmp, else a private tmpfs on the first RW_DIR's
# `.sandbox-tmp` (a tmpfs is gone with the sandbox), so a build writes nothing outside the RW_DIRs that outlives it.
# An overlay mounts an overlayfs over DIR whose upper layer is TOP/upper (TOP/work its work directory, TOP writable):
# CMD sees DIR and may change it, and every change lands in TOP/upper, a deletion as a whiteout, while DIR itself, which
# other processes read, is never written (the sampled recompile, check.py's `sampled_recompile`).
# With --at, TREE is also bound at FIXED, with every RW_DIR and overlay under it, and CMD starts in FIXED's copy of the
# directory it was started in (one under TREE). Lean writes a source's absolute path into what it compiles, so a build
# that reads its sources under FIXED gives the same bytes wherever TREE is. FIXED is a name in /var/tmp, made in this
# sandbox's private tmpfs there and gone with it: nothing is made on the host, and two sandboxes at once each have
# their own. Where /var/tmp got no private tmpfs, or is a link (CMD's physical directory is not under FIXED), --at
# refuses: a tree bound elsewhere would compile to other bytes.
# CMD is PID 1, so every process a build starts (compile-time code included) ends with it.
set -euo pipefail
rw=()
ro=()
ovl=()
at=
tree=
while [ "$1" != "--" ]; do
  case "$1" in
    --ro) ro+=("$2"); shift 2;;
    --overlay) ovl+=("$2" "$3"); shift 3;;
    --at) at="$2"; tree=$(realpath "$3"); shift 3;;
    *) rw+=("$1"); shift;;
  esac
done
shift
# overlayfs splits its options at commas and its layers at colons, and has no escape for either
for ((i = 0; i < ${#ovl[@]}; i += 2)); do
  case "${ovl[i]}${ovl[i+1]}" in *[,:\\]*) echo "sandbox.sh: an overlay's path holds a comma, a colon or a backslash: ${ovl[i]} ${ovl[i+1]}" >&2; exit 2;; esac
  ovl[i]=$(realpath "${ovl[i]}")
  mkdir -p "${ovl[i+1]}/upper" "${ovl[i+1]}/work"
  rw+=("${ovl[i+1]}")
done
lower=()
for ((i = 0; i < ${#ovl[@]}; i += 2)); do lower+=("${ovl[i]}"); done
# a bind mount and /proc/self/mountinfo name a directory by its resolved path, so keep and seen hold the resolved ones
for i in "${!rw[@]}"; do mkdir -p "${rw[i]}"; rw[i]=$(realpath "${rw[i]}"); done
for i in "${!ro[@]}"; do ro[i]=$(realpath -m "${ro[i]}"); done
mount --make-rprivate /
for d in "${rw[@]}"; do mount --bind "$d" "$d"; done
keep=" ${rw[*]} "
seen=" ${rw[*]} ${ro[*]} ${lower[*]} "
awk '{print $5}' /proc/self/mountinfo | sort -u | while read -r m; do
  case "$keep" in *" $m "*) continue;; esac
  mount -o remount,bind,ro "$m" 2>/dev/null || true
done
for ((i = 0; i < ${#ovl[@]}; i += 2)); do
  top=$(realpath "${ovl[i+1]}")
  mount -t overlay overlay -o "lowerdir=${ovl[i]},upperdir=$top/upper,workdir=$top/work,userxattr" "${ovl[i]}"
done
private_tmp=
private_var_tmp=
for t in /tmp /dev/shm /var/tmp; do
  [ -d "$t" ] || continue
  case "$seen" in *" $t/"*) continue;; esac
  mount -t tmpfs -o mode=1777 tmpfs "$t"
  [ "$t" != /tmp ] || private_tmp=1
  [ "$t" != /var/tmp ] || private_var_tmp=1
done
mount -t proc proc /proc
if [ -n "$at" ]; then
  case "$at" in
    /var/tmp/*/*|/var/tmp/.|/var/tmp/..) echo "sandbox.sh: --at $at: the fixed path is one name in /var/tmp" >&2; exit 2;;
    /var/tmp/?*) ;;
    *) echo "sandbox.sh: --at $at: the fixed path is one name in /var/tmp" >&2; exit 2;;
  esac
  if [ -z "$private_var_tmp" ]; then
    if [ -d /var/tmp ]; then why="an RW_DIR, an RO_DIR or an overlay's DIR lies under /var/tmp, so it stays the host's"
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
fi
export TMPDIR="$tmp" TEMP="$tmp" TMP="$tmp"
exec "$@"
