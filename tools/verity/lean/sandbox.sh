#!/usr/bin/env bash
# The build sandbox, run as
#   unshare --user --map-root-user --mount --net --pid --fork --kill-child -- \
#     bash sandbox.sh [--ro RO_DIR]... [--overlay DIR TOP]... RW_DIR... -- CMD...
# Inside new user, mount, network and PID namespaces: every mount read-only except the RW_DIRs (the build's .lake
# directories), a private /tmp, /dev/shm and /var/tmp unless an RW_DIR, an RO_DIR (read, never written: Lake's artifact
# cache) or an overlay's DIR lives under one, no network, and a fresh /proc. TMPDIR, TEMP and TMP name a writable
# directory: TMPDIR if it is under an RW_DIR, else the private /tmp, else a private tmpfs on the first RW_DIR's
# `.sandbox-tmp` (a tmpfs is gone with the sandbox), so a build writes nothing outside the RW_DIRs that outlives it.
# An overlay mounts an overlayfs over DIR whose upper layer is TOP/upper (TOP/work its work directory, TOP writable):
# CMD sees DIR and may change it, and every change lands in TOP/upper, a deletion as a whiteout, while DIR itself, which
# other processes read, is never written (the sampled recompile, check.py's `sampled_recompile`).
# CMD is PID 1, so every process a build starts (compile-time code included) ends with it.
set -euo pipefail
rw=()
ro=()
ovl=()
while [ "$1" != "--" ]; do
  case "$1" in
    --ro) ro+=("$2"); shift 2;;
    --overlay) ovl+=("$2" "$3"); shift 3;;
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
for t in /tmp /dev/shm /var/tmp; do
  [ -d "$t" ] || continue
  case "$seen" in *" $t/"*) continue;; esac
  mount -t tmpfs -o mode=1777 tmpfs "$t"
  [ "$t" != /tmp ] || private_tmp=1
done
mount -t proc proc /proc
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
