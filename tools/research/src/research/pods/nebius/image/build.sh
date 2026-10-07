#!/usr/bin/env bash
# Builds the base image (the Dockerfile beside this script) with the node's docker and imports it into k3s's containerd (namespace
# k8s.io) as verity-base:<the first 16 hex of the Dockerfile's sha256>, so a pod names it by content and never pulls it.
#
#   build.sh [--dry-run]     on a node, as research (sudo); prints the tag and every size it read
#
# Docker and k3s both keep their layers on /: the build holds the image once in docker and the import, at its peak, twice more in
# containerd (its layer tars and their unpacked snapshot; the tars are then left to containerd's garbage collector). The script
# refuses to start, or to import, when that peak would leave / under MIN_FREE_GB.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
MIN_FREE_GB=${MIN_FREE_GB:-40}
EST_GB=${EST_GB:-4}                      # unpacked, as built off the node: 3.9 GB, of which Lean's toolchain is 3.0
tag=verity-base:$(sha256sum "$here/Dockerfile" | cut -c1-16)
ref=docker.io/library/$tag
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }

echo "tag: $tag"
if sudo -n k3s ctr -n k8s.io images ls -q | grep -qxF "$ref"; then echo "already in k3s: $ref"; exit 0; fi
free=$(free_gb)
echo "/ free: ${free} GB; docker root: $(sudo -n docker info -f '{{.DockerRootDir}}'); estimated need: $((3 * EST_GB)) GB"
if [ $((free - 3 * EST_GB)) -lt "$MIN_FREE_GB" ]; then echo "refusing: / would keep under $MIN_FREE_GB GB" >&2; exit 1; fi
[ "${1:-}" = --dry-run ] && exit 0

sudo -n docker build -t "$tag" - < "$here/Dockerfile"
size_gb=$(( ($(sudo -n docker image inspect -f '{{.Size}}' "$tag") + (1 << 30) - 1) >> 30 ))
free=$(free_gb)
echo "built: $tag, ${size_gb} GB; / free: ${free} GB"
if [ $((free - 2 * size_gb)) -lt "$MIN_FREE_GB" ]; then echo "refusing to import: / would keep under $MIN_FREE_GB GB" >&2; exit 1; fi

sudo -n docker save "$tag" | sudo -n k3s ctr -n k8s.io images import --local --discard-unpacked-layers -
sudo -n k3s crictl images | grep -F verity-base
echo "imported: $ref; / free: $(free_gb) GB"
