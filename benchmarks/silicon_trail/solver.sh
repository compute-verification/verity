#!/usr/bin/env bash
# Fetch and build the SAT solver the search calls, outside the repository.  Prints the binary's path.
#
#   solver.sh [DIR]        DIR defaults to a fresh temporary directory
#
# Kissat (Armin Biere and others, MIT licence), release 4.0.4, pinned by commit.  Nothing of it is kept in the tree.
set -euo pipefail
dir="${1:-$(mktemp -d)}"
commit=8af8e56f174b778aef3aa45af9f739b2a5f492c2
mkdir -p "$dir"
if [ ! -x "$dir/kissat/build/kissat" ]; then
  rm -rf "$dir/kissat"
  git clone -q https://github.com/arminbiere/kissat.git "$dir/kissat" >&2
  git -C "$dir/kissat" checkout -q "$commit" >&2
  (cd "$dir/kissat" && ./configure >/dev/null && make -j"$(nproc)" >/dev/null) >&2
fi
test "$(git -C "$dir/kissat" rev-parse HEAD)" = "$commit"
echo "$dir/kissat/build/kissat"
