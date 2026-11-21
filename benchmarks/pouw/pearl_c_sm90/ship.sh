#!/usr/bin/env bash
# Pearl-C's sm_90a ship (the H100 and H200): ../pearl_c_ship.sh for this build.
# Usage: bash benchmarks/pouw/pearl_c_sm90/ship.sh [OUT]   (default /tmp/pearl-c-sm90-ship)
exec bash "$(dirname "$0")/../pearl_c_ship.sh" pearl_c_sm90 "$@"
