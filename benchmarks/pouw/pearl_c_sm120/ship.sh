#!/usr/bin/env bash
# Pearl-C's sm_120 ship (the RTX PRO 6000): ../pearl_c_ship.sh for this build.
# Usage: bash benchmarks/pouw/pearl_c_sm120/ship.sh [OUT]   (default /tmp/pearl-c-sm120-ship)
exec bash "$(dirname "$0")/../pearl_c_ship.sh" pearl_c_sm120 "$@"
