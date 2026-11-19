# shellcheck shell=bash
# Sourced: a shipped cubin's nvcc command line, stated once (the Pearl-C trees' build.sh, the ship frame's ship_cubin_flags and
# ship_probe.sh source it).
#
# cubin_flags GENCODE sets FLAGS for the -gencode value GENCODE, which the caller reads from the target's architecture
# file (machine.py gencode): no FTZ, fast-math, FMA contraction or approximate division or square root, and no line
# info, so a cubin's bytes don't depend on the directory it's built in. A GENCODE that brings such a flag is refused
# (returns 1). `-Xptxas -v` is ptxas's report, which changes no byte of the cubin.
cubin_flags() {
  FLAGS=(-O3 -std=c++17 -gencode "$1" -fmad=false -Xptxas -v -cubin)
  case " ${FLAGS[*]} " in
    *ftz*|*fast-math*|*fast_math*|*prec-div*|*prec-sqrt*|*lineinfo*) echo "a refused flag: ${FLAGS[*]}" >&2; return 1;;
  esac
}
