#!/usr/bin/env bash
# The bench's two edits to its scratch clone of CUTLASS, each a no-op unless its macro is defined:
#   device_kernel.h: OV_MAXNREG caps the kernel's launch registers with __maxnreg__ (instead of __launch_bounds__'s
#     minimum blocks), so a warp-specialized GEMM leaves registers for a hash CTA on the same SM;
#   sm_90's cooperative and pingpong kernels: OV_LOADR_V / OV_MMAR_V replace the producer's and the consumers'
#     setmaxnreg counts (40 / 232 by default), which must fit the capped budget.
#   bash patch_cutlass.sh CUTLASS_DIR
set -euo pipefail
python3 - "$1" <<'EOF'
import pathlib
import sys

c = pathlib.Path(sys.argv[1]) / "include" / "cutlass"
dk = c / "device_kernel.h"
s = dk.read_text()
old = "__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)\n#endif // __CUDACC__\nvoid device_kernel("
assert s.count(old) == 1, "device_kernel.h: launch bounds not found"
s = s.replace(old, "#ifdef OV_MAXNREG\n__maxnreg__(OV_MAXNREG)\n#else\n"
              "__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)\n#endif\n"
              "#endif // __CUDACC__\nvoid device_kernel(")
dk.write_text(s)
for name in ("sm90_gemm_tma_warpspecialized_cooperative.hpp", "sm90_gemm_tma_warpspecialized_pingpong.hpp"):
    p = c / "gemm" / "kernel" / name
    t = p.read_text()
    for var, default, macro in (("LoadRegisterRequirement", "!HeavyRegisterPressure ? 40 : 24", "OV_LOADR_V"),
                                ("MmaRegisterRequirement", "!HeavyRegisterPressure ? 232 : 240", "OV_MMAR_V")):
        line = f"static constexpr uint32_t {var} = {default};"
        assert t.count(line) == 1, f"{name}: {var} not found"
        t = t.replace(line, f"static constexpr uint32_t {var} =\n#ifdef {macro}\n    {macro};\n#else\n    {default};\n#endif")
    p.write_text(t)
print("patched device_kernel.h and the sm_90 cooperative and pingpong kernels")
EOF
