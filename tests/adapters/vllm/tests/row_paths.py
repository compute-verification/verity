"""The machine paths `verity-vllm row` takes with no default (`config.RunEnv`), as a test's environment names them under its own directory."""
import sys
from pathlib import Path


def row_paths(tmp: Path | str) -> dict[str, str]:
    d = Path(tmp)
    return {"PY": sys.executable, "SWEEP_DIR": str(d / "sweep"), "HF_HOME": str(d / "hf"), "CUDA_HOME": "/a/toolkit",
            "VERITY_BUILD_ROOT": str(d / "cp"), "GUARD_PRESERVE_FILE": str(d / "guard.preserve")}
