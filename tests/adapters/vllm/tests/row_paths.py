"""The machine paths `verity-vllm row` takes with no default (`config.RunEnv`), as a test's environment names them under its own directory."""
import sys
from pathlib import Path


def row_paths(tmp: Path | str) -> dict[str, str]:
    d = Path(tmp)
    so = ("HIDDEN_SO_FA2", "HIDDEN_SO_FA3", "HIDDEN_SO_FA2_GUARDED", "HIDDEN_SO_FA3_GUARDED", "NORM_TAP_SO", "ROUTER_TAP_SO")
    return {"PY": sys.executable, "SWEEP_DIR": str(d / "sweep"), "HF_HOME": str(d / "hf"), "CUDA_HOME": "/a/toolkit",
            "NATIVE_COLLECT_DEFAULT": str(d / "nc_build"), "GUARD_PRESERVE_FILE": str(d / "guard.preserve"),
            **{k: str(d / f"{k.lower()}.so") for k in so}}
