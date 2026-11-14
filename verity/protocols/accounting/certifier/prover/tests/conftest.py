"""The proof system's benchmark drivers the drivers under test stage, prove and scan with (`proof_drivers`), installed
once for the suite by the drivers' command line, `benchmarks/network_traces/zk_drivers.py`."""
import importlib.util
from pathlib import Path

_ENTRY = Path(__file__).resolve().parents[6] / "benchmarks" / "network_traces" / "zk_drivers.py"
_spec = importlib.util.spec_from_file_location("zk_drivers", _ENTRY)
_zk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_zk)
_zk.install()
