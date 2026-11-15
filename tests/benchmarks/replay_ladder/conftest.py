import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[2] / "benchmarks" / "replay_ladder") not in sys.path:
    sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "replay_ladder"))
