"""On this branch, class `shallow` runs island `wildcard` (islands/wildcard/train.py, whose source rides in each
submission's notes), for sets within the top 2 blocks and the final norm.

    python guard.py islands/shallow/train.py --start s50 --out SUB --set mlp-in:-1 [TRAIN_ARGS...]
"""
import runpy
import sys
from pathlib import Path

script = Path(__file__).resolve().parents[1] / "wildcard" / "train.py"
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), init_globals={"KLASS": "shallow"}, run_name="__main__")
