"""The environment the vstar difftests read, each variable declared once."""
from __future__ import annotations

import os

#: a recursion_e2e run's kept data/TAG (KEEP_DATA=1, after STEP=lean); empty, the deployed recursive tests skip
VSTAR_RECURSIVE_DATA = os.environ.get("VERITY_VSTAR_RECURSIVE_DATA", "")
#: how many flock-verify processes the deployed recursive difftest runs at once
LEAN_JOBS = int(os.environ.get("LEAN_JOBS", "4"))
