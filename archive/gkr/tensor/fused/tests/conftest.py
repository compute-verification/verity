import sys
from pathlib import Path

# ``archive/gkr`` on the path so ``import tensor.fused`` resolves without a workspace member.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
