"""``verity bench proof_throughput COMMAND ...``: the proof system's throughput over B (README.md; ``--help`` lists commands)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verity_proof_throughput.cli import main  # noqa: E402

sys.exit(main())
