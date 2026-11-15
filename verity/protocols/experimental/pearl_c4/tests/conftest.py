"""These tests extend PoUW's and import its verifier tests' helpers (`circuit_fixtures`, `epoch_coins`, the Pearl-C tests
they share checks with) and its clear audit (``prover/clear``) by name, as PoUW's own tests do."""
import sys
from pathlib import Path

POUW = Path(__file__).resolve().parents[3] / "resource_accounting" / "compute" / "pouw"
for d in (POUW / "prover" / "clear", POUW / "verifier" / "tests"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
