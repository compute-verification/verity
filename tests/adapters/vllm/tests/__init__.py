from pathlib import Path

# the root's tests/ too, so that `tests.X` names the tests of X here as from the root
__path__.append(str(Path(__file__).resolve().parents[3]))
