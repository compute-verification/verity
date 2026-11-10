"""LEAN-251, a synthetic file: one a branch adds in `benchmarks/nci/`, where the one map has a row a file and none for this."""
import exposure as X
import verity.core.protocols.network.nci as N
from benchmarks.nci import catalog
from benchmarks.nci.kappa import kappa
from verity.ml.circuits import Program

ROWS = "benchmarks/nci/rows.json"
HERE = "benchmarks/nci/x.py"
