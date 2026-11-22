"""``verity bench recursion_e2e``: refuses by name. The old recursion's benchmark is gone, since the first version has no outer
proof. The five metrics, `benchmarks/recursion-run-seconds`, `benchmarks/recursion-lean-verifier-seconds`,
`benchmarks/recursion-core-years-world-year`, `benchmarks/recursion-ks-error-log2-world-year` and
`benchmarks/recursion-stop-leak-bits-world-year`, keep their stored readings."""
import sys

REFUSAL = ("recursion_e2e: the old recursion's benchmark is gone, since the first version has no outer proof; the readings "
           "of record stand")

if __name__ == "__main__":
    sys.exit(REFUSAL)
