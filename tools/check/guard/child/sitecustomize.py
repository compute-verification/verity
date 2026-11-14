"""On the PYTHONPATH `suites.py` gives a suite's tests, so every Python a test starts traces what it reads (`verity_suite_reads.py`)
while a test module is collected or run (`VERITY_SUITE_MODULE` is set). A tracer that can't start says so in the reads file
(`E`), and `suites.py` then fails the suite. Then the interpreter's own `sitecustomize`, which this one shadows, is loaded as it
would have been. Programs that aren't Python are not traced."""
import os
import sys


def _trace() -> None:
    if not (os.environ.get("VERITY_SUITE_MODULE") and os.environ.get("VERITY_SUITE_READS")) or getattr(sys, "_verity_suite_traced", False):
        return
    sys._verity_suite_traced = True                  # one tracer per process, however many copies of this directory are on the path
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("verity_suite_reads", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                                         "verity_suite_reads.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.Reads().install()
    except Exception as e:  # noqa: BLE001  (an untraced child must not pass for a traced one)
        test, why = os.environ.get("VERITY_SUITE_TEST"), " ".join(f"{type(e).__name__}: {e}".split())     # one line, no tabs
        with open(os.environ["VERITY_SUITE_READS"], "a") as f:
            f.write(f"E\t{os.environ['VERITY_SUITE_MODULE']}\tthe tracer did not start: {why}" + (f"\t{test}" if test else "") + "\n")


def _next() -> None:
    import importlib.machinery
    import importlib.util

    def tracer(p: str) -> bool:                      # this directory, or another copy of it (a suite run inside a suite's test)
        return os.path.isfile(os.path.join(os.path.abspath(p or os.curdir), "verity_suite_reads.py"))

    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [p for p in sys.path if not tracer(p)])
    if spec is not None and spec.origin:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


_trace()
_next()
