# tools/difftest: the Lean against the Python that mirrors it

Verity's Lean is the definition and the verifier, and its Python mirrors it for the prover's speed. This harness holds
the two together: a differential test sends the same inputs, invalid ones included, to a Lean executable and to the
Python, and fails on the first answer where they differ, shrunk to a minimal input. Each piece of executable Lean that
has a Python mirror gets such a test, its port. The Python side is here (`verity_difftest`, whose docstring says how a
run compares and when it fails); the Lean side, the Lake package `verity_difftest`, is in `verity/circuits/difftest/`.

## Try it

~~~sh
uv run pytest tests/tools/difftest/test_example_port.py
~~~

This builds the harness's own example port with `lake` and runs it against its Python mirror. It needs the Lean
toolchain and takes about 15 seconds.

Run the suite with `uv run tools/check/suites.py verity-difftest`.
