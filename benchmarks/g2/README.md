# G2: what committing every value costs served vLLM

G2 is the served Commit's overhead over plain served vLLM (`stock`): per model, GPU, batch and phase (prefill, decode and the
whole run), each measure a Commit arm's ratio to stock in the same run, and the best counting measure in each cell. Serving
under Verity commits every value of every step, and this is what that costs the server. G2 has no pass/fail number: it is
this curve.

Nothing here measures. `benchmarks/served_commit` runs the arms, and these two scripts render its runs from the evidence store:

- `curve.py`: G2's curve, from every run labeled campaign=g2 and the hash-kernel loop's results (labeled campaign=hash-loop).
  Its docstring says which measures count (gates passed, the one hash, a pinned Build or eager) and how a cell takes the
  lowest counting ratio, split by value retention.
- `served.py`: the served arm's runs (`benchmarks/served_commit/served_arm.sh`): the Commit, compiled and eager, against
  compiled and batch-invariant eager stock in one run on one RTX PRO 6000.

Both read the store through the `research` client (`tools/research`), so run them where it is configured, from the repository
root:

~~~text
uv run --no-project --with matplotlib python3 benchmarks/g2/curve.py [--out DIR] [--no-refresh]
uv run --no-project --with matplotlib python3 benchmarks/g2/served.py RUN... [--runs-dir DIR] [--out DIR]
~~~

`curve.py` writes `g2-curve.md` (the curve, then every measure), `g2-curve.png` and `g2-curve.json` (every measure, whether it
counts, and why not). `served.py` writes `g2-served.md`, `g2-served.json` and `g2-served.png`. DIR defaults to `$OUT`, else
`out/`.

Status: current. `tests/` runs the selection logic (which arm is which, which measures count) on stand-in documents, without
the store, and `served_arm.sh` over stand-in arms.
