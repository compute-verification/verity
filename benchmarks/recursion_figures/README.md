# Recursion figures

A recursion bench run times one recursive session on one machine. This script turns that session into the figures a
result is read in: what auditing every machine so would cost, the knowledge-soundness error pooled over the sessions,
and how much their stops can leak, over monitoring periods and over all compute.

One recursive session's figures from a recursion bench result (`benchmarks/recursion_e2e`'s `recursion_e2e.json`, whose
`session_times` hold each session step's wall and CPU seconds): its cost, its knowledge-soundness error and its stop-leak
bound, for one window and pooled over a week, a month and a year (`verity.protocol.MONITORING_PERIODS`), over the reference
datacenter and the world's compute (`benchmarks/census`), each with its basis: measured (the run), proved (the Lean
theorem it reads and what that assumes) or scaled (a session a machine a window, each as this one). The
knowledge-soundness error is κ + t·ε a session (κ and ε proved), pooled by the union bound as sessions·κ + T·ε against the
SHA-512 calls of the world's GPUs and of the datacenter's, and as a share of the sessions and of the ANDs they cover.
`figures.py`'s docstring states every figure.

## Try it

~~~text
PYTHONPATH=benchmarks/recursion_e2e uv run python benchmarks/recursion_figures/figures.py RESULT.json          # the table
PYTHONPATH=benchmarks/recursion_e2e uv run python benchmarks/recursion_figures/figures.py RESULT.json --json   # the figures
PYTHONPATH=benchmarks/recursion_e2e uv run python benchmarks/recursion_figures/figures.py --from RUN           # a recorded
                                                                                    # run's records, from the evidence store
~~~

`RUN` is a recorded run of `verity bench recursion_e2e`; the run of record is the one `metrics.toml`'s command names. It
needs the evidence store's credentials and no GPU. It prints the session (its statement, parts, wall and CPU seconds,
and whether it was accepted), each figure's basis, then one table: the datacenter and the world, each over one session,
a week, a month and a year.

`benchmarks/recursion_e2e`, whose reader (`rec_profile`) it imports, is no package, so the command puts it on the path;
the rest it imports as the workspace installs it.

`--from RUN` fetches the `out/` of the run's record and reads it with this tree's reader, so a run recorded before
`session_times` gives its figures too. Under a research run (`metrics.toml`'s command, from the run of record) the
command also writes `METRICS` into the run's result: the world year's CPU core-years, log₂ knowledge-soundness error and
stop-leak bits, each modeled and written for each fleet's week, month and year, the world's year last. The validation
passes only for figures of an accepted recursive run with every figure there.

It sits beside the bench rather than in it because it reads the census and the monitoring periods, which the public
export doesn't copy.
