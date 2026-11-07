# Recursion figures

One recursive session's figures from a recursion bench result (`benchmarks/recursion_e2e`'s `recursion_e2e.json`, whose
`session_times` hold each session step's wall and CPU seconds): its cost, its knowledge-soundness error and its stop-leak
bound, for one window and pooled over a week, a month and a year (`wipe_monitoring.PERIODS`), over the reference
datacenter and the world's compute (`benchmarks/census`), each with its basis: measured (the run), proved (the Lean
theorem it reads and what that assumes) or scaled (a session a machine a window, each as this one). The
knowledge-soundness error is κ + t·ε a session (κ and ε proved), pooled by the union bound as sessions·κ + T·ε against the
SHA-512 calls of the world's GPUs and of the datacenter's, and as a share of the sessions and of the ANDs they cover.
`figures.py`'s docstring states every figure.

~~~text
python benchmarks/recursion_figures/figures.py RESULT.json          # the table
python benchmarks/recursion_figures/figures.py RESULT.json --json   # the figures
~~~

It sits beside the bench rather than in it because it reads the census and the monitoring periods, which the public
export doesn't copy.
