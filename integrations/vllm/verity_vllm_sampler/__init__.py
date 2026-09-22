"""Verity x vLLM POC, worker W3 ("sampler").

Owns the shared workload generator (`workload`), the stock production recorder for
stochastic sampling (`production`), the replay-side logits dump (`replay_logits`),
the `sampler.select` relation (`select`), the stopping-semantics relations
(`stopping`) and the N08/N09 negative tests (`negative`).

Everything GPU-related runs on the pod; this package is importable on CPU without
vLLM or torch for the pure-python parts (workload, relations over recorded JSON).
"""

__all__ = ["workload", "select", "stopping"]
