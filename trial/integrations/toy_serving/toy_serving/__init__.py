"""A toy serving engine that plays the developer in `verity.protocol.epoch.run`, the way vLLM would.

What it owns is what only an engine can: where the hooks go, running the work, keeping each execution's trace, and its
account. What it does not own: the phase order, the challenges, the checks, the configuration's refusals.
"""
