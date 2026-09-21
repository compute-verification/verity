"""research: shared tooling for executing, observing, and coordinating computational work during development.

Three user actions: run work (`research run`), inspect evidence (`research status`, `research inspect`), coordinate resources
(pods, budget, guards).  Internally: `research.telemetry` (the recorded-attempt harness: job/status/events/resources/failure),
`research.pods` (RunPod lifecycle, budget cap, memory guard, pod-side shell guards), `research.runs` (run ids and the local registry).

`research` imports nothing from Verity's `packages/`; it runs commands and records what happened.
"""
from __future__ import annotations

__version__ = "0.1.0"
