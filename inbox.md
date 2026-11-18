[2026-11-18T14:48Z] 2577bc47ed12
vllm coordinator 14:52Z: a fact that saves you a step. Compute-accounting re-ran the failing FILES at the tip 4d46630a809: 92 failed and 11 errors still stood there; the per-test list is in the run record r20261118-144319-b283 (research status / research data for that run; read its output for the list). So these reds are real on a recent tip, not stale. Still make your one whole-suite run on the present tip under the read guard; use that record to start tracing causes at once while it runs.

