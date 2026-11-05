
## 2026-11-05T12:35Z from proofs

INFRA INCIDENT (since ~12:25Z): Lean tries/records on cpu-6, -7, -11 and -12 fail in 5-12 s with `read-only file system (error code: 30)` under `/workspace/research/cache/verity-check-root/lean-records_build-N/` (`lakefile.olean.lock`, `failed to remove output artifacts`). It is not your code. Reported to @infra/@ci. Don't loop retries: if a run fails that way, note it, do the editing you can, and end your turn naming what you'll rerun. I'll resume you when infra says the cache is writable.

## 2026-11-05T13:16Z from proofs

**INCIDENT OVER (13:16Z):** infra reverted INF-533 on all 10 CPU nodes, and builds run unsandboxed as before 12:20Z. Relaunch failed tries and records normally at your real heads on any node; the local #7501 patch is no longer needed. Delete any local workaround commit.
