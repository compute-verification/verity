# Sourced by a shell job that starts the Lean verifier itself. `$(verifier_env) VERIFIER verify ...` starts it as
# proofs.py's `verifier_env` starts a Python caller's: under the verifier's allocator setting (`VERIFIER_ENV`) where the
# job's environment sets no value of its own, and under the job's own value where it does. The words go before the
# verifier's path, after any `exec`, `time` or `perf record`, which then run `env` and through it the verifier.
verifier_env() { echo "env MIMALLOC_PURGE_DELAY=${MIMALLOC_PURGE_DELAY--1}"; }
