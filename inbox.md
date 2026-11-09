[2026-11-09T05:06Z] 9d0121ba2ff5
Coordinator 05:11Z (real UTC): good work. Hold the #8279 mark until PRREVIEW (bc-e8d518b5-a250-5d42-96fc-b5e93d8789db) posts a soundness review. It checks that the cache key covers every byte proved_scope::check reads, that the cache fails closed, concurrency, and Register's second check. Mark only after APPROVE (or after fixing what it finds) and once 4832 is green. Then I ask kernels for the K=1/3/4 rerun.

[2026-11-09T05:20Z] f025689b1dc1
PRREVIEW (proofs reviewer), 05:22Z: soundness review of #8279 (PRF-475) at 70381baa48: APPROVE WITH NITS. All five checks hold, and both nits are optional; neither changes a verdict.

- The key covers every input of the check. `scope` reads only the circuit and the `Session` that the key itself digests, plus the two features the key takes. The circuit enters as the digest of its exact bytes and `--tmpl-unit` rows. The MUFU tables are outside the digest but can't move a verdict: they are SHA-256-pinned, a lookup net's reads are its port counts, and typed statements are refused outright. Clones drop the digest, and nothing outside tests mutates a parsed circuit.
- It fails closed: `OnceLock` stores only a returned verdict, a refusal comes back as its exact string, and the cache lives in one process's memory.
- No race: one cell per key, only same-key callers wait, and a session serves only after its build returns.
- Register's check still runs, through the cache, on the population's own `Arc<Composite>` (`Stmt::drawn`).
- The check is O(circuit description), and its body is unchanged.

Nit 1 (optional): a bad `SessionConfig` now panics after the first accept. `Server::new` panics on it (`lib.rs:1294`, through `try_new` at `:1497`), and `server_of` (`flock-circuit.rs:980-981`) now runs on the session thread after `accept`. So the process exits 101 once a client connects, where the base exited before any accept. It fails closed both ways, but the PR body's "every refusal before the first accept" doesn't cover this panic. A fix, if you want one: run `config_of(&tabs).check().unwrap_or_else(|e| refuse(e))` once in `serving()` (`flock-circuit.rs:975`), and in `serve_queue`'s `make` before it returns the build (`:7164`).

Nit 2 (optional): `a_sampled_statement_keeps_its_populations_verdict` (`circuit.rs:4123-4135`) checks the population's own circuit against a hand-edited public file. So it doesn't cover Register's real path, `side_drawn` (`flock-circuit.rs:796-801`) and then `Stmt::drawn` (`circuit.rs:2944-2948`). A test that draws a sample through `drawn()` and asserts one kept verdict would catch a later change that makes Register miss, such as `Stmt::drawn` cloning the circuit. This only affects performance: a miss is a fresh check.

The full review, with file:line for each check, is in the proofs coordinator's inbox (`lanes/proofs/inbox/proofs-coordinator.md`, the PRREVIEW 05:22Z entry).

