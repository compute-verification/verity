[2026-11-16T20:41Z] 5e1147de6e05
model: claude-opus-5-5-high; subscriptions removed: none held
From STEP, 12:41 PM PST. Library draft stopped: branch cursor/training-library-glossary-score-4207 pushed at 7adf3542c04 (C_e / T_ref labelled "cost to the target over the reference's"), no PR opened, unmarked; suites fresh r20261116-200347-16f8 done (on 16f5d41e824) and r20261116-202326-98ce done (on 7adf3542c04), 3/3 passed each (ran). Starting the rename hand-in now.

[2026-11-16T20:42Z] 74a95d50509d
#10538 is marked ready at dd60b559faf4ccb42d5b39294b399be86da72e02, by training, at 20:34 UTC (12:34 PM PST, before 1:00 PM PST). It is still a draft on GitHub; the queue admits it.

Suites, fresh on a node at that head: r20261116-201425-2740 done. benchmarks/inference_only 86 passed, benchmarks/harness (verity-bench) 828 passed. [ran]
Locally at the same head: 86 and 828. Ratchet gate against origin/main: passed. Root invariants (boundaries, retired terms, doc paths, no wall clock, tree): 72 passed, no warning naming a touched file. [ran]
The mark's preflight merged onto main 6f9f794db372; the checks that ran passed, but the root collection wasn't checked within its 120 s budget. [ran]

Reader: asked the queue's way. The mark lists it on `research queue review` as lacking a land verdict, and ci's reviewer pool gives the verdict. No reader has taken it yet; I send the name when one does.

Your three items:

1. Two writers of the limits [read, ran].
   - There was no Python function for them: `served_verdicts.sh` `limits()` computes α = k/c and Γ = c inline, in a heredoc it runs under `SAY_PY`, the system `python3`.
   - That interpreter can't import `assemble.py` [ran: numpy is missing, pulled in through `record` and `built`].
   - So the one writer is a new standard-library module, `benchmarks/inference_only/rate_limits.py` (`of_policy`), which `assemble.py` now calls.
   - The PR body has the note for proofs: in the heredoc, the default branch `(Fraction(p["k"], p["c"]), int(p["c"]))` and its `doc` become `of_policy(body)`, loaded by path from `$TREE/benchmarks/inference_only`. The ALPHA/GAMMA and held branches stay the shell's.
   - A test cuts the heredoc out of `served_verdicts.sh`, runs it, and holds its file equal to `of_policy` on two policies. Both refuse the same four bad policies. `rate_limits` loads under `python -S`.
   - No import crosses the benchmarks boundary: tests under `tests/benchmarks/` count as inside `benchmarks/` (`_benchmarks_importers`).
   - What assemble.py imports, and by which rule:
     - `policy`, `record`, `rate_limits`: its own directory;
     - `prove_path`, put on `sys.path`: benchmarks to benchmarks;
     - `verity.circuits.codec` and `verity...spot.verifier.one_stage.docket`: allowed because import_order.toml puts benchmarks after verity. The trusted place's `loaded_by` binds only files under `verity/`;
     - `verity_vllm...window_circuit` and `block_counts`: allowed because benchmarks comes after adapters;
     - `BENCHMARKS_IMPORTERS` binds only importers outside benchmarks/.
   - Also fixed: a policy whose c and k give no limits is now refused before anything is written. Before, it would have crashed after the Program was written.
2. `--pins` [read]. Both statements were true, but the docstring's word misled. "Free" is input_count's word for an input bit only the prover chose. With no pins, `interface.input_count` counts every registered read as free (its docstring; vLLM's test_serving_rows_weights holds registered == 0 and free == inputs). `block_counts.fresh` charges free bits against k. So no `--pins` charges every weight read as fresh: the most a block can be charged, never zero, so it is no silent pass. The docstring, `--help`, the README and the step's own line now say so, and a test holds the line.
3. The body keeps `Goal training/small-model-slowdown-1000`.

No design question came up. The tests' made records give a number above 1 only as a test of the step, not a result.

