[2026-11-15T18:23Z] 6f7f2c48aa1a
RECORD to the training lead. Step 1 of the outsider trial, at main 8c82d444d0d. Model: Claude Opus 5.5; no subscriptions or timers held. Your four facts are received; step 3 comes next, ahead of the PR.

HOW I RAN IT (ran). A fresh clone in a scratch directory. `uv sync --all-packages --extra torch-cpu` under `env -i` with only HOME, PATH=/usr/local/bin:/usr/bin:/bin, UV_CACHE_DIR and LANG: no research variables and no store credentials passed, and no elan or lake on PATH. The sync exited 0. The made record is the CPU test's own `made()` (benchmarks/inference_only/tests/test_served_record.py:67), written to disk twice: once with a development accept only, once with Pearl+'s accept of record.

WHAT RAN (ran)
- `verity bench inference_only --stated-only`: rc 0 in under 1 s. It prints "AS SERVED: 1" and says right below that the number is priced from served.toml's stated numbers and that no check ran.
- `verity bench inference_only` (the default form) with no lake: rc 1 after 136 s spent building the class's circuit. It then fails with a Python traceback whose last line is built.py:234: "`lake` isn't on PATH ... install the toolchain leanprover/lean4:v4.34.0 (elan) and run again".
- `--stated-only --served ...` on the made record with no lake: rc 1 with the same traceback, from record.py:185 through built.py:234.
- The same with this VM's elan put on PATH (an outsider who followed the message): rc 0. The first run took 31 s, the `lake build` included, and the second 1 s. Every Lean requirement is a path inside the clone: no Mathlib and no network.
  - Development accept only: "FROM RUN made's RECORD: 1 (measured), no PoUW accept of record, so s = 1". The PoUW line reads protocols.work.verdict = 'missing' and says "a development accept counts as none". Both rules' public forms say complies: true.
  - Pearl+'s accept: s = 33.83% and the tie compared equal. It gives "1 (partial)" on the made class. The PoUW line says the accept is the prover side's, "run on the served node ... no auditor device".
- No step reaches our store, a host or a key. I read the code path for this: nothing under benchmarks/inference_only imports research or opens a socket, and the harness kit only reads optional variables (below).

THE LIST (file:line at 8c82d444d0d; read unless marked ran)
1. README.md:11-12 and metrics.toml:11: the full command is `research run --on auto ...`, which an outsider has no way to run.
2. README.md:14-17: the default form "goes through `research run` in the run's own clone". Ran: it runs locally (136 s, then lake).
3. README.md:10 and :113-116: `--served RUN/out --program P --circuit C --log L0 L1 L2`. Nothing says where a run keeps P, C and the log, or how an outsider gets a served record at all. Only the test's made() writes one (tests/test_served_record.py:67).
4. The README never says lake is needed, or how to get it: --served needs it too (record.py:185, `B.lake_built(CHECK)`), and the toolchain is pinned in verity/protocols/policy/inference_only/verifier/lean/lean-toolchain.
5. built.py:229-235: the lake message is clear and actionable, but it comes as a traceback (a RuntimeError, uncaught in report.py:466-479). In the default form it comes only after 136 s of building (ran).
6. report.py:457-462: the first lines print the stated policy's "AS SERVED: 1" and "Not checked here: this tree builds no circuit of this class, so no check ran on it". This holds even when `--served` gave a record whose check did run (ran: those lines come first, and the record's lines come below them). None of the first lines says the number is a measurement and not an audit (your fact 2).
7. report.py:545: the JSON's `value` is the stated policy's even under `--served`. The record's value is at `record.block.value`, so an outsider who reads the last line reads the stated number.
8. report.py:554-557: `--served` writes no metric, and nothing gives the metric a record (the comment at :554 is the open item).
9. record.py:124: the run's name printed ("FROM RUN made's") is the `run` field of the record's own verdicts.json, not the directory the outsider named.
10. README.md:49-51, settings.py:47-50 and kit.py:497-504: optional environment variables only. `$OUT`, `RESEARCH_RUN_DIR`, `RESEARCH_RUN_ID` (default "local") and `RESEARCH_SOURCE_SHA` (else `git rev-parse HEAD`, report.py:88-92). The command works unset (ran). The README names `$RESEARCH_RUN_DIR` without saying it is optional.
11. kit.py:139-164: host_stamp calls nvidia-smi if one is present. Ran: none on this VM, and there was no error.
12. Out of this benchmark but on the outsider's road (read): `verity bench compute --from RUN` reads only our evidence store, through research's LocalStore (benchmarks/compute/run.py:181-228 and :590). An outsider can't run it.

DOES THE NUMBER COME WITH WHAT IT RESTS ON (ran). Yes for --stated-only: it says the number is priced from stated numbers, lists the premises (H) and (E), and says what is scaled. Under --served the record section prints the run, the verdict field and its value, why s = 1, the check's public verdicts and the inputs. But the headline and the JSON value above it stay the stated number (items 6 and 7).

Next: step 3 (the chain), then the PR on cursor/training-inference-only-outsider-4207. Within your fact 3, I change only argument parsing, how the record is read, and the first-lines sentence. The metric's write sits in the frozen end of main(), so the data-file reader will resolve the record, and I will say exactly what is left for after the kit change.

