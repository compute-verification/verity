[2026-11-15T18:41Z] 459fb4feefa2
RECORD to the training lead: the draft is open. PR #10354 on cursor/training-inference-only-outsider-4207, at 7d1e8a96de0. It is linked to TRN-158 with the line `Goal accounting/inference-only-small-model`, and it is not marked.

THE PR (ran)
- `--served RUN` alone takes a run's directory or its out/, and reads the Program, circuit record and log from out/registered/{program.json, circuit.json, log.jsonl}. `--program`, `--circuit` and `--log` still work, all three or none. `--log` also takes one .jsonl file.
- The data file benchmarks/inference_only/served_of_record.toml has a reader (record.of_record) and tests. It names no run. Where it names one, the command reads that record as if `--served` had named it.
- No lake: the command exits 1 before any work, with one line naming elan and the toolchain.
- The first line under the title: "A MEASUREMENT, NOT AN AUDIT: ... an audit runs the verifier and its coins as processes on a machine of their own.", followed by the record directory it read.
- The README is written for an outsider: what it measures, the one command, what may change, what is fixed, and how a cheat is refused. The unregistered weight now has a test: both rules refuse it, the count included, and the value is 1.
- The end of report.main is untouched (your fact 3). So a record named in the data file is printed but writes no metric. Writing the metric from it is a few lines in that end, once the kit change lands.
- Suite: `uv run tools/check/suites.py benchmarks/inference_only --fresh` ran here, where lake is on PATH. 57 passed, slow tests included; the compiled-check test ran; no reads outside the inputs. No `research run` was needed.

YOUR ITEM 2: A RECORD WITH NO PROGRAM (ran and read)
- What it does now: a record with no Program is read, not refused. The value is 1 (measured), and the reading says the record holds no Program, the check reads each template's Definition from it, and the circuit record and the registration name it only by its digest, so the check can't run. The check line says "not run". Test: test_a_record_with_no_program_is_one_and_says_the_check_did_not_run.
- Could the check take the Program from the circuit record or the registration? No (read). The record form compares the record's `program` with the SHA-512 of the Program's descriptor, and the run registration's `program` with the same digest. It then reads each template's Definition, its gates and its row widths from the Program's body (inference_only_record_reference.py, `front` and `templates_of`, around :215-235 and :670-690). The circuit record holds only the templates' descriptor ids and counts. So the bodies aren't there.
- What could work (inferred): rebuild the Program from the model, class and tables, which fix it (served_verdicts.sh says this of DAY_PROGRAM under H1_TIE), and hold it to the registered digest. That is the adapter's code (window_circuit), and the cheaper fix is for the serve to write the file.

YOUR ITEM 3: THE REHEARSAL RECORD (ran on the store's copy, a record not served from main)
- `--served` on its out/ is refused: "the run wrote no circuit record (out/registered/circuit.json)" (ran). The store's record holds no Program, no circuit record and no log in a form the check reads. The circuit record stays in the serve directory, and window-circuit-4-27-2.json is a 184-byte count.
- Even with those files the number would be 1 (read). Its record/verdicts.json PoUW row is `missing`, a development piece: "the window's registration or its audit's summary can't be read (... pearlplus/wK/tie/window/registration.json)". The compose took no accept of record, although pearlplus-lean.json accepts.
- The policy has no k_by_kind (read), so even an accept of record would give "no figure" for the tied class (record.py's NO_KINDS).
- What it does settle (read): the Lean verdict's work.macs = 108,984,795,136 = 4 × the policy's linears 27,246,198,784, so record.py:149 holds, as compute-accounting said.
- Arithmetic only, not the command's output (inferred): were that accept of record, W = 4 × theta_macs = 113,400,348,672 and C/W = 94.21%, so s = 5.79%, and the value would be capped at 1/s ≈ 17.3. compute's 96.08% is over linears + attention + head counted once each; this benchmark's W counts attention and the head twice. Nothing of this goes in the tree.

So the chain on main has three missing pieces, in order:
1. The serve writes out/registered/{program,circuit,log.jsonl}, and under the Pearl+ tie it writes a Program at all.
2. The compose's PoUW row is an accept of record: the tie's window registration present.
3. The policy carries k_by_kind (inference_only.py on main writes it; this rehearsal's tree didn't).

