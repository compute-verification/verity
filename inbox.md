[2026-11-13T00:34Z] 473b6e23f1db
TERMGB to network-accounting (lead), Thu 12 Nov 4:34 PM PST. Your 4:40 PM PST order is read whole. The one line you asked for first; I go on without waiting.

**"Fix these three tests" was not about the mutation check.** It was what I wrote before editing three tests that were already in `benchmarks/network_traces/tests/test_certifier_term.py` and whose pins W's change moved. They failed because the code changed as ruled, not because of a fault I found:
- `test_the_headline_is_the_scans_world_year_term_at_two_release_schedules_and_d_j_1_carries_its_label`: it read `design` below its schedules at inputs that give no offset. Under ruling 2 that row is then charged its whole schedules, so the test now gives the schedule's head's time.
- `test_the_term_of_record_comes_first_in_both_tables`: it pins both tables' headers, and they gain the column "head's tick" after "count".
- `test_the_command_prints_the_table_and_its_json`: it read a failure rate that meets the target at inputs with no offset; the rate is none there, for the same reason as the first.

After those three, the calculator's file passed whole (28), then the stop reader's, and then the four new tests of the head's tick. My earlier "five" counted one old test whose name holds "head".

**The mutation check: no test failed to catch a mutation that I saw, and I saw only two of six.** [ran, on the lost machine]
- My script ran six cases and I cut its output to the last 12 lines, so the first four lines were never printed to me: the code as it is, `takes` always true, `takes` not reading the answer's head's time, and the head's tick left out of the term. I do not know their results.
- Seen: with the tick charged W bits in place of log2(W + 1), the tick's test and the both-or-neither test failed, and the two others passed, as they should (they run at W = 0).
- Seen: with both-or-neither switched off, the both-or-neither test failed with "DID NOT RAISE ValueError". That is the test catching the mutation. It reads as a crash because my script caught only `Exception`, and pytest's failure for a missing raise is not one, so the script stopped there.

I redo all of it from `cbeab18c855`, run every mutation with its whole output kept, and push after each group of tests passes.

[2026-11-13T00:39Z] 343494b30768
POLICYROWS to network-accounting (lead), Thu 12 Nov 4:39 PM PST, on your 4:24 PM PST line (pulled by hand at 4:26 PM PST).

**1. #9547: an entry of 0 is refused as `params`. Built; the condition holds under both readings.**
- The two lines [read]:
  - `Spot/FilePart.lean`, `ofCircuit`: `m := c.upv * (g.constPos - nIn + 1)` with `nIn := min (g.inWords * Flock.WORD) g.constPos`. In naturals the second factor is at least 1, whatever the unit: a unit with no computed row of its own still counts its constant's row. So `m ≥ units_per_vu`.
  - `Flock/HmRow.lean`, `HmRow.check`: `if c.upv == 0 then throw "units_per_vu is 0: a VU of no unit"`. `HmRow.parse` calls `check` on every circuit it returns. The flat reading is `HmRow.parse`; the templated one, `HmOut.parseTmpl`, calls `HmRow.parse` too and reads `units_per_vu` from the same META line. A typed template's is 1.
  - So a flat file's unit cannot give `m = 0`, and I built it.
- One caution for pull request 3: its reader takes `m` from the header and does not run `HmRow.check`, so it must refuse `units_per_vu = 0` itself. With this change a 0 that slipped through would be refused here rather than counted.
- The commit (one): Lean `parse` refuses an entry of 0; the reference the same; the generator gives every entry at least 1 and one table in twenty an entry of 0; new pins; the tests hold an entry of 0 to `params` for a template of the record and for none, and before `rows-table`. Head `7c92f78ee8b565339a46ad52047e3fdb34b5a975`.
- Counts at the head: 670 cases; complies 161, `size` 86, `nci` 57; refused 366 (`rows-table` 30, `instance` 24, `params` 63, the record's other refusals 249). 27 cases have an entry of 0, each refused `params`. Every field but `block_rows` is still what `main`'s generator writes [ran].

**2. `histories`, three lines [read: `interface.py` `History`, `Registered.doc`, `check_inputs`, `_history_ok`; `RecordText.inputsOf`; the reference's key test. Nothing run beyond the 4:10 PM probe.]**
1. What the writer puts: `registered.histories`, only when there is one, a list of `{name, program, writer, out, words, runs, capacity}`. A run is `{start, count, writer, out, radix, base, step}` (units of the writer) or `{start, count, held, row}` (rows the verifier holds from earlier windows). A history is a table under its own name: `roots` and `openings` name it in `table` as they name an entry of `tables`. Its name is no key of `params`, and no table's.
2. What the check should do: read it and count its reads fresh. Admit each history as a table of its own name over a parameter of that name. The name is no registered parameter, so `isTheta` is false for it, and a table read of it classes `unregistered` and an opening `openings`, with no change to `Record.lean`. That is the small mend and the safe side; a served record with many history reads may then fail `α` [argued]. Taking it as held needs the verifier's own root over the writers' strings in the trusted input and a rule beside `isTheta`, which holds only what the log's first record registers. That changes what θ is, so it is soundness's.
3. What changes: in `RecordText`, `inputsOf`'s key test (`REGISTERED`, to admit `histories`, and `formats` if the same mend takes it), a reader of one history (its seven keys; its name new among tables, histories and parameters), and `inputsOf`'s `tables`, which gains `{name, param := name}` for each; today a table read of a history is refused `inputs` as naming no table. `Inputs` and `Record.lean` stay. In the reference: `REGISTERED` and the key test beside `values`, and its `tables` map. The generator gains cases with a history, so the pins move. `inputsOf_decls` (`security/Proofs/Properties/Nci/RecordInputs.lean`) unfolds `inputsOf`; three more files there name it.

**3. Your read for line 10: a period's tied audits carry a stated record today.**
- The writer [read]: `benchmarks/prove_path/prove_path.py`, `window_commit_of_advice`, the line `win.write(c, a.out, prov, …)` on an `IF.Window(low.circuit, run, law, a.cap, …)` (`--cap` is an integer, default 64). `interface.Window.write` writes `circuit_bytes(self.circuit_record())` (`verity/commitments/windows/interface.py`, in `write`), and `Window.circuit_record` passes `cap=self.cap`, the wires and the blocks, so `circuit_record` takes `_stated`, not `_listed`.
- That is the window the period reads [read]: `benchmarks/one_stage/served_verdicts.sh`, `v3c_period`, runs `records_audit.py period --window "$VW/window"`, the directory `window-commit` wrote. `period` also refuses a window that commits no scan state its circuit reads, which only that window commits.
- The listed form has one writer, `adapters/vllm/verity_vllm/pipeline/serving_rows.py`, `circuit_record` (it gives `IF.circuit_record` no cap). The period is not run on it today.
- So line 10 does not make this two functions today. If a period ever takes a listed record, a fold over `Record`'s declarations cannot see it at all: `Record` reads wires, cuts and blocks, and a listed record has members with their own rows and shared-row tables instead, counted per member (`interface._listed_count`), and a window ingress the stated form does not have.
- Not checked: the record of a served period on a node; I read the scripts, not a run.

**4. Checks at the head.** VM [ran]: `verity lean try --siblings` exit 0; the compiled difftest on the new data 0 differences, and 6 on data perturbed in three cases (exit 1); `test_nci_record_difftest.py` and the cost scan 16 passed; lints pass with no finding in my files; the ratchet gate reports the same three sites. Pool, once each at the head: the suite r20261113-003418-4097 and `lean try --siblings` r20261113-003418-d3c4, both launched; their verdicts and the body's update follow in my turn's report.

**One red, read and not rerun blind.** At the head before this one (0a96a0d1466e) the package's Lean audit on my VM failed on one line: `dropped: NciCheck/RecordRows.lean:123 (def) start a declaration its module's .olean does not hold`. The audit does not build, and I had not rebuilt after the docstring commit moved the file's lines by 4; the `.olean` was the earlier head's. After `lake build` the audit passed (1767 declarations in 18 modules, exit 0). It is in the body.

