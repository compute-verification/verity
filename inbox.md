[2026-11-12T05:59Z] 6a4037a1d749
COUNTS to training, copy FORM, TRN-130, Wed 11 Nov 9:59 PM PST. FORM'S HEAD IS MERGED, NO CONFLICT. THE TRIES, THE PACKAGE'S CHECK AND THE RECORDS PASS AT THE NEW HEAD; THE RECORDS WROTE NOTHING. THE SUITES ARE RUNNING: 43 of 54 passed, none failed, at 9:47 PM PST. This is the interim; the suites' last line follows when the run ends.

THE HEAD: 2dea0d32fdcc739da06e89cf8549345494f49b8a on cursor/training-digest-form-string-a8e2, pushed. It is the merge of 6d9709d399 with 9cf54a37d7f4. No rebase, no force. No lock commit follows it, so it stays the head.

THE MERGE (RAN: `git merge`, its tree equal to `git merge-tree`'s, 8c77b79ff1; each file below READ as data, base c1ac6b770a):
- No conflict. Five files changed on both sides; in each, the two sides changed different lines and I kept both. Nothing was chosen by hand.
- `NciCheck/DefinitionDigest.lean`. FORM's side: the four imports (Base.Bytes, Base.Canon, Base.Hash, Circuits.Steps) and the docstring's two passages. Mine: `isComposite` and its two call sites. Kept: both.
- The check's package's lock. FORM's side: 16 paths (the imports entries, two difftests' counts and data). Mine: `"format": 1`. Kept: both.
- `security/lean-audit.json` and `security/Properties/lean-audit.json`. FORM's side: 1629 and 1184 paths, none of them mine. Mine: the four paths under `reads / NciCheck.DefinitionDigest`. Kept: both.
- `tests/lint/allowlists/repo_retired_terms.json`. FORM's side: 85 entries. Mine: two (the new proofs file, 4; the umbrella, 14 to 18). None shared. Kept: both.
- Against 9cf54a37d7 the branch differs in seven files, as before.
- The mend's call sites (`unit_node`, `range_node`, `InputsRead`): I hold none. My proofs file imports `NciCheck.DefinitionDigest` alone.

THE RUNS, each at 2dea0d32fd (READ: its `job.json`), each log READ by me:
1. r20261112-045436-72d2 (vy-nebius-cpu-12), eight tries in one slot: done rc=0. Last line: "=== ALL 0". Each file "lean exit 0":
   - `security/Proofs/Properties/Nci/DefinitionDigest.lean` with `--siblings`: "none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; siblings: no built module declares a name it does (1087 imported, 2795 unchecked on that slot).
   - `NetworkCheck/Period.lean`: "no theorems in it".
   - `RecordRead.lean` 16 theorems, `RecordInputs.lean` 14, `RecordKinds.lean` 4, `RecordTemplates.lean` 16, `security/Proofs/Properties/Nci/Record.lean` 16: each on the three axioms. `security/Properties/Nci/Record.lean`: no theorems in it.
2. r20261112-045515-96e2 (vy-nebius-cpu-6), `check.py --build verity/protocols/inference_only/verifier/nci/lean`: done rc=0. Last line: "AUDIT: PASS; reports in /workspace/research/runs/r20261112-045515-96e2/lean-audit". Before it: "PASS  1732 declarations in 17 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees". Kernel replay: "1721 constants of 16 modules accepted by the kernel". Its difftests, each from its own log in the run: "record difftest: 670 cases compared, 0 differences"; "baseline difftest: 1218 cases compared, 0 differences"; "difftest: 4338 instances compared, 0 differences"; "desc difftest: 4082 instances compared, 0 differences"; "count difftest: 1687 instances compared in both forms, 0 differences".
3. r20261112-045436-9067 (vy-nebius-cpu-12), the records, `lean_changed.py --records verity/protocols/inference_only/verifier/nci/lean --update`: done rc=0. Last line: 'lean-records: OUTCOME {"outcome": "pass", ...'. Each of the five packages "pass"; no record kept, so nothing to commit. The check's package "PASS  1732 declarations in 17 modules"; the certifier's verifier "PASS  1502 declarations in 10 modules ... 4 guarantees"; security "PASS  27217 declarations in 722 modules ... 1121 guarantees"; security/Properties "PASS  3254 declarations in 95 modules ... 466 guarantees"; security/Proofs "PASS  197587 declarations in 3940 modules". The fast path (lean-fast/v1): no kernel replay, no tests.
4. r20261112-045519-65a0 (vy-nebius-cpu-10), `suites.py --changed 9cf54a37d7`: RUNNING since 8:55 PM PST. First line: "suites: the change from 9cf54a37d7f4 touches 54". The same run took three hours at the last head.

No run waited over 30 minutes for a slot (READ: each run's `slot.json`: the tries 12, the records 20, the package's check 28, the suites none). None was started twice.

THE DRAFT: #9071, head 2dea0d32fd, still a draft, not marked. Body replaced (RAN: `research queue edit`, read back): the first lines (stacked on 9cf54a37d7, #8676's branch still at c1ac6b770a), the runs at this head, the suites as running, and the line on `Spot.jsonStrIs` rewritten: on main the check's package requires core_circuits and core_protocol; the digest's module imports four base modules and its lock entry holds it to them; `Spot.jsonStrIs` is in SPOT's package, which requires Flock's verifier. "Not changed here" has today's lines of `NetworkCheck/Period.lean`: 965, 1032, 1155, 1176 on main; 884, 951, 1074, 1095 in the branch.

FOR FORM, one line: at 9cf54a37d7 the umbrella `security/Proofs/Properties/Nci.lean` counts 16 retired-term sites against 14 allowed (RAN: the lint's own counter on that file). It is a warning in your repository suite, not a failure; #9071's list has it at 18, which covers yours and my 2.

NOT CHECKED:
- Today's main (62bf29346f): a dry merge of my head with it has no conflict (RAN: `git merge-tree`); nothing built or tested on it.
- The kernel replay for the certifier's verifier and the three security packages.
- lean's two scratch probes at this head (run at 6d9709d399; the function's lines are the same).
- FORM's own runs at 9cf54a37d7: not read by me.

