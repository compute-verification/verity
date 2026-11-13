[2026-11-13T01:35Z] af2da817227d
TOPCODE, Thu 5:35 PM PST

**The guard is in its own commit (45c881f613), and the route, its tests and the 81M entry are pushed on `cursor/training-topcode-9478`; my entry's comparison is running, and the every-entry comparison launches when it shows the memory to ask.**

Commits on `cursor/training-topcode-9478`, its base GROW's b16502a23d (unchanged since):
- d64f33f928, then 574bab9f04 (a lint rename only): the `norm` rule, beside the record's plain normalisations, with its test. The present tests pass unchanged. RAN: `tests/test_of_record.py` on my VM, 19 pass.
- 45c881f613: the guard `(cfg[k][0] - n) % excess[scale][0] == 0` over the held backward keys, with the detached case as its test. RAN: without the guard, that test fires the rule; with it, the rule doesn't fire. The real entry's counts come from the comparison below; if they break the guard, I'll send you the counts and leave the guard as it is.
- dab4575423: `routes/topcode.py` (route `VERSION` 1, package still 4.24.0), with the packing in `codecs.wire` (`pack_codes`, `unpack_codes`, which refuse a width not a multiple of 8). R is a registered buffer built once in `build`, seeded 1000 + grown depth. The head is `model.top(h, len(model.blocks))`. The backward rounds on its own grid of 2^bits − 1 steps, not through bytes. Its tests, among them: the logits equal θ's bit for bit at the first step (f32 and bf16); R·Rᵀ = I within 1e-6, and the same R at every build; the round trip at 4 to 8 bits; and the tiny trace with no `refused`, `data_dependent` or `carried_scalars`, at `a_bits` = d·bits + 16.
- 14cf2170a1: the 81M entry's reference in `test_entries.py` and its entry in `of_record.toml`, with counted = 3 for now and no `record_only` yet.

RAN r20261113-013251-e432 (smoke run): a pool job takes only `--cwd source`. Torch 2.14.0+cpu loads on the node, and the record fetches with sha256 9dd58783…, as the brief says.

RAN r20261113-013457-5fe2 (running): of_record on my entry at counted 3, and at counted 4 from a scratch copy of the data file, each with `/usr/bin/time -v`. It gives condition 9's evidence of which first steps differ, the `record_only` lines, the guard on the real counts, `a_bits` against 3,088, and the memory for the every-entry jobs.

A finding outside my change. RAN on my VM: on torch 2.14.0+cpu at 4 threads, bf16 SDPA sometimes returns a different value for one tile of the same input (about once in 400 calls); at 1 thread it is deterministic. The suite's conftest pins 1 thread only in pytest's own process, so a job `run.main` starts in a child process uses every core. My slow resume test with its trace on diverged at step 2 for this reason, so it sets `OMP_NUM_THREADS=1` and its docstring says why; it then passed 3 times out of 3. INFERRED: the existing slow resume tests that start child processes carry the same risk. I haven't changed them.

[2026-11-13T01:40Z] 746f01e5b8eb
KEY, Thu 12 Nov, 5:40 PM PST.

**Final report: #9502 is at 9ddabfeafb with every check passed and nothing in flight; the two suites that read the changed files passed at the head with the new test among them, and the head merged with today's main compiles in every proof module that reads either side's changes.**

Branch `cursor/trn147-form-reads-formats-10f7`, head 9ddabfeafb, PR #9502 (a draft; the body is current). #9474's head is b23c853a35 (RAN `git ls-remote`, 01:25Z).

The last check open
- RAN r20261113-010201-8cc1 at 9ddabfeafb (`suites.py verity/protocols/certifier/verifier/network repository --changed ea620a89cf --quick`): `2 suites: 2 passed (0 cached), 0 failed; 16.5 min of test time`. The network suite: 90 passed; `repository`: 352 passed.
- The network suite also showed 90 passed at ea620a89cf, so I read both logs on the nodes (RAN r20261113-012558-2298, r20261113-012604-77c1). At 9ddabfeafb it collected 104 tests, deselected 14, and its line is `tests/test_nci_record_check.py .....`: five passed, the new test among them. At ea620a89cf it collected 103, deselected 13, and that file had four. The one fewer elsewhere is the record difftest's slow test, which `--changed ea620a89cf` no longer takes: 14 passed there at ea620a89cf, 13 now. INFERRED: nothing that slow test reads changed since ea620a89cf, where it passed (r20261112-232245-e582).

Main moved under this area, so I tried the merge
- READ: since this branch's base (5f0c008aa0), `main` (e86d920413) changed 11 proof files under `security/Proofs/Properties/Nci/` (`RecordPrim`, `RecordAligned`, `RecordRows`, `RecordLeafWide`, `RecordFields`, `DecodeKeys`, `LeafWidthsResolve`, `ResolvePassed`, `ResolveRows`, `ResolveWithin`, `BodiesOk/Nonneg`): lemmas two files proved, down to one statement each. None is a file #9502 touches, and `main` changed nothing under the nci package or the network suite.
- RAN: 9ddabfeafb merges with e86d920413 with no conflict. The merge is 8acb2a97eb on `cursor/trn147-main-merge-10f7`, a scratch branch, never a pull request. #9502's branch is unchanged: I did not merge `main` into it.
- RAN at 8acb2a97eb, `verity lean try` on the four modules that between them import every module of the proofs package reading one of those 11 or my three (49 modules), lean exit 0 each:
  - r20261113-012959-cc80, `Properties/Nci/RecordCounts.lean`: `lean exit 0 in 169 s`; it built all 14 changed modules; `axiom probe: no theorems in it`.
  - r20261113-012959-785e, `Proofs/Nci.lean`: `lean exit 0 in 29 s`; `none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound`.
  - r20261113-012959-b95f, `Properties/NetworkScan.lean`: `lean exit 0 in 24 s`; the same line for its 4 theorems.
  - r20261113-012959-1801, `Properties/Remote/Honest.lean`: `lean exit 0 in 37 s`; the same line for its 7 theorems.
- Not run at the merge: the whole proofs package's check and the suites. The candidate's check runs them. Say if you want `main` merged into #9502's branch and the package's check run there first.

Every check, with its run and last line
- At 9ddabfeafb: the two suites, r20261113-010201-8cc1, `2 suites: 2 passed (0 cached), 0 failed`. The ratchet against `origin/main` (e86d920413): `1 change(s), 1 failed`, `adds 3 retired-term site(s)`, all #9474's (`RecordCheckMain.lean` 3 and 15, `lakefile.toml` 57). Locally, `test_nci_record_check.py`: 5 passed.
- At efd15349eb (the option's tree), the served twelve: r20261112-233108-21af, r20261112-233205-6c00, r20261112-233259-86da, each `4 cases, 0 verdict(s) where the compiled command and the reference differ`.
- At ea620a89cf (the diff to the head is the script and its test file alone):
  - the nci package's Lean check, r20261112-232414-cced: `AUDIT: PASS`;
  - the security proofs package's Lean check, r20261112-232445-9cb4: `AUDIT: PASS`, 198512 declarations in 3967 modules;
  - tries, lean exit 0 and the axiom line clean: `RecordInputs.lean` r20261112-232515-59b3 (14 theorems), `RecordRead.lean` r20261112-232515-0fba (16), `RecordTemplates.lean` r20261112-232515-de9b (16);
  - the pre-mark suites, 57 in five groups: r20261112-232245-e582 `12 suites: 12 passed`, r20261112-232311-85ba `12 suites: 12 passed`, r20261112-232338-3ab1 `11 suites: 11 passed`, r20261112-232351-a95f `11 suites: 11 passed (3 cached)`, r20261112-232403-d24c `11 suites: 11 passed`, each `0 failed`.
- Not a check of this change: r20261112-234514-8867, the wrong suite, as my last message said.

What's left
1. Yours: read and mark. #9474 lands first.
2. Mine, if it happens: #9474's head moves (I merge it and run again what its diff touches), or you want `main` merged in.
3. Not built, by your ruling: the script reading `held` from vllm's limits file.

In flight: nothing. Branches of mine on origin beside the PR's: `cursor/trn147-served-cases-10f7` (efd15349eb, the served runs' tree) and `cursor/trn147-main-merge-10f7` (8acb2a97eb, the tries' tree); neither is a pull request. The handoff note is `lanes/training/20261112T2243Z-handoff-KEY.md`.

