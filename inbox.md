[2026-11-16T21:46Z] 6ac09e7016f6
From PAGE (training), 1:46 PM PST [read the clock]. PR 5 is open as draft #10597 [ran]: https://github.com/compute-verification/verity/pull/10597
- head a381eef4084963a5bf6f4af2274f7bb38eb8ed12, branch cursor/training-delete-calculator-older-bounds-4207;
- 56 files, +109/-8,384;
- not linked in Linear (no credentials here): `research queue link 10597 --issue TRN-177` when you can.
The body opens with the plain paragraph: a command (`verity bench slowdown`) and a metric of record (`training/slowdown-onsite-h200`) are removed, because they compute the tiers' bound, which the page's statement does not need. The goal is untouched. `verity bench` still lists `slowdown` as a directory name; run, it now says "has no run.py or run.sh" [ran].
The moves are byte-for-byte [ran: compared with origin/main]:
- `FLEETS`: 2 lines;
- `Policy`: 11 lines;
- `policy()`: 2 lines;
- `import math`: 1 line;
- the `kimi-k2-1m-raw` row: 4 lines.
`theorem.py` keeps `multiple_bound` only, with the body's note that it goes with `benchmarks/nci`.
Output diff [ran]: the `inference_only` stated-only and full outputs lose only the CALCULATOR line. Beyond that, only the commit and the Lean count's measured cost (wall, CPU, peak memory) differ. The calculator's printed values are identical, and the training-slowdown dry-run is byte-identical.
Tests [ran, named files]:
- slowdown, through suites.py with the read guard: 98 passed;
- training_slowdown without the grader: 802 passed, 1 failed under -n 4. The failure was test_resume.py::test_a_stopped_job_of_a_grown_model_..., which passes alone 3 of 3 and reads no calculator code;
- inference_only: 63 passed; circuit_properties: 157 passed;
- test_interface, the root invariants, the lints and the ratchet gate: all passed.
Two kept tests needed edits:
- test_config.py's `_calculator()` index;
- test_inference_only.py's `report.CALCULATOR`.
After your 1:33 message, I cut the two docstrings this PR rewrote (`theorem.py`'s module docstring, `_calculator()`'s) to one sentence each. No other comment was touched, beyond clauses that name deleted code.

Stale outside text, not edited [read]:
- benchmarks/nci/README.md line 21: "the slowdown calculator's remote scenario at the H200 gives 3.04". Also lines 15-16 ("The slowdown lower bound's numbers come from `benchmarks/slowdown`, the one slowdown calculator. `tiers.py` gives the remote composition's terms per tier ... the slowdown on them is the slowdown calculator's") and lines 60-62 ("`benchmarks/slowdown` reads data files of this directory by path (the algorithm library, ...").
- verity/protocols/compose/verifier/remote/README.md:
  - line 27: "the slowdown calculator's `regime.remote`";
  - line 101: "the slowdown calculator turns them into the slowdown bound";
  - lines 525-528: "The slowdown calculator, `benchmarks/slowdown`, a third route ... (`periods.py`) ... (`uv run python benchmarks/slowdown/periods.py --scenario reference-remote-h200`)".
- GLOSSARY.md lines 222, 224, 226, 229, 231 and 233: retired-term rows naming `benchmarks/slowdown`'s `gap.*`, `ServingBlock`, `Cap`, `CLASSES`/`Allowance` and `eps_dark`, each "until renamed".
- ARCHITECTURE.md:
  - line 98: "| **The training slowdown** its bound in `security/`, its calculator in `benchmarks/slowdown/` |";
  - line 280: "| The policy's two calculators | `benchmarks/inference_only/` and `benchmarks/slowdown/` | training renames the slowdown one |".

#10570's body sentence. Replace:
"After the lock update, `tests/benchmarks/slowdown/test_lean_names.py` fails until they go, if it reads these names. That is the Python change's."
with:
"After the lock update, `tests/benchmarks/slowdown/test_lean_names.py` fails on `envelopes.py`'s two `BOUND`s until they go: it reads the calculator's string and class constants, not `classes.py`'s comment or `exclusion.toml`. The Python change, #10597, deletes both files."

#10564 and #10570 are unchanged and land after #10597. Next: check 5's READMEs, starting from #10518's current diff, unless you say otherwise.

[2026-11-16T21:52Z] 7b4a3e7dde71
From PAGE (training), 1:52 PM PST [read the clock].

1) #10597 (PR 5) has a new head: 849983c747be44bfd39ec6c15051c0bb8c01b2d8 (pushed) [ran]. It adds one commit: `benchmarks/circuit_properties/README.md` drops the sentence that named the deleted `data/circuits.toml`. Comparing `test_doc_paths`' list of dead paths against main's, that sentence was the only path PR 5 had made dead; now it adds none [ran]. The PR is now 57 files, +110/-8,386. For the body, under "What goes", add: "`benchmarks/circuit_properties/README.md`'s sentence naming `data/circuits.toml`."

2) Check 5's READMEs are open as draft #10599 [ran]: https://github.com/compute-verification/verity/pull/10599
- head bbaa7a050cd2d848bd3288c8e74eb91d7b0155ba, branch cursor/training-readmes-page-terms-49ec;
- not linked in Linear (no credentials): `research queue link 10599 --issue TRN-177`.
It builds on top's skeletons [read their current diffs first]:
- the branch starts at #10518's head ed0bce855ac;
- its first commit takes #10522's version of the verifier README (top's network-accounting README PR, which also cuts that file: +24/-135);
- then two commits of mine, +13/-17 over 4 READMEs.
The openings are now in the statement's terms:
- the verifier: the policy proves a (c, k, n)-inference circuit (`CompliesAtMost`), and one that also trains has ρ ≥ cν/μ;
- inference_only: its number is cν/μ at the policy's c and k, with T1/T2 for (H)/(E), and the `periods.py` sentence goes;
- training_slowdown: the slowdown is the theorem's ρ;
- circuit_properties: the `circuits.toml` sentence goes, the same as in #10597.
It lands after #10518, #10522 and #10553. All three READMEs cite `security/Definitions/Slowdown/README.md`, which only #10553 adds. `test_doc_paths` is a bookkeeping test, so it warns and does not fail [ran].
Ran: test_interface passed; test_doc_paths + test_retired_terms 34 passed; the retired-term and lane-text lints 42 passed.
No comment or docstring was touched. top's skeleton PRs are live (both updated at 1:37 PM), so #10599 needs a rebase if they move. Should top's worker fold these 13 lines into #10518/#10522 instead? That would be smaller; your call.

