# FIRSTACCEPT: the first audit a driver writes that the compiled check of a whole audit accepts

Read `COMMON.md` beside this file first, then `## PROVED, Thu 12 Nov 21:01Z` (sections 4 and 5) and its correction at `21:04Z`, and `## PERIODREAD` entries of Thu 12 Nov `18:13Z`, `19:23Z` and `19:40Z` (the run `r20261112-185813-3849`).

**The gap.** For an accept to count as fully proved, one program has to say it: the compiled Lean check of a whole audit (`Spot.OneStageZk.checkBlobs`, `verity/protocols/spot/verifier/lean/Spot/OneStageZk.lean`, run under the composite's `remote check` or `onsite check`). On `main` it has no accept on record, in a test or in a run. The accept at the end of a served audit is put together by the Python driver from the Lean verifier's answers to single steps (`benchmarks/one_stage/a5.py`, `service.Stream.outcome`). The case the driver writes for the compiled check (`a5.py`, `write_case`; #9416 on `cursor/proofs-prf600-case-names-7d2e` makes it name its challenger) is refused: the one proof session's coin tree is not one the window's challenger session sampled (`Spot/ChallengerTrees.lean:254`), and `--challenger local` makes a key at small LMS parameters, which the compiled check refuses (`KEY_PARAMS`, `OneStageZk.lean:132`).

**Yours.** Make the smallest real one-stage audit of a one-member window end in the compiled check's accept, and keep it as a test and a kept run.
1. Read exactly why the compiled check refuses today's case, step by step, from the code and from `r20261112-185813-3849`'s files; list every refusal on the way to an accept, not only the first.
2. The driver change: a one-member window's proof session run on the window's challenger session, without the closing round (`a5.py:1398-1404` asks for two members or more for it; `:913-931`), under a key at the deployed parameters. PROVED's guess: 3 to 4 files of Python and Rust, no Lean. If an accept needs a Lean change, stop and write it up: a change to what the compiled check accepts is not yours to make.
3. A test that runs the compiled check on that case and expects `accept` (today every test of it expects a refusal, and `benchmarks/one_stage/tests/test_composite.py` puts a stand-in executable in its place), and one kept run through `research run` whose files anyone can replay the check on.
4. Then the record: no field the driver writes names the tree's commit or the verifier executable's SHA-512. Add both to the composite's record, and a check that the commit's lock lists the theorem the accept cites (under 100 lines of Python by PROVED's guess).

**Not yours.** The several-member form (the workers MEMBERS and WINDOWAT). The two-stage form. The period's check (PERIODREAD). `v2_case.py` writes a case naming no challenger: say what it needs, do not widen your change to it unless it is a few lines.

**Base.** `origin/cursor/proofs-prf600-case-names-7d2e` (#9416's head, today `76aaa72e7799`, marked and waiting to land), merged with `origin/main` if it is behind.

**Owed.** 20 min: first entry. By 23:30Z Thu: the list of refusals between today's case and an accept, each with its line and what removes it. By 03:00Z Fri: the driver change pushed and a run launched. By 15:00Z Fri: the accept on record, or exactly which refusal is left and why.
