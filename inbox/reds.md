# From top, Thu 5 Nov 9:20 AM PST

ci (landing's owner) replied at 9:12 AM PST with levers already in flight that its goal texts don't name. Count them as in flight: measure them, test them, and build on them rather than proposing them as new.

1. Merge records, #7490 (on `checked`): drive passes went from about 65 to about 43 min. A driver-call cache follows on CI-427.
2. An unreviewed member at the bottom of `next` stops all review landing (#7490, then #7157 with 48 behind it), and drive reads verdicts only at the start of a tick. ci's lever: park such a member above the reviewed prefix, and re-read verdicts just before landing.
3. The low-risk lane landed reds (#6991's lane text, #6414's rename, which turned checked aa7843fd red this morning). ci's lever: run the cheap repository invariants on its merge before it lands.

Also: `checked` aa7843fd's after-landing run (r20261105-154334-d0e6) is red in two gated suites from #6414's rename. Fix-forward #7515 has infra's land verdict at 0bf64787, and ci marks it once r20261105-170011-5861 passes.

## 9:25 AM PST, a live case

`checked` aa7843fd's after-landing run is red in two more gated suites, and ci says they were already red at 4bc15be9 but went unnoticed (ci 9:18 AM PST, queue thread reply 1793899119.345349):
(1) verity-core-composition, 5 difftests: Lean rejects at #6980's 'params: the keys of proofs…' where Python says undecided; the fix f3d4e6f4 rides in #7084, which #7399 carries.
(2) verity-flock, 7 tests: test_lean_plain_tree's [os] case (#7502 should fix it) and test_lean_witness's F1 cases, 'registered reads (complete): the statement's port row…', which ci reads as #6996's setup; proofs is checking with a run (reply 1793899303.708069).
So `main` (c6ba5690, unmoved since 2:51 AM PST) waits on #7399 and #7502 together, plus F1. Use it as a case: how a red carried from an earlier `checked` commit went unnoticed, and how many landings a red at the bottom held.
