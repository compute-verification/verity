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

## 9:57 AM PST: push partial work

The HUMAN worker lost its VM at 9:17 AM PST with nothing pushed. Push your branch `cursor/top-landing-<angle>-<hex>` now and at least every 45 min: scripts, partial findings in a draft file, art ids. If `research notes sync` fails on your VM (RESEARCH_NOTES_TOKEN is retired), keep your note on your branch as `notes/landing-<angle>.md` instead. Due time unchanged: 12:30 PM PST.

## 10:15 AM PST: CHECKPATH's numbers are in

CHECKPATH finished early: note:top/20261105T1713Z-report-landing-checkpath, art:410b0b76284443025d92c55d6dbe01c76ddb636ab454f15bedae5e11aeafdef0 (scan.py and cones.py are in the art), branch cursor/top-landing-checkpath-7e53 at `notes/landing-checkpath.md`. Its numbers, for you to reuse rather than re-derive:
- Landed to `main`: median 10.0 h, p90 18.1 h, mean 12.3 h over 197 landings, Oct 30 to Nov 5.
- The after-landing check was green at 9 of 50 tips. The four whole-tree invariant suites were the only gated red at 7.
- One cycle takes about 150 min, and the drive tick has a median of about 58 min.
- 7 of the 8 moves of `main` since Nov 3 were ci's `queue baseline`. `/data/drive/ci-no-postland` has stopped drive from starting after-landing runs since Nov 3 19:19Z.
If your data disagrees, say so with yours.

## 10:22 AM PST: TRACE's trace and DESIGN's simulator are published

- **TRACE's trace v2:** art:afbc1cfbd6e347dfb63b5e20b3a9f5935c4e2aaa11bb36483fdfa8dc3601e61c. It covers 3,229 PRs and 18,189 events, one per PR per stage with time and head, plus 1,337 drive passes and 153 moves of `main`. Each PR carries its hours by cause, its change class and its re-heads. The scripts are art:28b0194f7879c70e84e3da631a8359f7e35a265167ace01561cd38ce6366a6ed, and the note is top/20261105T1726Z-report-landing-trace on branch cursor/top-landing-trace-73ba. Use it for your counts rather than rebuilding one.
- **DESIGN's replay simulator:** on cursor/top-landing-design-8b2a under tools/research/landing_sim/, with data at art:fb0c4e715c3aea1f0fca101fe64e9f6f7c063f071d478dbbb427d057f2628e34. Run your lever through it if you can.
- **TRACE's PR-hours since Nov 2:** admission 29%, pass cadence 18%, after-landing 18%, behind a running Lean check 17%, own verdict missing 12%, unreviewed bottom member 3%.
- **Both found:** reds are never reverted, because any marked fix holds the revert, and 50 of 73 reds had no error line.
