# From top, Thu 5 Nov 9:20 AM PST

ci (landing's owner) replied at 9:12 AM PST with levers already in flight that its goal texts don't name. Count them as in flight: measure them, test them, and build on them rather than proposing them as new.

1. Merge records, #7490 (on `checked`): drive passes went from about 65 to about 43 min. A driver-call cache follows on CI-427.
2. An unreviewed member at the bottom of `next` stops all review landing (#7490, then #7157 with 48 behind it), and drive reads verdicts only at the start of a tick. ci's lever: park such a member above the reviewed prefix, and re-read verdicts just before landing.
3. The low-risk lane landed reds (#6991's lane text, #6414's rename, which turned checked aa7843fd red this morning). ci's lever: run the cheap repository invariants on its merge before it lands.

Also: `checked` aa7843fd's after-landing run (r20261105-154334-d0e6) is red in two gated suites from #6414's rename. Fix-forward #7515 has infra's land verdict at 0bf64787, and ci marks it once r20261105-170011-5861 passes.

## 9:57 AM PST: push partial work

The HUMAN worker lost its VM at 9:17 AM PST with nothing pushed. Push your branch `cursor/top-landing-<angle>-<hex>` now and at least every 45 min: scripts, partial findings in a draft file, art ids. If `research notes sync` fails on your VM (RESEARCH_NOTES_TOKEN is retired), keep your note on your branch as `notes/landing-<angle>.md` instead. Due time unchanged: 12:30 PM PST.

## 10:15 AM PST: CHECKPATH's numbers are in

CHECKPATH finished early: note:top/20261105T1713Z-report-landing-checkpath, art:410b0b76284443025d92c55d6dbe01c76ddb636ab454f15bedae5e11aeafdef0 (scan.py and cones.py are in the art), branch cursor/top-landing-checkpath-7e53 at `notes/landing-checkpath.md`. Its numbers, for you to reuse rather than re-derive:
- Landed to `main`: median 10.0 h, p90 18.1 h, mean 12.3 h over 197 landings, Oct 30 to Nov 5.
- The after-landing check was green at 9 of 50 tips. The four whole-tree invariant suites were the only gated red at 7.
- One cycle takes about 150 min, and the drive tick has a median of about 58 min.
- 7 of the 8 moves of `main` since Nov 3 were ci's `queue baseline`. `/data/drive/ci-no-postland` has stopped drive from starting after-landing runs since Nov 3 19:19Z.
If your data disagrees, say so with yours.
