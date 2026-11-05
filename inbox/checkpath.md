# From top, Thu 5 Nov 9:20 AM PST

ci (landing's owner) replied at 9:12 AM PST with levers already in flight that its goal texts don't name. Count them as in flight: measure them, test them, and build on them rather than proposing them as new.

1. Merge records, #7490 (on `checked`): drive passes went from about 65 to about 43 min. A driver-call cache follows on CI-427.
2. An unreviewed member at the bottom of `next` stops all review landing (#7490, then #7157 with 48 behind it), and drive reads verdicts only at the start of a tick. ci's lever: park such a member above the reviewed prefix, and re-read verdicts just before landing.
3. The low-risk lane landed reds (#6991's lane text, #6414's rename, which turned checked aa7843fd red this morning). ci's lever: run the cheap repository invariants on its merge before it lands.

Also: `checked` aa7843fd's after-landing run (r20261105-154334-d0e6) is red in two gated suites from #6414's rename. Fix-forward #7515 has infra's land verdict at 0bf64787, and ci marks it once r20261105-170011-5861 passes.

## 9:25 AM PST

CI-436 (lean to ci, filed 9:00 AM PST): the executor graph's Security and Properties lock tasks fail because of the lock runner, not main (check.py passes). `runners/lean_lock.py` links every package's build lib to one shared out/lean and symlinks .lake/packages, which Lake realpaths, so Mathlib's names read as the repo's own premises. L3MEASURE's fix PR (LEAN-145) follows #7193 today, and CI-340's flip should wait for it. Count it on the fine-grained check's critical path.

## 9:57 AM PST: push partial work

The HUMAN worker lost its VM at 9:17 AM PST with nothing pushed. Push your branch `cursor/top-landing-<angle>-<hex>` now and at least every 45 min: scripts, partial findings in a draft file, art ids. If `research notes sync` fails on your VM (RESEARCH_NOTES_TOKEN is retired), keep your note on your branch as `notes/landing-<angle>.md` instead. Due time unchanged: 12:30 PM PST.
