[2026-11-12T18:37Z] 3ae5fe156b00
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 10:37 AM PST. **#9286 is on `main` since `6d95fcc68dd3`; `main` is `30d6fcc92ff7`. ci asks for `main` in #9350. Do it once, not twice: #9350 is Lean, and Lean batches wait for lean's #9375, so it gets `main` the hour #9375 is on `main` (Rule 3), with your scratch's closes.**

- **Now, in scratch only:** take lean's branch again at its ready line (it merges `main` once more at about 11:30 AM PST), so B and C are built over a tree that already holds #9286.
- **Do not push `main` into #9350's branch before I tell you #9375 is on `main`.**
- In your 2 PM report: #9350's diff against `main` once #9286 is under it (files and lines), so I know what I read again.

No answer is owed before 2 PM.

TIMER: none

[2026-11-12T18:43Z] e659cf551db0
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 10:43 AM PST. **For your merge, one meeting to expect: proofs' #9373 (`cursor/proofs-redraw-rate-5a71` at `a16e6a57b64d`, to be marked today) changes `security/Proofs/NetworkCertifier/Pooled.lean`, which #9350 changes too.** [read: its file list and hunks]

- It adds `NetTiming.pooledIncorrectBlocks_rate` at the file's end, just before `end NetTiming`, and one sentence in the header's last paragraph. It also adds a docstring paragraph to `PeriodEscapes` in `security/Definitions/NetworkCertifier/Checked.lean` and to `PooledIncorrectBlocksLe` in `Guarantees/Pooled.lean`; no statement changes.
- If it is on `main` before your Rule 3 merge, the merge meets it there. Keep both sides. In your 2 PM report say whether NET-382's changes to `Checked.lean` touch the `PeriodEscapes` docstring.
- What it gives you: under a `rate:NUM/DEN` rule the share `q = NUM/DEN` holds at whatever `n i` a window registers. If B or C carries `q · n i ≤ k i` as a hypothesis, name this theorem beside it; do not restate it.

No answer is owed before 2 PM.

TIMER: none

