[2026-11-14T06:06Z] e3bc3830a098
## Coordinator, Sat 14 Nov 06:06Z: PRF-490: soundness's yes; the read first, then the refusal

From the proofs coordinator, Sat 14 Nov 06:06Z. Soundness answered on PRF-490 (Fri 13 Nov 9:14 PM and 9:45 PM PST; read both comments whole yourself, their words rule over my summary). This replaces "build nothing before Mon 16 Nov 20:00Z" in my word of 05:46Z; the order of work is now:

1. FIRST, the read `run-of-record-one-window` (my item 2 of 05:46Z), now wanted BEFORE the refusal lands, not by Mon: soundness asks us to confirm on the issue that no honest play of record is a one-window two-stage audit, and to name it if one is. So: every place a two-stage audit of record is played or pinned (`Spot/Run.lean`'s `interiorWindowOk`, `test_two_stage.py`'s stream tests, `benchmarks/two_stage/`'s runner and its kept records, the join's candidate) and, for each, one window or two. A note with that slug as soon as it is read, today if you can.

2. THEN the refusal, handed to you by this word: the two-stage form refuses one window outright (your second shape). Soundness's yes is in and the refusal is permanent; the order check is not built until a deployment needs one-window two-stage audits. A small pull request of its own from `origin/main`, unmarked, `Refs PRF-490`, `Lane proofs`. It still comes after `MemberAtFixed` piece 1 unless you judge it smaller to do first; say which in your next note.

3. In the same pull request or the next, as soundness asks: `SessionCoins.StmtsFit` becomes a refusal in the form's check, not a hypothesis. A statements message with 2^64 or more statements, or a statement of 2^64 bytes or more, is refused at parse; then `StmtsFit` is a theorem from the accept. Size it in the note of item 1 (lines, which records move).

A note with the day of each of 2 and 3 by Sat 14 Nov 20:00Z.

