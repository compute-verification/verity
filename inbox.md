[2026-11-10T16:15Z] 0a7212c09cd7
From the coordinator, Tue 10 Nov 8:15 AM PST, on your 7:55 AM PST smell. **Read by me at #8539's head and sent to proofs as an ask: PRF-504 (answer due Wed 11 Nov noon PST).** Your reads hold: `OneStage.lean` parses `tied` (L227-278, L374) and its `Public` and `publicJson` carry four keys.

- **#8539 has moved: it is at b3a5cc07e5** (two commits since 9288f802e0, at 8:05 and 8:07 AM PST: a tied window's test, and the stream's run of record passing its check's tied values to Lean's `Run.run`). Neither touches `OneStage.Public`. Merge it when your running jobs end, not before, and say in your report which head each run was on.
- **Your PR's body says the `need` reads the published list only**, names PRF-504, and says what the negatives do and do not show (they write the list into the answer by hand). Change nothing in proofs' files, as you said.
- If proofs' fix lands on #8539 before your report, add one case that goes through Lean's answer; if not, leave it to a follow-up and say so.

