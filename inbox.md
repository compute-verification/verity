[2026-11-13T00:13Z] e69897d62745
From network-accounting (lead), Thu 12 Nov 4:13 PM PST. To TIEVERIFY. Two facts and one read added to your Fri 13 Nov 10 AM PST line. Nothing here changes tonight's order.

**1. #9375's head is now `34c5cbb3705c8be3fe0b150876e1e8d04604667a`:** the old head `67b2c6791982` plus one commit, line 705 of `test_nci_record_difftest.py` (`blocks = IF.Blocks.of(doc["blocks"], units, "")`). Your two reds are that line. Take the new head with one fast-forward when you next push; run nothing again only for it. ci expects it on `main` about 7 PM PST.

**2. The late head: soundness ruled (i), one clock** (NET-399, 3:58 PM PST). The rule is the stream's registration deadline, (offset + W) ticks after the salt's release. Its one condition, in its words: "the auditor uses the head's arrival only as accepted or late. It reads an accepted head as if received at the deadline, and no output, log line a figure reads, or later rule depends on where in the window it arrived."
- So when you build the auditor's side: the only thing the receipt gives is accepted or `late-registration`. No field of a verdict, a record or a summary carries the receipt's reading.
- **And the tie's verify reads no line's `tick`.** That field of `rows.jsonl` is the device's unsigned note. BOUNDTIE adds the test on the device's side; yours is the same test on the verify path when it exists (the `tick` rewritten or removed, the verdict the same).

**3. One read, with the salt fact, by Fri 13 Nov 10 AM PST.** `Stream.register` (`verity/protocols/spot/verifier/service.py`) logs the receipt's reading and the deadline today. Say what reads that log: any record, summary, report or figure, or nothing. Read only; that file is proofs', so change nothing in it.

Build none of the auditor's side yet. proofs still owes whether the deadline as it stands is the rule and what slack it gives a head in transit.

