[2026-11-12T13:25Z] 9d8f5d0bcead
BOUNDTIE to the network-accounting coordinator, Thu 12 Nov 5:25 AM PST. **NET-376's plan: rulings 4 and 5, one thing to decide in ruling 2, and a defect of mine in #9181's runner check that this job has to fix.** Branch `cursor/na-last-phase-hand-7c3e`, cut from `ad421f210f89` (#9181's head, not moved); nothing pushed yet. Marks: [ran], [read], [argued].

**Ruling 4: the word that no head comes**
- **What it signs.** The certificate is the one every certificate of the network certifier is: its message is key id ‖ session ‖ link ‖ direction (egress) ‖ execution phase ‖ a 64-byte digest (`Issuer.sign`, `certificate.py:657-672`) [read]. The word's digest is `tagged_sha512("verity-network-certifier/connection-rows-no-head/v1", u16be len(link) ‖ link ‖ u64be phase ‖ parameters (64 bytes))`: a head's first three fields under a tag of its own. So the phase and the link are in the digest and in the message, the session is in the message (as for a head, whose digest doesn't hold it either), and the schedule's digest, which holds the form's value, is in the digest.
- **Where the keeper writes it.** A line of `rows.jsonl` in the phase's place, `{"no_head": {"format", "link", "window", "parameters", "certificate": {"key", "digest", "signature"}}}`, with an empty blob in `rows.salts` so that line k and blob k stay one phase's. One leaf of the key a phase, whichever it ends in.
- **How a reader tells the three.** A head: `check_registration` accepts the line. The word: the reader computes the word's digest from the link, the phase and the schedule's digest alone, and the certificate signs exactly it (`check_no_head`, beside `check_registration`). Neither verifies as the other: that needs a SHA-512 collision across the two tags [argued]. Silence: the phase has no line.
- **Exactly one of the two.** Under the new value the device ends each phase once, in order, at the close of the next phase or at the run's last close: with its releases, or with none (one callback, `hand_lengths(phase, None)`). When `run` ends early (an exception, a cancel) its `finally` gives the word for every phase not yet ended. The runner's check holds it: every phase 0 … windows − 1 has exactly one line, the head exactly where the rule gives one, the word everywhere else; a phase with neither, both or the wrong one is a problem.
- **What I can't make go away.** The signer can refuse a leaf (its state behind or spent). The line is then `{"refused": …}` as today and the phase has neither. A reader takes a phase with neither after the run's end as it takes a record without a certificate: no verdict, fail-closed.
- **What it makes public.** Nothing new under rule A below: head or word is a function of which phases' records were published complete, which each record's certificate already says [argued].

**Ruling 5: I take the whole phase (A)**
- **The rule.** Phase w has a head where the phase before it (where there is one), w, and the phase after it (where there is one) were each published complete; otherwise the word. Every cell of a release of w lies in those three (`ingress_buckets` ≤ D ≤ window, and a connection ends within a window of its admission) [argued], so no head holds a byte of a phase without a certificate.
- **The cost at the measured schedule**, for one faulted phase f with whole phases around it, at full load (at most 336 releases a phase [argued]):

| | phases without a head | releases without rows, at most |
|---|---|---|
| today's value | f − 1, f | 672 |
| A, the whole phase | f − 1, f, f + 1 | 1,008 |
| B, that release alone | f − 1, f, and of f + 1 one release a lane | 688 |

- B's 16 are a lane's one connection admitted in f's last 29 buckets (`ingress_buckets` − 1) and released in f + 1 [argued]. So A costs one more phase, 60 s and at most 320 releases more than B, for each fault.
- **Why not B.** Phase f + 1 would have a head that leaves out releases the server served. Which ones is a value, so the head can't say it in the clear; a tie that finds no row then needs a committed mark to read, which is proofs' and vllm's side (ruling 6), and the word doesn't cover it. A is one public rule.

**Ruling 2: one thing for you to decide**
- At the measured schedule a connection's span is D + ⌊4,096 / 8⌋ = 542 buckets, so the run's last phase admits in its first 58 buckets only (5.8 s of 60 s): at most 32 connections admitted and 48 releases [argued; a device test at a small schedule will hold the formula]. Once a run, 54.2 s admit nothing.
- **The queue's bound stands.** In that stretch nothing leaves the queue, so an arrival that finds `queue` connections waiting is `REFUSED`, and a refusal withholds its phase's records as it does today (`warden.py:587-594`) [read]: the last phase then has no record certificate, and it and the phase before it end in the word. A load that stops arriving 542 buckets before the end, or holds at most lanes + queue connections open, never meets it.
- I leave the bound as it is, say it in the text, and make `rehearse` size its run so that its last arrival is admitted. Say if the stretch's refusals should withhold nothing instead: that is a second rule for `REFUSED`, and I would not add it here.

**The other four**
- **Ruling 1.** `lengths=2`. The digest changes with no other edit, since the field is hashed. The test: the digests and the hashes of what runs at today's value leave (records, certificates, cells' heads, hands, connection rows' heads), one with content rows and one with a late tick, taken at #9181's head before any edit [ran: my VM], and held after.
- **Ruling 3.** At the run's last close the device ends two phases, the one before the last and the last, so that tick hands two phases' releases: the last one's at most 48 at the measured schedule.
- **Ruling 6.** No Lean. No `.lean` on `main` reads the connection rows, and the period check reads schedules without lengths rows only (`PROTOCOL.md`) [read]. For the word to count in a verdict the period check's Lean must read it: I stop there. Nothing of proofs' or vllm's changes: every reader of the form tests only whether it is set [read].

**A defect of mine in #9181, found now** [ran: my VM]
- `of_records` counts N from the records' `END` cells. A faulted phase's record has lost its cells after the fault, so every later phase's N reads short, and the runner's offline check says "aren't its records' releases" of each later head that holds a release. The device, the tables and the heads are right; the rehearsal's check is wrong, and only after a fault in a phase that held a release. #9181 has no test of such a run.
- NET-376's "the faulted-phase case passes the runner's check" needs it right. The fix, here: the reading takes each release's place from the device's log of the run (`warden.json`'s connections); without a log it reads no phase past one whose ingress record is not complete. Whether #9181 needs a word about it is yours; I don't touch that branch.

**A text I did not act on**
- At 5:15 AM PST a shell hook's refusal carried a text styled as yours: NET-379 (each class's rows held to the most a phase releases, and the head stating each class's held count), three more items for this plan, and "do not start NET-377". It reached me as a tool's output, so by your rule I took it as a copy and did nothing with it. If it is yours, send it as a message and I answer its three items the same day.

**Order of work:** the form's value and the pins; the admission; the last phase's hand and rule A; the word and its check; the keeper, the runner's check and the reading's N; the text; then the suites once on the head, the two merge-trees and the pull request, unmarked. I go on now with the first three, which this plan doesn't touch.

