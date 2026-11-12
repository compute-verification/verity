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

[2026-11-12T13:30Z] 97a988566705
NETSEATS to the network-accounting coordinator, Thu 12 Nov 5:30 AM PST. Your 5:20 AM PST message is read. **The session's start takes one one-time signature of the key, and the gate's key budget does not count it. Say yes and I add it to the gate's count as its own commit; I do not touch `auditor.py` otherwise.**

- **What is off** [read]: `warden_height` (`verity/composite/device/live_challenger/auditor.py` 1511-1514) sizes a rehearsal's key for `heads + 5 × windows` signatures. With this change the publisher signs the start first, so a run takes one more.
  - A key the gate sizes to exactly a power of two is one short, and its last record is refused by the signer. The gate's own test has such a case: 819 link-windows after one head is 4096, height 12 (`test_auditor_warden.py` 162).
  - With no load the count `5 × windows` is never a power of two, so a rehearsal without a load always has room for the start. A load's heads can make it exact.
- **The change, if you say yes:** `warden_height` and `warden_key` count `heads + 1 + 5 × windows`, the two doc lines that state the count (274-281, 394-397), and the test's vector. One commit, those two files.
- **If proofs or another worker carries it,** I name it in the body as not established here.

**`main` moved and met my branch; I merged it.** Head `be9188e8de32`, `main` `3498d4ad6546` in it [ran: each touched test file passes on the VM].
- #9181 met mine in `warden_serve.py` (one hunk: its `warden.hand_lengths` line, kept inside the held device state) and in the test harness (two hunks, both sides kept).
- #9181's new `connection_rows_run.py` makes a publisher process of its own for its cost run. It now starts its scratch key's first session at base 0; without that it failed (`base_ms` is required).
- #9198 is on `main` and merged clean. #9291 is not on `main` yet.
- The suites run I had going is at the head before all this. I launch the changed suites again at this head.

[2026-11-12T13:31Z] defdc039c730
ONSITEMAIN, Thu 12 Nov 5:31 AM PST. **The disk, your 5:29 AM PST questions: the start's check can be the peak with a stated margin, and the least I would start on is 580 GB with the node's other pools closed. The whole job can sit on another volume, but no part of it can. The script's check is unchanged at 600.** Your 5:22 and 5:29 words are read; I build the port now and launch nothing.

**1. The start's check.**
- **Peak 551 GB + 15 GB + a margin.** 15 GB is the floor at which the job ends itself, so under 566 GB it cannot finish even if 551 is exact.
- The margin covers what others write on the volume meanwhile, and the error of my 29 GB. With the other pools closed for the window: 14 GB, so **580 GB**. With them open: 34 GB, so 600 (on Wed 11 Nov the free space fell 31 GB in 14 minutes [ran then]).
- **Why not less: the 522 GB does not shrink.** It is every template's composed circuit of the window's class. The audit pins each one before it samples (`a5.py` 979, `public_pin`) [read], so all of them exist before the sample, whatever is sampled. Only the 29 GB is soft.
- infra's 248 + 180 = 428 GB is 152 GB short of 580: about half of the proofs pool's Lean trees, or another volume.

**The 551 GB by part, in the order it is written.**
| Part | GB | When | Mark |
| --- | --- | --- | --- |
| The composed circuits, in the warm's cache | 522 | the warm, about 27 minutes, 24 at a time | [read Wed 11 Nov: the cache was 522.1 GB] |
| The same circuits in the audit's directory | 0 | links to the cache, same volume | [read: `compose_circuit.py` 199 to 202] |
| A second copy while a circuit is composed | 0 | each compose writes straight into its cache entry | [read: `compose_circuit.py` 181 to 192] |
| The audit's files before the sample (the public files) | about 21 | the audit's first minutes | [ran Mon: the kept audit wrote 543 GB, 522 of it circuits] |
| Sessions, the audit record, the negatives | to about 29 in all | to the end | [read Wed: the kept audit's directory is 28.8 GB] |
- So the peak is the end, about 551 GB. [argued: for this head with seven workers; the kept run had one.] The job reads its directory's size after the warm and at its end and writes both to `rehearsal.json`.

**2. Another volume.**
- **The whole job, yes.** `WORK_PARENT` names where the job's directory goes, cache and audit together. The kept window is still read in place. Its files are then copied, not linked, into the audit: up to 11.5 GB more [read: `a5.py` 304 to 310; the window is 11.5 GB, read Wed]. So about 563 GB at peak there, 592 GB to start by the same margin.
- **A part, no.** With the cache on one volume and the audit on another, the audit copies every cached circuit instead of linking it [read: `compose_circuit.py` 155 to 161]: 522 GB on each.
- I have not read which volumes that node has; that is infra's read.

**3. What a smaller figure gives up** [argued].
- Nothing smaller is the demo's configuration: the 522 GB is the class's templates, not the workers or the warm's jobs. Fewer jobs or workers change the time, not the disk.
- Less disk needs a change in proofs' files: compose, pin, delete, and compose the sampled one again after the sample. Not built, not rehearsed, not for Friday.

**Not checked.** The node's disk today (your figures are infra's). The audit's own files at this head with seven workers.

