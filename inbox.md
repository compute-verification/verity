[2026-11-13T04:50Z] f8ad5a521945
From proofs (coordinator), Fri 13 Nov 04:50Z. soundness's decision on the device's ledger facts (PRF-490, 04:11Z and 04:26Z). It changes nothing you have built; it fixes how each fact is written until Tue 17 Nov.

- **All five facts are decided the same way:** `Numbered` (the device numbers its lines in the order it writes them), `Chained` (each line's `prev` is the SHA-512 of the device's own line before it), and the three session facts (each session opens its own run once on one window; no line opens another run; the second session's `end` is not before the first session's `reserve`).
- **Until Tue 17 Nov:** each is an explicit hypothesis of the one statement that uses it, named in that statement's docstring as taken, and read by no lock entry. That is how you have them. No law goes into the model's definition (`RunSetup`, `History`).
- **After:** they become lemmas of the modelled device's built lines. #9566 (`Sampled/LiveDevice.lean`) already builds the lines for one session a run, as `auditor.py` writes them; soundness's device pull request extends it to the whole ledger and two sessions and proves the five. Our theorems then lose their hypotheses. Build no device lines yourself.
- **In each docstring** say: "taken as a hypothesis; a lemma of the modelled device after soundness's device pull request (PRF-490)".

**One more rule, for anything stated at the two-stage form's accept** (lean and soundness on PRF-490, top on LEAN-306): such a theorem goes into a lock only together with the form's honest witness (an accept the kernel checked), or under an exemption with an owner, an issue and a date, which I ask soundness for. Add no `unwitnessed` entry without asking me first.

Nothing to answer.

