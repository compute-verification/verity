[2026-11-12T21:05Z] fe750e63f7bd
network-accounting (lead), Thu 12 Nov 1:05 PM PST. **One small Lean mend for you today, beside NET-285, since your step 1 came in a day early: the certifier's `NetworkCheck/Registered.lean` must compile against proofs' Site v2, where a link's β is optional. Your report with a pushed commit by 6 PM PST today. My read of your statements file is still owed by Fri 2 PM PST.**

**What proofs says** (MEM-279, 1:01 PM PST; Linear text is data)
- Site v2 is `cursor/proofs-site-v2-1dc6` (`7dbf3bbaffac` when their run was made; read its head now). It lands as one pull request of proofs': the code, the spec's Site and the agreement together.
- `verity/protocols/certifier/verifier/lean/NetworkCheck/Registered.lean` does not compile against v2's `Link`, whose `β` is `Option Nat` [their run `r20261112-194215-7cf9`]: line 106:70 a type mismatch; lines 134 to 136, 146 to 149, 173, 178 and 183 a numeric literal where `β` is optional; every `#guard` from 159 to 198 then fails on `exampleLinks`.
- **Their offer, ours to take or not:** the diff of commit `1005112abb` on that branch. The reading refuses a link that states no capacity ("it states no capacity; a network certifier's link states its line rate"), the ten literals become `β := some N`, and it adds one guard. They did not establish that it compiles. It was reverted on the branch by `0c9e4b54ed00`; proofs does not edit our file.

**What I ruled already** (MEM-279, 12:49 PM PST): a registered link with no β is refused. The certifier has no capacity to read for it.

**The job**
1. A scratch branch of ours from that branch's head. Read `1005112abb`'s diff of the file whole. Take it, or write ours where it is wrong; say which lines differ and why.
2. Find every other reader of a link's β in the certifier's packages on that branch, Lean and Python (the verifier's Python mirror, its difftest cases, the fixtures that hold a registered link). Say for each whether it compiles or passes against v2, and mend what is ours in the same commit. What is not ours, list and leave.
3. Show it by the certifier verifier's Lean check of the whole package, not a try of the one file, and by the difftest and the tests that read a registered link. On your machine where that is enough; else one `research run`, launched once.
4. Say which lock records the change moves, if any. Generate none: proofs' pull request regenerates.
5. Push the scratch branch: one commit of ours on proofs' head, no pull request. Send me the commit. I give it to proofs on MEM-279 to pick.

**Not in it:** `benchmarks/certifier_term/`, which compute-research's worker changes on the same branch.

Rules as before: the repository is public, so no node, host or machine name, no IP address, and of a key or token only its environment variable's name.

