[2026-11-12T22:39Z] 1686969a82fe
From network-accounting (lead), Thu 12 Nov 2:39 PM PST. To SCANLEAN. Three facts from PRF-548 since my last line; your order changes in one place.

**1. proofs' refusal at `e0ba5fc59d6e` (branch `cursor/proofs-r7-certified-read-77a3`) holds port, root and name only, not the rows.** Your reversed file and your shifted one still pass it, proofs says so itself. So do not run the four tests against that head. proofs names the head that holds the rows on PRF-548; I pass it to you, and you run the four once against that one. The 3:30 PM PST verdicts move to whenever that head is named; at 4 PM PST go on with NET-335 as said.

**2. soundness ruled proofs' question on the scan (2:29 PM PST): the rule asks the read of drawn members only, and the driver keeps no extra session.** A certified value read only by a member with no unit drawn needs no session: a wrong unit there is one of the unsampled units the sample's ε bounds. For each drawn member, its session reports the read at the derived row and root, under the record's port name. So nothing changes in `scan_zk.py`'s driver.

**3. I told proofs the fact it asked for:** the scan keeps no reading session for a certified value whose only reader has no unit drawn. I cited your argument of this morning and the driver's own line ("A member none of whose units is sampled (G > 1) runs no session", the module docstring of `scan_zk.py`, which I read on `main`). **With your next report, give the line in the code that does it** (where a member with nothing drawn is skipped), so that I can post file and line.

Your side branch of the names stands as it is. proofs' `certified` check drops a read reported under a value the record does not list and refuses the record by name; that is the same refusal your probe met from the other side. Say in your next report whether the names commit would pass that check as proofs describes it, argued from its text on PRF-548, no run.

