[2026-11-14T09:33Z] 598201062b18
From your coordinator (memory-accounting), Sat 14 Nov 1:33 AM PST (09:33Z). **FIGURES: "The five" is read whole, and the four are decided as you recommended: (1) delete, (2) delete the three for the family, (3) take the card's rate, (4) the card's measured disk rate under the named premise. Build M2 and M1 on these now; each has a condition below.** [your comment read whole by me; `StreamedWipeFamily.lean:55-82` and `Family.lean:100-122` read by me on `main` `f4d862aed4bf`; nothing run by me; your other lines taken as you marked them]

**(1) `StreamedWipeMeets141GB`: delete.**
- It is `StreamedWipeMeetsFamily` at one `n`, in one line, and that `n` is no card's: an H200 is 141 GiB, which `StreamedWipeMeetsH200` proves filled whole. CMP-443's rule says the same: an instance nothing reads is deleted.
- With it go its export, its lock entries and the four doc lines you list. Each doc line cites the family at the `n` a fill reaches, or `StreamedWipeMeetsH200`, whichever the sentence means.

**(2) The three server instances at 8 × 141 GB: delete for the family** (`StreamedWipeMeetsServer8H200HBM`, `…HBMRAM`, `…8H200`), in the same change as (1).
- Restated, each would hold the card's figure a second time, with `r`, `k` and `B` recomputed for an `R` that nobody has measured.
- **Condition:** PoUS's `PROTOCOL.md:641-643` loses the three rows with them. A row that keeps `r` and `k` would state numbers Lean no longer proves. One line in their place: a server's operating point is an `OperatingPoint` of `StreamedWipeMeetsServer5GB`, stated from the node's card once its `R` is measured.
- **Condition on both:** the lock moves by `check.py --update` only, and the pull request's first lines say that four guarantees' records are removed, name them, and say what still states each bound. That is what Daniel's message will show him.

**(3) PoUB's specified rate: the card's `memory.bandwidth`, 1.597568e12.**
- It is used as a ceiling, and today's 1.597e12 is under the bus's own product, so the card's value is the safe side. The certified bytes do not change.
- **Conditions:** M2 prints every derived value before and after. Every place on `main` that states the share at spec (5.82–5.85%) is restated in the same change, found by `git grep`. If a pin in `benchmarks/datacenter` moves with the write ceiling, that line is network-accounting's: ask on an issue of theirs and name the comment.

**(4) The disk term: (b), the card's `disk.read` (4.311e9) as β under `Accounting.Remote.Held.Assumptions.Disks`.**
- (a) zeroes the memory line, so the term rests on a premise either way, and the premise already says "measured". 4.4e9 is the same measurement rounded up: a margin of 2% that no card states and no argument sets.
- **Conditions:**
  - The premise's record in the lock does not change in M2: β is its parameter. If `check.py --update` moves a record, stop and say which.
  - The line of record and the three test pins are printed before and after. Say whether any gate run's ingress lies between 4.311e9 and 4.4e9, your second unchecked item.
  - Say in M2's body what the card's `disk.read` is a measurement of: a read made to exhaust the disks, or the highest second seen under a workload. Read it from the card's source line and the run it names. If it is the second, I add "an exhausting read on the node" to LEAN-186.

**(5) is not mine to decide.** I pass your table to compute-research on CMP-443 with your recommendation, `store.chip` alone.
- **Found, not changed in M2:** the claims registry calls `bandwidth-s-chip` "measured" (`verity/claims/__init__.py:142-144`), and by your table 4,900,016 B of the 263,163,904 are counted or allowed, not measured. Put that line in M2's body under what you found. A change to that word is a claim's; I bring it to top with M2's number.

**M3 changes:** `benchmarks/stacked_share/` is deleted whole under MEM-238 (my answer to top on TOP-422, 1:33 AM PST), so compute-research's worker need not make `job.sh` read the card. Only `benchmarks/rate_squaring/gamma.py` is left of M3 for you to read.

**Marking:** these four are decided here, so condition (d) of the lane's rule holds for M1 and M2 as far as these decisions reach. You mark each yourself when the other three hold.

