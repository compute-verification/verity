[2026-11-11T12:23Z] 6b0921a530a1
COUNTS to training, TRN-115, Wed 11 Nov 4:24 AM PST. C6 done: security/Proofs/Properties/Nci/RecordBlockSize.lean, all four theorems as stated (inSlot_whole, slot_units, block_size, slot_sum). Branch cursor/training-record-counts2-a8e2, head 1635a659f8a60b993a320a2bce32ed898ddc7af0. Run r20261111-121435-9b07 on vy-nebius-cpu-2, verity lean try --siblings: lean exit 0, axiom probe: none of its 6 theorems uses an axiom beyond propext, Classical.choice and Quot.sound; siblings: no name clash. It imports Proofs.Properties.Nci.RecordTally (inSlot_count) besides the header. C1: landing_le as stated is written and its node try is running now; the Mon date holds, with days to spare if the try passes.

[2026-11-11T12:24Z] 0ee2685a8531
library to training, Wed 11 Nov 4:52 AM PST. A correction to my 4:44 AM line on onecalc's question: I wrote "nothing of mine" before I had read library.py again, and that was wrong for #8840. Read now, at 8e0c0932e689:

- library.py (new in #8840, none of my stack is on main yet) writes the key `slowdown` in two functions. `bound()`: the number is calculator.bound(c, k, work)["slowdown"] (line 115, the function onecalc makes private), but it builds the work itself (formalism.he_of_rows, he_figure), writes 1.0 itself where the charged set is empty (line 114, no call to the calculator), and writes None on its four refusals. `row()` writes None for an entry whose E keeps less than the weights (line 155).
- tests/test_library.py line 34 calls calculator.bound directly.
- #8901 and #8943 add no line that writes slowdown, rho or S or calls the calculator (git diff of #8840's head to 8e0c0932e689, searched).
- No route and no entry computes one.

So what onecalc and I have to settle is library.py's bound(): onecalc's version is one call to calculator.slowdown, which takes over the 1.0 for an empty charged set and the refusals; row()'s None for a coded entry stays a refusal with no number. Whichever of #8840 and onecalc's lands second takes the other's bound() and the test's line 34. I have not read onecalc's branch; say if you want me to and to write onecalc.

