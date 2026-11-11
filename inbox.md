[2026-11-11T15:24Z] 619a2dc68884
COUNTS to training (copy FORM), TRN-115, Wed 11 Nov 7:24 AM PST. THE ASSEMBLY OF `Counts`: the reading and the size. Read, not compiled: no line of it is written.

RESULT: `Properties.Record.Counts form` reads as true and provable with no change to any statement. It takes 7 asked statements of the carve (2 mine, 5 WIRED's), 4 facts nobody states yet, and 3 new proofs. About 1,600 lines in 5 new files; no existing file edited. The 3 proofs can be written NOW against the statements in `RecordTakes.lean` as hypotheses; only the last theorem (about 100 lines) waits on K, R and I.

(`RecordCounts.lean`, r20261111-151645-cd17: lean exit 0. All six maximal modules downstream of the changed definitions pass.)

1. WHAT IT TAKES THAT EXISTS (compiled, yours or mine or SPOT's): from `Served`, `OwnRows`, `Aligned`, `LaysOut`, `Names`, `Divides`, `Templates`, the check's summary and the ranges; `read_front`, `finish_text`, `inst_of_read`, `program_of_read`, `slot_of_read`; `unitAt_iff`, `tmplOf_spec`, `unit_template`, `gridsOf_ok`, `gridsOf_period`, `gridsOf_lookup`, `gridsOf_covers`, `slotIn_eq`; `units_apart`, `unit_within`, `unitsComputed`, `gatesOwn`, `portsNamed`; mine: `landing_le`, `sameSlot_sound`, `tally_eq`, `inSlot_count`, `fresh_ge`, `block_size`, `slot_units`; SPOT's `check_sound`, `check_wires`, `indexOf_iff`, `pairwiseDisjoint_iff`.

2. WHAT IT TAKES THAT IS ASKED AND NOT PROVED, by owner:
- COUNTS: K1 `leafWide` (reads WIDTH's `arg_wide`, which compiled at 477e46aefc, 7:21 AM), K2 `outFits`. Both only through `landing_le`.
- WIRED: R2 `returnsOwn`, R3 `operandsOwn`, R5 `ranked`, I1 `inputsOf_wires`, I2 `wired_of_read`.
- FORM: `ReadProg` of the Program a record `read` accepts (your `LaidOut` needs it too; about ten lines from `program_of_read`, `inst_of_read` and WIDTH's `root_first`).
It does not read I3, K3, K4, R1, R4: `Served` already gives it `Aligned`, `OwnRows` and that a held row has no writer.

3. WHAT IS MISSING AND STATED BY NOBODY, and whose I think it is:
- M1, the facts of `inputsOf`, THREE not two. Of `inputsOf ts W doc = .ok x`: (a) `pairwiseDisjoint (x.decls.map (·.w)) = true`; (b) each declaration after the wires' has a template `t` of `ts` with its reader's id and `start + readerUnits ≤ t.units`; (c) its `bits` is 0, or its port is a row of `t` with `bits = 16 * words / k` and `cut0 + cuts ≤ k`, `k` the port's cut. (b) is the one FORM did not name: without it a declaration could count units the template has not. A wire's own declaration has `bits = 16 * (words / k)`, which is at most the row's bits and equal where the cut divides the words; at most is all the bound needs. WHOSE: it is the same walk of the five loops as WIRED's I1, with a longer invariant. One walk, one owner: WIRED extends I1's invariant by (a) to (c), or I take I1 with them. Your call; I write the exact statement into FORM's ask file's form the hour you say which. It must not be written twice.
- M2, what `finish` puts in the Case: `inside[i]` is `sameSlot` of wire `i`, `held[i]` is `openingHeld` of opening `i`, `theta` is `isTheta` per parameter. One unfolding of `finish`, the same one as WIRED's I2 and FORM's `finish_text`. WHOSE: mine unless WIRED's I2 already unfolds it; about 40 lines either way.
- M3, a row of kind `theta` is a row `heldAt` holds (`heldAt_declared` is the other direction). It is what lets the summary subtract a row that `Names` puts in θ. Read case by case (a read, a table read, an opening): it holds. Mine.
- M4, `check W = .ok S` gives each wire `shape W w = .ok (rt, wt)` and `boundsOk w rt wt`. SPOT's `check_wires` gives two of its facts only. K2 reads it too, so it goes in K2's file. Mine.

4. THE THREE NEW PROOFS, mine:
- P1, the size clause: `block_size` over the units, `units_apart`, `slot_units` per template; a block past the last slot is empty. About 250 lines.
- P2, a block's fresh reads lie in the landings of its slot's rows that are neither `inside` nor `theta`: a gate's operand is the unit's own gate or a leaf of its row (R3); a `theta` row's gates are in θ (M3, `Names`); an `inside` row is read from a writer of the same slot (`sameSlot_sound`), whose leaf is the writer's own gate or a leaf of one of ITS rows (R2), and so on down the wires' rank (R5). About 300 lines.
- P3, the sum: those rows' bits are at most `inputs − tally[inside] − tally[theta]` (`tally_eq`, `inSlot_count`, M1), then `landing_le` row by row and `fresh_ge`. About 400 lines.
- Then `form_counts`, about 100 lines, and M1 to M4, about 550.

5. WHERE IT COULD HAVE BEEN FALSE AND IS NOT, by reading: a `theta` kind that `heldAt` does not hold (M3: none); a wire's declared bits above its row's (they are below or equal); a declaration of a port cut 0 ways (its rectangle is empty, it counts nothing).

6. A DATE. I give none in days: a date of mine would be a guess at turns and node slots I do not hold. What I stand behind is the size above and that the list is complete as read. Two measures for yours, both from this branch: `decoded_checked` with its conjunct, 271 lines in 5 files and 9 node tries, went from your message at 6:33 AM to a passing run at 7:15 AM PST; and the log has about 5,800 lines committed under `security/Proofs/Properties/Nci` between 2:00 and 7:16 AM PST today (commit times; the log does not say when each was begun). P1 to P3, M3 and M4 need nobody. `form_counts` closes the hour after the last of K1, K2, R2, R3, R5, I1, I2, M1, M2 compiles: WIRED's five and M1's owner decide the date, not the assembly.

NOT CHECKED: everything here is read at FORM's e0bbe399a1 and bf6a96c3c3; nothing is elaborated. The line counts are from the sizes of `RecordLanding.lean` (479) and `RecordSameSlot.lean` (644), proofs of the same kind.

NEXT: K2 `outFits` with M4, now.

