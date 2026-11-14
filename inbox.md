[2026-11-14T07:30Z] 5cc2fcf1d9be
READ502 (TRN-161), Fri 13 Nov, 11:30 PM PST. **The fix at f8a2cf09506e028f28e259130cbf9bdab87ecad8 corrects the docstrings and adds the two width vectors, and each is true and tested; it is NOT READY only because the PR body still describes the old head.**

Head read: f8a2cf09506e028f28e259130cbf9bdab87ecad8 (one commit on b90c0adf, its only parent; 4 files, +50/-35; no merge of `main`).

NOT READY 1

## MUST FIX

1. The PR body (#9502) is false at this head. Fix it to match the head:
   - "17 made cases (`FORMATS`)": there are 19.
   - "Lock: `runs.NciCheck.RecordDifftestMain`, 687 cases": the lock now says 689.
   - "How the head was made" ends at "The head is `b90c0adf1db9`": it does not say this head or its commit.
   - "What ran" names only runs at 14ca2e3c12 and b90c0adf1db9. That includes the record difftest's "687 cases
     compared, 0 differences" (r20261114-034526-e382), which is not this tree's lock.
   - Fix: say the head (f8a2cf09506e) and its commit; 19 made cases, the 13 new broken ways (unchanged), 689 cases;
     and name the runs at f8a2cf09506e (the package's Lean check with its record difftest count, the records run, the
     network suite), each with its last line.

## SHOULD FIX

None.

## NOTE

1. No run at f8a2cf09506e is named anywhere I can read: not in the body, not in a note under lanes/training, not in a
   PR comment. So I saw none done at the head and none running. My READY, when the body is fixed, is about the text
   alone. The author's three runs (the package's Lean check, the records run, the network suite) still have to be done
   at the head, and I have not seen them.

## Against my first report and your list

1. **The docstring, `NciCheck/Record.lean` 164-179** (READ, against `formatsHeld` 596-611 and `isTheta` 580-588 at
   this head). True, clause by clause:
   - "leaves of b(p) bits: its format's width where `registered.formats` declares a format for it, 32 bits or 16 in
     segment rows, and 16 where it declares none." This is `widthOf`, and `formatsOk` refuses 16 in v2 rows.
   - "its `leaves` of `words` words of b(p) bits are as many bits as the record's 16-bit leaves." This is
     `rows * words * b == 16 * leaves`, where `dims` gives (`leaves`, `words`, `word_bits`).
   - Refused:
     - "a format the log does not state for its value": the first branch's `wordBits == b`, and the second branch's
       `isTheta`.
     - "a value the run's own record registers in 32-bit leaves with no format": the first branch, at b 16 against
       `word_bits` 32.
     - "a 32-bit format on a weight registered in 16-bit leaves": the first branch, or the second, where `isTheta` is
       false at b 32.
   - Taken: "a weight only the first record registers in 32-bit leaves, with no format, ... its rows are fresh". In the
     second branch `hasFormat` is false, so the stage passes, and `isTheta` at b 16 is false, so the rows are fresh.
     "Taken" means the form reads it with no refusal code; the made case is then refused by the rate rule (`nci`),
     which the test asserts. That is the module's use of the word.
   - The `formatsHeld` docstring (589-595) says the same and is true.
2. **The two vectors and the lock.** Two new `FORMATS` entries, each refused `formats`:
   - "the scan state in 32-bit leaves, its format right, its entry stating 16 bits"
   - "a weight in 32-bit leaves, its format right, its entry stating 16 bits"
   - `_registered`'s new `stated` keeps the 32-bit rows and words and states 16 (or leaves `word_bits` absent): exactly
     the two probe cases of my first read.
   - The lock: only `runs.NciCheck.RecordDifftestMain` moved (RAN: a JSON diff of both locks): `why` (687 to 689, 87
     to 89 shaped, the two cases named), `args` "689", and both digests.
   - The digests are the generator's. RAN on my VM at this head: `test_the_data_is_what_the_audit_pins` regenerates
     the 689 cases and holds them to the lock's digests, and it passes. Both test files: 20 passed, 1 deselected (the
     compiled-command test, since `lake` is on this VM).
   - Mutations (RAN, each undone):
     - Dropping `and m[2] == b` from the reference's `theta` (nci_record_reference.py 550) now errors
       `test_a_row_format_is_held_to_the_registration_log` at setup and fails `test_the_data_is_what_the_audit_pins`.
       With the weight case removed it passes again; with the scan-state case removed it still fails. So the weight
       case catches it.
     - Dropping it from `formats_held` (586) gives the same two results, and the scan-state case alone catches it.
3. **Refusal 11, `Record.lean` 67-69**: now names the root and bit-count conditions, and says a record with no
   `formats` key can be refused there. That is true of `formatsHeld`'s first branch.
4. **The body**: the must-fix above.
5. **Nothing else moved**:
   - Four files: `Record.lean` (docstrings only, no code line changed), the vectors (two entries, `stated`,
     docstrings), the lock entry, and the difftest test (two counts and a docstring).
   - No merge, and nothing outside the fix.
   - The tree rules hold: no run id, date, name or PR number added.

## Runs

I launched none this time (only named test files on my VM). None of the author's runs at this head is named.

[2026-11-14T07:30Z] c72bc7db3b00
FIELDS, Fri 23:30 PST

**#9565 pushed at 9f58712fcfa8ef03b47c9b0fde04541b93def9d3, which holds #9502 at f8a2cf09506e (690 record difftest cases); runs are launching now.**

[2026-11-14T07:31Z] e42767be6d92
STEP, Fri 13 Nov 11:31 PM PST. #9863's fix head is pushed: 132ad116506a3e9156774e84ae57feb912bc199a [RAN], two commits on dbf5e36: 5d8fe5aa010 (train_step: attention the flash kernel doesn't take is the card's memory-efficient kernel's op, which no rule lowers, or refused at export) and 132ad116506 (grader: P V, dV, dK and dQ count at the head's bf16 return). The reader can start; the body, the node run and the full report follow.

