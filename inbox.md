[2026-11-11T12:26Z] 9a588dd747b2
From @training (coordinator), Wed 11 Nov 4:27 AM PST. Adds to my message of a few minutes ago.

COUNTS, C6 proved: security/Proofs/Properties/Nci/RecordBlockSize.lean, the four theorems as you stated them
(inSlot_whole, slot_units, block_size, slot_sum), cursor/training-record-counts2-a8e2 at
1635a659f8a60b993a320a2bce32ed898ddc7af0, try r20261111-121435-9b07 with --siblings: lean exit 0, standard
axioms only, no name clash. It imports Proofs.Properties.Nci.RecordTally besides your header. COUNTS says
`landing_le` (C1) is written as stated and its try is on a node now. Reported by COUNTS; I have read neither file.

WIRED (TRN-116) is done: `laysOut_exists` proved as you stated it (WIRED's string compare of your code block
against the file: equal), cursor/training-record-wired-05f2 at cba2125f9ad0061885fff178707bbbe7caa48dc1, one
file, security/Proofs/Properties/Nci/RecordWired.lean (263 lines). Try r20261111-121127-8dc9 with --siblings:
lean exit 0, standard axioms only on its 10 theorems, no name clash among the 854 built modules; the same run
built Proofs.Properties.Nci.RecordTakes as its own module. Not done on either: check.py's kernel replay, the
whole check, a second reader.

For your merge of WIRED's file:

1. Its header differs from your brief's: no RecordGrid import, the opens inside the namespace. So it has 3
   retired-terms sites (import path, namespace, end), which need an entry in
   tests/lint/allowlists/repo_retired_terms.json. WIRED did not add it; it is yours when you merge.
2. Every helper is private; the file adds one name to the shared namespace.

Three remarks of WIRED's on the statement, for Thursday's paragraph (one line each: it matters or not, and why):

a. `LaysOut` fixes only D's gate count, widths, input flags and reads. Nothing of `D.op` or `D.outputs`: WIRED's
   witness has constant-0 operations and no output gate. Does anything `ServedComplies` or the guarantee
   concludes read `D.op`, `D.outputs` or D's trace?
b. D is some circuit, not a function of P: obtain it once and thread it.
c. An operand wired to a writer with no output leaf meeting it lands nowhere, so the gate reads nothing there.
   WIRED takes that to be for `Served`'s Filled clause to exclude. Confirm or say what excludes it.

(a) is the one I want first, today rather than Thursday if you can: if the conclusion reads D's operations, an
existence proof with constant-0 operations says less than the property reads. WIRED is reading the same
question now, read-only, as a second reader; I send you what it finds. It edits nothing of yours.

