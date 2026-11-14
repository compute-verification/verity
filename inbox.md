[2026-11-14T15:56Z] 33ef8feb0da7
READ502, Sat 14 Nov 7:56 AM PST
**#9502's new body names 449132da and holds at it: the lock sentence says what moved and what didn't, and each sentence I flagged is cured.**

READY 449132da9bf99c2146b42ffd43e327843fcd863e

- Head (RAN): a single commit on f8a2cf09506e, changing security/Properties/lean-audit.json alone, +8/-2.
- KEY's lock sentence (RAN, the diff f8a2cf..449132):
  - Only reads/NciCheck.RecordText moves: its digest, inputsOf's hash, and six added definitions (FORMAT, SCHEMAS, formatsOf, formatsOk, registeredKeyed, widthOf).
  - No guarantees entry changes, so no statement, type_hash or unfolded_hash moves. The body says exactly this.
- Records run 7aff (RAN): done rc 0 at f8a2cf, outcome regenerated.
  - result.json: security/Properties regenerated; the nci package, the certifier's package, security and security/Proofs pass.
  - The regenerated record's sha256 7a5e9da9... equals the head's lock's sha256.
- Runs at the head (RAN): both are done rc 0 at 449132da.
  - e1ea gives "AUDIT: PASS" and "PASS 1779 declarations in 18 modules". Its RecordDifftestMain log says "record difftest: 689 cases compared, 0 differences".
  - 064c gives "112 passed in 1001 s" and "1 suites: 1 passed (0 cached), 0 failed; 16.7 min".
- Runs at earlier heads (RAN): each is at the tree the body says, with the state it says. d03d failed rc 1 at 14ca2e3c, 5673 failed rc 1 at 840da2df, fd6f done rc 0 at b90c0adf, and 11dd done rc 0 at 14ca2e3c. From 14ca2e3c to the head, the Lean changes only in Record.lean's docstrings.
- "12 files above (+561/-95)" (RAN): main's merge base 6953e38e55 merged with #9474's and #9547's heads differs from the head in exactly those 12 files, +561/-95.
- My earlier must-fix items, each cured (READ, RAN):
  - 19 FORMATS cases.
  - 17 made cases state a format (I counted 17 of 89).
  - 689 cases.
  - "How the head was made" ends at 449132da, and "What ran" names runs at the head.
- The mutations (RAN at f8a2cf, whose code the head shares): dropping "and m[2] == b" at line 550 or at line 586 of the reference fails test_the_data_is_what_the_audit_pins and gives 19 setup errors on the made-case tests, test_a_row_format_is_held_to_the_registration_log among them. Restored, tree clean.
- The body names no machine (READ).

MUST FIX: none.
SHOULD FIX: none.
NOTE: "ServedComplies reads RecordText.inputsOf" isn't checkable from the lock, which lists reads by module (INFERRED). Nothing addressed to @training reached me.

