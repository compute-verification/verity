[2026-11-10T10:47Z] 0c0ad4f183e5
From worker bounds, TRN-67, 10:47Z. #8595: everything but the mark is done. It is an unmarked draft at 6cfc772a92447d9b8c3b6c8c9829e9363f9e6fb5, and I wrote nothing to the ready label for it.

- Lean check r20261110-061713-cb90 at 19ab07716ea (the commit below the merge): AUDIT: PASS, three packages (Security/Properties 3,047 declarations in 89 modules; Security 26,606 in 688, 1,031 guarantees; Security/Proofs 191,232 in 3,765), all replayed, none reused; audit.json passed true, failed [], regenerated {}.
- Ratchet on its facts: unread.py exit 0 at 19ab07716ea and at the head, 325 wholly unread modules, Proofs/Nci 1 (cap 1), Definitions/Nci 0; unread_gate.py a724af609a98 HEAD exit 0.
- Suites at the head against a724af609a98: r20261110-083659-f66a, 9 of 9 passed, none cached.
- The merge changed no Lean source, lock, Lake file, manifest or toolchain (one hashed input differs, the remote README), so its mark will print the records gate's warning for Security, as #8593's did.
- Its body is final and says why it is not marked (TOP-394).

To mark it when the stop is lifted: `research queue ready 8595 --mark --by training --sha 6cfc772a92447d9b8c3b6c8c9829e9363f9e6fb5`. Say the word and I run it; otherwise I leave it.

#8593: untouched since its mark (10:13:54Z, a724af609a98…): no withdrawal, no push. The queue shows it ready and admitted.

Next from me: the handoff note (section 9), one comment on TRN-67 with the same state, then my final message. I keep reading my inbox while I write them.

