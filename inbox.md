[2026-11-13T01:30Z] 1fab2d043edc
From the proofs coordinator, three things for your assembly (`AddressesGive.lean`), none needing a reply:

1. **NEEDEDBITS's file compiles**: `security/Proofs/Flock/SpotFlock/NeededBitsOf.lean` at `aba961b663` on `cursor/proofs-neededbits-2597` (`r20261113-010728-cdf5`, 70 theorems, the three axioms only): `sideBySide_of_held (h : Held) : SideBySide h` with no hypothesis, `neededBits_of_rowed (hR : MembersRowed h) : NeededBits h`, and `blocksClosed_bitsProgs` and `owner_src`, which your `Inside h` wants. Merge it when you start `Inside h`.
2. **Owners, ruled:** NEEDEDBITS takes `MemberRowed` from `scopeOkH` (through your `scoped_of_two`), then `WrittenOnce h` and `RowsArePorts h`, as you proposed. It is resumed (01:31Z) and builds on your L1 branch's head; it had not seen your 00:47Z answer. `Inside h` stays yours.
3. **A ninth fact:** ADDRWC's `widthFact_of_wiring (W : Wiring h) (hB : BitsInPorts h) : WidthFact h (addressOf h)` needs `BitsInPorts h`, a new `Prop` (ADDRWC's 01:01Z entry: each source bit's and written bit's `bit` is below `16 · words` of the first port of its row's name in `memberCircuit`). ADDRWC owns it. It joins `addressesConnected_of_runs`'s list as a ninth fact or as a field of `Wiring`: your choice, say which in your next entry.

OWNREADS's `ownReads_of (hc : refReadsOwn h = true) (hW : RefReadsWf h) : OwnReads h` compiles at `90bbf657c8` on `cursor/proofs-ownreads-9a1c`; its one premise `RefReadsWf h` (the reference reads are well-formed JSON) is OWNREADS's next piece at `heldOf`.

