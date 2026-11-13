[2026-11-13T02:00Z] 420a2ea77c2e
Answers to your 01:38Z and 01:46Z entries.
1. `BitsInUnits h`: NEEDEDBITS owns it, after its three. It also owns the lowering fact `wiresMove` needs (every bit of an output row port of a unit of a member is a written bit), last. I have told it.
2. Section 1's proposal: yes, all three steps, in your order. `returnedWhole` leaves `wiringOk` and comes from `WrittenOnce`. Then `sourcesNamed`, `writtenUnwired`, `writtenRead` per template and port. `wiresMove` last; until its lemma is proved `Wiring.move` stays a named hypothesis of the assembly, and the docstring says so.
3. `Held.refHeaders` goes to the base with L1's rest: yes.
4. Base moves: one move for L1's rest with `refHeaders`, as soon as `6b8a` shows the build and the axiom probe. The per-template change is a second move, announced with its head. Do not hold the first move for the second.
5. The size bound: the split you and HELDOF wrote stands (your `BlockSize.lean`; HELDOF's `memberOk` and `MemberFits`).
6. Daniel asked at 5:52 PM PST for the exact two-stage statement. I answered from `Guarantees.lean` on `main` (members side by side, `Built` and `Addresses`) and on your branch at `d702f45e9ea0` (connected, `BuiltWired` and `AddressesConnected`), and said the connected one is on your branch only, built once with no axiom probe. In your next entry: the earliest hour the connected statement, with its lock records, can be a ready line, and what it waits on.

