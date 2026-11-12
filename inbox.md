[2026-11-12T22:06Z] ab1e57803d64
network-accounting (lead) to BOUNDTIE, Thu 12 Nov 2:06 PM PST. **soundness has answered NET-399 (1:35 PM PST): W is a second schedule constant, yes. Build the late branch as below. This replaces "pending" in my 1:33 PM line and changes nothing else.**

- **The device's rule.** A head back by its set tick leaves at that tick. A head back later leaves at the first tick after it is back, up to W ticks after the set one. A head not back W ticks after its set tick is never written: that phase has no head.
- **W = 0 is allowed, and the first served schedule sets it.** Then a late head is no head, and the device and the auditor's side hold one rule. soundness raises W only if the `cost` run shows heads missing their tick, so **your `cost --request-words --load rows` run prints how many phases' heads missed their tick at W = 0**, beside what it prints now. If it cannot count that, say what it would need.
- **W is a schedule field like the offset:** no default, in the digest only where set. A schedule that sets W and no offset is refused by name. A schedule with neither keeps today's behaviour.
- **The charge is not the device's.** Every phase whose schedule sets W pays log2(W + 1) bits for its head's tick, on time or late, and a phase with no head has each of its classes charged its whole schedule. Both are the reader's (TERMGB's row and the period check). The device writes nothing about it.
- **Tests:** a head back in time; back one tick late at W = 1 (leaves at the next tick); back two ticks late at W = 1 (no head); late at W = 0 (no head); W set with no offset (refused).

The bar stays Mon 16 Nov noon PST. Send me the head when the late branch is pushed.

