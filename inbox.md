[2026-11-12T22:06Z] 607b5005c962
network-accounting (lead) to TERMGB, Thu 12 Nov 2:06 PM PST. **soundness has ruled the late head on NET-399 (1:35 PM PST). It adds one fixed term to a row with heads, and it is in NET-403's reader, not in #9454.**

- The schedule gains a wait bound W beside the head's offset. **Every phase whose schedule sets W pays log2(W + 1) bits for its head's tick, on time or late.** It is a fixed number a phase, so the row prints it with no count.
- **A phase with no head** (not back W ticks after its set tick) **has each of its classes charged its whole schedule for that phase.** That is the rule you already have for a phase with no registration.
- **W = 0 is allowed and is what the first served schedule sets:** the term is 0 bits and a late head is no head.
- **A schedule with no offset** keeps today's behaviour: its row says the head's time isn't held, and it gets no small figure.

In NET-403: W is read from the schedule's parameters (BOUNDTIE builds the field, no default). Say in your next line whether #9501's reader takes it as a parameter today or needs a change, and what the world's year reads at W = 1 beside W = 0 for the `design` rows.

