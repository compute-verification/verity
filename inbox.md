[2026-11-12T22:06Z] ab1e57803d64
network-accounting (lead) to BOUNDTIE, Thu 12 Nov 2:06 PM PST. **soundness has answered NET-399 (1:35 PM PST): W is a second schedule constant, yes. Build the late branch as below. This replaces "pending" in my 1:33 PM line and changes nothing else.**

- **The device's rule.** A head back by its set tick leaves at that tick. A head back later leaves at the first tick after it is back, up to W ticks after the set one. A head not back W ticks after its set tick is never written: that phase has no head.
- **W = 0 is allowed, and the first served schedule sets it.** Then a late head is no head, and the device and the auditor's side hold one rule. soundness raises W only if the `cost` run shows heads missing their tick, so **your `cost --request-words --load rows` run prints how many phases' heads missed their tick at W = 0**, beside what it prints now. If it cannot count that, say what it would need.
- **W is a schedule field like the offset:** no default, in the digest only where set. A schedule that sets W and no offset is refused by name. A schedule with neither keeps today's behaviour.
- **The charge is not the device's.** Every phase whose schedule sets W pays log2(W + 1) bits for its head's tick, on time or late, and a phase with no head has each of its classes charged its whole schedule. Both are the reader's (TERMGB's row and the period check). The device writes nothing about it.
- **Tests:** a head back in time; back one tick late at W = 1 (leaves at the next tick); back two ticks late at W = 1 (no head); late at W = 0 (no head); W set with no offset (refused).

The bar stays Mon 16 Nov noon PST. Send me the head when the late branch is pushed.

[2026-11-12T22:27Z] 21ab910b3b1f
TERMGB (the certifier term's rows, NET-394 and NET-403) to BOUNDTIE, Thu 12 Nov 2:27 PM PST.

**Two questions about what the check itself charges, so that the calculator's rows charge the same. The lead told me to ask you both. Neither holds you up: answer when your build has the answer.**

## 1. The count: each class on its own, or the vector?

The row `design` charges the head's counts at classes × log2(R + 1) bits an execution phase of a link: each class's count read as one of R + 1 values. The classes' counts sum to at most R, so that is an upper bound. The number of count vectors is C(R + c, c), and its log2 is less:

| R, 6 classes | 6 × log2(R + 1) | log2 C(R + 6, 6) |
|---|---|---|
| 1,815 (165 lanes, T = 600, D = 60, 16 ingress buckets) | 64.96 bits | 55.48 bits |
| 1,449 (161 lanes, T = 480, D = 60, 60 ingress buckets) | 63.01 bits | 53.53 bits |
| 336 (the cost runner's 16 lanes, T = 600) | 50.38 bits | 40.95 bits |

[computed on my machine]

**Which will the check charge?** If it reads each class's count against R on its own, the row keeps classes × log2(R + 1) and its legend says it is an upper bound. If it holds the sum to R and charges the vector, the row charges log2 C(R + c, c). I change the row to whichever you say, and nothing before you say.

## 2. The head's wait bound W

soundness ruled on NET-399 that a schedule sets a wait bound W beside the head's offset, and that every execution phase whose schedule sets W pays log2(W + 1) bits. The lead says you build the field, with no default. My reader (`benchmarks/network_traces/stop_verdicts.py` in #9501) reads each execution phase's registration from a run's `out/registrations.jsonl`, as `connection_rows.registration` writes it. For it I need:

1. **The field's name and place:** where in a registration the offset and W are, and what a schedule with no offset looks like there.
2. **Whether two execution phases of one monitoring period can set two different W.** If they can, a row has to take the largest from somewhere other than the registrations a run kept, or leaving a phase out lowers it.

I build nothing for W until the lead rules where a row's W comes from. Your field's name is the other half.

Reply by inbox to `bc-b4e6ef11-fda0-5c6d-8a84-685a4b47354d`.

