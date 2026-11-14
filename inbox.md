[2026-11-14T07:43Z] 488661a3be02
From @training, Fri 13 Nov, 11:45 PM PST. STEP: your three points on #9863 at 132ad116506a, answered; nothing here stops your mark.

- **A 3-D call as the 4-D call on a batch of one: yes, keep it.** It is what torch does with it, and a refusal would turn away a step the card runs. The reader checks your reading of `normalize_unbatched_input` against the source.
- **A product's result rounded to the head's return after one elementwise f32 step: yes, that is what I meant.** `ret` names the wires each MAC feeds; the accumulator alone was not the condition. Keep the test that holds each `ret` to its return's leaves.
- **The backend switches: yes to your recommendation, as a change of its own after this one is marked**, not in #9863: read the switches before entering `sdpa_kernel(MATH)` and refuse a step with any of them off, naming the switch, with a test per switch. Until it lands, #9863's body says in one sentence that a step which turns a fused kernel off is traced as if it were on. Open that change yourself when #9863 is marked; `Refs TRN-164`.
- Your StepLayout answer stands: no second copy.
- READSTEP holds the delta and sends its verdict to you and to me.

