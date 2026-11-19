[2026-11-19T23:35Z] 23c9ed6058b0
From vllm lead, 3:37 PM PST. Circuits one-lane change HAS LANDED: one-circuit at 9c3c0ded75e (ScanSlots_v4, one scan template a lane; every window digest moves; the q/k/v slices an allocation each; five prove-path files changed). So cut your provable-check branch and your processor test from a one-circuit that HOLDS it, and your step 3 merge takes it in: say what it moves in the served window at class 4,27,2 (template count, the three small templates still the three?).

