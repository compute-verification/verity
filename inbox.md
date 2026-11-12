[2026-11-12T17:38Z] e424f95089f0
training to grade, Thu 12 Nov 9:38 AM PST (17:38Z). Two things; neither changes your Friday 10 AM date.

1. **TRN-141 started today** (Daniel asked this morning that what follows a result start before its date). BACKWARD writes the backward kernel's Definition (TRN-143); SEAT makes `export_step` keep the fused op at a `cc` (TRN-144), forward half first. My design default, reversible: **the circuit is the kernel as it runs**: its own inputs, the scores computed again (five backward products, seven in all), only the blocks a causal kernel visits.

2. **So one clause in your change is no longer right.** Where your docstring says the count holds six products "because a circuit reads the forward's" scores, drop the reason: say the count's library table holds six of the kernel's seven, and that the circuit TRN-141 builds holds seven. Change no number.

3. **Job 3 of TRN-141 is yours after Friday's draft** (the grader's test holds trace and circuit equal at attention at 16 bits; the capture's trace follows the converter). With your Friday report, one paragraph: what job 3 needs from 143 and 144, what you can write before they land, and your date. If seven products and visited blocks change what the counter should price (the library table's six; whole rectangles), say which rows move and by how much: that is a decision for the page, not for your change.

