[2026-11-14T19:11Z] 62d539ec3d1c
## Coordinator, Sat 14 Nov 19:11Z: CALLER: soundness's two changes

From the proofs lead, Sat 14 Nov: two changes from soundness (TOP-415, 11:09 AM PST), both taken. Do them with your step 3, and say so in your report.
1. `acceptInferenceOnly` (and `stage_held`) take only the FORWARD half as their premise: "the caller's check returns no code, so the record keeps `CallerHeld`". Name that half as its own `Prop` (for example `Properties.Remote.CallerCheckSound`), keep `CallerCheckMeans` (the iff) as the exact statement to prove, and prove the half from it in one line. The guarantee's open premise is then the smallest one.
2. The lock lists that open premise under `acceptInferenceOnly` with owner proofs and the day Tue 17 Nov, and your note has a section "What changed in the guarantees" naming it, for the list Daniel sees at the branch's close.

