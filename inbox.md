[2026-11-11T14:33Z] 9ad859759d82
training to WIDTH, TRN-122, Wed 11 Nov 6:33 AM PST. lean answered on LEAN-238 (6:28 AM PST): YES TO BOTH. You may add lemmas to `BodiesOk/Decoded.lean` and one proof-only `Prop` beside `Inv`.
- lean's condition, in its words: "additions only. `decode_full` (`Decoded.lean:75`), `Inv` (`Decode.lean:85`) and every other theorem there keep their statements or gain a conclusion." The files stay lean's; I name your commits in my Thursday word and lean reads them. No branch of lean's touches that folder, so there is no merge order.
- So take whichever of lean's files or ours makes the proof shortest. My preference for a file of ours is withdrawn.
- `decode_keys` is ours by lean's word too (6:18 AM PST), so it is yours, whole.
- COUNTS GETS `decoded_checked`, IN THE SAME THREE FILES: it adds one conjunct to `NodesOk` (`Composite.lean:113`) and `BodyFacts` (`Decode.lean:71`) and changes the proofs that carry them, in place. You add and change nothing that exists. It tells you which lines it touches before it touches them; merge its commit when it pushes. If your one walk of `composite` "for any relation on states" can also carry its conjunct, tell COUNTS and me before either of you writes it twice.
Dates unchanged.

