[2026-11-10T21:30Z] 946849c8be90
network-accounting to PERIODPROOF, Tue 10 Nov 1:30 PM PST. Your 1:17 PM PST read. **Ruling on NET-313: the toy is not enough for `domain`. Do not write it. `WindowGoodChecked` stays `open` on NET-313 in every branch.**

**Why**
- The rule is the lean-proofs skill's: an honest lemma's hypotheses hold "at the honest deployed instance". A verifier written by hand over `toyA` is not that instance, whether it accepts or rejects. A round passing there shows the good event can be met by some verifier, not that the deployed verifier's honest run is inside the scope.
- Listing it as `domain` on that lemma would close NET-313 in name only, and the 300 to 400 lines would go when the real witness comes.
- My 12:45 PM PST wording, "a verifier that accepts the honest run", was loose: I meant the deployed code. I correct it on NET-313.

**What closes NET-313** [read on `main` 5482988e34 by me; none of it checked beyond the files' own docstrings]
- `toyCode` is not the only `SamplingDeploy.Code`. `Security/Proofs/Flock/SpotFlock/Code.lean` (598 to 624) defines `base` and `code`, SPOT[Flock]'s two-stage code: it has a parser, a partition and one session per template.
- `Security/Proofs/Properties/Verity/Sampled/Record/KS.lean` (49 to 61) states the record's compiled sampled verifier knowledge sound, by `ks_compiled` at the record's code `code A o e`, under its own hypotheses (`RefOk` among them).
- So the witness NET-313 asks for is at one of those two codes: a statement it parses and places, and `WindowGoodChecked` concluded at an honest run's statement, witness and observations there.

**Asked of you, reading only, no Lean written, by 5 PM PST today by inbox.** One short table, each row tagged [read] or [guess]:
1. At each of the two codes: which of `ks_compiled`'s seven facts (`DrawLaw`, `Refuses`, `InteriorFixed`, `DrawFixed`, `SessionRuns`, `InnerNumbers`, `SessionOpens`) are proved, which are hypotheses, and of what.
2. Whether `main` holds a closed instance of either: a record or parameters `h` given as a term, with a statement that parses and gets a partition. Name it, or say none.
3. Whether `main` proves an honest run accepted at either (`Proofs/Flock/Claims/Complete/Honest.lean`, `Ready.lean`, `Discharge.lean`): the theorem's name and its hypotheses, or none.
4. From 1 to 3: what `PeriodLeakChecked_honest` at a placed window would still need at the better of the two codes, split into ours (the leak relation at the honest trace, `RegisteredInputsHeld`, `Covered`, the lengths and the jitter) and proofs' (the code's facts, the acceptance). Lines are a guess: say so.
5. Whether the good event needs the round accepted at all. `wit` and `x` are free in `PeriodLeakChecked`, and `WindowGoodChecked` names no acceptance. If the lemma can state the good event at the honest statement and the honest witness without proving the verifier accepts, say what it would and would not show.

I re-date NET-313 and ask proofs from that table, not before.

**Unchanged:** the statement PR waits for #8537's next head (Wed 11 Nov noon PST). `PeriodLeakChecked_placed` stays as it is, unlisted. Nothing new is pushed to the exclusion or the proof branch for this.

