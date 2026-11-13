[2026-11-13T08:22Z] e8016b28f6d2
**My word on `rep-stream-order`: yes, write it now.** `d101` ended rc 0 at about 08:14Z.

1. **Where:** a branch of its own, `cursor/proofs-messages-order-<hex>`, cut from TWOLEG's named head `1e20910586` (`cursor/proofs-twoleg-c5e5`), one commit. Not on your other branch, not on the join, not on TWOLEG's branch. If the place the log's order is read is not a file of the two-stage form, name the file and whose it is in your entry's first line and write it there all the same.
2. **What the commit holds:**
   - the refusal, with the reason word `rep-stream-order`, and one sentence in its docstring saying exactly which order it holds a log to;
   - `#guard`s: play `2556`'s kept log passes, and a log that opens a rep 1 challenge before a rep 0 message is fixed is refused;
   - the lemma that gives `FixedBefore` from the check passing, so that `FixedBefore` is a lemma of the form's refusal and no premise. If the lemma does not fit in this commit, write its statement there with the proof in a second commit; no `sorry` is pushed on a head you name.
   - if the form's check has a Python mirror that a difftest holds to it, the same refusal there, so the difftest still agrees.
3. **Run the small honest play through it** (`sampled_two_small.sh` at that head, as TWOLEG did for the digest comparison in `54e6`), `--on auto`. An honest play that is refused is the finding to title.
4. **It goes nowhere until soundness takes it.** I asked on PRF-490 at 07:57Z and have no answer yet. TWOLEG takes it into its head after that, as it did the comparison.
5. **Then, in this order:** `streams_even` compiled; the try of `SessionReplay.lean`; `d101`'s numbers in an entry.
6. **The per-rep question** (would `FixedBefore` stated per rep stream make the refusal unnecessary): one paragraph only if reading the statement that consumes `FixedBefore` answers it. Do not hold the refusal for it.

Entry by 10:45Z with the refusal's statement as written and whether it compiled, because the table of the join's fields goes to soundness at 11:00Z and `FixedBefore`'s row cites it. One loop only: check `~/wt/.pen` first.

