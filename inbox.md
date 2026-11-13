[2026-11-13T07:24Z] c805bc6fbd5e
QATREST, Thu 12 Nov 11:25 PM PST (TRN-155, both parts)

**All thirteen QAT frontier runs are library entries of route `qat`, each held to its own run, in draft PR #9686 at head 17023468876a; nothing is left out and no step is refused.**

Verdicts:
- RAN r20261113-065131-29ea (vy-nebius-cpu-5, at 3d08bc603b3b): of_record on the 13 new entries and the 3 on `main`, last line `16 entries: 16 equal, 0 differ, 0 refused.`, exit 0. Every entry reads `equal, 1 expected difference, 3 named lines of the record's program alone`: the `norm` rule, and the three `record_only` lines `-top3`/`-top4` already name on `main`. No new rule.
- RAN locally: the three existing entries' config digests at the head equal `origin/main`'s.
- RAN r20261113-071445-322f at the head: `suites.py benchmarks/training_slowdown --changed 2e99b1cf48b8 --quick`, `1 suites: 1 passed (0 cached), 0 failed` (656 passed).
- RAN: ratchet gate at the head `1 change(s), 0 failed`; ruff `All checks passed!`.

Your six changes, as written:
1. Two dials, both in `LATER`: `grad_code` (fp32, rtn4, rtn8) and `low_state` (bool: bf16 latents with the stochastic update, plus Muon's int8 momentum with bf16 row scales).
2. No `grad_scale` dial. `-top3-grad4`'s config and the body's table say its scales are bf16 where the program kept f16, both 16 bits.
3. The code functions are in `routes/_codes.py` (`row_codes`/`row_values`, `round_bf16`, `round_int8`), on tensors with nothing of `qat`. It is its own commit (1634d4a37e).
4. `Muon(low_state=...)` puts its key in the defaults only when on, in the grow route's form, in its own commit (a9bce2c19a). A test holds Muon's defaults and state at the defaults to `main`'s keys.
5. The draw order is in `qat`'s docstring: per latent, the momentum's rounding, then the update's.
   - The continuation test passes: a stopped `low_state` job continues to the unbroken job's latents bit for bit, and to others when given the generator as it stood at the start.
   - `run.trace_step` runs only after `run.train` returns, so no trace runs between steps; that is one line in the body. READ run.py.
   - RAN locally, and asserted in a test: the capture leaves torch's generator untouched.
6. The trace's measures are reported RAN in the body, from each counted record's trace at the start's shape.

What the trace measured (RAN r20261113-065131-29ea). Top four blocks are 12,845,056 entries in 21,504 rows:
- Gradient cut: rtn8 103,104,512 (8 a weight + 16 a row); rtn4 51,724,288 (4 + 16 a row). `-top3-grad4`: 38,793,216 (4 a weight + 16 a row over 9,633,792 entries and 16,128 rows).
- Carried state under `low_state`: 308,625,408, which is 16 for the latents + 8 for the momentum's codes a weight, plus 16 a row for their scales. `-top3-grad4` (fp32 state): 616,562,688, which is 32 + 32 a weight.
- Seeds: none charged (`tau_other_bits` 0 in all five). The random values reach the next forward only through the latents' codes, and no token op reads a random op. I changed nothing for that.

One thing beyond the design (INFERRED, then RAN by test): torch's `Optimizer.load_state_dict` casts every state tensor of a float parameter to the parameter's dtype. That turned a continued job's int8 codes into bf16: same values, so the digests matched, but 16 bits carried. `Muon.load_state_dict` now casts the codes back to int8, inside the Muon commit; a test loads a state and checks the dtypes. The continuation test caught it through the counter's list of new weights.

Every entry's "what binds" is `library`'s `needs a record: the entry's E keeps less than the trained weights...` (RAN r20261113-065131-29ea), as for the three `qat` entries on `main`.

Done: both parts, every verdict, the suites in the body, head named (ahead of the Sun 15 Nov 6 PM date). Left: nothing unless you want changes; I keep merging `main`. No run in flight.

