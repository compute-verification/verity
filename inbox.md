[2026-11-13T10:09Z] 33831f8904ba
DGRAD, Fri 13 Nov 2:11 AM PST (TRN-156: LoRA and grow held, before-and-after 27 of 33 equal, two things I did wrong)

**All four LoRA entries and grow at step 164 are held at the dgrad head. 27 of main's 33 held entries print every line the same before and after the `per_row` commit. The other six are rerunning, and the draft opens when they're read.**

1. **One more route fix** (commit 57411bf29c). At rank 1, `.contiguous()` doesn't copy a transposed view, so q4r-mlp's B gradient was still the transposed product at 6e15d4ba1a. The route and the reference now build B with `clone(memory_format=torch.contiguous_format)`, and the comment says why.

2. **Held at 57411bf29c** (RAN r20261113-092839-1bcc, 12:10 wall, 2.7 GB). Each entry is `equal, 1 expected difference, 1 named line of the record's program alone`:
- s-t20-1.1x-dgrad-q4r-mlp: 288/288 uses, 57/58 products;
- s-t20-1.1x-dgrad-q4r and s-t20-1.1x-dgrad-q2r: 672/672 uses, 57/58 products each;
- s-t20-1.1x-dgrad-vote: 672/672 uses, 51/52 products.

3. **Grow at step 164 held** (RAN r20261113-090724-d211, at b5a022a0e4 on the scratch branch `cursor/dgrad-grow-scratch-8002`: 57411bf29c's route at rank 84, with `STEPS = 165` and a grow entry; 40:18 wall, 3.6 GB). It is `equal`: 896/896 uses and 72/73 products. Its entry waits on #9531's `step` field, as ruled. The scratch branch is not for merging.

4. **Suites** (RAN r20261113-085752-b441 at 6e15d4ba1a): all four suites pass, training_slowdown 647 passed. The change after that touches only dgrad's B, and its tests pass locally. A suites run at the final head goes with the draft.

5. **Before and after, the `per_row` commit** (your 3):
- before: main 9182900fb9, chunks r20261113-090508-657a, -090539-ca77, -090608-7eeb, -090623-c835 and -090652-d7df;
- after: 6e15d4ba1a, chunks r20261113-090157-df0a, -090230-6379, -090246-f201, -090304-24f2, -090334-cf86 and -090405-d556.
- Result: 27 entries are on both sides, and every verdict and every printed line is equal.
- Before-chunk r20261113-090437-1d02 hit the hour and printed nothing: xl-t20 was first and took it. Its six entries are rerunning: r20261113-100836-b898 (xl-t20 alone, cpu-3, 28 GB) and r20261113-100742-cfe2 (the other five, cpu-14). Each has an after line, all `equal`.
- The commits after 6e15d4ba1a change only `routes/dgrad.py` and the dgrad reference, which no main entry reaches (INFERRED from the diff). So the after lines stand for 57411bf29c.

6. **Two things I did wrong, both RAN:**
- Under `--on auto`, four runs went to vy-nebius-cpu-2: r20261113-090304-24f2, -090539-ca77 and -090608-7eeb, which finished, and -092803-8bbc, which I cancelled. Since then every run names its node.
- The after chunk with xl-t20 (r20261113-090157-df0a) peaked at 24.8 GB on a 12 GB ask. xl-t20's rerun asks 28 GB.

7. **The PR.** None of my tools opens a PR. I set `cursor/dgrad-route-8002` as this agent's active branch so the environment opens it as a draft against main. If it doesn't appear, the branch is pushed and ready for you to open.

8. **Two questions for the PR's table.** TRN-156 gives 14 frontier points over the six runs: grow has 6, and the per-run split of the other 8 was in the original brief, which I no longer have. What is it? And what should the "what binds" column hold: the slowdown's binding term for each run's step, or something else?

