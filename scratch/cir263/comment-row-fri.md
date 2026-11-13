**The three captures of record for the MTP draft layer's kernels ran on an H200 on Fri 13 Nov, 15:45Z to 15:52Z, at branch head `ce09de03c318`. Each sealed its record with `ok` true. Together they made 52,846,709 word comparisons (my sum) and found 0 words that differ.**

- **`eh_proj`**: run `r20261113-154513-8180`, record digest `41442bea4181c34d43a40ac11596b2e9bb85456f2b8c8376e24ebaee2cef5e1b`, sealed.
  - 24 launches from M = 1 to 8,192 tokens. 4,458 rows of 6,144 words each (every row up to M = 1,024, and 384 rows of each of the two launches above it): 27,389,952 words compared, 0 differ.
  - Route, the same as the first run scanned: split-K 2 at M 105-111, 129-159, 161-223, 272, 288 and 336; no split elsewhere up to 8,192.
  - The two split launches (M = 105 and 192): 297 rows, 1,824,768 words, 0 differ from `CublasSplitKGemm_v1{K=12288, N=6144, SPLIT_K=2, DOT=HopperBF16WgmmaDot16_v1}`. The unsplit launches are compared with `Gemm_v2{K=12288, N=6144, DOT=HopperBF16WgmmaDot16_v1}`.
  - Split order: the job's probe is 32 crafted words a split launch. Two contiguous halves of K, summed in f32, fit all 32; strided halves miss 1 word and bf16 partials miss 2. Computed afterwards on a CPU from the words the run kept, not in the job: of the 1,824,768 split words, 5,653 differ under strided halves and 675,620 under bf16 partials.
- **`eh_norm`**: run `r20261113-155013-668e`, record digest `5584f173348e50978eb87d2309a56f096521c291f473950f27f76b8ff446fc11`, sealed.
  - Two launches of 1,024 positions, each half of each compared whole: 4,096 rows, 25,165,824 words, 0 differ.
  - The kernel's PTX hash on the card equals the pinned one. Each alternative reading of the kernel is refuted on at least 16 words.
- **`indexer_logits`**: run `r20261113-155125-8b02`, record digest `253714da690d3c2e1b3ede3fc9a2a99ee20cf06a23688472f6302d31060bd35d`, sealed.
  - 290,933 comparisons, 0 differ, 0 unresolved.
  - **A correction to my comment of Fri 05:00Z, which said every logit was compared.** That holds for the 9-request launch: all 50,664 logits. In the 64-request launch the capture compares 59 of its 384 rows whole and 50,000 random (row, key) pairs: 179,605 of its 702,153 logits. The 290,933 counts repeats; the distinct logits compared are 230,269 (my count from the kept words).

What a word is compared with: each Definition's NumPy twin. The Definition's own evaluator runs on a sample (177 coordinates for `eh_proj`, 102 rows for `eh_norm`, 144 logits for the indexer) and equals the twin on all of it.

Each record names the vLLM it ran, `0.28.1rc1.dev472+gd9105ea80`, which is the pin's commit. I recomputed each digest from the fetched record. Each capture held one lease on card 0 (169 s, 30 s and 20 s), and the card was idle at each start.

Not done yet: `eh_proj`'s two bindings in a sidecar, and the pull request (by Mon 16 Nov 16:00Z).
