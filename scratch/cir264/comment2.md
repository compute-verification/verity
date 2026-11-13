**The sampled accept's kernels ran on a card and store the Definition's reference word for word on every request inside its domain: 409 requests of 1 + 5 rows, 80,226 words compared, 0 differ (run `r20261113-034537-8871`). Off the card, the registered `SampledAccept_v1{K=5,V=154880,TILE=8192}` evaluated on 11 of them gives the card's count and tokens: 264 words compared, 0 differ (run `r20261113-034958-1553`). PR_SENTENCE**

circuits' worker ACCEPT, STAMP.

**What ran.** vLLM `d9105ea80`'s `rejection_sample` kernels, launched alone with the arguments the sampler gives them, on node A's card 0 (an H200, compute capability 9.0), at GLM-5.3's vocabulary of 154,880 and five draft tokens, one request of 1 + 5 rows a launch. No model was loaded: a request's operands are its logits after top-p, its draft tokens, seed, positions and temperature. The tree was `903c545eb3d3`. One lease on card 0 with cores 0-9 declared, from 03:46:10Z to 03:47:35Z (85 s); nothing else of ours touched a card.

**What I read in the capture's run.**

- 410 requests (the 10 hand vectors and 400 random ones), 2,460 rows. 409 are inside the Definition's domain. Of those, 0 differ: 80,226 words compared (each draft row's tile maxima and sums, the accepted count, the appended tokens, the log-sum-exp of the last position tested), 0 differ.
- Accepted draft tokens a request, over the 410: none 236, one 57, two 37, three 19, four 17, all five 44.
- Each request was launched three times (the sampler's wrapper twice, the four kernels directly once, into buffers filled with `0xA5` first): 0 requests disagree between launches. The 410 requests as one batch, their state rows shuffled, give every request the tokens and count of its own launch: 0 differ.
- The four kernels' PTX and machine code in Triton's cache on the node are the ones the Definition was read from (SHA-256 of the PTX's instructions and of each cubin's code section; the PTX is kept in run `r20261112-135317-f990`).
- The gate held: H200, capability 9.0, vLLM `0.28.1rc1.dev472+gd9105ea80`, Triton 3.7.1, and the two source files of the pin by SHA-256.

**The one request outside the domain, `halves-split`.** It is the hand vector where lane 0 of the rejection kernel's warp accepts the first draft token and lanes 16 to 31 refuse it, which the kernel leaves undefined. On the card its count is 6 and its six tokens are lane 0's, as the reference has them; 1 of its 200 words differs: the kept log-sum-exp of the last position tested is `0xc1af360a`, where lane 0's arithmetic gives `0x40e00000`. The PTX interpreter I used off the card had given a count of 2 for it. So after a split the card's sums are neither lane 0's arithmetic nor my interpreter's. The tokens agree here because rows 1 to 4 of this vector put all the mass on the draft token [inferred from the vector; I tested no other split]. The Definition's domain stays as stated: a request whose halves split is outside it.

**The replay off the card.** The capture keeps every request's operands and stored words, so the comparison runs again without a card as the Definition changes. Run `r20261113-034958-1553` replayed the kept 410 requests at the same tree: the same counts against the reference (409 inside the domain, 80,226 words, 0 differ), and the registered Definition evaluated on the first 12 kept requests that are inside the domain, which is 11 (nine hand vectors and two random requests; the tenth hand vector is `halves-split`). For each, the Definition's 12 output words (five verdicts, the count, six tokens) are compared with the card's, where a verdict is read off the stored count, and with the reference's: 264 words compared, 0 differ. One evaluation of the Definition at this size takes over a minute [computed from the run's 16.5 minutes], which is why it is 11 requests and not 409.

**What this does not show.**

- That a serve launches these kernels at this specialization (one-hot draft, fp32 noise, no block verification): that is read from the pin's source, and the capture launches the kernels itself.
- Anything about a split other than the one hand vector.
- The masked row the `logits` port reads: its writer is CIR-273.

**Runs.** The capture `r20261113-034537-8871` on node A: launched 03:45:37Z, started 03:46:05Z, 90 s. The replay `r20261113-034958-1553`: waited 10.5 minutes for a check slot (03:50:44Z to 04:01:15Z), held it 16.5 minutes (to 04:17:46Z). CC_SENTENCE
