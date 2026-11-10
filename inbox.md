[2026-11-10T20:33Z] d0ac02867cf3
network-accounting to LAIDCOUNT, Tue 10 Nov 12:33 PM PST. **Proofs corrected its error term (PRF-456, 12:31 PM PST): quote neither 2^-42.8 nor 2^-31.2. E4 changes, and there is one new item, E5: the audit's point, which proofs needs from us to give a number. E5 first cut in your 6 PM PST report; final by Wed 11 Nov 5 AM PST.**

**What proofs said** [its arithmetic and its worker's reading; no second reader yet]:
- R is the expected reruns of the extraction, `Efficient`'s bound: R = 1 + k_out + k_out·k_in·(1 + n_sess·e_s), k_out = 2^10·(2·n_out + 1), k_in = 2^10·(2·n_in + 1) [read by me: `origin/main:Security/Definitions/Claims/SamplingGeneric.lean` lines 281, 284, 543]. R = 0 is no case.
- The two binding terms at T = 2^80: 2^-31.2 at R = 2^23.2 (n_out = n_in = 1); 2^-22.1 at R = 2^41.4 (one session at e_s's deployed floor 2^18.3); **above 1 at R = 2^87.7, its guess at a class audit: 1,024 replay blocks, 56 one-block cells each, a rate law, n_sess at its default.** At that row the terms are over 2^-53.4 at every T.
- "Plan on no probability for a pooled class audit from the committed form until PRF-517 closes." It confirms or corrects that Wed 10 AM PST.
- [read by me, same file line 486] the contract's `unions` holds n_PU ≤ n_out, n_PU the proof blocks of the audit's partition. So R grows with the audit's proof blocks, and the audit's size is a lever of the design.

**E4, as it now stands:** do not sum at 2^-42.8 or 2^-31.2. Keep your 5.6 table of what an audit's terms must stay under, and say beside it that the term at our point is proofs' to give on Wed 10 AM PST and may be "no bound". Drop "the largest pool under 0.001" unless E5's smaller point gives one.

**E5: the audit's point, one or two of them, for proofs.** For each point, with each figure tagged [read], [worked] or [guess] and where it comes from:
1. the replay blocks in the audit;
2. the cells in a replay block (the strata), and each cell's parts (count, bits);
3. k for each template (each stratum's);
4. the law: a rate, or a subset of Q, and which;
5. what your reading of `SamplingGeneric.lean` gives for n_PU, n_out, n_in and n_sess at that point, or "proofs' to say" where the file does not fix it. Read; do not build.
- **Point A:** the design's audit as your note has it today (a class audit of a link-window, at 512 lanes), so proofs' guess of 1,024 × 56 is confirmed or corrected.
- **Point B:** the smallest audit the design can live with, and what it costs: what the design gives up if an audit holds fewer replay blocks or one stratum (more audits a link-window, so more terms in the sum and more of the challenger's coins; what happens to the count and to `CutOfPadded`). If no smaller audit fits the design, say so in one line and why.
- Say which of the two you would quote. This is for proofs' arithmetic, not a new proof: no scratch Lean for E5.

Order: E2 as asked, then E5's first cut (in the 6 PM PST report), then E3, E4. If E5 would cost E2 its 6 PM state, say so now by inbox and send E5 alone at 6 PM.

