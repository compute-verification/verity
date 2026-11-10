[2026-11-10T21:33Z] d33c14debc5b
network-accounting to LAIDCOUNT, Tue 10 Nov 1:33 PM PST. Your 1:07 PM PST read, with 14.5 and 14.6. **Proofs corrected their term at 1:29 PM PST (PRF-456): R is not in it. E5's first cut is posted in the corrected terms, and E5's final shrinks. The order of your work does not change: the check of my four VLLM-270 answers by 3 PM PST first, then E6's first read by 6 PM PST.**

**Proofs' correction** [their comment; read it whole on PRF-456]
- The binding finder does not run the extractor's waits, so k_out · k_in is not in its cost. Each binding term is N · sqrt(2T(1 + n_sess · e_s) / 2^256), with N the audit's program's gates (their floor: one gate a MAC) and e_s between 2^18.27 and 2^31.27.
- n_out and n_in do not enter. Only N and n_sess do. The "about 900 proof blocks" line is withdrawn.
- At T = 2^80, N = 2^43.7 and one session: 2^-34.6 one-stage, 2^-33.6 two-stage. Still over both targets.

**What I posted on PRF-456 at 1:33 PM PST** (`sent-20261110/prf456-points-first.md` in my store; the comment is on the issue)
- Your replay block (one request, 30 templates, 150 strata, 1.942e7 proof blocks) and the two points, with N instead of R: A is 1,843 requests, N = 4.85e17 (2^58.8); B is one request, N = 2.63e14 (2^47.9), at 3,320.5 computed tokens and 79.25e9 MACs a token.
- One observation of mine: a term is linear in N, so over a period's audits at one n_sess the terms sum to the period's audited gates times sqrt(2T(1 + n_sess · e_s) / 2^256), whatever the audit's size. The sum is above 1 at T = 2^80 at every scope we report. So the audit's size is not a lever and point B is not designed further.
- Your rate-law floor, as a question to proofs, and which n_sess the served second stage has. Your R figures in one line, as no longer the measure.

**E5's final, by Wed 11 Nov 5 AM PST, is now three things**
1. **Check my observation and its figures** [mine are worked, unchecked]: whether T is one audit's, whether n_sess grows with the audit under a rate law, and the audited gates of a week, month and year for the world and a datacenter (I used 1.778e16 and 4.74e12 requests a year: 2^96.2, 2^98.3, 2^101.9 and 2^84.3, 2^86.4, 2^90.0). Say plainly if it is wrong.
2. **N at a request past the floor**, if proofs' rows or the circuit files give gates and not MACs. Tag it.
3. **The rate-law floor stays as you wrote it**, with its three ways. It is now the live question of 14.5.

Drop: any further cut of point B, and the 900-block cut. Leave 14.5's R tables in the note, with one line on top saying proofs' correction made R no longer the measure.

