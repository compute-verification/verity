proofs (lead), Tue 10 Nov 3:59 PM PST (23:59Z). **Candidate R: replace the two-run binding finder by one that repeats the prover until two runs open a node differently. If it holds, the binding error is about $2^{-96}$ to $2^{-117}$ at one session an audit, with SHA-512, no random oracle and a premise that is on `main`. It was derived by the coordinator in one sitting and nobody has checked it. Break it or prove it.**

Daniel, 3:51 PM PST: "What is the fastest way you think we can get clarity on whether we have something working? … I want to know straightaway because I'll be sending this to people."

## Why today's bound loses so much

Today's finder (`Bconf`) runs the prover twice at one tape. If the bad event has probability $\delta$, it finds a collision with probability about $(\delta/N)^2$, so the theorem reads $\delta\le N\sqrt{\beta'}$, and the linear premise then gives $\beta'\le E[\text{cost}]/2^{256}$. The loss is the square root and $N$. A finder with **constant** success pays in time and not in probability: under either premise form, a finder of cost $q$ with success at least $\tfrac14$ needs $q\gtrsim 2^{254}$, so $\delta\lesssim(\text{cost of one try})\times(\text{tries needed})/2^{254}$.

## The claim

Setting: the one-stage game of `Compiled.KSReg` (the draft's Theorem B.2). $C$ fixed, the prover's tape $\omega$, the commitment $c$ sent before any coin (so $c$ is a function of $\omega$), the committed trace $\hat\tau_\omega$ the plurality of Definition 3.14, $N'$ the number of boundary nodes (at most the gates $N$).

**Claim R.** There is an explicit finder $F_R$, built from the prover, that runs it at most $(L+1)R$ times in expectation-counted runs, each with the base argument's extraction, such that
$$\Pr_{\omega,r}\big[\text{accept}\wedge Q(r)\cap T_{\hat\tau_\omega}\ne\varnothing\big]\ \le\ \underbrace{\kappa_s+\sqrt{k\gamma_s\Pr[B]}}_{\text{as today}}+2^{-L}+c_0\,(L+1)^2\,\sqrt{N'}\;\frac{c_{run}}{2^{256}}$$
wherever `ExpectedCR` holds at $F_{R^*}$, with $c_{run}=T(1+n_{sess}e_s)+n_{PU}\cdot$`openCalls` the expected SHA-512 calls of one run with its extractions, $L=128$, and $c_0$ a small constant (32 in the sketch).

## The steps (each is a place it can fail)

Fix $\omega$. For a node $u$: $p_u(\sigma)=\Pr_{r,\zeta}[u\text{ is opened to }\sigma]$, $P_u=\max_\sigma p_u(\sigma)=p_u(\hat\tau_\omega(u))$, $a_u=\sum_\sigma p_u(\sigma)$, $m_u=a_u-P_u$. $\delta_\omega$ is the probability over $(r,\zeta)$ of the bad event with every accepting session's extraction giving a witness; the rest of the bad event is the base argument's term, as today.

1. **A bad run opens some node to a non-plurality value.** The sampled block $S$ is incorrect in $\hat\tau_\omega$ and correct under the extracted openings, so they differ at a boundary node $u$ of $S$; $u$ is opened on this run, so $\hat\tau_\omega(u)$ is defined. Pick one such node canonically: $b_u=\Pr[\text{the run is bad with canonical node }u]$, $\sum_u b_u=\delta_\omega$, $b_u\le m_u$.
2. **A second independent run at the same tape disagrees with it at $u$ with probability at least $b_u$.** If the first run opened $u$ to $\sigma\ne\hat\tau_\omega(u)$, the second opens $u$ to a value other than $\sigma$ with probability $a_u-p_u(\sigma)\ge a_u-P_u=m_u\ge b_u$. Two accepted openings of one position of one $c$ to different values give a collision (`findColl`). So an ordered pair of runs yields a collision with probability at least $\sum_u b_u\,m_u\ge\sum_u b_u^2\ge\delta_\omega^2/N'$ (Cauchy–Schwarz). This is today's lemma with $\sqrt{N'}$ in place of $N$; no alphabet size enters.
3. **$M$ runs at one tape find a disagreeing pair with probability at least $\tfrac12$ once $M\ge c'\sqrt{N'}/\delta_\omega$.** There are $\binom M2$ pairs, each disagreeing with probability at least $\delta_\omega^2/N'$, so the expected count is at least 1 at $M\approx\sqrt{2N'}/\delta_\omega$. **The step from an expected count to a constant probability is not written**: the pairs are dependent. Routes: a second-moment bound; or split the runs in two halves, take the bad runs of the first half (about $\delta_\omega M/2$ of them, each with a canonical node drawn from $b/\delta_\omega$) and ask whether the second half opens that node to its plurality value (probability at least $1-e^{-b_uM/2}$ a node).
4. **Tapes.** $\delta=E_\omega[\delta_\omega]$. Dyadic levels $j=0,\dots,L$: some level has $\Pr_\omega[\delta_\omega\in(2^{-j-1},2^{-j}]]\cdot2^{-j}\ge(\delta-2^{-L})/(L+1)$. At level $j$ the finder draws about $R/(c'\sqrt{N'}\,2^{j})$ tapes and makes $c'\sqrt{N'}\,2^{j+1}$ runs at each. It does every level, so $(L+1)R$ runs in all up to constants, and it succeeds with constant probability once $R\ge c''(L+1)\sqrt{N'}/\delta$.
5. **The premise.** Each run costs $c_{run}$ calls in expectation (its extractions included; the runs' coins are fresh, the tape is shared). `ExpectedCR` at $F_R$: $\Pr[\text{collision}]\le(L+1)R\,c_{run}/2^{256}$. Take $R^*$ with $(L+1)R^*c_{run}/2^{256}$ just under the constant of step 4. Then $\delta<2^{-L}+c''(L+1)\sqrt{N'}/R^*$, which is the claim. Under `StrictBirthday` the same holds with the finder stopped at four times its expected cost (Markov costs $\tfrac14$ of a constant success, which is why a cap is free here and was fatal for the two-run finder).

## The numbers, with the sketch's constants ($c_0(L+1)^2=2^{19}$; the coordinator's arithmetic)

$\log_2$ of the last term, at $T=2^{80}$; each cell is ($k=1{,}507$, $k=12{,}343{,}986$):

| Point | $n_{sess}=1$ | $n_{sess}=30$ | $n_{sess}=n_{PU}$ |
|---|---|---|---|
| Floor, $N=2^{43.73}$ | $-116.9$, $-103.9$ | $-112.0$, $-99.0$ | $-101.1$, $-88.1$ |
| B, $N=2^{47.9}$ | $-114.8$, $-101.8$ | $-109.9$, $-96.9$ | $-90.6$, $-77.6$ |
| A, $N=2^{58.8}$ | $-109.3$, $-96.3$ | $-104.4$, $-91.4$ | $-74.2$, $-61.2$ |

At $T=2^{64}$ every cell is 16 better. With one factor of $L$ and tight constants every cell is about 11 better. The base argument's own terms ($\kappa_s\le2^{-203}$ and a fork term of $2^{-141.5}$ at $T=2^{80}$, TWOPROVE's reading, not checked by the coordinator) then matter and need their own check.

## What to attack

- **Step 3:** is there a distribution of openings for which $c'\sqrt{N'}/\delta_\omega$ runs find no disagreeing pair with probability near 1? What are $c'$ and the real exponent of $N'$?
- **Step 2:** can one run open a node to two values, or to none while the block is still "correct under the extracted openings"? Read `Freg`, `regVal` and the opening event in `Security/Definitions/Claims/SamplingGeneric.lean` on `main`.
- **Step 4:** the count of levels and the $(L+1)^2$.
- **Step 5:** `ExpectedCR` is asked of a finder with about $2^{254}$ expected calls, built from about $2^{150}$ runs of the prover. That is the usual reading of collision resistance (no machine of time $t$ finds a collision with probability above $t/2^{256}$), and it is not false generically ($q/2^{256}\ge q^2/2^{513}$ up to $q=2^{257}$). Is it a premise the lane can state, or does it read as weaker than what the draft says ("an explicit algorithm, built from the prover")?
- **The base argument's terms:** are they multiplied by the session count, and is $2^{-141.5}$ right?
- **Sessions:** $c_{run}$ carries $n_{sess}$. Can the finder extract one session a run?
- **Two stages:** the second stage's commitments are made after the first sample, so $c$ is not a function of $\omega$ alone there. What survives?
- **What it changes:** the event is Theorem 3.15's, unchanged, so Corollary 3.16, Claim 3.19 and Section 4 stand. The bound is restated (`KSReg` is locked), so it is Daniel's to approve.
- **The literature:** is this the argument of Chiesa, Dall'Agnol, Guan, Spooner and Yogev (TCC 2024) for Kilian's protocol, and what do they get for the constant and the dependence on the proof's length?

## How to report

One comment on this issue within 60 minutes of starting, leading with `HOLDS | FAILS AT STEP n | HOLDS WITH <change>` and the bound you can defend; the final by Wed 11 Nov 02:30Z (Tue 6:30 PM PST). Mark each fact [read] with file and line at the `origin/main` SHA, [derived] with the steps shown, [literature] with the theorem's number, or [guess]. `research msg linear comment PRF-542 --file F.md --worker NAME` (no `--as`). Your final message to the coordinator repeats the final comment in full. Build no Lean and run no suite on your VM; a small Python simulation of step 3 is welcome. Push to no existing branch.
