proofs (lead), Tue 10 Nov 3:32 PM PST (23:32Z). **Daniel wants 100 bits of security for SPOT's binding, a swarm on every option, and a deep audit of every other concrete-security number. This issue is the shared context; each worker's brief is its own sub-issue.**

## Daniel's words (3:20 and 3:21 PM PST)

> "this is totally unacceptable. … we need to rapidly use our resources to figure out … what all of our options are for increasing the security. I'm looking for 100 bits of security. There should be a swarm investigating our options. Happy to get creative, walk back past decisions, etc."

> "this is such an enormous smell … we need to do a deep audit to ensure that there aren't other smells in terms of our concrete security. Also note this is our current draft: https://app.notion.com/p/SPOT-security-definitions-413399515d9e8162bbb7f8de48ca23e4 … Should find differences and resolve them."

## What is wrong, exactly

**The statement.** The draft's Theorem B.2, which is `Compiled.KSReg` on `main` (`Security/Definitions/Claims/SamplingGeneric.lean`), bounds the chance that the verifier accepts while the sample hits an incorrect block of the committed trace by

$$\kappa+\sqrt{\rho\,\gamma\,\beta}+N\sqrt{\beta'}+\beta'$$

In the Lean: `εreg = L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + L.nG x * Real.sqrt ((Bconf H L Ω P x).Pr H) + (Bconf H L Ω P x).Pr H`. The two-stage form (#8512, in review) adds a second $N\sqrt{\Pr[B_{conf,2}]}$.

- $N$ is the number of gates of the circuit (`nG`).
- $\beta'=\Pr[B_{conf}]$: the finder `Bconf` runs the prover twice, independently, at one tape, applies the base argument's extractor (`runExt`) to both runs, and looks for a collision among the SHA-512 calls made (`findColl`). Its cost is both runs' calls, both parties', plus both extractions' calls with their reruns.
- The committed trace (draft Definition 3.14) is the plurality, over the verifier's coins and the extractor's, of the value each boundary node is opened to.

**The premise.** `Security/Definitions/Core/Interaction.lean:262-312` has two forms. `StrictBirthday H F := ∀ q, (∀ s, F.cost s ≤ q) → F.Pr H ≤ q^2 / 2^513` needs a hard cap on the finder's cost. `ExpectedCR H F := F.Pr H ≤ Ex (cost) / 2^256` takes an expected cost. The base argument's extractor runs in expected time only, so today's bound uses `ExpectedCR`, which is linear, and then sits under a square root with $N$ outside it.

**The finder's cost** (TWOPROVE's derivation, confirmed by ACCOUNTD; C5 on `cursor/proofs-bconf-cost-5987` states it, not on `main`): $E[\text{cost}]\le 2T(1+n_{sess}\,e_s)+2\,n_{PU}\cdot$`openCalls`, with $T$ the cap on one run's SHA-512 calls (both parties'), $n_{sess}$ the number of sessions of the base argument, $e_s=1+210k$ the expected reruns of one session's extraction, and $k=\lceil 4N_0/e\rceil$ from $1{,}507$ to $12{,}343{,}986$ (`Security/Definitions/Claims.lean:314-323`; $\kappa$'s term there is `2 * D.k x / D.cap x`).

**Where the bits go** (one term, $N=2^{43.73}$, $T=2^{80}$, one session, $k=1507$): $128-40.5\ (\tfrac12\log_2 2T)-9.1\ (\tfrac12\log_2(1+e_s))-43.7\ (\log_2 N)=34.6$ bits. Two terms: $2^{-33.6}$. A plain birthday bound at $2^{80}$ calls is $2^{-353}$, so the loss is the proof's, not a known attack.

**The numbers** (two terms, $T=2^{80}$, $k=1507$; script below). Columns: today (`ExpectedCR`) / a capped extractor under `StrictBirthday` with the best cap / that plus a proposed premise `Claims.AdaptiveBirthday` (ASK-272, not on `main`).

| Point | $N$ | $n_{sess}=1$ | $n_{sess}=30$ | $n_{sess}=n_{PU}$ |
|---|---|---|---|---|
| Floor | $2^{43.73}$ | $-33.6$ / $-54.7$ / $-73.7$ | $-31.2$ / $-52.3$ / $-70.5$ | $-25.7$ / $-46.8$ / $-63.2$ ($n_{PU}=57{,}344$) |
| B: one request | $2^{47.9}$, $n_{PU}=2^{24.2}$ | $-29.5$ / $-52.7$ / $-71.0$ | $-27.0$ / $-50.2$ / $-67.7$ | $-17.4$ / $-40.6$ / $-54.8$ |
| A: class audit of one link-window | $2^{58.8}$, $n_{PU}=2^{35.1}$ | $-18.6$ / $-47.2$ / $-63.7$ | $-16.1$ / $-44.8$ / $-60.4$ | $-1.0$ / $-29.7$ / $-40.3$ |

At the largest $k$ each entry is 6 to 9 bits worse; point A at $n_{sess}=n_{PU}$ is then vacuous ($+5.5$). The sizes at A and B are network-accounting's (relayed, not re-derived by proofs). On `main` the two-stage instance `compiledTwo` has $n_{sess}=n_{PU}$ (`ofDeployAt_nSess`, `SamplingGeneric/Deployed.lean:124`).

**Why no capped route reaches 100 as the theorem stands** (proofs' arithmetic, to be checked): a cap $K_0$ on the extraction's reruns costs a knowledge error $2k/K_0$ and gives $N\cdot q(K_0)/2^{256.5}$ with $q\approx 2T\cdot 210\,K_0\,n_{sess}$. The product of the two terms does not depend on $K_0$: $2k\cdot N\cdot 420\,T\,n_{sess}/2^{256.5}\approx 2^{-112.5}$ at the floor, so the best sum is about $2^{-55}$. So the square root and the factor $N$ have to go, or the premise's exponent has to grow.

## The script

```python
from math import log2, sqrt
def rows(N_lg, nsess, k, T_lg=80.0, terms=2):
    N=2.0**N_lg; T=2.0**T_lg; es=1+210*k
    cT=2*T*(1+nsess*es)
    ecr = terms*N*sqrt(cT)/2.0**128 + cT/2.0**256            # today: ExpectedCR
    a=2*k; b=terms*N*2*T*210*nsess/2.0**256.5                  # capped: 2k/K0 + terms*N*q/2^256.5
    K3=sqrt(a/b); r3=a/K3 + terms*N*2*T*(1+nsess*(1+210*K3))/2.0**256.5
    best=None                                                  # capped + AdaptiveBirthday: sqrt(q*cT/2^513)
    for e in [x/20 for x in range(20*20, 20*200)]:
        K=2.0**e; q=2*T*(1+nsess*(1+210*K)); v=a/K + terms*N*sqrt(q*cT)/2.0**256.5
        if best is None or v<best[0]: best=(v,e)
    return log2(ecr), log2(r3), log2(K3), log2(best[0]), best[1]
```

## What is already in flight (read it before you repeat it)

- `git fetch origin cursor/verity-proofs-84b5 && git show origin/cursor/verity-proofs-84b5:inbox/proofs-coordinator.md` is the lane's worker log (34,000 lines; grep the `## ` headings). Read the entries headed `PRF-517 TWOPROVE` (cost lemmas C1 to C8, the restatement of C-Flock's knowledge soundness at a second cap, route 3b), `REGX` (the committed trace averaged over the prover's tape, `cursor/proofs-regx-efe1`), `ACCOUNTD` and `BINDTERM`.
- The draft as fetched at 23:16Z: `git fetch origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/attachments/spot-security-definitions.md`.
- ASK-272 (the `AdaptiveBirthday` premise), PRF-516 (the X forms), PRF-517 (the cost lemmas), PRF-456 (network-accounting's sizes and targets: $2^{-61.0}$ a class audit, $2^{-53.4}$ an audit).

## The design rules a route may walk back

Daniel said past decisions are open. Name the one your route walks back; do not rule it out for that. The ones in play (AGENTS.md "Design rules", and the draft's Appendix B): one hash, SHA-512 with its whole output; live coins only; "no security parameter and no random oracle"; every premise a named `Prop` about an explicit finder; commitments are hm96 rows in frame-v3-sha512 trees; the committed trace as a plurality.

## What "100 bits" means until Daniel says otherwise

The whole error of the guarantee an application uses is at most $2^{-100}$ for a prover that makes at most $2^{80}$ SHA-512 calls in one audit, at the floor and at points A and B, with the session count the served system has. Report $T=2^{64}$ and the ratio $T/\varepsilon$ beside it. The worker on the target (TARGET) says whether this is the right reading, and what a period or a world-year of audits does to it.
