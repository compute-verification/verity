import Definitions.Pouw.FiatShamir

/-!
# Relative security's named premises under Fiat–Shamir draws (trusted)

The premises of the Fiat–Shamir statements in `Models.Pouw.Guarantees.FiatShamir`, over decision 102's seeds
(`Definitions.Pouw.FiatShamir`). Each stage's seed is SHA-512 of commitments the prover has made, evaluated inside the
drawn unit's own proof (decision 105). The live coins are each round's check.

* `SeedExtraction`: the random-oracle premise (the card 9cd57182: PoUW only, each use a named premise), gap (4)'s
  `Pouw.WindowWork.StraightLineExtraction` at the seed's domain: the seeds the accepted proofs derive are the oracle's
  answers at the committed inputs, except with probability `ε_id(T, |J|)`.
* `SteerBound`: a try reaches the premium `π` with probability at most `τ` (decision 101's per-try `μ^m`).
  `FiatShamirFloor` takes it.
* `SteerMoment`: a try's gain has moment at most `M` at rate `θ`, GRIND's per-query hypothesis
  (`Accounting.Guarantees.TS.GrindPremium`), which gives `SteerBound` at `τ = M·e^{−θπ}` (`SteerBoundOfMoment`).
* `ServiceSoundStagedFS`: `ServiceSoundStagedPerProver` with its stage draws from the oracle, the staged service sound
  up to its proof layer at every oracle, its binding and zero-knowledge breaks bounded by a constant `κ` plus weights
  times the success of the analysis's collision finders for that adversary (`FindersFS`), binding at the draw, not at
  the circuit's cuts (Fiat–Shamir fixes nothing before a round's seed). It is the premise from
  which the staged service's live statement at unsteered draws (`LiveFloorUnsteered`) is to be derived with K's scope
  and the machine's charge; no guarantee reads it yet.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity

/-- `random-oracle` (the random-oracle card, 9cd57182: PoUW only, each use a named premise). **The seeds the accepted
proofs derive are the oracle's answers at the committed inputs**: gap (4)'s `Pouw.WindowWork.StraightLineExtraction` at
the seed's domain. For every adversary within `T` oracle calls, the probability that its run is accepted (its proofs
pass at the seeds they derive in-circuit) while some stage's seed is not the oracle's answer at the input its commitment
names is at most `ε_id(T, |J|)`, over a uniform oracle, the live coins and the adversary's coins:
$$\Pr_{H,l,r}\big[\mathrm{accepts}(c, l, a) \wedge \mathrm{seed}(c, a) \ne H \circ \mathrm{pre}(c, \cdot)\big]
\le \varepsilon_{id}(T, |J|).$$

Its content is a straight-line extraction, with no rewinding. Each drawn unit's knowledge soundness extracts the roots
and position its seed is derived from, and the extractor reads them off the adversary's query log. The commitment is an
oracle program's output (`FSWithin`, the antecedent, as `ProverModel.RegQueries` is gap (4)'s). `H` stands for SHA-512
domain-separated for the commitments' trees as well as the seeds, gap (4)'s one oracle. So the witnessed inputs differ
from the committed ones only if the run's queries hold a collision or open a node that no query answered, and
`ε_id(T, |J|) ≈ (T² + T·|J|)/2^512` is the expected instance, with `T` every evaluation under the epoch's salt.

It is false for a service whose proofs don't bind the seed to the committed roots and position: a position the prover
picks inside the unit picks the seed (decision 106's condition). -/
def SeedExtraction {J N Live : Type} [Fintype J] [Fintype N] [Fintype Live] (Mc : Machine) (S : FSService J N Live)
    (T : ℕ) (εid : ℕ → ℕ → ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T →
    prob (fun ω : ((S.Q → N) × Live) × Ω =>
      S.accepts (Mc.eval M ω).1 ω.1.2 (Mc.eval M ω).2 = true ∧
        S.seed (Mc.eval M ω).1 (Mc.eval M ω).2 ≠ S.draws ω.1.1 (Mc.eval M ω).1) ≤ εid T (Fintype.card J)

/-- `pouw-fs-steer` (proposed). **A try reaches the premium with probability at most `τ`** (decision 101's per-try
`μ^m`; GRIND's `M·e^{−θπ}`). For every adversary within `T` oracle calls, every value `r` of its own coins, every
oracle `H` and every input `x` (a stage's seed input, committed or only tried), redrawing the oracle at `x` alone makes
the adversary's run realize at least `π` through `x` with probability at most `τ`. The rest of the oracle is fixed, so
the gain may depend on the run before the try and after it. That is the adaptivity across stages (GRIND's
`GrindTail`): a later stage's roots may read every earlier seed.

Its instance is GRIND's per-query moment (`SteerMoment`, `SteerBoundOfMoment`) at `τ = M·e^{−θπ}`, so at the premium
`π* = ln(T·M/δ)/θ` the `T` tries cost `T·τ = δ`. The gain is relative to the stage's savings, so RT4's C1 (a run cheap
at other stages that carries a zero-saving correlated error at this one) gains nothing here. It fails if a cheaper
run's gain at a seed reads too few fresh coordinates of the draw (one-sided noise, or a zero operand on the skip set:
GRIND's `q = 1`), or if the seed's expansion into noise shares coordinates across the positions it seeds. -/
def SteerBound {J N Live : Type} [Fintype J] [Fintype N] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (gain : GainMap Mc S) (π τ : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T → ∀ (r : Ω) (H : S.Q → N) (x : S.Q),
    prob (fun d : N => π ≤ gain Ω M r (Function.update H x d) x) ≤ τ

/-- `pouw-fs-steer-moment` (proposed). **A try's gain has moment at most `M` at rate `θ`** (GRIND's per-query
hypothesis of `Accounting.Guarantees.TS.GrindPremium`). For every adversary within `T` oracle calls, every value `r`
of its own coins, every oracle `H` and every input `x`, over the oracle's answer at `x` alone,
`𝔼_d[e^{θ·gain(H[x ↦ d], x)}] ≤ M`.

For h1tile's linears it is `Accounting.Guarantees.TS.TemplateMoment` (`M = 1` at `κₓ = 4`, `θ = 1/(256·r)` a product)
over a seed group's segments and tiles through `PrivateMoment`, given E3 (the skip is fixed by the query) and that the
seed group decomposes into independent segments given `U` and the data, which GRIND argues and doesn't state. Other
templates take `TileLuck` and `LuckMoment`. -/
def SteerMoment {J N Live : Type} [Fintype J] [Fintype N] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (gain : GainMap Mc S) (θ M : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (A : FSAdv Mc S Ω), FSWithin Mc S A T → ∀ (r : Ω) (H : S.Q → N) (x : S.Q),
    ∑ d : N, Real.exp (θ * gain Ω A r (Function.update H x d) x) ≤ M * Fintype.card N

section StagedFiatShamir

variable {R St n L np B n' : ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-fs-staged-service` (proposed). **The staged service under Fiat–Shamir draws is sound up to its proof
layer.** `ServiceSoundStaged` with decision 102's seeds: stage `(w, s)`'s draw is the oracle's answer at the input its
commitment names (`pre`), and each round's check is live (S4, round protocol with C1 per stage and C4 per draw; S5's
live coins now the checks only, on `trusted-nebius-host`). For every adversary within `T` oracle calls, every value
`r` of its own coins and every oracle `H`, its run's play reads each stage's error table off the stage's operand
commitment, which the stage's seed input hashes (`root(A_j)`, `root(B_j)`) (the play's form, `StagedPlay`; a re-rolled
root is a new input and a new draw, which `SteerBound` prices). Every run accepted at the oracle's draws is accepted by
the play's game at its realized coins (`stagedCoins`), or breaks binding or zero knowledge in some round (N1, X-RO
commitments; F1 and F2, C-Flock's binding and zero knowledge). Some round breaks with probability at most
`κ + βb · Σ_w Pr[fb M r H w finds a collision of sha] + βz · Σ_w Pr[fz M r H w …]` over the live coins: `κ` the rounds'
constants, which read no prover, and the analysis's binding and zero-knowledge finders for that adversary at their
weights, since no constant bounds a break for an adversary holding a collision.

Binding is at the draw, not at the circuit's cuts. Fiat–Shamir fixes nothing before a round's seed, since the prover
evaluates the hash before it commits, so the breaks are measured at a fixed oracle, every seed answered and the
commitment fixed, over the live checks alone. At the cuts, binding would fail with no collision: a cheater that writes
two candidate operands, queries both seeds and keeps the first favorable one opens a value that departs from the cut's
plurality value with probability `1/4` (RT-COMPOSE r5's toy, one round, two queries), and a fork at the cut reselects a
query instead of finding a collision. That grinding is the premium's (`SteerBound`), not a break's.

A free state term in place of the finders (`κ + g(M, r, H)`) would carry no number: `g = 1` satisfies the premise, and
the floor's bound is then at least 1. Through the finders, the floor reads `ecr/sha-512` at each adversary's own
finders (`breakBoundFS`, `Claims.ecrBound`, the record the live floors per prover read), at budgets computed from the
adversary's time (`cb · time M + db` SHA-512 calls for binding, `cz · time M + dz` for zero knowledge), never at the
finders' declared costs. So a machine that hard-codes a collision gets 1, a finder that declares a large cost to meet
`ecr/sha-512` exceeds the budget, and every other adversary keeps the budgets over `2^256` at its weights. The
instances are `ServiceSoundStagedPerProver`'s, round by round:
- binding: the fork finder, two runs of `M` and the verifier's checks (`cb` about twice the calls a unit of time buys),
  its square root folded in by AM–GM, `δ' + |L|² x/(4δ')`;
- zero knowledge: C-Flock's knowledge bound, whose finder extracts the openings first (its link finders, about
  `2(1 + k)` runs, `linkBoundZC`). -/
def ServiceSoundStagedFS (Mc : Machine) (S : StagedFS R St n L np B n') (T : ℕ) (run : RunMapFS Mc S Ar)
    (κ : ℝ) (sha : Claims.Msg → Claims.Msg) (fb fz : FindersFS Mc S R) (βb βz : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T →
    ∀ (r : Ω) (H : S.Q → Finset (DBits n L np B n')),
      prob (fun l : Fin R → Fin St × DCells n n' B => ∃ w,
        (run Ω M r H).Bind w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l) ∨
          (run Ω M r H).ZK w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l)) ≤
        κ + βb * ∑ w, (fb Ω M r H w).Pr sha + βz * ∑ w, (fz Ω M r H w).Pr sha ∧
      ∀ l, S.verify (Mc.eval M ((H, l), r)).1 (S.draws H (Mc.eval M ((H, l), r)).1) l (Mc.eval M ((H, l), r)).2 = true →
        (run Ω M r H).play.game.accepts (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l) ∨
          ∃ w, (run Ω M r H).Bind w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l) ∨
            (run Ω M r H).ZK w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l)

end StagedFiatShamir

end Pouw.Assumptions
