module

public import Definitions.Claims.RecursionGeneric
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.SampledGeneric.Statements

/-!
# A claims-core proof system through the sampled compiler (B1)

The sampled audit (`Proofs.Flock.SampledGeneric`) runs a proof system's session on each sample, a
`FlockSoundness.Game` (`ProofSystem`). The recursion states its verifiers in the claims core (`Claims.Verifier`,
`Claims.run`). This file is the one bridge between the two:

* `sessionOf H V x`: the claims verifier `V`'s session at `x` as a game. The prover sends a claims prover (tape
  `Unit`), the verifier samples its coins, and the game returns `run`'s verdict. A strategy of it is a claims prover
  (`proverOf`), so a bound over every claims prover covers every strategy. The coins are live as `run` makes them: the
  prover's message at a round reads only the verifier's replies to the rounds before, and its SHA-512 calls are in
  the run's log.
* `proofSystemOf H V stmt`: the proof system whose session on the sample `S` against the registration `R` is
  `sessionOf` at the statement `stmt S R`.
* `jointOf`: the claims extractor `E` as the sampled compiler's textbook extractor (`JointExtractor`). After the
  judged session it samples the judged run's coins and `E`'s, and runs `E` on the prover's runs. Each sampled proof
  block's witness is `E`'s witness of the whole statement, and `opens` gives the opening it carries at each position.
  Its cost is the trial's SHA-512 calls, counted from its runs (`trialCalls`: the judged run's and `E`'s, its own and
  its reruns'), so the collision premise at the trials' finders (`Trials.CR`) is read at what the trials compute.
* `Flock.Guarantees.GenericSampledKSClaims` (**B1**): a claims verifier that is knowledge sound at every sampled
  statement, `Pr[run = acc ∧ ¬R(x, E(x))] ≤ err(x, P)` on every prover `P`, and whose witnesses open the
  registration and make their proof blocks correct (`JointExtractor.Sound`), gives the sampled audit's bound
  `Pr[accept] ≤ P_a(F) + E_S max_{u ∈ S} err + ε_bind`, by `GenericSampledKSJoint`.
* `Flock.Guarantees.RecursiveSampledKS`: the recursive audit, B1 at a compiled verifier with `err = Comp.err` where
  L1's hypotheses hold at every sampled statement.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims FlockSoundness FlockSoundness.Game

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **A claims verifier's session as a game**: the prover sends its claims prover, the verifier samples the run's coins
`r`, and the game returns whether `run` accepts. -/
def sessionOf (H : Msg → Msg) (V : Verifier Coin) (x : Msg) : FlockSoundness.Game Bool :=
  .send (Prover Unit) fun P => .coin (Fin (V.k x) → Coin) fun r => .ret (decide ((run H P V x () r).out = .acc))

/-- **The claims prover a strategy plays**: its first move. -/
def proverOf {H : Msg → Msg} {V : Verifier Coin} {x : Msg} (τ : Strategy (sessionOf H V x)) : Prover Unit := τ.1

variable {n : ℕ} {Reg Dh : Type} {Hc : List UInt8 → Dh}

/-- **The claims verifier as a proof system for proof blocks**: on the sample `S` against the registration `R`, its
session at the statement `stmt S R`. -/
abbrev proofSystemOf (H : Msg → Msg) (V : Verifier Coin) (stmt : Finset (Fin n) → Reg → Msg) :
    FlockSoundness.SampledGeneric.ProofSystem n Reg :=
  ⟨fun S R => sessionOf H V (stmt S R)⟩

/-- **A trial's SHA-512 calls**: the judged run's, both parties', and `E`'s on it, its own and its reruns'
(`Extractor.eval`). -/
def trialCalls (H : Msg → Msg) (P : Prover Unit) (V : Verifier Coin) (E : Claims.Extractor Coin W V) (x : Msg)
    (r₀ : Fin (V.k x) → Coin) (c : E.C x) : ℕ :=
  (run H P V x () r₀).pLog.length + (run H P V x () r₀).vLog.length + (E.eval H P x () r₀ c).2.1

/-- **The claims extractor as the sampled compiler's textbook extractor.** Its game, after the judged session, takes
the prover's claims prover, samples the judged run's coins `r₀` and `E`'s coins, and outputs the judged run's verdict
and, at each sampled proof block, `E`'s result at the statement beside the trial's SHA-512 calls (`trialCalls`). A
witness is one of `u`'s statement when `E` gave one and `Rl` holds of it at `stmt S R`; `reads u` are the positions `u`
reads, and `opens S R u w c` the opening of `c` that `w` carries. Its cost is the calls its output carries, counted
from the trial's runs: a trial's sample reads a position, so it has a sampled proof block. -/
noncomputable def jointOf (H : Msg → Msg) (V : Verifier Coin) (Rl : Rel W) (E : Claims.Extractor Coin W V)
    (stmt : Finset (Fin n) → Reg → Msg) (B : FlockSoundness.SampledGeneric.Binding Reg Hc)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) :
    FlockSoundness.SampledGeneric.JointExtractor (proofSystemOf H V stmt) B where
  Wit := Option W × ℕ
  Rel S R _ w := Rl.ext (stmt S R) w.1
  reads := reads
  opens S R u w c := (w.1.map fun w' => opens S R u w' c).getD default
  G S R := .send (Prover Unit) fun P => .coin (Fin (V.k (stmt S R)) → Coin) fun r₀ =>
    .coin (E.C (stmt S R)) fun c => .ret (decide ((run H P V (stmt S R) () r₀).out = .acc),
      fun u => if u ∈ S then some ((E.eval H P (stmt S R) () r₀ c).1, trialCalls H P V E (stmt S R) r₀ c) else none)
  strat _ _ τ := ⟨proverOf τ, fun _ _ => PUnit.unit⟩
  cost _ _ o := ((Finset.univ.sup fun u => ((o.2 u).map Prod.snd).getD 0 : ℕ) : ℝ)

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric
open scoped ENNReal

/-- **B1: a claims-core proof system through the sampled compiler.** For every claims verifier `V` with relation `Rl`
and extractor `E`, and every `H` at which `E`'s error at each sampled statement is at most `err` on every prover
(`PrBad ≤ err`), sampling design `L`, binding commitment `B` to the trace, proof blocks' correctness `Correct`,
statements `stmt` and witnesses' openings (`reads`, `opens`) that are sound for the proof blocks
(`JointExtractor.Sound`): at every audit prover `σ` whose link finders meet ecr/sha-512 at the trials' counted calls
(`Trials.CR`),
$$\Pr[\text{accept}] \le P_a(F) + \mathbb E_S \max_{u \in S} \mathrm{err}(\mathrm{stmt}(S, R), P_S)
+ \varepsilon_{\rm bind},$$
with `R` the registration, `P_S` the claims prover `σ` plays on the sample `S`, `F` the incorrect set of the plurality
of the trials and `ε_bind` their binding term. `E` is any rerun program and may depend on `H`; the trials' cost counts
its `H` calls and its reruns' (`trialCalls`), not its computation between them, so `F` is the committed trace's
incorrect set only at a counted `E` (Flock's `E^rep` at the deployed decoder). -/
def GenericSampledKSClaims : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Verifier Coin) (Rl : Rel W)
    (E : Claims.Extractor Coin W V) (H : Claims.Msg → Claims.Msg) (err : Claims.Msg → Prover Unit → ℝ) (n : ℕ)
    (Reg Dh : Type)
    (Hc : List UInt8 → Dh) (L : Law n) (B : Binding Reg Hc) (Correct : Reg → B.Trace → Fin n → Prop)
    (stmt : Finset (Fin n) → Reg → Claims.Msg)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) (M : ℕ),
    (∀ S R (P : Prover Unit), PrBad H V Rl E (stmt S R) P ≤ err (stmt S R) P) →
    (jointOf H V Rl E stmt B reads opens).Sound Correct → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg (proofSystemOf H V stmt).session),
      ((jointOf H V Rl E stmt B reads opens).toTrials L).CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg (proofSystemOf H V stmt).session) σ ≤
        L.escape (incorrect B Correct (reg σ)
            (((jointOf H V Rl E stmt B reads opens).toTrials L).plur (reg σ) (cont σ))) +
          avg (fun ω => ⨆ u ∈ L.draw ω,
            ENNReal.ofReal (err (stmt (L.draw ω) (reg σ)) (proverOf (x := stmt (L.draw ω) (reg σ)) (cont σ ω)))) +
          ((jointOf H V Rl E stmt B reads opens).toTrials L).bindErr (reg σ) (cont σ) M

/-- **The recursive audit through the sampled compiler.** B1 at a compiled verifier `K` over the inner protocol `I`
and the server `Sv`, its extractor `E*` (`Comp.E`) and relation (`Comp.R`), where the server binds (`Server.Binds`) and
L1's hypotheses hold at every sampled statement (`Comp.OuterKSAt`, nonnegative `κ_O` and `γ_O`, `Comp.Contract`):
$$\Pr[\text{accept}] \le P_a(F) + \mathbb E_S \max_{u \in S} \mathrm{Comp.err}(\mathrm{stmt}(S, R), P_S) +
\varepsilon_{\rm bind},$$
at the inner loop bounds `mI` and the link's `Mo`. It follows from L1 (`Claims.RecGeneric.CompositionKS`) and B1. -/
def RecursiveSampledKS : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (I : Proto Coin W)
    (Sv : Server) (K : Claims.RecGeneric.Comp Coin) (mI : Fin I.nT → ℕ) (Mo : ℕ) (n : ℕ) (Reg Dh : Type)
    (Hc : List UInt8 → Dh) (L : Law n) (B : Binding Reg Hc) (Correct : Reg → B.Trace → Fin n → Prop)
    (stmt : Finset (Fin n) → Reg → Claims.Msg)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) (M : ℕ),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → Sv.Binds →
    (∀ S R, K.OuterKSAt H (stmt S R) ∧ 0 ≤ K.κO (stmt S R) ∧ 0 ≤ K.γO (stmt S R) ∧
      K.Contract H (Sv.IVStar I.V I.ans) (fun v => I.opens (Sv.openView v)) (stmt S R)) →
    (jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).Sound Correct → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg (proofSystemOf H K.V stmt).session),
      ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg (proofSystemOf H K.V stmt).session) σ ≤
        L.escape (incorrect B Correct (reg σ)
            (((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).plur (reg σ) (cont σ))) +
          avg (fun ω => ⨆ u ∈ L.draw ω,
            ENNReal.ofReal (K.err I Sv H (proverOf (x := stmt (L.draw ω) (reg σ)) (cont σ ω))
              (stmt (L.draw ω) (reg σ)) mI Mo)) +
          ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).bindErr (reg σ) (cont σ) M

end Flock.Guarantees
