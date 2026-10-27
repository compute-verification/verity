import Proofs.Properties.Verity.SampledProofs
import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Bridge.Unopened

/-!
# Sampled proofs' target at the deployment, with no hypothesis on the strategy

`Vocab.SampledProofsSound` doesn't hold at every vocabulary: at `delta = 0`, `loss = 0` and `crBound = 1` the error is
`0`, and an extractor that returns a wrong transcript fails the claim at an honest play that passes. It holds at a
vocabulary whose sampled proofs' fields are the deployment's, C-Flock's live session at the committed files' parse
(`SampledDeploy`). `sampledProofsSound'` is the target there, at every `θ`, `F`, `c` and strategy, with no hypothesis
on the strategy.

`SampledDeploy` holds, for files that parse (`Vocab.parsed`), the session's setting at the parse (`A`, with its named
assumptions A3 and `RecordCustodyZKJ`), the session's strategy the play induces (`τOf`) and the play's break (`Brk`);
the facts relating them that no lemma gives yet, each its owner's:

* `accept_le`: outside the break, the verdict at most as likely as the session's acceptance (ONE-3t's
  `acc_of_accepts` and a decode of `ComponentAccepts`; ONE-4's coupling of the device's releases to the session's
  coins and `DrawnByDevice`);
* `break_le`: the break at the finder's loss (ONE-4's `findBreak` as the finder; LMS forgery to a SHA-512 collision);
* `unbound_le`: the unbound units at most the session's faulty ones (`LiveAt.faultyE`; `place`'s openings bridge,
  ONE-7);
* `cr`: the vocabulary's finder premise gives C-Flock's (C-Flock's finders as one `Claims.Finder`, ONE19);

the deployment's constants (`k = kDep`, `M = R_w = 2^192`, `ρ = 1/2`, `v ≤ 2^80`, at most `2^40` committed wires),
`t'` the strategy's declared cost; and what the product publishes: `ε ≥ 0` and `δ` at least the miss plus `2^-100` at
costs up to `2^80`, `⊤` above. From these `SampledDeploy.sampledIn` builds the coupling `SampledIn` at every strategy:
`units_eq` is ONE78's `unitsOfFile_eq`, `hε` is `LiveAt.sound_honest` at `K₀ = ⌊ε n⌋ + 1`, and `delta_le` is ONE19's
`deltaPub_le_deployed`. Files that don't parse have no units, and files that fail the commit phase's check no verdict.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **The committed files' parse**, as `Vocab.read` reads them. -/
noncomputable def Vocab.parsed (V : Vocab d α) (F : ByteArray) : Option Flock.Bridge.Parsed :=
  Flock.Bridge.parsedOf (V.read F).tags (V.read F).circuitFile (V.read F).tables (V.read F).publicFile

/-- **`deltaPub` at the deployment's constants** is the draw's miss plus at most `2^-100`, at a strategy whose cost
`t'` is at most `2^80`. -/
theorem deltaPub_le_of {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ}
    {desc : Claims.Msg} (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) (τ : FlockSoundness.Game.Strategy A.G)
    (hk : A.k = Flock.Bridge.kDep) (hM : A.M = 2 ^ 192) (hRw : A.Rw = 2 ^ 192) (hρ : A.ρ = 1 / 2)
    (hv : A.v ≤ 2 ^ 80) (hW : Flock.Bridge.ioSum A.P ≤ 2 ^ 40) (ht : A.t' τ ≤ 2 ^ 80) :
    Flock.Bridge.deltaPub n kd K₀ A.k (A.qF τ) (A.qS τ) (A.t' τ) A.M A.Rw (Flock.Bridge.ioSum A.P) A.ρ ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(100 : ℝ)) := by
  rw [hk, hM, hRw, hρ]
  have h0 := A.ht' τ
  have hv0 := A.hv
  have e82 : (2 : ℝ) ^ 82 = 4 * 2 ^ 80 := by norm_num
  exact Flock.Bridge.deltaPub_le_deployed n kd K₀ _ hW
    (show 0 ≤ 2 * (A.t' τ + A.v) by linarith) (show 2 * (A.t' τ + A.v) ≤ 2 ^ 82 by linarith)
    (show 0 ≤ A.t' τ + A.v by linarith) (show A.t' τ + A.v ≤ 2 ^ 82 by linarith) h0 ht

/-- Equal circuits, unit counts, partitions, openings and transcripts have as many unbound units. -/
theorem card_unbound_congr {C C' : Verity.Circuit} (hC : C = C') {n n' : ℕ} (hn : n = n')
    {P : Verity.Partition C n} {P' : Verity.Partition C' n'} (hP : HEq P P')
    {O : (g : Fin C.N) → BitVec (C.w g) → Prop} {O' : (g : Fin C'.N) → BitVec (C'.w g) → Prop} (hO : HEq O O')
    {X : C.Transcript} {X' : C'.Transcript} (hX : HEq X X') :
    (Accounting.unbound P O X).card = (Accounting.unbound P' O' X').card := by
  subst hC hn
  obtain rfl := eq_of_heq hP
  obtain rfl := eq_of_heq hO
  obtain rfl := eq_of_heq hX
  rfl

/-- The values the session's extractor opens at `τ`: its witness's openings in the sampling statement. -/
noncomputable abbrev _root_.Flock.Guarantees.LiveAt.valE {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit}
    {kd K₀ : ℕ} {desc : Claims.Msg} (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) (q : Claims.Sampling.Query)
    (τ : FlockSoundness.Game.Strategy A.G) : ℕ → Option Bool :=
  (Flock.Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)).val Claims.Sampling.sha512Msg A.Cd
    (Flock.Bridge.stmtOf A.P A.κ kd desc q A.roots)

/-- **`unbound_le` from the extractor and the openings.** Where the vocabulary's extractor is the transcript of the
values the session's extractor opens (`hX`, ONE-8's `extract`), and the vocabulary's openings are openings `O` of the
parse's circuit (`hO`) that hold wherever that witness opens a value other than a constant (`hopen`) and at the
constants (`hconst`) (`place`'s openings bridge, ONE-7's), the unbound units are at most the session's faulty ones
(`Flock.Bridge.card_unbound_le_faulty`). -/
theorem unbound_le_of {V : Vocab d α} (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    {r : Flock.Bridge.Parsed} (hr : V.parsed F = some r) {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt r.n r.D r.hU kd K₀ desc) (q : Claims.Sampling.Query)
    (τ : FlockSoundness.Game.Strategy A.G)
    (hX : HEq (V.extract θ F c σ) (Flock.Bridge.transcriptOf r.D r.hU r.n A.κ (A.valE q τ)))
    (O : (g : Fin (Flock.Bridge.circuitOf r.D r.hU r.n).N) → BitVec ((Flock.Bridge.circuitOf r.D r.hU r.n).w g) → Prop)
    (hO : HEq (V.Opens θ F c) O)
    (hopen : ∀ (g : Fin A.Cc.N) (b : Bool),
      (∀ b', (Flock.Bridge.progOf A.Cc A.κ).gates[Flock.Bridge.posOf A.Cc g]? ≠ some (.const b')) →
      A.valE q τ (Flock.Bridge.posOf A.Cc g) = some b → O g (BitVec.ofBool b))
    (hconst : ∀ (g : Fin A.Cc.N) (b : Bool),
      (Flock.Bridge.progOf A.Cc A.κ).gates[Flock.Bridge.posOf A.Cc g]? = some (.const b) →
      O g (Flock.Bridge.transcriptOf r.D r.hU r.n A.κ (A.valE q τ) g)) :
    (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card ≤ (A.faultyE q τ).card := by
  have hC : V.circuitOf F = Flock.Bridge.circuitOf r.D r.hU r.n := by
    show Flock.Bridge.Parsed.circuit (V.parsed F) = _
    rw [hr]
    rfl
  have hn : V.units F = r.n := by
    show Flock.Bridge.Parsed.units (V.parsed F) = _
    rw [hr]
    rfl
  have hP : HEq (V.partsOf F) (Flock.Bridge.partsOf r.D r.hU r.n) := by
    show HEq (Flock.Bridge.Parsed.parts (V.parsed F)) _
    rw [hr]
    rfl
  refine (card_unbound_congr hC hn hP hO hX).le.trans ?_
  exact Flock.Bridge.card_unbound_le_faulty A.P A.κ kd desc q A.roots Claims.Sampling.sha512Msg A.Cd
    (Flock.Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)) O hopen hconst

/-- **Sampled proofs at the deployment.** At files that parse to `r` (`Vocab.parsed`), every `θ`, `c` and strategy
`σ`: C-Flock's live session at the parse (`A`, `K₀ = ⌊ε n⌋ + 1`, its cost `t'` the strategy's), the session's
strategy the play induces (`τOf`), the play's break (`Brk`), and the facts relating them (`accept_le`, `break_le`,
`unbound_le`, `cr`), each its owner's to prove; the deployment's constants; and what the product publishes about
sampled proofs (`ε_nonneg`, `delta_ge`, `delta_top`). -/
structure SampledDeploy (V : Vocab d α) where
  kd : ByteArray → ℕ
  desc : ByteArray → Claims.Msg
  q : ByteArray → Claims.Sampling.Query
  A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
    Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ)
  τOf : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), FlockSoundness.Game.Strategy (A θ F c σ r hr).G
  Brk : ByteArray → ByteArray → ByteArray → Outcome → Prop
  accept_le : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), V.Registered θ F →
    Game.prob (fun o => V.Verdict .sampled o ∧ ¬ Brk θ F c o) (V.game θ F c) σ ≤
      FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkExec.LiveAcceptsZK (A θ F c σ r hr).I o)
        (A θ F c σ r hr).G (τOf θ F c σ r hr)
  break_le : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (Brk θ F c) (V.game θ F c) σ ≤ V.loss .sampled θ ((V.live θ F c).collides sha512Msg .sampled σ)
  unbound_le : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), V.Registered θ F →
    (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card ≤
      ((A θ F c σ r hr).faultyE (q θ) (τOf θ F c σ r hr)).card
  cr : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), V.Registered θ F →
    V.FinderCR .sampled θ F c σ → (A θ F c σ r hr).CR (τOf θ F c σ r hr)
  k_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).k = Flock.Bridge.kDep
  M_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).M = 2 ^ 192
  Rw_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).Rw = 2 ^ 192
  ρ_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).ρ = 1 / 2
  v_le : ∀ θ F c σ r hr, (A θ F c σ r hr).v ≤ 2 ^ 80
  ioSum_le : ∀ θ F c σ r hr, Flock.Bridge.ioSum (A θ F c σ r hr).P ≤ 2 ^ 40
  t'_eq : ∀ θ F c σ r hr τ, (A θ F c σ r hr).t' τ = V.cost θ F c σ
  ε_nonneg : ∀ θ, 0 ≤ (V.certOf θ).ε
  delta_ge : ∀ (θ F : ByteArray) (r : Flock.Bridge.Parsed), V.parsed F = some r → V.Registered θ F →
    ∀ t : ℕ, t ≤ 2 ^ 80 →
      ENNReal.ofReal (((r.n - (⌊(V.certOf θ).ε * r.n⌋₊ + 1)).choose (kd θ) : ℝ) / (r.n.choose (kd θ) : ℝ) +
        2 ^ (-(100 : ℝ))) ≤ V.delta θ .sampled t
  delta_top : ∀ (θ : ByteArray) (t : ℕ), 2 ^ 80 < t → V.delta θ .sampled t = ⊤

namespace SampledDeploy

variable {V : Vocab d α}

/-- **The coupling at every strategy**, at registered files that parse. -/
noncomputable def sampledIn (Dp : SampledDeploy V) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (r : Flock.Bridge.Parsed) (hr : V.parsed F = some r) (hreg : V.Registered θ F) : SampledIn V θ F c σ where
  n := r.n
  D := r.D
  hU := r.hU
  kd := Dp.kd θ
  K₀ := ⌊(V.certOf θ).ε * r.n⌋₊ + 1
  desc := Dp.desc θ
  A := Dp.A θ F c σ r hr
  τ := Dp.τOf θ F c σ r hr
  q := Dp.q θ
  units_eq := by
    show Flock.Bridge.unitsOfFile _ _ _ _ = r.n
    exact Flock.Bridge.unitsOfFile_eq hr
  hε := Flock.Guarantees.LiveAt.sound_honest (n := r.n) _ (mul_nonneg (Dp.ε_nonneg θ) (Nat.cast_nonneg _))
  Brk := Dp.Brk θ F c
  accept_le := Dp.accept_le θ F c σ r hr hreg
  break_le := Dp.break_le θ F c σ
  unbound_le := Dp.unbound_le θ F c σ r hr hreg
  cr := Dp.cr θ F c σ r hr hreg
  delta_le := by
    by_cases hc : V.cost θ F c σ ≤ 2 ^ 80
    · have ht : (Dp.A θ F c σ r hr).t' (Dp.τOf θ F c σ r hr) ≤ 2 ^ 80 := by
        rw [Dp.t'_eq]
        exact_mod_cast hc
      exact (ENNReal.ofReal_le_ofReal (deltaPub_le_of _ _ (Dp.k_eq θ F c σ r hr) (Dp.M_eq θ F c σ r hr)
        (Dp.Rw_eq θ F c σ r hr) (Dp.ρ_eq θ F c σ r hr) (Dp.v_le θ F c σ r hr) (Dp.ioSum_le θ F c σ r hr) ht)).trans
        (Dp.delta_ge θ F r hr hreg _ hc)
    · rw [Dp.delta_top θ _ (not_le.1 hc)]
      exact le_top

end SampledDeploy

/-- **Sampled proofs' target at the deployment** (`Vocab.SampledProofsSound`, ONE-8), with no hypothesis on the
strategy: at every vocabulary whose sampled proofs' fields are the deployment's (`SampledDeploy`), at every `θ`, `F`,
`c` and strategy, sampled proofs' verdict accepts while more than an `ε` share of the units are unbound in the
extractor's transcript with probability at most `Vocab.error .sampled`. -/
theorem sampledProofsSound' (V : Vocab d α) (Dp : SampledDeploy V) : V.SampledProofsSound := fun θ F c σ => by
  show Game.prob (fun o => V.Verdict .sampled o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
    V.error .sampled θ F c σ
  by_cases hreg : V.Registered θ F
  · cases hr : V.parsed F with
    | some r => exact (Dp.sampledIn θ F c σ r hr hreg).le_error
    | none =>
      have h0 : V.units F = 0 := by
        show Flock.Bridge.Parsed.units (V.parsed F) = 0
        rw [hr]
        rfl
      have hH : ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
          (V.certOf θ).ε * V.units F := by
        have hc : (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card ≤ V.units F :=
          (Finset.card_le_univ _).trans_eq (Fintype.card_fin _)
        have hc' : ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
            ((V.units F : ℕ) : ℝ) := by exact_mod_cast hc
        have h0' : ((V.units F : ℕ) : ℝ) = 0 := by rw [h0, Nat.cast_zero]
        rw [h0', mul_zero]
        exact hc'.trans h0'.le
      refine (Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hH) _ σ).trans ?_
      rw [Game.prob_false]
      exact zero_le
  · exact (prob_play_eq_zero (V.rounds θ) θ F c _ (fun _ _ h => hreg h.1.1) σ).trans_le zero_le

end Verity
