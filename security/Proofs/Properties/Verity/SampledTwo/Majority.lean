module

public import Proofs.Properties.Verity.SampledTwo.Projection
public import Proofs.Properties.Verity.SampledTwo.Claim
public import Proofs.Properties.Verity.Majority

/-!
# The two-stage leg's majority: the boundary values fixed at registration

`MajorityBound` asks that, where the claim fails on the vocabulary's transcript, an accepted run of the frame's prover
whose extracted witness is in the two-stage relation be at most `bind` likely. The transcript is the majority of the
boundary values the extractor's witnesses open (`majority`), the shared plurality (`Verity.Majority`) at the frame's
plays.

* **The plays** (`majorityPlays`): one sample, the frame's tape, the run's coins and the extractor's seed; a play binds
  the units its extracted witness's transcript binds, sampled or not.
* **What a witness in the relation binds** (`HoldsOnWitness`): the claim holds on the transcript of every witness in the
  relation. `holdsOnWitness_of` proves it from `holds_of_rTwo`, where the units are the replay blocks
  (`UnitsAreReplay`) and the design caps the claim's share (`CapsAt`): a witness in `RTwo` binds every replay block but
  at most `ε · n`, sampled or not.
* **Two readings of a bound wire are a collision** (`ReadsBind`): where two witnesses in the relation both bind a unit
  holding the wire `g` and read `g` differently, `collOf` returns a SHA-512 collision; the binding finder
  (`bindFinder`) runs the frame twice, independently, and returns it.
* **The bound** (`majorityBound`): where the claim fails on the majority, more than `ε · n` units are unbound in it and
  at most `ε · n` in an accepted witness's transcript, so the witness binds a unit the majority doesn't and differs from
  it at one of the unit's boundary bits (`Majority.unbound_drawn`); such plays weigh at most
  `√(min(W, 2n) · K)`, `K` the binding finder's collision probability (`Majority.Plays.ex_off_le`), and at most
  `√(min(W, 2n) · E[cost] / 2^256)` under `ecr/sha-512` at it (`majorityBound_ecr`).

The deployed device's history holds one play's signatures, so an earlier period's genuine challenger signature in the
evidence makes the strategy's error `⊤` at the deployed theorems; nothing here reads the device, and that hole stays
open on this leg until the key pins close it.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **A play of the frame with the extractor's seed**: the tape, the run's coins and the seed. -/
abbrev MajSample (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Type :=
  TapeOf V θ F c σ × Coins A x × A.L.E.C x

/-- **The frame's plays** (`Majority.Plays` at one sample): a play is the frame's tape, the run's coins and the
extractor's seed; it accepts where the frame's run does; its witness is the extractor's; the relation is the session's
at `x`; a witness is read as the vocabulary's transcript by `readX`; and a play binds the units its witness's
transcript binds (correct, and opened by the commitment at every wire). -/
noncomputable def majorityPlays (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) :
    Majority.Plays (V.circuitOf θ F) (V.units θ F) (V.partsOf θ F) where
  Ξ := Unit
  Smp _ := MajSample V A x θ F c σ
  Wit := Claims.Sampling.Wit
  acc _ s := (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2.1).out = .acc
  drawn _ s u := ∃ wt, (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1 = some wt ∧
    u ∉ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX wt)
  wit _ s := (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1
  R _ := A.R x
  read _ := readX

/-- **The majority**: each bit of each wire is the value the extracted witnesses in the relation read there most often,
over the accepted plays whose witness binds a unit holding the wire (`false` on a tie). -/
noncomputable def majority (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) : Verity.Circuit.Trace (V.circuitOf θ F) :=
  (majorityPlays V A x θ F c σ Fr readX).extract

/-- **The binding finder**: it plays the frame twice, with independent tapes, coins and seeds, runs the extractor on
both, and outputs `collOf`'s pair from the two witnesses. Its cost is the two runs' SHA-512 calls, the two
extractions' and `collOf`'s `dColl`. -/
noncomputable def bindFinder (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (collOf : Claims.Sampling.Wit → Claims.Sampling.Wit → Option (Msg × Msg)) (dColl : ℕ) : Claims.Finder where
  S := MajSample V A x θ F c σ × MajSample V A x θ F c σ
  out s :=
    match (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1.1 s.1.2.1 s.1.2.2).1,
      (A.L.E.eval FlockDeploy.sha512 Fr.P x s.2.1 s.2.2.1 s.2.2.2).1 with
    | some w₁, some w₂ => collOf w₁ w₂
    | _, _ => none
  cost s :=
    (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1.1 s.1.2.1).pLog.length +
      (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1.1 s.1.2.1).vLog.length +
      (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.2.1 s.2.2.1).pLog.length +
      (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.2.1 s.2.2.1).vLog.length +
      (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1.1 s.1.2.1 s.1.2.2).2.1 +
      (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1.1 s.1.2.1 s.1.2.2).2.2 +
      (A.L.E.eval FlockDeploy.sha512 Fr.P x s.2.1 s.2.2.1 s.2.2.2).2.1 +
      (A.L.E.eval FlockDeploy.sha512 Fr.P x s.2.1 s.2.2.1 s.2.2.2).2.2 + dColl

/-- **The claim holds on every witness the relation takes** at `x`, read as the vocabulary's transcript by `readX`:
at most an `ε` share of the units is unbound in it. At the two-stage parameters it is `holds_of_rTwo`'s conclusion
(`holdsOnWitness_of`), from `UnitsAreReplay` and `CapsAt`, which the instance owes. -/
def HoldsOnWitness (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) : Prop :=
  ∀ wt, A.R x wt → Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (readX wt) (V.certOf θ).ε

/-- **Two readings of a bound wire are a collision** at `x` (the instance owes it, from the commitment's binding at the
boundary registration): where two witnesses in the relation each bind a unit holding the wire `g` (correct there and
opened by the commitment `c` at each of its wires) and read `g` differently, `collOf` outputs a SHA-512 collision. -/
def ReadsBind (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (collOf : Claims.Sampling.Wit → Claims.Sampling.Wit → Option (Msg × Msg)) : Prop :=
  ∀ w₁ w₂ (g : Fin (V.circuitOf θ F).N), A.R x w₁ → A.R x w₂ →
    (∃ u, u ∉ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w₁) ∧ g ∈ (V.partsOf θ F).io u) →
    (∃ u, u ∉ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w₂) ∧ g ∈ (V.partsOf θ F).io u) →
    readX w₁ g ≠ readX w₂ g →
    ∃ m m', collOf w₁ w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m'

/-- **`HoldsOnWitness` from the claim at replay blocks**: where `x` parses with the session's partitions, the units are
the replay blocks (`UnitsAreReplay`) and the design caps the claim's share at the session's `δ` (`CapsAt`), every
witness in the relation makes the claim hold on its transcript (`holds_of_rTwo`). -/
theorem holdsOnWitness_of (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (hs : A.L.parse x = some s) (hP : A.L.partition s = some Pt)
    (hd : A.L.two s = some dd) (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (hU : UnitsAreReplay V θ F c FlockDeploy.sha512 A.L.commit s Pt dd readX)
    (hcap : CapsAt dd Pt ((V.certOf θ).ε * V.units θ F) A.δ) : HoldsOnWitness V A x θ F c readX :=
  fun wt hR => holds_of_rTwo V θ F c FlockDeploy.sha512 A.L.commit A.L.parse A.L.partition A.L.two A.δ x s Pt dd hs
    hP hd readX hU hcap wt hR

/-- The average over a nonempty type of a function bounded by `K` is at most `K`. -/
theorem ex_le_of_forall_le {β : Type} [Fintype β] [Nonempty β] {f : β → ℝ} {K : ℝ} (h : ∀ a, f a ≤ K) :
    Ex f ≤ K := by
  have hc : (Fintype.card β : ℝ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  calc Ex f ≤ Ex (fun _ : β => K) := Claims.Ex_mono h
    _ = K := by
      unfold Ex
      rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      field_simp

/-- **Two plays' collision**: the binding finder at the two plays `s₁`, `s₂` outputs a SHA-512 collision. -/
def collTwo (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (collOf : Claims.Sampling.Wit → Claims.Sampling.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (s₁ s₂ : MajSample V A x θ F c σ) : Prop :=
  ∃ m m', (bindFinder V A x θ F c σ Fr collOf dColl).out (s₁, s₂) = some (m, m') ∧
    Claims.Collision FlockDeploy.sha512 m m'

/-- **`ReadsBind` is the frame's plays' binding at the binding finder's collision.** -/
theorem majorityPlays_binds (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (collOf : Claims.Sampling.Wit → Claims.Sampling.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ReadsBind V A x θ F c readX collOf) :
    (majorityPlays V A x θ F c σ Fr readX).Binds
      (fun _ s₁ _ s₂ => collTwo V A x θ F c σ Fr collOf dColl s₁ s₂) := by
  intro ξ₁ s₁ ξ₂ s₂ w₁ w₂ g he₁ he₂ hR₁ hR₂ hd₁ hd₂ hne
  have he₁' : (A.L.E.eval FlockDeploy.sha512 Fr.P x s₁.1 s₁.2.1 s₁.2.2).1 = some w₁ := he₁
  have he₂' : (A.L.E.eval FlockDeploy.sha512 Fr.P x s₂.1 s₂.2.1 s₂.2.2).1 = some w₂ := he₂
  obtain ⟨u₁, ⟨wt₁, hwt₁, hu₁⟩, hg₁⟩ := hd₁
  obtain ⟨u₂, ⟨wt₂, hwt₂, hu₂⟩, hg₂⟩ := hd₂
  have e₁ : wt₁ = w₁ := Option.some.inj (hwt₁.symm.trans he₁')
  have e₂ : wt₂ = w₂ := Option.some.inj (hwt₂.symm.trans he₂')
  subst e₁ e₂
  obtain ⟨m, m', hm, hcol⟩ := hcoll wt₁ wt₂ g hR₁ hR₂ ⟨u₁, hu₁, hg₁⟩ ⟨u₂, hu₂, hg₂⟩ hne
  refine ⟨m, m', ?_, hcol⟩
  simp only [bindFinder, he₁', he₂']
  exact hm

/-- **An accepted play whose witness is in the relation is off the majority**: where the vocabulary's transcript is the
majority and the claim fails on it, while the claim holds on every witness the relation takes, the play's witness binds
a unit the majority leaves unbound and differs from the majority at a bit of one of its wires. -/
theorem off_of_accepts (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (hX : V.extract θ F c σ = majority V A x θ F c σ Fr readX) (hholds : HoldsOnWitness V A x θ F c readX)
    (hnot : ¬ Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε)
    (s : MajSample V A x θ F c σ)
    (hacc : (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2.1).out = .acc)
    (hR : Claims.Rel.ext A.R x (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1) :
    ∃ u, (majorityPlays V A x θ F c σ Fr readX).offEvent (majority V A x θ F c σ Fr readX) u () s := by
  classical
  cases hw : (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1 with
  | none => rw [hw] at hR; exact (show False from hR).elim
  | some wt =>
    rw [hw] at hR
    have hR' : A.R x wt := hR
    have hle := hholds wt hR'
    have hgt : ¬ ((Accounting.unbound (V.partsOf θ F) (V.Opens θ F c)
        (majority V A x θ F c σ Fr readX)).card : ℝ) ≤ (V.certOf θ).ε * V.units θ F := by
      rw [← hX]; exact hnot
    obtain ⟨u, huX, huW⟩ : ∃ u ∈ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (majority V A x θ F c σ Fr readX),
        u ∉ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX wt) := by
      by_contra hall
      push_neg at hall
      have hcard := Finset.card_le_card (show Accounting.unbound (V.partsOf θ F) (V.Opens θ F c)
          (majority V A x θ F c σ Fr readX) ⊆ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX wt) from
        fun u hu => hall u hu)
      have hle' : ((Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX wt)).card : ℝ) ≤
          (V.certOf θ).ε * V.units θ F := hle
      exact hgt (le_trans (by exact_mod_cast hcard) hle')
    have h' := huW
    simp only [Accounting.unbound, Finset.mem_filter, Finset.mem_univ, true_and, not_or, not_exists, not_and,
      not_not] at h'
    obtain ⟨g, hg, i, hi, hbit⟩ := Majority.unbound_drawn (V.partsOf θ F) (V.Opens θ F c) h'.1 h'.2 huX
    exact ⟨u, hacc, ⟨wt, hw, huW⟩, wt, hw, hR', g, hg, i, hi, hbit⟩

/-- **The witness fixed at registration**: where the vocabulary's transcript is the majority (`hX`), the claim holds on
every witness the relation takes (`hholds`), two readings of a bound wire are a collision (`hcoll`) and the binding
finder outputs one with probability at most `K` (`hbind`), an accepted run of the frame's prover whose extracted
witness is in the session's relation, where the claim fails on the vocabulary's transcript, is at most
`√(min(W, 2n) · K)` likely, `W` the circuit's wire bits and `n` its units. -/
theorem majorityBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (collOf : Claims.Sampling.Wit → Claims.Sampling.Wit → Option (Msg × Msg)) (dColl : ℕ) (K : ℝ)
    (hX : V.extract θ F c σ = majority V A x θ F c σ Fr readX)
    (hholds : HoldsOnWitness V A x θ F c readX) (hcoll : ReadsBind V A x θ F c readX collOf)
    (hbind : (bindFinder V A x θ F c σ Fr collOf dColl).Pr FlockDeploy.sha512 ≤ K) :
    MajorityBound V A x θ F c σ Fr (V.extract θ F c σ)
      (ENNReal.ofReal (Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) * K))) := by
  intro _ hnot
  refine ENNReal.ofReal_le_ofReal ?_
  have hK : Ex (fun p : (majorityPlays V A x θ F c σ Fr readX).Ξ × (majorityPlays V A x θ F c σ Fr readX).Ξ =>
      Pr (fun s : (majorityPlays V A x θ F c σ Fr readX).Smp p.1 × (majorityPlays V A x θ F c σ Fr readX).Smp p.2 =>
        collTwo V A x θ F c σ Fr collOf dColl s.1 s.2)) ≤ K := by
    show Ex (fun _ : Unit × Unit => Pr (fun s : MajSample V A x θ F c σ × MajSample V A x θ F c σ =>
      collTwo V A x θ F c σ Fr collOf dColl s.1 s.2)) ≤ K
    exact ex_le_of_forall_le fun _ => hbind
  have hoff := (majorityPlays V A x θ F c σ Fr readX).ex_off_le
    (fun _ s₁ _ s₂ => collTwo V A x θ F c σ Fr collOf dColl s₁ s₂)
    (majorityPlays_binds V A x θ F c σ Fr readX collOf dColl hcoll) hK
  have hunit : Ex (fun ξ : (majorityPlays V A x θ F c σ Fr readX).Ξ =>
      Pr (fun s : (majorityPlays V A x θ F c σ Fr readX).Smp ξ =>
        ∃ u, (majorityPlays V A x θ F c σ Fr readX).offEvent (majorityPlays V A x θ F c σ Fr readX).extract u ξ s)) =
      Pr (fun s : MajSample V A x θ F c σ =>
        ∃ u, (majorityPlays V A x θ F c σ Fr readX).offEvent (majority V A x θ F c σ Fr readX) u () s) :=
    Majority.ex_unit _
  refine le_trans (Claims.Pr_mono fun s hs => ?_) (le_trans (le_of_eq hunit.symm) hoff)
  exact off_of_accepts V A x θ F c σ Fr readX hX hholds hnot s hs.1 hs.2

/-- **The witness fixed at registration under `ecr/sha-512`**: `majorityBound` where the binding finder meets
`ExpectedCR`, so that `K` is its expected SHA-512 calls over `2^256`: the binding term
`√(min(W, 2n) · E[cost] / 2^256)` of the recursive leg (`Verity.Recursive.Dep.delta`). -/
theorem majorityBound_ecr (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (collOf : Claims.Sampling.Wit → Claims.Sampling.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hX : V.extract θ F c σ = majority V A x θ F c σ Fr readX)
    (hholds : HoldsOnWitness V A x θ F c readX) (hcoll : ReadsBind V A x θ F c readX collOf)
    (hbind : Claims.ExpectedCR FlockDeploy.sha512 (bindFinder V A x θ F c σ Fr collOf dColl)) :
    MajorityBound V A x θ F c σ Fr (V.extract θ F c σ)
      (ENNReal.ofReal (Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
        (Ex (fun s => ((bindFinder V A x θ F c σ Fr collOf dColl).cost s : ℝ)) / 2 ^ (256 : ℝ))))) :=
  majorityBound V A x θ F c σ Fr readX collOf dColl _ hX hholds hcoll hbind

end Verity.SampledTwo
