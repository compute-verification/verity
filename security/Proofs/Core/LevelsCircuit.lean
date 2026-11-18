module

public import Proofs.Core.Circuit
public import Proofs.Core.LevelsMiss

/-!
# Committing late does not help, at a circuit's two levels

Two levels drawn over a circuit's nested partitions: the inner blocks are the units of `Pf`, the outer blocks those of
`Pc`, and `Pf` refines `Pc` (`r`). The levels' blocks are sets of gates of some type `G`, and `blk`, `oblk` place the
inner and outer units among them, an inner unit inside an outer one exactly when the outer is its parent. The outer
trace is `τ₁` at the opened gates `O`. An inner trace extends it when it agrees with `τ₁` on `O`, and its incorrect
inner blocks are its incorrect units with those having an outer committed wire outside `O` (`Partition.unopened`).

* `Levels.late_le_missProb_opened`: after the outer sample is drawn, the prover picks at each outer block, by any rule,
  an inner trace extending `τ₁`, its interior. Every inner sample then misses that trace's incorrect inner blocks with
  probability at most `missProb L.design` of one trace's, a trace fixed in advance that extends `τ₁`. It is
  `Levels.late_le_missProb` at the incorrect sets of the traces extending `τ₁`, which glue
  (`Partition.Refines.glue`).
* `Levels.late_le_missProb_trace`: the same with every outer committed wire opened.
* `Levels.prCoin_accept_le_late`: the sampling step of soundness at a miss `δ` with interiors committed late. A run
  accepted only where the inner samples miss the late interiors' incorrect inner blocks, or a break occurs, is
  accepted while the largest miss over the interiors is at most `δ` with probability at most `δ` plus the break's.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

/-- The sampling step of soundness at a miss `δ`, at any event `E` each `p` bounds by `m p`. -/
theorem prCoin_accept_le_of {P Ω : Type} [Fintype P] [Nonempty P] [Fintype Ω] (E acc brk : P → Ω → Prop)
    (m : P → ℝ≥0∞) (hE : ∀ p, prCoin (E p) ≤ m p) (h : ∀ p ω, acc p ω → E p ω ∨ brk p ω) (δ : ℝ≥0∞) :
    prCoin (fun x : P × Ω => acc x.1 x.2 ∧ m x.1 ≤ δ) ≤ δ + prCoin (fun x : P × Ω => brk x.1 x.2) := by
  classical
  have hmiss : prCoin (fun x : P × Ω => E x.1 x.2 ∧ m x.1 ≤ δ) ≤ δ := by
    have hp : ∀ p : P, prCoin (fun ω : Ω => E p ω ∧ m p ≤ δ) ≤ δ := by
      intro p
      by_cases hδ : m p ≤ δ
      · exact (prCoin_mono fun ω hω => hω.1).trans ((hE p).trans hδ)
      · exact (prCoin_mono (q := fun _ => False) fun ω hω => hδ hω.2).trans (prCoin_false.trans_le zero_le)
    calc prCoin (fun x : P × Ω => E x.1 x.2 ∧ m x.1 ≤ δ)
        = avg fun p : P => prCoin fun ω : Ω => E p ω ∧ m p ≤ δ := by
          unfold prCoin
          exact Tape.avg_prod _
      _ ≤ avg fun _ : P => δ := avg_mono hp
      _ = δ := avg_const δ
  calc prCoin (fun x : P × Ω => acc x.1 x.2 ∧ m x.1 ≤ δ)
      ≤ prCoin (fun x : P × Ω => (E x.1 x.2 ∧ m x.1 ≤ δ) ∨ brk x.1 x.2) :=
        prCoin_mono fun x hx => (h x.1 x.2 hx.1).imp (fun hd => ⟨hd, hx.2⟩) id
    _ ≤ _ := (prCoin_or_le _ _).trans (add_le_add hmiss le_rfl)

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Committing late does not help, at a circuit's two levels, from a partly opened outer trace.** -/
theorem late_le_missProb_opened (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂])
    {C : Circuit} {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Pf.Refines Pc)
    (blk : Fin nf → Finset G) (oblk : Fin nc → Finset G) (hsub : ∀ v k, blk v ⊆ oblk k ↔ r.parent v = k)
    (hparts : ∀ X ∈ l₁.blocks.parts, ∃ k, oblk k = X) (τ₁ : C.Trace) (O : Finset (Fin C.N))
    (R : (l₁.design Finset.univ).Ω → Finset G → C.Trace) (hR : ∀ ω X, ∀ g ∈ O, R ω X g = τ₁ g) :
    ∃ τ₂ : C.Trace, (∀ g ∈ O, τ₂ g = τ₁ g) ∧
      prCoin (fun ω : (l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω) =>
          ∀ X ∈ (l₁.design Finset.univ).draw ω.1,
            Disjoint ((l₂.design X).draw (ω.2 X)) ((Pf.unopened Pc.committed O ∪ Pf.incorrect (R ω.1 X)).image blk)) ≤
        L.design.missProb ((Pf.unopened Pc.committed O ∪ Pf.incorrect τ₂).image blk) := by
  classical
  set U := Pf.unopened Pc.committed O
  let 𝒯 : Finset (Finset (Finset G)) := Finset.univ.filter fun T =>
    ∃ Y : C.Trace, (∀ g ∈ O, Y g = τ₁ g) ∧ T = (U ∪ Pf.incorrect Y).image blk
  have mem𝒯 : ∀ T, T ∈ 𝒯 ↔ ∃ Y : C.Trace, (∀ g ∈ O, Y g = τ₁ g) ∧ T = (U ∪ Pf.incorrect Y).image blk :=
    fun T => by simp [𝒯]
  have hfilter : ∀ (s : Finset (Fin nf)) (k : Fin nc),
      (s.image blk).filter (· ⊆ oblk k) = (s.filter (r.parent · = k)).image blk := by
    intro s k
    rw [Finset.filter_image]
    exact congrArg (Finset.image blk) (Finset.filter_congr fun v _ => hsub v k)
  have hglue : ∀ c : Finset G → Finset (Finset G), (∀ X, c X ∈ 𝒯) →
      ∃ T ∈ 𝒯, ∀ X ∈ l₁.blocks.parts, T.filter (· ⊆ X) = (c X).filter (· ⊆ X) := by
    intro c hc
    choose Yc hYc hcY using fun X => (mem𝒯 (c X)).1 (hc X)
    obtain ⟨hres, hfil⟩ := r.glue (X := τ₁) (Y := fun k => Yc (oblk k)) (O := O) fun k => hYc (oblk k)
    refine ⟨(U ∪ Pf.incorrect (Pc.splice τ₁ fun k => Yc (oblk k))).image blk, (mem𝒯 _).2 ⟨_, hres, rfl⟩, ?_⟩
    intro X hX
    obtain ⟨k, rfl⟩ := hparts X hX
    rw [hcY (oblk k), hfilter, hfilter, hfil k]
  obtain ⟨T, hT, hle⟩ := L.late_le_missProb hL 𝒯 hglue (fun ω X => (U ∪ Pf.incorrect (R ω X)).image blk)
    fun ω X => (mem𝒯 _).2 ⟨R ω X, hR ω X, rfl⟩
  obtain ⟨Y, hY, rfl⟩ := (mem𝒯 T).1 hT
  exact ⟨Y, hY, hle⟩

/-- **Committing late does not help, at a circuit's two levels**, from an outer trace opened on every outer committed
wire. -/
theorem late_le_missProb_trace (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂])
    {C : Circuit} {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Pf.Refines Pc)
    (blk : Fin nf → Finset G) (oblk : Fin nc → Finset G) (hsub : ∀ v k, blk v ⊆ oblk k ↔ r.parent v = k)
    (hparts : ∀ X ∈ l₁.blocks.parts, ∃ k, oblk k = X) (τ₁ : C.Trace)
    (R : (l₁.design Finset.univ).Ω → Finset G → C.Trace) (hR : ∀ ω X, ∀ g ∈ Pc.committed, R ω X g = τ₁ g) :
    ∃ τ₂ : C.Trace, (∀ g ∈ Pc.committed, τ₂ g = τ₁ g) ∧
      prCoin (fun ω : (l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω) =>
          ∀ X ∈ (l₁.design Finset.univ).draw ω.1,
            Disjoint ((l₂.design X).draw (ω.2 X)) ((Pf.incorrect (R ω.1 X)).image blk)) ≤
        L.design.missProb ((Pf.incorrect τ₂).image blk) := by
  obtain ⟨τ₂, h, hle⟩ := L.late_le_missProb_opened hL r blk oblk hsub hparts τ₁ Pc.committed R hR
  refine ⟨τ₂, h, ?_⟩
  simpa only [Pf.unopened_self, Finset.empty_union] using hle

/-- **The sampling step of soundness at a miss `δ`, interiors committed late.** The prover's coins `p`, drawn before
the samples and independently of them, fix the outer trace `τ₁ p` at the opened gates `O p`. After the outer sample the
prover picks at each outer block, by any rule `R p`, an interior extending it. Where every accepted run's inner samples
miss the late interiors' incorrect inner blocks or a break occurs, the run is accepted while the largest miss over
the interiors extending `τ₁ p` is at most `δ` with probability at most `δ` plus the break's. -/
theorem prCoin_accept_le_late (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂])
    {C : Circuit} {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Pf.Refines Pc)
    (blk : Fin nf → Finset G) (oblk : Fin nc → Finset G) (hsub : ∀ v k, blk v ⊆ oblk k ↔ r.parent v = k)
    (hparts : ∀ X ∈ l₁.blocks.parts, ∃ k, oblk k = X) {P : Type} [Fintype P] [Nonempty P]
    (τ₁ : P → C.Trace) (O : P → Finset (Fin C.N)) (R : P → (l₁.design Finset.univ).Ω → Finset G → C.Trace)
    (hR : ∀ p ω X, ∀ g ∈ O p, R p ω X g = τ₁ p g)
    (acc brk : P → (l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω) → Prop)
    (h : ∀ p ω, acc p ω → (∀ X ∈ (l₁.design Finset.univ).draw ω.1,
        Disjoint ((l₂.design X).draw (ω.2 X)) ((Pf.unopened Pc.committed (O p) ∪ Pf.incorrect (R p ω.1 X)).image blk)) ∨
      brk p ω) (δ : ℝ≥0∞) :
    prCoin (fun x : P × ((l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω)) => acc x.1 x.2 ∧
        (⨆ (τ₂ : C.Trace) (_ : ∀ g ∈ O x.1, τ₂ g = τ₁ x.1 g),
          L.design.missProb ((Pf.unopened Pc.committed (O x.1) ∪ Pf.incorrect τ₂).image blk)) ≤ δ) ≤
      δ + prCoin (fun x : P × ((l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω)) => brk x.1 x.2) := by
  refine prCoin_accept_le_of _ acc brk (fun p => ⨆ (τ₂ : C.Trace) (_ : ∀ g ∈ O p, τ₂ g = τ₁ p g),
    L.design.missProb ((Pf.unopened Pc.committed (O p) ∪ Pf.incorrect τ₂).image blk)) (fun p => ?_) h δ
  obtain ⟨τ₂, hτ₂, hle⟩ := L.late_le_missProb_opened hL r blk oblk hsub hparts (τ₁ p) (O p) (R p) (hR p)
  exact hle.trans (le_iSup₂ (f := fun (τ : C.Trace) (_ : ∀ g ∈ O p, τ g = τ₁ p g) =>
    L.design.missProb ((Pf.unopened Pc.committed (O p) ∪ Pf.incorrect τ).image blk)) τ₂ hτ₂)

end Levels

end Verity
