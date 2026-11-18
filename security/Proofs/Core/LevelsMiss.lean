module

public import Proofs.Core.Levels

/-!
# The sampling step of SPOT's guarantee, for a bad set fixed before the coins

The sampling step of SPOT's soundness at a miss `δ`, with no argument system in it: a set `T` of blocks that
does not depend on the coins escapes the sample with probability exactly `missProb T` (`prCoin_misses`), so a run that
is accepted only where the sample misses its bad set, or where a break occurs, is accepted with a bad set whose miss is
at most `δ` with probability at most `δ` plus the break's (`prCoin_accept_le`). The bad set may depend on the prover's
coins, drawn before the sample and independently of it.

Committing late does not help: at a design drawn in two stages (`SamplingDesign.bind`), a bad set chosen
at each outer block after the outer draw, by any rule (`prCoin_bind_late`), escapes the inner samples with probability
at most that of the best choice fixed in advance (`late_le_fixed`). At two levels (`Levels.late_le_missProb`), where the
bad sets on offer glue across outer blocks, as the incorrect inner blocks of traces that restrict to one outer trace do,
that is at most `missProb L.design T` at one of them, fixed in advance.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

namespace SamplingDesign

variable {U V : Type}

/-- **A set fixed before the coins escapes the sample with probability its miss.** -/
theorem prCoin_misses (D : SamplingDesign U) (T : Finset U) :
    prCoin (fun ω => Disjoint (D.draw ω) T) = D.missProb T := rfl

/-- A set of at least `K` units escapes the sample no more often than the count curve at `K`. -/
theorem missProb_le_miss (D : SamplingDesign U) {T : Finset U} {K : ℕ} (h : K ≤ T.card) :
    D.missProb T ≤ D.miss K :=
  le_iSup₂ (f := fun (B : Finset U) (_ : K ≤ B.card) => D.missProb B) T h

/-- **The sampling step of soundness at a miss `δ`.** The prover's coins `p`, drawn before the sample and
independently of it, fix its bad set `T p`. Where every accepted run's sample misses `T p` or a break occurs, the run
is accepted with a bad set whose miss is at most `δ` with probability at most `δ` plus the break's. -/
theorem prCoin_accept_le {P : Type} [Fintype P] [Nonempty P] (D : SamplingDesign U) (T : P → Finset U)
    (acc brk : P → D.Ω → Prop) (h : ∀ p ω, acc p ω → Disjoint (D.draw ω) (T p) ∨ brk p ω) (δ : ℝ≥0∞) :
    prCoin (fun x : P × D.Ω => acc x.1 x.2 ∧ D.missProb (T x.1) ≤ δ) ≤
      δ + prCoin (fun x : P × D.Ω => brk x.1 x.2) := by
  classical
  have hmiss : prCoin (fun x : P × D.Ω => Disjoint (D.draw x.2) (T x.1) ∧ D.missProb (T x.1) ≤ δ) ≤ δ := by
    have hp : ∀ p : P, prCoin (fun ω : D.Ω => Disjoint (D.draw ω) (T p) ∧ D.missProb (T p) ≤ δ) ≤ δ := by
      intro p
      by_cases hδ : D.missProb (T p) ≤ δ
      · exact (prCoin_mono fun ω hω => hω.1).trans hδ
      · exact (prCoin_mono (q := fun _ => False) fun ω hω => hδ hω.2).trans (prCoin_false.trans_le zero_le)
    calc prCoin (fun x : P × D.Ω => Disjoint (D.draw x.2) (T x.1) ∧ D.missProb (T x.1) ≤ δ)
        = avg fun p : P => prCoin fun ω : D.Ω => Disjoint (D.draw ω) (T p) ∧ D.missProb (T p) ≤ δ := by
          unfold prCoin
          exact Tape.avg_prod _
      _ ≤ avg fun _ : P => δ := avg_mono hp
      _ = δ := avg_const δ
  calc prCoin (fun x : P × D.Ω => acc x.1 x.2 ∧ D.missProb (T x.1) ≤ δ)
      ≤ prCoin (fun x : P × D.Ω =>
          (Disjoint (D.draw x.2) (T x.1) ∧ D.missProb (T x.1) ≤ δ) ∨ brk x.1 x.2) :=
        prCoin_mono fun x hx => (h x.1 x.2 hx.1).imp (fun hd => ⟨hd, hx.2⟩) id
    _ ≤ _ := (prCoin_or_le _ _).trans (add_le_add hmiss le_rfl)

variable [Fintype U] [DecidableEq U]

/-- **A bad set chosen after the outer draw**: at a two-stage design `D.bind F`, with the bad set `B ω₁ u` at each
outer unit `u` chosen after the outer coins `ω₁` by any rule, every inner sample misses its bad set with probability
`E_{ω₁}[∏_{u ∈ D.draw ω₁} missProb (F u) (B ω₁ u)]`, since each inner sample is drawn on coins of its own after the
choice. -/
theorem prCoin_bind_late (D : SamplingDesign U) (F : U → SamplingDesign V) (B : D.Ω → U → Finset V) :
    prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) =>
        ∀ u ∈ D.draw ω.1, Disjoint ((F u).draw (ω.2 u)) (B ω.1 u)) =
      avg fun ω₁ => ∏ u ∈ D.draw ω₁, (F u).missProb (B ω₁ u) := by
  calc prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) =>
          ∀ u ∈ D.draw ω.1, Disjoint ((F u).draw (ω.2 u)) (B ω.1 u))
      = avg fun ω₁ => prCoin fun ω₂ : (u : U) → (F u).Ω =>
          ∀ u ∈ D.draw ω₁, Disjoint ((F u).draw (ω₂ u)) (B ω₁ u) := by
        unfold prCoin
        exact Tape.avg_prod _
    _ = _ := congrArg avg (funext fun ω₁ =>
        prCoin_pi_forall (D.draw ω₁) fun u (c : (F u).Ω) => Disjoint ((F u).draw c) (B ω₁ u))

/-- **Committing late does not help**, at one outer and one inner stage: a bad set chosen at each outer unit after
the outer draw, from the sets `S u` on offer there, escapes the inner samples no more often than one choice `b` fixed
in advance, which picks at each `u` the set of `S u` that its inner design misses most often. -/
theorem late_le_fixed (D : SamplingDesign U) (F : U → SamplingDesign V) (S : U → Finset (Finset V))
    (hS : ∀ u, (S u).Nonempty) (B : D.Ω → U → Finset V) (hB : ∀ ω u, B ω u ∈ S u) :
    ∃ b : U → Finset V, (∀ u, b u ∈ S u) ∧
      prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) =>
          ∀ u ∈ D.draw ω.1, Disjoint ((F u).draw (ω.2 u)) (B ω.1 u)) ≤
        prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) =>
          ∀ u ∈ D.draw ω.1, Disjoint ((F u).draw (ω.2 u)) (b u)) := by
  choose b hbS hbmax using fun u => (S u).exists_max_image (fun s => (F u).missProb s) (hS u)
  refine ⟨b, hbS, ?_⟩
  rw [prCoin_bind_late D F B, prCoin_bind_late D F fun _ => b]
  exact avg_mono fun ω₁ => Finset.prod_le_prod fun u _ => hbmax u _ (hB ω₁ u)

end SamplingDesign

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Committing late does not help, at two levels.** The bad sets on offer `𝒯` are sets of inner blocks, those
of the traces for the inner blocks that restrict to one fixed outer trace, and they glue (`hglue`): any choice of one
of them at each outer block agrees, inside every outer block, with one of them. After the outer sample is drawn the
prover chooses, at each outer block `X`, a bad set `R ω₁ X` of `𝒯` by any rule. Every inner sample then misses the
bad set chosen at its outer block with probability at most `missProb L.design T` at one `T` of `𝒯`, fixed in
advance. -/
theorem late_le_missProb (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂])
    (𝒯 : Finset (Finset (Finset G)))
    (hglue : ∀ c : Finset G → Finset (Finset G), (∀ X, c X ∈ 𝒯) →
      ∃ T ∈ 𝒯, ∀ X ∈ l₁.blocks.parts, T.filter (· ⊆ X) = (c X).filter (· ⊆ X))
    (R : (l₁.design Finset.univ).Ω → Finset G → Finset (Finset G)) (hR : ∀ ω X, R ω X ∈ 𝒯) :
    ∃ T ∈ 𝒯,
      prCoin (fun ω : (l₁.design Finset.univ).Ω × ((X : Finset G) → (l₂.design X).Ω) =>
          ∀ X ∈ (l₁.design Finset.univ).draw ω.1, Disjoint ((l₂.design X).draw (ω.2 X)) (R ω.1 X)) ≤
        L.design.missProb T := by
  have hi := L.inside
  rw [hL] at hi
  simp only [Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
  obtain ⟨h₁, h₂⟩ := hi
  have hne : 𝒯.Nonempty := ⟨_, hR (Classical.arbitrary _) Finset.univ⟩
  obtain ⟨c, hc𝒯, hle⟩ := SamplingDesign.late_le_fixed (l₁.design Finset.univ) (fun X => l₂.design X)
    (fun _ => 𝒯) (fun _ => hne) R hR
  obtain ⟨T, hT𝒯, hT⟩ := hglue c hc𝒯
  refine ⟨T, hT𝒯, hle.trans (le_of_eq ?_)⟩
  rw [SamplingDesign.prCoin_bind_late (l₁.design Finset.univ) (fun X => l₂.design X) (fun _ => c),
    L.missProb_cons hL (List.cons_ne_nil _ _)]
  refine congrArg avg (funext fun ω => Finset.prod_congr rfl fun X hX => ?_)
  have hpart := (h₁ ω X hX).1
  have hin : ∀ ω' b, b ∈ (l₂.design X).draw ω' → b ⊆ X := fun ω' b hb => (h₂ X hpart ω' b hb).2
  show (l₂.design X).missProb (c X) = (l₂.design X).missProb T
  rw [← SamplingDesign.missProb_filter (l₂.design X) (· ⊆ X) hin (c X),
    ← SamplingDesign.missProb_filter (l₂.design X) (· ⊆ X) hin T, hT X hpart]

end Levels

end Verity
