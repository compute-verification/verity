module

public import Proofs.Core.Circuit
public import Proofs.Core.LevelsMiss

/-!
# Committing late does not help, at a circuit's two levels

Two levels drawn over a circuit's nested partitions: the inner blocks are the units of `Pf`, the outer blocks those of
`Pc`, and `Pf` refines `Pc` (`r`). The levels' blocks are sets of gates of some type `G`, and `blk`, `oblk` place the
inner and outer units among them, an inner unit inside an outer one exactly when the outer is its parent. A trace for
the inner blocks restricts to the outer trace `τ₁` when it agrees with `τ₁` on the outer committed wires.

`Levels.late_le_missProb_trace`: fix `τ₁`. After the outer sample is drawn, the prover picks at each outer block, by any
rule, a trace that restricts to `τ₁`, its interior. Every inner sample then misses that trace's incorrect inner blocks
with probability at most `missProb L.design` of one trace's incorrect inner blocks, a trace fixed in advance that
restricts to `τ₁`. It is `Levels.late_le_missProb` at the incorrect sets of the traces restricting to `τ₁`, which glue
(`Partition.Refines.glue`).
-/

@[expose] public section

namespace Verity

open Game

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Committing late does not help, at a circuit's two levels.** -/
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
  classical
  let 𝒯 : Finset (Finset (Finset G)) := Finset.univ.filter fun T =>
    ∃ Y : C.Trace, (∀ g ∈ Pc.committed, Y g = τ₁ g) ∧ T = (Pf.incorrect Y).image blk
  have mem𝒯 : ∀ T, T ∈ 𝒯 ↔ ∃ Y : C.Trace, (∀ g ∈ Pc.committed, Y g = τ₁ g) ∧ T = (Pf.incorrect Y).image blk :=
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
    obtain ⟨hres, hfil⟩ := r.glue (X := τ₁) (Y := fun k => Yc (oblk k)) fun k => hYc (oblk k)
    refine ⟨(Pf.incorrect (Pc.splice τ₁ fun k => Yc (oblk k))).image blk, (mem𝒯 _).2 ⟨_, hres, rfl⟩, ?_⟩
    intro X hX
    obtain ⟨k, rfl⟩ := hparts X hX
    rw [hcY (oblk k), hfilter, hfilter, hfil k]
  obtain ⟨T, hT, hle⟩ := L.late_le_missProb hL 𝒯 hglue (fun ω X => (Pf.incorrect (R ω X)).image blk)
    fun ω X => (mem𝒯 _).2 ⟨R ω X, hR ω X, rfl⟩
  obtain ⟨Y, hY, rfl⟩ := (mem𝒯 T).1 hT
  exact ⟨Y, hY, hle⟩

end Levels

end Verity
