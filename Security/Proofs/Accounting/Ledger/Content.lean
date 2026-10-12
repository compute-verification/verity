import Mathlib.Tactic.Ring
import Proofs.Accounting.Ledger.Basic
import Definitions.Accounting.Ledger.Pin

/-!
# Completeness and a crossing's capacity bound its content

If the rest of the actual circuit sees only its own inputs and what crossed region `ρ`'s boundary outward
(`Complete`), its own inputs don't depend on the secret, and at most `k` bits cross on every run, then no decoder of
what the rest sees recovers more than `2^(k+1) − 1` secrets: the content is at most the crossing
(`contentAtMost_of_complete`). With a capacity per tick, `k` is the region's ticks times the capacity's outward crossing
(`contentAtMost_of_capacity`). This is the upper pin a capacity gives with no protocol; a protocol's upper pin, such as
the network certifier's, is a smaller bound on the same content.
-/

namespace Accounting.Ledger

/-- At most `2^(k+1) − 1` bit strings have at most `k` bits. -/
theorem encard_length_le_add_one (k : ℕ) : {l : List Bool | l.length ≤ k}.encard + 1 ≤ 2 ^ (k + 1) := by
  induction k with
  | zero =>
    have : {l : List Bool | l.length ≤ 0} = {[]} := by
      ext l
      simp
    rw [this, Set.encard_singleton]
    norm_num
  | succ k ih =>
    have hsub : {l : List Bool | l.length ≤ k + 1} ⊆
        insert [] (List.cons true '' {l | l.length ≤ k} ∪ List.cons false '' {l | l.length ≤ k}) := by
      rintro (_ | ⟨b, l⟩) hl
      · exact Set.mem_insert _ _
      · replace hl : l.length ≤ k := by simpa using hl
        cases b
        · exact Or.inr (Or.inr ⟨l, hl, rfl⟩)
        · exact Or.inr (Or.inl ⟨l, hl, rfl⟩)
    have hle := (Set.encard_le_encard hsub).trans <| (Set.encard_insert_le _ _).trans <|
      add_le_add ((Set.encard_union_le _ _).trans
        (add_le_add (Set.encard_image_le _ _) (Set.encard_image_le _ _))) le_rfl
    calc {l : List Bool | l.length ≤ k + 1}.encard + 1
        ≤ ({l : List Bool | l.length ≤ k}.encard + {l : List Bool | l.length ≤ k}.encard + 1) + 1 :=
          add_le_add hle le_rfl
      _ = 2 * ({l : List Bool | l.length ≤ k}.encard + 1) := by ring
      _ ≤ 2 * 2 ^ (k + 1) := by gcongr
      _ = 2 ^ (k + 1 + 1) := by ring

variable {Λ G Z V R : Type} [Fintype Λ] [DecidableEq Λ]

/-- **Completeness and a bound on the crossing bound the content.** If what the rest sees (`wire`) is a function of its
own inputs (`rest`, the same for every secret) and what crossed `ρ` outward, and at most `k` bits cross on every run,
the content is at most `2^(k+1) − 1` secrets, whichever runs are accepted. -/
theorem contentAtMost_of_complete {A : Z → Annotation Λ G} {ρ : Region Λ} {wire : Z → V} {rest : Z → R}
    {crossed : Z → List Bool} (hc : Complete A ρ wire rest crossed) (hrest : ∀ z z', rest z = rest z') {k : ℕ}
    (hk : ∀ z, (A z).ledger ρ .crossOut ≤ k) (accepted : Z → Prop) :
    ContentAtMost accepted wire ((2 ^ (k + 1) - 1 : ℕ) : ℕ∞) := by
  intro Zs _ dec hdec
  have inj : Set.InjOn crossed Zs := fun z hz z' hz' h => by
    have hw : wire z = wire z' := hc.1 z z' (Prod.ext (hrest z z') h)
    rw [← hdec z hz, ← hdec z' hz', hw]
  have himg : crossed '' Zs ⊆ {l : List Bool | l.length ≤ k} := by
    rintro _ ⟨z, -, rfl⟩
    exact (hc.2 z).trans (hk z)
  have hZ : Zs.encard + 1 ≤ 2 ^ (k + 1) := by
    rw [← inj.encard_image]
    exact (add_le_add (Set.encard_le_encard himg) le_rfl).trans (encard_length_le_add_one k)
  have hfin : Zs.encard ≠ ⊤ := fun h => by simp [h] at hZ
  obtain ⟨n, hn⟩ := ENat.ne_top_iff_exists.mp hfin
  rw [← hn] at hZ ⊢
  have : n + 1 ≤ 2 ^ (k + 1) := by exact_mod_cast hZ
  exact_mod_cast (by omega : n ≤ 2 ^ (k + 1) - 1)

/-- **A crossing's capacity bounds its content**: with capacity `c` per tick on `ρ`'s part, at most
`#ticks · c(crossOut)` bits cross, so the content is at most `2^(#ticks · c(crossOut) + 1) − 1` secrets. -/
theorem contentAtMost_of_capacity {A : Z → Annotation Λ G} {ρ : Region Λ} {wire : Z → V} {rest : Z → R}
    {crossed : Z → List Bool} (hc : Complete A ρ wire rest crossed) (hrest : ∀ z z', rest z = rest z')
    {c : Ledger G} (hcap : Capacity A ρ.part c) (accepted : Z → Prop) :
    ContentAtMost accepted wire ((2 ^ (ρ.ticks.card * c .crossOut + 1) - 1 : ℕ) : ℕ∞) :=
  contentAtMost_of_complete hc hrest (fun z => ledger_le_card_smul hcap z ρ.ticks .crossOut) accepted

end Accounting.Ledger
