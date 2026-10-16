/-
`Returns x P`: every value the verdict `x` returns satisfies `P`, with its rules for `pure`, `throw`, `bind`, branches and
`Array.mapM`, and the facts a setup's `pure` closes on: a scheme the registry found has its id's registry scheme
(`MerkleScheme.Named`). `returns_walk` (`Flock.With.ReturnsWalk`) walks a do block with them; `Flock.With.Deployed` uses it
on the verifier's setups.
-/
import Flock.Verify
import Flock.With.Hashes

namespace Flock.With

/-- Every value `x` returns satisfies `P`. -/
def Returns {α : Type} (x : V α) (P : α → Prop) : Prop := ∀ a, x = .ok a → P a

namespace Returns

theorem pure' {α : Type} {a : α} {P : α → Prop} (h : P a) : Returns (pure a) P := by
  intro b hb; cases hb; exact h

theorem throw' {α : Type} {e : String} {P : α → Prop} : Returns (throw e : V α) P := by
  intro b hb; cases hb

theorem bind' {α β : Type} {x : V α} {f : α → V β} {P : β → Prop} (h : ∀ a, Returns (f a) P) : Returns (x >>= f) P := by
  intro b hb
  cases x with
  | error e => cases hb
  | ok a => exact h a b hb

theorem bindQ {α β : Type} {x : V α} {f : α → V β} {Q : α → Prop} {P : β → Prop} (hx : Returns x Q)
    (h : ∀ a, Q a → Returns (f a) P) : Returns (x >>= f) P := by
  intro b hb
  cases hxe : x with
  | error e => rw [hxe] at hb; cases hb
  | ok a => rw [hxe] at hb; exact h a (hx a hxe) b hb

theorem mapM_list {α β : Type} {f : α → V β} {Q : β → Prop} (h : ∀ a, Returns (f a) Q) :
    ∀ l : List α, Returns (l.mapM f) fun ys => ∀ y ∈ ys, Q y
  | [] => pure' fun _ hy => nomatch hy
  | a :: l => by
    rw [List.mapM_cons]
    refine bindQ (h a) fun b hb => bindQ (mapM_list h l) fun bs hbs => pure' fun y hy => ?_
    rcases List.mem_cons.mp hy with rfl | hy
    · exact hb
    · exact hbs y hy

theorem mapM {α β : Type} {f : α → V β} {Q : β → Prop} (h : ∀ a, Returns (f a) Q) (xs : Array α) :
    Returns (xs.mapM f) fun ys => ∀ y ∈ ys, Q y := by
  rw [Array.mapM_eq_mapM_toList]
  intro ys hys
  cases hl : List.mapM f xs.toList with
  | error e => rw [hl] at hys; cases hys
  | ok l =>
    rw [hl] at hys; cases hys
    intro y hy
    exact mapM_list h _ l hl y (List.mem_toArray.mp hy)

theorem ite {α : Type} {c : Prop} [Decidable c] {x y : V α} {P : α → Prop} (h₁ : c → Returns x P)
    (h₂ : ¬c → Returns y P) : Returns (if c then x else y) P := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact h₁ hc
  · rw [ite_eq_right hc]; exact h₂ hc

theorem dite {α : Type} {c : Prop} [Decidable c] {x : c → V α} {y : ¬c → V α} {P : α → Prop}
    (h₁ : ∀ h, Returns (x h) P) (h₂ : ∀ h, Returns (y h) P) : Returns (if h : c then x h else y h) P := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact h₁ hc
  · rw [dite_eq_right hc]; exact h₂ hc

/-- A bind on a statement that has a registry scheme. -/
theorem bindNamed {β : Type} {x : V Stmt} {f : Stmt → V β} {P : β → Prop} (hx : Returns x fun st => st.merkle.Named)
    (h : ∀ st : Stmt, st.merkle.Named → Returns (f st) P) : Returns (x >>= f) P :=
  bindQ hx h

/-- A bind on statements that each have a registry scheme. -/
theorem bindNamedMapM {α β : Type} {xs : Array α} {f : α → V Stmt} {g : Array Stmt → V β} {P : β → Prop}
    (hf : ∀ a, Returns (f a) fun st => st.merkle.Named) (hg : ∀ sts : Array Stmt, (∀ st ∈ sts, st.merkle.Named) →
      Returns (g sts) P) : Returns (xs.mapM f >>= g) P :=
  bindQ (mapM hf xs) hg

end Returns

theorem named_of_find? {id : String} {ms : MerkleScheme} (h : MerkleScheme.find? id = some ms) : ms.Named := by
  intro s hs
  have hid : ms.id = id := by simpa using List.find?_some h
  rw [hid, h] at hs
  exact (Option.some.inj hs).symm

theorem named_singleton {spec : SessionSpec} {st : Stmt} (h : st.merkle.Named) :
    ∀ s ∈ (spec, #[st]).2, s.merkle.Named := by
  intro s hs
  rw [List.mem_singleton.mp (Array.mem_toList_iff.mpr hs)]
  exact h

end Flock.With
