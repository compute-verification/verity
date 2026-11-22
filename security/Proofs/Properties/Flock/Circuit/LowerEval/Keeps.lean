module

public import Proofs.Properties.Flock.Circuit.LowerEval.Widths

/-! What the lift to bits keeps of its Program: its Definitions, and its Boolean map's targets. -/

@[expose] public section

namespace Flock.Chain

open Flock.Program
open Std.Do
set_option mvcgen.warning false

/-- `defs` and `index` extend `P`'s: its Definitions in place, its keys at their places. -/
def KInv (P : Prog) (defs : Array Defn) (index : Std.HashMap String ℕ) : Prop :=
  P.defs.size ≤ defs.size ∧ (∀ i < P.defs.size, defs[i]? = P.defs[i]?) ∧
    ∀ (k : String) (i : ℕ), P.index[k]? = some i → index[k]? = some i

theorem kinv_init (P : Prog) : KInv P P.defs P.index := ⟨Nat.le_refl _, fun _ _ => rfl, fun _ _ h => h⟩

theorem kinv_push {P : Prog} {defs : Array Defn} {index : Std.HashMap String ℕ} {l : Defn} {n : ℕ}
    (h : KInv P defs index) (hc : ¬ index.contains l.key = true) : KInv P (defs.push l) (index.insert l.key n) := by
  obtain ⟨hle, hpre, hidx⟩ := h
  refine ⟨by simp; omega, fun i hi => ?_, fun k i hki => ?_⟩
  · rw [Array.getElem?_push, if_neg (by omega)]; exact hpre i hi
  · have hk := hidx k i hki
    rw [Std.HashMap.getElem?_insert]
    split
    · rename_i he
      simp only [beq_iff_eq] at he
      subst he
      exact absurd (by rw [Std.HashMap.contains_eq_isSome_getElem?, hk]; rfl) hc
    · exact hk

theorem hashMap_forIn_spec {β : Type} (m : Std.HashMap ℕ ℕ) (b : β)
    (f : ℕ × ℕ → β → Except String (ForInStep β)) : ⦃⌜True⌝⦄ forIn m b f ⦃⇓? _ => ⌜True⌝⦄ := exc_spec _

theorem closure_keeps (P : Prog) (start : Array ℕ) :
    ⦃⌜True⌝⦄ Flock.Lift.closure P start ⦃⇓? r => ⌜KInv P r.1.defs r.1.index ∧ r.1.boolean = P.boolean⌝⦄ := by
  unfold Flock.Lift.closure
  mvcgen [hashMap_forIn_spec, exc_throw_spec, -Spec.throw_Except] invariants
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? ⟨_, defs, index, _⟩ => ⌜KInv P defs index⌝
  all_goals first
    | trivial
    | exact kinv_init P
    | (exact kinv_push (by assumption) (by assumption))
    | (exact ⟨by assumption, rfl⟩)
    | assumption
    | exact exc_wp_true _ _ (fun r => by cases r <;> exact True.intro)

/-- **`LiftKeeps`**: the lift to bits keeps `S`'s Definitions in place and the place of every Definition its Boolean map
names, where every value of the map is a key of its index. -/
theorem liftKeeps {S L : Prog} {d e : ℕ} (hcont : ∀ (k v : String), S.boolean[k]? = some v → S.index.contains v = true)
    (h : Flock.Lift.lift S d = .ok (L, e)) :
    S.defs.size ≤ L.defs.size ∧ (∀ i < S.defs.size, L.defs[i]? = S.defs[i]?) ∧ ∀ k, L.bodyOf? k = S.bodyOf? k := by
  unfold Flock.Lift.lift at h
  rcases hc : Flock.Lift.closure S #[d] with _ | ⟨Q, fnOf⟩
  · rw [hc] at h; simp [bind, Except.bind] at h
  rw [hc] at h
  have h' : (Q, fnOf d) = (L, e) := by simpa [bind, Except.bind, pure, Except.pure] using h
  obtain ⟨rfl, -⟩ := Prod.mk.inj h'
  obtain ⟨⟨hle, hpre, hidx⟩, hb⟩ := exc_of_triple (closure_keeps S #[d]) hc
  refine ⟨hle, hpre, fun k => ?_⟩
  unfold Prog.bodyOf?
  rw [hb]
  rcases hk : S.boolean[k]? with _ | v
  · rfl
  · have hv := hcont k v hk
    rw [Std.HashMap.contains_eq_isSome_getElem?] at hv
    obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1 hv
    simp only [Option.bind_eq_bind, Option.bind_some]
    rw [hi, hidx v i hi]

end Flock.Chain
