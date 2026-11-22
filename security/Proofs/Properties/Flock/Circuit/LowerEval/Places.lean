module

public import Proofs.Properties.Flock.Circuit.LowerEval.Lift
public import Proofs.Core.QSortPerm

/-! Where the lift to bits places a composite with word-primitive nodes: after `S`'s Definitions, each node that calls a
word primitive calling that primitive's Boolean Definition. -/

@[expose] public section

namespace Flock.Chain

open Flock.Program
open Std.Do
set_option mvcgen.warning false

/-- The closure's walk so far from `d`: what it has seen it has expanded, a composite's callees into `todo`, a word
primitive's Boolean Definition into `toB`, and `toB` holds only primitives. -/
def WalkInv (S : Prog) (d : ℕ) (seen : Std.HashSet ℕ) (todo : Array ℕ) (at_ : ℕ) (toB : Std.HashMap ℕ ℕ) : Prop :=
  0 < todo.size ∧ todo[0]! = d ∧ at_ ≤ todo.size ∧ (∀ j < at_, seen.contains todo[j]! = true) ∧
  (∀ x : ℕ, seen.contains x = true → ((S.defs[x]!).prim = false → ∀ nd ∈ (S.defs[x]!).body.nodes, nd.fn ∈ todo.toList) ∧
    (isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = true →
      ∃ bi, toB[x]? = some bi ∧ S.bodyOf? (S.defs[x]!).key = some bi)) ∧
  ∀ (x bi : ℕ), toB[x]? = some bi → isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = true

/-- Each primitive's entry in `wordy` is whether it is a word primitive, and a composite `d` of word-primitive nodes,
at least one, is wordy. -/
def WordyInv (S : Prog) (d : ℕ) (wordy : Std.HashMap ℕ Bool) : Prop :=
  ∀ (x : ℕ) (b : Bool), wordy[x]? = some b → ((S.defs[x]!).prim = true → b = isWordPrim (S.defs[x]!).prim (S.defs[x]!).key) ∧
    (x = d → b = true)

/-- `d` lifted: at `toL`'s place for it, after `S`'s Definitions, its key in the index, each node that calls a word
primitive calling that primitive's Boolean Definition. -/
def Placed (S : Prog) (d : ℕ) (defs : Array Defn) (index : Std.HashMap String ℕ) (toL : Std.HashMap ℕ ℕ) : Prop :=
  ∃ e, toL[d]? = some e ∧ S.defs.size ≤ e ∧ index.contains (Flock.Lift.liftedKey (S.defs[d]!).key) = true ∧
    ∃ fnOf primOf gatesOf, defs[e]? = some (Flock.Lift.liftDef (S.defs[d]!) fnOf primOf gatesOf) ∧
      ∀ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true → S.bodyOf? (S.defs[nd.fn]!).key = some (fnOf nd.fn)

/-- The lift loop so far: `defs` and `index` extend `S`'s, the word primitives `d`'s nodes call still at their Boolean
Definitions, and `d` placed once the loop has passed it. -/
def LiftInv (S : Prog) (d : ℕ) (pre : List ℕ) (defs : Array Defn) (index : Std.HashMap String ℕ)
    (toL : Std.HashMap ℕ ℕ) : Prop :=
  KInv S defs index ∧ (∀ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true →
    ∃ bi, S.bodyOf? (S.defs[nd.fn]!).key = some bi ∧ toL[nd.fn]? = some bi) ∧ (d ∈ pre → Placed S d defs index toL)

theorem arr_get!_push {a : Array ℕ} {x i : ℕ} (h : i < a.size) : (a.push x)[i]! = a[i]! := by
  simp only [getElem!_def, Array.getElem?_push, if_neg (show i ≠ a.size by omega)]

theorem arr_get!_append {a b : Array ℕ} {i : ℕ} (h : i < a.size) : (a ++ b)[i]! = a[i]! := by
  simp only [getElem!_def, Array.getElem?_append_left h]

theorem walk_init (S : Prog) (d : ℕ) : WalkInv S d ∅ #[d] 0 ∅ :=
  ⟨by simp, by simp, by simp, fun j hj => absurd hj (by omega), fun x hx => by simp at hx, fun x bi h => by simp at h⟩

theorem walk_seen {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ} {toB : Std.HashMap ℕ ℕ}
    (h : WalkInv S d seen todo at_ toB) (hlt : at_ < todo.size) (hc : seen.contains todo[at_] = true) :
    WalkInv S d seen todo (at_ + 1) toB := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  refine ⟨h1, h2, by omega, fun j hj => ?_, h5, h6⟩
  rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
  · exact h4 j hj
  · rw [getElem!_pos todo j hlt]; exact hc

theorem walk_word {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ} {toB : Std.HashMap ℕ ℕ}
    {defn : Defn} {bk : String} {bi : ℕ} (h : WalkInv S d seen todo at_ toB) (hlt : at_ < todo.size)
    (hx : S.defs[todo[at_]]? = some defn) (hw : isWordPrim defn.prim defn.key = true)
    (hbk : S.boolean[defn.key]? = some bk) (hbi : S.index[bk]? = some bi) :
    WalkInv S d (seen.insert todo[at_]) (todo.push bi) (at_ + 1) (toB.insert todo[at_] bi) := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  have hD : S.defs[todo[at_]]! = defn := by rw [getElem!_def, hx]
  have hpr : defn.prim = true := by simp only [isWordPrim, Bool.and_eq_true] at hw; exact hw.1.1
  refine ⟨by simp, by rw [arr_get!_push h1]; exact h2, by simp; omega, fun j hj => ?_, fun y hy => ?_,
    fun y b hy => ?_⟩
  · rw [arr_get!_push (by omega), Std.HashSet.contains_insert]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · rw [h4 j hj, Bool.or_true]
    · rw [getElem!_pos todo j hlt]; simp
  · rw [Std.HashSet.contains_insert] at hy
    by_cases hyx : todo[at_] = y
    · subst hyx
      rw [hD]
      refine ⟨fun hp => absurd hpr (by rw [hp]; decide), fun _ => ⟨bi, by simp, ?_⟩⟩
      unfold Prog.bodyOf?
      rw [hbk]
      exact hbi
    · have hy' : seen.contains y = true := by simpa [hyx] using hy
      obtain ⟨hc, hwd⟩ := h5 y hy'
      refine ⟨fun hp nd hnd => Array.mem_toList_iff.2 (Array.mem_push.2 (Or.inl (Array.mem_toList_iff.1
        (hc hp nd hnd)))), fun hw' => ?_⟩
      obtain ⟨b, hb, hb'⟩ := hwd hw'
      exact ⟨b, by rw [Std.HashMap.getElem?_insert, if_neg (by simpa using hyx)]; exact hb, hb'⟩
  · rw [Std.HashMap.getElem?_insert] at hy
    by_cases hyx : todo[at_] = y
    · subst hyx; rw [hD]; exact hw
    · rw [if_neg (by simpa using hyx)] at hy; exact h6 y b hy

theorem walk_comp {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ} {toB : Std.HashMap ℕ ℕ}
    {defn : Defn} (h : WalkInv S d seen todo at_ toB) (hlt : at_ < todo.size)
    (hx : S.defs[todo[at_]]? = some defn) (hcp : (!defn.prim) = true) :
    WalkInv S d (seen.insert todo[at_]) (todo ++ Array.map (fun x => x.fn) defn.body.nodes) (at_ + 1) toB := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  have hD : S.defs[todo[at_]]! = defn := by rw [getElem!_def, hx]
  have hpr : defn.prim = false := by simpa using hcp
  refine ⟨by simp; omega, by rw [arr_get!_append h1]; exact h2, by simp; omega, fun j hj => ?_, fun y hy => ?_, h6⟩
  · rw [arr_get!_append (by omega), Std.HashSet.contains_insert]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · rw [h4 j hj, Bool.or_true]
    · rw [getElem!_pos todo j hlt]; simp
  · rw [Std.HashSet.contains_insert] at hy
    by_cases hyx : todo[at_] = y
    · subst hyx
      rw [hD]
      refine ⟨fun _ nd hnd => Array.mem_toList_iff.2 (Array.mem_append.2 (Or.inr (Array.mem_map.2 ⟨nd, hnd, rfl⟩))),
        fun hw => ?_⟩
      simp [isWordPrim, hpr] at hw
    · have hy' : seen.contains y = true := by simpa [hyx] using hy
      obtain ⟨hc, hwd⟩ := h5 y hy'
      exact ⟨fun hp nd hnd => Array.mem_toList_iff.2 (Array.mem_append.2 (Or.inl (Array.mem_toList_iff.1
        (hc hp nd hnd)))), hwd⟩

theorem walk_prim {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ} {toB : Std.HashMap ℕ ℕ}
    {defn : Defn} (h : WalkInv S d seen todo at_ toB) (hlt : at_ < todo.size)
    (hx : S.defs[todo[at_]]? = some defn) (hnw : ¬ isWordPrim defn.prim defn.key = true)
    (hcp : ¬ (!defn.prim) = true) : WalkInv S d (seen.insert todo[at_]) todo (at_ + 1) toB := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  have hD : S.defs[todo[at_]]! = defn := by rw [getElem!_def, hx]
  have hpr : defn.prim = true := by simpa using hcp
  refine ⟨h1, h2, by omega, fun j hj => ?_, fun y hy => ?_, h6⟩
  · rw [Std.HashSet.contains_insert]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · rw [h4 j hj, Bool.or_true]
    · rw [getElem!_pos todo j hlt]; simp
  · rw [Std.HashSet.contains_insert] at hy
    by_cases hyx : todo[at_] = y
    · subst hyx
      rw [hD]
      exact ⟨fun hp => absurd hpr (by rw [hp]; decide), fun hw => absurd hw hnw⟩
    · exact h5 y (by simpa [hyx] using hy)

theorem walk_done {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ} {toB : Std.HashMap ℕ ℕ}
    (hp1 : (S.defs[d]!).prim = false) (h : WalkInv S d seen todo at_ toB) (hge : ¬ at_ < todo.size) :
    seen.contains d = true ∧ ∀ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true →
      ∃ bi, S.bodyOf? (S.defs[nd.fn]!).key = some bi ∧ toB[nd.fn]? = some bi := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  have hall : ∀ x ∈ todo.toList, seen.contains x = true := by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    have hj' : j < todo.size := by simpa using hj
    have := h4 j (by omega)
    rw [getElem!_pos todo j hj'] at this
    rw [Array.getElem_toList]; exact this
  have hd0 : seen.contains d = true := by
    have := h4 0 (by omega); rwa [h2] at this
  refine ⟨hd0, fun nd hnd hwn => ?_⟩
  obtain ⟨hc, -⟩ := h5 d hd0
  have hfn := hall _ (hc hp1 nd hnd)
  obtain ⟨-, hw⟩ := h5 _ hfn
  obtain ⟨bi, hb, hb'⟩ := hw hwn
  exact ⟨bi, hb', hb⟩

/-- At the walk's end, every composite it has seen has its callees seen, each word primitive among them at its Boolean
Definition in `toB`. -/
theorem walk_done_at {S : Prog} {d x : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ} {toB : Std.HashMap ℕ ℕ}
    (h : WalkInv S d seen todo at_ toB) (hge : ¬ at_ < todo.size) (hx : seen.contains x = true)
    (hpx : (S.defs[x]!).prim = false) :
    ∀ nd ∈ (S.defs[x]!).body.nodes, seen.contains nd.fn = true ∧
      (isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true →
        ∃ bi, S.bodyOf? (S.defs[nd.fn]!).key = some bi ∧ toB[nd.fn]? = some bi) := by
  obtain ⟨-, -, h3, h4, h5, -⟩ := h
  intro nd hnd
  have hin := (h5 x hx).1 hpx nd hnd
  obtain ⟨j, hj, hjx⟩ := List.getElem_of_mem hin
  have hj' : j < todo.size := by simpa using hj
  have hs : seen.contains nd.fn = true := by
    have := h4 j (by omega)
    rw [getElem!_pos todo j hj'] at this
    rw [← hjx, Array.getElem_toList]; exact this
  refine ⟨hs, fun hwn => ?_⟩
  obtain ⟨bi, hb, hb'⟩ := (h5 _ hs).2 hwn
  exact ⟨bi, hb', hb⟩

theorem wordy_init (S : Prog) (d : ℕ) :
    WordyInv S d ∅ ∧ ∀ x ∈ ([] : List ℕ), ((∅ : Std.HashMap ℕ Bool)[x]?).isSome = true :=
  ⟨fun x b h => by simp at h, fun x hx => by simp at hx⟩

theorem wordy_prim {S : Prog} {d : ℕ} {w : Std.HashMap ℕ Bool} {pref : List ℕ} {c : ℕ}
    (hp1 : (S.defs[d]!).prim = false) (h : WordyInv S d w ∧ ∀ x ∈ pref, (w[x]?).isSome = true)
    (hpr : (S.defs[c]!).prim = true) :
    WordyInv S d (w.insert c (isWordPrim (S.defs[c]!).prim (S.defs[c]!).key)) ∧
      ∀ x ∈ pref ++ [c], ((w.insert c (isWordPrim (S.defs[c]!).prim (S.defs[c]!).key))[x]?).isSome = true := by
  obtain ⟨hW, hS⟩ := h
  refine ⟨fun x b hx => ?_, fun x hx => ?_⟩
  · rw [Std.HashMap.getElem?_insert] at hx
    by_cases hcx : c = x
    · subst hcx
      simp only [beq_self_eq_true, ↓reduceIte, Option.some.injEq] at hx
      subst hx
      refine ⟨fun _ => rfl, fun e => ?_⟩
      subst e; rw [hp1] at hpr; cases hpr
    · rw [if_neg (by simpa using hcx)] at hx; exact hW x b hx
  · rw [Std.HashMap.getElem?_insert]
    by_cases hcx : c = x
    · subst hcx; simp
    · rw [if_neg (by simpa using hcx)]
      simp only [List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | hx
      · exact hS x hx
      · exact absurd hx.symm hcx

theorem inner_step {S : Prog} {d : ℕ} {w : Std.HashMap ℕ Bool} {Q : Prop} {pref : List Node} {c : Node}
    {b b' : Bool} (hW : WordyInv S d w ∧ Q) (hx : w[c.fn]? = some b')
    (h : b = true ∨ ∀ y ∈ pref, isWordPrim (S.defs[y.fn]!).prim (S.defs[y.fn]!).key = false) :
    (b || b') = true ∨ ∀ y ∈ pref ++ [c], isWordPrim (S.defs[y.fn]!).prim (S.defs[y.fn]!).key = false := by
  cases hwc : isWordPrim (S.defs[c.fn]!).prim (S.defs[c.fn]!).key
  · rcases h with h | h
    · left; simp [h]
    · right; intro y hy
      simp only [List.mem_append, List.mem_singleton] at hy
      rcases hy with hy | rfl
      · exact h y hy
      · exact hwc
  · left
    have hpr : (S.defs[c.fn]!).prim = true := by
      simp only [isWordPrim, Bool.and_eq_true] at hwc; exact hwc.1.1
    have := (hW.1 c.fn b' hx).1 hpr
    rw [this, hwc]; simp

theorem wordy_comp {S : Prog} {d : ℕ} {w : Std.HashMap ℕ Bool} {pref : List ℕ} {c : ℕ} {r : Bool}
    (hex : ∃ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true)
    (h : WordyInv S d w ∧ ∀ x ∈ pref, (w[x]?).isSome = true) (hpr : ¬ (S.defs[c]!).prim = true)
    (hr : r = true ∨ ∀ y ∈ (S.defs[c]!).body.nodes.toList, isWordPrim (S.defs[y.fn]!).prim (S.defs[y.fn]!).key = false) :
    WordyInv S d (w.insert c r) ∧ ∀ x ∈ pref ++ [c], ((w.insert c r)[x]?).isSome = true := by
  obtain ⟨hW, hS⟩ := h
  refine ⟨fun x b hx => ?_, fun x hx => ?_⟩
  · rw [Std.HashMap.getElem?_insert] at hx
    by_cases hcx : c = x
    · subst hcx
      simp only [beq_self_eq_true, ↓reduceIte, Option.some.injEq] at hx
      subst hx
      refine ⟨fun h' => absurd h' hpr, fun e => ?_⟩
      subst e
      rcases hr with hr | hr
      · exact hr
      · obtain ⟨nd, hnd, hw⟩ := hex
        have h1 := hr _ (Array.mem_toList_iff.2 hnd)
        rw [hw] at h1
        cases h1
    · rw [if_neg (by simpa using hcx)] at hx; exact hW x b hx
  · rw [Std.HashMap.getElem?_insert]
    by_cases hcx : c = x
    · subst hcx; simp
    · rw [if_neg (by simpa using hcx)]
      simp only [List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | hx
      · exact hS x hx
      · exact absurd hx.symm hcx

theorem lift_init {S : Prog} {d : ℕ} {toB : Std.HashMap ℕ ℕ}
    (hT : ∀ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true →
      ∃ bi, S.bodyOf? (S.defs[nd.fn]!).key = some bi ∧ toB[nd.fn]? = some bi) :
    LiftInv S d [] S.defs S.index toB := ⟨kinv_init S, hT, fun h => by simp at h⟩

theorem lift_step {S : Prog} {d : ℕ} {pref : List ℕ} {defs : Array Defn} {index : Std.HashMap String ℕ}
    {toL : Std.HashMap ℕ ℕ} {c : ℕ} {wordy : Std.HashMap ℕ Bool}
    (h : LiftInv S d pref defs index toL) (hc : (wordy[c]?.getD false && !(S.defs[c]!).prim) = true)
    (hk : ¬ index.contains (Flock.Lift.liftDef (S.defs[c]!) (fun f => toL[f]?.getD f) (fun f => (defs[f]!).prim)
      (fun f => (defs[f]!).gates)).key = true) :
    LiftInv S d (pref ++ [c])
      (defs.push (Flock.Lift.liftDef (S.defs[c]!) (fun f => toL[f]?.getD f) (fun f => (defs[f]!).prim)
        (fun f => (defs[f]!).gates)))
      (index.insert (Flock.Lift.liftDef (S.defs[c]!) (fun f => toL[f]?.getD f) (fun f => (defs[f]!).prim)
        (fun f => (defs[f]!).gates)).key defs.size)
      (toL.insert c defs.size) := by
  obtain ⟨hK, hT, hP⟩ := h
  have hcp : (S.defs[c]!).prim = false := by
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hc; exact hc.2
  refine ⟨kinv_push hK hk, fun nd hnd hw => ?_, fun hd => ?_⟩
  · obtain ⟨bi, hb, ht⟩ := hT nd hnd hw
    refine ⟨bi, hb, ?_⟩
    have hne : c ≠ nd.fn := by
      intro e
      have h1 := hw
      rw [← e, hcp] at h1
      simp [isWordPrim] at h1
    rw [Std.HashMap.getElem?_insert, if_neg (by simpa using hne)]; exact ht
  · simp only [List.mem_append, List.mem_singleton] at hd
    by_cases hcd : c = d
    · subst hcd
      refine ⟨defs.size, by simp, hK.1, ?_, (fun f => toL[f]?.getD f), (fun f => (defs[f]!).prim),
        (fun f => (defs[f]!).gates), by simp, fun nd hnd hw => ?_⟩
      · show (index.insert (Flock.Lift.liftedKey (S.defs[c]!).key) defs.size).contains
          (Flock.Lift.liftedKey (S.defs[c]!).key) = true
        simp
      · obtain ⟨bi, hb, ht⟩ := hT nd hnd hw
        rw [hb]
        show some bi = some (toL[nd.fn]?.getD nd.fn)
        rw [ht]; rfl
    · rcases hd with hd | hd
      · obtain ⟨e, he, hle, hcon, fnOf, primOf, gatesOf, hde, hfn⟩ := hP hd
        refine ⟨e, ?_, hle, ?_, fnOf, primOf, gatesOf, ?_, hfn⟩
        · rw [Std.HashMap.getElem?_insert, if_neg (by simpa using hcd)]; exact he
        · rw [Std.HashMap.contains_insert, hcon, Bool.or_true]
        · have he' := (Array.getElem?_eq_some_iff.1 hde).1
          rw [Array.getElem?_push, if_neg (by omega)]; exact hde
      · exact absurd hd.symm hcd

theorem lift_skip {S : Prog} {d : ℕ} {pref suff ord : List ℕ} {defs : Array Defn} {index : Std.HashMap String ℕ}
    {toL : Std.HashMap ℕ ℕ} {c : ℕ} {wordy : Std.HashMap ℕ Bool} (hp1 : (S.defs[d]!).prim = false)
    (hW : WordyInv S d wordy ∧ ∀ x ∈ ord, (wordy[x]?).isSome = true) (hord : ord = pref ++ c :: suff)
    (h : LiftInv S d pref defs index toL) (hc : ¬ (wordy[c]?.getD false && !(S.defs[c]!).prim) = true) :
    LiftInv S d (pref ++ [c]) defs index toL := by
  obtain ⟨hK, hT, hP⟩ := h
  refine ⟨hK, hT, fun hd => ?_⟩
  simp only [List.mem_append, List.mem_singleton] at hd
  rcases hd with hd | hd
  · exact hP hd
  · subst hd
    exfalso; apply hc
    have hm : d ∈ ord := by rw [hord]; simp
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (hW.2 d hm)
    have h1 := (hW.1 d b hb).2 rfl
    simp [hb, h1, hp1]

theorem lift_final {S : Prog} {d : ℕ} {defs : Array Defn} {index : Std.HashMap String ℕ} {toL : Std.HashMap ℕ ℕ}
    {seen : Std.HashSet ℕ} (hs : seen.contains d = true)
    (h : LiftInv S d (seen.toArray.qsort (fun x1 x2 => decide (x1 < x2))).toList defs index toL) :
    S.defs.size ≤ toL[d]?.getD d ∧ ∃ fnOf primOf gatesOf,
      defs[toL[d]?.getD d]? = some (Flock.Lift.liftDef (S.defs[d]!) fnOf primOf gatesOf) ∧
      ∀ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true → S.bodyOf? (S.defs[nd.fn]!).key = some (fnOf nd.fn) := by
  have hm : d ∈ (seen.toArray.qsort (fun x1 x2 => decide (x1 < x2))).toList := by
    rw [Array.mem_toList_iff, (Verity.QSort.qsort_perm _ _).mem_iff, Std.HashSet.mem_toArray,
      Std.HashSet.mem_iff_contains]
    exact hs
  obtain ⟨e, he, hle, -, fnOf, primOf, gatesOf, hde, hfn⟩ := h.2.2 hm
  simp only [he, Option.getD_some]
  exact ⟨hle, fnOf, primOf, gatesOf, hde, hfn⟩

theorem closure_places (S : Prog) (d : ℕ) (hd : d < S.defs.size) (hp1 : (S.defs[d]!).prim = false)
    (hex : ∃ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true) :
    ⦃⌜True⌝⦄ Flock.Lift.closure S #[d] ⦃⇓? r => ⌜S.defs.size ≤ r.2 d ∧ ∃ fnOf primOf gatesOf,
      r.1.defs[r.2 d]? = some (Flock.Lift.liftDef (S.defs[d]!) fnOf primOf gatesOf) ∧
      ∀ nd ∈ (S.defs[d]!).body.nodes, isWordPrim (S.defs[nd.fn]!).prim (S.defs[nd.fn]!).key = true → S.bodyOf? (S.defs[nd.fn]!).key = some (fnOf nd.fn)⌝⦄ := by
  unfold Flock.Lift.closure
  mvcgen [hashMap_forIn_spec, exc_throw_spec, -Spec.throw_Except] invariants
  · ⇓? ⟨_, seen, todo, at_, toB⟩ => ⌜WalkInv S d seen todo at_ toB⌝
  · ⇓? ⟨c, wordy⟩ => ⌜WordyInv S d wordy ∧ ∀ x ∈ c.prefix, (wordy[x]?).isSome = true⌝
  · ⇓? ⟨c, w⟩ => ⌜w = true ∨ ∀ y ∈ c.prefix, isWordPrim (S.defs[y.fn]!).prim (S.defs[y.fn]!).key = false⌝
  · ⇓? ⟨c, defs, index, toL⟩ => ⌜LiftInv S d c.prefix defs index toL⌝
  all_goals first
    | trivial
    | exact walk_init S d
    | (exact walk_seen (by assumption) (by assumption) (by assumption))
    | (exact walk_word (by assumption) (by assumption) (by assumption) (by assumption) (by assumption)
        (by assumption))
    | (exact walk_comp (by assumption) (by assumption) (by assumption) (by assumption))
    | (exact walk_prim (by assumption) (by assumption) (by assumption) (by assumption) (by assumption))
    | exact wordy_init S d
    | (exact wordy_prim hp1 (by assumption) (by assumption))
    | (exact inner_step (by assumption) (by assumption) (by assumption))
    | exact Or.inr (fun y hy => by simp at hy)
    | (exact wordy_comp hex (by assumption) (by assumption) (by assumption))
    | (exact lift_init (walk_done hp1 (by assumption) (by assumption)).2)
    | (exact lift_step (by assumption) (by assumption) (by assumption))
    | (exact lift_skip hp1 (by assumption) (by assumption) (by assumption) (by assumption))
    | (exact lift_final (walk_done hp1 (by assumption) (by assumption)).1 (by assumption))

/-- **`LiftPlaces`** holds. -/
theorem liftPlaces : LiftPlaces := by
  intro S L d e hcont h hd hp hn
  unfold Flock.Lift.lift at h
  rcases hc : Flock.Lift.closure S #[d] with _ | ⟨Q, fnOf⟩
  · rw [hc] at h; simp [bind, Except.bind] at h
  rw [hc] at h
  have h' : (Q, fnOf d) = (L, e) := by simpa [bind, Except.bind, pure, Except.pure] using h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj h'
  obtain ⟨hle, fnOf, primOf, gatesOf, hde, hfn⟩ :=
    exc_of_triple (closure_places S d hd hp.1 ⟨_, Array.getElem_mem hn, hp.2 _ (Array.getElem_mem hn)⟩) hc
  exact ⟨hle, fnOf, primOf, gatesOf, hde, fun nd hnd => hfn nd hnd (hp.2 nd hnd)⟩

/-- **`LiftCalls`** holds. -/
theorem liftCalls : LiftCalls := liftCalls_of liftPlaces

end Flock.Chain
