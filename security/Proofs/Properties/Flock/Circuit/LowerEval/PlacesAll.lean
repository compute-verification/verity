module

public import Proofs.Properties.Flock.Circuit.LowerEval.Places

/-! Where the lift to bits places every composite its walk reaches: a word primitive's Boolean Definition, a composite
that reaches no word primitive left where it is, and one that reaches a word primitive (`ReachesWord`), however deep,
lifted after `S`'s Definitions with each node calling where the lift places its callee (`closure_placesAll`). -/

@[expose] public section

namespace Flock.Chain

open Flock.Program
open Std.Do
set_option mvcgen.warning false
/-- The `wordy` pass so far over `pref`: its keys are `pref`'s, each entry whether its Definition reaches a word
primitive, and each composite's callees come before it. -/
@[irreducible] def WordyAll (S : Prog) (pref : List ℕ) (w : Std.HashMap ℕ Bool) : Prop :=
  (∀ x : ℕ, (w[x]?).isSome = true ↔ x ∈ pref) ∧ (∀ (x : ℕ) b, w[x]? = some b → (b = true ↔ ReachesWord S x)) ∧
  ∀ p y s, pref = p ++ y :: s → (S.defs[y]!).prim = false → ∀ nd ∈ (S.defs[y]!).body.nodes, nd.fn ∈ p

/-- The `toL` pass so far over `pref`: `defs` extends `S`'s, `toL` is `toB` at each primitive, a composite has an
entry only once passed and wordy, and each passed wordy composite is placed lifted after `S`'s Definitions, its nodes
calling what `toL` gives their callees. -/
@[irreducible] def LiftAll (S : Prog) (toB : Std.HashMap ℕ ℕ) (w : Std.HashMap ℕ Bool) (pref : List ℕ) (defs : Array Defn)
    (toL : Std.HashMap ℕ ℕ) : Prop :=
  S.defs.size ≤ defs.size ∧ (∀ x : ℕ, (S.defs[x]!).prim = true → toL[x]? = toB[x]?) ∧
  (∀ x : ℕ, (S.defs[x]!).prim = false → (toL[x]?).isSome = true → x ∈ pref ∧ w[x]? = some true) ∧
  ∀ y ∈ pref, (S.defs[y]!).prim = false → w[y]? = some true → ∃ e, toL[y]? = some e ∧ S.defs.size ≤ e ∧
    ∃ fnOf primOf gatesOf, defs[e]? = some (Flock.Lift.liftDef (S.defs[y]!) fnOf primOf gatesOf) ∧
      (∀ nd ∈ (S.defs[y]!).body.nodes, fnOf nd.fn = toL[nd.fn]?.getD nd.fn) ∧
      ∀ nd ∈ (S.defs[y]!).body.nodes, (S.defs[nd.fn]!).prim = false → ReachesWord S nd.fn → toL[nd.fn]?.getD nd.fn < e


theorem snoc_split {α : Type} {pref p s : List α} {c y : α} (h : pref ++ [c] = p ++ y :: s) :
    (s = [] ∧ p = pref ∧ y = c) ∨ ∃ s', s = s' ++ [c] ∧ pref = p ++ y :: s' := by
  rcases List.eq_nil_or_concat s with rfl | ⟨s', z, rfl⟩
  · have h' : pref ++ [c] = p ++ [y] := h
    have := List.append_inj' h' rfl
    exact Or.inl ⟨rfl, this.1.symm, by simpa using this.2.symm⟩
  · have h' : pref ++ [c] = (p ++ y :: s') ++ [z] := by simpa using h
    have := List.append_inj' h' rfl
    have hz : z = c := by simpa using this.2.symm
    exact Or.inr ⟨s', by rw [hz]; simp, this.1⟩

theorem reachesWord_comp_iff {S : Prog} {x : ℕ} (hp : (S.defs[x]!).prim = false) :
    ReachesWord S x ↔ ∃ nd ∈ (S.defs[x]!).body.nodes, ReachesWord S nd.fn := by
  constructor
  · intro h
    cases h with
    | word hw => simp [isWordPrim, hp] at hw
    | comp _ hn hr => exact ⟨_, hn, hr⟩
  · rintro ⟨nd, hn, hr⟩; exact .comp hp hn hr

theorem wordyAll_nil (S : Prog) : WordyAll S [] ∅ := by
  unfold WordyAll; exact ⟨fun x => by simp, fun x b h => by simp at h, fun p y s h => by simp at h⟩

theorem wordyAll_insert {S : Prog} {pref : List ℕ} {w : Std.HashMap ℕ Bool} {c : ℕ} {b : Bool}
    (h : WordyAll S pref w) (hb : b = true ↔ ReachesWord S c)
    (hc : (S.defs[c]!).prim = false → ∀ nd ∈ (S.defs[c]!).body.nodes, nd.fn ∈ pref) :
    WordyAll S (pref ++ [c]) (w.insert c b) := by
  unfold WordyAll at h ⊢
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨fun x => ?_, fun x b' hx => ?_, fun p y s hs => ?_⟩
  · rw [Std.HashMap.getElem?_insert]
    by_cases hcx : c = x
    · subst hcx; simp
    · rw [if_neg (by simpa using hcx), h1 x]; simp [Ne.symm hcx]
  · rw [Std.HashMap.getElem?_insert] at hx
    by_cases hcx : c = x
    · subst hcx; simp at hx; subst hx; exact hb
    · simp [hcx] at hx; exact h2 x b' hx
  · rcases snoc_split hs with ⟨rfl, rfl, rfl⟩ | ⟨s', rfl, hpre⟩
    · exact hc
    · exact h3 p y s' hpre

theorem wordyAll_prim {S : Prog} {pref : List ℕ} {w : Std.HashMap ℕ Bool} {c : ℕ}
    (hp : (S.defs[c]!).prim = true) (h : WordyAll S pref w) :
    WordyAll S (pref ++ [c]) (w.insert c (isWordPrim (S.defs[c]!).prim (S.defs[c]!).key)) := by
  refine wordyAll_insert h ⟨fun hw => .word hw, fun hr => ?_⟩ (fun hf => by simp [hf] at hp)
  cases hr with
  | word hw => exact hw
  | comp hf _ _ => simp [hf] at hp

theorem wordyAll_comp {S : Prog} {pref : List ℕ} {w : Std.HashMap ℕ Bool} {c : ℕ} {b : Bool}
    (hp : ¬ (S.defs[c]!).prim = true) (h : WordyAll S pref w)
    (hi : (b = true ↔ ∃ x, x ∈ (S.defs[c]!).body.nodes.toList ∧ w[x.fn]? = some true) ∧
      ∀ x : Node, x ∈ (S.defs[c]!).body.nodes.toList → (w[x.fn]?).isSome = true) :
    WordyAll S (pref ++ [c]) (w.insert c b) := by
  have hp' : (S.defs[c]!).prim = false := by simpa using hp
  obtain ⟨hi1, hi2⟩ := hi
  have h' := h; unfold WordyAll at h'
  refine wordyAll_insert h ?_ (fun _ nd hn => (h'.1 nd.fn).1 (hi2 nd (by simpa using hn)))
  rw [hi1, reachesWord_comp_iff hp']
  constructor
  · rintro ⟨x, hx, hw⟩; exact ⟨x, by simpa using hx, (h'.2.1 _ _ hw).1 rfl⟩
  · rintro ⟨x, hx, hr⟩
    have hx' : x ∈ (S.defs[c]!).body.nodes.toList := by simpa using hx
    obtain ⟨b', hb'⟩ := Option.isSome_iff_exists.1 (hi2 x hx')
    exact ⟨x, hx', by rw [hb', (h'.2.1 _ _ hb').2 hr]⟩

theorem wordyAll_inner_nil {w : Std.HashMap ℕ Bool} :
    (false = true ↔ ∃ x, x ∈ ([] : List Node) ∧ w[x.fn]? = some true) ∧
      ∀ x : Node, x ∈ ([] : List Node) → (w[x.fn]?).isSome = true := by simp

theorem wordyAll_inner_step {w : Std.HashMap ℕ Bool} {pref : List Node} {cur : Node} {acc b : Bool}
    (hb : w[cur.fn]? = some b)
    (h : (acc = true ↔ ∃ x, x ∈ pref ∧ w[x.fn]? = some true) ∧ ∀ x : Node, x ∈ pref → (w[x.fn]?).isSome = true) :
    ((acc || b) = true ↔ ∃ x, x ∈ pref ++ [cur] ∧ w[x.fn]? = some true) ∧
      ∀ x : Node, x ∈ pref ++ [cur] → (w[x.fn]?).isSome = true := by
  obtain ⟨h1, h2⟩ := h
  refine ⟨?_, fun x hx => ?_⟩
  · rw [Bool.or_eq_true, h1]
    constructor
    · rintro (⟨x, hx, hw⟩ | rfl)
      · exact ⟨x, List.mem_append_left _ hx, hw⟩
      · exact ⟨cur, by simp, hb⟩
    · rintro ⟨x, hx, hw⟩
      rcases List.mem_append.1 hx with hx | hx
      · exact Or.inl ⟨x, hx, hw⟩
      · simp at hx; subst hx; rw [hb] at hw; exact Or.inr (Option.some.inj hw)
  · rcases List.mem_append.1 hx with hx | hx
    · exact h2 x hx
    · simp at hx; subst hx; simp [hb]

theorem walk_reach {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ}
    {toB : Std.HashMap ℕ ℕ} (h : WalkInv S d seen todo at_ toB) (hge : ¬ at_ < todo.size) {x : ℕ}
    (hx : Reach S d x) : seen.contains x = true := by
  induction hx with
  | root => have := h.2.2.2.1 0 (by have := h.1; have := h.2.2.1; omega); rwa [h.2.1] at this
  | callee _ hp hn ih => exact (walk_done_at h hge ih hp _ hn).1

theorem order_perm (seen : Std.HashSet ℕ) :
    (seen.toArray.qsort fun x1 x2 => decide (x1 < x2)).toList.Perm seen.toList := by
  have h := Verity.QSort.qsort_perm seen.toArray (fun x1 x2 => decide (x1 < x2))
  rw [← Std.HashSet.toList_toArray]; exact Array.perm_iff_toList_perm.1 h

theorem order_nodup (seen : Std.HashSet ℕ) :
    (seen.toArray.qsort fun x1 x2 => decide (x1 < x2)).toList.Nodup :=
  (order_perm seen).nodup_iff.2 (Std.HashSet.distinct_toList.imp fun h e => by simp [e] at h)

theorem mem_order {seen : Std.HashSet ℕ} {x : ℕ} (h : seen.contains x = true) :
    x ∈ (seen.toArray.qsort fun x1 x2 => decide (x1 < x2)).toList :=
  (order_perm seen).mem_iff.2 (Std.HashSet.mem_toList.2 (Std.HashSet.mem_iff_contains.2 h))

theorem liftAll_nil {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ}
    {toB : Std.HashMap ℕ ℕ} (w : Std.HashMap ℕ Bool) (h : WalkInv S d seen todo at_ toB) :
    LiftAll S toB w [] S.defs toB ∧ ([] ++ (seen.toArray.qsort fun x1 x2 => decide (x1 < x2)).toList).Nodup := by
  unfold LiftAll
  refine ⟨⟨Nat.le_refl _, fun _ _ => rfl, fun x hp hs => ?_, fun y hy => by simp at hy⟩, by simpa using order_nodup seen⟩
  obtain ⟨bi, hbi⟩ := Option.isSome_iff_exists.1 hs
  have := h.2.2.2.2.2 x bi hbi; simp [isWordPrim, hp] at this

theorem liftAll_skip {S : Prog} {toB : Std.HashMap ℕ ℕ} {w : Std.HashMap ℕ Bool} {pref suff : List ℕ}
    {cur : ℕ} {defs : Array Defn} {toL : Std.HashMap ℕ ℕ}
    (hn : ¬ (w[cur]?.getD false && !(S.defs[cur]!).prim) = true)
    (h : LiftAll S toB w pref defs toL ∧ (pref ++ cur :: suff).Nodup) :
    LiftAll S toB w (pref ++ [cur]) defs toL ∧ (pref ++ [cur] ++ suff).Nodup := by
  unfold LiftAll at h ⊢
  obtain ⟨⟨h1, h2, h3, h4⟩, hnd⟩ := h
  refine ⟨⟨h1, h2, fun x hp hs => ?_, fun y hy hp hw => ?_⟩, by simpa using hnd⟩
  · obtain ⟨hx, hw⟩ := h3 x hp hs; exact ⟨List.mem_append_left _ hx, hw⟩
  · rcases List.mem_append.1 hy with hy | hy
    · exact h4 y hy hp hw
    · simp at hy; subst hy; simp [hw, hp] at hn

theorem liftAll_place {S : Prog} {toB : Std.HashMap ℕ ℕ} {w : Std.HashMap ℕ Bool} {L pref suff : List ℕ}
    {cur : ℕ} {defs : Array Defn} {toL : Std.HashMap ℕ ℕ} (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ)
    (hW : WordyAll S L w) (hL : L = pref ++ cur :: suff)
    (hc : (w[cur]?.getD false && !(S.defs[cur]!).prim) = true)
    (h : LiftAll S toB w pref defs toL ∧ (pref ++ cur :: suff).Nodup) :
    LiftAll S toB w (pref ++ [cur])
        (defs.push (Flock.Lift.liftDef (S.defs[cur]!) (fun f => toL[f]?.getD f) primOf gatesOf))
        (toL.insert cur defs.size) ∧ (pref ++ [cur] ++ suff).Nodup := by
  unfold LiftAll at h ⊢
  unfold WordyAll at hW
  obtain ⟨⟨h1, h2, h3, h4⟩, hnd⟩ := h
  have hpc : (S.defs[cur]!).prim = false := by
    cases hq : (S.defs[cur]!).prim <;> simp_all
  have hwc : w[cur]? = some true := by
    cases hq : w[cur]? with
    | none => simp [hq] at hc
    | some b => cases b <;> simp_all
  have hcp : cur ∉ pref := fun hm => by
    have hnd' := hnd; rw [List.nodup_append] at hnd'; exact (hnd'.2.2 cur hm cur (by simp)) rfl
  have hcal : ∀ y ∈ pref ++ [cur], (S.defs[y]!).prim = false → ∀ nd ∈ (S.defs[y]!).body.nodes, nd.fn ∈ pref := by
    intro y hy hp nd hn
    rcases List.mem_append.1 hy with hy | hy
    · obtain ⟨p, s0, rfl⟩ := List.mem_iff_append.1 hy
      have := hW.2.2 p y (s0 ++ cur :: suff) (by rw [hL]; simp) hp nd hn
      exact List.mem_append_left _ this
    · simp at hy; subst hy; exact hW.2.2 pref y suff hL hp nd hn
  have hne : ∀ y ∈ pref ++ [cur], (S.defs[y]!).prim = false → ∀ nd ∈ (S.defs[y]!).body.nodes, nd.fn ≠ cur :=
    fun y hy hp nd hn he => hcp (he ▸ hcal y hy hp nd hn)
  refine ⟨⟨by simp; omega, fun x hp => ?_, fun x hp hs => ?_, fun y hy hp hw => ?_⟩, by simpa using hnd⟩
  · have : cur ≠ x := fun e => by subst e; simp [hp] at hpc
    rw [Std.HashMap.getElem?_insert, if_neg (by simpa using this)]; exact h2 x hp
  · rw [Std.HashMap.getElem?_insert] at hs
    by_cases hcx : cur = x
    · subst hcx; exact ⟨by simp, hwc⟩
    · rw [if_neg (by simpa using hcx)] at hs
      obtain ⟨hx, hw⟩ := h3 x hp hs; exact ⟨List.mem_append_left _ hx, hw⟩
  · by_cases hcy : cur = y
    · subst hcy
      refine ⟨defs.size, by simp, h1, fun f => toL[f]?.getD f, primOf, gatesOf, by simp, fun nd hn => ?_,
        fun nd hn hp' hr => ?_⟩
      · rw [Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm (hne cur (by simp) hp nd hn))]
      · have hin := hcal cur (by simp) hp nd hn
        obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 ((hW.1 nd.fn).2 (by rw [hL]; exact List.mem_append_left _ hin))
        have hbt : b = true := (hW.2.1 nd.fn b hb).2 hr
        subst hbt
        obtain ⟨e', he', -, f', p', g', hd', -⟩ := h4 nd.fn hin hp' hb
        have hel : e' < defs.size := by
          rcases Nat.lt_or_ge e' defs.size with h | h
          · exact h
          · simp [Array.getElem?_eq_none h] at hd'
        rw [Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm (hne cur (by simp) hp nd hn)), he']
        exact hel
    · have hy' : y ∈ pref := by
        rcases List.mem_append.1 hy with hy | hy
        · exact hy
        · simp at hy; exact absurd hy.symm hcy
      obtain ⟨e, he, hse, fnOf, pO, gO, hd, hf, hlt⟩ := h4 y hy' hp hw
      have hel : e < defs.size := by
        rcases Nat.lt_or_ge e defs.size with h | h
        · exact h
        · simp [Array.getElem?_eq_none h] at hd
      refine ⟨e, ?_, hse, fnOf, pO, gO, ?_, fun nd hn => ?_, fun nd hn hp' hr => ?_⟩
      · rw [Std.HashMap.getElem?_insert, if_neg (by simpa using hcy)]; exact he
      · rw [Array.getElem?_push, if_neg (by omega)]; exact hd
      · rw [hf nd hn, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm (hne y hy hp nd hn))]
      · rw [Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm (hne y hy hp nd hn))]
        exact hlt nd hn hp' hr

theorem placesAll_final {S : Prog} {d : ℕ} {seen : Std.HashSet ℕ} {todo : Array ℕ} {at_ : ℕ}
    {toB : Std.HashMap ℕ ℕ} {w : Std.HashMap ℕ Bool} {defs : Array Defn} {toL : Std.HashMap ℕ ℕ}
    (hk : WalkInv S d seen todo at_ toB) (hge : ¬ at_ < todo.size)
    (hW : WordyAll S (seen.toArray.qsort fun x1 x2 => decide (x1 < x2)).toList w)
    (hL : LiftAll S toB w (seen.toArray.qsort fun x1 x2 => decide (x1 < x2)).toList defs toL ∧
      ((seen.toArray.qsort fun x1 x2 => decide (x1 < x2)).toList ++ []).Nodup) :
    (∀ x, Reach S d x → isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = true →
        S.bodyOf? (S.defs[x]!).key = some (toL[x]?.getD x)) ∧
      (∀ x, Reach S d x → (S.defs[x]!).prim = true → isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = false →
        toL[x]?.getD x = x) ∧
      (∀ x, Reach S d x → (S.defs[x]!).prim = false → ¬ ReachesWord S x → toL[x]?.getD x = x) ∧
      ∀ x, Reach S d x → (S.defs[x]!).prim = false → ReachesWord S x → S.defs.size ≤ toL[x]?.getD x ∧
        ∃ fnOf primOf gatesOf, defs[toL[x]?.getD x]? = some (Flock.Lift.liftDef (S.defs[x]!) fnOf primOf gatesOf) ∧
          (∀ nd ∈ (S.defs[x]!).body.nodes, fnOf nd.fn = toL[nd.fn]?.getD nd.fn) ∧
          ∀ nd ∈ (S.defs[x]!).body.nodes, (S.defs[nd.fn]!).prim = false → ReachesWord S nd.fn →
            toL[nd.fn]?.getD nd.fn < toL[x]?.getD x := by
  unfold LiftAll at hL
  unfold WordyAll at hW
  obtain ⟨⟨_, h2, h3, h4⟩, _⟩ := hL
  refine ⟨fun x hr hw => ?_, fun x _ hp hw => ?_, fun x hr hp hn => ?_, fun x hr hp hrw => ?_⟩
  · obtain ⟨bi, hb, hbo⟩ := (hk.2.2.2.2.1 x (walk_reach hk hge hr)).2 hw
    have hpr : (S.defs[x]!).prim = true := by simp only [isWordPrim, Bool.and_eq_true] at hw; exact hw.1.1
    rw [h2 x hpr, hb]; exact hbo
  · rw [h2 x hp]
    cases hq : toB[x]? with
    | none => rfl
    | some bi => rw [hk.2.2.2.2.2 x bi hq] at hw; cases hw
  · cases hq : toL[x]? with
    | none => rfl
    | some e =>
      obtain ⟨_, hw⟩ := h3 x hp (by simp [hq])
      exact absurd ((hW.2.1 x true hw).1 rfl) hn
  · have hm := mem_order (walk_reach hk hge hr)
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 ((hW.1 x).2 hm)
    have hbt : b = true := (hW.2.1 x b hb).2 hrw
    subst hbt
    obtain ⟨e, he, hse, fnOf, pO, gO, hd, hf, hlt⟩ := h4 x hm hp hb
    rw [he]; exact ⟨hse, fnOf, pO, gO, hd, hf, hlt⟩

theorem closure_placesAll (S : Prog) (d : ℕ) :
    ⦃⌜True⌝⦄ Flock.Lift.closure S #[d] ⦃⇓? r => ⌜
      (∀ x, Reach S d x → isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = true → S.bodyOf? (S.defs[x]!).key = some (r.2 x)) ∧
      (∀ x, Reach S d x → (S.defs[x]!).prim = true → isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = false →
        r.2 x = x) ∧
      (∀ x, Reach S d x → (S.defs[x]!).prim = false → ¬ ReachesWord S x → r.2 x = x) ∧
      ∀ x, Reach S d x → (S.defs[x]!).prim = false → ReachesWord S x → S.defs.size ≤ r.2 x ∧
        ∃ fnOf primOf gatesOf, r.1.defs[r.2 x]? = some (Flock.Lift.liftDef (S.defs[x]!) fnOf primOf gatesOf) ∧
          (∀ nd ∈ (S.defs[x]!).body.nodes, fnOf nd.fn = r.2 nd.fn) ∧
          ∀ nd ∈ (S.defs[x]!).body.nodes, (S.defs[nd.fn]!).prim = false → ReachesWord S nd.fn → r.2 nd.fn < r.2 x⌝⦄ := by
  unfold Flock.Lift.closure
  mvcgen [hashMap_forIn_spec, exc_throw_spec, -Spec.throw_Except] invariants
  · ⇓? ⟨_, seen, todo, at_, toB⟩ => ⌜WalkInv S d seen todo at_ toB⌝
  · ⇓? ⟨c, wordy⟩ => ⌜WordyAll S c.prefix wordy⌝
  · by exact ⇓? ⟨c, w⟩ => ⌜(w = true ↔ ∃ x ∈ c.prefix, ‹Std.HashMap ℕ Bool›[x.fn]? = some true) ∧
      ∀ x ∈ c.prefix, (‹Std.HashMap ℕ Bool›[x.fn]?).isSome = true⌝
  · by exact ⇓? ⟨c, defs, index, toL⟩ => ⌜LiftAll S ‹Std.HashSet ℕ × Array ℕ × ℕ × Std.HashMap ℕ ℕ›.2.2.2 ‹Std.HashMap ℕ Bool› c.prefix defs toL ∧
      (c.prefix ++ c.suffix).Nodup⌝
  all_goals first
    | trivial
    | exact walk_init S d
    | (exact walk_seen (by assumption) (by assumption) (by assumption))
    | (exact walk_word (by assumption) (by assumption) (by assumption) (by assumption) (by assumption)
        (by assumption))
    | (exact walk_comp (by assumption) (by assumption) (by assumption) (by assumption))
    | (exact walk_prim (by assumption) (by assumption) (by assumption) (by assumption) (by assumption))
    | exact wordyAll_nil S
    | exact wordyAll_inner_nil
    | (exact wordyAll_prim (by assumption) (by assumption))
    | (exact wordyAll_inner_step (by assumption) (by assumption))
    | (exact wordyAll_comp (by assumption) (by assumption) (by assumption))
    | (exact liftAll_nil _ (by assumption))
    | (exact liftAll_skip (by assumption) (by assumption))
    | (exact liftAll_place _ _ (by assumption) (by assumption) (by assumption) (by assumption))
    | (exact placesAll_final (by assumption) (by assumption) (by assumption) (by assumption))

theorem prim_lt {P : Prog} {i : ℕ} (h : (P.defs[i]!).prim = true) : i < P.defs.size := by
  refine Classical.byContradiction fun hn => ?_
  rw [getElem!_neg P.defs i hn] at h
  cases h

theorem wordPrim_lt {P : Prog} {i : ℕ} (h : isWordPrim (P.defs[i]!).prim (P.defs[i]!).key = true) : i < P.defs.size :=
  prim_lt (by simp only [isWordPrim, Bool.and_eq_true] at h; exact h.1.1)

theorem reachesWord_lt {P : Prog} {x : ℕ} (h : ReachesWord P x) : x < P.defs.size := by
  cases h with
  | word hw => exact wordPrim_lt hw
  | comp _ hmem _ =>
    refine Classical.byContradiction fun hn => ?_
    rw [getElem!_neg P.defs _ hn] at hmem
    simp [show (default : Defn).body.nodes = #[] from rfl] at hmem

/-- `ReachesWord`, read in any `S` that keeps `P`'s Definitions. -/
theorem reachesWord_restrict {P S : Prog} (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) {y : ℕ}
    (h : ReachesWord P y) : ReachesWord S y := by
  induction h with
  | word hw => exact .word (by rw [hpre _ (wordPrim_lt hw)]; exact hw)
  | comp hp hmem hr ih =>
    have hy := reachesWord_lt (.comp hp hmem hr)
    exact .comp (by rw [hpre _ hy]; exact hp) (by rw [hpre _ hy]; exact hmem) ih

/-- `ReachesWord` in `S` of a template of `ScanNested`, read back in `P`, whose Definitions `S` keeps. -/
theorem reachesWord_unrestrict {P S : Prog} (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) {y : ℕ}
    (h : ScanNested P y) : y < P.defs.size → ReachesWord S y → ReachesWord P y := by
  induction h with
  | @mk d hp hlt _ _ _ ih =>
    intro hy hr
    cases hr with
    | word hw => exact .word (by rw [← hpre d hy]; exact hw)
    | @comp _ nd hpS hmem hr' =>
      rw [hpre d hy] at hmem
      cases hc : (P.defs[nd.fn]!).prim with
      | false => exact .comp hp hmem (ih nd hmem hc (by have := hlt nd hmem hc; omega) hr')
      | true =>
        have hl := prim_lt hc
        cases hr' with
        | word hw => exact .comp hp hmem (.word (by rw [← hpre _ hl]; exact hw))
        | comp hpS' _ _ => rw [hpre _ hl, hc] at hpS'; cases hpS'

/-- **`closure_placesAll` through `lift`**: the lift of `d` returns where its closure places `d`. -/
theorem liftPlacesAll {S L : Prog} {d e : ℕ} (h : Flock.Lift.lift S d = .ok (L, e)) :
    ∃ pl : ℕ → ℕ, e = pl d ∧
      (∀ x, Reach S d x → isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = true → S.bodyOf? (S.defs[x]!).key = some (pl x)) ∧
      (∀ x, Reach S d x → (S.defs[x]!).prim = true → isWordPrim (S.defs[x]!).prim (S.defs[x]!).key = false →
        pl x = x) ∧
      (∀ x, Reach S d x → (S.defs[x]!).prim = false → ¬ ReachesWord S x → pl x = x) ∧
      ∀ x, Reach S d x → (S.defs[x]!).prim = false → ReachesWord S x → S.defs.size ≤ pl x ∧
        ∃ fnOf primOf gatesOf, L.defs[pl x]? = some (Flock.Lift.liftDef (S.defs[x]!) fnOf primOf gatesOf) ∧
          (∀ nd ∈ (S.defs[x]!).body.nodes, fnOf nd.fn = pl nd.fn) ∧
          ∀ nd ∈ (S.defs[x]!).body.nodes, (S.defs[nd.fn]!).prim = false → ReachesWord S nd.fn → pl nd.fn < pl x := by
  unfold Flock.Lift.lift at h
  rcases hc : Flock.Lift.closure S #[d] with _ | ⟨Q, fnOf⟩
  · rw [hc] at h; simp [bind, Except.bind] at h
  rw [hc] at h
  have h' : (Q, fnOf d) = (L, e) := by simpa [bind, Except.bind, pure, Except.pure] using h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj h'
  have hpost := exc_of_triple (closure_placesAll S d) hc
  exact ⟨fnOf, rfl, hpost⟩

end Flock.Chain
