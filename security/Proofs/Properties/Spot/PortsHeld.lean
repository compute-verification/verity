module

public import Properties.Spot.PortsHeld
import all Lean.Data.Json.Basic
import Proofs.Properties.Verity.Sampled.MemberForm.Passes
import Proofs.Properties.Verity.Sampled.Window.Members
import Proofs.Flock.SpotFlock.OwnReads
import Proofs.Flock.SpotFlock.ParamsWf
import Proofs.Flock.SpotFlock.RunsAs.Outcome
import Proofs.Flock.Claims.JsonNorm

/-! # Proof that the per-member form holds every port of every member's statement (`Properties.Spot.MembersPortsHeld`) -/

@[expose] public section

open Lean (Json)
open Spot
open Claims.Json (Wf)
open FlockSoundness.Refine (ebind_ok)

namespace Properties.Spot.PortsHeldProof

/-- The slice and header after dropping the ports `D` in order (`OneStageZk.dropPort`). -/
def dropAll (s : Json × Json) (D : List String) : Json × Json :=
  D.foldl (fun c q => OneStageZk.dropPort c.1 c.2 q) s

theorem dropAll_cons (s : Json × Json) (q : String) (D : List String) :
    dropAll s (q :: D) = dropAll (OneStageZk.dropPort s.1 s.2 q) D := rfl

theorem dropAll_append (s : Json × Json) (D E : List String) : dropAll s (D ++ E) = dropAll (dropAll s D) E := by
  unfold dropAll; rw [List.foldl_append]

theorem forIn_drops {α : Type} (P : String → Prop) (f : α → Json × Json → Except String (ForInStep (Json × Json)))
    : ∀ (xs : List α), (∀ a ∈ xs, ∀ s s', f a s = .ok s' →
        s' = .yield s ∨ ∃ q, P q ∧ s' = .yield (OneStageZk.dropPort s.1 s.2 q)) →
      ∀ {s0 s : Json × Json}, forIn xs s0 f = .ok s → ∃ D, (∀ q ∈ D, P q) ∧ s = dropAll s0 D
  | [], _, s0, s, h => by
    rw [List.forIn_nil] at h
    have e : s0 = s := by simpa [pure, Except.pure] using h
    exact ⟨[], by simp, by rw [← e]; rfl⟩
  | a :: as, hf, s0, s, h => by
    rw [List.forIn_cons] at h
    cases hfa : f a s0 with
    | error e => simp [hfa, bind, Except.bind] at h
    | ok s' =>
      simp only [hfa, bind, Except.bind] at h
      rcases hf a List.mem_cons_self s0 s' hfa with rfl | ⟨q, hq, rfl⟩
      · obtain ⟨D, hD, rfl⟩ := forIn_drops P f as (fun b hb => hf b (List.mem_cons_of_mem _ hb)) h
        exact ⟨D, hD, rfl⟩
      · obtain ⟨D, hD, rfl⟩ := forIn_drops P f as (fun b hb => hf b (List.mem_cons_of_mem _ hb)) h
        refine ⟨q :: D, fun x hx => ?_, rfl⟩
        rcases List.mem_cons.1 hx with rfl | hx
        · exact hq
        · exact hD x hx

theorem forIn_drops_arr {α : Type} (P : String → Prop)
    (f : α → Json × Json → Except String (ForInStep (Json × Json))) (xs : Array α)
    (hf : ∀ a ∈ xs, ∀ s s', f a s = .ok s' →
      s' = .yield s ∨ ∃ q, P q ∧ s' = .yield (OneStageZk.dropPort s.1 s.2 q))
    {s0 s : Json × Json} (h : forIn xs s0 f = .ok s) : ∃ D, (∀ q ∈ D, P q) ∧ s = dropAll s0 D := by
  rw [← Array.forIn_toList] at h
  exact forIn_drops P f xs.toList (fun a ha => hf a (Array.mem_def.2 ha)) h

/-! ## JSON objects -/

theorem getD_obj (m : Std.TreeMap.Raw String Json compare) (k : String) : getD (.obj m) k = (m[k]?).getD .null := by
  unfold getD
  rw [Properties.Challenger.CanonLine.field?_obj, Std.TreeMap.Raw.get?_eq_getElem?]

theorem field?_obj' (m : Std.TreeMap.Raw String Json compare) (k : String) : field? (.obj m) k = m[k]? := by
  rw [Properties.Challenger.CanonLine.field?_obj, Std.TreeMap.Raw.get?_eq_getElem?]

theorem wf_getD {j : Json} (hj : Wf j) (k : String) : Wf (getD j k) := by
  cases j with
  | obj m =>
    rw [getD_obj]
    obtain ⟨-, hv⟩ := SpotFlock.ReadsJson.wf_obj hj
    cases hk : m[k]? with
    | none => exact .null
    | some v => exact hv k v hk
  | _ => exact .null

theorem mem_entries_iff {j : Json} (hj : Wf j) {k : String} {v : Json} :
    (k, v) ∈ entries j ↔ field? j k = some v := by
  cases j with
  | obj m =>
    obtain ⟨hm, -⟩ := SpotFlock.ReadsJson.wf_obj hj
    rw [field?_obj']
    exact Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some hm
  | _ => simp [entries, field?, Json.getObjVal?, Except.toOption, throw, throwThe, MonadExceptOf.throw]

theorem isObj_of_mem_entries {j : Json} {k : String} {v : Json} (h : (k, v) ∈ entries j) :
    ∃ m, j = .obj m := by
  cases j with
  | obj m => exact ⟨m, rfl⟩
  | _ => simp [entries] at h

theorem isObj_of_mem_entries_getD {j : Json} {key k : String} {v : Json} (h : (k, v) ∈ entries (getD j key)) :
    ∃ m, j = .obj m := by
  cases j with
  | obj m => exact ⟨m, rfl⟩
  | _ =>
    have : getD _ key = Json.null := rfl
    rw [this] at h
    simp [entries] at h

theorem distinct_drop {j : Json} (hj : Wf j) (q : String) :
    FlockComplete.JsonRT.Distinct ((entries j).filter (·.1 != q)).toList := by
  rw [Array.toList_filter]
  cases j with
  | obj m =>
    obtain ⟨hm, -⟩ := SpotFlock.ReadsJson.wf_obj hj
    exact List.Pairwise.sublist List.filter_sublist (SpotFlock.ParamsWf.distinct_toList hm)
  | _ => exact List.Pairwise.nil

theorem field?_drop {j : Json} (hj : Wf j) {q k : String} (hk : k ≠ q) :
    field? (Json.mkObj ((entries j).filter (·.1 != q)).toList) k = field? j k := by
  have hd := distinct_drop hj q
  cases h : field? j k with
  | some v =>
    apply SpotFlock.ReadsJson.field?_mkObj hd
    exact Array.mem_toList_iff.2 (Array.mem_filter.2 ⟨(mem_entries_iff hj).2 h, by simpa using hk⟩)
  | none =>
    cases h' : field? (Json.mkObj ((entries j).filter (·.1 != q)).toList) k with
    | none => rfl
    | some v =>
      have hm := (Array.mem_filter.1 (Array.mem_toList_iff.1 (SpotFlock.ReadsJson.mem_of_field?_mkObj hd h'))).1
      rw [mem_entries_iff hj, h] at hm
      cases hm

theorem getD_drop {j : Json} (hj : Wf j) {q k : String} (hk : k ≠ q) :
    getD (Json.mkObj ((entries j).filter (·.1 != q)).toList) k = getD j k := by
  unfold getD; rw [field?_drop hj hk]

theorem wf_drop {j : Json} (hj : Wf j) (q : String) : Wf (Json.mkObj ((entries j).filter (·.1 != q)).toList) := by
  refine SpotFlock.ReadsJson.wf_mkObj (distinct_drop hj q) fun kv hkv => ?_
  have hm := (Array.mem_filter.1 (Array.mem_toList_iff.1 hkv)).1
  rw [mem_entries_iff hj] at hm
  have : getD j kv.1 = kv.2 := by unfold getD; rw [hm]; rfl
  rw [← this]
  exact wf_getD hj _

theorem setObjVal_eq_setKey (m : Std.TreeMap.Raw String Json compare) (k : String) (v : Json) :
    (Json.obj m).setObjVal! k v = SecondStage.setKey (.obj m) k v := rfl

theorem getD_setKey_ne {j v : Json} (hj : Wf j) {k k' : String} (hk : k ≠ k') :
    getD (SecondStage.setKey j k v) k' = getD j k' := by
  unfold getD; rw [SpotFlock.ReadsJson.field?_setKey_ne hj hk]

theorem getD_setKey_self {m : Std.TreeMap.Raw String Json compare} (hj : Wf (.obj m)) (k : String) (v : Json) :
    getD (SecondStage.setKey (.obj m) k v) k = v := by
  unfold getD; rw [SpotFlock.ReadsJson.field?_setKey_self hj]; rfl

/-! ## `Record.byPort`: the last root of each port -/

abbrev Entries := Array (String × Json)

theorem find_push_filter (acc : Entries) (p' p : String) (r : Json) :
    ((acc.filter fun (q : String × Json) => q.1 != p').push (p', r)).find? (·.1 == p) =
      if p' = p then some (p', r) else acc.find? (·.1 == p) := by
  rw [Array.find?_push, Array.find?_filter]
  by_cases h : p' = p
  · subst h
    have : acc.find? (fun a => decide ((a.1 != p') = true ∧ (a.1 == p') = true)) = none := by
      rw [Array.find?_eq_none]; intro a _; simp
    simp [this]
  · have : acc.find? (fun a => decide ((a.1 != p') = true ∧ (a.1 == p) = true)) = acc.find? (·.1 == p) := by
      congr 1; funext a
      by_cases ha : a.1 = p
      · simp [ha, Ne.symm h]
      · simp [ha]
    simp [this, h]

/-- What a step of `Record.byPort`'s fold does: a root naming a port replaces that port's entry, any other is skipped. -/
def StepSpec (F : Entries → Json → Entries) : Prop :=
  ∀ acc r, (∀ p, getD r "port" = .str p → F acc r = (acc.filter fun (q : String × Json) => q.1 != p).push (p, r)) ∧
    ((∀ p, getD r "port" ≠ .str p) → F acc r = acc)

theorem byPort_spec :
    ∃ F, StepSpec F ∧
      ∀ rec, Record.byPort rec = (match getD rec "roots" with | .arr rs => rs | _ => #[]).foldl F #[] := by
  refine ⟨fun (acc : Entries) r => match getD r "port" with
    | .str p => (acc.filter fun (q : String × Json) => q.1 != p).push (p, r)
    | _ => acc, fun acc r => ⟨fun p hp => ?_, fun hn => ?_⟩, fun rec => ?_⟩
  · dsimp only; rw [hp]
  · dsimp only
    split
    · next p hp => exact absurd hp (hn p)
    · rfl
  · unfold Record.byPort
    generalize getD rec "roots" = j
    cases j <;> rfl

theorem find_step {F : Entries → Json → Entries} (hF : StepSpec F) (acc : Entries) (r : Json) (p : String) :
    (F acc r).find? (·.1 == p) =
      if getD r "port" = .str p then some (p, r) else acc.find? (·.1 == p) := by
  by_cases hs : ∃ p', getD r "port" = .str p'
  · obtain ⟨p', hp'⟩ := hs
    rw [(hF acc r).1 p' hp', find_push_filter, hp']
    by_cases h : p' = p
    · subst h; simp
    · simp [h]
  · have hn : ∀ p', getD r "port" ≠ .str p' := fun p' h => hs ⟨p', h⟩
    rw [(hF acc r).2 hn, if_neg (hn p)]

def Good (acc : Entries) : Prop := ∀ e ∈ acc, acc.find? (·.1 == e.1) = some e

theorem good_fold {F : Entries → Json → Entries} (hF : StepSpec F) :
    ∀ (l : List Json) (acc : Entries), Good acc → Good (l.foldl F acc)
  | [], _, h => h
  | r :: l, acc, h => by
    apply good_fold hF l
    by_cases hs : ∃ p', getD r "port" = .str p'
    · obtain ⟨p', hp'⟩ := hs
      rw [(hF acc r).1 p' hp']
      intro e he
      rw [find_push_filter]
      rcases Array.mem_push.1 he with he | rfl
      · obtain ⟨heacc, hne⟩ := Array.mem_filter.1 he
        have hne' : p' ≠ e.1 := fun h' => by simp [h'] at hne
        rw [if_neg hne']
        exact h e heacc
      · simp
    · have hn : ∀ p', getD r "port" ≠ .str p' := fun p' h => hs ⟨p', h⟩
      rw [(hF acc r).2 hn]
      exact h

theorem byPort_good (rec : Json) : Good (Record.byPort rec) := by
  obtain ⟨F, hF, he⟩ := byPort_spec
  rw [he, ← Array.foldl_toList]
  exact good_fold hF _ _ (by intro e he; simp at he)

theorem fold_find_filter {F : Entries → Json → Entries} (hF : StepSpec F) {p q : String} (hpq : p ≠ q) :
    ∀ (l : List Json) (acc1 acc2 : Entries), acc1.find? (·.1 == p) = acc2.find? (·.1 == p) →
      ((l.filter fun r => getD r "port" != .str q).foldl F acc1).find? (·.1 == p) = (l.foldl F acc2).find? (·.1 == p)
  | [], _, _, h => h
  | r :: l, acc1, acc2, h => by
    rw [List.filter_cons]
    by_cases hr : getD r "port" = .str q
    · have : (getD r "port" != Json.str q) = false := by simp [hr]
      rw [this]
      apply fold_find_filter hF hpq l acc1 (F acc2 r)
      rw [find_step hF, hr, if_neg (fun h' => hpq (Json.str.inj h').symm)]
      exact h
    · have : (getD r "port" != Json.str q) = true := by simp [hr]
      rw [this]
      apply fold_find_filter hF hpq l (F acc1 r) (F acc2 r)
      rw [find_step hF, find_step hF, h]

/-! ## What `dropPort` keeps -/

/-- The header `hc` after the drops `D`: an object whose `frame_v3` is one, with every port outside `D` at the original
roots `R0` and domains `Dom0`. -/
def HInv (R0 Dom0 : Json) (D : List String) (hc : Json) : Prop :=
  Wf hc ∧ (∃ m, hc = .obj m) ∧ (∃ mf, getD hc "frame_v3" = .obj mf) ∧
    (∀ k, k ∉ D → field? (getD (getD hc "frame_v3") "roots") k = field? R0 k) ∧
    (∀ k, k ∉ D → getD (getD (getD hc "frame_v3") "domain_ids") k = getD Dom0 k)

/-- The slice `rc` after the drops `D`: an object whose `byPort` finds every port outside `D` as the slice `S`'s does. -/
def RInv (S : Json) (D : List String) (rc : Json) : Prop :=
  (∃ m : Std.TreeMap.Raw String Json compare, rc = .obj m ∧ m.WF) ∧
    ∀ k, k ∉ D → (Record.byPort rc).find? (·.1 == k) = (Record.byPort S).find? (·.1 == k)

theorem hinv_mono {R0 Dom0 : Json} {D D' : List String} {hc : Json} (hD : ∀ k, k ∉ D' → k ∉ D)
    (h : HInv R0 Dom0 D hc) : HInv R0 Dom0 D' hc :=
  ⟨h.1, h.2.1, h.2.2.1, fun k hk => h.2.2.2.1 k (hD k hk), fun k hk => h.2.2.2.2 k (hD k hk)⟩

theorem rinv_mono {S : Json} {D D' : List String} {rc : Json} (hD : ∀ k, k ∉ D' → k ∉ D)
    (h : RInv S D rc) : RInv S D' rc :=
  ⟨h.1, fun k hk => h.2 k (hD k hk)⟩

theorem hinv_step {R0 Dom0 : Json} {D : List String} {rc hc : Json} (q : String) (h : HInv R0 Dom0 D hc) :
    HInv R0 Dom0 (q :: D) (OneStageZk.dropPort rc hc q).2 := by
  obtain ⟨hw, ⟨m, rfl⟩, ⟨mf, hmf⟩, hR, hDm⟩ := h
  have hfv : Wf (.obj mf) := hmf ▸ wf_getD hw "frame_v3"
  have hRw : Wf (getD (.obj mf) "roots") := wf_getD hfv _
  have hDw : Wf (getD (.obj mf) "domain_ids") := wf_getD hfv _
  unfold OneStageZk.dropPort
  dsimp only
  rw [hmf, SpotFlock.RunsAs.Outcome.setObjVal_obj, setObjVal_eq_setKey, setObjVal_eq_setKey]
  have hfv1 : Wf (.obj (mf.insert "roots" (Json.mkObj ((entries (getD (.obj mf) "roots")).filter (·.1 != q)).toList))) :=
    SpotFlock.ReadsJson.wf_setKey hfv (wf_drop hRw q) "roots"
  have hfv2 := SpotFlock.ReadsJson.wf_setKey hfv1 (wf_drop hDw q) "domain_ids"
  have hself := getD_setKey_self hw "frame_v3"
    (SecondStage.setKey (.obj (mf.insert "roots" (Json.mkObj ((entries (getD (.obj mf) "roots")).filter (·.1 != q)).toList)))
      "domain_ids" (Json.mkObj ((entries (getD (.obj mf) "domain_ids")).filter (·.1 != q)).toList))
  refine ⟨SpotFlock.ReadsJson.wf_setKey hw hfv2 _, ⟨_, rfl⟩, ⟨_, ?_⟩, fun k hk => ?_, fun k hk => ?_⟩
  · rw [hself]; rfl
  · have hkq : k ≠ q := fun e => hk (e ▸ List.mem_cons_self)
    have hkD : k ∉ D := fun e => hk (List.mem_cons_of_mem _ e)
    rw [hself, getD_setKey_ne hfv1 (by decide : "domain_ids" ≠ "roots")]
    have h1 : getD (.obj (mf.insert "roots" (Json.mkObj ((entries (getD (.obj mf) "roots")).filter (·.1 != q)).toList)))
        "roots" = Json.mkObj ((entries (getD (.obj mf) "roots")).filter (·.1 != q)).toList :=
      getD_setKey_self hfv "roots" _
    rw [h1, field?_drop hRw hkq, ← hR k hkD, hmf]
  · have hkq : k ≠ q := fun e => hk (e ▸ List.mem_cons_self)
    have hkD : k ∉ D := fun e => hk (List.mem_cons_of_mem _ e)
    rw [hself, getD_setKey_self hfv1, getD_drop hDw hkq, ← hDm k hkD, hmf]

theorem rinv_step {S : Json} {D : List String} {rc hc : Json} (q : String) (h : RInv S D rc) :
    RInv S (q :: D) (OneStageZk.dropPort rc hc q).1 := by
  obtain ⟨⟨m, rfl, hm⟩, hf⟩ := h
  unfold OneStageZk.dropPort
  dsimp only
  rw [SpotFlock.RunsAs.Outcome.setObjVal_obj]
  refine ⟨⟨_, rfl, Std.TreeMap.Raw.WF.insert hm⟩, fun k hk => ?_⟩
  have hkq : k ≠ q := fun e => hk (e ▸ List.mem_cons_self)
  have hkD : k ∉ D := fun e => hk (List.mem_cons_of_mem _ e)
  rw [← hf k hkD]
  obtain ⟨F, hF, he⟩ := byPort_spec
  rw [he, he, getD_obj (m.insert _ _), Std.TreeMap.Raw.getElem?_insert_self hm]
  dsimp only [Option.getD]
  generalize getD (.obj m) "roots" = j
  cases j with
  | arr rs =>
    dsimp only
    rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_filter]
    exact fold_find_filter hF hkq _ _ _ rfl
  | _ => rfl

theorem dropAll_inv {R0 Dom0 S : Json} : ∀ (D : List String) (s : Json × Json) (E : List String),
    HInv R0 Dom0 E s.2 → RInv S E s.1 → HInv R0 Dom0 (D ++ E) (dropAll s D).2 ∧ RInv S (D ++ E) (dropAll s D).1
  | [], _, _, h1, h2 => ⟨h1, h2⟩
  | q :: D, s, E, h1, h2 => by
    obtain ⟨g1, g2⟩ := dropAll_inv D (OneStageZk.dropPort s.1 s.2 q) (q :: E)
      (hinv_step (rc := s.1) q h1) (rinv_step (hc := s.2) q h2)
    have hm : ∀ k, k ∉ q :: D ++ E → k ∉ D ++ q :: E := by
      intro k hk h
      apply hk
      simp only [List.mem_cons, List.mem_append] at h ⊢
      tauto
    exact ⟨hinv_mono hm g1, rinv_mono hm g2⟩

/-! ## What `headerHeld` and `headerHeldAt` accept -/

theorem sameDict_mem {a b : Array (String × Json)} (h : Record.sameDict a b = true) :
    ∀ e ∈ a, ∃ e' ∈ b, e.1 = e'.1 ∧ jeq e.2 e'.2 = true := by
  unfold Record.sameDict at h
  rw [Bool.and_eq_true, Array.all_eq_true'] at h
  intro ⟨k, v⟩ he
  have := h.2 (k, v) he
  dsimp only at this
  rw [Array.any_eq_true'] at this
  obtain ⟨⟨k', v'⟩, hm, hb⟩ := this
  dsimp only at hb
  rw [Bool.and_eq_true, beq_iff_eq] at hb
  exact ⟨(k', v'), hm, hb.1, hb.2⟩

theorem matchesPublic_ok {reg : Array (String × Json)} {rec s roots domains pop : Json}
    (h : (Record.matchesPublicBy reg rec s roots domains pop).codes.isEmpty = true) :
    (∀ pr ∈ reg, ∃ e' ∈ entries roots, pr.1 = e'.1 ∧ jeq (getD pr.2 "root") e'.2 = true) ∧
      (∀ pr ∈ reg, ∃ pr' ∈ reg, pr.1 = pr'.1 ∧ jeq (getD domains pr.1) (getD pr'.2 "domain") = true) := by
  unfold Record.matchesPublicBy at h
  simp only [Id.run] at h
  repeat' split at h
  all_goals first
    | (simp only [pure, _root_.Verity.Sampled.fail_codes] at h; cases h)
    | skip
  rename_i n1 n2 n3 n4
  have h2 := sameDict_mem (_root_.Verity.Sampled.PerMember.of_not_bnot n2)
  have h3 := sameDict_mem (_root_.Verity.Sampled.PerMember.of_not_bnot n3)
  refine ⟨fun ⟨p, r⟩ hpr => ?_, fun ⟨p, r⟩ hpr => ?_⟩
  · obtain ⟨e', he', h1, h2⟩ := h2 _ (Array.mem_map_of_mem hpr)
    exact ⟨e', he', h1, h2⟩
  · obtain ⟨e', he', h1, h2⟩ := h3 _ (Array.mem_map_of_mem hpr)
    obtain ⟨⟨p', r'⟩, hmem, rfl⟩ := Array.mem_map.1 he'
    exact ⟨(p', r'), hmem, h1, h2⟩

theorem headerHeld_ok {rec h : Json} (hh : OneStageZk.headerHeld rec h = .ok ()) :
    OneStageZk.unwired rec (getD (getD h "frame_v3") "roots") = none ∧
      (Record.matchesPublic rec (getD h "scheme") (getD (getD h "frame_v3") "roots")
        (getD (getD h "frame_v3") "domain_ids") (getD h "instances")).codes.isEmpty = true := by
  unfold OneStageZk.headerHeld at hh
  dsimp only at hh
  split at hh
  · first | (simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at hh; done) | cases hh
  · rename_i hu
    split at hh
    · exact ⟨hu, by assumption⟩
    · first | (simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at hh; done) | cases hh

/-- A port `headerHeldAt` drops before its `headerHeld`: one a read of the member's names at its root's tree, or one
the slice holds no root of that a registered read holds. -/
def Dropped (reads : Array (String × String)) (held : Array Flock.Registered.Port) (slice : Json) (q : String) : Prop :=
  (∃ r ∈ (match getD slice "roots" with | .arr rs => rs | _ => #[]), getD r "port" = .str q ∧
    ∃ t, getD r "tree" = .str t ∧ (q, t) ∈ reads) ∨
  ((∀ x ∈ Record.byPort slice, x.1 ≠ q) ∧ ∃ v, (q, v) ∈ reads ∧ OneStageZk.registeredAt held q v = true)

theorem headerHeldAt_drops {reads : Array (String × String)} {held : Array Flock.Registered.Port} {rec h : Json}
    (hh : OneStageZk.headerHeldAt reads held rec h = .ok ()) :
    ∃ D, (∀ q ∈ D, Dropped reads held rec q) ∧
      OneStageZk.headerHeld (dropAll (rec, h) D).1 (dropAll (rec, h) D).2 = .ok () := by
  unfold OneStageZk.headerHeldAt at hh
  dsimp only at hh
  split at hh
  · first
      | (simp only [throw, throwThe, MonadExceptOf.throw, bind, Except.bind, reduceCtorEq] at hh; done)
      | cases hh
  obtain ⟨s1, h1, hh⟩ := ebind_ok hh
  obtain ⟨s2, h2, hh⟩ := ebind_ok hh
  obtain ⟨D1, hD1, rfl⟩ := forIn_drops_arr (Dropped reads held rec) _ _ (by
    intro ⟨p, t, r⟩ ha s s' hf
    dsimp only at hf
    split at hf
    · rename_i hc
      cases hf
      refine Or.inr ⟨p, Or.inl ?_, rfl⟩
      obtain ⟨r0, hr0, hfm⟩ := Array.mem_filterMap.1 ha
      split at hfm
      · rename_i p' t' hp ht
        cases hfm
        exact ⟨r0, hr0, hp, t, ht, Array.contains_iff_mem.1 hc⟩
      · cases hfm
    · split at hf
      · first
          | (simp only [throw, throwThe, MonadExceptOf.throw, bind, Except.bind, reduceCtorEq] at hf; done)
          | cases hf
      · cases hf
        exact Or.inl rfl) h1
  obtain ⟨D2, hD2, rfl⟩ := forIn_drops_arr (Dropped reads held rec) _ _ (by
    intro ⟨p, v⟩ ha s s' hf
    dsimp only at hf
    split at hf
    · rename_i hc
      dsimp only at hf
      split at hf
      · first
          | (simp only [throw, throwThe, MonadExceptOf.throw, bind, Except.bind, reduceCtorEq] at hf; done)
          | cases hf
      · rename_i hr
        cases hf
        refine Or.inr ⟨p, Or.inr ⟨fun x hx hxp => ?_, v, ha, _root_.Verity.Sampled.PerMember.of_not_bnot hr⟩, rfl⟩
        have hany : (Record.byPort rec).any (·.1 == p) = true := Array.any_eq_true'.2 ⟨x, hx, by simp [hxp]⟩
        rw [hany] at hc
        simp at hc
    · cases hf
      exact Or.inl rfl) h2
  refine ⟨D1 ++ D2, fun q hq => ?_, by rw [dropAll_append]; exact hh⟩
  rcases List.mem_append.1 hq with hq | hq
  · exact hD1 q hq
  · exact hD2 q hq

/-! ## Every port of the header is held -/

theorem getD_nonobj {j : Json} (hj : ∀ m, j ≠ .obj m) (k : String) : getD j k = .null := by
  cases j with
  | obj m => exact absurd rfl (hj m)
  | _ => rfl

theorem isObj_of_getD {j : Json} {k : String} {m : Std.TreeMap.Raw String Json compare} (h : getD j k = .obj m) :
    ∃ m', j = .obj m' := by
  by_cases hj : ∃ m', j = .obj m'
  · exact hj
  · rw [getD_nonobj (fun m' e => hj ⟨m', e⟩)] at h
    cases h

theorem isObj_of_mem {j : Json} {k : String} {v : Json} (h : (k, v) ∈ entries j) :
    ∃ m : Std.TreeMap.Raw String Json compare, j = .obj m := by
  cases j with
  | obj m => exact ⟨m, rfl⟩
  | _ => simp [entries] at h

theorem portsHeld_of_headerHeldAt {reads : Array (String × String)} {held : Array Flock.Registered.Port}
    {slice hd : Json} (hw : Wf hd) (hs : ∃ m : Std.TreeMap.Raw String Json compare, slice = .obj m ∧ m.WF)
    (hh : OneStageZk.headerHeldAt reads held slice hd = .ok ()) : PortsHeld reads held slice hd := by
  obtain ⟨D, hD, hfin⟩ := headerHeldAt_drops hh
  intro ⟨p, x⟩ hpr
  by_cases hpD : p ∈ D
  · rcases hD p hpD with ha | hb
    · exact Or.inl ha
    · exact Or.inr (Or.inl hb)
  refine Or.inr (Or.inr ?_)
  obtain ⟨mr, hmr⟩ := isObj_of_mem hpr
  obtain ⟨mf, hmf⟩ := isObj_of_getD hmr
  obtain ⟨m, hm⟩ := isObj_of_getD hmf
  have hinit : HInv (getD (getD hd "frame_v3") "roots") (getD (getD hd "frame_v3") "domain_ids") [] hd :=
    ⟨hw, ⟨m, hm⟩, ⟨mf, hmf⟩, fun _ _ => rfl, fun _ _ => rfl⟩
  have rinit : RInv slice [] slice := ⟨hs, fun _ _ => rfl⟩
  obtain ⟨hI, rI⟩ := dropAll_inv D (slice, hd) [] hinit rinit
  rw [List.append_nil] at hI rI
  obtain ⟨hw', -, -, hR', hDom'⟩ := hI
  obtain ⟨hu, hc⟩ := headerHeld_ok hfin
  generalize dropAll (slice, hd) D = s at hR' hDom' hw' rI hu hc
  obtain ⟨rec', h'⟩ := s
  dsimp only at hR' hDom' hw' rI hu hc
  have hRw' : Wf (getD (getD h' "frame_v3") "roots") := wf_getD (wf_getD hw' _) _
  have hx0 : field? (getD (getD hd "frame_v3") "roots") p = some x :=
    (mem_entries_iff (wf_getD (wf_getD hw _) _)).1 hpr
  have hx' : (p, x) ∈ entries (getD (getD h' "frame_v3") "roots") :=
    (mem_entries_iff hRw').2 ((hR' p hpD).trans hx0)
  unfold OneStageZk.unwired at hu
  have hany := Array.findSome?_eq_none_iff.1 hu (p, x) hx'
  dsimp only at hany
  have hany' : (Record.byPort rec').any (·.1 == p) = true := by
    cases hb : (Record.byPort rec').any (·.1 == p)
    · rw [hb] at hany; simp at hany
    · rfl
  obtain ⟨r, hr, hrp⟩ := Array.any_eq_true'.1 hany'
  have hrp' : r.1 = p := by simpa using hrp
  have hfind' : (Record.byPort rec').find? (·.1 == p) = some r := by rw [← hrp']; exact byPort_good rec' r hr
  have hfind : (Record.byPort slice).find? (·.1 == p) = some r := (rI.2 p hpD).symm.trans hfind'
  unfold Record.matchesPublic at hc
  obtain ⟨hroots, hdoms⟩ := matchesPublic_ok hc
  obtain ⟨⟨k', v'⟩, he', he1, he2⟩ := hroots r hr
  have hv' : field? (getD (getD h' "frame_v3") "roots") k' = some v' := (mem_entries_iff hRw').1 he'
  dsimp only at he1 he2
  rw [← he1, hrp', (hR' p hpD).trans hx0] at hv'
  cases hv'
  obtain ⟨pr', hpr', hd1, hd2⟩ := hdoms r hr
  have hgood := byPort_good rec' pr' hpr'
  rw [← hd1, hrp', hfind'] at hgood
  cases hgood
  refine ⟨r, hfind, he2, ?_⟩
  rw [hrp', hDom' p hpD] at hd2
  exact hd2

theorem header_wf {pubB : ByteArray} {hd : Json} (h : FilePart.header pubB = .ok hd) : Wf hd := by
  unfold FilePart.header at h
  split at h
  · cases hx : (Flock.JsonParse.parse _).mapError _ with
    | error e => rw [hx] at h; cases h
    | ok j =>
      rw [hx] at h
      cases h
      exact FlockDeploy.wf_norm j
  · first | cases h | (simp only [throw, throwThe, MonadExceptOf.throw, reduceCtorEq] at h; done)

theorem membersPortsHeld : MembersPortsHeld := by
  intro p ms c ξ e sts bs pub prof h
  obtain ⟨r, hr, -, -⟩ := OneStageZk.checkMembersAt_accept h
  obtain ⟨roots, hroots, hfiles⟩ := OneStageZk.membersRun_files hr
  refine ⟨roots, hroots, fun m hm => ?_⟩
  obtain ⟨a, ha, -, -, -, hpop⟩ := hfiles m hm
  rw [(_root_.Verity.Sampled.MemberForm.statement_checks ha).2] at hpop
  unfold OneStageZk.Member.population at hpop
  obtain ⟨prog, -, hpop⟩ := ebind_ok hpop
  obtain ⟨strata, -, hpop⟩ := ebind_ok hpop
  obtain ⟨hd, hh, hpop⟩ := ebind_ok hpop
  obtain ⟨⟨⟩, -, hpop⟩ := ebind_ok hpop
  obtain ⟨rd, hrd, hpop⟩ := ebind_ok hpop
  obtain ⟨hl, hhl, hpop⟩ := ebind_ok hpop
  exact ⟨a, hd, rd, hl, ha, hh, hrd, hhl,
    portsHeld_of_headerHeldAt (header_wf hh) ⟨_, rfl, Std.TreeMap.Raw.WF.ofList⟩ hpop⟩

end Properties.Spot.PortsHeldProof
