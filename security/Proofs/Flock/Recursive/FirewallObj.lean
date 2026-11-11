module

public import Flock.Firewall.Honest
public import Proofs.Flock.Recursive.FirewallJson
import all Lean.Data.Json.Basic
import all Std.Data.Internal.List.Associative
import Std.Data.DTreeMap.Internal.WF.Lemmas

/-!
# Objects as the ZK server builds, reads and pins them

The ZK server builds objects with `Json.mkObj`, reads them with `getObjVal?` and pins them by canonical form. An object
built from members with distinct keys holds those members in key order (`perm_ofList`, `toListModel_ofList`), two
objects whose keys ascend and whose members have the same keys and canonical forms have one canonical form
(`canonForm_obj_perm`), and a key of an object whose keys ascend reads its member's value (`getObjVal?_of_mem`) or
nothing (`getObjVal?_of_not_mem`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallObj

open Lean (Json)
open Flock Flock.Firewall
open Std.DTreeMap.Internal (Impl)

abbrev Tree := Impl String (fun _ => Json)

/-- Strictly ascending in `key`. -/
def Asc {γ : Type} (key : γ → String) (l : List γ) : Prop := l.Pairwise fun a b => compare (key a) (key b) = .lt

theorem eq_of_perm {γ : Type} (key : γ → String) {l l' : List γ} (hp : l.Perm l') (h : Asc key l)
    (h' : Asc key l') : l = l' := by
  refine List.Perm.eq_of_pairwise (le := fun a b => compare (key a) (key b) = .lt) ?_ h h' hp
  intro a b _ _ hab hba
  rw [Std.OrientedCmp.gt_iff_lt.symm, hab] at hba
  cases hba

theorem asc_ordered {t : Tree} (h : t.Ordered) : Asc (fun e : (_ : String) × Json => e.1) t.toListModel := h

theorem ordered_of_ascending {t : Tree} (h : ascending t = true) : t.Ordered := by
  unfold ascending at h
  exact (of_decide_eq_true h : t.toListModel.Pairwise fun a b => compare a.1 b.1 = .lt)

/-- Ascending keys are distinct. -/
theorem Asc.ne {γ : Type} {key : γ → String} {l : List γ} (h : Asc key l) : l.Pairwise fun a b => key a ≠ key b :=
  h.imp fun hab he => by
    rw [he] at hab
    exact absurd hab (by simp)

theorem Asc.map {γ δ : Type} {key : δ → String} (f : γ → δ) {l : List γ} (h : Asc (key ∘ f) l) :
    Asc key (l.map f) := List.pairwise_map.mpr h

/-- An object's members, each key with its value's canonical form. -/
def forms (t : Tree) : List (String × List Tok) := t.toListModel.map fun e => (e.1, canonForm e.2)

theorem canonForm_obj_forms (t : Tree) :
    canonForm (.obj ⟨⟨t⟩⟩) = .obj :: (forms t).flatMap (fun e => .key e.1 :: e.2) ++ [.close] := by
  rw [FirewallJson.canonForm_obj, FirewallJson.canonFormObj_eq, forms, List.flatMap_map]

/-- **Two objects of one canonical form**: their keys ascend, and their members have the same keys and canonical forms. -/
theorem canonForm_obj_perm {t t' : Tree} (ht : t.Ordered) (ht' : t'.Ordered) (hp : (forms t).Perm (forms t')) :
    canonForm (.obj ⟨⟨t⟩⟩) = canonForm (.obj ⟨⟨t'⟩⟩) := by
  rw [canonForm_obj_forms, canonForm_obj_forms,
    eq_of_perm Prod.fst hp (Asc.map (key := Prod.fst) _ ht) (Asc.map (key := Prod.fst) _ ht')]

/-! ## Built from members -/

theorem mkObj_eq (L : List (String × Json)) : Json.mkObj L = .obj ⟨⟨Impl.Const.ofList L⟩⟩ := (rfl)

theorem wf_ofList (L : List (String × Json)) : (Impl.Const.ofList L : Tree).WF :=
  Impl.WF.constInsertMany Impl.WF.empty

theorem ordered_ofList (L : List (String × Json)) : (Impl.Const.ofList L : Tree).Ordered := (wf_ofList L).ordered

/-- An object built from members with distinct keys holds them. -/
theorem perm_ofList {L : List (String × Json)} (h : L.Pairwise fun a b => a.1 ≠ b.1) :
    (Impl.Const.ofList L : Tree).toListModel.Perm (L.map fun e => ⟨e.1, e.2⟩) := by
  refine (Impl.Const.toListModel_insertMany_list Impl.WF.empty).trans ?_
  refine (Std.Internal.List.perm_insertList Std.Internal.List.DistinctKeys.nil ?_ ?_).trans ?_
  · rw [List.pairwise_map]
    refine h.imp fun hab => ?_
    simp only [Std.Internal.List.Prod.toSigma, beq_eq_false_iff_ne]
    exact hab
  · intro a ha
    simp [Std.Internal.List.containsKey] at ha
  · simp only [List.nil_append]
    rfl

/-- An object built from members with ascending keys holds them in order. -/
theorem toListModel_ofList {L : List (String × Json)} (h : Asc Prod.fst L) :
    (Impl.Const.ofList L : Tree).toListModel = L.map fun e => ⟨e.1, e.2⟩ :=
  eq_of_perm (fun e : (_ : String) × Json => e.1) (perm_ofList h.ne) (asc_ordered (ordered_ofList L))
    (Asc.map (key := fun e : (_ : String) × Json => e.1) _ h)

theorem forms_ofList {L : List (String × Json)} (h : L.Pairwise fun a b => a.1 ≠ b.1) :
    (forms (Impl.Const.ofList L)).Perm (L.map fun e => (e.1, canonForm e.2)) := by
  have := (perm_ofList h).map fun e : (_ : String) × Json => (e.1, canonForm e.2)
  rw [List.map_map] at this
  exact this

/-- **Two objects built from members of one canonical form** have one canonical form. -/
theorem canonForm_mkObj_perm {L L' : List (String × Json)} (h : L.Pairwise fun a b => a.1 ≠ b.1)
    (h' : L'.Pairwise fun a b => a.1 ≠ b.1)
    (hp : (L.map fun e => (e.1, canonForm e.2)).Perm (L'.map fun e => (e.1, canonForm e.2))) :
    canonForm (Json.mkObj L) = canonForm (Json.mkObj L') :=
  canonForm_obj_perm (ordered_ofList L) (ordered_ofList L') ((forms_ofList h).trans (hp.trans (forms_ofList h').symm))

/-! ## Read by key -/

theorem getObjVal?_of_mem {t : Tree} (ht : t.Ordered) {k : String} {v : Json}
    (h : (⟨k, v⟩ : (_ : String) × Json) ∈ t.toListModel) : Json.getObjVal? (.obj ⟨⟨t⟩⟩) k = .ok v := by
  have e : Impl.Const.get? t k = some v := by
    rw [Impl.Const.get?_eq_getValue? ht, Std.Internal.List.getValue?_eq_getEntry?,
      Std.Internal.List.getEntry?_of_mem ht.distinctKeys (beq_self_eq_true k) h]
    rfl
  show (match Impl.Const.get? t k with | some v => pure v | none => throw _) = _
  rw [e]
  rfl

theorem getObjVal?_of_not_mem {t : Tree} (ht : t.Ordered) {k : String} (h : ∀ e ∈ t.toListModel, e.1 ≠ k) :
    (Json.getObjVal? (.obj ⟨⟨t⟩⟩) k).toOption = none := by
  have e : Impl.Const.get? t k = none := by
    rw [Impl.Const.get?_eq_getValue? ht, Std.Internal.List.getValue?_eq_none,
      Std.Internal.List.containsKey_eq_false_iff]
    intro b hb
    rw [beq_eq_false_iff_ne]
    exact fun he => h b hb he.symm
  show (match Impl.Const.get? t k with | some v => (pure v : Except String Json) | none => throw _).toOption = _
  rw [e]
  rfl

/-- A key some member has, read off an object whose keys ascend. -/
theorem mem_of_getObjVal? {t : Tree} (ht : t.Ordered) {k : String} {v : Json}
    (h : Json.getObjVal? (.obj ⟨⟨t⟩⟩) k = .ok v) : (⟨k, v⟩ : (_ : String) × Json) ∈ t.toListModel := by
  by_cases hk : ∃ e ∈ t.toListModel, e.1 = k
  · obtain ⟨⟨k', v'⟩, he, rfl⟩ := hk
    rw [getObjVal?_of_mem ht he] at h
    cases h
    exact he
  · have := getObjVal?_of_not_mem ht (k := k) fun e he h' => hk ⟨e, he, h'⟩
    rw [h] at this
    cases this

/-! ## The members of an object as `kvs.toArray.toList` reads them -/

theorem toArray_toList (t : Tree) :
    (Std.TreeMap.Raw.toArray (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json)).toList = t.toListModel.map fun e => (e.1, e.2) := by
  show (Impl.Const.toArray t).toList = _
  rw [Impl.Const.toArray_eq_toArray_map, List.toList_toArray]

/-! ## Rebuilt by the parser -/

/-- An object whose keys ascend, as the parser rebuilds it (`FirewallJson.norm`): each member's value rebuilt, in order. -/
theorem norm_obj {t : Tree} (ht : t.Ordered) :
    ∃ t' : Tree, FirewallJson.norm (.obj ⟨⟨t⟩⟩) = .obj ⟨⟨t'⟩⟩ ∧
      t'.toListModel = t.toListModel.map fun e => ⟨e.1, FirewallJson.norm e.2⟩ := by
  refine ⟨(FirewallJson.normObj t ∅).inner.inner, rfl, ?_⟩
  rw [FirewallJson.normObj_eq]
  have hf : t.toListModel.foldl (fun a e => a.insert e.1 (FirewallJson.norm e.2)) (∅ : Std.TreeMap.Raw String Json) =
      (t.toListModel.map fun e => (⟨e.1, FirewallJson.norm e.2⟩ : (_ : String) × Json)).foldl
        (fun a e => a.insert e.1 e.2) ∅ := by
    rw [List.foldl_map]
  rw [hf, FirewallJson.toListModel_foldl _ _ Impl.WF.empty (by
    show ([] ++ _).Pairwise _
    rw [List.nil_append, List.pairwise_map]
    exact ht)]
  rfl

theorem ascending_of_ordered {t : Tree} (h : t.Ordered) : ascending t = true := by
  unfold ascending
  exact decide_eq_true (h : t.toListModel.Pairwise fun a b => compare a.1 b.1 = .lt)

/-- An object built from plain members is plain. -/
theorem plain_mkObj {L : List (String × Json)} (h : L.Pairwise fun a b => a.1 ≠ b.1)
    (hp : ∀ e ∈ L, plainStr e.1 = true ∧ plain e.2 = true) : plain (Json.mkObj L) = true := by
  rw [mkObj_eq]
  show (plainObj _ && ascending _) = true
  rw [ascending_of_ordered (ordered_ofList L), Bool.and_true, FirewallJson.plainObj_iff]
  intro e he
  obtain ⟨e', he', rfl⟩ := List.mem_map.mp ((perm_ofList h).mem_iff.mp he)
  exact hp e' he'

/-! ## Hello with Σ -/

/-- Hello's member as `helloWith` writes it, Σ in its link. -/
def linked (sigma : String) (e : (_ : String) × Json) : String × Json :=
  (e.1, if e.1 == "link" then helloWith.linkWith sigma e.2 else e.2)

theorem helloWith_obj (t : Tree) (sigma : String) :
    helloWith (.obj ⟨⟨t⟩⟩) sigma = Json.mkObj (t.toListModel.map (linked sigma)) := by
  simp only [helloWith, toArray_toList, List.map_map]
  rfl

theorem linkWith_obj (t : Tree) (sigma : String) :
    helloWith.linkWith sigma (.obj ⟨⟨t⟩⟩) =
      Json.mkObj (t.toListModel.map (fun e => (e.1, e.2)) ++ [("sigma", .str sigma)]) := by
  simp only [helloWith.linkWith, toArray_toList]

theorem pairwise_ne_map {γ : Type} {key : γ → String} {l : List γ} (h : Asc key l) :
    (l.map fun e => (key e, ())).Pairwise fun a b => a.1 ≠ b.1 := List.pairwise_map.mpr h.ne

/-- **Hello as the ZK server reads the worker's**: at a schedule's Hello, an object whose keys ascend with plain members,
whose link (`A ++ link :: B`) is an object whose keys ascend with plain members and no `sigma`, and which has no
`coins`, the worker's Hello with a plain Σ is plain, the parser's value of its text has a link that names Σ and no
`coins`, and with Σ taken out of its link it has the schedule's Hello's canonical form. -/
theorem hello {H L : Tree} {A B : List ((_ : String) × Json)} (hH : H.Ordered) (hpH : plainObj H = true)
    (hL : L.Ordered) (hpL : plainObj L = true) (hsplit : H.toListModel = A ++ ⟨"link", .obj ⟨⟨L⟩⟩⟩ :: B)
    (hs : ∀ e ∈ L.toListModel, e.1 ≠ "sigma") (hc : ∀ e ∈ H.toListModel, e.1 ≠ "coins") {sigma : String}
    (hsig : plainStr sigma = true) :
    plain (helloWith (.obj ⟨⟨H⟩⟩) sigma) = true ∧
      ∃ T : Tree, (FirewallJson.norm (helloWith (.obj ⟨⟨H⟩⟩) sigma)).getObjVal? "link" = .ok (.obj ⟨⟨T⟩⟩) ∧
        (Json.obj ⟨⟨T⟩⟩).getObjVal? "sigma" = .ok (.str sigma) ∧
        ((FirewallJson.norm (helloWith (.obj ⟨⟨H⟩⟩) sigma)).getObjVal? "coins").toOption = none ∧
        canonForm (withField (FirewallJson.norm (helloWith (.obj ⟨⟨H⟩⟩) sigma)) "link"
          (without (.obj ⟨⟨T⟩⟩) "sigma")) = canonForm (.obj ⟨⟨H⟩⟩) := by
  have hpm := FirewallJson.plainObj_iff H |>.mp hpH
  have hpl := FirewallJson.plainObj_iff L |>.mp hpL
  -- the members of `A` and `B` are `H`'s, and none is the link
  have hHasc : Asc (fun e : (_ : String) × Json => e.1) (A ++ ⟨"link", .obj ⟨⟨L⟩⟩⟩ :: B) := hsplit ▸ hH
  have hAB : ∀ e ∈ A ++ B, e.1 ≠ "link" := by
    intro e he
    have hne := (hHasc.ne)
    rw [List.pairwise_append, List.pairwise_cons] at hne
    rcases List.mem_append.mp he with he | he
    · exact hne.2.2 e he _ List.mem_cons_self
    · exact fun h => hne.2.1.1 e he h.symm
  have hmemH : ∀ e ∈ A ++ B, e ∈ H.toListModel := by
    intro e he
    rw [hsplit]
    rcases List.mem_append.mp he with he | he
    · exact List.mem_append_left _ he
    · exact List.mem_append_right _ (List.mem_cons_of_mem _ he)
  have hlinkH : (⟨"link", .obj ⟨⟨L⟩⟩⟩ : (_ : String) × Json) ∈ H.toListModel := by
    rw [hsplit]
    exact List.mem_append_right _ List.mem_cons_self
  have hlinked : ∀ e ∈ A ++ B, linked sigma e = (e.1, e.2) := by
    intro e he
    simp only [linked, beq_eq_false_iff_ne.mpr (hAB e he), Bool.false_eq_true, ite_false]
  -- the link with Σ
  let Ls : List (String × Json) := L.toListModel.map (fun e => (e.1, e.2)) ++ [("sigma", .str sigma)]
  have hLs : Ls.Pairwise fun a b => a.1 ≠ b.1 := by
    refine List.pairwise_append.mpr ⟨?_, List.pairwise_singleton _ _, ?_⟩
    · exact List.pairwise_map.mpr (Asc.ne (asc_ordered hL))
    · intro a ha b hb
      rw [List.mem_singleton] at hb
      subst hb
      obtain ⟨a', ha', rfl⟩ := List.mem_map.mp ha
      exact hs a' ha'
  have hplw : plain (helloWith.linkWith sigma (.obj ⟨⟨L⟩⟩)) = true := by
    rw [linkWith_obj]
    refine plain_mkObj hLs fun e he => ?_
    rcases List.mem_append.mp he with he | he
    · obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
      exact hpl e' he'
    · rw [List.mem_singleton] at he
      subst he
      exact ⟨by simp [plainStr], hsig⟩
  -- Hello's members with Σ in its link
  let M := H.toListModel.map (linked sigma)
  have hMeq : M = (A.map fun e => (e.1, e.2)) ++ ("link", helloWith.linkWith sigma (.obj ⟨⟨L⟩⟩)) ::
      B.map fun e => (e.1, e.2) := by
    show H.toListModel.map (linked sigma) = _
    rw [hsplit, List.map_append, List.map_cons]
    congr 1
    · exact List.map_congr_left fun e he => hlinked e (List.mem_append_left _ he)
    · congr 1
      exact List.map_congr_left fun e he => hlinked e (List.mem_append_right _ he)
  have hMasc : Asc Prod.fst M := Asc.map (key := Prod.fst) (linked sigma) hH
  have hplain : plain (helloWith (.obj ⟨⟨H⟩⟩) sigma) = true := by
    rw [helloWith_obj]
    refine plain_mkObj hMasc.ne fun e he => ?_
    obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
    by_cases hk : e'.1 = "link"
    · have : e' = ⟨"link", .obj ⟨⟨L⟩⟩⟩ := by
        obtain ⟨k, v⟩ := e'
        simp only at hk
        subst hk
        have h1 := getObjVal?_of_mem hH he'
        rw [getObjVal?_of_mem hH hlinkH] at h1
        cases h1
        rfl
      subst this
      exact ⟨by simp [linked, plainStr], hplw⟩
    · simp only [linked, beq_eq_false_iff_ne.mpr hk, Bool.false_eq_true, ite_false]
      exact hpm e' he'
  refine ⟨hplain, ?_⟩
  -- the parser's value of the worker's Hello
  obtain ⟨Tj, hTj, hTjl⟩ := norm_obj (ordered_ofList M)
  rw [toListModel_ofList hMasc, List.map_map] at hTjl
  have hTjo : Tj.Ordered := by
    show Asc (fun e : (_ : String) × Json => e.1) Tj.toListModel
    rw [hTjl]
    exact Asc.map (key := fun e : (_ : String) × Json => e.1) _ hMasc
  obtain ⟨T, hT, hTl⟩ := norm_obj (ordered_ofList Ls)
  have hTo : T.Ordered := by
    show Asc (fun e : (_ : String) × Json => e.1) T.toListModel
    rw [hTl]
    exact Asc.map (key := fun e : (_ : String) × Json => e.1) _ (ordered_ofList Ls)
  have hj : FirewallJson.norm (helloWith (.obj ⟨⟨H⟩⟩) sigma) = .obj ⟨⟨Tj⟩⟩ := by
    rw [helloWith_obj, mkObj_eq]
    exact hTj
  have hlw : FirewallJson.norm (helloWith.linkWith sigma (.obj ⟨⟨L⟩⟩)) = .obj ⟨⟨T⟩⟩ := by
    rw [linkWith_obj, mkObj_eq]
    exact hT
  refine ⟨T, ?_, ?_, ?_, ?_⟩
  · rw [hj, getObjVal?_of_mem hTjo (v := .obj ⟨⟨T⟩⟩)]
    rw [hTjl, hMeq]
    refine List.mem_map.mpr ⟨_, List.mem_append_right _ List.mem_cons_self, ?_⟩
    simp only [Function.comp, hlw]
  · refine getObjVal?_of_mem hTo ?_
    rw [hTl]
    refine List.mem_map.mpr ⟨⟨"sigma", .str sigma⟩, (perm_ofList hLs).mem_iff.mpr ?_, rfl⟩
    exact List.mem_map.mpr ⟨_, List.mem_append_right _ (List.mem_singleton_self _), rfl⟩
  · rw [hj]
    refine getObjVal?_of_not_mem hTjo fun e he => ?_
    rw [hTjl] at he
    obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
    obtain ⟨h0, hh0, rfl⟩ := List.mem_map.mp he'
    exact hc h0 hh0
  · -- Σ out of the link: the schedule's link
    have hnormAB : ∀ e ∈ A ++ B, canonForm (FirewallJson.norm e.2) = canonForm e.2 :=
      fun e he => FirewallJson.canonForm_norm e.2 (hpm e (hmemH e he)).2
    let Xs := (T.toListModel.map fun e => (e.1, e.2)).filter (·.1 != "sigma")
    have hTp : (T.toListModel.map fun e => (e.1, e.2)).Pairwise fun a b : String × Json => a.1 ≠ b.1 :=
      List.pairwise_map.mpr (Asc.ne (asc_ordered hTo))
    have hXs : Xs.Pairwise fun a b => a.1 ≠ b.1 := hTp.filter _
    have hTpair : (T.toListModel.map fun e => (e.1, e.2)).Perm (Ls.map fun e => (e.1, FirewallJson.norm e.2)) := by
      rw [hTl, List.map_map]
      have := ((perm_ofList hLs).map fun e : (_ : String) × Json => (e.1, FirewallJson.norm e.2))
      rw [List.map_map] at this
      exact this
    have hX : canonForm (Json.mkObj Xs) = canonForm (.obj ⟨⟨L⟩⟩) := by
      rw [mkObj_eq]
      refine canonForm_obj_perm (ordered_ofList Xs) hL ((forms_ofList hXs).trans ?_)
      have h1 := (hTpair.filter (·.1 != "sigma")).map fun e : String × Json => (e.1, canonForm e.2)
      refine h1.trans (List.Perm.of_eq ?_)
      simp only [Ls, List.map_append, List.map_cons, List.map_nil, List.map_map, List.filter_append, List.filter_cons,
        List.filter_nil, bne_self_eq_false, Bool.false_eq_true, ite_false, List.append_nil, forms]
      rw [List.filter_eq_self.mpr, List.map_map]
      · refine List.map_congr_left fun e he => ?_
        simp only [Function.comp]
        rw [FirewallJson.canonForm_norm e.2 (hpl e he).2]
      · intro e he
        obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
        simp only [Function.comp, bne_iff_ne, ne_eq]
        exact hs e' he'
    -- Hello with the schedule's link back in its place
    have hTjpair : (Tj.toListModel.map fun e => (e.1, e.2)) =
        (A.map fun e => (e.1, FirewallJson.norm e.2)) ++
          ("link", FirewallJson.norm (helloWith.linkWith sigma (.obj ⟨⟨L⟩⟩))) ::
          B.map fun e => (e.1, FirewallJson.norm e.2) := by
      rw [hTjl, List.map_map, hMeq]
      simp only [List.map_append, List.map_cons, List.map_map]
      rfl
    have hfilt : (Tj.toListModel.map fun e => (e.1, e.2)).filter (·.1 != "link") =
        (A.map fun e => (e.1, FirewallJson.norm e.2)) ++ B.map fun e => (e.1, FirewallJson.norm e.2) := by
      rw [hTjpair, List.filter_append, List.filter_cons]
      simp only [bne_self_eq_false, Bool.false_eq_true, ite_false]
      rw [List.filter_eq_self.mpr, List.filter_eq_self.mpr]
      · intro e he
        obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
        simp only [bne_iff_ne, ne_eq]
        exact hAB e' (List.mem_append_right _ he')
      · intro e he
        obtain ⟨e', he', rfl⟩ := List.mem_map.mp he
        simp only [bne_iff_ne, ne_eq]
        exact hAB e' (List.mem_append_left _ he')
    rw [hj]
    show canonForm (Json.mkObj (((⟨⟨Tj⟩⟩ : Std.TreeMap.Raw String Json).toArray.toList.filter (·.1 != "link")) ++
      [("link", Json.mkObj ((⟨⟨T⟩⟩ : Std.TreeMap.Raw String Json).toArray.toList.filter (·.1 != "sigma")))])) = _
    rw [toArray_toList, toArray_toList, hfilt]
    let Ws := ((A.map fun e => (e.1, FirewallJson.norm e.2)) ++ B.map fun e => (e.1, FirewallJson.norm e.2)) ++
      [("link", Json.mkObj Xs)]
    have hWs : Ws.Pairwise fun a b => a.1 ≠ b.1 := by
      have hne := hHasc.ne
      rw [List.pairwise_append, List.pairwise_cons] at hne
      refine List.pairwise_append.mpr ⟨?_, List.pairwise_singleton _ _, ?_⟩
      · rw [← List.map_append]
        refine List.pairwise_map.mpr ?_
        exact List.pairwise_append.mpr ⟨hne.1, hne.2.1.2, fun a ha b hb => hne.2.2 a ha b (List.mem_cons_of_mem _ hb)⟩
      · intro a ha b hb
        rw [List.mem_singleton] at hb
        subst hb
        rw [← List.map_append] at ha
        obtain ⟨a', ha', rfl⟩ := List.mem_map.mp ha
        exact hAB a' ha'
    rw [mkObj_eq]
    refine canonForm_obj_perm (ordered_ofList Ws) hH ((forms_ofList hWs).trans ?_)
    simp only [Ws, forms, hsplit, List.map_append, List.map_cons, List.map_map]
    rw [show canonForm (Json.mkObj Xs) = canonForm (.obj ⟨⟨L⟩⟩) from hX, List.append_assoc]
    refine List.Perm.append (List.Perm.of_eq (List.map_congr_left fun e he => ?_)) ?_
    · simp only [Function.comp, hnormAB e (List.mem_append_left _ he)]
    · simp only [List.map_nil]
      refine (List.perm_append_singleton _ _).trans
        (List.Perm.cons _ (List.Perm.of_eq (List.map_congr_left fun e he => ?_)))
      simp only [Function.comp, hnormAB e (List.mem_append_right _ he)]

end FlockSoundness.Discharge.Recursion.FirewallObj
