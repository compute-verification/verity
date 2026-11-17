module

public import Proofs.Properties.Verity.Sampled.RoundsRead
public import Proofs.Properties.Verity.Sampled.Layout
public import Proofs.Properties.Verity.Sampled.Device
public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Flock.Claims.Complete.Json.Parse
public import Proofs.Flock.Soundness.Refine.Walk
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-! The record's coin tree is the session's: each stream opens at most `2^roundsLog` rounds. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Spot Spot.Challenger
open FlockDeploy (JsonEquiv)
open FlockSoundness.Discharge
open FlockSoundness.Refine (ebind_ok throw_ne_ok throw_bind')
open Properties.Challenger.CanonLine (nat?_equiv)
open Properties.Challenger.CanonRead (sorted_getD getD_equiv sortedKeys_of_wf)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

set_option maxHeartbeats 4000000 in
theorem setupHidden_coinTreeP {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Json} {program own : Option ByteArray}
    {st : Flock.Stmt} (h : Flock.Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    st.coinTree = coins.tree? := by
  cases draw <;> unfold Flock.Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

theorem SetsUp.spec_coinTree {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) : spec.coinTree = some A.coinSpec := by
  have hc : A.I.count = 1 := hA.count
  unfold Exec.setupOfP at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · exact (setupHidden_coinTreeP hst).trans rfl
  · exact absurd h1 (by show ¬ 1 < A.I.count; omega)

theorem checkRecord_spec {sp : Flock.CoinTree.Spec} {nonce key : String} {c : Json}
    (h : Flock.CoinTree.checkRecord sp nonce c = .ok key) :
    ∃ s, Flock.canon (getD c "spec") = .ok s ∧ Flock.canon sp.toJson = .ok s := by
  unfold Flock.CoinTree.checkRecord at h
  dsimp only at h
  try simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    rename_i hc
    simp only [bne_iff_ne, ne_eq, not_not] at hc
    subst hc
    exact ⟨_, ‹Flock.canon ((c.getObjVal? "spec").toOption.getD Json.null) = _›, ‹Flock.canon sp.toJson = _›⟩

set_option maxHeartbeats 1000000 in
theorem decode_coinTree {spec : Flock.SessionSpec} {text : String} {sess : Flock.RunRecord}
    {sp : Flock.CoinTree.Spec} (h : Flock.Record.decode spec text = .ok sess) (hc : spec.coinTree = some sp) :
    ∃ j c nonce key, Flock.JsonParse.parse text = .ok j ∧ Flock.Record.fld j "coin_tree" = .ok c ∧
      Flock.CoinTree.checkRecord sp nonce c = .ok key := by
  revert h
  rw [Flock.Record.decode.eq_def]
  cases hp : Flock.JsonParse.parse text with
  | error e => intro h; cases h
  | ok j =>
    intro h
    dsimp only at h
    rw [pure_bind] at h
    simp only [hc] at h
    repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
    all_goals
      first
        | exact ⟨j, _, _, _, rfl, ‹Flock.Record.fld j "coin_tree" = .ok _›,
            ‹Flock.CoinTree.checkRecord sp _ _ = .ok _›⟩
        | exact absurd ‹(throw _ : Flock.V _) = Except.ok _› (throw_ne_ok _ _)

theorem getD_of_fld {j c : Json} {k : String} (h : Flock.Record.fld j k = .ok c) : getD j k = c := by
  unfold Flock.Record.fld at h
  cases hv : j.getObjVal? k with
  | error e => rw [hv] at h; cases h
  | ok v =>
    rw [hv] at h
    have e : v = c := Except.ok.inj h
    subst e
    show ((j.getObjVal? k).toOption).getD Json.null = _
    rw [hv]
    try rfl

theorem Trees.coinTree_held {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, recordTree record = t.root ∧ jeq (getD record "coin_tree") t.record = true := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  rename_i t ht
  refine ⟨t, Array.mem_of_find?_eq_some ht, ?_, ?_⟩
  · have hp := Array.find?_some ht
    unfold Verity.Sampled.recordTree
    rw [Trees.strIs hp]
  split at h
  case isFalse => cases h
  case isTrue hj => exact hj

theorem record_sorted {t : LoggedTree} (h : sortedKeys t.spec = true) : sortedKeys t.record = true := by
  unfold LoggedTree.record
  refine Device.sortedKeys_mkObj (by simp only [List.map_cons, List.map_nil]; decide) fun p hp => ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl
  · exact (Device.printed_str _).1
  · exact (Device.printed_str _).1
  · exact (Device.printed_str _).1
  · exact h

theorem spec_equiv {ct : Json} {t : LoggedTree} (hct : sortedKeys ct = true) (hts : sortedKeys t.spec = true)
    (h : jeq ct t.record = true) : JsonEquiv (getD ct "spec") t.spec := by
  have hrec := record_sorted hts
  unfold jeq at h
  split at h
  · simp only [beq_iff_eq] at h
    subst h
    have e := getD_equiv hct hrec
      (Properties.Challenger.CanonLine.canon_inj hct hrec ‹Flock.canon ct = _› ‹Flock.canon t.record = _›) "spec"
    have he : getD t.record "spec" = t.spec := rfl
    rwa [he] at e
  · cases h

theorem treeStep_spec {records : List Json} {ts ts' : List (String × Json)} {d : Json} {r : Verity.Release}
    (h : treeStep records ts d = some (r, ts')) : ∃ more, ts' = ts ++ more ∧ ∀ x ∈ more, x.2 = getD d "spec" := by
  by_cases hT : jsonStrIs (getD d "event") TREE_EVENT = true
  · unfold treeStep at h
    rw [ite_eq_left hT] at h
    simp only [Option.bind_eq_bind] at h
    split at h
    · split at h
      · split at h
        · rename_i root _
          obtain ⟨spec, -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨ν, -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨r', -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨κ, -, h⟩ := Option.bind_eq_some_iff.1 h
          simp only [Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨-, rfl⟩ := h
          exact ⟨[(root, getD d "spec")], rfl, by simp⟩
        · simp at h
      · simp at h
    · simp at h
  · rcases treeStep_some h with ⟨_, _, _, _, _, _, hT'⟩ | ⟨_, _, _, _, hts, -⟩
    · exact absurd hT' hT
    · exact ⟨[], by rw [hts, List.append_nil], by simp⟩

theorem treeFold_sorted {records : List Json} :
    ∀ (ls : List String) (ts tsF : List (String × Json)), treeFold records ts ls = some tsF →
      (∀ x ∈ ts, sortedKeys x.2 = true) → ∀ x ∈ tsF, sortedKeys x.2 = true
  | [], ts, tsF, h, hs => by
    simp only [treeFold, Option.some.injEq] at h
    subst h
    exact hs
  | l :: ls, ts, tsF, h, hs => by
    simp only [treeFold] at h
    split at h
    · obtain ⟨⟨r, ts'⟩, hX, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX
      refine treeFold_sorted ls ts' tsF h ?_
      obtain ⟨more, rfl, hm⟩ := treeStep_spec hX
      intro x hx
      rcases List.mem_append.1 hx with hx | hx
      · exact hs x hx
      · rw [hm x hx]
        exact sorted_getD (Properties.Challenger.CanonLine.event_ok (event_of_toOption hj)).2 "spec"
    · exact treeFold_sorted ls ts tsF h hs

theorem trees_sorted {lines : List String} {trees : Array LoggedTree} (h : coinTreesOf lines = .ok trees) :
    ∀ t ∈ trees, sortedKeys t.spec = true := fun t ht =>
  treeFold_sorted lines [] (heldTrees trees) (coinTreesOf_fold [] h) (fun _ hx => by simp at hx)
    (t.root, t.spec) (List.mem_map.2 ⟨t, Array.mem_toList_iff.2 ht, rfl⟩)

theorem specValue_toJson (sp : Flock.CoinTree.Spec) : Lines.SpecValue sp.toJson sp :=
  ⟨rfl, by show nat? (Lean.toJson sp.roundsLog) = _; simp [nat?, int?, Lean.toJson, Lean.JsonNumber.fromNat],
    by show nat? (Lean.toJson sp.block) = _; simp [nat?, int?, Lean.toJson, Lean.JsonNumber.fromNat]⟩

theorem specValue_of_equiv {x y : Json} {sp : Flock.CoinTree.Spec} (hx : sortedKeys x = true)
    (hy : sortedKeys y = true) (he : JsonEquiv x y) (h : Lines.SpecValue y sp) : Lines.SpecValue x sp := by
  have h1 := getD_equiv hx hy he "streams"
  rw [h.streams] at h1
  exact ⟨Lines.equiv_strs h1, (nat?_equiv (getD_equiv hx hy he "rounds_log")).trans h.roundsLog,
    (nat?_equiv (getD_equiv hx hy he "block")).trans h.block⟩

theorem helloSpec_of_record {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Verity.Release} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    (hR : readLog label after records ξ lines = some log) {Lg : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) log Lg) {record : Json}
    {tables : Array (String × Option ℕ)} (h : sessionTreeReason trees tables record = none)
    (hrs : sortedKeys record = true) {sp : Flock.CoinTree.Spec}
    {nonce key : String} (hC : Flock.CoinTree.checkRecord sp nonce (getD record "coin_tree") = .ok key) {i si : ℕ}
    {name : String} (hk : KeyAt (heldTrees trees) i si (recordTree record) name) : (helloSpecs Lg)[i]? = some sp := by
  obtain ⟨t, ht, hroot, hj⟩ := Trees.coinTree_held h
  have hcs := sorted_getD hrs "coin_tree"
  have hts := trees_sorted hT t ht
  obtain ⟨s, hs1, hs2⟩ := checkRecord_spec hC
  have hsp : Lines.SpecValue t.spec sp :=
    specValue_of_equiv hts (Device.printed_spec sp).1
      (Eq.trans (Eq.symm (spec_equiv hcs hts hj))
        (Properties.Challenger.CanonLine.canon_inj (sorted_getD hcs "spec") (Device.printed_spec sp).1 hs1 hs2))
      (specValue_toJson sp)
  obtain ⟨-, sj, hg, -⟩ := hk
  have hg' : (trees.toList.map fun t => (t.root, t.spec))[i]? = some (recordTree record, sj) := hg
  rw [List.getElem?_map, Array.getElem?_toList] at hg'
  cases hti : trees[i]? with
  | none => simp [hti] at hg'
  | some t' =>
    rw [hti] at hg'
    obtain ⟨hr', hs'⟩ := Prod.mk.inj (Option.some.inj hg')
    have e : t' = t := List.inj_on_of_nodup_map (Lines.trees_log hT).nodup
      (Array.mem_toList_iff.2 (Array.mem_iff_getElem?.2 ⟨i, hti⟩)) (Array.mem_toList_iff.2 ht) (hr'.trans hroot)
    obtain ⟨sp', hsp', hv⟩ := forall₂_getElem? (readLog_specs hR (coinTreesOf_fold records hT)) hg
    have hv' : specOf sj = some sp' := hv
    rw [← hs', e, Lines.specOf_value hsp] at hv'
    rw [← helloSpecs_congr hl, hsp']
    exact hv'.symm

theorem helloSpec_of_decode {spec : Flock.SessionSpec} {text : String} {sess : Flock.RunRecord}
    {sp : Flock.CoinTree.Spec} (hd : Flock.Record.decode spec text = .ok sess) (hc : spec.coinTree = some sp)
    {record : Json} (hJ : Flock.JsonParse.parse text = .ok record) {label : String} {after : ByteArray}
    {records : List Json} {ξ : ByteArray} {lines : List String} {log : List Verity.Release}
    {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees) (hR : readLog label after records ξ lines = some log)
    {Lg : List Verity.Release} (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) log Lg)
    {tables : Array (String × Option ℕ)} (h : sessionTreeReason trees tables record = none) {i si : ℕ}
    {name : String} (hk : KeyAt (heldTrees trees) i si (recordTree record) name) : (helloSpecs Lg)[i]? = some sp := by
  obtain ⟨j, c, nonce, key, hp, hcj, hC⟩ := decode_coinTree hd hc
  have hjr : j = record := Except.ok.inj (hp.symm.trans hJ)
  rw [hjr] at hcj
  rw [← getD_of_fld hcj] at hC
  exact helloSpec_of_record hT hR hl h (sortedKeys_of_wf record (FlockComplete.JsonRT.parse_wf text record hJ)) hC hk

theorem SetsUp.helloSpec_session {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (hA : SetsUp V θ F A) {text : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own text = .ok (spec, sts)) {sess : Flock.RunRecord}
    (hd : Flock.Record.decode spec text = .ok sess) {record : Json} (hJ : Flock.JsonParse.parse text = .ok record)
    {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray} {lines : List String}
    {log : List Verity.Release} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    (hR : readLog label after records ξ lines = some log) {Lg : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) log Lg)
    {tables : Array (String × Option ℕ)} (h : sessionTreeReason trees tables record = none) {i si : ℕ}
    {name : String}
    (hk : KeyAt (heldTrees trees) i si (recordTree record) name) : (helloSpecs Lg)[i]? = some A.coinSpec :=
  helloSpec_of_decode hd (hA.spec_coinTree hs) hJ hT hR hl h hk

theorem LogServed.openedOf_le {Lg : List Verity.Release} (hS : LogServed Lg) {i : ℕ} {sp : Flock.CoinTree.Spec}
    (hsp : (helloSpecs Lg)[i]? = some sp) (s : ℕ) : openedOf Lg i s ≤ 2 ^ sp.roundsLog := by
  suffices h : ∀ pre post, Lg = pre ++ post → openedOf pre i s ≤ 2 ^ sp.roundsLog from
    h Lg [] (List.append_nil Lg).symm
  intro pre
  induction pre using List.reverseRecOn with
  | nil => intro _ _; exact Nat.zero_le _
  | append_singleton pre r ih =>
    intro post he
    rw [List.append_assoc, List.singleton_append] at he
    rw [openedOf_append]
    by_cases hr : r.frame.opensAt i s = true
    · obtain ⟨n, hn⟩ := opening_of_opensAt hr
      rw [ite_eq_left hr]
      obtain ⟨sp', hsp', -, hlt, -⟩ := hS pre r post i s n he hn
      have h2 := helloSpecs_getElem?_append hsp' (r :: post)
      rw [← he, hsp] at h2
      obtain rfl := Option.some.inj h2
      exact hlt
    · rw [ite_eq_right hr]
      exact ih (r :: post) he

theorem SetsUp.covers_openings {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (hA : SetsUp V θ F A) {Lg : List Verity.Release} (hS : LogServed Lg) {i : ℕ}
    (hsp : (helloSpecs Lg)[i]? = some A.coinSpec) {S : Finset (Fin A.n)} {R : A.Reg} (t : Fin (A.J S R)) (r : Fin 2)
    (s : ℕ) : ∀ k < openedOf Lg i s, ∃ j, A.layout j = .rep t.val r k :=
  hA.layout_covers_tree t r (hS.openedOf_le hsp s)

theorem SetsUp.covers_logged {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (hA : SetsUp V θ F A) {text : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own text = .ok (spec, sts)) {sess : Flock.RunRecord}
    (hd : Flock.Record.decode spec text = .ok sess) {record : Json} (hJ : Flock.JsonParse.parse text = .ok record)
    {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray} {lines : List String}
    {log : List Verity.Release} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    (hR : readLog label after records ξ lines = some log) {Lg : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) log Lg) (hS : LogServed Lg)
    {tables : Array (String × Option ℕ)} (h : sessionTreeReason trees tables record = none) {i si : ℕ}
    {name : String} (hk : KeyAt (heldTrees trees) i si (recordTree record) name) {S : Finset (Fin A.n)} {R : A.Reg}
    (t : Fin (A.J S R)) (r : Fin 2) (s : ℕ) : ∀ k < openedOf Lg i s, ∃ j, A.layout j = .rep t.val r k :=
  hA.covers_openings hS (hA.helloSpec_session hs hd hJ hT hR hl h hk) t r s

end Verity.Sampled
