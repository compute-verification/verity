import Proofs.Flock.Claims.Complete.ZkTables.Reps
import Proofs.Flock.Claims.Complete.Zk.Json.Record
import Proofs.Flock.Verifier.Hash

/-!
# What the masked record needs of the statement, at several tables

`Case.stmtAt`: at every good case of several tables, `StmtAt` holds under the JSON library's specification
(`JsonSpec`), from the case's setup alone.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open FlockSoundness.Refine
open Lean (Json)

/-! ## The hello -/

theorem good_of_same : ∀ {a b : Json}, Same a b → Good a → Good b
  | _, _, .null, _ => .null
  | _, _, .bool b, _ => .bool b
  | _, _, .num _, h => h
  | _, _, .str s, _ => .str s
  | _, _, .arr hs he, h => by
    cases h with
    | arr hg =>
    refine .arr fun x hx => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
    exact good_of_same (he i (hs ▸ hi) hi) (hg _ (Array.getElem_mem _))
  | _, _, .obj _ hn hk hv, h => by
    cases h with
    | obj _ hg =>
    refine .obj hn fun k v hb => ?_
    have hs := hk k
    rw [hb] at hs
    obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 hs
    exact good_of_same (hv k a v ha hb) (hg k a ha)

theorem good_insert {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hg : ∀ k v, m.get? k = some v → Good v)
    {k : String} {v : Json} (hv : Good v) : Good (.obj (m.insert k v)) := by
  refine .obj hm.insert fun k' v' h => ?_
  rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm] at h
  split at h
  · cases h; exact hv
  · rw [← Std.TreeMap.Raw.get?_eq_getElem?] at h
    exact hg _ _ h

theorem good_coins (ct : CoinTree.Spec) (nonce : String) : Good (ct.coins nonce) := by
  have hg := good_specJson ct
  unfold CoinTree.Spec.coins
  unfold CoinTree.Spec.toJson at hg ⊢
  cases hg with
  | obj hm hg => exact good_insert hm hg (.str _)

theorem hello_of_list (S : JsonSpec) {ct : CoinTree.Spec} {L : List (String × Json)} {hello : String}
    (hh : canon (Json.mkObj L) = .ok hello) (hd : Distinct L) (hg : ∀ p ∈ L, Good p.2)
    (hc : ("coins", ct.coins "") ∈ L) :
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Json.parse hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) := by
  have hgj := good_mkObj hd hg
  obtain ⟨j', hp, hs⟩ := S.parse_canon _ _ hgj hh
  obtain ⟨m, rfl, hm, -⟩ := same_obj_wf hs
  obtain ⟨cv, hcv, hcs⟩ := hs.getObjVal (getObjVal?_mkObj hd hc)
  have hco : ∃ M : Std.TreeMap.Raw String Json compare, ct.coins "" = .obj M := ⟨_, rfl⟩
  obtain ⟨M, hM⟩ := hco
  rw [hM] at hcs
  obtain ⟨kvs, rfl, hkv, -⟩ := same_obj_wf hcs
  have hmg : m.get? "coins" = some (.obj kvs) := by
    simp only [Json.getObjVal?] at hcv
    split at hcv
    · cases hcv; assumption
    · cases hcv
  have hgm : Good (.obj m) := good_of_same hs hgj
  have hgk : Good (.obj kvs) := good_of_same hcs (hM ▸ good_coins ct "")
  refine ⟨m, kvs, hp, hm, hkv, hmg, ?_⟩
  cases hgm with
  | obj _ hgm =>
  cases hgk with
  | obj _ hgk =>
  have e : helloJ (.obj m) = .obj (m.insert "coins" (.obj (kvs.insert "nonce" (Json.str (zeros 64))))) := by
    simp only [helloJ, Json.getObjVal?, hmg, Json.setObjVal!]
    rfl
  rw [e]
  exact good_insert hm hgm (good_insert hkv hgk (.str _))

/-- **The masked `Hello` reads back** as an object whose `coins` are an object, and the prover's version, with its
nonce put in, is printable. -/
theorem hello_ok (S : JsonSpec) {tags : Tags} {sigma : ByteArray} {mPts : ℕ} {ct : CoinTree.Spec}
    {names : Array String} {hello : String} (hh : Flock.helloOf tags sigma mPts (.tree ct) names = .ok hello) :
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Json.parse hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) := by
  have hlink : Good (Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts),
      ("points", Lean.toJson (2 : ℕ))]) := by
    refine good_mkObj (by simp) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨.str _, good_toJson _, good_toJson _⟩
  have harr : Good (Json.arr (names.map Json.str)) := good_arr_of fun x hx => by
    obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
    exact .str _
  unfold Flock.helloOf at hh
  dsimp only at hh
  rcases Bool.eq_false_or_eq_true tags.retainRounds with hr | hr <;>
    rcases Bool.eq_false_or_eq_true tags.hm96Rows with h9 | h9 <;>
    simp only [hr, h9, ite_true, Bool.false_eq_true, ite_false, List.cons_append, List.nil_append] at hh <;>
    refine hello_of_list S (ct := ct) hh (by simp) ?_ (by simp) <;>
    simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq] <;>
    first
      | exact ⟨.str _, good_toJson _, .str _, harr, hlink, good_coins ct "", .str _, .str _⟩
      | exact ⟨.str _, good_toJson _, .str _, harr, hlink, good_coins ct "", .str _⟩
      | exact ⟨.str _, good_toJson _, .str _, harr, hlink, good_coins ct ""⟩

/-! ## A subset draw that decodes is printable -/

private theorem key_mem {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {L : List String} (hL : L.Nodup)
    (hin : ∀ k ∈ L, k ∈ m) (hs : m.size = L.length) {k : String} (hk : k ∈ m) : k ∈ L := by
  by_contra hn
  have hsub : k :: L ⊆ m.keys := by
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact (Std.TreeMap.Raw.mem_keys hm).2 hk
    · exact (Std.TreeMap.Raw.mem_keys hm).2 (hin x hx)
  have hle := (List.subperm_of_subset (List.nodup_cons.2 ⟨hn, hL⟩) hsub).length_le
  rw [Std.TreeMap.Raw.length_keys hm, hs] at hle
  simp at hle

private theorem mem_of_getObjVal {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
    (h : (Json.obj m).getObjVal? k = .ok v) : k ∈ m ∧ m.get? k = some v := by
  simp only [Json.getObjVal?] at h
  cases hg : m.get? k with
  | none => rw [hg] at h; cases h
  | some a =>
    rw [hg] at h
    cases h
    refine ⟨(Std.TreeMap.Raw.mem_iff_isSome_getElem? hm).2 ?_, rfl⟩
    rw [← Std.TreeMap.Raw.get?_eq_getElem?, hg]
    rfl

private theorem getObjVal_of_get {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json}
    (h : m.get? k = some v) : (Json.obj m).getObjVal? k = .ok v := by
  simp only [Json.getObjVal?, h]
  rfl

private theorem getNat_ok {j : Json} {n : ℕ} (h : j.getNat? = .ok n) : j = .num ⟨Int.ofNat n, 0⟩ := by
  unfold Json.getNat? at h
  split at h
  · cases h; rfl
  · cases h

private theorem asNat_good {j : Json} {c : String} {n : ℕ} (h : J.asNat j c = .ok n) : Good j := by
  unfold J.asNat at h
  split at h
  · rename_i _ hg; cases h; rw [getNat_ok hg]; exact .num _
  · cases h

private theorem get_ok {j : Json} {k c : String} {v : Json} (h : J.get j k c = .ok v) : j.getObjVal? k = .ok v := by
  unfold J.get at h
  split at h
  · rename_i _ hg; cases h; exact hg
  · cases h

private theorem nat_good {j : Json} {k c : String} {n : ℕ} (h : J.nat j k c = .ok n) :
    ∃ v, j.getObjVal? k = .ok v ∧ Good v := by
  unfold J.nat at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  exact ⟨v, get_ok hv, asNat_good h⟩

private theorem str_good {j : Json} {k c : String} {s : String} (h : J.str j k c = .ok s) :
    ∃ v, j.getObjVal? k = .ok v ∧ Good v := by
  unfold J.str at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  refine ⟨v, get_ok hv, ?_⟩
  unfold J.asStr at h
  split at h
  · rename_i _ hg
    cases v with
    | str => exact .str _
    | _ => cases hg
  · cases h

private theorem arr_ok {j : Json} {k c : String} {a : Array Json} (h : J.arr j k c = .ok a) :
    j.getObjVal? k = .ok (.arr a) := by
  unfold J.arr at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [get_ok hv]
  unfold J.asArr at h
  split at h
  · rename_i _ hg
    cases h
    unfold Json.getArr? at hg
    split at hg
    · cases hg; rfl
    · cases hg
  · cases h

private theorem getObj_ok {j : Json} {m : Std.TreeMap.Raw String Json compare} (h : j.getObj? = .ok m) :
    j = .obj m := by
  cases j with
  | obj m' => cases h; rfl
  | _ => cases h

private theorem good_of_asNat_mapM {a : Array Json} {c : String} {b : Array ℕ}
    (h : a.mapM (J.asNat · c) = .ok b) : Good (.arr a) :=
  .arr fun x hx => by
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
    obtain ⟨hs, hb⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
    exact asNat_good (hb i hi (hs ▸ hi))

/-- An object with exactly the keys of `L`, each holding a printable value, is printable. -/
private theorem good_obj_of {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {L : List String} (hL : L.Nodup)
    (hs : m.toArray.size = L.length) (hk : ∀ k ∈ L, ∃ v, (Json.obj m).getObjVal? k = .ok v ∧ Good v) :
    Good (.obj m) := by
  refine .obj hm fun k v hv => ?_
  have hv' := getObjVal_of_get hv
  have hkL : k ∈ L := key_mem hm hL (fun k hk' => by
      obtain ⟨v, hv, -⟩ := hk k hk'
      exact (mem_of_getObjVal hm hv).1)
    (by rw [← Std.TreeMap.Raw.size_toArray hm]; exact hs) (mem_of_getObjVal hm hv').1
  obtain ⟨v', hv'', hg⟩ := hk k hkL
  rw [hv'] at hv''
  cases hv''
  exact hg

set_option maxHeartbeats 4000000 in
/-- **A subset draw that decodes is printable**, an object of exactly `law`, `population`, `units` and `k`, so it has no
closure. -/
theorem good_subset {j : Json} (hw : Wf j) {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) {k : ℕ}
    (hl : D.law = .subset k) : Good j ∧ D.closure = #[] := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : V Draw.Law) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    try simp only [reduceCtorEq] at hl
  all_goals
    obtain ⟨m, rfl⟩ : ∃ m, j = Json.obj m := ⟨_, getObj_ok ‹j.getObj? = _›⟩
    have hjm := ‹(Json.obj m).getObj? = Except.ok _›
    simp only [Json.getObj?, pure, Except.pure, Except.ok.injEq] at hjm
    subst hjm
    cases hw with
    | obj hm _ =>
    have hc := ‹¬ (_ != 4) = true›
    simp only [bne_iff_ne, ne_eq, not_not] at hc
    have hL : ["law", "population", "units", "k"].Nodup := by decide
    have hk : ∀ key ∈ ["law", "population", "units", "k"], ∃ v, (Json.obj m).getObjVal? key = .ok v ∧ Good v := by
      simp only [List.mem_cons, List.mem_nil_iff, or_false]
      rintro key (rfl | rfl | rfl | rfl)
      · exact str_good ‹J.str _ "law" _ = _›
      · exact nat_good ‹J.nat _ "population" _ = _›
      · exact ⟨_, arr_ok ‹J.arr _ "units" _ = _›, good_of_asNat_mapM ‹Array.mapM _ _ = Except.ok _›⟩
      · exact nat_good ‹J.nat _ "k" _ = _›
    refine ⟨good_obj_of hm hL (by simpa using hc) hk, ?_⟩
    first
      | (have hpc := ‹(pure #[] : V (Array ℕ)) = Except.ok _›
         cases hpc
         rfl)
      | exfalso
        have hcl := (mem_of_getObjVal hm ‹(Json.obj _).getObjVal? "closure" = Except.ok _›).1
        have := key_mem hm hL (fun key hkey => by
            obtain ⟨v, hv, -⟩ := hk key hkey
            exact (mem_of_getObjVal hm hv).1)
          (by rw [← Std.TreeMap.Raw.size_toArray hm]; simpa using hc) hcl
        simp at this

end FlockComplete.ZkTables
