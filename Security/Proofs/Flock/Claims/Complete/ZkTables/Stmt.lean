import Proofs.Flock.Claims.Complete.ZkTables.Reps
import Proofs.Flock.Claims.Complete.Zk.Json.Record
import Proofs.Flock.Verifier.Hash

/-!
# What the masked record needs of the statement, at several tables

`Case.stmtAt`: at every good case of several tables, `StmtAt` holds under the JSON library's specification
(`JsonSpec`), from the case's setup alone (`Case.setupTables`: the scope's statements hold no public inputs). So
`StmtOkJ` holds (`stmtOkJ`), and `completeJS`, `completeAllS` are `completeJO`, `completeAll` without it.

* A session of several tables (`setupTables_spec`) keeps the statement's draw, which is a subset or a stratified draw
  (the setup refuses any other law at several tables) whose units are the session's instances; its `Σ` is a SHA-512
  digest; and its `Hello` is `helloOf`'s at the tables' names.
* A subset draw that decodes is an object of exactly `law`, `population`, `k` and `units` (`UnitDraw.ofJson`'s key
  count), a string, natural numbers and an array of them, so it is printable and has no closure (`good_subset`); a
  stratified one has exactly `law`, `population`, `units`, `k` and `strata`, so it is printable
  (`JsonRT.good_stratified`) and has no closure either (`closure_stratified`); the statement's draw is JSON that
  `Json.parse` read, so it is well formed.
* A masked statement's coins are its keyed coin tree (`zkCoins`), so the `Hello`'s `coins` entry is the tree's spec with
  a nonce, an object; `canon` prints the `Hello` and `parse` reads back an alike object (`hello_ok`), which stays
  printable with the prover's nonce put in (`good_of_same`, `good_insert`). `canon` printed it, so it is ASCII
  (`ascii_of_canon`), as is the coin tree's spec it carries, and it stays so with the nonce put in (`ascii_insert`).
* The setup of several tables never prints the statement's draw, so its ASCII is the readiness check's
  (`Case.drawAscii`).
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
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) ∧ Ascii (helloJ (.obj m)) ∧ Ascii ct.toJson := by
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
  have ham : Ascii (.obj m) := ascii_of_same hs (ascii_of_canon hh)
  have hak : Ascii (.obj kvs) := ascii_get hm ham hmg
  obtain ⟨hA, hnk⟩ := ascii_coins (ct := ct) (by rw [hM]; exact ascii_of_same_rev hcs hak)
  have e : helloJ (.obj m) = .obj (m.insert "coins" (.obj (kvs.insert "nonce" (Json.str (zeros 64))))) := by
    simp only [helloJ, Json.getObjVal?, hmg, Json.setObjVal!]
    rfl
  refine ⟨m, kvs, hp, hm, hkv, hmg, ?_, ?_, hA⟩
  · cases hgm with
    | obj _ hgm =>
    cases hgk with
    | obj _ hgk =>
    rw [e]
    exact good_insert hm hgm (good_insert hkv hgk (.str _))
  · rw [e]
    exact ascii_insert hm ham (ascii_key hm ham hmg) (ascii_insert hkv hak hnk (.str (ascii_zeros 64)))

/-- **The masked `Hello` reads back** as an object whose `coins` are an object, and the prover's version, with its
nonce put in, is printable and ASCII, as is the coin tree's spec. -/
theorem hello_ok (S : JsonSpec) {tags : Tags} {sigma : ByteArray} {mPts : ℕ} {ct : CoinTree.Spec}
    {names : Array String} {hello : String} (hh : Flock.helloOf tags sigma mPts (.tree ct) names = .ok hello) :
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Json.parse hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) ∧ Ascii (helloJ (.obj m)) ∧ Ascii ct.toJson := by
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

theorem mem_of_getObjVal {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
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

theorem getObjVal_of_get {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json}
    (h : m.get? k = some v) : (Json.obj m).getObjVal? k = .ok v := by
  simp only [Json.getObjVal?, h]
  rfl

theorem getNat_ok {j : Json} {n : ℕ} (h : j.getNat? = .ok n) : j = .num ⟨Int.ofNat n, 0⟩ := by
  unfold Json.getNat? at h
  split at h
  · cases h; rfl
  · cases h

theorem asNat_good {j : Json} {c : String} {n : ℕ} (h : J.asNat j c = .ok n) : Good j := by
  unfold J.asNat at h
  split at h
  · rename_i _ hg; cases h; rw [getNat_ok hg]; exact .num _
  · cases h

theorem get_ok {j : Json} {k c : String} {v : Json} (h : J.get j k c = .ok v) : j.getObjVal? k = .ok v := by
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

theorem arr_ok {j : Json} {k c : String} {a : Array Json} (h : J.arr j k c = .ok a) :
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

theorem getObj_ok {j : Json} {m : Std.TreeMap.Raw String Json compare} (h : j.getObj? = .ok m) :
    j = .obj m := by
  cases j with
  | obj m' => cases h; rfl
  | _ => cases h

theorem good_of_asNat_mapM {a : Array Json} {c : String} {b : Array ℕ}
    (h : a.mapM (J.asNat · c) = .ok b) : Good (.arr a) :=
  .arr fun x hx => by
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
    obtain ⟨hs, hb⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
    exact asNat_good (hb i hi (hs ▸ hi))

/-- An object with exactly the keys of `L`, each holding a printable value, is printable. -/
theorem good_obj_of {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {L : List String} (hL : L.Nodup)
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

set_option maxHeartbeats 4000000 in
/-- **A stratified draw that decodes has no closure**: an object of exactly `law`, `population`, `units`, `k` and
`strata`, so it holds no `closure` key. -/
theorem closure_stratified {j : Json} (hw : Wf j) {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) {K : ℕ}
    {strata : Array Draw.Stratum} (hl : D.law = .stratified K strata) : D.closure = #[] := by
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
    have hc := ‹¬ (_ != 5 || _ || _) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hc
    have hL : ["law", "population", "units", "k", "strata"].Nodup := by decide
    have hk : ∀ key ∈ ["law", "population", "units", "k", "strata"], key ∈ m := by
      simp only [List.mem_cons, List.mem_nil_iff, or_false]
      rintro key (rfl | rfl | rfl | rfl | rfl)
      · obtain ⟨v, hv, -⟩ := str_good ‹J.str _ "law" _ = _›
        exact (mem_of_getObjVal hm hv).1
      · obtain ⟨v, hv, -⟩ := nat_good ‹J.nat _ "population" _ = _›
        exact (mem_of_getObjVal hm hv).1
      · exact (mem_of_getObjVal hm (arr_ok ‹J.arr _ "units" _ = _›)).1
      · obtain ⟨v, hv, -⟩ := nat_good ‹J.nat _ "k" _ = _›
        exact (mem_of_getObjVal hm hv).1
      · exact (mem_of_getObjVal hm ‹(Json.obj _).getObjVal? "strata" = Except.ok (Json.arr _)›).1
    first
      | (have hpc := ‹(pure #[] : V (Array ℕ)) = Except.ok _›
         cases hpc
         rfl)
      | exfalso
        have hcl := (mem_of_getObjVal hm ‹(Json.obj _).getObjVal? "closure" = Except.ok _›).1
        have := key_mem hm hL hk (by rw [← Std.TreeMap.Raw.size_toArray hm]; simpa using hc.1.1) hcl
        simp at this

/-! ## The setup of several tables -/

private theorem mapM_some_ok {d : Json} {u? : Option Draw.UnitDraw}
    (h : Option.mapM Draw.UnitDraw.ofJson (some d) = .ok u?) : ∃ u, u? = some u ∧ Draw.UnitDraw.ofJson d = .ok u := by
  simp only [Option.mapM_some] at h
  cases hd : Draw.UnitDraw.ofJson d with
  | error e => rw [hd] at h; cases h
  | ok u => rw [hd] at h; cases h; exact ⟨u, rfl, rfl⟩

set_option maxHeartbeats 4000000 in
/-- **The session spec of several tables**: its draw the statement's, a subset or a stratified draw whose units are its
instances, its
`Σ` a SHA-512 digest, and its `Hello` `helloOf`'s at the tables' names. -/
theorem setupTables_spec {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {draw partition : Option Json} {program : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))}
    {spec : SessionSpec} {sts : Array Flock.Stmt}
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) (hs : 1 < sts.size) :
    spec.unitDraw = draw ∧ spec.sigma.size = 64 ∧
      (∀ d, draw = some d → ∃ D, Draw.UnitDraw.ofJson d = .ok D ∧
        ((∃ k, D.law = .subset k) ∨ ∃ K st, D.law = .stratified K st) ∧ spec.instances = D.units.size) ∧
      Flock.helloOf tags spec.sigma spec.mPts coins (tags.tableNames count) = .ok spec.hello ∧
      spec.coinTree = coins.tree? := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals first
    | exact absurd hs (Nat.lt_irrefl 1)
    | refine ⟨rfl, Sha512.hash_size _, ?_, ‹_›, rfl⟩
  all_goals
    intro d hd
    subst hd
    obtain ⟨u, hu, hd⟩ := mapM_some_ok ‹Option.mapM Draw.UnitDraw.ofJson (some d) = _›
    cases hu <;> first
      | exact absurd ‹some _ = none› (by simp)
      | exact (‹∀ u, some _ = some u → False› _ rfl).elim
      | (cases ‹some _ = some _›
         first
           | exact ⟨_, hd, .inl ⟨_, ‹_ = Draw.Law.subset _›⟩, rfl⟩
           | exact ⟨_, hd, .inr ⟨_, _, ‹_ = Draw.Law.stratified _ _›⟩, rfl⟩)

/-! ## The statement at a good case of several tables -/


theorem ofJson_obj {j : Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) : j ≠ Json.null := by
  rintro rfl
  cases h

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **A masked statement in scope is set up** by `Stmt.setupTables` on what it reads, with no public inputs. -/
theorem setupTables_of {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hsc : Claims.Flock.Scope (deployZk rel) x) (hs : setupOf true x = some (s, spec, sts)) :
    Stmt.setupTables s.read.tags s.read.I.circuitFile s.read.I.publicFile s.read.coins s.read.I.count
      s.read.I.tables s.read.draw s.read.I.partition s.read.I.program s.read.I.work s.read.I.stratified
      s.read.I.closure = .ok (spec, sts) := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) x).1 hsc
  have hi := (setupOf_some hs).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some hs).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rw [ht] at hset
  exact hset

/-- **A good case's setup** is `Stmt.setupTables` on what the statement reads, with no public inputs. -/
theorem Case.setupTables (c : Case H rel) :
    Stmt.setupTables c.s.read.tags c.s.read.I.circuitFile c.s.read.I.publicFile c.s.read.coins c.s.read.I.count
      c.s.read.I.tables c.s.read.draw c.s.read.I.partition c.s.read.I.program c.s.read.I.work c.s.read.I.stratified
      c.s.read.I.closure = .ok (c.spec, c.sts) :=
  setupTables_of c.scope c.setup

/-- **What the masked record needs of the statement holds at every good case of several tables** (`StmtAt`), under
the JSON library's specification: `Σ` is a SHA-512 digest, the session's draw is the statement's, a subset or a
stratified draw that is
printable and decodes to the session's instances, and its `Hello` reads back with `coins` an object. -/
theorem Case.stmtAt (S : JsonSpec) (c : Case H rel) (hc : 1 < c.sts.size) : StmtAt c := by
  obtain ⟨hu, hsig, hd, hh, hct⟩ := setupTables_spec c.setupTables hc
  have hzk : c.s.read.zk = true := readOf_zk (x := c.x) (by
    unfold readOf
    rw [(setupOf_some c.setup).1]
    rfl)
  have hco : c.s.read.coins = .tree (Zk.coinSpec (c.s.read.I.tags.tableNames c.s.read.I.count)) := by
    simp [Readable.Read.coins, hzk, FlockSoundness.Discharge.ZkExec.zkCoins]
  rw [hco] at hh hct
  obtain ⟨m, kvs, hp, hm, hkv, hmg, hg, ha, hA⟩ := hello_ok S hh
  refine ⟨by rw [hsig]; omega, ?_, hu, m, kvs, hp, hm, hkv, hmg, hg, ha, fun ct hc' => ?_⟩
  swap
  · rw [hct] at hc'
    simp only [Coins.tree?, Option.some.injEq] at hc'
    subst hc'
    exact hA
  unfold DrawOk
  rw [hu]
  cases hdr : c.s.read.draw with
  | none => trivial
  | some d =>
    obtain ⟨D, hD, hlaw, hn⟩ := hd d hdr
    have hw := (insOf_draw_norm (setupOf_some c.setup).1 hdr).2
    obtain ⟨hgd, hcl⟩ : Good d ∧ D.closure = #[] := by
      rcases hlaw with ⟨k, hk⟩ | ⟨K, st, hK⟩
      · exact good_subset hw hD hk
      · exact ⟨JsonRT.good_stratified hw hD hK, closure_stratified hw hD hK⟩
    refine ⟨ofJson_obj hD, hgd, c.drawAscii hc d hdr, D, hD, ?_⟩
    rw [hn]
    simp [Draw.UnitDraw.proved, hcl]

/-- **`StmtOkJ` holds**, under the JSON library's specification. -/
theorem stmtOkJ (S : JsonSpec) : StmtOkJ H rel := fun c hc => c.stmtAt S hc

/-- `StmtAt` at every good case: one table's is `StmtOkZk`'s, several's `Case.stmtAt`. -/
theorem stmtAt_of (S : JsonSpec) (hS : StmtOkZk H rel) (c : Case H rel) : StmtAt c :=
  c.one_or_several.elim (fun h1 => hS (c.toZk h1)) (c.stmtAt S)

/-! ## Completeness without the statement's premise at several tables -/

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables: `completeJO` with the statement's premise at several tables proved (`stmtOkJ`). -/
theorem completeJS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel) (hR : JsonRT.RepShapedZk H rel)
    (hPR : JsonRT.ProverRunsZk H rel) (hRS : FlockComplete.RingSwitchZk H rel) (hLg : FlockComplete.LigeritoZk H rel)
    (hIn : FlockComplete.Inner H rel) (hPRJ : ProverRunsJ H rel) (hRJ : RepShapedJ H rel)
    (hCJ : RepChecksJ H rel) (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) :
    Claims.Flock.Complete H (deployZk rel) :=
  completeJO H rel S h64 hS hR hPR hRS hLg hIn (stmtOkJ S) hPRJ hRJ hCJ hMR hMF

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables: `completeAll` with the statement's premise `StmtOkZk`'s at one table and proved at several. -/
theorem completeAllS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel)
    (hPR : ∀ c : Case H rel, ProverRunsAt c) (hR : ∀ c : Case H rel, RepShapedAt c)
    (hC : ∀ o : Run H rel, RepChecksAt H rel o) (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) :
    Claims.Flock.Complete H (deployZk rel) :=
  completeAll H rel S h64 (stmtAt_of S hS) hPR hR hC hMR hMF

end FlockComplete.ZkTables
