import Proofs.Flock.Claims.Complete.ZkTables.RingSwitch
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.FailClosed.PublicTies

/-!
# What the masked record needs of the statement, at one table

`stmt_of_one`: a masked statement in scope whose drawn setup has one table gives `StmtAt`'s facts under the JSON
library's specification (`JsonSpec`), from the setup alone. So `StmtAt` holds at every good case of one table
(`Case.stmtOne`), and with `Case.stmtAt` at every good case (`stmtAt_all`); `StmtOkZk` holds (`stmtOkZk`); and
`completeRel`, `completeJRel` and `complete_zk_relation` hold without it (`completeRelS`, `completeJRelS`,
`complete_zk_relationS`). `complete_json` and `complete_jsonJ` are `completeRelS` and `completeJRelS` in
`Claims.Flock.Complete`'s form, at the masked record's pinned prover and error, with `Out64` the claim's own
antecedent.

* A setup of one table is `setupOne`'s (`setupTables_okAny`: one of several has more).
* Its public header is the population's with the draw put in (`HmRow.drawn`), so its `unit_draw` is the statement's
  draw (`unitDraw_of`), which decodes to the statement's instances (`PublicTies.instances`); its `Σ` is a SHA-512 digest
  and its `Hello` is `helloOf`'s (`setupOne_stmt`).
* The setup checked the draw's law (`ProvedScope.drawLaw`), under `--zk` a stratified or a subset draw. Either, when it
  decodes, is an object of exactly the keys it reads, each a string, a natural number, an array of them, or strata that
  decode (`good_stratified`, `good_subset`), so it is printable.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open FlockSoundness.Refine
open Lean (Json)

open private stratumOfJson from Flock.Draw

/-! ## The `J` readers' values -/

private theorem obj_of_getObjVal {j : Json} {k : String} {v : Json} (h : j.getObjVal? k = .ok v) :
    ∃ m, j = .obj m := by
  cases j with
  | obj m => exact ⟨m, rfl⟩
  | _ => cases h

private theorem asNat_ok {j : Json} {c : String} {n : ℕ} (h : J.asNat j c = .ok n) :
    j = .num ⟨Int.ofNat n, 0⟩ := by
  unfold J.asNat at h
  split at h
  · rename_i _ hg; cases h; exact getNat_ok hg
  · cases h

private theorem nat_ok {j : Json} {k c : String} {n : ℕ} (h : J.nat j k c = .ok n) :
    j.getObjVal? k = .ok (.num ⟨Int.ofNat n, 0⟩) := by
  unfold J.nat at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [get_ok hv, asNat_ok h]

private theorem str_ok {j : Json} {k c : String} {s : String} (h : J.str j k c = .ok s) :
    j.getObjVal? k = .ok (.str s) := by
  unfold J.str at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [get_ok hv]
  unfold J.asStr at h
  split at h
  · rename_i _ hg
    cases h
    unfold Json.getStr? at hg
    split at hg
    · cases hg; rfl
    · cases hg
  · cases h

private theorem mapM_ok_mem {α β : Type} {f : α → V β} {a : Array α} {b : Array β} (h : a.mapM f = .ok b) :
    ∀ x ∈ a, ∃ y, f x = .ok y := by
  intro x hx
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
  obtain ⟨hs, hb⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
  exact ⟨_, hb i hi (hs ▸ hi)⟩

private theorem getStr_ok {j : Json} {s : String} (h : j.getStr? = .ok s) : j = .str s := by
  cases j with
  | str s' => cases h; rfl
  | _ => cases h

private theorem isOk_ok {j : Json} {k : String} (h : (j.getObjVal? k).isOk = true) : ∃ v, j.getObjVal? k = .ok v := by
  cases e : j.getObjVal? k with
  | ok v => exact ⟨v, rfl⟩
  | error _ => rw [e] at h; cases h

/-! ## What the setup of one table set up -/

/-- A drawn session's setup checked its draw's law (`ProvedScope.drawLaw`). -/
private theorem setup_drawLaw {s : Ins} {p : SessionSpec × Array Flock.Stmt} (h : s.setup = .ok p)
    (hd : s.read.draw.isSome = true) :
    ProvedScope.drawLaw s.read.zk s.read.tags false s.read.draw = .ok () := by
  unfold Ins.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | assumption

private theorem hmOut_loadPublic {tags : Tags} {c : Circuit} {f : ByteArray} {pop : Public}
    (h : HmOut.loadPublic tags c f = .ok pop) : HmRow.loadPublic tags { c with outPorts := #[] } f = .ok pop := by
  unfold HmOut.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

set_option maxHeartbeats 4000000 in
/-- What `Stmt.setupH` set up: its public file loaded (at a draw, `HmRow.drawn` of it) and passed `HmRow.checkPublic`,
its instances, `Σ` and its `Hello`. -/
private theorem setupH_stmt {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    (∃ pop, HmRow.loadPublic tags st.c pf = .ok pop ∧ ∀ d, draw = some d → HmRow.drawn st.c pop d = .ok st.pub) ∧
      HmRow.checkPublic st.c st.pub = .ok () ∧ st.n = st.pub.n ∧
      st.sigma = Sha512.hash (tags.sigmaTag.toUTF8 ++ st.c.sha ++ st.digest ++ st.pub.sha) ∧
      Flock.helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello := by
  cases draw <;> unfold Stmt.setupH at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨⟨_, by assumption, fun d hd => by first | (cases hd; done) | (cases hd; assumption)⟩, by assumption, rfl,
      rfl, by assumption⟩

/-- **What a table's setup set up** (`setupOne`, either output mode): the public file a public-file load returned, at a
draw that file's `HmRow.drawn`; the header's draw decodes to the statement's instances; a SHA-512 `Σ`; and the
`Hello`. -/
private theorem setupOne_stmt {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}
    (h : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    (∃ c pop, HmRow.loadPublic tags c pf = .ok pop ∧ ∀ d, draw = some d → HmRow.drawn c pop d = .ok st.pub) ∧
      (∀ d, st.pub.header.getObjVal? "unit_draw" = .ok d →
        ∃ u, Draw.UnitDraw.ofJson d = .ok u ∧ u.proved.size = st.n) ∧
      0 < st.sigma.size ∧ Flock.helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello := by
  rcases Bool.eq_false_or_eq_true tags.hiddenOutputs with hh | hh
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh] at h
    have S := FlockSoundness.Discharge.Hidden.setupHidden_spec' h
    obtain ⟨pop, hl, -, hd⟩ := S.pub
    have T := FlockSoundness.Discharge.FailClosed.ties_of_check (hidden := true) (Or.inr ⟨rfl, S.checkPublic⟩)
    refine ⟨⟨_, pop, hmOut_loadPublic hl, hd⟩, fun d hd' => ?_, by rw [S.sigma, Sha512.hash_size]; omega, S.hello⟩
    obtain ⟨u, hu, -, hn⟩ := T.instances d hd'
    exact ⟨u, hu, by rw [S.n, hn]⟩
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_public hh] at h
    obtain ⟨⟨pop, hl, hd⟩, hc, hn, hs, hhello⟩ := setupH_stmt h
    have T := FlockSoundness.Discharge.FailClosed.ties_of_check (hidden := false) (Or.inl ⟨rfl, hc⟩)
    refine ⟨⟨_, pop, hl, hd⟩, fun d hd' => ?_, by rw [hs, Sha512.hash_size]; omega, hhello⟩
    obtain ⟨u, hu, -, hn'⟩ := T.instances d hd'
    exact ⟨u, hu, by rw [hn, hn']⟩

/-! ## The public header's draw -/

set_option maxHeartbeats 4000000 in
/-- The public file's header is a rebuilt object (`Flock.norm` of what `Json.parse` read). -/
private theorem loadPublic_header {tags : Tags} {c : Circuit} {f : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c f = .ok pop) : (∃ j, pop.header = Flock.norm j) ∧ ∃ m, pop.header = .obj m := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hpj := ‹(pure _ : V Json) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hpj
    subst hpj
    exact ⟨⟨_, rfl⟩, obj_of_getObjVal (str_ok (k := "format") (by assumption))⟩

set_option maxHeartbeats 4000000 in
/-- `HmRow.drawn` puts the draw in the population's header. -/
private theorem drawn_header {c : Circuit} {pub pub' : Public} {d : Json} {m : Std.TreeMap.Raw String Json compare}
    (hm : m.WF) (hp : pub.header = .obj m) (h : HmRow.drawn c pub d = .ok pub') :
    pub'.header.getObjVal? "unit_draw" = .ok d := by
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    dsimp only
    rw [hp]
    simp only [Json.setObjVal!]
    apply getObjVal_of_get
    rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert,
      Std.TreeMap.Raw.getElem?_insert hm.insert.insert, Std.TreeMap.Raw.getElem?_insert hm.insert,
      Std.TreeMap.Raw.getElem?_insert hm]
    simp

/-- **The statement's header names its draw**: a draw's statement over a loaded public file has that draw as its
`unit_draw`. -/
private theorem unitDraw_of {tags : Tags} {c : Circuit} {pf : ByteArray} {pop pub : Public} {d : Json}
    (hl : HmRow.loadPublic tags c pf = .ok pop) (hd : HmRow.drawn c pop d = .ok pub) :
    pub.header.getObjVal? "unit_draw" = .ok d := by
  obtain ⟨⟨j, hj⟩, m, hm⟩ := loadPublic_header hl
  have hw := FlockDeploy.wf_norm j
  rw [← hj, hm] at hw
  cases hw with
  | obj hwf _ => exact drawn_header hwf hm hd

/-! ## A stratified draw that decodes is printable -/

/-- A stratum's `[lo, hi)` range is a pair of natural numbers. -/
private theorem good_range {p : Array Json} (hs : p.size = 2) {c₀ c₁ : String} {lo hi : ℕ}
    (h₀ : J.asNat p[0]! c₀ = .ok lo) (h₁ : J.asNat p[1]! c₁ = .ok hi) : Good (.arr p) := by
  refine .arr fun x hx => ?_
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
  have : i = 0 ∨ i = 1 := by omega
  rcases this with rfl | rfl
  · rw [← getElem!_pos p 0 hi]; exact asNat_good h₀
  · rw [← getElem!_pos p 1 hi]; exact asNat_good h₁

set_option maxHeartbeats 4000000 in
/-- **A stratum that decodes is printable**: an object of exactly `template`, `units` and `k`, a string, ranges of two
natural numbers, and a natural number. -/
private theorem good_stratum {j : Json} (hw : Wf j) {s : Draw.Stratum} (h : stratumOfJson j = .ok s) : Good j := by
  unfold stratumOfJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hu : ∀ r ∈ _, Good r := fun r hr => by
      obtain ⟨y, h⟩ := mapM_ok_mem ‹Array.mapM _ _ = Except.ok _› r hr
      try dsimp only at h
      try simp only [throw_bind'] at h
      repeat' walk_step
      all_goals
        exact good_range (by simpa using ‹¬ (_ != 2) = true›) ‹J.asNat _[0]! _ = _› ‹J.asNat _[1]! _ = _›
    have hj := getObj_ok ‹j.getObj? = _›
    subst hj
    cases hw with
    | obj hm _ =>
    have hc := ‹¬ (_ != 3 || _) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not, List.any_cons, List.any_nil, Bool.or_false,
      Bool.not_eq_true', Bool.not_eq_false] at hc
    obtain ⟨hs, ht, hun, hk⟩ := hc
    refine good_obj_of hm (L := ["template", "units", "k"]) (by decide) (by simpa using hs) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false]
    rintro key (rfl | rfl | rfl)
    · obtain ⟨v, hv⟩ := isOk_ok ht
      refine ⟨v, hv, ?_⟩
      have hs' := ‹(Json.obj _).getObjValAs? String "template" = _›
      simp only [Json.getObjValAs?, Json.getObjValD, hv, Except.toOption, Option.getD_some] at hs'
      rw [getStr_ok hs']
      exact .str _
    · exact ⟨_, ‹(Json.obj _).getObjVal? "units" = _›, .arr hu⟩
    · obtain ⟨v, hv⟩ := isOk_ok hk
      refine ⟨v, hv, ?_⟩
      have hn := ‹J.asNat _ "U2: a stratum's k" = _›
      simp only [hv, Except.toOption, Option.getD_some] at hn
      exact asNat_good hn

set_option maxHeartbeats 4000000 in
/-- **A stratified draw that decodes is printable**: an object of exactly `law`, `population`, `units`, `k` and
`strata`, a string, natural numbers, an array of them, and strata that decode. -/
private theorem good_stratified {j : Json} (hw : Wf j) {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D)
    {K : ℕ} {strata : Array Draw.Stratum} (hl : D.law = .stratified K strata) : Good j := by
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
    have hj := getObj_ok ‹j.getObj? = _›
    subst hj
    cases hw with
    | obj hm hwv =>
    have hc := ‹¬ (_ != 5 || _ || _) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hc
    have hss := ‹(Json.obj _).getObjVal? "strata" = Except.ok (Json.arr _)›
    refine good_obj_of hm (L := ["law", "population", "units", "k", "strata"]) (by decide) (by simpa using hc.1.1) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false]
    rintro key (rfl | rfl | rfl | rfl | rfl)
    · exact ⟨_, str_ok ‹J.str _ "law" _ = _›, .str _⟩
    · exact ⟨_, nat_ok ‹J.nat _ "population" _ = _›, .num _⟩
    · exact ⟨_, arr_ok ‹J.arr _ "units" _ = _›, good_of_asNat_mapM ‹Array.mapM _ _ = Except.ok _›⟩
    · exact ⟨_, nat_ok ‹J.nat _ "k" _ = _›, .num _⟩
    · refine ⟨_, hss, .arr fun x hx => ?_⟩
      obtain ⟨_, hy⟩ := mapM_ok_mem ‹Array.mapM stratumOfJson _ = Except.ok _› x hx
      have hw' := hwv "strata" _ (mem_of_getObjVal hm hss).2
      cases hw' with
      | arr ha => exact good_stratum (ha x hx) hy

/-- Under `--zk` the setup admits a stratified or a subset draw. -/
private theorem law_of_lawWhy {tags : Tags} {D : Draw.UnitDraw} (h : ProvedScope.lawWhy true tags false D = none) :
    (∃ K strata, D.law = .stratified K strata) ∨ ∃ k, D.law = .subset k := by
  revert h
  cases hl : D.law with
  | stratified K s => exact fun _ => .inl ⟨K, s, rfl⟩
  | subset k => exact fun _ => .inr ⟨k, rfl⟩
  | _ => simp [ProvedScope.lawWhy, hl]

/-! ## The statement at one table -/

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **A masked setup of one table** is `setupOne`'s, at the statement's tags, coins and draw. -/
theorem setupOne_of {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hsc : Claims.Flock.Scope (deployZk rel) x) (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) :
    ∃ st, sts = #[st] ∧ spec = st.spec ∧
      FlockSoundness.Discharge.HiddenExec.setupOne s.read.tags s.read.I.circuitFile s.read.I.publicFile s.read.coins
        s.read.I.tables s.read.draw s.read.I.partition s.read.I.program = .ok st := by
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny (setupTables_of hsc hs) with
    ⟨-, st, hst, hspec, hsts⟩ | ⟨hc, hsz, -⟩
  · exact ⟨st, hsts, hspec, hst⟩
  · omega

/-- **What the masked record needs of the statement, at one table**, under the JSON library's specification: a masked
statement in scope whose drawn setup has one table has a non-empty `Σ`, a printable ASCII draw that decodes to its
instances and is the statement's, a `Hello` that reads back with `coins` an object, printable and ASCII with the
prover's nonce, and an ASCII coin tree spec. -/
theorem stmt_of_one (S : JsonSpec) {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hsc : Claims.Flock.Scope (deployZk rel) x) (hs : setupOf true x = some (s, spec, sts))
    (hdr : s.read.draw.isSome = true) (h1 : sts.size = 1) :
    0 < spec.sigma.size ∧ DrawOk spec ∧ spec.unitDraw = s.read.draw ∧
      ∃ m kvs : Std.TreeMap.Raw String Json compare, Json.parse spec.hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
        m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) ∧ Ascii (helloJ (.obj m)) ∧
        ∀ ct, spec.coinTree = some ct → Ascii ct.toJson := by
  obtain ⟨st, -, hspec, hst⟩ := setupOne_of hsc hs h1
  obtain ⟨⟨c, pop, hl, hd⟩, hinst, hsig, hhello, hct0⟩ := FlockComplete.JsonRT.setupOne_stmt hst
  have hzk : s.read.zk = true := readOf_zk (x := x) (by
    unfold readOf
    rw [(setupOf_some hs).1]
    rfl)
  have hco : s.read.coins = .tree (Zk.coinSpec (s.read.I.tags.tableNames s.read.I.count)) := by
    simp [Readable.Read.coins, hzk, FlockSoundness.Discharge.ZkExec.zkCoins]
  rw [hco] at hhello
  obtain ⟨d, hdd⟩ := Option.isSome_iff_exists.1 hdr
  have hu := unitDraw_of hl (hd d hdd)
  have hU : spec.unitDraw = some d := by
    rw [hspec]
    simp [Flock.Stmt.spec, hu, Except.toOption]
  obtain ⟨m, kvs, hp, hm, hkv, hmg, hg, ha, hA⟩ := hello_ok S hhello
  refine ⟨by rw [hspec]; exact hsig, ?_, by rw [hU, hdd], m, kvs, by rw [hspec]; exact hp, hm, hkv, hmg, hg, ha,
    fun ct hc' => ?_⟩
  swap
  · rw [hspec] at hc'
    change st.coinTree = _ at hc'
    rw [hct0, hco] at hc'
    simp only [Coins.tree?, Option.some.injEq] at hc'
    subst hc'
    exact hA
  unfold DrawOk
  rw [hU]
  dsimp only
  obtain ⟨D, hD, hn⟩ := hinst d hu
  have hdl := setup_drawLaw (setupOf_some hs).2 hdr
  rw [hdd, hzk] at hdl
  simp only [ProvedScope.drawLaw, hD] at hdl
  have hlw : ProvedScope.lawWhy true s.read.tags false D = none := by
    cases e : ProvedScope.lawWhy true s.read.tags false D with
    | none => rfl
    | some _ => rw [e] at hdl; cases hdl
  have hw := (insOf_draw_norm (setupOf_some hs).1 hdd).2
  refine ⟨ofJson_obj hD, ?_, ascii_of_drawn hl (hd d hdd), D, hD, by rw [hspec]; exact hn⟩
  rcases law_of_lawWhy hlw with ⟨K, strata, hK⟩ | ⟨k, hk⟩
  · exact good_stratified hw hD hK
  · exact (good_subset hw hD hk).1

/-- **`StmtAt` at every good case of one table**, under the JSON library's specification. -/
theorem Case.stmtOne (S : JsonSpec) (c : Case H rel) (h1 : c.sts.size = 1) : StmtAt c :=
  stmt_of_one S c.scope c.setup c.ready_parts.1 h1

/-- **`StmtAt` at every good case**, under the JSON library's specification: `Case.stmtOne` at one table,
`Case.stmtAt` at several. -/
theorem stmtAt_all (S : JsonSpec) (c : Case H rel) : StmtAt c :=
  c.one_or_several.elim (c.stmtOne S) (c.stmtAt S)

/-- **`StmtOkZk` holds**, under the JSON library's specification: a good masked case has one table. -/
theorem stmtOkZk (S : JsonSpec) : StmtOkZk H rel :=
  fun c => stmt_of_one S c.scope c.setup c.ready_parts.2.1 c.ready_parts.1

/-! ## Completeness without the statement's premise -/

/-- **Flock's masked verifier is complete at the deployed relation** for the honest masked prover on each table's
share, with error `δc H`, at any number of tables: `completeRel` with the statement's premise proved
(`stmtAt_all`). -/
theorem completeRelS (H : Claims.Msg → Claims.Msg) (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H)
    (hPR : ∀ c : Case H Readable.relation, ProverRunsAt c) (hR : ∀ c : Case H Readable.relation, RepShapedAt c)
    (hC : ∀ o : Run H Readable.relation, LgInnerAt H Readable.relation o) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  completeR H Readable.relation S h64 (stmtAt_all S) hPR hR Run.rsExtras hC hMR maskFree_relation

/-- **Flock's completeness at the deployed masked verifier**, at the deployed relation, of the record's reference
prover (`(deployZk Readable.relation).P`, `honestPR`) at the record's error (`(deployZk Readable.relation).δc`, `δc H`),
at any number of tables: `completeRelS`, the oracle's 64-byte outputs (`Out64`) being `Claims.Complete`'s own
antecedent. -/
theorem complete_json (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (hPR : ∀ c : Case H Readable.relation, ProverRunsAt c) (hR : ∀ c : Case H Readable.relation, RepShapedAt c)
    (hC : ∀ o : Run H Readable.relation, LgInnerAt H Readable.relation o) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) := fun h64 =>
  completeRelS H S h64 hPR hR hC hMR h64

/-- **Flock's masked verifier is complete at the deployed relation** for the honest masked prover on each table's
share, with error `δc H`, at any number of tables: `completeJRel` with `StmtOkZk` proved (`stmtOkZk`). -/
theorem completeJRelS (H : Claims.Msg → Claims.Msg) (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H)
    (hR : JsonRT.RepShapedZk H Readable.relation) (hPR : JsonRT.ProverRunsZk H Readable.relation)
    (hLg : FlockComplete.LigeritoZk H Readable.relation) (hIn : FlockComplete.Inner H Readable.relation)
    (hPRJ : ProverRunsJ H Readable.relation) (hRJ : RepShapedJ H Readable.relation)
    (hCJ : LgInnerJ H Readable.relation) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  completeJRel H S h64 (stmtOkZk S) hR hPR hLg hIn hPRJ hRJ hCJ hMR

/-- **Flock's completeness at the deployed masked verifier**, at the deployed relation, of the record's reference
prover at the record's error, at any number of tables: `completeJRelS`, from the one-table premises and the same
three at several tables (`ProverRunsJ`, `RepShapedJ`, `LgInnerJ`), with `Out64` `Claims.Complete`'s own
antecedent. -/
theorem complete_jsonJ (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (hR : JsonRT.RepShapedZk H Readable.relation) (hPR : JsonRT.ProverRunsZk H Readable.relation)
    (hLg : FlockComplete.LigeritoZk H Readable.relation) (hIn : FlockComplete.Inner H Readable.relation)
    (hPRJ : ProverRunsJ H Readable.relation) (hRJ : RepShapedJ H Readable.relation)
    (hCJ : LgInnerJ H Readable.relation) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  fun h64 => completeJRelS H S h64 hR hPR hLg hIn hPRJ hRJ hCJ hMR h64

/-- **Flock's masked verifier is complete at the deployed relation** for `honestPZk` with error `δcZk H`:
`complete_zk_relation`, which proves `StmtOkZk` from `S` itself. -/
theorem complete_zk_relationS (H : Claims.Msg → Claims.Msg) (S : JsonRT.JsonSpec)
    (hR : JsonRT.RepShapedZk H Readable.relation)
    (hPR : JsonRT.ProverRunsZk H Readable.relation) (hLg : FlockComplete.LigeritoZk H Readable.relation)
    (hIn : FlockComplete.Inner H Readable.relation) (hMR : MaskRank.MaskRankBound) :
    CompleteWith H (deployZk Readable.relation) honestPZk (δcZk H) :=
  FlockComplete.complete_zk_relation H S hR hPR hLg hIn hMR

end FlockComplete.ZkTables
