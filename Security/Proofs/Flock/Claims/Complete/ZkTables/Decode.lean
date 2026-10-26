import Proofs.Flock.Claims.Complete.ZkTables.Names
import Proofs.Flock.Claims.Complete.Zk.Json.Decode
import Proofs.Flock.Claims.JsonNorm

/-!
# The honest masked record decodes and reads live, at any number of tables

`Zk/Json/Decode.lean` at every number of tables, the tables named distinctly (`Names`): the masked record of one
`root_B` per table is printable (`good_recordObjZkJ`); a printable object with the plain record's fields decodes as
`sessOf` (`decodeZk_ofFieldsJ`: S5 reads each table's `root_B` under its name, and S11/S13 find each rep stream under
its name, `find?_name`), and `liveOf` reads it (`liveZk_ofFieldsJ`); so the masked record decodes
(`decode_recordObjZkJ`), reads live (`live_recordObjZkJ`) and carries the draw (`recordDraw_recordObjZkJ`).
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open Lean (Json JsonNumber)

theorem good_rootBList {spec : SessionSpec} (hn : spec.tables.toList.Nodup) (f : ByteArray → ByteArray)
    {rootB : Array ByteArray} (hs : rootB.size = spec.tables.size) : Good (Json.mkObj (rootBList spec f rootB)) := by
  refine good_mkObj (rootBList_distinct hn f hs) fun p hp => ?_
  obtain ⟨q, -, rfl⟩ := List.mem_map.1 hp
  exact good_hexJ _

theorem good_linkObjJ {spec : SessionSpec} (hn : spec.tables.toList.Nodup) (h : ByteArray → ByteArray)
    {rootB : Array ByteArray} (hs : rootB.size = spec.tables.size) (points : Array (Array F128)) :
    Good (linkObj spec h rootB points) := by
  refine good_mkObj (linkList_distinct spec h _ points) ?_
  simp only [linkList, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨.str _, good_hexJ _, good_rootBList hn _ hs, good_rootBList hn _ hs, good_hexJ _, good_hexJ _,
    good_mkObj (by simp) (by simp), good_mkObj (by simp) (by simp), good_arr_of fun x hx => ?_, good_arr_str _⟩
  obtain ⟨p, -, rfl⟩ := Array.mem_map.1 hx
  exact good_arr_str _

/-- **The masked record is printable** at any number of tables, named distinctly, when the statement's draw is. -/
theorem good_recordObjZkJ {spec : SessionSpec} (hn : spec.tables.toList.Nodup) (h : ByteArray → ByteArray)
    (hz : String) (ct : CoinTree.Spec) {rootB : Array ByteArray} (hs : rootB.size = spec.tables.size)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray)
    (hd : Good (spec.unitDraw.getD Json.null)) :
    Good (recordObjZk spec h hz ct rootB points streams proofs) := by
  refine good_mkObj (recordListZk_distinct spec h hz ct _ points streams proofs) ?_
  simp only [recordListZk, recordList, List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq]
  refine ⟨.str _, .null, .str _, good_mkObj (by simp) (by simp; exact .bool true), .str _,
    good_linkObjJ (spec := specZ spec hz) hn h hs points, hd, good_arr_of fun x hx => ?_, good_hexJ _,
    good_arr_of fun x hx => ?_, good_coinTreeJ ct⟩
  · obtain ⟨q, -, rfl⟩ := Array.mem_map.1 hx
    exact good_streamObj _ _ _ _
  · obtain ⟨q, -, rfl⟩ := Array.mem_map.1 hx
    exact good_proofObj _ _ _

theorem str_rootB {spec : SessionSpec} (hn : spec.tables.toList.Nodup) (f : ByteArray → ByteArray)
    {rootB : Array ByteArray} (hrbs : rootB.size = spec.tables.size) {ob : Json}
    (hobs : Same (Json.mkObj (rootBList spec f rootB)) ob) (i : ℕ) (h₁ : i < spec.tables.size)
    (h₂ : i < rootB.size) : Record.str ob spec.tables[i] = .ok (hexOf (f rootB[i])) :=
  str_same hobs (getObjVal?_mkObj (v := Json.str (hexOf (f rootB[i]))) (rootBList_distinct hn _ hrbs)
    (List.mem_map.2 ⟨(spec.tables[i], rootB[i]), Array.mem_toList_iff.2 (Array.mem_iff_getElem.2
      ⟨i, by simp; omega, by simp [Array.getElem_zip]⟩), rfl⟩))

/-- **Under distinct names, the element found by a name is the one at its index.** -/
theorem find?_name {α β : Type} [BEq β] [LawfulBEq β] (f : α → β) :
    ∀ (l : List α), (l.map f).Nodup → ∀ k (hk : k < l.length), l.find? (fun a => f a == f l[k]) = some l[k]
  | [], _, _, hk => absurd hk (by simp)
  | a :: l, hn, 0, _ => by simp
  | a :: l, hn, k + 1, hk => by
    rw [List.map_cons, List.nodup_cons] at hn
    have hk' : k < l.length := by simpa using hk
    have hne : (f a == f l[k]) = false := by
      rw [beq_eq_false_iff_ne]
      intro e
      exact hn.1 (e ▸ List.mem_map_of_mem (List.getElem_mem _))
    simp only [List.find?_cons, hne, List.getElem_cons_succ]
    exact find?_name f l hn.2 k hk'

/-- **A printable object with the honest record's fields decodes** (S1–S14) as `sessOf`, at any number of tables
named distinctly, as `decodeZk_ofFields`. -/
theorem decodeZk_ofFieldsJ (S : JsonSpec) {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64)
    {spec : SessionSpec} (hn : spec.tables.toList.Nodup) (hct : spec.coinTree = none)
    (hdraw : DrawOk spec) (hsig : 0 < spec.sigma.size)
    {rootB : Array ByteArray} (hrbs : rootB.size = spec.tables.size) (hrb : ∀ rb ∈ rootB, 0 < rb.size)
    {points : Array (Array F128)} (hpts : points.size = spec.points) (hpm : ∀ p ∈ points, p.size = spec.mPts)
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (hrs : ∀ rs ∈ streams, 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n)
    (proofs : Array ByteArray) {R : Json} (hgood : Good R)
    (gR : ∀ {k v}, (k, v) ∈ recordList spec h rootB points streams proofs → R.getObjVal? k = .ok v) :
    Record.decodeWith (Hashes.ofSha512 h : Hashes V) spec (printJ R) =
      .ok (sessOf spec h rootB points streams proofs) := by
  obtain ⟨j, hp, hs⟩ := S.parse_print _ hgood
  have hkind : Record.str j "kind" = .ok "flock-live-session/v1" := str_same hs (gR (by simp [recordList]))
  have habort : Record.isNull j "aborted" = true := isNull_null hs (gR (by simp [recordList]))
  have hhello : Record.str j "hello" = .ok spec.hello := str_same hs (gR (by simp [recordList]))
  obtain ⟨cfg, hcfg, hcfgs⟩ := fld_same hs
    (gR (k := "config") (v := Json.mkObj [("require_link", Json.bool true)]) (by simp [recordList]))
  have hreq : Record.fld cfg "require_link" = .ok (Json.bool true) :=
    fld_plain hcfgs (getObjVal?_mkObj (by simp) (by simp)) (.bool true)
  have hmode : Record.str j "link_mode" = .ok "exchange" := str_same hs (gR (by simp [recordList]))
  obtain ⟨l, hl, hls⟩ := fld_same hs (gR (k := "link") (v := linkObj spec h rootB points) (by simp [recordList]))
  unfold Record.decodeWith Record.decodeJWith
  simp only [hp, hct, pure_bind, liftM, monadLift_self, hkind, hhello, habort, hcfg, hreq, hmode, hl, ok_bind,
    beq_self_eq_true, check_true]
  have gL : ∀ {k v}, (k, v) ∈ linkList spec h rootB points → (linkObj spec h rootB points).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj (linkList_distinct ..) hm
  have hlm : Record.str l "mode" = .ok "exchange" := str_same hls (gL (by simp [linkList]))
  have hsg : Record.str l "sigma" = .ok (hexOf spec.sigma) :=
    str_same hls (gL (v := hexJ spec.sigma) (by simp [linkList]))
  obtain ⟨dv, hdv, hdvs⟩ := hs.getObjVal (gR (k := "unit_draw") (v := spec.unitDraw.getD Json.null)
    (by simp [recordList]))
  simp only [hlm, hsg, ok_bind, beq_self_eq_true, check_true, hdv]
  have hdc : ∀ o : Option Json, (dv = Json.null → o = none) → (dv ≠ Json.null → o = some dv) →
      Draw.check o spec.unitDraw spec.instances = .ok () := by
    intro o ho₁ ho₂
    unfold DrawOk at hdraw
    revert hdraw hdvs
    cases spec.unitDraw with
    | none =>
      intro _ hdvs
      simp only [Option.getD_none] at hdvs
      rw [ho₁ (same_null hdvs)]
      rfl
    | some d =>
      rintro ⟨hn, hg, D, hD, hDs⟩ hdvs
      simp only [Option.getD_some] at hdvs
      have hn' : dv ≠ Json.null := fun e => hn (same_null (e ▸ hdvs.symm))
      obtain ⟨c, hc⟩ := S.canon_ok d hg
      have hc' : canon dv = .ok c := by rw [← S.canon_same _ _ hdvs, hc]
      rw [ho₂ hn']
      simp only [Draw.check, hc, hc', ok_bind, bne_self_eq_false, Bool.false_eq_true, ite_false, hD, hDs]
      rfl
  rw [hdc _ (fun e => by subst e; rfl) (fun hn => by cases dv <;> first | rfl | exact absurd rfl hn)]
  simp only [ok_bind]
  -- S5
  obtain ⟨ob, hob, hobs⟩ := fld_same hls (gL (k := "root_b") (v := Json.mkObj (rootBList spec id rootB))
    (by simp [linkList]))
  obtain ⟨od, hod, hods⟩ := fld_same hls (gL (k := s!"root_b_{spec.dk}")
    (v := Json.mkObj (rootBList spec (dgOf spec h) rootB)) (by simp [linkList]))
  rw [array_mapM_ok_of _ spec.tables rootB hrbs.symm fun i h₁ h₂ => by
    simp only [hob, hod, str_rootB hn id hrbs hobs i h₁ h₂, str_rootB hn _ hrbs hods i h₁ h₂, ok_bind, unhex_hexOf,
      dgWith_ofSha512, id,
      decide_eq_true (hrb _ (Array.getElem_mem h₂)), check_true, beq_self_eq_true]
    rfl]
  simp only [ok_bind]
  -- S6
  have hrf : Record.str l "root_f" = .ok (hexOf spec.sigma) :=
    str_same hls (gL (v := hexJ spec.sigma) (by simp [linkList]))
  have hrfd : Record.str l s!"root_f_{spec.dk}" = .ok (hexOf (dgOf spec h spec.sigma)) :=
    str_same hls (gL (v := hexJ (dgOf spec h spec.sigma)) (by simp [linkList]))
  simp only [hrf, hrfd, ok_bind, unhex_hexOf, dgWith_ofSha512, decide_eq_true hsig, check_true, beq_self_eq_true,
    bytes_beq_self]
  -- S7
  obtain ⟨op, hop, hops⟩ := fld_same hls (gL (k := "publics") (v := Json.mkObj []) (by simp [linkList]))
  obtain ⟨oq, hoq, hoqs⟩ := fld_same hls (gL (k := s!"publics_{spec.dk}") (v := Json.mkObj []) (by simp [linkList]))
  simp only [hop, hoq, ok_bind]
  rw [array_mapM_ok _ (fun _ => ByteArray.empty) spec.tables fun t' _ => by
    simp only [isNull_none hops (k := t') rfl, isNull_none hoqs (k := t') rfl, ByteArray.size_empty, beq_self_eq_true,
      Bool.and_self, Bool.not_true, Bool.false_eq_true, ite_true, ite_false]
    rfl]
  simp only [ok_bind]
  -- S9, S10
  have hpt : Record.arr l "points" = .ok (points.map fun p => Json.arr (p.map f128J)) :=
    arr_plain hls (gL (by simp [linkList])) (.arr fun x hx => by
      obtain ⟨p, -, rfl⟩ := Array.mem_map.1 hx
      exact plain_arr_str p)
  have hy : Record.arr l "y" = .ok ((yOf spec).map f128J) := arr_plain hls (gL (by simp [linkList])) (plain_arr_str _)
  simp only [hpt, hy, ok_bind, f128s_map]
  rw [array_mapM_ok_of _ _ points (by simp) fun i h₁ h₂ => by
    simp only [Array.getElem_map]
    exact f128s_map _]
  have hall : (points.all fun x => x.size == spec.mPts) = true :=
    Array.all_eq_true.2 fun i hi => by simp [hpm _ (Array.getElem_mem hi)]
  have hyz : ((yOf spec).all fun x => x.isZero) = true := Array.all_eq_true.2 fun i hi => by simp [yOf]; decide
  simp only [hpts] at hall
  simp only [yOf, Array.size_replicate] at hyz
  simp only [ok_bind, hpts, hall, beq_self_eq_true, Bool.and_self, check_true, hyz, yOf, Array.size_replicate]
  -- S12
  obtain ⟨sj, hsj, hsjs, hsje⟩ := arr_same hs (gR (k := "streams")
    (v := Json.arr ((spec.repStreams.zip streams).map (streamObj spec h rootB))) (by simp [recordList]))
  simp only [hsj, ok_bind]
  have hdsz : ∀ b, (dgOf spec h b).size = (dgOf spec h ByteArray.empty).size := by
    intro b; unfold dgOf; split <;> simp [h64, Sha256.hash_size]
  rw [array_mapM_ok_of _ sj ((spec.repStreams.zip streams).map (streamRec spec h rootB))
    (by rw [← hsjs]; simp) fun i h₁ h₂ => by
    have hi : i < (spec.repStreams.zip streams).size := by simpa using h₂
    have hq := hsje i (by simpa using hi) h₁
    simp only [Array.getElem_map] at hq ⊢
    have hmem : (spec.repStreams.zip streams)[i].2 ∈ streams := by
      rw [Array.getElem_zip]; exact Array.getElem_mem _
    generalize (spec.repStreams.zip streams)[i] = q at hq hmem ⊢
    generalize sj[i] = s' at hq ⊢
    have gS : ∀ {k v}, (k, v) ∈ streamList spec h rootB q → (streamObj spec h rootB q).getObjVal? k = .ok v :=
      fun hm => getObjVal?_mkObj (streamList_distinct ..) hm
    have hname : Record.str s' "stream" = .ok q.1.2.2 := str_same hq (gS (by simp [streamList]))
    obtain ⟨rj, hrj, hrjs, hrje⟩ := arr_same hq (gS (k := "rounds")
      (v := Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!)) (by simp [streamList]))
    have hroot : Record.fld s' s!"root_{spec.dk}" = .ok (hexJ (dgOf spec h rootB[q.1.1]!)) :=
      fld_plain hq (gS (by simp [streamList])) (.str _)
    have hpar : Record.str s' "parent" = .ok "" := str_same hq (gS (by simp [streamList]))
    have hlab : Record.str s' "label" = .ok (spec.domain spec.tables[q.1.1]! q.1.2.1) :=
      str_same hq (gS (by simp [streamList]))
    have hat : Record.nat s' "at" = .ok 0 := nat_same hq (gS (by simp [streamList]))
    simp only [hname, hrj, ok_bind]
    rw [array_mapIdxM_ok_of _ rj (q.2.map (roundRec spec h)) (by rw [← hrjs]; simp) fun k h₁ h₂ => by
      have hk : k < q.2.size := by simpa using h₂
      have hr := hrje k (by simpa using hk) h₁
      simp only [Array.getElem_map, Array.getElem_range, getElem!_pos q.2 k hk] at hr ⊢
      obtain ⟨hn1, hn2, hcs⟩ := (hrs q.2 hmem).2 _ (Array.getElem_mem hk)
      generalize q.2[k] = r at hr hn1 hn2 hcs ⊢
      generalize rj[k] = r' at hr ⊢
      have gRd : ∀ {key v}, (key, v) ∈ roundList spec h k r → (roundObj spec h k r).getObjVal? key = .ok v :=
        fun hm => getObjVal?_mkObj (roundList_distinct ..) hm
      have hk' : Record.nat r' "k" = .ok k := nat_same hr (gRd (by simp [roundList]))
      have hn : Record.nat r' "n" = .ok r.n := nat_same hr (gRd (by simp [roundList]))
      have hsha : Record.str r' s!"msg_{spec.dk}" = .ok (hexOf (dgOf spec h (r.bytes.getD ByteArray.empty))) :=
        str_same hr (gRd (v := hexJ _) (by simp [roundList]))
      have hco : Record.arr r' "coins" = .ok (r.coins.map f128J) :=
        arr_plain hr (gRd (by simp [roundList])) (plain_arr_str _)
      have hmsg : Record.str r' "msg" = .ok (hexOf (r.bytes.getD ByteArray.empty)) :=
        str_same hr (gRd (v := hexJ (r.bytes.getD ByteArray.empty)) (by simp [roundList]))
      have hmn : Record.isNull r' "msg" = false :=
        isNull_str hr (gRd (v := hexJ (r.bytes.getD ByteArray.empty)) (by simp [roundList]))
      have hml : Record.nat r' "msg_len" = .ok (r.bytes.getD ByteArray.empty).size :=
        nat_same hr (gRd (by simp [roundList]))
      have hsz : ((dgOf spec h (r.bytes.getD ByteArray.empty)).size == (dgOf spec h ByteArray.empty).size) = true := by
        rw [hdsz]; exact beq_self_eq_true _
      simp only [hk', hn, hsha, hco, hmsg, hmn, hml, ok_bind, unhex_hexOf, f128s_map,
        beq_self_eq_true, check_true, bytes_beq_self, hsz, decide_eq_true hn1, decide_eq_true hn2, hcs, Bool.and_self,
        Bool.not_false, Bool.or_true, Bool.false_eq_true, ite_false]
      rfl]
    simp only [ok_bind, hroot, hpar, hlab, hat]
    rfl]
  simp only [ok_bind]
  -- S11, S13
  have hSRs : ((spec.repStreams.zip streams).map (streamRec spec h rootB)).size = spec.repStreams.size := by
    simp [hss]
  have hSRk : ∀ k (hk : k < spec.repStreams.size),
      ((spec.repStreams.zip streams).map (streamRec spec h rootB))[k]'(by rw [hSRs]; exact hk) =
        streamRec spec h rootB (spec.repStreams[k], streams[k]'(hss ▸ hk)) := by
    intro k hk
    simp [Array.getElem_zip]
  have hnames : (((spec.repStreams.zip streams).map (streamRec spec h rootB)).toList.map (·.name)).Nodup := by
    have e : ((spec.repStreams.zip streams).map (streamRec spec h rootB)).toList.map (·.name) =
        spec.repStreams.toList.map (·.2.2) := by
      refine List.ext_getElem (by simp [hss]) fun k h₁ h₂ => ?_
      simp [streamRec]
    rw [e]
    exact repNames_nodup hn
  have hfind : ∀ k (hk : k < spec.repStreams.size),
      ((spec.repStreams.zip streams).map (streamRec spec h rootB)).find? (fun x => x.name == spec.repStreams[k].2.2) =
        some (streamRec spec h rootB (spec.repStreams[k], streams[k]'(hss ▸ hk))) := by
    intro k hk
    have hk' : k < ((spec.repStreams.zip streams).map (streamRec spec h rootB)).toList.length := by simp [hss, hk]
    have := find?_name (·.name) _ hnames k hk'
    simp only [Array.getElem_toList, hSRk k hk, Array.find?_toList] at this
    simpa [streamRec] using this
  have hS11 : (((spec.repStreams.zip streams).map (streamRec spec h rootB)).size == spec.repStreams.size &&
      spec.repStreams.all fun a =>
        ((spec.repStreams.zip streams).map (streamRec spec h rootB)).any fun x => x.name == a.2.2) = true := by
    refine Bool.and_eq_true_iff.2 ⟨by rw [hSRs, beq_self_eq_true], Array.all_eq_true.2 fun k hk => ?_⟩
    refine Array.any_eq_true.2 ⟨k, by rw [hSRs]; exact hk, ?_⟩
    rw [hSRk k hk]
    simp [streamRec]
  simp only [hS11, check_true]
  rw [← Array.forIn_toList, forIn_ok _ _ fun a ha => ?_]
  swap
  · obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem ha
    have hk' : k < spec.repStreams.size := by simpa using hk
    simp only [Array.getElem_toList]
    rw [hfind k hk']
    have h0 := (hrs _ (Array.getElem_mem (hss ▸ hk'))).1
    simp only [streamRec, beq_self_eq_true, Bool.and_self, Array.size_map, decide_eq_true h0, check_true]
    rfl
  -- S14
  have hlk : Record.str j s!"link_{spec.dk}" = .ok (hexOf (dgOf spec h (lbOf spec rootB points))) :=
    str_same hs (gR (v := hexJ (dgOf spec h (lbOf spec rootB points))) (by simp [recordList]))
  simp only [lbOf, yOf] at hlk
  simp only [ok_bind, Array.forIn_pure_yield_eq_foldl, pure_bind, hlk, beq_self_eq_true, check_true]
  -- the proof digests
  obtain ⟨pj, hpj, hpjs, hpje⟩ := arr_same hs (gR (k := s!"proof_{spec.dk}")
    (v := Json.arr ((spec.repStreams.zip proofs).map (proofObj spec h))) (by simp [recordList]))
  simp only [hpj, ok_bind]
  rw [array_mapM_ok_of _ pj ((spec.repStreams.zip proofs).map fun q => (q.1.2.2, hexOf (dgOf spec h q.2)))
    (by rw [← hpjs]; simp) fun i h₁ h₂ => by
    have hi : i < (spec.repStreams.zip proofs).size := by simpa using h₂
    have hq := hpje i (by simpa using hi) h₁
    simp only [Array.getElem_map] at hq ⊢
    generalize (spec.repStreams.zip proofs)[i] = q at hq ⊢
    generalize pj[i] = p' at hq ⊢
    have gP : ∀ {k v}, (k, v) ∈ proofList spec h q → (proofObj spec h q).getObjVal? k = .ok v :=
      fun hm => getObjVal?_mkObj (proofList_distinct ..) hm
    have e₁ : Record.str p' "stream" = .ok q.1.2.2 := str_same hq (gP (by simp [proofList]))
    have e₂ : Record.str p' spec.dk = .ok (hexOf (dgOf spec h q.2)) :=
      str_same hq (gP (v := hexJ (dgOf spec h q.2)) (by simp [proofList]))
    simp only [e₁, e₂, ok_bind]
    rfl]
  simp only [ok_bind, sessOf, yOf]
  rfl

/-- **`liveOf` reads a printable object with the honest record's fields**, at any number of tables named distinctly,
as `liveZk_ofFields`: each rep stream is found under its name. -/
theorem liveZk_ofFieldsJ (S : JsonSpec) {h : ByteArray → ByteArray} {spec : SessionSpec}
    (hn : spec.tables.toList.Nodup) {rootB : Array ByteArray} (hrbs : rootB.size = spec.tables.size)
    {points : Array (Array F128)} {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (proofs : Array ByteArray) {R : Json} (hgood : Good R)
    (gR : ∀ {k v}, (k, v) ∈ recordList spec h rootB points streams proofs → R.getObjVal? k = .ok v) :
    liveOf spec (printJ R) =
      .ok ⟨rootB, spec.tables.map fun _ => ByteArray.empty, points, streams.map (·.map liveRound)⟩ := by
  obtain ⟨j, hp, hs⟩ := S.parse_print _ hgood
  obtain ⟨l, hl, hls⟩ := fld_same hs (gR (k := "link") (v := linkObj spec h rootB points) (by simp [recordList]))
  have gL : ∀ {k v}, (k, v) ∈ linkList spec h rootB points → (linkObj spec h rootB points).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj (linkList_distinct ..) hm
  obtain ⟨ob, hob, hobs⟩ := fld_same hls (gL (k := "root_b") (v := Json.mkObj (rootBList spec id rootB))
    (by simp [linkList]))
  obtain ⟨op, hop, hops⟩ := fld_same hls (gL (k := "publics") (v := Json.mkObj []) (by simp [linkList]))
  have hpt : Record.arr l "points" = .ok (points.map fun p => Json.arr (p.map f128J)) :=
    arr_plain hls (gL (by simp [linkList])) (.arr fun x hx => by
      obtain ⟨p, -, rfl⟩ := Array.mem_map.1 hx
      exact plain_arr_str p)
  obtain ⟨sj, hsj, hsjs, hsje⟩ := arr_same hs (gR (k := "streams")
    (v := Json.arr ((spec.repStreams.zip streams).map (streamObj spec h rootB))) (by simp [recordList]))
  unfold liveOf
  simp only [hp, pure_bind, hl, ok_bind]
  rw [array_mapM_ok_of _ spec.tables rootB hrbs.symm fun i h₁ h₂ => by
    simp only [hob, str_rootB hn id hrbs hobs i h₁ h₂, ok_bind, unhex_hexOf]
    rfl]
  simp only [ok_bind, hop]
  rw [array_mapM_ok _ (fun _ => ByteArray.empty) spec.tables fun t' _ => by
    simp only [isNull_none hops (k := t') rfl, ite_true]
    rfl]
  simp only [ok_bind, hpt]
  rw [array_mapM_ok_of _ _ points (by simp) fun i h₁ h₂ => by
    simp only [Array.getElem_map]
    exact f128s_map _]
  simp only [ok_bind, hsj]
  have hsjsz : sj.size = spec.repStreams.size := by rw [← hsjs]; simp [hss]
  have hq : ∀ k (hk : k < spec.repStreams.size),
      Same (streamObj spec h rootB (spec.repStreams[k], streams[k]'(hss ▸ hk))) (sj[k]'(hsjsz ▸ hk)) := by
    intro k hk
    have := hsje k (by simp [hss, hk]) (hsjsz ▸ hk)
    simpa [Array.getElem_zip] using this
  have hnm : ∀ k (hk : k < spec.repStreams.size),
      ((sj[k]'(hsjsz ▸ hk)).getObjValAs? String "stream").toOption = some spec.repStreams[k].2.2 := by
    intro k hk
    rw [(live_stream (hq k hk)).1]
    rfl
  have hnodup : (sj.toList.map fun s => (s.getObjValAs? String "stream").toOption).Nodup := by
    have e : sj.toList.map (fun s => (s.getObjValAs? String "stream").toOption) =
        (spec.repStreams.toList.map (·.2.2)).map some := by
      refine List.ext_getElem (by simp [hsjsz]) fun k h₁ h₂ => ?_
      simp only [List.getElem_map, Array.getElem_toList]
      exact hnm k (by simpa using h₂)
    rw [e]
    exact (repNames_nodup hn).map (Option.some_injective _)
  have hfnd : ∀ k (hk : k < spec.repStreams.size),
      sj.find? (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[k].2.2) =
        some (sj[k]'(hsjsz ▸ hk)) := by
    intro k hk
    have := find?_name (fun s : Json => (s.getObjValAs? String "stream").toOption) sj.toList hnodup k
      (by simp [hsjsz, hk])
    simp only [Array.getElem_toList, Array.find?_toList, hnm k hk] at this
    exact this
  have hrounds : ∀ (rs : Array Round) (rj : Array Json) (hsz : rs.size = rj.size)
      (he : ∀ (k : Nat) (h₁ : k < rs.size) (h₂ : k < rj.size), Same (roundObj spec h k rs[k]) rj[k]),
      rj.mapM (fun r => do
        let n ← Record.nat r "n"
        let coins ← Record.f128s (← Record.arr r "coins")
        let bytes ← if Record.isNull r "msg" then pure none else some <$> unhex (← Record.str r "msg")
        pure (⟨n, ByteArray.empty, coins, bytes⟩ : Round)) = .ok (rs.map liveRound) := by
    intro rs rj hsz he
    refine array_mapM_ok_of _ rj (rs.map liveRound) (by simp [hsz]) fun k h₁ h₂ => ?_
    have hk : k < rs.size := by simpa using h₂
    obtain ⟨e₁, e₂, e₃, e₄⟩ := live_round (he k hk h₁)
    simp only [e₁, e₂, e₃, e₄, ok_bind, f128s_map, unhex_hexOf, Array.getElem_map, liveRound, Bool.false_eq_true,
      ite_false]
    rfl
  rw [array_mapM_ok_of _ spec.repStreams (streams.map (·.map liveRound)) (by simp [hss]) fun k h₁ h₂ => by
    obtain ⟨-, rj, hrj, hrjs, hrje⟩ := live_stream (hq k h₁)
    have hf := hfnd k h₁
    have hr := hrounds _ rj hrjs hrje
    simp only [Array.getElem_map]
    generalize spec.repStreams[k] = x at hf hr hrj ⊢
    obtain ⟨a, b, name⟩ := x
    simp only [hf, hrj, ok_bind]
    exact hr]
  rfl

/-- **A printable object with the honest record's fields carries the statement's draw**, up to key order, at any root
`B`s, as `recordDrawZk_ofFields`. -/
theorem recordDrawZk_ofFieldsJ (S : JsonSpec) {h : ByteArray → ByteArray} {spec : SessionSpec}
    (hdraw : DrawOk spec) (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round))
    (proofs : Array ByteArray) {R : Json} (hgood : Good R)
    (gR : ∀ {k v}, (k, v) ∈ recordList spec h rootB points streams proofs → R.getObjVal? k = .ok v) :
    match spec.unitDraw with
    | none => FlockSoundness.Discharge.Exec.recordDraw (printJ R) = none
    | some d => ∃ d', FlockSoundness.Discharge.Exec.recordDraw (printJ R) = some d' ∧
        FlockDeploy.JsonEquiv d' d := by
  obtain ⟨j, hp, hs⟩ := S.parse_print _ hgood
  obtain ⟨dv, hdv, hdvs⟩ := hs.getObjVal (gR (k := "unit_draw") (v := spec.unitDraw.getD Json.null)
    (by simp [recordList]))
  unfold FlockSoundness.Discharge.Exec.recordDraw
  rw [hp]
  dsimp only
  rw [hdv]
  unfold DrawOk at hdraw
  revert hdraw hdvs
  cases spec.unitDraw with
  | none =>
    intro _ hdvs
    simp only [Option.getD_none] at hdvs
    rw [same_null hdvs]
  | some d =>
    rintro ⟨hn, -, -⟩ hdvs
    simp only [Option.getD_some] at hdvs
    have hn' : dv ≠ Json.null := fun e => hn (same_null (e ▸ hdvs.symm))
    refine ⟨Flock.norm dv, ?_,
      (FlockDeploy.jsonEquiv_norm dv (FlockDeploy.same_wf_right hdvs)).trans hdvs.jsonEquiv.symm⟩
    cases dv <;> first | rfl | exact absurd rfl hn'

/-! ## The masked record -/

/-- **The honest masked record decodes** (S1–S14) as `sessOf`, at any number of tables named distinctly, as
`decode_recordObjZk`. -/
theorem decode_recordObjZkJ (S : JsonSpec) {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64)
    {spec : SessionSpec} (hn : spec.tables.toList.Nodup) {ct : CoinTree.Spec} (hct : spec.coinTree = some ct)
    {hz : String} (hH : HelloOk spec hz) (hdraw : DrawOk spec) (hsig : 0 < spec.sigma.size)
    {rootB : Array ByteArray} (hrbs : rootB.size = spec.tables.size) (hrb : ∀ rb ∈ rootB, 0 < rb.size)
    {points : Array (Array F128)} (hpts : points.size = spec.points) (hpm : ∀ p ∈ points, p.size = spec.mPts)
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (hrs : ∀ rs ∈ streams, 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n)
    (proofs : Array ByteArray) :
    Record.decodeWith (Hashes.ofSha512 h : Hashes V) spec
      (printJ (recordObjZk spec h hz ct rootB points streams proofs)) =
      .ok (sessOf spec h rootB points streams proofs) := by
  obtain ⟨m, kvs, hown, hm, hkv, hco, hg, hc⟩ := hH.obj
  have hgood := good_recordObjZkJ hn h hz ct hrbs points streams proofs (good_draw hdraw)
  obtain ⟨j, hp, hs⟩ := S.parse_print _ hgood
  have hh : Record.str j "hello" = .ok hz :=
    str_same hs (recordObjZk_fields spec h hz ct _ points streams proofs (v := Json.str hz)
      (by simp [recordList, specZ]))
  obtain ⟨cj, hcj, hcjs⟩ := fld_same hs (recordObjZk_coinTree spec h hz ct rootB points streams proofs)
  obtain ⟨key, hk⟩ := checkRecord_ok S ct hcjs
  rw [decodeWith_specZ _ hct hp hown hh (checkHello_ok S hm hkv hco hg hc) hcj hk]
  exact decodeZk_ofFieldsJ S h64 (spec := specZ spec hz) hn rfl hdraw hsig hrbs hrb hpts hpm hss hrs proofs hgood
    fun hm => recordObjZk_fields _ _ _ _ _ _ _ _ hm

/-- **`liveOf` reads the printed honest masked record** at any number of tables named distinctly. -/
theorem live_recordObjZkJ (S : JsonSpec) {h : ByteArray → ByteArray} {spec : SessionSpec}
    (hn : spec.tables.toList.Nodup) (hd : Good (spec.unitDraw.getD Json.null)) (hz : String) (ct : CoinTree.Spec)
    {rootB : Array ByteArray} (hrbs : rootB.size = spec.tables.size) {points : Array (Array F128)}
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size) (proofs : Array ByteArray) :
    liveOf spec (printJ (recordObjZk spec h hz ct rootB points streams proofs)) =
      .ok ⟨rootB, spec.tables.map fun _ => ByteArray.empty, points, streams.map (·.map liveRound)⟩ :=
  liveZk_ofFieldsJ S (h := h) (spec := specZ spec hz) hn hrbs hss proofs
    (good_recordObjZkJ hn h hz ct hrbs points streams proofs hd) fun hm => recordObjZk_fields _ _ _ _ _ _ _ _ hm

/-- **The honest masked record's draw is the statement's**, up to key order (`JsonEquiv`), at any number of tables
named distinctly. -/
theorem recordDraw_recordObjZkJ (S : JsonSpec) {h : ByteArray → ByteArray} {spec : SessionSpec}
    (hn : spec.tables.toList.Nodup) (hdraw : DrawOk spec) (hz : String) (ct : CoinTree.Spec)
    {rootB : Array ByteArray} (hrbs : rootB.size = spec.tables.size) (points : Array (Array F128))
    (streams : Array (Array Round)) (proofs : Array ByteArray) :
    match spec.unitDraw with
    | none => FlockSoundness.Discharge.Exec.recordDraw (printJ (recordObjZk spec h hz ct rootB points streams proofs)) =
        none
    | some d => ∃ d', FlockSoundness.Discharge.Exec.recordDraw
        (printJ (recordObjZk spec h hz ct rootB points streams proofs)) = some d' ∧ FlockDeploy.JsonEquiv d' d :=
  recordDrawZk_ofFieldsJ S (h := h) (spec := specZ spec hz) hdraw rootB points streams proofs
    (good_recordObjZkJ hn h hz ct hrbs points streams proofs (good_draw hdraw))
    fun hm => recordObjZk_fields _ _ _ _ _ _ _ _ hm

end FlockComplete.ZkTables
