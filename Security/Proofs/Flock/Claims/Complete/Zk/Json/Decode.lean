import Proofs.Flock.Claims.Complete.Json.Live
import Proofs.Flock.Claims.Complete.Zk.Json.Record
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Claims.Complete.Json.Canon

/-!
# The honest masked record decodes, reads live and carries the draw

C's `decode_recordObj`, `live_recordObj` and `recordDraw_recordObj` read the printed record only through `parse` and its
fields, so they hold of any printable object with `recordList`'s fields (`decodeZk_ofFields`, `liveZk_ofFields`,
`recordDrawZk_ofFields`), the masked record among them (`recordObjZk_fields`: `recordList` at `specZ`, `coin_tree`
added). The masked decoder's S2 passes on it (`checkHello_ok`, `checkRecord_ok`), after which it decodes as the plain
record of `specZ` (`decodeWith_specZ`): `decode_recordObjZk`, `live_recordObjZk`, `recordDraw_recordObjZk`.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy
open Lean (Json JsonNumber)

/-- **A printable object with the honest record's fields decodes** (S1–S14) as `sessOf`, as `decode_recordObj`. -/
theorem decodeZk_ofFields {h : ByteArray → ByteArray}
    (h64 : ∀ b, (h b).size = 64)
    {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (hct : spec.coinTree = none)
    (hdraw : DrawOk spec) (hsig : 0 < spec.sigma.size)
    {rb : ByteArray} (hrb : 0 < rb.size)
    {points : Array (Array F128)} (hpts : points.size = spec.points) (hpm : ∀ p ∈ points, p.size = spec.mPts)
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (hrs : ∀ rs ∈ streams, 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n)
    (proofs : Array ByteArray) {R : Json} (hgood : Good R)
    (gR : ∀ {k v}, (k, v) ∈ recordList spec h #[rb] points streams proofs → R.getObjVal? k = .ok v) :
    Record.decodeWith (Hashes.ofSha512 h : Hashes V) spec (printJ R) =
      .ok (sessOf spec h #[rb] points streams proofs) := by
  obtain ⟨j, hp, hs⟩ := FlockComplete.JsonRT.parse_print _ hgood
  have hkind : Record.str j "kind" = .ok "flock-live-session/v1" := str_same hs (gR (by simp [recordList]))
  have habort : Record.isNull j "aborted" = true := isNull_null hs (gR (by simp [recordList]))
  have hhello : Record.str j "hello" = .ok spec.hello := str_same hs (gR (by simp [recordList]))
  obtain ⟨cfg, hcfg, hcfgs⟩ := fld_same hs
    (gR (k := "config") (v := Json.mkObj [("require_link", Json.bool true)]) (by simp [recordList]))
  have hreq : Record.fld cfg "require_link" = .ok (Json.bool true) :=
    fld_plain hcfgs (getObjVal?_mkObj (by simp) (by simp)) (.bool true)
  have hmode : Record.str j "link_mode" = .ok "exchange" := str_same hs (gR (by simp [recordList]))
  obtain ⟨l, hl, hls⟩ := fld_same hs (gR (k := "link") (v := linkObj spec h #[rb] points) (by simp [recordList]))
  unfold Record.decodeWith Record.decodeJWith
  simp only [hp, hct, pure_bind, liftM, monadLift_self, hkind, hhello, habort, hcfg, hreq, hmode, hl, ok_bind,
    beq_self_eq_true, check_true]
  have gL : ∀ {k v}, (k, v) ∈ linkList spec h #[rb] points → (linkObj spec h #[rb] points).getObjVal? k = .ok v :=
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
      rintro ⟨hn, hg, ha, D, hD, hDs⟩ hdvs
      simp only [Option.getD_some] at hdvs
      have hn' : dv ≠ Json.null := fun e => hn (same_null (e ▸ hdvs.symm))
      obtain ⟨c, hc⟩ := FlockComplete.JsonRT.canon_ok d hg ha
      have hc' : canon dv = .ok c := by rw [← FlockComplete.JsonRT.canon_same _ _ hdvs, hc]
      rw [ho₂ hn']
      simp only [Draw.check, hc, hc', ok_bind, bne_self_eq_false, Bool.false_eq_true, ite_false, hD, hDs]
      rfl
  rw [hdc _ (fun e => by subst e; rfl) (fun hn => by cases dv <;> first | rfl | exact absurd rfl hn)]
  simp only [ok_bind]
  -- S5
  obtain ⟨ob, hob, hobs⟩ := fld_same hls (gL (k := "root_b") (v := Json.mkObj (rootBList spec id #[rb]))
    (by simp [linkList]))
  obtain ⟨od, hod, hods⟩ := fld_same hls (gL (k := s!"root_b_{spec.dk}")
    (v := Json.mkObj (rootBList spec (dgOf spec h) #[rb])) (by simp [linkList]))
  rw [rootBList_one ht] at hobs hods
  have hobt : Record.str ob t = .ok (hexOf rb) :=
    str_same hobs (getObjVal?_mkObj (v := Json.str (hexOf rb)) (by simp) (by simp [hexJ]))
  have hodt : Record.str od t = .ok (hexOf (dgOf spec h rb)) :=
    str_same hods (getObjVal?_mkObj (v := Json.str (hexOf (dgOf spec h rb))) (by simp) (by simp [hexJ]))
  rw [array_mapM_ok_of _ spec.tables #[rb] (by simp [ht]) fun i h₁ h₂ => by
    have hi : i = 0 := by simp [ht] at h₁; omega
    subst hi
    simp only [ht, Array.getElem_singleton, hob, hod, hobt, hodt, ok_bind, unhex_hexOf, dgWith_ofSha512,
      decide_eq_true hrb, check_true, beq_self_eq_true]
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
    (v := Json.arr ((spec.repStreams.zip streams).map (streamObj spec h #[rb]))) (by simp [recordList]))
  simp only [hsj, ok_bind]
  have hdsz : ∀ b, (dgOf spec h b).size = (dgOf spec h ByteArray.empty).size := by
    intro b; unfold dgOf; split <;> simp [h64, Sha256.hash_size]
  rw [array_mapM_ok_of _ sj ((spec.repStreams.zip streams).map (streamRec spec h #[rb]))
    (by rw [← hsjs]; simp) fun i h₁ h₂ => by
    have hi : i < (spec.repStreams.zip streams).size := by simpa using h₂
    have hq := hsje i (by simpa using hi) h₁
    simp only [Array.getElem_map] at hq ⊢
    have hmem : (spec.repStreams.zip streams)[i].2 ∈ streams := by
      rw [Array.getElem_zip]; exact Array.getElem_mem _
    generalize (spec.repStreams.zip streams)[i] = q at hq hmem ⊢
    generalize sj[i] = s' at hq ⊢
    have gS : ∀ {k v}, (k, v) ∈ streamList spec h #[rb] q → (streamObj spec h #[rb] q).getObjVal? k = .ok v :=
      fun hm => getObjVal?_mkObj (streamList_distinct ..) hm
    have hname : Record.str s' "stream" = .ok q.1.2.2 := str_same hq (gS (by simp [streamList]))
    obtain ⟨rj, hrj, hrjs, hrje⟩ := arr_same hq (gS (k := "rounds")
      (v := Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!)) (by simp [streamList]))
    have hroot : Record.fld s' s!"root_{spec.dk}" = .ok (hexJ (dgOf spec h #[rb][q.1.1]!)) :=
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
  have hrs1 : spec.repStreams = #[(0, 0, s!"{t}/rep{(0 : Nat)}"), (0, 1, s!"{t}/rep{(1 : Nat)}")] := by
    unfold SessionSpec.repStreams
    rw [ht]
    simp only [List.size_toArray, List.length_singleton]
    rw [show Array.range 1 = #[0] from rfl]
    simp
  obtain ⟨s0, s1, rfl⟩ : ∃ s0 s1, streams = #[s0, s1] := by
    rw [hrs1] at hss
    rcases streams with ⟨_ | ⟨s0, _ | ⟨s1, _ | ⟨_, _⟩⟩⟩⟩ <;> simp at hss
    exact ⟨s0, s1, rfl⟩
  have hsr : (spec.repStreams.zip #[s0, s1]).map (streamRec spec h #[rb]) =
      #[streamRec spec h #[rb] ((0, 0, s!"{t}/rep{(0 : Nat)}"), s0),
        streamRec spec h #[rb] ((0, 1, s!"{t}/rep{(1 : Nat)}"), s1)] := by
    rw [hrs1]
    simp only [List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray, List.map_cons,
      List.map_nil]
  have h0 := (hrs s0 (by simp)).1
  have h1 := (hrs s1 (by simp)).1
  have hne : (s!"{t}/rep{(0 : Nat)}" == s!"{t}/rep{(1 : Nat)}") = false := by simp
  have hne' : (s!"{t}/rep{(1 : Nat)}" == s!"{t}/rep{(0 : Nat)}") = false := by simp
  rw [hsr, hrs1]
  simp only [List.all_toArray, List.any_toArray]
  simp only [pure_bind, List.size_toArray, List.length_cons, List.length_nil, List.all_cons,
    List.any_cons, List.all_nil, List.any_nil, streamRec, beq_self_eq_true, hne, hne', Bool.or_false, Bool.false_or,
    check_true, ok_bind, List.forIn_toArray, List.forIn_cons, List.forIn_nil, List.find?_toArray,
    List.find?_cons, Array.size_map, decide_eq_true h0, decide_eq_true h1, Bool.and_self]
  -- S14
  have hlk : Record.str j s!"link_{spec.dk}" = .ok (hexOf (dgOf spec h (lbOf spec #[rb] points))) :=
    str_same hs (gR (v := hexJ (dgOf spec h (lbOf spec #[rb] points))) (by simp [recordList]))
  simp only [lbOf, yOf] at hlk
  simp only [Array.forIn_pure_yield_eq_foldl, pure_bind, hlk, ok_bind, beq_self_eq_true, check_true]
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
  simp only [ok_bind, sessOf, hsr, yOf, streamRec]
  rfl

/-- **`liveOf` reads a printable object with the honest record's fields**, as `live_recordObj`. -/
theorem liveZk_ofFields {h : ByteArray → ByteArray} {spec : SessionSpec}
    {t : String}
    (ht : spec.tables = #[t]) {rb : ByteArray} {points : Array (Array F128)} {streams : Array (Array Round)}
    (hss : streams.size = spec.repStreams.size) (proofs : Array ByteArray) {R : Json} (hgood : Good R)
    (gR : ∀ {k v}, (k, v) ∈ recordList spec h #[rb] points streams proofs → R.getObjVal? k = .ok v) :
    liveOf spec (printJ R) =
      .ok ⟨#[rb], spec.tables.map fun _ => ByteArray.empty, points, streams.map (·.map liveRound)⟩ := by
  obtain ⟨j, hp, hs⟩ := FlockComplete.JsonRT.parse_print _ hgood
  obtain ⟨l, hl, hls⟩ := fld_same hs (gR (k := "link") (v := linkObj spec h #[rb] points) (by simp [recordList]))
  have gL : ∀ {k v}, (k, v) ∈ linkList spec h #[rb] points → (linkObj spec h #[rb] points).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj (linkList_distinct ..) hm
  obtain ⟨ob, hob, hobs⟩ := fld_same hls (gL (k := "root_b") (v := Json.mkObj (rootBList spec id #[rb]))
    (by simp [linkList]))
  rw [rootBList_one ht] at hobs
  have hobt : Record.str ob t = .ok (hexOf rb) :=
    str_same hobs (getObjVal?_mkObj (v := Json.str (hexOf rb)) (by simp) (by simp [hexJ]))
  obtain ⟨op, hop, hops⟩ := fld_same hls (gL (k := "publics") (v := Json.mkObj []) (by simp [linkList]))
  have hpt : Record.arr l "points" = .ok (points.map fun p => Json.arr (p.map f128J)) :=
    arr_plain hls (gL (by simp [linkList])) (.arr fun x hx => by
      obtain ⟨p, -, rfl⟩ := Array.mem_map.1 hx
      exact plain_arr_str p)
  obtain ⟨sj, hsj, hsjs, hsje⟩ := arr_same hs (gR (k := "streams")
    (v := Json.arr ((spec.repStreams.zip streams).map (streamObj spec h #[rb]))) (by simp [recordList]))
  unfold liveOf
  simp only [hp, pure_bind, hl, ok_bind]
  rw [array_mapM_ok_of _ spec.tables #[rb] (by simp [ht]) fun i h₁ h₂ => by
    have hi : i = 0 := by simp [ht] at h₁; omega
    subst hi
    simp only [ht, Array.getElem_singleton, hob, hobt, ok_bind, unhex_hexOf]]
  simp only [ok_bind, hop]
  rw [array_mapM_ok _ (fun _ => ByteArray.empty) spec.tables fun t' _ => by
    simp only [isNull_none hops (k := t') rfl, ite_true]
    rfl]
  simp only [ok_bind, hpt]
  rw [array_mapM_ok_of _ _ points (by simp) fun i h₁ h₂ => by
    simp only [Array.getElem_map]
    exact f128s_map _]
  simp only [ok_bind, hsj]
  have hrs1 := repStreams_one ht
  obtain ⟨s0, s1, rfl⟩ : ∃ s0 s1, streams = #[s0, s1] := by
    rw [hrs1] at hss
    rcases streams with ⟨_ | ⟨s0, _ | ⟨s1, _ | ⟨_, _⟩⟩⟩⟩ <;> simp at hss
    exact ⟨s0, s1, rfl⟩
  obtain ⟨a0, a1, rfl⟩ : ∃ a0 a1, sj = #[a0, a1] := by
    have : sj.size = 2 := by rw [← hsjs]; simp [hrs1]
    rcases sj with ⟨_ | ⟨a0, _ | ⟨a1, _ | ⟨_, _⟩⟩⟩⟩ <;> simp at this
    exact ⟨a0, a1, rfl⟩
  have hq0 := hsje 0 (by simp [hrs1]) (by simp)
  have hq1 := hsje 1 (by simp [hrs1]) (by simp)
  simp only [hrs1, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray, List.map_cons,
    List.map_nil, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ] at hq0 hq1
  obtain ⟨hn0, r0, hr0, hr0s, hr0e⟩ := live_stream hq0
  obtain ⟨hn1, r1, hr1, hr1s, hr1e⟩ := live_stream hq1
  dsimp only at hn0 hn1 hr0s hr1s hr0e hr1e
  have hne : (s!"{t}/rep{(0 : Nat)}" == s!"{t}/rep{(1 : Nat)}") = false := by simp
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
  rw [hrs1]
  rw [array_mapM_ok_of _ _ #[s0.map liveRound, s1.map liveRound] (by simp) fun i h₁ h₂ => by
    have hi : i = 0 ∨ i = 1 := by simp at h₁; omega
    rcases hi with rfl | rfl
    · simp only [List.getElem_toArray, List.getElem_cons_zero, List.find?_toArray, List.find?_cons, hn0,
        Except.toOption, beq_self_eq_true, hr0, ok_bind]
      exact hrounds s0 r0 hr0s hr0e
    · simp only [List.getElem_toArray, List.getElem_cons_succ, List.getElem_cons_zero, List.find?_toArray,
        List.find?_cons, hn0, hn1, Except.toOption, beq_self_eq_true, Option.some_beq_some, hne, hr1, ok_bind]
      exact hrounds s1 r1 hr1s hr1e]
  simp only [ok_bind, List.map_toArray, List.map_cons, List.map_nil]
  rfl

/-- **A printable object with the honest record's fields carries the statement's draw**, up to key order, as
`recordDraw_recordObj`. -/
theorem recordDrawZk_ofFields {h : ByteArray → ByteArray} {spec : SessionSpec}
    (hdraw : DrawOk spec) (rb : ByteArray) (points : Array (Array F128)) (streams : Array (Array Round))
    (proofs : Array ByteArray) {R : Json} (hgood : Good R)
    (gR : ∀ {k v}, (k, v) ∈ recordList spec h #[rb] points streams proofs → R.getObjVal? k = .ok v) :
    match spec.unitDraw with
    | none => FlockSoundness.Discharge.Exec.recordDraw (printJ R) = none
    | some d => ∃ d', FlockSoundness.Discharge.Exec.recordDraw (printJ R) = some d' ∧
        FlockDeploy.JsonEquiv d' d := by
  obtain ⟨j, hp, hs⟩ := FlockComplete.JsonRT.parse_print _ hgood
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

/-- **The masked record holds the plain record's fields at `specZ`.** -/
theorem recordObjZk_fields (spec : SessionSpec) (h : ByteArray → ByteArray) (hz : String) (ct : CoinTree.Spec)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray)
    {k : String} {v : Json} (hm : (k, v) ∈ recordList (specZ spec hz) h rootB points streams proofs) :
    (recordObjZk spec h hz ct rootB points streams proofs).getObjVal? k = .ok v :=
  getObjVal?_mkObj (recordListZk_distinct ..) (List.mem_append_left _ hm)

theorem recordObjZk_coinTree (spec : SessionSpec) (h : ByteArray → ByteArray) (hz : String) (ct : CoinTree.Spec)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    (recordObjZk spec h hz ct rootB points streams proofs).getObjVal? "coin_tree" = .ok (coinTreeJ ct) :=
  getObjVal?_mkObj (recordListZk_distinct ..) (List.mem_append_right _ (by simp))

theorem good_draw {spec : SessionSpec} (hdraw : DrawOk spec) : Good (spec.unitDraw.getD Json.null) := by
  unfold DrawOk at hdraw
  revert hdraw
  cases spec.unitDraw with
  | none => exact fun _ => .null
  | some d => exact fun hd => hd.2.1

/-- What the masked record needs of the verifier's `Hello`: a JSON object whose `coins` is an object, printable with
the prover's nonce put in, and the prover's `Hello` its `canon`. -/
structure HelloOk (spec : SessionSpec) (hz : String) : Prop where
  obj : ∃ m kvs, Flock.JsonParse.parse spec.hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧ m.get? "coins" = some (.obj kvs) ∧
    Good (helloJ (.obj m)) ∧ canon (helloJ (.obj m)) = .ok hz

/-- **The honest masked record decodes** (S1–S14) as `sessOf`: S2 passes on the prover's `Hello` and `coin_tree`, and
the rest is the plain record's at `specZ`. -/
theorem decode_recordObjZk {h : ByteArray → ByteArray}
    (h64 : ∀ b, (h b).size = 64)
    {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) {ct : CoinTree.Spec} (hct : spec.coinTree = some ct)
    (hA : Ascii ct.toJson) {hz : String} (hH : HelloOk spec hz) (hdraw : DrawOk spec) (hsig : 0 < spec.sigma.size)
    {rb : ByteArray} (hrb : 0 < rb.size)
    {points : Array (Array F128)} (hpts : points.size = spec.points) (hpm : ∀ p ∈ points, p.size = spec.mPts)
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (hrs : ∀ rs ∈ streams, 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n)
    (proofs : Array ByteArray) :
    Record.decodeWith (Hashes.ofSha512 h : Hashes V) spec
      (printJ (recordObjZk spec h hz ct #[rb] points streams proofs)) =
      .ok (sessOf spec h #[rb] points streams proofs) := by
  obtain ⟨m, kvs, hown, hm, hkv, hco, hg, hc⟩ := hH.obj
  have hgood := good_recordObjZk ht h hz ct rb points streams proofs (good_draw hdraw)
  obtain ⟨j, hp, hs⟩ := FlockComplete.JsonRT.parse_print _ hgood
  have hh : Record.str j "hello" = .ok hz :=
    str_same hs (recordObjZk_fields spec h hz ct _ points streams proofs (v := Json.str hz)
      (by simp [recordList, specZ]))
  obtain ⟨cj, hcj, hcjs⟩ := fld_same hs (recordObjZk_coinTree spec h hz ct #[rb] points streams proofs)
  obtain ⟨key, hk⟩ := checkRecord_ok ct hA hcjs
  rw [decodeWith_specZ _ hct hp hown hh (checkHello_ok hm hkv hco hg hc) hcj hk]
  exact decodeZk_ofFields h64 (spec := specZ spec hz) ht rfl hdraw hsig hrb hpts hpm hss hrs proofs hgood
    fun hm => recordObjZk_fields _ _ _ _ _ _ _ _ hm

/-- **`liveOf` reads the printed honest masked record**, as the plain one's. -/
theorem live_recordObjZk {h : ByteArray → ByteArray} {spec : SessionSpec}
    {t : String} (ht : spec.tables = #[t]) (hd : Good (spec.unitDraw.getD Json.null)) (hz : String) (ct : CoinTree.Spec)
    {rb : ByteArray} {points : Array (Array F128)} {streams : Array (Array Round)}
    (hss : streams.size = spec.repStreams.size) (proofs : Array ByteArray) :
    liveOf spec (printJ (recordObjZk spec h hz ct #[rb] points streams proofs)) =
      .ok ⟨#[rb], spec.tables.map fun _ => ByteArray.empty, points, streams.map (·.map liveRound)⟩ :=
  liveZk_ofFields (h := h) (spec := specZ spec hz) ht hss proofs
    (good_recordObjZk ht h hz ct rb points streams proofs hd) fun hm => recordObjZk_fields _ _ _ _ _ _ _ _ hm

/-- **The honest masked record's draw is the statement's**, up to key order (`JsonEquiv`). -/
theorem recordDraw_recordObjZk {h : ByteArray → ByteArray}
    {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (hdraw : DrawOk spec) (hz : String) (ct : CoinTree.Spec) (rb : ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    match spec.unitDraw with
    | none => FlockSoundness.Discharge.Exec.recordDraw (printJ (recordObjZk spec h hz ct #[rb] points streams proofs)) =
        none
    | some d => ∃ d', FlockSoundness.Discharge.Exec.recordDraw
        (printJ (recordObjZk spec h hz ct #[rb] points streams proofs)) = some d' ∧ FlockDeploy.JsonEquiv d' d :=
  recordDrawZk_ofFields (h := h) (spec := specZ spec hz) hdraw rb points streams proofs
    (good_recordObjZk ht h hz ct rb points streams proofs (good_draw hdraw))
    fun hm => recordObjZk_fields _ _ _ _ _ _ _ _ hm

end FlockComplete.JsonRT
