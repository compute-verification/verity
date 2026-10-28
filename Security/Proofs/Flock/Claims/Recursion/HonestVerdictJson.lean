import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Claims.Complete.ZkTables.Decode
import Proofs.Flock.Claims.Complete.Json.Canon

/-!
# The honest record survives the reader's substitutions

The reader's verdict reads the honest record `recordObj` with its proof digests written from the files (`shaSubJ`) and
its coins the live coins (`coinSubJ`). Where the files are the proofs' and each round's live coin opens to the round's
recorded coins, the value it reads is alike (`Same`) the honest record, key by key: each substitution writes back the
value the record already holds. A value alike the honest record decodes as the printed record does.

* **Decoding a value alike the record** (`decodeJ_recordObj`): `decode_recordObj`'s reads at any value alike the
  record, with no parse.
* **Alike values** (`same_refl`): a printable value is alike itself.
* **The substitutions** (`same_round_fill`, `same_link_points`, `same_stream_fill`, `same_coinSub_recordObj`): a round
  whose coin opens to its coins, the link at its own points, each rep stream at its coin offset, the whole record.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete
open Lean (Json JsonNumber)

/-- **A value alike the honest record decodes as `sessOf`**: `decode_recordObj`, read at the value. -/
theorem decodeJ_recordObj {h : ByteArray → ByteArray}
    (h64 : ∀ b, (h b).size = 64) {spec : SessionSpec} {t : String} (ht : spec.tables = #[t])
    (hct : spec.coinTree = none)
    (hdraw : DrawOk spec) (hsig : 0 < spec.sigma.size)
    {rb : ByteArray} (hrb : 0 < rb.size)
    {points : Array (Array F128)} (hpts : points.size = spec.points) (hpm : ∀ p ∈ points, p.size = spec.mPts)
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (hrs : ∀ rs ∈ streams, 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n)
    (proofs : Array ByteArray) {j : Json} (hs : Same (recordObj spec h #[rb] points streams proofs) j) :
    Record.decodeJWith (Hashes.ofSha512 h : Hashes V) spec j = .ok (sessOf spec h #[rb] points streams proofs) := by
  have gR : ∀ {k v}, (k, v) ∈ recordList spec h #[rb] points streams proofs →
      (recordObj spec h #[rb] points streams proofs).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj (recordList_distinct ..) hm
  have hkind : Record.str j "kind" = .ok "flock-live-session/v1" := str_same hs (gR (by simp [recordList]))
  have habort : Record.isNull j "aborted" = true := isNull_null hs (gR (by simp [recordList]))
  have hhello : Record.str j "hello" = .ok spec.hello := str_same hs (gR (by simp [recordList]))
  obtain ⟨cfg, hcfg, hcfgs⟩ := fld_same hs
    (gR (k := "config") (v := Json.mkObj [("require_link", Json.bool true)]) (by simp [recordList]))
  have hreq : Record.fld cfg "require_link" = .ok (Json.bool true) :=
    fld_plain hcfgs (getObjVal?_mkObj (by simp) (by simp)) (.bool true)
  have hmode : Record.str j "link_mode" = .ok "exchange" := str_same hs (gR (by simp [recordList]))
  obtain ⟨l, hl, hls⟩ := fld_same hs (gR (k := "link") (v := linkObj spec h #[rb] points) (by simp [recordList]))
  unfold Record.decodeJWith
  simp only [hct, pure_bind, liftM, monadLift_self, hkind, hhello, habort, hcfg, hreq, hmode, hl, FlockComplete.ok_bind,
    beq_self_eq_true, check_true]
  have gL : ∀ {k v}, (k, v) ∈ linkList spec h #[rb] points → (linkObj spec h #[rb] points).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj (linkList_distinct ..) hm
  have hlm : Record.str l "mode" = .ok "exchange" := str_same hls (gL (by simp [linkList]))
  have hsg : Record.str l "sigma" = .ok (hexOf spec.sigma) :=
    str_same hls (gL (v := hexJ spec.sigma) (by simp [linkList]))
  obtain ⟨dv, hdv, hdvs⟩ := hs.getObjVal (gR (k := "unit_draw") (v := spec.unitDraw.getD Json.null)
    (by simp [recordList]))
  simp only [hlm, hsg, FlockComplete.ok_bind, beq_self_eq_true, check_true, hdv]
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
      simp only [Draw.check, hc, hc', FlockComplete.ok_bind, bne_self_eq_false, Bool.false_eq_true, ite_false, hD, hDs]
      rfl
  rw [hdc _ (fun e => by subst e; rfl) (fun hn => by cases dv <;> first | rfl | exact absurd rfl hn)]
  simp only [FlockComplete.ok_bind]
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
    simp only [ht, Array.getElem_singleton, hob, hod, hobt, hodt, FlockComplete.ok_bind, unhex_hexOf, dgWith_ofSha512,
      decide_eq_true hrb, check_true, beq_self_eq_true]
    rfl]
  simp only [FlockComplete.ok_bind]
  -- S6
  have hrf : Record.str l "root_f" = .ok (hexOf spec.sigma) := str_same hls (gL (v := hexJ spec.sigma) (by simp [linkList]))
  have hrfd : Record.str l s!"root_f_{spec.dk}" = .ok (hexOf (dgOf spec h spec.sigma)) :=
    str_same hls (gL (v := hexJ (dgOf spec h spec.sigma)) (by simp [linkList]))
  simp only [hrf, hrfd, FlockComplete.ok_bind, unhex_hexOf, dgWith_ofSha512, decide_eq_true hsig, check_true, beq_self_eq_true,
    bytes_beq_self]
  -- S7
  obtain ⟨op, hop, hops⟩ := fld_same hls (gL (k := "publics") (v := Json.mkObj []) (by simp [linkList]))
  obtain ⟨oq, hoq, hoqs⟩ := fld_same hls (gL (k := s!"publics_{spec.dk}") (v := Json.mkObj []) (by simp [linkList]))
  simp only [hop, hoq, FlockComplete.ok_bind]
  rw [array_mapM_ok _ (fun _ => ByteArray.empty) spec.tables fun t' _ => by
    simp only [isNull_none hops (k := t') rfl, isNull_none hoqs (k := t') rfl, ByteArray.size_empty, beq_self_eq_true,
      Bool.and_self, Bool.not_true, Bool.false_eq_true, ite_true, ite_false]
    rfl]
  simp only [FlockComplete.ok_bind]
  -- S9, S10
  have hpt : Record.arr l "points" = .ok (points.map fun p => Json.arr (p.map f128J)) :=
    arr_plain hls (gL (by simp [linkList])) (.arr fun x hx => by
      obtain ⟨p, -, rfl⟩ := Array.mem_map.1 hx
      exact plain_arr_str p)
  have hy : Record.arr l "y" = .ok ((yOf spec).map f128J) := arr_plain hls (gL (by simp [linkList])) (plain_arr_str _)
  simp only [hpt, hy, FlockComplete.ok_bind, f128s_map]
  rw [array_mapM_ok_of _ _ points (by simp) fun i h₁ h₂ => by
    simp only [Array.getElem_map]
    exact f128s_map _]
  have hall : (points.all fun x => x.size == spec.mPts) = true :=
    Array.all_eq_true.2 fun i hi => by simp [hpm _ (Array.getElem_mem hi)]
  have hyz : ((yOf spec).all fun x => x.isZero) = true := Array.all_eq_true.2 fun i hi => by simp [yOf]; decide
  simp only [hpts] at hall
  simp only [yOf, Array.size_replicate] at hyz
  simp only [FlockComplete.ok_bind, hpts, hall, beq_self_eq_true, Bool.and_self, check_true, hyz, yOf, Array.size_replicate]
  -- S12
  obtain ⟨sj, hsj, hsjs, hsje⟩ := arr_same hs (gR (k := "streams")
    (v := Json.arr ((spec.repStreams.zip streams).map (streamObj spec h #[rb]))) (by simp [recordList]))
  simp only [hsj, FlockComplete.ok_bind]
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
    simp only [hname, hrj, FlockComplete.ok_bind]
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
      simp only [hk', hn, hsha, hco, hmsg, hmn, hml, FlockComplete.ok_bind, unhex_hexOf, f128s_map,
        beq_self_eq_true, check_true, bytes_beq_self, hsz, decide_eq_true hn1, decide_eq_true hn2, hcs, Bool.and_self,
        Bool.not_false, Bool.or_true, Bool.false_eq_true, ite_false]
      rfl]
    simp only [FlockComplete.ok_bind, hroot, hpar, hlab, hat]
    rfl]
  simp only [FlockComplete.ok_bind]
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
    check_true, FlockComplete.ok_bind, List.forIn_toArray, List.forIn_cons, List.forIn_nil, List.find?_toArray,
    List.find?_cons, Array.size_map, decide_eq_true h0, decide_eq_true h1, Bool.and_self]
  -- S14
  have hlk : Record.str j s!"link_{spec.dk}" = .ok (hexOf (dgOf spec h (lbOf spec #[rb] points))) :=
    str_same hs (gR (v := hexJ (dgOf spec h (lbOf spec #[rb] points))) (by simp [recordList]))
  simp only [lbOf, yOf] at hlk
  simp only [Array.forIn_pure_yield_eq_foldl, pure_bind, hlk, FlockComplete.ok_bind, beq_self_eq_true, check_true]
  -- the proof digests
  obtain ⟨pj, hpj, hpjs, hpje⟩ := arr_same hs (gR (k := s!"proof_{spec.dk}")
    (v := Json.arr ((spec.repStreams.zip proofs).map (proofObj spec h))) (by simp [recordList]))
  simp only [hpj, FlockComplete.ok_bind]
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
    simp only [e₁, e₂, FlockComplete.ok_bind]
    rfl]
  simp only [FlockComplete.ok_bind, sessOf, hsr, yOf, streamRec]
  rfl

end FlockComplete.JsonRT

namespace FlockDeploy.HonestJson

open Flock FlockComplete
open Lean (Json)
open Claims.Json (Same Good)
open FlockComplete.JsonRT (recordObj recordList recordList_distinct linkObj linkList linkList_distinct streamObj
  streamList streamList_distinct roundObj roundList roundList_distinct proofObj getObjVal?_mkObj get?_mkObj
  good_roundObj good_streamObj)

/-! ## Alike values -/

/-- **A printable value is alike itself.** -/
theorem same_refl {j : Json} (h : Good j) : Same j j := by
  induction h with
  | null => exact .null
  | bool b => exact .bool b
  | num m => exact .num _
  | str s => exact .str s
  | arr _ ih => exact .arr rfl fun i h₁ _ => ih _ (Array.getElem_mem h₁)
  | obj hm _ ih => exact .obj hm hm (fun _ => rfl) fun k a b ha hb => by rw [ha] at hb; cases hb; exact ih k a ha

theorem wf_of_good {m : Std.TreeMap.Raw String Json compare} (h : Good (.obj m)) : m.WF := by
  cases h with
  | obj hm _ => exact hm

theorem good_of_get {m : Std.TreeMap.Raw String Json compare} (h : Good (.obj m)) {k : String} {v : Json}
    (hv : m.get? k = some v) : Good v := by
  cases h with
  | obj _ hg => exact hg k v hv

theorem wf_map {m : Std.TreeMap.Raw String Json compare} (f : String → Json → Json) (hm : m.WF) : (m.map f).WF :=
  ⟨⟨Std.DTreeMap.Internal.Impl.WF.map hm.out.out⟩⟩

theorem mapKey_obj (k : String) (g : Json → Json) (m : Std.TreeMap.Raw String Json compare) :
    mapKey k g (.obj m) = .obj (m.map fun k' v => if k' = k then g v else v) := rfl

theorem fld_of_gv {j : Json} {k : String} {v : Json} (h : j.getObjVal? k = .ok v) : Record.fld j k = .ok v := by
  simp only [Record.fld, h]

/-! ## The substitutions' maps -/

/-- The erasure's round map. -/
def eraseG : String → ℕ → Json → Json := fun _ _ => mapKey "coins" fun _ => .null

/-- The fill's round map, at the erased value `j`. -/
def fillG (spec : SessionSpec) (mids : List (Claims.Msg × Coin)) (j : Json) : String → ℕ → Json → Json :=
  fun name k r => match mids[streamOffset spec j name + k]? with
    | some (_, c) => mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r
    | none => r

theorem eraseCoins_eq (j : Json) : eraseCoins j = mapRounds eraseG (setPoints .null j) := rfl

/-- **The coin substitution as five key maps**: the link's points and the streams' rounds erased, then filled. -/
theorem coinSubJ_eq (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) :
    coinSubJ spec c₁ mids j =
      mapKey "streams" (mapArr (mapStream (fillG spec mids (eraseCoins j))))
        (mapKey "link" (mapKey "points" fun _ => .arr ((pointsOf spec c₁).map hexArr))
          (mapKey "streams" (mapArr (mapStream eraseG)) (mapKey "link" (mapKey "points" fun _ => .null) j))) :=
  rfl

theorem arr_streams_rounds (g : String → ℕ → Json → Json) (v j : Json) :
    Record.arr (mapRounds g (setPoints v j)) "streams" = (·.map (mapStream g)) <$> Record.arr j "streams" := by
  unfold mapRounds setPoints
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link")]

theorem arr_mapStream (g : String → ℕ → Json → Json) (s : Json) :
    Record.arr (mapStream g s) "rounds" = (·.mapIdx (g (nameOf s))) <$> Record.arr s "rounds" := by
  unfold mapStream
  exact arr_mapKey_mapArrIdx _ _ _

/-! ## The honest record's streams -/

theorem streamObj_name (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : (streamObj spec h rb q).getObjValAs? String "stream" = .ok q.1.2.2 := by
  unfold streamObj
  simp only [Json.getObjValAs?, Json.getObjValD, getObjVal?_mkObj (streamList_distinct spec h rb q)
    (show ("stream", Json.str q.1.2.2) ∈ streamList spec h rb q by simp [streamList])] <;> rfl

theorem streamObj_rounds (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) :
    Record.arr (streamObj spec h rb q) "rounds" = .ok ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!) := by
  unfold streamObj
  simp only [Record.arr, fld_of_gv (getObjVal?_mkObj (streamList_distinct spec h rb q)
    (show ("rounds", Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!)) ∈
      streamList spec h rb q by simp [streamList]))] <;> rfl

/-- **Under distinct names, the stream found by a name is the one at its index.** -/
theorem find_named {A : Array Json} {names : List String} (hn : names.Nodup) (hl : A.size = names.length)
    (hnm : ∀ (k : ℕ) (hk : k < A.size),
      (A[k].getObjValAs? String "stream").toOption = some (names[k]'(hl ▸ hk)))
    (i : ℕ) (hi : i < A.size) :
    A.find? (fun s => (s.getObjValAs? String "stream").toOption == some (names[i]'(hl ▸ hi))) = some A[i] := by
  have hnodup : (A.toList.map fun s => (s.getObjValAs? String "stream").toOption).Nodup := by
    have e : A.toList.map (fun s => (s.getObjValAs? String "stream").toOption) = names.map some := by
      refine List.ext_getElem (by simp [hl]) fun k h₁ h₂ => ?_
      simp only [List.getElem_map, Array.getElem_toList]
      exact hnm k (by simpa using h₁)
    rw [e]
    exact hn.map (Option.some_injective _)
  have := FlockComplete.ZkTables.find?_name (fun s : Json => (s.getObjValAs? String "stream").toOption) A.toList
    hnodup i (by simpa using hi)
  simp only [Array.getElem_toList, Array.find?_toList, hnm i hi] at this
  exact this

theorem find_streamObj {spec : SessionSpec} {h : ByteArray → ByteArray} {rb : Array ByteArray}
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (i : ℕ) (hi : i < spec.repStreams.size) :
    ((spec.repStreams.zip streams).map (streamObj spec h rb)).find?
        (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[i].2.2) =
      some (streamObj spec h rb (spec.repStreams[i], streams[i]'(hss ▸ hi))) := by
  have hl : ((spec.repStreams.zip streams).map (streamObj spec h rb)).size =
      (spec.repStreams.toList.map (·.2.2)).length := by simp [hss]
  have hA : ∀ (k : ℕ) (hk : k < ((spec.repStreams.zip streams).map (streamObj spec h rb)).size),
      ((spec.repStreams.zip streams).map (streamObj spec h rb))[k] =
        streamObj spec h rb (spec.repStreams[k]'(by simp [hss] at hk; omega), streams[k]'(by simp at hk; omega)) := by
    intro k hk
    simp [Array.getElem_zip]
  have := find_named hnd hl (fun k hk => by rw [hA k hk, streamObj_name]; simp <;> rfl) i (by simp [hss, hi])
  rw [hA i (by simp [hss, hi])] at this
  simpa using this

/-- **Where each rep stream's rounds take their coins**, in the erased record: the rounds of the rep streams before
it. -/
theorem streamOffset_erase {spec : SessionSpec} {h : ByteArray → ByteArray} {rb : Array ByteArray}
    {S : Array (Array Round)} (hss : S.size = spec.repStreams.size) (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    {J : Json} (hJs : Record.arr J "streams" = .ok ((spec.repStreams.zip S).map (streamObj spec h rb))) :
    ∀ (i : ℕ) (hi : i < spec.repStreams.size),
      streamOffset spec (eraseCoins J) spec.repStreams[i].2.2 = ((S.toList.take i).map Array.size).sum := by
  have hJ2s : Record.arr (eraseCoins J) "streams" =
      .ok (((spec.repStreams.zip S).map (streamObj spec h rb)).map (mapStream eraseG)) := by
    rw [eraseCoins_eq, arr_streams_rounds, hJs] <;> rfl
  have hrounds : ∀ (j : ℕ) (hj : j < spec.repStreams.size),
      roundsOf (eraseCoins J) spec.repStreams[j].2.2 = (S[j]'(hss ▸ hj)).size := by
    intro j hj
    have hf : (((spec.repStreams.zip S).map (streamObj spec h rb)).map (mapStream eraseG)).find?
        (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[j].2.2) =
        some (mapStream eraseG (streamObj spec h rb (spec.repStreams[j], S[j]'(hss ▸ hj)))) := by
      rw [find_mapStream, find_streamObj hss hnd j hj]
      rfl
    have ha : Record.arr (mapStream eraseG (streamObj spec h rb (spec.repStreams[j], S[j]'(hss ▸ hj))))
        "rounds" = .ok (((Array.range (S[j]'(hss ▸ hj)).size).map fun k =>
          roundObj spec h k (S[j]'(hss ▸ hj))[k]!).mapIdx (eraseG (nameOf (streamObj spec h rb
            (spec.repStreams[j], S[j]'(hss ▸ hj)))))) := by
      rw [arr_mapStream, streamObj_rounds] <;> rfl
    rw [roundsOf_of hJ2s hf ha]
    simp
  intro i hi
  unfold streamOffset
  have hni : spec.repStreams[i].2.2 = (spec.repStreams.toList.map (·.2.2))[i]'(by simpa using hi) := by simp
  have e : (spec.repStreams.toList.map (·.2.2)).map (roundsOf (eraseCoins J)) = S.toList.map Array.size := by
    apply List.ext_getElem (by simp [hss])
    intro j h₁ h₂
    have hj : j < spec.repStreams.size := by simpa using h₁
    simp only [List.getElem_map, Array.getElem_toList]
    exact hrounds j hj
  rw [hni, takeWhile_ne_nodup _ hnd i (by simpa using hi), List.map_take, List.map_take, e]

/-! ## Each substitution writes back the record's value -/

/-- **A round whose coin opens to its coins** is alike itself with its coins erased and filled from the coin. -/
theorem same_round_fill (spec : SessionSpec) (h : ByteArray → ByteArray) (k : ℕ) (r : Round) (c : Coin)
    (hc : c.vals r.n = r.coins) :
    Same (roundObj spec h k r) (mapKey "coins" (fun _ => hexArr (c.vals (nOf (mapKey "coins" (fun _ => .null)
      (roundObj spec h k r))))) (mapKey "coins" (fun _ => .null) (roundObj spec h k r))) := by
  have gRd : ∀ {key : String} {v : Json}, (key, v) ∈ roundList spec h k r →
      (roundObj spec h k r).getObjVal? key = .ok v :=
    fun hm => getObjVal?_mkObj (roundList_distinct ..) hm
  have hn0 : Record.nat (roundObj spec h k r) "n" = .ok r.n := by
    simp only [Record.nat, fld_of_gv (gRd (key := "n") (v := Lean.toJson r.n) (by simp [roundList]))] <;> rfl
  have hnO : nOf (mapKey "coins" (fun _ => .null) (roundObj spec h k r)) = r.n := by
    simp only [nOf, nat_mapKey_ne _ _ (by decide : "n" ≠ "coins"), hn0]
  rw [hnO, hc]
  have hg : Good (.obj (Std.TreeMap.Raw.ofList (roundList spec h k r) compare)) := good_roundObj spec h k r
  have hm : ∀ {key : String} {v : Json}, (key, v) ∈ roundList spec h k r →
      (Std.TreeMap.Raw.ofList (roundList spec h k r) compare).get? key = some v :=
    fun hm => get?_mkObj (roundList_distinct ..) hm
  have hco := hm (key := "coins") (v := Json.arr (r.coins.map f128J)) (by simp [roundList])
  change Same (.obj (Std.TreeMap.Raw.ofList (roundList spec h k r) compare))
    (mapKey "coins" (fun _ => hexArr r.coins) (mapKey "coins" (fun _ => .null)
      (.obj (Std.TreeMap.Raw.ofList (roundList spec h k r) compare))))
  generalize Std.TreeMap.Raw.ofList (roundList spec h k r) compare = mr at hg hco ⊢
  simp only [mapKey_obj]
  refine .obj (wf_of_good hg) (wf_map _ (wf_map _ (wf_of_good hg))) (fun k' => ?_) fun k' a b ha hb => ?_
  · simp only [get?_map]
    cases mr.get? k' <;> rfl
  · simp only [get?_map, ha, Option.map_some, Option.some.injEq] at hb
    subst hb
    by_cases e : k' = "coins"
    · subst e
      rw [hco] at ha
      cases ha
      simp only [↓reduceIte, hexArr]
      exact same_refl (FlockComplete.JsonRT.good_arr_str r.coins)
    · simp only [e, ↓reduceIte]
      exact same_refl (good_of_get hg ha)

/-- **The link at its own points** is alike itself with its points erased and written from the points. -/
theorem same_link_points (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (pts : Array (Array F128)) (hg : Good (linkObj spec h rb pts)) :
    Same (linkObj spec h rb pts) (mapKey "points" (fun _ => .arr (pts.map hexArr))
      (mapKey "points" (fun _ => .null) (linkObj spec h rb pts))) := by
  have hm : ∀ {key : String} {v : Json}, (key, v) ∈ linkList spec h rb pts →
      (Std.TreeMap.Raw.ofList (linkList spec h rb pts) compare).get? key = some v :=
    fun hm => get?_mkObj (linkList_distinct ..) hm
  have hpt := hm (key := "points") (v := Json.arr (pts.map fun p => Json.arr (p.map f128J))) (by simp [linkList])
  change Good (.obj (Std.TreeMap.Raw.ofList (linkList spec h rb pts) compare)) at hg
  change Same (.obj (Std.TreeMap.Raw.ofList (linkList spec h rb pts) compare))
    (mapKey "points" (fun _ => .arr (pts.map hexArr)) (mapKey "points" (fun _ => .null)
      (.obj (Std.TreeMap.Raw.ofList (linkList spec h rb pts) compare))))
  generalize Std.TreeMap.Raw.ofList (linkList spec h rb pts) compare = ml at hg hpt ⊢
  simp only [mapKey_obj]
  refine .obj (wf_of_good hg) (wf_map _ (wf_map _ (wf_of_good hg))) (fun k' => ?_) fun k' a b ha hb => ?_
  · simp only [get?_map]
    cases ml.get? k' <;> rfl
  · simp only [get?_map, ha, Option.map_some, Option.some.injEq] at hb
    subst hb
    by_cases e : k' = "points"
    · subst e
      have hga := good_of_get hg ha
      rw [hpt] at ha
      cases ha
      simp only [↓reduceIte]
      exact same_refl hga
    · simp only [e, ↓reduceIte]
      exact same_refl (good_of_get hg ha)

/-- **A rep stream at its coin offset**, each round's coin opening to the round's coins, is alike itself with its
rounds' coins erased and filled. -/
theorem same_stream_fill (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) (mids : List (Claims.Msg × Coin)) (E : Json) (off : ℕ)
    (hoff : streamOffset spec E q.1.2.2 = off)
    (hmid : ∀ (k : ℕ) (hk : k < q.2.size), ∃ m c, mids[off + k]? = some (m, c) ∧ c.vals q.2[k].n = q.2[k].coins) :
    Same (streamObj spec h rb q) (mapStream (fillG spec mids E) (mapStream eraseG (streamObj spec h rb q))) := by
  have hname : nameOf (mapKey "rounds" (mapArrIdx (eraseG (nameOf (streamObj spec h rb q)))) (streamObj spec h rb q)) =
      q.1.2.2 := by
    unfold nameOf
    rw [getObjValAs_mapKey_ne _ _ (by decide : "stream" ≠ "rounds"), streamObj_name]
    rfl
  unfold mapStream
  rw [hname]
  generalize nameOf (streamObj spec h rb q) = n₀
  have hg : Good (.obj (Std.TreeMap.Raw.ofList (streamList spec h rb q) compare)) := good_streamObj spec h rb q
  have hm : ∀ {key : String} {v : Json}, (key, v) ∈ streamList spec h rb q →
      (Std.TreeMap.Raw.ofList (streamList spec h rb q) compare).get? key = some v :=
    fun hm => get?_mkObj (streamList_distinct ..) hm
  have hro := hm (key := "rounds")
    (v := Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!)) (by simp [streamList])
  change Same (.obj (Std.TreeMap.Raw.ofList (streamList spec h rb q) compare))
    (mapKey "rounds" (mapArrIdx (fillG spec mids E q.1.2.2)) (mapKey "rounds" (mapArrIdx (eraseG n₀))
      (.obj (Std.TreeMap.Raw.ofList (streamList spec h rb q) compare))))
  generalize Std.TreeMap.Raw.ofList (streamList spec h rb q) compare = ms at hg hro ⊢
  simp only [mapKey_obj]
  refine .obj (wf_of_good hg) (wf_map _ (wf_map _ (wf_of_good hg))) (fun k' => ?_) fun k' a b ha hb => ?_
  · simp only [get?_map]
    cases ms.get? k' <;> rfl
  · simp only [get?_map, ha, Option.map_some, Option.some.injEq] at hb
    subst hb
    by_cases e : k' = "rounds"
    · subst e
      rw [hro] at ha
      cases ha
      simp only [↓reduceIte, mapArrIdx]
      refine .arr (by simp) fun i h₁ h₂ => ?_
      have hi : i < q.2.size := by simpa using h₁
      obtain ⟨m, c, hmc, hcv⟩ := hmid i hi
      simp only [Array.getElem_mapIdx, Array.getElem_map, Array.getElem_range, fillG, eraseG, hoff, hmc]
      refine same_round_fill spec h i q.2[i]! c ?_
      rw [getElem!_pos q.2 i hi]
      exact hcv
    · simp only [e, ↓reduceIte]
      exact same_refl (good_of_get hg ha)

/-- **The honest record survives the reader's substitutions**: with the proof digests written from the proofs and the
coins from the live coins, where each rep stream's round `k` takes, at its offset, a coin that opens to its coins, the
value is alike the record. -/
theorem same_coinSub_recordObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray) (c₁ : Coin)
    (S : Array (Array Round)) (proofs : Array ByteArray) (mids : List (Claims.Msg × Coin))
    (hR : Good (recordObj spec h rb (pointsOf spec c₁) S proofs))
    (hss : S.size = spec.repStreams.size) (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hmid : ∀ (i : ℕ) (hi : i < S.size) (k : ℕ) (hk : k < S[i].size), ∃ m c,
      mids[((S.toList.take i).map Array.size).sum + k]? = some (m, c) ∧ c.vals S[i][k].n = S[i][k].coins) :
    Same (recordObj spec h rb (pointsOf spec c₁) S proofs)
      (coinSubJ spec c₁ mids (shaSubJ spec (proofs.map (dgOf spec h))
        (recordObj spec h rb (pointsOf spec c₁) S proofs))) := by
  have hm : ∀ {key : String} {v : Json}, (key, v) ∈ recordList spec h rb (pointsOf spec c₁) S proofs →
      (Std.TreeMap.Raw.ofList (recordList spec h rb (pointsOf spec c₁) S proofs) compare).get? key = some v :=
    fun hm => get?_mkObj (recordList_distinct ..) hm
  have hJs : Record.arr (shaSubJ spec (proofs.map (dgOf spec h)) (recordObj spec h rb (pointsOf spec c₁) S proofs))
      "streams" = .ok ((spec.repStreams.zip S).map (streamObj spec h rb)) := by
    unfold shaSubJ
    rw [arr_mapKey_ne _ _ (proofKey_ne_streams spec).symm]
    unfold recordObj
    simp only [Record.arr, fld_of_gv (getObjVal?_mkObj (recordList_distinct spec h rb (pointsOf spec c₁) S proofs)
      (show ("streams", Json.arr ((spec.repStreams.zip S).map (streamObj spec h rb))) ∈
        recordList spec h rb (pointsOf spec c₁) S proofs by simp [recordList]))] <;> rfl
  have hoff := streamOffset_erase hss hnd hJs
  rw [coinSubJ_eq]
  generalize eraseCoins (shaSubJ spec (proofs.map (dgOf spec h)) (recordObj spec h rb (pointsOf spec c₁) S proofs)) =
    E at hoff ⊢
  have hst := hm (key := "streams") (v := Json.arr ((spec.repStreams.zip S).map (streamObj spec h rb)))
    (by simp [recordList])
  have hli := hm (key := "link") (v := linkObj spec h rb (pointsOf spec c₁)) (by simp [recordList])
  have hpf := hm (key := s!"proof_{spec.dk}") (v := Json.arr ((spec.repStreams.zip proofs).map (proofObj spec h)))
    (by simp [recordList])
  change Good (.obj (Std.TreeMap.Raw.ofList (recordList spec h rb (pointsOf spec c₁) S proofs) compare)) at hR
  unfold shaSubJ
  change Same (.obj (Std.TreeMap.Raw.ofList (recordList spec h rb (pointsOf spec c₁) S proofs) compare))
    (mapKey "streams" _ (mapKey "link" _ (mapKey "streams" _ (mapKey "link" _ (mapKey _ _
      (.obj (Std.TreeMap.Raw.ofList (recordList spec h rb (pointsOf spec c₁) S proofs) compare)))))))
  generalize Std.TreeMap.Raw.ofList (recordList spec h rb (pointsOf spec c₁) S proofs) compare = mR at hR hst hli hpf ⊢
  simp only [mapKey_obj]
  have hw := wf_of_good hR
  refine .obj hw (wf_map _ (wf_map _ (wf_map _ (wf_map _ (wf_map _ hw))))) (fun k => ?_) fun k a b ha hb => ?_
  · simp only [get?_map]
    cases mR.get? k <;> rfl
  · simp only [get?_map, ha, Option.map_some, Option.some.injEq] at hb
    subst hb
    have hga := good_of_get hR ha
    have hps := proofKey_ne_streams spec
    have hpl := proofKey_ne_link spec
    by_cases hs : k = "streams"
    · subst hs
      rw [hst] at ha
      cases ha
      simp only [↓reduceIte, Ne.symm hps, mapArr]
      refine .arr (by simp) fun i h₁ h₂ => ?_
      have hi : i < spec.repStreams.size := by simp at h₁; omega
      have hiS : i < S.size := by omega
      simp only [Array.getElem_map, Array.getElem_zip]
      exact same_stream_fill spec h rb (spec.repStreams[i], S[i]) mids E _ (hoff i hi) fun k hk => hmid i hiS k hk
    · by_cases hl : k = "link"
      · subst hl
        rw [hli] at ha
        cases ha
        simp only [↓reduceIte, Ne.symm hpl]
        exact same_link_points spec h rb (pointsOf spec c₁) hga
      · by_cases hp : k = s!"proof_{spec.dk}"
        · subst hp
          rw [hpf] at ha
          cases ha
          simp only [↓reduceIte, hps, hpl, shaArr_honest]
          exact same_refl hga
        · simp only [hs, hl, hp, ↓reduceIte]
          exact same_refl hga

end FlockDeploy.HonestJson
