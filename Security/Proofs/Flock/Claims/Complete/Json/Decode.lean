import Proofs.Flock.Claims.Complete.Json.Record

/-!
# The honest record decodes (S1–S14)

`decode_recordObj`: under the JSON library's specification (`JsonSpec`), `Record.decodeWith` through
`Hashes.ofSha512 h` reads the printed honest record (`printJ` of `recordObj`) back as `sessOf`, its every check passing,
for one table, no keyed coin tree, a printable ASCII draw that decodes (`DrawOk`), a non-empty `Σ` and `root_B`, the
configured point shape, one recorded stream per rep stream and well-formed rounds, with `h`'s outputs 64 bytes (S12
compares each round's digest length with the empty message's).

* **Reads through `mapM` and `mapIdxM`** (`array_mapM_ok_of`, `array_mapIdxM_ok_of`): elementwise returns compose.
* **The decoded session** (`sessOf`): `Σ` as `root_F`, the `root_B`, no public words, the points, `y = 0`, each rep
  stream with its rounds' digests (`streamRec`, `roundRec`) and each proof file's digest.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete
open Lean (Json JsonNumber)

theorem list_mapM_ok_of {α β : Type} (f : α → V β) :
    ∀ (l : List α) (ys : List β), l.length = ys.length →
      (∀ (i : Nat) (h₁ : i < l.length) (h₂ : i < ys.length), f l[i] = .ok ys[i]) → l.mapM f = .ok ys
  | [], [], _, _ => rfl
  | a :: l, b :: ys, hs, hf => by
    have h0 := hf 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    rw [List.mapM_cons, h0, ok_bind,
      list_mapM_ok_of f l ys (by simpa using hs) fun i h₁ h₂ => by
        have := hf (i + 1) (by simp; omega) (by simp; omega)
        simpa only [List.getElem_cons_succ] using this]
    rfl
  | [], _ :: _, hs, _ => by simp at hs
  | _ :: _, [], hs, _ => by simp at hs

theorem array_mapM_ok_of {α β : Type} (f : α → V β) (a : Array α) (ys : Array β) (hs : a.size = ys.size)
    (hf : ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < ys.size), f a[i] = .ok ys[i]) : a.mapM f = .ok ys := by
  rw [Array.mapM_eq_mapM_toList, list_mapM_ok_of f a.toList ys.toList (by simpa using hs)
    fun i h₁ h₂ => by simpa using hf i (by simpa using h₁) (by simpa using h₂)]
  rfl

theorem f128s_map (a : Array F128) : Record.f128s (a.map f128J) = .ok a := by
  unfold Record.f128s
  rw [array_mapM_ok_of _ _ a (by simp) fun i h₁ h₂ => ?_]
  simp only [Array.getElem_map, f128J, Json.getStr?]
  exact f128OfHex_toHex _

theorem mapIdxM_go_ok_of {α β : Type} (f : Nat → α → V β) :
    ∀ (l : List α) (acc : Array β) (zs : List β), l.length = zs.length →
      (∀ (i : Nat) (h₁ : i < l.length) (h₂ : i < zs.length), f (acc.size + i) l[i] = .ok zs[i]) →
        List.mapIdxM.go f l acc = .ok (acc.toList ++ zs)
  | [], acc, [], _, _ => by simp [List.mapIdxM.go]; rfl
  | a :: l, acc, z :: zs, hs, hf => by
    have h0 := hf 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero, Nat.add_zero] at h0
    rw [List.mapIdxM.go, h0, ok_bind, mapIdxM_go_ok_of f l (acc.push z) zs (by simpa using hs) fun i h₁ h₂ => by
      have := hf (i + 1) (by simp; omega) (by simp; omega)
      simpa only [List.getElem_cons_succ, Array.size_push, show acc.size + 1 + i = acc.size + (i + 1) by omega]
        using this]
    simp
  | [], _, _ :: _, hs, _ => by simp at hs
  | _ :: _, _, [], hs, _ => by simp at hs

theorem array_mapIdxM_ok_of {α β : Type} (f : Nat → α → V β) (a : Array α) (ys : Array β) (hs : a.size = ys.size)
    (hf : ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < ys.size), f i a[i] = .ok ys[i]) : a.mapIdxM f = .ok ys := by
  rw [← Array.toArray_toList (xs := a), List.mapIdxM_toArray, List.mapIdxM,
    mapIdxM_go_ok_of f a.toList #[] ys.toList (by simpa using hs) fun i h₁ h₂ => by
      simpa using hf i (by simpa using h₁) (by simpa using h₂)]
  simp

/-- What the decoder's U1–U3 need of the statement's draw: none, or a non-null printable ASCII draw that decodes, its
proved units the statement's instances. -/
def DrawOk (spec : SessionSpec) : Prop :=
  match spec.unitDraw with
  | none => True
  | some d => d ≠ Json.null ∧ Good d ∧ Ascii d ∧ ∃ D, Draw.UnitDraw.ofJson d = .ok D ∧ D.proved.size = spec.instances

/-- A recorded round as the decoder reads it back: its digest the retained bytes'. -/
def roundRec (spec : SessionSpec) (h : ByteArray → ByteArray) (r : Round) : Round :=
  ⟨r.n, dgOf spec h (r.bytes.getD ByteArray.empty), r.coins, some (r.bytes.getD ByteArray.empty)⟩

/-- A rep stream as the decoder reads it back. -/
def streamRec (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (Nat × Nat × String) × Array Round) : StreamRec :=
  ⟨q.1.2.2, "", spec.domain spec.tables[q.1.1]! q.1.2.1, 0, some (hexOf (dgOf spec h rootB[q.1.1]!)),
    q.2.map (roundRec spec h)⟩

/-- **The honest record, decoded.** -/
def sessOf (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray) (points : Array (Array F128))
    (streams : Array (Array Round)) (proofs : Array ByteArray) : RunRecord :=
  { rootF := spec.sigma, rootB, publics := spec.tables.map fun _ => ByteArray.empty, points, y := yOf spec,
    streams := (spec.repStreams.zip streams).map (streamRec spec h rootB),
    proofSha := (spec.repStreams.zip proofs).map fun q => (q.1.2.2, hexOf (dgOf spec h q.2)) }

/-- **The honest record decodes** (S1–S14) as `sessOf`: one table `t`, no keyed coin tree, the statement's draw
`DrawOk`, a non-empty `Σ` and `root_B`, the configured points, one recorded stream per rep stream, each with rounds of
`1 ≤ n ≤ 65536` coins, and `h`'s outputs 64 bytes. -/
theorem decode_recordObj (S : Claims.JsonSpec printJ Flock.canon) {h : ByteArray → ByteArray}
    (h64 : ∀ b, (h b).size = 64) {spec : SessionSpec} {t : String} (ht : spec.tables = #[t])
    (hct : spec.coinTree = none)
    (hdraw : DrawOk spec) (hsig : 0 < spec.sigma.size)
    {rb : ByteArray} (hrb : 0 < rb.size)
    {points : Array (Array F128)} (hpts : points.size = spec.points) (hpm : ∀ p ∈ points, p.size = spec.mPts)
    {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (hrs : ∀ rs ∈ streams, 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n)
    (proofs : Array ByteArray) :
    Record.decodeWith (Hashes.ofSha512 h : Hashes V) spec (printJ (recordObj spec h #[rb] points streams proofs)) =
      .ok (sessOf spec h #[rb] points streams proofs) := by
  have hgood := good_recordObj ht h rb points streams proofs (by
    unfold DrawOk at hdraw
    revert hdraw
    cases spec.unitDraw with
    | none => exact fun _ => .null
    | some d => exact fun hd => hd.2.1)
  obtain ⟨j, hp, hs⟩ := S.parse_print _ hgood
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
      obtain ⟨c, hc⟩ := S.canon_ok d hg ha
      have hc' : canon dv = .ok c := by rw [← S.canon_same _ _ hdvs, hc]
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
  have hrf : Record.str l "root_f" = .ok (hexOf spec.sigma) := str_same hls (gL (v := hexJ spec.sigma) (by simp [linkList]))
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

end FlockComplete.JsonRT
