import Proofs.Flock.Claims.Complete.Json.Session
import Proofs.Flock.Soundness.Discharge.Exec.Event
import Proofs.Flock.Claims.JsonNorm

/-!
# The honest record's live session and draw

Under the JSON library's specification (`JsonSpec`), at one table:

* `live_recordObj`: `liveOf` reads the printed honest record back as its `root_B`, no public words, its points and each
  rep stream's rounds with no digest (`liveRound`).
* `recordDraw_recordObj`: the record's draw is the statement's, up to key order (`none` when the statement has none).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy
open Lean (Json JsonNumber)

theorem strAs_same {j j' : Json} (h : Same j j') {k s : String} (hv : j.getObjVal? k = .ok (.str s)) :
    j'.getObjValAs? String k = .ok s := by
  obtain ⟨v', e, hs⟩ := h.getObjVal hv
  simp only [Json.getObjValAs?, Json.getObjValD, e, same_str hs]
  rfl

/-- A recorded round as `liveOf` reads it back: its count, coins and retained bytes, no digest. -/
def liveRound (r : Round) : Round := ⟨r.n, ByteArray.empty, r.coins, some (r.bytes.getD ByteArray.empty)⟩

theorem live_round {spec : SessionSpec} {h : ByteArray → ByteArray} {k : Nat} {r : Round} {r' : Json}
    (hr : Same (roundObj spec h k r) r') :
    Record.nat r' "n" = .ok r.n ∧ Record.arr r' "coins" = .ok (r.coins.map f128J) ∧ Record.isNull r' "msg" = false ∧
      Record.str r' "msg" = .ok (hexOf (r.bytes.getD ByteArray.empty)) := by
  have gRd : ∀ {key v}, (key, v) ∈ roundList spec h k r → (roundObj spec h k r).getObjVal? key = .ok v :=
    fun hm => getObjVal?_mkObj (roundList_distinct ..) hm
  exact ⟨nat_same hr (gRd (by simp [roundList])), arr_plain hr (gRd (by simp [roundList])) (plain_arr_str _),
    isNull_str hr (gRd (v := hexJ (r.bytes.getD ByteArray.empty)) (by simp [roundList])),
    str_same hr (gRd (v := hexJ (r.bytes.getD ByteArray.empty)) (by simp [roundList]))⟩

theorem live_stream {spec : SessionSpec} {h : ByteArray → ByteArray} {rootB : Array ByteArray}
    {q : (Nat × Nat × String) × Array Round} {s' : Json} (hq : Same (streamObj spec h rootB q) s') :
    s'.getObjValAs? String "stream" = .ok q.1.2.2 ∧ ∃ rj : Array Json, Record.arr s' "rounds" = .ok rj ∧
      q.2.size = rj.size ∧ ∀ (k : Nat) (h₁ : k < q.2.size) (h₂ : k < rj.size), Same (roundObj spec h k q.2[k]) rj[k] := by
  have gS : ∀ {k v}, (k, v) ∈ streamList spec h rootB q → (streamObj spec h rootB q).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj (streamList_distinct ..) hm
  refine ⟨strAs_same hq (gS (by simp [streamList])), ?_⟩
  obtain ⟨rj, hrj, hrjs, hrje⟩ := arr_same hq (gS (k := "rounds")
    (v := Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!)) (by simp [streamList]))
  refine ⟨rj, hrj, by simpa using hrjs, fun k h₁ h₂ => ?_⟩
  have := hrje k (by simpa using h₁) h₂
  simpa [getElem!_pos q.2 k h₁] using this

/-- **`liveOf` reads the printed honest record** at one table: its `root_B`, no public words, its points and each rep
stream's rounds without digests. -/
theorem live_recordObj (S : Claims.JsonSpec printJ Flock.canon) {h : ByteArray → ByteArray} {spec : SessionSpec}
    {t : String} (ht : spec.tables = #[t]) (hd : Good (spec.unitDraw.getD Json.null)) {rb : ByteArray}
    {points : Array (Array F128)} {streams : Array (Array Round)} (hss : streams.size = spec.repStreams.size)
    (proofs : Array ByteArray) :
    liveOf spec (printJ (recordObj spec h #[rb] points streams proofs)) =
      .ok ⟨#[rb], spec.tables.map fun _ => ByteArray.empty, points, streams.map (·.map liveRound)⟩ := by
  obtain ⟨j, hp, hs⟩ := S.parse_print _ (good_recordObj ht h rb points streams proofs hd)
  have gR : ∀ {k v}, (k, v) ∈ recordList spec h #[rb] points streams proofs →
      (recordObj spec h #[rb] points streams proofs).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj (recordList_distinct ..) hm
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

/-- **The record's draw is the statement's**, up to key order (`JsonEquiv`). -/
theorem recordDraw_recordObj (S : Claims.JsonSpec printJ Flock.canon) {h : ByteArray → ByteArray} {spec : SessionSpec}
    {t : String} (ht : spec.tables = #[t]) (hdraw : DrawOk spec) (rb : ByteArray) (points : Array (Array F128))
    (streams : Array (Array Round)) (proofs : Array ByteArray) :
    match spec.unitDraw with
    | none => FlockSoundness.Discharge.Exec.recordDraw (printJ (recordObj spec h #[rb] points streams proofs)) = none
    | some d => ∃ d', FlockSoundness.Discharge.Exec.recordDraw (printJ (recordObj spec h #[rb] points streams proofs)) =
        some d' ∧ FlockDeploy.JsonEquiv d' d := by
  have hgood : Good (spec.unitDraw.getD Json.null) := by
    unfold DrawOk at hdraw
    revert hdraw
    cases spec.unitDraw with
    | none => exact fun _ => .null
    | some d => exact fun hd => hd.2.1
  obtain ⟨j, hp, hs⟩ := S.parse_print _ (good_recordObj ht h rb points streams proofs hgood)
  obtain ⟨dv, hdv, hdvs⟩ := hs.getObjVal (getObjVal?_mkObj (recordList_distinct ..)
    (k := "unit_draw") (v := spec.unitDraw.getD Json.null) (by simp [recordList]))
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

end FlockComplete.JsonRT
