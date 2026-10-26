import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.Claims.Recursion.NextShaDec

/-!
# Custody at the reader's record

At a decision of `nextR`, the record the checks read is the reader's own (`recJ`, through `decR`): the honest record's
form at the link's `root_B`, the first coin's points and the reader's streams (`recStreams`). Custody's reading
(`liveOfJ`) of that record with its proof digests and coins substituted (`jAt`) reads back the link's `root_B`, no
public words, and each round's message, so custody passes (`custody_decR`) at a statement with no public inputs (its
tables, and so its rep streams, are named apart) on as many coins as messages. No JSON premise is taken: the record is
a value the reader builds, never parsed text.

* **The record read back** (`liveOfJ_rec`): at any map of the rounds that `roundJ` reads with the round's message,
  `liveOfJ` reads the record's `root_B`, no public words, and each round's message, stream by stream.
* **The coin substitution fills every round** (`streamOffset_rec`): rep stream `i`'s rounds start at the rounds of the
  streams before it, so on as many coins as rounds each round finds its coin.
* **At `decR`** (`custody_decR`): the reader's streams carry the round messages in order (`recStreams_bytes`), as many
  as the messages (`readPs_length`).
-/

namespace FlockDeploy.RestCustody

open Flock
open Lean (Json)
open FlockComplete.JsonRT (recordObj recordList linkObj linkList streamObj streamList roundObj roundList rootBList
  Distinct getObjVal?_mkObj recordList_distinct linkList_distinct streamList_distinct roundList_distinct dk_cases
  f128s_map array_mapM_ok_of unhex_hexOf bytes_beq_self)

/-! ## Generic facts -/

theorem fld_ok {j : Json} {k : String} {v : Json} (h : j.getObjVal? k = .ok v) : Record.fld j k = .ok v := by
  unfold Record.fld
  rw [h]

theorem array_mapM_ok_ex {α β : Type} (f : α → Flock.V β) (a : Array α) (P : ℕ → β → Prop)
    (h : ∀ (i : ℕ) (hi : i < a.size), ∃ y, f a[i] = .ok y ∧ P i y) :
    ∃ ys : Array β, a.mapM f = .ok ys ∧ ys.size = a.size ∧ ∀ (i : ℕ) (hi : i < a.size) (h' : i < ys.size), P i ys[i] := by
  choose g hg using h
  refine ⟨Array.ofFn fun i : Fin a.size => g i.1 i.2, ?_, by simp, fun i hi h' => ?_⟩
  · exact array_mapM_ok_of f a _ (by simp) fun i h₁ h₂ => by simpa using (hg i h₁).1
  · simpa using (hg i hi).2

theorem flatMap_congr_idx {α β γ : Type} {f : α → List γ} {g : β → List γ} :
    ∀ (A : List α) (B : List β), A.length = B.length →
      (∀ (i : ℕ) (h₁ : i < A.length) (h₂ : i < B.length), f A[i] = g B[i]) → A.flatMap f = B.flatMap g
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl
  | a :: A, b :: B, hl, h => by
    have h0 := h 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    simp only [List.flatMap_cons, h0]
    congr 1
    exact flatMap_congr_idx A B (by simpa using hl) fun i h₁ h₂ => by
      simpa using h (i + 1) (by simpa using h₁) (by simpa using h₂)

theorem getElem_of_map_eq {α β : Type} {f : α → β} {L : List α} {M : List β} (h : L.map f = M) (t : ℕ)
    (h₁ : t < L.length) (h₂ : t < M.length) : f L[t] = M[t] := by
  subst h
  simp

theorem sum_take_add_getElem_le : ∀ (l : List ℕ) (i : ℕ) (hi : i < l.length), (l.take i).sum + l[i] ≤ l.sum
  | a :: l, 0, _ => by simp
  | a :: l, i + 1, hi => by
    have := sum_take_add_getElem_le l i (by simpa using hi)
    simp only [List.take_succ_cons, List.sum_cons, List.getElem_cons_succ]
    omega

theorem list_bytes_beq_self (l : List ByteArray) : (l == l) = true := by
  induction l with
  | nil => rfl
  | cons b l ih => simp only [List.cons_beq_cons, bytes_beq_self, ih, Bool.and_self]

theorem cutLens_length {α : Type} : ∀ (ns : List ℕ) (l : List α), (cutLens ns l).length = ns.length
  | [], _ => rfl
  | n :: ns, l => by simp [cutLens, cutLens_length ns]

theorem flatMap_cutLens_map {α β : Type} (f : α → β) :
    ∀ (ns : List ℕ) (l : List α), l.length ≤ ns.sum → (cutLens ns l).flatMap (fun b => b.map f) = l.map f
  | [], l, h => by
    cases l with
    | nil => rfl
    | cons a l => simp at h
  | n :: ns, l, h => by
    simp only [cutLens, List.flatMap_cons]
    rw [flatMap_cutLens_map f ns (l.drop n) (by simp only [List.length_drop, List.sum_cons] at h ⊢; omega),
      ← List.map_append, List.take_append_drop]

/-! ## Keys -/

theorem link_ne_proof (spec : SessionSpec) : ("link" : String) ≠ s!"proof_{spec.dk}" := by
  rcases dk_cases spec with e | e <;> rw [e] <;> decide

theorem streams_ne_proof (spec : SessionSpec) : ("streams" : String) ≠ s!"proof_{spec.dk}" := by
  rcases dk_cases spec with e | e <;> rw [e] <;> decide

/-- The rep streams' `root_B` object has distinct keys where the tables are named apart. -/
theorem rootBList_distinct (spec : SessionSpec) (f : ByteArray → ByteArray) {rootB : Array ByteArray}
    (htn : spec.tables.toList.Nodup) (hrb : rootB.size = spec.tables.size) : Distinct (rootBList spec f rootB) := by
  have h2 : ((spec.tables.zip rootB).toList.map Prod.fst).Nodup := by
    rw [Array.toList_zip, List.map_fst_zip (by simp [hrb])]
    exact htn
  unfold Distinct rootBList
  rw [List.pairwise_map]
  rw [List.Nodup, List.pairwise_map] at h2
  exact h2.imp fun h e => h (Std.LawfulEqOrd.eq_of_compare e)

/-! ## Reading the reader's record -/

section Read

variable (spec : SessionSpec) (hh : ByteArray → ByteArray) (rootB : Array ByteArray) (pts : Array (Array F128))
  (S : Array (Array Round)) (files ds : Array ByteArray)

theorem fld_record_link :
    Record.fld (recordObj spec hh rootB pts S files) "link" = .ok (linkObj spec hh rootB pts) :=
  fld_ok (j := recordObj spec hh rootB pts S files)
    (getObjVal?_mkObj (recordList_distinct ..) (by simp [recordList]))

theorem arr_record_streams :
    Record.arr (recordObj spec hh rootB pts S files) "streams" =
      .ok ((spec.repStreams.zip S).map (streamObj spec hh rootB)) := by
  unfold Record.arr
  rw [fld_ok (j := recordObj spec hh rootB pts S files) (getObjVal?_mkObj (recordList_distinct ..) (k := "streams")
    (v := Json.arr ((spec.repStreams.zip S).map (streamObj spec hh rootB))) (by simp [recordList]))]
  rfl

/-- The record's link, its points erased and then set to `P`. -/
theorem fld_link (g : String → ℕ → Json → Json) (P : Json) :
    Record.fld (mapKey "streams" (mapArr (mapStream g)) (mapKey "link" (mapKey "points" fun _ => P)
      (eraseCoins (shaSubJ spec ds (recordObj spec hh rootB pts S files))))) "link" =
      .ok (mapKey "points" (fun _ => P) (mapKey "points" (fun _ => .null) (linkObj spec hh rootB pts))) := by
  unfold eraseCoins mapRounds setPoints shaSubJ
  rw [fld_mapKey_ne _ _ (by decide : ("link" : String) ≠ "streams"), fld_mapKey_eq,
    fld_mapKey_ne _ _ (by decide : ("link" : String) ≠ "streams"), fld_mapKey_eq,
    fld_mapKey_ne _ _ (link_ne_proof spec), fld_record_link]
  rfl

theorem fld_link_rootB (P : Json) :
    Record.fld (mapKey "points" (fun _ => P) (mapKey "points" (fun _ => .null) (linkObj spec hh rootB pts)))
      "root_b" = .ok (Json.mkObj (rootBList spec id rootB)) := by
  rw [fld_mapKey_ne _ _ (by decide : ("root_b" : String) ≠ "points"),
    fld_mapKey_ne _ _ (by decide : ("root_b" : String) ≠ "points")]
  exact fld_ok (j := linkObj spec hh rootB pts) (getObjVal?_mkObj (linkList_distinct ..) (by simp [linkList]))

theorem fld_link_publics (P : Json) :
    Record.fld (mapKey "points" (fun _ => P) (mapKey "points" (fun _ => .null) (linkObj spec hh rootB pts)))
      "publics" = .ok (Json.mkObj []) := by
  rw [fld_mapKey_ne _ _ (by decide : ("publics" : String) ≠ "points"),
    fld_mapKey_ne _ _ (by decide : ("publics" : String) ≠ "points")]
  exact fld_ok (j := linkObj spec hh rootB pts) (getObjVal?_mkObj (linkList_distinct ..) (by simp [linkList]))

theorem arr_link_points (P : Array Json) :
    Record.arr (mapKey "points" (fun _ => Json.arr P) (mapKey "points" (fun _ => .null) (linkObj spec hh rootB pts)))
      "points" = .ok P := by
  unfold Record.arr
  rw [fld_mapKey_eq, fld_mapKey_eq, fld_ok (j := linkObj spec hh rootB pts) (getObjVal?_mkObj (linkList_distinct ..)
    (k := "points") (v := Json.arr (pts.map fun p => Json.arr (p.map FlockComplete.f128J))) (by simp [linkList]))]
  rfl

theorem str_rootB {rootB : Array ByteArray} (htn : spec.tables.toList.Nodup) (hrb : rootB.size = spec.tables.size)
    (i : ℕ) (hi : i < spec.tables.size) :
    Record.str (Json.mkObj (rootBList spec id rootB)) spec.tables[i] = .ok (hexOf (rootB[i]'(by omega))) := by
  have hmem : (spec.tables[i], FlockComplete.hexJ (id (rootB[i]'(by omega)))) ∈ rootBList spec id rootB := by
    unfold rootBList
    refine List.mem_map.2 ⟨(spec.tables[i], rootB[i]'(by omega)), ?_, rfl⟩
    rw [Array.mem_toList_iff, Array.mem_iff_getElem]
    exact ⟨i, by simp [hrb, hi], by simp [Array.getElem_zip]⟩
  unfold Record.str
  rw [fld_ok (getObjVal?_mkObj (rootBList_distinct spec id htn hrb) hmem)]
  rfl

theorem isNull_empty (t : String) : Record.isNull (Json.mkObj []) t = true := by
  have hn := FlockComplete.JsonRT.get?_mkObj_none (l := []) (k := t) rfl
  unfold Record.isNull
  simp only [Json.mkObj, Json.getObjVal?, hn] <;> rfl

theorem arr_erase :
    Record.arr (eraseCoins (shaSubJ spec ds (recordObj spec hh rootB pts S files))) "streams" =
      .ok (((spec.repStreams.zip S).map (streamObj spec hh rootB)).map
        (mapStream fun _ _ => mapKey "coins" fun _ => Json.null)) := by
  unfold eraseCoins mapRounds setPoints shaSubJ
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : ("streams" : String) ≠ "link"),
    arr_mapKey_ne _ _ (streams_ne_proof spec), arr_record_streams]
  rfl

theorem arr_streams (g : String → ℕ → Json → Json) (P : Json) :
    Record.arr (mapKey "streams" (mapArr (mapStream g)) (mapKey "link" (mapKey "points" fun _ => P)
      (eraseCoins (shaSubJ spec ds (recordObj spec hh rootB pts S files))))) "streams" =
      .ok ((((spec.repStreams.zip S).map (streamObj spec hh rootB)).map
        (mapStream fun _ _ => mapKey "coins" fun _ => Json.null)).map (mapStream g)) := by
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : ("streams" : String) ≠ "link"), arr_erase]
  rfl

theorem streamObj_name (q : (ℕ × ℕ × String) × Array Round) :
    (streamObj spec hh rootB q).getObjValAs? String "stream" = .ok q.1.2.2 := by
  have e : (streamObj spec hh rootB q).getObjVal? "stream" = .ok (Json.str q.1.2.2) :=
    getObjVal?_mkObj (streamList_distinct ..) (by simp [streamList])
  simp only [Json.getObjValAs?, Json.getObjValD, e] <;> rfl

theorem nameOf_streamObj (q : (ℕ × ℕ × String) × Array Round) : nameOf (streamObj spec hh rootB q) = q.1.2.2 := by
  simp only [nameOf, streamObj_name] <;> rfl

theorem nameOf_mapStream (g : String → ℕ → Json → Json) (s : Json) : nameOf (mapStream g s) = nameOf s := by
  simp only [nameOf, mapStream, getObjValAs_mapKey_ne _ _ (by decide : ("stream" : String) ≠ "rounds")]

theorem streamObj_rounds (q : (ℕ × ℕ × String) × Array Round) :
    Record.arr (streamObj spec hh rootB q) "rounds" =
      .ok ((Array.range q.2.size).map fun k => roundObj spec hh k q.2[k]!) := by
  unfold Record.arr
  rw [fld_ok (j := streamObj spec hh rootB q) (getObjVal?_mkObj (streamList_distinct ..) (k := "rounds")
    (v := Json.arr ((Array.range q.2.size).map fun k => roundObj spec hh k q.2[k]!)) (by simp [streamList]))]
  rfl

theorem arr_mapStream (g : String → ℕ → Json → Json) {s : Json} {name : String} {a : Array Json}
    (hn : nameOf s = name) (ha : Record.arr s "rounds" = .ok a) :
    Record.arr (mapStream g s) "rounds" = .ok (a.mapIdx (g name)) := by
  unfold mapStream
  rw [hn, arr_mapKey_mapArrIdx, ha]
  rfl

/-- **Rep stream `i` is the `i`-th recorded stream** where the rep streams are named apart. -/
theorem find_streamObj (hS : S.size = spec.repStreams.size) (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (i : ℕ) (hi : i < spec.repStreams.size) :
    ((spec.repStreams.zip S).map (streamObj spec hh rootB)).find?
        (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[i].2.2) =
      some (streamObj spec hh rootB (spec.repStreams[i], S[i]'(by omega))) := by
  have hz : ∀ (k : ℕ) (hk : k < spec.repStreams.size),
      ((spec.repStreams.zip S).map (streamObj spec hh rootB))[k]'(by simp [hS, hk]) =
        streamObj spec hh rootB (spec.repStreams[k], S[k]'(by omega)) := by
    intro k hk
    simp [Array.getElem_zip]
  rw [Array.find?_eq_some_iff_getElem]
  refine ⟨by rw [streamObj_name]; simp, i, by simp [hS, hi], hz i hi, fun k hk => ?_⟩
  have hk' : k < spec.repStreams.size := by omega
  rw [hz k hk', streamObj_name]
  have hne : spec.repStreams[k].2.2 ≠ spec.repStreams[i].2.2 := by
    intro e
    have e' : (spec.repStreams.toList.map (·.2.2))[k]'(by simpa using hk') =
        (spec.repStreams.toList.map (·.2.2))[i]'(by simpa using hi) := by simpa using e
    have := (List.Nodup.getElem_inj_iff hnd).1 e'
    omega
  simpa using hne

/-- **A round with its coins erased and then set reads them**, with its count and message. -/
theorem roundJ_sub (k : ℕ) (r : Round) (v : Array F128) :
    roundJ (mapKey "coins" (fun _ => hexArr v) (mapKey "coins" (fun _ => .null) (roundObj spec hh k r))) =
      .ok ⟨r.n, ByteArray.empty, v, some (r.bytes.getD ByteArray.empty)⟩ := by
  have gR : ∀ {key : String} {w : Json}, (key, w) ∈ roundList spec hh k r → (roundObj spec hh k r).getObjVal? key = .ok w :=
    fun hm => getObjVal?_mkObj (roundList_distinct ..) hm
  have hn : Record.nat (roundObj spec hh k r) "n" = .ok r.n := by
    unfold Record.nat
    rw [fld_ok (gR (key := "n") (w := Lean.toJson r.n) (by simp [roundList]))]
    rfl
  have hc : Record.arr (mapKey "coins" (fun _ => hexArr v) (mapKey "coins" (fun _ => .null) (roundObj spec hh k r)))
      "coins" = .ok (v.map FlockComplete.f128J) := by
    unfold Record.arr
    rw [fld_mapKey_eq, fld_mapKey_eq,
      fld_ok (gR (key := "coins") (w := Json.arr (r.coins.map FlockComplete.f128J)) (by simp [roundList]))]
    rfl
  have hm : (roundObj spec hh k r).getObjVal? "msg" = .ok (FlockComplete.hexJ (r.bytes.getD ByteArray.empty)) :=
    gR (by simp [roundList])
  have hnull : Record.isNull (mapKey "coins" (fun _ => hexArr v) (mapKey "coins" (fun _ => .null)
      (roundObj spec hh k r))) "msg" = false := by
    unfold Record.isNull
    rw [getObjVal?_mapKey_ne _ _ (by decide : ("msg" : String) ≠ "coins"),
      getObjVal?_mapKey_ne _ _ (by decide : ("msg" : String) ≠ "coins"), hm]
    rfl
  have hstr : Record.str (mapKey "coins" (fun _ => hexArr v) (mapKey "coins" (fun _ => .null)
      (roundObj spec hh k r))) "msg" = .ok (hexOf (r.bytes.getD ByteArray.empty)) := by
    unfold Record.str
    rw [fld_mapKey_ne _ _ (by decide : ("msg" : String) ≠ "coins"),
      fld_mapKey_ne _ _ (by decide : ("msg" : String) ≠ "coins"), fld_ok hm]
    rfl
  unfold roundJ
  rw [nat_mapKey_ne _ _ (by decide : ("n" : String) ≠ "coins"), nat_mapKey_ne _ _ (by decide : ("n" : String) ≠ "coins"),
    hn]
  simp only [FlockComplete.ok_bind, hc, f128s_map, hnull, hstr, unhex_hexOf, Bool.false_eq_true, ite_false]
  rfl

/-- **Where each rep stream's rounds start**: after the rounds of the streams before it. -/
theorem streamOffset_rec (hS : S.size = spec.repStreams.size) (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (i : ℕ) (hi : i < spec.repStreams.size) :
    streamOffset spec (eraseCoins (shaSubJ spec ds (recordObj spec hh rootB pts S files))) spec.repStreams[i].2.2 =
      ((S.toList.map Array.size).take i).sum := by
  have hro : ∀ (i' : ℕ) (hi' : i' < spec.repStreams.size),
      roundsOf (eraseCoins (shaSubJ spec ds (recordObj spec hh rootB pts S files))) spec.repStreams[i'].2.2 =
        (S[i']'(by omega)).size := by
    intro i' hi'
    have hf : (((spec.repStreams.zip S).map (streamObj spec hh rootB)).map
        (mapStream fun _ _ => mapKey "coins" fun _ => Json.null)).find?
          (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[i'].2.2) =
        some (mapStream (fun _ _ => mapKey "coins" fun _ => Json.null)
          (streamObj spec hh rootB (spec.repStreams[i'], S[i']'(by omega)))) := by
      rw [find_mapStream, find_streamObj spec hh rootB S hS hnd i' hi']
      rfl
    rw [roundsOf_of (arr_erase spec hh rootB pts S files ds) hf
      (arr_mapStream _ (nameOf_streamObj spec hh rootB _) (streamObj_rounds spec hh rootB _))]
    simp
  unfold streamOffset
  have hnames : spec.repStreams[i].2.2 = (spec.repStreams.toList.map (·.2.2))[i]'(by simpa using hi) := by simp
  rw [hnames, takeWhile_ne_nodup _ hnd i (by simpa using hi), List.map_take]
  congr 1
  apply List.ext_getElem (by simp [hS])
  intro i' h₁ h₂
  simp only [List.getElem_map, Array.getElem_toList]
  exact hro i' (by simpa using h₁)

/-- **`liveOfJ` reads the reader's record back**: at a map `g` of the rounds that `roundJ` reads with each round's
message, the record's `root_B`, no public words, and each round's message, stream by stream. -/
theorem liveOfJ_rec (g : String → ℕ → Json → Json) (htn : spec.tables.toList.Nodup)
    (hrb : rootB.size = spec.tables.size) (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hS : S.size = spec.repStreams.size)
    (hg : ∀ (i : ℕ) (hi : i < spec.repStreams.size) (k : ℕ) (hk : k < (S[i]'(by omega)).size), ∃ rd : Round,
      roundJ (g spec.repStreams[i].2.2 k
        (mapKey "coins" (fun _ => .null) (roundObj spec hh k ((S[i]'(by omega))[k]'hk)))) = .ok rd ∧
      rd.bytes = some (((S[i]'(by omega))[k]'hk).bytes.getD ByteArray.empty)) :
    ∃ lv : Live, liveOfJ spec (mapKey "streams" (mapArr (mapStream g)) (mapKey "link"
        (mapKey "points" fun _ => Json.arr (pts.map hexArr))
        (eraseCoins (shaSubJ spec ds (recordObj spec hh rootB pts S files))))) = .ok lv ∧
      lv.rootB = rootB ∧ lv.publics = spec.tables.map (fun _ => ByteArray.empty) ∧
      (lv.streams.toList.flatMap fun s : Array Round => s.toList).map Round.bytes =
        (S.toList.flatMap fun s : Array Round => s.toList).map fun r => some (r.bytes.getD ByteArray.empty) := by
  have hst : ∀ (i : ℕ) (hi : i < spec.repStreams.size), ∃ rs : Array Round,
      streamJ ((((spec.repStreams.zip S).map (streamObj spec hh rootB)).map
        (mapStream fun _ _ => mapKey "coins" fun _ => Json.null)).map (mapStream g)) spec.repStreams[i].2.2 = .ok rs ∧
      rs.size = (S[i]'(by omega)).size ∧ ∀ (k : ℕ) (hk : k < (S[i]'(by omega)).size) (h' : k < rs.size),
        rs[k].bytes = some (((S[i]'(by omega))[k]'hk).bytes.getD ByteArray.empty) := by
    intro i hi
    have hSi : i < S.size := by omega
    have hn0 : nameOf (mapStream (fun _ _ => mapKey "coins" fun _ => Json.null)
        (streamObj spec hh rootB (spec.repStreams[i], S[i]))) = spec.repStreams[i].2.2 := by
      rw [nameOf_mapStream, nameOf_streamObj]
    have ha := arr_mapStream g hn0 (arr_mapStream _ (nameOf_streamObj spec hh rootB _) (streamObj_rounds spec hh rootB _))
    obtain ⟨rs, hrs, hsz, hP⟩ := array_mapM_ok_ex (fun r => roundJ r)
      ((((Array.range (S[i]).size).map fun k => roundObj spec hh k (S[i])[k]!).mapIdx
        ((fun _ _ => mapKey "coins" fun _ => Json.null) (spec.repStreams[i], S[i]).1.2.2)).mapIdx
          (g spec.repStreams[i].2.2))
      (fun k rd => ∀ hk : k < (S[i]).size, rd.bytes = some (((S[i])[k]'hk).bytes.getD ByteArray.empty))
      (fun k hk => by
        have hk' : k < (S[i]).size := by simpa using hk
        obtain ⟨rd, h1, h2⟩ := hg i hi k hk'
        refine ⟨rd, ?_, fun _ => h2⟩
        simp only [Array.getElem_mapIdx, Array.getElem_map, Array.getElem_range, getElem!_pos (S[i]) k hk']
        exact h1)
    refine ⟨rs, ?_, by simpa using hsz, fun k hk h' => hP k (by simpa using hk) hk⟩
    unfold streamJ
    rw [find_mapStream, find_mapStream, find_streamObj spec hh rootB S hS hnd i hi]
    dsimp only [Option.map]
    rw [ha]
    simp only [FlockComplete.ok_bind]
    exact hrs
  choose rsOf hrsOf hszOf hbOf using hst
  obtain ⟨streams, hsz, hsget⟩ : ∃ streams : Array (Array Round), ∃ hsz : streams.size = spec.repStreams.size,
      ∀ (i : ℕ) (h₁ : i < spec.repStreams.size), streams[i]'(by omega) = rsOf i h₁ :=
    ⟨Array.ofFn fun i : Fin spec.repStreams.size => rsOf i.1 i.2, by simp, fun i h₁ => by simp⟩
  refine ⟨⟨rootB, spec.tables.map fun _ => ByteArray.empty, pts, streams⟩, ?_, rfl, rfl, ?_⟩
  · unfold liveOfJ
    rw [fld_link]
    simp only [FlockComplete.ok_bind]
    rw [array_mapM_ok_of _ spec.tables rootB hrb.symm fun i h₁ h₂ => by
      simp only [fld_link_rootB, FlockComplete.ok_bind, str_rootB spec htn hrb i h₁, unhex_hexOf]]
    simp only [FlockComplete.ok_bind, fld_link_publics]
    rw [FlockComplete.array_mapM_ok _ (fun _ => ByteArray.empty) spec.tables fun t _ => by
      simp only [isNull_empty, ite_true]
      rfl]
    simp only [FlockComplete.ok_bind, arr_link_points]
    rw [array_mapM_ok_of _ _ pts (by simp) fun i h₁ h₂ => by
      simp only [Array.getElem_map, hexArr, Json.getArr?]
      exact f128s_map _]
    simp only [FlockComplete.ok_bind, arr_streams]
    rw [array_mapM_ok_of _ spec.repStreams streams hsz.symm fun i h₁ h₂ => by
      rw [hsget i h₁]
      exact hrsOf i h₁]
    rfl
  · rw [List.map_flatMap, List.map_flatMap]
    apply flatMap_congr_idx _ _ (by simp [hsz, hS])
    intro i h₁ h₂
    have hi : i < spec.repStreams.size := by simpa [hsz] using h₁
    simp only [Array.getElem_toList, hsget i hi]
    apply List.ext_getElem (by simp [hszOf i hi])
    intro k hk₁ hk₂
    simp only [List.getElem_map, Array.getElem_toList]
    exact hbOf i hi k (by simpa using hk₂) (by simpa using hk₁)

end Read

/-! ## At a decision of `decR` -/

theorem readPs_sum {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) : (repLens spec sts).sum = rss.length := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · exact hc.2
  · rw [if_neg hc] at h
    cases h

/-- **The round messages are as many as the rounds after the link**: each rep's are its rounds' framed bytes. -/
theorem readPs_length {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) : pf.2.length = rss.length := by
  have hsum := readPs_sum h
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    have hrl : (repLens spec sts).length = 2 * sts.size := by simp [repLens]
    have hk : ∀ (k : ℕ) (hk : k < rs.length),
        rs[k].2.length = ((cutLens (repLens spec sts) rss)[k]'(by rw [cutLens_length]; omega)).length := by
      intro k hk
      have hg := hget k (by rw [List.length_range]; omega) hk
      rw [List.getElem_range] at hg
      obtain ⟨T, hT, he⟩ := readRep_fits (p := rs[k].1) (ms := rs[k].2) hg
      rw [he, framesOf, List.length_map, hT.rounds, List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem (by rw [cutLens_length]; omega), Option.getD_some]
    dsimp only
    rw [List.length_flatten, ← cutLens_flatten (repLens spec sts) rss hsum.ge, List.length_flatten]
    congr 1
    apply List.ext_getElem (by simp [cutLens_length, hlen, hrl])
    intro k h₁ h₂
    simp only [List.getElem_map]
    exact hk k (by simpa using h₁)
  · rw [if_neg hc] at h
    cases h

/-- **The reader's streams carry the round messages in order**, one per round. -/
theorem recStreams_bytes (spec : SessionSpec) (sts : Array Stmt) (ms : List Claims.Msg) (cs : List Coin)
    (hl : ms.length = (repLens spec sts).sum) :
    ((recStreams spec sts ms cs).toList.flatMap fun s : Array Round => s.toList).map
        (fun r => some (r.bytes.getD ByteArray.empty)) = ms.map fun m => some (⟨m.toArray⟩ : ByteArray) := by
  have hrl : (repLens spec sts).length = 2 * sts.size := by simp [repLens]
  rw [List.map_flatMap, ← flatMap_cutLens_map _ (repLens spec sts) ms hl.le]
  apply flatMap_congr_idx _ _ (by simp [recStreams, cutLens_length, hrl])
  intro k h₁ h₂
  have hk : k < 2 * sts.size := by simpa [recStreams] using h₁
  have hk' : k < (cutLens (repLens spec sts) ms).length := by rw [cutLens_length]; omega
  simp only [recStreams, List.toList_toArray, List.getElem_map, List.getElem_range, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem hk', Option.getD_some]
  apply List.ext_getElem (by simp)
  intro j hj₁ hj₂
  simp [List.getElem_mapIdx]

/-- **Custody passes at the reader's record**: at every decision of `decR` at a statement with no public inputs, on as
many coins as messages, what `liveOfJ` reads of the substituted record (`jAt` at SHA-512) passes custody against the
decision's link and round messages. -/
theorem custody_decR {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) (hpi : d.s.read.publicInputs = none) (hl : as.length = cs.length) :
    ∃ lv, liveOfJ d.spec (jAt sha512 d.spec d.sts d.c₁ d.mids d.f) = .ok lv ∧
      custody d.spec lv d.link d.c₁ d.mids = .ok () := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, h1, -, -, h7, h8, h9, rfl⟩ := decR_some h
  dsimp only at hpi ⊢
  have hs : s.setup = .ok (spec, sts) := (setupOf_some h1).2
  have htn : spec.tables.toList.Nodup := KS.tables_nodup hs hpi
  have hnd := KS.repNames_nodup htn
  have hts : spec.tables.size = sts.size := by
    have h' := Ins.setup_read (zk := false) hs
    unfold Readable.Read.setup at h'
    rw [hpi] at h'
    exact (FlockComplete.ZkTables.setupTables_size h').2.2
  have hrb : link.1.toArray.size = spec.tables.size := by
    rw [List.size_toArray, (linkR_rootB h8).1, List.length_map, cutLens_length, hts]
    simp [level0Lens]
  have hpf : pf.2.length = rss.length := readPs_length h9
  have hrss : rss.length = as.tail.dropLast.length := (mapM_some_get h7).1
  have hsum : (repLens spec sts).sum = rss.length := readPs_sum h9
  have hcs : cs.tail.dropLast.length = pf.2.length := by
    rw [hpf, hrss]
    simp [hl]
  have hmids : (pf.2.zip cs.tail.dropLast).length = pf.2.length := by simp [hcs]
  have hSsize : (recStreams spec sts pf.2 cs.tail.dropLast).size = spec.repStreams.size := by
    simp [recStreams, Claims.Bridge.Custody.repStreams_size, hts]
  have hby := recStreams_bytes spec sts pf.2 cs.tail.dropLast (by omega)
  have htot : ((recStreams spec sts pf.2 cs.tail.dropLast).toList.map Array.size).sum ≤
      (pf.2.zip cs.tail.dropLast).length := by
    have e := congrArg List.length hby
    rw [List.length_map, List.length_map, List.length_flatMap] at e
    rw [hmids, ← e]
    simp
  have key : ∃ lv : Live, liveOfJ spec (jAt sha512 spec sts c₁ (pf.2.zip cs.tail.dropLast)
        ⟨fl.stmt, recJ spec sts link c₁ pf.2 cs.tail.dropLast pf.1, pf.1⟩) = .ok lv ∧
      lv.rootB = link.1.toArray ∧ lv.publics = spec.tables.map (fun _ => ByteArray.empty) ∧
      (lv.streams.toList.flatMap fun s : Array Round => s.toList).map Round.bytes =
        ((recStreams spec sts pf.2 cs.tail.dropLast).toList.flatMap fun s : Array Round => s.toList).map
          fun r => some (r.bytes.getD ByteArray.empty) := by
    refine liveOfJ_rec spec _ _ _ _ _ _ _ htn hrb hnd hSsize ?_
    intro i hi k hk
    have hoff := streamOffset_rec spec (Claims.hOf sha512) link.1.toArray (pointsOf spec c₁)
      (recStreams spec sts pf.2 cs.tail.dropLast) (filesAt (tabsOf sts) (pointsOf spec c₁) pf.1)
      ((filesAt (tabsOf sts) (pointsOf spec c₁) pf.1).map (FlockComplete.dgOf spec (Claims.hOf sha512)))
      hSsize hnd i hi
    have hlt : streamOffset spec (eraseCoins (shaSubJ spec
        ((filesAt (tabsOf sts) (pointsOf spec c₁) pf.1).map (FlockComplete.dgOf spec (Claims.hOf sha512)))
        (recordObj spec (Claims.hOf sha512) link.1.toArray (pointsOf spec c₁)
          (recStreams spec sts pf.2 cs.tail.dropLast) (filesAt (tabsOf sts) (pointsOf spec c₁) pf.1))))
        spec.repStreams[i].2.2 + k < (pf.2.zip cs.tail.dropLast).length := by
      rw [hoff]
      have hi' : i < ((recStreams spec sts pf.2 cs.tail.dropLast).toList.map Array.size).length := by
        simp [hSsize, hi]
      have := sum_take_add_getElem_le _ i hi'
      simp only [List.getElem_map, Array.getElem_toList] at this
      omega
    obtain ⟨q, hq⟩ : ∃ q, (pf.2.zip cs.tail.dropLast)[streamOffset spec (eraseCoins (shaSubJ spec
        ((filesAt (tabsOf sts) (pointsOf spec c₁) pf.1).map (FlockComplete.dgOf spec (Claims.hOf sha512)))
        (recordObj spec (Claims.hOf sha512) link.1.toArray (pointsOf spec c₁)
          (recStreams spec sts pf.2 cs.tail.dropLast) (filesAt (tabsOf sts) (pointsOf spec c₁) pf.1))))
        spec.repStreams[i].2.2 + k]? = some q := ⟨_, List.getElem?_eq_getElem hlt⟩
    obtain ⟨m, c⟩ := q
    beta_reduce
    rw [hq]
    exact ⟨_, roundJ_sub spec _ k _ _, rfl⟩
  obtain ⟨lv, hlv, hrB, hpubs, hbytes⟩ := key
  refine ⟨lv, hlv, ?_⟩
  have hbytes' := hbytes.trans hby
  have hb1 : (lv.rootB.toList == link.1) = true := by
    rw [hrB]
    exact list_bytes_beq_self _
  have hb2 : (lv.publics.toList == link.2) = true := by
    rw [hpubs, linkR_publics h8, Array.toList_map, List.map_const', List.map_const', Array.length_toList,
      Array.length_toList, hts]
    exact list_bytes_beq_self _
  have hb3 : ((lv.streams.toList.flatMap fun s : Array Round => s.toList).length ==
      (pf.2.zip cs.tail.dropLast).length) = true := by
    have e := congrArg List.length hbytes'
    rw [List.length_map, List.length_map] at e
    rw [e, hmids]
    exact beq_self_eq_true _
  have hb4 : (((lv.streams.toList.flatMap fun s : Array Round => s.toList).zip (pf.2.zip cs.tail.dropLast)).all
      fun q => q.1.bytes == some ⟨q.2.1.toArray⟩) = true := by
    rw [List.all_eq_true]
    intro q hq
    obtain ⟨t, ht, rfl⟩ := List.mem_iff_getElem.1 hq
    have ht₁ : t < (lv.streams.toList.flatMap fun s : Array Round => s.toList).length := by
      simp only [List.length_zip] at ht
      omega
    have ht₂ : t < pf.2.length := by
      simp only [List.length_zip] at ht
      omega
    have e := getElem_of_map_eq hbytes' t ht₁ (by rw [List.length_map]; exact ht₂)
    rw [List.getElem_map] at e
    rw [List.getElem_zip, List.getElem_zip]
    dsimp only
    rw [e]
    exact bytes_beq_self _
  rw [custody_coinSubJ spec link c₁ (pf.2.zip cs.tail.dropLast) _ hnd hlv, hb1, hb2, hb3, hb4]
  rfl

end FlockDeploy.RestCustody
