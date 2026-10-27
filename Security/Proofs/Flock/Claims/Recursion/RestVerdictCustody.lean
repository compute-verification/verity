import Proofs.Flock.Claims.Recursion.RestVerdictBases
import Proofs.Flock.Claims.Complete.ZkTables.Decode
import Proofs.Flock.Claims.Complete.ZkTables.Scope

/-!
# Custody at the reader's record (`CustodyRecJ`)

At every decision of `nextR` whose statement passes its scope check, `liveOfJ` reads the value the verdict reads
(`jAt`: the reader's record `recJ`, its proof digests substituted, its coins the live coins) and custody passes there.
Nothing is assumed of the prover: the reader builds the record from the link and the round messages, so every field
custody compares is the link's or a round message's by construction.

* **The record's reads** (`liveOfJ_sub`): at any value whose link is the honest record's link object and whose streams
  are the honest record's stream objects, at tables named apart, `liveOfJ` reads the `root_B` and no public words, and
  each rep stream's rounds with the bytes they retain, every round's coin filled from the live coins (`fillG`) as long
  as there are as many live coins as rounds.
* **The reader's streams** (`recStreams_bytes`): the rounds `recStreams` records retain, in order, the round messages.
* **The reading's lengths** (`readPs_len`): the round messages are as many as the opened rounds after the link.
* **`custodyRecJ`**: custody's comparisons, the points and coins aside (`custody_coinSubJ`), are then identities.
-/

namespace RecDeploy

open Flock
open Lean (Json)
open Claims (Msg)
open FlockDeploy (Coin FinalP DecR decR jAt custody liveOfJ Live coinSubJ fillCoins eraseCoins mapRounds setPoints
  mapKey mapStream streamJ roundJ hexArr nameOf nOf streamOffset roundsOf shaSubJ pointsOf recStreams repLens cutLens
  readPs fld_mapKey_eq fld_mapKey_ne arr_mapKey_mapArr arr_mapKey_mapArrIdx arr_mapKey_ne nat_mapKey_ne
  getObjVal?_mapKey_ne getObjValAs_mapKey_ne find_mapStream roundsOf_of takeWhile_ne_nodup custody_coinSubJ)
open FlockComplete.JsonRT (recordObj recordList recordList_distinct linkObj linkList linkList_distinct rootBList
  streamObj streamList streamList_distinct roundObj roundList roundList_distinct getObjVal?_mkObj f128s_map
  array_mapM_ok_of unhex_hexOf bytes_beq_self)

/-! ## Reading objects -/

theorem fld_of_gv {j : Json} {k : String} {v : Json} (h : j.getObjVal? k = .ok v) : Record.fld j k = .ok v := by
  simp only [Record.fld, h]

theorem fld_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) {k : String} {v : Json}
    (hm : (k, v) ∈ l) : Record.fld (Json.mkObj l) k = .ok v :=
  fld_of_gv (getObjVal?_mkObj hd hm)

theorem isNull_mkObj_nil (k : String) : Record.isNull (Json.mkObj []) k = true := by
  first
  | rfl
  | simp only [Record.isNull, Json.mkObj, Json.getObjVal?, FlockComplete.JsonRT.get?_mkObj_none (l := []) (k := k) rfl]

theorem bytesList_beq_self (l : List ByteArray) : (l == l) = true := by
  induction l with
  | nil => rfl
  | cons b l ih => simp only [List.cons_beq_cons, bytes_beq_self, ih, Bool.and_self]

/-- **`mapM` returns where every element does**, its results related to their indices as each element's is. -/
theorem array_mapM_ok_ex {α β : Type} (f : α → Flock.V β) (a : Array α) (P : ℕ → β → Prop)
    (h : ∀ (i : ℕ) (hi : i < a.size), ∃ b, f a[i] = .ok b ∧ P i b) :
    ∃ ys : Array β, a.mapM f = .ok ys ∧ ys.size = a.size ∧ ∀ (i : ℕ) (hi : i < ys.size), P i ys[i] := by
  refine ⟨Array.ofFn fun i : Fin a.size => Classical.choose (h i i.2), ?_, by simp, fun i hi => ?_⟩
  · refine array_mapM_ok_of f a _ (by simp) fun i h₁ h₂ => ?_
    rw [Array.getElem_ofFn]
    exact (Classical.choose_spec (h i h₁)).1
  · rw [Array.getElem_ofFn]
    exact (Classical.choose_spec (h i (by simpa using hi))).2

theorem mapM_bind_ok_ex {α β γ : Type} (f : α → Flock.V β) (a : Array α) (k : Array β → γ) (P : ℕ → β → Prop)
    (Q : γ → Prop) (h : ∀ (i : ℕ) (hi : i < a.size), ∃ b, f a[i] = .ok b ∧ P i b)
    (hQ : ∀ ys : Array β, ys.size = a.size → (∀ (i : ℕ) (hi : i < ys.size), P i ys[i]) → Q (k ys)) :
    ∃ lv, (a.mapM f >>= fun ys => pure (k ys)) = .ok lv ∧ Q lv := by
  obtain ⟨ys, hys, hsz, hP⟩ := array_mapM_ok_ex f a P h
  exact ⟨k ys, by rw [hys]; rfl, hQ ys hsz hP⟩

theorem sum_take_lt {α : Type} (g : α → ℕ) :
    ∀ (L : List α) (i : ℕ) (hi : i < L.length) (k : ℕ), k < g L[i] → ((L.take i).map g).sum + k < (L.map g).sum
  | [], _, hi, _, _ => absurd hi (by simp)
  | a :: L, 0, _, k, hk => by
    simp only [List.take_zero, List.map_nil, List.sum_nil, List.map_cons, List.sum_cons, List.getElem_cons_zero] at hk ⊢
    omega
  | a :: L, i + 1, hi, k, hk => by
    have := sum_take_lt g L i (by simpa using hi) k (by simpa using hk)
    simp only [List.take_succ_cons, List.map_cons, List.sum_cons]
    omega

/-! ## The substituted value's maps -/

/-- The erasure's round map. -/
def eraseG : String → ℕ → Json → Json := fun _ _ => mapKey "coins" fun _ => .null

/-- The fill's round map, at the erased value `j`. -/
def fillG (spec : SessionSpec) (mids : List (Msg × Coin)) (j : Json) : String → ℕ → Json → Json :=
  fun name k r => match mids[streamOffset spec j name + k]? with
    | some (_, c) => mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r
    | none => r

theorem eraseCoins_eq (j : Json) : eraseCoins j = mapRounds eraseG (setPoints .null j) := rfl

theorem fillCoins_eq (spec : SessionSpec) (c₁ : Coin) (mids : List (Msg × Coin)) (j : Json) :
    fillCoins spec c₁ mids j = mapRounds (fillG spec mids j) (setPoints (.arr ((pointsOf spec c₁).map hexArr)) j) :=
  rfl

theorem fld_link_rounds (g : String → ℕ → Json → Json) (v j : Json) :
    Record.fld (mapRounds g (setPoints v j)) "link" = (mapKey "points" fun _ => v) <$> Record.fld j "link" := by
  unfold mapRounds setPoints
  rw [fld_mapKey_ne _ _ (by decide : "link" ≠ "streams"), fld_mapKey_eq]

theorem arr_streams_rounds (g : String → ℕ → Json → Json) (v j : Json) :
    Record.arr (mapRounds g (setPoints v j)) "streams" = (·.map (mapStream g)) <$> Record.arr j "streams" := by
  unfold mapRounds setPoints
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link")]

theorem arr_mapStream (g : String → ℕ → Json → Json) (s : Json) :
    Record.arr (mapStream g s) "rounds" = (·.mapIdx (g (nameOf s))) <$> Record.arr s "rounds" := by
  unfold mapStream
  exact arr_mapKey_mapArrIdx _ _ _

theorem nameOf_mapStream (g : String → ℕ → Json → Json) (s : Json) : nameOf (mapStream g s) = nameOf s := by
  unfold nameOf mapStream
  rw [getObjValAs_mapKey_ne _ _ (by decide : "stream" ≠ "rounds")]

/-! ## The honest record's objects -/

theorem streamObj_name (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : (streamObj spec h rb q).getObjValAs? String "stream" = .ok q.1.2.2 := by
  unfold streamObj
  simp only [Json.getObjValAs?, Json.getObjValD, getObjVal?_mkObj (streamList_distinct spec h rb q)
    (show ("stream", Json.str q.1.2.2) ∈ streamList spec h rb q by simp [streamList])] <;> rfl

theorem streamObj_rounds (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) :
    Record.arr (streamObj spec h rb q) "rounds" = .ok ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!) := by
  unfold streamObj
  simp only [Record.arr, fld_mkObj (streamList_distinct spec h rb q)
    (show ("rounds", Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!)) ∈
      streamList spec h rb q by simp [streamList])] <;> rfl

theorem str_rootB_mk {spec : SessionSpec} (hn : spec.tables.toList.Nodup) {rb : Array ByteArray}
    (hrbs : rb.size = spec.tables.size) (i : ℕ) (h₁ : i < spec.tables.size) (h₂ : i < rb.size) :
    Record.str (Json.mkObj (rootBList spec id rb)) spec.tables[i] = .ok (hexOf rb[i]) := by
  have hm : (spec.tables[i], FlockComplete.hexJ (id rb[i])) ∈ rootBList spec id rb :=
    List.mem_map.2 ⟨(spec.tables[i], rb[i]), Array.mem_toList_iff.2 (Array.mem_iff_getElem.2
      ⟨i, by simp; omega, by simp [Array.getElem_zip]⟩), rfl⟩
  simp only [Record.str, fld_mkObj (FlockComplete.ZkTables.rootBList_distinct hn id hrbs) hm] <;> rfl

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

/-! ## A recorded round, its coins filled -/

theorem roundJ_fill (spec : SessionSpec) (h : ByteArray → ByteArray) (k : ℕ) (r : Round) (c : Coin) :
    roundJ (mapKey "coins" (fun _ => hexArr (c.vals (nOf (mapKey "coins" (fun _ => .null) (roundObj spec h k r)))))
        (mapKey "coins" (fun _ => .null) (roundObj spec h k r))) =
      .ok ⟨r.n, ByteArray.empty, c.vals r.n, some (r.bytes.getD ByteArray.empty)⟩ := by
  have gRd : ∀ {key : String} {v : Json}, (key, v) ∈ roundList spec h k r →
      (roundObj spec h k r).getObjVal? key = .ok v :=
    fun hm => getObjVal?_mkObj (roundList_distinct ..) hm
  have hn0 : Record.nat (roundObj spec h k r) "n" = .ok r.n := by
    simp only [Record.nat, fld_of_gv (gRd (key := "n") (v := Lean.toJson r.n) (by simp [roundList]))] <;> rfl
  have hnO : nOf (mapKey "coins" (fun _ => .null) (roundObj spec h k r)) = r.n := by
    simp only [nOf, nat_mapKey_ne _ _ (by decide : "n" ≠ "coins"), hn0]
  rw [hnO]
  have hmsg : (roundObj spec h k r).getObjVal? "msg" = .ok (Json.str (hexOf (r.bytes.getD ByteArray.empty))) :=
    gRd (by simp [roundList, FlockComplete.hexJ])
  have hgv : ∀ (G G' : Json → Json),
      (mapKey "coins" G (mapKey "coins" G' (roundObj spec h k r))).getObjVal? "msg" =
        .ok (Json.str (hexOf (r.bytes.getD ByteArray.empty))) := by
    intro G G'
    rw [getObjVal?_mapKey_ne _ _ (by decide : "msg" ≠ "coins"), getObjVal?_mapKey_ne _ _ (by decide : "msg" ≠ "coins"),
      hmsg]
  have e1 : Record.nat (mapKey "coins" (fun _ => hexArr (c.vals r.n)) (mapKey "coins" (fun _ => .null)
      (roundObj spec h k r))) "n" = .ok r.n := by
    rw [nat_mapKey_ne _ _ (by decide : "n" ≠ "coins"), nat_mapKey_ne _ _ (by decide : "n" ≠ "coins"), hn0]
  have hc0 : Record.fld (roundObj spec h k r) "coins" = .ok (Json.arr (r.coins.map FlockComplete.f128J)) :=
    fld_of_gv (gRd (by simp [roundList]))
  have e2 : Record.arr (mapKey "coins" (fun _ => hexArr (c.vals r.n)) (mapKey "coins" (fun _ => .null)
      (roundObj spec h k r))) "coins" = .ok ((c.vals r.n).map FlockComplete.f128J) := by
    unfold Record.arr
    rw [fld_mapKey_eq, fld_mapKey_eq, hc0] <;> rfl
  have e3 : Record.isNull (mapKey "coins" (fun _ => hexArr (c.vals r.n)) (mapKey "coins" (fun _ => .null)
      (roundObj spec h k r))) "msg" = false := by
    simp only [Record.isNull, hgv]
  have e4 : Record.str (mapKey "coins" (fun _ => hexArr (c.vals r.n)) (mapKey "coins" (fun _ => .null)
      (roundObj spec h k r))) "msg" = .ok (hexOf (r.bytes.getD ByteArray.empty)) := by
    simp only [Record.str, fld_of_gv (hgv _ _)] <;> rfl
  unfold roundJ
  simp only [e1, e2, e3, e4, FlockComplete.ok_bind, f128s_map, unhex_hexOf, Bool.false_eq_true, ite_false] <;> rfl

/-! ## The record's reads at the substituted value -/

set_option maxHeartbeats 1000000 in
/-- **`liveOfJ` at the substituted value of a record with the honest link and streams**: its `root_B`, no public
words, and each rep stream's rounds retaining their bytes, where the tables are named apart, there is one `root_B` per
table and one stream per rep stream, and there are as many live coins as recorded rounds. -/
theorem liveOfJ_sub (spec : SessionSpec) (c₁ : Coin) (mids : List (Msg × Coin)) (J : Json)
    (h : ByteArray → ByteArray) (rb : Array ByteArray) (pts : Array (Array F128)) (streams : Array (Array Round))
    (htn : spec.tables.toList.Nodup) (hrb : rb.size = spec.tables.size) (hss : streams.size = spec.repStreams.size)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hJl : Record.fld J "link" = .ok (linkObj spec h rb pts))
    (hJs : Record.arr J "streams" = .ok ((spec.repStreams.zip streams).map (streamObj spec h rb)))
    (hmid : (streams.toList.flatMap fun s : Array Round => s.toList).length ≤ mids.length) :
    ∃ lv : Live, liveOfJ spec (coinSubJ spec c₁ mids J) = .ok lv ∧ lv.rootB = rb ∧
      lv.publics = spec.tables.map (fun _ => ByteArray.empty) ∧
      lv.streams.toList.map (fun s => s.toList.map Round.bytes) =
        streams.toList.map (fun s => s.toList.map fun r => some (r.bytes.getD ByteArray.empty)) := by
  have hJ2l : Record.fld (eraseCoins J) "link" = .ok (mapKey "points" (fun _ => .null) (linkObj spec h rb pts)) := by
    rw [eraseCoins_eq, fld_link_rounds, hJl] <;> rfl
  have hJ2s : Record.arr (eraseCoins J) "streams" =
      .ok (((spec.repStreams.zip streams).map (streamObj spec h rb)).map (mapStream eraseG)) := by
    rw [eraseCoins_eq, arr_streams_rounds, hJs] <;> rfl
  have hlink : Record.fld (coinSubJ spec c₁ mids J) "link" = .ok (mapKey "points"
      (fun _ => .arr ((pointsOf spec c₁).map hexArr)) (mapKey "points" (fun _ => .null) (linkObj spec h rb pts))) := by
    unfold coinSubJ
    rw [fillCoins_eq, fld_link_rounds, hJ2l] <;> rfl
  have hstr : Record.arr (coinSubJ spec c₁ mids J) "streams" =
      .ok ((((spec.repStreams.zip streams).map (streamObj spec h rb)).map (mapStream eraseG)).map
        (mapStream (fillG spec mids (eraseCoins J)))) := by
    unfold coinSubJ
    rw [fillCoins_eq, arr_streams_rounds, hJ2s] <;> rfl
  have hl_rootb : ∀ (P : Json), Record.fld (mapKey "points" (fun _ => P) (mapKey "points" (fun _ => .null)
      (linkObj spec h rb pts))) "root_b" = .ok (Json.mkObj (rootBList spec id rb)) := by
    intro P
    rw [fld_mapKey_ne _ _ (by decide : "root_b" ≠ "points"), fld_mapKey_ne _ _ (by decide : "root_b" ≠ "points")]
    unfold linkObj
    exact fld_mkObj (linkList_distinct ..) (by simp [linkList])
  have hl_pub : ∀ (P : Json), Record.fld (mapKey "points" (fun _ => P) (mapKey "points" (fun _ => .null)
      (linkObj spec h rb pts))) "publics" = .ok (Json.mkObj []) := by
    intro P
    rw [fld_mapKey_ne _ _ (by decide : "publics" ≠ "points"), fld_mapKey_ne _ _ (by decide : "publics" ≠ "points")]
    unfold linkObj
    exact fld_mkObj (linkList_distinct ..) (by simp [linkList])
  have hl_pts : Record.arr (mapKey "points" (fun _ => .arr ((pointsOf spec c₁).map hexArr))
      (mapKey "points" (fun _ => .null) (linkObj spec h rb pts))) "points" = .ok ((pointsOf spec c₁).map hexArr) := by
    have hp0 : Record.fld (linkObj spec h rb pts) "points" =
        .ok (Json.arr (pts.map fun p => Json.arr (p.map FlockComplete.f128J))) := by
      unfold linkObj
      exact fld_mkObj (linkList_distinct ..) (by simp [linkList])
    unfold Record.arr
    rw [fld_mapKey_eq, fld_mapKey_eq, hp0] <;> rfl
  -- the offsets at which each rep stream's rounds take their coins
  have hrounds : ∀ (j : ℕ) (hj : j < spec.repStreams.size),
      FlockDeploy.roundsOf (eraseCoins J) spec.repStreams[j].2.2 = (streams[j]'(hss ▸ hj)).size := by
    intro j hj
    have hf : (((spec.repStreams.zip streams).map (streamObj spec h rb)).map (mapStream eraseG)).find?
        (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[j].2.2) =
        some (mapStream eraseG (streamObj spec h rb (spec.repStreams[j], streams[j]'(hss ▸ hj)))) := by
      rw [find_mapStream, find_streamObj hss hnd j hj]
      rfl
    have ha : Record.arr (mapStream eraseG (streamObj spec h rb (spec.repStreams[j], streams[j]'(hss ▸ hj))))
        "rounds" = .ok (((Array.range (streams[j]'(hss ▸ hj)).size).map fun k =>
          roundObj spec h k (streams[j]'(hss ▸ hj))[k]!).mapIdx (eraseG (nameOf (streamObj spec h rb
            (spec.repStreams[j], streams[j]'(hss ▸ hj)))))) := by
      rw [arr_mapStream, streamObj_rounds] <;> rfl
    rw [roundsOf_of hJ2s hf ha]
    simp
  have hoff : ∀ (i : ℕ) (hi : i < spec.repStreams.size),
      streamOffset spec (eraseCoins J) spec.repStreams[i].2.2 =
        ((streams.toList.take i).map Array.size).sum := by
    intro i hi
    unfold streamOffset
    have hni : spec.repStreams[i].2.2 = (spec.repStreams.toList.map (·.2.2))[i]'(by simpa using hi) := by simp
    have e : (spec.repStreams.toList.map (·.2.2)).map (FlockDeploy.roundsOf (eraseCoins J)) =
        streams.toList.map Array.size := by
      apply List.ext_getElem (by simp [hss])
      intro j h₁ h₂
      have hj : j < spec.repStreams.size := by simpa using h₁
      simp only [List.getElem_map, Array.getElem_toList]
      exact hrounds j hj
    rw [hni, takeWhile_ne_nodup _ hnd i (by simpa using hi), List.map_take, List.map_take, e]
  have htot : ∀ (i : ℕ) (hi : i < streams.size) (k : ℕ), k < streams[i].size →
      ((streams.toList.take i).map Array.size).sum + k < mids.length := by
    intro i hi k hk
    have hlt := sum_take_lt Array.size streams.toList i (by simpa using hi) k (by simpa using hk)
    have hfl : (streams.toList.flatMap fun s : Array Round => s.toList).length = (streams.toList.map Array.size).sum := by
      simp [List.length_flatMap]
    omega
  unfold liveOfJ
  rw [hlink]
  simp only [FlockComplete.ok_bind]
  rw [array_mapM_ok_of _ spec.tables rb hrb.symm fun i h₁ h₂ => by
    simp only [hl_rootb, FlockComplete.ok_bind, str_rootB_mk htn hrb i h₁ h₂, unhex_hexOf] <;> rfl]
  simp only [FlockComplete.ok_bind, hl_pub]
  rw [FlockComplete.array_mapM_ok _ (fun _ => ByteArray.empty) spec.tables fun t _ => by
    simp only [isNull_mkObj_nil, ite_true] <;> rfl]
  simp only [FlockComplete.ok_bind, hl_pts]
  rw [array_mapM_ok_of _ ((pointsOf spec c₁).map hexArr) (pointsOf spec c₁) (by simp) fun i h₁ h₂ => by
    simp only [Array.getElem_map, hexArr, Json.getArr?]
    exact f128s_map _]
  simp only [FlockComplete.ok_bind, hstr]
  refine mapM_bind_ok_ex _ _ _
    (fun i (y : Array Round) => y.toList.map Round.bytes =
      (streams[i]!).toList.map fun r => some (r.bytes.getD ByteArray.empty))
    (fun lv : Live => lv.rootB = rb ∧ lv.publics = spec.tables.map (fun _ => ByteArray.empty) ∧
      lv.streams.toList.map (fun s => s.toList.map Round.bytes) =
        streams.toList.map (fun s => s.toList.map fun r => some (r.bytes.getD ByteArray.empty))) ?_ ?_
  · intro i hi
    have hi' : i < streams.size := by omega
    obtain ⟨⟨a, b, name⟩, hx⟩ : ∃ x, spec.repStreams[i] = x := ⟨_, rfl⟩
    have hname : spec.repStreams[i].2.2 = name := by rw [hx]
    rw [hx]
    dsimp only
    have hf := find_streamObj (h := h) (rb := rb) hss hnd i hi
    have hof := hoff i hi
    rw [hname] at hf hof
    unfold streamJ
    rw [find_mapStream, find_mapStream, hf]
    simp only [Option.map_some]
    have hnm : nameOf (streamObj spec h rb (spec.repStreams[i], streams[i])) = name := by
      unfold nameOf
      rw [streamObj_name]
      exact hname
    have e2 : Record.arr (mapStream (fillG spec mids (eraseCoins J)) (mapStream eraseG
        (streamObj spec h rb (spec.repStreams[i], streams[i])))) "rounds" =
        .ok ((((Array.range streams[i].size).map fun k => roundObj spec h k streams[i][k]!).mapIdx
          (eraseG name)).mapIdx (fillG spec mids (eraseCoins J) name)) := by
      rw [arr_mapStream, arr_mapStream, streamObj_rounds, nameOf_mapStream, hnm] <;> rfl
    rw [e2, FlockComplete.ok_bind]
    obtain ⟨ys, hys, hsz, hP⟩ := array_mapM_ok_ex (fun r => roundJ r)
      ((((Array.range streams[i].size).map fun k => roundObj spec h k streams[i][k]!).mapIdx
        (eraseG name)).mapIdx (fillG spec mids (eraseCoins J) name))
      (fun k (y : Round) => y.bytes = some ((streams[i][k]!).bytes.getD ByteArray.empty)) (fun k hk => by
        have hk' : k < streams[i].size := by simpa using hk
        have hlt := htot i hi' k hk'
        refine ⟨⟨(streams[i][k]!).n, ByteArray.empty,
          (mids[((streams.toList.take i).map Array.size).sum + k]'hlt).2.vals (streams[i][k]!).n,
          some ((streams[i][k]!).bytes.getD ByteArray.empty)⟩, ?_, rfl⟩
        simp only [Array.getElem_mapIdx, Array.getElem_map, Array.getElem_range]
        unfold fillG
        rw [hof, List.getElem?_eq_getElem hlt]
        exact roundJ_fill spec h k _ _)
    refine ⟨ys, hys, ?_⟩
    rw [getElem!_pos streams i hi']
    apply List.ext_getElem (by simp [hsz])
    intro k h₁ h₂
    have hk : k < ys.size := by simpa using h₁
    have hk' : k < streams[i].size := by simpa [hsz] using hk
    simp only [List.getElem_map, Array.getElem_toList]
    rw [hP k hk, getElem!_pos _ k hk']
  · intro ys hsz hP
    refine ⟨rfl, rfl, ?_⟩
    apply List.ext_getElem (by simp [hsz, hss])
    intro j h₁ h₂
    have hj : j < ys.size := by simpa using h₁
    have hj' : j < streams.size := by simpa using h₂
    simp only [List.getElem_map, Array.getElem_toList]
    rw [hP j hj, getElem!_pos _ j hj']

/-! ## The reader's record -/

theorem recJ_link (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray) (pts : Array (Array F128))
    (streams : Array (Array Round)) (proofs ds : Array ByteArray) :
    Record.fld (shaSubJ spec ds (recordObj spec h rb pts streams proofs)) "link" = .ok (linkObj spec h rb pts) := by
  unfold shaSubJ
  rw [fld_mapKey_ne _ _ (FlockDeploy.proofKey_ne_link spec).symm]
  unfold recordObj
  exact fld_mkObj (recordList_distinct ..) (by simp [recordList])

theorem recJ_streams (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (pts : Array (Array F128)) (streams : Array (Array Round)) (proofs ds : Array ByteArray) :
    Record.arr (shaSubJ spec ds (recordObj spec h rb pts streams proofs)) "streams" =
      .ok ((spec.repStreams.zip streams).map (streamObj spec h rb)) := by
  unfold shaSubJ
  rw [arr_mapKey_ne _ _ (FlockDeploy.proofKey_ne_streams spec).symm]
  unfold recordObj
  simp only [Record.arr, fld_mkObj (recordList_distinct ..)
    (show ("streams", Json.arr ((spec.repStreams.zip streams).map (streamObj spec h rb))) ∈
      recordList spec h rb pts streams proofs by simp [recordList])] <;> rfl

theorem cutLens_length {α : Type} : ∀ (ns : List ℕ) (l : List α), (cutLens ns l).length = ns.length
  | [], _ => rfl
  | n :: ns, l => by simp only [FlockDeploy.cutLens, List.length_cons, cutLens_length ns]

theorem range_map_getD {α : Type} (L : List (List α)) : (List.range L.length).map (fun k => L.getD k []) = L := by
  apply List.ext_getElem (by simp)
  intro k h₁ h₂
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h₂]

/-- **The reader's rounds retain the round messages**, in order, where they fit the reps' round counts. -/
theorem recStreams_bytes (spec : SessionSpec) (sts : Array Stmt) (M : List Msg) (cs : List Coin)
    (hM : M.length ≤ (repLens spec sts).sum) :
    ((recStreams spec sts M cs).toList.map fun s => s.toList.map fun r => some (r.bytes.getD ByteArray.empty)).flatten =
      M.map fun m => some (⟨m.toArray⟩ : ByteArray) := by
  have hL : (repLens spec sts).length = 2 * sts.size := by simp [FlockDeploy.repLens]
  have hB : (List.range (2 * sts.size)).map (fun k => (cutLens (repLens spec sts) M).getD k []) =
      cutLens (repLens spec sts) M := by
    rw [← hL, ← cutLens_length (repLens spec sts) M]
    exact range_map_getD _
  have e : (recStreams spec sts M cs).toList.map (fun s => s.toList.map fun r => some (r.bytes.getD ByteArray.empty)) =
      ((List.range (2 * sts.size)).map fun k => (cutLens (repLens spec sts) M).getD k []).map
        (List.map fun m => some (⟨m.toArray⟩ : ByteArray)) := by
    unfold recStreams
    simp only [List.toList_toArray, List.map_map]
    apply List.map_congr_left
    intro k _
    simp only [Function.comp_apply, List.toList_toArray]
    apply List.ext_getElem (by simp)
    intro j h₁ h₂
    simp [List.getElem_mapIdx]
  rw [e, ← List.map_flatten, hB, FlockDeploy.cutLens_flatten _ _ hM]

/-- **The reading's lengths**: the reps' round counts sum to the opened rounds after the link, and the round
messages are as many. -/
theorem readPs_len {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) :
    (repLens spec sts).sum = rss.length ∧ pf.2.length = rss.length := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := FlockDeploy.mapM_some_get hrs
    rw [List.length_range] at hlen
    refine ⟨hc.2, ?_⟩
    have hL : (repLens spec sts).length = 2 * sts.size := by simp [FlockDeploy.repLens]
    have hcl : (cutLens (repLens spec sts) rss).length = rs.length := by rw [cutLens_length, hL, hlen]
    dsimp only
    rw [List.length_flatten, ← FlockDeploy.cutLens_flatten (repLens spec sts) rss (by omega), List.length_flatten]
    congr 1
    apply List.ext_getElem (by simp [hcl])
    intro k h₁ h₂
    simp only [List.getElem_map]
    have hk : k < rs.length := by simpa using h₁
    have hg := hget k (by rw [List.length_range]; omega) hk
    rw [List.getElem_range] at hg
    obtain ⟨T, hT, hms⟩ := FlockDeploy.readRep_fits hg
    rw [show rs[k].2 = _ from hms]
    simp only [FlockDeploy.framesOf, List.length_map]
    rw [hT.rounds, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by simpa using h₂), Option.getD_some]
  · rw [if_neg hc] at h
    cases h

/-! ## Custody -/

/-- **Custody holds at the reader's record** at every decision whose statement passes its scope check. -/
theorem custodyRecJ (rd : Msg → Option FinalP) : CustodyRecJ rd := by
  intro x as cs d hlen hd hq
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hx, -, -, hrss, hlink, hpf, rfl⟩ := FlockDeploy.decR_some hd
  dsimp only at hq hlink hpf ⊢
  have hs := (FlockDeploy.setupOf_some hx).2
  have hpi := Claims.Bridge.Custody.publicInputs_none_of_check hq
  have htn := FlockDeploy.KS.tables_nodup hs hpi
  have hnd := FlockDeploy.KS.repNames_nodup htn
  have hts : spec.tables.size = sts.size := by
    have h := FlockDeploy.Ins.setup_read hs
    unfold Readable.Read.setup at h
    rw [hpi] at h
    exact (FlockComplete.ZkTables.setupTables_size h).2.2
  obtain ⟨hl1, -⟩ := FlockDeploy.linkR_rootB hlink
  have hpub := FlockDeploy.linkR_publics hlink
  obtain ⟨hsum, hpf2⟩ := readPs_len hpf
  have hrl : rss.length = as.tail.dropLast.length := (FlockDeploy.mapM_some_get hrss).1
  have hMc : pf.2.length ≤ cs.tail.dropLast.length := by
    simp only [List.length_dropLast, List.length_tail] at hrl ⊢
    omega
  have hrb : link.1.toArray.size = spec.tables.size := by
    rw [List.size_toArray, hl1, List.length_map, cutLens_length, hts]
    simp [FlockDeploy.level0Lens]
  have hss : (recStreams spec sts pf.2 cs.tail.dropLast).size = spec.repStreams.size := by
    simp [recStreams, FlockDeploy.repStreams_size, hts]
  have hbytes := recStreams_bytes spec sts pf.2 cs.tail.dropLast (hpf2.trans hsum.symm).le
  have hfl : ((recStreams spec sts pf.2 cs.tail.dropLast).toList.flatMap fun s : Array Round => s.toList).length =
      pf.2.length := by
    have := congrArg List.length hbytes
    simpa [List.length_flatten, List.length_flatMap, Function.comp_def] using this
  obtain ⟨lv, hL, hrbE, hpubE, hstE⟩ := liveOfJ_sub spec c₁ (pf.2.zip cs.tail.dropLast) _
    (Claims.hOf FlockDeploy.sha512) link.1.toArray (pointsOf spec c₁) (recStreams spec sts pf.2 cs.tail.dropLast)
    htn hrb hss hnd
    (recJ_link spec _ _ _ _ (FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (pointsOf spec c₁) pf.1)
      ((FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (pointsOf spec c₁) pf.1).map
        (FlockComplete.dgOf spec (Claims.hOf FlockDeploy.sha512))))
    (recJ_streams spec _ _ _ _ (FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (pointsOf spec c₁) pf.1)
      ((FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (pointsOf spec c₁) pf.1).map
        (FlockComplete.dgOf spec (Claims.hOf FlockDeploy.sha512))))
    (by rw [hfl, List.length_zip]; omega)
  refine ⟨lv, hL, ?_⟩
  rw [custody_coinSubJ spec link c₁ (pf.2.zip cs.tail.dropLast) _ hnd hL]
  have hFL : (lv.streams.toList.flatMap fun s : Array Round => s.toList).map Round.bytes =
      pf.2.map fun m => some (⟨m.toArray⟩ : ByteArray) := by
    rw [List.map_flatMap, List.flatMap_def, hstE]
    exact hbytes
  have hlenFL : (lv.streams.toList.flatMap fun s : Array Round => s.toList).length = pf.2.length := by
    have := congrArg List.length hFL
    simpa only [List.length_map] using this
  have b1 : (lv.rootB.toList == link.1) = true := by
    rw [hrbE, List.toList_toArray]
    exact bytesList_beq_self _
  have b2 : (lv.publics.toList == link.2) = true := by
    rw [hpubE, hpub, Array.toList_map, List.map_const', List.map_const', Array.length_toList, Array.length_toList, hts]
    exact bytesList_beq_self _
  have b3 : ((lv.streams.toList.flatMap fun s : Array Round => s.toList).length ==
      (pf.2.zip cs.tail.dropLast).length) = true := by
    rw [hlenFL, List.length_zip, Nat.min_eq_left hMc]
    exact beq_self_eq_true _
  have b4 : (((lv.streams.toList.flatMap fun s : Array Round => s.toList).zip (pf.2.zip cs.tail.dropLast)).all
      fun q => q.1.bytes == some ⟨q.2.1.toArray⟩) = true := by
    rw [List.all_eq_true]
    intro q hq
    obtain ⟨t, ht, rfl⟩ := List.mem_iff_getElem.1 hq
    have ht₁ : t < (lv.streams.toList.flatMap fun s : Array Round => s.toList).length :=
      lt_of_lt_of_le ht (by rw [List.length_zip]; exact Nat.min_le_left _ _)
    have ht₂ : t < (pf.2.zip cs.tail.dropLast).length :=
      lt_of_lt_of_le ht (by rw [List.length_zip]; exact Nat.min_le_right _ _)
    have ht₃ : t < pf.2.length := by
      rw [List.length_zip] at ht₂
      exact lt_of_lt_of_le ht₂ (Nat.min_le_left _ _)
    have e1 : (lv.streams.toList.flatMap fun s : Array Round => s.toList)[t].bytes = some ⟨pf.2[t].toArray⟩ := by
      have := congrArg (·[t]?) hFL
      simp only [List.getElem?_map, List.getElem?_eq_getElem ht₁, List.getElem?_eq_getElem ht₃, Option.map_some]
        at this
      exact Option.some.inj this
    rw [List.getElem_zip]
    dsimp only
    rw [e1, List.getElem_zip]
    exact bytes_beq_self _
  rw [b1, b2, b3, b4]
  rfl

end RecDeploy
