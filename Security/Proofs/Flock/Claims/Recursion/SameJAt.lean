import Proofs.Flock.Claims.Recursion.DecodeSame
import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.Claims.Recursion.NextShaDec

/-!
# The record the verdict reads is alike an honest record

`jAt` is the reader's record (`recordObj`) with its proof digests replaced (`shaSubJ`), then its coins erased and
filled from the live coins (`coinSubJ`). Each step maps the record's values in place, keeping its keys, so the result
is alike (`Same`) the honest record of the same parts, its rounds' coins the fill's (`fillStreams`), wherever the fill
finds a coin for every recorded round (`same_coinSubJ`). The fill finds one wherever `liveOf` reads the substituted
record (`fill_of_live`): an erased round's coins are no list.

* **By level** (`same_round`, `same_stream`, `same_link`, `same_record`): each object of the record, through the maps
  the substitutions make of it, by `same_mkObj`.
* **The offsets don't read the proof digests** (`streamOffset_shaSubJ`).
-/

namespace RecDeploy.DecodeAt

open Flock
open Lean (Json)
open Claims.Json (Same Good)
open FlockDeploy (Coin Live mapKey mapArr mapArrIdx mapStream mapRounds setPoints eraseCoins fillCoins coinSubJ
  shaSubJ shaArr_honest hexArr nOf nameOf streamOffset liveOfJ streamJ roundJ getObjVal?_mapKey_self
  getObjVal?_mapKey_ne getObjValAs_mapKey_ne nat_mapKey_ne arr_mapKey_mapArr arr_mapKey_ne bind_ok
  array_mapM_ok_inv streamJ_map roundJ_ok fld_mapKey_self proofKey_ne_streams proofKey_ne_link)
open FlockComplete (dgOf)
open FlockComplete.JsonRT (WfObj wfObj_mkObj same_mkObj same_of_good getObjVal?_mkObj getObjVal?_mkObj_ne
  ne_of_contains_false recordObj recordList recordList_distinct linkObj linkList linkList_distinct streamObj streamList
  streamList_distinct roundObj roundList roundList_distinct good_recordObj good_linkObj good_streamObj good_roundObj
  good_arr_str dk_cases proofObj Distinct get?_mkObj)

/-! ## In-place maps -/

theorem wfObj_mapKey {j : Json} (h : WfObj j) (k : String) (g : Json → Json) : WfObj (mapKey k g j) := by
  obtain ⟨n, rfl, hn⟩ := h
  exact ⟨_, rfl, ⟨⟨Std.DTreeMap.Internal.Impl.WF.map hn.out.out⟩⟩⟩

theorem good_of_mkObj {l : List (String × Json)} (hd : Distinct l) (hg : Good (Json.mkObj l)) {k : String}
    {v : Json} (hm : (k, v) ∈ l) : Good v := by
  unfold Json.mkObj at hg
  cases hg with
  | obj _ h => exact h k v (get?_mkObj hd hm)

theorem keep2 {k k₀ : String} (hk : k ≠ k₀) (g₁ g₂ : Json → Json) {l : List (String × Json)} (hd : Distinct l)
    {v : Json} (hm : (k, v) ∈ l) (hg : Good (Json.mkObj l)) :
    ∃ v', (mapKey k₀ g₂ (mapKey k₀ g₁ (Json.mkObj l))).getObjVal? k = .ok v' ∧ Same v v' := by
  rw [getObjVal?_mapKey_ne _ _ hk, getObjVal?_mapKey_ne _ _ hk]
  exact ⟨v, getObjVal?_mkObj hd hm, same_of_good (good_of_mkObj hd hg hm)⟩

theorem keep5 {k ks kl kp : String} (hs : k ≠ ks) (hl : k ≠ kl) (hp : k ≠ kp) (g₁ g₂ g₃ g₄ g₅ : Json → Json)
    {l : List (String × Json)} (hd : Distinct l) {v : Json} (hm : (k, v) ∈ l) (hg : Good (Json.mkObj l)) :
    ∃ v', (mapKey ks g₅ (mapKey kl g₄ (mapKey ks g₃ (mapKey kl g₂ (mapKey kp g₁ (Json.mkObj l)))))).getObjVal? k =
      .ok v' ∧ Same v v' := by
  rw [getObjVal?_mapKey_ne _ _ hs, getObjVal?_mapKey_ne _ _ hl, getObjVal?_mapKey_ne _ _ hs,
    getObjVal?_mapKey_ne _ _ hl, getObjVal?_mapKey_ne _ _ hp]
  exact ⟨v, getObjVal?_mkObj hd hm, same_of_good (good_of_mkObj hd hg hm)⟩

/-! ## A round -/

theorem nOf_erased (spec : SessionSpec) (h : ByteArray → ByteArray) (k : ℕ) (r : Round) :
    nOf (mapKey "coins" (fun _ => .null) (roundObj spec h k r)) = r.n := by
  have hn0 : (roundObj spec h k r).getObjVal? "n" = .ok (Lean.toJson r.n) :=
    getObjVal?_mkObj (roundList_distinct spec h k r) (by simp [roundList])
  have hn : Record.nat (roundObj spec h k r) "n" = .ok r.n := by
    simp only [Record.nat, Record.fld, hn0]
    rfl
  simp only [nOf, nat_mapKey_ne _ _ (show "n" ≠ "coins" by decide), hn]

/-- **A round with its coins erased and refilled from `c`** is alike the round whose coins are `c`'s first `n`. -/
theorem same_round (spec : SessionSpec) (h : ByteArray → ByteArray) (k : ℕ) (r : Round) (c : Coin) :
    Same (roundObj spec h k { r with coins := c.vals r.n })
      (mapKey "coins" (fun _ => hexArr (c.vals (nOf (mapKey "coins" (fun _ => .null) (roundObj spec h k r)))))
        (mapKey "coins" (fun _ => .null) (roundObj spec h k r))) := by
  rw [nOf_erased]
  have hg := good_roundObj spec h k r
  have hd := roundList_distinct spec h k r
  refine same_mkObj (roundList_distinct ..) (wfObj_mapKey (wfObj_mapKey (wfObj_mkObj _) _ _) _ _) ?_ ?_
  · intro key v hm
    simp only [roundList, List.mem_cons, List.mem_nil_iff, or_false, Prod.mk.injEq] at hm
    rcases hm with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · refine keep2 ?_ _ _ hd (by simp [roundList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep2 ?_ _ _ hd (by simp [roundList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep2 ?_ _ _ hd (by simp [roundList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · have hco : (roundObj spec h k r).getObjVal? "coins" = .ok (Json.arr (r.coins.map FlockComplete.f128J)) :=
        getObjVal?_mkObj hd (by simp [roundList])
      rw [getObjVal?_mapKey_self, getObjVal?_mapKey_self, hco]
      exact ⟨_, rfl, same_of_good (good_arr_str (c.vals r.n))⟩
    · refine keep2 ?_ _ _ hd (by simp [roundList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep2 ?_ _ _ hd (by simp [roundList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
  · intro key v hc
    have hk := ne_of_contains_false hc (k' := "coins")
      (v := Json.arr ((c.vals r.n).map FlockComplete.f128J)) (by simp [roundList])
    rw [getObjVal?_mapKey_ne _ _ hk, getObjVal?_mapKey_ne _ _ hk]
    have hkeys : (roundList spec h k r).map Prod.fst = (roundList spec h k { r with coins := c.vals r.n }).map Prod.fst := by
      simp [roundList]
    exact getObjVal?_mkObj_ne (by rw [hkeys]; exact hc) v

/-! ## A stream -/

theorem getObjValAs_streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : (streamObj spec h rootB q).getObjValAs? String "stream" = .ok q.1.2.2 := by
  show (Json.mkObj (streamList spec h rootB q)).getObjValAs? String "stream" = _
  unfold Json.getObjValAs? Json.getObjValD
  rw [getObjVal?_mkObj (streamList_distinct spec h rootB q) (k := "stream") (v := .str q.1.2.2)
    (by simp [streamList])]
  rfl

theorem nameOf_streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : nameOf (streamObj spec h rootB q) = q.1.2.2 := by
  unfold nameOf
  simp only [getObjValAs_streamObj]
  rfl

theorem nameOf_mapKey_rounds (g : Json → Json) (s : Json) : nameOf (mapKey "rounds" g s) = nameOf s := by
  unfold nameOf
  rw [getObjValAs_mapKey_ne _ _ (by decide : "stream" ≠ "rounds")]

theorem getObjValAs_mapStream (g : String → ℕ → Json → Json) (s : Json) :
    (mapStream g s).getObjValAs? String "stream" = s.getObjValAs? String "stream" :=
  getObjValAs_mapKey_ne _ _ (by decide)

/-- **A stream through two round maps** is alike the stream of rounds `rs'` where each round `k` is alike its two
maps. -/
theorem same_stream (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : ℕ × ℕ × String) (rs rs' : Array Round) (g₁ g₂ : String → ℕ → Json → Json) (hsz : rs'.size = rs.size)
    (hr : ∀ k (hk : k < rs.size),
      Same (roundObj spec h k rs'[k]!) (g₂ q.2.2 k (g₁ q.2.2 k (roundObj spec h k rs[k]!)))) :
    Same (streamObj spec h rootB (q, rs')) (mapStream g₂ (mapStream g₁ (streamObj spec h rootB (q, rs)))) := by
  have hg := good_streamObj spec h rootB (q, rs)
  have hd := streamList_distinct spec h rootB (q, rs)
  simp only [mapStream, nameOf_mapKey_rounds, nameOf_streamObj]
  refine same_mkObj (streamList_distinct ..) (wfObj_mapKey (wfObj_mapKey (wfObj_mkObj _) _ _) _ _) ?_ ?_
  · intro key v hm
    simp only [streamList, List.mem_cons, List.mem_nil_iff, or_false, Prod.mk.injEq] at hm
    rcases hm with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · refine keep2 ?_ _ _ hd (by simp [streamList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep2 ?_ _ _ hd (by simp [streamList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep2 ?_ _ _ hd (by simp [streamList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep2 ?_ _ _ hd (by simp [streamList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep2 ?_ _ _ hd (by simp [streamList]) hg
      first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · have hro : (streamObj spec h rootB (q, rs)).getObjVal? "rounds" =
          .ok (Json.arr ((Array.range rs.size).map fun k => roundObj spec h k rs[k]!)) :=
        getObjVal?_mkObj hd (by simp [streamList])
      rw [getObjVal?_mapKey_self, getObjVal?_mapKey_self, hro]
      refine ⟨_, rfl, ?_⟩
      simp only [mapArrIdx]
      refine .arr (by simp [hsz]) fun i h₁ h₂ => ?_
      simp only [Array.getElem_mapIdx, Array.getElem_map, Array.getElem_range]
      exact hr i (by simpa [hsz] using h₁)
  · intro key v hc
    have hk := ne_of_contains_false hc (k' := "rounds")
      (v := Json.arr ((Array.range rs'.size).map fun k => roundObj spec h k rs'[k]!)) (by simp [streamList])
    rw [getObjVal?_mapKey_ne _ _ hk, getObjVal?_mapKey_ne _ _ hk]
    have hkeys : (streamList spec h rootB (q, rs)).map Prod.fst = (streamList spec h rootB (q, rs')).map Prod.fst := by
      simp [streamList]
    exact getObjVal?_mkObj_ne (by rw [hkeys]; exact hc) v

/-! ## The link -/

/-- **The link with its points erased and set to `P`'s** is alike the link of points `P`. -/
theorem same_link {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (h : ByteArray → ByteArray)
    (rb : ByteArray) (P : Array (Array F128)) :
    Same (linkObj spec h #[rb] P)
      (mapKey "points" (fun _ => .arr (P.map hexArr)) (mapKey "points" (fun _ => .null) (linkObj spec h #[rb] P))) := by
  have hg := good_linkObj ht h rb P
  have hd := linkList_distinct spec h #[rb] P
  refine same_mkObj (linkList_distinct ..) (wfObj_mapKey (wfObj_mapKey (wfObj_mkObj _) _ _) _ _) ?_ ?_
  · intro key v hm
    by_cases hk : key = "points"
    · subst hk
      have e1 : (linkObj spec h #[rb] P).getObjVal? "points" = .ok v := getObjVal?_mkObj hd hm
      have e2 : (linkObj spec h #[rb] P).getObjVal? "points" =
          .ok (Json.arr (P.map fun p => Json.arr (p.map FlockComplete.f128J))) :=
        getObjVal?_mkObj hd (by simp [linkList])
      have hv : v = Json.arr (P.map fun p => Json.arr (p.map FlockComplete.f128J)) := by
        rw [e1] at e2
        exact Except.ok.inj e2
      subst hv
      rw [getObjVal?_mapKey_self, getObjVal?_mapKey_self, e2]
      exact ⟨_, rfl, same_of_good (good_of_mkObj (k := "points") hd hg (by simp [linkList]))⟩
    · exact keep2 hk _ _ hd hm hg
  · intro key v hc
    have hk := ne_of_contains_false hc (k' := "points")
      (v := Json.arr (P.map fun p => Json.arr (p.map FlockComplete.f128J))) (by simp [linkList])
    rw [getObjVal?_mapKey_ne _ _ hk, getObjVal?_mapKey_ne _ _ hk]
    exact getObjVal?_mkObj_ne hc v

/-! ## The record -/

/-- **The record through the substitutions' five maps** is alike the record of streams `SS`, where each stream of
`SS` is alike its stream of `S` through the two round maps. -/
theorem same_record {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (h : ByteArray → ByteArray)
    (rb : ByteArray) (P : Array (Array F128)) (S SS : Array (Array Round)) (F : Array ByteArray)
    (g₁ g₂ : String → ℕ → Json → Json) (hdg : Good (spec.unitDraw.getD Json.null)) (hsz : SS.size = S.size)
    (hst : ∀ i (h₁ : i < spec.repStreams.size) (h₂ : i < S.size),
      Same (streamObj spec h #[rb] (spec.repStreams[i], SS[i]'(by omega)))
        (mapStream g₂ (mapStream g₁ (streamObj spec h #[rb] (spec.repStreams[i], S[i]))))) :
    Same (recordObj spec h #[rb] P SS F)
      (mapRounds g₂ (setPoints (.arr (P.map hexArr)) (mapRounds g₁ (setPoints .null
        (shaSubJ spec (F.map (dgOf spec h)) (recordObj spec h #[rb] P S F)))))) := by
  have hg := good_recordObj ht h rb P S F hdg
  have hd := recordList_distinct spec h #[rb] P S F
  have hSL : ("streams" : String) ≠ "link" := by decide
  have hLS : ("link" : String) ≠ "streams" := by decide
  have hPS := proofKey_ne_streams spec
  have hPL := proofKey_ne_link spec
  simp only [mapRounds, setPoints, shaSubJ]
  refine same_mkObj (recordList_distinct ..) (wfObj_mapKey (wfObj_mapKey (wfObj_mapKey (wfObj_mapKey
    (wfObj_mapKey (wfObj_mkObj _) _ _) _ _) _ _) _ _) _ _) ?_ ?_
  · intro key v hm
    simp only [recordList, List.mem_cons, List.mem_nil_iff, or_false, Prod.mk.injEq] at hm
    rcases hm with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ |
      ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · refine keep5 ?_ ?_ ?_ _ _ _ _ _ hd (by simp [recordList]) hg <;>
        first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep5 ?_ ?_ ?_ _ _ _ _ _ hd (by simp [recordList]) hg <;>
        first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep5 ?_ ?_ ?_ _ _ _ _ _ hd (by simp [recordList]) hg <;>
        first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep5 ?_ ?_ ?_ _ _ _ _ _ hd (by simp [recordList]) hg <;>
        first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · refine keep5 ?_ ?_ ?_ _ _ _ _ _ hd (by simp [recordList]) hg <;>
        first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · -- the link
      have hl : (recordObj spec h #[rb] P S F).getObjVal? "link" = .ok (linkObj spec h #[rb] P) :=
        getObjVal?_mkObj hd (by simp [recordList])
      rw [getObjVal?_mapKey_ne _ _ hLS, getObjVal?_mapKey_self, getObjVal?_mapKey_ne _ _ hLS, getObjVal?_mapKey_self,
        getObjVal?_mapKey_ne _ _ hPL.symm, hl]
      exact ⟨_, rfl, same_link ht h rb P⟩
    · refine keep5 ?_ ?_ ?_ _ _ _ _ _ hd (by simp [recordList]) hg <;>
        first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · -- the streams
      have hs : (recordObj spec h #[rb] P S F).getObjVal? "streams" =
          .ok (Json.arr ((spec.repStreams.zip S).map (streamObj spec h #[rb]))) :=
        getObjVal?_mkObj hd (by simp [recordList])
      rw [getObjVal?_mapKey_self, getObjVal?_mapKey_ne _ _ hSL, getObjVal?_mapKey_self, getObjVal?_mapKey_ne _ _ hSL,
        getObjVal?_mapKey_ne _ _ hPS.symm, hs]
      refine ⟨_, rfl, ?_⟩
      simp only [mapArr]
      refine .arr (by simp [hsz]) fun i h₁ h₂ => ?_
      simp only [Array.getElem_map, Array.getElem_zip]
      exact hst i (by simp at h₁; omega) (by simp at h₂; omega)
    · refine keep5 ?_ ?_ ?_ _ _ _ _ _ hd (by simp [recordList]) hg <;>
        first | decide | (rcases dk_cases spec with e | e <;> simp [e, toString] <;> decide)
    · -- the proof digests
      have hp : (recordObj spec h #[rb] P S F).getObjVal? s!"proof_{spec.dk}" =
          .ok (Json.arr ((spec.repStreams.zip F).map (proofObj spec h))) :=
        getObjVal?_mkObj hd (by simp [recordList])
      rw [getObjVal?_mapKey_ne _ _ hPS, getObjVal?_mapKey_ne _ _ hPL, getObjVal?_mapKey_ne _ _ hPS,
        getObjVal?_mapKey_ne _ _ hPL, getObjVal?_mapKey_self, hp]
      refine ⟨_, rfl, ?_⟩
      show Same _ (FlockDeploy.shaArr spec (F.map (dgOf spec h)))
      rw [shaArr_honest]
      exact same_of_good (good_of_mkObj (k := s!"proof_{spec.dk}") hd hg (by simp [recordList]))
  · intro key v hc
    have hS := ne_of_contains_false hc (k' := "streams")
      (v := Json.arr ((spec.repStreams.zip SS).map (streamObj spec h #[rb]))) (by simp [recordList])
    have hL := ne_of_contains_false hc (k' := "link") (v := linkObj spec h #[rb] P) (by simp [recordList])
    have hP := ne_of_contains_false hc (k' := s!"proof_{spec.dk}")
      (v := Json.arr ((spec.repStreams.zip F).map (proofObj spec h))) (by simp [recordList])
    rw [getObjVal?_mapKey_ne _ _ hS, getObjVal?_mapKey_ne _ _ hL, getObjVal?_mapKey_ne _ _ hS,
      getObjVal?_mapKey_ne _ _ hL, getObjVal?_mapKey_ne _ _ hP]
    have hkeys : (recordList spec h #[rb] P S F).map Prod.fst = (recordList spec h #[rb] P SS F).map Prod.fst := by
      simp [recordList]
    exact getObjVal?_mkObj_ne (by rw [hkeys]; exact hc) v

/-! ## The coin substitution -/

/-- The coin the fill writes into round `k` of stream `name`, at the streams' offsets `off`. -/
def fillCoin (mids : List (Claims.Msg × Coin)) (off : String → ℕ) (name : String) (k : ℕ) : Coin :=
  ((mids[off name + k]?).map Prod.snd).getD default

/-- **The streams with the fill's coins**: round `k` of rep stream `i` holds the first `n` of its fill coin. -/
def fillStreams (spec : SessionSpec) (mids : List (Claims.Msg × Coin)) (off : String → ℕ)
    (S : Array (Array Round)) : Array (Array Round) :=
  S.mapIdx fun i rs => rs.mapIdx fun k r => { r with coins := (fillCoin mids off (spec.repStreams[i]!).2.2 k).vals r.n }

/-- **The substituted record is alike an honest record**: the record of streams `S` with its proof digests the files'
and its coins substituted is alike the record whose rounds hold the fill's coins, where the fill finds a coin for every
recorded round. -/
theorem same_coinSubJ {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (h : ByteArray → ByteArray)
    (rb : ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (S : Array (Array Round)) (F : Array ByteArray)
    (hdg : Good (spec.unitDraw.getD Json.null))
    (hfill : ∀ i (hi : i < spec.repStreams.size) (hS : i < S.size) k, k < S[i].size →
      (mids[streamOffset spec (eraseCoins (shaSubJ spec (F.map (dgOf spec h))
        (recordObj spec h #[rb] (FlockDeploy.pointsOf spec c₁) S F))) spec.repStreams[i].2.2 + k]?).isSome) :
    Same (recordObj spec h #[rb] (FlockDeploy.pointsOf spec c₁)
        (fillStreams spec mids (streamOffset spec (eraseCoins (shaSubJ spec (F.map (dgOf spec h))
          (recordObj spec h #[rb] (FlockDeploy.pointsOf spec c₁) S F)))) S) F)
      (coinSubJ spec c₁ mids (shaSubJ spec (F.map (dgOf spec h))
        (recordObj spec h #[rb] (FlockDeploy.pointsOf spec c₁) S F))) := by
  unfold coinSubJ fillCoins
  generalize streamOffset spec (eraseCoins (shaSubJ spec (F.map (dgOf spec h))
    (recordObj spec h #[rb] (FlockDeploy.pointsOf spec c₁) S F))) = off at hfill ⊢
  unfold eraseCoins
  refine same_record ht h rb _ S _ F _ _ hdg (by simp [fillStreams]) fun i h₁ h₂ => ?_
  refine same_stream spec h #[rb] _ S[i] _ _ _ (by simp [fillStreams]) fun k hk => ?_
  obtain ⟨⟨m, c⟩, hmc⟩ := Option.isSome_iff_exists.1 (hfill i h₁ h₂ k hk)
  have hk' : k < ((fillStreams spec mids off S)[i]'(by simp [fillStreams]; omega)).size := by
    simp [fillStreams]; exact hk
  have e : ((fillStreams spec mids off S)[i]'(by simp [fillStreams]; omega))[k]! =
      { S[i][k]! with coins := c.vals (S[i][k]!).n } := by
    rw [getElem!_pos _ k hk', getElem!_pos S[i] k hk]
    simp only [fillStreams, Array.getElem_mapIdx, fillCoin, getElem!_pos spec.repStreams i h₁, hmc]
    rfl
  rw [e]
  simp only [hmc]
  exact same_round spec h k S[i][k]! c

/-- **The offsets don't read the proof digests.** -/
theorem streamOffset_shaSubJ (spec : SessionSpec) (ds : Array ByteArray) (j : Json) :
    streamOffset spec (eraseCoins (shaSubJ spec ds j)) = streamOffset spec (eraseCoins j) := by
  have hSL : ("streams" : String) ≠ "link" := by decide
  have ha : Record.arr (eraseCoins (shaSubJ spec ds j)) "streams" = Record.arr (eraseCoins j) "streams" := by
    simp only [eraseCoins, mapRounds, setPoints, shaSubJ, arr_mapKey_mapArr, arr_mapKey_ne _ _ hSL,
      arr_mapKey_ne _ _ (proofKey_ne_streams spec).symm]
  have hro : FlockDeploy.roundsOf (eraseCoins (shaSubJ spec ds j)) = FlockDeploy.roundsOf (eraseCoins j) := by
    funext name
    unfold FlockDeploy.roundsOf
    rw [ha]
  funext name
  unfold streamOffset
  rw [hro]

/-! ## The fill finds a coin wherever `liveOf` reads -/

theorem find?_getElem' {α : Type} {a : Array α} {p : α → Bool} {k : ℕ} (hk : k < a.size) (hp : p a[k] = true)
    (hne : ∀ (j : ℕ) (hj : j < k), p (a[j]'(by omega)) = false) : a.find? p = some a[k] := by
  rw [Array.find?_eq_some_iff_getElem]
  exact ⟨hp, k, hk, rfl, fun j hj => by simp [hne j hj]⟩

theorem arr_of' {j : Json} {k : String} {a : Array Json} (h : j.getObjVal? k = .ok (.arr a)) :
    Record.arr j k = .ok a := by
  have hf : Record.fld j k = .ok (.arr a) := by simp only [Record.fld, h]
  simp only [Record.arr, hf]
  rfl

/-- An erased value is no list. -/
theorem arr_mapKey_null' {k : String} {j : Json} {a : Array Json}
    (h : Record.arr (mapKey k (fun _ => .null) j) k = .ok a) : False := by
  unfold Record.arr at h
  cases hf : Record.fld (mapKey k (fun _ => .null) j) k with
  | error e => rw [hf] at h; cases h
  | ok v =>
    obtain ⟨v₀, -, rfl⟩ := fld_mapKey_self hf
    rw [hf] at h
    cases h

set_option maxHeartbeats 1600000 in
/-- **The fill finds a coin for every recorded round** of the record of streams `S` wherever `liveOf` reads its
substituted value, at rep streams of distinct names. -/
theorem fill_of_live {spec : SessionSpec} {h : ByteArray → ByteArray} {rootB : Array ByteArray}
    {P : Array (Array F128)} {S : Array (Array Round)} {F : Array ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {lv : Live} (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hl : liveOfJ spec (coinSubJ spec c₁ mids (recordObj spec h rootB P S F)) = .ok lv) :
    ∀ i (hi : i < spec.repStreams.size) (hS : i < S.size) k, k < S[i].size →
      (mids[streamOffset spec (eraseCoins (recordObj spec h rootB P S F)) spec.repStreams[i].2.2 + k]?).isSome := by
  intro i hi hS k hk
  have hne : ∀ (t t' : ℕ) (ht : t < spec.repStreams.size) (ht' : t' < spec.repStreams.size), t ≠ t' →
      spec.repStreams[t].2.2 ≠ spec.repStreams[t'].2.2 := by
    intro t t' ht ht' htt e
    apply htt
    have e' : (spec.repStreams.toList.map (·.2.2))[t]'(by simpa using ht) =
        (spec.repStreams.toList.map (·.2.2))[t']'(by simpa using ht') := by simpa using e
    exact (List.Nodup.getElem_inj_iff hnd).1 e'
  unfold liveOfJ at hl
  obtain ⟨_, -, hl⟩ := bind_ok hl
  obtain ⟨_, -, hl⟩ := bind_ok hl
  obtain ⟨_, -, hl⟩ := bind_ok hl
  obtain ⟨_, -, hl⟩ := bind_ok hl
  obtain ⟨_, -, hl⟩ := bind_ok hl
  obtain ⟨_, -, hl⟩ := bind_ok hl
  obtain ⟨recs, hrecs, hl⟩ := bind_ok hl
  obtain ⟨streams, hstreams, -⟩ := bind_ok hl
  have hSL : ("streams" : String) ≠ "link" := by decide
  have hE : Record.arr (eraseCoins (recordObj spec h rootB P S F)) "streams" =
      .ok (((spec.repStreams.zip S).map (streamObj spec h rootB)).map
        (mapStream fun _ _ => mapKey "coins" fun _ => .null)) := by
    have h0 : (recordObj spec h rootB P S F).getObjVal? "streams" =
        .ok (Json.arr ((spec.repStreams.zip S).map (streamObj spec h rootB))) :=
      getObjVal?_mkObj (recordList_distinct ..) (by simp [recordList])
    simp only [eraseCoins, mapRounds, setPoints]
    rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ hSL, arr_of' h0]
    rfl
  simp only [coinSubJ, fillCoins, mapRounds, setPoints] at hrecs
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ hSL, hE] at hrecs
  injection hrecs with hrecs
  subst hrecs
  obtain ⟨hsz, hf⟩ := array_mapM_ok_inv _ _ _ hstreams
  obtain ⟨s₁, rs₁, hfind, ha, hm⟩ := streamJ_map (hf i hi)
  have hfind' : (((spec.repStreams.zip S).map (streamObj spec h rootB)).map
      (mapStream fun _ _ => mapKey "coins" fun _ => .null)).find?
        (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[i].2.2) =
      some ((((spec.repStreams.zip S).map (streamObj spec h rootB)).map
        (mapStream fun _ _ => mapKey "coins" fun _ => .null))[i]'(by simp; omega)) := by
    refine find?_getElem' (by simp; omega) ?_ fun j hj => ?_
    · simp only [Array.getElem_map, Array.getElem_zip, getObjValAs_mapStream, getObjValAs_streamObj]
      simp [Except.toOption]
    · simp only [Array.getElem_map, Array.getElem_zip, getObjValAs_mapStream, getObjValAs_streamObj]
      simpa [Except.toOption] using hne j i (by omega) hi (by omega)
  obtain rfl := Option.some.inj (hfind.symm.trans hfind')
  simp only [Array.getElem_map, Array.getElem_zip, mapStream, nameOf_streamObj] at ha
  rw [FlockDeploy.arr_mapKey_mapArrIdx] at ha
  have hro : Record.arr (streamObj spec h rootB (spec.repStreams[i], S[i])) "rounds" =
      .ok ((Array.range S[i].size).map fun k => roundObj spec h k S[i][k]!) := by
    have h0 : (streamObj spec h rootB (spec.repStreams[i], S[i])).getObjVal? "rounds" =
        .ok (Json.arr ((Array.range S[i].size).map fun k => roundObj spec h k S[i][k]!)) :=
      getObjVal?_mkObj (streamList_distinct ..) (by simp [streamList])
    exact arr_of' h0
  rw [hro] at ha
  injection ha with ha
  subst ha
  obtain ⟨hsz', hr⟩ := array_mapM_ok_inv _ _ _ hm
  have hr := hr k (by simpa using hk)
  simp only [Array.getElem_mapIdx, Array.getElem_map, Array.getElem_range] at hr
  cases hmc : mids[streamOffset spec (eraseCoins (recordObj spec h rootB P S F)) spec.repStreams[i].2.2 + k]? with
  | some mc => rfl
  | none =>
    exfalso
    simp only [List.get_eq_getElem, Array.getElem_toList] at hr
    rw [hmc] at hr
    obtain ⟨-, hc⟩ := roundJ_ok hr
    obtain ⟨a, ha, -⟩ := bind_ok hc
    exact arr_mapKey_null' ha

end RecDeploy.DecodeAt
