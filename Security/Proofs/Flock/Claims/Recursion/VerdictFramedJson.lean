import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.Claims.Recursion.NextShaDec
import Proofs.Flock.Claims.KS.PlainDrawJRun

/-!
# The decided record's rep streams: the reader's frames, through the substitutions

Row B's `framed` asks that rep `r`'s rounds in the decoded session hold the rep's frames. At a decision of `nextR`, the
record the verdict decodes (`jAt`) is the reader's (`recJ`) with its coins substituted and its proof digests replaced,
and the reader's rep streams carry, round by round, the round messages `readPs` read with its schedule's coin counts
(`recStreams`):

* **The round messages are the frames** (`readPs_framesR`, `decR_framesR`): rep `k`'s block of the round messages is its
  proof's framed bytes at the `T` it fits at.
* **The substitutions keep what custody reads** (`arr_streams_jAt`): every stream's name, and every round's count and
  message, read in `jAt` as in the reader's record (`KeptS`, `KeptR`); only the coins and the link's points change.
* **A round reads back** (`roundFields_roundObj`): where the decoder read a round of the reader's record, its count is
  the round's and its bytes are the round's message.
-/

namespace FlockDeploy

open Flock
open Lean (Json)
open FlockComplete.JsonRT (roundObj roundList streamObj streamList recordObj recordList getObjVal?_mkObj
  roundList_distinct streamList_distinct recordList_distinct)
open Claims.Bridge.Custody (RoundFields)

/-! ## The reader's round messages are the frames -/

/-- **Each rep's block of the round messages `readPs` reads is its proof's framed bytes**, at a `T` the proof fits its
opened rounds at. -/
theorem readPs_framesR {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) :
    ∀ k < 2 * sts.size, ∃ T, FitsRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k [])
      pf.1[k]! T ∧ (cutLens (repLens spec sts) pf.2).getD k [] = framesOf (repCtx spec sts c₁ r₀ cs k) pf.1[k]! T := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    have hfit : ∀ (k : ℕ) (hk : k < rs.length), ∃ T,
        FitsRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) rs[k].1 T ∧
          rs[k].2 = framesOf (repCtx spec sts c₁ r₀ cs k) rs[k].1 T := by
      intro k hk
      have hg := hget k (by rw [List.length_range]; omega) hk
      rw [List.getElem_range] at hg
      exact readRep_fits (p := rs[k].1) (ms := rs[k].2) hg
    have hmap : (rs.map Prod.snd).map List.length = repLens spec sts := by
      refine List.ext_getElem (by simp [repLens, hlen]) fun k h₁ h₂ => ?_
      have hk : k < rs.length := by simpa using h₁
      obtain ⟨T, hT, he⟩ := hfit k hk
      rw [List.getElem_map, List.getElem_map, he, framesOf, List.length_map, hT.rounds, ← hT.sched]
      simp [repLens, repCtx]
    have hcut : cutLens (repLens spec sts) (rs.map Prod.snd).flatten = rs.map Prod.snd := by
      rw [← hmap]
      exact cutLens_flatten_eq _
    intro k hk
    have hk' : k < rs.length := by omega
    obtain ⟨T, hT, he⟩ := hfit k hk'
    have hp : (rs.map Prod.fst).toArray[k]! = rs[k].1 := by
      rw [getElem!_pos _ k (by simpa using hk')]
      simp
    refine ⟨T, ?_, ?_⟩
    · dsimp only
      rw [hp]
      exact hT
    · dsimp only
      rw [hp, hcut, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hk']
      simpa using he
  · rw [if_neg hc] at h
    cases h

/-- **At a decision, the record's round messages are the reps' frames**: the reader's record is `recJ` at round
messages `ms` whose `k`-th block is rep `k`'s proof's framed bytes, at a `T` the proof fits its opened rounds at. -/
theorem decR_framesR {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) :
    ∃ (r₀ : List ByteArray) (rss : List (List ByteArray)) (ms : List Claims.Msg),
      (rowsOf (as.headD [])).map (List.map nodeOf) = some r₀ ∧ as.tail.dropLast.mapM rowsOf = some rss ∧
      d.f.j = recJ d.spec d.sts d.link d.c₁ ms cs.tail.dropLast d.f.ps ∧
      ∀ k < 2 * d.sts.size, ∃ T, FitsRep (repCtx d.spec d.sts d.c₁ r₀ cs.tail.dropLast k)
        ((cutLens (repLens d.spec d.sts) rss).getD k []) d.f.ps[k]! T ∧
        (cutLens (repLens d.spec d.sts) ms).getD k [] =
          framesOf (repCtx d.spec d.sts d.c₁ r₀ cs.tail.dropLast k) d.f.ps[k]! T := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, h6, h7, -, h9, rfl⟩ := decR_some h
  exact ⟨r₀, rss, pf.2, h6, h7, rfl, readPs_framesR h9⟩

/-! ## The substitutions keep the streams' names, counts and messages -/

theorem forall₂_getElem {α β : Type _} {R : α → β → Prop} {l₁ : List α} {l₂ : List β} :
    List.Forall₂ R l₁ l₂ ↔ l₁.length = l₂.length ∧ ∀ i (h₁ : i < l₁.length) (h₂ : i < l₂.length), R l₁[i] l₂[i] := by
  rw [List.forall₂_iff_get]
  simp only [List.get_eq_getElem]

/-- `b` reads as `a` at a round's count and message. -/
structure KeptR (a b : Json) : Prop where
  n : b.getObjVal? "n" = a.getObjVal? "n"
  msg : b.getObjVal? "msg" = a.getObjVal? "msg"

/-- `b` reads as `a` at a stream's name, and its rounds as `a`'s at their counts and messages. -/
structure KeptS (a b : Json) : Prop where
  stream : b.getObjVal? "stream" = a.getObjVal? "stream"
  rounds : ∀ rs, Record.arr a "rounds" = .ok rs →
    ∃ rs', Record.arr b "rounds" = .ok rs' ∧ List.Forall₂ KeptR rs.toList rs'.toList

theorem KeptR.trans {a b c : Json} (h : KeptR a b) (h' : KeptR b c) : KeptR a c := ⟨h'.n.trans h.n, h'.msg.trans h.msg⟩

theorem KeptS.trans {a b c : Json} (h : KeptS a b) (h' : KeptS b c) : KeptS a c := by
  refine ⟨h'.stream.trans h.stream, fun rs hrs => ?_⟩
  obtain ⟨rs', h1, h2⟩ := h.rounds rs hrs
  obtain ⟨rs'', h3, h4⟩ := h'.rounds rs' h1
  refine ⟨rs'', h3, ?_⟩
  rw [forall₂_getElem] at h2 h4 ⊢
  exact ⟨h2.1.trans h4.1, fun i h₁ h₂ => KeptR.trans (h2.2 i h₁ (by omega)) (h4.2 i (by omega) h₂)⟩

theorem keptR_coins (f : Json → Json) (r : Json) : KeptR r (mapKey "coins" f r) :=
  ⟨getObjVal?_mapKey_ne _ _ (by decide), getObjVal?_mapKey_ne _ _ (by decide)⟩

theorem keptS_mapStream (g : String → ℕ → Json → Json) (hg : ∀ n k r, KeptR r (g n k r)) (s : Json) :
    KeptS s (mapStream g s) := by
  refine ⟨getObjVal?_mapKey_ne _ _ (by decide), fun rs hrs => ⟨rs.mapIdx (g (nameOf s)), ?_, ?_⟩⟩
  · rw [mapStream, arr_mapKey_mapArrIdx, hrs]
    rfl
  · rw [forall₂_getElem]
    refine ⟨by simp, fun i h₁ h₂ => ?_⟩
    simp only [Array.getElem_toList, Array.getElem_mapIdx]
    exact hg _ _ _

theorem arr_eraseCoins (j : Json) :
    Record.arr (eraseCoins j) "streams" =
      (·.map (mapStream fun _ _ => mapKey "coins" fun _ => .null)) <$> Record.arr j "streams" := by
  unfold eraseCoins mapRounds setPoints
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link")]

/-- **The verdict's record keeps the reader's streams**: each rep stream of `jAt` reads as the reader's record's, at
its name and at its rounds' counts and messages. -/
theorem arr_streams_jAt {H : Claims.Msg → Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {recs : Array Json}
    (h : Record.arr (jAt H spec sts c₁ mids f) "streams" = .ok recs) :
    ∃ recs₀, Record.arr f.j "streams" = .ok recs₀ ∧ List.Forall₂ KeptS recs₀.toList recs.toList := by
  unfold jAt coinSubJ fillCoins at h
  simp only [mapRounds, setPoints] at h
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link"), arr_eraseCoins, shaSubJ,
    arr_mapKey_ne _ _ (proofKey_ne_streams spec).symm] at h
  cases ha : Record.arr f.j "streams" with
  | error e => rw [ha] at h; cases h
  | ok a =>
    rw [ha] at h
    cases h
    refine ⟨a, rfl, ?_⟩
    rw [forall₂_getElem]
    refine ⟨by simp, fun i h₁ h₂ => ?_⟩
    simp only [Array.getElem_toList, Array.getElem_map]
    refine (keptS_mapStream _ (fun _ _ r => keptR_coins _ r) _).trans (keptS_mapStream _ (fun n k r => ?_) _)
    split
    · exact keptR_coins _ _
    · exact ⟨rfl, rfl⟩

/-! ## The reader's record -/

theorem arr_recordObj_streams (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    Record.arr (recordObj spec h rootB points streams proofs) "streams" =
      .ok ((spec.repStreams.zip streams).map (streamObj spec h rootB)) := by
  have hv : (recordObj spec h rootB points streams proofs).getObjVal? "streams" =
      .ok (Json.arr ((spec.repStreams.zip streams).map (streamObj spec h rootB))) :=
    getObjVal?_mkObj (recordList_distinct ..) (by simp [recordList])
  simp only [Record.arr, Record.fld, hv]
  rfl

theorem streamObj_stream (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : (streamObj spec h rootB q).getObjVal? "stream" = .ok (Json.str q.1.2.2) :=
  getObjVal?_mkObj (streamList_distinct ..) (by simp [streamList])

theorem arr_streamObj_rounds (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) :
    Record.arr (streamObj spec h rootB q) "rounds" = .ok ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!) := by
  have hv : (streamObj spec h rootB q).getObjVal? "rounds" =
      .ok (Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!)) :=
    getObjVal?_mkObj (streamList_distinct ..) (by simp [streamList])
  simp only [Record.arr, Record.fld, hv]
  rfl

theorem roundObj_n (spec : SessionSpec) (h : ByteArray → ByteArray) (k : ℕ) (r : Round) :
    (roundObj spec h k r).getObjVal? "n" = .ok (Lean.toJson r.n) :=
  getObjVal?_mkObj (roundList_distinct ..) (by simp [roundList])

theorem roundObj_msg (spec : SessionSpec) (h : ByteArray → ByteArray) (k : ℕ) (r : Round) :
    (roundObj spec h k r).getObjVal? "msg" = .ok (FlockComplete.hexJ (r.bytes.getD ByteArray.empty)) :=
  getObjVal?_mkObj (roundList_distinct ..) (by simp [roundList])

/-- **A round of the reader's record reads back**: where the decoder read a round that reads as round `k` of the
reader's record, its count is the round's and its bytes the round's message. -/
theorem roundFields_roundObj {spec : SessionSpec} {h : ByteArray → ByteArray} {k : ℕ} {r : Round} {x : Json}
    (hx : KeptR (roundObj spec h k r) x) {rd : Round} (hf : RoundFields x rd.n rd.coins rd.bytes) :
    rd.n = r.n ∧ rd.bytes = some (r.bytes.getD ByteArray.empty) := by
  obtain ⟨hn, -, -, hm⟩ := hf
  have h1 : Record.nat x "n" = .ok r.n := by
    unfold Record.nat Record.fld
    rw [hx.n, roundObj_n]
    rfl
  have h2 : Record.isNull x "msg" = false := by
    unfold Record.isNull
    rw [hx.msg, roundObj_msg]
    rfl
  have h3 : Record.str x "msg" = .ok (hexOf (r.bytes.getD ByteArray.empty)) := by
    unfold Record.str Record.fld
    rw [hx.msg, roundObj_msg]
    rfl
  obtain ⟨s, b, hs, hb, hbytes⟩ := hm h2
  rw [h3] at hs
  cases hs
  rw [FlockComplete.JsonRT.unhex_hexOf] at hb
  cases hb
  rw [h1] at hn
  injection hn with hn
  exact ⟨hn.symm, hbytes⟩

/-- **The stream at a name is the one at its index**, where the names are distinct. -/
theorem find_name {l : List StreamRec} {name : String} {k : ℕ} (hk : k < l.length) (hn : l[k].name = name)
    (hnd : (l.map (·.name)).Nodup) : l.find? (·.name == name) = some l[k] := by
  rw [List.find?_eq_some_iff_getElem]
  refine ⟨by simp [hn], k, hk, rfl, fun j hj => ?_⟩
  have hne : l[j].name ≠ l[k].name := by
    intro e
    have e' : (l.map (·.name))[j]'(by simp; omega) = (l.map (·.name))[k]'(by simpa using hk) := by simpa using e
    have := (List.Nodup.getElem_inj_iff hnd).1 e'
    omega
  simp [← hn, hne]

end FlockDeploy
