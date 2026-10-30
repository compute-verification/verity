import Proofs.Flock.Claims.Recursion.VerdictRowDSide
import Proofs.Flock.Claims.Recursion.RecordCanon
import Proofs.Flock.Claims.Recursion.NextShaDec
import Proofs.Flock.Claims.Recursion.CallsAgree
import Proofs.Flock.Claims.DeployK

/-!
# Row D's coin tie: the position a `rec-open` piece reads

`DecAt.rowD_of_pieces` takes a `posAt` that reads each `rec-open` piece's position off the statement and the coins,
and asks it to be the one the verifier rows draw for the piece's query (`hpos`). This file fixes that reader and proves
the tie.

* `canonRounds`: rep `k`'s coins as the coins alone fix them: its schedule's round count, round `j` the first `n_j` (its
  schedule's) values of the coin at the rep's offset plus `j` among the coins after the link.
* `posAt`: at a statement that sets up, the verifier rows at table `table`'s shape, the extra claims at the first
  coin's link points and `canonRounds` (comb empty: the rows' `info` doesn't read it, `verifierRows_comb`), and the
  position they draw for the piece's query.
* `coins_of_decode`: the decoded session's rep stream `k` has `canonRounds`' coins. The record `decR` hands over
  (`recJ`) writes rep `k`'s rounds with its schedule's counts and `repLens` rounds per rep; the coin substitution
  (`coinSubJ`) erases every round's coins and fills round `j` of stream `name` with the first `n` values of the coin at
  `streamOffset + j`, and at distinct names `streamOffset` is the rounds of the reps before it.
* `DecAt.posAt_eq`, `DecAt.hpos`: at a decision of an inner statement read at the dense form, `posAt` reads the
  position the verifier rows at rep `r`'s values draw, in `runs_assembled`'s form.
-/

namespace RecDeploy.Rows

open Flock
open Lean (Json)
open FlockDeploy (Coin FinalP DecR decR repLens cutLens readPs readRep readRep_fits repCtx tabsOf pointsOf setupOf
  Ins recJ recStreams jAt coinSubJ eraseCoins mapStream mapKey nameOf nOf hexArr streamOffset shaSubJ
  arr_mapKey_mapArr arr_mapKey_mapArrIdx arr_mapKey_ne nat_mapKey_ne getObjValAs_mapKey_ne arr_mapKey_const
  fld_mapKey_ne fld_mapKey_self bind_ok takeWhile_ne_nodup roundsOf_of mapM_some_get RoundR StreamR framesOf)
open FlockComplete.JsonRT (recordObj streamObj streamList roundObj roundList getObjVal?_mkObj
  recordList_distinct streamList_distinct roundList_distinct f128s_map)
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Refine (shapeOf ebind_ok)
open FlockSoundness.Discharge.Recursion (HClass)

noncomputable section

/-! ## Reading the record's objects -/

theorem fld_of {j : Json} {k : String} {v : Json} (h : j.getObjVal? k = .ok v) : Record.fld j k = .ok v := by
  simp only [Record.fld, h]

theorem arr_of {j : Json} {k : String} {a : Array Json} (h : j.getObjVal? k = .ok (.arr a)) :
    Record.arr j k = .ok a := by
  simp only [Record.arr, fld_of h]
  rfl

theorem nat_of {j : Json} {k : String} {n : ℕ} (h : j.getObjVal? k = .ok (Lean.toJson n)) :
    Record.nat j k = .ok n := by
  simp only [Record.nat, fld_of h]
  rfl

theorem str_of {j : Json} {k s : String} (h : j.getObjVal? k = .ok (.str s)) : Record.str j k = .ok s := by
  simp only [Record.str, fld_of h]
  rfl

theorem str_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    Record.str (mapKey k g j) k' = Record.str j k' := by
  unfold Record.str
  rw [fld_mapKey_ne g j hk]

/-- An erased value is no list. -/
theorem arr_mapKey_null {k : String} {j : Json} {a : Array Json}
    (h : Record.arr (mapKey k (fun _ => .null) j) k = .ok a) : False := by
  unfold Record.arr at h
  cases hf : Record.fld (mapKey k (fun _ => .null) j) k with
  | error e => rw [hf] at h; cases h
  | ok v =>
    obtain ⟨v₀, -, rfl⟩ := fld_mapKey_self hf
    rw [hf] at h
    cases h

theorem arr_recordObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    Record.arr (recordObj spec h rootB points streams proofs) "streams" =
      .ok ((spec.repStreams.zip streams).map (streamObj spec h rootB)) :=
  arr_of (getObjVal?_mkObj (recordList_distinct spec h rootB points streams proofs)
    (by simp [FlockComplete.JsonRT.recordList]))

theorem arr_streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) :
    Record.arr (streamObj spec h rootB q) "rounds" =
      .ok ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!) :=
  arr_of (getObjVal?_mkObj (streamList_distinct spec h rootB q) (by simp [streamList]))

theorem str_streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : Record.str (streamObj spec h rootB q) "stream" = .ok q.1.2.2 :=
  str_of (getObjVal?_mkObj (streamList_distinct spec h rootB q) (by simp [streamList]))

theorem getObjValAs_streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : (streamObj spec h rootB q).getObjValAs? String "stream" = .ok q.1.2.2 := by
  show (Json.mkObj (streamList spec h rootB q)).getObjValAs? String "stream" = _
  unfold Json.getObjValAs? Json.getObjValD
  rw [getObjVal?_mkObj (streamList_distinct spec h rootB q) (k := "stream") (v := .str q.1.2.2)
    (by simp [streamList])]
  rfl

theorem nat_roundObj (spec : SessionSpec) (h : ByteArray → ByteArray) (k : ℕ) (r : Round) :
    Record.nat (roundObj spec h k r) "n" = .ok r.n :=
  nat_of (getObjVal?_mkObj (roundList_distinct spec h k r) (by simp [roundList]))

theorem nameOf_streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (ℕ × ℕ × String) × Array Round) : nameOf (streamObj spec h rootB q) = q.1.2.2 := by
  unfold nameOf
  simp only [getObjValAs_streamObj]
  rfl

theorem getObjValAs_mapStream (g : String → ℕ → Json → Json) (s : Json) :
    (mapStream g s).getObjValAs? String "stream" = s.getObjValAs? String "stream" :=
  getObjValAs_mapKey_ne _ _ (by decide)

theorem nameOf_mapStream (g : String → ℕ → Json → Json) (s : Json) : nameOf (mapStream g s) = nameOf s := by
  unfold nameOf
  rw [getObjValAs_mapStream]

theorem str_mapStream (g : String → ℕ → Json → Json) (s : Json) :
    Record.str (mapStream g s) "stream" = Record.str s "stream" :=
  str_mapKey_ne _ _ (by decide)

theorem arr_mapStream (g : String → ℕ → Json → Json) (s : Json) :
    Record.arr (mapStream g s) "rounds" = (·.mapIdx (g (nameOf s))) <$> Record.arr s "rounds" :=
  arr_mapKey_mapArrIdx "rounds" _ s

/-! ## The coin substitution's streams -/

/-- What `eraseCoins` does to a round. -/
def eraseG : String → ℕ → Json → Json := fun _ _ => mapKey "coins" fun _ => .null

/-- What `fillCoins` does to round `k` of stream `name`, at the streams' offsets `off`. -/
def fillG (mids : List (Claims.Msg × Coin)) (off : String → ℕ) (name : String) (k : ℕ) (r : Json) : Json :=
  match mids[off name + k]? with
  | some (_, c) => mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r
  | none => r

theorem arr_eraseCoins (j : Json) :
    Record.arr (eraseCoins j) "streams" = (·.map (mapStream eraseG)) <$> Record.arr j "streams" := by
  simp only [eraseCoins, FlockDeploy.mapRounds, FlockDeploy.setPoints]
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link")]
  rfl

theorem arr_coinSubJ (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) :
    Record.arr (coinSubJ spec c₁ mids j) "streams" =
      (·.map (mapStream (fillG mids (streamOffset spec (eraseCoins j))))) <$> Record.arr (eraseCoins j) "streams" := by
  simp only [coinSubJ, FlockDeploy.fillCoins, FlockDeploy.mapRounds, FlockDeploy.setPoints]
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link")]
  rfl

theorem arr_shaSubJ (spec : SessionSpec) (ds : Array ByteArray) (j : Json) :
    Record.arr (shaSubJ spec ds j) "streams" = Record.arr j "streams" :=
  arr_mapKey_ne _ _ (FlockDeploy.proofKey_ne_streams spec).symm

/-- **A substituted round reads the live coin**: where the decoder reads a round the substitution filled from an
honest round `rd₀`, there is a coin at its index, and the round counts `rd₀.n` and holds that coin's first `rd₀.n`
values. -/
theorem round_coins {mids : List (Claims.Msg × Coin)} {off : String → ℕ} {w : String} {j : ℕ} {spec : SessionSpec}
    {H : ByteArray → ByteArray} {rd₀ rd : Round}
    (h : RoundR (fillG mids off w j (eraseG w j (roundObj spec H j rd₀))) rd) :
    ∃ m c, mids[off w + j]? = some (m, c) ∧ rd.n = rd₀.n ∧ rd.coins = c.vals rd₀.n := by
  obtain ⟨hn, hc⟩ := h
  have hn0 : Record.nat (mapKey "coins" (fun _ => .null) (roundObj spec H j rd₀)) "n" = .ok rd₀.n := by
    rw [nat_mapKey_ne _ _ (by decide)]
    exact nat_roundObj spec H j rd₀
  simp only [fillG, eraseG] at hn hc
  cases hm : mids[off w + j]? with
  | none =>
    rw [hm] at hc
    obtain ⟨a, ha, -⟩ := bind_ok hc
    exact (arr_mapKey_null ha).elim
  | some mc =>
    obtain ⟨m, c⟩ := mc
    rw [hm] at hn hc
    dsimp only at hn hc
    rw [nat_mapKey_ne _ _ (by decide), hn0] at hn
    have hnr : nOf (mapKey "coins" (fun _ => .null) (roundObj spec H j rd₀)) = rd₀.n := by simp only [nOf, hn0]
    rw [hnr] at hc
    obtain ⟨a, ha, hc⟩ := bind_ok hc
    obtain rfl := arr_mapKey_const ha
    rw [f128s_map] at hc
    exact ⟨m, c, rfl, (Except.ok.inj hn).symm, (Except.ok.inj hc).symm⟩

/-! ## What the decoder reads of the streams -/

set_option maxHeartbeats 1000000 in
/-- **The decoder's streams at a parsed value**: one per stream record, each with its name and its rounds' coin
counts and coins (`StreamR`). -/
theorem decodeJWith_streams (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json} {sess : Session}
    (h : Record.decodeJWith hs spec j = .ok sess) :
    ∃ (recs : Array Json) (SF : Json → Flock.V StreamRec),
      Record.arr j "streams" = .ok recs ∧ recs.mapM SF = .ok sess.streams ∧ ∀ s x, SF s = .ok x → StreamR s x := by
  rw [Record.decodeJWith_eq_decodeJWithK] at h
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, by assumption, by assumption, fun s x hx => ?_⟩
    try dsimp only at hx
    obtain ⟨name, hname, hx⟩ := ebind_ok hx
    obtain ⟨rs, hrs, hx⟩ := ebind_ok hx
    obtain ⟨rounds, hrounds, hx⟩ := ebind_ok hx
    repeat' obtain ⟨_, _, hx⟩ := ebind_ok hx
    cases hx
    refine ⟨hname, rs, hrs, FlockDeploy.forall₂_of_mapIdxM (fun k r rd hr => ?_) hrounds⟩
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨n, hn, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨cs, hcs, hr⟩ := ebind_ok hr
    obtain ⟨coins, hcoins, hr⟩ := ebind_ok hr
    repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (split at hr))
    all_goals
      cases hr
      exact ⟨hn, by rw [show Record.arr r "coins" = .ok cs from hcs]; exact hcoins⟩

theorem forall₂_getElem {α β : Type} {R : α → β → Prop} {a : Array α} {b : Array β}
    (h : List.Forall₂ R a.toList b.toList) :
    a.size = b.size ∧ ∀ (t : ℕ) (h₁ : t < a.size) (h₂ : t < b.size), R a[t] b[t] := by
  obtain ⟨hl, hg⟩ := List.forall₂_iff_get.1 h
  refine ⟨by simpa using hl, fun t h₁ h₂ => ?_⟩
  have := hg t (by simpa using h₁) (by simpa using h₂)
  simpa using this

theorem find?_getElem {α : Type} {a : Array α} {p : α → Bool} {k : ℕ} (hk : k < a.size) (hp : p a[k] = true)
    (hne : ∀ (j : ℕ) (hj : j < k), p (a[j]'(by omega)) = false) : a.find? p = some a[k] := by
  rw [Array.find?_eq_some_iff_getElem]
  exact ⟨hp, k, hk, rfl, fun j hj => by simp [hne j hj]⟩

/-! ## The rounds per rep -/

theorem cutLens_getD_length {α : Type} : ∀ (ns : List ℕ) (l : List α), l.length = ns.sum → ∀ k,
    ((cutLens ns l).getD k []).length = ns.getD k 0
  | [], l, _, k => by simp [cutLens]
  | n :: ns, l, h, 0 => by
    simp only [cutLens, List.getD_cons_zero, List.length_take]
    simp only [List.sum_cons] at h
    omega
  | n :: ns, l, h, k + 1 => by
    simp only [cutLens, List.getD_cons_succ]
    exact cutLens_getD_length ns (l.drop n) (by simp only [List.length_drop, List.sum_cons] at h ⊢; omega) k

theorem cutLens_map_length {α : Type} : ∀ (L : List (List α)), cutLens (L.map List.length) L.flatten = L
  | [] => rfl
  | a :: L => by
    simp only [List.map_cons, List.flatten_cons, cutLens]
    rw [List.take_left, List.drop_left, cutLens_map_length L]

/-- **Rep `k`'s block of round messages has its schedule's rounds**: each rep's messages are its proof's framed
bytes, one per opened round (`FitsRep.rounds`), and the opened rounds are the `k`-th block of `repLens`. -/
theorem readPs_blocks {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) (k : ℕ) :
    ((cutLens (repLens spec sts) pf.2).getD k []).length = (repLens spec sts).getD k 0 := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    have hL : (rs.map Prod.snd).map List.length = repLens spec sts := by
      apply List.ext_getElem (by simp [repLens, hlen])
      intro t h₁ h₂
      have ht : t < rs.length := by simpa using h₁
      have hg := hget t (by rw [List.length_range]; omega) ht
      rw [List.getElem_range] at hg
      obtain ⟨T, hT, hms⟩ := readRep_fits hg
      rw [List.getElem_map, List.getElem_map, show rs[t].2 = _ from hms, framesOf, List.length_map, hT.rounds,
        cutLens_getD_length _ _ hc.2.symm, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h₂, Option.getD_some]
    dsimp only
    rw [← hL, cutLens_map_length, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
    simp only [List.getElem?_map]
    cases rs[k]? <;> rfl
  · rw [if_neg hc] at h
    cases h

theorem recStreams_size (spec : SessionSpec) (sts : Array Stmt) (ms : List Claims.Msg) (cs : List Coin) :
    (recStreams spec sts ms cs).size = 2 * sts.size := by
  simp [recStreams]

theorem recStreams_get {spec : SessionSpec} {sts : Array Stmt} {ms : List Claims.Msg} {cs : List Coin} {k : ℕ}
    (hk : k < (recStreams spec sts ms cs).size) :
    (recStreams spec sts ms cs)[k].size = ((cutLens (repLens spec sts) ms).getD k []).length ∧
      ∀ j < (recStreams spec sts ms cs)[k].size,
        ((recStreams spec sts ms cs)[k][j]!).n = ((RecDeploy.repSchedAt spec sts[k / 2]!).getD j default).1 := by
  refine ⟨by simp [recStreams], fun j hj => ?_⟩
  rw [getElem!_pos _ j hj]
  simp [recStreams]

/-! ## The coin tie at a decode -/

/-- **Rep `k`'s coins as the coins alone fix them**: its schedule's round count, round `j` the first `n_j` (its
schedule's) values of the coin at `k`'s offset plus `j` among the coins after the link `cs`. -/
def canonRounds (spec : SessionSpec) (sts : Array Stmt) (cs : List Coin) (k : ℕ) : Array (Array F128) :=
  (Array.range ((repLens spec sts).getD k 0)).map fun j =>
    (cs.getD (((repLens spec sts).take k).sum + j) default).vals
      ((RecDeploy.repSchedAt spec sts[k / 2]!).getD j default).1

/-- **The coin tie at a decode**: where the decoder reads the record `decR` hands over with its coins substituted, rep
`k`'s stream has `canonRounds`' coins, at rep streams of distinct names and round messages whose blocks have the
schedule's rounds. -/
theorem coins_of_decode (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {ms : List Claims.Msg} {cs : List Coin}
    {ps : Array Flock.Proof} {ds : Array ByteArray} {sess : Session}
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hlen : ∀ k, ((cutLens (repLens spec sts) ms).getD k []).length = (repLens spec sts).getD k 0)
    (hdec : Record.decodeJWith hs spec
      (coinSubJ spec c₁ (ms.zip cs) (shaSubJ spec ds (recJ spec sts link c₁ ms cs ps))) = .ok sess)
    {k : ℕ} (hk : k < 2 * sts.size) (hks : k < spec.repStreams.size) :
    ((sess.streams.find? (·.name == spec.repStreams[k].2.2)).getD default).rounds.map (·.coins) =
      canonRounds spec sts cs k := by
  have hSs := recStreams_size spec sts ms cs
  -- the record's streams, through the substitution
  have hQ : Record.arr (recJ spec sts link c₁ ms cs ps) "streams" =
      .ok ((spec.repStreams.zip (recStreams spec sts ms cs)).map
        (streamObj spec (Claims.hOf FlockDeploy.sha512) link.1.toArray)) :=
    arr_recordObj _ _ _ _ _ _
  have hJ2 : Record.arr (eraseCoins (shaSubJ spec ds (recJ spec sts link c₁ ms cs ps))) "streams" =
      .ok (((spec.repStreams.zip (recStreams spec sts ms cs)).map
        (streamObj spec (Claims.hOf FlockDeploy.sha512) link.1.toArray)).map (mapStream eraseG)) := by
    rw [arr_eraseCoins, arr_shaSubJ, hQ]
    rfl
  have hJ : Record.arr (coinSubJ spec c₁ (ms.zip cs) (shaSubJ spec ds (recJ spec sts link c₁ ms cs ps))) "streams" =
      .ok ((((spec.repStreams.zip (recStreams spec sts ms cs)).map
        (streamObj spec (Claims.hOf FlockDeploy.sha512) link.1.toArray)).map (mapStream eraseG)).map
          (mapStream (fillG (ms.zip cs)
            (streamOffset spec (eraseCoins (shaSubJ spec ds (recJ spec sts link c₁ ms cs ps))))))) := by
    rw [arr_coinSubJ, hJ2]
    rfl
  generalize hoffd : streamOffset spec (eraseCoins (shaSubJ spec ds (recJ spec sts link c₁ ms cs ps))) = off at hJ
  generalize hJ2d : eraseCoins (shaSubJ spec ds (recJ spec sts link c₁ ms cs ps)) = J₂ at hJ2 hoffd
  generalize hSd : recStreams spec sts ms cs = S at hSs hQ hJ2 hJ
  generalize hrecs₀ : (spec.repStreams.zip S).map (streamObj spec (Claims.hOf FlockDeploy.sha512) link.1.toArray) =
    recs₀ at hJ2 hJ
  have hsz₀ : recs₀.size = min spec.repStreams.size S.size := by rw [← hrecs₀]; simp
  have hrec₀ : ∀ (t : ℕ) (ht : t < recs₀.size), recs₀[t] = streamObj spec (Claims.hOf FlockDeploy.sha512)
      link.1.toArray (spec.repStreams[t]'(by omega), S[t]'(by omega)) := by
    intro t ht
    subst hrecs₀
    simp
  -- the decoder's streams
  obtain ⟨recs', SF, harr, hmap, hR⟩ := decodeJWith_streams hs hdec
  rw [hJ] at harr
  cases harr
  obtain ⟨hssz, hF⟩ := forall₂_getElem (FlockDeploy.forall₂_of_array_mapM hR hmap)
  simp only [Array.size_map] at hssz hF
  have hk₀ : k < recs₀.size := by omega
  have hname : ∀ (t : ℕ) (ht : t < sess.streams.size),
      (sess.streams[t]).name = (spec.repStreams[t]'(by omega)).2.2 := by
    intro t ht
    have h1 := (hF t (by omega) ht).1
    simp only [Array.getElem_map, str_mapStream, hrec₀ t (by omega), str_streamObj] at h1
    exact (Except.ok.inj h1).symm
  have hne : ∀ (t t' : ℕ) (ht : t < spec.repStreams.size) (ht' : t' < spec.repStreams.size), t ≠ t' →
      spec.repStreams[t].2.2 ≠ spec.repStreams[t'].2.2 := by
    intro t t' ht ht' htt e
    apply htt
    have e' : (spec.repStreams.toList.map (·.2.2))[t]'(by simpa using ht) =
        (spec.repStreams.toList.map (·.2.2))[t']'(by simpa using ht') := by simpa using e
    exact (List.Nodup.getElem_inj_iff hnd).1 e'
  have hfind : sess.streams.find? (·.name == spec.repStreams[k].2.2) = some (sess.streams[k]'(by omega)) := by
    refine find?_getElem (by omega) (by rw [hname k (by omega)]; simp) fun j hj => ?_
    rw [hname j (by omega)]
    simpa using hne j k (by omega) hks (by omega)
  rw [hfind, Option.getD_some]
  -- rep `k`'s rounds
  obtain ⟨-, rs, hrs, hRd⟩ := hF k (by omega) (by omega)
  simp only [Array.getElem_map, arr_mapStream, nameOf_mapStream, hrec₀ k hk₀, nameOf_streamObj,
    arr_streamObj] at hrs
  have hrs' := (Except.ok.inj hrs).symm
  obtain ⟨hrsz, hRj⟩ := forall₂_getElem hRd
  have hSk : k < S.size := by omega
  have hSksz : S[k].size = ((cutLens (repLens spec sts) ms).getD k []).length := by
    subst hSd
    exact (recStreams_get hSk).1
  have hSn : ∀ j < S[k].size, (S[k][j]!).n = ((RecDeploy.repSchedAt spec sts[k / 2]!).getD j default).1 := by
    subst hSd
    exact (recStreams_get hSk).2
  have hrsS : rs.size = S[k].size := by rw [hrs']; simp
  -- the offset of rep `k`'s rounds
  have hoffk : off spec.repStreams[k].2.2 = ((repLens spec sts).take k).sum := by
    rw [← hoffd]
    unfold streamOffset
    have hnames : spec.repStreams[k].2.2 = (spec.repStreams.toList.map (·.2.2))[k]'(by simpa using hks) := by simp
    rw [hnames, takeWhile_ne_nodup _ hnd k (by simpa using hks)]
    apply congrArg List.sum
    apply List.ext_getElem (by simp [repLens]; omega)
    intro t h₁ h₂
    have ht : t < k := by simp at h₁; omega
    simp only [List.getElem_map, List.getElem_take, Array.getElem_toList]
    have hfind₁ : (recs₀.map (mapStream eraseG)).find?
        (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[t].2.2) =
          some ((recs₀.map (mapStream eraseG))[t]'(by simp; omega)) := by
      refine find?_getElem (by simp; omega) ?_ fun j hj => ?_
      · simp only [Array.getElem_map, getObjValAs_mapStream, hrec₀ t (by omega), getObjValAs_streamObj]
        simp [Except.toOption]
      · simp only [Array.getElem_map, getObjValAs_mapStream, hrec₀ j (by omega), getObjValAs_streamObj]
        simpa [Except.toOption] using hne j t (by omega) (by omega) (by omega)
    have hro := roundsOf_of hJ2 hfind₁ (by
      simp only [Array.getElem_map, arr_mapStream, hrec₀ t (by omega), arr_streamObj]
      rfl)
    rw [hro, Array.size_mapIdx, Array.size_map, Array.size_range]
    obtain ⟨hSt, -⟩ := recStreams_get (spec := spec) (sts := sts) (ms := ms) (cs := cs) (k := t)
      (by rw [hSd]; omega)
    simp only [hSd] at hSt
    rw [hSt, hlen t, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem (by simp at h₂; omega), Option.getD_some]
  -- the rounds, one by one
  apply Array.ext
  · rw [Array.size_map, ← hrsz, hrsS, hSksz, hlen k]
    simp [canonRounds]
  · intro j h₁ h₂
    have hj : j < rs.size := by rw [hrsz]; simpa using h₁
    have hjS : j < S[k].size := by omega
    have hr := hRj j hj (by simpa using h₁)
    simp only [hrs', Array.getElem_mapIdx, Array.getElem_map, Array.getElem_range] at hr
    obtain ⟨m, c, hmc, -, hcoins⟩ := round_coins hr
    rw [hoffk] at hmc
    obtain ⟨-, hc⟩ := List.getElem?_zip_eq_some.1 hmc
    simp only [Array.getElem_map, hcoins, canonRounds, Array.getElem_range]
    rw [hSn j hjS]
    simp only [List.getD_eq_getElem?_getD, hc, Option.getD_some]

/-! ## The verifier rows don't read the comb -/

open Flock.RecAlgebra in
/-- **The verifier rows' `info` and `kappa` don't read the comb**: it is only appended to `v`. -/
theorem verifierRows_comb (sh : Shape) (rounds : Array (Array F128)) (extras : Array Claim) (comb : Array F128) :
    verifierRows sh rounds extras comb =
      (fun t => (t.1 ++ comb, t.2.1, t.2.2)) <$> verifierRows sh rounds extras #[] := by
  unfold verifierRows
  cases residualLins sh (CoinSource.ofRounds rounds) extras with
  | error e => rfl
  | ok p =>
    obtain ⟨lins, info⟩ := p
    dsimp only [bind, Except.bind]
    cases build sh lins with
    | error e => rfl
    | ok b =>
      dsimp only
      split <;> rfl

/-! ## The reader -/

/-- **The position a `rec-open` piece reads off the statement and the coins**: at a statement that sets up, the
verifier rows at table `table`'s shape (its `fast100` schedule and its extra claims at the first coin's link points),
those extra claims, rep `k = 2 table + rep`'s `canonRounds` at the coins after the link, and no comb (`info` doesn't
read it, `verifierRows_comb`); the position they draw for query `query` of level `level`. -/
def posAt (x : Claims.Msg) (cs : List Coin) (q : OpenPiece) : ℕ :=
  match setupOf false x with
  | some v =>
    match verifierRows
        (shapeOf (tabsOf v.2.2)[q.table]! ((fast100 (tabsOf v.2.2)[q.table]!.m).getD default)
          (((tabsOf v.2.2)[q.table]!.extra (pointsOf v.2.1 (cs.headD default))).toOption.getD #[]).size)
        (canonRounds v.2.1 v.2.2 cs.tail.dropLast (2 * q.table + q.rep))
        (((tabsOf v.2.2)[q.table]!.extra (pointsOf v.2.1 (cs.headD default))).toOption.getD #[]) #[] with
    | .ok (_, info, _) => (info.lg.conss[q.level]!).pos[q.query]!
    | .error _ => 0
  | none => 0

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ)

/-- **The rep streams' names are distinct** at a decision of an inner statement read at the dense form: the read
carries no public inputs (`dense_inner`), so its tables are distinct (`KS.tables_nodup`). -/
theorem names_nodup {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) :
    (D.spec.repStreams.toList.map (·.2.2)).Nodup := by
  obtain ⟨s, spec, st, sch, hset, -, hpi, -⟩ := dense_inner hst hf
  rw [← hx] at hset
  obtain ⟨hsp, -⟩ := D.setup_eq
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hset] at hv
  cases hv
  have hdspec : D.d.spec = spec := congrArg (·.spec) hd
  obtain ⟨-, hs⟩ := FlockDeploy.setupOf_some hset
  rw [hsp, hdspec]
  exact FlockDeploy.KS.repNames_nodup (FlockDeploy.KS.tables_nodup hs hpi)

/-- **A piece reads its query's drawn position**: at a decision where the rep streams' names are distinct, a piece of
table `i` and rep `r` reads the position the verifier rows at rep `r`'s values draw for its query. -/
theorem posAt_eq (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (r : Fin 2) {v : Array F128} {info : RecAlgebra.Info RecAlgebra.Lin}
    {kappa : Array (Array (Array F128))}
    (hv : verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
      .ok (v, info, kappa))
    (q : OpenPiece) (ht : q.table = i) (hr : q.rep = r.val) :
    posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]! := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hv₀, -, hc₁, -, -, -, hpf, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hsp' : D.spec = spec := by rw [hsp, hd]
  have hsts' : D.sts = sts := by rw [hsts, hd]
  have hdec := D.decode
  rw [hd, hsp', hsts'] at hdec
  simp only [jAt] at hdec
  have hpts := (FlockDeploy.decodeJWith_sub _ hdec).1
  have hhead : cs.headD default = c₁ := by
    cases cs with
    | nil => cases hc₁
    | cons c cs => exact Option.some.inj hc₁
  have hk : 2 * i + r.val < 2 * sts.size := by rw [← hsts']; omega
  rw [hsp'] at hnd hti
  obtain ⟨hks, e⟩ := Claims.Bridge.Custody.repStreams_getElem spec hti r.isLt
  have hrounds : (D.rep cls acc i r).rounds = canonRounds spec sts cs.tail.dropLast (2 * i + r.val) := by
    show ((D.files cls i).stream r).rounds.map (·.coins) = _
    have hs : (D.files cls i).stream r =
        (D.sess.streams.find? (·.name == spec.repStreams[2 * i + r.val].2.2)).getD default := by
      rw [e]
      show (D.sess.streams.find? fun y => y.name == toString D.spec.tables[i]! ++ toString "/rep" ++
        toString (r : ℕ)).getD default = _
      rw [hsp']
    rw [hs]
    exact coins_of_decode _ hnd (readPs_blocks hpf) hdec hk hks
  have hext : (D.rep cls acc i r).extras =
      ((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[] := by
    show ((tabsOf D.sts)[i]!.extra D.sess.points).toOption.getD #[] = _
    rw [hsts', hpts, hhead]
  have hsh : D.sh cls i = shapeOf (tabsOf sts)[i]! ((fast100 (tabsOf sts)[i]!.m).getD default)
      (((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]).size := by
    show shapeOf (tabsOf D.sts)[i]! ((fast100 (tabsOf D.sts)[i]!.m).getD default)
      (((tabsOf D.sts)[i]!.extra D.sess.points).toOption.getD #[]).size = _
    rw [hsts', hpts, hhead]
  rw [hsh, hrounds, hext, verifierRows_comb] at hv
  unfold posAt
  rw [hv₀]
  dsimp only
  rw [ht, hr]
  cases h0 : verifierRows (shapeOf (tabsOf sts)[i]! ((fast100 (tabsOf sts)[i]!.m).getD default)
      (((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]).size)
      (canonRounds spec sts cs.tail.dropLast (2 * i + r.val))
      (((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]) #[] with
  | error e => rw [h0] at hv; cases hv
  | ok t =>
    rw [h0] at hv
    obtain ⟨t₁, t₂, t₃⟩ := t
    simp only [Functor.map, Except.map, Except.ok.injEq, Prod.mk.injEq] at hv
    obtain ⟨-, rfl, -⟩ := hv
    rfl

/-- **`hpos`, the coin tie, in `runs_assembled`'s form**: at a decision of an inner statement read at the dense form
and a table `i` of the statement, at the verifier rows at rep `r`'s values, every piece of rep `r` of table `i` reads
the position the rows draw for its query through `posAt`. -/
theorem hpos {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (pieces : List OpenPiece) :
    ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]! :=
  fun r _ _ _ hv q _ ht hr => D.posAt_eq cls acc i (D.names_nodup hst hf hx) hi hti r hv q ht hr

end DecAt

end

end RecDeploy.Rows
