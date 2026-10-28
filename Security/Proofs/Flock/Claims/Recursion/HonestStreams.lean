import Proofs.Flock.Claims.Recursion.HonestVerdictJson
import Proofs.Flock.Claims.Recursion.HonestVerdictR
import Proofs.Flock.Claims.Recursion.RepSqCounts
import Proofs.Flock.Claims.KS.RoundsN
import Proofs.Flock.Claims.Complete.Json.Goals
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Soundness.Discharge.Exec.FramesRep
import Flock.With.Deployed

/-!
# The reader's streams at the honest run are the honest record's

The reader's record `recJ` rebuilds each rep stream from the round messages: rep `k` takes its block of the framed
proof bytes (`recFrames`), each round counting its schedule's coins (`RecDeploy.repSchedAt`) and holding the first
`n` values of its coin. At an honest single-table run whose goals hold at SHA-512, these are the recorded streams'
rounds as `liveOf` reads them (`recStreams_eq`), and the link the reader takes from round 0 is the record's `root_B`
(`recLink_eq`):

* **Counts** (`schedN_eq_repNs`): `RecVStar.repRounds`' coin counts (`schedN`) are the counts a verified rep's rounds
  issued (`repNs`, `verifyRepWith_readsN`), at every `fast100` schedule.
* **Bytes** (`frames_emitted`): the goals' zerocheck, lincheck, ring switch and Ligerito runs, at SHA-512 the plain
  ones, emit the rep's frames at the ring switch's batched value, one chunk per recorded round, each retained.
* **Coins** (`hs_coins`): the sends after the link are the recorded rounds (`Recs`), so the deployed verifier's coin
  at a round's offset opens to the round's coins.
* **Per rep** (`rep_facts`) and **the streams** (`recStreams_eq`): the three, at each rep's stream as the record
  decodes it.
-/

namespace FlockDeploy

open Flock FlockComplete
open FlockSoundness.Refine (Frame encs Retains AllRetained init_enc DimSeam LevelSeam)
open FlockSoundness.Discharge.Exec (Emitted chunks repFrames zcFrames lcFrames rsFrames ligFrames emitted_absorb
  zerocheck_emitted lincheck_emitted ringSwitch_emitted ligerito_frames)

/-! ## The schedule's counts are a verified rep's -/

theorem loop_passN (sch : Flock.Schedule)
    (hcs : ∀ i < sch.levels.size - 1, (sch.levels.getD (i + 1) default).cols + 3 = (sch.levels.getD i default).cols)
    (hq : ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q) :
    ∀ k a x, a + k = sch.levels.size - 1 → (0 < k → x = (sch.levels.getD a default).cols + 1) →
      loopNs (levelNs sch) (levelDim sch) a k x = (List.range' a k).flatMap (passN sch (sch.levels.size - 1))
  | 0, _, _, _, _ => rfl
  | k + 1, a, x, hk, hx => by
    have hxa := hx (by omega)
    simp only [loopNs]
    rw [List.range'_succ, List.flatMap_cons]
    by_cases hi : a + 1 < sch.levels.size - 1
    · have hc := hcs a (by omega)
      have e : x - 4 + 1 = (sch.levels.getD (a + 1) default).cols + 1 := by omega
      have hq1 := hq (a + 1) (by omega)
      rw [show levelDim sch a x = x - 4 + 1 by simp [levelDim, hi],
        loop_passN sch hcs hq k (a + 1) (x - 4 + 1) (by omega) (fun _ => e)]
      congr 1
      rw [levelNs, if_pos hi, passN, if_pos hi, lvN, qNs, e, bitLength_pred hq1]
      rfl
    · obtain rfl : k = 0 := by omega
      have hq1 := hq (sch.levels.size - 1) (by omega)
      simp only [loopNs, List.range'_zero, List.flatMap_nil, List.append_nil]
      rw [levelNs, if_neg hi, passN, if_neg hi, qNs, bitLength_pred hq1]
      rfl

/-- **`RecVStar.repRounds`' counts are a verified rep's**, at a `fast100` schedule. -/
theorem schedN_eq_repNs {m kLog claims : ℕ} {sch : Flock.Schedule} (hs : fast100 m = some sch) :
    schedN sch m kLog claims = repNs m kLog claims sch := by
  obtain ⟨⟨hn7, h2, hl0, hc0, hl4, hcs, hext⟩, hsm⟩ := FlockSoundness.Refine.fast100_dims _ _ hs
  have hm22 := FlockSoundness.Refine.exec_fast100_ge hs
  have hq := fast100_q_two hs
  have hq0 := hq 0 (by omega)
  have hloop := loop_passN sch hcs hq (sch.levels.size - 1) 0 (sch.n - sch.k0 + 1) (by omega) (fun _ => by omega)
  have h5 : List.replicate (m - 5) 1 = List.replicate (m - 6) 1 ++ [1] := by
    rw [show m - 5 = m - 6 + 1 by omega, List.replicate_succ']
  unfold schedN repNs zcNs lcNs rsNs ligNs
  rw [h5, hloop]
  simp only [lvN, qNs, bitLength_pred hq0, List.append_assoc, List.cons_append, List.nil_append]

/-! ## Names and schemes -/

theorem named_self {ms : MerkleScheme} (h : MerkleScheme.find? ms.id = some ms) : ms.Named :=
  Flock.With.named_of_find? h

/-- **Every registered scheme is named.** -/
theorem named_of_all {ms : MerkleScheme} (h : ms ∈ MerkleScheme.all) : ms.Named := by
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl <;> exact named_self rfl

theorem nodup_names {spec : SessionSpec} (h : NamesDistinct spec) : (spec.repStreams.toList.map (·.2.2)).Nodup := by
  unfold List.Nodup
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij heq
  simp only [List.getElem_map, Array.getElem_toList] at heq
  have := h i j (by simpa using hi) (by simpa using hj) heq
  omega

/-- At one table, rep stream `k` is table `k / 2`'s rep `k % 2`, by the name the verifier finds it under. -/
theorem repName_one {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) {k : ℕ}
    (hk : k < spec.repStreams.size) :
    spec.repStreams[k].2.2 = toString spec.tables[k / 2]! ++ toString "/rep" ++ toString (k % 2) := by
  have hR := JsonRT.repStreams_one ht
  have hk2 : k < 2 := by have h := hk; rw [hR] at h; simpa using h
  interval_cases k <;> simp [hR, ht]

/-- **The decoded record finds each rep stream under its name**, at distinct names. -/
theorem stream_find {spec : SessionSpec} {h : ByteArray → ByteArray} {rb : Array ByteArray}
    {HS : Array (Array Flock.Round)} (hss : HS.size = spec.repStreams.size)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) {k : ℕ} (hk : k < spec.repStreams.size) :
    ((spec.repStreams.zip HS).map (JsonRT.streamRec spec h rb)).find? (fun x => x.name == spec.repStreams[k].2.2) =
      some (JsonRT.streamRec spec h rb (spec.repStreams[k], HS[k]'(hss ▸ hk))) := by
  set A := (spec.repStreams.zip HS).map (JsonRT.streamRec spec h rb) with hAdef
  have hAs : A.size = spec.repStreams.size := by simp [hAdef, hss]
  have hA : ∀ j (hj : j < A.size),
      A[j] = JsonRT.streamRec spec h rb (spec.repStreams[j]'(by omega), HS[j]'(by omega)) := by
    intro j hj
    simp [hAdef, Array.getElem_zip]
  have hnames : A.toList.map (·.name) = spec.repStreams.toList.map (·.2.2) := by
    apply List.ext_getElem (by simp [hAs])
    intro j h1 h2
    simp only [List.getElem_map, Array.getElem_toList]
    rw [hA j (by simpa using h1)]
    rfl
  have hnd' : (A.toList.map (·.name)).Nodup := by rw [hnames]; exact hnd
  have hk' : k < A.toList.length := by rw [Array.length_toList, hAs]; exact hk
  have hf := FlockComplete.ZkTables.find?_name (·.name) A.toList hnd' k hk'
  simp only [Array.getElem_toList, Array.find?_toList] at hf
  have hname : (A[k]'(by omega)).name = spec.repStreams[k].2.2 := by rw [hA]; rfl
  rw [hname, hA k (by omega)] at hf
  exact hf

/-! ## The record's rounds, as `liveOf` reads them -/

theorem streamObj_live (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (q : ℕ × ℕ × String) (rs : Array Flock.Round) :
    JsonRT.streamObj spec h rb (q, rs.map JsonRT.liveRound) = JsonRT.streamObj spec h rb (q, rs) := by
  have e : ((Array.range (rs.map JsonRT.liveRound).size).map fun k =>
        JsonRT.roundObj spec h k (rs.map JsonRT.liveRound)[k]!) =
      (Array.range rs.size).map fun k => JsonRT.roundObj spec h k rs[k]! := by
    apply Array.ext (by simp)
    intro j h1 h2
    have hj : j < rs.size := by simpa using h2
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos (rs.map JsonRT.liveRound) j (by simpa using hj), getElem!_pos rs j hj, Array.getElem_map]
    rfl
  unfold JsonRT.streamObj JsonRT.streamList
  dsimp only
  rw [e]

/-- **The honest record is unchanged by reading its rounds live.** -/
theorem recordObj_liveRound (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (pts : Array (Array F128)) (S : Array (Array Flock.Round)) (proofs : Array ByteArray) :
    JsonRT.recordObj spec h rb pts (S.map (·.map JsonRT.liveRound)) proofs = JsonRT.recordObj spec h rb pts S proofs := by
  have e : (spec.repStreams.zip (S.map (·.map JsonRT.liveRound))).map (JsonRT.streamObj spec h rb) =
      (spec.repStreams.zip S).map (JsonRT.streamObj spec h rb) := by
    apply Array.ext (by simp)
    intro i h1 h2
    simp only [Array.getElem_map, Array.getElem_zip]
    exact streamObj_live spec h rb _ _
  unfold JsonRT.recordObj JsonRT.recordList
  rw [e]

/-! ## The honest run's reps -/

section

variable {rel : Readable.Parsed → Readable.Relation}

theorem run_sch {H : Claims.Msg → Claims.Msg} (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) :
    o.sch i = (fast100 o.tabs[i]!.m).getD default := by
  unfold Run.sch schedsOf
  simp [hi]

theorem run_tabs_size {H : Claims.Msg → Claims.Msg} (o : Run H rel) : o.tabs.size = o.sts.size := by
  unfold Run.tabs tabsOf
  rw [Array.size_map]

/-- **Rep `k`'s decoded proof is the session's file's**, where the record's points are the first coin's. -/
theorem run_proof_rec {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) (o : Run H rel) {k : ℕ}
    (hex : (tabsOf o.sts)[k / 2]!.extra (pointsOf o.spec o.recC₁) = .ok (o.extras (k / 2)))
    (hk : k < 2 * o.sts.size) : o.proof (k / 2) (k % 2) = o.recPs[k]! := by
  obtain ⟨-, hts⟩ := setup_tables o.setup
  have hdec : decodeProof (shapeAt (tabsOf o.sts) (pointsOf o.spec o.recC₁) k) o.proofs[k]! = .ok o.recPs[k]! :=
    ((sessionS_reps h64 (rfl : o.resp 0 = o.recC₁.bytes) hts.symm o.recTabsOk o.ran [] []).2 k hk).1
  have hi : k / 2 < o.tabs.size := by rw [run_tabs_size]; omega
  unfold Run.proof Run.bytes
  rw [show 2 * (k / 2) + k % 2 = k by omega]
  have hsh : FlockComplete.shapeOf o.tabs[k / 2]! (o.sch (k / 2)) (o.extras (k / 2)).size =
      shapeAt (tabsOf o.sts) (pointsOf o.spec o.recC₁) k := by
    rw [run_sch o hi]
    simp only [shapeAt, hex]
    rfl
  rw [hsh, hdec]
  rfl

/-- **A rep's recorded rounds retain its frames** at the ring switch's batched value: the goals' runs, at SHA-512 the
plain ones, emit `repFrames`, one chunk per round. -/
theorem frames_emitted (g : Goals sha512 rel) (o : Run sha512 rel) {i r : ℕ} (hi : i < o.tabs.size) (hr : r < 2)
    (hN : o.tabs[i]!.merkle.Named) (hf : fast100 o.tabs[i]!.m = some (o.sch i))
    (hret : AllRetained (o.stream i r).rounds) :
    (chunks (repFrames o.tabs[i]! (o.sch i) (o.spec.domain o.spec.tables[i]! r).toUTF8 (2 + (o.extras i).size)
        (o.proof i r) (o.sw i r).1.1)).1.length = (o.stream i r).rounds.size ∧
      ∀ j < (o.stream i r).rounds.size, Retains ((o.stream i r).rounds[j]!)
        (encs ((chunks (repFrames o.tabs[i]! (o.sch i) (o.spec.domain o.spec.tables[i]! r).toUTF8
          (2 + (o.extras i).size) (o.proof i r) (o.sw i r).1.1)).1.getD j [])) := by
  obtain ⟨hdim, -⟩ := FlockSoundness.Refine.fast100_dims _ _ hf
  have hm : 14 ≤ o.tabs[i]!.m := by have := FlockSoundness.Refine.exec_fast100_ge hf; omega
  have hlv : ∀ j < (o.sch i).levels.size, LevelSeam ((o.sch i).levels.getD j default) := fun j hj =>
    FlockSoundness.Refine.fast100_level _ _ hf _ (by
      unfold Array.getD; rw [dite_eq_left hj]; exact Array.getElem_mem hj)
  have hs : o.hs = Flock.With.std := by unfold Run.hs; rw [hOf_sha512]
  have h₁ := g.zerocheck o i hi r hr
  have h₂ := g.lincheck o i hi r hr
  have h₃ := g.ringSwitch o i hi r hr
  have h₄ := g.ligerito o i hi r hr
  have hc := g.consumed o i hi r hr
  rw [hs, Flock.With.bindAndZerocheckWith_std] at h₁
  rw [hs, Flock.With.lincheckWith_std] at h₂
  rw [hs, Flock.With.ringSwitchWith_std] at h₃
  rw [hs, Flock.With.ligeritoWith_std _ hN] at h₄
  have e₀ : Emitted ⟨(o.stream i r).rounds, 0, ByteArray.empty⟩ (o.t₀ i r)
      [.dom (o.spec.domain o.spec.tables[i]! r).toUTF8] :=
    emitted_absorb rfl rfl rfl (by unfold Run.t₀; rw [init_enc]; rfl)
  have r₀ : AllRetained (o.t₀ i r).rounds := hret
  have e₁ := zerocheck_emitted o.tabs[i]!.m hm o.tabs[i]!.digest (FlockComplete.flat (o.proof i r).cap)
    (o.proof i r).zerocheck (o.t₀ i r) (o.zc i r).2 (o.zc i r).1 r₀ h₁
  have e₂ := lincheck_emitted o.tabs[i]!.fold (o.zc i r).1 (o.proof i r).lincheck (o.zc i r).2 (o.lc i r).2
    (o.lc i r).1 (e₁.retained r₀) h₂
  have hsz : (abcClaims o.tabs[i]!.m o.tabs[i]!.kLog (o.zc i r).1 (o.lc i r).1 ++ o.extras i).size =
      2 + (o.extras i).size := by simp [abcClaims]
  have e₃ := ringSwitch_emitted o.tabs[i]!.m _ (o.proof i r).ringSwitches (by rw [hsz]; omega)
    (o.lc i r).2 (o.sw i r).2 (o.sw i r).1 (e₂.retained (e₁.retained r₀)) h₃
  rw [hsz] at e₃
  have e₄ := ligerito_frames o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.proof i r).cap (o.proof i r).ligerito
    (basis (o.sw i r).1.2) hdim hlv (t := (o.sw i r).2) (t' := (o.lg i r).2)
    (e₃.retained (e₂.retained (e₁.retained r₀))) h₄
  have e := e₀.trans (e₁.trans (e₂.trans (e₃.trans e₄)))
  have hfr : [Frame.dom (o.spec.domain o.spec.tables[i]! r).toUTF8] ++
      (zcFrames o.tabs[i]!.m o.tabs[i]!.digest (FlockComplete.flat (o.proof i r).cap) (o.proof i r).zerocheck ++
        (lcFrames o.tabs[i]!.fold.kLog (o.proof i r).lincheck ++
          (rsFrames (2 + (o.extras i).size) (o.proof i r).ringSwitches ++
            ligFrames (o.sch i) (o.sw i r).1.1 (o.proof i r).cap (o.proof i r).ligerito))) =
      repFrames o.tabs[i]! (o.sch i) (o.spec.domain o.spec.tables[i]! r).toUTF8 (2 + (o.extras i).size)
        (o.proof i r) (o.sw i r).1.1 := by
    simp [repFrames, FlockComplete.flat, List.append_assoc]
  rw [hfr] at e
  obtain ⟨-, hk, hR, -⟩ := e
  have hc' : (o.lg i r).2.k = (o.stream i r).rounds.size := by simpa using hc
  refine ⟨by rw [← hc', hk]; simp, fun j hj => ?_⟩
  have hj' : j < (chunks (repFrames o.tabs[i]! (o.sch i) (o.spec.domain o.spec.tables[i]! r).toUTF8
      (2 + (o.extras i).size) (o.proof i r) (o.sw i r).1.1)).1.length := by
    rw [← hc', hk] at hj; simpa using hj
  have := hR j hj'
  simpa using this

/-! ## Offsets and coins -/

theorem arrays_offset {α : Type} : ∀ (L : List (Array α)) (i : ℕ) (hi : i < L.length) (k : ℕ) (hk : k < L[i].size),
    (L.flatMap Array.toList)[((L.take i).map Array.size).sum + k]? = some L[i][k]
  | [], _, hi, _, _ => absurd hi (by simp)
  | a :: L, 0, _, k, hk => by
    simp only [List.flatMap_cons, List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add,
      List.getElem_cons_zero]
    rw [List.getElem?_append_left (by simpa using hk)]
    simp
  | a :: L, i + 1, hi, k, hk => by
    simp only [List.flatMap_cons, List.take_succ_cons, List.map_cons, List.sum_cons, List.getElem_cons_succ]
    rw [List.getElem?_append_right (by simp only [Array.length_toList]; omega)]
    have := arrays_offset L i (by simpa using hi) k (by simpa using hk)
    rw [show a.size + ((L.take i).map Array.size).sum + k - a.toList.length =
      ((L.take i).map Array.size).sum + k by simp only [Array.length_toList]; omega]
    exact this

/-- **The deployed verifier's coin at a recorded round's offset opens to the round's coins**: the sends after the link
are the recorded rounds, each answered by the run's coin. -/
theorem hs_coins {H : Claims.Msg → Claims.Msg} (o : Run H rel) {cs : List Coin} (hcs : o.CoinsAt cs)
    {HS : Array (Array Flock.Round)} {bs : List Claims.Msg} (hrl : repLens o.spec o.sts = HS.toList.map Array.size)
    (hrecs : JsonRT.Recs o.resp 1 (HS.toList.flatMap Array.toList) bs) {q : ℕ} (hq : q < HS.size) {j : ℕ}
    (hj : j < HS[q].size) :
    (cs.tail.dropLast.getD (((HS.toList.take q).map Array.size).sum + j) default).vals HS[q][j].n = HS[q][j].coins := by
  have hoff := arrays_offset HS.toList q (by simpa using hq) j (by simpa using hj)
  have hlenF : (HS.toList.flatMap Array.toList).length = (HS.toList.map Array.size).sum := by
    simp [List.length_flatMap]
  obtain ⟨hfl, he0⟩ := List.getElem?_eq_some_iff.1 hoff
  have he : (HS.toList.flatMap Array.toList)[((HS.toList.take q).map Array.size).sum + j] = HS[q][j] := by
    simpa using he0
  obtain ⟨b, hb⟩ := FlockComplete.ZkTables.Recs.getElem hrecs _ hfl
  rw [he] at hb
  obtain ⟨c, -, -, -, hco⟩ := hb
  have h2 := hcs.2
  rw [hrl] at h2
  rw [getD_tail_dropLast default cs (((HS.toList.take q).map Array.size).sum + j) (by omega),
    hcs.1 (((HS.toList.take q).map Array.size).sum + j + 1) (by omega), hco]
  unfold Coin.vals
  congr 1
  funext x
  rw [show 1 + (((HS.toList.take q).map Array.size).sum + j) = ((HS.toList.take q).map Array.size).sum + j + 1 by
    omega]
  exact (FlockComplete.JsonRT.coinOf_bytes (Claims.coinSeq o.r (((HS.toList.take q).map Array.size).sum + j + 1))
    x).symm

end

/-! ## Each rep of the honest single-table run -/

/-- **Rep `k` of the honest run**: its schedule has a round per recorded round, each counting the recorded round's
coins, and its framed proof bytes are a message per recorded round, each the bytes that round retained. -/
theorem rep_facts (h64 : Claims.Out64 sha512) (g : Goals sha512 Readable.relation) (o : Run sha512 Readable.relation)
    (cs : List Coin) {st : Stmt} {rb : ByteArray} {HS : Array (Array Flock.Round)}
    (hsts : o.sts = #[st]) (htab : o.spec.tables = #[st.tags.table]) (hss : HS.size = o.spec.repStreams.size)
    (hsess : o.sess = JsonRT.sessOf o.spec (Claims.hOf sha512) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) HS
      o.proofs)
    {k : ℕ} (hk : k < HS.size) :
    (RecDeploy.repSchedAt o.spec o.sts[k / 2]!).length = HS[k].size ∧
    (∀ j (hj : j < HS[k].size), HS[k][j].n = ((RecDeploy.repSchedAt o.spec o.sts[k / 2]!).getD j default).1) ∧
    (framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k)).length = HS[k].size ∧
    ∀ j (hj : j < HS[k].size), HS[k][j].bytes.getD ByteArray.empty =
      ⟨((framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k)).getD j []).toArray⟩ := by
  have hR := JsonRT.repStreams_one htab
  have hHS : HS.size = 2 := by rw [hss, hR]; rfl
  have hsz : o.sts.size = 1 := by rw [hsts]; rfl
  have hk2 : k < 2 * o.sts.size := by omega
  have hi : k / 2 < o.tabs.size := by rw [run_tabs_size]; omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by omega)
  obtain ⟨-, hts⟩ := setup_tables o.setup
  obtain ⟨-, hall⟩ := sessionS_reps h64 (rfl : o.resp 0 = o.recC₁.bytes) hts.symm o.recTabsOk o.ran
    (capRows o.recCaps) cs.tail.dropLast
  have hsch : SchedOf (o.recCtx cs k) := (hall k hk2).2.1
  unfold SchedOf at hsch
  obtain ⟨hf0, hkl0, rs, hrr0, hsch0⟩ := hsch
  have hf' : fast100 o.tabs[k / 2]!.m = some ((fast100 o.tabs[k / 2]!.m).getD default) := hf0
  have hf : fast100 o.tabs[k / 2]!.m = some (o.sch (k / 2)) := by rw [run_sch o hi]; exact hf'
  have hkl : o.tabs[k / 2]!.fold.kLog = o.tabs[k / 2]!.kLog := hkl0
  obtain ⟨stream, extras, hfind, hex, hns⟩ := verifyRepWith_readsN o.hs hkl hf (Run.rep_ok g o hi hr)
  obtain rfl : extras = o.extras (k / 2) := by
    have := g.repExtras o (k / 2) hi
    rw [hex] at this
    exact Except.ok.inj this
  have hpts : o.sess.points = pointsOf o.spec o.recC₁ := by rw [hsess]; rfl
  have hex' : (tabsOf o.sts)[k / 2]!.extra (pointsOf o.spec o.recC₁) = .ok (o.extras (k / 2)) := by
    rw [← hpts]; exact hex
  have hkr : k < o.spec.repStreams.size := by omega
  have hnd := nodup_names (namesDistinct_of_setup (setupOf_some o.setup).2 o.recPublicInputs)
  have hfs := stream_find (h := Claims.hOf sha512) (rb := #[rb]) hss hnd hkr
  rw [repName_one htab hkr] at hfs
  have hfind' : o.sess.streams.find?
      (fun x => x.name == toString o.spec.tables[k / 2]! ++ toString "/rep" ++ toString (k % 2)) = some stream :=
    hfind
  rw [hsess] at hfind'
  have hst : stream = JsonRT.streamRec o.spec (Claims.hOf sha512) #[rb] (o.spec.repStreams[k], HS[k]) :=
    Option.some.inj (hfind'.symm.trans hfs)
  have host : o.stream (k / 2) (k % 2) = stream := by
    unfold Run.stream
    have e : o.sess.streams.find? (·.name == s!"{o.spec.tables[k / 2]!}/rep{k % 2}") = some stream := hfind
    rw [e]
    rfl
  have hrounds : (o.stream (k / 2) (k % 2)).rounds = HS[k].map (JsonRT.roundRec o.spec (Claims.hOf sha512)) := by
    rw [host, hst]
    rfl
  have hsize : (o.stream (k / 2) (k % 2)).rounds.size = HS[k].size := by rw [hrounds, Array.size_map]
  have hn : (o.recCtx cs k).n = 2 + (o.extras (k / 2)).size := by
    unfold Run.recCtx repCtx
    simp only [hex']
  have hsc : (o.recCtx cs k).sch = o.sch (k / 2) := by rw [run_sch o hi]; rfl
  have hproof : o.proof (k / 2) (k % 2) = o.recPs[k]! := run_proof_rec h64 o hex' hk2
  have hN : o.tabs[k / 2]!.merkle.Named := by
    obtain ⟨st', hst', -, hmem, -⟩ := o.schemeOk
    have e : o.tabs[k / 2]! = Setup.ofCircuit st' := by
      rw [show k / 2 = 0 by omega]
      unfold Run.tabs tabsOf
      rw [hst']
      first | rfl | simp
    rw [e]
    exact named_of_all hmem
  have hret : AllRetained (o.stream (k / 2) (k % 2)).rounds := by
    intro j hj
    rw [hrounds] at hj ⊢
    rw [getElem!_pos _ j hj, Array.getElem_map]
    exact ⟨_, rfl⟩
  obtain ⟨hlen, hret'⟩ := frames_emitted g o hi hr hN hf hret
  have hF : framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k) =
      (chunks (repFrames o.tabs[k / 2]! (o.sch (k / 2)) (o.spec.domain o.spec.tables[k / 2]! (k % 2)).toUTF8
        (2 + (o.extras (k / 2)).size) (o.proof (k / 2) (k % 2)) (o.sw (k / 2) (k % 2)).1.1)).1.map encs := by
    rw [hproof, ← hn, ← hsc]
    rfl
  have hlenF : (framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k)).length = HS[k].size := by
    rw [hF, List.length_map, hlen, hsize]
  have hrr : RecVStar.repRounds o.tabs[k / 2]!.m o.tabs[k / 2]!.fold.kLog (2 + (o.extras (k / 2)).size) =
      some rs := by
    rw [← hn]; exact hrr0
  have hsched : RecDeploy.repSchedAt o.spec o.sts[k / 2]! = rs.toList := hsch0
  have hcnt := repRounds_sqs hf (FlockSoundness.Refine.fast100_dims _ _ hf).1 hrr
  rw [schedN_eq_repNs hf, hkl] at hcnt
  have hmap : (RecDeploy.repSchedAt o.spec o.sts[k / 2]!).map (·.1) = HS[k].toList.map Flock.Round.n := by
    rw [hsched, hcnt, ← hns, hst]
    simp [JsonRT.streamRec, JsonRT.roundRec]
  have hlen1 : (RecDeploy.repSchedAt o.spec o.sts[k / 2]!).length = HS[k].size := by
    simpa using congrArg List.length hmap
  refine ⟨hlen1, fun j hj => ?_, hlenF, fun j hj => ?_⟩
  · have h3 := List.getElem_of_eq hmap (i := j) (by rw [List.length_map]; omega)
    simp only [List.getElem_map, Array.getElem_toList] at h3
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega), Option.getD_some]
    exact h3.symm
  · have hj' : j < (o.stream (k / 2) (k % 2)).rounds.size := by rw [hsize]; exact hj
    have hRj := hret' j hj'
    rw [hrounds, getElem!_pos (HS[k].map (JsonRT.roundRec o.spec (Claims.hOf sha512))) j
      (by rw [Array.size_map]; exact hj), Array.getElem_map] at hRj
    simp only [Retains, JsonRT.roundRec, Option.map_some, Option.some.injEq] at hRj
    have hfj : (framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k)).getD j [] =
        encs ((chunks (repFrames o.tabs[k / 2]! (o.sch (k / 2)) (o.spec.domain o.spec.tables[k / 2]! (k % 2)).toUTF8
          (2 + (o.extras (k / 2)).size) (o.proof (k / 2) (k % 2)) (o.sw (k / 2) (k % 2)).1.1)).1.getD j []) := by
      rw [hF, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
      cases (chunks (repFrames o.tabs[k / 2]! (o.sch (k / 2)) (o.spec.domain o.spec.tables[k / 2]! (k % 2)).toUTF8
          (2 + (o.extras (k / 2)).size) (o.proof (k / 2) (k % 2)) (o.sw (k / 2) (k % 2)).1.1)).1[j]? <;> rfl
    rw [hfj]
    apply ByteArray.ext
    apply Array.toList_inj.1
    rw [hRj]

/-! ## The reader's streams and link -/

/-- **The reader's streams at the honest run are the record's, read live**: rep `k` of `recStreams` at the honest
proofs' framed bytes and the deployed verifier's coins is the recorded stream `k` as `liveOf` reads it. -/
theorem recStreams_eq (h64 : Claims.Out64 sha512) (g : Goals sha512 Readable.relation)
    (o : Run sha512 Readable.relation) {cs : List Coin} (hcs : o.CoinsAt cs) {st : Stmt} {rb : ByteArray}
    {HS : Array (Array Flock.Round)} {bs : List Claims.Msg}
    (hsts : o.sts = #[st]) (htab : o.spec.tables = #[st.tags.table]) (hss : HS.size = o.spec.repStreams.size)
    (hsess : o.sess = JsonRT.sessOf o.spec (Claims.hOf sha512) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) HS
      o.proofs)
    (hrecs : JsonRT.Recs o.resp 1 (HS.toList.flatMap Array.toList) bs) :
    recStreams o.spec o.sts (o.recFrames cs) cs.tail.dropLast = HS.map (·.map JsonRT.liveRound) := by
  have hHS : HS.size = 2 := by rw [hss, JsonRT.repStreams_one htab]; rfl
  have hsz : o.sts.size = 1 := by rw [hsts]; rfl
  have F := fun k (hk : k < HS.size) => rep_facts h64 g o cs hsts htab hss hsess hk
  have hrl : repLens o.spec o.sts = HS.toList.map Array.size := by
    apply List.ext_getElem (by simp [repLens, hsz, hHS])
    intro k h1 h2
    simp only [repLens, List.getElem_map, List.getElem_range, Array.getElem_toList]
    exact (F k (by simpa [repLens, hsz, hHS] using h1)).1
  have hps : o.recPs.size = 2 := by
    obtain ⟨-, hts⟩ := setup_tables o.setup
    unfold Run.recPs
    rw [filePs_size, (sessionS_reps h64 (rfl : o.resp 0 = o.recC₁.bytes) hts.symm o.recTabsOk o.ran [] []).1, hsz]
  have hcut : cutLens (repLens o.spec o.sts) (o.recFrames cs) =
      (List.range 2).map fun k => framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k) := by
    have hl : repLens o.spec o.sts =
        ((List.range 2).map fun k => framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k)).map List.length := by
      rw [hrl]
      apply List.ext_getElem (by simp [hHS])
      intro k h1 h2
      have hk : k < HS.size := by simpa [hHS] using h1
      simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
      exact (F k hk).2.2.1.symm
    unfold Run.recFrames
    rw [hps, hl, cutLens_flatten_eq]
  unfold recStreams
  rw [hcut]
  apply Array.ext
  · simp [hsz, hHS]
  · intro k h1 h2
    have hk : k < HS.size := by simpa [hsz, hHS] using h1
    obtain ⟨-, hn, hlen, hby⟩ := F k hk
    have hget : ((List.range 2).map fun k => framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k)).getD k [] =
        framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k) := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega)]
      rfl
    simp only [List.getElem_toArray, List.getElem_map, List.getElem_range, Array.getElem_map]
    rw [hget]
    apply Array.ext
    · simp [hlen]
    · intro j h3 h4
      have hj : j < HS[k].size := by simpa [hlen] using h3
      simp only [List.getElem_toArray, List.getElem_mapIdx, Array.getElem_map]
      have hc := hs_coins o hcs hrl hrecs hk hj
      rw [List.map_take, ← hrl, hn j hj] at hc
      have hb := hby j hj
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega), Option.getD_some] at hb
      show _ = (⟨HS[k][j].n, ByteArray.empty, HS[k][j].coins, some (HS[k][j].bytes.getD ByteArray.empty)⟩ :
        Flock.Round)
      rw [hb, ← hc, hn j hj]

/-- **The link the reader takes from the honest round 0 is the record's `root_B`.** -/
theorem recLink_eq (h64 : Claims.Out64 sha512) (g : Goals sha512 Readable.relation)
    (o : Run sha512 Readable.relation) {st : Stmt} {rb : ByteArray} {HS : Array (Array Flock.Round)}
    (hsts : o.sts = #[st])
    (hsess : o.sess = JsonRT.sessOf o.spec (Claims.hOf sha512) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) HS
      o.proofs) :
    o.recLink.1.toArray = #[rb] := by
  have hsz : o.sts.size = 1 := by rw [hsts]; rfl
  have hi : 0 < o.tabs.size := by rw [run_tabs_size]; omega
  have hpts : o.sess.points = pointsOf o.spec o.recC₁ := by rw [hsess]; rfl
  have hex : (tabsOf o.sts)[0 / 2]!.extra (pointsOf o.spec o.recC₁) = .ok (o.extras (0 / 2)) := by
    rw [← hpts]; exact g.repExtras o 0 hi
  have hp : o.proof 0 0 = o.recPs[0]! := run_proof_rec h64 o hex (k := 0) (by omega)
  have hc := bytes_eq_of_bne (g.capCommitted o 0 hi 0 (by omega))
  rw [hp, hsess] at hc
  have hc' : FlockComplete.flat (filePs o.spec o.sts o.recC₁ o.proofs)[0]!.cap = rb := hc
  have e : (o.recCaps.toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty) = [rb] := by
    unfold Run.recCaps fileCaps
    rw [hsz, show Array.range 1 = #[0] from rfl, ← hc']
    simp [FlockComplete.flat, Array.foldl_toList]
  unfold Run.recLink
  rw [e]

end FlockDeploy
