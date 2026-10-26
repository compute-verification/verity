import Proofs.Flock.Claims.KS.PlainRCaps
import Proofs.Flock.C3.Compose.HoloStreamSizes

/-!
# The cap two accepted runs of `VR rd` absorb, from the rows before its round

`CapPrefixR rd` (`plain_capPrefixK`'s form at `VR rd`) holds. At `nextR`, a rep's stream rounds are the framed bytes of
the proof `readRep` fits to the rep's opened rows (`readRep_fits`), and custody makes them the rounds the verdict reads.
So the decoded proof's chunk at the cap's round has the fitted proof's chunk's bytes (`plainR_capChunk`). That chunk is
the round's sumcheck message and the cap (`repFrames_capChunk`): the message is the round's absorbed row
(`FitsRep.elems`), and the cap the round's committed top (`FitsRep.tops`), the `ℓ`-th of the rep's tops
(`repSchedAt_cap`). Two runs whose coins agree before the plain fork round open the same rows up to it
(`depMsg_congr`), so their fitted chunks there have the same bytes, and their decoded proofs one cap
(`capFr_eq_of_chunk`).

* `repSchedAt_cap`: the rep's schedule has `repRounds` rounds, and commits its `ℓ`-th top alone, in an absorbing round,
  at `preLig + capRound k0 (ℓ + 1)` (evaluated at each `m` below 36, `repRounds_tail`).
* `repFrames_capChunk`: any proof's frames' chunk at that round is `kFr` of one message, the cap, and one squeeze.
* `plainR_capChunk`: an accepted run's decoded rep-`r` chunks have the fitted proof's bytes.
* `capPrefixR`: `CapPrefixR rd`. `plainR_tableBoundAbortLin_R`, `plainR_KSFlockLin_R`: the table bound and
  `KSFlockLin` at the deployed reader from `PlainRdInR` alone.
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody (checksAtJ_ok custody_ok)
open Claims.Bridge.PlainDraw (decideR_sha512)
open Lean (Json)

/-! ## The rep's schedule at the cap's round -/

/-- Compiled Ligerito's rounds `T` at `sch` commit each recursive cap alone, in an absorbing round, at `capRound`. -/
def ligTailOk (sch : Flock.Schedule) (T : List Flock.RecVStar.SchedRound) : Bool :=
  T.length == sch.k0 + 5 + 11 * (sch.levels.size - 1) &&
    (List.range (sch.levels.size - 1)).all fun ℓ =>
      ((T.take (capRound sch.k0 (ℓ + 1))).map (·.2.1.size)).sum == ℓ &&
        (T.getD (capRound sch.k0 (ℓ + 1)) default).2.1.size == 1 &&
        (T.getD (capRound sch.k0 (ℓ + 1)) default).2.2.2

/-- `repRounds` at `m` returns where `fast100` does, with a tail `ligTailOk` holds of. -/
def schedOk (m : ℕ) : Bool :=
  match fast100 m, Flock.RecVStar.repRounds m 6 1 with
  | none, _ => true
  | some _, none => false
  | some sch, some out => ligTailOk sch (out.toList.drop (FlockC3.Compose.repHead m 6 1).size)

theorem schedOk_lt36 : ∀ m < 36, schedOk m = true := by decide +kernel

theorem repHead_size {m kLog claims : ℕ} (hm : 6 ≤ m) :
    (FlockC3.Compose.repHead m kLog claims).size = (3 + (m - 6)) + (3 + (kLog - 6)) + (claims + 1) := by
  simp [FlockC3.Compose.repHead]
  omega

theorem repHead_tops (m kLog claims : ℕ) :
    ((FlockC3.Compose.repHead m kLog claims).toList.map (·.2.1.size)).sum = 0 := by
  simp [FlockC3.Compose.repHead, List.map_replicate]

/-- **The rep's schedule commits its `ℓ`-th top alone, in an absorbing round, at the cap's round**: table `st`'s rep
schedule has `repRounds false spec st` rounds; the rounds before `preLig + capRound k0 (ℓ + 1)` commit `ℓ` tops, and
that round one, and it absorbs. -/
theorem repSchedAt_cap {spec : SessionSpec} {st : Stmt} {sch : Flock.Schedule} (hsch : fast100 st.m = some sch) :
    (RecDeploy.repSchedAt spec st).length = repRounds false spec st ∧
      ∀ ℓ, ℓ + 1 < sch.levels.size →
        (((RecDeploy.repSchedAt spec st).take (preLig spec st + capRound sch.k0 (ℓ + 1))).map
            (·.2.1.size)).sum = ℓ ∧
          ((RecDeploy.repSchedAt spec st).getD (preLig spec st + capRound sch.k0 (ℓ + 1)) default).2.1.size = 1 ∧
          ((RecDeploy.repSchedAt spec st).getD (preLig spec st + capRound sch.k0 (ℓ + 1)) default).2.2.2 = true := by
  have h22 := (Flock.Guarantees.HoloRepCoins.fast100_facts hsch).1
  have hm36 : st.m < 36 := by
    refine Decidable.by_contra fun hn => ?_
    have hb : (st.m < 22 || st.m > 35) = true := by
      simp only [Bool.or_eq_true, decide_eq_true_eq]
      omega
    have h0 : Flock.fast100 st.m = none := by
      unfold Flock.fast100
      simp [hb]
    rw [h0] at hsch
    cases hsch
  have key := schedOk_lt36 st.m hm36
  unfold schedOk at key
  rw [hsch] at key
  cases hr : Flock.RecVStar.repRounds st.m 6 1 with
  | none => rw [hr] at key; cases key
  | some out =>
    rw [hr] at key
    obtain ⟨T, rfl, hall⟩ := FlockC3.Compose.repRounds_tail hr
    have eT : (FlockC3.Compose.repHead st.m 6 1 ++ T).toList.drop (FlockC3.Compose.repHead st.m 6 1).size =
        T.toList := by simp
    simp only [eT, ligTailOk, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at key
    obtain ⟨hlen, hlv⟩ := key
    have hS : RecDeploy.repSchedAt spec st =
        (FlockC3.Compose.repHead st.m st.c.kLog (2 + spec.points * st.regions.size)).toList ++ T.toList := by
      simp only [RecDeploy.repSchedAt, hall, Option.map_some, Option.getD_some, Array.toList_append]
    have hH : (FlockC3.Compose.repHead st.m st.c.kLog (2 + spec.points * st.regions.size)).toList.length =
        preLig spec st := by
      rw [Array.length_toList, repHead_size (by omega)]
      rfl
    rw [hS]
    refine ⟨?_, fun ℓ hℓ => ?_⟩
    · rw [List.length_append, hH, hlen, repRounds_plain hsch]
    · obtain ⟨⟨hsum, hone⟩, habs⟩ := hlv ℓ (by omega)
      have hd : ∀ d : Flock.RecVStar.SchedRound,
          ((FlockC3.Compose.repHead st.m st.c.kLog (2 + spec.points * st.regions.size)).toList ++ T.toList).getD
            (preLig spec st + capRound sch.k0 (ℓ + 1)) d = T.toList.getD (capRound sch.k0 (ℓ + 1)) d := by
        intro d
        rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), hH,
          Nat.add_sub_cancel_left]
      refine ⟨?_, by rw [hd]; exact hone, by rw [hd]; exact habs⟩
      rw [List.take_append, List.take_of_length_le (by omega), hH, Nat.add_sub_cancel_left, List.map_append,
        List.sum_append_nat, repHead_tops, hsum, Nat.zero_add]

/-! ## A proof's frames at the cap's round -/

theorem chunks_nc : ∀ B : List Frame, B.countP closes = 0 → chunks B = ([], B)
  | [], _ => rfl
  | f :: fs, h => by
    have hf : closes f = false := by
      rw [List.countP_cons] at h
      cases hc : closes f <;> simp_all
    have hfs : fs.countP closes = 0 := by
      rw [List.countP_cons] at h
      omega
    simp only [chunks, hf, Bool.false_eq_true, ite_false, chunks_nc fs hfs]

theorem pend_append_nc (pre B : List Frame) (hB : B.countP closes = 0) :
    (chunks (pre ++ B)).2 = (chunks pre).2 ++ B := by
  rw [chunks_append, chunks_nc B hB]
  rfl

theorem pend_close (zs : List Frame) {s : Frame} (hs : closes s = true) : (chunks (zs ++ [s])).2 = [] := by
  rw [chunks_append]
  have e : chunks [s] = ([[s]], []) := by simp only [chunks, hs, ↓reduceIte]
  rw [e]
  rfl

/-- **The frames pending after a run of folding challenges are its last message's.** -/
theorem pend_chFr (p : LigeritoProof) : ∀ (k msg : ℕ) (pre : List Frame), (chunks pre).2 = [] →
    (chunks (pre ++ (kFr p.messages[msg]! ++ chFr p (msg + 1) k))).2 = kFr p.messages[msg + k]!
  | 0, msg, pre, hX => by
    rw [show chFr p (msg + 1) 0 = [] from rfl, List.append_nil, pend_append_nc pre _ (countP_kFr _), hX,
      List.nil_append, Nat.add_zero]
  | k + 1, msg, pre, hX => by
    have e : pre ++ (kFr p.messages[msg]! ++ chFr p (msg + 1) (k + 1)) =
        (pre ++ kFr p.messages[msg]! ++ [.sq 2 2]) ++ (kFr p.messages[msg + 1]! ++ chFr p (msg + 1 + 1) k) := by
      simp only [chFr, List.append_assoc, List.cons_append, List.nil_append]
    rw [e, pend_chFr p k (msg + 1) _ (pend_close _ (by simp [closes])), show msg + 1 + k = msg + (k + 1) by omega]

/-- The sumcheck message the round absorbing the `ℓ`-th recursive cap carries. -/
def capMsg (k0 ℓ : ℕ) : ℕ := if ℓ = 0 then k0 else k0 + 7 + 7 * (ℓ - 1)

/-- The squeeze that closes the round absorbing the `ℓ`-th recursive cap. -/
def capSq (sch : Flock.Schedule) (ℓ : ℕ) : Frame :=
  if ℓ = 0 then .sq 2 (sch.n - sch.k0 + 1) else .sq 2 ((sch.levels.getD ℓ default).cols + 1)

/-- **A proof's frames' round absorbing its `ℓ`-th recursive cap** (`repFrames_capSplit`'s split, finer): for any
proof and `T`, the round `preLig + capRound k0 (ℓ + 1)` is `kFr` of message `capMsg k0 ℓ`, the cap, and `capSq`. -/
theorem repFrames_capChunk (st : Setup) (sch : Flock.Schedule) (domain : ByteArray) (n : ℕ) (p : Proof) (T : F128)
    (hm : 14 ≤ st.m) (hn : n ≠ 0) (hn0 : sch.n ≠ 0) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0)
    {ℓ : ℕ} (hℓ : ℓ + 1 < sch.levels.size) :
    (chunks (repFrames st sch domain n p T)).1.getD
        ((3 + (st.m - 6)) + (3 + (st.fold.kLog - 6)) + (n + 1) + capRound sch.k0 (ℓ + 1)) [] =
      kFr p.ligerito.messages[capMsg sch.k0 ℓ]! ++ [capFr p.ligerito.recursiveCaps[ℓ]!, capSq sch ℓ] := by
  rcases ℓ with _ | i
  · set zs : List Frame :=
      .dom domain :: (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck ++
        lcFrames st.fold.kLog p.lincheck ++ rsFrames n p.ringSwitches ++
        ([Frame.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr p.cap] ++
          [.sq 2 sch.n, .one p.ligerito.oodValues[0]!, .bytes (le64 p.ligerito.claimNonces[0]!)])) with hZ
    set a : List Frame :=
      (zs ++ [.sq 1 1]) ++ (kFr p.ligerito.messages[0]! ++ chFr p.ligerito (0 + 1) sch.k0) with ha
    set b : List Frame := (.one p.ligerito.oodValues[1]! :: kFr p.ligerito.messages[sch.k0 + 1]! ++
          [.bytes (le64 p.ligerito.claimNonces[1]!), .sq 1 1]) ++
        introFr p.ligerito (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
        queryFr p.ligerito (sch.levels.getD 0 default) 0 0 ++ kFr p.ligerito.messages[sch.k0 + 3]! ++
        [.bytes (le64 p.ligerito.claimNonces[3]!), .sq 1 1] ++
        (List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p.ligerito) with hb
    have hF : repFrames st sch domain n p T =
        a ++ capFr p.ligerito.recursiveCaps[0]! :: Frame.sq 2 (sch.n - sch.k0 + 1) :: b := by
      simp only [ha, hZ, hb, repFrames, ligFrames, introFr, List.append_assoc, List.cons_append, List.nil_append,
        Nat.zero_add]
    have hcnt : (chunks a).1.length =
        (3 + (st.m - 6)) + (3 + (st.fold.kLog - 6)) + (n + 1) + capRound sch.k0 (0 + 1) := by
      rw [chunks_length]
      simp only [ha, hZ, List.countP_cons, List.countP_append, countP_zcFrames hm, countP_lcFrames,
        countP_rsFrames hn, countP_kFr, countP_chFr]
      simp [closes, capFr, hn0, capRound]
      omega
    rw [hF, ← hcnt, chunks_getD_split a b (by simp [closes, capFr]) (by simp [closes]), ha,
      pend_chFr p.ligerito sch.k0 0 _ (pend_close zs (by simp [closes]))]
    simp [capMsg, capSq]
  · have hi : i + 1 < sch.levels.size - 1 := by omega
    have hL : levelFr sch p.ligerito i = chFr p.ligerito (sch.k0 + 4 + 7 * i) 4 ++
        capFr p.ligerito.recursiveCaps[i + 1]! ::
          (introFr p.ligerito ((sch.levels.getD (i + 1) default).cols + 1) (3 + 2 * i) (sch.k0 + 8 + 7 * i)
              (4 + 3 * i) ++
            introFr p.ligerito ((sch.levels.getD (i + 1) default).cols + 1) (4 + 2 * i) (sch.k0 + 9 + 7 * i)
              (5 + 3 * i) ++
            queryFr p.ligerito (sch.levels.getD (i + 1) default) (1 + i) (1 + i) ++
            kFr p.ligerito.messages[sch.k0 + 10 + 7 * i]! ++
            [.bytes (le64 p.ligerito.claimNonces[6 + 3 * i]!), .sq 1 1]) := by
      unfold levelFr
      rw [ite_eq_left hi]
    have hlev : ((List.range' 0 i).flatMap (levelFr sch p.ligerito)).countP closes = i * 11 :=
      countP_flatMap_range' i 0 fun j _ hj => countP_levelFr sch p.ligerito (by omega) (hq (j + 1) (by omega))
    set zs : List Frame :=
      .dom domain :: (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck ++
        lcFrames st.fold.kLog p.lincheck ++ rsFrames n p.ringSwitches ++
        ([Frame.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr p.cap] ++
          ([.sq 2 sch.n, .one p.ligerito.oodValues[0]!, .bytes (le64 p.ligerito.claimNonces[0]!), .sq 1 1] ++
            kFr p.ligerito.messages[0]! ++ chFr p.ligerito 1 sch.k0 ++ capFr p.ligerito.recursiveCaps[0]! ::
            (introFr p.ligerito (sch.n - sch.k0 + 1) 1 (sch.k0 + 1) 1 ++
              introFr p.ligerito (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
              queryFr p.ligerito (sch.levels.getD 0 default) 0 0 ++ kFr p.ligerito.messages[sch.k0 + 3]! ++
              [.bytes (le64 p.ligerito.claimNonces[3]!), .sq 1 1] ++
              (List.range' 0 i).flatMap (levelFr sch p.ligerito))))) with hZ
    set a : List Frame := (zs ++ [.sq 2 2]) ++
        (kFr p.ligerito.messages[sch.k0 + 4 + 7 * i]! ++ chFr p.ligerito (sch.k0 + 4 + 7 * i + 1) 3) with ha
    set b : List Frame := (.one p.ligerito.oodValues[3 + 2 * i]! :: kFr p.ligerito.messages[sch.k0 + 8 + 7 * i]! ++
          [.bytes (le64 p.ligerito.claimNonces[4 + 3 * i]!), .sq 1 1]) ++
        introFr p.ligerito ((sch.levels.getD (i + 1) default).cols + 1) (4 + 2 * i) (sch.k0 + 9 + 7 * i)
          (5 + 3 * i) ++
        queryFr p.ligerito (sch.levels.getD (i + 1) default) (1 + i) (1 + i) ++
        kFr p.ligerito.messages[sch.k0 + 10 + 7 * i]! ++
        [.bytes (le64 p.ligerito.claimNonces[6 + 3 * i]!), .sq 1 1] ++
        (List.range' (i + 1) (sch.levels.size - 1 - (i + 1))).flatMap (levelFr sch p.ligerito) with hb
    have hch : chFr p.ligerito (sch.k0 + 4 + 7 * i) 4 = .sq 2 2 :: kFr p.ligerito.messages[sch.k0 + 4 + 7 * i]! ++
        chFr p.ligerito (sch.k0 + 4 + 7 * i + 1) 3 := rfl
    have hF : repFrames st sch domain n p T =
        a ++ capFr p.ligerito.recursiveCaps[i + 1]! ::
          Frame.sq 2 ((sch.levels.getD (i + 1) default).cols + 1) :: b := by
      rw [repFrames, ligFrames, range'_split (show i < sch.levels.size - 1 by omega), List.flatMap_append,
        List.flatMap_cons, hL, hch]
      simp only [ha, hZ, hb, introFr, List.append_assoc, List.cons_append, List.nil_append]
    have hcnt : (chunks a).1.length =
        (3 + (st.m - 6)) + (3 + (st.fold.kLog - 6)) + (n + 1) + capRound sch.k0 (i + 1 + 1) := by
      rw [chunks_length]
      simp only [ha, hZ, List.countP_cons, List.countP_append, countP_zcFrames hm, countP_lcFrames,
        countP_rsFrames hn, countP_kFr, countP_chFr, countP_introFr p.ligerito (show sch.n - sch.k0 + 1 ≠ 0 by omega),
        countP_queryFr p.ligerito (hq 0 (by omega)), hlev]
      simp [closes, capFr, hn0, capRound]
      omega
    rw [hF, ← hcnt, chunks_getD_split a b (by simp [closes, capFr]) (by simp [closes]), ha,
      pend_chFr p.ligerito 3 (sch.k0 + 4 + 7 * i) _ (pend_close zs (by simp [closes])),
      show sch.k0 + 4 + 7 * i + 3 = sch.k0 + 7 + 7 * i by omega]
    simp [capMsg, capSq]

end FlockDeploy.KS
