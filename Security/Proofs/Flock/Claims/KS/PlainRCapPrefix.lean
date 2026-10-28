import Proofs.Flock.Claims.KS.PlainRCaps
import Proofs.Flock.C3.Compose.HoloStreamSizes
import Proofs.Flock.Claims.Recursion.NextRHonest

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

/-! ## An accepted run's decoded rep, round by round -/

/-- The first two lists' entries in a flattening, by list and position. -/
theorem flat_two_list {α : Type} (L : List (List α)) {n r j : ℕ} (h0 : (L.getD 0 []).length = n) (hr : r < 2)
    (hj : j < (L.getD r []).length) : L.flatten[r * n + j]? = (L.getD r [])[j]? := by
  subst h0
  obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
  · rcases L with _ | ⟨a, l⟩
    · simp at hj
    · change j < a.length at hj
      show (a ++ l.flatten)[0 * a.length + j]? = a[j]?
      rw [Nat.zero_mul, Nat.zero_add]
      exact List.getElem?_append_left hj
  · rcases L with _ | ⟨a, _ | ⟨b, l⟩⟩
    · simp at hj
    · simp at hj
    · change j < b.length at hj
      show (a ++ (b ++ l.flatten))[1 * a.length + j]? = b[j]?
      rw [Nat.one_mul, List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]
      exact List.getElem?_append_left hj

/-- **An accepted run of `VR rd` decodes rep `r` to a proof with its read proof's bytes, round by round.** At an
accepted run deciding at `actR`, rep `r < 2` of table 0's file of the read proofs, decoded at the table's `fast100`
schedule and `|regions| · points` extra claims, has frames (at some `T`) whose `j`-th chunk has, for every round `j` of
the rep's `repRounds`, the bytes of the `j`-th chunk of the proof `readRep` fitted to the rep's opened rows (at the
fit's `T'`): the verdict accepts the rep at the shipped oracle and the table's named scheme, so its stream's rounds
retained its chunks (`rep_rounds`), and custody makes the stream's rounds the decision's round messages
(`custody_ok`), which are the fitted proofs' framed bytes, rep by rep (`decR_some`, `readRep_fits`). The fit's context
is the decoding's. -/
theorem plainR_capChunk {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω c).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {e : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {st : Stmt} {sch : Flock.Schedule} {p : Proof} {r : ℕ}
    (hact : actR rd x ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst).length).map (Claims.coinSeq c)) =
      .decide s spec sts link e mids f)
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hr : r < 2)
    (hp : decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points))
      (filesAt (tabsOf sts) (pointsOf spec e) f.ps)[r]! = .ok p) :
    ∃ (rss : List (List ByteArray)) (cx : RepCtx) (q : Proof) (T T' : F128),
      ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst).tail.dropLast.mapM rowsOf = some rss ∧
      cx.st = Setup.ofCircuit st ∧ cx.sch = sch ∧ cx.dom = (spec.domain spec.tables[0]! r).toUTF8 ∧
      cx.n = 2 + st.regions.size * spec.points ∧ cx.sched = RecDeploy.repSchedAt spec st ∧
      FitsRep cx ((cutLens (repLens spec sts) rss).getD r []) q T' ∧
      ∀ j < repRounds false spec st,
        encs ((chunks (repFrames (Setup.ofCircuit st) sch (spec.domain spec.tables[0]! r).toUTF8
          (2 + st.regions.size * spec.points) p T)).1.getD j []) = encs ((chunksOf cx q T').getD j []) := by
  obtain ⟨s', spec', sts', link', e', mids', f', hact', hrun⟩ := accR_decide h
  rw [hact] at hact'
  injection hact' with e1 e2 e3 e4 e5 e6 e7
  subst e1 e2 e3 e4 e5 e6 e7
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω c h
  obtain ⟨-, hc, hsu, -, -⟩ := decideR_sha512 hact hacc
  have hv := verdictJ_within (decisionSF_runs hsu hrun)
  generalize (Claims.run sha512 P (VR rd) x ω c).vLog.map Prod.fst = L at hv
  have hdr := decR_some (actR_decide hact)
  generalize (Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst = ms at hdr ⊢
  generalize (List.range ms.length).map (Claims.coinSeq c) = cs at hdr
  obtain ⟨v, c₁', r₀, rss, fl, link'', pf, -, -, -, hrss, -, hread, hdd⟩ := hdr
  simp only [DecR.mk.injEq] at hdd
  obtain ⟨-, hsp, hsts, -, rfl, hmids, hf⟩ := hdd
  rw [← hsp, ← hsts] at hread
  have hcond : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length := by
    by_contra hn
    rw [readPs, if_neg hn] at hread
    cases hread
  rw [readPs, if_pos hcond] at hread
  obtain ⟨rs, hrs, hpf⟩ := Option.map_eq_some_iff.1 hread
  subst hpf
  dsimp only at hmids hf
  obtain ⟨hrsl, hget⟩ := mapM_some_get hrs
  rw [List.length_range] at hrsl
  obtain ⟨-, lv, hlv, hcus⟩ := checksAtJ_ok hc
  have hpts' := liveOfJ_points spec e mids _ hlv
  have hpn : lv.points.size = spec.points := by
    rw [hpts']
    simp [pointsOf]
  obtain ⟨hsts0, hst0⟩ := Array.getElem?_eq_some_iff.1 hst
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, schs, hdec, hsz, -, -, hall⟩ := verifyJWithMK_tables_cap id _ hv
  obtain ⟨hi', cB, hsch', h0r, h1r⟩ := hall 0 h0
  simp only [ht0] at hsch' h0r h1r
  have e0 : schs[0] = sch := Option.some.inj (hsch'.symm.trans hsch)
  rw [e0] at h0r h1r
  have hrep : verifyRepWith (hashesIn L) spec (Setup.ofCircuit st) sch sess 0 r
      (filesAt (tabsOf sts) (pointsOf spec e) f.ps)[r]! = .ok cB := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · simpa using h0r
    · simpa using h1r
  have named : (Setup.ofCircuit st).merkle.Named := by
    show st.merkle.Named
    rcases checksAtJ_leaf hc hsu 0 hsts0 with e' | ⟨-, e'⟩ <;> rw [hst0] at e' <;> rw [e']
    · exact With.named_of_find? (id := "hm96-sha512/v1") rfl
    · exact With.named_of_find? (id := "flock-leaf/sha512-unsalted") rfl
  have hrep' : verifyRep spec (Setup.ofCircuit st) sch sess 0 r
      (filesAt (tabsOf sts) (pointsOf spec e) f.ps)[r]! = .ok cB := by
    rw [← With.verifyRepWith_std _ _ named]
    exact (With.verifyRepWith_sim sub (sub_hashes L) spec _ sch sess 0 r _) cB hrep
  have h22 := exec_fast100_ge hsch
  obtain ⟨hdim, -⟩ := fast100_dims _ _ hsch
  have hlvs : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level _ _ hsch _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨stream, p', extras, T, hfind, hext, hp', hR⟩ :=
    rep_rounds spec (Setup.ofCircuit st) sch sess 0 r _ cB (by show 14 ≤ st.m; omega) hdim hlvs hrep'
  obtain ⟨-, hpts, -, hstreams⟩ := decodeJWithK_live id _ hdec hlv
  rw [hpts] at hext
  have hn : extras.size = st.regions.size * spec.points := by
    rw [extraClaims_size (st := st) hext, hpn]
  rw [hpts'] at hext
  rw [hn] at hp' hR
  obtain rfl := Except.ok.inj (hp.symm.trans hp')
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs'⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hr
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + r) hk'
  rw [hrs'] at hy
  obtain rfl : stream = y := Option.some.inj (hfind.symm.trans hy)
  obtain ⟨hsz2, hsizes⟩ := streams_size_plainJ id _ hv hlv hpn
  have hR0 : ∀ q (hq : q < lv.streams.size), q < 2 → lv.streams[q].size = repRounds false spec st := by
    intro q hq hq2
    rw [hsizes q hq, show q / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0]
  have hq2 : 2 ≤ lv.streams.size := by omega
  simp only [Nat.mul_zero, Nat.zero_add] at hfa
  rw [getElem!_pos lv.streams r (by omega)] at hfa
  have hsize : stream.rounds.size = repRounds false spec st := by
    have := hfa.length_eq
    rw [Array.length_toList, Array.length_toList, hR0 r _ hr] at this
    exact this
  have hfl := (custody_ok hcus).2.2.2
  have hR2 : 2 ≤ rs.length := by rw [hrsl]; omega
  have hread_k : ∀ k (hk : k < 2), ∃ T', FitsRep (repCtx spec sts e r₀ cs.tail.dropLast k)
      ((cutLens (repLens spec sts) rss).getD k []) (rs[k]'(by omega)).1 T' ∧
      (rs[k]'(by omega)).2 = framesOf (repCtx spec sts e r₀ cs.tail.dropLast k) (rs[k]'(by omega)).1 T' := by
    intro k hk
    have hg := hget k (by rw [List.length_range]; omega) (by omega)
    rw [List.getElem_range] at hg
    exact readRep_fits hg
  have hlenk : ∀ k (hk : k < 2), ((rs.map Prod.snd).getD k []).length = repRounds false spec st := by
    intro k hk
    obtain ⟨Tk, hfk, hmk⟩ := hread_k k hk
    rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem (show k < rs.length by omega),
      Option.map_some, Option.getD_some, hmk, framesOf, List.length_map, hfk.rounds, ← hfk.sched]
    dsimp only [repCtx]
    rw [show k / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0]
    exact (repSchedAt_cap hsch).1
  obtain ⟨T', hfit, hms⟩ := hread_k r hr
  have hchl : (chunksOf (repCtx spec sts e r₀ cs.tail.dropLast r) (rs[r]'(by omega)).1 T').length =
      repRounds false spec st := by
    rw [hfit.rounds, ← hfit.sched]
    dsimp only [repCtx]
    rw [show r / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0]
    exact (repSchedAt_cap hsch).1
  have hbytes : ∀ j < repRounds false spec st, (stream.rounds[j]!).bytes =
      some ⟨(encs ((chunksOf (repCtx spec sts e r₀ cs.tail.dropLast r) (rs[r]'(by omega)).1 T').getD j [])).toArray⟩ := by
    intro j hj
    have hjr : j < (lv.streams[r]'(by omega)).size := by rw [hR0 r _ hr]; exact hj
    have e1 := flat_two_getElem? lv.streams hq2 hr hjr
    rw [hR0 0 (by omega) (by omega), Array.getElem?_eq_getElem hjr] at e1
    obtain ⟨am, hm, hram⟩ := forall₂_getElem?_some hfl e1
    rw [hmids] at hm
    obtain ⟨hb, -⟩ := List.getElem?_zip_eq_some.1 hm
    have hsnd : (rs.map Prod.snd).getD r [] =
        framesOf (repCtx spec sts e r₀ cs.tail.dropLast r) (rs[r]'(by omega)).1 T' := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem (show r < rs.length by omega),
        Option.map_some, Option.getD_some, hms]
    have hj' : j < (chunksOf (repCtx spec sts e r₀ cs.tail.dropLast r) (rs[r]'(by omega)).1 T').length := by
      rw [hchl]; exact hj
    rw [flat_two_list _ (hlenk 0 (by omega)) hr (by rw [hlenk r hr]; exact hj), hsnd, framesOf,
      List.getElem?_map, List.getElem?_eq_getElem hj', Option.map_some, Option.some.injEq] at hb
    have hfj := Claims.Bridge.Custody.forall₂_getElem hfa (show j < stream.rounds.size by omega)
    rw [getElem!_pos stream.rounds j (by omega), hfj.2.2, getElem!_pos _ j hjr, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem hj', Option.getD_some, hb]
    exact hram.1
  have hret : AllRetained stream.rounds := fun i hi => ⟨_, hbytes i (hsize ▸ hi)⟩
  obtain ⟨-, hc'⟩ := hR hret
  refine ⟨rss, repCtx spec sts e r₀ cs.tail.dropLast r, (rs[r]'(by omega)).1, T, T', hrss, ?_, ?_, ?_, ?_, ?_,
    hfit, fun j hj => ?_⟩
  · dsimp only [repCtx]
    rw [show r / 2 = 0 by omega, getElem!_pos _ 0 h0, ht0]
  · dsimp only [repCtx]
    rw [show r / 2 = 0 by omega, getElem!_pos _ 0 h0, ht0]
    show (fast100 st.m).getD default = sch
    rw [hsch]
    rfl
  · dsimp only [repCtx]
    rw [show r / 2 = 0 by omega, Nat.mod_eq_of_lt hr]
  · dsimp only [repCtx]
    rw [show r / 2 = 0 by omega, getElem!_pos _ 0 h0, ht0, hext]
    exact congrArg (2 + ·) hn
  · dsimp only [repCtx]
    rw [show r / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0]
  · have e' := hc' j (hsize ▸ hj)
    unfold Retains at e'
    rw [hbytes j hj] at e'
    simpa using e'.symm

/-! ## Two accepted runs' caps -/

/-- **An accepted run's messages are `P`'s along its coins** (`run_depK`'s proof, at `VR rd`). -/
theorem run_depR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (c : Fin ((VR rd).k x) → Coin) :
    (Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst =
      (List.range (Claims.run sha512 P (VR rd) x ω c).msgs.length).map (depMsg sha512 P x ω (Claims.coinSeq c)) :=
  stepWith_dep sha512 P (VR rd) x ω _ (answersCoin_VR rd sha512 x) _ [] [] _ rfl rfl rfl

theorem mapM_getElem? {α β : Type} {f : α → Option β} : ∀ {l : List α} {r : List β}, l.mapM f = some r →
    ∀ t : ℕ, r[t]? = l[t]?.bind f
  | [], r, h, t => by
    rw [List.mapM_nil] at h
    cases h
    simp
  | a :: l, r, h, t => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.pure_def] at h
    obtain ⟨b, hb, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨r', hr', h⟩ := Option.bind_eq_some_iff.1 h
    have e : r = b :: r' := (Option.some.inj h).symm
    subst e
    cases t with
    | zero => simp [hb]
    | succ t => simpa using mapM_getElem? hr' t

/-- The first two blocks of `cutLens` at two equal leading lengths, by position. -/
theorem cutLens_two {α : Type} (n : ℕ) (ns : List ℕ) (l : List α) {r j : ℕ} (hr : r < 2) (hj : j < n) :
    ((cutLens (n :: n :: ns) l).getD r [])[j]? = l[r * n + j]? := by
  obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
  · show ((l.take n))[j]? = l[0 * n + j]?
    rw [List.getElem?_take, ite_eq_left hj, Nat.zero_mul, Nat.zero_add]
  · show ((l.drop n).take n)[j]? = l[1 * n + j]?
    rw [List.getElem?_take, ite_eq_left hj, List.getElem?_drop, Nat.one_mul]

theorem eq_cons_cons {α : Type} {l : List α} {a b : α} (h0 : l[0]? = some a) (h1 : l[1]? = some b) :
    l = a :: b :: l.drop 2 := by
  rcases l with _ | ⟨a', _ | ⟨b', l⟩⟩
  · simp at h0
  · simp at h1
  · simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, Option.some.injEq] at h0 h1
    subst h0 h1
    rfl

theorem repLens_getElem? {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} (hst : sts[0]? = some st) {k : ℕ}
    (hk : k < 2) : (repLens spec sts)[k]? = some (RecDeploy.repSchedAt spec st).length := by
  obtain ⟨h0, hst0⟩ := Array.getElem?_eq_some_iff.1 hst
  simp only [repLens, List.getElem?_map, List.getElem?_range (show k < 2 * sts.size by omega), Option.map_some]
  rw [show k / 2 = 0 by omega, getElem!_pos sts 0 h0, hst0]

/-- **A flattened zip's first entries read only its first rows**: where `A` and `B` agree below `N`, the flattening of
`g` over `S` zipped with them agrees below the first `N` rows' widths. -/
theorem flatMap_zip_congr {σ β γ : Type} (g : σ → β → List γ) (w : σ → ℕ) (hw : ∀ s b, (g s b).length = w s) :
    ∀ (S : List σ) (A B : List β) (N : ℕ), (∀ j < N, A[j]? = B[j]?) → N ≤ A.length →
      ∀ i < ((S.take N).map w).sum,
        ((S.zip A).flatMap fun q => g q.1 q.2)[i]? = ((S.zip B).flatMap fun q => g q.1 q.2)[i]?
  | [], _, _, _, _, _, i, hi => by simp at hi
  | _ :: _, _, _, 0, _, _, i, hi => by simp at hi
  | _ :: _, [], _, _ + 1, _, hA, _, _ => by simp at hA
  | _ :: _, _ :: _, [], _ + 1, hAB, _, _, _ => by simpa using hAB 0 (by omega)
  | s :: S, a :: A, b :: B, N + 1, hAB, hA, i, hi => by
    have hab : a = b := by simpa using hAB 0 (by omega)
    subst hab
    simp only [List.zip_cons_cons, List.flatMap_cons]
    simp only [List.take_succ_cons, List.map_cons, List.sum_cons] at hi
    by_cases hi' : i < w s
    · rw [List.getElem?_append_left (by rw [hw]; exact hi'), List.getElem?_append_left (by rw [hw]; exact hi')]
    · rw [List.getElem?_append_right (by rw [hw]; omega), List.getElem?_append_right (by rw [hw]; omega), hw]
      exact flatMap_zip_congr g w hw S A B N (fun j hj => by simpa using hAB (j + 1) (by omega))
        (by simpa using hA) _ (by omega)

theorem caps_getElem!_eq {α : Type} {a b : Array (Array α)} {ℓ : ℕ}
    (h : (a.toList.map Array.toList)[ℓ]? = (b.toList.map Array.toList)[ℓ]?) : a[ℓ]! = b[ℓ]! := by
  simp only [List.getElem?_map, Array.getElem?_toList] at h
  by_cases ha : ℓ < a.size <;> by_cases hb : ℓ < b.size
  · rw [Array.getElem?_eq_getElem ha, Array.getElem?_eq_getElem hb, Option.map_some, Option.map_some,
      Option.some.injEq] at h
    rw [getElem!_pos a ℓ ha, getElem!_pos b ℓ hb]
    exact Array.toList_inj.1 h
  · rw [Array.getElem?_eq_getElem ha, Array.getElem?_eq_none (by omega)] at h
    simp at h
  · rw [Array.getElem?_eq_none (by omega), Array.getElem?_eq_getElem hb] at h
    simp at h
  · rw [getElem!_neg a ℓ ha, getElem!_neg b ℓ hb]

theorem vals_capChunk (m : F256 × F256) (cap : Array ByteArray) (sch : Flock.Schedule) (ℓ : ℕ) :
    FlockDeploy.vals (kFr m ++ [capFr cap, capSq sch ℓ]) = [m.1.c0, m.1.c1, m.2.c0, m.2.c1] := by
  unfold capSq
  split <;> rfl

/-- **`CapPrefixR` holds.** At two accepted runs of `VR rd` whose coins agree before the plain fork round, the rows
they open agree up to the cap's round (`depMsg_congr`), so the proofs `readRep` fits there commit one cap there
(`FitsRep.tops`, `flatMap_zip_congr`) and absorb one message there (`FitsRep.elems`, `RowsAt.vals_eq`), and their
chunks there are one (`repFrames_capChunk`). The decoded proofs' chunks there have those bytes (`plainR_capChunk`), so
they absorb one cap (`capFr_eq_of_chunk`). -/
theorem capPrefixR (rd : Claims.Msg → Option FinalP) : CapPrefixR rd := by
  intro Ω P x ω c₁ c₂ h₁ h₂ s spec sts st sch r ℓ _ hst hsch hr hℓ hc link₁ link₂ e₁ e₂ mids₁ mids₂ f₁ f₂ p₁ p₂
    hact₁ hact₂ hp₁ hp₂
  obtain ⟨rss₁, cx₁, q₁, T₁, T₁', hrss₁, hst₁, hsch₁, hdom₁, hn₁, hsched₁, hfit₁, hT₁⟩ :=
    plainR_capChunk h₁ hact₁ hst hsch hr hp₁
  obtain ⟨rss₂, cx₂, q₂, T₂, T₂', hrss₂, hst₂, hsch₂, hdom₂, hn₂, hsched₂, hfit₂, hT₂⟩ :=
    plainR_capChunk h₂ hact₂ hst hsch hr hp₂
  have h22 := exec_fast100_ge hsch
  obtain ⟨hdim, hsm⟩ := fast100_dims _ _ hsch
  have hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0 := fun i hi =>
    (fast100_level _ _ hsch _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)).q_ne_zero
  have hn0 : sch.n ≠ 0 := by rw [hdim.1, hsm]; omega
  have hm : 14 ≤ (Setup.ofCircuit st).m := by show 14 ≤ st.m; omega
  obtain ⟨hRS, hcapS⟩ := repSchedAt_cap (spec := spec) hsch
  obtain ⟨hsum, hone, -⟩ := hcapS ℓ hℓ
  have hJR : preLig spec st + capRound sch.k0 (ℓ + 1) < repRounds false spec st := by
    rw [repRounds_plain hsch]
    have := capRound_lt (k0 := sch.k0) (show 2 ≤ sch.levels.size by omega) (show ℓ + 1 ≤ sch.levels.size by omega)
    omega
  have hlen : ∀ (c : Fin ((VR rd).k x) → Coin), (Claims.run sha512 P (VR rd) x ω c).out = .acc →
      (Claims.run sha512 P (VR rd) x ω c).msgs.length = (VR rd).k x := by
    intro c h
    have hk := accLast_VR rd x _ _ (by simp) (FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω c h)
    simpa using hk
  have hmsg : ∀ t ≤ forkRound spec st sch.k0 r ℓ,
      ((Claims.run sha512 P (VR rd) x ω c₁).msgs.map Prod.fst)[t]? =
        ((Claims.run sha512 P (VR rd) x ω c₂).msgs.map Prod.fst)[t]? := by
    intro t ht
    rw [run_depR rd P x ω c₁, run_depR rd P x ω c₂, hlen c₁ h₁, hlen c₂ h₂]
    rcases Nat.lt_or_ge t ((VR rd).k x) with htk | htk
    · rw [List.getElem?_map, List.getElem?_map, List.getElem?_range htk, Option.map_some, Option.map_some]
      congr 1
      refine depMsg_congr sha512 P x ω fun i hi => ?_
      by_cases hk : i < (VR rd).k x
      · simp only [Claims.coinSeq, hk, ↓reduceDIte]
        exact hc ⟨i, hk⟩ (show i < forkRound spec st sch.k0 r ℓ by omega)
      · simp only [Claims.coinSeq, hk, ↓reduceDIte]
    · rw [List.getElem?_eq_none, List.getElem?_eq_none] <;> simpa using htk
  have hrss : ∀ t, t + 1 ≤ forkRound spec st sch.k0 r ℓ → rss₁[t]? = rss₂[t]? := by
    intro t ht
    rw [mapM_getElem? hrss₁, mapM_getElem? hrss₂, List.getElem?_dropLast, List.getElem?_dropLast,
      List.length_tail, List.length_tail, List.length_map, List.length_map, hlen c₁ h₁, hlen c₂ h₂,
      List.getElem?_tail, List.getElem?_tail, hmsg (t + 1) ht]
  have hcut : ∀ (rss : List (List ByteArray)) (j : ℕ), j < repRounds false spec st →
      ((cutLens (repLens spec sts) rss).getD r [])[j]? = rss[r * repRounds false spec st + j]? := by
    intro rss j hj
    rw [eq_cons_cons (repLens_getElem? (spec := spec) hst (show 0 < 2 by omega))
      (repLens_getElem? (spec := spec) hst (show 1 < 2 by omega)), cutLens_two _ _ _ hr (by omega), hRS]
  have hrow : ∀ j ≤ preLig spec st + capRound sch.k0 (ℓ + 1),
      ((cutLens (repLens spec sts) rss₁).getD r [])[j]? = ((cutLens (repLens spec sts) rss₂).getD r [])[j]? := by
    intro j hj
    rw [hcut rss₁ j (by omega), hcut rss₂ j (by omega),
      hrss (r * repRounds false spec st + j) (by unfold forkRound; omega)]
  have hlen₁ : ((cutLens (repLens spec sts) rss₁).getD r []).length = repRounds false spec st := by
    rw [← hfit₁.sched, hsched₁, hRS]
  have hlen₂ : ((cutLens (repLens spec sts) rss₂).getD r []).length = repRounds false spec st := by
    rw [← hfit₂.sched, hsched₂, hRS]
  have hJS : preLig spec st + capRound sch.k0 (ℓ + 1) < (RecDeploy.repSchedAt spec st).length := by
    rw [hRS]; exact hJR
  have hone' : ((RecDeploy.repSchedAt spec st)[preLig spec st + capRound sch.k0 (ℓ + 1)]'hJS).2.1.size = 1 := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hJS, Option.getD_some] at hone
    exact hone
  have hsumJ : (((RecDeploy.repSchedAt spec st).take (preLig spec st + capRound sch.k0 (ℓ + 1) + 1)).map
      (·.2.1.size)).sum = ℓ + 1 := by
    rw [List.take_succ, List.map_append, List.sum_append_nat, hsum, List.getElem?_eq_getElem hJS]
    simp [hone']
  have htops : (q₁.ligerito.recursiveCaps.toList.map Array.toList)[ℓ]? =
      (q₂.ligerito.recursiveCaps.toList.map Array.toList)[ℓ]? := by
    rw [hfit₁.tops, hfit₂.tops, hsched₁, hsched₂]
    exact flatMap_zip_congr (fun (u : Flock.RecVStar.SchedRound) (rs : List ByteArray) =>
        (cutLens u.2.1.toList rs).map (List.map nodeOf))
      (fun u : Flock.RecVStar.SchedRound => u.2.1.size)
      (fun u b => by simp [cutLens_length]) _ _ _ (preLig spec st + capRound sch.k0 (ℓ + 1) + 1)
      (fun j hj => hrow j (by omega)) (by omega) ℓ (by rw [hsumJ]; omega)
  have hcap : q₁.ligerito.recursiveCaps[ℓ]! = q₂.ligerito.recursiveCaps[ℓ]! := caps_getElem!_eq htops
  have hfc : ∀ (cx : RepCtx) (q : Proof) (T : F128), cx.st = Setup.ofCircuit st → cx.sch = sch →
      cx.dom = (spec.domain spec.tables[0]! r).toUTF8 → cx.n = 2 + st.regions.size * spec.points →
      (chunksOf cx q T).getD (preLig spec st + capRound sch.k0 (ℓ + 1)) [] =
        kFr q.ligerito.messages[capMsg sch.k0 ℓ]! ++ [capFr q.ligerito.recursiveCaps[ℓ]!, capSq sch ℓ] := by
    intro cx q T h1 h2 h3 h4
    have hJf : (3 + ((Setup.ofCircuit st).m - 6)) + (3 + ((Setup.ofCircuit st).fold.kLog - 6)) +
        (2 + st.regions.size * spec.points + 1) = preLig spec st := by
      unfold preLig
      rw [Nat.mul_comm st.regions.size]
      rfl
    unfold chunksOf
    rw [h1, h2, h3, h4, ← hJf]
    exact repFrames_capChunk _ sch _ _ q T hm (by omega) hn0 hq hℓ
  have hJ1 : preLig spec st + capRound sch.k0 (ℓ + 1) < ((cutLens (repLens spec sts) rss₁).getD r []).length := by
    rw [hlen₁]; exact hJR
  have hJ2 : preLig spec st + capRound sch.k0 (ℓ + 1) < ((cutLens (repLens spec sts) rss₂).getD r []).length := by
    rw [hlen₂]; exact hJR
  have hel₁ := hfit₁.elems _ hJ1
  have hel₂ := hfit₂.elems _ hJ2
  rw [hsched₁, hfc cx₁ q₁ T₁' hst₁ hsch₁ hdom₁ hn₁, vals_capChunk] at hel₁
  rw [hsched₂, hfc cx₂ q₂ T₂' hst₂ hsch₂ hdom₂ hn₂, vals_capChunk] at hel₂
  have hrJ : ((cutLens (repLens spec sts) rss₁).getD r []).getD (preLig spec st + capRound sch.k0 (ℓ + 1)) [] =
      ((cutLens (repLens spec sts) rss₂).getD r []).getD (preLig spec st + capRound sch.k0 (ℓ + 1)) [] := by
    have hh := hrow _ le_rfl
    generalize (cutLens (repLens spec sts) rss₁).getD r [] = A at hh ⊢
    generalize (cutLens (repLens spec sts) rss₂).getD r [] = B at hh ⊢
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, hh]
  rw [hrJ] at hel₁
  have hvv := FlockDeploy.RowsAt.vals_eq hel₁ hel₂ (by simp only [List.length_cons, List.length_nil])
  have hkf : kFr q₁.ligerito.messages[capMsg sch.k0 ℓ]! = kFr q₂.ligerito.messages[capMsg sch.k0 ℓ]! := by
    simp only [List.cons.injEq, and_true] at hvv
    obtain ⟨e1, e2, e3, e4⟩ := hvv
    simp only [kFr, e1, e2, e3, e4]
  obtain ⟨a, b, s₁, hF, hs₁, ha⟩ := repFrames_capSplit (Setup.ofCircuit st) sch
    (spec.domain spec.tables[0]! r).toUTF8 (2 + st.regions.size * spec.points) p₁ T₁ hm (by omega) hn0 hq hℓ
  obtain ⟨a', b', s₂, hF', hs₂, ha'⟩ := repFrames_capSplit (Setup.ofCircuit st) sch
    (spec.domain spec.tables[0]! r).toUTF8 (2 + st.regions.size * spec.points) p₂ T₂ hm (by omega) hn0 hq hℓ
  have hcnt : a.countP closes = preLig spec st + capRound sch.k0 (ℓ + 1) := by
    rw [ha]
    unfold preLig
    rw [Nat.mul_comm st.regions.size]
    rfl
  refine capFr_eq_of_chunk (repFrames_hds (FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ hp₁)
    (FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ hp₂) _ T₁ T₂) hF hF' hs₁ hs₂ (ha.trans ha'.symm) ?_
  rw [hcnt, hT₁ _ hJR, hT₂ _ hJR, hfc cx₁ q₁ T₁' hst₁ hsch₁ hdom₁ hn₁, hfc cx₂ q₂ T₂' hst₂ hsch₂ hdom₂ hn₂, hcap,
    hkf]

/-! ## The table bound at the deployed reader -/

/-- **The table bound at the deployed reader's plain record, from `PlainRdInR` alone**
(`plainR_tableBoundAbortLin_of_capPrefix` at `capPrefixR`). -/
theorem plainR_tableBoundAbortLin_R (rd : Claims.Msg → Option FinalP)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc sha512 (VR rd) x → PlainRdInR rd x) :
    Claims.Recursion.TableBoundAbortLin sha512 (deployPlainR Readable.relation rd) (wordLen false) (wide false)
      CpDep :=
  plainR_tableBoundAbortLin_of_capPrefix rd hIn (capPrefixR rd)

/-- **`KSFlockLin` at the deployed reader's trimmed plain record, from `PlainRdInR` alone**
(`plainR_KSFlockLin_of_capPrefix` at `capPrefixR`). -/
theorem plainR_KSFlockLin_R (rd : Claims.Msg → Option FinalP)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc sha512 (VR rd) x → PlainRdInR rd x) :
    Claims.Recursion.KSFlockLin sha512 (deployPlainTrimR Readable.relation rd) CpDep :=
  plainR_KSFlockLin_of_capPrefix rd hIn (capPrefixR rd)

end FlockDeploy.KS
