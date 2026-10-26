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
  simp [FlockC3.Compose.repHead] <;> omega

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

end FlockDeploy.KS
