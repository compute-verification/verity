import Proofs.Flock.Claims.Recursion.VerdictRowDCoin

/-!
# Each rep stream's round count is the schedule's

The record a decision of `nextR` hands the verdict is the reader's (`recJ`), not the prover's JSON: rep `k`'s stream has
`repLens` rounds, round `j` counting its schedule's coins (`recStreams`). So the decoded session's rep stream has the
coins the live coins alone fix (`canonRounds`, `coins_of_decode`), stream by stream, not just in total. This is what the
shipped recursion enforces of the inner record by check 5 (`RecVStar.innerCommits`/`repCoins` hold each rep stream's
round count and coins to the coin server's, and `RecVStar.checkSchedule` the coin server's to the schedule).

* `DecAt.rounds_canon`: rep `r` of table `i`'s coin rounds are `canonRounds` at rep `2 i + r`.
* `DecAt.roundCount`: its round count is the schedule's (`repLens`).
* `DecAt.hz`: row L's coin tie (`TargetZ`'s form): the stream's round-2 coin is the live one the reader read `z` from.
* `DecAt.hsz_of_sched`: row R's `hsz`, where the schedule's coin counts are `roundSizes` at the table's shape.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP repLens cutLens repCtx)
open Flock.RecAlgebra (NoEmpty)
open FlockVBridge.Family (roundSizes)
open FlockSoundness.Discharge.Recursion (HClass)

theorem cutLens_getD_getD {α : Type} (d : α) : ∀ (ns : List ℕ) (l : List α) (k j : ℕ),
    ((cutLens ns l).getD k []).getD j d = if j < ns.getD k 0 then l.getD ((ns.take k).sum + j) d else d
  | [], l, k, j => by simp [cutLens]
  | n :: ns, l, 0, j => by
    simp only [cutLens, List.getD_cons_zero, List.take_zero, List.sum_nil, Nat.zero_add]
    by_cases h : j < n
    · rw [if_pos h]
      simp [List.getD_eq_getElem?_getD, List.getElem?_take, h]
    · rw [if_neg h]
      simp [List.getD_eq_getElem?_getD, List.getElem?_take, h]
  | n :: ns, l, k + 1, j => by
    simp only [cutLens, List.getD_cons_succ, List.take_succ_cons, List.sum_cons]
    rw [cutLens_getD_getD d ns (l.drop n) k j]
    split
    · simp [List.getD_eq_getElem?_getD, List.getElem?_drop, Nat.add_assoc]
    · rfl

theorem vals_getD (c : Coin) {n : ℕ} (hn : n ≠ 0) : (c.vals n).getD 0 0 = c.at 0 := by
  unfold FlockDeploy.Coin.vals Array.getD
  rw [dif_pos (by simp; omega)]
  simp

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ)

/-- **Rep `r` of table `i`'s coin rounds are the live coins'**: the schedule's round count, round `j` the first `n_j`
(its schedule's) values of the coin at the rep's offset plus `j` (`canonRounds`), at rep streams of distinct names. -/
theorem rounds_canon (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (r : Fin 2) :
    ((D.files cls i).stream r).rounds.map (·.coins) = canonRounds D.spec D.sts cs.tail.dropLast (2 * i + r) := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, -, -, -, -, -, -, hpf, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hsp' : D.spec = spec := by rw [hsp, hd]
  have hsts' : D.sts = sts := by rw [hsts, hd]
  have hdec := D.decode
  rw [hd, hsp', hsts'] at hdec
  simp only [FlockDeploy.jAt] at hdec
  have hk : 2 * i + r.val < 2 * sts.size := by rw [← hsts']; omega
  rw [hsp'] at hnd hti
  obtain ⟨hks, e⟩ := Claims.Bridge.Custody.repStreams_getElem spec hti r.isLt
  have hs : (D.files cls i).stream r =
      (D.sess.streams.find? (·.name == spec.repStreams[2 * i + r.val].2.2)).getD default := by
    rw [e]
    show (D.sess.streams.find? fun y => y.name == toString D.spec.tables[i]! ++ toString "/rep" ++
      toString (r : ℕ)).getD default = _
    rw [hsp']
  rw [hs, hsp', hsts']
  exact coins_of_decode _ hnd (readPs_blocks hpf) hdec hk hks

/-- **Each rep stream's round count is the schedule's** (`repLens`), stream by stream. -/
theorem roundCount (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (r : Fin 2) :
    ((D.files cls i).stream r).rounds.size = (repLens D.spec D.sts).getD (2 * i + r) 0 := by
  have h := congrArg Array.size (D.rounds_canon cls i hnd hi hti r)
  simpa [canonRounds] using h

/-- **Row L's coin tie**: rep `r`'s round-2 coin, as the session records it, is the live coin the reader took the
rep's zerocheck `z` from. -/
theorem hz (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (r : Fin 2) :
    (((D.files cls i).stream r).rounds[2]!).coins.getD 0 0 = (D.ctx (2 * i + r)).z := by
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hR := D.rounds_canon cls i hnd hi hti r
  have hsz := D.roundCount cls i hnd hi hti r
  have hne : NoEmpty (((D.files cls i).stream r).rounds.map (·.coins)) := D.noEmpty cls (fun _ => #[]) hti r
  show _ = (((cutLens (repLens D.d.spec D.d.sts) cs.tail.dropLast).getD (2 * i + r) []).getD 2 default).at 0
  rw [cutLens_getD_getD, ← hsp, ← hsts]
  generalize ((D.files cls i).stream r).rounds = R at hR hsz hne
  split
  · rename_i h2
    have h2R : 2 < R.size := by omega
    have h2' : 2 < (R.map (·.coins)).size := by rw [Array.size_map]; exact h2R
    have hmem : (R.map (·.coins))[2]! ∈ R.map (·.coins) := by
      rw [getElem!_pos _ 2 h2']
      exact Array.getElem_mem h2'
    have hX := hne _ hmem
    have e : (R[2]!).coins = (R.map (·.coins))[2]! := by
      rw [getElem!_pos (R.map (·.coins)) 2 h2', Array.getElem_map, getElem!_pos R 2 h2R]
    have hc : (canonRounds D.spec D.sts cs.tail.dropLast (2 * i + r))[2]! =
        (cs.tail.dropLast.getD (((repLens D.spec D.sts).take (2 * i + r)).sum + 2) default).vals
          ((RecDeploy.repSchedAt D.spec D.sts[(2 * i + r) / 2]!).getD 2 default).1 := by
      rw [getElem!_pos _ 2 (by rw [← hR, Array.size_map]; omega)]
      simp [canonRounds]
    rw [hR, hc] at hX
    rw [e, hR, hc]
    exact vals_getD _ (by simpa [FlockDeploy.Coin.vals] using hX)
  · have h2 : ¬ 2 < R.size := by omega
    rw [getElem!_neg R 2 h2]
    rfl

/-- **Row R's `hsz`**: rep `r`'s coin rounds have the shape's sizes, where the schedule's coin counts at table `i`
are `roundSizes` at its shape (`hsched`, an equation between the verifier's schedule and its algebra). -/
theorem hsz_of_sched (acc : ℕ → Array F128) (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup)
    (hi : i < D.sts.size) (hti : i < D.spec.tables.size) (r : Fin 2)
    (hsched : (RecDeploy.repSchedAt D.spec D.sts[i]!).map (·.1) = roundSizes (D.sh cls i)) :
    (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i) := by
  have hR := D.rounds_canon cls i hnd hi hti r
  have hd : (2 * i + (r : ℕ)) / 2 = i := by have := r.isLt; omega
  have hL : (repLens D.spec D.sts).getD (2 * i + r) 0 = (RecDeploy.repSchedAt D.spec D.sts[i]!).length := by
    have hk : 2 * i + r < 2 * D.sts.size := by omega
    simp [repLens, List.getD_eq_getElem?_getD, hk, hd]
  change (((D.files cls i).stream r).rounds.map (·.coins)).toList.map Array.size = _
  rw [hR, ← hsched]
  apply List.ext_getElem
  · simp only [List.length_map, Array.length_toList, canonRounds, Array.size_map, Array.size_range]
    exact hL
  · intro j h₁ h₂
    have hj : j < (RecDeploy.repSchedAt D.spec D.sts[i]!).length := by simpa using h₂
    simp only [canonRounds, hd]
    simp [FlockDeploy.Coin.vals, List.getD_eq_getElem?_getD, hj]

end DecAt

end RecDeploy.Rows
