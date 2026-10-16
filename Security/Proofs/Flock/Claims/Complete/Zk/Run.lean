import Proofs.Flock.Claims.Complete.Zk.Honest
import Proofs.Flock.Claims.Complete.Run

/-!
# The honest run against the masked verifier

`run_acc_of` at `V true` for `honestPZk`: the masked verifier's side is the plain one's (`act`, `next` read `zk` only
through `setupOf` and the verdict), so `act_coin` and `act_final` hold at any `zk` (`act_coin_any`, `act_final_any`),
and the prover's side is `feed_sim` and `stepWith_acc` as for `honestP`.
-/

namespace FlockComplete

open Flock FlockDeploy

/-- A last message of tag `0` is answered by its coin, at either verifier. -/
theorem act_coin_any {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Flock.Stmt} {t : Claims.Msg} {c : Coin} (hs : setupOf zk x = some (s, spec, sts))
    (ha : as.getLast? = some (0 :: t)) (hc : cs.getLast? = some c) : act zk x as cs = .coin c := by
  unfold act
  rw [hs, ha, hc]
  rfl

/-- The last message is decided, at either verifier. -/
theorem act_final_any {zk : Bool} {x : Claims.Msg} {a₁ : Claims.Msg} {rest : List Claims.Msg} {c₁ : Coin}
    {crest : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {f : Claims.Msg} {F : Final}
    {link : List ByteArray × List ByteArray} (hs : setupOf zk x = some (s, spec, sts))
    (hl : (a₁ :: rest).getLast? = some f) (hf : finalOf f = some F) (hlink : linkOf a₁ = some link) :
    act zk x (a₁ :: rest) (c₁ :: crest) =
      .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) F := by
  obtain ⟨b, rfl⟩ : ∃ b, f = 1 :: b := by
    unfold finalOf at hf
    split at hf
    · exact ⟨_, rfl⟩
    · cases hf
  unfold act
  rw [hs]
  dsimp only
  rw [hl]
  split
  · rename_i h _
    simp only [Option.some.injEq, List.cons.injEq] at h
    exact absurd h.1 (by decide)
  · rename_i h _
    simp only [Option.some.injEq] at h
    subst h
    rw [hf]
    dsimp only
    rw [hlink]
  · rename_i h
    cases h

/-- **The honest masked run** at the coins `r`: the program's sends, each answered by its coin's bytes, and its last
message. -/
def honestRunZk (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) {k : ℕ}
    (r : Fin k → Coin) : List Claims.Msg × Claims.Msg :=
  (progZk x w ω).sim H (fun j => (Claims.coinSeq r j).bytes) 0

/-- **The honest masked prover is accepted** at the coins `r` when the statement sets up, the honest run's sends are of
tag `0`, fit the verifier's rounds, and start with a link message, its last message decodes, and the masked verdict and
the checks pass on it. -/
theorem run_acc_of_zk (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (ω : TapeZk)
    (r : Fin ((V true).k x) → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    {link : List ByteArray × List ByteArray} {F : Final} (hs : setupOf true x = some (s, spec, sts))
    (hk : (honestRunZk H x w ω r).1.length + 1 ≤ (V true).k x)
    (htag : ∀ m ∈ (honestRunZk H x w ω r).1, ∃ t, m = 0 :: t)
    (hne : (honestRunZk H x w ω r).1 ≠ [])
    (hlink : linkOf ((honestRunZk H x w ω r).1.headD []) = some link)
    (hfin : finalOf (honestRunZk H x w ω r).2 = some F)
    (hv : verdictWith true (Claims.hOf H) spec (tabsOf sts) F.record F.proofs = .ok ())
    (hc : checks true s spec x link (Claims.coinSeq r 0)
      (midsOf ((honestRunZk H x w ω r).1 ++ [(honestRunZk H x w ω r).2])
        ((List.range ((honestRunZk H x w ω r).1.length + 1)).map (Claims.coinSeq r))) F = .ok ()) :
    (Claims.run H (honestPZk w) (V true) x ω r).out = .acc := by
  set ms := (honestRunZk H x w ω r).1 with hms
  set f := (honestRunZk H x w ω r).2 with hf
  let seq : ℕ → Claims.Msg := fun j => (ms ++ [f]).getD j []
  let resp : ℕ → Claims.Msg := fun j => (Claims.coinSeq r j).bytes
  have hseq : ∀ j (hj : j < ms.length), seq j = ms[j] := by
    intro j hj
    simp only [seq, List.getD_eq_getElem?_getD, List.getElem?_append_left hj, List.getElem?_eq_getElem hj,
      Option.getD_some]
  have hall : (List.range (ms.length + 1)).map seq = ms ++ [f] := by
    have := map_getD_range (ms ++ [f])
    rw [List.length_append, List.length_singleton] at this
    exact this
  unfold Claims.run
  refine stepWith_acc H (honestPZk w) x ω _ seq resp ms.length ?_ ?_ ?_ _ hk _
  · intro j hj hist hh
    show ((feed (progZk x w ω) (hist.map Prod.snd)).eval H).1 = _
    have := feed_sim H resp (progZk x w ω) 0 j hj
    simp only [Nat.zero_add] at this
    rw [hh]
    exact this
  · intro j hj
    obtain ⟨t, ht⟩ := htag ms[j] (List.getElem_mem hj)
    have ha : ((List.range (j + 1)).map seq).getLast? = some (0 :: t) := by
      simp only [List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel,
        Option.map_some]
      rw [hseq j hj, ht]
    have hc' : ((List.range ((List.range (j + 1)).map seq).length).map (Claims.coinSeq r)).getLast? =
        some (Claims.coinSeq r j) := by
      simp only [List.length_map, List.length_range, List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero,
        ite_false, Nat.add_sub_cancel, Option.map_some]
    exact (next_coin H (act_coin_any hs ha hc')).symm ▸ rfl
  · show ((next true x ((List.range (ms.length + 1)).map seq)
      ((List.range ((List.range (ms.length + 1)).map seq).length).map (Claims.coinSeq r))).eval H).1 = .acc
    rw [hall, List.length_append, List.length_singleton]
    obtain ⟨a₁, rest, hms1⟩ : ∃ a₁ rest, ms = a₁ :: rest := List.exists_cons_of_ne_nil hne
    have hl : (ms ++ [f]).getLast? = some f := by simp
    have hcs : (List.range (ms.length + 1)).map (Claims.coinSeq r) =
        Claims.coinSeq r 0 :: (List.range ms.length).map (fun j => Claims.coinSeq r (j + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      rfl
    have hlink' : linkOf a₁ = some link := by rw [hms1] at hlink; exact hlink
    rw [hcs]
    rw [hms1, List.cons_append] at hl ⊢
    have had := act_final_any (c₁ := Claims.coinSeq r 0) (crest := (List.range (a₁ :: rest).length).map
      (fun j => Claims.coinSeq r (j + 1))) hs hl hfin hlink'
    rw [next_decide H had]
    refine ⟨hv, ?_⟩
    rw [hcs, hms1, List.cons_append] at hc
    exact hc

end FlockComplete
