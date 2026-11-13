module

public import Proofs.Properties.Verity.SampledTwo.SessionMsgs
public import Proofs.Flock.Claims.Complete.Zk.Run

/-!
# A session's replaying prover against the masked verifier

`SessionMsgs.replayProver` sends a session's messages off the play's case whatever the verifier answers. Against
`FlockDeploy.V true` such a prover is accepted wherever the honest masked prover's run is (`run_acc_of_zk`), with its
messages in place of the honest run's:

* **A prover that sends a fixed list** `sends ++ [f]` (`run_acc_replay`): at the statement's setup, the sends of tag
  `0` fitting the verifier's rounds and starting with a link message, the last message decoding, and the masked
  verdict and the checks passing on it, the run accepts.
* **The replaying prover's next message** (`replayProver_next`): on a play whose case gives template `t`'s session's
  messages `ms`, its message after a history of length `i` is `ms`'s `i`-th.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionReplay

open Flock FlockDeploy FlockComplete

/-- **A prover that sends `sends ++ [f]` is accepted** at the coins `r`, where the statement sets up (against `H` as
at SHA-512), the sends are of tag `0`, fit the verifier's rounds and start with a link message, the last message
decodes, and the masked verdict and the checks pass on it. -/
theorem run_acc_replay (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (sends : List Claims.Msg) (f : Claims.Msg)
    (hP : ∀ hist : List (Claims.Msg × Claims.Msg), hist.length ≤ sends.length →
      ((P.next x hist ω).eval H).1 = (sends ++ [f]).getD hist.length [])
    (r : Fin ((V true).k x) → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    {link : List ByteArray × List ByteArray} {F : Final} (hs : setupOf true x = some (s, spec, sts))
    (hsu : setupWith (Claims.hOf H) s.read = .ok (spec, sts))
    (hk : sends.length + 1 ≤ (V true).k x)
    (htag : ∀ m ∈ sends, ∃ t, m = 0 :: t)
    (hne : sends ≠ [])
    (hlink : linkOf (sends.headD []) = some link)
    (hfin : finalOf f = some F)
    (hv : verdictWith true (Claims.hOf H) spec (tabsOf sts) F.record F.proofs = .ok ())
    (hc : checks true s spec x link (Claims.coinSeq r 0)
      (midsOf (sends ++ [f]) ((List.range (sends.length + 1)).map (Claims.coinSeq r))) F = .ok ()) :
    (Claims.run H P (V true) x ω r).out = .acc := by
  let seq : ℕ → Claims.Msg := fun j => (sends ++ [f]).getD j []
  let resp : ℕ → Claims.Msg := fun j => (Claims.coinSeq r j).bytes
  have hseq : ∀ j (hj : j < sends.length), seq j = sends[j] := by
    intro j hj
    simp only [seq, List.getD_eq_getElem?_getD, List.getElem?_append_left hj, List.getElem?_eq_getElem hj,
      Option.getD_some]
  have hall : (List.range (sends.length + 1)).map seq = sends ++ [f] := by
    have := map_getD_range (sends ++ [f])
    rw [List.length_append, List.length_singleton] at this
    exact this
  unfold Claims.run
  refine stepWith_acc H P x ω _ seq resp sends.length ?_ ?_ ?_ _ hk _
  · intro j hj hist hh
    have hl : hist.length = j := by
      simpa using congrArg List.length hh
    rw [hP hist (by omega), hl]
  · intro j hj
    obtain ⟨t, ht⟩ := htag sends[j] (List.getElem_mem hj)
    have ha : ((List.range (j + 1)).map seq).getLast? = some (0 :: t) := by
      simp only [List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel,
        Option.map_some]
      rw [hseq j hj, ht]
    have hc' : ((List.range ((List.range (j + 1)).map seq).length).map (Claims.coinSeq r)).getLast? =
        some (Claims.coinSeq r j) := by
      simp only [List.length_map, List.length_range, List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero,
        ite_false, Nat.add_sub_cancel, Option.map_some]
    exact (FlockDeploy.next_coin H (act_coin_any hs ha hc')).symm ▸ rfl
  · show ((FlockDeploy.next true x ((List.range (sends.length + 1)).map seq)
      ((List.range ((List.range (sends.length + 1)).map seq).length).map (Claims.coinSeq r))).eval H).1 = .acc
    rw [hall, List.length_append, List.length_singleton]
    obtain ⟨a₁, rest, hms1⟩ : ∃ a₁ rest, sends = a₁ :: rest := List.exists_cons_of_ne_nil hne
    have hl : (sends ++ [f]).getLast? = some f := by simp
    have hcs : (List.range (sends.length + 1)).map (Claims.coinSeq r) =
        Claims.coinSeq r 0 :: (List.range sends.length).map (fun j => Claims.coinSeq r (j + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      rfl
    have hlink' : linkOf a₁ = some link := by rw [hms1] at hlink; exact hlink
    rw [hcs]
    rw [hms1, List.cons_append] at hl ⊢
    have had := act_final_any (c₁ := Claims.coinSeq r 0) (crest := (List.range (a₁ :: rest).length).map
      (fun j => Claims.coinSeq r (j + 1))) hs hl hfin hlink'
    rw [FlockDeploy.next_decide H had]
    refine ⟨spec, sts, hsu, hv, ?_⟩
    rw [hcs, hms1, List.cons_append] at hc
    exact hc

/-- **The replaying prover's next message**: on a play whose case `k` gives template `t`'s session's messages `ms`,
after a history of length `i`, `ms`'s `i`-th message (none past its end). -/
theorem replayProver_next (H : Claims.Msg → Claims.Msg) (C : CaseOf) (o : Verity.Exchange) (t : ℕ) (x : Claims.Msg)
    {k : Case} {ms : List Claims.Msg} (hC : C o = some k) (hm : SessionMsgs.msgsAtCase k t = some ms)
    (hist : List (Claims.Msg × Claims.Msg)) :
    (((SessionMsgs.replayProver C o t).next x hist ()).eval H).1 = ms.getD hist.length [] := by
  simp only [SessionMsgs.replayProver, Messages.proverOf, Claims.OC.eval, hC, Option.bind_some,
    SessionMsgs.msgAtCase, hm, List.getD_eq_getElem?_getD]

end Verity.SampledTwo.SessionReplay
