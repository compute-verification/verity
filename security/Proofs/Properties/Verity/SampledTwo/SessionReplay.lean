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
* **An accepted masked session's rep streams split evenly** (`streams_even`): where the deployed masked verdict
  accepts a record at the statement's setup, each table has two rep streams, each of the table's `repRounds true`
  rounds. A record whose reps' rounds are split any other way is refused by the verdict itself.
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

/-- **A prover that sends a session's messages is accepted** at the coins `r`, where the statement sets up (against `H`
as at SHA-512), the messages fit the verifier's rounds, the record's roots, public words, statement, text and proof
files fit their length words, and the masked verdict and the checks pass on the record's link message, its rounds and
the last message. -/
theorem run_acc_sessionMsgs (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (ω : Ω)
    {spec : SessionSpec} {record : String} {proofs : Array ByteArray} {ms : List Claims.Msg} {lv : Live}
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms) (hlv : (liveOf spec record).toOption = some lv)
    (hP : ∀ hist : List (Claims.Msg × Claims.Msg), hist.length + 1 ≤ ms.length →
      ((P.next x hist ω).eval H).1 = ms.getD hist.length [])
    (r : Fin ((V true).k x) → Coin) {s : Ins} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (hsu : setupWith (Claims.hOf H) s.read = .ok (spec, sts))
    (hk : ms.length ≤ (V true).k x)
    (hr : lv.rootB.size < 2 ^ 64) (hp : lv.publics.size < 2 ^ 64) (hrb : ∀ b ∈ lv.rootB.toList, b.size < 2 ^ 64)
    (hpb : ∀ b ∈ lv.publics.toList, b.size < 2 ^ 64) (hxs : x.length < 2 ^ 64) (hrec : record.toUTF8.size < 2 ^ 64)
    (hps : proofs.size < 2 ^ 64) (hb : ∀ b ∈ proofs, b.size < 2 ^ 64)
    (hv : verdictWith true (Claims.hOf H) spec (tabsOf sts) record proofs = .ok ())
    (hc : checks true s spec x (lv.rootB.toList, lv.publics.toList) (Claims.coinSeq r 0)
      (midsOf ms ((List.range ms.length).map (Claims.coinSeq r))) ⟨x, record, proofs⟩ = .ok ()) :
    (Claims.run H P (V true) x ω r).out = .acc := by
  obtain ⟨lv', mids, hlv', hmids, hms⟩ := SessionMsgs.sessionMsgs_some hm
  rw [hlv] at hlv'
  have e := Option.some.inj hlv'
  subst e
  obtain ⟨a, ha, hla⟩ := SessionMsgs.linkOf_sessionMsgs hm hlv hr hp hrb hpb
  obtain ⟨b, hb', hfb⟩ := SessionMsgs.finalOf_sessionMsgs hm hxs hrec hps hb
  have hms' :
      ms = ((0 :: linkBody lv.rootB.toList lv.publics.toList) :: mids) ++ [Final.encode ⟨x, record, proofs⟩] :=
    hms
  have hlen : ms.length = ((0 :: linkBody lv.rootB.toList lv.publics.toList) :: mids).length + 1 := by
    rw [hms', List.length_append, List.length_singleton]
  have hah : a = 0 :: linkBody lv.rootB.toList lv.publics.toList := by
    rw [hms', List.cons_append, List.head?_cons, Option.mem_def, Option.some.injEq] at ha
    exact ha.symm
  have hbf : b = Final.encode ⟨x, record, proofs⟩ := by
    rw [hms', SessionMsgs.getLast?_cons_concat, Option.mem_def, Option.some.injEq] at hb'
    exact hb'.symm
  subst hah hbf
  refine run_acc_replay H P x ω ((0 :: linkBody lv.rootB.toList lv.publics.toList) :: mids)
    (Final.encode ⟨x, record, proofs⟩) ?_ r hs hsu (by omega) ?_ (List.cons_ne_nil _ _) hla hfb hv ?_
  · intro hist hl
    rw [hP hist (by omega), hms']
  · intro m hm'
    rcases List.mem_cons.1 hm' with rfl | hm'
    · exact ⟨_, rfl⟩
    · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hm'
      obtain ⟨hl, hall⟩ := SessionMsgs.mapM_some _ hmids
      obtain ⟨c, -, hc'⟩ := Option.map_eq_some_iff.1 (hall i (by omega) hi)
      exact ⟨_, hc'.symm⟩
  · rw [hlen, hms'] at hc
    exact hc

/-- **The replaying prover is accepted** at the coins `r` on a play whose case gives template `t`'s session's messages
at the statement `x`'s setup, under `run_acc_sessionMsgs`'s conditions. -/
theorem run_acc_replayProver (H : Claims.Msg → Claims.Msg) (C : CaseOf) (o : Verity.Exchange) (t : ℕ)
    (x : Claims.Msg)
    {k : Case} {spec : SessionSpec} {record : String} {proofs : Array ByteArray} {ms : List Claims.Msg} {lv : Live}
    (hC : C o = some k) (hmc : SessionMsgs.msgsAtCase k t = some ms)
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms) (hlv : (liveOf spec record).toOption = some lv)
    (r : Fin ((V true).k x) → Coin) {s : Ins} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (hsu : setupWith (Claims.hOf H) s.read = .ok (spec, sts))
    (hk : ms.length ≤ (V true).k x)
    (hr : lv.rootB.size < 2 ^ 64) (hp : lv.publics.size < 2 ^ 64) (hrb : ∀ b ∈ lv.rootB.toList, b.size < 2 ^ 64)
    (hpb : ∀ b ∈ lv.publics.toList, b.size < 2 ^ 64) (hxs : x.length < 2 ^ 64) (hrec : record.toUTF8.size < 2 ^ 64)
    (hps : proofs.size < 2 ^ 64) (hb : ∀ b ∈ proofs, b.size < 2 ^ 64)
    (hv : verdictWith true (Claims.hOf H) spec (tabsOf sts) record proofs = .ok ())
    (hc : checks true s spec x (lv.rootB.toList, lv.publics.toList) (Claims.coinSeq r 0)
      (midsOf ms ((List.range ms.length).map (Claims.coinSeq r))) ⟨x, record, proofs⟩ = .ok ()) :
    (Claims.run H (SessionMsgs.replayProver C o t) (V true) x () r).out = .acc :=
  run_acc_sessionMsgs H _ x () hm hlv (fun hist _ => replayProver_next H C o t x hC hmc hist) r hs hsu hk hr hp hrb
    hpb hxs hrec hps hb hv hc

/-- **An accepted masked session's rep streams split evenly**: at the statement's setup, where the deployed masked
verdict (`Flock.Zk.verify`) accepts the record and `liveOf` reads it, there are two rep streams per table, and stream
`k` (table `k / 2`'s rep `k % 2`) has `repRounds true spec sts[k / 2]` rounds (`FlockDeploy.streams_size`, the verdict
at SHA-512 by `FlockDeploy.verdictWith_sha512`). So the verdict refuses a record whose reps' rounds are split any
other way, and the form needs no refusal of its own for it. -/
theorem streams_even {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hs : FlockDeploy.setupOf true x = some (s, spec, sts))
    (hv : Flock.Zk.verify spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].size = FlockDeploy.repRounds true spec sts[k / 2]! := by
  have hw : verdictWith true (Claims.hOf FlockDeploy.sha512) spec (tabsOf sts) record proofs = .ok () :=
    (FlockDeploy.verdictWith_sha512 hs record proofs).trans hv
  exact FlockDeploy.streams_size true (Flock.Hashes.ofSha512 (Claims.hOf FlockDeploy.sha512)) hw hl

end Verity.SampledTwo.SessionReplay
