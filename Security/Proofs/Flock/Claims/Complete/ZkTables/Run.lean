import Proofs.Flock.Claims.Complete.ZkTables.Honest
import Proofs.Flock.Claims.Complete.Zk.Sim

/-!
# The honest run against the masked verifier, at any number of tables

`run_acc_of_zk` and `honestRunZk_eq` for `honestP`: the verifier's side is `act_coin_any` and `act_final_any`, the
prover's `feed_sim` and `stepWith_acc`, and the run is the pure masked session (`sessionPZk_sim`) on each table's share
of the witness (`sessionS`, `honestRun_eq`).
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- **The honest masked run** at the coins `r`: the program's sends, each answered by its coin's bytes, and its last
message. -/
def honestRun (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) {k : ℕ}
    (r : Fin k → Coin) : List Claims.Msg × Claims.Msg :=
  (prog x w ω).sim H (fun j => (Claims.coinSeq r j).bytes) 0

/-- **The honest masked prover is accepted** at the coins `r` when the statement sets up, against `H` as at SHA-512,
the honest run's sends are of tag `0`, fit the verifier's rounds, and start with a link message, its last message
decodes, and the masked verdict and the checks pass on it. -/
theorem run_acc_of (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (ω : TapeZk)
    (r : Fin ((V true).k x) → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    {link : List ByteArray × List ByteArray} {F : Final} (hs : setupOf true x = some (s, spec, sts))
    (hsu : setupWith (Claims.hOf H) s.read = .ok (spec, sts))
    (hk : (honestRun H x w ω r).1.length + 1 ≤ (V true).k x)
    (htag : ∀ m ∈ (honestRun H x w ω r).1, ∃ t, m = 0 :: t)
    (hne : (honestRun H x w ω r).1 ≠ [])
    (hlink : linkOf ((honestRun H x w ω r).1.headD []) = some link)
    (hfin : finalOf (honestRun H x w ω r).2 = some F)
    (hv : verdictWith true (Claims.hOf H) spec (tabsOf sts) F.record F.proofs = .ok ())
    (hc : checks true s spec x link (Claims.coinSeq r 0)
      (midsOf ((honestRun H x w ω r).1 ++ [(honestRun H x w ω r).2])
        ((List.range ((honestRun H x w ω r).1.length + 1)).map (Claims.coinSeq r))) F = .ok ()) :
    (Claims.run H (honestP w) (V true) x ω r).out = .acc := by
  set ms := (honestRun H x w ω r).1 with hms
  set f := (honestRun H x w ω r).2 with hf
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
  refine stepWith_acc H (honestP w) x ω _ seq resp ms.length ?_ ?_ ?_ _ hk _
  · intro j hj hist hh
    show ((feed (prog x w ω) (hist.map Prod.snd)).eval H).1 = _
    have := feed_sim H resp (prog x w ω) 0 j hj
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
    refine ⟨spec, sts, hsu, hv, ?_⟩
    rw [hcs, hms1, List.cons_append] at hc
    exact hc

/-- **The pure honest masked session** at the replies `resp`, on each table's share of the witness: its verdict (the
record and the proof files) and its sends' bodies. -/
def sessionS (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt)
    (w : Claims.Flock.Wit) (ω : TapeZk) : Except String (String × Array ByteArray) × Array Claims.Msg :=
  (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts) (traces sts w) ω).go #[]

/-- **The honest masked run is the pure masked session's**: its sends are the session's bodies, each of tag `0`, and its
last message the session's record and proof files after the statement. -/
theorem honestRun_eq (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) {k : ℕ}
    (r : Fin k → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (rec, pfs), bs)) :
    honestRun H x w ω r = (bs.toList.map (List.cons 0), Final.encode ⟨x, rec, pfs⟩) := by
  obtain ⟨bs', h1, h2⟩ := sessionPZk_sim H (fun j => (Claims.coinSeq r j).bytes) spec (tabsOf sts) (traces sts w) ω
    #[]
  unfold sessionS at hS
  rw [hS] at h1 h2
  dsimp only at h1 h2
  unfold honestRun prog
  rw [hs]
  dsimp only
  change (PI.bind (sessionPZk asks Hashes.pi spec (tabsOf sts) (traces sts w) ω).run _).sim _ _ 0 = _
  rw [PI.sim_bind]
  have h1' := h1
  simp only [Array.size_empty] at h1'
  rw [h1']
  rw [h2, Array.empty_append, List.toList_toArray]
  show (_ ++ [], _) = _
  rw [List.append_nil]
  rfl

end FlockComplete.ZkTables
