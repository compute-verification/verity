import Proofs.Flock.Claims.Complete.ZkTables.Faithful
import Proofs.Flock.Claims.Complete.ZkTables.Run
import Proofs.Flock.Claims.Complete.ZkTables.Count

/-!
# The masked prover after a refusal

A masked session that does not return (it throws at a refused coin) is sent by `prog` as the empty message, and the
prover sends it once each of the session's sends is answered.

* `feed_sim_ge`: fed at least as many answers as the program sends, `feed` returns the program's result.
* `honestP_err`: fed `1 + rounds` answers (the session's sends are at most that many, `sessionPZk_sends`), at a session
  that does not return at those answers, `honestP` sends `[]`; `honestPR_err` is the same for `honestPR`.
-/

namespace FlockComplete

open Flock FlockDeploy

/-- **Fed at least the program's sends, `feed` returns its result.** -/
theorem feed_sim_ge (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (p : PI Claims.Msg) :
    ∀ (i j : ℕ), (p.sim H resp i).1.length ≤ j →
      ((feed p ((List.range j).map fun k => resp (i + k))).eval H).1 = (p.sim H resp i).2 := by
  induction p with
  | ret a => intro i j _; rfl
  | query q k ih => intro i j hj; exact ih (H q) i j hj
  | ask m k ih =>
    intro i j hj
    cases j with
    | zero => simp [PI.sim] at hj
    | succ j =>
      simp only [PI.sim, List.length_cons, Nat.add_le_add_iff_right] at hj
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      have e : ((fun k => resp (i + k)) ∘ Nat.succ) = fun k => resp (i + 1 + k) := by
        funext k
        simp only [Function.comp, Nat.succ_eq_add_one]
        congr 1
        omega
      rw [e]
      simp only [feed, PI.sim]
      exact ih (resp i) (i + 1) j hj

end FlockComplete

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- **A masked session that does not return is sent as `[]`**: fed the answers `bs`, at least `1 + rounds` of them, where
the session answered by `bs` does not return, the honest masked prover's message is `[]`. -/
theorem honestP_err (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (bs : List Claims.Msg) (hl : 1 + rounds true spec sts ≤ bs.length)
    (herr : ∀ r l, (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
      (MaskRank.masksOf sts) (traces sts w) ω).go #[] ≠ (.ok r, l)) :
    ((feed (prog x w ω) bs).eval H).1 = [] := by
  obtain ⟨bs', h1, h2⟩ := sessionPZk_sim H (fun k => bs.getD k []) spec (tabsOf sts) (MaskRank.masksOf sts)
    (traces sts w) ω #[]
  generalize hY : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
    (MaskRank.masksOf sts) (traces sts w) ω).go #[] = res at h1 h2
  obtain ⟨R, L⟩ := res
  have hsend := sendsC_ran (sessionPZk_sends (fun k => bs.getD k []) (Claims.hOf H) spec sts (MaskRank.masksOf sts)
    (traces sts w) ω hspec hm) hY
  dsimp only at h1 h2
  obtain ⟨e, rfl⟩ : ∃ e, R = .error e := by
    cases R with
    | error e => exact ⟨e, rfl⟩
    | ok r => exact absurd hY (herr r L)
  rw [h2, Array.empty_append, List.size_toArray] at hsend
  simp only [Array.size_empty] at h1
  have hsim : (prog x w ω).sim H (fun k => bs.getD k []) 0 = (bs'.map (List.cons 0), []) := by
    unfold prog
    rw [hs]
    dsimp only
    change (PI.bind (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (traces sts w) ω).run _).sim
      _ _ 0 = _
    rw [PI.sim_bind, h1]
    show (_ ++ [], _) = _
    rw [List.append_nil]
    rfl
  have key := feed_sim_ge H (fun k => bs.getD k []) (prog x w ω) 0 bs.length (by
    rw [hsim, List.length_map]
    omega)
  simp only [Nat.zero_add, map_getD_range] at key
  rw [key, hsim]

/-- **`honestPR` sends `[]` after a session that does not return**: at a history whose answers `bs`, at least
`1 + rounds` of them, leave the session on the mask witness (`maskWOf`) not returning. -/
theorem honestPR_err (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (hist : List (Claims.Msg × Claims.Msg)) (bs : List Claims.Msg) (hh : hist.map Prod.snd = bs)
    (hl : 1 + rounds true spec sts ≤ bs.length)
    (herr : ∀ r l, (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
      (MaskRank.masksOf sts) (traces sts (maskWOf x w ω.mask)) ω).go #[] ≠ (.ok r, l)) :
    (((honestPR w).next x hist ω).eval H).1 = [] := by
  cases ho : outerHit x (hist.map Prod.snd)
  · rw [honestPR_next ho]
    show ((feed (prog x (maskWOf x w ω.mask) ω) (hist.map Prod.snd)).eval H).1 = []
    rw [hh]
    exact honestP_err H x _ ω hs hspec hm bs hl herr
  · rw [honestPR_stop ho]
    rfl

end FlockComplete.ZkTables
