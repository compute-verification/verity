import Proofs.Flock.Claims.Complete.ZkTables.Faithful
import Proofs.Flock.Claims.Complete.ZkTables.Run
import Proofs.Flock.Claims.Complete.ZkTables.Count

/-!
# The masked prover's message from the session's run

A masked session that does not return (it throws at a refused coin) is sent by `prog` as the empty message, and the
prover sends it once each of the session's sends is answered; before that, the prover sends the session's next body.

* `feed_sim_ge`: fed at least as many answers as the program sends, `feed` returns the program's result.
* `honestP_stops`: fed answers at which the session stops having sent at most that many, `honestP` sends `[]`.
  `honestP_err`: fed `1 + rounds` answers (the session's sends are at most that many, `sessionPZk_sends`), at a session
  that does not return at those answers. `honestPR_stops` and `honestPR_err` are the same for `honestPR`.
* `honestP_sent`: fed fewer answers than the session sends, `honestP` sends the session's body at their count, tagged
  `0`; `honestPR_sent` the same for `honestPR` where the answers show no outer coin `1`.
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

/-- **A masked session that stops within the answers is sent as `[]`**: fed the answers `bs`, where the session
answered by `bs` stops having sent at most `|bs|`, the honest masked prover's message is `[]`. -/
theorem honestP_stops (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (bs : List Claims.Msg) {e : String} {L : Array Claims.Msg}
    (hrun : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
      (MaskRank.masksOf sts) (traces sts w) ω).go #[] = (.error e, L)) (hl : L.size ≤ bs.length) :
    ((feed (prog x w ω) bs).eval H).1 = [] := by
  obtain ⟨bs', h1, h2⟩ := sessionPZk_sim H (fun k => bs.getD k []) spec (tabsOf sts) (MaskRank.masksOf sts)
    (traces sts w) ω #[]
  rw [hrun] at h1 h2
  dsimp only at h1 h2
  rw [h2, Array.empty_append, List.size_toArray] at hl
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

/-- **Fed fewer answers than the session sends, `honestP` sends its next body**: fed the answers `bs`, where the
session answered by `bs` sends more than `|bs|` bodies, the honest masked prover's message is body `|bs|`, tagged `0`. -/
theorem honestP_sent (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (bs : List Claims.Msg) {R : Except String (String × Array ByteArray)} {L : Array Claims.Msg}
    (hrun : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
      (MaskRank.masksOf sts) (traces sts w) ω).go #[] = (R, L)) (hl : bs.length < L.size) :
    ((feed (prog x w ω) bs).eval H).1 = 0 :: L[bs.length]! := by
  obtain ⟨bs', h1, h2⟩ := sessionPZk_sim H (fun k => bs.getD k []) spec (tabsOf sts) (MaskRank.masksOf sts)
    (traces sts w) ω #[]
  rw [hrun] at h1 h2
  dsimp only at h1 h2
  rw [h2, Array.empty_append] at hl ⊢
  rw [List.size_toArray] at hl
  simp only [Array.size_empty] at h1
  have hsim : ((prog x w ω).sim H (fun k => bs.getD k []) 0).1 = bs'.map (List.cons 0) := by
    unfold prog
    rw [hs]
    dsimp only
    change ((PI.bind (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (traces sts w) ω).run
      _).sim _ _ 0).1 = _
    rw [PI.sim_bind, h1]
    cases R <;> exact List.append_nil _
  have key := feed_sim H (fun k => bs.getD k []) (prog x w ω) 0 bs.length (by
    rw [hsim, List.length_map]
    omega)
  simp only [Nat.zero_add, map_getD_range] at key
  rw [key, hsim, List.getD_append _ _ _ _ (by rw [List.length_map]; exact hl),
    List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hl), List.getElem_map,
    getElem!_pos bs'.toArray bs.length (by rw [List.size_toArray]; exact hl), List.getElem_toArray]

/-- **A masked session that does not return is sent as `[]`**: fed the answers `bs`, at least `1 + rounds` of them, where
the session answered by `bs` does not return, the honest masked prover's message is `[]` (`honestP_stops`; the
session sends at most `1 + rounds`, `sessionPZk_sends`). -/
theorem honestP_err (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (bs : List Claims.Msg) (hl : 1 + rounds true spec sts ≤ bs.length)
    (herr : ∀ r l, (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
      (MaskRank.masksOf sts) (traces sts w) ω).go #[] ≠ (.ok r, l)) :
    ((feed (prog x w ω) bs).eval H).1 = [] := by
  rcases hY : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
    (MaskRank.masksOf sts) (traces sts w) ω).go #[] with ⟨R, L⟩
  have hsend := sendsC_ran (sessionPZk_sends (fun k => bs.getD k []) (Claims.hOf H) spec sts (MaskRank.masksOf sts)
    (traces sts w) ω hspec hm) hY
  cases R with
  | ok r => exact absurd hY (herr r L)
  | error e => exact honestP_stops H x w ω hs bs hY (by omega)

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

/-- **`honestPR` sends `[]` after a session that stops within the answers**: at a history whose answers `bs` leave the
session on the mask witness (`maskWOf`) stopped having sent at most `|bs|`. -/
theorem honestPR_stops (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hist : List (Claims.Msg × Claims.Msg)) (bs : List Claims.Msg) (hh : hist.map Prod.snd = bs)
    (hstop : ∀ R L, (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
      (MaskRank.masksOf sts) (traces sts (maskWOf x w ω.mask)) ω).go #[] = (R, L) →
      (∃ e, R = .error e) ∧ L.size ≤ bs.length) :
    (((honestPR w).next x hist ω).eval H).1 = [] := by
  rcases hY : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
    (MaskRank.masksOf sts) (traces sts (maskWOf x w ω.mask)) ω).go #[] with ⟨R, L⟩
  obtain ⟨⟨e, rfl⟩, hl⟩ := hstop R L hY
  cases ho : outerHit x (hist.map Prod.snd)
  · rw [honestPR_next ho]
    show ((feed (prog x (maskWOf x w ω.mask) ω) (hist.map Prod.snd)).eval H).1 = []
    rw [hh]
    exact honestP_stops H x _ ω hs bs hY hl
  · rw [honestPR_stop ho]
    rfl

/-- **`honestPR` sends the session's next body at a history with no outer hit**: where the answers `bs` show no outer
coin `1` and the session on the mask witness (`maskWOf`), answered by them, sends more than `|bs|` bodies, its message
is body `|bs|`, tagged `0` (`honestP_sent`). -/
theorem honestPR_sent (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hist : List (Claims.Msg × Claims.Msg)) (bs : List Claims.Msg) (hh : hist.map Prod.snd = bs)
    (ho : outerHit x bs = false) {R : Except String (String × Array ByteArray)} {L : Array Claims.Msg}
    (hrun : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
      (MaskRank.masksOf sts) (traces sts (maskWOf x w ω.mask)) ω).go #[] = (R, L)) (hl : bs.length < L.size) :
    (((honestPR w).next x hist ω).eval H).1 = 0 :: L[bs.length]! := by
  subst hh
  rw [honestPR_next ho]
  exact honestP_sent H x _ ω hs _ hrun hl

end FlockComplete.ZkTables
