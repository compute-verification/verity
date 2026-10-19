import Proofs.Flock.Claims.ZK.Split

/-!
# The deployed plan's last message is the honest prover's (`FinExe`)

`FinExe` (`Sent`'s `hfin` at the deployed plan) asks that after `refK x` openings `ops` whose coins every table admits,
`finExe` of the tables' views is `honestPR`'s last message. Both are `Final.encode` of a session's verdict, `[]` where it
throws (`finMsg`):

* **`honestPR`'s side.** Where every table admits, no outer coin is `1` (`outerHit_false`), so `honestPR` runs `honestP`
  on the mask witness, and fed `refK x = 1 + rounds` answers it sends `finMsg` of the session answered by them
  (`honestPR_finMsg`, `ZkTables.honestP_err`'s argument at a session that returns): `liveRunAt`'s verdict.
* **`finExe`'s side** is `finMsg` of `runCut` at the views' source (`finExe_eq`).

`LiveEnd` (the session answered by the openings returns what the session answered by the opened coins and stopped
after the last round returns) and `SimFin` (the session read from the views returns the session's verdict) close the
gap: `finExe_of`.
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- **The last message of a session's verdict at `x`**: `Final.encode` of its record and proof files, `[]` where it threw
(`prog`'s). -/
def finMsg (x : Claims.Msg) : Except String (String × Array ByteArray) → Claims.Msg
  | .ok (record, proofs) => Final.encode ⟨x, record, proofs⟩
  | .error _ => []

/-- **Fed `1 + rounds` answers, the honest masked prover sends the session's verdict**: `finMsg` of the session answered
by the answers `bs` (its sends are at most `1 + rounds`, `sessionPZk_sends`). -/
theorem honestP_finMsg (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (bs : List Claims.Msg) (hl : 1 + rounds true spec sts ≤ bs.length) :
    ((feed (prog x w ω) bs).eval H).1 =
      finMsg x ((sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
        (MaskRank.masksOf sts) (traces sts w) ω).go #[]).1 := by
  obtain ⟨bs', h1, h2⟩ := sessionPZk_sim H (fun k => bs.getD k []) spec (tabsOf sts) (MaskRank.masksOf sts)
    (traces sts w) ω #[]
  generalize hY : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
    (MaskRank.masksOf sts) (traces sts w) ω).go #[] = res at h1 h2 ⊢
  obtain ⟨R, L⟩ := res
  have hsend := sendsC_ran (sessionPZk_sends (fun k => bs.getD k []) (Claims.hOf H) spec sts (MaskRank.masksOf sts)
    (traces sts w) ω hspec hm) hY
  dsimp only at h1 h2 ⊢
  rw [h2, Array.empty_append, List.size_toArray] at hsend
  simp only [Array.size_empty] at h1
  have hsim : (prog x w ω).sim H (fun k => bs.getD k []) 0 = (bs'.map (List.cons 0), finMsg x R) := by
    unfold prog
    rw [hs]
    dsimp only
    change (PI.bind (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (traces sts w) ω).run _).sim
      _ _ 0 = _
    rw [PI.sim_bind, h1]
    rcases R with e | ⟨record, proofs⟩
    · show (_ ++ [], _) = _
      rw [List.append_nil]
      rfl
    · show (_ ++ [], _) = _
      rw [List.append_nil]
      rfl
  have key := feed_sim_ge H (fun k => bs.getD k []) (prog x w ω) 0 bs.length (by
    rw [hsim, List.length_map]
    omega)
  simp only [Nat.zero_add, map_getD_range] at key
  rw [key, hsim]

/-- **`honestPR`'s last message is the session's verdict** at a history whose answers `bs`, at least `1 + rounds` of
them, hold no outer coin `1`: `finMsg` of the session on the mask witness answered by them. -/
theorem honestPR_finMsg (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (hist : List (Claims.Msg × Claims.Msg)) (bs : List Claims.Msg) (hh : hist.map Prod.snd = bs)
    (hl : 1 + rounds true spec sts ≤ bs.length) (ho : outerHit x bs = false) :
    (((honestPR w).next x hist ω).eval H).1 =
      finMsg x ((sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
        (MaskRank.masksOf sts) (traces sts (maskWOf x w ω.mask)) ω).go #[]).1 := by
  rw [honestPR_next (show outerHit x (hist.map Prod.snd) = false by rw [hh]; exact ho)]
  show ((feed (prog x (maskWOf x w ω.mask) ω) (hist.map Prod.snd)).eval H).1 = _
  rw [hh]
  exact honestP_finMsg H x _ ω hs hspec hm bs hl

end FlockComplete.ZkTables

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **`finExe` is the verdict's last message** of the session read from the views (`runCut` at `tabSrcView`). -/
theorem finExe_eq {x : Msg} {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (c : Fin (refK x) → FlockDeploy.Coin) (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.View) :
    finExe PE x c v =
      FlockComplete.ZkTables.finMsg x (runCut x c (refK x) spec (tabSrcView PE x (czExe PE x c) v)).1 := by
  unfold finExe
  rw [hsx]
  dsimp only
  generalize (runCut x c (refK x) spec (tabSrcView PE x (czExe PE x c) v)).1 = R
  rcases R with e | ⟨record, proofs⟩ <;> rfl

/-- **Where every table admits the opened coins, no outer coin is `1`**: after `refK x` openings `ops`, every
zerocheck outer coin among their answers (`outerSlotsZk`) is table `j`'s rep `r`'s outer coin `i` at the opened coins
(`outerAt`), which `czExe`'s `rOuter` reads (`PointReads`) and `Admits` keeps off `1`. -/
theorem outerHit_false {x : Msg} (hx : Live x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x (padOps ops) j)) :
    FlockComplete.ZkTables.outerHit x (ops.map fun p => coinOfD p.2) = false := by
  rw [← Bool.not_eq_true]
  intro h
  unfold FlockComplete.ZkTables.outerHit at h
  rw [List.any_eq_true] at h
  obtain ⟨⟨k, i⟩, hmem, hk⟩ := h
  unfold FlockComplete.outerSlotsZk at hmem
  rw [hsx] at hmem
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq] at hmem
  obtain ⟨q, hq, i', hi', rfl, rfl⟩ := hmem
  rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff, List.length_map] at hk
  obtain ⟨hk, h1⟩ := hk
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hlt : q / 2 < (stsOf x).size := by omega
  have hj : q / 2 < jOf x := hJ ▸ hlt
  have hr : q % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hq' : 2 * (q / 2) + q % 2 = q := Nat.div_add_mod q 2
  rw [getElem!_pos (stsOf x) (q / 2) hlt] at hi'
  have hm35 := ((FlockComplete.ZkTables.tables_of_setup hx.1 hsx).2.2.2 (q / 2) hlt).2.1
  have hslot : outerSlot x (q / 2) (q % 2) = FlockComplete.repStartZk spec (stsOf x) q + 1 := by
    unfold outerSlot
    rw [hsx, hq']
  have hval : outerAt (padOps (x := x) ops) (q / 2) (q % 2) i' = 1 := by
    have e : outerAt (padOps (x := x) ops) (q / 2) (q % 2) i' =
        slotAt (padOps (x := x) ops) (outerSlot x (q / 2) (q % 2)) i' := rfl
    rw [e, hslot, slotAt_padOps ops hl (by omega) (by omega)]
    exact h1
  exact (hA ⟨q / 2, hj⟩).2.2.2 ⟨q % 2, hr⟩ _ (List.mem_ofFn.2 ⟨⟨i', hi'⟩, rfl⟩) hval

/-- **The tables' real views at the opened coins `c`**, on `w`'s table witnesses and the tape `τ` (`viewsAt`; it reads
neither the rep rounds nor the round plan's encodings). -/
noncomputable def viewsE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.View :=
  (session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ

/-- **What the deployed session's last message reads from the views** (`tabSrcView`). -/
noncomputable def srcV (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ℕ → FlockComplete.TabSrc :=
  tabSrcView PE x (czExe PE x c) (viewsE x c w τ)

/-- **The session answered by the openings returns what the session answered by the opened coins does**, stopped after
the last round (`PrefixLive` at `n = refK x`, executable code only): after `refK x` openings `ops` whose coins every
table admits. -/
def LiveEnd : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)), ops.length = refK x →
    (∀ j, (tabC PE x j).Admits (czExe PE x (padOps ops) j)) →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (liveRunAt x w ω ops spec).1 = (runExe x w ω (padOps (x := x) ops) (refK x) spec).1

/-- **The session read from the views returns the session's verdict** (`SimSession` at the last round, with the final
message's data): at a live statement and a witness in the deployed relation, at opened coins every table admits,
`sessionCut` on the views' source (`srcV`) returns what the masked session returns, both answered by the coins. -/
def SimFin : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin),
    (∀ j, (tabC PE x j).Admits (czExe PE x c j)) → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (runCut x c (refK x) spec (srcV x c w τ)).1 = (runExe x w (UE.tapeZk τ) c (refK x) spec).1

/-- **`FinExe` from the last round's two sessions**: `finExe` of the views is `finMsg` of `runCut` on them
(`finExe_eq`), which is the session's verdict (`SimFin`), which is the live session's (`LiveEnd`), which `honestPR`
sends (`honestPR_finMsg`, no outer coin `1` by `outerHit_false`). Where `x` sets up nothing both are `[]`. -/
theorem finExe_of (hL : LiveEnd) (hS : SimFin) : FinExe := by
  intro H hH nr x hx w hR ν t ops hl hA
  show finExe PE x (padOps ops) (viewsE x (padOps ops) w t) =
    (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (UE.tapeZk t)).eval H).1
  cases hsx0 : FlockDeploy.setupOf true x with
  | none =>
    have e1 : finExe PE x (padOps ops) (viewsE x (padOps ops) w t) = [] := by
      unfold finExe
      rw [hsx0]
    rw [e1]
    by_cases ho : FlockComplete.ZkTables.outerHit x ((ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd) = true
    · rw [FlockComplete.ZkTables.honestPR_stop ho]
      rfl
    · rw [FlockComplete.ZkTables.honestPR_next (by simpa using ho)]
      show [] = ((FlockComplete.feed (FlockComplete.ZkTables.prog x _ _) _).eval H).1
      unfold FlockComplete.ZkTables.prog
      rw [hsx0]
      rfl
  | some v =>
    obtain ⟨s', spec', sts'⟩ := v
    have hst : stsOf x = sts' := by
      unfold stsOf
      rw [hsx0]
    have hpos : 0 < (stsOf x).size := by
      rw [hst]
      exact (FlockComplete.ZkTables.tables_of_setup hx.1 hsx0).1
    obtain ⟨s, spec, hsx⟩ := setupOf_of_pos hpos
    obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
    have hK := refK_eq hsx
    have hh : (ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd = ops.map fun p => coinOfD p.2 := by
      rw [List.map_map]
      rfl
    rw [finExe_eq hsx, FlockComplete.ZkTables.honestPR_finMsg H x w _ hsx hspec
      (fun i hi => by have := (hall i hi).1; omega) _ _ hh (by rw [List.length_map]; omega)
      (outerHit_false hx hsx ops hl hA), hH]
    exact congrArg (FlockComplete.ZkTables.finMsg x)
      ((hS x hx w hR t (padOps ops) hA s spec hsx).trans (hL x hx w hR _ ops hl hA s spec hsx).symm)

end Masked

end Claims.ZKMal
