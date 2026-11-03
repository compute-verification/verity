import Proofs.Flock.SessionLoop.FinPiopRet
import Proofs.Flock.SessionLoop.FinRepOpen
import Proofs.Flock.SessionLoop.FinRepRows
import Proofs.Flock.Claims.ZK.RepCutFw

/-!
# The views' cut rep returns where the firewall's rep did, with no rows

The cut rep is answered by the opened coins, so its control flow reads only coins, sizes and points
(`FlockComplete.CutFree.repCut_of_repFwC`): where the session's rep on the loop's table and request returned
(`RepAt`, as `finRepCut` runs it at the opened coins), the cut rep at the same table and schedule, from any hash,
level-0 commitment, pads commitment and source, returns from the rep's base, at the views' extra claims, the earlier
reps' points and the mask slot's words, and ends where the firewall's does, at the next rep's base (`finRepCut_size`).
No premise on the views' masked values, and none on the rows.

**`Loop.repCut_views_ret`**: that return, at any rep stream `k`.

**`Loop.prevCut_views`**: the earlier streams' returns from the switched views' source, in the form `FinPiop`, `FinFc`,
`swSrcE_hmk_fw` and `swSrcE_l0_views` take as `hprev`: at table `j`'s rep `r`, every stream below `2j + r` returns
from its base and ends at the next one's, at any pads commitment.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 1000000 in
/-- **The views' cut rep returns at the next rep's base**: in the masked scope, at a witness in the relation, at a
`Finish` `FinishIs` reads, rep stream `k`'s cut rep at its table and `--zk` schedule, answered by the opened coins, from
any hash, level-0 commitment, pads commitment and source, at the views' extra claims, the earlier reps' points and the
mask slot's words, returns from the rep's base and ends at rep `k + 1`'s: the firewall's rep stream `k` returned
(`proofsOf_at`, `proofFile_ok`, `finRepCut`) and ends there (`finRepCut_size`), and the cut rep returns with it
(`repCut_of_repFwC`), its rank check by `rank_at_admits`. -/
theorem repCut_views_ret (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    {k : ℕ} (hk : k < 2 * spec.tables.size) {extras : Array _root_.Flock.Claim}
    (hex : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (h' : ByteArray → ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
    (rs : FlockComplete.RepSrc) {l : Array Msg} (hl : l.size = Masked.repBase x k) :
    ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
        (_root_.Flock.Hashes.ofSha512 h') (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
        (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cm0 extras ic rs
        (Masked.priorE (finCoins x ops) (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
        (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1) := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
  -- the firewall's rep stream `k`, at the opened coins
  obtain ⟨files, hp⟩ := proofsOf_ok_of_ctx L₀ src (realOf src) wk x w hF hsx
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  obtain ⟨tb, hi, q, hRep⟩ := proofsOf_at hp (k := k) (by rw [hrs]; exact hk)
  obtain ⟨ic', p, tr, hic', hfw, -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hRep.2.2
  have hsz := finRepCut_size L₀ src (realOf src) wk x w hsc ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk tb hi q _ hRep ic' p
    tr hic' hfw l hl
  have hgo := go_of_fst (finRepCut L₀ src (realOf src) wk x w hsc ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk tb hi q _ hRep
    ic' p tr hic' hfw l hl)
  -- the session's table and schedule
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hs hk hRep
  obtain ⟨h25, -, -, -⟩ := finRepBounds (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hk hRep
  have hst := Masked.tabsOf_stAt x ⟨k / 2, by omega⟩
  have hstj : (Masked.stsOf x)[k / 2]! = Masked.stAt x ⟨k / 2, by omega⟩ := getElem!_pos (Masked.stsOf x) (k / 2) hj
  have hsm : tb.st.m = (Masked.stsOf x)[k / 2]!.m := by
    rw [← htb, hst, hstj]
    rfl
  have hsk : tb.st.kLog = (Masked.stsOf x)[k / 2]!.c.kLog := by
    rw [← htb, hst, hstj]
    rfl
  have hfk : tb.st.fold.kLog = tb.st.kLog := by
    rw [← htb, hst]
    rfl
  -- as many extra claims as the firewall's
  have hxs := finRepXs L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk extras hex tb hi q _ hRep
  have hcx := congrArg Array.size (_root_.Flock.Firewall.Finish.claimsOf_map_x q.xs)
  have hex' := congrArg Array.size hxs
  rw [Array.size_map] at hcx hex'
  have hes : extras.size = (_root_.Flock.Firewall.Finish.claimsOf q.xs).size := by omega
  have hl' : l.size = Masked.repBase x (2 * (k / 2) + k % 2) := by
    rw [hl]
    congr 1
    omega
  obtain ⟨q', t₅, L₅, hq, hL⟩ := FlockComplete.CutFree.repCut_of_repFwC (Masked.ligCoin (finCoins x ops)) (refK x)
    (resp := Masked.FinRep.respOf (finCoins x ops)) (fun k => Masked.FinRep.coinOf_respOf (finCoins x ops) k)
    (by omega) hfk.symm hgo h' (k % 2) (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cm0 hes ic rs
    (Masked.priorE (finCoins x ops) (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]! rfl
    (fun _ _ _ _ _ _ _ _ h₁ h₂ => by
      rw [FlockComplete.MaskRank.masksOf_getElem! hj]
      exact Masked.rank_at_admits hx hA hj (Nat.mod_lt _ (by norm_num)) (Masked.ligCoin (finCoins x ops)) (refK x)
        (fun _ _ _ => rfl) tb.st hsm hsk hfk _ _ _ _ rfl hl' h₁ h₂)
  refine ⟨q', t₅, L₅, ?_, hL.trans hsz⟩
  rw [htb, hsb]
  exact hq

set_option maxHeartbeats 1000000 in
/-- **The earlier streams' cut reps return, with no rows**: at table `j`'s rep `r`, every rep stream `k < 2j + r`'s cut
rep from the switched views' source, at the views' extra claims and any pads commitment, returns from its base and ends
at rep `k + 1`'s (`repCut_views_ret`): the `hprev` that `FinPiop`, `FinFc`, `swSrcE_hmk_fw` and `swSrcE_l0_views`
take. -/
theorem prevCut_views (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    (cms : Array FlockComplete.Commit) (j : Fin (Masked.jOf x)) (r : Fin 2) :
    ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut Masked.hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))).go l =
          (.ok ic, l)) →
      ∀ l : Array Msg, l.size = Masked.repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x)) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))
          (Masked.priorE (finCoins x ops) (k / 2) (k % 2))
          (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1) := by
  intro k hk extras hex ic _ l hl
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hk' : k < 2 * spec.tables.size := by
    have := r.isLt
    omega
  exact repCut_views_ret L₀ src wk x w hsc hR hServe hF hsx hs hk' hex _root_.Flock.Sha512.hash cms[k / 2]! ic _ hl

end Loop

end Claims.ZKServer
