import Proofs.Flock.SessionLoop.FinRepErasedCut
import Proofs.Flock.SessionLoop.FinRepXs
import Proofs.Flock.SessionLoop.FinRepView
import Proofs.Flock.Claims.ZK.PiopFree

/-!
# The views' cut PIOP returns where the firewall's rep did

The cut PIOP returns on its coins alone (`Masked.piopCut_of_repFwC`), so where the session's rep on the loop's table
and request returned (`RepAt`, as `finRepCut` runs it at the opened coins), the cut PIOP from any source, layout, cap,
root and domain returns from the rep's base, at table `j`'s setup, the views' extra claims, the earlier reps' points
and the mask slot's words. No premise on the views' masked values, and none on the rows.

**`Loop.piopCut_views_ret`**: that return. At the views' source, `flat cms[j]!.cap`, `ic.root` and the session's
domain it is `FinPiop`'s first conjunct and the `hpiop` that `swSrcE_hmk_cut` and `swSrcE_l0_live` take.
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
/-- **The views' cut PIOP returns**: in the masked scope, at a witness in the relation, where the session's rep on the
loop's table and request at table `j`'s rep `r` returned (`RepAt`), the cut PIOP at table `j`'s setup, answered by the
opened coins, from any source, layout, cap, root and domain, at the views' extra claims, the earlier reps' points and
the mask slot's words, returns from the rep's base. -/
theorem piopCut_views_ret (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    (j : Fin (Masked.jOf x)) (r : Fin 2) {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    {q : _root_.Flock.Firewall.Finish.RepIn} {f : ByteArray}
    (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hex : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras)
    (lay : _root_.Flock.Zk.PadLayout) (rs : FlockComplete.RepSrc) (cap root dom : ByteArray) {l : Array Msg}
    (hl : l.size = Masked.repBase x (2 * j.val + r.val)) :
    ∃ res tr₁ l₁, ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! lay rs cap root extras (Masked.priorE (finCoins x ops) j.val r.val)
      (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run (_root_.Flock.Transcript.init #[] dom)).go l =
      (.ok (res, tr₁), l₁) := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    have := r.isLt
    omega
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
  obtain ⟨ic', p, tr, hic', hfw, -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hRep.2.2
  have hcut := finRepCut L₀ src (realOf src) wk x w hsc ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hk tb hi q _ hRep ic' p tr
    hic' hfw l hl
  obtain ⟨htb, -⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hs hk hRep
  rw [hk2] at htb
  rw [← htb] at hcut
  have hxs := finRepXs L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hk extras
    (by rw [hk2]; exact hex) tb hi q _ hRep
  have hcx := congrArg Array.size (_root_.Flock.Firewall.Finish.claimsOf_map_x q.xs)
  have hex' := congrArg Array.size hxs
  rw [Array.size_map] at hcx hex'
  have hes : extras.size = (_root_.Flock.Firewall.Finish.claimsOf q.xs).size := by omega
  have hst := Masked.tabsOf_stAt x j
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hsm : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m = (Masked.stsOf x)[j.val]!.m := by
    rw [hst, hstj]
    rfl
  have hsk : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog = (Masked.stsOf x)[j.val]!.c.kLog := by
    rw [hst, hstj]
    rfl
  have hfk : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.fold.kLog =
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog := by
    rw [hst]
    rfl
  obtain ⟨res, t₃, l₃, h⟩ := Masked.piopCut_of_repFwC hx (c := finCoins x ops) hA hj r.isLt (refK x) hsm hsk hfk
    (go_eq_of_run_fst hcut) hl lay rs cap root hes (t₀' := _root_.Flock.Transcript.init #[] dom) rfl
  rw [FlockComplete.MaskRank.masksOf_getElem! hj]
  exact ⟨res, t₃, l₃, h⟩

end Loop

end Claims.ZKServer
