import Proofs.Flock.SessionLoop.FinRep
import Proofs.Flock.SessionLoop.FinRounds
import Proofs.Flock.Claims.ZK.FinHmkLive

/-!
# `hmk` at the loop's `Finish`: the views' masked values below `zp 64`

At (c.3)'s context (`Loop.FinRepAt`'s), the session's rep (`repFwC` on the loop's table `tb` and request `q`) returns
`(p, tr)` and rep stream `k`'s run from the views (`Loop.viewRep`) returns `(p', tr)` with `eraseZ p' = eraseZ p`. Both
run at one setup: the loop's table is `tabOfLevel0` at the statement's (`Loop.realOf_st`), and the views' is the
deployed tables' at `k / 2`. So `HmkLive.masked_of_eraseZ` reads the views' masked values below `zp 64` off the two
proofs: they are the request's values plus its pads, `tauFw_at`'s `hmk`, with no history read.

* `Loop.realOf_st`: the loop's table for rep stream `k` is at the setup of the statement's table `k / 2`.
* **`Loop.finSrc_hmk_live`**: the views' source (`Loop.finSrc`) holds `q.vals + pads.h` below `zp 64`.
* **`Loop.swSrcE_hmk_live`**: so does the switched cut's (`Masked.swSrcE` at the opened coins and the workers' values
  `vs`), where every table admits the opened coins (`Loop.admits_of_finCtx`, from `Serves`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **The loop's table for rep stream `k` is at table `k / 2`'s setup** (`realOf`'s `tabOfLevel0`). -/
theorem realOf_st {R : Type} {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {k : ℕ} {s0 : _} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} (hr : realOf src x w ρ k = .ok (tb, hi)) :
    tb.st = _root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]! := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  unfold realOf at hr
  rw [hsx] at hr
  try dsimp only at hr
  obtain ⟨⟨g, d, hi'⟩, -, hr⟩ := bok hr
  obtain ⟨l, -, hr⟩ := bok hr
  obtain ⟨tb', htb, hr⟩ := bok hr
  have htb' : tb' = tb := by cases hr; rfl
  rw [htb'] at htb
  exact (Count.tabOfLevel0_sched htb).1

/-- **The views' masked values below `zp 64` are the request's plus its pads**: at (c.3)'s context, where the session's
rep on the loop's table `tb` and request `q` returns `(p, tr)` and rep stream `k`'s run from the views returns
`(p', tr)` with `eraseZ p' = eraseZ p`, below `zp 64` the views' source's masked values are `q.vals + pads.h`. -/
theorem finSrc_hmk_live (vs : finS.SwVals) {L₀ : LeanRelease Flock.Wit Tp}
    {src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    {ν : finS.Nonce} {t : finS.Tape} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {s0 : _} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)} {cms : Array FlockComplete.Commit} {k : ℕ}
    (hk : k < 2 * spec.tables.size) {extras : Array _root_.Flock.Claim} {ic : FlockComplete.InnerCommit}
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hR : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f)
    {ic' : _root_.Flock.Firewall.Finish.InnerCommit} {l : Array Msg} {p : _root_.Flock.Zk.ZkProof}
    {tr : _root_.Flock.Transcript}
    (hrun : (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr))
    {p' : _root_.Flock.Zk.ZkProof} {l₀ l' : Array Msg}
    (hv : (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l₀ = (.ok (p', tr), l'))
    (he : FlockComplete.eraseZ p' = FlockComplete.eraseZ p) :
    ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := by
  obtain ⟨h25, -, -, -⟩ := finRepBounds (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hk hR
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hi' : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨hr, -, -⟩ := hR
  have hst := realOf_st hsx hr
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! = tb.st := by
    rw [hst, getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) (by simpa [FlockDeploy.tabsOf] using hi'),
      getElem!_pos (Masked.stsOf x) (k / 2) hi']
    simp [FlockDeploy.tabsOf]
  have hkL : tb.st.fold.kLog = tb.st.kLog := by
    rw [hst]
    rfl
  unfold viewRep at hv
  exact HmkLive.masked_of_eraseZ _ _ _ _ _ _ htab (by omega) hkL hrun hv he

/-- **`swSrcE_hmk_live`: the switched cut's masked values below `zp 64` are the request's plus its pads**: at (c.3)'s
context at a served statement (`Serves`), where every table admits the opened coins (`admits_of_finCtx`), the switched
cut's source at the opened coins and the workers' values `vs` is the views' up to `eraseS` (`eraseS_finSrc`), which keeps
the masked values, so below `zp 64` they are `q.vals + pads.h` (`finSrc_hmk_live`). -/
theorem swSrcE_hmk_live (vs : finS.SwVals) {L₀ : LeanRelease Flock.Wit Tp}
    {src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hRel : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)} {cms : Array FlockComplete.Commit} {k : ℕ}
    (hk : k < 2 * spec.tables.size) {extras : Array _root_.Flock.Claim} {ic : FlockComplete.InnerCommit}
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hR : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f)
    {ic' : _root_.Flock.Firewall.Finish.InnerCommit} {l : Array Msg} {p : _root_.Flock.Zk.ZkProof}
    {tr : _root_.Flock.Transcript}
    (hrun : (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr))
    {p' : _root_.Flock.Zk.ZkProof} {l₀ l' : Array Msg}
    (hv : (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l₀ = (.ok (p', tr), l'))
    (he : FlockComplete.eraseZ p' = FlockComplete.eraseZ p) :
    ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2)).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := by
  intro i hi
  have hx : Masked.Live x := Masked.live_of hsc hRel
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hA := admits_of_finCtx L₀ src real wk hServe hF
  have e := congrArg FlockComplete.RepSrc.masked
    (eraseS_finSrc vs w ν t hA (i := k / 2) (r := k % 2) (by omega) (Nat.mod_lt _ (by omega)))
  have e' : ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2)).masked =
      ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)).masked := e.symm
  rw [congrFun e' i]
  exact finSrc_hmk_live vs hsc hsx hk hR hrun hv he i hi

end Loop

end Claims.ZKServer
