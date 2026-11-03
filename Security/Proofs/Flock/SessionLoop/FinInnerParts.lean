import Proofs.Flock.SessionLoop.FinInner

/-!
# The loop's finish at a rep's inner rounds on the worker's own histories, in parts

`LeanRelease.finOf_finalC_own` and `LeanRelease.finOf_tau_own` read the rounds the loop's finish hands a rep's frame
(`Loop.finOf`) at a history `LiveInnerOwn` reads at rep `2j + r`'s inner round `e` (`Loop.InnerHist`), at the release
`withInner` on those rounds (`Loop.relIn`). Each part is named here, none a premise of a guarantee:

* **`Loop.innerHist_fin`**: there the loop's table and request for rep stream `2j + r` return; the table's pad layout
  and the round's configuration are table `j`'s (`Masked.layE`) and the configuration's pads the tape's; and where the
  finish's live inner proof (`finInner`) returns at the request with `final_c = a` (`askOf`) and at no other `final_c`,
  `finOf` is its rounds. The history's next round is the stream's round `nSched + e`, past its scheduled rounds
  (`innerAt_slot`).
* **`Loop.innerReq_own`**: the request's values below `zp 64` but `final_c`'s are `vs`'s (`vsWord`), and its pads the
  tape's. The history's rounds of the rep's PIOP are the frame's releases of the worker's rounds (`OwnHist`), which carry
  `vs`'s values under the pads (`TakesValsOwn`).
* **`Loop.innerReq_coins`**: the request's coins of the rep's rounds before the history's next are the opened coins from
  the rep's first round (`Masked.repBase`).
* **`Loop.finInner_fc`**: where `finInner` returns, its `final_c'` is the request's `final_c` under its pad.
* **`Loop.finInner_accepts_own`**: `finInner` returns at the request with `final_c = a` only at `vs`'s `final_c`: its
  `zerocheck c_eval` constraint holds on the pads (`consWhy`), which at the request's round-1c values and the opened
  coins' zerocheck point fixes `a` (`Loop.FinalCFixed`).
* **`Loop.finInner_returns_own`**: it returns there.
* **`Loop.finInner_tau_own`**: at the inner proof's round 2, its `τ` commitment is `vs`'s (`Loop.vsTau`), where the rows
  hold (`Loop.RowsOK`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS)
open _root_.Flock.Firewall.Finish (Tab RepIn)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **The session loop's release at its finish's inner rounds**: `withInner` on `finOf` at `L₁`'s frame draws, with the
loop's own link and `Finish` (`ofLoopLink`). -/
noncomputable abbrev relIn (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) : LeanRelease Flock.Wit Tp :=
  LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real

/-- **A history `LiveInnerOwn` reads at rep `2j + r`'s inner round `e`**, at the release `L`: after `Hello` `(m₀, ρ₀)`
and `0 < ops.length < refK x` openings, live, whose openings pass, no coin refused, the release's own run
(`ZkServer.OwnHist`), whose next round is an inner proof's (`InnerAt`), rep `2j + r`'s round `innerOff e`. -/
def InnerHist (L : LeanRelease Flock.Wit Tp) (wk : L.toSrv.Worker)
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce)
    (t : finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (e : ℕ) : Prop :=
  0 < ops.length ∧ ops.length < refK x ∧ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true ∧
    ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true ∧
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false ∧
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) ∧
    InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) ∧
    L.rep x (ops.length + 1) = 2 * j.val + r.val ∧
    ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff e)

variable (L₁ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))

/-- **The finish at a rep's inner round**: at a history `InnerHist` holds at with `e < 4`, on the loop's own tables at a
statement in the masked scope with a witness that the loop serves, the loop's table and request for rep stream `2j + r`
return; the table's pad layout and the round's configuration are table `j`'s, the configuration's pads the tape's
(`Loop.RepSrc`); and where `finInner` on them returns at `final_c = a` after the stream's `nSched + e` rounds of coins,
and at no other `final_c`, `finOf` is its round 0 and round 2. -/
theorem innerHist_fin (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (wk : (relIn L₁ src real).toSrv.Worker) {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk}
    {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e) :
    ∃ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (tr : Tab) (hi : Array ByteArray) (q₀ : RepIn),
      FlockDeploy.setupOf true x = some (s0, spec, sts) ∧
      real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi) ∧
      repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀ ∧
      _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j ∧
      ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay = Masked.layE x j ∧
      ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h =
        (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h ∧
      ∀ (a fc' : F128) (tc : ByteArray),
        finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
          (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a) = .ok (fc', tc) →
        (∀ a', (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val))
          (nSched spec sts (2 * j.val + r.val))
          (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a')).isOk = true → a' = a) →
        finOf real L₁.draws x w (ν, t) ((m₀, ρ₀) :: ops) = #[inner0 fc'.toBytes, .empty, inner2 tc, .empty] := by
  sorry

/-- **The loop's request at a rep's inner round, on the worker's own history**: at a history `InnerHist` holds at, where
the worker takes `vs`'s values at its own histories (`TakesValsOwn`) and `vs` is causal, the request for rep stream
`2j + r` (`repInOf`) has `zp 64` values, `vs`'s below it but at `final_c`'s pad (`vsWord`, at the coins the openings
open), and the tape's pads. -/
theorem innerReq_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) {ν : finS.Nonce} {t : finS.Tape}
    {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2}
    {e : ℕ} (he : e < 4) (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    q₀.vals.size = (Masked.layE x j).zp 64 ∧
      (∀ i < (Masked.layE x j).zp 64, i ≠ (Masked.layE x j).fc →
        q₀.vals[i]! = vsWord vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r i) ∧
      (_root_.Flock.Firewall.Finish.padsOf q₀.draws).toC =
        FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) := by
  sorry

/-- **The request's coins** at a history `InnerHist` holds at: it holds a round of coins for each of rep stream
`2j + r`'s rounds, and below each round's count its rounds before the history's next, `nSched + e` of them, are the
opened coins from the rep's first round (`Masked.repBase`). -/
theorem innerReq_coins (x : Msg) (w : Flock.Wit) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (wk : (relIn L₁ src real).toSrv.Worker) {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk}
    {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    q₀.coins.size = (roundsOf spec sts).getD (2 * j.val + r.val) 0 ∧
      nSched spec sts (2 * j.val + r.val) + e < q₀.coins.size ∧
      ∀ k' < nSched spec sts (2 * j.val + r.val) + e, ∀ i < (q₀.coins[k']!).size,
        Masked.slotAt (finS.pad (ops.map fun p => finS.decOf p.2)) (Masked.repBase x (2 * j.val + r.val) + k') i =
          (q₀.coins[k']!)[i]! := by
  sorry

/-- **`finInner` returns the request's `final_c` under its pad**: the masked `final_c` its round 0 absorbs is
`repCons`'s zerocheck proof's, the PIOP's masked value at `final_c`'s pad. -/
theorem finInner_fc {t : Tab} {rep : ℕ} {dom : ByteArray} {n : ℕ} {q : RepIn} {fc' : F128} {tc : ByteArray}
    (h : finInner t rep dom n q = .ok (fc', tc)) :
    fc' = q.vals[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]! +
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]! := by
  sorry

/-- **The finish accepts the request only at `vs`'s `final_c`**: at a history `InnerHist` holds at, on the worker's own
values (`innerReq_own`), where `finInner` returns at the request with `final_c = a`, its `zerocheck c_eval` constraint
holds on the pads (`consWhy`): `a` is the c_eval of the request's round-1c values at the replay's zerocheck point, the
opened coins' (`zcE`), which is `vs`'s `final_c` where no outer coin is `1` (`Loop.FinalCFixed`,
`outer_ne_one_ofLoopLink`). -/
theorem finInner_accepts_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) (hC : FinalCFixed vs x w) {ν : finS.Nonce} {t : finS.Tape}
    {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2}
    {e : ℕ} (he : e < 4) (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀)
    (a : F128)
    (hok : (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a)).isOk = true) :
    a = vsWord vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r (Masked.layE x j).fc := by
  sorry

/-- **The finish accepts the request at `vs`'s `final_c`**: at a history `InnerHist` holds at with `e < 4`, `finInner`
returns at the request with `final_c` `vs`'s. Its `repCons` returns at the stream's scheduled coins (the session's cut of
the rep returns there and `repCons` with it, `repCons_asksPad`), taking `nSched` rounds; its `zerocheck c_eval`
constraint holds on the pads at `vs`'s `final_c` (`Loop.FinalCFixed`), so `consWhy` passes; and `innerP` returns on the
coins past them, the stream's and then ones. -/
theorem finInner_returns_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) (hC : FinalCFixed vs x w) (hrows : RowsOK vs x w)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)}
    {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    ∃ (fc' : F128) (tc : ByteArray),
      finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
        (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e)
          (vsWord vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r (Masked.layE x j).fc)) =
        .ok (fc', tc) := by
  sorry

/-- **The finish's `τ` commitment at a rep's inner round 2**: at a history `InnerHist` holds at with `e = 2`, where the
rows hold (`Loop.RowsOK`), where `finInner` returns at the request, its `τ` commitment is `vs`'s (`Loop.vsTau`): it
accepts only at `vs`'s `final_c` (`finInner_accepts_own`), and its inner proof commits to `InZ.tau` over its constraints
at the coins of the inner proof's rounds 0 and 1, the stream's, which where the request's values are `vs`'s and its pads
the tape's are the switched view's first rows (`Composed.tauFw_at_coef`, `Masked.swTau_of_rows`). -/
theorem finInner_tau_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) (hC : FinalCFixed vs x w) (hrows : RowsOK vs x w)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)}
    {j : Fin (Masked.jOf x)} {r : Fin 2} (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r 2)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀)
    {a fc' : F128} {tc : ByteArray}
    (hok : finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + 2) a) = .ok (fc', tc)) :
    tc = vsTau vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r := by
  sorry

end Loop

end Claims.ZKServer
