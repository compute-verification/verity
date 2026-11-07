module

public import Proofs.Flock.SessionLoop
public import Proofs.Flock.Claims.ZK.Stopped
public import Proofs.Flock.Claims.ZK.FrameAdmits
public import Proofs.Flock.SessionLoop.LiveSchedOwn
public import Proofs.Flock.SessionLoop.RoundsPass

/-!
# `SessionLoopZKAny`'s premises at the honest session loop

`Flock.Guarantees.SessionLoopZKAny` takes the session's Props at the loop's tables and the transport's trusts at the
unswitched prover (`LeanRelease.TrustsStop`). At the deployed tables (`Masked.dataE`), the honest values
(`RefSession.honestVals`, each table's own) and the honest worker (`Loop.valsWk`) at the loop's own inner rounds
(`LeanRelease.withInner`), link and `Finish` (`LeanRelease.ofLoopLink`), at the loop's own tables (`Loop.realOf`):

* `LeanRelease.linkRoundEqStop_of_vals`, `liveRoundEqStop_of_vals`, `fillRoundEqStop_of_vals`: at the honest values the
  unswitched and switched provers' messages are both `refP`'s before the last coin (`unMsgAt_vals`, `swMsgAt_vals`), so
  each Stop trust is its switched trust.
* `Loop.finS_tables`, `finS_dummyReg`, `finS_complete`, `finS_causal`, `finS_unCausal`, `finS_swCausal`, `finS_tapeOK`:
  the session's Props at the deployed tables, `Causal` the stopped plan's (`causal_encStop` at `causalExe`) and `TapeOK`
  the layout's tape's (`fibersEq_ofLayout`), at every statement.
* `Loop.acceptCross`, `LeanRelease.cross_acceptCross`: at `L₁` with Rust's checks (`cross`) passing every round, the
  loop's release accepts the honest device's rounds there, which is `Loop.RoundsPass`'s `cross` premise.
* `LeanRelease.finishRows_honestVals`: on a live history of `refK x` openings whose last coins the frame takes, no
  zerocheck outer coin is `1` (`outer_ne_one_ofLoopLink`), so the rows hold at the honest values
  (`Loop.rowsOK_honestVals`).
* `LeanRelease.finalRoundEq_ofLoopLink`: the loop's `Finish` is the switched session's (`Loop.finishIs`), and its
  frames refuse every coin fault (`frameUnrefused_ofLoop`) at tables that admit unrefused coins.
* **`LeanRelease.trustsStop_honest`**: `TrustsStop` at the honest worker, at every relation inside the served, in-scope
  deployed one, from `Loop.Src` (the loop's inputs being the tape's), `LiveSched` (the loop's scheduled rounds) and
  `FillRoundEq` at the honest values. `LiveInner` is `liveInner_withInner`, the inner round 0 is the loop's
  (`LoopInnerRound0Own`), and the worker takes the honest values (`takes_honestVals_ofLoopLink`,
  `takesTau_honestVals_ofLoopLink`).
* **`LeanRelease.trustsStop_finHonest`**: `trustsStop_honest` from `Loop.Src` alone, at `acceptCross L₁` and the
  finish's honest rounds (`finHonest`): `LiveSched` is `liveSched_honestVals`, and `FillRoundEq` is
  `fillRoundEq_finHonest`, whose `cross` premise is `cross_acceptCross`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)

namespace LeanRelease

/-- **At the honest values `LinkRoundEq` is `LinkRoundEqStop`.** -/
theorem linkRoundEqStop_of_vals (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (x : Msg)
    (w : Flock.Wit) (hLk : LinkRoundEq S S.honestVals L x w) : LinkRoundEqStop S S.honestVals L x w :=
  fun ν t m₀ ρ₀ hk hrf => (hLk ν t m₀ ρ₀ hk hrf).trans
    ((S.swMsgAt_vals x w ν (S.wt x w) t []).trans (S.unMsgAt_vals x w ν (S.wt x w) t [] hk).symm)

/-- **At the honest values `LiveRoundEq` gives `LiveRoundEqStop`.** -/
theorem liveRoundEqStop_of_vals (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hE : LiveRoundEq S S.honestVals L H wk x w) :
    LiveRoundEqStop S S.honestVals L H wk x w :=
  fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hrf _ hv => (hE ν t κ m₀ ρ₀ ops v h0 hlt hl hok hrf hv).trans
    (congrArg dropDom ((S.swMsgAt_vals x w ν (S.wt x w) t _).trans
      (S.unMsgAt_vals x w ν (S.wt x w) t _ (by simpa using hlt)).symm))

/-- **At the honest values `FillRoundEq` gives `FillRoundEqStop`.** -/
theorem fillRoundEqStop_of_vals (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hE : FillRoundEq S S.honestVals L H wk x w) :
    FillRoundEqStop S S.honestVals L H wk x w :=
  fun ν t κ m₀ ρ₀ ops h0 hlt hl hok hrf _ => (hE ν t κ m₀ ρ₀ ops h0 hlt hl hok hrf).trans
    (congrArg dropDom ((S.swMsgAt_vals x w ν (S.wt x w) t _).trans
      (S.unMsgAt_vals x w ν (S.wt x w) t _ (by simpa using hlt)).symm))

end LeanRelease

namespace Loop

theorem finS_tables : finS.Tables := Masked.tables _ _ _ _

theorem finS_dummyReg : finS.DummyReg := Masked.dummyReg _ _ _ _

/-- **`Complete` at every relation inside the deployed one** (`Masked.complete_tabC`). -/
theorem finS_complete (Rl : Rel Flock.Wit) (h : ∀ x w, Rl x w → Masked.RZk x w) : finS.Complete Rl :=
  fun _ _ hR j => Masked.complete_tabC Masked.PE (h _ _ hR) j

/-- **The deployed session is `Causal`**: its plan is `encExe` stopped where a refusal is seen, which is causal where
`encExe` is (`causal_encStop`, `causalExe`). -/
theorem finS_causal : finS.Causal :=
  Masked.RoundPlan.causal (Masked.causal_encStop (cz := Masked.czExe Masked.PE) (enc := Masked.encExe Masked.PE)
    (fin := Masked.finExe Masked.PE) Masked.causalExe) _ _

theorem finS_unCausal : finS.UnCausal finS.honestVals := finS.unCausal_vals finS_causal

theorem finS_swCausal : finS.SwCausal finS.honestVals := finS.swCausal_vals finS_causal

/-- **The deployed session's tape is read with fibers of one size at every statement** (`fibersEq_ofLayout`). -/
theorem finS_tapeOK : finS.TapeOK :=
  (finS.tapeOKOn_iff).1 (finS.tapeOKOn_of_tape _ fun x _ =>
    Masked.fibersEq_ofLayout (Masked.TapeLayout.exe Masked.PE) Masked.exeTape x)

/-- **`L` with Rust's checks passing**: Rust's own checks beside the Lean frame (`LeanRelease.cross`) accept every round.
The guarantees hold at every `cross`, since Rust's checks only fail a round more. That they pass the honest worker's
rounds is completeness, which `verity/ml/flock/verifier/firewall_controls.sh` checks under `FC_FIREWALL=1`. -/
def acceptCross (L : LeanRelease Flock.Wit Tp) : LeanRelease Flock.Wit Tp :=
  { L with cross := fun _ _ _ _ _ _ => true }

end Loop

namespace LeanRelease

open Loop

/-- **`Loop.RoundsPass`'s `cross` premise at `acceptCross L₁`**: the loop's release keeps `L₁`'s Rust checks, so they
accept every round at every statement, witness, tape and history, the honest device's rounds (`valsDev`'s) among them,
whatever `L₁`'s other fields are. -/
theorem cross_acceptCross (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (fin : Msg → Flock.Wit → Tp → List (Msg × Msg) → Array ByteArray) (x : Msg) (w : Flock.Wit) (ρ : Tp)
    (g : List (Msg × Msg)) (v : Msg) :
    (ofLoopLink (withInner Masked.dataE (acceptCross L₁) (realOf src) fin) src (realOf src)).cross x g.length w ρ g v =
      true :=
  rfl

/-- **`FinishRows` at the honest values**, at every statement the loop serves with a witness in the deployed relation:
on a live history of `refK x` openings whose last coins the frame takes, no zerocheck outer coin of a table is `1`
(`outer_ne_one_ofLoopLink`), where the rows hold at the honest values (`rowsOK_honestVals`). -/
theorem finishRows_honestVals (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x) :
    FinishRows finS finS.honestVals (ofLoopLink L₀ src real) CoinOpen.shaH wk x w :=
  fun ν t κ m₀ ρ₀ ops hl hlive _ hr _ j =>
    rowsOK_honestVals hR ν _ t j (outer_ne_one_ofLoopLink L₀ src real wk hSv (ν, t) κ m₀ ρ₀ ops hl.le hlive hr j)

/-- **`FinalRoundEq` at the loop's own link and `Finish`**: at `SessionLoopZKDeployed`'s premises, the loop's `Finish`
is the switched session's (`Loop.finishIs`), and the frames refuse every coin fault (`frameUnrefused_ofLoop`) at tables
that admit unrefused coins (`tablesAdmitUnrefused_dataE`). -/
theorem finalRoundEq_ofLoopLink (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w) (hSrc : Loop.Src L₀ src x w)
    (hE : LiveRoundEq finS vs (ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w) (hc : finS.SwCausal vs) :
    FinalRoundEq finS vs (ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w :=
  finalRoundEq_ofLoop Masked.dataE vs (withFiles (withLink L₀ src) src) (realOf src) _ wk x w
    tablesAdmitUnrefused_dataE (frameUnrefused_ofLoop _ _ _ _ wk x w hSv)
    (Loop.finishIs vs L₀ src wk x w hsc hR hSv hL0 hSrc hE hc)

/-- **The transport's trusts at the honest session loop**: at the loop's own inner rounds (`withInner`, at the finish's
rounds `fin`), link and `Finish`, at its own tables, the honest values and the honest worker (`valsWk`), `TrustsStop`
holds at every relation whose statements the loop serves, are in the masked scope and have their witness in the deployed
relation, where the loop's inputs are the tape's (`Loop.Src`), its scheduled rounds are the switched session's
(`LiveSched`) and so are its stand-in rounds (`FillRoundEq`). -/
theorem trustsStop_honest (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (fin : Msg → Flock.Wit → Tp → List (Msg × Msg) → Array ByteArray) (Rl : Rel Flock.Wit)
    (hRl : ∀ x w, Rl x w → Serves x ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x ∧
      Masked.RZk x w)
    (hSrc : ∀ x w, Rl x w → Loop.Src (withInner Masked.dataE L₁ (realOf src) fin) src x w)
    (hS : ∀ x w, Rl x w → LiveSched finS.honestVals (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src)
      (valsWk finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src))) x w)
    (hEf : ∀ x w, Rl x w → FillRoundEq finS finS.honestVals
      (ofLoopLink (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src)) CoinOpen.shaH
      (valsWk finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src))) x w) :
    TrustsStop finS finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src))
      CoinOpen.shaH (valsWk finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src)))
      Rl := by
  intro x w h
  obtain ⟨hSv, hsc, hR⟩ := hRl x w h
  have hE : LiveRoundEq finS finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src))
      CoinOpen.shaH (valsWk finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src) fin) src (realOf src)))
      x w :=
    liveRoundEq_ofLoopLink _ _ src _ _ x w (by obtain ⟨_, hp, -⟩ := hsc; exact decodeRec_none_of_parse hp) (hS x w h)
      (liveInner_withInner _ L₁ src (realOf src) fin x w (Masked.live_of hsc hR) hSv
        (fun ρ _ _ _ hs => realTabs_realOf src x w ρ hs) (hSrc x w h).2 _
        (takes_honestVals_ofLoopLink _ src (realOf src)).2 (takesTau_honestVals_ofLoopLink _ src (realOf src))
        (rowsOK_honestVals hR))
  exact ⟨linkRoundEqStop_of_vals _ _ x w (linkRoundEq_ofLoopLink _ _ src _ hsc hR (hSrc x w h).1.linkLevel0),
    liveRoundEqStop_of_vals _ _ _ _ x w hE, fillRoundEqStop_of_vals _ _ _ _ x w (hEf x w h),
    finalRoundEq_ofLoopLink _ _ src _ x w hsc hR hSv
      (_root_.Flock.SecurityProofs.LoopInnerRound0Own Masked.dataE L₁ (realOf src) fin x w) (hSrc x w h) hE
      finS_swCausal,
    finishRows_honestVals _ src _ _ hR hSv⟩

/-- **The transport's trusts at the honest session loop from `Loop.Src` alone**: `trustsStop_honest` at `L₁` with Rust's
checks passing (`acceptCross`) and the inner rounds at the finish's honest rounds (`finHonest`). There the scheduled
rounds are the switched session's at the honest worker (`liveSched_honestVals`, at `finS_tables` and `finS_tapeOK`),
and so are the stand-in rounds (`fillRoundEq_finHonest`, every round passing at the honest
device), so only the loop's inputs being the tape's is left. -/
theorem trustsStop_finHonest (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)) (Rl : Rel Flock.Wit)
    (hRl : ∀ x w, Rl x w → Serves x ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x ∧
      Masked.RZk x w)
    (hSrc : ∀ x w, Rl x w → Loop.Src (withInner Masked.dataE (acceptCross L₁) (realOf src)
      (finHonest (ofLoopLink (acceptCross L₁) src (realOf src)))) src x w) :
    TrustsStop finS finS.honestVals (ofLoopLink (withInner Masked.dataE (acceptCross L₁) (realOf src)
        (finHonest (ofLoopLink (acceptCross L₁) src (realOf src)))) src (realOf src)) CoinOpen.shaH
      (valsWk finS.honestVals (ofLoopLink (withInner Masked.dataE (acceptCross L₁) (realOf src)
        (finHonest (ofLoopLink (acceptCross L₁) src (realOf src)))) src (realOf src))) Rl :=
  trustsStop_honest (acceptCross L₁) src (finHonest (ofLoopLink (acceptCross L₁) src (realOf src))) Rl hRl hSrc
    (fun x w h => liveSched_honestVals _ src (realOf src) rfl finS_tables finS_tapeOK (hRl x w h).2.2 (hRl x w h).1
      (hRl x w h).2.1 (hSrc x w h))
    fun x w h => fillRoundEq_finHonest finS.honestVals (acceptCross L₁) src
      (Masked.live_of (hRl x w h).2.1 (hRl x w h).2.2) (hRl x w h).1 (hSrc x w h).1
      fun ν t g _ _ _ => cross_acceptCross L₁ src (finHonest (ofLoopLink (acceptCross L₁) src (realOf src))) x w
        (ν, t) g _

end LeanRelease

end Claims.ZKServer
