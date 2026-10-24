import Proofs.Flock.Claims.ZK.FinishSwInner
import Proofs.Flock.Claims.ZK.InnerTauVal
import Proofs.Flock.Claims.ZK.InnerLeaf

/-!
# `τ`'s commitment at the switched source

`FinishSwInner.swSrcE_innerSrc_frame` leaves `InnerSrc`'s `τ` commitment open. Where table `j`'s rows hold at its base,
the switched prover's `τ` leaf commits the worker's view's `τ` (`Coined.swContent`), under the leaf salt the honest view
reads as `y_τ`, so it is that `τ` committed under the honest pads' `τ` salt (`ytau_saltBytes`). That `τ` is `λ`'s batch
against `μ` on the worker's view's rows, which is `InZ.tau` at any constraints whose first seven rows are those rows
(`tau_of_rows`): the triple's three are the honest view's (`tripleE`), since `ρ_in`, `σ_in` don't move.

* `swViewE`, `rowsSwE`: table `j`'s view at the workers' values and its rep `r`'s ten rows.
* `swCutE_tauCom`, **`swSrcE_tauCom`**: the switched source's `τ` commitment is the worker's view's `τ` under the honest
  pads' `τ` salt.
* `SwRows`: `cons` is `rowsSwE`'s first seven rows, each of the layout's length (`TauRows` at the worker's view).
* **`swTau_of_rows`**: then `InZ.tau` over `cons` at the honest pads is the worker's view's `τ`.
* **`swSrcE_tau_frame`**, **`swSrcE_innerSrc_rows`**: `swSrcE_innerSrc_frame`'s `τ` premise from `SwRows`.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf)

namespace Masked

/-- Table `j`'s view at the workers' values `v`, at its base on the tape `τ`. -/
noncomputable abbrev swViewE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    ((tabC PE x j).T (czExe PE x c j)).tab.View :=
  ((tabC PE x j).T (czExe PE x c j)).tab.toBase
    (((tabC PE x j).T (czExe PE x c j)).tab.onVals.view (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0))

/-- Table `j`'s rep `r` ten constraint rows (`Composed.rowsOf`) on its view at the workers' values. -/
noncomputable abbrev rowsSwE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    Fin (7 + 3) → ZkSession.Lin FlockLevel3.GF128 ((tabC PE x j).z.nE + 4) :=
  Composed.rowsOf (zkTab x j).table (stAt x j) (czExe PE x c j) r ((swViewE x c w v τ j).pre.2.1 r)
    (swViewE x c w v τ j).pre.1 ((swViewE x c w v τ j).pre.2.2.2.2.2 r).1 ((swViewE x c w v τ j).pre.2.2.2.2.1 r)

set_option smartUnfolding false in
/-- **Where table `j`'s rows hold at its base, the switched cut's `τ` commitment is the worker's view's `τ`** under the
honest view's `y_τ` (the leaf salt at `τ`'s position). -/
theorem swCutE_tauCom (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    (swCutE x c w v τ j).tauCom r = ZkView.leafT ((swViewE x c w v τ j).tau r) ((viewE x c w τ j).ytau r) := by
  have hb : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j) ((UE.tape x τ).1 j).1
      ((UE.tape x τ).1 j).2.1.2.1 := hrows
  have e : (swCutE x c w v τ j).tauCom r = _root_.Flock.Hm96.Default512.leaf
      (((tabC PE x j).T (czExe PE x c j)).swContent (wtOf PE x w j) (v j) ((UE.tape x τ).1 j) (.inr (.inr r)))
      (ZkView.saltBytes ((UE.tape x τ).2.1 ⟨j, .inr (.inr r)⟩)) := rfl
  rw [e, ZkView.Coined.swContent, if_pos hb]
  rfl

/-- **Where table `j`'s rows hold at its base, the switched source's `τ` commitment is the worker's view's `τ`**
committed through SHA-512 under the honest pads' `τ` salt (`ytau_saltBytes`). -/
theorem swSrcE_tauCom (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    ((swSrcE x c w v τ j.val).rep r.val).tauCom =
      FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash (FlockLevel3.GF128.val ((swViewE x c w v τ j).tau r))
        (innerPadsE x τ j r).tauSalt := by
  rw [swSrcE_rep, ← ytau_saltBytes x c w τ j r]
  show (swCutE x c w v τ j).tauCom r = _
  rw [swCutE_tauCom x c w v τ j r hrows]
  rfl

/-- The worker's view's `τ` is `λ`'s batch against the tape's `μ`, on its rows. -/
theorem swViewTau_eq (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    fOf (FlockLevel3.GF128.val ((swViewE x c w v τ j).tau r)) =
      ∑ i, ZkSession.batchCoef (Composed.innerGamma (zkTab x j).table (stAt x j) (czExe PE x c j) r)
        (rowsSwE x c w v τ j r) (Composed.padEquiv _ i) * muE x τ j r i :=
  rfl

/-- The worker's view's `ρ_in`, `σ_in` are the honest view's: they read the pads alone. -/
theorem swViewE_inner (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    (swViewE x c w v τ j).pre.2.2.2.2.1 r = (viewE x c w τ j).pre.2.2.2.2.1 r :=
  rfl

/-- The inner proof's three triple rows are the worker's view's rows' last three. -/
theorem tripleSwE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : Fin 3)
    (i : Fin ((tabC PE x j).z.nE + 4)) :
    fOf ((_root_.Flock.Zk.triple (innerLayE x j) ((innerCoinsE c j r).ce 0)
        (FlockComplete.Replay.InZ.rho (innerLayE x j) (innerPadsE x τ j r) (innerCoinsE c j r))
        (FlockComplete.Replay.InZ.sigma (innerLayE x j) (innerPadsE x τ j r)))[q.val]!).2.a[i.val]! =
      (rowsSwE x c w v τ j r (Fin.natAdd 7 q)).coef i := by
  rw [tripleE x c w τ j r q i]
  show (Fin.append _ _ (Fin.natAdd 7 q) : ZkSession.Lin FlockLevel3.GF128 _).coef i =
    (Fin.append _ _ (Fin.natAdd 7 q) : ZkSession.Lin FlockLevel3.GF128 _).coef i
  rw [Fin.append_right, Fin.append_right]
  all_goals rfl

/-- **The seven constraints before the triple at the worker's view**: `cons` is `rowsSwE`'s first seven rows by
coefficient, each of the layout's length (`TauRows` at the worker's view). -/
def SwRows (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (cons : Array (String × _root_.Flock.Zk.Lin)) : Prop :=
  cons.size = 7 ∧ (∀ k < 7, (cons[k]!).2.a.size = (innerLayE x j).len) ∧
    ∀ (k : Fin 7) (i : Fin ((tabC PE x j).z.nE + 4)),
      fOf (cons[k.val]!).2.a[i.val]! = (rowsSwE x c w v τ j r (Fin.castAdd 3 k)).coef i

/-- **`τ` over the worker's view's rows**: where `cons` is `rowsSwE`'s first seven rows, `InZ.tau` over it at the
honest pads and the rep's coins is the worker's view's `τ` (`tau_of_rows`, the triple by `tripleSwE`). -/
theorem swTau_of_rows {x : Msg} {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (cons : Array (String × _root_.Flock.Zk.Lin)) (h : SwRows x c w v τ j r cons) :
    FlockComplete.Replay.InZ.tau (innerLayE x j) (innerPadsE x τ j r) cons (innerCoinsE c j r) =
      FlockLevel3.GF128.val ((swViewE x c w v τ j).tau r) := by
  obtain ⟨hsz, hasz, hrow⟩ := h
  apply FlockSoundness.Refine.fOf_injective
  rw [swViewTau_eq]
  refine tau_of_rows (innerLayE x j) (innerPadsE x τ j r) cons (innerCoinsE c j r) _ (rowsSwE x c w v τ j r)
    (muE x τ j r) (innerLayE_len x j) hsz ?_ ?_ (gammaE hs c j r) (muE_eq x τ j r)
  · intro k hk
    unfold FlockComplete.Replay.InZ.all
    by_cases h7 : k < 7
    · rw [ZkExec.append_get!_left _ _ (by omega)]
      exact hasz k h7
    · rw [ZkExec.append_get!_right _ _ (j := k - 7) (by omega) (by simp [_root_.Flock.Zk.triple]; omega)]
      exact triple_size _ _ _ _ _ (by omega)
  · intro k i
    unfold FlockComplete.Replay.InZ.all
    induction k using Fin.addCases with
    | left k =>
      rw [Fin.val_castAdd, ZkExec.append_get!_left _ _ (by omega)]
      exact hrow k i
    | right q =>
      rw [Fin.val_natAdd, ZkExec.append_get!_right _ _ (j := q.val) (by omega)
        (by simp [_root_.Flock.Zk.triple])]
      exact tripleSwE x c w v τ j r q i

/-- **`τ`'s commitment at the switched source, at the firewall's pads**: where table `j`'s rows hold at its base, the
firewall's pads are the request's draws and the tape's pads are those draws' (`pads_frame_eq`'s premises), and `cons`
is the worker's view's first seven rows (`SwRows`), the switched source's `τ` commitment is `τ` over `cons` at the
firewall's pads and the inner proof's coins, committed through SHA-512 under the pads' `τ` salt. -/
theorem swSrcE_tau_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (cons : Array (String × _root_.Flock.Zk.Lin)) (hcons : SwRows x c w v τ j r cons) {L₀ : ℕ}
    (hL : L₀ = repSlot x (2 * j.val + r.val) (innerOff 0)) :
    ((swSrcE x c w v τ j.val).rep r.val).tauCom =
      FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash
        (FlockComplete.Replay.InZ.tau (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC cons
          (FlockComplete.SimInner.cutCoins (ligCoin c) L₀))
        (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.tauSalt := by
  have hP := pads_frame_eq L ρ t₀ rep x τ j r q hq hlay hz hpads
  rw [hP, ← innerLayE_layE, hL, swSrcE_tauCom x c w v τ j r hrows]
  exact congrArg (fun t => FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash t (innerPadsE x τ j r).tauSalt)
    (swTau_of_rows hs c w v τ j r cons hcons).symm

/-- **`repCut_fw`'s `hinner` at the switched source from the worker's view's rows**: `swSrcE_innerSrc_frame` with its
`τ` premise met by `swSrcE_tau_frame`, so `InnerSrc` holds at the switched source and the firewall's pads wherever the
batched constraint holds there and `cons` is the worker's view's first seven rows. -/
theorem swSrcE_innerSrc_rows {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hn : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (cons : Array (String × _root_.Flock.Zk.Lin)) (hcons : SwRows x c w v τ j r cons) {L₀ : ℕ}
    (hL : L₀ = repSlot x (2 * j.val + r.val) (innerOff 0))
    (hbat : L₀ + 1 < n →
      ((FlockComplete.Replay.InZ.lin (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC cons
          (FlockComplete.SimInner.cutCoins (ligCoin c) L₀)).c +
        _root_.Flock.dot (FlockComplete.Replay.InZ.lin (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
          cons (FlockComplete.SimInner.cutCoins (ligCoin c) L₀)).a
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h).isZero = true) :
    FlockComplete.SimInner.InnerSrc (ligCoin c) n _root_.Flock.Sha512.hash (layE x j)
      (_root_.Flock.Firewall.Finish.padsOf q.draws).toC cons ((swSrcE x c w v τ j.val).rep r.val) L₀ :=
  swSrcE_innerSrc_frame L ρ t₀ rep x hx w τ c n hn s spec hs j r v q hq hlay hz hpads hrows cons hL hbat
    fun _ => swSrcE_tau_frame L ρ t₀ rep x w τ c hs j r v q hq hlay hz hpads hrows cons hcons hL

end Masked
end Claims.ZKMal
