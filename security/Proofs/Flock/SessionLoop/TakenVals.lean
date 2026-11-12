module

public import Proofs.Flock.Claims.ZK.FinishSwSrc
public import Proofs.Flock.Claims.ZK.FrameFree
public import Proofs.Flock.Claims.ZK.FrameTape
public import Proofs.Flock.Claims.ZK.Round1Root
import Init.Data.Nat.Power2.Basic
public import Proofs.Flock.Claims.ZK.FrameLive
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.FinishSwInner
import Proofs.Flock.Claims.ZK.FinishSwMkAny
import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FrameRanked
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask

/-!
# The values the frame took, as the switched session's values

The session loop's frame takes each value of a rep's algebraic phase as the free word the worker sent there, and
releases it masked, `v + h_i` under the rep's pad `h_i` (`Frame.Masker.absorb`); the loop reads the value back as the
released word plus the pad (`Loop.valsOf`). The switched session at the deployed tables (`CoinLayer.session Loop.loopNr
Masked.dataE`) masks the values `vs` gives at the opened coins the same way (`Masked.swSrcE_masked_pad_any`). This file
states the switched session's side by pad index, as the ZK server's words:

* **`Loop.vsWord`**: table `j`'s rep `r` value `k` of `vs` at the opened coins and what the workers see of the tape
  (`RefSession.seenOf`, as the switched session reads `vs`), at the table's base on the tape (its mask slot and level
  0's hiding, which a table's values may read), as the ZK server's word; `0` from `n_E` on.
* **`Loop.RowsOK`**: each table's rows hold at `vs` at its base, at every nonce, opened coins and tape at which none of
  the table's zerocheck outer coins is `1`. Where they fail the switched session commits its pads tree over the moved
  pads (`Table.swPads`), while the frame commits it over the sampled ones, so a rep's first round, which carries the pads
  root, is the switched message only where they hold. `Loop.rowsOK_honestVals`: they hold at the honest values.
* **`Loop.TakesVals`**: at every live round, each word the frame frees in the worker's round is `vs`'s value there.
* **`Loop.TakesFinalC`**: at the inner proof's round 0, which the frame frees nothing of, the worker's `final_c'` plus
  its pad is `vs`'s `final_c`.
* **`Loop.swSrcE_masked_vsWord`**: below `n_E`, the switched source's masked value is `vsWord` plus the session's pad
  (`padsOf`), with no premise on the rows.
* **`Loop.vsTau`**: table `j`'s rep `r` `τ` commitment at `vs`, the switched source's, a SHA-512 digest whatever the
  values (`swSrcE_tauCom_size`).

`LiveRoundEq` at `SessionLoopZKDeployed`'s release (`LeanRelease.ofLoopLink`) splits where the loop schedules a round of
its own and where it doesn't:

* `LeanRelease.scheduled_ofLoopLink`: past `Hello`, the round in a device round's place is the loop's own
  (`Loop.schedOf`) on the rep's table and salts `real`, and `L₀`'s where the loop schedules none.
* `Loop.InnerAt`: where it schedules none at a session's history: the round is rep stream `k`'s round `i`
  (`CoinLayer.locate`) at or past the end of the rounds `Loop.schedSoFar` schedules on the stream's table at the coins
  it answered, the inner proof's (`Frame.INNER_ROUNDS` of them, round `i - sched.size`). `Loop.schedOf_none_of_innerAt`:
  `schedOf` is none there. `Loop.schedOf_eq_none_iff`: at a statement that sets up a session (`0 < refK x`), no V\*
  part's, and a history past the link, `schedOf` is none exactly there.
* `LeanRelease.LiveSched`, `LeanRelease.LiveInner`: `LiveRoundEq`'s equation at the rounds the loop schedules and at
  the inner proof's. `LeanRelease.liveRoundEq_ofLoopLink`: the two give `LiveRoundEq` at a statement that is no V\*
  part's.
* `LeanRelease.outer_ne_one_ofLoopLink`: on a live history whose last coins the frame takes, the coins the openings
  open meet `RowsOK`'s outer-coin condition, so `RowsOK` gives the rows there.

`TakesVals` and `TakesFinalC` are satisfiable by a live worker:

* **`Loop.valsWk vs L`**: the honest worker at the device `Loop.valsDev vs L`, which sends the scheduled round with `vs`'s
  value in each word the frame frees, at the inner proof's round 0 `vs`'s `final_c` under its pad, and at its round 2
  `vs`'s `τ` commitment (`Loop.vsTauAt`, `vsTau` at the round's rep), which `Loop.TakesTau` reads.
  `Loop.live_valsWk`: its run is live short of the `Finish` wherever its rounds pass the frame's checks
  (`LeanRelease.live_withDevice`: the server's checks don't read the device).
* `Loop.takesVals_valsWk`, `Loop.takesFinalC_valsWk`: it meets both at every `vs`; `Loop.takes_honestVals_ofLoopLink`:
  at the honest values at the session loop. That the deployed device's rounds are `valsWk honestVals`'s is the device's
  completeness, checked empirically.
-/

@[expose] public section

namespace FlockSoundness.ZK.Table

set_option smartUnfolding false in
/-- **The rows at the honest values are the table's own**: where each rep's batched constraint holds at `w`'s view at
every sample, the rows hold at `w` and the table's values. -/
theorem rowsAt_vals {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (u : T.Mask) (R : T.Vis)
    (h : ∀ (ω : T.Rand) (r : Fin 2),
      ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view w ω).pre) :
    T.onVals.RowsAt (w, T.vals) u R :=
  fun ω _ _ r => h ω r

end FlockSoundness.ZK.Table

namespace Claims.ZKServer

open Claims Claims.ZKMal

section Vals

open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

/-- **`vs`'s value at pad `k`** of table `j`'s rep `r`, at the opened coins `c` and what the workers see of the tape `τ`
(`RefSession.seenOf`), at table `j`'s base on `τ` (its mask slot and level 0's hiding), as the ZK server's word; `0`
from `n_E` on. -/
noncomputable def vsWord (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (k : ℕ) :
    _root_.Flock.F128 :=
  if hk : k < (tabC PE x j).z.nE then
    FlockLevel3.GF128.val (vs x w ν c ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 ⟨k, hk⟩)
  else 0

/-- **The rows hold at `vs`**: at every nonce, opened coins and tape, at each table none of whose zerocheck outer coins
is `1`, the table's rows hold at `vs`'s values at its base (`Table.RowsAt`). At an outer coin `1` the honest pads miss
the inner proof's batched constraint (`CoinedSetup.Admits`), so the rows are asked for only where they hold at the
honest values (`rowsOK_honestVals`). -/
def RowsOK (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)),
    (∀ (r : Fin 2), ∀ y ∈ (tabC PE x j).rOuter (czExe PE x c j) r, y ≠ 1) →
    ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt
      (wtOf PE x w j, vs x w ν c ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x τ) j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1

/-- **The rows hold at the honest values** at every witness in the deployed relation: each table's share is valid
(`complete_tabC`), so it carries the public region data (`coinedOf_reg_of`), and the honest pads satisfy each rep's
batched constraint where its outer coins aren't `1` (`innerOf_holds_rep_of`). -/
theorem rowsOK_honestVals {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) :
    RowsOK (CoinLayer.session Loop.loopNr Masked.dataE).honestVals x w := by
  intro ν c τ j hout
  have hv := complete_tabC PE hR j
  have hreg := FlockSoundness.Discharge.Composed.coinedOf_reg_of PE (zkTab x j).table (stAt x j) (zkTab x j).wf.1
    (zkTab x j).wf.2 rfl (zkTab x j).whole (czExe PE x c j) (wtOf PE x w j)
    (FlockSoundness.Discharge.Composed.RegionsHold.of_satisfies hv)
  exact FlockSoundness.ZK.Table.rowsAt_vals ((tabC PE x j).T (czExe PE x c j)).tab (wtOf PE x w j) _ _
    fun ω r => FlockSoundness.Discharge.Composed.innerOf_holds_rep_of PE (zkTab x j).table (stAt x j)
      (zkTab x j).wf.1 rfl (zkTab x j).mask
      (FlockSoundness.Discharge.FailClosed.avoidsMask_of_shapeOk (zkTab x j).wf.2 (zkTab x j).hzk)
      (czExe PE x c j) r (hout r) (wtOf PE x w j) hv hreg ω

/-- **The worker's free words are `vs`'s values**: after every history `LiveRoundEq` reads (`0 < ops.length < refK x`
openings, live, whose openings pass under the tape's nonce), where the frame takes the stream's coins (`answered`) and
the worker's round `v` of rep stream `2j + r`, each word the frame frees in `v` (`Frame.roundFree`: the algebraic phase's
values, `v`'s `k`-th at the absorbed order's place `m.pos + k`) is `vs`'s value at that pad index (`vsWord`), at the coins
the openings open and the tape. -/
def TakesVals (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker)
    (out : ByteArray) (s : _root_.Flock.Firewall.Frame.Masker) (fr : Array ℕ) (j : Fin (jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m →
    _root_.Flock.Firewall.Frame.roundFree (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false } ⟨v.toArray⟩
      (L.coins x (ops.length + 1)) = .ok (out, s, fr) →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    ∀ k < fr.size,
      (_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]! =
        vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
          (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
          (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]!

/-- **The worker's `final_c'` is `vs`'s `final_c` under its pad**: after every history `LiveRoundEq` reads, where the
worker's round `v` of rep stream `2j + r` is the inner proof's round 0 (its label frame, then the 16-byte `final_c'`
between its headers, the only shape `Frame.innerOwn` takes there), `final_c'` plus the rep's pad at `final_c`'s index
(the value the frame took there, `Loop.valsOf`) is `vs`'s value at that index (`vsWord`). -/
def TakesFinalC (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (w' : ByteArray) (j : Fin (jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    w'.size = 16 →
    v = (_root_.Flock.Firewall.Frame.labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ _root_.Flock.Transcript.header 3 1 1 ++
      w' ++ _root_.Flock.Transcript.header 4 1 1).data.toList →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    finalCOf ⟨v.toArray⟩ + (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.getD (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc 0 =
      vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
        (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc

/-- **The switched source's masked value below `n_E` is `vsWord` plus the session's pad**, whatever the rows. -/
theorem swSrcE_masked_vsWord (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {k : ℕ}
    (hk : k < (tabC PE x j).z.nE) :
    ((swSrcE x c w (vs x w ν c ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x τ)) τ j.val).rep r.val).masked k =
      vsWord vs x w ν c τ j r k + (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  rw [swSrcE_masked_pad_any x c w (vs x w ν c ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x τ)) τ j r hk]
  unfold vsWord
  split
  · rfl
  · contradiction

/-- **`vs`'s `τ` commitment** at table `j`'s rep `r`, at the opened coins `c` and at table `j`'s base on the tape `τ`:
the switched source's (`Masked.swSrcE`), which where the rows hold commits the worker's view's `τ` (`swSrcE_tauCom`). -/
noncomputable def vsTau (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    ByteArray :=
  ((swSrcE x c w (vs x w ν c ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x τ)) τ j.val).rep r.val).tauCom

set_option smartUnfolding false in
/-- The switched source's `τ` commitment is a SHA-512 digest at any values. -/
theorem swSrcE_tauCom_size (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    ((swSrcE x c w v τ j.val).rep r.val).tauCom.size = 64 := by
  have e : (swCutE x c w v τ j).tauCom r = _root_.Flock.Hm96.Default512.leaf
      (((tabC PE x j).T (czExe PE x c j)).swContent (wtOf PE x w j) (v j) ((UE.tape x τ).1 j) (.inr (.inr r)))
      (FlockSoundness.Discharge.ZkView.saltBytes ((UE.tape x τ).2.1 ⟨j, .inr (.inr r)⟩)) := rfl
  rw [swSrcE_rep]
  show ((swCutE x c w v τ j).tauCom r).size = 64
  rw [e]
  exact FlockSoundness.Refine.sha512_size _

theorem vsTau_size (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    (vsTau vs x w ν c τ j r).size = 64 :=
  swSrcE_tauCom_size x c w (vs x w ν c ((CoinLayer.session Loop.loopNr Masked.dataE).seenOf x τ)) τ j r

end Loop

end Vals

section Inner

open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- Where `x` sets up no session, Flock's or a V\* part's, there is no round past the link. -/
theorem refK_of_none {x : Msg} (hs : FlockDeploy.setupOf true x = none) (hd : RecDeploy.decodeRec x = none) :
    refK x = 0 :=
  refK_none (sessOf_none hs (RecDeploy.setupO_none hd))

/-- **The inner proof's rounds** after the decoded coins `cs` at `x`, on the rep streams' tables and salts `real`: `x`
sets up a session, the round is rep stream `k`'s round `i` (`CoinLayer.locate` of rep round `cs.length - 1`), the
stream's table and salts are `real k`'s, and `i` is at or past the end of the rounds `schedSoFar` schedules there at the
coins the stream answered (`schedOf`'s): the stream's rounds past its scheduled ones, the inner proof's round
`i - sched.size`. -/
def InnerAt (real : ℕ → V (Tab × Array ByteArray)) (x : Msg) (cs : List FlockDeploy.Coin) : Prop :=
  ∃ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (q k i : ℕ) (t : Tab) (hi : Array ByteArray)
    (sched : Array (ByteArray × ℕ)),
    FlockDeploy.setupOf true x = some (s0, spec, sts) ∧ cs.length = q + 1 ∧
    CoinLayer.locate (roundsOf spec sts) 0 q = (k, i) ∧ real k = .ok (t, hi) ∧
    schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
      hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) = .ok sched ∧
    sched.size ≤ i

/-- `schedOf` at a session's history past the link, rep stream `k`'s round `i`: the stream's scheduled round `i` on its
table, the empty round where its table or schedule is refused. -/
theorem schedOf_at {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} {cs : List FlockDeploy.Coin}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q k i : ℕ}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    (hl : CoinLayer.locate (roundsOf spec sts) 0 q = (k, i)) :
    schedOf real x cs =
      match real k with
      | .ok (t, hi) =>
        match schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
            (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
              (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
            hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) with
        | .ok sched => sched[i]?.map (·.1)
        | .error _ => some .empty
      | .error _ => some .empty := by
  simp only [schedOf, hs, hq, hl]
  rfl

/-- **At the inner proof's rounds the loop schedules no round.** -/
theorem schedOf_none_of_innerAt {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} {cs : List FlockDeploy.Coin}
    (hI : Loop.InnerAt real x cs) : schedOf real x cs = none := by
  obtain ⟨s0, spec, sts, q, k, i, t, hi, sched, hs, hq, hl, hr, hss, hle⟩ := hI
  rw [schedOf_at hs hq hl, hr]
  dsimp only
  rw [hss]
  simp [hle]

/-- **Exactly the inner proof's rounds have no scheduled round**: at a statement that sets up a session, no V\* part's
(whose session the loop doesn't frame), and a history past the link, `schedOf` is none exactly at `InnerAt`. -/
theorem schedOf_eq_none_iff {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} {cs : List FlockDeploy.Coin}
    (hd : RecDeploy.decodeRec x = none) (hk : 0 < refK x) (hc : 0 < cs.length) :
    schedOf real x cs = none ↔ Loop.InnerAt real x cs := by
  refine ⟨fun hn => ?_, schedOf_none_of_innerAt⟩
  obtain ⟨q, hq⟩ : ∃ q, cs.length = q + 1 := ⟨cs.length - 1, by omega⟩
  rcases hs : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · exact absurd (refK_of_none hs hd) (by omega)
  rcases hl : CoinLayer.locate (roundsOf spec sts) 0 q with ⟨k, i⟩
  rw [schedOf_at hs hq hl] at hn
  rcases hr : real k with e | ⟨t, hi⟩
  · rw [hr] at hn
    simp at hn
  · rw [hr] at hn
    dsimp only at hn
    rcases hss : schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
          (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
        hi ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) with e | sched
    · rw [hss] at hn
      simp at hn
    · rw [hss] at hn
      exact ⟨s0, spec, sts, q, k, i, t, hi, sched, hs, hq, hl, hr, hss, by simpa using hn⟩

end Loop

namespace LeanRelease

/-- **The round in a device round's place at `ofLoopLink`**: past `Hello`, the loop's own (`Loop.schedOf`) on the rep's
table and salts `real`, and `L₀`'s where the loop schedules none (`Loop.InnerAt`: the inner proof's rounds). -/
theorem scheduled_ofLoopLink
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (p : Msg × Msg) (ops : List (Msg × Msg)) (vals : String → Option ByteArray) :
    (ofLoopLink L₀ src real).scheduled x w ρ (p :: ops) vals =
      match Loop.schedOf (real x w ρ) x (ops.map fun p => CoinLayer.rd p.2) with
      | some r => r.data.toList
      | none => L₀.scheduled x w ρ (p :: ops) vals :=
  rfl

/-- **`LiveRoundEq` at the rounds the loop schedules**: its equation after every history it reads at which the loop
schedules its own round `r` (`Loop.schedOf`), against which the frame passed the worker's round. -/
def LiveSched (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`LiveRoundEq` at the inner proof's rounds** (`Loop.InnerAt`): its equation after every history it reads at which
the loop schedules no round of its own, `L₀`'s round in the device round's place. -/
def LiveInner (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`LiveRoundEq` from the scheduled rounds and the inner proof's**: at `SessionLoopZKDeployed`'s release, where the
loop schedules its own round (`LiveSched`) and at the inner proof's rounds (`LiveInner`), at a statement that is no V\*
part's. -/
theorem liveRoundEq_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hd : RecDeploy.decodeRec x = none)
    (hS : LiveSched vs L₀ src real wk x w) (hI : LiveInner vs L₀ src real wk x w) :
    LiveRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  intro ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hv
  rcases hsc : Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) with _ | r
  · exact hI ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hv
      ((Loop.schedOf_eq_none_iff hd (by omega) (by simpa using hpos)).mp hsc)
  · exact hS ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hv hsc

/-- **The opened coins meet `RowsOK`'s outer-coin condition**: at a statement the loop serves, on a live history of at
most `refK x` openings whose last coins the frame takes, no zerocheck outer coin of a table at the coins the openings
open is `1`. One whose slot is among the openings the frame refused at `1` (`outer_unrefused`); one past them is the
default coin's `0`. -/
theorem outer_ne_one_ofLoopLink
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hSv : Loop.Serves x)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoopLink L₀ src real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoopLink L₀ src real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false) (j : Fin (Masked.jOf x)) :
    ∀ (r : Fin 2), ∀ y ∈ (Masked.tabC Masked.PE x j).rOuter (Masked.czExe Masked.PE x
      ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) j) r, y ≠ 1 := by
  intro r y hy
  obtain ⟨e, he, rfl⟩ := Masked.outerReads_czExe Masked.PE x _ j r y hy
  have hj := Masked.lt_of_jOf j
  rcases Nat.lt_or_ge (Masked.outerSlot x j.val r.val) ops.length with hb | hb
  · exact outer_unrefused Masked.dataE (withFiles (withLink L₀ src) src) real wk x w (Masked.live_of_lt j) hSv ρ κ
      m₀ ρ₀ ops hl hlive hr hj r.isLt hb (by rw [getElem!_pos (Masked.stsOf x) j.val hj]; exact he)
  · unfold Masked.outerAt
    split
    · show ((ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2).getD
        (Masked.outerSlot x j.val r.val) default).at e ≠ 1
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by rw [List.length_map]; exact hb), Option.getD_none]
      unfold FlockDeploy.Coin.at
      split
      · show (0 : F128) ≠ 1
        decide
      · decide
    · decide

end LeanRelease

end Inner

section Worker

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame roundFree INNER_ROUNDS)
open FlockSoundness.Refine (Frame encs toBytes_length)
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace ZkServer

theorem copy_withHonest {W R : Type} (S : ZkServer W R) (hn : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val)
    (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) :
    ({ S with honest := hn } : ZkServer W R).copy wk x w ρ h = S.copy wk x w ρ h :=
  rfl

theorem check_withHonest {W R : Type} (S : ZkServer W R) (hn : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val)
    (x : Msg) (j s : ℕ) (w : W) (ρ : R) (h : List (Msg × Msg)) (cp : ℕ → ℕ → Option S.Val) (κ : S.Chk)
    (u : Option S.Val) :
    ({ S with honest := hn } : ZkServer W R).check x j s w ρ h cp κ u = S.check x j s w ρ h cp κ u := by
  unfold check
  dsimp only
  split <;> rfl

theorem pass_withHonest {W R : Type} (S : ZkServer W R) (hn : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val)
    (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk) (h : List (Msg × Msg)) :
    ({ S with honest := hn } : ZkServer W R).pass wk x w ρ κ h = S.pass wk x w ρ κ h := by
  unfold pass
  rw [copy_withHonest]
  simp only [check_withHonest]

/-- **Liveness doesn't read the honest workers' values.** -/
theorem live_withHonest {W R : Type} (S : ZkServer W R) (hn : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val)
    (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk) (h : List (Msg × Msg)) :
    ({ S with honest := hn } : ZkServer W R).live wk x w ρ κ h = S.live wk x w ρ κ h := by
  unfold live
  simp only [pass_withHonest]

end ZkServer

namespace LeanRelease

/-- `L` with the device `d`. -/
def withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) : LeanRelease W R :=
  { L with device := d }

/-- **The server's checks don't read the device**: they read the workers' values and the ZK server's own rounds, so a run
is live at `L` with any device exactly where it is at `L`. -/
theorem live_withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) :
    (L.withDevice d).toSrv.live wk x w ρ κ h = L.toSrv.live wk x w ρ κ h :=
  L.toSrv.live_withHonest (L.withDevice d).toSrv.honest wk x w ρ κ h

theorem kindAt_withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) (x : Msg) (j : ℕ) :
    (L.withDevice d).kindAt x j = L.kindAt x j :=
  rfl

/-- Rounds `2` to the `Finish`'s exclusive are frame rounds. -/
theorem kindAt_masked_of {W R : Type} (L : LeanRelease W R) {x : Msg} {j : ℕ} (h1 : 2 ≤ j) (h2 : j ≠ L.last x) :
    L.kindAt x j = .masked := by
  unfold kindAt
  split_ifs with ha hb
  · omega
  · omega
  · rfl

/-- **A live frame round passes the frame's check**: where a live run's worker sends the device round `v` at a frame
round, the frame takes the stream's coins and `Frame.check` passes `v` against the scheduled round. -/
theorem check_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (v : Msg) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hk : L.kindAt x h.length = .masked) (hv : wk x w ρ h 0 = some (v, false)) :
    ∃ m out s', L.answered x ρ h = .ok m ∧
      _root_.Flock.Firewall.Frame.check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) =
        .ok (out, s', none) := by
  have hp := L.toSrv.pass_of_live wk x w ρ κ h hl (le_refl h.length)
  rw [List.take_length] at hp
  have hc : L.toSrv.check x h.length 0 w ρ h (L.toSrv.copy wk x w ρ h) κ (wk x w ρ h 0) = true :=
    List.all_eq_true.mp (ZkServer.slots_of_pass hp) 0 (List.mem_range.mpr Nat.one_pos)
  rw [hv] at hc
  have hrel : (!(v, false).2 && decide (h.length < L.last x) && L.cross x h.length w ρ h (v, false).1 &&
      L.pass x w ρ h (v, false).1 && L.takes x ρ h (v, false).1) = true := by
    unfold ZkServer.check at hc
    rw [show L.toSrv.kind x h.length 0 = .masked from hk] at hc
    exact hc
  simp only [Bool.and_eq_true] at hrel
  have hpass : L.pass x w ρ h v = true := hrel.1.2
  unfold pass at hpass
  cases ha : L.answered x ρ h with
  | error e => rw [ha] at hpass; cases hpass
  | ok m =>
    rw [ha] at hpass
    simp only at hpass
    cases hc' : _root_.Flock.Firewall.Frame.check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
    | error e => rw [hc'] at hpass; cases hpass
    | ok q =>
      obtain ⟨out, s', why⟩ := q
      cases why with
      | some e => rw [hc'] at hpass; cases hpass
      | none => exact ⟨m, out, s', rfl, hc'⟩

end LeanRelease

namespace Loop

/-! ### A round with values written into its free words -/

/-- `L` with its 16 bytes from `o` set to `a`'s. -/
def setWord (L : List UInt8) (o : ℕ) (a : F128) : List UInt8 := L.take o ++ (a.toBytes.data.toList ++ L.drop (o + 16))

/-- **`L` with each word the offsets `fr` name set to `vals`' value**: the `k`-th, at `fr[k]`, to `vals k`. -/
def fill (L : List UInt8) (fr : Array ℕ) (vals : ℕ → F128) : List UInt8 :=
  (List.range fr.size).foldl (fun L k => setWord L fr[k]! (vals k)) L

theorem setWord_length {L : List UInt8} {o : ℕ} (a : F128) (h : o + 16 ≤ L.length) :
    (setWord L o a).length = L.length := by
  simp only [setWord, List.length_append, List.length_take, List.length_drop, toBytes_length]
  omega

theorem setWord_drop {L : List UInt8} {o : ℕ} (a : F128) (h : o + 16 ≤ L.length) :
    (setWord L o a).drop o = a.toBytes.data.toList ++ L.drop (o + 16) :=
  List.drop_left' (by rw [List.length_take]; omega)

theorem setWord_getElem?_lt {L : List UInt8} {o i : ℕ} (a : F128) (h : o ≤ L.length) (hi : i < o) :
    (setWord L o a)[i]? = L[i]? := by
  unfold setWord
  rw [List.getElem?_append_left (by rw [List.length_take]; omega)]
  simp [hi]

/-- A word before the one `setWord` sets is kept. -/
theorem setWord_window {L : List UInt8} {o p : ℕ} (a : F128) (h : o + 16 ≤ L.length) (hp : p + 16 ≤ o) :
    ((setWord L o a).drop p).take 16 = (L.drop p).take 16 := by
  apply List.ext_getElem?
  intro i
  simp only [List.getElem?_take, List.getElem?_drop]
  split
  · exact setWord_getElem?_lt a (by omega) (by omega)
  · rfl

theorem fill_length (L : List UInt8) (fr : Array ℕ) (vals : ℕ → F128)
    (hb : ∀ i < fr.size, fr[i]! + 16 ≤ L.length) :
    ∀ m, m ≤ fr.size → ((List.range m).foldl (fun L k => setWord L fr[k]! (vals k)) L).length = L.length
  | 0, _ => rfl
  | m + 1, hm => by
    have ih := fill_length L fr vals hb m (by omega)
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
      setWord_length _ (by rw [ih]; exact hb m (by omega)), ih]

theorem fill_window (L : List UInt8) (fr : Array ℕ) (vals : ℕ → F128)
    (hs : ∀ i j, i < j → j < fr.size → fr[i]! + 16 ≤ fr[j]!) (hb : ∀ i < fr.size, fr[i]! + 16 ≤ L.length) (k : ℕ) :
    ∀ m, k < m → m ≤ fr.size →
      (((List.range m).foldl (fun L k => setWord L fr[k]! (vals k)) L).drop fr[k]!).take 16 =
        (vals k).toBytes.data.toList
  | 0, hk, _ => absurd hk (Nat.not_lt_zero k)
  | m + 1, hk, hm => by
    have hl := fill_length L fr vals hb m (by omega)
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    rcases Nat.lt_or_ge k m with hlt | hge
    · rw [setWord_window _ (by rw [hl]; exact hb m (by omega)) (hs k m hlt (by omega))]
      exact fill_window L fr vals hs hb k m hlt (by omega)
    · obtain rfl : k = m := by omega
      rw [setWord_drop _ (by rw [hl]; exact hb k (by omega)), List.take_left' (toBytes_length _)]

/-- **The words `fill` sets read back**, at offsets each 16 bytes past the one before and inside `L`. -/
theorem fill_wd (L : List UInt8) (fr : Array ℕ) (vals : ℕ → F128)
    (hs : ∀ i j, i < j → j < fr.size → fr[i]! + 16 ≤ fr[j]!) (hb : ∀ i < fr.size, fr[i]! + 16 ≤ L.length)
    {k : ℕ} (hk : k < fr.size) : wd (fill L fr vals) fr[k]! = vals k := by
  have hw : ((fill L fr vals).drop fr[k]!).take 16 = (vals k).toBytes.data.toList :=
    fill_window L fr vals hs hb k fr.size hk le_rfl
  apply wd_drop (rest := ((fill L fr vals).drop fr[k]!).drop 16)
  rw [← hw, List.take_append_drop]

/-! ### The words a round frees are in order, 16 bytes apart -/

/-- **Offsets of 16-byte words in order**, each starting past the one before and ending by `p`. -/
def Spaced (fr : List ℕ) (p : ℕ) : Prop := fr.Pairwise (fun a b => a + 16 ≤ b) ∧ ∀ a ∈ fr, a + 16 ≤ p

theorem Spaced.mono {fr : List ℕ} {p p' : ℕ} (h : Spaced fr p) (hp : p ≤ p') : Spaced fr p' :=
  ⟨h.1, fun a ha => Nat.le_trans (h.2 a ha) hp⟩

theorem Spaced.push {fr : List ℕ} {p : ℕ} (h : Spaced fr p) : Spaced (fr ++ [p]) (p + 16) := by
  refine ⟨List.pairwise_append.2 ⟨h.1, List.pairwise_singleton _ _, fun a ha b hb => ?_⟩, fun a ha => ?_⟩
  · rw [List.mem_singleton] at hb
    subst hb
    exact h.2 a ha
  · rcases List.mem_append.1 ha with ha | ha
    · exact Nat.le_trans (h.2 a ha) (by omega)
    · rw [List.mem_singleton] at ha
      omega

theorem Spaced.get {fr : Array ℕ} {p : ℕ} (h : Spaced fr.toList p) :
    (∀ i j, i < j → j < fr.size → fr[i]! + 16 ≤ fr[j]!) ∧ ∀ i < fr.size, fr[i]! + 16 ≤ p := by
  refine ⟨fun i j hij hj => ?_, fun i hi => ?_⟩
  · rw [getElem!_pos fr i (by omega), getElem!_pos fr j hj]
    have := List.pairwise_iff_getElem.1 h.1 i j (by rw [Array.length_toList]; omega)
      (by rw [Array.length_toList]; exact hj) hij
    simpa using this
  · rw [getElem!_pos fr i hi]
    exact h.2 _ (by simp)

theorem absorb1_spaced (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (v : F128)
    (h : absorb1 c r v = .ok r') (hs : Spaced r.2.2.1.toList r.2.2.2) : Spaced r'.2.2.1.toList r'.2.2.2 := by
  unfold absorb1 at h
  cases hb : r.1.absorb c v with
  | error e => rw [hb] at h; cases h
  | ok b =>
    rw [hb] at h
    cases h
    dsimp only
    split
    · rw [Array.toList_push]
      exact hs.push
    · exact hs.mono (by omega)

theorem foldlM_absorb1_spaced (c : Cfg) : ∀ (vs : List F128) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    vs.foldlM (absorb1 c) r = .ok r' → Spaced r.2.2.1.toList r.2.2.2 → Spaced r'.2.2.1.toList r'.2.2.2
  | [], r, r', h, hs => by cases h; exact hs
  | v :: vs, r, r', h, hs => by
    rw [List.foldlM_cons] at h
    cases ha : absorb1 c r v with
    | error e => rw [ha] at h; cases h
    | ok a =>
      rw [ha] at h
      exact foldlM_absorb1_spaced c vs a r' h (absorb1_spaced c r a v ha hs)

theorem stepAt_spaced (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (f : Frame) (h : stepAt c r f = .ok r')
    (hs : Spaced r.2.2.1.toList r.2.2.2) : Spaced r'.2.2.1.toList r'.2.2.2 := by
  cases f with
  | dom b => cases h; exact hs.mono (by dsimp only; omega)
  | label b =>
    simp only [stepAt] at h
    cases hl : r.1.label c b with
    | error e => rw [hl] at h; cases h
    | ok q => rw [hl] at h; cases h; exact hs.mono (by dsimp only; omega)
  | bytes b => cases h; exact hs.mono (by dsimp only; omega)
  | sq k n => cases h; exact hs.mono (by dsimp only; omega)
  | one v => exact foldlM_absorb1_spaced c [v] _ r' h (hs.mono (by dsimp only; omega))
  | vec vs => exact foldlM_absorb1_spaced c vs _ r' h (hs.mono (by dsimp only; omega))

theorem foldlM_stepAt_spaced (c : Cfg) : ∀ (fs : List Frame) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    fs.foldlM (stepAt c) r = .ok r' → Spaced r.2.2.1.toList r.2.2.2 → Spaced r'.2.2.1.toList r'.2.2.2
  | [], r, r', h, hs => by cases h; exact hs
  | f :: fs, r, r', h, hs => by
    rw [List.foldlM_cons] at h
    cases ha : stepAt c r f with
    | error e => rw [ha] at h; cases h
    | ok a =>
      rw [ha] at h
      exact foldlM_stepAt_spaced c fs a r' h (stepAt_spaced c r a f ha hs)

/-- **The words a round of frames frees are in order, 16 bytes apart, inside its frames.** -/
theorem roundFree_spaced (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hs : s.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) {q : ByteArray × Masker × Array ℕ}
    (hq : roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok q) :
    Spaced q.2.2.toList (encs fs).length := by
  rw [roundFree_encs c s fs k n hs hm hk h0 hn] at hq
  cases ha : fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) with
  | error e => rw [ha] at hq; cases hq
  | ok a =>
    rw [ha] at hq
    have := exMap_ok_eq hq
    subst this
    have hp := foldlM_stepAt_pos c fs _ a ha
    have hsp := foldlM_stepAt_spaced c fs _ a ha ⟨List.Pairwise.nil, fun _ h => by simp at h⟩
    dsimp only at hp
    rw [hp, Nat.zero_add] at hsp
    exact hsp

/-! ### The inner proof's round 0 frees nothing -/

/-- The inner proof's label leaves the masked phases shut: it needs them ended, and nothing reopens one. -/
theorem label_inner_on (c : Cfg) (s : Masker) {q : Masker × ByteArray}
    (h : s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 = .ok q) : q.1.on = false := by
  obtain ⟨p1, p2, p3, p4⟩ := Ranked.inner_other
  cases hi : s.inner with
  | some k =>
    simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi] at h
    cases h
  | none =>
    cases hd : s.done
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd] at h
      cases h
    · cases hon : s.on
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
        split at h
        · cases h
        · cases h
          rfl
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
        cases h

theorem absorb1_off (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (v : F128) (hon : r.1.on = false)
    (h : absorb1 c r v = .ok r') : r'.2.2.1 = r.2.2.1 := by
  unfold absorb1 at h
  cases hb : r.1.absorb c v with
  | error e => rw [hb] at h; cases h
  | ok b =>
    rw [hb] at h
    cases h
    simp [hon]

/-- **The inner proof's round 0 frees no word**: at its label, `final_c'` and `ε`'s squeeze, the frame frees nothing
wherever it takes the round. -/
theorem roundFree_inner0 {c : Cfg} {s : Masker} {w' : ByteArray} (hw : w'.size = 16) {n : ℕ}
    {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1) n =
      .ok q) : q.2.2 = #[] := by
  have hs := roundFree_ok_inner h
  have ht : (ByteArray.mk (Array.replicate 64 0)).size = 64 := by simp [ByteArray.size]
  have hX := ownRound_encs (c := c) (s := s) (n := n) hw ht (k := 0) (by decide)
  have hE : labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1 =
      ByteArray.mk (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0) ++ (Frame.sq 1 1).enc).toArray := by
    rw [show Frame.sq 1 1 = ownSq n 0 from rfl, ← hX, Array.toArray_toList]
    rfl
  rw [hE] at h
  have hsq := roundFree_sq hs h
  obtain ⟨-, -, hn1, -⟩ := hsq
  have hsz : (ByteArray.mk (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0) ++
      (Frame.sq 1 1).enc).toArray).size - 16 = (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0)).length := by
    rw [size_list, List.length_append, Frame.enc_length]
    simp [Frame.payLen]
  rw [hsz] at hn1
  have hh := (header_at (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0) ++ (Frame.sq 1 1).enc)
    (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0)).length (.sq 1 1) []
    (by rw [List.drop_left, List.append_nil]) (by decide)).2.2
  rw [hh] at hn1
  obtain rfl : n = 1 := hn1.symm
  rw [roundFree_encs c s _ 1 1 hs (ownFrames0_mid c s w' _) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)] at h
  obtain ⟨a, ha, h⟩ := bindV_ok h
  obtain rfl := exMap_ok_eq h
  simp only [ownFrames, List.foldlM_cons, List.foldlM_nil] at ha
  obtain ⟨b, hb, ha⟩ := bindV_ok ha
  obtain ⟨d, hd, ha⟩ := bindV_ok ha
  obtain rfl : d = a := Except.ok.inj ha
  simp only [stepAt] at hb
  cases hl : s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 with
  | error e => rw [hl] at hb; cases hb
  | ok ql =>
    rw [hl] at hb
    cases hb
    have hon := label_inner_on c s hl
    simp only [stepAt, List.foldlM_cons, List.foldlM_nil] at hd
    obtain ⟨e, he, hd⟩ := bindV_ok hd
    obtain rfl : e = d := Except.ok.inj hd
    exact absorb1_off c _ e _ hon he

/-! ### The worker that sends the scheduled rounds with `vs`'s values -/

/-- The inner proof's round 0 with `final_c'` `w'`: its label, `w'` and `ε`'s squeeze (`Frame.innerOwn`'s shape). -/
def inner0 (w' : ByteArray) : ByteArray :=
  labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1

theorem finalCOf_inner0 (a : F128) : finalCOf ⟨(inner0 a.toBytes).data.toList.toArray⟩ = a := by
  unfold finalCOf
  dsimp only
  apply wd_drop (rest := (header 4 1 1).data.toList)
  have hl : (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16 =
      ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).data.toList ++ (header 3 1 1).data.toList).length := by
    rw [List.length_append, FlockSoundness.Refine.Frame.header_length, Array.length_toList]
    try rfl
  simp only [inner0, ByteArray.toList_data_append]
  rw [hl, List.append_assoc, List.append_assoc, ← List.append_assoc (labelFrame _ ).data.toList, List.drop_left]

open Classical in
/-- `u` with its `final_c'` set to `a`'s bytes where `u` is the inner proof's round 0; else `u`. -/
noncomputable def setFinalC (a : F128) (u : List UInt8) : List UInt8 :=
  if ∃ w' : ByteArray, w'.size = 16 ∧ u = (inner0 w').data.toList then (inner0 a.toBytes).data.toList else u

/-- The inner proof's round 2 with `τ`'s commitment `t`: its bytes frame and `β`'s squeeze (`Frame.innerOwn`'s
shape). -/
def inner2 (t : ByteArray) : ByteArray := header 5 0 64 ++ t ++ header 4 1 1

open Classical in
/-- `u` with its `τ` commitment set to `t` where the frame's state `m` is at the inner proof's round 2 and `u` has that
round's shape; else `u`. Elsewhere a round of that shape (a 64-byte digest and one coin) is the frame's own. -/
noncomputable def setTau (m : Masker) (t : ByteArray) (u : List UInt8) : List UInt8 :=
  if m.inner = some 2 ∧ ∃ t' : ByteArray, t'.size = 64 ∧ u = (inner2 t').data.toList then (inner2 t).data.toList
  else u

theorem eq_setTau {v : List UInt8} {m : Masker} {t : ByteArray} {u : List UInt8} (h : v = setTau m t u) :
    v = (inner2 t).data.toList ∨ v = u := by
  unfold setTau at h
  split_ifs at h
  · exact Or.inl h
  · exact Or.inr h

theorem head_header_append (op kind : UInt8) (n : ℕ) (b : List UInt8) :
    ((header op kind n).data.toList ++ b).head? = some op := by
  rw [FlockSoundness.Refine.header_data]
  rfl

theorem inner2_head (t : ByteArray) : (inner2 t).data.toList.head? = some 5 := by
  simp only [inner2, ByteArray.toList_data_append, List.append_assoc, head_header_append]

theorem inner0_head (w : ByteArray) : (inner0 w).data.toList.head? = some 2 := by
  simp only [inner0, labelFrame, ByteArray.toList_data_append, List.append_assoc, head_header_append]

/-- Round 2's shape is never round 0's: their first frames differ. -/
theorem inner2_ne_inner0 (t w : ByteArray) : (inner2 t).data.toList ≠ (inner0 w).data.toList := by
  intro h
  have e := congrArg List.head? h
  rw [inner2_head, inner0_head] at e
  exact absurd (Option.some.inj e) (by decide)

/-- **The inner proof's round 2 frees no word**: at `τ`'s commitment and `β`'s squeeze, the frame frees nothing
wherever it takes the round. -/
theorem roundFree_inner2 {c : Cfg} {s : Masker} {t : ByteArray} (ht : t.size = 64) {n : ℕ}
    {q : ByteArray × Masker × Array ℕ} (h : roundFree c s (inner2 t) n = .ok q) : q.2.2 = #[] := by
  have hs := roundFree_ok_inner h
  have hw : (ByteArray.mk (Array.replicate 16 0)).size = 16 := by simp [ByteArray.size]
  have hX := ownRound_encs (c := c) (s := s) (n := n) hw ht (k := 2) (by decide)
  have hE : inner2 t =
      ByteArray.mk (encs (ownFrames c s (ByteArray.mk (Array.replicate 16 0)) t 2) ++ (Frame.sq 1 1).enc).toArray := by
    rw [show Frame.sq 1 1 = ownSq n 2 from rfl, ← hX, Array.toArray_toList]
    rfl
  have hmid : ∀ f ∈ ownFrames c s (ByteArray.mk (Array.replicate 16 0)) t 2, midFrame f := by
    intro f hf
    simp only [ownFrames, List.mem_cons, List.not_mem_nil, or_false] at hf
    subst hf
    show t.size < 2 ^ 64
    rw [ht]
    norm_num
  rw [hE] at h
  have hsq := roundFree_sq hs h
  obtain ⟨-, -, hn1, -⟩ := hsq
  have hsz : (ByteArray.mk (encs (ownFrames c s (ByteArray.mk (Array.replicate 16 0)) t 2) ++
      (Frame.sq 1 1).enc).toArray).size - 16 =
      (encs (ownFrames c s (ByteArray.mk (Array.replicate 16 0)) t 2)).length := by
    rw [size_list, List.length_append, Frame.enc_length]
    simp [Frame.payLen]
  rw [hsz] at hn1
  have hh := (header_at (encs (ownFrames c s (ByteArray.mk (Array.replicate 16 0)) t 2) ++ (Frame.sq 1 1).enc)
    (encs (ownFrames c s (ByteArray.mk (Array.replicate 16 0)) t 2)).length (.sq 1 1) []
    (by rw [List.drop_left, List.append_nil]) (by decide)).2.2
  rw [hh] at hn1
  obtain rfl : n = 1 := hn1.symm
  rw [roundFree_encs c s _ 1 1 hs hmid (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)] at h
  obtain ⟨a, ha, h⟩ := bindV_ok h
  obtain rfl := exMap_ok_eq h
  simp only [ownFrames, List.foldlM_cons, List.foldlM_nil] at ha
  obtain ⟨b, hb, ha⟩ := bindV_ok ha
  obtain rfl : b = a := Except.ok.inj ha
  cases hb
  rfl

/-- A round that is the bytes of frames, each before its squeeze, then its squeeze of `n` coins (`roundFree_encs`'s
form). -/
def Frames (r : List UInt8) (n : ℕ) : Prop :=
  ∃ (fs : List Frame) (k : UInt8), r = encs fs ++ (Frame.sq k n).enc ∧ (∀ f ∈ fs, midFrame f) ∧
    (k = 2 ∨ (k = 1 ∧ n = 1)) ∧ ¬ (n = 0 ∧ k = 2) ∧ n < 2 ^ 64

/-- **`vs`'s value at pad `i` for the round after `h`**: `vsWord` at rep stream `L.rep x |h|`'s table and rep, at the
coins `h`'s openings open and the tape; `0` past the tables. -/
noncomputable def vsAt (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (h : List (Msg × Msg)) (i : ℕ) : F128 :=
  if hj : L.rep x h.length / 2 < jOf x then
    vsWord vs x w ρ.1 ((CoinLayer.session Loop.loopNr Masked.dataE).pad
      (h.tail.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) ρ.2
      ⟨L.rep x h.length / 2, hj⟩ ⟨L.rep x h.length % 2, Nat.mod_lt _ (by decide)⟩ i
  else 0

theorem vsAt_eq (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (h : List (Msg × Msg)) (i : ℕ) {j : Fin (jOf x)} {r : Fin 2} (hrep : L.rep x h.length = 2 * j.val + r.val) :
    vsAt vs L x w ρ h i = vsWord vs x w ρ.1 ((CoinLayer.session Loop.loopNr Masked.dataE).pad
      (h.tail.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) ρ.2 j r i := by
  have hr := r.isLt
  have hj : L.rep x h.length / 2 < jOf x := by rw [hrep]; omega
  unfold vsAt
  rw [dif_pos hj]
  have e1 : (⟨L.rep x h.length / 2, hj⟩ : Fin (jOf x)) = j := Fin.ext (by dsimp only; rw [hrep]; omega)
  have e2 : (⟨L.rep x h.length % 2, Nat.mod_lt _ (by decide)⟩ : Fin 2) = r :=
    Fin.ext (by dsimp only; rw [hrep]; omega)
  rw [e1, e2]

/-- **`vs`'s `τ` commitment for the round after `h`**: `vsTau` at rep stream `L.rep x |h|`'s table and rep, at the
coins `h`'s openings open and the tape; empty past the tables. -/
noncomputable def vsTauAt (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (h : List (Msg × Msg)) : ByteArray :=
  if hj : L.rep x h.length / 2 < jOf x then
    vsTau vs x w ρ.1 ((CoinLayer.session Loop.loopNr Masked.dataE).pad
      (h.tail.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) ρ.2
      ⟨L.rep x h.length / 2, hj⟩ ⟨L.rep x h.length % 2, Nat.mod_lt _ (by decide)⟩
  else ByteArray.empty

theorem vsTauAt_eq (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (h : List (Msg × Msg)) {j : Fin (jOf x)} {r : Fin 2} (hrep : L.rep x h.length = 2 * j.val + r.val) :
    vsTauAt vs L x w ρ h = vsTau vs x w ρ.1 ((CoinLayer.session Loop.loopNr Masked.dataE).pad
      (h.tail.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) ρ.2 j r := by
  have hr := r.isLt
  have hj : L.rep x h.length / 2 < jOf x := by rw [hrep]; omega
  unfold vsTauAt
  rw [dite_eq_left hj]
  have e1 : (⟨L.rep x h.length / 2, hj⟩ : Fin (jOf x)) = j := Fin.ext (by dsimp only; rw [hrep]; omega)
  have e2 : (⟨L.rep x h.length % 2, Nat.mod_lt _ (by decide)⟩ : Fin 2) = r :=
    Fin.ext (by dsimp only; rw [hrep]; omega)
  rw [e1, e2]

theorem vsTauAt_size (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (h : List (Msg × Msg)) {j : Fin (jOf x)} {r : Fin 2} (hrep : L.rep x h.length = 2 * j.val + r.val) :
    (vsTauAt vs L x w ρ h).size = 64 := by
  rw [vsTauAt_eq vs L x w ρ h hrep]
  exact vsTau_size vs x w _ _ _ j r

open Classical in
/-- **The device round with `vs`'s values**: the scheduled round `r` in the device round's place after `h`, with `vs`'s
value written into each word the frame frees in `r` (`fill`, the `k`-th at the absorbed order's place `m.pos + k`), at
the inner proof's round 0 `final_c'` set to `vs`'s `final_c` under its pad (`setFinalC`), and at its round 2 `τ`'s
commitment set to `vs`'s (`setTau`). Where the frame refuses the stream's coins or `r` itself, the round fails whatever
the device sends, and it sends `r`; where `r` is not the bytes of frames and a squeeze, it sends the empty round, which
the frame refuses. -/
noncomputable def valsDev (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)) :
    Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → (String → Option ByteArray) → Msg :=
  fun x w ρ h cv =>
    match L.answered x ρ h with
    | .error _ => L.scheduled x w ρ h cv
    | .ok m =>
      match roundFree (L.cfg x ρ h) { m with stopped := false } ⟨(L.scheduled x w ρ h cv).toArray⟩
          (L.coins x h.length) with
      | .error _ => L.scheduled x w ρ h cv
      | .ok (_, _, fr) =>
        if Frames (L.scheduled x w ρ h cv) (L.coins x h.length) then
          setTau m (vsTauAt vs L x w ρ h)
            (setFinalC (vsAt vs L x w ρ h (L.cfg x ρ h).lay.fc + (L.cfg x ρ h).h.getD (L.cfg x ρ h).lay.fc 0)
              (fill (L.scheduled x w ρ h cv) fr fun k => vsAt vs L x w ρ h ((L.cfg x ρ h).order[m.pos + k]!)))
        else []

/-- **The worker that sends the scheduled rounds with `vs`'s values**: `L`'s honest worker at the device `valsDev vs
L`. Its run is live wherever its rounds pass the frame's checks (`live_valsWk`), and it meets `TakesVals` and
`TakesFinalC` at `vs` (`takesVals_valsWk`, `takesFinalC_valsWk`). That the deployed device's rounds are `valsWk
honestVals`'s at the honest values is the device's completeness, checked empirically under `FC_FIREWALL=1`. -/
noncomputable def valsWk (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)) :
    L.toSrv.Worker :=
  (L.withDevice (valsDev vs L)).honestWk

theorem valsWk_masked (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    {x : Msg} {w : Flock.Wit}
    {ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand}
    {h : List (Msg × Msg)} (hk : L.kindAt x h.length = .masked) :
    valsWk vs L x w ρ h 0 = some (valsDev vs L x w ρ h (computedOf (L.computed x w ρ h)), false) := by
  show some (if (L.withDevice (valsDev vs L)).kindAt x h.length = .masked then
      ((L.withDevice (valsDev vs L)).device x w ρ h (computedOf ((L.withDevice (valsDev vs L)).computed x w ρ h)),
        false)
    else ((L.withDevice (valsDev vs L)).ownMsg x w ρ h, false)) = _
  rw [if_pos ((L.kindAt_withDevice (valsDev vs L) x h.length).trans hk)]
  rfl

/-- **`valsWk` is live short of the `Finish`** wherever each frame round so far passes the frame's checks at its round
(`roundOk` at the device `valsDev`): `Hello` is the server's own, at the link it sends the ZK server's own message, and
it never sends a stand-in. -/
theorem live_valsWk (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : h.length < L.last x)
    (hr : ∀ i, 2 ≤ i → i ≤ h.length → (L.withDevice (valsDev vs L)).roundOk x w ρ (h.take i) = true) :
    L.toSrv.live (valsWk vs L) x w ρ κ h = true := by
  rw [← L.live_withDevice (valsDev vs L)]
  exact (L.withDevice (valsDev vs L)).toSrv_live_honest x w ρ κ h hl hr

/-- **`valsWk` takes `vs`'s values**: at every `vs` and every release whose `Finish` comes after the scheduled rounds
(`refK x < L.last x`: at `ofLoopLink` and `withInner`, `loop_last`), `valsWk vs L` meets `TakesVals`. A live round
passes the frame's check, so it is the scheduled round's frames but for their values (`passes_frames`), the frame frees
the same words in it as in the scheduled round (`roundFree_alike_ok`), and `fill` wrote `vs`'s value into each
(`fill_wd`); at the inner proof's round 0 the frame frees none (`roundFree_inner0`). -/
theorem takesVals_valsWk (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    {x : Msg} {w : Flock.Wit} (hlast : refK x < L.last x) : TakesVals vs L (valsWk vs L) x w := by
  intro ν t κ m₀ ρ₀ ops v m out s fr j r hpos hlt hl _ hv hm hfr hrep k hk
  have hkd : L.kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    L.kindAt_masked_of (by simp only [List.length_cons]; omega) (by simp only [List.length_cons]; omega)
  have hv' := hv
  rw [valsWk_masked vs L hkd] at hv'
  obtain ⟨m', out', s', hm', hc⟩ := LeanRelease.check_of_live L _ x w (ν, t) κ _ v hl hkd hv
  rw [hm] at hm'
  obtain rfl := Except.ok.inj hm'
  obtain ⟨qr, hqr, -, -⟩ := passes_none (check_none_passes hc)
  obtain ⟨o₁, s₁, frr⟩ := qr
  have hvd : v = valsDev vs L x w (ν, t) ((m₀, ρ₀) :: ops)
      (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops))) := (congrArg Prod.fst (Option.some.inj hv')).symm
  unfold valsDev at hvd
  rw [hm] at hvd
  simp only at hvd
  rw [hqr] at hvd
  simp only at hvd
  split at hvd
  · rename_i hF
    obtain ⟨fs, kk, hr, hmid, hkk, h0, hn⟩ := hF
    have hs0 := roundFree_ok_inner hqr
    have hR : (⟨(L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops)
        (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops)))).toArray⟩ : ByteArray) =
        ByteArray.mk (encs fs ++ (Frame.sq kk (L.coins x ((m₀, ρ₀) :: ops).length)).enc).toArray := by rw [hr]
    rw [hR] at hqr
    have hpass := check_none_passes hc
    rw [hR] at hpass
    obtain ⟨fs', hA, hmid', hV⟩ := passes_frames _ _ fs kk _ hs0 hmid hkk h0 hn _ hpass
    have hfr' := hfr
    rw [hV] at hfr'
    have hc' : CfgAlike (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) :=
      ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
    obtain ⟨-, hff⟩ := roundFree_alike_ok hc' _ false fs fs' kk _ hs0 hmid hmid' hkk h0 hn hA hqr hfr'
    dsimp only at hff
    subst hff
    obtain ⟨hsp, hbd⟩ := (roundFree_spaced _ _ fs kk _ hs0 hmid hkk h0 hn hqr).get
    rcases eq_setTau hvd with hvd | hvd
    · subst hvd
      have h0' := hfr
      rw [Array.toArray_toList] at h0'
      have h00 : frr = #[] := roundFree_inner2 (vsTauAt_size vs L x w (ν, t) ((m₀, ρ₀) :: ops) hrep) h0'
      subst h00
      exact absurd hk (Nat.not_lt_zero k)
    by_cases hS : ∃ w' : ByteArray, w'.size = 16 ∧ fill (L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops)
        (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops)))) frr
        (fun k => vsAt vs L x w (ν, t) ((m₀, ρ₀) :: ops) ((L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]!)) =
        (inner0 w').data.toList
    · rw [setFinalC, if_pos hS] at hvd
      subst hvd
      have h0' := hfr
      rw [Array.toArray_toList] at h0'
      have h00 : frr = #[] := roundFree_inner0 (FlockSoundness.Discharge.ZkView.toBytes_size _) h0'
      subst h00
      exact absurd hk (Nat.not_lt_zero k)
    · rw [setFinalC, if_neg hS] at hvd
      subst hvd
      show wd _ frr[k]! = _
      rw [fill_wd _ frr _ hsp (fun i hi => le_trans (hbd i hi) (by rw [hr, List.length_append]; omega)) hk]
      exact vsAt_eq vs L x w (ν, t) ((m₀, ρ₀) :: ops) _ hrep
  · subst hvd
    have hs := roundFree_ok_inner hfr
    obtain ⟨h16, -⟩ := roundFree_sq hs hfr
    simp [ByteArray.size] at h16

/-- **`valsWk` takes `vs`'s `final_c`**: at every `vs` and every release whose `Finish` comes after the scheduled rounds,
`valsWk vs L` meets `TakesFinalC`: where its round is the inner proof's round 0, `setFinalC` wrote `vs`'s `final_c`
under its pad there. -/
theorem takesFinalC_valsWk (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    {x : Msg} {w : Flock.Wit} (hlast : refK x < L.last x) : TakesFinalC vs L (valsWk vs L) x w := by
  intro ν t κ m₀ ρ₀ ops v w' j r hpos hlt hl _ hv hw' hshape hrep
  have hkd : L.kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    L.kindAt_masked_of (by simp only [List.length_cons]; omega) (by simp only [List.length_cons]; omega)
  have hv' := hv
  rw [valsWk_masked vs L hkd] at hv'
  obtain ⟨m, out', s', hm, hc⟩ := LeanRelease.check_of_live L _ x w (ν, t) κ _ v hl hkd hv
  obtain ⟨qr, hqr, -, -⟩ := passes_none (check_none_passes hc)
  obtain ⟨o₁, s₁, frr⟩ := qr
  have hvd : v = valsDev vs L x w (ν, t) ((m₀, ρ₀) :: ops)
      (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops))) := (congrArg Prod.fst (Option.some.inj hv')).symm
  unfold valsDev at hvd
  rw [hm] at hvd
  simp only at hvd
  rw [hqr] at hvd
  simp only at hvd
  split at hvd
  · rcases eq_setTau hvd with hvd | hvd
    · exact absurd (hvd.symm.trans hshape) (inner2_ne_inner0 _ w')
    by_cases hS : ∃ w'' : ByteArray, w''.size = 16 ∧ fill (L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops)
        (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops)))) frr
        (fun k => vsAt vs L x w (ν, t) ((m₀, ρ₀) :: ops) ((L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]!)) =
        (inner0 w'').data.toList
    · rw [setFinalC, if_pos hS] at hvd
      subst hvd
      rw [finalCOf_inner0, FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self]
      exact vsAt_eq vs L x w (ν, t) ((m₀, ρ₀) :: ops) _ hrep
    · rw [setFinalC, if_neg hS] at hvd
      exact absurd ⟨w', hw', hvd.symm.trans hshape⟩ hS
  · have e := congrArg List.length (hvd.symm.trans hshape)
    rw [ByteArray.toList_data_append, List.length_append, FlockSoundness.Refine.Frame.header_length,
      List.length_nil] at e
    omega

/-- **`TakesVals` and `TakesFinalC` hold at the honest values** at the session loop: `valsWk honestVals`, live wherever
its rounds pass the frame's checks (`live_valsWk`), meets both. -/
theorem takes_honestVals_ofLoopLink (L₀ src real) {x : Msg} {w : Flock.Wit} :
    TakesVals (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
        (valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)) x w ∧
      TakesFinalC (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
        (valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)) x w := by
  have he : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last Loop.loopNr Masked.dataE _ x
  have hlast : refK x < (LeanRelease.ofLoopLink L₀ src real).last x := by omega
  exact ⟨takesVals_valsWk _ _ hlast, takesFinalC_valsWk _ _ hlast⟩

end Loop

end Worker

end Claims.ZKServer
