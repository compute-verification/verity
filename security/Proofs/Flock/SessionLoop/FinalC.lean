module

public import Proofs.Flock.SessionLoop.TakenVals
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

/-!
# `vs`'s `final_c` is fixed by its round-1c values

The ZK server's finish sends the inner proof's round 0 only where the request's `final_c` is the one its passed values
fix: the rep's `zerocheck c_eval` constraint, `Σ_i λ_i(z)·r1c_i + final_c` with each value under its own pad, holds on
the pads (`Flock.Firewall.Finish.consWhy`). This file states that identity on the switched session's values, a premise
over `vs` beside `Loop.TakesVals` and `Loop.RowsOK`:

* **`Loop.FinalCFixed`**: at each table and rep, at every nonce, opened coins and tape at which none of the table's
  zerocheck outer coins is `1` (`RowsOK`'s restriction), `vs`'s `final_c` (`vsWord` at `fc`) is the c_eval of its 64
  round-1c values at the opened coins' zerocheck point (`zcE`).
* **`Loop.finalCFixed_honestVals`**: it holds at the honest values, at every coins: there they are the honest clear
  zerocheck's (`Composed.honestAt_fc`, `honestAt_r1c`), whose two sides of `c_eval` agree (`Zerocheck.c_eval`). With
  `Loop.takes_honestVals_ofLoopLink` and `Loop.rowsOK_honestVals`, the honest values meet every premise over `vs` at the
  session loop.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

/-- **`vs`'s `final_c` is the c_eval of its round-1c values**: at every nonce, opened coins and tape at which none of
table `j`'s zerocheck outer coins is `1`, each rep's value at `final_c`'s pad index is `Σ_i λ_i(z)·r1c_i`, the Lagrange
weights at the rep's zerocheck point `z` (the opened coins', `zcE`) against its values at the 64 round-1c indices, as the
ZK server's finish computes it (`Flock.dot`, `Flock.lagrange`). -/
def FinalCFixed (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)),
    (∀ (r : Fin 2), ∀ y ∈ (tabC PE x j).rOuter (czExe PE x c j) r, y ≠ 1) → ∀ (r : Fin 2),
    vsWord vs x w ν c τ j r (layE x j).fc =
      _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
        ((Array.range 64).map fun i => vsWord vs x w ν c τ j r ((layE x j).r1c i))

/-- **`vsWord` at the honest values is the clear value the cut's view masks** (`Composed.honestAt`), below `n_E`. -/
theorem vsWord_honestVals (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hk : k < (tabC PE x j).z.nE) :
    vsWord (CoinLayer.session Loop.loopNr Masked.dataE).honestVals x w ν c τ j r k =
      FlockLevel3.GF128.val (FlockSoundness.Discharge.Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r
        (FlockSoundness.Discharge.Composed.zWith (stAt x j) (wtOf PE x w j)
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1)
        ((((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 r) k) := by
  unfold vsWord
  rw [dite_eq_left hk]
  rfl

/-- **`FinalCFixed` holds at the honest values**, at every coins, the outer coins' restriction unused: the honest
values at `final_c`'s and the round-1c indices are the honest clear zerocheck's (`honestAt_fc`, `honestAt_r1c`), and
its `final_c` is its round-1c values' c_eval at its point (`Zerocheck.c_eval`), the opened coins' (`zcE`). -/
theorem finalCFixed_honestVals (x : Msg) (w : Flock.Wit) :
    FinalCFixed (CoinLayer.session Loop.loopNr Masked.dataE).honestVals x w := by
  intro ν c τ j _ r
  have hE : (tabC PE x j).z.nE = (layE x j).hab := by
    show (layE x j).len - 4 = (layE x j).hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hr1c : ∀ i < 64, (layE x j).r1c i < (tabC PE x j).z.nE := fun i _ => by
    rw [hE]
    exact lt_hab (layE x j) (by
      simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.r1c, _root_.Flock.Zk.PadLayout.fc,
        _root_.Flock.Zk.PadLayout.fa]
      omega)
  rw [vsWord_honestVals x w ν c τ j r (k := (layE x j).fc) (by rw [hE]; exact lt_hab (layE x j) le_rfl)]
  refine (congrArg FlockLevel3.GF128.val (FlockSoundness.Discharge.Composed.honestAt_fc (zkTab x j).table
    (stAt x j) (czExe PE x c j) r _ _)).trans ?_
  show (FlockSoundness.Discharge.Composed.zcH (zkTab x j).table (stAt x j) (czExe PE x c j) r _).finalC = _
  unfold FlockSoundness.Discharge.Composed.zcH
  rw [← FlockSoundness.Discharge.Composed.Zerocheck.c_eval]
  refine congrArg₂ _root_.Flock.dot rfl (Array.ext ?_ fun i h1 h2 => ?_)
  · simp [FlockSoundness.Discharge.Composed.Zerocheck.honest]
  · have hi : i < 64 := by simpa using h2
    simp only [Array.getElem_map, Array.getElem_range]
    rw [vsWord_honestVals x w ν c τ j r (hr1c i hi), ← getElem!_pos _ i h1]
    exact (congrArg FlockLevel3.GF128.val (FlockSoundness.Discharge.Composed.honestAt_r1c (zkTab x j).table
      (stAt x j) (czExe PE x c j) r _ _ i hi)).symm

end Loop

end Claims.ZKServer
