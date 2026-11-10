module

public import Proofs.Accounting.TS.Session.Checked.Layer
public import Proofs.Accounting.TS.Session.Assumptions
public import Proofs.Flock.SoundAtBridge.Served
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
# A checked unit's proof layer is sound through its Flock sessions (the statement)

`Accounting.Guarantees.TS.CheckedSessionSound`: the realization `Accounting.Served`'s docstring asks of a proof layer
for a value-checked `zTree` window, at `Accounting.Guarantees.TS.SessionSoundAt`'s premises, for the checked layer
`Accounting.TS.Checked.checkedLayer` (its unit `PouwH1TileZy_v1`, its check word `ok`). Of `SessionSoundAt`'s four
premises on the layer, it proves three and the window's sampling design stays a premise:

* `AcceptedLive`: the layer accepts exactly where flock-verify `--zk --registered` accepts;
* `ServedChecksHold`: the relation, the unit's Definition with `ok` registered, holds at the session's value layer
  wherever its sampled proof blocks are right (`RowsSession.RowsAt` is the circuit's evaluation there);
* `ServedRootsLinked` at `δs = 0`: under full checking (`kd = n`) every outcome of the session's game samples every unit,
  and `Flock.SecurityProofs.ServedRowsLinked` puts the registered rows at the value layer on each sampled proof block, where
  every unit's served roots are registered (`RowsSession.ServedRegistered`);

and so `pf.SoundAt δ`. It sits in the proofs package because the sessions do (`reads_exempt`).

It proves the checked layer only. Its relation, `CheckedRel`, says the opened rows are one wire assignment the circuit's
evaluation agrees with, and nothing of what the circuit computes: don't cite it for the product or `dq`. The tile's
Definition with `ok = 1` (each cell `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_b]`) is
`Accounting.Guarantees.TS.TileSessionSound`'s, under its premise `Accounting.TS.Tile.TileComputes`; the declared
circuit's checks are `Accounting.Guarantees.TS.HeldSessionSound`'s.
-/

@[expose] public section

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A checked unit's proof layer is sound at `δ` through its Flock sessions, fully checked.** For any window whose
proof `i` of round `w` on the path `x` is the Flock session `sess x w i` (a `RowsSession`) with outcome
`out x w i`, and the checked layer `checkedLayer sess out ok right` (accepted where flock-verify `--zk --registered`
accepts, opening the committed rows at the registered reads, its relation the unit's Definition with the port `ok`
registered): given `Flock.Guarantees.EndToEndRegisteredRows`' named assumptions at every session and every unit's
opening (`RowsSession.CR`), the window's sampling design (`Accounting.Assumptions.TS.WindowDesign`), and at every session full
checking (`kd = n`), every unit's served roots registered as ports of `own`
(`RowsSession.ServedRegistered`) and a port named `ok` in the circuit and in `own`, and `δ ≥ 0` with
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta$$
at every session a path runs, the layer meets `SessionSoundAt`'s premises `AcceptedLive`, `ServedChecksHold` and
`ServedRootsLinked` at `δs = 0` with `rows := rowsReg`, and each proof is accepted with committed rows that fail its
relation on the path with probability at most `δ`. -/
def CheckedSessionSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) (δ : ℝ),
    (∀ x w i, (sess x w i).CR) →
    Accounting.Assumptions.TS.WindowDesign sess out →
    (∀ x w i, (sess x w i).kd = (sess x w i).n) →
    (∀ x w i u, (sess x w i).ServedRegistered u) →
    (∀ x w i, (∃ q : Fin (sess x w i).c.ports.size, (sess x w i).c.ports[q].name = ok) ∧
      ∃ pt ∈ (sess x w i).own, pt.name = ok) →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    Accounting.Assumptions.TS.AcceptedLive (checkedLayer sess out ok right) sess out ∧
      Accounting.Assumptions.TS.ServedChecksHold (checkedLayer sess out ok right) sess out rowsReg ∧
      Accounting.Assumptions.TS.ServedRootsLinked (checkedLayer sess out ok right) sess out rowsReg 0 ∧
      (checkedLayer sess out ok right).SoundAt δ

end Accounting.Guarantees.TS
