module

public import Proofs.Flock.Claims.Recursion.HonestHalf
public import Proofs.Flock.Claims.Recursion.IrDep
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
# The honest deployed recursive run (`HonestFixture`)

The one fact the honest half of `RecursiveKSAt`'s non-vacuity (`honest_half_fixture`) takes, and nothing in Lean proves:
one honest deployed recursive run, at one statement `x` and with the deployed inner prover `IrDep x`, on which every
premise of `honest_half_tops` holds at once. Its conjuncts are the named Props that theorem reads, with their statements
unchanged where they are stated (`HonestRunAccepted` and `NoCollAt` in `HonestHalf`, `HonestWitSatisfies` in
`HonestWitR`), and the tape exception, named here (`TopsAt`). That the inner prover's openings fit V\*'s levels
(`IrShaped`) is proved at `IrDep` (`irShaped_dep`).
-/

@[expose] public section

namespace RecDeploy

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest
open RecDeploy.HonestHalf

/-- **The two reps' tops differ at every level above the cap** (the tape exception, `honest_half_tops`'s `hTops`): where
the stager's inputs at the inner prover `Ir`'s run (`srcOf`, at the inner witness `w`, tape `ρ` and inner rounds `hist`)
are read, the two reps' committed tops at each level `l > 0` are not one top (`VStar.In.sameTop`). A fact of the run's
tape, not of the honest prover: `HonestFixture` takes it of its run. -/
def TopsAt (Ir : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : (Ir.fit x).Tape) (hist : List (Msg × Msg)) : Prop :=
  ∀ src, srcOf (Ir.fit x) x w ρ hist = some src → ∀ l, 0 < l → ∀ a b,
    (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false

/-- **One honest deployed recursive run has every premise of the honest half** (witnessed outside Lean, all on one run
at one statement `x`): at the deployed brings at `x` (`bLAt`), the inner verifier `IAt decI` and the compiled record
`DAt`, the reference prover's run on tape `ω` and coins `r` is accepted (`HonestRunAccepted`), V\*'s gate-level
completeness holds at it (`HonestWitSatisfies`), no part's recorded calls collide (`NoCollAt`), and at the run's inner
rounds before the last (`hist.take (kI x - 1)`) the reps' tops differ (`TopsAt`). The inner prover is the deployed one,
`IrDep x`: the rows `rec_live` commits and the openings, salts, κ, eq(α) and v `rec_vstage`, `rec_vstar` and
`verifier_rows` build, at Flock's proofs.

How the witness is reproduced. The fixture is one session of the recursive audit that `benchmarks/recursion_e2e`
generates in its one-pass mode on the CPU prover, under the ZK server (`--e2e --cpu`, with `KEEP_DATA=1`, which keeps
the session's data): the toy deployment at K = 64 and N = 4, its challenge tree sampled by the node's operating system
(`--tree-test-os`), a test stand-in for a challenger's device. The data holds the inner session, V\*'s staged
statements (each part's `inst.bin`, with `fold.json`, `points.json` and `acc-salts.bin`) and each part's proved
session. The session the tests below read is the one
`verity/protocols/proofs/recursion/verifier/vstar_cli/tests/recursive_data.json` names, or any other the generator
kept. Each conjunct on it:
* `HonestRunAccepted`: the whole-session tests of
  `verity/protocols/proofs/recursion/verifier/vstar_cli/tests/test_vstar_recursive.py`, which run only under
  `VERITY_VSTAR_RECURSIVE_DATA` (the check's opt-in tier). V\*'s compiled check over the whole session
  (`vstar-recursive verdicts`) and Flock's compiled verifier at each part's flags accept every V\* part, each part's
  verdict the same in both; and over the inner session with one challenge flipped, some part refuses, the same parts
  in both.
* `TopsAt`, `HonestWitSatisfies` and `NoCollAt`: taken of the same session, and no test in the tree checks them on it.
  A check of `HonestWitSatisfies` holds each part's trace (`VStar.Trace.assign`) to the table the compiled verifier
  sets up at the sample the part's session records; one of `NoCollAt` finds no two inputs with one SHA-512 output
  among the calls each part records (`findColl`); one of `TopsAt` reads the two reps' committed tops at each level
  above the first.

What the tests witness is weaker than the Prop in two ways: the verdicts are the compiled programs' on the session's
files, with no Lean term `x` formed, so they are not `DAt`'s verifier evaluated at one; and the session's prover is
Python's builder with Flock's CPU prover, not `refP`.

What ties the two provers is `verity/protocols/proofs/recursion/verifier/vstar_cli/tests/test_vstar_trace_difftest.py`.
`refP`'s V\* parts are `bringsAt`'s `refPartsFin (A0 mask)`: each part's `VStar.Trace.assign` at its fitted port inputs
(`VStar.In.partIn`, whose level, algebra and cap builders `test_vstar_in_difftest` holds to Python's stager). The trace
difftest holds `VStar.Trace.assign` (`vstar-trace`) to Flock's CPU prover's witness (`flock-trace`: `Stmt::row_inputs`,
`unit_inputs`, `converge`, `witness`), block for block and range for range, on parts Python's `rec_vstage.stage`
staged: on the small sessions it stages itself (`test_the_trace_is_the_deployed_provers`) and, under
`VERITY_VSTAR_IN_DEPLOYED`, on every staged statement of a recorded session
(`test_the_recorded_plain_leaf_sessions_traces`).

Neither difftest compares `refP`'s inner rounds with Python's builder;
`verity/protocols/proofs/recursion/verifier/vstar_cli/tests/test_irdep_difftest.py` does, on a session's own tree
(`VERITY_IRDEP_TREE`, with `IrDepSession.lean`'s output at it as `VERITY_IRDEP_LEAN`): `IrDep x`'s rows at every inner
round, openings, running-sum salts, κ, eq(α) and v, evaluated by `IrDepSession.lean` (the setup-level copies
`IrDep.lean` proves equal to `IrDep`'s own), are what `rec_live`, `rec_vstar` and `rec_vstage` built there, and a
planted byte in any of them is refused. -/
def HonestFixture : Prop :=
  ∃ (mask : ℕ → ℕ → Bool) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (w : Claims.Flock.Wit) (st : Claims.Recursion.CompiledStatement)
    (ω : Tape (bLAt mask (IrDep x) b0 x aux nOf hd recRest))
    (r : Fin ((DAt mask (IrDep x) b0 x aux nOf hd recRest).V.k x) → Coin),
    HonestRunAccepted mask (IrDep x) b0 x aux nOf hd recRest decI w st ω r ∧
    HonestWitSatisfies FlockDeploy.sha512 (IAt decI).V (bLAt mask (IrDep x) b0 x aux nOf hd recRest) x w ω r ∧
    NoCollAt mask (IrDep x) b0 x aux nOf hd recRest decI w ω r ∧
    TopsAt (IrDep x) x w ω.1
      ((((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask (IrDep x) b0 x aux nOf hd recRest) w)
        (DAt mask (IrDep x) b0 x aux nOf hd recRest).V x ω r).msgs.take ((IAt decI).V.k (xIOf x))).map histOf).take
          (RestRead.kI x - 1))

end RecDeploy
