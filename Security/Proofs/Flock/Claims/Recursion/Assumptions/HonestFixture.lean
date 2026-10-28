import Proofs.Flock.Claims.Recursion.HonestHalf

/-!
# The honest deployed recursive run (`HonestFixture`)

The one fact the honest half of `RecursiveKSAt`'s non-vacuity (`honest_half_fixture`) takes, and nothing in Lean proves:
one honest deployed recursive run, at one statement `x`, on which every premise of `honest_half_tops` holds at once. Its
conjuncts are the named Props that theorem reads, with their statements unchanged where they are stated
(`HonestRunAccepted` and `NoCollAt` in `HonestHalf`, `HonestWitSatisfies` in `HonestWitR`, `IrShaped` in `VStarDense`),
and the tape exception, named here (`TopsAt`).
-/

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
tape, not of the honest prover: witnessed on `HonestFixture`'s run. -/
def TopsAt (Ir : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : (Ir.fit x).Tape) (hist : List (Msg × Msg)) : Prop :=
  ∀ src, srcOf (Ir.fit x) x w ρ hist = some src → ∀ l, 0 < l → ∀ a b,
    (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false

/-- **One honest deployed recursive run has every premise of the honest half** (witnessed outside Lean, all on one run
at one statement `x`): at the deployed brings at `x` (`bLAt`), the inner verifier `IAt decI` and the compiled record
`DAt`, the reference prover's run on tape `ω` and coins `r` is accepted (`HonestRunAccepted`), V\*'s gate-level
completeness holds at it (`HonestWitSatisfies`), no part's recorded calls collide (`NoCollAt`), and at the run's inner
rounds before the last (`hist.take (kI x - 1)`) the inner prover's openings fit V\*'s levels (`IrShaped`) and the reps'
tops differ (`TopsAt`).

The run is r20261027-014625-06c6 (`benchmarks/recursion_e2e --e2e --cpu`, the toy deployment, K = 64, N = 16, under the
firewall, at f1asm 16a44e3b86). Its outer tree, with each part's `inst.bin`, `fold.json`, `points.json` and
`acc-salts.bin`, is art:886efa1d995543a70ba46e6ef713a2afe89a7022b7fc4988c9cc2dbd629a43a8. Each conjunct on it:
* `HonestRunAccepted`: the compiled `flock-verify` printed `VERDICT {"accepted":true}` (why `""`) on each of the 10 V\*
  parts (L0–L3, alg-p0, alg-p1, cap, cap-L1–L3) and on the inner session, and the parts that read a flipped challenge
  or query coin refused it.
* `IrShaped`: passed in r20261027-025051-8aa6 (`vstar-difftest ir_shaped` on the stager's inputs from the run's tree:
  both reps open, 448 openings each; one opening fewer, or one past the tops, is refused).
* `TopsAt`: holds in 8aa6 (the reps' tops differ at levels 1–3).
* `HonestWitSatisfies`: passed in 8aa6 (`test_the_deployed_runs_parts_satisfy`, `vstar-trace --sat --session`): all 10
  parts, each at the table `verify --zk` sets up at the draw its session records, under the zero mask and a random
  one, 20 of 20 lines with no bad rows, pin or region bits.
* `NoCollAt`: passed in r20261027-033500-eeb1 (`test_the_deployed_runs_calls_have_no_collision`, `vstar-trace
  --nocoll`): per part and over all 10 together, no SHA-512 output has two inputs (`findColl = none` on full input
  bytes): 44,952 calls, 32,323 distinct outputs for 32,323 distinct inputs. A planted pair (a recorded output under an
  input one byte longer) is caught. The calls are a superset of `NoCollAt`'s: every hm96 salt and row call the part's
  trace computes at every port and instance slot (padding included, each row digest held to the trace's own last
  compression), not only `layR`'s recorded ones; and for `vcallsAt`, every input the verifier's setup hashes at the
  part's statement and draw, logged by the logging oracle's setup (`setupLogged`), which is proved equal to the fast
  one.

16a44e3b86's verifier has `checkRoundWords`, `VStar/In` and `VStar/Trace`, and beside `DAt`'s it only refuses more
(`checkEndReads`, and `--ks-form`'s port roots, which the run doesn't pass). What the run witnesses is weaker than the
Prop in two ways: only Python joins the 10 per-part verdicts (no single compiled verdict exists), and its prover is
Python's builder with C-Flock's CPU prover, not `refP`.

What ties the two provers is `verity/core/service/vstar/tests/test_vstar_trace_difftest.py`. `refP`'s V\* parts are
`bringsAt`'s `refPartsFin (A0 mask)`: each part's `VStar.Trace.assign` at its fitted port inputs (`VStar.In.partIn`,
which `test_vstar_in_difftest` holds to Python's stager). The trace difftest holds `VStar.Trace.assign` (`vstar-trace`) to
C-Flock's CPU prover's witness (`flock-trace`: `Stmt::row_inputs`, `unit_inputs`, `converge`, `witness`), block for
block and range for range, on parts Python's `rec_vstage.stage` staged:
* r20261026-232210-2d0b (`cursor/proofs-vstartrace-e146` 547915a62c, `test_the_recorded_plain_leaf_sessions_traces`)
  passed on all 10 staged statements of `rec-coins`' recorded (16,16,8) session at 64 instances (L0–L3, alg/p0, alg/p1,
  cap, cap/L1–L3);
* r20261027-014612-19c5 (e146 93dac12374, `vstar-trace --sat`) checks the trace against the table `verify --zk` sets up:
  its drawn level and cap passed (`test_the_trace_satisfies_the_verifiers_tables`), and so did L0 of the recorded
  session (`test_the_recorded_plain_leaf_sessions_satisfy`).

8aa6's and eeb1's checks run on 06c6's own tree. Neither difftest compares `refP`'s inner rounds (the inner prover `Ir`'s
messages) with Python's builder. -/
def HonestFixture : Prop :=
  ∃ (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (w : Claims.Flock.Wit) (st : Claims.Recursion.Stmt) (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
    (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin),
    HonestRunAccepted mask Ir b0 x aux nOf hd recRest decI w st ω r ∧
    HonestWitSatisfies FlockDeploy.sha512 (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) x w ω r ∧
    NoCollAt mask Ir b0 x aux nOf hd recRest decI w ω r ∧
    IrShaped Ir x w ω.1
      ((((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
        (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.take ((IAt decI).V.k (xIOf x))).map histOf).take
          (RestRead.kI x - 1)) ∧
    TopsAt Ir x w ω.1
      ((((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
        (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.take ((IAt decI).V.k (xIOf x))).map histOf).take
          (RestRead.kI x - 1))

end RecDeploy
