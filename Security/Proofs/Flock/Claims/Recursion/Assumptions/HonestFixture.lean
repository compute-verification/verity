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

The run is r20261026-223243-c2a7 (`benchmarks/recursion_e2e --e2e --cpu`, the toy deployment, K = 64, N = 16, under the
firewall, at main 4331ec922a), outer tree art:d872affc203376ec503fbbc949b6f2ab3d191b866e6a8cbd7c13656a836734c8: the
compiled `flock-verify` printed `VERDICT accepted` on each of its 10 V\* parts, and the inner session was accepted.
CAPLOWER's one run over that tree checks the other four conjuncts (all hashed inputs distinct for `NoCollAt`); its id is
pending. What the run witnesses is weaker than the Prop in three ways: only Python joins the 10 per-part verdicts (no
single compiled verdict exists); its `flock-verify` was built at 4331ec922a, before the verifier's `checkRoundWords`,
`VStar/In` and `VStar/Trace`, which `DAt`'s verifier includes; and its prover is Python's builder with C-Flock's CPU
prover, which agrees with `refP` only through the difftest. -/
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
