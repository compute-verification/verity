import FlockSoundness.Discharge.Composed.Defs
import FlockSoundness.Discharge.Composed.Base
import FlockSoundness.Discharge.Composed.Inner
import FlockSoundness.Protocol.Discharge.Composed.Coined

/-!
# One `--zk` sampled session: the draw's tables as `Coined` tables

* `CoinedSetup`, `coinedOf`, `CoinedOK`, `CoinedOfSetup`: the `--zk` table of a draw as `ZkView`'s `Coined` tables at the
  executable's leaves, at each coin vector, and the structural facts Lemma B reads of them.
* `ZkViewAt`: the view statement the composed headline proves at each draw.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-! ## The session's tables at the executable's leaves -/

/-- **The structural facts Lemma B reads**, of a `CoinedSetup` at the arithmetic `A` and M1's padding constant `κ`: M1's
level-0 padding geometry (`cols ≤ 40`, `t_pad ≤ 2^cols`, at most `t_pad` opened positions), and at every coin vector the
honest prover doesn't refuse (`Admits`): the opened positions distinct and inside `2^(cols+1)`, the table reads its coins (`β`, `ρ`, the rank check's
weights `eq(p_k, x)` and `embed` the inclusion of F128 in F256), the rank check, M1's padding columns `Gp`, each rep's
`W^r` bijective, the pads onto, `β ≠ 0`, and the inner relation at every valid witness whose registered values are the
public ones. -/
def CoinedOK (A : Arith 𝔽 FlockLevel3.GF256) (κ : 𝔽) (C : CoinedSetup) : Prop :=
  C.cols ≤ 40 ∧ C.z.tpad ≤ 2 ^ C.cols ∧ C.z.nS ≤ C.z.tpad ∧
    ∀ c : CoinsZK C.Rest, C.Admits c →
      (Function.Injective (C.pos c) ∧ ∀ q, C.pos c q < 2 ^ (C.cols + 1)) ∧
      ((∀ r, (C.T c).β r = fOf (c.beta r)) ∧
        (∀ r, (C.T c).ρ r 0 = FlockLevel3.GF256.of (c.rho r).1 ∧ (C.T c).ρ r 1 = FlockLevel3.GF256.of (c.rho r).2) ∧
        (∀ k x, (C.T c).eqw k x = fOf (eqWord (C.points c k) (C.maskWords x))) ∧
        ∀ a : F128, (C.T c).embed (fOf a) = FlockLevel3.GF256.of ⟨a, 0⟩) ∧
      (C.T c).tab.Rank ∧
      (∀ q i, (C.T c).tab.Gp q i = κ * A.Xhat i.val (C.cols + 1) (A.omega (C.pos c q))
        + A.Xhat (2 ^ C.cols + i.val) (C.cols + 1) (A.omega (C.pos c q))) ∧
      (∀ r, Function.Bijective fun p : 𝔽 × 𝔽 => (C.T c).tab.Wr r p.1 p.2) ∧ (C.T c).tab.PadsOnto ∧
      (∀ r, (C.T c).tab.β r ≠ 0) ∧
      ∀ w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg → (C.T c).tab.InnerHolds w

/-- **The session's tables satisfy Lemma B's facts** (proved: `coinedOf_setup`): at every draw `verify --zk` sets up (`y`), the
`Coined` tables of the draw's `--zk` table at its executable statement (`coinedOf P (tablesAtZK dj y₀ S R) y.x.st`) are over
the table's statement, check rank over the `RANK_POINTS` claim points at the statement's mask words (`maskWordsOf`), have
other coins, and are `CoinedOK` at `execArith` and `PAD_C`. -/
def CoinedOfSetup (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
    (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).S = (tablesAtZK (Reg := Reg) dj y₀ S R).S ∧
    (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).z.nC = Zk.RANK_POINTS ∧
    (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).z.nMask = (maskWordsOf y.x.st).length ∧
    (∀ x, (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).maskWords x = (maskWordsOf y.x.st).getD x 0) ∧
    Nonempty (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Rest ∧
    CoinedOK execArith (fOf Zk.PAD_C) (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st)


end FlockSoundness.Discharge.Composed
