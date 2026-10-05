import FlockSoundness.Protocol.Discharge.ZkExec.Live
import FlockSoundness.Protocol.Discharge.Exec.Setup
import FlockSoundness.Protocol.Refine.Frames

/-!
# The target statements of flock-verify `--zk`'s acceptance and its refinement

Each statement is a `Prop` here; its theorem (`theorem … : G`) is in the file that proves it.

**Acceptance** (what `AcceptsZK` gives, peeled check by check, at any tag and any table count `J`):
* `ZkVerifyOk` (`Verify.lean`): `Zk.verify` accepts only on the keyed coin tree, a record that decodes, one statement
  per table, one proof file per rep stream, and at every table both reps' checks accepting with one cap.
* `ZkRepOk` (`Verify.lean`): a rep's check accepts only on its stream and proof file, the pinned PCS parameters, a
  64-byte pads root, the transcript program `repProg` running on the stream's rounds and consuming every round, and
  the proof's cap equal to the table's `root_B`.
* `ZkInnerOk` (`Inner.lean`): the inner proof accepts only with its pinned sizes, `β ≠ 0`, `τ` opening its
  `hm96-sha512/v1` commitment, the batched constraint `⟨c, Y⟩ = t + β·τ`, and at each of the `Q` queried columns `Y`'s
  code equal to `a + β·b` and the column's `hm96` leaf climbing to the pads' root.

**Acceptance and the statement** (under tags without hidden outputs, `hh`, as `Exec`'s; with them `setupTables` calls
`setupHidden`, [wave2-hidden-exec]'s):
* `AcceptsZKSetupH`: every table is `setupH`'s statement under the `--zk` tags (`tagsAt (zkTags I) J j`) and the coin
  tree, at some draw (the record's, for one table), and passes `Zk.shapeOk`.
* `AcceptsZKOne`: a one-table acceptance is one `setupH` statement's, with `Zk.verify` on `#[Setup.ofCircuit st]`.
* `AcceptsZKRetained`: under tags that retain rounds, every stream of the accepted record retained every round's bytes.

**Refinement** (one table, against the compiled `--zk` table `G, out, acc` named in `Live.lean`):
* `ExecRefinesZKOne`: R11 at `--zk`: against any live prover, flock-verify `--zk`'s acceptance is no likelier than the
  compiled table's acceptance against the simulated prover `simAuditZK … τ`, with `bad = ⊥`, given `DecodesZK`.
* `ZkHeadlineExecOne`: the `--zk` headline at flock-verify's acceptance, carried from any bound `B` on the compiled
  audit's `Pr[accept ∧ Q]` at the simulated prover, for any event `Q` of that prover (at least `K` wrong units of its
  committed values): `Pr[flock-verify --zk accepts ∧ Q] ≤ B`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Lean (Json)
open Game Audit Exec Refine Flock
open scoped ENNReal

/-! ## Acceptance -/

/-- The inner code's length: the pads message (`len + K_PAD`) rounded up to a power of two, at rate `1/8`. -/
def innerN (lay : Flock.Zk.PadLayout) : ℕ :=
  (lay.len + Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ Flock.Zk.INNER_LOG_INV_RATE

/-! ## Acceptance and the statement -/

/-! ## Refinement, one table -/

section Refine

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {P Cp Op Out : Finset (Fin n) → Reg → Type}
  (G : ∀ S R, (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (P S R))
  (out : ∀ S R, Cp S R → P S R × P S R → Op S R → Out S R) (acc : ∀ S R, Out S R → Bool)
  [∀ S R, Nonempty (Op S R)] (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ S R, Array ByteArray → Op S R) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ S R, ByteArray → Cp S R) (N : ℕ) (L : Law n)

end Refine

end FlockSoundness.Discharge.ZkExec
