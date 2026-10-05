import FlockSoundness.Model.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.Card
import FlockSoundness.Protocol.Model.Statement

/-!
# Statements, witnesses, claims and the committed values

Every statement proves one block R1CS over GF(2) (PROTOCOL.md §16.1): `A·z ∘ B·z = z` with
`A = I ⊗ A₀`, `B = I ⊗ B₀`, a pinned constant column equal to 1 in every block, and public
regions whose bits the verifier knows (§16.2). This block R1CS is the circuit's lowered layout;
that it means the named circuit is the lowering theorem (level 3), outside this lane.

**The committed values.** The prover commits a level-0 table `C₀` (§13.1: `2^k₀` lanes, one
Reed–Solomon codeword position per column). In Flock's list-decoding regime a table is close to
up to `L₀` codewords, and no out-of-domain sample follows the commitment (§9.1, paper Remark 11),
so the commitment binds to that short list, not to one witness. `CommittedSatisfies` says that
some message whose encoding is within the level-0 proximity radius of `C₀` unpacks to a witness
satisfying the statement. This is the claim soundness certifies.
-/

namespace FlockSoundness.Model

namespace Statement

variable (S : Statement)

end Statement

/-! ## Claims -/

variable {F : Type} [Field F]

/-- The value a claim asserts: `Σ_s skip[s]·Σ_{x'} eq(x, x')·z[s + 64·x']`, the evaluation of
the witness's extension that is multilinear in address bits 6… and has the given weights on the
64 skip positions (the quirky extension for `ab` and `c`, the multilinear one for regions). -/
def Claim.eval {m : ℕ} (c : Claim F m) (z : Witness m) : F :=
  ∑ a : Fin (2 ^ m), c.skip ⟨a.val % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt c.x (a.val / 64)
    * ((z a).val : F)

/-- A claim holds for a witness. -/
def Claim.Holds {m : ℕ} (c : Claim F m) (z : Witness m) : Prop := c.eval z = c.value

/-! ## The committed values -/

variable {K : Type} [Field K]

end FlockSoundness.Model
