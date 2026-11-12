module

public import Proofs.Flock.Claims.ZK.Split
public import Proofs.Flock.Claims.KS.ZkTables
public import Proofs.Flock.Claims.KS.TableErrJ
public import Definitions.Claims.SamplingTable
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
# Flock's masked knowledge soundness and zero knowledge at the deployed record (trusted)

`MaskedKS` and `MaskedZK` are `def G : Prop`s in the form of `Flock.Guarantees.EndToEnd`: two claims of
`Definitions.Claims`, `Claims.Flock.KS` and `Claims.Flock.ZK`, at the deployed masked record of the deployed relation
(`Flock.Readable.relation`), with SHA-512 as the oracle (`FlockDeploy.sha512`). Their proofs are
`Flock.SecurityProofs.MaskedKS : Flock.Guarantees.MaskedKS` and
`Flock.SecurityProofs.MaskedZK : Flock.Guarantees.MaskedZK`. `MaskedZK` holds `Claims.Flock.ZK`'s text in the lock in
place of `Flock.Guarantees.ClaimsStated`.

`MaskedKSTable` is a third, beside `MaskedKS`: `Claims.Flock.KSTable`, knowledge soundness at one table of a session,
at the same record and oracle, at each table's own word length and error. Its proof is
`Flock.SecurityProofs.MaskedKSTable : Flock.Guarantees.MaskedKSTable`.

Flock's deployed records (`FlockDeploy.deployZk`, `Claims.ZKMal.zkDeployed` at the deployed plan) are still in its
proofs package, so this module imports them from `Proofs.Flock.Claims`, and `Definitions.lean`, which the `security`
package builds without `security_proofs`, does not import it: `Proofs.Flock.Masked` does.
-/

@[expose] public section

namespace Flock.Guarantees

/-- **Masked knowledge soundness (G1).** `Claims.Flock.KS` at SHA-512 at the untrimmed masked record of the deployed
relation (`FlockDeploy.deployZk Flock.Readable.relation`): at the record's extractor and every decoder that meets the
list-decoding spec, for every statement and every prover, the probability that the verifier accepts and the extractor
recovers no witness under the relation is at most `κ(x) + √(γ(x)·Pr[B outputs a collision])`, at the record's `κ` and
`γ`, with `B` the claim's collision finder built from the prover. It takes no hypothesis. -/
def MaskedKS : Prop :=
  Claims.Flock.KS FlockDeploy.sha512 (FlockDeploy.deployZk _root_.Flock.Readable.relation)

/-- **Masked knowledge soundness at one table of a session.** `Claims.Flock.KSTable` at SHA-512 at the untrimmed masked
record of the deployed relation, at each table's own word length (`FlockDeploy.KS.wordLenJ true`) and error
(`FlockDeploy.KS.tableErrJ true`): at every decoder that meets the list-decoding spec and every index `j`, for every
statement and every prover, the probability that the verifier accepts the session and the extractor at table `j`
(`Claims.Flock.ETable`) recovers no witness of the statement the record's `table` gives at `j` (`Claims.Flock.RTable`)
is at most `κ_j(x) + √(γ(x)·Pr[B outputs a collision])`, with `κ_j` the knowledge error at table `j`
(`Claims.Flock.κTable`), which reads table `j`'s error and its `k_j = ⌈4·N_j/e⌉` accepted reruns a wait, `γ` the
record's, and `B` the claim's collision finder built from the prover. It takes no hypothesis.

What it does not say:

* It is knowledge soundness only: it says nothing of zero knowledge or of the extractor's expected work.
* At a statement of one table it is `MaskedKS`'s bound and no better: the two probabilities are equal there
  (`Claims.Flock.prBad_zero_eq`, `Claims.Flock.ksTable_one_table`, `Claims.Flock.ksTable_zero_of_ks`; at this record,
  `FlockDeploy.KS.ks_one_table_zk`).
* No compiled verifier is shown to use it: the sampled compiler is knowledge sound at the table's extractor only
  where a compiled verifier meets its contract at tables (`Claims.SamplingGeneric.Compiled.atTables_ks_complete`, its
  `hS`), and none is shown to. The Lean verifier refuses a session above `Tags.MAX_TABLES` tables.
* It extracts one table's witness. No witness of a whole session is assembled from the tables' extractions.
* Past the statement's tables it says nothing: there the word is empty and `κ_j ≥ 1`
  (`FlockDeploy.KS.one_le_κTable_past`).
* It is a statement about the record. Nothing served reads it. -/
def MaskedKSTable : Prop :=
  Claims.Flock.KSTable FlockDeploy.sha512 (FlockDeploy.deployZk _root_.Flock.Readable.relation)
    (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true)

open Claims.ZKMal.Masked FlockSoundness.Discharge in
open Classical in
/-- **Masked zero knowledge (G2).** `Claims.Flock.ZK` at SHA-512 at the deployed masked record
(`Claims.ZKMal.zkDeployed`: the deployed coin layer `coin-tree/hm96-sha512/v2`, the deployed relation, and the deployed
plan's tables: the prover of levels ≥ 1 `PE`, the round plan of `czExe`, `encExe` stopped where a refusal is seen
(`encStop`) and `finExe`, and the executable's tape plan `UE`), at every rep rounds' coin counts `nr` and every
repetition count `0 < t`.
Against every malicious verifier, the record's simulator's view is within `ε_zk(x) + γ·Pr[fork]` of the real one, where
for a statement `x` in scope (`Claims.Flock.Scope` of the masked record)
$$\varepsilon_{zk}(x) = 8\,n_L(x)\,\delta_1 + 8\,t\,n_L(x)\,\delta_1 + 2 \cdot 3^{-t},$$
with `n_L(x)` the deployed plan's leaf count at `x` (`nL PE x`) and `δ₁` the hm96 salted-leaf hiding gap at the pinned
key (`FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash`); outside scope `ε_zk` is `1`. The fork term's factor
is `γ = 4·t·3^t`. It takes no other hypothesis. -/
def MaskedZK : Prop :=
  ∀ (nr : Claims.Msg → ℕ → ℕ) (t : ℕ), 0 < t →
    Claims.Flock.ZK FlockDeploy.sha512 (Claims.ZKMal.zkDeployed _root_.Flock.Readable.relation nr
        (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t

end Flock.Guarantees
