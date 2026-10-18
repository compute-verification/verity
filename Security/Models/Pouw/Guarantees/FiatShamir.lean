import Definitions.Pouw.FiatShamir
import Models.Pouw.Assumptions.FiatShamir

/-!
# Relative security under Fiat–Shamir draws: the grinding term (trusted)

Decision 102 derives each matmul's draw from SHA-512 of commitments the prover has already made (its operand roots, at
the window's live salt), in place of `ServiceSoundStaged`'s live per-stage coins. The prover can then re-roll a root
and try again. The statement here prices that: under a uniform oracle, `T` tries cost the live floor `T·μ^m`, where
`μ^m` bounds one try's chance to steer (`Pouw.Assumptions.SteerBound`). Its proof is to be
`Pouw.SecurityProofs.FiatShamir.FiatShamirFloor : Pouw.Guarantees.FiatShamir.FiatShamirFloor`.
-/

namespace Pouw.Guarantees.FiatShamir

open Pouw.RelativeSecurity Pouw.Assumptions

/-- **Fiat–Shamir draws cost the live floor `T·μ^m`** (proofs' gap (7), note:20261018T0309Z-draft-pouw-spec). For
every service whose stage draws are a uniform oracle's answers at the inputs its commitment names (the random-oracle
model: SHA-512 at the window's salt, decision 102's `tag ‖ s ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)`), whose
other coins are live and drawn after the commitment, and every analysis's steer events `Steer`: if a try steers with
probability at most `μ^m` (`SteerBound`) and the live coins alone bound accepted runs charged below `F` at unsteered
draws by `ε` (`LiveFloorUnsteered`, the live statement), then every adversary within `T` oracle calls, the inputs its
commitment names counted, has its accepted runs charged below `F` with probability at most

  `ε + T·μ^m`

over the oracle, the live coins and its own coins (`ServiceFloorFS`). At `T = |J|`, one try a stage, it is the live
floor's split: a live adversary's draws are uniform, so they steer with probability at most `|J|·μ^m`. -/
def FiatShamirFloor : Prop :=
  ∀ (J N Live : Type) [Fintype J] [Fintype N] [Fintype Live] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (Steer : SteerMap Mc S) (F ε μ : ℝ) (m : ℕ), 0 ≤ ε → 0 ≤ μ →
    SteerBound Mc S T Steer μ m → LiveFloorUnsteered Mc S T Steer F ε →
      ServiceFloorFS Mc S T F (ε + T * μ ^ m)

end Pouw.Guarantees.FiatShamir
