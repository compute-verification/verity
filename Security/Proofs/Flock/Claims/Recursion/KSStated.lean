import Proofs.Flock.Claims.Recursion.KS

/-!
# The recursion's KS at its stated errors

The recursion's claims state `γ` and `γ_e` at the errors `KS_of` proves (`γ = γ_A`, `γ_e = γ_eA`), so the KS that
`KS_of` proves is the claim: `KSA` is `KS` (`KSA_iff_KS`), and `KSFlockA` is `KSFlock` (`KSFlockA_iff_KSFlock`).
-/

namespace Claims.Recursion

theorem γA_eq_γ {Coin W : Type} (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : γA D I x = γ D I x :=
  rfl

theorem γeA_eq_γe {Coin W : Type} (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : γeA D I x = γe D I x :=
  rfl

theorem KSA_iff_KS {Coin W Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) :
    KSA H D I DO ↔ KS H D I DO :=
  Iff.rfl

theorem KSFlockA_iff_KSFlock {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [DecidableEq Col'] [Zero Col] [Zero Col'] (H : Msg → Msg) (D : Deploy Coin)
    (DI : Flock.Deploy Coin Col T) (DO : Flock.Deploy Coin Col' T') :
    KSFlockA H D DI DO ↔ KSFlock H D DI DO :=
  Iff.rfl

end Claims.Recursion
