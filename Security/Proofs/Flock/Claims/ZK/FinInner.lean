import Proofs.Flock.Claims.ZK.FinPads
import Proofs.Flock.Claims.ZK.FinTau

/-!
# `FinExe` from the hybrids, the live end and `InnerTauVal`

`repOpenRest` proves `RepOpenRest` from its two halves: level 0's opened rows and salts (`repOpenRest_l0`) and the
pads' opened columns and salts (`repOpenRest_pads`). With `RepTau` from `InnerTauVal` (`repTau_of`), `finExe_of_rest`
gives `finExe_of_inner`.
-/

namespace Claims.ZKMal

namespace Masked

/-- **The final-only data at opened positions read from the views is the session's** (`RepOpenRest`): level 0's
opened rows and salts (`repOpenRest_l0`) and the pads' opened columns and salts (`repOpenRest_pads`). -/
theorem repOpenRest : RepOpenRest := by
  intro x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  obtain ⟨h1, h2⟩ := repOpenRest_l0 x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  obtain ⟨h3, h4⟩ := repOpenRest_pads x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  exact ⟨h1, h2, h3, h4⟩

/-- **`FinExe` from the hybrids, the live end and `InnerTauVal`**: `finExe_of_rest` with `RepTau` from `InnerTauVal`
(`repTau_of`) and `RepOpenRest` proved (`repOpenRest`). -/
theorem finExe_of_inner (hL : LiveEnd) (hSC : SimCommit) (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner)
    (hT : InnerTauVal) : FinExe :=
  finExe_of_rest hL hSC h0 h1 h2 (repTau_of hT) repOpenRest

end Masked

end Claims.ZKMal
