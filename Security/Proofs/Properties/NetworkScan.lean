import Properties.NetworkScan
import Proofs.Properties.NetworkScan.Fits

/-!
# An accepted scan's statements are in the sampled audit's scope, from its layout fitting

`scanScopeAcc_of_scanFits`: the sampling half is the scope check's (`ScopeMatches`), and the layout half is `ScanFits`
at the statement's descriptor, the scan's canonical bytes, flattened (`StmtFlattens`).
-/

namespace Properties.NetworkScan

theorem scanScopeAcc_of_scanFits (h : ScanFits) : ScanScopeAcc := by
  intro Coin Col T _ _ _ _ _ D j program s sc c x hflat hmatch hsc hscan hc hdesc
  have hS : Claims.Sampling.Scope D x := (hmatch x).2 hsc
  refine ⟨hS, ?_⟩
  obtain ⟨st, _, hx, _⟩ := hS
  obtain ⟨P, hP, hF⟩ := hflat x st hx
  refine ⟨st, hx, ?_⟩
  unfold layOf
  rw [hP]
  rw [hdesc st hx] at hP
  exact h j program s sc c P st.prog hscan hc hP hF

end Properties.NetworkScan
