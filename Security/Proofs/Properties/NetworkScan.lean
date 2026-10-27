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

/-- **`ScanFits`' conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a Program and a flattening of
it, built by hand, which a scan check accepting everything would pass, the layout doesn't fit. The Program has no
definitions, so its root is the default definition, of no gates and no return, and its flattening returns nothing:
step 0's egress presence gate isn't returned. The instance is the model's: no file decoding to it can be exhibited,
since `Flock.Program.decode` reads through `Lean.Json.parse`, a `partial def` the kernel can't unfold. -/
theorem scanFits_nontrivial :
    ∃ (P : Flock.Program.Prog) (C : Claims.Sampling.Prog) (s : NetworkCheck.Period.Schedule),
      Flattens P C ∧ ¬ (scanLayout P s).Fits C s.window := by
  refine ⟨default, ⟨[], []⟩, ⟨0, 0, 0, 0, 0, 0, 1, 0, 1, 1⟩,
    ⟨rfl, fun g hg => absurd hg (Nat.not_lt_zero g), fun g => ⟨fun h => by simp at h, ?_⟩⟩, ?_⟩
  · rintro ⟨j, hj, -⟩
    exact absurd hj (Nat.not_lt_zero j)
  · rintro ⟨h, -, -⟩
    have := h 0 (by decide) 0 ⟨0, by decide⟩ _ (List.mem_cons.2 (Or.inl rfl))
    simp at this

end Properties.NetworkScan
