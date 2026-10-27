import Proofs.Accounting.TS.Session.Tile.Guarantees
import Proofs.Accounting.TS.HeldSound

/-!
# A tile unit's proof layer with its Definition's checks is sound through its C-Flock sessions (the proof)

`Accounting.SecurityProofs.TS.TileSessionSound`, as `Accounting.SecurityProofs.TS.HeldSessionSound` is proved. The tile
layer accepts where the checked layer does and flock-verify's known-content and complete checks pass, and opens the
same rows, so `AcceptedLive` is its acceptance's first conjunct. Its unsound event is inside the checked layer's
(`Pouw.RelativeSecurity.prob_mono`), because on every path an accepted proof whose rows meet `CheckedRel` meets
`TileRel` (`tile_of_checked`): each `ok` wire is at the `ok` row unless the `ok` tie collides
(`Accounting.TS.Held.tie_ok`), the checked relation's assignment is the opened rows there, and
`Accounting.Assumptions.TS.TileChecks`, which `TileComputes` gives at the tile's `K`
(`Accounting.TS.Tile.tileChecks_of_computes`), gives each drawn unit's decoded cells.
-/

namespace Accounting.TS.Tile

open Accounting.TS Accounting.TS.Checked Accounting.TS.Held Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg

/-- **An accepted proof's rows that meet `CheckedRel` meet `TileRel`**: at a session whose known-content check passed,
naming `ok` with rows each the word `1`, whose served roots are registered and whose circuit computes the tile's
Definition at `dec`, at a `K` the tile binds at. -/
theorem tile_of_checked {R C K : ℕ} (s : RowsSession) (o : s.Out) (ok : String)
    (known : Array (String × Array (Array ℕ × ByteArray))) {ks : Array (Array ℕ × ByteArray)} {dq : List ℤ → ℤ}
    {dec : (ℕ → ℕ → Option Bool) → ℕ → TileView R C K}
    (hk : Flock.Registered.checkKnownAll s.own known = .ok ()) (hm : (ok, ks) ∈ known)
    (hks : ∀ r ∈ ks, r.1 = okRow) (hS : ∀ u, s.ServedRegistered u)
    (hT : Accounting.Assumptions.TS.TileChecks dq s ok dec) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17)
    (hc : CheckedRel s o ok (openedReg s)) : TileRel s o ok dq dec known (openedReg s) := by
  refine ⟨hc, ?_⟩
  by_cases hcol : OkColl s ok ks
  · exact Or.inr ⟨ks, hm, hcol⟩
  obtain ⟨-, vals, hv, hagree⟩ := hc
  refine Or.inl fun u hu => ?_
  rw [hv]
  refine hT hK32 hK vals u (hagree u hu) fun g hw => ⟨_, tie_ok s ok hk hm hks hS hcol hw, ?_⟩
  obtain ⟨hg, hr, -⟩ := hw
  have e := congrFun (congrFun hv u.val) g.val
  simp only [openedReg, rowsReg] at e
  rw [regRead_at s _ hg hr, regRead_at s _ hg hr] at e
  exact (Option.some.inj e).symm

end Accounting.TS.Tile

open Accounting.TS Accounting.TS.Checked Accounting.TS.Tile in
theorem Accounting.SecurityProofs.TS.TileSessionSound : Accounting.Guarantees.TS.TileSessionSound := by
  intro Ω _ S nπ R C K P sess out ok known form dqW rd right δ hCR hLaw hk hS hok hKn hK32 hK hT hδ hδ0
  refine ⟨fun _ _ _ h => h.1, fun w i => ?_⟩
  have hC := (Accounting.SecurityProofs.TS.CheckedSessionSound sess out ok right δ hCR hLaw hk hS hok hδ
    hδ0).2.2.2
  have hsub : ∀ x, (tileLayer sess out ok known (Lanes.dqOf dqW) (fun x w i v u => tileView (rd x w i v u))
      right).Unsound x w i → (checkedLayer sess out ok right).Unsound x w i := by
    intro x hx
    obtain ⟨⟨hacc, hkn, -⟩, hnot⟩ := hx
    refine ⟨hacc, fun hc => hnot ?_⟩
    obtain ⟨ks, hm, hks⟩ := hKn x w i
    exact tile_of_checked (sess x w i) (out x w i) ok (known x w i) hkn hm hks (hS x w i)
      (tileChecks_of_computes form dqW (sess x w i) ok (rd x w i) (hT x w i)) hK32 hK hc
  exact le_trans (Pouw.RelativeSecurity.prob_mono hsub) (hC w i)
