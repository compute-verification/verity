import FlockSoundness.Discharge.ZkBind.Bind
import FlockSoundness.Discharge.Layout.Keyed
import FlockSoundness.Protocol.Discharge.ZkBind.Shadow

/-!
# A `--zk` layout from a public one at the same statements

`Binding.Layout` and `Binding.HmRows` are about plans of `TabSpec` tables, `LayoutZ` and `HmRowsZ` about plans of
`TabZK` tables. Their facts read a drawn unit's table only through its statement and the witness a message packs. So a
public layout at plan `planS` gives a `--zk` one at plan `planZ` whenever every drawn unit's `--zk` table has the
statement of its public table (`e`): the addresses move along `e` (`adr`), and every witness of the public table's
statement is one a message packs (`wit_surj`, the `fast100` schedule's `m = 7 + k₀ + cols`), so a fact about every
satisfying public message is one about every satisfying `--zk` message.

* `LayoutZ.ofShadow`: the `--zk` layout, given that the two decoders agree on messages packing one witness (`hdec`);
* `HmRowsZ.ofShadow`: its rows, and `HmRowsZ.ofShadow_computes`: `HmRowComputes` moves along.
-/

namespace FlockSoundness.Discharge.ZkBind

open Game Model Accounting Refine Audit ZkSession ZkLink Binding

local notation "𝔽" => FlockLevel3.GF128

/-! ## Witnesses along an equality of statements -/

/-! ## The `--zk` layout of a public one -/

section Shadow

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length}
  {planS : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tabS : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planS S R).length}
  (e : ∀ S R u, (tabOfZ planZ tab S R u).S = (Partition.tabOf planS tabS S R u).S)
  {decZ : DecoderZK (C := C) planZ tab} {decS : Partition.DecoderB (C := C) (Reg := Reg) planS tabS}

variable (lay : Binding.Layout P planS tabS decS) (hdec : DecAgree e (decZ := decZ) (decS := decS) (P := P))

variable {lay} (rs : Binding.HmRows lay)

/-- **`HmRowComputes` moves along**: the public rows' `HmRowComputes` gives the `--zk` rows'. -/
theorem HmRowsZ.ofShadow_computes (hHm : rs.Computes) : (HmRowsZ.ofShadow e hdec rs).Computes := by
  rintro ⟨S, R, u, M', g⟩ ⟨hu, hM', hg, hrow⟩
  obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
  obtain ⟨hr, hs⟩ := row_ofShadow e lay hdec S R u M' M hM (lay.pos g)
  have h := hHm ⟨S, R, u, M, g⟩ ⟨hu, hS, hg, hrow⟩
  simp only at h
  rw [← hr, ← hs, hM] at h
  exact h

end Shadow

end FlockSoundness.Discharge.ZkBind
