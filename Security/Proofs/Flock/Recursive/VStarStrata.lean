import Proofs.Flock.Recursive.Flock

/-!
# Recursion: V*'s laws from their executable strata (gap 7)

`RecursiveAudit` takes `_hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St`: each of V*'s laws'
executable strata are its model's. `VLaw`'s model fields (`σs`, `ks`) are independent of its executable strata `St`, so
nothing the verifier checks gives `_hS` for an arbitrary `VLaw`. Here the model is read off the executable strata.

* `Law.Tiles S n`: the executable strata `S` tile the units `[0, n)`: each stratum's `j`-th unit, for `j` below its
  size, is a unit, distinct `j` give distinct units, every unit is one of exactly one stratum's, and each stratum draws
  at most its size. `Tiles.stratumOf` sends a unit to the stratum holding it.
* `Law.Tiles.execStrata`: tiling strata are the model's at `stratumOf` and their own draw counts.
  `Law.execStrata_iff` is the converse: `ExecStrata σ k S` holds exactly when `S` tiles, `σ` is `stratumOf` and `k` the
  strata's own, so `_hS` says no more than that.
* `VLaw.ofExec`: a law of V*'s units built from tiling executable strata, its model read off them, and
  `vstar_execStrata`: such a law satisfies `_hS`. `VLaw.execStrata_iff`: a law satisfies `_hS` exactly when its strata
  tile and its model is `ofExec`'s.

What gives `Tiles` at the executable is the verifier's U2 check of a stratified draw (`Flock.Draw.strataProblem`, with
`stratumOfJson`'s ranges: non-empty, ascending, disjoint, the strata's ranges tiling the population), once the draw
file is tied to `St` (`DrawFileZK`); or, at V*'s statements fixed, the strata themselves (gap 4).
-/

namespace FlockSoundness.Audit.Law

open Finset

variable {n m : ℕ} {S : Fin m → Flock.Draw.Stratum}

/-- **The executable strata tile the units `[0, n)`**: each stratum draws at most its size, its `j`-th unit for `j`
below its size is a unit, distinct `j` give distinct units, and every unit is one of exactly one stratum's. -/
structure Tiles (S : Fin m → Flock.Draw.Stratum) (n : ℕ) : Prop where
  k_le : ∀ s, (S s).k ≤ (S s).size
  lt : ∀ s, ∀ j < (S s).size, (S s).unit j < n
  inj : ∀ s, ∀ i < (S s).size, ∀ j < (S s).size, (S s).unit i = (S s).unit j → i = j
  cover : ∀ u : Fin n, ∃! s, u.val ∈ (range (S s).size).image (S s).unit

namespace Tiles

/-- The stratum holding a unit. -/
noncomputable def stratumOf (h : Tiles S n) (u : Fin n) : Fin m := (h.cover u).exists.choose

theorem mem_stratumOf (h : Tiles S n) (u : Fin n) :
    u.val ∈ (range (S (h.stratumOf u)).size).image (S (h.stratumOf u)).unit :=
  (h.cover u).exists.choose_spec

theorem stratumOf_eq (h : Tiles S n) {u : Fin n} {s : Fin m}
    (hs : u.val ∈ (range (S s).size).image (S s).unit) : h.stratumOf u = s :=
  (h.cover u).unique (h.mem_stratumOf u) hs

/-- `stratumOf`'s stratum `s` is stratum `s`'s units. -/
theorem stratum_map (h : Tiles S n) (s : Fin m) :
    (stratum h.stratumOf s).map Fin.valEmbedding = (range (S s).size).image (S s).unit := by
  ext x
  simp only [Finset.mem_map, mem_stratum, Fin.valEmbedding_apply]
  constructor
  · rintro ⟨u, rfl, rfl⟩
    exact h.mem_stratumOf u
  · intro hx
    obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hx
    exact ⟨⟨(S s).unit j, h.lt s j (Finset.mem_range.1 hj)⟩, h.stratumOf_eq hx, rfl⟩

theorem card_stratum (h : Tiles S n) (s : Fin m) : (stratum h.stratumOf s).card = (S s).size := by
  rw [← Finset.card_map Fin.valEmbedding, h.stratum_map, Finset.card_image_of_injOn, Finset.card_range]
  intro i hi j hj hij
  exact h.inj s i (Finset.mem_range.1 (Finset.mem_coe.1 hi)) j (Finset.mem_range.1 (Finset.mem_coe.1 hj)) hij

/-- **Tiling strata are the model's**, at `stratumOf` and their own draw counts. -/
theorem execStrata (h : Tiles S n) : ExecStrata h.stratumOf (fun s => (S s).k) S :=
  fun s => ⟨rfl, (h.card_stratum s).symm, (h.stratum_map s).symm⟩

end Tiles

/-- **`ExecStrata` is exactly tiling strata with the model read off them**: given `k s ≤ |σ⁻¹ s|`, the executable
strata `S` are `σ`'s at draw counts `k` iff they tile the units, `σ` sends each unit to the stratum holding it, and `k`
is the strata's own. -/
theorem execStrata_iff {σ : Fin n → Fin m} {k : Fin m → ℕ} (hk : ∀ s, k s ≤ (stratum σ s).card) :
    ExecStrata σ k S ↔ ∃ h : Tiles S n, σ = h.stratumOf ∧ k = fun s => (S s).k := by
  constructor
  · intro hS
    have hmem : ∀ s (x : ℕ), x ∈ (range (S s).size).image (S s).unit ↔ ∃ u : Fin n, σ u = s ∧ u.val = x := by
      intro s x
      rw [(hS s).2.2]
      simp [mem_stratum]
    have hT : Tiles S n :=
      { k_le := fun s => by rw [(hS s).1, (hS s).2.1]; exact hk s
        lt := fun s j hj => by
          obtain ⟨u, -, hu⟩ := (hmem s _).1 (Finset.mem_image_of_mem _ (Finset.mem_range.2 hj))
          rw [← hu]
          exact u.isLt
        inj := fun s i hi j hj hij => by
          have hc : ((range (S s).size).image (S s).unit).card = (range (S s).size).card := by
            rw [(hS s).2.2, Finset.card_map, Finset.card_range, (hS s).2.1]
          exact Finset.card_image_iff.1 hc (Finset.mem_coe.2 (Finset.mem_range.2 hi))
            (Finset.mem_coe.2 (Finset.mem_range.2 hj)) hij
        cover := fun u => ⟨σ u, (hmem _ _).2 ⟨u, rfl, rfl⟩, fun s hs => by
          obtain ⟨v, hv, hvu⟩ := (hmem s _).1 hs
          have hv' : v = u := Fin.ext hvu
          subst hv'
          exact hv.symm⟩ }
    refine ⟨hT, funext fun u => (hT.stratumOf_eq ((hmem _ _).2 ⟨u, rfl, rfl⟩)).symm, funext fun s => (hS s).1.symm⟩
  · rintro ⟨h, hσ, hk'⟩
    rw [hσ, hk']
    exact h.execStrata

end FlockSoundness.Audit.Law

namespace FlockSoundness.Discharge.Recursion

open Audit

/-- **A law of V*'s units from its executable strata**: the live stratified law on `St` (`VLaw.live`), and the model
read off `St` (`VLaw.model`): each unit's stratum the one holding it, each stratum's draw count its own. -/
noncomputable def VLaw.ofExec {nv m : ℕ} (St : Fin m → Flock.Draw.Stratum) (hT : Law.Tiles St nv) (Kd pop : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256) :
    VLaw nv where
  m := m
  σs := hT.stratumOf
  ks s := (St s).k
  hks s := by rw [hT.card_stratum]; exact hT.k_le s
  St := St
  Kd := Kd
  pop := pop
  Ω := Ω
  fintypeΩ := inferInstance
  nonemptyΩ := inferInstance
  Lb := Lb
  src := src

/-- **Gap 7: a law built from its executable strata has them as its model's**, `RecursiveAudit`'s `_hS` at it. -/
theorem vstar_execStrata {nv m : ℕ} (St : Fin m → Flock.Draw.Stratum) (hT : Law.Tiles St nv) (Kd pop : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256) :
    Law.ExecStrata (VLaw.ofExec St hT Kd pop Ω Lb src).σs (VLaw.ofExec St hT Kd pop Ω Lb src).ks
      (VLaw.ofExec St hT Kd pop Ω Lb src).St :=
  hT.execStrata

/-- **`_hS` at a law is exactly `ofExec`'s model**: its executable strata tile its units, and its model is the one
`VLaw.ofExec` reads off them. -/
theorem VLaw.execStrata_iff {nv : ℕ} (law : VLaw nv) :
    Law.ExecStrata law.σs law.ks law.St ↔
      ∃ hT : Law.Tiles law.St nv, law.σs = hT.stratumOf ∧ law.ks = fun s => (law.St s).k :=
  Law.execStrata_iff law.hks

end FlockSoundness.Discharge.Recursion
