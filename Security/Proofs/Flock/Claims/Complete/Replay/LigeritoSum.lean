import Proofs.Flock.Claims.Complete.Replay.LigeritoBasis

/-!
# Ring switching's claim is the packed table's

`RsSum` at the deployed relation (`rsSum_relation`): at every rep, `Σ_y packed(y)·basisVals(y)` is ring switching's
batched value `T = Σ_k γ_k·Σ_u eq(r''_k, u)·t_k[u]`.

Both sides are `Σ_k γ_k Σ_u eq(r''_k, u) Σ_v x^v Σ_y z[128y + v]·bit_u(eq(z_k, y))` (`Lg.rsSum_eq`). On the right,
`t_k[u] = Σ_v bit_u(s_k[v])·x^v` with `s_k[v] = Σ_y z[128y + v]·eq(z_k, y)`, and a bit of a sum is the sum of the
bits (`Lg.bF_bit_sumF`: bits are the packing basis's coordinates, `unpack`, which is additive).
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

namespace Lg

/-- A bit as a field element. -/
noncomputable def bF (b : Bool) : 𝔽 := if b then 1 else 0

theorem fOf_ite (b : Bool) (a : F128) : fOf (if b = true then a else 0) = bF b * fOf a := by
  cases b <;> simp [bF, fOf_zero]

theorem fOf_foldl_if (c : ℕ → Bool) (g : ℕ → F128) : ∀ (k : ℕ) (acc : F128),
    fOf ((List.range k).foldl (fun acc v => if c v = true then acc + g v else acc) acc) =
      fOf acc + ∑ v ∈ Finset.range k, bF (c v) * fOf (g v) := by
  intro k
  induction k with
  | zero => intro acc; simp
  | succ k ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Finset.sum_range_succ]
      cases hc : c k
      · simp only [Bool.false_eq_true, ite_false, ih, bF, zero_mul, add_zero]
      · simp only [ite_true, fOf_add, ih, bF, one_mul, add_assoc]

theorem list_sum_rng {α : Type} [Inhabited α] (a : Array α) (h : α → 𝔽) :
    (a.toList.map h).sum = ∑ k ∈ Finset.range a.size, h a[k]! := by
  rw [← FlockLevel3.list_sum_range]
  congr 1
  refine List.ext_getElem (by simp) fun i h₁ _ => ?_
  simp only [List.getElem_map, Array.getElem_toList, List.getElem_range]
  rw [getElem!_pos a i (by simpa using h₁)]

theorem fOf_dot_rng (a b : Array F128) (n : ℕ) (ha : a.size = n) :
    fOf (dot a b) = ∑ i ∈ Finset.range n, fOf a[i]! * fOf b[i]! := by
  rw [fOf_dot a b n ha, Fin.sum_univ_eq_sum_range (fun i => fOf a[i]! * fOf b[i]!) n]

/-! ## A bit of a sum -/

/-- **Bits are the packing basis's coordinates**, as field elements. -/
theorem bF_bit (a : F128) (u : ℕ) (hu : u < 128) :
    bF (a.bit u) = ZMod.castHom (dvd_refl 2) 𝔽 (FlockLevel3.unpack (fOf a) ⟨u, hu⟩) := by
  rw [unpack_fOf]
  cases a.bit u <;> simp [bF]

/-- **A bit of a sum is the sum of the bits**: bit `u` of `Σ_y [c_y]·e_y` is `Σ_y [c_y]·bit_u(e_y)` in the field. -/
theorem bF_bit_sumF (N : ℕ) (c : ℕ → Bool) (e : ℕ → F128) (u : ℕ) (hu : u < 128) :
    bF ((sumF N fun y => if c y = true then e y else 0).bit u) =
      ∑ y ∈ Finset.range N, bF (c y) * bF ((e y).bit u) := by
  rw [bF_bit _ u hu]
  have hs : fOf (sumF N fun y => if c y = true then e y else 0) =
      ∑ y ∈ Finset.range N, fOf (if c y = true then e y else 0) := Zc.of_sumF N _
  have hu' := congrFun (map_sum FlockLevel3.unpackHom (fun y => fOf (if c y = true then e y else 0))
    (Finset.range N)) ⟨u, hu⟩
  rw [Finset.sum_apply] at hu'
  change FlockLevel3.unpack _ _ = ∑ y ∈ Finset.range N, FlockLevel3.unpack _ _ at hu'
  rw [hs, hu', map_sum]
  refine Finset.sum_congr rfl fun y _ => ?_
  cases c y
  · simp [bF, fOf_zero, FlockLevel3.unpack_zero]
  · simp only [ite_true]
    rw [← bF_bit _ u hu]
    simp [bF]

/-! ## The two sides -/

theorem fOf_packed (tr : Trace) (m y : ℕ) (hy : y < 2 ^ (m - 7)) :
    fOf (packed tr m)[y]! = ∑ v ∈ Finset.range 128, bF tr.z[128 * y + v]! * fOf (F128.monomial v) := by
  unfold packed
  rw [FlockSoundness.Refine.getElem!_range_map _ hy, fOf_foldl_if (fun v => tr.z[128 * y + v]!) F128.monomial,
    fOf_zero, zero_add]

theorem fOf_basisVals_raw (sws : Array Switched) (l w : ℕ) (hw : w < 2 ^ l) :
    fOf (basisVals sws l)[w]! = ∑ k ∈ Finset.range sws.size, fOf (sws[k]!).gamma *
      ∑ u ∈ Finset.range 128, bF (((eqTable (sws[k]!).z)[w]!).bit u) * fOf (eqTable (sws[k]!).rpp)[u]! := by
  unfold basisVals
  rw [FlockSoundness.Refine.getElem!_range_map _ hw, Array.foldl_map, ← Array.foldl_toList]
  dsimp only
  rw [fOf_foldl_acc, fOf_zero, zero_add, list_sum_rng]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [fOf_mul]
  congr 1
  show FlockLevel3.GF128.of _ = _
  rw [Zc.of_sumF]
  exact Finset.sum_congr rfl fun u _ => fOf_ite _ _

theorem fOf_tOf_sum (s : Array F128) (u : ℕ) (hu : u < 128) :
    fOf (tOf s)[u]! = ∑ v ∈ Finset.range 128, bF (s[v]!.bit u) * fOf (F128.monomial v) := by
  unfold tOf
  rw [FlockSoundness.Refine.getElem!_range_map _ hu, ← Array.foldl_toList, Array.toList_range,
    fOf_foldl_if (fun v => s[v]!.bit u) F128.monomial, fOf_zero, zero_add]

theorem bF_switchVec (tr : Trace) (c : Claim) (m : ℕ) (hx : c.x.size = m - 6) (v u : ℕ) (hv : v < 128)
    (hu : u < 128) :
    bF ((switchVec tr c)[v]!.bit u) = ∑ y ∈ Finset.range (2 ^ (m - 7)),
      bF tr.z[128 * y + v]! * bF (((eqTable (c.x.extract 1 c.x.size))[y]!).bit u) := by
  unfold switchVec
  dsimp only
  rw [FlockSoundness.Refine.getElem!_range_map _ hv, bF_bit_sumF _ (fun y => tr.z[128 * y + v]!) _ u hu,
    FlockLevel3.eqTable_size, Array.size_extract, Nat.min_self, hx, show m - 6 - 1 = m - 7 by omega]

theorem sws_get (claims : Array Claim) (cs : Rs.Challenges) (k : ℕ) (hk : k < claims.size) :
    (Rs.sws claims cs)[k]! =
      { gamma := cs.cg k, rpp := cs.rpp k, z := claims[k]!.x.extract 1 claims[k]!.x.size } := by
  rw [Rs.sws, FlockSoundness.Refine.getElem!_range_map _ hk, Rs.perClaim, pushes_get _ _ k hk, Rs.gamma,
    FlockSoundness.Refine.getElem!_range_map _ hk]

/-- The sums' reordering (over any commutative semiring: `Finset.mul_sum` doesn't rewrite at `GF128`'s instances,
but they unify with it). -/
theorem sum_reorder {K : Type} [CommSemiring K] (N n : ℕ) (c : ℕ → ℕ → K) (b : ℕ → ℕ → ℕ → K) (xv G : ℕ → K)
    (R : ℕ → ℕ → K) :
    ∑ y ∈ Finset.range N, (∑ v ∈ Finset.range 128, c y v * xv v) *
      ∑ k ∈ Finset.range n, G k * ∑ u ∈ Finset.range 128, b k y u * R k u =
    ∑ k ∈ Finset.range n, G k * ∑ u ∈ Finset.range 128, R k u *
      ∑ v ∈ Finset.range 128, (∑ y ∈ Finset.range N, c y v * b k y u) * xv v := by
  simp only [Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_comm.trans (Finset.sum_congr rfl fun k _ => Finset.sum_comm.trans
    (Finset.sum_congr rfl fun u _ => Finset.sum_comm.trans
      (Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun y _ => ?_)))
  ring

/-- **Ring switching's claim is the packed table's**: at claims of `m − 6` coordinates,
`Σ_y packed(y)·basisVals(y) = T`. -/
theorem rsSum_eq (tr : Trace) (m : ℕ) (claims : Array Claim) (cs : Rs.Challenges)
    (hx : ∀ c ∈ claims, c.x.size = m - 6) :
    dot (packed tr m) (basisVals (Rs.sws claims cs) (m - 7)) = Rs.T tr claims cs := by
  have hxk : ∀ k < claims.size, claims[k]!.x.size = m - 6 := fun k hk => by
    rw [getElem!_pos claims k hk]
    exact hx _ (Array.getElem_mem hk)
  have hrpp : ∀ k, (eqTable (cs.rpp k)).size = 128 := fun k => by
    rw [FlockLevel3.eqTable_size]
    simp [Rs.Challenges.rpp]
  apply fOf_injective
  rw [fOf_dot_rng _ _ (2 ^ (m - 7)) (by simp [packed]), Rs.T,
    fOf_dot_rng _ _ claims.size (by simp [Rs.gamma])]
  have hl : ∀ y ∈ Finset.range (2 ^ (m - 7)), fOf (packed tr m)[y]! *
      fOf (basisVals (Rs.sws claims cs) (m - 7))[y]! =
      (∑ v ∈ Finset.range 128, bF tr.z[128 * y + v]! * fOf (F128.monomial v)) *
        ∑ k ∈ Finset.range claims.size, fOf (cs.cg k) * ∑ u ∈ Finset.range 128,
          bF (((eqTable (claims[k]!.x.extract 1 claims[k]!.x.size))[y]!).bit u) *
            fOf (eqTable (cs.rpp k))[u]! := by
    intro y hy
    rw [Finset.mem_range] at hy
    rw [fOf_packed tr m y hy, fOf_basisVals_raw _ _ y hy,
      show (Rs.sws claims cs).size = claims.size by simp [Rs.sws]]
    refine congrArg (_ * ·) (Finset.sum_congr rfl fun k hk => ?_)
    rw [sws_get claims cs k (Finset.mem_range.1 hk)]
  have hr : ∀ k ∈ Finset.range claims.size, fOf (Rs.gamma cs claims.size)[k]! * fOf (Rs.betas tr claims cs)[k]! =
      fOf (cs.cg k) * ∑ u ∈ Finset.range 128, fOf (eqTable (cs.rpp k))[u]! *
        ∑ v ∈ Finset.range 128, (∑ y ∈ Finset.range (2 ^ (m - 7)), bF tr.z[128 * y + v]! *
          bF (((eqTable (claims[k]!.x.extract 1 claims[k]!.x.size))[y]!).bit u)) * fOf (F128.monomial v) := by
    intro k hk
    rw [Finset.mem_range] at hk
    rw [Rs.gamma, FlockSoundness.Refine.getElem!_range_map _ hk, Rs.betas, pushes_get _ _ k hk,
      fOf_dot_rng _ _ 128 (hrpp k)]
    refine congrArg (_ * ·) (Finset.sum_congr rfl fun u hu => ?_)
    rw [Finset.mem_range] at hu
    rw [fOf_tOf_sum _ u hu]
    refine congrArg (_ * ·) (Finset.sum_congr rfl fun v hv => ?_)
    rw [bF_switchVec tr _ m (hxk k hk) v u (Finset.mem_range.1 hv) hu]
  rw [Finset.sum_congr rfl hl, Finset.sum_congr rfl hr]
  exact sum_reorder _ _ (fun y v => bF tr.z[128 * y + v]!)
    (fun k y u => bF (((eqTable (claims[k]!.x.extract 1 claims[k]!.x.size))[y]!).bit u))
    (fun v => fOf (F128.monomial v)) (fun k => fOf (cs.cg k)) (fun k u => fOf (eqTable (cs.rpp k))[u]!)

end Lg

/-- **`RsSum` at the deployed relation**: at every rep, ring switching's claim is the packed table's. -/
theorem rsSum_relation (H : Claims.Msg → Claims.Msg) : RsSum H Readable.relation :=
  fun o i hi r _ => Lg.rsSum_eq (o.trace i) _ (o.claims i r) (o.rsCoins i r) (o.claims_xsize hi r)

/-- **Ligerito's final identity at the deployed relation** (`LgFinal`), at every rep the walk runs on its packed table,
from the honest Ligerito's. -/
theorem lgFinal_relation (H : Claims.Msg → Claims.Msg)
    (hw : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalkP o i r) (hH : Lg.Honest) :
    LgFinal H Readable.relation :=
  lgFinal_of H _ hw hH (rsSum_relation H) (basisOk_relation H)

end FlockComplete.Replay
