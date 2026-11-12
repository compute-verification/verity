module

public import Proofs.Flock.Claims.Sampling.Select.SecondCount
public import Proofs.Flock.Claims.Sampling.Select.Bounds
public import Proofs.Flock.Claims.Sampling.Reg
public import Proofs.Flock.Soundness.Discharge.Exec.LawDraw
import Spot.Draw
import Proofs.Core.QSort
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.SamplingGeneric.TwoCode

/-!
# The second stage's proof blocks by template (`sum_proved_selectSecond_le`)

`size_proved_selectSecond` counts an accepted second stage's proof blocks, `|sample| · Σ_t k_t`, every one alike. A
weight that differs by template sums over them to at most `|sample| · Σ_t k_t · M_t`, where `M_t` bounds it on template
`t`'s cells: each cell's stratum samples `k_t` of its units (`subset_size`), each below the stratum's size
(`subset_spec`), so each in the cell (`stratum_unit`), and the stratified sample returns the strata's samples together
(`stratifiedWith_sum_le`). At a registration, a unit's proof block is `RegTwo.blockAt`, which on template `t`'s cells
is a block of template `t` (`RegTwo.blockAt_cell`), so a weight on blocks at most `M_t` on template `t`'s blocks sums
over the proved units' blocks to at most `|sample| · Σ_t k_t · M_t` (`RegTwo.sum_proved_blockAt_le`), and over the
checked blocks of an accepted second stage, with a replay sample of at most `K0` blocks, to at most the always-checked
blocks' weight plus `K0 · Σ_t k_t · M_t` (`RegTwo.sum_checked_le_strata`).
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open FlockSoundness FlockSoundness.Audit

/-- A stratum's unit weighs at most the largest weight on its units. -/
theorem le_sup_unit (W : ℕ → ℕ) (s : Flock.Draw.Stratum) {j : ℕ} (hj : j < s.size) :
    W (s.unit j) ≤ (expand s.units.toList).toFinset.sup W := by
  rw [stratum_size'] at hj
  rw [stratum_unit, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, Option.getD_some]
  exact Finset.le_sup (List.mem_toFinset.2 (List.getElem_mem hj))

/-- **A stratified sample's weights**: the units a run of `stratifiedWith` on `take` returns weigh at most each
stratum's `k` times the largest weight on its units, summed over the strata. -/
theorem stratifiedWith_sum_le (src : ByteArray) (strata : Array Flock.Draw.Stratum) (p p' : ℕ) (D : Array ℕ)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run p = .ok (some D, p'))
    (W : ℕ → ℕ) :
    (D.toList.map W).sum ≤ (strata.toList.map fun s => s.k * (expand s.units.toList).toFinset.sup W).sum := by
  unfold Flock.Draw.stratifiedWith at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl] at h
  have loop : ∀ (l : List ℕ) (out : Array ℕ) (pos : ℕ) (r : Option (Option (Array ℕ))) (out' : Array ℕ) (pos₁ : ℕ),
      (forIn (m := StateT ℕ (Except String)) l ((none : Option (Option (Array ℕ))), out) (fun i __s => do
          let __x ← Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = .ok ((r, out'), pos₁) →
      r = none ∧ (out'.toList.map W).sum ≤ (out.toList.map W).sum +
        (l.map fun i => strata[i]!.k * (expand strata[i]!.units.toList).toFinset.sup W).sum := by
    intro l
    induction l with
    | nil =>
      intro out pos r out' pos₁ hl
      simp only [List.forIn_nil, StateT.run_pure] at hl
      obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj hl)
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
      exact ⟨rfl, by simp⟩
    | cons i is ih =>
      intro out pos r out' pos₁ hl
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind] at hl
      cases ho : (Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
          (Flock.Draw.subset strata[i]!.k strata[i]!.size)).run pos with
      | error e => rw [ho] at hl; exact absurd hl (by rw [bind_error, bind_error]; simp)
      | ok x =>
        obtain ⟨v, pos₂⟩ := x
        obtain ⟨j, Dl, rfl, rfl, hst⟩ := onWith_take src _ (by unfold Flock.Draw.Stratum.need; omega) _ pos _ _ ho
        rw [ho, bind_ok] at hl
        dsimp only at hl
        rw [StateT.run_pure, pure_bind] at hl
        dsimp only at hl
        obtain ⟨hr, hsum⟩ := ih _ _ r out' pos₁ hl
        refine ⟨hr, hsum.trans ?_⟩
        obtain ⟨π, hkn, hπ, hmem⟩ := FlockSoundness.Discharge.Exec.subset_spec hst.2.2.1
        have hb : ∀ y ∈ (Dl.map strata[i]!.unit).toList.map W,
            y ≤ (expand strata[i]!.units.toList).toFinset.sup W := by
          intro y hy
          obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hy
          rw [Array.toList_map] at hu
          obtain ⟨j', hj', rfl⟩ := List.mem_map.1 hu
          obtain ⟨t, ht, rfl⟩ := (hmem j').1 (Array.mem_toList_iff.1 hj')
          exact le_sup_unit W _ (hπ t (by omega))
        have hs := List.sum_le_length_nsmul _ _ hb
        rw [List.length_map, Array.length_toList, Array.size_map, subset_size hst.2.2.1, smul_eq_mul] at hs
        simp only [Array.toList_append, List.map_append, List.sum_append, List.map_cons, List.sum_cons]
        omega
  rw [StateT.run_bind] at h
  cases hl : (forIn (m := StateT ℕ (Except String)) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run p with
  | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨⟨r, out'⟩, pos₁⟩ := y
    rw [hl, bind_ok] at h
    dsimp only at h
    obtain ⟨rfl, hsum⟩ := loop _ _ _ _ _ _ hl
    rw [StateT.run_pure] at h
    obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj h)
    have hD : D = out'.qsort (fun x1 x2 => decide (x1 < x2)) := (Option.some.inj h1).symm
    rw [hD, ((Array.perm_iff_toList_perm.1 (DrawExec.qsort_perm out' _)).map W).sum_eq]
    refine hsum.trans (le_of_eq ?_)
    rw [Array.toList_empty, List.map_nil, List.sum_nil, Nat.zero_add]
    exact sum_range'_getElem! strata fun s => s.k * (expand s.units.toList).toFinset.sup W

/-- **A work sampling design's sample by stratum**: `units` weighs at most each stratum's `k` times the largest weight
on its units, summed over the strata. -/
theorem workUnits_sum_le {K n read : ℕ} {ws : Array Flock.Draw.WorkStratum} {src : ByteArray}
    {units : Array ℕ} (h : Draw.drawUnits (.work K ws) n src = .ok (units, read)) (W : ℕ → ℕ) :
    (units.toList.map W).sum ≤ (ws.toList.map fun s => s.k * (expand s.units.toList).toFinset.sup W).sum := by
  unfold Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Draw.take src need) (.work K ws) n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok x =>
    obtain ⟨r, read'⟩ := x
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hu, -⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hu
    unfold Flock.Draw.drawWith at hw
    dsimp only at hw
    rw [StateT.run_bind] at hw
    cases ho : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) (ws.map (·.toStratum))).run 0 with
    | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨v, p⟩ := y
      rw [ho, bind_ok, StateT.run_pure] at hw
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
      cases v with
      | none => exact absurd h1 (by simp)
      | some D =>
        obtain rfl := (Option.some.inj h1).symm
        refine (stratifiedWith_sum_le src _ 0 p D ho W).trans (le_of_eq ?_)
        rw [Array.toList_map, List.map_map]
        rfl

theorem filterMap_map_some {α β γ : Type} (xs : Array α) (g : α → β) (f : β → Option γ) (k : α → γ)
    (hf : ∀ a, f (g a) = some (k a)) : (xs.map g).filterMap f = xs.map k := by
  rw [Array.filterMap_map]
  have : (f ∘ g) = fun a => some (k a) := funext hf
  rw [this, Array.filterMap_eq_map' rfl]

/-- A sample object whose `units` are `units`, with no closure, proves those units: a weight sums alike over both. -/
theorem sum_proved_of_units {draw : Json} {units : Array ℕ} (hu : getD draw "units" = natsJson units)
    (hc : getD draw "closure" = Json.null) (W : ℕ → ℕ) :
    ((Draw.proved draw).toList.map fun j => W j.toNat).sum = (units.toList.map W).sum := by
  unfold Draw.proved
  dsimp only
  rw [hu, hc, ((Array.perm_iff_toList_perm.1 (DrawExec.qsort_perm _ _)).map _).sum_eq]
  simp only [natsJson]
  rw [filterMap_map_some units _ int? (fun a : ℕ => (a : Int)) int_toJson]
  simp [Function.comp_def]

/-- The cells over templates `L`, a weight on each cell's template: each template's, once for each of the `q` sampled
replay blocks. -/
theorem sum_cells_fst (q : ℕ) (f : ℕ → ℕ) (L : List ℕ) :
    ((L.flatMap fun t => (List.range q).map fun i => (t, i)).map fun z => f z.1).sum = q * (L.map f).sum := by
  induction L with
  | nil => simp
  | cons t L ih =>
    rw [List.flatMap_cons, List.map_append, List.sum_append, ih, List.map_cons, List.sum_cons, Nat.mul_add]
    congr 1
    rw [List.map_map]
    have hc : ((fun z : ℕ × ℕ => f z.1) ∘ fun i => (t, i)) = fun _ => f t := rfl
    rw [hc, sum_map_const, List.length_range]

theorem mem_expand_cell {lo hi x : ℕ} (hx : x ∈ expand (#[(lo, hi)] : Array (ℕ × ℕ)).toList) : lo ≤ x ∧ x < hi := by
  have he : expand (#[(lo, hi)] : Array (ℕ × ℕ)).toList = List.range' lo (hi - lo) ++ [] := rfl
  rw [he, List.append_nil, List.mem_range'_1] at hx
  omega

/-- **The second stage's proof blocks by template**: where a weight `W` on units is at most `M t` on template `t`'s
cells, it sums over an accepted second stage's proved units to at most `|sample| · Σ_t k_t · M t`. -/
theorem sum_proved_selectSecond_le {sample ns ks : Array ℕ} {src : ByteArray} {draw : Json} {read : ℕ}
    (h : Select.selectSecond sample ns ks src = .ok (draw, read)) (W M : ℕ → ℕ)
    (hW : ∀ t i j, t < ns.size → i < sample.size → j < ns.getD t 0 →
      W (cellOff sample.size ns t + i * ns.getD t 0 + j) ≤ M t) :
    ((Draw.proved draw).toList.map fun j => W j.toNat).sum ≤
      sample.size * ((List.range ns.size).map fun t => ks.getD t 0 * M t).sum := by
  obtain ⟨-, parsed, units, hp, hu, hdu, hdc⟩ := selectSecond_units h
  obtain ⟨ws, rfl, hmap⟩ := flockLaw_second sample ns ks _ parsed hp
  rw [sum_proved_of_units hdu hdc W]
  refine (workUnits_sum_le hu W).trans ?_
  have hws : (ws.toList.map fun s => s.k * (expand s.units.toList).toFinset.sup W) =
      (cellPairs sample.size ns.size).map fun z =>
        (cellOf sample.size ns ks z).2 * (expand (cellOf sample.size ns ks z).1.toList).toFinset.sup W := by
    have := congrArg (List.map fun c : Array (ℕ × ℕ) × ℕ => c.2 * (expand c.1.toList).toFinset.sup W) hmap
    simpa only [List.map_map, Function.comp_def] using this
  rw [hws, cellPairs, ← sum_cells_fst]
  refine List.sum_le_sum fun z hz => ?_
  obtain ⟨t, ht, hz⟩ := List.mem_flatMap.1 hz
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hz
  rw [List.mem_range] at ht hi
  show ks.getD t 0 * (expand (#[(cellOff sample.size ns t + i * ns.getD t 0,
      cellOff sample.size ns t + (i + 1) * ns.getD t 0)] : Array (ℕ × ℕ)).toList).toFinset.sup W ≤ ks.getD t 0 * M t
  refine Nat.mul_le_mul_left _ (Finset.sup_le fun x hx => ?_)
  obtain ⟨hlo, hhi⟩ := mem_expand_cell (List.mem_toFinset.1 hx)
  rw [Nat.succ_mul] at hhi
  have hj : x - (cellOff sample.size ns t + i * ns.getD t 0) < ns.getD t 0 := by omega
  have hx' := hW t i _ ht hi hj
  rwa [show cellOff sample.size ns t + i * ns.getD t 0 + (x - (cellOff sample.size ns t + i * ns.getD t 0)) = x by
    omega] at hx'

end Properties.Spot.SecurityProofs

namespace SamplingDeploy

open Properties.Spot Properties.Spot.SecurityProofs
open Lean (Json)

/-- **The proved units' blocks by template, at a registration**: a weight `w` on blocks, at most `M t` on template
`t`'s, sums over the blocks of an accepted second stage's proved units to at most `|sample| · Σ_t k_t · M t`. At the
replay sample `g.sample src₁` (at most `K` blocks), this is the by-template count `K · Σ_t k_t · M_t`. -/
theorem RegTwo.sum_proved_blockAt_le (g : RegTwo) (sample : Array ℕ) (src : ByteArray) {draw : Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond sample g.ns g.ks src = .ok (draw, n)) (w M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < g.ns.size → i < g.ns.getD t 0 → w (g.blockOf ρ t i) ≤ M t) :
    ((Spot.Draw.proved draw).toList.map fun j => w (g.blockAt sample j.toNat)).sum ≤
      sample.size * ((List.range g.ns.size).map fun t => g.ks.getD t 0 * M t).sum :=
  sum_proved_selectSecond_le hsel (fun j => w (g.blockAt sample j)) M fun t i j ht hi hj => by
    show w (g.blockAt sample (cellOff sample.size g.ns t + i * g.ns.getD t 0 + j)) ≤ M t
    rw [g.blockAt_cell sample ht hi hj]
    exact hM _ t j ht hj

theorem sum_toFinset_le (l : List ℕ) (w : ℕ → ℕ) : ∑ v ∈ l.toFinset, w v ≤ (l.map w).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    rw [List.toFinset_cons, List.map_cons, List.sum_cons]
    by_cases ha : a ∈ l.toFinset
    · rw [Finset.insert_eq_of_mem ha]; omega
    · rw [Finset.sum_insert ha]; omega

/-- The units below `N` that some `j ∈ P` maps to weigh at most the weight summed over `P`'s images. -/
theorem sum_filter_hit_le (N : ℕ) (P : Array ℤ) (f : ℤ → ℕ) (w : ℕ → ℕ) :
    (((List.range N).filter fun v => P.any fun j => f j == v).map w).sum ≤ (P.toList.map fun j => w (f j)).sum := by
  have hsub : ((List.range N).filter fun v => P.any fun j => f j == v).toFinset ⊆ (P.toList.map f).toFinset := by
    intro v hv
    obtain ⟨-, hq⟩ := List.mem_filter.1 (List.mem_toFinset.1 hv)
    rw [← Array.any_toList] at hq
    obtain ⟨j, hj, hjv⟩ := List.any_eq_true.1 hq
    exact List.mem_toFinset.2 (List.mem_map.2 ⟨j, hj, beq_iff_eq.1 hjv⟩)
  rw [← List.sum_toFinset w (List.nodup_range.filter _)]
  refine (Finset.sum_le_sum_of_subset hsub).trans ?_
  have := sum_toFinset_le (P.toList.map f) w
  rw [List.map_map] at this
  exact this

/-- **The checked proof blocks by template, at a registration**: on an accepted second stage over a replay sample of
at most `K0` blocks, a weight `w` on blocks, at most `M t` on template `t`'s, sums over the checked blocks below `N` to
at most the always-checked blocks' weight plus `K0 · Σ_t k_t · M t`. -/
theorem RegTwo.sum_checked_le_strata (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) {draw : Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) (w M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < g.ns.size → i < g.ns.getD t 0 → w (g.blockOf ρ t i) ≤ M t) {K0 : ℕ}
    (hK : (g.sample src₁).size ≤ K0) :
    ((g.checked N src₁ src₂).map w).sum ≤
      (((List.range N).filter fun v => g.always v).map w).sum +
        K0 * ((List.range g.ns.size).map fun t => g.ks.getD t 0 * M t).sum := by
  have hc : g.checked N src₁ src₂ = (List.range N).filter fun v =>
      g.always v || (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v := by
    unfold RegTwo.checked
    rw [hsel]
  have hor : ∀ (xs : List ℕ) (p q : ℕ → Bool),
      ((xs.filter fun v => p v || q v).map w).sum ≤ ((xs.filter p).map w).sum + ((xs.filter q).map w).sum := by
    intro xs p q
    induction xs with
    | nil => simp
    | cons a xs ih => by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq] <;> omega
  rw [hc]
  refine (hor _ _ _).trans (Nat.add_le_add_left ?_ _)
  exact (sum_filter_hit_le N _ _ w).trans ((g.sum_proved_blockAt_le _ src₂ hsel w M hM).trans
    (Nat.mul_le_mul_right _ hK))

end SamplingDeploy
