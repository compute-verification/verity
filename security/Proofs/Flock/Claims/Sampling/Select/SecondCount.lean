module

public import Proofs.Flock.Claims.Sampling.Select.Second
public import Proofs.Flock.Claims.Sampling.Select.Size
import Spot.Draw
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import Proofs.Core.QSort
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# How many proof blocks the second stage proves (`size_proved_selectSecond`)

The second stage's law has one cell per template `t` and sampled replay block, each sampling `k_t` of its `n_t` proof
blocks (`flockLaw_second`); its draw returns each cell's count (`stratifiedWith_size`), and its sample object carries
those units and no closure. So an accepted second stage proves exactly `|sample| · Σ_t k_t` proof blocks. At a
registration the replay sample is `RegTwo.sample src₁`, the units the first stage proves, and this is the count
`Kfirst · Σ_t k_t` of the drawn blocks whose gates the checked proof blocks need.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)

theorem getD_draw_closure (sample ns ks : Array ℕ) (m : ℕ) (units : Array ℕ) :
    getD (Json.mkObj ((entries (Select.secondLaw sample ns ks)).toList ++
      [("population", Lean.toJson m), ("units", natsJson units)])) "closure" = Json.null := (rfl)

/-- **An accepted second stage's sample object**: it passed `secondProblem`, drew `units` under `secondLaw`'s Flock law
over the `|sample|·Σ n_t` proof blocks, and carries those units and no closure. -/
theorem selectSecond_units {sample ns ks : Array ℕ} {src : ByteArray} {draw : Json} {read : ℕ}
    (h : Select.selectSecond sample ns ks src = .ok (draw, read)) :
    Select.secondProblem sample ns ks = none ∧
      ∃ parsed units, Draw.flockLaw (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) = .ok parsed ∧
        Draw.drawUnits parsed (sample.size * ns.toList.sum) src = .ok (units, read) ∧
          getD draw "units" = natsJson units ∧ getD draw "closure" = Json.null := by
  unfold Select.selectSecond at h
  cases hq : Select.secondProblem sample ns ks with
  | some e => rw [hq] at h; dsimp only at h; exact absurd h (throw_bind_ne _ _ _)
  | none =>
    rw [hq] at h
    dsimp only at h
    refine ⟨rfl, ?_⟩
    obtain ⟨⟨draw', read'⟩, hd, h⟩ := bind_eq_ok.1 h
    split at h
    · exact absurd h (throw_bind_ne _ _ _)
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    unfold Draw.drawFirstRead at hd
    obtain ⟨parsed, hp, hd⟩ := bind_eq_ok.1 hd
    obtain ⟨⟨units, read''⟩, hu, hd⟩ := bind_eq_ok.1 hd
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hd
    obtain ⟨rfl, rfl⟩ := hd
    exact ⟨parsed, units, hp, hu, getD_draw_units _ _ _ _ _, getD_draw_closure _ _ _ _ _⟩

theorem size_filterMap_map_some {α β γ : Type} (xs : Array α) (g : α → β) (f : β → Option γ) (k : α → γ)
    (hf : ∀ a, f (g a) = some (k a)) : ((xs.map g).filterMap f).size = xs.size := by
  rw [Array.filterMap_map]
  have : (f ∘ g) = fun a => some (k a) := funext hf
  rw [this, Array.filterMap_eq_map' rfl, Array.size_map]

/-- A sample object whose `units` are `units`, with no closure, proves exactly as many units. -/
theorem size_proved_of_units {draw : Json} {units : Array ℕ} (hu : getD draw "units" = natsJson units)
    (hc : getD draw "closure" = Json.null) : (Draw.proved draw).size = units.size := by
  unfold Draw.proved
  dsimp only
  rw [hu, hc, Verity.QSort.size_qsort, Array.size_append]
  simp only [natsJson, Array.size_empty, Nat.add_zero]
  exact size_filterMap_map_some units _ int? (fun a : ℕ => (a : Int)) int_toJson

/-- **A work law's draw returns its strata's counts**: `units` has `Σ_s k_s` entries. -/
theorem drawUnits_work_size {K n read : ℕ} {ws : Array Flock.Draw.WorkStratum} {src : ByteArray} {units : Array ℕ}
    (h : Draw.drawUnits (.work K ws) n src = .ok (units, read)) : units.size = (ws.toList.map (·.k)).sum := by
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
        refine (stratifiedWith_size src _ 0 p D ho).trans ?_
        rw [Array.toList_map, List.map_map]
        rfl

/-- A second stage `secondProblem` passes has a count for each template's cells. -/
theorem secondProblem_size {sample ns ks : Array ℕ} (h : Select.secondProblem sample ns ks = none) :
    ks.size = ns.size := by
  unfold Select.secondProblem at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hns
      by_contra hne
      have hb : (ks.size != ns.size) = true := bne_iff_ne.2 hne
      exact hns (by rw [hb, Bool.or_true, Bool.true_or])

theorem sum_map_const {α : Type} (l : List α) (c : ℕ) : (l.map fun _ => c).sum = l.length * c := by
  induction l with
  | nil => simp
  | cons a l ih => rw [List.map_cons, List.sum_cons, ih, List.length_cons, Nat.succ_mul, Nat.add_comm]

/-- The cells' counts over templates `L`: each template's `k_t`, once for each of the `q` sampled replay blocks. -/
theorem sum_cells_k (q : ℕ) (ns ks : Array ℕ) (L : List ℕ) :
    ((L.flatMap fun t => (List.range q).map fun i => (t, i)).map fun z => (cellOf q ns ks z).2).sum =
      q * (L.map fun t => ks.getD t 0).sum := by
  induction L with
  | nil => simp
  | cons t L ih =>
    rw [List.flatMap_cons, List.map_append, List.sum_append, ih, List.map_cons, List.sum_cons, Nat.mul_add]
    congr 1
    rw [List.map_map]
    have hc : ((fun z => (cellOf q ns ks z).2) ∘ fun i => (t, i)) = fun _ => ks.getD t 0 := rfl
    rw [hc, sum_map_const, List.length_range]

theorem map_range_getD (l : List ℕ) : (List.range l.length).map (fun t => l.getD t 0) = l := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [Function.comp_def, List.getD_cons_succ, List.getD_cons_zero, ih]

theorem sum_range_getD (a : Array ℕ) : ((List.range a.size).map fun t => a.getD t 0).sum = a.toList.sum := by
  simp only [getD_toList]
  rw [← Array.length_toList, map_range_getD]

/-- **The second stage proves `|sample| · Σ_t k_t` proof blocks**: an accepted second stage's sample proves exactly
`k_t` proof blocks of each template `t`'s cell in each sampled replay block, and no others. At a registration,
`sample` is the replay sample `RegTwo.sample src₁`, and this is the count `Kfirst · Σ_t k_t` of the drawn proof
blocks. -/
theorem size_proved_selectSecond {sample ns ks : Array ℕ} {src : ByteArray} {draw : Json} {read : ℕ}
    (h : Select.selectSecond sample ns ks src = .ok (draw, read)) :
    (Draw.proved draw).size = sample.size * ks.toList.sum := by
  obtain ⟨hq, parsed, units, hp, hu, hdu, hdc⟩ := selectSecond_units h
  obtain ⟨ws, rfl, hmap⟩ := flockLaw_second sample ns ks _ parsed hp
  rw [size_proved_of_units hdu hdc, drawUnits_work_size hu]
  have hk : ws.toList.map (·.k) = (cellPairs sample.size ns.size).map fun z => (cellOf sample.size ns ks z).2 := by
    have := congrArg (List.map Prod.snd) hmap
    simpa only [List.map_map, Function.comp_def] using this
  rw [hk, cellPairs, sum_cells_k, ← secondProblem_size hq, sum_range_getD]

end Properties.Spot.SecurityProofs
