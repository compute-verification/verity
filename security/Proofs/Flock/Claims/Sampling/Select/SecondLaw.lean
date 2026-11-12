module

public import Proofs.Flock.Claims.Sampling.Select.SecondStrata
public import Proofs.Flock.Bridge.QsRows.FirstReads
import Spot.Draw
import Proofs.Flock.Claims.Sampling.Select.Misses
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# The second stage's sampling design on its cells, and the bytes its strata read first

`secondLaw` is a work sampling design at `k = 1` over one stratum per cell, each of unit work and floor `k_t`. Where
`secondProblem` passes, each cell samples `1 ≤ k_t ≤ n_t` of its `n_t` proof blocks, and the work rule gives each cell
exactly its floor, so what `Draw.flockLaw` reads from it is a sampling design on the cells (`Flock.Bridge.LawOn`) that
samples `|sample| · Σ_t k_t` proof blocks.

* `secondProblem_counts`: `secondProblem` passing gives `1 ≤ k_t ≤ n_t` for each template.
* `cellSizes`, `cellSizes_sum`: the cells' sizes in the strata's order, `|sample| · Σ_t n_t` in all.
* `lawOn_second`: the parsed sampling design is `LawOn` the cells, sampling `|sample| · Σ_t k_t`.
* `secondRead`, `firstRead_second`: the bytes the strata read first, `|sample| · Σ_t (2·k_t·width n_t + 64)`.
* `drawUnits_second_error_le`: on a uniform stream that holds those bytes, the second stage's units are not sampled
  with probability at most `|sample| · Σ_t k_t · 2⁻⁶⁴` (`Flock.Bridge.units_error_le`).
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open FlockSoundness FlockSoundness.Game Finset
open scoped ENNReal

/-- A second stage `secondProblem` passes samples `1 ≤ k_t ≤ n_t` of each template's cell. -/
theorem secondProblem_counts {sample ns ks : Array ℕ} (h : Select.secondProblem sample ns ks = none) {t : ℕ}
    (ht : t < ns.size) : 1 ≤ ks.getD t 0 ∧ ks.getD t 0 ≤ ns.getD t 0 := by
  have hsz := secondProblem_size h
  have hall : ∀ k n, (k, n) ∈ ks.toList.zip ns.toList → 1 ≤ k ∧ k ≤ n := by
    unfold Select.secondProblem at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · cases h
          · rename_i hk
            simpa using hk
  have hlen : t < (ks.toList.zip ns.toList).length := by
    rw [List.length_zip, Array.length_toList, Array.length_toList, hsz]
    omega
  have hm := List.getElem_mem hlen
  rw [List.getElem_zip] at hm
  have hkt : ks.getD t 0 = ks.toList[t]'(by rw [Array.length_toList]; omega) := by
    simp [Array.getD, hsz, ht]
  have hnt : ns.getD t 0 = ns.toList[t]'(by rw [Array.length_toList]; omega) := by
    simp [Array.getD, ht]
  rw [hkt, hnt]
  exact hall _ _ hm

/-- The sizes of `secondLaw`'s strata, in their order: template by template, `q` cells of `n_t` proof blocks. -/
def cellSizes (q : ℕ) (ns : Array ℕ) : List ℕ := (List.range ns.size).flatMap fun t => List.replicate q (ns.getD t 0)

theorem cellSizes_take (q : ℕ) (ns : Array ℕ) : ∀ T : ℕ,
    ((List.range T).flatMap fun t => List.replicate q (ns.getD t 0)).sum = cellOff q ns T
  | 0 => by simp [cellOff]
  | T + 1 => by
    rw [List.range_succ, List.flatMap_append, List.sum_append, cellSizes_take q ns T, List.flatMap_singleton,
      cellOff_succ, List.sum_replicate_nat]

/-- The cells hold the interior's `q · Σ_t n_t` proof blocks. -/
theorem cellSizes_sum (q : ℕ) (ns : Array ℕ) : (cellSizes q ns).sum = q * ns.toList.sum := by
  rw [cellSizes, cellSizes_take, cellOff_size]

/-- A cell's stratum holds its template's `n_t` proof blocks. -/
theorem cellStratum_size (sample ns ks : Array ℕ) (z : ℕ × ℕ) : (cellStratum sample ns ks z).size = ns.getD z.1 0 := by
  unfold Flock.Draw.Stratum.size
  simp [cellStratum, Nat.succ_mul]
  omega

/-- The work rule at `k = 1` and unit work gives a stratum of `n` units its floor `k`, where `1 ≤ k ≤ n` and the
population's work is at least `n`. -/
theorem workK_unit {n W k : ℕ} (hk : 1 ≤ k) (hkn : k ≤ n) (hW : n ≤ W) : Flock.Draw.workK 1 1 n W k = k := by
  unfold Flock.Draw.workK
  have hdiv : (1 * 1 * n + W - 1) / W = 1 := Nat.div_eq_of_lt_le (by omega) (by omega)
  rw [hdiv]
  omega

/-- A cell `(t, i)` of the cells' list is one of a template `t < T`. -/
theorem cellPairs_fst {q T : ℕ} {z : ℕ × ℕ} (hz : z ∈ cellPairs q T) : z.1 < T := by
  unfold cellPairs at hz
  obtain ⟨t, ht, hz⟩ := List.mem_flatMap.1 hz
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hz
  exact List.mem_range.1 ht

/-- **`secondLaw` is a sampling design on its cells that samples `|sample| · Σ_t k_t` proof blocks**: where
`secondProblem` passes, what `Draw.flockLaw` reads from it has the cells as its strata, in order and tiling the
interior, each stratum's count the work rule's. -/
theorem lawOn_second {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) {m : ℕ}
    {parsed : Flock.Draw.Law} (h : Draw.flockLaw (Select.secondLaw sample ns ks) m = .ok parsed) :
    Flock.Bridge.LawOn parsed (cellSizes sample.size ns) (sample.size * ks.toList.sum) := by
  obtain ⟨ws, rfl, hws⟩ := flockLaw_second_strata sample ns ks m parsed h
  have hmap : ws.toList.map (fun s => (s.units, s.k)) =
      (cellPairs sample.size ns.size).map (cellOf sample.size ns ks) := by
    rw [hws, List.map_map]
    exact List.map_congr_left fun _ _ => rfl
  show Flock.Bridge.StrataOn (ws.toList.map (·.toStratum)) (cellSizes sample.size ns) ∧
    (∀ s ∈ ws.toList, s.k = Flock.Draw.workK 1 s.work s.size (Flock.Draw.totalWork ws) s.floor) ∧
    (ws.toList.map (·.k)).sum = sample.size * ks.toList.sum
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · have hpos := positions_second hmap
    rw [Array.toList_map] at hpos
    rw [hpos, cellSizes_sum]
  · have h1 := congrArg (List.map Prod.fst) (sizes_second hmap)
    have h2 : ((List.range ns.size).flatMap fun t => List.replicate sample.size (ns.getD t 0, ks.getD t 0)).map
        Prod.fst = cellSizes sample.size ns := by
      rw [cellSizes, List.map_flatMap]
      simp only [List.map_replicate]
    rw [h2] at h1
    rw [← h1]
    simp only [Array.toList_map, List.map_map, Function.comp_def]
  · intro s hs
    rw [hws] at hs
    obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hs
    obtain ⟨hk1, hkn⟩ := secondProblem_counts hp (cellPairs_fst hz)
    have hW : (cellStratum sample ns ks z).size * (cellStratum sample ns ks z).work ≤ Flock.Draw.totalWork ws := by
      rw [totalWork_eq]
      exact List.le_sum_of_mem (List.mem_map.2 ⟨_, hws ▸ List.mem_map_of_mem hz, rfl⟩)
    rw [cellStratum_size] at hW ⊢
    have hW' : ns.getD z.1 0 ≤ Flock.Draw.totalWork ws := by
      have h1 : (cellStratum sample ns ks z).work = 1 := rfl
      rw [h1, Nat.mul_one] at hW
      exact hW
    exact (workK_unit hk1 hkn hW').symm
  · have hk : ws.toList.map (·.k) = (cellPairs sample.size ns.size).map fun z => (cellOf sample.size ns ks z).2 := by
      have := congrArg (List.map Prod.snd) hmap
      simpa only [List.map_map, Function.comp_def] using this
    rw [hk, cellPairs, sum_cells_k, ← secondProblem_size hp, sum_range_getD]

/-- **The bytes the second stage's strata read first, in all**: each cell's `2·k_t·width n_t + 64`, a replay sample of
`q` blocks over cells of `ns` proof blocks sampling `ks`. -/
def secondRead (q : ℕ) (ns ks : Array ℕ) : ℕ :=
  q * ((List.range ns.size).map fun t => 2 * ks.getD t 0 * Flock.Draw.width (ns.getD t 0) + 64).sum

/-- The strata's first reads by index are the strata's. -/
theorem range_map_need (St : Array Flock.Draw.Stratum) :
    (List.range' 0 St.size).map (fun i => St[i]!.need) = St.toList.map (·.need) := by
  refine List.ext_getElem (by simp) fun i h₁ _ => ?_
  have hi : i < St.size := by simpa using h₁
  simp [getElem!_pos St i hi]

/-- The stages of strata `St` read their `need`s first. -/
theorem stages_need (St : Array Flock.Draw.Stratum) :
    (((List.range' 0 St.size).map fun i => stage St[i]! fun _ => True).map (·.need)).sum =
      (St.toList.map (·.need)).sum := by
  rw [List.map_map, ← range_map_need]
  rfl

/-- A work sampling design's stages read its strata's `need`s first. -/
theorem firstRead_work (K n : ℕ) (ws : Array Flock.Draw.WorkStratum) :
    Flock.Bridge.firstRead (.work K ws) n = ((ws.toList.map (·.toStratum)).map (·.need)).sum := by
  have h := stages_need (ws.map (·.toStratum))
  rw [Array.toList_map] at h
  exact h

/-- **The second stage's strata read `secondRead` bytes first.** -/
theorem firstRead_second {sample ns ks : Array ℕ} {m : ℕ} {parsed : Flock.Draw.Law}
    (h : Draw.flockLaw (Select.secondLaw sample ns ks) m = .ok parsed) :
    Flock.Bridge.firstRead parsed m = secondRead sample.size ns ks := by
  obtain ⟨ws, rfl, hws⟩ := flockLaw_second_strata sample ns ks m parsed h
  have hf : (((cellPairs sample.size ns.size).map (cellStratum sample ns ks)).map (·.toStratum)).map (·.need) =
      (cellPairs sample.size ns.size).map
        fun z : ℕ × ℕ => 2 * ks.getD z.1 0 * Flock.Draw.width (ns.getD z.1 0) + 64 := by
    rw [List.map_map, List.map_map]
    refine List.map_congr_left fun z _ => ?_
    show 2 * (cellStratum sample ns ks z).k * Flock.Draw.width (cellStratum sample ns ks z).size + 64 = _
    rw [cellStratum_size]
    rfl
  rw [firstRead_work, hws, hf, cellPairs,
    sum_cells_fst sample.size (fun t => 2 * ks.getD t 0 * Flock.Draw.width (ns.getD t 0) + 64)]
  rfl

/-- **The second stage's units are not sampled with probability at most `|sample| · Σ_t k_t · 2⁻⁶⁴`**, on a uniform
stream of at least `secondRead` bytes: where `secondProblem` passes, `Draw.drawUnits` under what `Draw.flockLaw` reads
from `secondLaw` returns no sample only where some cell's first read samples nothing. -/
theorem drawUnits_second_error_le {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none)
    {parsed : Flock.Draw.Law}
    (h : Draw.flockLaw (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) = .ok parsed) (len : ℕ)
    (hlen : secondRead sample.size ns ks ≤ len) :
    prCoin (fun f : Fin len → Fin 256 => ¬ ∃ units read,
      Draw.drawUnits parsed (sample.size * ns.toList.sum) (bytesOf f) = .ok (units, read)) ≤
      ((sample.size * ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ :=
  Flock.Bridge.units_error_le (lawOn_second hp h) (cellSizes_sum _ _) len (by rw [firstRead_second h]; exact hlen)

end Properties.Spot.SecurityProofs
