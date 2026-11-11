module

public import Proofs.Flock.Claims.Sampling.Select.DrawLaw
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
public import Proofs.Flock.Soundness.ExecStrataOf

/-!
# The deployed sample at the registration's strata

`SampledKSLiveV5` samples with the deployed sample (`Spot.Draw.drawUnits`, `Flock.Draw.drawWith` on the
challenger's coins read in order as one stream) under a Flock law over the registration's strata, and its `δ` is that
sampling design's own `P_a`. This file holds what the statement reads and how the sample meets it:

* `StrataOn St ns`: the strata `St`'s ranges, in order, are the slots `0, …, Σ ns − 1`, at sizes `ns`, so each stratum
  is one run of consecutive slots;
* `LawOn law ns kd`: `law` is `subset kd`, or `stratified K` or `work K` over strata at `ns`, each stratum's `k` the
  rule's (`stratumK`, `workK`) and `kd` their sum. A Bernoulli law is none (`NotBernoulli`);
* `pa_stratified`, `pa_work`: such a sampling design's claims design (`oneStageOf`) misses `B` with `strataPa` of its strata's sizes
  and samples; `pa_empty_of_lawOn`: it misses nothing with probability `1`;
* `wellFormed_of_lawOn`: such a sampling design, where it is the one the verifier's agreed law gives (`deriveLaw`, else
  `flockLaw` of `derive`'s law object), is well formed over `ns` (`OneStage.WellFormed`), by the code's own refusals;
* `drawUnits_miss_le`: the deployed sample misses a set `U` of slots with probability at most `P_a(U)` (`one_bound`,
  `strata_bound`);
* `execSel src law pop`: the deployed sample as a sampling design on the verifier's bytes `src`, a sample that runs out of them sampling
  every unit; `execSel_escape_le`: under A3 it escapes `W` with probability at most `P_a` of `W`'s slots;
* `lawSized_size`: a sample file that decodes to the verifier's sample under a sampling design that is `subset kd`, or whose strata
  hold each slot once and sample `kd` units in all (`LawSized`, as a sampling design on strata is, `lawSized_of_lawOn`), has `kd`
  units, at any population (`ofJson_shape`, and `UnitDraw.ofJson`'s per-stratum count, `ofJson_count`,
  `ofJson_count_work`, in strata whose ranges it checks tile the population, `strataProblem_none`,
  `workProblem_chain`); `lawOn_size` at a sampling design on strata;
* `drawUnits_size`: the deployed sample under such a sampling design that ends holds `kd` units, at any population (`subset_size`,
  `stratifiedWith_size`);
* `All`: the rule's law is `subset n` over the `n` units (`Select.Rule.lawOver`), on the strata at `kd = n`
  (`lawOn_all`), and under it the deployed sample holds every unit whatever its bytes (`drawUnits_all`, `execSel_all`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Flock Exec ZkExec Finset FlockSoundness.Refine

set_option maxHeartbeats 2000000 in
/-- **`UnitDraw.ofJson`'s per-stratum count, at a work sampling design** (`ofJson_count`): a sample it decodes under a work sampling design holds
exactly `k_s` of its units in each stratum `s`'s ranges. -/
theorem ofJson_count_work {j : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) {K : ℕ}
    {strata : Array Draw.WorkStratum} (hl : D.law = .work K strata) (s : Draw.WorkStratum) (hs : s ∈ strata) :
    D.units.foldl (fun acc u => if (s.units.any fun x => decide (x.1 ≤ u) && decide (u < x.2)) = true then acc + 1
      else acc) 0 = s.k := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : V Draw.Law) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    simp only [reduceCtorEq, Draw.Law.work.injEq] at hl
  all_goals
    obtain ⟨rfl, rfl⟩ := hl
    have hf := ‹forIn _ _ _ = Except.ok _›
    rw [← Array.forIn_toList] at hf
    simpa using forIn_throw_ok _ _ _ _ hf s (by simpa using hs)

end FlockSoundness.Discharge.ZkLink

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game Finset
open scoped ENNReal

/-- **Strata at `ns`**: the strata `St`'s ranges, in order, are the slots `0, …, Σ ns − 1`, and their sizes are `ns`. -/
def StrataOn (St : List Flock.Draw.Stratum) (ns : List ℕ) : Prop :=
  St.flatMap (fun s => Properties.Spot.SecurityProofs.expand s.units.toList) = List.range ns.sum ∧
    St.map (·.size) = ns

/-- **A sampling design at the strata `ns` that samples `kd` units**: `subset kd`, or `stratified K` or `work K` over strata at `ns`
(`StrataOn`), each stratum's `k` the rule's (`stratumK`, `workK`) and `kd` their sum. A Bernoulli law is none. -/
def LawOn (law : Flock.Draw.Law) (ns : List ℕ) (kd : ℕ) : Prop :=
  match law with
  | .subset k => k = kd ∧ kd ≤ ns.sum
  | .bernoulli _ _ => False
  | .stratified K St => StrataOn St.toList ns ∧ K ≤ ns.sum ∧ (∀ x ∈ ns, 1 ≤ x) ∧
      (∀ s ∈ St.toList, s.k = Flock.Draw.stratumK K s.size ns.sum) ∧ (St.toList.map (·.k)).sum = kd
  | .work K ws => StrataOn (ws.toList.map (·.toStratum)) ns ∧
      (∀ s ∈ ws.toList, s.k = Flock.Draw.workK K s.work s.size (Flock.Draw.totalWork ws) s.floor) ∧
      (ws.toList.map (·.k)).sum = kd

/-- **No Bernoulli law**: `law` samples a count its strata or its `k` fix, not one the coins set, as `LawOn` and
`LawSized` require. -/
def NotBernoulli (law : Flock.Draw.Law) : Prop := ∀ a b, law ≠ .bernoulli a b

/-- A stratified sampling design's design misses `B` with `strataPa` of its strata's sizes and samples, when its strata have sizes `ns`
and each stratum's `k` is the rule's. -/
theorem pa_stratified {K : ℕ} {St : Array Flock.Draw.Stratum} {ns : List ℕ} (hsz : St.toList.map (·.size) = ns)
    (hk : ∀ s ∈ St.toList, s.k = Flock.Draw.stratumK K s.size ns.sum) (B : Finset ℕ) :
    (Properties.Spot.oneStageOf (.stratified K St)).Pa ns B =
      Claims.Sampling.strataPa (St.toList.map fun s => (s.size, s.k)) 0 B := by
  subst hsz
  simp only [Properties.Spot.oneStageOf, Claims.Sampling.OneStage.Pa, List.map_map]
  congr 1
  refine List.map_congr_left fun s hs => ?_
  rw [Function.comp_apply, hk s hs]
  rfl

/-- A work sampling design's design misses `B` with `strataPa` of its strata's sizes and samples, when its strata have sizes `ns` and
each stratum's `k` is the work rule's. -/
theorem pa_work {K : ℕ} {ws : Array Flock.Draw.WorkStratum} {ns : List ℕ}
    (hsz : (ws.toList.map (·.toStratum)).map (·.size) = ns)
    (hk : ∀ s ∈ ws.toList, s.k = Flock.Draw.workK K s.work s.size (Flock.Draw.totalWork ws) s.floor) (B : Finset ℕ) :
    (Properties.Spot.oneStageOf (.work K ws)).Pa ns B =
      Claims.Sampling.strataPa ((ws.map (·.toStratum)).toList.map fun s => (s.size, s.k)) 0 B := by
  subst hsz
  have hW : ((((ws.toList.map (·.toStratum)).map (·.size)).zip (ws.toList.map (·.work))).map
      (fun p => p.1 * p.2)).sum = Flock.Draw.totalWork ws := by
    rw [Properties.Spot.SecurityProofs.totalWork_eq, List.map_map, List.zip_map', List.map_map]
    rfl
  simp only [Properties.Spot.oneStageOf, Claims.Sampling.OneStage.Pa]
  rw [hW, Array.toList_map, List.map_map, List.map_map, List.zip_map', List.zip_map', List.map_map]
  congr 1
  refine List.map_congr_left fun s hs => ?_
  simp only [Function.comp_apply]
  rw [hk s hs]
  rfl

theorem stratumK_le {K n N : ℕ} (hK : K ≤ N) (hn : 1 ≤ n) (hnN : n ≤ N) : Flock.Draw.stratumK K n N ≤ n := by
  refine max_le hn (Nat.lt_succ_iff.1 ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_))
  have : K * n ≤ N * n := Nat.mul_le_mul_right n hK
  nlinarith

/-- **Such a sampling design misses nothing with probability `1`**: each stratum samples at most its size. -/
theorem pa_empty_of_lawOn {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (h : LawOn law ns kd) :
    (Properties.Spot.oneStageOf law).Pa ns ∅ = 1 := by
  cases law with
  | subset k =>
    obtain ⟨rfl, hk⟩ := h
    simp only [Properties.Spot.oneStageOf, Claims.Sampling.OneStage.Pa, Finset.card_empty, Nat.sub_zero]
    exact div_self (Nat.cast_ne_zero.2 (Nat.choose_pos hk).ne')
  | bernoulli a b => exact h.elim
  | stratified K St =>
    obtain ⟨⟨-, hsz⟩, hK, hn, hk, -⟩ := h
    rw [pa_stratified hsz hk]
    refine Claims.Sampling.strataPa_empty _ 0 fun p hp => ?_
    obtain ⟨s, hs, rfl⟩ := List.mem_map.1 hp
    have hm : s.size ∈ ns := hsz ▸ List.mem_map_of_mem hs
    rw [hk s hs]
    exact stratumK_le hK (hn _ hm) (List.le_sum_of_mem hm)
  | work K ws =>
    obtain ⟨⟨-, hsz⟩, hk, -⟩ := h
    rw [pa_work hsz hk]
    refine Claims.Sampling.strataPa_empty _ 0 fun p hp => ?_
    obtain ⟨s, hs, rfl⟩ := List.mem_map.1 hp
    rw [Array.toList_map] at hs
    obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hs
    show w.k ≤ w.size
    rw [hk w hw]
    exact min_le_left _ _

/-- `UnitDraw.ofJson`'s range test is membership in the ranges' slots. -/
theorem any_iff_mem_expand (rs : Array (ℕ × ℕ)) (u : ℕ) :
    (rs.any fun x => decide (x.1 ≤ u) && decide (u < x.2)) = true ↔
      u ∈ Properties.Spot.SecurityProofs.expand rs.toList := by
  simp only [Array.any_eq_true', Properties.Spot.SecurityProofs.expand, List.mem_flatMap,
    Array.mem_toList_iff, List.mem_range'_1, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨x, hx, h1, h2⟩; exact ⟨x, hx, h1, by omega⟩
  · rintro ⟨x, hx, h1, h2⟩; exact ⟨x, hx, h1, by omega⟩

/-- Units in strata whose slots hold each unit once, counted stratum by stratum, are all the units. -/
theorem sum_count_strata {α : Type} (E : α → List ℕ) :
    ∀ (St : List α) (l : List ℕ), (St.flatMap E).Nodup → (∀ u ∈ l, u ∈ St.flatMap E) →
      (St.map fun s => (l.filter fun u => decide (u ∈ E s)).length).sum = l.length
  | [], l, _, hl => by
    obtain rfl : l = [] := List.eq_nil_iff_forall_not_mem.2 fun u hu => by simpa using hl u hu
    rfl
  | s :: St, l, hnd, hl => by
    rw [List.flatMap_cons, List.nodup_append] at hnd
    obtain ⟨-, hnd', hdis⟩ := hnd
    have hrest : ∀ s' ∈ St, (l.filter fun u => decide (u ∈ E s')) =
        ((l.filter fun u => !decide (u ∈ E s)).filter fun u => decide (u ∈ E s')) := by
      intro s' hs'
      rw [List.filter_filter]
      refine List.filter_congr fun u _ => ?_
      by_cases h : u ∈ E s'
      · have : u ∉ E s := fun h' => hdis u h' u (List.mem_flatMap.2 ⟨s', hs', h⟩) rfl
        simp [h, this]
      · simp [h]
    have hmap : (St.map fun s' => (l.filter fun u => decide (u ∈ E s')).length) =
        St.map fun s' => (((l.filter fun u => !decide (u ∈ E s)).filter fun u => decide (u ∈ E s')).length) :=
      List.map_congr_left fun s' hs' => by rw [hrest s' hs']
    rw [List.map_cons, List.sum_cons, hmap, sum_count_strata E St _ hnd' fun u hu => ?_]
    · exact (List.length_eq_length_filter_add _).symm
    · obtain ⟨hul, hus⟩ := List.mem_filter.1 hu
      rcases List.mem_append.1 (List.flatMap_cons ▸ hl u hul) with h | h
      · simp [h] at hus
      · exact h

/-- `strata_bound` at strata whose ranges are the slots `0, …, n − 1` in order: the sample misses the slots `U` themselves. -/
theorem strata_bound_on (law : Flock.Draw.Law) (St : Array Flock.Draw.Stratum)
    (hSt : (∃ k, law = .stratified k St) ∨ ∃ k ws, law = .work k ws ∧ St = ws.map (·.toStratum)) (n : ℕ)
    (hon : St.toList.flatMap (fun s => Properties.Spot.SecurityProofs.expand s.units.toList) = List.range n)
    (U : Finset ℕ) (hU : U ⊆ Finset.range n) (L : ℕ) :
    Claims.Pr (fun src : Fin L → Fin 256 => ∃ units read,
        Spot.Draw.drawUnits law n (Properties.Spot.bytesOf src) = .ok (units, read) ∧
          ∀ u ∈ U, u ∉ units) ≤
      Claims.Sampling.strataPa (St.toList.map fun s => (s.size, s.k)) 0 U := by
  refine le_trans (Claims.Pr_mono fun src ⟨units, read, hu, hm⟩ => ⟨units, read, hu, fun v hv => ?_⟩)
    (Properties.Spot.SecurityProofs.strata_bound law St hSt n U L)
  have hv' : v < (List.range n).length := by rw [List.length_range]; exact Finset.mem_range.1 (hU hv)
  rw [hon, List.getD_eq_getElem _ _ hv', List.getElem_range]
  exact hm v hv

/-- **The deployed sample misses `U`** with probability at most its sampling design's `P_a(U)` on the strata `ns` it is on. -/
theorem drawUnits_miss_le {law : Flock.Draw.Law} {ns : List ℕ} {kd n : ℕ} (hlaw : LawOn law ns kd) (hns : ns.sum = n)
    (U : Finset ℕ) (hU : U ⊆ Finset.range n) (L : ℕ) :
    Claims.Pr (fun src : Fin L → Fin 256 => ∃ units read,
        Spot.Draw.drawUnits law n (Properties.Spot.bytesOf src) = .ok (units, read) ∧
          ∀ u ∈ U, u ∉ units) ≤
      (Properties.Spot.oneStageOf law).Pa ns U := by
  cases law with
  | subset k =>
    obtain ⟨rfl, hk⟩ := hlaw
    refine ((Properties.Spot.SecurityProofs.one_bound n U hU L).1 _ (by rw [← hns]; exact hk)).trans_eq ?_
    simp only [Properties.Spot.oneStageOf, Claims.Sampling.OneStage.Pa, hns]
  | bernoulli a b => exact hlaw.elim
  | stratified K St =>
    obtain ⟨⟨hon, hsz⟩, -, -, hk, -⟩ := hlaw
    rw [pa_stratified hsz hk]
    exact strata_bound_on _ St (Or.inl ⟨K, rfl⟩) n (by rw [← hns]; exact hon) U hU L
  | work K ws =>
    obtain ⟨⟨hon, hsz⟩, hk, -⟩ := hlaw
    rw [pa_work hsz hk]
    refine strata_bound_on _ _ (Or.inr ⟨K, ws, rfl, rfl⟩) n ?_ U hU L
    rw [Array.toList_map, ← hns]
    exact hon

/-- **The deployed sample as a sampling design**: `Spot.Draw.drawUnits` under `law` over `[0, pop)` on the source `t`'s bytes,
read in order as one stream; a sample that runs out of them samples every unit. -/
def execSel {n : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ} (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256)
    (law : Flock.Draw.Law) (pop : ℕ) : Audit.Law n where
  Ω := Ω
  draw ω := match Spot.Draw.drawUnits law pop (Properties.Spot.bytesOf (t ω 0)) with
    | .ok r => univ.filter fun u => u.val ∈ r.1
    | .error _ => univ

/-- **The deployed sample escapes `W`** with probability at most its sampling design's `P_a` of `W`'s slots, on a uniform byte source
(A3), at a sampling design on strata `ns` of the `n` units. -/
theorem execSel_escape_le {n : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ}
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) {law : Flock.Draw.Law}
    {ns : List ℕ} {kd : ℕ} (hlaw : LawOn law ns kd) (hns : ns.sum = n) (W : Finset (Fin n)) :
    (execSel (n := n) t law n).escape W ≤
      ENNReal.ofReal ((Properties.Spot.oneStageOf law).Pa ns (W.map Fin.valEmbedding)) := by
  classical
  rcases W.eq_empty_or_nonempty with rfl | ⟨b, hb⟩
  · rw [Finset.map_empty, pa_empty_of_lawOn hlaw, ENNReal.ofReal_one]
    exact Audit.Law.escape_le_one _ _
  have hU : W.map Fin.valEmbedding ⊆ Finset.range n := fun u hu => by
    obtain ⟨v, -, rfl⟩ := Finset.mem_map.1 hu
    exact Finset.mem_range.2 v.isLt
  let E : (i : Fin 1) → (Fin (L i) → Fin 256) → Prop := fun _ g => ∃ units read,
    Spot.Draw.drawUnits law n (Properties.Spot.bytesOf g) = .ok (units, read) ∧
      ∀ u ∈ W.map Fin.valEmbedding, u ∉ units
  have hfin : prCoin (E 0) ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (Audit.DrawExec.prCoin_le_one _)
  calc (execSel (n := n) t law n).escape W ≤ prCoin (fun ω => E 0 (t ω 0)) := by
        refine prCoin_mono fun ω hd => ?_
        change Disjoint (match Spot.Draw.drawUnits law n (Properties.Spot.bytesOf (t ω 0)) with
          | .ok r => univ.filter fun u => u.val ∈ r.1
          | .error _ => univ) W at hd
        split at hd
        · rename_i r hdu
          refine ⟨r.1, r.2, hdu, fun u hu hun => ?_⟩
          obtain ⟨v, hv, rfl⟩ := Finset.mem_map.1 hu
          exact Finset.disjoint_left.1 hd (Finset.mem_filter.2 ⟨Finset.mem_univ _, hun⟩) hv
        · exact absurd hb (Finset.disjoint_left.1 hd (Finset.mem_univ b))
    _ = prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => E 0 (f 0)) := ht fun f => E 0 (f 0)
    _ ≤ prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => ∀ s ∈ (univ : Finset (Fin 1)), E s (f s)) :=
        prCoin_mono fun f h s _ => by obtain rfl : s = 0 := Subsingleton.elim _ _; exact h
    _ = ∏ s : Fin 1, prCoin (E s) := prCoin_pi_forall (Ω := fun s => Fin (L s) → Fin 256) univ E
    _ = prCoin (E 0) := Fin.prod_univ_one _
    _ = ENNReal.ofReal (Claims.Pr (E 0)) := by rw [Claims.Sampling.Pr_eq_toReal_prCoin, ENNReal.ofReal_toReal hfin]
    _ ≤ _ := ENNReal.ofReal_le_ofReal (drawUnits_miss_le hlaw hns _ hU (L 0))

/-- **`All`'s law is on the strata**: `subset n` over the `n` units, sampling all of them. -/
theorem lawOn_all (ns : List ℕ) : LawOn (.subset ns.sum) ns ns.sum := ⟨rfl, le_rfl⟩

/-- **Under `All` the deployed sample holds every unit**: a sample of `subset n` over `[0, n)` that ends is the image of
`[0, n)` under a permutation that keeps it (`subset_spec`), so all of `[0, n)`, whatever the bytes. -/
theorem drawUnits_all (n : ℕ) (src : ByteArray) (r : Array ℕ × ℕ)
    (h : Spot.Draw.drawUnits (.subset n) n src = .ok r) (u : ℕ) (hu : u < n) : u ∈ r.1 := by
  have hs : Properties.Spot.SecurityProofs.SeqMiss
      [⟨2 * n * Flock.Draw.width n + 64, Flock.Draw.subset n n, fun D => ∀ x ∈ D, x ∈ r.1⟩] src 0 :=
    Properties.Spot.SecurityProofs.drawUnits_seq (.subset n) n src r.1 r.2 h
  obtain ⟨j, D, ⟨-, -, hD, -⟩, hm, -⟩ := hs
  change Flock.Draw.subset n n _ = some D at hD
  change ∀ x ∈ D, x ∈ r.1 at hm
  obtain ⟨π, -, hπ, hmem⟩ := Discharge.Exec.subset_spec hD
  have himg : (Finset.range n).image π = Finset.range n :=
    Finset.eq_of_subset_of_card_le (fun x hx => by
        obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hx
        exact Finset.mem_range.2 (hπ i (Finset.mem_range.1 hi)))
      (by rw [Finset.card_image_of_injective _ π.injective])
  have hu' : u ∈ (Finset.range n).image π := by rw [himg]; exact Finset.mem_range.2 hu
  obtain ⟨i, hi, hiu⟩ := Finset.mem_image.1 hu'
  exact hm u ((hmem u).2 ⟨i, Finset.mem_range.1 hi, hiu⟩)

/-- **Under `All` the verifier samples every unit, for any coins**: `execSel` at `subset n` over the `n` units. -/
theorem execSel_all {n : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ}
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ω : Ω) : (execSel (n := n) t (.subset n) n).draw ω = univ := by
  change (match Spot.Draw.drawUnits (.subset n) n (Properties.Spot.bytesOf (t ω 0)) with
    | .ok r => univ.filter fun u : Fin n => u.val ∈ r.1
    | .error _ => univ) = univ
  split
  · rename_i r hr
    exact Finset.filter_true_of_mem fun u _ => drawUnits_all n _ r hr u.val u.isLt
  · rfl

/-- A unit in one of a stratum's ranges is in its slots. -/
theorem mem_expand_of_mem {rs : List (ℕ × ℕ)} {r : ℕ × ℕ} (hr : r ∈ rs) {u : ℕ} (h1 : r.1 ≤ u) (h2 : u < r.2) :
    u ∈ Properties.Spot.SecurityProofs.expand rs := by
  simp only [Properties.Spot.SecurityProofs.expand, List.mem_flatMap, List.mem_range'_1]
  exact ⟨r, hr, h1, by omega⟩

/-- **A sampling design whose sample files hold `kd` units** (`lawSized_size`): `subset kd`, or a stratified or work sampling design whose
strata's slots hold each unit once and whose strata's `k`s sum to `kd`. A Bernoulli law is none. -/
def LawSized (law : Flock.Draw.Law) (kd : ℕ) : Prop :=
  match law with
  | .subset k => k = kd
  | .bernoulli _ _ => False
  | .stratified _ St =>
    (St.toList.flatMap fun s => Properties.Spot.SecurityProofs.expand s.units.toList).Nodup ∧
      (St.toList.map (·.k)).sum = kd
  | .work _ ws =>
    ((ws.toList.map (·.toStratum)).flatMap fun s =>
        Properties.Spot.SecurityProofs.expand s.units.toList).Nodup ∧
      (ws.toList.map (·.k)).sum = kd

/-- A sampling design on strata at `ns` is sized: its strata's slots are `0, …, Σ ns − 1`, each once. -/
theorem lawSized_of_lawOn {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (h : LawOn law ns kd) : LawSized law kd := by
  cases law with
  | subset k => exact h.1
  | bernoulli a b => exact h.elim
  | stratified K St =>
    obtain ⟨⟨hon, -⟩, -, -, -, hsum⟩ := h
    exact ⟨by rw [hon]; exact List.nodup_range, hsum⟩
  | work K ws =>
    obtain ⟨⟨hon, -⟩, -, hsum⟩ := h
    exact ⟨by rw [hon]; exact List.nodup_range, hsum⟩

/-- **A sample file of the verifier's sample under a sized law decodes only to `kd` units**, at any population: a subset
sample has its `k` (`ofJson_shape`), and a stratified or work sample `k_s` units in each stratum (`ofJson_count`,
`ofJson_count_work`), whose slots hold each unit once and whose ranges hold each of its units, since `UnitDraw.ofJson`
checks that they tile the sample's population (`strataProblem_none`, `workProblem_chain`). -/
theorem lawSized_size {n : ℕ} {law : Flock.Draw.Law} {kd : ℕ} (hlaw : LawSized law kd)
    {dj : Finset (Fin n) → Lean.Json} (hdj : Discharge.ZkLink.DrawFileZK dj law n) (S : Finset (Fin n))
    (D : Flock.Draw.UnitDraw) (hD : Flock.Draw.UnitDraw.ofJson (dj S) = .ok D) : D.units.size = kd := by
  obtain rfl := hdj S D hD
  have hlt : ∀ u ∈ (Discharge.ZkLink.drawOf law n S).units.toList, u < n := by
    intro u hu
    obtain ⟨v, -, rfl⟩ := (Discharge.ZkLink.drawOf_mem law n S u).1 (Array.mem_toList_iff.1 hu)
    exact v.isLt
  cases law with
  | subset k =>
    obtain rfl := hlaw
    exact (Discharge.Composed.ofJson_shape hD).2.2 _ rfl
  | bernoulli a b => exact hlaw.elim
  | stratified K St =>
    obtain ⟨hnd, hsum⟩ := hlaw
    have hch := (Audit.StrataExec.strataProblem_none (Audit.StrataExec.ofJson_stratified hD rfl).1).2.2.2
    have hin : ∀ u ∈ (Discharge.ZkLink.drawOf (.stratified K St) n S).units.toList,
        u ∈ St.toList.flatMap fun s => Properties.Spot.SecurityProofs.expand s.units.toList := by
      intro u hu
      obtain ⟨r, hr, h1, h2⟩ := Audit.StrataExec.chain_covers hch u (Nat.zero_le u) (hlt u hu)
      have hr' := (Audit.StrataExec.qsort_perm _ _).subset hr
      rw [Audit.StrataExec.foldl_units] at hr'
      obtain ⟨s, hs, hrs⟩ := List.mem_flatMap.1 hr'
      exact List.mem_flatMap.2 ⟨s, hs, mem_expand_of_mem hrs h1 h2⟩
    have hc : ∀ s ∈ St.toList, ((Discharge.ZkLink.drawOf (.stratified K St) n S).units.toList.filter fun u =>
        decide (u ∈ Properties.Spot.SecurityProofs.expand s.units.toList)).length = s.k := by
      intro s hs
      have h := Discharge.ZkLink.ofJson_count hD rfl s (Array.mem_toList_iff.1 hs)
      rw [← Array.foldl_toList, Discharge.ZkLink.foldl_count_filter, zero_add] at h
      rw [← h]
      congr 1
      refine List.filter_congr fun u _ => ?_
      rw [Bool.eq_iff_iff, decide_eq_true_iff]
      exact (any_iff_mem_expand s.units u).symm
    have hcount := sum_count_strata (fun s : Flock.Draw.Stratum =>
      Properties.Spot.SecurityProofs.expand s.units.toList) St.toList
      (Discharge.ZkLink.drawOf (.stratified K St) n S).units.toList hnd hin
    calc (Discharge.ZkLink.drawOf (.stratified K St) n S).units.size
        = (Discharge.ZkLink.drawOf (.stratified K St) n S).units.toList.length := Array.length_toList.symm
      _ = (St.toList.map fun s => ((Discharge.ZkLink.drawOf (.stratified K St) n S).units.toList.filter fun u =>
            decide (u ∈ Properties.Spot.SecurityProofs.expand s.units.toList)).length).sum := hcount.symm
      _ = (St.toList.map (·.k)).sum := congrArg List.sum (List.map_congr_left hc)
      _ = kd := hsum
  | work K ws =>
    obtain ⟨hnd, hsum⟩ := hlaw
    have hch := Audit.StrataExec.workProblem_chain (Audit.StrataExec.ofJson_work hD rfl)
    have hin : ∀ u ∈ (Discharge.ZkLink.drawOf (.work K ws) n S).units.toList,
        u ∈ (ws.toList.map (·.toStratum)).flatMap fun s =>
          Properties.Spot.SecurityProofs.expand s.units.toList := by
      intro u hu
      obtain ⟨r, hr, h1, h2⟩ := Audit.StrataExec.chain_covers hch u (Nat.zero_le u) (hlt u hu)
      have hr' := (Audit.StrataExec.qsort_perm _ _).subset hr
      rw [Audit.StrataExec.foldl_units_work] at hr'
      obtain ⟨w, hw, hrw⟩ := List.mem_flatMap.1 hr'
      exact List.mem_flatMap.2 ⟨w.toStratum, List.mem_map_of_mem hw, mem_expand_of_mem hrw h1 h2⟩
    have hc : ∀ s ∈ ws.toList, ((Discharge.ZkLink.drawOf (.work K ws) n S).units.toList.filter fun u =>
        decide (u ∈ Properties.Spot.SecurityProofs.expand s.units.toList)).length = s.k := by
      intro s hs
      have h := Discharge.ZkLink.ofJson_count_work hD rfl s (Array.mem_toList_iff.1 hs)
      rw [← Array.foldl_toList, Discharge.ZkLink.foldl_count_filter, zero_add] at h
      rw [← h]
      congr 1
      refine List.filter_congr fun u _ => ?_
      rw [Bool.eq_iff_iff, decide_eq_true_iff]
      exact (any_iff_mem_expand s.units u).symm
    have hcount := sum_count_strata (fun s : Flock.Draw.Stratum =>
      Properties.Spot.SecurityProofs.expand s.units.toList) (ws.toList.map (·.toStratum))
      (Discharge.ZkLink.drawOf (.work K ws) n S).units.toList hnd hin
    calc (Discharge.ZkLink.drawOf (.work K ws) n S).units.size
        = (Discharge.ZkLink.drawOf (.work K ws) n S).units.toList.length := Array.length_toList.symm
      _ = ((ws.toList.map (·.toStratum)).map fun s => ((Discharge.ZkLink.drawOf (.work K ws) n S).units.toList.filter
            fun u => decide (u ∈ Properties.Spot.SecurityProofs.expand s.units.toList)).length).sum :=
          hcount.symm
      _ = (ws.toList.map fun s => ((Discharge.ZkLink.drawOf (.work K ws) n S).units.toList.filter fun u =>
            decide (u ∈ Properties.Spot.SecurityProofs.expand s.units.toList)).length).sum := by
          rw [List.map_map]
          rfl
      _ = (ws.toList.map (·.k)).sum := congrArg List.sum (List.map_congr_left hc)
      _ = kd := hsum

open Properties.Spot.SecurityProofs in
/-- **The law the verifier agrees is well formed over its strata**: where `deriveLaw` expands the named law, or
`flockLaw` reads `derive`'s law object, and the law is one at `ns` (`LawOn`), its claims design is well formed over
`ns`, by the code's own refusals. A subset draws at least one unit (`flockLaw`); a stratified `K` is in
`[1, population]` (`Flock.Draw.derive`); a work law's strata are nonempty (`strataOf` of a covered partition), its
floors at least one (`workTableOfJson`) and its work nonzero (`deriveWork`). -/
theorem wellFormed_of_lawOn {p : Spot.Program} {query law lawJ : Lean.Json} {table : Option Lean.Json}
    {n : ℕ} {derived : Option Flock.Draw.Law} {parsed : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ}
    (hJ : Spot.Draw.derive p query law table = .ok lawJ)
    (hd : Spot.Draw.deriveLaw p query law table = .ok derived)
    (hp : Spot.Draw.flockLaw lawJ n = .ok parsed) (hlaw : LawOn (derived.getD parsed) ns kd) :
    (Properties.Spot.oneStageOf (derived.getD parsed)).WellFormed ns := by
  rcases deriveLaw_cases p query law table derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S, k, law', hS, rfl, hlaw'⟩
  · have hJ' : lawJ = law := by
      unfold Spot.Draw.derive at hJ
      rw [hd] at hJ
      exact (Except.ok.inj hJ).symm
    subst hJ'
    rcases flockLaw_cases _ _ _ hp with ⟨k, rfl, -, -, hk⟩ | ⟨a, b, rfl, hb, hab⟩ | hk | hk
    · obtain ⟨rfl, hkd⟩ := hlaw
      exact ⟨hk, hkd⟩
    · exact ⟨hb, hab⟩
    · exact absurd hk hk1
    · exact absurd hk hk2
  · rcases hlaw' with hder | ⟨t, tab, ht, hwork⟩
    · obtain ⟨hk1, -⟩ := derive_bounds k S law' hder
      obtain rfl := derive_ok k S law' hder
      obtain ⟨-, hK, hpos, -⟩ := hlaw
      exact ⟨hk1, hK, hpos⟩
    · obtain ⟨ws, hF, htot, rfl⟩ := deriveWork_wf k S tab law' hwork
      have hSpos : ∀ q ∈ S.toList, 1 ≤ rsSize q.2 := by
        unfold Spot.Partition.strata at hS
        obtain ⟨rs, he, hS⟩ := bind_eq_ok.1 hS
        obtain rfl := (Except.ok.inj hS).symm
        exact strataOf_pos _ (partition_covered p query rs he)
      have hfl : ∀ s ∈ ws.toList, 1 ≤ s.floor := fun s hs => by
        obtain ⟨_, -, -, e, he, hfe⟩ := forall₂_mem_right hF s hs
        rw [hfe]
        exact workTable_floor _ tab ht e he
      have hpos : ∀ s ∈ ws.toList, 1 ≤ s.size := fun s hs => by
        obtain ⟨q, hq, hu, -⟩ := forall₂_mem_right hF s hs
        rw [stratum_size, hu]
        exact hSpos q hq
      obtain ⟨⟨-, hsz⟩, -⟩ := hlaw
      have hns : ns = ws.toList.map (·.size) := by
        rw [← hsz]
        simp only [Array.toList_map, List.map_map]
        rfl
      subst hns
      have hW : (((ws.toList.map (·.size)).zip ((ws.map fun s => { s with
          k := Flock.Draw.workK k s.work s.size (Flock.Draw.totalWork ws) s.floor }).toList.map (·.work))).map
            (fun p => p.1 * p.2)).sum = Flock.Draw.totalWork ws := by
        rw [totalWork_eq, Array.toList_map, List.map_map, List.zip_map', List.map_map]
        rfl
      refine ⟨by simp, by simp, fun x hx => ?_, by rw [hW]; exact Nat.pos_of_ne_zero htot, fun x hx => ?_⟩
      · rw [Array.toList_map, List.map_map, List.mem_map] at hx
        obtain ⟨w, hw, rfl⟩ := hx
        exact hfl w hw
      · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hx
        exact hpos w hw

/-- **A sample file of the verifier's sample under a sampling design on strata decodes only to `kd` units** (`lawSized_size`). -/
theorem lawOn_size {n : ℕ} {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (hlaw : LawOn law ns kd)
    {dj : Finset (Fin n) → Lean.Json} (hdj : Discharge.ZkLink.DrawFileZK dj law n) (S : Finset (Fin n))
    (D : Flock.Draw.UnitDraw) (hD : Flock.Draw.UnitDraw.ofJson (dj S) = .ok D) : D.units.size = kd :=
  lawSized_size (lawSized_of_lawOn hlaw) hdj S D hD

/-- A `subset k n` sample that ends holds `k` units. -/
theorem subset_size {k n : ℕ} {src : ByteArray} {D : Array ℕ} (h : Flock.Draw.subset k n src = some D) :
    D.size = k := by
  rw [Audit.DrawExec.subset_eq] at h
  obtain ⟨sw, -, rfl⟩ := Option.map_eq_some_iff.1 h
  rw [← Array.length_toList, (Audit.StrataExec.qsort_perm _ _).length_eq, Array.length_toList, Array.size_map,
    Array.size_range]

open Properties.Spot.SecurityProofs (bind_ok bind_error onWith_take) in
/-- **`stratifiedWith` on the verifier's bytes samples each stratum's `k`**: a run that returns a sample holds `∑ k_s` units,
each stratum's `subset` returning its `k` (`subset_size`). -/
theorem stratifiedWith_size (src : ByteArray) (strata : Array Flock.Draw.Stratum) (p p' : ℕ) (D : Array ℕ)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Spot.Draw.take src need) strata).run p =
      .ok (some D, p')) :
    D.size = (strata.toList.map (·.k)).sum := by
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
          let __x ← Flock.Draw.onWith (fun need => Spot.Draw.take src need) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = .ok ((r, out'), pos₁) →
      r = none ∧ out'.size = out.size + (l.map fun i => strata[i]!.k).sum := by
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
      cases ho : (Flock.Draw.onWith (fun need => Spot.Draw.take src need) strata[i]!.need
          (Flock.Draw.subset strata[i]!.k strata[i]!.size)).run pos with
      | error e => rw [ho] at hl; exact absurd hl (by rw [bind_error, bind_error]; simp)
      | ok x =>
        obtain ⟨v, pos₂⟩ := x
        obtain ⟨j, Dl, rfl, rfl, hst⟩ := onWith_take src _ (by unfold Flock.Draw.Stratum.need; omega) _ pos _ _ ho
        obtain ⟨-, -, hrun, -⟩ := hst
        rw [ho, bind_ok] at hl
        dsimp only at hl
        rw [StateT.run_pure, pure_bind] at hl
        dsimp only at hl
        obtain ⟨hr, hsz⟩ := ih _ _ r out' pos₁ hl
        refine ⟨hr, ?_⟩
        rw [hsz, Array.size_append, Array.size_map, subset_size hrun, List.map_cons, List.sum_cons]
        omega
  rw [StateT.run_bind] at h
  cases hl : (forIn (m := StateT ℕ (Except String)) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run p with
  | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨⟨r, out'⟩, pos₁⟩ := y
    rw [hl, bind_ok] at h
    dsimp only at h
    obtain ⟨rfl, hsz⟩ := loop _ _ _ _ _ _ hl
    rw [StateT.run_pure] at h
    obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj h)
    rw [← Option.some.inj h1, ← Array.length_toList, (Audit.StrataExec.qsort_perm _ _).length_eq,
      Array.length_toList, hsz, Array.size_empty, Nat.zero_add]
    congr 1
    refine List.ext_getElem (by simp) fun i h₁ _ => ?_
    have hi : i < strata.size := by simpa using h₁
    simp [getElem!_pos strata i hi]

open Properties.Spot.SecurityProofs (bind_ok bind_error onWith_take) in
/-- **The deployed sample under such a sampling design samples `kd` units**: a sample from `src` that ends holds `kd` units, `subset k`'s
`k` (`subset_size`), and a stratified or work sampling design's `k_s` in each stratum (`stratifiedWith_size`), at any population. -/
theorem drawUnits_size {law : Flock.Draw.Law} {ns : List ℕ} {kd n : ℕ} (hlaw : LawOn law ns kd) {src : ByteArray}
    {units : Array ℕ} {read : ℕ} (h : Spot.Draw.drawUnits law n src = .ok (units, read)) :
    units.size = kd := by
  unfold Spot.Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take src need) law n).run 0 with
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
    cases law with
    | subset k =>
      obtain ⟨hk, -⟩ := hlaw
      unfold Flock.Draw.drawWith at hw
      dsimp only at hw
      rw [StateT.run_bind] at hw
      cases ho : (Flock.Draw.onWith (fun need => Spot.Draw.take src need)
          (2 * k * Flock.Draw.width n + 64) (Flock.Draw.subset k n)).run 0 with
      | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
      | ok y =>
        obtain ⟨v, p⟩ := y
        obtain ⟨j, D, rfl, -, hst⟩ := onWith_take src _ (by omega) _ 0 _ _ ho
        obtain ⟨-, -, hrun, -⟩ := hst
        rw [ho, bind_ok, StateT.run_pure] at hw
        obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
        obtain rfl := (Option.some.inj h1).symm
        exact (subset_size hrun).trans hk
    | bernoulli a b => exact hlaw.elim
    | stratified K St =>
      obtain ⟨-, -, -, -, hsum⟩ := hlaw
      unfold Flock.Draw.drawWith at hw
      dsimp only at hw
      rw [StateT.run_bind] at hw
      cases ho : (Flock.Draw.stratifiedWith (fun _ need => Spot.Draw.take src need) St).run 0 with
      | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
      | ok y =>
        obtain ⟨v, p⟩ := y
        rw [ho, bind_ok, StateT.run_pure] at hw
        obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
        cases v with
        | none => exact absurd h1 (by simp)
        | some D =>
          obtain rfl := (Option.some.inj h1).symm
          exact (stratifiedWith_size src St 0 p D ho).trans hsum
    | work K ws =>
      obtain ⟨-, -, hsum⟩ := hlaw
      unfold Flock.Draw.drawWith at hw
      dsimp only at hw
      rw [StateT.run_bind] at hw
      cases ho : (Flock.Draw.stratifiedWith (fun _ need => Spot.Draw.take src need)
          (ws.map (·.toStratum))).run 0 with
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
          rw [← hsum, Array.toList_map, List.map_map]
          rfl

end Flock.Bridge
