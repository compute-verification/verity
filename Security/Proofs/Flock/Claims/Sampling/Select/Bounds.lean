import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Claims.Sampling.ExecBernoulli

/-!
# The draw `drawUnits` makes misses a set of units no more often than its law's `P_a`

`drawUnits` runs its law's stages on the coins in order (`drawUnits_seq`), and each stage is bounded by its sampler's
escape on a stream of any length (`Law.subset_exec_escape_le`, `Law.bernoulli_exec_escape_le`), so the stopping-time
bound (`seqMiss_le`) multiplies them.

* `subset_none_of_lt`: `subset k n` draws nothing when `k > n`, so such a stratum's stage misses with probability `0`.
* `one_bound`: a subset or Bernoulli draw misses a set `U` of units with probability at most
  `C(n − |U|, k)/C(n, k)` or `((den − num)/den)^|U|`.
* `strata_bound`: a stratified or work draw misses the units of its strata at the slots of `W` with probability at most
  `strataPa` of its strata's sizes and draws.
-/

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit Finset Claims.Sampling
open scoped ENNReal

theorem uniform_zero (src : ByteArray) : ∀ fuel pos, Flock.Draw.uniform 0 src fuel pos = none
  | 0, _ => rfl
  | fuel + 1, pos => by
    unfold Flock.Draw.uniform
    dsimp only
    split
    · rfl
    · simp only [Nat.mod_zero, Nat.sub_self, Nat.not_lt_zero, ↓reduceIte]
      exact uniform_zero src fuel _

theorem loopSw_none (n : ℕ) (src : ByteArray) :
    ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ) (pos : ℕ), n ∈ l → DrawExec.loopSw n src l sw pos = none
  | [], _, _, h => absurd h List.not_mem_nil
  | i :: is, sw, pos, h => by
    unfold DrawExec.loopSw
    split
    · rfl
    · rename_i r p hu
      by_cases hi : i = n
      · subst hi
        rw [Nat.sub_self, uniform_zero] at hu
        exact absurd hu (by simp)
      · exact loopSw_none n src is _ p ((List.mem_cons.1 h).resolve_left fun e => hi e.symm)

/-- **`subset k n` draws nothing when `k > n`**: its `n`-th step draws from `[0, 0)`. -/
theorem subset_none_of_lt (k n : ℕ) (hk : n < k) (src : ByteArray) : Flock.Draw.subset k n src = none := by
  rw [DrawExec.subset_eq, loopSw_none n src _ _ _ (List.mem_range'.2 ⟨n, hk, by omega⟩), Option.map_none]

/-- Stages with the same reads and samplers, each condition implying the other's. -/
theorem seqMiss_imp : ∀ (l l' : List Stage),
    List.Forall₂ (fun s t : Stage => s.need = t.need ∧ s.run = t.run ∧ ∀ D, s.miss D → t.miss D) l l' →
    ∀ src p, SeqMiss l src p → SeqMiss l' src p
  | [], [], _, _, _, _ => trivial
  | s :: l, t :: l', .cons ⟨hn, hr, hm⟩ h, src, p, ⟨j, D, hst, hmiss, hrest⟩ => by
    refine ⟨j, D, hn ▸ hr ▸ hst, hm D hmiss, hn ▸ seqMiss_imp l l' h src _ hrest⟩

/-- A subset stage missing `U`, bounded by `subset`'s escape. -/
theorem subset_stage_bounded (n k : ℕ) (hk : k ≤ n) (N : ℕ) (hN : 0 < N) (U : Finset ℕ) :
    Stage.Bounded ⟨N, Flock.Draw.subset k n, fun D => ∀ u ∈ U, u ∉ D⟩ ((Law.subset n k hk).escape (finOf n U)) := by
  refine ⟨hN, fun a b D h => Law.subset_append k n a b D h, fun L => ?_⟩
  refine le_trans (prCoin_mono fun f ⟨D, hD, hm⟩ => ⟨D, hD, fun u hu => hm u.val ?_⟩)
    (Law.subset_exec_escape_le n k L hk (finOf n U))
  simpa [finOf] using hu

/-- A Bernoulli stage missing `U`, bounded by `bernoulli`'s escape. -/
theorem bernoulli_stage_bounded (n num den : ℕ) [NeZero den] (h : num ≤ den) (N : ℕ) (hN : 0 < N) (U : Finset ℕ) :
    Stage.Bounded ⟨N, Flock.Draw.bernoulli num den n, fun D => ∀ u ∈ U, u ∉ D⟩
      ((Law.bernoulli n num den).escape (finOf n U)) := by
  refine ⟨hN, fun a b D h => Law.bernoulli_append num den n a b D h, fun L => ?_⟩
  refine le_trans (prCoin_mono fun f ⟨D, hD, hm⟩ => ⟨D, hD, fun u hu => hm u.val ?_⟩)
    (Law.bernoulli_exec_escape_le n num den L h (finOf n U))
  simpa [finOf] using hu

theorem escape_ne_top {n : ℕ} (L : Law n) (B : Finset (Fin n)) : L.escape B ≠ ⊤ :=
  ne_top_of_le_ne_top ENNReal.one_ne_top (DrawExec.prCoin_le_one _)

/-- The claims' probability of an event below a finite bound on its `prCoin`. -/
theorem Pr_le_of_prCoin {L : ℕ} (E : (Fin L → Fin 256) → Prop) (q : ℝ≥0∞) (hq : q ≠ ⊤) (h : prCoin E ≤ q) :
    Claims.Pr E ≤ q.toReal := by
  rw [Pr_eq_toReal_prCoin]
  exact ENNReal.toReal_mono hq h

/-- **A subset or Bernoulli draw misses `U`** with probability at most `C(n − |U|, k)/C(n, k)`, or
`((den − num)/den)^|U|`. -/
theorem one_bound (n : ℕ) (U : Finset ℕ) (hU : U ⊆ range n) (L : ℕ) :
    (∀ k, k ≤ n → Claims.Pr (fun src : Fin L → Fin 256 => ∃ units read,
        Draw.drawUnits (.subset k) n (bytesOf src) = .ok (units, read) ∧ ∀ u ∈ U, u ∉ units) ≤
      ((n - U.card).choose k : ℝ) / (n.choose k : ℝ)) ∧
    (∀ num den, 0 < den → num ≤ den → Claims.Pr (fun src : Fin L → Fin 256 => ∃ units read,
        Draw.drawUnits (.bernoulli num den) n (bytesOf src) = .ok (units, read) ∧ ∀ u ∈ U, u ∉ units) ≤
      (((den - num : ℕ) : ℝ) / den) ^ U.card) := by
  refine ⟨fun k hk => ?_, fun num den hden h => ?_⟩
  · have hle := seqMiss_le [(⟨2 * k * Flock.Draw.width n + 64, Flock.Draw.subset k n, fun D => ∀ u ∈ U, u ∉ D⟩,
      (Law.subset n k hk).escape (finOf n U))] (fun p hp => by
        rw [List.mem_singleton.1 hp]; exact subset_stage_bounded n k hk _ (by omega) U) L
    rw [List.map_singleton, List.map_singleton, List.prod_singleton] at hle
    dsimp only at hle
    refine le_trans (Pr_le_of_prCoin _ _ (escape_ne_top _ _) (le_trans (prCoin_mono fun f hf => ?_) hle)) ?_
    · obtain ⟨units, read, hd, hm⟩ := hf
      refine seqMiss_imp _ _ ?_ _ _ (drawUnits_seq _ n _ units read hd)
      exact .cons ⟨rfl, rfl, fun D hD u hu huD => hm u hu (hD u huD)⟩ .nil
    · rw [Law.subset_escape, card_finOf hU, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]
  · have : NeZero den := ⟨by omega⟩
    have hle := seqMiss_le [(⟨2 * n * Flock.Draw.width den + 64, Flock.Draw.bernoulli num den n,
      fun D => ∀ u ∈ U, u ∉ D⟩, (Law.bernoulli n num den).escape (finOf n U))] (fun p hp => by
        rw [List.mem_singleton.1 hp]; exact bernoulli_stage_bounded n num den h _ (by omega) U) L
    rw [List.map_singleton, List.map_singleton, List.prod_singleton] at hle
    dsimp only at hle
    refine le_trans (Pr_le_of_prCoin _ _ (escape_ne_top _ _) (le_trans (prCoin_mono fun f hf => ?_) hle)) ?_
    · obtain ⟨units, read, hd, hm⟩ := hf
      refine seqMiss_imp _ _ ?_ _ _ (drawUnits_seq _ n _ units read hd)
      exact .cons ⟨rfl, rfl, fun D hD u hu huD => hm u hu (hD u huD)⟩ .nil
    · rw [Law.bernoulli_escape h, card_finOf hU, ENNReal.toReal_pow, bernFactor_zero num den]

/-- A stratum's units, in order: `Stratum.unit` over its ranges. -/
theorem stratum_unit (s : Flock.Draw.Stratum) (j : ℕ) : s.unit j = (expand s.units.toList).getD j 0 := by
  unfold Flock.Draw.Stratum.unit
  simp only [Id.run]
  rw [← Array.forIn_toList]
  generalize s.units.toList = l
  induction l generalizing j with
  | nil => rfl
  | cons p rs ih =>
    obtain ⟨lo, hi⟩ := p
    have he : expand ((lo, hi) :: rs) = List.range' lo (hi - lo) ++ expand rs := rfl
    simp only [List.forIn_cons, he]
    split
    · rename_i hj
      rw [List.getD_append _ _ _ _ (by simpa using hj)]
      simp [hj]
      rfl
    · rename_i hj
      rw [List.getD_append_right _ _ _ _ (by simp; omega), List.length_range']
      simpa using ih _

theorem stratum_size' (s : Flock.Draw.Stratum) : s.size = (expand s.units.toList).length := stratum_size s

/-- The stage of stratum `s` at slot `o`, missing the slots of `W`, and its bound: `subset`'s escape, `0` when the stratum
draws more units than it has. -/
noncomputable def stratumQ (W : Finset ℕ) (s : Flock.Draw.Stratum) (o : ℕ) : ℝ≥0∞ :=
  if h : s.k ≤ s.size then (Law.subset s.size s.k h).escape (finOf s.size (window o s.size W)) else 0

/-- The strata's stages from slot `o`, each missing its window of `W`, with their bounds. -/
noncomputable def strataStages (W : Finset ℕ) : List Flock.Draw.Stratum → ℕ → List (Stage × ℝ≥0∞)
  | [], _ => []
  | s :: ss, o => (stage s (fun D => ∀ j ∈ window o s.size W, j ∉ D), stratumQ W s o) :: strataStages W ss (o + s.size)

theorem stratum_stage_bounded (W : Finset ℕ) (s : Flock.Draw.Stratum) (o : ℕ) :
    (stage s (fun D => ∀ j ∈ window o s.size W, j ∉ D)).Bounded (stratumQ W s o) := by
  have hN : 0 < s.need := by unfold Flock.Draw.Stratum.need; omega
  unfold stratumQ
  split
  · rename_i h
    exact subset_stage_bounded s.size s.k h s.need hN _
  · rename_i h
    refine ⟨hN, fun a b D h => Law.subset_append _ _ a b D h, fun L => ?_⟩
    have : (fun f : Fin L → Fin 256 => ∃ D, (stage s fun D => ∀ j ∈ window o s.size W, j ∉ D).run (Law.streamOf f) =
        some D ∧ (stage s fun D => ∀ j ∈ window o s.size W, j ∉ D).miss D) = fun _ => False := by
      funext f
      simp only [stage, subset_none_of_lt s.k s.size (by omega), reduceCtorEq, false_and, exists_false]
    rw [this, prCoin_false]

theorem stratumQ_toReal (W : Finset ℕ) (s : Flock.Draw.Stratum) (o : ℕ) :
    (stratumQ W s o).toReal = ((s.size - (window o s.size W).card).choose s.k : ℝ) / (s.size.choose s.k : ℝ) := by
  unfold stratumQ
  split
  · rw [Law.subset_escape, card_finOf (window_subset _ _ _), ENNReal.toReal_div, ENNReal.toReal_natCast,
      ENNReal.toReal_natCast]
  · rw [Nat.choose_eq_zero_of_lt (n := s.size) (k := s.k) (by omega), Nat.cast_zero, div_zero, ENNReal.toReal_zero]

theorem stratumQ_le_one (W : Finset ℕ) (s : Flock.Draw.Stratum) (o : ℕ) : stratumQ W s o ≤ 1 := by
  unfold stratumQ
  split
  · exact DrawExec.prCoin_le_one _
  · exact zero_le_one

theorem strataStages_bounded (W : Finset ℕ) :
    ∀ (l : List Flock.Draw.Stratum) (o : ℕ), ∀ p ∈ strataStages W l o, p.1.Bounded p.2
  | [], _, _, h => absurd h List.not_mem_nil
  | s :: ss, o, p, h => by
    rcases List.mem_cons.1 h with rfl | h
    · exact stratum_stage_bounded W s o
    · exact strataStages_bounded W ss _ p h

theorem strataStages_prod (W : Finset ℕ) : ∀ (l : List Flock.Draw.Stratum) (o : ℕ),
    ((strataStages W l o).map fun p : Stage × ℝ≥0∞ => p.2).prod ≤ 1 ∧
      ((strataStages W l o).map fun p : Stage × ℝ≥0∞ => p.2).prod.toReal = strataPa (l.map fun s => (s.size, s.k)) o W
  | [], _ => by simp [strataStages, strataPa]
  | s :: ss, o => by
    obtain ⟨h1, h2⟩ := strataStages_prod W ss (o + s.size)
    simp only [strataStages, List.map_cons, List.prod_cons, strataPa]
    exact ⟨mul_le_one' (stratumQ_le_one W s o) h1, by rw [ENNReal.toReal_mul, h2, stratumQ_toReal]⟩

/-- **The strata's stages as `drawUnits` runs them miss their windows**: stage `i` of the strata `St` draws its
`i`-th stratum's units among `units`, and no unit at a slot of `W` is among them. -/
theorem strataStages_of (St : Array Flock.Draw.Stratum) (units : Array ℕ) (W : Finset ℕ)
    (hW : ∀ v ∈ W, (St.toList.flatMap fun s => expand s.units.toList).getD v 0 ∉ units) :
    ∀ (pre l : List Flock.Draw.Stratum), St.toList = pre ++ l →
      List.Forall₂ (fun s t : Stage => s.need = t.need ∧ s.run = t.run ∧ ∀ D, s.miss D → t.miss D)
        ((List.range' pre.length l.length).map fun i => stage St[i]! fun Dl => ∀ x ∈ Dl, St[i]!.unit x ∈ units)
        ((strataStages W l (pre.flatMap fun s => expand s.units.toList).length).map (·.1))
  | _, [], _ => .nil
  | pre, s :: l, hL => by
    have hs : St[pre.length]! = s := by rw [← Array.getElem!_toList, hL]; simp
    rw [List.length_cons, List.range'_succ, List.map_cons]
    simp only [strataStages, List.map_cons]
    refine .cons ⟨by rw [hs]; rfl, by rw [hs]; rfl, fun D hD j hj hjD => ?_⟩ ?_
    · have hjs : j < s.size := Finset.mem_range.1 (window_subset _ _ _ hj)
      have hv := (mem_window hjs).1 hj
      refine hW _ hv ?_
      have hu := hD j hjD
      rw [hs, stratum_unit] at hu
      rw [hL, List.flatMap_append, List.getD_append_right _ _ _ _ (by omega), Nat.add_sub_cancel_left,
        List.flatMap_cons, List.getD_append _ _ _ _ (by rw [← stratum_size']; exact hjs)]
      exact hu
    · have := strataStages_of St units W hW (pre ++ [s]) l (by rw [hL]; simp)
      simp only [List.length_append, List.length_singleton, List.flatMap_append, List.flatMap_singleton,
        List.length_append, ← stratum_size'] at this
      exact this

/-- **A stratified or work draw misses the units at the slots of `W`** with probability at most `strataPa` of its strata's
sizes and draws, its strata's units in order as slots. -/
theorem strata_bound (law : Flock.Draw.Law) (St : Array Flock.Draw.Stratum)
    (hSt : (∃ k, law = .stratified k St) ∨ ∃ k ws, law = .work k ws ∧ St = ws.map (·.toStratum)) (n : ℕ)
    (W : Finset ℕ) (L : ℕ) :
    Claims.Pr (fun src : Fin L → Fin 256 => ∃ units read, Draw.drawUnits law n (bytesOf src) = .ok (units, read) ∧
      ∀ v ∈ W, (St.toList.flatMap fun s => expand s.units.toList).getD v 0 ∉ units) ≤
    strataPa (St.toList.map fun s => (s.size, s.k)) 0 W := by
  have hstages : ∀ units : Array ℕ, lawStages law n (fun i Dl => ∀ x ∈ Dl, unitOf law i x ∈ units) =
      (List.range' 0 St.size).map fun i => stage St[i]! fun Dl => ∀ x ∈ Dl, St[i]!.unit x ∈ units := by
    intro units
    rcases hSt with ⟨k, rfl⟩ | ⟨k, ws, rfl, rfl⟩ <;> rfl
  obtain ⟨hle1, heq⟩ := strataStages_prod W St.toList 0
  have hle := seqMiss_le (strataStages W St.toList 0) (strataStages_bounded W _ _) L
  rw [← heq]
  refine Pr_le_of_prCoin _ _ (ne_top_of_le_ne_top ENNReal.one_ne_top hle1) (le_trans (prCoin_mono fun f hf => ?_) hle)
  obtain ⟨units, read, hd, hm⟩ := hf
  have hs := drawUnits_seq law n _ units read hd
  rw [hstages] at hs
  have := strataStages_of St units W hm [] St.toList rfl
  simp only [List.length_nil, List.flatMap_nil, Array.length_toList] at this
  exact seqMiss_imp _ _ this _ _ hs

end Properties.SampledProofs.SecurityProofs
