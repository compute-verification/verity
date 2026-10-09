import Proofs.Flock.Recursive.Target
import Proofs.Flock.Recursive.ForkRuns

/-!
# Recursion: the compile lemma on the runs of one prover

The compile lemma (`flock_inner_sound_compiled`, `Compile`) for one prover `s` of `flockInnerCL` and one `G`: two runs
of `s` satisfying `G` play the compiled table to outcomes that clash, or `s` makes the inner verifier accept a transcript
satisfying `G` with probability at most `tableError`. The clash is any relation `Cl` that a self-clash, a conflict at the
level-0 cap and a conflict at a rep's recorded cap each imply, so that `Compile` reads it at its `ClashCL`.

* `RunOut`: the outcomes the compiled table ends at on the runs of `s` that satisfy `G`.
* `sched_prob_le_value`: the schedule's simulation at one statement, the body of `sched_sound`: against `s`, the run on
  the schedule's steps ends at an outcome satisfying `p` no more often than the game's `value`.
* `table_value_clashFree` (**open**): the compiled table's value on a set of outcomes no two of which clash is at most
  `tableError`.
* `compiled_runs_sound`: the lemma, from the three. With no clash between two runs satisfying `G`, `RunOut` is clash
  free, every accepted run satisfying `G` ends in it, and the simulation reads the table's value there.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

section Clash

/-- **The compiled table on a clash-free set of outcomes.** On a statement no witness satisfies, for every set `W` of
outcomes none of which self-clashes, no two of which open a position of the first's level-0 cap to different rows, and no
two of which open a position of a rep's level-`ℓ` cap that the first records to different rows: every prover makes the
compiled table end accepted in `W` with probability at most `tableError`.

Its proof is open. It is `table_sound_compiledL`'s argument with the plurality chooser replaced by `W`'s: the level-0
table at a cap `c` is the row a `W`-outcome with `cap₀ = c` opens at each position (any row where none does), and a rep's
table at `(slot, d)` the row a `W`-outcome recording `d` there opens. By the hypotheses an accepted run in `W` is off
none of these tables and does not self-clash, so `bad_splitL` leaves it agreeing with the oracle run that
`lockTableL` couples to it (the chooser's table is the one `Lock.getPair` records at each slot), and
`Lock.expect_le_coupled` with `table_sound` bounds it by `tableError` with no bad term, as in `compiled_le_badL`. Needs:
the chooser, the lemma that a lockstep run records the chooser's table at each slot it fills, the agreement from the
three hypotheses, and the conversions between `prob`, `expect` and `value` (`prob_le_value` in reverse is `iSup` over
the cap, then `prob_toReal`). -/
theorem table_value_clashFree (A : Arith F K) (hA : A.Correct) [Inhabited D] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (hno : ¬ ∃ z, S.Satisfies z) (W : OutCL F D Sl → Prop)
    (hW₀ : ∀ o, W o → ¬ SelfClashL H E Lf sch o)
    (hW₁ : ∀ o o', W o → W o' → ∀ p, ¬ Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p)
    (hW₂ : ∀ o o', W o → W o' → ∀ i < 2, ∀ ℓ, 1 ≤ ℓ → ∀ cap,
      (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap →
      ∀ p, ¬ Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p) :
    value (fun o : OutCL F D Sl => o.accepted = true ∧ W o) (tableCL A H E Lf S sch hm ptLocal mPts) ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) := by
  sorry

end Clash

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

open Classical in
/-- **The schedule's simulation at one statement**: against every prover on the schedule, the run on the schedule's
steps ends at an outcome of `g` satisfying `p` no more often than `g`'s value. `sched_sound`'s body. -/
theorem sched_prob_le_value {α : Type} (g : Game α) (p : α → Prop)
    (s : Strategy (rounds (List (SMsg T)) ((k : Fin r) → CT k) r .ret)) :
    prob (fun t => ∃ a, PlaysT g (schedSteps T CT 0 t) a ∧ p a) (rounds (List (SMsg T)) ((k : Fin r) → CT k) r .ret)
      s ≤ value p g := by
  refine le_trans (le_of_eq ?_) (leafAvg_plays_le T CT p r 0 (by omega) .ret s g)
  exact prob_rounds _ r _ s

/-- **The outcomes of the runs satisfying `G`**: the outcomes the compiled table ends at on the transcripts `s` reaches
that satisfy `G`. -/
def RunOut (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (M : Hidden cls)
    (G : List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop)
    (s : Strategy (flockInnerCL T CT A H E Lf cls regs sch).game) (o : OutCL F D Sl) : Prop :=
  ∃ t, Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t ∧ G t ∧
    PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t) o

/-- **The compile lemma on one prover's runs.** For a hidden template no witness satisfies, every `G` and every prover
`s` of `flockInnerCL`: two transcripts `s` reaches that satisfy `G` play the compiled table to outcomes related by `Cl`,
or `s` makes the inner verifier accept a transcript satisfying `G` with probability at most `tableError`. `Cl` is any
relation that a self-clash of either outcome, a conflict at the first's level-0 cap and a conflict at a rep's cap the
first records each imply. -/
theorem compiled_runs_sound (A : Arith F K) (hA : A.Correct) [Inhabited D]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (hsch : fast100 cls.m = some sch)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts)
    (Cl : OutCL F D Sl → OutCL F D Sl → Prop)
    (hself : ∀ o o', SelfClashL H E Lf sch o → Cl o o')
    (h₀ : ∀ o o' p, Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p → Cl o o')
    (hR : ∀ o o', ∀ i < 2, ∀ ℓ, 1 ≤ ℓ → ∀ cap, (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap →
      ∀ p, Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p → Cl o o')
    (M : Hidden cls) (hno : ¬ (flockInnerCL T CT A H E Lf cls regs sch).holds M)
    (G : List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop)
    (s : Strategy (flockInnerCL T CT A H E Lf cls regs sch).game) :
    (∃ t t' o o', Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t ∧
      Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t' ∧ G t ∧ G t' ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t) o ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t') o' ∧
      Cl o o') ∨
    prob (fun t => (flockInnerCL T CT A H E Lf cls regs sch).accepts M t ∧ G t)
        (flockInnerCL T CT A H E Lf cls regs sch).game s ≤
      ENNReal.ofReal (tableError sch ⟨cls.kLog, regs.length, cls.mPts⟩) := by
  by_cases hc : ∃ t t' o o', Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t ∧
      Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t' ∧ G t ∧ G t' ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t) o ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t') o' ∧
      Cl o o'
  · exact Or.inl hc
  right
  have hclean : ∀ o o', RunOut T CT A H E Lf cls regs sch M G s o → RunOut T CT A H E Lf cls regs sch M G s o' →
      ¬ Cl o o' := by
    rintro o o' ⟨t, ht, hG, ho⟩ ⟨t', ht', hG', ho'⟩ hcl
    exact hc ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', hcl⟩
  have hval := table_value_clashFree A hA H E Lf (M.toStatement regs) sch hsch cls.hm cls.ptLocal cls.mPts (hlay M) hno
    (RunOut T CT A H E Lf cls regs sch M G s)
    (fun o ho h => hclean o o ho ho (hself o o h))
    (fun o o' ho ho' p h => hclean o o' ho ho' (h₀ o o' p h))
    (fun o o' ho ho' i hi ℓ hℓ cap hcap p h => hclean o o' ho ho' (hR o o' i hi ℓ hℓ cap hcap p h))
  calc prob (fun t => (flockInnerCL T CT A H E Lf cls regs sch).accepts M t ∧ G t)
        (flockInnerCL T CT A H E Lf cls regs sch).game s
      ≤ prob (fun t => ∃ a, PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)
          (schedSteps T CT 0 t) a ∧ (a.accepted = true ∧ RunOut T CT A H E Lf cls regs sch M G s a))
        (flockInnerCL T CT A H E Lf cls regs sch).game s := by
        refine prob_mono_outcome _ s fun t ht h => ?_
        obtain ⟨⟨a, ha, hacc⟩, hG⟩ := h
        exact ⟨a, ha, hacc, t, ht, hG, ha⟩
    _ ≤ value (fun a : OutCL F D Sl => a.accepted = true ∧ RunOut T CT A H E Lf cls regs sch M G s a)
          (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) :=
        sched_prob_le_value T CT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)
          (fun a : OutCL F D Sl => a.accepted = true ∧ RunOut T CT A H E Lf cls regs sch M G s a) s
    _ ≤ _ := hval

end FlockSoundness.Discharge.Recursion
