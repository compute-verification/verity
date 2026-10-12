import Proofs.Flock.C3.RegLevels.SrcOpened
import Proofs.Flock.VBridge.Runs.Lg

/-!
# n7's levels: R's Ligerito run's contexts are its levels (`ConssAt`)

G's `Conss.ligerito_conss` is a statement about the symbolic run (`LM Lin`), and a run over F does not give one over
`Lin` (`Sim` goes from `Lin` to F). `levelsLoop_starts` and `ligeritoRun_starts` are `Runs.levelsLoop_count` and
`Runs.ligeritoRun_conss` with each context's level and query count in place of their number, and `conssAt_of_lig`
gives `ConssAt` wherever V*'s Ligerito over F returns zero residuals at a `fast100` schedule (`levelTie_of_lig`'s
hypotheses).
-/

namespace FlockC3.RegLevels

open Flock FlockSoundness.Refine
open Flock.RecAlgebra (Messages LSt Coins)

variable {c : Array (Array F128)}

/-- V*'s level loop records level `i + 1`'s context at iteration `i`, with the level's `q` positions. -/
theorem levelsLoop_starts [Fact (RecAlgebra.NoEmpty c)] {M : Messages F128} {sch : Flock.Schedule} {R : ℕ}
    (hq : ∀ i ≤ R, LevelSeam (sch.levels.getD i default)) :
    ∀ (fuel i split : ℕ) (levels : Array (Array F256)) (e : ℕ) (st : LSt F128) (k : ℕ)
      (out : Array (Array F256) × ℕ) (st' : LSt F128) (k' : ℕ), i + fuel ≤ R →
      RecAlgebra.levelsLoop sch M R fuel i split levels e st k (Coins.ofRounds c) = .ok ((out, st'), k') →
      st'.conss.toList.map (fun x => (x.start, x.pos.size)) =
        st.conss.toList.map (fun x => (x.start, x.pos.size)) ++
          (List.range' (i + 1) fuel).map fun l => (l, (sch.levels.getD l default).q)
  | 0, _, _, _, _, st, k, out, st', k', _, h => by
    simp only [RecAlgebra.levelsLoop] at h
    obtain ⟨-, rfl, -⟩ := RecAlgebra.vpure h
    simp only [List.range'_zero, List.map_nil, List.append_nil]
  | fuel + 1, i, split, levels, e, st, k, out, st', k', hi, h => by
    simp only [RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (RecAlgebra.vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := RecAlgebra.vbind_ok h
    obtain ⟨-, -, -, -, -, hc₁, -⟩ := RecAlgebra.challenges_conv (p := default) 4 0 #[] st st₁ k k₁ chs h₁
    try dsimp only at h
    by_cases hi' : i + 1 < R
    · rw [ite_eq_left hi'] at h
      obtain ⟨u₂, st₂, k₂, h₂, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨u₃, st₃, k₃, h₃, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨u₄, st₄, k₄, h₄, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨-, -, -, hc₂, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
      obtain ⟨-, -, -, hc₃, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
      obtain ⟨-, -, -, -, mm, -, -, -, -, hc₄, -⟩ := RecAlgebra.vqueryFold (hq (i + 1) (by omega)).q_ne_zero h₄
      have hp : ∀ cs, (positions (sch.levels.getD (i + 1) default) cs).size = (sch.levels.getD (i + 1) default).q :=
        positions_size _ (hq (i + 1) (by omega))
      rw [levelsLoop_starts hq fuel (i + 1) _ _ _ st₄ k₄ out st' k' (by omega) h, hc₄, Array.toList_push, hc₃, hc₂,
        hc₁, List.range'_succ]
      simp only [List.map_append, List.map_cons, List.map_nil, hp, List.append_assoc, List.cons_append,
        List.nil_append]
    · rw [ite_eq_right hi'] at h
      obtain ⟨r, st₂, k₂, h₂, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨rfl, rfl, -, -, rfl⟩ := RecAlgebra.vquery (hq R le_rfl).q_ne_zero h₂
      try dsimp only at h
      obtain ⟨-, h⟩ := RecAlgebra.vcoinOne_bind h
      obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨e₃, rfl⟩ := RecAlgebra.vmodify_ok h₃
      have hR : R = i + 1 := by omega
      have hp : ∀ cs, (positions (sch.levels.getD R default) cs).size = (sch.levels.getD R default).q :=
        positions_size _ (hq R le_rfl)
      rw [levelsLoop_starts hq fuel (i + 1) _ _ _ st₃ _ out st' k' (by omega) h, e₃, Array.toList_push, hc₁,
        List.range'_succ]
      simp only [List.map_append, List.map_cons, List.map_nil, hp, List.append_assoc, List.cons_append,
        List.nil_append]
      rw [hR]

/-- **V*'s Ligerito up to its final check records level `ℓ`'s context `ℓ`-th**, with the level's `q` positions
(`Runs.ligeritoRun_conss`, which counts them). -/
theorem ligeritoRun_starts [Fact (RecAlgebra.NoEmpty c)] {sch : Flock.Schedule} {M : Messages F128} {T : F128}
    {k₀ k' : ℕ} {initCh : Array F256} {levels : Array (Array F256)} {ext : ℕ} {stV : LSt F128}
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hrun : (RecAlgebra.ligeritoRun sch M).run
        { s := RecAlgebra.kofF T, q := ((RecAlgebra.Val.zero, RecAlgebra.Val.zero),
          (RecAlgebra.Val.zero, RecAlgebra.Val.zero), (RecAlgebra.Val.zero, RecAlgebra.Val.zero)) } k₀
        (Coins.ofRounds c) = .ok (((initCh, levels, ext), stV), k')) :
    stV.conss.toList.map (fun x => (x.start, x.pos.size)) =
      (List.range sch.levels.size).map fun l => (l, (sch.levels[l]!).q) := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hq : ∀ i ≤ sch.levels.size - 1, LevelSeam (sch.levels.getD i default) := fun i hi => hlv i (by omega)
  have hp : ∀ cs, (positions (sch.levels[0]!) cs).size = (sch.levels[0]!).q := fun cs => by
    rw [← FlockVBridge.Family.Conss.getD_default]
    exact positions_size _ (hlv 0 (by omega)) cs
  rw [StateT.run] at hrun
  unfold RecAlgebra.ligeritoRun at hrun
  obtain ⟨-, h⟩ := RecAlgebra.vcoinVec_bind (by omega) hrun
  obtain ⟨y0, -, h⟩ := RecAlgebra.vnextOod_bind h
  obtain ⟨-, h⟩ := RecAlgebra.vcoinOne_bind h
  obtain ⟨⟨⟩, stZ, kZ, hZ, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨eZ, rfl⟩ := RecAlgebra.vmodify_ok hZ
  subst eZ
  obtain ⟨m0, -, h⟩ := RecAlgebra.vnextMsg_bind h
  obtain ⟨⟨⟩, stA, kA, hA, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨eA, rfl⟩ := RecAlgebra.vmodify_ok hA
  obtain ⟨ic, st₁, k₁, h₁, h⟩ := RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, st₂, k₂, h₂, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₄, k₄, h₄, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨⟨lv', ex'⟩, st₅, k₅, h₅, h⟩ := RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨-, e5, -⟩ := RecAlgebra.vpure h
  subst e5
  obtain ⟨-, -, -, -, -, hc₁, -⟩ := RecAlgebra.challenges_conv (p := default) sch.k0 0 #[] _ _ _ _ _ h₁
  obtain ⟨-, -, -, hc₂, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
  obtain ⟨-, -, -, hc₃, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
  obtain ⟨-, -, -, -, mq, -, -, -, -, hc₄, -⟩ := RecAlgebra.vqueryFold (hlv 0 (by omega)).q_ne_zero h₄
  have hA0 : stA.conss = #[] := by rw [eA]
  rw [levelsLoop_starts hq _ _ _ _ _ st₄ _ _ _ _ (by omega) h₅, hc₄, Array.toList_push, hc₃, hc₂, hc₁, hA0,
    FlockVBridge.Family.Conss.range_eq_cons (n := sch.levels.size) (by omega)]
  simp only [Array.toList_empty, List.nil_append, List.map_cons, List.map_nil, List.cons_append, hp, Nat.zero_add,
    FlockVBridge.Family.Conss.getD_default]

/-- **(n7's levels) R's run's contexts are its levels** (`ConssAt`): where V*'s Ligerito over F returns zero residuals
at a `fast100` schedule on rounds with no empty round, it records level `ℓ`'s context `ℓ`-th with the level's `q`
positions. -/
theorem conssAt_of_lig {m : ℕ} {sch : Flock.Schedule} {M : Messages F128} {tIn : F128} {sws : Array Switched}
    {k₂ k₃ : ℕ} {lgr : Array F128} {lg : RecAlgebra.LgPts} (hf : Flock.fast100 m = some sch)
    (hlg : RecAlgebra.ligerito sch M tIn sws k₂ (Coins.ofRounds c) = .ok ((lgr, lg), k₃)) (hz : ∀ x ∈ lgr, x = 0)
    (hne : RecAlgebra.NoEmpty c) : ConssAt sch lg := by
  obtain ⟨ext, stV, hrun, -, -, -, -, hlc, -, -⟩ := RecAlgebra.ligerito_checks hlg hz
  have : Fact (RecAlgebra.NoEmpty c) := ⟨hne⟩
  have hdim := (fast100_dims m sch hf).1
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi => by
    have he : sch.levels.getD i default = sch.levels[i] := by simp [Array.getD, hi]
    rw [he]
    exact fast100_level m sch hf _ (Array.getElem_mem hi)
  unfold ConssAt
  rw [hlc]
  exact ligeritoRun_starts hdim hlv hrun

end FlockC3.RegLevels
