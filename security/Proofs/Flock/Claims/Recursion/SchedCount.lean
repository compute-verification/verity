module

public import Proofs.Flock.Claims.Deploy
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import VCVio.Prelude.Core

/-! The plain schedule's size, and the plain verifier's count of `2 + 2R` rounds at it (`k_plain`). -/

@[expose] public section

namespace RecDeploy

open Flock

theorem forIn_opt_size {α : Type} (g : ℕ → ℕ) (f : ℕ → Array α × ℕ → Option (ForInStep (Array α × ℕ)))
    (hf : ∀ i s t, f i s = some t → ∃ u, t = .yield u ∧ u.1.size = s.1.size + g i) :
    ∀ (l : List ℕ) (s0 s : Array α × ℕ), forIn l s0 f = some s → s.1.size = s0.1.size + (l.map g).sum := by
  intro l
  induction l with
  | nil => intro s0 s h; simp at h; subst h; simp
  | cons a l ih =>
    intro s0 s h
    simp only [List.forIn_cons] at h
    cases hfa : f a s0 with
    | none => rw [hfa] at h; cases h
    | some t =>
      rw [hfa] at h
      obtain ⟨u, rfl, hu⟩ := hf a s0 t hfa
      have := ih u s h
      simp only [List.map_cons, List.sum_cons]
      omega

theorem sum_map_one (l : List ℕ) : (l.map fun _ => 1).sum = l.length := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [List.map_cons, List.sum_cons, ih, List.length_cons]; omega

theorem fast100_sizes {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    22 ≤ m ∧ m ≤ 35 ∧ sch.n = m - 7 ∧ sch.k0 = (if m = 28 then 4 else if m = 29 then 5 else 6) ∧
      sch.levels.size = (if m ≤ 24 then 3 else if m ≤ 27 then 4 else if m ≤ 30 then 5 else if m ≤ 33 then 6 else 7) := by
  unfold fast100 at h
  by_cases hm : (m < 22 || m > 35) = true
  · simp [hm] at h
  · simp only [hm] at h
    simp at hm
    simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at h
    simp only [Option.bind_eq_bind, Option.pure_def] at h
    simp at h
    simp only [Option.bind_eq_some_iff] at h
    obtain ⟨s, hs, he⟩ := h
    cases he
    have := forIn_opt_size (fun _ => 1) _ (fun i s t ht => by
      simp only [Option.some.injEq] at ht
      exact ⟨_, ht.symm, by simp⟩) _ _ _ hs
    rw [sum_map_one, List.length_range'] at this
    exact ⟨hm.1, hm.2, rfl, rfl, by simpa using this⟩

def loopG (r i : ℕ) : ℕ := 4 + (if i + 1 < r then 7 else 3)

theorem repRounds_size {m kLog claims : ℕ} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.repRounds m kLog claims = some rs) :
    ∃ sch, fast100 m = some sch ∧ rs.size = (3 + (m - 5) + 1 + (kLog - 6) + 1 + claims + 1) + (2 + sch.k0 + 7) +
      ((List.range' 0 (sch.levels.size - 1)).map (loopG (sch.levels.size - 1))).sum := by
  unfold RecVStar.repRounds at h
  cases hs : fast100 m with
  | none => simp [hs] at h
  | some sch =>
    refine ⟨sch, rfl, ?_⟩
    simp only [hs, Std.Legacy.Range.forIn_eq_forIn_range'] at h
    simp only [Option.bind_eq_bind, Option.pure_def, Option.bind_some] at h
    simp only [Option.bind_eq_some_iff] at h
    obtain ⟨s, hl, he⟩ := h
    cases he
    have := forIn_opt_size (loopG (sch.levels.size - 1)) _ (fun i s t ht => by
      by_cases h4 : s.2 < 4
      · simp [h4] at ht
      · by_cases hi : i + 1 < sch.levels.size - 1
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_⟩
          simp [loopG, hi]
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_⟩
          simp [loopG, hi]) _ _ _ hl
    simp at this
    rw [this]
    simp
    omega

theorem repRounds_closed {m kLog claims : ℕ} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.repRounds m kLog claims = some rs) (hk : 6 ≤ kLog) : ∃ sch, fast100 m = some sch ∧
      rs.size = (3 + (m - 6)) + (3 + (kLog - 6)) + (claims + 1) + (sch.k0 + 5 + 11 * (sch.levels.size - 1)) := by
  obtain ⟨sch, hs, hsz⟩ := repRounds_size h
  obtain ⟨h1, -, -, -, hL⟩ := fast100_sizes hs
  have hsum : ((List.range' 0 (sch.levels.size - 1)).map (loopG (sch.levels.size - 1))).sum =
      11 * (sch.levels.size - 1) - 4 ∧ 1 ≤ sch.levels.size - 1 := by
    rw [hL]; repeat' (first | exact ⟨rfl, by decide⟩ | split)
  refine ⟨sch, hs, ?_⟩
  omega

theorem flockRepRounds_eq {spec : SessionSpec} {st : Stmt} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size) = some rs) (hk : 6 ≤ st.c.kLog) :
    FlockDeploy.repRounds false spec st = rs.size := by
  obtain ⟨sch, hs, hsz⟩ := repRounds_closed h hk
  rw [hsz]
  unfold FlockDeploy.repRounds
  simp only [Bool.false_eq_true, ↓reduceIte, hs]

theorem k_plain {xI : Claims.Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {st : Stmt} {sh : RecVStar.Inner}
    {rs : Array RecVStar.SchedRound} (hset : FlockDeploy.setupOf false xI = some (s, spec, #[st]))
    (hm : sh.m = st.m) (hk : sh.kLog = st.c.kLog) (hc : sh.claims = 2 + spec.points * st.regions.size)
    (hk6 : 6 ≤ st.c.kLog) (hrs : RecVStar.repRounds sh.m sh.kLog sh.claims = some rs) :
    (FlockDeploy.V false).k xI = 2 + 2 * rs.size := by
  rw [hm, hk, hc] at hrs
  have he := flockRepRounds_eq hrs hk6
  simp only [FlockDeploy.V, hset]
  simp [FlockDeploy.rounds, he]

end RecDeploy
