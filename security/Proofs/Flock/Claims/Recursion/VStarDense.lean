module

public import Proofs.Flock.Claims.Recursion.VStarAccept
public import Proofs.Flock.Claims.Recursion.VStarFit
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.C3.Reps.AtList
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.VStarFitHeld

/-! V\*'s parts at the dense form are levels, algebra parts, caps and the link table's part (`partAt_dense`), and the stager accepts each (`hok_dense`, `hok_shaped`). -/

@[expose] public section

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

theorem sparse_none_of_dense {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) : ∃ r, readX x = some r ∧ r.shape.sparse = none := by
  obtain ⟨r, hr, -, rfl⟩ := parse_read hx
  refine ⟨r, hr, ?_⟩
  have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hf.2
  cases h : r.shape.sparse with
  | none => rfl
  | some sp => rw [h] at h2; cases h2

theorem partAt_dense {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) {s : ℕ} {sv : _root_.VStar.Stmt} {c : Circuit}
    (hp : partAt x s = some (sv, c)) :
    (∃ l L H n, sv = .recOpen l L H n) ∨ (∃ j nm p, sv = .inner j nm p) ∨ (∃ l N, sv = .cap l N) ∨
      ∃ UB PB N, sv = .link UB PB N := by
  obtain ⟨r, hr, hsp⟩ := sparse_none_of_dense hx hf
  obtain ⟨ss, hss, -⟩ := readX_parts hr
  have hsv : ss[s]? = some sv := by
    have h := (partAt_some hr hp).1
    rw [hss] at h
    exact h
  obtain ⟨sch, ps, C, Ks, -, hC, hK, hget⟩ := FlockC3.Reps.getElem?_list hss
  rw [hget s, (FlockC3.VStarList.candidate3_ok hC).1 hsp] at hsv
  split_ifs at hsv
  · exact .inl ⟨_, _, _, _, (Option.some.inj hsv).symm⟩
  · exact .inr (.inl ⟨_, _, _, (Option.some.inj hsv).symm⟩)
  · rw [Array.empty_append] at hsv
    exact .inr (.inr (hK _ (Array.mem_iff_getElem?.2 ⟨_, hsv⟩)))

theorem hok_dense {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)}
    {st : Claims.Recursion.CompiledStatement} {s : ℕ} (hx : parse x = some st) (hf : Claims.Recursion.Form.Dense st.form)
    (hL : ∀ l L H n c src, partAt x s = some (.recOpen l L H n, c) → srcOf I x w ρ hist = some src →
      LevelInFits src l L H n)
    (hA : ∀ (j : ℕ) (nm : String) (p : _root_.VStar.Part) c src, partAt x s = some (.inner j nm p, c) → srcOf I x w ρ hist = some src →
      AlgInFits src p.S.ports)
    (hC : ∀ l N c src, partAt x s = some (.cap l N, c) → srcOf I x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.capIn src l N = .ok pin) :
    ∀ sv c src, partAt x s = some (sv, c) → srcOf I x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin := by
  refine hok_of_fits fun sv c src hp hs => ?_
  rcases partAt_dense hx hf hp with ⟨l, L, H, n, rfl⟩ | ⟨j, nm, p, rfl⟩ | ⟨l, N, rfl⟩ | ⟨UB, PB, N, rfl⟩
  · exact hL l L H n c src hp hs
  · exact hA j nm p c src hp hs
  · exact hC l N c src hp hs
  · trivial

def IrShaped (Ir : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : (Ir.fit x).Tape) (hist : List (Msg × Msg)) :
    Prop :=
  (∃ o ∈ ((Ir.fit x).rest w ρ.1 hist).openings, o.rep = 0) ∧
  (∃ o ∈ ((Ir.fit x).rest w ρ.1 hist).openings, o.rep = 1) ∧
  ∀ s l L H n c, partAt x s = some (.recOpen l L H n, c) →
    (((Ir.fit x).rest w ρ.1 hist).openings.filter (·.level == l)).size = n ∧
    (∀ q ∈ ((Ir.fit x).rest w ρ.1 hist).openings, q.level = l → q.row.size = L ∧ q.sibs.size = H) ∧
    ∀ src, srcOf (Ir.fit x) x w ρ hist = some src → ∀ T, _root_.VStar.In.topsAt src l = .ok T →
      ∀ q ∈ ((Ir.fit x).rest w ρ.1 hist).openings, q.level = l → T.first[q.rep]! + q.pos / 2 ^ H < T.rows.size

theorem srcOf_openings {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)}
    {src : _root_.VStar.In.Src} (hs : srcOf I x w ρ hist = some src) :
    src.openings = (I.rest w ρ.1 hist).openings := by
  cases hr : readX x with
  | none => simp [srcOf, hr] at hs
  | some r =>
    cases hrs : repSched r.shape with
    | none => simp [srcOf, hr, hrs] at hs
    | some rs => rw [srcOf_eq hr hrs hs]

theorem levelFits_of_shaped {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape}
    {hist : List (Msg × Msg)} (hI : IrShaped Ir x w ρ hist) {s l L H n : ℕ} {c : Circuit} {src : _root_.VStar.In.Src}
    (hp : partAt x s = some (.recOpen l L H n, c)) (hs : srcOf (Ir.fit x) x w ρ hist = some src)
    (hT : ∃ T, _root_.VStar.In.topsAt src l = .ok T) : LevelInFits src l L H n := by
  obtain ⟨hn, hsz, hpos⟩ := hI.2.2 s l L H n c hp
  have ho := srcOf_openings hs
  obtain ⟨T, hT⟩ := hT
  refine ⟨?_, fun q hq => ?_, T, hT, fun q hq => ?_⟩
  · rw [ho]; exact hn
  · rw [ho] at hq; exact hsz q hq
  · rw [ho] at hq; exact hpos src hs T hT q hq

theorem algFits_of_shaped {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape}
    {hist : List (Msg × Msg)} (hI : IrShaped Ir x w ρ hist) {src : _root_.VStar.In.Src}
    (hs : srcOf (Ir.fit x) x w ρ hist = some src) {ports : List (String × ℕ)}
    (hN : ∀ xr rs, readX x = some xr → repSched xr.shape = some rs → ∀ nm ∈ ports.map Prod.fst,
      nm = "acc0" ∨ nm = "acc1" ∨ ∃ i, RecVStar.mIndex nm = some i ∧ i < (RecVStar.commitRounds rs).length) :
    AlgInFits src ports := by
  cases hr : readX x with
  | none => simp [srcOf, hr] at hs
  | some r =>
    cases hrs : repSched r.shape with
    | none => simp [srcOf, hr, hrs] at hs
    | some rs => exact algFits_srcOf hr hrs hs hI.1 hI.2.1 (hN r rs hr hrs)

theorem hok_shaped {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape} {hist : List (Msg × Msg)}
    {st : Claims.Recursion.CompiledStatement} {s : ℕ} (hx : parse x = some st) (hf : Claims.Recursion.Form.Dense st.form)
    (hI : IrShaped Ir x w ρ hist)
    (hT : ∀ l L H n c src, partAt x s = some (.recOpen l L H n, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ T, _root_.VStar.In.topsAt src l = .ok T)
    (hN : ∀ (j : ℕ) (nm : String) (p : _root_.VStar.Part) c, partAt x s = some (.inner j nm p, c) → ∀ xr rs, readX x = some xr →
      repSched xr.shape = some rs → ∀ pn ∈ p.S.ports.map Prod.fst,
        pn = "acc0" ∨ pn = "acc1" ∨ ∃ i, RecVStar.mIndex pn = some i ∧ i < (RecVStar.commitRounds rs).length)
    (hC : ∀ l N c src, partAt x s = some (.cap l N, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.capIn src l N = .ok pin) :
    ∀ sv c src, partAt x s = some (sv, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin :=
  hok_dense hx hf (fun l L H n c src hp hs => levelFits_of_shaped hI hp hs (hT l L H n c src hp hs))
    (fun j nm p c _ hp hs => algFits_of_shaped hI hs (hN j nm p c hp)) hC

end RecDeploy.VStarBuild
