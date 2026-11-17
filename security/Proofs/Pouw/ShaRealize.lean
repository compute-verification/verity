module

public import Definitions.Pouw.Guarantees.ShaRealize
public import Proofs.Pouw.MerkleExtract
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
import Proofs.Pouw.RelativeSecurity

/-! Commit-and-open as a theorem: `ExtractAdv` from SHA-512 as a random oracle and its realization. -/

@[expose] public section

namespace Pouw.SecurityProofs.ShaRealize

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

theorem prob_bad_le {P : RoundProto} {k : ℕ} {X D W : Type} [Fintype X] [DecidableEq X] [Fintype D]
    [DecidableEq D] [Inhabited D] (E : Enc X D W) (eo : P.Orc ≃ (X → D)) {q : ℕ} {τ : Fin k → ℕ}
    (L : QueryLog P k X D q τ) (Bad : P.Orc → (Fin k → P.Draw × P.Chk) → Prop)
    (hlink : ∀ o x u, Consistent (L.qry x) (eo o) u → Bad o x →
      (∃ w, ExtractFails E (L.prover x w) u) ∨ ¬ Function.Injective u) :
    prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => Bad ox.1 ox.2) ≤
      ((q * q + ∑ w, q * (2 * τ w + 1) : ℕ) : ℝ) / Fintype.card D := by
  classical
  refine prob_le_of_slices (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => Bad ox.1 ox.2) (by positivity)
    fun x => ?_
  show prob (fun o : P.Orc => Bad o x) ≤ _
  rw [prob_equiv eo]
  refine (prob_mono (q := fun h : X → D => ∃ u, Consistent (L.qry x) h u ∧
      ((∃ w, ExtractFails E (L.prover x w) u) ∨ ¬ Function.Injective u)) fun h hb => ?_).trans ?_
  · obtain ⟨u, hu⟩ := exists_consistent (L.qry x) (L.qry_dep x) h
    exact ⟨u, hu, hlink _ x u (by rwa [Equiv.apply_symm_apply]) hb⟩
  · rw [prob_eq_card]
    have hc := card_fails_oracle_le E (L.qry x) (L.fresh x) (L.prover x)
    have hD : (0 : ℝ) < Fintype.card D := by exact_mod_cast Fintype.card_pos
    have hXD : (0 : ℝ) < Fintype.card (X → D) := by exact_mod_cast Fintype.card_pos
    rw [div_le_div_iff₀ hXD hD]
    convert (Nat.cast_le (α := ℝ)).mpr hc using 1 <;> push_cast <;> rfl

theorem link_noFold {G : Accel} {nf : NoFold} {k : ℕ} {X D : Type} [Inhabited D] (E : Enc X D Word)
    (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) {q : ℕ} {τ : Fin k → ℕ}
    (hopens : ∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c)
    (A : Prover G (noFoldRound nf) k) (Reg : Fin k → Set (noFoldRound nf).Inst)
    (L : QueryLog (noFoldRound nf) k X D q τ) {Adv : Type}
    (adv : nf.Orc → (Fin k → nf.Draw × nf.Chk) → Fin k → Adv)
    (Mw : Fin k → G.Mc.Alg (Adv × nf.Draw) (Fin nf.n → Fin nf.n' → Fin nf.nb → Word))
    (o : nf.Orc) (x : Fin k → nf.Draw × nf.Chk)
    (hroot : ∀ u, Consistent (L.qry x) (eo o) u → ∀ w, rD (A.root o x w) = L.root x w u)
    (hN2 : ∀ u, Consistent (L.qry x) (eo o) u → A.Accepted Reg o x → ∀ w (j : Fin nf.s),
      ∀ y ∈ pts E (pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
        ∃ i, L.qry x i u = y)
    (hwalk : ∀ u, Consistent (L.qry x) (eo o) u → Function.Injective u →
      ∀ w (c : Fin nf.n × Fin nf.n' × Fin nf.nb), G.Mc.eval (Mw w) (adv o x w, (x w).1) c.1 c.2.1 c.2.2 =
        (ext ((L.prover x w).oracle u) E ((L.prover x w).pre u) (L.root x w u) (dirs c)).getD 0)
    (u : Fin q → D) (hu : Consistent (L.qry x) (eo o) u) (hinj : Function.Injective u) (hacc : A.Accepted Reg o x)
    (hbad : ∃ w, ¬ ((noFoldRound nf).ok (A.inst o x w) (x w).1 (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1)) ∧
      (noFoldRound nf).opens o (A.root o x w) (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1)))) :
    ∃ w, ExtractFails E (L.prover x w) u := by
  obtain ⟨w, hw⟩ := hbad
  have hw' : ¬ (noFoldRound nf).ok (A.inst o x w) (x w).1 (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1)) := by
    intro hok
    refine hw ⟨hok, fun j => ⟨((A.ans o x w : Fin nf.s → Word × nf.Path) j).2, ?_⟩⟩
    have e : G.Mc.eval (Mw w) (adv o x w, (x w).1) (nf.cells (x w).2 j).1 (nf.cells (x w).2 j).2.1
        (nf.cells (x w).2 j).2.2 = ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1 :=
      (hok j).trans ((hacc w).2 j).2.symm
    show nf.opens o (A.root o x w) (nf.cells (x w).2 j) (G.Mc.eval (Mw w) (adv o x w, (x w).1)
      (nf.cells (x w).2 j).1 (nf.cells (x w).2 j).2.1 (nf.cells (x w).2 j).2.2)
      ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2
    rw [e]
    exact ((hacc w).2 j).1
  obtain ⟨j, hj⟩ := not_forall.mp hw'
  have hA : nf.opens o (A.root o x w) (nf.cells (x w).2 j) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1
        ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2 ∧
      ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1 = noFoldWord nf (nf.noise (A.inst o x w) (x w).1)
        (nf.cells (x w).2 j).1 (nf.cells (x w).2 j).2.1 (nf.cells (x w).2 j).2.2 := (hacc w).2 j
  obtain ⟨hO, hdirs⟩ := hopens _ _ _ _ _ hA.1
  rw [hroot u hu w] at hO
  have hcov := hN2 u hu hacc w j
  refine ⟨w, pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2, ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
    opens_congr E (fun y hy => ?_) hO, hcov, ?_⟩
  · obtain ⟨i, rfl⟩ := hcov y hy
    rw [show L.qry x i u = (L.prover x w).qry i u from rfl, Prover.oracle_qry]
    exact (hu i).symm
  · rw [hdirs]
    intro hext
    have hwc := hwalk u hu hinj w (nf.cells (x w).2 j)
    rw [show (L.prover x w).root u = L.root x w u from rfl] at hext
    rw [hext, Option.getD_some] at hwc
    exact hj (hwc.trans hA.2)

theorem extract_of_realization {G : Accel} {nf : NoFold} {k : ℕ} {Reg : Fin k → Set (noFoldRound nf).Inst}
    {sel : (noFoldRound nf).Chk → Prop} {Q : ℝ} {X D : Type}
    [Fintype X] [DecidableEq X] [Fintype D] [DecidableEq D] [Inhabited D] (E : Enc X D Word) (eo : nf.Orc ≃ (X → D))
    (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D)) (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool)
    (hopens : ∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c)
    {ξw ξl : G.Res → ℝ} (A : Prover G (noFoldRound nf) k) {q : ℕ} {τ : Fin k → ℕ} {Badv : ℕ}
    (hR : Realization G nf k Reg sel Q E eo rD pl dirs ξw ξl A q τ Badv) :
    ExtractAdvAt G (noFoldRound nf) k Reg sel Q Badv (fun r => k * ξw r + q * ξl r)
      (((q * q + ∑ w, q * (2 * τ w + 1) : ℕ) : ℝ) / Fintype.card D) A := by
  obtain ⟨L, Adv, adv, Mw, Mh, henc, hadv, hMh, huse, hext, hroot, hN2, hwalk⟩ := hR
  refine ⟨Adv, adv, Mw, Mh, henc, hadv, hMh, huse, hext, ?_⟩
  exact prob_bad_le E eo L
    (fun o x => A.Accepted Reg o x ∧ A.WithinCap sel Q o x ∧
      ∃ w, ¬ ((noFoldRound nf).ok (A.inst o x w) (x w).1 (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1)) ∧
        (noFoldRound nf).opens o (A.root o x w) (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1))))
    fun o x u hu hb => (Classical.em (Function.Injective u)).imp (fun hinj => link_noFold E eo rD pl dirs hopens A
      Reg L adv Mw o x (hroot o x hb.2.1) (hN2 o x hb.2.1) (hwalk o x hb.2.1) u hu hinj hb.1 hb.2.2) id

/-- `Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`. -/
theorem ExtractOfRealize : Pouw.Guarantees.ShaRealize.ExtractOfRealize := by
  intro G nf k Reg sel Q X D _ _ _ _ _ E eo rD pl dirs ξw ξl q τ Badv hopens horc hR hG1 hG2 hG3 hP1 hP2 A
  exact extract_of_realization E eo rD pl dirs hopens A (hR hG1 hG2 hG3 horc hP1 hP2 A)

theorem state_congr {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k) (o : P.Orc)
    {x x' : Fin k → P.Draw × P.Chk} :
    ∀ n : ℕ, (∀ v : Fin k, (v : ℕ) < n → x v = x' v) → A.state o x n = A.state o x' n
  | 0, _ => rfl
  | n + 1, h => by
    have ih := state_congr A o n fun v hv => h v (by omega)
    simp only [Prover.state]
    split_ifs with hn
    · rw [ih, h ⟨n, hn⟩ (by simp)]
    · exact ih

theorem commit_congr {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k) (o : P.Orc)
    {x x' : Fin k → P.Draw × P.Chk} (w : Fin k) (h : ∀ v : Fin k, v < w → x v = x' v) :
    G.Mc.eval (A.commit w) (A.commitIn o x w) = G.Mc.eval (A.commit w) (A.commitIn o x' w) := by
  simp only [Prover.commitIn, state_congr A o (w : ℕ) fun v hv => h v hv]

theorem inst_congr {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k) (o : P.Orc)
    {x x' : Fin k → P.Draw × P.Chk} (w : Fin k) (h : ∀ v : Fin k, v < w → x v = x' v) :
    A.inst o x w = A.inst o x' w :=
  congrArg Prod.snd (commit_congr A o w h)

end Pouw.SecurityProofs.ShaRealize
