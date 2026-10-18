import Models.Hash.Guarantees
import Mathlib.Tactic
import Mathlib.Data.Fintype.Pigeonhole
import Mathlib.Data.Fintype.EquivFin

/-! Red-team probes for #2474 (hash lane). Scratch, never for a PR. -/

open Hash

namespace HashRedteam

open Classical in
theorem pr_one {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) (h : ∀ a, p a) : Pr p = 1 := by
  unfold Pr
  rw [Finset.filter_true_of_mem (fun a _ => h a), Finset.card_univ]
  exact div_self (by exact_mod_cast Fintype.card_ne_zero)

instance : Finite UInt8 := Finite.of_injective (fun x : UInt8 => x.toFin) fun _ _ h => UInt8.toFin_inj.1 h

/-- Every keyed family has, for every key, a collision (pigeonhole: messages are infinite, digests finite). -/
theorem coll_exists {K : Type} (h : K → Msg → Digest) (k : K) : ∃ u v : Msg, u ≠ v ∧ h k u = h k v :=
  Finite.exists_ne_map_eq_of_infinite (h k)

/-- **P1. `KeyedWindowCR` is false at some finder, for every keyed family `h` and every `lam`**: the finder that makes
no evaluation and outputs a collision of `h k` chosen per key (any function of the key, as `Keyed.Finder.prog` allows)
succeeds with probability 1, against the bound `0² / 2^(2·lam+1) = 0`. So the Lean form cannot separate full SHA-512
from a linear universal family: both fail it at this finder, and both may pass it at `reduce F V` for a given `V`. -/
theorem keyedWindowCR_refuted {K : Type} [Fintype K] [Nonempty K] (h : K → Msg → Digest) (lam : ℕ) :
    ∃ F : Keyed.Finder K, ¬ Assumptions.KeyedWindowCR h lam F := by
  choose u v hne heq using coll_exists h
  refine ⟨{ S := Unit, prog := fun k _ => .done (some (u k, v k)) }, fun H => ?_⟩
  have hb := H 0 (fun _ _ => by simp [Keyed.Prog.evals])
  have hPr : Pr (fun p : K × Unit => ∃ a b, (Keyed.Prog.done (some (u p.1, v p.1)) : Keyed.Prog _).run (h p.1)
      = some (a, b) ∧ Collision (h p.1) a b) = 1 :=
    pr_one _ fun p => ⟨u p.1, v p.1, rfl, hne p.1, heq p.1⟩
  simp only at hb
  rw [hPr] at hb
  norm_num at hb

/-- **P2. `SISHard` is false at a finder of no work when `β = hi − lo ≥ q`** (nothing in `Sis.Params` refuses it): the
vector `(q, …, q)` is a solution of every matrix. -/
theorem sisHard_refuted (P : Sis.Params) (hm : 0 < P.m) (hβ : (P.q : ℤ) ≤ P.hi - P.lo) (hb : 1 ≤ P.bits) :
    ∃ F : Sis.Finder P, ¬ Assumptions.SISHard P F := by
  have hq : (0 : ℤ) < P.q := by have := P.one_lt_q; omega
  let z : Sis.Digits P := fun _ => (P.q : ℤ)
  have hz : ∀ A, Sis.Solution P A z := by
    intro A
    refine ⟨fun h0 => ?_, fun j => ?_, ?_⟩
    · have := congrFun h0 ⟨0, hm⟩
      simp [z] at this
      omega
    · simp only [z, abs_of_pos hq]; exact hβ
    · funext i
      simp [Sis.image, z]
  refine ⟨{ S := Unit, out := fun _ _ => some z, cost := fun _ _ => 0 }, fun H => ?_⟩
  have hPr : Pr (fun s : Sis.Mat P × Unit => ∃ w, (some z : Option (Sis.Digits P)) = some w ∧ Sis.Solution P s.1 w)
      = 1 :=
    pr_one _ fun s => ⟨z, rfl, hz s.1⟩
  unfold Assumptions.SISHard at H
  simp only at H
  rw [hPr] at H
  have hEx : Ex (fun _ : Sis.Mat P × Unit => ((0 : ℕ) : ℝ)) = 0 := by simp [Ex]
  rw [hEx, zero_add] at H
  have h2 : (2 : ℝ) ≤ 2 ^ P.bits := by
    calc (2 : ℝ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ P.bits := pow_le_pow_right₀ (by norm_num) hb
  have hpos : (0 : ℝ) < 2 ^ P.bits := by positivity
  have : (1 : ℝ) / 2 ^ P.bits < 1 := (div_lt_one hpos).2 (by linarith)
  linarith

/-- **P3. The window game doesn't bound precomputation.** For every keyed family, format and depth, a prover of `q = 0`
evaluations in the window (its program a function of the key, as `Keyed.Prover.prog` allows, holding a collision of
the leaf hash under every key) outputs a binding break with probability 1. `EpochBinding`'s conclusion at it is
`1 ≤ (2·(depth+1))² / 2^(2·lam+1)`, false at any useful `lam`, so `KeyedWindowCR` at `Keyed.reduce F V` must be false:
the game is per-key collision resistance with unbounded advice, and the benefit of keying (work before the key is
drawn doesn't count) is not in it. -/
theorem epochBinding_precomputed {K : Type} [Fintype K] [Nonempty K] (h : K → Msg → Digest) (F : Keyed.Format) :
    ∃ V : Keyed.Prover K 0, Pr (fun p : K × V.Ω => Keyed.Break F (h p.1) ((V.prog p.1 p.2).run (h p.1))) = 1 := by
  have hc : ∀ k, ∃ u v : Msg, u ≠ v ∧ h k (Keyed.leafIn F u) = h k (Keyed.leafIn F v) :=
    fun k => Finite.exists_ne_map_eq_of_infinite (fun x => h k (Keyed.leafIn F x))
  choose u v hne heq using hc
  let sibs : List Digest := List.replicate F.depth (fun _ => 0)
  refine ⟨{ Ω := Unit
            prog := fun k _ => .done (Path.root (h k) F.nodeTag 0 (Keyed.leafIn F (u k)) sibs, 0, ⟨u k, sibs⟩, ⟨v k, sibs⟩)
            bounded := fun _ _ _ => by simp [Keyed.Prog.evals] }, ?_⟩
  refine pr_one _ fun p => ?_
  simp only [Keyed.Break, Keyed.accepts, Path.accepts, Keyed.Prog.run]
  refine ⟨⟨by simp [sibs], Nat.two_pow_pos _, trivial⟩, ⟨by simp [sibs], Nat.two_pow_pos _, ?_⟩, hne p.1⟩
  simp only [Path.root]
  rw [heq p.1]

end HashRedteam

#print axioms HashRedteam.keyedWindowCR_refuted
#print axioms HashRedteam.sisHard_refuted
#print axioms HashRedteam.epochBinding_precomputed
