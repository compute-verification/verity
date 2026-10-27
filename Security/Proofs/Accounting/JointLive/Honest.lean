import Definitions.Accounting.Guarantees.JointBlocks
import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfRemoteLinePeriod
import Proofs.Core.Circuit
import Proofs.Pouw.Prob

/-!
# The honest and nontrivial lemmas of `JointHorizonBlocks` and `BandwidthOfRemoteLinePeriod`

At the honest deployed horizon (`Accounting.Honest.honest`) the committed transcript is the circuit's evaluation, so no
unit is wrong (`wrong_eval`); the node's step is the same at every tick, so each account's charges over a block are the
block's length times the step's (`read_const`, `ops_const`, `held_const`), which are its certificates; the wipe keeps
nothing; and the capacity account holds all of PoUS's store at every tick, so no memory is undeclared. Each hypothesis's
bad event is empty, so its probability is 0.

At the idle horizon (`Accounting.Honest.idleBlocks`) the bandwidth account reads nothing, so PoUB's claim at the honest
node's certificate `L · 38,400` fails on every coin of every block of positive length (`idle_poub_fails`) while its
verdict accepts: the bad event has probability 1.
-/

namespace Accounting.Honest

open Verity

/-- A unit recomputed from the circuit's evaluation is the evaluation. -/
theorem localEval_eval {C : Circuit} {n : ℕ} (P : Partition C n) (a : C.Transcript) (u : Fin n) :
    P.localEval (C.eval a) u = C.eval a := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  by_cases hu : P.unit g = some u
  · have hni : C.isInput g = false := by
      cases h : C.isInput g
      · rfl
      · rw [(P.input_iff g).2 h] at hu
        exact absurd hu (by simp)
    rw [P.localEval_of_mem _ hu, C.eval_op a hni]
    exact C.op_congr g _ _ fun x hx => ih x (C.reads_lt g x hx)
  · exact P.localEval_of_not_mem _ hu

/-- The circuit's evaluation has no wrong unit. -/
theorem wrong_eval {C : Circuit} {n : ℕ} (P : Partition C n) (a : C.Transcript) : P.wrong (C.eval a) = ∅ :=
  Finset.eq_empty_of_forall_notMem fun u hu =>
    (P.mem_wrong.1 hu) fun g _ _ => (congrFun (localEval_eval P a u) g).symm

theorem ops_const (st : Step Unit (Fin 3)) (w : Window) (a : Fin 3) (o : Op) :
    Schedule.ops (fun _ => st) w a o = w.len * st.ops a o := by
  simp [Schedule.ops, Window.ticks]

theorem read_const (st : Step Unit (Fin 3)) (w : Window) (a : Fin 3) :
    Schedule.read (fun _ => st) w a = w.len * st.read a := by
  simp [Schedule.read, Window.ticks]

theorem held_const (st : Step Unit (Fin 3)) (w : Window) (a : Fin 3) :
    Schedule.held (fun _ => st) w a = w.len * st.mem a := by
  simp [Schedule.held, Window.ticks]

/-- Every audit's claim holds on the honest node's block at its certificates, at every coin. -/
theorem period_holds (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n)
    (v : C.Transcript) (B L E : ℕ) (p : Price) (b : Fin B) (x : Audit) (ξ : Fin B × Audit → Bool) :
    ((honest rate cap P v B L E).block b).Holds (certs rate L E p) p x ξ := by
  cases x
  · show ((P.wrong (C.eval v)).card : ℝ) ≤ 0 * n
    simp [wrong_eval]
  · show ((∑ o, p o * (L * rate o) : ℕ) : ℝ) ≤
      ((Schedule.work (fun _ => step (h200 rate cap)) ⟨(b : ℕ) * L, L⟩ 0 p : ℕ) : ℝ)
    refine Nat.cast_le.2 (le_of_eq ?_)
    simp only [Schedule.work, ops_const]
    simp [step, h200]
  · show (weights : ℝ) * L ≤ ((Schedule.held (fun _ => step (h200 rate cap)) ⟨(b : ℕ) * L, L⟩ 2 : ℕ) : ℝ)
    rw [held_const]
    simp [step, mul_comm]
  · show ((L * 38400 : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => step (h200 rate cap)) ⟨(b : ℕ) * L, L⟩ 1 : ℕ) : ℝ)
    rw [read_const]
    simp [step, h200]
  · exact le_rfl

/-- The honest node's capacity account holds all of PoUS's store at every tick: no memory is undeclared. -/
theorem memory (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    (B L E : ℕ) (b : Fin B) (ξ : Fin B × Audit → Bool) : (honest rate cap P v B L E).MemoryTerm b ξ := by
  intro t _
  have h0 : (honest rate cap P v B L E).undeclared b ξ t = 0 := by
    show (weights : ℝ) - ((if (2 : Fin 3) = 2 then weights else 0 : ℕ) : ℝ) = 0
    simp
  rw [h0]
  positivity

/-- The idle node reads nothing. -/
theorem idle_read (w : Window) (a : Fin 3) : Schedule.read (fun _ => idle) w a = 0 := by
  simp [Schedule.read, idle]

/-- PoUB's claim at the honest node's certificate fails on every coin of an idle block of positive length. -/
theorem idle_poub_fails (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n)
    (v : C.Transcript) (B L E : ℕ) (p : Price) (hL : 0 < L) (b : Fin B) (ξ : Fin B × Audit → Bool) :
    ¬ ((idleBlocks rate cap P v B L E).block b).Holds (certs rate L E p) p .poub ξ := by
  intro h
  have h' : ((L * 38400 : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => idle) ⟨(b : ℕ) * L, L⟩ 1 : ℕ) : ℝ) := h
  rw [idle_read, Nat.cast_zero] at h'
  exact absurd h' (not_le.2 (Nat.cast_pos.2 (Nat.mul_pos hL (by norm_num))))

end Accounting.Honest

namespace Accounting.SecurityProofs

open Accounting

theorem JointHorizonBlocksHonest : Accounting.Guarantees.JointHorizonBlocksHonest := by
  intro rate cap C n P v B L E p
  refine ⟨fun _ => ⟨fun _ _ => trivial, fun _ => trivial⟩, fun b x ξ => ?_, fun b => ?_, fun b => ?_⟩
  · exact le_of_eq (Pouw.pr_eq_zero fun y h => h.2 (Honest.period_holds rate cap P v B L E p b x _))
  · exact le_of_eq (Pouw.pr_eq_zero fun ξ h => h.2 (Nat.zero_le _))
  · exact le_of_eq (Pouw.pr_eq_zero fun ξ h => h.2 (Honest.memory rate cap P v B L E b ξ))

theorem JointHorizonBlocksNontrivial : Accounting.Guarantees.JointHorizonBlocksNontrivial := by
  intro rate cap C n P v B L E p δ εWipe δPous hB hL hlt hle
  refine absurd hle (not_le.2 (lt_of_lt_of_le hlt (le_of_eq (Pouw.pr_eq_one ?_).symm)))
  intro ξ
  exact Or.inl ⟨⟨0, hB⟩, .poub, trivial, Honest.idle_poub_fails rate cap P v B L E p hL _ ξ⟩

namespace Bridge

theorem BandwidthOfRemoteLinePeriodHonest : Accounting.Guarantees.Bridge.BandwidthOfRemoteLinePeriodHonest := by
  intro rate cap C n P v B E p b
  refine ⟨⟨?_⟩, le_of_eq (Pouw.pr_eq_zero fun _ h => lt_irrefl _ h.2)⟩
  refine
    { coin := Equiv.refl Bool
      accepts_iff := fun _ _ => Iff.rfl
      cert_le := fun _ _ _ => le_rfl
      read_le := fun ξ y => ?_ }
  show ((Honest.sessionBits : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => Honest.step (Honest.h200 rate cap))
    ⟨(b : ℕ) * Honest.sessionSpan, Honest.sessionSpan⟩ 1 : ℕ) : ℝ)
  rw [Honest.read_const]
  refine Nat.cast_le.2 ?_
  simp [Honest.step, Honest.h200, Honest.sessionBits, Honest.sessionSpan]

theorem BandwidthOfRemoteLinePeriodNontrivial :
    Accounting.Guarantees.Bridge.BandwidthOfRemoteLinePeriodNontrivial := by
  intro rate cap C n P v B L E p b δ hL hδ hS
  have h1 := hS fun _ => false
  rw [Pouw.pr_eq_one fun y => ⟨trivial, Honest.idle_poub_fails rate cap P v B L E p hL b _⟩] at h1
  linarith

end Bridge

end Accounting.SecurityProofs
