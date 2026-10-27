import Definitions.Accounting.Guarantees.JointLive.Honest
import Proofs.Accounting.JointLive

/-!
# The honest and nontrivial lemmas of `JointHorizonBlocksLive` and `Bridge.BandwidthOfRemoteLineLive`

At the honest deployed horizon (`Accounting.Honest.honestBlocks`) the extractor reads the circuit's evaluation, which
the commitment opens at every gate, so no unit is unbound (`Accounting.SecurityProofs.Live.unbound_eval`); the node's
step is the same at every tick, so each account's charges over a block are the block's length times the step's
(`read_const`, `held_const`), which are its certificates; the wipe keeps nothing; and the capacity account holds all of
PoUS's store at every tick, so no memory is undeclared. Each hypothesis's bad event fails on every run, so its
probability is 0 at every strategy (`prob_eq_zero_of_none`).

At the idle horizon (`Accounting.Honest.idleBlocks`) the bandwidth account reads nothing, so PoUB's claim at the honest
node's certificate `L · 38,400` fails on every run of every block of positive length (`idle_poub_fails`) while its
verdict accepts: the bad event holds on every run, so its probability is 1 at every strategy.
-/

namespace Accounting.Honest

open Verity Verity.Game

/-- An event that fails on every run has probability 0 at every strategy. -/
theorem prob_eq_zero_of_none {O : Type} {p : O → Prop} (h : ∀ o, ¬ p o) (G : Game O) (s : Strategy G) :
    prob p G s = 0 :=
  le_antisymm ((prob_mono (fun o ho => h o ho) G s).trans (prob_false G s).le) (zero_le _)

theorem read_const (st : Step Unit (Fin 3)) (w : Window) (a : Fin 3) :
    Schedule.read (fun _ => st) w a = w.len * st.read a := by
  simp [Schedule.read, Window.ticks]

theorem held_const (st : Step Unit (Fin 3)) (w : Window) (a : Fin 3) :
    Schedule.held (fun _ => st) w a = w.len * st.mem a := by
  simp [Schedule.held, Window.ticks]

/-- Every subprotocol's claim holds on the honest node's block at its certificates, on every run of every strategy. -/
theorem period_holds (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    {O : Type} (G : Game O) (L E b : ℕ) (s : Strategy G) (x : Audit) (o : O) :
    (honestPeriod rate cap P v G L E b).Holds (certs C L E) s x o := by
  cases x
  · show IntegrityHolds P (fun g y => y = C.eval v g) (C.eval v) 0
    rw [IntegrityHolds, SecurityProofs.Live.unbound_eval P (fun g y => y = C.eval v g) v fun _ => rfl]
    simp
  · exact le_rfl
  · show (weights : ℝ) * L ≤ ((Schedule.held (fun _ => step (h200 rate cap)) ⟨b * L, L⟩ 2 : ℕ) : ℝ)
    rw [held_const]
    simp [step, mul_comm]
  · show ((L * 38400 : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => step (h200 rate cap)) ⟨b * L, L⟩ 1 : ℕ) : ℝ)
    rw [read_const]
    simp [step, h200]
  · exact le_rfl

/-- The honest node's capacity account holds all of PoUS's store at every tick: no memory is undeclared. -/
theorem memory (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    {O : Type} (G : Game O) (B L E : ℕ) (s : Strategy G) (b : Fin B) (o : O) :
    (honestBlocks rate cap P v G B L E).MemoryTerm s b o := by
  intro t _
  have h0 : (weights : ℝ) - (((step (h200 rate cap)).mem 2 : ℕ) : ℝ) = 0 := by simp [step]
  exact h0.le.trans (by positivity)

/-- The idle node reads nothing. -/
theorem idle_read (w : Window) (a : Fin 3) : Schedule.read (fun _ => idle) w a = 0 := by
  simp [Schedule.read, idle]

/-- PoUB's claim at the honest node's certificate fails on every run of an idle block of positive length. -/
theorem idle_poub_fails (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n)
    (v : C.Transcript) {O : Type} (G : Game O) (L E b : ℕ) (hL : 0 < L) (s : Strategy G) (o : O) :
    ¬ (idlePeriod rate cap P v G L E b).Holds (certs C L E) s .poub o := by
  intro h
  have h' : ((L * 38400 : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => idle) ⟨b * L, L⟩ 1 : ℕ) : ℝ) := h
  rw [idle_read, Nat.cast_zero] at h'
  exact absurd h' (not_le.2 (Nat.cast_pos.2 (Nat.mul_pos hL (by norm_num))))

end Accounting.Honest

namespace Accounting.SecurityProofs

open Accounting Verity.Game
open scoped ENNReal

theorem JointHorizonBlocksLive_honest : Accounting.Guarantees.JointHorizonBlocksLive_honest := by
  intro rate cap C n P v O G B L E
  refine ⟨fun _ => ⟨fun _ _ => trivial, fun _ => trivial⟩, fun b a s => ?_, fun b s => ?_, fun b s => ?_⟩
  · exact Honest.prob_eq_zero_of_none (fun o h => h.2 (Honest.period_holds rate cap P v G L E b s a o)) G s
  · exact Honest.prob_eq_zero_of_none (fun o h => h.2 (Nat.zero_le _)) G s
  · exact Honest.prob_eq_zero_of_none (fun o h => h.2 (Honest.memory rate cap P v G B L E s b o)) G s

theorem JointHorizonBlocksLive_nontrivial : Accounting.Guarantees.JointHorizonBlocksLive_nontrivial := by
  intro rate cap C n P v O G B L E s e hB hL he hle
  refine absurd hle (not_le.2 (lt_of_lt_of_le he ?_))
  calc (1 : ℝ≥0∞) = prob (fun _ => True) G s := (prob_true G s).symm
    _ ≤ _ := prob_mono (fun o _ => Or.inl ⟨⟨0, hB⟩, .poub, trivial,
        Honest.idle_poub_fails rate cap P v G L E 0 hL s o⟩) G s

namespace Bridge

theorem BandwidthOfRemoteLineLive_honest : Accounting.Guarantees.Bridge.BandwidthOfRemoteLineLive_honest := by
  intro rate cap C n P v O G E b
  refine ⟨⟨fun _ => Iff.rfl, fun _ _ => le_rfl, fun s o => ?_⟩,
    fun s => Honest.prob_eq_zero_of_none (fun _ h => lt_irrefl _ h.2) G s⟩
  show ((Honest.sessionBits : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => Honest.step (Honest.h200 rate cap))
    ⟨b * Honest.sessionSpan, Honest.sessionSpan⟩ 1 : ℕ) : ℝ)
  rw [Honest.read_const]
  refine Nat.cast_le.2 ?_
  simp [Honest.step, Honest.h200, Honest.sessionBits, Honest.sessionSpan]

theorem BandwidthOfRemoteLineLive_nontrivial :
    Accounting.Guarantees.Bridge.BandwidthOfRemoteLineLive_nontrivial := by
  intro rate cap C n P v O G L E b s e hL he hle
  refine absurd hle (not_le.2 (lt_of_lt_of_le he ?_))
  calc (1 : ℝ≥0∞) = prob (fun _ => True) G s := (prob_true G s).symm
    _ ≤ _ := prob_mono (fun o _ => ⟨trivial, Honest.idle_poub_fails rate cap P v G L E b hL s o⟩) G s

end Bridge

end Accounting.SecurityProofs
