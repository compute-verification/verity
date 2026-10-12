import Mathlib
import Definitions.Accounting.Confidence
import Proofs.Accounting.Actual.Split

/-!
# The held-tick channel is closed by content-causality

Memory accounting's `held_tick` (`note:ra-ledger-lean/20261012T0917Z-finding-held-tick-memory`) is a strategy over one
coin `b` at leaves `true` (the verifier) and `false` (`X`) in which a width-0 gate held at `X` sits at a tick that is
the coin, and `X` later computes `b` from it while nothing reaches it. Under v2.2's causality, which read the state *as
seen* (ticks included), it was causal from every tick at `M_X = 0`. The circuit at coin `b`:
* `g0`, the verifier's input; `g1`, at the verifier at tick `1 + b`, width 0, reading `g0`; `g2`, at the verifier at
  tick 10, reading `g0`;
* `g3`, at `X` at tick `2 + b`, width 0, reading `g1`; `g4`, at `X` at tick 6, width 1, reading `g3`, value `b`.

v2.4 reads the state's content, its values without their positions. Whatever the leaves hold (`Holding`), `X`'s state at
tick 5 is `g3` alone, whose content is the empty string at both coins, and nothing crosses into `X` during `[5, 7)`;
but `g4`'s value is the coin. So no holding makes the strategy causal from every tick (`held_tick_closed`), and the
circuit admits a holding (`tick_holding`), so the closure turns on `CausalFrom`.
-/

namespace Accounting.SecurityProofs.Actual

open Finset

/-- The actual circuit of `held_tick` at coin `b`. Every gate is the constant `b` at its width. -/
abbrev tickPlaced (b : Bool) : Placed Bool where
  N := 5
  w g := if g.val = 1 ∨ g.val = 3 then 0 else 1
  isInput g := decide (g = 0)
  reads g := if g = 1 ∨ g = 2 then {0} else if g = 3 then {1} else if g = 4 then {3} else ∅
  reads_lt g a ha := by fin_cases g <;> simp_all
  input_reads g hg := by fin_cases g <;> simp_all
  op g _ := BitVec.ofNat _ b.toNat
  op_congr _ _ _ _ := rfl
  outputs := ∅
  x g := decide (g.val < 3)
  t g := if g.val = 1 then 1 + b.toNat else if g.val = 2 then 10 else if g.val = 3 then 2 + b.toNat else
    if g.val = 4 then 6 else 0

/-- Its run at coin `b`, with what each leaf holds given by `held`. -/
abbrev tickRun (b : Bool) (held : Bool → ℕ → Finset (Fin 5)) (h : (tickPlaced b).Holding held) :
    Actual.Run Bool where
  A := tickPlaced b
  val _ := BitVec.ofNat _ b.toNat
  faithful _ _ := rfl
  held := held
  holding := h

/-- Whatever the leaves hold, `X`'s state at tick 5 is `g3`: it is held there until `g4` reads it, and it is the only
gate of `X` before tick 5. -/
theorem tick_state (b : Bool) (held : Bool → ℕ → Finset (Fin 5)) (h : (tickPlaced b).Holding held) :
    (tickRun b held h).state {false} 5 = {3} := by
  have hb := Bool.toNat_le b
  ext a
  simp only [Actual.Run.state, mem_filter, mem_univ, true_and, mem_singleton]
  constructor
  · rintro (⟨y, hy, ha⟩ | ⟨ha, hx⟩)
    · obtain rfl : y = false := by simpa using hy
      have := h.1 false 5 a ha
      fin_cases a <;> simp_all
    · obtain ⟨-, g, hg, hag⟩ := (mem_into _).1 ha
      have hg' := (mem_region _).1 hg
      fin_cases g <;> fin_cases a <;> simp_all
  · rintro rfl
    refine Or.inl ⟨false, by simp, h.2.2 4 3 (by simp) (by simp) 5 (Or.inr ?_) (by simp)⟩
    show 2 + b.toNat < 5
    omega

/-- **`held_tick` is closed** (v2.4): no holding makes it causal from every tick. `X`'s state at 5 has the empty content
at both coins and nothing crosses into `X` during `[5, 7)`, yet `g4`'s value is the coin. -/
theorem held_tick_closed (held : Bool → Bool → ℕ → Finset (Fin 5)) (h : ∀ b, (tickPlaced b).Holding (held b)) :
    ¬CausalFrom fun b => tickRun b (held b) (h b) := by
  intro hc
  have hx : ∀ b, (tickRun b (held b) (h b)).crossing (↑({false} : Finset Bool))ᶜ ↑({false} : Finset Bool)
      (Set.Ico 5 7) = ∅ := fun b => by
    ext p
    simp only [Actual.Run.crossing, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    intro g a ha hxa hxg hg
    have hb := Bool.toNat_le b
    fin_cases g <;> fin_cases a <;> simp_all
    omega
  have key := congrArg Prod.fst
    (hc {false} 5 7 false true
      (by simp only [Actual.Run.content, tick_state]; simp [Actual.chunk, Actual.position, Actual.bits, Actual.Run.seen])
      (by simp only [hx]))
  have : ((6 : ℕ), false, 1, 1) ∈ ((tickRun true (held true) (h true)).acts ↑({false} : Finset Bool)
      (Set.Ico 5 7)).1 := ⟨4, by simp, by simp, rfl⟩
  rw [← key] at this
  obtain ⟨g, hg, ht, he⟩ := this
  fin_cases g <;> simp_all [Actual.Run.seen]

/-- The circuit admits a holding at each coin: the verifier holds its input until tick 10, and `X` holds `g3` from the
tick after its placement to `g4`'s. -/
theorem tick_holding (b : Bool) :
    (tickPlaced b).Holding fun y t =>
      if y = true then (if t ≤ 10 then {0} else ∅) else if 2 + b.toNat < t ∧ t ≤ 6 then {3} else ∅ := by
  have hb := Bool.toNat_le b
  refine ⟨fun y t a ha => ?_, fun a ha => ?_, fun g a ha hx s hs hst => ?_⟩
  · simp only at ha
    split_ifs at ha <;> fin_cases a <;> simp_all
  · fin_cases a <;> simp_all
  · fin_cases g <;> fin_cases a <;> simp_all
    split_ifs <;> simp_all
    omega

end Accounting.SecurityProofs.Actual
