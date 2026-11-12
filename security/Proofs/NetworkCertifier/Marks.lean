module

public import Definitions.Core.Assumptions
public import Definitions.NetworkCertifier.Guarantees.Marks
meta import Definitions.NetworkCertifier.Marks

/-!
# A mark's execution phase orders it, and what was read by a mark is sent within the tail

`Definitions.NetworkCertifier.Marks`. A cell `j` of a lane that holds at most `N` buckets of egress has `⌊j / r⌋ < N`,
so a connection admitted before bucket `t` sends it in a bucket at most `t + (D + N − 2)`, and that many buckets past
`t` end within `⌈(D + N − 2) / T⌉` execution phases of `t`'s (`add_div_le_add_ceil`), no more than the tail:
`Guarantees.HeldCutInTail`. A mark's execution phase is `ticks / T`, and division by `T` is monotone:
`Guarantees.MarkOrders`. The two together are `Guarantees.SpanCovers`. Under `trusted-host` the network certifier's
run stamps a mark with that code, so its stamp orders the mark (`markOrders_run`).

The `#guard`s hold the tail to the schedules the span count's own guards read: a response of 17 cells released two a
bucket after a delay of 8 buckets, whose last cell leaves 16 buckets after its admission, has a tail of one execution
phase of 16 buckets, and a response two cells longer a tail of two.
-/

@[expose] public section

namespace NetTiming

/-- `b` buckets past bucket `a` end within `⌈b / T⌉` execution phases of `a`'s. -/
theorem add_div_le_add_ceil (a b T : ℕ) : (a + b) / T ≤ a / T + (b + T - 1) / T := by
  rcases Nat.eq_zero_or_pos T with h0 | hT
  · subst h0; simp
  · have ha := Nat.div_add_mod a T
    have hm := Nat.mod_lt a hT
    have hb : b ≤ T * ((b + T - 1) / T) := by
      have h1 := Nat.div_add_mod (b + T - 1) T
      have h2 := Nat.mod_lt (b + T - 1) hT
      omega
    have key : a + b < T * (a / T + (b + T - 1) / T + 1) := by
      rw [Nat.mul_add, Nat.mul_add, Nat.mul_one]
      omega
    exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul hT).mpr (by rw [Nat.mul_comm]; exact key))

theorem heldCutInTail : Guarantees.HeldCutInTail := by
  intro κ s N T t c j hc ha hj
  have hjr : j / s.r < N := Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le hj hc)
  have hb : s.bucket c j ≤ t + (s.D + N - 2) := by
    unfold CertifierSchedule.bucket; omega
  calc s.bucket c j / T ≤ (t + (s.D + N - 2)) / T := Nat.div_le_div_right hb
    _ ≤ t / T + (s.D + N - 2 + T - 1) / T := add_div_le_add_ceil _ _ _
    _ ≤ t / T + spanTail s.D N T := by
        apply Nat.add_le_add_left
        unfold spanTail
        exact Nat.div_le_div_right (by omega)

theorem markOrders : Guarantees.MarkOrders := by
  intro T ticks b
  exact ⟨fun h => Nat.div_le_div_right h, fun h => Nat.div_le_div_right (Nat.le_of_lt h)⟩

theorem spanCovers : Guarantees.SpanCovers := by
  intro κ s N T ts te c j hc hj ha hs
  exact ⟨(markOrders T ts _).1 hs, heldCutInTail s N T te c j hc ha hj⟩

/-- Under `trusted-host`, a network certifier whose code on a mark is `markPhase T` stamps each mark with an execution
phase that orders it against every bucket. -/
theorem markOrders_run {Dev : Type} (T : ℕ) (dev : Dev) (run : Dev → ℕ → ℕ)
    (htrust : Verity.Assumptions.TrustedHost dev run (markPhase T)) (ticks b : ℕ) :
    (ticks ≤ b → run dev ticks ≤ b / T) ∧ (b < ticks → b / T ≤ run dev ticks) := by
  rw [htrust ticks]
  exact markOrders T ticks b

-- D = 8, r = 2, T = 16: 17 cells are N = 9 buckets of egress, 19 are 10
#guard spanTail 8 9 16 == 1 && spanTail 8 10 16 == 2
-- a response's one cell after a delay of one execution phase is still one execution phase, and a bucket more is two
#guard spanTail 16 1 16 == 1 && spanTail 17 1 16 == 2 && spanTail 1 1 1 == 1
-- D = 10, r = 4, T = 600 and 4096 cells, N = 1024 buckets: two execution phases
#guard spanTail 10 1024 600 == 2

end NetTiming

namespace NetTiming.SecurityProofs

theorem HeldCutInTail : NetTiming.Guarantees.HeldCutInTail := NetTiming.heldCutInTail

theorem MarkOrders : NetTiming.Guarantees.MarkOrders := NetTiming.markOrders

theorem SpanCovers : NetTiming.Guarantees.SpanCovers := NetTiming.spanCovers

end NetTiming.SecurityProofs
