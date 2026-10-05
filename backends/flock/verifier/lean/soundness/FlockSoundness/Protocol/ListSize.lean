import ArkLib.Data.CodingTheory.JohnsonBound.Family
import FlockSoundness.Protocol.Defs

/-!
# The list size: at most `L₀` codewords near any level-0 table

Every phase of `rep_sound` pays a union bound over the codewords within the level-0 proximity
radius of the committed table (`closeMsgs`). This file bounds their number by `listBound₀ =
100·2^r` at rate `2^-r` (`closeMsgs_card_le`), from ArkLib's Johnson bound for interleaved
Reed–Solomon codes (`irs_lambda_le_johnson_mds`: at radius `1 − √ρ − η`, at most `1/(2ηρ)`
codewords). The bridge:

* a table, read position first (`tr`), is an interleaved word; an encoded message is a codeword of
  `irsCode` on the level's evaluation points, because each row is the evaluation of a polynomial of
  degree below `2^cols` (the novel basis, `Arith.Correct.xhat_poly`);
* "close" (`Close`: every lane agrees on a `1 − γ` fraction of positions) is relative block
  Hamming distance at most `γ`;
* encoding is injective (a nonzero polynomial of degree below `2^cols ≤ 2^d` cannot vanish on `2^d`
  distinct points), so distinct close messages are distinct close codewords.
-/

namespace FlockSoundness

open Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]


/-! ## The list size -/

/-- A `fast100` schedule is for the statement's `m`. -/
theorem fast100_m : ∀ m s, fast100 m = some s → s.m = m := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; rfl)

end FlockSoundness
