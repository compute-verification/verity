import Mathlib
import Definitions.Accounting.Actual.WipeDraw
import Definitions.Pous.Model.StreamedWipe

/-!
# The remote audit's opening draw (trusted)

The draw `audit.pooled` reads, as the verifier runs it. A session has `n` windows, in order. Window `t` registers its
`U_t ≤ m` receipts and their roots, which fix which of them are correct. Then the challenger releases window `t`'s
openings coin, live, to the verifier and the prover, and the verifier opens `s'_t = min(s_t, U_t)` distinct receipts of
the `U_t` (`RemoteAudit.draws`, `Session.close`): `Key.subset(s'_t, U_t, "openings")` under the key
`derive(coin_t, OPENINGS_DOMAIN, {registration, window: t})` (`verity/ml/randomness`), a partial Fisher–Yates whose step
`i` swaps position `i` with position `i + Key.uniform(U_t − i)`, read from the keyed SHA-512 stream at
`("s", "openings", i)`. `Key.uniform(N)` is exact rejection sampling: it reads `w`-bit big-endian words,
`w = 8 ⌈(bitlen N + 64) / 8⌉`, and returns the first word below `2^w − (2^w mod N)`, reduced mod `N`
(`Actual.wordIndex`). For `N < 2^448` the first word is the top `w` bits of the stream's first block.

* `Blocks m`: a window's first blocks, draw `i`'s first SHA-512 block as its big-endian value. The deployed function from
  window `t`'s coin to its first blocks is the guarantees' `prf t`.
* `word N b`: the block's first word at `N`, its top `width N` bits; `offset N r b` is the sampler's draw on it, or `r`
  when the sampler rejects it (what it then draws from the stream's later words, left free).
* `arrange j i`: the partial Fisher–Yates' arrangement after `i` swaps, positions to receipts, as `Key.subset` keeps it
  (`swapped`); `openings U k b r` is the set of the receipts at its first `k` positions, the window's openings.
* `test U k C b`: the window test on first blocks `b`: when the sampler accepts each of the `k` draws' first words at
  `U`, whether the openings lie in `C`, and otherwise true. It reads `b`'s first `k` blocks and runs the Fisher–Yates.
* `below δ P u`: the threshold coin. `u`, 512 uniform bits, falls below `min(1, δ/P)` rounded down to a multiple of
  `2^-512`, and always when `P ≤ δ`.
* `Coins n κ X`: a session's coins, the other coins `x` (the prover's, the world's, every window's challenge) and each
  window's `κ`-bit openings coin.
* `Session`: each window's size `U_t` and correct receipts, read off the coins. The prover is sent every coin once it is
  released (`coins.coin_reason`), so it may read an earlier window's whole coin, not only its openings. That window
  `t`'s are read only off the earlier windows' openings coins is `Assumptions.Live`. `Session.lands t ω b` is window
  `t`'s test at its own `U_t`, `s'_t` and correct receipts, and `Session.bound t ω` the most it lands on uniform
  blocks, `(g_t/U_t)^(s'_t)` with `g_t` its correct receipts, `0` when `g_t < s'_t`.
* `Session.distinguisher prf δ t`: **window `t`'s distinguisher**, in `Pous.StreamedWipe.prfAdvantage`'s form, one a
  window and fixed by the session. On coins `(ω, u)` and blocks `b` it runs the session through window `t`: windows
  `j < t` on `prf j` at their own coins in `ω`, which it computes itself, window `t` on `b`. It outputs whether every
  one of them lands in its correct receipts and `u` falls below `∏_{j ≤ t} bound j`. It reads `ω`'s other coins and
  the earlier windows' coins (window `t`'s and the later ones' are unread, by `Live`), and `b`'s first `s'_t ≤ s_t`
  blocks: a distinguisher with `s_t` queries to window `t`'s stream, as efficient as the session's prover and verifier.
  It is the hybrid over windows, the earlier windows real and window `t` given, with the later windows replaced by
  their bound as a coin: they read window `t`'s whole coin, which a distinguisher given `b` instead of a key doesn't
  have, and on uniform blocks each lands with probability at most its bound given the windows before.
* `pooled n U s' δ`: `audit.pooled`'s horizon reading, the most wrong receipts `Σ_t f_t·U_t` over the shares `f_t` with
  `0 ≤ f_t ≤ 1 − s'_t/U_t` and `Σ_t s'_t·f_t ≤ budget S δ = S·(1 − δ^(1/S))`, `S = Σ_t s'_t`. `audit.pooled` computes it
  greedily, the windows taken by `U_t/s'_t` falling (the fractional knapsack's optimum); that the greedy order reaches
  this supremum is not stated here.
-/

namespace Accounting.Remote.Draw

open Finset
open scoped ENNReal

/-- A window's first blocks: draw `i`'s first SHA-512 block of its stream, read as its big-endian value. -/
abbrev Blocks (m : ℕ) : Type := Fin m → Pous.Bits 512

/-- The sampler's word width at `N`: `8 ⌈(bitlen N + 64) / 8⌉` bits (`Key._uniform`). -/
def width (N : ℕ) : ℕ := 8 * ((Nat.size N + 71) / 8)

/-- The first word of an `L`-bit block at `N`: its top `width N` bits, the stream's first `width N / 8` bytes read
big-endian. -/
def word (N : ℕ) {L : ℕ} (b : Pous.Bits L) : Pous.Bits (width N) :=
  (Pous.bitsEquiv (width N)).symm ⟨Pous.bitsVal b / 2 ^ (L - width N), by
    have hb : Pous.bitsVal b < 2 ^ L := (Pous.bitsEquiv L b).isLt
    refine Nat.div_lt_of_lt_mul (hb.trans_le ?_)
    rw [← pow_add]
    exact Nat.pow_le_pow_right (by norm_num) (by omega)⟩

/-- A draw's offset at `N`: the sampler's on the block's first word when it accepts it, else `r`. -/
def offset (N r : ℕ) (b : Pous.Bits 512) : ℕ := ((Actual.wordIndex N (word N b)).map Fin.val).getD r

/-- The arrangement after `i` swaps, positions to receipts: step `l` swaps positions `l` and `l + j l`
(`swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)`). -/
def arrange (j : ℕ → ℕ) : ℕ → Equiv.Perm ℕ
  | 0 => 1
  | i + 1 => arrange j i * Equiv.swap i (i + j i)

/-- The offsets of a window of `U` receipts with first blocks `b` and fallbacks `r`: step `i` draws below `U − i`. -/
def offsets {m : ℕ} (U : ℕ) (b : Blocks m) (r : ℕ → ℕ) (i : ℕ) : ℕ :=
  if h : i < m then offset (U - i) (r i) (b ⟨i, h⟩) else r i

/-- The receipts a window of `U` opens with `k` draws: those at the arrangement's first `k` positions. -/
def openings {m : ℕ} (U k : ℕ) (b : Blocks m) (r : ℕ → ℕ) : Finset ℕ :=
  (range k).image (arrange (offsets U b r) k)

/-- Every one of the `k` draws of a window of `U` takes its offset from its first word. -/
def accepted {m : ℕ} (U k : ℕ) (b : Blocks m) : Prop :=
  ∀ i : Fin m, (i : ℕ) < k → (Actual.wordIndex (U - i) (word (U - i) (b i))).isSome = true

/-- **The window test**: on first blocks whose `k` first words at `U` the sampler accepts, whether the openings lie in
`C`; true when it rejects one. -/
def test {m : ℕ} (U k : ℕ) (C : Finset ℕ) (b : Blocks m) : Prop :=
  accepted U k b → openings U k b (fun _ => 0) ⊆ C

/-- **The threshold coin**: `u` read as a number below `2^512` is below `⌊δ·2^512/P⌋`, or `P ≤ δ`. -/
def below (δ P : ℝ) (u : Pous.Bits 512) : Prop := P ≤ δ ∨ Pous.bitsVal u < ⌊δ * 2 ^ 512 / P⌋₊

/-- A session's coins: the other coins and window `t`'s openings coin. -/
abbrev Coins (n κ : ℕ) (X : Type) : Type := X × (Fin n → Pous.Bits κ)

/-- A session of `n` windows of at most `m` receipts, window `t` drawing `min(s_t, U_t)` of its `U_t`. -/
structure Session (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X : Type) where
  /-- Window `t`'s receipt count `U_t`. -/
  size : Fin n → Coins n κ X → Fin (m + 1)
  /-- Window `t`'s correct receipts, by position. -/
  correct : Fin n → Coins n κ X → Finset ℕ

/-- `audit.pooled`'s budget: `S·(1 − δ^(1/S))` (`-expm1(log δ / S)·S`). -/
noncomputable def budget (S : ℕ) (δ : ℝ) : ℝ := S * (1 - δ ^ ((S : ℝ)⁻¹))

/-- The pooled bound of windows of sizes `U` drawing `d`: the most `Σ_t f_t·U_t` over the shares `f` with
`0 ≤ f_t ≤ 1 − d_t/U_t` and `Σ_t d_t·f_t ≤ budget (Σ_t d_t) δ`. -/
noncomputable def pooled (n : ℕ) (U d : Fin n → ℕ) (δ : ℝ) : ℝ :=
  sSup ((fun f : Fin n → ℝ => ∑ t, f t * U t) ''
    {f | (∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t) ∧ ∑ t, (d t : ℝ) * f t ≤ budget (∑ t, d t) δ})

namespace Session

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Session n m s κ X)

/-- Window `t`'s draw count, `min(s_t, U_t)`. -/
def draws (t : Fin n) (ω : Coins n κ X) : ℕ := min (s t) (D.size t ω)

/-- The receipts window `t` opens at coins `ω`: its draw on the first blocks `prf t` gives its coin, with the fallbacks
`rest ω t` where the sampler rejects a first word. -/
def opened (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n)
    (ω : Coins n κ X) : Finset ℕ :=
  openings (D.size t ω) (D.draws t ω) (prf t (ω.2 t)) (rest ω t)

/-- Window `t`'s correct receipts among its `U_t`. -/
def good (t : Fin n) (ω : Coins n κ X) : ℕ :=
  ((range (D.size t ω)).filter (· ∈ D.correct t ω)).card

/-- Window `t`'s bound on uniform first blocks: `(g_t/U_t)^(s'_t)`, `0` when `g_t < s'_t`. -/
noncomputable def bound (t : Fin n) (ω : Coins n κ X) : ℝ :=
  if D.draws t ω ≤ D.good t ω then ((D.good t ω : ℝ) / (D.size t ω : ℕ)) ^ D.draws t ω else 0

/-- Window `t`'s test at outcome `ω` on first blocks `b`: its draw of `s'_t` of its `U_t` lands in its correct
receipts. -/
def lands (t : Fin n) (ω : Coins n κ X) (b : Blocks m) : Prop :=
  test (D.size t ω) (D.draws t ω) (D.correct t ω) b

/-- **Window `t`'s distinguisher** on coins `y.1 = (ω, u)` and first blocks `y.2`: windows `j < t` land on `prf j` at
their coins in `ω`, window `t` lands on `y.2`, and the threshold coin `u` falls below `∏_{j ≤ t} bound j`. -/
def distinguisher (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n)
    (y : (Coins n κ X × Pous.Bits 512) × Blocks m) : Prop :=
  (∀ j, j < t → D.lands j y.1.1 (prf j (y.1.1.2 j))) ∧ D.lands t y.1.1 y.2 ∧
    below δ (∏ j ∈ univ.filter (· ≤ t), D.bound j y.1.1) y.1.2

/-- The session's wrong receipts, `Σ_t (U_t − good_t)`. -/
def wrong (ω : Coins n κ X) : ℕ := ∑ t, ((D.size t ω : ℕ) - D.good t ω)

/-- The session's pooled bound at `δ`, at the sizes and draw counts of outcome `ω`. -/
noncomputable def pooled (δ : ℝ) (ω : Coins n κ X) : ℝ :=
  Draw.pooled n (fun t => D.size t ω) (fun t => D.draws t ω) δ

end Session

end Accounting.Remote.Draw
