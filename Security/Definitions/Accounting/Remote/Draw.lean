import Mathlib
import Definitions.Accounting.Actual.Sampler
import Definitions.Accounting.TS.Drawn
import Definitions.Hash.RowLeaf
import Definitions.Pous.Model.StreamedWipe

/-!
# The remote audit's opening draw (trusted)

The draw `audit.pooled` reads, as the verifier runs it. A session has `n` windows, in order. Window `t` registers its
`U_t ≤ m` receipts and their roots, which fix which of them are correct unless an opening collides (`Receipts`). Then
the challenger releases window `t`'s
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
* `DrawCoins n κ X`: a session's coins, the other coins `x` (the prover's, the world's, every window's challenge) and each
  window's `κ`-bit openings coin.
* `Session`: each window's size `U_t` and correct receipts, read off the coins. The prover is sent every coin once it is
  released (`coins.coin_reason`), so it may read an earlier window's whole coin, not only its openings. That window
  `t`'s are read only off the earlier windows' openings coins is `Assumptions.Live`. `Session.lands t ω b` is window
  `t`'s test at its own `U_t`, `s'_t` and correct receipts, and `Session.bound t ω` the most it lands on uniform
  blocks, `(g_t/U_t)^(s'_t)` with `g_t` its correct receipts, `0` when `g_t < s'_t`.
* `Committed`: **a session read through a coin commitment.** The challenger commits every coin before the first is
  released, as hm96-sha512 leaves (192-byte salts) in a frame-v3-sha512 tree (`coins.py`). It sends the prover each
  coin as it releases it and the root only after the prover's last root, with the outcome (`audit.Session.disclose`;
  soundness's ruling of 27 Oct, no hiding premise on the root), so the session the prover plays is a `Session` and the
  guarantees take it plainly. A root is a function of every coin, so a prover holding it from the start could size a
  window off window `t`'s openings coin, and that session is not `Live` (red team RD-12). `view c` is the session
  played holding commitment `c`; `play com` holds `com ω` at coins `ω` (the salts among the other coins `x`).
  `junk com t` is the commitment with openings coin `t`'s leaf committing zeros, the other coins and every salt as
  they are: it reads no bit of coin `t`. The memory line over the prover's records reads its records through it
  (`Properties.RemoteAudit.Line.recordCommitted`).
* `Session.potential prf δ t`: the hybrid's potential after `t` windows: windows `j < t` land on `prf j` at their own
  coins, and a uniform 512-bit threshold coin `u` falls below `∏_{j<t} bound j`.
* `Committed.hide com prf δ t`: **window `t`'s hiding term**, what the junk-commitment hybrid costs at window `t`
  (`0` at a view that reads no commitment):
  how much likelier the potential after `t + 1` windows is at the deployed commitment than at `junk com t`, plus how
  much likelier the potential after `t` windows is at `junk com t` than at the deployed one. Each is an event of the
  coins and the commitment the prover holds, read off the prover's sizes and correct receipts of windows `j ≤ t`,
  which are fixed before coin `t` and its salt are released: a distinguisher of hm96 commitments at leaf `t` between
  coin `t` and zeros (claim `statistical-hiding`). At the pinned key (`common-reference-string`, `hash-derived-key`)
  hm96 puts each within the pair distance of leaf `t`'s commit strings, and over a session's `n` openings leaves those
  sum to at most `n·2^-192` for all but a `2^-64` fraction of keys (`verity/ml/commitments/hm96/PROTOCOL.md` §3), so
  the session's hiding terms total at most `n·2^-191` (`ledger.hiding`). That bound reads a prover whose view before
  coin `t` holds no salt of a coin not yet released, as the deployed prover's doesn't: a `view` that reads leaf `t`'s
  salt off the other coins gets no bound from hm96, rightly.
* `Session.distinguisher prf δ t`: **window `t`'s distinguisher**, in `Pous.StreamedWipe.prfAdvantage`'s form, one a
  window and fixed by the session. On coins `(ω, u)` and blocks `b` it runs the session through window `t`: windows
  `j < t` on `prf j` at their own coins in `ω`, which it computes itself, window `t` on `b`. It outputs whether every
  one of them lands in its correct receipts and `u` falls below `∏_{j ≤ t} bound j`. It reads `ω`'s other coins and
  the earlier windows' coins (window `t`'s and the later ones' are unread, by `Live`), and `b`'s first `s'_t ≤ s_t`
  blocks: a distinguisher with `s_t` queries to window `t`'s stream, as efficient as the session's prover and verifier.
  It is the hybrid over windows, the earlier windows real and window `t` given, with the later windows replaced by
  their bound as a coin: they read window `t`'s whole coin, which a distinguisher given `b` instead of a key doesn't
  have, and on uniform blocks each lands with probability at most its bound given the windows before.
* `cont t ω ζ`: **window `t`'s continuation** of coins `ω` on `ζ`, the run that shares `ω`'s cut (its other coins,
  the prover's, the world's and every salt, and its openings coins before window `t`'s) and continues on `ζ`'s openings
  coins from window `t`'s on. It has `Accounting.TS.spliceAt`'s two laws, so the sampled window's forking lemma reads
  it.
* `Receipts`: **the receipts as the window's roots hold them.** Window `t` registers its receipts as
  `hm96-sha512/row/v2` leaves, 16 receipts a leaf, in frame-v3-sha512 trees, and sends their roots before its coin is
  released (`RemoteAudit.verifyWindow`). `opened t ω p` is the opening the prover presents for receipt `p` when the
  window opens it. `Shows D prf rest t p ω v`: the window opens `p` at `ω` and the verifier accepts the presented
  opening, of row `v`, at the receipt's root and rank as that run has them. `held D prf rest t ω p` is **the row the
  root holds for it**, a definition, not the prover's: the plurality of the rows `p` shows over window `t`'s
  continuations of `ω` (`Accounting.TS.plural` at `cont t`, ties broken by `Classical.epsilon`), as
  `Accounting.TS.DrawnLayer.committed` is a position's. It reads only `ω`'s cut, and no row is shown on more of the
  cut's continuations (`Accounting.SecurityProofs.Remote.Binding.held_cont`, `tally_le_held`). A root alone fixes no
  row: the prover may present another one. `Moved D prf rest t ω p` is the verifier accepting the presented opening at
  the receipt's root and rank with a row other than the held one. `binding D prf rest` is **the receipts' binding
  term**: the probability that some window's opened receipt moved. Its bound is the fork form (`Remote.Fork`,
  `ReceiptsBindingOfFork`): two runs from window `t`'s cut show the receipt two accepted rows, which at one root, rank
  and verifier hash a SHA-512 collision (`Hash.Guarantees.RowBinding`, proved), and otherwise split the receipt's
  roots. Without a split it is a collision a finder outputs (`ReceiptsBindingOfCR`, claim `cr/sha-512`);
  `ledger.binding` charges that, the split at 0 (the open gap `binding-split`).
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

/-- A session's coins: the other coins and window `t`'s openings coin. Not the live coins (`Verity.Coins`): each
openings coin seeds a draw without replacement over its window's own receipts, and the other coins hold the prover's. -/
abbrev DrawCoins (n κ : ℕ) (X : Type) : Type := X × (Fin n → Pous.Bits κ)

/-- A session of `n` windows of at most `m` receipts, window `t` drawing `min(s_t, U_t)` of its `U_t`. -/
structure Session (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X : Type) where
  /-- Window `t`'s receipt count `U_t`. -/
  size : Fin n → DrawCoins n κ X → Fin (m + 1)
  /-- Window `t`'s correct receipts, by position. -/
  correct : Fin n → DrawCoins n κ X → Finset ℕ

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
def draws (t : Fin n) (ω : DrawCoins n κ X) : ℕ := min (s t) (D.size t ω)

/-- The receipts window `t` opens at coins `ω`: its draw on the first blocks `prf t` gives its coin, with the fallbacks
`rest ω t` where the sampler rejects a first word. -/
def opened (prf : Fin n → Pous.Bits κ → Blocks m) (rest : DrawCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n)
    (ω : DrawCoins n κ X) : Finset ℕ :=
  openings (D.size t ω) (D.draws t ω) (prf t (ω.2 t)) (rest ω t)

/-- Window `t`'s correct receipts among its `U_t`. -/
def good (t : Fin n) (ω : DrawCoins n κ X) : ℕ :=
  ((range (D.size t ω)).filter (· ∈ D.correct t ω)).card

/-- Window `t`'s bound on uniform first blocks: `(g_t/U_t)^(s'_t)`, `0` when `g_t < s'_t`. -/
noncomputable def bound (t : Fin n) (ω : DrawCoins n κ X) : ℝ :=
  if D.draws t ω ≤ D.good t ω then ((D.good t ω : ℝ) / (D.size t ω : ℕ)) ^ D.draws t ω else 0

/-- Window `t`'s test at outcome `ω` on first blocks `b`: its draw of `s'_t` of its `U_t` lands in its correct
receipts. -/
def lands (t : Fin n) (ω : DrawCoins n κ X) (b : Blocks m) : Prop :=
  test (D.size t ω) (D.draws t ω) (D.correct t ω) b

/-- **Window `t`'s distinguisher** on coins `y.1 = (ω, u)` and first blocks `y.2`: windows `j < t` land on `prf j` at
their coins in `ω`, window `t` lands on `y.2`, and the threshold coin `u` falls below `∏_{j ≤ t} bound j`. -/
def distinguisher (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n)
    (y : (DrawCoins n κ X × Pous.Bits 512) × Blocks m) : Prop :=
  (∀ j, j < t → D.lands j y.1.1 (prf j (y.1.1.2 j))) ∧ D.lands t y.1.1 y.2 ∧
    below δ (∏ j ∈ univ.filter (· ≤ t), D.bound j y.1.1) y.1.2

/-- **The hybrid's potential after `t` windows** at coins `z = (ω, u)`: windows `j < t` land on `prf j` at their coins,
and the threshold coin `u` falls below the product of their bounds. -/
def potential (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : ℕ) (z : DrawCoins n κ X × Pous.Bits 512) : Prop :=
  (∀ j : Fin n, (j : ℕ) < t → D.lands j z.1 (prf j (z.1.2 j))) ∧
    below δ (∏ j ∈ univ.filter (fun j : Fin n => (j : ℕ) < t), D.bound j z.1) z.2

/-- The session's wrong receipts, `Σ_t (U_t − good_t)`. -/
def wrong (ω : DrawCoins n κ X) : ℕ := ∑ t, ((D.size t ω : ℕ) - D.good t ω)

/-- The session's pooled bound at `δ`, at the sizes and draw counts of outcome `ω`. -/
noncomputable def pooled (δ : ℝ) (ω : DrawCoins n κ X) : ℝ :=
  Draw.pooled n (fun t => D.size t ω) (fun t => D.draws t ω) δ

end Session

/-- **A session that reads the coin commitment**: `view c` is the session the prover plays holding commitment `c`. -/
structure Committed (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X Cm : Type) where
  /-- The session the prover plays holding commitment `c`. -/
  view : Cm → Session n m s κ X

namespace Committed

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X Cm : Type} (F : Committed n m s κ X Cm)

/-- The session the prover plays holding commitment `com ω` at coins `ω`. -/
def play (com : DrawCoins n κ X → Cm) : Session n m s κ X where
  size t ω := (F.view (com ω)).size t ω
  correct t ω := (F.view (com ω)).correct t ω

/-- The commitment with openings coin `t`'s leaf committing zeros: `com` at the coins with window `t`'s openings coin
replaced by zeros, the other coins and every salt as they are. -/
def junk (com : DrawCoins n κ X → Cm) (t : Fin n) (ω : DrawCoins n κ X) : Cm :=
  com (ω.1, Function.update ω.2 t fun _ => false)

/-- **Window `t`'s hiding term**: how much likelier the potential after `t + 1` windows is at the deployed commitment
than with openings coin `t`'s leaf committing zeros, plus how much likelier the potential after `t` windows is with
that leaf committing zeros than at the deployed commitment (each truncated at 0). -/
noncomputable def hide [Fintype X] (com : DrawCoins n κ X → Cm) (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ)
    (t : Fin n) : ℝ≥0∞ :=
  (Pous.pr ((F.play com).potential prf δ ((t : ℕ) + 1)) -
      Pous.pr ((F.play (junk com t)).potential prf δ ((t : ℕ) + 1))) +
    (Pous.pr ((F.play (junk com t)).potential prf δ t) - Pous.pr ((F.play com).potential prf δ t))

end Committed

/-- **Window `t`'s continuation** of coins `ω` on `ζ`: `ω`'s other coins and its openings coins before window `t`'s,
then `ζ`'s openings coins from window `t`'s on. -/
def cont {n κ : ℕ} {X : Type} (t : Fin n) (ω ζ : DrawCoins n κ X) : DrawCoins n κ X :=
  (ω.1, fun j => if j < t then ω.2 j else ζ.2 j)

/-- **A session's receipts as the windows' roots hold them**: receipt `p` of window `t` at coins `ω` is a row of an
hm96 bit-row leaf at rank `rank t ω p` of the tree whose root `root t ω p` the window registered, checked by
`verifier t ω p` under `hash` (SHA-512). `opened t ω p` is the opening the prover presents when the window opens it;
the row the root holds for it is `Receipts.held`, read off the prover's runs. -/
structure Receipts (n κ : ℕ) (X : Type) where
  /-- The hash, SHA-512. -/
  hash : Hash.Msg → Hash.Digest
  /-- What the verifier holds for receipt `p`'s leaf: its domain, the hm96 key and the format (`row/v2`). -/
  verifier : Fin n → DrawCoins n κ X → ℕ → Hash.RowLeaf.Verifier
  /-- The root window `t` registered for receipt `p`. -/
  root : Fin n → DrawCoins n κ X → ℕ → Hash.Digest
  /-- Receipt `p`'s leaf's rank. -/
  rank : Fin n → DrawCoins n κ X → ℕ → ℕ
  /-- The opening the prover presents for receipt `p`. -/
  opened : Fin n → DrawCoins n κ X → ℕ → Hash.RowLeaf.Opening

namespace Receipts

variable {n κ : ℕ} {X : Type} (R : Receipts n κ X)

/-- **Receipt `p` of window `t` shows row `v` at coins `ω`**: the window opens it, the verifier accepts the opening the
prover presents at the receipt's root and rank, and its row is `v`. -/
def Shows {m : ℕ} {s : Fin n → ℕ} (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : DrawCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (p : ℕ) (ω : DrawCoins n κ X) (v : Hash.Row) : Prop :=
  p ∈ D.opened prf rest t ω ∧
    Hash.RowLeaf.accepts R.hash R.hash (R.verifier t ω p) (R.root t ω p) (R.rank t ω p) (R.opened t ω p) ∧
    (R.opened t ω p).row = v

/-- **The row receipt `p`'s root holds** at window `t` and coins `ω`: the plurality of the rows it shows over window
`t`'s continuations of `ω`, a row that the most of them show (`Accounting.TS.plural` at `cont t`, ties broken by
`Classical.epsilon`). -/
noncomputable def held [Fintype X] {m : ℕ} {s : Fin n → ℕ} (D : Session n m s κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : DrawCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : DrawCoins n κ X)
    (p : ℕ) : Hash.Row :=
  Accounting.TS.plural (cont t) (R.Shows D prf rest t p) ω

/-- **Receipt `p` of window `t` moved**: the verifier accepts the opening presented, at the receipt's root and rank,
with a row other than the one its root holds. -/
def Moved [Fintype X] {m : ℕ} {s : Fin n → ℕ} (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : DrawCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : DrawCoins n κ X) (p : ℕ) : Prop :=
  Hash.RowLeaf.accepts R.hash R.hash (R.verifier t ω p) (R.root t ω p) (R.rank t ω p) (R.opened t ω p) ∧
    (R.opened t ω p).row ≠ R.held D prf rest t ω p

/-- **The receipts' binding term**: the probability that some window opens a receipt that moved. -/
noncomputable def binding [Fintype X] {m : ℕ} {s : Fin n → ℕ} (D : Session n m s κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : DrawCoins n κ X → Fin n → ℕ → ℕ) : ℝ≥0∞ :=
  Pous.pr fun ω => ∃ t, ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p

end Receipts

end Accounting.Remote.Draw
