import Definitions.Accounting.Carrier

/-!
# Stacking's named assumption: bound keys (trusted)

What the stacking guarantees (`Models/Accounting/Guarantees/Stacking.lean`) assume beyond the remote audit's own and
`Accounting.Assumptions.EncodeChainHardness`, a named `Prop` taken as a hypothesis, never an `axiom`:

* `KeyBinding T M δ enc`: under the append-coin rule (`Accounting.appendCoinBlocks`), whatever the writers choose for
  each append from the codec and the coins of the appends before it, the blocks a pass reads at any layout fixed apart
  from the encodings are on every event at most `δ` likelier than uniform blocks (claim `key-binding`).
-/

namespace Accounting.Assumptions

open Pous (pr)
open scoped ENNReal

/-- **`key-binding`: under the append-coin rule, the blocks a pass reads are uniform but for `δ`.** For the codec
`enc` of `T` appends of `M` block positions each, every way `choose` the writers make each append's choices from the
codec and the coins of the appends before it, every layout `slot` of `N` distinct positions, and every set `ev` of the
`N` blocks' encodings, the blocks at a uniform codec `ω` and uniform coins `κ` are in `ev` with probability at most
`δ` more than uniform blocks:

  `Pr_{ω,κ}[appendCoinBlocks enc choose slot (ω, κ) ∈ ev] ≤ Pr_E[E ∈ ev] + δ`.

`ω` is the codec in the ideal encoding model (its rounds' random function, below). The coins are the appends'
(remote_bandwidth's PROTOCOL.md §3–§4), drawn at the node's one live registration and after: the registration is
append `0`, whose coin is the seed, drawn after its payload root; every other append's coin `c_a` is 32 bytes, drawn
once, when the append's root arrives, after that root has committed every digest. A registered block is its payload
encoded under the seed's key. An appended block is its payload `p_j` encoded with the bound tweak
`b_j(c_a) = H(DOMAIN ‖ seed ‖ j ‖ d_j ‖ c_a)`, `d_j = sha512/row/v2(p_j)`, added to every round, and it decodes only
under the digest and the coin it was written with. The writers' choices for an append (its payloads, which may be a
circuit's values over other blocks' encodings, their padding, the run inputs behind them, how many blocks, the
schedule up to it) are any function of `ω` and the coins before its own: the game fixes them before its coin, as the
protocol does, and no writer is indexed or counted. The layout is any embedding, chosen apart from the encodings: in
the stacked guarantees it is the memory map's schedule at the session's coins before the pass.

**Why it should hold.** The ideal-encoding step. A round `x ↦ ρ((x + c_r + t) mod p)` is not a random permutation:
`ρ(p − x) = p − ρ(x)`, and the parity of `ρ(x)` is the Legendre symbol of `x` (`p ≡ 3 mod 4`). The ideal model is a
random function on pairs `{x, p − x}` to even residues `e`, the round's output `e` or `p − e` by the sign of its input.
Under it a word's encoding, given its key, tweak and payload, is uniform on `[0, p)`: its last round's input lies in a
fresh pair, which takes a fresh value `e`, and its sign is uniform given the pair. Two words' encodings are independent
unless some round's inputs agree up to sign. An appended block's tweak reads its append's coin, uniform over `2^256`
values and drawn after its payload: whatever the writers chose from `ω` and the earlier coins, the tweak is one they
could not aim at (it is 64 bytes, so `2^512` of the `p` values), and the block is uniform and independent of the other
blocks and of the earlier coins. What its payload carries about other blocks is a carrier (`RateSketch.Carrier`),
charged apart. So `δ` is the writers' advantage at anticipating an append's tweak before its coin (an unbounded writer
gains about the bits it chooses over `2^256`), the chance that some round's inputs agree up to sign, and the words'
values `p` and above (`2^n − p` of `2^n`, at `p = 2^n − c`). Not covered: a second registration redraws the seed, a
choice among the registrations, which the one live registration rules out; and a layout chosen from the encodings
after the coins, which the stacked guarantees fix before. Under the seed's key alone a writer picks an appended
block's encoding by writing its decoding: measured in
`verity/core/protocols/remote_bandwidth/tests/test_key_binding.py`, which also fails 64 tries under the bound key and
keys a block under each coin apart. -/
def KeyBinding {Ω Ca Ch B : Type} [Fintype Ω] [Fintype Ca] [Fintype B] (T M : ℕ) (δ : ℝ≥0∞)
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B) : Prop :=
  ∀ (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch) (N : ℕ) (slot : Fin N ↪ Fin T × Fin M)
    (ev : Set (Fin N → B)),
    pr (fun y : Ω × (Fin T → Ca) => appendCoinBlocks enc choose slot y ∈ ev) ≤ pr (fun E : Fin N → B => E ∈ ev) + δ

end Accounting.Assumptions
