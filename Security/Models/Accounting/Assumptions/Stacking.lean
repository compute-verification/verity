import Definitions.Accounting.Carrier

/-!
# Stacking's named assumptions: encoding costs budgeted gates, and bound keys (trusted)

What the stacking guarantees (`Models/Accounting/Guarantees/Stacking.lean`) assume beyond the remote audit's own, named
`Prop`s taken as hypotheses, never `axiom`s:

* `CircuitEncodeHardness b rEnc cost`: computing `x` of a block's `b` encoded bits costs at least `x/b` of `rEnc`
  budgeted gates (claim `circuit-encode-hardness`).
* `KeyBinding δ store`: a store whose blocks written after the seed are encoded under bound keys is, on every event,
  at most `δ` likelier than a uniform store (claim `key-binding`).
-/

namespace Accounting.Assumptions

open Pous (pr)
open scoped ENNReal

/-- **`circuit-encode-hardness`: a circuit needs `rEnc` budgeted gates to encode a fresh block, `x/b` of them for `x` of
its `b` bits.** For every `x ≤ b`:

  `x · rEnc ≤ b · cost x`,

where `cost x` is what computing `x` bits of a block's encoding costs. The stacking guarantees take
`cost = encodeBitsCost F k f w εnum εden`, the circuit model's encode cost (`Gates.encodeCost`, budgeted gates of
`Definitions/Accounting/Gates`) of the codec words `x` bits fill. It is what `RateSketch.Provenance.roundings` charges
an `encode` step of `x` bits. With a budget of `R` budgeted gates over a carrier's life it bounds the bits a circuit
encodes by `b·R/rEnc`: at the property's `c_ρ` budgeted gates per byte of certified read, `q ≤ c_ρ·reads·L/rEnc` blocks
over a life of `L` windows.

At that cost it follows from `Accounting.Assumptions.EncodeChainHardness` at `E` a word, at `rEnc = (b/(8·bytes))·E`,
wherever the circuit model has a circuit computing the words (`Accounting.Guarantees.CircuitEncodeHardnessOfChain`); at
parameters where it has none, `encodeCost` is `0` and this premise is false.

**Why it should hold.** In the circuit model, s8-family products in their proven window and constant affine maps are
free; every other gate is budgeted, a floor, a carry or a select among them. An `n`-limb modular squaring needs about
one budgeted gate a limb to return its column sums to the window (`Accounting.Guarantees.Gates.Renormalization`; the
carry bound proves 63 bits of it at p1024), and encoding a word is `k` roots of `S` squarings each, so `rEnc` should be
at least `k·S` times that a word, times the block's words. It is conjectured, not proved or measured: the carry bound is
per squaring, not per encoding (`EncodeChainHardness` is that composition), and the charge is for whole words: a
function of a block's encoding cheaper than its words (a word's parity is a Jacobi symbol, which is why the words'
targets leave it out) must enter a carrier's provenance as an input, not an encode. -/
def CircuitEncodeHardness (b rEnc : ℕ) (cost : ℕ → ℕ) : Prop :=
  ∀ x, x ≤ b → x * rEnc ≤ b * cost x

/-- **`key-binding`: a store whose blocks written after the seed are under bound keys is a uniform store, but for
`δ`.** For every set `ev` of stores, the programmed store `store ω`, at a uniform `ω`, is in `ev` with probability at
most `δ` more than a uniform store:

  `Pr_ω[store ω ∈ ev] ≤ Pr_E[E ∈ ev] + δ`.

`ω` is everything the store's contents depend on: the codec, its seed, and every writer's choices. A block registered
before the seed is its payload encoded under the seed's key. A block written after it (a KV page: its payload is a
circuit's value, which may depend on other blocks' encodings) is its payload `p_j` encoded under the bound key
`K_j = H(seed ‖ j ‖ d_j)`, `d_j = sha512/row/v2(p_j)`, and it decodes only under the digest it was written with.

**Why it should hold.** In the ideal encoding model a block's encoding is uniform given its key and payload, and a
bound block's key is fresh for each payload, so a payload chosen in any way from other blocks leaves the block's
encoding uniform and independent of them; what the payload carries about other blocks is a carrier
(`RateSketch.Carrier`), charged apart. Choosing the encoding instead needs a payload that decodes from it under the key
of its own digest, a SHA-512 search, so `δ` is at most the writers' searches over `2^512`. Under the seed's key the
writer picks the encoding by writing its decoding: measured in
`verity/core/protocols/remote_bandwidth/tests/test_key_binding.py`, which also fails 64 tries under the bound key.
Not covered: a writer that chooses among several payloads by their encodings (grinding) moves its block's
distribution, by about the logarithm of its tries in bits; and a block written during a session, whose payload may
depend on the session's coins. -/
def KeyBinding {St Ω : Type} [Fintype St] [Fintype Ω] (δ : ℝ≥0∞) (store : Ω → St) : Prop :=
  ∀ ev : Set St, pr (fun ω => store ω ∈ ev) ≤ pr (fun E : St => E ∈ ev) + δ

end Accounting.Assumptions
