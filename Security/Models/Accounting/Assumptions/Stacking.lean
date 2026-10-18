import Definitions.Accounting.Carrier

/-!
# Stacking's named assumptions: bound keys, the declared circuit's carriers, the steered layout (trusted)

What the stacking guarantees (`Models/Accounting/Guarantees/Stacking.lean`, `DeclaredCircuit.lean`) assume beyond the
remote audit's own and `Accounting.Assumptions.EncodeChainHardness`, each a named `Prop` taken as a hypothesis, never an
`axiom`:

* `KeyBinding T M δ enc`: under the append-coin rule (`Accounting.appendCoinBlocks`), whatever the writers choose for
  each append from the codec and the coins of the appends before it, the blocks a pass reads at any layout fixed apart
  from the encodings are on every event at most `δ` likelier than uniform blocks (claim `key-binding`).
* `CircuitCarriers A X life B εnum εden W Aud`: every pass device of the session `A` rebuilds at most `X` bits besides
  the sectors of one carrier, and that carrier is the declared circuit `W`'s computation over at most `life` of the
  audit's windows (`RateSketch.Provenance.OfCircuit`).
* `SteeredLayout a layout`: at each prefix of the session's coins, a pass's layout reads the codec and the append coins
  through at most `a` bits.
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

/-- **Every rebuild beyond `X` is decoded from a carrier of the declared circuit's.** For the session `A` at the
deployed parameters, every pass device (window `w` after the earlier windows' coins `η`, pass `j` after the earlier
passes' coins `pre`, candidate `c`, at signs `σ`) rebuilds at most `X` bits besides the sectors of one carrier
(`RateSketch.Carrier`), and that carrier's provenance is `W`'s over a set `I` of at most `life` of the audit `Aud`'s
windows: its inputs are at most `W`'s run-input bits there, and its encodes, at the gate model's `encodeBitsCost` for
the deployed codec (p1024, 3 rounds) in the restricted class (`B` operand bits, more than `εnum/εden` of the draws),
cost at most `W`'s budgeted units there (`RateSketch.Provenance.OfCircuit`).

**Why it should hold.** In the verbatim class a device obtains a sector it doesn't hold, read or take over the link by
rebuilding it: computing it from memory whose contents are a circuit's values. The rebuilds the node's compute pays
for, beyond the declared circuit's own, are `X`: the squaring ceiling's bits a pass (`b·F/(rounds·c_root)`,
`SquaringCeiling` with `WorkHardness`, within the pass's deadline). Every other rebuild comes from memory the declared
circuit wrote after the seed, a credited KV page, so its provenance is `W`'s computation. The windows are the life of
that provenance back to values that depend on no store sector: the stacking property's life clause on run-produced
values (registered inputs, the weights, are payloads fixed before the seed, constants in a provenance), and a block
re-keyed at the life cap re-encodes under a fresh coin, so an older computation's encodes of it are constants too. The
encodes are in the restricted class when `W` is (`Gates.Restricted`, the IR's one-word rule). Not covered: a carrier
older than `life` windows of a block it encodes that is never re-keyed, such as a prefix-cached KV page computed from
the registered weights' encodings more than `life` windows before the pass; a deployment caps a cached page's age at
`life` or takes `life` as that age. -/
def CircuitCarriers {n P N ρ S L q G : ℕ} {ι : Type}
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S L q 16 G 2773) (X life B εnum εden : ℕ) (W : Gates.Dag ι)
    (Aud : Gates.Audit) : Prop :=
  ∀ w η j pre c σ, ∃ (K : RateSketch.Carrier N 128) (I : Finset ℕ),
    (∀ Es R O,
      let dv := (A.window w η).device j pre c
      let D := RateSketch.Data.ofParts Es R
      (dv.prog σ D).rebuilds O D * 256 ≤ X + K.carried.card * 256) ∧
    I ⊆ Finset.range Aud.windows ∧ I.card ≤ life ∧
    K.provenance.OfCircuit (encodeBitsCost Gates.p1024 3 B εnum εden) W Aud I

/-- **A pass's layout reads the codec and the append coins through at most `a` bits.** At every prefix `c` of the
session's coins, the layout `layout c y` takes at most `2^a` values as the codec and the append coins `y` vary: it is
`slot c` of an `a`-bit summary `steer c y`.

**Why it should hold.** The layout is the memory map's schedule: which blocks are resident and planned when the pass's
step begins. The memory map makes it a function of the served steps (frees at last read, writes at production, re-keys
at the life cap only), and the served steps are a function of what the prover chose: the requests' run inputs, their
arrival times and their scheduling. Whatever of those it chose after an append's coin can follow that append's
encodings. `a` counts those choices within the life of the blocks the pass holds: the schedule's arrivals, lengths,
prefix-cache hits and preemptions, and every run input that moves them. A schedule-determinism check (the served steps
a function of the run inputs, their arrival order and the coins) bounds `a` by those choices; without it, timing is any
function of the encodings and `a` is unbounded. -/
def SteeredLayout {Pfx Y : Type} {N T M : ℕ} (a : ℕ) (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) : Prop :=
  ∃ (slot : Pfx → Pous.Bits a → Fin N ↪ Fin T × Fin M) (steer : Pfx → Y → Pous.Bits a),
    ∀ c y, layout c y = slot c (steer c y)

end Accounting.Assumptions
