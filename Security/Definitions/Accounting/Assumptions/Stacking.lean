import Definitions.Accounting.Carrier

/-!
# Stacking's named assumptions: bound keys, the declared circuit's carriers, the steered layout (trusted)

What the stacking guarantees (`Definitions/Accounting/Guarantees/Stacking.lean`, `DeclaredCircuit.lean`) assume beyond the
remote audit's own and the gate model's `Accounting.Assumptions.SquaringFloor` and `Accounting.Assumptions.ChainSum`,
each a named `Prop` taken as a hypothesis, never an `axiom`:

* `KeyBinding T M δ enc`: under the append-coin rule (`Accounting.appendCoinBlocks`), whatever the writers choose for
  each append from the codec and the coins of the appends before it, the blocks a pass reads at any layout fixed apart
  from the encodings are on every event at most `δ` likelier than uniform blocks (claim `key-binding`).
* `CircuitCarriers A store layout joined X L B εnum εden W Aud`: when each append's blocks are read within `L` windows
  of the one it joined (`RateSketch.LifetimeCapped`), every pass device rebuilds at most `X` bits of its appended tiles
  besides the sectors of one carrier, and that carrier is the declared circuit `W`'s computation over at most `L` of
  the audit's windows (`RateSketch.Provenance.OfCircuit`). Registered tiles, whose keys are fixed for the
  registration, are not bounded.
* `SteeredLayout layout steer`: at each prefix of the session's coins, a pass's layout reads the codec and the append
  coins only through the `a`-bit summary `steer` its devices are told. What `a` is, is the schedule check's
  (`Accounting.ScheduleDeterministic`, `RemoteAudit.Schedule.check`): the bits of the schedule each credited plan is
  the verifier's plan of.
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

/-- **Under the lifetime cap `L`, every rebuild of an appended tile beyond `X` is decoded from a carrier of the
declared circuit's over `L` windows.** Let each append's blocks be read within `L` windows of the one it joined
(`RateSketch.LifetimeCapped layout joined L`, `joined x t` the window append `t` joined). Then every pass device, on the
store `store c y` the pass certifies, rebuilds at most `X` bits of its appended tiles (`RateSketch.appendedTiles`: those
the layout `layout c y` puts at positions of appends after the registration) besides the sectors of one carrier
(`RateSketch.Carrier`). The device is the one the pass runs: at the prefix `c` of the session's coins before the pass
and the codec and the append coins `y`, the session `A c y`'s device at candidate `g`. The carrier's provenance is
`W`'s over a set `I` of at most `L` of the audit `Aud`'s windows, `L` the lifetime cap itself: its inputs are at most
`W`'s run-input bits there, and its encodes, at the gate model's `encodeBitsCost` for the deployed codec (p1024,
3 rounds) in the restricted class (`B` operand bits, more than `εnum/εden` of the draws), cost at most `W`'s nonlinear
units there, its budgeted units and its data products (`RateSketch.Provenance.OfCircuit`). Rebuilds of registered tiles (append `0`: the weights, the pool's
filler) are not bounded, so a guarantee that reads this credits no registered block.

**Why it should hold.** In the verbatim class a device obtains a sector it doesn't hold, read or take over the link by
rebuilding it: computing it from memory whose contents are a circuit's values. The rebuilds the node's compute pays
for, beyond the declared circuit's own, are `X`: the squaring ceiling's bits a pass (`b·F/(rounds·c_root)`,
`SquaringCeiling` with `WorkHardness`, within the pass's deadline). Every other rebuild of an appended block comes from
memory the declared circuit wrote, a credited KV page, so its provenance is `W`'s computation. An appended block's key
reads its append's coin, drawn after its payload is committed (`KeyBinding`), so what was computed before that coin
carries nothing about its encoding, and only work after the coin helps rebuild it. Two claims about the code back the
rest:

* no pass reads an append's blocks `L` or more windows after the one it joined (remote_bandwidth's `lifetime`,
  `Properties.RemoteAudit.LifetimeCapped`, proved), whose game form is this premise's hypothesis;
* a re-key appends the block again and draws a fresh coin (`Properties.RemoteAudit.AppendsInOrder`, proved
  with the live coins uniform): each append has its own uniform coin, as in the game's coins.

So the work that helps rebuild an appended tile a pass reads lies in the `L` windows ending at the pass, counting the
window its append joined (whose census holds the steps after its coin), however old the carrier's own chain. The
encodes are in the restricted class when `W` is (`Gates.Restricted`, the IR's one-word rule). Without either claim an
appended block's carrier outlives `L` windows and this is false. Registered tiles are left out because it fails for
them: their keys never change, so a carrier of one built at any time still helps, it survives same-payload re-keys of
its own block and free copies, and the stock of such carriers grows without decay. -/
def CircuitCarriers {n P N ρ S Lk q G T M : ℕ} {ι Y : Type}
    (A : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Y →
      RateSketch.SketchDeployedSession n P N 128 ρ 512 S Lk q 16 G 2773)
    (store : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Y → RateSketch.SketchStore N 128)
    (layout : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Y → Fin N ↪ Fin T × Fin M)
    (joined : Y × (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) → Fin T → ℕ)
    (X L B εnum εden : ℕ) (W : Gates.IntDag ι) (Aud : Gates.Audit) : Prop :=
  RateSketch.LifetimeCapped layout joined L →
  ∀ (c : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773) (g : Fin G) (y : Y),
    ∃ (K : RateSketch.Carrier N 128) (I : Finset ℕ),
      (∀ R O,
        let dv := ((A c y).window c.w c.windows).device c.j c.passes g
        let D := RateSketch.Data.ofParts (store c y) R
        (dv.prog c.signs D).rebuildsIn (RateSketch.appendedTiles layout c y) O D * 256 ≤ X + K.carried.card * 256) ∧
      I ⊆ Finset.range Aud.windows ∧ I.card ≤ L ∧
      K.provenance.OfCircuit (encodeBitsCost Gates.p1024 3 B εnum εden) W Aud I

/-- **A pass's layout reads the codec and the append coins only through the `a`-bit summary its devices are told.**
At every prefix `c` of the session's coins, the layout `layout c y` is `slot c` of the summary `steer c y`: it takes at
most `2^a` values as the codec and the append coins `y` vary, and the pass's devices, told `steer c y`
(`RateSketch.SketchSummarySession`), know the layout they run on.

**Why it should hold.** The layout is the memory map's schedule: which blocks are resident and planned when the pass's
step begins. The memory map makes it a function of the served steps (frees at the values' last structural reader,
writes at production, re-keys at the life cap only). What the prover chose after an append's coin can follow that append's encodings, and the
summary is the part of it the layout reads. The prover that steers the layout knows that part, so its devices are told
it, and this premise is only that the layout reads nothing more of `y`. It holds of every layout at `a = |Y|`, so `a`
is what the schedule check bounds (`Accounting.ScheduleDeterministic`): the verifier credits nothing after the seed
unless every pass's plan is its own plan (`RemoteAudit.Schedule.plan`) of the schedule the prover publishes (when each
block is written and freed, and each pass's step), and that schedule is within the cap on its bits the prover
committed before the session's first coin (`audit.Params.schedule_cap`), which is what it credits
(`audit.Ledger.steering`). `a` is one number for every coin, so it is that cap, not the bits of the schedule
published after the coins, whose length can follow them. A layout the check accepts at cap `a` at every coin is
`SteeredLayout` at `a + 1` (`Properties.RemoteAudit.steered_of_checked`, under
`Properties.RemoteAudit.ScheduleLayoutGap`: the model's layout is the verifier's positions of the plan the pass
sent), and a layout that follows a
coin at a counted schedule is refused (`Properties.RemoteAudit.coinLayout_not_scheduled`, `ScheduleFixesLayout`). The
schedule holds the run inputs and arrival steps the prover chose after the coins of the blocks the pass holds
(arrivals, lengths, prefix-cache hits and preemptions, and every run input that moves them), and more: the whole
session's, so its bits bound theirs from above. -/
def SteeredLayout {Pfx Y : Type} {N T M a : ℕ} (layout : Pfx → Y → Fin N ↪ Fin T × Fin M)
    (steer : Pfx → Y → Pous.Bits a) : Prop :=
  ∃ slot : Pfx → Pous.Bits a → Fin N ↪ Fin T × Fin M, ∀ c y, layout c y = slot c (steer c y)

end Accounting.Assumptions
