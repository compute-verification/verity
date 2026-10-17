import Definitions.Accounting.Carrier

/-!
# Stacking's named assumption: bound keys (trusted)

What the stacking guarantees (`Models/Accounting/Guarantees/Stacking.lean`) assume beyond the remote audit's own and
`Accounting.Assumptions.EncodeChainHardness`, a named `Prop` taken as a hypothesis, never an `axiom`:

* `KeyBinding δ view`: for each index (the writers' choices, and in the session forms the coins drawn before the pass),
  the store the writers make, its blocks written after the seed encoded under bound keys, is on every event at most `δ`
  likelier than a uniform store (claim `key-binding`).
-/

namespace Accounting.Assumptions

open Pous (pr)
open scoped ENNReal

/-- **`key-binding`: a store whose blocks written after the seed are under bound keys is, at each index, a uniform
store but for `δ`.** For every index `α` and every set `ev` of stores, the store `view α ω` the writers make at `α`, at a
uniform `ω`, is in `ev` with probability at most `δ` more than a uniform store:

  `Pr_ω[view α ω ∈ ev] ≤ Pr_E[E ∈ ev] + δ`.

`ω` is the codec and its seed, drawn once at the node's one live registration, and everything else the store's contents
depend on but the index. A block registered before the seed is its payload encoded under the seed's key. A block
written after it (a KV page: its payload is a circuit's value, which may depend on other blocks' encodings) is its
payload `p_j` encoded with the bound tweak `b_j = H(seed ‖ j ‖ d_j)`, `d_j = sha512/row/v2(p_j)`, added to every round,
and it decodes only under the digest it was written with. The index is everything the store depends on besides `ω`:

* **the coins seen so far.** In the session forms the index holds the session's coins drawn before the pass, so a page
  written from earlier passes' receipts is in the view; the pass's own coins are not, which the audit rule secures (a
  block appended after a pass's seed is in no plan of that pass);
* **the writers' choices** made by looking at encodings: a writer that encodes `T` candidate payloads and keeps one
  chooses `log₂ T` bits. The guarantees count these choices (`2^a` of them) and pay for them. When the memory map is
  the declared circuit's at every step (`Requirement.StoreMatches`, with the schedule and layout fixed before the seed),
  frees, append order, indices and dummy appends are not choices: what is left is the payload bits outside live values
  (padding, dead values) and run inputs chosen by their encodings, re-key times, and a schedule chosen after the seed.

Given the index, a payload may be any function of other blocks' encodings and of the coins seen, but no writer keeps a
payload for its encoding beyond the index.

**Why it should hold.** The ideal-encoding step. A round `x ↦ ρ((x + c_r + t) mod p)` is not a random permutation:
`ρ(p − x) = p − ρ(x)`, and the parity of `ρ(x)` is the Legendre symbol of `x` (`p ≡ 3 mod 4`). The ideal model is a
random function on pairs `{x, p − x}` to even residues `e`, the round's output `e` or `p − e` by the sign of its input.
Under it a word's encoding, given its key, tweak and payload, is uniform on `[0, p)`: its last round's input lies in a
fresh pair, which takes a fresh value `e`, and its sign is uniform given the pair. Two words' encodings are independent
unless some round's inputs agree up to sign. A bound block's tweak is fresh for each payload in the tweak space (`b_j`
is 64 bytes, so `2^512` of the `p` values), so a payload chosen from other blocks and from the coins seen leaves its
block uniform and independent of them, and what the payload carries about other blocks is a carrier
(`RateSketch.Carrier`), charged apart. Making two inputs agree up to sign, or choosing a block's encoding, needs a
payload whose digest gives the tweak it was chosen for, a SHA-512 search. So `δ` is the writers' searches over `2^512`,
the chance that inputs agree up to sign, and the words' values `p` and above (`2^n − p` of `2^n`, at `p = 2^n − c`). A
second registration redraws `ω`, a choice of `log₂` the registrations, which the one live registration rules out.
Under the seed's key the writer picks the encoding by writing its decoding: measured in
`verity/core/protocols/remote_bandwidth/tests/test_key_binding.py`, which also fails 64 tries under the bound key. -/
def KeyBinding {St Ω Ch : Type} [Fintype St] [Fintype Ω] (δ : ℝ≥0∞) (view : Ch → Ω → St) : Prop :=
  ∀ (α : Ch) (ev : Set St), pr (fun ω => view α ω ∈ ev) ≤ pr (fun E : St => E ∈ ev) + δ

end Accounting.Assumptions
