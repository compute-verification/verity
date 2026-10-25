import Definitions.Pous.P2.Rounds

/-!
# The mutable store: versioned, coin-keyed blocks and the snapshot an audit window reads (trusted)

PoUS as an obligation keeps the device's memory full of encodings while the prover serves: the weights are encoded once,
and each KV page is re-encoded when it is written, under the key pair of a fresh tag `salt ‖ page ‖ version ‖ coin`, the
coin drawn by the memory challenger after the page's plaintext commitment. In the ring model (`ringRO`) the store's oracle
is the ring's labeling and one key pair per tag (`Oracle w p T`).

An audit window's reads are answered against the store as it stood at the window's start, its **snapshot**: block `j` at
its current version, whose tag is `tag j`. A tag names its page, so `tag` is injective. `snap tag ω` is the snapshot's
primitive, a point of `ringRO N w p`, and `rest tag ω` the key pairs of every tag outside it (earlier versions, and pages
the snapshot doesn't hold).

`CommittedBefore π W` says a payload drawn from the oracle was committed block by block, each before its key coin: block
`j` reads the oracle only through the labeling and the key pairs of the blocks committed before it in `π`, never its own.
-/

namespace Pous.Mutable

variable {N w p : ℕ} [NeZero p] {T : Type}

/-- The mutable store's oracle: the ring's random odd labeling and one key pair (mask, offset) per tag. -/
abbrev Oracle (w p : ℕ) [NeZero p] (T : Type) : Type := OddLabeling p × (T → Bits w × Fin p)

/-- The tags outside the snapshot `tag`. -/
abbrev Rest (tag : Fin N → T) : Type := {t : T // ∀ j, tag j ≠ t}

/-- The key pairs of the tags outside the snapshot. -/
abbrev RestKeys (tag : Fin N → T) (w p : ℕ) : Type := Rest tag → Bits w × Fin p

/-- The snapshot's ring-model primitive: the labeling, and block `j`'s key pair at its current tag. -/
def snap (tag : Fin N → T) (ω : Oracle w p T) : (ringRO N w p).Ω := (ω.1, fun j => ω.2 (tag j))

/-- The key pairs of every tag outside the snapshot. -/
def rest (tag : Fin N → T) (ω : Oracle w p T) : RestKeys tag w p := fun t => ω.2 t.1

/-- The payload `W ω`, as `N` blocks of `b` bits, was committed block by block before its key coin: block `j` reads the
oracle only through the labeling and the key pairs of the blocks committed before it (`π i < π j`), never its own. -/
def CommittedBefore {b : ℕ} (π : Fin N → ℕ) (W : (ringRO N w p).Ω → Bits (N * b)) : Prop :=
  ∀ (j : Fin N) (ω ω' : (ringRO N w p).Ω), ω.1 = ω'.1 → (∀ i, π i < π j → ω.2 i = ω'.2 i) →
    (codeEquiv N b).symm (W ω) j = (codeEquiv N b).symm (W ω') j

end Pous.Mutable
