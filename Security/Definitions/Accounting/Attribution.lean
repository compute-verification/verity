import Definitions.Accounting.Physical

/-!
# Compute attribution's vocabulary: the outside's advice about a block's coin (trusted)

The physical layer that takes PoUW's logical certificate (decisions 111 and 113: the certificate says the work was done,
the layer says which spatial unit did it) reads one quantity besides the actual circuit's placement: how much the rest
of the datacenter can learn, during the window, about the coin that only the block holds. PoUW's floor counts the work
that depends on the window's salt; a helper outside the block that learns `k` bits about the salt can do at most what
`k` bits of guessing buy, so the block's own credit is bounded below once `k` is.

* `OutsideAdvice S cert X a b k`: across the strategy `S`, what the outside `(X ∪ cert)ᶜ` does during `[a, b)` takes
  at most `2^k` values (`Actual.Run.acts`). Under salt custody (the coin reaches the window only through `X`'s secret
  inputs, `Secrets`) this is `k` bits of advice about the coin, whatever the outside computes from them.
* `advice H krec a b`: the bits B4 (`Accounting.Guarantees.B4Covered`) charges the outside of a spatial unit of `H`
  during `[a, b)` when the certifiers' records placed in it take at most `2^krec` values: `krec`, and each certifier's
  jitter bits on each bucket the interval meets.

Crossing bytes attribute nothing: a helper holding the weights needs only the salt in and a root out, so ingress
volume bounds no outsourced work. What bounds it is the outside's advice about the salt, which the certifiers' records
and their jitter carry and nothing else does (B4).
-/

namespace Accounting.Attribution

/-- **The outside's advice** (across the strategy `S`): what the outside `(X ∪ cert)ᶜ` of the location `X` and the
certifiers `cert` does during `[a, b)`, its gates, what it sends and its state (`Actual.Run.acts`), takes at most `2^k`
values. When the coins reach the window only through `X`'s secret inputs (`Secrets`), everything the outside does with
the coins in the window is a function of `k` bits about them. -/
def OutsideAdvice {Ω L : Type*} [LinearOrder L] [Fintype L] (S : Ω → Actual.Run L) (cert X : Finset L) (a b k : ℕ) :
    Prop :=
  (Set.range fun r => (S r).acts (X ∪ cert)ᶜ (Set.Ico a b)).encard ≤ ((2 ^ k : ℕ) : ℕ∞)

/-- **The advice bits B4 charges** a spatial unit of `H` during `[a, b)`: `krec` bits for the certifiers' records placed
in the interval, and `jitter` bits for each certifier on each bucket of ticks the interval meets. -/
noncomputable def advice {L E : Type*} (H : Hardware L E) (krec a b : ℕ) : ℕ :=
  krec + H.jitter * ((Finset.Ico a b).image H.bucket).card * H.cert.card

end Accounting.Attribution
