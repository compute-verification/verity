import Definitions.Accounting.RateAuditRemote

/-!
# What reading the remote audit's live pass coins as fresh oracles assumes (trusted)

`Accounting.Guarantees.RateLiveSeedPaidRO` and `RateLiveSeedPaid` bound a window whose pass `j` is seeded by the
challenger's live coin `c_j` (`Accounting.Rate.LiveChain`, `Accounting.Rate.atSeeds`) under one premise, a named `Prop`
taken as a hypothesis:

* `Accounting.Assumptions.LiveSeedRandom H P s ε`: the random-oracle premise (`random-oracle`). The pass oracles
  SHA-512 gives at `P` uniform live coins are within `ε` of `P` independent uniform oracles: no event of the store and
  the window's coins is more than `ε` likelier at the hashed coins than at uniform ones.
-/

namespace Accounting.Assumptions

open Accounting.Rate
open Pous (Bits)

variable {Hs : Type} {N b h ρ κ : ℕ}

/-- `random-oracle` (the random-oracle card, 9cd57182, ASK-16: PoUW only, each use a named premise; this use, for the
remote audit's live pass coins, is asked of Daniel with `SketchHashRandom`'s). **The pass oracles SHA-512 gives at
the window's live coins are within `ε` of independent uniform oracles**: over SHA-512's outputs `x`, a uniform store
and each pass's uniform live coin `c_j` of `κ` bits, stream and sample, no event `E` of the store and the window's
coins (the live coins among them) is more than `ε` likelier with pass `j`'s oracle the hash keyed by `c_j`
(`atSeeds (H x)`) than with a uniform oracle of its own, independent of `c_j`:
$$\Pr_{x, E, \omega}\big[E(E, \mathrm{atSeeds}(H_x, \omega))\big] \le \Pr\big[E\big] + \varepsilon.$$

It is the one random-oracle step `RateLiveSeedPaidRO` and `RateLiveSeedPaid` take: `RateLivePaidRO` and `RateLivePaid`
bound the window over the game's coins (each pass's oracle uniform and independent of the store, the other passes'
coins, its own live coin and the devices they pick), and this premise carries the bound to the oracles at the deployed
coins, at a cost of `ε`. The oracle at a coin is the receipt hash keyed by it: `remote_bandwidth.receipts`' receipt, sketch signs and tree
binding are SHA-512 with `seed/v1`'s seed, the pass coin's 64 bytes, in their prefix (`κ = 512`).

SHA-512 is one fixed function, and for it the premise is false in Lean at small `ε` (with `Hs` a point, the oracle at a
coin is one fixed function of the coin, not a uniform oracle). So it is a premise, read in the random-oracle model, and
not a lemma. There `Hs` is SHA-512 drawn as a uniform function and `H x c` its receipt hash keyed by `c`: coins that
differ give independent uniform oracles, so the law of the window's coins is within the probability that two of its
`P` coins are equal of uniform, at most `P²/2^(κ+1)`. A prover's other queries to SHA-512 (any it made before a coin
was released, on the node or off it) are independent of the oracles at coins it never queried, so a device may depend
on them as on any fixed value; a query at a coin before its release, at most `Q` queries over the session, adds at most
`P·Q/2^κ`. So `ε ≤ P·(P + Q)/2^512`. No grinding term: the coin is drawn after the root it follows, so the prover has
no seed to try before committing. -/
def LiveSeedRandom [Fintype Hs] (H : Hs → Bits κ → Oracle N b h ρ) (P s : ℕ) (ε : ℝ) : Prop :=
  ∀ E : Store N b × (Fin P → LiveCoins N b h ρ κ s) → Prop,
    Pous.pr (fun x : Hs × Store N b × (Fin P → SeedCoins N ρ κ s) => E (x.2.1, atSeeds (H x.1) x.2.2))
      ≤ Pous.pr E + ENNReal.ofReal ε

end Accounting.Assumptions
