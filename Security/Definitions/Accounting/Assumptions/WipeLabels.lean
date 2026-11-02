import Definitions.Accounting.WipeLabels

/-!
# What the streamed wipe's labels assume (trusted)

The survivor bound an accepted `verity/wipe/v0` session publishes (`MemoryChallenger.Properties.WipeSurvivorBound`) pays
the labels' PRF advantage (`Pous.StreamedWipe.prfAdvantage`) against the distinguisher the adversary's moves make
(`Accounting.Wipe.Strategy.distinguisher`). The labels are the deployed ones (`Accounting.Wipe.deployedLabels`:
`SHA-512(LABEL_TAG ‖ K ‖ u64be i)` under the challenger's 256-bit key `K`), and `WipeLabelsPrf` is the premise that
bounds that advantage: SHA-512 keyed by `K` this way is a PRF (`prf/sha-512`). It is the wipe's own premise, about these
labels alone.
-/

namespace Accounting.Assumptions

open scoped ENNReal

/-- **The deployed labels are a PRF against the adversary's distinguisher** (`prf/sha-512`). At the hash `H`, the
distinguisher `D`, with coins `ω`, outputs true on the `N` deployed labels of a uniform 256-bit key
(`Accounting.Wipe.deployedLabels H N`) at most `ε t N` more often than on `N` uniform labels:
$$\Pr_{x, K}\big[D(x, \mathrm{labels}_H(K))\big] \le \Pr_{x, L}\big[D(x, L)\big] + \varepsilon(t, N).$$
It bounds the PRF advantage `Pous.StreamedWipe.prfAdvantage (deployedLabels H N) D` by `ε t N`
(`Accounting.Guarantees.WipeLabelsAdvantage`).

A statement takes it at the distinguisher the adversary's moves make (`Accounting.Wipe.Strategy.distinguisher`: it runs
the strategy on the labels it is given and outputs whether the observed event holds, the strategy reads the foreign
content back and its egress is in the certified set), at `t` the strategy's SHA-512 calls and `H` the compiled SHA-512,
as `Hash.Assumptions.LmsUnforgeable` is taken at one forger. Lean has no cost model for `D`, and at every `D` the premise
is false for a fixed `H` at small `ε`: the distinguisher that asks whether the labels are `deployedLabels H N K` for
some key `K` holds on every key, and on uniform labels with probability at most `2^256 / 2^(512 N)`.

The key is the challenger's (`derive(secret, KEY_DOMAIN, {server, session})`), read as uniform: the challenger keeps its
secret (`Secret`), so nothing of the key reaches the device except through the labels it streams. -/
def WipeLabelsPrf (H : Hash.Msg → Hash.Msg) (ε : ℕ → ℕ → ℝ≥0∞) {N : ℕ} {ω : Type} [Fintype ω]
    (D : ω × Accounting.Wipe.Labels N → Prop) (t : ℕ) : Prop :=
  Pous.pr (fun x : ω × Pous.Bits 256 => D (x.1, Accounting.Wipe.deployedLabels H N x.2)) ≤ Pous.pr D + ε t N

end Accounting.Assumptions
