module

public import Definitions.Accounting.Guarantees.RateAuditSketch

/-!
# The sketch rate audit's honest device and its table: lemmas

Statements beside `Definitions.Accounting.Guarantees.RateAuditSketch`'s guarantees that are none: devices accepted on
every coin. The honest one shows `RateAuditSketch`'s event is not vacuous, and is no witness of it (a witness constructs
its instance, and it quantifies over every digest and size). The table is a counterexample outside the verbatim class,
which the claims registry cites for `rate-sketch-verbatim`, not a property of a function. Each is a `def G : Prop` under
the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and audited with the proofs: no
record pins it.

* `RateSketchHonest`: an honest device that reads every sector once is accepted on every coin, with footprint `N·m·w`.
* `RateSketchTable`: outside the verbatim class, for a prover whose memory holds any function of the store, every
  digest of at most `2^d` values has a device accepted on every coin that reads `d` bits per tile.
* `RateTensorSketchTable`: the same at the deployed tag, `32·k_s` bits per tile, at any number of rows `k_s`.
-/

@[expose] public section

namespace Accounting.Guarantees

open Accounting Accounting.RateSketch

/-- **Completeness of `RateAuditSketch`: an honest device that reads every sector once is accepted on every coin.**
For every digest and every `N`, `m`, `ρ`, `h` and `w` there is a device that holds nothing on chip, whose link carries
exactly the certified stream's `ρ` sectors (`L = ρ·w`) and which makes `N` oracle queries (a
`Prover N m ρ h 0 (ρ·w) N w`), that is accepted at every signs, oracle, data and sample, so with probability 1; its
footprint is the store, `N·m·w` bits, and its compute is `N·c_tag`, with nothing rebuilt. It takes in the stream, reads
every sector once and queries the oracle at each tile's digest under the stream. Against `RateAuditSketch` at `S = 0`,
`L = ρ·w`, `q = N` and `F = N·c_tag`, the honest `N·m·w` sits above the floor `k·m·w − (N − k)·c_tag·w/c_enc` by the
sample's slack `(N − k)·m·w` and the tagging it leaves unspent. Proof: `Accounting.SecurityProofs.RateSketchHonest`. -/
def RateSketchHonest : Prop :=
  ∀ (N m ρ h w : ℕ) (X Sig Dg : Type) [Inhabited X] (dig : Sig → (Fin m → X) → Dg),
    ∃ A : RateSketch.Prover N m ρ h 0 (ρ * w) N w X Sig Dg,
      ∀ (cTag cEnc : ℕ) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (s : ℕ) (I : Fin s → Fin N),
        A.Accepts dig σ O D I ∧ A.footprint σ O D = N * m * w ∧ A.compute cTag cEnc σ O D = N * cTag

/-- **Outside the verbatim class the sketch rate audit charges a prover only for its digest: the table.** For every
digest `dig σ` of a tile of `m` sectors that takes at most `2^d` values, there is a prover of `LayoutDevice`'s class
(`Security/Definitions/Accounting/RateAuditSketchLayout.lean`: memory laid out before the signs as any function of
the store and read one bit at an address, nothing held on chip, a side link that carries the certified stream's `ρ`
sectors and nothing else, nothing rebuilt) whose memory is `|Sig|·N·d` bits, the table `T[σ, t] = dig σ (tile t)` over
every signs `σ` and tile `t`, and which, at every signs, oracle and data, commits every tile's right tag, so is
accepted at every sample, makes `N` oracle queries and reads `N·d` bits of memory.

**The reading.** At `RateSketchHonest`'s budgets (`S = 0`, `L = ρ·w`, `q = N`, `F = N·c_tag`), `RateAuditSketch`'s
floor is `k·m·w − (N − k)·c_tag·w/c_enc` bits. This prover is accepted on every coin and reads `N·d` bits, every bit
that conservation counts (one prover, the reads of every tier of its memory). At 4 KiB tiles
(`m = 128`, `w = 256`) and `k_s = 16` rows (`d = 512`) that is 1/64 of the store, at `k_s = 64` (`d = 2048`) 1/16
(`RateTensorSketchTable`). So no bound of `RateAuditSketch`'s form, at any number of rows and any rank of the signs,
holds for a prover whose memory may hold any function of the store, which is `RateAudit`'s model: a digest that
compresses the tile is what the verbatim class (`rate-sketch-verbatim`) is there for. `RateAudit`'s tag pays in that
model because it hashes the tile itself, so its table is the store. What rules the table out is capacity, not reads:
it is `|Sig|·N·d` bits (`2^65536·N·512` at `k_s = 16`, `2^κ·N·d` with signs sampled from the oracle at a `κ`-bit seed),
and one cut to `M` bits covers a fraction `M/(|Sig|·N·d)` of the signs. A bound for an arbitrary prover must bound its
memory, and with memory near the store's size it is a static cell-probe lower bound for random `±1` sketches, which
this statement neither proves nor refutes.

Proof: `Accounting.SecurityProofs.RateSketchTable`: the chip takes in the stream, then for each tile reads the `d`
bits of its table entry at the signs and queries the oracle at the digest they encode. -/
def RateSketchTable : Prop :=
  ∀ (N m ρ h d : ℕ) (X Sig Dg : Type) [Inhabited X] [Fintype Sig] [Fintype Dg] (dig : Sig → (Fin m → X) → Dg),
    Fintype.card Dg ≤ 2 ^ d →
    ∃ A : LayoutDevice N m ρ h X Sig Dg,
      (∀ (E : Fin N × Fin m → X) (a : ℕ), Fintype.card Sig * (N * d) ≤ a → A.mem E a = false) ∧
      ∀ (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X),
        (∀ t, A.tags σ O D t = O (D.stream, t, dig σ (D.tile t))) ∧ (A.queries σ O D).length = N ∧
          A.reads σ O D = N * d ∧ A.linkSteps σ O D = ρ

/-- **The table at the deployed tag.** `RateSketchTable` for 32-byte sectors, uniform `k_s × n` signs (`2^(32·k_s·m)`
of them) and the tensor-core sketch into `k_s` INT32 accumulators (`d = 32·k_s`): a prover whose memory is a table of
`2^(32·k_s·m)·N·32·k_s` bits, accepted on every coin, that reads `32·k_s` bits per tile. At `k_s = 64` the signs bind
sectors but for a small chance (`SketchBinds`), and the table still reads 1/16 of a 4 KiB tile: more rows make the
digest bind, which prices a guess, and do not make the reads pay for the tile. Proof:
`Accounting.SecurityProofs.RateTensorSketchTable`, from `RateSketchTable`. -/
def RateTensorSketchTable : Prop :=
  ∀ (N m ρ h ks : ℕ), ∃ A : LayoutDevice N m ρ h Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)),
    (∀ (E : Fin N × Fin m → Sector) (a : ℕ), 2 ^ (ks * m * 32) * (N * (32 * ks)) ≤ a → A.mem E a = false) ∧
    ∀ (σ : SketchSigns ks m) (O : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) (D : Data N m ρ Sector),
      (∀ t, A.tags σ O D t = O (D.stream, t, sketch σ (D.tile t))) ∧ (A.queries σ O D).length = N ∧
        A.reads σ O D = N * (32 * ks) ∧ A.linkSteps σ O D = ρ

end Accounting.Guarantees
