import Definitions.Accounting.Assumptions.EncodeChain

/-!
# Bits for free: the encode chain's conjecture fails in the circuit model (trusted statement)

In `Definitions/Accounting/Gates/Model.lean` as it stands, a Boolean circuit of any size costs no unit: AND of two
values in `{0, 1}` is an s8-family step `0 + u·v` with u8 operands, whose derived interval is `[0,1]·[0,1] = [0, 1]`,
and NOT is the affine map `1 − u`, derived `[0, 1]`; neither is budgeted or a gather, so `units` charges neither, and
`Proven` holds with every intermediate value declared `[0, 1]`. A coin byte's bits enter through table reads
(`gather (some 8) 0` over constant `0`/`1` columns), at most `9` units each. So any function of the codec's coin
bytes, its `k` round constants, its mask and word 1's tweak (`(k + 2)·n/8` bytes), is computed for a number of units
that doesn't grow with the number of words: every word's tweak is `i·t_1 mod p`.

* `EncodeChainRefuted`: inside its floor, `Accounting.Assumptions.EncodeChainHardness F k B E ε` is false at every
  `E ≥ 1` and every `B`, at a field whose width is whole bytes with room below `p` for more distinct words than the
  units' bound, p1024 at `k = 3` among them.

The units bound here (`9` a bit read, `72` a key byte) is crude; a port of the model counts `8` a key byte,
`8·(k + 2)·n/8 = 5,120` at p1024 and `k = 3` (`benchmarks/property_redteam/bitwise_codec.py`, measured at `⟨8, 5⟩` and
`⟨16, 17⟩`). Its proof is `Accounting.SecurityProofs.Gates.EncodeChainRefuted`, in
`Security/Proofs/Accounting/Gates/Bitwise.lean`.
-/

namespace Accounting.Guarantees.Gates

open Accounting.Gates

/-- **The encode chain's conjecture is false in the model.** For every field `F` whose width is whole bytes
(`8 ∣ n`), inside `EncodeChainHardness`'s floor (`ε ≥ 2^−64`, `k ≥ 1`, `p ≥ 2^66`, `p ≡ 3 mod 4`), at every `ε < 1`,
every `B` and every `E ≥ 1`, with `72·(k + 2)·n/8 < p` (room for that many distinct words below `p`), the conjecture
`EncodeChainHardness F k B E ε` fails: a restricted circuit with no budgeted gate and no run input computes
`72·(k + 2)·n/8 + 1` distinct words on every draw at no more than `72·(k + 2)·n/8` units. Every guarantee that takes
it as a hypothesis at such an `F` and `k` (p1024 at `k = 3` among them) holds vacuously. -/
def EncodeChainRefuted : Prop :=
  ∀ (F : Field) (k B E εnum εden : ℕ), εden ≤ 2 ^ 64 * εnum → εnum < εden → 0 < k → 2 ^ 66 ≤ F.p →
    F.p % 4 = 3 → F.n % 8 = 0 → 72 * ((k + 2) * F.bytes) < F.p → 0 < E →
    ¬ Accounting.Assumptions.EncodeChainHardness F k B E εnum εden

end Accounting.Guarantees.Gates
