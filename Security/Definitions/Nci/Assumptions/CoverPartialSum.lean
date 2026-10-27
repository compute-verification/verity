import Definitions.Nci.CoverPartialSum

/-!
# The cover theorem's partial-sum tightening: the new premise

The partial-sum bounds rest on one new premise beside the cover theorem's `UpdateHardness`: `ActivationReads`, B2's
(X), which charges a block for the activations it uses one by one. At `f = 1` it is no premise but a theorem of the
keyed structure (`Guarantees.KeyedActivationReads`); above that it is a named `Prop`, so that what each bit per
activation costs is visible. It is a statement about the training circuit `C`, its products, a keyed token structure
(`Keyed`) and a cube (`Cube`), at block size `s` and read cap `k`.
-/

namespace Nci.Cover.Assumptions

/-- NEW PREMISE. **Activation reads** ([ACTIVATION-READS]), B2's (X), at `f` bits per activation and entry width `a`:
an inference block that is handed fewer than `f · a` gates of a token `t` (`Keyed.tokReads`; the token is *low* for
it) computes whole no entry of `t` (some counted product of each entry of `t` is not in `M.whole I`), and is handed `f`
gates of `t` for each distinct activation of the counted products of `t` it computes whole.

At `f = 1` it holds with `a = min(d − E, n)` under `ColumnsIncompressible`, given entries of width `n` (`Cube.Wide`):
`Guarantees.KeyedActivationReads` proves it. A block that holds no root of `t` reaches, from each activation it uses, a
value of `t` it reads, distinct activations distinct values (their key paths are disjoint by `Keyed`'s (I) and (F));
holding a root costs `d − E` reads (the fan-out branch's `rootCost`); and an entry whole in the block has `n` distinct
activations.

Rationale for `f > 1`: an activation is a word, and a product's output reads `f` of its bits (A2's cap: `f` is at most
the activation bits a product's output gate reads). A block handed fewer of them computes the rest from upstream: under
`Keyed.PerValueCut g` it holds an exit of all but `g − 1` of the root's legs, and its token costs `d − E − (g − 1)`
reads (Kimi K2: `g = 1` at `f = 4`). No guarantee here derives `ActivationReads` at `f > 1` from it.

It fails, beside where `Keyed`'s (I) fails (grouped-query attention, copies kept as values of their own), when an
activation's bits are cheap functions of fewer handed gates: a block handed one bit of each of many activations, which
computes their other bits from that bit and registered or already-paid values, pays about one gate per activation, so
`f = 1`. -/
def ActivationReads {C : Circuit} {J L Tok Val Mat Row Col : Type*} [DecidableEq J] [DecidableEq Val]
    [DecidableEq Tok] {M : MulBlocks C J L} {d aOut c V w : ℕ} {K : Keyed M Tok Val d aOut c V w}
    (Q : Cube K Mat Row Col) (s k f a : ℕ) : Prop :=
  ∀ I, C.InferenceBlock s k I → ∀ t : Tok, K.tokReads I t < f * a →
    (∀ p ∈ M.P, K.tok (K.act p) = t → ∃ p' ∈ M.P, Q.entry p' = Q.entry p ∧ p' ∉ M.whole I) ∧
    f * (((M.whole I).filter fun p => K.tok (K.act p) = t).image K.act).card ≤ K.tokReads I t

end Nci.Cover.Assumptions
