import Definitions.Nci.Cover

/-!
# The cover theorem's fan-in/fan-out tightening: the vocabulary

The covering bound (`Nci.Guarantees.CoveringBound`) charges a block for the weights and the tokens of the
multiplications it completes. Fan-in and fan-out charge it for the values it reads: a block that reads a value of a
token (an activation, a word) finishes
only the products whose activations key back to that value, at most `aOut` of them, and to do more of a token's work
without reading more of it, it must hold one of the token's accumulation chains whole, which costs `d` reads less the
embedding lookups it can afford. The structure that makes this precise is `Keyed`, a structural property of the training
computation, checkable on a named circuit, as `MulBlocks.Upstream` is for the bridge to the NCI check.

* `Circuit.exits`, `Circuit.Link`: where a gate set's value leaves it, and a path from one gate set into another.
* `Follows`: following keys from value to value.
* `Keyed`: the keyed token structure, with values, injective keys, roots with legs, lookups with the columns whose
  entries they read, activations read `f` bits wide, and reach.
* `Keyed.readSet`, `Keyed.heldSet`, `Keyed.Spent`: the values a block reads, the ones it holds, and the reads the
  fan-out argument charges.
-/

namespace Nci.Cover

namespace Circuit

/-- **The exits of a gate set `X`**: its gates that some gate outside `X` reads. A value reaches the rest of the circuit
only through its exits. -/
def exits (C : Circuit) (X : Finset (Fin C.N)) : Finset (Fin C.N) :=
  X.filter fun g => ∃ h, g ∈ C.pred h ∧ h ∉ X

/-- **A link from `U` into `X` ending at `g`**: a gate `x` of `X` reads an exit of `U` and reaches `g` inside `X`. The
path from that exit to `g` enters `X` at once and stays in it, so a block that computes `g` and is handed no gate of `X`
holds the exit of `U` the link starts from (`SecurityProofs.entry`). -/
def Link (C : Circuit) (U X : Finset (Fin C.N)) (g : Fin C.N) : Prop :=
  ∃ h ∈ C.exits U, ∃ x ∈ X, h ∈ C.pred x ∧ C.ReachesIn X x g

end Circuit

/-- **Following keys** from `v`, through values outside `stop`, reaches `u` (the paper's `v ⤳ u`, with `stop` the roots
and lookups). A key is a function, so the values `v` reaches lie on one path. -/
inductive Follows {Val : Type*} (key : Val → Option Val) (stop : Finset Val) : Val → Val → Prop
  | refl (v : Val) : Follows key stop v v
  | step {v κ u : Val} : v ∉ stop → key v = some κ → Follows key stop κ u → Follows key stop v u

/-- **A keyed token structure** for the counted products of `M` (the fan-out certificate): roots of at least `d` legs,
at most `aOut` counted products reached from any one value, at most `c` per token and parameter, lookup columns of at
least `V` updated parameters, and activations read `f` bits wide. Like `MulBlocks.Upstream`, it is a structural property
of the training computation, checkable on a named circuit.

Each token's gates are cut into *values*: words such as an activation, a residual or a whole accumulation chain,
pairwise disjoint and holding no registered parameter. Every value is a *stop* or has a *key*, a value of its token from
which a link enters it at each of its exits (K). No two values share a key (I), so a copy, a capacity slot, or a
router-scaled or re-quantized version of a value is a gate of the value, not a value keyed to it, and the values keyed
back from any one form a single chain. No value is keyed to a counted product's activation or to a root's leg (F): a
version of one is part of it. The stops are:

* *roots*, whole accumulation chains (a dot product's products and partial sums), entered at each exit by a link from
  each of at least `d` *legs* whose key paths are disjoint (R). A link's gates after its first lie in the root, so the
  root holds a whole path from every leg: a partial sum is a gate of its root, never a value of its own, and a block
  that enters a chain midway reads its root;
* *lookups*, embedding gathers, each selecting from a *column* of at least `V` parameters, the columns of one token's
  lookups disjoint. A lookup's gates are exactly the gates after an entry of its column on a path from the entry to one
  of the lookup's exits: every mux level and every shard's partial gather is inside, and the one-hot decode of the
  token id is outside. A column lists only entries updated at that step, outside `θ`, each feeding its lookup (L).

Each counted product reads an *activation*, a value of its token from which a link enters the product at its output
gate (K), and the output gate reads `f` of the activation's exits (W): the product's output (`MulBlocks.o`) is declared
at a gate that reads the whole `f`-bit word. At most `aOut` counted products have activations whose key paths reach
any one value, and none reaches a lookup (A).

Kimi K2, forward products (`c = 1`, `f = 4`). A token is one position at one step. Keys run from a residual to its
branch, from a norm past its shared scale, from an attention output through its diagonal `v`, and from an MoE output to
its shared expert. A dispatch copy belongs to its routed token's MoE-input value (I), and so does the NVFP4 code the
routed experts read, since the router's products read the BF16 word it codes (F). The roots are the weight products'
accumulation chains, the narrowest kv_b's over the 512-wide kv latent (`d = 512`). The lookups are the trained
embedding's gathers (`V = 163840`). The value reaching the most products is the MoE input: the router's 384 products
plus 9 × (gate + up) × 2048, so `aOut = 37248`. Merging a version into its value doesn't change this, since the
products whose key paths reach the version already reach the value.

With the backward's Dgrad products counted, backward values share their forward token (`c = 2`: a weight's forward
product and its Dgrad, per token). Each keys into that token's forward values (dk_nope to a q_nope root, dgate to an
up root), `d` stays 512, and the merged MoE-output gradient reaches 9 × 2048 = 18432 products. Dgrad can't be counted
as its own tokens' (`c = 1`): a backward-only token has no valid lookup, and even with one, (I) would merge a head's 128
dk_nope into one value reaching 65536 products, more than `aOut`. Counting Dgrad raises `α₀` from 0.148 to 0.301
(`kappa.py` at sequence 4096).

Scope:

* the LM head's forward products are not counted, since their activation reaches 163840 products, more than `aOut`;
* layer 0's q_a and kv_a products, whose activation is keyed to an embedding gather (about 0.05%), are not counted
  (`reach_lookups`);
* norm-γ and NVFP4 block-scale products are not counted: either kind, keyed into the MoE input, would raise its reach
  past `aOut` (to 37249 and 74112). The counted products are `kappa.py`'s matrix products, 8 routed slots per token,
  without recomputed ones;
* with Dgrad counted, position 0's dq_rope has 4096 entries on 128 keys, so (I) merges them 32 to a value, reaching
  49152 products. The W_q_b Dgrad products of 1024 of those entries per layer are not counted (3.8·10⁻⁷ of the counted
  products), which leaves at most 24 to a value; keeping them takes `aOut = 49152`;
* with `d ≥ 1`, roots' legs key down to lookups, since every exit of a value is entered by a link and no link ends at an
  input gate, and a lookup's gates come after its column's updated entries. So a model with a frozen embedding has no
  such structure, and the guarantees say nothing about it. -/
structure Keyed {C : Circuit} {J L : Type*} (M : MulBlocks C J L) (Tok Val : Type*) [DecidableEq Val]
    (d aOut c V f : ℕ) where
  /-- The values. -/
  vals : Finset Val
  /-- The gates of each value. -/
  gates : Val → Finset (Fin C.N)
  /-- The token of each value. -/
  tok : Val → Tok
  gates_disjoint : (vals : Set Val).PairwiseDisjoint gates
  gates_θ : ∀ v ∈ vals, Disjoint (gates v) C.θ
  /-- The activation each counted product reads. -/
  act : J × L → Val
  act_mem : ∀ p ∈ M.P, act p ∈ vals
  /-- (K), for products: a link from the activation enters the product at its output gate. -/
  act_link : ∀ p ∈ M.P, C.Link (gates (act p)) (M.m p) (M.o p)
  /-- (W): the output gate reads `f` bits of the activation: from each of `f` of its exits, a link enters the product
  at the output gate. -/
  act_width : ∀ p ∈ M.P, ∃ H ⊆ C.exits (gates (act p)), f ≤ H.card ∧
    ∀ h ∈ H, ∃ x ∈ M.m p, h ∈ C.pred x ∧ C.ReachesIn (M.m p) x (M.o p)
  /-- At most `c` counted products per token and updated parameter (`c = 1` for the forward pass without weight
  sharing, `c = 2` with Dgrad on the forward's tokens). -/
  mult_le : ∀ (i : Tok) (l : L) (X : Finset (J × L)), X ⊆ M.P → (∀ p ∈ X, tok (act p) = i ∧ p.2 = l) → X.card ≤ c
  /-- The roots: whole accumulation chains. -/
  roots : Finset Val
  /-- The lookups: embedding gathers. -/
  lookups : Finset Val
  /-- The key of a value. -/
  key : Val → Option Val
  /-- (K), for values: every value but a stop has a key in its token, and a link from the key enters the value at
  each of its exits. -/
  keyed : ∀ v ∈ vals, v ∉ roots ∪ lookups → ∃ κ ∈ vals, key v = some κ ∧ tok κ = tok v ∧
    ∀ g ∈ C.exits (gates v), C.Link (gates κ) (gates v) g
  /-- (I): no two values share a key. -/
  key_inj : ∀ v ∈ vals, ∀ v' ∈ vals, ∀ κ, key v = some κ → key v' = some κ → v = v'
  /-- The legs of a root. -/
  legs : Val → Finset Val
  legs_mem : ∀ r ∈ roots, ∀ q ∈ legs r, q ∈ vals ∧ tok q = tok r
  /-- (R): at least `d` legs. -/
  legs_card : ∀ r ∈ roots, d ≤ (legs r).card
  /-- (R): a link from every leg enters the root at each of its exits. -/
  legs_link : ∀ r ∈ roots, ∀ g ∈ C.exits (gates r), ∀ q ∈ legs r, C.Link (gates q) (gates r) g
  /-- (R): the legs' key paths are disjoint. -/
  legs_disjoint : ∀ r ∈ roots, ∀ q ∈ legs r, ∀ q' ∈ legs r, q ≠ q' → ∀ u,
    Follows key (roots ∪ lookups) q u → ¬ Follows key (roots ∪ lookups) q' u
  /-- (F): no value is keyed to a counted product's activation or to a root's leg. -/
  key_fresh : ∀ v ∈ vals, ∀ κ, key v = some κ → (∀ p ∈ M.P, act p ≠ κ) ∧ ∀ r ∈ roots, κ ∉ legs r
  /-- The column of a lookup: the updated parameters it selects from. -/
  col : Val → Finset L
  /-- (L): at least `V` parameters per column. -/
  col_card : ∀ v ∈ lookups, V ≤ (col v).card
  /-- (L): one token's lookups have disjoint columns. -/
  col_disjoint : ∀ v ∈ lookups, ∀ v' ∈ lookups, v ≠ v' → tok v = tok v' → Disjoint (col v) (col v')
  /-- The gates of each updated parameter, a column's entry. -/
  entry : L → Finset (Fin C.N)
  /-- (L), the cone: a lookup's gates are exactly the gates after an entry of its column on a path from the entry to an
  exit of the lookup. -/
  lookup_cone : ∀ v ∈ lookups, ∀ g, g ∈ gates v ↔
    ∃ l ∈ col v, ∃ x ∈ entry l, ∃ y ∈ C.exits (gates v), ∃ π, C.IsPath π x y ∧ g ∈ π ∧ g ≠ x
  /-- (L), the column: its entries are updated at that step (none is registered), each feeds the lookup, and they are
  disjoint. -/
  col_entries : ∀ v ∈ lookups, ∀ l ∈ col v, Disjoint (entry l) C.θ ∧
    (∃ x ∈ entry l, ∃ y ∈ C.exits (gates v), ∃ π, C.IsPath π x y) ∧
    ∀ l' ∈ col v, l ≠ l' → Disjoint (entry l) (entry l')
  /-- (A), reach: at most `aOut` counted products have an activation whose key path reaches any one value. -/
  reach : ∀ (u : Val) (X : Finset (J × L)), X ⊆ M.P → (∀ p ∈ X, Follows key (roots ∪ lookups) (act p) u) →
    X.card ≤ aOut
  /-- (A), entry: no counted product's activation has a key path to a lookup. -/
  reach_lookups : ∀ p ∈ M.P, ∀ u ∈ lookups, ¬ Follows key (roots ∪ lookups) (act p) u

namespace Keyed

variable {C : Circuit} {J L Tok Val : Type*} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V f : ℕ}

/-- The stops: the roots and the lookups. -/
def stops (K : Keyed M Tok Val d aOut c V f) : Finset Val :=
  K.roots ∪ K.lookups

/-- **The values `I` reads**: those with a gate among `I`'s inputs. -/
def readSet (K : Keyed M Tok Val d aOut c V f) (I : Finset (Fin C.N)) : Finset Val :=
  K.vals.filter fun v => (K.gates v ∩ C.inputs I).Nonempty

/-- **The values `I` holds**: it reads none of their gates, but computes one of their exits. -/
def heldSet (K : Keyed M Tok Val d aOut c V f) (I : Finset (Fin C.N)) : Finset Val :=
  K.vals.filter fun v => ¬ (K.gates v ∩ C.inputs I).Nonempty ∧ (C.exits (K.gates v) ∩ C.computed I).Nonempty

/-- **`I` spends `u`**: `I` reads `u`, and `u` is the first value `I` reads on the key path from a value `I` uses, the
activation of a counted product `I` computes whole or a leg of a root `I` holds. These are the reads the fan-out
argument charges. A read that buys no product, such as one bit of a value feeding a single gate, is not spent. -/
def Spent [DecidableEq J] (K : Keyed M Tok Val d aOut c V f) (I : Finset (Fin C.N)) (u : Val) : Prop :=
  u ∈ K.readSet I ∧ ∃ v, ((∃ p ∈ M.whole I, K.act p = v) ∨ (∃ r ∈ K.roots ∩ K.heldSet I, v ∈ K.legs r)) ∧
    Follows K.key (K.stops ∪ K.readSet I) v u

end Keyed

end Nci.Cover
