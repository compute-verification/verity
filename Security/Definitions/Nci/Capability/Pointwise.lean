import Definitions.Nci.Capability.Consumers

/-!
# The pointwise slowdown: decompositions of a counted computation

The accounting's side of the slowdown, one execution at a time. A **counted computation** (`Counted`) is a circuit,
the actual circuit of an execution, with its `N` operation gates (`Counted.N`) and its counted occurrences in one weight
convention: occurrence `o` has weight `ω o` and is computed by the gates `G o`, and the weighted total is `Ω`
(`Counted.Ω`). A **decomposition** at block size `c` (`Counted.Decomposition`) splits the operation gates into `B`
disjoint qualifying regions of exactly `c` gates each and the remainder, of `U` gates (`Decomposition.U`), and gives
every occurrence exactly one representative gate, among its own. So `N = c·B + U`
(`Nci.Guarantees.DecompositionCount`).

A region's **represented** weighted count (`Decomposition.represented`) is that of the occurrences represented there. It
is what the region computes whole (`Counted.whole`) plus what is **split** there (`Decomposition.split`): represented in
the region, computed partly outside it. The four obligations a decomposition meets are each one named hypothesis:

* `WholeCap q_whole`: every region computes whole at most `q_whole`, the local bounds' per-block cap
  (`Nci.Guarantees.WholeCapLocal`);
* `SplitCap q_split`: every region's split count is at most `q_split`, its own hypothesis, not derived here;
* `RemainderCeiling ν_out`: the remainder's represented count is at most `ν_out` a gate;
* `Utilization f`: the remainder is at most a share `f` of the operation gates.

`Decomposes` says some decomposition meets all four. `Gamma q c ν f`, with `q = q_whole + q_split`, is the weighted
count a gate can carry under them: `Γ = q/c + f·max(ν − q/c, 0)`.
-/

namespace Nci.Capability

open Verity Finset

/-- **A counted computation**: a circuit `C`, whose operation gates are `C.computed`, and its counted occurrences `κ`
in one weight convention: occurrence `o` has weight `ω o` and is computed by the gates `G o`. -/
structure Counted where
  /-- The circuit: an execution's actual circuit. -/
  C : Circuit
  /-- The counted occurrences. -/
  κ : Type
  [fintype : Fintype κ]
  /-- The gates computing each occurrence. -/
  G : κ → Finset (Fin C.N)
  /-- Each occurrence's weight, in the one convention coverage's `Ω_*` uses. -/
  ω : κ → ℝ

attribute [instance] Counted.fintype

namespace Counted

variable (X : Counted)

/-- **`N`**: the number of operation gates. -/
def N : ℕ := X.C.computed.card

/-- **`Ω`**: the weighted total of the counted occurrences. -/
noncomputable def Ω : ℝ := ∑ o, X.ω o

/-- **The weighted count the gates `S` compute whole**: of the occurrences all of whose gates lie in `S`. -/
noncomputable def whole (S : Finset (Fin X.C.N)) : ℝ := ∑ o ∈ univ.filter (fun o => X.G o ⊆ S), X.ω o

/-- **A protocol-to-execution decomposition** at block size `c`: `B` disjoint qualifying regions of the operation gates,
each of exactly `c` gates, and for every occurrence one representative gate among its own gates. The operation gates in
no region are the remainder. -/
structure Decomposition (c : ℕ) where
  /-- The number of qualifying regions. -/
  B : ℕ
  /-- The qualifying regions. -/
  region : Fin B → Finset (Fin X.C.N)
  region_computed : ∀ b, region b ⊆ X.C.computed
  region_card : ∀ b, (region b).card = c
  region_disjoint : ∀ b b', b ≠ b' → Disjoint (region b) (region b')
  /-- Each occurrence's representative gate. -/
  rep : X.κ → Fin X.C.N
  rep_mem : ∀ o, rep o ∈ X.G o
  rep_computed : ∀ o, rep o ∈ X.C.computed

namespace Decomposition

variable {X} {c : ℕ} (D : X.Decomposition c)

/-- **The remainder**: the operation gates in no qualifying region. -/
def remainder : Finset (Fin X.C.N) := X.C.computed \ univ.biUnion D.region

/-- **`U`**: the number of gates in the remainder. -/
def U : ℕ := D.remainder.card

/-- **The weighted count represented in the gates `S`**: of the occurrences whose representative lies in `S`. -/
noncomputable def represented (S : Finset (Fin X.C.N)) : ℝ :=
  ∑ o ∈ univ.filter (fun o => D.rep o ∈ S), X.ω o

/-- **The weighted count split at the gates `S`**: of the occurrences represented in `S` that `S` doesn't compute
whole. -/
noncomputable def split (S : Finset (Fin X.C.N)) : ℝ :=
  ∑ o ∈ univ.filter (fun o => D.rep o ∈ S ∧ ¬ X.G o ⊆ S), X.ω o

/-- **The whole cap** `q_whole`: every qualifying region computes whole a weighted count of at most `q_whole`. -/
def WholeCap (qWhole : ℝ) : Prop := ∀ b, X.whole (D.region b) ≤ qWhole

/-- **The split cap** `q_split`: every qualifying region's split weighted count is at most `q_split`. -/
def SplitCap (qSplit : ℝ) : Prop := ∀ b, D.split (D.region b) ≤ qSplit

/-- **The remainder ceiling** `ν_out`: the remainder's represented weighted count is at most `ν_out` a gate of it. -/
def RemainderCeiling (ν : ℝ) : Prop := D.represented D.remainder ≤ ν * D.U

/-- **Utilization** `f`: the remainder is at most the share `f` of the operation gates, `U ≤ f·N`. -/
def Utilization (f : ℝ) : Prop := (D.U : ℝ) ≤ f * X.N

end Decomposition

/-- **The decomposition hypotheses**: some decomposition of `X` at block size `c` meets the whole cap `q_whole`, the split
cap `q_split`, the remainder ceiling `ν_out` and utilization `f`. -/
def Decomposes (c : ℕ) (qWhole qSplit ν f : ℝ) : Prop :=
  ∃ D : X.Decomposition c, D.WholeCap qWhole ∧ D.SplitCap qSplit ∧ D.RemainderCeiling ν ∧ D.Utilization f

/-- **The counted products as occurrences**: occurrence `o` is the counted product `prod o` of `B`, computed by the
gates `G o`, weighted as `Nci.Guarantees.WeightedProductCap` weighs its product, its slot's price times its use's
weight `ω`. Several occurrences may share a product. -/
@[reducible] noncomputable def ofProducts {d : Declaration} (B : Nci.Covering.MulBlocks d) (W : Items d.D (Fin B.nw))
    (ω : Fin B.nt → ℝ) {κ : Type} [Fintype κ] (prod : κ → Fin B.nw × Fin B.nt) (G : κ → Finset (Fin d.D.N)) :
    Counted where
  C := d.D
  κ := κ
  G := G
  ω := fun o => W.price (prod o).1 * ω (prod o).2

end Counted

/-- **`Γ`**, the weighted count a gate carries under a per-region cap `q` on regions of `c` gates, the remainder
ceiling `ν_out` and utilization `f`: `Γ = q/c + f·max(ν_out − q/c, 0)`. -/
noncomputable def Gamma (q : ℝ) (c : ℕ) (ν f : ℝ) : ℝ := q / c + f * max (ν - q / c) 0

end Nci.Capability
