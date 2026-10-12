import Models.Pouw.Assumptions.Dimension.H100Frag
import Models.Pouw.Assumptions.Dimension.H100Measured

/-!
# How the H100 pins are cited (trusted statements; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §1.1; red team Phase 18 in
`internal/pouw-fp8/genuine-fp8-interface-red-team.md`).

The red team accepted the 25 FP8 and H100 pins as statements. Its findings are about what a citation may claim, and
three of them are stated here.

* **Credit per register, codes per register** (`DistinctWritesPacked`, F18-1).
  * The targets of `DistinctWritesMH100` are 32-bit words, and the credit is `c` per register. A trace word narrower
    than 32 bits is a target only through the 32-bit slot that holds it.
  * With at most `κ` codes per slot, `c·|codes| ≤ κ·cost`, so a code earns `c/κ`. At `c = 32` that is 16 per code
    when each two-code (x2) word sits in its own 32-bit slot (`κ = 2`), and 8 when two x2 words share a slot (`κ = 4`).
  * Padding helps only because each padded slot is a distinct non-free function. A plain whole-word store is a free
    `move` that copies its register, and splitting a four-code register into two padded slots is sub-word extraction,
    `other`.
* **The k8 lemma needs `FragLanes`** (`DenseWritesFragLanes`, F18-2). The named hypothesis `FragLanes S p q W`
  (`Pouw.Dimension.H100Assumptions`) says that every multiplicand in `W` depends on at most `q` lanes. It restricts the
  adversary and can fail (`FragShareGain`). So the k8 lemma is cited only against adversaries whose cheap tensor-core
  multiplicands meet `FragLanes` (red team F18b-5). `FragLanesSat` is its witness, and shows that it has content: the
  sum of `m + 1` lanes fails it at `q = m`.
* **`FragLanes` is not needed at Round 8's loop-free prices** (`DistinctWritesLoopFree`, `DistinctWritesFloor` in
  `Pouw.Dimension.H100Floor`; red team F18b-1). There legacy k8 costs 32.00 and `H32` holds with no slack. So every
  well-formed `MH100w` program matching targets pairwise distinct on `S`, none free, costs at least 32 per target, with
  no hypothesis on any instruction's accumulator or multiplicands. `DistinctWritesMeasured` says the same at Round 7's
  `h100Measured`. Since Round 8 that table does not bound the hardware from below, so it is not the hardware citation.

**Open (F18-6): a dense-word witness on the pinned atom `HOPPER_E4M3_K32`.** `DenseLaneWitness` is at `β = 0` on
`ℚ^(m+1)`. The package does not model the atom's semantics (E4M3 decoding, alignment to the largest product and the
14-bit truncation, which set `β`), so a witness on it is not stated. A finite checker can supply it per slice, as
`DenseCert`'s certificate: more than `m + r` lanes, each contributing more than `2β`.
-/

namespace Pouw.Dimension.H100

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **Credit per register, codes per register** (draft semantics, pending review; F18-1): under `Floor c`, if a
well-formed `MH100w` program on free inputs matches on `S` a finite set `T` of 32-bit slots, pairwise distinct on `S`
and none equal on `S` to a free value, and every code of `codes` sits in a slot of `T`, at most `κ` codes per slot,
then `c·|codes| ≤ κ·cost`. Each code earns `c/κ`. -/
def DistinctWritesPacked : Prop :=
  ∀ (c : ℕ) (pr : Prices100), Floor c pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ))
      (ι : Type) (codes : Finset ι) (slot : ι → G → ℚ) (κ : ℕ),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      (∀ i ∈ codes, slot i ∈ T) → (∀ t ∈ T, ∀ s : Finset ι, s ⊆ codes → (∀ i ∈ s, slot i = t) → s.card ≤ κ) →
      c * codes.card ≤ κ * cost100 pr P

/-- **The k8 lemma under the named hypothesis `FragLanes`** (draft semantics, pending review; F18-2): `DenseWritesFrag`
with its lane bound on the multiplicands stated as `FragLanes S p q W`. It is the form in which the k8 lemma is cited:
only against adversaries whose cheap tensor-core instructions have the accumulator form (`AccFragIn`) with
multiplicands in `W` that meet `FragLanes` (red team F18b-5). -/
def DenseWritesFragLanes : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c →
    ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F W : Set (G → ℚ)) (K q : ℕ) (β : ℚ),
    (∀ x : ℚ, (fun _ : G => x) ∈ F) → 0 ≤ β → FragLanes S p q W →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccFragIn a K β fpSem S F W P L →
      DenseOn S F (LaneFns (K * q) β S p) T₁ → SeparatedOn S (LaneFns (K * q) β S p) T₁ → DistinctOn S F T₂ →
      (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S (LaneFns (K * q) β S p) t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      a * T₁.card + c * T₂.card ≤ cost100 pr P

/-- **`FragLanes` is satisfiable, and has content** (F18-2): the lane words meet `FragLanes 1`, and on the grid
`Fin (m + 1) → ℚ` with the coordinates as lanes, the sum of the coordinates fails `FragLanes m`. -/
def FragLanesSat : Prop :=
  (∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ), FragLanes S p 1 (Set.range p)) ∧
    ∀ m : ℕ, ¬ FragLanes Set.univ (fun j (g : Fin (m + 1) → ℚ) => g j) m {sumCoords m}

/-- **At Round 7's prices the distinct-writes bound needs no hypothesis on the tensor-core instructions** (draft
semantics, pending review; F18-2): at `h100Measured`, every well-formed `MH100w` program on free inputs that matches on
`S` a finite set `T` of targets, pairwise distinct on `S` and none equal on `S` to a free value, costs at least
`32·|T|`. The k8 lemma and `FragLanes` are not needed there.

It is not a hardware citation (red team F18b-1). Since Round 8, `h100Measured` does not bound the hardware from below:
two-source IADD measured 34.68 against its `int32` of 64. Cite `DistinctWritesLoopFree` for programs of the measured
forms, or `DistinctWritesFloor` for every scalar form and every `wgmma` shape, with legacy tensor-core forms as
measured (`Pouw.Dimension.H100Floor`; red team F18c-1). -/
def DistinctWritesMeasured : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100Measured F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100Measured P

end Pouw.Dimension.H100
