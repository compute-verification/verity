import Models.Pouw.Assumptions.Dimension.H100W

/-!
# Dense words and the legacy k8 class in `MH100w` (trusted statements; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §1.1; red team X-H32-2).

`DistinctWritesLegacy` charges `a` for targets outside a set `K` that holds every register a cheap tensor-core
instruction (legacy k8 `mma.sync`, below `a` per register) can write. Arity alone does not bound `K`: `MH100w`'s
priced instructions write any function, so one earlier register can be injective on the grid, and a cheap instruction
reading it could then write any word. What bounds `K` is that the accumulator enters additively. A k8 register is
`C + (a function of its 8 lane products)`, up to the rounding of the add, where `C` is its accumulator input.

* **`AccIn a S F D`** states that form for a program: every register written by a tensor-core instruction below `a`
  per register equals, on `S`, an earlier register or a free value plus an increment in `D`. For a k8 instruction whose
  multiplicands are lane words, `D = LaneFns 8 β S p`: within `β` (the rounding of the add) of a function of at most
  8 lane products `p j`.
* **Dense and apart targets** (`DenseOn`, `ApartOn`, `SeparatedOn`): a dense target is no free value plus an
  increment. Two targets are apart if they are not both an increment away from one common value. Both are conditions
  on the target family alone.
* **`DenseWritesLegacy`**: under `FloorTc a c` with `c ≤ a ≤ 2c`, dense targets pairwise apart cost `a` each, and the
  other targets `c` each. A dense target written by a cheap instruction needs an accumulator that is an earlier
  non-free register, and apart targets cannot share one, so the chain costs at least `2c`.
* **Certificates** (`DenseCert`, `ApartCert`): with `Contributes` (lane `j` changes `t` by more than a margin between
  two grid points that differ only in lane `j`), more than `m + r` contributing lanes make a word dense, when every
  free value depends on at most `r` lanes, and more than `2m` contributing lanes of `t − t'` make two words apart.
  A finite checker verifies them from witness pairs.
* **`LegacyGamma`**: with honest price `a` per checked word, `DistinctLive(ε)` and a fraction `μ` of coverable words,
  `γ ≤ ε + μ·(a − c)/a`.

**Scope (red team F18-2).** `AccIn` with `LaneFns` covers cheap instructions whose multiplicands are lane words. A
legacy `mma` reading computed registers is outside it, and `LegacyGamma` bounds `γ` only over `AccIn` programs.
`Pouw.Dimension.H100Frag` extends the lemma to computed multiplicands. It is cited only against adversaries whose cheap
tensor-core multiplicands meet the named hypothesis `FragLanes` (`DenseWritesFragLanes`, `Pouw.Dimension.H100Cite`;
red team F18b-5). At Round 8's loop-free prices, where the H100 SXM's legacy k8 costs 32.00 and `H32` holds, none of
this is needed (`DistinctWritesLoopFree`, `Pouw.Dimension.H100Floor`).

**The witness (red team F18-6).** `DenseLaneWitness` is at `β = 0` on `ℚ^(m+1)`. It shows that the definitions are
satisfiable, not that a real atom word is dense once `β` carries the atom's truncation. A witness on the pinned atom
`HOPPER_E4M3_K32` is open, since the package does not model the atom's semantics.
-/

namespace Pouw.Dimension.H100

variable {G : Type}

/-- `d` depends only on the lanes `J` on `S`: two grid points of `S` whose lane products in `J` agree give `d` the same
value. -/
def DependsOn {Λ : Type} (S : Set G) (p : Λ → G → ℚ) (J : Finset Λ) (d : G → ℚ) : Prop :=
  ∀ g ∈ S, ∀ g' ∈ S, (∀ j ∈ J, p j g = p j g') → d g = d g'

/-- The increments of a cheap tensor-core instruction with `m` lanes: functions within `β` on `S` of a function of at
most `m` lane products. -/
def LaneFns {Λ : Type} (m : ℕ) (β : ℚ) (S : Set G) (p : Λ → G → ℚ) : Set (G → ℚ) :=
  {d | ∃ J : Finset Λ, J.card ≤ m ∧ ∃ d₀ : G → ℚ, DependsOn S p J d₀ ∧ ∀ g ∈ S, |d g - d₀ g| ≤ β}

/-- Lane `j` contributes to `t` on `S` by more than `θ`: two grid points of `S`, whose lane products agree except at
`j`, where `t` differs by more than `θ`. -/
def Contributes {Λ : Type} (S : Set G) (p : Λ → G → ℚ) (θ : ℚ) (t : G → ℚ) (j : Λ) : Prop :=
  ∃ g ∈ S, ∃ g' ∈ S, (∀ i, i ≠ j → p i g = p i g') ∧ θ < |t g - t g'|

/-- The accumulator form of one instruction: a tensor-core instruction outside the `wgmma` shapes that costs less than
`a` per register writes only registers equal on `S` to an earlier register or a free value plus an increment in `D`. -/
def Op100.AccOK (a : ℕ) (S : Set G) (F D : Set (G → ℚ)) (Rs : List (G → ℚ)) : Op100 G → Prop
  | .mma outs p => p < a * outs.length →
      ∀ f ∈ outs, ∃ C : G → ℚ, (C ∈ Rs ∨ C ∈ F) ∧ ∃ d ∈ D, ∀ g ∈ S, f g = C g + d g
  | _ => True

/-- **`AccIn a fpSem S F D P L`**: every instruction of `P`, run from the registers `L`, has the accumulator form. -/
def AccIn (a : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (F D : Set (G → ℚ)) :
    List (Op100 G) → List (G → ℚ) → Prop
  | [], _ => True
  | o :: P, Rs => o.AccOK a S F D Rs ∧ AccIn a fpSem S F D P (Rs ++ o.outs fpSem Rs)

/-- Dense targets: none equals on `S` a free value plus an increment in `D`. -/
def DenseOn (S : Set G) (F D : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∀ f ∈ F, ∀ d ∈ D, ¬ ∀ g ∈ S, t g = f g + d g

/-- Two words are apart on `S`: they are not both an increment in `D` away from one common value. -/
def ApartOn (S : Set G) (D : Set (G → ℚ)) (t t' : G → ℚ) : Prop :=
  ∀ d ∈ D, ∀ d' ∈ D, ¬ ∀ g ∈ S, t g - d g = t' g - d' g

/-- Targets pairwise apart on `S`. -/
def SeparatedOn (S : Set G) (D : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∀ t' ∈ T, t ≠ t' → ApartOn S D t t'

/-- The sum of the coordinates on `Fin (m + 1) → ℚ`, a word to which every lane contributes. -/
def sumCoords (m : ℕ) : (Fin (m + 1) → ℚ) → ℚ := fun g => ∑ j, g j

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **Dense words cost `a` against the legacy class** (draft semantics, pending review): under `FloorTc a c` with
`c ≤ a ≤ 2c`, for a well-formed `MH100w` program on free inputs in which every cheap tensor-core register is an
earlier register or a free value plus an increment in `D` (`AccIn`), dense targets `T₁` pairwise apart, and targets
`T₂` distinct and not free on `S`, each of `T₂` apart from each of `T₁`, a program matching them all on `S` costs at
least `a·|T₁| + c·|T₂|`. -/
def DenseWritesLegacy : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c →
    ∀ (G : Type) (F D : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) → (0 : G → ℚ) ∈ D →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G))
      (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccIn a fpSem S F D P L →
      DenseOn S F D T₁ → SeparatedOn S D T₁ → DistinctOn S F T₂ → (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S D t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      a * T₁.card + c * T₂.card ≤ cost100 pr P

/-- **The dense certificate** (draft semantics, pending review): if every free value depends on at most `r` lanes on
`S`, a word with more than `m + r` lanes each contributing more than `2β` is no free value plus an increment of
`LaneFns m β S p`. -/
def DenseCert : Prop :=
  ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F : Set (G → ℚ)) (m r : ℕ) (β : ℚ) (T : Finset (G → ℚ)),
    (∀ f ∈ F, ∃ J : Finset Λ, J.card ≤ r ∧ DependsOn S p J f) →
    (∀ t ∈ T, ∃ W : Finset Λ, m + r < W.card ∧ ∀ j ∈ W, Contributes S p (2 * β) t j) →
    DenseOn S F (LaneFns m β S p) T

/-- **The apart certificate** (draft semantics, pending review): if more than `2m` lanes each contribute more than `4β`
to `t − t'` on `S`, then `t` and `t'` are apart for `LaneFns m β S p`. -/
def ApartCert : Prop :=
  ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (m : ℕ) (β : ℚ) (t t' : G → ℚ),
    (∃ W : Finset Λ, 2 * m < W.card ∧ ∀ j ∈ W, Contributes S p (4 * β) (t - t') j) →
    ApartOn S (LaneFns m β S p) t t'

/-- **The γ bound with a legacy class** (draft semantics, pending review): with honest price `a` per checked word,
under `FloorTc a c` with `c ≤ a ≤ 2c`, if the live distinct checked words split into dense words `T₁`, pairwise
apart, and coverable words `T₂` (`DistinctLive(ε)`: `|T₁| + |T₂| ≥ (1 − ε)·n`, with `|T₂| ≤ μ·n`), every program
matching them on `S` has `γ = 1 − cost/(a·n) ≤ ε + μ·(a − c)/a`. -/
def LegacyGamma : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c → 0 < a →
    ∀ (G : Type) (F D : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) → (0 : G → ℚ) ∈ D →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (ts : List (G → ℚ))
      (T₁ T₂ : Finset (G → ℚ)) (ε μ : ℚ),
      0 < ts.length → (∀ t ∈ T₁, t ∈ ts) → (∀ t ∈ T₂, t ∈ ts) →
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccIn a fpSem S F D P L →
      DenseOn S F D T₁ → SeparatedOn S D T₁ → DistinctOn S F T₂ → (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S D t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      (1 - ε) * ts.length ≤ (T₁.card + T₂.card : ℚ) → (T₂.card : ℚ) ≤ μ * ts.length →
      1 - (cost100 pr P : ℚ) / (a * ts.length) ≤ ε + μ * (a - c) / a

/-- **Dense words exist for every lane count** (draft semantics, pending review): on the grid `Fin (m + 1) → ℚ` with
the coordinates as lane products and the constants as free values, the sum of the coordinates is dense for
`LaneFns m 0`, and so is a singleton family, pairwise apart. It is at `β = 0` on a synthetic grid. A witness on the
pinned atom, with `β` carrying its truncation, is open (red team F18-6). -/
def DenseLaneWitness : Prop :=
  ∀ m : ℕ,
    DenseOn Set.univ {f | ∃ q : ℚ, f = fun _ => q} (LaneFns m 0 Set.univ fun j (g : Fin (m + 1) → ℚ) => g j)
        {sumCoords m} ∧
      SeparatedOn Set.univ (LaneFns m 0 Set.univ fun j (g : Fin (m + 1) → ℚ) => g j) {sumCoords m}

end Pouw.Dimension.H100
