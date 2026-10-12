import Models.Pouw.Assumptions.Dimension.H100Dense

/-!
# Computed fragments in the k8 lemma (trusted statements; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §1.1; red team X-H32-2).

`AccIn` with `LaneFns m` covers a cheap tensor-core instruction whose A and B fragments are lane words. Here the
fragments may be computed registers. A cheap register is then, up to the rounding `β`, its accumulator plus a function
of at most `K` multiplicand registers (`AccFragIn`). A k8 output element reads 8 A and 8 B values, 4 registers of
each, so `K = 8`.

A computed fragment costs at least `c` to write, but that does not save the accounting. One fragment shared across
column tiles is paid once and read by every tile, so its price per word goes to 0 (`FragShareGain`). The count in
`DenseWritesLegacy` never charges fragments. It charges each dense word and its accumulator. What it needs is that the
increment still depends on few lanes:

* **`AccInOfFrag`**: if every multiplicand register depends on at most `q` lanes, the increments are in
  `LaneFns (K·q) β`. So `AccFragIn` implies `AccIn` at `m = K·q`, and `DenseCert`, `ApartCert` and `LegacyGamma` apply
  at `m = K·q`, however widely each fragment is shared.
* **`DenseWritesFrag`**: the k8 bound with computed fragments, `a·|T₁| + c·|T₂| ≤ cost`, for targets dense and apart
  for `LaneFns (K·q) β`.
* **`FragShareGain`**: the lane bound cannot be dropped. With one multiplicand that depends on every lane, `n` dense,
  pairwise apart words cost `32 + 31·n` at `h100Legacy`'s prices, below `32·n` once `n > 32`.

The lane bound on the multiplicands is the named hypothesis `FragLanes S p q W` (`Pouw.Dimension.H100Assumptions`, red
team F18-2). `DenseWritesFragLanes` (`Pouw.Dimension.H100Cite`) is `DenseWritesFrag` with it. The k8 lemma is cited
only against adversaries whose cheap tensor-core multiplicands meet it (red team F18b-5). It is not needed at Round 8's
loop-free prices (`DistinctWritesLoopFree`, `Pouw.Dimension.H100Floor`).
-/

namespace Pouw.Dimension.H100

variable {G : Type}

/-- `d` is determined on `S` by the functions `ws`: two grid points of `S` where every `w ∈ ws` agrees give `d` the
same value. -/
def DetBy (S : Set G) (ws : List (G → ℚ)) (d : G → ℚ) : Prop :=
  ∀ g ∈ S, ∀ g' ∈ S, (∀ w ∈ ws, w g = w g') → d g = d g'

/-- The accumulator form with computed multiplicands, for one instruction: a tensor-core instruction outside the
`wgmma` shapes that costs less than `a` per register writes only registers within `β` on `S` of an accumulator `C` (an
earlier register or a free value) plus a function of at most `K` multiplicands, each an earlier register or a free
value, and each in `W`. -/
def Op100.FragOK (a K : ℕ) (β : ℚ) (S : Set G) (F W : Set (G → ℚ)) (Rs : List (G → ℚ)) : Op100 G → Prop
  | .mma outs p => p < a * outs.length →
      ∀ f ∈ outs, ∃ C : G → ℚ, (C ∈ Rs ∨ C ∈ F) ∧ ∃ ws : List (G → ℚ), ws.length ≤ K ∧
        (∀ w ∈ ws, (w ∈ Rs ∨ w ∈ F) ∧ w ∈ W) ∧
        ∃ d₀ : G → ℚ, DetBy S ws d₀ ∧ ∀ g ∈ S, |f g - (C g + d₀ g)| ≤ β
  | _ => True

/-- **`AccFragIn a K β fpSem S F W P L`**: every instruction of `P`, run from the registers `L`, has the accumulator
form with at most `K` multiplicands from `W`. -/
def AccFragIn (a K : ℕ) (β : ℚ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (F W : Set (G → ℚ)) :
    List (Op100 G) → List (G → ℚ) → Prop
  | [], _ => True
  | o :: P, Rs => o.FragOK a K β S F W Rs ∧ AccFragIn a K β fpSem S F W P (Rs ++ o.outs fpSem Rs)

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **Computed fragments reduce to lane increments** (draft semantics, pending review): if every multiplicand in `W`
depends on at most `q` lanes on `S`, a program in the accumulator form with at most `K` multiplicands per register is
in the accumulator form with increments in `LaneFns (K·q) β S p`. -/
def AccInOfFrag : Prop :=
  ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F W : Set (G → ℚ)) (a K q : ℕ) (β : ℚ)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Op100 G)) (L : List (G → ℚ)),
    (∀ w ∈ W, ∃ J : Finset Λ, J.card ≤ q ∧ DependsOn S p J w) →
    AccFragIn a K β fpSem S F W P L → AccIn a fpSem S F (LaneFns (K * q) β S p) P L

/-- **Dense words cost `a` with computed fragments** (draft semantics, pending review): `DenseWritesLegacy` with
`AccFragIn` in place of `AccIn`, for multiplicands that each depend on at most `q` lanes, and targets dense and apart
for `LaneFns (K·q) β S p`. -/
def DenseWritesFrag : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c →
    ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F W : Set (G → ℚ)) (K q : ℕ) (β : ℚ),
    (∀ x : ℚ, (fun _ : G => x) ∈ F) → 0 ≤ β → (∀ w ∈ W, ∃ J : Finset Λ, J.card ≤ q ∧ DependsOn S p J w) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccFragIn a K β fpSem S F W P L →
      DenseOn S F (LaneFns (K * q) β S p) T₁ → SeparatedOn S (LaneFns (K * q) β S p) T₁ → DistinctOn S F T₂ →
      (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S (LaneFns (K * q) β S p) t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      a * T₁.card + c * T₂.card ≤ cost100 pr P

/-- **A shared fragment amortizes its price** (draft semantics, pending review): without a lane bound on the
multiplicands, a fragment's price does not save the accounting. For all `m` and `n`, on the grid
`Fin (2m + 1) → ℚ` with the coordinates as lanes and the constants free, `n` targets dense for `LaneFns m 0` and
pairwise apart are written at `h100Legacy`'s prices for `32 + 31·n`, by a program in the accumulator form with one
multiplicand per register (`AccFragIn 32 1 0`, multiplicands unrestricted): one `other` register, the sum of the
coordinates, read by `n` cheap instructions. -/
def FragShareGain : Prop :=
  ∀ (m n : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ),
    ∃ (P : List (Op100 (Fin (2 * m + 1) → ℚ))) (T : Finset ((Fin (2 * m + 1) → ℚ) → ℚ)),
      T.card = n ∧ (∀ o ∈ P, o.WFw h100Legacy {f | ∃ x : ℚ, f = fun _ => x}) ∧
      AccFragIn 32 1 0 fpSem Set.univ {f | ∃ x : ℚ, f = fun _ => x} Set.univ P [] ∧
      DenseOn Set.univ {f | ∃ x : ℚ, f = fun _ => x}
        (LaneFns m 0 Set.univ fun j (g : Fin (2 * m + 1) → ℚ) => g j) T ∧
      SeparatedOn Set.univ (LaneFns m 0 Set.univ fun j (g : Fin (2 * m + 1) → ℚ) => g j) T ∧
      OutputsOn fpSem Set.univ P [] T ∧ cost100 h100Legacy P = 32 + 31 * n

end Pouw.Dimension.H100
