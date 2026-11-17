module

public import Definitions.Pouw.Assumptions.Dimension.H100

/-! `MH100w`, straight-line programs with oblivious addresses over `MH100`'s instructions with W1's free values, and the distinct-writes bound on their additive-W1 work, not time. -/

@[expose] public section

namespace Pouw.Dimension.H100

variable {G : Type}

/-- **`MH100w`'s well-formedness** for the free values `F`: a tensor-core instruction outside
`Shape100` costs at least `pr.tc` per accumulator register it writes, and an operand word is a free value. -/
def Op100.WFw (pr : Prices100) (F : Set (G → ℚ)) : Op100 G → Prop
  | .mma outs p => pr.tc * outs.length ≤ p
  | .operand f => f ∈ F
  | _ => True

/-- `K` holds what the cheap class (legacy `mma.sync`) can compute. -/
def Op100.CheapIn (a : ℕ) (K : Set (G → ℚ)) : Op100 G → Prop
  | .mma outs p => p < a * outs.length → ∀ f ∈ outs, f ∈ K
  | _ => True

def DistinctOn (S : Set G) (F : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  (∀ t ∈ T, ∀ t' ∈ T, (∀ g ∈ S, t g = t' g) → t = t') ∧ ∀ t ∈ T, ∀ f ∈ F, ¬ ∀ g ∈ S, t g = f g

def OutputsOn (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (P : List (Op100 G)) (L : List (G → ℚ))
    (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∃ r ∈ regs100 fpSem P L, ∀ g ∈ S, r g = t g

/-- At most an `ε` fraction of the target words `ts` are free or duplicates on `S`. -/
def DistinctLiveAt (ε : ℚ) (S : Set G) (F : Set (G → ℚ)) (ts : List (G → ℚ)) : Prop :=
  ∃ D : Finset (G → ℚ), (∀ t ∈ D, t ∈ ts) ∧ DistinctOn S F D ∧ (1 - ε) * ts.length ≤ D.card

/-- A hypothetical part, below `H32`, with the other tensor-core instructions at 31 per register. -/
def h100Legacy : Prices100 := { h100Nominal with tc := 31 }

/-- **The distinct-writes bound for `MH100w`**. The targets are 32-bit register functions: for code words packed `κ` to a
register the credit is `32/κ` per code (`DistinctWritesPacked`). -/
def DistinctWritesMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → (∀ t ∈ T, t ∉ F) → (∀ t ∈ T, t ∈ regs100 fpSem P L) →
      32 * T.card ≤ cost100 pr P

/-- `S` is the set of grid points on which the program passes. -/
def DistinctWritesMH100On : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 pr P

def DistinctWritesMH100Floor : Prop :=
  ∀ (c : ℕ) (pr : Prices100), Floor c pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      c * T.card ≤ cost100 pr P

def DistinctWritesLegacy : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a →
    ∀ (G : Type) (F K : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G))
      (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → (∀ o ∈ P, o.CheapIn a K) →
      DistinctOn S F T₁ → DistinctOn S F T₂ → (∀ t ∈ T₁, ∀ t' ∈ T₂, ¬ ∀ g ∈ S, t g = t' g) →
      (∀ t ∈ T₁, ∀ f ∈ K, ¬ ∀ g ∈ S, t g = f g) →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      a * T₁.card + c * T₂.card ≤ cost100 pr P

/-- **`FloorTc` below `H32` is satisfiable**: `h100Legacy` meets `FloorTc 32 31` and fails `H32`. -/
def FloorTcSat : Prop := FloorTc 32 31 h100Legacy ∧ ¬ H32 h100Legacy

/-- A nominal-model statement: on hardware it holds only for `wgmma` at `N ≥ 64` (`Shape100`). -/
def DistinctWritesAttained : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ)) (Fr L : List (G → ℚ)),
    (∀ f ∈ Fr, f ∈ F) →
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Nominal F) ∧ (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧
      (∀ f ∈ Fr, f ∈ regs100 fpSem P L) ∧ cost100 h100Nominal P = 32 * T.card

/-- **The C-2 corollary** for the split-K words: distinct words count separately, whatever their rounding. -/
def C2DistinctWords : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) (T : ℕ) (L : List (G → ℚ)),
      (∀ o ∈ splitK κ I L.length T, o.WFw h100Nominal F) ∧
      ∀ (pr : Prices100) (P : List (Op100 G)), H32 pr → (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) →
        (∀ τ < T, I τ ∈ regs100 fpSem P L ∧ runWords fpSem κ I (τ + 1) ∈ regs100 fpSem P L) →
        ∀ D : Finset (G → ℚ), (∀ w ∈ D, (∃ τ < T, w = I τ ∨ w = runWords fpSem κ I (τ + 1)) ∧ w ∉ F) →
          32 * D.card ≤ cost100 pr P

/-- FADD meets the hypothesis on most inputs but not all (`+0 + −0`, NaNs, FTZ flushes, rounding), so for FADD
`R 1 = I 0` holds only where `I 0` takes none of those values. -/
def C2FirstRunCopy : Prop :=
  ∀ (G : Type) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ),
    (∀ x : ℚ, fpSem κ 0 x 0 = x) → runWords fpSem κ I 1 = I 0

/-- The applicable pin for the truncated k32 `HOPPER_E4M3_K32` words, which are not linear, so `PerWordMH100Words` does
not cover them. `DistinctLiveAt ε` is what each scheme must certify. -/
def DistinctLiveMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (ε : ℚ) (S : Set G) (ts : List (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ))
      (P : List (Op100 G)),
      DistinctLiveAt ε S F ts → (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) →
      (∀ t ∈ ts, ∃ r ∈ regs100 fpSem P L, ∀ g ∈ S, r g = t g) →
      32 * (1 - ε) * ts.length ≤ (cost100 pr P : ℚ)

end Pouw.Dimension.H100
