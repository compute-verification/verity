import Models.Pouw.Assumptions.Dimension.H100

/-!
# `MH100w`: W1's free operations only, and the distinct-writes bound (trusted statements; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §1.1).

`MH100` has no free arithmetic instruction: its only free instructions are `move` (an oblivious whole-word move or
copy) and `operand` (a free value). But its free values are a subspace `V`, so `operand` hands out every rational
combination of free values, and `PerWordMH100` goes through the embedding into the priced model, whose ℚ-linear
combinations are free. W1 has neither, and that relaxation is what makes `PerWordMH100` need linear independence.

`MH100w` is `MH100`'s instruction set `Op100`, with W1's free values: a set `F` of functions on the grid (the
constants, pre-salt data, raw XOF or hash output words), not a subspace. Its only free operations are `move` and
`operand f` with `f ∈ F` (`Op100.WFw`). Every other instruction is priced by its table: `wgmma`, other tensor-core
instructions at `pr.tc` or more per register, FFMA, FADD and FMUL at `pr.ffma`, INT32 at `pr.int32`, FP64 at
`pr.fp64`, and everything else (sub-word extraction, data-dependent loads and selects, memory-side reductions) at
`pr.other`. Every register value is therefore free or written by an instruction that pays for it.

* **Work, not time.** `MH100w`'s cost is additive W1: the sum of the prices of the instructions written. The hardware
  co-issues writers on separate pipes. In Round 8 (`internal/pouw/gpu-constants.md` §13), 1:1 DADD + FADD wrote a
  register every 26.75 units with two register sources, and as often as every 21.45 loop-free with immediates, and
  21.33 at full dispatch, below `H32`'s 32 per register. So every bound here is on work. A bound on time needs a
  maximum over resources (`docs/pouw/w1-time-model.md`).

* **Every salt-independent function of the committed input is free** (the corrected §0 rule, 16:52Z, and red team
  F18b-4). The committed input is X_q (the model's own FP8 quantization of the activations: E4M3 codes and a per-row
  scale), s, and the offline weight codes and scales, all fixed before the salt. Under the §0 game, `F` holds every
  salt-independent function of it, not just X_q: the decoded X_q, the row RMS or partner maximum, the cell widths,
  the registered weight tables, and Y16 if it is committed. The program may start from any of them (`L ⊆ F`) at no
  cost. The honest's cost of computing them is in W_ref, which Lean does not state. The casts of the noised copies
  are in the program and are priced. No statement changes: `F` is a parameter, and every statement here requires
  only that it hold the constants. A salt-tied target equals no salt-independent function on an `S` that spans the
  salts, so the credit is unchanged, and so is `DenseCert`'s `r`.
* **Targets are 32-bit words** (red team F18-1). A target is one register's function, and the credit is per
  register. A trace word narrower than 32 bits counts only through the 32-bit slot holding it, so with `κ` codes per
  slot a code earns `32/κ`: 16 with each two-code word in its own slot, and 8 packed four codes to a slot
  (`DistinctWritesPacked`, `Pouw.Dimension.H100Cite`). A plain whole-word store is a free `move`, so storing a register
  adds no target, and padding a register's codes into separate slots is sub-word extraction, `other`.

* **`DistinctWritesMH100`**: under `H32`, a program whose registers include a finite set `T` of target functions, none
  of them free, costs at least `32·|T|`. Equal targets share one write through a free copy; distinct ones cannot.
  There is no linear independence, no `V₀`, and no A1 or A2.
* **`DistinctWritesMH100On`**, the lifted form: the same for a program that is right only on a set `S` of grid points
  (the salts on which it passes), for targets pairwise distinct on `S`, none equal on `S` to a free value.
* **`DistinctWritesMH100Floor`**: under `Floor c`, `c` per target.
* **`DistinctWritesLegacy`**: under `FloorTc a c` (every writer at `a` or more, except the tensor-core instructions
  outside the `wgmma` shapes, at `c` or more), where an instruction below `a` per register writes only functions in a
  set `K`: targets no function of `K` matches on `S` cost `a` each, and the others `c` each. So each target a cheap
  instruction can write costs a penalty of `(a − c)/a` of its `a`.
* **Tightness** (`DistinctWritesAttained`), the C-2 corollary (`C2DistinctWords`, `C2FirstRunCopy`), and the scheme's
  side (`DistinctLiveAt`, `DistinctLiveMH100`).
-/

namespace Pouw.Dimension.H100

variable {G : Type}

/-- **`MH100w`'s well-formedness** for the free values `F` (draft semantics): a tensor-core instruction outside
`Shape100` costs at least `pr.tc` per accumulator register it writes, and an operand word is a free value. -/
def Op100.WFw (pr : Prices100) (F : Set (G → ℚ)) : Op100 G → Prop
  | .mma outs p => pr.tc * outs.length ≤ p
  | .operand f => f ∈ F
  | _ => True

/-- A tensor-core instruction outside `Shape100` that costs less than `a` per register writes only functions in `K`:
`K` holds what the cheap class (legacy `mma.sync`) can compute. -/
def Op100.CheapIn (a : ℕ) (K : Set (G → ℚ)) : Op100 G → Prop
  | .mma outs p => p < a * outs.length → ∀ f ∈ outs, f ∈ K
  | _ => True

/-- Targets `T` on the grid points `S`: pairwise distinct on `S`, and none equal on `S` to a free value. -/
def DistinctOn (S : Set G) (F : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  (∀ t ∈ T, ∀ t' ∈ T, (∀ g ∈ S, t g = t' g) → t = t') ∧ ∀ t ∈ T, ∀ f ∈ F, ¬ ∀ g ∈ S, t g = f g

/-- The program's registers include every target of `T` on the grid points `S`. -/
def OutputsOn (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (P : List (Op100 G)) (L : List (G → ℚ))
    (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∃ r ∈ regs100 fpSem P L, ∀ g ∈ S, r g = t g

/-- **`DistinctLiveAt ε S F ts`**: on the grid points `S`, the target words `ts` contain a set of at least
`(1 − ε)·|ts|` words that are pairwise distinct on `S` and none equal on `S` to a free value. So at most an `ε`
fraction of the target words are free or duplicates. -/
def DistinctLiveAt (ε : ℚ) (S : Set G) (F : Set (G → ℚ)) (ts : List (G → ℚ)) : Prop :=
  ∃ D : Finset (G → ℚ), (∀ t ∈ D, t ∈ ts) ∧ DistinctOn S F D ∧ (1 - ε) * ts.length ≤ D.card

/-- The nominal table with the other tensor-core instructions at 31 per register, below `H32`. It is a hypothetical part:
X-H32-1's H800 PCIe figure for legacy k8 `mma.sync` (31.2–31.8), rounded down. The H100 SXM measured 34.0 (Round 7;
red team F18-5). -/
def h100Legacy : Prices100 := { h100Nominal with tc := 31 }

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **The distinct-writes bound for `MH100w`** (draft semantics, pending review): under `H32`, for every grid, free
values `F` containing the constants, and FP32 semantics, a well-formed `MH100w` program on free inputs whose
registers include every target of a finite set `T`, none of them free, costs at least `32·|T|`.

The targets are 32-bit register functions, so the credit is 32 per register. For code words packed `κ` to a register it
is `32/κ` per code (red team F18-1; `DistinctWritesPacked`). -/
def DistinctWritesMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → (∀ t ∈ T, t ∉ F) → (∀ t ∈ T, t ∈ regs100 fpSem P L) →
      32 * T.card ≤ cost100 pr P

/-- **The distinct-writes bound on a success set** (draft semantics, pending review): under `H32`, for every set `S`
of grid points, a well-formed `MH100w` program on free inputs that matches on `S` every target of a finite set `T`,
pairwise distinct on `S` and none equal on `S` to a free value, costs at least `32·|T|`. -/
def DistinctWritesMH100On : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 pr P

/-- **The distinct-writes bound at any floor** (draft semantics, pending review): under `Floor c`, the same with `c`
per target. -/
def DistinctWritesMH100Floor : Prop :=
  ∀ (c : ℕ) (pr : Prices100), Floor c pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      c * T.card ≤ cost100 pr P

/-- **The distinct-writes bound with a cheap legacy class** (draft semantics, pending review): under `FloorTc a c`
with `c ≤ a`, if every tensor-core instruction below `a` per register writes only functions in `K`, then for targets
`T₁` and `T₂`, pairwise distinct on `S` within and across, none equal on `S` to a free value, and no target of `T₁`
equal on `S` to a function of `K`, a program matching them all on `S` costs at least `a·|T₁| + c·|T₂|`. -/
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

/-- **The distinct-writes bound is attained** at the nominal prices: for every finite set `T` of functions and every
list `Fr` of free values, one FP8 `wgmma` register per target of `T` and one operand word per free value is a
well-formed `MH100w` program whose registers include them all, at exactly `32·|T|`. This is a nominal-model
statement. On hardware it holds only for `wgmma` at `N ≥ 64` (red team F18-4; `Shape100`). -/
def DistinctWritesAttained : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ)) (Fr L : List (G → ℚ)),
    (∀ f ∈ Fr, f ∈ F) →
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Nominal F) ∧ (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧
      (∀ f ∈ Fr, f ∈ regs100 fpSem P L) ∧ cost100 h100Nominal P = 32 * T.card

/-- **The C-2 corollary** (draft semantics, pending review): for the split-K words (fresh-step words `I τ` and
running words `R (τ + 1)`, `τ < T`), the honest schedule `splitK` is a well-formed `MH100w` program. Under `H32`,
every well-formed `MH100w` program on free inputs whose registers include all of them costs at least 32 per word of
any set `D` of them that are not free. Distinct words count separately, whatever their rounding. -/
def C2DistinctWords : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) (T : ℕ) (L : List (G → ℚ)),
      (∀ o ∈ splitK κ I L.length T, o.WFw h100Nominal F) ∧
      ∀ (pr : Prices100) (P : List (Op100 G)), H32 pr → (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) →
        (∀ τ < T, I τ ∈ regs100 fpSem P L ∧ runWords fpSem κ I (τ + 1) ∈ regs100 fpSem P L) →
        ∀ D : Finset (G → ℚ), (∀ w ∈ D, (∃ τ < T, w = I τ ∨ w = runWords fpSem κ I (τ + 1)) ∧ w ∉ F) →
          32 * D.card ≤ cost100 pr P

/-- **The first running word is a copy**: if `κ` adds `+0` exactly (`fpSem κ 0 x 0 = x`, as FADD does), then
`R 1 = I 0`. -/
def C2FirstRunCopy : Prop :=
  ∀ (G : Type) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ),
    (∀ x : ℚ, fpSem κ 0 x 0 = x) → runWords fpSem κ I 1 = I 0

/-- **`DistinctLive` gives the per-word cost** (draft semantics, pending review): under `H32`, if `DistinctLiveAt ε S F
ts`, every well-formed `MH100w` program on free inputs that matches every target word of `ts` on `S` costs at least
`32·(1 − ε)·|ts|`.

This is the applicable pin for the truncated k32 `HOPPER_E4M3_K32` words of C-2, A-1 and D-2, which are not linear, so
`PerWordMH100Words` does not cover them (red team F18-3). `DistinctLiveAt ε` is what each scheme must certify. -/
def DistinctLiveMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (ε : ℚ) (S : Set G) (ts : List (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ))
      (P : List (Op100 G)),
      DistinctLiveAt ε S F ts → (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) →
      (∀ t ∈ ts, ∃ r ∈ regs100 fpSem P L, ∀ g ∈ S, r g = t g) →
      32 * (1 - ε) * ts.length ≤ (cost100 pr P : ℚ)

end Pouw.Dimension.H100
