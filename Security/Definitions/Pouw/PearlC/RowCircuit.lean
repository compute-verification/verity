import Definitions.Pouw.PearlC.RowPrims
import Definitions.Core.Fp32
import Definitions.Core.Prims
import Definitions.Core.Scalar

/-!
# The row units as core circuits (definitions only)

`pc8`'s row templates `Pc8RowA_v1{K, DEV}` and `Pc8RowB_v1{K, DEV}` (`verity.protocols.accounting.work.pouw.circuit.pc8`, their
`_row_a_body`, `_row_b_body`, `_stats`, `_form` and `_chain`) as gate tables (`Rows`, each a `TileCircuit.Spec`),
generic in the depth `K`; `Row4Circuit` writes `pc4`'s the same way. Unlike the tile's, a row gate names its
primitive (`Prim`, the template's primitive id in its docstring) and lists its operands in order (`Arg`): an earlier
gate, read plain or through core's wiring (`Wire`, which the IR's `CallGraph` folds into its readers), or a constant.
Its value is the primitive's word function on the operands' values (`Prim.eval`):

* core's primitives are the fields of `RowOps`, a parameter named as core's Lean (`Verity.Protocol.Fp32`, `.Scalar`,
  `.Prims`), which `coreOps` fills from it;
* PoUW's own (`PouwF32SqrtRn_v1`, `PouwETilde_v1`, `PouwBf16Neg_v1`, `PouwSq*`) are `RowPrims`' word functions;
* `F32Add_v2` is `TileCircuit.f32Add`, the E4M3 k32 atom is the device's `p.step` (`TileCircuit.atomF`) and the NVF4
  atom is `nvf4Word` (`TileCircuit.atom4F`), as in the tile.

Every value is reduced to the primitive's width, as the circuit's `BitVec` does.

A row unit's table (`Rows`) is its input gates, the template's parameter leaves in order, then runs of computed gates
(`Run`), each a section of the template's body. The sections both families share are the statistics (`statsRuns`): the
running max of `|x|`, the FMA partials of `x²` over every 8th position, their butterfly and `ρ`. Which gate reads which,
in which order and through which wiring, the constants, the widths and the outputs are checked against the templates'
own gates and their `CallGraph`, with descriptor digests pinned, by `tests/test_circuit_row_lean.py`; each primitive's
semantics is the trusted part.
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

/-! ## Core's primitives, as a parameter -/

/-- The word functions of core's primitives that the row units read, each on words as naturals, named as core's
(`Verity.Protocol.Fp32`, `.Scalar`, `.Prims`). `coreOps` is core's own. -/
structure RowOps where
  /-- `F32Mul_v2` -/
  f32MulV2 : ℕ → ℕ → ℕ
  /-- `F32Div_v2` -/
  f32DivV2 : ℕ → ℕ → ℕ
  /-- `F32Fma_v2`: `a·b + c` -/
  f32FmaV2 : ℕ → ℕ → ℕ → ℕ
  /-- `F32Fmaxf_v1` -/
  f32Fmaxf : ℕ → ℕ → ℕ
  /-- `F32Neg_v1` -/
  f32Neg : ℕ → ℕ
  /-- `F32ToE4m3Sat_v1` -/
  f32ToE4m3Sat : ℕ → ℕ
  /-- `F32ToE2m1Sat_v1` -/
  f32ToE2m1Sat : ℕ → ℕ
  /-- `E4m3ToF32_v1` -/
  e4m3ToF32 : ℕ → ℕ
  /-- `F2fpBf16_v1` -/
  f2fpBf16 : ℕ → ℕ
  /-- `I32Le_v1` -/
  i32Le : ℕ → ℕ → ℕ
  /-- `SelectI32_v1`: the condition, then the values for `1` and for `0` -/
  selectI32 : ℕ → ℕ → ℕ → ℕ
  /-- `SelectBf16_v1` -/
  selectBf16 : ℕ → ℕ → ℕ → ℕ
  /-- `BitAnd_v1` -/
  bitAnd : ℕ → ℕ → ℕ
  /-- `I32Add_v1` -/
  i32Add : ℕ → ℕ → ℕ
  /-- `F32Fabs_v1`, wiring -/
  f32Fabs : ℕ → ℕ
  /-- `Bf16ToF32_v1`, wiring -/
  bf16ToF32 : ℕ → ℕ

/-- Core's primitives, as core's Lean defines them. -/
def coreOps : RowOps where
  f32MulV2 := Verity.Fp32.f32MulV2
  f32DivV2 := Verity.Fp32.f32DivV2
  f32FmaV2 := Verity.Fp32.f32FmaV2
  f32Fmaxf := Verity.Scalar.f32Fmaxf
  f32Neg := Verity.Scalar.f32Neg
  f32ToE4m3Sat := Verity.Prims.f32ToE4m3Sat
  f32ToE2m1Sat := Verity.Prims.f32ToE2m1Sat
  e4m3ToF32 := Verity.Prims.e4m3ToF32
  f2fpBf16 := Verity.Prims.f2fpBf16
  i32Le := Verity.Scalar.i32Le
  selectI32 := Verity.Scalar.selectI32
  selectBf16 := Verity.Scalar.selectBf16
  bitAnd := Verity.Scalar.bitAnd
  i32Add := Verity.Scalar.i32Add
  f32Fabs := Verity.Scalar.f32Fabs
  bf16ToF32 := Verity.Prims.bf16ToF32

/-! ## Gates that name their primitive -/

/-- The row units' primitives. -/
inductive Prim
  /-- `F32Add_v2` -/
  | add
  /-- `F32Mul_v2` -/
  | mul
  /-- `F32Div_v2` -/
  | div
  /-- `F32Fma_v2` -/
  | fma
  /-- `PouwF32SqrtRn_v1` -/
  | sqrt
  /-- `F32Fmaxf_v1` -/
  | fmaxf
  /-- `F32Neg_v1` -/
  | neg
  /-- `F32ToE4m3Sat_v1` -/
  | toE4m3
  /-- `F32ToE2m1Sat_v1` -/
  | toE2m1
  /-- `E4m3ToF32_v1` -/
  | e4m3ToF32
  /-- `F2fpBf16_v1` -/
  | f2fpBf16
  /-- `I32Le_v1` -/
  | le
  /-- `SelectI32_v1` -/
  | selI32
  /-- `SelectBf16_v1` -/
  | selBf16
  /-- `BitAnd_v1` -/
  | band
  /-- `I32Add_v1` -/
  | iadd
  /-- The device's E4M3 k32 atom on the step `p` (`BlackwellE4m3QmmaDot32_v1` on sm_120, `HopperE4m3QgmmaDot32_v1` on
  the H100) -/
  | atom8 (p : Pipeline)
  /-- `BlackwellNvf4OmmaDot64_v1` -/
  | atom4
  /-- `PouwETilde_v1` -/
  | eTilde
  /-- `PouwBf16Neg_v1` -/
  | bf16Neg
  /-- `PouwSqAcc_v1` -/
  | sqAcc
  /-- `PouwSqCeil_v1` -/
  | sqCeil
  /-- `PouwSqGe_v1` -/
  | sqGe

/-- A primitive's width, its return type's. -/
def Prim.w : Prim → ℕ
  | .toE4m3 => 8
  | .toE2m1 => 4
  | .f2fpBf16 | .selBf16 | .eTilde | .bf16Neg => 16
  | .le | .band | .sqGe => 1
  | .sqAcc => 576
  | .sqCeil => 64
  | _ => 32

/-- A primitive's value on its operands' values in order, reduced to its width. -/
def Prim.eval (ops : RowOps) (q : Prim) (l : List ℕ) : ℕ :=
  let x := l.getD 0 0
  let y := l.getD 1 0
  let z := l.getD 2 0
  (match q with
    | .add => f32Add x y
    | .mul => ops.f32MulV2 x y
    | .div => ops.f32DivV2 x y
    | .fma => ops.f32FmaV2 x y z
    | .sqrt => sqrtWord x
    | .fmaxf => ops.f32Fmaxf x y
    | .neg => ops.f32Neg x
    | .toE4m3 => ops.f32ToE4m3Sat x
    | .toE2m1 => ops.f32ToE2m1Sat x
    | .e4m3ToF32 => ops.e4m3ToF32 x
    | .f2fpBf16 => ops.f2fpBf16 x
    | .le => ops.i32Le x y
    | .selI32 => ops.selectI32 x y z
    | .selBf16 => ops.selectBf16 x y z
    | .band => ops.bitAnd x y
    | .iadd => ops.i32Add x y
    | .atom8 p => atomF p false l
    | .atom4 => atom4F false l
    | .eTilde => eTildeWord x y
    | .bf16Neg => bf16NegWord x
    | .sqAcc => sqAccWord x y
    | .sqCeil => sqCeilWord x y
    | .sqGe => sqGeWord x y) % 2 ^ q.w

/-- How an operand reads its gate: plain, or through core's wiring, the widening first. -/
inductive Wire
  | id
  /-- `F32Fabs_v1` -/
  | fabs
  /-- `Bf16ToF32_v1` -/
  | bf16
  /-- `F32Fabs_v1` of `Bf16ToF32_v1` -/
  | bf16Fabs

/-- The wiring's value on the gate's. -/
def Wire.eval (ops : RowOps) : Wire → ℕ → ℕ
  | .id, x => x
  | .fabs, x => ops.f32Fabs x
  | .bf16, x => ops.bf16ToF32 x
  | .bf16Fabs, x => ops.f32Fabs (ops.bf16ToF32 x)

/-- One operand: gate `a` read through `w`, or a constant. -/
inductive Arg
  | g (a : ℕ) (w : Wire := .id)
  | c (v : ℕ)

/-- The gates an operand reads. -/
def Arg.reads : Arg → List ℕ
  | .g a _ => [a]
  | .c _ => []

/-- An operand's value, the gates' values being `V`. -/
def Arg.eval (ops : RowOps) (V : ℕ → ℕ) : Arg → ℕ
  | .g a w => w.eval ops (V a)
  | .c v => v

/-- A computed gate: its primitive and its operands in order. -/
structure RGate where
  prim : Prim
  args : List Arg

/-- The gates a computed gate reads, in its operands' order. -/
def RGate.reads (G : RGate) : List ℕ := G.args.flatMap Arg.reads

/-- The operands' values from the values `l` of the gates they read, in order. -/
def fill (ops : RowOps) : List Arg → List ℕ → List ℕ
  | [], _ => []
  | .c v :: as, l => v :: fill ops as l
  | .g _ w :: as, l => w.eval ops (l.headD 0) :: fill ops as l.tail

/-- A computed gate as a `TileCircuit.GateRow`. -/
def RGate.toGate (ops : RowOps) (G : RGate) : GateRow :=
  ⟨G.prim.w, false, G.reads, fun l => G.prim.eval ops (fill ops G.args l)⟩

/-! ## Tables of runs -/

/-- A run of `n` computed gates, the `r`-th being `gate r`. -/
structure Run where
  n : ℕ
  gate : ℕ → RGate

/-- `m` blocks of `L` gates, gate `r` of block `q` being `blk q r`. -/
def Run.rep (m L : ℕ) (blk : ℕ → ℕ → RGate) : Run := ⟨m * L, fun r => blk (r / L) (r % L)⟩

/-- A run of the listed gates. -/
def Run.ofList (l : List RGate) : Run := ⟨l.length, fun r => l.getD r ⟨.add, []⟩⟩

/-- A run from gate `o` on reads earlier gates only. -/
def Run.OK (R : Run) (o : ℕ) : Prop := ∀ r < R.n, ∀ a ∈ (R.gate r).reads, a < o + r

/-- The runs' gate count. -/
def runsLen : List Run → ℕ
  | [] => 0
  | R :: L => R.n + runsLen L

/-- Gate `g ≥ o` of runs laid out from gate `o`. -/
def runsAt : List Run → ℕ → ℕ → Option RGate
  | [], _, _ => none
  | R :: L, o, g => if g < o + R.n then some (R.gate (g - o)) else runsAt L (o + R.n) g

/-- Runs laid out from gate `o` read earlier gates only. -/
def RunsOK : List Run → ℕ → Prop
  | [], _ => True
  | R :: L, o => R.OK o ∧ RunsOK L (o + R.n)

theorem runsAt_reads : ∀ (L : List Run) (o : ℕ), RunsOK L o → ∀ g G, o ≤ g → runsAt L o g = some G →
    ∀ a ∈ G.reads, a < g
  | [], _, _, _, _, _, h => by simp [runsAt] at h
  | R :: L, o, ⟨hR, hL⟩, g, G, hog, h => by
    unfold runsAt at h
    split_ifs at h with hg
    · cases h
      intro a ha
      have := hR (g - o) (by omega) a ha
      omega
    · exact runsAt_reads L (o + R.n) hL g G (by omega) h

theorem runsOK_cons {R : Run} {L : List Run} {o o' : ℕ} (h : R.OK o) (ho : o + R.n = o') (hL : RunsOK L o') :
    RunsOK (R :: L) o :=
  ⟨h, ho ▸ hL⟩

theorem runsOK_append : ∀ {L₁ L₂ : List Run} {o o' : ℕ}, RunsOK L₁ o → o + runsLen L₁ = o' → RunsOK L₂ o' →
    RunsOK (L₁ ++ L₂) o
  | [], _, _, _, _, ho, h₂ => by simp only [runsLen, Nat.add_zero] at ho; subst ho; exact h₂
  | _ :: L₁, _, _, _, ⟨hR, h₁⟩, ho, h₂ => ⟨hR, runsOK_append h₁ (by simp only [runsLen] at ho; omega) h₂⟩

@[simp] theorem runsLen_append : ∀ L₁ L₂ : List Run, runsLen (L₁ ++ L₂) = runsLen L₁ + runsLen L₂
  | [], _ => by simp [runsLen]
  | R :: L₁, L₂ => by simp only [List.cons_append, runsLen, runsLen_append L₁ L₂]; omega

/-- An operand reads only gates below `n`. -/
def Arg.Below (n : ℕ) : Arg → Prop
  | .g a _ => a < n
  | .c _ => True

theorem RGate.reads_lt {G : RGate} {n : ℕ} (h : ∀ x ∈ G.args, x.Below n) : ∀ a ∈ G.reads, a < n := by
  intro a ha
  obtain ⟨x, hx, hax⟩ := List.mem_flatMap.1 ha
  have := h x hx
  cases x with
  | g b w => simp only [Arg.reads, List.mem_singleton] at hax; subst hax; exact this
  | c v => simp [Arg.reads] at hax

@[simp] theorem Arg.below_g (n a : ℕ) (w : Wire) : (Arg.g a w).Below n ↔ a < n := Iff.rfl

@[simp] theorem Arg.below_c (n v : ℕ) : (Arg.c v).Below n := trivial

@[simp] theorem Arg.below_ite {p : Prop} [Decidable p] (n : ℕ) (x y : Arg) :
    (if p then x else y).Below n ↔ (p → x.Below n) ∧ (¬p → y.Below n) := by
  split_ifs with h <;> simp [h]

theorem Run.rep_ok {m L o : ℕ} {blk : ℕ → ℕ → RGate}
    (h : ∀ q < m, ∀ r < L, ∀ x ∈ (blk q r).args, x.Below (o + q * L + r)) : (Run.rep m L blk).OK o := by
  intro r hr
  have hL : 0 < L := Nat.pos_of_ne_zero (by rintro rfl; simp [Run.rep] at hr)
  have hq : r / L < m := (Nat.div_lt_iff_lt_mul hL).2 (by simpa [Run.rep, Nat.mul_comm] using hr)
  have := RGate.reads_lt (h (r / L) hq (r % L) (Nat.mod_lt _ hL))
  rwa [Nat.add_assoc, Nat.div_add_mod'] at this

/-- Listed gates from gate `o` on read earlier gates only. -/
def ListOK : List RGate → ℕ → Prop
  | [], _ => True
  | G :: l, o => (∀ x ∈ G.args, x.Below o) ∧ ListOK l (o + 1)

theorem Run.ofList_ok : ∀ {l : List RGate} {o : ℕ}, ListOK l o → (Run.ofList l).OK o
  | [], _, _ => by intro r hr; simp [Run.ofList] at hr
  | G :: l, o, ⟨hG, hl⟩ => by
    intro r hr
    rcases r with _ | r
    · exact RGate.reads_lt hG
    · have := Run.ofList_ok hl r (by simpa [Run.ofList] using hr)
      intro a ha
      have := this a (by simpa [Run.ofList] using ha)
      omega

/-! ## The statistics both families compute -/

/-- The FMA partials' first gate, after the running max's `2·(K − 1)` from `b`. -/
def fmaAt (K b : ℕ) : ℕ := b + 2 * (K - 1)

/-- The butterfly's rounds' first gates, after the `K/8` partials: rounds of `16`, `8`, `4`, `2` and `1` sums, each no
more than the `K/8` live accumulators. -/
def bf0At (K b : ℕ) : ℕ := fmaAt K b + K / 8

def bf1At (K b : ℕ) : ℕ := bf0At K b + min 16 (K / 8)

def bf2At (K b : ℕ) : ℕ := bf1At K b + min 8 (K / 8)

def bf3At (K b : ℕ) : ℕ := bf2At K b + min 4 (K / 8)

def bf4At (K b : ℕ) : ℕ := bf3At K b + min 2 (K / 8)

/-- `ρ`'s two gates, the mean square's division and its root, after the butterfly. -/
def rhoAt (K b : ℕ) : ℕ := bf4At K b + min 1 (K / 8)

/-- The last FMA partial into accumulator `j < 32`: position `8i` goes to accumulator `i mod 32`. -/
def lastFma (K b j : ℕ) : ℕ := fmaAt K b + j + 32 * ((K / 8 - 1 - j) / 32)

/-- The running max `s` of `|x_t|` (`ab` its wiring, `|x_0|` the start): per `t = 1, …, K − 1` an `I32Le_v1` of `s` and
`|x_t|`, then the `SelectI32_v1` of `|x_t|` where it holds, else `s`. -/
def selRun (ab : Wire) (K b : ℕ) : Run := ⟨2 * (K - 1), fun r =>
  if r % 2 = 0 then ⟨.le, [if r = 0 then .g 0 ab else .g (b + r - 1), .g (r / 2 + 1) ab]⟩
  else ⟨.selI32, [.g (b + r - 1), .g (r / 2 + 1) ab, if r = 1 then .g 0 ab else .g (b + r - 2)]⟩⟩

/-- The FMA partials: position `8i`'s `fma(x, x, ·)` (`w` the words' wiring) into accumulator `i mod 32`, from `+0`. -/
def fmaRun (w : Wire) (K b : ℕ) : Run := ⟨K / 8, fun i =>
  ⟨.fma, [.g (8 * i) w, .g (8 * i) w, if i < 32 then .c 0 else .g (fmaAt K b + i - 32)]⟩⟩

/-- The butterfly's first round: accumulator `j` plus accumulator `j + 16` (`+0` past the live ones). -/
def bfly0Run (K b : ℕ) : Run := ⟨min 16 (K / 8), fun j =>
  ⟨.add, [.g (lastFma K b j), if j + 16 < K / 8 then .g (lastFma K b (j + 16)) else .c 0]⟩⟩

/-- A later round of the butterfly, the previous one's from `pb`: sum `j` plus sum `j + h`. -/
def bflyRun (h K pb : ℕ) : Run := ⟨min h (K / 8), fun j =>
  ⟨.add, [.g (pb + j), if j + h < K / 8 then .g (pb + j + h) else .c 0]⟩⟩

/-- `ρ`: the butterfly's sum divided by the FP32 word of `K/8`, then its root. -/
def rhoRun (K b : ℕ) : Run :=
  Run.ofList [⟨.div, [.g (rhoAt K b - 1), .c (f32Word ((K / 8 : ℕ) : ℚ))]⟩, ⟨.sqrt, [.g (rhoAt K b)]⟩]

/-- The statistics' runs from gate `b` (`_stats` up to `ρ`), `x` at gates `0, …, K − 1` read through `w` (`ab` for
`|x|`). -/
def statsRuns (ab w : Wire) (K b : ℕ) : List Run :=
  [selRun ab K b, fmaRun w K b, bfly0Run K b, bflyRun 8 K (bf0At K b), bflyRun 4 K (bf1At K b),
    bflyRun 2 K (bf2At K b), bflyRun 1 K (bf3At K b), rhoRun K b]

theorem statsRuns_ok (ab w : Wire) {K b : ℕ} (hb : K < b) : RunsOK (statsRuns ab w K b) b := by
  refine runsOK_cons (o' := fmaAt K b) ?_ (by simp [selRun, fmaAt]) <|
    runsOK_cons (o' := bf0At K b) ?_ (by simp [fmaRun, bf0At]) <|
    runsOK_cons (o' := bf1At K b) ?_ (by simp [bfly0Run, bf1At]) <|
    runsOK_cons (o' := bf2At K b) ?_ (by simp [bflyRun, bf2At]) <|
    runsOK_cons (o' := bf3At K b) ?_ (by simp [bflyRun, bf3At]) <|
    runsOK_cons (o' := bf4At K b) ?_ (by simp [bflyRun, bf4At]) <|
    runsOK_cons (o' := rhoAt K b) ?_ (by simp [bflyRun, rhoAt]) <|
    runsOK_cons (o' := rhoAt K b + 2) ?_ (by simp [rhoRun, Run.ofList]) trivial
  · intro r hr
    apply RGate.reads_lt
    simp only [selRun] at hr ⊢
    split_ifs <;> simp <;> omega
  · intro i hi
    apply RGate.reads_lt
    simp only [fmaRun] at hi ⊢
    simp [fmaAt]
    omega
  · intro j hj
    apply RGate.reads_lt
    simp only [bfly0Run] at hj ⊢
    simp [lastFma, bf0At, fmaAt]
    omega
  · intro j hj
    apply RGate.reads_lt
    simp only [bflyRun] at hj ⊢
    simp [bf0At, bf1At]
    omega
  · intro j hj
    apply RGate.reads_lt
    simp only [bflyRun] at hj ⊢
    simp [bf1At, bf2At]
    omega
  · intro j hj
    apply RGate.reads_lt
    simp only [bflyRun] at hj ⊢
    simp [bf2At, bf3At]
    omega
  · intro j hj
    apply RGate.reads_lt
    simp only [bflyRun] at hj ⊢
    simp [bf3At, bf4At]
    omega
  · apply Run.ofList_ok
    simp [ListOK, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
    omega

theorem statsRuns_len (ab w : Wire) (K b : ℕ) : b + runsLen (statsRuns ab w K b) = rhoAt K b + 2 := by
  simp [statsRuns, runsLen, selRun, fmaRun, bfly0Run, bflyRun, rhoRun, Run.ofList, rhoAt, bf4At, bf3At, bf2At, bf1At,
    bf0At, fmaAt]
  omega

/-! ## A row unit's table -/

/-- A row unit's table: `nIn` input gates, gate `a`'s width `inW a`, then the computed gates' runs from gate `nIn`, and
the outputs in the template's return order. -/
structure Rows where
  nIn : ℕ
  inW : ℕ → ℕ
  runs : List Run
  outs : List ℕ
  ok : RunsOK runs nIn

/-- The gate count. -/
def Rows.N (T : Rows) : ℕ := T.nIn + runsLen T.runs

/-- Gate `g` of the table, with `ops` for core's primitives. -/
def Rows.gate (ops : RowOps) (T : Rows) (g : ℕ) : GateRow :=
  if g < T.nIn then ⟨T.inW g, true, [], fun _ => 0⟩
  else match runsAt T.runs T.nIn g with
    | some G => G.toGate ops
    | none => ⟨0, true, [], fun _ => 0⟩

/-- **A row unit as a word circuit**, with `ops` for core's primitives. -/
def Rows.spec (ops : RowOps) (T : Rows) : Spec where
  N := T.N
  gate := T.gate ops
  outs := T.outs.toFinset
  rd_lt g _ a ha := by
    unfold Rows.gate at ha
    split_ifs at ha with h
    · simp at ha
    · split at ha
      · next G hG => exact runsAt_reads T.runs T.nIn T.ok g G (by omega) hG a ha
      · simp at ha
  input_rd g _ hi := by
    unfold Rows.gate at hi ⊢
    split_ifs with h
    · rfl
    · split
      · next G hG => simp [h, hG, RGate.toGate] at hi
      · rfl

/-! ## `pc8`'s rows -/

/-- `pc8`'s row inputs: `x`'s `K` FP32 words (gates `0, …, K − 1`), the noise `e`'s 32 E4M3 codes (from `K`), the form
basis' and the chain basis' `32·K` codes (from `K + 32` and `33·K + 32`, row `t`'s 32 at `·+ 32·t`), then on RowB `g`'s
1024 FP32 words (from `65·K + 32`, `g_{s,q}` at `·+ 32·s + q`). -/
def rowIn (K : ℕ) (b : Bool) : ℕ := 65 * K + 32 + if b then 1024 else 0

/-- `α`'s run's first gate, after `ρ`. -/
def al8At (K b : ℕ) : ℕ := rhoAt K b + 2

/-- The exact sum of squares' run, after `α`'s six gates. -/
def sq8At (K b : ℕ) : ℕ := al8At K b + 6

/-- The threshold `θ`, after the `K/8` sums. -/
def th8At (K b : ℕ) : ℕ := sq8At K b + K / 8

/-- The dead-position count's run, after `θ`. -/
def dead8At (K b : ℕ) : ℕ := th8At K b + 1

/-- The flag's run, after the count's `3·K` gates. -/
def ok8At (K b : ℕ) : ℕ := dead8At K b + 3 * K

/-- The rescaled noise' run, after the flag's eight gates. -/
def ep8At (K b : ℕ) : ℕ := ok8At K b + 8

/-- The rescaled noise' 32 FP32 words, after its `96` gates. -/
def ew8At (K b : ℕ) : ℕ := ep8At K b + 96

/-- The forming's run, after the 32 words. -/
def code8At (K b : ℕ) : ℕ := ew8At K b + 32

/-- The side's own runs, after the forming's `3·K` gates. -/
def post8At (K b : ℕ) : ℕ := code8At K b + 3 * K

/-- `α` and `β`: `α = 448 / max(s + 4·ρ, 2⁻³²)` (`s` the running max, the last gate before the partials), then
`β = ρ·α·2^(−19/2)`, each gate as the template computes it. -/
def alpha8Run (K b : ℕ) : Run :=
  Run.ofList [⟨.mul, [.c 0x40800000, .g (rhoAt K b + 1)]⟩, ⟨.add, [.g (fmaAt K b - 1), .g (al8At K b)]⟩,
    ⟨.fmaxf, [.g (al8At K b + 1), .c 0x2F800000]⟩, ⟨.div, [.c 0x43E00000, .g (al8At K b + 2)]⟩,
    ⟨.mul, [.g (rhoAt K b + 1), .g (al8At K b + 3)]⟩, ⟨.mul, [.g (al8At K b + 4), .c 0x3AB504F3]⟩]

/-- The exact scaled sum of `x²` over every 8th position, from `0`. -/
def sq8Run (K b : ℕ) : Run := ⟨K / 8, fun i =>
  ⟨.sqAcc, [if i = 0 then .c 0 else .g (sq8At K b + i - 1), .g (8 * i)]⟩⟩

/-- The threshold `θ` of the sum over the `K/8` positions. -/
def th8Run (K b : ℕ) : Run := Run.ofList [⟨.sqCeil, [.g (th8At K b - 1), .c (K / 8)]⟩]

/-- The dead positions' count: per position `t`, `x_t²`'s comparison with `θ`, its `SelectI32_v1` to `1` or `0`, and
the running `I32Add_v1` from `0`. -/
def dead8Run (K b : ℕ) : Run := ⟨3 * K, fun r =>
  if r % 3 = 0 then ⟨.sqGe, [.g (r / 3), .g (th8At K b)]⟩
  else if r % 3 = 1 then ⟨.selI32, [.g (dead8At K b + r - 1), .c 1, .c 0]⟩
  else ⟨.iadd, [if r = 2 then .c 0 else .g (dead8At K b + r - 3), .g (dead8At K b + r - 1)]⟩⟩

/-- The flag: `ρ·α − 1 ≥ 0`, `s` and `ρ` finite, and at most `K/64` dead positions; the last gate is the flag. -/
def ok8Run (K b : ℕ) : Run :=
  let o := ok8At K b
  Run.ofList [⟨.fma, [.g (rhoAt K b + 1), .g (al8At K b + 3), .c 0xBF800000]⟩, ⟨.le, [.c 0, .g o]⟩,
    ⟨.le, [.g (fmaAt K b - 1), .c 0x7F7FFFFF]⟩, ⟨.le, [.g (rhoAt K b + 1), .c 0x7F7FFFFF]⟩,
    ⟨.band, [.g (o + 2), .g (o + 3)]⟩, ⟨.le, [.g (o - 1), .c (K / 64)]⟩, ⟨.band, [.g (o + 1), .g (o + 5)]⟩,
    ⟨.band, [.g (o + 4), .g (o + 6)]⟩]

/-- The rescaled noise: per `e`'s code, its FP32 value, times `β`, cast to E4M3. -/
def ep8Run (K b : ℕ) : Run := ⟨96, fun r =>
  if r % 3 = 0 then ⟨.e4m3ToF32, [.g (K + r / 3)]⟩
  else if r % 3 = 1 then ⟨.mul, [.g (al8At K b + 5), .g (ep8At K b + r - 1)]⟩
  else ⟨.toE4m3, [.g (ep8At K b + r - 1)]⟩⟩

/-- The rescaled noise' codes as FP32 words. -/
def ew8Run (K b : ℕ) : Run := ⟨32, fun l => ⟨.e4m3ToF32, [.g (ep8At K b + 3 * l + 2)]⟩⟩

/-- The forming on the step `p`: per position `t`, `α·x_t`, one atom adding the rescaled noise' codes against the form
basis' row `t`, and its E4M3 cast, `A′`'s or `B̃`'s code at `t`. -/
def code8Run (p : Pipeline) (K b : ℕ) : Run := ⟨3 * K, fun r =>
  if r % 3 = 0 then ⟨.mul, [.g (al8At K b + 3), .g (r / 3)]⟩
  else if r % 3 = 1 then
    ⟨.atom8 p, (.g (code8At K b + r - 1) :: (List.range 32).map fun l => .g (ep8At K b + 3 * l + 2)) ++
      (List.range 32).map fun l => .g (K + 32 + 32 * (r / 3) + l)⟩
  else ⟨.toE4m3, [.g (code8At K b + r - 1)]⟩⟩

/-- Position `t`'s code: the forming's cast. -/
def code8 (K b t : ℕ) : ℕ := code8At K b + 3 * t + 2

/-- The `pc8` row units' runs both sides share, from gate `b`. -/
def pre8Runs (p : Pipeline) (K b : ℕ) : List Run :=
  statsRuns .fabs .id K b ++ [alpha8Run K b, sq8Run K b, th8Run K b, dead8Run K b, ok8Run K b, ep8Run K b, ew8Run K b,
    code8Run p K b]

/-- Gate `r < 5·(K/128)` of a chain on the step `p` at gate `gt`, the codes at `code t` and the other side at `col t`:
per window of 4 atoms, four atoms from `+0` and the promotion into the total (`+0` first), as the tile's chain. -/
def chain8 (p : Pipeline) (code col : ℕ → ℕ) (gt r : ℕ) : RGate :=
  if r % 5 < 4 then
    ⟨.atom8 p, ((if r % 5 = 0 then Arg.c 0 else .g (gt - 1)) ::
      (List.range 32).map fun l => .g (code (32 * (4 * (r / 5) + r % 5) + l))) ++
      (List.range 32).map fun l => .g (col (32 * (4 * (r / 5) + r % 5) + l))⟩
  else ⟨.add, [if r / 5 = 0 then .c 0 else .g (gt - 5), .g (gt - 1)]⟩

/-- RowA's `P_A` chains: per column `q < 32` of the chain basis, the chain of `A′`'s codes against it and its BF16
cast, `P_A`'s word `q`. -/
def af8Run (p : Pipeline) (K b : ℕ) : Run :=
  Run.rep 32 (5 * (K / 128) + 1) fun q s =>
    if s < 5 * (K / 128) then
      chain8 p (code8 K b) (fun t => 33 * K + 32 + 32 * t + q) (post8At K b + q * (5 * (K / 128) + 1) + s) s
    else ⟨.f2fpBf16, [.g (post8At K b + q * (5 * (K / 128) + 1) + s - 1)]⟩

/-- The rescaled noise' 32 words as BF16. -/
def bew8Run (K b : ℕ) : Run := ⟨32, fun l => ⟨.f2fpBf16, [.g (ew8At K b + l)]⟩⟩

/-- The flag as a BF16 word, `1` or `0`. -/
def okwRun (ok : ℕ) : Run := Run.ofList [⟨.selBf16, [.g ok, .c 1, .c 0]⟩]

theorem pre8Runs_ok (p : Pipeline) {K b : ℕ} (hb : 65 * K + 32 ≤ b) : RunsOK (pre8Runs p K b) b := by
  refine runsOK_append (statsRuns_ok .fabs .id (by omega)) (statsRuns_len .fabs .id K b) ?_
  refine runsOK_cons (o' := sq8At K b) ?_ (by simp [alpha8Run, Run.ofList, sq8At, al8At]) <|
    runsOK_cons (o' := th8At K b) ?_ (by simp [sq8Run, th8At]) <|
    runsOK_cons (o' := dead8At K b) ?_ (by simp [th8Run, Run.ofList, dead8At]) <|
    runsOK_cons (o' := ok8At K b) ?_ (by simp [dead8Run, ok8At]) <|
    runsOK_cons (o' := ep8At K b) ?_ (by simp [ok8Run, Run.ofList, ep8At]) <|
    runsOK_cons (o' := ew8At K b) ?_ (by simp [ep8Run, ew8At]) <|
    runsOK_cons (o' := code8At K b) ?_ (by simp [ew8Run, code8At]) <|
    runsOK_cons (o' := post8At K b) ?_ (by simp [code8Run, post8At]) trivial
  · apply Run.ofList_ok
    simp [ListOK, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
    omega
  · intro i hi
    apply RGate.reads_lt
    simp only [sq8Run] at hi ⊢
    split_ifs <;> simp [sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt] <;> omega
  · apply Run.ofList_ok
    simp [ListOK, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
  · intro r hr
    apply RGate.reads_lt
    simp only [dead8Run] at hr ⊢
    split_ifs <;> simp [dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt] <;> omega
  · apply Run.ofList_ok
    simp [ListOK, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
    omega
  · intro r hr
    apply RGate.reads_lt
    simp only [ep8Run] at hr ⊢
    split_ifs <;> simp [ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At,
      fmaAt] <;> omega
  · intro l hl
    apply RGate.reads_lt
    simp only [ew8Run] at hl ⊢
    simp [ew8At]
    omega
  · intro r hr
    apply RGate.reads_lt
    simp only [code8Run] at hr ⊢
    split_ifs
    · simp [code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At,
        fmaAt]
      omega
    · simp only [List.mem_cons, List.mem_append, List.mem_map, List.mem_range]
      rintro x ((rfl | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩) <;> simp [code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At,
        al8At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt] <;> omega
    · simp [code8At]

theorem pre8Runs_len (p : Pipeline) (K b : ℕ) : b + runsLen (pre8Runs p K b) = post8At K b := by
  have := statsRuns_len .fabs .id K b
  simp only [pre8Runs, runsLen_append] at this ⊢
  simp [runsLen, alpha8Run, sq8Run, th8Run, dead8Run, ok8Run, ep8Run, ew8Run, code8Run, Run.ofList, post8At, code8At,
    ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At]
  omega

theorem chain8_ok (p : Pipeline) {K o : ℕ} {code col : ℕ → ℕ} {gt r : ℕ} (hr : r < 5 * (K / 128))
    (hcode : ∀ t < K, code t < o) (hcol : ∀ t < K, col t < o) (ho : o ≤ gt) (hgt : r ≤ gt) :
    ∀ x ∈ (chain8 p code col gt r).args, x.Below gt := by
  unfold chain8
  split_ifs with h1 h2 h3 <;>
    simp only [List.forall_mem_cons, List.forall_mem_append, List.forall_mem_map, List.mem_range, Arg.below_g,
      Arg.below_c] <;> and_intros <;> intros <;> (try trivial) <;>
    (first
      | omega
      | exact Nat.lt_of_lt_of_le (hcode _ (by omega)) ho
      | exact Nat.lt_of_lt_of_le (hcol _ (by omega)) ho)

theorem af8Run_ok (p : Pipeline) {K b : ℕ} (hb : 65 * K + 32 ≤ b) : (af8Run p K b).OK (post8At K b) := by
  apply Run.rep_ok
  intro q hq s hs
  generalize q * (5 * (K / 128) + 1) = Q
  split_ifs with h
  · apply chain8_ok p h (o := post8At K b)
    · intro t ht; simp only [code8, post8At]; omega
    · intro t ht; simp only [post8At, code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At,
        bf2At, bf1At, bf0At, fmaAt]; omega
    · omega
    · omega
  · simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, Arg.below_g]
    simp only [post8At, code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At,
      bf0At, fmaAt]
    omega

theorem bew8Run_ok (K b : ℕ) {o : ℕ} (ho : post8At K b ≤ o) : (bew8Run K b).OK o := by
  intro l hl
  apply RGate.reads_lt
  simp only [bew8Run] at hl ⊢
  simp [post8At, code8At] at ho ⊢
  omega

theorem okwRun_ok {ok o : ℕ} (h : ok < o) : (okwRun ok).OK o := by
  apply Run.ofList_ok
  simp [ListOK, h]

/-- **`Pc8RowA_v1{K}`** on the step `p`: its outputs are `A′`'s `K` codes, the rescaled noise' 32 BF16 words, `P_A`'s
32 and the flag. -/
def row8A (p : Pipeline) (K : ℕ) : Rows where
  nIn := rowIn K false
  inW a := if a < K then 32 else 8
  runs := pre8Runs p K (rowIn K false) ++ [af8Run p K (rowIn K false), bew8Run K (rowIn K false),
    okwRun (ok8At K (rowIn K false) + 7)]
  outs := (List.range K).map (code8 K (rowIn K false)) ++
    (List.range 32).map (post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + ·) ++
    (List.range 32).map (fun l => post8At K (rowIn K false) + l * (5 * (K / 128) + 1) + 5 * (K / 128)) ++
    [post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + 32]
  ok := by
    have hb : 65 * K + 32 ≤ rowIn K false := by simp [rowIn]
    refine runsOK_append (pre8Runs_ok p hb) (pre8Runs_len p K _) ?_
    refine runsOK_cons (af8Run_ok p hb) rfl <| runsOK_cons (bew8Run_ok K _ (by simp [af8Run, Run.rep])) rfl <|
      runsOK_cons (okwRun_ok ?_) rfl trivial
    simp [af8Run, Run.rep, bew8Run, post8At, code8At, ew8At, ep8At]
    omega

/-- Gate `s` of RowB's block `q`, at gate `gt`: the 32 FMAs of `Σ_s ẽ_s·g_{s,q}` (from `+0`), the chain of `B̃`'s codes
against the chain basis' column `q`, its negation, the sum, and its BF16 cast, `P_B`'s word `q`. -/
def mid8 (p : Pipeline) (K b q s gt : ℕ) : RGate :=
  if s < 32 then ⟨.fma, [.g (ew8At K b + s), .g (65 * K + 32 + 32 * s + q), if s = 0 then .c 0 else .g (gt - 1)]⟩
  else if s < 32 + 5 * (K / 128) then chain8 p (code8 K b) (fun t => 33 * K + 32 + 32 * t + q) gt (s - 32)
  else if s = 32 + 5 * (K / 128) then ⟨.neg, [.g (gt - 1)]⟩
  else if s = 33 + 5 * (K / 128) then ⟨.add, [.g (gt - 2 - 5 * (K / 128)), .g (gt - 1)]⟩
  else ⟨.f2fpBf16, [.g (gt - 1)]⟩

/-- RowB's `P_B` blocks, one per column `q < 32`. -/
def mid8Run (p : Pipeline) (K b : ℕ) : Run :=
  Run.rep 32 (5 * (K / 128) + 35) fun q s => mid8 p K b q s (post8At K b + q * (5 * (K / 128) + 35) + s)

/-- The rescaled noise' words as BF16, each followed by its negation. -/
def nb8Run (K b o : ℕ) : Run := ⟨64, fun r =>
  if r % 2 = 0 then ⟨.f2fpBf16, [.g (ew8At K b + r / 2)]⟩ else ⟨.bf16Neg, [.g (o + r - 1)]⟩⟩

theorem mid8Run_ok (p : Pipeline) {K b : ℕ} (hb : 65 * K + 1056 ≤ b) : (mid8Run p K b).OK (post8At K b) := by
  apply Run.rep_ok
  intro q hq s hs
  generalize q * (5 * (K / 128) + 35) = Q
  unfold mid8
  by_cases h1 : s < 32
  · simp only [h1, ↓reduceIte]
    simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, true_and, Arg.below_g,
      Arg.below_ite, Arg.below_c]
    simp only [post8At, code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At,
      bf0At, fmaAt]
    omega
  simp only [h1, ↓reduceIte]
  by_cases h2 : s < 32 + 5 * (K / 128)
  · simp only [h2, ↓reduceIte]
    apply chain8_ok p (K := K) (by omega) (o := post8At K b)
    · intro t ht; simp only [code8, post8At]; omega
    · intro t ht; simp only [post8At, code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At,
        bf2At, bf1At, bf0At, fmaAt]; omega
    · omega
    · omega
  simp only [h2, ↓reduceIte]
  split_ifs <;>
    simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, Arg.below_g] <;>
    simp only [post8At, code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At,
      bf0At, fmaAt] <;> omega

theorem nb8Run_ok (K b : ℕ) {o : ℕ} (ho : post8At K b ≤ o) : (nb8Run K b o).OK o := by
  intro r hr
  apply RGate.reads_lt
  simp only [nb8Run] at hr ⊢
  split_ifs <;> simp [post8At, code8At] at ho ⊢ <;> omega

/-- **`Pc8RowB_v1{K}`** on the step `p`: its outputs are `B̃`'s `K` codes, `P_B`'s 32 words, the rescaled noise' 32
negated BF16 words and the flag. -/
def row8B (p : Pipeline) (K : ℕ) : Rows where
  nIn := rowIn K true
  inW a := if a < K then 32 else if a < 65 * K + 32 then 8 else 32
  runs := pre8Runs p K (rowIn K true) ++ [mid8Run p K (rowIn K true),
    nb8Run K (rowIn K true) (post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35)), okwRun (ok8At K (rowIn K true) + 7)]
  outs := (List.range K).map (code8 K (rowIn K true)) ++
    (List.range 32).map (fun q => post8At K (rowIn K true) + q * (5 * (K / 128) + 35) + 5 * (K / 128) + 34) ++
    (List.range 32).map (fun l => post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 2 * l + 1) ++
    [post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 64]
  ok := by
    have hb : 65 * K + 1056 ≤ rowIn K true := by simp [rowIn]
    refine runsOK_append (pre8Runs_ok p (by omega)) (pre8Runs_len p K _) ?_
    refine runsOK_cons (mid8Run_ok p hb) rfl <| runsOK_cons (nb8Run_ok K _ (Nat.le_add_right _ _)) rfl <|
      runsOK_cons (okwRun_ok ?_) rfl trivial
    simp [mid8Run, Run.rep, nb8Run, post8At, code8At, ew8At, ep8At]
    omega

end Pouw.PearlC.RowCircuit
