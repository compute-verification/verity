import Definitions.Pouw.PearlC.RowCircuit

/-!
# Pearl-C4's row units as core circuits (definitions only)

`pc4`'s row templates `Pc4RowA_v1{K}` and `Pc4RowB_v1{K}` (`verity.protocols.accounting.work.pouw.circuit.pc4`, their `_row_a_body`,
`_row_b_body`, `_alpha`, `_form` and `_chain`) as gate tables (`RowCircuit.Rows`), generic in the depth `K`, as
`RowCircuit` writes `pc8`'s. The inputs are laid out as `pc8`'s (`rowIn`), with `x` as BF16 words and the codes as
E2M1 nibbles; the statistics read `x` widened (`Wire.bf16`). Then:

* **`α`'s run** (`alpha4Run`): `α = 2688 / max(s + 4·δρ, 2⁻³²)` with `δ = 1/4`, `β = δρ·α·k_β`, `a_E` its E4M3 cast,
  and the flag (`β` in range, `s` finite);
* **the forming** (`d4Run`): per position, `α·x_t` and one NVF4 firing adding `e` at scale `a_E` against the form basis'
  row at scale `24`;
* **the quantisation** (`blk4Run`): per block of 16 positions, the block's max of `|d|`, its UE4M3 scale byte, and each
  position's E2M1 code;
* **the sides**: RowA's `P_A` chains (`af4Run`) and the noise' `Ẽ` (`et4Run`); RowB's `Ẽ`, its `P_B` blocks (`mid4Run`)
  and `−Ẽ`.

`tests/test_circuit_row_lean.py` checks every gate against the templates, as for `pc8`.
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

/-- `α`'s run's first gate, after `ρ`. -/
def a4At (K b : ℕ) : ℕ := rhoAt K b + 2

/-- The forming's run, after `α`'s 13 gates. -/
def d4At (K b : ℕ) : ℕ := a4At K b + 13

/-- The quantisation's run, after the forming's `2·K` gates. -/
def bk4At (K b : ℕ) : ℕ := d4At K b + 2 * K

/-- The side's own runs, after the quantisation's `52` gates per block of 16. -/
def post4At (K b : ℕ) : ℕ := bk4At K b + 52 * (K / 16)

/-- `α`, `β`, `a_E` and the flag: `δρ` (`δ = 1/4`), `4·δρ`, `s + 4·δρ` (`s` the running max), its floor at `2⁻³²`,
`α = 2688 / ·`, `δρ·α`, `β = δρ·α·k_β`, `a_E = e4m3(β)`, `β ≥ 2⁻⁷·15/8`, `β` finite, both, `s` finite, and the flag. -/
def alpha4Run (K b : ℕ) : Run :=
  let a := a4At K b
  Run.ofList [⟨.mul, [.g (rhoAt K b + 1), .c 0x3E800000]⟩, ⟨.mul, [.c 0x40800000, .g a]⟩,
    ⟨.add, [.g (fmaAt K b - 1), .g (a + 1)]⟩, ⟨.fmaxf, [.g (a + 2), .c 0x2F800000]⟩,
    ⟨.div, [.c 0x45280000, .g (a + 3)]⟩,
    ⟨.mul, [.g a, .g (a + 4)]⟩, ⟨.mul, [.g (a + 5), .c 0x3EA920A7]⟩, ⟨.toE4m3, [.g (a + 6)]⟩,
    ⟨.le, [.c 0x3C700000, .g (a + 6)]⟩, ⟨.le, [.g (a + 6), .c 0x7F7FFFFF]⟩, ⟨.band, [.g (a + 8), .g (a + 9)]⟩,
    ⟨.le, [.g (fmaAt K b - 1), .c 0x7F7FFFFF]⟩, ⟨.band, [.g (a + 11), .g (a + 10)]⟩]

/-- The forming: per position `t`, `α·x_t` (`x_t` widened) and one NVF4 firing from it over `e`'s 32 codes (then 32
zeros) at scale `a_E` against the form basis' row `t` (then 32 zeros) at scale `24`. -/
def d4Run (K b : ℕ) : Run := ⟨2 * K, fun r =>
  if r % 2 = 0 then ⟨.mul, [.g (a4At K b + 4), .g (r / 2) .bf16]⟩
  else ⟨.atom4, (.g (d4At K b + r - 1) :: (List.range 32).map (fun l => .g (K + l))) ++ List.replicate 32 (.c 0) ++
    (List.range 32).map (fun l => .g (K + 32 + 32 * (r / 2) + l)) ++ List.replicate 32 (.c 0) ++
    List.replicate 4 (.g (a4At K b + 7)) ++ List.replicate 4 (.c 24)⟩⟩

/-- Position `t`'s formed FP32 word. -/
def d4 (K b t : ℕ) : ℕ := d4At K b + 2 * t + 1

/-- Cell `s < 52` of block `q`'s quantisation at gate `base`: the running `fmaxf` of `|d|` over the block's 16
positions, `·/6`, its floor at `2⁻⁹`, the UE4M3 scale byte, its value, its reciprocal, then per position `d·(1/scale)`
and its E2M1 cast. -/
def blk4 (K b q s base : ℕ) : RGate :=
  if s < 15 then
    ⟨.fmaxf, [if s = 0 then .g (d4 K b (16 * q)) .fabs else .g (base + s - 1), .g (d4 K b (16 * q + s + 1)) .fabs]⟩
  else if s = 15 then ⟨.mul, [.g (base + 14), .c 0x3E2AAAAB]⟩
  else if s = 16 then ⟨.fmaxf, [.g (base + 15), .c 0x3B000000]⟩
  else if s = 17 then ⟨.toE4m3, [.g (base + 16)]⟩
  else if s = 18 then ⟨.e4m3ToF32, [.g (base + 17)]⟩
  else if s = 19 then ⟨.div, [.c 0x3F800000, .g (base + 18)]⟩
  else if (s - 20) % 2 = 0 then ⟨.mul, [.g (d4 K b (16 * q + (s - 20) / 2)), .g (base + 19)]⟩
  else ⟨.toE2m1, [.g (base + s - 1)]⟩

/-- The quantisation's blocks. -/
def blk4Run (K b : ℕ) : Run := Run.rep (K / 16) 52 fun q s => blk4 K b q s (bk4At K b + 52 * q)

/-- Position `t`'s E2M1 code. -/
def code4 (K b t : ℕ) : ℕ := bk4At K b + 52 * (t / 16) + 21 + 2 * (t % 16)

/-- Block `i`'s UE4M3 scale byte. -/
def scale4 (K b i : ℕ) : ℕ := bk4At K b + 52 * i + 17

/-- The `pc4` row units' runs both sides share, from gate `b`. -/
def pre4Runs (K b : ℕ) : List Run :=
  statsRuns .bf16Fabs .bf16 K b ++ [alpha4Run K b, d4Run K b, blk4Run K b]

/-- Firing `r < K/64` of a chain at gate `gt`: from `+0` (then the previous firing), the codes at `code t` and the other
side at `col t`, the first side's scale bytes at `scale i` and the other's `56`, `1.0`. -/
def chain4 (code col scale : ℕ → ℕ) (gt r : ℕ) : RGate :=
  ⟨.atom4, ((if r = 0 then Arg.c 0 else .g (gt - 1)) :: (List.range 64).map (fun l => .g (code (64 * r + l)))) ++
    (List.range 64).map (fun l => .g (col (64 * r + l))) ++ (List.range 4).map (fun l => .g (scale (4 * r + l))) ++
    List.replicate 4 (.c 56)⟩

/-- RowA's `P_A` chains: per column `q < 32` of the chain basis, the chain of `A′`'s codes against it and its BF16
cast, `P_A`'s word `q`. -/
def af4Run (K b : ℕ) : Run :=
  Run.rep 32 (K / 64 + 1) fun q s =>
    if s < K / 64 then
      chain4 (code4 K b) (fun t => 33 * K + 32 + 32 * t + q) (scale4 K b) (post4At K b + q * (K / 64 + 1) + s) s
    else ⟨.f2fpBf16, [.g (post4At K b + q * (K / 64 + 1) + s - 1)]⟩

/-- The noise' `Ẽ`: per code of `e`, `PouwETilde_v1` at `a_E`. -/
def et4Run (K b : ℕ) : Run := ⟨32, fun l => ⟨.eTilde, [.g (K + l), .g (a4At K b + 7)]⟩⟩

/-- Cell `s` of RowB's block `q`, at gate `gt`: the 32 FMAs of `Σ_s Ẽ_s·g_{s,q}` (from `+0`, `Ẽ` widened, at gates
from `et`), the chain of `B̃`'s codes against the chain basis' column `q`, its negation, the sum, and its BF16 cast,
`P_B`'s word `q`. -/
def mid4 (K b et q s gt : ℕ) : RGate :=
  if s < 32 then ⟨.fma, [.g (et + s) .bf16, .g (65 * K + 32 + 32 * s + q), if s = 0 then .c 0 else .g (gt - 1)]⟩
  else if s < 32 + K / 64 then chain4 (code4 K b) (fun t => 33 * K + 32 + 32 * t + q) (scale4 K b) gt (s - 32)
  else if s = 32 + K / 64 then ⟨.neg, [.g (gt - 1)]⟩
  else if s = 33 + K / 64 then ⟨.add, [.g (gt - 2 - K / 64), .g (gt - 1)]⟩
  else ⟨.f2fpBf16, [.g (gt - 1)]⟩

/-- RowB's `P_B` blocks, one per column `q < 32`, after `Ẽ`'s 32 gates. -/
def mid4Run (K b : ℕ) : Run :=
  Run.rep 32 (K / 64 + 35) fun q s => mid4 K b (post4At K b) q s (post4At K b + 32 + q * (K / 64 + 35) + s)

/-- `−Ẽ`. -/
def nb4Run (et : ℕ) : Run := ⟨32, fun l => ⟨.bf16Neg, [.g (et + l)]⟩⟩

@[simp] theorem Arg.forall_below_replicate (n m : ℕ) (x : Arg) :
    (∀ y ∈ List.replicate m x, y.Below n) ↔ (m = 0 ∨ x.Below n) := by
  constructor
  · intro h
    rcases m with _ | m
    · exact .inl rfl
    · exact .inr (h x (List.mem_replicate.2 ⟨by omega, rfl⟩))
  · rintro (rfl | h) y hy
    · simp at hy
    · rw [(List.mem_replicate.1 hy).2]; exact h

theorem alpha4Run_ok {K b : ℕ} (hb : K < b) : (alpha4Run K b).OK (rhoAt K b + 2) := by
  apply Run.ofList_ok
  simp [ListOK, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
  omega

theorem d4Run_ok {K b : ℕ} (hb : 65 * K + 32 ≤ b) : (d4Run K b).OK (d4At K b) := by
  intro r hr
  apply RGate.reads_lt
  simp only [d4Run] at hr ⊢
  split_ifs <;>
    simp only [List.forall_mem_cons, List.forall_mem_append, List.forall_mem_map, List.mem_range,
      Arg.forall_below_replicate, List.mem_nil_iff, false_implies, implies_true, and_true, Arg.below_g,
      Arg.below_c, OfNat.ofNat_ne_zero, false_or] <;>
    simp only [d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt] <;> and_intros <;> intros <;> omega

theorem blk4Run_ok (K b : ℕ) : (blk4Run K b).OK (bk4At K b) := by
  apply Run.rep_ok
  intro q hq s hs
  unfold blk4
  split_ifs <;>
    simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, true_and, Arg.below_g,
      Arg.below_c] <;>
    simp only [d4, bk4At] <;> omega

theorem pre4Runs_ok {K b : ℕ} (hb : 65 * K + 32 ≤ b) : RunsOK (pre4Runs K b) b := by
  refine runsOK_append (statsRuns_ok .bf16Fabs .bf16 (by omega)) (statsRuns_len .bf16Fabs .bf16 K b) ?_
  refine runsOK_cons (o' := d4At K b) (alpha4Run_ok (by omega)) (by simp [alpha4Run, Run.ofList, d4At, a4At]) <|
    runsOK_cons (o' := bk4At K b) (d4Run_ok hb) (by simp [d4Run, bk4At]) <|
    runsOK_cons (o' := post4At K b) (blk4Run_ok K b) (by simp [blk4Run, Run.rep, post4At]; omega) trivial

theorem pre4Runs_len (K b : ℕ) : b + runsLen (pre4Runs K b) = post4At K b := by
  have := statsRuns_len .bf16Fabs .bf16 K b
  simp only [pre4Runs, runsLen_append] at this ⊢
  simp [runsLen, alpha4Run, d4Run, blk4Run, Run.rep, Run.ofList, post4At, bk4At, d4At, a4At]
  omega

theorem chain4_ok {K o : ℕ} {code col scale : ℕ → ℕ} {gt r : ℕ} (hr : r < K / 64)
    (hcode : ∀ t < 64 * (K / 64), code t < o) (hcol : ∀ t < 64 * (K / 64), col t < o)
    (hscale : ∀ i < 4 * (K / 64), scale i < o) (ho : o ≤ gt)
    (hgt : r ≤ gt) : ∀ x ∈ (chain4 code col scale gt r).args, x.Below gt := by
  unfold chain4
  split_ifs <;>
    simp only [List.forall_mem_cons, List.forall_mem_append, List.forall_mem_map, List.mem_range,
      Arg.forall_below_replicate, Arg.below_g, Arg.below_c, OfNat.ofNat_ne_zero, false_or, and_true, true_and] <;>
    and_intros <;> intros <;>
    (first
      | omega
      | exact Nat.lt_of_lt_of_le (hcode _ (by omega)) ho
      | exact Nat.lt_of_lt_of_le (hcol _ (by omega)) ho
      | exact Nat.lt_of_lt_of_le (hscale _ (by omega)) ho)

theorem code4_lt {K b t : ℕ} (ht : t < 16 * (K / 16)) : code4 K b t < post4At K b := by
  simp only [code4, post4At]; omega

theorem scale4_lt {K b i : ℕ} (hi : i < K / 16) : scale4 K b i < post4At K b := by
  simp only [scale4, post4At]; omega

theorem af4Run_ok {K b : ℕ} (hb : 65 * K + 32 ≤ b) : (af4Run K b).OK (post4At K b) := by
  apply Run.rep_ok
  intro q hq s hs
  generalize q * (K / 64 + 1) = Q
  split_ifs with h
  · apply chain4_ok h (o := post4At K b) (fun t ht => code4_lt (by omega)) _ (fun i hi => scale4_lt (by omega))
      (by omega) (by omega)
    intro t ht
    simp only [post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
    omega
  · simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, Arg.below_g]
    simp only [post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
    omega

theorem et4Run_ok (K b : ℕ) {o : ℕ} (ho : post4At K b ≤ o) : (et4Run K b).OK o := by
  intro l hl
  apply RGate.reads_lt
  simp only [et4Run] at hl ⊢
  simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, Arg.below_g]
  simp only [post4At, bk4At, d4At] at ho
  omega

theorem mid4Run_ok {K b : ℕ} (hb : 65 * K + 1056 ≤ b) : (mid4Run K b).OK (post4At K b + 32) := by
  apply Run.rep_ok
  intro q hq s hs
  generalize q * (K / 64 + 35) = Q
  unfold mid4
  by_cases h1 : s < 32
  · simp only [h1, ↓reduceIte]
    simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, true_and, Arg.below_g,
      Arg.below_ite, Arg.below_c]
    simp only [post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
    omega
  simp only [h1, ↓reduceIte]
  by_cases h2 : s < 32 + K / 64
  · simp only [h2, ↓reduceIte]
    apply chain4_ok (K := K) (by omega) (o := post4At K b) (fun t ht => code4_lt (by omega)) _
      (fun i hi => scale4_lt (by omega))
      (by omega) (by omega)
    intro t ht
    simp only [post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
    omega
  simp only [h2, ↓reduceIte]
  split_ifs <;>
    simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, Arg.below_g] <;>
    simp only [post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt] <;> omega

theorem nb4Run_ok {et o : ℕ} (ho : et + 32 ≤ o) : (nb4Run et).OK o := by
  intro l hl
  apply RGate.reads_lt
  simp only [nb4Run] at hl ⊢
  simp only [List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true, and_true, Arg.below_g]
  omega

/-- **`Pc4RowA_v1{K}`**: its outputs are `A′`'s `K` codes and `K/16` scale bytes, the noise' 32 `Ẽ` words, `P_A`'s 32
and the flag. -/
def row4A (K : ℕ) : Rows where
  nIn := rowIn K false
  inW a := if a < K then 16 else 4
  runs := pre4Runs K (rowIn K false) ++ [af4Run K (rowIn K false), et4Run K (rowIn K false),
    okwRun (a4At K (rowIn K false) + 12)]
  outs := (List.range K).map (code4 K (rowIn K false)) ++ (List.range (K / 16)).map (scale4 K (rowIn K false)) ++
    (List.range 32).map (post4At K (rowIn K false) + 32 * (K / 64 + 1) + ·) ++
    (List.range 32).map (fun l => post4At K (rowIn K false) + l * (K / 64 + 1) + K / 64) ++
    [post4At K (rowIn K false) + 32 * (K / 64 + 1) + 32]
  ok := by
    have hb : 65 * K + 32 ≤ rowIn K false := by simp [rowIn]
    refine runsOK_append (pre4Runs_ok hb) (pre4Runs_len K _) ?_
    refine runsOK_cons (af4Run_ok hb) rfl <|
      runsOK_cons (et4Run_ok K _ (Nat.le_add_right _ _)) rfl <| runsOK_cons (okwRun_ok ?_) rfl trivial
    simp [af4Run, Run.rep, et4Run, post4At, bk4At, d4At]
    omega

/-- **`Pc4RowB_v1{K}`**: its outputs are `B̃`'s `K` codes and `K/16` scale bytes, `P_B`'s 32 words, the noise' 32
`−Ẽ` words and the flag. -/
def row4B (K : ℕ) : Rows where
  nIn := rowIn K true
  inW a := if a < K then 16 else if a < 65 * K + 32 then 4 else 32
  runs := pre4Runs K (rowIn K true) ++ [et4Run K (rowIn K true), mid4Run K (rowIn K true),
    nb4Run (post4At K (rowIn K true)), okwRun (a4At K (rowIn K true) + 12)]
  outs := (List.range K).map (code4 K (rowIn K true)) ++ (List.range (K / 16)).map (scale4 K (rowIn K true)) ++
    (List.range 32).map (fun q => post4At K (rowIn K true) + 32 + q * (K / 64 + 35) + K / 64 + 34) ++
    (List.range 32).map (post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + ·) ++
    [post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + 32]
  ok := by
    have hb : 65 * K + 1056 ≤ rowIn K true := by simp [rowIn]
    refine runsOK_append (pre4Runs_ok (by omega)) (pre4Runs_len K _) ?_
    refine runsOK_cons (o' := post4At K (rowIn K true) + 32) (et4Run_ok K _ le_rfl) (by simp [et4Run]) <|
      runsOK_cons (mid4Run_ok hb) rfl <| runsOK_cons (nb4Run_ok (Nat.le_add_right _ _)) rfl <|
      runsOK_cons (okwRun_ok ?_) rfl trivial
    simp [mid4Run, Run.rep, nb4Run, post4At, bk4At, d4At]
    omega

end Pouw.PearlC.RowCircuit
