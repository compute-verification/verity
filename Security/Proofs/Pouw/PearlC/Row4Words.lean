import Proofs.Pouw.PearlC.RowCircuit
import Proofs.Pouw.PearlC.Tile4Circuit

/-!
# Pearl-C4's row units compute their codes, scale bytes, `P_A` and `P_B` (proofs)

For every `ops` (core's primitives as a parameter), on the runs `pc4`'s two row units share (`pre4Runs`) and in a
transcript whose one unit is correct:

* **the forming** (`pre4_d`, `d4W`): position `t`'s formed word is one NVF4 firing (`form4`) from `α·x_t` over `e`'s
  32 codes at `a_E = e4m3(β)` against the form basis' row `t` at `24`, each padded with 32 zeros;
* **the quantisation** (`pre4_max`, `pre4_scale`, `pre4_code`): block `q`'s running max of `|d|` (`blkMax4`), its
  scale byte `e4m3(max(amax·f32(1/6), 2⁻⁹))` (`scale4W`, `scale4V`) and each position's E2M1 code
  `e2m1(d·(1/scale))` (`code4W`, `code4V`), so RowA's and RowB's code and scale outputs (`row4A_code`,
  `row4A_scale`, `row4B_code`, `row4B_scale`);
* **the chains** (`Chain4Holds`, `chain4_atom`, `chain4_word`): a row chain of `n` NVF4 firings from `+0` over codes,
  the other side and the first side's scale bytes, the other's at `1.0` (`0x38`), writes `cTilde sm120Nvf4`'s word
  wherever it is `some` (`nvf4Word_of_step`), so RowA's `P_A` word `q` is that word over `A′`'s row and the chain
  basis' column `q`, cast to BF16 (`row4A_pa`), and RowB's `P_B` word `q` is the BF16 cast of `Σ_s Ẽ_s·g_{s,q}` (the
  FMAs from `+0` of the widened `Ẽ` against `G`'s column `q`, `pbAcc`) plus the negated chain word (`row4B_pb`).

These are word formulas on `ops` and on the units' own inputs. They are not yet tied to `DeviceFp4`'s `formRow`, whose
forming is on values: that needs core's primitives to agree with the protocol's word functions where both are defined,
as `pc8`'s `Pc8OpsAgree` does.
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

variable (ops : RowOps)

/-! ## The words -/

/-- `pc4`'s formed word: `α·x` (`x` a BF16 word, widened), then one NVF4 firing from it over `e`'s 32 codes at the
byte `aE` against `f`'s 32 codes at `24` (`2⁻⁴`), each operand padded with 32 zeros. -/
def form4 (α x aE : ℕ) (e f : ℕ → ℕ) : ℕ :=
  Prim.eval ops .atom4 ((Prim.eval ops .mul [α, ops.bf16ToF32 x] :: (List.range 32).map e) ++
    List.replicate 32 0 ++ (List.range 32).map f ++ List.replicate 32 0 ++ List.replicate 4 aE ++ List.replicate 4 24)

/-- A block's running max of `|d_0|, …, |d_{s+1}|` by `F32Fmaxf_v1`. -/
def blkMax4 (d : ℕ → ℕ) : ℕ → ℕ
  | 0 => Prim.eval ops .fmaxf [ops.f32Fabs (d 0), ops.f32Fabs (d 1)]
  | s + 1 => Prim.eval ops .fmaxf [blkMax4 d s, ops.f32Fabs (d (s + 2))]

/-- A block's UE4M3 scale byte from its max `|d|`: `e4m3(max(amax·f32(1/6), 2⁻⁹))`. -/
def scale4W (amax : ℕ) : ℕ :=
  Prim.eval ops .toE4m3 [Prim.eval ops .fmaxf [Prim.eval ops .mul [amax, 0x3E2AAAAB], 0x3B000000]]

/-- A position's E2M1 code from its formed word `d` and its block's scale byte `s`: `e2m1(d·(1/s))`. -/
def code4W (d s : ℕ) : ℕ :=
  Prim.eval ops .toE2m1 [Prim.eval ops .mul [d, Prim.eval ops .div [0x3F800000, Prim.eval ops .e4m3ToF32 [s]]]]

/-- Position `t`'s formed word on a row unit's inputs `x` (the row, `e`'s codes, the form basis). -/
def d4W (x : ℕ → ℕ) (K t : ℕ) : ℕ :=
  form4 ops (alpha4 ops x K) (x t) (Prim.eval ops .toE4m3 [beta4 ops x K]) (fun l => x (K + l))
    (fun l => x (K + 32 + 32 * t + l))

/-- Block `q`'s scale byte on a row unit's inputs `x`. -/
def scale4V (x : ℕ → ℕ) (K q : ℕ) : ℕ := scale4W ops (blkMax4 ops (fun s => d4W ops x K (16 * q + s)) 14)

/-- Position `t`'s E2M1 code on a row unit's inputs `x`. -/
def code4V (x : ℕ → ℕ) (K t : ℕ) : ℕ := code4W ops (d4W ops x K t) (scale4V ops x K (t / 16))

/-! ## `pc4`'s shared runs -/

section Pre4

variable {ops} {V : ℕ → ℕ} {K b : ℕ}

theorem pre4_runs (h : RunsHold ops V (pre4Runs K b) b) :
    RunsHold ops V (statsRuns .bf16Fabs .bf16 K b) b ∧ RunHolds ops V (alpha4Run K b) (a4At K b) ∧
      RunHolds ops V (d4Run K b) (d4At K b) ∧ RunHolds ops V (blk4Run K b) (bk4At K b) := by
  obtain ⟨hs, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  obtain ⟨ha, hd, hb, -⟩ := hr
  exact ⟨hs, ha, hd, hb⟩

/-- **Position `t`'s formed word**: `d4W` of the values of the row's inputs. -/
theorem pre4_d (hK : 8 ≤ K) (h : RunsHold ops V (pre4Runs K b) b) : ∀ t < K, V (d4 K b t) = d4W ops V K t := by
  intro t ht
  obtain ⟨-, -, hd, -⟩ := pre4_runs h
  obtain ⟨hα, -, haE⟩ := pre4_alpha hK h
  have g0 := hd (2 * t) (by simp [d4Run]; omega)
  have g1 := hd (2 * t + 1) (by simp [d4Run]; omega)
  have e0 : 2 * t % 2 = 0 := by omega
  have e1 : (2 * t + 1) % 2 = 1 := by omega
  have d0 : 2 * t / 2 = t := by omega
  have d1 : (2 * t + 1) / 2 = t := by omega
  simp only [d4Run, e0, e1, d0, d1, ↓reduceIte, one_ne_zero, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval] at g0 g1
  rw [show d4 K b t = d4At K b + (2 * t + 1) by unfold d4; omega, g1,
    show d4At K b + (2 * t + 1) - 1 = d4At K b + 2 * t by omega]
  simp only [List.map_append, List.map_cons, List.map_map, List.map_replicate, Arg.eval, Wire.eval]
  rw [g0, hα, haE]
  rfl

theorem pre4_blkGate (h : RunsHold ops V (pre4Runs K b) b) {q s : ℕ} (hq : q < K / 16) (hs : s < 52) :
    V (bk4At K b + 52 * q + s) = (blk4 K b q s (bk4At K b + 52 * q)).prim.eval ops
      ((blk4 K b q s (bk4At K b + 52 * q)).args.map (Arg.eval ops V)) := by
  obtain ⟨-, -, -, hb⟩ := pre4_runs h
  have hr : 52 * q + s < (blk4Run K b).n := by simp only [blk4Run, Run.rep]; omega
  have hd : (52 * q + s) / 52 = q := by omega
  have hm : (52 * q + s) % 52 = s := by omega
  have := hb _ hr
  simp only [blk4Run, Run.rep, hd, hm] at this
  rw [← Nat.add_assoc] at this
  exact this

/-- **A block's running max**: gate `s < 15` of block `q` is `blkMax4` of the block's formed words. -/
theorem pre4_max (h : RunsHold ops V (pre4Runs K b) b) {q : ℕ} (hq : q < K / 16) :
    ∀ s < 15, V (bk4At K b + 52 * q + s) = blkMax4 ops (fun s => V (d4 K b (16 * q + s))) s := by
  intro s
  induction s with
  | zero =>
    intro _
    rw [pre4_blkGate h hq (by omega)]
    simp only [blk4, show (0 : ℕ) < 15 by omega, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval,
      blkMax4, Nat.add_zero]
  | succ s ih =>
    intro hs
    rw [pre4_blkGate h hq (by omega)]
    simp only [blk4, hs, ↓reduceIte, Nat.add_one_ne_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval,
      blkMax4]
    rw [show bk4At K b + 52 * q + (s + 1) - 1 = bk4At K b + 52 * q + s by omega, ih (by omega),
      show 16 * q + (s + 1) + 1 = 16 * q + (s + 2) by omega]

/-- **A block's scale byte**: `scale4W` of its running max. -/
theorem pre4_scaleW (h : RunsHold ops V (pre4Runs K b) b) {q : ℕ} (hq : q < K / 16) :
    V (scale4 K b q) = scale4W ops (blkMax4 ops (fun s => V (d4 K b (16 * q + s))) 14) := by
  have g15 := pre4_blkGate h hq (s := 15) (by omega)
  have g16 := pre4_blkGate h hq (s := 16) (by omega)
  have g17 := pre4_blkGate h hq (s := 17) (by omega)
  simp only [blk4, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
    show ¬ (15 : ℕ) < 15 by omega, show ¬ (16 : ℕ) < 15 by omega, show ¬ (17 : ℕ) < 15 by omega,
    show (16 : ℕ) ≠ 15 by omega, show (17 : ℕ) ≠ 15 by omega, show (17 : ℕ) ≠ 16 by omega] at g15 g16 g17
  unfold scale4 scale4W
  rw [g17, g16, g15, pre4_max h hq 14 (by omega)]

/-- **A position's code**: `code4W` of its formed word and its block's scale byte. -/
theorem pre4_codeW (h : RunsHold ops V (pre4Runs K b) b) {t : ℕ} (ht : t < 16 * (K / 16)) :
    V (code4 K b t) = code4W ops (V (d4 K b t)) (V (scale4 K b (t / 16))) := by
  have hq : t / 16 < K / 16 := by omega
  have g18 := pre4_blkGate h hq (s := 18) (by omega)
  have g19 := pre4_blkGate h hq (s := 19) (by omega)
  have gm := pre4_blkGate h hq (s := 20 + 2 * (t % 16)) (by omega)
  have gc := pre4_blkGate h hq (s := 21 + 2 * (t % 16)) (by omega)
  simp only [blk4, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
    show ¬ (18 : ℕ) < 15 by omega, show ¬ (19 : ℕ) < 15 by omega, show (18 : ℕ) ≠ 15 by omega,
    show (19 : ℕ) ≠ 15 by omega, show (18 : ℕ) ≠ 16 by omega, show (19 : ℕ) ≠ 16 by omega,
    show (18 : ℕ) ≠ 17 by omega, show (19 : ℕ) ≠ 17 by omega, show (19 : ℕ) ≠ 18 by omega] at g18 g19
  simp only [blk4, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
    show ¬ 20 + 2 * (t % 16) < 15 by omega, show 20 + 2 * (t % 16) ≠ 15 by omega,
    show 20 + 2 * (t % 16) ≠ 16 by omega, show 20 + 2 * (t % 16) ≠ 17 by omega,
    show 20 + 2 * (t % 16) ≠ 18 by omega, show 20 + 2 * (t % 16) ≠ 19 by omega,
    show (20 + 2 * (t % 16) - 20) % 2 = 0 by omega, show (20 + 2 * (t % 16) - 20) / 2 = t % 16 by omega,
    show 16 * (t / 16) + t % 16 = t by omega] at gm
  simp only [blk4, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
    show ¬ 21 + 2 * (t % 16) < 15 by omega, show 21 + 2 * (t % 16) ≠ 15 by omega,
    show 21 + 2 * (t % 16) ≠ 16 by omega, show 21 + 2 * (t % 16) ≠ 17 by omega,
    show 21 + 2 * (t % 16) ≠ 18 by omega, show 21 + 2 * (t % 16) ≠ 19 by omega,
    show ¬ (21 + 2 * (t % 16) - 20) % 2 = 0 by omega] at gc
  unfold code4 code4W
  rw [show bk4At K b + 52 * (t / 16) + 21 + 2 * (t % 16) = bk4At K b + 52 * (t / 16) + (21 + 2 * (t % 16)) by omega,
    gc, show bk4At K b + 52 * (t / 16) + (21 + 2 * (t % 16)) - 1 = bk4At K b + 52 * (t / 16) + (20 + 2 * (t % 16))
      by omega, gm, g19, g18]
  rfl

theorem blkMax4_congr {d d' : ℕ → ℕ} : ∀ n, (∀ s ≤ n + 1, d s = d' s) → blkMax4 ops d n = blkMax4 ops d' n
  | 0, h => by simp only [blkMax4, h 0 (by omega), h 1 (by omega)]
  | n + 1, h => by
    simp only [blkMax4, h (n + 2) (by omega), blkMax4_congr n fun s hs => h s (by omega)]

/-- **Block `q`'s scale byte** on values meeting `pc4`'s shared runs: `scale4V` of the row's inputs' values. -/
theorem pre4_scale (hK : 8 ≤ K) (h : RunsHold ops V (pre4Runs K b) b) {q : ℕ} (hq : q < K / 16) :
    V (scale4 K b q) = scale4V ops V K q := by
  rw [pre4_scaleW h hq, scale4V,
    blkMax4_congr 14 fun s hs => pre4_d hK h (16 * q + s) (by omega)]

/-- **Position `t`'s code** on values meeting `pc4`'s shared runs: `code4V` of the row's inputs' values. -/
theorem pre4_code (hK : 8 ≤ K) (h : RunsHold ops V (pre4Runs K b) b) {t : ℕ} (ht : t < 16 * (K / 16)) :
    V (code4 K b t) = code4V ops V K t := by
  rw [pre4_codeW h ht, pre4_d hK h t (by omega), pre4_scale hK h (by omega), code4V]

end Pre4

/-! ## `pc4`'s codes and scale bytes -/

section Rows4W

variable {ops}

theorem form4_congr {α x aE : ℕ} {e e' f f' : ℕ → ℕ} (he : ∀ l < 32, e l = e' l) (hf : ∀ l < 32, f l = f' l) :
    form4 ops α x aE e f = form4 ops α x aE e' f' := by
  unfold form4
  rw [List.map_congr_left fun l hl => he l (List.mem_range.1 hl),
    List.map_congr_left fun l hl => hf l (List.mem_range.1 hl)]

theorem d4W_congr {x y : ℕ → ℕ} {K t : ℕ} (hK : 8 ≤ K) (ht : t < K) (h : ∀ l < 33 * K + 32, x l = y l) :
    d4W ops x K t = d4W ops y K t := by
  unfold d4W
  rw [alpha4_congr hK fun l hl => h l (by omega), beta4_congr hK fun l hl => h l (by omega), h t (by omega)]
  exact form4_congr (fun l hl => h _ (by omega)) fun l hl => h _ (by omega)

theorem scale4V_congr {x y : ℕ → ℕ} {K q : ℕ} (hK : 8 ≤ K) (hq : q < K / 16) (h : ∀ l < 33 * K + 32, x l = y l) :
    scale4V ops x K q = scale4V ops y K q := by
  unfold scale4V
  rw [blkMax4_congr 14 fun s hs => d4W_congr hK (by omega) h]

theorem code4V_congr {x y : ℕ → ℕ} {K t : ℕ} (hK : 8 ≤ K) (ht : t < 16 * (K / 16))
    (h : ∀ l < 33 * K + 32, x l = y l) : code4V ops x K t = code4V ops y K t := by
  unfold code4V
  rw [d4W_congr hK (by omega) h, scale4V_congr hK (by omega) h]

/-- **A `pc4` row unit computes its codes**: a table whose runs start with `pc4`'s shared runs, in a transcript whose
one unit is correct, holds on position `t`'s code output `code4V` of the row's inputs. -/
theorem rows4_code (T : Rows) {K : ℕ} (hK : 8 ≤ K) (hb : 65 * K + 32 ≤ T.nIn) {rest : List Run}
    (hr : T.runs = pre4Runs K T.nIn ++ rest) (X : (T.spec ops).toCircuit.Transcript)
    (hX : (T.spec ops).part.Correct X 0) {t : ℕ} (ht : t < 16 * (K / 16)) (hout : code4 K T.nIn t ∈ T.outs) :
    val X (code4 K T.nIn t) = code4V ops (val X) K t := by
  have hlen := pre4Runs_len K T.nIn
  have hlt : code4 K T.nIn t < (T.spec ops).N := by
    show code4 K T.nIn t < T.nIn + runsLen T.runs
    rw [hr, runsLen_append]
    have := code4_lt (b := T.nIn) ht
    omega
  rw [Spec.out_eq (T.spec ops) X hX (List.mem_toFinset.2 hout) hlt]
  have hh := Rows.holds ops T X
  rw [hr] at hh
  rw [pre4_code hK (runsHold_append ops _ hh).1 ht]
  exact code4V_congr hK ht fun l hl => Rows.val_input ops T X (by omega)

/-- **A `pc4` row unit computes its scale bytes**: as `rows4_code`, block `q`'s scale output is `scale4V`. -/
theorem rows4_scale (T : Rows) {K : ℕ} (hK : 8 ≤ K) (hb : 65 * K + 32 ≤ T.nIn) {rest : List Run}
    (hr : T.runs = pre4Runs K T.nIn ++ rest) (X : (T.spec ops).toCircuit.Transcript)
    (hX : (T.spec ops).part.Correct X 0) {q : ℕ} (hq : q < K / 16) (hout : scale4 K T.nIn q ∈ T.outs) :
    val X (scale4 K T.nIn q) = scale4V ops (val X) K q := by
  have hlen := pre4Runs_len K T.nIn
  have hlt : scale4 K T.nIn q < (T.spec ops).N := by
    show scale4 K T.nIn q < T.nIn + runsLen T.runs
    rw [hr, runsLen_append]
    have := scale4_lt (b := T.nIn) hq
    omega
  rw [Spec.out_eq (T.spec ops) X hX (List.mem_toFinset.2 hout) hlt]
  have hh := Rows.holds ops T X
  rw [hr] at hh
  rw [pre4_scale hK (runsHold_append ops _ hh).1 hq]
  exact scale4V_congr hK hq fun l hl => Rows.val_input ops T X (by omega)

/-- **RowA computes `A′`'s codes** (`rows4_code` at `Pc4RowA_v1{K}`). -/
theorem row4A_code {K : ℕ} (hK : 8 ≤ K) (X : ((row4A K).spec ops).toCircuit.Transcript)
    (hX : ((row4A K).spec ops).part.Correct X 0) {t : ℕ} (ht : t < 16 * (K / 16)) :
    val X (code4 K (rowIn K false) t) = code4V ops (val X) K t :=
  rows4_code (row4A K) hK (by simp [row4A, rowIn]) rfl X hX ht
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_map.2 ⟨t, List.mem_range.2 (by omega), rfl⟩)))))

/-- **RowA computes `A′`'s scale bytes** (`rows4_scale` at `Pc4RowA_v1{K}`). -/
theorem row4A_scale {K : ℕ} (hK : 8 ≤ K) (X : ((row4A K).spec ops).toCircuit.Transcript)
    (hX : ((row4A K).spec ops).part.Correct X 0) {q : ℕ} (hq : q < K / 16) :
    val X (scale4 K (rowIn K false) q) = scale4V ops (val X) K q :=
  rows4_scale (row4A K) hK (by simp [row4A, rowIn]) rfl X hX hq
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨q, List.mem_range.2 hq, rfl⟩)))))

/-- **RowB computes `B̃`'s codes** (`rows4_code` at `Pc4RowB_v1{K}`). -/
theorem row4B_code {K : ℕ} (hK : 8 ≤ K) (X : ((row4B K).spec ops).toCircuit.Transcript)
    (hX : ((row4B K).spec ops).part.Correct X 0) {t : ℕ} (ht : t < 16 * (K / 16)) :
    val X (code4 K (rowIn K true) t) = code4V ops (val X) K t :=
  rows4_code (row4B K) hK (by simp [row4B, rowIn]) rfl X hX ht
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_map.2 ⟨t, List.mem_range.2 (by omega), rfl⟩)))))

/-- **RowB computes `B̃`'s scale bytes** (`rows4_scale` at `Pc4RowB_v1{K}`). -/
theorem row4B_scale {K : ℕ} (hK : 8 ≤ K) (X : ((row4B K).spec ops).toCircuit.Transcript)
    (hX : ((row4B K).spec ops).part.Correct X 0) {q : ℕ} (hq : q < K / 16) :
    val X (scale4 K (rowIn K true) q) = scale4V ops (val X) K q :=
  rows4_scale (row4B K) hK (by simp [row4B, rowIn]) rfl X hX hq
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨q, List.mem_range.2 hq, rfl⟩)))))

end Rows4W

/-! ## The chains -/

/-- A chain firing's operands after its accumulator: 64 codes, the other side's 64, the first side's 4 scale bytes
and the other's 4 at `1.0`. -/
def chainArgs4 (f h u : ℕ → ℕ) : List ℕ :=
  (List.range 64).map f ++ (List.range 64).map h ++ (List.range 4).map u ++ List.replicate 4 56

theorem chainArgs4_getD (f h u : ℕ → ℕ) :
    (∀ i < 64, (chainArgs4 f h u).getD i 0 = f i ∧ (chainArgs4 f h u).getD (64 + i) 0 = h i) ∧
      ∀ i < 4, (chainArgs4 f h u).getD (128 + i) 0 = u i ∧ (chainArgs4 f h u).getD (132 + i) 0 = 56 := by
  unfold chainArgs4
  refine ⟨fun i hi => ⟨?_, ?_⟩, fun i hi => ⟨?_, ?_⟩⟩
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append, hi]
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append, hi]
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append, show ¬ 128 + i < 64 by omega,
      show ¬ 128 + i - 64 < 64 by omega, show 128 + i - 64 - 64 = i by omega, hi]
  · simp only [List.getD_eq_getElem?_getD, List.getElem?_append, List.length_append, List.length_map,
      List.length_range, show ¬ 132 + i < 64 + 64 + 4 by omega,
      show 132 + i - (64 + 64 + 4) = i by omega, ↓reduceIte, List.getElem?_replicate, hi, Option.getD_some]

section Chain4

variable {ops} {V : ℕ → ℕ} {cd cl sc : ℕ → ℕ} {o n : ℕ}

/-- Values meeting `n` firings of a chain from gate `o` (`chain4`): the codes at `cd t`, the other side at `cl t` and
the first side's scale bytes at `sc i`. -/
def Chain4Holds (V : ℕ → ℕ) (cd cl sc : ℕ → ℕ) (o n : ℕ) : Prop :=
  ∀ r < n, V (o + r) =
    (chain4 cd cl sc (o + r) r).prim.eval ops ((chain4 cd cl sc (o + r) r).args.map (Arg.eval ops V))

/-- **A chain's firing**: from `+0` (then the previous firing's word), on the codes, the other side and the scale
bytes it reads, the other side's at `0x38`. -/
theorem chain4_atom (h : Chain4Holds (ops := ops) V cd cl sc o n) {r : ℕ} (hr : r < n) :
    V (o + r) = nvf4Word (if r = 0 then 0 else V (o + r - 1))
      (List.ofFn fun l : Fin 64 => nib (V (cd (64 * r + l)))) (List.ofFn fun l : Fin 64 => nib (V (cl (64 * r + l))))
      (List.ofFn fun l : Fin 4 => byte (V (sc (4 * r + l)))) (List.ofFn fun _ : Fin 4 => byte 56) % 2 ^ 32 := by
  rw [h r hr]
  have hc : Arg.eval ops V (if r = 0 then Arg.c 0 else .g (o + r - 1)) = if r = 0 then 0 else V (o + r - 1) := by
    split_ifs <;> rfl
  have hl : ((chain4 cd cl sc (o + r) r).args.map (Arg.eval ops V)) = (if r = 0 then 0 else V (o + r - 1)) ::
      chainArgs4 (fun l => V (cd (64 * r + l))) (fun l => V (cl (64 * r + l))) (fun l => V (sc (4 * r + l))) := by
    simp only [chain4, List.map_cons, List.map_append, List.map_map, List.map_replicate, List.cons_append, hc,
      chainArgs4]
    rfl
  rw [hl]
  show atom4F false _ % 2 ^ 32 = _
  obtain ⟨q1, q2⟩ := chainArgs4_getD (fun l => V (cd (64 * r + l))) (fun l => V (cl (64 * r + l)))
    (fun l => V (sc (4 * r + l)))
  simp only [atom4F, Bool.false_eq_true, ↓reduceIte, List.headD_cons, List.tail_cons]
  set L := chainArgs4 (fun l => V (cd (64 * r + l))) (fun l => V (cl (64 * r + l))) (fun l => V (sc (4 * r + l)))
  have a1 : (List.ofFn fun l : Fin 64 => nib (L.getD l 0)) = List.ofFn fun l : Fin 64 => nib (V (cd (64 * r + l))) :=
    congrArg List.ofFn (funext fun l => by rw [(q1 l l.isLt).1])
  have a2 : (List.ofFn fun l : Fin 64 => nib (L.getD (64 + l) 0)) =
      List.ofFn fun l : Fin 64 => nib (V (cl (64 * r + l))) :=
    congrArg List.ofFn (funext fun l => by rw [(q1 l l.isLt).2])
  have a3 : (List.ofFn fun l : Fin 4 => byte (L.getD (128 + l) 0)) =
      List.ofFn fun l : Fin 4 => byte (V (sc (4 * r + l))) :=
    congrArg List.ofFn (funext fun l => by rw [(q2 l l.isLt).1])
  have a4 : (List.ofFn fun l : Fin 4 => byte (L.getD (132 + l) 0)) = List.ofFn fun _ : Fin 4 => byte 56 :=
    congrArg List.ofFn (funext fun l => by rw [(q2 l l.isLt).2])
  rw [a1, a2, a3, a4]

theorem onesScales_window {n t : ℕ} (ht : t < n) :
    ((onesScales sm120Nvf4 (64 * n)).drop (4 * t)).take 4 = List.ofFn fun _ : Fin 4 => byte 56 := by
  unfold onesScales
  rw [show 64 * n / sm120Nvf4.block = 4 * n by show 64 * n / 16 = 4 * n; omega, List.drop_replicate,
    List.take_replicate, show min 4 (4 * n - 4 * t) = 4 by omega, List.ofFn_const]
  rfl

/-- **A row's chain**: its `n` firings from `+0` write `cTilde sm120Nvf4`'s word over the codes `x`, the other side `y`
and the scale bytes `sx`, the other side's at `1.0` (`onesScales`), wherever it is `some`. -/
theorem chain4_word (h : Chain4Holds (ops := ops) V cd cl sc o n) (hn : 0 < n) (x y : List Nib)
    (sx : List (Fin 256)) (hx : x.length = 64 * n) (hA : ∀ t < 64 * n, V (cd t) = (x.getD t 0).val)
    (hB : ∀ t < 64 * n, V (cl t) = (y.getD t 0).val) (hS : ∀ i < 4 * n, V (sc i) = (sx.getD i 0).val) {c : ℕ}
    (hc : cTilde sm120Nvf4 x y sx (onesScales sm120Nvf4 (64 * n)) = some c) : V (o + n - 1) = c := by
  have hk : sm120Nvf4.k = 64 := rfl
  have hS4 : sm120Nvf4.scales = 4 := rfl
  unfold cTilde Fp4Atom.chain at hc
  split_ifs at hc with hl
  obtain ⟨hyx, -, hsx, -⟩ := hl
  rw [hx, hk, hS4, show 64 * n / 64 = n by omega] at hsx
  rw [hx] at hyx
  rw [hx, hk, show 64 * n / 64 = n by omega] at hc
  have key : ∀ m t c0 w, t + m = n → (if t = 0 then 0 else V (o + t - 1)) = c0 →
      sm120Nvf4.chainN m c0 (x.drop (64 * t)) (y.drop (64 * t)) (sx.drop (4 * t))
        ((onesScales sm120Nvf4 (64 * n)).drop (4 * t)) = some w →
      V (o + n - 1) = w := by
    intro m
    induction m with
    | zero =>
      intro t c0 w ht hc0 h'
      simp only [Fp4Atom.chainN, Option.some.injEq] at h'
      subst h'
      obtain rfl : t = n := by omega
      rwa [ite_eq_right (by omega)] at hc0
    | succ m ih =>
      intro t c0 w ht hc0 h'
      simp only [Fp4Atom.chainN] at h'
      obtain ⟨d, hd, h'⟩ := Option.bind_eq_some_iff.1 h'
      rw [hk, hS4] at hd h'
      have e1 : (List.ofFn fun l : Fin 64 => nib (V (cd (64 * t + l)))) = (x.drop (64 * t)).take 64 := by
        rw [← ofFn_getD _ _ (by omega)]
        congr 1; funext l
        rw [hA _ (by omega), nib_val]
      have e2 : (List.ofFn fun l : Fin 64 => nib (V (cl (64 * t + l)))) = (y.drop (64 * t)).take 64 := by
        rw [← ofFn_getD _ _ (by omega)]
        congr 1; funext l
        rw [hB _ (by omega), nib_val]
      have e3 : (List.ofFn fun l : Fin 4 => byte (V (sc (4 * t + l)))) = (sx.drop (4 * t)).take 4 := by
        rw [← ofFn_getD _ _ (by omega)]
        congr 1; funext l
        rw [hS _ (by omega), byte_val]
      have hv : V (o + t) = d := by
        rw [chain4_atom h (by omega), hc0, e1, e2, e3, ← onesScales_window (n := n) (t := t) (by omega), nvf4Word_of_step hd,
          Nat.mod_eq_of_lt (step4_lt hd)]
      refine ih (t + 1) d w (by omega) ?_ ?_
      · rw [ite_eq_right (by omega), show o + (t + 1) - 1 = o + t by omega, hv]
      · rw [List.drop_drop, List.drop_drop, List.drop_drop, List.drop_drop,
          show 64 * t + 64 = 64 * (t + 1) by ring, show 4 * t + 4 = 4 * (t + 1) by ring] at h'
        exact h'
  exact key n 0 0 c (by omega) rfl (by simpa using hc)

end Chain4

/-! ## `P_A` and `P_B` -/

section Sides4

variable {ops}

/-- **RowA computes `P_A`'s words**: at depth `K = 64·nw`, a transcript correct on RowA whose code and scale outputs
hold a row `x` of `A′` and whose chain basis' column `q` holds `y` has on `P_A`'s word `q` the BF16 cast of
`cTilde sm120Nvf4`'s word over `x` and `y`, `y` at `1.0`, wherever it is `some`. -/
theorem row4A_pa {K nw : ℕ} (hnw : 0 < nw) (hK : K = 64 * nw) (X : ((row4A K).spec ops).toCircuit.Transcript)
    (hX : ((row4A K).spec ops).part.Correct X 0) (x : Row4) (hx : x.1.length = K)
    (hA : ∀ t < K, val X (code4 K (rowIn K false) t) = (x.1.getD t 0).val)
    (hS : ∀ i < K / 16, val X (scale4 K (rowIn K false) i) = (x.2.getD i 0).val) {q : ℕ} (hq : q < 32)
    (y : List Nib) (hB : ∀ t < K, val X (33 * K + 32 + 32 * t + q) = (y.getD t 0).val) {c : ℕ}
    (hc : cTilde sm120Nvf4 x.1 y x.2 (onesScales sm120Nvf4 K) = some c) :
    val X (post4At K (rowIn K false) + q * (K / 64 + 1) + K / 64) = Prim.eval ops .f2fpBf16 [c] := by
  have hKn : K / 64 = nw := by omega
  have hlen := pre4Runs_len K (rowIn K false)
  have hN : (row4A K).N = post4At K (rowIn K false) + 32 * (nw + 1) + 32 + 1 := by
    show rowIn K false + runsLen (row4A K).runs = _
    simp only [row4A, runsLen_append, runsLen, af4Run, Run.rep, et4Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil, hKn]
    omega
  have hout : ∀ g ∈ (row4A K).outs, val X g = val (((row4A K).spec ops).toCircuit.eval X) g := fun g hg =>
    Spec.out_eq ((row4A K).spec ops) X hX (List.mem_toFinset.2 hg) (by
      simp only [row4A, List.mem_append, List.mem_map, List.mem_range, List.mem_singleton] at hg
      show g < (row4A K).N
      rw [hN]
      rcases hg with (((⟨t, ht, rfl⟩ | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩) | rfl
      · have := code4_lt (b := rowIn K false) (show t < 16 * (K / 16) by omega); omega
      · have := scale4_lt (b := rowIn K false) hl; omega
      · rw [hKn]; omega
      · rw [hKn]
        have : l * (nw + 1) + nw < 32 * (nw + 1) := by nlinarith
        omega
      · rw [hKn]; omega)
  have hmem : post4At K (rowIn K false) + q * (K / 64 + 1) + K / 64 ∈ (row4A K).outs := by
    simp only [row4A, List.mem_append, List.mem_map, List.mem_range]
    exact Or.inl (Or.inr ⟨q, hq, rfl⟩)
  rw [hout _ hmem]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row4A K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row4A K) X
  rw [← hV] at hh
  have hpost := (runsHold_append ops V hh).2
  rw [show (row4A K).nIn + runsLen (pre4Runs K (rowIn K false)) = post4At K (rowIn K false) from hlen] at hpost
  have haf : RunHolds ops V (af4Run K (rowIn K false)) (post4At K (rowIn K false)) := hpost.1
  have hgate : ∀ s < nw + 1, V (post4At K (rowIn K false) + q * (nw + 1) + s) =
      ((af4Run K (rowIn K false)).gate (q * (nw + 1) + s)).prim.eval ops
        (((af4Run K (rowIn K false)).gate (q * (nw + 1) + s)).args.map (Arg.eval ops V)) := by
    intro s hs
    rw [Nat.add_assoc]
    exact haf _ (by simp only [af4Run, Run.rep, hKn]; nlinarith)
  have hsel : ∀ s < nw + 1, (af4Run K (rowIn K false)).gate (q * (nw + 1) + s) =
      if s < nw then chain4 (code4 K (rowIn K false)) (fun t => 33 * K + 32 + 32 * t + q) (scale4 K (rowIn K false))
        (post4At K (rowIn K false) + q * (nw + 1) + s) s
      else ⟨.f2fpBf16, [.g (post4At K (rowIn K false) + q * (nw + 1) + s - 1)]⟩ := by
    intro s hs
    have hd : (q * (nw + 1) + s) / (nw + 1) = q := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hs, Nat.zero_add]
    have hm : (q * (nw + 1) + s) % (nw + 1) = s := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]
    simp only [af4Run, Run.rep, hKn, hd, hm]
  have hch : Chain4Holds (ops := ops) V (code4 K (rowIn K false)) (fun t => 33 * K + 32 + 32 * t + q)
      (scale4 K (rowIn K false)) (post4At K (rowIn K false) + q * (nw + 1)) nw := by
    intro r hr
    rw [hgate r (by omega), hsel r (by omega), ite_eq_left hr]
  have hA' : ∀ t < 64 * nw, V (code4 K (rowIn K false) t) = (x.1.getD t 0).val := by
    intro t ht
    have hm : code4 K (rowIn K false) t ∈ (row4A K).outs := by
      simp only [row4A, List.mem_append, List.mem_map, List.mem_range]
      exact Or.inl (Or.inl (Or.inl (Or.inl ⟨t, by omega, rfl⟩)))
    rw [hV, ← hout _ hm, hA t (by omega)]
  have hS' : ∀ i < 4 * nw, V (scale4 K (rowIn K false) i) = (x.2.getD i 0).val := by
    intro i hi
    have hm : scale4 K (rowIn K false) i ∈ (row4A K).outs := by
      simp only [row4A, List.mem_append, List.mem_map, List.mem_range]
      exact Or.inl (Or.inl (Or.inl (Or.inr ⟨i, by omega, rfl⟩)))
    rw [hV, ← hout _ hm, hS i (by omega)]
  have hB' : ∀ t < 64 * nw, V (33 * K + 32 + 32 * t + q) = (y.getD t 0).val := by
    intro t ht
    rw [hV, Rows.val_input ops (row4A K) X (by show _ < rowIn K false; simp only [rowIn]; omega), hB t (by omega)]
  have hw := chain4_word hch hnw x.1 y x.2 (by omega) hA' hB' hS' (by rw [← hK]; exact hc)
  have hl := hgate nw (by omega)
  rw [hsel nw (by omega), ite_eq_right (Nat.lt_irrefl nw)] at hl
  simp only [List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at hl
  rw [hKn, hl, hw]

/-- **RowB computes `P_B`'s words**: at depth `K = 64·nw`, a transcript correct on RowB whose code and scale outputs
hold a row `x` of `B̃` and whose chain basis' column `q` holds `y` has on `P_B`'s word `q` the BF16 cast of
`Σ_s Ẽ_s·g_{s,q}` (the FMAs from `+0` of the widened `Ẽ` against `G`'s column `q`) plus the negated word of
`cTilde sm120Nvf4` over `x` and `y`, `y` at `1.0`, wherever it is `some`. -/
theorem row4B_pb {K nw : ℕ} (hnw : 0 < nw) (hK : K = 64 * nw) (X : ((row4B K).spec ops).toCircuit.Transcript)
    (hX : ((row4B K).spec ops).part.Correct X 0) (x : Row4) (hx : x.1.length = K)
    (hA : ∀ t < K, val X (code4 K (rowIn K true) t) = (x.1.getD t 0).val)
    (hS : ∀ i < K / 16, val X (scale4 K (rowIn K true) i) = (x.2.getD i 0).val) {q : ℕ} (hq : q < 32)
    (y : List Nib) (hB : ∀ t < K, val X (33 * K + 32 + 32 * t + q) = (y.getD t 0).val) {c : ℕ}
    (hc : cTilde sm120Nvf4 x.1 y x.2 (onesScales sm120Nvf4 K) = some c) :
    val X (post4At K (rowIn K true) + 32 + q * (K / 64 + 35) + K / 64 + 34) =
      Prim.eval ops .f2fpBf16 [Prim.eval ops .add
        [pbAcc ops (fun s => ops.bf16ToF32 (Prim.eval ops .eTilde
            [val X (K + s), Prim.eval ops .toE4m3 [beta4 ops (val X) K]]))
          (fun s => val X (65 * K + 32 + 32 * s + q)) 31, Prim.eval ops .neg [c]]] := by
  have hKn : K / 64 = nw := by omega
  have h8 : 8 ≤ K := by omega
  have hlen := pre4Runs_len K (rowIn K true)
  have hN : (row4B K).N = post4At K (rowIn K true) + 32 + 32 * (nw + 35) + 32 + 1 := by
    show rowIn K true + runsLen (row4B K).runs = _
    simp only [row4B, runsLen_append, runsLen, et4Run, mid4Run, Run.rep, nb4Run, okwRun, Run.ofList,
      List.length_cons, List.length_nil, hKn]
    omega
  have hout : ∀ g ∈ (row4B K).outs, val X g = val (((row4B K).spec ops).toCircuit.eval X) g := fun g hg =>
    Spec.out_eq ((row4B K).spec ops) X hX (List.mem_toFinset.2 hg) (by
      simp only [row4B, List.mem_append, List.mem_map, List.mem_range, List.mem_singleton] at hg
      show g < (row4B K).N
      rw [hN]
      rcases hg with (((⟨t, ht, rfl⟩ | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩) | rfl
      · have := code4_lt (b := rowIn K true) (show t < 16 * (K / 16) by omega); omega
      · have := scale4_lt (b := rowIn K true) hl; omega
      · rw [hKn]
        have : l * (nw + 35) + nw + 34 < 32 * (nw + 35) := by nlinarith
        omega
      · omega
      · omega)
  have hmem : post4At K (rowIn K true) + 32 + q * (K / 64 + 35) + K / 64 + 34 ∈ (row4B K).outs := by
    simp only [row4B, List.mem_append, List.mem_map, List.mem_range]
    exact Or.inl (Or.inl (Or.inr ⟨q, hq, rfl⟩))
  rw [hout _ hmem]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row4B K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row4B K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre4Runs K (rowIn K true)) (rowIn K true) := hsplit.1
  have hpost := hsplit.2
  rw [show (row4B K).nIn + runsLen (pre4Runs K (rowIn K true)) = post4At K (rowIn K true) from hlen] at hpost
  have het : RunHolds ops V (et4Run K (rowIn K true)) (post4At K (rowIn K true)) := hpost.1
  have hmid : RunHolds ops V (mid4Run K (rowIn K true)) (post4At K (rowIn K true) + 32) := hpost.2.1
  have hin : ∀ l < rowIn K true, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row4B K) X hl
  set B := post4At K (rowIn K true) + 32 + q * (nw + 35) with hBdef
  have hgate : ∀ s < nw + 35, V (B + s) =
      (mid4 K (rowIn K true) (post4At K (rowIn K true)) q s (B + s)).prim.eval ops
        ((mid4 K (rowIn K true) (post4At K (rowIn K true)) q s (B + s)).args.map (Arg.eval ops V)) := by
    intro s hs
    have hr : q * (nw + 35) + s < (mid4Run K (rowIn K true)).n := by
      simp only [mid4Run, Run.rep, hKn]
      nlinarith
    have := hmid _ hr
    have hd : (q * (nw + 35) + s) / (nw + 35) = q := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hs, Nat.zero_add]
    have hm : (q * (nw + 35) + s) % (nw + 35) = s := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]
    simp only [mid4Run, Run.rep, hKn, hd, hm] at this
    rw [← Nat.add_assoc] at this
    exact this
  have hacc : ∀ s, s < 32 → V (B + s) =
      pbAcc ops (fun s => ops.bf16ToF32 (V (post4At K (rowIn K true) + s)))
        (fun s => V (65 * K + 32 + 32 * s + q)) s := by
    intro s
    induction s with
    | zero =>
      intro _
      rw [hgate 0 (by omega)]
      simp only [mid4, show (0 : ℕ) < 32 by omega, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval,
        Wire.eval, pbAcc, Nat.add_zero]
    | succ s ih =>
      intro hs
      rw [hgate (s + 1) (by omega)]
      simp only [mid4, hs, ↓reduceIte, Nat.add_one_ne_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval,
        pbAcc]
      rw [show B + (s + 1) - 1 = B + s by omega, ih (by omega)]
  have hch : Chain4Holds (ops := ops) V (code4 K (rowIn K true)) (fun t => 33 * K + 32 + 32 * t + q)
      (scale4 K (rowIn K true)) (B + 32) nw := by
    intro r hr
    have := hgate (32 + r) (by omega)
    simp only [mid4, show ¬ (32 + r < 32) by omega, show 32 + r < 32 + K / 64 by omega, ↓reduceIte,
      show 32 + r - 32 = r by omega] at this
    rw [show B + 32 + r = B + (32 + r) by omega]
    exact this
  have hA' : ∀ t < 64 * nw, V (code4 K (rowIn K true) t) = (x.1.getD t 0).val := by
    intro t ht
    have hm : code4 K (rowIn K true) t ∈ (row4B K).outs := by
      simp only [row4B, List.mem_append, List.mem_map, List.mem_range]
      exact Or.inl (Or.inl (Or.inl (Or.inl ⟨t, by omega, rfl⟩)))
    rw [hV, ← hout _ hm, hA t (by omega)]
  have hS' : ∀ i < 4 * nw, V (scale4 K (rowIn K true) i) = (x.2.getD i 0).val := by
    intro i hi
    have hm : scale4 K (rowIn K true) i ∈ (row4B K).outs := by
      simp only [row4B, List.mem_append, List.mem_map, List.mem_range]
      exact Or.inl (Or.inl (Or.inl (Or.inr ⟨i, by omega, rfl⟩)))
    rw [hV, ← hout _ hm, hS i (by omega)]
  have hB' : ∀ t < 64 * nw, V (33 * K + 32 + 32 * t + q) = (y.getD t 0).val := by
    intro t ht
    rw [hin _ (by simp only [rowIn]; omega), hB t (by omega)]
  have hw := chain4_word hch hnw x.1 y x.2 (by omega) hA' hB' hS' (by rw [← hK]; exact hc)
  have hneg := hgate (32 + nw) (by omega)
  have hadd := hgate (33 + nw) (by omega)
  have hcast := hgate (34 + nw) (by omega)
  simp only [mid4, hKn] at hneg hadd hcast
  simp only [show ¬ (32 + nw < 32) by omega, show ¬ (32 + nw < 32 + nw) from Nat.lt_irrefl _,
    ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at hneg
  simp only [show ¬ (33 + nw < 32) by omega, show ¬ (33 + nw < 32 + nw) by omega,
    show ¬ (33 + nw = 32 + nw) by omega, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id] at hadd
  simp only [show ¬ (34 + nw < 32) by omega, show ¬ (34 + nw < 32 + nw) by omega,
    show ¬ (34 + nw = 32 + nw) by omega, show ¬ (34 + nw = 33 + nw) by omega, ↓reduceIte,
    List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at hcast
  rw [hKn, show post4At K (rowIn K true) + 32 + q * (nw + 35) + nw + 34 = B + (34 + nw) by omega, hcast,
    show B + (34 + nw) - 1 = B + (33 + nw) by omega, hadd, show B + (33 + nw) - 2 - nw = B + 31 by omega,
    show B + (33 + nw) - 1 = B + (32 + nw) by omega, hneg, show B + (32 + nw) - 1 = B + 32 + nw - 1 by omega, hw,
    hacc 31 (by omega)]
  have hE : pbAcc ops (fun s => ops.bf16ToF32 (V (post4At K (rowIn K true) + s)))
      (fun s => V (65 * K + 32 + 32 * s + q)) 31 =
      pbAcc ops (fun s => ops.bf16ToF32 (Prim.eval ops .eTilde
          [val X (K + s), Prim.eval ops .toE4m3 [beta4 ops (val X) K]]))
        (fun s => val X (65 * K + 32 + 32 * s + q)) 31 := by
    refine pbAcc_congr 31 fun s hs => ⟨?_, hin _ (by simp only [rowIn, ↓reduceIte]; omega)⟩
    have g := het s (by simp [et4Run]; omega)
    simp only [et4Run, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g
    rw [g, (pre4_alpha h8 hpre).2.2, hin _ (by simp only [rowIn]; omega),
      beta4_congr h8 fun l hl => hin l (by simp only [rowIn]; omega)]
  rw [hE]

end Sides4

end Pouw.PearlC.RowCircuit
