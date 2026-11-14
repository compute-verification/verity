module

public import Proofs.Pouw.PearlPlus.Stats

/-!
# Pearl+'s bridge, part (b), the peel and a row's forming

`Words.peel` takes a cell's checked word `C̃` through four `hopperBf16Dot16` steps against its rows' 16-word slices of
`P_A` and `P_B`, refusing a step whose word is not finite. `peel_uWord` says it is `PeelExact.uWord?`, the spec's four
`k16Step`s on the rows as lists.

For one row, `formV1` forms `E′` (each noise code widened, multiplied by `β` and cast back) and the row's codes (the
cast of one atom from `α·x_t` over `E′` against basis row `t`). `formRow_quant8` says that on a row of finite words
passing `Row8OK` these are `lineCodes` at the row's `β` and `pc8Sem`'s `quant8` at the H100 atom: the statistics are
the row's (`statsAreTheRow`), each `E′` code is `eTilde8`'s and so `lineCodes`' (`eTilde8_line`), and each formed word
is `form8`'s at the row's words, which reads a `−0` word as `+0` (`form8_zeroSign`), and is finite, so its cast is
`e4m3`'s.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

namespace PeelSim

open Pouw.PearlC.RowCircuit WordsSim

theorem wslice_length {a : Array ℕ} {s : ℕ} (hs : s + 16 ≤ a.size) : (Words.slice a s 16).length = 16 := by
  simp only [Words.slice, Array.length_toList, Array.size_extract]
  omega

theorem wslice_getD {a : Array ℕ} {s l : ℕ} (hl : l < 16) (hs : s + 16 ≤ a.size) :
    (Words.slice a s 16).getD l 0 = a.toList.getD (s + l) 0 := by
  rw [List.getD_eq_getElem _ _ (by rw [wslice_length hs]; exact hl),
    List.getD_eq_getElem _ _ (by rw [Array.length_toList]; omega)]
  simp only [Words.slice, Array.getElem_toList, Array.getElem_extract]

theorem mulV2_lt (a b : ℕ) : Verity.Fp32.f32MulV2 a b < 2 ^ 32 := by
  unfold Verity.Fp32.f32MulV2
  generalize (Verity.Fp32.decode a).mul (Verity.Fp32.decode b) = d
  cases d with
  | nan => decide
  | inf s => cases s <;> decide
  | fin s m =>
    unfold Verity.Fp32.roundRn
    dsimp only
    have key : ∀ w : ℕ, (if w ≥ 0x7F800000 then 0x7F800000 else w) < 2 ^ 32 := fun w => by split <;> omega
    exact Nat.or_lt_two_pow (by cases s <;> decide) (key _)

theorem e4m3ToF32_lt : ∀ c : Code, Verity.Prims.e4m3ToF32 c.val < 2 ^ 32 := by decide +kernel

theorem toE4m3_lt (u : ℕ) : Verity.Prims.f32ToE4m3Sat u < 2 ^ 8 := by
  have h : Verity.Prims.f32ToE4m3Sat u = (e4m3Word u).val := CoreOps.pc8OpsAgree_coreOps.cast u
  rw [h]
  exact (e4m3Word u).isLt.trans_le (by norm_num)

theorem eval_toE4m3 (a : ℕ) : Prim.eval coreOps .toE4m3 [a] = Verity.Prims.f32ToE4m3Sat a % 2 ^ 8 := rfl
theorem eval_e4m3ToF32 (c : ℕ) : Prim.eval coreOps .e4m3ToF32 [c] = Verity.Prims.e4m3ToF32 c % 2 ^ 32 := rfl

/-- A row of 32 codes as the atom reads it: the check's list of the row's words, each read as a code. -/
theorem map_code_ofFn {a : Array ℕ} {g : ℕ → ℕ} (ha : a.size = 32) (h : ∀ l < 32, a[l]! = g l) :
    a.toList.map TileCircuit.code = List.ofFn fun l : Fin 32 => TileCircuit.code (g l) := by
  apply List.ext_getElem
  · simp [ha]
  · intro l h1 _
    have hl : l < a.size := by simpa using h1
    simp only [List.getElem_map, Array.getElem_toList, List.getElem_ofFn]
    rw [← getElem!_pos a l hl, h l (by omega)]

end PeelSim

open PeelSim WordsSim in
/-- **The peel is the spec's**: the check's `peel` of a word `c` below `2^32` against rows `P_A` and `P_B` of at least
64 words, where it returns `.ok`, is `uWord?` of `c` and the rows as lists. -/
theorem peel_uWord {c u : ℕ} {pa pb : Array ℕ} (hc : c < 2 ^ 32) (hpa : 64 ≤ pa.size) (hpb : 64 ≤ pb.size)
    (h : Words.peel c pa pb = .ok u) : PeelExact.uWord? c pa.toList pb.toList = some u := by
  unfold Words.peel at h
  obtain ⟨r, hr, h⟩ := bind_ok h
  obtain rfl := pure_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
  have := StatsSim.forIn_inv
    (fun i acc => acc < 2 ^ 32 ∧ (List.range i).foldlM (PeelExact.uStep? pa.toList pb.toList) c = some acc) _ 0
    ⟨hc, rfl⟩ (fun j hj z r hz hr => by
      rw [List.getElem_range'] at hr
      dsimp only at hr
      obtain ⟨v, hv, hr⟩ := bind_ok hr
      obtain rfl := pure_ok hr
      refine ⟨v, rfl, ?_⟩
      have hj4 : j < 4 := by simpa [Std.Legacy.Range.size, PouwCheck.PearlPlus.PEEL_K, PouwCheck.PearlPlus.R] using hj
      simp only [Nat.zero_add] at hz hv ⊢
      obtain ⟨rfl, hf⟩ := finite_ok hv
      rw [bf16Dot_of_lt z (Words.slice pa (16 * j) 16) (Words.slice pb (16 * j) 16) hz.1
        (PeelSim.wslice_length (by omega)) (PeelSim.wslice_length (by omega))] at hf ⊢
      have hu : PeelExact.uStep? pa.toList pb.toList z j = some (PeelExact.k16Step z
          (fun l => (Words.slice pa (16 * j) 16).getD l.val 0)
          (fun l => (Words.slice pb (16 * j) 16).getD l.val 0)) := by
        unfold PeelExact.uStep?
        rw [← finite?_of hf]
        congr 2 <;> funext l <;> unfold PeelExact.slice16 <;> rw [PeelSim.wslice_getD l.isLt (by omega)]
      refine ⟨by unfold PeelExact.k16Step; exact AccumProofs.packValue_lt _, ?_⟩
      rw [List.range_succ, List.foldlM_append, hz.2]
      simp only [List.foldlM_cons, List.foldlM_nil, Option.bind_eq_bind, Option.bind_some, hu, Option.pure_def]) hr
  simp only [List.length_range', Std.Legacy.Range.size, PouwCheck.PearlPlus.PEEL_K, PouwCheck.PearlPlus.R,
    Nat.zero_add] at this
  exact this.2

open PeelSim WordsSim in
/-- **A row's forming is `pc8Sem`'s**: on a row `x` of `k` finite words `w` passing `Row8OK` (`8 ∣ k`, `k ≤ 2^27`), its
noise line's codes `e` and its form basis' rows `f`, where `s5_stats` and the forming's two maps return `.ok`, `E′` is
`lineCodes` at the row's `β` and the row's codes are `quant8`'s at the H100 atom. -/
theorem formRow_quant8 {k : ℕ} {w : ℕ → ℕ} {E : Fin Params.pi.r → Code} {F : Matrix (Fin k) (Fin Params.pi.r) Code}
    {x e : Array ℕ} {f : Array (Array ℕ)} {s ρw αw βw : ℕ} {ep codes : Array ℕ}
    (hk0 : 0 < k) (hk8 : 8 ∣ k) (hkK : k ≤ 2 ^ 27)
    (hxs : x.size = k) (hx : ∀ l < k, x[l]! = w l) (hw : ∀ l < k, w l < 2 ^ 32)
    (hA : WholeCircuit.Row8OK RowCircuit.coreOps k w)
    (hes : e.size = 32) (hE : ∀ q : Fin Params.pi.r, e[q.val]! = (E q).val)
    (hfs : ∀ t : Fin k, (f[t.val]!).size = 32)
    (hF : ∀ (t : Fin k) (q : Fin Params.pi.r), (f[t.val]!)[q.val]! = (F t q).val)
    (hs : Words.s5Stats x = .ok (s, ρw, αw, βw))
    (hep : e.mapM (fun c => do Words.toFp8 (Evaluate.Fp32.f32MulV2 βw (← Words.ofFp8 c))) = .ok ep)
    (hc : (Array.range x.size).mapM (fun t => do
      Words.toFp8 (← Words.finite (Words.atom (Evaluate.Fp32.f32MulV2 αw x[t]!) ep.toList f[t]!.toList))) = .ok codes) :
    ep.size = 32 ∧
      (∀ q : Fin Params.pi.r, ep[q.val]! = (lineCodes (WholeCircuit.betaQ8 RowCircuit.coreOps k w) E q).val) ∧
      codes.size = k ∧
      ∀ t : Fin k, codes[t.val]! = (WholeCircuit.quant8 RowCircuit.coreOps hopperE4m3K32 w E F t).val := by
  have eMul : Evaluate.Fp32.f32MulV2 = Verity.Fp32.f32MulV2 := Properties.Evaluator.Fp32.f32MulV2
  have eWid : Evaluate.Casts.e4m3ToF32 = Verity.Prims.e4m3ToF32 := Properties.Evaluator.Casts.e4m3ToF32
  have eCast : Evaluate.Casts.f32ToE4m3Sat = Verity.Prims.f32ToE4m3Sat := Properties.Evaluator.Casts.f32ToE4m3Sat
  have hops := CoreOps.pc8OpsAgree_coreOps
  obtain ⟨-, hα, hβ⟩ := statsAreTheRow k x s ρw αw βw hxs hk0 hk8 hkK
    (fun l hl => by rw [hx l hl]; exact ⟨hw l hl, hA.1 l hl⟩) hs
  have hrv : rowVals k x = WholeCircuit.rowQ k w :=
    funext fun l => by simp only [rowVals, WholeCircuit.rowQ, hx l l.isLt]
  rw [hrv] at hα hβ
  obtain ⟨hαS, hβS, hβpos⟩ := hA.2
  rw [← hα] at hαS
  rw [← hβ] at hβS hβpos
  have hβQ : WholeCircuit.betaQ8 RowCircuit.coreOps k w = PearlC.wordVal βw := by
    unfold WholeCircuit.betaQ8; rw [hβ]
  have hαW : f32Word (WholeCircuit.alphaQ8 RowCircuit.coreOps k w) = αw := by
    rw [show WholeCircuit.alphaQ8 RowCircuit.coreOps k w = PearlC.wordVal αw by unfold WholeCircuit.alphaQ8; rw [hα]]
    exact RowCircuit.f32Word_wordVal hαS
  have hβW : f32Word (WholeCircuit.betaQ8 RowCircuit.coreOps k w) = βw := by
    rw [hβQ]; exact RowCircuit.f32Word_wordVal hβS
  obtain ⟨hepS, hepA⟩ := mapM_ok hep
  have hep32 : ep.size = 32 := hepS.trans hes
  have hE' : ∀ q : Fin Params.pi.r,
      ep[q.val]! = RowCircuit.eTilde8 RowCircuit.coreOps βw (RowCircuit.encCodes E q) ∧ (E q).isNaN = false := by
    intro q
    obtain ⟨y, hy, ho⟩ := bind_ok (hepA q.val (by rw [hes]; exact q.isLt))
    obtain ⟨hn, rfl⟩ := ofFp8_ok hy
    obtain ⟨-, ho⟩ := toFp8_ok ho
    rw [hE q] at hn ho
    rw [eCast, eMul, eWid] at ho
    refine ⟨?_, by simpa [Code.isNaN] using hn⟩
    rw [ho, RowCircuit.eTilde8, WholeCircuit.encCodes_of_lt q.isLt, PeelSim.eval_e4m3ToF32,
      Nat.mod_eq_of_lt (PeelSim.e4m3ToF32_lt _), StatsSim.eval_mul, Nat.mod_eq_of_lt (PeelSim.mulV2_lt _ _),
      PeelSim.eval_toE4m3, Nat.mod_eq_of_lt (PeelSim.toE4m3_lt _)]
  obtain ⟨hcS, hcA⟩ := mapM_ok hc
  refine ⟨hep32, fun q => ?_, by rw [hcS, Array.size_range, hxs], fun t => ?_⟩
  · obtain ⟨h1, hn⟩ := hE' q
    rw [h1, hβQ]
    exact WholeCircuit.eTilde8_line hops hβS hβpos q hn
  have ht : t.val < (Array.range x.size).size := by rw [Array.size_range, hxs]; exact t.isLt
  have h1 := hcA t.val ht
  rw [getElem!_pos (Array.range x.size) t.val ht, Array.getElem_range] at h1
  obtain ⟨D, hD, h1⟩ := bind_ok h1
  obtain ⟨hDe, hDf⟩ := finite_ok hD
  rw [← hDe] at hDf
  obtain ⟨-, h1⟩ := toFp8_ok h1
  have hD' : D = hopperE4m3K32.step (Verity.Fp32.f32MulV2 αw (w t)) (ep.toList.map TileCircuit.code)
      ((f[t.val]!).toList.map TileCircuit.code) := by
    rw [hDe, ← eMul, ← hx t t.isLt]
    exact atomIsTheStep _ _ _
  have hDS : AccumProofs.StepWord D :=
    ⟨by rw [hD']; exact AccumProofs.step_lt _ _ _ _, by rw [hD']; exact AccumProofs.step_ne_negZero _ _ _ _,
      finite?_of hDf⟩
  have hform : RowCircuit.form8 RowCircuit.coreOps hopperE4m3K32 αw (w t)
      (fun l => RowCircuit.eTilde8 RowCircuit.coreOps βw (RowCircuit.encCodes E l)) (RowCircuit.encCodes (F t)) =
        D := by
    rw [hD']
    unfold RowCircuit.form8
    rw [List.cons_append, RowCircuit.atom8_eval, StatsSim.eval_mul, Nat.mod_eq_of_lt (PeelSim.mulV2_lt _ _),
      PeelSim.map_code_ofFn (g := fun l => RowCircuit.eTilde8 RowCircuit.coreOps βw (RowCircuit.encCodes E l)) hep32
        (fun l hl => (hE' ⟨l, hl⟩).1),
      PeelSim.map_code_ofFn (g := RowCircuit.encCodes (F t)) (hfs t)
        (fun l hl => (hF t ⟨l, hl⟩).trans (WholeCircuit.encCodes_of_lt hl).symm)]
  have hz : wordQ D = some 0 → (D >>> 31) % 2 = 0 := fun h0 => by
    rw [AccumProofs.wordQ_zero_word hDS h0]; rfl
  have hcast : Verity.Prims.f32ToE4m3Sat D = (e4m3Word D).val := hops.cast D
  unfold WholeCircuit.quant8 RowCircuit.pc8Form
  dsimp only
  rw [hαW, hβW, ← WholeCircuit.form8_zeroSign hops hopperE4m3K32 hαS (WholeCircuit.rowQ_zeroSign hA.1 t).1,
    Nat.mod_eq_of_lt (hw t t.isLt), hform, RowCircuit.e4m3_wordVal (fun h => absurd h hDf) hz, h1, eCast, hcast]

end Pouw.PearlPlus.Bridge
