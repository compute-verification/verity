/-
The read's check accepts its own template: `Flock.ReadNet.check V H (readTmpl V H)` for every vocabulary of `2 ≤ V < 2³²`
rows of `1 ≤ H ≤ 2¹⁶` words (`check_readTmpl`). Its runs write fresh variables (`Accepts/Fresh.lean`), and its root has at
most `2³²` columns: each part's Definition has at most `partColsMax H` (`Accepts/Parts.lean`, over `Accepts/Cost.lean`) and
the root lays out its calls' columns past its inputs and before its outputs.
-/
module

public import Proofs.Accounting.ReadUnit.Accepts.Fresh
public import Proofs.Accounting.ReadUnit.Accepts.Parts
public import Proofs.Accounting.ReadUnit.Check.Copies

@[expose] public section

namespace Accounting.ReadUnit.Accepts

open Flock.ReadNet
open Accounting.ReadUnit.Gates

theorem log2ceil_le {n k : ℕ} (h : n ≤ 2 ^ k) : Flock.log2ceil n ≤ k := by
  unfold Flock.log2ceil
  split
  · exact Nat.zero_le _
  · have := (Nat.log2_lt (n := n - 1) (k := k) (by omega)).2 (by omega)
    omega

theorem isPowerOfTwo_two_pow : ∀ k : ℕ, (2 ^ k).isPowerOfTwo
  | 0 => by rw [Nat.pow_zero]; exact Nat.isPowerOfTwo_one
  | k + 1 => by rw [Nat.pow_succ]; exact Nat.isPowerOfTwo_mul_two_of_isPowerOfTwo (isPowerOfTwo_two_pow k)

theorem nextPowerOfTwo_le_two_mul (x : ℕ) : x.nextPowerOfTwo ≤ 2 * x + 1 := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · exact Nat.nextPowerOfTwo_le (m := 1) (Nat.zero_le 1) Nat.isPowerOfTwo_one
  · have h1 := Nat.log2_self_le (n := x) (by omega)
    have h2 := Nat.nextPowerOfTwo_le (Nat.lt_log2_self (n := x)).le (isPowerOfTwo_two_pow (x.log2 + 1))
    rw [Nat.pow_succ] at h2
    omega

theorem partDefs_cols_le {V H : ℕ} (hV : V < 2 ^ 32) (i : ℕ) :
    ((partDefs V H).getD i default).cols ≤ partColsMax H := by
  rw [Array.getD_eq_getD_getElem?]
  unfold partDefs
  rw [Array.getElem?_map]
  cases h : (parts V H)[i]? with
  | none => exact Nat.zero_le _
  | some p => exact partDef_cols_le hV p (Array.mem_of_getElem? h)

theorem root_cols_le {V H : ℕ} (hV : V < 2 ^ 32) (hH : H ≤ 2 ^ 16) :
    (rootDef V H (partDefs V H)).cols ≤ 2 ^ 32 := by
  have hw : Flock.WORD = 128 := rfl
  have hC : COMPRESS = 121856 := rfl
  have hr : Flock.HmRow.ROW_SEG_BITS = 8192 := rfl
  have hH' : H ≤ 65536 := hH
  have hD := depth_le hV
  have hs : segs H ≤ 128 := by unfold segs; rw [hr]; omega
  have hcols : partColsMax H ≤ 18793472 := by unfold partColsMax; rw [Nat.add_mul, hC]; omega
  have hn : nCalls V H ≤ 165 := by unfold nCalls; omega
  have hsum : ((List.range (nCalls V H)).map fun i => ((partDefs V H).getD (partOf V H i) default).cols).sum ≤
      165 * 18793472 :=
    (sum_map_le (List.range (nCalls V H)) _ (partColsMax H) fun i _ => partDefs_cols_le hV (partOf V H i)).trans
      (by rw [List.length_range]; exact Nat.mul_le_mul hn hcols)
  have hin : (lay V H).inW ≤ 2 * ((2592 + 16 * H + 512 * depth V + Flock.WORD - 1) / Flock.WORD) + 1 := by
    show Nat.nextPowerOfTwo ((2592 + 16 * H + 512 * depth V + Flock.WORD - 1) / Flock.WORD) ≤ _
    exact nextPowerOfTwo_le_two_mul _
  have hout : (lay V H).outW ≤ 2 * ((16 * H + 16 + Flock.WORD - 1) / Flock.WORD) + 1 := by
    show Nat.nextPowerOfTwo ((16 * H + 16 + Flock.WORD - 1) / Flock.WORD) ≤ _
    exact nextPowerOfTwo_le_two_mul _
  have hO : (lay V H).oStart ≤ (lay V H).base (nCalls V H) + (lay V H).outW * Flock.WORD - 1 := by
    show ((lay V H).base (nCalls V H) + (lay V H).outW * Flock.WORD - 1) / ((lay V H).outW * Flock.WORD) *
      ((lay V H).outW * Flock.WORD) ≤ _
    exact Nat.div_mul_le_self _ _
  have h32 : (2 : ℕ) ^ 32 = 4294967296 := rfl
  rw [rootDef_cols, lay_const, h32]
  rw [lay_base, lay_zero] at hO
  rw [hw] at hin hout hO ⊢
  omega

theorem check_of_checked {V H : ℕ} {t : Flock.Tmpl.Tmpl} (h : Checked V H t) : check V H t = true := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨h.hV, h.hV32⟩, h.hH⟩, h.hH64⟩, h.ok⟩, h.root⟩, h.inG⟩, h.outG⟩, h.defs⟩, h.log⟩

/-- **The read's check accepts its template** for every `2 ≤ V < 2³²` and `1 ≤ H ≤ 2¹⁶`. -/
theorem check_readTmpl {V H : ℕ} (hV : 2 ≤ V) (hV32 : V < 2 ^ 32) (hH : 1 ≤ H) (hH16 : H ≤ 2 ^ 16) :
    check V H (readTmpl V H) = true := by
  refine check_of_checked ⟨hV, hV32, hH, lt_of_le_of_lt (Nat.mul_le_mul_left 16 hH16) (by norm_num),
    runOk_parts V H, rfl, rfl, rfl, rfl, ?_⟩
  show Flock.log2ceil ((readTmpl V H).defs.getD ROOT default).cols ≤ 32
  rw [readTmpl_root]
  exact log2ceil_le (root_cols_le hV32 hH16)

end Accounting.ReadUnit.Accepts
