import Definitions.Pouw.PearlC.W1
-- `W1.Instr.exec`'s equation lemmas, realized there first: two modules that each realize them don't import together
import Proofs.Pouw.PearlC.W1BudgetAbort
import Proofs.Pouw.PearlC.RowSeedGamma

/-!
# The row-seed reductions' cost-model conditions at W1

The row-seed readers of TT_OUT (`RowSeedGamma.lean`, `RowSeedKGamma.lean`, `RowSeedCode.lean`, `RowCapAtGamma.lean`)
take two conditions on the cost model: (C3) `CostModel.SplitClosed`, and for the code-form seed (C5)
`CostModel.RelabelClosed`. At `W1.w1Priced dv salt`, the cost model TT_OUT is stated at (`TTOutW1`):

- (C5) holds (`w1Priced_relabelClosed`): W1 runs no oracle, so relabelling the oracle changes nothing.
- (C3) fails, for every device and salt encoding, as soon as there is an oracle and a salt
  (`w1Priced_not_splitClosed`). `splitTranscript` commits row `i` of a unit as every row of a one-row matrix
  (`rowCodes`), so a split unit's ticket matrix repeats that row at every row index. W1 reads a unit's ticket words
  off memory (`W1.readout`), and a W1 program writes finitely many words (`exec_frame`) into a ticket region that starts
  at 0. The one-move program `ticketOne` sets unit 0's ticket word `(0, 0)` to 1 on a one-unit workload, and no W1
  program on the split workload commits a ticket word `(i, 0)` of 1 at every `i`.

So the row-seed readers can't be restated at W1 through `SplitClosed` as it is defined: a W1 reader that kept it as a
hypothesis would be vacuous.
-/

namespace Pouw.PearlC.TTOutW1RowSeed

open Pouw.PearlC

variable {Q R S : Type}

/-- W1 runs no oracle, so it meets (C5): the program itself, against `H`, commits and costs what it does against
`H ∘ π`. -/
theorem w1Priced_relabelClosed (dv : W1.W1Device) (salt : S → List ℕ) :
    (W1.w1Priced dv salt : CostModel Q R S).RelabelClosed :=
  fun _ prog _ _ => ⟨prog, fun _ _ _ => ⟨rfl, le_rfl, le_rfl⟩⟩

/-! ## A W1 run writes below a bound fixed by its code -/

/-- An index past every word the instruction writes, in any region; a register-addressed store writes below
`2^32 + off`. -/
def instrBound : W1.Instr → ℕ
  | .mov d _ => d.2 + 1
  | .ld d _ _ _ => d.2 + 1
  | .st _ _ off _ => 2 ^ 32 + off
  | .iop _ d _ _ => d.2 + 1
  | .iadd3 d _ _ _ => d.2 + 1
  | .lop3 d _ _ _ _ => d.2 + 1
  | .prmt d _ _ _ => d.2 + 1
  | .sel d _ _ _ => d.2 + 1
  | .fop _ d _ _ _ => d.2 + 1
  | .cvt _ d _ _ _ => d.2 + 1
  | .mma f d _ _ _ _ _ => d.2 + f.m * f.n
  | .sha d _ _ => d.2 + 16

/-- An index past every word the code writes, on every path. -/
def codeBound : W1.Code → ℕ
  | .done => 0
  | .op i k => instrBound i + codeBound k
  | .ite _ a b k => codeBound a + codeBound b + codeBound k

theorem instr_frame (dv : W1.W1Device) (i : W1.Instr) (M : W1.Mem) {x : W1.Addr} (h : instrBound i ≤ x.2) :
    i.exec dv M x = M x := by
  have upd : ∀ (d : W1.Addr) (v : ℕ), d.2 + 1 ≤ x.2 → Function.update M d v x = M x := fun d v hd =>
    Function.update_of_ne (by rintro rfl; omega) v M
  have fil : ∀ (d : W1.Addr) (n : ℕ) (g : ℕ → ℕ), d.2 + n ≤ x.2 → W1.Mem.fill M d n g x = M x := fun d n g hd => by
    show (if x.1 = d.1 ∧ d.2 ≤ x.2 ∧ x.2 < d.2 + n then W1.w32 (g (x.2 - d.2)) else M x) = M x
    exact ite_eq_right fun hc => absurd hc.2.2 (by omega)
  cases i with
  | mov d _ => exact upd d _ h
  | ld d _ _ _ => exact upd d _ h
  | st rg r off _ =>
    have hr : W1.w32 (M r) < 2 ^ 32 := Nat.mod_lt _ (by norm_num)
    have h' : 2 ^ 32 + off ≤ x.2 := h
    exact upd (rg, W1.w32 (M r) + off) _ (by show W1.w32 (M r) + off + 1 ≤ x.2; omega)
  | iop _ d _ _ => exact upd d _ h
  | iadd3 d _ _ _ => exact upd d _ h
  | lop3 d _ _ _ _ => exact upd d _ h
  | prmt d _ _ _ => exact upd d _ h
  | sel d _ _ _ => exact upd d _ h
  | fop _ d _ _ _ => exact upd d _ h
  | cvt f d _ _ _ =>
    simp only [W1.Instr.exec]
    by_cases hf : f = .s8
    · rw [ite_eq_left hf]
    · rw [ite_eq_right hf]
      exact upd d _ h
  | mma f d _ _ _ _ _ =>
    simp only [W1.Instr.exec]
    by_cases hh : dv.has f = true
    · rw [ite_eq_left hh]
      exact fil d _ _ h
    · rw [ite_eq_right hh]
  | sha d _ _ =>
    simp only [W1.Instr.exec]
    exact fil d 16 _ h

/-- **A run leaves every word at index `codeBound c` or past it as it found it.** -/
theorem exec_frame (dv : W1.W1Device) :
    ∀ (c : W1.Code) (t : Bool) (st : W1.St) (x : W1.Addr), codeBound c ≤ x.2 → (W1.exec dv t c st).mem x = st.mem x
  | .done, _, _, _, _ => rfl
  | .op i k, t, st, x, h => by
      have h' : instrBound i + codeBound k ≤ x.2 := h
      simp only [W1.exec]
      rw [exec_frame dv k t _ x (by omega)]
      exact instr_frame dv i st.mem (by omega)
  | .ite p a b k, t, st, x, h => by
      have h' : codeBound a + codeBound b + codeBound k ≤ x.2 := h
      simp only [W1.exec]
      rw [exec_frame dv k t _ x (by omega)]
      split_ifs
      · exact exec_frame dv b true _ x (by omega)
      · exact exec_frame dv a true _ x (by omega)

/-- A run leaves every ticket word at row `codeBound` or past it at 0. -/
theorem final_ticket_zero (dv : W1.W1Device) (salt : List ℕ) (pr : W1.CodeImage) (U : Workload) (act : ℕ → Codes)
    (σ : State) (u i j : ℕ) (hi : codeBound (pr.code U.layout) ≤ i) :
    (W1.final dv salt pr U act σ).mem (W1.ticketAt u i j) = 0 := by
  have hx : codeBound (pr.code U.layout) ≤ (W1.ticketAt u i j).2 :=
    hi.trans ((Nat.left_le_pair i j).trans (Nat.right_le_pair u (Nat.pair i j)))
  unfold W1.final
  rw [exec_frame dv _ false _ _ hx]
  simp [W1.init, W1.ticketAt]

/-! ## The counterexample -/

/-- One unit of shape `1 × 1 × 1`, weight 0, all zero. -/
def oneU : Workload := ⟨⟨1, fun _ => ⟨1, 1, 1⟩, fun _ => 0⟩, fun _ _ _ => 0⟩

/-- Activations of 0 everywhere. -/
def zeroAct : ℕ → Codes := fun _ _ _ => 0

/-- Unit 0's ticket word `(0, 0)` set to 1, the constant bank's one word: one free move, two nodes of text, within W1's
bound. -/
def ticketOne : W1.CodeImage := ⟨fun _ => [1], fun _ => .op (.mov (W1.ticketAt 0 0 0) (W1.constAt 0)) .done,
  fun _ => by norm_num [W1.Code.size]⟩

theorem ticketOne_ticket (dv : W1.W1Device) (salt : List ℕ) (U : Workload) (act : ℕ → Codes) (σ : State) :
    (W1.final dv salt ticketOne U act σ).mem (W1.ticketAt 0 0 0) = 1 := by
  simp [W1.final, ticketOne, W1.exec, W1.Instr.exec, W1.init, W1.constAt, W1.w32]

theorem readout_ticket (U : Workload) (act : ℕ → Codes) (M : W1.Mem) (hN : 0 < U.layout.N) (i j : ℕ) :
    (W1.readout U act M 0).map (fun c => c.ticket i j) = some (M (W1.ticketAt 0 i j)) := by
  simp only [W1.readout, ite_eq_left hN]
  rfl

/-- **W1 priced is not closed under splitting rows into units**, for every device and salt encoding, given an oracle
`H` and a salt `s`. On `oneU`, `ticketOne` commits ticket word `(0, 0)` of 1; the split transcript gives the one
split unit that word at row `i` for every `i`, and a program on the split workload leaves its ticket words from row
`codeBound` on at 0. -/
theorem w1Priced_not_splitClosed (dv : W1.W1Device) (salt : S → List ℕ) (H : Q → R) (s : S) :
    ¬ (W1.w1Priced dv salt : CostModel Q R S).SplitClosed := by
  intro hCM
  obtain ⟨prog', act', h⟩ := hCM ticketOne oneU zeroAct
  have hrun : W1.readout (splitWorkload oneU) act' (W1.final dv (salt s) prog' (splitWorkload oneU) act' []).mem =
      splitTranscript oneU.layout (W1.readout oneU zeroAct (W1.final dv (salt s) ticketOne oneU zeroAct []).mem) :=
    (h H [] s).1
  have hsplit := splitTranscript_rowIdx
    (W1.readout oneU zeroAct (W1.final dv (salt s) ticketOne oneU zeroAct []).mem)
    (L := oneU.layout) (u := 0) (i := 0) (by decide) (by decide)
  have h00 : rowIdx oneU.layout 0 0 = 0 := rfl
  rw [h00, ← hrun] at hsplit
  obtain ⟨B, hB⟩ : ∃ B, codeBound (W1.CodeImage.code prog' (splitWorkload oneU).layout) ≤ B := ⟨_, le_rfl⟩
  have hN' : 0 < (splitWorkload oneU).layout.N := by decide
  have hL := readout_ticket (splitWorkload oneU) act' (W1.final dv (salt s) prog' (splitWorkload oneU) act' []).mem
    hN' B 0
  have hR : Option.map (fun c : Commit => c.ticket B 0)
      (Option.map (fun c => rowCommit c 0)
        (W1.readout oneU zeroAct (W1.final dv (salt s) ticketOne oneU zeroAct []).mem 0)) =
      some ((W1.final dv (salt s) ticketOne oneU zeroAct []).mem (W1.ticketAt 0 0 0)) := by
    rw [Option.map_map]
    exact readout_ticket oneU zeroAct _ (by decide) 0 0
  have key := Option.some.inj (hL.symm.trans ((congrArg (Option.map fun c : Commit => c.ticket B 0) hsplit).trans hR))
  rw [final_ticket_zero dv (salt s) prog' (splitWorkload oneU) act' [] 0 B 0 hB, ticketOne_ticket] at key
  omega

end Pouw.PearlC.TTOutW1RowSeed
