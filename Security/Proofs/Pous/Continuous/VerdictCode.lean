import Proofs.Pous.Continuous.Verdict
import PousCheck.Verdict

/-!
# The PoUS verdict unit's code is the analysis's

`PousCheck.Verdict` (`verity/core/protocols/pous/check/PousCheck/Verdict.lean`) is the verdict unit
`PousVerdict_v1{K, M, LIMIT}` as executable code: its gates (`prog`), a run of a program (`Prog.eval`), and the
verdicts and acceptance a run decides. The difftest `verity/core/protocols/pous/check/tests/test_verdict_difftest.py`
runs `prog` against the Definition's compiled circuit (`verdict_circuit.PousVerdict`). This file proves the code is
`Continuous/Verdict.lean`'s, so the difftest holds the Definition to `verdictProg`.

* `prog_eq`: the code's `prog`, gate for gate, is `verdictProg`; so `VerdictIs` holds of it (`verdictIs_prog`).
* `eval_eq`: the code runs a program as `Claims.Sampling.Prog.eval` does.
* `verdicts_eq`, `encode_eq`, `accepts_iff`: the code's verdicts, input layout and acceptance are the analysis's.
* `prog_eval`, `prog_accepts`: `verdictProg_eval` and `verdict_accepts` read through the code: run by the code, `prog`'s
  output is `accepts M` of the code's verdicts, and `1` iff `contAccepts M` holds on records with those verdicts.

What stays a test: the deployed unit is the descriptor flattened (`AuditStatement.prog`), and no Lean reads a descriptor into
gates, so that the Definition's circuit is `prog` at given statics is the difftest's, at the statics it runs.
-/

namespace PousVerdict.Code

open Claims.Sampling Claims.Sampling.Univ Pous

/-! ## The gates -/

/-- A gate of the code as the analysis's. -/
def toB : PousCheck.Verdict.Gate → BGate
  | .input => .input
  | .const b => .const b
  | .not i => .not i
  | .and ws => .and ws
  | .xor ws => .xor ws

/-- The code's gates as the analysis's list. -/
def toL (gs : Array PousCheck.Verdict.Gate) : List BGate := gs.toList.map toB

/-- The code's gates and the wires it returns beside them, as the analysis's. -/
def lift {α : Type} (st : Array PousCheck.Verdict.Gate × α) : List BGate × α := (toL st.1, st.2)

/-- The code's program as the analysis's. -/
def toProg (C : PousCheck.Verdict.Prog) : Claims.Sampling.Prog := ⟨toL C.gates, C.outs⟩

theorem toL_push (gs : Array PousCheck.Verdict.Gate) (g : PousCheck.Verdict.Gate) :
    toL (gs.push g) = toL gs ++ [toB g] := by
  simp only [toL, Array.toList_push, List.map_append, List.map_cons, List.map_nil]

theorem toL_append (gs : Array PousCheck.Verdict.Gate) (l : List PousCheck.Verdict.Gate) :
    toL (gs ++ l) = toL gs ++ l.map toB := by
  simp only [toL, Array.toList_appendList, List.map_append]

theorem length_toL (gs : Array PousCheck.Verdict.Gate) : (toL gs).length = gs.size := by
  simp only [toL, List.length_map, Array.length_toList]

theorem map_toB_xor {α : Type} (f : α → List ℕ) (l : List α) :
    (l.map fun a => PousCheck.Verdict.Gate.xor (f a)).map toB = l.map fun a => BGate.xor (f a) := by
  simp only [List.map_map, Function.comp_def, toB]

theorem recordBits_eq : PousCheck.Verdict.recordBits = recordBits := rfl
theorem digestBits_eq : PousCheck.Verdict.digestBits = digestBits := rfl
theorem inputs_eq : PousCheck.Verdict.inputs = inputs := rfl
theorem beWires_eq : PousCheck.Verdict.beWires = beWires := rfl
theorem constWires_eq : PousCheck.Verdict.constWires = constWires := rfl

theorem carries_eq : ∀ (l : List (ℕ × ℕ)) (gs : Array PousCheck.Verdict.Gate) (c : ℕ),
    lift (PousCheck.Verdict.carries gs l c) = carries (toL gs) l c
  | [], _, _ => rfl
  | (x, p) :: r, gs, c => by
    have ih := carries_eq r
      (((gs.push (.xor [x, c])).push (.and [p, gs.size])).push (.xor [x, gs.size + 1])) (gs.size + 2)
    simp only [toL_push, List.append_assoc, List.cons_append, List.nil_append, toB] at ih
    simp only [PousCheck.Verdict.carries, carries, length_toL]
    rw [← ih]
    rfl

theorem chain_eq (w1 : ℕ) (gs : Array PousCheck.Verdict.Gate) (a s : List ℕ) (c0 : ℕ) :
    lift (PousCheck.Verdict.chain w1 gs a s c0) = chain w1 (toL gs) a s c0 := by
  have h := carries_eq (a.zip ((List.range (a.zip s).length).map (gs.size + ·)))
    (gs ++ (a.zip s).map fun xy => PousCheck.Verdict.Gate.xor [xy.1, xy.2, w1]) c0
  simp only [toL_append, map_toB_xor] at h
  simp only [PousCheck.Verdict.chain, chain, length_toL, ← h, lift]

theorem sums_eq (gs : Array PousCheck.Verdict.Gate) (ps cs : List ℕ) :
    lift (PousCheck.Verdict.sums gs ps cs) = sums (toL gs) ps cs := by
  simp only [PousCheck.Verdict.sums, sums, lift, toL_append, map_toB_xor, length_toL]

theorem litGates_eq (K w0 w1 j : ℕ) : (PousCheck.Verdict.litGates K w0 w1 j).map toB = litGates K w0 w1 j := by
  simp only [PousCheck.Verdict.litGates, litGates, List.map_append, map_toB_xor, recordBits_eq, digestBits_eq,
    beWires_eq, constWires_eq]

theorem roundGates_eq (K LIMIT w0 w1 j : ℕ) (gs : Array PousCheck.Verdict.Gate) :
    lift (PousCheck.Verdict.roundGates K LIMIT w0 w1 j gs) = roundGates K LIMIT w0 w1 j (toL gs) := by
  have hl : toL gs ++ litGates K w0 w1 j = toL (gs ++ PousCheck.Verdict.litGates K w0 w1 j) := by
    rw [toL_append, litGates_eq]
  have hn : (litGates K w0 w1 j).length = (PousCheck.Verdict.litGates K w0 w1 j).length := by
    rw [← litGates_eq, List.length_map]
  simp only [roundGates, hl, hn, ← chain_eq, lift, ← sums_eq, length_toL]
  simp only [PousCheck.Verdict.roundGates, toL_push, toB, recordBits_eq, beWires_eq, constWires_eq]

theorem initGates_eq (K : ℕ) : toL (PousCheck.Verdict.initGates K) = initGates K := by
  simp only [toL, PousCheck.Verdict.initGates, inputs_eq, Array.toList_push, Array.toList_replicate,
    List.append_assoc, List.cons_append, List.nil_append, List.map_append, List.map_replicate, toB, List.map_cons,
    List.map_nil, initGates]

theorem foldl_lift {α β γ : Type} (l : α → β) (f : α → γ → α) (g : β → γ → β)
    (h : ∀ st j, l (f st j) = g (l st) j) :
    ∀ (js : List γ) (st : α), l (js.foldl f st) = js.foldl g (l st)
  | [], _ => rfl
  | j :: js, st => by rw [List.foldl_cons, List.foldl_cons, foldl_lift l f g h js, h]

theorem hitStep_eq (K LIMIT : ℕ) (st : Array PousCheck.Verdict.Gate × List ℕ) (j : ℕ) :
    lift (PousCheck.Verdict.hitStep K LIMIT st j) = hitStep K LIMIT (lift st) j := by
  have h := roundGates_eq K LIMIT (inputs K) (inputs K + 1) j st.1
  simp only [lift] at h
  simp only [PousCheck.Verdict.hitStep, hitStep, lift, inputs_eq, ← h]

theorem hitGates_eq (K LIMIT : ℕ) : lift (PousCheck.Verdict.hitGates K LIMIT) = hitGates K LIMIT := by
  rw [PousCheck.Verdict.hitGates, hitGates, foldl_lift lift _ _ (hitStep_eq K LIMIT)]
  simp only [lift, initGates_eq]

/-- `Nat.size` is the bit length. -/
theorem size_bitLength (n : ℕ) : Nat.size n = PousCheck.Verdict.bitLength n := by
  unfold PousCheck.Verdict.bitLength
  split
  · subst_vars
    exact Nat.size_zero
  · rename_i h
    apply le_antisymm
    · exact Nat.size_le.2 Nat.lt_log2_self
    · exact Nat.lt_size.2 (Nat.log2_self_le h)

theorem counterWidth_eq (K M : ℕ) : PousCheck.Verdict.counterWidth K M = counterWidth K M := by
  rw [PousCheck.Verdict.counterWidth, counterWidth, size_bitLength]

theorem countStep_eq (w1 w : ℕ) (st : Array PousCheck.Verdict.Gate × List ℕ) (h : ℕ) :
    lift (PousCheck.Verdict.countStep w1 w st h) = countStep w1 w (lift st) h := by
  have hc := chain_eq w1 (st.1.push (.not h)) st.2 (List.replicate w w1) st.1.size
  simp only [toL_push, toB, lift] at hc
  simp only [PousCheck.Verdict.countStep, countStep, lift, length_toL]
  rw [← hc]
  exact sums_eq _ _ _

theorem countGates_eq (K M LIMIT : ℕ) : lift (PousCheck.Verdict.countGates K M LIMIT) = countGates K M LIMIT := by
  have hh := hitGates_eq K LIMIT
  simp only [lift] at hh
  rw [PousCheck.Verdict.countGates, countGates, foldl_lift lift _ _ (countStep_eq _ _), counterWidth_eq, inputs_eq]
  simp only [lift, ← hh]

/-- **The code's verdict unit is `verdictProg`**, gate for gate. -/
theorem prog_eq (K M LIMIT : ℕ) : toProg (PousCheck.Verdict.prog K M LIMIT) = verdictProg K M LIMIT := by
  have hc := countGates_eq K M LIMIT
  simp only [lift] at hc
  have h := chain_eq (inputs K + 1) (PousCheck.Verdict.countGates K M LIMIT).1
    (constWires (inputs K) (inputs K + 1) (counterWidth K M) M) (PousCheck.Verdict.countGates K M LIMIT).2
    (inputs K + 1)
  simp only [lift] at h
  simp only [verdictProg, ← hc, ← h, toProg, PousCheck.Verdict.prog, inputs_eq, constWires_eq, counterWidth_eq]

/-- **`VerdictIs` holds of the code's verdict unit.** -/
theorem verdictIs_prog (K M LIMIT : ℕ) : VerdictIs (toProg (PousCheck.Verdict.prog K M LIMIT)) K M LIMIT :=
  fun _ _ => by rw [prog_eq]

/-! ## Running a program -/

theorem getD_toList {α : Type} (a : Array α) (i : ℕ) (d : α) : a.toList.getD i d = a.getD i d := by
  simp only [List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getD_eq_getD_getElem?]

theorem val_toB (f : ℕ → Bool) : ∀ g : PousCheck.Verdict.Gate, g ≠ .input →
    ((toB g).val (fun i => some (f i)) none).getD false = PousCheck.Verdict.gval f g
  | .input, h => absurd rfl h
  | .const _, _ => rfl
  | .not _, _ => rfl
  | .and ws, _ => by
    simp only [toB, BGate.val, mapM_some, Option.map_some, Option.getD_some, PousCheck.Verdict.gval]
  | .xor ws, _ => by
    simp only [toB, BGate.val, mapM_some, Option.map_some, Option.getD_some, PousCheck.Verdict.gval]

theorem wires_foldl (x : List Bool) : ∀ (gs : List PousCheck.Verdict.Gate) (st : Array Bool × ℕ),
    (gs.map toB).foldl (fun (st : List Bool × ℕ) g =>
      match g with
      | .input => (st.1 ++ [x.getD st.2 false], st.2 + 1)
      | g => (st.1 ++ [(g.val (fun i => some (st.1.getD i false)) none).getD false], st.2)) (st.1.toList, st.2) =
      ((gs.foldl (PousCheck.Verdict.wireStep x.toArray) st).1.toList,
        (gs.foldl (PousCheck.Verdict.wireStep x.toArray) st).2)
  | [], _ => rfl
  | g :: gs, st => by
    rw [List.map_cons, List.foldl_cons, List.foldl_cons, ← wires_foldl x gs]
    congr 1
    cases g with
    | input =>
      simp only [toB, List.getD_eq_getElem?_getD, PousCheck.Verdict.wireStep, Array.toList_push,
        Array.getD_eq_getD_getElem?, List.getElem?_toArray]
    | const b =>
      simp only [toB, BGate.val, Option.getD_some, PousCheck.Verdict.wireStep, PousCheck.Verdict.gval,
        Array.toList_push]
    | not i =>
      simp only [PousCheck.Verdict.wireStep, Array.toList_push]
      rw [← val_toB _ (.not i) nofun]
      simp only [toB, getD_toList]
    | and ws =>
      simp only [PousCheck.Verdict.wireStep, Array.toList_push]
      rw [← val_toB _ (.and ws) nofun]
      simp only [toB, getD_toList]
    | xor ws =>
      simp only [PousCheck.Verdict.wireStep, Array.toList_push]
      rw [← val_toB _ (.xor ws) nofun]
      simp only [toB, getD_toList]

theorem wires_eq (C : PousCheck.Verdict.Prog) (x : List Bool) : (toProg C).wires x = (C.wires x).toList := by
  unfold Claims.Sampling.Prog.wires PousCheck.Verdict.Prog.wires toProg toL
  rw [← Array.foldl_toList]
  exact congrArg Prod.fst (wires_foldl x C.gates.toList (#[], 0))

/-- **The code runs a program as the analysis does.** -/
theorem eval_eq (C : PousCheck.Verdict.Prog) (x : List Bool) : (toProg C).eval x = C.eval x := by
  rw [Claims.Sampling.Prog.eval, wires_eq, PousCheck.Verdict.Prog.eval]
  simp only [toProg, getD_toList]

/-! ## The openings and their verdicts -/

/-- An opening of the code as the analysis's. -/
def toO (o : PousCheck.Verdict.Opening) : RoundOpening := ⟨o.row, o.expected⟩

/-- A verdict of the code as the analysis's. -/
def toCV : PousCheck.Verdict.RoundVerdict → ContVerdict
  | .hit => .hit
  | .wrong => .wrong
  | .late => .late
  | .reject => .reject

theorem bitsVal_eq : PousCheck.Verdict.bitsVal = bitsVal := by
  funext l
  induction l with
  | nil => rfl
  | cons b l ih => simp only [PousCheck.Verdict.bitsVal, bitsVal, ih]

theorem j_eq (o : PousCheck.Verdict.Opening) : o.j = (toO o).j := by
  simp only [PousCheck.Verdict.Opening.j, RoundOpening.j, PousCheck.Verdict.beField, beField, bitsVal_eq,
    beWires_eq, toO]

theorem send_eq (o : PousCheck.Verdict.Opening) : o.send = (toO o).send := by
  simp only [PousCheck.Verdict.Opening.send, RoundOpening.send, PousCheck.Verdict.beField, beField, bitsVal_eq,
    beWires_eq, toO]

theorem receipt_eq (o : PousCheck.Verdict.Opening) : o.receipt = (toO o).receipt := by
  simp only [PousCheck.Verdict.Opening.receipt, RoundOpening.receipt, PousCheck.Verdict.beField, beField, bitsVal_eq,
    beWires_eq, toO]

theorem digest_eq (o : PousCheck.Verdict.Opening) : o.digest = (toO o).digest := rfl

theorem expected_eq (o : PousCheck.Verdict.Opening) : o.expected = (toO o).expected := rfl

theorem roundVerdict_eq (LIMIT j : ℕ) (o : PousCheck.Verdict.Opening) :
    toCV (PousCheck.Verdict.roundVerdict LIMIT j o) = roundVerdict LIMIT j (toO o) := by
  rw [PousCheck.Verdict.roundVerdict, roundVerdict, j_eq, send_eq, receipt_eq, digest_eq, expected_eq]
  split_ifs <;> rfl

/-- **The code's verdicts are the analysis's.** -/
theorem verdicts_eq (LIMIT : ℕ) (os : List PousCheck.Verdict.Opening) :
    (PousCheck.Verdict.verdicts LIMIT os).map toCV = verdicts LIMIT (os.map toO) := by
  apply List.ext_getElem (by simp only [PousCheck.Verdict.verdicts, List.length_map, List.length_mapIdx, verdicts])
  intro i h1 h2
  simp only [PousCheck.Verdict.verdicts, List.getElem_map, List.getElem_mapIdx, roundVerdict_eq, verdicts]

theorem encode_eq (os : List PousCheck.Verdict.Opening) : PousCheck.Verdict.encode os = encode (os.map toO) := by
  simp only [PousCheck.Verdict.encode, encode, List.flatMap_map, toO]

theorem misses_eq (vs : List PousCheck.Verdict.RoundVerdict) :
    ((vs.map toCV).filter (· ≠ .hit)).length = PousCheck.Verdict.misses vs := by
  rw [List.filter_map, List.length_map, PousCheck.Verdict.misses]
  congr 1
  apply List.filter_congr
  intro v _
  cases v <;> rfl

/-- **The code accepts as the memory challenger does**: `accepts M` of the verdicts is `contAccepts M` of any records
with them. -/
theorem accepts_iff {M : ℕ} {vs : List PousCheck.Verdict.RoundVerdict} {rs : List ContRecord}
    (hrs : rs.map (·.verdict) = vs.map toCV) :
    PousCheck.Verdict.accepts M vs = true ↔ contAccepts (M : ℝ) rs := by
  rw [contAccepts, records_filter, hrs, misses_eq, PousCheck.Verdict.accepts, decide_eq_true_iff]
  exact Nat.cast_le.symm

theorem fits_map {os : List PousCheck.Verdict.Opening}
    (hfit : ∀ o ∈ os, o.row.length = recordBits ∧ o.expected.length = digestBits) :
    ∀ o ∈ os.map toO, o.Fits := by
  intro o ho
  obtain ⟨o', ho', rfl⟩ := List.mem_map.1 ho
  exact hfit o' ho'

/-- **The code's verdict unit outputs what the code accepts**: on the openings of `K ≤ 2³²` rounds of the unit's
widths, `prog`'s output, run by the code, is `accepts M` of the code's verdicts. -/
theorem prog_eval {K M LIMIT : ℕ} {os : List PousCheck.Verdict.Opening} (hK : os.length = K) (hK32 : K ≤ 2 ^ 32)
    (hfit : ∀ o ∈ os, o.row.length = recordBits ∧ o.expected.length = digestBits) :
    (PousCheck.Verdict.prog K M LIMIT).eval (PousCheck.Verdict.encode os) =
      [PousCheck.Verdict.accepts M (PousCheck.Verdict.verdicts LIMIT os)] := by
  rw [← eval_eq, prog_eq, encode_eq, verdictProg_eval (by rw [List.length_map, hK]) hK32
    (fits_map hfit), ← verdicts_eq, misses_eq, PousCheck.Verdict.accepts]

/-- **The code's verdict unit decides `contAccepts`**: on the openings of `K ≤ 2³²` rounds of the unit's widths, its
output, run by the code, is `1` iff the memory challenger with tolerance `M` accepts the records the code's verdicts
are. -/
theorem prog_accepts {K M LIMIT : ℕ} {os : List PousCheck.Verdict.Opening} (hK : os.length = K) (hK32 : K ≤ 2 ^ 32)
    (hfit : ∀ o ∈ os, o.row.length = recordBits ∧ o.expected.length = digestBits) {rs : List ContRecord}
    (hrs : rs.map (·.verdict) = (PousCheck.Verdict.verdicts LIMIT os).map toCV) :
    (PousCheck.Verdict.prog K M LIMIT).eval (PousCheck.Verdict.encode os) = [true] ↔ contAccepts (M : ℝ) rs := by
  rw [← eval_eq, encode_eq]
  rw [verdicts_eq] at hrs
  exact verdict_accepts (verdictIs_prog K M LIMIT) (by rw [List.length_map, hK]) hK32 (fits_map hfit) hrs

end PousVerdict.Code
