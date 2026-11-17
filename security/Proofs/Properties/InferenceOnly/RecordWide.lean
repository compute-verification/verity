module

public import Proofs.Properties.InferenceOnly.RecordGates
public import Proofs.Properties.InferenceOnly.RecordRowBits
public import Proofs.Properties.InferenceOnly.RecordTakes
public import Proofs.Properties.InferenceOnly.RecordWire

/-! The output row a reader row is wired to has no more bits than the row (`outFits`). -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope fieldsOf)
open Flock.TemplateQuery (TRange)
open Spot.RecordText (Row outerOf)
open InferenceOnlyCheck.Record.SecurityProofs (mem_of_lookup)
open Spot.Bits (bitsOf)

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

theorem outFits (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : OutFits P rs W S := by
  intro r wt wu out sc sc' i f hw _ hsc hsc' hi hp hout hf
  obtain ⟨w, hwm, idx, hidx, rfl, rfl, rfl⟩ := writerOf_wire hS hw
  obtain ⟨hrd, hpt, -, -, -, hrow, -⟩ := indexOf_some hidx
  obtain ⟨rt, wtm, hsh, -⟩ := wire_shape hS w hwm
  obtain ⟨hlr, hlw, p, ow, hlp, hlo, -, hk, -, -, hwd⟩ := shape_spec hsh
  rw [hrd] at hlr
  rw [hpt] at hlp
  rw [hrd, hpt] at hk hwd
  obtain ⟨-, hTr⟩ := hT r.tid rt (mem_of_lookup _ _ _ hlr)
  obtain ⟨hnames, h16, hrows, -⟩ := hTr r.unit sc hsc
  obtain ⟨-, hTw⟩ := hT w.writer wtm (mem_of_lookup _ _ _ hlw)
  obtain ⟨-, -, -, houts⟩ := hTw _ sc' hsc'
  obtain ⟨i', hi', hpi', hwords, houter⟩ := (hrows r.port p).1 (mem_of_lookup _ _ _ hlp)
  obtain rfl : i' = i := hnames i' i hi' hi (hpi'.symm.trans hp.symm)
  obtain ⟨f', -, hof', hn⟩ := (houts w.out ow).1 (mem_of_lookup _ _ _ hlo)
  obtain rfl : f' = f := outName_inj f' f (hof'.symm.trans hout)
  obtain ⟨hk0, hko, -⟩ := cut_divides hS hlr hlp
  rw [outRow_bits, inRow_bits _ _ _ _ (by rw [← houter]; exact hko) (by omega)]
  obtain ⟨m, hm⟩ := h16 i' hi
  have hA : bitsOf (P.defs[sc.d]!).params[i']!.2 = 16 * p.words := by omega
  have h1 : 16 * (p.words / W.cutsOf r.tid r.port) ≤ 16 * p.words / W.cutsOf r.tid r.port := by
    rw [Nat.le_div_iff_mul_le hk0, Nat.mul_assoc]
    exact Nat.mul_le_mul_left 16 (Nat.div_mul_le_self _ _)
  have h2 : bitsOf (fieldsOf (P.defs[sc'.d]!).ret)[f']! ≤ 16 * (p.words / W.cutsOf r.tid r.port) := by
    rw [hwd, hn]
    omega
  rw [hA]
  exact h2.trans h1

end InferenceOnlyCheck.Properties.Record.SecurityProofs
