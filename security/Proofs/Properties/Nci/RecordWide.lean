module

public import Proofs.Properties.Nci.RecordGates
public import Proofs.Properties.Nci.RecordRowBits
public import Proofs.Properties.Nci.RecordTakes
public import Proofs.Properties.Nci.RecordWire

/-!
# A wired row holds its output

* `outFits` (`OutFits`): the output row a reader row is wired to has no more bits than the row. The output's words
  are its bits rounded up to whole 16-bit words, and they are the words of a row of the reader's cut
  (`wire_shape`, `shape_spec`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row fieldsOf outerOf)
open NciCheck.Record.SecurityProofs (mem_of_lookup)
open Spot.Bits (bitsOf)

/-! ## A wired row holds its output -/

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

/-- **A wired row holds its output** (`OutFits`), of a wiring `Spot.Wires.check` accepts whose templates are the
Program's. The wire's shape gives a row of the reader's cut the words of the writer's output; the output's words are
its bits rounded up to whole words, and the row's bits are sixteen times its words or more. -/
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
  -- the row's port is parameter `i`
  obtain ⟨i', hi', hpi', hwords, houter⟩ := (hrows r.port p).1 (mem_of_lookup _ _ _ hlp)
  obtain rfl : i' = i := hnames i' i hi' hi (hpi'.symm.trans hp.symm)
  -- the writer's output is field `f`
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

end NciCheck.Properties.Record.SecurityProofs
