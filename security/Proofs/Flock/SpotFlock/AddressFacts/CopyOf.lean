module

public import Proofs.Flock.SpotFlock.AddressFacts.Copy
public import Proofs.Flock.SpotFlock.AddressFacts.WidthOf

/-! `CopyFact` at the address map (`copyFact_of_wiring`), from the wiring, `BitsInPorts`, `RowFact` and `CopiesAt`. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

def CopiesAt : Prop :=
  ∀ t k i rp, BitOf h t k i rp → ∃ m, h.ms[t]? = some m ∧
    CopiesRef m.1 (pickAny h t k i) (portNamed h t rp.row) rp.bit

variable {h}

namespace CheckedRun

variable {r : CheckedRun h.stage}

theorem copyAt_of_wiring (W : Wiring h) (hB : BitsInPorts h) (hC : CopiesAt h) (hr : r.Ok h) (hn : r.Counted h)
    {i : ℕ} (hi : r.s.prog.Needs r.b i) : r.CopyAt h (addressOf h i) i := by
  obtain ⟨hω, rp, tw, rq, hb, hw, hrb, hbit, -⟩ := target_of_wiring W hB hr hi
  obtain ⟨m, hm, hx⟩ := hC _ _ _ _ hb
  have haddr := (addressOf_eq h hω i).trans (addressAt_of_bitOf hb hw)
  have ep : (addressOf h i).port r.u = portNamed h (templateOf h.ms r.u) rp.row := by rw [haddr]
  have e : ((addressOf h i).place.2 : ℕ) = rp.bit := by
    rw [haddr]
    show ((placeAt h tw rq).2 : ℕ) = rp.bit
    rw [placeAt, dite_eq_left hbit]
    exact hrb
  have hx' : CopiesRef m.1 (r.pick h i) ((addressOf h i).port r.u) (addressOf h i).place.2 := by
    rw [ep, e]
    exact hx
  exact copy_of_copies hr (inBlock_of_counted hr hn) hm (copies_of_ref hr hm hx')

end CheckedRun

theorem copyFact_of_wiring {ω : ℕ → ℕ} {keep : ℕ → Bool} (W : Wiring h) (hB : BitsInPorts h)
    (hrow : RowFact h ω keep (addressOf h)) (hC : CopiesAt h) : CopyFact h (addressOf h) :=
  fun _ hr _ hi => CheckedRun.copyAt_of_wiring W hB hC hr (hrow _ hr _ hi).1 hi

end SpotFlock
