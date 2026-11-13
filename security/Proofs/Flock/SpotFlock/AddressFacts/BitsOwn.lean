module

public import Proofs.Flock.SpotFlock.AddressFacts.BitsIn

/-!
# Each source bit inside its own port

`bitsOwn_of_two`: `BitsOwn`'s source half where the second stage's design evaluates (`two`), and its written half
from `WrittenOwn`.

A source bit's commitment position is Bits' (`Spot.Bits.rowBit`): bit `8 · byte + bit` of the row port `p` its key is
a message bit of (`srcPos`). Bits keys an input a message bit only below its port's `bits` (`msgAt`), so `p` is one of
the circuit's ports and the bit is inside its `bits` (`rowBit_bits`). The design evaluates only where every template's
reference statement's ports are rows the hidden-output proof covers (`templateOk`'s `ProvedScope.v2Row`), each of
`16 · words` bits, and that statement's circuit is the member's (`memberCircuit`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-! ## Bits' message bits -/

/-- A leaf word's message bit is inside its port's `bits`. -/
theorem msgAt_bits {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x t : ℕ} {wide : Bool} {p byte bit : ℕ}
    (hm : Spot.Bits.msgAt c ix x t wide = .msg p byte bit) : 8 * byte + bit < (c.ports.getD p default).bits := by
  unfold Spot.Bits.msgAt at hm
  dsimp only at hm
  split at hm
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    injection hm with hp hb hbit
    subst hp hb hbit
    omega
  · cases hm

/-- A unit input's message bit is inside its port's `bits`. -/
theorem srcAt_bits {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {k p' t p byte bit : ℕ}
    (hs : Spot.Bits.srcAt c ix k p' t = .msg p byte bit) : 8 * byte + bit < (c.ports.getD p default).bits := by
  unfold Spot.Bits.srcAt at hs
  dsimp only at hs
  split at hs
  · split at hs
    · cases hs
    · exact msgAt_bits hs
  · split at hs
    · exact msgAt_bits hs
    · split at hs
      · cases hs
      · cases hs

/-- A stacked input's message-bit key is inside its port's `bits`. -/
theorem srcKey_bits {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x p byte bit : ℕ}
    (hs : Spot.Bits.srcKey c ix x = .msg p byte bit) : 8 * byte + bit < (c.ports.getD p default).bits := by
  unfold Spot.Bits.srcKey at hs
  dsimp only at hs
  split at hs
  · exact srcAt_bits hs
  · cases hs

/-- Bits' commitment position of a source is a message bit's: bit `8 · byte + bit` of the row port `p`. -/
theorem srcPos_msg {c : _root_.Flock.Circuit} {u : ℕ} {k : Spot.Bits.Src} {r : Spot.Bits.RowPos}
    (hs : Spot.Bits.srcPos c u k = some r) :
    ∃ p byte bit, k = .msg p byte bit ∧ r.row = (c.ports.getD p default).name ∧ r.bit = 8 * byte + bit := by
  unfold Spot.Bits.srcPos at hs
  split at hs
  · rename_i p byte bit
    cases hs
    exact ⟨p, byte, bit, rfl, rfl, rfl⟩
  · cases hs

/-- **A source gate's commitment position is a bit inside a port of its row's name.** -/
theorem rowBit_bits {c : _root_.Flock.Circuit} {n g : ℕ} {r : Spot.Bits.RowPos}
    (hr : Spot.Bits.rowBit c n g = some r) :
    ∃ p, p < c.ports.size ∧ c.ports[p]!.name = r.row ∧ r.bit < c.ports[p]!.bits := by
  unfold Spot.Bits.rowBit at hr
  dsimp only at hr
  split at hr
  · cases hr
  · obtain ⟨p, byte, bit, hk, hrow, hbit⟩ := srcPos_msg hr
    have hb := srcKey_bits hk
    have hp : p < c.ports.size := by
      by_contra hp
      have h0 : c.ports.getD p default = default := by simp [Array.getD, hp]
      rw [h0] at hb
      exact Nat.not_lt_zero _ hb
    have hg : c.ports.getD p default = c.ports[p]! := by
      rw [getElem!_pos c.ports p hp]
      simp [Array.getD, hp]
    rw [hg] at hb hrow
    exact ⟨p, hp, hrow.symm, by rw [hbit]; exact hb⟩

/-! ## At the held code -/

variable {h : Held}

/-- A source bit is a member's source gate with Bits' commitment position. -/
theorem rowBit_of_mem {x t : ℕ} {r : Spot.Bits.RowPos} (hs : (x, t, r) ∈ sourceBits h.ms) :
    ∃ ht : t < h.ms.length, ∃ g, Spot.Bits.rowBit h.ms[t].1 h.ms[t].2.size g = some r := by
  unfold sourceBits at hs
  obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hs
  have hm' := List.mem_range.1 hm
  obtain ⟨g, -, hgx⟩ := List.mem_filterMap.1 hx
  have hci : h.ms.getD m default = h.ms[m] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
  rw [hci] at hgx
  cases hκ : Spot.Bits.consts h.ms[m].1 h.ms[m].2.size g with
  | some b =>
    rw [hκ] at hgx
    simp at hgx
  | none =>
    rw [hκ] at hgx
    have hgx' : (Spot.Bits.rowBit h.ms[m].1 h.ms[m].2.size g).map (fun r => (Spot.Bits.base h.ms m + g, m, r)) =
        some (x, t, r) := hgx
    obtain ⟨r', hr', he⟩ := Option.map_eq_some_iff.1 hgx'
    have h2 : m = t := congrArg (fun p => p.2.1) he
    have h3 : r' = r := congrArg (fun p => p.2.2) he
    subst h2 h3
    exact ⟨hm', g, hr'⟩

/-- Where the design evaluates, each port of a template's member circuit is a row of `16 · words` bits. -/
theorem bits_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t p : ℕ} (ht : t < h.ms.length)
    (hp : p < (memberCircuit h t).ports.size) :
    (memberCircuit h t).ports[p]!.bits = 16 * (memberCircuit h t).ports[p]!.words := by
  have hok := templateOk_of_two h hd ht
  unfold templateOk at hok
  cases hr : FlockDeploy.readOf true (h.xS₀ t) with
  | none => rw [hr] at hok; cases hok
  | some r =>
    rw [hr] at hok
    simp only [Bool.and_eq_true] at hok
    have hm : memberCircuit h t = r.stmt := by simp [memberCircuit, stmtOf, hr]
    rw [hm] at hp ⊢
    have hv := Array.all_eq_true_iff_forall_mem.1 hok.2 r.stmt.ports[p] (Array.getElem_mem hp)
    rw [getElem!_pos r.stmt.ports p hp]
    simp only [_root_.Flock.ProvedScope.v2Row, Bool.and_eq_true, beq_iff_eq] at hv
    exact hv.1.1.2

variable (h) in
/-- **Each written bit's row is a port of its member's circuit and its bit inside that port's `16 w` bits**:
`BitsOwn`'s written half. -/
def WrittenOwn : Prop :=
  ∀ y ∈ writtenBits h.ms, ∃ q, q < (memberCircuit h y.2.1).ports.size ∧
    (memberCircuit h y.2.1).ports[q]!.name = y.2.2.row ∧
    y.2.2.bit < 16 * (memberCircuit h y.2.1).ports[q]!.words

/-- **`BitsOwn` where the design evaluates**, its written half from `WrittenOwn`. -/
theorem bitsOwn_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hW : WrittenOwn h) :
    BitsOwn h := by
  refine ⟨?_, hW⟩
  rintro ⟨x, t, r⟩ hs
  obtain ⟨ht, g, hg⟩ := rowBit_of_mem hs
  obtain ⟨p, hp, hn, hb⟩ := rowBit_bits hg
  have hm : memberCircuit h t = h.ms[t].1 := (by simp [Held.ms] : h.ms[t].1 = stmtOf (h.xS₀ t)).symm
  have hp' : p < (memberCircuit h t).ports.size := by rw [hm]; exact hp
  have h16 := bits_of_two hd ht hp'
  refine ⟨p, hp', ?_, ?_⟩
  · show (memberCircuit h t).ports[p]!.name = r.row
    rw [hm]
    exact hn
  · show r.bit < 16 * (memberCircuit h t).ports[p]!.words
    rw [← h16, hm]
    exact hb

/-- **`BitsInPorts` where the design evaluates**, each member naming its ports once (`PortsOnce`) and each written
bit inside its own port (`WrittenOwn`). -/
theorem bitsInPorts_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hO : PortsOnce h)
    (hW : WrittenOwn h) : BitsInPorts h :=
  bitsInPorts_of hO (bitsOwn_of_two hd hW)

end SpotFlock
