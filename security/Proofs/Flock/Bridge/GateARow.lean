module

public import Properties.Verity.GateARow
public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Bridge.BitsKeyProg

/-! `Verity.GateARow`, proved (`gateARow`): unit `u`'s gates of the committed circuit are exactly its `nComp` rows. -/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

theorem keyProg_unit_iff (g : Fin (Layout.keyN c hU n)) (u : Fin n) :
    (Placed.keyProg c hU n).unit g = some u ↔
      ∃ i : Fin (Placed.circuitRows c hU).nComp,
        g.val = 2 + n * (Placed.circuitRows c hU).nIn + u.val * (Placed.circuitRows c hU).nComp + i.val := by
  constructor
  · intro h
    by_cases hin : g.val < 2 + n * (Placed.circuitRows c hU).nIn
    · rw [keyProg_unit_input g hin] at h
      cases h
    have hlt : g.val < 2 + n * (Placed.circuitRows c hU).nIn + n * (Placed.circuitRows c hU).nComp := by
      have := g.isLt
      simp only [Layout.keyN, Placed.inGates, Layout.keyB, gates_val] at this
      exact this
    set k := 2 + n * (Placed.circuitRows c hU).nIn
    set m := (Placed.circuitRows c hU).nComp
    have hm : 0 < m := by
      rcases Nat.eq_zero_or_pos m with h0 | h0
      · rw [h0, Nat.mul_zero] at hlt; omega
      · exact h0
    have hd : g.val - k < n * m := by omega
    have hu' : (g.val - k) / m < n := (Nat.div_lt_iff_lt_mul hm).2 hd
    have hg : g.val = k + (g.val - k) / m * m + ((g.val - k) % m) := by
      have := Nat.div_add_mod' (g.val - k) m
      omega
    have h' := keyProg_unit_row hu' ⟨(g.val - k) % m, Nat.mod_lt _ hm⟩ g hg
    rw [h'] at h
    obtain rfl : (⟨(g.val - k) / m, hu'⟩ : Fin n) = u := Option.some.inj h
    exact ⟨⟨(g.val - k) % m, Nat.mod_lt _ hm⟩, hg⟩
  · rintro ⟨i, hi⟩
    exact keyProg_unit_row u.isLt i g hi

theorem keyProg_unit_card (u : Fin n) :
    (Finset.univ.filter fun g : Fin (Layout.keyN c hU n) => (Placed.keyProg c hU n).unit g = some u).card =
      (Placed.circuitRows c hU).nComp := by
  set k := 2 + n * (Placed.circuitRows c hU).nIn
  set m := (Placed.circuitRows c hU).nComp
  have hlt : ∀ i : Fin m, k + u.val * m + i.val < Layout.keyN c hU n := fun i => by
    show k + u.val * m + i.val < Placed.gates (Placed.inGates n (Layout.keyB c hU)) (Placed.circuitRows c hU) n
    rw [gates_val]
    have h1 : u.val * m + i.val < (u.val + 1) * m := by rw [Nat.succ_mul]; omega
    have h2 : (u.val + 1) * m ≤ n * m := Nat.mul_le_mul_right _ u.isLt
    show k + u.val * m + i.val < k + n * m
    omega
  have himg : (Finset.univ.filter fun g : Fin (Layout.keyN c hU n) => (Placed.keyProg c hU n).unit g = some u) =
      Finset.univ.image fun i : Fin m => (⟨k + u.val * m + i.val, hlt i⟩ : Fin (Layout.keyN c hU n)) := by
    ext g
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
    rw [keyProg_unit_iff]
    constructor
    · rintro ⟨i, hi⟩
      exact ⟨i, Fin.ext hi.symm⟩
    · rintro ⟨i, rfl⟩
      exact ⟨i, rfl⟩
  rw [himg, Finset.card_image_of_injective _ fun i j h => Fin.ext (by
    have := congrArg Fin.val h
    simp only at this
    omega), Finset.card_univ, Fintype.card_fin]

theorem gateARow : Verity.GateARow := by
  intro tags cf tables pf r hr
  refine (ofFile_iff hr
    (fun C n P => ∀ u : Fin n, C.size (P.gates u) = (Spot.FilePart.ofCircuit r.D.unroll r.n).m)).2 fun u => ?_
  show (Finset.univ.filter fun g : Fin (Layout.keyN r.D.unroll r.hU r.n) =>
    (Placed.keyProg r.D.unroll r.hU r.n).unit g = some u).card = _
  rw [keyProg_unit_card]
  rfl

end Flock.Bridge
