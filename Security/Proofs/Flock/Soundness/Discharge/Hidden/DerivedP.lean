import Proofs.Flock.Soundness.Discharge.Hidden.PortsP

/-!
# The public inputs the verifier derives itself

* `derivedGroup_some`: a port `HmIn.derivedGroup` derives is one word, seed `instance`, derivation `mod:G` with
  `0 < G ≤ 2^16`: exactly the position `HmIn.checkReads` reads beside a shared read.
* `derivedGroups_some`: with public input ports, each derived, port `k`'s group at `k`.
* `checkDerived_words`: an accepted `HmIn.checkDerived` pins every instance `i`'s word of port `k` to `i mod G_k`.
-/

open Flock

namespace FlockSoundness.Discharge.Hidden

theorem derivedGroup_some {pi : HmIn.Ports} {k g : Nat} (h : HmIn.derivedGroup pi k = some g) :
    (((pi.json.getArrVal? k).toOption.getD Lean.Json.null).getObjValAs? String "seed").toOption = some "instance" ∧
      (((pi.json.getArrVal? k).toOption.getD Lean.Json.null).getObjValAs? String "derivation").toOption =
        some s!"mod:{g}" ∧
      (pi.ports.getD k default).2 = 1 ∧ 0 < g ∧ g ≤ 2 ^ 16 := by
  unfold HmIn.derivedGroup at h
  simp only at h
  split at h
  · exact absurd h (by simp)
  · rename_i hc
    rw [Option.filter_eq_some_iff] at h
    obtain ⟨-, hp⟩ := h
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true', not_or, Classical.not_not,
      Bool.not_eq_false] at hc
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
    obtain ⟨⟨hd, hg0⟩, hg1⟩ := hp
    obtain ⟨⟨hs, hw⟩, -⟩ := hc
    refine ⟨?_, ?_, hw, hg0, hg1⟩
    · revert hs
      cases (((pi.json.getArrVal? k).toOption.getD Lean.Json.null).getObjValAs? String "seed").toOption <;>
        simp
    · revert hd
      cases (((pi.json.getArrVal? k).toOption.getD Lean.Json.null).getObjValAs? String "derivation").toOption
      · intro hd
        exact absurd hd (by simp)
      · simp

theorem derivedGroups_some {pi : HmIn.Ports} {gs : Array Nat} (h : HmIn.derivedGroups pi = some gs) :
    pi.ports.isEmpty = false ∧ gs.size = pi.ports.size ∧
      ∀ k < pi.ports.size, HmIn.derivedGroup pi k = some (gs.getD k 1) := by
  unfold HmIn.derivedGroups at h
  split at h
  · rename_i hc
    obtain rfl := (Option.some.inj h).symm
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true, List.mem_range] at hc
    refine ⟨hc.1, by simp, fun k hk => ?_⟩
    have hs := hc.2 k hk
    have hgd : ((Array.range pi.ports.size).map fun k => (HmIn.derivedGroup pi k).getD 0).getD k 1 =
        (HmIn.derivedGroup pi k).getD 0 := by
      simp [Array.getD_eq_getD_getElem?, hk]
    rw [hgd]
    revert hs
    cases HmIn.derivedGroup pi k <;> simp
  · exact absurd h (by simp)

theorem checkDerived_words {pi : HmIn.Ports} {pop : Public} {gs : Array Nat}
    (h : HmIn.checkDerived pi pop gs = .ok ()) {i k : Nat}
    (hi : i < (pop.population.map (·.1)).getD pop.n) (hk : k < pi.ports.size) :
    ((pop.population.map (·.2.2)).getD pop.outputs).getD (i * pi.ports.size + k) 0 = i % gs.getD k 1 := by
  unfold HmIn.checkDerived at h
  simp only at h
  split at h
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
  · rename_i hn
    have hpos : 0 < pi.ports.size := Nat.lt_of_le_of_lt (Nat.zero_le _) hk
    have hd : (i * pi.ports.size + k) / pi.ports.size = i := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hk, Nat.zero_add]
    have hm : (i * pi.ports.size + k) % pi.ports.size = k := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hk]
    have hx : i * pi.ports.size + k < (pop.population.map (·.1)).getD pop.n * pi.ports.size :=
      Nat.lt_of_lt_of_le (Nat.add_lt_add_left hk _)
        (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hi))
    have hf := List.find?_eq_none.1 (Option.eq_none_iff_forall_ne_some.2 fun x hx' => hn x hx')
      (i * pi.ports.size + k) (List.mem_range.2 hx)
    simpa [hd, hm] using hf

end FlockSoundness.Discharge.Hidden
