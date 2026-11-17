module

public import Proofs.Flock.SpotFlock.Honest.Any

/-! The two stages check a proof block at every play the verifier accepts (`checked_ne_nil_of_template`), where the
registration has a template, which is a hypothesis. -/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot

theorem of_not_bnot {b : Bool} (h : ¬ (!b) = true) : b = true := by
  cases b
  · exact absurd rfl h
  · rfl

theorem secondProblem_cells_pos {sample ns ks : Array ℕ}
    (h : _root_.Spot.Select.secondProblem sample ns ks = none) : 0 < ns.size := by
  unfold _root_.Spot.Select.secondProblem at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hns
      refine Nat.pos_of_ne_zero fun h0 => ?_
      have hb : ns.isEmpty = true := Array.isEmpty_iff_size_eq_zero.mpr h0
      exact hns (Bool.or_eq_true_iff.2 (Or.inl (Bool.or_eq_true_iff.2 (Or.inl hb))))

theorem secondProblem_counts_pos {sample ns ks : Array ℕ}
    (h : _root_.Spot.Select.secondProblem sample ns ks = none) : 0 < ks.toList.sum := by
  have hsz : ks.size = ns.size := Properties.Spot.SecurityProofs.secondProblem_size h
  have hne : ns.size ≠ 0 := Nat.pos_iff_ne_zero.1 (secondProblem_cells_pos h)
  unfold _root_.Spot.Select.secondProblem at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · rename_i hall
          have hall' := of_not_bnot hall
          cases hkl : ks.toList with
          | nil =>
            have hlen := congrArg List.length hkl
            rw [Array.length_toList, List.length_nil] at hlen
            exact absurd (hsz.symm.trans hlen) hne
          | cons k ks' =>
            cases hnl : ns.toList with
            | nil =>
              have hlen := congrArg List.length hnl
              rw [Array.length_toList, List.length_nil] at hlen
              exact absurd hlen hne
            | cons n ns' =>
              rw [hkl, hnl, List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at hall'
              have hf : (decide (1 ≤ k) && decide (k ≤ n)) = true := hall'.1
              have hk1 : 1 ≤ k := of_decide_eq_true (Bool.and_eq_true_iff.1 hf).1
              rw [List.sum_cons]
              omega

theorem checked_of_proved (g : RegTwo) {N : ℕ} {j : ℤ} {src₁ src₂ : ByteArray} {draw : Lean.Json} {read : ℕ}
    (h : _root_.Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, read))
    (hj : j ∈ _root_.Spot.Draw.proved draw) (hv : g.blockAt (g.sample src₁) j.toNat < N) :
    g.blockAt (g.sample src₁) j.toNat ∈ g.checked N src₁ src₂ := by
  unfold RegTwo.checked
  rw [h]
  refine List.mem_filter.2 ⟨List.mem_range.2 hv, ?_⟩
  rw [Bool.or_eq_true, Array.any_eq_true']
  exact Or.inr ⟨j, hj, by simp⟩

theorem sampled_block {h : Held} {s : AuditStatement} {d : Design.Two} (h2 : two h s = some d)
    (hSS : AddressRow.SampleSlots h) {b₁ b₂ : ByteArray} {draw : Lean.Json} {read : ℕ}
    (hsel : _root_.Spot.Select.selectSecond (h.reg.sample b₁) h.reg.ns h.reg.ks b₂ = .ok (draw, read)) :
    ∃ j ∈ _root_.Spot.Draw.proved draw,
      h.reg.blockAt (h.reg.sample b₁) j.toNat < (membersPart h.ms).blocks.length := by
  have hq := (Properties.Spot.SecurityProofs.selectSecond_units hsel).1
  have hs0 : 0 < (h.reg.sample b₁).size := by
    have hne := (Properties.Spot.SecurityProofs.secondProblem_sample hq).1
    refine Nat.pos_of_ne_zero fun h0 => ?_
    exact absurd ((Array.isEmpty_iff_size_eq_zero.mpr h0).symm.trans hne) (by decide)
  have hpos : 0 < (_root_.Spot.Draw.proved draw).size := by
    rw [Properties.Spot.SecurityProofs.size_proved_selectSecond hsel]
    exact Nat.mul_pos hs0 (secondProblem_counts_pos hq)
  obtain ⟨j, hj⟩ : ∃ j, j ∈ _root_.Spot.Draw.proved draw := ⟨_, Array.getElem_mem hpos⟩
  obtain ⟨t, i, p, ht, hi, hpn, hjt⟩ := AddressRow.proved_in_cell hsel hj
  have htm : t < h.ms.length := by
    obtain ⟨l, -, -, -, -, hfo⟩ := two_eq h h2
    have hT := (filesOk_sound hfo).1
    omega
  obtain ⟨hb, hlt⟩ := AddressRow.blockAt_slot h2 hSS b₁ htm hi hpn
  refine ⟨j, hj, ?_⟩
  rw [hjt, hb]
  exact _root_.Flock.Bridge.Members.unitBase_add_lt h.ms htm hlt

theorem units_pos_of_sampled {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) (hSS : AddressRow.SampleSlots h) {b₁ b₂ : ByteArray} {draw : Lean.Json} {read : ℕ}
    (hsel : _root_.Spot.Select.selectSecond (h.reg.sample b₁) h.reg.ns h.reg.ks b₂ = .ok (draw, read)) :
    0 < units (base h) x := by
  obtain ⟨j, -, hlt⟩ := sampled_block h2 hSS hsel
  have hP : partition h s = some (membersPart h.ms) := rfl
  rw [units_base h hp hP]
  exact Nat.lt_of_le_of_lt (Nat.zero_le _) hlt

theorem checked_ne_nil_of_sampled {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) (hSS : AddressRow.SampleSlots h) {b₁ b₂ : ByteArray} {draw : Lean.Json} {read : ℕ}
    (hsel : _root_.Spot.Select.selectSecond (h.reg.sample b₁) h.reg.ns h.reg.ks b₂ = .ok (draw, read)) :
    h.reg.checked (units (base h) x) b₁ b₂ ≠ [] := by
  obtain ⟨j, hj, hlt⟩ := sampled_block h2 hSS hsel
  have hP : partition h s = some (membersPart h.ms) := rfl
  rw [← units_base h hp hP] at hlt
  exact List.ne_nil_of_mem (checked_of_proved h.reg hsel hj hlt)

theorem checked_ne_nil {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) (hSS : AddressRow.SampleSlots h) (hN : 0 < units (base h) x)
    (c₀ : FlockDeploy.Coin) (c₁ : Stage h) :
    h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) ≠ [] := by
  cases hsel : _root_.Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (stageSrc c₁) with
  | error e => exact List.ne_nil_of_mem (RegTwo.error_mem h.reg hsel hN)
  | ok dr =>
    obtain ⟨draw, read⟩ := dr
    exact checked_ne_nil_of_sampled hp h2 hSS hsel

theorem checked_ne_nil_of_play {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hSS : AddressRow.SampleSlots h) (hN : 0 < units (base h) x) :
    h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) ≠ [] := by
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo h H x Ω P ω r hacc
  have hp' : parse h x = some s := hp
  exact checked_ne_nil hp' h2 hSS hN _ _

theorem template_of_cells {h : Held} {s : AuditStatement} {d : Design.Two} (h2 : two h s = some d)
    (hpos : 0 < h.reg.ns.size) : 0 < h.ms.length := by
  obtain ⟨l, -, -, -, -, hfo⟩ := two_eq h h2
  have hT := (filesOk_sound hfo).1
  omega

theorem sampled_of_play {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hm : 0 < h.ms.length) :
    ∃ draw read, _root_.Spot.Select.selectSecond (h.reg.sample (src (r ⟨0, k_pos h x⟩))) h.reg.ns h.reg.ks
      (stageSrc (stageCoins h x r)) = .ok (draw, read) := by
  obtain ⟨m₀, m₁, m₂, rest, -, hok⟩ := run_seenAt h H P x ω r hacc
  obtain ⟨draw, hd⟩ := stmtOk_sample h hok hm
  obtain ⟨read, hs⟩ := AddressRow.select_of_secondSample h hd
  exact ⟨draw, read, hs⟩

theorem units_pos_of_play {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hSS : AddressRow.SampleSlots h) (hm : 0 < h.ms.length) : 0 < units (base h) x := by
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo h H x Ω P ω r hacc
  have hp' : parse h x = some s := hp
  obtain ⟨draw, read, hsel⟩ := sampled_of_play H hacc hm
  exact units_pos_of_sampled hp' h2 hSS hsel

theorem checked_ne_nil_of_template {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hSS : AddressRow.SampleSlots h) (hm : 0 < h.ms.length) :
    h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) ≠ [] := by
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo h H x Ω P ω r hacc
  have hp' : parse h x = some s := hp
  obtain ⟨draw, read, hsel⟩ := sampled_of_play H hacc hm
  exact checked_ne_nil_of_sampled hp' h2 hSS hsel

theorem checked_ne_nil_of_firstK {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hm : 0 < h.ms.length) :
    h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) ≠ [] :=
  checked_ne_nil_of_template H hacc (AddressRow.sampleSlots_of_firstK hp hK) hm

end SpotFlock.Honest
