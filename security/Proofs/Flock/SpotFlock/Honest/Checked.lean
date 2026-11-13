module

public import Proofs.Flock.SpotFlock.Honest.Any

/-!
# The two stages check a proof block at every accepted play

`AddressesConnected` speaks of the proof blocks the two stages check (`RegTwo.checked`), and holds vacuously where
they check none. At an auditor's copies `h` that have a template and whose first stage samples slots that name
themselves (`AddressRow.SampleSlots`), they check one at the statement and the coins of every play the verifier
accepts:

* `secondProblem_cells_pos`, `secondProblem_counts_pos`: a second stage `secondProblem` passes has a cell, and
  samples a proof block of each sampled replay block, since its counts are one per template, of one template or
  more, each at least `1`;
* `checked_of_proved`: a proof block the second stage proves is checked, where it is below the statement's count;
* `drawn_unit`: a second stage that draws proves a proof block of a member. Its replay sample is not empty, so it
  proves `|sample| · Σ_t k_t ≥ 1` positions (`size_proved_selectSecond`), each a position of a template's cell at a
  sampled replay block (`AddressRow.proved_in_cell`), which the registration maps to a unit the template's member
  has (`AddressRow.blockAt_slot`), below the members' units (`unitBase_add_lt`);
* `units_pos_of_drawn`, `checked_ne_nil_of_drawn`: so where the second stage draws, the statement has a proof block
  and the two stages check one;
* `checked_ne_nil`, `checked_ne_nil_of_play`: on any coins, at a statement that has a proof block: where the second
  stage is refused, the checked proof blocks are every proof block;
* `drawn_of_play`: at the coins of a play the verifier accepts, the second stage draws, where the registration has a
  template: the accept passed the session statements' check, whose check of the first template's statement reads the
  draw (`stmtOk_draw`);
* `units_pos_of_play`, `checked_ne_nil_of_template`: so the accepted play's statement has a proof block and the two
  stages check one, with nothing asked of the statement's count;
* `checked_ne_nil_of_firstK`: the same, with `AddressRow.SampleSlots` from the run's population check and a first
  stage of a count `K` (`AddressRow.sampleSlots_of_firstK`), the facts `premises_of_checks` takes;
* `template_of_cells`: a registration with a cell has a template, where the design evaluates.

## What this does not show

That the registration has a template (`0 < h.ms.length`): it is a hypothesis. With no template the statement's count
of proof blocks (`units (base h) x`), which is the members' units, is zero, and the two stages check none. The count
is a fact of the held copies and not of the play.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot

/-- A Boolean whose negation is not `true` is `true`. -/
theorem of_not_bnot {b : Bool} (h : ¬ (!b) = true) : b = true := by
  cases b
  · exact absurd rfl h
  · rfl

/-- **A second stage `secondProblem` passes has a cell**: a template's count of proof blocks. -/
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

/-- **A second stage `secondProblem` passes samples a proof block of each sampled replay block**: its counts are one
per template, of one template or more, each at least `1`, so their sum is positive. -/
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

/-- **A proof block the second stage proves is checked**, where it is below `N`: `RegTwo.ok_mem` at a proved position
as the draw holds it, an integer. -/
theorem checked_of_proved (g : RegTwo) {N : ℕ} {j : ℤ} {src₁ src₂ : ByteArray} {draw : Lean.Json} {read : ℕ}
    (h : _root_.Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, read))
    (hj : j ∈ _root_.Spot.Draw.proved draw) (hv : g.blockAt (g.sample src₁) j.toNat < N) :
    g.blockAt (g.sample src₁) j.toNat ∈ g.checked N src₁ src₂ := by
  unfold RegTwo.checked
  rw [h]
  refine List.mem_filter.2 ⟨List.mem_range.2 hv, ?_⟩
  rw [Bool.or_eq_true, Array.any_eq_true']
  exact Or.inr ⟨j, hj, by simp⟩

/-- **A drawn second stage proves a proof block of a member**: where the design evaluates and the first stage samples
slots that name themselves, a second stage that draws proves a position the registration maps below the members'
units. The replay sample is not empty and the counts sum to at least `1` (`secondProblem`), so the stage proves a
position (`size_proved_selectSecond`), one of a template's cell at a sampled replay block
(`AddressRow.proved_in_cell`), which is a unit the template's member has (`AddressRow.blockAt_slot`). -/
theorem drawn_unit {h : Held} {s : AuditStatement} {d : Design.Two} (h2 : two h s = some d)
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

/-- **A statement has a proof block where the second stage draws**: the drawn stage proves a unit of a member
(`drawn_unit`), and the statement's proof blocks are the members' units. -/
theorem units_pos_of_drawn {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) (hSS : AddressRow.SampleSlots h) {b₁ b₂ : ByteArray} {draw : Lean.Json} {read : ℕ}
    (hsel : _root_.Spot.Select.selectSecond (h.reg.sample b₁) h.reg.ns h.reg.ks b₂ = .ok (draw, read)) :
    0 < units (base h) x := by
  obtain ⟨j, -, hlt⟩ := drawn_unit h2 hSS hsel
  have hP : partition h s = some (membersPart h.ms) := rfl
  rw [units_base h hp hP]
  exact Nat.lt_of_le_of_lt (Nat.zero_le _) hlt

/-- **The two stages check a proof block where the second stage draws**, at a statement that parses, where the
design evaluates and the first stage samples slots that name themselves: the proof block the drawn stage proves
(`drawn_unit`). It asks nothing of the statement's count. -/
theorem checked_ne_nil_of_drawn {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) (hSS : AddressRow.SampleSlots h) {b₁ b₂ : ByteArray} {draw : Lean.Json} {read : ℕ}
    (hsel : _root_.Spot.Select.selectSecond (h.reg.sample b₁) h.reg.ns h.reg.ks b₂ = .ok (draw, read)) :
    h.reg.checked (units (base h) x) b₁ b₂ ≠ [] := by
  obtain ⟨j, hj, hlt⟩ := drawn_unit h2 hSS hsel
  have hP : partition h s = some (membersPart h.ms) := rfl
  rw [← units_base h hp hP] at hlt
  exact List.ne_nil_of_mem (checked_of_proved h.reg hsel hj hlt)

/-- **The two stages check a proof block**, at a statement that parses and has one, where the design evaluates and
the first stage samples slots that name themselves, on any coins. Where the second stage is refused, every proof
block is checked, and this is where the statement's having one is used. Where it draws, the proof block it proves is
checked (`checked_ne_nil_of_drawn`). -/
theorem checked_ne_nil {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) (hSS : AddressRow.SampleSlots h) (hN : 0 < units (base h) x)
    (c₀ : FlockDeploy.Coin) (c₁ : Stage h) :
    h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) ≠ [] := by
  cases hsel : _root_.Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (stageSrc c₁) with
  | error e => exact List.ne_nil_of_mem (RegTwo.error_mem h.reg hsel hN)
  | ok dr =>
    obtain ⟨draw, read⟩ := dr
    exact checked_ne_nil_of_drawn hp h2 hSS hsel

/-- **The two stages check a proof block at a play the verifier accepts**, at its statement and its coins, where the
statement has a proof block and the first stage samples slots that name themselves: the accept passed the session
statements' check, at which the statement parses and the design evaluates (`refusesTwo`). -/
theorem checked_ne_nil_of_play {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hSS : AddressRow.SampleSlots h) (hN : 0 < units (base h) x) :
    h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) ≠ [] := by
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo h H x Ω P ω r hacc
  have hp' : parse h x = some s := hp
  exact checked_ne_nil hp' h2 hSS hN _ _

/-! ## Where the registration has a template, the accepted play's second stage draws -/

/-- **A registration with a cell has a template**, where the design evaluates: its cells are one per template
(`filesOk`). -/
theorem template_of_cells {h : Held} {s : AuditStatement} {d : Design.Two} (h2 : two h s = some d)
    (hpos : 0 < h.reg.ns.size) : 0 < h.ms.length := by
  obtain ⟨l, -, -, -, -, hfo⟩ := two_eq h h2
  have hT := (filesOk_sound hfo).1
  omega

/-- **The second stage draws at the coins of a play the verifier accepts**, where the registration has a template:
the accept passed the session statements' check (`run_seenAt`), whose check of the first template's statement reads
the second stage's draw (`stmtOk_draw`). -/
theorem drawn_of_play {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hm : 0 < h.ms.length) :
    ∃ draw read, _root_.Spot.Select.selectSecond (h.reg.sample (src (r ⟨0, k_pos h x⟩))) h.reg.ns h.reg.ks
      (stageSrc (stageCoins h x r)) = .ok (draw, read) := by
  obtain ⟨m₀, m₁, m₂, rest, -, hok⟩ := run_seenAt h H P x ω r hacc
  obtain ⟨draw, hd⟩ := stmtOk_draw h hok hm
  obtain ⟨read, hs⟩ := AddressRow.select_of_secondDraw h hd
  exact ⟨draw, read, hs⟩

/-- **The statement of a play the verifier accepts has a proof block**, where the registration has a template and the
first stage samples slots that name themselves: the play's second stage draws (`drawn_of_play`), and a drawn stage
proves a unit of a member (`units_pos_of_drawn`). -/
theorem units_pos_of_play {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hSS : AddressRow.SampleSlots h) (hm : 0 < h.ms.length) : 0 < units (base h) x := by
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo h H x Ω P ω r hacc
  have hp' : parse h x = some s := hp
  obtain ⟨draw, read, hsel⟩ := drawn_of_play H hacc hm
  exact units_pos_of_drawn hp' h2 hSS hsel

/-- **The two stages check a proof block at a play the verifier accepts**, at its statement and its coins, where the
registration has a template and the first stage samples slots that name themselves. It asks nothing of the
statement's count: the play's second stage draws (`drawn_of_play`), and the proof block it proves is checked
(`checked_ne_nil_of_drawn`). -/
theorem checked_ne_nil_of_template {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc)
    (hSS : AddressRow.SampleSlots h) (hm : 0 < h.ms.length) :
    h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) ≠ [] := by
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo h H x Ω P ω r hacc
  have hp' : parse h x = some s := hp
  obtain ⟨draw, read, hsel⟩ := drawn_of_play H hacc hm
  exact checked_ne_nil_of_drawn hp' h2 hSS hsel

/-- **The same, from the facts `premises_of_checks` takes**: the boundary registration passes the run's population
check at `n` units and its first stage samples a uniform `K`-subset, so it samples slots that name themselves
(`AddressRow.sampleSlots_of_firstK`). -/
theorem checked_ne_nil_of_firstK {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hm : 0 < h.ms.length) :
    h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) ≠ [] :=
  checked_ne_nil_of_template H hacc (AddressRow.sampleSlots_of_firstK hp hK) hm

end SpotFlock.Honest
