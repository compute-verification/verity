module

public import Proofs.Flock.SpotFlock.Frame
public import Proofs.Flock.SpotFlock.FrameIsBuilt
public import Proofs.Flock.SpotFlock.KS

/-!
# No proof block is always checked

A two-stage registration names proof blocks checked whatever the stages sample (`RegTwo.always`), and the code's
checked units are those and the ones the second stage draws (`RegTwo.checked`). The field is the auditor's, a part of
the model's registration and of nothing the verifier runs: the verifier's step function, its parser, partition
evaluation, design, sessions and base code are the same at the registration with the field emptied (`Held.noAlways`:
`V_noAlways` and the lemmas beside it), so no prover sets it and no check of the verifier has to bound it.

What makes SPOT[Flock]'s two-stage verifier knowledge sound at a registration makes it so at the emptied one
(`ksTwo_noAlways`, from `sessionOpensTwo_noAlways`): the emptied registration's checked units are among the
registration's own (`code_noAlways_checked_sub`), so the frame and the layout that serve the one serve the other
(`frameIs_noAlways`, `embeds_noAlways`), and the addresses premise only weakens (`addresses_noAlways`). There the
checked units are the drawn ones alone (`RegTwo.mem_noAlways_checked`, `code_noAlways_checked_drawn`): the gates they
need number at most the drawn blocks' own (`code_card_biUnion_noAlways_le`, `code_card_biUnion_noAlways_le_mul`), with
no term for always-checked blocks, and the replay cap is the `K · Σ_t n_t` proof blocks of `K` replay blocks
(`cap_noAlways`).
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling

/-- **The registration with no always-checked proof block**: every other field its own. -/
def RegTwo.noAlways (g : RegTwo) : RegTwo := { g with always := fun _ => false }

/-- **The emptied registration's checked proof blocks are the drawn ones**, where the second stage draws: those below
`N` at a position the second stage proves. -/
theorem RegTwo.mem_noAlways_checked (g : RegTwo) {N u : ℕ} {src₁ src₂ : ByteArray} {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) :
    u ∈ g.noAlways.checked N src₁ src₂ ↔
      u < N ∧ ∃ j ∈ Spot.Draw.proved draw, g.blockAt (g.sample src₁) j.toNat = u := by
  have hsel' : Spot.Select.selectSecond (g.noAlways.sample src₁) g.noAlways.ns g.noAlways.ks src₂ =
      .ok (draw, n) := hsel
  unfold RegTwo.checked
  rw [hsel']
  show u ∈ (List.range N).filter (fun v => false ||
    (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v) ↔ _
  constructor
  · intro hu
    obtain ⟨hN, hor⟩ := List.mem_filter.1 hu
    refine ⟨List.mem_range.1 hN, ?_⟩
    rcases Bool.or_eq_true_iff.1 hor with hal | hany
    · exact absurd hal (by decide)
    · obtain ⟨j, hj, hjv⟩ := Array.any_eq_true'.1 hany
      exact ⟨j, hj, beq_iff_eq.1 hjv⟩
  · rintro ⟨hN, j, hj, rfl⟩
    refine List.mem_filter.2 ⟨List.mem_range.2 hN, ?_⟩
    rw [Bool.or_eq_true, Array.any_eq_true']
    exact Or.inr ⟨j, hj, beq_iff_eq.2 rfl⟩

/-- **The emptied registration's checked proof blocks are among the registration's own.** -/
theorem RegTwo.noAlways_checked_sub (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) :
    ∀ u ∈ g.noAlways.checked N src₁ src₂, u ∈ g.checked N src₁ src₂ := by
  intro u hu
  cases hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ with
  | error e => exact g.error_mem hsel (SpotFlock.mem_checked_lt hu)
  | ok p =>
    obtain ⟨draw, n⟩ := p
    obtain ⟨hN, j, hj, rfl⟩ := (g.mem_noAlways_checked hsel).1 hu
    unfold RegTwo.checked
    rw [hsel]
    refine List.mem_filter.2 ⟨List.mem_range.2 hN, ?_⟩
    rw [Bool.or_eq_true, Array.any_eq_true']
    exact Or.inr ⟨j, hj, beq_iff_eq.2 rfl⟩

end SamplingDeploy

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **The auditor's copies with no always-checked proof block**: the registration emptied of them
(`RegTwo.noAlways`), every other copy its own. -/
def Held.noAlways (h : Held) : Held := { h with reg := h.reg.noAlways }

variable (h : Held)

theorem noAlways_always (v : ℕ) : h.noAlways.reg.always v = false := rfl

/-! ## The verifier doesn't read the field -/

/-- **The verifier's step function is the same.** -/
theorem V_noAlways : V h.noAlways = V h := rfl

theorem parse_noAlways : parse h.noAlways = parse h := rfl

theorem partition_noAlways : partition h.noAlways = partition h := rfl

theorem two_noAlways : two h.noAlways = two h := rfl

theorem session_noAlways : session h.noAlways = session h := rfl

/-- **The code before its samples is the same**: with it the reveal, the frame, the sessions and their counts. -/
theorem base_noAlways : base h.noAlways = base h := rfl

/-! ## The checked units -/

/-- **The emptied registration's code checks the emptied registration's proof blocks**, on the bytes of the two stages'
coins. -/
theorem code_noAlways_checked_reg (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) :
    ∀ u ∈ (code h.noAlways).checked x r, u ∈ h.noAlways.reg.checked
      (SamplingDeploy.units ((base h).atOrder (order h)) x) (src (r (order h x).first))
      (src (r (order h x).second)) := by
  intro u hu
  change u ∈ ((reg h.noAlways x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
    (src (r (order h x).first)) (src (r (order h x).second))).getD [] at hu
  cases hd : decode x with
  | none => simp [reg, hd] at hu
  | some _ => simpa [reg, hd] using hu

/-- **The emptied registration's code checks only units the registration's code checks.** -/
theorem code_noAlways_checked_sub (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) :
    ∀ u ∈ (code h.noAlways).checked x r, u ∈ (code h).checked x r := by
  intro u hu
  have hsub : u ∈ h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
      (src (r (order h x).first)) (src (r (order h x).second)) :=
    h.reg.noAlways_checked_sub _ _ _ u (code_noAlways_checked_reg h x r u hu)
  change u ∈ ((reg h.noAlways x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
    (src (r (order h x).first)) (src (r (order h x).second))).getD [] at hu
  change u ∈ ((reg h x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
    (src (r (order h x).first)) (src (r (order h x).second))).getD []
  cases hd : decode x with
  | none => simp [reg, hd] at hu
  | some _ => simpa [reg, hd] using hsub

/-- **Where the second stage draws, the emptied registration's code checks drawn proof blocks only**: each checked
unit is the proof block at a position the second stage proves. -/
theorem code_noAlways_checked_drawn (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin)
    {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks
      (src (r (order h x).second)) = .ok (draw, n)) :
    ∀ u ∈ (code h.noAlways).checked x r, ∃ j ∈ Spot.Draw.proved draw,
      h.reg.blockAt (h.reg.sample (src (r (order h x).first))) j.toNat = u :=
  fun u hu => ((h.reg.mem_noAlways_checked hsel).1 (code_noAlways_checked_reg h x r u hu)).2

/-- **K at the emptied registration's code**: where the second stage draws, the gates the checked proof blocks need
number at most each drawn block's, summed over the drawn blocks. -/
theorem code_card_biUnion_noAlways_le (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin)
    (needs : ℕ → Finset ℕ) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks
      (src (r (order h x).second)) = .ok (draw, n)) :
    (((code h.noAlways).checked x r).toFinset.biUnion needs).card ≤
      ∑ v ∈ ((Spot.Draw.proved draw).toList.map fun j =>
        h.reg.blockAt (h.reg.sample (src (r (order h x).first))) j.toNat).toFinset, (needs v).card := by
  refine (Finset.card_le_card (Finset.biUnion_subset_biUnion_of_subset_left needs fun u hu => ?_)).trans
    Finset.card_biUnion_le
  obtain ⟨j, hj, rfl⟩ := code_noAlways_checked_drawn h x r hsel u (List.mem_toFinset.1 hu)
  exact List.mem_toFinset.2 (List.mem_map.2 ⟨j, Array.mem_toList_iff.2 hj, rfl⟩)

/-- **K at the emptied registration's code, by the most a block needs**: at most the drawn blocks' count times the
most gates a block needs. -/
theorem code_card_biUnion_noAlways_le_mul (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin)
    (needs : ℕ → Finset ℕ) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks
      (src (r (order h x).second)) = .ok (draw, n)) (M : ℕ) (hM : ∀ u, (needs u).card ≤ M) :
    (((code h.noAlways).checked x r).toFinset.biUnion needs).card ≤ (Spot.Draw.proved draw).size * M := by
  refine (code_card_biUnion_noAlways_le h x r needs hsel).trans ?_
  refine (Finset.sum_le_card_nsmul _ _ M fun v _ => hM v).trans ?_
  rw [smul_eq_mul]
  refine Nat.mul_le_mul_right M ((List.toFinset_card_le _).trans ?_)
  rw [List.length_map, Array.length_toList]

/-- **The replay cap at the emptied registration**: the `K · Σ_t n_t` proof blocks of `K` replay blocks where the
first stage samples a uniform `K`-subset, never more than every proof block; every proof block otherwise. -/
theorem cap_noAlways (x : Msg) :
    cap h.noAlways x = (firstK h.reg.first).elim (nBlocks h) fun K => min (nBlocks h) (K * perBlock h) := by
  have h0 : ((Finset.range (nBlocks h.noAlways)).filter fun v => h.noAlways.reg.always v = true) = ∅ :=
    Finset.filter_false_of_mem fun v _ hv => absurd (show false = true from hv) (by decide)
  have hc : cap h.noAlways x = (firstK h.noAlways.reg.first).elim (nBlocks h.noAlways)
      fun K => min (nBlocks h.noAlways) (K * perBlock h.noAlways) := by
    unfold cap
    rw [h0, Finset.card_empty]
    cases firstK h.noAlways.reg.first <;> rfl
  exact hc

/-! ## Knowledge soundness at the emptied registration -/

/-- **A frame that opens the session's rows at the registration opens them at the emptied one**, at the same
layout. -/
theorem frameIs_noAlways {L : Layout} (hF : FrameIs h L) : FrameIs h.noAlways L := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu
  exact hF x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok
    (h.reg.noAlways_checked_sub _ _ _ u hu)

/-- **A layout that holds each checked unit's gates at the registration holds them at the emptied one.** -/
theorem embeds_noAlways {L : Layout} (he : Embeds h L) : Embeds h.noAlways L := by
  intro x s Pt d m₁ m₂ c₀ c₁ u b ρ rb S hx hP h2 hin hok hu
  exact he x s Pt d m₁ m₂ c₀ c₁ u b ρ rb S hx hP h2 hin hok (h.reg.noAlways_checked_sub _ _ _ u hu)

/-- **The sessions' openings at the emptied registration**, from the registration's own frame and addresses, at the
registration's layout. -/
theorem sessionOpensTwo_noAlways (hB : Built h) (hS : Addresses h) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (deploy _root_.Flock.Readable.relation (code h.noAlways)) (two h.noAlways) iroots :=
  sessionOpensTwo_of_frameIs h.noAlways (layoutFor h) (embeds_noAlways h (embeds_layoutFor h))
    (frameIs_noAlways h (frameIs_built h hB hS))

/-- **SPOT[Flock]'s two-stage knowledge soundness at the emptied registration**, from the registration's own frame
and addresses: the verifier, statement, partition, design and reveal are the registration's (`V_noAlways`,
`base_noAlways`), the checked units the drawn ones alone (`code_noAlways_checked_drawn`), and the inner count the
replay cap with no always-checked block (`cap_noAlways`). -/
theorem ksTwo_noAlways (hB : Built h) (hS : Addresses h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h.noAlways) iroots
      (two h.noAlways) (rs h.noAlways) (cap h.noAlways)).KS FlockDeploy.sha512 :=
  ksTwo_spot h.noAlways (sessionOpensTwo_noAlways h hB hS)

/-! ## The addresses premise only weakens -/

/-- **An accepted run's checked unit at the emptied registration is one at the registration.** -/
theorem CheckedRun.ok_noAlways {r : CheckedRun} (hr : r.Ok h.noAlways) : r.Ok h := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hrest⟩ := hr
  exact ⟨Pt, d, spec, hx, hP, h2, hin, hok, h.reg.noAlways_checked_sub _ _ _ _ hu, hrest⟩

/-- **The addresses at the registration are addresses at the emptied one**: the same address holds at each of its
runs, which are runs of the registration. -/
theorem addresses_noAlways (hS : Addresses h) : Addresses h.noAlways := fun g =>
  (hS g).imp fun _ hA r hr hg => hA r (CheckedRun.ok_noAlways h hr) hg

end SpotFlock
