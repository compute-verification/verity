module

public import Proofs.Slowdown.Secure
public import Proofs.ResourceAccounting.Actual.Device
public import Proofs.Pouw.PerRunStop
public import Proofs.Poub.BandwidthAudit
public import Proofs.Pous.Period
import Proofs.Pouw.PerRunJoint
import Proofs.Pouw.PerRunPriced
import Proofs.Pouw.Prob
public import Proofs.Core.Game.Draw

/-!
# The audits fill `RunSecure`'s slots

Each audit's per-run guarantee is soundness for its clause of `𝒜` (`Verity.Sound`), at the run of the audits' coins,
one uniform coin (`Game.draw`): the slot `RunSecure` leaves for it. At each coin the computation is a run of the
actual-circuit model, its declared circuit's gates those charged to an account of the audit's, and the joint audit's
schedule is the one the run induces on the device (`Actual.Run.schedule`). A slot is filled at any frame, at a window
that is the audit's window on the audit's device.

* `work_slot`: the compute audit's guarantee on the priced sum at certificate `W` fills work's slot (`WorkAtLeast`)
  at `⌈W − pRead · read⌉`, `read` the bits the compute account read from device memory on that run; `w1_slot` is
  Pearl-C's audit at W1 priced, from the gap property per coin (`w1Stop_computeSoundPerRunPriced`).
* `held_slot`: the capacity audit's guarantee at certificate `S` over one tick fills memory's slot (`HeldAtLeast`) at
  `⌈S⌉`, at a window whose cut is the end of that tick: the schedule charges the capacity account what the device
  holds there (`mem_le_state`). `pous_slot` is PoUS's period, from `periodCapacity`.
* `delivered_slot`: the bandwidth audit's guarantee at certificate `Q` fills bandwidth's slot (`DeliveredAtLeast`) at
  `⌈Q⌉`, on the chip alone, whose memory link ends at a leaf of the chip with its other end outside it. `poub_slot`
  is PoUB's audit, from `BandwidthOfAuditPerRun`.
* The count's slot (`DeliveredAtMost`) is filled with no audit at the links' capacity `b ω`
  (`delivered_le_capacity`), and a slot at the amount 0 by anything (`sound_of_imp`).

`RunSecure` at these slots, read back as a bound on the coins' probability (`pr_le_of_sound_draw`), on a frame of one
server over the ticks `[0, t)`, one window (`wholeCuts`) or two (`cutAt`):

* `secure_compute`: the device `chip ∪ mem` over one window that the memory and bandwidth audits miss.
  The probability that the compute audit accepts a run holding a reference training algorithm of more than
  `Φ t − ⌈W − pRead · read⌉ + o_w` counted multiplications is at most `ε`.
* `secure_bandwidth`: the chip over one covered window, with no memory audit. The count is the fit
  bound at `S' = (m + o_s) + (b t − ⌈Q⌉ + o_b)`.
* `secure_capacity`: the device cut at the end of the capacity audit's tick `c`; the window before the
  cut is missed and charged the device's whole work, the window after it is covered. The count is the fit bound at
  `S' = (m − ⌈S⌉ + o_s) + (b ω + o_b)` plus `Φ (c + 1) + o_w`.
* `secure_device`: work, memory and a count of ingress in one bound, on that frame. The compute audit's window is
  the ticks before the cut and the capacity audit's the last of them, both charging one account, and a verdict
  `counted` bounds what the device's links deliver after the cut by `y`. The error is the two audits' together.

`work_le_window` and `held_le_state` take the schedule's charges to the run: the compute account's work over its
window is work its gates do on the device, and the capacity account's charge over one tick is held on the device at
the end of it.
-/

@[expose] public section

namespace Slowdown.SecurityProofs

open Verity ResourceAccounting ResourceAccounting.SecurityProofs.Actual Finset Slowdown.Security

open scoped ENNReal

/-- One window: the ticks `[0, t)`. -/
def wholeCuts (t : ℕ) : Cuts t where
  J := 1
  τ j := j.val * t
  mono _ _ h := Nat.mul_le_mul_right t h
  zero := Nat.zero_mul t
  last := Nat.one_mul t

theorem wholeCuts_castSucc (t : ℕ) (j : Fin (wholeCuts t).J) : (wholeCuts t).τ j.castSucc = 0 := by
  have h0 : (j : ℕ) = 0 := Nat.lt_one_iff.1 j.isLt
  change (j : ℕ) * t = 0
  rw [h0, Nat.zero_mul]

theorem wholeCuts_succ (t : ℕ) (j : Fin (wholeCuts t).J) : (wholeCuts t).τ j.succ = 0 + t := by
  have h0 : (j : ℕ) = 0 := Nat.lt_one_iff.1 j.isLt
  change ((j : ℕ) + 1) * t = 0 + t
  rw [h0, Nat.zero_add, Nat.one_mul, Nat.zero_add]

/-- The one window of `wholeCuts t` is `t` ticks long. -/
theorem whole_ω {L E : Type*} (hw : Hardware L E) (X : Finset L) (hX : X ∈ hw.units) (t : ℕ)
    (missed : Finset (Fin (1 * (wholeCuts t).J))) : (Frame.single hw X hX t (wholeCuts t) missed).ω = t := by
  change (univ : Finset (Fin 1)).sup (fun j => j.succ.val * t - j.castSucc.val * t) = t
  simp

/-- The one window of `wholeCuts t` is `t` ticks long. -/
theorem whole_len {L E : Type*} (hw : Hardware L E) (X : Finset L) (hX : X ∈ hw.units) (t : ℕ)
    (missed : Finset (Fin (1 * (wholeCuts t).J))) (j : Fin (wholeCuts t).J) :
    (Frame.single hw X hX t (wholeCuts t) missed).len j = t := by
  change (wholeCuts t).τ j.succ - (wholeCuts t).τ j.castSucc = t
  rw [wholeCuts_succ, wholeCuts_castSucc]
  omega

section Audit

variable {L : Type*} {ι α E ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] {d : Device ι} (r : Run d α ΩW ΩQ ΩS)
  (ρ : ΩW × ΩQ × ΩS → Actual.Run L) (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
  (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true) (acct : ∀ ω, Fin (ρ ω).A.N → α)

/-- **The computation at the coins `ω`**: the run `ρ ω` with its usage and registered inputs, its declared circuit's
gates those charged to the account of the resource `res`. -/
abbrev computation (res : Resource) (ω : ΩW × ΩQ × ΩS) : Computation L E :=
  ⟨ρ ω, usage ω, θ ω, hθ ω, univ.filter fun g => acct ω g = r.acct res⟩

variable [DecidableEq L] (chip mem : Finset L) (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass) (hbm : E × L)
  (side : ι → E × L)

/-- The work the compute audit charges its account over its window `[a, a + n)` is work the account's gates do on the
device in those ticks, at a pricing that is the audit's price on each op class. -/
theorem work_le_window {a n : ℕ} (hwin : r.win .compute = ⟨a, n⟩) (ω : ΩW × ΩQ × ΩS)
    (hsched : r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) (pc : Price)
    (hπ : ∀ g, (ρ ω).A.x g ∈ chip ∪ mem → (usage ω).π g = (cls ω g).elim 0 pc) :
    r.work pc ω ≤ (ρ ω).A.work (usage ω).π ((ρ ω).A.region (chip ∪ mem) (Set.Ico a (a + n)) ∩
      univ.filter fun g => acct ω g = r.acct .compute) := by
  have hwork := covers_work (ρ ω) chip mem (cls ω) (acct ω) (usage ω).link hbm side
    (fun g hg => hπ g (by convert hg)) ⟨a, n⟩ (r.acct .compute)
  rw [Run.work, hsched, hwin, ← hwork]
  refine sum_le_sum_of_subset fun g hg => ?_
  obtain ⟨hg, ha⟩ := mem_filter.1 hg
  exact mem_inter.2 ⟨by convert hg, mem_filter.2 ⟨mem_univ _, ha⟩⟩

/-- What the capacity audit charges its account over the one tick `c` is held on the device at the end of that tick
(`mem_le_state`). -/
theorem held_le_state {c : ℕ} (hwin : r.win .capacity = ⟨c, 1⟩) (ω : ΩW × ΩQ × ΩS)
    (hsched : r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) :
    r.held ω ≤ (ρ ω).A.width ((ρ ω).state (chip ∪ mem) (c + 1) ∩
      univ.filter fun g => acct ω g = r.acct .capacity) := by
  rw [Run.held, hsched, hwin, Schedule.held]
  change ∑ τ ∈ Finset.Ico c (c + 1), _ ≤ _
  rw [Nat.Ico_succ_singleton, sum_singleton]
  convert mem_le_state (ρ ω) chip mem (cls ω) (acct ω) (usage ω).link hbm side c (r.acct .capacity)

section Slots

open scoped ENNReal

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS] (F : Frame L E)
  (i : Fin F.n) (j : Fin F.cuts.J)

/-- **Work's slot, filled by the compute audit.** The window is the compute audit's window `[a, a + n)` on the
device `chip ∪ mem`, the declared circuit's gates are those of an account `res` the compute audit charges, and the
pricing is the compute audit's price `pc` on each op class and nothing on a gate of none. Where the compute audit has
its per-run guarantee on the priced sum at certificate `W`, read price `pRead` and error `ε`, it is sound for
`WorkAtLeast` at `⌈W − pRead · read⌉` with error `ε`, `read` the bits the compute account read from device memory on
that run. -/
theorem work_slot (res : Resource) (hres : r.acct res = r.acct .compute) (hserver : F.server i = chip ∪ mem)
    {a n : ℕ} (hwin : r.win .compute = ⟨a, n⟩) (hspan : F.span j = Set.Ico a (a + n))
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) (pc : Price)
    (hπ : ∀ ω g, (ρ ω).A.x g ∈ chip ∪ mem → (usage ω).π g = (cls ω g).elim 0 pc) {pRead W ε : ℝ}
    (hsound : r.ComputeSoundPerRunPriced pc pRead W ε) :
    Sound (Game.draw (ΩW × ΩQ × ΩS)) (r.accepts .compute)
      (fun ω => computation r ρ usage θ hθ acct res ω ∈ WorkAtLeast F i j ⌈W - pRead * r.pouwRead ω⌉₊)
      (ENNReal.ofReal ε) := by
  refine sound_draw_of_pr ((Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => hω.2 ?_⟩).trans
    (r.pr_computePricedLt_le hsound))
  have hx : ⌈W - pRead * r.pouwRead ω⌉₊ ≤ r.work pc ω := Nat.ceil_le.2 (by linarith)
  refine hx.trans ((work_le_window r ρ usage acct chip mem cls hbm side hwin ω (hsched ω) pc (hπ ω)).trans
    (le_of_eq ?_))
  change _ = (ρ ω).A.work (usage ω).π ((ρ ω).A.region (F.server i) (F.span j) ∩
    univ.filter fun g => acct ω g = r.acct res)
  rw [hserver, hspan, hres]

/-- **Memory's slot, filled by the capacity audit.** The capacity audit's window is the one tick `c`, the window's
cut is the end of that tick on the device `chip ∪ mem`, and the declared circuit's gates are those of an account
`res` the capacity audit charges. Where the capacity audit has its per-run guarantee at certificate `S` and error
`ε`, it is sound for `HeldAtLeast` at `⌈S⌉` with error `ε`. -/
theorem held_slot (res : Resource) (hres : r.acct res = r.acct .capacity) (hserver : F.server i = chip ∪ mem)
    {c : ℕ} (hwin : r.win .capacity = ⟨c, 1⟩) (hcut : F.cuts.τ j.castSucc = c + 1)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) {S ε : ℝ}
    (hsound : r.CapacitySoundPerRun S ε) :
    Sound (Game.draw (ΩW × ΩQ × ΩS)) (r.accepts .capacity)
      (fun ω => computation r ρ usage θ hθ acct res ω ∈ HeldAtLeast F i j ⌈S⌉₊) (ENNReal.ofReal ε) := by
  refine sound_draw_of_pr ((Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => hω.2 ?_⟩).trans
    (r.pr_capacityLt_le hsound))
  have hx : ⌈S⌉₊ ≤ r.held ω := Nat.ceil_le.2 (by
    rw [hwin] at hge
    simpa using hge)
  refine hx.trans ((held_le_state r ρ usage acct chip mem cls hbm side hwin ω (hsched ω)).trans (le_of_eq ?_))
  change _ = (ρ ω).A.width ((ρ ω).state (F.server i) (F.cuts.τ j.castSucc) ∩
    univ.filter fun g => acct ω g = r.acct res)
  rw [hserver, hcut, hres]

/-- **Bandwidth's slot, filled by the bandwidth audit.** The window is the bandwidth audit's window `[a, a + n)` on
the chip, the memory link `hbm` ends at a leaf of `chip` and its other end is outside `chip`, every run is feasible,
and the declared circuit's gates are those charged to the bandwidth account. Where the bandwidth audit has its
per-run guarantee at certificate `Q` and error `ε`, it is sound for `DeliveredAtLeast` at `⌈Q⌉` with error `ε`. -/
theorem delivered_slot (hw : Hardware L E) (hserver : F.server i = chip) {a n : ℕ}
    (hwin : r.win .bandwidth = ⟨a, n⟩) (hlo : F.cuts.τ j.castSucc = a) (hhi : F.cuts.τ j.succ = a + n)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side)
    (hchip : hbm.2 ∈ chip)
    (hfar : ∀ y, hw.ends hbm.1 = (y, hbm.2) ∨ hw.ends hbm.1 = (hbm.2, y) → y ∉ chip)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .bandwidth ω ∈ H hw) {Q ε : ℝ}
    (hsound : r.BandwidthSoundPerRun Q ε) :
    Sound (Game.draw (ΩW × ΩQ × ΩS)) (r.accepts .bandwidth)
      (fun ω => computation r ρ usage θ hθ acct .bandwidth ω ∈ DeliveredAtLeast F i j ⌈Q⌉₊)
      (ENNReal.ofReal ε) := by
  refine sound_draw_of_pr ((Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => hω.2 ?_⟩).trans
    (r.pr_bandwidthLt_le hsound))
  have hx : ⌈Q⌉₊ ≤ r.read ω := Nat.ceil_le.2 hge
  refine hx.trans ?_
  change _ ≤ (computation r ρ usage θ hθ acct .bandwidth ω).delivered F i j _
  unfold Computation.delivered
  rw [hserver, hlo, hhi, Run.read, hsched, hwin]
  refine sum_le_sum fun τ _ => ?_
  have h := read_le_delivered (ρ ω) chip mem (cls ω) (acct ω) (usage ω).link hbm side hchip
    (fun g s hs hne hl hy => hfar _ (by
      have he := ((hH ω).2.2.2.1 g s hs hne).2
      rwa [hl, hy] at he)) τ (r.acct .bandwidth)
  convert h

end Slots

/-- A bound on the probability of an event of coins that don't exist holds at every error of at least 0. -/
theorem pr_le_of_isEmpty {Ω : Type} [Fintype Ω] [IsEmpty Ω] (P : Ω → Prop) {ε : ℝ} (hε : 0 ≤ ε) : Pouw.pr P ≤ ε := by
  classical
  refine le_of_eq_of_le ?_ hε
  simp [Pouw.pr, Claims.Pr, Claims.prUniform]

end Audit

section Compute

variable {L : Type*} {E : Type} [DecidableEq L] (hw : Hardware L E) (chip mem : Finset L)
  (hX : chip ∪ mem ∈ hw.units) (t : ℕ)

/-- The frame of the device `chip ∪ mem` over the one window `[0, t)`, which the memory and bandwidth audits miss. -/
abbrev deviceFrame : Frame L E := Frame.single hw (chip ∪ mem) hX t (wholeCuts t) univ

theorem deviceFrame_span (j : Fin (deviceFrame hw chip mem hX t).cuts.J) :
    (deviceFrame hw chip mem hX t).span j = Set.Ico 0 (0 + t) := by
  rw [Frame.span]
  exact congrArg₂ Set.Ico (wholeCuts_castSucc t j) (wholeCuts_succ t j)

variable [Fintype E] {ι α ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩQ]
  [Nonempty ΩS] {d : Device ι} (r : Run d α ΩW ΩQ ΩS) (ρ : ΩW × ΩQ × ΩS → Actual.Run L)
  (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
  (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true)
  (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass) (acct : ∀ ω, Fin (ρ ω).A.N → α)

/-- **An accepted compute audit's run is not prohibited, outside the audit's error.** Every run is feasible on `hw`;
the joint audit's schedule is the one the run induces on the device `chip ∪ mem`, whose memory link is `hbm` and side
channels `side`; the compute window is `[0, t)`; the pricing is the compute audit's price `pc` on each op class and
nothing on a gate of none; and the compute audit has its per-run guarantee on the priced sum at certificate `W`, read
price `pRead` and error `ε`. Then the probability that the compute audit accepts and the computation holds a reference
training algorithm of more than `Φ t − ⌈W − pRead · read⌉ + o_w` counted multiplications that meets the recipe's
premises is at most `ε`, with `Φ` the device's compute a tick and `read` the bits the compute account read from device
memory on that run. -/
theorem secure_compute (hwin : r.win .compute = ⟨0, t⟩) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) (pc : Price)
    (hπ : ∀ ω g, (ρ ω).A.x g ∈ chip ∪ mem → (usage ω).π g = (cls ω g).elim 0 pc)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .compute ω ∈ H hw) {pRead W ε : ℝ}
    (hsound : r.ComputeSoundPerRunPriced pc pRead W ε)
    (p : Recipe) :
    Pouw.pr (fun ω => r.accepts .compute ω ∧
      computation r ρ usage θ hθ acct .compute ω ∈ B (deviceFrame hw chip mem hX t) p
        ((hw.compute (chip ∪ mem) * t - ⌈W - pRead * r.pouwRead ω⌉₊ + p.o_w : ℕ) : ℝ)) ≤ ε := by
  have hε : 0 ≤ ε := (Pouw.pr_nonneg _).trans (r.pr_computePricedLt_le hsound)
  rcases isEmpty_or_nonempty ΩW with hΩ | hΩ
  · exact pr_le_of_isEmpty _ hε
  have hY : ∀ ω, (deviceFrame hw chip mem hX t).BoundAtMost 0 0 ⌈W - pRead * r.pouwRead ω⌉₊ 0 p
      ((hw.compute (chip ∪ mem) * t - ⌈W - pRead * r.pouwRead ω⌉₊ + p.o_w : ℕ) : ℝ) := fun ω => by
    refine ⟨fun h => absurd rfl h, ((bound_of_missed _ rfl _ _ _ _ p).trans ?_).le⟩
    rw [single_Φ]
    simp only [whole_len]
    rw [sum_const, card_univ, Fintype.card_fin]
    simp [wholeCuts]
  have h := RunSecure (Game.draw (ΩW × ΩQ × ΩS)) (computation r ρ usage θ hθ acct .compute)
    (deviceFrame hw chip mem hX t) (fun _ => 0) (fun _ => 0) (fun ω => ⌈W - pRead * r.pouwRead ω⌉₊) (fun _ => 0) p
    (fun ω => ((hw.compute (chip ∪ mem) * t - ⌈W - pRead * r.pouwRead ω⌉₊ + p.o_w : ℕ) : ℝ))
    (fun _ => True) (fun _ => True) (fun _ => True) (r.accepts .compute) 0 0 0 (ENNReal.ofReal ε) hY hH
    (fun i j h => absurd (mem_univ _) h) (fun i j h => absurd (mem_univ _) h) (fun i j h => absurd (mem_univ _) h)
    (fun i j _ => work_slot r ρ usage θ hθ acct chip mem cls hbm side (deviceFrame hw chip mem hX t) i j .compute
      rfl rfl hwin (deviceFrame_span hw chip mem hX t j) hsched pc hπ hsound)
  refine (Pouw.pr_mono fun ω hω => ?_).trans (pr_le_of_sound_draw hε
    (h.mono (fun a ha => ⟨trivial, trivial, trivial, ha⟩) (fun _ ha => ha) (le_of_eq ?_)))
  · exact ⟨hω.1, not_not.2 hω.2⟩
  · have hcard : (deviceFrame hw chip mem hX t).missed.card = 1 := by
      change (univ : Finset (Fin (1 * (wholeCuts t).J))).card = 1
      rw [card_univ, Fintype.card_fin]
      rfl
    rw [hcard]
    change ((1 : ℕ) : ℝ≥0∞) * _ + ((1 * 1 - 1 : ℕ) : ℝ≥0∞) * _ = _
    simp

/-- **Work's slot at Pearl-C's audit, W1 priced**: `work_slot` with the compute audit's guarantee from the gap
property per coin over `w1Priced`, for a run that realizes `W1.w1Stop`. The certificate is
`(1 − g) (1 − ε_s) ∑ W_ref` and the error `(1 − ε_s)^k + δ` at the device's oracle queries over the window, `k` the
tickets. -/
theorem w1_slot {Q O S : Type} [Fintype Q] [DecidableEq Q] [Fintype O] [Fintype S] (dv : Pouw.PearlC.W1.W1Device)
    (hdv : Pouw.PearlC.W1BudgetAbort.Positive dv) (salt : S → List ℕ) {P : Pouw.PearlC.Protocol Q O S}
    {D : Pouw.PearlC.Layout → Prop} {g : ℝ} {δ : ℕ → ℕ → ℝ} (hg : g < 1)
    (hG : Pouw.PearlC.GGapPerCoin (Pouw.PearlC.W1.w1Priced dv salt) P D g δ) {U : Pouw.PearlC.Workload}
    (hU : U.InDomain P D) {w : ℕ → ℕ} {κ : ℝ} (hκ : 0 ≤ κ) (hw' : ∀ u < U.layout.N, P.Wref U.layout u = κ * w u)
    {k : ℕ} {εs : ℝ} (hεs : εs ≤ 1) [Nonempty (PearlCCoins Q O S U w k)]
    (r : Run d α (PearlCCoins Q O S U w k) ΩQ ΩS) (ρ : PearlCCoins Q O S U w k × ΩQ × ΩS → Actual.Run L)
    (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
    (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true) (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass)
    (acct : ∀ ω, Fin (ρ ω).A.N → α) (F : Frame L E) (i : Fin F.n) (j : Fin F.cuts.J) (res : Resource)
    (hres : r.acct res = r.acct .compute) (hserver : F.server i = chip ∪ mem) {a n : ℕ}
    (hwin : r.win .compute = ⟨a, n⟩) (hspan : F.span j = Set.Ico a (a + n)) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) (pc : Price)
    (hπ : ∀ ω g, (ρ ω).A.x g ∈ chip ∪ mem → (usage ω).π g = (cls ω g).elim 0 pc) {pRead : ℝ}
    (hq : d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64)
    (real : RealizesPearlCPriced (Pouw.PearlC.W1.w1Stop dv salt) P U w r pc pRead) :
    Sound (Game.draw (PearlCCoins Q O S U w k × ΩQ × ΩS)) (r.accepts .compute)
      (fun ω => computation r ρ usage θ hθ acct res ω ∈ WorkAtLeast F i j
        ⌈(1 - g) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u - pRead * r.pouwRead ω⌉₊)
      (ENNReal.ofReal ((1 - εs) ^ k + δ (d.rate .oracle * (r.win .compute).len) U.layout.N)) :=
  work_slot r ρ usage θ hθ acct chip mem cls hbm side F i j res hres hserver hwin hspan hsched pc hπ
    (w1Stop_computeSoundPerRunPriced dv hdv salt hg hG hU hκ hw' hεs r hq real)

end Compute

section Bandwidth

variable {L : Type*} {E : Type} [DecidableEq L] [Fintype E] (hw : Hardware L E) (chip mem : Finset L)
  (hX : chip ∈ hw.units) (t : ℕ)

/-- The frame of the chip over the one window `[0, t)`, which the memory and bandwidth audits cover. -/
abbrev chipFrame : Frame L E := Frame.single hw chip hX t (wholeCuts t) ∅

variable {ι α ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
  [Nonempty ΩS] {d : Device ι} (r : Run d α ΩW ΩQ ΩS) (ρ : ΩW × ΩQ × ΩS → Actual.Run L)
  (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
  (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true)
  (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass) (acct : ∀ ω, Fin (ρ ω).A.N → α)

/-- **An accepted bandwidth audit's run is not prohibited, outside the audit's error.** The bandwidth audit's window
is `[0, t)`, the joint audit's schedule at each coin is the one the run induces on the device with its chip at `chip`,
the memory link `hbm` ends at a leaf of `chip` and its other end is outside `chip`, every run is feasible, and the
bandwidth audit has its per-run guarantee at certificate `Q` and error `ε`. Then, with the declared circuit's gates
those charged to the bandwidth account and `Y` at least the fit bound of the chip's frame at `x_b = ⌈Q⌉` and no other
amount, the probability that the audit accepts a run holding a reference training algorithm of more than `Y` counted
multiplications is at most `ε`. -/
theorem secure_bandwidth (hwin : r.win .bandwidth = ⟨0, t⟩) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side)
    (hchip : hbm.2 ∈ chip)
    (hfar : ∀ y, hw.ends hbm.1 = (y, hbm.2) ∨ hw.ends hbm.1 = (hbm.2, y) → y ∉ chip)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .bandwidth ω ∈ H hw) {Q ε : ℝ}
    (hsound : r.BandwidthSoundPerRun Q ε) (p : Recipe) (Y : ℝ)
    (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e)
    (hgap : ((chipFrame hw chip hX t).unaccounted 0 ⌈Q⌉₊
      ((chipFrame hw chip hX t).b * (chipFrame hw chip hX t).ω) p : ℝ) < p.Nstar * p.b_w)
    (hY : (chipFrame hw chip hX t).bound 0 ⌈Q⌉₊ 0
      ((chipFrame hw chip hX t).b * (chipFrame hw chip hX t).ω) p ≤ Y) :
    Pouw.pr (fun ω => r.accepts .bandwidth ω ∧
      computation r ρ usage θ hθ acct .bandwidth ω ∈ B (chipFrame hw chip hX t) p Y) ≤ ε := by
  have hε : 0 ≤ ε := (Pouw.pr_nonneg _).trans (r.pr_bandwidthLt_le hsound)
  rcases isEmpty_or_nonempty ΩQ with hΩ | hΩ
  · exact pr_le_of_isEmpty _ hε
  have h := RunSecure (Game.draw (ΩW × ΩQ × ΩS)) (computation r ρ usage θ hθ acct .bandwidth)
    (chipFrame hw chip hX t) (fun _ => 0) (fun _ => ⌈Q⌉₊) (fun _ => 0)
    (fun _ => (chipFrame hw chip hX t).b * (chipFrame hw chip hX t).ω) p (fun _ => Y)
    (fun _ => True) (r.accepts .bandwidth) (fun _ => True) (fun _ => True) 0 (ENNReal.ofReal ε) 0 0
    (fun _ => ⟨fun _ => ⟨hbw, hwe, hgap⟩, hY⟩) hH
    (fun i j _ => sound_of_imp _ fun _ _ => Nat.zero_le _)
    (fun i j _ => delivered_slot r ρ usage θ hθ acct chip mem cls hbm side (chipFrame hw chip hX t) i j hw rfl hwin
      (wholeCuts_castSucc t j) (wholeCuts_succ t j) hsched hchip hfar hH hsound)
    (fun i j _ => sound_of_imp _ fun ω _ => delivered_le_capacity _ _ (hH ω) i j)
    (fun i j h => absurd h (notMem_empty _))
  refine (Pouw.pr_mono fun ω hω => ?_).trans (pr_le_of_sound_draw hε
    (h.mono (fun a ha => ⟨trivial, ha, trivial, trivial⟩) (fun _ ha => ha) (le_of_eq ?_)))
  · exact ⟨hω.1, not_not.2 hω.2⟩
  · change ((0 : ℕ) : ℝ≥0∞) * _ + ((1 * 1 - 0 : ℕ) : ℝ≥0∞) * _ = _
    simp

/-- **Bandwidth's slot at PoUB's audit**: `delivered_slot` at the bandwidth audit's game with a store of `N` blocks
of `b` bits, `h`-bit tags, `s` sampled blocks and at most `G` regenerations (`RealizesBandwidth`), at the certificate
and the error of `BandwidthOfAuditPerRun`. -/
theorem poub_slot [Fintype ι] {N b h s G : ℕ} (k lam : ℕ) [Nonempty (BandwidthCoins N b h s)]
    (r : Run d α ΩW (BandwidthCoins N b h s) ΩS)
    (ρ : ΩW × BandwidthCoins N b h s × ΩS → Actual.Run L) (usage : ∀ ω, Usage (ρ ω).A E)
    (θ : ∀ ω, Finset (Fin (ρ ω).A.N)) (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true)
    (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass) (acct : ∀ ω, Fin (ρ ω).A.N → α) (F : Frame L E) (i : Fin F.n)
    (j : Fin F.cuts.J) (hserver : F.server i = chip) {a n : ℕ} (hwin : r.win .bandwidth = ⟨a, n⟩)
    (hlo : F.cuts.τ j.castSucc = a) (hhi : F.cuts.τ j.succ = a + n) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side)
    (hchip : hbm.2 ∈ chip)
    (hfar : ∀ y, hw.ends hbm.1 = (y, hbm.2) ∨ hw.ends hbm.1 = (hbm.2, y) → y ∉ chip)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .bandwidth ω ∈ H hw) (hN : 0 < N)
    (real : RealizesBandwidth G r) :
    Sound (Game.draw (ΩW × BandwidthCoins N b h s × ΩS)) (r.accepts .bandwidth)
      (fun ω => computation r ρ usage θ hθ acct .bandwidth ω ∈ DeliveredAtLeast F i j
        ⌈((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * (r.win .bandwidth).len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam⌉₊)
      (ENNReal.ofReal ((((k - 1 : ℕ) : ℝ) / N) ^ s + (s : ℝ) / 2 ^ h +
        (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ) / 2 ^ lam)) :=
  delivered_slot r ρ usage θ hθ acct chip mem cls hbm side F i j hw hserver hwin hlo hhi hsched hchip hfar hH
    (ResourceAccounting.SecurityProofs.BandwidthOfAuditPerRun N b h s G k lam ι α ΩW ΩS d r hN real)

end Bandwidth

section Capacity

variable {L : Type*} {E : Type} [DecidableEq L] [Fintype E] (hw : Hardware L E) (chip mem : Finset L)
  (hX : chip ∪ mem ∈ hw.units) (c t : ℕ) (hc : c + 1 ≤ t)

/-- Two windows: the ticks `[0, c + 1)` and `[c + 1, t)`. -/
def cutAt : Cuts t where
  J := 2
  τ := ![0, c + 1, t]
  mono := Fin.monotone_iff_le_succ.2 fun i => by
    fin_cases i
    · exact Nat.zero_le _
    · exact hc
  zero := rfl
  last := rfl

/-- The frame of the device `chip ∪ mem` cut at the end of tick `c`: the memory and bandwidth audits miss the window
before the cut and cover the one after. -/
abbrev cutFrame : Frame L E :=
  Frame.single hw (chip ∪ mem) hX t (cutAt c t hc) (univ.filter fun u => ((finProdFinEquiv.symm u).2 : ℕ) = 0)

omit [Fintype E] in
/-- The window the memory and bandwidth audits cover is the one after the cut. -/
theorem cutFrame_covered {i : Fin (cutFrame hw chip mem hX c t hc).n} {j : Fin (cutFrame hw chip mem hX c t hc).cuts.J}
    (hj : finProdFinEquiv (i, j) ∉ (cutFrame hw chip mem hX c t hc).missed) :
    (cutFrame hw chip mem hX c t hc).cuts.τ j.castSucc = c + 1 := by
  have hj1 : j = (1 : Fin 2) := by
    have h0 : (j : ℕ) ≠ 0 := fun h => hj (mem_filter.2 ⟨mem_univ _, by simpa using h⟩)
    have h2 : (j : ℕ) < 2 := j.isLt
    exact Fin.ext (by simp only [Fin.val_one]; omega)
  subst hj1
  rfl

omit [Fintype E] in
/-- The window they miss is the ticks before the cut. -/
theorem cutFrame_missed {i : Fin (cutFrame hw chip mem hX c t hc).n} {j : Fin (cutFrame hw chip mem hX c t hc).cuts.J}
    (hj : finProdFinEquiv (i, j) ∈ (cutFrame hw chip mem hX c t hc).missed) :
    (cutFrame hw chip mem hX c t hc).span j = Set.Ico 0 (0 + (c + 1)) := by
  have hj0 : j = (0 : Fin 2) := Fin.ext (by simpa using (mem_filter.1 hj).2)
  subst hj0
  rw [Nat.zero_add]
  rfl

omit [Fintype E] in
/-- They miss one window of the two. -/
theorem cutFrame_card : (cutFrame hw chip mem hX c t hc).missed.card = 1 := by
  change (univ.filter fun u : Fin (1 * 2) => ((finProdFinEquiv.symm u).2 : ℕ) = 0).card = 1
  decide

variable {ι α ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
  [Nonempty ΩQ] {d : Device ι} (r : Run d α ΩW ΩQ ΩS) (ρ : ΩW × ΩQ × ΩS → Actual.Run L)
  (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
  (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true)
  (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass) (acct : ∀ ω, Fin (ρ ω).A.N → α)

/-- **An accepted capacity audit's run is not prohibited, outside the audit's error.** The capacity audit's window is
the one tick `c`, the joint audit's schedule at each coin is the one the run induces on the device `chip ∪ mem`, every
run is feasible, and the capacity audit has its per-run guarantee at certificate `S` and error `ε`. Then, with the
declared circuit's gates those charged to the capacity account and `Y` at least the fit bound of the device's frame
cut at the end of tick `c`, at `x_s = ⌈S⌉` and no other amount, the probability that the audit accepts a run holding a
reference training algorithm of more than `Y` counted multiplications is at most `ε`. The bound charges the ticks
before the cut the device's whole work. -/
theorem secure_capacity (hwin : r.win .capacity = ⟨c, 1⟩) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .capacity ω ∈ H hw) {S ε : ℝ}
    (hsound : r.CapacitySoundPerRun S ε) (p : Recipe) (Y : ℝ) (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e)
    (hgap : ((cutFrame hw chip mem hX c t hc).unaccounted ⌈S⌉₊ 0
      ((cutFrame hw chip mem hX c t hc).b * (cutFrame hw chip mem hX c t hc).ω) p : ℝ) < p.Nstar * p.b_w)
    (hY : (cutFrame hw chip mem hX c t hc).bound ⌈S⌉₊ 0 0
      ((cutFrame hw chip mem hX c t hc).b * (cutFrame hw chip mem hX c t hc).ω) p ≤ Y) :
    Pouw.pr (fun ω => r.accepts .capacity ω ∧
      computation r ρ usage θ hθ acct .capacity ω ∈ B (cutFrame hw chip mem hX c t hc) p Y) ≤ ε := by
  have hε : 0 ≤ ε := (Pouw.pr_nonneg _).trans (r.pr_capacityLt_le hsound)
  rcases isEmpty_or_nonempty ΩS with hΩ | hΩ
  · exact pr_le_of_isEmpty _ hε
  have h := RunSecure (Game.draw (ΩW × ΩQ × ΩS)) (computation r ρ usage θ hθ acct .capacity)
    (cutFrame hw chip mem hX c t hc) (fun _ => ⌈S⌉₊) (fun _ => 0) (fun _ => 0)
    (fun _ => (cutFrame hw chip mem hX c t hc).b * (cutFrame hw chip mem hX c t hc).ω) p (fun _ => Y)
    (r.accepts .capacity) (fun _ => True) (fun _ => True) (fun _ => True) (ENNReal.ofReal ε) 0 0 0
    (fun _ => ⟨fun _ => ⟨hbw, hwe, hgap⟩, hY⟩) hH
    (fun i j hj => held_slot r ρ usage θ hθ acct chip mem cls hbm side (cutFrame hw chip mem hX c t hc) i j
      .capacity rfl rfl hwin (cutFrame_covered hw chip mem hX c t hc hj) hsched hsound)
    (fun i j _ => sound_of_imp _ fun _ _ => Nat.zero_le _)
    (fun i j _ => sound_of_imp _ fun ω _ => delivered_le_capacity _ _ (hH ω) i j)
    (fun i j _ => sound_of_imp _ fun _ _ => Nat.zero_le _)
  refine (Pouw.pr_mono fun ω hω => ?_).trans (pr_le_of_sound_draw hε
    (h.mono (fun a ha => ⟨ha, trivial, trivial, trivial⟩) (fun _ ha => ha) (le_of_eq ?_)))
  · exact ⟨hω.1, not_not.2 hω.2⟩
  · rw [cutFrame_card]
    change ((1 : ℕ) : ℝ≥0∞) * _ + ((1 * 2 - 1 : ℕ) : ℝ≥0∞) * _ = _
    simp

open Pous Pous.EndToEnd PousVerifier PousVerifier.Properties Pous.Period in
set_option exponentiation.threshold 20000 in
/-- **Memory's slot at PoUS's period**: `held_slot` at PoUS's per-period guarantee
(`Pous.Period.SecurityProofs.periodCapacity`), under P2's security in the ring model at `(D, K, R)`, for a prover
strategy under the device model whose session is the tick `c`, a capacity verdict that is the verifier's whole
decision outside a break, any certificate `S` that with the chip's state and what the links carry in a round is at
most `S₀`, and the error `δ + ε_max`. -/
theorem pous_slot [Fintype ι] {D K R : ℕ} (hP : Pous.P2.Assumptions.P2RingAuditSecure D K R)
    [Nonempty ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)]
    (r : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (ρ : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Actual.Run L)
    (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
    (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true) (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass)
    (acct : ∀ ω, Fin (ρ ω).A.N → α) (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ)
    (hdev : DeviceModel r P τ D K R round) (hinst : Instantiates P τ)
    (hS : S + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S)
    (hv : ∀ ω, r.accepts .capacity ω → P.Accepts ω ∧ ¬ P.Break ω) (F : Frame L E) (i : Fin F.n)
    (j : Fin F.cuts.J) (res : Resource) (hres : r.acct res = r.acct .capacity)
    (hserver : F.server i = chip ∪ mem) (hwin : r.win .capacity = ⟨c, 1⟩)
    (hcut : F.cuts.τ j.castSucc = c + 1) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) :
    Sound (Game.draw (ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)))
      (r.accepts .capacity) (fun ω => computation r ρ usage θ hθ acct res ω ∈ HeldAtLeast F i j ⌈S⌉₊)
      (ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal)) :=
  held_slot r ρ usage θ hθ acct chip mem cls hbm side F i j res hres hserver hwin hcut hsched
    (Pous.Period.SecurityProofs.periodCapacity D K R hP ι α ΩW ΩQ d r P τ round S hdev hinst (by
      rw [hwin]
      simpa using hS) hv).1

/-- **Work, memory and a count of ingress in one bound.** The compute audit's window is the ticks `[0, c + 1)` before
the cut, the capacity audit's the tick `c`, and both charge one account, whose gates are the declared circuit's; each
has its per-run guarantee, at certificates `W` and `S` and errors `ε_W` and `ε_S`. `counted` is any verdict under
which the device's links deliver at most `y` bits in the window after the cut. Then the probability that both audits
accept, the count holds, and the run holds a reference training algorithm of more counted multiplications than the
fit bound at `S' = (m − ⌈S⌉ + o_s) + (min (b ω) y + o_b)` after the cut plus
`Φ (c + 1) − ⌈W − pRead · read⌉ + o_w` before it, is at most `ε_W + ε_S`. With no count, `counted` is `True` and `y`
the links' capacity `b ω` (`delivered_le_capacity`). -/
theorem secure_device [Nonempty ΩS] (hwinW : r.win .compute = ⟨0, c + 1⟩) (hwinS : r.win .capacity = ⟨c, 1⟩)
    (hacct : r.acct .compute = r.acct .capacity) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) (pc : Price)
    (hπ : ∀ ω g, (ρ ω).A.x g ∈ chip ∪ mem → (usage ω).π g = (cls ω g).elim 0 pc)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .capacity ω ∈ H hw)
    {pRead W S εW εS : ℝ} (hW : r.ComputeSoundPerRunPriced pc pRead W εW) (hS : r.CapacitySoundPerRun S εS)
    (counted : ΩW × ΩQ × ΩS → Prop) (y : ℕ)
    (hy : ∀ ω i j, counted ω → finProdFinEquiv (i, j) ∉ (cutFrame hw chip mem hX c t hc).missed →
      (computation r ρ usage θ hθ acct .capacity ω).delivered (cutFrame hw chip mem hX c t hc) i j univ ≤ y)
    (p : Recipe) (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e)
    (hgap : ((cutFrame hw chip mem hX c t hc).unaccounted ⌈S⌉₊ 0 y p : ℝ) < p.Nstar * p.b_w) :
    Pouw.pr (fun ω => r.accepts .compute ω ∧ r.accepts .capacity ω ∧ counted ω ∧
      computation r ρ usage θ hθ acct .capacity ω ∈ B (cutFrame hw chip mem hX c t hc) p
        ((cutFrame hw chip mem hX c t hc).bound ⌈S⌉₊ 0 ⌈W - pRead * r.pouwRead ω⌉₊ y p)) ≤ εW + εS := by
  have hεW : 0 ≤ εW := (Pouw.pr_nonneg _).trans (r.pr_computePricedLt_le hW)
  have hεS : 0 ≤ εS := (Pouw.pr_nonneg _).trans (r.pr_capacityLt_le hS)
  have h := RunSecure (Game.draw (ΩW × ΩQ × ΩS)) (computation r ρ usage θ hθ acct .capacity)
    (cutFrame hw chip mem hX c t hc) (fun _ => ⌈S⌉₊) (fun _ => 0) (fun ω => ⌈W - pRead * r.pouwRead ω⌉₊)
    (fun _ => y) p (fun ω => (cutFrame hw chip mem hX c t hc).bound ⌈S⌉₊ 0 ⌈W - pRead * r.pouwRead ω⌉₊ y p)
    (r.accepts .capacity) (fun _ => True) counted (r.accepts .compute) (ENNReal.ofReal εS) 0 0 (ENNReal.ofReal εW)
    (fun _ => ⟨fun _ => ⟨hbw, hwe, hgap⟩, le_rfl⟩) hH
    (fun i j hj => held_slot r ρ usage θ hθ acct chip mem cls hbm side (cutFrame hw chip mem hX c t hc) i j
      .capacity rfl rfl hwinS (cutFrame_covered hw chip mem hX c t hc hj) hsched hS)
    (fun i j _ => sound_of_imp _ fun _ _ => Nat.zero_le _)
    (fun i j hj => sound_of_imp _ fun ω hcount => hy ω i j hcount hj)
    (fun i j hj => work_slot r ρ usage θ hθ acct chip mem cls hbm side (cutFrame hw chip mem hX c t hc) i j
      .capacity hacct.symm rfl hwinW (cutFrame_missed hw chip mem hX c t hc hj) hsched pc hπ hW)
  refine (Pouw.pr_mono fun ω hω => ?_).trans (pr_le_of_sound_draw (add_nonneg hεW hεS)
    (h.mono (acc' := fun ω => r.accepts .compute ω ∧ r.accepts .capacity ω ∧ counted ω)
      (fun a ha => ⟨ha.2.1, trivial, ha.2.2, ha.1⟩) (fun _ ha => ha) (le_of_eq ?_)))
  · exact ⟨⟨hω.1, hω.2.1, hω.2.2.1⟩, not_not.2 hω.2.2.2⟩
  · rw [cutFrame_card, ENNReal.ofReal_add hεW hεS]
    change ((1 : ℕ) : ℝ≥0∞) * _ + ((1 * 2 - 1 : ℕ) : ℝ≥0∞) * _ = _
    simp

end Capacity

end Slowdown.SecurityProofs
