module

public import Proofs.Slowdown.Secure
public import Proofs.ResourceAccounting.Actual.Device
public import Proofs.Pouw.PerRunStop
public import Proofs.Poub.BandwidthAudit
public import Proofs.Pous.Period
import Proofs.Pouw.PerRunJoint
import Proofs.Pouw.PerRunPriced
import Proofs.Pouw.Prob

/-!
# An accepted audit's run is not prohibited

`Secure` over an audit's coins, with `A`'s amount from the audit's own guarantee and not a hypothesis. The frame is
one server over the ticks `[0, t)`, one window (`wholeCuts`) or two (`cutAt`). At each coin the computation is a run of the
actual-circuit model, its declared circuit's gates those charged to the audit's account, and the joint audit's
schedule is the one the run induces on the device (`Actual.Run.schedule`).

Work. The server is the device `chip ∪ mem` and the window is missed by the memory and bandwidth audits; the pricing
is the compute audit's price on each op class.

* `secure_compute`: where every run is feasible and the compute audit has its per-run guarantee on the priced sum at
  certificate `W` and error `ε`, the probability that it accepts a run holding a reference training algorithm of more
  than `Φ t − ⌈W − pRead · read⌉ + o_w` counted multiplications is at most `ε`, `read` the bits the compute account
  read from device memory on that run.
* `secure_w1`: the same at Pearl-C's audit at W1 priced, from the gap property per coin
  (`w1Stop_computeSoundPerRunPriced`).

Bandwidth. The server is the chip alone and the window is covered, with no memory audit: the chip's own state is
what it carries. The memory link ends at a leaf of the chip and its other end is outside the chip.

* `secure_bandwidth`: where every run is feasible and the bandwidth audit has its per-run guarantee at certificate `Q`
  and error `ε`, the probability that it accepts a run holding a reference training algorithm of more counted
  multiplications than the fit bound at `S' = (m + o_s) + (b t − ⌈Q⌉ + o_b)` is at most `ε`.
* `secure_poub`: the same at PoUB's audit, from `BandwidthOfAuditPerRun`.

Memory. The server is the device `chip ∪ mem` and the capacity audit's window is one tick, `c`. The frame is cut at
the end of that tick: the window before the cut is missed and charged the device's whole work over its ticks, the
window after it is covered.

* `secure_capacity`: where every run is feasible and the capacity audit has its per-run guarantee at certificate `S`
  and error `ε`, the probability that it accepts a run holding a reference training algorithm of more counted
  multiplications than the fit bound at `S' = (m − ⌈S⌉ + o_s) + (b ω + o_b)` plus `Φ (c + 1) + o_w` is at most `ε`.
  The schedule charges the capacity account what the device holds at the end of the tick (`mem_le_state`), which is
  `A`'s memory clause at the cut.
* `secure_pous`: the same at PoUS's period, from `periodCapacity`.
* `secure_device`: work, memory and a count of ingress in one bound, on that frame. The compute audit's window is
  the ticks before the cut and the capacity audit's the last of them, both charging one account, and a verdict
  `counted` bounds what the device's links deliver after the cut by `y`. The probability that both audits accept, the
  count holds and the run holds a reference training algorithm of more counted multiplications than the fit bound at
  `S' = (m − ⌈S⌉ + o_s) + (min (b ω) y + o_b)` plus `Φ (c + 1) − ⌈W − pRead · read⌉ + o_w` is at most the two
  audits' errors together.

`work_le_window` and `held_le_state` take the schedule's charges to the run: the compute account's work over its
window is work its gates do on the device, and the capacity account's charge over one tick is held on the device at
the end of it.
-/

@[expose] public section

namespace Slowdown.SecurityProofs

open Verity ResourceAccounting ResourceAccounting.SecurityProofs.Actual Finset Slowdown.Security

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
  refine (Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => ?_⟩).trans (r.pr_computePricedLt_le hsound)
  have hx : ⌈W - pRead * r.pouwRead ω⌉₊ ≤ r.work pc ω := Nat.ceil_le.2 (by linarith)
  refine secure_work (deviceFrame hw chip mem hX t) rfl 0 0 ⌈W - pRead * r.pouwRead ω⌉₊ 0 p _ ?_ _ (hH ω)
    (fun i j => ⟨fun h => absurd (mem_univ _) h, fun _ => hx.trans ?_⟩) hω.2
  · rw [single_Φ]
    simp only [whole_len]
    rw [sum_const, card_univ, Fintype.card_fin]
    exact le_of_eq (by simp [wholeCuts])
  · refine (work_le_window r ρ usage acct chip mem cls hbm side hwin ω (hsched ω) pc (hπ ω)).trans (le_of_eq ?_)
    change _ = (ρ ω).A.work (usage ω).π ((ρ ω).A.region (chip ∪ mem) ((deviceFrame hw chip mem hX t).span j) ∩ _)
    rw [deviceFrame_span]

/-- **At Pearl-C's audit, W1 priced**: `secure_compute` with the compute audit's guarantee from the gap property per
coin over `w1Priced`, for a run that realizes `W1.w1Stop`. The certificate is `(1 − g) (1 − ε_s) ∑ W_ref` and the
error `(1 − ε_s)^k + δ` at the device's oracle queries over the window, `k` the tickets. -/
theorem secure_w1 {Q O S : Type} [Fintype Q] [DecidableEq Q] [Fintype O] [Fintype S] (dv : Pouw.PearlC.W1.W1Device)
    (hdv : Pouw.PearlC.W1BudgetAbort.Positive dv) (salt : S → List ℕ) {P : Pouw.PearlC.Protocol Q O S}
    {D : Pouw.PearlC.Layout → Prop} {g : ℝ} {δ : ℕ → ℕ → ℝ} (hg : g < 1)
    (hG : Pouw.PearlC.GGapPerCoin (Pouw.PearlC.W1.w1Priced dv salt) P D g δ) {U : Pouw.PearlC.Workload}
    (hU : U.InDomain P D) {w : ℕ → ℕ} {κ : ℝ} (hκ : 0 ≤ κ) (hw' : ∀ u < U.layout.N, P.Wref U.layout u = κ * w u)
    {k : ℕ} {εs : ℝ} (hεs : εs ≤ 1) [Nonempty (PearlCCoins Q O S U w k)]
    (r : Run d α (PearlCCoins Q O S U w k) ΩQ ΩS) (ρ : PearlCCoins Q O S U w k × ΩQ × ΩS → Actual.Run L)
    (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
    (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true) (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass)
    (acct : ∀ ω, Fin (ρ ω).A.N → α) (hwin : r.win .compute = ⟨0, t⟩) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side) (pc : Price)
    (hπ : ∀ ω g, (ρ ω).A.x g ∈ chip ∪ mem → (usage ω).π g = (cls ω g).elim 0 pc)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .compute ω ∈ H hw) {pRead : ℝ}
    (hq : d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64)
    (real : RealizesPearlCPriced (Pouw.PearlC.W1.w1Stop dv salt) P U w r pc pRead) (p : Recipe) :
    Pouw.pr (fun ω => r.accepts .compute ω ∧
      computation r ρ usage θ hθ acct .compute ω ∈ B (deviceFrame hw chip mem hX t) p
        ((hw.compute (chip ∪ mem) * t -
          ⌈(1 - g) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u - pRead * r.pouwRead ω⌉₊ +
            p.o_w : ℕ) : ℝ)) ≤
      (1 - εs) ^ k + δ (d.rate .oracle * (r.win .compute).len) U.layout.N :=
  secure_compute hw chip mem hX t r ρ usage θ hθ cls acct hwin hbm side hsched pc hπ hH
    (w1Stop_computeSoundPerRunPriced dv hdv salt hg hG hU hκ hw' hεs r hq real) p

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
  refine (Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => ?_⟩).trans (r.pr_bandwidthLt_le hsound)
  have hx : ⌈Q⌉₊ ≤ r.read ω := Nat.ceil_le.2 hge
  refine Secure (chipFrame hw chip hX t) 0 ⌈Q⌉₊ 0 _ p Y ⟨hbw, hwe, hgap, hY⟩ _ (hH ω)
    (fun i j => ⟨fun _ => ⟨Nat.zero_le _, hx.trans ?_, delivered_le_capacity _ _ (hH ω) i j⟩,
      fun h => absurd h (notMem_empty _)⟩) hω.2
  unfold Computation.delivered
  rw [wholeCuts_castSucc t j, wholeCuts_succ t j, Run.read, hsched, hwin]
  refine sum_le_sum fun τ _ => ?_
  have h := read_le_delivered (ρ ω) chip mem (cls ω) (acct ω) (usage ω).link hbm side hchip
    (fun g s hs hne hl hy => hfar _ (by
      have he := ((hH ω).2.2.2.1 g s hs hne).2
      rwa [hl, hy] at he)) τ (r.acct .bandwidth)
  convert h

/-- **An accepted PoUB audit's run is not prohibited, outside PoUB's error**: `secure_bandwidth` at the bandwidth
audit's game with a store of `N` blocks of `b` bits, `h`-bit tags, `s` sampled blocks and at most `G` regenerations
(`RealizesBandwidth`), at the certificate and the error of `BandwidthOfAuditPerRun`. -/
theorem secure_poub [Fintype ι] {N b h s G : ℕ} (k lam : ℕ) (r : Run d α ΩW (BandwidthCoins N b h s) ΩS)
    (ρ : ΩW × BandwidthCoins N b h s × ΩS → Actual.Run L) (usage : ∀ ω, Usage (ρ ω).A E)
    (θ : ∀ ω, Finset (Fin (ρ ω).A.N)) (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true)
    (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass) (acct : ∀ ω, Fin (ρ ω).A.N → α)
    (hwin : r.win .bandwidth = ⟨0, t⟩) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side)
    (hchip : hbm.2 ∈ chip)
    (hfar : ∀ y, hw.ends hbm.1 = (y, hbm.2) ∨ hw.ends hbm.1 = (hbm.2, y) → y ∉ chip)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .bandwidth ω ∈ H hw) (hN : 0 < N)
    (real : RealizesBandwidth G r) (p : Recipe) (Y : ℝ)
    (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e)
    (hgap : ((chipFrame hw chip hX t).unaccounted 0
      ⌈((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * (r.win .bandwidth).len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam⌉₊
      ((chipFrame hw chip hX t).b * (chipFrame hw chip hX t).ω) p : ℝ) < p.Nstar * p.b_w)
    (hY : (chipFrame hw chip hX t).bound 0
      ⌈((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * (r.win .bandwidth).len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam⌉₊ 0
      ((chipFrame hw chip hX t).b * (chipFrame hw chip hX t).ω) p ≤ Y) :
    Pouw.pr (fun ω => r.accepts .bandwidth ω ∧
      computation r ρ usage θ hθ acct .bandwidth ω ∈ B (chipFrame hw chip hX t) p Y) ≤
      (((k - 1 : ℕ) : ℝ) / N) ^ s + (s : ℝ) / 2 ^ h +
        (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ) / 2 ^ lam :=
  secure_bandwidth hw chip mem hX t r ρ usage θ hθ cls acct hwin hbm side hsched hchip hfar hH
    (ResourceAccounting.SecurityProofs.BandwidthOfAuditPerRun N b h s G k lam ι α ΩW ΩS d r hN real) p Y hbw hwe hgap hY

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
  refine (Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => ?_⟩).trans (r.pr_capacityLt_le hsound)
  have hx : ⌈S⌉₊ ≤ r.held ω := Nat.ceil_le.2 (by
    rw [hwin] at hge
    simpa using hge)
  refine Secure (cutFrame hw chip mem hX c t hc) ⌈S⌉₊ 0 0 _ p Y ⟨hbw, hwe, hgap, hY⟩ _ (hH ω)
    (fun i j => ⟨fun hj => ⟨hx.trans ?_, Nat.zero_le _, delivered_le_capacity _ _ (hH ω) i j⟩,
      fun _ => Nat.zero_le _⟩) hω.2
  have hj1 : j = (1 : Fin 2) := by
    have h0 : (j : ℕ) ≠ 0 := fun h => hj (mem_filter.2 ⟨mem_univ _, by simpa using h⟩)
    have h2 : (j : ℕ) < 2 := j.isLt
    exact Fin.ext (by simp only [Fin.val_one]; omega)
  subst hj1
  exact held_le_state r ρ usage acct chip mem cls hbm side hwin ω (hsched ω)

open Pous Pous.EndToEnd PousVerifier PousVerifier.Properties Pous.Period in
set_option exponentiation.threshold 20000 in
/-- **An accepted PoUS period's run is not prohibited, outside PoUS's error**: `secure_capacity` at PoUS's per-period
guarantee (`Pous.Period.SecurityProofs.periodCapacity`), under P2's security in the ring model at `(D, K, R)`, for a
prover strategy under the device model whose session is the tick `c`, a capacity verdict that is the verifier's whole
decision outside a break, any certificate `S` that with the chip's state and what the links carry in a round is at
most `S₀`, and the error `δ + ε_max`. -/
theorem secure_pous [Fintype ι] {D K R : ℕ} (hP : Pous.P2.Assumptions.P2RingAuditSecure D K R)
    (r : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (ρ : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) → Actual.Run L)
    (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
    (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true) (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass)
    (acct : ∀ ω, Fin (ρ ω).A.N → α) (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (S : ℝ)
    (hdev : DeviceModel r P τ D K R round) (hinst : Instantiates P τ)
    (hS : S + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S)
    (hv : ∀ ω, r.accepts .capacity ω → P.Accepts ω ∧ ¬ P.Break ω)
    (hwin : r.win .capacity = ⟨c, 1⟩) (hbm : E × L) (side : ι → E × L)
    (hsched : ∀ ω, r.schedule ω = (ρ ω).schedule chip mem (cls ω) (acct ω) (usage ω).link hbm side)
    (hH : ∀ ω, computation r ρ usage θ hθ acct .capacity ω ∈ H hw) (p : Recipe)
    (Y : ℝ) (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e)
    (hgap : ((cutFrame hw chip mem hX c t hc).unaccounted ⌈S⌉₊ 0
      ((cutFrame hw chip mem hX c t hc).b * (cutFrame hw chip mem hX c t hc).ω) p : ℝ) < p.Nstar * p.b_w)
    (hY : (cutFrame hw chip mem hX c t hc).bound ⌈S⌉₊ 0 0
      ((cutFrame hw chip mem hX c t hc).b * (cutFrame hw chip mem hX c t hc).ω) p ≤ Y) :
    Pouw.pr (fun ω => r.accepts .capacity ω ∧
      computation r ρ usage θ hθ acct .capacity ω ∈ B (cutFrame hw chip mem hX c t hc) p Y) ≤
      (Params.δ : ℝ) + Params.εMax.toReal :=
  secure_capacity hw chip mem hX c t hc r ρ usage θ hθ cls acct hwin hbm side hsched hH
    (Pous.Period.SecurityProofs.periodCapacity D K R hP ι α ΩW ΩQ d r P τ round S hdev hinst (by
      rw [hwin]
      simpa using hS) hv).1 p Y hbw hwe hgap hY

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
  refine (Pouw.pr_mono fun ω hω => ?_).trans ((Pouw.pr_or_le _ _).trans
    (add_le_add (r.pr_computePricedLt_le hW) (r.pr_capacityLt_le hS)))
  by_contra hno
  obtain ⟨hnw, hns⟩ := not_or.1 hno
  have hxw : ⌈W - pRead * r.pouwRead ω⌉₊ ≤ r.work pc ω :=
    Nat.ceil_le.2 (by linarith [not_lt.1 fun h => hnw ⟨hω.1, h⟩])
  have hxs : ⌈S⌉₊ ≤ r.held ω := Nat.ceil_le.2 (by
    have h := not_lt.1 fun h => hns ⟨hω.2.1, h⟩
    rw [hwinS] at h
    simpa using h)
  refine Secure (cutFrame hw chip mem hX c t hc) ⌈S⌉₊ 0 _ y p _ ⟨hbw, hwe, hgap, le_rfl⟩ _ (hH ω)
    (fun i j => ⟨fun hj => ⟨hxs.trans ?_, Nat.zero_le _, hy ω i j hω.2.2.1 hj⟩, fun hj => hxw.trans ?_⟩) hω.2.2.2
  · have hj1 : j = (1 : Fin 2) := by
      have h0 : (j : ℕ) ≠ 0 := fun h => hj (mem_filter.2 ⟨mem_univ _, by simpa using h⟩)
      have h2 : (j : ℕ) < 2 := j.isLt
      exact Fin.ext (by simp only [Fin.val_one]; omega)
    subst hj1
    exact held_le_state r ρ usage acct chip mem cls hbm side hwinS ω (hsched ω)
  · have hj0 : j = (0 : Fin 2) := Fin.ext (by simpa using (mem_filter.1 hj).2)
    subst hj0
    refine (work_le_window r ρ usage acct chip mem cls hbm side hwinW ω (hsched ω) pc (hπ ω)).trans (le_of_eq ?_)
    change _ = (ρ ω).A.work (usage ω).π ((ρ ω).A.region (chip ∪ mem) (Set.Ico 0 (c + 1)) ∩
      univ.filter fun g => acct ω g = r.acct .capacity)
    rw [hacct, Nat.zero_add]

end Capacity

end Slowdown.SecurityProofs
