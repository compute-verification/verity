module

public import Proofs.Slowdown.Secure
public import Proofs.ResourceAccounting.Actual.Device
public import Proofs.Pouw.PerRunStop
public import Proofs.Poub.BandwidthAudit
import Proofs.Pouw.PerRunJoint
import Proofs.Pouw.PerRunPriced
import Proofs.Pouw.Prob

/-!
# An accepted audit's run is not prohibited

`secure` over an audit's coins, with `A`'s amount from the audit's own guarantee and not a hypothesis. The frame is
one server over one window, the audit's `[0, t)` (`wholeCuts`). At each coin the computation is a run of the
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

section Audit

variable {L : Type*} {ι α E ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] {d : Device ι} (r : Run d α ΩW ΩQ ΩS)
  (ρ : ΩW × ΩQ × ΩS → Actual.Run L) (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
  (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true) (acct : ∀ ω, Fin (ρ ω).A.N → α)

/-- **The computation at the coins `ω`**: the run `ρ ω` with its usage and registered inputs, its declared circuit's
gates those charged to the account of the resource `res`. -/
abbrev computation (res : Resource) (ω : ΩW × ΩQ × ΩS) : Computation L E :=
  ⟨(ρ ω).A, usage ω, θ ω, hθ ω, univ.filter fun g => acct ω g = r.acct res⟩

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
    (hH : ∀ ω, Feasible hw (ρ ω).A (usage ω)) {pRead W ε : ℝ} (hsound : r.ComputeSoundPerRunPriced pc pRead W ε)
    (p : Recipe) :
    Pouw.pr (fun ω => r.accepts .compute ω ∧
      computation r ρ usage θ hθ acct .compute ω ∈ B (deviceFrame hw chip mem hX t) p
        ((hw.compute (chip ∪ mem) * t - ⌈W - pRead * r.pouwRead ω⌉₊ + p.o_w : ℕ) : ℝ)) ≤ ε := by
  refine (Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => ?_⟩).trans (r.pr_computePricedLt_le hsound)
  have hx : ⌈W - pRead * r.pouwRead ω⌉₊ ≤ r.work pc ω := Nat.ceil_le.2 (by linarith)
  refine secure_work (deviceFrame hw chip mem hX t) rfl 0 0 ⌈W - pRead * r.pouwRead ω⌉₊ 0 p _ ?_ _ (hH ω)
    (fun i j => ⟨fun h => absurd (mem_univ _) h, fun _ => hx.trans ?_⟩) hω.2
  · rw [single_Φ, whole_ω]
    exact le_of_eq (by simp [wholeCuts])
  · have hwork := covers_work (ρ ω) chip mem (cls ω) (acct ω) (usage ω).link hbm side
      (fun g hg => hπ ω g (by convert hg)) ⟨0, t⟩ (r.acct .compute)
    rw [Run.work, hsched, hwin, ← hwork]
    refine sum_le_sum_of_subset fun g hg => ?_
    obtain ⟨hg, ha⟩ := mem_filter.1 hg
    refine mem_inter.2 ⟨?_, mem_filter.2 ⟨mem_univ _, ha⟩⟩
    change g ∈ (ρ ω).A.region (chip ∪ mem) ((deviceFrame hw chip mem hX t).span j)
    rw [deviceFrame_span]
    convert hg

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
    (hH : ∀ ω, Feasible hw (ρ ω).A (usage ω)) {pRead : ℝ}
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
    (hH : ∀ ω, Feasible hw (ρ ω).A (usage ω)) {Q ε : ℝ} (hsound : r.BandwidthSoundPerRun Q ε) (p : Recipe) (Y : ℝ)
    (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e)
    (hgap : ((chipFrame hw chip hX t).unaccounted 0 ⌈Q⌉₊
      ((chipFrame hw chip hX t).b * (chipFrame hw chip hX t).ω) p : ℝ) < p.Nstar * p.b_w)
    (hY : (chipFrame hw chip hX t).bound 0 ⌈Q⌉₊ 0
      ((chipFrame hw chip hX t).b * (chipFrame hw chip hX t).ω) p ≤ Y) :
    Pouw.pr (fun ω => r.accepts .bandwidth ω ∧
      computation r ρ usage θ hθ acct .bandwidth ω ∈ B (chipFrame hw chip hX t) p Y) ≤ ε := by
  refine (Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => ?_⟩).trans (r.pr_bandwidthLt_le hsound)
  have hx : ⌈Q⌉₊ ≤ r.read ω := Nat.ceil_le.2 hge
  refine secure (chipFrame hw chip hX t) 0 ⌈Q⌉₊ 0 _ p Y hbw hwe hgap hY _ (hH ω)
    (fun i j => ⟨fun _ => ⟨Nat.zero_le _, hx.trans ?_, delivered_le_capacity _ _ (hH ω) i j⟩,
      fun h => absurd h (notMem_empty _)⟩) hω.2
  unfold Computation.delivered
  rw [wholeCuts_castSucc t j, wholeCuts_succ t j, Run.read, hsched, hwin]
  refine sum_le_sum fun τ _ => ?_
  have h := read_le_delivered (ρ ω) chip mem (cls ω) (acct ω) (usage ω).link hbm side hchip
    (fun g s hs hne hl hy => hfar _ (by
      have he := ((hH ω).2.2.2.2.1 g s hs hne).2
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
    (hH : ∀ ω, Feasible hw (ρ ω).A (usage ω)) (hN : 0 < N) (real : RealizesBandwidth G r) (p : Recipe) (Y : ℝ)
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

end Slowdown.SecurityProofs
