module

public import Proofs.Slowdown.Secure
public import Proofs.ResourceAccounting.Actual.Device
public import Proofs.Pouw.PerRunStop
import Proofs.Pouw.PerRunPriced
import Proofs.Pouw.Prob

/-!
# An accepted compute audit's run is not prohibited

`secure_work` over the compute audit's coins, with `A`'s amount of work from the audit's own guarantee and not a
hypothesis. The frame is one server, the device `chip ∪ mem`, over one window, the compute audit's `[0, t)`
(`wholeCuts`), which the memory and bandwidth audits miss. At each coin the computation is a run of the actual-circuit
model, its declared circuit's gates those charged to the compute account, and the joint audit's schedule is the one
the run induces on the device (`Actual.Run.schedule`), at a pricing that is the compute audit's price on each op class.

* `secure_compute`: where every run is feasible and the compute audit has its per-run guarantee on the priced sum at
  certificate `W` and error `ε`, the probability that it accepts a run holding a reference training algorithm of more
  than `Φ t − ⌈W − pRead · read⌉ + o_w` counted multiplications is at most `ε`, `read` the bits the compute account
  read from device memory on that run.
* `secure_w1`: the same at Pearl-C's audit at W1 priced, from the gap property per coin
  (`w1Stop_computeSoundPerRunPriced`).
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

section Compute

variable {L : Type*} {E : Type} [DecidableEq L] (hw : Hardware L E) (chip mem : Finset L)
  (hX : chip ∪ mem ∈ hw.units) (t : ℕ)

/-- The frame of the device `chip ∪ mem` over the one window `[0, t)`, which the memory and bandwidth audits miss. -/
abbrev deviceFrame : Frame L E := Frame.single hw (chip ∪ mem) hX t (wholeCuts t) univ

theorem deviceFrame_span (j : Fin (deviceFrame hw chip mem hX t).cuts.J) :
    (deviceFrame hw chip mem hX t).span j = Set.Ico 0 (0 + t) := by
  have h0 : (j : ℕ) = 0 := Nat.lt_one_iff.1 j.isLt
  have h1 : (deviceFrame hw chip mem hX t).cuts.τ j.castSucc = 0 := by
    change (j : ℕ) * t = 0
    rw [h0, Nat.zero_mul]
  have h2 : (deviceFrame hw chip mem hX t).cuts.τ j.succ = 0 + t := by
    change ((j : ℕ) + 1) * t = 0 + t
    rw [h0, Nat.zero_add, Nat.one_mul, Nat.zero_add]
  rw [Frame.span, h1, h2]

theorem deviceFrame_ω : (deviceFrame hw chip mem hX t).ω = t := by
  change (univ : Finset (Fin 1)).sup (fun j => j.succ.val * t - j.castSucc.val * t) = t
  simp

variable [Fintype E] {ι α ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩQ]
  [Nonempty ΩS] {d : Device ι} (r : Run d α ΩW ΩQ ΩS) (ρ : ΩW × ΩQ × ΩS → Actual.Run L)
  (usage : ∀ ω, Usage (ρ ω).A E) (θ : ∀ ω, Finset (Fin (ρ ω).A.N))
  (hθ : ∀ ω, ∀ a ∈ θ ω, (ρ ω).A.isInput a = true)
  (cls : ∀ ω, Fin (ρ ω).A.N → Option OpClass) (acct : ∀ ω, Fin (ρ ω).A.N → α)

/-- **The computation at the coins `ω`**: the run's actual circuit with its use of compute and links and its registered
inputs, the declared circuit's gates those charged to the compute account. -/
abbrev computation (ω : ΩW × ΩQ × ΩS) : Computation L E :=
  ⟨(ρ ω).A, usage ω, θ ω, hθ ω, univ.filter fun g => acct ω g = r.acct .compute⟩

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
      computation r ρ usage θ hθ acct ω ∈ B (deviceFrame hw chip mem hX t) p
        ((hw.compute (chip ∪ mem) * t - ⌈W - pRead * r.pouwRead ω⌉₊ + p.o_w : ℕ) : ℝ)) ≤ ε := by
  refine (Pouw.pr_mono fun ω hω => ⟨hω.1, lt_of_not_ge fun hge => ?_⟩).trans (r.pr_computePricedLt_le hsound)
  have hx : ⌈W - pRead * r.pouwRead ω⌉₊ ≤ r.work pc ω := Nat.ceil_le.2 (by linarith)
  refine secure_work (deviceFrame hw chip mem hX t) rfl 0 0 ⌈W - pRead * r.pouwRead ω⌉₊ 0 p _ ?_ _ (hH ω)
    (fun i j => ⟨fun h => absurd (mem_univ _) h, fun _ => hx.trans ?_⟩) hω.2
  · rw [single_Φ, deviceFrame_ω]
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
      computation r ρ usage θ hθ acct ω ∈ B (deviceFrame hw chip mem hX t) p
        ((hw.compute (chip ∪ mem) * t -
          ⌈(1 - g) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u - pRead * r.pouwRead ω⌉₊ +
            p.o_w : ℕ) : ℝ)) ≤
      (1 - εs) ^ k + δ (d.rate .oracle * (r.win .compute).len) U.layout.N :=
  secure_compute hw chip mem hX t r ρ usage θ hθ cls acct hwin hbm side hsched pc hπ hH
    (w1Stop_computeSoundPerRunPriced dv hdv salt hg hG hU hκ hw' hεs r hq real) p

end Compute

end Slowdown.SecurityProofs
