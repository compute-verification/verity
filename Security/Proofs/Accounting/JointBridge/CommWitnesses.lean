import Proofs.Accounting.JointBridge.HonestH200
import Proofs.NetworkCertifier.WitnessToys

/-!
# `CommOfPeriodLeakAccepted`'s nontrivial witness: the network audit that accepts everything

`Accounting.Guarantees.Bridge.CommOfPeriodLeakAccepted` concludes the network audit's `PeriodRun.Sound` at
`n · (δ + ε_ks + ε_bind + ε_tap)`. Its conclusion fails at a period whose network audit accepts at every coin while its
claim fails at every coin (`CommWitness.acceptAll`): the node moved one bit through the network certifiers, and the
claim is that it moved none (`K = 0`). The error is the one the accepting sampled verifier `tapD` gives at SHA-512 at
one window at `δ = 1/8`, with `ε_ks`, `ε_bind` and `ε_tap` its knowledge error, extraction and collision terms at the
statement `[]`. That error is below one (`tapD_κ_eighth`, `prB_accV`, `tapD_bext`, `tapD_btap`), while the network
audit accepts with its claim failing with probability one (`CommOfPeriodLeakAccepted_nontrivial`, which states that
some period run fails the conclusion, and takes this one).
-/

namespace Accounting.SecurityProofs.Bridge.CommWitness

/-- **The network audit that accepts everything**, on `HonestH200`'s period of one tick at no windows, its coins one
coordinate of one value: every audit accepts at every coin, and the network certifier's claim at `K` is that the one
bit the node moved through the network certifiers is within `K`. -/
def acceptAll : Accounting.PeriodRun (HonestH200.device 0) Unit HonestH200.emptyCircuit (fun _ : Unit => Unit) where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  own := fun _ => ()
  acct := fun _ => ()
  win := ⟨0, 1⟩
  schedule := fun _ _ => HonestH200.step 0
  fits := fun _ _ => HonestH200.step_fits 0
  transcript := fun _ g => g.elim0
  commWithin := fun K _ => (1 : ℝ) ≤ K
  accepts := fun _ _ => True

end Accounting.SecurityProofs.Bridge.CommWitness

namespace Accounting.SecurityProofs.Bridge

open Claims Claims.Witness NetTiming NetTiming.Witness

local notation "shaD" => FlockDeploy.sha512

/-- **`CommOfPeriodLeakAccepted`'s conclusion fails at a period run**: the network audit that accepts everything
(`CommWitness.acceptAll`), at one window at `δ = 1/8`, with `ε_ks`, `ε_bind` and `ε_tap` the accepting sampled
verifier `tapD`'s terms at SHA-512 at the statement `[]`: that error is below one, and the network audit accepts with
its claim failing at every coin. The proof names the run: a statement in the lock reads no proof-only definition. -/
theorem CommOfPeriodLeakAccepted_nontrivial :
    ∃ (d : Accounting.Device Unit) (C : Verity.Circuit) (r : Accounting.PeriodRun d Unit C fun _ : Unit => Unit),
      ¬ r.Sound ⟨0, 0, 0, 0, 0⟩ (fun _ => 1) .network
        (((1 : ℕ) : ℝ) * (1 / 8 +
          (Claims.Sampling.κ (tapD Unit) (1 / 8) [] - 1 / 8 +
            Real.sqrt (Claims.Sampling.γ (tapD Unit) [] * PrB shaD (tapD Unit).V silentP [])) +
          (Claims.Sampling.Bext shaD (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) Unit silentP []).Pr
            shaD +
          (Btap shaD (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) (tapD Unit).commit (tapD Unit).parse
            (fun _ => tapOps) Unit silentP []).Pr shaD)) := by
  refine ⟨HonestH200.device 0, HonestH200.emptyCircuit, CommWitness.acceptAll, fun h => ?_⟩
  have h1 := h fun _ => ()
  have hev : ∀ x : Unit,
      CommWitness.acceptAll.accepts .network
          (Function.update (fun _ => ()) (CommWitness.acceptAll.own .network) x) ∧
        ¬ CommWitness.acceptAll.Holds ⟨0, 0, 0, 0, 0⟩ (fun _ => 1) .network
          (Function.update (fun _ => ()) (CommWitness.acceptAll.own .network) x) := fun _ =>
    ⟨trivial, fun hK => (by norm_num : ¬ (1 : ℝ) ≤ 0) hK⟩
  have hpr : Pouw.pr (fun x : Unit =>
      CommWitness.acceptAll.accepts .network
          (Function.update (fun _ => ()) (CommWitness.acceptAll.own .network) x) ∧
        ¬ CommWitness.acceptAll.Holds ⟨0, 0, 0, 0, 0⟩ (fun _ => 1) .network
          (Function.update (fun _ => ()) (CommWitness.acceptAll.own .network) x)) = 1 :=
    Pouw.pr_eq_one hev
  rw [hpr] at h1
  have hb : PrB shaD (tapD Unit).V silentP [] = 0 := prB_accV shaD silentP []
  simp only [hb, mul_zero, Real.sqrt_zero, add_zero, tapD_bext, tapD_btap, Nat.cast_one, one_mul] at h1
  linarith [tapD_κ_eighth (Coin := Unit) []]

end Accounting.SecurityProofs.Bridge
