module

public import Proofs.Flock.Recursive.FirewallZK
public import Proofs.Flock.Soundness.ZK.HidingSalted
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.ZK.Hiding

/-!
# The ZK server's commitments when the worker reads the salts

The live firewall answers the worker each salt it drew, so a worker's later messages may read it: digest `i` reads the
salts before `i` (`encSalted`). `FirewallZK.recursiveZK_salted` is `recursiveZK_enc` there, at the same
`ε_o + n_h·δ`, with T1 over leaves whose digests read the earlier salts (`ZK.ideal_leaves_salted`) in place of T1 over
adaptive leaves.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- The commitments to the digests `dg` under the salts `y`, digest `i` reading the salts before `i`, through an encoding
`H` of their two halves. -/
def encSalted {D : Type} (H : ZkView.Bits × ByteArray → D) {n : ℕ}
    (dg : (i : Fin n) → (Fin i → ZkView.Salt) → ByteArray) (y : Fin n → ZkView.Salt) : Fin n → D :=
  fun i => H (ZkView.bitsOf (dg i fun j => y (Fin.castLE i.isLt.le j)) + ZkView.My (y i), ZkView.saltHash (y i))

/-- **`recursiveZK_enc` when digest `i` reads the salts before `i`**: the outer session's simulator at the commitments
(`hOuter`), then T1 once per commitment, the last first (`ZK.ideal_leaves_salted`). -/
theorem recursiveZK_salted {D Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (H : ZkView.Bits × ByteArray → D) (nh : ℕ)
    (dg : Ωa → Ωp → (i : Fin nh) → (Fin i → ZkView.Salt) → ByteArray)
    (view : Ωa → (Fin nh → D) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → D) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun q : Ωo => E (view a (encSalted H (dg a p) y) p y q))
        ≤ prCoin (fun s : Ωs => E (sim a (encSalted H (dg a p) y) s)) + εo)
    {δ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (E : Ωa × (Fin nh → D) × V → Prop) :
    prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
        E (w.1.1.1, encSalted H (dg w.1.1.1 w.1.1.2) w.1.2,
          view w.1.1.1 (encSalted H (dg w.1.1.1 w.1.1.2) w.1.2) w.1.1.2 w.1.2 w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2))
        + εo + nh * δ := by
  -- the outer session, simulated at the commitments
  have h1 : prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
        E (w.1.1.1, encSalted H (dg w.1.1.1 w.1.1.2) w.1.2,
          view w.1.1.1 (encSalted H (dg w.1.1.1 w.1.1.2) w.1.2) w.1.1.2 w.1.2 w.2)) ≤
      prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encSalted H (dg w.1.1.1 w.1.1.2) w.1.2,
          sim w.1.1.1 (encSalted H (dg w.1.1.1 w.1.1.2) w.1.2) w.2))
        + εo :=
    calc _ = avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => prCoin fun q : Ωo =>
            E (ωy.1.1, encSalted H (dg ωy.1.1 ωy.1.2) ωy.2,
              view ωy.1.1 (encSalted H (dg ωy.1.1 ωy.1.2) ωy.2) ωy.1.2 ωy.2 q)) :=
          ZK.prCoin_prod_split fun (ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt)) (q : Ωo) =>
            E (ωy.1.1, encSalted H (dg ωy.1.1 ωy.1.2) ωy.2,
              view ωy.1.1 (encSalted H (dg ωy.1.1 ωy.1.2) ωy.2) ωy.1.2 ωy.2 q)
      _ ≤ avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => (prCoin fun s : Ωs =>
            E (ωy.1.1, encSalted H (dg ωy.1.1 ωy.1.2) ωy.2,
              sim ωy.1.1 (encSalted H (dg ωy.1.1 ωy.1.2) ωy.2) s)) + εo) :=
          avg_mono fun ωy => hOuter ωy.1.1 ωy.1.2 ωy.2 fun v =>
            E (ωy.1.1, encSalted H (dg ωy.1.1 ωy.1.2) ωy.2, v)
      _ = avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => prCoin fun s : Ωs =>
            E (ωy.1.1, encSalted H (dg ωy.1.1 ωy.1.2) ωy.2,
              sim ωy.1.1 (encSalted H (dg ωy.1.1 ωy.1.2) ωy.2) s))
            + εo := by
          rw [avg_add, avg_const]
      _ = _ :=
          congrArg (· + εo) (ZK.prCoin_prod_split fun (ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt)) (s : Ωs) =>
            E (ωy.1.1, encSalted H (dg ωy.1.1 ωy.1.2) ωy.2,
              sim ωy.1.1 (encSalted H (dg ωy.1.1 ωy.1.2) ωy.2) s)).symm
  -- T1 once per commitment, the last first: digest `i` reads the salts before `i`, and the simulated session reads the
  -- salts only through the commitments
  have h2 : prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encSalted H (dg w.1.1.1 w.1.1.2) w.1.2,
          sim w.1.1.1 (encSalted H (dg w.1.1.1 w.1.1.2) w.1.2) w.2)) ≤
      prCoin (fun w : ((Ωa × Ωp) × Ωs) × (Fin nh → ZkView.Bits × ZkView.Salt) =>
        E (w.1.1.1, encIdeal H w.2, sim w.1.1.1 (encIdeal H w.2) w.1.2)) + nh * δ :=
    (ZK.prCoin_congr_equiv (Flock.SecurityProofs.swapLast (Ωa × Ωp) (Fin nh → ZkView.Salt) Ωs)
      (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encSalted H (dg w.1.1.1 w.1.1.2) w.1.2,
          sim w.1.1.1 (encSalted H (dg w.1.1.1 w.1.1.2) w.1.2) w.2))).symm.trans_le
      (le_of_eq_of_le rfl (ZK.ideal_leaves_salted ZkView.My ZkView.saltHash hT1 H nh
        (fun (ω : (Ωa × Ωp) × Ωs) i pre => ZkView.bitsOf (dg ω.1.1 ω.1.2 i pre))
        (fun ω l => E (ω.1.1, l, sim ω.1.1 l ω.2))))
  -- the developer's randomness is no longer read
  have h3 : prCoin (fun w : ((Ωa × Ωp) × Ωs) × (Fin nh → ZkView.Bits × ZkView.Salt) =>
        E (w.1.1.1, encIdeal H w.2, sim w.1.1.1 (encIdeal H w.2) w.1.2)) =
      prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
        E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2)) :=
    calc _ = prCoin (fun z : Ωp × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
          E (z.2.1.1, encIdeal H z.2.1.2, sim z.2.1.1 (encIdeal H z.2.1.2) z.2.2)) :=
          ZK.prCoin_congr_equiv (Flock.SecurityProofs.pullMid Ωa Ωp Ωs (Fin nh → ZkView.Bits × ZkView.Salt))
            (fun z : Ωp × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
              E (z.2.1.1, encIdeal H z.2.1.2, sim z.2.1.1 (encIdeal H z.2.1.2) z.2.2))
      _ = avg (fun _ : Ωp => prCoin fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
            E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2)) :=
          ZK.prCoin_prod_split fun (_ : Ωp) (w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
            E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2)
      _ = _ := avg_const _
  refine h1.trans ((add_le_add h2 (le_refl εo)).trans (le_of_eq ?_))
  rw [h3, add_right_comm]

end FlockSoundness.Discharge.Recursion.FirewallZK
