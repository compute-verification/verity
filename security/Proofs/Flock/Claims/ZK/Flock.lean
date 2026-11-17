module

public import Proofs.Flock.Claims.ZK.GKExp
public import Proofs.Flock.Claims.ZK.CoinCheck
public import Proofs.Flock.Claims.ZK.Prefix
public import Proofs.Flock.Claims.DeployCode
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.GKCost

/-! Flock's zero knowledge against a malicious verifier, from its committed coins. -/

@[expose] public section

namespace Claims.ZKMal

open Claims

theorem Flock.zkR_of_proto {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (N : ℕ) (δA δB : Msg → ℝ) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x)
    (hA : G.PrefixHides H (Flock.R D.parse) δA) (hB : G.SHVZK H (Flock.R D.parse) δB) (hB0 : ∀ x, 0 ≤ δB x)
    (hC : G.Binding H) :
    Claims.Flock.ZKAt H D G.prover (G.simulator N) (fun x => 1 / ((N : ℝ) + 1) + δA x + N * δB x) fun _ => N := by
  refine ⟨hzk, ?_⟩
  rw [show (fun x => D.V.k x + 1) = G.k from funext hk]
  exact G.gk_zk H _ N δA δB hA hB hB0 hC

def _root_.Claims.Flock.Deploy.withK {Coin Col T : Type} (D : Flock.Deploy Coin Col T) (k : Msg → ℕ) :
    Flock.Deploy Coin Col T :=
  { D with V := { D.V with k := k } }

theorem Flock.zkAt_withK {Coin Col T N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {D : Flock.Deploy Coin Col T} {k : Msg → ℕ} {P : Flock.Wit → Prover (N × Ω)} {S : Simulator} {εzk γ : Msg → ℝ}
    (hk : ∀ x w, Flock.R D.parse x w → k x = D.V.k x) (h : Claims.Flock.ZKAt H (D.withK k) P S εzk γ) :
    Claims.Flock.ZKAt H D P S εzk γ :=
  ⟨h.1, fun x w hR Θ Vs θ hv Ev => h.2 x w hR Θ Vs θ (hv.trans (congrArg (· + 1) (hk x w hR).symm)) Ev⟩

theorem deployV_k_pos (zk : Bool) (x : Msg) : 0 < (FlockDeploy.V zk).k x := by
  simp only [FlockDeploy.V]
  split <;> omega

theorem deployZk_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (x : Msg) : (FlockDeploy.deployZk rel).V.k x = G.K x + 1 := by
  have := deployV_k_pos true x
  rw [hK x]
  show (FlockDeploy.V true).k x = _
  omega

theorem deployZk_proto_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (x : Msg) : G.k x = (FlockDeploy.deployZk rel).V.k x + 1 := by
  rw [deployZk_k rel G hK x]
  rfl

theorem Flock.zkR_deployZk (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (G : Proto Flock.Wit) (N : ℕ) (δA δB : Msg → ℝ) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1)
    (hA : G.PrefixHides H (Flock.R (FlockDeploy.deployZk rel).parse) δA)
    (hB : G.SHVZK H (Flock.R (FlockDeploy.deployZk rel).parse) δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover (G.simulator N)
      (fun x => 1 / ((N : ℝ) + 1) + δA x + N * δB x) fun _ => N :=
  Flock.zkR_of_proto H _ G N δA δB rfl (fun x => (deployZk_proto_k rel G hK x).symm) hA hB hB0 hC

theorem Flock.simCost_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (N : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCost H (G.simulator N) fun x => (G.simBound N cChk cMsg cSim x : ℝ) :=
  (G.gk_cost H N cChk cMsg cSim hc).simCost

theorem Flock.zkR_gkE_of_proto {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (t : ℕ) (ht : 0 < t) (δA δB : Msg → ℝ) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x)
    (hA : G.PrefixHides H (Flock.R D.parse) δA) (hB : G.SHVZK H (Flock.R D.parse) δB) (hB0 : ∀ x, 0 ≤ δB x)
    (hC : G.Binding H) :
    Claims.Flock.ZKAt H D G.prover (G.simulatorE t (t * 3 ^ t)) (fun x => 4 * δA x + 4 * t * δB x + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  refine ⟨hzk, ?_⟩
  rw [show (fun x => D.V.k x + 1) = G.k from funext hk]
  exact G.gkE_zk_pow H _ t ht δA δB hA hB hB0 hC

theorem Flock.zkR_gkE_deployZk (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) (t : ℕ) (ht : 0 < t)
    (δA δB : Msg → ℝ) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1)
    (hA : G.PrefixHides H (Flock.R (FlockDeploy.deployZk rel).parse) δA)
    (hB : G.SHVZK H (Flock.R (FlockDeploy.deployZk rel).parse) δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 4 * δA x + 4 * t * δB x + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zkR_gkE_of_proto H _ G t ht δA δB rfl (fun x => (deployZk_proto_k rel G hK x).symm) hA hB hB0 hC

theorem Flock.zkR_gkE_hm96_bind {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (t : ℕ) (ht : 0 < t) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x) (δ₁ : ENNReal) (nL nB : Msg → ℕ)
    (hA : ∀ x w, Flock.R D.parse x w → ∀ (ν : G.Nonce) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
      (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤ 2 * nL x * δ₁.toReal)
    (hB : G.SHVZK H (Flock.R D.parse) fun x => 2 * nB x * δ₁.toReal) (hC : G.Binding H) :
    Claims.Flock.ZKAt H D G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  have h := Flock.zkR_gkE_of_proto H D G t ht (fun x => 2 * nL x * δ₁.toReal) (fun x => 2 * nB x * δ₁.toReal) hzk hk
    (G.prefixHides_of_runs H _ _ hA) hB (fun x => by positivity) hC
  have e : (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3 : ℝ) ^ t) =
      fun x => 4 * (2 * nL x * δ₁.toReal) + 4 * t * (2 * nB x * δ₁.toReal) + 2 * (1 / 3) ^ t :=
    funext fun x => by ring
  rw [e]
  exact h

theorem Flock.zkR_gkE_hm96 {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (t : ℕ) (ht : 0 < t) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x) (δ₁ : ENNReal) (nL nB : Msg → ℕ)
    (hA : ∀ x w, Flock.R D.parse x w → ∀ (ν : G.Nonce) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
      (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤ 2 * nL x * δ₁.toReal)
    (hB : G.SHVZK H (Flock.R D.parse) fun x => 2 * nB x * δ₁.toReal) {CoinT Salt : Type}
    (Tr : Msg → Msg → FlockSoundness.ZK.GK.CoinTree Dg CoinT Salt) (nb : G.Nonce → List UInt8)
    (root : Msg → Msg → Dg) (parse : Msg → Option (FlockSoundness.ZK.GK.CoinOpening Dg CoinT Salt))
    (pos : Msg → Msg → ℕ → List UInt8) (idx n depth : Msg → Msg → ℕ → ℕ) (dc : List CoinT → G.Coin)
    (hchk : ∀ x ν ρ j m, G.chk x ν ρ j m =
      treeChk (Tr x ρ) (nb ν) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) (parse m))
    (hdec : ∀ m op, parse m = some op → G.dec m = dc op.cs) :
    Claims.Flock.ZKAt H D G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zkR_gkE_hm96_bind H D G t ht hzk hk δ₁ nL nB hA hB
    (G.binding_of_tree H Tr nb root parse pos idx n depth dc hchk hdec)

theorem Flock.zkR_gkE_hm96_deployZk (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (δ₁ : ENNReal) (nL nB : Msg → ℕ)
    (hA : ∀ x w, Flock.R (FlockDeploy.deployZk rel).parse x w → ∀ (ν : G.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤ 2 * nL x * δ₁.toReal)
    (hB : G.SHVZK H (Flock.R (FlockDeploy.deployZk rel).parse) fun x => 2 * nB x * δ₁.toReal) {CoinT Salt : Type}
    (Tr : Msg → Msg → FlockSoundness.ZK.GK.CoinTree Dg CoinT Salt) (nb : G.Nonce → List UInt8)
    (root : Msg → Msg → Dg) (parse : Msg → Option (FlockSoundness.ZK.GK.CoinOpening Dg CoinT Salt))
    (pos : Msg → Msg → ℕ → List UInt8) (idx n depth : Msg → Msg → ℕ → ℕ) (dc : List CoinT → G.Coin)
    (hchk : ∀ x ν ρ j m, G.chk x ν ρ j m =
      treeChk (Tr x ρ) (nb ν) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) (parse m))
    (hdec : ∀ m op, parse m = some op → G.dec m = dc op.cs) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zkR_gkE_hm96 H _ G t ht rfl (fun x => (deployZk_proto_k rel G hK x).symm) δ₁ nL nB hA hB Tr nb root parse
    pos idx n depth dc hchk hdec

theorem Flock.simCostExp_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCost H (G.simulatorE t M) fun x => (1 + 5 * t) * G.runBound cChk cMsg cSim x :=
  G.gkE_costExp H t M cChk cMsg cSim hc

theorem Flock.simCost_gkE_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCostLe H (G.simulatorE t M) fun x => (1 + 5 * M) * G.runBound cChk cMsg cSim x :=
  G.gkE_cost H t M cChk cMsg cSim hc

end Claims.ZKMal
