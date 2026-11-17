module

public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.ZK.Hypotheses
public import Proofs.Flock.Claims.ZK.Indexed
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
import Proofs.Flock.Claims.ZK.Congr
import Proofs.Flock.Soundness.CROnly.Hiding

/-! Sampling's zero knowledge from one committed-coin protocol for the whole sampled run, whose models no session proof gives; the route of record is `SamplingDeploy.zk_seq`. -/

@[expose] public section

namespace Claims.Sampling

open Claims Claims.ZKMal

/-- **A committed-coin protocol with leakage**: its dummy-witness messages and SHVZK simulator also given the leakage; `G`'s
own `dmsg` and `sim` are not read. -/
structure Leaky (W : Type) where
  G : Proto W
  dmsg : List Msg → Msg → G.Nonce → G.Rand → List G.Coin → OC Msg
  sim : List Msg → (x : Msg) → G.Nonce → (Fin (G.K x) → G.Coin) → G.SimRand x → OC (List Msg)

namespace Leaky

variable {W : Type}

def proto (L : Leaky W) (lx : List Msg) : Proto W := { L.G with dmsg := L.dmsg lx, sim := L.sim lx }

def simLeak (L : Leaky W) (N : ℕ) : SimLeak where
  C x := L.G.Nonce × L.G.Rand × (Fin N → L.G.SimRand x)
  prog x lx c := ((L.proto lx).simulator N).prog x c

def simLeakE (L : Leaky W) (t M : ℕ) : SimLeak where
  C x := L.G.Nonce × L.G.Rand × (Fin M → L.G.Rand) × (Fin (4 * M) → L.G.SimRand x)
  prog x lx c := ((L.proto lx).simulatorE t M).prog x c

end Leaky

variable {Coin Col T : Type}

def RAt (H : Msg → Msg) (D : Deploy Coin Col T) (lx : List Msg) : Rel Wit :=
  fun x w => Rhon H D.commit D.parse D.partition x w ∧ leafLayer H D.commit D.parse x w = lx

def ZKAt {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (D : Deploy Coin Col T)
    (P : Wit → Prover (N × Ω)) (S : SimLeak) (εzk γ : Msg → ℝ) : Prop :=
  D.flock.zk = true ∧ ∀ l : Msg → List Msg,
    ZKmalR H D.V.k P (fun x w => Rhon H D.commit D.parse D.partition x w ∧ leafLayer H D.commit D.parse x w = l x)
      (S.given l) εzk γ

theorem zk_mask_of_zkAt {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (D : Deploy Coin Col T) (P : Wit → Prover (N × Ω)) (tapeOf : N × Ω → D.Tape) (S : SimLeak) (εzk γ : Msg → ℝ)
    (hP : ∀ w x (hist : List (Msg × Msg)) ω, hist.length < D.V.k x →
      ((D.P w).next x hist (tapeOf ω)).eval H = ((P w).next x hist ω).eval H)
    (h : ZKAt H D P S εzk γ) : ZK H (D.mask N Ω tapeOf S) εzk γ :=
  ⟨h.1, fun l => (zkmalRx_const_of H D.V.k (D.mask N Ω tapeOf S).prover [] (fun _ _ => rfl) _ _ εzk γ).2
    (zkmalR_congr_eq (fun w x hist ω hh => (hP w x hist ω hh).symm) (h.2 l))⟩

theorem zk_of_leaky (H : Msg → Msg) (D : Deploy Coin Col T) (L : Leaky Wit) (N : ℕ) (δA δB : Msg → ℝ)
    (hzk : D.flock.zk = true) (hk : ∀ x s, D.parse x = some s → L.G.k x = D.V.k x)
    (hA : ∀ lx, (L.proto lx).PrefixHides H (RAt H D lx) δA) (hB : ∀ lx, (L.proto lx).SHVZK H (RAt H D lx) δB)
    (hB0 : ∀ x, 0 ≤ δB x) (hC : L.G.Binding H) :
    ZKAt H D L.G.prover (L.simLeak N) (fun x => 1 / ((N : ℝ) + 1) + δA x + N * δB x) fun _ => N := by
  refine ⟨hzk, fun l x w hR Θ Vs θ hkV Ev => ?_⟩
  obtain ⟨s, _, hs, _⟩ := hR.1
  exact (L.proto (l x)).gk_zk H (RAt H D (l x)) N δA δB (hA _) (hB _) hB0 hC x w hR Θ Vs θ
    (hkV.trans (hk x s hs).symm) Ev

theorem zk_of_leakyE (H : Msg → Msg) (D : Deploy Coin Col T) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (δA δB : Msg → ℝ) (hzk : D.flock.zk = true) (hk : ∀ x s, D.parse x = some s → L.G.k x = D.V.k x)
    (hA : ∀ lx, (L.proto lx).PrefixHides H (RAt H D lx) δA) (hB : ∀ lx, (L.proto lx).SHVZK H (RAt H D lx) δB)
    (hB0 : ∀ x, 0 ≤ δB x) (hC : L.G.Binding H) :
    ZKAt H D L.G.prover (L.simLeakE t (t * 3 ^ t)) (fun x => 4 * δA x + 4 * t * δB x + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  refine ⟨hzk, fun l x w hR Θ Vs θ hkV Ev => ?_⟩
  obtain ⟨s, _, hs, _⟩ := hR.1
  exact (L.proto (l x)).gkE_zk_pow H (RAt H D (l x)) t ht δA δB (hA _) (hB _) hB0 hC x w hR Θ Vs θ
    (hkV.trans (hk x s hs).symm) Ev

/-- The leaf layer is the leakage, so the claim reads no hiding of it; that it hides the unopened rows is the leaf's own
hm96 hiding, outside this claim. -/
theorem zk_of_models (H : Msg → Msg) (D : Deploy Coin Col T) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hzk : D.flock.zk = true) (hk : ∀ x s, D.parse x = some s → L.G.k x = D.V.k x) {Salt B Cc : Type}
    [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B)
    (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H D lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H D lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZKAt H D L.G.prover (L.simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  have h := zk_of_leakyE H D L t ht (fun x => 2 * nL x * δ₁.toReal) (fun x => 2 * nB x * δ₁.toReal) hzk hk
    (fun lx => (L.proto lx).prefixHides_of_runs H _ _
      ((L.proto lx).prefix_of_models H _ My cs hT1 hδ₁ (Mp lx) (Mc lx) nL (hnL lx) (hL lx)))
    (fun lx => (L.proto lx).shvzk_of_models H _ My cs hT1 hδ₁ (Mp lx) (Mc lx) nB (hnB lx) (hL lx) (hS lx) (hN lx)
      (hI lx))
    (fun x => by positivity) (L.G.binding_of_coinCheck H hC)
  have e : (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3 : ℝ) ^ t) =
      fun x => 4 * (2 * nL x * δ₁.toReal) + 4 * t * (2 * nB x * δ₁.toReal) + 2 * (1 / 3) ^ t :=
    funext fun x => by ring
  rw [e]
  exact h

end Claims.Sampling

namespace SamplingDeploy

open Claims Claims.Sampling Claims.ZKMal

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

theorem zk_deploy_of_models (H : Msg → Msg) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZKAt H (deploy rel c) L.G.prover (L.simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  zk_of_models H (deploy rel c) L t ht rfl hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC

theorem zk_deploy_of_models_gap (H : Msg → Msg) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [DecidableEq Cc] (My : Salt → B)
    (cs : Salt → Cc) (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZKAt H (deploy rel c) L.G.prover (L.simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  zk_deploy_of_models rel c H L t ht hK My cs (FlockSoundness.CROnly.hm96Hiding_gap My cs) ENNReal.ofReal_ne_top Mp
    Mc nL nB hnL hnB hL hS hN hI hC

def Honest (H : Msg → Msg) (G : Proto Wit) (tapeOf : G.Nonce × G.Rand → Reference.Tape) : Prop :=
  ∀ w x (hist : List (Msg × Msg)) ω, hist.length < c.V.k x →
    ((Reference.refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w).next x hist
        (tapeOf ω)).eval H =
      ((G.prover w).next x hist ω).eval H

theorem zk_deploy_mask_of_models_gap (H : Msg → Msg) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [DecidableEq Cc] (My : Salt → B)
    (cs : Salt → Cc) (tapeOf : L.G.Nonce × L.G.Rand → Reference.Tape) (hP : Honest c H L.G tapeOf)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZK H ((deploy rel c).mask L.G.Nonce L.G.Rand tapeOf (L.simLeakE t (t * 3 ^ t)))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  zk_mask_of_zkAt H (deploy rel c) L.G.prover tapeOf _ _ _ hP
    (zk_deploy_of_models_gap rel c H L t ht hK My cs Mp Mc nL nB hnL hnB hL hS hN hI hC)

set_option synthInstance.maxSize 1024 in
theorem sampling_deployed_models (H : Msg → Msg) (hslots : DrawSlots (deploy rel c))
    (hprog : ProgramScope (deploy rel c)) (hComplete : Complete H (deploy rel c)) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [DecidableEq Cc] (My : Salt → B)
    (cs : Salt → Cc) (tapeOf : L.G.Nonce × L.G.Rand → Reference.Tape) (hP : Honest c H L.G tapeOf)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      Complete H (deploy rel c) ∧
      ZK H ((deploy rel c).mask L.G.Nonce L.G.Rand tapeOf (L.simLeakE t (t * 3 ^ t)))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
        fun _ => 4 * t * 3 ^ t :=
  sampling_deployed rel c H hslots hprog hComplete tapeOf (L.simLeakE t (t * 3 ^ t)) _ _
    (zk_deploy_mask_of_models_gap rel c H L t ht hK My cs tapeOf hP Mp Mc nL nB hnL hnB hL hS hN hI hC)

end SamplingDeploy
