module

public import Proofs.Flock.RecGeneric.AtFlockLinkBits
public import Proofs.Flock.Claims.Recursion.KappaEpsDeployed

/-!
# The gated link's binding term at the deployment's sizes: `ε_L ≤ 2^{−175.65}`

`Claims.RecGeneric.linkTerm_le` bounds `KSLinkOn`'s binding term at `M = 2^256` by `t·ε_L`, `ε_L = β₂·c_L/2^256`
(`Claims.RecGeneric.epsLink`). At `kappa_eps_deployed`'s sizes (`V_I ≤ 4402`, `k_I ≤ 12343986`, `m′ = 210`,
`W_I ≤ 5591040`, at most `16` parts) and call bounds `|S(x_I)|, n, v, w ≤ 2^32`, `β₂ ≤ 11825274530258` and
`c_L ≤ 125901258885`, so `ε_L ≤ 2^{−175.65}` (`epsLink_deployed`).
-/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`ε_L ≤ 2^{−175.65}` from its numbers**, at the sizes `kappa_eps_core` reads and call bounds `|S(x_I)|`, `n`, `v`,
`w` at most `2^32` each. -/
theorem epsLink_core {nS VI kI wI W NI cBe nSet n v w : ℕ} {ws ks : Fin nS → ℕ} (hNI : NI = 1 + wI * kI)
    (hcBe : cBe = 4 + 2 * ∑ s, ws s * ks s) (hSet : nSet ≤ 2 ^ 32) (hn : n ≤ 2 ^ 32) (hv : v ≤ 2 ^ 32)
    (hw : w ≤ 2 ^ 32) (hVI : VI ≤ 4402) (hkI : kI ≤ 12343986) (hwI : wI = 210) (hW : W ≤ 5591040) (hnS : nS ≤ 16)
    (hws : ∀ s, ws s = 210) (hks : ∀ s, ks s ≤ 12343986) :
    ((NI : ℝ) + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe + 2 * (nSet + 2 * n + v + w) + 1 : ℕ) : ℝ) /
      2 ^ (256 : ℝ) ≤ 2 ^ (-(175.65 : ℝ)) := by
  have hNI' : NI ≤ 2592237061 := by
    rw [hNI, hwI]
    calc 1 + 210 * kI ≤ 1 + 210 * 12343986 := by gcongr
      _ = 2592237061 := by norm_num
  have hcBe' : cBe ≤ 4 + 2 * (16 * (210 * 12343986)) := by
    have hsum : ∑ s, ws s * ks s ≤ nS * (210 * 12343986) :=
      sum_fin_le_nat _ fun s => by rw [hws s]; exact Nat.mul_le_mul_left 210 (hks s)
    rw [hcBe]
    gcongr
    exact hsum.trans (Nat.mul_le_mul_right _ hnS)
  have hL : cBe + 2 * (nSet + 2 * n + v + w) + 1 ≤ 125901258885 := by omega
  have hLR : ((cBe + 2 * (nSet + 2 * n + v + w) + 1 : ℕ) : ℝ) ≤ 125901258885 := by exact_mod_cast hL
  have hNIR : (NI : ℝ) ≤ 2592237061 := by exact_mod_cast hNI'
  have hVIR : (VI : ℝ) ≤ 4402 := by exact_mod_cast hVI
  have hkIR : (kI : ℝ) ≤ 12343986 := by exact_mod_cast hkI
  have hWR : (W : ℝ) ≤ 5591040 := by exact_mod_cast hW
  have hβ : ((NI : ℝ) + 8 * W + 4 * ((kI : ℝ) + 1)) * VI ≤ (2592237061 + 8 * 5591040 + 4 * (12343986 + 1)) * 4402 :=
    mul_le_mul (by linarith) hVIR (Nat.cast_nonneg _) (by norm_num)
  have hY : ((NI : ℝ) + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe + 2 * (nSet + 2 * n + v + w) + 1 : ℕ) : ℝ) ≤
      ((1488816950020209223842330 : ℕ) : ℝ) := by
    calc _ ≤ ((2592237061 + 8 * 5591040 + 4 * (12343986 + 1)) * 4402 : ℝ) * 125901258885 :=
          mul_le_mul hβ hLR (Nat.cast_nonneg _) (by norm_num)
      _ = ((1488816950020209223842330 : ℕ) : ℝ) := by norm_num
  -- `2^1607` is past `exponentiation.threshold`, so the comparison is the kernel's, on `ℕ`.
  have hK : ((1488816950020209223842330 : ℕ) : ℝ) ≤ 2 ^ (((1607 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) :=
    le_two_rpow_of_pow_le (Nat.cast_nonneg _) (by norm_num) (cast_pow_le_two_pow (by decide +kernel))
  have e : (2 : ℝ) ^ (((1607 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) * 2 ^ (-(256 : ℝ)) = 2 ^ (-(175.65 : ℝ)) := by
    rw [← Real.rpow_add (by norm_num)]
    congr 1
    norm_num
  rw [div_eq_mul_inv, ← Real.rpow_neg (by norm_num)]
  calc _ ≤ ((1488816950020209223842330 : ℕ) : ℝ) * 2 ^ (-(256 : ℝ)) :=
        mul_le_mul_of_nonneg_right hY (by positivity)
    _ ≤ 2 ^ (((1607 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) * 2 ^ (-(256 : ℝ)) := mul_le_mul_of_nonneg_right hK (by positivity)
    _ = 2 ^ (-(175.65 : ℝ)) := e

/-- **`ε_L ≤ 2^{−175.65}` at a compiled record and an inner protocol**, from bounds on the numbers it reads. -/
theorem epsLink_of {C W' : Type} [Fintype C] [Nonempty C] [Inhabited C] (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W') (Cp : Claims.Recursion.Caps C) (x : Msg) (n v w : ℕ)
    (hVI : I.V.k (Claims.Recursion.xI D x) ≤ 4402) (hkI : I.k (Claims.Recursion.xI D x) ≤ 12343986)
    (hwI : I.waits (Claims.Recursion.xI D x) = 210) (hW : Cp.width (Claims.Recursion.xI D x) ≤ 5591040)
    (hnS : D.nS x ≤ 16) (hSet : Claims.Recursion.setupLen D x ≤ 2 ^ 32) (hU : n ≤ 2 ^ 32) (hV : v ≤ 2 ^ 32)
    (hw : w ≤ 2 ^ 32) (hws : ∀ s : Fin (D.nS x), (D.nums x s).waits = 210)
    (hks : ∀ s : Fin (D.nS x), (D.nums x s).k ≤ 12343986) :
    Claims.RecGeneric.epsLink D I Cp x n v w ≤ 2 ^ (-(175.65 : ℝ)) := by
  unfold Claims.RecGeneric.epsLink Claims.RecGeneric.linkCalls Claims.Recursion.βdev2
  exact epsLink_core (nS := D.nS x) (VI := I.V.k (Claims.Recursion.xI D x)) (kI := I.k (Claims.Recursion.xI D x))
    (wI := I.waits (Claims.Recursion.xI D x)) (W := Cp.width (Claims.Recursion.xI D x))
    (NI := Claims.Recursion.NI D I x) (cBe := Claims.Recursion.cBe D x) (nSet := Claims.Recursion.setupLen D x)
    (ws := fun s => (D.nums x s).waits) (ks := fun s => (D.nums x s).k) rfl rfl hSet hU hV hw hVI hkI hwI hW hnS
    hws hks

/-- **The gated link's binding term at the deployment's sizes: `ε_L ≤ 2^{−175.65}`**, at `kappa_eps_deployed`'s
record, inner Flock, caps and statements, and call bounds `n`, `v` and `w` (the calls `E_O`'s witnesses record) at
most `2^32`. With `Claims.RecGeneric.linkTerm_le` it makes `KSLinkOn`'s binding term at `M = 2^256` at most
`t·2^{−175.65}`. -/
theorem epsLink_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts) (h1 : sts.size = 1)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts' ∧ sts'.size = 1)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 32) {n : ℕ} (hU : n ≤ 2 ^ 32) {v : ℕ} (hV : v ≤ 2 ^ 32) {c : ℕ}
    (hc : c ≤ 2 ^ 32) :
    Claims.RecGeneric.epsLink
        ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
        FlockDeploy.KS.CpDep x n v c ≤ 2 ^ (-(175.65 : ℝ)) := by
  have hn := nS_le_sixteen hO
  have hVI : (FlockDeploy.deployPlain rel).V.k (xIOf x) ≤ 4402 := by
    have := FlockDeploy.Vk_plain_le rel hin
    omega
  have hkI : (FlockDeploy.deployPlain rel).k (xIOf x) ≤ 12343986 := by
    obtain ⟨-, hhi⟩ := FlockDeploy.N₀_le hin.setup hin.size_pos fun st hst =>
      have := hin.tab st hst
      FlockDeploy.tableLen_plain_bounds st this.1 this.2.1
    rw [h1, Nat.cast_one, one_mul] at hhi
    exact k_le_one (N := FlockDeploy.N₀ false (xIOf x)) hhi
  refine epsLink_of ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
    (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
    FlockDeploy.KS.CpDep x n v c hVI hkI rfl (CpDep_width_le_tight (xIOf x)) hn hS hU hV hc (fun _ => rfl)
    (fun j => ?_)
  obtain ⟨i, spec', sts', hp, hp1⟩ := hO j j.isLt
  obtain ⟨-, hhi⟩ := N₀O_le hp
  rw [hp1, Nat.cast_one, one_mul] at hhi
  exact (numsOf_k _).le.trans (k_le_one (N := N₀O (staged b.st x (pick b.st x j) j)) hhi)

end RecDeploy
