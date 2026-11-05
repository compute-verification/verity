module

public import Proofs.Flock.Recursive.BitsKCoinTreeStatement
public import Proofs.Flock.Recursive.KSCoinTreeWitnesses

/-!
# `RecursiveBitsKDeployed`'s witnesses (the Lean check's `honest` and `nontrivial`)

`BitsKAtC` bounds the committed run's acceptance by the extractor's success at the simulating prover plus `a + t·e`
and `2 · k(x) · δ₁`, under `BitsKAt`'s hypotheses at that prover. At the verifier that accepts everything
(`RecAt.acceptAll`), at `δ₁ = 0` and the one-salt tree whose leaf is its salt, which hides at `0`
(`Witness.Hm96Hiding_holds`), the silent prover is accepted on every committed run (`Pr_runCM_acc`) and the extractor
recovers nothing (`Pr_E_none`); the hypotheses hold at `t = 1` and `n = v = 0`, since neither the silent prover nor the
accepting verifier and its record call SHA-512, so every finder finds nothing; and the bound is `a + e`
(`not_bitsKAtC_acceptAll`). Where `a + e < 1` it fails, at any statement `G` admits with a V\* session.

* **The honest lemma** (`recursiveBitsKDeployed_honest`), from `RecDeploy.HonestFixture`: `recursiveKSDeployed_honest`'s
  conclusions, with the run's statement at the deployment's sizes (`SizesAt`) beside `DenseAt`, so the honest run's
  statement is one `RecursiveBitsKDeployed` reads.
* **The nontrivial lemma** (`recursiveBitsKDeployed_nontrivial`), from `RecDeploy.KSFixture`: `BitsKAtC` at
  `RecursiveBitsKDeployed`'s values and `δ₁ = 0` is false at the verifier that accepts everything, at every brings: the
  fixture's statement is dense and at the deployment's sizes, has a V\* session, and `2^{−167.7} + 2^{−174.95} < 1`.
-/

@[expose] public section

namespace Claims.Recursion

section Committed

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- A prover that answers every round with `[]` and no SHA-512 call makes none on a run its verifier ends. -/
theorem stepWith_pLog_silent {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out)
    (hP : ∀ hist, P.next x hist ω = .ret []) (hv : ∀ as, vnext as = .ret .acc) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), 0 < n →
      (stepWith H P x ω vnext n hist as t).pLog = t.pLog
  | 0, _, _, _, h => absurd h (Nat.lt_irrefl 0)
  | _ + 1, _, _, _, _ => by simp [stepWith, hP, hv, OC.eval]

omit [Fintype Coin] [Nonempty Coin] in
/-- At a verifier that accepts at its first round, a prover that answers with `[]` and no call makes one round and no
SHA-512 call. -/
theorem calls_silent_acc {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (hP : ∀ hist ω, P.next x hist ω = .ret []) (hv : ∀ as cs, V.next x as cs = .ret .acc) (hk : 0 < V.k x) (ω : Ω)
    (r : Fin (V.k x) → Coin) : (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ 1 := by
  have hp : (run H P V x ω r).pLog = [] :=
    stepWith_pLog_silent H P x ω _ (hP · ω) (fun _ => hv _ _) (V.k x) [] [] ⟨[], [], [], .rej⟩ hk
  rw [hp, run_vLog_acc H P V x hv hk ω r]
  simp

/-- `B` finds nothing at a verifier that accepts at its first round (`prB_acc`). -/
theorem strictBirthday_B_acc {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (hv : ∀ as cs, V.next x as cs = .ret .acc) (hk : 0 < V.k x) : StrictBirthday H (B H V P x hk) := by
  have h := prB_acc H V P x hv
  unfold PrB at h
  rw [dite_eq_left hk] at h
  exact fun q _ => by rw [h]; positivity

/-- `FCI` finds nothing where the outer witnesses record no call (`prFCI_noCalls`). -/
theorem fCI_pr_noCalls (H : Msg → Msg) (D : Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) {W : Type} (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) (hW : 0 < Cp.width (xI D x)) :
    (FCI H D I DO decO Cp P x M hW).Pr H = 0 := by
  have h := prFCI_noCalls H D hc I DO decO Cp P x M
  unfold PrFCI at h
  rwa [dite_eq_left hW] at h

/-- `FCI` meets `ExpectedCR` where it finds nothing. -/
theorem fCI_expectedCR_noCalls (H : Msg → Msg) (D : Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) {W : Type} (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) (hW : 0 < Cp.width (xI D x)) :
    ExpectedCR H (FCI H D I DO decO Cp P x M hW) := by
  unfold ExpectedCR
  rw [fCI_pr_noCalls H D hc I DO decO Cp P x M hW]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- **`BitsKAtC` fails at the verifier that accepts everything**, at `δ₁ = 0`, at any statement of `G` where the
compiled verifier takes a round and the statement has a V\* session, where the outer Flock's parser refuses the empty
statement and `a + e < 1`: at the one-salt tree whose leaf is its salt, which hides at `0`, the silent prover is
accepted on every committed run (`Pr_runCM_acc`) and the extractor, run on it with any simulator's root and openings,
recovers nothing (`Pr_E_none`). Its run makes no SHA-512 call, and neither the accepting verifier nor its record does,
so `t = 1`, `n = v = 0`, every finder charged nothing finds nothing, and `BitsKAtC`'s hypotheses hold; its bound is
then `0 + (a + e) + 0 < 1`. -/
theorem not_bitsKAtC_acceptAll [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (N V : ℕ) (a e : ℝ)
    (decI : List (ℕ × Col) → List T) (hdI : decOK Bd.deploy decI) (decO : List (ℕ × Col') → List T')
    (hdO : decOK DO decO) (x : Msg) (hx : G x) (hk : 0 < (D.record Bd.IVStar).V.k x)
    (hS : 0 < (D.record Bd.IVStar).nS x) (hP : DO.parse [] = none) (hae : a + e < 1) :
    ¬ BitsKAtC H D.acceptAll Bd DO Cp G N V a e 0 := by
  intro h
  have hc : ∀ y (r : Fin ((D.acceptAll.record Bd.IVStar).V.k y) → Coin) ws,
      (D.acceptAll.record Bd.IVStar).callsAt H y r ws = [] := fun _ _ _ => rfl
  have hs : ∀ y ps, (D.acceptAll.record Bd.IVStar).vstar y ps = [] := fun _ _ => rfl
  have hv : ∀ as cs, (D.acceptAll.record Bd.IVStar).V.next x as cs = .ret .acc := fun _ _ => rfl
  have hk' : 0 < (D.acceptAll.record Bd.IVStar).V.k x := hk
  have hS' : 0 < (D.acceptAll.record Bd.IVStar).nS x := hS
  obtain ⟨S, hSim⟩ := h ENNReal.zero_ne_top decI hdI decO hdO x hx Unit Unit Unit Unit
    ⟨fun _ _ => (), id, fun _ => (), fun _ _ => [], fun _ _ _ _ _ => []⟩ Witness.Hm96Hiding_holds
  have hrun := calls_silent_acc H
    (Bridge.simProverM S (D.acceptAll.record Bd.IVStar).V.k (⟨fun _ _ _ _ _ => .ret []⟩ : Bridge.CProver Unit))
    (D.acceptAll.record Bd.IVStar).V x (fun _ _ => rfl) hv hk'
  have h1 := hSim Unit ⟨fun _ _ _ _ _ => .ret []⟩ 1 le_rfl hrun 0 (Nat.zero_le N) (fun _ _ => le_rfl) 0
    (Nat.zero_le V) (fun _ => le_rfl) (fun hk => strictBirthday_B_acc H _ _ x hv hk)
    (fun hk => Witness.Recursion.expectedCR_charge_zero (Witness.Recursion.binnerV_expectedCR_noCalls H
      (D.acceptAll.record Bd.IVStar) hc hs _ _ _ _ x hk) rfl)
    (fun M hk => Witness.Recursion.expectedCR_charge_zero (Witness.Recursion.fdevV_expectedCR_noCalls H
      (D.acceptAll.record Bd.IVStar) hc hs _ _ _ _ x M hk) rfl)
    (fun M hW => Witness.Recursion.expectedCR_charge_zero (fCI_expectedCR_noCalls H
      (D.acceptAll.record Bd.IVStar) hc _ _ _ _ _ x M hW) rfl)
    (fun R => Witness.Recursion.expectedCR_charge_zero (Witness.Recursion.f0I_expectedCR_noCalls H
      (D.acceptAll.record Bd.IVStar) hc _ _ _ _ x R) rfl)
  rw [Pr_runCM_acc H _ _ x _ hv hk', Pr_E_none H _ _ DO decO x (hs x) hS' hP, ENNReal.toReal_zero] at h1
  simp only [mul_zero, add_zero, zero_add, Nat.cast_one, one_mul] at h1
  linarith

end Committed

end Claims.Recursion

namespace RecDeploy

open Claims (Msg)

/-- **`BitsKAtC` at the verifier that accepts everything fails** at `δ₁ = 0`, at any statement of `G` with a V\* session
where `a + e < 1`, at the deployed decoders: `Claims.Recursion.not_bitsKAtC_acceptAll` at the deployed records, whose
compiled verifier takes a round at every statement (`recBound_k_pos`) and whose outer parser refuses the empty statement
(`zkRec_parse_nil`). -/
theorem not_bitsKAtC_acceptAll (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (Cp : Claims.Recursion.Caps FlockDeploy.Coin) (G : Msg → Prop) (N V : ℕ) (a e : ℝ) (x : Msg) (hx : G x)
    (hS : 0 < nS x) (hae : a + e < 1) :
    ¬ Claims.Recursion.BitsKAtC FlockDeploy.sha512 (recBoundAt rel b L).acceptAll
      (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO) Cp G N V a e 0 :=
  Claims.Recursion.not_bitsKAtC_acceptAll _ _ _ _ _ _ _ _ _ _ Claims.GS.plainDecodeRD
    (FlockDeploy.listWithin_plainTrim rel) Claims.GS.zkDecodeRD (Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl) x hx
    (recBound_k_pos rel b L nextE colsE wE join rowIn final w x) hS (zkRec_parse_nil relO) hae

/-- `recursiveBitsK_deployed`'s numbers sum below `1`: `2^{−167.7} + 2^{−174.95} < 1`. -/
theorem bitsK_ae_lt_one : (2 : ℝ) ^ (-(167.7 : ℝ)) + 2 ^ (-(174.95 : ℝ)) < 1 := by
  have h1 : (2 : ℝ) ^ (-(167.7 : ℝ)) < 2 ^ (-1 : ℝ) := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num)
  have h2 : (2 : ℝ) ^ (-(174.95 : ℝ)) < 2 ^ (-1 : ℝ) := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num)
  rw [Real.rpow_neg_one] at h1 h2
  linarith [show (2 : ℝ)⁻¹ = 1 / 2 by norm_num]

end RecDeploy

namespace RecDeploy.HonestHalfFixture

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest
open RecDeploy.HonestHalf

/-- **`RecursiveBitsKDeployed`'s honest lemma**: `recursiveKSDeployed_honest`'s conclusions at `HonestFixture`'s run,
with its statement at the deployment's sizes beside `DenseAt` (`SizesAt`, at the run's coin-server record), where
`RecursiveBitsKDeployed` reads it: every V\* part accepts, the reference prover is accepted at the run's tape and
coins committed in every coin tree, the deployed decoders meet the list-decoding spec, and hm96 leaves under the pinned
key hide at its hiding gap, a finite `δ₁`. -/
theorem recursiveBitsKDeployed_honest (h : HonestFixture) :
    ∃ (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
      (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
      (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
      (w : Claims.Flock.Wit) (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
      (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin),
      (∀ s < nS x,
        Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 (DAt mask Ir b0 x aux nOf hd recRest)
          (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR OpenRows.elemR) Claims.Recursion.Form.Dense
          (showsPart recRead ({ bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } : Brings))
          (cmtTr recRead) s x (Claims.coinSeq r)
          ((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
            (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.map Prod.fst)) ∧
      (DenseAt x ∧ SizesAt { bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } x) ∧
      (∀ (T : Type) (C : Claims.Bridge.CoinTree Coin T ((DAt mask Ir b0 x aux nOf hd recRest).V.k x)) (τ : T),
        (Claims.Bridge.runCM FlockDeploy.sha512
          (Claims.Bridge.CProver.ofProver (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w))
          (DAt mask Ir b0 x aux nOf hd recRest).V x C ω r τ).out = .acc) ∧
      Claims.Recursion.decOK DeployedR.bdR.deploy Claims.GS.plainDecodeRD ∧
      Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) Claims.GS.zkDecodeRD ∧
      Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash
        (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
          FlockSoundness.Discharge.ZkView.saltHash)) ∧
      ENNReal.ofReal (FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
        FlockSoundness.Discharge.ZkView.saltHash) ≠ ⊤ := by
  obtain ⟨mask, b0, x, aux, nOf, hd, recRest, decI, w, st, ω, r, hR, hW, hN, hT, hZ⟩ := h
  refine ⟨mask, IrDep x, b0, x, aux, nOf, hd, recRest, decI, w, ω, r,
    HonestHalfTops.honest_half_tops mask b0 x aux nOf hd recRest decI ω r hR hW hN hT, ⟨fun s hs => ?_, hZ⟩,
    fun _ C τ => ?_, FlockDeploy.listWithin_plainTrim _root_.Flock.Readable.relation,
    Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl, FlockSoundness.CROnly.hm96Hiding_gap _ _,
    ENNReal.ofReal_ne_top⟩
  · obtain rfl : st = s := Option.some.inj (hR.2.1.symm.trans hs)
    exact hR.2.2.1
  · rw [Claims.Bridge.runCM_ofProver]
    exact hR.2.2.2

end RecDeploy.HonestHalfFixture

namespace Flock.Guarantees

open Claims (Msg Out)

/-- **`RecursiveBitsKDeployed`'s nontrivial lemma**: its conclusion, `BitsKAtC` at the deployed values, is false at the
verifier that accepts everything (`RecAt.acceptAll` of the deployment's compiled record), at `δ₁ = 0` and every brings,
from the named assumption `RecDeploy.KSFixture`: the fixture's statement is dense (`denseAt_of_parses`), at the
deployment's sizes (`SizesAt`: its inner statement in the plain scope with one table, `plainIn_one_of_parses`, and its
parts, their number and its inner setup's calls as `PartsInScope` reads them) and has a V\* session, and
`2^{−167.7} + 2^{−174.95} < 1` (`RecDeploy.bitsK_ae_lt_one`), so `RecDeploy.not_bitsKAtC_acceptAll` applies. -/
theorem recursiveBitsKDeployed_nontrivial (h : RecDeploy.KSFixture) :
    ∃ (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
      (recRest : Msg → List (Msg × Out) → List UInt8), ∀ b : RecDeploy.Brings,
      ¬ Claims.Recursion.BitsKAtC FlockDeploy.sha512
        (RecDeploy.recBoundAt _root_.Flock.Readable.relation
          { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)).acceptAll
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
          RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
          RecDeploy.DeployedR.wR)
        (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep
        (fun x => RecDeploy.DenseAt x ∧
          RecDeploy.SizesAt { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } x)
        (2 ^ 32) (2 ^ 32) (2 ^ (-(167.7 : ℝ))) (2 ^ (-(174.95 : ℝ))) 0 := by
  obtain ⟨x, aux, nOf, hd, recRest, hP, hO, hn, hS⟩ := h
  refine ⟨aux, nOf, hd, recRest, fun b => ?_⟩
  exact RecDeploy.not_bitsKAtC_acceptAll _root_.Flock.Readable.relation _root_.Flock.Readable.relation
    { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
    (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols RecDeploy.DeployedR.wER
    FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wR FlockDeploy.KS.CpDep
    (fun x => RecDeploy.DenseAt x ∧
      RecDeploy.SizesAt { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } x)
    (2 ^ 32) (2 ^ 32) (2 ^ (-(167.7 : ℝ))) (2 ^ (-(174.95 : ℝ))) x
    -- `SizesAt` states the parts' count, or, where the verifier refuses past 16 parts, only the inner setup's calls
    ⟨denseAt_of_parses hP, plainIn_one_of_parses hP, hO, by first | exact ⟨hn, hS⟩ | exact hS⟩ hP.2.1
    RecDeploy.bitsK_ae_lt_one

end Flock.Guarantees
