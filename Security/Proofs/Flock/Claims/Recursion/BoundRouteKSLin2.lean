import Proofs.Flock.Claims.Recursion.BoundRouteMergedLin
import Proofs.Flock.Claims.Recursion.KSLin2
import Proofs.Flock.Claims.Recursion.LastBindsEC
import Proofs.Flock.Claims.Recursion.CallsAgree
import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Recursion.SetupCalls

/-!
# `KSLin2` at the deployment (route 2C)

`RecDeploy.ksBoundRouteLin_merged` reaches `KSBoundLin` from (P1) at `DE` and pays the inner root `√(γ_BI·Pr[B_I])`.
`RecDeploy.ksLin2_route` reaches `KSLin2` from `KSFlockLin` at `DE` instead, on the same facts at `deployRecC` and with
no weight premise: `Bound.ksLin2_of_flockLin_binds` there, carried to the deployed record. It is stated at the dense
statements (`KSLin2On … DenseAt`), from V*'s facts at the dense shape only: the bound at one statement reads
`InnerAcceptsColl` and `ForksBind` only there (`InnerAcceptsCollAt`, `ForksBindAt`). `E*`'s output is the same
(`E_eval_C`), and of the bound's terms only the composite finders read an inner view, through `accAtI`, which decides
alike at the two records (`accAtI_C`, from `innerView_C`).
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

section Cols

variable {Col T : Type}

/-- `dec` decides at `deployRecC` as at the deployed record (`innerView_C`). -/
theorem dec_out_C [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T) (ivScope : Msg → Bool) (b : Brings)
    (hCF : Bd.LastCoinFree H) (x : Msg) (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
    (o : Option (List Claims.Flock.Wit)) :
    (Claims.Recursion.dec H (deployRecC Bd ivScope b) x r o).out =
      (Claims.Recursion.dec H (deployRec Bd.IVStar ivScope b) x r o).out := by
  cases o with
  | none => rfl
  | some ws => exact (innerView_C H Bd ivScope b hCF x r ws).1

/-- A draw's inner view accepts at `deployRecC` where it does at the deployed record: `E_O` reads no inner view, and
`dec` decides alike (`dec_out_C`). -/
theorem accAtI_C {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (hCF : Bd.LastCoinFree H) {Ω : Type} (P : Prover Ω) (x : Msg) :
    Claims.Recursion.accAtI H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x =
      Claims.Recursion.accAtI H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x := by
  funext ω d
  unfold Claims.Recursion.accAtI
  rw [dec_out_C H Bd ivScope b hCF x d.1]
  try rfl

/-- `FCI` at `deployRecC` is `FCI` at the deployed record: it reads the inner view only through `accAtI`. -/
theorem PrFCI_C {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (hCF : Bd.LastCoinFree H) (Cp : Claims.Recursion.Caps Coin) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) :
    Claims.Recursion.PrFCI H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M =
      Claims.Recursion.PrFCI H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M := by
  have h := accAtI_C H Bd decI ivScope b DO decO hCF P x
  unfold Claims.Recursion.PrFCI Claims.Recursion.FCI Claims.Recursion.opensAtI
  rw [h]
  try rfl

/-- `F0I` at `deployRecC` is `F0I` at the deployed record: it reads the inner view only through `accAtI`. -/
theorem PrF0I_C {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (hCF : Bd.LastCoinFree H) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (R : ℕ) :
    Claims.Recursion.PrF0I H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x R =
      Claims.Recursion.PrF0I H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x R := by
  have h := accAtI_C H Bd decI ivScope b DO decO hCF P x
  unfold Claims.Recursion.PrF0I Claims.Recursion.F0I
  rw [h]
  try rfl

/-- **`KSLin2`'s bound at `deployRecC` is its bound at the deployed record**: `E*`'s output is the same
(`E_eval_C`), and of the bound's terms only `PrFCI` and `PrF0I` read an inner view (`PrFCI_C`, `PrF0I_C`). -/
theorem ksLin2_of_C {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] [Zero Col'] (H : Msg → Msg)
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (hCF : Bd.LastCoinFree H)
    (Cp : Claims.Recursion.Caps Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M R : ℕ)
    (h : Claims.PrBad H (deployRecC Bd ivScope b).V
        (Claims.Recursion.R (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI))
        (Claims.Recursion.E H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      Claims.Recursion.κ (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (Claims.Recursion.γ (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) x *
          Claims.PrB H (deployRecC Bd ivScope b).V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (Claims.Recursion.xI (deployRecC Bd ivScope b) x) + 1) *
          Claims.Recursion.PrBinner H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x +
        4 * Cp.width (Claims.Recursion.xI (deployRecC Bd ivScope b) x) *
          (Claims.Recursion.PrFCI H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M +
            1 / ((M : ℝ) + 1)) +
        4 * (Claims.Recursion.PrF0I H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (Claims.Recursion.xI (deployRecC Bd ivScope b) x) : ℝ) + 1) /
            ((R : ℝ) + 1)) +
        Claims.Recursion.βdev2 (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) Cp x *
          (Claims.Recursion.PrFdev H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x M +
            1 / ((M : ℝ) + 1))) :
    Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
        (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
        (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      Claims.Recursion.κ (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (Claims.Recursion.γ (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x *
          Claims.PrB H (deployRec Bd.IVStar ivScope b).V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (Claims.Recursion.xI (deployRec Bd.IVStar ivScope b) x) + 1) *
          Claims.Recursion.PrBinner H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x +
        4 * Cp.width (Claims.Recursion.xI (deployRec Bd.IVStar ivScope b) x) *
          (Claims.Recursion.PrFCI H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M +
            1 / ((M : ℝ) + 1)) +
        4 * (Claims.Recursion.PrF0I H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (Claims.Recursion.xI (deployRec Bd.IVStar ivScope b) x) : ℝ) + 1) /
            ((R : ℝ) + 1)) +
        Claims.Recursion.βdev2 (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) Cp x *
          (Claims.Recursion.PrFdev H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x M +
            1 / ((M : ℝ) + 1)) := by
  rw [PrFCI_C H Bd decI ivScope b DO decO hCF Cp P x M, PrF0I_C H Bd decI ivScope b DO decO hCF P x R] at h
  have hE : Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P =
    Claims.PrBad H (deployRecC Bd ivScope b).V
      (Claims.Recursion.R (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P := by
    unfold Claims.PrBad
    exact Claims.Pr_congr fun s => by
      rw [E_eval_C H Bd decI ivScope b DO decO hCF x P s.1 s.2.1 s.2.2]
      try exact Iff.rfl
  exact hE.trans_le h

/-- **`ksLin2_of_C` at the V finders**: the finders read the calls at each draw's coins (`Deploy.callsAt`), which are
the same at both records, and the inner view only through `accAtI` (`PrFCI_C`, `PrF0I_C`). -/
theorem ksLin2_of_C_setup {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] [Zero Col'] (H : Msg → Msg)
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (hCF : Bd.LastCoinFree H)
    (Cp : Claims.Recursion.Caps Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M R : ℕ)
    (h : Claims.PrBad H (deployRecC Bd ivScope b).V
        (Claims.Recursion.R (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI))
        (Claims.Recursion.E H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      Claims.Recursion.κ (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (Claims.Recursion.γ (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) x *
          Claims.PrB H (deployRecC Bd ivScope b).V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (Claims.Recursion.xI (deployRecC Bd ivScope b) x) + 1) *
          Claims.Recursion.PrBinnerV H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO
            P x +
        4 * Cp.width (Claims.Recursion.xI (deployRecC Bd ivScope b) x) *
          (Claims.Recursion.PrFCI H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO Cp
            P x M + 1 / ((M : ℝ) + 1)) +
        4 * (Claims.Recursion.PrF0I H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P
            x R +
          (((Inner.ofFlock Bd.deploy decI).k (Claims.Recursion.xI (deployRecC Bd ivScope b) x) : ℝ) + 1) /
            ((R : ℝ) + 1)) +
        Claims.Recursion.βdev2 (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) Cp x *
          (Claims.Recursion.PrFdevV H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P
            x M + 1 / ((M : ℝ) + 1))) :
    Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
        (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
        (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      Claims.Recursion.κ (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (Claims.Recursion.γ (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x *
          Claims.PrB H (deployRec Bd.IVStar ivScope b).V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (Claims.Recursion.xI (deployRec Bd.IVStar ivScope b) x) + 1) *
          Claims.Recursion.PrBinnerV H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
            decO P x +
        4 * Cp.width (Claims.Recursion.xI (deployRec Bd.IVStar ivScope b) x) *
          (Claims.Recursion.PrFCI H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
            decO Cp P x M + 1 / ((M : ℝ) + 1)) +
        4 * (Claims.Recursion.PrF0I H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
            decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (Claims.Recursion.xI (deployRec Bd.IVStar ivScope b) x) : ℝ) + 1) /
            ((R : ℝ) + 1)) +
        Claims.Recursion.βdev2 (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) Cp x *
          (Claims.Recursion.PrFdevV H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO
            decO P x M + 1 / ((M : ℝ) + 1)) := by
  rw [PrFCI_C H Bd decI ivScope b DO decO hCF Cp P x M, PrF0I_C H Bd decI ivScope b DO decO hCF P x R] at h
  have hE : Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P =
    Claims.PrBad H (deployRecC Bd ivScope b).V
      (Claims.Recursion.R (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P := by
    unfold Claims.PrBad
    exact Claims.Pr_congr fun s => by
      rw [E_eval_C H Bd decI ivScope b DO decO hCF x P s.1 s.2.1 s.2.2]
      try exact Iff.rfl
  exact hE.trans_le h

end Cols

/-- **`VStarAcceptsV` at V\*'s bound layout**: `vstarAccepts_bound` with each part's check holding off a collision
among its recorded calls and the compiled verifier's own at its V\* statement (`PartChecksBV`), which are among the
run's evaluations (`vcallsAt_sub`). -/
theorem vstarAcceptsV_bound {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) (decC : Msg → Msg → Coin)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg)
    (hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (Claims.run H (Claims.replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x))
    (hRP : Claims.Recursion.VStar.RunPairs H (deployRec I.V ivScope { b with rd := L.readers }) I decC)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hShows : Claims.Recursion.VStarB.VstarShowsB (deployRec I.V ivScope { b with rd := L.readers }) I L shape decC
      shows cmt)
    (hChk : ∀ s, Claims.Recursion.VStarB.PartChecksBV H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt chk s)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompB H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape chk)
    (hRows : L.ReadsRows H)
    (hRRP : ReadersRunPlayed H I.V { b with rd := L.readers } DO) :
    Claims.Recursion.VStarAcceptsV H (deployRec I.V ivScope { b with rd := L.readers }) I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  have hS := Claims.Recursion.setupCalls_true H (deployRec I.V ivScope { b with rd := L.readers })
    (Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r)
  obtain ⟨hlenR, hpr⟩ := hRP x Ω P ω r hacc
  obtain ⟨hK, hcl, hlast⟩ := hHeld x st hst hf
  change I.V.k (xIOf x) + kS x ≤ _ at hlenR
  change I.V.k (xIOf x) = L.rounds x at hK
  have hpl : ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
      (I.V.k (xIOf x) + kS x)).length = I.V.k (xIOf x) + kS x := by
    simp only [List.length_take]
    omega
  have hc : ∀ i < ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
      (I.V.k (xIOf x) + kS x)).length, ∃ m,
      (((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
        (I.V.k (xIOf x) + kS x)).getD i ([], .rej)).2 = .msg m ∧ decC x m = Claims.coinSeq r i := by
    intro i hi
    rw [hpl] at hi
    obtain ⟨m, hm, hd⟩ := hpr i hi
    refine ⟨m, ?_, hd⟩
    rwa [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hi, ← List.getD_eq_getElem?_getD]
  have hsh := hShows x st hst hf _ hpl (Claims.coinSeq r) hc
  have hlA := runAs_length H I.V { b with rd := L.readers } x P ω r hacc
  have hcA : (runCs (I.V.k (xIOf x) + kS x) r).length = _ :=
    (runCs_length I.V { b with rd := L.readers } x r).trans hlA.symm
  have hoA := ho
  rw [runVstar_staged H I ivScope { b with rd := L.readers } P ω r hacc] at hoA
  rcases hRRP x _ _ ws hlA hcA (stagedAll_run H I.V { b with rd := L.readers } x P ω r hacc) hoA htrue with
    hag | hcoll
  swap
  · exact Or.inr (Claims.Recursion.coll_append htrue hS hcoll)
  have hrd : ∀ j, j + 1 < I.V.k (xIOf x) → (L.msgs x ws).getD j [] =
      (((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
        (I.V.k (xIOf x) + kS x)).map Prod.fst).getD j [] := by
    intro j hj
    have hj' : j < (Claims.run H (Claims.replay (L.msgs x ws)) I.V (xIOf x) ()
        (fun i => (runCs (I.V.k (xIOf x) + kS x) r).getD i.val default)).msgs.length := by
      rw [hplay]
      omega
    have h1 : ((Claims.run H (Claims.replay (L.msgs x ws)) I.V (xIOf x) ()
        (fun i => (runCs (I.V.k (xIOf x) + kS x) r).getD i.val default)).msgs.map Prod.fst).getD j [] = _ :=
      hag j hj' hj
    rw [Claims.Recursion.VStar.run_replay_fst H I.V (xIOf x) (L.msgs x ws) _ hj'] at h1
    exact h1
  by_cases hpc : ∃ s, ∃ hs : s < ws.length, (Claims.findColl (L.calls x s ws[s] ++
      Claims.Recursion.VStarB.vcallsAt H (deployRec I.V ivScope { b with rd := L.readers })
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s []))).isSome
  · obtain ⟨s, hs, h⟩ := hpc
    have hsv : s < (Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).length :=
      lt_of_lt_of_eq (lt_of_lt_of_eq hs ho.1) (vstar_length _ _ _).symm
    refine Or.inr (Claims.Recursion.VStar.coll_of_sub (fun e he => (List.mem_append.1 he).elim (htrue e) (hS e))
      (fun p hp => ?_) h)
    rcases List.mem_append.1 hp with hp | hp
    · exact List.mem_append_left _ (Claims.Recursion.VStar.mem_calls_of_part hs hp)
    · exact List.mem_append_right _ (Claims.Recursion.VStarB.vcallsAt_sub hsv p hp)
  have hpc' : ∀ s (hs : s < ws.length), ¬ (Claims.findColl (L.calls x s ws[s] ++
      Claims.Recursion.VStarB.vcallsAt H (deployRec I.V ivScope { b with rd := L.readers })
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s []))).isSome :=
    fun s hs h => hpc ⟨s, hs, h⟩
  have hchk : ∀ s < (deployRec I.V ivScope { b with rd := L.readers }).nS x,
      chk s x (Claims.coinSeq r) (L.msgs x ws) := by
    intro s hs
    have hsw : s < ws.length := ho.1 ▸ hs
    have hts : ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 :=
      fun e he => htrue e (Claims.Recursion.VStar.mem_calls_of_part hsw he)
    have hshs := hsh s hs
    rcases hOpen s x st hst hf _ ws[s] (ho.2 s hsw) hts with hop | hop
    swap
    · exact absurd (Claims.Recursion.findColl_isSome_of_sub H (Claims.Recursion.VStarB.vcallsAt_append_true hts)
        (fun _ he => List.mem_append_left _ he) hop) (hpc' s hsw)
    rcases hChk s x st hst hf _ ws[s] (ho.2 s hsw) hts with hck | hck
    swap
    · exact absurd hck (hpc' s hsw)
    refine hck _ hshs.1 _ (fun i hi m hm => ?_) (fun i hi o m hm => ?_) ?_ ?_
    · have hi1 : i + 1 < L.rounds x := by
        by_contra hne
        have hil : i = I.V.k (xIOf x) - 1 := by omega
        have hn := hlast s
        change L.msgAt x s (I.V.k (xIOf x) - 1) = none at hn
        rw [← hil] at hn
        simp [Claims.Recursion.VStarB.LayoutB.held, hn] at hm
      exact (hrd i (by omega)).trans ((hshs.2 i hi1).symm.trans ((hop _ hshs.1 i hi1).1 m hm).symm)
    · obtain ⟨pre, p, post, hl, hp, he⟩ := Claims.Recursion.VStarB.LayoutB.mem_piecesOf hm
      have hne : L.pieceAt x i ≠ [] := by
        rw [hl]
        simp
      have hfx : L.fixedAt x i = none := by
        rcases hfi : L.fixedAt x i with _ | m'
        · rfl
        · exact absurd ((hcl i (by change i < I.V.k (xIOf x); omega)).2 (by simp [hfi])) hne
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
      rw [Claims.Recursion.VStarB.LayoutB.msgs_getD L x ws hi,
        Claims.Recursion.VStarB.LayoutB.msg_of_pieces hfx hne]
      exact Claims.Recursion.VStarB.LayoutB.assembled_piece hsw hl hp
    · exact Claims.Recursion.VStarB.LayoutB.cmtAgrees_of_eq fun i hi1 _ =>
        (hrd i (by omega)).trans (hshs.2 i hi1).symm
    · exact hRows x ws htrue s ws[s] (List.getElem?_eq_getElem hsw)
  rcases hDec x st hst hf (Claims.coinSeq r) ws ho.1 htrue hchk with hd | hd
  · exact Or.inl hd
  · exact Or.inr (Claims.Recursion.VStar.coll_of_sub (fun e he => (List.mem_append.1 he).elim (htrue e) (hS e))
      (fun _ h => List.mem_append_left _ h) hd)

/-- **A dense statement**: every parse of `x` has a dense inner form (`Form.Dense`), the form the deployed
`KSLin2` corollaries are stated at. -/
def DenseAt (x : Msg) : Prop := ∀ s, parse x = some s → s.form.Dense

/-- **`KSLin2` at the deployed records' dense statements from `KSFlockLin` at `DE`**: `KSBoundRouteLinMerged`'s
premises with `KSFlockLin` for (P1), no weight premise, and V*'s facts at the dense shape only give `KSLin2` at the
dense statements (`DenseAt`) of the record whose inner view runs `IVStar`: `InnerAcceptsColl` at a dense statement
(`innerAcceptsCollAtV_of_dense`), `ksBoundRouteLin_merged`'s facts at `deployRecC` there,
`Bound.ksLin2_of_flockLin_binds` there, carried back by `ksLin2_of_C`. -/
theorem ksLin2_route (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (Cp : Claims.Recursion.Caps Coin) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      Claims.Recursion.KSFlockLin FlockDeploy.sha512 Bd.DE Cp → Bd.Binds → Bd.LastBindsEC FlockDeploy.sha512 →
      Bd.LastCoinFree FlockDeploy.sha512 →
      Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
      (∃ decC : Msg → Msg → Coin,
        (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
          (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
        ∀ shape ∈ [Claims.Recursion.Form.Dense],
          ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
            (cmt : Msg → ℕ → Msg → Msg),
            (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
            (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
              decC shows cmt) ∧
            (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk
              s) ∧
            (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
            Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO) Cp DenseAt := by
  intro nextE colsE wE join rowIn final w Bd D bL hKS hBinds hEC hCF hF hst hT hRRP decI hdI decO hdO x hx Ω _ _ P
    M R
  obtain ⟨decC, hRP, hshapes⟩ := hst
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (run FlockDeploy.sha512 (replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x) :=
    fun x ms c => Bd.run_ivstar_length FlockDeploy.sha512 (xIOf x) ms c
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarAcceptsV FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨chk, shows, cmt, hHe, hSh, hCk, hOp, hDc⟩ := hshapes shape hs
    exact vstarAcceptsV_bound FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape decC chk shows cmt hplay
      (hRP decI) (hHe decI) (hSh decI) hCk hOp hDc hRows hRRP
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hA := Claims.Recursion.innerAcceptsCollAtV_of_dense (refuses I.V ivScope bL FlockDeploy.sha512) hTc
    (hacc _ (by simp)) hx
  have hLc := callsLogCollAt_deployRec FlockDeploy.sha512 I ivScope bL (deployZkRec relO)
  have hFk := forksBindAt_C_setup FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hEC hCF hTc hLc
    (fun decO x Ω P ω r c ws hg => readersRun_played FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hplay hT hRRP
      hg) x hA
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bL fun x => V_k_pos_flock false x
  have hLn := callsLogCollAt_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hLc
  have hAn := innerAcceptsCollAtV_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF x hA
  have hbody := Claims.Recursion.Bound.ksLin2_of_flockLin_binds (Bd := Bd) (decI := decI)
    (H := FlockDeploy.sha512) (D := deployRecC Bd ivScope bL) (DO := deployZkRec relO) (decO := decO) (P := P)
    (x := x) FlockDeploy.decCoin Cp hF hKS hdI hdO hdecC hBinds (refuses I.V ivScope bL FlockDeploy.sha512)
    (innerRounds I ivScope bL) (sessionNums I.V ivScope bL relO) (tableErrNonneg I.V ivScope bL)
    (sessionRounds I ivScope bL relO) (sessionAccepts I.V ivScope bL) (sessionRuns_sha512 I ivScope bL relO)
    (innerViewRun_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO)) hTc hLn hFk hAn hM
    (callsNil I.V ivScope bL FlockDeploy.sha512) M R
  exact ksLin2_of_C_setup FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) decO hCF Cp x P M R hbody

end RecDeploy
