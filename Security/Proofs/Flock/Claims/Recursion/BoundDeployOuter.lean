import Proofs.Flock.Claims.KS.ZkOuter
import Proofs.Flock.Claims.Recursion.DeployLayout
import Proofs.Flock.Claims.Recursion.BoundRouteCols
import Proofs.Flock.Claims.Recursion.LastBindsDeployed
import Proofs.Flock.Claims.Recursion.BoundWeight
import Proofs.Flock.Claims.Recursion.PlainKSAbortDeployed

/-!
# The deployed recursion, with Flock's KS at the outer record proved

The deployed recursion theorems take Flock's KS at the outer `--zk` record, `Claims.Flock.KS FlockDeploy.sha512
(deployZkRec relO)`, as a premise. At the deployed outer relation, `relO = Readable.relation` (the V* relation the
honest staging meets, `DeployHonest`), it is `FlockDeploy.KS.zk_KS_outer`. Each theorem here is one of them at that
`relO`, without the premise:

* **The outer relation alone** (`_outer`, the inner `rel` still free): `ksA_deployed_outer` and `ksFlock_deployed_outer`
  (`DeployPremises`), `ksA_deployed_reg_outer`, `ksFlock_deployed_reg_outer` and `ksA_deployed_parts_outer`
  (`DeployReg`), `ksA_deployed_concrete_outer` and `ksFlock_deployed_concrete_outer` (`DeployLayout`); `KSBoundRoute`'s
  ∀-form (`ksBoundRoute_outer`) and the A′ route's, with `LastBindsEC` and `LastCoinFree` for `LastBinds`
  (`ksBoundRouteCols_outer`, at the bridge `bridge_bdDeployed`).
* **Both relations** (`_relation`, `rel = Readable.relation` too): the `ksA` family reads the plain record's table bound
  (`TableBoundθOnAcc`) only for `Inner.KSAbort` at the plain record, which `plain_KSAbortR` gives there with no premise
  (`plain_tableBoundθOnAccR`), so the bound goes as well.
* **The A′ route at the intended choice** (`ksBoundDeployed_intended`): (P1) to (P4′) at the deployed plain
  verifier's own steps, columns and weight and the deployed rows' digest input, from `PlainLastBinds` and `γ`'s weight
  at the floor.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-! ## At the deployed outer relation -/

/-- **`ksA_deployed` at the deployed outer relation**, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksA_deployed_outer (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) :=
  ksA_deployed (relO := Flock.Readable.relation) (hF := FlockDeploy.KS.zk_KS_outer) rel ivScope b opening htab hT hL hO
    hD hH hS decI hdec

/-- **`ksFlock_deployed` at the deployed outer relation**, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksFlock_deployed_outer (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec Flock.Readable.relation) :=
  ksFlock_deployed (relO := Flock.Readable.relation) (hF := FlockDeploy.KS.zk_KS_outer) rel ivScope b opening htab hT hL
    hO hD hH hS

/-- **`ksA_deployed_reg` at the deployed outer relation**, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksA_deployed_reg_outer (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogCollR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayedR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) :=
  ksA_deployed_reg (relO := Flock.Readable.relation) (hF := FlockDeploy.KS.zk_KS_outer) rel ivScope b opening htab hT hL
    hO hD hH hS decI hdec

/-- **`ksFlock_deployed_reg` at the deployed outer relation**, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksFlock_deployed_reg_outer (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogCollR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayedR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec Flock.Readable.relation) :=
  ksFlock_deployed_reg (relO := Flock.Readable.relation) (hF := FlockDeploy.KS.zk_KS_outer) rel ivScope b opening htab
    hT hL hO hD hH hS

/-- **`ksA_deployed_parts` at the deployed outer relation**, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksA_deployed_parts_outer (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (L : Claims.Recursion.VStar.Layout) (hb : b.rd = readersOf L) (opening : Claims.Recursion.Opening)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI)
    (hT : ∀ s, Claims.Recursion.VStar.PartCallsTrue FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (deployZkRec Flock.Readable.relation) L s)
    (hPub : Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) L)
    (hFin : Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      opening)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) L
      opening Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) L
      opening Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) L
      opening Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) :=
  ksA_deployed_parts (relO := Flock.Readable.relation) (hF := FlockDeploy.KS.zk_KS_outer) rel ivScope b L hb opening
    htab decI hdec hT hPub hFin hD hH hS

/-- **`ksA_deployed_concrete` at the deployed outer relation**, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksA_deployed_concrete_outer (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI)
    (hPub : Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf)
    (hFin : Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      opening)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf opening Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf opening Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf opening Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation) :=
  ksA_deployed_concrete (relO := Flock.Readable.relation) (hF := FlockDeploy.KS.zk_KS_outer) rel ivScope b opening htab
    decI hdec hPub hFin hD hH hS

/-- **`ksFlock_deployed_concrete` at the deployed outer relation**, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksFlock_deployed_concrete_outer (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (ivScope : Msg → Bool) (b : Brings) (opening : Claims.Recursion.Opening)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hPub : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI → Claims.Recursion.VStar.ReadsPub
      FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf)
    (hFin : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI → Claims.Recursion.VStar.FinalOpens
      FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      opening)
    (hD : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI → Claims.Recursion.VStar.VStarParts
      FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf opening Claims.Recursion.Form.Dense)
    (hH : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI → Claims.Recursion.VStar.VStarParts
      FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf opening Claims.Recursion.Form.Holo)
    (hS : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI → Claims.Recursion.VStar.VStarParts
      FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec Flock.Readable.relation)
      layoutOf opening Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec Flock.Readable.relation) :=
  ksFlock_deployed_concrete (relO := Flock.Readable.relation) (hF := FlockDeploy.KS.zk_KS_outer) rel ivScope b opening
    htab hPub hFin hD hH hS

/-! ## The routes at the deployed outer relation -/

/-- **`KSBoundRoute` at the deployed outer relation, with no premise on the outer record**: `ksBoundRoute`, where
Flock's KS at the outer record is `zk_KS_outer`. -/
theorem ksBoundRoute_outer (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.WeightCovers → Bd.LastBinds FlockDeploy.sha512 →
      (∃ decC : Msg → Msg → Coin,
        (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
          (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
        ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
          ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
            (cmt : Msg → ℕ → Msg → Msg),
            (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
            (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
              decC shows cmt) ∧
            (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec Flock.Readable.relation) L
              shape shows cmt chk s) ∧
            (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec Flock.Readable.relation) L shape
              shows cmt s) ∧
            Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
      ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
      ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
      KSBoundDeployed rel Flock.Readable.relation b L nextE colsE wE join rowIn final w := by
  intro nextE colsE wE join rowIn final w Bd D bL hP1 hBinds hWC hLB hst hT hL hRRP
  exact ksBoundRoute rel Flock.Readable.relation b L hRows nextE colsE wE join rowIn final w hP1 hBinds hWC hLB
    FlockDeploy.KS.zk_KS_outer hst hT hL hRRP

/-- **`KSBoundRouteCols` at the deployed outer relation, with no premise on the outer record and none on the
bridge**: `ksBoundRoute_of_bridge_cols` at `bridge_bdDeployed`, where Flock's KS at the outer record is
`zk_KS_outer`. -/
theorem ksBoundRouteCols_outer (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.WeightCovers → Bd.LastBindsEC FlockDeploy.sha512 →
      Bd.LastCoinFree FlockDeploy.sha512 →
      (∃ decC : Msg → Msg → Coin,
        (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
          (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
        ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
          ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
            (cmt : Msg → ℕ → Msg → Msg),
            (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
            (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
              decC shows cmt) ∧
            (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec Flock.Readable.relation) L
              shape shows cmt chk s) ∧
            (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec Flock.Readable.relation) L shape
              shows cmt s) ∧
            Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
      ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
      ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
      KSBoundDeployed rel Flock.Readable.relation b L nextE colsE wE join rowIn final w := by
  intro nextE colsE wE join rowIn final w Bd D bL hP1 hBinds hWC hEC hCF hst hT hL hRRP
  exact ksBoundRoute_of_bridge_cols rel Flock.Readable.relation b L hRows (bridge_bdDeployed rel) nextE colsE wE join
    rowIn final w
    hP1 hBinds hWC hEC hCF FlockDeploy.KS.zk_KS_outer hst hT hL hRRP

/-! ## At the deployed relations, inner and outer -/

/-- **`ksA_deployed` at the deployed relations**, inner and outer, with neither premise: Flock's KS at the
outer record is `zk_KS_outer`, and the plain record's `Inner.KSAbort` is `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem ksA_deployed_relation (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) :=
  let I := Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI
  Claims.Recursion.KS_of_flockKS_coll FlockDeploy.sha512 (deployRec I.V ivScope b) I
    (deployZkRec Flock.Readable.relation) opening FlockDeploy.decCoin FlockDeploy.KS.zk_KS_outer
    (FlockDeploy.KS.plain_KSAbortR Flock.Readable.relation FlockDeploy.KS.plain_tableBoundθOnAccR decI hdec)
    (refuses I.V ivScope b FlockDeploy.sha512) (innerRounds I ivScope b)
    (sessionNums I.V ivScope b Flock.Readable.relation) (tableErrNonneg I.V ivScope b)
    (sessionRounds I ivScope b Flock.Readable.relation) (sessionAccepts I.V ivScope b)
    (sessionRuns_sha512 I ivScope b Flock.Readable.relation)
    (innerViewRun FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation))
    (callsTrue FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) hT)
    (callsLogColl FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) hT hL)
    (callsOpenPlayed FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) opening hO)
    (innerAcceptsColl FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) hT hD hH hS)
    (roundsMargin_deployRec_V false FlockDeploy.sha512 I rfl ivScope b)
    (FlockDeploy.coinsShown_V FlockDeploy.sha512 false) (callsNil I.V ivScope b FlockDeploy.sha512)

/-- **`ksFlock_deployed` at the deployed relations**, inner and outer, with neither premise: Flock's KS at the
outer record is `zk_KS_outer`, and the plain record's `Inner.KSAbort` is `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem ksFlock_deployed_relation (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (FlockDeploy.deployPlainTrim Flock.Readable.relation) (deployZkRec Flock.Readable.relation) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec => ksA_deployed_relation ivScope b opening hT hL hO hD
    hH hS decI hdec

/-- **`ksA_deployed_reg` at the deployed relations**, inner and outer, with neither premise: Flock's KS at the
outer record is `zk_KS_outer`, and the plain record's `Inner.KSAbort` is `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem ksA_deployed_reg_relation (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogCollR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayedR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) :=
  let I := Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI
  Claims.Recursion.VStar.KS_of_flockKS_played FlockDeploy.sha512 (deployRec I.V ivScope b) I
    (deployZkRec Flock.Readable.relation) opening FlockDeploy.decCoin FlockDeploy.KS.zk_KS_outer
    (FlockDeploy.KS.plain_KSAbortR Flock.Readable.relation FlockDeploy.KS.plain_tableBoundθOnAccR decI hdec)
    (refuses I.V ivScope b FlockDeploy.sha512) (innerRounds I ivScope b)
    (sessionNums I.V ivScope b Flock.Readable.relation) (tableErrNonneg I.V ivScope b)
    (sessionRounds I ivScope b Flock.Readable.relation) (sessionAccepts I.V ivScope b)
    (sessionRuns_sha512 I ivScope b Flock.Readable.relation)
    (innerViewRun FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation))
    (callsTrue FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) hT)
    (callsLogCollR FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) hT hL)
    (callsOpenPlayedR FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) opening hT hO)
    (innerAcceptsR FlockDeploy.sha512 I ivScope b (deployZkRec Flock.Readable.relation) hT hD hH hS)
    (roundsMargin_deployRec_V false FlockDeploy.sha512 I rfl ivScope b)
    (FlockDeploy.coinsShown_V FlockDeploy.sha512 false) (callsNil I.V ivScope b FlockDeploy.sha512)

/-- **`ksFlock_deployed_reg` at the deployed relations**, inner and outer, with neither premise: Flock's KS at the
outer record is `zk_KS_outer`, and the plain record's `Inner.KSAbort` is `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem ksFlock_deployed_reg_relation (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hL : ReadersLogCollR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation))
    (hO : ReadersOpenPlayedR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation) opening)
    (hD : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Dense)
    (hH : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Holo)
    (hS : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec Flock.Readable.relation)
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (FlockDeploy.deployPlainTrim Flock.Readable.relation) (deployZkRec Flock.Readable.relation) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec => ksA_deployed_reg_relation ivScope b opening hT hL
    hO hD hH hS decI hdec

/-- **`ksA_deployed_parts` at the deployed relations**, inner and outer, with neither premise: Flock's KS at the
outer record is `zk_KS_outer`, and the plain record's `Inner.KSAbort` is `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem ksA_deployed_parts_relation (ivScope : Msg → Bool)
    (b : Brings) (L : Claims.Recursion.VStar.Layout) (hb : b.rd = readersOf L) (opening : Claims.Recursion.Opening)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
    (hT : ∀ s, Claims.Recursion.VStar.PartCallsTrue FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (deployZkRec Flock.Readable.relation) L s)
    (hPub : Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) L)
    (hFin : Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) opening)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) L opening Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) L opening Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) L opening Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) :=
  let I := Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI
  have hR := readsThrough_deployRec FlockDeploy.sha512 I.V ivScope b L hb
  have hcf := Claims.Recursion.VStar.codeFacts_of_parts (refuses I.V ivScope b FlockDeploy.sha512) hR
    (vstarLen_deployRec I.V ivScope b) hT hPub hFin hD hH hS
  Claims.Recursion.VStar.KS_of_flockKS_played FlockDeploy.sha512 (deployRec I.V ivScope b) I
    (deployZkRec Flock.Readable.relation) opening FlockDeploy.decCoin FlockDeploy.KS.zk_KS_outer
    (FlockDeploy.KS.plain_KSAbortR Flock.Readable.relation FlockDeploy.KS.plain_tableBoundθOnAccR decI hdec)
    (refuses I.V ivScope b FlockDeploy.sha512) (innerRounds I ivScope b)
    (sessionNums I.V ivScope b Flock.Readable.relation) (tableErrNonneg I.V ivScope b)
    (sessionRounds I ivScope b Flock.Readable.relation) (sessionAccepts I.V ivScope b)
    (sessionRuns_sha512 I ivScope b Flock.Readable.relation) hcf.1 hcf.2.2.1 hcf.2.2.2.2.1 hcf.2.2.2.1 hcf.2.2.2.2.2
    (roundsMargin_deployRec_V false FlockDeploy.sha512 I rfl ivScope b)
    (FlockDeploy.coinsShown_V FlockDeploy.sha512 false) hcf.2.1

/-- **`ksA_deployed_concrete` at the deployed relations**, inner and outer, with neither premise: Flock's KS at the
outer record is `zk_KS_outer`, and the plain record's `Inner.KSAbort` is `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem ksA_deployed_concrete_relation (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
    (hPub : Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf)
    (hFin : Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) opening)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf opening Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf opening Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf opening Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) :=
  ksA_deployed_parts_relation ivScope b.atPorts layoutOf rfl opening decI hdec (partCallsTrue_layoutOf _ _ _) hPub hFin
    hD hH hS

/-- **`ksFlock_deployed_concrete` at the deployed relations**, inner and outer, with neither premise: Flock's KS at the
outer record is `zk_KS_outer`, and the plain record's `Inner.KSAbort` is `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem ksFlock_deployed_concrete_relation
    (ivScope : Msg → Bool) (b : Brings) (opening : Claims.Recursion.Opening)
    (hPub : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI →
      Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf)
    (hFin : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI →
      Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) opening)
    (hD : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf opening Claims.Recursion.Form.Dense)
    (hH : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf opening Claims.Recursion.Form.Holo)
    (hS : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim Flock.Readable.relation) decI)
      (deployZkRec Flock.Readable.relation) layoutOf opening Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (FlockDeploy.deployPlainTrim Flock.Readable.relation) (deployZkRec Flock.Readable.relation) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec => ksA_deployed_concrete_relation ivScope b opening
    decI hdec (hPub decI hdec) (hFin decI hdec) (hD decI hdec) (hH decI hdec) (hS decI hdec)

/-! ## The A′ route at the intended choice -/

/-- **The A′ route at the intended choice, at the deployed relations**: `KSBoundDeployed` at the deployed bound record
whose element-level verifier's steps, columns and weight are the deployed plain verifier's own, whose rows' digest input
is the deployed one (`rowInDeployed`), at any join, `final` and `γ` weight at or above the floor. (P1) is
`plainKSAbort_deployed`, (P2) `binds_deployed`, (P3) the floor (`weightCovers_deployed_iff`), (P4′)
`lastBindsEC_deployed` from the plain verifier's binding of its last message (`PlainLastBinds`), `LastCoinFree`
`lastCoinFree_deployed`, the bridge `bridge_bdDeployed` and Flock's KS at the outer record `zk_KS_outer`. What stays is
`PlainLastBinds` and the code facts of the staging, V*'s parts and the readers. -/
theorem ksBoundDeployed_intended (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hw : ∀ x, coverW (FlockDeploy.deployPlainTrim Flock.Readable.relation).weight x ≤ w x)
    (hPL : PlainLastBinds FlockDeploy.sha512) :
    let Bd := bdDeployed Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim Flock.Readable.relation).cols
      (FlockDeploy.deployPlainTrim Flock.Readable.relation).weight join rowInDeployed final w
    let D := recBound Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec Flock.Readable.relation) L
            shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec Flock.Readable.relation) L
            shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec Flock.Readable.relation) →
    KSBoundDeployed Flock.Readable.relation Flock.Readable.relation b L (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim Flock.Readable.relation).cols
      (FlockDeploy.deployPlainTrim Flock.Readable.relation).weight join rowInDeployed final w := by
  intro Bd D bL hst hT hL hRRP
  exact ksBoundRouteCols_outer Flock.Readable.relation b L hRows (FlockDeploy.V false).next
    (FlockDeploy.deployPlainTrim Flock.Readable.relation).cols
    (FlockDeploy.deployPlainTrim Flock.Readable.relation).weight join rowInDeployed final w
    (plainKSAbort_deployed join rowInDeployed final w) (binds_deployed _ _ _ _ join final w)
    ((weightCovers_deployed_iff _ _ _ _ join rowInDeployed final w).2 hw)
    (lastBindsEC_deployed _ _ join final w FlockDeploy.sha512 hPL)
    (lastCoinFree_deployed _ _ _ join rowInDeployed final w FlockDeploy.sha512) hst hT hL hRRP

end RecDeploy
