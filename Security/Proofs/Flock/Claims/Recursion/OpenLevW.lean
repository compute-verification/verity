import Proofs.Flock.Claims.Recursion.ResLowerD
import Proofs.Flock.Claims.Recursion.CapLowerPin
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.RowDAccept
import Proofs.Flock.Claims.Recursion.VerdictRowDCover
import Proofs.Flock.Claims.Recursion.NonVacuous

/-!
# A row D piece's copy at the outer witnesses: its outputs (`LevelOutsW`) and openings (`OpenCopyW`)

`accHolds_of_copy` steps the chain at a piece's level copy from three facts at the copy: its columns are a level
table's outputs at what they read (`LevelOuts`), at the piece's level's lanes (`OpenCopy`'s first conjunct), reading
the piece's opened row (`OpenCopy`'s third) at the side `Rows.sideAt rd` that `Top.ChainHolds` reads. These are
the per-piece facts `StepsW` reads, at its joint acceptance with each part at its own witness (`JointP`) and a
decision, at the copy `copyR` of the piece's owner's witness `ws[s]` (`AtPiecesP` at row D's pieces).

* `stagedAt_of_jointP`: under `JointP`, every part `s` has a staged statement at its own witness (`StagedAt`).
* `levelOutsW`: the copy's outputs, outright. The staged table's copies are the level's (`levelCopies`), and the piece's
  copy is one of them (`copyFitsR_rowD`).
* `openCopy_sideAt`: at a decision, `OpenCopy` moves from the assembly's side `sideAtR rd` to `Rows.sideAt rd`. Both read
  the decided side's schedule and openings (`DecAt.sideAtR_opens`, `DecAt.sideAt_eq`), and that is all `OpenCopy`
  reads.
* `openCopyW_deployed`: the copy's openings at the deployed readers (`DeployedR.rdR`, `OpenRows.elemR`,
  `DeployedR.finalR`), from `openCopyR_of` at `sideAtR` (`copyOpensR_layR`, `copyFitsR_rowD`) and `openCopy_sideAt`.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **The context of a row D piece's copy** at `StepsW`'s joint acceptance: `AtPiecesP` at the deployed layout and
`Rows.rowDPieces key`, whose check is `AccP`'s. -/
abbrev AtRowDW (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (rd : Msg → Option FinalP) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (Fact : Msg → (ℕ → Coin) → List Claims.Flock.Wit → ℕ → ℕ → ℕ → (xO : Msg) → Claims.Flock.Wit →
      (p : _root_.Flock.Readable.Parsed) → 0 < p.sts.size → OpenPiece → Prop) : Prop :=
  OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) shape untr b cmt (DeployedR.layR elem)
    (Rows.rowDPieces key) Fact

/-- **A row D piece's copy is a level table's outputs** (`LevelOuts`), at its owner's witness. -/
def LevelOutsW (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  AtRowDW key rd wE final w b elem shape untr cmt fun x _ _ LANES HH _ _ wt p ht q =>
    _root_.Flock.Readable.LevelOuts
      (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q) LANES HH

/-- **A row D piece's copy reads the piece's opening** (`OpenCopy`) at the side `Top.ChainHolds` reads
(`Rows.sideAt rd`), at every reading of the last message and every decision of the opened messages, as `StepsW` has
them. -/
def OpenCopyW (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  AtRowDW key rd wE final w b elem shape untr cmt fun x c ws LANES HH _ _ wt p ht q =>
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (((DeployedR.layR elem).msgs x ws).getD
          ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
    ∀ _ : Rows.DecAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c),
      OpenCopy (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
          ((_root_.RecDeploy.bdR rd wE final w).opened os last) q) q LANES HH
        (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q)

section Facts

variable {rd : Msg → Option FinalP} {wE : Msg → ℝ} {final : Msg → Option (List (List (Msg × Salt)) × Msg)}
  {w : Msg → ℝ} {b : Brings} {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
  {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}

/-- **Every part has a staged statement at its own witness** under `JointP`: part `s`'s check at `ws[s]` gives its
staged statement `xO`, its parse `p`, and the copies of its V\* statement (`stmtCopies_of_allOuts`). -/
theorem stagedAt_of_jointP {x : Msg} {c : ℕ → Coin} {ws : List Claims.Flock.Wit}
    (hJ : OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws)
    {s : ℕ} (hs : s < ws.length) {r : XR} {ss : Array VStar.Stmt} {sv : VStar.Stmt} (hr : readX x = some r)
    (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss) (hsv : ss[s]? = some sv) :
    ∃ (xO : Msg) (p : _root_.Flock.Readable.Parsed),
      OpenLower.StagedAt FlockDeploy.sha512 (DRL rd wE final w b elem) untr b cmt (DeployedR.layR elem) s x c
        ((DeployedR.layR elem).msgs x ws) sv xO ws[s] p := by
  obtain ⟨-, xO, hck, hT, hc, hsh, hh, hp, hm, hrw⟩ := hJ.2.2.2 s hs
  obtain ⟨p, r', sv', hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, hsz, htab⟩ := stagedReads hsh
  have hck' : Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO ws[s] =
      true := hck
  have hcp := _root_.Flock.Readable.stmtCopies_of_allOuts htab
    (_root_.Flock.Readable.allOuts_of_check hck' hx ht hpi)
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  exact ⟨xO, p, hx, hsz, hcp, ⟨hh, hp, hm, hrw⟩, hT, hc, hsh.1, hck, hsh.2⟩

end Facts

/-- **A row D piece's copy is a level table's outputs**, outright at the dense form: the staged table's copies are its
level's (`levelCopies`), and the piece's copy is in range (`copyFitsR_rowD`). -/
theorem levelOutsW (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) :
    LevelOutsW rd key wE final w b elem shape untr cmt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hw hst q hq ho ht
  obtain ⟨hob, hgb, hub⟩ := OpenLower.atPiecesP_of (H := FlockDeploy.sha512) (D := DRL rd wE final w b elem)
    (shape := shape) (untr := untr) (b := b) (cmt := cmt) (L := DeployedR.layR elem)
    (OpenLower.copyFitsR_rowD (H := FlockDeploy.sha512) (D := DRL rd wE final w b elem) (shape := shape)
      (untr := untr) (b := b) (cmt := cmt) (L := DeployedR.layR elem) key (fun _ => rfl) hS)
    x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hw hst q hq ho ht
  exact _root_.Flock.Readable.levelCopies ht hob hgb hub hst.2.2.1

/-- **`OpenCopy` at the decided side**: at a decision, a row D piece's `OpenCopy` at the assembly's side `sideAtR rd`
is one at `Rows.sideAt rd`. Both read the decided side's schedule and openings (`DecAt.sideAtR_opens`,
`DecAt.sideAt_eq`), and `OpenCopy` reads only those. -/
theorem openCopy_sideAt {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}
    (Dc : Rows.DecAt rd x ms cs) {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {q : OpenPiece}
    (hq : q ∈ Rows.rowDPieces key x) {LANES HH : ℕ} {v : ℕ → Bool}
    (h : OpenCopy (OpenLower.sideAtR rd x ms q) q LANES HH v) : OpenCopy (Rows.sideAt rd x ms q) q LANES HH v := by
  rw [Dc.rowDPieces_eq key] at hq
  obtain ⟨hi, hr, -⟩ := Rows.mem_rowDPiecesOf.1 hq
  obtain ⟨hsch, -, hop⟩ := Dc.sideAtR_opens Top.cls₀ q.table hi ⟨q.rep, hr⟩ q rfl rfl
  rw [Dc.sideAt_eq Top.cls₀ q.table hi ⟨q.rep, hr⟩ q rfl rfl]
  unfold OpenCopy at h ⊢
  rw [hsch, hop] at h
  exact h

/-- **A row D piece's copy reads the piece's opening at the deployed readers**: `openCopyR_of` at the assembly's side
`sideAtR`, from the side's schedule (`sideSchR_sideAtR`), the copy's range (`copyFitsR_rowD`) and its opened row and
siblings at the layout of record (`copyOpensR_layR`, which reads of the bound record only `final`, `DE`'s rounds and
`opened`), each part at its own witness (`atPiecesP_of`), then at the decision's side (`openCopy_sideAt`). -/
theorem openCopyW_deployed (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE w : Msg → ℝ) (b : Brings) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg) :
    OpenCopyW DeployedR.rdR key wE DeployedR.finalR w b OpenRows.elemR Claims.Recursion.Form.Dense untr cmt := by
  have hOp := OpenRows.copyOpensR_layR FlockDeploy.sha512 (DRL DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR)
    Claims.Recursion.Form.Dense untr b cmt (Rows.rowDPieces key) (fun _ => rfl) (fun _ h => h) (fun _ => rfl)
    (OpenLower.piecesInRangeR_rowDPieces key)
  have hO : OpenLower.CopyOpensR FlockDeploy.sha512 (DRL DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR)
      Claims.Recursion.Form.Dense untr b cmt (DeployedR.layR OpenRows.elemR)
      (_root_.RecDeploy.bdR DeployedR.rdR wE DeployedR.finalR w) (OpenLower.sideAtR DeployedR.rdR)
      (Rows.rowDPieces key) :=
    fun x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hst q hq ho ht os last hfin hl =>
      hOp x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hst q hq ho ht os last hfin hl
  have hC := OpenLower.openCopyR_of (H := FlockDeploy.sha512)
    (D := DRL DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR) (shape := Claims.Recursion.Form.Dense)
    (untr := untr) (b := b) (cmt := cmt) (L := DeployedR.layR OpenRows.elemR)
    (Bd := _root_.RecDeploy.bdR DeployedR.rdR wE DeployedR.finalR w) (side := OpenLower.sideAtR DeployedR.rdR)
    (Rows.rowDPieces key) (fun _ => rfl) (fun _ h => h)
    (OpenLower.sideSchR_sideAtR (H := FlockDeploy.sha512)
      (D := DRL DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR) (shape := Claims.Recursion.Form.Dense)
      (untr := untr) (b := b) (cmt := cmt) (L := DeployedR.layR OpenRows.elemR)
      (Bd := _root_.RecDeploy.bdR DeployedR.rdR wE DeployedR.finalR w) DeployedR.rdR (Rows.rowDPieces key))
    (OpenLower.copyFitsR_rowD (H := FlockDeploy.sha512)
      (D := DRL DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR) (shape := Claims.Recursion.Form.Dense)
      (untr := untr) (b := b) (cmt := cmt) (L := DeployedR.layR OpenRows.elemR) key (fun _ => rfl) (fun _ h => h))
    hO
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hw hst q hq ho ht os last hfin hl Dc
  obtain ⟨-, -, -, hcp⟩ := OpenLower.atPiecesP_of (H := FlockDeploy.sha512)
    (D := DRL DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR) (shape := Claims.Recursion.Form.Dense)
    (untr := untr) (b := b) (cmt := cmt) (L := DeployedR.layR OpenRows.elemR) hC
    x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hw hst q hq ho ht
  exact openCopy_sideAt Dc hq (hcp os last hfin hl)

end RecDeploy.ResLower
