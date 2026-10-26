import Proofs.Flock.Claims.Recursion.DecompBDeployed

/-!
# The rest's lowering (`RestLowR`, `SaltRowRest`) at the layout of record, and its joint form

`recKSDecompL_of_leaves` takes two leaves on the checks that are no pieces: `SaltRowRest` (the salts and rows of a
round before the last that no part holds) and `RestLowR` (`hlowR`: where part `s`'s circuit accepts at messages `ms`,
the salts and rows of the rounds `s` holds first, and the record's reading where it holds the last round first).

**The record's reading is never a part's** at `DeployedR.layR elem`: its `msgAt` is `layoutBTops`', so no part holds
the last round (`firstHolder_last_layR`), and `RestLowR`'s second conjunct holds outright (`restLowR_read`).

**`RestLowR`'s first conjunct is asked where it can't hold.** A salt or row check of round `j` reads the opening
`final` takes off the last message, `ms.getD (k - 1) []`. `PartAccepts` ties that message to part `s` only through
`s`'s own pieces (each a window `((ms.getD i []).drop o).take m.length` at an offset the statement fixes): its other
bytes, its length and so the trailer `finalR` reads off its end (`finalR_trailer`) are free in such an `ms`, and with
them the cut, so the opening it reads for round `j` is any bytes. `CmtAgrees` leaves the last round out. This is
`OpenLower`'s obstruction for `hcopy`, at `hlowR`. `DecompB` reads the parts' checks only at the parts' own messages
`L.msgs x ws`, every part accepting there (`decompB_of_joint`), so the lowering is asked there:

* `decompB_lastSplitP_sharesJR`: `decompB_lastSplitP_sharesJ` with `hlowR` asked only at `Joint`, as `hlowP` is.
* `RestLowJ`: `RestLowR` at the parts' own messages, every part accepting there.
* `SaltRowsJ`: every round before the last, each commit string's salt and row, at the readers' messages of witnesses
  one a part whose calls are true. It gives `SaltRowRest` (`saltRowRest_of`) and, at the layout of record, `RestLowJ`
  (`restLowJ_of`), the reading's half being `restLowR_read`'s.
* `recKSDecompL_of_leavesJ` is `recKSDecompL_of_leaves` with `RestLowJ` for `RestLowR`, and
  `recKSDecompD_of_leavesJ` its instance at `DeployedR.layR elem`, where `SaltRowsJ` alone stands for both leaves.
-/

namespace RecDeploy.RestLow

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR)
open Claims.Recursion.VStarB
open RecDeploy.VPieces (VPiece vrows vholds noB vown)
open RecDeploy.OpenLower (ownQR Joint)
open RecDeploy.RestRead (DenseX kI)
open RecDeploy.DecompBR

noncomputable section

/-! ## The composition with the rest's share at the parts' own messages -/

section Compose

variable (H : Msg → Msg) (rd : Msg → Option FinalP) {Q : Type} (pieces : Msg → List Q)
  (pholds : Q → Msg → List Msg → List Coin → Prop)
  (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ReadOkR rd x ms cs →
    (∀ b ∈ VerdictP.bases, FlockDeploy.holdsR H rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
      FlockDeploy.holdsR H rd .verdict x ms cs)
  (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
  (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
  (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)

/-- **The composition with both shares at the parts' own messages**: `decompB_lastSplitP_sharesJ` with the rest's
share `hlowR` asked, like the pieces' share, only where every part accepts at the parts' own messages (`Joint`). -/
theorem decompB_lastSplitP_sharesJR (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Q → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ (lastSplitP H rd pieces pholds hpieces).checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s,
        ownP ownB ownQ x r = some s → s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
          (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
            (lastSplitP H rd pieces pholds hpieces).toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ))) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
            (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
              (lastSplitP H rd pieces pholds hpieces).toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlowR : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
        Joint H (recBound rel' b L IV) L (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) x c ws →
        ∀ s < (recBound rel' b L IV).nS x,
          RestComputes H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
            (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
              (lastSplitP H rd pieces pholds hpieces).toLast)
            (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ))))
            s x c (L.msgs x ws))
    (hlowP : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
        Joint H (recBound rel' b L IV) L (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) x c ws →
        ∀ s < (recBound rel' b L IV).nS x,
          PiecesComputes (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w) (recBound rel' b L IV)
            pieces pholds ownQ s x c (L.msgs x ws)) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w).IVStar L
      shape (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) := by
  have hD := ivstar_of_parts H (recBound rel' b L IV) _ L shape
    (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast) _ _
    (ownedBy_deploy H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)) _ L shape
      (ownR_lt (lastSplitP H rd pieces pholds hpieces) (ownP ownB ownQ) _ L shape hR))
    (stmtFixes_deploy H rel (nextR rd) colsE wE FlockDeploy.joinR_head rowIn final w (nextR_coin rd H)
      (lastSplitP H rd pieces pholds hpieces).toLast (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)) rel' b L IV
      shape hrest)
  refine decompB_of_joint ?_ hD
  intro x st hx hf c ws hJ s hs
  exact partComputes_of_shares H rd pieces pholds hpieces rel colsE wE rowIn final w ownB ownQ _ L
    (hlowR x st hx hf c ws hJ s hs) (hlowP x st hx hf c ws hJ s hs)

end Compose

/-! ## The record's reading is never a part's at the layout of record -/

/-- **No part holds the last round at the layout of record**: `DeployedR.layR elem` holds what `layoutBTops` does
(`firstHolder_last`). -/
theorem firstHolder_last_layR (D : Claims.Recursion.Deploy Coin)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg} (hx : DenseX x) :
    firstHolder D (DeployedR.layR elem) x (kI x - 1) = none :=
  RestRead.firstHolder_last D (DeployedR.restR elem) hx

section Leaves

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)

/-- **`RestLowR`'s second conjunct at the layout of record**: wherever part `s` accepts, it holds the last round
first nowhere, so the record's reading is never its. -/
theorem restLowR_read (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (hacc : acceptsL rd b aux nOf hd rest wE final w (DeployedR.layR elem) s x c ms) :
    firstHolder (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem)) (DeployedR.layR elem) x (kI x - 1) =
        some s →
      ∀ os last, (bdR rd wE final w).final (ms.getD (kI x - 1) []) = some (os, last) → os.length + 1 = kI x →
        ReadOkR rd (xIOf x) ((bdR rd wE final w).opened os last) ((List.range (kI x)).map c) := by
  obtain ⟨st, hst, hf⟩ := hacc.1
  intro h
  rw [firstHolder_last_layR _ elem ⟨st, hst, hf⟩] at h
  cases h

/-- **Every commit string before the last round has its salt and row at the parts' own messages**: at a dense compiled
statement, round `j < k - 1`, commit string `i` the schedule counts, at the readers' messages of witnesses one a part
whose recorded calls are true, at any coins and any split of the element-level checks. -/
def SaltRowsJ : Prop :=
  ∀ x, DenseX x → ∀ j < kI x - 1, ∀ i < nAtSched (xIOf x) j, ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    ws.length = nS x → (∀ e ∈ L.allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
    ∀ {Q : Type} (S : ElemSplit FlockDeploy.sha512 (bdR rd wE final w) Q),
      (IVCheck.salt j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c (L.msgs x ws) ∧
      (IVCheck.row j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c (L.msgs x ws)

/-- **`RestLowR` at the parts' own messages** (`hlowR` as `decompB_lastSplitP_sharesJR` asks it): where every part
accepts at the parts' own messages, part `s`'s salts and rows of the rounds it holds first, and the record's reading
where it holds the last round first, there. -/
def RestLowJ : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
      Joint FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) L
        (acceptsL rd b aux nOf hd rest wE final w L) x c ws →
      ∀ s < nS x,
        (∀ j < kI x - 1, firstHolder (recL rd b aux nOf hd rest wE final w L) L x j = some s →
          ∀ i < nAtSched (xIOf x) j, ∀ {Q : Type} (S : ElemSplit FlockDeploy.sha512 (bdR rd wE final w) Q),
            (IVCheck.salt j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c (L.msgs x ws) ∧
            (IVCheck.row j i : IVCheck Q).Holds FlockDeploy.sha512 (bdR rd wE final w) S (xIOf x) c (L.msgs x ws)) ∧
        (firstHolder (recL rd b aux nOf hd rest wE final w L) L x (kI x - 1) = some s →
          ∀ os last, (bdR rd wE final w).final ((L.msgs x ws).getD (kI x - 1) []) = some (os, last) →
            os.length + 1 = kI x →
            ReadOkR rd (xIOf x) ((bdR rd wE final w).opened os last) ((List.range (kI x)).map c))

/-- **`SaltRowRest` from `SaltRowsJ`**: the rounds no part holds are among all of them. -/
theorem saltRowRest_of (h : SaltRowsJ rd wE final w L) : SaltRowRest rd b aux nOf hd rest wE final w L :=
  fun x hx j hj _ i hi c ws hl hT _ S => h x hx j hj i hi c ws hl hT S

/-- **`RestLowJ` at the layout of record from `SaltRowsJ`**: the salts and rows at every round, and the record's
reading never a part's (`firstHolder_last_layR`). -/
theorem restLowJ_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (h : SaltRowsJ rd wE final w (DeployedR.layR elem)) :
    RestLowJ rd b aux nOf hd rest wE final w (DeployedR.layR elem) := by
  intro x st hx hf c ws hJ s _
  have hX : DenseX x := ⟨st, hx, hf⟩
  refine ⟨fun j hj _ i hi _ S => h x hX j hj i hi c ws hJ.1 hJ.2.1 S, fun hs => ?_⟩
  rw [firstHolder_last_layR _ elem hX] at hs
  cases hs

end Leaves

/-! ## The assembly with the rest's share at the parts' own messages -/

section Assembly

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)
  (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
  (elemAt : Msg → List Msg → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
  (ownR : Msg → ResPiece → Option ℕ) (chainAt : Msg → List Msg → ℕ → ℕ → Array _root_.Flock.F128)

/-- **`RecKSDecompR` at layout `L`, from its leaves, the rest's share at the parts' own messages**:
`recKSDecompL_of_leaves` with `RestLowJ` for `RestLowR`. -/
theorem recKSDecompL_of_leavesJ
    (hRdep : Top.TargetRDep rd (PD resP key elemAt vvAt side posAt) (accR rd chainAt))
    (hDm : Top.TargetDMerkle rd (PD resP key elemAt vvAt side posAt) (accR rd chainAt))
    (hholds : RowEHolds rd resP key elemAt vvAt side posAt chainAt)
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon)
    (hB : BasesHold rd) (hF : FinalReadsL final L) (hLn : RoundLensL L) (hRd : ReadsAtL rd final L)
    (hSR : SaltRowRest rd b aux nOf hd rest wE final w L) (hOwn : PiecesOwned resP key ownR)
    (hlowR : RestLowJ rd b aux nOf hd rest wE final w L)
    (hres : ResLowJ rd b aux nOf hd rest wE final w L resP elemAt vvAt ownR)
    (hSch : OpenLower.SideSchR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key))
    (hO : OpenLower.CopyOpensR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key))
    (hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w)
      (Rows.rowDPieces key) posAt)
    (hTop : OpenLower.CopyTopR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key) posAt) :
    RecKSDecompL rd b aux nOf hd rest wE final w L := by
  have h := Top.targets_of_rowsJ S rd (rowsD resP key) (holdsD elemAt vvAt side posAt) chainAt key hRdep hDm hholds
  have hP : ∀ x, (recL rd b aux nOf hd rest wE final w L).parse x = parse x := fun _ => rfl
  have hS : ∀ f, Claims.Recursion.Form.Dense f → Claims.Recursion.Form.Dense f := fun _ h => h
  have hC := OpenLower.openCopyR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
    Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
    (Rows.rowDPieces key) hP hS hSch
    (OpenLower.copyFitsR_rowD FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L key hP hS) hO
  have hT := OpenLower.openTieR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
    Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
    (Rows.rowDPieces key) posAt
    (OpenLower.copyBitsR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key) posAt hP hS hSch
      (OpenLower.copyDirsR_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
        Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w)
        (Rows.rowDPieces key) posAt hDirs)) hTop
  refine decompB_lastSplitP_sharesJR FlockDeploy.sha512 rd (Top.pieces (rowsD resP key))
    (Top.pholds (holdsD elemAt vvAt side posAt))
    (Top.hpieces_of_targets rd (rowsD resP key) (holdsD elemAt vvAt side posAt) (accR rd chainAt) h)
    _root_.Flock.Readable.relation (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE
    rowInDeployed final w noB (ownD ownR) _root_.Flock.Readable.relation (bB b aux nOf hd rest) L
    (bdR rd wE final w).IVStar Claims.Recursion.Form.Dense (deployZkRec _root_.Flock.Readable.relation)
    (showsPart recRead (bB b aux nOf hd rest)) (cmtTr recRead) ?_ ?_ ?_ ?_
  · -- hR
    intro x st hx hf r hr s hs
    exact VPieces.ownP_lt ownR ownQR (fun _ _ => none) hs
  · -- hrest
    intro x st hx hf q hq hown hlast c ws hl hT'
    have hX : DenseX x := ⟨st, hx, hf⟩
    rcases mem_ivChecks _ _ _ hq with rfl | ⟨j, hj, rfl⟩ | ⟨j, hj, i, hi, rfl | rfl⟩ | ⟨q', hq', rfl⟩
    · obtain ⟨os, last, hfin, hos, hcnt⟩ := hF x hX ws hl
      exact ⟨RestRead.kI_pos x, os, last, hfin, hos, hcnt⟩
    · have hj' : j < kI x - 1 := hj
      obtain ⟨os, last, hfin, hos, hcnt⟩ := hF x hX ws hl
      intro os' last' hf'
      have hf'' : final ((L.msgs x ws).getD (kI x - 1) []) = some (os', last') := hf'
      rw [hfin] at hf''
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf'')
      rw [hLn x hX ws hl j hj']
      exact congrArg (128 * ·) (hcnt j (by omega)).symm
    · exact (hSR x hX j hj hown i hi c ws hl hT' _).1
    · exact (hSR x hX j hj hown i hi c ws hl hT' _).2
    · obtain ⟨r', rfl⟩ := hlast q' rfl
      have hr' := mem_last_checks hq'
      rcases List.mem_cons.1 hr' with rfl | hr'
      · intro os last hfin hos
        exact hRd x hX c ws hl hT' os last hfin hos
      · obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hr'
        rcases List.mem_append.1 hv with hv | hv
        · obtain ⟨b', hb', rfl⟩ := List.mem_map.1 hv
          exact restBase_holds rd hB _ _ _ _ _ wE _ rowInDeployed final w _ _ L _ hx hb' c _
        · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hv
          exfalso
          obtain ⟨r, hr, -, -⟩ := parse_read hx
          obtain ⟨s, hs⟩ := VPieces.vown_some ownR ownQR (fun _ _ => none) resP (Rows.rowDPieces key)
            (fun _ => []) (VPieces.nS_pos hr) (hOwn x hX).1 (hOwn x hX).2 (fun q hq => by simp at hq) p hp
          have hown' : vown ownR ownQR (fun _ _ => none) x p = none := hown
          rw [hs] at hown'
          cases hown'
  · -- hlowR
    intro x st hx hf c ws hJ s hs q hq hown hnp
    obtain ⟨hSRl, hRdl⟩ := hlowR x st hx hf c ws hJ s hs
    rcases mem_ivChecks _ _ _ hq with rfl | ⟨j, hj, rfl⟩ | ⟨j, hj, i, hi, rfl | rfl⟩ | ⟨q', hq', rfl⟩
    · cases (hown : (none : Option ℕ) = some s)
    · cases (hown : (none : Option ℕ) = some s)
    · exact (hSRl j hj hown i hi _).1
    · exact (hSRl j hj hown i hi _).2
    · cases q' with
      | setup => cases (hown : (none : Option ℕ) = some s)
      | tag _ => cases (hown : (none : Option ℕ) = some s)
      | last r' =>
        cases r' with
        | read =>
          intro os last hfin hos
          exact hRdl hown os last hfin hos
        | verdict v =>
          cases v with
          | base _ => cases (hown : (none : Option ℕ) = some s)
          | piece p => exact (hnp p rfl).elim
  · -- hlowP
    intro x st hx hf c ws hJ s hs
    refine VPieces.piecesComputes_vown ownR ownQR (fun _ _ => none) (bdR rd wE final w)
      (recL rd b aux nOf hd rest wE final w L) resP (Rows.rowDPieces key) (fun _ => []) (resHolds elemAt vvAt)
      (openHolds side posAt) (fun _ _ _ _ => True) ⟨x, st, hx, hf, rfl⟩ (hres x st hx hf c ws hJ s hs) ?_ ?_
    · exact OpenLower.openPieces_deployed FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
        Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
        (Rows.rowDPieces key) posAt hC hT hx hf hJ hs
    · intro q hq
      simp at hq

end Assembly

section AtNext

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
  (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
  (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
  (elemAt : Msg → List Msg → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (posAt : Msg → List Coin → OpenPiece → ℕ)
  (ownR : Msg → ResPiece → Option ℕ) (chainAt : Msg → List Msg → ℕ → ℕ → Array _root_.Flock.F128)

/-- **`RecKSDecompR`'s premise at the layout of record, the rest's two leaves as one**: `recKSDecompD_of_leaves`
with `SaltRowRest` and `RestLowR` replaced by `SaltRowsJ` (`saltRowRest_of`, `restLowJ_of`), through
`recKSDecompL_of_leavesJ`. -/
theorem recKSDecompD_of_leavesJ
    (hRdep : Top.TargetRDep rd (PD resP key elemAt vvAt (OpenLower.sideAtR rd) posAt) (accR rd chainAt))
    (hDm : Top.TargetDMerkle rd (PD resP key elemAt vvAt (OpenLower.sideAtR rd) posAt) (accR rd chainAt))
    (hholds : RowEHolds rd resP key elemAt vvAt (OpenLower.sideAtR rd) posAt chainAt)
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon)
    (hB : BasesHold rd) (hF : RestRead.FinalReads final (DeployedR.restR elem))
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hSRJ : SaltRowsJ rd wE final w (DeployedR.layR elem)) (hOwn : PiecesOwned resP key ownR)
    (hres : ResLowJ rd b aux nOf hd rest wE final w (DeployedR.layR elem) resP elemAt vvAt ownR)
    (hO : OpenLower.CopyOpensR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (OpenLower.sideAtR rd) (Rows.rowDPieces key))
    (hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (Rows.rowDPieces key) posAt)
    (hOut : CapLower.OutBindR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (Rows.rowDPieces key) posAt)
    (hCap : CapLower.SideCapR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key) posAt) :
    RecKSDecompL rd b aux nOf hd rest wE final w (DeployedR.layR elem) :=
  recKSDecompL_of_leavesJ rd b aux nOf hd rest wE final w (DeployedR.layR elem) resP key elemAt vvAt
    (OpenLower.sideAtR rd) posAt ownR chainAt hRdep hDm hholds S hB hF (RestRead.roundLens (DeployedR.restR elem))
    (fun x hx c ws hl hT os last hf hos =>
      hRd x hx c ws hl (fun e he => hT e (allCalls_layR elem x ws he)) os last hf hos)
    (saltRowRest_of rd b aux nOf hd rest wE final w _ hSRJ) hOwn
    (restLowJ_of rd b aux nOf hd rest wE final w elem hSRJ) hres
    (OpenLower.sideSchR_sideAtR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key))
    hO hDirs
    (CapLower.capTop_of FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key) posAt (fun _ => rfl) (fun _ h => h) hOut hCap)

end AtNext

end

end RecDeploy.RestLow
