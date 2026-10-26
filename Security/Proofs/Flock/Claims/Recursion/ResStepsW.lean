import Proofs.Flock.Claims.Recursion.ResChainW
import Proofs.Flock.Claims.Recursion.VerdictRowEAccR
import Proofs.Flock.Claims.Recursion.CapLowerPin
import Proofs.Flock.Claims.Recursion.ResPiecesD

/-!
# The steps half of P3 (`StepsW`): the outer witnesses' chain carries row D's running sums

`chainW x ws` reads row `p` as its owner part holds it (`accCanon`, `accSegs`). A row D piece's level copy reads the
chain's row at the piece's position `pin` on its `acc` columns and writes the next on its `acc_out` columns; V\* holds
the copy's part to `acc` at row `pin` (`RecVStar.accReads`) and registers `acc_out` at `pin + 1`. So the piece steps
`chainW` (`accHolds_of_copy`) wherever both of the copy's rows are their rows' owners':

* `rows_eq_of_calls'`, `held_rows_eq`, `out_rows_eq`: `CapLower.rows_eq_of_calls`, `held_rows_eq_layR`,
  `out_rows_eq_layR` with no premise on the two ports' row prefixes: both ports are their circuits', and every
  `sha512/row/v1` prefix is 128 bytes, so the prefixes cancel by length.
* `chainW_owner`: `chainW` at a row with an owner reads the owner's holding.
* `accHolds_chainW`: a piece holds its running sum at `chainW`, from its copy's facts (`accHolds_of_copy`'s) and one
  commit string between each of the copy's two rows and its row's owner's holding.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-! ## Two slots with one commit string hold one row, at ports of any word counts -/

/-- A port's row prefix is 128 bytes (`HmRow.rowPrefix`), at a port the circuit has. -/
theorem rowPfx_length {c : Circuit} {nm : String} {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) :
    (rowPfx c q).length = 128 := by
  obtain ⟨hlt, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  simp only [rowPfx, Array.getElem?_eq_getElem hlt, Option.map_some, Option.getD_some, Array.length_toList]
  exact FlockSoundness.Discharge.Hm.size_rowPrefix _

/-- **Two hm96 slots with one commit string hold one row** (`CapLower.rows_eq_of_calls`), where the two row prefixes
have one length rather than being one prefix. -/
theorem rows_eq_of_calls' {l : List (Msg × Msg)} (hT : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {c c' : Circuit} {q q' n n' : ℕ} {w w' : Claims.Flock.Wit}
    {row row' : List HmNets.Lin}
    (hs : (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n),
      readForms w ((hmCommit c q n).drop 512)) ∈ l)
    (hr : (rowPfx c q ++ readForms w row, readForms w (hmDigest c q n)) ∈ l)
    (hs' : (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n'),
      readForms w' ((hmCommit c' q' n').drop 512)) ∈ l)
    (hr' : (rowPfx c' q' ++ readForms w' row', readForms w' (hmDigest c' q' n')) ∈ l)
    (hlen : (rowPfx c q).length = (rowPfx c' q').length)
    (hcm : readForms w (hmCommit c q n) = readForms w' (hmCommit c' q' n')) :
    readForms w row = readForms w' row' := by
  have hsp : FlockSoundness.Refine.K512.saltPrefix = Hm96.Default512.saltPrefix.data.toList := by
    unfold FlockSoundness.Refine.K512; rfl
  have hs1 : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n)) =
      readForms w ((hmCommit c q n).drop 512) := hT _ hs
  have hr1 : FlockDeploy.sha512 (rowPfx c q ++ readForms w row) = readForms w (hmDigest c q n) := hT _ hr
  have hs1' : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n')) =
      readForms w' ((hmCommit c' q' n').drop 512) := hT _ hs'
  have hr1' : FlockDeploy.sha512 (rowPfx c' q' ++ readForms w' row') = readForms w' (hmDigest c' q' n') := hT _ hr'
  have e := CapLower.commit_of_true (c := c) (q := q) (n := n) (w := w) (R := rowPfx c q ++ readForms w row) hs1 hr1
  have e' := CapLower.commit_of_true (c := c') (q := q') (n := n') (w := w') (R := rowPfx c' q' ++ readForms w' row')
    hs1' hr1'
  have hy : FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst := by
    rw [hsp]; exact List.mem_map.2 ⟨_, hs, rfl⟩
  have hy' : FlockSoundness.Refine.K512.saltPrefix ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst := by
    rw [hsp]; exact List.mem_map.2 ⟨_, hs', rfl⟩
  have hr0 : rowPfx c q ++ readForms w row ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr, rfl⟩
  have hr0' : rowPfx c' q' ++ readForms w' row' ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr', rfl⟩
  have hb := CapLower.commit_bind (r := rowPfx c q ++ readForms w row) (r' := rowPfx c' q' ++ readForms w' row')
    (y := readForms w (hmSalt c q n)) (y' := readForms w' (hmSalt c' q' n')) hT hnc hr0 hr0' hy hy'
    (e.symm.trans (hcm.trans e'))
  exact (List.append_inj hb hlen).2

section AtLayR

variable (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

/-- **Two holdings of the chain with one commit string hold one row** (`held_rows_eq_layR`, at ports of any word
counts). -/
theorem held_rows_eq {x : Msg} {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    {s s' : ℕ} {c c' : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    (hc' : r.w.parts[s']?.bind partCircuit = some c') {w w' : Claims.Flock.Wit} (hw : ws[s]? = some w)
    (hw' : ws[s']? = some w') {h h' : String × ℕ} (hh : h ∈ accHeld r s) (hh' : h' ∈ accHeld r s') {q q' : ℕ}
    (hq : c.ports.findIdx? (·.name == h.1) = some q) (hq' : c'.ports.findIdx? (·.name == h'.1) = some q')
    (hcm : readForms w (hmCommit c q h.2) = readForms w' (hmCommit c' q' h'.2)) :
    readSegs w (accSegs c h) = readSegs w' (accSegs c' h') := by
  obtain ⟨h1, h2⟩ := heldCalls_layR elem hr hc hw hh hq
  obtain ⟨h3, h4⟩ := heldCalls_layR elem hr hc' hw' hh' hq'
  rw [readSegs_accSegs] at h2 ⊢
  rw [readSegs_accSegs] at h4 ⊢
  exact rows_eq_of_calls' hT hnc h1 h2 h3 h4 ((rowPfx_length hq).trans (rowPfx_length hq').symm) hcm

/-- **A unit's `acc_out` row is the holding of the chain with its commit string** (`out_rows_eq_layR`, at ports of
any word counts). -/
theorem out_rows_eq {x : Msg} {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    {s s' : ℕ} {c c' : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    (hc' : r.w.parts[s']?.bind partCircuit = some c') {w w' : Claims.Flock.Wit} (hw : ws[s]? = some w)
    (hw' : ws[s']? = some w') {u : ℕ} (hh : ("acc", u) ∈ accHeld r s) {h' : String × ℕ} (hh' : h' ∈ accHeld r s')
    {qo q' : ℕ} (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo)
    (hq' : c'.ports.findIdx? (·.name == h'.1) = some q')
    (hcm : readForms w (hmCommit c qo u) = readForms w' (hmCommit c' q' h'.2)) :
    readForms w (accOutForms c qo u) = readSegs w' (accSegs c' h') := by
  obtain ⟨h1, h2⟩ := outCalls_layR elem hr hc hw hh hqo
  obtain ⟨h3, h4⟩ := heldCalls_layR elem hr hc' hw' hh' hq'
  rw [readSegs_accSegs] at h4 ⊢
  exact rows_eq_of_calls' hT hnc h1 h2 h3 h4 ((rowPfx_length hqo).trans (rowPfx_length hq').symm) hcm

end AtLayR

/-! ## `chainW` at a row's owner -/

theorem rowW_owner {x : Msg} {ws : List Claims.Flock.Wit} {r : XR} (hr : readX x = some r) {p o : ℕ}
    {c : Circuit} {h : String × ℕ} (hcan : accCanon r p = some (o, c, h)) {w : Claims.Flock.Wit}
    (hw : ws[o]? = some w) : rowW x ws p = some (readSegs w (accSegs c h)) := by
  simp only [rowW, hr, hcan, hw, Option.bind_some, Option.map_some]

/-- **`chainW` reads row `p`'s owner's holding**, where the row has an owner with a witness. -/
theorem chainW_owner {x : Msg} {ws : List Claims.Flock.Wit} {r : XR} (hr : readX x = some r) {p o : ℕ}
    {c : Circuit} {h : String × ℕ} (hcan : accCanon r p = some (o, c, h)) {w : Claims.Flock.Wit}
    (hw : ws[o]? = some w) (t : ℕ) :
    chainW x ws t p = #[elemsOf (readSegs w (accSegs c h)) 2 0, elemsOf (readSegs w (accSegs c h)) 2 1] := by
  unfold chainW
  rw [rowW_owner hr hcan hw]

/-- An owner's first holding is a holding of the chain. -/
theorem accHeld_of_canon {r : XR} {p o : ℕ} {c : Circuit} {h : String × ℕ} (hcan : accCanon r p = some (o, c, h)) :
    r.w.parts[o]?.bind partCircuit = some c ∧ h ∈ accHoldings r o p := by
  obtain ⟨-, hc, hh⟩ := accCanon_spec hcan
  exact ⟨hc, List.mem_of_mem_head? (Option.mem_def.2 hh)⟩

/-! ## A piece steps `chainW` -/

/-- **A piece holds its running sum at `chainW`**: `accHolds_of_copy` at `chainW x ws`, where part `s`'s copy (`v`)
computes the piece's level (`LevelOuts`, `hLA`, `hrow`, `hcf`), its `acc` and `acc_out` columns hold its witness's
rows there (`hbi'`, `hbo'`), it holds `acc` at the piece's position `pin` at its unit `u`, and the copy's `acc` row
and `acc_out` row carry the commit strings of rows `pin`'s and `pin + 1`'s owners' holdings (`hcm₀`, `hcm₁`), at
`layR`'s true, collision-free calls. -/
theorem accHolds_chainW (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {x : Msg} {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (coefAt : Msg → List Coin → OpenPiece → Array _root_.Flock.F128)
    (q : OpenPiece) (xI : Msg) (ms : List Msg) (cs : List Coin) {v : ℕ → Bool} {LANES HH : ℕ}
    (hO : _root_.Flock.Readable.LevelOuts v LANES HH)
    (hLA : LANES = 2 ^ ((sideAt xI ms q).sch.levels[q.level]!).lanes)
    (hrow : _root_.Flock.Readable.openRow v LANES =
      (FlockSoundness.Refine.openAt (sideAt xI ms q).lig ((sideAt xI ms q).sch.levels.size - 1) q.level).1[q.query]!)
    (hcf : _root_.Flock.Readable.openCoefs v LANES HH = coefAt xI cs q)
    {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {wt : Claims.Flock.Wit} (hw : ws[s]? = some wt) {u pin : ℕ} (hh : ("acc", u) ∈ accHoldings r s pin)
    (hpin : pin = Rows.chainPos (sideAt xI ms q).sch q.rep q.level q.query)
    {qa qo : ℕ} (hqa : c.ports.findIdx? (·.name == "acc") = some qa)
    (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo)
    {bi bo : ByteArray} (hbi : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) bi)
    (hszi : 32 ≤ bi.size) (hbo : FlockVBridge.ColsHold v (FlockVBridge.Family.outCol LANES HH + 1024) bo)
    (hszo : 32 ≤ bo.size)
    (hbi' : bi.data.toList = readSegs wt (accSegs c ("acc", u)))
    (hbo' : bo.data.toList = readForms wt (accOutForms c qo u))
    {o₀ : ℕ} {c₀ : Circuit} {h₀ : String × ℕ} (hcan₀ : accCanon r pin = some (o₀, c₀, h₀))
    {w₀ : Claims.Flock.Wit} (hw₀ : ws[o₀]? = some w₀) {q₀ : ℕ}
    (hq₀ : c₀.ports.findIdx? (·.name == h₀.1) = some q₀)
    (hcm₀ : readForms wt (hmCommit c qa u) = readForms w₀ (hmCommit c₀ q₀ h₀.2))
    {o₁ : ℕ} {c₁ : Circuit} {h₁ : String × ℕ} (hcan₁ : accCanon r (pin + 1) = some (o₁, c₁, h₁))
    {w₁ : Claims.Flock.Wit} (hw₁ : ws[o₁]? = some w₁) {q₁ : ℕ}
    (hq₁ : c₁.ports.findIdx? (·.name == h₁.1) = some q₁)
    (hcm₁ : readForms wt (hmCommit c qo u) = readForms w₁ (hmCommit c₁ q₁ h₁.2)) :
    Rows.accHolds sideAt coefAt (fun _ _ => chainW x ws) q xI ms cs := by
  obtain ⟨hc₀, hm₀⟩ := accHeld_of_canon hcan₀
  obtain ⟨hc₁, hm₁⟩ := accHeld_of_canon hcan₁
  have e₀ := held_rows_eq elem hT hnc hr hc hc₀ hw hw₀ (mem_accHeld hh) (mem_accHeld hm₀) hqa hq₀ hcm₀
  have e₁ := out_rows_eq elem hT hnc hr hc hc₁ hw hw₁ (mem_accHeld hh) (mem_accHeld hm₁) hqo hq₁ hcm₁
  subst hpin
  refine accHolds_of_copy sideAt coefAt (fun _ _ => chainW x ws) q xI ms cs hO hLA hrow hcf hbi hszi hbo hszo ?_ ?_
  · refine (chainW_owner hr hcan₀ hw₀ q.table).trans ?_
    rw [hbi', e₀]
  · refine (chainW_owner hr hcan₁ hw₁ q.table).trans ?_
    rw [hbo', e₁]

/-! ## `StepsW` from four facts at each row D piece's copy -/

/-- **`StepsW`'s reading of the last message and decision** at `x`, `c`, `ws`: `F` at the opened messages, wherever
the last message reads as `os` and `last` and a decision of the opened messages exists. -/
abbrev AtDecW (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) (F : List Msg → Prop) : Prop :=
  ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
      (((DeployedR.layR elem).msgs x ws).getD
        ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
        []) = some (os, last) →
    os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
  ∀ _ : Rows.DecAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
      ((_root_.RecDeploy.bdR rd wE final w).opened os last)
      ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
        (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c),
    F ((_root_.RecDeploy.bdR rd wE final w).opened os last)

/-- **A row D piece's copy's coefficients are `coefRead`'s** at the coins `StepsW` reads, at its owner's witness
(`AtPiecesP` at row D's pieces). -/
def CoefW (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) shape untr b cmt (DeployedR.layR elem)
    (Rows.rowDPieces key) fun x c _ LANES HH _ _ wt p ht q =>
      _root_.Flock.Readable.openCoefs
          (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q) LANES HH =
        Rows.coefRead (Claims.Recursion.xI (DRL rd wE final w b elem) x)
          ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
            (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c) q

/-- **A row D piece's copy reads `chainW`'s rows at its position and the next** on its `acc` and `acc_out` columns,
at its owner's witness, at every reading of the last message and decision (`AtDecW`): bytes `bi` on the copy's `acc`
columns and `bo` on its `acc_out` columns whose first two elements are `chainW`'s rows at the piece's position
(`Rows.chainPos` at the decided side's schedule) and the next. -/
def RowsW (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) shape untr b cmt (DeployedR.layR elem)
    (Rows.rowDPieces key) fun x c ws LANES HH _ _ wt p ht q =>
      AtDecW rd wE final w b elem x c ws fun ms =>
        ∃ bi bo : ByteArray,
          FlockVBridge.ColsHold (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q)
            (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) bi ∧ 32 ≤ bi.size ∧
          FlockVBridge.ColsHold (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q)
            (FlockVBridge.Family.outCol LANES HH + 1024) bo ∧ 32 ≤ bo.size ∧
          chainW x ws q.table (Rows.chainPos (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x) ms q).sch
              q.rep q.level q.query) = #[elemsOf bi.data.toList 2 0, elemsOf bi.data.toList 2 1] ∧
          chainW x ws q.table (Rows.chainPos (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x) ms q).sch
              q.rep q.level q.query + 1) = #[elemsOf bo.data.toList 2 0, elemsOf bo.data.toList 2 1]

section Assembly

variable {rd : Msg → Option FinalP} {wE : Msg → ℝ} {final : Msg → Option (List (List (Msg × Salt)) × Msg)}
  {w : Msg → ℝ} {b : Brings} {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
  {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}

/-- Under `JointP`, every part has a staged statement at its own witness (`OpenLevW.stagedAt_of_jointP`'s proof). -/
private theorem stagedAt_jointP {x : Msg} {c : ℕ → Coin} {ws : List Claims.Flock.Wit}
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

end Assembly

/-- **`StepsW` from four facts at each row D piece's copy**, at the dense form: the copy is a level table's outputs
(`hL`, `OpenLevW.LevelOutsW`'s), reads the piece's opening at the decided side (`hO`, `OpenLevW.OpenCopyW`'s), its
coefficients are `coefRead`'s (`hC`), and its `acc` and `acc_out` columns read `chainW`'s rows at its position and the
next (`hR`). Each piece has its level's part as owner (`openOwned_D`), staged at its own witness under `JointP`, and
`accHolds_of_copy` steps `chainW` there. -/
theorem stepsW_of (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg)
    (hL : OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) Claims.Recursion.Form.Dense untr b cmt
      (DeployedR.layR elem) (Rows.rowDPieces key) fun x _ _ LANES HH _ _ wt p ht q =>
        _root_.Flock.Readable.LevelOuts
          (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q) LANES HH)
    (hO : OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) Claims.Recursion.Form.Dense untr b cmt
      (DeployedR.layR elem) (Rows.rowDPieces key) fun x c ws LANES HH _ _ wt p ht q =>
        AtDecW rd wE final w b elem x c ws fun ms =>
          OpenCopy (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x) ms q) q LANES HH
            (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q))
    (hC : CoefW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt)
    (hR : RowsW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt) :
    StepsW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt := by
  intro x st hx hf c ws hJ os last hfin hos hD q hq
  obtain ⟨s, hs, ho⟩ := DecompBR.openOwned_D key x ⟨st, hx, hf⟩ q hq
  obtain ⟨-, r, ss, hr, hss, LANES, HH, n, hsv⟩ := OpenLower.ownQR_some ho
  have hsw : s < ws.length := lt_of_lt_of_eq hs hJ.1.symm
  have hw : ws[s]? = some ws[s] := List.getElem?_eq_getElem hsw
  obtain ⟨xO, p, hst⟩ := stagedAt_jointP hJ hsw hr hss hsv
  have ht : 0 < p.sts.size := hst.2.1
  have hLO := hL x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO ws[s] p hw hst q hq ho ht
  have hOC := hO x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO ws[s] p hw hst q hq ho ht os last hfin hos
    hD.some
  have hCf := hC x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO ws[s] p hw hst q hq ho ht
  obtain ⟨bi, bo, hbi, hszi, hbo, hszo, hin, hout⟩ :=
    hR x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO ws[s] p hw hst q hq ho ht os last hfin hos hD.some
  exact accHolds_of_copy (Rows.sideAt rd) Rows.coefRead (fun _ _ => chainW x ws) q _ _ _ hLO hOC.1 hOC.2.2.1 hCf
    hbi hszi hbo hszo hin hout

end RecDeploy.ResLower
