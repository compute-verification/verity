module

public import Proofs.Flock.Claims.Recursion.CapLowerPin
public import Proofs.Flock.Claims.Recursion.ResChainW
public import Proofs.Flock.Claims.Recursion.VerdictRowEAccR
public import Proofs.Flock.Claims.Recursion.VerdictRowEBytes
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
import Mathlib.Data.String.Basic
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Claims.Recursion.RowDAccept
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Soundness.Discharge.Hidden.Delta

/-! The steps half of P3 (`StepsW`): the outer witnesses' chain carries row D's running sums. -/

@[expose] public section

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

theorem rowPfx_length {c : Circuit} {nm : String} {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) :
    (rowPfx c q).length = 128 := by
  obtain ⟨hlt, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  simp only [rowPfx, Array.getElem?_eq_getElem hlt, Option.map_some, Option.getD_some, Array.length_toList]
  exact FlockSoundness.Discharge.Hm.size_rowPrefix _

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
      readForms w ((hmCommit c q n).drop 512) := by
    have h := hT _ hs; exact h
  have hr1 : FlockDeploy.sha512 (rowPfx c q ++ readForms w row) = readForms w (hmDigest c q n) := by
    have h := hT _ hr; exact h
  have hs1' : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n')) =
      readForms w' ((hmCommit c' q' n').drop 512) := by
    have h := hT _ hs'; exact h
  have hr1' : FlockDeploy.sha512 (rowPfx c' q' ++ readForms w' row') = readForms w' (hmDigest c' q' n') := by
    have h := hT _ hr'; exact h
  have e := CapLower.commit_of_true (c := c) (q := q) (n := n) (w := w) (R := rowPfx c q ++ readForms w row) hs1 hr1
  have e' := CapLower.commit_of_true (c := c') (q := q') (n := n') (w := w') (R := rowPfx c' q' ++ readForms w' row')
    hs1' hr1'
  have hy0 : Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst :=
    List.mem_map.2 ⟨_, hs, rfl⟩
  have hy0' : Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst :=
    List.mem_map.2 ⟨_, hs', rfl⟩
  have hy : FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0
  have hy' : FlockSoundness.Refine.K512.saltPrefix ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0'
  have hr0 : rowPfx c q ++ readForms w row ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr, rfl⟩
  have hr0' : rowPfx c' q' ++ readForms w' row' ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr', rfl⟩
  have hb := CapLower.commit_bind (r := rowPfx c q ++ readForms w row) (r' := rowPfx c' q' ++ readForms w' row')
    (y := readForms w (hmSalt c q n)) (y' := readForms w' (hmSalt c' q' n')) hT hnc hr0 hr0' hy hy'
    (e.symm.trans (hcm.trans e'))
  exact (List.append_inj hb hlen).2

section AtLayR

variable (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

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

theorem rowW_owner {x : Msg} {ws : List Claims.Flock.Wit} {r : XR} (hr : readX x = some r) {p o : ℕ}
    {c : Circuit} {h : String × ℕ} (hcan : accCanon r p = some (o, c, h)) {w : Claims.Flock.Wit}
    (hw : ws[o]? = some w) : rowW x ws p = some (readSegs w (accSegs c h)) := by
  simp only [rowW, hr, hcan, hw, Option.bind_some, Option.map_some]

theorem chainW_owner {x : Msg} {ws : List Claims.Flock.Wit} {r : XR} (hr : readX x = some r) {p o : ℕ}
    {c : Circuit} {h : String × ℕ} (hcan : accCanon r p = some (o, c, h)) {w : Claims.Flock.Wit}
    (hw : ws[o]? = some w) (t : ℕ) :
    chainW x ws t p = #[elemsOf (readSegs w (accSegs c h)) 2 0, elemsOf (readSegs w (accSegs c h)) 2 1] := by
  unfold chainW
  rw [rowW_owner hr hcan hw]

theorem accHeld_of_canon {r : XR} {p o : ℕ} {c : Circuit} {h : String × ℕ} (hcan : accCanon r p = some (o, c, h)) :
    r.w.parts[o]?.bind partCircuit = some c ∧ h ∈ accHoldings r o p := by
  obtain ⟨-, hc, hh⟩ := accCanon_spec hcan
  exact ⟨hc, List.mem_of_mem_head? (Option.mem_def.2 hh)⟩

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

def RowsW (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (Φ : Msg → (ℕ → Coin) → List Claims.Flock.Wit → Prop) : Prop :=
  OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) shape untr b cmt (DeployedR.layR elem)
    (Rows.rowDPieces key) fun x c ws LANES HH _ _ wt p ht q => Φ x c ws →
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

def StepsWG (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (Φ : Msg → (ℕ → Coin) → List Claims.Flock.Wit → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws → Φ x c ws →
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (((DeployedR.layR elem).msgs x ws).getD
          ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
      Nonempty (Rows.DecAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c)) →
      Top.ChainHolds rd key (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c) (chainW x ws)

def XsW (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (A : ℕ → Msg → (ℕ → Coin) → Msg → Claims.Flock.Wit → List Msg → Prop)
    (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) : Prop :=
  ∃ xs : List Msg, (∀ s' (hs' : s' < ws.length), A s' x c (xs.getD s' []) ws[s'] ((DeployedR.layR elem).msgs x ws)) ∧
    (Claims.findColl ((DeployedR.layR elem).allCalls x ws ++
      Claims.Recursion.setupCalls FlockDeploy.sha512 (DRL rd wE final w b elem) xs)).isSome = false ∧
    XsStaged b.st x xs ws.length

theorem stepsW_of_G {rd : Msg → Option FinalP} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {wE : Msg → ℝ}
    {final : Msg → Option (List (List (Msg × Salt)) × Msg)} {w : Msg → ℝ} {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}
    (h : StepsWG rd key wE final w b elem shape untr cmt fun _ _ _ => True) :
    StepsW rd key wE final w b elem shape untr cmt :=
  fun x st hx hf c ws hJ => h x st hx hf c ws hJ trivial

section Assembly

variable {rd : Msg → Option FinalP} {wE : Msg → ℝ} {final : Msg → Option (List (List (Msg × Salt)) × Msg)}
  {w : Msg → ℝ} {b : Brings} {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
  {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}

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

theorem stepsWG_of (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg) (Φ : Msg → (ℕ → Coin) → List Claims.Flock.Wit → Prop)
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
    (hR : RowsW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt Φ) :
    StepsWG rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt Φ := by
  intro x st hx hf c ws hJ hΦ os last hfin hos hD q hq
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
    hR x st hx hf c ws hJ r ss s q.level LANES HH n hr hss hsv xO ws[s] p hw hst q hq ho ht hΦ os last hfin hos
      hD.some
  exact accHolds_of_copy (Rows.sideAt rd) Rows.coefRead (fun _ _ => chainW x ws) q _ _ _ hLO hOC.1 hOC.2.2.1 hCf
    hbi hszi hbo hszo hin hout

theorem schAt_fast100 {x : Msg} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt}
    (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss) {s lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) :
    fast100 r.shape.m = some (OpenLower.schAt (xIOf x) 0) := by
  obtain ⟨r', hr', -, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hh : r.shape.holo = false := hd.1
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hd.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  obtain ⟨sch, hsch, -, -, -, -⟩ := OpenLower.recOpen_dense hss hsp hh hsv
  obtain ⟨s', spec, t, hset, htm⟩ := OpenLower.inner_m hst hr
  have e : OpenLower.schAt (xIOf x) 0 = sch := by
    unfold OpenLower.schAt
    rw [hset]
    have htabs : FlockDeploy.tabsOf #[t] = #[_root_.Flock.Setup.ofCircuit t] := by simp [FlockDeploy.tabsOf]
    have h0 : (FlockDeploy.tabsOf #[t])[0]!.m = t.m := by
      rw [htabs]
      rfl
    dsimp only
    rw [h0, htm, hsch]
    rfl
  rw [e]
  exact hsch

theorem chainPos_eq (sch : Schedule) (rep l j : ℕ) :
    Rows.chainPos sch rep l j =
      rep * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l + j := by
  simp only [Rows.chainPos, Rows.nOpen, RecVStar.opensBelow, FlockSoundness.Discharge.Hidden.getElem!_eq_getD']

theorem sideAt_sch {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}
    (Dc : Rows.DecAt rd x ms cs) {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {q : OpenPiece}
    (hq : q ∈ Rows.rowDPieces key x) : (Rows.sideAt rd x ms q).sch = OpenLower.schAt x q.table := by
  rw [Dc.rowDPieces_eq key] at hq
  obtain ⟨hi, hr, -⟩ := Rows.mem_rowDPiecesOf.1 hq
  obtain ⟨hsch, -, -⟩ := Dc.sideAtR_opens Top.cls₀ q.table hi ⟨q.rep, hr⟩ q rfl rfl
  rw [Dc.sideAt_eq Top.cls₀ q.table hi ⟨q.rep, hr⟩ q rfl rfl]
  exact hsch.symm

theorem acc_holding {r : XR} {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {s lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {sch : Schedule}
    (hf : fast100 r.shape.m = some sch) {rep j : ℕ} (hrep : rep < 2) (hj : j < (sch.levels[lv]!).q) :
    ("acc", rep * (sch.levels[lv]!).q + j) ∈
      accHoldings r s (rep * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch lv + j) := by
  have hA : RecVStar.accReads r.shape (.recOpen lv LANES HH n) =
      .ok (some ((Array.range (2 * (sch.levels[lv]!).q)).map fun u =>
        some (u / (sch.levels[lv]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch lv +
          u % (sch.levels[lv]!).q))) := by
    unfold RecVStar.accReads
    simp only [hf]
    try rfl
  have hq : 0 < (sch.levels[lv]!).q := by omega
  have hu : rep * (sch.levels[lv]!).q + j < 2 * (sch.levels[lv]!).q := by
    interval_cases rep <;> omega
  have hdiv : (rep * (sch.levels[lv]!).q + j) / (sch.levels[lv]!).q = rep := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hq, Nat.div_eq_of_lt hj, Nat.zero_add]
  have hmod : (rep * (sch.levels[lv]!).q + j) % (sch.levels[lv]!).q = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
  have hst : (_root_.Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen lv LANES HH n) := by
    rw [hss]
    exact hsv
  unfold accHoldings
  rw [hst]
  dsimp only
  refine List.mem_filter.2 ⟨?_, ?_⟩
  · unfold accCands
    refine List.mem_append.2 (Or.inr (List.mem_map.2 ⟨_, List.mem_range.2 ?_, rfl⟩))
    simpa [hA, Except.toOption] using hu
  · unfold accRead
    simp [hA, Except.toOption, hu, hdiv, hmod]

theorem findIdx_isSome_of_outs {c : Circuit} {sub : VStar.Sub} {u : VStar.UnitC}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = RecVStar.vstarRows sub u) {nm : String}
    (hnm : nm ∈ sub.outputs) (hlen : sub.outputs.length ≤ u.outPorts.length) :
    (c.ports.findIdx? (·.name == nm)).isSome = true := by
  have hz : nm ∈ (sub.outputs.zip u.outPorts).map Prod.fst := by
    rw [List.map_fst_zip] <;> assumption
  obtain ⟨⟨a, k⟩, hm, rfl⟩ := List.mem_map.1 hz
  show (c.ports.findIdx? (·.name == a)).isSome = true
  have hv : (a, (k + 63) / 64 * 64, 16 * ((k + 63) / 64 * 64)) ∈ RecVStar.vstarRows sub u := by
    unfold RecVStar.vstarRows
    exact List.mem_append_right _ (List.mem_map.2 ⟨(a, k), hm, rfl⟩)
  rw [← hr] at hv
  obtain ⟨pt, hpt, he⟩ := List.mem_map.1 hv
  have hname : pt.name = a := congrArg Prod.fst he
  cases hf : c.ports.findIdx? (·.name == a) with
  | some _ => rfl
  | none =>
    have hn := Array.findIdx?_eq_none_iff.1 hf pt (Array.mem_toList_iff.1 hpt)
    simp [hname] at hn

theorem accOutPort_recOpen {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {ss : Array VStar.Stmt}
    (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) :
    (c.ports.findIdx? (·.name == "acc_out")).isSome = true := by
  obtain ⟨vs, hvs, -, hall⟩ := readX_partVStar hr
  obtain rfl : vs = ss := Except.ok.inj (hvs.symm.trans hss)
  obtain ⟨hs, hst'⟩ := Array.getElem?_eq_some_iff.1 hsv
  obtain ⟨-, p, c', -, hp, hpc, -, hpar, -⟩ := hall s hs
  have hcc : c' = c := by
    rw [hp, Option.bind_some, hpc] at hc
    exact Option.some_inj.1 hc
  subst hcc
  obtain ⟨u, hu, -, -, -, hports⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hpar
  rw [hst'] at hu hports
  obtain ⟨-, -, -, hout⟩ := FlockVBridge.OfVStar.recOpenUnit_ok hu
  refine findIdx_isSome_of_outs hports ?_ ?_
  · show "acc_out" ∈ ["out", "acc_out"]
    exact List.mem_cons_of_mem _ List.mem_cons_self
  · rw [hout]
    show ["out", "acc_out"].length ≤ [VStar.OUT_WORDS, VStar.ACC_WORDS].length
    simp

/-- Discharged at two level parts by `accCommit_eq`, and at a level part and an algebra part's `acc0`/`acc1` by
`innerAccCommit_eq`. -/
def AccCommitW (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ (r : XR), readX x = some r → ∀ (s s' : ℕ) (cc cc' : Circuit) (wt wt' : Claims.Flock.Wit),
      r.w.parts[s]?.bind partCircuit = some cc → r.w.parts[s']?.bind partCircuit = some cc' →
      ws[s]? = some wt → ws[s']? = some wt' →
    ∀ (p u : ℕ) (nm : String) (d : ℕ), (nm, d) ∈ [("acc", 0), ("acc_out", 1)] → ("acc", u) ∈ accHoldings r s p →
    ∀ h' ∈ accHoldings r s' (p + d), ∀ (qn q' : ℕ), cc.ports.findIdx? (·.name == nm) = some qn →
      cc'.ports.findIdx? (·.name == h'.1) = some q' →
      readForms wt (hmCommit cc qn u) = readForms wt' (hmCommit cc' q' h'.2)

def AccCommitWG (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (Φ : Msg → (ℕ → Coin) → List Claims.Flock.Wit → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws → Φ x c ws →
    ∀ (r : XR), readX x = some r → ∀ (s s' : ℕ) (cc cc' : Circuit) (wt wt' : Claims.Flock.Wit),
      r.w.parts[s]?.bind partCircuit = some cc → r.w.parts[s']?.bind partCircuit = some cc' →
      ws[s]? = some wt → ws[s']? = some wt' →
    ∀ (p u : ℕ) (nm : String) (d : ℕ), (nm, d) ∈ [("acc", 0), ("acc_out", 1)] → ("acc", u) ∈ accHoldings r s p →
    ∀ h' ∈ accHoldings r s' (p + d), ∀ (qn q' : ℕ), cc.ports.findIdx? (·.name == nm) = some qn →
      cc'.ports.findIdx? (·.name == h'.1) = some q' →
      readForms wt (hmCommit cc qn u) = readForms wt' (hmCommit cc' q' h'.2)

def NextHeldW (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (s lv LANES HH n : ℕ), ss[s]? = some (.recOpen lv LANES HH n) →
    ∀ (u p : ℕ), ("acc", u) ∈ accHoldings r s p → (accCanon r (p + 1)).isSome

theorem owner_of_holding {x : Msg} {r : XR} (hr : readX x = some r) {ws : List Claims.Flock.Wit}
    (hwl : ws.length = r.w.parts.length) {p : ℕ} (hp : (accCanon r p).isSome) :
    ∃ o c h w q, accCanon r p = some (o, c, h) ∧ ws[o]? = some w ∧ c.ports.findIdx? (·.name == h.1) = some q := by
  obtain ⟨⟨o, c, h⟩, hcan⟩ := Option.isSome_iff_exists.1 hp
  obtain ⟨ho, hc, hm⟩ := accCanon_spec hcan
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1
    (accPort_deployed hr hc (mem_accHeld (List.mem_of_mem_head? (Option.mem_def.2 hm))))
  exact ⟨o, c, h, ws[o], q, hcan, List.getElem?_eq_getElem (by omega), hq⟩

theorem rowsW_of (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg) (Φ : Msg → (ℕ → Coin) → List Claims.Flock.Wit → Prop)
    (hCm : AccCommitWG rd wE final w b elem Claims.Recursion.Form.Dense untr cmt Φ)
    (hN : NextHeldW rd wE final w b elem Claims.Recursion.Form.Dense) :
    RowsW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt Φ := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hw hst q hq ho ht hΦ os last hfin hos Dc
  beta_reduce
  have hxI : Claims.Recursion.xI (DRL rd wE final w b elem) x = xIOf x := rfl
  obtain ⟨h0, r', ss', hr', hss', LANES', HH', n', hsv'⟩ := OpenLower.ownQR_some ho
  obtain rfl : r = r' := Option.some.inj (hr.symm.trans hr')
  obtain rfl : ss = ss' := Except.ok.inj (hss.symm.trans hss')
  have hlv : lv = q.level := by
    rw [hsv] at hsv'
    cases hsv'
    rfl
  subst hlv
  obtain ⟨vs, hvs, hlen, hall⟩ := readX_partVStar hr
  obtain rfl : ss = vs := Except.ok.inj (hss.symm.trans hvs)
  have hs : s < ss.size := (Array.getElem?_eq_some_iff.1 hsv).1
  obtain ⟨-, pt, cc, -, hpt, hpc, -⟩ := hall s hs
  have hc : r.w.parts[s]?.bind partCircuit = some cc := by rw [hpt, Option.bind_some, hpc]
  have hwl : ws.length = r.w.parts.length := by
    rw [hJ.1]
    show RecDeploy.nS x = _
    unfold RecDeploy.nS
    rw [hr]
  have hfast := schAt_fast100 hx hf hr hss hsv
  obtain ⟨s', spec, t, hset, -⟩ := OpenLower.inner_m hx hr
  have hRS : OpenLower.schAt (xIOf x) 0 = Rows.schAt #[t] 0 := by
    unfold OpenLower.schAt Rows.schAt
    rw [hset]
  have hq1 := hq
  rw [hxI] at hq1
  unfold Rows.rowDPieces at hq1
  rw [hset] at hq1
  obtain ⟨-, hrep, -, hj, -⟩ := Rows.mem_rowDPiecesOf.1 hq1
  rw [h0, ← hRS] at hj
  have hh : ("acc", q.rep * OpenLower.qAt (Claims.Recursion.xI (DRL rd wE final w b elem) x) q.level + q.query) ∈
      accHoldings r s (Rows.chainPos (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last) q).sch q.rep q.level q.query) := by
    rw [sideAt_sch Dc hq, h0, hxI, chainPos_eq]
    exact acc_holding hss hsv hfast hrep hj
  have hS := hst.2.2.2.2.2.2.2.2
  obtain ⟨bi, hbi, hszi, hbi'⟩ := accBytes_staged hS hst.1 ht hr hss hsv hc wt
    (q.rep * OpenLower.qAt (Claims.Recursion.xI (DRL rd wE final w b elem) x) q.level + q.query)
  obtain ⟨qo, hqo⟩ := Option.isSome_iff_exists.1 (accOutPort_recOpen hr hc hss hsv)
  obtain ⟨bo, hbo, hszo, hbo'⟩ := accOutBytes_hidden hS hst.1 ht (staged_hidden hS hst.1 ht) hr hss hsv hc hqo wt
    (q.rep * OpenLower.qAt (Claims.Recursion.xI (DRL rd wE final w b elem) x) q.level + q.query)
  obtain ⟨qa, hqa⟩ := Option.isSome_iff_exists.1 (accPort_deployed hr hc (mem_accHeld hh))
  have hsP : s < r.w.parts.length := by omega
  have hown₀ : (accCanon r (Rows.chainPos (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
      ((_root_.RecDeploy.bdR rd wE final w).opened os last) q).sch q.rep q.level q.query)).isSome := by
    obtain ⟨e, he⟩ : ∃ e, accHolder r (Rows.chainPos (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last) q).sch q.rep q.level q.query) s = some e := by
      unfold accHolder
      rw [hc, Option.bind_some]
      revert hh
      cases accHoldings r s (Rows.chainPos (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last) q).sch q.rep q.level q.query) with
      | nil => intro hh; cases hh
      | cons a l => intro _; exact ⟨_, rfl⟩
    exact accCanon_isSome hsP he
  obtain ⟨o₀, c₀, h₀, w₀, q₀, hcan₀, hw₀, hq₀⟩ := owner_of_holding hr hwl hown₀
  obtain ⟨o₁, c₁, h₁, w₁, q₁, hcan₁, hw₁, hq₁⟩ := owner_of_holding hr hwl (hN x st hx hf r hr ss hss s q.level LANES HH
    n hsv _ _ hh)
  obtain ⟨hc₀, hm₀⟩ := accHeld_of_canon hcan₀
  obtain ⟨hc₁, hm₁⟩ := accHeld_of_canon hcan₁
  have hcm₀ := hCm x st hx hf c ws hJ hΦ r hr s o₀ cc c₀ wt w₀ hc hc₀ hw hw₀ _ _ "acc" 0 List.mem_cons_self hh h₀
    hm₀ qa q₀ hqa hq₀
  have hcm₁ := hCm x st hx hf c ws hJ hΦ r hr s o₁ cc c₁ wt w₁ hc hc₁ hw hw₁ _ _ "acc_out" 1
    (List.mem_cons_of_mem _ List.mem_cons_self) hh h₁ hm₁ qo q₁ hqo hq₁
  have e₀ := held_rows_eq elem hJ.2.1 hJ.2.2.1 hr hc hc₀ hw hw₀ (mem_accHeld hh) (mem_accHeld hm₀) hqa hq₀ hcm₀
  have e₁ := out_rows_eq elem hJ.2.1 hJ.2.2.1 hr hc hc₁ hw hw₁ (mem_accHeld hh) (mem_accHeld hm₁) hqo hq₁ hcm₁
  refine ⟨bi, bo, hbi, hszi, hbo, hszo, ?_, ?_⟩
  · rw [chainW_owner hr hcan₀ hw₀ q.table, hbi', e₀]
  · rw [chainW_owner hr hcan₁ hw₁ q.table, hbo', e₁]

theorem stepsWG_of_commits (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg) (Φ : Msg → (ℕ → Coin) → List Claims.Flock.Wit → Prop)
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
    (hCm : AccCommitWG rd wE final w b elem Claims.Recursion.Form.Dense untr cmt Φ)
    (hN : NextHeldW rd wE final w b elem Claims.Recursion.Form.Dense) :
    StepsWG rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt Φ :=
  stepsWG_of rd key wE final w b elem untr cmt Φ hL hO hC (rowsW_of rd key wE final w b elem untr cmt Φ hCm hN)

theorem stepsW_of_commits (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
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
    (hCm : AccCommitW rd wE final w b elem Claims.Recursion.Form.Dense untr cmt)
    (hN : NextHeldW rd wE final w b elem Claims.Recursion.Form.Dense) :
    StepsW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt :=
  stepsW_of_G (stepsWG_of_commits rd key wE final w b elem untr cmt (fun _ _ _ => True) hL hO hC
    (fun x st hx hf c ws hJ _ => hCm x st hx hf c ws hJ) hN)

end RecDeploy.ResLower
