module

public import Proofs.Flock.Claims.Recursion.DeployLayout
public import Proofs.Flock.Claims.Recursion.VStarBound
public import Proofs.Flock.Recursive.InnerRows
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

/-! The link clause at the deployment (`LinksHold`), which holds the link table's rows to the registered table at `T`, not to one the prover picks. -/

@[expose] public section

namespace RecDeploy.Links

open Flock
open Claims (Msg)

def linkAt (r : XR) : Option (ℕ × ℕ) :=
  (RecVStar.statements r.shape).toOption.bind fun ss =>
    match ss[ss.size - 2]? with
    | some (VStar.Stmt.link _ _ N) => some (ss.size - 2, N)
    | _ => none

def linkCircuit (r : XR) : Option (ℕ × ℕ × Circuit) := do
  let (s, N) ← linkAt r
  let c ← r.w.parts[s]?.bind partCircuit
  pure (s, N, c)

/-- **Instance `n`'s `trow`, as a part's witness holds it**: the bytes of the row's hidden values at the instance's input
columns (`rowHidden`), the row its hm96 slot commits behind its prefix (`SlotCommits`). -/
def trowAt (c : Circuit) (w : Claims.Flock.Wit) (n : ℕ) : List UInt8 :=
  Claims.Recursion.VStarB.readForms w (((rowHidden c "trow" n).getD []).map HmNets.Lin.var)

def tableRows (x : Msg) (ws : List Claims.Flock.Wit) : List (List UInt8) :=
  match readX x with
  | none => []
  | some r => match linkCircuit r with
    | none => []
    | some (s, N, c) => (List.range N).map (trowAt c (ws.getD s ⟨[]⟩))

def tableDomain (c : Circuit) (N : ℕ) : Option ByteArray := do
  let prog ← (c.metaJ.getObjValAs? String "program_sha512").toOption
  (Registered.domain prog RecLinkPart.LINKS_VALUE VStar.LINK_ROW_WORDS N 16).toOption

/-- `Registered.opens` reads only its root, leaves, domain and schema. -/
def tablePort (dom T : ByteArray) (N : ℕ) : Registered.Port where
  name := "trow"
  value := RecLinkPart.LINKS_VALUE
  root := T
  leaves := N
  words := some VStar.LINK_ROW_WORDS
  wordBits := 16
  domain := dom
  readsRel := #[]

def rowMsgOf (row : List UInt8) : ByteArray := FlockSoundness.Discharge.Hm.rowMsg VStar.LINK_ROW_WORDS row

def RowOpens (dom T : ByteArray) (N n : ℕ) (row : List UInt8) : Prop :=
  ∃ salt path : ByteArray,
    Registered.opens (tablePort dom T N) (Hm96.Default512.commitString (Sha512.hash (rowMsgOf row)) salt) n path = true

theorem rowMsgOf_inj {r r' : List UInt8} (h : rowMsgOf r = rowMsgOf r') : r = r' := by
  have h := congrArg (fun b : ByteArray => b.data.toList) h
  simp only [rowMsgOf, FlockSoundness.Discharge.Hm.rowMsg, List.toList_toArray] at h
  exact List.append_cancel_left h

theorem rowOpens_binding {dom T : ByteArray} {N n : ℕ} {row row' : List UInt8} (h : RowOpens dom T N n row)
    (h' : RowOpens dom T N n row') :
    row = row' ∨ (∃ u v : ByteArray, u ≠ v ∧ Sha512.hash u = Sha512.hash v) := by
  obtain ⟨salt, path, ho⟩ := h
  obtain ⟨salt', path', ho'⟩ := h'
  by_cases hrow : row = row'
  · exact .inl hrow
  refine .inr ?_
  by_cases hm : rowMsgOf row = rowMsgOf row'
  · exact absurd (rowMsgOf_inj hm) hrow
  by_cases hd : Sha512.hash (rowMsgOf row) = Sha512.hash (rowMsgOf row')
  · exact ⟨_, _, hm, hd⟩
  have hne : (Sha512.hash (rowMsgOf row), salt) ≠ (Sha512.hash (rowMsgOf row'), salt') :=
    fun he => hd (Prod.mk.inj he).1
  by_cases hl : Hm96.Default512.leaf (Sha512.hash (rowMsgOf row)) salt =
      Hm96.Default512.leaf (Sha512.hash (rowMsgOf row')) salt'
  · exact Hm96.Default512.leaf_binding _ _ _ _ (Sha512.hash_size _) (Sha512.hash_size _) hne hl
  · exact Registered.opensLeaf_collision (tablePort dom T N) n (Hm96.Default512.leaf (Sha512.hash (rowMsgOf row)) salt)
      path (Hm96.Default512.leaf (Sha512.hash (rowMsgOf row')) salt') path' (Hm96.Default512.leaf_size _ _)
      (Hm96.Default512.leaf_size _ _) hl ho ho'

theorem RecIns.setup_linkPart {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    sts.forM (RecLinkPart.check · s.base.registered s.src s.links) = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem checkLinks_ok {st : Stmt} {own : Option (Array Registered.Port)} {files : Option RecCoins.Source}
    {links : Option (ByteArray × ℕ)} {UB PB : ℕ} (h : RecLinkPart.checkLinks st own files links UB PB = .ok ()) :
    ∃ T N own' f sf R W, links = some (T, N) ∧ own = some own' ∧ files = some (.files f) ∧
      RecLinkPart.checkTable st.c st.pub own' T N = .ok () ∧ f.stmt st.c.unit.name = .ok sf ∧
      RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).1 = .ok R ∧
      RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).2 = .ok W ∧
      (∃ what, RecCoins.checkValue st.c st.pub own' "rin" RecLinkPart.IN_VALUE
        (Array.replicate N (RecLinkRef.expected R).toArray) what = .ok ()) ∧
      ∃ what, RecCoins.checkValue st.c st.pub own' "rout" RecLinkPart.OUT_VALUE
        (Array.replicate N (RecLinkRef.expected W).toArray) what = .ok () := by
  unfold RecLinkPart.checkLinks at h
  rcases links with _ | ⟨T, N⟩
  · cases h
  rcases own with _ | own'
  · cases h
  rcases files with _ | ⟨f⟩ | ⟨j, rec⟩
  · cases h
  · obtain ⟨_, h1, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨sf, h2, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨R, h3, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨W, h4, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨_, h5, h6⟩ := FlockSoundness.Refine.ebind_ok h
    have h3' : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).1 = .ok R := by
      change Except.mapError _ (RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).1) = Except.ok R at h3
      cases e : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).1 with
      | error e' => rw [e] at h3; cases h3
      | ok R' => rw [e] at h3; cases h3; rfl
    have h4' : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).2 = .ok W := by
      change Except.mapError _ (RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).2) = Except.ok W at h4
      cases e : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).2 with
      | error e' => rw [e] at h4; cases h4
      | ok W' => rw [e] at h4; cases h4; rfl
    exact ⟨T, N, own', f, sf, R, W, rfl, rfl, rfl, h1, h2, h3', h4', ⟨_, h5⟩, ⟨_, h6⟩⟩
  · cases h

local instance lawfulBEqByteArrayLinks : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

theorem checkTable_ok {c : Circuit} {pub : Public} {own : Array Registered.Port} {T : ByteArray} {N : ℕ}
    (h : RecLinkPart.checkTable c pub own T N = .ok ()) :
    ∃ reg, own.find? (·.name == "trow") = some reg ∧ reg.value = RecLinkPart.LINKS_VALUE ∧ reg.root = T ∧
      reg.leaves = N ∧ reg.words = some VStar.LINK_ROW_WORDS ∧ reg.wordBits = 16 ∧
      reg.schema = HmRow.ROW_SCHEMA ∧ tableDomain c N = some reg.domain ∧
      reg.reads = (Array.range N).map some ∧ Registered.unitIndices pub = .ok (Array.range pub.n) := by
  unfold RecLinkPart.checkTable at h
  cases hf : own.find? (·.name == "trow") with
  | none => simp only [hf] at h; cases h
  | some reg =>
    simp only [hf] at h
    by_cases hv : (reg.value != RecLinkPart.LINKS_VALUE) = true
    · simp only [hv, ↓reduceIte] at h; cases h
    simp only [hv, Bool.false_eq_true, ↓reduceIte] at h
    by_cases hr : (reg.root != T) = true
    · simp only [hr, ↓reduceIte] at h; cases h
    simp only [hr, Bool.false_eq_true, ↓reduceIte] at h
    by_cases hl : (reg.leaves != N || reg.words != some VStar.LINK_ROW_WORDS || reg.wordBits != 16 ||
        reg.schema != HmRow.ROW_SCHEMA) = true
    · simp only [hl, ↓reduceIte] at h; cases h
    simp only [hl, Bool.false_eq_true, ↓reduceIte] at h
    cases hp : c.metaJ.getObjValAs? String "program_sha512" with
    | error e => simp only [hp] at h; cases h
    | ok prog =>
      simp only [hp] at h
      cases hd : Registered.domain prog RecLinkPart.LINKS_VALUE VStar.LINK_ROW_WORDS N 16 with
      | error e => simp only [hd, bind, Except.bind] at h; cases h
      | ok dom =>
        simp only [hd, bind, Except.bind] at h
        by_cases hdm : (reg.domain != dom) = true
        · simp only [hdm, ↓reduceIte] at h; cases h
        simp only [hdm, Bool.false_eq_true, ↓reduceIte] at h
        by_cases hrd : (reg.reads != (Array.range N).map some) = true
        · simp only [hrd, ↓reduceIte] at h; cases h
        simp only [hrd, Bool.false_eq_true, ↓reduceIte] at h
        cases hi : Registered.unitIndices pub with
        | error e => simp only [hi] at h; cases h
        | ok idx =>
          simp only [hi] at h
          by_cases hidx : (idx != Array.range pub.n) = true
          · simp only [hidx, ↓reduceIte] at h; cases h
          simp only [Bool.not_eq_true, bne_eq_false_iff_eq, Bool.or_eq_false_iff] at hv hr hl hdm hrd hidx
          obtain ⟨⟨⟨hl1, hl2⟩, hl3⟩, hl4⟩ := hl
          refine ⟨reg, rfl, hv, hr, hl1, hl2, hl3, hl4, ?_, hrd, by rw [hidx]⟩
          unfold tableDomain
          simp [hp, hd, Except.toOption, hdm]

theorem opens_tablePort {reg : Registered.Port} {dom T : ByteArray} {N : ℕ} (hr : reg.root = T) (hl : reg.leaves = N)
    (hd : reg.domain = dom) (hs : reg.schema = HmRow.ROW_SCHEMA) (bc : ByteArray) (n : ℕ) (path : ByteArray) :
    Registered.opens reg bc n path = Registered.opens (tablePort dom T N) bc n path := by
  unfold Registered.opens Registered.opensLeaf
  rw [hr, hl, hd, hs]
  rfl

def innerParsed (r : XR) : Option _root_.Flock.Readable.Parsed :=
  _root_.Flock.Readable.parseBytes (FlockDeploy.readOf false) r.w.inner

/-- A row that is not a link (`KIND_LINK`) says nothing. -/
def LinkHolds (p : _root_.Flock.Readable.Parsed) (w : Claims.Flock.Wit) (row : List UInt8) : Prop :=
  let f := RecLinkRef.field ⟨row.toArray⟩
  f 0 = RecLinks.KIND_LINK →
    f 1 < p.sts[0]!.pub.n ∧ f 2 < HmOut.nIn p.sts[0]!.c ∧ f 3 < p.sts[0]!.pub.n ∧ f 4 < p.sts[0]!.c.outRowPorts.size ∧
      InnerRows.rowBytes p w 0 (f 1) (f 2) = InnerRows.rowBytes p w 0 (f 3) (HmOut.nIn p.sts[0]!.c + f 4)

/-- It says nothing at a statement carrying no table. -/
def LinksHold (x : Msg) (w : Claims.Flock.Wit) (rs : List (List UInt8)) : Prop :=
  ∀ r, readX x = some r → ∀ T N, r.w.links = some (T, N) →
    rs.length = N ∧
      (∀ s N' c dom, linkCircuit r = some (s, N', c) → tableDomain c N = some dom →
        ∀ n (hn : n < rs.length), RowOpens dom T N n rs[n]) ∧
      ∀ p, innerParsed r = some p → ∀ row ∈ rs, LinkHolds p w row

theorem linksHold_none {x : Msg} {r : XR} (hr : readX x = some r) (hl : r.w.links = none) (w : Claims.Flock.Wit)
    (rs : List (List UInt8)) : LinksHold x w rs := by
  intro r' hr' T N hl'
  rw [hr] at hr'
  cases hr'
  rw [hl] at hl'
  cases hl'

theorem linksHold_rows {x : Msg} {r : XR} (hr : readX x = some r) {T : ByteArray} {N : ℕ}
    (hl : r.w.links = some (T, N)) {s N' : ℕ} {c : Circuit} {dom : ByteArray} (hc : linkCircuit r = some (s, N', c))
    (hd : tableDomain c N = some dom) {w w' : Claims.Flock.Wit} {rs rs' : List (List UInt8)} (h : LinksHold x w rs)
    (h' : LinksHold x w' rs') : rs = rs' ∨ (∃ u v : ByteArray, u ≠ v ∧ Sha512.hash u = Sha512.hash v) := by
  obtain ⟨hn, ho, -⟩ := h r hr T N hl
  obtain ⟨hn', ho', -⟩ := h' r hr T N hl
  by_cases hc' : (∃ u v : ByteArray, u ≠ v ∧ Sha512.hash u = Sha512.hash v)
  · exact .inr hc'
  refine .inl (List.ext_getElem (hn.trans hn'.symm) fun n h1 h2 => ?_)
  rcases rowOpens_binding (ho s N' c dom hc hd n h1) (ho' s N' c dom hc hd n h2) with he | he
  · exact he
  · exact absurd he hc'

end RecDeploy.Links
