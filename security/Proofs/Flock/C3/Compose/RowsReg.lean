module

public import Proofs.Flock.C3.Compose.Honest
public import Proofs.Flock.VBridge.Family.Registered
import Flock.Desc
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
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenDec.Assemble
import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteHP
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-! The rows `own` names are registered reads at candidate 3's family (`RowsRegisteredC3`), from the premise
`RegistrantCovers`. -/

@[expose] public section

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled

def CoversAt {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') : Prop :=
  ∀ (u : Fin Q.nv) (q : Fin Q.c.ports.size), (∃ pt ∈ z.own, pt.name = Q.c.ports[q].name) → z.rd.isRead (.inl (u, q))

theorem fresh_not_named {c : Flock.Circuit} (q : Fin c.ports.size) {pt : Flock.Registered.Port}
    (hpt : pt ∈ freshOwn c) : pt.name ≠ c.ports[q].name := by
  intro h
  have hpt' : pt = { (default : Flock.Registered.Port) with name := freshName c } := Array.mem_singleton.mp hpt
  have hn : pt.name = freshName c := by rw [hpt']
  exact freshName_ne c q.isLt (by rw [getElem!_pos c.ports q.val q.isLt]; exact h.symm.trans hn)

theorem coversAt_keepOwn {Q : VStmt} {Reg' : Type} {z : Q.Outer Reg'}
    {P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop} (hz : CoversAt z) :
    CoversAt (Flock.Guarantees.keepOwn z P) := by
  intro u q ⟨pt, hpt, hn⟩
  by_cases hp : ∃ R' t, z.Writes R' t ∧ AcceptsZKR z.I z.own t ∧ P z.I z.own t
  · have ho : (Flock.Guarantees.keepOwn z P).own = z.own := ite_eq_left hp
    rw [ho] at hpt
    exact hz u q ⟨pt, hpt, hn⟩
  · have ho : (Flock.Guarantees.keepOwn z P).own = freshOwn Q.c := ite_eq_right hp
    rw [ho] at hpt
    exact absurd hn (fresh_not_named q hpt)

theorem coversAt_refuse {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') : CoversAt z.refuse :=
  fun _ q ⟨_, hpt, hn⟩ => absurd hn (fresh_not_named q hpt)

theorem coversAt_zsOf {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    {zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg'} (hcov : Flock.Premises.RegistrantCovers zr)
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr) (R : A) (ω : B) (t : C) (R₂ : D) (j : Fin S) :
    CoversAt (zsOf zr hRec h R ω t R₂ j) := by
  by_cases hx : (zr R ω t R₂ j).Typable
  · rw [zsOf_pos zr hRec h R ω t R₂ j hx]
    exact fun u q hq => hcov R ω t R₂ j _ _ _ _ _ _ u q hq
  · rw [zsOf_neg zr hRec h R ω t R₂ j hx]
    exact coversAt_refuse _

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

namespace C3Family

variable (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)

/-- Acceptance does not give it: a prover's record with no registered reads meets `hown` and `ReadsX`
vacuously and stays accepted. At the checked family `RegistrantCovers` gives it (`rowsCovered_checked`). -/
def RowsRegisteredC3 (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = (F.Vs j).c.ports[q].name) →
      (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))

def RowsCovered : Prop :=
  ∀ R ω t R₂ j, CoversAt (F.zs R ω t R₂ j)

theorem rowsRegisteredC3_of_covered (h : F.RowsCovered) (dg : ByteArray → ByteArray) (N k Rw : ℕ) :
    F.RowsRegisteredC3 dg N k Rw :=
  fun R ω t R₂ _ _ _ j u q hq => h R ω t R₂ j u q hq

end C3Family

theorem rowsCovered_checked (ish : RecVStar.Inner) {ss : Array VStar.Stmt} (hss : RecVStar.statements ish = .ok ss)
    (bind : Fin (vstarCount ish) → VStar.Bindings) {Vs : Fin (vstarCount ish) → VStmt}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr) (h : Typed zr) (coins : String → Option Flock.RecCoins.Source)
    (rows : Option (Array Flock.Registered.Port))
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json)
    (hcov : Flock.Premises.RegistrantCovers zr) :
    (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).RowsCovered :=
  fun R ω t R₂ j => by
    refine coversAt_keepOwn ?_
    dsimp only [Flock.Guarantees.custodied]
    exact coversAt_zsOf hcov _ h R ω t R₂ j

end FlockC3.Compose

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL Composed
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

theorem reads_of_acceptsZKR (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hTags : TagsOkZHJ I)
    {own : Array Registered.Port} {tr : Exec.Transcript} (hacc : AcceptsZKR I own tr) {reg : Registered.Port}
    (hmem : reg ∈ own) {u : ℕ} {d : Lean.Json} {D : Draw.UnitSample} {dp : Public} {idx : Array ℕ}
    (hd : recordDraw tr.record = some d) (hD : Draw.UnitSample.ofJson (drawJ I d (partOf I d u)) = .ok D)
    (hdp : HmOut.drawn c pub (drawJ I d (partOf I d u)) = .ok dp) (hidx : Registered.unitIndices dp = .ok idx)
    (hu : u ∈ D.proved) : ∃ x, reg.reads[idx.getD (D.proved.idxOf u) 0]? = some (some x) := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hTags.1 hd hs
  have hjs : (partOf I d u).val < sts.size := by rw [hsz]; exact (partOf I d u).isLt
  have s := Hidden.setupHidden_spec' (hstJ _ hjs)
  have hcst : sts[(partOf I d u).val].c = c := by
    have h1 := s.parseF ((FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl hc))
    rw [parseH_tagsAt ((zkTags_typed I).trans ht), hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, loadPublicH_tagsAt, hpub] at hpop'
  cases hpop'
  have hdr' := hdr _ rfl
  rw [hcst, hdp] at hdr'
  have hpd : sts[(partOf I d u).val].pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub (drawJ I d (partOf I d u)) = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg _ hjs)
  rw [hcst, hpd] at hck
  obtain ⟨tb, idx', -, hidx', hports⟩ := check_tables hck
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions, paths, -, -, hall⟩ := checkPort_at (hports _ hmem)
  obtain ⟨D', hD', -, hn, -, -⟩ := Layout.sampled_spec hdpR
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, -⟩ := hall _ (by rw [hn]; exact hi)
  exact ⟨_, hreads⟩

theorem sampled_of_acceptsZKR (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hTags : TagsOkZHJ I)
    {own : Array Registered.Port} {tr : Exec.Transcript} (hacc : AcceptsZKR I own tr) {d : Lean.Json}
    (hd : recordDraw tr.record = some d) (u : ℕ) :
    ∃ D dp idx, Draw.UnitSample.ofJson (drawJ I d (partOf I d u)) = .ok D ∧
      HmOut.drawn c pub (drawJ I d (partOf I d u)) = .ok dp ∧ Registered.unitIndices dp = .ok idx := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hTags.1 hd hs
  have hjs : (partOf I d u).val < sts.size := by rw [hsz]; exact (partOf I d u).isLt
  have s := Hidden.setupHidden_spec' (hstJ _ hjs)
  have hcst : sts[(partOf I d u).val].c = c := by
    have h1 := s.parseF ((FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl hc))
    rw [parseH_tagsAt ((zkTags_typed I).trans ht), hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, loadPublicH_tagsAt, hpub] at hpop'
  cases hpop'
  have hdp := hdr _ rfl
  rw [hcst] at hdp
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub (drawJ I d (partOf I d u)) =
      .ok sts[(partOf I d u).val].pub := hdp
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec hdpR
  have hck := checkRegistered_ok (array_forM_ok hreg _ hjs)
  rw [hcst] at hck
  obtain ⟨-, idx, -, hidx, -⟩ := check_tables hck
  exact ⟨D, _, idx, hD, hdp, hidx⟩

theorem names_of_acceptsZKR (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hTags : TagsOkZHJ I)
    {own : Array Registered.Port} {tr : Exec.Transcript} (hacc : AcceptsZKR I own tr) {d : Lean.Json}
    (hd : recordDraw tr.record = some d) :
    ∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q' := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hTags.1 hd hs
  have hjs : (partOf I d 0).val < sts.size := by rw [hsz]; exact (partOf I d 0).isLt
  have hcst : sts[(partOf I d 0).val].c = c := by
    have h1 := (Hidden.setupHidden_spec' (hstJ _ hjs)).parseF
      ((FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl hc))
    rw [parseH_tagsAt ((zkTags_typed I).trans ht), hc] at h1
    exact (Except.ok.inj h1).symm
  have hdn := dupName_of_checkRegistered (array_forM_ok hreg _ hjs)
  rw [hcst] at hdn
  exact names_once_of_dupName hdn

end FlockSoundness.Discharge.ZkReg

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.FailClosed (pubZ)

abbrev NamedBy (own : Array Flock.Registered.Port) (c : Flock.Circuit) (q : Fin c.ports.size) : Prop :=
  ∃ pt ∈ own, pt.name = c.ports[q].name

abbrev NamedRow {nv : ℕ} {G : Type} (own : Array Flock.Registered.Port) (c : Flock.Circuit) :
    (Fin nv × Fin c.ports.size) ⊕ G → Prop :=
  fun p => ∃ u q, p = .inl (u, q) ∧ NamedBy own c q

def portOf (own : Array Flock.Registered.Port) (nm : String) : Flock.Registered.Port :=
  (own.toList.find? (·.name == nm)).getD default

theorem portOf_spec {own : Array Flock.Registered.Port} {nm : String} (h : ∃ pt ∈ own, pt.name = nm) :
    portOf own nm ∈ own ∧ (portOf own nm).name = nm := by
  obtain ⟨pt, hpt, hn⟩ := h
  have hs : (own.toList.find? (·.name == nm)).isSome = true :=
    List.find?_isSome.mpr ⟨pt, by simpa using hpt, by simpa using hn⟩
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp hs
  have h1 : portOf own nm = a := by
    show (own.toList.find? (·.name == nm)).getD default = a
    rw [ha]
    rfl
  rw [h1]
  refine ⟨by simpa using List.mem_of_find?_eq_some ha, ?_⟩
  have hb := List.find?_some ha
  simpa using hb

def NamesOnce (c : Flock.Circuit) : Prop :=
  ∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q'

theorem find_of_names {c : Flock.Circuit} (hN : NamesOnce c) (q : Fin c.ports.size) :
    (List.range c.ports.size).find? (fun p => c.ports[p]!.name == c.ports[q].name) = some q.val := by
  rw [List.find?_eq_some_iff_getElem]
  refine ⟨by rw [getElem!_pos c.ports q.val q.isLt]; simp, q.val, by simp, by simp, fun k hk => ?_⟩
  have hk' : k < c.ports.size := by omega
  simp only [List.getElem_range]
  rw [getElem!_pos c.ports k hk']
  have hne : c.ports[k].name ≠ c.ports[q].name := fun he =>
    Nat.ne_of_lt hk (congrArg Fin.val (hN ⟨k, hk'⟩ q he))
  simpa using hne

section Honest

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nv}

/-- The label is the one the sampled statement gives the unit's instance, which the verifier derives
(`Registered.derived`); `loadPublic` does not make it the unit's own index. -/
def RegistrantOpens (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) : Prop :=
  ∀ (ht : x.I.tags.typed = false) (S₀ : Finset (Fin nv)) (d₀ : Lean.Json)
    (y₀ : ZkHidden.SampleSetupZHJ (inputsJ x.I 0) mPts S₀ d₀)
    (hc : HmOut.parse (zkTags x.I) x.I.circuitFile x.I.tables = .ok c)
    (hpub : HmOut.loadPublic (zkTags x.I) c x.I.publicFile = .ok (pubZ x.I)),
    ∃ rdH : FlockSoundness.Registered.ReadsX
        (ZkHidden.layShHJ (Reg := Reg') x.dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU (pub := pubZ x.I))
        (rsShHJS x.dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU hpub),
      ∀ (u : Fin nv) (q : Fin c.ports.size), NamedBy x.own c q →
        rdH.isRead (.inl (u, q)) ∧ rdH.port (.inl (u, q)) = portOf x.own c.ports[q].name ∧
          ∀ d D dp idx, Flock.Draw.UnitSample.ofJson (drawJ x.I d (partOf x.I d u.val)) = .ok D →
            HmOut.drawn c (pubZ x.I) (drawJ x.I d (partOf x.I d u.val)) = .ok dp →
            Flock.Registered.unitIndices dp = .ok idx → u.val ∈ D.proved → ∀ pos,
              (portOf x.own c.ports[q].name).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some pos) →
                rdH.readAt (.inl (u, q)) = pos

theorem labelsSelf (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) (hRec : RecordsLiveZKJ x.I x.dj (fun _ _ => J) x.wr)
    (ht : x.I.tags.typed = false) (hc : HmOut.parse (zkTags x.I) x.I.circuitFile x.I.tables = .ok c)
    (hpub : HmOut.loadPublic (zkTags x.I) c x.I.publicFile = .ok (pubZ x.I)) (ω : Lv.Ω) (R' : Reg')
    (o : Tables.LiveOutJ J mPts) (hacc : LiveAcceptsZKR x.I x.own (R', Lv.draw ω, x.wr (Lv.draw ω) R' o))
    (u : Fin nv) (hu : u ∈ Lv.draw ω) :
    ∃ d D dp idx, recordDraw (x.wr (Lv.draw ω) R' o).record = some d ∧
      Flock.Draw.UnitSample.ofJson (drawJ x.I d (partOf x.I d u.val)) = .ok D ∧
      HmOut.drawn c (pubZ x.I) (drawJ x.I d (partOf x.I d u.val)) = .ok dp ∧
      Flock.Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved := by
  have hd := (hRec _ _ _ (acceptsZK_of_R (t := x.wr (Lv.draw ω) R' o) hacc)).1
  obtain ⟨D, dp, idx, hD, hdp, hidx⟩ :=
    FlockSoundness.Discharge.ZkReg.sampled_of_acceptsZKR x.I ht hc hpub x.hTags (tr := x.wr (Lv.draw ω) R' o) hacc hd
      u.val
  exact ⟨_, D, dp, idx, hd, hD, hdp, hidx, ((x.hDraw ω) (partOf x.I _ u.val)).1 D hD u (mem_partS hu)⟩

theorem ownReads_cover (x : ZkOuterR mPts nv J Reg' c hU hscope Lv)
    (hRec : RecordsLiveZKJ x.I x.dj (fun _ _ => J) x.wr) (ht : x.I.tags.typed = false) {S₀ : Finset (Fin nv)}
    {d₀ : Lean.Json}
    (y₀ : ZkHidden.SampleSetupZHJ (inputsJ x.I 0) mPts S₀ d₀)
    (hc : HmOut.parse (zkTags x.I) x.I.circuitFile x.I.tables = .ok c)
    (hpub : HmOut.loadPublic (zkTags x.I) c x.I.publicFile = .ok (pubZ x.I))
    (rdH : FlockSoundness.Registered.ReadsX
        (ZkHidden.layShHJ (Reg := Reg') x.dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU (pub := pubZ x.I))
        (rsShHJS x.dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU hpub))
    (hH : ∀ (u : Fin nv) (q : Fin c.ports.size), NamedBy x.own c q →
      rdH.isRead (.inl (u, q)) ∧ rdH.port (.inl (u, q)) = portOf x.own c.ports[q].name ∧
        ∀ d D dp idx, Flock.Draw.UnitSample.ofJson (drawJ x.I d (partOf x.I d u.val)) = .ok D →
          HmOut.drawn c (pubZ x.I) (drawJ x.I d (partOf x.I d u.val)) = .ok dp →
          Flock.Registered.unitIndices dp = .ok idx → u.val ∈ D.proved → ∀ pos,
            (portOf x.own c.ports[q].name).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some pos) →
              rdH.readAt (.inl (u, q)) = pos)
    (ω : Lv.Ω) (R' : Reg') (o : Tables.LiveOutJ J mPts)
    (hacc : LiveAcceptsZKR x.I x.own (R', Lv.draw ω, x.wr (Lv.draw ω) R' o)) :
    OwnReadsHJ x.I x.own c (pubZ x.I) (x.wr (Lv.draw ω) R' o)
      ((rdH.restrict (NamedRow x.own c)).restrict (rowIn (Lv.draw ω))).isRead rdH.port rdH.readAt := by
  rintro p ⟨⟨-, u, q, rfl, hn⟩, hu⟩
  obtain ⟨-, hport, hat⟩ := hH u q hn
  obtain ⟨hmem, hname⟩ := portOf_spec hn
  obtain ⟨d, D, dp, idx, hd, hD, hdp, hidx, hmD⟩ := labelsSelf x hRec ht hc hpub ω R' o hacc u hu
  have hN : NamesOnce c :=
    FlockSoundness.Discharge.ZkReg.names_of_acceptsZKR x.I ht hc x.hTags (tr := x.wr (Lv.draw ω) R' o) hacc hd
  obtain ⟨pos, hpos⟩ := FlockSoundness.Discharge.ZkReg.reads_of_acceptsZKR x.I ht hc hpub x.hTags
    (tr := x.wr (Lv.draw ω) R' o) hacc hmem hd hD hdp hidx hmD
  refine ⟨u, q, d, D, dp, idx, rfl, ?_, ?_, hd, hD, hdp, hidx, hmD, ?_⟩
  · rw [hport]; exact hmem
  · rw [hport, hname]; exact find_of_names hN q
  · rw [hport, hpos, hat d D dp idx hD hdp hidx hmD pos hpos]

noncomputable def coveredSession (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) (hO : RegistrantOpens x)
    (hRec : RecordsLiveZKJ x.I x.dj (fun _ _ => J) x.wr) : ZkOuterR mPts nv J Reg' c hU hscope Lv :=
  { x with
    rd := fun ht S₀ d₀ y₀ hc hpub => (hO ht S₀ d₀ y₀ hc hpub).choose.restrict (NamedRow x.own c)
    hown := fun ht S₀ d₀ y₀ hc hpub ω R' o hacc =>
      ownReads_cover x hRec ht y₀ hc hpub (hO ht S₀ d₀ y₀ hc hpub).choose (hO ht S₀ d₀ y₀ hc hpub).choose_spec ω
        R' o hacc }

end Honest

noncomputable def coveredR {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') (hO : ∀ R ω t R₂ j, RegistrantOpens (zr R ω t R₂ j))
    (hCust : Flock.Premises.RecordCustody zr) : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg' :=
  fun R ω t R₂ j => coveredSession (zr R ω t R₂ j) (hO R ω t R₂ j) (hCust R ω t R₂ j).1

theorem registrantCovers_coveredR {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') (hO : ∀ R ω t R₂ j, RegistrantOpens (zr R ω t R₂ j))
    (hCust : Flock.Premises.RecordCustody zr) : Flock.Premises.RegistrantCovers (coveredR zr hO hCust) :=
  fun R ω t R₂ j ht S₀ d₀ y₀ hc hpub u q hn =>
    And.intro ((hO R ω t R₂ j ht S₀ d₀ y₀ hc hpub).choose_spec u q hn).1 ⟨u, q, rfl, hn⟩

/-- `RegistrantCovers` is not vacuous: where the prover's sessions meet the guarantee's other hypotheses and
their registrants hold the rows their reads files name, the same sessions with the registrants' records
(`coveredR`) meet all of them, with the same game. -/
theorem registrantCovers_attained {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') (hO : ∀ R ω t R₂ j, RegistrantOpens (zr R ω t R₂ j))
    (hCust : Flock.Premises.RecordCustody zr) (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {k : ℕ} {ρ : ℝ}
    (hr : ∀ R ω t R₂ j S₀ (d₀ : Lean.Json) (y₀ : ZkHidden.SampleSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀)
      S' R' i, rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    (hT : Typed zr) (dg : ByteArray → ByteArray) (N : ℕ) :
    ∃ zr' : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg',
      Flock.Premises.RegistrantCovers zr' ∧ Flock.Premises.RecordCustody zr' ∧
      (∀ R ω t R₂ j, TagsFound (zr' R ω t R₂ j).I) ∧
      (∀ R ω t R₂ j S₀ (d₀ : Lean.Json)
        (y₀ : ZkHidden.SampleSetupZHJ (inputsJ (zr' R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
        rateZ (ZkHidden.planZHJ (Reg := Reg') (zr' R ω t R₂ j).dj y₀ S' R') i k ≤ ρ) ∧
      Typed zr' ∧ (∀ R ω t, outerVR (zr' R ω t) dg N = outerVR (zr R ω t) dg N) ∧
      ∀ R ω t R₂ j, (zr' R ω t R₂ j).I = (zr R ω t R₂ j).I ∧ (zr' R ω t R₂ j).own = (zr R ω t R₂ j).own ∧
        (zr' R ω t R₂ j).wr = (zr R ω t R₂ j).wr :=
  ⟨coveredR zr hO hCust, registrantCovers_coveredR zr hO hCust, hCust, hF, hr, hT, fun _ _ _ => rfl,
    fun _ _ _ _ _ => ⟨rfl, rfl, rfl⟩⟩

end FlockC3.Compose
