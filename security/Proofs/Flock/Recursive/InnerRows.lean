module

public import Proofs.Flock.Claims.Scope.Tables
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
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
import Mathlib.Tactic.ENatToNat
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
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Claims.Scope.Sizes
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.Hidden.Committed
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Hidden.Unrolled
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Block
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Upv
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.Hm.Site
import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Layout.QSort
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows

@[expose] public section

set_option Elab.async false

/-! A unit's rows read off the inner witness: each row's commit string is the one its table's public file publishes (`rows_commit`), with no premise. -/

namespace Flock.InnerRows

open FlockSoundness FlockSoundness.Discharge Exec Readable

/-- **Bit `y` of block `o`** of table `j`'s share of an inner witness's hidden values (`Readable.zOf`: of `J` tables,
table `j`'s bit `a` is hidden value `a·J + j`). -/
def bitAt (p : Parsed) (w : Claims.Flock.Wit) (j o y : ℕ) : Bool :=
  w.hid.getD ((o * 2 ^ p.sts[j]!.c.kLog + y) * p.sts.size + j) false

def rowBytes (p : Parsed) (w : Claims.Flock.Wit) (j i q : ℕ) : List UInt8 :=
  Hidden.Any.rowMsg (p.sts[j]!.c.ports[q]!)
    (Hm.bytesOf (bitAt p w j (i / p.sts[j]!.c.g)) (2 * (p.sts[j]!.c.ports[q]!).words)
      (Typed.msgCol p.sts[j]!.c (i % p.sts[j]!.c.g) q))

/-- **Row `q` of unit `i` of table `j`'s salt**: the 192 bytes at its `hm96` slot's salt columns (`Hm.saltCol`). -/
def salt (p : Parsed) (w : Claims.Flock.Wit) (j i q : ℕ) : ByteArray :=
  ⟨(Hm.bytesOf (bitAt p w j (i / p.sts[j]!.c.g)) 192 (Hm.saltCol p.sts[j]!.c (i % p.sts[j]!.c.g) q)).toArray⟩

theorem bytesOf_congr {f f' : ℕ → Bool} {N K : ℕ} {col : ℕ → ℕ → ℕ} (hc : ∀ j < N, ∀ t < 8, col j t < K)
    (h : ∀ y < K, f y = f' y) : Hm.bytesOf f N col = Hm.bytesOf f' N col := by
  unfold Hm.bytesOf
  congr 1
  funext j
  congr 1
  funext t
  exact h _ (hc j j.isLt t t.isLt)

theorem bit_toZ : ∀ b : Bool, FlockSoundness.bit (toZ b) = b := by decide

theorem eq_of_data {a b : ByteArray} (h : a.data.toList = b.data.toList) : a = b := by
  cases a
  cases b
  simp only [ByteArray.mk.injEq]
  exact Array.ext' h

theorem commitString_data (r : List UInt8) (salt : ByteArray) :
    Binding.commitString r salt.data.toList =
      (Hm96.Default512.commitString (Sha512.hash ⟨r.toArray⟩) salt).data.toList := by
  show Merkle.xorBytes (Refine.enc512.dig (Refine.H512 r)) (Refine.K512.mask salt.data.toList) ++
      Refine.enc512.dig (Refine.H512 (Refine.K512.saltPrefix ++ salt.data.toList)) = _
  rw [Refine.dig_eq, Refine.dig_eq, Refine.H512_val, Refine.H512_val, Refine.K512_mask, Refine.bytes_of_list,
    ← Refine.xor_range _ _ 64 (Refine.sha512_size _) (Refine.mask_size salt)]
  show _ ++ (Sha512.hash ⟨(Hm96.Default512.saltPrefix.data.toList ++ salt.data.toList).toArray⟩).data.toList = _
  rw [← Refine.data_append Hm96.Default512.saltPrefix salt, Refine.bytes_of_list]
  exact (Refine.data_append _ _).symm

theorem pub_digest_size {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) {i q : ℕ}
    (hi : i < st.pub.n) (hq : q < st.c.ports.size) : (st.pub.digest i q st.c.ports.size).size = 128 := by
  obtain ⟨pop, hpop, hnone, hsome⟩ := (Hidden.setupHidden_spec' h).pub
  have hload := Hidden.loadPublic_nil hpop
  cases draw with
  | none =>
    rw [hnone rfl] at hi ⊢
    exact Hidden.pop_digest_sizeW hload hi hq
  | some d =>
    have hdrawn : HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hsome d rfl
    obtain ⟨D, hD, hpn, hn, -⟩ := Layout.sampled_spec hdrawn
    have hlt := Layout.proved_lt_population hD
    have hr : i < D.proved.size := hn ▸ hi
    exact (congrArg ByteArray.size (Hidden.sampled_digestW hload hdrawn hD hlt hr hq)).trans
      (Hidden.pop_digest_sizeW hload (by rw [← hpn]; exact hlt _ (Array.getElem_mem hr)) hq)

theorem site_commit {T : Model.Statement} (a : HiddenExec.AcceptedH T)
    (hregs : ∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs)
    (hrf : Hidden.Any.RowsFitS a.st.c) {z : Model.Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl))
    {g q : ℕ} (hg : g < a.st.c.g) (hq : q < a.st.c.ports.size)
    (hsz : ∀ g' ≤ g, (Layout.bcChunk a.st.c a.st.pub o.val g' q).size = 128) :
    Binding.commitString (Hidden.Any.rowMsg a.st.c.ports[q]!
        (Hm.bytesOf (Hm.chunkBit T z o) (2 * a.st.c.ports[q]!.words) (Typed.msgCol a.st.c g q)))
      (Hm.bytesOf (Hm.chunkBit T z o) 192 (Hm.saltCol a.st.c g q)) =
      (Layout.bcChunk a.st.c a.st.pub o.val g q).data.toList := by
  obtain ⟨hty, ⟨ls, hls⟩, -⟩ := Hidden.Committed.check a.ht a.committed
  have hcom := HiddenExec.bc_site_commitH a (Hidden.check_upv_pos hty hls) hrf hz o hg hq
  rw [HiddenExec.ofFn_siteH a z o _ _ (fun b _ t ht => Hidden.Committed.msgCol_lt a.ht a.committed hg hq b t ht),
    HiddenExec.bytesAt_siteH a z o 192 _ (fun b hb t ht => Hm.saltCol_lt a.rowRanges hg hq hb ht)] at hcom
  rw [← hcom]
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hregs hrf hg hq hsz hty hls ⊢
  have hg2 := Layout.vu_lt_pow2_of (check_facts hty hls) hg
  obtain ⟨hrsz, hreg⟩ := Hidden.regions_get h.self.regions
  have hpr : q < st.regions.size := hrsz ▸ hq
  exact Hidden.bc_pinned' (st := Hidden.unpin st) L regs hregs hz o hg hg2 (Array.getElem_mem_toList hpr)
    (hreg _ hpr) hsz

theorem unpin_self {st : Stmt} (hda : st.da = (HmOut.delta st.c st.pin).1) (hdb : st.db = (HmOut.delta st.c st.pin).2)
    (hpin : HmRow.pinnedCount st.c.metaJ = 0) : Hidden.unpin st = st := by
  rw [Hidden.hidden_delta_unpinned hpin] at hda hdb
  cases st
  dsimp only at hda hdb
  subst hda hdb
  rfl

theorem model_congr {s s' : Stmt} (e : s' = s) (L : FlockSoundness.Layout s) (L' : FlockSoundness.Layout s')
    (regs : List (Model.Region s.c.kLog (s.m - s.c.kLog)))
    (regs' : List (Model.Region s'.c.kLog (s'.m - s'.c.kLog))) (hr : HEq regs' regs) :
    L'.model regs' = L.model regs := by
  subst e
  cases hr
  rfl

theorem unit_commit {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (hd : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) (hrf : Hidden.Any.RowsFitS st.c)
    (L : FlockSoundness.Layout st) {z : Model.Witness st.m} (hz : (L.model (Refine.regionsOf st)).Satisfies z)
    {i q : ℕ} (hi : i < st.pub.n) (hq : q < st.c.ports.size) {f : ℕ → Bool}
    (hf : ∀ (h : i / st.c.g < 2 ^ (L.model (Refine.regionsOf st)).nbl) (y : ℕ), y < 2 ^ st.c.kLog →
      Hm.chunkBit (L.model (Refine.regionsOf st)) z ⟨i / st.c.g, h⟩ y = f y) :
    Binding.commitString (Hidden.Any.rowMsg st.c.ports[q]!
        (Hm.bytesOf f (2 * st.c.ports[q]!.words) (Typed.msgCol st.c (i % st.c.g) q)))
      (Hm.bytesOf f 192 (Hm.saltCol st.c (i % st.c.g) q)) = (st.pub.digest i q st.c.ports.size).data.toList := by
  have s := Hidden.setupHidden_spec' hd
  have hC := (Hidden.setupHidden_specCC hd).committed
  obtain ⟨hty, ⟨ls, hls⟩, -⟩ := Hidden.Committed.check ht hC
  have hgpos : 0 < st.c.g := Hidden.check_g_pos hty hls
  have hu : Hidden.unpin st = st := unpin_self s.da s.db (Hidden.setupHidden_unpinned ht hd hrf.1.1)
  have hT : (Hidden.unpin_layout ht hd).model (Refine.regionsOf st) = L.model (Refine.regionsOf st) :=
    model_congr hu L _ _ _ HEq.rfl
  have hob : i / st.c.g < 2 ^ (L.model (Refine.regionsOf st)).nbl := by
    show i / st.c.g < 2 ^ (st.m - st.c.kLog)
    rw [s.m, Nat.add_sub_cancel_left, s.nbl, s.blocksReal, s.n]
    exact (Nat.div_lt_iff_lt_mul hgpos).2
      (lt_of_lt_of_le hi ((Integrate.fits_blocks hgpos).trans_eq (Nat.mul_comm _ _)))
  have hg : i % st.c.g < st.c.g := Nat.mod_lt _ hgpos
  have hdm : i / st.c.g * st.c.g + i % st.c.g = i := Nat.div_add_mod' i st.c.g
  have hin : ∀ g' ≤ i % st.c.g, i / st.c.g * st.c.g + g' < st.pub.n := fun g' hg' =>
    lt_of_le_of_lt ((Nat.add_le_add_left hg' _).trans_eq hdm) hi
  have hchunk : ∀ g' ≤ i % st.c.g,
      Layout.bcChunk st.c st.pub (i / st.c.g) g' q = st.pub.digest (i / st.c.g * st.c.g + g') q st.c.ports.size :=
    fun g' hg' => by
      unfold Layout.bcChunk Circuit.instOf
      simp only [hin g' hg', ↓reduceIte]
  have key : Binding.commitString (Hidden.Any.rowMsg st.c.ports[q]!
        (Hm.bytesOf (Hm.chunkBit (L.model (Refine.regionsOf st)) z ⟨i / st.c.g, hob⟩) (2 * st.c.ports[q]!.words)
          (Typed.msgCol st.c (i % st.c.g) q)))
      (Hm.bytesOf (Hm.chunkBit (L.model (Refine.regionsOf st)) z ⟨i / st.c.g, hob⟩) 192
        (Hm.saltCol st.c (i % st.c.g) q)) =
      (Layout.bcChunk st.c st.pub (i / st.c.g) (i % st.c.g) q).data.toList :=
    site_commit ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, Hidden.setupHidden_specCC hd,
        Hidden.unpin_layout ht hd, HiddenExec.unitRowsFlat_unpin ht hd hrf.1.toP, Refine.regionsOf st, hT⟩
      (fun r hr => List.mem_map_of_mem hr) hrf hz ⟨i / st.c.g, hob⟩ hg hq fun g' hg' => by
        show (Layout.bcChunk st.c st.pub (i / st.c.g) g' q).size = 128
        rw [hchunk g' hg']
        exact pub_digest_size hd (hin g' hg') hq
  have hR : Hm.RowRanges st.c := HiddenExec.AcceptedH.rowRanges
    ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, Hidden.setupHidden_specCC hd,
      Hidden.unpin_layout ht hd, HiddenExec.unitRowsFlat_unpin ht hd hrf.1.toP, Refine.regionsOf st, hT⟩
  rw [hchunk _ le_rfl, hdm,
    bytesOf_congr (N := 2 * st.c.ports[q]!.words) (fun b _ t h8 => Hidden.Committed.msgCol_lt ht hC hg hq b t h8)
      (hf hob),
    bytesOf_congr (N := 192) (fun b hb t h8 => Hm.saltCol_lt hR hg hq hb h8) (hf hob)] at key
  exact key

theorem rows_commit_check {p : Parsed} {w : Claims.Flock.Wit} (hP : Parses p) (hc : p.check = .ok ())
    (hh : p.tags.hiddenOutputs = true) (hR : Claims.Flock.RelS (toReadable relation p) w)
    {j i q : ℕ} (hj : j < p.sts.size) (hi : i < p.sts[j].pub.n) (hq : q < p.sts[j].c.ports.size) :
    Hm96.Default512.commitString (Sha512.hash ⟨(rowBytes p w j i q).toArray⟩) (salt p w j i q) =
      p.sts[j].pub.digest i q p.sts[j].c.ports.size := by
  obtain ⟨hRu, c0, -, hc0, -, -, hok⟩ := (check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  have ht : p.tags.typed = false := hRu.2.1
  obtain ⟨spec, hset⟩ := setup_of_tablesOf (tablesOf_of_parses hP)
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt p.tags p.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  obtain ⟨hf, hsat⟩ := (relS_relation p w).1 hR j hj
  have e! : p.sts[j]! = p.sts[j] := getElem!_pos p.sts j hj
  revert hf hsat
  unfold rowBytes salt
  rw [e!]
  generalize hst : p.sts[j] = st at hd hi hq ⊢
  intro hf hsat
  have s := Hidden.setupHidden_spec' hd
  have hcst : HmOut.parseCommit p.tags p.I.circuitFile p.I.tables = .ok st.c :=
    (parseCommit_tagsAt ht _ _ _ _).symm.trans s.parse
  have hc0' : c0 = st.c := by
    unfold ProvedScope.circuitOf at hc0
    rw [hh] at hc0
    simp only [↓reduceIte] at hc0
    exact Except.ok.inj (hc0.symm.trans hcst)
  rw [hc0'] at hok
  have hrows : (Layout.v1Ports st.c || Layout.bitRows st.c) = true := by
    unfold FormOk at hok
    split at hok
    · unfold ProvedScope.scopeOkT at hok
      simp only [Bool.and_eq_true] at hok
      have e : ProvedScope.v2Row = Layout.v2Row := rfl
      have e' : ProvedScope.segRow = Layout.segRow := rfl
      have hr := hok.2
      simp only [ProvedScope.v1Ports, Layout.v1Ports, ProvedScope.v1Rows, Layout.v1Rows, ProvedScope.unpinned,
        Layout.unpinned, ProvedScope.bitRows, Layout.bitRows, ProvedScope.bitRowsU, Layout.bitRowsU, e, e'] at hr ⊢
      exact hr
    · rw [hh] at hok
      simp only [↓reduceIte] at hok
      rw [FailClosed.scopeOkI_eq] at hok
      unfold Layout.scopeOkI at hok
      simp only [Bool.and_eq_true] at hok
      exact hok.2.2
  have hC := (Hidden.setupHidden_specCC hd).committed
  have hrf : Hidden.Any.RowsFitS st.c := HiddenExec.rowsFit_of_committed ht' hC hrows
  have L : FlockSoundness.Layout st :=
    (Hidden.setupHidden_specCC hd).layout ht' (Hidden.setupHidden_unpinned ht' hd hrf.1.1)
  refine eq_of_data ((commitString_data _ _).symm.trans ?_)
  have hk : p.sts[j]!.c.kLog = st.c.kLog := by rw [e!, hst]
  refine unit_commit ht' hd hrf L hsat hi hq fun h y hy => ?_
  refine (Hm.chunkBit_of_lt (L.model (Refine.regionsOf st)) (zOf p.sts.size w j st.m) ⟨i / st.c.g, h⟩ hy).trans ?_
  show FlockSoundness.bit (toZ (w.hid.getD ((i / st.c.g * 2 ^ st.c.kLog + y) * p.sts.size + j) false)) =
    w.hid.getD ((i / st.c.g * 2 ^ p.sts[j]!.c.kLog + y) * p.sts.size + j) false
  rw [bit_toZ, hk]

theorem outRows_le {p : Parsed} (hP : Parses p) (hc : p.check = .ok ()) (hh : p.tags.hiddenOutputs = true) {j : ℕ}
    (hj : j < p.sts.size) : p.sts[j].c.outRowPorts.size ≤ p.sts[j].c.ports.size := by
  obtain ⟨hRu, -⟩ := (check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  have ht : p.tags.typed = false := hRu.2.1
  obtain ⟨spec, hset⟩ := setup_of_tablesOf (tablesOf_of_parses hP)
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨_, hd, -⟩ := hall j hj
  have ht' : (tagsAt p.tags p.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  exact HiddenExec.checkRows_le (Hidden.Committed.check ht' (Hidden.setupHidden_specCC hd).committed).2.2

theorem rows_commit {p : Parsed} {w : Claims.Flock.Wit} (hP : Parses p) (hsc : p.scope = .ok ())
    (hh : p.tags.hiddenOutputs = true) (hR : Claims.Flock.RelS (toReadable relation p) w)
    {j i q : ℕ} (hj : j < p.sts.size) (hi : i < p.sts[j].pub.n) (hq : q < p.sts[j].c.ports.size) :
    Hm96.Default512.commitString (Sha512.hash ⟨(rowBytes p w j i q).toArray⟩) (salt p w j i q) =
      p.sts[j].pub.digest i q p.sts[j].c.ports.size :=
  rows_commit_check hP ((scope_ok_iff p).1 hsc).1 hh hR hj hi hq

end Flock.InnerRows
