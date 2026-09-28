import Proofs.Flock.Soundness.Discharge.ZkBind.Headline
import Proofs.Flock.Soundness.Discharge.ZkBind.Shadow
import Proofs.Flock.Soundness.Binding.RowsX

/-!
# `--zk` rows each under its own encoding

`HmRowsZ`, `HmRowsZ.ofShadow` and `zk_flock_countC_execOS_keyProg` for rows whose bytes need not be injective
(`Binding.HmRowsX`, so that `sha512/row-seg/v1` rows sit beside `/row/v1` and `/row/v2` ones):

* `HmRowsZX`: a `--zk` layout's rows, each commit string's bytes `enc p` with an extractor (`ext_spec`), its scheme
  `hm96X` at the rows, `HmRowsZX.Computes` and the registration fact (`HmRowsZX.registered`), and the value binding
  (`HmRowsZX.bindingZ`);
* `HmRowsZX.ofShadow`: the `--zk` rows of a public layout's `HmRowsX`, and `HmRowsZX.ofShadow_computes`;
* `zk_flock_countC_execOS_keyProgX`: the compiled `--zk` count curve at such rows' binding.
-/

namespace FlockSoundness.Discharge.ZkBind

open Game Model Accounting Refine Audit ZkSession ZkLink Binding

section Rows

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  {A : Arith F K} {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type}
  {planZ : Finset (Fin n) → Reg → List (TabZK F mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length}
  {decode : DecoderZK (C := C) planZ tab}

/-- **A `--zk` layout's rows, each under its own encoding** (`Binding.HmRowsX`' fields over `TabZK` plans). -/
structure HmRowsZX (lay : LayoutZ A P planZ tab decode) where
  isRow : lay.Pos → Prop
  [decRow : DecidablePred isRow]
  enc : lay.Pos → lay.Val → List UInt8
  ext : lay.Pos → lay.Val → lay.Val → List UInt8 × List UInt8
  ext_spec : ∀ p v v', v ≠ v' → enc p v = enc p v' → Collides (ext p v v')
  other : lay.Pos → ExplicitLeaf lay.Val
  bcAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → lay.Pos → Fin 128 → Fin 8 →
    Fin (2 ^ (tabOfZ planZ tab S R u).S.m)
  bc_registered : ∀ S R u, u ∈ S → ∀ M, SatZ A planZ tab S R u M → ∀ g ∈ P.io u, isRow (lay.pos g) →
    H512 (K512.leafPrefix ++ bytesAt (witZ A planZ tab S R u M) (bcAt S R u (lay.pos g))) = lay.regLeaf R (lay.pos g)
  other_registered : ∀ S R u, u ∈ S → ∀ M, SatZ A planZ tab S R u M → ∀ g ∈ P.io u, ¬ isRow (lay.pos g) →
    (other (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)

attribute [instance] HmRowsZX.decRow

namespace HmRowsZX

variable {lay : LayoutZ A P planZ tab decode} (rs : HmRowsZX lay)

/-- Each commit string's leaf scheme: `hm96X` over its encoding at a row, `other` elsewhere. -/
def scheme (p : lay.Pos) : ExplicitLeaf lay.Val :=
  if rs.isRow p then ExplicitLeaf.hm96X (rs.enc p) (rs.ext p) (rs.ext_spec p) else rs.other p

/-- **`HmRowComputes` at the drawn units' rows of `--zk` tables**, each row under its own encoding. -/
def Computes : Prop :=
  Assumptions.HmRowComputes (fun l => digBytes (H512 l)) K512.saltPrefix K512.mask
    (fun w : RowSiteZ planZ tab C.N =>
      w.u ∈ w.S ∧ SatZ A planZ tab w.S w.R w.u w.M ∧ w.g ∈ P.io w.u ∧ rs.isRow (lay.pos w.g))
    (fun w => rs.enc (lay.pos w.g) (lay.row w.S w.R w.u w.M (lay.pos w.g)))
    (fun w => lay.salt w.S w.R w.u w.M (lay.pos w.g))
    (fun w => bytesAt (witZ A planZ tab w.S w.R w.u w.M) (rs.bcAt w.S w.R w.u (lay.pos w.g)))

omit [Fintype K] [DecidableEq K] in
/-- **The registration fact from `HmRowComputes`**, for `--zk` tables, each row under its own encoding. -/
theorem registered (hHm : rs.Computes) : lay.Registered rs.scheme := by
  intro S R u hu M hM g hg
  unfold scheme
  split_ifs with hrow
  · have hc : bytesAt (witZ A planZ tab S R u M) (rs.bcAt S R u (lay.pos g)) =
        commitString (rs.enc (lay.pos g) (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g)) :=
      hHm ⟨S, R, u, M, g⟩ ⟨hu, hM, hg, hrow⟩
    have hb := rs.bc_registered S R u hu M hM g hg hrow
    rw [hc] at hb
    exact hb
  · exact rs.other_registered S R u hu M hM g hg hrow

/-- **The value binding of a `--zk` layout's rows, each under its own encoding.** -/
noncomputable def bindingZ (hHm : rs.Computes) : ValueBindingZ A P planZ tab decode H512 :=
  lay.bindingZ rs.scheme (rs.registered hHm)

end HmRowsZX

end Rows

/-! ## From a public layout's rows -/

section Shadow

local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length}
  {planS : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tabS : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planS S R).length}
  (e : ∀ S R u, (tabOfZ planZ tab S R u).S = (Partition.tabOf planS tabS S R u).S)
  {decZ : DecoderZK (C := C) planZ tab} {decS : Partition.DecoderB (C := C) (Reg := Reg) planS tabS}
  {lay : Binding.Layout P planS tabS decS} (hdec : DecAgree e (decZ := decZ) (decS := decS) (P := P))
  (rs : Binding.HmRowsX lay)

/-- **The `--zk` rows of a public layout's rows, each under its own encoding.** -/
noncomputable def HmRowsZX.ofShadow : HmRowsZX (LayoutZ.ofShadow e lay hdec) where
  isRow := rs.isRow
  decRow := rs.decRow
  enc := rs.enc
  ext := rs.ext
  ext_spec := rs.ext_spec
  other := rs.other
  bcAt S R u p j t := adrOf (e S R u) (rs.bcAt S R u p j t)
  bc_registered S R u hu M' hM' g hg hrow := by
    obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
    have h := rs.bc_registered S R u hu M hS g hg hrow
    rw [hM] at h
    exact h
  other_registered S R u hu M' hM' g hg hrow := by
    obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
    obtain ⟨hr, hs⟩ := row_ofShadow e lay hdec S R u M' M hM (lay.pos g)
    show (rs.other (lay.pos g)).leaf ((LayoutZ.ofShadow e lay hdec).row S R u M' (lay.pos g))
      ((LayoutZ.ofShadow e lay hdec).salt S R u M' (lay.pos g)) = lay.regLeaf R (lay.pos g)
    rw [hr, hs]
    exact rs.other_registered S R u hu M hS g hg hrow

/-- **`HmRowComputes` moves along**, each row under its own encoding. -/
theorem HmRowsZX.ofShadow_computes (hHm : rs.Computes) : (HmRowsZX.ofShadow e hdec rs).Computes := by
  rintro ⟨S, R, u, M', g⟩ ⟨hu, hM', hg, hrow⟩
  obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
  obtain ⟨hr, hs⟩ := row_ofShadow e lay hdec S R u M' M hM (lay.pos g)
  have h := hHm ⟨S, R, u, M, g⟩ ⟨hu, hS, hg, hrow⟩
  simp only at h
  rw [← hr, ← hs, hM] at h
  exact h

end Shadow

/-! ## The compiled `--zk` count curve at such rows -/

section Curve

open StrictCR Placed
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith F K) (Hs : Hashes F D Sl) {n : ℕ} {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **`zk_flock_countC_execOS_keyProg` at rows each under its own encoding** (`HmRowsZX`). -/
theorem zk_flock_countC_execOS_keyProgX (hA : A.Correct) {c : Flock.Circuit} (hU : FlockSoundness.Built c.unit)
    (si : SitesZ planZ tab)
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      Flock.HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n)))
    (lay : LayoutZ A ((keyProg c hU n).partition outs) planZ tab
      (decodeOfPlace A ((keyProg c hU n).partition outs) planZ tab (si.placeKey hU hparse outs)))
    (rs : HmRowsZX lay) (hHm : rs.Computes)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition outs).wrong
        (XplurZC A Hs ((keyProg c hU n).partition outs) planZ tab (rs.bindingZ hHm) (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ planZ (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC A Hs ((keyProg c hU n).partition outs) planZ tab (rs.bindingZ hHm) k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_cr A Hs ((keyProg c hU n).partition outs) planZ tab hA σs ks hks St hS Kd pop Lb src hA3
    (si.placeKey hU hparse outs) (rs.bindingZ hHm) hk hRw hM hρ hr ht K σ hCR tr hT

end Curve

end FlockSoundness.Discharge.ZkBind
