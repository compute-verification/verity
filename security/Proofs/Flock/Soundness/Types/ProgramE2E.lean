module

public import Proofs.Flock.Soundness.Types.ProgramPlaces
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-! The end-to-end theorems on a program of units carried to its Boolean circuit; `hExec`, `dp` and `vb` stay
premises, and the `_classes` forms leave the statement's zero at the plurality, so they say nothing about the
zero padding. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

theorem _root_.FlockSoundness.UProg.toRows_unit_of_lt {k one : ℕ} (hk : one < k) :
    ∀ {outs : List ℕ} (p : FlockSoundness.UProg k one outs) (g : Fin p.rowsN), g.val < k → (p.toRows hk).unit g = none
  | _, .inputs, _, _ => rfl
  | _, .snoc p s src valid, g, hg => by
    have h' : g.val < p.rowsN := lt_of_lt_of_le hg p.k_le_rowsN
    have ih := FlockSoundness.UProg.toRows_unit_of_lt hk p ⟨g.val, h'⟩ hg
    change (Prog.snoc (p.toRows hk) s.R (p.insOf s src valid) (p.oneOf hk) (p.insOf_ne hk s src valid)).unit g = none
    simp only [Prog.unit, h', ↓reduceDIte, ih, Option.map_none]

namespace Partition

variable {kIn one : ℕ} (hone : one < kIn) {outs : List ℕ} (p : FlockSoundness.UProg kIn one outs) {Reg : Type} {D : Type}
  [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
  {L : SamplingDesign p.n}

theorem _root_.FlockSoundness.UProg.flock_e2e_count
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin p.n → FlockSoundness.Rows} (dp : (p.P hone).DerivedPlaces plan tab unitRows)
    (hConst : (p.P hone).ConstCols plan tab dp.place {p.oneOf hone})
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab ((p.P hone).placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).incorrect
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun _ h => ⟨h.1, h.2.trans (Finset.card_le_card (p.rowsL1 hone _ fun _ hg => Xpub_of_mem_ones hg))⟩) _
    σ).trans
    ((p.P hone).flock_e2e_count H E plan tab hExec dp {p.oneOf hone} hConst ∅
      (fun _ _ _ _ _ h => absurd h (Finset.notMem_empty _)) vb hk hRw hM hρ hr ht K σ hCR)

theorem _root_.FlockSoundness.UProg.flock_e2e_sampled
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin p.n → FlockSoundness.Rows} (dp : (p.P hone).DerivedPlaces plan tab unitRows)
    (hConst : (p.P hone).ConstCols plan tab dp.place {p.oneOf hone})
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab ((p.P hone).placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).incorrect
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun _ h => ⟨h.1, fun hd => h.2 (Finset.disjoint_of_subset_left
    (p.rowsL1 hone _ fun _ hg => Xpub_of_mem_ones hg) hd)⟩) _ σ).trans
    ((p.P hone).flock_e2e_sampled H E plan tab hExec dp {p.oneOf hone} hConst ∅
      (fun _ _ _ _ _ h => absurd h (Finset.notMem_empty _)) vb hk hRw hM hρ hr ht σ hCR)

theorem _root_.FlockSoundness.UProg.constCols_of_classes
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u)) :
    (p.P hone).ConstCols plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place
      {p.oneOf hone} :=
  (p.P hone).constCols_singleton plan tab _ _ fun _ _ u _ => Fin.ext (p.pairs_oneGate hone u)

theorem _root_.FlockSoundness.UProg.zeroCols_of_classes
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < kIn) (hz1 : zer ≠ one)
    (hzr : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hone u).t.inBits → p.srcOf u i = .input zer →
      (cls S R u h).copyPos (slot S R u h) i ∈ (cls S R u h).zeros) :
    (p.P hone).ZeroCols plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place
      {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩} := by
  intro S R u h c hc z hz
  change Fin (p.rowsOf hone u).w at c
  rw [Finset.mem_singleton] at hc
  change (p.instOf hone u).wire c = _ at hc
  have hi : (p.rowsOf hone u).nIn = (p.spec hone u).t.inBits := (p.spec hone u).nIn_eq
  have hn : (p.rowsOf hone u).nIn = (cls S R u h).rows.nIn :=
    congrArg Rows.nIn (UnitSpec.R_eq_of_reads (hrd S R u h))
  by_cases hc1 : c.val < (p.spec hone u).t.inBits
  · have hv := congrArg Fin.val hc
    rw [p.instOf_wire hone u c hc1] at hv
    have hs : p.srcOf u c.val = .input zer := p.srcGate_inj (p.srcOf_valid hone u hc1) ⟨hzk, hz1⟩ hv
    have hcol := RowsAt.cast_col (UnitSpec.R_eq_of_reads (hrd S R u h)).symm
      ((cls S R u h).atSlot (slot S R u h) (blk S R u h)) c
    change (tabOf plan tab S R u).S.block z
        (((cls S R u h).atSlot (slot S R u h) (blk S R u h)).cast (UnitSpec.R_eq_of_reads (hrd S R u h)).symm).o
        ((((cls S R u h).atSlot (slot S R u h) (blk S R u h)).cast (UnitSpec.R_eq_of_reads (hrd S R u h)).symm).col c) = 0
    rw [RowsAt.cast_o, hcol]
    change (tabOf plan tab S R u).S.block z (blk S R u h) ((cls S R u h).col (slot S R u h) _) = 0
    rw [(cls S R u h).copy (slot S R u h) _ (by rw [Fin.val_cast, ← hn, hi]; exact hc1) z hz (blk S R u h),
      Fin.val_cast]
    exact (cls S R u h).zero _ (hzr S R u h c.val hc1 hs) z hz (blk S R u h)
  · exfalso
    have hw : c.val < (p.rowsOf hone u).nIn + (p.rowsOf hone u).nComp + 1 := c.isLt
    by_cases hc2 : c.val = (p.rowsOf hone u).nIn + (p.rowsOf hone u).nComp
    · have hco : c = (p.rowsOf hone u).one := Fin.ext hc2
      subst hco
      have h1 := congrArg Fin.val hc
      have h2 : ((p.instOf hone u).wire (p.rowsOf hone u).one).val = one := p.pairs_oneGate hone u
      exact hz1 (h1.symm.trans h2)
    · have hci : (p.rowsOf hone u).comp ⟨c.val - (p.rowsOf hone u).nIn, by omega⟩ = c :=
        Fin.ext (show (p.rowsOf hone u).nIn + (c.val - (p.rowsOf hone u).nIn) = c.val by omega)
      have hm := (p.instOf hone u).mem ⟨c.val - (p.rowsOf hone u).nIn, by omega⟩
      rw [hci, hc] at hm
      change (p.toRows hone).unit ⟨zer, _⟩ = some u at hm
      rw [UProg.toRows_unit_of_lt hone p ⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩ hzk] at hm
      cases hm

theorem _root_.FlockSoundness.UProg.flock_e2e_count_classes
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).incorrect
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  UProg.flock_e2e_count hone p H E plan tab hExec (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp)
    (p.constCols_of_classes hone plan tab cls slot blk hrd hcp) vb hk hRw hM hρ hr ht K σ hCR

theorem _root_.FlockSoundness.UProg.flock_e2e_sampled_classes
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).incorrect
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  UProg.flock_e2e_sampled hone p H E plan tab hExec (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp)
    (p.constCols_of_classes hone plan tab cls slot blk hrd hcp) vb hk hRw hM hρ hr ht σ hCR

theorem _root_.FlockSoundness.UProg.flock_e2e_count_classes_zero
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < kIn) (hz1 : zer ≠ one)
    (hzr : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hone u).t.inBits → p.srcOf u i = .input zer →
      (cls S R u h).copyPos (slot S R u h) i ∈ (cls S R u h).zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).incorrect (Xpub {p.oneOf hone} {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩}
          ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun _ h => ⟨h.1, h.2.trans (Finset.card_le_card (p.rowsL1 hone _ fun _ hg => Xpub_of_mem_ones hg))⟩) _
    σ).trans
    ((p.P hone).flock_e2e_count H E plan tab hExec (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp)
      {p.oneOf hone} (p.constCols_of_classes hone plan tab cls slot blk hrd hcp) _
      (p.zeroCols_of_classes hone plan tab cls slot blk hrd hcp hzk hz1 hzr) vb hk hRw hM hρ hr ht K σ hCR)

theorem _root_.FlockSoundness.UProg.flock_e2e_sampled_classes_zero
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < kIn) (hz1 : zer ≠ one)
    (hzr : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hone u).t.inBits → p.srcOf u i = .input zer →
      (cls S R u h).copyPos (slot S R u h) i ∈ (cls S R u h).zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).incorrect (Xpub {p.oneOf hone} {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩}
          ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun _ h => ⟨h.1, fun hd => h.2 (Finset.disjoint_of_subset_left
    (p.rowsL1 hone _ fun _ hg => Xpub_of_mem_ones hg) hd)⟩) _ σ).trans
    ((p.P hone).flock_e2e_sampled H E plan tab hExec (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp)
      {p.oneOf hone} (p.constCols_of_classes hone plan tab cls slot blk hrd hcp) _
      (p.zeroCols_of_classes hone plan tab cls slot blk hrd hcp hzk hz1 hzr) vb hk hRw hM hρ hr ht σ hCR)

end Partition

end FlockSoundness.Audit
