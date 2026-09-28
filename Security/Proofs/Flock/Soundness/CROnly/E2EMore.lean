import Proofs.Flock.Soundness.CROnly.Link
import Proofs.Flock.Soundness.ExecStratified
import Proofs.Flock.Soundness.Types.ProgramE2E

/-!
# The executable-law and program-of-units e2e forms under `cr/sha-512` alone

`Partition.flock_e2e_drawn_exec`, `flock_e2e_count_exec`, `UProg.flock_e2e_count`, `UProg.flock_e2e_drawn` and the four
`UProg.flock_e2e_*_classes` forms, restated as in `CROnly/E2E.lean`: `LinkCRStrict` at budget `q` in place of `LinkCR`, and
`strictRunCost t' q` in place of `t'`.
-/

namespace FlockSoundness.CROnly

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

section Exec

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

/-- **`Partition.flock_e2e_drawn_exec` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at `δ_link = linkBoundE`
at the per-run cost `strictRunCost t' q`. -/
theorem flock_e2e_drawn_exec
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := Law.execStratified St Ls) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  P.flock_e2e_drawn_exec H E plan tab St Ls hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le σ
    (linkCR_of_strict P H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **`Partition.flock_e2e_count_exec` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at `δ_link = linkBoundE`
at the per-run cost `strictRunCost t' q`. -/
theorem flock_e2e_count_exec (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (hS : Law.ExecStrata σs ks St)
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      (Law.stratified σs ks hks).miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        P.linkBoundE (L := Law.execStratified St Ls) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  P.flock_e2e_count_exec H E plan tab St Ls σs ks hks hS hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le K σ (linkCR_of_strict P H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

end Exec

section UProg

variable {kIn one : ℕ} (hone : one < kIn) {outs : List ℕ} (p : FlockSoundness.UProg kIn one outs) {Reg : Type}
  {D : Type} [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
  {L : Law p.n}

/-- **`UProg.flock_e2e_count` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at
`δ_link = linkBoundE` at the per-run cost `strictRunCost t' q`. -/
theorem uprog_e2e_count
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin p.n → FlockSoundness.Rows} (dp : (p.P hone).DerivedPlaces plan tab unitRows)
    (hConst : (p.P hone).ConstCols plan tab dp.place {p.oneOf hone})
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab ((p.P hone).placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.P hone) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  UProg.flock_e2e_count hone p H E plan tab hExec dp hConst vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict (p.P hone) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **`UProg.flock_e2e_drawn` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at
`δ_link = linkBoundE` at the per-run cost `strictRunCost t' q`. -/
theorem uprog_e2e_drawn
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin p.n → FlockSoundness.Rows} (dp : (p.P hone).DerivedPlaces plan tab unitRows)
    (hConst : (p.P hone).ConstCols plan tab dp.place {p.oneOf hone})
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab ((p.P hone).placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.P hone) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  UProg.flock_e2e_drawn hone p H E plan tab hExec dp hConst vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le σ
    (linkCR_of_strict (p.P hone) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **`UProg.flock_e2e_count_classes` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at `δ_link = linkBoundE`
at the per-run cost `strictRunCost t' q`. -/
theorem uprog_e2e_count_classes
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (Audit.Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.P hone) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  UProg.flock_e2e_count_classes hone p H E plan tab hExec cls slot blk hrd hcp vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict (p.P hone) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **`UProg.flock_e2e_drawn_classes` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at `δ_link = linkBoundE`
at the per-run cost `strictRunCost t' q`. -/
theorem uprog_e2e_drawn_classes
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (Audit.Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.P hone) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  UProg.flock_e2e_drawn_classes hone p H E plan tab hExec cls slot blk hrd hcp vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le σ
    (linkCR_of_strict (p.P hone) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **`UProg.flock_e2e_count_classes_zero` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at `δ_link = linkBoundE`
at the per-run cost `strictRunCost t' q`. -/
theorem uprog_e2e_count_classes_zero
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (Audit.Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < kIn) (hz1 : zer ≠ one)
    (hzr : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hone u).t.inBits → p.srcOf u i = .input zer →
      (cls S R u h).copyPos (slot S R u h) i ∈ (cls S R u h).zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.P hone) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).wrong (Xpub {p.oneOf hone} {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩}
          ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  UProg.flock_e2e_count_classes_zero hone p H E plan tab hExec cls slot blk hrd hcp hzk hz1 hzr vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict (p.P hone) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **`UProg.flock_e2e_drawn_classes_zero` under `cr/sha-512`**: `LinkCRStrict` at budget `q` (`hCRs`) for `LinkCR`, at `δ_link = linkBoundE`
at the per-run cost `strictRunCost t' q`. -/
theorem uprog_e2e_drawn_classes_zero
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (Audit.Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < kIn) (hz1 : zer ≠ one)
    (hzr : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hone u).t.inBits → p.srcOf u i = .input zer →
      (cls S R u h).copyPos (slot S R u h) i ∈ (cls S R u h).zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.P hone) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).wrong (Xpub {p.oneOf hone} {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩}
          ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  UProg.flock_e2e_drawn_classes_zero hone p H E plan tab hExec cls slot blk hrd hcp hzk hz1 hzr vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le σ
    (linkCR_of_strict (p.P hone) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

end UProg

end FlockSoundness.CROnly
