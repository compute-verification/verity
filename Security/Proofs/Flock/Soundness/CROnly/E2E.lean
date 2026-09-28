import Proofs.Flock.Soundness.Headline
import Proofs.Flock.Soundness.CROnly.Link

/-!
# The end-to-end theorems under `cr/sha-512` alone

The end-to-end theorems (`E2E.lean`, `Binding/E2E.lean`) and the headline (`Headline.lean`) take `SHA512CRExpected`
(`ecr/sha-512`) for the prover's link finders (`hCR : LinkCR …`). Each is restated here with `hCRs : LinkCRStrict …`
in its place: `SHA512CRStrict` (`cr/sha-512`) for the same finders stopped at `q` SHA-512 evaluations, built from the
prover bounded as `LinkCR`'s are. The bound is the original's at the per-run cost `strictRunCost t' q` in place of `t'`
(`linkCR_of_strict`), and every other hypothesis is the original's, unchanged. So `flock_headline`'s cryptographic
hypotheses are all instances of `cr/sha-512` (`hCRs`, `hKS`, `hT`), beside A3 (`hA3`), which is about the verifier's
byte source.

At the best budget, `q³ = t'·2^512`, the link term's `2t'(1 + k)/2^256` becomes `3(1 + k)·q²/2^512`
(`linkBoundE_at_cube`).

**Scope.** Every theorem here, in `E2EMore` and in `Registered`, is about the non-ZK session (`batchedSession`), whose
verifier checks `final_c` in the clear. None is about `--zk`, whose session model (`Discharge/ZkSession`) receives
`final_c'` before the inner proof, as main's verifier does not until #793.
-/

namespace FlockSoundness.CROnly

open Game Model Accounting Extract LinkFinder Refine Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

section Generic

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The link bound at the best budget**: at `q³ = t'·2^512`, `linkBoundE` at the per-run cost `strictRunCost t' q` is
`Q_s/(1 − ρ)·(3(1 + k)·q²/2^512 + 1/(eM) + k/(eR_w))`. -/
theorem linkBoundE_at_cube {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : P.ValueBinding plan tab decode Hc) {k Rw M : ℕ} {ρ t' q : ℝ} (hq : 0 < q)
    (hcube : q ^ 3 = t' * 2 ^ (512 : ℝ)) :
    P.linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) =
      ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω) *
        (3 * (1 + k) * (q ^ 2 / 2 ^ (512 : ℝ)) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) := by
  unfold linkBoundE
  rw [strictRunCost_at_cube k hq hcube]

/-- **The end-to-end count curve (skeleton) under `cr/sha-512`**: `Partition.flock_e2e_count` with `LinkCRStrict` at
budget `q` (`hCRs`) for `LinkCR`, at `δ_link = linkBoundE` at the per-run cost `strictRunCost t' q`. -/
theorem flock_e2e_count
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        P.linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  P.flock_e2e_count H E plan tab hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict P H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **The end-to-end drawn-unit bound (skeleton) under `cr/sha-512`**: `Partition.flock_e2e_drawn` with `LinkCRStrict`
for `LinkCR`, as in `flock_e2e_count`. -/
theorem flock_e2e_drawn
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  P.flock_e2e_drawn H E plan tab hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le σ
    (linkCR_of_strict P H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **The end-to-end count curve with the rows' value binding, under `cr/sha-512` alone**:
`Binding.flock_e2e_count_hm96` with `LinkCRStrict` for `LinkCR`, so `ε_ks`, `δ_link` and `δ_tree` are all under
`SHA512CRStrict` (`hKS`, `hCRs`, `hT`). -/
theorem flock_e2e_count_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_count_hm96 P H E plan tab dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le K σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT

/-- **The end-to-end drawn-unit bound with the rows' value binding, under `cr/sha-512` alone**:
`Binding.flock_e2e_drawn_hm96` with `LinkCRStrict` for `LinkCR`, as in `flock_e2e_count_hm96`. -/
theorem flock_e2e_drawn_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_drawn_hm96 P H E plan tab dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT

variable {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

/-- **The e2e drawn-unit bound at the executable's law, with the rows' value binding, under `cr/sha-512` alone**:
`Binding.flock_e2e_drawn_exec_hm96` with `LinkCRStrict` for `LinkCR`. -/
theorem flock_e2e_drawn_exec_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((Law.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan (Law.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := Law.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_drawn_exec_hm96 P H E plan tab St Ls dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT

/-- **The e2e count curve at the executable's law, with the rows' value binding, under `cr/sha-512` alone**:
`Binding.flock_e2e_count_exec_hm96` with `LinkCRStrict` for `LinkCR`. -/
theorem flock_e2e_count_exec_hm96 (σs : Fin n → Fin m) (ks : Fin m → ℕ)
    (hks : ∀ s, ks s ≤ (Law.stratum σs s).card) (hS : Law.ExecStrata σs ks St)
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((Law.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      (Law.stratified σs ks hks).miss K + StrictCR.ksAvgStrict plan (Law.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := Law.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_count_exec_hm96 P H E plan tab St Ls σs ks hks hS dp ones hConst zeros hZero lay rs hHm hk hRw
    hM hρ hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs) hKS tr hT

end Generic

section Prog

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

section Session

variable {D : Type} [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {L : Law n}

/-- **The end-to-end count curve on the circuit of the statement's rows, under `cr/sha-512` for the link finders**:
`Prog.flock_e2e_count` with `LinkCRStrict` for `LinkCR`. -/
theorem prog_e2e_count
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.partition outs) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  p.flock_e2e_count outs plan tab H E hExec pp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le
    K σ (linkCR_of_strict (p.partition outs) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

/-- **The end-to-end drawn-unit bound on the circuit of the statement's rows, under `cr/sha-512` for the link
finders**: `Prog.flock_e2e_drawn` with `LinkCRStrict` for `LinkCR`. -/
theorem prog_e2e_drawn
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.partition outs) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.partition outs).wrong
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  p.flock_e2e_drawn outs plan tab H E hExec pp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le
    σ (linkCR_of_strict (p.partition outs) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

end Session

/-- **The headline under `cr/sha-512` alone: the count curve on the statement's rows, under SHA-512, at the live
draw.** `Prog.flock_headline` with `LinkCRStrict` for `LinkCR`: every cryptographic hypothesis is an instance of
`SHA512CRStrict` (`cr/sha-512`), for finders built from the prover bounded: the link finders stopped at `q` (`hCRs`),
the tables' knowledge finders (`hKS`) and the serving roots' rewinding finder (`hT`). The other hypotheses are the
headline's, A3 (`hA3`) among them. The bound is `miss(K) + ε_ks + δ_link + δ_tree`, with `δ_link = linkBoundE` at the
per-run cost `strictRunCost t' q`; at `q³ = t'·2^512` its hash part is `3(1 + k)·q²/2^512` (`linkBoundE_at_cube`). -/
theorem flock_headline
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSession H512 enc512 plan)))
    (hCRs : LinkCRStrict (p.partition outs) H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H512 enc512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStrat H512 enc512 plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H512
      (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H512 qT) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros ((p.partition outs).Xplur H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  p.flock_headline outs plan tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst zeros hZero lay rs hHm hk hRw hM hρ
    hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict (p.partition outs) H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs) hKS tr hT

end Prog

end FlockSoundness.CROnly
