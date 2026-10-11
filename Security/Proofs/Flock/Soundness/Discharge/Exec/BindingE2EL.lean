import Proofs.Flock.Soundness.Discharge.Exec.E2EL
import Proofs.Flock.Soundness.Discharge.Exec.StrictCRL
import Proofs.Flock.Soundness.Binding.E2E
import Proofs.Flock.Soundness.Binding.RowsX

/-!
# The end-to-end theorem with the value binding of `hm96-sha512/row/v1` rows

`flock_e2e_countL` and `flock_e2e_drawnL` (`E2E.lean`), and their forms at the executable's draw law
(`ExecStratified.lean`), with `vb : ValueBinding` built from the serving rows (`HmRows.binding`). The binding's hash is
SHA-512 (`Refine.H512`), so `SHA512CRExpected` (`hCR`) is about SHA-512 itself, and what `vb` carried becomes:
* `HmRowComputes` (`hHm`), the circuit's row hashing (phase 2i), the one gap of the binding;
* the layout's facts, in `lay : Layout` (`decode_row`: the decoder reads the rows the witness holds) and `rs : HmRows`
  (`bc_registered`: the `b ‖ c` the witness holds hashes to the registered leaf; `other_registered`: the returned
  outputs' leaves).

`collide_spec` is proved (`ExplicitLeaf.hm96`): the link finder's collisions are explicit SHA-512 collisions. It is asked
of the finders built from the prover bounded (`hCR : LinkCRL … (reg σ) (cont σ) …`). The other hypotheses (`dp`, `hConst`,
`hZero`) are the skeleton's, unchanged (`assumption-notes/e2e-checklist.md`).

**The two forms of SHA-512's collision resistance.** The link term `δ_link` is under `SHA512CRExpected` (`ecr/sha-512`,
through `hCR`), of SHA-512 itself, since the rows' binding hashes with it. The knowledge term and `δ_tree` are under
`SHA512CRStrict` (`cr/sha-512`, through `hKS` and `hT`), of the session's hash `H`, which `Prog.flock_headline` takes to
be SHA-512; each at its finders' budgets:
* the knowledge term is `ksAvgStrict` (`ksAvgBE_le_strictL`), for tables whose rewinding finders satisfy
  `SHA512CRStrict` at budget `qF` and whose self-clash finders at `qS` (`hKS`);
* `δ_tree` is `TreeRewind.bound qT = √(N_s·Q_s·qT²/2^513)` (`DESIGN.md` §1), for a rewinding at the serving roots (`tr`)
  whose finder satisfies `SHA512CRStrict` at budget `qT` (`hT`).

**`δ_tree`.** "Accept" is `execAccept`, read from the registered roots, and the bound counts it with the Lean
protocol's acceptance at the registered leaves (`Layout.regLeaf`). Where they differ is `δ_tree`: `tr` covers the event
that the executable accepts and the Lean protocol rejects by runs that open a serving root off its plurality
(`TreeRewind.covers`). That cover is the link from the executable's openings to the Lean protocol's (R11), still open,
so `tr` is a named hypothesis. The skeleton's `hExec` (every accept is the Lean protocol's) gives a cover with no
positions, at `δ_tree = 0` (`TreeRewind.ofNever`).

**A3.** None of these forms takes A3 (`uniform/io-getrandombytes`). The generic forms hold at every law `L`, the live
draw `Law.execOS` included, and read it only through their terms. The `_exec` forms are at `Law.execStratified`, whose
outcomes are the byte tapes themselves, uniform by construction. The live verifier's bytes enter through `Law.execOS`, and
A3 compares its count curve with `Law.stratified`'s (`Law.execOS_miss_le`): `Prog.flock_headline` composes the two.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Refine Binding Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The end-to-end count curve, with the rows' value binding**: `flock_e2e_countL` at `vb := HmRows.binding rs hHm`.
If `execAccept` holds, the probability that at least `K` units of the committed circuit are wrong at the committed values
is at most `miss_L(K) + ε_ks + δ_link + δ_tree`: `ε_ks` and `δ_tree` under `SHA512CRStrict` of `H` (`cr/sha-512`,
through `hKS` and `hT`), `δ_link` under `SHA512CRExpected` of SHA-512 (`ecr/sha-512`, through `hCR`). Beyond the
skeleton's `dp`, `hConst` and `hZero`, it takes the rows' layout (`lay`, `rs`), `HmRowComputes` (`hHm`) and the serving
roots' rewinding (`tr`). It takes no A3: `miss_L` is the law's own, and `Prog.flock_headline` bounds the live draw's by
`Law.stratified`'s under A3. -/
theorem flock_e2e_count_hm96L
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H E Lf (plan (L.draw ω) (reg σ)) j
      (batchStratL H E Lf plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSessionL H E Lf plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (XplurL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' + ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((flock_e2e_countL P H E Lf plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (ksAvgBE_le_strictL H E Lf plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

/-- `flock_e2e_count_hm96L` at rows each under its own encoding (`HmRowsX`), `vb := HmRowsX.binding rs hHm`. -/
theorem flock_e2e_count_hm96LX
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H E Lf (plan (L.draw ω) (reg σ)) j
      (batchStratL H E Lf plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSessionL H E Lf plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (XplurL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' + ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((flock_e2e_countL P H E Lf plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (ksAvgBE_le_strictL H E Lf plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

/-- **The end-to-end drawn-unit bound, with the rows' value binding**: `flock_e2e_drawnL` at
`vb := HmRows.binding rs hHm`, at most `ε_ks + δ_link + δ_tree`, as in `flock_e2e_count_hm96L` (`cr/sha-512` through `hKS`
and `hT`, `ecr/sha-512` through `hCR`). It reads the law only through `ε_ks` and `δ_link`, both at `L`, so it holds as
stated at the live draw (`L := Law.execOS …`), with no A3. -/
theorem flock_e2e_drawn_hm96L
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H E Lf (plan (L.draw ω) (reg σ)) j
      (batchStratL H E Lf plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSessionL H E Lf plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (XplurL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) + P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((flock_e2e_drawnL P H E Lf plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht σ hCR).trans
      (add_le_add (ksAvgBE_le_strictL H E Lf plan k (reg σ) (cont σ) hKS) le_rfl))
    (tr.le hT))

end FlockSoundness.Discharge.Exec
