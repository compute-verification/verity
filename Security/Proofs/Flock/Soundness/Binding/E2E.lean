import Proofs.Flock.Soundness.E2E
import Proofs.Flock.Soundness.ExecStratified
import Proofs.Flock.Soundness.Binding.Weights
import Proofs.Flock.Soundness.StrictCR

/-!
# The end-to-end theorem with the value binding of `hm96-sha512/row/v1` rows

`flock_e2e_count` and `flock_e2e_drawn` (`E2E.lean`), and their forms at the executable's draw law
(`ExecStratified.lean`), with `vb : ValueBinding` built from the serving rows (`HmRows.binding`). The binding's hash is
SHA-512 (`Refine.H512`), so `SHA512CRExpected` (`hCR`) is about SHA-512 itself, and what `vb` carried becomes:
* `HmRowComputes` (`hHm`), the circuit's row hashing (phase 2i), the one gap of the binding;
* the layout's facts, in `lay : Layout` (`decode_row`: the decoder reads the rows the witness holds) and `rs : HmRows`
  (`bc_registered`: the `b ‖ c` the witness holds hashes to the registered leaf; `other_registered`: the returned
  outputs' leaves).

`collide_spec` is proved (`ExplicitLeaf.hm96`): the link finder's collisions are explicit SHA-512 collisions. It is asked
of the finders built from the prover bounded (`hCR : LinkCR … (reg σ) (cont σ) …`). The other hypotheses (`dp`, `hConst`,
`hZero`) are the skeleton's, unchanged (`E2E.lean`).

**The two forms of SHA-512's collision resistance.** The link term `δ_link` is under `SHA512CRExpected` (`ecr/sha-512`,
through `hCR`), of SHA-512 itself, since the rows' binding hashes with it. The knowledge term and `δ_tree` are under
`SHA512CRStrict` (`cr/sha-512`, through `hKS` and `hT`), of the session's hash `H`, which `Prog.flock_headline` takes to
be SHA-512; each at its finders' budgets:
* the knowledge term is `ksAvgStrict` (`StrictCR.ksAvgBE_le_strict`), for tables whose rewinding finders satisfy
  `SHA512CRStrict` at budget `qF` and whose self-clash finders at `qS` (`hKS`);
* `δ_tree` is `TreeRewind.bound qT = √(N_s·Q_s·qT²/2^513)` (`DESIGN.md` §1), for a rewinding at the serving roots (`tr`)
  whose finder satisfies `SHA512CRStrict` at budget `qT` (`hT`).

**`δ_tree`.** "Accept" is `execAccept`, read from the registered roots, and the bound counts it with the model's
acceptance at the registered leaves (`Layout.regLeaf`). Where they differ is `δ_tree`: `tr` covers the event that the
executable accepts and the model rejects by runs that open a serving root off its plurality (`TreeRewind.covers`). That
cover is the link from the executable's openings to the model's (R11), still open, so `tr` is a named hypothesis. The
skeleton's `hExec` (every accept is the model's) gives a cover with no positions, at `δ_tree = 0` (`TreeRewind.ofNever`).

**A3.** None of these forms takes A3 (`uniform/io-getrandombytes`). The generic forms hold at every law `L`, the live
draw `Law.execOS` included, and read it only through their terms. The `_exec` forms are at `Law.execStratified`, whose
outcomes are the byte tapes themselves, uniform by construction. The live verifier's bytes enter through `Law.execOS`, and
A3 compares its count curve with `Law.stratified`'s (`Law.execOS_miss_le`): `Prog.flock_headline` composes the two.
-/

namespace FlockSoundness.Binding

open Game Model Accounting Extract LinkFinder Refine Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

/-- An event on the audit's outcome splits on the model's verdict: the part the model accepts, and the part it
rejects. -/
theorem prob_split_accept {n : ℕ} (G : Game (Bool × Finset (Fin n))) (S : Strategy G)
    (A B : Bool × Finset (Fin n) → Prop) :
    prob (fun o => A o ∧ B o) G S ≤
      prob (fun o => (o.1 = true ∧ A o) ∧ B o) G S + prob (fun o => A o ∧ o.1 = false) G S :=
  (prob_mono (p := fun o => A o ∧ B o) (q := fun o => ((o.1 = true ∧ A o) ∧ B o) ∨ (A o ∧ o.1 = false))
    (fun o h => (Bool.eq_false_or_eq_true o.1).elim (fun h1 => Or.inl ⟨⟨h1, h.1⟩, h.2⟩)
      fun h1 => Or.inr ⟨h.1, h1⟩) G S).trans (prob_or_le _ _ G S)

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The end-to-end count curve, with the rows' value binding**: `flock_e2e_count` at `vb := HmRows.binding rs hHm`.
If `execAccept` holds, the probability that at least `K` units of the committed circuit are wrong at the committed values
is at most `miss_L(K) + ε_ks + δ_link + δ_tree`: `ε_ks` and `δ_tree` under `SHA512CRStrict` of `H` (`cr/sha-512`,
through `hKS` and `hT`), `δ_link` under `SHA512CRExpected` of SHA-512 (`ecr/sha-512`, through `hCR`). Beyond the
skeleton's `dp`, `hConst` and `hZero`, it takes the rows' layout (`lay`, `rs`), `HmRowComputes` (`hHm`) and the serving
roots' rewinding (`tr`). It takes no A3: `miss_L` is the law's own, and `Prog.flock_headline` bounds the live draw's by
`Law.stratified`'s under A3. -/
theorem flock_e2e_count_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' + ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_count H E plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

/-- **The end-to-end drawn-unit bound, with the rows' value binding**: `flock_e2e_drawn` at
`vb := HmRows.binding rs hHm`, at most `ε_ks + δ_link + δ_tree`, as in `flock_e2e_count_hm96` (`cr/sha-512` through `hKS`
and `hT`, `ecr/sha-512` through `hCR`). It reads the law only through `ε_ks` and `δ_link`, both at `L`, so it holds as
stated at the live draw (`L := Law.execOS …`), with no A3. -/
theorem flock_e2e_drawn_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) + P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_drawn H E plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht σ hCR).trans
      (add_le_add (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS) le_rfl))
    (tr.le hT))

variable {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

/-- **The e2e drawn-unit bound at the executable's law, with the rows' value binding**: `flock_e2e_drawn_exec` at
`vb := HmRows.binding rs hHm`, at most `ε_ks + δ_link + δ_tree`, as in `flock_e2e_count_hm96` (`cr/sha-512` through `hKS`
and `hT`, `ecr/sha-512` through `hCR`). `Law.execStratified`'s outcomes are uniform byte tapes, so it takes no A3; the live
verifier's bytes are `flock_e2e_drawn_hm96`'s at `L := Law.execOS …`. -/
theorem flock_e2e_drawn_exec_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((Law.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan (Law.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := Law.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_drawn_exec H E plan tab St Ls (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1)
      dp ones hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht σ hCR).trans
      (add_le_add (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS) le_rfl))
    (tr.le hT))

/-- **The e2e count curve at the executable's law, with the rows' value binding**: `flock_e2e_count_exec` at
`vb := HmRows.binding rs hHm`, its count curve bounded by `Law.stratified`'s `miss`, at most
`miss(K) + ε_ks + δ_link + δ_tree`, as in `flock_e2e_count_hm96` (`cr/sha-512` through `hKS` and `hT`, `ecr/sha-512`
through `hCR`). `Law.execStratified`'s outcomes are uniform byte tapes, so it takes no A3; at the live verifier's bytes
the same curve is `Prog.flock_headline`'s, under A3. -/
theorem flock_e2e_count_exec_hm96 (σs : Fin n → Fin m) (ks : Fin m → ℕ)
    (hks : ∀ s, ks s ≤ (Law.stratum σs s).card) (hS : Law.ExecStrata σs ks St)
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((Law.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      (Law.stratified σs ks hks).miss K + StrictCR.ksAvgStrict plan (Law.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := Law.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_count_exec H E plan tab St Ls σs ks hks hS (execAccept := fun o => o.1 = true ∧ execAccept o)
      (fun _ h => h.1) dp ones hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

end FlockSoundness.Binding
