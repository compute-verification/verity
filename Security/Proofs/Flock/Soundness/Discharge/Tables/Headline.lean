import Proofs.Flock.Soundness.Discharge.Tables.Session

/-!
# The headline at the executable, for sessions of `J` tables

`Prog.flock_headline_live_on` (`Exec/LiveOn.lean`) is the headline for one-table plans. Here the plan is any list of
tables at each draw, `plan S R` of length `J`, and the live session is `liveSession` over them.

* `Prog.flock_headline_live_tables`: `Prog.flock_headline_execL` with `hRef` discharged by `execRefines_tables`, at the
  live stratified draw, which flock-verify splits over `J > 1` tables as one draw (`Stmt.setupTables`, `Draw.tablesWhy`:
  several tables need a subset or stratified draw).
* `Prog.flock_headline_live_tables_subset`: the same at the live subset draw `Law.execOS src (.subset kd) n`, whose
  count curve is the uniform `kd`-subset's, `C(n − K, kd) / C(n, kd)` (`Law.subset_miss`), under A3.
* `Prog.flock_headline_live_tables_subset_missX`: the same at rows each under its own encoding (`HmRowsX`), the rows of
  hidden-output tables.

The error term is `miss(K) + ksAvgStrict + linkBoundE`. None of the three grows with `J`: `miss(K)` is the law's;
`ksAvgStrict` averages over draws the supremum over the session's tables of each table's knowledge error, so `J`
tables cost their worst table's, not a sum, at `J` per-table hypotheses (`hKS` at every table `j`); and `linkBoundE`
reads the plan only through the positions the draw reads and `ρ`, a bound on every table's rate.
-/

namespace FlockSoundness.Discharge.Tables

open Game Model Accounting Refine Audit Audit.Partition Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {N' n : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N')) {Reg : Type} {mPts : ℕ}

/-- **The headline at the executable, for sessions of `J` tables of the challenger, at the live stratified
draw.** Let the verifier draw the units as in `Prog.flock_headline_L` (the live stratified draw under A3), let each
draw's session be the plan's tables `plan S R`, and let the live session be `liveSession` over them (the challenger,
its coins uniform per round and drawn after the round's retained bytes, on the `2J` rep streams), written out as a
session directory by `wr`. Against any live prover `τ`, flock-verify accepts with at least `K` wrong units of the
committed values of the simulated prover (`simAuditJ … τ`) with probability at most `miss(K) + ε_ks + δ_link`:
`Prog.flock_headline_execL` with `hRef` discharged by `execRefines_tables`, so `Pr[bad] = 0`. Beyond
`Prog.flock_headline_L`'s (`pp`, `hConst`, `hZero`, `lay`, `rs`, `hHm`, and the assumptions through `hA3`, `hKS`, `hCR`),
it takes `hdec`: the decoder is right on the sessions flock-verify accepts, at the draws the law makes, table by table
(R11b). -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_tables
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs) (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOfJ dg plan N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have h := p.flock_headline_execL outs plan tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst
    zeros hZero lay rs hHm hk hRw hM hρ hr ht K I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
  rwa [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h

/-- The live subset draw's count curve is at most the uniform `kd`-subset's, under A3. -/
theorem execOS_subset_miss_le (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src) (c : ℕ) :
    (Law.execOS (n := n) src (.subset kd) n).miss c ≤ (Law.subset n kd hkd).miss c :=
  iSup₂_mono fun B _ => Law.execOS_subset_escape_le n kd hkd Lb src hA3 B

/-- **The headline at the executable, for sessions of `J` tables of the challenger, at the live subset draw**,
the draw flock-verify takes at more than one table. As `Prog.flock_headline_live_tables`, with the verifier's draw the
live `kd`-subset draw of the `n` units under A3 (`hA3`), whose count curve at `K ≤ n` wrong units is
`C(n − K, kd) / C(n, kd)`. -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_tables_subset
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (hK : K ≤ n) (I : Inputs)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf (Law.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.subset kd) n).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)) τ ≤
      ((n - K).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.subset kd) n)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := flock_e2e_count_hm96L (p.partition outs) H512 enc512 leaf512 plan tab (execAccept := fun o => o.1 = true)
    pp.derived ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := flock_headline_exec_L plan I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) _ hH
  rw [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h
  refine h.trans (add_le_add (add_le_add ?_ le_rfl) le_rfl)
  rw [← Law.subset_miss hkd hK]
  exact execOS_subset_miss_le kd hkd Lb src hA3 K

/-- **`Prog.flock_headline_live_tables_subset` at every `K`**, its count curve the uniform `kd`-subset law's
(`Law.subset n kd hkd`), which is `C(n − K, kd) / C(n, kd)` at `K ≤ n` (`Law.subset_miss`). -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_tables_subset_miss
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf (Law.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.subset kd) n).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)) τ ≤
      (Law.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.subset kd) n)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := flock_e2e_count_hm96L (p.partition outs) H512 enc512 leaf512 plan tab (execAccept := fun o => o.1 = true)
    pp.derived ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := flock_headline_exec_L plan I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) _ hH
  rw [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h
  exact h.trans (add_le_add (add_le_add (execOS_subset_miss_le kd hkd Lb src hA3 K) le_rfl) le_rfl)

/-- `Prog.flock_headline_live_tables_subset_miss` at rows each under its own encoding (`HmRowsX`, the hidden-output
rows), through `flock_e2e_count_hm96LX`. -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_tables_subset_missX
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf (Law.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.subset kd) n).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)) τ ≤
      (Law.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.subset kd) n)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := flock_e2e_count_hm96LX (p.partition outs) H512 enc512 leaf512 plan tab
    (execAccept := fun o => o.1 = true) pp.derived ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := flock_headline_exec_L plan I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) _ hH
  rw [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h
  exact h.trans (add_le_add (add_le_add (execOS_subset_miss_le kd hkd Lb src hA3 K) le_rfl) le_rfl)

end FlockSoundness.Discharge.Tables
