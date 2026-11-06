import Proofs.Flock.Soundness.Discharge.Tables.Headline
import Proofs.Flock.Soundness.Discharge.Tables.Accepts

/-!
# Sessions of `J` tables: the statements

The open guarantee is sessions of more than one table (`--session-tables J`, `J > 1`). Each statement is a `Prop`
here, and its theorem is below it.

* `SessionTransfer`: the protocol-level transfer at `J` tables. Against any live prover of `liveSession` (one `Commit`
  of `J` roots, one link-point draw every table reads, the `2J` rep streams in any order the prover picks with each
  round's coins drawn after its bytes, then every proof file), the live verdict holds with probability at most the
  model session's acceptance of every table (`modelSession`, the tables concurrently through `batch`) against the
  simulated prover `simSession`, whenever every table's decoder is right (`DecodesTable`, one per table).
* `SessionBLModel`: the salted session on the tables `ts` is `modelSession`'s.
* `ExecRefinesTables`: R11 for any plan, `ExecRefines … (LiveAccepts I) ⊥ (audit … (batchedSessionL … plan)) (simAuditJ …)`
  given `DecodesOnJ`.
* `HeadlineTables`: `Prog.flock_headline_live_on` with `plan S R` of any length `J`, at the live stratified draw.
* `HeadlineTablesSubset`: the same at the live subset draw, with count curve `C(n − K, kd) / C(n, kd)`; flock-verify
  takes no other draw at `J > 1`.
* `AcceptsTables`: flock-verify's acceptance at `count > 1`: the record's draw, when there is one, is a subset draw;
  the session's units (the draw's, without one the population's) split into `count` equal parts; and table `j` is
  `setupH`'s statement under `Tags.tableIdentity count j` at the subset draw `partDraw` of part `j`.

The decoder (`DecodesTable`, inside `DecodesOnJ`) is a named hypothesis per draw, registration and table: at one table
it is `DecodesOn`'s, which `decodesOn_exec` (`Discharge/DecodeZL`) discharges for the plain headline's tables, and its
coin half is `decodesOnOfRuns`'s.
-/

namespace FlockSoundness.Discharge.Tables

open Game Model Accounting Refine Audit Audit.Partition Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- **The transfer at `J` tables.** -/
def SessionTransfer : Prop :=
  ∀ {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
    (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp) (N : ℕ)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (hinh : ∀ j pts r, AllInh (G j pts r))
    (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j) (Pr : Strategy (liveSession dg J mPts N)),
    prob pL (liveSession dg J mPts N) Pr ≤
      prob (AllOk ok dflt) (modelSession out G)
        ((simSession out ok dflt pL opn dg dec one fin capOf N G hinh hdec).strategy Pr)

theorem session_transfer : SessionTransfer :=
  fun out ok dflt pL opn dg dec one fin capOf N G hinh hdec Pr =>
    live_le_session out ok dflt pL opn dg dec one fin capOf N G hinh hdec Pr

/-- **The salted session on `J` tables is `modelSession`'s.** -/
def SessionBLModel : Prop :=
  ∀ {K D Sl : Type} [Field K] [DecidableEq K] {mPts : ℕ} (A : Arith 𝔽 K) (H : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) (ts : List (TabSpec mPts)),
    modelSession (J := ts.length) (fun _ cap pend os => tableOutL H E Lf cap pend os)
      (fun j => repsOf (D := D) A (ts.get j)) = sessionBL A H E Lf ts

theorem sessionBL_model : SessionBLModel :=
  fun A H E Lf ts => sessionBL_eq_modelSession A H E Lf ts

/-- **R11 for sessions of `J` tables.** -/
def ExecRefinesTables : Prop :=
  ∀ {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type} (H : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (I : Inputs) (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 D × Salts Sl)
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap D) (N : ℕ)
    {L : Law n} (hdec : DecodesOnJ H E Lf plan I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfJ dg plan N wr))),
    ExecRefines (auditLive L Reg (liveOfJ dg plan N wr)) τ (LiveAccepts I) (fun _ => False)
      (audit L Reg (batchedSessionL H E Lf plan)) (simAuditJ H E Lf plan I wr opn dg dec one fin capOf N hdec τ)

theorem execRefines_tables_stmt : ExecRefinesTables :=
  fun H E Lf plan I wr opn dg dec one fin capOf N _ hdec τ =>
    execRefines_tables H E Lf plan I wr opn dg dec one fin capOf N hdec τ

/-- **The headline at the executable for sessions of `J` tables, at the live stratified draw.** -/
def HeadlineTables : Prop :=
  ∀ {N' n : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N')) {Reg : Type} {mPts : ℕ}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (_hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (_hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (_hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (_hk : 1 ≤ k) {Rw M : ℕ} (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (_ht : 0 ≤ t')
    (K : ℕ) (I : Inputs) (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOfJ dg plan N wr))),
    let σ := simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ
    LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' →
    ∀ {qF qS : ℝ}, (∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 plan (cont σ ω)) qF qS) →
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOfJ dg plan N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          plan tab (rs.binding hHm) k Rw M ρ t'

theorem headline_tables : HeadlineTables := by
  intro N' n p outs Reg mPts m σs ks hks St hS Kd pop Ω _ _ Lb src hA3 plan tab pp ones hConst zeros hZero lay rs hHm
    k hk Rw M hRw hM ρ hρ hr t' ht K I wr opn dg dec one fin capOf N hdec τ σ hCR qF qS hKS
  exact p.flock_headline_live_tables outs σs ks hks St hS Kd pop Lb src hA3 plan tab pp ones hConst zeros hZero lay rs
    hHm hk hRw hM hρ hr ht K I wr opn dg dec one fin capOf N hdec τ hCR hKS

/-- **The headline at the executable for sessions of `J` tables, at the live subset draw.** -/
def HeadlineTablesSubset : Prop :=
  ∀ {N' n : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N')) {Reg : Type} {mPts : ℕ}
    (kd : ℕ) (_hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (_hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (_hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (_hk : 1 ≤ k) {Rw M : ℕ} (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (_ht : 0 ≤ t')
    (K : ℕ) (_hK : K ≤ n) (I : Inputs)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf (Law.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr))),
    let σ := simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ
    LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' →
    ∀ {qF qS : ℝ}, (∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 plan (cont σ ω)) qF qS) →
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)) τ ≤
      ((n - K).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.subset kd) n)
          plan tab (rs.binding hHm) k Rw M ρ t'

theorem headline_tables_subset : HeadlineTablesSubset := by
  intro N' n p outs Reg mPts kd hkd Ω _ _ Lb src hA3 plan tab pp ones hConst zeros hZero lay rs hHm k hk Rw M hRw hM ρ
    hρ hr t' ht K hK I wr opn dg dec one fin capOf N hdec τ σ hCR qF qS hKS
  exact p.flock_headline_live_tables_subset outs kd hkd Lb src hA3 plan tab pp ones hConst zeros hZero lay rs hHm hk
    hRw hM hρ hr ht K hK I wr opn dg dec one fin capOf N hdec τ hCR hKS

open Lean (Json) in
/-- Part `j` of `count` equal consecutive parts of the session's units, as the subset draw of the population that
`Stmt.setupTables` sets table `j` up at. -/
def partDraw (population : ℕ) (units : Array ℕ) (count j : ℕ) : Json :=
  Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population),
    ("k", Lean.toJson (units.size / count)),
    ("units", Json.arr ((units.extract (j * (units.size / count)) (j * (units.size / count) + units.size / count)).map
      Lean.toJson))]

/-- **flock-verify's acceptance at more than one table.** When `Accepts I t` at `count > 1` under a tag without hidden
outputs: the tag allows several tables and `count ≤ 64`; the record's draw, when there is one, is a subset or stratified
draw, whose
population and units are the session's (without one, the population's every unit); the units split into `count`
nonempty equal parts; and table `j` is `setupH`'s statement under `Tags.tableIdentity count j` at the subset draw of
part `j`. -/
def AcceptsTables : Prop :=
  ∀ {I : Inputs} {t : Transcript}, I.tags.hiddenOutputs = false → 1 < I.count → Accepts I t →
    ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
      Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧
      I.tags.manyTables.isSome ∧ I.count ≤ Flock.Tags.MAX_TABLES ∧
      ∃ (population : ℕ) (units : Array ℕ),
        (∀ d, recordDraw t.record = some d → ∃ u, Flock.Draw.UnitDraw.ofJson d = .ok u ∧
          ((∃ k', u.law = .subset k') ∨ ∃ K st, u.law = .stratified K st) ∧
          u.population = population ∧ u.units = units) ∧
        (recordDraw t.record = none → units = Array.range population) ∧
        0 < units.size ∧ units.size % I.count = 0 ∧ sts.size = I.count ∧
        ∀ (j : ℕ) (hj : j < sts.size),
          Flock.Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables
            (some (partDraw population units I.count j)) I.partition I.program = .ok sts[j]

theorem accepts_tables : AcceptsTables := by
  intro I t hh hc h
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨hmt, hmax, population, units, hD1, hD2, hpos, hmod, hsz, hall⟩ := setupTables_parts hh hc hs
  refine ⟨spec, sts, hs, hv, hmt, hmax, population, units, hD1, hD2, hpos, hmod, hsz, fun j hj => ?_⟩
  rw [tagsAt, ite_eq_right (show ¬ I.count ≤ 1 by omega)]
  exact hall j hj

end FlockSoundness.Discharge.Tables
