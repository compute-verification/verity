import Proofs.Flock.Claims.KS.PlainRawL
import Proofs.Flock.Claims.KS.PlainCapsDep
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The plain table bound at one tape, each slot at the run's own level columns

`plain_accPr_le_rdL` bounds a plain run's acceptance by `ε_table`, the one-run event's probability, one read-set term
per fork slot and the chance of opening off the table, at read sets `rdD` holding every decoded live opening and
tables `w` that read the coins before the slot's fork round. Here its two simulator premises are discharged, so the
read sets need only hold what the run itself opens:

* `plainRawLevelCols_top`: an accepted run has no level columns at or past its schedule's last level.
* `plainRel_of`: a decoded live opening at a cap slot, where the outputs record the deployed reps' later entries, is
  a row of the run's own level columns there, at an accepted run (`plain_liveRow`, `opRL_eq_some`).
* `plain_accPr_le_rdRun`: `plain_accPr_le_rdL` at any leaf with the slot alignment from the live schedule
  (`plainLive_schedL`, `leadsAfter_simAt`, `slotRSL_lead`) and the read sets' premise from `plainRel_of`.
* `plain_accPr_le_rdAll`: at every statement some run accepts, its one table's schedule, and the bound at every
  prover, tape and table off which the model statement isn't committed: the public, hm96 and plain-leaf cases
  (`plain_laterEntriesAt`, `plain_laterEntriesHAt`, `plain_laterEntriesHUAt`), each slot's term at the run's own
  level columns.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **An accepted plain run has no level columns at or past its schedule's last level**: its reader's shape is table
0's, whose schedule is the statement's `fast100` one. -/
theorem plainRawLevelCols_topK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c : Fin ((V false).k x) → Coin} (h : (Claims.run sha512 P (VK keep) x ω c).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule}
    (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch)
    (r : ℕ) {ℓ : ℕ} (hℓ : sch.levels.size ≤ ℓ) :
    plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c).seen r ℓ = [] := by
  obtain ⟨s', spec', sts', link, c₁, mids, f, -, hf, hc, -, -⟩ := plain_acc_midsK keep h
  have hx := (Claims.Bridge.Custody.checks_ok hc).1
  rcases hro : runOf false (Claims.run sha512 P (VK keep) x ω c).seen 0 with _ | rr
  · simp only [plainRawLevelCols, hro]
  · obtain ⟨f', s'', spec'', sts'', st', hf', hs', hst', hm⟩ := runOf_m hro
    obtain rfl := Option.some.inj (hf.symm.trans hf')
    rw [hx, hs] at hs'
    obtain ⟨-, -, rfl⟩ : s = s'' ∧ spec = spec'' ∧ sts = sts'' := by simpa using hs'
    obtain rfl := Option.some.inj (hst.symm.trans hst')
    have e : rr.plainShape.sched = sch := by
      show Claims.Flock.plainSched rr.m = sch
      unfold Claims.Flock.plainSched
      rw [hm, hsch]
      rfl
    have htop : ∀ (a : Array Round) (b : ByteArray), plainLevelColsOf rr.plainShape a b ℓ = [] := fun a b => by
      unfold plainLevelColsOf
      rw [if_neg]
      rintro ⟨-, hl⟩
      rw [e] at hl
      omega
    simp only [plainRawLevelCols, hro]
    split <;> simp [htop]

/-- `plainRawLevelCols_topK` at the deployed plain verifier (`keep := id`). -/
theorem plainRawLevelCols_top {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c : Fin ((V false).k x) → Coin} (h : (Claims.run sha512 P (V false) x ω c).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule}
    (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch)
    (r : ℕ) {ℓ : ℕ} (hℓ : sch.levels.size ≤ ℓ) :
    plainRawLevelCols (Claims.run sha512 P (V false) x ω c).seen r ℓ = [] :=
  plainRawLevelCols_topK id (Ω := Ω) (P := P) (x := x) (ω := ω) (c := c) (h := h) (s := s) (spec := spec) (sts := sts) (st := st) (sch := sch) (hs := hs) (hst := hst) (hsch := hsch) (r := r) (ℓ := ℓ) (hℓ := hℓ)

/-- **A decoded live opening at a cap slot is a row of the run's own level columns**: where the decoded outputs
record the deployed reps' later entries (`hE`), an output the concrete simulator's live outcome returns that opens
rep `r`'s level-`ℓ + 1` position `q` at its recorded `ℓ`-th cap to `v` is one where `P`'s run accepts, and `v` is a
row the run's level-`ℓ + 1` columns of rep `r` hold at `q` (`plain_liveRow`). -/
theorem plainRel_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512) (T : TabSpec spec.mPts)
    (sch : Accounting.Schedule) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAt sch ins.read.I d (wrPD ins.read.I d o)
          (tableOutL H512 enc512 Lf (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x)
    (rr : Fin (plainN x + 2) → LCoin 65536) (a : List (Model.OutCL 𝔽 Dig (List UInt8)))
    (ha : Claims.Bridge.RetOK (fun _ cap pend os => tableOutL H512 enc512 Lf cap pend os)
      (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Dig) execArith ([T].get j))
      (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
        sha512 (plainSim P x spec st) x (ω, u) rr) a)
    (r : Fin 2) (ℓ : ℕ) (q : Fin (2 ^ (lvOf sch (ℓ + 1)).logLen)) (v : Fin (2 ^ (lvOf sch (ℓ + 1)).lanes) → 𝔽)
    (hv : opCapL H512 enc512 Lf sch r.val ℓ a q = some v) :
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, rr))).out = .acc ∧
      ∃ e, (q.val, e) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, rr))).seen r.val
        (ℓ + 1) ∧ rowFin (lvOf sch (ℓ + 1)).lanes e = v := by
  obtain ⟨hV, pend, hpl, rfl⟩ := ha
  have hacc := plain_acc_of_abort (plainSimAbortAt x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C
    (plainRoot P x ω)) hV
  refine ⟨hacc, ?_⟩
  unfold opCapL at hv
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hv
  obtain ⟨dc, -, hdc⟩ := Option.bind_eq_some_iff.1 hv
  obtain ⟨y, hy, hyq, hyv⟩ := opRL_eq_some (Option.isSome_iff_exists.2 ⟨v, hdc⟩)
  refine ⟨y.2.1, ?_, Option.some.inj (hyv.symm.trans hdc)⟩
  rw [← hyq]
  exact plain_liveRow hs hin hset h hd P ω hacc hV.1 r.isLt (hE _ hV.1 (pend 0) (hpl 0) r.val r.isLt) (ℓ + 1)
    (by omega) y.1 y.2.1 (List.mem_map_of_mem hy)

open Classical in
/-- **A plain run accepts with probability at most `ε_table`, its one-run event's, each slot's read-set term and
that of opening off the table, at read sets holding the run's own level columns**, at any leaf `Lf`, its decoders
`hdec` and openings `opn` whose outputs record the deployed reps' later entries (`hE`): `plain_accPr_le_rdL`, the
simulated strategy meeting each slot at its fork round (`plainLive_schedL`, `leadsAfter_simAt`, `slotRSL_lead`) and
each decoded opening a row of the run's columns (`plainRel_of`). -/
theorem plain_accPr_le_rdRun {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (regionsOf st)).m) (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL H512 enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL H512 enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
          o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
            (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAt ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPD ins.read.I d o)
          (tableOutL H512 enc512 Lf (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hsc : PlainClashRun P x ω spec.mPts plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
      (plainSim P x spec st) (plainΦ x spec st)
      (LiveSelfClash H512 enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Dig) execArith
          ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j)))
    (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
      (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) → w r ℓ c = w r ℓ c')
    (hrd : ∀ r ℓ (c : Fin ((V false).k x) → Coin), (Claims.run sha512 P (V false) x ω c).out = .acc →
      ∀ q a, (q.val, a) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω c).seen r.val (ℓ.val + 1) →
        rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes a ∈ rdD r ℓ c q) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁ P x ω s) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
        Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_accPr_le_rdL Lf hset h hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay opn hdec (plain_mPts hs hin hset).1
    (plain_mPts hs hin hset).2.1 (plain_mPts hs hin hset).2.2 P ω C hT hsc
    (fun r ℓ u => slotRSL_lead L (plain_mPts hs hin hset).2.2 execArith H512 enc512 Lf m k0 l₀ l₁ rest ext hsch hk0
      hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!)
      (leadsAfter_simAt execArith H512 enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ _ opn
        (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (fun _ => rfl)
        (plainRoot P x ω) (plainLive x spec st (plain_mPts hs hin hset).1 P ω u)
        (plainSim_root P x spec st ω (plain_mPts hs hin hset).1 u)
        (plainLive_schedL hset h hsch (by simp) (plain_mPts hs hin hset).1 P ω u)) r ℓ.val)
    rdD w hwj fun r ℓ u rr a ha q v hv => by
      obtain ⟨hacc, e, he, rfl⟩ := plainRel_of hs hin hset h hd Lf _ _ opn hE P ω C u rr a ha r ℓ.val q v hv
      exact hrd r ℓ _ hacc q e he

open Classical in
/-- **At every statement some run accepts, the plain bound at the run's own level columns**: the statement's setup,
its one table `st`, that table's `fast100` schedule `sch` and its model schedule, the statement's tables and the
model schedule as `Accounting.fast100`'s, and at every prover, tape and table
of full-width columns that is not good, `plain_accPr_le_rdRun` at read sets holding the run's own level columns and
tables reading the coins before each slot's fork round: at `leaf512` where the outputs are public or hidden under
hm96 leaves (`plain_hdecAt`, `plain_hdecHAt`; `plain_laterEntriesAt`, `plain_laterEntriesHAt`; the live self-clash's
clash `plainSimClashAt_of_later`), at `leafPlain` under plain SHA-512 leaves (`plain_hdecHUAt`,
`plain_laterEntriesHUAt`, `plainSimClashHU`). -/
theorem plain_accPr_le_rdAll {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V false) x) :
    ∃ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (m k0 : ℕ)
      (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ),
      setupOf false x = some (ins, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      schedOf sch = ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧ tables false x = [(spec, st)] ∧
      Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧
      ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
        (∀ p, wide false x (C p)) → ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
        ∀ (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
              Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
          (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
              Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽),
        (∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
          (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) →
            w r ℓ c = w r ℓ c') →
        (∀ r ℓ (c : Fin ((V false).k x) → Coin), (Claims.run sha512 P (V false) x ω c).out = .acc →
          ∀ q a, (q.val, a) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω c).seen r.val (ℓ.val + 1) →
            rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes a ∈ rdD r ℓ c q) →
        Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
          (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁ P x ω s) +
          ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
            Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
          Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
            ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
              Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧
                u ≠ C p := by
  obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := acc_draw hno
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  obtain ⟨sch, hF, hF'⟩ := exec_sched hsch
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs hin hset h0 L C hw hg
    rwa [e] at this
  refine ⟨ins, spec, sts, sts[0]'h0, sch, (sts[0]'h0).m, k0, l₀, l₁, rest, ext, hset,
    Array.getElem?_eq_getElem h0, hF, hF'.symm, ht, hsch, ?_⟩
  intro Ω _ _ P ω C hw hg rdD w hwj hrd
  cases hh : ins.read.I.tags.hiddenOutputs
  · refine plain_accPr_le_rdRun hs hin hset ht hd leaf512 L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay _
      (plain_hdecAt hs hin hset ht hh hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay)
      (plain_laterEntriesAt hs hin hset ht hh hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay) P ω C (hnc C hw hg)
      (plainClashRun_of_rawL (plainRawClashRunL_of_abort (fun u r' hna hq => ?_)
        ((plainSimClashAt_of_later (plainSimLaterClashAt x) s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest
          ext hsch hm13 hlay Ω P ω).1 hh C)))
      rdD w hwj hrd
    obtain ⟨a, ha, -⟩ := hq
    exact plainSimAbortAt x s hs hin ins spec sts hset _ ht Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1
  · by_cases hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · refine plain_accPr_le_rdRun hs hin hset ht hd leaf512 L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay _
        (plain_hdecHAt hs hin hset ht hh hl hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay)
        (plain_laterEntriesHAt hs hin hset ht hh hl hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay) P ω C (hnc C hw hg)
        (plainClashRun_of_rawL (plainRawClashRunL_of_abort (fun u r' hna hq => ?_)
          ((plainSimClashAt_of_later (plainSimLaterClashAt x) s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁
            rest ext hsch hm13 hlay Ω P ω).2 hh hl C)))
        rdD w hwj hrd
      obtain ⟨a, ha, -⟩ := hq
      exact plainSimAbortAt x s hs hin ins spec sts hset _ ht Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1
    · have hl' := plain_leafU hs hin hset hl
      refine plain_accPr_le_rdRun hs hin hset ht hd leafPlain L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay _
        (plain_hdecHUAt hs hin hset ht hh hl' hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay)
        (plain_laterEntriesHUAt hs hin hset ht hh hl' hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay) P ω C (hnc C hw hg)
        (plainClashRun_of_rawL (plainRawClashRunL_of_abort (fun u r' hna hq => ?_)
          (plainSimClashHU hs hin hset ht hh hl' hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay P ω C)))
        rdD w hwj hrd
      obtain ⟨a, ha, -⟩ := hq
      exact plainSimAbortAt x s hs hin ins spec sts hset _ ht Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1

end FlockDeploy.KS
