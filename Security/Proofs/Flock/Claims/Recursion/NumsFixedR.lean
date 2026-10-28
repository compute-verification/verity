import Proofs.Flock.Claims.Recursion.DeployHonest
import Proofs.Flock.Claims.Recursion.ResLowerDeployed
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.VStarLinks
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Layout.Drawn

/-!
# A staged part's numbers do not depend on the inner rounds (`numsFixed`)

`NumsFixed st`: wherever part `s`'s staged statement sets up at two inner rounds `ms` and `ms'`, its numbers (`numsOf`:
`VO`'s rounds and the table error, `N₀` and weight) agree. The two statements are `partRaw` at the same compiled
statement and part, so they differ only in what the stage reads off the rounds: the part's public file, the inner
session's record and the coin server's transcript. The record and transcript are recursive inputs the setup's tables
don't read (`RecIns.setup_read`), and the public file changes nothing else Flock's parsers read (`rawIns_pf`). So the
two setups are `Stmt.setupTables` at the same arguments but the public file; the recursive setup holds each to one
table (`RecSetup.oneFull`) under `verity/flock-circuit`'s hidden outputs (`staged_one`), which is `Stmt.setupHidden`'s
statement (`setupTables_okH'`).

The numbers read a table's `m`, `mPts`, `k_log` and region count and the spec's `points` and `mPts`. `setupHidden_key`
reads those off the circuit (`HmOut.parse`, the same file) and the statement's `n`: without a draw, the loaded file's
instance count, which `RecVStar.checkCircuit` holds to the instances of V*'s statement `s` at the verifier's own inner
shape (`staged_pop`); with one, the draw's proved units (`Layout.drawn_spec`), the same draw.
-/

namespace RecDeploy.NumsFix

open Flock
open Claims (Msg Out)
open FlockSoundness.Refine
open FlockSoundness.Discharge.Exec (range_mapM_ok mPts_of_any)
open Lean (Json)

/-! ## Flock's parsers with the public file swapped -/

set_option maxHeartbeats 4000000 in
/-- **A raw statement with its public file swapped reads the same**, but for the public file. -/
theorem rawIns_pf {w : FlockDeploy.Raw} {pf : ByteArray} {b b' : FlockDeploy.Ins} (h : w.ins = .ok b)
    (h' : ({ w with publicFile := pf } : FlockDeploy.Raw).ins = .ok b') :
    b' = { b with read := { b.read with I := { b.read.I with publicFile := pf } } } := by
  unfold FlockDeploy.Raw.ins at h h'
  simp only [bind, Except.bind, pure, Except.pure] at h h'
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    repeat' split at h'
    all_goals first | (cases h'; done) | skip
  all_goals (cases h; cases h'; simp_all)
  all_goals exact ‹∀ (k : ℕ) (t : Array (String × ℕ × ℕ)), _ = k → ¬_ = some t› _ _ rfl (‹some _ = _› : some _ = _).symm

/-! ## `Stmt.setupTables` at one table, with the verifier's own public input words -/

set_option maxHeartbeats 4000000 in
/-- `HiddenExec.setupTables_okH` at any `--public-inputs`. -/
theorem setupTables_okH' {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {draw partition : Option Json} {program : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))}
    {q : Option ByteArray} {spec : SessionSpec} {sts : Array Flock.Stmt} (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure q =
      .ok (spec, sts)) :
    (count ≤ 1 ∧ ∃ st, Stmt.setupHidden tags cf pf coins tables draw partition program q = .ok st ∧ spec = st.spec ∧
        sts = #[st]) ∨
      (1 < count ∧ sts.size = count) := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind, ite_eq_left hh] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  · exact Or.inl ⟨by assumption, _, FlockSoundness.Discharge.Exec.ok_of_refuse (by assumption), rfl, rfl⟩
  all_goals
    obtain ⟨hsz, -⟩ := range_mapM_ok (by assumption)
    exact Or.inr ⟨by omega, hsz⟩

/-! ## What a statement's numbers read off `Stmt.setupHidden` -/

theorem outRegions_size {c : Circuit} {pub : Public} {r : Array Region} (h : HmOut.regions c pub = .ok r) :
    r.size = c.ports.size := by
  unfold HmOut.regions at h
  try dsimp only at h
  exact (range_mapM_ok h).1

theorem inRegions_size {c : Circuit} {pi : HmIn.Ports} {pub : Public} {r : Array Region}
    (h : HmIn.regions c pi pub = .ok r) :
    r.size = if pi.ports.isEmpty then 0 else (HmIn.inRegions c pi).size := by
  unfold HmIn.regions at h
  cases he : pi.ports.isEmpty
  · simp only [he, Bool.false_eq_true, ↓reduceIte] at h ⊢
    try dsimp only at h
    exact (range_mapM_ok h).1
  · simp only [he, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h ⊢
    subst h
    rfl

set_option maxHeartbeats 2000000 in
/-- `HmIn.loadPublic` is `HmRow.loadPublic`'s file, with checks. -/
theorem inLoad_row {tags : Tags} {c : Circuit} {pi : HmIn.Ports} {f : ByteArray} {pop : Public}
    (h : HmIn.loadPublic tags c pi f = .ok pop) : ∃ c', HmRow.loadPublic tags c' f = .ok pop := by
  unfold HmIn.loadPublic at h
  try dsimp only at h
  try simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, by assumption⟩

set_option maxHeartbeats 4000000 in
/-- **What `Stmt.setupHidden` set up**, at any `--public-inputs`: its circuit is `HmOut.parse`'s, its public file the
loaded one or its draw, and its `n`, `m`, `mPts` and region count are the circuit's and the public file's. -/
theorem setupHidden_key {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program q : Option ByteArray} {st : Flock.Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program q = .ok st) :
    HmOut.parse tags cf tables = .ok st.c ∧ ∃ pi pop, HmIn.parse st.c = .ok pi ∧
      HmIn.loadPublic tags st.c pi pf = .ok pop ∧ (draw = none → st.pub = pop) ∧
      (∀ d, draw = some d → HmIn.drawn st.c pi pop d = .ok st.pub) ∧
      st.regions.size = st.c.ports.size + (if pi.ports.isEmpty then 0 else (HmIn.inRegions st.c pi).size) ∧
      st.n = st.pub.n ∧ st.m = st.c.kLog + max 3 (log2ceil (max ((st.pub.n + st.c.g - 1) / st.c.g) 1)) ∧
      st.mPts = PT_LOCAL + max 3 (log2ceil (max ((st.pub.n + st.c.g - 1) / st.c.g) 1)) := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨by assumption, _, _, by assumption, by assumption, ?_, ?_, ?_, rfl, rfl, rfl⟩
    · intro hd
      first | exact absurd hd (Option.some_ne_none _) | rfl
    · intro d' hd
      first | exact absurd hd.symm (Option.some_ne_none _) | (cases hd; assumption)
    · dsimp only
      rw [Array.size_append, outRegions_size ‹HmOut.regions _ _ = _›, inRegions_size ‹HmIn.regions _ _ _ = _›]

/-- **Two setups at the same arguments but the public file have the same numbers' inputs**, where they hold the same
population without a draw. -/
theorem hidden_same {tags : Tags} {cf pf pf' : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program q : Option ByteArray} {t t' : Flock.Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program q = .ok t)
    (h' : Stmt.setupHidden tags cf pf' coins tables draw partition program q = .ok t')
    (hpop : draw = none → RecVStar.population t = RecVStar.population t') :
    t.c = t'.c ∧ t.m = t'.m ∧ t.mPts = t'.mPts ∧ t.regions.size = t'.regions.size := by
  obtain ⟨hc, pi, pop, hpi, hl, hn0, hd0, hr, hn, hm, hmp⟩ := setupHidden_key h
  obtain ⟨hc', pi', pop', hpi', hl', hn0', hd0', hr', hn', hm', hmp'⟩ := setupHidden_key h'
  have ec : t.c = t'.c := Except.ok.inj (hc.symm.trans hc')
  have epi : pi = pi' := by
    rw [ec] at hpi
    exact Except.ok.inj (hpi.symm.trans hpi')
  have en : t.pub.n = t'.pub.n := by
    cases draw with
    | none =>
      obtain ⟨c₁, hl₁⟩ := inLoad_row hl
      obtain ⟨c₂, hl₂⟩ := inLoad_row hl'
      have a : RecVStar.population t = t.n := by
        unfold RecVStar.population
        rw [hn0 rfl, FlockSoundness.Discharge.Recursion.loadPublic_population hl₁] <;> rfl
      have a' : RecVStar.population t' = t'.n := by
        unfold RecVStar.population
        rw [hn0' rfl, FlockSoundness.Discharge.Recursion.loadPublic_population hl₂] <;> rfl
      rw [← hn, ← hn', ← a, ← a']
      exact hpop rfl
    | some d =>
      have e₁ := hd0 d rfl
      have e₂ := hd0' d rfl
      unfold HmIn.drawn at e₁ e₂
      obtain ⟨D, hD, -, hnD, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec e₁
      obtain ⟨D', hD', -, hnD', -⟩ := FlockSoundness.Discharge.Layout.drawn_spec e₂
      obtain rfl : D' = D := Except.ok.inj (hD'.symm.trans hD)
      rw [hnD, hnD']
  refine ⟨ec, ?_, ?_, ?_⟩
  · rw [hm, hm', ec, en]
  · rw [hmp, hmp', ec, en]
  · rw [hr, hr', ec, epi]

/-! ## A staged part's setup -/

/-- A staged part's inputs are Flock's parsers on the raw part at the staged public file. -/
theorem staged_base {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns}
    (hins : insO (staged st x ms s) = some i) :
    ∃ r p, readX x = some r ∧ r.w.parts[s]? = some p ∧
      ({ p.base with publicFile := st.pub x ms s } : FlockDeploy.Raw).ins = .ok i.base := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at hins; cases hins
  obtain ⟨r, p, hr, hp, hd⟩ := staged_decode hne
  refine ⟨r, p, hr, hp, ?_⟩
  unfold insO at hins
  rw [hd] at hins
  simp only [Option.bind_eq_bind, Option.bind_some] at hins
  split_ifs at hins
  · simp at hins
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using hins
  exact RecRaw.ins_base (toOption_eq_some h')

/-- **A staged part sets up one table, with hidden outputs**: the recursive setup holds a session at the verifier's inner
shape to one table (`RecSetup.oneFull`), and its `checkStmt` to `verity/flock-circuit` (`ResLower.staged_hidden`'s
argument). -/
theorem staged_one {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (hins : insO (staged st x ms s) = some i) (hset : i.setup = .ok (spec, sts)) :
    sts.size = 1 ∧ i.base.read.tags.hiddenOutputs = true := by
  obtain ⟨r, -, hinner, -⟩ := insO_staged hins
  have hone := RecIns.setup_oneFull hset
  rw [hinner] at hone
  have h1 := (oneFull_ok hone).1
  refine ⟨h1, ?_⟩
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hn := ResLower.checkStmt_name (checks_stmt hC 0 (by omega))
  obtain ⟨⟨nm, hf⟩, coins, hI⟩ := ResLower.insO_tags hins
  have hf' := hf
  unfold Flock.Tags.find? at hf'
  have hnm : nm = Flock.Tags.circuit.name := by
    have h2 : i.base.stmtTags.name = nm := by simpa using List.find?_some hf'
    rw [← h2, hn]
  subst hnm
  rw [FlockSoundness.Discharge.Recursion.find?_circuit] at hf
  have hT : i.base.stmtTags = Flock.Tags.circuit := (Option.some.inj hf).symm
  have hI' : i.base.read.I.tags.hiddenOutputs = true := by
    rw [hI, hT]
    unfold Flock.Tags.withCoins
    split <;> rfl
  show i.base.read.tags.hiddenOutputs = true
  unfold Flock.Readable.Read.tags
  split
  · rw [FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs]
    exact hI'
  · exact hI'

/-- **A staged part's first table's population is V\*'s statement `s`'s instances**, at the verifier's own inner shape
(`RecVStar.checkCircuit`'s unit check). -/
theorem staged_pop {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (hins : insO (staged st x ms s) = some i) (hset : i.setup = .ok (spec, sts))
    (h0 : 0 < sts.size) :
    ∃ r ss sv, readX x = some r ∧ RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some sv ∧
      RecVStar.population sts[0] = sv.instances := by
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  obtain ⟨j, f, sv, hj, -, hcc⟩ := checkStmt_circuit (checks_stmt hC 0 h0)
  obtain rfl := Option.some.inj hj
  obtain ⟨hck, -⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hcc
  obtain ⟨ss, hss, hjs, -, hpop, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck
  exact ⟨r, ss, sv, hr, hss, hjs, hpop⟩

/-- **Part `s`'s staged statements at two inner rounds set up the same numbers' inputs**: one table each, at the same
`m`, `mPts`, `k_log` and region count, each the spec's one table. -/
theorem staged_same {st : Stage} {x : Msg} {ms ms' : List (Msg × Out)} {s : ℕ} {i i' : RecIns}
    {spec spec' : Flock.SessionSpec} {sts sts' : Array Flock.Stmt}
    (h1 : setupO (staged st x ms s) = some (i, spec, sts)) (h2 : setupO (staged st x ms' s) = some (i', spec', sts')) :
    ∃ t t', sts = #[t] ∧ sts' = #[t'] ∧ spec = t.spec ∧ spec' = t'.spec ∧ t.m = t'.m ∧ t.mPts = t'.mPts ∧
      t.c.kLog = t'.c.kLog ∧ t.regions.size = t'.regions.size := by
  obtain ⟨hins1, hset1⟩ := setupO_some h1
  obtain ⟨hins2, hset2⟩ := setupO_some h2
  obtain ⟨hsz1, hh⟩ := staged_one hins1 hset1
  obtain ⟨hsz2, -⟩ := staged_one hins2 hset2
  obtain ⟨r, p, hr, hp, hb1⟩ := staged_base hins1
  obtain ⟨r', p', hr', hp', hb2⟩ := staged_base hins2
  obtain rfl : r = r' := Option.some.inj (hr.symm.trans hr')
  obtain rfl : p = p' := Option.some.inj (hp.symm.trans hp')
  have hb2' : ({ ({ p.base with publicFile := st.pub x ms s } : FlockDeploy.Raw) with
      publicFile := st.pub x ms' s } : FlockDeploy.Raw).ins = .ok i'.base := hb2
  have e := rawIns_pf hb1 hb2'
  have hs1 : Stmt.setupTables i.base.read.tags i.base.read.I.circuitFile i.base.read.I.publicFile i.base.read.coins
      i.base.read.I.count i.base.read.I.tables i.base.read.draw i.base.read.I.partition i.base.read.I.program
      i.base.read.I.work i.base.read.I.stratified i.base.read.I.closure i.base.read.publicInputs = .ok (spec, sts) :=
    RecIns.setup_read hset1
  have hs2 : Stmt.setupTables i.base.read.tags i.base.read.I.circuitFile (st.pub x ms' s) i.base.read.coins
      i.base.read.I.count i.base.read.I.tables i.base.read.draw i.base.read.I.partition i.base.read.I.program
      i.base.read.I.work i.base.read.I.stratified i.base.read.I.closure i.base.read.publicInputs = .ok (spec', sts') := by
    have := RecIns.setup_read hset2
    rw [e] at this
    exact this
  rcases setupTables_okH' hh hs1 with ⟨-, t, ht, rfl, rfl⟩ | ⟨hc, hsz⟩
  swap
  · omega
  rcases setupTables_okH' hh hs2 with ⟨-, t', ht', rfl, rfl⟩ | ⟨hc, hsz⟩
  swap
  · omega
  have hpop : i.base.read.draw = none → RecVStar.population t = RecVStar.population t' := fun _ => by
    obtain ⟨r₁, ss, sv, hr₁, hss, hj, hp₁⟩ := staged_pop hins1 hset1 (by simp)
    obtain ⟨r₂, ss', sv', hr₂, hss', hj', hp₂⟩ := staged_pop hins2 hset2 (by simp)
    obtain rfl : r₁ = r₂ := Option.some.inj (hr₁.symm.trans hr₂)
    obtain rfl : ss = ss' := Except.ok.inj (hss.symm.trans hss')
    obtain rfl : sv = sv' := Option.some.inj (hj.symm.trans hj')
    exact (hp₁ : RecVStar.population t = _).trans (hp₂ : RecVStar.population t' = _).symm
  obtain ⟨hc, hm, hmp, hrs⟩ := hidden_same ht ht' hpop
  exact ⟨t, t', rfl, rfl, rfl, rfl, hm, hmp, by rw [hc], hrs⟩

/-! ## The numbers -/

theorem rounds_one (zk : Bool) (spec : Flock.SessionSpec) (t : Flock.Stmt) :
    FlockDeploy.rounds zk spec #[t] = 0 + 2 * FlockDeploy.repRounds zk spec t := by
  first | rfl | simp [FlockDeploy.rounds]

/-- **A one-table statement's numbers** are its table's. -/
theorem numsOf_one {xO : Msg} {i : RecIns} {spec : Flock.SessionSpec} {t : Flock.Stmt}
    (h : setupO xO = some (i, spec, #[t])) :
    numsOf xO = ⟨FlockDeploy.tableErr true spec t, FlockDeploy.tableLen true t, FlockDeploy.tableWeight true t,
      2 + (0 + 2 * FlockDeploy.repRounds true spec t), Claims.Flock.Numbers.waits, Claims.Flock.Numbers.cap, 0, 0⟩ := by
  have ht : tablesO xO = [(spec, t)] := by simp [tablesO, h]
  have hk : VO.k xO = 2 + FlockDeploy.rounds true spec #[t] := by simp only [VO, h]
  rw [rounds_one] at hk
  simp only [numsOf, εtableO, N₀O, weightO, ht, hk, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    add_zero]

end RecDeploy.NumsFix

namespace RecDeploy

open NumsFix

/-- **`NumsFixed` holds at every stage**: part `s`'s staged statements that set up at two inner rounds have the same
numbers (`staged_same`, `numsOf_one`). -/
theorem numsFixed (st : Stage) : NumsFixed st := by
  intro x ms ms' s h1 h2
  obtain ⟨⟨i, spec, sts⟩, hs1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨⟨i', spec', sts'⟩, hs2⟩ := Option.isSome_iff_exists.1 h2
  obtain ⟨t, t', rfl, rfl, rfl, rfl, hm, hmp, hk, hr⟩ := staged_same hs1 hs2
  rw [numsOf_one hs1, numsOf_one hs2]
  simp only [FlockDeploy.tableErr, FlockDeploy.acct, FlockDeploy.tableLen, FlockDeploy.tableWeight,
    FlockDeploy.repRounds, Flock.Stmt.spec, hm, hmp, hk, hr]

end RecDeploy
