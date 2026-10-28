import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.ResLowerDeployed
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Exec.Setup

/-!
# A staged part's numbers don't read its public file

Part `s`'s staged statement at inner rounds `ms` (`RecDeploy.staged`) differs from the one at `ms'` only in what
the rounds put in it: the part's public file, the inner session's record and the transcript. Its numbers (`numsOf`:
`ε_table`, `N₀`, the weight and the rounds) read only its one table's `m`, `k_log`, region count and `m_pts`, and
those read the circuit, the tables, the draw and the unit count `n`:

* `RecSetup.oneFull`, under the inner shape every staged statement carries, passes one table only.
* `RecVStar.checkStmt` passes only `verity/flock-circuit`, a statement with hidden outputs, so the table is
  `Stmt.setupHidden`'s.
* With a draw, `n` is the draw's proved units (`HmRow.drawn`). Without one, the public file records no population
  (`HmRow.loadPublic`), so `n` is the table's population, which `checkStmt` pins to V*'s statement `s` at the inner
  shape.

So the numbers agree at any two rounds where the staged statement sets up (`numsOf_staged_eq`), they are `nums`'
(`numsOf_staged`), and `vstarAt` is the staged statement itself (`vstarAt_staged`). Nothing here assumes the verifier's
own public input words (`--public-inputs`) are absent: the walks of `Stmt.setupTables` and `Stmt.setupHidden` take any.
-/

set_option Elab.async false

namespace Verity.Recursive.Numbers

open Flock RecDeploy FlockSoundness.Refine
open Lean (Json)

section Walks

variable {tags : Tags} {cf pf : ByteArray} {coins : Flock.Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {publicInputs : Option ByteArray}
  {spec : SessionSpec} {sts : Array Flock.Stmt}

set_option maxHeartbeats 4000000 in
/-- Under hidden outputs, `setupTables` at any public input words is `setupHidden`'s one table, or several. -/
theorem setupTables_hidden (hh : tags.hiddenOutputs = true)
    (h : Flock.Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure
      publicInputs = .ok (spec, sts)) :
    (∃ st, Flock.Stmt.setupHidden tags cf pf coins tables draw partition program publicInputs = .ok st ∧
        spec = st.spec ∧ sts = #[st]) ∨ (1 < count ∧ sts.size = count) := by
  unfold Flock.Stmt.setupTables at h
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
  all_goals
    first
    | exact Or.inl ⟨_, by assumption, rfl, rfl⟩
    | exact absurd hh ‹_›
    | (obtain ⟨hsz, -⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok (by assumption)
       exact Or.inr ⟨by omega, hsz⟩)

set_option maxHeartbeats 4000000 in
/-- What `setupHidden`'s table is made of, at any public input words: its circuit, its population and draw, the unit
count `n` the shape reads, and its regions. -/
theorem setupHidden_shape {st : Flock.Stmt}
    (h : Flock.Stmt.setupHidden tags cf pf coins tables draw partition program publicInputs = .ok st) :
    ∃ pi pop A B, HmOut.parse tags cf tables = .ok st.c ∧ HmIn.parse st.c = .ok pi ∧
      HmIn.loadPublic tags st.c pi pf = .ok pop ∧ (draw = none → st.pub = pop) ∧
      (∀ d, draw = some d → HmIn.drawn st.c pi pop d = .ok st.pub) ∧ st.n = st.pub.n ∧
      st.m = st.c.kLog + max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) ∧
      st.mPts = PT_LOCAL + max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) ∧
      HmOut.regions st.c st.pub = .ok A ∧ HmIn.regions st.c pi st.pub = .ok B ∧ st.regions = A ++ B := by
  cases draw <;> unfold Flock.Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨_, _, _, _, by assumption, by assumption, by assumption, fun hd => ?_, fun d' hd => ?_, rfl, rfl, rfl,
      by assumption, by assumption, rfl⟩
    · first | rfl | cases hd
    · first | (cases hd; assumption) | cases hd

/-- `HmIn.loadPublic`'s population is `HmRow.loadPublic`'s, at the circuit with its public input row. -/
theorem hmIn_loadPublic_row {c : Flock.Circuit} {pi : HmIn.Ports} {f : ByteArray} {pop : Public}
    (h : HmIn.loadPublic tags c pi f = .ok pop) : ∃ c', HmRow.loadPublic tags c' f = .ok pop := by
  unfold HmIn.loadPublic at h
  dsimp only at h
  try simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, by assumption⟩

/-- `HmIn.regions` returns one region per public input block, a count of the circuit's alone. -/
theorem hmIn_regions_size {c : Flock.Circuit} {pi : HmIn.Ports} {pub : Public} {B : Array Region}
    (h : HmIn.regions c pi pub = .ok B) : B.size = if pi.ports.isEmpty then 0 else (HmIn.inRegions c pi).size := by
  unfold HmIn.regions at h
  dsimp only at h
  split at h
  · rename_i he
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp [he]
  · rename_i he
    rw [if_neg he]
    exact (FlockSoundness.Discharge.Exec.range_mapM_ok h).1

end Walks

/-! ## What the statement's readers take from the staged statement -/

/-- The verifier's readers keep the statement's circuit, table count and tables. -/
theorem raw_ins_files {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    s.read.I.circuitFile = w.circuit ∧ s.read.I.count = w.count ∧ s.read.I.tables = w.tables := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨rfl, rfl, rfl⟩)

/-- A `--zk` statement's coins are its keyed coin tree. -/
theorem raw_ins_coins {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) (hz : w.zk = true) :
    ∃ cs, s.read.I.coins = .tree cs := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals
    first
    | (cases h; done)
    | (cases h
       first
       | exact ⟨_, rfl⟩
       | (try dsimp only
          rw [if_pos hz]
          exact ⟨_, rfl⟩)
       | exact absurd hz ‹_›)

/-- A `--zk` statement's tags are its statement's under the circuit's mask and its table count. -/
theorem read_tags_of {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) (hz : w.zk = true) :
    s.read.tags = Zk.tags s.stmtTags (Zk.maskLogOf w.circuit) w.count := by
  obtain ⟨cs, hcs⟩ := raw_ins_coins h hz
  obtain ⟨-, hT⟩ := ResLower.raw_ins_tags h
  obtain ⟨hc, hn, -⟩ := raw_ins_files h
  have hz' : s.read.zk = true := (FlockDeploy.Raw.ins_zk h).trans hz
  unfold Flock.Readable.Read.tags
  rw [if_pos hz', FlockSoundness.Discharge.ZkExec.zkTags_eq, hT, hcs, FlockSoundness.Discharge.ZkExec.withCoins_tree,
    hc, hn]

/-- Part `s`'s staged statement is read from its part's raw statement with the staged public file. -/
theorem staged_ins {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) :
    ∃ r p, readX x = some r ∧ r.w.parts[s]? = some p ∧ p.base.zk = true ∧
      FlockDeploy.Raw.ins { p.base with publicFile := st.pub x ms s } = .ok i.base := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, hr, hp, hd⟩ := staged_decode hne
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h with hz
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  exact ⟨r, p, hr, hp, by simpa [partRaw] using hz, RecRaw.ins_base (toOption_eq_some h')⟩

/-- **The staged statements at any two rounds are read alike**: the same tags, circuit file, tables and draw. -/
theorem reads_eq {st : Stage} {x : Claims.Msg} {ms ms' : List (Claims.Msg × Claims.Out)} {s : ℕ} {i i' : RecIns}
    (hi : insO (staged st x ms s) = some i) (hi' : insO (staged st x ms' s) = some i') :
    i.base.read.tags = i'.base.read.tags ∧ i.base.read.I.circuitFile = i'.base.read.I.circuitFile ∧
      i.base.read.I.tables = i'.base.read.I.tables ∧ i.base.read.draw = i'.base.read.draw := by
  obtain ⟨r, p, hr, hp, hz, hb⟩ := staged_ins hi
  obtain ⟨r', p', hr', hp', -, hb'⟩ := staged_ins hi'
  obtain rfl : r = r' := Option.some.inj (hr.symm.trans hr')
  obtain rfl : p = p' := Option.some.inj (hp.symm.trans hp')
  have e1 : i.base.stmtTags = i'.base.stmtTags :=
    Option.some.inj ((ResLower.raw_ins_tags hb).1.symm.trans (ResLower.raw_ins_tags hb').1)
  obtain ⟨hc, -, ht⟩ := raw_ins_files hb
  obtain ⟨hc', -, ht'⟩ := raw_ins_files hb'
  obtain ⟨d, hd, hdr⟩ := FlockDeploy.Raw.ins_draw hb
  obtain ⟨d', hd', hdr'⟩ := FlockDeploy.Raw.ins_draw hb'
  have edd : d = d' := Except.ok.inj (hd.symm.trans hd')
  refine ⟨?_, hc.trans hc'.symm, ht.trans ht'.symm, by rw [hdr, hdr', edd]⟩
  rw [read_tags_of hb hz, read_tags_of hb' hz, e1] <;> rfl

/-! ## One staged table's shape -/

/-- The unit count a staged table must have: the draw's proved units, else V*'s statement `s`'s instances. -/
def nOf (r : XR) (s : ℕ) : Option Json → Option ℕ
  | some d => (Draw.UnitDraw.ofJson d).toOption.map (·.proved.size)
  | none => ((RecVStar.statements r.shape).toOption.bind (·[s]?)).map (·.instances)

/-- **A staged statement that sets up has one table**, from `setupHidden` at its circuit, with the unit count `nOf`. -/
theorem side {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {i : RecIns}
    {spec : SessionSpec} {sts : Array Flock.Stmt} (h : setupO (staged st x ms s) = some (i, spec, sts)) :
    ∃ r t, readX x = some r ∧ spec = t.spec ∧ sts = #[t] ∧
      HmOut.parse i.base.read.tags i.base.read.I.circuitFile i.base.read.I.tables = .ok t.c ∧
      some t.n = nOf r s i.base.read.draw ∧
      (∃ pi B, HmIn.parse t.c = .ok pi ∧ HmIn.regions t.c pi t.pub = .ok B ∧
        t.regions.size = t.c.ports.size + B.size) ∧
      t.m = t.c.kLog + max 3 (log2ceil (max ((t.n + t.c.g - 1) / t.c.g) 1)) ∧
      t.mPts = PT_LOCAL + max 3 (log2ceil (max ((t.n + t.c.g - 1) / t.c.g) 1)) := by
  obtain ⟨hins, hset⟩ := setupO_some h
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  have h1 := RecIns.setup_oneFull hset
  rw [hinner] at h1
  obtain ⟨hsz, -⟩ := oneFull_ok h1
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hck := checks_stmt hC 0 (by omega)
  have hname := ResLower.checkStmt_name hck
  obtain ⟨⟨nm, hf⟩, coins, hI⟩ := ResLower.insO_tags hins
  have hf' := hf
  unfold Flock.Tags.find? at hf'
  have hnm : nm = Flock.Tags.circuit.name := by
    have h2 : i.base.stmtTags.name = nm := by simpa using List.find?_some hf'
    rw [← h2, hname]
  subst hnm
  rw [FlockSoundness.Discharge.Recursion.find?_circuit] at hf
  have hT : i.base.stmtTags = Flock.Tags.circuit := (Option.some.inj hf).symm
  have hI' : i.base.read.I.tags.hiddenOutputs = true := by
    rw [hI, hT]
    unfold Flock.Tags.withCoins
    split <;> rfl
  have hh : i.base.read.tags.hiddenOutputs = true := by
    unfold Flock.Readable.Read.tags
    split
    · rw [FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs]
      exact hI'
    · exact hI'
  obtain ⟨j, f, sv, hj, -, hc⟩ := checkStmt_circuit hck
  cases hj
  obtain ⟨hck', -⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hc
  obtain ⟨ss, hss, hs, -, hpop, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck'
  have hread := RecIns.setup_read hset
  unfold Flock.Readable.Read.setup at hread
  rcases setupTables_hidden hh hread with ⟨t, ht, hspec, hsts⟩ | ⟨hc1, hc2⟩
  swap
  · exfalso
    omega
  subst hspec
  subst hsts
  have hpop' : RecVStar.population t = sv.instances := by first | exact hpop | simpa using hpop
  obtain ⟨pi, pop, A, B, hpar, hpi, hload, hnone, hsome, hn, hm, hmP, hA, hB, hreg⟩ := setupHidden_shape ht
  refine ⟨r, t, hr, rfl, rfl, hpar, ?_, ⟨pi, B, hpi, hB, ?_⟩, hm, hmP⟩
  · cases hd : i.base.read.draw with
    | none =>
      have hpub := hnone hd
      obtain ⟨c', hc'⟩ := hmIn_loadPublic_row hload
      have hnil := FlockSoundness.Discharge.Recursion.loadPublic_population hc'
      have hp : RecVStar.population t = t.n := by
        unfold RecVStar.population
        rw [hpub, hnil]
        all_goals rfl
      have e : nOf r s none = some sv.instances := by simp [nOf, hss, hs, Except.toOption]
      rw [e, ← hpop', hp]
    | some d =>
      have hdr := hsome d hd
      unfold HmIn.drawn at hdr
      obtain ⟨D, hD, -, hnD, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdr
      have e : nOf r s (some d) = some D.proved.size := by simp [nOf, hD, Except.toOption]
      rw [e, hn, hnD]
  · rw [hreg, Array.size_append, (FlockSoundness.Discharge.Hidden.regions_get hA).1]

/-! ## The numbers of one table -/

/-- The numbers of a session of one table `t`. -/
noncomputable def numsT (t : Flock.Stmt) : Claims.Recursion.Nums :=
  ⟨FlockDeploy.tableErr true t.spec t, FlockDeploy.tableLen true t, FlockDeploy.tableWeight true t,
    2 + 2 * FlockDeploy.repRounds true t.spec t, Claims.Flock.Numbers.waits, Claims.Flock.Numbers.cap, 0, 0⟩

theorem rounds_one (zk : Bool) (spec : SessionSpec) (t : Flock.Stmt) :
    FlockDeploy.rounds zk spec #[t] = 2 * FlockDeploy.repRounds zk spec t := by
  simp [FlockDeploy.rounds]

theorem numsOf_one {xO : Claims.Msg} {i : RecIns} {t : Flock.Stmt} (h : setupO xO = some (i, t.spec, #[t])) :
    numsOf xO = numsT t := by
  have ht : tablesO xO = [(t.spec, t)] := by
    unfold tablesO
    rw [h]
    all_goals rfl
  have hk : VO.k xO = 2 + FlockDeploy.rounds true t.spec #[t] := by simp only [VO, h]
  unfold numsOf εtableO N₀O weightO numsT
  rw [ht, hk, rounds_one]
  all_goals simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]

/-- One table's numbers read only its `m`, `k_log`, region count and `m_pts`. -/
theorem numsT_congr {t u : Flock.Stmt} (hm : t.m = u.m) (hk : t.c.kLog = u.c.kLog)
    (hr : t.regions.size = u.regions.size) (hp : t.mPts = u.mPts) : numsT t = numsT u := by
  simp only [numsT, FlockDeploy.tableErr, FlockDeploy.tableLen, FlockDeploy.tableWeight, FlockDeploy.repRounds,
    FlockDeploy.acct, Flock.Stmt.spec, hm, hk, hr, hp]

end Verity.Recursive.Numbers

namespace Verity.Recursive

open RecDeploy Verity.Recursive.Numbers

/-- **A staged part's numbers don't read its rounds** (C2): at any two inner rounds where part `s`'s staged statement
sets up, its numbers agree. -/
theorem numsOf_staged_eq {st : Stage} {x : Claims.Msg} {ms ms' : List (Claims.Msg × Claims.Out)} {s : ℕ}
    {o o' : RecIns × Flock.SessionSpec × Array Flock.Stmt}
    (h : setupO (staged st x ms s) = some o) (h' : setupO (staged st x ms' s) = some o') :
    numsOf (staged st x ms s) = numsOf (staged st x ms' s) := by
  obtain ⟨i, spec, sts⟩ := o
  obtain ⟨i', spec', sts'⟩ := o'
  obtain ⟨r, t, hr, rfl, rfl, hpar, hn, ⟨pi, B, hpi, hB, hreg⟩, hm, hmP⟩ := side h
  obtain ⟨r', t', hr', rfl, rfl, hpar', hn', ⟨pi', B', hpi', hB', hreg'⟩, hm', hmP'⟩ := side h'
  obtain rfl : r = r' := Option.some.inj (hr.symm.trans hr')
  obtain ⟨eT, eC, eTb, eD⟩ := reads_eq (setupO_some h).1 (setupO_some h').1
  rw [eT, eC, eTb] at hpar
  have ec : t.c = t'.c := Except.ok.inj (hpar.symm.trans hpar')
  rw [eD] at hn
  have en : t.n = t'.n := Option.some.inj (hn.trans hn'.symm)
  rw [ec] at hpi hB
  obtain rfl : pi = pi' := Except.ok.inj (hpi.symm.trans hpi')
  have eB : B.size = B'.size := by rw [hmIn_regions_size hB, hmIn_regions_size hB']
  rw [numsOf_one h, numsOf_one h']
  apply numsT_congr
  · rw [hm, hm', ec, en]
  · rw [ec]
  · rw [hreg, hreg', ec, eB]
  · rw [hmP, hmP', ec, en]

/-- **Part `s`'s numbers are `nums`'** at any inner rounds where its staged statement sets up. -/
theorem numsOf_staged {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ}
    {o : RecIns × Flock.SessionSpec × Array Flock.Stmt} (h : setupO (staged st x ms s) = some o) :
    numsOf (staged st x ms s) = nums st x s := by
  have hex : ∃ ms, (setupO (staged st x ms s)).isSome := ⟨ms, by simp [h]⟩
  obtain ⟨o', ho'⟩ := Option.isSome_iff_exists.1 (Classical.choose_spec hex)
  unfold nums pick
  rw [dif_pos hex]
  exact numsOf_staged_eq h ho'

/-- **Part `s`'s V\* statement is the staged one** at any inner rounds where it sets up. -/
theorem vstarAt_staged {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ}
    {o : RecIns × Flock.SessionSpec × Array Flock.Stmt} (h : setupO (staged st x ms s) = some o) :
    vstarAt st x ms s = staged st x ms s := by
  unfold vstarAt
  rw [if_pos (numsOf_staged h)]

end Verity.Recursive
