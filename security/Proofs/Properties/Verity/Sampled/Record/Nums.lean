module

public import Proofs.Properties.Verity.Sampled.Record
public import Proofs.Properties.Verity.Recursive.BaseSetup
public import Proofs.Properties.Verity.Recursive.Numbers
public import Proofs.Flock.Claims.Sampling.Select.Size
public import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import all Lean.Data.Json.Basic
import Proofs.Flock.Claims.Recursion.ResLowerDeployed

/-!
# The session's numbers at a draw file

The session's statement at a draw file `d` (`sessStmt A o d`) has the statement's numbers (`numsOk`) wherever it sets up
one table of `A.kd` units (`RefOk`, `numsOk_of_refOk`):

* **one table, hidden** (`shape_sess`): the statement is `verity/flock-circuit`, whose outputs are hidden, at one table,
  so its setup is `Stmt.setupHidden`'s one table, whose circuit, regions and `m` read the circuit file, the tables and
  the unit count `n` alone (`Verity.Recursive.Numbers.setupHidden_shape`);
* **the numbers read the table** (`nums_one`): the rounds and `N₀` of a one-table session are its table's, so two draw
  files whose tables have one `n` give one set of numbers (`nums_congr`), the reference draw file's (`d₀`) among them.

And what `RefOk`'s unit count reads at a draw v2 accepts: a draw file's units are its `units` field's
(`unitsAt_size`), a draw by a law `LawOn` holds draws `A.kd` units (`drawn_length`), and a session's draw that
`ProvedScope.drawLaw` keeps proves its drawn units alone (`proved_of_drawLaw`).
-/

@[expose] public section

namespace Verity.Sampled.Record

open Lean (Json)
open Claims (Msg)
open FlockDeploy (Ins)
open Verity.Recursive.BaseSetup (bind_ok)

/-! ## Reading sizes -/

theorem list_mapM_option_length {α β : Type} (f : α → Option β) :
    ∀ (l : List α) (ys : List β), l.mapM f = some ys → ys.length = l.length
  | [], ys, h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | none => rw [ha] at h; cases h
    | some b =>
      rw [ha] at h
      cases hl : l.mapM f with
      | none => rw [hl] at h; cases h
      | some cs =>
        rw [hl] at h
        cases h
        simp [list_mapM_option_length f l cs hl]

theorem list_mapM_except_length {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys → ys.length = l.length
  | [], ys, h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, -, h⟩ := bind_ok h
    obtain ⟨cs, hl, h⟩ := bind_ok h
    cases h
    simp [list_mapM_except_length f l cs hl]

theorem size_mapM_option {α β : Type} {f : α → Option β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = some ys) : ys.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => rw [hl] at h; cases h
  | some l =>
    rw [hl] at h
    cases h
    simp [list_mapM_option_length f _ l hl]

theorem size_mapM_except {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ys.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    cases h
    simp [list_mapM_except_length f _ l hl]

theorem get_ok {j : Json} {k ctx : String} {v : Json} (h : Flock.J.get j k ctx = .ok v) : j.getObjVal? k = .ok v := by
  unfold Flock.J.get at h
  cases hg : j.getObjVal? k with
  | error e => rw [hg] at h; cases h
  | ok w => rw [hg] at h; cases h; rfl

/-- A field `Flock.J.arr` reads as an array is that array, as `Spot.getD` reads it. -/
theorem getD_of_arr {j : Json} {k ctx : String} {a : Array Json} (h : Flock.J.arr j k ctx = .ok a) :
    _root_.Spot.getD j k = .arr a := by
  unfold Flock.J.arr at h
  obtain ⟨v, hv, h⟩ := bind_ok h
  have hg := get_ok hv
  have hva : v = .arr a := by
    try dsimp only at h
    unfold Flock.J.asArr at h
    cases v <;> first
      | (cases h; rfl)
      | cases h
  rw [← hva]
  simp only [_root_.Spot.getD, _root_.Spot.field?, hg, Except.toOption, Option.getD_some]

/-- **A draw file's units, as v2 credits them, are as many as its draw's**: `unitsAt` reads one unit per entry of the
draw's `units` field, which `UnitDraw.ofJson` reads as its units. -/
theorem unitsAt_size {h j : Json} {xs : Array ℕ} {D : Flock.Draw.UnitDraw}
    (hu : _root_.Properties.Spot.Units.unitsAt h j = some xs) (hD : Flock.Draw.UnitDraw.ofJson j = .ok D) :
    xs.size = D.units.size := by
  obtain ⟨⟨a, ha, hm⟩, -, -⟩ := FlockSoundness.Discharge.Composed.ofJson_shape hD
  have hga := getD_of_arr ha
  unfold _root_.Properties.Spot.Units.unitsAt at hu
  obtain ⟨_, -, hu⟩ := Option.bind_eq_some_iff.1 hu
  try dsimp only at hu
  rw [hga] at hu
  try dsimp only at hu
  obtain ⟨us, hus, hX⟩ := Option.bind_eq_some_iff.1 hu
  cases hX
  rw [Verity.QSort.size_qsort, size_mapM_option hus, size_mapM_except hm]

/-- **The coins draw `A.kd` units** where the draw ends: `A`'s law is on its strata and draws `A.kd` (`Live.hlaw`). -/
theorem drawn_length (A : Live) {src : ByteArray} {u : Array ℕ} {rd : ℕ}
    (h : _root_.Spot.Draw.drawUnits A.design A.n src = .ok (u, rd)) : (drawn A src).length = A.kd := by
  unfold drawn
  simp only [h, Array.length_toList]
  exact _root_.Properties.Spot.SecurityProofs.drawUnits_size A.hlaw h

/-- **A draw `ProvedScope.drawLaw` keeps proves its drawn units alone**: it is by a subset or stratified law, whose
draws carry no closure. -/
theorem proved_of_drawLaw {T : Flock.Tags} {J : Json} {D : Flock.Draw.UnitDraw}
    (hD : Flock.Draw.UnitDraw.ofJson (Flock.norm J) = .ok D)
    (h : Flock.ProvedScope.drawLaw true T false (some (Flock.norm J)) = .ok ()) : D.proved = D.units := by
  unfold Flock.ProvedScope.drawLaw at h
  try dsimp only at h
  rw [hD] at h
  try dsimp only at h
  cases hw : Flock.ProvedScope.lawWhy true T false D with
  | some why => rw [hw] at h; cases h
  | none =>
    have hcl : D.closure = #[] := by
      cases hl : D.law with
      | subset k => exact (FlockComplete.ZkTables.good_subset (FlockDeploy.wf_norm J) hD hl).2
      | stratified K St => exact FlockComplete.ZkTables.closure_stratified (FlockDeploy.wf_norm J) hD hl
      | bernoulli a b => simp [Flock.ProvedScope.lawWhy, hl] at hw
      | work K ws => simp [Flock.ProvedScope.lawWhy, hl] at hw
    unfold Flock.Draw.UnitDraw.proved
    rw [hcl]
    rfl

/-! ## The session statement's one table -/

section Sess

variable {A : Live} {o : Own}

/-- The session's statement is `verity/flock-circuit`'s. -/
theorem stmtTags_sess {d : Msg} {s : Ins}
    (hins : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s) :
    s.stmtTags = Flock.Tags.circuit := by
  have hf : Flock.Tags.byName? _root_.Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit :=
    Flock.Tags.byName?_of_find? FlockSoundness.Discharge.Recursion.find?_circuit
  exact Option.some.inj ((RecDeploy.ResLower.raw_ins_tags hins).1.symm.trans hf)

/-- The session's outputs are hidden. -/
theorem hidden_sess {d : Msg} {s : Ins}
    (hins : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s) :
    s.read.tags.hiddenOutputs = true := by
  have hI' : s.read.I.tags.hiddenOutputs = true := by
    rw [(RecDeploy.ResLower.raw_ins_tags hins).2, stmtTags_sess hins]
    all_goals rfl
  unfold Flock.Readable.Read.tags
  split
  · rw [FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs]
    exact hI'
  · exact hI'

/-- **The session's statement at any draw file sets up one hidden table**: its circuit, regions and `m` read the
session's tags, circuit file and tables and its unit count `n`, which is its public file's, the draw's. -/
theorem shape_sess {d : Msg} {s : Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hins : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s)
    (hsu : s.setup = .ok (spec, sts)) :
    ∃ t pi pop B, spec = t.spec ∧ sts = #[t] ∧
      Flock.HmOut.parseCommit s.read.tags s.read.I.circuitFile s.read.I.tables = .ok t.c ∧
      Flock.HmIn.parse t.c = .ok pi ∧ Flock.HmIn.regions t.c pi t.pub = .ok B ∧
      t.regions.size = t.c.ports.size + B.size ∧
      t.m = t.c.kLog + max 3 (Flock.log2ceil (max ((t.n + t.c.g - 1) / t.c.g) 1)) ∧
      (∀ dj, s.read.draw = some dj → Flock.HmIn.drawn t.c pi pop dj = .ok t.pub) ∧ t.n = t.pub.n := by
  have hread := FlockDeploy.Ins.setup_read hsu
  unfold Flock.Readable.Read.setup at hread
  rcases Verity.Recursive.Numbers.setupTables_hidden (hidden_sess hins) hread with ⟨t, ht, hspec, hsts⟩ | ⟨hc1, -⟩
  swap
  · exfalso
    have h1 : s.read.I.count = 1 := (Verity.Recursive.Numbers.raw_ins_files hins).2.1
    omega
  obtain ⟨pi, pop, R, B, hpar, hpi, -, -, hsome, hn, hm, -, hR, hB, hreg⟩ :=
    Verity.Recursive.Numbers.setupHidden_shape ht
  refine ⟨t, pi, pop, B, hspec, hsts, hpar, hpi, hB, ?_, hm, hsome, hn⟩
  rw [hreg, Array.size_append, (FlockSoundness.Discharge.Hidden.regions_get hR).1]

/-- **A one-table session's numbers are its table's.** -/
theorem nums_one {x : Msg} {s : Ins} {t : Flock.Stmt} (h : FlockDeploy.setupOf true x = some (s, t.spec, #[t])) :
    Nums x = (2 + 2 * FlockDeploy.repRounds true t.spec t, FlockDeploy.tableLen true t) := by
  have hk : (FlockDeploy.V true).k x = 2 + FlockDeploy.rounds true t.spec #[t] := by
    simp only [FlockDeploy.V, h]
  have ht : FlockDeploy.tables true x = [(t.spec, t)] := by
    unfold FlockDeploy.tables
    rw [h]
    all_goals rfl
  unfold Nums FlockDeploy.N₀
  rw [hk, ht, Verity.Recursive.Numbers.rounds_one]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]

/-- **The session's numbers read only its table's unit count**: at two draw files whose sessions set up one table each,
of one unit count, the numbers agree. -/
theorem nums_congr {d d' : Msg} {s s' : Ins} {t t' : Flock.Stmt}
    (hins : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s)
    (hset : FlockDeploy.setupOf true (sessStmt A o d) = some (s, t.spec, #[t]))
    (hins' : ({ baseRaw A o with draw := some ⟨d'.toArray⟩ } : FlockDeploy.Raw).ins = .ok s')
    (hset' : FlockDeploy.setupOf true (sessStmt A o d') = some (s', t'.spec, #[t'])) (hn : t.n = t'.n) :
    Nums (sessStmt A o d) = Nums (sessStmt A o d') := by
  rw [nums_one hset, nums_one hset']
  obtain ⟨u, pi, _pop, B, -, hst, hpar, hpi, hB, hreg, hm, -, -⟩ := shape_sess hins (FlockDeploy.setupOf_some hset).2
  obtain ⟨u', pi', _pop', B', -, hst', hpar', hpi', hB', hreg', hm', -, -⟩ :=
    shape_sess hins' (FlockDeploy.setupOf_some hset').2
  have htu : t = u := Array.singleton_inj.1 hst
  have htu' : t' = u' := Array.singleton_inj.1 hst'
  rw [← htu] at hpar hpi hB hreg hm
  rw [← htu'] at hpar' hpi' hB' hreg' hm'
  have eT : s.read.tags = s'.read.tags := by
    rw [Verity.Recursive.Numbers.read_tags_of hins rfl, Verity.Recursive.Numbers.read_tags_of hins' rfl,
      stmtTags_sess hins, stmtTags_sess hins']
  have eC : s.read.I.circuitFile = s'.read.I.circuitFile :=
    (Verity.Recursive.Numbers.raw_ins_files hins).1.trans (Verity.Recursive.Numbers.raw_ins_files hins').1.symm
  have eTb : s.read.I.tables = s'.read.I.tables :=
    (Verity.Recursive.Numbers.raw_ins_files hins).2.2.trans (Verity.Recursive.Numbers.raw_ins_files hins').2.2.symm
  rw [eT, eC, eTb] at hpar
  have ec : t.c = t'.c := Except.ok.inj (hpar.symm.trans hpar')
  rw [ec] at hpi hB
  obtain rfl : pi = pi' := Except.ok.inj (hpi.symm.trans hpi')
  have eB : B.size = B'.size := by
    rw [Verity.Recursive.Numbers.hmIn_regions_size hB, Verity.Recursive.Numbers.hmIn_regions_size hB']
  have em : t.m = t'.m := by rw [hm, hm', ec, hn]
  have ek : t.c.kLog = t'.c.kLog := by rw [ec]
  have er : t.regions.size = t'.regions.size := by rw [hreg, hreg', ec, eB]
  simp only [FlockDeploy.repRounds, FlockDeploy.tableLen, Flock.Stmt.spec, em, ek, er]

/-- **A draw file whose session sets up one table of `A.kd` units has the statement's numbers**: the reference draw
file's session does too (`d₀`), and the numbers read only the unit count (`nums_congr`). -/
theorem numsOk_of_refOk {d : Msg} (h : RefOk A o d) : numsOk A o d := by
  have h0 : RefOk A o (d₀ A o) := Classical.epsilon_spec ⟨d, h⟩
  obtain ⟨s, t, hins, hset, hn⟩ := h
  obtain ⟨s', t', hins', hset', hn'⟩ := h0
  exact nums_congr hins hset hins' hset' (hn.trans hn'.symm)

end Sess

end Verity.Sampled.Record
