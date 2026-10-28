import Proofs.Flock.Claims.KS.ZkDrawJ
import Proofs.Flock.Claims.KS.ZkAccJ
import Proofs.Flock.Claims.KS.PlainWrP

/-!
# The deployed `--zk` decoders at the masked statement

`ZkLiveAccAtJ`'s decoders (`hdec`) at the deployed statement, from `ZkPlanJ`, `ZkPlanHJ` and `ZkDrawJ`: where the
statement parses in scope and `Deploy.V` sets it up at a draw, the old `--zk` openings and decoder, at public or hidden
outputs, are right at C-Flock ZK's padded tables (`zkTabs`) on the outcomes `verify --zk` accepts at the setup at that
draw at `wrS`.

* `acc_draw_of_setup`: wherever some run accepts, the statement carries a draw at the inputs `Deploy.V` reads of it
  (`acc_draw`, the checks refusing a session without one).
* `setupAt_of_setupOf`: `Deploy.V`'s setup of the statement is `verify --zk`'s at its draw (`setupAt`), with no
  `--public-inputs` (`InScope` refuses them, `publicInputs_none_of_inScope`).
* `tags_of_inScope`: the scope gives `verify --zk`'s inputs untyped tags with retained rounds and `hm96-sha512/v1`
  leaves (`Read.form`).
* `decodesTable_deploy`: the deployed decoders `opnD` and `decD` are right at `zkTabs` (`decodesTable_zkTabs` at
  public outputs, `decodesTableH_zkTabs` at hidden outputs, flat or templated).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (outJ repJ)
open FlockSoundness.Discharge.Tables (DecodesTable)

local notation "𝔽" => FlockLevel3.GF128

namespace Gen

variable [S : ZkSetup]

/-- **A statement some run accepts carries a draw**, at the inputs `Deploy.V` reads of it (`acc_draw`). -/
theorem acc_draw_of_setup {zk : Bool} {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts)) :
    ∃ d, ins.read.draw = some d := by
  obtain ⟨ins', spec', sts', d, hset', hd⟩ := acc_draw hacc
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj (hset'.symm.trans hset))
  exact ⟨d, hd⟩

end Gen

/-- **A statement some run accepts carries a draw**, at the inputs `Deploy.V` reads of it (`acc_draw`). -/
theorem acc_draw_of_setup {zk : Bool} {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts)) :
    ∃ d, ins.read.draw = some d :=
  Gen.acc_draw_of_setup (S := S₀) (zk := zk) (x := x) (hacc := hacc) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset)

namespace Gen

variable [S : ZkSetup]

/-- **`Deploy.V`'s setup of a masked statement in scope is `verify --zk`'s at its draw**: `Read.setup` under `--zk`
reads `zkTags` and `zkCoins` (`zkInputs`), and the scope refuses `--public-inputs`. -/
theorem setupAt_of_setupOf {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) :
    ins.read.zk = true ∧ setupAt (ZkExec.zkInputs ins.read.I) ins.read.draw = .ok (spec, sts) := by
  obtain ⟨hins, hsetup⟩ := setupOf_some hset
  have hzk : ins.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hpi := Claims.Bridge.Custody.publicInputs_none_of_inScope hs hin hins
  refine ⟨hzk, ?_⟩
  rw [← Ins.setup_read hsetup]
  simp only [setupAt, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, ZkExec.zkInputs, hzk, hpi,
    ite_true]

end Gen

/-- **`Deploy.V`'s setup of a masked statement in scope is `verify --zk`'s at its draw**: `Read.setup` under `--zk`
reads `zkTags` and `zkCoins` (`zkInputs`), and the scope refuses `--public-inputs`. -/
theorem setupAt_of_setupOf {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) :
    ins.read.zk = true ∧ setupAt (ZkExec.zkInputs ins.read.I) ins.read.draw = .ok (spec, sts) :=
  Gen.setupAt_of_setupOf (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset)

namespace Gen

variable [S : ZkSetup]

/-- **The scope fixes `verify --zk`'s tags**: at a masked statement in scope, the inputs `Deploy.V` reads are untyped,
with retained rounds and `hm96-sha512/v1` leaves (`InScope`'s `form`, which is `Read.form` at `zkTags`). -/
theorem tags_of_inScope {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) :
    ins.read.I.tags.typed = false ∧ ins.read.I.tags.retainRounds = true ∧
      ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" := by
  have hzk := (setupAt_of_setupOf hs hin hset).1
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨-, hpr, -⟩ := parseBytes_setup hp hset
  obtain ⟨-, -, -, -, -, ht, hrr, hl, -⟩ := hin
  replace hl := hl.resolve_right fun h => h.1 rfl
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at ht hrr hl
  have htags : p.toRead.tags = ZkExec.zkTags ins.read.I := by
    rw [hpr]
    simp [Readable.Read.tags, hzk]
  rw [htags] at ht hrr hl
  exact ⟨(ZkExec.zkTags_typed _).symm.trans ht, (ZkExec.zkTags_retainRounds _).symm.trans hrr,
    (ZkExec.zkTags_merkleLeaf _).symm.trans hl⟩

end Gen

/-- **The scope fixes `verify --zk`'s tags**: at a masked statement in scope, the inputs `Deploy.V` reads are untyped,
with retained rounds and `hm96-sha512/v1` leaves (`InScope`'s `form`, which is `Read.form` at `zkTags`). -/
theorem tags_of_inScope {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) :
    ins.read.I.tags.typed = false ∧ ins.read.I.tags.retainRounds = true ∧
      ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" :=
  Gen.tags_of_inScope (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset)

/-- **The deployed `--zk` decoders are right at `zkTabs` at a masked statement in scope**: where the statement parses
in scope, `Deploy.V` sets it up at the draw `d` and at least one table, at every layout and link layout at the
session's link points, the openings `opnD` and decoder `decD` are right at `zkTabs` on the outcomes `verify --zk`
accepts at the setup at `d` at `wrS` (`DecodesTable`, `ZkLiveAccAtJ`'s `hdec` at the verdict `accepted`): at public
outputs `decodesTable_zkTabs` under `TagsOkZJ`, at hidden outputs `decodesTableH_zkTabs` under `TagsOkZHJ`, every
table passing the shape check (`setup_shapeOk`). -/
theorem decodesTable_deploy {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d)
    (i : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (fun _ o => o.accepted = true)
      (fun o => Composed.AcceptsAt ins.read.I d (Claims.Bridge.Custody.wrS ins.read.I d o))
      (opnD sts (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (decD (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) i := by
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hset]
    simp
  revert i
  rw [hlen]
  obtain ⟨hzk, hsA⟩ := setupAt_of_setupOf hs hin hset
  rw [hd] at hsA
  obtain ⟨ht, hrr, hl⟩ := tags_of_inScope hs hin hset
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact decodesTable_zkTabs ht ⟨hh, hrr, hl⟩ (fun _ : Finset (Fin 0) => d) hsA h0 L hz hk hlay
  · have hshape := Claims.Bridge.Custody.setup_shapeOk (setupOf_some hset).2 hzk
    exact decodesTableH_zkTabs ht ⟨hh, hrr, hl⟩ (fun _ : Finset (Fin 0) => d) hsA hshape h0 L hz hk hlay

end FlockDeploy.KS
