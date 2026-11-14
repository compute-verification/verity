module

public import Proofs.Properties.Verity.Sampled.HonestSession.LiveFacts
public import Proofs.Flock.Soundness.Discharge.Hidden.Built
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Sound

/-!
# The honest sampled session

The concrete accepted transcript and its registered files determine a `Live` session. All parser and file facts below
are closed computations checked by Lean's kernel; structural fields follow from the compiled verifier's setup.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open FlockSoundness.Discharge

/-- The registered Program's descriptor bytes. -/
def desc : Claims.Msg := programFile.data.toList

/-- The query named by the registered Program and partition. -/
def query : Json :=
  jsonOfText """{"name":"Q_template_instance","params":{"template":"RoPEHead_v1{D=64}"},"version":0}"""

/-- The subset sampling design named by the parameters. -/
def law : Json := jsonOfText """{"k":16,"law":"subset"}"""

/-- The accepted setup consists of one table, as the one-stage statement requires. -/
theorem setup_size : setup.2.size = 1 :=
  (ZkHidden.setupOf_tableJHP (I := inputs) (own := none) (d := draw)
    (by decide) record_draw setup_ok).1

/-- The concrete table is the hidden statement at the transcript's recorded draw. -/
theorem table_setup :
    Flock.Stmt.setupHidden (Composed.inputsJ inputs 0).tags
      (Composed.inputsJ inputs 0).circuitFile (Composed.inputsJ inputs 0).publicFile
      (Composed.inputsJ inputs 0).coins (Composed.inputsJ inputs 0).tables
      (Exec.recordDraw transcript.record) (Composed.inputsJ inputs 0).partition
      (Composed.inputsJ inputs 0).program none = .ok setup.2[0] := by
  have h := (ZkHidden.setupOf_tableJHP (I := inputs) (own := none) (d := draw)
    (by decide) record_draw setup_ok).2 0 (by rw [setup_size]; decide)
  simpa [inputs] using h

/-- The concrete table passes the ZK shape check. -/
theorem table_shape : Flock.Zk.shapeOk setup.2[0] = .ok () :=
  ZkExec.array_forM_ok shape_ok 0 (by rw [setup_size]; decide)

/-- The descriptor parsed from the registered circuit. -/
def D : Flock.Desc := .single setup.2[0].c

theorem desc_ok :
    Flock.Desc.parse (ZkExec.zkTags inputs) inputs.circuitFile inputs.tables = .ok D := by
  have hp := HiddenExec.setupHidden_parseCP
    (tags := (Composed.inputsJ inputs 0).tags) (by decide) table_setup
  have ht :
      (Composed.inputsJ inputs 0).tags = ZkExec.zkTags inputs := by
    simp [Composed.inputsJ, Exec.tagsAt, inputs]
  rw [ht] at hp
  exact Flock.Desc.parse_single hp

/-- The registered public file as the circuit reads it. -/
def publicResult : Flock.V Flock.Public :=
  Flock.HmOut.loadPublic (ZkExec.zkTags inputs) D.unroll inputs.publicFile

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
theorem public_some : publicResult.toOption.isSome = true := by decide +kernel

def pubO : Flock.Public := publicResult.toOption.get public_some

theorem public_ok : publicResult = .ok pubO := by
  have hps := public_some
  generalize h : publicResult = r at hps ⊢
  cases r <;> simp_all [pubO]

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- The circuit and public-file checks needed by `Live`, as closed kernel computations. -/
theorem file_checks :
    pubO.n = 32 ∧
    Layout.scopeOkH D.unroll = true ∧
    Flock.PortRoots.checkOf (.ok D.unroll) (ZkExec.zkTags inputs) inputs.publicFile = .ok () ∧
    Flock.PortRoots.checkOf
      (Flock.ProvedScope.circuitOf (ZkExec.zkTags inputs) inputs.circuitFile inputs.tables)
      (ZkExec.zkTags inputs) inputs.publicFile = .ok () := by
  decide +kernel

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- The registered Program's query has the fixture's 32 proof blocks. -/
theorem population_matches : Flock.Bridge.PopMatches desc query 32 := by
  decide +kernel

/-- The sampling design read from the registered data. -/
def design : Flock.Draw.Law := Flock.Bridge.designAt desc query law none 32

set_option maxHeartbeats 0 in
theorem design_is_subset :
    (match design with | .subset _ => true | _ => false) = true := by
  decide +kernel

/-- The registered sampling design is subset sampling. -/
theorem subset_design :
    Flock.Bridge.designAt desc query law none 32 =
      .subset (Flock.Bridge.lawK (Flock.Bridge.designAt desc query law none 32)) := by
  change design = .subset (Flock.Bridge.lawK design)
  cases h : design <;> simp_all [Flock.Bridge.lawK]

/-- The transcript records the concrete ascending subset draw. -/
theorem record_drawn :
    recordDrawn (.subset units.size) 32 units transcript.record := by
  have h := ofJson_subset_eq (Claims.Bridge.Custody.recordDraw_norm record_draw) draw_decodes unit_law
  constructor
  · simpa [units, unit_population] using h.1
  · rw [← exec_recordDraw, record_draw]
    apply congrArg some
    simpa [units, unit_population] using h.2

/-- The concrete accepted setup supplies every field of a `Live` session. -/
theorem exists_live : ∃ A : Live,
    A.I = inputs ∧ A.n = 32 ∧ A.desc = desc ∧ A.qr = query ∧ A.lj = law ∧ A.tb = none ∧
      A.Lb = deployedLb ∧ A.ε = 0 ∧ A.OpensReady := by
  have hD := desc_ok
  have hpubO := public_ok
  have hnO := file_checks.1
  have hscope := file_checks.2.1
  have hports := file_checks.2.2.1
  have hready := file_checks.2.2.2
  have ht : inputs.tags.typed = false := by decide
  have hU : FlockSoundness.Built D.unroll.unit :=
    Hidden.built_of_parseStmt (Flock.Desc.parseStmt_of_parse hD) ((ZkExec.zkTags_typed inputs).trans ht)
  obtain ⟨hpi, pub, hpub, hdup, hn⟩ :=
    pub_of_setup (I := inputs) (D := D) (pubO := pubO) (n := 32) rfl hD
      (by rw [record_draw]; exact table_setup) hpubO hnO hports
  obtain ⟨S₀, d₀, hy⟩ :=
    y₀_of_setup 32 (I := inputs) (D := D) (kd := 16) (n' := 32) (units := units)
      rfl ht hD record_drawn (by decide) table_setup table_shape
  have hnTab : Composed.nTab inputs = 1 := by simp [Composed.nTab, inputs]
  let A : Live :=
    { n := 32
      I := inputs
      own := none
      ht
      D
      hD
      hU
      hpi
      pub
      hpub
      pubO
      hpubO
      hscope
      hdup
      hn
      hnO
      desc
      qr := query
      hpop := population_matches
      lj := law
      tb := none
      hsub := subset_design
      hdiv := by rw [hnTab]; exact Nat.mod_one _
      S₀
      d₀
      y₀ := Classical.choice hy
      Ω := DeployedCoins
      fΩ := inferInstance
      nΩ := inferInstance
      Lb := deployedLb
      src := deployedSrc
      hA3 := uniformRandomBytes_deployed
      hTags := by decide
      k := Instance.Honest.k
      Rw := 1
      M := 1
      hk := Instance.Honest.one_le_k
      hRw := Nat.le_refl 1
      hM := Nat.le_refl 1
      ρ := Instance.Honest.ρ
      hρ := Instance.Honest.ρ_lt_one
      hr := fun S R j => hr_planZHJP _ _ Instance.Honest.kρ S R j
      ε := 0
      v := 0
      w := 0
      h := 0 }
  exact ⟨A, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨hready⟩⟩

/-- The deployed honest sampled session. -/
noncomputable def live : Live := Classical.choose exists_live

theorem live_inputs : live.I = inputs := (Classical.choose_spec exists_live).1
theorem live_n : live.n = 32 := (Classical.choose_spec exists_live).2.1
theorem live_desc : live.desc = desc := (Classical.choose_spec exists_live).2.2.1
theorem live_query : live.qr = query := (Classical.choose_spec exists_live).2.2.2.1
theorem live_law : live.lj = law := (Classical.choose_spec exists_live).2.2.2.2.1
theorem live_table : live.tb = none := (Classical.choose_spec exists_live).2.2.2.2.2.1
theorem live_lb : live.Lb = deployedLb := (Classical.choose_spec exists_live).2.2.2.2.2.2.1
theorem live_epsilon : live.ε = 0 := (Classical.choose_spec exists_live).2.2.2.2.2.2.2.1
theorem live_ready : live.OpensReady := (Classical.choose_spec exists_live).2.2.2.2.2.2.2.2

end Verity.Sampled.Instance.HonestSession
