module

public import Proofs.Properties.Verity.Sampled.HonestSession.LiveFacts
public import Proofs.Properties.Verity.Sampled.HonestSession.Setup
public import Proofs.Properties.Verity.Sampled.HonestSession.Scope
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Sound

/-!
# The honest session

The session v2 sets up on `θ₀` and the registered files: each of `Live`'s binders the value the pieces name (the circuit
`cH`, the public file as each side reads it, the descriptor, query and law), the reference table at `draw₀`
(`LiveFacts.y₀_of_setup` at `Setup`'s setup), the device's deployed bytes, and the honest bound (`k = 2^23`,
`ρ = 3/8`). Its fields' facts are `exists_live`'s, read off by `live_*`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open Flock
open FlockSoundness.Discharge

/-- The descriptor the circuit file parses to. -/
def D : Flock.Desc := .single cH

theorem desc_ok : Flock.Desc.parse (ZkExec.zkTags inputs) inputs.circuitFile inputs.tables = .ok D := by
  show Flock.Desc.parse tagsZ circuitFile [] = .ok (.single cH)
  unfold Flock.Desc.parse
  rw [CircuitFile.parse_ok] <;> rfl

theorem nTab_inputs : Composed.nTab inputs = 1 := rfl

theorem proved_size (D' : Flock.Draw.UnitDraw) (h : Flock.Draw.UnitDraw.ofJson draw₀ = .ok D') :
    D'.proved.size = Flock.Bridge.lawK (Flock.Bridge.designAt desc query law none nUnits) := by
  rw [Setup.draw_ofJson] at h
  cases h
  rw [Program.design_eq] <;> rfl

/-- The honest session supplies every field of a `Live`. -/
theorem exists_live : ∃ A : Live,
    A.I = inputs ∧ A.n = nUnits ∧ A.desc = desc ∧ A.qr = query ∧ A.lj = law ∧ A.tb = none ∧ A.own = none ∧
      A.Lb = deployedLb ∧ A.ε = 0 ∧ A.OpensReady := by
  have hD := desc_ok
  have ht : inputs.tags.typed = false := typed_false
  have hU : FlockSoundness.Built D.unroll.unit :=
    Hidden.built_of_parseStmt (Flock.Desc.parseStmt_of_parse hD) ((ZkExec.zkTags_typed inputs).trans ht)
  obtain ⟨hpi, pub, hpub, hdup, hn⟩ :=
    LiveFacts.pub_of_setup (I := inputs) (D := D) (pubO := pubO) (n := nUnits) rfl hD Setup.setupHidden_ok
      PublicFile.hmOut_loadPublic PublicFile.pubO_n Scope.ports_ok
  obtain ⟨S₀, d₀, hy⟩ :=
    LiveFacts.y₀_of_setup nUnits (I := inputs) (D := D)
      (kd := Flock.Bridge.lawK (Flock.Bridge.designAt desc query law none nUnits)) rfl ht hD proved_size
      Setup.setupHidden_ok Setup.shapeOk_ok
  let A : Live :=
    { n := nUnits
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
      hpubO := PublicFile.hmOut_loadPublic
      hscope := Scope.scopeOkH
      hdup
      hn
      hnO := PublicFile.pubO_n
      desc
      qr := query
      hpop := Program.popMatches
      lj := law
      tb := none
      hsub := by rw [Program.design_eq] <;> rfl
      hdiv := by rw [nTab_inputs]; exact Nat.mod_one _
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
      hr := fun S R j => LiveFacts.hr_planZHJP _ _ Instance.Honest.kρ S R j
      ε := 0
      v := 0
      w := 0
      h := 0 }
  exact ⟨A, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨Scope.ready⟩⟩

/-- The honest session. -/
noncomputable def live : Live := Classical.choose exists_live

theorem live_inputs : live.I = inputs := (Classical.choose_spec exists_live).1
theorem live_n : live.n = nUnits := (Classical.choose_spec exists_live).2.1
theorem live_desc : live.desc = desc := (Classical.choose_spec exists_live).2.2.1
theorem live_query : live.qr = query := (Classical.choose_spec exists_live).2.2.2.1
theorem live_law : live.lj = law := (Classical.choose_spec exists_live).2.2.2.2.1
theorem live_table : live.tb = none := (Classical.choose_spec exists_live).2.2.2.2.2.1
theorem live_own : live.own = none := (Classical.choose_spec exists_live).2.2.2.2.2.2.1
theorem live_lb : live.Lb = deployedLb := (Classical.choose_spec exists_live).2.2.2.2.2.2.2.1
theorem live_epsilon : live.ε = 0 := (Classical.choose_spec exists_live).2.2.2.2.2.2.2.2.1
theorem live_ready : live.OpensReady := (Classical.choose_spec exists_live).2.2.2.2.2.2.2.2.2

end Verity.Sampled.Instance.HonestSession
