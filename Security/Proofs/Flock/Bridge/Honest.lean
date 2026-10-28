import Proofs.Flock.Bridge.Assumptions.Served
import Proofs.Flock.Bridge.Public
import Proofs.Flock.C3.VStarList.Parse
import Proofs.Flock.Soundness.Discharge.Hidden.Facts

/-!
# `SampledKSLive`'s file hypotheses hold at the deployed session

`SampledKSLiveV2`, `SampledKSLiveV3` and `SampledKSLivePub` say something only if their hypotheses on the files can
hold. `Flock.Guarantees.SampledKSLiveHonest` concludes them at the deployed session's files, from the one named premise
that those files parse (`Flock.Bridge.Assumptions.ServedFilesParse`): inputs `I` whose files have the deployed SHA-512
digests, at `verity/flock-circuit`'s tags with one table and no session tables, at which the circuit file parses to `D`,
`D`'s unit is built (`Built`), the public file loads with no shared-row tables, `D` is in `scopeOkH`, and the tags are
the ones the statements take (`TagsOkZHJ`). The finders' collision resistance is no hypothesis of these: the statements
take it as the named assumptions `Assumptions.TableCRZAt` and `Assumptions.LinkCRZCAt`.
-/

namespace Flock.Bridge

/-- **`verify --zk`'s inputs for one session of the files `cf` and `pf`**: `verity/flock-circuit`'s tags, one table, no
session tables, on the coin tree `verify --zk` draws them on. The partition and the program it also reads are none here:
no hypothesis on the files reads them. -/
def servedI (cf pf : ByteArray) : FlockSoundness.Discharge.Exec.Inputs :=
  { tags := Flock.Tags.circuit, circuitFile := cf, publicFile := pf,
    coins := .tree (Flock.Zk.coinSpec (Flock.Tags.circuit.tableNames 1)), count := 1, tables := [], partition := none,
    program := none, work := none, stratified := none, closure := none }

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkHidden

/-- **`SampledKSLive`'s file hypotheses hold at the deployed session.** Given that the deployed files parse
(`Bridge.Assumptions.ServedFilesParse`), there are inputs `I`, a descriptor `D` and a public file `pub` at which every
hypothesis `SampledKSLiveV2`, `SampledKSLiveV3` and `SampledKSLivePub` take of the files holds: `I`'s files are the
deployed ones (their SHA-512 digests), its tags `verity/flock-circuit`'s with one table and no session tables, untyped,
the circuit file parses to `D` and `D`'s unit is built, the public file loads at `D` with no shared-row tables, `D` is in
`scopeOkH`, and the tags are hidden-output ones (`TagsOkZHJ`). -/
def SampledKSLiveHonest : Prop :=
  Bridge.Assumptions.ServedFilesParse →
    ∃ (I : Inputs) (D : Flock.Desc) (pub : Flock.Public),
      I.tags = Flock.Tags.circuit ∧ I.count = 1 ∧ I.tables = [] ∧
      Flock.hexOf (Flock.Sha512.hash I.circuitFile) =
        "0c1d6ad42ebf34efa4370690f077ae1302319ba7dbde4f3ac18565ae97c06f11d2633d210e2b9a923f3cb1a398b34d31419b3ffea2d56a820927b0b1ca4db099" ∧
      Flock.hexOf (Flock.Sha512.hash I.publicFile) =
        "56861c780e639f52be33394f86b369de7bec45c3e5290d0150720673ffb6032b465ca3a5cb00953301661aa97a3e7a3e6fe4525651ecee7cc0e7cc13412e9209" ∧
      I.tags.typed = false ∧
      Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D ∧ Built D.unroll.unit ∧
      HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub ∧ pub.tables = none ∧
      Discharge.Layout.scopeOkH D.unroll = true ∧ TagsOkZHJ I

end Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge.ZkExec in
/-- **`SampledKSLive`'s file hypotheses hold at the deployed session** (`Flock.Guarantees.SampledKSLiveHonest`): the
unit is built because `HmRow.parse` built it (`built_of_parse`, through `parse_untyped` at untyped tags), and the
verifier's `scopeOkH` is the analysis's (`scopeOkH_eq`). -/
theorem Flock.SecurityProofs.SampledKSLiveHonest : Flock.Guarantees.SampledKSLiveHonest := by
  rintro ⟨cf, pf, hcf, hpf, D, hD, hsc, pub, hpub, hpt⟩
  have hD' : Flock.Desc.parse (zkTags (Flock.Bridge.servedI cf pf)) cf [] = .ok D := hD
  have hpub' : Flock.HmOut.loadPublic (zkTags (Flock.Bridge.servedI cf pf)) D.unroll pf = .ok pub := hpub
  have ht : (Flock.Bridge.servedI cf pf).tags.typed = false := rfl
  refine ⟨Flock.Bridge.servedI cf pf, D, pub, rfl, rfl, rfl, hcf, hpf, ht, hD', ?_, hpub', hpt, ?_, rfl, rfl, rfl⟩
  · exact FlockC3.VStarList.built_of_parse
      (FlockSoundness.Discharge.Hidden.parse_untyped ((zkTags_typed _).trans ht) (Flock.Desc.parse_flat hD')).1
  · rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
    exact hsc
