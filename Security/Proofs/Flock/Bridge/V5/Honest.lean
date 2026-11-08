module

public import Proofs.Flock.Bridge.Assumptions.ServedV5
public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.Deployed
public import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# `SampledKSLiveV5`'s file hypotheses hold at the served registered session

`SampledKSLiveV5` says something only if its hypotheses on the files can hold. `Flock.Guarantees.SampledKSLiveV5Honest`
concludes them at the served registered session's files, from the one named premise that those files parse at public
inputs (`Flock.Bridge.Assumptions.ServedFilesParseV5`): inputs `I` whose files have the served SHA-512 digests, at
`verity/flock-circuit`'s tags with one table and no session tables, at which the circuit file parses to `D`, its public
input ports `pi` parse and the public file loads over them, `D` is in `scopeOkH` and has
the row layout it needs (`RowsOk`), no two of its ports share a name, every sample size divides into the session's tables,
and the tags are the hidden-output ones at which the statement gives its bound (`TagsOkZHJ`). The registration's
population (`PopMatches`) is a statement about the registration, not the files, and is no conjunct here; the statement takes no hypothesis on the
sampling design it samples under (it is `designAt`'s, never a Bernoulli law) or on the registration's roots (its `ecr/sha-512` term at the roots finder pays for them), and the finders' collision resistance
is no hypothesis either: the statement takes it as `TableCRZ` and `LinkCRZC`.
-/

@[expose] public section

namespace Flock.Bridge

/-- **`verify --zk`'s inputs for one session of the files `cf` and `pf`**: `verity/flock-circuit`'s tags, one table, no
session tables, on the coin tree `verify --zk` samples them on. The partition and the program it also reads are none here:
no hypothesis on the files reads them. -/
def servedIV5 (cf pf : ByteArray) : FlockSoundness.Discharge.Exec.Inputs :=
  { tags := Flock.Tags.circuit, circuitFile := cf, publicFile := pf,
    coins := .tree (Flock.Zk.coinSpec (Flock.Tags.circuit.tableNames 1)), count := 1, tables := [], partition := none,
    program := none, work := none, stratified := none, closure := none }

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.Composed

/-- **`SampledKSLiveV5`'s file hypotheses hold at the served registered session.** Given that the served files parse at
public inputs (`Bridge.Assumptions.ServedFilesParseV5`), there are inputs `I`, a descriptor `D`, public input ports `pi`
and a public file `pub` at which every hypothesis `SampledKSLiveV5` takes of the files holds: `I`'s files are the served
ones (their SHA-512 digests), its tags `verity/flock-circuit`'s with one table and no session tables, untyped, the
circuit file parses to `D`, `pi` parses at `D` and the public file loads over it, `D` is in
`scopeOkH` with `RowsOk` and no duplicate port name, every sample size is a multiple of the session's tables, and the tags
are hidden-output ones (`TagsOkZHJ`), the case in which `SampledKSLiveV5` gives its bound. -/
def SampledKSLiveV5Honest : Prop :=
  Bridge.Assumptions.ServedFilesParseV5 →
    ∃ (I : Inputs) (D : Flock.Desc) (pi : Flock.HmIn.Ports) (pub : Flock.Public),
      I.tags = Flock.Tags.circuit ∧ I.count = 1 ∧ I.tables = [] ∧
      Flock.hexOf (Flock.Sha512.hash I.circuitFile) =
        "7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0" ∧
      Flock.hexOf (Flock.Sha512.hash I.publicFile) =
        "6bea4647565a87a873bc66670482ec0cdb86c4424d061ccb03354b81ec55cbe302564836f08c4a9757085746f6000b904657749c2f84b028fe56fde816c26dd1" ∧
      I.tags.typed = false ∧
      Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D ∧
      Flock.HmIn.parse D.unroll = .ok pi ∧
      Flock.HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub ∧
      Discharge.Layout.scopeOkH D.unroll = true ∧ Bridge.RowsOk D.unroll ∧
      Flock.Registered.dupName D.unroll = none ∧
      (∀ kd : ℕ, kd % nTab I = 0) ∧ TagsOkZHJ I

end Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge.ZkExec in
/-- **`SampledKSLiveV5`'s file hypotheses hold at the served registered session**
(`Flock.Guarantees.SampledKSLiveV5Honest`): the verifier's `scopeOkH` is the analysis's (`scopeOkH_eq`), which carries
`RowsOk` (`rowsOk_of_scopeOkH`), and one table divides every sample size. -/
theorem Flock.SecurityProofs.SampledKSLiveV5Honest : Flock.Guarantees.SampledKSLiveV5Honest := by
  rintro ⟨cf, pf, hcf, hpf, D, hD, hsc, hdup, pi, hpi, pub, hpub⟩
  have hD' : Flock.Desc.parse (zkTags (Flock.Bridge.servedIV5 cf pf)) cf [] = .ok D := hD
  have hpub' : Flock.HmIn.loadPublic (zkTags (Flock.Bridge.servedIV5 cf pf)) D.unroll pi pf = .ok pub := hpub
  have ht : (Flock.Bridge.servedIV5 cf pf).tags.typed = false := rfl
  have hsc' : Discharge.Layout.scopeOkH D.unroll = true := by
    rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
    exact hsc
  refine ⟨Flock.Bridge.servedIV5 cf pf, D, pi, pub, rfl, rfl, rfl, hcf, hpf, ht, Flock.Desc.parseStmt_of_parse hD', hpi, hpub', hsc',
    Flock.Bridge.rowsOk_of_scopeOkH hsc', hdup, ?_, rfl, rfl, rfl⟩
  · intro kd
    have h1 : FlockSoundness.Discharge.Composed.nTab (Flock.Bridge.servedIV5 cf pf) = 1 := by
      simp [FlockSoundness.Discharge.Composed.nTab, Flock.Bridge.servedIV5]
    rw [h1]
    exact Nat.mod_one kd
