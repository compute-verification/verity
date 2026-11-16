module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Inputs
public import Proofs.Properties.Verity.Sampled.HonestSession.Hex
public import Proofs.Flock.Recursive.FirewallJson

/-!
# The circuit file parses

`HmOut.parse` at `verify --zk`'s tags reads the circuit file to `cH`, whatever the fixed nets' content: the walk is
`Proofs.Flock.SessionLoop.Concrete.Fwd`'s, at the hidden-output tags (`Tags.circuit`) and this file's META, with the fixed
nets as parameters and `Properties.Flock.Nets.fixedNetsBuild` at them. The facts the other pieces read of `cH` are the
named proposition `CircuitFileFacts`, which is not proved: the pieces that use it take it as a hypothesis.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Flock

/-- **Facts about the concrete circuit file, not proved**: `HmOut.parse` and `parseCommit` at `verify --zk`'s tags read it
to `cH`, `HmIn.parse` finds no public input port, `cH` has the one 64-word output row, its META is `norm metaJ`, it has one
unit per VU and pins the dummy row (`dummyBc`, output `0`), and its unit net reads only its leaf group, returns the outputs,
has at most `2^32` rows and reads only port bits. -/
structure CircuitFileFacts : Prop where
  /-- The zero-knowledge mask size the file's META states is `14`. -/
  maskLog : Zk.maskLogOf circuitFile = 14
  /-- `HmOut.parse` reads the file to `cH`. -/
  parse_ok : HmOut.parse tagsZ circuitFile [] = .ok cH
  /-- `HmOut.parseCommit` reads the file to `cH`. -/
  parseCommit_ok : HmOut.parseCommit tagsZ circuitFile [] = .ok cH
  /-- `cH` names no public input port. -/
  hmIn_parse : HmIn.parse cH = .ok {}
  /-- `cH` carries the file's SHA-512, `circuitSha`. -/
  cH_sha : cH.sha = circuitSha
  /-- `cH` carries the class SHA-512, `classSha`. -/
  cH_classSha : cH.classSha = classSha
  /-- `cH`'s one port is `yPort`. -/
  cH_ports : cH.ports = #[yPort]
  /-- `cH`'s one output row holds one port. -/
  cH_outRowPorts : cH.outRowPorts = #[1]
  /-- `cH` has one gate group. -/
  cH_g : cH.g = 1
  /-- `cH`'s `k_log` is `22`. -/
  cH_kLog : cH.kLog = 22
  /-- `cH`'s META is the parser's reading of `metaJ`. -/
  cH_metaJ : cH.metaJ = FlockSoundness.Discharge.Recursion.FirewallJson.norm metaJ
  /-- `cH` has one unit per VU. -/
  cH_upv : cH.upv = 1
  /-- `cH` pins the dummy row's leaf string `dummyBc`. -/
  cH_dummyDigests : cH.dummyDigests = #[dummyBc]
  /-- `cH` pins the dummy output word `0`. -/
  cH_dummyOutputs : cH.dummyOutputs = #[0]
  /-- `cH`'s unit reads one input group. -/
  unit_inGroups : cH.unit.inGroups.size = 1
  /-- `cH`'s output net is its unit net. -/
  unit_outNet : cH.outNet = cH.unitNet
  /-- `cH`'s unit has at most `2^32` rows. -/
  unit_unitLog : cH.unit.unitLog ≤ 32
  /-- `cH`'s unit reads only port bits among its inputs. -/
  unit_portReads : FlockSoundness.Discharge.Layout.portReadsB cH.unit = true

end Verity.Sampled.Instance.HonestSession
