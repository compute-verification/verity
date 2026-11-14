module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Inputs
public import Proofs.Properties.Verity.Sampled.HonestSession.Hex
public import Proofs.Flock.Recursive.FirewallJson

/-!
# The circuit file parses

`HmOut.parse` at `verify --zk`'s tags reads the circuit file to `cH` (`parse_ok`), whatever the fixed nets' content: the
walk is `Proofs.Flock.SessionLoop.Concrete.Fwd`'s, at the hidden-output tags (`Tags.circuit`) and this file's META, with
the fixed nets as parameters and `Properties.Flock.Nets.fixedNetsBuild` at them. The circuit's fields the other pieces
read are named here; a piece that needs another field asks for it here.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.CircuitFile

open Flock

theorem maskLog : Zk.maskLogOf circuitFile = 14 := by
  sorry -- piece P1 (circuit file)

theorem parse_ok : HmOut.parse tagsZ circuitFile [] = .ok cH := by
  sorry -- piece P1 (circuit file)

theorem parseCommit_ok : HmOut.parseCommit tagsZ circuitFile [] = .ok cH := by
  sorry -- piece P1 (circuit file)

theorem hmIn_parse : HmIn.parse cH = .ok {} := by
  sorry -- piece P1 (circuit file)

theorem cH_sha : cH.sha = circuitSha := by
  sorry -- piece P1 (circuit file)

theorem cH_classSha : cH.classSha = classSha := by
  sorry -- piece P1 (circuit file)

theorem cH_ports : cH.ports = #[yPort] := by
  sorry -- piece P1 (circuit file)

theorem cH_outRowPorts : cH.outRowPorts = #[1] := by
  sorry -- piece P1 (circuit file)

theorem cH_g : cH.g = 1 := by
  sorry -- piece P1 (circuit file)

theorem cH_kLog : cH.kLog = 22 := by
  sorry -- piece P1 (circuit file)

theorem cH_metaJ : cH.metaJ = FlockSoundness.Discharge.Recursion.FirewallJson.norm metaJ := by
  sorry -- piece P1 (circuit file)

end Verity.Sampled.Instance.HonestSession.CircuitFile
