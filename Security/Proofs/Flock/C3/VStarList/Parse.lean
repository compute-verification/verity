import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# Piece (h): V*'s files parse

What `vstarParses_of_list` rests on. `circuitFile_parses` is open: it is the write-then-parse fact of V*'s circuit
writer (`RecVStar.circuitFile`, `VStar.emitCircuit`) against the verifier's parser (`HmRow.parse` and its checks) and
`Layout.scopeOk`, at every statement the verifier lists and every binding. `built_of_parse` is proved: a parsed
circuit's unit is a net `Net.ofRows` built.
-/

namespace FlockC3.VStarList

/-- **V*'s file for a listed statement parses, and its circuit is in scope** (open, circuits' write-then-parse fact):
at an inner shape the verifier lists statements for, the file `circuitFile` writes for one of them under any bindings
parses as `VStarParses` reads it (V*'s tags at the file's own mask, one instance per session), and `Layout.scopeOk`
accepts the circuit. -/
theorem circuitFile_parses {ish : Flock.RecVStar.Inner} {ss : Array VStar.Stmt} {s : VStar.Stmt}
    {b : VStar.Bindings} {cf : ByteArray} (hss : Flock.RecVStar.statements ish = .ok ss) (hs : s ∈ ss)
    (hcf : Flock.RecVStar.circuitFile s b = .ok cf) :
    ∃ c, Flock.HmRow.parse (Flock.Zk.tags (Flock.Tags.circuit.withCoins
        (.tree (Flock.Zk.coinSpec #[Flock.Tags.circuit.table]))) (Flock.Zk.maskLogOf cf) 1) cf [] = .ok c ∧
      FlockSoundness.Discharge.Layout.scopeOk c = true := by
  sorry

/-- **A parsed circuit's unit is built**: `HmRow.parse`'s nets are `Net.ofRows`'s or lookup slots, and its check
refuses a lookup slot as the unit. -/
theorem built_of_parse {tags : Flock.Tags} {cf : ByteArray} {c : Flock.Circuit}
    (h : Flock.HmRow.parse tags cf [] = .ok c) : FlockSoundness.Built c.unit := by
  have P := FlockSoundness.parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  exact FlockSoundness.unit_built P (FlockSoundness.check_facts P.typed hls)

end FlockC3.VStarList
