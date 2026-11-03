import SampledProofs.Fleet

/-!
# The proof system's halt: detection is terminal (`SampledProofs.Stream.run`, `SampledProofs.Fleet.run`)

The properties of R5-all as the service's Lean of record runs it: when a stream of an operator reaches its failure budget,
the fleet halts the operator (`Fleet.failures`), and each of the operator's streams then takes a `halt` op
(`Stream.halt`), which ends every outcome not yet final as `halted` and keeps those already final. Each is stated against
the function as it runs, for every state and op its decoders produce, and takes no premise.

* `HaltIsFinal`: a halted stream refuses every op `halted` and its state stays as it is, so nothing after its halt is
  accepted, registered or ended.
* `HaltEndsOpen`: a halt the stream accepts names the entries that were open, makes each of them final, leaves none open
  and changes nothing else the stream counts: its size, failures, salt and end.
* `NoAcceptAfterHalt`: over any run of ops (`runs`), every decision after an accepted halt is the `halted` refusal.
* `FleetHaltIsFinal`: an operator the fleet has halted stays halted with its cause, and every op that names it (an
  `attach` of it, or a `register` or `failures` of one of its streams) is refused `halted` and changes nothing.
* `HaltCoversOperator`: a stream's failures at its budget leave its operator halted, with the fleet's streams unchanged;
  when the operator was not yet halted, the cause is that stream at the fleet's position, and the decision names every
  stream of the operator (`Fleet.streamsOf`), each of which the service then halts.
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

/-- A halted stream refuses every op `halted`, and its state stays as it is. -/
def HaltIsFinal : Prop :=
  ∀ (p : Stream.Params) (s : Stream.State) (op : Json) (d : Stream.Decision) (s' : Stream.State),
    Stream.run p s op = .ok (d, s') → ∀ c, s.halted = some c → d = .refused #["halted"] ∧ s' = s

/-- A halt the stream accepts names its open entries and makes each final, leaves none open, records its cause, and
changes nothing else the stream counts. -/
def HaltEndsOpen : Prop :=
  ∀ (p : Stream.Params) (s : Stream.State) (op : Json) (d : Stream.Decision) (s' : Stream.State),
    Stream.run p s op = .ok (d, s') → ∀ v, d = .ok v → getD op "op" = .str "halt" →
      v = natsJson s.opened ∧ s'.halted = some (getD op "cause") ∧ s'.opened = #[] ∧
      s'.entries = s.entries.map (fun e => if s.opened.contains e.order then { e with final := true } else e) ∧
      s'.size = s.size ∧ s'.failures = s.failures ∧ s'.saltedAt = s.saltedAt ∧ s'.ended = s.ended

/-- The stream's rules over ops in turn, from `s`: each op's decision, and the state after the last. -/
def runs (p : Stream.Params) : Stream.State → List Json → Flock.V (List Stream.Decision × Stream.State)
  | s, [] => pure ([], s)
  | s, op :: ops => do
    let (d, s') ← Stream.run p s op
    let (ds, s'') ← runs p s' ops
    pure (d :: ds, s'')

/-- Every decision after an accepted halt is the `halted` refusal. -/
def NoAcceptAfterHalt : Prop :=
  ∀ (p : Stream.Params) (s : Stream.State) (ops : List Json) (ds : List Stream.Decision) (s' : Stream.State),
    runs p s ops = .ok (ds, s') → ∀ (k : Nat) (v : Json), ds[k]? = some (.ok v) →
      ops[k]?.map (getD · "op") = some (.str "halt") → ∀ d ∈ ds.drop (k + 1), d = .refused #["halted"]

/-- The operator an op of the fleet names: `attach`'s, or that of the stream `register` or `failures` names. -/
def opOperator (s : Fleet.State) (op : Json) : Option String :=
  match getD op "op" with
  | .str "attach" => match getD op "operator" with
    | .str o => some o
    | _ => none
  | .str "register" | .str "failures" => (nat? (getD op "stream")).bind fun i => s.streams[i]?.map (·.operator)
  | _ => none

/-- A halted operator stays halted with its cause, and every op that names it is refused `halted` and changes nothing. -/
def FleetHaltIsFinal : Prop :=
  ∀ (s : Fleet.State) (op : Json) (d : Stream.Decision) (s' : Fleet.State),
    Fleet.run s op = .ok (d, s') → ∀ o c, Fleet.haltOf s o = some c →
      Fleet.haltOf s' o = some c ∧ (opOperator s op = some o → d = .refused #["halted"] ∧ s' = s)

/-- A stream's failures at its budget leave its operator halted and the streams unchanged; a new halt's cause is that
stream at the fleet's position, and its decision names every stream of the operator. -/
def HaltCoversOperator : Prop :=
  ∀ (s : Fleet.State) (op : Json) (d : Stream.Decision) (s' : Fleet.State),
    Fleet.run s op = .ok (d, s') → ∀ (i f : Nat) (a : Fleet.Attached),
      getD op "op" = .str "failures" → nat? (getD op "stream") = some i → nat? (getD op "failures") = some f →
      s.streams[i]? = some a → a.budget ≤ f →
        ∃ c, Fleet.haltOf s' a.operator = some c ∧ s'.streams = s.streams ∧
          (Fleet.haltOf s a.operator = none →
            c = Fleet.causeOf a.operator i s.size ∧ d = .ok (Fleet.haltValue c (Fleet.streamsOf s a.operator)))

end Properties.SampledProofs
