module

public import CoreProtocol.Protocol
public import Spot.Registration

@[expose] public section

/-!
SPOT's caller's check: what the caller hands SPOT's verifier beside the registration record and SPOT's own log, which
SPOT's check (`Spot.Registration.check`) reads none of. The record names its day's one declared program (`declared`;
`R2-undeclared`, `R2-declared-twice`, `R2-declared`); its window names the stream and the prover the verifier is bound
to (`binding`; `R6-stream`, `R6-prover`), the seed salt the verifier read from the challenger's log (`seedSalt`;
`R6-seed-salt`), the pool's reset, week and spent mark, at most one reset a week (`pool`; `R6-pool-reset`,
`R6-pool-week`, `R6-pool-spent`, `R6-pool-reset-week`), and the registration log's compiler and advice within its
length (`compiler`; `R6-compiler`, `R6-advice-length`); and, when the verifier expects values registered, the record's
entry for the pool's `pin` is the pool its window opens (`R7-pool-chain`).

Each step is one term (`Spot.Registration.kept`), so the check is one composition with no join point. `gate` runs it
after SPOT's check accepts: sampled proofs' one-stage form (`VStar.Recursive.Sampled.checkBlobs`) and the recursive
audit (`VStar.Recursive.checkBlobs`) both read SPOT's acceptance through it. What its emptiness means is
`Properties.Remote.CallerCheckMeans`.
-/

namespace VStar.Recursive.Caller

open Lean (Json)
open Spot
open Spot.Registration (Verifier Pool kept optStr)

/-- The day's declared program (`registration._declared`): `R2-undeclared` when the window's day has no declaration,
`R2-declared-twice` when it has more than one, `R2-declared` when the record's `program` is not the day's declared one,
and none when it is or the verifier reads no declarations. -/
def declaredCode (program : Json) : Option (Nat × Array (Nat × String)) → Option String
  | none => none
  | some (day, ds) => match Registration.declaredOf day ds with
    | [] => some "R2-undeclared"
    | [p] => if jeq program (Json.str p) then none else some "R2-declared"
    | _ => some "R2-declared-twice"

/-- The declared day's code added to `c` (`declaredCode`). -/
def declaredMarks (program : Json) (d : Option (Nat × Array (Nat × String))) (c : Array String) : Array String :=
  match declaredCode program d with
  | some code => addCode c code
  | none => c

/-- The binding's codes added to `c`: the window `w` names the stream and the prover the verifier is bound to
(`R6-stream`, `R6-prover`); none without a binding. -/
def bindingMarks (w : Json) : Option (String × String) → Array String → Array String
  | none, c => c
  | some (stream, prover), c =>
    kept (jeq (getD w "prover") (Json.str prover)) (kept (jeq (getD w "stream") (Json.str stream)) c "R6-stream")
      "R6-prover"

def optReset : Option Pool → Json
  | some p => Json.bool p.reset
  | none => Json.null

def optWeek : Option Pool → Json
  | some p => Lean.toJson p.week
  | none => Json.null

def optSpent (pool : Option Pool) (log : Array Json) : Json :=
  match pool with
  | some p => Json.bool (p.reset || p.spentBefore log)
  | none => Json.null

def onceOk (pool : Option Pool) (log : Array Json) : Bool :=
  match pool with
  | some p => !(p.reset && p.spentBefore log)
  | none => true

/-- The pool's codes added to `c`: the window `w` names the pool's reset, week and spent mark (`R6-pool-reset`,
`R6-pool-week`, `R6-pool-spent`), and a reset where the week's is spent before it is refused (`R6-pool-reset-week`). -/
def poolMarks (w : Json) (pool : Option Pool) (log : Array Json) (c : Array String) : Array String :=
  kept (onceOk pool log)
    (kept (jeq (getD w "pool_spent") (optSpent pool log))
      (kept (jeq (getD w "pool_week") (optWeek pool))
        (kept (jeq (getD w "pool_reset") (optReset pool)) c "R6-pool-reset") "R6-pool-week") "R6-pool-spent")
    "R6-pool-reset-week"

/-- The registered compiler's codes added to `c` (`Registration.compilerOk`, `R6-compiler`; `Registration.adviceOk`,
`R6-advice-length`). -/
def compilerMarks (w : Json) (rc : Option (Option (String × Nat))) (c : Array String) : Array String :=
  kept (Registration.adviceOk w rc) (kept (Registration.compilerOk w rc) c "R6-compiler") "R6-advice-length"

/-- The pool chain (`registration._pool_chain`): with a pool, the record's entry for `pin` (`Registration.entryOf`) has
the domain and root of the pool its window opens (`Registration.poolOpen`). -/
def poolChained (rec : Json) (v : Verifier) : Bool :=
  match v.pool with
  | none => true
  | some p => match Registration.entryOf rec p.pin, Registration.poolOpen rec v p with
    | some e, some (d, r) => jeq (getD e "domain") (Json.str d) && jeq (getD e "root") (Json.str r)
    | _, _ => false

/-- **The caller's check**: its codes, in the order `Spot.Registration.check` once added them; empty when the record
keeps the caller's facts against `v`. -/
def check (rec : Json) (v : Verifier) : Array String :=
  let w := getD rec "window"
  let c := declaredMarks (getD rec "program") v.declared #[]
  let c := bindingMarks w v.binding c
  let c := kept (jeq (getD w "seed_salt") (optStr v.seedSalt)) c "R6-seed-salt"
  let c := poolMarks w v.pool v.log c
  let c := compilerMarks w v.compiler c
  kept (!v.v2 || poolChained rec v) c "R7-pool-chain"

/-- SPOT's outcome `o` held to the caller's check: an acceptance stands only when `check` finds nothing, and is refused
by its codes otherwise; any other outcome is SPOT's own. -/
def gate {α β : Type} (rec : Json) (v : Verifier) : Core.Outcome α β → Core.Outcome α β
  | .accept pub prof =>
    if (check rec v).isEmpty then .accept pub prof
    else .reject s!"caller: {", ".intercalate (check rec v).toList}"
  | .reject s => .reject s
  | .undecided s => .undecided s

/-- What `gate` accepts, SPOT accepted, and the caller's check found nothing. -/
theorem gate_accept {α β : Type} {rec : Json} {v : Verifier} {o : Core.Outcome α β} {pub : α} {prof : β}
    (h : gate rec v o = .accept pub prof) : check rec v = #[] ∧ o = .accept pub prof := by
  cases o with
  | accept pub' prof' =>
    by_cases hc : (check rec v).isEmpty = true
    · simp only [gate, hc, ↓reduceIte, Core.Outcome.accept.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨Array.isEmpty_iff.mp hc, rfl⟩
    · simp [gate, hc] at h
  | reject s => simp [gate] at h
  | undecided s => simp [gate] at h

end VStar.Recursive.Caller
