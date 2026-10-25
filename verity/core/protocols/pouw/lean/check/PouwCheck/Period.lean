import PouwCheck.Credit

/-!
# The Pearl-C period's credit (`pearl_c_period.py`)

`pearlCPeriodCredit` is `pearl_c_period.pearl_c_period_credit` (CMP-82): a live Pearl-C pass earns credit only inside a
Pearl-C period, and the period earns only when every clause below holds, else nothing of it does. Its credit is the
passes' manifests, in the order presented. Its refusals are numbered by the clause each enforces (`REFUSALS`; `form`, 0,
is its inputs'):

1. `exact`: `Credit.period` passes with the passes' epoch roots (each manifest's `session.root`, the distinct ones in the
   order the passes first name them), so a complete salt-only session the close lists but no pass presents refuses the
   period.
2. `salt-only`: every session the period's close lists is reserved for the salt coin alone, its open binding that
   reserve (`saltOnly`), so H1's sessions and Pearl-C's never share a period.
3. `session`: each pass's session counts toward the period (`creditedSession`, with the pass's `descriptor`), within the
   period's pin floor (`Credit.pinHygiene`).
4. `pass`: each pass passes `passEpoch` under the verifier's live inputs; one refused pass refuses the period.
5. `index`: for each weights' root W, the period's epochs on W, in the close's order, are numbered on from the lane's
   `earlier` credited epochs on W with no gap or repeat.
6. `repeated-call`: every call index is below 2^32 and no two passes of one epoch share one (`distinctCalls`), so the
   credited passes meet `Properties.Pouw.LiveSalt.PassesDistinct` (`Properties.Pouw.LiveSalt.creditDistinct`).

Clauses 0 to 5 are `periodChecks`; `pearlCPeriodCredit` runs clause 6 on what they return. Clause 6 groups passes by the
epoch each manifest names (`namedEpoch`, `PassesDistinct`'s `epochOf`) and reads their call indices as `callsOf`
(`PassesDistinct`'s `callIndices`), so the property reads this check's own functions.
-/

namespace PouwCheck.Period

open Lean (Json)
open SampledProofs.Challenger (ChallengerKey creditedSession familiesJson)

def FORM : String := "form"
def EXACT : String := "exact"
def SALT_ONLY : String := "salt-only"
def SESSION : String := "session"
def PASS : String := "pass"
def INDEX : String := "index"
def REPEATED_CALL : String := "repeated-call"
/-- The period's refusals, numbered by the rule's clause each enforces (`form`, 0, is its inputs'). -/
def REFUSALS : List String := [FORM, EXACT, SALT_ONLY, SESSION, PASS, INDEX, REPEATED_CALL]

/-- A presented pass: the window's context naming its manifest, the descriptor of the circuit its period committed (hex),
its manifest's bytes (hex) and its epoch's session's files. -/
def PASS_KEYS : List String := ["context", "descriptor", "manifest", "session"]
def PASSES_FORM : String :=
  "the period's passes are a list of {context, descriptor, manifest, session}: the window's context naming " ++
    "the pass's manifest by its SHA-512 (a string), the circuit descriptor's bytes and the manifest's bytes " ++
    "in lowercase hex, and its epoch's session's files"
def EARLIER_FORM : String :=
  "the lane's earlier credited epochs are a list of [W, k] pairs: a weights' root (a SHA-512 digest in " ++
    "lowercase hex) and an epoch number, an integer in [0, 2^64)"
def NOT_SALT_ONLY : String :=
  "is not a Pearl-C salt session: the auditor's ledger doesn't reserve it for the salt coin alone, or " ++
    "opens it binding no such reserve, and H1's sessions and Pearl-C's never share a period"
def OUT_OF_TURN : String :=
  "a weights' root's epochs run 0, 1, 2 and on across its lane's credited periods with no gap or repeat, " ++
    "so a root whose numbering broke is spent"
def REPEATED : String := "a call index names one call across its epoch, so no call's work is credited twice"

/-- The epoch the pass `m` names: its manifest's session's `root`. -/
def namedEpoch (m : Json) : Json := SampledProofs.getD (at? m "session") "root"

/-- The call indices of the pass `m`'s matmuls, in its manifest's order (none when its workload doesn't parse). -/
def callsOf (m : Json) : List Int :=
  match workloadOf m with
  | .ok (us, _) => us.map (·.index)
  | .error _ => []

/-- Whether two passes' epochs are one, by a match the kernel reduces (`jsonStrIs`'s): two equal strings, or either not a
string, which `presented` refuses (`form`), so it holds of any two equal epochs. -/
def sameEpoch : Json → Json → Bool
  | .str a, .str b => a == b
  | _, _ => true

def isStr : Json → Bool
  | .str _ => true
  | _ => false

/-- `_manifest`'s reading, `passEpoch`'s: the bytes as UTF-8, a JSON object. -/
def manifestOf? (raw : ByteArray) : Option Json :=
  (String.fromUTF8? raw).bind fun s => match Json.parse s with
    | .ok j@(.obj _) => some j
    | _ => none

/-- `_presented`: each pass's manifest bytes, the manifest and the epoch root it names, or the `form` refusal. -/
def presented (passes : Json) : R (List (ByteArray × Json × String)) := do
  let .arr ps := passes | throw (FORM, PASSES_FORM)
  unless ps.all (fun p => keys p == some PASS_KEYS && isStr (get p "context")) do throw (FORM, PASSES_FORM)
  ps.toList.zipIdx.mapM fun (p, i) => do
    let some raw := (match get p "manifest" with
        | .str s => if Credit.isHexBytes s then hex? (.str s) else none
        | _ => none) | throw (FORM, s!"pass {i}'s manifest is not its bytes in lowercase hex")
    let some m := manifestOf? raw | throw (FORM, s!"pass {i}'s manifest is not a JSON object")
    let .str r := namedEpoch m
      | throw (FORM, s!"pass {i}'s manifest names no epoch: its session's root is not a string")
    pure (raw, m, r)

/-- `n` with W's next epoch at least `v`. -/
def bump (n : List (String × Nat)) (w : String) (v : Nat) : List (String × Nat) :=
  (w, max ((n.lookup w).getD 0) v) :: n.filter (·.1 != w)

/-- `_next_epochs`: n_W for each W `earlier` lists, one more than the largest k it lists on W, or none when `earlier` isn't
a list of `[W, k]` pairs (`EARLIER_FORM`). -/
def nextEpochs (earlier : Json) : Option (List (String × Nat)) := do
  let .arr es := earlier | none
  es.toList.foldlM (init := []) fun n e => do
    let .arr #[w, k] := e | none
    let some _ := Credit.digest? w | none
    let .str w := w | none
    let some k := (nat? k).filter (· < 2 ^ 64) | none
    pure (bump n w (k + 1))

/-- `_salt_only`, clause 2, over the close of period `nm` in `ledger`, which `Credit.period` has trusted: each session the
close lists is reserved for the salt coin alone (`PouwWindow.SALT_FAMILIES`), its open binding that reserve, as
`liveSaltCheck` reads a pass's session. -/
def saltOnly (ledger : Json) (nm : String) : R Unit := do
  let es := (match ledger with | .arr ls => ls.toList | _ => []).filterMap SampledProofs.Challenger.ledgerEntry?
  let some close := (es.find? fun e => e.kind == "period-close" && get e.body "period" == Json.str nm).map (·.body)
    | pure ()
  let ws := match get close "windows" with
    | .arr a => a.toList
    | _ => []
  for (w, j) in ws.zipIdx do
    let sid := (str? (get w "session")).getD ""
    let named := sessionEntries ledger sid
    let ok := match named.find? (·.kind == "reserve"), named.find? (·.kind == "open") with
      | some rv, some o =>
        SampledProofs.jeq (SampledProofs.getD rv.body "families") (familiesJson PouwWindow.SALT_FAMILIES) &&
          SampledProofs.int? (SampledProofs.getD o.body "reserve") == some (rv.seq : Int)
      | _, _ => false
    unless ok do throw (SALT_ONLY, s!"window {j} of period {nm} (session {sid}) {NOT_SALT_ONLY}")

/-- A step's refusal as the period's `code`, `why` of the step's code and reason; the port's own errors stay errors. -/
def wrap (code : String) (why : String → String) (e : String × String) : String × String :=
  if e.1 == "error" then e else (code, why s!"{e.1}: {e.2}")

/-- Clauses 0 to 5 of `pearl_c_period_credit`: the passes' manifests, in the order presented, or the first refusal. -/
def periodChecks (key ledger pins main name passes earlier : Json) : R (List Json) := do
  let got ← presented passes
  let some next := nextEpochs earlier | throw (FORM, EARLIER_FORM)
  let roots := got.foldl (fun rs (_, _, r) => if rs.contains r then rs else rs ++ [r]) []
  (Credit.period key ledger name (Json.arr (roots.map Json.str).toArray)).mapError
    (wrap EXACT fun why => s!"the period's sessions are not exactly its passes' epochs, so it earns nothing: {why}")
  let nm := (str? name).getD ""
  saltOnly ledger nm
  let ck ← (ChallengerKey.ofJson key).mapError fun e => ("error", s!"the verifier's challenger key is malformed: {e}")
  let periodPin := Credit.committedPin ledger nm
  let floor := Credit.committedFloor ledger nm
  let ps := match passes with
    | .arr a => a.toList
    | _ => []
  for ((p, (_, _, r)), i) in (ps.zip got).zipIdx do
    let j := roots.idxOf r
    let some d := (match get p "descriptor" with
        | .str s => if Credit.isHexBytes s then hex? (.str s) else none
        | _ => none) | throw (SESSION, s!"pass {i} of period {nm}: {Credit.DESCRIPTOR_HEX}")
    let a ← (creditedSession ledger (Json.mkObj [("root", .str r)]) ck pins main d).mapError fun e =>
      (SESSION, s!"pass {i} of period {nm}, of its window {j}, doesn't count toward it, so the period earns nothing: {e}")
    if let some (c, why) := Credit.pinHygiene nm j main a.pinned periodPin floor then throw (SESSION, s!"{c}: {why}")
  for ((p, (raw, _, _)), i) in (ps.zip got).zipIdx do
    let live := Json.mkObj [("key", key), ("ledger", ledger), ("main", main), ("pins", pins), ("session", get p "session")]
    discard <| (passEpoch .null (some raw) ((str? (get p "context")).getD "") false none none none none live).mapError
      (wrap PASS fun why => s!"pass {i} of period {nm} is refused, so the period earns nothing: {why}")
  let mut n := next
  for (r, j) in roots.zipIdx do
    let some (_, m, _) := got.find? (·.2.2 == r) | throw ("error", "a root no pass names")
    let (_, ws) ← workloadOf m
    let w := Flock.hexOf (weightsRootAt 64 "verity/pouw/weights/v1" ws)
    let k := (nat? (at? m "pass_index")).getD 0
    let want := (n.lookup w).getD 0
    if k != want then
      throw (INDEX, s!"window {j} of period {nm} is epoch {k} of weights' root {w.take 16}..., not epoch {want}: " ++
        OUT_OF_TURN)
    n := (w, want + 1) :: n.filter (·.1 != w)
  pure (got.map (·.2.1))

/-- Clause 6 for pass `i` of period `nm`, of epoch `e`, over its call indices: each below 2^32 (`CALL_RANGE`), and none
an index of a pass of `seen` (each its position, its epoch and its call indices) of the same epoch. -/
def freshCalls (nm : String) (i : Nat) (e : Json) (seen : List (Nat × Json × List Int)) : List Int → R Unit
  | [] => pure ()
  | u :: us =>
    if 0 ≤ u ∧ u < CALL_BOUND then
      match seen.find? (fun p => sameEpoch p.2.1 e && decide (u ∈ p.2.2)) with
      | some (q, _, _) => throw (REPEATED_CALL, s!"passes {q} and {i} of period {nm} both hold call index {u} of " ++
          s!"epoch {((str? e).getD "").take 16}...: {REPEATED}")
      | none => freshCalls nm i e seen us
    else throw (REPEATED_CALL, s!"pass {i} of period {nm} holds call index {u}: {CALL_RANGE}")

/-- Clause 6, `repeated-call`, over the passes `ms` from position `i` on, after the passes `seen`: no two passes of one
epoch share a call index, and every call index is below 2^32. A pass presented twice shares its indices with itself. -/
def distinctCalls (nm : String) : Nat → List (Nat × Json × List Int) → List Json → R Unit
  | _, _, [] => pure ()
  | i, seen, m :: ms => do
    freshCalls nm i (namedEpoch m) seen (callsOf m)
    distinctCalls nm (i + 1) (seen ++ [(i, namedEpoch m, callsOf m)]) ms

end PouwCheck.Period

namespace PouwCheck

/-- `pearl_c_period.pearl_c_period_credit`: Pearl-C period `name`'s credit, the manifests of `passes` in order, under the
pinned challenger key `key`, the auditor's ledger `ledger`, the host's pins log `pins` and the verifier's copy of main's
first-parent history `main`; `passes` each `{context, descriptor, manifest, session}` and `earlier` the lane's epochs
credited in earlier Pearl-C periods, `[W, k]` pairs. Else the refusal of the first clause that fails
(`Period.REFUSALS`): clauses 0 to 5 are `Period.periodChecks`, and clause 6 `Period.distinctCalls`. -/
def pearlCPeriodCredit (key ledger pins main name passes earlier : Json) : R (List Json) := do
  let ms ← Period.periodChecks key ledger pins main name passes earlier
  Period.distinctCalls ((str? name).getD "") 0 [] ms
  pure ms

end PouwCheck
