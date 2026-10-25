import PouwCheck.Credit

/-!
# The Pearl-C period's credit (`credit.pearl_c_period_credit`)

`pearlCPeriodCredit` is `credit.pearl_c_period_credit` (CMP-82): a live Pearl-C pass earns credit only inside a Pearl-C
period, and the period earns only when every clause below holds, else nothing of it does, each refused with its own
code. The credit is the period's epochs as `(W, k)` (W hex), in the close's order, which the lane's next Pearl-C period
takes in its `earlier`, and the credited passes' manifests, in the order presented.

0. The inputs' form: `period` (`PEARL_C_FORM`) for the passes, `index` (`EARLIER_FORM`) for `earlier`, and `pass` for a
   pass whose manifest names no salt epoch (`liveEpoch`).
1. Exact match (`Credit.period`'s codes): the passes' salt epochs' roots R = `epochOpenRoot W k`, of each manifest's
   weights' root and `pass_index`, in the order they first appear, are exactly the windows the close lists, in its
   order, so a complete salt-only session no pass presents refuses the period.
2. Salt-only sessions (`salt-only`): no pass's session is one `liveSaltCheck` refuses with `salt`.
3. Credited sessions (`circuit`): each pass's session passes `creditedSession` with its `descriptor`.
4. Accepted passes (`pass`): each pass passes `passEpoch` under the verifier's live inputs.
5. Consecutive epochs (`index`): for each W, the period's epochs on W, in the close's order, are numbered on from the
   lane's `earlier` credited epochs on W with no gap or repeat.
6. Distinct calls (`passesDistinct`: `call-range`, `repeated-call`): no two passes of one salt hold one call index,
   and none is 2^32 or more.

Clauses 0 to 5 are `periodChecks`. After clause 6, `pearlCPeriodCredit` runs `distinctCalls`, which groups the passes by
the epoch each manifest names (`namedEpoch`, its session's root, `Properties.Pouw.LiveSalt.PassesDistinct`'s `epochOf`)
rather than by salt, so the property reads this check's own functions (`Properties.Pouw.LiveSalt.creditDistinct`). It
refuses only two passes that name one root under two salts with a call index in common, which every clause above
accepting needs the auditor's ledger to open one root twice, or two openings of one live coin.
-/

namespace PouwCheck.Period

open Lean (Json)
open SampledProofs.Challenger (ChallengerKey creditedSession)

/-- `credit.PEARL_C_KEYS`: a Pearl-C pass's evidence in its period. -/
def PEARL_C_KEYS : List String := ["context", "descriptor", "manifest", "session"]
def PEARL_C_FORM : String :=
  "the period's passes are not a nonempty list of each pass's evidence {context, descriptor, manifest, " ++
    "session}: the window context naming its manifest by SHA-512, the descriptor in lowercase hex of the " ++
    "circuit it proves, its manifest's bytes in lowercase hex and its salt epoch's session's files"
def EARLIER_FORM : String :=
  "the lane's earlier credited Pearl-C epochs are not a list of [weights' root, k] pairs: a SHA-512 digest " ++
    "(128 lowercase hex) and an integer in [0, 2^64)"
def PERIOD : String := "period"
def SALT_ONLY : String := "salt-only"
def CIRCUIT : String := "circuit"
def PASS_REFUSED : String := "pass"
def INDEX : String := "index"
def CALL_RANGE_CODE : String := "call-range"
def REPEATED_CALL : String := "repeated-call"

/-- The epoch the pass `m` names: its manifest's session's `root`. -/
def namedEpoch (m : Json) : Json := SampledProofs.getD (at? m "session") "root"

/-- The call indices of the pass `m`'s matmuls, in its manifest's order (none when its workload doesn't parse). -/
def callsOf (m : Json) : List Int :=
  match workloadOf m with
  | .ok (us, _) => us.map (·.index)
  | .error _ => []

/-- Whether two passes' epochs are one, by a match the kernel reduces (`jsonStrIs`'s): two equal strings, or either not a
string, so it holds of any two equal epochs. -/
def sameEpoch : Json → Json → Bool
  | .str a, .str b => a == b
  | _, _ => true

/-- `_manifest`'s reading, `passEpoch`'s: the bytes as UTF-8, a JSON object. -/
def manifestOf? (raw : ByteArray) : Option Json :=
  (String.fromUTF8? raw).bind fun s => match Json.parse s with
    | .ok j@(.obj _) => some j
    | _ => none

/-- `core_check.live_epoch`: a live Pearl-C pass's manifest, read from its bytes `raw`, and its salt epoch: W, the weights'
root of its workload's weights, and k, its `pass_index`, an integer in [0, 2^64). -/
def liveEpoch (raw : ByteArray) : R (Json × ByteArray × Nat) := do
  let some m := manifestOf? raw | manifest "the pass's manifest is not a JSON object"
  let (_, ws) ← workloadOf m
  let some k := (PouwWindow.intOf? (at? m "pass_index")).bind fun k => if 0 ≤ k && k < U64 then some k.toNat else none
    | manifest PASS_FORM
  pure (m, weightsRootAt 64 "verity/pouw/weights/v1" ws, k)

/-- `earlier` as `(W, k)` pairs, or none when it isn't a list of `[W, k]` pairs (`EARLIER_FORM`). -/
def earlierOf (earlier : Json) : Option (List (String × Nat)) := do
  let .arr es := earlier | none
  es.toList.mapM fun e => do
    let .arr #[w, k] := e | none
    let some _ := Credit.digest? w | none
    let .str w := w | none
    let some k := (nat? k).filter (· < 2 ^ 64) | none
    pure (w, k)

def hexBytes? : Json → Option ByteArray
  | .str s => if Credit.isHexBytes s then hex? (.str s) else none
  | _ => none

/-- Clauses 0 to 5 of `pearl_c_period_credit`: the period's epochs and the passes' manifests, or the first refusal. -/
def periodChecks (key ledger pins main name passes earlier : Json) : R (List (String × Nat) × List Json) := do
  let ps := match passes with
    | .arr a => a.toList
    | _ => []
  unless (match passes with | .arr _ => true | _ => false) && !ps.isEmpty && ps.all (keys · == some PEARL_C_KEYS) do
    throw (PERIOD, PEARL_C_FORM)
  let some ek := earlierOf earlier | throw (INDEX, EARLIER_FORM)
  let some raws := ps.mapM (fun p => hexBytes? (get p "manifest")) | throw (PERIOD, PEARL_C_FORM)
  let nm := PouwWindow.pyStr name
  let got ← raws.zipIdx.mapM fun (raw, i) => (liveEpoch raw).mapError fun e =>
    if e.1 == "error" then e
    else (PASS_REFUSED, s!"pass {i} of period {nm} names no salt epoch, so the period earns nothing: {e.2}")
  let epochs := got.map fun (_, w, k) => (w, k, Flock.hexOf (epochOpenRoot w k))
  let roots := epochs.foldl (fun rs (_, _, r) => if rs.contains r then rs else rs ++ [r]) []
  Credit.period key ledger name (Json.arr (roots.map Json.str).toArray)
  let ck ← (ChallengerKey.ofJson key).mapError fun e => ("error", s!"the verifier's challenger key is malformed: {e}")
  for ((p, (m, _, _)), i) in (ps.zip got).zipIdx do
    match liveSaltCheck key ledger pins main (get p "session") m with
    | .error (c, why) =>
      if c == "salt" then
        throw (SALT_ONLY, s!"pass {i} of period {nm} is not of a Pearl-C salt session, so the period earns nothing: {why}")
      if c == "error" then throw (c, why)
    | .ok _ => pure ()
  for ((p, (_, _, r)), i) in (ps.zip epochs).zipIdx do
    let some d := hexBytes? (get p "descriptor")
      | throw (CIRCUIT, s!"pass {i} of period {nm}: the descriptor is not the bytes in lowercase hex of the circuit it proves")
    discard <| (creditedSession ledger (Json.mkObj [("root", .str r)]) ck pins main d).mapError fun e =>
      (CIRCUIT, s!"pass {i} of period {nm} doesn't count toward it, so the period earns nothing: {e}")
  for ((p, raw), i) in (ps.zip raws).zipIdx do
    let live := Json.mkObj [("key", key), ("ledger", ledger), ("main", main), ("pins", pins), ("session", get p "session")]
    discard <| (passEpoch .null (some raw) ((str? (get p "context")).getD "") false none none none none live).mapError
      fun e => if e.1 == "error" then e
        else (PASS_REFUSED, s!"pass {i} of period {nm} is refused, so the period earns nothing: {e.1}: {e.2}")
  let mut nxt : List (String × Nat) := []
  for (w, k) in ek do
    nxt := (w, max ((nxt.lookup w).getD 0) (k + 1)) :: nxt.filter (·.1 != w)
  let mut out : List (String × Nat) := []
  for r in roots do
    let some (w, k, _) := epochs.find? (·.2.2 == r) | throw ("error", "a root no pass names")
    let w := Flock.hexOf w
    let want := (nxt.lookup w).getD 0
    if k != want then
      throw (INDEX, s!"period {nm}'s epoch {k} over weights' root {w.take 16}... is not epoch {want}: a " ++
        "weights' root's salt epochs are numbered on from its lane's earlier credited periods " ++
        "with no gap or repeat, so a root whose numbering broke is spent: re-key hm96 for a new " ++
        "weights' root, its epochs numbered from 0")
    nxt := (w, k + 1) :: nxt.filter (·.1 != w)
    out := out ++ [(w, k)]
  pure (out, got.map (·.1))

/-- `credit.passes_distinct`, clause 6: every call index below 2^32 (`call-range`), and no index held by two passes of
one salt, nor twice by one pass (`repeated-call`). -/
def passesDistinct (ms : List Json) : R Unit := do
  let mut seen : List (String × List (Int × Nat)) := []
  for (m, p) in ms.zipIdx do
    let some salt := str? (at? m "salt")
      | throw ("manifest", s!"pass {p} is not a Pearl-C manifest with a string salt and a list of matmuls")
    let some us := (match at? m "matmuls" with
        | .arr a => if a.all Credit.isObj then some a.toList else none
        | _ => none)
      | throw ("manifest", s!"pass {p} is not a Pearl-C manifest with a string salt and a list of matmuls")
    let mut held := (seen.lookup salt).getD []
    for u in us do
      let iv := at? u "index"
      let some i := (PouwWindow.strictInt? iv).filter fun i => 0 ≤ i && i < CALL_BOUND
        | throw (CALL_RANGE_CODE, s!"pass {p} holds call index {PouwWindow.pyRepr iv}, outside [0, 2^32) where " ++
            "-h3's row seed packs it: end the epoch (a new salt) before its 2^32nd call")
      if let some q := held.lookup i then
        let both := if q == p then s!"pass {p} holds call index {i} twice" else s!"passes {q} and {p} both hold call index {i}"
        throw (REPEATED_CALL, s!"{both} under salt {salt.take 16}...: one epoch's calls repeat their seeds and " ++
          "tickets, so the device numbers calls across the whole epoch and no index is credited twice")
      held := (i, p) :: held
    seen := (salt, held) :: seen.filter (·.1 != salt)

/-- The root-grouped clause for pass `i` of period `nm`, of epoch `e`, over its call indices: each below 2^32, and none
an index of a pass of `seen` (each its position, its epoch and its call indices) of the same epoch. -/
def freshCalls (nm : String) (i : Nat) (e : Json) (seen : List (Nat × Json × List Int)) : List Int → R Unit
  | [] => pure ()
  | u :: us =>
    if 0 ≤ u ∧ u < CALL_BOUND then
      match seen.find? (fun p => sameEpoch p.2.1 e && decide (u ∈ p.2.2)) with
      | some (q, _, _) => throw (REPEATED_CALL, s!"passes {q} and {i} of period {nm} both hold call index {u} of " ++
          s!"epoch {((str? e).getD "").take 16}... under two salts: an epoch's root opens one session, whose salt " ++
          "coin opens once")
      | none => freshCalls nm i e seen us
    else throw (CALL_RANGE_CODE, s!"pass {i} of period {nm} holds call index {u}: {CALL_RANGE}")

/-- The root-grouped clause over the passes `ms` from position `i` on, after the passes `seen`: no two passes naming one
epoch share a call index, and every call index is below 2^32. -/
def distinctCalls (nm : String) : Nat → List (Nat × Json × List Int) → List Json → R Unit
  | _, _, [] => pure ()
  | i, seen, m :: ms => do
    freshCalls nm i (namedEpoch m) seen (callsOf m)
    distinctCalls nm (i + 1) (seen ++ [(i, namedEpoch m, callsOf m)]) ms

end PouwCheck.Period

namespace PouwCheck

open Lean (Json)

/-- `credit.pearl_c_period_credit`: Pearl-C period `name`'s credit, its epochs as `(W, k)` in the close's order and the
manifests of `passes` in order, under the pinned challenger key `key`, the auditor's ledger `ledger`, the host's pins log
`pins` and the verifier's copy of main's first-parent history `main`; `passes` each `{context, descriptor, manifest,
session}` and `earlier` the lane's epochs credited in earlier Pearl-C periods, `[W, k]` pairs. Else the refusal of the
first clause that fails: clauses 0 to 5 are `Period.periodChecks`, clause 6 `Period.passesDistinct`, and then
`Period.distinctCalls`. -/
def pearlCPeriodCredit (key ledger pins main name passes earlier : Json) : R (List (String × Nat) × List Json) := do
  let r ← Period.periodChecks key ledger pins main name passes earlier
  Period.passesDistinct r.2
  Period.distinctCalls (PouwWindow.pyStr name) 0 [] r.2
  pure r

end PouwCheck
