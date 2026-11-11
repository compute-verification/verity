module

public import PouwPearlPlus.Coins

/-!
# Reading a retained pass (`verify_run.load`, `credit.passes_distinct`, `pearl_c.manifest_epoch`)

`load coins man` is a pass's manifest held to what it published, in `load`'s order, each refusal in the words `verify_passes`
gives it (`TYPE: why`, the exception Python raises):

1. the manifest's schema (`manifest_schema`: the v2 run, not one recorded before epoch coins), its call indices
   (`passes_distinct`), its scheme (Pearl-C v1 -h3 on the H100's record) and its matmuls' shapes;
2. its epoch (`manifest_epoch`, `Epoch.start`, `host.epoch_of`): the pass's coin commitment must be the verifier's (an
   in-process challenger's, or a live session's when the pass's coins are `window.LIVE_SCHEME`'s), its salt coin must open
   against it after the weights' root (a live session's after the salt epoch's open root, `Coins.liveSalt`), and the salt
   must be the manifest's;
3. its records (`records_of`, `UnitRecord`): real rows, |K|, excluded and voluntary rows, held to V-EX's caps
   (`checkRecords`) and to the records root the pass published;
4. its domain (`PearlC.check`) and the commitment's tile count.

`passesDistinct` refuses passes credited together whose call indices leave `[0, 2^32)` (`call-range`) or repeat under one
salt (`repeated-call`), or one that is not a manifest with a string salt and a list of matmuls (`manifest`).

Out of this port's scope, each refused by name: the sm_120 records' schemes and a manifest's `instance` other than `k_min`
and `tile`.
-/

@[expose] public section

namespace PouwCheck.PearlPlus

open Lean PouwCommit
open Spot.Challenger (LiveCommitment LiveOpening)

def MANIFEST_SCHEMA : String := Domains.VLLM_PEARL_C_RUN
def MANIFEST_RECORDED : String := Domains.VLLM_PEARL_C_RUN_V1
def CALL_RANGE : Nat := 2 ^ 32

/-- Lowercase hex, `bytes.hex()`. -/
def hex (b : ByteArray) : String :=
  String.ofList (b.toList.flatMap fun x => [Nat.digitChar (x.toNat / 16), Nat.digitChar (x.toNat % 16)])

/-- A window class's public caps (`window_class.WindowClass`'s, `verify_passes`' ``window_class``): its slots, each of at most
`prompt` prompt ids and `outputs` outputs. Its fixed load is every slot's prompt at round 0 and a row a slot at each
later round of its `outputs` (`Tie.heldToLoad`). -/
structure LoadClass where
  slots : Nat
  prompt : Nat
  outputs : Nat
  deriving Repr, DecidableEq

/-- `verify_run.tie`'s answer: the window record's digest (hex), the number of tables it ties (`Tie.tieOf`), for a
version-1 W, `Tie.WEIGHTS_UNTIED`, and the class whose fixed load the passes were held to where the verdict was given
one (`Tie.tieAt`). -/
structure Tied where
  record : String
  tables : Nat
  weights : Option String := none
  load : Option LoadClass := none
  deriving Repr

/-- A refusal: Python's (code, why) where it has one (`passes_distinct`'s and `AFTER_COIN`), the reason it rejects with,
and the tie when the passes were held to W before the refusal (Python's verdict carries it then). -/
structure Refusal where
  refused : Option (String × String) := none
  reason : String
  tie : Option Tied := none
  deriving Repr

def refuse {α : Type} (type why : String) : Except Refusal α := throw { reason := s!"{type}: {why}" }

/-- `passes_distinct(passes)`: (code, why), or none. -/
def passesDistinct (passes : List Json) : Option (String × String) := Id.run do
  let mut seen : List (String × List (Nat × Nat)) := []
  for (m, p) in passes.zipIdx do
    let (salt, us) ← match get? m "salt", get? m "matmuls" with
      | some (.str s), some (.arr us) =>
        if us.all (fun u => match u with | .obj _ => true | _ => false) then (s, us.toList)
        else return some ("manifest", s!"pass {p} is not a Pearl-C manifest with a string salt and a list of matmuls")
      | _, _ => return some ("manifest", s!"pass {p} is not a Pearl-C manifest with a string salt and a list of matmuls")
    let mut held := (seen.lookup salt).getD []
    for u in us do
      let i? := get? u "index"
      let i ← match i? >>= jsonInt? with
        | some i => if 0 ≤ i && i < CALL_RANGE then i.toNat else
            return some ("call-range", s!"pass {p} holds call index {pyReprJson (i?.getD .null)}, outside [0, 2^32) where " ++
              "-h3's row seed packs it: end the epoch (a new salt) before its 2^32nd call")
        | none => return some ("call-range", s!"pass {p} holds call index {pyReprJson (i?.getD .null)}, outside " ++
              "[0, 2^32) where -h3's row seed packs it: end the epoch (a new salt) before its 2^32nd call")
      if let some q := held.lookup i then
        let both := if q == p then s!"pass {p} holds call index {i} twice" else s!"passes {q} and {p} both hold call index {i}"
        return some ("repeated-call", s!"{both} under salt {String.ofList (salt.toList.take 16)}...: one epoch's calls " ++
          "repeat their seeds and tickets, so the device numbers calls across the whole epoch and no index is credited twice")
      held := (i, p) :: held
    seen := (salt, held) :: seen.filter (·.1 != salt)
  return none

/-- A JSON field Python reads as `man[k]`, refused as its `KeyError`. -/
def key (j : Json) (k : String) : Except Refusal Json :=
  match get? j k with
  | some v => pure v
  | none => refuse "KeyError" (pyRepr k)

def natOf? (j : Json) : Option Nat := match jsonInt? j with
  | some i => if 0 ≤ i then some i.toNat else none
  | none => none

def nats (j : Json) (what : String) : Except Refusal (List Nat) :=
  match j with
  | .arr xs => xs.toList.mapM fun x => match natOf? x with
    | some n => pure n
    | none => refuse "ValueError" s!"{what}: rows are integers"
  | _ => refuse "ValueError" s!"{what}: rows are a list"

def hexOf (j : Json) (what : String) : Except Refusal ByteArray :=
  match j with
  | .str s => match pyFromHex s with
    | .ok b => pure b
    | .error _ => refuse "ValueError" s!"{what}: not hex"
  | _ => refuse "TypeError" s!"{what}: not a string"

/-- `PD.scheme_of(man["scheme"], **man.get("instance", {}))`, this port's scheme alone. -/
def schemeOf (man : Json) : Except Refusal Scheme := do
  let name ← match ← key man "scheme" with
    | .str s => pure s
    | j => refuse "TypeError" s!"the scheme {j.compress} is not a name"
  if !SCHEMES.contains name then
    refuse "KeyError" ("\"" ++ pyRepr name ++ s!": this device path runs {", ".intercalate SCHEMES}\"")
  if name != NAME then
    refuse "NotImplementedError" s!"{name}: this port checks {NAME} alone (the sm_120 records' schemes are not ported)"
  match get? man "instance" with
  | none => pure {}
  | some (.obj kv) => kv.toList.foldlM instanceField {}
  | some _ => refuse "NotImplementedError" "the instance is not an object"
where
  instanceField (s : Scheme) : String × Json → Except Refusal Scheme
    | ("k_min", v) => match natOf? v with
      | some n => pure { s with kMin := n }
      | none => refuse "NotImplementedError" "the instance's k_min is not an integer"
    | ("tile", .arr #[h, w]) => match natOf? h, natOf? w with
      | some h, some w => if h = 0 || w = 0 then refuse "NotImplementedError" "a tile is at least 1 x 1"
        else pure { s with tileM := h, tileN := w }
      | _, _ => refuse "NotImplementedError" "the instance's tile is two integers"
    | (k, _) => refuse "NotImplementedError" s!"the instance's {k} is not ported (k_min and tile are)"

/-- `Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"])`. -/
def matmulOf (u : Json) : Except Refusal Matmul := do
  let idx ← key u "index"
  let m ← key u "m"
  let k ← key u "k"
  let n ← key u "n"
  let w ← key u "weight"
  let pos (j : Json) := match jsonInt? j with | some i => decide (1 ≤ i) | none => false
  if !(pos m && pos k && pos n) then refuse "DomainError" "a shape has m, k, n >= 1"
  match natOf? idx, natOf? w with
  | some i, some wt =>
    if i < 2 ^ 64 && wt < 2 ^ 64 then pure ⟨i, (natOf? m).get!, (natOf? k).get!, (natOf? n).get!, wt⟩
    else refuse "DomainError" "a matmul's index and weight id are 64-bit"
  | _, _ => refuse "DomainError" "a matmul's index and weight id are 64-bit"

/-- `manifest_schema(man)`: the v2 run's schema, not one recorded before epoch coins. -/
def manifestSchema (man : Json) : Except Refusal Unit := do
  let schema := get? man "schema"
  if fieldIs schema MANIFEST_RECORDED then
    refuse "DomainError" (s!"the pass's manifest is {MANIFEST_RECORDED}, recorded before epoch coins: only a replay of a " ++
      "recorded pass reads it")
  if !fieldIs schema MANIFEST_SCHEMA then
    refuse "DomainError" s!"schema {pyReprJson (schema.getD .null)}, not {MANIFEST_SCHEMA}"

/-- `Workload(matmuls)`'s refusals. -/
def workloadReason (us : List Matmul) : Option String := Id.run do
  if (us.map (·.index)).eraseDups.length != us.length then return some "matmul indices are unique"
  let mut dims : List (Nat × Nat × Nat) := []
  for u in us do
    match dims.lookup u.weight with
    | some d => if d != (u.n, u.k) then return some s!"every matmul on weight {u.weight} has the weight's (n, k)"
    | none => dims := (u.weight, u.n, u.k) :: dims
  return none

def throwRefusal {α : Type} (why : Option (String × String)) (k : Unit → Except Refusal α) : Except Refusal α :=
  match why with
  | some (c, w) => throw { refused := some (c, w), reason := s!"Refused: {c}: {w}" }
  | none => k ()

def malformed {α : Type} (e : String) : Except Refusal α := refuse "DomainError" s!"the pass's coins are malformed: {e}"

/-- `load(path, coins)`: the retained pass `man` held to what it published under the epoch's coin commitment `coins`, the
verifier's copy (`verify_run.epoch_coins`). -/
def load (coins : EpochCoins) (man : Json) : Except Refusal Pass := do
  manifestSchema man
  throwRefusal (passesDistinct [man]) fun _ => do
  let scheme ← schemeOf man
  let us ← match ← key man "matmuls" with
    | .arr us => pure us.toList
    | _ => refuse "TypeError" "the matmuls are a list"
  let matmuls ← us.mapM matmulOf
  let weights ← match ← key man "weights" with
    | .obj kv => pure kv.toList
    | _ => refuse "AttributeError" "'weights' is not a dict"
  let roots ← weights.mapM fun (w, v) => do
    let some wi := w.toNat? | refuse "ValueError" s!"invalid literal for int() with base 10: {pyRepr w}"
    pure (wi, ← hexOf (← key v "root") "a weight's root")
  let some openingJ := get? man "salt_opening" | refuse "DomainError" "the pass has no salt coin"
  let claimedJ? := get? man "coins"
  let live := match claimedJ? with
    | some j => fieldIs (get? j "scheme") LIVE_SCHEME
    | none => false
  let opening ← match live, openingJ with
    | true, o => match LiveOpening.ofJson o with
      | .ok o => pure (Sum.inr o)
      | .error e => malformed e
    | false, o => match CoinOpening.ofJson o with
      | .ok o => pure (Sum.inl o)
      | .error e => malformed e
  let some claimedJ := claimedJ? | malformed (pyRepr "coins")
  let claimed : EpochCoins ← if live then match LiveCommitment.ofJson claimedJ with
      | .ok c => pure (.live c)
      | .error e => malformed e
    else match CoinCommitment.ofJson claimedJ with
      | .ok c => pure (.inproc c)
      | .error e => malformed e
  let same := match claimed, coins with
    | .inproc a, .inproc b => a.epoch == b.epoch && a.root == b.root
    | .live a, .live b => sameLive a b
    | _, _ => false
  if !same then refuse "DomainError" "the pass's coin commitment is not the epoch's"
  if let some why := workloadReason matmuls then refuse "DomainError" why
  let ws := (matmuls.map (·.weight)).eraseDups
  if ws.length != roots.length || !ws.all (fun w => (roots.lookup w).isSome) then
    refuse "DomainError" "every weight of the workload is committed, and only those"
  let weightsRoot ← match PouwCommit.weightsRoot roots with
    | .ok r => pure r
    | .error e => refuse "ValueError" e
  let salt ← match coins, opening with
    | .inproc c, .inl o => match epochSalt c o weightsRoot with
      | .ok (.ok s) => pure s
      | .ok (.error why) => refuse "DomainError" why
      | .error e => refuse "ValueError" e
    | .live c, .inr o => match liveSalt c o weightsRoot ((get? man "pass_index").getD .null) with
      | .ok s => pure s
      | .error e => throw { reason := e }
    | _, _ => refuse "DomainError" "the pass's coin commitment is not the epoch's"
  if !fieldIs (get? man "salt") (hex salt) then
    refuse "DomainError" "the pass's salt is not its coin's and the weights'"
  let wj (w : Nat) : Json := (weights.lookup (toString w)).getD .null
  let records ← (us.zip matmuls).mapM fun (u, mm) => do
    let wt := wj mm.weight
    let real ← key u "real_rows"
    let kk ← key u "k_kept"
    let some real := natOf? real | refuse "TypeError" "a record's real rows are an integer"
    let some kk := natOf? kk | refuse "TypeError" "a record's kept width is an integer"
    let ea ← nats (← key u "excluded_a") "excluded_a"
    let eb ← nats (← key wt "excluded_b") "excluded_b"
    let va ← nats ((get? u "voluntary_a").getD (.arr #[])) "voluntary_a"
    let vb ← nats ((get? wt "voluntary_b").getD (.arr #[])) "voluntary_b"
    let r : UnitRecord := ⟨mm, real, kk, ea, eb, va, vb⟩
    if let some why := r.reason then refuse "ValueError" why
    pure r
  let splits ← ws.mapM fun w => do
    match natOf? (← key (wj w) "k") with
    | some k => pure (w, k)
    | none => refuse "TypeError" "a split's k is an integer"
  if let some why := checkRecords splits records then refuse "ValueError" why
  let published ← hexOf (← key man "records_root") "the records root"
  if recordsRoot salt weightsRoot splits records != published then
    refuse "ValueError" "the records are not the ones whose root the pass published"
  let rootsA ← us.mapM fun u => do hexOf (← key u "root_a") "an activation root"
  let c ← key man "commitment"
  let root ← hexOf (← key c "root") "the transcript root"
  let some count := natOf? (← key c "count") | refuse "TypeError" "the commitment's count is an integer"
  let yRoots ← us.mapM fun u => match get? u "y" with
    | some y => do pure (some (← hexOf (← key y "root") "a y root"))
    | none => pure none
  for u in matmuls do
    if let some why := scheme.domainReason u then refuse "DomainError" why
  let p : Pass := { scheme := scheme, salt := salt, weightRoots := roots, weightsRoot := weightsRoot, records := records,
                    rootsA := rootsA, transcriptRoot := root, count := count, recordsRoot := published, yRoots := yRoots }
  if p.tileCount != count then
    refuse "ValueError" s!"the commitment counts {count} tiles; the workload has {p.tileCount}"
  pure p

end PouwCheck.PearlPlus
