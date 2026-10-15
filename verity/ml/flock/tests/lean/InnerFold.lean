import Flock

/-!
What V* computes natively for the inner session's algebra, from the Lean verifier of record and nothing the prover sends:
the extra claims at the session's link points (`Stmt.extraClaims`) and, per rep, the lincheck's circuit fold `comb`
(`CircuitFold.folded`) at the rep's fold points. The points are the verifier's own coins (the link points from the
session's public record, the fold points from each rep's recorded transcript), so this reads no proof and no session
record. `verity_flock.rec_vstage` hands `comb` to the algebra as a public input; its tests hold it to
`tests/fixtures/inner_claims.json`, which `InnerClaims.lean` dumps while running the whole inner verifier.

    lean --run InnerFold.lean [--plain-leaves] [--partition FILE] [--registered FILE] [--hide NET] STATEMENT CIRCUIT PUBLIC POINTS [os]

POINTS: `{"link": [[hex, ...], ...], "reps": [{"alpha", "beta", "lw": [64], "rho_in": [k_log - 6], "ts": [k_log - 6]}]}`.
`os`: the statement's tags on the verifier's live OS coins (`verify --coins os`), else seeded coins.
`--partition`: the verifier's copy of the `verity/partition/v1` object a circuit binds (`verify --partition`).
`--registered`: the verifier's own registered reads (`verify --registered`, `verity/registered-reads/v0`) of the inner
statement's description: its rows, read from the developer's registration (`Registered.checkDescription`, PROTOCOL.md
§16.16), or the fold is refused; without it, a statement naming registered reads is refused.
`--plain-leaves`: the statement's tags with the inner statement's plain SHA-512 leaves (`leafSchemePlain`) in place of
their own leaf-commitment scheme: a recursive session's inner statement, whose tree tops the firewall salts.
`--hide NET`: circuit NET's unit circuits are hidden (`verity_flock.rec_session`): each rep's `comb` is the fold of the
statement without NET's range (its slots' A₀ and B₀; Δ and the pin stay), `full` the whole fold, and `hidden` names NET.
-/

open Lean Flock

/-- META `leaf_scheme` of a recursive session's inner statement (`verity_flock.circuit.LEAF_SCHEME_PLAIN`). -/
def leafSchemePlain : Json :=
  Json.mkObj [("id", .str "flock-leaf/sha512-unsalted"), ("inner", .str "sha512 column digest"), ("salt_bytes", toJson (0 : Nat)),
    ("use", .str "a recursive session's inner statement: the gateway commits every tree top under a salt, ZK off")]

def hx (a : F128) : Json := .str a.toHex

def hxs (a : Array F128) : Json := .arr (a.map hx)

def claimJson (c : Claim) : Json := Json.mkObj [("value", hx c.value), ("skip", hxs c.skip), ("x", hxs c.x)]

def one (j : Json) (k : String) : V F128 := do f128OfHex (← Record.str j k)

def many (j : Json) (k : String) : V (Array F128) := do Record.f128s (← Record.arr j k)

/-- `hide`: the hidden circuit's name and the fold of the statement without it. -/
def run (st : Setup) (pts : Json) (hide : Option (String × CircuitFold) := none) : V Json := do
  let link ← (← Record.arr pts "link").mapM fun p => match p.getArr? with
    | .ok a => Record.f128s a
    | .error _ => throw "points: a link point is not a list"
  let extras ← st.extra link
  let mut reps := #[]
  for rep in ← Record.arr pts "reps" do
    let (alpha, beta, lw, rho, ts) := (← one rep "alpha", ← one rep "beta", ← many rep "lw", ← many rep "rho_in", ← many rep "ts")
    let full ← st.fold.folded alpha beta lw rho ts
    if full.size != 64 then throw "the circuit fold has the wrong width"
    match hide with
    | none => reps := reps.push (Json.mkObj [("comb", hxs full)])
    | some (_, pub) =>
      let comb ← pub.folded alpha beta lw rho ts
      if comb.size != 64 then throw "the public fold has the wrong width"
      reps := reps.push (Json.mkObj [("comb", hxs comb), ("full", hxs full)])
  let out : List (String × Json) := [("m", .num st.m), ("k_log", .num st.kLog), ("extras", .arr (extras.map claimJson)), ("reps", .arr reps)]
  return Json.mkObj (match hide with
    | none => out
    | some (net, _) => out ++ [("hidden", Json.arr #[.str net])])

def main (args : List String) : IO UInt32 := do
  let (args, plain) := match args with
    | "--plain-leaves" :: rest => (rest, true)
    | _ => (args, false)
  let (args, partFile) := match args with
    | "--partition" :: f :: rest => (rest, some f)
    | _ => (args, none)
  let (args, regFile) := match args with
    | "--registered" :: f :: rest => (rest, some f)
    | _ => (args, none)
  let (args, hideNet) := match args with
    | "--hide" :: n :: rest => (rest, some n)
    | _ => (args, none)
  let (args, coins) := match args with
    | [sid, circuit, pub, points, "os"] => ([sid, circuit, pub, points], Coins.os)
    | _ => (args, Coins.seed)
  let [sid, circuit, pub, points] := args
    | IO.eprintln "usage: lean --run InnerFold.lean [--plain-leaves] [--partition FILE] [--registered FILE] [--hide NET] STATEMENT CIRCUIT PUBLIC POINTS [os]"; return 2
  let some tags := Tags.find? sid | IO.eprintln s!"unknown statement {sid}"; return 2
  let tags := tags.withCoins coins
  let tags := if plain then { tags with merkleLeaf := "flock-leaf/sha512-unsalted", pinnedLeafScheme := some leafSchemePlain }
    else tags
  let circuitB ← IO.FS.readBinFile circuit
  let pubB ← IO.FS.readBinFile pub
  let pts ← match Json.parse (← IO.FS.readFile points) with
    | .ok j => pure j
    | .error e => IO.eprintln s!"{points}: not JSON: {e}"; return 2
  let partition ← match partFile with
    | none => pure none
    | some f => match Json.parse (← IO.FS.readFile f) with
      | .ok j => pure (some j)
      | .error e => IO.eprintln s!"--partition {f}: not JSON: {e}"; return 2
  let registered ← match regFile with
    | none => pure none
    | some f => match Json.parse (← IO.FS.readFile f) >>= Registered.ofJson with
      | .ok own => pure (some own)
      | .error e => IO.eprintln s!"--registered {f}: {e}"; return 2
  let (_, sts) ← match Stmt.setupTables tags circuitB pubB coins 1 (partition := partition) with
    | .ok x => pure x
    | .error e => IO.eprintln s!"setup: {e}"; return 1
  let checked := match registered with
    | some own => Registered.checkDescription sts[0]!.c sts[0]!.pub own
    | none => Registered.check sts[0]!.c sts[0]!.pub none false
  if let .error e := checked then IO.eprintln s!"setup: {e}"; return 1
  let st := sts[0]!
  let hide ← match hideNet with
    | none => pure none
    | some net => match st.c.nets.findIdx? (·.1 == net) with
      | some i =>
        let c : Circuit := { st.c with ranges := st.c.ranges.filter (·.kind != Flock.Kind.net i) }
        pure (some (net, ({ st with c := c } : Stmt).circuitFold))
      | none => IO.eprintln s!"--hide {net}: the circuit has no circuit {net}"; return 2
  match run (Setup.ofCircuit st) pts hide with
  | .ok j => IO.println j.compress; return 0
  | .error e => IO.eprintln e; return 1
