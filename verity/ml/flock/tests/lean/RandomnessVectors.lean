import Proofs.Flock.Soundness.PlanDraw

/-!
Vectors for `verity.randomness` v2 and the PoUW window draw, from the soundness package's Lean spec
(`FlockSoundness/Randomness.lean`, `FlockSoundness/PlanDraw.lean`), for `tests/fixtures/randomness_lean.json`.

* `kat`: `verity/primitives/randomness/tests/vectors.json` recomputed from its inputs, which the tests hold equal to it.
* `extra`: keys, streams, `uniform` and `subset` values on inputs of the spec's own, among them PoUW's closure-draw domain
  and `plan.draw`'s `("call", forward, position, "stratum", name)` contexts, which Python's `Key` must match.
* `window`: `PlanDraw.window` on small windows: the draw of each, and a repeated call index refused.

Regenerate with `tests/lean/generate.sh`.
-/

open Lean FlockSoundness.Randomness FlockSoundness.PlanDraw

def hexOf (b : ByteArray) : String :=
  let d := "0123456789abcdef".toList.toArray
  String.ofList (b.toList.flatMap fun x => [d[(x.toNat / 16)]!, d[(x.toNat % 16)]!])

def unhex (s : String) : ByteArray :=
  let v := fun (c : Char) => if c.isDigit then c.toNat - 48 else c.toNat - 87
  let cs := s.toList.toArray
  ⟨(Array.range (cs.size / 2)).map fun i => (16 * v cs[2 * i]! + v cs[2 * i + 1]!).toUInt8⟩

def partOfJson (j : Json) : Part :=
  match j with
  | .str s => .str s
  | .num n => .int n.mantissa.toNat
  | _ => match j.getObjVal? "bytes", j.getObjVal? "int", j.getObjVal? "str" with
    | .ok (.str h), _, _ => .bytes (unhex h)
    | _, .ok (.num n), _ => .int n.mantissa.toNat
    | _, _, .ok (.str s) => .str s
    | _, _, _ => .int 0

def partToJson : Part → Json
  | .bytes b => Json.mkObj [("bytes", .str (hexOf b))]
  | .str s => .str s
  | .int n => .num (JsonNumber.fromNat n)

def indexOfJson (j : Json) : List Part :=
  match j with
  | .arr a => a.toList.map partOfJson
  | _ => [partOfJson j]

def partOfRepr (r : String) : Part :=
  if r.startsWith "b'" then .bytes ((r.drop 2).dropEnd 1).toString.toUTF8
  else if r.startsWith "'" then .str ((r.drop 1).dropEnd 1).toString
  else .int r.toNat!

/-- Bytes read per stream: every vector below finishes well inside it. -/
def L : Nat := 4096

def num (n : Nat) : Json := .num (JsonNumber.fromNat n)

def kat (j : Json) : Json := Id.run do
  let .ok (.arr fr) := j.getObjVal? "frame" | return .null
  let .ok (.arr cases) := j.getObjVal? "cases" | return .null
  let frames := fr.map fun v => match v with
    | .arr #[.str r, _] => Json.arr #[.str r, .str (hexOf (frame (partOfRepr r)))]
    | _ => Json.null
  let cs := cases.map fun c =>
    let src := unhex ((c.getObjValAs? String "source").toOption.getD "")
    let dom := (c.getObjValAs? String "domain").toOption.getD ""
    let ctx : List (String × Part) := match c.getObjVal? "context" with
      | .ok (.obj kvs) => kvs.toArray.toList.map fun (k, v) => (k, partOfJson v)
      | _ => []
    let key := derive src dom ctx
    let sidx := ((c.getObjVal? "stream").toOption.bind fun s => (s.getObjVal? "index").toOption).getD Json.null
    let slen := ((c.getObjVal? "stream").toOption.bind fun s => (s.getObjValAs? String "bytes").toOption).getD ""
    let hidx := ((c.getObjVal? "shard").toOption.bind fun s => (s.getObjVal? "index").toOption).getD Json.null
    let us := match c.getObjVal? "uniform" with
      | .ok (.arr a) => a.map fun u => match u with
        | .arr #[.num n, i, _] => Json.arr #[.num n, i, match uniform key n.mantissa.toNat (indexOfJson i) L with
            | some v => num v | none => .null]
        | _ => .null
      | _ => #[]
    let bs := match c.getObjVal? "bernoulli" with
      | .ok (.arr a) => a.map fun u => match u with
        | .arr #[.str p, i, _] =>
          let (nu, de) := match p.splitOn "/" with
            | [a, b] => (a.toNat!, b.toNat!)
            | [a] => (a.toNat!, 1)
            | _ => (0, 1)
          Json.arr #[.str p, i, match bernoulli key nu de (indexOfJson i) L with
            | some v => .bool v | none => .null]
        | _ => .null
      | _ => #[]
    let ss := match c.getObjVal? "subset" with
      | .ok (.arr a) => a.map fun u => match u with
        | .arr #[.num k, .num n, i, _] =>
          Json.arr #[.num k, .num n, i, match subset key k.mantissa.toNat n.mantissa.toNat (indexOfJson i) L with
            | some v => .arr (v.map num) | none => .null]
        | _ => .null
      | _ => #[]
    Json.mkObj [("source", c.getObjValD "source"), ("domain", c.getObjValD "domain"), ("context", c.getObjValD "context"),
      ("key", .str (hexOf key)),
      ("stream", Json.mkObj [("index", sidx), ("bytes", .str (hexOf (stream key (indexOfJson sidx) (slen.length / 2))))]),
      ("shard", Json.mkObj [("index", hidx), ("key", .str (hexOf (shard key (indexOfJson hidx))))]),
      ("uniform", .arr us), ("bernoulli", .arr bs), ("subset", .arr ss)]
  return Json.mkObj [("frame", .arr frames), ("cases", .arr cs)]

/-- The spec's own inputs: `(source, domain, context)`. -/
def keys : List (ByteArray × String × List (String × Part)) :=
  [("verifier-secret".toUTF8, "verity/pouw/closure-draw/v1", [("receipt", .bytes (unhex "00112233445566778899aabbccddeeff"))]),
   ("verifier-secret".toUTF8, "verity/pouw/closure-draw/v1", [("receipt", .bytes (unhex "ffeeddccbbaa99887766554433221100"))]),
   ("beacon-round-7".toUTF8, "verity/sampled-proofs/ru-sample/v1", [("epoch", .int 7), ("program", .str "p")])]

/-- `(k, n, index)`: subsets on PoUW's contexts and a few others. -/
def subsets : List (Nat × Nat × List Part) :=
  [(3, 10, context [.int 0, .int 5] "NcpTile_v1"), (1, 1, context [.int 0, .int 5] "NcpDigest_v1"),
   (0, 7, context [.int 1, .int 2] "NcpTile_v1"), (7, 7, context [.int 2, .int 0] "NcpFormX_v1"),
   (12, 6171648, context [.int 3, .int 9] "NcpTile_v1"), (5, 300, context [] "NcpTile_v1"),
   (4, 1000, [.str "s", .int 3]), (2, 256, [.bytes (unhex "00ff"), .str "é", .int 256])]

def uniforms : List (Nat × List Part) :=
  [(1, [.int 0]), (6171648, [.str "call", .int 3]), (170141183460469231731687303715884105729, [.int 9]), (1000, [])]

def extra : Json := .arr <| keys.toArray.map fun (src, dom, ctx) =>
  let key := derive src dom ctx
  Json.mkObj [("source", .str (hexOf src)), ("domain", .str dom),
    ("context", Json.mkObj (ctx.map fun (k, v) => (k, partToJson v))), ("key", .str (hexOf key)),
    ("streams", .arr <| (subsets.toArray.map fun (_, _, idx) => (swapIndex idx 0)).map fun idx =>
      Json.mkObj [("index", .arr (idx.toArray.map partToJson)), ("bytes", .str (hexOf (stream key idx 100)))]),
    ("uniform", .arr <| uniforms.toArray.map fun (n, idx) =>
      Json.arr #[num n, .arr (idx.toArray.map partToJson), match uniform key n idx L with
        | some v => num v | none => .null]),
    ("subset", .arr <| subsets.toArray.map fun (k, n, idx) =>
      Json.arr #[num k, num n, .arr (idx.toArray.map partToJson), match subset key k n idx L with
        | some v => .arr (v.map num) | none => .null])]

def stratumJson (s : Stratum) : Json :=
  Json.mkObj [("name", .str s.name), ("units", .arr (s.units.toArray.map num)), ("k", num s.k)]

/-- Small windows: two calls, and the same with a call index repeated. -/
def windows : List (List Call) :=
  [[⟨[.int 0, .int 5], [⟨"NcpTile_v1", [0, 1, 2, 3, 4, 5], 2⟩, ⟨"NcpDigest_v1", [6, 7], 1⟩]⟩,
    ⟨[.int 1, .int 5], [⟨"NcpTile_v1", [8, 9, 10, 11], 2⟩, ⟨"NcpDigest_v1", [12], 1⟩]⟩],
   [⟨[.int 0, .int 5], [⟨"NcpTile_v1", [0, 1, 2], 1⟩]⟩, ⟨[.int 0, .int 5], [⟨"NcpTile_v1", [3, 4, 5], 1⟩]⟩]]

def windowVectors : Json := .arr <| windows.toArray.map fun calls =>
  let (src, dom, ctx) := keys.head!
  Json.mkObj [("source", .str (hexOf src)), ("domain", .str dom),
    ("context", Json.mkObj (ctx.map fun (k, v) => (k, partToJson v))),
    ("calls", .arr <| calls.toArray.map fun c =>
      Json.mkObj [("index", .arr (c.index.toArray.map partToJson)), ("strata", .arr (c.strata.toArray.map stratumJson))]),
    ("drawn", match window (derive src dom ctx) calls L with
      | some d => .arr (d.toArray.map num) | none => .null)]

def main (args : List String) : IO Unit := do
  let .ok j := Json.parse (← IO.FS.readFile args[0]!) | throw (IO.userError "vectors.json is not JSON")
  IO.println (Json.mkObj [("source", .str "verity/ml/flock/tests/lean/RandomnessVectors.lean via tests/lean/generate.sh"),
    ("kat", kat j), ("extra", extra), ("window", windowVectors)]).compress
