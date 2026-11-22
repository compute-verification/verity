module

public import Difftest
public import Base.Bytes
public import Circuits.Build
public import Circuits.Encode
public import Circuits.Program

/-!
# The program decoder's port: `Flock.Program.decode` behind `Difftest.serve`

`program-difftest` serves the decoder of `verity/circuits/PROTOCOL.md` §4, schema v2 with its Boolean map (§4.9), so that
`codec.decode_program` is held to it on the format's vectors (`tests/tools/difftest/test_program_decode_port.py`, its cases
`tests/verity/circuits/decode_difftest.py`'s). `decode` reads `{"descriptor": TEXT}` and answers
`{"ok": true, "program_digest": HEX, "boolean": MAP}` or `{"ok": false, "error": CODE}`, CODE the refusal's first `E_` code
or null: a refusal is an output, so which refusal is compared, and not only the fact of one. `encode` answers with
`Flock.Program.Prog.encode` of the decoded Program, which the same test holds to `codec.descriptor_chunks` byte for byte.
`build_eq` and `build_product` answer with the bytes `Flock.Program.Build.finish` writes for the word equality on bits and
for `Gemm_v2{K, N}` at a size, each held to its Python twin's. `input_bits` answers with `Flock.Program.inputBits`,
held to `program.input_bits_of` (`tests/tools/difftest/test_input_bits_port.py`). -/

@[expose] public section

namespace ProgramDifftest

open Lean (Json)

/-- The first `E_[A-Z_]+` of `s` that starts a word (`re.search(r"\bE_[A-Z_]+", s)`), or none. -/
def code? (s : String) : Option String := Id.run do
  let cs := s.toList.toArray
  let word := fun (c : Char) => c.isAlphanum || c == '_'
  for i in [0:cs.size] do
    if cs[i]! == 'E' && cs[i + 1]? == some '_' && (i == 0 || !word cs[i - 1]!) then
      let mut j := i + 2
      while j < cs.size && (cs[j]!.isUpper || cs[j]! == '_') do j := j + 1
      return some (String.ofList (cs.extract i j).toList)
  return none

def decode : Difftest.Fn := fun j => do
  let text ← match j.getObjValAs? String "descriptor" with
    | .ok s => pure s
    | .error _ => throw "descriptor: not a string"
  return match Flock.Program.decode text.toUTF8 with
    | .ok P => Json.mkObj [("ok", true), ("program_digest", Flock.hexOf P.sha512),
        ("boolean", Json.mkObj ((P.boolean.toList.mergeSort (fun a b => a.1 ≤ b.1)).map fun (k, v) => (k, Json.str v)))]
    | .error e => Json.mkObj [("ok", false), ("error", match code? e with | some c => Json.str c | none => Json.null)]

/-- `decode`'s refusal text, for the cases that hold it: `{"ok": true}` or `{"ok": false, "message": TEXT}`. -/
def decodeText : Difftest.Fn := fun j => do
  let text ← match j.getObjValAs? String "descriptor" with
    | .ok s => pure s
    | .error _ => throw "descriptor: not a string"
  return match Flock.Program.decode text.toUTF8 with
    | .ok _ => Json.mkObj [("ok", true)]
    | .error e => Json.mkObj [("ok", false), ("message", Json.str e)]

/-- `decodeWith`, the decoder the verifier's setups run over a hash oracle, at SHA-512 in `Except`: `decode`'s answer
(`{"ok": true, "program_digest", "boolean"}`), or `{"ok": false, "message": TEXT}`. -/
def decodeWithText : Difftest.Fn := fun j => do
  let text ← match j.getObjValAs? String "descriptor" with
    | .ok s => pure s
    | .error _ => throw "descriptor: not a string"
  return match Flock.Program.decodeWith (M := Except String) (fun b => pure (Flock.Sha512.hash b)) text.toUTF8 with
    | .ok P => Json.mkObj [("ok", true), ("program_digest", Flock.hexOf P.sha512),
        ("boolean", Json.mkObj ((P.boolean.toList.mergeSort (fun a b => a.1 ≤ b.1)).map fun (k, v) => (k, Json.str v)))]
    | .error e => Json.mkObj [("ok", false), ("message", Json.str e)]

/-- `encode` of what `decode` reads: `{"ok": true, "encoded": TEXT}`, or `{"ok": false, "message": TEXT}`. -/
def encode : Difftest.Fn := fun j => do
  let text ← match j.getObjValAs? String "descriptor" with
    | .ok s => pure s
    | .error _ => throw "descriptor: not a string"
  return match Flock.Program.decode text.toUTF8 >>= Flock.Program.Prog.encode with
    | .ok b => match String.fromUTF8? b with
      | some s => Json.mkObj [("ok", true), ("encoded", Json.str s)]
      | none => Json.mkObj [("ok", false), ("message", "encode: not UTF-8")]
    | .error e => Json.mkObj [("ok", false), ("message", Json.str e)]

def written (r : Except String (ByteArray × Flock.Program.Prog)) : Json :=
  match r with
  | .ok (b, Q) => match String.fromUTF8? b with
    | some s => Json.mkObj [("ok", true), ("encoded", Json.str s), ("program_digest", Flock.hexOf Q.sha512)]
    | none => Json.mkObj [("ok", false), ("message", "build: not UTF-8")]
  | .error e => Json.mkObj [("ok", false), ("message", Json.str e)]

/-- `Build.eqBits k` as the Program `root` with program function `program_fn`. -/
def buildEq : Difftest.Fn := fun j => do
  let (.ok k, .ok root, .ok fn) := (j.getObjValAs? Nat "k", j.getObjValAs? String "root", j.getObjValAs? String "program_fn")
    | throw "build_eq: {k, root, program_fn}"
  return written (Flock.Program.Build.finish root fn (Flock.Program.Build.eqBits k))

open Flock.Program Flock.Program.Build in
/-- `Gemm_v2{K, N}` with `DOT` the Ampere BF16 tensor-core step, as `Program` lowers it: `x` and `w` the root's input nodes,
the batch of `GemmCoordinate_v2` over `w`'s rows; its primitives and their Boolean Definitions adopted from `src`. -/
def product (src : Prog) (K N : Nat) : M Val := do
  let input ← adopt src "Input16_v1"
  let zero ← adopt src "Const32[0x00000000]_v1"
  let dot ← adopt src "AmpereBF16TcDot16_v2"
  let f2fp ← adopt src "F2fpBf16_v1"
  let rec_ := #[("K", Lean.toJson K), ("DOT", Json.mkObj [("fn", .str "AmpereBF16TcDot16_v2")])]
  let word := Ty.value 16
  let row := Ty.array K word
  let dotBf16 ← composite "DotBf16_v2" rec_ #[("x", row), ("w", row)] fun ps => do
    let (some x, some w) := (ps[0]?, ps[1]?) | throw "DotBf16: two parameters"
    let mut acc ← call zero #[]
    for s in [0:K / 16] do
      acc ← call dot #[acc, ← x.span (.array 16 word) (16 * s) (16 * s + 16), ← w.span (.array 16 word) (16 * s) (16 * s + 16)]
    pure acc
  let coord ← composite "GemmCoordinate_v2" rec_ #[("x", row), ("wrow", row)] fun ps => do
    let (some x, some w) := (ps[0]?, ps[1]?) | throw "GemmCoordinate: two parameters"
    call f2fp #[← call dotBf16 #[x, w]]
  let x ← batch input K #[]
  let w ← batch input (N * K) #[]
  let y ← batch coord N #[(x, none), (← w.as (.array N row), some 0)]
  adoptBooleans src
  pure y

/-- `product` at `K`, `N`, its registry Definitions adopted from the Program `registry` decodes to. -/
def buildProduct : Difftest.Fn := fun j => do
  let (.ok text, .ok k, .ok n) := (j.getObjValAs? String "registry", j.getObjValAs? Nat "K", j.getObjValAs? Nat "N")
    | throw "build_product: {registry, K, N}"
  let some src := (Flock.Program.decode text.toUTF8).toOption | throw "build_product: the registry does not decode"
  let dot := "AmpereBF16TcDot16_v2"
  return written (Flock.Program.Build.finish s!"Root[Gemm_v2\{K={k},N={n},DOT={dot}}]_v1"
    s!"Gemm_v2\{K={k},N={n},DOT=\{\"fn\":\"{dot}\"}}" (product src k n))

/-- `Flock.Program.inputBits` of what `decode` reads, with `rebind` (`{"node", "key", "binding"}`, `binding` a string or
null), where given, setting the key and binding of root node `node`'s callee first, a description no decoder gives:
`{"ok": true, "run", "registration", "supplied"}`, or `{"ok": false, "error": CODE}`, CODE the refusal's text before its
first colon. -/
def inputBits : Difftest.Fn := fun j => do
  let text ← match j.getObjValAs? String "descriptor" with
    | .ok s => pure s
    | .error _ => throw "descriptor: not a string"
  let P ← match Flock.Program.decode text.toUTF8 with
    | .ok P => pure P
    | .error e => throw s!"descriptor: {e}"
  let P ← match j.getObjVal? "rebind" with
    | .error _ => pure P
    | .ok r => do
      let (.ok k, .ok key) := (r.getObjValAs? Nat "node", r.getObjValAs? String "key")
        | throw "rebind: {node, key, binding}"
      let binding := (r.getObjValAs? String "binding").toOption
      let some f := ((P.defs[P.root]?).bind (·.body.nodes[k]?)).map (·.fn) | throw "rebind: no such root node"
      pure { P with defs := P.defs.modify f fun d => { d with key, binding } }
  return match Flock.Program.inputBits P with
    | .ok b => Json.mkObj [("ok", true), ("run", b.run), ("registration", b.registration), ("supplied", b.supplied)]
    | .error e => Json.mkObj [("ok", false), ("error", Json.str ((e.splitOn ":").headD e))]

def table : String → Option Difftest.Fn :=
  Difftest.table [("decode", decode), ("decode_text", decodeText), ("decode_with", decodeWithText), ("encode", encode),
    ("build_eq", buildEq), ("build_product", buildProduct), ("input_bits", inputBits)]

end ProgramDifftest
