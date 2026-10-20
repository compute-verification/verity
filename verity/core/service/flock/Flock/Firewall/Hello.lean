/-
# The session's `Hello`, made in Lean (live PROTOCOL.md §9, §10)

A `--zk` session's first message is the prover's `Hello`: the verifier's own session parameters (`Stmt.helloOf`, at the
session's statement after the verifier's draw) with the prover's fresh coin-tree nonce in `coins.nonce`, the text R7
compares byte for byte (`CoinTree.checkHello`). Under the firewall that message is made here, from the statement's own
files, and never by the prover's workers.

* `json`: the `Hello` at a nonce's hex, canonical (`canon`); a session without a coin tree, its own.
* `frame`: `Req::Hello`'s frame of it: tag `1`, `u32le` of its UTF-8 length, its UTF-8 bytes. `msg` is both.
* `cmd`: `flock-firewall hello`, served on stdin and stdout, which draws each session's nonce from the operating system.

The analysis holds these to the coin layer's `Hello` (`Claims.ZKMal.CoinLayer.helloD`, `Claims/ZK/HelloLean.lean`).
-/
import Flock.Witness
import Flock.Firewall.Coins

namespace Flock.Firewall.Hello

open Lean (Json)

/-- **The session's `Hello` at the nonce's hex** `nonce`: the verifier's own (`SessionSpec.hello`) with `nonce` in
`coins.nonce`, canonical, which is what `CoinTree.checkHello` compares the received one to; a session without a coin tree,
its own `Hello`. -/
def json (spec : SessionSpec) (nonce : String) : V String :=
  match spec.coinTree with
  | none => pure spec.hello
  | some _ => do
    let some own := (Json.parse spec.hello).toOption | throw "S2/R7: the verifier's Hello is not JSON"
    let coins := (own.getObjVal? "coins").toOption.getD Json.null
    canon (own.setObjVal! "coins" (coins.setObjVal! "nonce" (Json.str nonce)))

/-- `Req::Hello`'s frame of the text `h`: tag `1`, `u32le` of its UTF-8 length, its UTF-8 bytes. -/
def frame (h : String) : List UInt8 := 1 :: ((Coins.u32be h.utf8ByteSize).reverse ++ h.toUTF8.data.toList)

/-- **The `Hello` message** of the session `spec` at the nonce `nonce`: `json` framed; the empty message where it fails. -/
def msg (spec : SessionSpec) (nonce : ByteArray) : List UInt8 :=
  match json spec (hexOf nonce) with
  | .ok h => frame h
  | .error _ => []

/-- The bytes of a coin-tree nonce. -/
def NONCE_BYTES : Nat := 32

/-- A request's draw: the verifier's answer to `Register`, JSON text, `null` where it drew none. -/
def drawOf (text : String) : Except String (Option Json) :=
  match Json.parse text with
  | .ok Json.null => .ok none
  | .ok d => .ok (some d)
  | .error e => .error s!"the draw is not JSON: {e}"

/-- The nonce a request names: 64 lowercase hex digits. -/
def nonceOf (hex : String) : Except String ByteArray :=
  if CoinTree.isHex (2 * NONCE_BYTES) hex then unhex hex
  else .error s!"a nonce `{hex}` that is not {2 * NONCE_BYTES} lowercase hex digits"

/-- `flock-firewall hello [--statement ID] --circuit F --public F [--tables DIR] [--session-tables J] [--partition F]
[--program F]`: each session's `Hello`, served on stdin and stdout, at `flock-verify verify --zk`'s statement of those
files (`Witness.zkTags`, `zkCoins`, `Stmt.setupTables`; the public file the verifier's own, without rows).
`draw LEN` followed by the `LEN` bytes of the session's draw (`drawOf`) is answered `ok N` followed by the `N` bytes of
the session's `Hello` (`json`), at a nonce of 32 bytes this command draws from the operating system; `draw LEN NONCE`
takes the nonce's 64 hex digits instead, for a simulator's rewinds, which restart a session after its `Hello`. Each
draw's statement is set up once. A statement the setup refuses, or one without a coin tree, is answered `refused WHY`,
and the session goes on; anything else (a malformed line, a draw that is not JSON) is answered `err WHY` and ends it. -/
def cmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-firewall hello [--statement ID] --circuit F --public F [--tables DIR] [--session-tables J] [--partition F] [--program F]"
  let (some cf, some pf) := (Witness.argOf args "--circuit", Witness.argOf args "--public") | IO.eprintln usage; return 2
  let some tags := Tags.find? ((Witness.argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  if tags.plainLeaves then
    IO.eprintln s!"hello: refused: {tags.name}, a statement with plain SHA-512 leaves, has no --zk session"; return 2
  let some count := ((Witness.argOf args "--session-tables").getD "1").toNat? | IO.eprintln usage; return 2
  let (partition, program) ← match ← Witness.bindingsOf args with
    | .ok b => pure b
    | .error e => IO.eprintln e; return 2
  let circuitFile ← IO.FS.readBinFile cf
  let publicFile ← IO.FS.readBinFile pf
  let tables ← Witness.readTables args
  let I : Witness.Inputs := { tags, circuitFile, publicFile, count, tables, partition, program }
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let say := fun (line : String) => do
    stdout.putStrLn (line.replace "\n" " ")
    stdout.flush
  let mut specs : Array (String × V SessionSpec) := #[]
  repeat
    let line := (← stdin.getLine).replace "\n" ""
    if line.isEmpty then break
    let req : Option (Option Nat × Option String) := match line.splitOn " " with
      | ["draw", len] => some (len.toNat?, none)
      | ["draw", len, nonce] => some (len.toNat?, some nonce)
      | _ => none
    let some (some len, given) := req | say s!"err a request `{line}`"; return 1
    let b ← Coins.readExact "hello" stdin len
    let some text := String.fromUTF8? b | say "err a draw that is not UTF-8"; return 1
    let draw ← match drawOf text with
      | .ok d => pure d
      | .error e => say s!"err {e}"; return 1
    let nonce ← match given with
      | none => IO.getRandomBytes NONCE_BYTES.toUSize
      | some h => match nonceOf h with
        | .ok n => pure n
        | .error e => say s!"err {e}"; return 1
    let key := (draw.map fun d => (canon d).toOption.getD "").getD "null"
    let spec ← match specs.find? (·.1 == key) with
      | some (_, s) => pure s
      | none => do
        let s := (Stmt.setupTables (Witness.zkTags I) I.circuitFile I.publicFile (Witness.zkCoins I) I.count I.tables draw
          I.partition I.program).map (·.1)
        specs := specs.push (key, s)
        pure s
    match spec with
    | .error e => say s!"refused setup: {e}"
    | .ok spec =>
      if spec.coinTree.isNone then say "refused the session has no coin tree" else
      match json spec (hexOf nonce) with
      | .error e => say s!"refused {e}"
      | .ok h =>
        let u := h.toUTF8
        say s!"ok {u.size}"
        stdout.write u
        stdout.flush
  return 0

end Flock.Firewall.Hello
