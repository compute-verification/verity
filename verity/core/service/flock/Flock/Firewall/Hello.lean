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

/-- The verifier's own draw-law inputs, as `flock-verify verify` reads them: `--work K` with `--work-table F`
(`Draw.workTableOfJson`), `--stratified K` and `--closure F` (`Draw.closureMapOfJson`), each optional. A stratified or work
draw sets up only on them (`Stmt.setupTables`, U2), so they are the session's statement's as much as its files are. -/
def lawOf (args : List String) :
    IO (Except String (Option (Nat × Array (String × Nat × Nat)) × Option Nat × Option (Std.HashMap Nat (Array Nat)))) := do
  let nat : String → Except String (Option Nat) := fun flag => match Witness.argOf args flag with
    | none => .ok none
    | some k => match k.toNat? with
      | some k => .ok (some k)
      | none => .error s!"{flag} K: K is a natural number"
  let table : Except String (Option (Array (String × Nat × Nat))) ← match Witness.argOf args "--work-table" with
    | none => pure (.ok none)
    | some f => do pure ((← Witness.readJson f) >>= fun j => (Draw.workTableOfJson j).map some)
  let closure : Except String (Option (Std.HashMap Nat (Array Nat))) ← match Witness.argOf args "--closure" with
    | none => pure (.ok none)
    | some f => do pure ((← Witness.readJson f) >>= fun j => (Draw.closureMapOfJson j).map some)
  return do
    let work ← match ← nat "--work", ← table with
      | some k, some t => pure (some (k, t))
      | none, none => pure none
      | some _, none => throw "--work needs --work-table (the verifier's own work per template)"
      | none, some _ => throw "--work-table needs --work"
    pure (work, ← nat "--stratified", ← closure)

/-- A firewalled statement's `Hello` inputs, read once: `flock-verify verify --zk`'s statement of its files
(`Witness.Inputs`) and the draw law's inputs (`lawOf`). -/
structure Setup where
  I : Witness.Inputs
  work : Option (Nat × Array (String × Nat × Nat))
  stratified : Option Nat
  closure : Option (Std.HashMap Nat (Array Nat))

/-- The statement flags a `Setup` reads. -/
def FLAGS : String :=
  "[--statement ID] --circuit F --public F [--tables DIR] [--session-tables J] [--partition F] [--program F] [--stratified K] [--work K --work-table F] [--closure F]"

/-- The `Setup` the statement flags `args` name, or why they name none (`usage`, the usage line). -/
def Setup.ofArgs (usage : String) (args : List String) : IO (Except String Setup) := do
  let (some cf, some pf) := (Witness.argOf args "--circuit", Witness.argOf args "--public") | return .error usage
  let some tags := Tags.find? ((Witness.argOf args "--statement").getD Tags.circuit.name)
    | return .error s!"unknown statement; known: {Tags.all.map (·.name)}"
  if tags.plainLeaves then
    return .error s!"hello: refused: {tags.name}, a statement with plain SHA-512 leaves, has no --zk session"
  let some count := ((Witness.argOf args "--session-tables").getD "1").toNat? | return .error usage
  let (partition, program) ← match ← Witness.bindingsOf args with
    | .ok b => pure b
    | .error e => return .error e
  let (work, stratified, closure) ← match ← lawOf args with
    | .ok l => pure l
    | .error e => return .error s!"hello: {e}"
  let circuitFile ← IO.FS.readBinFile cf
  let publicFile ← IO.FS.readBinFile pf
  let tables ← Witness.readTables args
  return .ok { I := { tags, circuitFile, publicFile, count, tables, partition, program }, work, stratified, closure }

/-- The session's spec and its tables' statements at the verifier's draw `draw` (`Stmt.setupTables`, `setup_eq`; the public
file the verifier's own, without rows), its circuit parsed with each net's rows in tasks (`HmOut.parseCommitFast`). -/
def Setup.setup (S : Setup) (draw : Option Json) : V (SessionSpec × Array Stmt) :=
  Stmt.setupTablesOf (HmOut.parseCommitFast (Witness.zkTags S.I) S.I.circuitFile S.I.tables) (Witness.zkTags S.I) S.I.circuitFile
    S.I.publicFile (Witness.zkCoins S.I) S.I.count S.I.tables draw S.I.partition S.I.program S.work S.stratified S.closure

theorem Setup.setup_eq (S : Setup) (draw : Option Json) :
    S.setup draw = Stmt.setupTables (Witness.zkTags S.I) S.I.circuitFile S.I.publicFile (Witness.zkCoins S.I) S.I.count
      S.I.tables draw S.I.partition S.I.program S.work S.stratified S.closure := by
  rw [Setup.setup, Stmt.setupTables_eq, HmOut.parseCommit_eq_fast]

/-- The session's spec at `draw`. -/
def Setup.spec (S : Setup) (draw : Option Json) : V SessionSpec := (S.setup draw).map (·.1)

/-- The key a draw's statement is set up once under: its canonical text, `null` without one. -/
def drawKey (draw : Option Json) : String := (draw.map fun d => (canon d).toOption.getD "").getD "null"

/-- The `Hello` text at the set-up statement `spec` and the nonce `nonce` (`json`), or why there is none: a statement the
setup refused, one without a coin tree, or a `Hello` that is not JSON. -/
def textAt (spec : V SessionSpec) (nonce : ByteArray) : Except String String :=
  match spec with
  | .error e => .error s!"setup: {e}"
  | .ok spec =>
    if spec.coinTree.isNone then .error "the session has no coin tree"
    else json spec (hexOf nonce)

/-- The statements set up so far, by `drawKey`: one process sets each draw's statement up once. -/
abbrev Specs := IO.Ref (Array (String × V (SessionSpec × Array Stmt)))

/-- The spec and statements at `draw` (`Setup.setup`), set up on first use. -/
def Setup.setupAt (S : Setup) (specs : Specs) (draw : Option Json) : IO (V (SessionSpec × Array Stmt)) := do
  let key := drawKey draw
  match (← specs.get).find? (·.1 == key) with
  | some (_, s) => pure s
  | none =>
    let s := S.setup draw
    specs.modify (·.push (key, s))
    pure s

/-- The spec at `draw`, set up on first use. -/
def Setup.specAt (S : Setup) (specs : Specs) (draw : Option Json) : IO (V SessionSpec) :=
  return (← S.setupAt specs draw).map (·.1)

/-- `flock-firewall hello [--statement ID] --circuit F --public F [--tables DIR] [--session-tables J] [--partition F]
[--program F] [--stratified K] [--work K --work-table F] [--closure F]`: each session's `Hello`, served on stdin and stdout,
at `flock-verify verify --zk`'s statement of those files (`Setup`).
`draw LEN` followed by the `LEN` bytes of the session's draw (`drawOf`) is answered `ok N` followed by the `N` bytes of
the session's `Hello` (`json`), at a nonce of 32 bytes this command draws from the operating system; `draw LEN NONCE`
takes the nonce's 64 hex digits instead, for a simulator's rewinds, which restart a session after its `Hello`. Each
draw's statement is set up once. A statement the setup refuses, or one without a coin tree, is answered `refused WHY`,
and the session goes on; anything else (a malformed line, a draw that is not JSON) is answered `err WHY` and ends it. -/
def cmd (args : List String) : IO UInt32 := do
  let S ← match ← Setup.ofArgs s!"usage: flock-firewall hello {FLAGS}" args with
    | .ok S => pure S
    | .error e => IO.eprintln e; return 2
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let say := fun (line : String) => do
    stdout.putStrLn (line.replace "\n" " ")
    stdout.flush
  let specs : Specs ← IO.mkRef #[]
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
    match textAt (← S.specAt specs draw) nonce with
    | .error e => say s!"refused {e}"
    | .ok h =>
      let u := h.toUTF8
      say s!"ok {u.size}"
      stdout.write u
      stdout.flush
  return 0

end Flock.Firewall.Hello
