import Flock.Canon
import PoubCheck.Unboxed
import PoubCheck.Window

/-!
# The store binding's decision: `store.py`

What the verifier decides of a registered store bound to a pinned model (`verifier.py --model`), from what `store.py`
reads of the files: the pin from its file's bytes (`pinOf`, `read_pin`'s decision, and `modelPin`, the binding's,
whose pin must be the role's), the checkpoint's digest (`checkpointSha512`), the store's payload salts (`storeSalt`),
the payload root the store has under the verifier's key (`expectedRoot`, from the weights blocks' digests, each
subtree of `2^CHUNK_LOG2` leaves a task's), the binding the verifier starts with (`bind`: the pin and the store's
shape) and the hello's decision (`hold`). Every refusal is `store.py`'s, word for word. The verifier's process decides
the binding with these (`PoubCheck.Verifier`'s `bind` and `hold`), `store.Binding` beside it as the reference.

~~~text
pin        UTF-8; JSON, else JSON lines (split at "\n", a line of spaces, tabs and "\r" skipped), a registry's lines of
           the role pinning one checkpoint (compared as the canonical JSON of a map to strings, any other value one
           more); core's pin, an object of objects, read as its role's entry {files, checkpoint_sha512}; then a
           nonempty name, a nonempty {file: SHA-512 hex} (no "/", no leading "."), checkpoint_sha512 its digest, and
           config.json and a .safetensors file among the files
digest     SHA512(canonical JSON of {file: SHA-512 hex}): keys sorted by code point, "," and ":", ensure_ascii
salt_j     SHAKE256("verity/store/salt/v0\0" ‖ key ‖ SHA512("verity/remote-session/payload-salts/v0") ‖ u64be j), 192 bytes
root       payload_domain(node, N)'s tree, N = E·A + H: leaf j hidden_digest(d, salt_j), d block j's digest in its
           arena (j mod A < W) or a zero block's, then pad leaves up to 2^depth N
hold       the hello's N at least E·A, its layout (if any) the store's regions, its node 1 to 64 bytes, its payload
           root that root
~~~

JSON is what both sides read as JSON: Lean's `Json.parse`, and `json.loads` with numbers opaque and an exponent past
2^64 refused as Lean's parser refuses it, no NaN or Infinity (`store._loads`); a text that escapes a lone surrogate is
not JSON on either side (`loneSurrogate`). No number enters a decision: a pin holds none.

What stays Python (`store.py` alone): reading the files (the pin's bytes; `check`, each pinned file's SHA-512; the
safetensors headers and the tensors' bytes), the arena's layout from config.json and those headers (`tensors`,
`arena`), and the weights blocks' digests (`weight_digests`, `codec.digest` over the arena's bytes). `bind` and `hold`
take the arena as its digests, 64 bytes a block.
-/

namespace PoubCheck.Store

open Lean (Json)
open Flock

def SALT_DOMAIN : ByteArray := tag0 "verity/store/salt/v0"
def PAYLOAD_LABEL : ByteArray := sha512 "verity/remote-session/payload-salts/v0".toUTF8

/-- `store_salt(key, label, c)`, SHAKE256 with its lanes unboxed (`Unboxed.shake256`): one a block of the store. -/
def storeSalt (key label : ByteArray) (c : Nat) : ByteArray :=
  Unboxed.shake256 (SALT_DOMAIN ++ key ++ label ++ be 8 c) SALT_BYTES

private def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

/-- `json.dumps(s)` with `ensure_ascii`: `"` and `\` escaped, `\b \f \n \r \t`, every other character outside space …
`~` as `\uXXXX` in lowercase, one past U+FFFF as its UTF-16 surrogate pair. -/
def pyStr (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    let n := c.toNat
    out := match c with
      | '"' => out ++ "\\\""
      | '\\' => out ++ "\\\\"
      | '\n' => out ++ "\\n"
      | '\r' => out ++ "\\r"
      | '\t' => out ++ "\\t"
      | _ =>
        if n == 8 then out ++ "\\b" else if n == 12 then out ++ "\\f"
        else if 0x20 ≤ n && n ≤ 0x7e then out.push c
        else if n < 0x10000 then out ++ "\\u" ++ hex4 n
        else out ++ "\\u" ++ hex4 (0xD800 + ((n - 0x10000) >>> 10)) ++ "\\u" ++ hex4 (0xDC00 + (n - 0x10000) % 0x400)
  return out ++ "\""

/-- `canonical_json(v)` of a value with no number (a pin holds none); `none` for one with a number. -/
def pyJson : Json → Option String
  | Json.null => some "null"
  | Json.bool b => some (if b then "true" else "false")
  | Json.num _ => none
  | Json.str s => some (pyStr s)
  | Json.arr a => do
    let parts ← a.attach.mapM fun x => pyJson x.1
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | Json.obj kvs => do
    let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← pyJson p.1.2)
    let items := vals.qsort (fun a b => a.1 < b.1)
    pure ("{" ++ ",".intercalate (items.toList.map fun kv => pyStr kv.1 ++ ":" ++ kv.2) ++ "}")
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact sizeOf_lt_of_mem_toArray p.2

/-- `canonical_json(files)` of a checkpoint's `{file: SHA-512 hex}`. -/
def filesJson (files : Array (String × String)) : String :=
  "{" ++ ",".intercalate ((files.qsort (fun a b => a.1 < b.1)).toList.map fun (f, h) => pyStr f ++ ":" ++ pyStr h) ++ "}"

/-- `checkpoint_sha512(files)`, lowercase hex. -/
def checkpointSha512 (files : Array (String × String)) : String := hexOf (sha512 (filesJson files).toUTF8)

private def hexDigit? (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - 48)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87)
  else if 'A' ≤ c && c ≤ 'F' then some (c.toNat - 55)
  else none

/-- Whether a JSON text escapes a UTF-16 surrogate that isn't a high one followed by a low one, which `json.loads`
keeps as a lone code point and Lean's parser reads as U+FFFD. In a text that isn't JSON the answer doesn't matter. -/
def loneSurrogate (text : String) : Bool := Id.run do
  let cs := text.toList.toArray
  let unit := fun (i : Nat) =>
    if i + 4 ≤ cs.size then [0, 1, 2, 3].foldlM (fun acc k => (acc * 16 + ·) <$> hexDigit? cs[i + k]!) 0 else none
  let mut inString := false
  let mut i := 0
  for _ in [0:cs.size] do
    if i ≥ cs.size then break
    let c := cs[i]!
    if !inString then
      if c == '"' then inString := true
      i := i + 1
    else if c == '"' then
      inString := false
      i := i + 1
    else if c == '\\' && cs[i + 1]? == some 'u' then
      let some v := unit (i + 2) | return false
      if 0xDC00 ≤ v && v < 0xE000 then return true
      if 0xD800 ≤ v && v < 0xDC00 then
        if cs[i + 6]? != some '\\' || cs[i + 7]? != some 'u' then return true
        let some w := unit (i + 8) | return false
        if !(0xDC00 ≤ w && w < 0xE000) then return true
        i := i + 12
      else
        i := i + 6
    else if c == '\\' then
      i := i + 2
    else
      i := i + 1
  return false

/-- A text as `store._loads` reads it. -/
def loads (text : String) : Option Json :=
  match Json.parse text with
  | .ok j => if loneSurrogate text then none else some j
  | .error _ => none

/-- A pin: the model's name, its checkpoint's files with their SHA-512s (lowercase hex) by name, and
`checkpoint_sha512`. -/
structure Pin where
  name : String
  checkpoint : Array (String × String)
  checkpointSha512 : String

def Pin.toJson (p : Pin) : Json :=
  Json.mkObj [("name", .str p.name), ("checkpoint", Json.mkObj (p.checkpoint.toList.map fun (f, h) => (f, .str h))),
    ("checkpoint_sha512", .str p.checkpointSha512)]

private def isObj : Json → Bool
  | .obj _ => true
  | _ => false

/-- `d.get(k, default)` of an object. -/
private def getOr (j : Json) (k : String) (otherwise : Json) : Json :=
  match j.getObjVal? k with
  | .ok v => v
  | .error _ => otherwise

private def field (j : Json) (k : String) : Json := getOr j k .null

/-- `store._pinned`: a registry line's checkpoint as its lines are compared. -/
private def pinned (checkpoint : Json) : Option String :=
  match checkpoint with
  | .obj kvs => if kvs.toArray.all (fun p => match p.2 with | .str _ => true | _ => false) then pyJson checkpoint
      else none
  | _ => none

private def isSha512Hex (h : String) : Bool :=
  h.length == 128 && h.all fun c => ('0' ≤ c && c ≤ '9') || ('a' ≤ c && c ≤ 'f')

private def isFile (f : String) : Bool := !f.isEmpty && !f.contains '/' && !f.startsWith "."

private def short (h : String) : String := String.ofList (h.toList.take 16)

/-- `pin_of(data, path, role)`: the pin the file at `path` holds, or why not. -/
def pinOf (data : ByteArray) (path : String) (role : Option String) : Except String Pin := do
  let some text := String.fromUTF8? data | throw s!"the pin {path} is not UTF-8"
  let mut doc ← match loads text with
    | some d => pure d
    | none => do
      let mut lines : Array Json := #[]
      let mut k := 0
      for x in text.splitOn "\n" do
        k := k + 1
        if x.any fun c => c != ' ' && c != '\t' && c != '\r' then
          match loads x with
          | some j => lines := lines.push j
          | none => throw s!"the pin {path} is neither JSON nor JSON lines: its line {k} is not JSON"
      let mine := lines.filter fun x => isObj x && (match role with
        | some r => getOr x "model" (field x "name") == .str r
        | none => false)
      let keys := mine.foldl (fun acc x =>
        let key := pinned (field x "checkpoint")
        if acc.contains key then acc else acc.push key) (#[] : Array (Option String))
      if keys.size ≠ 1 then
        throw s!"the registry {path} has {mine.size} lines of {role.getD "None"}, pinning {keys.size} checkpoints: one is the pin"
      pure mine.back!
  let mut form := "{name, checkpoint: {file: SHA-512 hex}, checkpoint_sha512}"
  if let .obj kvs := doc then
    let items := kvs.toArray
    if items.size > 0 && items.all (fun p => isObj p.2) then
      let roles := "[" ++ ",".intercalate (((items.map (·.1)).qsort (· < ·)).toList.map pyStr) ++ "]"
      let some r := role | throw s!"the pin {path} pins no role null: its roles are {roles}"
      let .ok entry := doc.getObjVal? r | throw s!"the pin {path} pins no role {pyStr r}: its roles are {roles}"
      form := "{ROLE: {checkpoint_sha512, files: {file: SHA-512 hex}, ...}}"
      doc := Json.mkObj [("name", .str r), ("checkpoint", field entry "files"),
        ("checkpoint_sha512", field entry "checkpoint_sha512")]
  let malformed := s!"the pin {path} is not {form} with a file name (no directory) a file"
  let name := if isObj doc then getOr doc "name" (field doc "model") else .null
  let files := if isObj doc then field doc "checkpoint" else .null
  let .str n := name | throw malformed
  let .obj fs := files | throw malformed
  let some checkpoint := fs.toArray.mapM (fun (f, h) => match h with
      | .str h => if isFile f && isSha512Hex h then some (f, h) else none
      | _ => none) | throw malformed
  if n.isEmpty || checkpoint.isEmpty then throw malformed
  let digest := checkpointSha512 checkpoint
  if field doc "checkpoint_sha512" != .str digest then
    throw s!"the pin {path}'s checkpoint_sha512 is not the SHA-512 of its checkpoint's canonical JSON ({short digest}...)"
  if !checkpoint.any (·.1 == "config.json") || !checkpoint.any (·.1.endsWith ".safetensors") then
    throw s!"the pin {path} pins no config.json or no .safetensors file: the layout reads both"
  pure { name := n, checkpoint := checkpoint.qsort (fun a b => a.1 < b.1), checkpointSha512 := digest }

/-- `model_pin(data, path, role)`: `pinOf`, the pin the binding of `--model role` reads, which must be `role`'s. -/
def modelPin (data : ByteArray) (path role : String) : Except String Pin := do
  let pin ← pinOf data path (some role)
  if pin.name ≠ role then throw s!"the pin {path} is {pin.name}'s, not --model {role}'s"
  pure pin

/-- A region of the store as a hello's layout names it (`verifier.regions_of`): kind, tier, first block, blocks. -/
abbrev Region := String × String × Int × Int

/-- `Store.regions`: each arena's weights blocks (weights, hbm) and the rest of it (filler, hbm), then the host
tier (filler, host). -/
def regions (engines arenaBlocks weightsBlocks hostBlocks : Nat) : Array Region := Id.run do
  let mut out : Array Region := #[]
  for e in [0:engines] do
    out := out.push ("weights", "hbm", ((e * arenaBlocks : Nat) : Int), (weightsBlocks : Int))
    if arenaBlocks > weightsBlocks then
      out := out.push ("filler", "hbm", ((e * arenaBlocks + weightsBlocks : Nat) : Int),
        ((arenaBlocks - weightsBlocks : Nat) : Int))
  if hostBlocks > 0 then
    out := out.push ("filler", "host", ((engines * arenaBlocks : Nat) : Int), (hostBlocks : Int))
  return out

/-- Python's repr of a list's first four regions (`list(layout)[:4]`), whose kinds and tiers (`regions_of`'s) are
letters alone. -/
def reprRegions (rs : Array Region) : String :=
  "[" ++ ", ".intercalate ((rs.toList.take 4).map fun (k, t, f, n) => s!"('{k}', '{t}', {f}, {n})") ++ "]"

/-- `payload_domain(node, N)`. -/
def payloadDomain (node : ByteArray) (blocks : Nat) : FrameV3.Domain :=
  FrameV3.Domain.make (sha512 (PAYLOAD_DOMAIN ++ frameBytes node)) 0 0 blocks

/-- `codec.digest` of a block of zeros. -/
def ZERO_DIGEST : ByteArray := Codec.digest (zeros Codec.BLOCK_BYTES)

/-- `store.CHUNK_LOG2`: a subtree's leaves, as a power of two. -/
def CHUNK_LOG2 : Nat := 16

/-- The weights blocks `digests` holds, 64 bytes a block. -/
def weightsOf (digests : ByteArray) : Nat := digests.size / 64

/-- Leaf `j` of the payload tree over `N` blocks, `engines` arenas of `arenaBlocks` (each its weights blocks, `digests`
64 bytes a block, then zeros) and then zeros, salted under `key`; a pad leaf past `N`. -/
def payloadLeaf (d : FrameV3.Domain) (engines arenaBlocks N : Nat) (digests key : ByteArray) (j : Nat) : ByteArray :=
  if j ≥ N then d.pad j
  else
    let W := weightsOf digests
    let r := if j < engines * arenaBlocks then j % arenaBlocks else W
    let x := if r < W then digests.extract (64 * r) (64 * r + 64) else ZERO_DIGEST
    d.leaf j j SCHEMA (Unboxed.hiddenDigest x (storeSalt key PAYLOAD_LABEL j))

/-- `store._up(dom, level, height, first)`: a level of nodes at `height`, the first of them under leaf `first`,
climbed to its root. -/
def climbUp (d : FrameV3.Domain) (level : Array ByteArray) (height first : Nat) : ByteArray := Id.run do
  let mut level := level
  let mut h := height
  for _ in [0:level.size] do
    if level.size ≤ 1 then break
    let base := first >>> (h + 1)
    level := (Array.range (level.size / 2)).map fun i => d.node h (base + i) level[2 * i]! level[2 * i + 1]!
    h := h + 1
  return level[0]!

/-- `store._subtree`: the root of the subtree over leaves `first … first + 2^log2 − 1`. -/
def subtree (d : FrameV3.Domain) (engines arenaBlocks N : Nat) (digests key : ByteArray) (first log2 : Nat) :
    ByteArray :=
  climbUp d ((Array.range (2 ^ log2)).map fun i => payloadLeaf d engines arenaBlocks N digests key (first + i)) 0 first

/-- `expected_root(Store(arena, engines, arena_blocks, host_blocks), node, key, digests, workers, chunk_log2)`: the
payload root of `engines` arenas of `arenaBlocks`, each its weights blocks (`digests`, 64 bytes a block) then zeros,
then `hostBlocks` of zeros, salted under `key`. Each subtree of `2^chunkLog2` leaves is a task's, on `workers` threads
(subtree `i` on thread `i mod workers`), and their roots are climbed to the tree's. -/
def expectedRoot (engines arenaBlocks hostBlocks : Nat) (digests node key : ByteArray) (workers : Nat := 1)
    (chunkLog2 : Nat := CHUNK_LOG2) : ByteArray :=
  let N := engines * arenaBlocks + hostBlocks
  let d := payloadDomain node N
  let height := FrameV3.depth N
  let c := min chunkLog2 height
  let chunks := 2 ^ (height - c)
  let w := max 1 (min workers chunks)
  let tasks := (Array.range w).map fun t => Task.spawn (prio := .dedicated) fun _ =>
    (Array.range ((chunks - t + w - 1) / w)).map fun q =>
      subtree d engines arenaBlocks N digests key ((t + q * w) <<< c) c
  let parts := tasks.map Task.get
  climbUp d ((Array.range chunks).map fun i => parts[i % w]![i / w]!) c 0

/-- `Store(arena, engines, arena_blocks)`, as the binding starts: one or more arenas, each holding the weights. -/
def shape (engines arenaBlocks : Int) (weightsBlocks : Nat) : Except String Unit := do
  if engines < 1 then
    throw s!"a store is one or more arenas and no fewer than zero host blocks, not {engines} and 0"
  if arenaBlocks < weightsBlocks then
    throw s!"the weights take {weightsBlocks} blocks, more than an arena's {arenaBlocks}"

/-- What `Binding.hold` leaves: the store's regions when the hello's store is bound, else why not, and the expected
root once it was computed (its `record()`'s `expected_root`). -/
structure Held where
  regions : Except String (Array Region)
  expectedRoot : Option ByteArray

/-- `Binding.hold(node, blocks, root, key, layout)` of the binding of `role`'s pin, its arena's weights blocks'
digests `digests` (64 bytes a block) and its `engines` arenas of `arenaBlocks` (`shape`'s): bound when the hello's
store of `blocks` is the arenas then host filler, its layout (if any) that store's regions and its payload root `root`
that store's under the verifier's key `key`, the root on `workers` threads. -/
def hold (role : String) (digests : ByteArray) (engines arenaBlocks : Nat) (node : ByteArray) (blocks : Int)
    (root key : ByteArray) (layout : Option (Array Region)) (workers : Nat := 1) : Held := Id.run do
  if blocks < ((engines * arenaBlocks : Nat) : Int) then
    return ⟨.error s!"the hello's store is {blocks} blocks, fewer than the pinned model's {engines} arenas of {arenaBlocks}",
      none⟩
  let host := (blocks - ((engines * arenaBlocks : Nat) : Int)).toNat
  let W := weightsOf digests
  let rs := regions engines arenaBlocks W host
  if let some l := layout then
    if l != rs then
      return ⟨.error s!"the hello's layout {reprRegions l}... is not the pinned model {role}'s {reprRegions rs}...", none⟩
  if !(1 ≤ node.size && node.size ≤ 64) then return ⟨.error "a node's id is 1 to 64 bytes", none⟩
  let want := expectedRoot engines arenaBlocks host digests node key workers
  if want != root then
    return ⟨.error (s!"the hello's payload root {short (hexOf root)}... is not the pinned model {role}'s store " ++
      s!"({engines} arenas of {arenaBlocks} blocks, {W} of weights each, then {host} host filler blocks) " ++
      s!"under this verifier's payload key, whose root is {short (hexOf want)}...: register the store " ++
      "`poub.store` lays out, salted with the key the verifier sent"), some want⟩
  return ⟨.ok rs, some want⟩

/-- A binding as the verifier starts it (`Binding(role, …)` past what it reads of the files): `role`'s pin, its
arena's weights blocks' digests (64 bytes a block) and its `engines` arenas of `arenaBlocks`. -/
structure Binding where
  role : String
  pin : Pin
  digests : ByteArray
  engines : Nat
  arenaBlocks : Nat

/-- The binding of `--model role` from its pin file's bytes `data` (read from `path`) and its arena's weights blocks'
digests, or why not: the pin (`modelPin`), then the store's shape (`shape`), in `Binding`'s order. -/
def bind (data : ByteArray) (path role : String) (digests : ByteArray) (engines arenaBlocks : Int) :
    Except String Binding := do
  let pin ← modelPin data path role
  shape engines arenaBlocks (weightsOf digests)
  pure { role, pin, digests, engines := engines.toNat, arenaBlocks := arenaBlocks.toNat }

/-- `Binding.hold` of the binding `b`. -/
def Binding.hold (b : Binding) (node : ByteArray) (blocks : Int) (root key : ByteArray) (layout : Option (Array Region))
    (workers : Nat := 1) : Held :=
  Store.hold b.role b.digests b.engines b.arenaBlocks node blocks root key layout workers

end PoubCheck.Store
