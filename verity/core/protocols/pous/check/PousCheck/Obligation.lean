import Flock.Canon
import PousCheck.Setup

/-!
# PoUS as an obligation at test scale: the verdict on the memory challenger's page log

`verity/core/protocols/pous/obligation.py`'s `verdict`, which `tests/test_obligation_difftest.py` holds this to through
`pous-obligation-difftest`. The KV region is `pages` pages of `blocks` P2 blocks at the ladder's 64-bit width, one
segment a page. Page `p`'s version `v` is committed (`commit`), keyed by the memory challenger's coin (`coin`), encoded
under its tag `pageSalt salt p v coin` and stored (`store`, which the verifier decodes and checks against the
commitment, setup's public-encoder check per page); a window's `snapshot` is the root over every page's latest root,
and its `read`s are checked against it. The log is judged in order and rejected at the first event that breaks a rule,
for one of the reasons `obligation.REASONS` lists (`page`, `version`, `recommit`, `coin`, `early`, `restore`, `decode`,
`snapshot`, `order`, `late`, `stale`, `wrong`, `reads`); a log that isn't one is refused.
-/

namespace PousCheck.Obligation

open Lean Flock

def W : Nat := 64
def C0 : Nat := 189
def SALT_BYTES : Nat := 24
def COIN_BYTES : Nat := 32
def MAX : Nat := 2 ^ 16
def TAG_DOMAIN : ByteArray := "verity/pous/kv-page/tag/v0".toUTF8
def COMMIT_DOMAIN : ByteArray := "verity/pous/kv-page/commit/v0".toUTF8

/-- A page's codec: one P2 segment of `blocks` blocks at the page width. -/
def params (blocks : Nat) : P2.Params := ⟨W, C0, blocks, 1⟩

/-- A page version's tag, as the P2 salt its blocks' keys expand from:
`SHA512(TAG_DOMAIN ‖ salt ‖ u64be(page) ‖ u64be(version) ‖ coin)` cut to `SALT_BYTES`. -/
def pageSalt (salt : ByteArray) (page version : Nat) (coin : ByteArray) : ByteArray :=
  (Sha512.hash (TAG_DOMAIN ++ salt ++ Merkle.be page 8 ++ Merkle.be version 8 ++ coin)).extract 0 SALT_BYTES

/-- The commitment to a page's plaintext: `SHA512(COMMIT_DOMAIN ‖ u64be(page) ‖ u64be(version) ‖ plaintext)`. -/
def commitment (page version : Nat) (plain : ByteArray) : ByteArray :=
  Sha512.hash (COMMIT_DOMAIN ++ Merkle.be page 8 ++ Merkle.be version 8 ++ plain)

/-- The root over a page's blocks' leaves under its tag. -/
def pageRoot (tagSalt : ByteArray) (blocks : Array ByteArray) : ByteArray :=
  (Merkle.root (Setup.segmentLeaves tagSalt 0 blocks)).getD .empty

/-- A committed version: its commitment, its coin once drawn, and its tag and root once stored. -/
structure Version where
  c : ByteArray
  coin : Option ByteArray := none
  tagSalt : ByteArray := .empty
  root : Option ByteArray := none

structure State where
  vers : Array (Array Version)
  stored : Array (Option Nat)
  window : Option (Array Nat) := none
  reads : Nat := 0
  windows : Nat := 0
  writes : Nat := 0

abbrev V := Except String

/-- The log's header: the store's salt, the page and block counts and the read deadline. -/
structure Header where
  salt : ByteArray
  pages : Nat
  blocks : Nat
  limit : Nat

def bytesAt (j : Json) (k ctx : String) : V ByteArray := do unhex (← J.str j k ctx)

def bytesArr (j : Json) (k ctx : String) : V (Array ByteArray) := do
  (← J.arr j k ctx).mapM fun x => do unhex (← J.asStr x s!"{ctx}.{k}")

/-- Whether `got` with `path` is block `j` of the version `v` of a page of `n` blocks. -/
def answers (n j : Nat) (got : ByteArray) (path : Array ByteArray) (v : Version) : Bool :=
  match v.root with
  | some r => Merkle.verifyPath r n j (Merkle.packageLeaf v.tagSalt 0 j got) path
  | none => false

/-- One event: a malformed event is refused (the outer `Except`); a judged one gives the next state or the reason it
is rejected for. -/
def step (h : Header) (st : State) (e : Json) : V (Except String State) := do
  let q := params h.blocks
  match ← J.str e "op" "event" with
  | "commit" =>
    let (page, version, c) := (← J.nat e "page" "commit", ← J.nat e "version" "commit", ← bytesAt e "c" "commit")
    if page ≥ h.pages then return .error "page"
    let vs := st.vers[page]!
    if version < vs.size then return .error "recommit"
    if version > vs.size then return .error "version"
    return .ok { st with vers := st.vers.set! page (vs.push { c }) }
  | "coin" =>
    let (page, version, coin) := (← J.nat e "page" "coin", ← J.nat e "version" "coin", ← bytesAt e "coin" "coin")
    if coin.size != COIN_BYTES then throw s!"a coin is {COIN_BYTES} bytes"
    if page ≥ h.pages then return .error "page"
    let vs := st.vers[page]!
    match vs[version]? with
    | some v =>
      if v.coin.isSome then return .error "coin"
      return .ok { st with vers := st.vers.set! page (vs.set! version { v with coin := some coin }) }
    | none => return .error "coin"
  | "store" =>
    let (page, version, bs) := (← J.nat e "page" "store", ← J.nat e "version" "store", ← bytesArr e "blocks" "store")
    if page ≥ h.pages then return .error "page"
    let vs := st.vers[page]!
    let some v := vs[version]? | return .error "early"
    let some coin := v.coin | return .error "early"
    if let some s := st.stored[page]! then
      if version ≤ s then return .error "restore"
    let ts := pageSalt h.salt page version coin
    match Setup.decodeSegment q ts 0 bs with
    | .error _ => return .error "decode"
    | .ok plain =>
      if commitment page version plain != v.c then return .error "decode"
      let v' := { v with tagSalt := ts, root := some (pageRoot ts bs) }
      return .ok { st with vers := st.vers.set! page (vs.set! version v'), stored := st.stored.set! page (some version),
                           writes := st.writes + 1 }
  | "snapshot" =>
    let root ← bytesAt e "root" "snapshot"
    let some table := st.stored.mapM id | return .error "snapshot"
    let roots := (Array.range h.pages).map fun p => (st.vers[p]![table[p]!]!).root.getD .empty
    if Merkle.root roots != some root then return .error "snapshot"
    return .ok { st with window := some table, windows := st.windows + 1 }
  | "read" =>
    let (page, j, got) := (← J.nat e "page" "read", ← J.nat e "block" "read", ← bytesAt e "answer" "read")
    let (path, ns) := (← bytesArr e "path" "read", ← J.nat e "ns" "read")
    let some table := st.window | return .error "order"
    if page ≥ h.pages || j ≥ h.blocks then return .error "page"
    if ns > h.limit then return .error "late"
    if got.size != q.blockBytes then return .error "wrong"
    let vs := st.vers[page]!
    if answers h.blocks j got path vs[table[page]!]! then return .ok { st with reads := st.reads + 1 }
    if (vs.extract 0 table[page]!).any (answers h.blocks j got path) then return .error "stale"
    return .error "wrong"
  | op => throw s!"an event's op is {op}, not one of commit, coin, store, snapshot, read"

/-- The verdict on a log (`obligation.verdict`): `{"verdict": "accept", "reads", "windows", "writes"}`, or
`{"verdict": "reject", "reason", "event"}` at the first event a rule rejects (the log's length for `reads`), or a
refusal of a log that isn't one. -/
def verdict (log : Json) : V Json := do
  let salt ← bytesAt log "salt" "log"
  let (pages, blocks, limit) := (← J.nat log "pages" "log", ← J.nat log "blocks" "log", ← J.nat log "limit_ns" "log")
  if salt.size != SALT_BYTES then throw s!"the salt is {SALT_BYTES} bytes"
  if pages < 1 || pages > MAX || blocks < 1 || blocks > MAX then throw s!"pages and blocks are 1 to {MAX}"
  let events ← J.arr log "events" "log"
  let h : Header := ⟨salt, pages, blocks, limit⟩
  let mut st : State := ⟨Array.replicate pages #[], Array.replicate pages none, none, 0, 0, 0⟩
  let reject (r : String) (i : Nat) : Json :=
    Json.mkObj [("verdict", .str "reject"), ("reason", .str r), ("event", toJson i)]
  for h' : i in [0:events.size] do
    match ← step h st events[i] with
    | .ok s => st := s
    | .error r => return reject r i
  if st.reads == 0 then return reject "reads" events.size
  return Json.mkObj [("verdict", .str "accept"), ("reads", toJson st.reads), ("windows", toJson st.windows),
    ("writes", toJson st.writes)]

end PousCheck.Obligation
