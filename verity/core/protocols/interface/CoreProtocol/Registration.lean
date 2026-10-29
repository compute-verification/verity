import CoreProtocol.Protocol
import CoreProtocol.Site
import Flock.Hash

/-!
# The registration log

A deployment's registration is one append-only log of entries, each naming the head of the log before it
(`Entry.previous`): the digest of the log's last entry, or of its composite's genesis when the log is empty. So the head
names the whole log. An entry is one of two kinds, each under its own versioned format:

* **site** (`verity/registration/site/v1`): the Site's own registration entry (`Core.Site.Registration.Entry`), the
  sheets, locations and links it adds to the Site (`Core.Site`), its JSON, digest and check the Site's.
* **information** (`verity/registration/information/v1`): the items it registers, each once in the log (`Item.key`):
  θ, each registered value (weights, initial state, tables) by its committed root, and the builder `B` that compiles
  advice into the circuit (its digest, its advice's format and its public cap `ℓ` on one declaration's advice, in
  bits). Registration registers the site, θ and `B`, not the circuit `B` builds; a subprotocol's Program and
  parameters, and the proof system, are that protocol's public verifier parameters, not registration.

An entry's digest is `SHA-512(format ‖ 0 ‖ canon(entry))` over the entry's canonical JSON, its format and `previous`
among its keys, and the genesis of a composite's log is the Site's (`Core.Site.Registration.sha512`), so a log of site
entries alone has the Site's head. Registration is ASCII-only without DEL: a log or parameters holding any other byte
is refused (`ascii`, in `Core.canonicalJson`'s words) before anything is decoded.

The check (`check`) holds each entry to the log before it, and refuses the first entry that fails by the first check it
fails: a site entry by the Site's checks (`Core.Site.Registration.Entry.refusal`, `previous` first), and an information
entry by `previous` and then `duplicate-information`, no item registered twice. A site entry's checks hold of an
information entry whenever they held of the log before it, which it adds no hardware to.

A component reads its view of the log by projection (`View`): the Site, θ's values and the builder, at the head. As a
core protocol (`protocol`, `verity/registration/log/v1`) the log is the commitment, its head the digest, and the
parameters the auditor's own: the composite and the Site's auditor (`Core.Site.Registration.Auditor`: its data sheets,
the tolerance it allows a sheet's figures around their estimates, and its devices' keys).
-/

namespace Core.Registration

open Lean (Json)
open Flock (V)
open Core.Site.Registration (Composite Auditor)

def INFORMATION : String := "verity/registration/information/v1"
def protocolId : String := "verity/registration/log/v1"

/-- One of θ's registered values (weights, initial state, tables) by its committed root: its name, the root as hex, its
rows (`leaves`) of `words` words of `wordBits` bits (16 or 32), the hex id of the domain its leaves are hashed under,
which binds no window, and for public weights their checkpoint's SHA-512. -/
structure Theta where
  value : String
  root : String
  leaves : Nat
  words : Nat
  wordBits : Nat := 16
  domain : String
  checkpoint : Option String := none
  deriving DecidableEq, Repr

/-- The builder `B`: its digest, the format of the advice it compiles, and its public cap `ℓ` on one declaration's
advice, in bits. -/
structure Builder where
  digest : String
  advice : String
  cap : Nat
  deriving DecidableEq, Repr

inductive Item
  | theta (t : Theta)
  | builder (b : Builder)
  deriving DecidableEq, Repr

/-- What an item registers, which the log registers once: each θ value by its name, and one builder. -/
def Item.key : Item → String
  | .theta t => "theta " ++ t.value
  | .builder _ => "builder"

/-- **An entry of the registration log**: the Site's site entry, or an information entry naming the head of the log
it follows and the items it registers. -/
inductive Entry
  | site (e : Site.Registration.Entry)
  | information (previous : String) (items : List Item)
  deriving DecidableEq

/-- The head of the log the entry follows. -/
def Entry.previous : Entry → String
  | .site e => e.previous
  | .information p _ => p

/-- **The registration log**, oldest entry first. -/
abbrev Log := List Entry

/-! ## Canonical JSON -/

def Item.toJson : Item → Json
  | .theta t => Json.mkObj ((t.checkpoint.map fun c => [("checkpoint_sha512", Json.str c)]).getD [] ++
      [("domain", .str t.domain), ("kind", .str "theta"), ("leaves", Lean.toJson t.leaves), ("root", .str t.root),
       ("value", .str t.value)] ++ (if t.wordBits == 16 then [] else [("word_bits", Lean.toJson t.wordBits)]) ++
      [("words", Lean.toJson t.words)])
  | .builder b => Json.mkObj [("advice", .str b.advice), ("cap", Lean.toJson b.cap), ("digest", .str b.digest),
      ("kind", .str "builder")]

def Entry.format : Entry → String
  | .site _ => Site.SITE
  | .information .. => INFORMATION

def Entry.toJson : Entry → Json
  | .site e => e.toJson
  | .information p items =>
    Json.mkObj [("format", .str INFORMATION), ("items", .arr (items.map Item.toJson).toArray), ("previous", .str p)]

/-- **An entry's digest**: `SHA-512(format ‖ 0 ‖ canon(entry))`, a site entry's the Site's. -/
def Entry.digest (e : Entry) : String := Site.tagged e.format e.toJson

/-- The head of a composite's empty log, the Site's. -/
def genesis (c : Composite) : String := Site.Registration.sha512.genesis c

/-- **The log's head**, which names it: its last entry's digest, or its composite's genesis when it is empty. -/
def head (c : Composite) (log : Log) : String :=
  match log.getLast? with
  | none => genesis c
  | some e => e.digest

/-- A site entry adding `s`, to chain. -/
def site (s : Site.Entry) : String → Entry := fun p => .site (.site p s)

/-- An information entry registering `items`, to chain. -/
def information (items : List Item) : String → Entry := fun p => .information p items

/-- The log of `entries` in order, each following the head of the log before it. -/
def chain (c : Composite) (entries : List (String → Entry)) : Log :=
  (entries.foldl (fun (acc : Log × String) f =>
    let e := f acc.2
    (acc.1 ++ [e], e.digest)) ([], genesis c)).1

/-! ## Decoding: exactly what `toJson` writes -/

open Core.Site (exactly)

def hex64 (j : Json) (what : String) : V String := do
  let s ← Flock.J.asStr j what
  let b ← (Flock.unhex s).mapError (s!"{what}: " ++ ·)
  if b.size != 64 then throw s!"{what}: not a SHA-512 digest (64 bytes)"
  pure s

def Item.ofJson (j : Json) : V Item := do
  match ← Flock.J.str j "kind" "an item" with
  | "theta" =>
    let has := fun k => (j.getObjVal? k).isOk
    exactly j ((if has "checkpoint_sha512" then ["checkpoint_sha512"] else []) ++
      ["domain", "kind", "leaves", "root", "value"] ++ (if has "word_bits" then ["word_bits"] else []) ++ ["words"])
      "a theta item"
    let wordBits ← if has "word_bits" then Flock.J.nat j "word_bits" "a theta item" else pure 16
    if has "word_bits" && wordBits != 32 then throw "a theta item: word_bits is 32 when present (16 is left out)"
    pure (.theta { value := ← Flock.J.str j "value" "a theta item",
                   root := ← hex64 (← Flock.J.get j "root" "a theta item") "a theta item's root",
                   leaves := ← Flock.J.nat j "leaves" "a theta item", words := ← Flock.J.nat j "words" "a theta item",
                   wordBits, domain := ← hex64 (← Flock.J.get j "domain" "a theta item") "a theta item's domain",
                   checkpoint := ← if has "checkpoint_sha512" then
                     some <$> hex64 (← Flock.J.get j "checkpoint_sha512" "a theta item") "a theta item's checkpoint"
                     else pure none })
  | "builder" =>
    exactly j ["advice", "cap", "digest", "kind"] "a builder item"
    pure (.builder { digest := ← hex64 (← Flock.J.get j "digest" "the builder") "the builder's digest",
                     advice := ← Flock.J.str j "advice" "the builder", cap := ← Flock.J.nat j "cap" "the builder" })
  | k => throw s!"an item: kind {k} is not one of [theta, builder]"

def Entry.ofJson (j : Json) : V Entry := do
  let format ← Flock.J.str j "format" "an entry"
  if format == Site.SITE then Entry.site <$> Site.Registration.Entry.ofJson j
  else if format == INFORMATION then
    exactly j ["format", "items", "previous"] "an information entry"
    pure (.information (← Flock.J.str j "previous" "an entry")
      (← (← Flock.J.arr j "items" "an information entry").toList.mapM Item.ofJson))
  else throw s!"an entry: format {format} is not one of [{Site.SITE}, {INFORMATION}]"

/-- The log as committed: its composite and its entries, `{"composite": NAME, "entries": [...]}`. -/
structure Committed where
  composite : Composite
  log : Log

def Committed.toJson (c : Committed) : Json :=
  Json.mkObj [("composite", .str c.composite.name), ("entries", .arr (c.log.map Entry.toJson).toArray)]

def Committed.ofJson (j : Json) : V Committed := do
  exactly j ["composite", "entries"] "the log"
  pure { composite := ← Site.Registration.Composite.ofJson (← Flock.J.get j "composite" "the log"),
         log := ← (← Flock.J.arr j "entries" "the log").toList.mapM Entry.ofJson }

/-! ## The check -/

/-- A refusal: a site entry's, by the Site's check that fails, or `duplicate-information`. -/
inductive Refusal
  | site (r : Site.Registration.Refusal)
  | duplicateInformation
  deriving DecidableEq, Repr

def Refusal.name : Refusal → String
  | .site r => r.name
  | .duplicateInformation => "duplicate-information"

def nodup : List String → Bool
  | [] => true
  | x :: xs => !xs.contains x && nodup xs

/-- The log so far, folded: its Site, its information items in order, its head and its length. -/
structure State where
  site : Site
  items : List Item
  head : String
  entries : Nat

def State.empty (c : Composite) : State := { site := ⟨[], [], []⟩, items := [], head := genesis c, entries := 0 }

/-- The first check the entry `e` fails against the log folded to `s`, or `none`: a site entry's the Site's, under its
deployed digests. -/
def Entry.refusal (A : Auditor) (c : Composite) (s : State) : Entry → Option Refusal
  | .site e => (e.refusal Site.Registration.sha512 A c s.site s.head).map .site
  | .information p items =>
    if p ≠ s.head then some (.site .previous)
    else if nodup ((s.items ++ items).map Item.key) then none else some .duplicateInformation

def State.add (s : State) (e : Entry) : State :=
  match e with
  | .site x => { s with site := s.site.add x.added, head := e.digest, entries := s.entries + 1 }
  | .information _ items => { s with items := s.items ++ items, head := e.digest, entries := s.entries + 1 }

/-- **The check**: the log folded entry by entry, each held to the log before it; or the first entry that fails, by its
index and the first check it fails. -/
def check (A : Auditor) (c : Composite) (log : Log) : Except (Nat × Refusal) State :=
  log.foldlM (fun s e => match e.refusal A c s with
    | some r => .error (s.entries, r)
    | none => .ok (s.add e)) (State.empty c)

/-! ## Views: what a component reads of the log -/

/-- What a component reads of an accepted log, at its head: the Site, θ's registered values in order, and the builder. -/
structure View where
  site : Site
  theta : List Theta
  builder : Option Builder
  head : String

def State.view (s : State) : View :=
  { site := s.site, head := s.head,
    theta := s.items.filterMap fun | .theta t => some t | _ => none,
    builder := s.items.findSome? fun | .builder b => some b | _ => none }

/-- θ's registered value named `value`. -/
def View.value? (v : View) (value : String) : Option Theta := v.theta.find? (·.value == value)

def View.toJson (v : View) : Json :=
  Json.mkObj [("builder", match v.builder with | some b => Item.toJson (.builder b) | none => .null),
    ("head", .str v.head), ("site", v.site.toJson), ("theta", .arr (v.theta.map (Item.toJson ∘ .theta)).toArray)]

/-! ## The core protocol -/

structure Params where
  composite : Composite
  auditor : Auditor

/-- Registration is ASCII without DEL, refused in `Core.canonicalJson`'s words before anything is decoded. -/
def ascii (b : ByteArray) : V Unit :=
  if b.data.any (· ≥ 0x7f) then throw "not ASCII without DEL (a byte at or above 0x7f)" else pure ()

def Params.ofJson (j : Json) : V Params := do
  ascii j.compress.toUTF8
  exactly j ["auditor", "composite"] "the params"
  pure { composite := ← Site.Registration.Composite.ofJson (← Flock.J.get j "composite" "the params"),
         auditor := ← Site.Registration.Auditor.ofJson (← Flock.J.get j "auditor" "the params") }

/-- The log as committed, and its head as bytes. -/
structure Commitment where
  committed : Committed
  head : ByteArray

def commitmentOf (b : ByteArray) : V Commitment := do
  ascii b
  let committed ← Committed.ofJson (← canonicalJson b)
  pure { committed, head := ← Flock.unhex (head committed.composite committed.log) }

def evidenceOf (b : ByteArray) : V Unit := do
  exactly (← canonicalJson b) [] "the evidence"

def checkLog (p : Params) (c : Commitment) (_ : Unit) : Outcome Json Json :=
  if c.committed.composite != p.composite then
    .reject s!"composite: the log is {c.committed.composite.name}'s, not {p.composite.name}'s"
  else match check p.auditor p.composite c.committed.log with
    | .error (i, r) => .reject s!"entry {i}: {r.name}"
    | .ok s => .accept s.view.toJson (Json.mkObj [("entries", Lean.toJson s.entries)])

/-- **The registration log's check** as a core protocol, `verity/registration/log/v1`: the parameters are the auditor's
(`{"auditor": AUDITOR, "composite"}`, the auditor as `Core.Site.Registration.Auditor.ofJson` reads it), the commitment
is the log (`{"composite", "entries": [ENTRY]}`) with its head as digest, the evidence is `{}`. It accepts a log the
check passes, making public its view (`View.toJson`) and keeping its length, and rejects the first entry the check
refuses as `entry I: REFUSAL`, and a log or parameters outside ASCII without DEL as not ASCII. It is never undecided. -/
def protocol : Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Unit
  Public := Json
  Profile := Json
  params := Params.ofJson
  commitment := commitmentOf
  evidence := evidenceOf
  digest c := c.head
  check := checkLog
  publicJson := id
  profileJson := id

/-! ## An example: one node's onsite site, then its information -/

namespace Example

open Core.Site (Sheet Loc Link Figure outside)

/-- The figure `n / d`. -/
def fig (n : Nat) (d : Nat := 1) : Figure := if h : 0 ≤ mkRat n d then ⟨_, h⟩ else ⟨0, by decide⟩

def sheet : Sheet :=
  { figures := { sku := "sim-a", clock := fig 2000000000,
                 peakRate := fun
                   | .uniformS8 => fig 1000000000000000 | .uniformS2 => fig 2000000000000000
                   | .uniformU2 => fig 2000000000000000 | .uniformU4 => fig 1500000000000000
                   | .zero => fig 4000000000000000 },
    bw := 3200, chip := 1 <<< 25, mem := 64 <<< 30 }

/-- `sheet` with its clock at `n` Hz. -/
def clocked (n : Nat) : Sheet := { sheet with figures := { sheet.figures with clock := fig n } }

/-- `sheet` with its peak rate of class `c` at `n`. -/
def rated (c : Site.OperandClass) (n : Nat) : Sheet :=
  { sheet with figures := { sheet.figures with peakRate := fun x => if x = c then fig n else sheet.figures.peakRate x } }

def keys : List (String × String) := [("cert0", "k0"), ("chal0", "k1")]

def auditor : Auditor :=
  { sheet := fun sku => [sheet].find? (·.sku == sku), tolerance := fig 1 20, key := fun i => (keys.find? (·.1 == i)).map (·.2) }

def hex (c : Char) : String := String.ofList (List.replicate 128 c)

def locs : List Loc :=
  [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" },
   { id := "gpu0", kind := .device, parent := some "node0", sheet := Site.Registration.sha512.sheet sheet },
   { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := 1 <<< 30 },
   { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "k0" },
   { id := "chal0", kind := .challenger, parent := some "site0", owner := .auditor, key := "k1" }]

def links : List Link :=
  [{ id := "up0", ends := ("node0", "cert0"), β := 100, latency := 1 },
   { id := "up1", ends := ("cert0", outside), β := 100, latency := 1 },
   { id := "mc0", ends := ("chal0", "node0"), β := 10, latency := 1 }]

def weights : Theta :=
  { value := "weights", root := hex 'a', leaves := 4, words := 64, domain := hex 'b', checkpoint := some (hex 'c') }

def items : List Item :=
  [.theta weights, .theta { value := "state", root := hex 'd', leaves := 1, words := 8, wordBits := 32, domain := hex 'e' },
   .builder { digest := hex 'f', advice := "verity/advice/v1", cap := 32 }]

def entries : List (String → Entry) := [site ⟨[sheet], locs, links⟩, information items]

/-- A site entry adding only `k`. -/
def stating (k : Sheet) : List (String → Entry) := [site ⟨[k], [], []⟩]

def refusal (c : Composite) (entries : List (String → Entry)) (A : Auditor := auditor) : Option (Nat × Refusal) :=
  match check A c (chain c entries) with
  | .ok _ => none
  | .error r => some r

/-- A tolerance of `n / d` around the auditor's sheets. -/
def at_ (n : Nat) (d : Nat := 1) : Auditor := { auditor with tolerance := fig n d }

#guard refusal .onsite entries == none
#guard ((check auditor .onsite (chain .onsite entries)).toOption.map (·.view.value? "weights")) == some (some weights)
#guard (chain .onsite entries).all fun e => match Entry.ofJson e.toJson with | .ok e' => e' == e | .error _ => false
#guard head .onsite (chain .onsite entries) != head .remote (chain .remote entries)
-- a log of site entries alone is the Site's log, with the Site's head
#guard head .onsite (chain .onsite [site ⟨[sheet], locs, links⟩]) ==
  Site.Registration.head Site.Registration.sha512 .onsite [.site (genesis .onsite) ⟨[sheet], locs, links⟩]
-- a log chained for remote is not onsite's
#guard (match check auditor .onsite (chain .remote entries) with | .error r => some r | .ok _ => none) ==
  some (0, .site .previous)
-- onsite: an uplink no certifier splits; remote takes it
private def bare : List (String → Entry) :=
  [site ⟨[sheet], locs.take 4, [{ id := "up0", ends := ("node0", outside), β := 100, latency := 1 }]⟩]
#guard refusal .onsite bare == some (0, .site .uncertified)
#guard refusal .remote bare == none
-- onsite: memory no challenger reaches
#guard refusal .onsite [site ⟨[sheet], locs.take 5, links.take 2⟩] == some (0, .site .challengerReach)
-- a sheet with no data sheet, a device's memory over its sheet's, a certifier without its key
#guard refusal .remote (stating { sheet with figures := { sheet.figures with sku := "sim-b" } }) ==
  some (0, .site .sheetDigest)
#guard refusal .remote bare { auditor with sheet := fun _ => none } == some (0, .site .sheetDigest)
-- a clock or a peak rate stated five percent off is within a twentieth either way; past it, slow or fast, is refused
#guard refusal .remote (stating (clocked 1900000000)) == none
private def fast : Sheet :=
  { sheet with figures := { (clocked 2100000000).figures with peakRate := (rated .zero 3800000000000000).figures.peakRate } }
#guard refusal .remote (stating fast) == none
#guard refusal .remote (stating (clocked 1899999999)) == some (0, .site .figures)
#guard refusal .remote (stating (rated .uniformS2 2100000000000001)) == some (0, .site .figures)
#guard refusal .remote (stating { sheet with mem := 1 }) == some (0, .site .figures)
-- at tolerance 0 only the sheet's figures; at 1 or more nothing below is refused
#guard refusal .remote (stating (clocked 1999999999)) (at_ 0) == some (0, .site .figures)
#guard refusal .remote (stating (clocked 0)) (at_ 1) == none
#guard refusal .remote (stating (clocked 4000000001)) (at_ 1) == some (0, .site .figures)
-- a zero peak rate is refused even where the tolerance allows it
#guard refusal .remote (stating (rated .uniformU4 0)) (at_ 1) == some (0, .site .figures)
private def over : Loc := { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := (64 <<< 30) + 1 }
#guard refusal .remote [site ⟨[sheet], locs.take 3 ++ [over], []⟩] == some (0, .site .capacity)
#guard refusal .onsite entries { auditor with key := fun _ => some "k1" } == some (0, .site .owner)
-- an id twice, a parent nobody registered, a host under a host
#guard refusal .remote [site ⟨[sheet], locs.take 2, []⟩, site ⟨[], locs.drop 1 |>.take 1, []⟩] == some (1, .site .duplicateId)
#guard refusal .remote [site ⟨[], [{ id := "node1", kind := .host, parent := some "site9" }], []⟩] ==
  some (0, .site .unknownRef)
#guard refusal .remote [site ⟨[], locs.take 2 ++ [{ id := "node1", kind := .host, parent := some "node0" }], []⟩] ==
  some (0, .site .kind)
-- θ's weights registered twice, and a second builder; another θ value is a new item
#guard refusal .onsite (entries ++ [information (items.take 1)]) == some (2, .duplicateInformation)
#guard refusal .onsite (entries ++ [information (items.drop 2)]) == some (2, .duplicateInformation)
#guard refusal .onsite (entries ++ [information [.theta { weights with value := "tables" }]]) == none

private def bytes (j : Json) : ByteArray := ((Flock.canon j).toOption.getD "").toUTF8

private def params (c : Composite) : Json :=
  Json.mkObj [("auditor", Json.mkObj [("keys", .arr #[Json.mkObj [("id", "cert0"), ("key", "k0")],
      Json.mkObj [("id", "chal0"), ("key", "k1")]]), ("sheets", .arr #[sheet.toJson]), ("tolerance", "1/20")]),
    ("composite", .str c.name)]

private def answer (c : Composite) (log : Committed) : Answer :=
  protocol.answer (bytes (params c)) (bytes log.toJson) "{}".toUTF8

#guard (answer .onsite { composite := .onsite, log := chain .onsite entries }).verdict == .accept
#guard (answer .onsite { composite := .onsite, log := chain .onsite entries }).digest ==
  (Flock.unhex (head .onsite (chain .onsite entries))).toOption
#guard (answer .onsite { composite := .remote, log := chain .remote entries }).reason ==
  some "composite: the log is remote's, not onsite's"
#guard (answer .onsite { composite := .onsite, log := chain .onsite bare }).reason == some "entry 0: uncertified"
private def accented : Committed :=
  { composite := .onsite, log := chain .onsite [information [.theta { weights with value := "poids\u00e9" }]] }
-- a log or parameters outside ASCII without DEL is refused before anything is decoded (`ascii`)
#guard (protocol.answer (bytes (params .onsite)) accented.toJson.compress.toUTF8 "{}".toUTF8).reason ==
  some "commitment: not ASCII without DEL (a byte at or above 0x7f)"
private def deleted : Committed :=
  { composite := .onsite, log := chain .onsite [information [.theta { weights with value := "poids\x7f" }]] }
#guard (protocol.answer (bytes (params .onsite)) deleted.toJson.compress.toUTF8 "{}".toUTF8).reason ==
  some "commitment: not ASCII without DEL (a byte at or above 0x7f)"

end Example

end Core.Registration
