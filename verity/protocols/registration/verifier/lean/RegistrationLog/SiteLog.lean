module

public import CoreProtocol.Site

/-!
# The site log

The registration log the Site is folded from (`Site.Registration`, `Site.ofLog`), its entries, the digests it names
things by (`Registration.Digests`, the deployed `Registration.sha512`), the refusals its check names, and the JSON
`verity.protocols.registration.verifier.site_log` writes for it, under `verity/registration/site/v2` and
`verity/registration/genesis/v1`. The Site's
types, its views, the auditor's figures, the composites and the sheet's tag are the interface's (`CoreProtocol.Site`);
the checks an entry passes are `RegistrationLog.SiteCheck`'s.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core

/-- An entry of the registration, by its kind, naming the head of the log it follows: the site entry. -/
inductive Site.Registration.Entry
  | site (previous : String) (entry : Site.Entry)
  deriving DecidableEq

/-- **The registration**: the log the Site is folded from, oldest entry first. -/
abbrev Site.Registration : Type := List Site.Registration.Entry

namespace Site

namespace Registration.Entry

/-- The head of the log the entry follows. -/
def previous : Entry → String
  | .site p _ => p

/-- What the entry adds to the Site. -/
def added : Entry → Site.Entry
  | .site _ s => s

end Registration.Entry

/-- The Site an entry's additions extend `S` to, unchecked. -/
def add (S : Site) (e : Entry) : Site :=
  ⟨S.sheets ++ e.sheets, S.locs ++ e.locs, S.links ++ e.links⟩

/-- **The Site of a log**: its entries' sheets, locations and links, in the log's order. -/
def ofLog (log : Registration) : Site :=
  ⟨log.flatMap (·.added.sheets), log.flatMap (·.added.locs), log.flatMap (·.added.links)⟩

end Site

namespace Site.Registration

/-- The digests the log names things by: a sheet's, an entry's, and the head of a composite's empty log. -/
structure Digests where
  sheet : Sheet → String
  entry : Entry → String
  genesis : Composite → String

/-- A refusal, named by the check that fails. -/
inductive Refusal
  | previous
  | duplicateId
  | unknownRef
  | kind
  | sheetDigest
  | figures
  | capacity
  | owner
  | inventory
  | uncertified
  | challengerReach
  deriving DecidableEq, Repr

end Site.Registration

/-! ## The JSON `site_log` writes, and the deployed digests -/

namespace Site

open Lean (Json)
open Flock (V)

def SITE : String := "verity/registration/site/v2"
def GENESIS : String := "verity/registration/genesis/v1"

/-- A refusal's name in `site_log` (`REFUSALS`): the constructor's, in kebab case. -/
def Registration.Refusal.name : Registration.Refusal → String
  | .previous => "previous" | .duplicateId => "duplicate-id" | .unknownRef => "unknown-ref" | .kind => "kind"
  | .sheetDigest => "sheet-digest" | .figures => "figures" | .capacity => "capacity" | .owner => "owner"
  | .inventory => "inventory" | .uncertified => "uncertified" | .challengerReach => "challenger-reach"

def Registration.Entry.toJson : Registration.Entry → Json
  | .site p s => Json.mkObj [("format", .str SITE), ("links", .arr (s.links.map Link.toJson).toArray),
      ("locs", .arr (s.locs.map Loc.toJson).toArray), ("previous", .str p),
      ("sheets", .arr (s.sheets.map Sheet.toJson).toArray)]

/-- The deployed digests: SHA-512 over the JSON `site_log` writes, under the registration log's tags. -/
def Registration.sha512 : Registration.Digests where
  sheet k := tagged SHEET k.toJson
  entry e := tagged SITE e.toJson
  genesis c := tagged GENESIS (Json.mkObj [("composite", .str c.name)])

def Registration.Entry.ofJson (j : Json) : V Registration.Entry := do
  exactly j ["format", "links", "locs", "previous", "sheets"] "an entry"
  if (← Flock.J.str j "format" "an entry") != SITE then throw s!"an entry: format is not {SITE}"
  let sheets ← (← Flock.J.arr j "sheets" "an entry").toList.mapM Sheet.ofJson
  let locs ← (← Flock.J.arr j "locs" "an entry").toList.mapM Loc.ofJson
  let links ← (← Flock.J.arr j "links" "an entry").toList.mapM Link.ofJson
  pure (.site (← Flock.J.str j "previous" "an entry") ⟨sheets, locs, links⟩)

/-- A log as `Registration.to_json` writes it: its composite and its entries. -/
def Registration.ofJson (j : Json) : V (Registration.Composite × Registration) := do
  exactly j ["composite", "entries"] "the log"
  pure (← Registration.Composite.ofJson (← Flock.J.get j "composite" "the log"),
    ← (← Flock.J.arr j "entries" "the log").toList.mapM Registration.Entry.ofJson)

end Site

end Core
