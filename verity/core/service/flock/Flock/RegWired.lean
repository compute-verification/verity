/-
Wired reads: a statement's rows of a port the window's wiring fills from another unit's output, each opened against the
writer's committed root. A wired row is committed under its writer row's salt, so its `b ‖ c` is the writer's leaf: the
verifier's own `--wired` file (`FORMAT`) names, per wired port of the statement, the writers' output roots as the
window's registration holds them (each a frame-v3-sha512 root, domain and leaf count, accepted before any coin), and the
public header states, per table row of the port, which writer's root it opens, at what position and through what path
(`HEADER_KEY`). `check` opens every table row; which writer and position each instance must read is the layout's, which
the audit derives from the wires and holds the stated ones to (`SampledProofs.WiredReads`). Nothing is opened in the
clear: the header holds only `b ‖ c`'s tree siblings, positions and the writers' registration names.
-/
import Flock.Registered

namespace Flock
namespace RegWired

open Lean (Json)

def FORMAT : String := "verity/wired-reads/v1"
/-- The keys a `FORMAT` file holds, its ports' and its writers'. -/
def FILE_KEYS : List String := ["format", "ports"]
def PORT_KEYS : List String := ["writers"]
def WRITER_KEYS : List String := ["root", "domain", "leaves"]
/-- The public header's key: `{port: {"writers", "positions", "paths"}}`, one of each per table row of the port. -/
def HEADER_KEY : String := "wired"
def ENTRY_KEYS : List String := ["writers", "positions", "paths"]

/-- A writer's output as the window's registration holds it: its registration port (`<member>/<out>`), the
frame-v3-sha512 root and domain of its rows and their count. -/
structure Writer where
  name : String
  root : ByteArray
  domain : ByteArray
  leaves : Nat
  deriving Inhabited

/-- A wired port of the statement and the writers' outputs its rows may open. -/
structure Port where
  name : String
  writers : Array Writer
  deriving Inhabited

def ofJson (j : Json) : V (Array Port) := do
  let ctx := "wired reads"
  if (← J.str j "format" ctx) != FORMAT then throw s!"{ctx}: not a {FORMAT} file"
  if !Registered.keysIn j FILE_KEYS then throw s!"{ctx}: a {FORMAT} file holds the keys {FILE_KEYS} and no other"
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw s!"{ctx}: ports is not an object"
  if kvs.toArray.isEmpty then throw s!"{ctx}: no port"
  kvs.toArray.mapM fun (name, e) => do
    if !Registered.keysIn e PORT_KEYS then throw s!"{ctx}: port {name}: it holds the keys {PORT_KEYS} and no other"
    let .ok (.obj ws) := e.getObjVal? "writers" | throw s!"{ctx}: port {name}: writers is not an object"
    if ws.toArray.isEmpty then throw s!"{ctx}: port {name} names no writer"
    let writers ← ws.toArray.mapM fun (w, x) => do
      if !Registered.keysIn x WRITER_KEYS then
        throw s!"{ctx}: port {name}: writer {w}: it holds the keys {WRITER_KEYS} and no other"
      let root ← unhex (← J.str x "root" ctx)
      let domain ← unhex (← J.str x "domain" ctx)
      let leaves ← J.nat x "leaves" ctx
      if root.size != 64 || domain.size != 64 || leaves == 0 then
        throw s!"{ctx}: port {name}: writer {w}: a 64-byte root and domain, and at least one leaf"
      pure ({ name := w, root, domain, leaves } : Writer)
    pure { name, writers }

/-- The writer's output as `Registered.opens` reads a root: its leaves framed under `schema`, the reading port's. -/
def Writer.asPort (w : Writer) (schema : String) : Registered.Port :=
  { name := w.name, value := w.name, root := w.root, leaves := w.leaves, words := none, wordBits := 16,
    domain := w.domain, readsRel := #[], schema }

/-- Port `q`'s table rows: a shared-row file's table, else one row an instance. -/
def rows (pub : Public) (q : Nat) : Nat :=
  match pub.tables with
  | some t => t.counts.getD q 0
  | none => pub.n

/-- Table row `r` of port `q`, among `np` ports: a shared-row file's table row, else instance `r`'s own row. -/
def row (pub : Public) (np q r : Nat) : ByteArray :=
  match pub.tables with
  | some t => Registered.tableRow pub t q r
  | none => pub.digests.getD (r * np + q) ByteArray.empty

/-- The table row instance `i` reads of port `q`: its ref, else its own. -/
def refOf (pub : Public) (np q i : Nat) : Nat :=
  match pub.tables with
  | some t => Registered.ref t np q i
  | none => i

/-- Instance `i`'s row of port `q` is the table row it reads. -/
theorem digest_eq (pub : Public) (np q i : Nat) : pub.digest i q np = row pub np q (refOf pub np q i) := by
  cases h : pub.tables <;> simp [Public.digest, row, refOf, h, Registered.tableRow, Registered.ref]

/-- Port `port`'s header entry: each table row's writer, position and path. -/
def entry (pub : Public) (port : Port) : V (Array String × Array Nat × Array ByteArray) := do
  let ctx := "public file.wired"
  let .ok e := pub.header.getObjVal? HEADER_KEY >>= (·.getObjVal? port.name)
    | throw s!"wired reads: the statement states no writer for port {port.name}'s rows: a wired row opens its writer's root"
  if !Registered.keysIn e ENTRY_KEYS then throw s!"wired reads: port {port.name}'s entry holds the keys {ENTRY_KEYS} and no other"
  let writers ← (← J.arr e "writers" ctx).mapM (J.asStr · "a wired row's writer")
  let positions ← (← J.arr e "positions" ctx).mapM (J.asNat · "a wired row's position")
  let paths ← (← J.arr e "paths" ctx).mapM fun x => do unhex (← J.asStr x "a wired row's path")
  return (writers, positions, paths)

/-- A row's `b ‖ c` opens the root of the port's writer `writer` at `pos` through `path`, its leaf framed under `schema`. -/
def rowOpens (port : Port) (schema : String) (bc : ByteArray) (writer : String) (pos : Nat) (path : ByteArray) : Bool :=
  port.writers.any fun w => w.name == writer && Registered.opens (w.asPort schema) bc pos path

/-- Port `port` of the circuit is a bit row (`row/v2` or `row-seg/v1`, as a writer's output row is), its header entry
states one writer, position and path per table row, and every table row opens its stated writer's root (`rowOpens`). -/
def portOk (c : Circuit) (pub : Public) (port : Port) : Bool :=
  match Registered.portOf c port.name, entry pub port with
  | some q, .ok (ws, ps, pa) =>
    let n := rows pub q
    let schema := Registered.portSchema c.ports[q]!
    schema != HmRow.ROW_SCHEMA && ws.size == n && ps.size == n && pa.size == n &&
      (List.range n).all fun r => rowOpens port schema (row pub c.ports.size q r) (ws.getD r "") (ps.getD r 0)
        (pa.getD r ByteArray.empty)
  | _, _ => false

/-- Why `portOk` refuses `port`. -/
def portWhy (c : Circuit) (pub : Public) (port : Port) : String :=
  match Registered.portOf c port.name, entry pub port with
  | none, _ => s!"wired reads: {port.name} is no port of the circuit"
  | some _, .error e => e
  | some q, .ok (ws, ps, pa) =>
    let n := rows pub q
    let schema := Registered.portSchema c.ports[q]!
    if schema == HmRow.ROW_SCHEMA then
      s!"wired reads: port {port.name} is a {schema} row, and a wired row is a bit row, as its writer's output row is"
    else if ws.size != n || ps.size != n || pa.size != n then
      s!"wired reads: port {port.name}: one writer, position and path per table row ({n})"
    else
      let r := ((List.range n).find? fun r => !rowOpens port schema (row pub c.ports.size q r) (ws.getD r "")
        (ps.getD r 0) (pa.getD r ByteArray.empty)).getD 0
      if !port.writers.any (·.name == ws.getD r "") then
        s!"wired reads: port {port.name}: table row {r} names writer {ws.getD r ""}, whose root the verifier's wired reads do not hold"
      else
        s!"wired reads: port {port.name}: table row {r} does not open {ws.getD r ""}'s root at position {ps.getD r 0}"

/-- A port the header names wired that the verifier's own reads do not. -/
def unheld (pub : Public) (ports : Array Port) : Option String :=
  match pub.header.getObjVal? HEADER_KEY with
  | .ok (.obj kvs) => (kvs.toArray.find? fun (n, _) => !ports.any (·.name == n)).map (·.1)
  | _ => none

/-- **Each wired row opens its writer's committed root** (`check_at`): without the verifier's wired reads, nothing; with
them, the header names no wired port they don't, and every port of them is a bit row of the circuit whose every table row's
`b ‖ c`, at its stated position and through its stated path, opens the root of its stated writer among the port's
(`portOk`). -/
def check (c : Circuit) (pub : Public) (wired : Option (Array Port)) : V Unit :=
  match wired with
  | none => .ok ()
  | some ports =>
    match unheld pub ports with
    | some n => .error s!"wired reads: the statement names port {n} wired, and the verifier's wired reads do not"
    | none =>
      match ports.find? (!portOk c pub ·) with
      | none => .ok ()
      | some port => .error (portWhy c pub port)

theorem check_none (c : Circuit) (pub : Public) : check c pub none = .ok () := rfl

theorem portOk_at {c : Circuit} {pub : Public} {port : Port} (h : portOk c pub port = true) :
    ∃ q ws ps pa, Registered.portOf c port.name = some q ∧ entry pub port = .ok (ws, ps, pa) ∧
      Registered.portSchema c.ports[q]! ≠ HmRow.ROW_SCHEMA ∧
      ws.size = rows pub q ∧ ps.size = rows pub q ∧ pa.size = rows pub q ∧
      ∀ r < rows pub q, ∃ w ∈ port.writers, w.name = ws.getD r "" ∧
        Registered.opens (w.asPort (Registered.portSchema c.ports[q]!)) (row pub c.ports.size q r) (ps.getD r 0)
          (pa.getD r ByteArray.empty) = true := by
  unfold portOk at h
  split at h
  · rename_i q ws ps pa hq he
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, List.all_eq_true, List.mem_range] at h
    obtain ⟨⟨⟨⟨hs, hw⟩, hp⟩, ha⟩, hr⟩ := h
    refine ⟨q, ws, ps, pa, hq, he, hs, hw, hp, ha, fun r hrn => ?_⟩
    have := hr r hrn
    simp only [rowOpens, Array.any_eq_true', Bool.and_eq_true, beq_iff_eq] at this
    obtain ⟨w, hm, hn, ho⟩ := this
    exact ⟨w, hm, hn, ho⟩
  · cases h

/-- **An accepted check opens every wired row** (ODSESSION's W1): with the verifier's wired reads `ports`, every port of
them passes `portOk` (`portOk_at`: each table row's `b ‖ c` opens its stated writer's root at its stated position). -/
theorem check_at {c : Circuit} {pub : Public} {ports : Array Port} (h : check c pub (some ports) = .ok ()) :
    ∀ port ∈ ports, portOk c pub port = true := by
  intro port hp
  simp only [check] at h
  split at h
  · cases h
  · split at h
    · rename_i hn
      simpa using Array.find?_eq_none.mp hn port hp
    · cases h

end RegWired
end Flock
