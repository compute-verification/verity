/-
Wired reads: a statement's rows of a port the window's wiring fills from another unit's output, each opened against the
writer's committed root. A wired row is committed under its writer row's salt, so its `b ‖ c` is the writer's leaf. The
verifier's own wired reads (`Port`) give, per wired port of the statement and per instance of the session (a drawn
session's drawn instances), the writer's output as the window's registration holds it (a frame-v3-sha512 root, domain and
leaf count, accepted before any coin) and the writer unit's position among its leaves; the audit derives both from the
wires (`SampledProofs.WiredReads`). The public header states only one path per row of the population (`HEADER_KEY`).
`check` opens the row each session instance reads, at the reading port's leaf schema, so its work is the session's
instances times the wired ports and the paths' depth. Nothing is opened in the clear: the header holds only `b ‖ c`'s tree
siblings.
-/
import Flock.Registered

namespace Flock
namespace RegWired

/-- The public header's key: `{port: [path, …]}`, one frame-v3-sha512 path (leaf level first) per row of the population's
port: its table's rows in a shared-row file, else one per instance of the population. -/
def HEADER_KEY : String := "wired"

/-- A writer's output as the window's registration holds it: its registration port (`<member>/<out>`), the
frame-v3-sha512 root and domain of its rows and their count. -/
structure Writer where
  name : String
  root : ByteArray
  domain : ByteArray
  leaves : Nat
  deriving Inhabited

/-- What one session instance's wired row reads: its writer's output and the writer unit's position among its leaves. -/
structure Read where
  writer : Writer
  pos : Nat
  deriving Inhabited

/-- A wired port of the statement and each session instance's read, in instance order. -/
structure Port where
  name : String
  reads : Array Read
  deriving Inhabited

/-- The writer's output as `Registered.opens` reads a root: its leaves framed under `schema`, the reading port's. -/
def Writer.asPort (w : Writer) (schema : String) : Registered.Port :=
  { name := w.name, value := w.name, root := w.root, leaves := w.leaves, words := none, wordBits := 16,
    domain := w.domain, readsRel := #[], schema }

/-- The population's rows of port `q`: its table's in a shared-row file, else one per instance of the population. -/
def popRows (pub : Public) (q : Nat) : Nat :=
  match pub.tables with
  | some t => t.counts.getD q 0
  | none => (pub.population.map (·.1)).getD pub.n

/-- The population's row of port `q` that session instance `i` reads, which its path is indexed by: its ref in a
shared-row file, else its population instance (`drawn[i]`, the draw's proved units, in a drawn session). -/
def popRow (pub : Public) (drawn : Option (Array Nat)) (np q i : Nat) : Nat :=
  match pub.tables with
  | some t => Registered.ref t np q i
  | none => (drawn.bind (·[i]?)).getD i

/-- The session's drawn units (`unit_draw`'s proved units), none for an undrawn one. -/
def drawnOf (pub : Public) : V (Option (Array Nat)) :=
  match pub.header.getObjVal? "unit_draw" with
  | .ok d => (some ·.proved) <$> Draw.UnitDraw.ofJson d
  | .error _ => pure none

/-- Port `name`'s header entry: one path per row of the population's port. -/
def entry (pub : Public) (name : String) : V (Array ByteArray) := do
  let .ok e := pub.header.getObjVal? HEADER_KEY >>= (·.getObjVal? name)
    | throw s!"wired reads: the statement states no path for port {name}'s rows: a wired row opens its writer's root"
  (← J.asArr e s!"wired reads: port {name}'s paths").mapM fun x => do unhex (← J.asStr x "a wired row's path")

/-- Session instance `i`'s row of port `q` (`Public.digest`, the row its session binds its input to) opens its read's
writer root at the read's position, its leaf framed under port `q`'s schema, through the path of the row it reads. -/
def readOpens (c : Circuit) (pub : Public) (drawn : Option (Array Nat)) (port : Port) (q : Nat) (paths : Array ByteArray)
    (i : Nat) : Bool :=
  match port.reads[i]? with
  | some rd => Registered.opens (rd.writer.asPort (Registered.portSchema c.ports[q]!)) (pub.digest i q c.ports.size) rd.pos
      (paths.getD (popRow pub drawn c.ports.size q i) ByteArray.empty)
  | none => false

/-- Port `port` of the circuit is a bit row (`row/v2` or `row-seg/v1`, as a writer's output row is), the verifier reads
one row per session instance, the header states one path per row of the population, and every session instance's row
opens its read's writer root (`readOpens`). -/
def portOk (c : Circuit) (pub : Public) (drawn : Option (Array Nat)) (port : Port) : Bool :=
  match Registered.portOf c port.name, entry pub port.name with
  | some q, .ok paths =>
    Registered.portSchema c.ports[q]! != HmRow.ROW_SCHEMA && port.reads.size == pub.n && paths.size == popRows pub q &&
      (List.range pub.n).all (readOpens c pub drawn port q paths)
  | _, _ => false

/-- Why `portOk` refuses `port`. -/
def portWhy (c : Circuit) (pub : Public) (drawn : Option (Array Nat)) (port : Port) : String :=
  match Registered.portOf c port.name, entry pub port.name with
  | none, _ => s!"wired reads: {port.name} is no port of the circuit"
  | some _, .error e => e
  | some q, .ok paths =>
    let schema := Registered.portSchema c.ports[q]!
    if schema == HmRow.ROW_SCHEMA then
      s!"wired reads: port {port.name} is a {schema} row, and a wired row is a bit row, as its writer's output row is"
    else if port.reads.size != pub.n then
      s!"wired reads: port {port.name}: the verifier reads {port.reads.size} rows, one per session instance ({pub.n})"
    else if paths.size != popRows pub q then
      s!"wired reads: port {port.name}: one path per row of the population ({popRows pub q})"
    else
      let i := ((List.range pub.n).find? (!readOpens c pub drawn port q paths ·)).getD 0
      let rd := port.reads[i]!
      s!"wired reads: port {port.name}: instance {i}'s row does not open {rd.writer.name}'s root at position {rd.pos}"

/-- A port the header names wired that the verifier's own reads do not. -/
def unheld (pub : Public) (ports : Array Port) : Option String :=
  match pub.header.getObjVal? HEADER_KEY with
  | .ok (.obj kvs) => (kvs.toArray.find? fun (n, _) => !ports.any (·.name == n)).map (·.1)
  | _ => none

/-- A wired port the verifier's registered reads also name: its rows open a registered root, never a writer's. -/
def registeredToo (registered : Option (Array Registered.Port)) (ports : Array Port) : Option String :=
  ((ports.find? fun p => registered.any (·.any (·.name == p.name))).map (·.name))

/-- **Each session instance's wired row opens its writer's committed root** (`check_at`): without the verifier's wired
reads, nothing; with them, the header names no wired port they don't, no wired port is a registered one, and every port
of them passes `portOk`. -/
def check (c : Circuit) (pub : Public) (registered : Option (Array Registered.Port)) (wired : Option (Array Port)) :
    V Unit :=
  match wired with
  | none => .ok ()
  | some ports =>
    match unheld pub ports, registeredToo registered ports, drawnOf pub with
    | some n, _, _ => .error s!"wired reads: the statement names port {n} wired, and the verifier's wired reads do not"
    | none, some n, _ => .error s!"wired reads: port {n} is a registered port and wired: each port is one or the other"
    | none, none, .error e => .error e
    | none, none, .ok drawn =>
      match ports.find? (!portOk c pub drawn ·) with
      | none => .ok ()
      | some port => .error (portWhy c pub drawn port)

theorem check_none (c : Circuit) (pub : Public) (registered : Option (Array Registered.Port)) :
    check c pub registered none = .ok () := rfl

theorem portOk_at {c : Circuit} {pub : Public} {drawn : Option (Array Nat)} {port : Port}
    (h : portOk c pub drawn port = true) :
    ∃ q, Registered.portOf c port.name = some q ∧ Registered.portSchema c.ports[q]! ≠ HmRow.ROW_SCHEMA ∧
      port.reads.size = pub.n ∧ ∀ (i : Nat) (hi : i < port.reads.size), ∃ path,
        Registered.opens (port.reads[i].writer.asPort (Registered.portSchema c.ports[q]!)) (pub.digest i q c.ports.size)
          port.reads[i].pos path = true := by
  unfold portOk at h
  split at h
  · rename_i q paths hq _
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, List.all_eq_true, List.mem_range] at h
    obtain ⟨⟨⟨hs, hn⟩, _⟩, hr⟩ := h
    refine ⟨q, hq, hs, hn, fun i hi => ?_⟩
    have := hr i (hn ▸ hi)
    simp only [readOpens, Array.getElem?_eq_getElem hi] at this
    exact ⟨_, this⟩
  · cases h

/-- **An accepted check opens every session instance's wired row**: with the verifier's wired reads `ports`, every port
of them is a bit row of the circuit, read once per session instance, and instance `i`'s row (`Public.digest`, the row its
session binds its input to), as a leaf of the reading port's schema, opens its read's writer root at the read's position. -/
theorem check_at {c : Circuit} {pub : Public} {registered : Option (Array Registered.Port)} {ports : Array Port}
    (h : check c pub registered (some ports) = .ok ()) :
    ∀ port ∈ ports, ∃ q, Registered.portOf c port.name = some q ∧
      Registered.portSchema c.ports[q]! ≠ HmRow.ROW_SCHEMA ∧ port.reads.size = pub.n ∧
      ∀ (i : Nat) (hi : i < port.reads.size), ∃ path,
        Registered.opens (port.reads[i].writer.asPort (Registered.portSchema c.ports[q]!)) (pub.digest i q c.ports.size)
          port.reads[i].pos path = true := by
  intro port hp
  simp only [check] at h
  split at h
  · cases h
  · cases h
  · cases h
  · split at h
    · rename_i hn
      exact portOk_at (by simpa using Array.find?_eq_none.mp hn port hp)
    · cases h

end RegWired
end Flock
