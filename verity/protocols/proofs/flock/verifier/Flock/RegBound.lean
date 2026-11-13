/-
`Flock.Registered`'s completeness with bound ports: a `--registered` file of `Registered.FORMAT` may name, beside its
registered ports, every other port of the statement bound to the window's committed values (`KEY`, `bound`): its rows'
frame-v3-sha512 root and domain as the window's registration holds them, which the verifier accepted before any coin.
`HmOut.checkPublic` recomputes each port's root from the population's `b ‖ c` against the public header's, so a bound
port's rows are the rows the window committed. A port the verifier's own program reads a registered input through
(`Registered.regParams`, the template's public structure, never the file) is never bound: `Registered.checkCover`
holds it to a registered port, opened at its registered root. Nor is a port the window's registered interface declares
registered (`declaredOf`: its circuit record's registered reads, table reads and openings' verdicts, outputs included): a
file that binds ports needs that record (`flock-verify --interface`), and each such port of the statement's template must
be a registered port of the file, never bound and never left out, whatever the file's builder omitted. Without `bound`,
`check` is `Registered.checkComplete`.
-/
module

public import Flock.Registered
public import Flock.JsonKeys

/-!
# A `--registered` file's bound ports, and the check that every port is registered or bound

`KEY`, `Bound` and `ofJson` read the ports a `Registered.FORMAT` file binds to the window's committed values, and
`ownOf` the file without them; `declaredOf` and `declaredOfBytes` read the ports a window's registered circuit record
(`INTERFACE`) declares registered; `regParamsOf` gives the template parameters the verifier's own program reads a
registered input through. `check` holds every port of a statement to a registered port or a bound one, with
`check_empty` (no bound port: `Registered.checkComplete`), `check_ok` (every port `held`) and `check_declared` (every
declared port registered). `flock-verify` (`FlockVerify`) reads a `--registered` file through `ofJson` and `ownOf`, its
`--interface` record through `declaredOfBytes`, and runs `check` on each statement; `Flock.Test.RegBound` holds `check`
to its refusals.
-/

@[expose] public section

namespace Flock
namespace RegBound

open Lean (Json)

/-- The key of a `Registered.FORMAT` file naming its bound ports: `{port: {"root", "domain"}}`, each 64 bytes in hex. -/
def KEY : String := "bound"
/-- The keys a bound port holds. -/
def KEYS : List String := ["root", "domain"]

/-- A port bound to the window's committed values: the frame-v3-sha512 root and domain of its rows. -/
structure Bound where
  name : String
  root : ByteArray
  domain : ByteArray
  deriving Inhabited

/-- The file without `KEY`: the registered reads `Registered.ofJson` and its siblings read. -/
def ownOf (j : Json) : Json :=
  match j with
  | .obj kvs => Json.mkObj (kvs.toArray.filter (·.1 != KEY)).toList
  | x => x

/-- The file's bound ports (`KEY`), none when it names none. Only a `Registered.FORMAT` file names them, each a port
of 64-byte root and domain and no other key. -/
def ofJson (j : Json) : V (Array Bound) := do
  let ctx := "registered reads"
  let .ok b := j.getObjVal? KEY | return #[]
  if (j.getObjValAs? String "format").toOption != some Registered.FORMAT then
    throw s!"{ctx}: {KEY} is a {Registered.FORMAT} key"
  let .obj kvs := b | throw s!"{ctx}: {KEY} is not an object"
  if kvs.toArray.isEmpty then throw s!"{ctx}: {KEY} names no port"
  kvs.toArray.mapM fun (name, e) => do
    if !Registered.keysIn e KEYS then throw s!"{ctx}: bound port {name}: it holds the keys {KEYS} and no other"
    let root ← unhex (← J.str e "root" ctx)
    let domain ← unhex (← J.str e "domain" ctx)
    if root.size != 64 || domain.size != 64 then throw s!"{ctx}: bound port {name}: a 64-byte root and domain"
    return { name, root, domain }

/-- The format of a window's registered circuit record (`verity.commitments.windows.interface.CIRCUIT`), the interface whose
registered reads declare which ports read registered values. -/
def INTERFACE : String := "verity/window-circuit/v1"

/-- The `(reader, field)` of each entry of the interface's registered `key`. -/
def entriesOf (reg : Json) (key field : String) : V (Array (String × String)) := do
  let .ok (.arr xs) := reg.getObjVal? key | throw s!"registered interface: registered {key} is not a list"
  xs.mapM fun x => do
    let .ok r := x.getObjValAs? String "reader" | throw s!"registered interface: an entry of registered {key} names no reader"
    let .ok q := x.getObjValAs? String field | throw s!"registered interface: an entry of registered {key} names no {field}"
    pure (r, q)

/-- The ports a window's registered circuit record `j` declares registered, as `(reader template, port)`: each registered
read's port, each table read's port (its table's root or domain id) and each opening's verdict, an output (the record's
`registered` `reads`, `roots` and `openings`, as `interface.committed_reads` reads them); none when it declares no
registered read. Its `program` must be `program`, the partition object's, so it is the interface of the statement's own
window program. -/
def declaredOf (j : Json) (program : Option String) : V (Array (String × String)) := do
  if (j.getObjValAs? String "format").toOption != some INTERFACE then
    throw s!"registered interface: not a {INTERFACE} record"
  let some p := program
    | throw "registered interface: its program is held to the partition object's, and the verifier holds none (--partition)"
  if (j.getObjValAs? String "program").toOption != some p then
    throw "registered interface: the record's program is not the partition object's: it is another window's interface"
  let .ok reg := j.getObjVal? "registered" | return #[]
  return (← entriesOf reg "reads" "port") ++ (← entriesOf reg "roots" "port") ++ (← entriesOf reg "openings" "verdict")

/-- `declaredOf` of the record's bytes `b`: UTF-8 JSON naming no key twice in one object (`JsonKeys.dupKey`), which a
JSON parser would read as once, as a `--registered` file is read. -/
def declaredOfBytes (b : ByteArray) (program : Option String) : V (Array (String × String)) := do
  let some t := String.fromUTF8? b | throw "registered interface: not UTF-8"
  let j ← match JsonParse.parse t with
    | .ok j => pure j
    | .error e => throw s!"registered interface: not JSON: {e}"
  if let some k := JsonKeys.dupKey t then
    throw s!"registered interface: an object of the record names the key {k} twice, which a JSON parser would read as once"
  declaredOf j program

/-- Port `q` of a circuit is the declared port `p`: `p` itself, or for an array of rows one row `p[i]`. -/
def namesPort (q p : String) : Bool := q == p || (q.startsWith (p ++ "[") && q.endsWith "]")

/-- Port `q` is one of the declared ports `ds`. -/
def declaredIn (ds : Array String) (q : String) : Bool := ds.any (namesPort q)

/-- The ports `declared` (`declaredOf`'s) names for the circuit's template (`HmRow.queryTemplate`), which a file binding
ports needs. -/
def templatePorts (c : Circuit) : Option (Array (String × String)) → V (Array String)
  | none => .error s!"registered reads (declared): the file binds ports ({KEY}), and the verifier holds no registered interface (--interface) to tell the ports the window declares registered, which are never bound"
  | some d => match HmRow.queryTemplate c with
    | none => .error "registered reads (declared): META program_digests does not name the circuit's one template, whose ports the window's interface declares registered"
    | some (t, _) => .ok ((d.filter (·.1 == t)).map (·.2))

/-- Port `name` is named after the template parameter `p`: `p`'s name, or for an array of rows `name[i]` of one row. -/
def namesParam (name : String) (p : String × Program.Ty) : Bool :=
  name == p.1 || match p.2 with
    | .array n _ => (List.range n).any fun i => name == s!"{p.1}[{i}]"
    | _ => false

/-- The template parameters the verifier's own program reads a registered input through (`Registered.regParams`), for the
circuit's template (`HmRow.queryTemplate`); none for a program with no `RegisteredInput<w>_v1` root node. A bound port
needs the program: refused without it, or when it does not decode. -/
def regParamsOf (c : Circuit) (program : Option ByteArray) : V (Array (String × Program.Ty)) := do
  let some b := program
    | throw s!"registered reads (bound): the file binds ports ({KEY}), and the verifier holds no program (--program) to tell its registered ports"
  let P ← Program.decode b
  if !Registered.regRoots P then return #[]
  let some (t, _) := HmRow.queryTemplate c
    | throw "registered reads (bound): META program_digests does not name the circuit's one template, whose registered parameters are never bound"
  return Registered.regParams P t

/-- Bound port `b` is a port of the circuit, not a registered port of `own`, none of the declared ports `ds`, and named
after no registered parameter of `regs`. -/
def boundOk (c : Circuit) (own : Array Registered.Port) (regs : Array (String × Program.Ty)) (ds : Array String)
    (b : Bound) : Bool :=
  c.ports.any (·.name == b.name) && !own.any (·.name == b.name) && !declaredIn ds b.name && !regs.any (namesParam b.name)

/-- Why `boundOk` refuses `b`. -/
def boundWhy (c : Circuit) (own : Array Registered.Port) (regs : Array (String × Program.Ty)) (ds : Array String)
    (b : Bound) : String :=
  if !c.ports.any (·.name == b.name) then s!"registered reads (bound): {b.name} is no port of the statement's circuit"
  else if own.any (·.name == b.name) then
    s!"registered reads (bound): port {b.name} is a registered port and bound: each port is one or the other"
  else if declaredIn ds b.name then
    s!"registered reads (declared): port {b.name} is one the window's interface declares registered: it reads at a registered root, never a window's (name it among the ports)"
  else
    let p := ((regs.find? (namesParam b.name)).map (·.1)).getD ""
    s!"registered reads (bound): port {b.name} reads parameter {p}, a registered input of the verifier's program: its rows open the registered root, never a window's (name it among the ports)"

/-- Port `p` of the circuit is held: a registered port of `own`, or bound by `bound` at the public header's
frame-v3-sha512 root and domain of its rows. -/
def held (pub : Public) (own : Array Registered.Port) (bound : Array Bound) (p : Flock.Port) : Bool :=
  own.any (·.name == p.name) || bound.any fun b => b.name == p.name &&
    (pub.fv3Bytes "roots" p.name).toOption == some b.root && (pub.fv3Bytes "domain_ids" p.name).toOption == some b.domain

/-- Port `p` of the circuit is one of the declared ports `ds` and no registered port of `own` names it. -/
def declaredMiss (own : Array Registered.Port) (ds : Array String) (p : Flock.Port) : Bool :=
  declaredIn ds p.name && !own.any (·.name == p.name)

/-- **Every port a statement's outputs read is registered or bound to the window's committed values**: without bound
ports, `Registered.checkComplete`; with them, the ports the window's interface declares registered for the circuit's
template (`templatePorts` of `declared`, `declaredOf`'s, none refused), each bound port a port of the circuit that no
registered port names, none of those declared ports and reading no registered input of the verifier's program
(`boundOk`); every declared port of the circuit a registered port of `own` (`declaredMiss`, `check_declared`); and every
port of the circuit a registered port of `own` or a bound port whose rows' root and domain in the public header are the
bound ones (`held`, `check_ok`). A port that is neither is refused: its rows would be committed afresh, stated by the
public file. -/
def check (c : Circuit) (pub : Public) (program : Option ByteArray) (own : Array Registered.Port) (bound : Array Bound)
    (declared : Option (Array (String × String))) : V Unit :=
  if bound.isEmpty then Registered.checkComplete c own
  else match templatePorts c declared with
    | .error e => .error e
    | .ok ds =>
      match regParamsOf c program with
      | .error e => .error e
      | .ok regs =>
        match bound.find? (!boundOk c own regs ds ·) with
        | some b => .error (boundWhy c own regs ds b)
        | none =>
          match c.ports.find? (declaredMiss own ds ·) with
          | some p =>
            .error s!"registered reads (declared): the statement's port {p.name} is one the window's interface declares registered, and no port of the verifier's registered reads: it reads at a registered root, never bound or committed afresh"
          | none =>
            match c.ports.find? (!held pub own bound ·) with
            | none => .ok ()
            | some p =>
              if bound.any (·.name == p.name) then
                .error s!"registered reads (bound): port {p.name}'s frame-v3-sha512 root or domain in the public file is not the window's committed one the verifier binds it to"
              else
                .error s!"registered reads (complete): the statement's port {p.name} is neither a port of the verifier's registered reads nor bound to the window's committed values ({KEY}): its rows would be committed afresh, stated by the public file"

theorem check_empty (c : Circuit) (pub : Public) (program : Option ByteArray) (own : Array Registered.Port)
    (declared : Option (Array (String × String))) :
    check c pub program own #[] declared = Registered.checkComplete c own := rfl

/-- An accepted `check` holds every port of the circuit (`held`). -/
theorem check_ok {c : Circuit} {pub : Public} {program : Option ByteArray} {own : Array Registered.Port}
    {bound : Array Bound} {declared : Option (Array (String × String))}
    (h : check c pub program own bound declared = .ok ()) : ∀ p ∈ c.ports, held pub own bound p = true := by
  intro p hp
  unfold check at h
  split at h
  · obtain ⟨r, hr, hn⟩ := Registered.checkComplete_ok h p hp
    simp only [held, Bool.or_eq_true]
    exact .inl (Array.any_eq_true'.mpr ⟨r, hr, by simp [hn]⟩)
  · split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · cases h
          · split at h
            · rename_i hn
              simpa using Array.find?_eq_none.mp hn p hp
            · split at h <;> cases h

/-- An accepted `check` registers every port of the circuit that the window's interface declares registered for its
template: such a port is a registered port of `own`, never bound and never committed afresh. -/
theorem check_declared {c : Circuit} {pub : Public} {program : Option ByteArray} {own : Array Registered.Port}
    {bound : Array Bound} {declared : Option (Array (String × String))} {ds : Array String}
    (h : check c pub program own bound declared = .ok ()) (hd : templatePorts c declared = .ok ds) :
    ∀ p ∈ c.ports, declaredIn ds p.name = true → own.any (·.name == p.name) = true := by
  intro p hp hdi
  unfold check at h
  split at h
  · obtain ⟨r, hr, hn⟩ := Registered.checkComplete_ok h p hp
    exact Array.any_eq_true'.mpr ⟨r, hr, by simp [hn]⟩
  · split at h
    · cases h
    · rename_i ds' hds'
      have hds : ds' = ds := Except.ok.inj (hds'.symm.trans hd)
      subst hds
      split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · cases h
          · rename_i hn
            have hm : ¬ (declaredMiss own ds' p = true) := Array.find?_eq_none.mp hn p hp
            unfold declaredMiss at hm
            rw [hdi] at hm
            simpa using hm

end RegBound
end Flock
