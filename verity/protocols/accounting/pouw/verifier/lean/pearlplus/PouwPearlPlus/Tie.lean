module

public import PouwPearlPlus.Tile
public import PouwPearlPlus.Load
public import Flock.WindowRecord
meta import PouwPearlPlus.Tile
meta import PouwPearlPlus.Load
meta import Flock.WindowRecord

/-!
# The window record's tie (`verify_run.tie`, `verify_run.tie_entries`)

`tieOf w passes rounds dropped` holds the run's loaded passes to the window record W (`--tie W`, a
`verity/registration/v2` record that `served_passes.py retain --window-record` writes): among W's ``pearl-a/``,
``pearl-y/`` and ``pearl-b/`` values, W registers exactly each call's A rows and y and each weight's B rows, pass i as round
``rounds[i]`` (default i), beside any of the ``dropped`` rounds' calls' (passes left out of the retention before the coin).
For pass s's call c, and each weight w the passes read:

* ``pearl-a/{s}/{c}``: the manifest's ``root_a``, its m committed rows of K 32-bit words, under `activationBinding`'s
  `hm96-sha512` domain;
* ``pearl-y/{s}/{c}``: the manifest's y root, its real rows of N 32-bit words, under `served_y.y_domain`'s;
* ``pearl-b/{w}``: the pass's weight root (`Pass.weightRoots`, which the weights' root folds and `checkTile` opens B
  against), its n rows of K 32-bit words, under `weightBinding`'s, one entry however many passes read it.

That is the tie of a W at version 2, its ``window`` header's ``version`` (`wVersion`, `pearl_c.window.version_of`). A
version-2 W that registers no ``pearl-b/`` value at all is refused as untied (`UNTIED_WEIGHTS`), whatever its calls'
entries: the run's B is then no registered value. A W that states no version is version 1, written before W registered
the weights: it is held to the calls alone, as it was then, and the answer names its weights untied (`WEIGHTS_UNTIED`);
one that registers a ``pearl-b/`` value is refused (`V1_WEIGHTS`), and a W stating any other version is refused
(`UNKNOWN_VERSION`).

An entry that names no ``word_bits`` is at `registered.WORD_BITS` (16). The answer is `{record: digest(W), tables}`, with
`weights` for a version-1 W; the first mismatch is Python's `ValueError`, in its words. Every JSON comparison is by a
match (`Want.holds`).

`tieAt load w passes rounds dropped` is the verdict's tie step (`verify_passes`' ``window`` and ``window_class``): `tieOf`, and
then, given the window's class, the passes must be that class's fixed load (`heldToLoad`, `verify_run.held_to_load`):
the retained and the dropped rounds are each of the class's `outputs` rounds once, and every call of a retained round
has that round's rows, `slots × prompt` at round 0 and `slots` after, committed in rows padded to `ROW_ALIGN`
(`window_class.PearlTables`). Passes of another load are refused in the class's own words (`offLoad`), which name no
round, call or count of the passes'; W's header names no class, so the verdict takes it as an input. Only at a held load
are the passes', the calls' and W's tables' counts the class's and the model's, and so protocol structure.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace PouwCheck.PearlPlus

open Lean PouwCommit

def TIE_PREFIXES : List String := ["pearl-a/", "pearl-y/", "pearl-b/"]
/-- `verify_run.TIE_CALLS`: a call's values, a round's. -/
def TIE_CALLS : List String := ["pearl-a/", "pearl-y/"]

/-- `verify_run.UNTIED_WEIGHTS`: why a version-2 W that ties the calls alone is refused. -/
def UNTIED_WEIGHTS : String :=
  "W registers no weight's B rows (no pearl-b/ value: a window record written before W registered them)"
/-- `verify_run.WEIGHTS_UNTIED`: the tie's `weights` for a version-1 W, which ties the calls alone. -/
def WEIGHTS_UNTIED : String := "weights untied (W v1)"
/-- `verify_run.V1_WEIGHTS`: why a version-1 W that registers a weight's B rows is refused. -/
def V1_WEIGHTS : String := "W states no version (version 1), whose values are the calls' alone, and registers"
/-- `pearl_c.window.VERSION`: the version of a W that registers the weights' B rows. -/
def W_VERSION : Nat := 2
/-- `pearl_c.window.UNKNOWN_VERSION`: why a W stating a version other than `W_VERSION` is refused. -/
def UNKNOWN_VERSION : String :=
  s!"W's window states a version other than {W_VERSION}: a W states version {W_VERSION}, or none (version 1, " ++
    "written before W registered the weights)"

/-- `pearl_c.window.version_of`: W's ``window`` header's ``version``, 1 where it states none, else `UNKNOWN_VERSION`. -/
def wVersion (w : Json) : Except String Nat :=
  match (get? w "window").bind (get? · "version") with
  | none => pure 1
  | some (.num ⟨n, 0⟩) => if n == (W_VERSION : Int) then pure W_VERSION else throw UNKNOWN_VERSION
  | some _ => throw UNKNOWN_VERSION

/-- `verify_run._owner`: what a value is of, in a mismatch's words. -/
def tieOwner (v : String) : String := if v.startsWith "pearl-b/" then "weight" else "call"
/-- `registered.WORD_BITS`: the word width of an entry that names none. -/
def REGISTERED_WORD_BITS : Nat := 16

/-- A value an entry must hold: a string, or a JSON integer (Python's `int`, never a `bool` or a `float`). -/
inductive Want where
  | str (s : String)
  | nat (n : Nat)
  deriving BEq

def Want.holds : Want → Json → Bool
  | .str s, j => fieldIs (some j) s
  | .nat n, .num ⟨m, 0⟩ => m == (n : Int)
  | .nat _, _ => false

/-- One call's entry as `tie_entries` wants it: root, leaves, words, word_bits and domain, in that order. -/
abbrev TieEntry := List (String × Want)

private def rawStr (j : Option Json) : String := match j with | some (.str s) => s | _ => ""

private def domainHex (d : Except String RowsDomain) : Except String String := do
  pure (hex (← d).frame.id)

/-- `tie_entries`' B rows: pass by pass (each with its round s), each weight its records read, by ascending id, as
``pearl-b/{w}``: its n rows of K 32-bit words at the pass's weight root, under `weightBinding`'s domain. A weight is one
entry, the first pass's; a later pass that commits it under another root or shape is Python's `ValueError`. -/
def tieWeights (passes : List (Nat × Pass)) : Except String (List (String × TieEntry)) := do
  let mut out : List (String × TieEntry) := []
  for (s, p) in passes do
    let us := p.records.map (·.matmul)
    for w in ((us.map (·.weight)).eraseDups.toArray.qsort (· < ·)).toList do
      let some u := us.find? (·.weight == w) | continue
      let d ← domainHex (rowsDomain (weightBinding w) (-2) u.n HM96 false)
      let x : TieEntry := [("root", .str (hex ((p.weightRoots.lookup w).getD ByteArray.empty))), ("leaves", .nat u.n),
        ("words", .nat u.k), ("word_bits", .nat 32), ("domain", .str d)]
      let v := s!"pearl-b/{w}"
      match out.lookup v with
      | none => out := out ++ [(v, x)]
      | some y =>
        if y != x then
          throw s!"pass {s} commits weight {w}'s B rows under another root or shape than an earlier pass, and W registers one"
  pure out

/-- `tie_entries(passes, rounds, weights)`: what W must register for each call of the run, then, with `weights` (a
version-2 W's), for each weight the passes read, in Python's order. `passes` are the manifests with their loaded passes. -/
def tieEntries (passes : List (Json × Pass)) (rounds : Option (List Nat)) (weights : Bool := true) :
    Except String (List (String × TieEntry)) := do
  let ss := rounds.getD (List.range passes.length)
  let calls ← go ss passes
  pure (calls ++ (← if weights then tieWeights (ss.zip (passes.map (·.2))) else pure []))
where
  go : List Nat → List (Json × Pass) → Except String (List (String × TieEntry))
    | [], [] => pure []
    | [], _ :: _ => throw "zip() argument 2 is longer than argument 1"
    | _ :: _, [] => throw "zip() argument 2 is shorter than argument 1"
    | s :: rs, (man, p) :: ps => do
      let us := match get? man "matmuls" with | some (.arr us) => us.toList | _ => []
      let mut out : List (String × TieEntry) := []
      for ((r, rec), c) in (p.records.zip us).zipIdx do
        let u := r.matmul
        let some y := get? rec "y" | throw s!"pass {s}'s call {c} (index {u.index}) commits no y to tie"
        let da ← domainHex (rowsDomain (activationBinding p.salt u.index) u.index u.m HM96 false)
        let dy ← domainHex (rowsDomain (yBinding p.salt u.index) u.index r.realRows HM96 false)
        out := out ++ [(s!"pearl-a/{s}/{c}", [("root", .str (rawStr (get? rec "root_a"))), ("leaves", .nat u.m),
            ("words", .nat u.k), ("word_bits", .nat 32), ("domain", .str da)]),
          (s!"pearl-y/{s}/{c}", [("root", .str (rawStr (get? y "root"))), ("leaves", .nat r.realRows),
            ("words", .nat u.n), ("word_bits", .nat 32), ("domain", .str dy)])]
      pure (out ++ (← go rs ps))

/-- `tie(w, passes, rounds, dropped)`: W's digest and the number of tables it ties, with `WEIGHTS_UNTIED` for a
version-1 W, or the first mismatch. -/
def tieOf (w : Json) (passes : List (Json × Pass)) (rounds : Option (List Nat)) (dropped : List Nat) :
    Except String Tied := do
  let regs ← match w, get? w "registered" with
    | .obj _, some (.arr es) =>
      if fieldIs (get? w "format") Flock.WindowRecord.FORMAT_V2 then pure es.toList
      else throw s!"W is no {Flock.WindowRecord.FORMAT_V2} record with registered values"
    | _, _ => throw s!"W is no {Flock.WindowRecord.FORMAT_V2} record with registered values"
  let tied := (← wVersion w) == W_VERSION
  let mut got : List (String × Json) := []
  for e in regs do
    let v := match e with | .obj _ => get? e "value" | _ => none
    if let some (.str v) := v then
      if TIE_PREFIXES.any (fun t => v.startsWith t) then
        if (got.lookup v).isSome then throw s!"W registers {v} twice"
        got := got ++ [(v, e)]
  if !tied then
    if let some (b, _) := got.find? (·.1.startsWith "pearl-b/") then
      throw s!"{V1_WEIGHTS} {b}: a W that registers the weights states version {W_VERSION}"
  let want ← tieEntries passes rounds tied
  let skip := dropped.map toString
  let extra := got.map (·.1) |>.filter fun v =>
    (want.lookup v).isNone &&
      !(TIE_CALLS.any (fun t => v.startsWith t) && skip.contains ((v.splitOn "/")[1]?.getD ""))
  if let x :: xs := extra then
    let first := xs.foldl (fun m y => if y < m then y else m) x
    throw s!"W registers {first}, no {tieOwner first} of the run's"
  if let some (b, _) := want.find? (·.1.startsWith "pearl-b/") then
    if !got.any (·.1.startsWith "pearl-b/") then
      throw s!"{UNTIED_WEIGHTS}, so the run's rootB is no W's ({b} first)"
  for (v, x) in want do
    let some e := got.lookup v | throw s!"W registers no {v}"
    let held (k : String) : Json :=
      if k == "word_bits" then (get? e k).getD (.num ⟨REGISTERED_WORD_BITS, 0⟩) else (get? e k).getD .null
    let bad := x.filterMap fun (k, want) => if want.holds (held k) then none else some k
    if !bad.isEmpty then throw s!"W's {v}: its {", ".intercalate bad} not the {tieOwner v}'s"
  pure { record := ← Flock.WindowRecord.digestHex w, tables := want.length,
         weights := if tied then none else some WEIGHTS_UNTIED }

/-- `pearl_c.ROW_ALIGN`: a call's A rows are committed in rows padded to a multiple of it. -/
def ROW_ALIGN : Nat := 16

/-- `verify_run.off_load`: why passes that aren't the class's fixed load are refused, in the class's words alone. -/
def offLoad (c : LoadClass) : String :=
  s!"the passes are not the fixed load of the window's class ({c.slots} slots, {c.prompt} prompt ids, {c.outputs} " ++
    s!"outputs): a pass a round of its {c.outputs}, each retained or dropped, every call of round 0 of " ++
    s!"{c.slots * c.prompt} rows and of a later round of {c.slots}, committed in rows padded to a multiple of {ROW_ALIGN}"

/-- `verify_run.load_rows`: a round-`s` call's (rows, committed A rows) under the class's fixed load
(`window_class.PearlTables`' `rows` and `a_rows`). -/
def loadRows (c : LoadClass) (s : Nat) : Nat × Nat :=
  let rows := if s == 0 then c.slots * c.prompt else c.slots
  (rows, (rows + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN)

/-- `verify_run.held_to_load`: whether the retained passes, pass i as round `rounds[i]` (default i), are the class's
fixed load: the retained and the dropped rounds are each of the class's `outputs` rounds once, and every call of a
retained round has that round's `loadRows`. -/
def heldToLoad (c : LoadClass) (passes : List Pass) (rounds : Option (List Nat)) (dropped : List Nat) : Bool :=
  let ss := rounds.getD (List.range passes.length)
  ss.length == passes.length &&
    ((ss ++ dropped).toArray.qsort (· < ·)).toList == List.range c.outputs &&
    (ss.zip passes).all fun (s, p) => p.records.all fun r => (r.realRows, r.matmul.m) == loadRows c s

/-- `verify_passes`' tie step: `tieOf`, then, given the window's class `load`, the passes are its fixed load
(`heldToLoad`) and the tie names it, or the refusal is `offLoad`'s. -/
def tieAt (load : Option LoadClass) (w : Json) (passes : List (Json × Pass)) (rounds : Option (List Nat))
    (dropped : List Nat) : Except String Tied := do
  let t ← tieOf w passes rounds dropped
  match load with
  | none => pure t
  | some c =>
    if heldToLoad c (passes.map (·.2)) rounds dropped then pure { t with load := some c } else throw (offLoad c)

#guard loadRows ⟨4, 32, 4⟩ 0 == (128, 128) && loadRows ⟨4, 32, 4⟩ 1 == (4, 16) && loadRows ⟨2, 8, 2⟩ 0 == (16, 16)
#guard loadRows ⟨3, 7, 2⟩ 0 == (21, 32) && loadRows ⟨3, 7, 2⟩ 5 == (3, 16)
#guard offLoad ⟨2, 8, 2⟩ ==
  "the passes are not the fixed load of the window's class (2 slots, 8 prompt ids, 2 outputs): a pass a round of " ++
    "its 2, each retained or dropped, every call of round 0 of 16 rows and of a later round of 2, committed in rows " ++
    "padded to a multiple of 16"

end PouwCheck.PearlPlus
