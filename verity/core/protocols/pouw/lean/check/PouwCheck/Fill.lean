import PouwCheck.Credit

/-!
# A period's filler windows and the `fill` it credits beside its `work` (`fill.py`; CMP-172)

The verifier of record of `verity/core/protocols/pouw/fill.py`, which `tests/test_period_difftest.py` holds to it
through `pouw-period` (`PouwCheck.Period`), string for string. `periodFill` is the one entry.

A filler window is an H1 window the prover runs on tensor pipes its served windows leave idle: the served window's
Program (`Credit.layoutOf` of its descriptor) at one round, its units J10-R tiles of the period's template, credited
every unit's work (`work`, M·n·K a call: no served rows, since every row of its A is the pad). It counts toward the
companion period `<name>:fill` (`periodName`), which the auditor's ledger opens and closes as it does a served period,
its open committing the circuit, the draw K and the weights' root and model. `Credit.periodCredit` is unchanged.

* **The pad is the verifier's** (`padEntry`): K's A table `<PREFIX>a/<K>@0` holds M rows for each call of K, row i its
  K int8 codes SHAKE-256(`PAD_TAG` ‖ u64be(K) ‖ u64be(i)) and the f32 scale 1.0 (`PAD_SCALE`), each row
  `Credit.chunks?` K leaves, under the public salts `padSalt` of the window's open root, registered binding it
  (`Credit.knownRoot`). The window's entry must be exactly it, so a filler unit can't choose its A.
* **The salt is the window's live coin** (`coins`): the window's challenger session (`FILL_FAMILIES`: one salt coin and
  one draw coin) is read through the auditor's ledger (`trustedSessionOf`); its log shows the window's open root as the
  one root its salt coin followed, and the Program salt is that coin's round-0 salt after it (`h1tile.live_salt`).
* **The open root binds the weights and the session** (`openRoot`): `windowOpenRoot` of the weights' root, the constant
  `FILL_REQUESTS` in the request table's place, and the session, which `Credit.periodWeights` checks.
* **Every value is fixed before the draw** (`entriesRoot`, `drawRoot`): the draw root is over the Program salt and the
  digest of every registered entry, and the challenger's log shows it as the one root the draw coin followed.
* **No partial credit**: every drawn unit has exactly one accepted outcome (`Credit.proved`), each opening its K's pad
  and weights at their registered roots (`Credit.Internal.registration`), and a refused window refuses the fill.

Every refusal is (code, why), the first rule the input breaks, with Python's words.
-/

namespace PouwCheck.Fill

open Lean (Json)
open PouwCommit (Part tagged)
open SampledProofs.Challenger (ChallengerKey LiveCommitment LiveOpening trustedSessionOf liveReason)

/-- `fill.SUFFIX`: the companion period a served period's filler windows count toward, `<name>:fill`. -/
def SUFFIX : String := ":fill"
/-- `fill.FILL_FAMILIES`: a filler window's coin layout, its salt coin, then its draw coin. -/
def FILL_FAMILIES : List (String × Nat) := [(PouwWindow.SALT, 1), (PouwWindow.DRAW, 1)]
/-- `fill.FILL_REQUESTS`: what stands in a filler window's open root for a served window's round-0 request table's
root. -/
def FILL_REQUESTS : ByteArray := Flock.Sha512.hash ("verity/pouw/fill/requests/v1".toUTF8.push 0)
def DRAW_TAG : String := "verity/pouw/fill/draw/v1"
def ENTRIES_TAG : String := "verity/pouw/fill/entries/v1"
def PAD_TAG : ByteArray := "verity/pouw/fill/pad/v1".toUTF8.push 0
def PAD_SALT_TAG : ByteArray := "verity/pouw/fill/pad-salt/v1".toUTF8.push 0
/-- `fill.PAD_SCALE`: every pad row's scale s_a, the f32 1.0 little-endian. -/
def PAD_SCALE : ByteArray := ⟨#[0x00, 0x00, 0x80, 0x3f]⟩
/-- `core_check.SALT_BYTES`: a pad leaf's hm96 salt. -/
def PAD_SALT_BYTES : Nat := 192
def FILL_KEYS : List String := ["descriptor", "draw", "entries", "opening", "proofs", "salt", "session", "window"]
def FILLS_FORM : String :=
  "the period's filler windows are not a list of each filler window's evidence {" ++ ", ".intercalate FILL_KEYS ++
  "}, in the order its fill period's close lists them"
def FILL_EPOCH : String := "the coin commitment's coins are not a filler window's: its salt coin, then its draw coin"
def SALT_NOT_RECEIVED : String :=
  "the challenger's log does not show the filler window's open root as the one root its salt coin followed"
def FILL_SALT : String :=
  "the filler window's Program salt is not its live salt coin's round-0 salt after its open root " ++
  "(h1tile.live_salt): seed its units from the coin the challenger opened"
def ENTRIES_FORM : String :=
  "the filler window's registered entries are not a list of one-stage's {value, root, leaves, words, domain}"
def RECEIPTS : String :=
  "the filler window's draw receipts are not the digest of its registered entries (entries_root): register " ++
  "every table before the draw coin and draw over them"
def DRAW_ROOT : String :=
  "the filler window's draw root is not over its Program salt and its registered entries (draw_root)"
def CHANGED : String :=
  "the draw root over the filler window's Program salt and registered entries is not the one root the " ++
  "challenger's log shows its draw coin followed: a value changed after the draw coin"

/-- `fill.period_name`: the fill period of served period `name`. -/
def periodName (name : String) : String := name ++ SUFFIX

/-- `fill.fill_epoch_reason`: why `c` is not a filler window's coin commitment, `none` when it is. -/
def fillEpochReason (c : LiveCommitment) : Option String :=
  if c.scheme != PouwWindow.LIVE_SCHEME then
    some s!"the coin commitment's scheme is {c.scheme}, not {PouwWindow.LIVE_SCHEME}"
  else if c.families == FILL_FAMILIES then none else some FILL_EPOCH

/-- `fill.open_root`: a filler window's open root, `windowOpenRoot` of its weights' root, `FILL_REQUESTS` and its
session. -/
def openRoot (weights : ByteArray) (session : String) : ByteArray := windowOpenRoot weights FILL_REQUESTS session

/-- `fill.pad_codes`: pad row `row`'s K int8 codes at K = `k`. -/
def padCodes (k row : Nat) : ByteArray :=
  Flock.Keccak.shake256 (PAD_TAG ++ PouwCommit.be8 k ++ PouwCommit.be8 row) k

/-- `weights.row_words`: bytes as u16 words, little-endian in pairs. -/
def words16 (b : ByteArray) : Array Nat :=
  (Array.range (b.size / 2)).map fun i => b[2 * i]!.toNat ||| (b[2 * i + 1]!.toNat <<< 8)

/-- `fill.pad_leaves`: the leaves of `rows` pad rows at K = `k`, row i's x = `Credit.chunks?` K leaves, chunk c its
K/x codes then `PAD_SCALE`, as u16 words. -/
def padLeaves (k rows : Nat) : Array (Array Nat) := Id.run do
  let x := (Credit.chunks? k).getD 1
  let kc := k / x
  let mut out : Array (Array Nat) := #[]
  for i in [0:rows] do
    let codes := padCodes k i
    for c in [0:x] do
      out := out.push (words16 (codes.extract (c * kc) ((c + 1) * kc) ++ PAD_SCALE))
  pure out

/-- `fill.pad_salt`: pad leaf `leaf`'s public hm96 salt in the window whose open root is `window` (hex). -/
def padSalt (window : String) (leaf : Nat) : ByteArray :=
  Flock.Keccak.shake256 (PAD_SALT_TAG ++ window.toUTF8.push 0 ++ PouwCommit.be8 leaf) PAD_SALT_BYTES

/-- `fill.pad_rows`: each K of `layout` (one `Credit.strata` accepted) and its pad table's rows, M for each call of K,
the Ks in the order of their first call. -/
def padRows (layout : Json) : List (Nat × Nat) := Id.run do
  let m := (nat? (get layout "m")).getD 0
  let lins : List (String × Nat) := match get layout "linears" with
    | .arr xs => xs.toList.filterMap fun x => match x with
      | .arr #[.str s, k, _] => (s, ·) <$> nat? k
      | _ => none
    | _ => []
  let calls : List String := match get layout "calls" with
    | .arr xs => xs.toList.filterMap fun x => match x with
      | .arr #[.str s, _] => some s
      | _ => none
    | _ => []
  let mut out : List (Nat × Nat) := []
  for lin in calls do
    let some k := lins.lookup lin | continue
    out := if (out.lookup k).isSome then out.map fun (k', r) => (k', if k' == k then r + m else r)
      else out ++ [(k, m)]
  pure out

/-- `fill.pad_entry`: K's pad table `<prefix>a/<K>@0` of `rows` rows as the filler window whose open root is `window`
registers it, under the circuit whose descriptor's SHA-512 is `program`. -/
def padEntry (program prefix_ window : String) (k rows : Nat) : R Json := do
  let leaves := padLeaves k rows
  let v := s!"{prefix_}a/{k}@0"
  let words := Credit.leafWords k
  let root ← (Credit.knownRoot program window v ((Array.range leaves.size).map fun i => (leaves[i]!, padSalt window i))
    words).mapError fun (_, why) => ("pad", why)
  let d ← (SampledProofs.Registered.domainAt program v words leaves.size 16 (some window)).mapError fun why =>
    ("pad", why)
  pure (Json.mkObj [("value", .str v), ("root", .str root), ("leaves", Lean.toJson leaves.size),
    ("words", Lean.toJson words), ("domain", .str (Flock.hexOf d))])

/-- `fill.entries_root`: the digest of a filler window's registered entries, in order: `tagged` of each one's canonical
JSON (`credit._canonical`). -/
def entriesRoot (entries : List Json) : ByteArray :=
  tagged ENTRIES_TAG (entries.map fun e => Part.bytes ((SampledProofs.PyJson.canonical e).toOption.getD "").toUTF8)

/-- `fill.draw_root`: a filler window's draw root, over its Program salt and `entriesRoot` of its entries. -/
def drawRoot (salt receipts : ByteArray) : ByteArray := tagged DRAW_TAG [.bytes salt, .bytes receipts]

/-- `fill.work`: a filler window's work, every unit's R·C·K, M·n·K a call. -/
def work (layout : Json) : Nat := (Credit.callWork layout).foldl (fun a (s, w) => a + s * w) 0

/-- `fill.eps`: the ε a draw of K = `k` establishes at `Credit.DELTA` over the whole of a filler window's work, or none
when it establishes none. -/
def eps (layout : Json) (k : Nat) : Option Rat :=
  (PouwWindow.epsAt k Credit.DELTA (work layout) (work layout) 1).toOption

/-- `fill._coins`: none when the filler window's salt and draw coins are live, read through its challenger's session;
else the refusal (`coin`, `salt`, `fill` or `draw`). -/
def coins (key ledger pins main window session salt opening draw entries : Json) : R Unit := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let some r := Credit.digest? (get window "root") | throw ("fill", Credit.WINDOW_ROOT_FORM)
  let (c, received) ← (trustedSessionOf ledger window k pins main session).mapError fun e =>
    ("coin", s!"the challenger's session is not trusted: {e}")
  if let some why := fillEpochReason c then throw ("coin", why)
  unless (received.lookup PouwWindow.SALT).getD [] == [r] do throw ("salt", SALT_NOT_RECEIVED)
  let o ← (LiveOpening.ofJson opening).mapError fun e => ("coin", s!"the salt's {e}")
  let why := match liveReason c o PouwWindow.SALT 0 r with
    | .error e => some e
    | .ok w => w
  if let some w := why then throw ("coin", s!"the salt's {w}")
  let rk ← port (PouwWindow.roundKey c o.coin 0 r)
  let want := FlockSoundness.Randomness.stream rk [.int 0] SALT_BYTES
  unless str? salt == some (Flock.hexOf want) do throw ("salt", FILL_SALT)
  let some es := (match entries with
      | .arr a => if a.all Credit.entryOk then some a.toList else none
      | _ => none) | throw ("fill", ENTRIES_FORM)
  let receipts := entriesRoot es
  unless Credit.isObj draw && str? (get draw "receipts") == some (Flock.hexOf receipts) do throw ("draw", RECEIPTS)
  let root := drawRoot want receipts
  unless str? (get draw "root") == some (Flock.hexOf root) do throw ("draw", DRAW_ROOT)
  let got := (received.lookup PouwWindow.DRAW).getD []
  if got.isEmpty then throw ("draw", Credit.NO_DRAW)
  unless got == [root] do throw ("draw", CHANGED)
  let why := match LiveOpening.ofJson (get draw "coin") with
    | .error e => some e
    | .ok o => match liveReason c o PouwWindow.DRAW 0 root with
      | .error e => some e
      | .ok w => w
  if let some w := why then throw ("coin", s!"the draw coin: {w}")

def noPad (v : String) : String :=
  s!"the filler window registers no pad table {v}, or registers it twice: each K's A rows are its one pad " ++
  "table, which the verifier builds (pad_entry)"

def otherPad (v : String) : String :=
  s!"the filler window's {v} is not the pad (pad_entry: each row its public codes and scale 1.0 under the " ++
  "public pad salts of the window's open root, registered binding it): a filler unit's A rows are the pad"

def otherValue (v pre : String) : String :=
  s!"the filler window registers {v}, which is none of its values: a filler window registers each K's pad " ++
  s!"{pre}a/<K>@0 and weights {pre}b/<K>, and each call's {pre}zy@<c> and {pre}receipt@<c>, " ++
  "nothing else"

/-- An entry equal to `want`, field for field, as Python's dicts compare. -/
def sameEntry (e want : Json) : Bool :=
  keys e == keys want && ["domain", "root", "value"].all (fun f => str? (get e f) == str? (get want f))
    && ["leaves", "words"].all (fun f => nat? (get e f) == nat? (get want f))

/-- `fill._pad`: none when the filler window's `entries` register each K's pad table (`padEntry`) once, and no value but
those, each K's weights and each call's z/y and receipt tables; else the refusal (`pad`). -/
def pad (entries : List Json) (layout : Json) (program prefix_ window : String) : R Unit := do
  let rows := padRows layout
  let pads := rows.map fun (k, _) => (s!"{prefix_}a/{k}@0", k)
  let calls := match get layout "calls" with
    | .arr a => a.size
    | _ => 0
  let others := rows.map (fun (k, _) => s!"{prefix_}b/{k}") ++
    (List.range calls).flatMap fun c => [s!"{prefix_}zy@{c}", s!"{prefix_}receipt@{c}"]
  let names := entries.map fun e => (str? (get e "value")).getD ""
  for v in names do
    unless (pads.lookup v).isSome || others.contains v do throw ("pad", otherValue v prefix_)
  for (v, k) in pads do
    unless names.count v == 1 do throw ("pad", noPad v)
    let want ← padEntry program prefix_ window k ((rows.lookup k).getD 0)
    let e := (entries.find? fun e => str? (get e "value") == some v).getD .null
    unless sameEntry e want do throw ("pad", otherPad v)

/-- `fill._window`: a filler window's layout, else the refusal of the first step that refuses: `coins`, the layout of
its circuit (`Credit.layoutOf` of `descriptor`), its weights (`Credit.periodWeights` at `openRoot`), `pad`, the draw at
the fill period's committed K `k` (`Credit.windowDrawn`), each drawn unit's one accepted outcome (`Credit.proved`) and
its registration (`Credit.Internal.registration`, one round). -/
def window (key ledger pins main window session salt opening draw descriptor entries proofs k : Json) : R Json := do
  coins key ledger pins main window session salt opening draw entries
  let some d := (match descriptor with
      | .str s => if Credit.isHexBytes s then hex? (.str s) else none
      | _ => none) | throw ("layout", Credit.DESCRIPTOR_HEX)
  let layout ← Credit.layoutOf d
  let program := Flock.hexOf (Flock.Sha512.hash d)
  let pre := Credit.prefixOf d
  let (sid, _) ← Credit.periodWeights key ledger window
    (.arr #[Json.mkObj [("requests", .str (Flock.hexOf FILL_REQUESTS))]]) layout entries program pre
  let ss ← Credit.strata layout
  let es := match entries with
    | .arr a => a.toList
    | _ => []
  let w := (str? (get window "root")).getD ""
  let p := (str? pre).getD ""
  pad es layout program p w
  Credit.proved (← Credit.windowDrawn window draw k ss) proofs
  Credit.Internal.registration w program p entries proofs ss layout 1 sid
  pure layout

/-- `fill._period_outcomes`' fill: each filler window's work in its fill period's close order, the fill period's K, the
δ the ε is reported at and each window's ε. -/
structure FillCredit where
  work : List Nat
  k : Nat
  delta : Rat
  eps : List (Option Rat)

def FillCredit.toJson (c : FillCredit) : Json :=
  Json.mkObj [("work", Lean.toJson c.work), ("k", c.k), ("delta", toString c.delta),
    ("eps", Json.arr (c.eps.map fun e => (e.map fun e => Json.str (toString e)).getD .null).toArray)]

/-- `fill._period_outcomes`: served period `name`'s fill. Its filler windows `fills` (`FILL_KEYS` each) are exactly
the close's of the fill period (`periodName`, `Credit.period`), whose open committed K (`Credit.committedK`) and
signed the weights' root (`Credit.committedWeights`); each window's session ran a commit the host's pins log `pins`
records on main's first-parent history `main`, the window proves the circuit the fill period's open committed
(`circuit`, `creditedSession`) within its freshness floor (`pin`, `Credit.pinHygiene`), and `window` credits it
(`fill`). A refused window refuses the fill. -/
def periodFill (key ledger pins main : Json) (name : String) (fills : Json) : R FillCredit := do
  let some ws := (match fills with
      | .arr a => if a.all (fun w => keys w == some FILL_KEYS) then some a.toList else none
      | _ => none) | throw ("fill", FILLS_FORM)
  let nm := periodName name
  let roots := Json.arr (ws.toArray.map fun w => match get w "window" with
    | j@(.obj _) => get j "root"
    | _ => .null)
  Credit.period key ledger (.str nm) roots
  let bigK ← Credit.committedK ledger nm
  discard <| Credit.committedWeights ledger nm
  let periodPin := Credit.committedPin ledger nm
  let floor := Credit.committedFloor ledger nm
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let mut out : List Nat := []
  let mut eps_ : List (Option Rat) := []
  for (w, i) in ws.zipIdx do
    let f := get w
    let some d := (match f "descriptor" with
        | .str s => if Credit.isHexBytes s then hex? (.str s) else none
        | _ => none) | throw ("circuit", s!"filler window {i} of period {nm}: {Credit.DESCRIPTOR_HEX}")
    let a ← (SampledProofs.Challenger.creditedSession ledger (f "window") k pins main d).mapError fun e =>
      ("circuit", s!"filler window {i} of period {nm} doesn't count toward it, so the fill earns nothing: {e}")
    if let some why := Credit.pinHygiene nm i main a.pinned periodPin floor then throw why
    let layout ← (window key ledger pins main (f "window") (f "session") (f "salt") (f "opening") (f "draw")
      (f "descriptor") (f "entries") (f "proofs") (Lean.toJson bigK)).mapError fun (c, why) =>
        ("fill", s!"filler window {i} of period {nm} is refused, so the fill earns nothing: {c}: {why}")
    out := out ++ [work layout]
    eps_ := eps_ ++ [eps layout bigK]
  pure ⟨out, bigK, Credit.DELTA, eps_⟩

end PouwCheck.Fill
