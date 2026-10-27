import PouwCheck.Credit

/-!
# A period's filler windows and the `fill` it credits beside its `work` (`fill.py`)

The verifier of record of `verity/core/protocols/pouw/fill.py`, which `tests/test_period_difftest.py` holds to it
through `pouw-period` (`PouwCheck.Period`), string for string. `periodFill` is the one entry.

A filler window is Dither calls (`circuit.dither`) on tensor pipes the served windows leave idle, each call's A its
per-call pad and its B a published checkpoint's s8 weights, committing its T_1 .. T_S; its proof blocks are
`PouwDitherBlock_v1` blocks, one cell and one segment each. It counts toward the companion period `<name>:fill`
(`periodName`), whose open commits the circuit and the draw K. `Credit.periodCredit` is unchanged.

* **The circuit is public structure** (`descriptorOf`): canonical JSON `{L, calls, checkpoint, k, lp, m, n, row, unit}`,
  and the window's open root is `openRoot` of its SHA-512 and the window's session.
* **The pad and noise are live coins** (`coins`): the window's challenger session (`families`: one salt coin, one call
  coin a call, one draw coin) shows the open root as the one root its salt coin and every call coin followed, and the
  window's salt is the salt coin's round-0 salt after it (`h1tile.live_salt`).
* **Every value is fixed before the draw** (`receipts`, `drawRoot`): the draw root is over the salt and every call's
  value root, and the challenger's log shows it as the one root the draw coin followed. The draw is over one stratum
  (`stratum`) of every call's blocks, L MACs each.
* **No partial credit**: every drawn block has exactly one accepted outcome (`Credit.proved`), verified against its
  block's `statement` with its `t` port opened at its call's value root (`registered`).

Every refusal is (code, why), the first rule the input breaks, with Python's words.
-/

namespace PouwCheck.Fill

open Lean (Json)
open PouwCommit (Part tagged)
open SampledProofs.Challenger (ChallengerKey LiveCommitment LiveOpening trustedSessionOf liveReason)

/-- `fill.SUFFIX`: the companion period a served period's filler windows count toward, `<name>:fill`. -/
def SUFFIX : String := ":fill"
/-- `fill.CALL`: a filler window's call coins, one a call, between its salt coin and its draw coin. -/
def CALL : String := "pouw/fill/call"
def UNIT : String := "PouwDitherBlock_v1"
def DESCRIPTOR_KEYS : List String := ["L", "calls", "checkpoint", "k", "lp", "m", "n", "row", "unit"]
def OPEN_TAG : String := "verity/pouw/fill/open/v2"
def ROOTS_TAG : String := "verity/pouw/fill/roots/v1"
def DRAW_TAG : String := "verity/pouw/fill/draw/v2"
def STATEMENT_TAG : String := "verity/pouw/fill/statement/v1"
/-- `fill.TEMPLATE`: the one stratum of a filler window's draw. -/
def TEMPLATE : String := "dither"
/-- `dither.TILE`, `dither.K_LIMIT` and `dither.TILE_WORD_BYTES`. -/
def TILE : Nat := 128
def K_LIMIT : Nat := 131072
def TILE_WORD_BYTES : Nat := 32768
/-- `fill.T_MAX`: the most L′-tiles a block's words span, t = min(L, K) / L′. -/
def T_MAX : Nat := 8
def FILL_KEYS : List String := ["calls", "descriptor", "draw", "opening", "proofs", "roots", "salt", "session", "window"]
def FILLS_FORM : String :=
  "the period's filler windows are not a list of each filler window's evidence {" ++ ", ".intercalate FILL_KEYS ++
  "}, in the order its fill period's close lists them"
def DESCRIPTOR_HEX : String :=
  "the filler window's descriptor is not the bytes in lowercase hex of the circuit its fill period committed"
def DESCRIPTOR_FORM : String :=
  "the filler window's descriptor is not canonical JSON {" ++ ", ".intercalate DESCRIPTOR_KEYS ++ "}: the " ++
  "Dither filler's shape and call count as positive integers, B's checkpoint digest (128 lowercase hex) and the unit " ++
  UNIT
def SHAPE : String :=
  s!"the filler window's shape is not a Dither filler's: m and n whole tiles of {TILE}, lp a multiple of 32 " ++
  s!"dividing k, L a multiple of lp dividing k with t = min(L, k) / lp at most {T_MAX}, k below {K_LIMIT}, " ++
  s!"and row a multiple of 16 dividing {TILE_WORD_BYTES}"
def OPEN : String :=
  "the filler window's open root is not open_root of its circuit and its session: open each filler window on its own " ++
  "root, after the circuit is fixed"
def FILL_EPOCH : String :=
  "the coin commitment's coins are not a filler window's: its salt coin, a call coin for each of its calls, then its " ++
  "draw coin"
def SALT_NOT_RECEIVED : String :=
  "the challenger's log does not show the filler window's open root as the one root its salt coin followed"
def FILL_SALT : String :=
  "the filler window's salt is not its live salt coin's round-0 salt after its open root (h1tile.live_salt): seed its " ++
  "calls from the coin the challenger opened"
def CALLS_FORM : String :=
  "the filler window's calls are not a list of each call's coin opening, one a call of its descriptor"
def CALLS_NOT_RECEIVED : String :=
  "the challenger's log does not show the filler window's open root as the one root each of its call coins " ++
  "followed: open every call coin after the open root, before the call's values"
def ROOTS_FORM : String :=
  "the filler window's roots are not a list of each call's value root (SHA-512 digests), one a call"
def RECEIPTS : String :=
  "the filler window's draw receipts are not the digest of its calls' value roots (receipts): commit every call's " ++
  "values before the draw coin and draw over them"
def DRAW_ROOT : String :=
  "the filler window's draw root is not over its salt and its calls' value roots (draw_root)"
def CHANGED : String :=
  "the draw root over the filler window's salt and its calls' value roots is not the one root the challenger's log " ++
  "shows its draw coin followed: a value changed after the draw coin"

/-- `fill.period_name`: the fill period of served period `name`. -/
def periodName (name : String) : String := name ++ SUFFIX

/-- `fill.families`: a filler window's coin layout, its salt coin, `calls` call coins, then its draw coin. -/
def families (calls : Nat) : List (String × Nat) := [(PouwWindow.SALT, 1), (CALL, calls), (PouwWindow.DRAW, 1)]

/-- A filler shape (`dither.Shape` with `o=False`) and its call count. -/
structure Shape where
  m : Nat
  n : Nat
  k : Nat
  lp : Nat
  L : Nat
  row : Nat
  calls : Nat

def Shape.S (sh : Shape) : Nat := sh.k / sh.L

/-- `fill.blocks`: a call's proof blocks, m·n cells of S segments each. -/
def Shape.blocks (sh : Shape) : Nat := sh.m * sh.n * sh.S

/-- `fill.work`: a filler window's work, m·n·K MACs a call. -/
def Shape.work (sh : Shape) : Nat := sh.calls * sh.m * sh.n * sh.k

/-- `fill.descriptor_of`: the filler shape and call count the descriptor `raw` names, else the refusal
(`descriptor`). -/
def descriptorOf (raw : ByteArray) : R Shape := do
  let some j := (String.fromUTF8? raw).bind Credit.loads? | throw ("descriptor", DESCRIPTOR_FORM)
  let pos (x : String) : Option Nat := (nat? (get j x)).filter (1 ≤ ·)
  let some sh := (do
      unless keys j == some DESCRIPTOR_KEYS && str? (get j "unit") == some UNIT
        && (Credit.digest? (get j "checkpoint")).isSome do none
      pure ({ m := ← pos "m", n := ← pos "n", k := ← pos "k", lp := ← pos "lp", L := ← pos "L", row := ← pos "row",
              calls := ← pos "calls" } : Shape)) | throw ("descriptor", DESCRIPTOR_FORM)
  unless sh.m % TILE == 0 && sh.n % TILE == 0 && sh.lp % 32 == 0 && sh.k % sh.lp == 0 && sh.k < K_LIMIT
    && sh.L % sh.lp == 0 && sh.row % 16 == 0 && TILE_WORD_BYTES % sh.row == 0 && sh.k % sh.L == 0
    && min sh.L sh.k / sh.lp ≤ T_MAX do
    throw ("descriptor", SHAPE)
  pure sh

/-- `fill.open_root`: a filler window's open root, `tagged` of its circuit and its session. -/
def openRoot (circuit : ByteArray) (session : String) : ByteArray := tagged OPEN_TAG [.bytes circuit, .str session]

/-- `fill.receipts`: the digest of a filler window's calls' value roots, in call order. -/
def receipts (roots : List ByteArray) : ByteArray := tagged ROOTS_TAG (roots.map Part.bytes)

/-- `fill.draw_root`: a filler window's draw root, over its salt and `receipts` of its calls' value roots. -/
def drawRoot (salt receipts : ByteArray) : ByteArray := tagged DRAW_TAG [.bytes salt, .bytes receipts]

/-- `fill.block_of`: unit `u`'s block (c, i, j, s). -/
def blockOf (sh : Shape) (u : Nat) : Nat × Nat × Nat × Nat :=
  let r := u % sh.blocks
  let cell := r / sh.S
  (u / sh.blocks, cell / sh.n, cell % sh.n, r % sh.S + 1)

/-- `fill.stratum`: the draw's one stratum, every call's blocks, L MACs each. -/
def stratum (sh : Shape) : Array Flock.Draw.WorkStratum :=
  #[{ template := TEMPLATE, units := #[(0, sh.calls * sh.blocks)], k := 0, work := sh.L, floor := 1 }]

/-- `fill.statement`: the statement a drawn block's proof is verified against. -/
def statement (circuit salt coin : ByteArray) (b : Nat × Nat × Nat × Nat) : ByteArray :=
  tagged STATEMENT_TAG [.bytes circuit, .bytes salt, .bytes coin, .int b.1, .int b.2.1, .int b.2.2.1, .int b.2.2.2]

/-- `fill.registered`: a block proof's registered finding, its `t` port opened at call c's value root. -/
def registered (reg : Json) (c : Nat) (root : String) : Bool :=
  let t := get reg "t"
  keys reg == some ["t"] && keys t == some ["root", "value"] && str? (get t "root") == some root
    && str? (get t "value") == some s!"fill/T@{c}"

/-- `fill.eps`: the ε a draw of K = `k` establishes at `Credit.DELTA` over the whole of a filler window's work, or none
when it establishes none. -/
def eps (sh : Shape) (k : Nat) : Option Rat := (PouwWindow.epsAt k Credit.DELTA sh.work sh.work 1).toOption

def live (c : LiveCommitment) (doc : Json) (label : String) (index : Nat) (after : ByteArray) (what : String) :
    R Unit := do
  let why := match LiveOpening.ofJson doc with
    | .error e => some e
    | .ok o => match liveReason c o label index after with
      | .error e => some e
      | .ok w => w
  if let some w := why then throw ("coin", s!"{what}: {w}")

/-- `fill._coins`: each call's coin when the filler window's salt, call and draw coins are live, read through its
challenger's session; else the refusal (`coin`, `salt`, `calls` or `draw`). -/
def coins (k : ChallengerKey) (ledger pins main window w : Json) (calls : Nat) : R (List ByteArray) := do
  let f := get w
  let r := (Credit.digest? (get window "root")).getD .empty
  let (c, received) ← (trustedSessionOf ledger window k pins main (f "session")).mapError fun e =>
    ("coin", s!"the challenger's session is not trusted: {e}")
  if c.scheme != PouwWindow.LIVE_SCHEME then
    throw ("coin", s!"the coin commitment's scheme is {c.scheme}, not {PouwWindow.LIVE_SCHEME}")
  unless c.families == families calls do throw ("coin", FILL_EPOCH)
  unless (received.lookup PouwWindow.SALT).getD [] == [r] do throw ("salt", SALT_NOT_RECEIVED)
  live c (f "opening") PouwWindow.SALT 0 r "the salt's coin"
  let coin := (hex? (get (f "opening") "coin")).getD .empty
  let rk ← port (PouwWindow.roundKey c coin 0 r)
  let salt := FlockSoundness.Randomness.stream rk [.int 0] SALT_BYTES
  unless str? (f "salt") == some (Flock.hexOf salt) do throw ("salt", FILL_SALT)
  let some os := (match f "calls" with
      | .arr a => if a.size == calls then some a.toList else none
      | _ => none) | throw ("calls", CALLS_FORM)
  unless (received.lookup CALL).getD [] == List.replicate calls r do throw ("calls", CALLS_NOT_RECEIVED)
  let mut out : List ByteArray := []
  for (o, i) in os.zipIdx do
    live c o CALL i r s!"call {i}'s coin"
    out := out ++ [(hex? (get o "coin")).getD .empty]
  let some roots := (match f "roots" with
      | .arr a => if a.size == calls then a.toList.mapM Credit.digest? else none
      | _ => none) | throw ("roots", ROOTS_FORM)
  let rc := receipts roots
  let draw := f "draw"
  unless Credit.isObj draw && str? (get draw "receipts") == some (Flock.hexOf rc) do throw ("draw", RECEIPTS)
  let root := drawRoot salt rc
  unless str? (get draw "root") == some (Flock.hexOf root) do throw ("draw", DRAW_ROOT)
  let got := (received.lookup PouwWindow.DRAW).getD []
  if got.isEmpty then throw ("draw", Credit.NO_DRAW)
  unless got == [root] do throw ("draw", CHANGED)
  live c (get draw "coin") PouwWindow.DRAW 0 root "the draw coin"
  pure out

def otherStatement (u : Nat) : String :=
  s!"drawn block {u}'s proof was verified against another statement than its own (statement: the circuit, the " ++
  "window's salt, its call's coin and the block): verify every drawn block against its own statement"

def otherRegistered (u c : Nat) : String :=
  s!"drawn block {u}'s proof opened something other than its t port at call {c}'s value root fill/T@{c}: " ++
  "verify every drawn block with its call's values registered"

/-- `fill._window`: a filler window's shape, else the refusal of the first step that refuses: its root's form, its
descriptor (`descriptorOf`), its open root (`openRoot`, `open`), `coins`, the draw at the fill period's committed K `k`
(`Credit.windowDrawn` over `stratum`), each drawn block's one accepted outcome (`Credit.proved`), its `statement` and its
`registered` finding. -/
def window (k : ChallengerKey) (ledger pins main w : Json) (session : String) (circuit : ByteArray) (bigK : Nat) :
    R Shape := do
  let f := get w
  let window := f "window"
  unless keys window == some ["root"] && (Credit.digest? (get window "root")).isSome do
    throw ("fill", Credit.WINDOW_ROOT_FORM)
  let sh ← descriptorOf ((hex? (f "descriptor")).getD .empty)
  unless str? (get window "root") == some (Flock.hexOf (openRoot circuit session)) do throw ("open", OPEN)
  let cs ← coins k ledger pins main window w sh.calls
  Credit.proved (← Credit.windowDrawn window (f "draw") (Lean.toJson bigK) (stratum sh)) (f "proofs")
  let salt := (hex? (f "salt")).getD .empty
  let roots := match f "roots" with
    | .arr a => a.toList.map fun x => (str? x).getD ""
    | _ => []
  let .arr ps := f "proofs" | pure sh
  for p in ps.toList do
    let u := (nat? (get p "unit")).getD 0
    let b := blockOf sh u
    unless str? (get p "program") == some (Flock.hexOf (statement circuit salt (cs.getD b.1 .empty) b)) do
      throw ("statement", otherStatement u)
    unless registered (get p "registered") b.1 (roots.getD b.1 "") do throw ("registered", otherRegistered u b.1)
  pure sh

/-- `fill.period_fill`'s fill: each filler window's work in its fill period's close order, the fill period's K, the
δ the ε is reported at and each window's ε. -/
structure FillCredit where
  work : List Nat
  k : Nat
  delta : Rat
  eps : List (Option Rat)

def FillCredit.toJson (c : FillCredit) : Json :=
  Json.mkObj [("work", Lean.toJson c.work), ("k", c.k), ("delta", toString c.delta),
    ("eps", Json.arr (c.eps.map fun e => (e.map fun e => Json.str (toString e)).getD .null).toArray)]

/-- `fill.period_fill`: served period `name`'s fill. Its filler windows `fills` (`FILL_KEYS` each) are exactly the
close's of the fill period (`periodName`, `Credit.period`), whose open committed K (`Credit.committedK`); each window's
session ran a commit the host's pins log `pins` records on main's first-parent history `main`, the window proves the
circuit the fill period's open committed (`circuit`, `creditedSession`) within its freshness floor (`pin`,
`Credit.pinHygiene`), and `window` credits it (`fill`). A refused window refuses the fill. -/
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
  let periodPin := Credit.committedPin ledger nm
  let floor := Credit.committedFloor ledger nm
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let mut out : List Nat := []
  let mut eps_ : List (Option Rat) := []
  for (w, i) in ws.zipIdx do
    let f := get w
    let some d := (match f "descriptor" with
        | .str s => if Credit.isHexBytes s then hex? (.str s) else none
        | _ => none) | throw ("circuit", s!"filler window {i} of period {nm}: {DESCRIPTOR_HEX}")
    let a ← (SampledProofs.Challenger.creditedSession ledger (f "window") k pins main d).mapError fun e =>
      ("circuit", s!"filler window {i} of period {nm} doesn't count toward it, so the fill earns nothing: {e}")
    if let some why := Credit.pinHygiene nm i main a.pinned periodPin floor then throw why
    let sh ← (window k ledger pins main w a.session (Flock.Sha512.hash d) bigK).mapError fun (c, why) =>
      ("fill", s!"filler window {i} of period {nm} is refused, so the fill earns nothing: {c}: {why}")
    out := out ++ [sh.work]
    eps_ := eps_ ++ [eps sh bigK]
  pure ⟨out, bigK, Credit.DELTA, eps_⟩

end PouwCheck.Fill
