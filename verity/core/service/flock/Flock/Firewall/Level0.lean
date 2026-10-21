/-
# A table's level 0, as the session loop holds it

The session loop draws each firewalled table's level-0 hiding itself, from the operating system (as it draws `Hello`'s
nonce), and commits level 0 on it (`Link`): the prover hands it only the table's packed witness. What it keeps of the
table is one `Level0`, from which everything of level 0 that leaves is read: the root in the session's `Commit`, and
`Finish`'s level-0 openings (`Firewall.Finish`, ZKFIN's).

* `Level0`: the table's `m`, level 0 of its `--zk` schedule and its `t_pad`, the packed witness `g`, the hiding the loop
  drew (the four extra lanes, every lane's padding, a salt per tree position), the padded codeword's rows and the root.
* `ofDraws`: a table's `Level0` from its witness and the draws' bytes, in `Link.parse`'s layout after the witness: the
  extra lanes, the paddings, the salts. `draw` reads those bytes from the OS in one read.
* `Registry`, `commit`: the loop's tables, and a session's `Commit` held to them: each root it carries must be a
  `Level0`'s the loop drew at the table's own `m`, which the session then owns.

The prover gets the draws back (`drawsOf`) to run its own commitment over the same codeword; what leaves is the loop's.
-/
import Flock.Firewall.Link

namespace Flock.Firewall.Level0

/-- **A table's level 0**: its `m`, level 0 of its `--zk` schedule (`lv`) and `t_pad`; the host's packed witness `g`
(`2^(lanes + cols)` words); the hiding the loop drew: the four extra lanes (`extra`, `2^cols` words each), every lane's
padding coefficients (`pads`, `t_pad` each, the witness lanes' then the extra lanes'), and the salts (`salts`, 192 bytes
for each of the `2^d` tree positions, `salt q`); the padded codeword's `2^d` rows (`Link.rows`) and the root
(`Link.root`'s: the cap, concatenated). -/
structure Level0 where
  m : Nat
  lv : Level
  tPad : Nat
  g : Array F128
  extra : Array (Array F128)
  pads : Array (Array F128)
  salts : ByteArray
  rows : Array (Array F128)
  root : ByteArray
  deriving Inhabited

/-- The salt of tree position `q`. -/
def Level0.salt (l : Level0) (q : Nat) : ByteArray :=
  if q < 2 ^ l.lv.d then l.salts.extract (192 * q) (192 * (q + 1)) else .empty

/-- The table as `flock-firewall link` reads one (`Link.In`). -/
def Level0.toIn (l : Level0) : Link.In where
  m := l.m
  g := l.g
  extra e c := (l.extra[e]!)[c]!
  lanePad i k := (l.pads[i]!)[k]!
  salt := l.salt

/-- The OS bytes a table's level 0 draws at `m`: the extra lanes' words, the paddings' and the salts. -/
def drawBytes (m : Nat) : V Nat := do
  let (_, nC, nL, tPad, nQ) ← Link.sizes m
  return 16 * Zk.EXTRA_LANES * nC + 16 * nL * tPad + 192 * nQ

/-- **A table's level 0** at `m` from its packed witness `g` and the draws' bytes `d` (`drawBytes m`). -/
def ofDraws (m : Nat) (g : Array F128) (d : ByteArray) : V Level0 := do
  let (s, tPad) ← Zk.schedule m
  let lv := s.levels.getD 0 default
  let (nG, nC, nL, _, nQ) ← Link.sizes m
  if g.size != nG then throw s!"level 0: a {g.size}-word witness where m = {m} takes {nG}"
  let want ← drawBytes m
  if d.size != want then throw s!"level 0: {d.size} bytes drawn where m = {m} takes {want}"
  let extra := (Array.range Zk.EXTRA_LANES).map fun e => Link.wordsAt d (16 * nC * e) nC
  let oP := 16 * Zk.EXTRA_LANES * nC
  let pads := (Array.range nL).map fun l => Link.wordsAt d (oP + 16 * tPad * l) tPad
  let salts := d.extract (oP + 16 * nL * tPad) d.size
  let l : Level0 := { m, lv, tPad, g, extra, pads, salts, rows := #[], root := .empty }
  let i := l.toIn
  let rows := Link.rows lv tPad i.g i.extra i.lanePad
  let k := min lv.d Link.CHUNK
  let lvs := Link.parRange (fun q => Link.leaf rows[q]! (i.salt q)) (2 ^ (lv.d - k)) (2 ^ k)
  let root := Link.flat (Link.layer lv.d (fun q => lvs[q]!) (lv.d - lv.cap))
  if nQ != rows.size then throw s!"level 0: {rows.size} rows where m = {m} has {nQ} tree positions"
  return { l with rows, root }

/-- **A table's level 0** at `m` from its packed witness `g`, its hiding drawn from `get` in one read. -/
def draw (get : USize → IO ByteArray) (m : Nat) (g : Array F128) : IO (V Level0) := do
  match drawBytes m with
  | .error e => return .error e
  | .ok n =>
    let d ← get n.toUSize
    return ofDraws m g d

/-- What the prover gets back of its table's level 0: the draws, in `ofDraws`'s layout, then the root. -/
def drawsOf (l : Level0) : ByteArray :=
  let w := fun (v : Array F128) => v.foldl F128.pushBytes .empty
  (l.extra ++ l.pads).foldl (fun b v => b ++ w v) .empty ++ l.salts ++ l.root

/-- A table's level 0 in the loop, and the session and table it went to, once a session's `Commit` carried its root. -/
structure Slot where
  l0 : Level0
  bound : Option (Nat × String) := none
  deriving Inhabited

abbrev Registry := Array Slot

/-- **Session `sid`'s `Commit`**, of the session's tables `tables` (each with its statement's `m`), carrying `roots`: each
root, in table order, a `Level0`'s the loop drew at its table's `m` and no other session's or table's, which the session
then owns; else why it does not leave. -/
def commit (R : Registry) (sid : Nat) (tables : Array (String × Nat)) (roots : List (String × ByteArray)) :
    Except String Registry := do
  if roots.map (·.1) != tables.toList.map (·.1) then
    throw s!"a Commit whose roots are of {roots.map (·.1)}, where the session's tables are {tables.toList.map (·.1)}"
  let mut R := R
  for ((t, r), (_, m)) in roots.zip tables.toList do
    let ok := fun (s : Slot) => s.l0.root == r && (s.bound.isNone || s.bound == some (sid, t))
    match R.findIdx? ok with
    | none => throw s!"a Commit whose root of {t} no Lean level 0 drew"
    | some i =>
      let s := R[i]!
      if s.l0.m != m then throw s!"a Commit whose root of {t} was drawn at m = {s.l0.m}, where its statement's is {m}"
      R := R.set! i { s with bound := some (sid, t) }
  return R

/-- Session `sid`'s table `t`'s level 0, once its `Commit` left. -/
def ofSession (R : Registry) (sid : Nat) (t : String) : Option Level0 :=
  (R.find? (·.bound == some (sid, t))).map (·.l0)

end Flock.Firewall.Level0
