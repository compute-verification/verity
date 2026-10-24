/-
# A table's level 0, as the session loop holds it

The session loop draws each firewalled table's level-0 hiding itself, from the operating system (as it draws `Hello`'s
nonce), and commits level 0 on it (`Link`): the prover hands it only the table's packed witness. What it keeps of the
table is one `Level0`, from which everything of level 0 that leaves is read: the root in the session's `Commit`, and
`Finish`'s level-0 openings (`Firewall.Finish`, ZKFIN's).

* `Level0`: the table's `m`, level 0 of its `--zk` schedule and its `t_pad`, the packed witness `g`, the hiding the loop
  drew (the four extra lanes, every lane's padding, a salt per tree position), the padded codeword's rows, the Merkle
  tree's layers from the leaves to the cap, and the root.
* `ofDraws`: a table's `Level0` from its witness and the draws' bytes, in `Link.parse`'s layout after the witness: the
  extra lanes, the paddings, the salts. `draw` reads those bytes from the OS in one read.
* `ofDraws_ok`: what `ofDraws` keeps is what it commits: the rows, layers and root of a `Level0` it returns are `Link`'s
  of its own inputs (`toIn`), its root `Link.root toIn`, so a statement over `toIn` needs none of the stored fields.
* `Registry`, `commit`: the loop's tables, each drawn for one table of a live session named by its coin tree's root
  (`nameOf`, `Registry.draw`), once: a second draw for the same table, or one for a table the session does not have, is
  refused. Each is gone when that session ends (`Registry.close`). A session's `Commit` is held to them: the root it
  carries for its table `i` must be the `Level0` the loop drew for that session's table `i`, at that table's own `m`. So
  the level 0 a session binds is the one draw the loop made for the table, and the prover has no choice among draws.

The prover gets the draws back (`drawsOf`) to run its own commitment over the same codeword; what leaves is the loop's.
-/
import Flock.Firewall.Link

namespace Flock.Firewall.Level0

/-- **A table's level 0**: its `m`, level 0 of its `--zk` schedule (`lv`) and `t_pad`; the host's packed witness `g`
(`2^(lanes + cols)` words); the hiding the loop drew: the four extra lanes (`extra`, `2^cols` words each), every lane's
padding coefficients (`pads`, `t_pad` each, the witness lanes' then the extra lanes'), and the salts (`salts`, 192 bytes
for each of the `2^d` tree positions, `salt q`); and what level 0 commits of them (`build`, `ofDraws_ok`): the padded
codeword's `2^d` rows (`Link.rows`), the Merkle tree's layers from the leaves (`layers[0]`) to the cap
(`layers[d − cap]`, `Link.layer`'s), and the root (`Link.root`'s: the cap, concatenated). -/
structure Level0 where
  m : Nat
  lv : Level
  tPad : Nat
  g : Array F128
  extra : Array (Array F128)
  pads : Array (Array F128)
  salts : ByteArray
  rows : Array (Array F128)
  layers : Array (Array ByteArray)
  root : ByteArray
  deriving Inhabited

/-- The salt of tree position `q`. -/
def Level0.salt (l : Level0) (q : Nat) : ByteArray :=
  if q < 2 ^ l.lv.d then l.salts.extract (192 * q) (192 * (q + 1)) else .empty

/-- The table as `Link.parse` reads one (`Link.In`). -/
def Level0.toIn (l : Level0) : Link.In where
  m := l.m
  g := l.g
  extra e c := (l.extra[e]!)[c]!
  lanePad i k := (l.pads[i]!)[k]!
  salt := l.salt

/-- The layer above `below`: each node `SHA-512(l ‖ r)` of a pair (`Link.layer`'s step). -/
def up (below : Array ByteArray) : Array ByteArray :=
  (Array.range (below.size / 2)).map fun i => Sha512.hash (below[2 * i]! ++ below[2 * i + 1]!)

/-- A Merkle tree over the leaves `lf q` (`q < 2^d`), `k` layers up: its top layer, and its layers from the leaves to it,
each hashed once (`tower_fst`, `tower_snd`). -/
def tower (d : Nat) (lf : Nat → ByteArray) : Nat → Array ByteArray × Array (Array ByteArray)
  | 0 => let l := Link.layer d lf 0; (l, #[l])
  | k + 1 => let p := tower d lf k; let t := up p.1; (t, p.2.push t)

theorem layer_succ (d : Nat) (lf : Nat → ByteArray) (h : Nat) : Link.layer d lf (h + 1) = up (Link.layer d lf h) := rfl

theorem tower_fst (d : Nat) (lf : Nat → ByteArray) (k : Nat) : (tower d lf k).1 = Link.layer d lf k := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [tower, ih, layer_succ]

theorem tower_snd (d : Nat) (lf : Nat → ByteArray) (k : Nat) :
    (tower d lf k).2 = (Array.range (k + 1)).map (Link.layer d lf) := by
  induction k with
  | zero =>
    simp only [tower, Array.range_succ, Array.append_singleton, Array.range_eq_empty_iff.mpr (rfl : 0 = 0),
      Array.push_empty, Array.map_singleton]
  | succ k ih =>
    simp only [tower, ih, tower_fst, layer_succ, Array.range_succ, Array.append_singleton, Array.map_push]

/-- **A table's level 0** at level `lv` with `t_pad` paddings, from its packed witness `g` and the drawn hiding: the rows,
layers and root committed on them, as `Link` commits `toIn` (`build_rows`, `build_layers`, `build_root`). -/
def build (m : Nat) (lv : Level) (tPad : Nat) (g : Array F128) (extra pads : Array (Array F128)) (salts : ByteArray) :
    Level0 :=
  let l : Level0 := { m, lv, tPad, g, extra, pads, salts, rows := #[], layers := #[], root := .empty }
  let rows := Link.rows lv tPad l.toIn.g l.toIn.extra l.toIn.lanePad
  let lvs := Link.leaves lv tPad l.toIn
  let t := tower lv.d (fun q => lvs[q]!) (lv.d - lv.cap)
  { l with rows, layers := t.2, root := Link.flat t.1 }

/-- The OS bytes level 0 draws at level `lv` with `t_pad` paddings: the extra lanes' words, the paddings' and the
salts. -/
def drawSize (lv : Level) (tPad : Nat) : Nat :=
  16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad + 192 * 2 ^ lv.d

/-- The OS bytes a table's level 0 draws at `m` (`drawSize` at level 0 of its `--zk` schedule). -/
def drawBytes (m : Nat) : V Nat := do
  let (s, tPad) ← Zk.schedule m
  return drawSize (s.levels.getD 0 default) tPad

/-- **A table's level 0** at `m` from its packed witness `g` and the draws' bytes `d` (`drawBytes m`), in
`Link.parse`'s layout after the witness: the extra lanes, the paddings, the salts. -/
def ofDraws (m : Nat) (g : Array F128) (d : ByteArray) : V Level0 := do
  let (s, tPad) ← Zk.schedule m
  let lv := s.levels.getD 0 default
  let (nG, nC, nL) := (2 ^ (lv.lanes + lv.cols), 2 ^ lv.cols, 2 ^ lv.lanes + Zk.EXTRA_LANES)
  if g.size != nG then throw s!"level 0: a {g.size}-word witness where m = {m} takes {nG}"
  if d.size != drawSize lv tPad then throw s!"level 0: {d.size} bytes drawn where m = {m} takes {drawSize lv tPad}"
  let oP := 16 * Zk.EXTRA_LANES * nC
  let l := build m lv tPad g ((Array.range Zk.EXTRA_LANES).map fun e => Link.wordsAt d (16 * nC * e) nC)
    ((Array.range nL).map fun l => Link.wordsAt d (oP + 16 * tPad * l) tPad) (d.extract (oP + 16 * nL * tPad) d.size)
  if l.rows.size != 2 ^ lv.d then throw s!"level 0: {l.rows.size} rows where m = {m} has {2 ^ lv.d} tree positions"
  return l

section
variable {m : Nat} {lv : Level} {tPad : Nat} {g : Array F128} {extra pads : Array (Array F128)} {salts : ByteArray}
  {l : Level0}

theorem build_rows (h : l = build m lv tPad g extra pads salts) :
    l.rows = Link.rows l.lv l.tPad l.toIn.g l.toIn.extra l.toIn.lanePad := by
  subst h; rfl

theorem build_layers (h : l = build m lv tPad g extra pads salts) :
    l.layers = (Array.range (l.lv.d - l.lv.cap + 1)).map
      (Link.layer l.lv.d fun q => (Link.leaves l.lv l.tPad l.toIn)[q]!) := by
  subst h; exact tower_snd lv.d _ (lv.d - lv.cap)

theorem build_root (h : l = build m lv tPad g extra pads salts) :
    l.root = Link.flat (Link.cap l.lv l.tPad l.toIn) := by
  subst h; exact congrArg Link.flat (tower_fst lv.d _ (lv.d - lv.cap))

end

/-- `Link.root` of a table at level 0 of its `--zk` schedule is its cap's, concatenated. -/
theorem root_of (l : Level0) (s : Schedule) (hz : Zk.schedule l.m = .ok (s, l.tPad))
    (hl : l.lv = s.levels.getD 0 default) : Link.root l.toIn = Link.flat (Link.cap l.lv l.tPad l.toIn) := by
  unfold Link.root
  rw [show l.toIn.m = l.m from rfl, hz, hl]

/-- **What `ofDraws` keeps is what it commits**: a `Level0` it returns is `build` at level 0 of `m`'s `--zk` schedule, so
its rows, layers and root are `Link`'s of its own inputs (`toIn`), and its root is `Link.root`'s. -/
theorem ofDraws_ok {m : Nat} {g : Array F128} {d : ByteArray} {l : Level0} (h : ofDraws m g d = .ok l) :
    l.m = m ∧ l.g = g ∧ (∃ s, Zk.schedule m = .ok (s, l.tPad) ∧ l.lv = s.levels.getD 0 default) ∧
      l.rows = Link.rows l.lv l.tPad l.toIn.g l.toIn.extra l.toIn.lanePad ∧
      l.layers = (Array.range (l.lv.d - l.lv.cap + 1)).map
        (Link.layer l.lv.d fun q => (Link.leaves l.lv l.tPad l.toIn)[q]!) ∧
      l.root = Link.root l.toIn := by
  unfold ofDraws at h
  cases hz : Zk.schedule m with
  | error e => rw [hz] at h; cases h
  | ok p =>
    obtain ⟨s, tPad⟩ := p
    rw [hz] at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    repeat (split at h; · cases h)
    injection h with h
    have hb := h.symm
    have hm : l.m = m := by subst hb; rfl
    have ht : l.tPad = tPad := by subst hb; rfl
    have hl : l.lv = s.levels.getD 0 default := by subst hb; rfl
    refine ⟨hm, by subst hb; rfl, ⟨s, by rw [ht], hl⟩, build_rows hb, build_layers hb, ?_⟩
    rw [root_of l s (by rw [hm, ht]; exact hz) hl]
    exact build_root hb

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

/-- A table's level 0 in the loop: the session it was drawn for (`owner`), the session's table it was drawn for (`table`,
its index), and the table it went to, once that session's `Commit` carried its root. -/
structure Slot where
  l0 : Level0
  owner : Nat
  table : Nat
  bound : Option (Nat × String) := none
  deriving Inhabited

/-- **The loop's tables' level 0**: each drawn one (`slots`), and the sessions a level-0 request may name (`sessions`):
each live session's name, the hex of its coin tree's root (`nameOf`), its number and its count of tables. A slot lives
only as long as the session it was drawn for (`close`). -/
structure Registry where
  slots : Array Slot := #[]
  sessions : Array (String × Nat) := #[]
  tables : Array (Nat × Nat) := #[]
  deriving Inhabited

/-- A session's name: the hex of its coin tree's root, the first 64 bytes of the words of `Hello`'s answer `a`
(`Resp::Coins`: tag, `u32le` count, then the words), which the prover receives as it is. -/
def nameOf (a : ByteArray) : String := hexOf (a.extract 5 69)

/-- Session `sid`, named `name` once its `Hello`'s answer checked, with `n` tables. -/
def Registry.name (R : Registry) (name : String) (sid n : Nat) : Registry :=
  { R with sessions := R.sessions.push (name, sid), tables := R.tables.push (sid, n) }

/-- **A level 0 drawn for table `j` of the session named `name`**, kept for it; else why not: a name no live session has,
a table the session does not have, or a table the loop already drew a level 0 for in this session. One draw a table is
what makes the level 0 the session binds the loop's one draw, not one the prover picked among several. -/
def Registry.draw (R : Registry) (name : String) (j : Nat) (l : Level0) : Except String Registry :=
  match R.sessions.find? (·.1 == name) with
  | none => .error s!"a level-0 request for a session the loop is not serving (coin tree root {name})"
  | some (_, sid) =>
    let n := ((R.tables.find? (·.1 == sid)).map (·.2)).getD 0
    if j ≥ n then .error s!"a level-0 request for table {j} of a session with {n} tables"
    else if R.slots.any fun s => s.owner == sid && s.table == j then
      .error s!"a second level-0 request for table {j} of this session: the loop draws a table's level 0 once"
    else .ok { R with slots := R.slots.push { l0 := l, owner := sid, table := j } }

/-- Session `sid` ended: its name and every level 0 drawn for it, bound or not, are gone. -/
def Registry.close (R : Registry) (sid : Nat) : Registry :=
  { slots := R.slots.filter (·.owner != sid), sessions := R.sessions.filter (·.2 != sid),
    tables := R.tables.filter (·.1 != sid) }

/-- **Session `sid`'s `Commit`**, of the session's tables `tables` (each with its statement's `m`), carrying `roots`: the
root of table `i`, in table order, the `Level0` the loop drew for this session's table `i`, at its table's `m`, which the
session then owns; else why it does not leave. -/
def commit (R : Registry) (sid : Nat) (tables : Array (String × Nat)) (roots : List (String × ByteArray)) :
    Except String Registry := do
  if roots.map (·.1) != tables.toList.map (·.1) then
    throw s!"a Commit whose roots are of {roots.map (·.1)}, where the session's tables are {tables.toList.map (·.1)}"
  let mut S := R.slots
  for (i, ((t, r), (_, m))) in (List.range roots.length).zip (roots.zip tables.toList) do
    let ok := fun (s : Slot) => s.l0.root == r && s.owner == sid && s.table == i &&
      (s.bound.isNone || s.bound == some (sid, t))
    match S.findIdx? ok with
    | none => throw s!"a Commit whose root of {t} is not the level 0 the loop drew for this session's table {i}"
    | some k =>
      let s := S[k]!
      if s.l0.m != m then throw s!"a Commit whose root of {t} was drawn at m = {s.l0.m}, where its statement's is {m}"
      S := S.set! k { s with bound := some (sid, t) }
  return { R with slots := S }

/-- Session `sid`'s table `t`'s level 0, once its `Commit` left. -/
def ofSession (R : Registry) (sid : Nat) (t : String) : Option Level0 :=
  (R.slots.find? (·.bound == some (sid, t))).map (·.l0)

end Flock.Firewall.Level0
