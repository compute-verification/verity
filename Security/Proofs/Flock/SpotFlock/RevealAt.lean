module

public import Proofs.Flock.SpotFlock.FrameIsAt
public import Proofs.Flock.SpotFlock.LayoutAt

/-!
# The base frame: a reveal that opens each gate's row at its site

`revealCode h site c₀` is a frame for `frameAt`: `c₀` but for its reveal, its own ports and how it reads an opening.
Gate `g`'s site (`site g`, a `Site`) names the port whose row holds its value, the value's place in that row and the
frame's slot for the row, all fixed before the run. Nothing here reads `h.frame`.

* **An opening's data** (`dataOf`): the row's bits, its salt and its path, each escaped (`esc`) so that the three
  read back (`unesc_esc`). `bitsOf`, `saltOf` and `pathOf` are the frame's `bits`, `salt` and `π`.
* **The unit's instance** (`rankAt`): a checked unit's instance in its session is the first unit of the session's
  sample that the registration maps to it (`RegTwo.blockAt`, over the replay blocks the interior record lists).
  `layoutFor h` is `layoutAt` at that rank and `pickAny`.
* **The opening** (`openingOf`, `openAt`): at gate `g` of unit `u`, the session witness's bit at the unit's block and
  the layout's column (`bitAt`), under the replay block's roots where `g` is a gate of the replay block it doesn't
  commit and the run's roots otherwise (`rootsAt`). Its data are row `q` of the unit's instance in the session's
  first table (`rowBitsAt`), that row's salt and its path (`pathAt`): the session header's path for the registered
  read named like the port where there is one, else the path in the port's published tree (`pubPath`). The reveal
  (`revealAt`) opens every gate the unit's proof block could need (`needsOf`) that has a site.
* **The opening facts**, at a checked unit of an accepted run (`CheckedRun`, `CheckedRun.Ok`): the reveal holds gate
  `g`'s opening (`open_mem`), under `FrameIs`'s roots (`roots_opening`); its row is the one the row hash reads
  (`row_opening`) where the site's port is an unsegmented bit row of its place's width whose bit at the place is the
  one the layout reads (`RowAt`); its salt is that row's (`salt_opening`) and its path `pathAt`'s (`path_opening`).

The frame's own ports are one port of `2^64` leaves (`revealOwn`), so its depth is 64 (`depth_revealOwn`): round 2
reads no port of more leaves.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

/-! ## An opening's data -/

/-- **A byte string escaped**: each byte behind a `1`, then a `0`. -/
def esc : List UInt8 → List UInt8
  | [] => [0]
  | b :: bs => 1 :: b :: esc bs

/-- **An escaped byte string read back**, and the bytes after it. -/
def unesc : List UInt8 → List UInt8 × List UInt8
  | x :: b :: bs => if x = 1 then (b :: (unesc bs).1, (unesc bs).2) else ([], b :: bs)
  | _ => ([], [])

theorem unesc_esc : ∀ a r : List UInt8, unesc (esc a ++ r) = (a, r)
  | [], [] => rfl
  | [], b :: r => by
    show unesc (0 :: b :: r) = ([], b :: r)
    rw [unesc.eq_1, if_neg (show ¬ ((0 : UInt8) = 1) by decide)]
  | b :: a, r => by
    show unesc (1 :: b :: (esc a ++ r)) = (b :: a, r)
    rw [unesc.eq_1, if_pos rfl, unesc_esc a r]

/-- **An opening's data**: its row's bits, a byte each, then its salt, then its path. -/
def dataOf (row : List Bool) (salt path : Msg) : Msg :=
  esc (row.map fun b => if b then 1 else 0) ++ (esc salt ++ path)

/-- **The row bits an opening's data hold** (`dataOf`'s first part). -/
def bitsOf (d : Msg) (i : ℕ) : Bool := (unesc d).1.getD i 0 == 1

/-- **An opening's salt** (`dataOf`'s second part). -/
def saltOf (o : Opening Bool) : Msg := (unesc (unesc o.data).2).1

/-- **An opening's path** (`dataOf`'s rest). -/
def pathOf (o : Opening Bool) : Msg := (unesc (unesc o.data).2).2

theorem bitsOf_dataOf (row : List Bool) (salt path : Msg) (i : ℕ) :
    bitsOf (dataOf row salt path) i = row.getD i false := by
  simp only [bitsOf, dataOf, unesc_esc, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases row[i]? with
  | none => rfl
  | some b => cases b <;> rfl

theorem saltOf_dataOf (o : Opening Bool) (row : List Bool) (salt path : Msg) (h : o.data = dataOf row salt path) :
    saltOf o = salt := by
  simp only [saltOf, h, dataOf, unesc_esc]

theorem pathOf_dataOf (o : Opening Bool) (row : List Bool) (salt path : Msg) (h : o.data = dataOf row salt path) :
    pathOf o = path := by
  simp only [pathOf, h, dataOf, unesc_esc]

/-- **A deployed opening's row is a list of bits** with the list's length at its place, the opened value at the place
and the data's bits elsewhere. -/
theorem rowBits_eq_of {place : ℕ → (w : ℕ) × Fin w} {bits : Msg → ℕ → Bool} {o : Opening Bool} {l : List Bool}
    (hl : (place o.pos).1 = l.length) (hv : l.getD (place o.pos).2 false = o.val)
    (hb : ∀ i, i ≠ ((place o.pos).2 : ℕ) → bits o.data i = l.getD i false) : rowBits place bits o = l := by
  apply List.ext_getElem (by rw [rowBits, List.length_ofFn, hl])
  intro i h1 h2
  have e : l[i] = l.getD i false := by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2, Option.getD_some]
  simp only [rowBits, List.getElem_ofFn]
  rw [e]
  split_ifs with hi
  · rw [← hv, ← hi]
  · exact hb i fun e => hi (Fin.ext e)

/-- **`sha512/row/v2`'s bytes of `8 N` bits read at `a`** are the `N` bytes read at `a` (`Hm.bytesOf`): bit `8 j + t` is
bit `t` of byte `j` in both. -/
theorem packBits_ofFn (f : ℕ → Bool) (N : ℕ) (a : ℕ → ℕ → ℕ) :
    _root_.Flock.HmRow.packBits (List.ofFn fun i : Fin (8 * N) => f (a (i.val / 8) (i.val % 8))) =
      FlockSoundness.Discharge.Hm.bytesOf f N a := by
  apply List.ext_getElem
  · simp only [_root_.Flock.HmRow.packBits, List.length_map, List.length_range, List.length_ofFn,
      FlockSoundness.Discharge.Hm.length_bytesOf]
    omega
  · intro j h1 h2
    have hj : j < N := by rwa [FlockSoundness.Discharge.Hm.length_bytesOf] at h2
    simp only [_root_.Flock.HmRow.packBits, List.getElem_map, List.getElem_range,
      FlockSoundness.Discharge.Hm.bytesOf, List.getElem_ofFn, FlockSoundness.Binding.byteOf]
    rw [SamplingDeploy.bitsVal_eq_sum]
    have hl : ((List.drop (8 * j) (List.ofFn fun i : Fin (8 * N) => f (a (i.val / 8) (i.val % 8)))).take 8).length =
        8 := by
      simp only [List.length_take, List.length_drop, List.length_ofFn]
      omega
    show UInt8.ofNat _ = UInt8.ofNat _
    congr 1
    rw [← Fin.sum_congr' _ hl]
    refine Finset.sum_congr rfl fun t _ => ?_
    have e1 : (8 * j + t.val) / 8 = j := by omega
    have e2 : (8 * j + t.val) % 8 = t.val := by omega
    simp only [Fin.val_cast, List.getElem_take, List.getElem_drop, List.getElem_ofFn, e1, e2]

/-! ## The frame's own ports -/

/-- **The frame's own ports**: one port of `2^64` leaves, the most round 2 reads. -/
def revealOwn : Array _root_.Flock.Registered.Port :=
  #[{ (default : _root_.Flock.Registered.Port) with leaves := 2 ^ 64 }]

theorem depth_revealOwn : depth revealOwn = 64 := by
  have h1 : Nat.log2 (2 ^ 64 - 1) = 63 := (Nat.log2_eq_iff (by norm_num)).2 ⟨by norm_num, by norm_num⟩
  have h2 : ¬ ((2 : ℕ) ^ 64 ≤ 1) := by norm_num
  show max (_root_.Flock.log2ceil (2 ^ 64)) 0 = 64
  rw [_root_.Flock.log2ceil, if_neg h2, h1]
  omega

/-! ## The unit's instance -/

/-- **The units a session's sample proves**, in its order (`unitOf`'s); none where it has no sample. -/
def drawnOf (draw : Option Json) : List ℕ :=
  match draw with
  | some d =>
    match _root_.Flock.Draw.UnitDraw.ofJson d with
    | .ok D => D.proved.toList
    | .error _ => []
  | none => []

variable (h : Held)

/-- **Unit `u`'s instance in its session statement `xS`**: the first unit of the session's sample (`drawnOf`) that the
registration maps to `u` (`RegTwo.blockAt`), its template's cells starting at `cellOff` over the replay blocks the
interior record `m₁` lists (`blocksOf`). It reads neither the run's statement nor its values. -/
def rankAt (_ : Msg) (m₁ xS : Msg) (u : ℕ) : ℕ :=
  let sample := (((jsonMsg m₁).map blocksOf).getD []).toArray
  let lo := Properties.Spot.cellOff sample.size h.reg.ns (templateOf h.ms u)
  (drawnOf ((FlockDeploy.readOf true xS).bind (·.draw))).findIdx fun p => h.reg.blockAt sample (lo + p) == u

/-- **The base frame's layout**: `layoutAt` at each unit's instance in its session (`rankAt`) and `pickAny`. -/
noncomputable abbrev layoutFor : Layout := layoutAt h (rankAt h) (pickAny h)

/-! ## The opening -/

/-- **Where gate `g`'s value sits**, fixed before the run: the port whose row holds it, its place in that row (the
row's bits and the value's bit), and the frame's slot for the row. -/
structure Site where
  port : ℕ
  place : (w : ℕ) × Fin w
  slot : Slot

/-- **Gate `g`'s place**: its site's; a one-bit row where it has none. -/
def placeOf (site : ℕ → Option Site) (g : ℕ) : (w : ℕ) × Fin w := ((site g).map Site.place).getD ⟨1, 0⟩

/-- **Row `q` of instance `n` of a session's first table, as bits**: bit `8 i + t` is bit `t` of the row's byte `i` as
the row hash reads it (`rowBytes`, after its prefix). -/
def rowBitsAt (p : _root_.Flock.Readable.Parsed) (w : Claims.Flock.Wit) (n q : ℕ) : List Bool :=
  List.ofFn fun i : Fin (8 * (2 * (p.sts[0]!.c.ports[q]!).words)) =>
    _root_.Flock.InnerRows.bitAt p w 0 (n / p.sts[0]!.c.g)
      (FlockSoundness.Typed.msgCol p.sts[0]!.c (n % p.sts[0]!.c.g) q (i.val / 8) (i.val % 8))

/-- **The registered read named like port `q`** of the session's first table, the first. -/
def regNamed (ins : FlockDeploy.Ins) (sts : Array _root_.Flock.Stmt) (q : ℕ) :
    Option _root_.Flock.Registered.Port :=
  ins.registered.bind fun own => own.find? fun reg => reg.name == (sts[0]!.c.ports[q]!).name

/-- **Row `q` of instance `n`'s path**: the session header's path for it (`Registered.entry`) where a registered read
is named like the port, else its path in the port's published tree (`pubPath`). -/
noncomputable def pathAt (ins : FlockDeploy.Ins) (sts : Array _root_.Flock.Stmt) (n q : ℕ) : ByteArray :=
  match regNamed ins sts q with
  | some reg =>
    match sts[0]!.pub.tables, _root_.Flock.Registered.entry sts[0]!.pub reg with
    | some t, .ok (_, paths) => paths.getD (_root_.Flock.Registered.ref t sts[0]!.c.ports.size q n) ByteArray.empty
    | _, _ => ByteArray.empty
  | none =>
    match _root_.Flock.HmOut.loadPublic ins.read.tags sts[0]!.c ins.read.I.publicFile with
    | .ok pop => pubPath sts[0]!.c pop q (rowAt sts[0]!.c pop q (unitOf ins.read.draw n))
    | .error _ => ByteArray.empty

/-- **The roots gate `g`'s opening for unit `u` is under** (`FrameIs`'s): the replay block's interior roots
(`iroots`) where `g` is a gate of `u`'s replay block that the block doesn't commit, the run's roots otherwise. -/
noncomputable def rootsAt (s : AuditStatement) (v : Seen) (u g : ℕ) : List Msg :=
  match partition h s, two h s with
  | some Pt, some d =>
    match replayOf Pt d.part u with
    | some ρ =>
      match d.part.blocks[ρ]? with
      | some rb => if g ∈ rb ∧ s.prog.Boundary rb g = false then iroots [] v ρ else s.roots
      | none => s.roots
    | none => s.roots
  | _, _ => s.roots

/-- **The gates a proof block `b` could need** (`Prog.Needs`): its own, then those its gates read. -/
def needsOf (C : Prog) (b : List ℕ) : List ℕ :=
  b ++ b.flatMap fun g' => (C.gates[g']?.map BGate.reads).getD []

theorem mem_needsOf {C : Prog} {b : List ℕ} {g : ℕ} (hg : C.Needs b g) : g ∈ needsOf C b := by
  rcases hg with ⟨hg, -⟩ | ⟨⟨g', hg', x, hx, hr⟩, -, -⟩
  · exact List.mem_append_left _ hg
  · exact List.mem_append_right _ (List.mem_flatMap.2 ⟨g', hg', by simpa [hx] using hr⟩)

/-- **Gate `g`'s opening at site `S` for unit `u`**, at the round-1 and round-2 messages `m₁` and `m₂` and the unit's
session's setup (the module's docstring). -/
noncomputable def openingOf (s : AuditStatement) (v : Seen) (wF : Claims.Flock.Wit) (u : ℕ) (m₁ m₂ : Msg)
    (ins : FlockDeploy.Ins) (sts : Array _root_.Flock.Stmt) (S : Site) (g : ℕ) : List Msg × Opening Bool :=
  let xS := unitStmt h m₂ u
  let n := rankAt h [] m₁ xS u
  (rootsAt h s v u g,
    { pos := g
      val := _root_.Flock.InnerRows.bitAt { ins.read with sts } wF 0 (((layoutFor h).blockOf [] m₁ xS u).getD 0)
        ((layoutFor h).wire [] m₁ xS u g)
      data := dataOf (rowBitsAt { ins.read with sts } wF n S.port)
        (_root_.Flock.InnerRows.salt { ins.read with sts } wF 0 n S.port).data.toList
        (pathAt ins sts n S.port).data.toList })

/-- **Gate `g`'s opening for unit `u`** (`openingOf`); none where `g` has no site or the unit's session statement
doesn't set up. -/
noncomputable def openAt (site : ℕ → Option Site) (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit)
    (m₁ m₂ : Msg) (g : ℕ) : Option (List Msg × Opening Bool) :=
  match site g, FlockDeploy.setupOf true (unitStmt h m₂ u) with
  | some S, some (ins, _, sts) => some (openingOf h s v wF u m₁ m₂ ins sts S g)
  | _, _ => none

/-- **The reveal**: at a transcript whose rounds 1 and 2 sent `m₁` and `m₂`, each gate unit `u`'s proof block could need
(`needsOf`) that has a site, opened (`openAt`). -/
noncomputable def revealAt (site : ℕ → Option Site) (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    List (List Msg × Opening Bool) :=
  match v.msgs with
  | _ :: (m₁, _) :: (m₂, _) :: _ =>
    (needsOf s.prog (((partition h s).bind (·.blocks[u]?)).getD [])).filterMap (openAt h site s u v wF m₁ m₂)
  | _ => []

/-- **The base frame**: `c₀` with the reveal `revealAt`, the own ports `revealOwn`, each gate's place its site's, and
an opening's bits, salt and path read off its data. -/
noncomputable def revealCode (site : ℕ → Option Site) (c₀ : Code) : Code :=
  { c₀ with
    reveal := revealAt h site
    own := revealOwn
    ownAt := fun _ _ => revealOwn
    ownAt_depth := fun _ _ => le_rfl
    place := placeOf site
    bits := bitsOf
    salt := saltOf
    π := pathOf }

theorem depth_revealCode (site : ℕ → Option Site) (c₀ : Code) : depth (revealCode h site c₀).own = 64 :=
  depth_revealOwn

/-! ## The opening facts -/

/-- **The proof blocks at `x`**: `units (base h) x`, read without the frame. -/
def unitsAt (x : Msg) : ℕ :=
  match parse h x with
  | some s => ((partition h s).getD ⟨[]⟩).blocks.length
  | none => 0

theorem units_eq_unitsAt (x : Msg) : units (base h) x = unitsAt h x := rfl

/-- **A checked unit of a run, with its session's setup**: the values `SlotsAt` quantifies over. The run's statement
`x` and its parse `s`, its first three messages and the coins of rounds 0 and 1, the rest of its transcript, the
session witness `wF`, the unit `u`, its proof block `b`, its replay block `ρ` with gates `rb`, and its session's
setup. -/
structure CheckedRun where
  x : Msg
  s : AuditStatement
  m₀ : Msg
  m₁ : Msg
  m₂ : Msg
  c₀ : FlockDeploy.Coin
  c₁ : FlockDeploy.Coin
  rest : List (Msg × Out)
  wF : Claims.Flock.Wit
  u : ℕ
  b : List ℕ
  ρ : ℕ
  rb : List ℕ
  ins : FlockDeploy.Ins
  sts : Array _root_.Flock.Stmt

namespace CheckedRun

variable (r : CheckedRun)

/-- **The run is accepted and `u` is one of its checked units**: `SlotsAt`'s premises. -/
def Ok : Prop :=
  ∃ Pt d spec, parse h r.x = some r.s ∧ partition h r.s = some Pt ∧ two h r.s = some d ∧ InScopeTwo Pt d r.s ∧
    stmtOk h r.x r.m₁ r.m₂ r.c₀ r.c₁ = true ∧ r.u ∈ h.reg.checked (unitsAt h r.x) (src r.c₀) (src r.c₁) ∧
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h r.m₂ r.u) r.wF = true ∧
    Pt.blocks[r.u]? = some r.b ∧ replayOf Pt d.part r.u = some r.ρ ∧ d.part.blocks[r.ρ]? = some r.rb ∧
    FlockDeploy.setupOf true (unitStmt h r.m₂ r.u) = some (r.ins, spec, r.sts)

/-- The unit's session statement. -/
abbrev xS : Msg := unitStmt h r.m₂ r.u

/-- The run's transcript. -/
abbrev seen : Seen := seenAt r.m₀ r.m₁ r.m₂ r.c₀ r.c₁ r.rest

/-- The session statement as the verifier parses it. -/
abbrev p : _root_.Flock.Readable.Parsed := { r.ins.read with sts := r.sts }

/-- The unit's block in its session. -/
noncomputable abbrev β : ℕ := ((layoutFor h).blockOf [] r.m₁ (r.xS h) r.u).getD 0

/-- The layout's column for gate `g` of the unit. -/
noncomputable abbrev wire (g : ℕ) : ℕ := (layoutFor h).wire [] r.m₁ (r.xS h) r.u g

/-- The unit's instance in its session. -/
abbrev n : ℕ := rankAt h [] r.m₁ (r.xS h) r.u

/-- **Gate `g`'s opening at site `S`** (`openingOf`). -/
noncomputable abbrev opening (S : Site) (g : ℕ) : List Msg × Opening Bool :=
  openingOf h r.s r.seen r.wF r.u r.m₁ r.m₂ r.ins r.sts S g

/-- **The site's row is the one the row hash reads**, at gate `g`: the site's port of the session's first table is an
unsegmented bit row (`sha512/row/v2`) of `16 w` bits, its place's width, and the bit the row holds at the place, at the
unit's instance, is the one the layout reads. -/
def RowAt (S : Site) (g : ℕ) : Prop :=
  (r.sts[0]!.c.ports[S.port]!).v2 = true ∧ (r.sts[0]!.c.ports[S.port]!).segs = none ∧
    S.place.1 = 16 * (r.sts[0]!.c.ports[S.port]!).words ∧
    _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / r.sts[0]!.c.g)
      (FlockSoundness.Typed.msgCol r.sts[0]!.c (r.n h % r.sts[0]!.c.g) S.port (S.place.2 / 8) (S.place.2 % 8)) =
      _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.β h) (r.wire h g)

variable {h r}

theorem open_mem {site : ℕ → Option Site} {S : Site} {g : ℕ} (hr : r.Ok h) (hs : site g = some S)
    (hg : r.s.prog.Needs r.b g) : r.opening h S g ∈ revealAt h site r.s r.u r.seen r.wF := by
  obtain ⟨Pt, d, spec, -, hP, -, -, -, -, -, hb, -, -, hset⟩ := hr
  have hb' : ((partition h r.s).bind (·.blocks[r.u]?)).getD [] = r.b := by
    rw [hP]
    simp [hb]
  show r.opening h S g ∈ (needsOf r.s.prog (((partition h r.s).bind (·.blocks[r.u]?)).getD [])).filterMap
    (openAt h site r.s r.u r.seen r.wF r.m₁ r.m₂)
  rw [hb']
  refine List.mem_filterMap.2 ⟨g, mem_needsOf hg, ?_⟩
  simp only [openAt, hs, hset]

theorem roots_opening (hr : r.Ok h) (S : Site) (g : ℕ) :
    (r.opening h S g).1 = if g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false then iroots r.x r.seen r.ρ else r.s.roots := by
  obtain ⟨Pt, d, spec, -, hP, h2, -, -, -, -, -, hρ, hrb, -⟩ := hr
  show rootsAt h r.s r.seen r.u g = _
  simp only [rootsAt, hP, h2, hρ, hrb]
  rfl

theorem row_opening {site : ℕ → Option Site} {c₀ : Code} {S : Site} {g : ℕ} (hs : site g = some S)
    (hR : r.RowAt h S g) :
    row (revealCode h site c₀).place (revealCode h site c₀).bits (r.opening h S g).2 =
      _root_.Flock.InnerRows.rowBytes r.p r.wF 0 (r.n h) S.port := by
  obtain ⟨hv2, hseg, hw, hcopy⟩ := hR
  have hplace : (revealCode h site c₀).place g = S.place := by
    show ((site g).map Site.place).getD ⟨1, 0⟩ = S.place
    simp [hs]
  have hlen : (rowBitsAt r.p r.wF (r.n h) S.port).length = 16 * (r.sts[0]!.c.ports[S.port]!).words := by
    show (List.ofFn fun i : Fin (8 * (2 * (r.sts[0]!.c.ports[S.port]!).words)) => _).length = _
    rw [List.length_ofFn]
    omega
  have hlt : (S.place.2 : ℕ) < (rowBitsAt r.p r.wF (r.n h) S.port).length := by
    rw [hlen, ← hw]
    exact S.place.2.isLt
  have hrow : rowBits (revealCode h site c₀).place (revealCode h site c₀).bits (r.opening h S g).2 =
      rowBitsAt r.p r.wF (r.n h) S.port := by
    apply rowBits_eq_of
    · show ((revealCode h site c₀).place g).1 = _
      rw [hplace, hlen, hw]
    · show (rowBitsAt r.p r.wF (r.n h) S.port).getD ((revealCode h site c₀).place g).2 false =
        _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.β h) (r.wire h g)
      rw [hplace, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt, Option.getD_some, ← hcopy]
      simp only [rowBitsAt, List.getElem_ofFn]
    · intro i _
      exact bitsOf_dataOf _ _ _ i
  rw [row, hrow]
  show (_root_.Flock.HmRow.rowPrefixV2 (rowBitsAt r.p r.wF (r.n h) S.port).length).data.toList ++
      _root_.Flock.HmRow.packBits (rowBitsAt r.p r.wF (r.n h) S.port) =
    FlockSoundness.Discharge.Hidden.Any.rowMsg (r.sts[0]!.c.ports[S.port]!)
      (FlockSoundness.Discharge.Hm.bytesOf (_root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / r.sts[0]!.c.g))
        (2 * (r.sts[0]!.c.ports[S.port]!).words)
        (FlockSoundness.Typed.msgCol r.sts[0]!.c (r.n h % r.sts[0]!.c.g) S.port))
  rw [FlockSoundness.Discharge.Hidden.Any.rowMsg, hseg, hlen]
  simp only [Option.isSome_none, Bool.false_eq_true, FlockSoundness.Discharge.Hidden.rowPre, hv2,
    ↓reduceIte]
  congr 1
  exact packBits_ofFn (_root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / r.sts[0]!.c.g))
    (2 * (r.sts[0]!.c.ports[S.port]!).words) (FlockSoundness.Typed.msgCol r.sts[0]!.c (r.n h % r.sts[0]!.c.g) S.port)

theorem salt_opening (site : ℕ → Option Site) (c₀ : Code) (S : Site) (g : ℕ) :
    (revealCode h site c₀).salt (r.opening h S g).2 =
      (_root_.Flock.InnerRows.salt r.p r.wF 0 (r.n h) S.port).data.toList := by
  show saltOf (r.opening h S g).2 = _
  rw [saltOf_dataOf (r.opening h S g).2 _ _ _ rfl]

theorem path_opening (site : ℕ → Option Site) (c₀ : Code) (S : Site) (g : ℕ) :
    bytes ((revealCode h site c₀).π (r.opening h S g).2) = pathAt r.ins r.sts (r.n h) S.port := by
  show bytes (pathOf (r.opening h S g).2) = _
  rw [pathOf_dataOf (r.opening h S g).2 _ _ _ rfl]
  exact bytes_bs _

end CheckedRun

end SpotFlock
