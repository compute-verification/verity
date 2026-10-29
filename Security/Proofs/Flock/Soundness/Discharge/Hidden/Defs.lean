import Proofs.Flock.Soundness.Discharge.Hm.Bc
import Proofs.Flock.Soundness.Discharge.Layout.Drawn

/-!
# A hidden-output statement (`Stmt.setupHidden`): what it is, and its frozen statements

With hidden outputs (`Tags.hiddenOutputs`, PROTOCOL.md §16.13) flock-verify sets a table up with `Stmt.setupHidden`,
`Stmt.setupH`'s checks on `HmOut`'s rows: the input rows, then one output row per output port. The public file holds no
output word; every row's `b ‖ c` is a region; Δ is `HmRow.delta`'s, then each output row's message bits copied from
the output bits the circuit computes (`outPairs`), zero past the port's words, then each pinned output bit's copy of the
forced-zero row (`pinList`; the pinned outputs, META `pinned_outputs`, have no row).

The definitions here say what that statement is: `Spec` (what `setupHidden` checked and produced), `outPairs` (Δ's pairs
past `HmRow.delta`'s), `pinList`, `outBytes` (an output row's message: the output words a template instance computes, zero-padded,
u16 little-endian) and `commitPair` (the extractor for the row commitment `b ‖ c`). The `Prop`s below are the frozen
statements `Theorems.lean` proves.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

/-! ## Δ's output pairs -/

/-- Output port `j`'s first word among a template instance's output words (`HmOut.delta`'s `at_`). -/
def outAt (c : Circuit) (j : ℕ) : ℕ := ((c.outPorts.toList.take j).map (·.2)).sum

/-- **The column `HmOut.delta` copies into output row `j`'s word `k`, bit `t`, at template instance `g`**: bit `t` of
the template instance's output word `at_ + k` where the circuit computes it (`HmOut.outCol`), `at_` the first output
word of the row's run of output ports (`HmOut.outGroup`), below the run's words; the unit range's first slot's
forced-zero row past them. -/
def outSrc (c : Circuit) (g j k t : ℕ) : ℕ :=
  if k < (HmOut.outGroup c j).2.2.2 then HmOut.outCol c g ((HmOut.outGroup c j).2.2.1 + k) + t else Hm.zeroCol c

/-- Row `p`'s message bit `t` of byte `byte` at template instance `g`, as `HmOut.delta` writes it: in the row's chain's
`sha512x3` slot for the byte's compression (`HmRow.byteAt`; block `byte / 128` of an unsegmented row), or, past the
row's compressions, in its `hm96` slot's `pad` (an exact-length row's last block). -/
def deltaMsgCol (c : Circuit) (g p byte t : ℕ) : ℕ :=
  let ko := HmRow.byteAt (c.ports.getD p default) byte
  if ko.1 < ((HmRow.rowSlots c.ports).2[p]!).size then
    let sl := ((HmRow.rowSlots c.ports).2[p]!)[ko.1]!
    c.slotCol c.shaNet g sl.1 + (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * sl.2 + 1]!.1 +
      64 * (ko.2 / 8) + 56 - 8 * (ko.2 % 8) + t
  else c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.1 +
      64 * (ko.2 / 8) + 56 - 8 * (ko.2 % 8) + t

/-- **Δ's pairs past `HmRow.delta`'s** (`HmOut.delta`): per template instance `g`, output row `j` (one per run of output
ports, `outRowPorts`), row word `k` and bit `t < 16`, the message column `i` of row `nIn + j`'s byte `2k + t / 8`, bit
`t % 8`, gets `(i, i)` and `(i, outSrc c g j k t)`. -/
def outPairs (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range c.outRowPorts.size).flatMap fun j =>
    (List.range (c.ports.getD (HmOut.nIn c + j) default).words).flatMap fun k =>
      (List.range 16).flatMap fun t =>
        [(deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8),
            deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)),
          (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)]

/-- **Δ's pairs on the pinned outputs** (`HmOut.pinPairs`): per template instance `g`, pinned output word `o` (from
`HmOut.pinnedFrom`) and bit `t < 16`, its column `i` gets `(i, i)` and `(i, Hm.zeroCol c)`. -/
def pinList (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmOut.pinnedWords c)).flatMap fun o =>
    (List.range 16).flatMap fun t =>
      [(HmOut.outCol c g (HmOut.pinnedFrom c + o) + t, HmOut.outCol c g (HmOut.pinnedFrom c + o) + t),
        (HmOut.outCol c g (HmOut.pinnedFrom c + o) + t, Hm.zeroCol c)]

/-! ## An output row's message -/

/-- **Output row `j`'s message at template instance `g`**, under the bits `w`: the row's `2 · words` bytes, byte `b` the
low (`b` even) or high byte of the template instance's output word `outAt c j + b / 2` as the circuit computes it
(`HmOut.outCol`), u16 little-endian, below the port's words, and zero past them. -/
def outBytes (w : ℕ → Bool) (c : Circuit) (g j : ℕ) : List UInt8 :=
  List.ofFn fun b : Fin (2 * (c.ports.getD (HmOut.nIn c + j) default).words) =>
    if b.val / 2 < (c.outPorts.getD j default).2 then
      Binding.byteOf fun t => w (HmOut.outCol c g (outAt c j + b.val / 2) + 8 * (b.val % 2) + t.val)
    else 0

/-! ## The row commitment's extractor -/

/-- **The extractor for the row commitment `b ‖ c`** (`Binding.commitString`), on two rows' bytes and salts: the salt
digest's inputs when the salts differ, else the rows' bytes. -/
def commitPair (r y r' y' : List UInt8) : List UInt8 × List UInt8 :=
  if y = y' then (r, r') else (Refine.K512.saltPrefix ++ y, Refine.K512.saltPrefix ++ y')

/-! ## What `setupHidden` checked and produced -/

/-- **What `Stmt.setupHidden` accepted**: the circuit `HmOut.parse` accepted, the leaf scheme it names, the public file
`HmOut.loadPublic` loaded (at a draw, `HmOut.drawn` of it) and `HmOut.checkPublic` passed, `n`, the block counts and `m`
as the executable computes them (`k_log ≤ 27`, `m ≤ 35`), the pin `HmRow.pin` gave, Δ from `HmOut.delta`, the regions
`HmOut.regions` built, which passed `checkInRange`, Σ over the circuit, the statement's
digest and the public file, and the hello `helloOf` built. -/
structure Spec (tags : Tags) (cf pf : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Lean.Json) (st : Stmt) : Prop where
  tagsEq : st.tags = tags
  leaf : st.leaf = default
  parse : HmOut.parse tags cf tables = .ok st.c
  merkle : MerkleScheme.find? st.c.leafScheme = some st.merkle
  pub : ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ (draw = none → st.pub = pop) ∧
    ∀ d, draw = some d → HmOut.drawn st.c pop d = .ok st.pub
  checkPublic : HmOut.checkPublic st.c st.pub = .ok ()
  n : st.n = st.pub.n
  blocksReal : st.blocksReal = (st.n + st.c.g - 1) / st.c.g
  nbl : st.nbl = max 3 (log2ceil (max st.blocksReal 1))
  m : st.m = st.c.kLog + st.nbl
  kLog : st.c.kLog ≤ 27
  m35 : st.m ≤ 35
  mPts : st.mPts = PT_LOCAL + st.nbl
  pin : HmRow.pin st.c = .ok st.pin
  da : st.da = (HmOut.delta st.c st.pin).1
  db : st.db = (HmOut.delta st.c st.pin).2
  regions : HmOut.regions st.c st.pub = .ok st.regions
  inRange : checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok ()
  sigma : st.sigma = Sha512.hash (tags.sigmaTag.toUTF8 ++ st.c.sha ++ st.digest ++ st.pub.sha)
  hello : helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello

/-! ## The frozen statements -/

/-- **`SetupHiddenSpec`**: every statement `Stmt.setupHidden` accepts is its `Spec`. -/
def SetupHiddenSpec : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st → Spec tags cf pf coins tables draw st

/-- **`HiddenDelta`**: `HmOut.delta` is `HmRow.delta`, then `outPairs`, then `pinList`, in Δ_A and in Δ_B. -/
def HiddenDelta : Prop :=
  ∀ (c : Circuit) (pin : ℕ),
    HmOut.delta c pin = ((HmRow.delta c pin).1 ++ (outPairs c).toArray ++ (pinList c).toArray,
      (HmRow.delta c pin).2 ++ (outPairs c).toArray ++ (pinList c).toArray)

/-- **`HiddenLayout`**: an untyped statement `Stmt.setupHidden` accepts that pins no output (META `pinned_outputs`) has
the layout the placement reads (`FlockSoundness.Layout`), so its block R1CS is `Layout.model`. -/
def HiddenLayout : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st →
      HmRow.pinnedCount st.c.metaJ = 0 → Layout st

/-- **`PublicExposes`**: what a statement `Stmt.setupHidden` accepts takes from the public file: no output word, and one
region per row, region `p` ("digest p") every template instance's `b ‖ c` of row `p` (`Layout.hmBytes`: an instance's
public `b ‖ c`, the pinned dummy past the population) at the row's `hm96` slot output (`Layout.hmFixed`,
`Layout.hmFree`). -/
def PublicExposes : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st →
    st.pub.outputs = #[] ∧ st.regions.size = st.c.ports.size ∧
      ∀ p < st.c.ports.size, ∃ r, st.regions[p]? = some r ∧
        mkRegion st.c.kLog s!"digest {p}" (Layout.hmFree st.c) (Layout.hmFixed st.c p) (Layout.hmBytes st.c st.pub p) =
          .ok r

/-- **`OutputsCommitted`** (the protocol meaning): in a drawn statement `Stmt.setupHidden` accepts, in every satisfying
witness of its block R1CS (`Layout.model`) at regions containing the executable's, at block `o` and template instance
`g` (below `2^⌊log₂ g⌋`), output row `j`'s `b ‖ c` in the registered population's public file, at the unit the template
instance proves, is the commit string of the row prefix and the output words the template instance computes
(`outBytes`), under the salt the witness holds at the row's `hm96` slot (`Hm.saltCol`). For a population file without
shared rows, a draw whose proved units are in the population, a unit that returns the outputs (`ho`) and `sha512/row/v1`
rows (`hv1`). -/
def OutputsCommitted : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    st.c.outNet = st.c.unitNet →
    (∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
          outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

/-- **`CommitPairSpec`**: the row commitment's extractor outputs a SHA-512 collision whenever two different rows'
bytes have one commit string `b ‖ c`, under any salts. -/
def CommitPairSpec : Prop :=
  ∀ r y r' y' : List UInt8, r ≠ r' → Binding.commitString r y = Binding.commitString r' y' →
    Binding.Collides (commitPair r y r' y')

/-- **`OutputsBinding`** (the protocol meaning, under `cr/sha-512`): two drawn statements `Stmt.setupHidden` accepts on
one circuit file, public file and tables, under `OutputsCommitted`'s conditions each, and satisfying witnesses of their
block R1CS, that put one registered unit at block `o`, template instance `g` and at block `o'`, template instance `g'`:
when the output words the two template instances compute for output row `j` differ, `commitPair` of their rows' bytes
and salts is a SHA-512 collision. -/
def OutputsBinding : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D D' : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitDraw.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    st.c.outNet = st.c.unitNet →
    (∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.blockBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
        outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.rowPrefix (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words).data.toList ++
        outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.blockBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

end FlockSoundness.Discharge.Hidden
