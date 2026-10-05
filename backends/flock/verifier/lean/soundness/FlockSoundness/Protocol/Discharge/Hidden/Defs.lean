import FlockSoundness.Protocol.Discharge.Hm.Bc
import FlockSoundness.Protocol.Discharge.Layout.Drawn

/-!
# A hidden-output statement (`Stmt.setupHidden`): what it is, and its frozen statements

With hidden outputs (`Tags.hiddenOutputs`, PROTOCOL.md §16.13) flock-verify sets a table up with `Stmt.setupHidden`,
`Stmt.setupH`'s checks on `HmOut`'s rows: the input rows, then one output row per output port. The public file holds no
output word; every row's `b ‖ c` is a region; Δ is `HmRow.delta`'s, then each output row's message bits copied from
the output bits the circuit computes (`outPairs`), zero past the port's words.

The definitions here say what that statement is: `Spec` (what `setupHidden` checked and produced), `outPairs` (Δ's pairs
past `HmRow.delta`'s), `outBytes` (an output row's message: the output words a VU computes, zero-padded, u16
little-endian) and `commitPair` (the extractor for the row commitment `b ‖ c`). The `Prop`s below are the frozen
statements `Theorems.lean` proves.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

/-! ## Δ's output pairs -/

/-- Output port `j`'s first word among a VU's output words (`HmOut.delta`'s `at_`). -/
def outAt (c : Circuit) (j : ℕ) : ℕ := ((c.outPorts.toList.take j).map (·.2)).sum

/-- **The column `HmOut.delta` copies into output row `j`'s word `k`, bit `t`, at VU `g`**: bit `t` of the VU's output
word `at_ + k` where the circuit computes it (`HmOut.outCol`), `at_` the first output word of the row's run of output ports
(`HmOut.outGroup`), below the run's words; the unit range's first slot's forced-zero row past them. -/
def outSrc (c : Circuit) (g j k t : ℕ) : ℕ :=
  if k < (HmOut.outGroup c j).2.2.2 then HmOut.outCol c g ((HmOut.outGroup c j).2.2.1 + k) + t else Hm.zeroCol c

/-- Row `p`'s message bit `t` of byte `byte` at VU `g`, as `HmOut.delta` writes it: in the row's chain's `sha512x3`
slot for the byte's compression (`HmRow.byteAt`; block `byte / 128` of an unsegmented row), or, past the row's
compressions, in its `hm96` slot's `pad` (an exact-length row's last block). -/
def deltaMsgCol (c : Circuit) (g p byte t : ℕ) : ℕ :=
  let ko := HmRow.byteAt (c.ports.getD p default) byte
  if ko.1 < ((HmRow.rowSlots c.ports).2[p]!).size then
    let sl := ((HmRow.rowSlots c.ports).2[p]!)[ko.1]!
    c.slotCol c.shaNet g sl.1 + (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * sl.2 + 1]!.1 +
      64 * (ko.2 / 8) + 56 - 8 * (ko.2 % 8) + t
  else c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.1 +
      64 * (ko.2 / 8) + 56 - 8 * (ko.2 % 8) + t

/-- **Δ's pairs past `HmRow.delta`'s** (`HmOut.delta`): per VU `g`, output row `j` (one per run of output ports,
`outRowPorts`), row word `k` and bit `t < 16`, the message column `i` of row `nIn + j`'s byte `2k + t / 8`, bit `t % 8`,
gets `(i, i)` and `(i, outSrc c g j k t)`. -/
def outPairs (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range c.outRowPorts.size).flatMap fun j =>
    (List.range (c.ports.getD (HmOut.nIn c + j) default).words).flatMap fun k =>
      (List.range 16).flatMap fun t =>
        [(deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8),
            deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)),
          (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)]

/-! ## An output row's message -/

/-! ## The row commitment's extractor -/

/-! ## What `setupHidden` checked and produced -/

/-- **What `Stmt.setupHidden` accepted**: the circuit `HmOut.parse` accepted, the leaf scheme it names, the public file
`HmOut.loadPublic` loaded (at a draw, `HmOut.drawn` of it) and `HmOut.checkPublic` passed, `n`, the block counts and `m`
as the executable computes them (`k_log ≤ 27`, `m ≤ 35`), the pin `HmRow.pin` gave, Δ from `HmOut.delta`, the regions
`HmOut.regions` built, which passed `HmRow.checkRegionWords` and `checkInRange`, Σ over the circuit, the statement's
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
  regionWords : HmRow.checkRegionWords st.c st.da st.db st.regions = .ok ()
  inRange : checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok ()
  sigma : st.sigma = Sha512.hash (tags.sigmaTag.toUTF8 ++ st.c.sha ++ st.digest ++ st.pub.sha)
  hello : helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello

/-! ## The frozen statements -/

/-- **`SetupHiddenSpec`**: every statement `Stmt.setupHidden` accepts is its `Spec`. -/
def SetupHiddenSpec : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st → Spec tags cf pf coins tables draw st

/-- **`HiddenDelta`**: `HmOut.delta` is `HmRow.delta`, then `outPairs`, in Δ_A and in Δ_B. -/
def HiddenDelta : Prop :=
  ∀ (c : Circuit) (pin : ℕ),
    HmOut.delta c pin = ((HmRow.delta c pin).1 ++ (outPairs c).toArray, (HmRow.delta c pin).2 ++ (outPairs c).toArray)

end FlockSoundness.Discharge.Hidden
