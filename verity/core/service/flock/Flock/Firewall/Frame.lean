/-
# The outer frame (live PROTOCOL.md §10.1): a masked rep's randomness, its refusals, and what leaves

The outer firewall's release function, with the rep's prover randomness and its two refusals. At the rep's start the frame
draws from the operating system's bytes (`IO.getRandomBytes`) everything the rep's hiding rests on (`Draws`): the pads `h`
(with `h_ab = h_a h_b` and the triple's `w = x y`), the dot-product mask `μ`, each inner row's padding, `τ`'s salt and the
pads tree's leaf salts, and it commits them itself (`padsRoot`, the tree the inner proof opens). The prover proves with
exactly these draws, and the frame masks with them.

`round` takes one device round as the device frames it (transcript-v2, its values unmasked, ending in its squeeze) and the
rep's masking state, and returns the round that leaves for the verifier's stream with the state after it: every label,
message and squeeze as the device framed it, the pads' label and root right after the zerocheck's label, and each value
absorbed in a masked phase sent as `v + h_i` (GF(2^128) addition) under the next pad of the layout's absorbed order
(`absorbed`). The masked phases are the algebraic phase, from the zerocheck's label to the opening's, and, with a masked
Ligerito level 0, the values between `Zk.LABEL_OPEN` and `Zk.LABEL_CLOSE`. The domain frame is the stream's own and never
leaves.

The frame decides the rep's refusals from the verifier's coins alone (`Masker.answer`, the coins the stream answered each
round it released): a zerocheck outer coin `r_j = 1`; at the opening's label, a mask slot whose weights on the claim
points of the table's reps so far (`prior`'s, then this rep's, each from its coins: `claimPoints`) have less than full rank
(`maskRank`); level 0's `ρ_0, ρ_1` dependent over F128 (`Zk.rhoIndependent`, the coins after its last masked value); and
`β = 0` in the inner proof (its third round's coin, counted from `Zk.LABEL_INNER`). Each is a stop the public coins decide
(`STOP`): nothing of the round leaves. The inner proof is the firewall's own, and its rounds leave through the frame
unmasked, in the form the frame fixes (`innerOwn`): `(ρ, σ)` and `Y` are the frame's, from its draws and the coins, and of
its other two rounds only `final_c'` and `τ`'s commitment are the firewall's.

The frame makes the pass decision on each device round (`check`, `passes`): the firewall hands it the device round with
the statement's scheduled round in its place, the skeleton's round with every value zero in the algebraic phase and the
firewall's own opening round in the opening, and the device round passes when it is the scheduled round byte for byte but
for the values the algebraic phase masks. Rust keeps no round check of its own (live PROTOCOL.md §10.1): what fails a
rep outside the frame (its worker, the end check on its proof, Rust's masking cross-checks) can only fail a round more,
the rep's rounds then reaching the frame as no device round, which it fails.

Any other stop comes at `Finish`, never before (§10.1): once a round fails, or `round` refuses one, the rep is `stopped`.
The scheduled round then leaves in the device's place, and every value of the algebraic phase leaves as its pad alone
(`0 + h_i`), whatever value the round carries, so what leaves before `Finish` has the same form and distribution as an
honest rep's. A stopped rep's `finish` is `stop`.

`flock-firewall frame` serves this to the firewall (`cmd`), which releases only what it returns, and the session loop
serves it on connections of its own (`serveWith`, `Session.frames`), where the coins a frame takes are the ones the loop
opened (`Frames`). Each request is one pure step of the rep's state (`step`, folded by `run`); `serveWith` only reads the
request's bytes, runs the host's hooks and writes the answer. Rust's `zk_veil::Masking`
and `zk_veil::Pads` are its speed references: the firewall holds Rust's masked rounds to this release byte for byte, and
Rust's tree over these draws to `padsRoot`.
-/
import Flock.Zk
import Flock.ZkLigerito
import Flock.Firewall.Coins
import Flock.JsonParse

namespace Flock.Firewall.Frame

open Lean (Json)
open Transcript (header padded)

def LABEL_ZC : String := "flock-zerocheck-v0"
def LABEL_PCS : String := "flock-pcs-open-batch-v0"
/-- The univariate skip's evaluations per sent vector (`zk_veil::SKIP`). -/
def SKIP : Nat := 64
/-- The skip's coins, and the zerocheck's pinned inner coordinates (`zk_veil::K_SKIP`, `N_INNER`). -/
def K_SKIP : Nat := 6
def N_INNER : Nat := 7
/-- An `hm96-sha512/v1` leaf salt's bytes. -/
def SALT : Nat := 192
/-- The prefix of a stop the verifier's coins decide (`zk_veil`'s refusals). -/
def STOP : String := "PROVER-STOPPED: "
/-- Level 0's combination coins `ρ_0, ρ_1 ∈ F256`, as the stream answers them: four F128 words. -/
def RHO_COINS : Nat := 4
/-- The inner proof's rounds (`Zk.verifyInner`: `ε`, the batching coins, `β`, the queries), and `β`'s among them. -/
def INNER_ROUNDS : Nat := 4
def INNER_BETA : Nat := 2

/-- How many of the absorbed values belong to a masked Ligerito level 0 (none when `k0 = 0`: level 0 is not masked). -/
def nLig (l : Zk.PadLayout) : Nat := if l.k0 > 0 then l.nOod0 + 4 * l.k0 + 4 else 0

/-- The pads in the order the prover absorbs the values they mask (`zk_veil::Layout::absorbed`): the zerocheck's
`round1_ab`, `round1_c`, multilinear rounds, `final_a`, `final_b` (`final_c` is never absorbed), the lincheck's rounds and
`z_partial`, then level 0's OOD values, lane-round messages and the pair's sums. -/
def absorbed (l : Zk.PadLayout) : Array Nat := Id.run do
  let mut v : Array Nat := #[]
  for i in [0:SKIP] do v := v.push (l.r1ab i)
  for i in [0:SKIP] do v := v.push (l.r1c i)
  for j in [0:l.nMlv] do v := (v.push (l.ml j 0)).push (l.ml j 1)
  v := (v.push l.fa).push l.fb
  for j in [0:l.nLc] do v := (v.push (l.lc j 0)).push (l.lc j 1)
  for i in [0:SKIP] do v := v.push (l.zp i)
  if l.k0 > 0 then
    for i in [0:l.nOod0] do v := v.push (l.lgOod i)
    for j in [0:l.k0] do
      for t in [0:4] do v := v.push (l.lgMsg j t)
    v := v ++ #[l.lgE 0 0, l.lgE 0 1, l.lgE 1 0, l.lgE 1 1]
  return v

/-! ## The rep's draws -/

/-- The inner code's length for layout `l`, the pads tree's leaves (`zk_veil::Code::new`). -/
def codeLen (l : Zk.PadLayout) : Nat := (l.len + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE

/-- The OS bytes a rep's draws take: `h`, `μ` and the two rows' padding (16 bytes a word), `τ`'s salt, a salt per leaf. -/
def drawBytes (l : Zk.PadLayout) : Nat := 16 * (2 * l.len + 2 * Zk.K_PAD) + Zk.TAU_SALT_BYTES + SALT * codeLen l

/-- `n` words of `b` from byte `o`, 16 bytes each, low then high, little-endian. -/
def words (b : ByteArray) (o n : Nat) : Array F128 :=
  (Array.range n).map fun i => ⟨u64OfLE b (o + 16 * i), u64OfLE b (o + 16 * i + 8)⟩

def wordBytes (v : Array F128) : ByteArray := v.foldl F128.pushBytes .empty

/-- A rep's prover randomness (`zk_veil::Pads`, with its pads tree's leaf salts). -/
structure Draws where
  h : Array F128
  mu : Array F128
  padH : Array F128
  padMu : Array F128
  tauSalt : ByteArray
  salts : ByteArray
  deriving Inhabited

/-- The draws in the bytes `b` (`drawBytes l` of them): `h`, `μ`, the rows' padding, `τ`'s salt and the leaf salts, in that
order, with `h_ab := h_a h_b` and `w := x y` (the triple) set from the others. -/
def Draws.ofBytes (l : Zk.PadLayout) (b : ByteArray) : Draws :=
  let s := l.len
  let h := words b 0 s
  let h := h.set! l.hab (h[l.fa]! * h[l.fb]!)
  let h := h.set! l.tw (h[l.tx]! * h[l.ty]!)
  let o := 32 * s + 32 * Zk.K_PAD
  { h, mu := words b (16 * s) s, padH := words b (32 * s) Zk.K_PAD, padMu := words b (32 * s + 16 * Zk.K_PAD) Zk.K_PAD,
    tauSalt := b.extract o (o + Zk.TAU_SALT_BYTES),
    salts := b.extract (o + Zk.TAU_SALT_BYTES) (o + Zk.TAU_SALT_BYTES + SALT * codeLen l) }

/-- The draws in the bytes `b` one read returned, refused unless there are exactly `drawBytes l` of them: `Draws.ofBytes`
reads past a short input as zeros, which would be pads no one drew. -/
def Draws.ofRead (l : Zk.PadLayout) (b : ByteArray) : Except String Draws :=
  if b.size = drawBytes l then .ok (Draws.ofBytes l b)
  else .error s!"{b.size} bytes where the draws take {drawBytes l}"

/-- The rep's draws from a byte source, in one read of `drawBytes l` bytes from `get` (`cmd`'s is `IO.getRandomBytes`). -/
def drawsWith {m : Type → Type} [Monad m] (get : USize → m ByteArray) (l : Zk.PadLayout) : m (Except String Draws) :=
  Draws.ofRead l <$> get (drawBytes l).toUSize

/-- The root over leaves `l` (a power of two of them): each node `SHA-512(left ‖ right)`, a level at a time. -/
def treeRoot (l : Array ByteArray) : ByteArray := go l l.size
where
  go (l : Array ByteArray) : Nat → ByteArray
    | 0 => l[0]!
    | fuel + 1 =>
      if l.size ≤ 1 then l[0]!
      else go ((Array.range (l.size / 2)).map fun i => MerkleScheme.hm96Sha512.node l[2 * i]! l[2 * i + 1]!) fuel

/-- The pads' commitment (`zk_veil::InnerCommit`): leaf `j` is column `j` of the encoded rows `(h ‖ pad_h)` and
`(μ ‖ pad_μ)` (`Zk.codeAt`) under the `j`-th salt, as the inner proof opens it (`Zk.verifyInner`). -/
def padsRoot (l : Zk.PadLayout) (d : Draws) : ByteArray :=
  let rh := d.h ++ d.padH
  let rm := d.mu ++ d.padMu
  treeRoot ((Array.range (codeLen l)).map fun j =>
    Hm96.Default512.leaf (Sha512.hash ((Zk.codeAt rh j).toBytes ++ (Zk.codeAt rm j).toBytes)) (d.salts.extract (SALT * j) (SALT * j + SALT)))

/-- The draws as the start's answer carries them, then the pads' root. -/
def Draws.toBytes (d : Draws) (root : ByteArray) : ByteArray :=
  wordBytes d.h ++ wordBytes d.mu ++ wordBytes d.padH ++ wordBytes d.padMu ++ d.tauSalt ++ d.salts ++ root

/-! ## The refusals -/

/-- The two z-claims' word coordinates (`x[1..]` of each claim point the opening takes) from the algebraic phase's coins
alone, in the verifier's draw order (`Zk.replay`: the skip's, the outer ones, `z`, the multilinear rounds', `α`, `β`, the
lincheck's, `r'`); `zk_veil::points_from_coins` and `claim_word_points`. -/
def claimPoints (l : Zk.PadLayout) (c : Array F128) : V (Array (Array F128)) := do
  let nOuter := l.nMlv - N_INNER
  let need := K_SKIP + nOuter + 1 + l.nMlv + 2 + l.nLc + 1
  if l.nMlv < N_INNER || c.size != need then
    throw s!"{c.size} coins in the algebraic phase, not {need}"
  let lcAt := K_SKIP + nOuter + 1 + l.nMlv + 2
  let zc : ZcOut := { z := c[K_SKIP + nOuter]!, rho := c.extract (K_SKIP + nOuter + 1) (K_SKIP + nOuter + 1 + l.nMlv),
                      rRest := Flock.pinned ++ c.extract K_SKIP (K_SKIP + nOuter), vA := 0, vB := 0, vC := 0 }
  let lc : LcOut := { w := 0, rPrime := c[lcAt + l.nLc]!, rIn := (c.extract lcAt (lcAt + l.nLc)).reverse }
  return (abcClaims (l.nMlv + K_SKIP) (l.nLc + K_SKIP) zc lc).map fun cl => cl.x.extract 1 cl.x.size

/-- `eq(p, x)` at word `x`: `∏_i (p_i` if bit `i` of `x` is set, else `1 + p_i)` (`zk_veil::eq_word`). -/
def eqWord (p : Array F128) (x : Nat) : F128 := Id.run do
  let mut acc : F128 := 1
  for i in [0:p.size] do
    acc := acc * (if x.testBit i then p[i]! else 1 + p[i]!)
  return acc

/-- The index of `v`'s lowest set bit (`v ≠ 0`). -/
def lowBit (v : UInt64) : Nat := Nat.log2 (v &&& (0 - v)).toNat

/-- The mask slot (`flock-circuit`'s `mask_words`): `n` words from word `w0` of each of `blocks` blocks. -/
structure MaskSlot where
  w0 : Nat
  n : Nat
  blocks : Nat
  deriving Inhabited

/-- Word `i` of the slot, in a circuit of `2^kLog` coordinates a block (128 a word). -/
def MaskSlot.word (m : MaskSlot) (kLog i : Nat) : Nat := (i / m.n) * 2 ^ (kLog - 7) + m.w0 + i % m.n

/-- The GF(2)-rank of the rows `(eq(p_1, x), …, eq(p_k, x))` over the slot's words `x` (`zk_veil::mask_rank_ok`): each
row reduced into an echelon basis, its pivot its lowest set bit, until the basis holds all `128 k` columns. Each reduction
clears the row's lowest set bit and sets none below it, so `128 k` of them settle a row. -/
def maskRank (points : Array (Array F128)) (m : MaskSlot) (kLog : Nat) : Nat := Id.run do
  let limbs := 2 * points.size
  let full := 64 * limbs
  let mut basis : Array (Option (Array UInt64)) := Array.replicate full none
  let mut rank := 0
  for i in [0:m.blocks * m.n] do
    if rank == full then break
    let x := m.word kLog i
    let mut row : Array UInt64 := points.foldl (fun acc p => let e := eqWord p x; (acc.push e.lo).push e.hi) #[]
    let mut w := 0
    for _ in [0:full + 1] do
      match (List.range' w (limbs - w)).find? (fun k => row[k]! != 0) with
      | none => break
      | some k =>
        w := k
        let col := 64 * w + lowBit row[w]!
        match basis[col]! with
        | some b => row := (Array.range limbs).map fun t => if t ≥ w then row[t]! ^^^ b[t]! else row[t]!
        | none =>
          basis := basis.set! col (some row)
          rank := rank + 1
          break
  return rank

/-! ## The release -/

/-- A rep's masking configuration, fixed at its start: its layout, pads `h`, the absorbed order, the pads' root and the mask
slot. Every step takes it and returns only the rep's `Masker`, so no step changes it. -/
structure Cfg where
  lay : Zk.PadLayout
  h : Array F128
  order : Array Nat
  /-- how many of `order` the algebraic phase absorbs -/
  nPiop : Nat
  lig : Bool
  root : ByteArray
  /-- the mask slot the rank check at the opening's label reads (none: no check) -/
  mask : Option MaskSlot := none
  /-- the pads tree's two rows, `(h ‖ pad_h)` and `(μ ‖ pad_μ)`, which the inner proof's `Y` combines -/
  rowH : Array F128 := #[]
  rowMu : Array F128 := #[]
  deriving Inhabited

/-- The configuration of a rep of layout `l` with the draws `d` and the statement's mask slot: `d`'s pads and rows, and the
root the frame computes over `d` (`padsRoot`). -/
def Cfg.ofDraws (l : Zk.PadLayout) (d : Draws) (mask : Option MaskSlot) : Cfg :=
  { lay := l, h := d.h, order := absorbed l, nPiop := (absorbed l).size - nLig l, lig := l.k0 > 0,
    root := padsRoot l d, mask, rowH := d.h ++ d.padH, rowMu := d.mu ++ d.padMu }

/-- Whether a rep can start from `c`: a 64-byte root, and every pad of the order among the rep's pads. -/
def Cfg.check (c : Cfg) : V Unit := do
  if c.root.size != 64 then throw s!"the pads' root has {c.root.size} bytes, not 64"
  if let some i := c.order.find? (· ≥ c.h.size) then throw s!"pad {i} is outside the rep's {c.h.size} pads"

/-- A rep's masking state: where its stream is, and what its refusals read. It starts as `{}`. -/
structure Masker where
  /-- the algebraic phase's coins so far, as the stream answered them -/
  coins : Array F128 := #[]
  /-- the claim points of the table's earlier reps -/
  prior : Array (Array F128) := #[]
  /-- the coin count of the round last released, until the stream's coins for it come -/
  owed : Option Nat := none
  /-- the rank check's claim points, mask words and rank, once it ran -/
  rank : Option (Nat × Nat × Nat) := none
  /-- values absorbed masked so far -/
  pos : Nat := 0
  on : Bool := false
  done : Bool := false
  doneL0 : Bool := false
  l0 : Bool := false
  /-- level 0's coins after its last masked value, as the stream answered them: `ρ_0`'s words, then `ρ_1`'s -/
  rho : Array F128 := #[]
  /-- the inner proof's rounds answered so far, once its label came -/
  inner : Option Nat := none
  /-- the inner proof's `ε` and `β`, once the stream answered them -/
  eps : F128 := 0
  beta : F128 := 0
  /-- a round failed the firewall's check or was refused: the algebraic phase's values leave as their pads alone, and the
  rep ends in `stop` -/
  stopped : Bool := false
  deriving Inhabited

def labelFrame (b : ByteArray) : ByteArray := header 2 0 b.size ++ padded b

def bytesFrame (b : ByteArray) : ByteArray := header 5 0 b.size ++ padded b

def isLabel (b : ByteArray) (l : String) : Bool := b.data == l.toUTF8.data

/-- A label as the masking challenger takes it: the phase it ends, its frame, and the phase it starts (after the
zerocheck's label, the pads' label and root). At the opening's label, the rank check on the coins so far: a slot short of
full rank is a stop. -/
def Masker.label (c : Cfg) (s : Masker) (l : ByteArray) : V (Masker × ByteArray) := do
  let mut s := s
  if isLabel l LABEL_PCS && s.on then
    if s.pos != c.nPiop then
      throw s!"the opening began after {s.pos} of the algebraic phase's {c.nPiop} masked values"
    s := { s with on := false, done := true }
    if let some m := c.mask then
      let all := s.prior ++ (← claimPoints c.lay s.coins)
      let r := maskRank all m (c.lay.nLc + K_SKIP)
      s := { s with rank := some (all.size, m.blocks * m.n, r) }
      if r != 128 * all.size then
        throw s!"{STOP}degenerate coins: the mask slot's weights have rank {r} < {128 * all.size} over {all.size} claim points"
  if isLabel l Zk.LABEL_CLOSE && s.l0 then
    if s.pos != c.order.size then
      throw s!"level 0 closed after {s.pos} of the layout's {c.order.size} masked values"
    if s.rho.size != RHO_COINS then
      throw s!"level 0 closed after {s.rho.size} of its {RHO_COINS} ρ coins"
    s := { s with l0 := false, on := false, doneL0 := true }
  if isLabel l Zk.LABEL_INNER then
    if s.inner.isSome then throw "a second inner proof"
    if !s.done || s.on || (c.lig && !s.doneL0) then throw "the inner proof began before the opening ended"
    s := { s with inner := some 0 }
  let mut out := labelFrame l
  if isLabel l LABEL_ZC && !s.done then
    s := { s with on := true }
    out := out ++ labelFrame Zk.LABEL_PADS.toUTF8 ++ bytesFrame c.root
  if isLabel l Zk.LABEL_OPEN && s.done && c.lig && !s.doneL0 then
    s := { s with on := true, l0 := true }
  return (s, out)

/-- An absorbed value: `v + h_i` under the next pad of the order in a masked phase, `v` outside one. In a stopped rep the
algebraic phase's values are `0`, so each leaves as its pad `h_i`. -/
def Masker.absorb (c : Cfg) (s : Masker) (v : F128) : V (Masker × F128) := do
  if !s.on then return (s, v)
  let some i := c.order[s.pos]? | throw s!"more masked values than the layout's {c.order.size}"
  let some p := c.h[i]? | throw s!"pad {i} is outside the rep's pads"
  let v := if s.stopped && s.pos < c.nPiop then 0 else v
  return ({ s with pos := s.pos + 1 }, v + p)

/-- The coins `c` of the inner proof's round `k`: `β = 0` at its third is a stop (`zk_veil::prove_inner`,
`Zk.verifyInner`), at which the simulator would divide by `β`. -/
def innerCoins (k : Nat) (c : Array F128) : V Unit := do
  if k == INNER_BETA then
    if c.size != 1 then throw s!"the inner proof's β round squeezed {c.size} coins, not 1"
    if c[0]!.isZero then throw s!"{STOP}degenerate coin β = 0"

/-- Level 0's coins `c` after its last masked value, its `ρ` words so far being `r`: once all four are in, `ρ_0, ρ_1`
dependent over F128 is a stop (`zk::rho_independent`, `Zk.rhoIndependent`), at which `ρ_0 R_0 + ρ_1 R_1` is not uniform. -/
def rhoCoins (r c : Array F128) : V (Array F128) := do
  let r' := r ++ c
  if r.size < RHO_COINS && RHO_COINS ≤ r'.size && !Zk.rhoIndependent (⟨r'[0]!, r'[1]!⟩, ⟨r'[2]!, r'[3]!⟩) then
    throw s!"{STOP}degenerate coins: rho_0, rho_1 dependent over F128"
  return r'

/-- The coins `c` the stream answered the round last released. The algebraic phase's are kept (the claim points), and a
zerocheck outer coin `r_j = 1` among them is a stop (`zk_veil::Masking::refuse_outer`): at `r_j = 1` the replay's round `j`
divides by zero. Level 0's after its last masked value are `ρ`'s (`rhoCoins`), and the inner proof's are counted, its
first being `ε` and its third `β` (`innerCoins`), which its next round reads (`innerOwn`). -/
def Masker.answer (cfg : Cfg) (s : Masker) (c : Array F128) : V Masker := do
  let some n := s.owed | throw "coins for no round"
  if c.size != n then throw s!"{c.size} coins for a round that squeezed {n}"
  let s := { s with owed := none }
  if let some k := s.inner then
    innerCoins k c
    let s := if k == 0 then { s with eps := c[0]! } else if k == INNER_BETA then { s with beta := c[0]! } else s
    return { s with inner := some (k + 1) }
  if s.l0 then
    if s.pos != cfg.order.size then return s
    return { s with rho := ← rhoCoins s.rho c }
  if !s.on then return s
  let start := s.coins.size
  let s := { s with coins := s.coins ++ c }
  let nOuter := cfg.lay.nMlv - N_INNER
  for i in [start:s.coins.size] do
    if K_SKIP ≤ i && i < K_SKIP + nOuter && (s.coins[i]! + 1).isZero then
      throw s!"{STOP}degenerate coin: zerocheck outer coin r_{i - K_SKIP} = 1"
  return s

/-- Why the rep's stream cannot end here, if it cannot: a round's coins unanswered, a pad of the order unused, a masked
phase open, or the inner proof's rounds not all released. -/
def Masker.unfinished (c : Cfg) (s : Masker) : Option String :=
  if let some k := s.owed then some s!"the rep ended before the {k} coins of its last round"
  else if s.pos != c.order.size then some s!"the rep ended after {s.pos} of the layout's {c.order.size} masked values"
  else if !s.done || (c.lig && !s.doneL0) then some "the rep ended inside a masked phase"
  else if s.inner != some INNER_ROUNDS then
    some s!"the rep ended after {s.inner.getD 0} of the inner proof's {INNER_ROUNDS} rounds"
  else none

/-- The inner proof's round `k` (`zk_veil::prove_inner`, `Zk.verifyInner`), `content` squeezing `n` coins, as the frame
fixes it: every frame, kind and length, and every value the frame can compute from its draws and the coins. Round 0 is the
inner label, `final_c'` and `ε`'s squeeze; round 1 is `(ρ, σ) = (ε h_fa + h_tx, h_fb + h_ty)` and the batching coins;
round 2 is `τ`'s 64-byte commitment and `β`'s squeeze; round 3 is `Y = (h ‖ pad_h) + β (μ ‖ pad_μ)` and the queries. Only
`final_c'` and `τ`'s commitment are the firewall's: `final_c'` masks the device's `final_c` under `h_fc`, and
`τ = ⟨c, μ ‖ pad_μ⟩` reads the batched constraints the frame does not derive. -/
def innerOwn (c : Cfg) (s : Masker) (k : Nat) (content : ByteArray) (n : Nat) : V Unit := do
  let fixedBut := fun (pre : ByteArray) (free : Nat) (post : ByteArray) =>
    content.size == pre.size + free + post.size && content.extract 0 pre.size == pre
      && content.extract (pre.size + free) content.size == post
  match k with
  | 0 =>
    if !fixedBut (labelFrame Zk.LABEL_INNER.toUTF8 ++ header 3 1 1) 16 (header 4 1 1) then
      throw "the inner proof's round 0 is not its label, final_c' and ε's squeeze"
  | 1 =>
    let l := c.lay
    let rs := #[s.eps * c.h[l.fa]! + c.h[l.tx]!, c.h[l.fb]! + c.h[l.ty]!]
    if content != header 3 2 2 ++ wordBytes rs ++ header 4 2 n then
      throw "the inner proof's round 1 is not the rep's (ρ, σ) = (ε h_fa + h_tx, h_fb + h_ty) and the batching coins"
  | 2 =>
    if !fixedBut (header 5 0 64) 64 (header 4 1 1) then
      throw "the inner proof's round 2 is not τ's 64-byte commitment and β's squeeze"
  | 3 =>
    let y := (c.rowH.zip c.rowMu).map fun (a, b) => a + s.beta * b
    if content != header 3 2 y.size ++ wordBytes y ++ header 4 2 n then
      throw "the inner proof's round 3 is not the rep's Y = (h ‖ pad_h) + β (μ ‖ pad_μ) and the queries"
  | _ => throw s!"a round after the inner proof's {INNER_ROUNDS}"

/-- `round`, with the offsets in `content` of the values it absorbs in the algebraic phase (16 bytes each, ascending): the
values a device round carries freely, since each leaves masked. A round of the inner proof is the one `innerOwn` fixes. -/
def roundFree (c : Cfg) (s : Masker) (content : ByteArray) (n : Nat) : V (ByteArray × Masker × Array Nat) := do
  if s.inner == some INNER_ROUNDS then throw s!"a round after the inner proof's {INNER_ROUNDS}"
  let mut s := s
  let mut out := ByteArray.empty
  let mut free : Array Nat := #[]
  let mut pos := 0
  for _ in [0:content.size + 1] do
    if pos + 16 > content.size then
      if pos == content.size then throw "a round without its squeeze"
      throw "a truncated frame header"
    let op := content.get! pos
    let kind := content.get! (pos + 1)
    let len := (u64OfLE content (pos + 8)).toNat
    pos := pos + 16
    let pad := (len + 15) / 16 * 16
    let body := fun (w : Nat) => if pos + w > content.size then throw "a frame overruns its round" else
      pure (content.extract pos (pos + w))
    match op.toNat with
    | 1 =>
      let _ ← body pad
      pos := pos + pad
    | 2 =>
      let (s', f) ← s.label c (← body len)
      s := s'
      out := out ++ f
      pos := pos + pad
    | 5 =>
      out := out ++ bytesFrame (← body len)
      pos := pos + pad
    | 3 =>
      let b ← body (16 * len)
      if !(kind == 2 || (kind == 1 && len == 1)) then throw s!"an observe of kind {kind} and length {len}"
      out := out ++ header 3 kind len
      for i in [0:len] do
        if s.on && s.pos < c.nPiop then free := free.push (pos + 16 * i)
        let (s', v) ← s.absorb c ⟨u64OfLE b (16 * i), u64OfLE b (16 * i + 8)⟩
        s := s'
        out := F128.pushBytes out v
      pos := pos + 16 * len
    | 4 =>
      if len == 0 && kind == 2 then
        out := out ++ header 4 2 0
      else
        if pos != content.size then throw "a squeeze before the round's end"
        if len != n then throw s!"the round squeezes {len} coins, the caller asked for {n}"
        if !(kind == 2 || (kind == 1 && n == 1)) then throw s!"a squeeze of kind {kind} and length {n}"
        if let some k := s.inner then innerOwn c s k content n
        return (out ++ header 4 kind n, { s with owed := some n }, free)
    | x => throw s!"unknown frame op {x}"
  throw "a round without its squeeze"

/-- One round `content` of the rep configured by `c` that squeezes `n` coins: the round that leaves, and the state after
it, which owes the round's coins; or why the round is not a round of the rep (a frame that overruns it, an unknown op, a
squeeze before its end or of other than `n` coins, a round without its squeeze, more masked values than the layout's, a
masked phase closed early), or a stop the coins decide. -/
def round (c : Cfg) (s : Masker) (content : ByteArray) (n : Nat) : V (ByteArray × Masker) := do
  let (out, s', _) ← roundFree c s content n
  return (out, s')

/-- Whether byte `i` lies in one of the 16-byte words at the offsets `free`. -/
def inFree (free : Array Nat) (i : Nat) : Bool := free.any fun f => f ≤ i && i < f + 16

/-- The first byte at which `v` differs from `r` outside the 16-byte words at the offsets `free`, if any. -/
def firstDiff (v r : ByteArray) (free : Array Nat) : Option Nat :=
  if v == r then none else (List.range v.size).find? fun i => !inFree free i && v.get! i != r.get! i

/-- **The pass decision** on the device round `v` against the statement's scheduled round `r` in its place (`n` coins), at
the rep's state `s`: why `v` fails, if it does. `v` passes when it is `r` byte for byte but for the values `r` absorbs in
the algebraic phase (`roundFree`), which may be any, since each leaves masked: in the algebraic phase `r` is the skeleton's
round with every value zero, so `v` must have its frames, labels, bytes, kinds, lengths, padding and coin count; in the
opening `r` is the firewall's own opening round (`Drive.openStep`), which `v` must equal byte for byte, level 0's masked
values included. The frame's refusal of `r` itself, or a stop the coins decide at
it, is an error. -/
def passes (c : Cfg) (s : Masker) (v r : ByteArray) (n : Nat) : V (Option String) := do
  let (_, _, free) ← roundFree c s r n
  if v.size != r.size then return some s!"the device's round has {v.size} bytes where the statement's has {r.size}"
  return (firstDiff v r free).map fun i => s!"the device's round differs from the statement's at byte {i}"

/-- The scheduled round `r` in the place of a device round that failed (`w`, why): the rep stops, and `r` leaves as a
stopped rep's. -/
def standIn (c : Cfg) (s : Masker) (r : ByteArray) (n : Nat) (w : String) : V (ByteArray × Masker × Option String) := do
  let (out, s') ← round c { s with stopped := true } r n
  return (out, s', some w)

/-- **The device round `v`, with the statement's scheduled round `r` in its place** (`n` coins): the round that leaves,
the state after it, and why `v` failed, if it did. A device round of a rep not stopped that `passes` and that `round`
takes leaves (`round`); else the rep stops and `r` leaves in its place (`standIn`). A stop the coins decide is a stop
whichever round carries it. -/
def check (c : Cfg) (s : Masker) (v r : ByteArray) (n : Nat) : V (ByteArray × Masker × Option String) := do
  let why ← if s.stopped then pure (some "the rep is stopped") else passes c s v r n
  match why with
  | some w => standIn c s r n w
  | none =>
    match round c s v n with
    | .ok (out, s') => return (out, s', none)
    | .error e => if e.startsWith STOP then throw e else standIn c s r n e

/-- The rep's start, from `flock-firewall frame`'s first line: `{"layout": {"n_mlv", "n_lc", "n_ood0", "k0"}, "mask":
{"w0", "n", "blocks"}}` (`mask` optional: without it, no rank check). Anything else is refused: the frame draws the rep's
randomness itself. -/
def start (j : Json) : V (Zk.PadLayout × Option MaskSlot) := do
  let keys := (← j.getObj?).keys
  if let some k := keys.find? (fun k => k != "layout" && k != "mask") then
    throw s!"a start with `{k}`: the frame takes only the layout and the mask slot, and draws the rep's randomness itself"
  let lay ← j.getObjVal? "layout"
  let nat := fun (o : Json) (k : String) => o.getObjValAs? Nat k
  let l : Zk.PadLayout := ⟨← nat lay "n_mlv", ← nat lay "n_lc", ← nat lay "n_ood0", ← nat lay "k0"⟩
  let mask ← match j.getObjVal? "mask" with
    | .ok m => do
      let slot : MaskSlot := ⟨← nat m "w0", ← nat m "n", ← nat m "blocks"⟩
      if slot.n == 0 then throw "a mask slot of no words"
      pure (some slot)
    | .error _ => pure none
  return (l, mask)

/-- `n` bytes of `h` (`Coins.readExact`). -/
def readExact (h : IO.FS.Stream) (n : Nat) : IO ByteArray := Coins.readExact "frame" h n

/-- Where a frame is served: its requests, a line or `n` bytes at a time, and its answers. -/
structure Chan where
  getLine : IO String
  readExact : Nat → IO ByteArray
  write : ByteArray → IO Unit
  flush : IO Unit

/-- Standard input and output. -/
def stdio : IO Chan := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  return { getLine := stdin.getLine, readExact := readExact stdin, write := stdout.write, flush := stdout.flush }

/-- What the frame's host does beside it: `released out n` runs before the round `out`, which squeezes `n` coins, leaves;
`coins k b` is the coins the frame takes for `coins k` followed by the bytes `b`, or why it takes none. On its own
(`cmdWith`) a frame takes the coins in `b`; the session loop's frames (`Session.frames`) take the ones the loop opened. -/
structure Hooks where
  released : ByteArray → Nat → IO Unit := fun _ _ => pure ()
  coins : Nat → ByteArray → IO (Except String (Array F128)) := fun k b => pure (.ok (words b 0 k))

/-- What stands in a round's place: the statement's scheduled round (`round N LEN REFLEN`), the rep's stop (`round N LEN
fail`), or nothing, at the inner proof's own rounds (`round N LEN`). -/
inductive Place where
  | sched (r : ByteArray)
  | fail
  | own

/-- One request to a rep's frame after its start, as `serveWith` reads it: a round `round N LEN …` squeezing `n` coins,
its `LEN` bytes and what stands in its place; `coins N` at what the host's `coins` gave for its `16 N` bytes; an earlier
rep's coins `prior N` with its `16 N` bytes; or `finish`. -/
inductive Req where
  | round (n : Nat) (content : ByteArray) (place : Place)
  | coins (cs : Except String (Array F128))
  | prior (k : Nat) (b : ByteArray)
  | finish

/-- The frame's answer to a request: a round that leaves (`ok LEN'` with its verdict, then its bytes), `ok`, a round it
refuses (`err WHY`, the rep stopping), `finish`'s `ok` or `stop` (`done`), or the rep's end at a refusal (`err WHY`, exit
1) or at a stop (`stop WHY`, exit 0). -/
inductive Reply where
  | round (out : ByteArray) (verdict : String)
  | ok
  | err (why : String)
  | done (stopped : Bool)
  | refuse (why : String)
  | stop (why : String)

/-- Whether the rep ends at the answer. -/
def Reply.ends : Reply → Bool
  | .refuse _ | .stop _ => true
  | _ => false

/-- A rep's frame between requests: its masking state, and whether a round came yet, after which `prior` is refused. -/
structure State where
  m : Masker := {}
  begun : Bool := false
  deriving Inhabited

/-- The verdict a round's answer carries: `pass`, or `fail WHY`, where the scheduled round stands in its place. -/
def verdict : Place → Option String → String
  | .sched _, none => " pass"
  | .sched _, some w => s!" fail {w.replace "\n" " "}"
  | _, _ => ""

/-- **The frame's step**: its answer to the request `q` at the rep's configuration `c` and state `st`, and the state
after it. A round before the last one's coins is refused. A round with its scheduled round is the pass decision (`check`);
one at the rep's stop stops the rep first (`round` of a stopped rep); one alone must be the inner proof's (`round`).
A round `round` refuses stops the rep, and a stop the coins decide ends it. Coins are the frame's taking of them
(`Masker.answer`); an earlier rep's coins add their claim points before the first round (`claimPoints`); `finish` reads
`Masker.unfinished`. -/
def step (c : Cfg) (st : State) : Req → Reply × State
  | .round n content place =>
    match st.m.owed with
    | some k => (.refuse s!"a round before the {k} coins of the last one", st)
    | none =>
      let s := st.m
      let res : V (ByteArray × Masker × Option String) := match place with
        | .sched r => check c s content r n
        | .fail => do
          let (out, s') ← round c { s with stopped := true } content n
          return (out, s', none)
        | .own => do
          let (out, s') ← round c s content n
          if s'.inner.isNone then throw "a round with no scheduled round in its place, outside the inner proof"
          return (out, s', none)
      match res with
      | .ok (out, s', why) => (.round out (verdict place why), { m := s', begun := true })
      | .error e =>
        if e.startsWith STOP then (.stop e, { st with begun := true })
        else (.err e, { m := { s with stopped := true }, begun := true })
  | .coins (.error e) => (.refuse e, st)
  | .coins (.ok cs) =>
    match st.m.answer c cs with
    | .ok s' => (.ok, { st with m := s' })
    | .error e => (if e.startsWith STOP then .stop e else .refuse e, st)
  | .prior k b =>
    if st.begun then (.refuse "an earlier rep's coins after the rep's first round", st)
    else match claimPoints c.lay (words b 0 k) with
      | .ok p => (.ok, { st with m := { st.m with prior := st.m.prior ++ p } })
      | .error e => (.refuse s!"an earlier rep's coins: {e}", st)
  | .finish =>
    match st.m.unfinished c with
    | some why => (.refuse why, st)
    | none => (.done st.m.stopped, st)

/-- The rep's state after the requests `qs` from `st`, and whether it ended: `step` over them in order, up to the first
whose answer ends the rep. -/
def run (c : Cfg) : State → List Req → State × Bool
  | st, [] => (st, false)
  | st, q :: qs =>
    let (a, st') := step c st q
    if a.ends then (st', true) else run c st' qs

/-- One rep's draws, refusals and release, served on `ch` beside the host's hooks `hk`, its draws read from `get`.
The first line starts the rep (`start`): the frame draws `drawBytes` bytes from `get` in one read (`drawsWith`, refusing
any other count), commits them (`padsRoot`), and answers `ok LEN` followed by the
`LEN` bytes of `Draws.toBytes`, the draws then the root. `prior N` followed by `16 N` bytes, before the first round, is an
earlier rep of the table's algebraic-phase coins, whose claim points the rank check covers too; it is answered `ok`.

`round N LEN REFLEN` followed by the device round's `LEN` bytes, then the `REFLEN` bytes of the statement's scheduled round
in its place, is the pass decision (`check`): `ok LEN' pass` followed by the `LEN'` bytes that leave, the device round's
release, or `ok LEN' fail WHY` followed by the scheduled round's release as a stopped rep's, the rep stopping. Each
round that leaves goes to `hk.released` first. Then `coins N` followed by the `16 N` bytes of the coins the stream
answered it is answered `ok`, at the coins `hk.coins` gives for them; no round is taken before the last one's coins. `round N LEN fail` is the scheduled round where no device round passed the firewall's own check (a
worker that failed, or one Rust's checks fail): it stops the rep first, and is answered `ok LEN'`. `round N LEN` alone is
one of the inner proof's `INNER_ROUNDS` rounds, the firewall's own, after the opening's: answered `ok LEN'` when it is the
round `innerOwn` fixes, and refused anywhere else. A round `round` refuses is answered `err WHY`, releases nothing and stops the rep. A round or coins the
refusals stop are answered `stop WHY` (`STOP`), nothing of the round leaving, and end the rep. `finish` is answered `ok`,
or `stop` for a stopped rep, when every masked phase closed with every pad used and the inner proof's rounds all left.
Anything else, and coins `hk.coins` refuses, is answered `err WHY` and ends the rep: nothing after it leaves. Past the
start, the answers and the state are `step`'s: this loop only reads each request's bytes, runs the hooks and writes the
answer. -/
def serveWith (get : USize → IO ByteArray) (ch : Chan) (hk : Hooks) : IO UInt32 := do
  let stderr ← IO.getStderr
  let say := fun (line : String) => ch.write (line ++ "\n").toUTF8
  let refuse := fun (why : String) => do
    say s!"err {why.replace "\n" " "}"
    ch.flush
    return (1 : UInt32)
  let stop := fun (why : String) => do
    say s!"stop {why.replace "\n" " "}"
    ch.flush
    return (0 : UInt32)
  let first ← ch.getLine
  let (l, mask) ← match JsonParse.parse first >>= start with
    | .ok x => pure x
    | .error e => return (← refuse s!"the rep's start: {e}")
  let d ← match ← drawsWith get l with
    | .ok d => pure d
    | .error e => return (← refuse s!"the rep's start: {e}")
  let c := Cfg.ofDraws l d mask
  if let .error e := c.check then return (← refuse s!"the rep's start: {e}")
  let mut st : State := {}
  let drawn := d.toBytes c.root
  say s!"ok {drawn.size}"
  ch.write drawn
  ch.flush
  repeat
    let line := (← ch.getLine).replace "\n" ""
    if line.isEmpty then break
    let q : Req ← match line.splitOn " " with
      | "round" :: n :: len :: flag =>
        let (some n, some len) := (n.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
        let ref? : Option (Option Nat) ← match flag with
          | [] => pure none
          | ["fail"] => pure (some none)
          | [r] => match r.toNat? with
            | some r => pure (some (some r))
            | none => return (← refuse s!"a request `{line}`")
          | _ => return (← refuse s!"a request `{line}`")
        let content ← ch.readExact len
        let place ← match ref? with
          | some (some r) => pure (Place.sched (← ch.readExact r))
          | some none => pure Place.fail
          | none => pure Place.own
        pure (Req.round n content place)
      | ["coins", k] =>
        let some k := k.toNat? | return (← refuse s!"a request `{line}`")
        let b ← ch.readExact (16 * k)
        pure (Req.coins (← hk.coins k b))
      | ["prior", k] =>
        let some k := k.toNat? | return (← refuse s!"a request `{line}`")
        pure (Req.prior k (← ch.readExact (16 * k)))
      | ["finish"] => pure Req.finish
      | _ => return (← refuse s!"a request `{line}`")
    let (a, st') := step c st q
    match a with
    | .round out v =>
      if let (none, some (p, w, r)) := (st.m.rank, st'.m.rank) then
        stderr.putStrLn s!"FRAMERANK\t\{\"points\": {p}, \"mask_words\": {w}, \"rank\": {r}}"
        stderr.flush
      st := st'
      if let .round n _ _ := q then hk.released out n
      say s!"ok {out.size}{v}"
      ch.write out
      ch.flush
    | .ok =>
      st := st'
      say "ok"
      ch.flush
    | .err e =>
      st := st'
      say s!"err {e.replace "\n" " "}"
      ch.flush
    | .done stopped =>
      st := st'
      say (if stopped then "stop" else "ok")
      ch.flush
    | .refuse why => return (← refuse why)
    | .stop why => return (← stop why)
  return 0

/-- `flock-firewall frame`: `serveWith` on standard input and output, taking the coins `coins N` carries. -/
def cmdWith (get : USize → IO ByteArray) (_ : List String) : IO UInt32 := do
  serveWith get (← stdio) {}

/-- `flock-firewall frame` at the operating system's bytes: `cmdWith`, the rep's draws read from `IO.getRandomBytes`. -/
def cmd : List String → IO UInt32 := cmdWith IO.getRandomBytes

/-- The most bytes one `tap` request takes. -/
def TAP_MAX : Nat := 2 ^ 24

/-- `flock-firewall tap`: the OS bytes of every other draw of a firewalled prover (the mask slot's words, level 0's hiding,
the leaf salts of every tree but the pads', the coin tree's nonce), which the prover takes from here and never from the OS
itself. Each request is a line `N`, `N ≤ TAP_MAX`, answered with `N` bytes of `IO.getRandomBytes`. Anything else ends it
with exit 1, and its end of input with exit 0. -/
def tapCmd (_ : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let line := (← stdin.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.toNat? with
    | some n =>
      if n > TAP_MAX then
        IO.eprintln s!"flock-firewall tap: {n} bytes in one request, past {TAP_MAX}"
        return 1
      stdout.write (← IO.getRandomBytes n.toUSize)
      stdout.flush
    | none =>
      IO.eprintln s!"flock-firewall tap: a request `{line}`"
      return 1
  return 0

end Flock.Firewall.Frame
