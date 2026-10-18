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
(`STOP`): nothing of the round leaves. The inner proof is the firewall's own, and its rounds leave through the frame as
every other round does, unmasked.

Any other stop comes at `Finish`, never before (§10.1): once a round fails the firewall's check, or `round` refuses one, the
rep is `stopped`. The firewall then sends the statement's scheduled round in the device's place, and every value of the
algebraic phase leaves as its pad alone (`0 + h_i`), whatever value the round carries, so what leaves before `Finish` has
the same form and distribution as an honest rep's. A stopped rep's `finish` is `stop`.

`flock-firewall frame` serves this to the firewall (`cmd`), which releases only what it returns. Rust's `zk_veil::Masking`
and `zk_veil::Pads` are its speed references: the firewall holds Rust's masked rounds to this release byte for byte, and
Rust's tree over these draws to `padsRoot`.
-/
import Flock.Zk
import Flock.ZkLigerito

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
  deriving Inhabited

/-- The configuration of a rep of layout `l` with the draws `d` and the statement's mask slot: `d`'s pads, and the root the
frame computes over `d` (`padsRoot`). -/
def Cfg.ofDraws (l : Zk.PadLayout) (d : Draws) (mask : Option MaskSlot) : Cfg :=
  { lay := l, h := d.h, order := absorbed l, nPiop := (absorbed l).size - nLig l, lig := l.k0 > 0,
    root := padsRoot l d, mask }

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
third being `β` (`innerCoins`). -/
def Masker.answer (cfg : Cfg) (s : Masker) (c : Array F128) : V Masker := do
  let some n := s.owed | throw "coins for no round"
  if c.size != n then throw s!"{c.size} coins for a round that squeezed {n}"
  let s := { s with owed := none }
  if let some k := s.inner then
    innerCoins k c
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

/-- One device round `content` of the rep configured by `c` that squeezes `n` coins: the round that leaves, and the state
after it, which owes the
round's coins; or why the round is not a device round (a frame that overruns it, an unknown op, a squeeze before its end or
of other than `n` coins, a round without its squeeze, more masked values than the layout's, a masked phase closed early),
or a stop the coins decide. -/
def round (c : Cfg) (s : Masker) (content : ByteArray) (n : Nat) : V (ByteArray × Masker) := do
  if s.inner == some INNER_ROUNDS then throw s!"a round after the inner proof's {INNER_ROUNDS}"
  let mut s := s
  let mut out := ByteArray.empty
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
        return (out ++ header 4 kind n, { s with owed := some n })
    | x => throw s!"unknown frame op {x}"
  throw "a round without its squeeze"

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

/-- `n` bytes of `h`. Each read that returns nothing ends the input, so `n` reads that each return a byte or more
suffice. -/
def readExact (h : IO.FS.Stream) (n : Nat) : IO ByteArray := go n .empty
where
  go : Nat → ByteArray → IO ByteArray
    | 0, acc =>
      if acc.size ≥ n then pure acc
      else throw (IO.userError s!"frame: {acc.size} bytes after {n} reads of a {n}-byte round")
    | fuel + 1, acc => do
      if acc.size ≥ n then return acc
      let b ← h.read (n - acc.size).toUSize
      if b.isEmpty then throw (IO.userError s!"frame: input closed {acc.size} bytes into a {n}-byte round")
      go fuel (acc ++ b)

/-- `flock-firewall frame`: one rep's draws, refusals and release, served on stdin and stdout. The first line starts the
rep (`start`): the frame draws `drawBytes` OS bytes, commits them (`padsRoot`), and answers `ok LEN` followed by the
`LEN` bytes of `Draws.toBytes`, the draws then the root. `prior N` followed by `16 N` bytes, before the first round, is an
earlier rep of the table's algebraic-phase coins, whose claim points the rank check covers too; it is answered `ok`.

`round N LEN` followed by the device round's `LEN` bytes is answered `ok LEN'` followed by the `LEN'` bytes that leave, and
then `coins N` followed by the `16 N` bytes of the coins the stream answered it is answered `ok`; no round is taken before
the last one's coins. `round N LEN fail` is the statement's scheduled round in place of a device round that failed the
firewall's check: it stops the rep first. A round `round` refuses is answered `err WHY`, releases nothing and stops the rep,
which stays open for the scheduled round in its place. A round or coins the refusals stop are answered `stop WHY` (`STOP`),
nothing of the round leaving, and end the rep. The inner proof's `INNER_ROUNDS` rounds, after the opening's, are rounds
like any other. `finish` is answered `ok`, or `stop` for a stopped rep, when every masked phase closed with every pad used
and the inner proof's rounds all left. Anything else is answered `err WHY` and ends the rep: nothing after it leaves. -/
def cmd (_ : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let refuse := fun (why : String) => do
    stdout.putStrLn s!"err {why.replace "\n" " "}"
    stdout.flush
    return (1 : UInt32)
  let stop := fun (why : String) => do
    stdout.putStrLn s!"stop {why.replace "\n" " "}"
    stdout.flush
    return (0 : UInt32)
  let first ← stdin.getLine
  let (l, mask) ← match Json.parse first >>= start with
    | .ok x => pure x
    | .error e => return (← refuse s!"the rep's start: {e}")
  let d := Draws.ofBytes l (← IO.getRandomBytes (drawBytes l).toUSize)
  let c := Cfg.ofDraws l d mask
  if let .error e := c.check then return (← refuse s!"the rep's start: {e}")
  let mut s : Masker := {}
  let drawn := d.toBytes c.root
  stdout.putStrLn s!"ok {drawn.size}"
  stdout.write drawn
  stdout.flush
  let mut begun := false
  repeat
    let line := (← stdin.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.splitOn " " with
    | "round" :: n :: len :: flag =>
      let (some n, some len) := (n.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let fail ← match flag with
        | [] => pure false
        | ["fail"] => pure true
        | _ => return (← refuse s!"a request `{line}`")
      let content ← readExact stdin len
      if let some k := s.owed then return (← refuse s!"a round before the {k} coins of the last one")
      begun := true
      if fail then s := { s with stopped := true }
      match round c s content n with
      | .ok (out, s') =>
        if let (none, some (p, w, r)) := (s.rank, s'.rank) then
          stderr.putStrLn s!"FRAMERANK\t\{\"points\": {p}, \"mask_words\": {w}, \"rank\": {r}}"
          stderr.flush
        s := s'
        stdout.putStrLn s!"ok {out.size}"
        stdout.write out
        stdout.flush
      | .error e =>
        if e.startsWith STOP then return (← stop e)
        s := { s with stopped := true }
        stdout.putStrLn s!"err {e.replace "\n" " "}"
        stdout.flush
    | ["coins", k] =>
      let some k := k.toNat? | return (← refuse s!"a request `{line}`")
      let b ← readExact stdin (16 * k)
      match s.answer c (words b 0 k) with
      | .ok s' =>
        s := s'
        stdout.putStrLn "ok"
        stdout.flush
      | .error e => return (← if e.startsWith STOP then stop e else refuse e)
    | ["prior", k] =>
      let some k := k.toNat? | return (← refuse s!"a request `{line}`")
      let b ← readExact stdin (16 * k)
      if begun then return (← refuse "an earlier rep's coins after the rep's first round")
      match claimPoints c.lay (words b 0 k) with
      | .ok p =>
        s := { s with prior := s.prior ++ p }
        stdout.putStrLn "ok"
        stdout.flush
      | .error e => return (← refuse s!"an earlier rep's coins: {e}")
    | ["finish"] =>
      if let some why := s.unfinished c then return (← refuse why)
      stdout.putStrLn (if s.stopped then "stop" else "ok")
      stdout.flush
    | _ => return (← refuse s!"a request `{line}`")
  return 0

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
