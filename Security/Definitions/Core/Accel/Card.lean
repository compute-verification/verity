import Definitions.Core.Accel.Isa

/-!
# A GPU as an accelerator (trusted)

What a GPU instance of `Machine` is, as a table of its data sheet's numbers (`Card`) and the parameters no document or
capture fixes yet (`Open`). Both instances (`Accel.H200`, `Accel.Sm120`) are tables of this shape.

* **Stores** (`Store`): the register file, shared memory (with L1), L2 and device memory (HBM or GDDR), each pooled over
  the card.
* **Resources** (`Res`): the tensor core (one pipe every format shares, a MAC of format `f` using `weight f` units:
  the formats' peaks are one pipe's rate at different widths), twice: at the tick's clock (`tensor`, the per-clock
  peak) and at a measured sustained clock (`tensorLong`, which over any window longer than the backlog caps the MMAs
  there; at `tensor`'s on both instances for now), the warp schedulers' issue slots, the integer `alu` and
  `imad` pipes, the shuffle pipe, the FP64 pipe, shared memory's banks, and device memory's bandwidth. The FP32 pipe's
  rate is the issue rate, so it is not a resource of its own; L2's bandwidth and the register file's ports are in no
  document, so they are free.
* **Units** (`use`): an MMA uses `m·n·k·weight` tensor units and an issue slot; a lane function `1/32` of a slot
  (a warp instruction is 32 lanes' functions) and one lane of its pipe; a uniform instruction a whole slot (the
  uniform datapath has its own ALU, but its instruction takes the warp's issue slot); a copy `1/32` of a slot and, per
  word, one
  unit of each endpoint's bandwidth (a bank word for shared memory, a word for device memory; a register-to-register
  move a lane of the `alu`); an atomic `1/32` of a slot and one word of its store's bandwidth; a branch `1/32` of a
  slot.

Each of these is a relaxation of the card where it isn't exact, so every program the card runs is a program here at no
more units (the instances list them).
-/

open scoped NNRat

namespace Verity.Accel

/-- A GPU's stores. -/
inductive Store
  /-- the register file -/
  | reg
  /-- shared memory, with L1 -/
  | smem
  /-- the L2 cache -/
  | l2
  /-- device memory (HBM or GDDR) -/
  | dram
  deriving DecidableEq

/-- A GPU's resources. -/
inductive Res
  /-- the tensor core, in s8 MACs, at the tick's clock: its per-clock peak, the short-window cap -/
  | tensor
  /-- the tensor core over long windows, in s8 MACs: a sustained clock below the tick's, where a measurement shows
  the card can't exceed it (θ never caps a rate below what the card might reach, so both instances
  set it at `tensor`'s until a clock check on the H200 rules on 1830 MHz). Every tensor-core instruction, DMMA included (one
  datapath for every format, FP64 too: A100 whitepaper, 'Third-generation Tensor Cores'), uses as many units of it as
  of `tensor`, so the formats share one long-window budget -/
  | tensorLong
  /-- the warp schedulers' issue slots, in warp instructions -/
  | issue
  /-- the integer ALU pipe, in lanes -/
  | alu
  /-- the integer multiply-add pipe, in lanes -/
  | imad
  /-- the shuffle pipe, in lanes (its own pipe beside the `alu`; SHFL measured 32 lanes a SM-clock on both cards) -/
  | shfl
  /-- the FP64 pipe, in lanes (`Lane.dfma`) -/
  | f64
  /-- shared memory's banks, in words -/
  | smem
  /-- device memory's bandwidth, in words -/
  | dram
  /-- the L2's read bandwidth, in words the L2 serves -/
  | l2
  deriving DecidableEq

/-- A copy's units, per word, at one endpoint: shared memory's banks or device memory's bandwidth. -/
def port : Store → Res → ℚ≥0
  | .smem, .smem => 1
  | .dram, .dram => 1
  | _, _ => 0

/-- A read's units, per word, at its source: `port`, and a word of the L2's read bandwidth when the L2 serves it. -/
def rd (s : Store) (r : Res) : ℚ≥0 := port s r + if s = .l2 ∧ r = .l2 then 1 else 0

/-- **The units an opcode uses**, `weight f` tensor units a MAC of format `f` at shape scale `scale sh`. An MMA's
operand words read from shared memory pay its banks, a word a bank unit, as a copy's do (`wgmma` reads `B`, and
maybe `A`, there). Every unit is per word or per product, so a price vector over the resources prices an MMA per
product, a lane function per result, and a copy per word moved; an MMA with `D ≠ C` delivers its accumulators to
new words at no units beyond its own (P2's fork), so no such pricing covers a per-seed delivery cost.
An `IMAD.HI` uses `hi` lanes of `imad`, its own
price, not tied to `IMAD`'s, and an `IMAD.WIDE` 2 (CUDA C++ Programming Guide §6.4.1 Table 7, footnote 6: 32 lanes
for extended precision against 64). A broadcast load to `k` lanes pays `k` lanes' share of a slot and its `n` words once at
its source, as the card's port serves lanes that read one word at once (CUDA C++ Programming Guide, 'Shared
Memory'; as `k` copies it would pay `k·n` bank units, 32 × the card's on a whole-warp broadcast). -/
def opUse (weight : Fmt → ℚ≥0) (scale : Shape → ℚ≥0) (hi : ℚ≥0) : Opcode Store → Res → ℚ≥0
  | .mma fa fb _ sh sa sb, r =>
    (if r = .tensor ∨ r = .tensorLong then (sh.m * sh.n * sh.k : ℕ) * weight fa * scale sh else 0) + (if r = .issue then 1 else 0) +
      ((if sa = .smem then (sh.wa fa : ℚ≥0) else 0) + (if sb = .smem then (sh.wb fb : ℚ≥0) else 0)) * port .smem r
  | .mmaSp fa fb _ sh sa sb, r =>
    (if r = .tensor ∨ r = .tensorLong then (sh.m * sh.n * (sh.k / 2) : ℕ) * weight fa * scale sh else 0) + (if r = .issue then 1 else 0) +
      ((if sa = .smem then (sh.waSp fa : ℚ≥0) else 0) + (if sb = .smem then (sh.wb fb : ℚ≥0) else 0)) * port .smem r
  | .lane g, r =>
    match r, g.pipe with
    | .issue, _ => 1 / 32
    | .alu, .alu => 1
    | .imad, .imad => if g matches .madHi _ then hi else if g = .madWide then 2 else 1
    | .shfl, .shfl => 1
    | .f64, .f64 => 1
    | _, _ => 0
  | .uni _, r => if r = .issue then 1 else 0
  | .copy s t n, r =>
    (if r = .issue then 1 / 32 else 0) + (n : ℚ≥0) * (rd s r + port t r) +
      (if s = .reg ∧ t = .reg ∧ r = .alu then (n : ℚ≥0) else 0)
  | .bcast s n k, r => (if r = .issue then ((max k 1 : ℕ) : ℚ≥0) / 32 else 0) + (n : ℚ≥0) * rd s r
  | .atom s _, r => (if r = .issue then 1 / 32 else 0) + rd s r
  | .red s t n _, r => (if r = .issue then 1 / 32 else 0) + (n : ℚ≥0) * (port s r + rd t r)
  | .dmma sh, r =>
    (if r = .tensor ∨ r = .tensorLong then (sh.m * sh.n * sh.k : ℕ) * weight .f64 else 0) + (if r = .issue then 1 else 0)

/-- What no document states and no capture has pinned yet: a parameter of the instance until it is measured. -/
structure Open where
  /-- the in-flight backlog, in ticks: work issued in a window completes within `slack` ticks after it -/
  slack : ℕ
  /-- the tensor units a `b1` MAC uses: no data sheet gives a `b1` peak. On the H200 it measured 1/8;
  sm_120's is unmeasured. A floor that needs `1/8 ≤ b1` (the Pearl-C model's) takes it as a hypothesis on this
  parameter. -/
  b1 : ℚ≥0
  /-- on a card whose data-sheet tensor rate is another instruction's (the H200's is `wgmma`'s), the factor by which
  an `mma.sync` MAC uses more tensor units than that instruction's; unmeasured until a calibration run measures it -/
  sync : ℚ≥0
  /-- the `imad` lanes an `IMAD.HI` uses. The Programming Guide gives `IMAD.HI` 32 lanes a SM-clock against `IMAD`'s 64
  (§6.4.1 Table 7), so 2, θ's floor on both cards; on the H200 it measured 30.6 lanes a SM-clock -/
  madHi : ℚ≥0
  /-- the dot of every format pair and `k` that no capture pins, by formats, saturation and `k` -/
  dot : Fmt → Fmt → Bool → ℕ → Dot
  /-- a 2:4 sparse MMA's dot at logical `k`, over its `k/2` kept pairs (`mmaSpEval`), by formats and saturation: no
  capture pins one on either card -/
  dotSp : Fmt → Fmt → Bool → ℕ → Dot

/-- **A card's table**: its data sheet's numbers, and the dots its captures pin. -/
structure Card where
  /-- the words each store holds -/
  size : Store → ℕ
  size_pos : ∀ s, 0 < size s
  /-- each resource's units a tick -/
  ρ : Res → ℚ≥0
  /-- the tensor units a MAC of each format uses (at `b1`, `Open.b1` instead) -/
  weight : Fmt → ℚ≥0
  /-- the factor on an MMA's tensor units by its shape, from the open parameters (the H200's `mma.sync`, `Open.sync`) -/
  scale : Open → Shape → ℚ≥0
  /-- the opcodes the card has -/
  Avail : Opcode Store → Prop
  /-- the dots a capture pins: by formats, saturation and `k` -/
  captured : Fmt → Fmt → Bool → ℕ → Option Dot
  /-- the captured dots whose capture fixed a zero accumulator: the card's dot is the captured one only there -/
  zeroOnly : Fmt → Fmt → Bool → ℕ → Bool
  /-- the instructions a program's code holds, all threads' texts together: device memory over 16 bytes an instruction -/
  text : ℕ
  /-- the device's coarse limits: s8 tensor MACs, CUDA-core lanes and device-memory bits a tick -/
  tc : ℕ
  cuda : ℕ
  bw : ℕ
  /-- θ's floors on the open parameters: what a data sheet or a cited measurement bounds each one by. The card's
  true parameters are some `Open` meeting them (`Assumptions.CardWithinAccel`), so a theorem over the instance holds at
  every `Open` that does. -/
  floors : Open → Prop

/-- The integer a float code `c` represents (IEEE 754 binary: `F` fraction bits, `E` exponent bits above them, then
the sign), if it is one within ±2^(F+1), the format's exact integers; `none` for every other code. -/
def floatInt (E F c : ℕ) : Option ℤ :=
  let m := c % 2 ^ F
  let e := c / 2 ^ F % 2 ^ E
  let bias := 2 ^ (E - 1) - 1
  let w := (2 ^ F + m) * 2 ^ (e - bias)
  let sign : ℤ := if c / 2 ^ (F + E) % 2 = 1 then -1 else 1
  if e = 0 ∧ m = 0 then some 0
  else if bias ≤ e ∧ e ≤ bias + F + 1 ∧ w % 2 ^ F = 0 ∧ w / 2 ^ F ≤ 2 ^ (F + 1) then some (sign * (w / 2 ^ F : ℕ))
  else none

/-- A `bits`-wide float code as the `bits`-wide two's complement of its exact integer (`floatInt`), unchanged if it
represents none. -/
def decInt (E F bits c : ℕ) : ℕ :=
  match floatInt E F c with
  | some n => (BitVec.ofInt bits n).toNat
  | none => c

/-- How the refinement reads a register word of the card (`Card.decode`): as it is, or as a float code of an
over-approximated unit (`Accel.Isa`'s docstring), an FP64 code's low or high word. -/
inductive Enc
  | raw | f32 | f16x2 | bf16x2 | f64lo | f64hi
  deriving DecidableEq

/-- Word `w` read under encoding `e`, `w'` an FP64 code's other word. -/
def Enc.dec : Enc → Word → Word → Word
  | .raw, w, _ => w
  | .f32, w, _ => BitVec.ofNat 32 (decInt 8 23 32 w.toNat)
  | .f16x2, w, _ => BitVec.ofNat 32 (decInt 5 10 16 (w.toNat % 2 ^ 16) + decInt 5 10 16 (w.toNat / 2 ^ 16) * 2 ^ 16)
  | .bf16x2, w, _ => BitVec.ofNat 32 (decInt 8 7 16 (w.toNat % 2 ^ 16) + decInt 8 7 16 (w.toNat / 2 ^ 16) * 2 ^ 16)
  | .f64lo, w, hi => BitVec.ofNat 32 (decInt 11 52 64 (hi.toNat * 2 ^ 32 + w.toNat))
  | .f64hi, w, lo => BitVec.ofNat 32 (decInt 11 52 64 (w.toNat * 2 ^ 32 + lo.toNat) / 2 ^ 32)

namespace Card

/-- The card's stores' contents: a word at every location (`Machine.Mem` of `machine` at any parameters). -/
abbrev Mem (c : Card) : Type := (Σ s : Store, Fin (c.size s)) → Word

/-- **The card's contents as the model reads them** under register encodings `e`: register word `q`
read by `e q` (an FP64 code's other word at `q + 1` for `f64lo`, `q - 1` for `f64hi`), so a float code that
represents an integer exactly, within its format's exact range, reads as that integer's two's complement, and every
other word, and every word of every other store, as it is. Each word's reading depends on at most it and one
neighbour, so the reading computes nothing a program would otherwise pay for. -/
def decode (c : Card) (e : ℕ → Enc) (M : c.Mem) : c.Mem := fun l =>
  if l.1 = .reg then
    let q := l.2.val
    let q' := if e q = .f64lo then q + 1 else q - 1
    (e q).dec (M l) (M ⟨.reg, ⟨q' % c.size .reg, Nat.mod_lt _ (c.size_pos _)⟩⟩)
  else M l

variable (c : Card) (x : Open)

/-- The tensor units a MAC of each format uses, `b1`'s from `x`. -/
def weightOf (f : Fmt) : ℚ≥0 := if f = .b1 then x.b1 else c.weight f

/-- The dot of each format pair, saturation and `k`: the capture's where one pins it (at a zero accumulator only, for
a capture that fixed it there), `x`'s otherwise. -/
def dot (fa fb : Fmt) (sat : Bool) (k : ℕ) : Dot := fun acc a b =>
  match c.captured fa fb sat k with
  | some d => if c.zeroOnly fa fb sat k && acc != 0 then x.dot fa fb sat k acc a b else d acc a b
  | none => x.dot fa fb sat k acc a b

/-- **The card as an accelerator**, with side channels `cap` and `oracle` oracle queries a tick for its coarse
`Device` (neither is a fact about the card's data flow). -/
def machine {ι : Type} (cap : ι → ℕ) (oracle : ℕ) : Machine ι where
  chip := 32 * (c.size .reg + c.size .smem + c.size .l2)
  mem := 32 * c.size .dram
  rate := fun
    | .tc => c.tc
    | .cuda => c.cuda
    | .oracle => oracle
  bw := c.bw
  cap := cap
  Store := Store
  size := c.size
  size_pos := c.size_pos
  reg := .reg
  Res := Res
  ρ := c.ρ
  slack := x.slack
  Op := {o : Opcode Store // c.Avail o}
  nIn o := o.1.nIn
  nOut o := o.1.nOut
  inStore o := o.1.inStore .reg
  outStore o := o.1.outStore .reg
  eval o := o.1.eval (c.dot x) x.dotSp
  use o := opUse (c.weightOf x) (c.scale x) x.madHi o.1
  braUse r := if r = .issue then 1 / 32 else 0
  text := c.text
  fetch := .dram
  fetchUse := 4

/-- **The card refines its coarse device** (`Accounting.Device`): the machine's tensor units a tick, its issue slots' 32
lanes and its device-memory words' 32 bits a tick are at most the device's s8 MACs, CUDA-core lanes and bits, and every
s8 MAC uses a tensor unit. The stores' bits are the device's by `machine`'s definition. -/
def Refines : Prop := c.ρ .tensor ≤ c.tc ∧ 32 * c.ρ .issue ≤ c.cuda ∧ 32 * c.ρ .dram ≤ c.bw ∧ c.weight .s8 = 1

end Card

/-- The formats an MMA of the integer path pairs: `s8` and `u8` either way, `b1` with `b1`. -/
def intPair : Fmt → Fmt → Bool
  | .s8, .s8 | .s8, .u8 | .u8, .s8 | .u8, .u8 | .b1, .b1 => true
  | _, _ => false

/-- The integer path's dots: `intDot`, exact, for every `k`. -/
def intCaptured (fa fb : Fmt) (sat : Bool) : Option Dot :=
  if intPair fa fb then some (intDot fa fb sat) else none

/-- What one multiplier forms from operand `x` of format `fa` and `y` of `fb`: the classes of the two operands, the
product's sign, the product of their significands and the sum of their exponent fields (a subnormal's as 1), which
is the exact product with its exponent; for an integer pair, the exact product, its sign and magnitude. -/
def mulOut (fa fb : Fmt) (x y : ℕ) : TC.Spec.Cls × TC.Spec.Cls × Bool × ℕ × ℕ :=
  if intPair fa fb then
    let p := fa.intVal x * fb.intVal y
    (.finite, .finite, decide (p < 0), p.natAbs, 0)
  else
    let f := fa.mini
    let g := fb.mini
    let u := fa.code x
    let v := fb.code y
    let eu := (u >>> f.manBits) &&& (2 ^ f.expBits - 1)
    let ev := (v >>> g.manBits) &&& (2 ^ g.expBits - 1)
    let su := (u &&& (2 ^ f.manBits - 1)) ||| (if eu ≠ 0 then 2 ^ f.manBits else 0)
    let sv := (v &&& (2 ^ g.manBits - 1)) ||| (if ev ≠ 0 then 2 ^ g.manBits else 0)
    (f.cls u, g.cls v, ((u >>> (f.bits - 1)) ^^^ (v >>> (g.bits - 1))) &&& 1 = 1, su * sv,
      (if eu ≠ 0 then eu else 1) + (if ev ≠ 0 then ev else 1))

/-- A dot of rows of format `fa` and `fb` and `k` products reads its operand pairs only through their multiplier
outputs: on rows of the format's length, it is a function of the accumulator, the `k` pairs' `mulOut` and the rows'
block scales (empty unless the format is block-scaled). -/
def ProductForm (fa fb : Fmt) (k : ℕ) (d : Dot) : Prop :=
  ∃ F : Word → List (TC.Spec.Cls × TC.Spec.Cls × Bool × ℕ × ℕ) → List ℕ → List ℕ → Word,
    ∀ (c : Word) (a b : List ℕ), a.length = fa.rowLen k → b.length = fb.rowLen k →
      d c a b = F c (List.zipWith (mulOut fa fb) (a.take k) (b.take k)) (a.drop k) (b.drop k)

end Verity.Accel
