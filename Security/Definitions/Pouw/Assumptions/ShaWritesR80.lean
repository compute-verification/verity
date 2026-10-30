import Definitions.Pouw.PearlC.W1

/-!
# ShaWrites-R80: SHA-512's round function, credited per word absorbed (draft; a trusted statement, not yet listed)

A fold that absorbs the committed words into SHA-512's own rounds, so the premise it reads is about SHA-512's round
function as `ShaWrites` is (`Pouw.Assumptions.Lead.ShaWrites`, which credits a SHA-512 compression its registers).

* `block x cv w`: **R80** (`x = 0`) is SHA-512's rounds, `K_t` at round `t`, one a word of `w` (at most 80) in place
  of the message schedule, from the state `cv`, then the feed-forward. **R80X** (`x = xc`) XORs word `t` with the
  constant `xc t` first, so that a word that enters only through the round's add can't be skipped: at zero data a word
  is a row walk plus a column walk, and the round reads it only through `h + W_t + K_t`.
* `chain x cv ws`: the blocks absorbed in turn. One chain a thread, carried across calls: a call's words for the chain
  are 80-word blocks and one short last block, and the protocol commits the chain's state as the call leaves it, under
  a hiding SHA-512 leaf, never the words.
* `ShaWritesR80 x dv R ε inst`: for every data the prover chooses, every program on the W1 machine
  (`Pouw.PearlC.W1`) pays, in expectation over the coins, at least `R / 80` INT32 instructions for each word of each
  chain of a window it gets right, less `ε` of the window's words.

**The additive form.** The premise credits each chain the program gets right, rather than bounding the fraction a
cheap program gets right: a program honest on a fraction `f` of the chains and idle on the rest is credited `f` of
the work, so the protocol's gap (additive against stock, in priced work) composes with it term by term. Honest R80X
pays `30·j + 14` writes for a block of `j` words, at least `2,414 / 80 · j` for every `j ≤ 80`, so the credit at
`R = 2,414` never exceeds honest's cost, short blocks included.

**Its relation to `ShaWrites`.** R80 over the 80 words of a block's message schedule is SHA-512's compression of that
block (the `example` below, on FIPS 180-2's `"abc"`). So for programs right on every input, or on any set holding the
padding block's schedule, R80's premise follows from `ShaWrites`' padding-block count, and from its chained-block count
less the schedule's 64 steps. On the protocol's words, which are neither, it is a premise of its own, at `ShaWrites`'
rank: the same function and the same 28 writes a round, on a distribution `ShaWrites` doesn't speak about.

**The machine.** Programs are `W1.Code` with a constant bank, run by `W1.exec` from `initMem`, priced by `W1`'s
instruction prices (fetches unpriced, which only helps the program). `W1.Instr` has no predicate register, so a carry
costs an INT32 write there, while on sm_120 `IADD3` writes its carries as predicates, which W1 does not price. Honest
R80 is 2,254 INT32 writes a block and R80X 2,414 with free carries; at `W1.Instr`, without them, every program pays
more. The premise the protocol needs is this statement at a W1 whose `iadd3` writes its carries as free predicates.
That machine runs every `W1.Instr` program at its price, so its premise implies this one; until `W1.Instr` has them,
this one does not imply it.

**Its falsification test** is `benchmarks/pouw/theory/hash_r80_premise.py` (`falsify`, with this Prop's parameters
as its arguments): a candidate program on a machine with free carry predicates, run on windows of `m` chains of whole
blocks drawn as the protocol draws them, with the verdict this Prop's inequality at confidence `1 - 2^-20`.
-/

namespace Pouw.Assumptions.R80

open Pouw.PearlC Pouw.PearlC.W1

/-- `k` rotated right a byte at a time, at most `r` times, until both 32-bit halves have a nonzero bit in 4..6. -/
def xcRot : ℕ → ℕ → ℕ
  | 0, k => k
  | r + 1, k => if (k >>> 4) % 8 ≠ 0 ∧ (k >>> 36) % 8 ≠ 0 then k else xcRot r (Sha512.rotr k 8)

/-- **R80X's constant for word `t`** of a block: `K_{t+40}` rotated by `xcRot`, then each 32-bit half's low nibble
set to `0x1`. Each half is odd and not `2^32 - 1` or `2^31 - 1`, so the XOR is no add (it would be at `0`, `2^31`,
`2^32 - 1` and `2^31 - 1`); bits 4..6 keep each half's low byte off `0x01` and `0x81`, where a one-sided add and a
`LOP3` match the XOR on the low byte. -/
def xc (t : ℕ) : ℕ :=
  let k := xcRot 8 (Sha512.K.getD ((t + 40) % 80) 0)
  k - k % 16 - (k >>> 32) % 16 * 2 ^ 32 + 2 ^ 32 + 1

/-- A state's 8 words as the working registers. -/
def regsOf (cv : List ℕ) : Sha512.Regs :=
  ⟨cv.getD 0 0, cv.getD 1 0, cv.getD 2 0, cv.getD 3 0, cv.getD 4 0, cv.getD 5 0, cv.getD 6 0, cv.getD 7 0⟩

/-- The words of a block that its rounds read: its first 80. -/
def roundsOf (w : List ℕ) : ℕ := min w.length 80

/-- **One block** with mask `x`: SHA-512's rounds from the state `cv`, round `t` with `K_t` and word `t` of `w` XORed
with `x t` in place of the schedule's `W_t`, one round a word up to 80, then the feed-forward (64-bit words, read mod
`2^64`). -/
def block (x : ℕ → ℕ) (cv w : List ℕ) : List ℕ :=
  let cv := cv.map (· % 2 ^ 64)
  let s := (List.range (roundsOf w)).foldl
    (fun s t => Sha512.round s (Sha512.K.getD t 0) ((w.getD t 0 ^^^ x t) % 2 ^ 64)) (regsOf cv)
  [Sha512.add (cv.getD 0 0) s.a, Sha512.add (cv.getD 1 0) s.b, Sha512.add (cv.getD 2 0) s.c,
    Sha512.add (cv.getD 3 0) s.d, Sha512.add (cv.getD 4 0) s.e, Sha512.add (cv.getD 5 0) s.f,
    Sha512.add (cv.getD 6 0) s.g, Sha512.add (cv.getD 7 0) s.h]

/-- **A chain**: the blocks absorbed in turn from the state `cv`. -/
def chain (x : ℕ → ℕ) : List ℕ → List (List ℕ) → List ℕ
  | cv, [] => cv
  | cv, w :: ws => chain x (block x cv w) ws

/-- A chain's instance: its start state's 8 words and its blocks. -/
abbrev Inst := List ℕ × List (List ℕ)

/-- The words a chain's rounds absorb. -/
def wordCount (c : Inst) : ℕ := (c.2.map roundsOf).sum

/-- An instance's input as 32-bit words, high first (`W1.word64`'s order): the start state's 8 words, then each
block's words its rounds read. The blocks' lengths are the layout's, public. -/
def inputWords (c : Inst) : List ℕ :=
  ((List.range 8).map (c.1.getD · 0) ++ c.2.flatMap fun w => w.take 80).flatMap
    fun v => [v / 2 ^ 32 % 2 ^ 32, v % 2 ^ 32]

/-- The memory a window's run starts from: the program's constants in region 3, chain `i`'s input words in region 4
at `Nat.pair i j`, every other word 0. -/
def initMem {m : ℕ} (consts : List ℕ) (win : Fin m → Inst) : Mem := fun a =>
  if a.1 = 3 then w32 (consts.getD a.2 0)
  else if a.1 = 4 then
    if h : (Nat.unpair a.2).1 < m then w32 ((inputWords (win ⟨_, h⟩)).getD (Nat.unpair a.2).2 0) else 0
  else 0

/-- Chain `i`'s 8 output words, read from region 7 at `Nat.pair i j` as 16 32-bit words, high first. -/
def outOf (M : Mem) (i : ℕ) : List ℕ :=
  (List.range 8).map fun j => w32 (M (7, Nat.pair i (2 * j))) * 2 ^ 32 + w32 (M (7, Nat.pair i (2 * j + 1)))

/-- The run of a program, its constant bank and its code, on a window, from `initMem` with no taint. -/
def run (dv : W1Device) {m : ℕ} (consts : List ℕ) (code : Code) (win : Fin m → Inst) : St :=
  W1.exec dv false code ⟨initMem consts win, [], 0⟩

/-- The chains of a window the run gets right. -/
def right (x : ℕ → ℕ) (dv : W1Device) {m : ℕ} (consts : List ℕ) (code : Code) (win : Fin m → Inst) :
    Finset (Fin m) :=
  Finset.univ.filter fun i => outOf (run dv consts code win).mem i = chain x (win i).1 (win i).2

/-- `pouw-sha-writes-r80` (proposed; `ShaWrites` carried to raw round words, at its rank). **SHA-512's round
function costs its registers on the protocol's words**: for every data `d` the prover chooses, every program of
bounded text, run on the window `inst d s` of `m` chains at each coin value `s`, costs in total over the coins at
least `R / 80` INT32 instructions for each word of each chain it gets right, less `ε` of the window's words:

`R / 80 · int32 · Σ_s (Σ_{i right at s} words(i) - ε · Σ_i words(i)) ≤ Σ_s cost at s`.

The window is the protocol's: `inst d s i` is chain `i`'s start state and its blocks, derived from the data and the
coins, so the program may share work across the window's chains (at zero data their words are row walks plus column
walks), which is what the batch form lets an attack exploit.

Rationale: the round is 28 writes on sm_120 (four 64-bit adds of two 32-bit writes each, `Σ₀` and `Σ₁` of four
funnel shifts and a `LOP3` a half, `Ch` and `Maj` of a `LOP3` a half), and nothing in it is skippable on words of
the prover's choosing; `ShaWrites` accepts the same count of the same function, through its padding block. -/
def ShaWritesR80 (x : ℕ → ℕ) (dv : W1Device) (R ε : ℚ) {D S : Type} [Fintype S] {m : ℕ}
    (inst : D → S → Fin m → Inst) : Prop :=
  ∀ d : D, ∀ (consts : List ℕ) (code : Code), code.size ≤ 2 ^ 64 → consts.length ≤ 2 ^ 64 →
    R / 80 * dv.prices.int32 * ∑ s, ((∑ i ∈ right x dv consts code (inst d s), (wordCount (inst d s i) : ℚ)) -
        ε * ∑ i, (wordCount (inst d s i) : ℚ)) ≤
      ∑ s, (run dv consts code (inst d s)).cost

/-- **ShaWrites-R80X**: `ShaWritesR80` at R80X's constants `xc`. -/
def ShaWritesR80X (dv : W1Device) (R ε : ℚ) {D S : Type} [Fintype S] {m : ℕ} (inst : D → S → Fin m → Inst) :
    Prop :=
  ShaWritesR80 xc dv R ε inst

example : xc 0 = 0xa2bfe8a14cf10361 ∧ xc 1 = 0x01a81a614bbc4231 ∧ xc 79 = 0x3b92722185148231 := by decide +kernel

example : block (fun _ => 0) Sha512.IV (Sha512.schedule 64 (Sha512.blockWords (Sha512.pad [0x61, 0x62, 0x63]))) =
    Sha512.compress Sha512.IV (Sha512.blockWords (Sha512.pad [0x61, 0x62, 0x63])) := by decide +kernel

end Pouw.Assumptions.R80
