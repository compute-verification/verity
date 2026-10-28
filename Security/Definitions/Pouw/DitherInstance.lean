import Definitions.Pouw.Dither
import Definitions.Core.Accel.Machine
import Definitions.Core.Accel.H200
import Definitions.Core.Accel.Sm120

/-!
# The headline's model on the cards: the H200 and the RTX PRO 6000 (trusted)

`Definitions.Pouw.Guarantees.Dither.WorkSound` is stated over an abstract accelerator `Accel`. This file ties that
`Accel` to the machine 𝓜 (`Definitions.Core.Accel`) at a card's numbers, so the headline can be read on
`Verity.Accel.H200.card` and `Verity.Accel.Sm120.card` (`Definitions.Pouw.Guarantees.DitherInstance`):

* **the period's devices** (`Devices`): `n` cards of one table pooled into one machine (`poolCard`, a coarsening:
  the pooled cards share their stores and resources, so the links between them are free), and the period's other
  devices (`Others`: the host CPUs and DPUs the cards reach without crossing a network certifier), each resource
  at its θ rate;
* **where a round's coin and words sit in the cards' memory** (`RoundPlan`);
* **`G` is the devices at the card's semantics `x`** (`OnCard`): `G`'s resources are the pooled cards' and the
  others', at their rates; its memory is theirs; its (P1) and (P2) are the machine's at `x`; and every words program
  the cards run is a program of `G` with the run's units as its use, its seconds as its wall time and no import;
* **the public best methods** (`PubBest`), fixed per card and priced at the period's dual price (`pubCost`): the
  public-best menu's named programs (`pubBestH200`, `pubBestSm120`);
* **the cards' hosts** (`HostRow`, `HostsCounted`): the AMX and SHA-512 rates of each card's share of its node's CPUs,
  which the period's devices count;
* **an oracle call on the card** (`CallsAtPub`): one compression of a public SHA-512 schedule, the call step's use;
* **the card's runs at `x`** (`CardRunsAt`): `Assumptions.CardWithinAccel`'s refinement at the semantics it gives;
* **the network certifier's counts** (`Counted`): the verifier refuses a period whose signed counts into
  or out of the devices, read from the devices' readings (`Readings`), pass the caps `E_in`, `E_out`;
* **the auditor's clocks** (`Timed`): the challenger issues each sampling block's dither coins only after the
  block's operand root is in, at least `Δ` before the period's end, and the next block's only after every root of
  this one is in. The verifier refuses a period in which the network certifier saw a root arrive more than `Δ`
  after its block's coins, or a selected block's answer more than `τ_ans` after its check coin. Only selected
  blocks are answered, and an answer may arrive after later blocks' coins. `TimedWorkFits` and `AnswersFit` are
  what that buys: the sampling blocks' commit phases and work within the period's capacity, at any number of
  blocks, and each selected answer within the period.

The card's semantics `x` are not fixed here: the headline holds at every `x` meeting P1 (`Card.Exact`) and P2
(`Open.ProductForm`, as `Assumptions.AccelProductForm`).
-/

open scoped NNRat

namespace Pouw.Dither

open Pouw.RelativeSecurity

section Cards

instance instFintypeAccelRes : Fintype Verity.Accel.Res where
  elems := {.tensor, .issue, .alu, .imad, .shfl, .f64, .sfu, .smem, .l2, .dram}
  complete r := by cases r <;> simp

/-- **`n` cards pooled into one**: every store, every resource's rate and the code space `n` times the card's, the
same weights and range. A coarsening, as the card's own pools over its SMs are: any thread names any card's
locations, so the links between the cards (NVLink, PCIe peer-to-peer) cost nothing, and a lower bound in the pool
holds on the cards. -/
def poolCard (c : Verity.Accel.Card) (n : ℕ) (hn : 0 < n) : Verity.Accel.Card :=
  { c with
    size := fun s => n * c.size s
    size_pos := fun s => Nat.mul_pos hn (c.size_pos s)
    ρ := fun r => (n : ℚ≥0) * c.ρ r
    text := n * c.text }

/-- **The period's other devices**: every processor the period's cards reach without
crossing a network certifier other than the cards themselves (host CPUs over PCIe or NVLink-C2C, DPUs), each
resource with its θ rate in units a second (θ's per-node rows: SHA-512 and INT8 rates, DRAM, PCIe, C2C) and
their memory in bits. Their programs are `Accel`'s like the cards', priced at the period's dual price, so the honest
optimum may use them and the capacity `C` includes them. That the list is complete is the inventory premise
(`Assumptions.PeriodInventory`). -/
structure Others where
  Res : Type
  [instRes : Fintype Res]
  rate : Res → ℝ
  mem : ℝ

attribute [instance] Others.instRes

/-- **The period's devices**: `n` cards of the table `c`, at the table's rates a second (its clock's ceiling), and
the other devices. -/
structure Devices (c : Verity.Accel.Card) where
  n : ℕ
  n_pos : 0 < n
  oth : Others

/-- The period's cards as one machine at the card's semantics `x`. -/
def Devices.machine {c : Verity.Accel.Card} (D : Devices c) (x : Verity.Accel.Open) : Verity.Accel.Machine :=
  ⟨poolCard c D.n D.n_pos, x⟩

/-- **Where a round's coin and words sit in the cards' memory**: the dither coin written into the memory a words
program starts from (`putDraw`), and the words read from the memory it ends with (`getTr`), as the deployed kernel lays
them out. A round plan moves words; it computes nothing. The headline holds at every round plan. -/
structure RoundPlan (m : Verity.Accel.Machine) (P : RoundProto) where
  putDraw : P.Draw → m.Mem → m.Mem
  getTr : m.Mem → P.Tr

/-- A run of `Prg` on the schedule `sched` from the memory `M`, every thread at its text's start. -/
def cardRun (m : Verity.Accel.Machine) (Prg : m.Prog ℕ) (M : m.Mem) (sched : List ℕ) :
    m.State ℕ × List m.Instr :=
  Verity.Accel.Machine.run Prg ⟨M, fun _ => 0⟩ sched

/-- The units of resource `r` a run's instructions use. -/
def cardUnits (m : Verity.Accel.Machine) (iss : List m.Instr) (r : Verity.Accel.Res) : ℚ≥0 :=
  Verity.Accel.Machine.units iss r

/-- **`G` is the period's devices at the card's semantics `x`**: under the equivalence `e` of `G`'s resources with the
pooled cards' and the other devices',

* each card resource runs at `n ρ_r` units a second, and each other device's at its θ rate;
* `G`'s memory is the pooled cards' bits and the other devices';
* `G`'s (P1) and (P2) are the machine's at `x` (`Card.Exact`, `Open.ProductForm`);
* every words program the cards run is one of `G`'s: for every memory `M₀` the round starts from (whatever the prover
  carried into it) and every program and schedule of the pooled machine, fixed before the coin, meeting (G1) and (G2)
  from `M₀` with the coin written in, some words program of `G` computes the words the run leaves, uses exactly the
  run's units on each card resource and nothing of the other devices, takes the run's seconds as wall time, exchanges
  nothing and makes no oracle call (the machine has no oracle instruction).

So `SAKMatmul` at `G` speaks of every words program of the card at `x`, at the card's rates; the dual price, the honest
cost `H` and the capacity `C` are in the card's units. `G.ext` is the bits a run exchanges, which the network certifier
counts (`Assumptions.CertifierSeesImports`). The commitment and the round's other programs read the oracle `P.Orc`,
which the machine has no instruction for: they reach it through `G`'s priced calls (G3), each at a public SHA-512
schedule's units on the card (`CallsAtPub`); that the card can't query the oracle more cheaply is the realization
premise `Assumptions.ShaRealize`, which is open. A program of `G` cheaper than any of the card's makes `SAKMatmul` or
`SAKSha` false at `G`, and the headline then says nothing at that `G`; the `G` the headline is read at has the card's
programs and no others. -/
def OnCard (G : Accel) (c : Verity.Accel.Card) (x : Verity.Accel.Open) (D : Devices c) (P : RoundProto)
    (L : RoundPlan (D.machine x) P) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) : Prop :=
    (∀ r, G.rate (e.symm (.inl r)) = ((poolCard c D.n D.n_pos).ρ r : ℝ)) ∧
    (∀ h, G.rate (e.symm (.inr h)) = D.oth.rate h) ∧
    G.mem = ((D.machine x).bits : ℝ) + D.oth.mem ∧
    (G.P1 ↔ c.Exact x) ∧ (G.P2 ↔ x.ProductForm) ∧
    ∀ (M₀ : (D.machine x).Mem) (Prg : (D.machine x).Prog ℕ) (sched : List ℕ) (T : P.Draw → ℚ≥0),
      (∀ d, Verity.Accel.Machine.Fits Prg ⟨L.putDraw d M₀, fun _ => 0⟩ sched (T d)) →
      ∃ Mw : G.Mc.Alg ((D.machine x).Mem × P.Draw) P.Tr, ∀ d,
        G.Mc.eval Mw (M₀, d) = L.getTr (cardRun (D.machine x) Prg (L.putDraw d M₀) sched).1.mem ∧
        (∀ r, G.use Mw (M₀, d) (e.symm (.inl r)) =
          (cardUnits (D.machine x) (cardRun (D.machine x) Prg (L.putDraw d M₀) sched).2 r : ℝ)) ∧
        (∀ h, G.use Mw (M₀, d) (e.symm (.inr h)) = 0) ∧
        G.wall Mw (M₀, d) = (T d : ℝ) ∧ G.ext Mw (M₀, d) = 0 ∧ G.calls Mw (M₀, d) = 0

/-- A program's units on the card's resources: a naive MAC's for a matmul program, a compression's for a SHA-512
schedule. -/
abbrev ResUnits := Verity.Accel.Res → ℚ≥0

/-- **The public best methods on the card table `c`**: the named public matmul programs (`mm`, units a naive MAC)
and the pinned public SHA-512 schedules (`sha`, units a compression), each in the model's units, at the cheapest
instance a prover can register (one constant must hold at every instance the prover may pick), with advice of the
allowed size (forms of the fixed weights and lookup tables, priced as the public-best menu prices them). The
headline prices them at the period's dual price (`pubCost`): the cheapest of the named programs there, times the
round's MACs or compressions. So `pub_mm` and `pub_sha` are fixed per card, not chosen per reading. -/
structure PubBest (c : Verity.Accel.Card) where
  mm : List ResUnits
  sha : List ResUnits

/-- Units on the card's resources priced at `μ` through the resource match `e`. -/
noncomputable def unitCost {G : Accel} {c : Verity.Accel.Card} {D : Devices c}
    (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (μ : G.Res → ℝ) (u : ResUnits) : ℝ :=
  ∑ r, μ (e.symm (.inl r)) * (u r : ℝ)

/-- The public best's cost at `μ` for `n` MACs or compressions: the cheapest of the named programs at `μ`, `n` times
(0 for no program). -/
noncomputable def pubCost {G : Accel} {c : Verity.Accel.Card} {D : Devices c}
    (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (μ : G.Res → ℝ) (us : List ResUnits) (n : ℝ) : ℝ :=
  n * ((us.map (unitCost e μ)).min?.getD 0)

/-- A matmul program's units a naive MAC: `t` on the tensor core, `i` issue slots, `a` ALU lanes, `m` IMAD lanes and
`s` shared-memory words. -/
def mmUnits (t i a m s : ℚ≥0) : Verity.Accel.Res → ℚ≥0
  | .tensor => t
  | .issue => i
  | .alu => a
  | .imad => m
  | .smem => s
  | _ => 0

/-- A SHA-512 schedule's units a compression: `a` ALU lanes, `m` IMAD lanes and `i` issue slots. -/
def shaUnits (a m i : ℚ≥0) : Verity.Accel.Res → ℚ≥0
  | .alu => a
  | .imad => m
  | .issue => i
  | _ => 0

/-- **The public-best menu's named matmul programs on the H200**
(`art:4eb6719838c94485978f03fe075cb004e7693a318c6c5b980590f4b3d2f01263`, computed by
`benchmarks/pouw/theory/d5_v2.py` at `34866fd9f8`): protocol v3's round at A = B = 0, the frozen qkv shape
(K 16384, n 16512) at the served caps (t 2, L' 512), the hull of every route in the menu (the tile route, the
O-route, the zero-high-digit peel, Strassen's orbit, the integral catalogue, every route beside the MMA) priced at
the honest period's duals μ° without the hashing. Served (μ° on `alu` and `joint0`): the dense product with O by the
O-route, pub_mm 0.4853 against H_mm 0.4984. Job-only (μ° on `alu` alone, the matmul unpriced): the dense product
with the peels, which is also the honest kernel. At a fixed price the cheapest listed program is at most any mixture
of them, so the hull's pub at μ° is the list's minimum. The menu's units in the model's: its tensor MAC-times as
they are, its lane-issues over 32 (a warp instruction), its FMA-pipe lanes over 2 (an `IMAD` or `IDP.4A` takes two
of θ's 128, one of the model's 64 `imad` lanes) and its shared-memory bytes over 4 (words). -/
def pubMmH200 : List ResUnits :=
  [mmUnits (35 / 32) (99 / 2097152) 0 0 (207 / 32768), mmUnits (9 / 8) (25 / 524288) 0 0 (53 / 8192)]

/-- **The public-best menu's named matmul programs on the RTX PRO 6000** (as on the H200, at the served caps t 3,
L' 256). Served (μ° on `alu`, `rf` and `joint1`): Strassen's 2×2 orbit at depth 1, pub_mm 0.8255 against H_mm 0.8586.
Job-only (μ° on `alu` and `rf`): the dense product with O by the zero-high-digit peel, pub_mm 0.2930 against H_mm
0.2974. And the honest kernel. Values off a dyadic are rounded down at the seventh decimal. The menu's register-file
words aren't a resource of the model and are dropped, which only lowers `pub_mm`. -/
def pubMmSm120 : List ResUnits :=
  [mmUnits (59 / 64) (56829 / 40000000) (63 / 4000) 0 (1678031 / 40000000),
   mmUnits (67 / 64) (329 / 3145728) 0 0 (233 / 24576), mmUnits (17 / 16) (83 / 786432) 0 0 (59 / 6144)]

/-- **The pinned public SHA-512 schedules on the H200, at the model's prices** (`SAKSha` is in model units, so
`c_sha = 1` in the model): the default build of `commit_gpu` (measured 4.79 G/s; SHF 1,728, LOP3 912, IADD3 760, IMAD
152), A-wide + R-hi (the menu's `pub_sha.H200`; IMAD 546, IMAD.WIDE 546, IMAD.HI 394) and A-wide + R-shf (IMAD 850,
IMAD.WIDE 850), each schedule's SASS counts a compression. Priced as the model prices a lane op (`Card.use`): one ALU
or `imad` lane each, `IMAD.WIDE` and `IMAD.HI` included (a lane op takes up to four words to two), and `1/32` of an
issue slot. These are the menu's schedules too, with the same ALU and issue counts. -/
def pubShaH200 : List ResUnits :=
  [shaUnits 3400 152 (3608 / 32), shaUnits 2460 1486 (4002 / 32), shaUnits 2550 1700 (4306 / 32)]

/-- **The pinned public SHA-512 schedules on the RTX PRO 6000, at the model's prices** (as on the H200): the default
build (measured 7.13 G/s; SHF 1,728, LOP3 927, IADD.64 760), R-shf (IMAD 852, IMAD.WIDE 852)
and R-hi (the menu's `pub_sha.RTXPRO6000`; IMAD 487, IMAD.WIDE 487, IMAD.HI 487). -/
def pubShaSm120 : List ResUnits :=
  [shaUnits 3415 0 (3415 / 32), shaUnits 2563 1704 (4267 / 32), shaUnits 2441 1461 (3902 / 32)]

/-- The H200's public best methods: the public-best menu's named programs. -/
def pubBestH200 : PubBest Verity.Accel.H200.card := ⟨pubMmH200, pubShaH200⟩

/-- The RTX PRO 6000's public best methods: the public-best menu's named programs. -/
def pubBestSm120 : PubBest Verity.Accel.Sm120.card := ⟨pubMmSm120, pubShaSm120⟩

/-- **An oracle call costs a public SHA-512 compression on the card**: `G`'s oracle-call step (G3)
uses, on the card's resources, the units of one compression of one of the public schedules
`pb.sha`, and nothing of the other devices. A call is one evaluation of a tree input: a leaf's is one compression and
a node's (two digests) two, so one compression is the floor, and a card resource the schedule uses caps the calls at
its capacity over the schedule's use of it (WorkSound's `cap_r ≤ Q callUse_r`). Host CPUs that hash are other work's
devices; their calls are priced there. -/
def CallsAtPub (G : Accel) {c : Verity.Accel.Card} {D : Devices c} (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res)
    (pb : PubBest c) : Prop :=
  ∃ u ∈ pb.sha, (∀ r, G.callUse (e.symm (.inl r)) = (u r : ℝ)) ∧ ∀ h, G.callUse (e.symm (.inr h)) = 0

/-- **The card's runs are its model's at `x`** (`Assumptions.CardWithinAccel`'s clause, at the semantics it gives):
every memory-to-memory map `f` the card computes within `T` seconds on the memories `D` (`can f T D`) is a program and
schedule of the machine `⟨c, x⟩` that meet (G1) and (G2) in `T` seconds and take `M` to `f M` on every `M` in `D`. -/
def CardRunsAt (c : Verity.Accel.Card) (x : Verity.Accel.Open)
    (can : (c.Mem → c.Mem) → ℚ≥0 → (c.Mem → Prop) → Prop) : Prop :=
  ∀ f T D, can f T D → ∃ (P : (Verity.Accel.Machine.mk c x).Prog ℕ) (sched : List ℕ),
    ∀ σ : (Verity.Accel.Machine.mk c x).State ℕ, D σ.mem → (∀ θ, σ.pc θ = 0) →
      Verity.Accel.Machine.Fits P σ sched T ∧ (Verity.Accel.Machine.run P σ sched).1.mem = f σ.mem

/-- The host resources a card's node brings: the CPUs' AMX, in s8 MACs, and their SHA-512, in
compressions. -/
inductive HostRes
  | amx
  | sha
  deriving DecidableEq

instance instFintypeHostRes : Fintype HostRes where
  elems := {.amx, .sha}
  complete r := by cases r <;> simp

/-- **A card's share of its node's hosts**, in units a second (θ's `host.cpu` rows at `188907341d`). -/
structure HostRow where
  amx : ℝ
  sha : ℝ

/-- A row's hosts for `n` cards as other devices: AMX and SHA-512 at `n` times the row's rates, and `mem` bits. -/
noncomputable def HostRow.others (h : HostRow) (n : ℕ) (mem : ℝ) : Others where
  Res := HostRes
  rate r := match r with
    | .amx => n * h.amx
    | .sha => n * h.sha
  mem := mem

/-- **The period's devices count the cards' hosts**: the other devices have an AMX and a SHA-512 resource for the `n`
cards' hosts, at least `n` times the row's rates. So the honest optimum may use them, and the capacity `C` includes
them. -/
def HostsCounted {c : Verity.Accel.Card} (D : Devices c) (h : HostRow) : Prop :=
  ∃ j : HostRes → D.oth.Res, Function.Injective j ∧
    D.n * h.amx ≤ D.oth.rate (j .amx) ∧ D.n * h.sha ≤ D.oth.rate (j .sha)

/-- **The H200 node's hosts, a card's share**: 8 Xeon 8468 cores a GPU; AMX at 3.15% of the H200's dense
INT8 (1979 TOPS, so 989.5 · 10¹² MACs a second), SHA-512 by AVX-512 multi-buffer at 2.934 · 10⁷ compressions a
second a core (4.90% of the card's measured 4.79 G/s). -/
noncomputable def hostH200 : HostRow := ⟨315 / 10000 * (1979 * 10 ^ 12 / 2), 8 * (2934 * 10 ^ 4)⟩

/-- **The RTX PRO 6000 node's hosts, a card's share**: 12 Xeon 6776P cores a GPU; AMX at 9.51% of the card's
dense INT8 (10¹⁵ ops, 5 · 10¹⁴ MACs a second), SHA-512 at 3.012 · 10⁷ compressions a second a core (5.07% of the
card's measured 7.13 G/s). -/
noncomputable def hostSm120 : HostRow := ⟨951 / 10000 * (5 * 10 ^ 14), 12 * (3012 * 10 ^ 4)⟩

end Cards

section Certifier

/-- **The verdict with the network certifier's counts**: the service's verdict, and the certifier's signed counts of the
bits into (`cin`) and out of (`cout`) the period's devices, as the verdict reads them from the devices' readings, are
within the caps `E_in` and `E_out`, fixed before the period at served traffic; a period past either is refused. The
counts exclude the challenger's coin stream and nothing else, as `Assumptions.CertifierSeesImports` defines it: the
dither coin is the sampling block's `Draw`, an input of its programs, not an exchanged bit (`Accel.ext`). A replica on
several nodes has one pair of counts, its certifiers' together. -/
def Counted {P : RoundProto} {k : ℕ} {Obs : Type} (verdict : Verdict P k Obs) (cin cout : Obs → ℝ)
    (Ein Eout : ℝ) : Verdict P k Obs :=
  fun o x tr ob => verdict o x tr ob ∧ cin ob ≤ Ein ∧ cout ob ≤ Eout

end Certifier

section Clock

/-- **The verdict with the auditor's clocks**: the service's verdict, read with the challenger's signed issue times
and the network certifier's signed arrival times, as the verdict reads them from the devices' readings `ob`. Each
sampling block runs its commit phase, then its challenge phase, then its response phase, every commitment bound
before any of its coins.

* **The commit phase.** The network certifier signs the arrival of sampling block `b`'s operand root
  (`opArrive ob b`), and the challenger issues the block's dither coins at `t ob b`, after it: the period is refused
  if `t ob b` came before `opArrive ob b`. So the block's registered instance is read at `opArrive`, before any coin
  of the block. The coins' message is derived from that receipt (the derivation `OwnDither` rests on,
  with the receipt's digest in its context), so the coins can't be issued against an operand root that wasn't in.
* **The challenge phase.** The dither coins go out at least `Δ` before the period's end, and block `b + 1`'s only
  after every root of block `b` is in, so the blocks run one at a time. The network certifier watches the replica's
  boundary; the period is refused if it saw a root of block `b` arrive (`arrive ob b`) more than `Δ` after the block's
  dither coins.
* The challenger issues every sampling block's check coin, which carries whether the block is selected, at `tc ob b`,
  and the period is refused if that was before every root of the block was in: a prover told early
  that its block isn't selected could skip the block. The coin's message carries the digest of the certifier's
  signed receipt of the roots, so the two times are on one clock.
* Only selected sampling blocks are answered (`sel`, read from the block's check coin). A selected block's check coin
  is issued at least `τans` before the period's end, and the period is refused if its answer arrived (`ansAt ob b`)
  more than `τans` after it. An answer may arrive after later blocks' coins.

The number of sampling blocks and of selected ones is bounded by nothing here. The clocks are the auditor's devices,
and nothing here reads a time the prover reports. -/
def Timed {G : Accel} {P : RoundProto} {nsb : ℕ} {Obs : Type} (verdict : Verdict P nsb Obs) (Pd : Jobs G) (Δ : ℝ)
    (opArrive t arrive : Obs → Fin nsb → ℝ) (sel : P.Chk → Prop) (τans : ℝ) (tc ansAt : Obs → Fin nsb → ℝ) :
    Verdict P nsb Obs :=
  fun o x tr ob => verdict o x tr ob ∧ (∀ b, 0 ≤ t ob b ∧ t ob b + Δ ≤ Pd.T ∧ arrive ob b ≤ t ob b + Δ) ∧
    (∀ b, opArrive ob b ≤ t ob b) ∧
    (∀ b b' : Fin nsb, (b : ℕ) + 1 = b' → arrive ob b ≤ t ob b') ∧ (∀ b, arrive ob b ≤ tc ob b) ∧
    ∀ b, sel (x b).2 → 0 ≤ tc ob b ∧ tc ob b + τans ≤ Pd.T ∧ ansAt ob b ≤ tc ob b + τans

/-- **An accepted period's sampling blocks' commit phases and work fit the period**: for every prover whose period the
verdict accepts, its sampling blocks' timed use (`Prover.timedUse`: each block's commit phase and work, from the
previous block's work to its commitment) is, resource by resource, at most the period's capacity `Jobs.cap`. This is what lets the headline count that use
against the period's devices: it was done on them, inside the period. -/
def TimedWorkFits (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (verdict : Verdict P nsb Obs)
    (dev : Readings G P nsb Obs) (Pd : Jobs G) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk), A.Passes verdict dev o x →
    ∀ r, A.timedUse o x r ≤ Pd.cap r

end Clock

end Pouw.Dither
