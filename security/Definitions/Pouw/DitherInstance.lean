module

public import Definitions.Pouw.Dither
public import Definitions.Core.Accel.Machine
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! The no-fold Dither headline's accelerator tied to the machine 𝓜 at a card's numbers, at every card semantics `x` meeting P1 and P2. -/

@[expose] public section

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
crossing a network attester other than the cards themselves (host CPUs over PCIe or NVLink-C2C, DPUs), each
resource with its θ rate in units a second (θ's per-node rows: SHA-512 and INT8 rates, DRAM, PCIe, C2C) and
their memory in bits. Their programs are `Accel`'s like the cards', priced at the period's dual price, so the honest
optimum may use them and the capacity `C` includes them. That the list is complete is the inventory premise
(`Assumptions.MonitoringPeriodInventory`). -/
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
cost `H` and the capacity `C` are in the card's units. `G.ext` is the bits a run exchanges, which the network attester
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

/-- The public-best menu's matmul programs on the H200 at the served caps, priced at the honest period's duals without
the hashing. The menu's units in the model's: tensor MAC-times as they are, lane-issues over 32, FMA-pipe lanes over 2
(an `IMAD` takes two of θ's 128) and shared-memory bytes over 4. -/
def pubMmH200 : List ResUnits :=
  [mmUnits (35 / 32) (99 / 2097152) 0 0 (207 / 32768), mmUnits (9 / 8) (25 / 524288) 0 0 (53 / 8192)]

/-- Values off a dyadic are rounded down at the seventh decimal; the menu's register-file words aren't a resource of the
model and are dropped, which only lowers `pub_mm`. -/
def pubMmSm120 : List ResUnits :=
  [mmUnits (59 / 64) (56829 / 40000000) (63 / 4000) 0 (1678031 / 40000000),
   mmUnits (67 / 64) (329 / 3145728) 0 0 (233 / 24576), mmUnits (17 / 16) (83 / 786432) 0 0 (59 / 6144)]

/-- Each schedule's SASS counts a compression, priced at one ALU or `imad` lane a lane op (`IMAD.WIDE` and `IMAD.HI`
included) and `1/32` of an issue slot. -/
def pubShaH200 : List ResUnits :=
  [shaUnits 3400 152 (3608 / 32), shaUnits 2460 1486 (4002 / 32), shaUnits 2550 1700 (4306 / 32)]

def pubShaSm120 : List ResUnits :=
  [shaUnits 3415 0 (3415 / 32), shaUnits 2563 1704 (4267 / 32), shaUnits 2441 1461 (3902 / 32)]

/-- At the card table `c` that states `hardware/cards/h200-sxm-141gb.toml`. -/
def pubBestH200 (c : Verity.Accel.Card) : PubBest c := ⟨pubMmH200, pubShaH200⟩

/-- At the card table `c` that states `hardware/cards/rtx-pro-6000-server.toml`. -/
def pubBestSm120 (c : Verity.Accel.Card) : PubBest c := ⟨pubMmSm120, pubShaSm120⟩

/-- **An oracle call costs a public SHA-512 compression on the card**: `G`'s oracle-call step (G3)
uses, on the card's resources, the units of one compression of one of the public schedules
`pb.sha`, and nothing of the other devices. A call is one evaluation of a tree input: a leaf's is one compression and
a node's (two digests) two, so one compression is the floor, and a card resource the schedule uses bounds the calls by
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

/-- A card's share of the H200 node's hosts, 8 Xeon 8468 cores: AMX in MACs a second, SHA-512 in compressions a second. -/
noncomputable def hostH200 : HostRow := ⟨315 / 10000 * (1979 * 10 ^ 12 / 2), 8 * (2934 * 10 ^ 4)⟩

/-- A card's share of the RTX PRO 6000 node's hosts, 12 Xeon 6776P cores: AMX in MACs a second, SHA-512 in compressions a second. -/
noncomputable def hostSm120 : HostRow := ⟨951 / 10000 * (5 * 10 ^ 14), 12 * (3012 * 10 ^ 4)⟩

end Cards

section Certifier

/-- **The verdict with the network attester's counts**: the service's verdict, and the certifier's signed counts of the
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
and the network attester's signed arrival times, as the verdict reads them from the devices' readings `ob`. Each
sampling block runs its commit phase, then its challenge phase, then its response phase, every commitment bound
before any of its coins.

* **The commit phase.** The network attester signs the arrival of sampling block `b`'s operand root
  (`opArrive ob b`), and the challenger issues the block's dither coins at `t ob b`, after it: the period is refused
  if `t ob b` came before `opArrive ob b`. So the block's registered instance is read at `opArrive`, before any coin
  of the block. The coins' message is derived from that receipt (the derivation `OwnDither` rests on,
  with the receipt's digest in its context), so the coins can't be issued against an operand root that wasn't in.
* **The challenge phase.** The dither coins go out at least `Δ` before the period's end, and block `b + 1`'s only
  after every root of block `b` is in, so the blocks run one at a time. The network attester watches the replica's
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
def Timed {G : Accel} {P : RoundProto} {nsb : ℕ} {Obs : Type} (verdict : Verdict P nsb Obs) (Pd : Period G) (Δ : ℝ)
    (opArrive t arrive : Obs → Fin nsb → ℝ) (sel : P.Chk → Prop) (τans : ℝ) (tc ansAt : Obs → Fin nsb → ℝ) :
    Verdict P nsb Obs :=
  fun o x tr ob => verdict o x tr ob ∧ (∀ b, 0 ≤ t ob b ∧ t ob b + Δ ≤ Pd.T ∧ arrive ob b ≤ t ob b + Δ) ∧
    (∀ b, opArrive ob b ≤ t ob b) ∧
    (∀ b b' : Fin nsb, (b : ℕ) + 1 = b' → arrive ob b ≤ t ob b') ∧ (∀ b, arrive ob b ≤ tc ob b) ∧
    ∀ b, sel (x b).2 → 0 ≤ tc ob b ∧ tc ob b + τans ≤ Pd.T ∧ ansAt ob b ≤ tc ob b + τans

/-- **An accepted period's sampling blocks' commit phases and work fit the period**: for every prover whose period the
verdict accepts, its sampling blocks' timed use (`Prover.timedUse`: each block's commit phase and work, from the
previous block's work to its commitment) is, resource by resource, at most the period's capacity `Period.cap`. This is what lets the headline count that use
against the period's devices: it was done on them, inside the period. -/
def TimedWorkFits (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (verdict : Verdict P nsb Obs)
    (dev : Readings G P nsb Obs) (Pd : Period G) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk), A.Passes verdict dev o x →
    ∀ r, A.timedUse o x r ≤ Pd.cap r

/-- **The work bound at a verdict**, what the headline concludes of a period under the verdict `verdict` and the
devices' readings `dev`: freed compute beyond the honest optimum is at most `ε_A W` but with probability `κ_A`, prover
by prover (`FreedSound` at λ = 1); every accepted period's sampling blocks' commit phases and work fit the period's
capacity (`TimedWorkFits`); and every accepted period's selected answers ran within the period (`AnswersFit`, at the
selection `sel`). It is one `Prop`, so that the statement that it fails under a verdict that accepts everything is the
negation of it. -/
def WorkBound (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (verdict : Verdict P nsb Obs)
    (dev : Readings G P nsb Obs) (Pd : Period G) (ε : Prover G P nsb → ℝ) (W : ℝ) (κ : Prover G P nsb → ℝ)
    (sel : P.Chk → Prop) : Prop :=
  FreedSound G P nsb verdict dev Pd 1 ε W κ ∧ TimedWorkFits G P nsb verdict dev Pd ∧
    AnswersFit G P nsb verdict dev sel Pd

end Clock

end Pouw.Dither
