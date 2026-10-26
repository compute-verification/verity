import Definitions.Pouw.Dither
import Definitions.Core.Accel.Principles

/-!
# The headline's model on the cards: the H200 and the RTX PRO 6000 at θ (trusted)

`Definitions.Pouw.Guarantees.Dither.WorkSound` is stated over an abstract accelerator `Accel`. This file ties that
`Accel` to MODEL-SPEC's model (`Definitions.Core.Accel`) at a card's θ, so the headline can be read on
`Verity.Accel.H200.card` and `Verity.Accel.Sm120.card` (`Definitions.Pouw.Guarantees.DitherInstance`):

* **the period's devices** (`Devices`): `n` cards of one table pooled into one machine (`poolCard`, a coarsening:
  the pooled cards share their stores and resources, so the links between them are free), at `f` ticks a second, and
  the period's other devices (`Others`: the host CPUs and DPUs the cards reach without crossing a network certifier,
  RT-PROTO P9, RT-UNITS U9), each resource at its θ rate;
* **where a round's coin and words sit in the cards' memory** (`Layout`);
* **`G` is the devices at the open parameters `x`** (`OnCard`): `G`'s resources are the pooled cards' and the others',
  at θ's rates; its memory is theirs; its (P1) and (P2) are MODEL-SPEC's; and every words program the cards run is a
  program of `G` with the run's units as its use, its ticks and backlog as its wall time and no import;
* **the public best methods** (`PubBest`), fixed per card and priced at the headline's dual price (`pubCost`), with
  placeholders until D5;
* **the network certifier's counts** (`Counted`, RT-PROTO P2): the verifier refuses a period whose signed counts into
  or out of the devices pass the caps `E_in`, `E_out`;
* **the auditor's clocks** (`Timed`, RT-PROTO P5): the verifier refuses a period in which the network certifier saw
  some round's commitment arrive more than `Δ` after the challenger's coin, and draws every round's coin at least `Δ`
  before the period's end; `TimedWorkFits` is what that buys, the rounds' work within the period's capacity.

The open parameters `x` are not fixed here: the headline holds at every `x` meeting θ's floors (`Card.floors`: on the
H200 `sync ≥ 1`, `b1 ≥ 1/8`, `madHi ≥ 2`, `slack ≥ 1`; on sm_120 `madHi ≥ 2` and no native `b1`, RT-MODEL M12, M13) with the
product form (`Assumptions.AccelProductForm`). Without the floors the model runs an `IMAD.HI`, a `b1` MAC or an
`mma.sync` MAC for nothing, `ShaSpeed` at a card's rates is false in it, and the headline says nothing.
-/

open scoped NNRat

namespace Pouw.Dither

open Pouw.RelativeSecurity

section Cards

instance instFintypeAccelRes : Fintype Verity.Accel.Res where
  elems := {.tensor, .tensorLong, .issue, .alu, .imad, .shfl, .f64, .smem, .dram, .l2}
  complete r := by cases r <;> simp

/-- **`n` cards pooled into one**: every store and every resource `n` times the card's, the code space too, and the
same formats, opcodes, dots and floors. A coarsening, as the card's own pools over its SMs are (`Accel.Card`): any
thread names any card's locations, so the links between the cards (NVLink, PCIe peer-to-peer) cost nothing, and a
lower bound in the pool holds on the cards. -/
def poolCard (c : Verity.Accel.Card) (n : ℕ) (hn : 0 < n) : Verity.Accel.Card :=
  { c with
    size := fun s => n * c.size s
    size_pos := fun s => Nat.mul_pos hn (c.size_pos s)
    ρ := fun r => (n : ℚ≥0) * c.ρ r
    text := n * c.text
    tc := n * c.tc
    cuda := n * c.cuda
    bw := n * c.bw }

/-- **The period's other devices** (RT-PROTO P9, RT-UNITS U9): every processor the period's cards reach without
crossing a network certifier other than the cards themselves (host CPUs over PCIe or NVLink-C2C, DPUs), each
resource with its θ rate in units a second (RT-MODEL's per-node rows: SHA-512 and INT8 rates, DRAM, PCIe, C2C) and
their memory in bits. Their programs are `Accel`'s like the cards', priced at the period's dual price, so the honest
optimum may use them and the capacity `C` includes them. That the list is complete is the inventory premise
(`Assumptions.PeriodInventory`). -/
structure Others where
  Res : Type
  [instRes : Fintype Res]
  rate : Res → ℝ
  mem : ℝ

attribute [instance] Others.instRes

/-- **The period's devices**: `n` cards of the table `c`, a tick `1/f` seconds (the clock the table's per-tick rates
are at; the fleet's transfer reads CALIB's θ, #4513 `clocks.fmax`: 1980 MHz on the H200, 2430 MHz on the RTX PRO
6000), and the other devices. The headline holds at every clock. -/
structure Devices (c : Verity.Accel.Card) where
  n : ℕ
  n_pos : 0 < n
  f : ℝ
  oth : Others

/-- The period's cards as one machine at the open parameters `x` (no side channels: what enters from outside is the
network certifier's count, `Accel.ext`). -/
def Devices.machine {c : Verity.Accel.Card} (D : Devices c) (x : Verity.Accel.Open) : Verity.Accel.Machine Empty :=
  (poolCard c D.n D.n_pos).machine x (fun e => e.elim) 0

/-- **Where a round's coin and words sit in the cards' memory**: the dither coin written into the memory a words
program starts from (`putDraw`), and the words read from the memory it ends with (`getTr`), as the deployed kernel lays
them out. A layout moves words; it computes nothing. The headline holds at every layout. -/
structure Layout (m : Verity.Accel.Machine Empty) (P : RoundProto) where
  putDraw : P.Draw → m.Mem → m.Mem
  getTr : m.Mem → P.Tr

variable {ι : Type}

/-- A run of `Prg` on the schedule `sched` from the memory `M`, every thread at its text's start. -/
def cardRun (m : Verity.Accel.Machine ι) (Prg : m.Prog ℕ) (M : m.Mem) (sched : List (List ℕ)) :
    m.State ℕ × List (List m.Instr) :=
  Verity.Accel.Machine.run Prg ⟨M, fun _ => 0⟩ sched

/-- The units of resource `r` a run's instructions use over its ticks. -/
def cardUnits (m : Verity.Accel.Machine ι) (iss : List (List m.Instr)) (r : m.Res) : ℚ≥0 :=
  ∑ t ∈ Finset.range iss.length, Verity.Accel.Machine.usage iss t r

/-- **`G` is the period's devices at the open parameters `x`** (RT-LEAN L4, L9, L31): under the equivalence `e` of
`G`'s resources with the pooled cards' and the other devices',

* each card resource runs at `n ρ_r` units a tick, `n ρ_r f` a second, and each other device's at its θ rate;
* `G`'s memory is the pooled cards' bits (on-chip and device memory) and the other devices';
* `G`'s (P1) is MODEL-SPEC's integer path with the card's captured dots in product form, and its (P2) MODEL-SPEC's four
  data-flow properties (`Definitions.Core.Accel.Principles`);
* every words program the cards run is one of `G`'s: for every memory `M₀` the round starts from (whatever the prover
  carried into it) and every program and schedule of the pooled machine meeting (G1) and (G2) from `M₀` with the coin
  written in, some words program of `G` computes the words the run leaves, uses exactly the run's units on each card
  resource and nothing of the other devices, takes the run's ticks and its backlog as wall time, and imports nothing.

So `SAKMatmul` at `G` speaks of every words program of the card at `x`, at the card's rates; the dual price, the honest
cost `H` and the capacity `C` are in the card's units. `G.ext` is the bits a run imports, which the network certifier
counts (`Assumptions.CertifierSeesImports`). The commitment and the round's other programs read the oracle
`P.Orc`, which the card machine has no instruction for: their reading on the card is ABSORB-THM's realization (an oracle
query priced at the public best's speed, R-ACCEL-7), open. A program of `G` cheaper than any of the card's makes
`SAKMatmul` or `ShaSpeed` false at `G`, and the headline then says nothing at that `G`; the `G` the headline is read at
has the card's programs and no others. -/
def OnCard (G : Accel) (c : Verity.Accel.Card) (x : Verity.Accel.Open) (D : Devices c) (P : RoundProto)
    (L : Layout (D.machine x) P) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) : Prop :=
    (∀ r, G.rate (e.symm (.inl r)) = ((poolCard c D.n D.n_pos).ρ r : ℝ) * D.f) ∧
    (∀ h, G.rate (e.symm (.inr h)) = D.oth.rate h) ∧
    G.mem = (((D.machine x).chip + (D.machine x).mem : ℕ) : ℝ) + D.oth.mem ∧
    (G.P1 ↔ Verity.Accel.P1Int ∧ Verity.Accel.CapturedProductForm c) ∧
    (G.P2 ↔ Verity.Accel.P2Local ∧ Verity.Accel.P2Distinct ∧ Verity.Accel.P2Frame ∧ Verity.Accel.P2Priced) ∧
    ∀ (M₀ : (D.machine x).Mem) (Prg : (D.machine x).Prog ℕ) (sched : P.Draw → List (List ℕ)),
      (∀ d, Verity.Accel.Machine.Fits Prg ⟨L.putDraw d M₀, fun _ => 0⟩ (sched d)) →
      ∃ Mw : G.Mc.Alg ((D.machine x).Mem × P.Draw) P.Tr, ∀ d,
        G.Mc.eval Mw (M₀, d) = L.getTr (cardRun (D.machine x) Prg (L.putDraw d M₀) (sched d)).1.mem ∧
        (∀ r, G.use Mw (M₀, d) (e.symm (.inl r)) =
          (cardUnits (D.machine x) (cardRun (D.machine x) Prg (L.putDraw d M₀) (sched d)).2 r : ℝ)) ∧
        (∀ h, G.use Mw (M₀, d) (e.symm (.inr h)) = 0) ∧
        G.wall Mw (M₀, d) = (((sched d).length + x.slack : ℕ) : ℝ) / D.f ∧
        G.ext Mw (M₀, d) = 0

/-- **The public best methods on the card table `c`** (RT-LEAN L3, L27, L31): the units, resource by resource, that the
public best matmul method takes for one round's words (`mm`) and that the pinned public SHA-512 implementation takes
for the round's commitment (`sha`), each a named public program run on the card, at the cheapest instance a prover can
register (the coordinator's L27 decision: one constant must hold at every instance the prover may pick). The headline
prices them at its dual price (`pubCost`), so `pub_mm` and `pub_sha` are fixed per card, not chosen per reading. -/
structure PubBest (c : Verity.Accel.Card) where
  mm : Verity.Accel.Res → ℚ≥0
  sha : Verity.Accel.Res → ℚ≥0

/-- Units on the card's resources priced at `μ` through the resource match `e`. -/
noncomputable def pubCost {G : Accel} {c : Verity.Accel.Card} {D : Devices c}
    (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (μ : G.Res → ℝ) (u : Verity.Accel.Res → ℚ≥0) : ℝ :=
  ∑ r, μ (e.symm (.inl r)) * (u r : ℝ)

/-- **Placeholder until D5 (12:00Z)**: the H200's public best methods, zero units, so `c_mm = c_sha = 0` and the
headline's model term is 1 until D5 names the programs and their units. -/
def pubBestH200Placeholder : PubBest Verity.Accel.H200.card := ⟨0, 0⟩

/-- **Placeholder until D5 (12:00Z)**: the RTX PRO 6000's public best methods, zero units, as on the H200. -/
def pubBestSm120Placeholder : PubBest Verity.Accel.Sm120.card := ⟨0, 0⟩

end Cards

section Certifier

/-- A count the network certifier signs over a prover's period, at the oracle and the verifier's coins. -/
abbrev Count (G : Accel) (P : RoundProto) (k : ℕ) : Type 1 :=
  Prover G P k → P.Orc → (Fin k → P.Draw × P.Chk) → ℝ

/-- **The verdict with the network certifier's counts** (RT-PROTO P2, RT-LEAN L32): the service's verdict, and the
certifier's signed counts of the bits into (`cin`) and out of (`cout`) the period's devices are within the caps
`E_in` and `E_out`, fixed before the period; a period past either is refused. -/
def Counted {G : Accel} {P : RoundProto} {k : ℕ} (verdict : Verdict G P k) (cin cout : Count G P k)
    (Ein Eout : ℝ) : Verdict G P k :=
  fun A o x => verdict A o x ∧ cin A o x ≤ Ein ∧ cout A o x ≤ Eout

end Certifier

section Clock

/-- The times, by round, at which the network certifier saw each round's commitment arrive, over a prover's period at
the oracle and the verifier's coins: its signed clock, an auditor device. -/
abbrev Arrival (G : Accel) (P : RoundProto) (k : ℕ) : Type 1 :=
  Prover G P k → P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → ℝ

/-- **The verdict with the auditor's clocks** (RT-PROTO P5, O-REDTEAM NF5, RT-LEAN L33): the service's verdict, the
challenger's schedule `t` draws every round's dither coin at least `Δ` before the period's end, and the period is
refused if the network certifier saw some round's commitment arrive (`arrive`) more than `Δ` after that round's coin.
Both clocks are the auditor's; nothing here reads a time the prover reports. -/
def Timed {G : Accel} {P : RoundProto} {k : ℕ} (verdict : Verdict G P k) (Pd : Period G) (Δ : ℝ)
    (t : Fin k → ℝ) (arrive : Arrival G P k) : Verdict G P k :=
  fun A o x => verdict A o x ∧ (∀ w, 0 ≤ t w ∧ t w + Δ ≤ Pd.T) ∧ ∀ w, arrive A o x w ≤ t w + Δ

/-- **An accepted period's rounds' work fits the period**: for every prover the verdict accepts, the work its rounds
do from coin to commitment uses, resource by resource, at most the period's capacity `Period.cap`. This is what lets
the headline count that work against the period's devices: it was done on them, inside the period. -/
def TimedWorkFits {G : Accel} {P : RoundProto} {k : ℕ} (verdict : Verdict G P k) (Pd : Period G) : Prop :=
  ∀ (A : Prover G P k) (o : P.Orc) (x : Fin k → P.Draw × P.Chk), verdict A o x →
    ∀ r, ∑ w, G.use (A.work w) (A.workIn o x w) r ≤ Pd.cap r

end Clock

end Pouw.Dither
