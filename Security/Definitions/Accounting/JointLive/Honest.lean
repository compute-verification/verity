import Definitions.Accounting.JointLive
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise

/-!
# The honest deployed horizon over a causal game, and an idle one whose verdicts accept everything (trusted)

The instances the joint guarantee's honest and nontrivial lemmas are stated at: every security guarantee has an honest
lemma (its hypotheses hold at the honest deployed instance) and a nontrivial one (its conclusion fails for a verifier
that accepts everything).

The device (`h200`) is one H200 of `vy-dc-h200-2`, at a tick of one nanosecond, with one side channel, its PCIe link:

* on-chip state: `32 · 31,001,912` bits, the register file (with its uniform registers and predicates), shared memory
  and L2 of `Verity.Accel.H200.size`;
* device memory: `8 · 150,109,880,320` bits, the HBM the device reports (`prover_store.total_bytes` of
  `r20261026-003454-bfd2`'s honest wipe);
* bandwidth: the H200 data sheet's 4.8 TB/s, `38,400` bits a nanosecond;
* the op rates and the link's rate: parameters, which the honest node runs at in full.

Every block is `L` ticks and is opened by the honest wipe of `r20261026-003454-bfd2` (`fill`: 2,319,974,400 labels of
512 bits), which leaves no foreign content (`kept = 0`). Its verdict accepted and published `foreign_bits` as the whole
fill, uncertified: its egress count, 32,767 MiB, is outside the 0 to 32,766 MiB it bounds, so it bounds nothing. Once the
count is in range the verdict publishes the streamed wipe's `⌈fill/19⌉` plus the egress.
PoUS's store (`weights`) is P2's deployed point, `2^19` blocks of 2,056 bytes, challenged at every tick against P2's
read deadline `Δ + RTT = 854,000` ns. The commitment opens each gate to the circuit's evaluation on the inputs, which
is what the extractor reads; the finders output nothing; every computed gate is computed after its coin; the network
certifiers saw the node move `E` bits; and every verdict accepts. The game is any causal game: neither node's world
reads the strategy or the run.

* **The honest node** (`honestPeriod`, `honestBlocks`) runs every op class at the device's rate on the compute
  account, reads device memory at the full bandwidth on the bandwidth account, and holds PoUS's store on the capacity
  account. Its certificates (`certs`) are what it does: no unbound unit, every computed gate, the store held, those
  reads, and `E` bits moved.
* **The idle node** (`idlePeriod`, `idleBlocks`) does nothing, on the same blocks, under the same verdicts, at the same
  certificates.

The remote bandwidth audit's honest session is `r20261026-180902-bc3d`'s (`BW VERDICT accept q_bytes=275963022560
receipts=34013184 wrong_at_most=457369 span_s=121.966 credited_GB_s=2.263`): `sessionBits` credited as read over
`sessionSpan` ticks, at the certificates `sessionCerts`.
-/

namespace Accounting.Honest

open Verity

/-- The deployed device: one H200 of `vy-dc-h200-2` at a tick of one nanosecond, with one side channel (its PCIe link),
at op rates `rate` and the link's rate `cap`. -/
def h200 (rate : Op → ℕ) (cap : Unit → ℕ) : Device Unit where
  chip := 32 * (132 * (65536 + 4494) + 132 * 65536 + 50 * 2 ^ 20 / 4)
  mem := 8 * 150109880320
  rate := rate
  bw := 38400
  cap := cap

/-- The bits of PoUS's store at P2's deployed point (`p2-p16448-c21065`): `2^19` blocks of 2,056 bytes. -/
def weights : ℕ := 8 * 2056 * 2 ^ 19

/-- The bits the honest wipe of `r20261026-003454-bfd2` filled: 2,319,974,400 labels of 512 bits. -/
def fill : ℕ := 512 * 2319974400

/-- P2's read deadline, `Δ + RTT = 854,000` ns. -/
def readDeadline : ℕ := 854000

/-- The bits the honest remote bandwidth session of `r20261026-180902-bc3d` credited as read: `8 · q_bytes`. -/
def sessionBits : ℕ := 8 * 275963022560

/-- That session's span, in ticks (`span_ns`). -/
def sessionSpan : ℕ := 121965969250

/-- The accounts: compute (0), bandwidth (1) and capacity (2). -/
def acct : Resource → Fin 3
  | .compute => 0
  | .bandwidth => 1
  | .capacity => 2

/-- The honest node's step on device `d`: every op class at its rate on the compute account, the full bandwidth on the
bandwidth account, and PoUS's store on the capacity account. -/
def step (d : Device Unit) : Step Unit (Fin 3) where
  ops a o := if a = 0 then d.rate o else 0
  read a := if a = 1 then d.bw else 0
  side _ _ := 0
  chip _ := 0
  mem a := if a = 2 then weights else 0

/-- The idle node's step: nothing. -/
def idle : Step Unit (Fin 3) where
  ops _ _ := 0
  read _ := 0
  side _ _ := 0
  chip _ := 0
  mem _ := 0

/-- Block `b` of `L` ticks on the deployed device, read off any run of any strategy of the causal game `G`, over the
circuit `C` partitioned by `P` on the inputs `v`, the node taking the step `st` at every tick and the network
certifiers seeing it move `E` bits: the commitment opens each gate to the evaluation's value there, the extractor reads
the evaluation, the finders output nothing, every computed gate is computed after its coin, and every verdict
accepts. -/
def period (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    {O : Type} (G : Game O) (L E b : ℕ) (st : Step Unit (Fin 3)) (fits : st.Fits (h200 rate cap)) :
    PeriodLive (h200 rate cap) (Fin 3) C G where
  units := n
  parts := P
  opens g x := x = C.eval v g
  extract _ _ := C.eval v
  finder _ _ := ⟨Unit, fun _ => none, fun _ => 0⟩
  computed _ _ := C.computed
  world _ _ :=
    { acct := acct
      win := ⟨b * L, L⟩
      schedule := fun _ => st
      fits := fun _ => fits
      commWithin := fun K => (E : ℝ) ≤ K }
  accepts _ _ := True

/-- **The honest node's block** `b` of `L` ticks. -/
def honestPeriod (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    {O : Type} (G : Game O) (L E b : ℕ) : PeriodLive (h200 rate cap) (Fin 3) C G :=
  period rate cap P v G L E b (step (h200 rate cap)) (by simp [Step.Fits, step, h200, weights])

/-- **The idle node's block** `b` of `L` ticks, under verdicts that accept everything. -/
def idlePeriod (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    {O : Type} (G : Game O) (L E b : ℕ) : PeriodLive (h200 rate cap) (Fin 3) C G :=
  period rate cap P v G L E b idle (by simp [Step.Fits, idle])

/-- The honest wipe of `r20261026-003454-bfd2`, opening a block: accepted, publishing the foreign bits it published
(`foreign_bits = 1,187,826,892,800`, the whole fill, since its egress count bounds nothing), and leaving no foreign
content. -/
def honestWipe (rate : Op → ℕ) (cap : Unit → ℕ) (O : Type) : WipeLive (h200 rate cap) O where
  fill := fill
  fill_le := by
    show fill ≤ 8 * 150109880320
    norm_num [fill]
  accepts _ := True
  foreignBits _ := fill
  kept _ := 0

/-- `B` blocks of `L` ticks on the deployed device, block `b` being `blk b`, each opened by the honest wipe, with
PoUS's store challenged at every tick of the block. -/
def blocks (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {O : Type} (G : Game O) (B L : ℕ)
    (blk : Fin B → PeriodLive (h200 rate cap) (Fin 3) C G) : BlocksLive (h200 rate cap) (Fin 3) B (fun _ => C) G where
  block := blk
  wipe _ := honestWipe rate cap O
  weights _ := weights
  chal b _ := (⟨(b : ℕ) * L, L⟩ : Window).ticks
  readDeadline := readDeadline

/-- **The honest deployed horizon**: `B` blocks of `L` ticks of the honest node. -/
def honestBlocks (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    {O : Type} (G : Game O) (B L E : ℕ) : BlocksLive (h200 rate cap) (Fin 3) B (fun _ => C) G :=
  blocks rate cap G B L fun b => honestPeriod rate cap P v G L E b

/-- **The idle horizon**: `B` blocks of `L` ticks of a node that does nothing, under verdicts that accept everything. -/
def idleBlocks (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n) (v : C.Transcript)
    {O : Type} (G : Game O) (B L E : ℕ) : BlocksLive (h200 rate cap) (Fin 3) B (fun _ => C) G :=
  blocks rate cap G B L fun b => idlePeriod rate cap P v G L E b

/-- The honest node's certificates for a block of `L` ticks over the circuit `C`: no unbound unit, every computed gate,
PoUS's store held, every bit of the bandwidth read, and the `E` bits moved. -/
def certs (C : Circuit) (L E : ℕ) : Certificates where
  ε := 0
  W := C.size C.computed
  S := weights
  Q := ((L * 38400 : ℕ) : ℝ)
  K := E

/-- The certificates of a block as long as the remote bandwidth session, with the bandwidth certificate the bits the
session credited as read. -/
def sessionCerts (C : Circuit) (E : ℕ) : Certificates :=
  { certs C sessionSpan E with Q := sessionBits }

end Accounting.Honest
