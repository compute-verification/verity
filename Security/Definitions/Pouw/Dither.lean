import Definitions.Pouw.RelativeSecurity

/-!
# Dither's work-soundness: the vocabulary its guarantee reads (trusted)

The final guarantee of the ACCEL program (`Models.Pouw.Guarantees.Dither.DitherWorkSound`) is stated over:

* **the accelerator model** `G = (Σ, M, R, ρ, I, u)` (`Accel`): a machine's programs, the period's devices'
  resources with their data-sheet rates `ρ`, their memory `M`, a run's use of each resource `u`, and the two device
  premises the model is held to, (G1) capacity (`Accel.G1`) and (G2) rate (`Accel.G2`), beside the two arithmetic
  premises (P1) exact operand formats and (P2) a product lives in one accumulator (`Accel.P1`, `Accel.P2`), which
  MODEL-SPEC (CMP-109) defines; until it lands, this structure is the abstraction MODEL-SPEC instantiates;
* **prices** (`Simplex`, `Accel.priced`): a run's cost at a price vector `λ` on the resources, `Σ_r λ_r u_r / ρ_r`,
  in device-seconds when `λ` is a probability vector;
* **a round** of a commit-then-check protocol (`RoundProto`), parametric in the protocol (R-ACCEL-2), with today's
  Dither as one instance (`ditherRound`) whose knobs (`Knobs`: `L`, U's range and placement) are parameters;
* **Theorem O's slot** (`TheoremO`): a statement about the model alone, to be proved in it (O-CHARGE, CMP-111;
  O-BITS, CMP-112), never an assumption about algorithms;
* **a monitoring period's play** round by round (`PeriodPlay`, `PeriodRun`, `PeriodMap`), **ABSORB's slot**
  (`Absorb`: the rounds an accepted run is cut into are pieces of it; a named slot whose standing is Daniel's open
  decision, never an unnamed premise), its priced floors, on the rounds (`RoundsFloor`) and on the whole run
  (`WorkFloor`, `WorkSound`), and the error `epsDither`;
* **freed compute** (`Period`, `Period.FreedWithin`): R-ACCEL-1's definition, kept in this one place.
-/

namespace Pouw.Dither

open Pouw.RelativeSecurity Finset

section Model

/-- **The accelerator model** at the granularity this guarantee reads: a machine `Mc` (its programs, what they compute
and their composition), the period's devices' resources `Res` (tensor cores, ALUs, shared-memory and register ports,
global memory, links), each at its data-sheet rate `rate` (`ρ`, units a second), their memory `mem` (`M`, bits), and
per run its use of each resource `use` (`u`), the bits it holds at its peak `held`, its wall time `wall`, and the bits
it reads from outside the period's devices `ext` (what the network certifier counts). `P1` and `P2` are the arithmetic
premises, exact operand formats and one accumulator per product, as MODEL-SPEC (CMP-109) defines them. `θ = (M, ρ, u)`
are the model's parameters; the guarantee's error reads them only through Theorem O's constant. -/
structure Accel where
  /-- The programs, what they compute and how they compose. -/
  Mc : Machine
  /-- The period's devices' resources. -/
  Res : Type
  [instRes : Fintype Res]
  /-- `ρ`: each resource's data-sheet rate, units a second, over the period's devices. -/
  rate : Res → ℝ
  /-- `M`: the period's devices' memory, bits. -/
  mem : ℝ
  /-- `u`: a run's use of each resource, units. -/
  use : ∀ {I O : Type}, Mc.Alg I O → I → Res → ℝ
  /-- The bits a run holds at its peak. -/
  held : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- A run's wall time, seconds. -/
  wall : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- The bits a run reads from outside the period's devices (imported). -/
  ext : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- (P1) exact operand formats, as MODEL-SPEC defines it. -/
  P1 : Prop
  /-- (P2) a product lives in one accumulator, as MODEL-SPEC defines it. -/
  P2 : Prop

attribute [instance] Accel.instRes

namespace Accel

variable (G : Accel)

/-- **(G1) capacity**: every program, its code and constants with what it holds at its peak, fits in the period's
devices' memory. A table of answers indexed by the live coins is a program, so G1 bounds it. -/
def G1 : Prop := ∀ {I O : Type} (f : G.Mc.Alg I O) (x : I), G.Mc.size f + G.held f x ≤ G.mem

/-- **(G2) rate**: every resource runs at a positive rate, and a run uses each at most at its rate over its wall time. -/
def G2 : Prop :=
  (∀ r, 0 < G.rate r) ∧
    ∀ {I O : Type} (f : G.Mc.Alg I O) (x : I) (r : G.Res), 0 ≤ G.use f x r ∧ G.use f x r ≤ G.rate r * G.wall f x

/-- **A run's priced cost** at the price vector `λ`: `Σ_r λ_r u_r / ρ_r`, the device-seconds it takes at `λ`. At a
point mass on one resource it is that resource's busy time; per-resource counting (tensor-core time alone) is one such
point, and the guarantee holds at every `λ` at once. -/
noncomputable def priced (p : G.Res → ℝ) {I O : Type} (f : G.Mc.Alg I O) (x : I) : ℝ :=
  ∑ r, p r * (G.use f x r / G.rate r)

end Accel

/-- A price vector is a probability vector on the resources. -/
def Simplex {G : Accel} (p : G.Res → ℝ) : Prop := (∀ r, 0 ≤ p r) ∧ ∑ r, p r = 1

end Model

section Round

/-- **A round of a commit-then-check protocol**: an instance the prover committed before the round (`Inst`, the
workload's operands), the round's first live coin `Draw` (σ₁), the transcript the prover commits after it `Tr`, the
check's live coin `Chk` (σ₂), and whether the check passes on the committed transcript (`ok`). The proof layer (C-Flock's
proof that the committed transcript passes, the receipt fold) is the service's, not the round's. -/
structure RoundProto where
  Inst : Type
  Draw : Type
  Chk : Type
  [instDraw : Fintype Draw]
  [instChk : Fintype Chk]
  Tr : Type
  ok : Inst → Draw → Chk → Tr → Prop

attribute [instance] RoundProto.instDraw RoundProto.instChk

/-- A word of the transcript: integers mod `2^32`. -/
abbrev Word : Type := ZMod (2 ^ 32)

/-- **Dither's knobs** (R-ACCEL-2): `A` is `n × m`, `B` is `m × n'` with `m = nb · L`, the prefix stride `L` (32
today), U's range `±1, …, ±u` (64 today) and its placement `place` (`S_tj = U_{j, place t}`; `t mod L` today). -/
structure Knobs where
  n : ℕ
  n' : ℕ
  nb : ℕ
  L : ℕ
  u : ℕ
  place : Fin (nb * L) → Fin L

/-- U's value from its index: `k < u` is `k + 1`, `k ≥ u` is `−(k − u + 1)`. -/
def uVal (u : ℕ) (k : Fin (2 * u)) : Word :=
  if k.val < u then ((k.val + 1 : ℕ) : Word) else -((k.val - u + 1 : ℕ) : Word)

/-- **Dither's prefix word** at cell `(i, j)` and block `b`: `Σ_{t < (b+1)L} Ã_it B̃_tj` mod `2^32`, with
`Ã = A + R` (`R` a 0/1 matrix) and `B̃ = B + S`, `S_tj = U_{j, place t}`. -/
def ditherPrefix (k : Knobs) (a : (Fin k.n → Fin (k.nb * k.L) → Word) × (Fin (k.nb * k.L) → Fin k.n' → Word))
    (d : (Fin k.n → Fin (k.nb * k.L) → Fin 2) × (Fin k.n' → Fin k.L → Fin (2 * k.u)))
    (i : Fin k.n) (j : Fin k.n') (b : Fin k.nb) : Word :=
  ∑ t : Fin (k.nb * k.L), if t.val < (b.val + 1) * k.L then
    (a.1 i t + ((d.1 i t).val : Word)) * (a.2 t j + uVal k.u (d.2 j (k.place t))) else 0

/-- **Today's Dither round, at its knobs**: the instance is the committed operands `(A, B)`, the first coin the dither
`(R, U)`, the transcript every prefix word (`n n' nb` words, `n³/L` for square operands), the check a cell and block,
and the check passes where the committed word there is the prefix word. -/
def ditherRound (k : Knobs) : RoundProto where
  Inst := (Fin k.n → Fin (k.nb * k.L) → Word) × (Fin (k.nb * k.L) → Fin k.n' → Word)
  Draw := (Fin k.n → Fin (k.nb * k.L) → Fin 2) × (Fin k.n' → Fin k.L → Fin (2 * k.u))
  Chk := Fin k.n × Fin k.n' × Fin k.nb
  Tr := Fin k.n → Fin k.n' → Fin k.nb → Word
  ok a d c T := T c.1 c.2.1 c.2.2 = ditherPrefix k a d c.1 c.2.1 c.2.2

/-- **The honest transcript**: every prefix word of the dithered product. -/
def ditherHonest (k : Knobs) (a : (ditherRound k).Inst) (d : (ditherRound k).Draw) : (ditherRound k).Tr :=
  fun i j b => ditherPrefix k a d i j b

end Round

section TheoremO

/-- The uniform average of `f`. -/
noncomputable def avg {Ω : Type} [Fintype Ω] (f : Ω → ℝ) : ℝ := (∑ ω, f ω) / Fintype.card Ω

/-- **A round's shortfall** at the model constant `c`: `(c − (cost + β · imported)/H)₊`, how far a round's priced
cost, with each imported bit worth `β`, falls short of `c` honest rounds' `H`. -/
noncomputable def roundShortfall (c β H cost imp : ℝ) : ℝ := max 0 (c - (cost + β * imp) / H)

open Classical in
/-- **A round program's moment**: the average over the round's coins of `e^{η · shortfall}` where the check passes,
`0` where it fails. The program `M` is fixed before the round's coins and computes the transcript from the first coin
alone, so the transcript is committed before the check is drawn. -/
noncomputable def roundMoment (G : Accel) (P : RoundProto) (p : G.Res → ℝ) (c η β H : ℝ) (a : P.Inst)
    (M : G.Mc.Alg P.Draw P.Tr) : ℝ :=
  avg fun x : P.Draw × P.Chk =>
    if P.ok a x.1 x.2 (G.Mc.eval M x.1) then
      Real.exp (η * roundShortfall c β H (G.priced p M x.1) (G.ext M x.1)) else 0

/-- **Theorem O's slot**: in the accelerator model `G` at its device premises, every round program of the protocol
`P`, on every committed instance, has moment at most 1 at price `λ`, constant `c`, rate `η` and import weight `β`:
`E_σ[1_pass(σ) · e^{η (c − (cost_λ + β · imported)/H)₊}] ≤ 1`.

The program is chosen after the instance, so it may carry as constants anything computed from the instance before the
round (the undithered product `A B`, say), within (G1)'s memory. That precomputation is not in the round's cost, so
Theorem O's constant must hold against it: the rounds' floor counts only what the rounds themselves use.

**This is a theorem to be proved in the model, not an assumption.** It is true or false of `G` and `P` alone, with
nothing about which algorithms anyone runs: O-CHARGE (CMP-111) and O-BITS (CMP-112) prove it, and the constant they
reach is the guarantee's model slack `1 − c`. A round program outputs its transcript, so Theorem O is a counting
statement (per-word costs of each instruction class, and the ring count for the tensor cores), outside ABSORB-THM's
barrier. The barrier sits in whether an accepted run is cut into such programs (`Absorb`). Where Theorem O holds,
`c H` is at most the cost of every program that always passes, so at `H` the honest kernel's priced cost of a round,
`c ≤ 1` is the honest kernel's share of the model's optimum. Theorem K's floor of record (`StagedFloorKOneSided`)
reaches the same period floor on its scope, at the parity luck `η = 0.146` and `c H = St κ L n n' B`
(`ChargeVF3j16`'s charge); outside that scope Theorem O holds on paper only, at `c ≥ 1 − δ` given CoincCharged_θ and
at `c = ¼` without it (VF3e/VF3f price LDS.128 and LDG.128 at 8 MACs a word against the honest kernel's 32). -/
def TheoremO (G : Accel) (P : RoundProto) (p : G.Res → ℝ) (c η β H : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ (a : P.Inst) (M : G.Mc.Alg P.Draw P.Tr), roundMoment G P p c η β H a M ≤ 1

end TheoremO

section Period

/-- **A monitoring period's play**, round by round: per round, the instance committed before it, the round program
(fixed before the round's coins, on the history), then the rest of the play, which may depend on the round's coins. -/
inductive PeriodPlay (G : Accel) (P : RoundProto) : ℕ → Type
  | done : PeriodPlay G P 0
  | round {k : ℕ} (a : P.Inst) (M : G.Mc.Alg P.Draw P.Tr) (next : P.Draw × P.Chk → PeriodPlay G P k) :
      PeriodPlay G P (k + 1)

namespace PeriodPlay

variable {G : Accel} {P : RoundProto}

/-- Every round's check passes. -/
def accepts : {k : ℕ} → PeriodPlay G P k → Paths (P.Draw × P.Chk) k → Prop
  | _, .done, _ => True
  | _, .round a M next, x => P.ok a x.1.1 x.1.2 (G.Mc.eval M x.1.1) ∧ (next x.1).accepts x.2

/-- The rounds' use of a resource, summed along the realized coins. -/
def use : {k : ℕ} → PeriodPlay G P k → Paths (P.Draw × P.Chk) k → G.Res → ℝ
  | _, .done, _, _ => 0
  | _, .round _ M next, x, r => G.use M x.1.1 r + (next x.1).use x.2 r

/-- The rounds' imported bits, summed along the realized coins. -/
def ext : {k : ℕ} → PeriodPlay G P k → Paths (P.Draw × P.Chk) k → ℝ
  | _, .done, _ => 0
  | _, .round _ M next, x => G.ext M x.1.1 + (next x.1).ext x.2

/-- The rounds' priced cost at `λ` along the realized coins: `Σ_r λ_r (Σ_w u_{w,r}) / ρ_r`. -/
noncomputable def priced {k : ℕ} (pl : PeriodPlay G P k) (p : G.Res → ℝ) (x : Paths (P.Draw × P.Chk) k) : ℝ :=
  ∑ r, p r * (pl.use x r / G.rate r)

end PeriodPlay

/-- **An adversary's period as the analysis reads it**: its play, and per round the binding and zero-knowledge break
events of the proof layer. -/
structure PeriodRun (G : Accel) (P : RoundProto) (R : ℕ) where
  play : PeriodPlay G P R
  Bind : Fin R → Paths (P.Draw × P.Chk) R → Prop
  ZK : Fin R → Paths (P.Draw × P.Chk) R → Prop

/-- **The analysis's map from adversaries to their periods**: for each adversary `M` against the service `S` and each
value of its own coins, its run. -/
abbrev PeriodMap (G : Accel) (S : Service) (P : RoundProto) (R : ℕ) : Type 1 :=
  ∀ (Ω' : Type) [Fintype Ω'], G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans) → Ω' → PeriodRun G P R

/-- **ABSORB's slot**: in the accelerator model `G` at its device premises, the analysis `run` cuts every accepted
period of every adversary that commits first into rounds that are pieces of the adversary's run: on every run the
verifier accepts, the rounds' uses of each resource sum to at most the run's, and their imported bits to at most the
run's. The cut is at the commitments, so the rounds are disjoint stretches of the period's devices' time (rounds are
period-wide epochs: round `w`'s transcript is committed before round `w + 1`'s first coin).

**This is ABSORB, the one statement beyond the model's parameters, the cryptography and the live coins that the
guarantee rests on, and its standing is Daniel's open decision.** What is to hold is that some analysis map meets it
beside `Models.Pouw.Assumptions.ServiceSoundDither`: every accepted run computes the transcripts it commits to (through
the hash and the fold) within its own resources. It is a statement over every program of `G`, true or false of `G`,
the service and the protocol. ABSORB-THM's Theorem B shows that proving it over the full instruction set would be an
explicit superlinear circuit lower bound. It may become a named computational assumption, follow from SHA-512 as a
random oracle at a cost in overhead, or be dropped for the unconditional floor alone. Until Daniel rules, it is a
slot, never an unnamed premise. It carries the floor's whole link to the adversary: without it, an analysis could cut
any accepted run into honest rounds, and the floor would read nothing of the run. This is ABSORB's cut form. It
implies the cost form (every committed word is written into a register) wherever the model charges a program's output
words, and it is the weakest form this route needs, because Theorem O reads a round program's whole cost. -/
def Absorb (G : Accel) (S : Service) (P : RoundProto) {R : ℕ} (run : PeriodMap G S P R)
    (e : S.Seed ≃ Paths (P.Draw × P.Chk) R) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 →
    ∀ (Ω' : Type) [Fintype Ω'] (M : G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
      (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
        S.verify (G.Mc.eval M (e.symm x, r)).1 (e.symm x) (G.Mc.eval M (e.symm x, r)).2 = true →
          (∀ q, (run Ω' M r).play.use x q ≤ G.use M (e.symm x, r) q) ∧
            (run Ω' M r).play.ext x ≤ G.ext M (e.symm x, r)

/-- **The priced floor, adversary by adversary**: every adversary that commits first has its accepted runs priced
below `F` at `λ` (its size or its priced cost, whichever is larger) with probability at most `κ Ω M`.
`ServiceFloorPerProver` with the run's cost read at `λ`. -/
def WorkFloor (G : Accel) (S : Service) (p : G.Res → ℝ) (F : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)),
    (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) →
    ENNReal.ofReal (prob fun ω : S.Seed × Ω => S.verify (G.Mc.eval M ω).1 ω.1 (G.Mc.eval M ω).2 = true ∧
      max (G.Mc.size M) (G.priced p M ω) < F) ≤ κ Ω M

/-- **The rounds' priced floor, adversary by adversary**: every adversary that commits first has its accepted periods
whose rounds (the credited part, as the analysis `run` cuts it out of the adversary's run, along the period's coins
`e`) are priced below `F` at `λ` with probability at most `κ Ω M`. The rest of the adversary's run is its other work,
so this is the floor freed compute reads (`Period.FreedWithin`); it implies `WorkFloor` wherever the rounds' use is at
most the run's. -/
def RoundsFloor (G : Accel) (S : Service) (P : RoundProto) {R : ℕ} (run : PeriodMap G S P R)
    (e : S.Seed ≃ Paths (P.Draw × P.Chk) R) (p : G.Res → ℝ) (F : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)),
    (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) →
    ENNReal.ofReal (prob fun ω : S.Seed × Ω => S.verify (G.Mc.eval M ω).1 ω.1 (G.Mc.eval M ω).2 = true ∧
      (run Ω M ω.2).play.priced p (e ω.1) < F) ≤ κ Ω M

/-- **`(1 − ε)`-work-soundness** at `λ` against the honest period's priced cost `Hp`: accepted runs priced below
`(1 − ε) Hp` have probability at most `κ`. -/
def WorkSound (G : Accel) (S : Service) (p : G.Res → ℝ) (Hp ε : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal) : Prop :=
  WorkFloor G S p ((1 - ε) * Hp) κ

/-- **Dither's error over a monitoring period of `k` rounds**: `ε = (1 − c) + D/k + β K/(k H)`, the model slack
(Theorem O's constant), the sampling luck (`D = ln(1/δ)/η` rounds' worth of shortfall at luck `e^{−η D} = δ`), and the
network import (`K` bits at most, the network certifier's, each worth `β`). -/
noncomputable def epsDither (c D β K H : ℝ) (k : ℕ) : ℝ := (1 - c) + D / k + β * K / (k * H)

end Period

section Freed

/-- **A monitoring period's devices and the other work they could do**, in resource units: the period's length `T`
(seconds), the other jobs `J`, each with its value `val` per unit and its use `needs` per unit, and the honest use
`hon` per unit of credited work (the honest kernel's, Dither included). -/
structure Period (G : Accel) where
  T : ℝ
  J : Type
  [instJ : Fintype J]
  val : J → ℝ
  needs : J → G.Res → ℝ
  hon : G.Res → ℝ

attribute [instance] Period.instJ

namespace Period

variable {G : Accel} (Pd : Period G)

/-- What the devices can do in the period (G2): `ρ_r T` of resource `r`. -/
def cap (r : G.Res) : ℝ := G.rate r * Pd.T

/-- A strategy fits the devices: its credited part (its rounds, `PeriodPlay.use`) uses `X`, its other work `z ≥ 0`
units of each job, and the two together use no resource past its capacity. -/
def Fits (X : G.Res → ℝ) (z : Pd.J → ℝ) : Prop := (∀ j, 0 ≤ z j) ∧ ∀ r, X r + ∑ j, z j * Pd.needs j r ≤ Pd.cap r

/-- A strategy's value: its credited work `W` plus its other work's. -/
def value (W : ℝ) (z : Pd.J → ℝ) : ℝ := W + ∑ j, Pd.val j * z j

/-- **Freed compute is at most `ε C`** (R-ACCEL-1, Daniel 25 Oct): a strategy credited `W` with other work `z` gets
no more credited work plus other work out of the period's devices than an honest strategy on the same devices, plus
`ε C`. An honest strategy is credited `W'` at the honest use `W' · hon` and does other work `z'` beside it. Its dual is
a price vector on the resources (`Models.Pouw.Guarantees.Dither.FreedOfFloor`). This is the definition's one place. -/
def FreedWithin (W : ℝ) (z : Pd.J → ℝ) (ε C : ℝ) : Prop :=
  ∃ (W' : ℝ) (z' : Pd.J → ℝ), 0 ≤ W' ∧ Pd.Fits (fun r => W' * Pd.hon r) z' ∧ Pd.value W z ≤ Pd.value W' z' + ε * C

end Period

end Freed

end Pouw.Dither
