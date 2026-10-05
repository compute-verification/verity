import FlockSoundness.Protocol.Discharge.Exec.FlockBatchedL
import FlockSoundness.Protocol.Audit.FlockLink

/-!
# The link theorem: `δ_link` under expected-time collision resistance (DESIGN.md §3)

The batched audit's link term (`analysisBEL`) is discharged here for the plurality value layer,
under expected-time collision resistance of SHA-512 for an explicit finder
(`Assumptions.SHA512CRExpected`) and the binding of wires to commit strings (`ValueBinding`,
named).

* **The binding** (`ValueBinding`): each wire is held by a commit string (a leaf under the serving
  roots, `pos`). A satisfying witness of a drawn unit's table opens the commit string of each of the
  unit's wires, registered at `R`, to a value whose bit on the wire is the decoded one. Two openings
  of one string with different values give a SHA-512 collision. M0's `hm96-sha512` leaf layout
  discharges it.
* **Readings** (`readsAtL`): after an extraction, the values drawn units open a commit string to.
* **The trial** (`trialGL`) for a commit string `c`: a draw that reads `c`, uniform among such draws;
  a fresh session; if the session accepts, `k` waits of at most `Rw` reruns each for a rerun in which
  the session accepts. It reads what the extraction from the runs found reads at `c` (`rdTL`), and
  costs `t'` per run it plays (`cTL`), `t'·(1 + k)` in expectation.
* **The value layer** (`XplurL`) is the plurality of the trial's single readings at each commit
  string. It is a function of the registration state.
* **The finder** for `c` is `LinkFinder.finderG` on the trial. Its successes output collisions
  (`outCL`), so under A2 the trial's link mass at `c` is at most `2t'(1 + k)/2^256 + 1/(eM)`.

A2 is asked of one prover at a time (`LinkCRL`): some prover hard-codes two openings of a commit string, and its
finders find a collision, so no bound holds for every prover under A2. The theorem bounds the link term at each prover
whose finders satisfy A2, and gives `⊤` at any other (`linkBoundCRL`).

The link event at a drawn unit gives a reading other than the plurality at some commit string, so
the audit's link term is at most the ideal link mass, summed over commit strings. The ideal extractor
and the trial differ only by the waits that give up (`k/(eRw)`). Each commit string is weighted by
the probability `π c` that a draw reads it (`flock_batched_linkSoundEL`):
$$\delta_{\rm link}\le\frac{Q_s}{1-\rho}\Big(\frac{2(1+k)\,t'}{2^{256}}+\frac{1}{eM}+
\frac{k}{eR_w}\Big),\qquad Q_s=\sum_c\pi_c,$$
with `ρ ≥ N₀/(ek)` for every table. `Q_s` is the expected number of commit strings the drawn units
read, at most the `N_s` leaves under the serving roots. At `k = ⌈2N₀/e⌉`, `ρ = 1/2`, this is the
record's `t·N_s(8N₀/e)/2^256` with `Q_s` in place of `N_s`, up to lower-order terms.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

variable {L : Law n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSessionL H E Lf plan) R)

/-- The session strategy on draw `ω`. -/
noncomputable abbrev stratωL (ω : L.Ω) : Strategy (sessionBL execArith H E Lf (plan (L.draw ω) R)) :=
  batchStratL H E Lf plan (τ ω)

/-- Everything after `Commit` on draw `ω`. -/
noncomputable abbrev gameωL (ω : L.Ω) : Game (List (OutCL 𝔽 D Sl)) :=
  sessAfterBL (gamesBL execArith H E Lf (plan (L.draw ω) R) (stratωL H E Lf plan R τ ω).1)

/-- The session accepts. -/
abbrev accωL (ω : L.Ω) : List (OutCL 𝔽 D Sl) → Prop := acceptedBL (plan (L.draw ω) R)

/-- The table extracted for drawn unit `u` from the reruns `bs`. -/
noncomputable abbrev tabExtL (ω : L.Ω) (u : Fin n) (bs : List (List (OutCL 𝔽 D Sl))) : Oracle 𝔽 :=
  extractTableSL H E Lf (Partition.tabOf plan tab (L.draw ω) R u).sch (tab (L.draw ω) R u)
    ((stratωL H E Lf plan R τ ω).1 (tab (L.draw ω) R u)) bs

open Classical in
/-- The message the extraction settles on for unit `u`'s table: a close message packing a
satisfying witness, when there is one. -/
noncomputable def candL (ω : L.Ω) (u : Fin n) (bs : List (List (OutCL 𝔽 D Sl))) :
    Option (Message 𝔽 (Partition.tabOf plan tab (L.draw ω) R u).sch.k0
      (Partition.tabOf plan tab (L.draw ω) R u).sch.level₀.logCols) :=
  if h : Committed execArith (Partition.tabOf plan tab (L.draw ω) R u).S (Partition.tabOf plan tab (L.draw ω) R u).sch
      (tabExtL H E Lf plan tab R τ ω u bs) then some (Classical.choose h) else none

/-- An extraction opens `c` to value `v`, with the opening `o`: a drawn unit that reads `c` opens it
so from the message its table settles on. -/
def OpensAtL (ω : L.Ω) (bs : List (List (OutCL 𝔽 D Sl))) (c : vb.Pos) (v : vb.Val) (o : vb.Opening) :
    Prop :=
  ∃ u ∈ L.draw ω, P.Reads plan tab vb u c ∧ ∃ M, candL H E Lf plan tab R τ ω u bs = some M ∧
    o = vb.opening (L.draw ω) R u M c ∧ vb.val o = v

open Classical in
/-- **The readings of commit string `c`** after an extraction from the reruns `bs`. -/
noncomputable def readsAtL (ω : L.Ω) (bs : List (List (OutCL 𝔽 D Sl))) (c : vb.Pos) : Finset vb.Val :=
  Finset.univ.filter fun v => ∃ o, OpensAtL P H E Lf plan tab vb R τ ω bs c v o

/-- A trial's outcome: the draw, the fresh run, and the waits' runs. -/
abbrev TrialOutL (L : Law n) (D Sl : Type) := L.Ω × List (OutCL 𝔽 D Sl) × List (List (List (OutCL 𝔽 D Sl)))

/-- **The trial for commit string `c`**: a draw that reads `c`, a fresh session, and `k` waits of at
most `Rw` reruns each. -/
noncomputable def trialGL (c : vb.Pos) [Nonempty (P.Ωc plan tab vb L c)] (k Rw : ℕ) : Game (TrialOutL L D Sl) :=
  (Game.draw (P.Ωc plan tab vb L c)).bind fun w => (gameωL H E Lf plan R τ w.1).bind fun fresh =>
    (runsG (runsG (gameωL H E Lf plan R τ w.1) Rw) k).map fun bl => (w.1, fresh, bl)

/-- The prover's session strategy in every run. -/
noncomputable def trialStratL (c : vb.Pos) [Nonempty (P.Ωc plan tab vb L c)] (k Rw : ℕ) :
    Strategy (trialGL P H E Lf plan tab vb R τ c k Rw) :=
  Strategy.bind _ _ (drawStrat _) fun w => Strategy.bind _ _ (stratωL H E Lf plan R τ w.1).2 fun _ =>
    Strategy.map _ _ (runsStrat _ (runsStrat _ (stratωL H E Lf plan R τ w.1).2 Rw) k)

open Classical in
/-- **What a trial reads at `c`**: nothing unless the fresh session accepts and every wait finds an
accepted rerun; then the readings of the extraction from those reruns. -/
noncomputable def rdTL (c : vb.Pos) (e : TrialOutL L D Sl) : Finset vb.Val :=
  if accωL plan R e.1 e.2.1 then
    (firsts (accωL plan R e.1) e.2.2).elim ∅ fun bs => readsAtL P H E Lf plan tab vb R τ e.1 bs c
  else ∅

open Classical in
/-- **A trial's cost**: `t'` per run it plays, the fresh one and, if it accepts, each wait's. -/
noncomputable def cTL (t' : ℝ) (e : TrialOutL L D Sl) : ℝ :=
  t' * (1 + if accωL plan R e.1 e.2.1 then (e.2.2.map (waitCost (accωL plan R e.1))).sum else 0)

open Classical in
/-- An opening a trial found for value `v` at `c`, when it found one. -/
noncomputable def opOfVL (c : vb.Pos) (e : TrialOutL L D Sl) (v : vb.Val) : vb.Opening :=
  Classical.epsilon fun o => accωL plan R e.1 e.2.1 ∧
    ∃ bs, firsts (accωL plan R e.1) e.2.2 = some bs ∧ OpensAtL P H E Lf plan tab vb R τ e.1 bs c v o

open Classical in
/-- **The finder's output**: a collision from two openings of `c` with different values, the first
from its first trial, the second from that trial or from the first further trial that reads `c`. -/
noncomputable def outCL (c : vb.Pos) (x : TrialOutL L D Sl × List (TrialOutL L D Sl)) :
    Option (List UInt8 × List UInt8) :=
  if h : ∃ p : TrialOutL L D Sl × vb.Val × vb.Val, p.2.1 ≠ p.2.2 ∧
      p.2.1 ∈ rdTL P H E Lf plan tab vb R τ c x.1 ∧ p.2.2 ∈ rdTL P H E Lf plan tab vb R τ c p.1 ∧
      (p.1 = x.1 ∨ nextRead (rdTL P H E Lf plan tab vb R τ c) x.2 = some p.1) then
    some (vb.collide (opOfVL P H E Lf plan tab vb R τ c x.1 (Classical.choose h).2.1)
      (opOfVL P H E Lf plan tab vb R τ c (Classical.choose h).1 (Classical.choose h).2.2))
  else none

open Classical in
/-- **The plurality value at `c`**: the value the trial reads alone most often. -/
noncomputable def XvalL (k Rw : ℕ) (c : vb.Pos) : vb.Val :=
  if h : Nonempty (P.Ωc plan tab vb L c) then
    haveI := h
    Classical.choose (Finset.exists_max_image Finset.univ
      (pRd (trialGL P H E Lf plan tab vb R τ c k Rw) (trialStratL P H E Lf plan tab vb R τ c k Rw)
        (rdTL P H E Lf plan tab vb R τ c)) Finset.univ_nonempty)
  else default

/-- **The value layer**: each wire's bit in the plurality value of its commit string. -/
noncomputable def XplurL (k Rw : ℕ) : Fin C.N → Bool :=
  fun g => vb.bit (XvalL P H E Lf plan tab vb R τ k Rw (vb.pos g)) g

/-! ## Openings, and the finder's output -/

/-! ## The trial's expectations -/

/-! ## The link mass at one commit string -/

/-! ## The link theorem -/

/-- **A2 for one prover's link finders** (`Assumptions.SHA512CRExpected`): at the registration `R` and the
prover's continuation `τ`, for the finder built for each commit string a draw can read. A2 is a hypothesis about one
finder, since some finder outputs a collision after no evaluations: a prover that hard-codes two openings of one commit
string has link finders that find one. So the link theorem asks A2 only of the finders of the prover it bounds. -/
def LinkCRL (k Rw M : ℕ) (t' : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (P.Ωc plan tab vb L c)), Assumptions.SHA512CRExpected Hc
    (finderG (trialGL P H E Lf plan tab vb R τ c k Rw) M)
    (finderStrat _ (trialStratL P H E Lf plan tab vb R τ c k Rw) M) (outCL P H E Lf plan tab vb R τ c)
    (finderCost (rdTL P H E Lf plan tab vb R τ c) (cTL plan R t'))

end FlockSoundness.Discharge.Exec
