import FlockSoundness.Protocol.Discharge.ZkLink.Compiled
import FlockSoundness.Protocol.Discharge.ZkLink.Link

/-!
# `δ_link` for the compiled `--zk` session, under expected-time collision resistance

`Discharge/Exec/FlockLinkL.lean`'s link theorem for the compiled `--zk` session (`batchedSessionZC`), whose extractor
fills each drawn unit's level-0 oracle (68 lanes: the witness lanes and both reps' extra lanes) from the verifying
openings of `k` accepted reruns (`extractTableZ`). It holds at the compiled analysis `analysisZC` at any knowledge bound,
since the link term does not read it, under expected-time collision resistance of SHA-512 for an explicit finder
(`Assumptions.SHA512CRExpected`) and the binding of wires to commit strings (`ValueBindingZ`, named):

* **Readings** (`readsAtZC`): after an extraction, the values drawn units open a commit string to, from the padded
  witness each one's table settles on (`candZC`).
* **The trial** (`trialGZC`) for a commit string `c`: a draw that reads `c`, uniform among such draws; a fresh compiled
  session; if it accepts, `k` waits of at most `Rw` reruns each for a rerun in which it accepts. It reads what the
  extraction from the runs found reads at `c` (`rdTZC`), and costs `t'` per run it plays (`cTZC`), `t'(1 + k)` in
  expectation.
* **The value layer** (`XplurZC`) is the plurality of the trial's single readings at each commit string.
* **The finder** for `c` is `LinkFinder.finderG` on the trial. Its successes output collisions (`outCZC_spec`), so under
  A2 the trial's link mass at `c` is at most `2t'(1 + k)/2^256 + 1/(eM)`.

A2 is asked of one prover at a time (`LinkCRZC`), and the bound is `⊤` at any other (`linkBoundZC`). With `π c` the
probability that a draw reads `c` (`zk_linkSoundC`):
$$\delta_{\rm link}\le\frac{Q_s}{1-\rho}\Big(\frac{2(1+k)\,t'}{2^{256}}+\frac{1}{eM}+
\frac{k}{eR_w}\Big),\qquad Q_s=\sum_c\pi_c,$$
with `ρ ≥ N₀/(ek)` for every table, `N₀` its level-0 positions.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

/-- The compiled session strategy on draw `ω`. -/
noncomputable abbrev stratωZC (ω : L.Ω) : Strategy (sessionZC A Hs (planZ (L.draw ω) R)) :=
  zcStrat A Hs planZ (τ ω)

/-- Everything after `Commit` on draw `ω`. -/
noncomputable abbrev gameωZC (ω : L.Ω) : Game (List (OutZC F D Sl)) :=
  sessAfterZC (gamesZC A Hs (planZ (L.draw ω) R) (stratωZC A Hs planZ R τ ω).1)

/-- The compiled session accepts. -/
abbrev accωZC (ω : L.Ω) : List (OutZC F D Sl) → Prop := acceptedZC (planZ (L.draw ω) R)

/-- The level-0 oracle extracted for drawn unit `u`'s table from the reruns `bs`. -/
noncomputable abbrev tabExtZC (ω : L.Ω) (u : Fin n) (bs : List (List (OutZC F D Sl))) : Oracle F :=
  extractTableZ Hs (tabOfZ planZ tab (L.draw ω) R u) (tab (L.draw ω) R u)
    ((stratωZC A Hs planZ R τ ω).1 (tab (L.draw ω) R u)) bs

open Classical in
/-- The padded witness the extraction settles on for unit `u`'s table: one near the extracted oracle that satisfies the
statement, when there is one. -/
noncomputable def candZC (ω : L.Ω) (u : Fin n) (bs : List (List (OutZC F D Sl))) :
    Option (Message F (tabOfZ planZ tab (L.draw ω) R u).sch.k0 (tabOfZ planZ tab (L.draw ω) R u).sch.level₀.logCols) :=
  if h : (tabOfZ planZ tab (L.draw ω) R u).Committed A (tabExtZC A Hs planZ tab R τ ω u bs) then
    some (Classical.choose h) else none

/-- An extraction opens `c` to value `v`, with the opening `o`: a drawn unit that reads `c` opens it
so from the message its table settles on. -/
def OpensAtZC (ω : L.Ω) (bs : List (List (OutZC F D Sl))) (c : vb.Pos) (v : vb.Val) (o : vb.Opening) :
    Prop :=
  ∃ u ∈ L.draw ω, ReadsZ A P planZ tab vb u c ∧ ∃ M, candZC A Hs planZ tab R τ ω u bs = some M ∧
    o = vb.opening (L.draw ω) R u M c ∧ vb.val o = v

open Classical in
/-- **The readings of commit string `c`** after an extraction from the reruns `bs`. -/
noncomputable def readsAtZC (ω : L.Ω) (bs : List (List (OutZC F D Sl))) (c : vb.Pos) : Finset vb.Val :=
  Finset.univ.filter fun v => ∃ o, OpensAtZC A Hs P planZ tab vb R τ ω bs c v o

/-- A trial's outcome: the draw, the fresh run, and the waits' runs. -/
abbrev TrialOutZC (F : Type) (L : Law n) (D Sl : Type) := L.Ω × List (OutZC F D Sl) × List (List (List (OutZC F D Sl)))

/-- **The trial for commit string `c`**: a draw that reads `c`, a fresh session, and `k` waits of at
most `Rw` reruns each. -/
noncomputable def trialGZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) : Game (TrialOutZC F L D Sl) :=
  (Game.draw (ΩcZ A P planZ tab vb L c)).bind fun w => (gameωZC A Hs planZ R τ w.1).bind fun fresh =>
    (runsG (runsG (gameωZC A Hs planZ R τ w.1) Rw) k).map fun bl => (w.1, fresh, bl)

/-- The prover's session strategy in every run. -/
noncomputable def trialStratZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) :
    Strategy (trialGZC A Hs P planZ tab vb R τ c k Rw) :=
  Strategy.bind _ _ (drawStrat _) fun w => Strategy.bind _ _ (stratωZC A Hs planZ R τ w.1).2 fun _ =>
    Strategy.map _ _ (runsStrat _ (runsStrat _ (stratωZC A Hs planZ R τ w.1).2 Rw) k)

open Classical in
/-- **What a trial reads at `c`**: nothing unless the fresh session accepts and every wait finds an
accepted rerun; then the readings of the extraction from those reruns. -/
noncomputable def rdTZC (c : vb.Pos) (e : TrialOutZC F L D Sl) : Finset vb.Val :=
  if accωZC planZ R e.1 e.2.1 then
    (firsts (accωZC planZ R e.1) e.2.2).elim ∅ fun bs => readsAtZC A Hs P planZ tab vb R τ e.1 bs c
  else ∅

open Classical in
/-- **A trial's cost**: `t'` per run it plays, the fresh one and, if it accepts, each wait's. -/
noncomputable def cTZC (t' : ℝ) (e : TrialOutZC F L D Sl) : ℝ :=
  t' * (1 + if accωZC planZ R e.1 e.2.1 then (e.2.2.map (waitCost (accωZC planZ R e.1))).sum else 0)

open Classical in
/-- An opening a trial found for value `v` at `c`, when it found one. -/
noncomputable def opOfVZC (c : vb.Pos) (e : TrialOutZC F L D Sl) (v : vb.Val) : vb.Opening :=
  Classical.epsilon fun o => accωZC planZ R e.1 e.2.1 ∧
    ∃ bs, firsts (accωZC planZ R e.1) e.2.2 = some bs ∧ OpensAtZC A Hs P planZ tab vb R τ e.1 bs c v o

open Classical in
/-- **The finder's output**: a collision from two openings of `c` with different values, the first
from its first trial, the second from that trial or from the first further trial that reads `c`. -/
noncomputable def outCZC (c : vb.Pos) (x : TrialOutZC F L D Sl × List (TrialOutZC F L D Sl)) :
    Option (List UInt8 × List UInt8) :=
  if h : ∃ p : TrialOutZC F L D Sl × vb.Val × vb.Val, p.2.1 ≠ p.2.2 ∧
      p.2.1 ∈ rdTZC A Hs P planZ tab vb R τ c x.1 ∧ p.2.2 ∈ rdTZC A Hs P planZ tab vb R τ c p.1 ∧
      (p.1 = x.1 ∨ nextRead (rdTZC A Hs P planZ tab vb R τ c) x.2 = some p.1) then
    some (vb.collide (opOfVZC A Hs P planZ tab vb R τ c x.1 (Classical.choose h).2.1)
      (opOfVZC A Hs P planZ tab vb R τ c (Classical.choose h).1 (Classical.choose h).2.2))
  else none

open Classical in
/-- **The plurality value at `c`**: the value the trial reads alone most often. -/
noncomputable def XvalZC (k Rw : ℕ) (c : vb.Pos) : vb.Val :=
  if h : Nonempty (ΩcZ A P planZ tab vb L c) then
    haveI := h
    Classical.choose (Finset.exists_max_image Finset.univ
      (pRd (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
        (rdTZC A Hs P planZ tab vb R τ c)) Finset.univ_nonempty)
  else default

/-- **The value layer**: each wire's bit in the plurality value of its commit string. -/
noncomputable def XplurZC (k Rw : ℕ) : Fin C.N → Bool :=
  fun g => vb.bit (XvalZC A Hs P planZ tab vb R τ k Rw (vb.pos g)) g

/-! ## Openings, and the finder's output -/

/-! ## The trial's expectations -/

/-! ## The link mass at one commit string -/

/-! ## The link theorem -/

/-- **A2 for one prover's link finders** (`Assumptions.SHA512CRExpected`): at the registration `R` and the
prover's continuation `τ`, for the finder built for each commit string a draw can read. A2 is a hypothesis about one
finder, since some finder outputs a collision after no evaluations: a prover that hard-codes two openings of one commit
string has link finders that find one. So the link theorem asks A2 only of the finders of the prover it bounds. -/
def LinkCRZC (k Rw M : ℕ) (t' : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)), Assumptions.SHA512CRExpected Hc
    (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
    (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)
    (finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t'))

open Classical in
/-- **The link bound, prover by prover**: `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))` at a prover whose link
finders satisfy A2 (`LinkCRZC`), and `⊤` at any other. -/
noncomputable def linkBoundZC (k Rw M : ℕ) (ρ t' : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) :
    ℝ≥0∞ :=
  if LinkCRZC A Hs P planZ tab vb R τ k Rw M t' then
    ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
      (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
  else ⊤

end FlockSoundness.Discharge.ZkLink
