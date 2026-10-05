import FlockSoundness.Protocol.Discharge.ZkSession.Session

/-!
# `δ_link` for `--zk` sessions under expected-time collision resistance (interactive-oracle layer)

`zk_flock_count_execOS_m1` leaves the link term as the named hypothesis `LinkSound δ_link`. It is discharged here in the
shape of `Audit.Partition.flock_batched_linkSoundE`, for the protocol as Lean defines it at the interactive-oracle layer
(`sessionZK`), under expected-time collision resistance of SHA-512 for an explicit finder
(`Assumptions.SHA512CRExpected`):

* **The binding** (`ValueBindingZ`): `Audit.Partition.ValueBinding` for `--zk` tables. A message that packs a witness
  satisfying a drawn unit's statement opens the commit string of each of the unit's wires to a value whose commitment is
  the registered one and whose bit on the wire is the decoded one; two openings of one commit string with different values
  give a collision of `Hc`.
* **Readings** (`readsZ`): on a draw, the values that drawn units reading a commit string open it to, from every padded
  witness near their table's level-0 oracle that satisfies the statement. At this layer the oracle is the prover's
  `Commit` message, so a draw fixes it: the prover's strategy on the draw is deterministic.
* **The trial** for a commit string `c` is a draw that reads `c`, uniform among such draws, at cost `t'` (one run of the
  prover's `Commit`, and the decoding of its oracles). **The value layer** (`XplurZ`) is the plurality of the trial's
  single readings at each commit string. **The finder** is `LinkFinder.finderG` on the trial; its successes output
  collisions (`outZ_spec`), so under A2 the trial's link mass at `c` is at most `2t'/2^256 + 1/(eM)` (`trialZ_mass_le`).

The link event at a drawn unit gives a reading other than the plurality at some commit string, whatever the session's
coins (`linkEvZK_le_psi`, with `prob_out_le`: a table's output oracle is the one `Commit` sent). So
(`zk_linkSoundE`)
$$\delta_{\rm link}\le Q_s\Big(\frac{2t'}{2^{256}}+\frac{1}{eM}\Big),\qquad Q_s=\sum_c\pi_c,$$
with `π_c` the probability that a draw reads `c`. At M0 (`flock_batched_linkSoundE`) the extraction from accepted reruns
adds `k/(eR_w)` and the factor `1/(1 − ρ)`; here extraction is the oracle itself. A2 is asked of one prover at a time
(`LinkCRZ`), and the bound is `⊤` at any other (`linkBoundZ`).

`zk_flock_count_execOS_m1_cr`: `zk_flock_count_execOS_m1` with the committed transcript the plurality value layer and
`δ_link` that bound.

**What this does not cover.** The finder reads the oracle the prover sends at `Commit`; the protocol as run sends a Merkle
cap, and the oracle is what extraction from the cap's openings recovers. The compiled counterpart, whose finder runs the
compiled prover and extracts, is `Discharge/ZkLink/Compiled.lean`'s open guarantee.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit LinkFinder ZkSession
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

/-! ## A table's output oracle is the one `Commit` sent -/

/-! ## The binding, readings and the trial -/

section Link

variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The binding of wires to registered commit strings for `--zk` tables** (`Audit.Partition.ValueBinding`'s form,
over `TabZK` plans). Each wire is held by a commit string (`pos`). A message that packs a witness satisfying drawn unit
`u`'s table's statement opens the commit string of each of `u`'s wires: the opening's commitment is the string registered
at `R` (`cs`), and its value's bit on the wire is the decoded one. Two openings of one string with different values give
a collision of `Hc` (`collide`). -/
structure ValueBindingZ (decode : DecoderZK (C := C) planZ tab) {Dh : Type} (Hc : List UInt8 → Dh) where
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  Val : Type
  [finVal : Fintype Val]
  [decVal : DecidableEq Val]
  [inhVal : Inhabited Val]
  pos : Fin C.N → Pos
  bit : Val → Fin C.N → Bool
  Opening : Type
  [inhOp : Inhabited Opening]
  Com : Type
  val : Opening → Val
  com : Opening → Com
  cs : Reg → Pos → Com
  opening : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
    Message F (tabOfZ planZ tab S R u).sch.k0 (tabOfZ planZ tab S R u).sch.level₀.logCols → Pos → Opening
  opening_spec : ∀ S R u, u ∈ S → ∀ M, (tabOfZ planZ tab S R u).S.Satisfies
      (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
    ∀ g ∈ P.io u, com (opening S R u M (pos g)) = cs R (pos g) ∧
      bit (val (opening S R u M (pos g))) g = decode S R u M g
  collide : Opening → Opening → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ Hc (collide o o').1 = Hc (collide o o').2

attribute [instance] ValueBindingZ.finPos ValueBindingZ.decPos ValueBindingZ.finVal ValueBindingZ.decVal
  ValueBindingZ.inhVal ValueBindingZ.inhOp

variable {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) (R : Reg) (τ : Cont L Reg (batchedSessionZK A planZ) R)

/-- Drawn unit `u` reads commit string `c`. -/
def ReadsZ (u : Fin n) (c : vb.Pos) : Prop := ∃ g ∈ P.io u, vb.pos g = c

/-- A draw reads commit string `c`. -/
def ReadsDrawZ (c : vb.Pos) (ω : L.Ω) : Prop := ∃ u ∈ L.draw ω, ReadsZ A P planZ tab vb u c

/-- The draws that read `c`. -/
abbrev ΩcZ (L : Law n) (c : vb.Pos) : Type := {ω : L.Ω // ReadsDrawZ A P planZ tab vb c ω}

open Classical in
noncomputable instance (L : Law n) (c : vb.Pos) : Fintype (ΩcZ A P planZ tab vb L c) := Subtype.fintype _

/-! ## Openings, and the finder's output -/

/-! ## The link mass at one commit string -/

/-! ## The link theorem -/

end Link

/-! ## The count curve at M1's parameters -/

section Count

variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

end Count

end FlockSoundness.Discharge.ZkLink
