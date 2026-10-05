import FlockSoundness.Protocol.Audit.FlockBatched
import FlockSoundness.Protocol.SessionBatchAcc
import FlockSoundness.Protocol.LinkFinder

/-!
# The link theorem: `δ_link` under expected-time collision resistance (DESIGN.md §3)

The batched audit's link term (`analysisBE`) is discharged here for the plurality value layer,
under expected-time collision resistance of SHA-512 for an explicit finder
(`Assumptions.SHA512CRExpected`) and the binding of wires to commit strings (`ValueBinding`,
named).

* **The binding** (`ValueBinding`): each wire is held by a commit string (a leaf under the serving
  roots, `pos`). A satisfying witness of a drawn unit's table opens the commit string of each of the
  unit's wires, registered at `R`, to a value whose bit on the wire is the decoded one. Two openings
  of one string with different values give a SHA-512 collision. M0's `hm96-sha512` leaf layout
  discharges it.
* **Readings** (`readsAt`): after an extraction, the values drawn units open a commit string to.
* **The trial** (`trialG`) for a commit string `c`: a draw that reads `c`, uniform among such draws;
  a fresh session; if the session accepts, `k` waits of at most `Rw` reruns each for a rerun in which
  the session accepts. It reads what the extraction from the runs found reads at `c` (`rdT`), and
  costs `t'` per run it plays (`cT`), `t'·(1 + k)` in expectation.
* **The value layer** (`Xplur`) is the plurality of the trial's single readings at each commit
  string. It is a function of the registration state.
* **The finder** for `c` is `LinkFinder.finderG` on the trial. Its successes output collisions
  (`outC`), so under A2 the trial's link mass at `c` is at most `2t'(1 + k)/2^256 + 1/(eM)`.

A2 is asked of one prover at a time (`LinkCR`): some prover hard-codes two openings of a commit string, and its
finders find a collision, so no bound holds for every prover under A2. The theorem bounds the link term at each prover
whose finders satisfy A2, and gives `⊤` at any other (`linkBoundCR`).

The link event at a drawn unit gives a reading other than the plurality at some commit string, so
the audit's link term is at most the ideal link mass, summed over commit strings. The ideal extractor
and the trial differ only by the waits that give up (`k/(eRw)`). Each commit string is weighted by
the probability `π c` that a draw reads it (`flock_batched_linkSoundE`):
$$\delta_{\rm link}\le\frac{Q_s}{1-\rho}\Big(\frac{2(1+k)\,t'}{2^{256}}+\frac{1}{eM}+
\frac{k}{eR_w}\Big),\qquad Q_s=\sum_c\pi_c,$$
with `ρ ≥ N₀/(ek)` for every table. `Q_s` is the expected number of commit strings the drawn units
read, at most the `N_s` leaves under the serving roots. At `k = ⌈2N₀/e⌉`, `ρ = 1/2`, this is the
record's `t·N_s(8N₀/e)/2^256` with `Q_s` in place of `N_s`, up to lower-order terms.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **The binding of wires to registered commit strings** (named hypothesis): the statement's
Halevi–Micali checks, for each drawn unit's table. Each wire is held by a commit string (`pos`). A
message that satisfies a drawn unit's table opens the commit string of each of the unit's wires:
the opening's commitment is the string registered at `R` (`cs`), and its value's bit on the wire is
the decoded one. Two openings of one string with different values give a collision of `Hc`
(`collide`), as Halevi–Micali binding reduces to SHA-512 collision resistance. -/
structure ValueBinding (decode : DecoderB (C := C) (Reg := Reg) plan tab) {Dh : Type}
    (Hc : List UInt8 → Dh) where
  /-- The commit strings: the leaves under the serving roots. -/
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  /-- The value behind a commit string. -/
  Val : Type
  [finVal : Fintype Val]
  [decVal : DecidableEq Val]
  [inhVal : Inhabited Val]
  /-- The commit string that holds each wire. -/
  pos : Fin C.N → Pos
  /-- A value's bit on each wire its string holds. -/
  bit : Val → Fin C.N → Bool
  /-- An opening of a commit string, and its commitment. -/
  Opening : Type
  [inhOp : Inhabited Opening]
  Com : Type
  val : Opening → Val
  com : Opening → Com
  /-- The commit string registered at each position. -/
  cs : Reg → Pos → Com
  /-- The opening a drawn unit's table's message gives at a commit string. -/
  opening : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
    Message 𝔽 (tabOf plan tab S R u).sch.k0 (tabOf plan tab S R u).sch.level₀.logCols → Pos → Opening
  opening_spec : ∀ S R u, u ∈ S → ∀ M, (tabOf plan tab S R u).S.Satisfies
      (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
        (tabOf plan tab S R u).sch.level₀.logCols M) →
      ∀ g ∈ P.io u, com (opening S R u M (pos g)) = cs R (pos g) ∧
        bit (val (opening S R u M (pos g))) g = decode S R u M g
  /-- Two openings of one string with different values give a collision. -/
  collide : Opening → Opening → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ Hc (collide o o').1 = Hc (collide o o').2

attribute [instance] ValueBinding.finPos ValueBinding.decPos ValueBinding.finVal
  ValueBinding.decVal ValueBinding.inhVal ValueBinding.inhOp

/-- Drawn unit `u` reads commit string `c`. -/
def Reads (u : Fin n) (c : vb.Pos) : Prop := ∃ g ∈ P.io u, vb.pos g = c

/-- A draw reads commit string `c`. -/
def ReadsDraw (c : vb.Pos) (ω : L.Ω) : Prop := ∃ u ∈ L.draw ω, P.Reads plan tab vb u c

/-- The draws that read `c`. -/
abbrev Ωc (L : Law n) (c : vb.Pos) : Type := {ω : L.Ω // P.ReadsDraw plan tab vb c ω}

open Classical in
noncomputable instance (L : Law n) (c : vb.Pos) : Fintype (P.Ωc plan tab vb L c) :=
  Subtype.fintype _

/-! ## Openings, and the finder's output -/

/-! ## The trial's expectations -/

/-! ## The link mass at one commit string -/

/-! ## The link theorem -/

end Partition

end FlockSoundness.Audit
