import Definitions.NetworkCertifier.SampledCharge

/-!
# A monitoring period of sampled audits: each window's leak relation at its witness's faulty set

Definitions only. `RLeak δ` puts a window's observations among the sequences accepted with some faulty set that `M_δ`
allows (`AccSeqsSampled`), so a monitoring period charged by it pays `log2 #faultySets(δ)` and `δ` in every window. Over
a monitoring period the faulty set that matters is the one of the witness computed from each window's registration,
before its draw: the draws bound how many units those sets hold in total, and the count pays for which units they are
once. `LeakAt` is the leak relation at one faulty set, and `PooledGood` is every window's at its own set, with the
sets' total below a bound.

A window's scan registers the outputs of its tied audits, the compute audits whose returned values are its registered
inputs (`TiedAudit`). An incorrect proof block of a tied audit can set every returned value in its downstream cone
(`coneOf`): the scan inputs it reaches (`TiedAudit.reach`) are registered at values the scan's registration no longer
fixes, so the window's leak relation holds with those inputs unpinned. `PooledGood` counts the tied audits' incorrect
proof blocks over the monitoring period too, and unpins what they reach.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **A window's leak relation at the faulty set `F`**: at the window registered as `s` (partition `P`, values `ρ` at
the registered inputs, operator `op`), the program fits its layout over the audit's scope `k s`, and the observations
`o` have that length, jitter below `Dj`, and are accepted with the faulty set `F` and the registered inputs pinned to
`ρ` (`AcceptedFaulty` at `AsReg`). -/
def LeakAt {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (s : AuditStatement)
    (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ)
    (o : List (Obs L K S)) : Prop :=
  (lay s).Fits s.prog (k s) ∧ o.length = k s ∧ (∀ y ∈ o, y.jitter < Dj) ∧
    AcceptedFaulty s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) F op o

/-- **The downstream cone** of the gates `S` of `C`, gate by gate in layout order: gate `g` is in it when it is one of
`S` or reads an earlier gate in it. A gate outside it computes the value it computes when no gate of `S` is incorrect. -/
def coneOf (C : Prog) (S : List ℕ) : List Bool :=
  C.gates.zipIdx.foldl (fun acc p => acc ++ [decide (p.2 ∈ S) || p.1.reads.any fun r => acc.getD r false]) []

/-- **A tied audit**: a compute audit whose returned values a window's scan registers, its Program `prog`, its
partition `part` into proof blocks, and `wire`, the scan's input gate each returned value is (`wire[j]` for
`prog.outs[j]`). -/
structure TiedAudit where
  prog : Prog
  part : Partition
  wire : List ℕ

/-- **What proof block `v` of a tied audit reaches**: the scan's input gates whose values are returned values of the
audit's Program in the downstream cone of `v`'s gates (`coneOf`), the registered inputs an incorrect `v` can set. -/
def TiedAudit.reach (a : TiedAudit) (v : ℕ) : Finset ℕ :=
  (((a.wire.zip a.prog.outs).filter fun p =>
    (coneOf a.prog (RU.gates (a.part.rus.getD v []))).getD p.2 false).map (·.1)).toFinset

/-- A window's tied audits' proof blocks, audit by audit, each as what it reaches: the `u`th is the window's tied proof
block `u`. -/
def tiedReaches (T : List TiedAudit) : List (Finset ℕ) :=
  T.flatMap fun a => (List.range a.part.rus.length).map a.reach

/-- What the window's tied proof blocks `F` reach together. -/
def tiedReach (T : List TiedAudit) (F : Finset ℕ) : Finset ℕ := F.biUnion fun u => (tiedReaches T).getD u ∅

/-- **`R`**: the most scan inputs one tied proof block of the window reaches. -/
def tiedWidth (T : List TiedAudit) : ℕ := (tiedReaches T).toFinset.sup Finset.card

/-- The scan inputs some tied proof block of the window reaches. -/
def tiedSpan (T : List TiedAudit) : Finset ℕ := tiedReach T (Finset.range (tiedReaches T).length)

/-- **A monitoring period's good event**: over the windows registered as `sw i`, the faulty sets `W i` hold fewer than
`Fmax` units in total and the incorrect sets `Wt i` of the windows' tied audits `T i` at most `Ft`, and every window's
observations `o i` meet its leak relation at its own faulty set (`LeakAt`), with the registered inputs its tied
audits' incorrect proof blocks reach unpinned (`tiedReach`). -/
def PooledGood {L K S m : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool)
    (T : Fin m → List TiedAudit) (Fmax Ft : ℕ) (W Wt : Fin m → Finset ℕ) (o : Fin m → List (Obs L K S)) : Prop :=
  ∑ i, (W i).card < Fmax ∧ ∑ i, (Wt i).card ≤ Ft ∧
    ∀ i, LeakAt lay k (fun s => reg s \ tiedReach (T i) (Wt i)) Dj (sw i) (Pw i) (ρ i) (op i) (W i) (o i)

end NetTiming
