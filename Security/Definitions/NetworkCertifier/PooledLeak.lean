import Definitions.NetworkCertifier.SampledCharge

/-!
# A monitoring period of sampled audits: each window's leak relation at its witness's incorrect proof blocks

Definitions only. `RLeak δ` puts a window's observations among the sequences accepted with some set of incorrect proof
blocks that `M_δ` allows (`AccSeqsSampled`), so a monitoring period charged by it pays `log2 #faultySets(δ)` and `δ` in
every window. Over a monitoring period the set that matters is the one of the witness computed from each window's
registration, before its draw: the draws bound how many proof blocks those sets hold in total, and the count pays for
which proof blocks they are once. `LeakAt` is the leak relation at one set, and `PooledGood` is every window's at its
own set, with the sets' total below a bound.

A window's scan registers the returned values of its tied audits, the compute audits whose returned values are its
registered inputs (`TiedAudit`). An incorrect proof block of a tied audit can change only the returned values in its
downstream cone (`inCone`): the scan inputs it reaches (`TiedAudit.reach`) are registered at values the tied audit's
registration no longer fixes, so the window's leak relation holds with those inputs unpinned. `PooledGood` counts the
tied audits' incorrect proof blocks over the monitoring period too, and unpins what they reach. How many there are is
sampled proofs' pooled bound at the tied audits' samples (`TiedIncorrectBlocksLe`).
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **A window's leak relation at the incorrect proof blocks `F`**: at the window registered as `s` (partition `P`,
values `ρ` at the registered inputs, operator `op`), the program fits its layout over the audit's scope `k s`, and the
observations `o` have that length, jitter below `Dj`, and are accepted with the proof blocks `F` incorrect and the
registered inputs pinned to `ρ` (`AcceptedFaulty` at `AsReg`). -/
def LeakAt {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (s : AuditStatement)
    (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ)
    (o : List (Obs L K S)) : Prop :=
  (lay s).Fits s.prog (k s) ∧ o.length = k s ∧ (∀ y ∈ o, y.jitter < Dj) ∧
    AcceptedFaulty s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) F op o

/-- **Gate `g` is in the downstream cone of the gates `S` of `C`**: it is one of `S` or reads an earlier gate in it, as
`valOver` reads. A gate outside it computes the value it computes when no gate of `S` holds a free value. -/
def inCone (C : Prog) (S : List ℕ) (g : ℕ) : Bool :=
  decide (g ∈ S) ||
    match C.gates[g]? with
    | some x => x.reads.any fun i => if _h : i < g then inCone C S i else false
    | none => false
termination_by g

/-- **A tied audit**: a compute audit whose returned values a window's scan registers, its Program `prog`, its
partition `part` into proof blocks, and `wire`, the scan's input gate each returned value is (`wire[j]` for
`prog.outs[j]`). -/
structure TiedAudit where
  prog : Prog
  part : Partition
  wire : List ℕ

/-- **What proof block `v` of a tied audit reaches**: the scan's input gates whose values are returned values of the
audit's Program in the downstream cone of `v`'s gates (`inCone`), the registered inputs an incorrect `v` can set. -/
def TiedAudit.reach (a : TiedAudit) (v : ℕ) : Finset ℕ :=
  (((a.wire.zip a.prog.outs).filter fun p => inCone a.prog (a.part.blocks.getD v []) p.2).map (·.1)).toFinset

/-- A window's tied audits' proof blocks, audit by audit, each as what it reaches: the `u`th is the window's tied proof
block `u`. -/
def tiedReaches (T : List TiedAudit) : List (Finset ℕ) :=
  T.flatMap fun a => (List.range a.part.blocks.length).map a.reach

/-- What the window's tied proof blocks `F` reach together. -/
def tiedReach (T : List TiedAudit) (F : Finset ℕ) : Finset ℕ := F.biUnion fun u => (tiedReaches T).getD u ∅

/-- **`R`**: the most scan inputs one tied proof block of the window reaches. -/
def tiedWidth (T : List TiedAudit) : ℕ := (tiedReaches T).toFinset.sup Finset.card

/-- The scan inputs some tied proof block of the window reaches. -/
def tiedSpan (T : List TiedAudit) : Finset ℕ := tiedReach T (Finset.range (tiedReaches T).length)

/-- **A monitoring period's good event**: over the windows registered as `sw i`, the incorrect proof blocks `W i` number
fewer than `Fmax` in total and the incorrect proof blocks `Wt i` of the windows' tied audits `T i` fewer than `Ft`, and
every window's observations `o i` meet its leak relation at its own set (`LeakAt`), with the registered inputs its tied
audits' incorrect proof blocks reach unpinned (`tiedReach`). -/
def PooledGood {L K S m : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool)
    (T : Fin m → List TiedAudit) (Fmax Ft : ℕ) (W Wt : Fin m → Finset ℕ) (o : Fin m → List (Obs L K S)) : Prop :=
  ∑ i, (W i).card < Fmax ∧ ∑ i, (Wt i).card < Ft ∧
    ∀ i, LeakAt lay k (fun s => reg s \ tiedReach (T i) (Wt i)) Dj (sw i) (Pw i) (ρ i) (op i) (W i) (o i)

/-- **The tied audits' incorrect proof blocks over a monitoring period, pooled**: for each secret `z ∈ Zs`, every
window's audits accept at the period's coins (`acc z i ω (ω i)`) while the windows' tied audits hold `Ft` or more
incorrect proof blocks in total (`Wt z ω i`) with probability at most `et`. It is sampled proofs' pooled bound at the
tied audits' samples: `NetTiming.tiedIncorrectBlocksLe_of_samples` derives it, at `et = (1 - q_t)^Ft + Σ_i ε_i`, from
`Guarantees.PooledIncorrectBlocksLe` wherever each window's tied audits sample a uniform subset of their proof blocks
at its coins and accept while the sample meets an incorrect one with probability at most `ε_i`. With no incorrect tied
proof block, an honest prover's, it holds at every `Ft ≥ 1` and `et ≥ 0`. -/
def TiedIncorrectBlocksLe {m : ℕ} {Ω Z : Type} [Fintype Ω] (Zs : Set Z) (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (Wt : Z → (Fin m → Ω) → Fin m → Finset ℕ) (Ft : ℕ) (et : ℝ) : Prop :=
  ∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧ Ft ≤ ∑ i, (Wt z ω i).card) ≤ et

end NetTiming
