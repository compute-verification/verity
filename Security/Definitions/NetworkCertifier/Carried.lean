module

public import Definitions.Hash.RowSeg
public import Definitions.NetworkCertifier.Checked

/-!
# The carried pool: runs chained through the pool's root

A run whose pool carries over (`PoolWindow_v1`) opens the pool the run before it left. Its circuit reads the opened row
at its opening gates `prow` and checks it in-circuit against the opening root `pin`, the row's `sha512/row-seg/v1` digest
(`Hash.RowSeg.digest`). Its offline memory check runs from that row on the run's own live coins. It leaves the row at its
closing gates `nrow`, and its check holds `nrow` to the run and its digest to the closing root `pout`. The runs connect only
through the root: under the record rule (`R7-pool-chain`) run `i`'s `pin` is run `i - 1`'s `pout`, one registered row
both read, where run `i` carries the pool, and the empty pool's root otherwise.

The opened row is no checked input: its root pins it only up to the 512 bits of a digest, so no check pins it as
`CheckedUnique` asks. It is a *carried* input, which the chain fixes: it is the row the previous run left, which that run's
check pins (`CheckedClosing`), unless two different rows have one root, a collision of the hash among what the two digests
hashed. So a period's count charges no bit of an opened row, and its escape bound adds a collision term for each opening.

* `poolOpen` and `PoolChained`: the record rule over a period's runs. Run `i` opens the row run `i - 1` left where it
  carries the pool (`chains i`, the record rule's `Registration.Pool.chains`), else `genesis i`, the empty pool's row (for
  the period's first run, the row the period starts from).
* `CheckedClosing`: the check pins the closing row, as `CheckedUnique` pins the egress.
* `AcceptedCarried` and `LeakAtCarried`: `AcceptedChecked` and `LeakAtChecked` with the run reading the row `rin` at its
  opening gates and leaving `rout` at its closing gates.
* `PooledGoodCarried`: a period's good event, the runs' observations accepted at rows that chain. Run `i`'s operator reads
  the runs before it (`runsBefore`): a request cut at a run's edge continues in the next run from its prompt and released
  outputs, both certified.
* `CarriedRoots`: accepted runs open the root of the row the record rule names, which each run's check of its roots and
  the record rule give, with no probability.
* `ResetsIn`: where every audit accepts, the restarts serving picks are among registered patterns, whose number a
  period's count charges.
* `PeriodLeaksCarried`: a period's input at each run's extracted rows. `PoolColl`: a collision at an opening.
* `AccSeqsCarried`, `OpensAtInputs` and `RLeakCarried`: one run. The sequences accepted from an opened row, and the leak
  relation at the rows the trace gives, which the extractor's witness fixes.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- The observations of the runs before run `i`, in order. -/
def runsBefore {m : ℕ} {α : Type} (o : Fin m → α) (i : Fin m) : List α := (List.ofFn o).take i.val

/-- **The row run `i` opens under the record rule**: the row run `i - 1` left (`rout (i - 1)`) where run `i` carries the
pool (`chains i`), and `genesis i` otherwise. Over roots it is `Registration.poolOpen`. -/
def poolOpen {m : ℕ} {R : Type} (chains : Fin m → Bool) (genesis rout : Fin m → R) (i : Fin m) : R :=
  if 0 < i.val ∧ chains i = true then rout ⟨i.val - 1, lt_of_le_of_lt (Nat.sub_le i.val 1) i.isLt⟩ else genesis i

/-- **The pool chain**: every run opens the row `poolOpen` names. Over roots it is the record rule: run `i`'s `pin` is
run `i - 1`'s `pout` where it carries the pool, and the empty pool's root otherwise. -/
def PoolChained {m : ℕ} {R : Type} (chains : Fin m → Bool) (genesis rin rout : Fin m → R) : Prop :=
  ∀ i, rin i = poolOpen chains genesis rout i

/-- **The check pins the closing row** at the gates `nrow`: two runs over the horizon `n` on one ingress, whose inputs
agree off `chk`, and which hold one set of values, only at committed gates of proof blocks of no class sampled `all`,
both `valid`, leave one row there. -/
def CheckedClosing {L S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (chk : Finset ℕ) (valid : (ℕ → Bool) → Prop) (nrow : List ℕ) : Prop :=
  ∀ (ins : List (List Bool)) (a a' : ℕ → Bool) (free : ℕ → Option Bool), ins.length = n → (∀ g ∉ chk, a g = a' g) →
    (∀ g b, free g = some b → ∃ v ∉ allSlots d P, FaultyGate C P {v} g) →
    valid (valOver C (lay.inputOf n ins a) free) → valid (valOver C (lay.inputOf n ins a') free) →
    nrow.map (valOver C (lay.inputOf n ins a) free) = nrow.map (valOver C (lay.inputOf n ins a') free)

/-- **Accepted with a valid run from the row `rin`, leaving `rout`**: `AcceptedChecked`, with the run's inputs at the
opening gates `prow` reading `rin` and its values at the closing gates `nrow` being `rout`. -/
def AcceptedCarried {L K S : ℕ} (C : Prog) (P : Partition) (lay : BoundaryGates L S) (k : ℕ) (As : Set (ℕ → Bool))
    (valid : (ℕ → Bool) → Prop) (prow nrow : List ℕ) (rin rout : Hash.Row) (F : Finset ℕ)
    (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)) : Prop :=
  ∃ a ∈ As, ∃ free : ℕ → Option Bool, (∀ g b, free g = some b → FaultyGate C P F g) ∧
    valid (valOver C (lay.inputOf k ((List.range k).map fun i => op i (o.take i)) a) free) ∧
    prow.map (lay.inputOf k ((List.range k).map fun i => op i (o.take i)) a) = rin ∧
    nrow.map (valOver C (lay.inputOf k ((List.range k).map fun i => op i (o.take i)) a) free) = rout ∧
    Accepted (circuitProgram C lay k) op k o (a, free)

/-- **A run's leak relation from the row `rin`, leaving `rout`**: `LeakAtChecked` at `AcceptedCarried`. -/
def LeakAtCarried {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (prow nrow : AuditStatement → List ℕ)
    (Dj : ℕ) (s : AuditStatement) (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool)
    (rin rout : Hash.Row) (F : Finset ℕ) (o : List (Obs L K S)) : Prop :=
  (lay s).Fits s.prog (k s) ∧ Disjoint F (allSlots s.design P) ∧ o.length = k s ∧ (∀ y ∈ o, y.jitter < Dj) ∧
    AcceptedCarried s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) (valid s) (prow s) (nrow s) rin rout F
      op o

/-- **A monitoring period's good event over a carried pool**: the incorrect sets `W i` hold fewer than `Fmax` units in
total, and some rows that chain (`PoolChained`) carry every run's observations: run `i`'s are accepted with a valid run
from its row at its own set (`LeakAtCarried`), its operator reading the runs before it. -/
def PooledGoodCarried {L K S m : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (prow nrow : AuditStatement → List ℕ)
    (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (chains : Fin m → Bool)
    (genesis : Fin m → Hash.Row) (Fmax : ℕ) (W : Fin m → Finset ℕ) (o : Fin m → List (Obs L K S)) : Prop :=
  ∑ i, (W i).card < Fmax ∧ ∃ rin rout : Fin m → Hash.Row, PoolChained chains genesis rin rout ∧
    ∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i) (rout i) (W i)
      (o i)

/-- **The leak relation from each run's extracted rows**: run `i`'s audit accepts while its observations `obs z ω i` are
not accepted with a valid run at `W z ω i` from the row `rin z ω i`, leaving `rout z ω i` (`LeakAtCarried`), with
probability at most `e' i`. -/
def PeriodLeaksCarried {L K S m : ℕ} {Ω Z : Type} [Fintype Ω] (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (Zs : Set Z)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop) (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (rin rout : Z → (Fin m → Ω) → Fin m → Hash.Row) (obs : Z → (Fin m → Ω) → Fin m → List (Obs L K S))
    (e' : Fin m → ℝ) : Prop :=
  ∀ z ∈ Zs, ∀ i, Pr (fun ω => acc z i ω (ω i) ∧
    ¬LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i)) (rin z ω i)
      (rout z ω i) (W z ω i) (obs z ω i)) ≤ e' i

/-- **Accepted runs open the root of the row the record rule names**: rows at which every run's observations are
accepted with a valid run (`LeakAtCarried`) have, at each run, the opened row's root equal to the root of the row
`poolOpen` names (the row run `i - 1` left, or `genesis i`). Each run's check holds its opened row's root to its `pin`
and its closing row's to its `pout`, on runs whose faults avoid the proof blocks of classes sampled `all`, and the record
rule makes run `i`'s `pin` run `i - 1`'s `pout` where it carries the pool, else the empty pool's root. -/
def CarriedRoots {L K S m : ℕ} (G : Hash.Msg → Hash.Digest) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    (chains : Fin m → Bool) (genesis : Fin m → Hash.Row) : Prop :=
  ∀ (o : Fin m → List (Obs L K S)) (W : Fin m → Finset ℕ) (rin rout : Fin m → Hash.Row),
    (∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i) (rout i) (W i)
      (o i)) →
    ∀ i, Hash.RowSeg.digest G (rin i) = Hash.RowSeg.digest G (poolOpen chains genesis rout i)

/-- **The restarts serving picks are registered patterns**: at each secret in `Zs` and draw of the coins at which every
run's audit accepts, the restarts `chains z ω` and opened rows `genesis z ω` serving picked are one of the patterns of
`Cs`. Under the public restart rule a run restarts there only at the registration's first run or at a chosen reset, of
which the record rule refuses a second in a week. -/
def ResetsIn {m : ℕ} {Ω Z : Type} (Zs : Set Z) (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (Cs : Finset ((Fin m → Bool) × (Fin m → Hash.Row))) (chains : Z → (Fin m → Ω) → Fin m → Bool)
    (genesis : Z → (Fin m → Ω) → Fin m → Hash.Row) : Prop :=
  ∀ z ∈ Zs, ∀ ω, (∀ i, acc z i ω (ω i)) → (chains z ω, genesis z ω) ∈ Cs

/-- **The sequences a sampled audit accepts at `δ` from the row `rin`**: `AccSeqsChecked`, with the run reading `rin` at its
opening gates `prow` and leaving any row at its closing gates `nrow`. -/
def AccSeqsCarried {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (As : Set (ℕ → Bool)) (valid : (ℕ → Bool) → Prop) (prow nrow : List ℕ) (rin : Hash.Row) (δ : ℝ)
    (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧
    ∃ F ∈ faultySets d P δ, ∃ rout, AcceptedCarried C P lay n As valid prow nrow rin rout F op o}

/-- **The opening gates are inputs** of each statement's circuit: a run reads its opened row where values enter it. -/
def OpensAtInputs (prow : AuditStatement → List ℕ) : Prop :=
  ∀ s, ∀ g ∈ prow s, s.prog.gates[g]? = some .input

/-- **The leak relation with a valid run from the trace's rows** at `δ`: `RLeakChecked`, with the accepted run reading the
trace's values at the opening gates `prow` and leaving at the closing gates `nrow` the values of the trace's run (`C` on
the trace's values, the incorrect proof blocks' committed gates held at them, `heldAdv`). Both rows are functions of the
witness, so two runs' rows that differ under one root give a collision found from the two witnesses. -/
def RLeakCarried (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧
    ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
      Carries (lay s) s.prog (w.val H Cm s) op (k s) o → Covered (k s) o → o.length = k s → (∀ y ∈ o, y.jitter < Dj) →
        ∃ F ∈ faultySets s.design P δ, AcceptedCarried s.prog P (lay s) (k s)
          (AsReg s.prog (lay s) (k s) (reg s) (traceVal s.prog (w.val H Cm s))) (valid s) (prow s) (nrow s)
          ((prow s).map (traceVal s.prog (w.val H Cm s)))
          ((nrow s).map (valOver s.prog (traceVal s.prog (w.val H Cm s))
            (heldAdv s.prog P (faulty H Cm s P w) (w.val H Cm s))))
          F op o

/-- **A collision at run `i`'s opening**: two different messages with one digest among those the digests of the row run
`i` opened (`rin i`) and of the row it must open (`poolOpen`) hashed. -/
def PoolColl (G : Hash.Msg → Hash.Digest) {m : ℕ} (chains : Fin m → Bool) (genesis rin rout : Fin m → Hash.Row)
    (i : Fin m) : Prop :=
  Hash.CollisionIn G (Hash.RowSeg.hashed G (rin i) ++ Hash.RowSeg.hashed G (poolOpen chains genesis rout i))

end NetTiming
