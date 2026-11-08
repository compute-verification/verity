module

public import Definitions.Core.Assumptions
public import Definitions.NetworkCertifier.Carried

/-!
# The leak bound over a carried pool

`PooledLeakChecked` over runs that carry the pool: each run opens the row the run before it left, or the empty pool's
where the chain restarts (`PoolChained`, the record rule), and runs its own offline memory check from it on its own live
coins. An opened row costs no advice bits: the chain fixes it, the previous run's closing row, which that run's check pins
(`CheckedClosing`), unless two different rows have one root. So the count is the product of the runs' counts, each
leaving out its opened row (`prow`), and the escape bound adds, for each opening, the chance that its root is not the root
of the row it must open (`PeriodRoots`) and a collision of the hash among what the two rows' digests hashed (`PoolColl`).

It is conditional as `PooledLeakChecked` is: its inputs are named (`PeriodChannel`, `PeriodAdaptive`, `PeriodEscapes`,
`PeriodLeaksCarried`, `PeriodRoots`), none discharged yet.
-/

@[expose] public section

namespace NetTiming.Guarantees

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

open Claims Claims.Sampling Finset

/-- **The leak through the network certifier over a monitoring period whose runs carry the pool** (conditional on its
named inputs). Over `m` runs on independent live coins, run `i` registered as `sw i` (partition `Pw i`, values `ρ i` at
the registered inputs, operator `op i`, which reads the runs before it), with `n i` proof blocks, each run's check pinning
its checked inputs (`CheckedUnique`) and its closing row at `nrow` (`CheckedClosing`); the record rule's restarts
`chains` and rows `genesis` (the empty pool's); with the network certifier running its code by `trusted-host`, its one run
giving the cells the client receives and its record of them, the runs' observations; a covert sender's channel
(`PeriodChannel`), each run's acceptance and incorrect set reading only the earlier runs' coins (`PeriodAdaptive`), the
conditional escape (`PeriodEscapes`), the leak relation at each run's own incorrect set from its extracted rows `rin`,
`rout` (`PeriodLeaksCarried`), and each run opening the root of the row the record rule names (`PeriodRoots`, at the
`sha512/row-seg/v1` digest over `G`):

* for every secret, every audit accepts and the good event fails (`PooledGoodCarried`) with probability at most
  `(1 - q)^Fmax + Σ e + Σ e' + Σ eo + Σ_i Pr[PoolColl i]`, `PoolColl i` a collision of `G` among the messages the digests
  of run `i`'s opened row and of the row it must open hashed;
* at every sample of the coins, the secrets that meet the good event number at most
  `#Ss · ∏_c (cap c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ (reg_i ∪ chk_i ∪ prow_i)) · Dj^k_i`,
  `N = Σ_i n i` and `b` the most committed bits of one of the period's proof blocks. -/
def PooledLeakCarried : Prop :=
  ∀ {L K S : ℕ} (G : Hash.Msg → Hash.Digest) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (n : Fin m → ℕ)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (hcls : ∀ i, CheckedClosing (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i))
      (nrow (sw i)))
    (chains : Fin m → Bool) (genesis : Fin m → Hash.Row)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (Zs : Set Z)
    (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : PeriodChannel f cap Ss Zs schedOf srvOf dec fun p => (run tap p).1)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop) (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (had : PeriodAdaptive n acc W) (rin rout : Z → (Fin m → Ω) → Fin m → Hash.Row) (e e' eo : Fin m → ℝ)
    (he : ∀ i, 0 ≤ e i) (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : PeriodEscapes Zs acc W q e)
    (hleak : PeriodLeaksCarried lay k reg valid prow nrow Dj sw Pw ρ op Zs acc W rin rout
      (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) e')
    (hroot : PeriodRoots G Zs acc chains genesis rin rout eo)
    (Fmax : ℕ),
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op chains genesis Fmax (W z ω)
          (run tap (schedOf z ω, srvOf z ω)).2) ≤
        (1 - q) ^ Fmax + ∑ i, e i + ∑ i, e' i + ∑ i, eo i +
          ∑ i, Pr (fun ω => PoolColl G chains genesis (rin z ω) (rout z ω) i)) ∧
      ∀ ω, {z | z ∈ Zs ∧ PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op chains genesis Fmax (W z ω)
          (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax *
          2 ^ ((univ.sup fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
          ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
              (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card * Dj ^ k (sw i)) : ℕ) : ℕ∞)

end NetTiming.Guarantees
