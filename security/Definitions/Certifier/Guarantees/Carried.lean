module

public import Definitions.Core.Assumptions
public import Definitions.Certifier.Carried
public import Definitions.Certifier.Guarantees.SampledAudit

/-! The leak bounds over a carried pool, conditional as `PooledLeakAccepted` is on inputs no proof discharges. -/

@[expose] public section

namespace NetTiming.Guarantees

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

/-- **An accepted sampled audit bounds a carried run's leak.** `SampledLeakChecked` for a run that reads an opened row at
its opening gates `prow`, inputs of its circuit (`OpensAtInputs`), and leaves a row at its closing gates `nrow`, at every
compiled sampled verifier `D` where `Claims.SamplingGeneric.Compiled.KS` holds:

* on the statements in scope whose trace run the check holds on except with probability `εchk` (`CheckedWitness` at
  `traceRun`), `Pr[acc ∧ ¬RLeakCarried δ] ≤ δ + ε_ks + ε_bind + εchk`: except with that probability, every observation
  sequence of length `k` with jitter below `Dj` that the trace's transfer gates carry with every status complete is
  accepted with a valid run from the trace's values at `prow`, leaving the trace run's values at `nrow`, both read off
  the witness;
* on a statement in scope whose check pins `chk` (`CheckedUnique`), at every `ρ` and every opened row `rin`, the
  sequences accepted from `rin` (`AccSeqsCarried`) number at most
  `2^#(freeInputs \ (reg ∪ chk ∪ prow)) · #incorrectSets(δ) · 2^β · Dj^k`, `β` the most committed bits of a possible
  incorrect set. -/
def SampledLeakCarried : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (hopen : OpensAtInputs prow) (εchk : Claims.Msg → ℝ) (Dj : ℕ),
    KSLinOn H D.V (fun x => ScopeAcc D lay k x ∧ CheckedWitness H D (traceRun H D.commit) valid εchk x)
        (RLeakCarried H D.commit D.parse D.partition L K S lay k δ reg valid prow nrow Dj)
        D.E (fun x => D.κ δ x + εchk x) D.γ (fun _ => 1) (Bext H D.V D.E) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        CheckedUnique s.prog P s.design (lay s) (k s) (chk s) (valid s) →
        ∀ (op : ℕ → List (Obs L K S) → List Bool) (ρ : ℕ → Bool) (rin : Hash.Row),
          (AccSeqsCarried s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) (valid s) (prow s)
              (nrow s) rin δ op Dj).encard ≤
            ((2 ^ (freeInputs s.prog (lay s) (k s) \ (reg s ∪ chk s ∪ (prow s).toFinset)).card *
              (incorrectSets s.design P δ).card * 2 ^ ((incorrectSets s.design P δ).sup (faultyBits s.prog P)) *
              Dj ^ k s : ℕ) : ℕ∞)

/-- **The leak through the network attester over a monitoring period whose runs carry the pool** (conditional on its
named inputs). Over `m` runs on independent live coins, run `i` registered as `sw i` (partition `Pw i`, values `ρ i` at
the registered inputs, operator `op i`, which reads the runs before it), with `n i` proof blocks, each run's check pinning
its checked inputs (`CheckedUnique`) and its closing row at `nrow` (`CheckedClosing`); the record rule's restarts
`chains` and opened rows `genesis` (at run 0 the previous period's closing row or the empty pool's, and the empty pool's
at a restart), serving's choice at each secret and sample of the coins, which where every audit accepts lie in a
registered finite set `Cs` (`ResetsIn`) and have the runs open the root of the row the rule names (`CarriedRoots`, at the
`sha512/row-seg/v1` digest over `G`); with the network attester running its code by `trusted-device`, its one run giving
the cells the client receives and its record of them, the runs' observations; a covert sender's channel
(`MonitoringPeriodChannel`), each run's acceptance and incorrect set reading only the earlier runs' coins (`MonitoringPeriodAdaptive`), the
conditional escape (`MonitoringPeriodMissProb`), and the leak relation at each run's own incorrect set from its extracted rows `rin`,
`rout` (`MonitoringPeriodLeaksCarried`):

* for every secret, every audit accepts and the good event at the picked pattern fails (`PooledGoodCarried`) with
  probability at most `(1 - q)^Fmax + Σ e + Σ e' + Σ_i Pr[PoolColl i]`, `PoolColl i` a collision of `G` among the
  messages the digests of run `i`'s opened row and of the row it must open hashed;
* at every sample of the coins, the secrets whose audits all accept and that meet the good event at the pattern picked
  for them number at most `#Cs · #Ss · ∏_c (cap c + 1) · (N + 1)^Fmax · 2^(b · Fmax) ·
  ∏_i 2^#(freeInputs_i \ (reg_i ∪ chk_i ∪ prow_i)) · Dj^k_i`, `N = Σ_i n i` and `b` the most committed bits of one of the
  period's proof blocks.

The pattern is advice of `log₂ #Cs` bits. Under the public restart rule, where every audit accepts a run restarts only
at the registration's first run or at a chosen reset, at most one a week, so a week's period has `m + 1` patterns. A
restart the rule makes after a refused run is not charged, since no audit is refused where the count applies. -/
def PooledLeakCarried : Prop :=
  ∀ {L K S : ℕ} (G : Hash.Msg → Hash.Digest) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (n : Fin m → ℕ)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (hcls : ∀ i, CheckedClosing (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i))
      (nrow (sw i)))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (Zs : Set Z)
    (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : MonitoringPeriodChannel f cap Ss Zs schedOf srvOf dec fun p => (run tap p).1)
    (Cs : Finset ((Fin m → Bool) × (Fin m → Hash.Row))) (chains : Z → (Fin m → Ω) → Fin m → Bool)
    (genesis : Z → (Fin m → Ω) → Fin m → Hash.Row) (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (hcs : ResetsIn Zs acc Cs chains genesis)
    (hroot : ∀ z ∈ Zs, ∀ ω, (∀ i, acc z i ω (ω i)) →
      CarriedRoots G lay k reg valid prow nrow Dj sw Pw ρ op (chains z ω) (genesis z ω))
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (had : MonitoringPeriodAdaptive n acc W) (rin rout : Z → (Fin m → Ω) → Fin m → Hash.Row) (e e' : Fin m → ℝ)
    (he : ∀ i, 0 ≤ e i) (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : MonitoringPeriodMissProb Zs acc W q e)
    (hleak : MonitoringPeriodLeaksCarried lay k reg valid prow nrow Dj sw Pw ρ op Zs acc W rin rout
      (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) e')
    (Fmax : ℕ),
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op (chains z ω) (genesis z ω) Fmax (W z ω)
          (run tap (schedOf z ω, srvOf z ω)).2) ≤
        (1 - q) ^ Fmax + ∑ i, e i + ∑ i, e' i +
          ∑ i, Pr (fun ω => PoolColl G (chains z ω) (genesis z ω) (rin z ω) (rout z ω) i)) ∧
      ∀ ω, {z | z ∈ Zs ∧ (∀ i, acc z i ω (ω i)) ∧
          PooledGoodCarried lay k reg valid prow nrow Dj sw Pw ρ op (chains z ω) (genesis z ω) Fmax (W z ω)
            (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
        (Cs.card : ℕ∞) * (Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax *
          2 ^ ((univ.sup fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
          ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
              (reg (sw i) ∪ chk (sw i) ∪ (prow (sw i)).toFinset)).card * Dj ^ k (sw i)) : ℕ) : ℕ∞))

end NetTiming.Guarantees
