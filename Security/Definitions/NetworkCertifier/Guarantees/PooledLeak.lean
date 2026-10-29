import Definitions.NetworkCertifier.Guarantees.Period
import Definitions.NetworkCertifier.PooledLeak

/-!
# The leak through the network certifier over a monitoring period of sampled audits

A monitoring period is `m` windows, each audited on its own live coins (the period's coins are `Fin m → Ω`, window
`i`'s at `ω i`) at a statement registered as `sw i`. `PooledLeakAccepted` is the count `K_pooled`: what a covert
sender in the server moves through the network certifier over the whole monitoring period, with the faulty units paid
for once over the period rather than at `δ` in every window. At each window the witness computed from the
registration, before the window's draw, has a faulty set `W i` (it reads only the earlier windows' coins), and the
statement takes two inputs per window:

* the conditional escape (`hesc`): given the earlier windows' coins, the window's audit accepts with probability at most
  `(1 - q)^|W_i| + e_i`, `q` the deployed design's per-window detection probability (a uniform draw of `kd_i` of the
  window's `n_i` units meets it at `q = kd_i / n_i`; the statement fixes no draw's form);
* the leak relation at that same witness (`hleak`): the audit accepts while the window's observations in the network
  certifier's record are not accepted with the faulty set `W_i` (`LeakAt`) with probability at most `e'_i`.

A window's scan registers the returned values of its tied audits `T i` (`TiedAudit`), compute audits sampled at their
own design. Their witness has an incorrect set `Wt i` of their proof blocks, also fixed before the window's draw, with
its own conditional escape (`hesct`, `(1 - q_t)^|Wt_i| + e^t_i`), and the leak relation holds with the scan inputs
those proof blocks reach unpinned (`tiedReach`: the returned values in each one's downstream cone in its audit's
Program). An empty list of tied audits is the statement without them: no tied proof block, nothing unpinned, and the
tied factor and error `1` and `0` at `q_t = 1`, `F_t = 0`, `δ_t = 0`.

From them, every audit accepts while the faulty sets hold `Fmax ≥ ln(1/δ)/q` units or more, or the incorrect sets more
than `F_t` with `(1 - q_t)^(F_t + 1) ≤ δ_t`, or some window's leak relation fails, with probability at most
`δ + δ_t + Σ_i e_i + Σ_i e^t_i + Σ_i e'_i` (the escapes multiply, `PooledIncorrectBlocksLe`'s argument with the escape
as the input, once for each); and the secrets whose every window meets its relation, with the faulty sets under `Fmax`
and the incorrect sets at most `F_t` (`PooledGood`), number at most
`#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ reg_i) · Dj^k_i` times the tied factor
`min((N_t + 1)^F_t · 2^(R · F_t), 2^(Σ_i #span_i))`, `N = Σ_i n_i` the period's units, `b` each unit's committed bits,
`N_t` the period's tied proof blocks, `R` the most scan inputs one of them reaches (`tiedWidth`) and `span_i` the scan
inputs any tied proof block of window `i` reaches (`tiedSpan`): `(ln(1/δ)/q + 1) · (b + log2 (N + 1))` bits for the
faulty units, `min(F_t · (R + log2 (N_t + 1)), Σ_i #span_i)` for the tied audits' incorrect proof blocks, plus the
advice and timing terms and the schedule terms. A proof unit's outputs are at most 16 bits, so `b = 16`.

The network certifier is one device on the server's uplink, as in `PeriodLeakAccepted`
(`Definitions.NetworkCertifier.Tap`): one run of its code gives the cells the client receives (it functions as an
active warden) and its record of them, the observations the audits read, so on every schedule its record format holds
the client's cells read back from the record. Its one named assumption is `trusted-host`.

The statement is conditional: it holds for any audits that satisfy the three inputs. Over a monitoring period of live
audits they are `Flock.Guarantees.SampledKSLive` at each window's history (at `K₀ = |W_i|` for the scan and
`K₀ = |Wt_i|` for its tied audits) and the live `SampledLeakAccepted` (P2) at the unpinned inputs, none discharged yet, so
nothing here bounds the deployed system over a monitoring period until that discharge lands.
-/

namespace NetTiming.Guarantees

open Claims Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of accepted sampled audits** (conditional on its
three per-window inputs). Over `m` windows on independent live coins, window `i` registered as `sw i` (partition `Pw i`,
values `ρ i` at the registered inputs, operator `op i`, tied audits `T i`), with `n i` units, each committing at most
`b` bits; with the network certifier running its code by `trusted-host` on schedules in `Ss`, each held by its record
format `f` (`TapRecord.fits`) and giving connection `c` at most `cap c` cells, its one run giving the cells the client
receives and its record of them in the format `f`,
the windows' observations; and a covert sender in the server whose secrets `z ∈ Zs` a decoder recovers from the
client's cells at the period's coins. For each secret, `acc z i ω x` is window `i`'s audits accepting at its coins `x`
after the earlier windows' coins `ω`, `W z ω i ⊆ range (n i)` is the faulty set of the witness computed from window
`i`'s registration, and `Wt z ω i` its tied audits' incorrect proof blocks (indices into `tiedReaches (T i)`); all read
only the windows before `i`. Given

* `hesc`, the conditional escape: `Pr_x[acc z i ω x] ≤ (1 - q)^|W z ω i| + e i`, `q` the design's per-window
  detection probability (`Flock.Guarantees.SampledKSLive` at the history, not yet discharged);
* `hesct`, the tied audits' conditional escape: `Pr_x[acc z i ω x] ≤ (1 - qt)^|Wt z ω i| + et i`, `qt` their design's
  per-window detection probability (`SampledKSLive` at the tied audits' history, not yet discharged);
* `hleak`, the leak relation at that witness: `Pr[acc_i ∧ ¬LeakAt (W z ω i)] ≤ e' i` on the record's observations, with
  the registered inputs `tiedReach (T i) (Wt z ω i)` unpinned (the live `SampledLeakAccepted`, P2, not yet discharged);

for `0 ≤ q ≤ 1`, `Fmax ≥ ln(1/δ)/q`, `0 ≤ qt ≤ 1` and `(1 - qt)^(Ft + 1) ≤ δt`:

* for every secret, every audit accepts and the monitoring period's good event fails (`PooledGood`: the faulty sets
  hold under `Fmax` units and the incorrect sets at most `Ft`, and every window meets its leak relation at its own) with
  probability at most `δ + δt + Σ e + Σ et + Σ e'`;
* at every draw of the coins, the secrets that meet the good event number at most
  `#Ss · ∏_c (cap c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ reg_i) · Dj^k_i` times
  `min((Nt + 1)^Ft · 2^(R · Ft), 2^(Σ_i #tiedSpan (T i)))`, `N = Σ_i n i`, `Nt` the period's tied proof blocks and `R`
  the most scan inputs one reaches (`tiedWidth`). -/
def PooledLeakAccepted : Prop :=
  ∀ {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m : ℕ}
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool)
    (T : Fin m → List TiedAudit)
    (n : Fin m → ℕ) (b : ℕ) (hb : ∀ i, ∀ v < n i, faultyBits (sw i).prog (Pw i) {v} ≤ b)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → (Fin m → Ω) → WardenSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (hs : ∀ z ∈ Zs, ∀ ω, schedOf z ω ∈ Ss) (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hdz : ∀ z ∈ Zs, ∀ ω, dec ω (run tap (schedOf z ω, srvOf z ω)).1 = z)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (hacc : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → acc z i ω = acc z i ω')
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (hW : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → W z ω i = W z ω' i)
    (hWn : ∀ z ω i, W z ω i ⊆ range (n i))
    (Wt : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (hWt : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → Wt z ω i = Wt z ω' i)
    (hWtn : ∀ z ω i, Wt z ω i ⊆ range (tiedReaches (T i)).length)
    (e et e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i) (het : ∀ i, 0 ≤ et i)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (qt : ℝ) (hqt0 : 0 ≤ qt) (hqt1 : qt ≤ 1)
    (hesc : ∀ z ∈ Zs, ∀ i ω, Pr (acc z i ω) ≤ (1 - q) ^ (W z ω i).card + e i)
    (hesct : ∀ z ∈ Zs, ∀ i ω, Pr (acc z i ω) ≤ (1 - qt) ^ (Wt z ω i).card + et i)
    (hleak : ∀ z ∈ Zs, ∀ i, Pr (fun ω => acc z i ω (ω i) ∧
      ¬LeakAt lay k (fun s => reg s \ tiedReach (T i) (Wt z ω i)) Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i)
        ((run tap (schedOf z ω, srvOf z ω)).2 i)) ≤ e' i)
    (δ : ℝ) (hδ : 0 < δ) (Fmax : ℕ) (hF : Real.log (1 / δ) ≤ q * Fmax)
    (δt : ℝ) (Ft : ℕ) (hFt : (1 - qt) ^ (Ft + 1) ≤ δt),
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬PooledGood lay k reg Dj sw Pw ρ op T Fmax Ft (W z ω) (Wt z ω) (run tap (schedOf z ω, srvOf z ω)).2) ≤
        δ + δt + ∑ i, e i + ∑ i, et i + ∑ i, e' i) ∧
      ∀ ω, {z | z ∈ Zs ∧
          PooledGood lay k reg Dj sw Pw ρ op T Fmax Ft (W z ω) (Wt z ω) (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
          ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i)) *
          min (((∑ i, (tiedReaches (T i)).length) + 1) ^ Ft * 2 ^ ((univ.sup fun i => tiedWidth (T i)) * Ft))
            (2 ^ ∑ i, (tiedSpan (T i)).card) : ℕ) : ℕ∞)

end NetTiming.Guarantees
