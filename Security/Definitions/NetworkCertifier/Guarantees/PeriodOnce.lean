import Definitions.NetworkCertifier.Guarantees.SampledCharge
import Definitions.NetworkCertifier.Period
import Definitions.NetworkCertifier.RecordRows
import Definitions.Core.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The leak through the network certifier over a monitoring period, each bit charged once

`PeriodLeakAccepted` reads the client's view back from the network certifier's record under the schedule, so it
charges every schedule the server can induce: `#Ss`, each connection's admission and cell count among them, and
`∏_c (cap_c + 1)`. In the rows format (`Definitions.NetworkCertifier.RecordRows`) each frame names its connection, and
every frame the network certifier sends is in its record: the `r` padding frames a bucket from a connection's admission
to `adm c + D`, its cells, and a failed lane's fail cell. Connection `c`'s frames over the monitoring period's buckets,
in order and past the first `r·D`, are its cells, so the client's view reads back from the record and the schedule's
`(r, D)` alone, and `PeriodLeakOnce` charges the schedule only for those: `#Θ`, the registered `(r, D)`, in place of
`#Ss · ∏_c (cap_c + 1)`. Where the network certifier registers its observations (`TapOpens`), the record is the
circuit's values at the egress gates, which the content term counts, so each connection's cell count is charged there
and only there: in `#(freeInputs \ reg)` when the circuit reads it from the advice, and in nothing beyond its committed
values when the circuit computes it from them. A failure is charged in that term too: its fail cell is a frame of the
record.

Its other differences from `PeriodLeakAccepted` keep each premise named or a range: a window's partition is the one
its statement's registration gives (`Deploy.part`; a window that meets `WindowGood` has one), its faulty sets'
committed bits are charged at their maximum, and what a secret needs to be counted (its schedule's `(r, D)` in `Θ`, the
schedule fitting the format, the decoder recovering the secret from the client's view) is in the set counted.
-/

namespace NetTiming.Guarantees

open Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period, each bit charged once.** At every deployment
`D` of the sampled verifier where `Claims.Sampling.KS` holds and the trace's commitment binds
(`CommitmentScheme.Binds`), with a decoder that lists every trace within Flock's radius, over a monitoring period of `n`
windows registered as `sw i` (registered values `ρ i`, operator `op i`), with the network certifier's openings `tops s`
at each statement, and a network certifier that runs its code by `trusted-host`, its one run giving the cells the client
receives (`wardenCode`) and the observations of the monitoring period's windows, its record of those cells in a rows
format `F` (`RowsFormat.Wf`):

* each window's audit, at a statement in its scope, has `Pr[acc ∧ ¬RLeakTap δ] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`;
* where at every window's statement the network certifier registers its observations at that window's operator
  (`TapOpens`), a covert sender in the server: its secrets on a schedule whose `(r, D)` is in `Θ` and which fits the
  format, which a decoder recovers from the client's view, and whose every window meets `WindowGood` at that window's
  statement `x z i` and extracted witness `wit z i`, number at most
  `#Θ · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^(max_F faultyBits_i F) · Dj^k_i`: the sender moves at most
  `log2 #Θ + Σ_i (#(freeInputs_i \ reg_i) + log2 #faultySets_i(δ) + max_F faultyBits_i F + k_i·log2 Dj)` bits through
  the network certifier in the monitoring period. -/
def PeriodLeakOnce : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (hks : KS H D) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool)
    {Conn Cell Srv Dev Z : Type} (arr : Srv → Conn → Arrivals Cell) (F : RowsFormat Conn Cell L K) (hF : F.Wf)
    (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p =>
      (wardenCode p.1 (arr p.2), F.record S n p.1 (arr p.2) (wardenCode p.1 (arr p.2))))
    (Θ : Set (ℕ × ℕ)) (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool)) (hb : D.commit.Binds H),
    (∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
          (E D decode) m P ≤
        κ D δ m + Real.sqrt (γ D m * Claims.PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
          (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H) ∧
      ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (tops s)
          ((run tap (schedOf z, srvOf z)).2 i)) →
        {z | z ∈ Zs ∧ ((schedOf z).r, (schedOf z).D) ∈ Θ ∧ F.Fits S n (schedOf z) ∧
            dec (run tap (schedOf z, srvOf z)).1 = z ∧
            ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
              ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
          Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
            (faultySets (sw i).design (D.part (sw i)) δ).card *
            2 ^ ((faultySets (sw i).design (D.part (sw i)) δ).sup (faultyBits (sw i).prog (D.part (sw i)))) *
            Dj ^ k (sw i)) : ℕ) : ℕ∞))

end NetTiming.Guarantees
