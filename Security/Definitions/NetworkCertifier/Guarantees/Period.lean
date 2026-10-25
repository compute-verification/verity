import Models.NetworkCertifier.Guarantees.Warden
import Definitions.NetworkCertifier.Guarantees.SampledCharge
import Definitions.NetworkCertifier.Period

/-!
# The leak through the wardens over a period of sampled audits

A period is `n` windows, each audited by the sampled verifier at a statement it registered (`sw i`, with its partition
`Pw i` and the values `ρ i` the registration commits at its registered inputs). The server's data through the wardens
is its egress over the period's windows, framed into cells (`cells`). `PeriodLeakAccepted` composes the sampled leak
bound (`SampledLeakAccepted`, with the registered inputs pinned: `AsReg`) with the warden's view count
(`EncardWardenViewsLe`), so the content term of the warden's bound comes from the audits, not from a premise:

* each window's audit, at a statement in its scope, fails the leak relation `RLeak` with probability at most
  `δ + ε_ks + ε_bind` at that statement (`KSLinOn`, `Claims.Sampling.KS`'s errors);
* the secrets of a covert sender in the server whose every window meets its good event (`WindowGood`) number at most
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`: the sender moves at most
  `log2 #Ss + Σ_c log2 (cap_c + 1) + Σ_i (#(freeInputs_i \ reg_i) + log2 #faultySets_i(δ) + β_i + k_i·log2 Dj)` bits
  through the wardens in the period.

The union of the windows' failures over the period's live coins is not stated here. `WindowGood` is the event the
first part bounds, plus two conditions on the run that no proof here gives: the registration's values (which the
registration's commitment binds) and the transfer gates' tie to the certifier's rows.
-/

namespace NetTiming.Guarantees

open Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the wardens over a period of accepted sampled audits.** At every deployment `D` of the sampled
verifier where `Claims.Sampling.KS` holds, with a decoder that lists every trace within Flock's radius, over a period of
`n` windows registered as `sw i` (partition `Pw i`, registered values `ρ i`, operator `op i`, and `β i` bounding the
committed bits of each of its possible faulty sets), and wardens that run their code by `trusted-nebius-host` on
schedules in `Ss`, each giving connection `c` at most `cap c` cells:

* each window's audit, at a statement in its scope, has `Pr[acc ∧ ¬RLeak δ] ≤ δ + ε_ks + ε_bind`;
* a covert sender in the server, whose data through the wardens is its observations' egress framed into cells, and
  whose secrets a decoder recovers from the client's view of the server side: its secrets whose every window meets
  `WindowGood`, at that window's statement `x z i` and extracted witness `wit z i`, number at most
  `#Ss · ∏_c (cap c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`. -/
def PeriodLeakAccepted : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (hks : KS H D) (L K S : ℕ) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : Stmt → Finset ℕ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → Stmt) (Pw : Fin n → Partition) (hPw : ∀ i, D.partition (sw i) = some (Pw i))
    (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
    (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell) (w : Dev)
    (run : Dev → WardenSchedule Conn × Srv → Conn → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedNebiusHost w run fun p => wardenCode p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run w (schedOf z, srvOf z)) = z)
    (obs : Z → Fin n → List (Obs L K S)) (cells : (Fin n → List (Obs L K S)) → Conn → List Cell)
    (hdata : ∀ z ∈ Zs, (fun c => (arr (srvOf z) c).stream) = cells (obs z))
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit),
    KSLinOn H D.V (ScopeAcc D lay k)
        (RLeak H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj)
        (E D decode) (κ D δ) (γ D) (fun _ => 1) (Bext H D.V (E D decode)) ∧
      {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) (op i) (x z i) (wit z i) (obs z i)}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
          (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞)

end NetTiming.Guarantees
