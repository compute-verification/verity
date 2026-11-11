module

public import Proofs.Flock.SessionLoop.SampledRelay
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Bridge.Hm96Tree

/-!
# The session loop's error per outcome as a number, and the sampled audit at it

`Loop.anyErr` reads three things that are not numbers: the session's leaf count, hm96's hiding gap at the pinned key,
and the go and unswitched provers' collision finders' success (`Fork`). Each is bounded here by a parameter: a leaf cap
`L`, the bound `δ₁` at which hm96 hides (`Claims.Hm96Hiding`), and the SHA-512 calls per run `q`, the prover's and
`V*`'s together, at which `ecr/sha-512` (`ExpectedCR`) bounds each finder's success by `2q/2^256`.

* `CROnly.hidingGap_le_of_hiding`: where hm96 hides at a finite `δ₁`, its gap is at most `δ₁`, at the event where the
  real pair is no less likely than the ideal one (`CROnly.half_sum_abs_eq`).
* `Fork.pr_le`: `Fork`'s success is at most `2q/2^256` where it meets `ExpectedCR` and each of its runs makes at most
  `q` calls, the prover's and the verifier's together.
* `Loop.Budget`: that at the go and unswitched provers.
* `Loop.errAt`: `Loop.anyErr` at the leaf cap, a bound on the gap and the calls, a number.
* `Loop.anyErr_le`: `anyErr ≤ errAt` at those bounds.
* **`Flock.SecurityProofs.SampledAuditAnyDeployedAt`** and **`SampledAuditAnyRelayDeployedAt`**:
  `SampledAuditAnyDeployed` and `SampledAuditAnyRelayDeployed` with `errAt` at each session's verdict in place of the
  premise `εS`, taking hm96's hiding at `δ₁`, the leaf cap at every statement a session may take and `Loop.Budget` at
  every session.
-/

@[expose] public section

namespace FlockSoundness.CROnly

open Finset
open scoped ENNReal

/-- **Half the ℓ¹ distance of two masses of equal total is one event's advantage**: the event where the first is no
less than the second. -/
theorem half_sum_abs_eq {V : Type} (S : Finset V) (μ ν : V → ℝ) (h : ∑ v ∈ S, μ v = ∑ v ∈ S, ν v) :
    1 / 2 * ∑ v ∈ S, |μ v - ν v| =
      ∑ v ∈ S, (if ν v ≤ μ v then (1 : ℝ) else 0) * μ v - ∑ v ∈ S, (if ν v ≤ μ v then (1 : ℝ) else 0) * ν v := by
  have hpt : ∀ v ∈ S, |μ v - ν v| = 2 * ((if ν v ≤ μ v then (1 : ℝ) else 0) * μ v) -
      2 * ((if ν v ≤ μ v then (1 : ℝ) else 0) * ν v) - (μ v - ν v) := fun v _ => by
    split_ifs with hv
    · rw [abs_of_nonneg (sub_nonneg.2 hv)]; ring
    · rw [abs_of_neg (sub_neg.2 (not_le.1 hv))]; ring
  rw [sum_congr rfl hpt, sum_sub_distrib, sum_sub_distrib, sum_sub_distrib, ← mul_sum, ← mul_sum, h]
  ring

/-- **hm96's gap is at most any finite bound it hides at**: `Hm96Hiding My c δ₁` at the event where `(My y, c y)` is
no less likely than `(U, c y)` bounds the gap, half their ℓ¹ distance (`half_sum_abs_eq`). -/
theorem hidingGap_le_of_hiding {A B C : Type} [Fintype A] [Fintype B] [DecidableEq B] [DecidableEq C] [Nonempty A]
    [Nonempty B] {My : A → B} {c : A → C} {δ₁ : ℝ≥0∞} (h : Claims.Hm96Hiding My c δ₁) (hδ : δ₁ ≠ ⊤) :
    hidingGap My c ≤ δ₁.toReal := by
  have hS : ∀ y, (My y, c y) ∈ (univ : Finset B) ×ˢ univ.image c :=
    fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ y)⟩
  have hS' : ∀ p : B × A, (p.1, c p.2) ∈ (univ : Finset B) ×ˢ univ.image c :=
    fun p => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ p.2)⟩
  have hE : ∀ (e : B × C → Prop) [DecidablePred e],
      ∑ v ∈ (univ : Finset B) ×ˢ univ.image c,
          (if e v then (1 : ℝ) else 0) * ((#{y | (My y, c y) = v} : ℝ) / Fintype.card A) ≤
        ∑ v ∈ (univ : Finset B) ×ˢ univ.image c,
          (if e v then (1 : ℝ) else 0) * ((#{p : B × A | (p.1, c p.2) = v} : ℝ) / Fintype.card (B × A)) +
          δ₁.toReal := by
    intro e _
    have := Claims.Bridge.hiding_toReal h hδ e
    rwa [Claims.Bridge.Pr_eq_card, Claims.Bridge.Pr_eq_card, Fintype.card_subtype, Fintype.card_subtype,
      card_filter_comp_div _ _ hS e, card_filter_comp_div _ _ hS' e] at this
  unfold hidingGap
  rw [half_sum_abs_eq _ _ _ ((sum_card_fiber_div _ _ hS).trans (sum_card_fiber_div _ _ hS').symm)]
  linarith [hE fun v => (#{p : B × A | (p.1, c p.2) = v} : ℝ) / Fintype.card (B × A) ≤
    (#{y | (My y, c y) = v} : ℝ) / Fintype.card A]

end FlockSoundness.CROnly

namespace Claims

/-- **`Fork`'s success at a call budget**: where `Fork` meets `ExpectedCR` and each run of the prover against `V*`
makes at most `q` SHA-512 calls, the prover's and `V*`'s together, it finds a collision with probability at most
`2q/2^256`: its cost is two runs'. -/
theorem Fork.pr_le {N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {P : Prover (N × Ω)} {Vs : VStar Θ} {θ : Θ} {x : Msg} {q : ℕ} (hF : ExpectedCR H (Fork H P Vs θ x))
    (hq : ∀ ω, (runStar H P Vs x ω θ).pLog.length + (runStar H P Vs x ω θ).vLog.length ≤ q) :
    (Fork H P Vs θ x).Pr H ≤ 2 * q / 2 ^ (256 : ℝ) := by
  refine le_trans hF (div_le_div_of_nonneg_right ?_ (by positivity))
  calc Ex (fun s => ((Fork H P Vs θ x).cost s : ℝ)) ≤ Ex (fun _ : (Fork H P Vs θ x).S => (2 * q : ℝ)) :=
        Ex_mono fun s => by
          have h : (Fork H P Vs θ x).cost s ≤ 2 * q := by
            dsimp only [Fork]
            have := hq (s.1, s.2.1); have := hq (s.1, s.2.2)
            omega
          exact_mod_cast h
    _ = 2 * q := Ex_const _

end Claims

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal Claims.ZKServer
open FlockSoundness.Discharge FlockSoundness.CROnly

/-- **The provers' calls at a budget**, at the statement `x`, witness `w`, `V*` and its tape `θ`: at the go and the
unswitched prover, each run against `V*` makes at most `q` SHA-512 calls, the prover's and `V*`'s together (as the
finders' bounds count a run's calls), and the prover's `Fork` meets `ExpectedCR` (`ecr/sha-512`). -/
def Budget (T : CoinLayer.TableData) (vs : (CoinLayer.session loopNr T).SwVals)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (q : ℕ) : Prop :=
  (ExpectedCR CoinOpen.shaH (Fork CoinOpen.shaH (((anyStops T L₀ real).goP vs).prover w) Vs θ x) ∧
    ∀ ω, (runStar CoinOpen.shaH (((anyStops T L₀ real).goP vs).prover w) Vs x ω θ).pLog.length +
      (runStar CoinOpen.shaH (((anyStops T L₀ real).goP vs).prover w) Vs x ω θ).vLog.length ≤ q) ∧
  (ExpectedCR CoinOpen.shaH (Fork CoinOpen.shaH (((anyStops T L₀ real).unP vs).prover w) Vs θ x) ∧
    ∀ ω, (runStar CoinOpen.shaH (((anyStops T L₀ real).unP vs).prover w) Vs x ω θ).pLog.length +
      (runStar CoinOpen.shaH (((anyStops T L₀ real).unP vs).prover w) Vs x ω θ).vLog.length ≤ q)

/-- **The error per outcome as a number**, at `t` repetitions, a leaf cap `L`, `q` SHA-512 calls per run and a bound
`δ₁` on hm96's gap: where the release goes on, `8Lδ₁ + 24tLδ₁ + 2·3^{-t} + 4t·3^t·2q/2^256`; where it stops, the same
with `8t` in place of `24t`. -/
noncomputable def errAt (t L q : ℕ) (δ₁ : ℝ) (o : Bool) : ℝ :=
  8 * L * δ₁ + (if o then 24 else 8) * t * L * δ₁ + 2 * (1 / 3) ^ t + 4 * t * 3 ^ t * (2 * q / 2 ^ (256 : ℝ))

/-- **`anyErr` is at most `errAt`** where the session has at most `L` leaves, hm96's hiding gap is at most `δ₁`, and
the provers' calls are at the budget (`Budget`). -/
theorem anyErr_le (T : CoinLayer.TableData) (vs : (CoinLayer.session loopNr T).SwVals)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (t : ℕ) {x : Msg} {w : Flock.Wit}
    {Θ : Type} {Vs : VStar Θ} {θ : Θ} {L q : ℕ} {δ₁ : ℝ} (hL : (CoinLayer.session loopNr T).nLeaves x ≤ L)
    (hδ : hidingGap ZkView.My ZkView.saltHash ≤ δ₁) (hB : Budget T vs L₀ real x w Vs θ q) (o : Bool) :
    anyErr T vs L₀ real t x w Θ Vs θ o ≤ errAt t L q δ₁ o := by
  have hn : ((CoinLayer.session loopNr T).nLeaves x : ℝ) ≤ L := by exact_mod_cast hL
  have hg := hidingGap_nonneg ZkView.My ZkView.saltHash
  have hnd : ((CoinLayer.session loopNr T).nLeaves x : ℝ) * hidingGap ZkView.My ZkView.saltHash ≤ L * δ₁ :=
    mul_le_mul hn hδ hg (Nat.cast_nonneg _)
  have h4 : (0 : ℝ) ≤ 4 * t * 3 ^ t := by positivity
  obtain ⟨⟨hFg, hcg⟩, hFu, hcu⟩ := hB
  have hPg := Fork.pr_le hFg hcg
  have hPu := Fork.pr_le hFu hcu
  have h8 := mul_le_mul_of_nonneg_left hnd (by norm_num : (0 : ℝ) ≤ 8)
  cases o
  · have h8t := mul_le_mul_of_nonneg_left hnd (by positivity : (0 : ℝ) ≤ 8 * t)
    have hF := mul_le_mul_of_nonneg_left hPu h4
    simp only [anyErr, errAt, Bool.false_eq_true, ↓reduceIte]
    linarith
  · have h24t := mul_le_mul_of_nonneg_left hnd (by positivity : (0 : ℝ) ≤ 24 * t)
    have hF := mul_le_mul_of_nonneg_left hPg h4
    simp only [anyErr, errAt, ↓reduceIte]
    linarith

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **A sampled audit's view at any values, at the deployed session loop, its error a number.**
`SampledAuditAnyDeployed` with its premise `εS` replaced by `Loop.errAt` at a leaf cap `L`, a hiding bound `δ₁` and
`q` SHA-512 calls per run, the prover's and `V*`'s together. In place of that premise it takes that hm96 hides at the pinned key
at a finite `δ₁` (`Claims.Hm96Hiding`), that every statement a session may take has at most `L` leaves, and that at
each session the go and unswitched provers' runs against its `V*` are at the budget, each prover's `Fork` meeting
`ExpectedCR` (`Loop.Budget`). Then the bound is `SampledAuditAnyDeployed`'s with `N·2δ₁` for the record and `errAt` at
each session's verdict, at `δ₁` in reals. -/
def _root_.Flock.Guarantees.SampledAuditAnyDeployedAt : Prop :=
  ∀ {K S V O T R Xs Θ D : Type} [Fintype K] [Fintype S] [Fintype T] [Fintype R] [Nonempty S] [Nonempty T] [Nonempty R]
    [Fintype Xs] {N : ℕ} {CB : ℕ → Type} [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)]
    (sv : Service K S (Fin N → ZkView.Salt) V O T R) (Hc : ZkView.Bits × ByteArray → D)
    (dig : V → Fin N → ZkView.Bits) (g : (Fin N → D) → O),
    (∀ v y, sv.out v y = g fun i => Hc (dig v i + ZkView.My (y i), ZkView.saltHash (y i))) →
    ∀ (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K) (ℓ t : ℕ), 0 < t →
    ∀ (Rl : Rel Claims.Flock.Wit), (CoinLayer.session Loop.loopNr Masked.dataE).Complete Rl →
    ∀ (L₀ : LeanRelease Claims.Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Claims.Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
      (Q : Sessions (RegistrationRecord S O T R) Xs (K × (Fin m → S))
        ((CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker) Claims.Flock.Wit
        (((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) Θ (List (Msg × Out)))
      (eC : ∀ j ρ χ b, CB j ≃
        (Loop.anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t).C
          (Q.x j ρ χ b))
      (L q : ℕ) (δ₁ : ℝ≥0∞),
    ZkServerOwnUser st →
    (∀ j p χ π b, (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal (Q.i j p χ π b).1 ∧
      LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) (Q.i j p χ π b).1
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH (Q.i j p χ π b).2 Rl) →
    (∀ j p χ π b, Rl (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b)) →
    (∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = refK (Q.x j ρ χ b) + 2) →
    Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁ → δ₁ ≠ ⊤ →
    (∀ j ρ χ b, (CoinLayer.session Loop.loopNr Masked.dataE).nLeaves (Q.x j ρ χ b) ≤ L) →
    (∀ j p χ π b, Loop.Budget Masked.dataE (Q.i j p χ π b).1
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) (Q.x j (recOf sv n v st p) χ b)
        (Q.w j p χ π b) (Q.Vs j (recOf sv n v st p) χ b) (Q.θ j (recOf sv n v st p) χ b) q) →
    ∀ (s : Tup (fun _ => Bool) ℓ) (E : RegistrationRecord S O T R → Xs → Tup (fun _ => List (Msg × Out)) ℓ → Prop),
      prCoin (fun z : (K × (Fin m → S)) × Xs × Tup (fun _ =>
          ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
            (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
            (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) ℓ =>
          sv.admits z.1.1 (recOf sv n v st z.1) ∧
            Tup.outs (Q.outc (fun x τ => goesOnAt (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).stop (refK x + 1) τ)
                CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                (recOf sv n v st) z.1 z.2.1)
              (Q.rel CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                (recOf sv n v st) z.1 z.2.1) ℓ z.2.2 = s ∧
              E (recOf sv n v st z.1) z.2.1
                (Tup.rels (Q.rel CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                  (recOf sv n v st) z.1 z.2.1) ℓ z.2.2)) ≤
        m * (avg (fun z : Xs × Tup CB ℓ =>
            sv.ideal n m v₀ (fun ρ => E ρ z.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
              (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ z.1) ℓ s
                z.2)) +
              sv.prfAdvantage n m v (fun ρ => E ρ z.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ z.1) ℓ
                  s z.2))) +
            N * (2 * δ₁)) +
          sv.forgeAdvantage n m v (st k₀ v) +
            Tup.errs (fun _ o => ENNReal.ofReal (Loop.errAt t L q δ₁.toReal o)) ℓ s

theorem SampledAuditAnyDeployedAt : Flock.Guarantees.SampledAuditAnyDeployedAt := by
  intro K S V O T R Xs Θ D _ _ _ _ _ _ _ _ N CB _ _ sv Hc dig g hout n m v v₀ st k₀ ℓ t ht Rl hV L₀ src Q eC L q δ₁
    hown hok hR hk hT1 hδ₁ hL hB s E
  have hδ := FlockSoundness.CROnly.hidingGap_le_of_hiding hT1 hδ₁
  refine (SampledAuditAnyDeployed sv Hc dig g hout n m v v₀ st k₀ ℓ t ht Rl hV L₀ src Q eC
    (fun _ o => ENNReal.ofReal (Loop.errAt t L q δ₁.toReal o)) hown hok hR hk
    (fun j p χ π b o => ENNReal.ofReal_le_ofReal (Loop.anyErr_le _ _ _ _ t (hL _ _ _ _) hδ (hB j p χ π b) o))
    s E).trans ?_
  have hg : ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash) ≤ δ₁ :=
    (ENNReal.ofReal_le_iff_le_toReal hδ₁).2 hδ
  gcongr

/-- **A sampled audit's view at any values, at the deployed held relay's frames, its error a number.**
`SampledAuditAnyRelayDeployed` with its premise `εS` replaced by `Loop.errAt`, at the premises
`SampledAuditAnyDeployedAt` takes in its place. -/
def _root_.Flock.Guarantees.SampledAuditAnyRelayDeployedAt : Prop :=
  ∀ {K S V O T R Xs Θ D : Type} [Fintype K] [Fintype S] [Fintype T] [Fintype R] [Nonempty S] [Nonempty T] [Nonempty R]
    [Fintype Xs] {N : ℕ} {CB : ℕ → Type} [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)]
    (sv : Service K S (Fin N → ZkView.Salt) V O T R) (Hc : ZkView.Bits × ByteArray → D)
    (dig : V → Fin N → ZkView.Bits) (g : (Fin N → D) → O),
    (∀ v y, sv.out v y = g fun i => Hc (dig v i + ZkView.My (y i), ZkView.saltHash (y i))) →
    ∀ (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K) (ℓ t : ℕ), 0 < t →
    ∀ (Rl : Rel Claims.Flock.Wit), (CoinLayer.session Loop.loopNr Masked.dataE).Complete Rl →
    ∀ (L₀ : LeanRelease Claims.Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Claims.Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
      (Q : RelaySessions (RegistrationRecord S O T R) Xs (K × (Fin m → S))
        ((CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker) Claims.Flock.Wit
        (((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) Θ)
      (eC : ∀ j ρ χ b, CB j ≃
        (Loop.anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t).C
          (Q.x j ρ χ b))
      (L q : ℕ) (δ₁ : ℝ≥0∞),
    ZkServerOwnUser st →
    (∀ j p χ π b, (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal (Q.i j p χ π b).1 ∧
      LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) (Q.i j p χ π b).1
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH (Q.i j p χ π b).2 Rl) →
    (∀ j p χ π b, Rl (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b)) →
    (∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = refK (Q.x j ρ χ b) + 2) →
    (∀ j p χ π b, RelayEnvAt (Q.spec j (recOf sv n v st p) χ b) (Q.sts j (recOf sv n v st p) χ b)
      (Q.ins j p χ π b)) →
    (∀ j p χ π b, RelayReads (Q.spec j (recOf sv n v st p) χ b) (Q.ins j p χ π b)
      fun ω => (Q.run CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
        (recOf sv n v st) p χ j π b ω).msgs.map (·.1)) →
    Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁ → δ₁ ≠ ⊤ →
    (∀ j ρ χ b, (CoinLayer.session Loop.loopNr Masked.dataE).nLeaves (Q.x j ρ χ b) ≤ L) →
    (∀ j p χ π b, Loop.Budget Masked.dataE (Q.i j p χ π b).1
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) (Q.x j (recOf sv n v st p) χ b)
        (Q.w j p χ π b) (Q.Vs j (recOf sv n v st p) χ b) (Q.θ j (recOf sv n v st p) χ b) q) →
    ∀ (s : Tup (fun _ => Bool × ℕ) ℓ) (E : RegistrationRecord S O T R → Xs → Tup (fun _ => List ByteArray) ℓ → Prop),
      prCoin (fun z : (K × (Fin m → S)) × Xs × Tup (fun _ =>
          ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
            (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
            (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) ℓ =>
          sv.admits z.1.1 (recOf sv n v st z.1) ∧
            Tup.outs (Q.outc (fun x τ => goesOnAt (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).stop (refK x + 1) τ)
                CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                (recOf sv n v st) z.1 z.2.1)
              (Q.rel z.1 z.2.1) ℓ z.2.2 = s ∧
              E (recOf sv n v st z.1) z.2.1 (Tup.rels (Q.rel z.1 z.2.1) ℓ z.2.2)) ≤
        m * (avg (fun z : Xs × Tup CB ℓ =>
            sv.ideal n m v₀ (fun ρ => E ρ z.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
              (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ z.1) ℓ s
                z.2)) +
              sv.prfAdvantage n m v (fun ρ => E ρ z.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ z.1) ℓ
                  s z.2))) +
            N * (2 * δ₁)) +
          sv.forgeAdvantage n m v (st k₀ v) +
            Tup.errs (fun _ o => ENNReal.ofReal (Loop.errAt t L q δ₁.toReal o.1)) ℓ s

theorem SampledAuditAnyRelayDeployedAt : Flock.Guarantees.SampledAuditAnyRelayDeployedAt := by
  intro K S V O T R Xs Θ D _ _ _ _ _ _ _ _ N CB _ _ sv Hc dig g hout n m v v₀ st k₀ ℓ t ht Rl hV L₀ src Q eC L q δ₁
    hown hok hR hk hE hRd hT1 hδ₁ hL hB s E
  have hδ := FlockSoundness.CROnly.hidingGap_le_of_hiding hT1 hδ₁
  refine (SampledAuditAnyRelayDeployed sv Hc dig g hout n m v v₀ st k₀ ℓ t ht Rl hV L₀ src Q eC
    (fun _ o => ENNReal.ofReal (Loop.errAt t L q δ₁.toReal o)) hown hok hR hk hE hRd
    (fun j p χ π b o => ENNReal.ofReal_le_ofReal (Loop.anyErr_le _ _ _ _ t (hL _ _ _ _) hδ (hB j p χ π b) o))
    s E).trans ?_
  have hg : ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash) ≤ δ₁ :=
    (ENNReal.ofReal_le_iff_le_toReal hδ₁).2 hδ
  gcongr

end Flock.SecurityProofs
