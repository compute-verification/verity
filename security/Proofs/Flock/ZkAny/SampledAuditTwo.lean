module

public import Proofs.Flock.ZkAny.SampledAudit
public import Proofs.Flock.ZkAny.RecordHiding

/-!
# A two-stage sampled audit's view at any values

A two-stage audit registers its boundary values with the ZK server's value service before any coin. The verifier's live
coins then pick the first-stage sample, the replay blocks, uniformly. The serving stack registers the sampled replay
blocks' interior values with the same service, under the same key, before the second stage's coins; fresh live coins
then pick the second-stage sample, and each sampled proof block's proof is one session through the ZK server. The
auditor sees the admitted records, both samples, and the sampled sessions' releases with their outcomes.

`Flock.Guarantees.SampledAuditTwoAny` states that view over a stream of `r` registrations of one value service (every
registration on the server's record up to the audit's interior registration), whose values and whose stack's choices are
functions of the first-stage sample `χ₁`: the interior values are the sampled replay blocks'. Per outcome of the sampled
sessions, on every event, the view is no likelier than a simulator's that reads none of the values: an ideal stream of
dummy values `v₀ χ₁` (a function of the public sample, as the interior registration's layout is), both samples, and the
sessions' simulator given the records, the samples and the outcome. The terms are

* for each choice `d` of the stack's sessions that the loop can admit (`streamChoices`), the simulator's probability on
  an ideal stream of the dummy values plus the PRF gap of the key-derived salts at `d`, averaged over both samples and
  the simulator's coins, plus `δ`, how far an ideal stream of the values is from one of the dummy values (at hm96's
  leaves `ideal_hm96_stream`'s `r·N·2δ₁`, the dummy values of the values' layouts);
* the forge term (`streamForgeAdvantage`), averaged over the first-stage sample;
* and the sessions' error at the outcome, `ε s`.

One key salts every registration, so the PRF and forge terms are the stream's
(`Verity.Guarantees.RegisteredStreamLeakage`) at each first-stage sample, not two registrations' bounds added. The
first-stage sample is uniform and independent of the key and the sessions; one the auditor chooses is not covered.

`Flock.SecurityProofs.SampledAuditTwoAny` proves it with the one-stage audit's swap (`prCoin_admits_perOutcome`) at the
registrations' randomness and the first-stage sample together (`prCoin_admits_perOutcome_two`). At each first-stage
sample, second-stage sample and simulator's coins, what is left is the probability that the loop admits a stream meeting
one event, which `RegisteredStreamLeakage` bounds at that sample's values and stack; the ideal stream of the values is
then within `δ` of the dummy's.

* `streamRecOf`: the records a stream stack hands the loop.
* `prCoin_admits_perOutcome_two`: the swap, a first-stage sample between the registrations and the sessions.
* `ideal_hm96_stream`: an ideal stream hides its values at hm96 commitments, `r·N·2δ₁` over `r` registrations of `N`
  leaves each (`leaves_hybrid` over the registrations, `leaves_swap_values` within each).
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity Verity.Game Verity.ValueSalts
open scoped ENNReal

/-- **The records a stream stack hands the loop**, registration `i`'s values `v i`, at the key and every registration's
sessions `p`. -/
def streamRecOf {K S Ys V O T R : Type} (sv : Service K S Ys V O T R) {r : ℕ} {m : Fin r → ℕ} (v : Fin r → V)
    (st : StreamStack K V S O T R r m) (p : K × ((i : Fin r) → Fin (m i) → S)) : StreamRecord r S O T R :=
  sv.streamHanded p.1 v p.2 (st p.1 v (sv.streamViews p.1 v p.2))

/-- **The sessions' release swapped for its simulator, a first-stage sample between.** Where, at every value `p`
of the registrations' randomness and `χ₁` of the first-stage sample and `χ₂` of the second, the sessions' release is
simulated per outcome by `Sim` given the record `ρ p χ₁` and both samples, the probability that `p` is admitted at `χ₁`
and the audit shows the outcome `s` and an event on the record, both samples and the bytes is at most, averaged over
both samples and the simulator's coins, the probability over `p` alone that it is admitted at `χ₁` and its record meets
that event with the simulator's bytes, plus `ε s`. -/
theorem prCoin_admits_perOutcome_two {P X₁ X₂ Ω Ωs St B Rec : Type} [Fintype P] [Fintype X₁] [Fintype X₂] [Fintype Ω]
    [Fintype Ωs] (adm : P → X₁ → Prop) (ρ : P → X₁ → Rec) (out : P → X₁ → X₂ → Ω → St)
    (rel : P → X₁ → X₂ → Ω → B) (Sim : Rec → X₁ → X₂ → St → Ωs → B) (ε : St → ℝ≥0∞)
    (hS : ∀ p χ₁ χ₂, PerOutcome (out p χ₁ χ₂) (rel p χ₁ χ₂) (Sim (ρ p χ₁) χ₁ χ₂) ε) (s : St)
    (E : Rec → X₁ → X₂ → B → Prop) :
    prCoin (fun q : P × X₁ × X₂ × Ω => adm q.1 q.2.1 ∧ out q.1 q.2.1 q.2.2.1 q.2.2.2 = s ∧
        E (ρ q.1 q.2.1) q.2.1 q.2.2.1 (rel q.1 q.2.1 q.2.2.1 q.2.2.2)) ≤
      avg (fun q : X₁ × X₂ × Ωs => prCoin fun p : P => adm p q.1 ∧
        E (ρ p q.1) q.1 q.2.1 (Sim (ρ p q.1) q.1 q.2.1 s q.2.2)) + ε s := by
  let e : P × X₁ × X₂ × Ω ≃ (X₁ × P) × X₂ × Ω :=
    { toFun := fun q => ((q.2.1, q.1), q.2.2.1, q.2.2.2)
      invFun := fun q => (q.1.2, q.1.1, q.2.1, q.2.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have h1 := prCoin_admits_perOutcome (fun p : X₁ × P => adm p.2 p.1) (fun p => (ρ p.2 p.1, p.1))
    (fun p => out p.2 p.1) (fun p => rel p.2 p.1) (fun ρχ χ₂ => Sim ρχ.1 ρχ.2 χ₂) ε (fun p χ₂ => hS p.2 p.1 χ₂) s
    (fun ρχ χ₂ bs => E ρχ.1 ρχ.2 χ₂ bs)
  have h2 : ∀ q : X₂ × Ωs, prCoin (fun p : X₁ × P => adm p.2 p.1 ∧
      E (ρ p.2 p.1) p.1 q.1 (Sim (ρ p.2 p.1) p.1 q.1 s q.2)) =
      avg fun χ₁ => prCoin fun p : P => adm p χ₁ ∧ E (ρ p χ₁) χ₁ q.1 (Sim (ρ p χ₁) χ₁ q.1 s q.2) := fun q =>
    prCoin_prod_split (fun (χ₁ : X₁) (p : P) => adm p χ₁ ∧ E (ρ p χ₁) χ₁ q.1 (Sim (ρ p χ₁) χ₁ q.1 s q.2))
  calc prCoin (fun q : P × X₁ × X₂ × Ω => adm q.1 q.2.1 ∧ out q.1 q.2.1 q.2.2.1 q.2.2.2 = s ∧
        E (ρ q.1 q.2.1) q.2.1 q.2.2.1 (rel q.1 q.2.1 q.2.2.1 q.2.2.2))
      = prCoin (fun q : (X₁ × P) × X₂ × Ω => adm q.1.2 q.1.1 ∧ out q.1.2 q.1.1 q.2.1 q.2.2 = s ∧
          E (ρ q.1.2 q.1.1) q.1.1 q.2.1 (rel q.1.2 q.1.1 q.2.1 q.2.2)) :=
        prCoin_congr_equiv e (fun q : (X₁ × P) × X₂ × Ω => adm q.1.2 q.1.1 ∧ out q.1.2 q.1.1 q.2.1 q.2.2 = s ∧
          E (ρ q.1.2 q.1.1) q.1.1 q.2.1 (rel q.1.2 q.1.1 q.2.1 q.2.2))
    _ ≤ avg (fun q : X₂ × Ωs => prCoin fun p : X₁ × P => adm p.2 p.1 ∧
          E (ρ p.2 p.1) p.1 q.1 (Sim (ρ p.2 p.1) p.1 q.1 s q.2)) + ε s := h1
    _ = avg (fun q : X₂ × Ωs => avg fun χ₁ => prCoin fun p : P => adm p χ₁ ∧
          E (ρ p χ₁) χ₁ q.1 (Sim (ρ p χ₁) χ₁ q.1 s q.2)) + ε s :=
        congrArg (fun f => avg f + ε s) (funext h2)
    _ = avg (fun χ₁ : X₁ => avg fun q : X₂ × Ωs => prCoin fun p : P => adm p χ₁ ∧
          E (ρ p χ₁) χ₁ q.1 (Sim (ρ p χ₁) χ₁ q.1 s q.2)) + ε s :=
        congrArg (· + ε s) (avg_swap fun (q : X₂ × Ωs) (χ₁ : X₁) => prCoin fun p : P => adm p χ₁ ∧
          E (ρ p χ₁) χ₁ q.1 (Sim (ρ p χ₁) χ₁ q.1 s q.2))
    _ = avg (fun q : X₁ × X₂ × Ωs => prCoin fun p : P => adm p q.1 ∧
          E (ρ p q.1) q.1 q.2.1 (Sim (ρ p q.1) q.1 q.2.1 s q.2.2)) + ε s :=
        congrArg (· + ε s) (avg_prod_split fun q : X₁ × X₂ × Ωs => prCoin fun p : P => adm p q.1 ∧
          E (ρ p q.1) q.1 q.2.1 (Sim (ρ p q.1) q.1 q.2.1 s q.2.2)).symm

/-- **An ideal stream hides its values, at hm96 commitments.** Where the value service's commitment of the values `v`
under the salts `y` is a function `g`, of the values' public shape `sh v`, of `N` hm96 leaves, leaf `i` being
`H(dig v i + M·y_i, c(y_i))` under its own uniform salt `y_i`, an ideal stream of `r` registrations' values `v` is
within `r·N·2δ₁` of one of values `v₀` of the same shapes on every event, whichever registrations `P` it passes over,
`δ₁` the hiding gap: each registration's commitment, of its own salts, is swapped at `N·2δ₁` (`leaves_swap_values`),
one registration at a time (`leaves_hybrid`). -/
theorem ideal_hm96_stream {K S Ys B C D V O T R Sh : Type} [Fintype S] [Nonempty S] [Fintype Ys] [Nonempty Ys]
    [Fintype T] [Nonempty T] [Fintype R] [Nonempty R] [AddCommGroup B] [Fintype B] [Nonempty B] [DecidableEq B]
    [DecidableEq C] {N : ℕ} (sv : Service K S (Fin N → Ys) V O T R) (My : Ys → B) (c : Ys → C) (H : B × C → D)
    (dig : V → Fin N → B) (sh : V → Sh) (g : Sh → (Fin N → D) → O)
    (hout : ∀ v y, sv.out v y = g (sh v) fun i => H (dig v i + My (y i), c (y i))) {r : ℕ} (m : Fin r → ℕ)
    (v v₀ : Fin r → V) (hsh : ∀ i, sh (v i) = sh (v₀ i)) (P : Finset (Fin r)) (A : StreamRecord r S O T R → Prop) :
    sv.streamIdeal m v P A ≤
      sv.streamIdeal m v₀ P A + r * (N * (2 * ENNReal.ofReal (CROnly.hidingGap My c))) := by
  let e : (Fin r → S × T × R) × (Fin r → Fin N → Ys) ≃ (Fin r → S × (Fin N → Ys) × T × R) :=
    { toFun := fun q i => ((q.1 i).1, q.2 i, (q.1 i).2.1, (q.1 i).2.2)
      invFun := fun p => (fun i => ((p i).1, (p i).2.2.1, (p i).2.2.2), fun i => (p i).2.1)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hv : ∀ u : Fin r → V, sv.streamIdeal m u P A =
      prCoin (fun q : (Fin r → S × T × R) × (Fin r → Fin N → Ys) =>
        A fun i => if i ∈ P then none
          else some ((i : ℕ), m i, ((q.1 i).1, sv.out (u i) (q.2 i), (q.1 i).2.1), (q.1 i).2.2)) := fun u => by
    unfold Service.streamIdeal
    exact (prCoin_congr_equiv e _).symm
  rw [hv v, hv v₀]
  refine leaves_hybrid (fun (a : {a : V × V // sh a.1 = sh a.2}) (y : Fin N → Ys) => sv.out a.1.1 y)
    (fun a y => sv.out a.1.2 y) (δ := N * (2 * ENNReal.ofReal (CROnly.hidingGap My c))) (fun x G => ?_) r
    (fun (_ : Fin r → S × T × R) i => ⟨(v i, v₀ i), hsh i⟩)
    (fun (ω : Fin r → S × T × R) (os : Fin r → O) => A fun i => if i ∈ P then none
      else some ((i : ℕ), m i, ((ω i).1, os i, (ω i).2.1), (ω i).2.2))
  have h₁ : ∀ ω y, sv.out (x ω).1.1 y = g (sh (x ω).1.1) fun i => H (dig (x ω).1.1 i + My (y i), c (y i)) :=
    fun ω y => hout _ y
  have h₂ : ∀ ω y, sv.out (x ω).1.2 y = g (sh (x ω).1.1) fun i => H (dig (x ω).1.2 i + My (y i), c (y i)) :=
    fun ω y => by rw [hout, (x ω).2]
  simp only [h₁, h₂]
  exact leaves_swap_values My c H N (fun ω i => dig (x ω).1.1 i) (fun ω i => dig (x ω).1.2 i)
    (fun ω ls => G ω (g (sh (x ω).1.1) ls))

end FlockSoundness.ZK

namespace Flock.SecurityProofs

open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open scoped ENNReal

/-- **A two-stage sampled audit's view at any values, per outcome of its sampled sessions.** For every value service
`sv` with finite keys, sessions, salts, tags and record tags; every stream of `r` registrations, registration `i` of an
epoch of `m i` sessions, at most `b` of them passed over; every finite space `X₁` of first-stage samples, one taken
uniformly after the boundary registration, and values `v`, dummy values `v₀` and a stack `st` that are functions of
the first-stage sample; a key `k₀`; every finite space `X₂` of second-stage samples, one taken uniformly after the
interior registration; and the sampled sessions' outcome `out` and released bytes `rel`, functions of the
registrations' randomness (the key and every registration's sessions), both samples and the sessions' own randomness:

if the stack reads no key (`ZkServerOwnUser`) at any first-stage sample, the sessions' release is simulated per outcome
(`PerOutcome`) within `ε` at every key, sessions and samples by `Sim` given the admitted records and both samples, and
at every first-stage sample an ideal stream of the values is within `δ` of one of the dummy values there on every
event, then at every outcome `s` and event `E` on the records, both samples and the bytes, the probability that the
loop admits the stream and the audit shows `s` and `E` is at most the sum, over the choices `d` the loop can admit, of
(the average, over both samples and `Sim`'s coins, of `E` on an ideal stream of the dummy values with `Sim`'s bytes
plus the PRF gap at `d` there, plus `δ`), plus the probability, averaged over the first-stage sample, that the stack
run at `k₀` forges a record tag, plus `ε s`. -/
def _root_.Flock.Guarantees.SampledAuditTwoAny : Prop :=
  ∀ {K S Y V O T R X₁ X₂ Ω Ωs St B : Type} [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R] [Fintype X₁]
    [Fintype X₂] [Fintype Ω] [Fintype Ωs] (sv : Service K S Y V O T R) (r : ℕ) (m : Fin r → ℕ) (b : ℕ)
    (v v₀ : X₁ → Fin r → V) (st : X₁ → StreamStack K V S O T R r m) (k₀ : K)
    (out : K × ((i : Fin r) → Fin (m i) → S) → X₁ → X₂ → Ω → St)
    (rel : K × ((i : Fin r) → Fin (m i) → S) → X₁ → X₂ → Ω → B)
    (Sim : StreamRecord r S O T R → X₁ → X₂ → St → Ωs → B) (ε : St → ℝ≥0∞) (δ : ℝ≥0∞),
    (∀ χ₁, ZkServerOwnUser (st χ₁)) →
    (∀ p χ₁ χ₂, PerOutcome (out p χ₁ χ₂) (rel p χ₁ χ₂) (Sim (streamRecOf sv (v χ₁) (st χ₁) p) χ₁ χ₂) ε) →
    (∀ χ₁ P A, sv.streamIdeal m (v χ₁) P A ≤ sv.streamIdeal m (v₀ χ₁) P A + δ) →
    ∀ (s : St) (E : StreamRecord r S O T R → X₁ → X₂ → B → Prop),
      prCoin (fun q : (K × ((i : Fin r) → Fin (m i) → S)) × X₁ × X₂ × Ω =>
          sv.streamAdmits q.1.1 b (streamRecOf sv (v q.2.1) (st q.2.1) q.1) ∧
            out q.1 q.2.1 q.2.2.1 q.2.2.2 = s ∧
              E (streamRecOf sv (v q.2.1) (st q.2.1) q.1) q.2.1 q.2.2.1 (rel q.1 q.2.1 q.2.2.1 q.2.2.2)) ≤
        (∑ d ∈ streamChoices r m b,
          (avg (fun q : X₁ × X₂ × Ωs =>
              sv.streamIdeal m (v₀ q.1) (passed d) (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2)) +
                sv.streamPrfAdvantage (v q.1) d (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2))) + δ)) +
          avg (fun χ₁ => sv.streamForgeAdvantage (v χ₁) (st χ₁ k₀ (v χ₁))) + ε s

theorem SampledAuditTwoAny : Flock.Guarantees.SampledAuditTwoAny := by
  intro K S Y V O T R X₁ X₂ Ω Ωs St B _ _ _ _ _ _ _ _ _ sv r m b v v₀ st k₀ out rel Sim ε δ hown hS hδ s E
  have hc : ∀ {C : Type} [Fintype C] (x : ℝ≥0∞), avg (fun _ : C => x) ≤ x := by
    intro C _ x
    rcases isEmpty_or_nonempty C with hC | hC
    · simp [avg]
    · exact (avg_const x).le
  have hsum : ∀ {C ι : Type} [Fintype C] (t : Finset ι) (f : ι → C → ℝ≥0∞),
      avg (fun c => ∑ i ∈ t, f i c) = ∑ i ∈ t, avg (f i) := by
    intro C ι _ t f
    simp only [avg, Finset.mul_sum]
    exact Finset.sum_comm
  have hstream : ∀ q : X₁ × X₂ × Ωs,
      prCoin (fun p : K × ((i : Fin r) → Fin (m i) → S) =>
          sv.streamAdmits p.1 b (streamRecOf sv (v q.1) (st q.1) p) ∧
            E (streamRecOf sv (v q.1) (st q.1) p) q.1 q.2.1
              (Sim (streamRecOf sv (v q.1) (st q.1) p) q.1 q.2.1 s q.2.2)) ≤
        (∑ d ∈ streamChoices r m b,
          ((sv.streamIdeal m (v₀ q.1) (passed d) (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2)) +
            sv.streamPrfAdvantage (v q.1) d (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2))) + δ)) +
          sv.streamForgeAdvantage (v q.1) (st q.1 k₀ (v q.1)) := by
    intro q
    refine (Verity.SecurityProofs.RegisteredStreamLeakage sv r m b (v q.1) (st q.1)
      (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2)) k₀ (hown q.1)).trans
      (add_le_add (Finset.sum_le_sum fun d _ => ?_) le_rfl)
    exact (add_le_add (hδ q.1 (passed d) _) le_rfl).trans_eq (add_right_comm _ _ _)
  refine (prCoin_admits_perOutcome_two (fun p χ₁ => sv.streamAdmits p.1 b (streamRecOf sv (v χ₁) (st χ₁) p))
    (fun p χ₁ => streamRecOf sv (v χ₁) (st χ₁) p) out rel Sim ε hS s E).trans (add_le_add ?_ le_rfl)
  calc avg (fun q : X₁ × X₂ × Ωs => prCoin fun p : K × ((i : Fin r) → Fin (m i) → S) =>
          sv.streamAdmits p.1 b (streamRecOf sv (v q.1) (st q.1) p) ∧
            E (streamRecOf sv (v q.1) (st q.1) p) q.1 q.2.1
              (Sim (streamRecOf sv (v q.1) (st q.1) p) q.1 q.2.1 s q.2.2))
      ≤ avg (fun q : X₁ × X₂ × Ωs => (∑ d ∈ streamChoices r m b,
          ((sv.streamIdeal m (v₀ q.1) (passed d) (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2)) +
            sv.streamPrfAdvantage (v q.1) d (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2))) + δ)) +
          sv.streamForgeAdvantage (v q.1) (st q.1 k₀ (v q.1))) := avg_mono hstream
    _ = (∑ d ∈ streamChoices r m b, avg fun q : X₁ × X₂ × Ωs =>
          (sv.streamIdeal m (v₀ q.1) (passed d) (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2)) +
            sv.streamPrfAdvantage (v q.1) d (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2))) + δ) +
          avg (fun q : X₁ × X₂ × Ωs => sv.streamForgeAdvantage (v q.1) (st q.1 k₀ (v q.1))) := by
        rw [avg_add, hsum]
    _ ≤ (∑ d ∈ streamChoices r m b,
          (avg (fun q : X₁ × X₂ × Ωs =>
              sv.streamIdeal m (v₀ q.1) (passed d) (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2)) +
                sv.streamPrfAdvantage (v q.1) d (fun ρs => E ρs q.1 q.2.1 (Sim ρs q.1 q.2.1 s q.2.2))) + δ)) +
          avg (fun χ₁ => sv.streamForgeAdvantage (v χ₁) (st χ₁ k₀ (v χ₁))) := by
        refine add_le_add (Finset.sum_le_sum fun d _ => ?_) ?_
        · rw [avg_add]
          exact add_le_add le_rfl (hc δ)
        · calc avg (fun q : X₁ × X₂ × Ωs => sv.streamForgeAdvantage (v q.1) (st q.1 k₀ (v q.1)))
              = avg (fun χ₁ : X₁ => avg fun _ : X₂ × Ωs => sv.streamForgeAdvantage (v χ₁) (st χ₁ k₀ (v χ₁))) :=
                avg_prod_split _
            _ ≤ avg (fun χ₁ => sv.streamForgeAdvantage (v χ₁) (st χ₁ k₀ (v χ₁))) := avg_mono fun χ₁ => hc _

-- The honest witness's value service: two keys, one session, salt and tag, a bit of values; every public commitment is
-- `false`, and the record tag is the key.
local notation "toyB" => (Service.mk (fun _ _ => ()) (fun _ _ => false) (fun _ _ _ => ()) (fun k _ _ _ => k) :
  Service Bool Unit Unit Bool Bool Unit Bool)

/-- **The premises hold at a stack that registers its sessions, the interior values following the first-stage
sample.** At the honest witness's value service, on a stream of a boundary and an interior registration, each of an
epoch of one session, the interior values the first-stage bit: the stack that registers each session reads no key,
sessions that release nothing are simulated exactly, and an ideal stream of the values is one of `false`'s, the
commitments reading no value. -/
theorem SampledAuditTwoAny_honest :
    (∀ _ : Bool, ZkServerOwnUser
        ((fun _ _ _ _ => some (Sum.inl 0)) : StreamStack Bool Bool Unit Bool Unit Bool 2 fun _ => 1)) ∧
      (∀ (_ : Bool × ((i : Fin 2) → Fin 1 → Unit)) (_ : Bool) (_ : Unit),
        PerOutcome (fun _ : Unit => ()) (fun _ : Unit => ()) (fun (_ : Unit) (_ : Unit) => ()) fun _ => 0) ∧
      ∀ (χ₁ : Bool) (P : Finset (Fin 2)) (A : StreamRecord 2 Unit Bool Unit Bool → Prop),
        Service.streamIdeal toyB (fun _ => 1) (fun i => if (i : ℕ) = 1 then χ₁ else false) P A ≤
          Service.streamIdeal toyB (fun _ => 1) (fun _ => false) P A + 0 :=
  ⟨fun _ _ _ => rfl, fun _ _ _ _ _ => le_add_right (prCoin_mono fun _ h => h.2), fun _ _ _ => le_add_right le_rfl⟩

-- The nontrivial witness's instance, `RegisteredStreamLeakage_nontrivial`'s: its value service, and its stack that
-- hands a record of its own, public commitment `true` and record tag the key.
local notation "toyS₁" => (Service.mk (fun _ _ => ()) (fun _ _ => false) (fun _ _ _ => ()) (fun k _ _ _ => k) :
  Service Bool Unit Unit Unit Bool Unit Bool)
local notation "toyF₁" =>
  (fun k _ _ _ => some (Sum.inr (0, 1, ((), true, ()), k)) : StreamStack Bool Unit Unit Bool Unit Bool 1 fun _ => 1)

/-- **The bound fails at a stack that reads the key.** At `RegisteredStreamLeakage_nontrivial`'s instance, one
registration of an epoch of one session, none passed over, with one first-stage and one second-stage sample and sessions
that release nothing, simulated exactly, and the values hidden exactly: the stack that hands a record of its own, public
commitment `true` and record tag the key (`RegisteredStreamLeakage_nontrivial_stack`: it reads the key), is always
admitted with a `true` commitment, while the bound is that stream's, `1 ≤ 0 + 1/2`. -/
theorem SampledAuditTwoAny_nontrivial :
    ¬ prCoin (fun q : (Bool × ((i : Fin 1) → Fin 1 → Unit)) × Unit × Unit × Unit =>
          Service.streamAdmits toyS₁ q.1.1 0 (streamRecOf toyS₁ (fun _ => ()) toyF₁ q.1) ∧ () = () ∧
            ∃ ρ, streamRecOf toyS₁ (fun _ => ()) toyF₁ q.1 0 = some ρ ∧ ρ.2.2.1.2.1 = true) ≤
        (∑ d ∈ streamChoices 1 (fun _ => 1) 0,
          (avg (fun _ : Unit × Unit × Unit =>
              Service.streamIdeal toyS₁ (fun _ => 1) (fun _ => ()) (passed d)
                  (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) +
                Service.streamPrfAdvantage toyS₁ (fun _ => ()) d
                  (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true)) +
            0)) +
          avg (fun _ : Unit => Service.streamForgeAdvantage toyS₁ (fun _ => ()) (toyF₁ false fun _ => ())) + 0 := by
  intro h
  have hone : ∀ p : Bool × ((i : Fin 1) → Fin 1 → Unit),
      Service.streamAdmits toyS₁ p.1 0 (streamRecOf toyS₁ (fun _ => ()) toyF₁ p) ∧
        ∃ ρ, streamRecOf toyS₁ (fun _ => ()) toyF₁ p 0 = some ρ ∧ ρ.2.2.1.2.1 = true := by
    intro p
    refine ⟨⟨?_, ?_⟩, _, rfl, rfl⟩
    · simp [streamRecOf, Service.streamHanded]
    · intro i ρ h
      obtain rfl : i = 0 := Subsingleton.elim _ _
      cases h
      exact ⟨rfl, rfl⟩
  have hL : prCoin (fun q : (Bool × ((i : Fin 1) → Fin 1 → Unit)) × Unit × Unit × Unit =>
      Service.streamAdmits toyS₁ q.1.1 0 (streamRecOf toyS₁ (fun _ => ()) toyF₁ q.1) ∧ () = () ∧
        ∃ ρ, streamRecOf toyS₁ (fun _ => ()) toyF₁ q.1 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 1 :=
    Verity.ValueSalts.prCoin_of_forall fun q => ⟨(hone q.1).1, rfl, (hone q.1).2⟩
  have hreal : Service.streamReal toyS₁ 0 (fun _ => ()) toyF₁ (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 1 :=
    Verity.ValueSalts.prCoin_of_forall hone
  refine Verity.SecurityProofs.RegisteredStreamLeakage_nontrivial
    (hreal.trans_le (hL.symm.trans_le (h.trans_eq ?_)))
  simp only [avg_const, add_zero]

end Flock.SecurityProofs
