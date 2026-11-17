module

public import Proofs.Flock.ZkAny.SampledAudit
public import Proofs.Flock.ZkAny.RecordHiding

/-! A two-stage sampled audit's view at any values, its first-stage sample uniform and independent of the key and the sessions (`Flock.Guarantees.SampledAuditTwoAny`). -/

@[expose] public section

namespace FlockSoundness.ZK

open Verity Verity.Game Verity.ValueSalts
open scoped ENNReal

def streamRecOf {K S Ys V O T R : Type} (sv : Service K S Ys V O T R) {r : ℕ} {m : Fin r → ℕ} (v : Fin r → V)
    (st : StreamStack K V S O T R r m) (p : K × ((i : Fin r) → Fin (m i) → S)) : StreamRecord r S O T R :=
  sv.streamHanded p.1 v p.2 (st p.1 v (sv.streamViews p.1 v p.2))

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

/-- `SampledAuditAny` over a stream of `r` registrations under one key, its values and stack functions of the first-stage
sample, the bound summed over the choices the loop can admit. The first-stage sample is uniform and independent of
the key and the sessions; one the auditor chooses is not covered. -/
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

/-- The premises hold at the honest witness's value service, on a boundary and an interior registration of one session
each, the interior values the first-stage bit: the stack registers its sessions and reads no key. -/
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

/-- The bound fails at a stack that reads the key, at `RegisteredStreamLeakage_nontrivial`'s instance: the stack is
always admitted with a `true` commitment, a probability of 1 against a bound of 1/2. -/
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
