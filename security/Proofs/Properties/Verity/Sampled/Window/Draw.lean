module

public import Proofs.Properties.Verity.Sampled.Window.Walk
public import Proofs.Properties.Verity.Sampled.CoinMap
public import Proofs.Properties.Verity.Sampled.Members.Widen

/-! A window's one sample, its members' parts, and the coin map over the window. -/

@[expose] public section

namespace Flock.Guarantees.JointAuditV5.Laid

variable {w : JointAuditV5}

theorem inj_of_mem (h : w.Laid) : ∀ m ∈ w.ms, Function.Injective m.emb := fun m hm => by
  obtain ⟨k, rfl⟩ := List.mem_iff_get.1 hm
  exact h.emb_injective k

theorem pairwise (h : w.Laid) : w.ms.Pairwise fun m m' => ∀ u u', m.emb u ≠ m'.emb u' :=
  List.pairwise_iff_get.2 fun _ _ hij _ _ e => absurd (h.member_eq e) (ne_of_lt hij)

theorem cover (h : w.Laid) (v : Fin w.nJ) : ∃ m ∈ w.ms, ∃ u, m.emb u = v := by
  obtain ⟨k, u, hu⟩ := h.exists_emb v
  exact ⟨w.ms.get k, List.get_mem _ _, u, hu⟩

end Flock.Guarantees.JointAuditV5.Laid

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

theorem Member.mem_part {nJ : ℕ} (m : Member nJ) (S : Finset (Fin nJ)) (u : Fin m.n) :
    u ∈ m.part S ↔ m.emb u ∈ S := by
  have h : ∀ (m' : Flock.Guarantees.MemberV5 ByteArray nJ) (u' : Fin m'.n), u' ∈ m'.part S ↔ m'.emb u' ∈ S :=
    fun m' u' => by simp [Flock.Guarantees.MemberV5.part]
  exact h (m.toV5 0) u

structure ExecutionPhase.Partitions (W : ExecutionPhase) : Prop where
  inj : ∀ m ∈ W.ms, Function.Injective m.emb
  disjoint : W.ms.Pairwise fun m m' => ∀ u u', m.emb u ≠ m'.emb u'
  cover : ∀ v : Fin W.nJ, ∃ m ∈ W.ms, ∃ u, m.emb u = v

open Classical in
theorem Member.card_part {nJ : ℕ} (m : Member nJ) (h : Function.Injective m.emb) (S : Finset (Fin nJ)) :
    (m.part S).card = (S.filter fun v => ∃ u, m.emb u = v).card := by
  rw [← Finset.card_image_of_injective _ h]
  refine congrArg Finset.card (Finset.ext fun v => ?_)
  rw [Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨u, hu, rfl⟩
    exact ⟨(m.mem_part S u).1 hu, u, rfl⟩
  · rintro ⟨hv, u, rfl⟩
    exact ⟨u, (m.mem_part S u).2 hv, rfl⟩

open Classical in
theorem sum_part_card_filter {nJ : ℕ} (S : Finset (Fin nJ)) : ∀ l : List (Member nJ),
    (∀ m ∈ l, Function.Injective m.emb) → (l.Pairwise fun m m' => ∀ u u', m.emb u ≠ m'.emb u') →
    (l.map fun m => (m.part S).card).sum = (S.filter fun v => ∃ m ∈ l, ∃ u, m.emb u = v).card
  | [], _, _ => by
    rw [List.map_nil, List.sum_nil, Finset.filter_false_of_mem fun v _ h => by
      obtain ⟨m, hm, _⟩ := h
      exact List.not_mem_nil hm]
    rfl
  | m :: l, hi, hp => by
    have hd : Disjoint (S.filter fun v => ∃ u, m.emb u = v) (S.filter fun v => ∃ m' ∈ l, ∃ u, m'.emb u = v) :=
      Finset.disjoint_left.2 fun v hv hv' => by
        obtain ⟨_, u, rfl⟩ := Finset.mem_filter.1 hv
        obtain ⟨_, m', hm', u', hu'⟩ := Finset.mem_filter.1 hv'
        exact (List.pairwise_cons.1 hp).1 m' hm' u u' hu'.symm
    rw [List.map_cons, List.sum_cons, sum_part_card_filter S l (fun m' hm' => hi m' (List.mem_cons_of_mem _ hm'))
      (List.pairwise_cons.1 hp).2, m.card_part (hi m List.mem_cons_self) S, ← Finset.card_union_of_disjoint hd]
    refine congrArg Finset.card (Finset.ext fun v => ?_)
    rw [Finset.mem_union, Finset.mem_filter, Finset.mem_filter, Finset.mem_filter]
    constructor
    · rintro (⟨hv, u, hu⟩ | ⟨hv, m', hm', u, hu⟩)
      · exact ⟨hv, m, List.mem_cons_self, u, hu⟩
      · exact ⟨hv, m', List.mem_cons_of_mem _ hm', u, hu⟩
    · rintro ⟨hv, m', hm', u, hu⟩
      rcases List.mem_cons.1 hm' with rfl | hm'
      · exact Or.inl ⟨hv, u, hu⟩
      · exact Or.inr ⟨hv, m', hm', u, hu⟩

theorem ExecutionPhase.sum_part_card (W : ExecutionPhase) (h : W.Partitions) (S : Finset (Fin W.nJ)) :
    (W.ms.map fun m => (m.part S).card).sum = S.card := by
  classical
  rw [sum_part_card_filter S W.ms h.inj h.disjoint]
  exact congrArg Finset.card (Finset.filter_true_of_mem fun v _ => h.cover v)

def ExecutionPhase.Laid (W : ExecutionPhase) : Prop := (W.joint 0).Laid

theorem ExecutionPhase.Laid.partitions {W : ExecutionPhase} (h : W.Laid) : W.Partitions where
  inj m hm := Flock.Guarantees.JointAuditV5.Laid.inj_of_mem h (m.toV5 0) (List.mem_map_of_mem hm)
  disjoint := List.pairwise_map.1 (Flock.Guarantees.JointAuditV5.Laid.pairwise h)
  cover v := by
    obtain ⟨m', hm', u, hu⟩ := Flock.Guarantees.JointAuditV5.Laid.cover h v
    obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hm'
    exact ⟨m, hm, u, hu⟩

theorem ExecutionPhase.Laid.sum_part_card {W : ExecutionPhase} (h : W.Laid) (S : Finset (Fin W.nJ)) :
    (W.ms.map fun m => (m.part S).card).sum = S.card :=
  W.sum_part_card h.partitions S

theorem ExecutionPhase.avg_design (W : ExecutionPhase) (t : ℕ) (k : Fin (W.joint t).ms.length)
    (G : Finset (Fin ((W.joint t).ms.get k).n) → ℝ≥0∞) :
    avg (fun ω : ((W.joint t).design k).Ω => G (((W.joint t).design k).draw ω)) =
      avg fun ω : W.L.Ω => G (((W.joint t).ms.get k).part (W.L.draw ω)) :=
  rfl

theorem ExecutionPhase.sampleMiss_extend (W : ExecutionPhase) (S : Widening) : (W.extend S).sampleMiss = W.sampleMiss :=
  rfl

theorem ExecutionPhase.joint_extend (W : ExecutionPhase) (S : Widening) (t : ℕ) : (W.extend S).joint t = (W.joint t).extend S :=
  rfl

/-- By `dsimp`, not `rfl`: elaborating the two samples as definitionally equal reduces their `match` and
evaluates `sampleUnits`. -/
theorem ExecutionPhase.sample_extend (W : ExecutionPhase) (S : Widening) (ω : (W.extend S).L.Ω) :
    (W.extend S).L.draw ω = W.L.draw ω.1 := by
  dsimp only [ExecutionPhase.L, Flock.Bridge.execSel, ExecutionPhase.extend]

theorem ExecutionPhase.eps_extend (W : ExecutionPhase) (S : Widening) (t : ℕ) : ((W.extend S).joint t).eps = (W.joint t).eps :=
  (congrArg Flock.Guarantees.JointAuditV5.eps (W.joint_extend S t)).trans ((W.joint t).eps_extend S)

theorem ExecutionPhase.sessionDelta_extend (W : ExecutionPhase) (S : Widening) (t : ℕ) :
    (W.extend S).sessionDelta t = W.sessionDelta t := by
  unfold ExecutionPhase.sessionDelta
  rw [W.sampleMiss_extend S, W.eps_extend S t, W.rootsErr_extend S]

section SessionErr

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : PosDecoders g σ R Reg) (J : ℕ)
  (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞)

noncomputable def stopErrK (fin e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (q : Pos g σ Claims.Msg (Fin R)) :
    Option (SubmitJ J) → ℝ≥0∞
  | none => fin q
  | some s => γ q J s + Pos.resample e q.consume

noncomputable def roundsErrK (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) : ℕ → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | 0, p => e p
  | N + 1, p => stopErrK J γ e (roundsErrK e N) p.toStop (D.sub J p.toStop.modes)

noncomputable def sessionErrK (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) : ℝ≥0∞ :=
  γL p.toStop + Pos.resample (roundsErrK D J γ e N) p.toStop.consume

noncomputable def memberErrK (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) : Bool → ℝ≥0∞
  | true => sessionErrK D J γ γL (fun q => Pos.resample e q.toStop.consume) N p
  | false => e p

variable (M : CoinMap g σ Ω R mPts) (B : RoundHold)

theorem roundsOffK_le_one {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (he : ∀ q, e q ≤ 1) :
    ∀ (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)), roundsOffK D M J B e N p ≤ 1
  | 0, p => he p
  | N + 1, p => by
    show stopOffK M J B e (roundsOffK D M J B e N) p.toStop (D.sub J p.toStop.modes) ≤ 1
    cases D.sub J p.toStop.modes with
    | none => exact he _
    | some s =>
      refine (avg_mono fun c => ?_).trans_eq (avg_const 1)
      split_ifs
      · exact zero_le_one
      · exact roundsOffK_le_one he N _

theorem sessionOffK_le_one (BL : LinkHold mPts) {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (he : ∀ q, e q ≤ 1) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : sessionOffK D M J B BL e N p ≤ 1 := by
  refine (avg_mono fun pts => ?_).trans_eq (avg_const 1)
  split_ifs
  · exact zero_le_one
  · exact roundsOffK_le_one D J M B he N _

theorem sessionOffK_toStop (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : sessionOffK D M J B BL e N p.toStop = sessionOffK D M J B BL e N p := by
  unfold sessionOffK
  rw [Pos.toStop_toStop]

theorem sessionErrK_toStop (γL e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    sessionErrK D J γ γL e N p.toStop = sessionErrK D J γ γL e N p := by
  unfold sessionErrK
  rw [Pos.toStop_toStop]

theorem memberOffK_le_one (BL : LinkHold mPts) {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (he : ∀ q, e q ≤ 1) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : ∀ b : Bool, memberOffK D M J B BL e N p b ≤ 1
  | true => sessionOffK_le_one D J M B BL (fun q => he q.toStop.consume) N p
  | false => he p

theorem memberOffK_toStop (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (he : ∀ q, e q.toStop = e q) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ b : Bool, memberOffK D M J B BL e N p.toStop b = memberOffK D M J B BL e N p b
  | true => sessionOffK_toStop D J M B BL (fun q => e q.toStop.consume) N p
  | false => he p

theorem memberErrK_toStop (γL e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (he : ∀ q, e q.toStop = e q) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ b : Bool, memberErrK D J γ γL e N p.toStop b = memberErrK D J γ γL e N p b
  | true => sessionErrK_toStop D J γ γL (fun q => Pos.resample e q.toStop.consume) N p
  | false => he p

variable {D J γ M B} in
theorem avg_roundsOffK_le (hR : M.RoundClose γ) (Q : CoinTape g σ → Prop)
    {eO eE : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (hO : ∀ q, eO q ≤ 1)
    (hK : ∀ q : Pos g σ Claims.Msg (Fin R), avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
      avg (fun z : CoinTape q.g' q.s' => eO (q.withTape z)) + avg fun z : CoinTape q.g' q.s' => eE (q.withTape z)) :
    ∀ (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)),
      avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z)) ≤
        avg (fun z : CoinTape p.g' p.s' => roundsOffK D M J B eO N (p.withTape z)) +
        avg fun z : CoinTape p.g' p.s' =>
          roundsErrK D J (fun q j s => γ q j s + prCoin (B q.g' j s)) eE N (p.withTape z)
  | 0, p => hK p
  | N + 1, p => by
    let γB : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞ := fun q j s => γ q j s + prCoin (B q.g' j s)
    let Ψ : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q =>
      stopOffK M J B eO (roundsOffK D M J B eO N) q (D.sub J q.modes)
    let E : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q =>
      stopErrK J γB eE (roundsErrK D J γB eE N) q (D.sub J q.modes)
    have hstop : ∀ (q : Pos g σ Claims.Msg (Fin R)) (o : Option (SubmitJ J)),
        avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
          avg (fun z : CoinTape q.g' q.s' => stopOffK M J B eO (roundsOffK D M J B eO N) (q.withTape z) o) +
          avg fun z : CoinTape q.g' q.s' => stopErrK J γB eE (roundsErrK D J γB eE N) (q.withTape z) o :=
      fun q o => by
      cases o with
      | none => exact hK q
      | some s =>
        let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => roundsOffK D M J B eO N (q.consume.withTape z)
        have hγ : avg G ≤ avg (fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 =>
              G (M.round (q.withTape z) J s c z)) + avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s :=
          calc avg G = avg fun _ : CoinTape q.g' q.s' => avg G := (avg_const _).symm
            _ ≤ avg fun z₀ : CoinTape q.g' q.s' => avg (fun c : Fin s.n → 𝔽 => avg fun z : CoinTape q.g' q.s' =>
                  G (M.round (q.withTape z) J s c z)) + γ (q.withTape z₀) J s :=
                avg_mono fun z₀ => hR (q.withTape z₀) J s G fun _ => roundsOffK_le_one D J M B hO N _
            _ = _ := by
                rw [avg_add, avg_const, avg_comm]
        have hB : (avg fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 => G (M.round (q.withTape z) J s c z)) ≤
            (avg fun z : CoinTape q.g' q.s' =>
              stopOffK M J B eO (roundsOffK D M J B eO N) (q.withTape z) (some s)) + prCoin (B q.g' J s) :=
          (avg_mono fun z : CoinTape q.g' q.s' => avg_le_off (B q.g' J s)
            (fun c : Fin s.n → 𝔽 => G (M.round (q.withTape z) J s c z))
              fun _ => roundsOffK_le_one D J M B hO N _).trans_eq (by rw [avg_add, avg_const]; rfl)
        have h2 : avg (fun z : CoinTape q.g' q.s' =>
              stopErrK J γB eE (roundsErrK D J γB eE N) (q.withTape z) (some s)) =
            (avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s) + prCoin (B q.g' J s) +
              avg fun z : CoinTape q.g' q.s' => roundsErrK D J γB eE N (q.consume.withTape z) := by
          show avg (fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s + prCoin (B q.g' J s) +
            Pos.resample (roundsErrK D J γB eE N) q.consume) = _
          rw [avg_add, avg_add, avg_const, avg_const]
          rfl
        calc avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z))
            ≤ avg G + avg fun z : CoinTape q.g' q.s' => roundsErrK D J γB eE N (q.consume.withTape z) :=
              avg_roundsOffK_le hR Q hO hK N q.consume
          _ ≤ ((avg fun z : CoinTape q.g' q.s' =>
                  stopOffK M J B eO (roundsOffK D M J B eO N) (q.withTape z) (some s)) +
                prCoin (B q.g' J s) + avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s) +
              avg fun z : CoinTape q.g' q.s' => roundsErrK D J γB eE N (q.consume.withTape z) :=
              add_le_add (hγ.trans (add_le_add hB le_rfl)) le_rfl
          _ = _ := by
              rw [h2]
              ring
    calc avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z))
        = avg fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z).toStop :=
          congrArg avg (funext fun z => (tapeInd_toStop Q _).symm)
      _ = avg fun z : CoinTape p.g' p.s' => Pos.resample (tapeInd Q) (p.withTape z).toStop := Pos.avg_toStop _ p
      _ ≤ avg fun z : CoinTape p.g' p.s' =>
            Pos.resample Ψ (p.withTape z).toStop + Pos.resample E (p.withTape z).toStop :=
          avg_mono fun z => hstop _ (D.sub J (p.withTape z).toStop.modes)
      _ = (avg fun z : CoinTape p.g' p.s' => Pos.resample Ψ (p.withTape z).toStop) +
            avg fun z : CoinTape p.g' p.s' => Pos.resample E (p.withTape z).toStop := avg_add _ _
      _ = (avg fun z : CoinTape p.g' p.s' => Ψ (p.withTape z).toStop) +
            avg fun z : CoinTape p.g' p.s' => E (p.withTape z).toStop := by
          rw [Pos.avg_toStop Ψ p, Pos.avg_toStop E p]
      _ = _ := rfl

open Classical in
variable {D J γ M B} in
theorem avg_sessionOffK_le {γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (hL : M.LinkClose γL) (hR : M.RoundClose γ)
    (BL : LinkHold mPts) (Q : CoinTape g σ → Prop) {eO eE : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (hO : ∀ q, eO q ≤ 1)
    (hK : ∀ q : Pos g σ Claims.Msg (Fin R), avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
      avg (fun z : CoinTape q.g' q.s' => eO (q.withTape z)) + avg fun z : CoinTape q.g' q.s' => eE (q.withTape z))
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z)) ≤
      avg (fun z : CoinTape p.g' p.s' => sessionOffK D M J B BL eO N (p.withTape z)) +
      avg fun z : CoinTape p.g' p.s' =>
        sessionErrK D J (fun q j s => γ q j s + prCoin (B q.g' j s)) (fun q => γL q + prCoin (BL q.g')) eE N
          (p.withTape z) := by
  let γB : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞ := fun q j s => γ q j s + prCoin (B q.g' j s)
  let Ψ : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
    if BL q.g' pts then 0 else roundsOffK D M J B eO N (q.rewrite (M.link q pts)).consume
  let E : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q =>
    γL q + prCoin (BL q.g') + Pos.resample (roundsErrK D J γB eE N) q.consume
  have hstop : ∀ q : Pos g σ Claims.Msg (Fin R),
      Pos.resample (tapeInd Q) q ≤ Pos.resample Ψ q + Pos.resample E q := fun q => by
    let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => roundsOffK D M J B eO N (q.consume.withTape z)
    have hγ : avg G ≤ avg (fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
          G (M.link (q.withTape z) pts z)) + avg fun z : CoinTape q.g' q.s' => γL (q.withTape z) :=
      calc avg G = avg fun _ : CoinTape q.g' q.s' => avg G := (avg_const _).symm
        _ ≤ avg fun z₀ : CoinTape q.g' q.s' => avg (fun pts : Fin 2 → Fin mPts → 𝔽 =>
              avg fun z : CoinTape q.g' q.s' => G (M.link (q.withTape z) pts z)) + γL (q.withTape z₀) :=
            avg_mono fun z₀ => hL (q.withTape z₀) G fun _ => roundsOffK_le_one D J M B hO N _
        _ = _ := by rw [avg_add, avg_const, avg_comm]
    have hB : (avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
          G (M.link (q.withTape z) pts z)) ≤
        (avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
          if BL q.g' pts then 0 else G (M.link (q.withTape z) pts z)) + prCoin (BL q.g') :=
      (avg_mono fun z : CoinTape q.g' q.s' => avg_le_off (BL q.g')
        (fun pts : Fin 2 → Fin mPts → 𝔽 => G (M.link (q.withTape z) pts z))
          fun _ => roundsOffK_le_one D J M B hO N _).trans_eq (by rw [avg_add, avg_const])
    calc Pos.resample (tapeInd Q) q
        ≤ avg G + Pos.resample (roundsErrK D J γB eE N) q.consume := avg_roundsOffK_le hR Q hO hK N q.consume
      _ ≤ ((avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
              if BL q.g' pts then 0 else G (M.link (q.withTape z) pts z)) + prCoin (BL q.g') +
            avg fun z : CoinTape q.g' q.s' => γL (q.withTape z)) +
          Pos.resample (roundsErrK D J γB eE N) q.consume :=
          add_le_add (hγ.trans (add_le_add hB le_rfl)) le_rfl
      _ = _ := by
          show _ = (avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
                if BL q.g' pts then 0 else G (M.link (q.withTape z) pts z)) +
              avg fun z : CoinTape q.g' q.s' =>
                γL (q.withTape z) + prCoin (BL q.g') + Pos.resample (roundsErrK D J γB eE N) q.consume
          rw [avg_add, avg_add, avg_const, avg_const]
          ring
  calc avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z))
      = avg fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z).toStop :=
        congrArg avg (funext fun z => (tapeInd_toStop Q _).symm)
    _ = avg fun z : CoinTape p.g' p.s' => Pos.resample (tapeInd Q) (p.withTape z).toStop := Pos.avg_toStop _ p
    _ ≤ avg fun z : CoinTape p.g' p.s' =>
          Pos.resample Ψ (p.withTape z).toStop + Pos.resample E (p.withTape z).toStop :=
        avg_mono fun z => hstop _
    _ = (avg fun z : CoinTape p.g' p.s' => Pos.resample Ψ (p.withTape z).toStop) +
          avg fun z : CoinTape p.g' p.s' => Pos.resample E (p.withTape z).toStop := avg_add _ _
    _ = (avg fun z : CoinTape p.g' p.s' => Ψ (p.withTape z).toStop) +
          avg fun z : CoinTape p.g' p.s' => E (p.withTape z).toStop := by
        rw [Pos.avg_toStop Ψ p, Pos.avg_toStop E p]
    _ = _ := rfl

open Classical in
theorem avg_tapeInd_consume (Q : CoinTape g σ → Prop) {eO eE : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞}
    (hK : ∀ q : Pos g σ Claims.Msg (Fin R), avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
      avg (fun z : CoinTape q.g' q.s' => eO (q.withTape z)) + avg fun z : CoinTape q.g' q.s' => eE (q.withTape z))
    (q : Pos g σ Claims.Msg (Fin R)) :
    avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
      avg (fun z : CoinTape q.g' q.s' => eO (q.withTape z).toStop.consume) +
      avg fun z : CoinTape q.g' q.s' => Pos.resample eE (q.withTape z).toStop.consume :=
  calc avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z))
      = avg fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z).toStop :=
        congrArg avg (funext fun z => (tapeInd_toStop Q _).symm)
    _ = avg fun z : CoinTape q.g' q.s' => Pos.resample (tapeInd Q) (q.withTape z).toStop := Pos.avg_toStop _ q
    _ ≤ avg fun z : CoinTape q.g' q.s' =>
          Pos.resample eO (q.withTape z).toStop.consume + Pos.resample eE (q.withTape z).toStop.consume :=
        avg_mono fun z => hK (q.withTape z).toStop.consume
    _ = (avg fun z : CoinTape q.g' q.s' => Pos.resample eO (q.withTape z).toStop.consume) +
          avg fun z : CoinTape q.g' q.s' => Pos.resample eE (q.withTape z).toStop.consume := avg_add _ _
    _ = _ :=
        congrArg (fun a => a + avg fun z : CoinTape q.g' q.s' => Pos.resample eE (q.withTape z).toStop.consume)
          (Pos.avg_toStop (fun r => eO r.consume) q).symm

open Classical in
variable {D J γ M B} in
theorem avg_memberOffK_le {γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (hL : M.LinkClose γL) (hR : M.RoundClose γ)
    (BL : LinkHold mPts) (Q : CoinTape g σ → Prop) {eO eE : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (hO : ∀ q, eO q ≤ 1)
    (hK : ∀ q : Pos g σ Claims.Msg (Fin R), avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
      avg (fun z : CoinTape q.g' q.s' => eO (q.withTape z)) + avg fun z : CoinTape q.g' q.s' => eE (q.withTape z))
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ b : Bool, avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z)) ≤
      avg (fun z : CoinTape p.g' p.s' => memberOffK D M J B BL eO N (p.withTape z) b) +
      avg fun z : CoinTape p.g' p.s' =>
        memberErrK D J (fun q j s => γ q j s + prCoin (B q.g' j s)) (fun q => γL q + prCoin (BL q.g')) eE N
          (p.withTape z) b
  | true => avg_sessionOffK_le hL hR BL Q (eO := fun q => eO q.toStop.consume)
      (eE := fun q => Pos.resample eE q.toStop.consume) (fun q => hO q.toStop.consume) (avg_tapeInd_consume Q hK) N p
  | false => hK p

end SessionErr

section SeqErr

variable {g : Game Exchange} {σ : Strategy g} {Reg : Type} {R nJ : ℕ} (D : ℕ → PosDecoders g σ R Reg)
  (γ : ℕ → Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
  (γL : ℕ → (mPts : ℕ) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (S : Finset (Fin nJ)) (Rg : Reg)

noncomputable def seqErr : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    ℝ≥0∞
  | _, [], _ => 0
  | i, m :: l, p =>
    memberErrK (D i) (memberTabs S Rg m) (γ i) (γL i (memberPts m)) (fun q => seqErr (i + 1) l q) m.N p
      (memberSampled S m)

theorem seqErr_toStop : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)),
    seqErr D γ γL S Rg i l p.toStop = seqErr D γ γL S Rg i l p
  | [], _, _ => rfl
  | m :: l, i, p =>
    memberErrK_toStop (D i) (memberTabs S Rg m) (γ i) (γL i (memberPts m))
      (fun q => seqErr D γ γL S Rg (i + 1) l q) (fun q => seqErr_toStop l (i + 1) q) m.N p (memberSampled S m)

variable {Ω : Type} (M : ExecutionPhaseCoinMap g σ Ω R) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold)

theorem seqOff_le_one (Q : CoinTape g σ → Prop) : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)), seqOff D M BL B S Rg Q i l p ≤ 1
  | [], _, p => tapeInd_le_one Q p
  | m :: l, i, p =>
    memberOffK_le_one (D i) (memberTabs S Rg m) (M.member i (memberPts m)) (B i) (BL i (memberPts m))
      (fun q => seqOff_le_one Q l (i + 1) q) m.N p (memberSampled S m)

theorem seqOff_toStop (Q : CoinTape g σ → Prop) : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)), seqOff D M BL B S Rg Q i l p.toStop = seqOff D M BL B S Rg Q i l p
  | [], _, p => tapeInd_toStop Q p
  | m :: l, i, p =>
    memberOffK_toStop (D i) (memberTabs S Rg m) (M.member i (memberPts m)) (B i) (BL i (memberPts m))
      (fun q => seqOff D M BL B S Rg Q (i + 1) l q) (fun q => seqOff_toStop Q l (i + 1) q) m.N p (memberSampled S m)

open Classical in
variable {D γ γL M} in
theorem avg_seqOff_le (hR : ∀ i mPts, (M.member i mPts).RoundClose (γ i))
    (hL : ∀ i mPts, (M.member i mPts).LinkClose (γL i mPts)) (Q : CoinTape g σ → Prop) :
    ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)),
      avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z)) ≤
        avg (fun z : CoinTape p.g' p.s' => seqOff D M BL B S Rg Q i l (p.withTape z)) +
        avg fun z : CoinTape p.g' p.s' =>
          seqErr D (fun i q j s => γ i q j s + prCoin (B i q.g' j s))
            (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g')) S Rg i l (p.withTape z)
  | [], _, _ => le_self_add
  | m :: l, i, p =>
    avg_memberOffK_le (D := D i) (J := memberTabs S Rg m) (B := B i) (hL i (memberPts m)) (hR i (memberPts m))
      (BL i (memberPts m)) Q (fun q => seqOff_le_one D S Rg M BL B Q l (i + 1) q)
      (fun q => avg_seqOff_le hR hL Q l (i + 1) q) m.N p (memberSampled S m)

open Classical in
variable {D γ γL M} in
theorem seqOff_couples (Ks : Fin R → DigestCut Claims.Msg g) {C : Type} [Fintype C] (draw : C → Finset (Fin nJ))
    (start : C → CoinTape g σ) {ε₀ : ℝ≥0∞} (hR : ∀ i mPts, (M.member i mPts).RoundClose (γ i))
    (hL : ∀ i mPts, (M.member i mPts).LinkClose (γL i mPts))
    (hF : ∀ G : Finset (Fin nJ) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞,
      avg (fun ω : C => G (draw ω) (Pos.root Ks (start ω)).toStop) =
        avg fun ω : C => Pos.resample (G (draw ω)) (Pos.root Ks (start ω)).toStop)
    (hS : ∀ H : CoinTape g σ → ℝ≥0∞, (∀ x, H x ≤ 1) → avg H ≤ avg (fun ω : C => H (start ω)) + ε₀)
    (Q : CoinTape g σ → Prop) (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ) :
    prCoin Q ≤ avg (fun ω : C => seqOff D M BL B (draw ω) Rg Q i l (Pos.root Ks (start ω))) +
      (ε₀ + avg fun ω : C => seqErr D (fun i q j s => γ i q j s + prCoin (B i q.g' j s))
        (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g')) (draw ω) Rg i l (Pos.root Ks (start ω))) := by
  let GO : Finset (Fin nJ) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun s q => seqOff D M BL B s Rg Q i l q
  let GE : Finset (Fin nJ) → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun s q =>
    seqErr D (fun i q j s => γ i q j s + prCoin (B i q.g' j s))
      (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g')) s Rg i l q
  have hU : prCoin Q = avg fun x : CoinTape g σ => Pos.resample (tapeInd Q) (Pos.root Ks x).toStop := by
    refine Eq.trans ?_ (Pos.avg_walk (tapeInd Q) g σ Ks id)
    refine congrArg avg (funext fun x => ?_)
    show _ = tapeInd Q (Pos.root Ks x).toStop
    rw [tapeInd, Pos.toStop_tape]
    rfl
  rw [hU]
  calc avg (fun x : CoinTape g σ => Pos.resample (tapeInd Q) (Pos.root Ks x).toStop)
      ≤ avg (fun ω : C => Pos.resample (tapeInd Q) (Pos.root Ks (start ω)).toStop) + ε₀ :=
        hS _ fun x => resample_tapeInd_le_one Q _
    _ ≤ avg (fun ω : C => Pos.resample (GO (draw ω)) (Pos.root Ks (start ω)).toStop +
          Pos.resample (GE (draw ω)) (Pos.root Ks (start ω)).toStop) + ε₀ :=
        add_le_add (avg_mono fun ω => avg_seqOff_le (draw ω) Rg BL B hR hL Q l i _) le_rfl
    _ = (avg fun ω : C => GO (draw ω) (Pos.root Ks (start ω)).toStop) +
          (avg fun ω : C => GE (draw ω) (Pos.root Ks (start ω)).toStop) + ε₀ := by
        rw [avg_add, ← hF GO, ← hF GE]
    _ = _ := by
        rw [show (fun ω : C => GO (draw ω) (Pos.root Ks (start ω)).toStop) =
            fun ω : C => seqOff D M BL B (draw ω) Rg Q i l (Pos.root Ks (start ω)) from
          funext fun ω => seqOff_toStop D (draw ω) Rg M BL B Q l i _,
          show (fun ω : C => GE (draw ω) (Pos.root Ks (start ω)).toStop) =
            fun ω : C => seqErr D (fun i q j s => γ i q j s + prCoin (B i q.g' j s))
              (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g')) (draw ω) Rg i l (Pos.root Ks (start ω)) from
          funext fun ω => seqErr_toStop D _ _ (draw ω) Rg l i _]
        ring

end SeqErr

variable (e : 𝔽 ≃ (Fin 128 → Bool))

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

def ExecutionPhaseCouplesWithin (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold)
    (B : ExecutionPhaseRoundHold) (ε : ℝ≥0∞) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    executionPhaseOff V θ F c W t rd σ D M BL B Q + ε

theorem executionPhaseCouples_of_within {W : ExecutionPhase} {t : ℕ} {rd : Reading W.rounds} {σ : Strategy (V.game θ F c)}
    {D : ExecutionPhaseDecoders V θ F c W σ} {M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds} {BL : ExecutionPhaseLinkHold}
    {B : ExecutionPhaseRoundHold} {ε : ℝ≥0∞} (h : ExecutionPhaseCouplesWithin V θ F c W t rd σ D M BL B ε)
    (hε : ε ≤ W.hidingTerm) : ExecutionPhaseCouples V θ F c W t rd σ D M BL B := fun Q =>
  (h Q).trans (add_le_add le_rfl hε)

def ExecutionPhaseStartWithin (W : ExecutionPhase) (σ : Strategy (V.game θ F c))
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (ε₀ : ℝ≥0∞) : Prop :=
  ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞, (∀ x, H x ≤ 1) →
    avg H ≤ avg (fun ω : (W.extend M.widening).L.Ω => H (M.start ω)) + ε₀

noncomputable def executionPhaseFirstStop (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (ω : (W.extend M.widening).L.Ω) :
    Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) :=
  (Pos.root (roundCuts V θ F c rd) (M.start ω)).toStop

def ExecutionPhaseFirstStopFresh (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) : Prop :=
  ∀ G : Finset (Fin W.nJ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞,
    avg (fun ω : (W.extend M.widening).L.Ω =>
        G ((W.extend M.widening).L.draw ω) (executionPhaseFirstStop V θ F c W rd σ M ω)) =
      avg fun ω : (W.extend M.widening).L.Ω =>
        Pos.resample (G ((W.extend M.widening).L.draw ω)) (executionPhaseFirstStop V θ F c W rd σ M ω)

noncomputable def executionPhaseErr (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞) : ℝ≥0∞ :=
  avg fun ω : (W.extend M.widening).L.Ω =>
    seqErr D γ γL ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω))

open Classical in
theorem executionPhaseCouples_of_close (W : ExecutionPhase) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ExecutionPhaseDecoders V θ F c W σ) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞} {ε₀ : ℝ≥0∞}
    (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold) (hR : ∀ i mPts, (M.member i mPts).RoundClose (γ i))
    (hL : ∀ i mPts, (M.member i mPts).LinkClose (γL i mPts)) (hF : ExecutionPhaseFirstStopFresh V θ F c W rd σ M)
    (hS : ExecutionPhaseStartWithin V θ F c W σ M ε₀) :
    ExecutionPhaseCouplesWithin V θ F c W t rd σ D M BL B (ε₀ + executionPhaseErr V θ F c W t rd σ D M
      (fun i q j s => γ i q j s + prCoin (B i q.g' j s)) (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g'))) :=
  fun Q => by
  rw [ofReal_pr_eq_prCoin]
  exact seqOff_couples (executionPhaseReg V θ F c W rd σ D) BL B (roundCuts V θ F c rd)
    (C := (W.extend M.widening).L.Ω) (fun ω => (W.extend M.widening).L.draw ω) (fun ω => M.start ω) hR hL hF hS Q
    ((W.extend M.widening).joint t).ms 0

noncomputable def executionPhaseOpenMap {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ) :
    ExecutionPhaseCoinMap g σ Ω R where
  T := T
  start := start
  link _ p _ pts := writeOpening E p.s' (linkBits e pts)
  round _ p _ _ c := writeOpening E p.s' (roundBits e c)

theorem executionPhaseOpenMap_roundClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ)
    (i mPts : ℕ) :
    ((executionPhaseOpenMap e (R := R) E T start).member i mPts).RoundClose fun q _ s =>
      openGap E q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) := fun q _ _ G hG =>
  avg_le_writeOpening E (roundBits e) q.s' G hG

theorem executionPhaseOpenMap_linkClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ)
    (i mPts : ℕ) :
    ((executionPhaseOpenMap e (R := R) E T start).member i mPts).LinkClose fun q =>
      openGap E q.g' (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool) := fun q G hG =>
  avg_le_writeOpening E (linkBits e) q.s' G hG

noncomputable def executionPhaseSampleMap (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool) :
    ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds :=
  executionPhaseOpenMap e (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ)
    (sampleStart V θ F c rd σ label after κ)

theorem executionPhaseFirstStop_eq (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (ω : (W.extend M.widening).L.Ω) :
    executionPhaseFirstStop V θ F c W rd σ M ω = Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd) (M.start ω) id :=
  rfl

theorem executionPhaseSampleMap_start_eq (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool)
    (ω : (W.extend (executionPhaseSampleMap e V θ F c W rd σ label after κ).widening).L.Ω) :
    (executionPhaseSampleMap e V θ F c W rd σ label after κ).start ω =
      fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2 :=
  rfl

theorem executionPhaseFirstStop_sampleMap (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool)
    (ω : (W.extend (executionPhaseSampleMap e V θ F c W rd σ label after κ).widening).L.Ω) :
    executionPhaseFirstStop V θ F c W rd σ (executionPhaseSampleMap e V θ F c W rd σ label after κ) ω =
      Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd)
        (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2) id :=
  (executionPhaseFirstStop_eq V θ F c W rd σ (executionPhaseSampleMap e V θ F c W rd σ label after κ) ω).trans
    (congrArg (fun t => Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd) t id)
      (executionPhaseSampleMap_start_eq e V θ F c W rd σ label after κ ω))

theorem executionPhaseSampleMap_fresh (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool) :
    ExecutionPhaseFirstStopFresh V θ F c W rd σ (executionPhaseSampleMap e V θ F c W rd σ label after κ) := fun G => by
  have h₁ := congrArg avg
    (funext fun ω : (W.extend (executionPhaseSampleMap e V θ F c W rd σ label after κ).widening).L.Ω =>
      congrArg₂ G (W.sample_extend _ ω) (executionPhaseFirstStop_sampleMap e V θ F c W rd σ label after κ ω))
  have h₂ := congrArg avg
    (funext fun ω : (W.extend (executionPhaseSampleMap e V θ F c W rd σ label after κ).widening).L.Ω =>
      congrArg₂ (fun s q => Pos.resample (G s) q) (W.sample_extend _ ω)
        (executionPhaseFirstStop_sampleMap e V θ F c W rd σ label after κ ω))
  refine h₁.trans (Eq.trans ?_ h₂.symm)
  generalize W.L.draw = dr
  exact avg_prod_walk_fill (C := W.Ω) (sampleSel V θ F c σ label after) (fun ω₁ : W.Ω => G (dr ω₁))
    (roundCuts V θ F c rd) m κ

theorem executionPhaseSampleMap_start (W : ExecutionPhase) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    ExecutionPhaseStartWithin V θ F c W σ (executionPhaseSampleMap e V θ F c W rd σ label after κ) 0 := fun H _ => by
  rw [add_zero]
  refine le_of_eq (Eq.trans ?_ (avg_prod (C := W.Ω) (D := CoinTape (V.game θ F c) σ) _).symm)
  exact (avg_fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m H).symm.trans
    (hκ fun κ' => avg fun t =>
      H (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m κ' t)).symm

theorem ExecutionPhase.avg_byteCoins (W : ExecutionPhase) {m : ℕ} (hL : W.Lb 0 = m * 64) (H : (Fin m → Fin 512 → Bool) → ℝ≥0∞) :
    avg (fun ω : W.Ω => H (byteCoins W.src hL ω)) = avg H :=
  _root_.Verity.Sampled.avg_byteCoins W.hA3 hL H

omit θ F c in
noncomputable def executionPhaseSampleMaps (rd : ExecutionPhaseReadings)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {m : ℕ}
    (κ : (W : ExecutionPhase) → W.Ω → Fin m → Fin 512 → Bool) (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds :=
  executionPhaseSampleMap e V θ F c W (rd W θ F c) σ (label θ F c σ) (after θ F c σ) (κ W)

variable (k : Verity.Registration.Composite)

omit θ F c in
theorem executionPhaseCouplingTarget_of_close (rd : ExecutionPhaseReadings) (K : ExecutionPhaseConstructions V)
    (D : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseDecoders V θ F c W σ)
    (M : (W : ExecutionPhase) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (W : ExecutionPhase) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseLinkHold)
    (B : (c : ByteArray) → Strategy (V.game θ F c) → ExecutionPhaseRoundHold)
    (γ : (c : ByteArray) → (σ : Strategy (V.game θ F c)) → ℕ →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : (c : ByteArray) → (σ : Strategy (V.game θ F c)) → ℕ → (mPts : ℕ) →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞)
    (ε₀ : (c : ByteArray) → Strategy (V.game θ F c) → ℝ≥0∞)
    (ha : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hR : ∀ c σ i mPts, ((M W θ F c σ).member i mPts).RoundClose (γ c σ i))
    (hL : ∀ c σ i mPts, ((M W θ F c σ).member i mPts).LinkClose (γL c σ i mPts))
    (hF : ∀ c σ, ExecutionPhaseFirstStopFresh V θ F c W (rd W θ F c) σ (M W θ F c σ))
    (hS : ∀ c σ, ExecutionPhaseStartWithin V θ F c W σ (M W θ F c σ) (ε₀ c σ))
    (hε : ∀ c σ, ε₀ c σ + executionPhaseErr V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ)
      (fun i q j s => γ c σ i q j s + prCoin (B c σ i q.g' j s))
      (fun i mPts q => γL c σ i mPts q + prCoin (BL c σ i mPts q.g')) ≤ W.hidingTerm)
    (hD : ∀ c σ, ExecutionPhaseDecodes V θ F c k (fun o => P o ∧ Q c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
      (M W θ F c σ) (BL c σ) (B c σ)) :
    ExecutionPhaseCouplingTarget V k (K.withProject V rd D M) W θ F :=
  executionPhaseCouplingTarget_of_logged V k rd K D M W P θ F Q BL B ha hq
    (fun c σ => executionPhaseCouples_of_within V θ F c
      (executionPhaseCouples_of_close V θ F c W _ _ σ _ _ (BL c σ) (B c σ) (hR c σ) (hL c σ) (hF c σ) (hS c σ)) (hε c σ))
    hD

end Verity.Sampled
