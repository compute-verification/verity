module

public import Proofs.Properties.Verity.Sampled.Window.Walk
public import Proofs.Properties.Verity.Sampled.CoinMap
public import Proofs.Properties.Verity.Sampled.Members.Widen

/-!
# A window's one sample, its members' parts, and the coin map over the window

The period's one sample covers a window's population of all its members' proof blocks; each member's session proves
the member's part of it (`Member.part`), and the window's projection hands each member's session that part
(`seqP`). What the leg reads of the one sample and of the coin map over the members' sessions:

* **The members' parts** (`Member.mem_part`, `Window.Partitions`, `Window.sum_part_card`): a member's part of a
  sample is its proof blocks the sample holds, and where the members' places partition the population the parts'
  sizes add up to the sample's.
* **The registered layout** (`Window.Laid`, `Window.Laid.partitions`): the members in unit order, member `k`'s proof
  block `v` at the population's place `base k + v`, its base the sum of the counts of the members before it
  (`JointAuditV5.Laid`), both fixed by the registration before the sample. A sampled unit `u` is then the proof block
  of the one member with `base ≤ u < base + n` (`JointAuditV5.Laid.mem_iff`, `Laid.member_unique`), so the members'
  places partition the population.
* **The marginals** (`Window.avg_design`, `Window.drawMiss_extend`, `Window.joint_extend`, `Window.eps_extend`,
  `Window.sessionDelta_extend`): member `k`'s sampling design in Flock's window is its part of the window's one
  sample, a mean of a reading of the part over the sample's coins; the members' knowledge and binding terms read only
  that, so the window's error is the same at the window widened by the coins a projection reads.
* **The coin map's premise within a term** (`stopErrK`, `roundsErrK`, `sessionErrK`, `seqErr`, `windowErr`,
  `WindowStartWithin`, `WindowFirstStopFresh`, `WindowCouplesWithin`, `windowCouples_of_close`): `CoinMap`'s
  (`couples_of_close`), over the members' sessions in order. Wherever each member's coin map writes its link points
  in within `γL` of the tape's own and each round's coins within `γ`, the start leaves the tape from the first stop
  fresh and is within `ε₀` of the uniform tape, the joint law's tape off the held-off events is within `ε₀` and the
  stops' term, every member's session's summed along the walk, of the uniform tape at every event; so the coin map's
  premise holds wherever that is at most the hiding term (`windowCouples_of_within`). The induction is one session's
  with what follows it (`avg_roundsOffK_le`, `avg_sessionOffK_le`), then over the members (`avg_seqOff_le`,
  `seqOff_couples`); a session reads the position it starts from only at its next stop (`Pos.toStop_toStop`), which
  is how the start's freshness at the first stop reaches the first member's session.
* **The coin map into the openings, over the window** (`windowOpenMap`, `windowSampleMap`, `windowSampleMaps`):
  `openMap` and `sampleMap` with one widening and one start for the window, every member's link points and round
  coins written into the openings at its own stops. The sample-coupled start leaves the tape from the first stop
  fresh (`windowSampleMap_fresh`) and is the uniform tape wherever the sample's coins are uniform
  (`windowSampleMap_start`), and each member's map writes its coins in within the stop's opening's gap
  (`windowOpenMap_roundClose`, `windowOpenMap_linkClose`), so the coupling at the window follows from the decoders'
  premise wherever the stops' term is at most the hiding term (`windowCouplingTarget_of_close`).
-/

@[expose] public section

namespace Flock.Guarantees.JointAuditV5.Laid

variable {w : JointAuditV5}

/-- Under the layout each member's proof blocks sit at distinct places. -/
theorem inj_of_mem (h : w.Laid) : ∀ m ∈ w.ms, Function.Injective m.emb := fun m hm => by
  obtain ⟨k, rfl⟩ := List.mem_iff_get.1 hm
  exact h.emb_injective k

/-- Under the layout no two members share a place. -/
theorem pairwise (h : w.Laid) : w.ms.Pairwise fun m m' => ∀ u u', m.emb u ≠ m'.emb u' :=
  List.pairwise_iff_get.2 fun _ _ hij _ _ e => absurd (h.member_eq e) (ne_of_lt hij)

/-- Under the layout every place is some member's. -/
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

/-! ## The members' parts of the one sample -/

/-- **A member's part of a sample is its proof blocks the sample holds.** -/
theorem Member.mem_part {nJ : ℕ} (m : Member nJ) (S : Finset (Fin nJ)) (u : Fin m.n) :
    u ∈ m.part S ↔ m.emb u ∈ S := by
  have h : ∀ (m' : Flock.Guarantees.MemberV5 ByteArray nJ) (u' : Fin m'.n), u' ∈ m'.part S ↔ m'.emb u' ∈ S :=
    fun m' u' => by simp [Flock.Guarantees.MemberV5.part]
  exact h (m.toV5 0) u

/-- **The members' places partition the window's population**: each member's proof blocks sit at distinct places,
no two members share a place, and every place is some member's. -/
structure Window.Partitions (W : Window) : Prop where
  inj : ∀ m ∈ W.ms, Function.Injective m.emb
  disjoint : W.ms.Pairwise fun m m' => ∀ u u', m.emb u ≠ m'.emb u'
  cover : ∀ v : Fin W.nJ, ∃ m ∈ W.ms, ∃ u, m.emb u = v

open Classical in
/-- **A member's part of a sample has the size of the sample's places that are the member's**, where the member's
proof blocks sit at distinct places. -/
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
/-- **Members' parts of a sample have, between them, the size of the sample's places that are theirs**, where each
member's proof blocks sit at distinct places and no two members share a place. -/
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

/-- **The members' parts of a sample have its size between them**, where the members' places partition the
population. -/
theorem Window.sum_part_card (W : Window) (h : W.Partitions) (S : Finset (Fin W.nJ)) :
    (W.ms.map fun m => (m.part S).card).sum = S.card := by
  classical
  rw [sum_part_card_filter S W.ms h.inj h.disjoint]
  exact congrArg Finset.card (Finset.filter_true_of_mem fun v _ => h.cover v)

/-! ## The registered layout -/

/-- **The window's members are laid out in unit order** (`JointAuditV5.Laid`): their proof-block counts sum to the
population, and member `k`'s proof block `v` sits at the population's place `base k + v`, its base the sum of the
counts of the members before it. The counts, the bases and the places are the registration's, fixed before the
sample; a sampled unit `u` is the proof block of the member with `base ≤ u < base + n`
(`JointAuditV5.Laid.mem_iff`). -/
def Window.Laid (W : Window) : Prop := (W.joint 0).Laid

/-- **Members laid out in unit order partition the window's population.** -/
theorem Window.Laid.partitions {W : Window} (h : W.Laid) : W.Partitions where
  inj m hm := Flock.Guarantees.JointAuditV5.Laid.inj_of_mem h (m.toV5 0) (List.mem_map_of_mem hm)
  disjoint := List.pairwise_map.1 (Flock.Guarantees.JointAuditV5.Laid.pairwise h)
  cover v := by
    obtain ⟨m', hm', u, hu⟩ := Flock.Guarantees.JointAuditV5.Laid.cover h v
    obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hm'
    exact ⟨m, hm, u, hu⟩

/-- **The members' parts of a sample have its size between them**, where the members are laid out in unit order. -/
theorem Window.Laid.sum_part_card {W : Window} (h : W.Laid) (S : Finset (Fin W.nJ)) :
    (W.ms.map fun m => (m.part S).card).sum = S.card :=
  W.sum_part_card h.partitions S

/-! ## The marginals -/

/-- **Member `k`'s sampling design is its part of the window's one sample**: a mean of a reading of the member's
sample over its design's coins is the mean, over the window's sample's coins, of the reading at the member's part. -/
theorem Window.avg_design (W : Window) (t : ℕ) (k : Fin (W.joint t).ms.length)
    (G : Finset (Fin ((W.joint t).ms.get k).n) → ℝ≥0∞) :
    avg (fun ω : ((W.joint t).design k).Ω => G (((W.joint t).design k).draw ω)) =
      avg fun ω : W.L.Ω => G (((W.joint t).ms.get k).part (W.L.draw ω)) :=
  rfl

/-- The sample's term doesn't read the widening. -/
theorem Window.drawMiss_extend (W : Window) (S : Widening) : (W.extend S).drawMiss = W.drawMiss :=
  rfl

/-- **Flock's window at the widened window is Flock's window, widened** (`JointAuditV5.extend`). -/
theorem Window.joint_extend (W : Window) (S : Widening) (t : ℕ) : (W.extend S).joint t = (W.joint t).extend S :=
  rfl

/-- The sample at widened coins is the sample at the device's coins. By `dsimp`, not `rfl`: elaborating the two
samples as definitionally equal reduces their `match` and evaluates `drawUnits`. -/
theorem Window.sample_extend (W : Window) (S : Widening) (ω : (W.extend S).L.Ω) :
    (W.extend S).L.draw ω = W.L.draw ω.1 := by
  dsimp only [Window.L, Flock.Bridge.execSel, Window.extend]

/-- **The members' terms read only their parts of the one sample**: the members' knowledge and binding terms summed
are the same at the window widened by the coins `S`. -/
theorem Window.eps_extend (W : Window) (S : Widening) (t : ℕ) : ((W.extend S).joint t).eps = (W.joint t).eps :=
  (congrArg Flock.Guarantees.JointAuditV5.eps (W.joint_extend S t)).trans ((W.joint t).eps_extend S)

/-- **The window's error doesn't read the widening** (`Live.sessionDelta_extend`). -/
theorem Window.sessionDelta_extend (W : Window) (S : Widening) (t : ℕ) :
    (W.extend S).sessionDelta t = W.sessionDelta t := by
  unfold Window.sessionDelta
  rw [W.drawMiss_extend S, W.eps_extend S t, W.rootsErr_extend S]

/-! ## The coin map's premise within a term, over the members' sessions -/

section SessionErr

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : PosDecoders g σ R Reg) (J : ℕ)
  (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞)

/-- **One stop's error, then what follows the session** (`stopErr`): `fin` at the session's end there; `γ` there and
the next rounds' error over the tapes from the stop where a submission is decoded. -/
noncomputable def stopErrK (fin e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (q : Pos g σ Claims.Msg (Fin R)) :
    Option (SubmitJ J) → ℝ≥0∞
  | none => fin q
  | some s => γ q J s + Pos.resample e q.consume

/-- **The rounds' error, then what follows the session** (`roundsErr`) from the position `p`: `γ` at each stop a
uniform tape from `p` reaches, and `e` at the position the session's walk ends at. -/
noncomputable def roundsErrK (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) : ℕ → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | 0, p => e p
  | N + 1, p => stopErrK J γ e (roundsErrK e N) p.toStop (D.sub J p.toStop.modes)

/-- **The session's error, then what follows it**, from the position `p`: `γL` at the session's first stop and the
rounds' error over the tapes from it. -/
noncomputable def sessionErrK (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) : ℝ≥0∞ :=
  γL p.toStop + Pos.resample (roundsErrK D J γ e N) p.toStop.consume

/-- **A member's error, then what follows it**: its session's (`sessionErrK`) where the sample holds one of its proof
blocks (`true`), then what follows over the tapes from the session's closing round's stop, past it (a closing round
releases no coin, so its stop adds no error of its own); else what follows, from the same position. -/
noncomputable def memberErrK (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) : Bool → ℝ≥0∞
  | true => sessionErrK D J γ γL (fun q => Pos.resample e q.toStop.consume) N p
  | false => e p

variable (M : CoinMap g σ Ω R mPts) (B : RoundHold)

/-- The rounds' share off `B`, then what follows the session, is at most 1 wherever what follows is. -/
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

/-- The session's share off `BL` and `B`, then what follows it, is at most 1 wherever what follows is. -/
theorem sessionOffK_le_one (BL : LinkHold mPts) {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (he : ∀ q, e q ≤ 1) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : sessionOffK D M J B BL e N p ≤ 1 := by
  refine (avg_mono fun pts => ?_).trans_eq (avg_const 1)
  split_ifs
  · exact zero_le_one
  · exact roundsOffK_le_one D J M B he N _

/-- The session's share off `BL` and `B` reads the position only at its next stop. -/
theorem sessionOffK_toStop (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : sessionOffK D M J B BL e N p.toStop = sessionOffK D M J B BL e N p := by
  unfold sessionOffK
  rw [Pos.toStop_toStop]

/-- The session's error reads the position only at its next stop. -/
theorem sessionErrK_toStop (γL e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    sessionErrK D J γ γL e N p.toStop = sessionErrK D J γ γL e N p := by
  unfold sessionErrK
  rw [Pos.toStop_toStop]

/-- A member's share off `BL` and `B`, then what follows it, is at most 1 wherever what follows is. -/
theorem memberOffK_le_one (BL : LinkHold mPts) {e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (he : ∀ q, e q ≤ 1) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) : ∀ b : Bool, memberOffK D M J B BL e N p b ≤ 1
  | true => sessionOffK_le_one D J M B BL (fun q => he q.toStop.consume) N p
  | false => he p

/-- A member's share off `BL` and `B` reads the position only at its next stop, wherever what follows does. -/
theorem memberOffK_toStop (BL : LinkHold mPts) (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞)
    (he : ∀ q, e q.toStop = e q) (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ b : Bool, memberOffK D M J B BL e N p.toStop b = memberOffK D M J B BL e N p b
  | true => sessionOffK_toStop D J M B BL (fun q => e q.toStop.consume) N p
  | false => he p

/-- A member's error reads the position only at its next stop, wherever what follows does. -/
theorem memberErrK_toStop (γL e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (he : ∀ q, e q.toStop = e q) (N : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)) :
    ∀ b : Bool, memberErrK D J γ γL e N p.toStop b = memberErrK D J γ γL e N p b
  | true => sessionErrK_toStop D J γ γL (fun q => Pos.resample e q.toStop.consume) N p
  | false => he p

variable {D J γ M B} in
/-- **The rounds' share off `B`, then what follows the session, is within the rounds' error of uniform from where it
starts** (`avg_roundsOff_le`): wherever the coin map writes each round's coins in within `γ` of the tape's own, and
what follows the session has, over a uniform tape from each position, the event's share within its own error, an
event of the tape has, over a uniform tape from the position `p`, at most its share off `B` plus the rounds' error at
`γ` and `B`'s probability at each stop. -/
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
/-- **The session's share off `BL` and `B`, then what follows it, is within the session's error of uniform from where
it starts** (`resample_sessionOff_le`): wherever the coin map writes the link points in within `γL` of the tape's own
and each round's coins within `γ`, and what follows the session has the event's share within its own error, an event
of the tape has, over a uniform tape from the position `p`, at most its share off `BL` and `B` plus the session's
error: `γL` and `BL`'s probability at the first stop and the rounds' error after it. -/
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
/-- **Past a stop with nothing written in, what follows keeps its share within its error**: wherever what follows
has, over a uniform tape from each position, the event's share within its own error, it has from a position the
event's share within its error past the position's next stop, consumed as it is, the error taken over the tapes from
that stop. The tape from the next stop is uniform and independent of the walk to it (`Pos.avg_toStop`), and consuming
a stop changes no tape. -/
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
/-- **A member's share off `BL` and `B`, then what follows it, is within the member's error of uniform from where it
starts**: its session's (`avg_sessionOffK_le`) where the sample holds one of its proof blocks, with what follows
taken past the session's closing round's stop (`avg_tapeInd_consume`); else what follows has the event's share within
its own error, from the same position. -/
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

/-- **The sessions' error, in the members' order** from the position `p`: each member's session's, at its own `γ`
and `γL`, then the rest's from the position its walk ends at. A member with no proof block in the sample has no
session and adds nothing (`memberErrK`). -/
noncomputable def seqErr : (i : ℕ) → (l : List (Flock.Guarantees.MemberV5 Reg nJ)) → Pos g σ Claims.Msg (Fin R) →
    ℝ≥0∞
  | _, [], _ => 0
  | i, m :: l, p =>
    memberErrK (D i) (memberTabs S Rg m) (γ i) (γL i (memberPts m)) (fun q => seqErr (i + 1) l q) m.N p
      (memberDrawn S m)

/-- The sessions' error reads the position only at its next stop. -/
theorem seqErr_toStop : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)),
    seqErr D γ γL S Rg i l p.toStop = seqErr D γ γL S Rg i l p
  | [], _, _ => rfl
  | m :: l, i, p =>
    memberErrK_toStop (D i) (memberTabs S Rg m) (γ i) (γL i (memberPts m))
      (fun q => seqErr D γ γL S Rg (i + 1) l q) (fun q => seqErr_toStop l (i + 1) q) m.N p (memberDrawn S m)

variable {Ω : Type} (M : WindowCoinMap g σ Ω R) (BL : WindowLinkHold) (B : WindowRoundHold)

/-- The sessions' share off the held-off events is at most 1. -/
theorem seqOff_le_one (Q : CoinTape g σ → Prop) : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)), seqOff D M BL B S Rg Q i l p ≤ 1
  | [], _, p => tapeInd_le_one Q p
  | m :: l, i, p =>
    memberOffK_le_one (D i) (memberTabs S Rg m) (M.member i (memberPts m)) (B i) (BL i (memberPts m))
      (fun q => seqOff_le_one Q l (i + 1) q) m.N p (memberDrawn S m)

/-- The sessions' share off the held-off events reads the position only at its next stop. -/
theorem seqOff_toStop (Q : CoinTape g σ → Prop) : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ)
    (p : Pos g σ Claims.Msg (Fin R)), seqOff D M BL B S Rg Q i l p.toStop = seqOff D M BL B S Rg Q i l p
  | [], _, p => tapeInd_toStop Q p
  | m :: l, i, p =>
    memberOffK_toStop (D i) (memberTabs S Rg m) (M.member i (memberPts m)) (B i) (BL i (memberPts m))
      (fun q => seqOff D M BL B S Rg Q (i + 1) l q) (fun q => seqOff_toStop Q l (i + 1) q) m.N p (memberDrawn S m)

open Classical in
variable {D γ γL M} in
/-- **The sessions' share off the held-off events is within the sessions' error of uniform from where it starts**:
wherever each member's coin map writes its link points in within `γL` of the tape's own and each round's coins within
`γ`, an event of the tape has, over a uniform tape from the position `p`, at most its share off the held-off events
through the members' sessions in order, plus every session's error. -/
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
      (fun q => avg_seqOff_le hR hL Q l (i + 1) q) m.N p (memberDrawn S m)

open Classical in
variable {D γ γL M} in
/-- **The sessions' share off the held-off events from a coupled start is within the start's term and the sessions'
error of uniform**: over coins `ω` that give a sample and a start of the period's tape, wherever the start leaves the
tape from the first stop fresh (`hF`) and is within `ε₀` of the uniform tape (`hS`), and each member's coin map
writes its live coins in within `γL` and `γ` of the tape's own. -/
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

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The window's coin map's premise within `ε`, off the held-off events** (`CouplesWithin`). -/
def WindowCouplesWithin (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold)
    (B : WindowRoundHold) (ε : ℝ≥0∞) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    windowOff V θ F c W t rd σ D M BL B Q + ε

/-- **The window's coin map's premise at a term within the hiding term.** -/
theorem windowCouples_of_within {W : Window} {t : ℕ} {rd : Reading W.rounds} {σ : Strategy (V.game θ F c)}
    {D : WindowDecoders V θ F c W σ} {M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds} {BL : WindowLinkHold}
    {B : WindowRoundHold} {ε : ℝ≥0∞} (h : WindowCouplesWithin V θ F c W t rd σ D M BL B ε)
    (hε : ε ≤ W.hidingTerm) : WindowCouples V θ F c W t rd σ D M BL B := fun Q =>
  (h Q).trans (add_le_add le_rfl hε)

/-- **The window's coin map's start is within `ε₀` of the uniform tape** (`StartWithin`). -/
def WindowStartWithin (W : Window) (σ : Strategy (V.game θ F c))
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (ε₀ : ℝ≥0∞) : Prop :=
  ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞, (∀ x, H x ≤ 1) →
    avg H ≤ avg (fun ω : (W.extend M.widening).L.Ω => H (M.start ω)) + ε₀

/-- **The window's projection's first stop** on the widened coins `ω`: where the first member's `Commit`'s roots are
decoded. -/
noncomputable def windowFirstStop (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (ω : (W.extend M.widening).L.Ω) :
    Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) :=
  (Pos.root (roundCuts V θ F c rd) (M.start ω)).toStop

/-- **The window's coin map's start leaves the tape from the first stop fresh** (`FirstStopFresh`): over the widened
coins, any reading of the sample and the projection's first stop has the mean of its average over the tapes from the
stop. -/
def WindowFirstStopFresh (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) : Prop :=
  ∀ G : Finset (Fin W.nJ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞,
    avg (fun ω : (W.extend M.widening).L.Ω =>
        G ((W.extend M.widening).L.draw ω) (windowFirstStop V θ F c W rd σ M ω)) =
      avg fun ω : (W.extend M.widening).L.Ω =>
        Pos.resample (G ((W.extend M.widening).L.draw ω)) (windowFirstStop V θ F c W rd σ M ω)

/-- **The stops' term over the window**: over the widened coins, every member's session's error along the walk from
the play's start, in the members' order (`seqErr`). -/
noncomputable def windowErr (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞) : ℝ≥0∞ :=
  avg fun ω : (W.extend M.widening).L.Ω =>
    seqErr D γ γL ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0
      ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω))

open Classical in
/-- **The window's coin map's premise within a term, off its local ones** (`couples_of_close`): the joint law's tape
off the held-off events is within the start's term and the stops' of the uniform tape at every event, wherever each
member's coin map writes its link points in within `γL` of the tape's own and each round's coins within `γ`, the start
leaves the tape from the first stop fresh, and the start is within `ε₀` of the uniform tape. The stops' term carries,
beside each member's `γL` and `γ`, the probability of its held-off events at each stop. -/
theorem windowCouples_of_close (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : WindowDecoders V θ F c W σ) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞} {ε₀ : ℝ≥0∞}
    (BL : WindowLinkHold) (B : WindowRoundHold) (hR : ∀ i mPts, (M.member i mPts).RoundClose (γ i))
    (hL : ∀ i mPts, (M.member i mPts).LinkClose (γL i mPts)) (hF : WindowFirstStopFresh V θ F c W rd σ M)
    (hS : WindowStartWithin V θ F c W σ M ε₀) :
    WindowCouplesWithin V θ F c W t rd σ D M BL B (ε₀ + windowErr V θ F c W t rd σ D M
      (fun i q j s => γ i q j s + prCoin (B i q.g' j s)) (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g'))) :=
  fun Q => by
  rw [ofReal_pr_eq_prCoin]
  exact seqOff_couples (windowReg V θ F c W rd σ D) BL B (roundCuts V θ F c rd)
    (C := (W.extend M.widening).L.Ω) (fun ω => (W.extend M.widening).L.draw ω) (fun ω => M.start ω) hR hL hF hS Q
    ((W.extend M.widening).joint t).ms 0

/-! ## The coin map into the openings, over the window -/

/-- **The window's coin map that writes the live coins into the openings** (`openMap`), at the widening `T` and the
start `start`: at every member's stops, the link points' bits or a round's coins' bits under `e`, written into the
stop's opening in a play that ends in `E` (`writeOpening`). -/
noncomputable def windowOpenMap {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ) :
    WindowCoinMap g σ Ω R where
  T := T
  start := start
  link _ p _ pts := writeOpening E p.s' (linkBits e pts)
  round _ p _ _ c := writeOpening E p.s' (roundBits e c)

/-- **Every member's coin map writes each round's coins in within the stop's opening's gap**
(`openMap_roundClose`). -/
theorem windowOpenMap_roundClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ)
    (i mPts : ℕ) :
    ((windowOpenMap e (R := R) E T start).member i mPts).RoundClose fun q _ s =>
      openGap E q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) := fun q _ _ G hG =>
  avg_le_writeOpening E (roundBits e) q.s' G hG

/-- **Every member's coin map writes the link points in within the stop's opening's gap** (`openMap_linkClose`). -/
theorem windowOpenMap_linkClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ)
    (i mPts : ℕ) :
    ((windowOpenMap e (R := R) E T start).member i mPts).LinkClose fun q =>
      openGap E q.g' (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool) := fun q G hG =>
  avg_le_writeOpening E (linkBits e) q.s' G hG

/-- **The sample-coupled coin map over the window** (`sampleMap`): widened by a tape of the period's game, its start
writes the sample's coins `κ` into the fresh frames of the sample's family (`sampleStart`), once for the window, and
at each member's stops it writes the live coins into the stop's opening in `σ`'s play, whose log ends in the close
after `σ`'s request (`windowOpenMap`). -/
noncomputable def windowSampleMap (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool) :
    WindowCoinMap (V.game θ F c) σ W.Ω W.rounds :=
  windowOpenMap e (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ)
    (sampleStart V θ F c rd σ label after κ)

/-- The window's projection's first stop is the walk from the start's tape to the first stop. -/
theorem windowFirstStop_eq (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (ω : (W.extend M.widening).L.Ω) :
    windowFirstStop V θ F c W rd σ M ω = Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd) (M.start ω) id :=
  rfl

/-- The sample-coupled start is the fill of the widening's tape with the coins `κ` at the device's coins. -/
theorem windowSampleMap_start_eq (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool)
    (ω : (W.extend (windowSampleMap e V θ F c W rd σ label after κ).widening).L.Ω) :
    (windowSampleMap e V θ F c W rd σ label after κ).start ω =
      fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2 :=
  rfl

/-- The sample-coupled map's first stop is the walk to it from the filled tape. -/
theorem windowFirstStop_sampleMap (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool)
    (ω : (W.extend (windowSampleMap e V θ F c W rd σ label after κ).widening).L.Ω) :
    windowFirstStop V θ F c W rd σ (windowSampleMap e V θ F c W rd σ label after κ) ω =
      Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd)
        (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2) id :=
  (windowFirstStop_eq V θ F c W rd σ (windowSampleMap e V θ F c W rd σ label after κ) ω).trans
    (congrArg (fun t => Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd) t id)
      (windowSampleMap_start_eq e V θ F c W rd σ label after κ ω))

/-- **The sample-coupled start leaves the tape from the first stop fresh** (`sampleMap_fresh`): it writes nothing
from the first stop on, and the window's one sample reads only the device's coins (`avg_prod_walk_fill`). -/
theorem windowSampleMap_fresh (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool) :
    WindowFirstStopFresh V θ F c W rd σ (windowSampleMap e V θ F c W rd σ label after κ) := fun G => by
  have h₁ := congrArg avg
    (funext fun ω : (W.extend (windowSampleMap e V θ F c W rd σ label after κ).widening).L.Ω =>
      congrArg₂ G (W.sample_extend _ ω) (windowFirstStop_sampleMap e V θ F c W rd σ label after κ ω))
  have h₂ := congrArg avg
    (funext fun ω : (W.extend (windowSampleMap e V θ F c W rd σ label after κ).widening).L.Ω =>
      congrArg₂ (fun s q => Pos.resample (G s) q) (W.sample_extend _ ω)
        (windowFirstStop_sampleMap e V θ F c W rd σ label after κ ω))
  refine h₁.trans (Eq.trans ?_ h₂.symm)
  generalize W.L.draw = dr
  exact avg_prod_walk_fill (C := W.Ω) (sampleSel V θ F c σ label after) (fun ω₁ : W.Ω => G (dr ω₁))
    (roundCuts V θ F c rd) m κ

/-- **The sample-coupled start is the uniform tape** wherever the sample's coins are uniform (`sampleMap_start`). -/
theorem windowSampleMap_start (W : Window) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : W.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    WindowStartWithin V θ F c W σ (windowSampleMap e V θ F c W rd σ label after κ) 0 := fun H _ => by
  rw [add_zero]
  refine le_of_eq (Eq.trans ?_ (avg_prod (C := W.Ω) (D := CoinTape (V.game θ F c) σ) _).symm)
  exact (avg_fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m H).symm.trans
    (hκ fun κ' => avg fun t =>
      H (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m κ' t)).symm

/-- **The window's bytes the sample reads, as fresh frames' coins, are uniform** (`avg_byteCoins`, A3 at the
window's bytes), the premise `windowSampleMap_start` takes of the sample's coins. -/
theorem Window.avg_byteCoins (W : Window) {m : ℕ} (hL : W.Lb 0 = m * 64) (H : (Fin m → Fin 512 → Bool) → ℝ≥0∞) :
    avg (fun ω : W.Ω => H (byteCoins W.src hL ω)) = avg H :=
  _root_.Verity.Sampled.avg_byteCoins W.hA3 hL H

omit θ F c in
/-- **The sample-coupled coin maps over the windows** at the readings `rd` (`sampleMaps`): at each window, commitment
and strategy, `windowSampleMap` at the family `label θ F c σ` answering `after θ F c σ` and the window's coins
`κ W`. -/
noncomputable def windowSampleMaps (rd : WindowReadings)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {m : ℕ}
    (κ : (W : Window) → W.Ω → Fin m → Fin 512 → Bool) (W : Window) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds :=
  windowSampleMap e V θ F c W (rd W θ F c) σ (label θ F c σ) (after θ F c σ) (κ W)

variable (k : Verity.Registration.Composite)

omit θ F c in
/-- **The coupling at the window's projection, off the coin map's premises within a term**
(`couplingTarget_withProject_of_close`): `windowCouplingTarget_of_logged` at the cut `P` and `Q`, off the held-off
events `BL` and `B`, with the coin map's premise discharged by `windowCouples_of_close` at each commitment and
strategy, wherever the start's term and the stops' (each member's `γL` and `BL`'s probability at its first stop, `γ`
and `B`'s at each later one) sum to at most the hiding term. -/
theorem windowCouplingTarget_of_close (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
    (M : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (W : Window) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (c : ByteArray) → Strategy (V.game θ F c) → WindowLinkHold)
    (B : (c : ByteArray) → Strategy (V.game θ F c) → WindowRoundHold)
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
    (hF : ∀ c σ, WindowFirstStopFresh V θ F c W (rd W θ F c) σ (M W θ F c σ))
    (hS : ∀ c σ, WindowStartWithin V θ F c W σ (M W θ F c σ) (ε₀ c σ))
    (hε : ∀ c σ, ε₀ c σ + windowErr V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ)
      (fun i q j s => γ c σ i q j s + prCoin (B c σ i q.g' j s))
      (fun i mPts q => γL c σ i mPts q + prCoin (BL c σ i mPts q.g')) ≤ W.hidingTerm)
    (hD : ∀ c σ, WindowDecodes V θ F c k (fun o => P o ∧ Q c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
      (M W θ F c σ) (BL c σ) (B c σ)) :
    WindowCouplingTarget V k (K.withProject V rd D M) W θ F :=
  windowCouplingTarget_of_logged V k rd K D M W P θ F Q BL B ha hq
    (fun c σ => windowCouples_of_within V θ F c
      (windowCouples_of_close V θ F c W _ _ σ _ _ (BL c σ) (B c σ) (hR c σ) (hL c σ) (hF c σ) (hS c σ)) (hε c σ))
    hD

end Verity.Sampled
