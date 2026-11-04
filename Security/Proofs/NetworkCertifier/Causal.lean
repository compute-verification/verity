import Proofs.NetworkCertifier.Accepted
import Proofs.Properties.CertifierNetwork.Instance

/-!
# A later window's sync never changes an earlier window's rows (5c)

`Properties.NetworkCertifier.ExpectedGridCausal` and `Properties.NetworkCertifier.EncardProgramRunsLeWindows`, with the
latter's honest and nontrivial lemmas.

The program hands frame `f` at `max(eligible f, last)` in canonical order (`NetworkCertifier.handingFrom`), and a frame
anchored in window `w` is eligible no earlier than its anchor, at or after `w·T`. So two runs of `handingFrom` whose
syncs agree on the windows before `m` give, frame by frame, hand ticks that are equal or both at or after `m·T`
(`handingFrom_causal`): the enqueue counts of the buckets before `m·T` agree (`enqueued_causal`), and so do the FIFO's
rows there (`fifoGrid_congr`). The row at bucket `t` reads the syncs of windows `0 … t / T` only
(`expectedRow_congr_syncs`), and the grid the check expects of window `w` the syncs the link declares for windows
`0 … w` (`expectedGridCausal`).

At the reference's constant-rate Program of one execution block of the `W` windows, sent at the program's hand ticks
(`programFrames`), a step's rows read the content advice and the syncs of the windows up to its own
(`tiedRows_causal`). So an accepted sequence of the first `m` windows is fixed by the content advice, the syncs of its
windows below `min W m` and its jitter classes, and those sequences number at most `#Ac · |Σ_sync|^min(W, m) · D^(m·T)`,
by the injection of `NetTiming.encard_accSeqs_le_constantRate_windows` (`encardProgramRunsLeWindows`). An accepted run of
the check is a run of that Program (`acceptedRunsAreProgramRuns`), whose first `m·T` buckets it accepts
(`NetTiming.accepted_take`), so the check's accepted runs over their first `m ≤ W` windows number at most
`#Ac · |Σ_sync|^m · D^(m·T)`, or one carries a break (`encard_acceptedObservations_take_le`).

The honest lemma counts the honest network certifier's record at `Instance.config`, one window of one bucket: the
Program's own row of one frame anchored at bucket 0. The nontrivial lemma takes the prediction that reads its rows off
the advice, which accepts every sequence with complete statuses: there two sequences of the one bucket differ in the
payload of its one slot, against the bound's one at `Instance.config`'s parameters.
-/

open NetworkCheck.Check (expectedGrid noSync)
open NetworkCertifier (ClockSync LogicalFrame expectedRow)
open Properties.NetworkCertifier
open Properties.CertifierNetwork.Instance (config oneFrame oneAdvice honestObs)

namespace Proofs.NetworkCertifier

/-! ## The program's hand ticks -/

/-- The tick `handingFrom` hands frame `f` at, after the tick `last` of the frame before. -/
def tick (T W : ℕ) (syncs : Fin W → ClockSync) (last : ℕ) (f : LogicalFrame) : ℕ :=
  if hw : f.anchor / T < W then max (_root_.NetworkCertifier.eligible f (syncs ⟨f.anchor / T, hw⟩)) last
  else max (max last (W * T)) f.anchor

theorem handingFrom_cons (T W : ℕ) (syncs : Fin W → ClockSync) (last : ℕ) (f : LogicalFrame)
    (fs : List LogicalFrame) :
    _root_.NetworkCertifier.handingFrom T W syncs last (f :: fs) =
      tick T W syncs last f :: _root_.NetworkCertifier.handingFrom T W syncs (tick T W syncs last f) fs := rfl

theorem anchor_le_eligible (f : LogicalFrame) (s : ClockSync) : f.anchor ≤ _root_.NetworkCertifier.eligible f s :=
  le_trans (Nat.le_add_right _ s.delta) (Nat.le_add_right _ _)

/-- One step of `handingFrom` under syncs that agree on the windows before `m`: from ticks that are equal or both at or
after `m·T`, the next ticks are too. -/
theorem tick_causal (T W m : ℕ) (syncs syncs' : Fin W → ClockSync)
    (hs : ∀ w : Fin W, (w : ℕ) < m → syncs w = syncs' w) (f : LogicalFrame) {last last' : ℕ}
    (hl : last = last' ∨ m * T ≤ last ∧ m * T ≤ last') :
    tick T W syncs last f = tick T W syncs' last' f ∨
      m * T ≤ tick T W syncs last f ∧ m * T ≤ tick T W syncs' last' f := by
  by_cases hm : f.anchor / T < m
  · rcases hl with rfl | ⟨hl, hl'⟩
    · left
      by_cases hw : f.anchor / T < W
      · simp only [tick, dite_eq_left hw, hs ⟨_, hw⟩ hm]
      · simp only [tick, dite_eq_right hw]
    · right
      by_cases hw : f.anchor / T < W
      · simp only [tick, dite_eq_left hw]
        exact ⟨le_trans hl (le_max_right _ _), le_trans hl' (le_max_right _ _)⟩
      · simp only [tick, dite_eq_right hw]
        exact ⟨le_trans hl (le_trans (le_max_left _ _) (le_max_left _ _)),
          le_trans hl' (le_trans (le_max_left _ _) (le_max_left _ _))⟩
  · right
    have ha : m * T ≤ f.anchor :=
      le_trans (Nat.mul_le_mul_right T (Nat.le_of_not_lt hm)) (Nat.div_mul_le_self _ _)
    by_cases hw : f.anchor / T < W
    · simp only [tick, dite_eq_left hw]
      exact ⟨le_trans ha (le_trans (anchor_le_eligible f _) (le_max_left _ _)),
        le_trans ha (le_trans (anchor_le_eligible f _) (le_max_left _ _))⟩
    · simp only [tick, dite_eq_right hw]
      exact ⟨le_trans ha (le_max_right _ _), le_trans ha (le_max_right _ _)⟩

/-- **The hand ticks are causal.** Under syncs that agree on the windows before `m`, from ticks that are equal or both
at or after `m·T`, every frame's hand tick is equal under both, or at or after `m·T` under both. -/
theorem handingFrom_causal (T W m : ℕ) (syncs syncs' : Fin W → ClockSync)
    (hs : ∀ w : Fin W, (w : ℕ) < m → syncs w = syncs' w) :
    ∀ (frames : List LogicalFrame) (last last' : ℕ), (last = last' ∨ m * T ≤ last ∧ m * T ≤ last') →
      List.Forall₂ (fun x y => x = y ∨ m * T ≤ x ∧ m * T ≤ y)
        (_root_.NetworkCertifier.handingFrom T W syncs last frames)
        (_root_.NetworkCertifier.handingFrom T W syncs' last' frames)
  | [], _, _, _ => List.Forall₂.nil
  | f :: fs, _, _, hl => by
    rw [handingFrom_cons, handingFrom_cons]
    have hk := tick_causal T W m syncs syncs' hs f hl
    exact List.Forall₂.cons hk (handingFrom_causal T W m syncs syncs' hs fs _ _ hk)

theorem count_eq_of_forall₂ {N u : ℕ} (hu : u < N) {xs ys : List ℕ}
    (h : List.Forall₂ (fun x y => x = y ∨ N ≤ x ∧ N ≤ y) xs ys) : xs.count u = ys.count u := by
  induction h with
  | nil => rfl
  | @cons x y _ _ hxy _ ih =>
    rw [List.count_cons, List.count_cons, ih]
    rcases hxy with rfl | ⟨hx, hy⟩
    · rfl
    · have hx' : x ≠ u := by omega
      have hy' : y ≠ u := by omega
      simp [hx', hy']

/-- The enqueue counts of the buckets before `m·T` read the syncs of the windows before `m` only. -/
theorem enqueued_causal (T W m : ℕ) (syncs syncs' : Fin W → ClockSync) (frames : List LogicalFrame)
    (hs : ∀ w : Fin W, (w : ℕ) < m → syncs w = syncs' w) {u : ℕ} (hu : u < m * T) :
    _root_.NetworkCertifier.enqueued T W syncs frames u = _root_.NetworkCertifier.enqueued T W syncs' frames u := by
  simp only [_root_.NetworkCertifier.enqueued, _root_.NetworkCertifier.handingBuckets,
    count_eq_of_forall₂ hu (handingFrom_causal T W m syncs syncs' hs frames 0 0 (Or.inl rfl))]

/-! ## The FIFO reads the enqueue counts up to its bucket -/

theorem queue_congr (S : ℕ) {enq enq' : ℕ → ℕ} :
    ∀ u : ℕ, (∀ v < u, enq v = enq' v) → _root_.NetworkCertifier.queue S enq u = _root_.NetworkCertifier.queue S enq' u
  | 0, _ => rfl
  | u + 1, h => by
    simp only [_root_.NetworkCertifier.queue, queue_congr S u fun v hv => h v (Nat.lt_succ_of_lt hv),
      h u (Nat.lt_succ_self u)]

theorem released_congr (S : ℕ) {enq enq' : ℕ → ℕ} {u : ℕ} (h : ∀ v ≤ u, enq v = enq' v) :
    _root_.NetworkCertifier.released S enq u = _root_.NetworkCertifier.released S enq' u :=
  Fin.ext (show min (_root_.NetworkCertifier.queue S enq u + enq u) S =
      min (_root_.NetworkCertifier.queue S enq' u + enq' u) S by
    rw [queue_congr S u fun v hv => h v hv.le, h u le_rfl])

theorem fifoGrid_congr (T S : ℕ) {F : Type} (f : ℕ → F) {enq enq' : ℕ → ℕ} (h : ∀ u < T, enq u = enq' u) :
    _root_.NetworkCertifier.fifoGrid T S f enq = _root_.NetworkCertifier.fifoGrid T S f enq' := by
  have hr : (fun t : Fin T => _root_.NetworkCertifier.released S enq t) =
      fun t : Fin T => _root_.NetworkCertifier.released S enq' t :=
    funext fun t => released_congr S fun v hv => h v (lt_of_le_of_lt hv t.2)
  unfold _root_.NetworkCertifier.fifoGrid
  exact congrArg (_root_.NetworkCertifier.fill f) hr

/-- **The program's row at bucket `t` reads the syncs of windows `0 … t / T` only.** -/
theorem expectedRow_congr_syncs (S T W : ℕ) (syncs syncs' : Fin W → ClockSync) (frames : List LogicalFrame) (t : ℕ)
    (h : ∀ w : Fin W, (w : ℕ) ≤ t / T → syncs w = syncs' w) :
    expectedRow S T W syncs frames t = expectedRow S T W syncs' frames t := by
  have henq : ∀ u < t + 1, _root_.NetworkCertifier.enqueued T W syncs frames u =
      _root_.NetworkCertifier.enqueued T W syncs' frames u := by
    intro u hu
    rcases Nat.eq_zero_or_pos T with hT | hT
    · simp [_root_.NetworkCertifier.enqueued, hT]
    · refine enqueued_causal T W (t / T + 1) syncs syncs' frames (fun w hw => h w (Nat.lt_succ_iff.mp hw)) ?_
      calc u ≤ t := Nat.lt_succ_iff.mp hu
        _ < t / T * T + T := Nat.lt_div_mul_add hT
        _ = (t / T + 1) * T := by rw [Nat.add_mul, Nat.one_mul]
  unfold _root_.NetworkCertifier.expectedRow
  rw [fifoGrid_congr (t + 1) S _ henq]

theorem expectedGridCausal : ExpectedGridCausal := by
  intro g W frames early later later' w
  simp only [expectedGrid, List.map_toArray]
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hT : i < g.buckets := List.mem_range.mp hi
  have hdiv : ((w : ℕ) * g.buckets + i) / g.buckets = w := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (lt_of_le_of_lt (Nat.zero_le i) hT), Nat.div_eq_of_lt hT,
      Nat.zero_add]
  rw [expectedRow_congr_syncs g.allocation g.buckets W (fun v => (early ++ later).getD v.1 noSync)
    (fun v => (early ++ later').getD v.1 noSync) frames ((w : ℕ) * g.buckets + i) fun v hv => ?_]
  have hvw : (v : ℕ) ≤ w := by simpa only [hdiv] using hv
  have hv' : (v : ℕ) < early.size := lt_of_le_of_lt hvw w.2
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_append_left hv']

/-! ## The first `m` windows -/

/-- The tied Program's rows at bucket `t` under the content advice `c` read the syncs of windows `0 … t / T` only. -/
theorem tiedRows_causal (cfg : _root_.CertifierDevice.Config) {In Ac : Type}
    (frames : List In → Ac → Fin 1 → List LogicalFrame) (t : ℕ) (ins : List In) (c : Ac)
    (σ σ' : Fin cfg.windows → SyncValue cfg.syncSet)
    (hσ : ∀ w : Fin cfg.windows, (w : ℕ) ≤ t / cfg.egress.buckets → σ w = σ' w) :
    (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets cfg.windows 1
        (programFrames cfg frames)).rows t ins (fun _ => (c, σ)) =
      (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets cfg.windows 1
        (programFrames cfg frames)).rows t ins (fun _ => (c, σ')) := by
  funext ℓ
  simp only [NetTiming.constantRate]
  split_ifs with h
  · have h0 : t / (cfg.windows * cfg.egress.buckets) = 0 := Nat.lt_one_iff.mp h
    have hmod : t % (cfg.windows * cfg.egress.buckets) = t := by
      rcases Nat.eq_zero_or_pos (cfg.windows * cfg.egress.buckets) with hz | hz
      · rw [hz, Nat.mod_zero]
      · have := (Nat.div_lt_iff_lt_mul hz).mp h
        exact Nat.mod_eq_of_lt (by omega)
    have e : ∀ v, programFrames cfg frames ins ⟨t / (cfg.windows * cfg.egress.buckets), h⟩ v ℓ =
        programFrames cfg frames ins 0 v ℓ := fun _ => rfl
    rw [e, e, h0, Nat.zero_mul, hmod]
    obtain ⟨hp, hs⟩ := programFrames_fields cfg frames ins 0 c σ ℓ
    obtain ⟨hp', hs'⟩ := programFrames_fields cfg frames ins 0 c σ' ℓ
    rw [← NetTiming.SecurityProofs.CodeExpectedRow _ _ _ _ _ _ hp hs t,
      ← NetTiming.SecurityProofs.CodeExpectedRow _ _ _ _ _ _ hp' hs' t]
    exact expectedRow_congr_syncs _ _ _ _ _ _ _ fun w hw => congrArg SyncValue.sync (hσ w hw)
  · rfl

theorem encardProgramRunsLeWindows : EncardProgramRunsLeWindows := by
  intro cfg K In Ac _ frames op m D
  classical
  by_cases hne : Nonempty (Fin 1 → Ac × (Fin cfg.windows → SyncValue cfg.syncSet))
  swap
  · have : NetTiming.AccSeqs (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets cfg.windows 1
        (programFrames cfg frames)) op (m * cfg.egress.buckets) D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, -⟩; exact hne ⟨a⟩
    rw [this, Set.encard_empty]; exact bot_le
  let a0 : Fin 1 → Ac × (Fin cfg.windows → SyncValue cfg.syncSet) := Classical.choice hne
  let adv : List (NetTiming.Obs 1 K cfg.egress.allocation) → Fin 1 → Ac × (Fin cfg.windows → SyncValue cfg.syncSet) :=
    fun o => if h : NetTiming.AcceptedUpTo (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets
      cfg.windows 1 (programFrames cfg frames)) op (m * cfg.egress.buckets) o then h.choose else a0
  have adv_spec : ∀ o ∈ NetTiming.AccSeqs (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets
      cfg.windows 1 (programFrames cfg frames)) op (m * cfg.egress.buckets) D,
      NetTiming.Accepted (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets cfg.windows 1
        (programFrames cfg frames)) op (m * cfg.egress.buckets) o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec
  let cut : (Fin 1 → Ac × (Fin cfg.windows → SyncValue cfg.syncSet)) →
      Ac × (Fin (min cfg.windows m) → SyncValue cfg.syncSet) :=
    fun a => ((a 0).1, fun i => (a 0).2 ⟨i, lt_of_lt_of_le i.2 (min_le_left _ _)⟩)
  let code : List (NetTiming.Obs 1 K cfg.egress.allocation) →
      (Ac × (Fin (min cfg.windows m) → SyncValue cfg.syncSet)) × (Fin (m * cfg.egress.buckets) → ℕ) :=
    fun o => (cut (adv o), NetTiming.jitterCode o)
  have hone : ∀ o, adv o = fun _ => ((adv o 0).1, (adv o 0).2) :=
    fun o => funext fun i => by rw [Subsingleton.elim i 0]
  have hinj : Set.InjOn code (NetTiming.AccSeqs (NetTiming.constantRate (S := cfg.egress.allocation)
      cfg.egress.buckets cfg.windows 1 (programFrames cfg frames)) op (m * cfg.egress.buckets) D) := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨hσ, hj⟩ := h
    have acc := adv_spec o ho
    have acc' := adv_spec o' ho'
    apply NetTiming.eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    apply NetTiming.Obs.ext'
    · rw [hr, hr', NetTiming.ingressUpTo_congr op hpre, hone o, hone o']
      have hc : (adv o 0).1 = (adv o' 0).1 := by have h := congrArg Prod.fst hσ; exact h
      rw [hc]
      apply tiedRows_causal
      intro w hw
      have hwm : (w : ℕ) < m :=
        lt_of_le_of_lt hw (Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact ht))
      have := congrFun (congrArg Prod.snd hσ) ⟨w, lt_min w.2 hwm⟩
      simpa only [cut, Fin.eta] using this
    · exact NetTiming.jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  calc (NetTiming.AccSeqs (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets cfg.windows 1
        (programFrames cfg frames)) op (m * cfg.egress.buckets) D).encard
      ≤ (((Finset.univ : Finset (Ac × (Fin (min cfg.windows m) → SyncValue cfg.syncSet))) ×ˢ
          Fintype.piFinset fun _ : Fin (m * cfg.egress.buckets) => Finset.range D).card : ℕ∞) := by
        apply NetTiming.encard_le_card_of_injOn code hinj
        intro o ho
        simp only [code, Finset.mem_product, Finset.mem_univ, true_and]
        exact NetTiming.jitterCode_mem o ho.1 ho.2.1
    _ = _ := by
        rw [Finset.card_product, Fintype.card_piFinset]
        simp only [Finset.card_univ, Finset.card_range, Finset.prod_const, Fintype.card_prod, Fintype.card_fun,
          Fintype.card_fin, Fintype.card_coe, List.toFinset_card_of_nodup cfg.syncSet.deltas_nodup,
          List.toFinset_card_of_nodup cfg.syncSet.rhos_nodup]

section Check

open NetworkCheck hiding Params
open NetworkCheck.Check
open _root_.CertifierDevice

/-- **The check's accepted runs over their first `m` windows.** For a device that runs as built, the first `m ≤ W`
windows of the observation sequences of its accepted runs number at most `#Ac · |Σ_sync|^m · D^(m·T)`, or some accepted
run carries a break (`ObservationsBounded`'s): an accepted run is a run of the tied Program
(`acceptedRunsAreProgramRuns`), the Program accepts its first `m·T` buckets (`NetTiming.accepted_take`), and those
number at most `encardProgramRunsLeWindows`'s count. -/
theorem encard_acceptedObservations_take_le (device : DeviceInput → DeviceOutput) (cfg : Config)
    (key : Lms.PrivateKey) {K : ℕ} {In Ac : Type} [Fintype Ac]
    (frames : List In → Ac → Fin 1 → List _root_.NetworkCertifier.LogicalFrame)
    (op : ℕ → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String) (m D : ℕ)
    (hm : m ≤ cfg.windows) (hint : NetTiming.DeviceIntact device deviceCode) :
    ((List.take (m * cfg.egress.buckets)) '' AcceptedObservations cfg key device frames op linkOf D).encard ≤
        ((Fintype.card Ac * (cfg.syncSet.deltas.length * cfg.syncSet.rhos.length) ^ m *
          D ^ (m * cfg.egress.buckets) : ℕ) : ℕ∞) ∨
      ∃ o x c e a, AcceptedRun cfg key device frames op linkOf o x c e a ∧
        ((∃ y ∈ runStrings x (device x) c e, ∃ z ∈ runStrings x (device x) c e, y ≠ z ∧
            Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
          ∃ r ∈ c.records.toList, ∃ msg, certMessage x.issuer.session r = some msg ∧
            LmsVerifies (deviceParams x) msg r.certificate.signature ∧ ¬ Signed x.issuer.session (device x) msg) := by
  by_cases hb : ∃ o x c e a, AcceptedRun cfg key device frames op linkOf o x c e a ∧
      ((∃ y ∈ runStrings x (device x) c e, ∃ z ∈ runStrings x (device x) c e, y ≠ z ∧
          Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
        ∃ r ∈ c.records.toList, ∃ msg, certMessage x.issuer.session r = some msg ∧
          LmsVerifies (deviceParams x) msg r.certificate.signature ∧ ¬ Signed x.issuer.session (device x) msg)
  · exact Or.inr hb
  have hbound := encardProgramRunsLeWindows cfg frames op m D
  rw [min_eq_right hm] at hbound
  refine Or.inl (le_trans (Set.encard_le_encard ?_) hbound)
  rintro _ ⟨o, ⟨hlen, hjit, x, c, e, a, hrun⟩, rfl⟩
  obtain ⟨hx, hk, hacc, hlink, hobs, hopen⟩ := hrun
  subst hx hk
  rcases acceptedRunsAreProgramRuns device x c e hacc hint with hr | hbr
  · obtain ⟨σ, -, hA⟩ := hr frames a op o linkOf hlink hlen hobs hopen
    have hmT : m * x.config.egress.buckets ≤ x.config.windows * x.config.egress.buckets :=
      Nat.mul_le_mul_right _ hm
    exact ⟨by rw [List.length_take, hlen]; exact Nat.min_eq_left hmT,
      fun y hy => hjit y (List.mem_of_mem_take hy), fun _ => (a, σ), NetTiming.accepted_take _ op hmT hA⟩
  · exact (hb ⟨o, x, c, e, a, ⟨rfl, rfl, hacc, hlink, hobs, hopen⟩, hbr⟩).elim

end Check

/-! ## Its honest and nontrivial lemmas -/

/-- **The program's honest run is counted**: at `Instance.config` (one window of one bucket, allocation 1, Σ_sync of
one value), the honest record of one frame anchored at bucket 0 is among the sequences `EncardProgramRunsLeWindows`
counts over the first window. -/
theorem encardProgramRunsLeWindows_honest :
    honestObs ∈ NetTiming.AccSeqs (NetTiming.constantRate (S := config.egress.allocation)
      config.egress.buckets config.windows 1 (programFrames config oneFrame))
      (fun _ _ => ()) (1 * config.egress.buckets) 1 := by
  refine ⟨rfl, fun x hx => ?_, oneAdvice, fun t ht => ?_⟩
  · simp only [honestObs, List.mem_singleton] at hx
    subst hx
    exact Nat.one_pos
  · have hb : config.egress.buckets = 1 := rfl
    obtain rfl : t = 0 := by omega
    exact ⟨Nat.one_pos, rfl, fun _ => rfl⟩

/-- At the prediction that reads its rows off the advice, which accepts every sequence whose statuses are complete,
two sequences of one step differ in the payload of its first slot. -/
theorem two_le_acceptAll {S : ℕ} (hS : 0 < S) :
    2 ≤ (NetTiming.AccSeqs (L := 1) (K := 1) (S := S) (In := Unit) (Adv := ℕ → Fin 1 → Fin S → Option (List Bool))
      ⟨fun t _ a => a t⟩ (fun _ _ => ()) 1 1).encard := by
  let o : Bool → List (NetTiming.Obs 1 1 S) := fun b => [⟨fun _ _ => some [b], 0, fun _ => .complete⟩]
  have hmem : ∀ b, o b ∈ NetTiming.AccSeqs (L := 1) (K := 1) (S := S) (In := Unit)
      (Adv := ℕ → Fin 1 → Fin S → Option (List Bool)) ⟨fun t _ a => a t⟩ (fun _ _ => ()) 1 1 := by
    intro b
    refine ⟨rfl, fun x hx => ?_, fun _ _ _ => some [b], fun t ht => ?_⟩
    · simp only [o, List.mem_singleton] at hx
      subst hx
      exact Nat.one_pos
    · obtain rfl : t = 0 := by omega
      exact ⟨Nat.one_pos, rfl, fun _ => rfl⟩
  have hne : o true ≠ o false := by
    intro he
    have := congrArg (fun l : List (NetTiming.Obs 1 1 S) => l.head?.map fun x => x.rows 0 ⟨0, hS⟩) he
    simp [o] at this
  rw [← Set.encard_pair hne]
  exact Set.encard_le_encard (by rintro x (rfl | rfl) <;> exact hmem _)

/-- **`EncardProgramRunsLeWindows`'s bound fails at the verifier that accepts everything**: there, at
`Instance.config`'s parameters (one window of one bucket, allocation 1, Σ_sync of one value, the content advice `Unit`,
`D = 1`), the bucket's one slot carries `[true]` or `[false]`, two sequences against the bound's one. -/
theorem encardProgramRunsLeWindows_nontrivial :
    ¬ (NetTiming.AccSeqs (L := 1) (K := 1) (S := config.egress.allocation) (In := Unit)
        (Adv := ℕ → Fin 1 → Fin config.egress.allocation → Option (List Bool)) ⟨fun t _ a => a t⟩
        (fun _ _ => ()) (1 * config.egress.buckets) 1).encard ≤
      ((Fintype.card Unit * (config.syncSet.deltas.length * config.syncSet.rhos.length) ^
        min config.windows 1 * 1 ^ (1 * config.egress.buckets) : ℕ) : ℕ∞) := by
  have hb : Fintype.card Unit * (config.syncSet.deltas.length * config.syncSet.rhos.length) ^
      min config.windows 1 * 1 ^ (1 * config.egress.buckets) = 1 := by
    simp [config, Properties.CertifierNetwork.Instance.syncSet]
  have hn : 1 * config.egress.buckets = 1 := rfl
  rw [hb, hn]
  intro h
  have := le_trans (two_le_acceptAll (S := config.egress.allocation) Nat.one_pos) h
  norm_num at this

end Proofs.NetworkCertifier
