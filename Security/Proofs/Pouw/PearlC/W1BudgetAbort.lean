import Definitions.Pouw.PearlC.W1

/-!
# W1 doesn't truncate

`BudgetAbortPearlC` is `Accounting.BudgetAbort`'s statement over Pearl-C's `CostModel` (ra-per-run's hypothesis for the
per-run bound). W1 (`Pouw.PearlC.W1.w1`) fails it whenever its prices are positive (`Positive`), because a correct
unit's operand fetches are free and whether a unit is correct is decided on the final transcript under the oracle,
which no W1 program reads.

- `copyAct` moves unit 0's activation word `(0, 0)` into its ticket word `(0, 0)`: one free move and one fetch, which
  W1 doesn't charge when unit 0 comes out correct.
- A program that costs nothing on a run fetches no pre-salt word and runs only free moves and fragments the device
  lacks, so every word outside the pre-salt regions stays below 256 (`small_exec_code`).
- `w1_not_budgetAbort`: at budget 0, a copy of `copyAct` must commit its transcript under every oracle, since W1's run
  doesn't read the oracle. Under an oracle where unit 0 is wrong, nothing is free, so the copy must cost nothing and
  can't write a ticket word of 256 or more.
- `sm120_not_budgetAbort`: on sm_120 at any positive SHA-512 and fetch prices, with a one-unit protocol whose cap holds
  exactly when the oracle says so (`toyP`), so the theorem's hypotheses hold of a concrete instance.

Without the refund, W1's charge is fixed by the run's path and fetches, but its language still has no free abort: an
`ite`'s continuation is shared by both branches, and moving it into them taints it. That case is open here.
-/

namespace Pouw.PearlC.W1BudgetAbort

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.W1

variable {Q R S : Type}

/-- **The model truncates**: `Accounting.BudgetAbort`'s statement over Pearl-C's `CostModel`. -/
def BudgetAbortPearlC (CM : CostModel Q R S) : Prop :=
  (∀ (prog : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      0 ≤ CM.cost prog H U act σ s) ∧
    ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Codes) (b : ℝ), 0 ≤ b → ∃ prog' : CM.Prog,
      ∀ (H : Q → R) (σ : State) (s : S),
        CM.cost prog' H U act σ s ≤ b ∧ CM.calls prog' H U act σ s ≤ CM.calls prog H U act σ s ∧
          (CM.cost prog H U act σ s ≤ b → CM.run prog' H U act σ s = CM.run prog H U act σ s)

/-- W1's prices are positive: INT32, FP32, a cast, a compression, a fetch, and every fragment the device has. -/
def Positive (dv : W1Device) : Prop :=
  0 < dv.prices.int32 ∧ 0 < dv.prices.fp32 ∧ 0 < dv.prices.cvt ∧ 0 < dv.prices.sha512 ∧ 0 < dv.prices.fetch ∧
    ∀ f : Frag, dv.has f = true → 0 < ((f.m * f.n * dv.depth f.fmt : ℕ) : ℚ) * dv.prices.mac f.fmt

section Prices

variable {dv : W1Device}

theorem base_nonneg (h : Positive dv) (i : Instr) : 0 ≤ i.base dv := by
  obtain ⟨hi, hf, hc, hs, -, hm⟩ := h
  cases i with
  | mov => exact le_rfl
  | ld => exact le_rfl
  | st => exact le_rfl
  | iop => exact hi.le
  | iadd3 => exact hi.le
  | lop3 => exact hi.le
  | prmt => exact hi.le
  | sel => exact hi.le
  | fop => exact hf.le
  | cvt => exact hc.le
  | mma f =>
    simp only [Instr.base]
    split_ifs with hh
    · exact (hm f hh).le
    · exact le_rfl
  | sha => exact hs.le

theorem base_le_price (h : Positive dv) (t : Bool) (i : Instr) : i.base dv ≤ i.price dv t := by
  unfold Instr.price
  have : 0 ≤ (if t || i.dynamic then dv.prices.int32 * i.words dv else 0) := by
    split_ifs
    · exact mul_nonneg h.1.le (Nat.cast_nonneg _)
    · exact le_rfl
  linarith

theorem price_nonneg (h : Positive dv) (t : Bool) (i : Instr) : 0 ≤ i.price dv t :=
  (base_nonneg h i).trans (base_le_price h t i)

end Prices

/-- A run only adds to its cost and its fetches. -/
theorem exec_mono (dv : W1Device) (h : Positive dv) :
    ∀ (c : W1.Code) (t : Bool) (st : St), st.cost ≤ (exec dv t c st).cost ∧ st.fetched <+: (exec dv t c st).fetched
  | .done, _, _ => ⟨le_rfl, List.prefix_refl _⟩
  | .op i k, t, st => by
      obtain ⟨h1, h2⟩ := exec_mono dv h k t
        ⟨i.exec dv st.mem, st.fetched ++ (i.reads dv st.mem).filter preSalt, st.cost + i.price dv t⟩
      dsimp only at h1 h2
      simp only [exec]
      exact ⟨by linarith [price_nonneg h t i], (List.prefix_append _ _).trans h2⟩
  | .ite p a b k, t, st => by
      have key : ∀ mid : St, st.cost + dv.prices.int32 ≤ mid.cost → st.fetched ++ [p].filter preSalt <+: mid.fetched →
          st.cost ≤ (exec dv t k mid).cost ∧ st.fetched <+: (exec dv t k mid).fetched := by
        intro mid hc hf
        obtain ⟨k1, k2⟩ := exec_mono dv h k t mid
        exact ⟨by linarith [h.1], ((List.prefix_append _ _).trans hf).trans k2⟩
      let st1 : St := ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + dv.prices.int32⟩
      simp only [exec]
      split_ifs
      · exact key _ (exec_mono dv h b true st1).1 (exec_mono dv h b true st1).2
      · exact key _ (exec_mono dv h a true st1).1 (exec_mono dv h a true st1).2

/-- Every word outside the pre-salt regions is below 256: the salt's bytes and zeros. -/
def Small (M : Mem) : Prop := ∀ a : Addr, preSalt a = false → M a < 256

theorem small_init (salt : List ℕ) (pr : CodeImage) (U : Workload) (act : ℕ → Codes) (σ : State) :
    Small (init salt pr U act σ) := by
  intro a ha
  have h2 : a.1 ≠ 2 := fun e => by simp [preSalt, e] at ha
  have h3 : a.1 ≠ 3 := fun e => by simp [preSalt, e] at ha
  have h4 : a.1 ≠ 4 := fun e => by simp [preSalt, e] at ha
  have h5 : a.1 ≠ 5 := fun e => by simp [preSalt, e] at ha
  simp only [init]
  by_cases h1 : a.1 = 1
  · rw [if_pos h1]
    exact Nat.mod_lt _ (by norm_num)
  · rw [if_neg h1, if_neg h2, if_neg h3, if_neg h4, if_neg h5]
    norm_num

/-- An instruction W1 charges nothing, that fetches no pre-salt word, keeps the words outside the pre-salt regions small. -/
theorem small_exec {dv : W1Device} (h : Positive dv) {t : Bool} {i : Instr} {M : Mem} (hp : i.price dv t = 0)
    (hr : (i.reads dv M).filter preSalt = []) (hM : Small M) : Small (i.exec dv M) := by
  have pos : ∀ x : ℚ, 0 < x → x ≤ i.base dv → False := fun x hx hle => by
    have := base_le_price h t i
    linarith
  have dyn : i.dynamic = true → False := fun hd => by
    have hw : 1 ≤ i.words dv := by cases i <;> simp_all [Instr.dynamic, Instr.words]
    have := base_nonneg h i
    have hq : (1 : ℚ) ≤ (i.words dv : ℚ) := by exact_mod_cast hw
    have : dv.prices.int32 ≤ dv.prices.int32 * (i.words dv : ℚ) := by nlinarith [h.1]
    simp only [Instr.price, hd, Bool.or_true, if_true] at hp
    linarith [h.1]
  obtain ⟨hi, hf, hc, hs, -, hm⟩ := h
  cases i with
  | mov d a =>
    have ha : preSalt a = false := by simpa [Instr.reads] using hr
    intro x hx
    simp only [Instr.exec, Function.update_apply]
    split_ifs
    · exact lt_of_le_of_lt (Nat.mod_le _ _) (hM a ha)
    · exact hM x hx
  | ld => exact (dyn rfl).elim
  | st => exact (dyn rfl).elim
  | iop => exact (pos _ hi le_rfl).elim
  | iadd3 => exact (pos _ hi le_rfl).elim
  | lop3 => exact (pos _ hi le_rfl).elim
  | prmt => exact (pos _ hi le_rfl).elim
  | sel => exact (pos _ hi le_rfl).elim
  | fop => exact (pos _ hf le_rfl).elim
  | cvt => exact (pos _ hc le_rfl).elim
  | sha => exact (pos _ hs le_rfl).elim
  | mma f d a b c sa sb =>
    by_cases hh : dv.has f = true
    · exact (pos _ (hm f hh) (by simp only [Instr.base, hh, if_true]; exact le_rfl)).elim
    · have e : Instr.exec dv M (.mma f d a b c sa sb) = M := by simp only [Instr.exec, hh]; rfl
      rw [e]
      exact hM

/-- **A run W1 charges nothing, with no pre-salt fetch, keeps the words outside the pre-salt regions small.** -/
theorem small_exec_code (dv : W1Device) (h : Positive dv) :
    ∀ (c : W1.Code) (t : Bool) (st : St), (exec dv t c st).cost = st.cost → (exec dv t c st).fetched = st.fetched →
      Small st.mem → Small (exec dv t c st).mem
  | .done, _, _ => fun _ _ hM => by simpa only [exec] using hM
  | .op i k, t, st => fun hc hf hM => by
      simp only [exec] at hc hf ⊢
      obtain ⟨m1, m2⟩ := exec_mono dv h k t
        ⟨i.exec dv st.mem, st.fetched ++ (i.reads dv st.mem).filter preSalt, st.cost + i.price dv t⟩
      dsimp only at m1 m2
      have hp0 : i.price dv t = 0 := by linarith [price_nonneg h t i]
      rw [hf] at m2
      have hl := m2.length_le
      rw [List.length_append] at hl
      have hr : (i.reads dv st.mem).filter preSalt = [] := List.eq_nil_of_length_eq_zero (by omega)
      refine small_exec_code dv h k t _ ?_ ?_ (small_exec h hp0 hr hM)
      · dsimp only
        rw [hc, hp0, add_zero]
      · dsimp only
        rw [hf, hr, List.append_nil]
  | .ite p a b k, t, st => fun hc _ _ => by
      exfalso
      simp only [exec] at hc
      split_ifs at hc
      · have h1 := (exec_mono dv h b true ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + dv.prices.int32⟩).1
        have h2 := (exec_mono dv h k t
          (exec dv true b ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + dv.prices.int32⟩)).1
        dsimp only at h1
        linarith [h.1]
      · have h1 := (exec_mono dv h a true ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + dv.prices.int32⟩).1
        have h2 := (exec_mono dv h k t
          (exec dv true a ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + dv.prices.int32⟩)).1
        dsimp only at h1
        linarith [h.1]

/-! ## The counterexample -/

/-- Unit 0's activation word `(0, 0)` moved into its ticket word `(0, 0)`: two nodes of text, within W1's bound. -/
def copyAct : CodeImage := ⟨fun _ => [], fun _ => .op (.mov (ticketAt 0 0 0) (actAt 0 0 0)) .done,
  fun _ => by norm_num [W1.Code.size]⟩

/-- What `copyAct` commits, on every oracle. -/
abbrev copyTranscript (dv : W1Device) (salt : List ℕ) (U : Workload) (act : ℕ → Codes) : Transcript :=
  readout U act (final dv salt copyAct U act []).mem

theorem copy_cost (dv : W1Device) (salt : List ℕ) (U : Workload) (act : ℕ → Codes) :
    (final dv salt copyAct U act []).cost = 0 := by
  simp [final, copyAct, exec, Instr.price, Instr.base, Instr.dynamic]

theorem copy_fetched (dv : W1Device) (salt : List ℕ) (U : Workload) (act : ℕ → Codes) :
    (final dv salt copyAct U act []).fetched = [actAt 0 0 0] := by
  simp [final, copyAct, exec, Instr.reads, preSalt, actAt]

theorem copy_ticket (dv : W1Device) (salt : List ℕ) (U : Workload) (act : ℕ → Codes) :
    (final dv salt copyAct U act []).mem (ticketAt 0 0 0) = w32 (act 0 0 0) := by
  simp [final, copyAct, exec, Instr.exec, init, actAt, ticketAt, w32]

open Classical in
/-- **W1 doesn't truncate**: given positive prices, a one-unit workload whose first activation word is 256 or more, and
oracles under which `copyAct`'s unit 0 is correct and isn't, no program of W1 is `copyAct` aborted at budget 0. -/
theorem w1_not_budgetAbort (dv : W1Device) (hdv : Positive dv) (P : Protocol Q R S) (salt : S → List ℕ)
    (U : Workload) (act : ℕ → Codes) (s : S) (H₁ H₂ : Q → R) (hN : U.layout.N = 1) (hm : 0 < (U.layout.shape 0).m)
    (hk : 0 < (U.layout.shape 0).k) (hbig : 256 ≤ w32 (act 0 0 0))
    (h₁ : 0 ∈ correctSet P U H₁ s (copyTranscript dv (salt s) U act))
    (h₂ : 0 ∉ correctSet P U H₂ s (copyTranscript dv (salt s) U act)) :
    ¬ BudgetAbortPearlC (w1 dv P salt) := by
  rintro ⟨-, hab⟩
  obtain ⟨p', hp'⟩ := hab copyAct U act 0 le_rfl
  have hfree : FreeFetch P U H₁ s (copyTranscript dv (salt s) U act) (actAt 0 0 0) := by
    refine Or.inl ⟨rfl, ?_, ?_, ?_⟩ <;> simp only [actAt, Nat.unpair_pair]
    · exact h₁
    · exact hm
    · exact hk
  have c1 : (w1 dv P salt).cost copyAct H₁ U act [] s ≤ 0 := by
    simp only [w1, copy_cost, copy_fetched, List.dedup_eq_self.mpr (List.nodup_singleton (actAt 0 0 0))]
    simp [hfree]
  have hrun := (hp' H₁ [] s).2.2 c1
  have hc2 := (hp' H₂ [] s).1
  simp only [w1] at hrun hc2
  have hK : correctSet P U H₂ s (readout U act (final dv (salt s) p' U act []).mem) = ∅ := by
    rw [hrun]
    ext u
    simp only [Finset.notMem_empty, iff_false]
    intro hu
    have hu0 : u < U.layout.N := Finset.mem_range.mp (Finset.mem_filter.mp hu).1
    rw [hN] at hu0
    obtain rfl : u = 0 := by omega
    exact h₂ hu
  have hnf : ∀ a, ¬ FreeFetch P U H₂ s (readout U act (final dv (salt s) p' U act []).mem) a := by
    intro a ha
    simp [FreeFetch, hK] at ha
  rw [List.filter_eq_self.mpr ?_] at hc2
  swap
  · intro a _
    exact decide_eq_true (hnf a)
  have hq : (final dv (salt s) p' U act []).cost +
      dv.prices.fetch * ((final dv (salt s) p' U act []).fetched.dedup.length : ℚ) ≤ 0 := by
    exact_mod_cast hc2
  have hc0 : 0 ≤ (final dv (salt s) p' U act []).cost := (exec_mono dv hdv _ false _).1
  have hlen : (final dv (salt s) p' U act []).fetched.dedup.length = 0 := by
    by_contra hne
    have h1 : (1 : ℚ) ≤ ((final dv (salt s) p' U act []).fetched.dedup.length : ℚ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.mpr hne
    have := mul_le_mul_of_nonneg_left h1 hdv.2.2.2.2.1.le
    linarith [hdv.2.2.2.2.1]
  have hfetch : (final dv (salt s) p' U act []).fetched = [] := by
    have hd := List.eq_nil_of_length_eq_zero hlen
    rcases hf : (final dv (salt s) p' U act []).fetched with _ | ⟨a, l⟩
    · rfl
    · have ha : a ∈ (final dv (salt s) p' U act []).fetched.dedup := List.mem_dedup.mpr (by simp [hf])
      simp [hd] at ha
  have hcost : (final dv (salt s) p' U act []).cost = 0 := by
    rw [hlen, Nat.cast_zero, mul_zero, add_zero] at hq
    linarith
  have hsmall : Small (final dv (salt s) p' U act []).mem :=
    small_exec_code dv hdv _ false _ hcost hfetch (small_init _ _ _ _ _)
  have key : ∀ M : Mem, (readout U act M 0).map (fun c => c.ticket 0 0) = some (M (ticketAt 0 0 0)) := by
    intro M
    simp [readout, hN]
  have e := congrArg (Option.map fun c : Commit => c.ticket 0 0) (congrFun hrun 0)
  rw [key, key, copy_ticket] at e
  have hlt := hsmall (ticketAt 0 0 0) (by simp [preSalt, ticketAt])
  rw [Option.some.inj e] at hlt
  omega

/-! ## A concrete instance -/

/-- One unit of shape `1 × 1 × 1`, weight 0, all zero. -/
def toyU : Workload := ⟨⟨1, fun _ => ⟨1, 1, 1⟩, fun _ => 0⟩, fun _ _ _ => 0⟩

/-- Activations of 256 everywhere. -/
def toyAct : ℕ → Codes := fun _ _ _ => 256

/-- A protocol whose ticket word `(0, 0)` is the unit's activation `(0, 0)`, whose useful word is 0, and whose cap holds
exactly when the oracle answers `true`. -/
def toyP : Protocol Unit Bool Unit where
  ticket := fun _ _ _ _ a i j => wordQ (if i = 0 ∧ j = 0 then a 0 0 else 0)
  useful := fun _ _ _ _ _ _ _ => wordQ 0
  actOK := fun _ _ => True
  weightOK := fun _ _ _ => True
  credit := fun _ _ _ _ _ => 0
  capOK := fun H _ _ _ _ => H () = true
  Wref := fun _ _ => 0

theorem toy_correct (dv : W1Device) (H : Unit → Bool) :
    0 ∈ correctSet toyP toyU H () (copyTranscript dv [] toyU toyAct) ↔ H () = true := by
  simp [correctSet, Correct, readout, toyP, toyU, toyAct, final, copyAct, exec, Instr.exec, init, actAt, ticketAt,
    usefulAt, Function.update_apply, w32]

theorem sm120_positive (sha fetch : ℚ) (hs : 0 < sha) (hf : 0 < fetch) : Positive (W1Device.sm120 sha fetch) := by
  refine ⟨by norm_num [W1Device.sm120, W1Device.sm120At, W1Prices.sm120],
    by norm_num [W1Device.sm120, W1Device.sm120At, W1Prices.sm120],
    by norm_num [W1Device.sm120, W1Device.sm120At, W1Prices.sm120],
    by simpa [W1Device.sm120, W1Device.sm120At, W1Prices.sm120] using hs,
    by simpa [W1Device.sm120, W1Device.sm120At, W1Prices.sm120] using hf, ?_⟩
  intro f hh
  simp only [W1Device.has, W1Device.sm120, W1Device.sm120At, sm120Frags, List.mem_cons, List.not_mem_nil, or_false,
    decide_eq_true_eq] at hh
  rcases hh with rfl | rfl | rfl
  · norm_num [W1Device.sm120, W1Device.sm120At, W1Device.depth, sm120E4m3, sm120E4m3K32, W1Prices.sm120]
  · norm_num [W1Device.sm120, W1Device.sm120At, W1Device.depth, PeelExact.hopperBf16K16, W1Prices.sm120]
  · norm_num [W1Device.sm120, W1Device.sm120At, W1Device.depth, Fp4.sm120Nvf4, W1Prices.sm120]

/-- **W1 on sm_120 doesn't truncate**, at any positive SHA-512 and fetch prices. -/
theorem sm120_not_budgetAbort (sha fetch : ℚ) (hs : 0 < sha) (hf : 0 < fetch) :
    ¬ BudgetAbortPearlC (w1 (W1Device.sm120 sha fetch) toyP (fun _ => [])) :=
  w1_not_budgetAbort _ (sm120_positive sha fetch hs hf) toyP _ toyU toyAct () (fun _ => true) (fun _ => false) rfl
    (by decide) (by decide) (by decide) ((toy_correct _ _).mpr rfl) (fun h => by simpa using (toy_correct _ _).mp h)

end Pouw.PearlC.W1BudgetAbort
