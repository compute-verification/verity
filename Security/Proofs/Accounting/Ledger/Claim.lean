import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Proofs.Pouw.Prob
import Proofs.Accounting.Ledger.Basic
import Definitions.Accounting.Ledger.Pins

/-!
# Claims, monotonicity and composition of pins

**The claim** (`Experiment.claim`): if the protocol run alongside a task pins `π + R` and the region's capacity is `c`
with `c − π < R`, the protocol accepts with the task run there with probability at most the pin's error, in either
reading. With additivity, the pin alongside comes from the protocol's pin and the task's requirement
(`Experiment.claim_of_additive`), and a capacity per tick bounds the meter of a region (`Experiment.read_le_capacity`).

**Monotonicity.** A pin weakens in its bound and its error (`Experiment.PinAs.mono`), and carries to any meter and region
whose reading is at least its own on every run (`Experiment.PinAs.of_le`): a larger meter (`PinAs.mono_meter`), more ticks
(`PinAs.mono_ticks`), and for a meter that reads no crossing a larger part (`PinAs.mono_part`). A per-run pin gives its
budget form (`Experiment.Pin.budget`).

**Composition.** Per-run pins on disjoint regions add, bounds and errors, by the union bound: over disjoint ticks for any
meter (`Experiment.Pin.union_ticks`), over disjoint parts for a meter that reads no crossing
(`Experiment.Pin.union_part`), and for two meters of one region (`Experiment.Pin.add_meter`). Pins in budget form don't
(`budget_not_union_ticks`): each bounds its region's worst case over the coins, and the two worst cases may sit on
different coins. Lower pins on crossings don't compose over parts in either reading: the bits moving between the parts
cross neither boundary of their union (`ledger_union_part_crossOut_le`).

The rate audit's claim (`rate_claim`) and PoUS's graded pin giving its peak-graded one (`pousHeldPeak_of_graded`) are
instances. An upper pin on content excludes any decodable set of accepted secrets larger than its bound
(`ContentAtMost.claim`).
-/

namespace Accounting.Ledger

open Pouw

namespace Experiment

variable {Λ G Adv Ω : Type} [Fintype Λ] [DecidableEq Λ] [Fintype G] {E : Experiment Λ G Adv Ω}

theorem Requires.mono {P : Task Adv Ω} {ρ : Region Λ} {m : Meter G} {R R' : ℝ} (h : E.Requires P ρ m R)
    (hR : R' ≤ R) : E.Requires P ρ m R' :=
  fun a ω hP => hR.trans (h a ω hP)

theorem Requires.mono_task {P P' : Task Adv Ω} {ρ : Region Λ} {m : Meter G} {R : ℝ} (h : E.Requires P ρ m R)
    (hP : ∀ a ω, P' a ω → P a ω) : E.Requires P' ρ m R :=
  fun a ω hP' => h a ω (hP a ω hP')

/-- A capacity of `c` per tick bounds a meter of a region of `n` ticks by `n` times the meter of `c`. -/
theorem read_le_capacity {ρ : Region Λ} {c : Ledger G} (hc : Capacity (fun x : Adv × Ω => E.circuit x.1 x.2) ρ.part c)
    (m : Meter G) (a : Adv) (ω : Ω) : (E.read ρ m a ω : ℝ) ≤ ρ.ticks.card * m.eval c := by
  exact_mod_cast eval_le_capacity hc m (a, ω) ρ.ticks

variable [Fintype Ω]

section Mono

variable {ρ ρ' : Region Λ} {m m' : Meter G} {π π' ε ε' : ℝ}

theorem Pin.mono (h : E.Pin ρ m π ε) (hπ : π' ≤ π) (hε : ε ≤ ε') : E.Pin ρ m π' ε' :=
  fun a => (pr_mono fun _ hω => ⟨hω.1, hω.2.trans_le hπ⟩).trans ((h a).trans hε)

theorem PinBudget.mono (h : E.PinBudget ρ m π ε) (hπ : π' ≤ π) (hε : ε ≤ ε') : E.PinBudget ρ m π' ε' :=
  fun a B hB hle => (pr_mono fun _ hω => ⟨hω.1, hω.2.trans_le hπ⟩).trans ((h a B hB hle).trans hε)

theorem PinAs.mono {r : Reading} (h : E.PinAs r ρ m π ε) (hπ : π' ≤ π) (hε : ε ≤ ε') : E.PinAs r ρ m π' ε' := by
  cases r
  · exact Pin.mono h hπ hε
  · exact PinBudget.mono h hπ hε

/-- **A per-run pin gives its budget form**: an adversary whose meter is at most `B` on every coin and accepts with
`B < π` accepts with its meter below `π`. -/
theorem Pin.budget (h : E.Pin ρ m π ε) : E.PinBudget ρ m π ε :=
  fun a _ _ hB => (pr_mono fun ω hω => ⟨hω.1, (hB ω).trans_lt hω.2⟩).trans (h a)

/-- A pin carries to any meter and region whose reading is at least its own on every run. -/
theorem PinAs.of_le {r : Reading} (hle : ∀ a ω, E.read ρ m a ω ≤ E.read ρ' m' a ω) (h : E.PinAs r ρ m π ε) :
    E.PinAs r ρ' m' π ε := by
  cases r
  · exact fun a => (pr_mono fun ω hω => ⟨hω.1, lt_of_le_of_lt (by exact_mod_cast hle a ω) hω.2⟩).trans (h a)
  · exact fun a B hB hB' => h a B hB fun ω => le_trans (by exact_mod_cast hle a ω) (hB' ω)

theorem PinAs.mono_meter {r : Reading} (hm : m ≤ m') (h : E.PinAs r ρ m π ε) : E.PinAs r ρ m' π ε :=
  h.of_le fun _ _ => Meter.eval_mono_meter hm _

theorem PinAs.mono_ticks {r : Reading} {p : Finset Λ} {T T' : Finset ℕ} (hT : T ⊆ T') (h : E.PinAs r ⟨p, T⟩ m π ε) :
    E.PinAs r ⟨p, T'⟩ m π ε :=
  h.of_le fun a ω => Meter.eval_mono m (ledger_mono_ticks (E.circuit a ω) p hT)

theorem PinAs.mono_part {r : Reading} (hm : m.NoCrossing) {p q : Finset Λ} {T : Finset ℕ} (hpq : p ⊆ q)
    (h : E.PinAs r ⟨p, T⟩ m π ε) : E.PinAs r ⟨q, T⟩ m π ε :=
  h.of_le fun a ω => eval_mono_part (E.circuit a ω) hm hpq T

end Mono

section Claim

variable {P : Task Adv Ω} {ρ : Region Λ} {m : Meter G} {π R ε c : ℝ}

/-- **The claim.** If the protocol run alongside task `P` pins `π + R` of meter `m` in region `ρ`, and the region's
meter is at most `c` on every run with `c − π < R`, the protocol accepts with `P` run with probability at most the pin's
error, in either reading. -/
theorem claim (r : Reading) (hpin : (E.alongside P).PinAs r ρ m (π + R) ε) (hc : 0 ≤ c)
    (hcap : ∀ a ω, (E.read ρ m a ω : ℝ) ≤ c) (hgap : c - π < R) (a : Adv) :
    pr (fun ω => E.accepts a ω ∧ P a ω) ≤ ε := by
  have hlt : c < π + R := by linarith
  cases r
  · exact (pr_mono fun ω hω => ⟨hω, (hcap a ω).trans_lt hlt⟩).trans (hpin a)
  · exact (pr_mono fun _ hω => ⟨hω, hlt⟩).trans (hpin a c hc (hcap a))

/-- **The claim from additivity**: the protocol's pin `π`, task `P`'s requirement `R` and capacity `c` with
`c − π < R` exclude `P`, except with the pin's error. -/
theorem claim_of_additive {r : Reading} {Indep : Task Adv Ω → Prop} (hadd : E.Additive r Indep ρ m) (hP : Indep P)
    (hpin : E.PinAs r ρ m π ε) (hreq : E.Requires P ρ m R) (hc : 0 ≤ c) (hcap : ∀ a ω, (E.read ρ m a ω : ℝ) ≤ c)
    (hgap : c - π < R) (a : Adv) : pr (fun ω => E.accepts a ω ∧ P a ω) ≤ ε :=
  claim r (hadd P π R ε hP hpin hreq) hc hcap hgap a

/-- The claim from additivity with a capacity `c` per tick: the region's capacity is its number of ticks times the
meter of `c`. -/
theorem claim_of_capacity {r : Reading} {Indep : Task Adv Ω → Prop} (hadd : E.Additive r Indep ρ m) (hP : Indep P)
    (hpin : E.PinAs r ρ m π ε) (hreq : E.Requires P ρ m R) {c : Ledger G}
    (hc : Capacity (fun x : Adv × Ω => E.circuit x.1 x.2) ρ.part c) (hgap : ρ.ticks.card * m.eval c - π < R)
    (a : Adv) : pr (fun ω => E.accepts a ω ∧ P a ω) ≤ ε :=
  claim_of_additive hadd hP hpin hreq (by positivity) (read_le_capacity hc m) hgap a

end Claim

section Compose

variable {ρ ρ' ρ'' : Region Λ} {m m' m'' : Meter G} {π π' ε ε' : ℝ}

/-- Per-run pins add over two readings whose sum is at most a third's on every run, bounds and errors. -/
theorem Pin.add_of_le (hle : ∀ a ω, E.read ρ m a ω + E.read ρ' m' a ω ≤ E.read ρ'' m'' a ω) (h : E.Pin ρ m π ε)
    (h' : E.Pin ρ' m' π' ε') : E.Pin ρ'' m'' (π + π') (ε + ε') := by
  intro a
  refine (pr_mono fun ω hω => ?_).trans ((pr_or_le _ _).trans (add_le_add (h a) (h' a)))
  have hs : (E.read ρ m a ω : ℝ) + E.read ρ' m' a ω ≤ E.read ρ'' m'' a ω := by exact_mod_cast hle a ω
  rcases lt_or_ge (E.read ρ m a ω : ℝ) π with h1 | h1
  · exact Or.inl ⟨hω.1, h1⟩
  · exact Or.inr ⟨hω.1, by linarith [hω.2]⟩

/-- **Composition over time**: per-run pins on one part over disjoint sets of ticks add. -/
theorem Pin.union_ticks {p : Finset Λ} {T T' : Finset ℕ} (hd : Disjoint T T') (h : E.Pin ⟨p, T⟩ m π ε)
    (h' : E.Pin ⟨p, T'⟩ m π' ε') : E.Pin ⟨p, T ∪ T'⟩ m (π + π') (ε + ε') :=
  Pin.add_of_le (fun a ω => by simp only [read, ledger_union_ticks _ p hd, Meter.eval_add, le_refl]) h h'

/-- **Composition over space**: per-run pins of a meter that reads no crossing on disjoint parts add. -/
theorem Pin.union_part (hm : m.NoCrossing) {p q : Finset Λ} {T : Finset ℕ} (hd : Disjoint p q)
    (h : E.Pin ⟨p, T⟩ m π ε) (h' : E.Pin ⟨q, T⟩ m π' ε') : E.Pin ⟨p ∪ q, T⟩ m (π + π') (ε + ε') :=
  Pin.add_of_le (fun a ω => by simp only [read, eval_union_part _ hm hd, le_refl]) h h'

/-- Per-run pins of two meters of one region add. -/
theorem Pin.add_meter (h : E.Pin ρ m π ε) (h' : E.Pin ρ m' π' ε') : E.Pin ρ (m + m') (π + π') (ε + ε') :=
  Pin.add_of_le (fun a ω => by simp only [read, Meter.eval_add_meter, le_refl]) h h'

end Compose

end Experiment

open Experiment

/-- **Budget pins don't compose over time.** An experiment with one adversary, one leaf and a coin: the leaf holds a
bit at tick 0 on one coin and at tick 1 on the other, and the protocol always accepts. Each tick's bits held are
pinned at `1` with error `0` in budget form, since each tick's worst case over the coins is a bit, but the two ticks'
are not pinned at `2` with error `0`: their sum is one bit on every coin. -/
theorem budget_not_union_ticks : ∃ E : Experiment Unit Unit Unit Bool,
    E.PinBudget ⟨Finset.univ, {0}⟩ (Meter.entry .held) 1 0 ∧
    E.PinBudget ⟨Finset.univ, {1}⟩ (Meter.entry .held) 1 0 ∧
    ¬ E.PinBudget ⟨Finset.univ, {0} ∪ {1}⟩ (Meter.entry .held) (1 + 1) 0 := by
  let E : Experiment Unit Unit Unit Bool :=
    ⟨fun _ ω => ⟨fun _ _ _ => 0, fun _ t => if t = (if ω then 0 else 1) then 1 else 0, fun _ _ _ => 0⟩,
      fun _ _ => True⟩
  have r0 : ∀ ω, E.read ⟨Finset.univ, {0}⟩ (Meter.entry .held) () ω = if ω then 1 else 0 := by
    intro ω; cases ω <;> simp [E, Experiment.read, Meter.eval_entry, Annotation.ledger]
  have r1 : ∀ ω, E.read ⟨Finset.univ, {1}⟩ (Meter.entry .held) () ω = if ω then 0 else 1 := by
    intro ω; cases ω <;> simp [E, Experiment.read, Meter.eval_entry, Annotation.ledger]
  have r01 : ∀ ω, E.read ⟨Finset.univ, {0} ∪ {1}⟩ (Meter.entry .held) () ω = 1 := by
    intro ω; cases ω <;> simp [E, Experiment.read, Meter.eval_entry, Annotation.ledger]
  refine ⟨E, fun _ B _ hB => ?_, fun _ B _ hB => ?_, fun h => ?_⟩
  · have := hB true
    rw [r0] at this
    exact (pr_eq_zero fun _ hω => absurd hω.2 (not_lt.mpr (by simpa using this))).le
  · have := hB false
    rw [r1] at this
    exact (pr_eq_zero fun _ hω => absurd hω.2 (not_lt.mpr (by simpa using this))).le
  · have h1 := h () 1 zero_le_one fun ω => by rw [r01]; simp
    rw [pr_eq_one fun _ => ⟨trivial, by norm_num⟩] at h1
    norm_num at h1

variable {Λ G Adv Ω : Type} [Fintype Λ] [DecidableEq Λ] [Fintype G] [DecidableEq G] [Fintype Ω]
  {E : Experiment Λ G Adv Ω}

/-- **PoUS's graded pin gives its peak-graded pin** over a nonempty set of challenge ticks: bits held at most `B` at
every challenge make at most `B·|C|` bit-ticks over them. -/
theorem pousHeldPeak_of_graded {dev : Finset Λ} {C : Finset ℕ} {S err : ℝ} (hC : C.Nonempty)
    (h : E.PousHeldGraded dev C S err) : E.PousHeldPeak dev C S err := by
  intro a B hB hle
  have hsum : ∀ ω, (E.read ⟨dev, C⟩ (Meter.entry .held) a ω : ℝ) ≤ B * C.card := by
    intro ω
    have hs : E.read ⟨dev, C⟩ (Meter.entry .held) a ω = ∑ c ∈ C, E.read ⟨dev, {c}⟩ (Meter.entry .held) a ω := by
      simp only [Experiment.read]
      rw [ledger_eq_sum_ticks, Meter.eval_sum]
    rw [hs, Nat.cast_sum]
    calc ∑ c ∈ C, (E.read ⟨dev, {c}⟩ (Meter.entry .held) a ω : ℝ) ≤ ∑ _c ∈ C, B :=
          Finset.sum_le_sum fun c hc => hle ω c hc
      _ = B * C.card := by rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
  have hcard : (0 : ℝ) < C.card := by exact_mod_cast hC.card_pos
  exact (pr_mono fun _ hω => ⟨hω.1, mul_lt_mul_of_pos_right hω.2 hcard⟩).trans
    (h a (B * C.card) (by positivity) hsum)

/-- **The rate audit's claim.** The rate audit pins device memory's crossing plus `b` bits a rebuilt word at
`(1 − μ)·nE`, with error `errR`. Rebuilding `k` words requires `k·c` work on the cores (`RebuildGates`); with the
declared work's pin `πw` there, compute additivity, and the cores' capacity `F` over the window with `F − πw < k·c`, the
device rebuilt fewer than `k` words except with error `errW`. So it accepts with the crossing below `(1 − μ)·nE − b·k`
with probability at most `errR + errW`. -/
theorem rate_claim {mem cores : Finset Λ} {w : Window} {rebuilt : Adv → Ω → ℕ} {b nE : ℕ} {μ errR : ℝ}
    (hrate : E.RateRead mem w rebuilt b nE μ errR) {p : G → ℕ} {c : ℝ}
    (hreb : E.RebuildGates cores w rebuilt p c) (k : ℕ) {r : Reading} {Indep : Task Adv Ω → Prop}
    (hadd : E.ComputeAdditive r Indep (Region.ofWindow cores w) p) (hind : Indep fun a ω => k ≤ rebuilt a ω)
    {πw errW F : ℝ} (hwork : E.PinAs r (Region.ofWindow cores w) (Meter.priced p) πw errW) (hF : 0 ≤ F)
    (hcap : ∀ a ω, (E.read (Region.ofWindow cores w) (Meter.priced p) a ω : ℝ) ≤ F) (hgap : F - πw < k * c)
    (a : Adv) :
    pr (fun ω => E.accepts a ω ∧
      (E.read (Region.ofWindow mem w) (Meter.entry .crossOut) a ω : ℝ) + b * k < (1 - μ) * nE) ≤ errR + errW := by
  refine (pr_mono fun ω hω => ?_).trans ((pr_or_le _ _).trans
    (add_le_add (hrate a) (claim_of_additive hadd hind hwork (hreb k) hF hcap hgap a)))
  by_cases hk : k ≤ rebuilt a ω
  · exact Or.inr ⟨hω.1, hk⟩
  · have : (b : ℝ) * rebuilt a ω ≤ b * k := by gcongr; exact_mod_cast (Nat.lt_of_not_le hk).le
    exact Or.inl ⟨hω.1, by linarith [hω.2]⟩

/-- **The content claim**: an upper pin of `N` on content excludes any set of more than `N` accepted secrets that a
decoder recovers from the wire. -/
theorem ContentAtMost.claim {Z W : Type} {accepted : Z → Prop} {wire : Z → W} {N : ℕ∞}
    (h : ContentAtMost accepted wire N) {Zs : Set Z} (hacc : ∀ z ∈ Zs, accepted z) (dec : W → Z)
    (hdec : ∀ z ∈ Zs, dec (wire z) = z) (hN : N < Zs.encard) : False :=
  absurd (h Zs hacc dec hdec) (not_le.mpr hN)

end Accounting.Ledger
