import Mathlib
import Models.Accounting.Guarantees.Stacking
import Proofs.Accounting.RateAudit.DeployedSketchSum
import Proofs.Accounting.RateAudit.DeployedRelay
import Proofs.Accounting.Requirement

/-!
# Stacking: the carrier lemma and the stacked session

`Accounting.SecurityProofs.CarrierBits` counts a circuit's values: a step of `x` bits splits the stores into at most
`2^x` classes, in each of which the rest of the provenance has `x` bits fewer, so the values take at most `2^n`
contents when every run has at most `n` input and encoded bits (`Provenance.card_image_le`). A carrier's values are its
sectors at every store, which take every value, so its sectors are at most `n` bits (`Carrier.card_le`). Under
`EncodeChainHardness` an encode of `x` bits costs at least `x/(8·bytes)` words' `E` budgeted gates
(`encodeBitsCost_ge`, with `⊤` where the gate model has no circuit), so a gate budget bounds the encoded bits
(`Provenance.encoded_mul_le`).

`KeyBinding` moves an event of the store made with one choice and the session's coins to a uniform store at a cost of
`δ` (`KeyBinding.pr_le`), fiber by fiber over the coins, and an event of the store made with the writers' choices to
`|Ch|` times that (`KeyBinding.pr_le_advice`), by a union over the choices. The session's draws hold at every store, so
only its unpaid words' event pays for the choices.

`Accounting.SecurityProofs.StackedSessionConservationSketchDeployed` charges each carrier's sectors in the pass's
rebuilds and applies the deployed guarantee at rebuilt bits `X + C`.

`Accounting.SecurityProofs.StackedSessionConservationRelay` simulates each pass device by one that reruns its carrier's
circuit first, reading every step's bits over the side link at odd addresses (`PIChip.walk`, within `C` bits), and then
runs the device with its carrier reads answered from the circuit's values (`PIChip.answer`). The simulating device
(`PIDevice.stack`) has the device's state, memory, receipts, tier reads and compute, and side link `L + C`, so
`RemoteSessionConservationRelay` bounds its session, which is the stacked session's event.

`Accounting.SecurityProofs.StackedRequirementSketchDeployed` and `Accounting.SecurityProofs.StackedRequirementRelay`
bound a declared circuit's held and read bits over at most `k` steps by the store's `k·N·B` when the store and the plans
are its own (`Requirement.cap_of_matches`, from `RequirementHeld` and `RequirementMoved` with no block held), so a
shortfall against the requirement is one against the store.
-/

namespace Accounting

/-- **Under the gate model's conjecture, `x` bits of a block's encoding cost `x/(8·bytes)` words' `E` budgeted
gates**: every circuit of `Gates.encodeCosts` at the `⌈x/(8·bytes)⌉` words those bits fill has at least that many words'
`E` budgeted gates, and `encodeBitsCost` is `⊤` where there is none. -/
theorem encodeBitsCost_ge {F : Gates.Field} {k f w E εnum εden : ℕ}
    (hE : Assumptions.EncodeChainHardness F k f w E εnum εden) (hbytes : 0 < F.bytes) (x : ℕ) :
    ((x * E : ℕ) : ℕ∞) ≤ (8 * F.bytes : ℕ) * encodeBitsCost F k f w εnum εden x := by
  have hn : 0 < F.n - 1 := by
    have h := hbytes
    simp only [Gates.Field.bytes] at h
    omega
  have hwb : 0 < 8 * F.bytes := by omega
  set q := (x + 8 * F.bytes - 1) / (8 * F.bytes) with hq
  have hlt := Nat.lt_mul_div_succ (x + 8 * F.bytes - 1) hwb
  rw [← hq, mul_add, mul_one] at hlt
  have hxq : x ≤ 8 * F.bytes * q := by omega
  have hcost : ((q * E : ℕ) : ℕ∞) ≤ encodeBitsCost F k f w εnum εden x := by
    refine le_iInf₂ fun c hc => ?_
    obtain ⟨ι, C, m, xs, words, out, hlen, hnd, hp, hres, hrun, hin, hcomp, rfl⟩ := hc
    have h := hE C m xs words out hres hin hnd hp hcomp
    rw [hrun, zero_mul, add_zero, hlen] at h
    exact_mod_cast Nat.le_of_mul_le_mul_right h hn
  calc ((x * E : ℕ) : ℕ∞) ≤ ((8 * F.bytes * q * E : ℕ) : ℕ∞) := by
        exact_mod_cast Nat.mul_le_mul_right E hxq
    _ = (8 * F.bytes : ℕ) * ((q * E : ℕ) : ℕ∞) := by push_cast; ring
    _ ≤ (8 * F.bytes : ℕ) * encodeBitsCost F k f w εnum εden x := by gcongr

end Accounting

namespace Accounting.RateSketch

open Pous (Bits)

namespace Provenance

variable {N m : ℕ} {α : Type}

/-- One step of `x` bits: the stores split by the step's answer `g`, and the values in each class are at most
`2^(n − x)`, so at most `2^n` in all. -/
theorem card_image_step [DecidableEq α] {x n : ℕ} (T : Finset (SketchStore N m)) (g : SketchStore N m → Bits x)
    (F : Bits x → SketchStore N m → α) (hx : x ≤ n)
    (hF : ∀ v, ((T.filter fun E => g E = v).image (F v)).card ≤ 2 ^ (n - x)) :
    (T.image fun E => F (g E) E).card ≤ 2 ^ n := by
  calc (T.image fun E => F (g E) E).card
      ≤ (Finset.univ.biUnion fun v => (T.filter fun E => g E = v).image (F v)).card := by
        apply Finset.card_le_card
        intro a ha
        obtain ⟨E, hE, rfl⟩ := Finset.mem_image.mp ha
        exact Finset.mem_biUnion.mpr ⟨g E, Finset.mem_univ _,
          Finset.mem_image.mpr ⟨E, Finset.mem_filter.mpr ⟨hE, rfl⟩, rfl⟩⟩
    _ ≤ ∑ v : Bits x, ((T.filter fun E => g E = v).image (F v)).card := Finset.card_biUnion_le
    _ ≤ ∑ _v : Bits x, 2 ^ (n - x) := Finset.sum_le_sum fun v _ => hF v
    _ = 2 ^ n := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul]
        simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
        rw [← pow_add, Nat.add_sub_cancel' hx]

/-- **The circuit's values take at most `2^n` contents** over a set of stores at each of which it has at most `n`
input and encoded bits. -/
theorem card_image_le [DecidableEq α] (c : Provenance N m α) :
    ∀ (T : Finset (SketchStore N m)) (n : ℕ), (∀ E ∈ T, c.inputs E + c.encoded E ≤ n) →
      (T.image fun E => c.run E).card ≤ 2 ^ n := by
  induction c with
  | done a =>
    intro T n _
    calc (T.image fun E => (done a : Provenance N m α).run E).card ≤ ({a} : Finset α).card := by
          apply Finset.card_le_card
          intro b hb
          obtain ⟨E, _, rfl⟩ := Finset.mem_image.mp hb
          exact Finset.mem_singleton_self a
      _ = 1 := Finset.card_singleton a
      _ ≤ 2 ^ n := Nat.one_le_two_pow
  | input x f k ih =>
    intro T n hT
    rcases T.eq_empty_or_nonempty with rfl | ⟨E₀, hE₀⟩
    · simp
    have hx : x ≤ n := by
      have := hT E₀ hE₀
      simp only [inputs, encoded] at this
      omega
    exact card_image_step T f (fun v E => (k v).run E) hx fun v =>
      ih v _ _ fun E hE => by
        obtain ⟨hET, rfl⟩ := Finset.mem_filter.mp hE
        have := hT E hET
        simp only [inputs, encoded] at this
        omega
  | encode t x _ f k ih =>
    intro T n hT
    rcases T.eq_empty_or_nonempty with rfl | ⟨E₀, hE₀⟩
    · simp
    have hx : x ≤ n := by
      have := hT E₀ hE₀
      simp only [inputs, encoded] at this
      omega
    exact card_image_step T (fun E => f fun u => E (t, u)) (fun v E => (k v).run E) hx fun v =>
      ih v _ _ fun E hE => by
        obtain ⟨hET, rfl⟩ := Finset.mem_filter.mp hE
        have := hT E hET
        simp only [inputs, encoded] at this
        omega

/-- **A gate budget bounds the encoded bits**: under `EncodeChainHardness` at `E` budgeted gates a word of `8·bytes`
bits, the encoded bits times `E` are at most `8·bytes` times the encodes' budgeted gates. -/
theorem encoded_mul_le {F : Gates.Field} {kr fan wid E εnum εden : ℕ}
    (hE : Assumptions.EncodeChainHardness F kr fan wid E εnum εden) (hbytes : 0 < F.bytes) (c : Provenance N m α)
    (Es : SketchStore N m) :
    ((c.encoded Es * E : ℕ) : ℕ∞) ≤ (8 * F.bytes : ℕ) * c.gates (encodeBitsCost F kr fan wid εnum εden) Es := by
  induction c with
  | done a => simp [encoded, gates]
  | input x g k ih => simpa [encoded, gates] using ih (g Es)
  | encode t x hx g k ih =>
    simp only [encoded, gates]
    rw [add_mul, Nat.cast_add, mul_add]
    exact add_le_add (ih _) (encodeBitsCost_ge hE hbytes x)

end Provenance

namespace Carrier

variable {N m : ℕ}

/-- **A carrier's sectors are at most its input and encoded bits.** Its values are its sectors at every store, and
those take every value, `2^(256·|carried|)` of them. -/
theorem card_le (K : Carrier N m) {n : ℕ} (hK : ∀ E, K.provenance.inputs E + K.provenance.encoded E ≤ n) :
    K.carried.card * 256 ≤ n := by
  classical
  have h := K.provenance.card_image_le Finset.univ n fun E _ => hK E
  have himg : (Finset.univ.image fun E => K.provenance.run E) = Finset.univ := by
    refine Finset.eq_univ_of_forall fun v => Finset.mem_image.mpr ?_
    refine ⟨fun p => if hp : p ∈ K.carried then v ⟨p, hp⟩ else fun _ => 0, Finset.mem_univ _, ?_⟩
    rw [K.holds]
    funext p
    simp [p.2]
  rw [himg, Finset.card_univ, Fintype.card_fun, Fintype.card_coe] at h
  have hs : Fintype.card Sector = 2 ^ 256 := by simp [Sector]
  rw [hs, ← pow_mul, mul_comm] at h
  exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h

/-- A carrier whose circuit takes in at most `I` bits and spends at most `R` budgeted gates on its encodes carries at
most `I + 8·bytes·R/E` bits, under `EncodeChainHardness` at `E` budgeted gates a word of `8·bytes` bits. -/
theorem card_le_of_budget (K : Carrier N m) {F : Gates.Field} {kr fan wid E εnum εden I R : ℕ}
    (hE : Assumptions.EncodeChainHardness F kr fan wid E εnum εden) (hbytes : 0 < F.bytes) (hE0 : 0 < E)
    (hK : ∀ Es, K.provenance.inputs Es ≤ I ∧
      K.provenance.gates (encodeBitsCost F kr fan wid εnum εden) Es ≤ (R : ℕ)) :
    K.carried.card * 256 ≤ I + 8 * F.bytes * R / E := by
  refine K.card_le fun Es => ?_
  have h2 : K.provenance.encoded Es ≤ 8 * F.bytes * R / E := by
    rw [Nat.le_div_iff_mul_le hE0]
    have h := (Provenance.encoded_mul_le hE hbytes K.provenance Es).trans (mul_le_mul_right (hK Es).2 _)
    exact_mod_cast h
  have h1 := (hK Es).1
  omega

end Carrier

end Accounting.RateSketch

namespace Accounting.Assumptions

open Pous (pr)
open scoped ENNReal

/-- A probability over a pair is the average over the second coordinate of its fibers' probabilities. -/
theorem pr_prod_eq_sum_div {X Y : Type} [Fintype X] [Fintype Y] (Q : X × Y → Prop) :
    pr Q = (∑ y, pr fun x => Q (x, y)) / Fintype.card Y := by
  classical
  unfold pr
  have e : (Finset.univ.filter Q).card = ∑ y, (Finset.univ.filter fun x => Q (x, y)).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type_right]
    simp only [Finset.card_filter]
  rw [e, Nat.cast_sum, Fintype.card_prod, Nat.cast_mul, ← ENNReal.sum_div]
  simp only [ENNReal.div_eq_inv_mul]
  rw [ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  ring

/-- **Key binding moves an event to a uniform store, at a cost of `δ`**: for each choice `α`, an event of the store
made with `α` and of independent coins `C` is at most as likely as on a uniform store, plus `δ`. -/
theorem KeyBinding.pr_le {St Ω Ch C : Type} [Fintype St] [Fintype Ω] [Fintype C] {δ : ℝ≥0∞} {view : Ch → Ω → St}
    (h : KeyBinding δ view) (α : Ch) (P : St → C → Prop) :
    pr (fun x : Ω × C => P (view α x.1) x.2) ≤ pr (fun y : St × C => P y.1 y.2) + δ := by
  rw [pr_prod_eq_sum_div (fun x : Ω × C => P (view α x.1) x.2), pr_prod_eq_sum_div (fun y : St × C => P y.1 y.2)]
  calc (∑ c, pr fun ω => P (view α ω) c) / Fintype.card C
      ≤ (∑ c, (pr (fun E => P E c) + δ)) / Fintype.card C :=
        ENNReal.div_le_div_right (Finset.sum_le_sum fun c _ => h α {E | P E c}) _
    _ = (∑ c, pr fun E => P E c) / Fintype.card C + Fintype.card C * δ / Fintype.card C := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ENNReal.add_div]
    _ ≤ (∑ c, pr fun E => P E c) / Fintype.card C + δ :=
        add_le_add le_rfl (ENNReal.div_le_of_le_mul (le_of_eq (mul_comm _ _)))

/-- **The writers' choices are advice**: an event of the store made with the choices `choice ω` and of independent
coins `C` is at most `|Ch|` times as likely as on a uniform store, plus `δ`. -/
theorem KeyBinding.pr_le_advice {St Ω Ch C : Type} [Fintype St] [Fintype Ω] [Fintype Ch] [Fintype C] {δ : ℝ≥0∞}
    {view : Ch → Ω → St} (h : KeyBinding δ view) (choice : Ω → Ch) (P : St → C → Prop) :
    pr (fun x : Ω × C => P (view (choice x.1) x.1) x.2) ≤
      Fintype.card Ch * (pr (fun y : St × C => P y.1 y.2) + δ) :=
  calc pr (fun x : Ω × C => P (view (choice x.1) x.1) x.2)
      ≤ pr (fun x : Ω × C => ∃ α, P (view α x.1) x.2) := Bandwidth.pr_mono fun _ hx => ⟨_, hx⟩
    _ ≤ ∑ α, pr (fun x : Ω × C => P (view α x.1) x.2) :=
        Bandwidth.pr_exists_le fun α (x : Ω × C) => P (view α x.1) x.2
    _ ≤ ∑ _α : Ch, (pr (fun y : St × C => P y.1 y.2) + δ) := Finset.sum_le_sum fun α _ => h.pr_le α P
    _ = Fintype.card Ch * (pr (fun y : St × C => P y.1 y.2) + δ) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

end Accounting.Assumptions

namespace Accounting.Rate

open Pous (Bits pr)

namespace PassProvenance

variable {N b h ρ : ℕ} {α : Type}

/-- The provenance's step answers at store `E`, stream `R` and oracle `O`, one after another: bit `i` of the run's
bits, and `false` past them. -/
def transcript (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : PassProvenance N b h ρ α → ℕ → Bool
  | done _, _ => false
  | input x f k, i => if hi : i < x then f E R O ⟨i, hi⟩ else (k (f E R O)).transcript E R O (i - x)
  | encode j x _ f k, i => if hi : i < x then f (E j) ⟨i, hi⟩ else (k (f (E j))).transcript E R O (i - x)

/-- **A gate budget bounds the encoded bits**: under `EncodeChainHardness` at `E` budgeted gates a word of `8·bytes`
bits, the encoded bits times `E` are at most `8·bytes` times the encodes' budgeted gates. -/
theorem encoded_mul_le {F : Gates.Field} {kr fan wid E εnum εden : ℕ}
    (hE : Assumptions.EncodeChainHardness F kr fan wid E εnum εden) (hbytes : 0 < F.bytes)
    (c : PassProvenance N b h ρ α) (Es : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    ((c.encoded Es R O * E : ℕ) : ℕ∞) ≤
      (8 * F.bytes : ℕ) * c.gates (encodeBitsCost F kr fan wid εnum εden) Es R O := by
  induction c with
  | done a => simp [encoded, gates]
  | input x g k ih => simpa [encoded, gates] using ih (g Es R O)
  | encode j x hx g k ih =>
    simp only [encoded, gates]
    rw [add_mul, Nat.cast_add, mul_add]
    exact add_le_add (ih _) (encodeBitsCost_ge hE hbytes x)

end PassProvenance

namespace PIChip

variable {N b h ρ : ℕ} {α β : Type}

/-- Read `n` bits from the side sources at the odd addresses `2·(o + i) + 1`, then continue with them. -/
def readOdd : (n : ℕ) → ℕ → (Bits n → PIChip N b h ρ α) → PIChip N b h ρ α
  | 0, _, k => k Fin.elim0
  | n + 1, o, k => side (2 * o + 1) (fun γ => readOdd n (o + 1) (fun z => k (Fin.cons γ z)))

theorem cons_side (w : IWorld N b h ρ) (v : List Bool) {n : ℕ} (o : ℕ) :
    (Fin.cons (w.side v (2 * o + 1)) (fun i : Fin n => w.side v (2 * (o + 1 + i) + 1)) : Bits (n + 1)) =
      fun i : Fin (n + 1) => w.side v (2 * (o + i) + 1) := by
  funext i
  refine Fin.cases ?_ (fun j => ?_) i
  · simp
  · simp only [Fin.cons_succ, Fin.val_succ]
    congr 1
    omega

theorem elim0_side (w : IWorld N b h ρ) (v : List Bool) (o : ℕ) :
    (Fin.elim0 : Bits 0) = fun i : Fin 0 => w.side v (2 * (o + i) + 1) :=
  funext fun i => i.elim0

theorem readOdd_run (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n o : ℕ) (k : Bits n → PIChip N b h ρ α),
      (readOdd n o k).run w v = (k fun i => w.side v (2 * (o + i) + 1)).run w v
  | 0, o, k => by simp only [readOdd]; rw [← elim0_side w v o]
  | n + 1, o, k => by
    simp only [readOdd, run]
    rw [readOdd_run w v n, cons_side]

theorem readOdd_linkBits (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n o : ℕ) (k : Bits n → PIChip N b h ρ α),
      (readOdd n o k).linkBits w v = (k fun i => w.side v (2 * (o + i) + 1)).linkBits w v + n
  | 0, o, k => by simp only [readOdd, add_zero]; rw [← elim0_side w v o]
  | n + 1, o, k => by
    simp only [readOdd, linkBits]
    rw [readOdd_linkBits w v n, cons_side]
    omega

theorem readOdd_queries (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n o : ℕ) (k : Bits n → PIChip N b h ρ α),
      (readOdd n o k).queries w v = (k fun i => w.side v (2 * (o + i) + 1)).queries w v
  | 0, o, k => by simp only [readOdd]; rw [← elim0_side w v o]
  | n + 1, o, k => by
    simp only [readOdd, queries]
    rw [readOdd_queries w v n, cons_side]

theorem readOdd_tierBits {τ : ℕ} (tier : ℕ → Fin τ) (t : Fin τ) (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n o : ℕ) (k : Bits n → PIChip N b h ρ α),
      (readOdd n o k).tierBits tier t w v = (k fun i => w.side v (2 * (o + i) + 1)).tierBits tier t w v
  | 0, o, k => by simp only [readOdd]; rw [← elim0_side w v o]
  | n + 1, o, k => by
    simp only [readOdd, tierBits]
    rw [readOdd_tierBits tier t w v n, cons_side]

theorem readOdd_work (cost : ℕ → ℕ) (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n o : ℕ) (k : Bits n → PIChip N b h ρ α),
      (readOdd n o k).work cost w v = (k fun i => w.side v (2 * (o + i) + 1)).work cost w v
  | 0, o, k => by simp only [readOdd]; rw [← elim0_side w v o]
  | n + 1, o, k => by
    simp only [readOdd, work]
    rw [readOdd_work cost w v n, cons_side]

/-- Rerun a provenance over the side link and continue with `k` at its values: each step's bits are read at the next
odd addresses from `2·o + 1`, within a budget of `B` bits, past which the program stops at `dflt`. -/
def walk (dflt : α) (k : β → PIChip N b h ρ α) : ℕ → ℕ → PassProvenance N b h ρ β → PIChip N b h ρ α
  | _, _, .done a => k a
  | B, o, .input x _ k' => if x ≤ B then readOdd x o (fun z => walk dflt k (B - x) (o + x) (k' z)) else done dflt
  | B, o, .encode _ x _ _ k' =>
    if x ≤ B then readOdd x o (fun z => walk dflt k (B - x) (o + x) (k' z)) else done dflt

/-- Where the side sources answer the provenance's transcript from `2·o + 1` on, within its budget, the rerun program
returns and reads what `k` does at the provenance's values. -/
theorem walk_eq {τ : ℕ} (dflt : α) (k : β → PIChip N b h ρ α) (tier : ℕ → Fin τ) (t : Fin τ) (w : IWorld N b h ρ)
    (v : List Bool) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    ∀ (c : PassProvenance N b h ρ β) (B o : ℕ), (∀ i, w.side v (2 * (o + i) + 1) = c.transcript E R O i) →
      c.inputs E R O + c.encoded E R O ≤ B →
      (walk dflt k B o c).run w v = (k (c.run E R O)).run w v ∧
        (walk dflt k B o c).tierBits tier t w v = (k (c.run E R O)).tierBits tier t w v := by
  intro c
  induction c with
  | done a => intro B o _ _; exact ⟨rfl, rfl⟩
  | input x f k' ih =>
    intro B o hw hB
    simp only [PassProvenance.inputs, PassProvenance.encoded] at hB
    have hx : x ≤ B := by omega
    have hz : (fun i : Fin x => w.side v (2 * (o + i) + 1)) = f E R O := by
      funext i
      rw [hw i]
      simp [PassProvenance.transcript, i.2]
    have hw' : ∀ i, w.side v (2 * (o + x + i) + 1) = (k' (f E R O)).transcript E R O i := by
      intro i
      rw [show o + x + i = o + (x + i) by omega, hw (x + i)]
      simp [PassProvenance.transcript]
    obtain ⟨h₁, h₂⟩ := ih (f E R O) (B - x) (o + x) hw' (by omega)
    simp only [walk, hx, ↓reduceIte, readOdd_run, readOdd_tierBits, hz, PassProvenance.run]
    exact ⟨h₁, h₂⟩
  | encode j x hx' f k' ih =>
    intro B o hw hB
    simp only [PassProvenance.inputs, PassProvenance.encoded] at hB
    have hx : x ≤ B := by omega
    have hz : (fun i : Fin x => w.side v (2 * (o + i) + 1)) = f (E j) := by
      funext i
      rw [hw i]
      simp [PassProvenance.transcript, i.2]
    have hw' : ∀ i, w.side v (2 * (o + x + i) + 1) = (k' (f (E j))).transcript E R O i := by
      intro i
      rw [show o + x + i = o + (x + i) by omega, hw (x + i)]
      simp [PassProvenance.transcript]
    obtain ⟨h₁, h₂⟩ := ih (f (E j)) (B - x) (o + x) hw' (by omega)
    simp only [walk, hx, ↓reduceIte, readOdd_run, readOdd_tierBits, hz, PassProvenance.run]
    exact ⟨h₁, h₂⟩

/-- In every world the rerun program takes in at most its budget `B` over the side link beyond what `k` does, and
queries what `k` does. -/
theorem walk_bounded (dflt : α) (k : β → PIChip N b h ρ α) (w : IWorld N b h ρ) (v : List Bool) {L q : ℕ}
    (hk : ∀ γ, (k γ).linkBits w v ≤ L ∧ ((k γ).queries w v).length ≤ q) :
    ∀ (c : PassProvenance N b h ρ β) (B o : ℕ),
      (walk dflt k B o c).linkBits w v ≤ L + B ∧ ((walk dflt k B o c).queries w v).length ≤ q := by
  intro c
  induction c with
  | done a =>
    intro B o
    have := hk a
    simp only [walk]
    omega
  | input x f k' ih =>
    intro B o
    by_cases hx : x ≤ B
    · simp only [walk, hx, ↓reduceIte, readOdd_linkBits, readOdd_queries]
      have := ih (fun i => w.side v (2 * (o + i) + 1)) (B - x) (o + x)
      omega
    · simp [walk, hx, linkBits, queries]
  | encode j x hx' f k' ih =>
    intro B o
    by_cases hx : x ≤ B
    · simp only [walk, hx, ↓reduceIte, readOdd_linkBits, readOdd_queries]
      have := ih (fun i => w.side v (2 * (o + i) + 1)) (B - x) (o + x)
      omega
    · simp [walk, hx, linkBits, queries]

/-- In every world the rerun program's compute is at most what `k`'s is at most. -/
theorem walk_work_le (dflt : α) (k : β → PIChip N b h ρ α) (cost : ℕ → ℕ) (w : IWorld N b h ρ) (v : List Bool)
    {F : ℕ} (hk : ∀ γ, (k γ).work cost w v ≤ F) :
    ∀ (c : PassProvenance N b h ρ β) (B o : ℕ), (walk dflt k B o c).work cost w v ≤ F := by
  intro c
  induction c with
  | done a => intro B o; exact hk a
  | input x f k' ih =>
    intro B o
    by_cases hx : x ≤ B
    · simp only [walk, hx, ↓reduceIte, readOdd_work]
      exact ih _ _ _
    · simp [walk, hx, work]
  | encode j x hx' f k' ih =>
    intro B o
    by_cases hx : x ≤ B
    · simp only [walk, hx, ↓reduceIte, readOdd_work]
      exact ih _ _ _
    · simp [walk, hx, work]

/-- The program with its side reads at odd addresses answered on chip from `κ`, at `κ (a / 2)` for address `a`. -/
def answer (κ : ℕ → Bool) : PIChip N b h ρ α → PIChip N b h ρ α
  | done a => done a
  | tag x k => tag x fun r => answer κ (k r)
  | ingress a k => ingress a fun γ => answer κ (k γ)
  | side a k => if a % 2 = 0 then side a fun γ => answer κ (k γ) else answer κ (k (κ (a / 2)))
  | stream i k => stream i fun γ => answer κ (k γ)
  | complete j x hx g k => complete j x hx g fun z => answer κ (k z)
  | send γ p => send γ (answer κ p)

/-- The world whose side sources answer odd addresses from `κ`, at `κ (a / 2)` for address `a`. -/
def withOdd (κ : ℕ → Bool) (w : IWorld N b h ρ) : IWorld N b h ρ :=
  { w with side := fun v a => if a % 2 = 0 then w.side v a else κ (a / 2) }

@[simp] theorem withOdd_oracle (κ : ℕ → Bool) (w : IWorld N b h ρ) : (withOdd κ w).oracle = w.oracle := rfl
@[simp] theorem withOdd_mem (κ : ℕ → Bool) (w : IWorld N b h ρ) : (withOdd κ w).mem = w.mem := rfl
@[simp] theorem withOdd_stream (κ : ℕ → Bool) (w : IWorld N b h ρ) : (withOdd κ w).stream = w.stream := rfl
@[simp] theorem withOdd_store (κ : ℕ → Bool) (w : IWorld N b h ρ) : (withOdd κ w).store = w.store := rfl
theorem withOdd_side (κ : ℕ → Bool) (w : IWorld N b h ρ) (v : List Bool) (a : ℕ) :
    (withOdd κ w).side v a = if a % 2 = 0 then w.side v a else κ (a / 2) := rfl

theorem answer_run (κ : ℕ → Bool) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (v : List Bool), (answer κ c).run w v = c.run (withOdd κ w) v := by
  intro c
  induction c with
  | done a => intro v; rfl
  | tag x k ih => intro v; simp only [answer, run, withOdd_oracle]; exact ih _ v
  | ingress a k ih => intro v; simp only [answer, run, withOdd_mem]; exact ih _ v
  | side a k ih =>
    intro v
    by_cases ha : a % 2 = 0
    · simp only [answer, ha, ↓reduceIte, run, withOdd_side]; exact ih _ v
    · simp only [answer, ha, ↓reduceIte, run, withOdd_side]; exact ih _ v
  | stream i k ih => intro v; simp only [answer, run, withOdd_stream]; exact ih _ v
  | complete j x hx g k ih => intro v; simp only [answer, run, withOdd_store]; exact ih _ v
  | send γ p ih => intro v; simp only [answer, run]; exact ih _

theorem answer_tierBits {τ : ℕ} (tier : ℕ → Fin τ) (t : Fin τ) (κ : ℕ → Bool) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (v : List Bool),
      (answer κ c).tierBits tier t w v = c.tierBits tier t (withOdd κ w) v := by
  intro c
  induction c with
  | done a => intro v; rfl
  | tag x k ih => intro v; simp only [answer, tierBits, withOdd_oracle]; exact ih _ v
  | ingress a k ih => intro v; simp only [answer, tierBits, withOdd_mem]; rw [ih]
  | side a k ih =>
    intro v
    by_cases ha : a % 2 = 0
    · simp only [answer, ha, ↓reduceIte, tierBits, withOdd_side]; exact ih _ v
    · simp only [answer, ha, ↓reduceIte, tierBits, withOdd_side]; exact ih _ v
  | stream i k ih => intro v; simp only [answer, tierBits, withOdd_stream]; exact ih _ v
  | complete j x hx g k ih => intro v; simp only [answer, tierBits, withOdd_store]; exact ih _ v
  | send γ p ih => intro v; simp only [answer, tierBits]; exact ih _

theorem answer_linkBits (κ : ℕ → Bool) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (v : List Bool), (answer κ c).linkBits w v = c.linkBitsEven (withOdd κ w) v := by
  intro c
  induction c with
  | done a => intro v; rfl
  | tag x k ih => intro v; simp only [answer, linkBits, linkBitsEven, withOdd_oracle]; exact ih _ v
  | ingress a k ih => intro v; simp only [answer, linkBits, linkBitsEven, withOdd_mem]; exact ih _ v
  | side a k ih =>
    intro v
    by_cases ha : a % 2 = 0
    · simp only [answer, ha, ↓reduceIte, linkBits, linkBitsEven, withOdd_side]; rw [ih]
    · simp only [answer, ha, ↓reduceIte, linkBitsEven, withOdd_side]; exact ih _ v
  | stream i k ih => intro v; simp only [answer, linkBits, linkBitsEven, withOdd_stream]; rw [ih]
  | complete j x hx g k ih => intro v; simp only [answer, linkBits, linkBitsEven, withOdd_store]; exact ih _ v
  | send γ p ih => intro v; simp only [answer, linkBits, linkBitsEven]; exact ih _

theorem answer_queries (κ : ℕ → Bool) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (v : List Bool), (answer κ c).queries w v = c.queries (withOdd κ w) v := by
  intro c
  induction c with
  | done a => intro v; rfl
  | tag x k ih => intro v; simp only [answer, queries, withOdd_oracle]; rw [ih]
  | ingress a k ih => intro v; simp only [answer, queries, withOdd_mem]; exact ih _ v
  | side a k ih =>
    intro v
    by_cases ha : a % 2 = 0
    · simp only [answer, ha, ↓reduceIte, queries, withOdd_side]; exact ih _ v
    · simp only [answer, ha, ↓reduceIte, queries, withOdd_side]; exact ih _ v
  | stream i k ih => intro v; simp only [answer, queries, withOdd_stream]; exact ih _ v
  | complete j x hx g k ih => intro v; simp only [answer, queries, withOdd_store]; exact ih _ v
  | send γ p ih => intro v; simp only [answer, queries]; exact ih _

theorem answer_work (κ : ℕ → Bool) (cost : ℕ → ℕ) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (v : List Bool), (answer κ c).work cost w v = c.work cost (withOdd κ w) v := by
  intro c
  induction c with
  | done a => intro v; rfl
  | tag x k ih => intro v; simp only [answer, work, withOdd_oracle]; exact ih _ v
  | ingress a k ih => intro v; simp only [answer, work, withOdd_mem]; exact ih _ v
  | side a k ih =>
    intro v
    by_cases ha : a % 2 = 0
    · simp only [answer, ha, ↓reduceIte, work, withOdd_side]; exact ih _ v
    · simp only [answer, ha, ↓reduceIte, work, withOdd_side]; exact ih _ v
  | stream i k ih => intro v; simp only [answer, work, withOdd_stream]; exact ih _ v
  | complete j x hx g k ih => intro v; simp only [answer, work, withOdd_store]; rw [ih]
  | send γ p ih => intro v; simp only [answer, work]; exact ih _

end PIChip

namespace PIDevice

variable {N b h ρ S L' q : ℕ}

/-- The device that reruns its carrier's circuit `K` over the side link and answers the carrier's reads, `B`'s side
reads at odd addresses, from the circuit's values on chip: its side sources are `B`'s at even addresses and `K`'s
transcript at odd ones, of which it reads at most `C` bits. -/
def stack (B : PIDevice N b h ρ S L' q) (L C : ℕ) (K : PassProvenance N b h ρ (ℕ → Bool))
    (hL : ∀ st w, (B.chip st).linkBitsEven w [] ≤ L) : PIDevice N b h ρ S (L + C) q where
  state := B.state
  mem := B.mem
  side E R O v a := if a % 2 = 0 then B.side E R O v a else K.transcript E R O (a / 2)
  chip st := PIChip.walk (fun _ _ => false) (fun κ => (B.chip st).answer κ) C 0 K
  bounded st w := PIChip.walk_bounded _ _ w [] (fun κ =>
    ⟨(PIChip.answer_linkBits κ w (B.chip st) []).trans_le (hL st _),
      by rw [PIChip.answer_queries]; exact (B.bounded st _).2⟩) K C 0

theorem stack_side_odd (B : PIDevice N b h ρ S L' q) (L C : ℕ) (K : PassProvenance N b h ρ (ℕ → Bool)) (hL)
    (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (v : List Bool) (i : ℕ) :
    (B.stack L C K hL).side E R O v (2 * (0 + i) + 1) = K.transcript E R O i := by
  simp only [stack, show (2 * (0 + i) + 1) % 2 ≠ 0 by omega, ↓reduceIte, show (2 * (0 + i) + 1) / 2 = i by omega]

/-- Where `B`'s odd side reads answer `K`'s values, the stacked device's world with them answered on chip is `B`'s. -/
theorem stack_withOdd (B : PIDevice N b h ρ S L' q) (L C : ℕ) (K : PassProvenance N b h ρ (ℕ → Bool)) (hL)
    (hside : ∀ E R O v i, B.side E R O v (2 * i + 1) = K.run E R O i) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) :
    PIChip.withOdd (K.run E R O) ((B.stack L C K hL).world E R O) = B.world E R O := by
  have hs : ∀ v a, (if a % 2 = 0 then (B.stack L C K hL).side E R O v a else K.run E R O (a / 2)) =
      B.side E R O v a := by
    intro v a
    by_cases ha : a % 2 = 0
    · simp [stack, ha]
    · simp only [ha, ↓reduceIte]
      rw [← hside E R O v (a / 2), show 2 * (a / 2) + 1 = a by omega]
  simp only [PIChip.withOdd, world, hs]
  rfl

/-- **The stacked device commits `B`'s receipts and reads what `B` reads**, where `B`'s odd side reads answer `K`'s
values and `K` is within the budget `C`. -/
theorem stack_eq {τ : ℕ} (B : PIDevice N b h ρ S L' q) (L C : ℕ) (K : PassProvenance N b h ρ (ℕ → Bool)) (hL)
    (hside : ∀ E R O v i, B.side E R O v (2 * i + 1) = K.run E R O i) (tier : ℕ → Fin τ) (t : Fin τ)
    (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (hbud : K.inputs E R O + K.encoded E R O ≤ C) :
    (B.stack L C K hL).tags E R O = B.tags E R O ∧
      (B.stack L C K hL).tierReads tier t E R O = B.tierReads tier t E R O := by
  obtain ⟨h₁, h₂⟩ := PIChip.walk_eq (fun _ _ => false) (fun κ => (B.chip (B.state E)).answer κ) tier t
    ((B.stack L C K hL).world E R O) [] E R O K C 0 (stack_side_odd B L C K hL E R O []) hbud
  refine ⟨?_, ?_⟩
  · calc (B.stack L C K hL).tags E R O
        = ((B.chip (B.state E)).answer (K.run E R O)).run ((B.stack L C K hL).world E R O) [] := h₁
      _ = (B.chip (B.state E)).run (PIChip.withOdd (K.run E R O) ((B.stack L C K hL).world E R O)) [] :=
          PIChip.answer_run _ _ _ _
      _ = B.tags E R O := by rw [stack_withOdd B L C K hL hside]; rfl
  · calc (B.stack L C K hL).tierReads tier t E R O
        = ((B.chip (B.state E)).answer (K.run E R O)).tierBits tier t ((B.stack L C K hL).world E R O) [] := h₂
      _ = (B.chip (B.state E)).tierBits tier t
            (PIChip.withOdd (K.run E R O) ((B.stack L C K hL).world E R O)) [] :=
          PIChip.answer_tierBits _ _ _ _ _ _
      _ = B.tierReads tier t E R O := by rw [stack_withOdd B L C K hL hside]; rfl

theorem stack_correct (B : PIDevice N b h ρ S L' q) (L C : ℕ) (K : PassProvenance N b h ρ (ℕ → Bool)) (hL)
    (hside : ∀ E R O v i, B.side E R O v (2 * i + 1) = K.run E R O i) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) (hbud : K.inputs E R O + K.encoded E R O ≤ C) :
    (B.stack L C K hL).correct E R O = B.correct E R O := by
  have h := (stack_eq B L C K hL hside (fun _ => (0 : Fin 1)) 0 E R O hbud).1
  simp only [correct, h]

/-- The stacked device is within `B`'s squaring ceiling. -/
theorem stack_ceiling (B : PIDevice N b h ρ S L' q) (L C : ℕ) (K : PassProvenance N b h ρ (ℕ → Bool)) (hL)
    {cost : ℕ → ℕ} {F : ℕ} (hF : Assumptions.SquaringCeiling cost F B) :
    Assumptions.SquaringCeiling cost F (B.stack L C K hL) := fun st w =>
  PIChip.walk_work_le _ _ cost w [] (fun κ => (PIChip.answer_work κ cost w (B.chip st) []).trans_le (hF st _)) K C 0

end PIDevice

namespace DeployedSession

variable {n P N b h ρ S L' q r s d G : ℕ}

/-- The session with every pass device stacked on its carrier (`PIDevice.stack`), with the same candidates and
picks. -/
def stack (A : DeployedSession n P N b h ρ S L' q r s d G) (L C : ℕ)
    (K : ∀ (w : Fin n) (_ : {v : Fin n // v < w} → WindowCoins P N b h ρ r s d) (j : Fin P)
      (_ : {i : Fin P // i < j} → PassCoins N b h ρ r s) (_ : Fin G), PassProvenance N b h ρ (ℕ → Bool))
    (hL : ∀ w η j ι c st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L) :
    DeployedSession n P N b h ρ S (L + C) q r s d G where
  window w η :=
    { device := fun j ι c => ((A.window w η).device j ι c).stack L C (K w η j ι c) (hL w η j ι c)
      cand := (A.window w η).cand
      pick := (A.window w η).pick }

/-- **The stacked session's windows pass and read what the session's do**, where every device's odd side reads answer
its carrier's values and every carrier is within the budget `C`. -/
theorem stack_win {τ : ℕ} (A : DeployedSession n P N b h ρ S L' q r s d G) (L C : ℕ)
    (K : ∀ (w : Fin n) (_ : {v : Fin n // v < w} → WindowCoins P N b h ρ r s d) (j : Fin P)
      (_ : {i : Fin P // i < j} → PassCoins N b h ρ r s) (_ : Fin G), PassProvenance N b h ρ (ℕ → Bool))
    (hL : ∀ w η j ι c st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L)
    (hside : ∀ w η j ι c E R O v i, ((A.window w η).device j ι c).side E R O v (2 * i + 1) = (K w η j ι c).run E R O i)
    (hbud : ∀ w η j ι c E R O, (K w η j ι c).inputs E R O + (K w η j ι c).encoded E R O ≤ C) (tier : ℕ → Fin τ)
    (w : Fin n) (ω : Fin n → WindowCoins P N b h ρ r s d) (E : Store N b) (ω' : Fin P → PassCoins N b h ρ r s) :
    ((A.stack L C K hL).win w ω).correctAll E ω' = (A.win w ω).correctAll E ω' ∧
      ((A.stack L C K hL).win w ω).reads tier E ω' = (A.win w ω).reads tier E ω' := by
  refine ⟨Finset.filter_congr fun p _ => ?_, Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun t _ => ?_⟩
  · have := PIDevice.stack_correct ((A.win w ω).dev p.1 E ω') L C _ (hL w _ p.1 _ _) (hside w _ p.1 _ _) E
      (ω' p.1).1 ((A.win w ω).oracle p.1 E ω') (hbud w _ p.1 _ _ _ _ _)
    exact Iff.of_eq (congrArg (fun T => p.2 ∈ T) this)
  · exact (PIDevice.stack_eq ((A.win w ω).dev j E ω') L C _ (hL w _ j _ _) (hside w _ j _ _) tier t E (ω' j).1
      ((A.win w ω).oracle j E ω') (hbud w _ j _ _ _ _ _)).2

end DeployedSession

end Accounting.Rate

namespace Accounting.Requirement

variable {V : Type} [DecidableEq V] (W : Wiring V) (e : Execution V) (ℓ : Layout V)

/-- Over a window of at most `k` steps at each of which the store is `W`'s with at most `N` resident blocks and the plan
is `W`'s reads of at most `N` blocks, `W` holds and reads at most `k·N·B` bits. -/
theorem cap_of_matches (s : Store) (plan : ℕ → Finset ℕ) (B N k : ℕ) (win : Window) (hwin : win.len ≤ k)
    (hm : ∀ t ∈ win.ticks, StoreMatches W e ℓ s B t ∧ (s.resident t).card ≤ N ∧
      PlanMatches W e ℓ (plan t) B t ∧ (plan t).card ≤ N) :
    held W e win ≤ k * N * B ∧ read W e win ≤ k * N * B := by
  have hsum : ∑ _t ∈ win.ticks, N * B ≤ k * N * B := by
    rw [Finset.sum_const, Window.ticks, Nat.card_Ico, smul_eq_mul, Nat.add_sub_cancel_left, mul_assoc]
    exact Nat.mul_le_mul_right _ hwin
  refine ⟨(Finset.sum_le_sum fun t ht => ?_).trans hsum, (Finset.sum_le_sum fun t ht => ?_).trans hsum⟩
  · have h1 := SecurityProofs.RequirementHeld W e ℓ s B t ∅ (hm t ht).1
    simp only [bitsAt, bits, Finset.notMem_empty, Finset.filter_false, Finset.sum_empty, Finset.sdiff_empty,
      zero_add] at h1
    exact h1.trans (Nat.mul_le_mul_right B (hm t ht).2.1)
  · have h1 := SecurityProofs.RequirementMoved W e ℓ (plan t) B t ∅ (hm t ht).2.2.1
    simp only [bitsAt, bits, Finset.notMem_empty, Finset.filter_false, Finset.sum_empty, Finset.sdiff_empty,
      zero_add] at h1
    exact h1.trans (Nat.mul_le_mul_right B (hm t ht).2.2.2)

/-- A count short of `c` times a requirement's held or read bits, each at most `Y`, is short of `c·Y`. -/
theorem lt_of_cap {a H R Y : ℕ} {c : ℝ} (hH : H ≤ Y) (hR : R ≤ Y) (h : (a : ℝ) < c * H ∨ (a : ℝ) < c * R) :
    (a : ℝ) < c * Y := by
  have key : ∀ {X : ℕ}, X ≤ Y → (a : ℝ) < c * X → (a : ℝ) < c * Y := by
    intro X hX hlt
    rcases le_or_gt 0 c with hc | hc
    · exact hlt.trans_le (mul_le_mul_of_nonneg_left (Nat.cast_le.mpr hX) hc)
    · nlinarith [(Nat.cast_nonneg a : (0 : ℝ) ≤ a), (Nat.cast_nonneg X : (0 : ℝ) ≤ X)]
  exact h.elim (key hH) (key hR)

end Accounting.Requirement

namespace Accounting.SecurityProofs

open Accounting.RateSketch
open Pous (pr)
open scoped ENNReal

/-- **The carrier lemma**, `Accounting.Guarantees.CarrierBits`. -/
theorem CarrierBits : Accounting.Guarantees.CarrierBits := by
  intro N m I X α c hc
  classical
  have h := c.card_image_le Finset.univ (I + X) fun E _ => add_le_add (hc E).1 (hc E).2
  rwa [Nat.card_coe_set_eq, Set.ncard_eq_toFinset_card', Set.toFinset_range]

/-- **The remote audit at the deployed parameters on a programmed store, with stacking**,
`Accounting.Guarantees.StackedSessionConservationSketchDeployed`. -/
theorem StackedSessionConservationSketchDeployed :
    Accounting.Guarantees.StackedSessionConservationSketchDeployed := by
  intro n P N ρ S L q G X qtot Rin cρ reads life E fan wid εnum εden a ε δ Ω Ch _ _ view choice A hn hP hN hE0 hCh
    hrank hKB hE hB hK C
  have hX : ∀ w η j ι c Es σ R O,
      let B := (A.window w η).device j ι c
      let D := RateSketch.Data.ofParts Es R
      (B.prog σ D).rebuilds O D * 256 ≤ X + C := by
    intro w η j ι c Es σ R O
    obtain ⟨K, hS, hKb⟩ := hK w η j ι c
    have h1 := hS Es σ R O
    have h2 := K.card_le_of_budget hE (by decide) hE0 hKb
    dsimp only at h1 ⊢
    calc _ ≤ X + K.carried.card * 256 := h1
      _ ≤ X + C := Nat.add_le_add_left h2 X
  have hu := hKB.pr_le_advice choice fun (Es : SketchStore N 128)
      (ω : Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
    ∃ w, (A.win w ω).reads Es (ω w).1 (ω w).2.1 + P * (S + (L - ρ * 256) + (X + C)) <
      ((A.win w ω).correctAll Es (ω w).1 (ω w).2.1).card * (128 * 256)
  have hunif := SketchDeployedSession.pr_unpaid_sum_le A 12 (X + C) qtot ε (by norm_num) hrank hX hB
  have hCh' : (Fintype.card Ch : ℝ≥0∞) ≤ 2 ^ a := by exact_mod_cast hCh
  refine (SketchDeployedSession.pr_deployed_of_unpaid (fun ω => view (choice ω) ω) A (X + C) hn hP hN _
    (hu.trans (mul_le_mul_right (add_le_add hunif le_rfl) _))).trans
    (le_of_le_of_eq (add_le_add le_rfl (mul_le_mul_left hCh' _)) ?_)
  rw [show 8 * 12 = 96 from rfl]
  ring

/-- **The remote audit with relaying side sources on a programmed store, with stacking**,
`Accounting.Guarantees.StackedSessionConservationRelay`. -/
theorem StackedSessionConservationRelay : Accounting.Guarantees.StackedSessionConservationRelay := by
  intro n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E a cost Fc k fan wid εnum εden tier δ
    Ω Ch _ _ view choice A dbar dw hn hP hN hb hrc hdw hE0 hbytes hCh hKB hW hF hE hK C
  choose K hside hL hbud using hK
  have hC : ∀ w η j ι c Es R O, (K w η j ι c).inputs Es R O + (K w η j ι c).encoded Es R O ≤ C := by
    intro w η j ι c Es R O
    have h1 := (hbud w η j ι c Es R O).1
    have h2 : (K w η j ι c).encoded Es R O ≤ 8 * Fc.bytes * (cρ * reads * life) / E := by
      rw [Nat.le_div_iff_mul_le hE0]
      have h := ((K w η j ι c).encoded_mul_le hE hbytes Es R O).trans
        (mul_le_mul_right (hbud w η j ι c Es R O).2 _)
      exact_mod_cast h
    exact Nat.add_le_add h1 h2
  have hu := hKB.pr_le_advice choice fun (Es : Rate.Store N b) (ω : Fin n → Rate.WindowCoins P N b h ρ r s d) =>
    ∃ w, ((A.stack L C K hL).win w ω).reads tier Es (ω w).1 +
        P * (S + (L + C - ρ) + Rate.Deployed.rebuildBits b rounds cRoot F + (lam + m * b)) <
      (((A.stack L C K hL).win w ω).correctAll Es (ω w).1).card * b
  have hunif := Rate.DeployedSession.pr_unpaid_le (A.stack L C K hL) tier _ _
    (Remote.window_relay_le (lam := lam) (m := m) (A.stack L C K hL) tier hrc hW
      (fun w η j ι c => Rate.PIDevice.stack_ceiling _ L C (K w η j ι c) (hL w η j ι c) (hF w η j ι c)))
  have base := Remote.session_relay_of_unpaid (fun ω => view (choice ω) ω) (A.stack L C K hL) tier lam m rounds
    cRoot F dbar dw hn hP hN hb hdw _ (hu.trans (mul_le_mul_right (add_le_add hunif le_rfl) _))
  have hCh' : (Fintype.card Ch : ℝ≥0∞) ≤ 2 ^ a := by exact_mod_cast hCh
  refine ((Bandwidth.pr_mono fun y hy => ?_).trans base).trans
    (le_of_le_of_eq (add_le_add le_rfl (mul_le_mul_left hCh' _)) ?_)
  · have e := fun w => Rate.DeployedSession.stack_win A L C K hL hside hC tier w y.2 (view (choice y.1) y.1) (y.2 w).1
    simp only [fun w => (e w).1, fun w => (e w).2]
    exact hy
  · simp only [div_eq_mul_inv, pow_add]
    ring

/-- **The stacked remote audit at the deployed parameters, read against a declared circuit's requirement**,
`Accounting.Guarantees.StackedRequirementSketchDeployed`. -/
theorem StackedRequirementSketchDeployed : Accounting.Guarantees.StackedRequirementSketchDeployed := by
  intro n P N ρ S L q G X qtot Rin cρ reads life E fan wid εnum εden a ε δ Ω Ch _ _ view choice A V _ W e ℓ mm plan
    win hn hP hN hE0 hCh hrank hKB hE hB hK hwin hm C γ
  have hcap := Requirement.cap_of_matches W e ℓ mm plan 32768 N (n * P) win hwin hm
  have base := StackedSessionConservationSketchDeployed n P N ρ S L q G X qtot Rin cρ reads life E fan wid εnum εden
    a ε δ view choice A hn hP hN hE0 hCh hrank hKB hE hB hK
  dsimp only at base
  refine (Bandwidth.pr_mono fun x hx => ⟨hx.1, Or.inl ?_⟩).trans base
  have h := Requirement.lt_of_cap hcap.1 hcap.2 hx.2
  simp only [Nat.cast_mul, Nat.cast_ofNat] at h
  exact h

/-- **The stacked remote audit with relaying side sources, read against a declared circuit's requirement**,
`Accounting.Guarantees.StackedRequirementRelay`. -/
theorem StackedRequirementRelay : Accounting.Guarantees.StackedRequirementRelay := by
  intro n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E a cost Fc k fan wid εnum εden tier δ
    Ω Ch _ _ view choice A dbar dw V _ W e ℓ mm plan win hn hP hN hb hrc hdw hE0 hbytes hCh hKB hW hF hE hK hwin hm
    C γ
  have hcap := Requirement.cap_of_matches W e ℓ mm plan b N (n * P) win hwin hm
  have base := StackedSessionConservationRelay n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life
    E a cost Fc k fan wid εnum εden tier δ view choice A dbar dw hn hP hN hb hrc hdw hE0 hbytes hCh hKB hW hF hE hK
  dsimp only at base
  refine (Bandwidth.pr_mono fun x hx => ⟨hx.1, Or.inl ?_⟩).trans base
  have h := Requirement.lt_of_cap hcap.1 hcap.2 hx.2
  simp only [Nat.cast_mul] at h
  exact h

end Accounting.SecurityProofs
