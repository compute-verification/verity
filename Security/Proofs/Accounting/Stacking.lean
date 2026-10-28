import Mathlib
import Definitions.Accounting.Guarantees.Stacking
import Proofs.Accounting.RateAudit.DeployedSketchSum
import Proofs.Accounting.RateAudit.DeployedRelay
import Proofs.Accounting.Requirement
import Proofs.Accounting.Gates

/-!
# Stacking: the carrier lemma and the stacked session

`Accounting.SecurityProofs.CarrierBits` counts a circuit's values: a step of `x` bits splits the stores into at most
`2^x` classes, in each of which the rest of the provenance has `x` bits fewer, so the values take at most `2^n`
contents when every run has at most `n` input and encoded bits (`Provenance.card_image_le`). A carrier's values are its
sectors at every store, which take every value, so its sectors are at most `n` bits (`Carrier.card_le`). Under
`SquaringFloor` and `ChainSum` (the encode bound at `E ≤ k · S · e`, `SecurityProofs.Gates.encodeBound_of`) an encode
of `x` bits costs at least `x/(8·bytes)` words' `E` nonlinear units (`encodeBitsCost_ge`, with `⊤` where the gate model
has no circuit), so a gate budget bounds the encoded bits (`Provenance.encoded_mul_le`).

`KeyBinding` makes a pass's store uniform but for `δ` at each prefix of the session's coins (`UniformBut`): at the
prefix the writers' choices and the layout are fixed, so the blocks are `KeyBinding`'s, and the tiles' sectors are a
uniform store's when the blocks are uniform (`KeyBinding.uniformBut_coinStore`; the general class's one store,
`KeyBinding.uniformBut_blocks`). A store uniform but for `δ` at an index read from the coins moves an event of the
store and the coins to a uniform store at a cost of `δ` (`UniformBut.pr_le_of`), fiber by fiber over the coins.

`Accounting.SecurityProofs.StackedSessionConservationSketchDeployed` charges each carrier's sectors in the pass's
rebuilds, so every pass rebuilds at most `X + C` bits, and bounds the session with a store per pass. A window that
doesn't pay for its correct receipts has a sign matrix of low rank, paid once a window, or a pass whose kept candidate
is unpaid on its own store (`SketchDeployedWindow.exists_unpaidOn`). That store is the append-coin rule's at the
prefix before the pass, which `KeyBinding` moves to a uniform store independent of the session's coins, where the
deployed guarantee's one-pass family bound holds (`SketchDeployedSession.pr_pass_le`). So the unpaid windows cost
`n·ε + n·P·(G·N/2^h + qtot/2^(8r) + δ)` (`SketchDeployedSession.pr_unpaidOn_le`). The passes' correct receipts are a
causal family of the windows' draws, whatever stores they are on (`pr_draws_causal_le`), so the draws' terms are the
deployed guarantee's (`pr_deployed_of_unpaid_causal`).

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

/-- **Under the encode bound at `E`, `x` bits of a block's encoding cost `x/(8·bytes)` words' `E` capped
units**: the encode cost of the `⌈x/(8·bytes)⌉` words those bits fill is at least that many words' `E`
(`SecurityProofs.Gates.encodeCost_ge`), and `⊤` where the gate model has no circuit. -/
theorem encodeBitsCost_ge {F : Gates.Field} {k B E εnum εden : ℕ}
    (hE : Gates.EncodeBound F k B E εnum εden) (hbytes : 0 < F.bytes) (x : ℕ) :
    ((x * E : ℕ) : ℕ∞) ≤ (8 * F.bytes : ℕ) * encodeBitsCost F k B εnum εden x := by
  have hn : 2 ≤ F.n := by
    have h := hbytes
    simp only [Gates.Field.bytes] at h
    omega
  have hwb : 0 < 8 * F.bytes := by omega
  set q := (x + 8 * F.bytes - 1) / (8 * F.bytes) with hq
  have hlt := Nat.lt_mul_div_succ (x + 8 * F.bytes - 1) hwb
  rw [← hq, mul_add, mul_one] at hlt
  have hxq : x ≤ 8 * F.bytes * q := by omega
  have hcost : ((q * E : ℕ) : ℕ∞) ≤ encodeBitsCost F k B εnum εden x :=
    SecurityProofs.Gates.encodeCost_ge hE hn q
  calc ((x * E : ℕ) : ℕ∞) ≤ ((8 * F.bytes * q * E : ℕ) : ℕ∞) := by
        exact_mod_cast Nat.mul_le_mul_right E hxq
    _ = (8 * F.bytes : ℕ) * ((q * E : ℕ) : ℕ∞) := by push_cast; ring
    _ ≤ (8 * F.bytes : ℕ) * encodeBitsCost F k B εnum εden x := by gcongr

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

/-- **A gate budget bounds the encoded bits**: under the encode bound at `E` nonlinear units a word of `8·bytes` bits,
the encoded bits times `E` are at most `8·bytes` times the encodes' nonlinear units. -/
theorem encoded_mul_le {F : Gates.Field} {kr B E εnum εden : ℕ}
    (hE : Gates.EncodeBound F kr B E εnum εden) (hbytes : 0 < F.bytes) (c : Provenance N m α)
    (Es : SketchStore N m) :
    ((c.encoded Es * E : ℕ) : ℕ∞) ≤ (8 * F.bytes : ℕ) * c.gates (encodeBitsCost F kr B εnum εden) Es := by
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

/-- A carrier whose circuit takes in at most `I` bits and spends at most `R` nonlinear units on its encodes carries at
most `I + 8·bytes·R/E` bits, under the encode bound at `E` nonlinear units a word of `8·bytes` bits. -/
theorem card_le_of_budget (K : Carrier N m) {F : Gates.Field} {kr B E εnum εden I R : ℕ}
    (hE : Gates.EncodeBound F kr B E εnum εden) (hbytes : 0 < F.bytes) (hE0 : 0 < E)
    (hK : ∀ Es, K.provenance.inputs Es ≤ I ∧
      K.provenance.gates (encodeBitsCost F kr B εnum εden) Es ≤ (R : ℕ)) :
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

end Accounting.Assumptions

namespace Accounting

open Pous (pr)
open scoped ENNReal

/-- A family of stores, one at each index `α`, each a uniform store but for `δ` on every event. -/
def UniformBut {St Ω I : Type} [Fintype St] [Fintype Ω] (δ : ℝ≥0∞) (view : I → Ω → St) : Prop :=
  ∀ (α : I) (ev : Set St), pr (fun ω => view α ω ∈ ev) ≤ pr (fun E : St => E ∈ ev) + δ

/-- **A store uniform but for `δ` at each index moves an event to a uniform store, at a cost of `δ`, at an index read
from the coins**: an event of the store at index `f c` and of coins `c` independent of `ω` is at most as likely as on
a uniform store independent of them, plus `δ`. -/
theorem UniformBut.pr_le_of {St Ω I C : Type} [Fintype St] [Fintype Ω] [Fintype C] {δ : ℝ≥0∞}
    {view : I → Ω → St} (h : UniformBut δ view) (f : C → I) (P : St → C → Prop) :
    pr (fun x : Ω × C => P (view (f x.2) x.1) x.2) ≤ pr (fun y : St × C => P y.1 y.2) + δ := by
  rw [Assumptions.pr_prod_eq_sum_div (fun x : Ω × C => P (view (f x.2) x.1) x.2),
    Assumptions.pr_prod_eq_sum_div (fun y : St × C => P y.1 y.2)]
  calc (∑ c, pr fun ω => P (view (f c) ω) c) / Fintype.card C
      ≤ (∑ c, (pr (fun E => P E c) + δ)) / Fintype.card C :=
        ENNReal.div_le_div_right (Finset.sum_le_sum fun c _ => h (f c) {E | P E c}) _
    _ = (∑ c, pr fun E => P E c) / Fintype.card C + Fintype.card C * δ / Fintype.card C := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ENNReal.add_div]
    _ ≤ (∑ c, pr fun E => P E c) / Fintype.card C + δ :=
        add_le_add le_rfl (ENNReal.div_le_of_le_mul (le_of_eq (mul_comm _ _)))

/-- **A store uniform but for `δ` moves an event to a uniform store, at a cost of `δ`**: at each index `α`, an event
of the store and of independent coins `C` is at most as likely as on a uniform store, plus `δ`. -/
theorem UniformBut.pr_le {St Ω I C : Type} [Fintype St] [Fintype Ω] [Fintype C] {δ : ℝ≥0∞} {view : I → Ω → St}
    (h : UniformBut δ view) (α : I) (P : St → C → Prop) :
    pr (fun x : Ω × C => P (view α x.1) x.2) ≤ pr (fun y : St × C => P y.1 y.2) + δ :=
  h.pr_le_of (fun _ => α) P

/-- **Under key binding the general class's store is uniform but for `δ`**: the blocks at the layout `slot`, the
writers choosing each append before its coin. -/
theorem Assumptions.KeyBinding.uniformBut_blocks {Ω Ca Ch B : Type} [Fintype Ω] [Fintype Ca] [Fintype B] {T M N : ℕ}
    {δ : ℝ≥0∞} {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B} (h : Assumptions.KeyBinding T M δ enc)
    (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch) (slot : Fin N ↪ Fin T × Fin M) :
    UniformBut δ (fun (_ : Unit) (y : Ω × (Fin T → Ca)) => appendCoinBlocks enc choose slot y) :=
  fun _ ev => h choose N slot ev

/-- **Under key binding a pass's store is uniform but for `δ` at each prefix**: at the prefix `c`, the writers'
choices `choose c` and the layout `slot c` are fixed, and the tiles' sectors are a uniform store's when the blocks
are uniform. -/
theorem Assumptions.KeyBinding.uniformBut_coinStore {n P N m ρ h ks G d T M : ℕ} {Ω Ca Ch : Type} [Fintype Ω]
    [Fintype Ca] {δ : ℝ≥0∞} {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin m → RateSketch.Sector)}
    (hKB : Assumptions.KeyBinding T M δ enc)
    (choose : RateSketch.SketchPrefix n P N m ρ h ks G d → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (slot : RateSketch.SketchPrefix n P N m ρ h ks G d → Fin N ↪ Fin T × Fin M) :
    UniformBut δ (RateSketch.coinStore enc choose slot) := by
  intro c ev
  have h1 := hKB (choose c) N (slot c) {E | (fun p => E p.1 p.2) ∈ ev}
  have e := Rate.pr_equiv (Equiv.curry (Fin N) (Fin m) RateSketch.Sector).symm
    fun E : Fin N → Fin m → RateSketch.Sector => E ∈ {E | (fun p => E p.1 p.2) ∈ ev}
  have hl : pr (fun y : Ω × (Fin T → Ca) => RateSketch.coinStore enc choose slot c y ∈ ev) =
      pr (fun y : Ω × (Fin T → Ca) =>
        appendCoinBlocks enc (choose c) (slot c) y ∈ {E | (fun p => E p.1 p.2) ∈ ev}) := rfl
  have hr : pr (fun E : Fin N → Fin m → RateSketch.Sector => E ∈ {E | (fun p => E p.1 p.2) ∈ ev}) =
      pr (fun E : RateSketch.SketchStore N m => E ∈ ev) := by
    rw [e]
    rfl
  rw [hl, ← hr]
  exact h1

end Accounting

namespace Accounting.Rate

open Pous (Bits pr)

namespace PassProvenance

variable {N b h ρ : ℕ} {α : Type}

/-- The provenance's step answers at store `E`, stream `R` and oracle `O`, one after another: bit `i` of the run's
bits, and `false` past them. -/
def transcript (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : PassProvenance N b h ρ α → ℕ → Bool
  | done _, _ => false
  | input x f k, i => if hi : i < x then f E R O ⟨i, hi⟩ else (k (f E R O)).transcript E R O (i - x)
  | encode j x _ f k, i => if hi : i < x then f (E j) ⟨i, hi⟩ else (k (f (E j))).transcript E R O (i - x)

/-- **A gate budget bounds the encoded bits**: under the encode bound at `E` nonlinear units a word of `8·bytes` bits,
the encoded bits times `E` are at most `8·bytes` times the encodes' nonlinear units. -/
theorem encoded_mul_le {F : Gates.Field} {kr B E εnum εden : ℕ}
    (hE : Gates.EncodeBound F kr B E εnum εden) (hbytes : 0 < F.bytes)
    (c : PassProvenance N b h ρ α) (Es : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    ((c.encoded Es R O * E : ℕ) : ℕ∞) ≤
      (8 * F.bytes : ℕ) * c.gates (encodeBitsCost F kr B εnum εden) Es R O := by
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
    (v : List Bool) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) :
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
    (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) (v : List Bool) (i : ℕ) :
    (B.stack L C K hL).side E R O v (2 * (0 + i) + 1) = K.transcript E R O i := by
  simp only [stack, show (2 * (0 + i) + 1) % 2 ≠ 0 by omega, ↓reduceIte, show (2 * (0 + i) + 1) / 2 = i by omega]

/-- Where `B`'s odd side reads answer `K`'s values, the stacked device's world with them answered on chip is `B`'s. -/
theorem stack_withOdd (B : PIDevice N b h ρ S L' q) (L C : ℕ) (K : PassProvenance N b h ρ (ℕ → Bool)) (hL)
    (hside : ∀ E R O v i, B.side E R O v (2 * i + 1) = K.run E R O i) (E : Payloads N b) (R : Bits ρ)
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
    (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) (hbud : K.inputs E R O + K.encoded E R O ≤ C) :
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
    (hside : ∀ E R O v i, B.side E R O v (2 * i + 1) = K.run E R O i) (E : Payloads N b) (R : Bits ρ)
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
    (w : Fin n) (ω : Fin n → WindowCoins P N b h ρ r s d) (E : Payloads N b) (ω' : Fin P → PassCoins N b h ρ r s) :
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

namespace Accounting.RateSketch

open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_exists_le pr_mono pr_or_le)

/-- The oracle family of a pass's `G` grinding candidates is finite. -/
local instance instFintypeOracleFamStack (N ρ h ks G : ℕ) :
    Fintype (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := by
  letI : Fintype (Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  exact inferInstance

/-- Signs whose sector blocks all have rank at least `r`. -/
def RankGood {ks m : ℕ} (r : ℕ) (σ : SketchSigns ks m) : Prop :=
  ∀ u, r ≤ ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank

namespace SketchDeployedWindow

variable {P N m ρ h S L q ks G : ℕ}

/-- Correct receipts counted pass by pass, each pass on its own store. -/
theorem card_correctOn (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : Fin P → SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) :
    (A.correctOn E σ ω).card = ∑ j,
      let D := Data.ofParts (E j) (ω j).1
      ((A.dev j (E j) σ ω).correct sketch σ (A.oracle j (E j) σ ω) D).card := by
  classical
  unfold correctOn
  rw [Finset.card_filter, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [Finset.sum_boole]
  simp

/-- If a window's reads, each pass on its own store, plus one slack per pass do not pay for its correct receipts, some
pass's kept candidate is unpaid on its own store. -/
theorem exists_unpaidOn (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : Fin P → SketchStore N m)
    (σ : SketchSigns ks m) (slack : ℕ) (ω : Fin P → SketchPassCoins N ρ h ks G)
    (hx : A.readsOn E σ ω + P * slack < (A.correctOn E σ ω).card * (m * 256)) :
    ∃ j c, UnpaidAt A σ slack j c (E j, ω) := by
  by_contra hno
  push Not at hno
  have hle := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) =>
    (not_lt.1 (hno j (A.kept j (E j) σ ω)) :
      let D := Data.ofParts (E j) (ω j).1
      ((A.dev j (E j) σ ω).correct sketch σ (A.oracle j (E j) σ ω) D).card * (m * 256) ≤
        (A.dev j (E j) σ ω).footprint σ (A.oracle j (E j) σ ω) D + slack))
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul] at hle
  rw [card_correctOn, Finset.sum_mul] at hx
  exact (not_lt_of_ge (by simpa [readsOn] using hle)) hx

/-- **One pass on a uniform store**: at signs of rank at least `r`, pass `j` has an unpaid candidate with at most the
one-pass family bound, the store uniform and independent of the window's coins. -/
theorem pr_pass_le {d : ℕ} (A : SketchDeployedWindow P N m ρ h S L q ks G) (r X qtot : ℕ) (hr32 : r ≤ 32)
    (hX : ∀ j η c σ O D, ((A.device j η c).prog σ D).rebuilds O D * 256 ≤ X)
    (hq : ∀ j η σ (Os : Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) (D : Data N m ρ Sector),
      ∑ c, (((A.device j η c).prog σ D).queries (Os c) D).length ≤ qtot) (j : Fin P) :
    pr (fun y : SketchStore N m × SketchWindowCoins P N m ρ h ks G d =>
      RankGood r y.2.1 ∧ ∃ c, UnpaidAt A y.2.1 (S + (L - ρ * 256) + X) j c (y.1, y.2.2.1))
      ≤ (G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h) + (qtot : ℝ≥0∞) / 2 ^ (8 * r) := by
  classical
  let e : SketchStore N m × SketchWindowCoins P N m ρ h ks G d ≃
      (SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G)) × (SketchSigns ks m × (Fin d → Fin P × Fin N)) :=
    { toFun := fun y => ((y.1, y.2.2.1), (y.2.1, y.2.2.2))
      invFun := fun z => (z.1.1, (z.2.1, (z.1.2, z.2.2)))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [Rate.pr_equiv e]
  refine Rate.pr_le_of_fiber _ _ (fun sd => ?_)
  by_cases hr : RankGood r sd.1
  · exact (pr_mono fun z hz => hz.2).trans (pr_unpaidPass_le A sd.1 _ _ j fun η =>
      Device.pr_unpaid_rank_family_le (fun c => A.device j η c) sd.1 r X qtot hr32 hr
        (fun c => hX j η c sd.1) (hq j η sd.1))
  · calc _ ≤ pr (fun _ : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G) => False) :=
          pr_mono fun _ hz => hr hz.1
      _ ≤ _ := by
          rw [Rate.Pooled.pr_false]
          exact bot_le

end SketchDeployedWindow

namespace SketchDeployedSession

variable {n P N m ρ h S L q ks G d : ℕ}

/-- **One pass of a session on a uniform store**: at window `w`'s signs of rank at least `r`, pass `j` has an unpaid
candidate with at most the one-pass family bound, the store uniform and independent of the session's coins. -/
theorem pr_pass_le (A : SketchDeployedSession n P N m ρ h S L q ks G d) (r X qtot : ℕ) (hr32 : r ≤ 32)
    (hX : ∀ w η j ι c E σ R O,
      let B := (A.window w η).device j ι c
      let D := Data.ofParts E R
      (B.prog σ D).rebuilds O D * 256 ≤ X)
    (hq : ∀ w η j ι, Accounting.Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot)
    (w : Fin n) (j : Fin P) :
    pr (fun y : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      RankGood r (y.2 w).1 ∧
        ∃ c, SketchDeployedWindow.UnpaidAt (A.win w y.2) (y.2 w).1 (S + (L - ρ * 256) + X) j c (y.1, (y.2 w).2.1))
      ≤ (G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h) + (qtot : ℝ≥0∞) / 2 ^ (8 * r) := by
  have hX' : ∀ η j ι c σ O D, (((A.window w η).device j ι c).prog σ D).rebuilds O D * 256 ≤ X := by
    intro η j ι c σ O D
    let E : SketchStore N m := fun p => D (.inl p)
    let R : SketchStream ρ := fun i => D (.inr i)
    have hD : Data.ofParts E R = D := by
      funext p
      rcases p with p | p <;> rfl
    simpa [hD] using hX w η j ι c E σ R O
  exact pr_window_le A w (fun C y => RankGood r y.2.1 ∧
      ∃ c, SketchDeployedWindow.UnpaidAt C y.2.1 (S + (L - ρ * 256) + X) j c (y.1, y.2.2.1)) _
    fun η => SketchDeployedWindow.pr_pass_le (A.window w η) r X qtot hr32 (hX' η)
      (fun j ι σ Os D => hq w η j ι σ Os D) j

/-- The rank tail of window `w`'s signs, on any `ω` drawn before the session. -/
theorem pr_rank_le {Ω : Type} [Fintype Ω] (r : ℕ) (ε : ℝ≥0∞) (hRank : Accounting.Assumptions.SketchRank ks m r ε)
    (w : Fin n) :
    pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) => ¬ RankGood r (x.2 w).1) ≤ ε := by
  classical
  rw [Rate.pr_equiv (Equiv.prodComm _ _)]
  refine Rate.pr_le_of_fiber _ _ fun _ => ?_
  rw [Rate.pr_equiv (Equiv.funSplitAt w _)]
  refine Rate.pr_le_of_fiber _ _ fun rest => ?_
  refine Rate.pr_le_of_fiber _ _ fun pd => ?_
  refine (pr_mono fun σ hσ => ?_).trans hRank
  have e1 : ((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm ((σ, pd), rest)) w = (σ, pd) := by
    simp
  simp only [Equiv.prodComm_symm, Equiv.prodComm_apply, Prod.swap, e1, RankGood] at hσ
  push Not at hσ
  exact hσ

/-- **A session's unpaid windows with a store per pass**: when every pass's store is `view` at the prefix before the
pass, a uniform store but for `δ` at each prefix, some window fails to pay for its correct receipts with probability at
most `n·ε + n·P·(G·N/2^h + qtot/2^(8r) + δ)`: the rank tail once a window, and each pass's family bound and `δ`. -/
theorem pr_unpaidOn_le {Ω : Type} [Fintype Ω] (A : SketchDeployedSession n P N m ρ h S L q ks G d)
    (view : SketchPrefix n P N m ρ h ks G d → Ω → SketchStore N m) (δ : ℝ≥0∞)
    (hKB : Accounting.UniformBut δ view) (r X qtot : ℕ) (ε : ℝ≥0∞) (hr32 : r ≤ 32)
    (hRank : Accounting.Assumptions.SketchRank ks m r ε)
    (hX : ∀ w η j ι c E σ R O,
      let B := (A.window w η).device j ι c
      let D := Data.ofParts E R
      (B.prog σ D).rebuilds O D * 256 ≤ X)
    (hq : ∀ w η j ι, Accounting.Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot) :
    pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      ∃ w, (A.win w x.2).readsOn (programmed view x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 +
          P * (S + (L - ρ * 256) + X) <
        ((A.win w x.2).correctOn (programmed view x.1 x.2 w) (x.2 w).1 (x.2 w).2.1).card * (m * 256))
      ≤ (n : ℝ≥0∞) * ε + (n * P : ℝ≥0∞) *
        ((G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h) + (qtot : ℝ≥0∞) / 2 ^ (8 * r) + δ) := by
  classical
  let pass : Fin n → Fin P → Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Prop := fun w j x =>
    RankGood r (x.2 w).1 ∧ ∃ c, SketchDeployedWindow.UnpaidAt (A.win w x.2) (x.2 w).1 (S + (L - ρ * 256) + X) j c
      (view (.of x.2 w j) x.1, (x.2 w).2.1)
  have hsub : ∀ x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d),
      (∃ w, (A.win w x.2).readsOn (programmed view x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 +
          P * (S + (L - ρ * 256) + X) <
        ((A.win w x.2).correctOn (programmed view x.1 x.2 w) (x.2 w).1 (x.2 w).2.1).card * (m * 256)) →
      (∃ w, ¬ RankGood r (x.2 w).1) ∨ ∃ w j, pass w j x := by
    rintro x ⟨w, hw⟩
    by_cases hr : RankGood r (x.2 w).1
    · obtain ⟨j, c, hc⟩ := SketchDeployedWindow.exists_unpaidOn _ _ _ _ _ hw
      exact Or.inr ⟨w, j, hr, c, hc⟩
    · exact Or.inl ⟨w, hr⟩
  let hash := (G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h) + (qtot : ℝ≥0∞) / 2 ^ (8 * r)
  have hpass : ∀ w j, pr (pass w j) ≤ hash + δ := fun w j =>
    (hKB.pr_le_of (fun x : Fin n → SketchWindowCoins P N m ρ h ks G d => SketchPrefix.of x w j)
      (fun E x => RankGood r (x w).1 ∧
        ∃ c, SketchDeployedWindow.UnpaidAt (A.win w x) (x w).1 (S + (L - ρ * 256) + X) j c (E, (x w).2.1))).trans
      (add_le_add (pr_pass_le A r X qtot hr32 hX hq w j) le_rfl)
  refine (pr_mono hsub).trans ((pr_or_le _ _).trans (add_le_add ?_ ?_))
  · refine (pr_exists_le _).trans ?_
    calc ∑ w, pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) => ¬ RankGood r (x.2 w).1)
        ≤ ∑ _w : Fin n, ε := Finset.sum_le_sum fun w _ => pr_rank_le r ε hRank w
      _ = (n : ℝ≥0∞) * ε := by simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  · refine (pr_exists_le _).trans ?_
    calc ∑ w, pr (fun x => ∃ j, pass w j x)
        ≤ ∑ w, ∑ j, pr (pass w j) := Finset.sum_le_sum fun w _ => pr_exists_le _
      _ ≤ ∑ _w : Fin n, ∑ _j : Fin P, (hash + δ) :=
          Finset.sum_le_sum fun w _ => Finset.sum_le_sum fun j _ => hpass w j
      _ = (n * P : ℝ≥0∞) * (hash + δ) := by
          simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring

/-- **The draws are causal in any correct sets that read each window's earlier windows, signs and passes**, on any `ω`
drawn before the session: both pooled-draw readings of `pr_draws_le_map`, for correct sets `cor` in place of one
store's. -/
theorem pr_draws_causal_le {Ω : Type} [Fintype Ω]
    (cor : Ω → (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin n → Finset (Fin P × Fin N))
    (hcor : ∀ o x x' w, (∀ v, v < w → x v = x' v) → (x w).1 = (x' w).1 → (x w).2.1 = (x' w).2.1 →
      cor o x w = cor o x' w) (K Kw : ℕ) :
    pr (fun z : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (z.2 w).2.2 t ∈ cor z.1 z.2 w) ∧
        (∑ w, (cor z.1 z.2 w).card ≤ K ∨ ∃ w, (cor z.1 z.2 w).card ≤ Kw))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) := by
  classical
  rw [Rate.pr_equiv (Equiv.prodComm _ _)]
  refine Rate.pr_le_of_fiber _ _ (fun o => ?_)
  let X : Fin n → Type := fun _ => SketchSigns ks m × (Fin P → SketchPassCoins N ρ h ks G)
  let e : (Fin n → SketchWindowCoins P N m ρ h ks G d) ≃
      ((w : Fin n) → X w × (Fin d → Fin (P * N))) :=
    Equiv.piCongrRight fun _ =>
      (Equiv.prodAssoc _ _ _).symm |>.trans
        ((Equiv.refl _).prodCongr (Equiv.arrowCongr (Equiv.refl _) finProdFinEquiv))
  rw [Rate.pr_equiv e]
  let Gs : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N)) :=
    fun w ω => (cor o (e.symm ω) w).map finProdFinEquiv.toEmbedding
  have hG : Rate.Causal (s := fun _ => d) Gs := by
    intro w ω ω' hlt hw
    have h1 : ∀ v, v < w → e.symm ω v = e.symm ω' v := fun v hv => by simp [e, hlt v hv]
    have hσ : (e.symm ω w).1 = (e.symm ω' w).1 := by simp [e, hw]
    have hp : (e.symm ω w).2.1 = (e.symm ω' w).2.1 := by simp [e, hw]
    simp only [Gs]
    rw [hcor o _ _ w h1 hσ hp]
  have hmem : ∀ ω w t, (e.symm ω w).2.2 t ∈ cor o (e.symm ω) w → (ω w).2 t ∈ Gs w ω := by
    intro ω w t ht
    simp only [Gs, Finset.mem_map_equiv]
    exact ht
  have hcard : ∀ ω w, (Gs w ω).card = (cor o (e.symm ω) w).card := fun ω w => Finset.card_map _
  refine (pr_mono fun ω hω => ?_).trans ((pr_or_le _ _).trans (add_le_add
    ((Rate.Pooled.pr_causal_le' n (P * N) (d * K) X (fun _ => d) Gs hG).trans (le_of_eq ?_))
    ((Rate.Pooled.pr_floor_le n (P * N) Kw d X Gs hG).trans (le_of_eq ?_))))
  · obtain ⟨hall, hK | ⟨w, hw⟩⟩ := hω
    · refine Or.inl ⟨fun w t => hmem ω w t (hall w t), ?_⟩
      show ∑ i, d * (Gs i ω).card ≤ d * K
      simp only [hcard]
      rw [← Finset.mul_sum]
      exact Nat.mul_le_mul_left d hK
    · exact Or.inr ⟨fun w t => hmem ω w t (hall w t), w, (hcard ω w).trans_le hw⟩
  · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    rcases Nat.eq_zero_or_pos d with rfl | hd
    · simp
    rw [mul_comm n d]
    congr 1
    have hd' : (d : ℝ≥0∞) ≠ 0 := by exact_mod_cast hd.ne'
    push_cast
    rw [show (P : ℝ≥0∞) * N * (d * n) = d * (n * P * N) by ring]
    exact ENNReal.mul_div_mul_left _ _ hd' (ENNReal.natCast_ne_top d)
  · simp only [Nat.cast_mul, mul_assoc]

/-- Session conservation in counts from the pooled draws and an unpaid bound `ε_u`, for causal correct sets `cor`
and reads `rd`: `pr_paid_of_unpaid_map` with them in place of one store's. -/
theorem pr_paid_of_unpaid_causal {Ω : Type} [Fintype Ω]
    (cor : Ω → (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin n → Finset (Fin P × Fin N))
    (rd : Ω → (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin n → ℕ)
    (hcor : ∀ o x x' w, (∀ v, v < w → x v = x' v) → (x w).1 = (x' w).1 → (x w).2.1 = (x' w).2.1 →
      cor o x w = cor o x' w) (slack K Kw : ℕ) (εu : ℝ≥0∞)
    (hu : pr (fun z : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      ∃ w, rd z.1 z.2 w + P * slack < (cor z.1 z.2 w).card * (m * 256)) ≤ εu) :
    pr (fun z : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (z.2 w).2.2 t ∈ cor z.1 z.2 w) ∧
        (∑ w, rd z.1 z.2 w + n * (P * slack) < (K + 1) * (m * 256) ∨
          ∃ w, rd z.1 z.2 w + P * slack < (Kw + 1) * (m * 256)))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) + εu := by
  have h₁ := pr_draws_causal_le cor hcor K Kw
  refine (pr_mono (fun x hx => ?_)).trans ((pr_or_le _ _).trans (add_le_add h₁ hu))
  obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx
  · by_cases hD : ∑ w, (cor x.1 x.2 w).card ≤ K ∨ ∃ w, (cor x.1 x.2 w).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    refine Or.inr ?_
    push Not at hD
    by_contra hno
    push Not at hno
    have hle := Finset.sum_le_sum fun w (_ : w ∈ Finset.univ) => hno w
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
      ← Finset.sum_mul] at hle
    have hK : (K + 1) * (m * 256) ≤ (∑ w, (cor x.1 x.2 w).card) * (m * 256) :=
      Nat.mul_le_mul_right _ hD.1
    omega
  · by_cases hD : ∑ w, (cor x.1 x.2 w).card ≤ K ∨ ∃ w, (cor x.1 x.2 w).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    push Not at hD
    exact Or.inr ⟨w, hw.trans_le (Nat.mul_le_mul_right _ (hD.2 w))⟩

/-- `pr_conservation_of_unpaid` for causal correct sets `cor` and reads `rd`. -/
theorem pr_conservation_of_unpaid_causal {Ω : Type} [Fintype Ω]
    (cor : Ω → (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin n → Finset (Fin P × Fin N))
    (rd : Ω → (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin n → ℕ)
    (hcor : ∀ o x x' w, (∀ v, v < w → x v = x' v) → (x w).1 = (x' w).1 → (x w).2.1 = (x' w).2.1 →
      cor o x w = cor o x' w) (X : ℕ) (dbar dw : ℝ) (hn : 0 < n) (hP : 0 < P)
    (hN : 0 < N) (hm : 0 < m) (hdw : dw ≤ 1) (εu : ℝ≥0∞)
    (hu : pr (fun z : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      ∃ w, rd z.1 z.2 w + P * (S + (L - ρ * 256) + X) < (cor z.1 z.2 w).card * (m * 256)) ≤ εu) :
    pr (fun z : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (z.2 w).2.2 t ∈ cor z.1 z.2 w) ∧
        ((((∑ w, rd z.1 z.2 w : ℕ)) : ℝ) < (1 - (dbar + gammaSketch N m ρ S L X)) * (n * P * N * m * 256) ∨
          ∃ w, ((rd z.1 z.2 w : ℕ) : ℝ) < (1 - (dw + gammaSketch N m ρ S L X)) * (P * N * m * 256)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) + εu := by
  let slack := S + (L - ρ * 256) + X
  have main := pr_paid_of_unpaid_causal cor rd hcor slack
    ⌊(1 - dbar) * ((n : ℝ) * P * N)⌋₊ ⌊(1 - dw) * ((P : ℝ) * N)⌋₊ εu hu
  have hγ := gammaSketch_mul N m ρ S L X hN hm
  have hγ' : gammaSketch N m ρ S L X * ((N : ℝ) * (m * 256)) = (slack : ℝ) := by
    simpa [slack, mul_assoc] using hγ
  have hb : (0 : ℝ) < m * 256 := by positivity
  refine (pr_mono (fun x hx => ?_)).trans (main.trans ?_)
  · obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx
    · refine ⟨hall, Or.inl ?_⟩
      have hc := Rate.Deployed.count_lt
        (R := ((∑ w, rd x.1 x.2 w : ℕ) : ℝ))
        (t := dbar) (g := gammaSketch N m ρ S L X) (c := (n : ℝ) * P)
        (Nn := N) (b := (m : ℝ) * 256) (sl := slack) hb hγ'
        (by simpa [mul_assoc] using hT)
      rw [← mul_assoc]
      exact_mod_cast hc
    · refine ⟨hall, Or.inr ⟨w, ?_⟩⟩
      have hc := Rate.Deployed.count_lt
        (R := ((rd x.1 x.2 w : ℕ) : ℝ))
        (t := dw) (g := gammaSketch N m ρ S L X) (c := (P : ℝ))
        (Nn := N) (b := (m : ℝ) * 256) (sl := slack) hb hγ'
        (by simpa [mul_assoc] using hw)
      exact_mod_cast hc
  · have e1 : ((n : ℝ) * P * N) = ((n * P * N : ℕ) : ℝ) := by push_cast; ring
    have e2 : ((P : ℝ) * N) = ((P * N : ℕ) : ℝ) := by push_cast; ring
    rw [e1, e2]
    refine add_le_add (add_le_add ?_ ?_) le_rfl
    · rw [show ((n : ℝ≥0∞) * P * N) = ((n * P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
      exact Rate.Deployed.pooled_le dbar (n * P * N) (d * n) (by positivity)
    · rw [show ((n : ℝ≥0∞) * P * N) = (n : ℝ≥0∞) * ((P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
      exact Rate.Deployed.window_le dw n (P * N) (d * n) hn (by positivity) hdw

/-- `pr_deployed_of_unpaid` for causal correct sets `cor` and reads `rd`: the draws' terms are at most `2·2^-40`. -/
theorem pr_deployed_of_unpaid_causal {Ω : Type} [Fintype Ω]
    (cor : Ω → (Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) → Fin n → Finset (Fin P × Fin N))
    (rd : Ω → (Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) → Fin n → ℕ)
    (hcor : ∀ o x x' w, (∀ v, v < w → x v = x' v) → (x w).1 = (x' w).1 → (x w).2.1 = (x' w).2.1 →
      cor o x w = cor o x' w) (X : ℕ) (hn : 7 ≤ n) (hP : 0 < P) (hN : 0 < N) (εu : ℝ≥0∞)
    (hu : pr (fun z : Ω × (Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      ∃ w, rd z.1 z.2 w + P * (S + (L - ρ * 256) + X) < (cor z.1 z.2 w).card * (128 * 256)) ≤ εu) :
    pr (fun z : Ω × (Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      (∀ w t, (z.2 w).2.2 t ∈ cor z.1 z.2 w) ∧
        ((((∑ w, rd z.1 z.2 w : ℕ)) : ℝ) < (1 - (0.0015 + gammaSketch N 128 ρ S L X)) * (n * P * N * 32768) ∨
          ∃ w, ((rd z.1 z.2 w : ℕ) : ℝ) < (1 - (0.01 + gammaSketch N 128 ρ S L X)) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + εu := by
  have h := pr_conservation_of_unpaid_causal (S := S) (L := L) cor rd hcor X 0.0015 0.01 (by omega) hP hN
    (by norm_num) (by norm_num) εu hu
  refine (le_of_eq ?_).trans (h.trans ?_)
  · norm_num [mul_assoc]
  rw [show (2 : ℝ≥0∞) / 2 ^ 40 = 1 / 2 ^ 40 + 1 / 2 ^ 40 by
    rw [ENNReal.div_add_div_same]; norm_num]
  gcongr
  · exact Rate.Deployed.pooled_deployed n hn
  · exact Rate.Deployed.window_deployed n (by omega)

end SketchDeployedSession

end Accounting.RateSketch

namespace Accounting.Requirement

variable {V : Type} [DecidableEq V] (W : Wiring V) (e : Execution V) (ℓ : Store.Placement V)

/-- Over a window of at most `k` steps at each of which the store holds `W`'s credited blocks with at most `N` resident
blocks and the plan covers `W`'s reads in at most `N` blocks, the live bits the audit credits and the bits read from
complete blocks total at most `k·N·B` each. -/
theorem cap_of_matches (s : Store) (plan : ℕ → Finset ℕ) (B N k : ℕ) (win : Window) (hwin : win.len ≤ k)
    (hm : ∀ t ∈ win.steps, StoreMatches W e ℓ s B t ∧ (s.resident t).card ≤ N ∧
      PlanMatches W e ℓ (plan t) B t ∧ (plan t).card ≤ N) :
    ∑ t ∈ win.steps, creditedBits W e ℓ t ≤ k * N * B ∧ ∑ t ∈ win.steps, creditedReadBits W e ℓ t ≤ k * N * B := by
  have hsum : ∑ _t ∈ win.steps, N * B ≤ k * N * B := by
    rw [Finset.sum_const, Window.steps, Nat.card_Ico, smul_eq_mul, Nat.add_sub_cancel_left, mul_assoc]
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

/-- **The remote audit at the deployed parameters with a store per pass, with stacking**,
`Accounting.Guarantees.StackedSessionConservationSketchDeployed`. -/
theorem StackedSessionConservationSketchDeployed :
    Accounting.Guarantees.StackedSessionConservationSketchDeployed := by
  intro n P N ρ S L q G X qtot Rin cρ reads life E es B εnum εden T M ε δ Ω Ca Ch _ _ enc choose slot A hn hP hN
    hE0 hrank hKB hSq hCs hEle hε hB hK C
  have hE : Gates.EncodeBound Gates.p1024 3 B E εnum εden :=
    SecurityProofs.Gates.encodeBound_of hSq hCs hEle hε (by decide) SecurityProofs.Gates.p1024_floor.1
      SecurityProofs.Gates.p1024_floor.2
  have hX : ∀ w η j ι c Es σ R O,
      let dv := (A.window w η).device j ι c
      let D := RateSketch.Data.ofParts Es R
      (dv.prog σ D).rebuilds O D * 256 ≤ X + C := by
    intro w η j ι c Es σ R O
    obtain ⟨K, hS, hKb⟩ := hK w η j ι c σ
    have h1 := hS Es R O
    have h2 := K.card_le_of_budget hE (by decide) hE0 hKb
    dsimp only at h1 ⊢
    calc _ ≤ X + K.carried.card * 256 := h1
      _ ≤ X + C := Nat.add_le_add_left h2 X
  let view := coinStore enc choose slot
  have hcor : ∀ (o : Ω × (Fin T → Ca)) (x x' : Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) (w : Fin n),
      (∀ v, v < w → x v = x' v) → (x w).1 = (x' w).1 → (x w).2.1 = (x' w).2.1 →
      (A.win w x).correctOn (programmed view o x w) (x w).1 (x w).2.1 =
        (A.win w x').correctOn (programmed view o x' w) (x' w).1 (x' w).2.1 := by
    intro o x x' w hlt hσ hp
    have hη : (fun v : {v : Fin n // v < w} => x v.1) = fun v => x' v.1 := funext fun v => hlt v.1 v.2
    have hwin : A.win w x = A.win w x' := by
      simp only [SketchDeployedSession.win]
      rw [hη]
    have hpre : ∀ j, SketchPrefix.of x w j = SketchPrefix.of x' w j := fun j => by
      simp only [SketchPrefix.of]
      rw [hη, hσ, hp]
    have hprog : programmed view o x w = programmed view o x' w := funext fun j => by
      simp only [programmed]
      rw [hpre j]
    rw [hwin, hprog, hσ, hp]
  have hu := SketchDeployedSession.pr_unpaidOn_le A view δ (hKB.uniformBut_coinStore choose slot) 12 (X + C) qtot ε
    (by norm_num) hrank hX hB
  refine (SketchDeployedSession.pr_deployed_of_unpaid_causal
    (fun o x w => (A.win w x).correctOn (programmed view o x w) (x w).1 (x w).2.1)
    (fun o x w => (A.win w x).readsOn (programmed view o x w) (x w).1 (x w).2.1)
    hcor (X + C) hn hP hN _ hu).trans (le_of_eq ?_)
  rw [show 8 * 12 = 96 from rfl]
  ring

/-- **The remote audit with relaying side sources on a programmed store, with stacking**,
`Accounting.Guarantees.StackedSessionConservationRelay`. -/
theorem StackedSessionConservationRelay : Accounting.Guarantees.StackedSessionConservationRelay := by
  intro n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E T M cost Fc k B es εnum εden tier
    δ Ω Ca Ch _ _ enc choose slot A dbar dw hn hP hN hb hrc hdw hE0 hbytes hKB hW hF hSq hCs hEle hε hk hp hp4 hK C
  have hE : Gates.EncodeBound Fc k B E εnum εden := SecurityProofs.Gates.encodeBound_of hSq hCs hEle hε hk hp hp4
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
  have hu := (hKB.uniformBut_blocks choose slot).pr_le ()
    fun (Es : Rate.Payloads N b) (ω : Fin n → Rate.WindowCoins P N b h ρ r s d) =>
      ∃ w, ((A.stack L C K hL).win w ω).reads tier Es (ω w).1 +
          P * (S + (L + C - ρ) + Rate.Deployed.rebuildBits b rounds cRoot F + (lam + m * b)) <
        (((A.stack L C K hL).win w ω).correctAll Es (ω w).1).card * b
  have hunif := Rate.DeployedSession.pr_unpaid_le (A.stack L C K hL) tier _ _
    (Remote.window_relay_le (lam := lam) (m := m) (A.stack L C K hL) tier hrc hW
      (fun w η j ι c => Rate.PIDevice.stack_ceiling _ L C (K w η j ι c) (hL w η j ι c) (hF w η j ι c)))
  have base := Remote.session_relay_of_unpaid (fun y => appendCoinBlocks enc choose slot y) (A.stack L C K hL) tier
    lam m rounds cRoot F dbar dw hn hP hN hb hdw _ (hu.trans (add_le_add hunif le_rfl))
  refine ((Bandwidth.pr_mono fun y hy => ?_).trans base).trans (le_of_eq ?_)
  · have e := fun w => Rate.DeployedSession.stack_win A L C K hL hside hC tier w y.2
      (appendCoinBlocks enc choose slot y.1) (y.2 w).1
    simp only [fun w => (e w).1, fun w => (e w).2]
    exact hy
  · simp only [div_eq_mul_inv]
    ring

/-- **The stacked remote audit at the deployed parameters, read against a declared circuit's requirement**,
`Accounting.Guarantees.StackedRequirementSketchDeployed`. -/
theorem StackedRequirementSketchDeployed : Accounting.Guarantees.StackedRequirementSketchDeployed := by
  intro n P N ρ S L q G X qtot Rin cρ reads life E es B εnum εden T M ε δ Ω Ca Ch _ _ enc choose slot A V _ W e ℓ mm
    plan win hn hP hN hE0 hrank hKB hSq hCs hEle hε hB hK hwin hm C γ
  have hcap := Requirement.cap_of_matches W e ℓ mm plan 32768 N (n * P) win hwin hm
  have base := StackedSessionConservationSketchDeployed n P N ρ S L q G X qtot Rin cρ reads life E es B εnum εden
    T M ε δ enc choose slot A hn hP hN hE0 hrank hKB hSq hCs hEle hε hB hK
  dsimp only at base
  refine (Bandwidth.pr_mono fun x hx => ⟨hx.1, Or.inl ?_⟩).trans base
  have h := Requirement.lt_of_cap hcap.1 hcap.2 hx.2
  simp only [Nat.cast_mul, Nat.cast_ofNat] at h
  exact h

/-- **The stacked remote audit with relaying side sources, read against a declared circuit's requirement**,
`Accounting.Guarantees.StackedRequirementRelay`. -/
theorem StackedRequirementRelay : Accounting.Guarantees.StackedRequirementRelay := by
  intro n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E T M cost Fc k B es εnum εden tier
    δ Ω Ca Ch _ _ enc choose slot A dbar dw V _ W e ℓ mm plan win hn hP hN hb hrc hdw hE0 hbytes hKB hW hF hSq hCs
    hEle hε hk hp hp4 hK hwin hm C γ
  have hcap := Requirement.cap_of_matches W e ℓ mm plan b N (n * P) win hwin hm
  have base := StackedSessionConservationRelay n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life
    E T M cost Fc k B es εnum εden tier δ enc choose slot A dbar dw hn hP hN hb hrc hdw hE0 hbytes hKB hW hF hSq hCs
    hEle hε hk hp hp4 hK
  dsimp only at base
  refine (Bandwidth.pr_mono fun x hx => ⟨hx.1, Or.inl ?_⟩).trans base
  have h := Requirement.lt_of_cap hcap.1 hcap.2 hx.2
  simp only [Nat.cast_mul] at h
  exact h

end Accounting.SecurityProofs
