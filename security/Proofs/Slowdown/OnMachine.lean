module

public import Proofs.Slowdown.CutCharged
public import Proofs.ResourceAccounting.Actual.Outside
public import Proofs.ResourceAccounting.Actual.Device

/-!
# Training work on a machine whose work is mostly used by inference blocks

The registered circuit runs at a location `X` over `[0, t)` (an actual circuit, `ResourceAccounting.Placed`), and the
blocks `G` of a partition, each reading at most `k` gates outside `θ`, do at least `x` of its work. At a pricing of at
least one a gate, the gates of the other blocks number at most what the location has left, `r = Φ_X t − x`
(`ResourceAccounting.SecurityProofs.Actual.gates_outside`), and they hold whole at most `r / γ` MACs of at least `γ`
gates each. With `HECharged`'s count for the blocks of `G`: `γ N ≤ r + γ (T' + ½ √(S O / q) + ½ √(S T' / (q b)))`
(`on_machine`).

`on_machine_priced` is the same in the machine's own units: at a pricing where each MAC costs at least one unit, `N ≤ r + (T' + …)`, with
no floor on a gate's price or a MAC's gates.

`on_device` reads `x` off the device model: when the blocks of `G` hold every gate of an account `a` and the pricing is
`p` on each op class, `x` is any amount at most `a`'s work in the schedule the run induces
(`ResourceAccounting.SecurityProofs.Actual.covers_work`), which is what the compute audit's accept certifies.
-/

@[expose] public section

namespace Slowdown.Cut.SecurityProofs

open Verity Finset InferenceOnly ResourceAccounting

/-- The gates of the blocks outside `G` are computed gates outside the blocks of `G`. -/
theorem card_rest_le {C : Circuit} {m : ℕ} (P : Partition C m) (G : Finset (Fin m)) :
    ∑ u ∈ Gᶜ, (P.gates u).card ≤ C.size (C.computed \ G.biUnion P.gates) := by
  have hd : ∀ u ∈ Gᶜ, ∀ v ∈ Gᶜ, u ≠ v → Disjoint (P.gates u) (P.gates v) := fun u _ v _ huv =>
    disjoint_left.mpr fun g hu hv => huv (Option.some.inj ((mem_filter.mp hu).2.symm.trans (mem_filter.mp hv).2))
  rw [← card_biUnion hd]
  refine card_le_card fun g hg => ?_
  obtain ⟨u, hu, hgu⟩ := mem_biUnion.mp hg
  have hunit := (mem_filter.mp hgu).2
  refine mem_sdiff.mpr ⟨mem_filter.mpr ⟨mem_univ g, ?_⟩, fun hS => ?_⟩
  · by_contra hin
    rw [(P.input_iff g).mpr (by simpa using hin)] at hunit
    exact absurd hunit (by simp)
  · obtain ⟨v, hv, hgv⟩ := mem_biUnion.mp hS
    have : u = v := Option.some.inj (hunit.symm.trans (mem_filter.mp hgv).2)
    exact (mem_compl.mp hu) (this ▸ hv)

/-- **Training work on a machine.** The registered circuit, placed by `px`, `pt` and `pcell`, runs at the location `X`
over `[0, t)` at compute rate `Φ_X`, at a pricing of at least one a gate, and the blocks of `G` do at least `x` of the work. Under `HECharged`'s
premises, with MACs of at least `γ` gates, `γ N` is at most what the location has left, `Φ_X t − x`, and `γ` times the
count for the blocks of `G`. -/
theorem on_machine {ι V : Type} {L : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit)
    (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) (k m γ T' S : ℕ) (P : Partition d.D m) (G : Finset (Fin m))
    (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k)
    (hE : Assumptions.Exports W b) (hR : ReadsOn d P G k T' S) (hγ : ∀ s, γ ≤ (W.mac s).card)
    (px : Fin d.D.N → L) (pt pcell : Fin d.D.N → ℕ) (pinj : Function.Injective fun g => (px g, pcell g, pt g))
    {𝒳 : Set (Finset L)} {π : Fin d.D.N → ℕ} {compute : Finset L → ℕ}
    (hrate : (⟨d.D, px, pt, pcell, pinj⟩ : Placed L).ComputeRate 𝒳 π compute)
    (hπ : ∀ g, d.D.isInput g = false → 1 ≤ π g) {X : Finset L} (hX : X ∈ 𝒳) (t : ℕ)
    (hall : ∀ g, d.D.isInput g = false → px g ∈ X ∧ pt g < t) {x : ℕ}
    (hx' : x ≤ ∑ g ∈ d.D.computed ∩ G.biUnion P.gates, π g) :
    (γ : ℝ) * W.N ≤ ((compute X * t - x : ℕ) : ℝ) + γ * workHEOn T' S (βw * βx) b W.O := by
  set A : Placed L := ⟨d.D, px, pt, pcell, pinj⟩
  have hreg : A.region X (Set.Ico 0 t) = d.D.computed := by
    ext g
    rw [ResourceAccounting.SecurityProofs.Actual.mem_region A]
    simp only [Circuit.computed, mem_filter, mem_univ, true_and, Set.mem_Ico, Nat.zero_le]
    exact ⟨fun h => h.1, fun h => ⟨h, hall g h⟩⟩
  have hx'' : x ≤ A.work π (A.region X (Set.Ico 0 t) ∩ G.biUnion P.gates) := by
    rw [hreg]
    exact hx'
  have hout := ResourceAccounting.SecurityProofs.Actual.gates_outside A hrate hπ hX 0 t (G.biUnion P.gates) hx''
  rw [hreg, Nat.sub_zero] at hout
  have hN := (HECharged ι V d W val k m 0 γ T' S P G βw βx b hw hx hb hval hshare hH hE hR).1
  have hin : (∑ u ∈ Gᶜ, (W.inside (P.gates u)).card) * γ ≤ compute X * t - x := by
    rw [sum_mul]
    exact ((sum_le_sum fun u _ => card_inside_mul_le W (P.gates u) hγ).trans (card_rest_le P G)).trans hout
  have hin' : ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) * γ ≤ ((compute X * t - x : ℕ) : ℝ) := by
    exact_mod_cast hin
  have hγ0 : (0 : ℝ) ≤ γ := Nat.cast_nonneg γ
  nlinarith [mul_le_mul_of_nonneg_left hN hγ0]

/-- The blocks outside `G` cost at most the computed gates outside the blocks of `G`, at any pricing. -/
theorem sum_rest_le {C : Circuit} {m : ℕ} (P : Partition C m) (G : Finset (Fin m)) (π : Fin C.N → ℕ) :
    ∑ u ∈ Gᶜ, ∑ g ∈ P.gates u, π g ≤ ∑ g ∈ C.computed \ G.biUnion P.gates, π g := by
  have hd : ∀ u ∈ Gᶜ, ∀ v ∈ Gᶜ, u ≠ v → Disjoint (P.gates u) (P.gates v) := fun u _ v _ huv =>
    disjoint_left.mpr fun g hu hv => huv (Option.some.inj ((mem_filter.mp hu).2.symm.trans (mem_filter.mp hv).2))
  rw [← sum_biUnion hd]
  refine sum_le_sum_of_subset fun g hg => ?_
  obtain ⟨u, hu, hgu⟩ := mem_biUnion.mp hg
  have hunit := (mem_filter.mp hgu).2
  refine mem_sdiff.mpr ⟨mem_filter.mpr ⟨mem_univ g, ?_⟩, fun hS => ?_⟩
  · by_contra hin
    rw [(P.input_iff g).mpr (by simpa using hin)] at hunit
    exact absurd hunit (by simp)
  · obtain ⟨v, hv, hgv⟩ := mem_biUnion.mp hS
    have : u = v := Option.some.inj (hunit.symm.trans (mem_filter.mp hgv).2)
    exact (mem_compl.mp hu) (this ▸ hv)

/-- A set `A` of gates holds whole at most as many MACs as it costs, at a pricing where each MAC costs at least one
unit: MACs are disjoint. -/
theorem card_inside_le_cost {C : Circuit} {ι : Type} [Fintype ι] (W : Work C ι) (A : Finset (Fin C.N))
    {π : Fin C.N → ℕ} (hmac : ∀ s, 1 ≤ ∑ g ∈ W.mac s, π g) : (W.inside A).card ≤ ∑ g ∈ A, π g :=
  calc (W.inside A).card = ∑ _s ∈ W.inside A, 1 := by simp
    _ ≤ ∑ s ∈ W.inside A, ∑ g ∈ W.mac s, π g := sum_le_sum fun s _ => hmac s
    _ = ∑ g ∈ (W.inside A).biUnion W.mac, π g := (sum_biUnion fun s _ t _ hst => W.mac_disjoint s t hst).symm
    _ ≤ ∑ g ∈ A, π g := sum_le_sum_of_subset (biUnion_subset.2 fun _ hs => (mem_filter.1 hs).2)

/-- **Training work on a machine, in the machine's own units.** As `on_machine`, at any pricing where each MAC costs at
least one unit (a tensor unit, with `Φ_X` the location's tensor rate): `N` is at most what the location has left,
`Φ_X t − x`, and the count for the blocks of `G`. No gate need cost anything, and no MAC need have many gates. -/
theorem on_machine_priced {ι V : Type} {L : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit)
    (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) (k m T' S : ℕ) (P : Partition d.D m) (G : Finset (Fin m))
    (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k)
    (hE : Assumptions.Exports W b) (hR : ReadsOn d P G k T' S)
    (px : Fin d.D.N → L) (pt pcell : Fin d.D.N → ℕ) (pinj : Function.Injective fun g => (px g, pcell g, pt g))
    {𝒳 : Set (Finset L)} {π : Fin d.D.N → ℕ} {compute : Finset L → ℕ}
    (hrate : (⟨d.D, px, pt, pcell, pinj⟩ : Placed L).ComputeRate 𝒳 π compute)
    (hmac : ∀ s, 1 ≤ ∑ g ∈ W.mac s, π g) {X : Finset L} (hX : X ∈ 𝒳) (t : ℕ)
    (hall : ∀ g, d.D.isInput g = false → px g ∈ X ∧ pt g < t) {x : ℕ}
    (hx' : x ≤ ∑ g ∈ d.D.computed ∩ G.biUnion P.gates, π g) :
    (W.N : ℝ) ≤ ((compute X * t - x : ℕ) : ℝ) + workHEOn T' S (βw * βx) b W.O := by
  set A : Placed L := ⟨d.D, px, pt, pcell, pinj⟩
  have hreg : A.region X (Set.Ico 0 t) = d.D.computed := by
    ext g
    rw [ResourceAccounting.SecurityProofs.Actual.mem_region A]
    simp only [Circuit.computed, mem_filter, mem_univ, true_and, Set.mem_Ico, Nat.zero_le]
    exact ⟨fun h => h.1, fun h => ⟨h, hall g h⟩⟩
  have hx'' : x ≤ A.work π (A.region X (Set.Ico 0 t) ∩ G.biUnion P.gates) := by
    rw [hreg]
    exact hx'
  have hout := ResourceAccounting.SecurityProofs.Actual.work_outside A hrate hX 0 t (G.biUnion P.gates) hx''
  rw [hreg, Nat.sub_zero] at hout
  have hN := (HECharged ι V d W val k m 0 0 T' S P G βw βx b hw hx hb hval hshare hH hE hR).1
  have hin : ∑ u ∈ Gᶜ, (W.inside (P.gates u)).card ≤ compute X * t - x :=
    ((sum_le_sum fun u _ => card_inside_le_cost W (P.gates u) hmac).trans (sum_rest_le P G π)).trans hout
  have hin' : ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) ≤ ((compute X * t - x : ℕ) : ℝ) := by
    exact_mod_cast hin
  linarith

/-- `on_machine_priced` with the work of the blocks of `G` read off an account: every gate charged to `a` is in a block of
`G`, and `x` is at most `a`'s work over the device's gates in `[0, t)` at the pricing `π'`. -/
theorem on_machine_acct {ι V α : Type} {L : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit)
    (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) (k m T' S : ℕ) (P : Partition d.D m) (G : Finset (Fin m))
    (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k)
    (hE : Assumptions.Exports W b) (hR : ReadsOn d P G k T' S)
    (px : Fin d.D.N → L) (pt pcell : Fin d.D.N → ℕ) (pinj : Function.Injective fun g => (px g, pcell g, pt g))
    {𝒳 : Set (Finset L)} {π : Fin d.D.N → ℕ} {compute : Finset L → ℕ}
    (hrate : (⟨d.D, px, pt, pcell, pinj⟩ : Placed L).ComputeRate 𝒳 π compute)
    (hmac : ∀ s, 1 ≤ ∑ g ∈ W.mac s, π g) {X : Finset L} (hX : X ∈ 𝒳) (t : ℕ)
    (hall : ∀ g, d.D.isInput g = false → px g ∈ X ∧ pt g < t) {x : ℕ}
    (A : Placed L) (hA : A = ⟨d.D, px, pt, pcell, pinj⟩) (π' : Fin A.N → ℕ) (acct : Fin A.N → α) (a : α)
    [DecidablePred fun g => acct g = a]
    (hπ' : ∀ g, π' g = π (Fin.cast (by rw [hA]) g))
    (hacct : ∀ g, acct g = a → Fin.cast (by rw [hA]) g ∈ G.biUnion P.gates)
    (hW : x ≤ A.work π' ((A.region X (Set.Ico 0 (0 + t))).filter fun g => acct g = a)) :
    (W.N : ℝ) ≤ ((compute X * t - x : ℕ) : ℝ) + workHEOn T' S (βw * βx) b W.O := by
  subst hA
  refine on_machine_priced d W val k m T' S P G βw βx b hw hx hb hval hshare hH hE hR px pt pcell pinj hrate hmac hX t
    hall (hW.trans ?_)
  unfold Placed.work
  rw [sum_congr rfl fun g _ => hπ' g]
  refine sum_le_sum_of_subset fun g hg => ?_
  obtain ⟨hg, ha⟩ := mem_filter.mp hg
  obtain ⟨hc, -, -⟩ := (ResourceAccounting.SecurityProofs.Actual.mem_region _).1 hg
  exact mem_inter.mpr ⟨mem_filter.mpr ⟨mem_univ _, hc⟩, hacct g ha⟩

open Classical in
/-- **Training work on a device whose compute account is the inference blocks.** The run `R` is the registered circuit
placed at the device `chip ∪ mem` over `[0, t)`, every gate charged to the account `a` is in a block of `G`, and `a`'s
work in the schedule the run induces is at least `x` at the price `p`, at which each MAC costs at least one unit. Then
`N` is at most what the device has left, `Φ t − x`, and `HECharged`'s count for the blocks of `G`. -/
theorem on_device {ι V α ι' E : Type} {L : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq V] [DecidableEq α]
    (d : RegisteredCircuit)
    (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) (k m T' S : ℕ) (P : Partition d.D m) (G : Finset (Fin m))
    (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k)
    (hE : Assumptions.Exports W b) (hR : ReadsOn d P G k T' S)
    (px : Fin d.D.N → L) (pt pcell : Fin d.D.N → ℕ) (pinj : Function.Injective fun g => (px g, pcell g, pt g))
    {𝒳 : Set (Finset L)} {π : Fin d.D.N → ℕ} {compute : Finset L → ℕ}
    (hrate : (⟨d.D, px, pt, pcell, pinj⟩ : Placed L).ComputeRate 𝒳 π compute)
    (hmac : ∀ s, 1 ≤ ∑ g ∈ W.mac s, π g) (chip mem : Finset L) (hX : chip ∪ mem ∈ 𝒳) (t : ℕ)
    (hall : ∀ g, d.D.isInput g = false → px g ∈ chip ∪ mem ∧ pt g < t) {x : ℕ}
    (R : Actual.Run L) (hA : R.A = ⟨d.D, px, pt, pcell, pinj⟩) (cls : Fin R.A.N → Option OpClass)
    (acct : Fin R.A.N → α) (link : Fin R.A.N → Fin R.A.N → E) (hbm : E × L) (side : ι' → E × L) (p : Price) (a : α)
    (hcls : ∀ g, (cls g).elim 0 p = π (Fin.cast (by rw [hA]) g))
    (hacct : ∀ g, acct g = a → Fin.cast (by rw [hA]) g ∈ G.biUnion P.gates)
    (hW : x ≤ (R.schedule chip mem cls acct link hbm side).work ⟨0, t⟩ a p) :
    (W.N : ℝ) ≤ ((compute (chip ∪ mem) * t - x : ℕ) : ℝ) + workHEOn T' S (βw * βx) b W.O := by
  rw [← ResourceAccounting.SecurityProofs.Actual.covers_work R chip mem cls acct link hbm side
    (π := fun g => (cls g).elim 0 p) (fun _ _ => rfl) ⟨0, t⟩ a] at hW
  exact on_machine_acct d W val k m T' S P G βw βx b hw hx hb hval hshare hH hE hR px pt pcell pinj hrate hmac hX t
    hall R.A hA _ acct a hcls hacct hW

end Slowdown.Cut.SecurityProofs
