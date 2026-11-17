module

public import Proofs.Flock.Bridge.V5.OpeningsCanon
public import Proofs.Flock.Bridge.V5.Transports
public import Proofs.Flock.Bridge.QsRows
public import Proofs.Flock.Bridge.QsRows.FirstReads

/-! `Q_s` at public inputs, by the rows and public gates a sample reads, at any sampling design on strata. -/

@[expose] public section

namespace Flock.Bridge

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open scoped ENNReal

section QsRowsP

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge
  FlockSoundness.Discharge.Composed

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} {own : Option ByteArray} {dj : Finset (Fin n) → Lean.Json}
  {S₀ : Finset (Fin n)} {d₀ : Lean.Json} {y₀ : SampleSetupZHJP (inputsJ I 0) own mPts S₀ d₀}
  {ht : I.tags.typed = false} {c : Flock.Circuit} {hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c}
  {hscope : Discharge.Layout.scopeOkH c = true} {hU : Built c.unit} {pi : Flock.HmIn.Ports}
  {hpi : Flock.HmIn.parse c = .ok pi} {pub : Flock.Public}
  {hpub : Flock.HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub}
  {g : Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N}

theorem vbP_pos_none (h : HiddenExec.rowOfHP c hU n pi (Discharge.Layout.outsOfP c hU n pi) g = none) :
    (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).pos g = (Sum.inr g : PosHP c hU n pi) := by
  dsimp only [vbP, HmRowsZX.bindingZ, LayoutZ.bindingZ, rsZHJP, ZkHidden.layZHJP, LayoutZ.ofShadow,
    ZkHidden.layShHJP, HiddenExec.keyLayoutHP, Discharge.Layout.Readers.layout, Discharge.Layout.ofBits,
    HiddenExec.keyReadersHP]
  exact Discharge.Layout.ofRows_pos_none _ h

theorem vbP_pos_io (u : Fin n)
    (hg : g ∈ ((keyProgP c pi hU n).partition (Discharge.Layout.outsOfP c hU n pi)).io u) :
    (∃ q : Fin c.ports.size,
        (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).pos g = (Sum.inl (u, q) : PosHP c hU n pi)) ∨
      (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).pos g =
        (Sum.inr (Discharge.Layout.zeroG (c := c) (hU := hU) (n := n)) : PosHP c hU n pi) ∨
      (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).pos g =
        (Sum.inr (Discharge.Layout.oneG (c := c) (hU := hU) (n := n)) : PosHP c hU n pi) ∨
      ((vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).pos g = (Sum.inr g : PosHP c hU n pi) ∧
        ∃ q t, Discharge.Layout.keyPubOf c hU n pi g = some (u, q, t)) := by
  obtain ⟨x, hx⟩ := Discharge.Layout.exists_col_of_io
    (Discharge.Const.Prog.instC (keyProgP c pi hU n) (Discharge.Layout.outsOfP c hU n pi) u) hg
  rcases hr : HiddenExec.rowOfHP c hU n pi (Discharge.Layout.outsOfP c hU n pi) g with _ | ⟨u', q, k⟩
  · right
    rw [vbP_pos_none hr]
    by_cases h0 : g = Discharge.Layout.zeroG (c := c) (hU := hU) (n := n)
    · exact Or.inl (by rw [h0])
    by_cases h1 : g = Discharge.Layout.oneG (c := c) (hU := hU) (n := n)
    · exact Or.inr (Or.inl (by rw [h1]))
    rcases hp : Discharge.Layout.keyPubOf c hU n pi g with _ | ⟨u', q, t⟩
    · exact absurd hr (HiddenExec.rowOfHP_ne_none_of_io (Reg := Unit) (R := ()) (S := {u})
        (sitesShHJP_parse (Reg := Unit) dj y₀ ht hc hscope) hpi (Discharge.Layout.outsOfP c hU n pi)
        (Discharge.Layout.scopeOkH_spec hU hscope).2.1
        (HiddenExec.rowsFit_of_stmt (FlockSoundness.Discharge.ZkBind.zkTyped ht) hc hscope)
        (Discharge.Layout.outsOutP_outsOfP c hU n pi) (keysRowP_of_scopeZ ht hc hscope hU)
        (Finset.mem_singleton_self u) hg h1 h0 hp)
    · subst hx
      obtain ⟨rfl, -, -⟩ := Discharge.Layout.keyPubOf_wire (Discharge.Layout.outsOfP c hU n pi) x hp
      exact Or.inr (Or.inr ⟨rfl, q, t, rfl⟩)
  · left
    subst hx
    have hu := HiddenExec.rowOfHP_wire_unit (Discharge.Layout.outsOfP c hU n pi) x hr
    rw [hu] at hr
    exact ⟨q, vbP_pos_row hr⟩

open Classical in
noncomputable def pubBits (c : Flock.Circuit) (hU : Built c.unit) (pi : Flock.HmIn.Ports) : ℕ :=
  ((Finset.range (Discharge.Layout.keyB c hU)).filter fun b =>
    ∃ q t, Discharge.Aliased.srcKeyP c pi b = .pub q t).card

open Classical in
theorem card_pubGates_le (u : Fin n) :
    (Finset.univ.filter fun g' : Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N =>
      ∃ q t, Discharge.Layout.keyPubOf c hU n pi g' = some (u, q, t)).card ≤ pubBits c hU pi := by
  have hinj : Set.InjOn
      (fun g' : Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N =>
        (g'.val - 2) % Discharge.Layout.keyB c hU)
      ((Finset.univ.filter fun g' : Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N =>
        ∃ q t, Discharge.Layout.keyPubOf c hU n pi g' = some (u, q, t) : Finset _) : Set _) := by
    intro g₁ hg₁ g₂ hg₂ he
    obtain ⟨q₁, t₁, h₁⟩ := (Finset.mem_filter.1 (Finset.mem_coe.1 hg₁)).2
    obtain ⟨q₂, t₂, h₂⟩ := (Finset.mem_filter.1 (Finset.mem_coe.1 hg₂)).2
    obtain ⟨h21, -, hu₁, -⟩ := Discharge.Layout.keyPubOf_spec h₁
    obtain ⟨h22, -, hu₂, -⟩ := Discharge.Layout.keyPubOf_spec h₂
    have hd : (g₁.val - 2) / Discharge.Layout.keyB c hU = (g₂.val - 2) / Discharge.Layout.keyB c hU :=
      hu₁.symm.trans hu₂
    have hm : (g₁.val - 2) % Discharge.Layout.keyB c hU = (g₂.val - 2) % Discharge.Layout.keyB c hU := he
    have e₁ := Nat.div_add_mod (g₁.val - 2) (Discharge.Layout.keyB c hU)
    have e₂ := Nat.div_add_mod (g₂.val - 2) (Discharge.Layout.keyB c hU)
    rw [hd, hm] at e₁
    have e : g₁.val - 2 = g₂.val - 2 := e₁.symm.trans e₂
    exact Fin.ext (by omega)
  rw [← Finset.card_image_of_injOn hinj]
  unfold pubBits
  refine Finset.card_le_card fun b hb => ?_
  obtain ⟨g', hg', rfl⟩ := Finset.mem_image.1 hb
  obtain ⟨q, t, h⟩ := (Finset.mem_filter.1 hg').2
  obtain ⟨h2, hlt, -, hk⟩ := Discharge.Layout.keyPubOf_spec h
  have hB : 0 < Discharge.Layout.keyB c hU := Nat.pos_of_ne_zero fun hz => by
    rw [hz, Nat.mul_zero] at hlt
    omega
  exact Finset.mem_filter.2 ⟨Finset.mem_range.2 (Nat.mod_lt _ hB), q, t, hk⟩

theorem qs_vbP_le_split (L : FlockSoundness.Audit.SamplingDesign n) (kd : ℕ) (bad : Finset L.Ω)
    (hgood : ∀ ω, ω ∉ bad → (L.draw ω).card ≤ kd) :
    ∑ p : (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).Pos,
        (Fintype.card (ΩcZ execArith ((keyProgP c pi hU n).partition (Discharge.Layout.outsOfP c hU n pi))
          (planZHJP dj y₀) (tabZHJP dj y₀) (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)) L p) : ℝ) /
          Fintype.card L.Ω ≤
      ((kd : ℝ) * ((c.ports.size + pubBits c hU pi : ℕ) : ℝ) + 2) +
        (bad.card : ℝ) / Fintype.card L.Ω * ((n : ℝ) * ((c.ports.size + pubBits c hU pi : ℕ) : ℝ)) := by
  classical
  have h := qs_le_split execArith ((keyProgP c pi hU n).partition (Discharge.Layout.outsOfP c hU n pi))
    (planZHJP dj y₀) (tabZHJP dj y₀) (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)) L
    (fun u => (Finset.univ.image fun q : Fin c.ports.size => (Sum.inl (u, q) : PosHP c hU n pi)) ∪
      (Finset.univ.filter fun g' : Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N =>
        ∃ q t, Discharge.Layout.keyPubOf c hU n pi g' = some (u, q, t)).image
        fun g' => (Sum.inr g' : PosHP c hU n pi))
    {(Sum.inr (Discharge.Layout.zeroG (c := c) (hU := hU) (n := n)) : PosHP c hU n pi),
      Sum.inr (Discharge.Layout.oneG (c := c) (hU := hU) (n := n))} kd (c.ports.size + pubBits c hU pi)
    (fun u => (Finset.card_union_le _ _).trans (add_le_add
      (Finset.card_image_le.trans (le_of_eq (by rw [Finset.card_univ, Fintype.card_fin])))
      (Finset.card_image_le.trans (card_pubGates_le u))))
    (fun u g hg => by
      rcases vbP_pos_io (dj := dj) (y₀ := y₀) (ht := ht) (hc := hc) (hscope := hscope) (hpi := hpi) (hpub := hpub)
        (Reg := Reg) u hg with ⟨q, hq⟩ | h0 | h1 | ⟨hp, q, t, hpq⟩
      · exact Or.inl (Finset.mem_union_left _ (Finset.mem_image.2 ⟨q, Finset.mem_univ _, hq.symm⟩))
      · exact Or.inr (by rw [h0]; exact Finset.mem_insert_self _ _)
      · exact Or.inr (by rw [h1]; exact Finset.mem_insert_of_mem (Finset.mem_singleton_self _))
      · exact Or.inl (Finset.mem_union_right _ (Finset.mem_image.2
          ⟨g, Finset.mem_filter.2 ⟨Finset.mem_univ _, q, t, hpq⟩, hp.symm⟩)))
    bad hgood
  refine h.trans (add_le_add (add_le_add le_rfl ?_) le_rfl)
  exact_mod_cast Finset.card_le_two

theorem qs_vbP_execSel_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src)
    {samplingDesign : Flock.Draw.SamplingDesign} {ns : List ℕ} {kd : ℕ} (hlaw : SamplingDesignOn samplingDesign ns kd) (hns : ns.sum = n)
    (hLb : firstRead samplingDesign n ≤ Lb 0) :
    ∑ p : (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).Pos,
        (Fintype.card (ΩcZ execArith ((keyProgP c pi hU n).partition (Discharge.Layout.outsOfP c hU n pi))
          (planZHJP dj y₀) (tabZHJP dj y₀) (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg))
          (execSel (n := n) src samplingDesign n) p) : ℝ) / Fintype.card (execSel (n := n) src samplingDesign n).Ω ≤
      ((kd : ℝ) * ((c.ports.size + pubBits c hU pi : ℕ) : ℝ) + 2) +
        (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ * ((n : ℝ) * ((c.ports.size + pubBits c hU pi : ℕ) : ℝ)) := by
  classical
  have hbad : ((Finset.univ.filter fun ω : Ω => ¬ ∃ units read,
      Spot.Draw.sampleUnits samplingDesign n (Properties.Spot.bytesOf (src ω 0)) = .ok (units, read)).card : ℝ) /
      Fintype.card Ω ≤ (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ :=
    card_div_le_of_prCoin_le _ kd (execSel_error_le src hA3 hlaw hns hLb)
  have h := qs_vbP_le_split (dj := dj) (y₀ := y₀) (ht := ht) (hc := hc) (hscope := hscope) (hU := hU) (hpi := hpi)
    (hpub := hpub) (Reg := Reg) (execSel (n := n) src samplingDesign n) kd
    (Finset.univ.filter fun ω : Ω => ¬ ∃ units read,
      Spot.Draw.sampleUnits samplingDesign n (Properties.Spot.bytesOf (src ω 0)) = .ok (units, read))
    (fun ω hω => execSel_card_le src hlaw ω (by
      by_contra h0
      exact hω (Finset.mem_filter.2 ⟨Finset.mem_univ _, h0⟩)))
  exact h.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hbad (by positivity)))

end QsRowsP

end Flock.Bridge
