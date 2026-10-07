import Proofs.Pous.Dense.DenseReCert
import Definitions.Pous.Model.SegLift

/-!
# The 14 GB theorem for the implementation's tagged keying

`seg_meets_14` is proved for `Pous.Dense.segScheme`, whose random oracle is keyed by the **global** block index and
global parents. The implementation keys segment `s`'s oracle by `tag_s = salt ‖ u64 s` and **local** queries: segment
`s` is the dense scheme on `B` blocks with oracle `H(tag_s, ·)`. That is the trusted `Pous.Dense.segTagScheme`, over the
random oracle `Bits 256 × LabelQry B ℓ → Bits ℓ`. Its queries are not the global scheme's queries, so the transfer is a
statement about the two oracle games, proved here (`segTag_auditSecure`):

1. **An injective query map** `f : Bits 256 × LabelQry B ℓ → LabelQry (N*B) ℓ` sends `(tag_s, (i, p))` to the global
   query `(s·B + i, p placed in segment s)` (`tagEmbed`; injective for `2 ≤ N ≤ 2^64` and `ℓ ≥ 256`). Queries under a
   tag that is no segment's go injectively to global queries no segment asks.
2. **The codewords agree:** the tagged scheme under `H ∘ f` encodes exactly as the global scheme under `H`
   (`label_tag_eq`).
3. **Adversaries transfer:** a tagged preprocessor and responder, with their queries sent through `f`, are a global
   preprocessor and responder with the same rounds and work (`liftProg`), and they accept on `H` exactly when the
   tagged ones accept on `H ∘ f`.
4. **The oracle transfers:** for injective `f`, `H ∘ f` is uniform when `H` is (`pr_comp_injective`).

So every tagged adversary's acceptance probability is a global adversary's, and `segTag_meets_14` follows from
`seg_meets_14`, for every salt.
-/

namespace PousDenseReCert

open Pous Pous.Labelling Pous.Dense
open scoped ENNReal

/-! ### The tagged scheme decodes -/

/-- Bit-exact public decoding for the tagged scheme. -/
theorem segTagScheme_correct (N B ℓ : ℕ) [NeZero (N * B)] (salt : Bits 192) : (segTagScheme N B ℓ salt).Correct := by
  intro (W : Bits ((N * B) * ℓ)) (H : Bits (192 + 64) × LabelQry B ℓ → Bits ℓ) _
  dsimp only [segTagScheme]
  conv_rhs => rw [← (codeEquiv (N * B) ℓ).apply_symm_apply W]
  congr 1
  funext g
  obtain ⟨⟨s, i⟩, rfl⟩ := (segIndexEquiv N B).surjective g
  simp only [Equiv.symm_apply_apply]
  rw [PousB5.label_eq]
  exact PousDense.bxor_bxor_self _ _

/-! ### The query map -/

/-- A 256-bit tag written into an `ℓ`-bit label slot (zero-padded; injective when `ℓ ≥ 256`). -/
def padTag (ℓ : ℕ) (τ : Bits (192 + 64)) : Bits ℓ := fun j => if h : j.val < 192 + 64 then τ ⟨j.val, h⟩ else false

theorem padTag_injective {ℓ : ℕ} (hℓ : 192 + 64 ≤ ℓ) : Function.Injective (padTag ℓ) := by
  intro τ τ' h
  funext j
  have := congrFun h ⟨j.val, lt_of_lt_of_le j.isLt hℓ⟩
  simpa [padTag, j.isLt] using this

open Classical in
/-- **The query map.** A query `(tag_s, (i, p))` under segment `s`'s tag goes to the global query `(s·B + i, p placed
in segment s)`. A query under any other tag `τ` goes to `(s0·B + i, p placed in segment s0, with τ written at block
b0 of segment s1)`, a global query no segment asks, since a segment's queries have no parent outside it. -/
noncomputable def tagEmbed (N B ℓ : ℕ) (salt : Bits 192) (s0 s1 : Fin N) (b0 : Fin B) :
    Bits (192 + 64) × LabelQry B ℓ → LabelQry (N * B) ℓ := fun x =>
  if h : ∃ s : Fin N, tag salt s.val = x.1 then
    (segIndexEquiv N B (h.choose, x.2.1), liftPar N B ℓ h.choose x.2.2)
  else
    (segIndexEquiv N B (s0, x.2.1), fun g =>
      if ((segIndexEquiv N B).symm g).1 = s0 then x.2.2 ((segIndexEquiv N B).symm g).2
      else if (segIndexEquiv N B).symm g = (s1, b0) then some (padTag ℓ x.1) else none)

theorem tagEmbed_tag {N B ℓ : ℕ} (salt : Bits 192) (s0 s1 : Fin N) (b0 : Fin B) (hN : N ≤ 2 ^ 64)
    (s : Fin N) (i : Fin B) (p : Fin B → Option (Bits ℓ)) :
    tagEmbed N B ℓ salt s0 s1 b0 (tag salt s.val, (i, p)) = (segIndexEquiv N B (s, i), liftPar N B ℓ s p) := by
  have hex : ∃ s' : Fin N, tag salt s'.val = tag salt s.val := ⟨s, rfl⟩
  have hs : hex.choose = s := tag_injective salt hN hex.choose_spec
  simp only [tagEmbed, dif_pos hex, hs]

theorem liftPar_apply {N B ℓ : ℕ} (s s' : Fin N) (j : Fin B) (p : Fin B → Option (Bits ℓ)) :
    liftPar N B ℓ s p (segIndexEquiv N B (s', j)) = if s' = s then p j else none := by
  simp [liftPar]

theorem tagEmbed_injective {N B ℓ : ℕ} (salt : Bits 192) (s0 s1 : Fin N) (b0 : Fin B) (hN : N ≤ 2 ^ 64)
    (hs : s1 ≠ s0) (hℓ : 192 + 64 ≤ ℓ) : Function.Injective (tagEmbed N B ℓ salt s0 s1 b0) := by
  rintro ⟨τ, i, p⟩ ⟨τ', i', p'⟩ hxy
  by_cases h : ∃ s : Fin N, tag salt s.val = τ <;> by_cases h' : ∃ s : Fin N, tag salt s.val = τ'
  · obtain ⟨s, rfl⟩ := h
    obtain ⟨s', rfl⟩ := h'
    rw [tagEmbed_tag salt s0 s1 b0 hN, tagEmbed_tag salt s0 s1 b0 hN] at hxy
    obtain ⟨h1, h2⟩ := Prod.mk.inj hxy
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj ((segIndexEquiv N B).injective h1)
    have hp : p = p' := by
      funext j
      have := congrFun h2 (segIndexEquiv N B (s, j))
      simpa [liftPar_apply] using this
    rw [hp]
  · obtain ⟨s, rfl⟩ := h
    rw [tagEmbed_tag salt s0 s1 b0 hN] at hxy
    simp only [tagEmbed, dif_neg h'] at hxy
    obtain ⟨h1, h2⟩ := Prod.mk.inj hxy
    obtain ⟨rfl, -⟩ := Prod.mk.inj ((segIndexEquiv N B).injective h1)
    have := congrFun h2 (segIndexEquiv N B (s1, b0))
    simp [liftPar_apply, hs] at this
  · obtain ⟨s', rfl⟩ := h'
    rw [tagEmbed_tag salt s0 s1 b0 hN] at hxy
    simp only [tagEmbed, dif_neg h] at hxy
    obtain ⟨h1, h2⟩ := Prod.mk.inj hxy
    obtain ⟨rfl, -⟩ := Prod.mk.inj ((segIndexEquiv N B).injective h1)
    have := congrFun h2 (segIndexEquiv N B (s1, b0))
    simp [liftPar_apply, hs] at this
  · simp only [tagEmbed, dif_neg h, dif_neg h'] at hxy
    obtain ⟨h1, h2⟩ := Prod.mk.inj hxy
    obtain ⟨-, rfl⟩ := Prod.mk.inj ((segIndexEquiv N B).injective h1)
    have hp : p = p' := by
      funext j
      have := congrFun h2 (segIndexEquiv N B (s0, j))
      simpa using this
    have hτ : padTag ℓ τ = padTag ℓ τ' := by
      have := congrFun h2 (segIndexEquiv N B (s1, b0))
      simpa [hs] using this
    rw [hp, padTag_injective hℓ hτ]

/-! ### The codewords agree -/

theorem mem_segDAG_par {N B : ℕ} (s s' : Fin N) (i j : Fin B) :
    segIndexEquiv N B (s', j) ∈ (segDAG N B).par (segIndexEquiv N B (s, i)) ↔ s' = s ∧ j < i := by
  have hB : 0 < B := Fin.pos i
  simp only [segDAG, segIndexEquiv, Finset.mem_filter, Finset.mem_univ, true_and, finProdFinEquiv_apply_val]
  have e1 : (j.val + B * s'.val) / B = s'.val := by
    rw [Nat.add_mul_div_left _ _ hB, Nat.div_eq_of_lt j.isLt, zero_add]
  have e2 : (i.val + B * s.val) / B = s.val := by
    rw [Nat.add_mul_div_left _ _ hB, Nat.div_eq_of_lt i.isLt, zero_add]
  rw [e1, e2, Fin.lt_def]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨Fin.ext h1, by rw [h1] at h2; omega⟩
  · rintro ⟨rfl, h2⟩
    exact ⟨rfl, by omega⟩

/-- **The tagged labels under `H ∘ f` are the global labels under `H`**, for any query map that sends segment `s`'s
queries to their global form. -/
theorem label_tag_eq {N B ℓ : ℕ} (salt : Bits 192) (f : Bits (192 + 64) × LabelQry B ℓ → LabelQry (N * B) ℓ)
    (htag : ∀ (s : Fin N) i p, f (tag salt s.val, (i, p)) = (segIndexEquiv N B (s, i), liftPar N B ℓ s p))
    (H : LabelQry (N * B) ℓ → Bits ℓ) (W : Bits ((N * B) * ℓ)) (s : Fin N) (i : Fin B) :
    label (completeDAG B) (segBlocks N B ℓ W s) (fun q => H (f (tag salt s.val, q))) i =
      label (segDAG N B) (blocks (N * B) ℓ W) H (segIndexEquiv N B (s, i)) := by
  induction i using WellFoundedLT.induction with
  | _ i ih =>
    rw [PousB5.label_eq, PousB5.label_eq]
    simp only [PousB5.realQ]
    rw [htag]
    congr 2
    congr 1
    funext g
    obtain ⟨⟨s', j⟩, rfl⟩ := (segIndexEquiv N B).surjective g
    rw [liftPar_apply]
    by_cases hs : s' = s
    · subst hs
      rw [if_pos rfl]
      by_cases hj : j < i
      · have hmem : segIndexEquiv N B (s', j) ∈ (segDAG N B).par (segIndexEquiv N B (s', i)) :=
          (mem_segDAG_par s' s' i j).2 ⟨rfl, hj⟩
        have hmem' : j ∈ (completeDAG B).par i := by simp [completeDAG, hj]
        rw [if_pos hmem, if_pos hmem', ih j hj]
      · have hmem : segIndexEquiv N B (s', j) ∉ (segDAG N B).par (segIndexEquiv N B (s', i)) :=
          fun h => hj ((mem_segDAG_par s' s' i j).1 h).2
        have hmem' : j ∉ (completeDAG B).par i := by simp [completeDAG, hj]
        rw [if_neg hmem, if_neg hmem']
    · have hmem : segIndexEquiv N B (s', j) ∉ (segDAG N B).par (segIndexEquiv N B (s, i)) :=
        fun h => hs ((mem_segDAG_par s s' i j).1 h).1
      rw [if_neg hs, if_neg hmem]

/-! ### Programs, with their queries sent through `f` -/

variable {X Y Z : Type} [Fintype X] [DecidableEq X] [Fintype Y] [DecidableEq Y] [Fintype Z] [Nonempty Z]

/-- A program against the random oracle on `X`, run against the random oracle on `Y` by sending each query through
`f`. -/
def liftProg (f : X → Y) {α : Type} : Prog (IdealModel.randomOracle X Z) α → Prog (IdealModel.randomOracle Y Z) α
  | .ret a => .ret a
  | .round qs k => .round (qs.map f) fun as => liftProg f (k as)

omit [Fintype X] [DecidableEq X] [Fintype Y] [DecidableEq Y] [Fintype Z] [Nonempty Z] in
theorem map_map_comp (f : X → Y) (H : Y → Z) (l : List X) :
    (l.map f).map (fun y => H y) = l.map (fun x => H (f x)) := by
  rw [List.map_map]; rfl

theorem run_liftProg (f : X → Y) {α : Type} (H : Y → Z) (p : Prog (IdealModel.randomOracle X Z) α) :
    (liftProg f p).run (fun y => H y) = p.run (fun x => H (f x)) := by
  induction p with
  | ret a => rfl
  | round qs k ih =>
    have hm := map_map_comp f H qs
    exact (congrArg (fun as => (liftProg f (k as)).run (fun y => H y)) hm).trans (ih _)

theorem depth_liftProg (f : X → Y) {α : Type} (H : Y → Z) (p : Prog (IdealModel.randomOracle X Z) α) :
    (liftProg f p).depth (fun y => H y) = p.depth (fun x => H (f x)) := by
  induction p with
  | ret a => rfl
  | round qs k ih =>
    have hm := map_map_comp f H qs
    exact (congrArg (fun as => (liftProg f (k as)).depth (fun y => H y) + 1) hm).trans (congrArg (· + 1) (ih _))

theorem work_liftProg (f : X → Y) {α : Type} (H : Y → Z) (p : Prog (IdealModel.randomOracle X Z) α) :
    (liftProg f p).work (fun y => H y) = p.work (fun x => H (f x)) := by
  induction p with
  | ret a => rfl
  | round qs k ih =>
    have hm := map_map_comp f H qs
    have hl : ((qs : List X).map f).length = (qs : List X).length := List.length_map _
    exact congrArg₂ (· + ·) hl ((congrArg (fun as => (liftProg f (k as)).work (fun y => H y)) hm).trans (ih _))

theorem bounded_liftProg (f : X → Y) {α : Type} {D Q : ℕ} {p : Prog (IdealModel.randomOracle X Z) α}
    (h : p.Bounded D Q) : (liftProg f p).Bounded D Q := fun H => by
  have := h (fun x => H (f x))
  exact ⟨(depth_liftProg f H p).trans_le this.1, (work_liftProg f H p).trans_le this.2⟩

/-! ### The oracle, through an injective `f` -/

/-- For injective `f`, restricting a uniform oracle on `Y` to `f`'s image gives a uniform oracle on `X`. -/
theorem pr_comp_injective {Γ : Type} [Fintype Γ] (f : X → Y) (hf : Function.Injective f)
    (E : (X → Z) × Γ → Prop) : pr (fun x : (Y → Z) × Γ => E (fun a => x.1 (f a), x.2)) = pr E := by
  classical
  let eX : X ≃ Set.range f := Equiv.ofInjective f hf
  let e : (Y → Z) ≃ (X → Z) × ({y // y ∉ Set.range f} → Z) :=
    (Equiv.piEquivPiSubtypeProd (fun y => y ∈ Set.range f) (fun _ => Z)).trans
      (Equiv.prodCongr (Equiv.arrowCongr eX.symm (Equiv.refl Z)) (Equiv.refl _))
  let e2 : (Y → Z) × Γ ≃ ((X → Z) × Γ) × ({y // y ∉ Set.range f} → Z) :=
    (Equiv.prodCongr e (Equiv.refl Γ)).trans
      ((Equiv.prodAssoc _ _ _).trans ((Equiv.prodCongr (Equiv.refl _) (Equiv.prodComm _ _)).trans
        (Equiv.prodAssoc _ _ _).symm))
  have he2 : ∀ x : (Y → Z) × Γ, (e2 x).1 = (fun a => x.1 (f a), x.2) := fun x => rfl
  have h1 := PousThm1.pr_equiv e2 (fun y : ((X → Z) × Γ) × ({y // y ∉ Set.range f} → Z) => E y.1)
  have h2 := PousThm1.pr_fst (β := {y // y ∉ Set.range f} → Z) E
  simp only [he2] at h1
  rw [h1]
  convert h2 using 2

theorem seqAnswers_liftProg (f : X → Y) (H : Y → Z) {ℓ S B : ℕ}
    (step : ℕ → Bits S → Fin B → Prog (IdealModel.randomOracle X Z) (Bits ℓ × Bits S)) :
    ∀ j is σ, seqAnswers ((IdealModel.randomOracle Y Z).answer H) (fun j σ i => liftProg f (step j σ i)) j is σ =
      seqAnswers ((IdealModel.randomOracle X Z).answer fun x => H (f x)) step j is σ
  | _, [], _ => rfl
  | j, i :: is, σ => by
    have hr := run_liftProg f H (step j σ i)
    simp only [seqAnswers]
    rw [show (liftProg f (step j σ i)).run ((IdealModel.randomOracle Y Z).answer H) =
      (step j σ i).run ((IdealModel.randomOracle X Z).answer fun x => H (f x)) from hr]
    rw [seqAnswers_liftProg f H step (j + 1) is]

/-! ### The audit game transfers -/

/-- **The tagged scheme is as secure as the global one.** For a query map `f` that is injective and sends segment
`s`'s queries to their global form, every sequential-audit bound of `segScheme` holds for `segTagScheme`: a tagged
adversary, with its queries sent through `f`, is a global adversary accepted with the same probability. -/
theorem segTag_auditSecure {N B ℓ : ℕ} [NeZero (N * B)] (salt : Bits 192)
    (f : Bits (192 + 64) × LabelQry B ℓ → LabelQry (N * B) ℓ) (hf : Function.Injective f)
    (htag : ∀ (s : Fin N) i p, f (tag salt s.val, (i, p)) = (segIndexEquiv N B (s, i), liftPar N B ℓ s p))
    {S D Q k : ℕ} {δ ε : ℝ≥0∞} (h : AuditSecure (segScheme N B ℓ) .sequential S D Q k δ ε) :
    AuditSecure (segTagScheme N B ℓ salt) .sequential S D Q k δ ε := by
  intro W A₁ A₂
  let A₁' : Preprocessor (segScheme N B ℓ) S := fun W pp C H => A₁ W pp C (fun x => H (f x))
  let A₂' : SeqResponder (segScheme N B ℓ) S D Q :=
    ⟨fun W pp j σ i => liftProg f (A₂.step W pp j σ i), fun W pp j σ i => bounded_liftProg f (A₂.bounded W pp j σ i)⟩
  have henc : ∀ H : LabelQry (N * B) ℓ → Bits ℓ,
      ((segTagScheme N B ℓ salt).enc W (fun x => H (f x)) ()).2 = ((segScheme N B ℓ).enc W H ()).2 := by
    intro H
    funext g
    obtain ⟨⟨s, i⟩, rfl⟩ := (segIndexEquiv N B).surjective g
    dsimp only [segTagScheme, segScheme]
    simp only [Equiv.symm_apply_apply]
    exact label_tag_eq salt f htag H W s i
  have hpt : ∀ x : (LabelQry (N * B) ℓ → Bits ℓ) × Unit × (Fin k → Fin (N * B)),
      seqAccepts (segTagScheme N B ℓ salt) k W A₁ A₂ (fun q => x.1 (f q), x.2) ↔
        seqAccepts (segScheme N B ℓ) k W A₁' A₂' x := by
    rintro ⟨H, r, I⟩
    cases r
    have hs := seqAnswers_liftProg f H (A₂.step W (fun i => i.elim0)) 0 (List.ofFn I)
      (A₁ W (fun i => i.elim0) ((segScheme N B ℓ).enc W H ()).2 (fun x => H (f x)))
    simp only [seqAccepts]
    rw [henc H]
    exact Iff.of_eq (congrArg (· = _) hs.symm)
  calc pr (seqAccepts (segTagScheme N B ℓ salt) k W A₁ A₂)
      = pr (fun x : (LabelQry (N * B) ℓ → Bits ℓ) × (Unit × (Fin k → Fin (N * B))) =>
          seqAccepts (segTagScheme N B ℓ salt) k W A₁ A₂ (fun q => x.1 (f q), x.2)) :=
        (pr_comp_injective f hf _).symm
    _ = pr (seqAccepts (segScheme N B ℓ) k W A₁' A₂') := by
        congr 1
        funext x
        exact propext (hpt x)
    _ ≤ _ := h W A₁' A₂'

/-! ### The 14 GB theorem, tagged -/

/-- **Satisfiability witness** for the query map's hypotheses at 14 GB: segments `0 ≠ 1`, `448 ≤ 2^64` tags and
`256 ≤ 524288`-bit labels. -/
theorem tagEmbed_hyps_14 : (1 : Fin 448) ≠ 0 ∧ 448 ≤ 2 ^ 64 ∧ 192 + 64 ≤ 524288 := by decide

/-- The query map at 14 GB is injective. -/
theorem tagEmbed_14_injective (salt : Bits 192) :
    Function.Injective (tagEmbed 448 (2 ^ 9) 524288 salt 0 1 ⟨0, by norm_num⟩) :=
  tagEmbed_injective salt 0 1 _ tagEmbed_hyps_14.2.1 tagEmbed_hyps_14.1 tagEmbed_hyps_14.2.2

/-- The query map at 14 GB sends every segment's queries to their global form. -/
theorem tagEmbed_14_tag (salt : Bits 192) (s : Fin 448) (i : Fin (2 ^ 9)) (p : Fin (2 ^ 9) → Option (Bits 524288)) :
    tagEmbed 448 (2 ^ 9) 524288 salt 0 1 ⟨0, by norm_num⟩ (tag salt s.val, (i, p)) =
      (segIndexEquiv 448 (2 ^ 9) (s, i), liftPar 448 (2 ^ 9) 524288 s p) :=
  tagEmbed_tag salt 0 1 _ tagEmbed_hyps_14.2.1 s i p

/-- **Satisfiability witness:** at 14 GB the tagged scheme, run on the restricted oracle, writes exactly the global
scheme's codeword, for every `W`, oracle and salt. -/
theorem segTag_code_eq_14 (salt : Bits 192) (W : Bits ((448 * 2 ^ 9) * 524288))
    (H : LabelQry (448 * 2 ^ 9) 524288 → Bits 524288) :
    ((segTagScheme 448 (2 ^ 9) 524288 salt).enc W
        (fun x => H (tagEmbed 448 (2 ^ 9) 524288 salt 0 1 ⟨0, by norm_num⟩ x)) ()).2 =
      ((segScheme 448 (2 ^ 9) 524288).enc W H ()).2 := by
  funext g
  obtain ⟨⟨s, i⟩, rfl⟩ := (segIndexEquiv 448 (2 ^ 9)).surjective g
  dsimp only [segTagScheme, segScheme]
  simp only [Equiv.symm_apply_apply]
  exact label_tag_eq salt _ (tagEmbed_14_tag salt) H W s i

/-- **The 14 GB theorem for the implementation's keying.** The tagged segmented dense scheme (448 segments of `2^9`
blocks of 64 KB, oracle `H(salt ‖ u64 s, local query)`) meets the requirement for every salt, random-oracle `H`,
sequential reveal, `D = 5`, `Q = 2^20`, `k = 106`, `ε = 2^-128`. -/
theorem segTag_meets_14 (salt : Bits 192) :
    Pous.Meets (segTagScheme 448 (2 ^ 9) 524288 salt) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128) := by
  obtain ⟨hε, -, hW, hS, hA⟩ := seg_meets_14
  exact ⟨hε, segTagScheme_correct _ _ _ _, hW, hS,
    segTag_auditSecure salt _ (tagEmbed_14_injective salt) (tagEmbed_14_tag salt) hA⟩

end PousDenseReCert
