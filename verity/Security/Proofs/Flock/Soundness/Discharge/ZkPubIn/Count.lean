import Proofs.Flock.Soundness.Discharge.ZkPubIn.Bind
import Proofs.Flock.Soundness.Audit.FlockPublic

set_option Elab.async false

/-!
# The public-input step: the count curve at the public transcript

On a public gate the plurality value layer `XplurZC` is the prover's only in form: every satisfying message of every
drawn unit's table decodes it to the verifier's own word's bit, which the `in` region pins at its column. So the
committed transcript here is `Audit.Xpub ones zeros XplurZC`, 1 on the public gates at 1, 0 on those at 0, and the
plurality elsewhere, and the count curve holds for it with the same bound (`Audit.FlockPublic`'s
`flock_batched_count_public`, at the compiled `--zk` analysis):

* `PubDecodesZ`: every satisfying message of a drawn unit's `--zk` table decodes, on the unit's committed wires, each
  gate of `ones` to 1 and each gate of `zeros` outside `ones` to 0 (`Audit.Partition.PubDecodes` at `--zk`).
* `linkEvZK_of_pub`, `linkSoundZC_mono`, `zk_linkSoundC_pub`, `zk_flock_countC_pub`: the link event at the public
  transcript is the link event at `XplurZC` (`Audit.Partition.linkEvB_of_pub`), the link term is monotone in it, and
  `zk_flock_countC` holds at `Xpub ones zeros XplurZC` under `PubDecodesZ`;
* `zk_flock_countC_execOS_cr_pub`: `zk_flock_countC_execOS_cr` at the public transcript;
* `pubDecodesZ_placeZCP`: `PubDecodesZ` at the canonical session's places (`placeZCP`) for the public gates
  `pubOnes`, `pubZeros`.
-/


namespace FlockSoundness.Discharge.ZkPubIn

open Game Model Accounting Audit ZkSession StrictCR ZkLink Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

section Generic

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The public wires decode to their values at `--zk`**: every message whose witness satisfies a drawn unit's table
decodes, on the unit's committed wires, each gate of `ones` to 1 and each gate of `zeros` outside `ones` to 0. -/
def PubDecodesZ (decode : DecoderZK (C := C) planZ tab) (ones zeros : Finset (Fin C.N)) : Prop :=
  ∀ S R u, u ∈ S → ∀ M, (tabOfZ planZ tab S R u).S.Satisfies
      (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
    ∀ g ∈ P.io u, (g ∈ ones → decode S R u M g = true) ∧ (g ∉ ones → g ∈ zeros → decode S R u M g = false)

omit [Fintype K] [DecidableEq K] in
/-- **The `--zk` link event at the public transcript is the link event at `vals`** (`Partition.linkEvB_of_pub`). -/
theorem linkEvZK_of_pub {decode : DecoderZK (C := C) planZ tab} {ones zeros : Finset (Fin C.N)}
    (hpub : PubDecodesZ A P planZ tab decode ones zeros) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    {vals : Fin C.N → Bool} {T : Oracle F} (h : LinkEvZK A P planZ tab decode S R u (Xpub ones zeros vals) T) :
    LinkEvZK A P planZ tab decode S R u vals T := by
  refine ⟨h.1, fun M hM hsat => ?_⟩
  obtain ⟨g, hg, hne⟩ := h.2 M hM hsat
  refine ⟨g, hg, fun heq => hne ?_⟩
  obtain ⟨h1, h0⟩ := hpub S R u hu M hsat g hg
  by_cases hg1 : g ∈ ones
  · rw [Xpub_of_mem_ones hg1]; exact h1 hg1
  · by_cases hg0 : g ∈ zeros
    · rw [Xpub_of_mem_zeros hg1 hg0]; exact h0 hg1 hg0
    · rw [Xpub_of_not_mem hg1 hg0]; exact heq

omit [Fintype K] in
/-- **The compiled link term is monotone in the link event** (`Partition.linkSoundE_mono` at `analysisZC`). -/
theorem linkSoundZC_mono {L : Law n} {decode : DecoderZK (C := C) planZ tab}
    {Xc Xc' : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)} {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a)
    (hsub : ∀ R τ S u, u ∈ S → ∀ T, LinkEvZK A P planZ tab decode S R u (Xc' R τ) T →
      LinkEvZK A P planZ tab decode S R u (Xc R τ) T)
    {δlink : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → ℝ≥0∞}
    (h : (analysisZC A Hs P planZ tab decode Xc k hr hlow a hj).LinkSound δlink) :
    (analysisZC A Hs P planZ tab decode Xc' k hr hlow a hj).LinkSound δlink := by
  classical
  intro R τ tgt
  refine le_trans (avg_mono fun ω => ?_) (h R τ tgt)
  cases htg : tgt ω with
  | none => simp only [atTgt, Option.elim_none, le_refl]
  | some u =>
    simp only [atTgt, Option.elim_some]
    split_ifs with hu
    · show linkTermZC A Hs P planZ tab decode k (L.draw ω) R (τ ω) u (Xc' R τ) ≤
        linkTermZC A Hs P planZ tab decode k (L.draw ω) R (τ ω) u (Xc R τ)
      unfold linkTermZC
      apply ENNReal.ofReal_le_ofReal
      have hr1 : 0 ≤ 1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k := by
        linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
      refine div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left ?_ (accProbZC_nonneg _ _ _ _)) hr1
      unfold extractProbAccZ
      refine Extract.expectAcc_mono _ _ _ _ fun bs => ?_
      unfold ind
      split_ifs with h1 h2
      · exact le_refl _
      · exact absurd (hsub R τ _ u hu _ h1) h2
      · exact zero_le_one
      · exact le_refl _
    · exact le_refl _

/-- **The compiled link theorem at the public transcript** (`zk_linkSoundC` at `Xpub ones zeros XplurZC`). -/
theorem zk_linkSoundC_pub {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab decode Hc) (ones zeros : Finset (Fin C.N))
    (hpub : PubDecodesZ A P planZ tab decode ones zeros)
    (hlow : LoweringSoundZK A P planZ tab decode) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') :
    (analysisZC (L := L) A Hs P planZ tab decode
        (fun R τ => Xpub ones zeros (XplurZC A Hs P planZ tab vb R τ k Rw)) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).LinkSound
      (linkBoundZC A Hs P planZ tab vb k Rw M ρ t') :=
  linkSoundZC_mono A Hs P planZ tab _ hlow hj (fun _ _ _ _ hu _ h => linkEvZK_of_pub A P planZ tab hpub hu h)
    (zk_linkSoundC A Hs P planZ tab vb hlow hRw hM hρ hr hj ht)

/-- **The compiled `--zk` count curve at the public transcript** (`zk_flock_countC` at `Xpub ones zeros XplurZC`). -/
theorem zk_flock_countC_pub {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab decode Hc) (ones zeros : Finset (Fin C.N))
    (hpub : PubDecodesZ A P planZ tab decode ones zeros)
    (hlow : LoweringSoundZK A P planZ tab decode) {k Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        K ≤ (P.wrong (Xpub ones zeros (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  extraction_audit_count (analysisZC A Hs P planZ tab decode
      (fun R τ => Xpub ones zeros (XplurZC A Hs P planZ tab vb R τ k Rw)) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow a hj)
    (zc_knowledgeSound A Hs P planZ tab decode _ _ hlow hj)
    (zk_linkSoundC_pub A Hs P planZ tab vb ones zeros hpub hlow hRw hM hρ hr hj ht) K σ

/-- **The compiled `--zk` count curve at the public transcript**: `zk_flock_countC_execOS_cr` with the committed
transcript 1 on `ones`, 0 on `zeros` and the plurality elsewhere, when the places' decoder reads the public wires as
those values (`PubDecodesZ`), at the same bound. -/
theorem zk_flock_countC_execOS_cr_pub (hA : A.Correct)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc)
    (ones zeros : Finset (Fin C.N)) (hpub : PubDecodesZ A P planZ tab (decodeOfPlace A P planZ tab place) ones zeros)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ planZ (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) + ENNReal.ofReal (tr.bound qT) := by
  have hlow := loweringSoundZK_of_place A P planZ tab place
  refine (Binding.prob_split_accept _ σ execAccept _).trans (add_le_add ?_ (tr.le hT))
  refine (prob_mono (q := fun o => o.1 = true ∧
    K ≤ (P.wrong (Xpub ones zeros (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))).card)
    (fun o h => ⟨h.1.1, h.2⟩) _ _).trans ?_
  exact (zk_flock_countC_pub A Hs P planZ tab vb ones zeros hpub hlow hRw hM hρ hr (jointZC_ksBound A hA Hs planZ hk)
    ht K σ).trans
    (add_le_add (add_le_add (Law.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K)
      (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR)) le_rfl)

end Generic

open Flock FlockLevel3 Refine ZkExec ZkBind HiddenExec Placed ZkHidden Composed
open FlockSoundness.Discharge.Canonical (formatOk)

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} {own : ByteArray} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hfmt : formatOk c = true) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pub : Public}
  (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub) (hU : Built c.unit)

include hpub in
/-- **The public gates decode to the verifier's own bits at `placeZCP`**: in every satisfying witness of a drawn unit's
canonical `--zk` table, the unit's column on a public gate (`keyProgP`'s gate of a public leaf's bit `(q, t)` of input
`x`) holds bit `t` of the verifier's own word (`keyPlaceCP_decode_pub`, through the public plan's places, which read
one witness alike: `decAgreeCP`). -/
theorem pubDecodesZ_placeZCP :
    PubDecodesZ execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP (Reg := Reg) dj y₀)
      (tabZHJP dj y₀)
      (decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
        (tabZHJP dj y₀) (placeZCP dj y₀ ht hc hfmt hpi hU))
      (pubOnes c hU n pi own) (pubZeros c hU n pi own) := by
  intro S R u hu M hsat g hg
  obtain ⟨Ms, hws, hMs⟩ := shadow_msg (planZHJP_S (Reg := Reg) dj y₀) S R u M hsat
  have hag := decAgreeCP dj y₀ ht hc hfmt hpi hU S R u hu M Ms hws hsat g hg
  have hval : ∀ b, pubBit c hU n pi own g = some b →
      decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
        (tabZHJP dj y₀) (placeZCP dj y₀ ht hc hfmt hpi hU) S R u M g = b := fun b hb => by
    rw [hag]
    simp only [Partition.placeDecoder, hu, ↓reduceDIte]
    exact keyPlaceCP_decode_pub (sitesShCP_parse dj y₀ ht hc hfmt) pi hpi own (Layout.outsOf c hU n)
      (fun _ => pub) hfmt (sitesShCP_bcSite dj y₀ ht hc hfmt hpi hpub) (sitesShCP_own dj y₀ ht hc hfmt) hu hMs hg hb
  refine ⟨fun h1 => hval true ?_, fun _ h0 => hval false ?_⟩
  · exact ((@Finset.mem_filter _ (fun g => pubBit c hU n pi own g = some true) (Classical.decPred _) _ _).1 h1).2
  · exact ((@Finset.mem_filter _ (fun g => pubBit c hU n pi own g = some false) (Classical.decPred _) _ _).1 h0).2

end FlockSoundness.Discharge.ZkPubIn
