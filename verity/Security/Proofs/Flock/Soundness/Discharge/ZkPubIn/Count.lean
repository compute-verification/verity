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
* `zk_flock_countC_execOS_cr_pub`: `zk_flock_countC_execOS_cr` at `Xpub ones zeros XplurZC`, under `PubDecodesZ` at the
  places' decoder: the link event at the public transcript is the link event at `XplurZC`
  (`Audit.Partition.linkEvB_of_pub`), and the link term is monotone in it (`linkSoundE_mono`).
* `pubDecodesZ_placeZHP`: `PubDecodesZ` at `placeZHP` for the public gates `pubOnes`, `pubZeros`.
* `zk_flock_countC_execOS_atHP`: the count curve at `planAtZKHP` with `placeZHP` and `rsZHP`, at the public
  transcript.
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

open Flock FlockLevel3 Refine ZkExec ZkBind HiddenExec Placed ZkHidden

variable {n : ℕ} {I : Inputs} {own : ByteArray} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀)) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOk c = true) (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
  {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)

include hpi hpub in
/-- **The public gates decode to the verifier's own bits at `placeZHP`**: in every satisfying witness of a drawn unit's
`--zk` table, the unit's column on a public gate (`keyProgP`'s gate of a public leaf's bit `(q, t)` of input `x`)
holds bit `t` of the `in` region's word at its column, which is the drawn statement's public word `q` of the unit's
instance (`HmIn.regions`), the population's (`HmIn.drawn`) and so the verifier's own (`OwnWords`, `HmIn.checkOwn`). -/
theorem pubDecodesZ_placeZHP :
    PubDecodesZ execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP (Reg := Reg) dj y₀)
      (tabAtZKHP dj y₀)
      (decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
        (tabAtZKHP dj y₀) (placeZHP dj y₀ ht hc hscope hU hpi))
      (pubOnes c hU n pi own) (pubZeros c hU n pi own) := by
  intro S R u hu M hsat g hg
  obtain ⟨Ms, hws, hMs⟩ := shadow_msg (planAtZKHP_S (Reg := Reg) dj y₀) S R u M hsat
  have hag := decAgreeHP dj y₀ ht hc hscope hU hpi S R u hu M Ms hws hsat g hg
  have hval : ∀ b, pubBit c hU n pi own g = some b →
      decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
        (tabAtZKHP dj y₀) (placeZHP dj y₀ ht hc hscope hU hpi) S R u M g = b := fun b hb => by
    rw [hag]
    simp only [Partition.placeDecoder, hu, ↓reduceDIte]
    exact keyPlaceHP_decode_pub (sitesShHP_parse dj y₀ ht hc hscope) pi hpi own (Layout.outsOf c hU n)
      (fun _ => pub) (sitesShHP_bcSite dj y₀ ht hc hscope hU hpi hpub) (fun _ _ _ _ => rfl) hu hMs hg hb
  refine ⟨fun h1 => hval true ?_, fun _ h0 => hval false ?_⟩
  · exact ((@Finset.mem_filter _ (fun g => pubBit c hU n pi own g = some true) (Classical.decPred _) _ _).1 h1).2
  · exact ((@Finset.mem_filter _ (fun g => pubBit c hU n pi own g = some false) (Classical.decPred _) _ _).1 h0).2

/-- **The public-input step: the compiled `--zk` count curve at the public transcript of the `--zk` tables with the
verifier's public inputs.** `zk_flock_countC_execOS_cr_pub` at `planAtZKHP` with `place := placeZHP`,
`vb := rsZHP.bindingZ rsZHP_computes` and the public gates `pubOnes`, `pubZeros` (`pubDecodesZ_placeZHP`): at the
outcomes it counts, the extracted value at every public-input gate is the verifier's own word's bit, except on the
knowledge (`ε_ks`, under `cr/sha-512` through `hCR`) and link (`δ_link`, under `ecr/sha-512`) events and the serving
roots' rewinding (`δ_tree`, under `cr/sha-512` through `hT`). Past the count curve's own hypotheses it takes the
session's inputs: untyped tags (`ht`), one circuit file `HmOut.parse` accepts as `c` under the `--zk` tags (`hc`) with
public input ports `pi` (`hpi`), the circuit in `rs`'s scope (`hscope`), and a public file `HmIn.loadPublic` loads
(`hpub`). -/
theorem zk_flock_countC_execOS_atHP (Hs : Hashes 𝔽 Dig (List UInt8))
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKHP dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith Hs
      (planAtZKHP dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat execArith Hs (planAtZKHP dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKHP dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
        (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
          (XplurZC execArith Hs ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
            (tabAtZKHP dj y₀)
            ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
            (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planAtZKHP dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKHP dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_cr_pub execArith Hs ((keyProgP c hU n pi).partition (Layout.outsOf c hU n))
    (planAtZKHP dj y₀) (tabAtZKHP dj y₀) execArith_correct σs ks hks St hS Kd pop Lb src hA3
    (placeZHP dj y₀ ht hc hscope hU hpi)
    ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
    (pubOnes c hU n pi own) (pubZeros c hU n pi own) (pubDecodesZ_placeZHP dj y₀ ht hc hscope hU hpi hpub)
    hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ZkPubIn
