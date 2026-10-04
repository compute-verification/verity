import FlockSoundness.Discharge.ZkReg.Vals
import FlockSoundness.Discharge.ZkPubIn.Sound

/-!
# The drawn units and the registrant's values of a `--zk --registered --public-inputs` session

`zk_session_drawnR` and `zk_session_regValsR` at `ZkPubIn.zk_session_soundHRP`'s session: `flock-verify verify --zk
--registered R --public-inputs F` sessions over hidden outputs, at the committed transcript `zk_session_soundHRP` reads,
`Xpub (pubOnes …) (pubZeros …) XplurZC` at `simZA … τ` (the verifier's own bit on every public-input gate, the plurality
value elsewhere), with the registered reads of the shadow layout `layShHP`.

* `zk_headline_exec_drawnA`: `zk_headline_exec_drawn` at flock-verify's acceptance `acc` (`ZkPubIn.simZA`);
* `zk_flock_drawnC_cr_pub`: `zk_flock_drawnC_cr` at the public transcript, under `ZkPubIn.PubDecodesZ` (the link event
  at the public transcript is the link event at `XplurZC`, `ZkPubIn.zk_linkSoundC_pub`);
* `linkEvRegZK_of_pub`, `linkSoundRegZC_mono`, `zk_linkSoundRegC_pub`: the same for the registered link event, so
* `zk_flock_regValsC_cr_pub`: `zk_flock_regValsC_cr` at the public transcript;
* `zk_session_drawnHRP`: no drawn unit is wrong at the public transcript, except with probability `ε_ks + δ_link`; the
  `hwrong` of the bridge's `FlockBridge.Registered.registered_core` (`bridge/FlockBridge/Registered.lean`), with its
  `Off` false;
* `zk_session_regValsHRP`: wherever a drawn unit reads the registrant's leaf at a registered read, the public
  transcript's value is the registrant's, except with probability `ε_ks + δ_link + δ_reg`: the bridge's
  `FlockBridge.RegisteredReads.RegValsSound` (`bridge/FlockBridge/RegisteredReads.lean`) at `layShHP`'s reads `rd`.
-/

namespace FlockSoundness.Discharge.ZkReg

open scoped ENNReal

section Generic

open Game Model Accounting Extract Audit Audit.Partition ZkSession ZkLink StrictCR ZkPubIn

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Audit.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

omit [Fintype K] [DecidableEq K] in
/-- **The registered link event at the public transcript is the registered link event at `vals`**
(`ZkPubIn.linkEvZK_of_pub`): a satisfying witness decodes each public gate to its public value. -/
theorem linkEvRegZK_of_pub {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab decode Hc) (wo : vb.Pos → vb.Opening) {ones zeros : Finset (Fin C.N)}
    (hpub : PubDecodesZ A P planZ tab decode ones zeros) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    {R' : Reg} {vals : Fin C.N → Bool} {T : Oracle F}
    (h : LinkEvRegZK A P planZ tab vb wo S R u R' (Xpub ones zeros vals) T) :
    LinkEvRegZK A P planZ tab vb wo S R u R' vals T := by
  refine ⟨h.1, fun M hM hsat => ?_⟩
  obtain ⟨g, hg, hor⟩ := h.2 M hM hsat
  refine ⟨g, hg, hor.imp (fun hne heq => hne ?_) id⟩
  obtain ⟨h1, h0⟩ := hpub S R u hu M hsat g hg
  by_cases hg1 : g ∈ ones
  · rw [Xpub_of_mem_ones hg1]; exact h1 hg1
  · by_cases hg0 : g ∈ zeros
    · rw [Xpub_of_mem_zeros hg1 hg0]; exact h0 hg1 hg0
    · rw [Xpub_of_not_mem hg1 hg0]; exact heq

omit [Fintype K] in
/-- **The registered link term is monotone in the link event** (`ZkPubIn.linkSoundZC_mono` at `analysisRegZC`). -/
theorem linkSoundRegZC_mono {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab decode Hc) (wo : vb.Pos → vb.Opening)
    {Xc Xc' : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)} {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a)
    (hsub : ∀ R τ S u, u ∈ S → ∀ R' T, LinkEvRegZK A P planZ tab vb wo S R u R' (Xc' R τ) T →
      LinkEvRegZK A P planZ tab vb wo S R u R' (Xc R τ) T)
    {δlink : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → ℝ≥0∞}
    (h : (analysisRegZC A Hs P planZ tab vb wo Xc k hr hlow a hj).LinkSound δlink) :
    (analysisRegZC A Hs P planZ tab vb wo Xc' k hr hlow a hj).LinkSound δlink := by
  classical
  intro R τ tgt
  refine le_trans (avg_mono fun ω => ?_) (h R τ tgt)
  cases htg : tgt ω with
  | none => simp only [atTgt, Option.elim_none, le_refl]
  | some u =>
    simp only [atTgt, Option.elim_some]
    split_ifs with hu
    · show linkTermRegZC A Hs P planZ tab vb wo k (L.draw ω) R (τ ω) u (R, Xc' R τ) ≤
        linkTermRegZC A Hs P planZ tab vb wo k (L.draw ω) R (τ ω) u (R, Xc R τ)
      unfold linkTermRegZC
      apply ENNReal.ofReal_le_ofReal
      have hr1 : 0 ≤ 1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k := by
        linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
      refine div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left ?_ (accProbZC_nonneg _ _ _ _)) hr1
      unfold extractProbAccZ
      refine Extract.expectAcc_mono _ _ _ _ fun bs => ?_
      unfold ind
      split_ifs with h1 h2
      · exact le_refl _
      · exact absurd (hsub R τ _ u hu _ _ h1) h2
      · exact zero_le_one
      · exact le_refl _
    · exact le_refl _

/-- **The registered link theorem at the public transcript** (`zk_linkSoundRegC` at `Xpub ones zeros XplurZC`). -/
theorem zk_linkSoundRegC_pub {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab decode Hc) (wo : vb.Pos → vb.Opening) (ones zeros : Finset (Fin C.N))
    (hpub : PubDecodesZ A P planZ tab decode ones zeros)
    (hlow : LoweringSoundZK A P planZ tab decode) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (qW : ℝ) :
    (analysisRegZC (L := L) A Hs P planZ tab vb wo
        (fun R τ => Xpub ones zeros (XplurZC A Hs P planZ tab vb R τ k Rw)) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).LinkSound
      (fun R τ => linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ +
        regBoundZC A Hs P planZ tab vb wo k Rw ρ qW R τ) := by
  exact linkSoundRegZC_mono A Hs P planZ tab vb wo _ hlow hj
    (fun _ _ _ _ hu _ _ h => linkEvRegZK_of_pub A P planZ tab vb wo hpub hu h)
    (zk_linkSoundRegC A Hs P planZ tab vb wo hlow hRw hM hρ hr hj ht qW)

/-- **The compiled `--zk` audit's drawn units at the public transcript** (`zk_flock_drawnC_cr` at
`Xpub ones zeros XplurZC`, under `PubDecodesZ` at the places' decoder). -/
theorem zk_flock_drawnC_cr_pub (hA : A.Correct) {L : Law n}
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (ones zeros : Finset (Fin C.N))
    (hpub : PubDecodesZ A P planZ tab (decodeOfPlace A P planZ tab place) ones zeros)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS) :
    prob (fun o => o.1 = true ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) := by
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  exact (extraction_audit_drawn
      (analysisZC A Hs P planZ tab (decodeOfPlace A P planZ tab place)
        (fun R τ => Xpub ones zeros (XplurZC A Hs P planZ tab vb R τ k Rw)) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
      (zc_knowledgeSound A Hs P planZ tab _ _ _ hlow hj)
      (zk_linkSoundC_pub A Hs P planZ tab vb ones zeros hpub hlow hRw hM hρ hr hj ht) σ).trans
    (add_le_add (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR) le_rfl)

/-- **The compiled `--zk` audit's drawn units agree with the registered values at the public transcript**
(`zk_flock_regValsC_cr` at `Xpub ones zeros XplurZC`, under `PubDecodesZ` at the places' decoder). -/
theorem zk_flock_regValsC_cr_pub (hA : A.Correct) {L : Law n}
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (wo : vb.Pos → vb.Opening)
    (ones zeros : Finset (Fin C.N))
    (hpub : PubDecodesZ A P planZ tab (decodeOfPlace A P planZ tab place) ones zeros)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {qW : ℝ} (hW : RegCRZC A Hs P planZ tab vb wo (reg σ) (cont σ) k Rw qW) :
    prob (fun o => o.1 = true ∧ ∃ u ∈ o.2, ∃ g ∈ P.io u,
        RegBadZ A P planZ tab vb wo (reg σ)
          (Xpub ones zeros (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)) g)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
        regTermZC A P planZ tab vb L k Rw ρ qW := by
  classical
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  have h := extraction_audit_drawn
    (analysisRegZC A Hs P planZ tab vb wo (fun R τ => Xpub ones zeros (XplurZC A Hs P planZ tab vb R τ k Rw)) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
    (reg_knowledgeSoundZC A Hs P planZ tab vb wo _ _ hlow hj)
    (zk_linkSoundRegC_pub A Hs P planZ tab vb wo ones zeros hpub hlow hRw hM hρ hr hj ht qW) σ
  have hb : regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ) =
      regTermZC A P planZ tab vb L k Rw ρ qW := ite_eq_left hW
  refine le_trans (prob_mono ?_ _ _) (h.trans ?_)
  · rintro o ⟨h1, u, hu, g, hg, hbad⟩
    exact ⟨h1, Finset.not_disjoint_iff.2 ⟨u, Finset.mem_union_right _
      (Finset.mem_filter.2 ⟨Finset.mem_univ _, g, hg, hbad⟩), hu⟩⟩
  · show ksAvgZC A Hs planZ k _ (reg σ) (cont σ) +
        (linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
          regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ)) ≤ _
    rw [hb, ← add_assoc]
    exact add_le_add (add_le_add (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR) le_rfl) le_rfl

end Generic

section OnA

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL Composed ZkPubIn

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (acc : Exec.Transcript → Prop)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) {L : Law n}

/-- **The `--zk` headline at flock-verify's acceptance `acc` and an event of the registration and the draw**
(`zk_headline_exec_drawn` at `simZA`). -/
theorem zk_headline_exec_drawnA (hdec : DecodesOnZKA A Hs T acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (Q : Reg → Finset (Fin n) → Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q (reg (simZA A Hs T acc wr opn dg dec one fin capOf N hdec τ)) o.2)
      (audit L Reg (batchedSessionZC A Hs (planOf T))) (simZA A Hs T acc wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => acc o.2.2 ∧ Q o.1 o.2.1) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤ B := by
  have hc : ∀ (g : Finset (Fin n) → Reg → Game Bool)
      (h : ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = g),
      prob (fun o => o.1 = true ∧ Q (reg (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
          (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ))) o.2) (audit L Reg g)
          (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
            (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ)) =
        prob (fun o => o.1 = true ∧ Q (reg (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ)) o.2)
          (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
          (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ) := by
    intro g h
    subst h
    rfl
  have hB' := (hc _ (sessionZC_planOf A Hs T)).symm.trans_le hB
  refine exec_le_of_refines_drawn τ (simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ) rfl acc
    (fun ω => ?_) Q hB'
  have h : prob (fun o => acc o.2.2) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t))
      (τ.2 ω) ≤
      prob (fun o => o.1 = true)
        ((ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simAuditZKOnA A Hs T acc wr opn dg dec one fin capOf N hdec τ).2 ω) := by
    unfold simAuditZKOnA ZkExec.sessionZC
    rw [prob_strategy_map, prob_strategy_map, prob_map]
    unfold liveOf
    rw [prob_map]
    exact live_le (outz Hs T (L.draw ω) τ.1) _ (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg
      (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N
      (Gz A Hs T (L.draw ω) τ.1) (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1) _
  rw [prob_map] at h
  exact h.trans_eq (prob_map _ _ _ _)

end OnA

section Session

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden ZkPubIn

variable {mPts n : ℕ} {Reg : Type}

/-- **The registrant's opening** of each commit string of `rsZHP`'s value binding: the row and the salt `rd` reads
there. -/
def regOpenZHP {I : Inputs} {own : ByteArray} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀)) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOk c = true)
    (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pub : Public}
    (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub)) :
    ((rsZHP (Reg := Reg) dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub)).Pos →
      ((rsZHP (Reg := Reg) dj y₀ ht hc hscope hU hpi hpub).bindingZ
        (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub)).Opening :=
  fun p => (p, rd.wv p, rd.ws p)

/-- **No drawn unit of a `--zk --registered --public-inputs` session is wrong at its public transcript, except with
probability `ε_ks + δ_link`**: `zk_session_drawnR` at `zk_session_soundHRP`'s session, on `zk_session_soundHRP`'s
assumptions without A3 (the law's count curve is not read). -/
theorem zk_session_drawnHRP
    (I : Inputs) (ht : I.tags.typed = false) (rports : Array Flock.Registered.Port) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit, its public input ports and the public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHP I own dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkH I) (hRec : RecordsLiveZKP I own dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKHP dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKHP dj y₀)
        (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKRP I rports own o ∧
        ¬ Disjoint (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
          (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
            (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
              ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
              (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
              (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw)))
          o.2.1)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ksAvgStrictZ (planAtZKHP dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) := by
  have hdec := decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _
    (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)
  have hc' := zk_flock_drawnC_cr_pub execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
    (tabAtZKHP dj y₀) execArith_correct (placeZHP dj y₀ ht hc hscope hU hpi) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
    (pubOnes c hU n pi own) (pubZeros c hU n pi own) (pubDecodesZ_placeZHP dj y₀ ht hc hscope hU hpi hpub)
    hk hRw hM hρ hr ht' (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ) hCR
  exact le_trans (prob_mono (fun o h => ⟨acceptsZKP_of_RP h.1, h.2⟩) _ _)
    (zk_headline_exec_drawnA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ (fun _ S => ¬ Disjoint
      (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
          (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
            (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ)) (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ)) k Rw))) S) _ hc')

/-- **The public transcript of a `--zk --registered --public-inputs` session is the registered values wherever a
drawn unit reads the registrant's leaf**, except with probability `ε_ks + δ_link + δ_reg`: `zk_session_regValsR` at
`zk_session_soundHRP`'s session, for every registered-reads record `rd` of `layShHP`, on `zk_session_soundHRP`'s
assumptions without A3 and with `cr/sha-512` at budget `qW` for the registered-value finders of `τ`'s simulation
(`hW`). -/
theorem zk_session_regValsHRP
    (I : Inputs) (ht : I.tags.typed = false) (rports : Array Flock.Registered.Port) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit, its public input ports and the public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHP I own dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkH I) (hRec : RecordsLiveZKP I own dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the registered reads
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    -- the live prover, and cr/sha-512 for the knowledge and registered-value finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKHP dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKHP dj y₀)
        (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
      ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
      (regOpenZHP dj y₀ ht hc hscope hU hpi hpub rd)
      (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
      (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKRP I rports own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).io u,
          rd.isRead ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
          (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).regLeaf o.1 ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) =
            rd.leaf ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
          (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
            (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
              ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
              (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
              (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw)) g ≠
            (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).bit (rd.wv ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g)) g)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ksAvgStrictZ (planAtZKHP dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) +
        regTermZC execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP (Reg := Reg) dj y₀)
          (tabAtZKHP dj y₀) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k Rw ρ qW := by
  have hdec := decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _
    (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)
  have hc' := zk_flock_regValsC_cr_pub execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
    (tabAtZKHP dj y₀) execArith_correct (placeZHP dj y₀ ht hc hscope hU hpi) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
    (regOpenZHP dj y₀ ht hc hscope hU hpi hpub rd)
    (pubOnes c hU n pi own) (pubZeros c hU n pi own) (pubDecodesZ_placeZHP dj y₀ ht hc hscope hU hpi hpub)
    hk hRw hM hρ hr ht' (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ) hCR hW
  refine le_trans (prob_mono (fun o h => ⟨acceptsZKP_of_RP h.1, h.2⟩) _ _)
    (zk_headline_exec_drawnA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ (fun R S => ∃ u ∈ S,
      ∃ g ∈ ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).io u,
        rd.isRead ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
        (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).regLeaf R ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) = rd.leaf ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
        (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
          (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
            (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ)) (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ)) k Rw)) g ≠
          (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).bit (rd.wv ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g)) g) _
      (le_trans (prob_mono ?_ _ _) hc'))
  rintro o ⟨hacc, u, hu, g, hg, hread, hleaf, hne⟩
  refine ⟨hacc, u, hu, g, hg, ?_, hne⟩
  refine Prod.ext rfl ?_
  show ((rsZHP dj y₀ ht hc hscope hU hpi hpub).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZ.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

end Session

end FlockSoundness.Discharge.ZkReg
