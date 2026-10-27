import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.RowsC
import Proofs.Flock.EndToEndRegistered
import Proofs.Flock.Soundness.Registered.ReadsAt

/-!
# The sampled audit's knowledge soundness at the registrant's rows (`SampledKSLiveV2`)

`Bridge.SampledKSLiveXr` reads the claims' values off the extracted plurality (`XplurZC`): every wire carries its value
in the circuit's evaluation on `XplurZC`'s inputs, so a unit that computes correctly on whatever rows the session
committed passes, whatever rows were registered. `SampledKSLiveV2` reads them at the rows the registrant opens: at a
registered read the verifier owns at the outcome's draw (`OwnReadsHJ`, as in `EndToEndRegisteredRows`), the bit of the
registrant's row (`rd.wv`), and elsewhere the evaluation on `XplurZC`'s inputs. So a unit that is correct on the
session's rows but not on the registrant's fails the claim. Its bound is `Bridge.SampledKSLiveXr`'s plus
`EndToEndRegisteredRows`' two terms: `δ_rows` (`δ_link`'s closed form, for the rows finders, `RowsCRZC`) and
`δ_reg = q_R²/2^513` (`Registered.ReadsX.prob_off_le_at`, `cr/sha-512` at budget `q_R` for the registered reads'
finder).

`Flock.Bridge.sampledKSLiveV2_of_xr` proves it from `Bridge.SampledKSLiveXr`, `TransportsLiveC`, `CountTarget` and
`OpeningsCanonTarget`. The registered reads' leaves are the public file's at every registration (`layShHJ`'s
`regLeaf` reads only `pub`), so whether some read is off the registrant's leaf (`rd.Off`) is one proposition:

* **off**: an outcome whose reads the verifier owns is accepted while a read is off, which `EndToEndRegisteredDrawn`'s
  registered check bounds by `δ_reg`; at any other, the claims' values are `Bridge.SampledKSLiveXr`'s;
* **not off**: the registrant's openings open the registered commitments at every read, and a faulty population slot
  is a unit wrong on the registration with canonical openings or off the registrant's rows (`faultyRows`: a unit in
  neither carries the evaluation at every committed wire, by `Partition.correct_eval`), so the count curve at
  `wrongRegZc ∪ rowsWrongZ` (`ZkLink.zk_flock_countC_rowsC`) bounds the event by `miss + ε_ks + δ_link + δ_rows`.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **The sampled audit's knowledge soundness, live, at the registrant's rows.** `Bridge.SampledKSLiveXr`'s setting
(circuit files of `sha512/row/v1`, `sha512/row/v2` or `sha512/row-seg/v1` ports, `Bridge.RowsOk`, each of one name,
with no colliding pair of roots, and a descriptor and query that evaluate to the game's `n` units in order), with F's
registered roots `own`, registered before the statement and its coin, and a registrant's opening `rd` of them at the
session's registered reads (`Registered.ReadsX`, as in `EndToEndRegisteredRows`). The claims' values at an outcome
(`vreg`) are, at each wire whose commit string is a registered read the verifier owns at the outcome's draw
(`OwnReadsHJ`), the bit of the registrant's row there, and at every other wire its value in the circuit's evaluation on
the extracted plurality's inputs (`XplurZC`). Then for every live prover,
$$\Pr[\text{flock-verify accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witRows}(v_{reg}))] \le
  \binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \varepsilon_{ks} + \delta_{link} + \delta_{rows} + \delta_{reg},$$
`ε_ks`, `δ_link` `Bridge.SampledKSLiveXr`'s, `δ_rows` `δ_link`'s closed form for the rows finders, and
`δ_reg = q_R²/2^513`. Its named assumptions are `Bridge.SampledKSLiveXr`'s (`Assumptions.TableCRZAt`,
`Assumptions.LinkCRZCAt`, A3 and `live-verifier`) and `EndToEndRegisteredRows`' two more: `ecr/sha-512` for the rows
finders at the registrant's openings (`RowsCRZC`) and `cr/sha-512` at budget `q_R` for the registered reads' finder
(`StrictCR.Finder.CR`). -/
def SampledKSLiveV2 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- F's registered roots: registered before the statement and its coin
    (own : Array Flock.Registered.Port)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.Registered.dupName D.unroll = none) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
    -- the statement's descriptor, and the registration's query, whose population is the game's n units in order
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    -- the registrant's opening of `own`'s roots at the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- one opening's verification, and the registered reads' finder's budget, in SHA-512 evaluations
    (v qR : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => Discharge.ZkHidden.drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    -- the deployed commitment's layout: the game's rows, read off the circuit file
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- the registration's frame-v3 roots, over the public file's rows
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    -- the circuit's constants, public
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    -- the claims' values at an outcome: the registrant's row's bit at a registered read the verifier owns at the
    -- outcome's draw, and the evaluation on the extracted plurality's inputs at every other wire
    let vreg : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Fin Cc.N → Bool := fun o g =>
      if OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt ∧ rd.isRead (vb.pos g) then
        vb.bit (vb.val (wo (vb.pos g))) g
      else Cc.eval Xr g
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs, at their cost: two runs of the
    -- prover and two verifications for a rewinding finder, one of each for the self-clash finder
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    -- ecr/sha-512 for the prover's link finders
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      k Rw M t' →
    -- ecr/sha-512 for the rows finders, at the registrant's openings
    RowsCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) rd.isRead wo k Rw t' →
    -- cr/sha-512 for the registered reads' finder: one play of the audit, and the extractor's pair at an off read
    (StrictCR.oneRun G τ fun o => rd.pick o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port)).CR H512 qR →
    prob (fun o => LiveAcceptsZKR I own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o)))
        G τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgStrictZ (planZHJ dj y₀) L k (2 * (t' + v)) (t' + v) (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
        ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ))

/-- **`SampledKSLiveV2` from `Bridge.SampledKSLiveXr`**, canonical transport, the count lemma and canonical openings. -/
theorem _root_.Flock.Bridge.sampledKSLiveV2_of_xr (hX : Bridge.SampledKSLiveXr) (hT : Bridge.TransportsLiveC)
    (hC : Bridge.CountTarget) (hOC : Bridge.OpeningsCanonTarget) : SampledKSLiveV2 := by
  intro n Reg own I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω
    _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ tv qR q L G P vb wo σ Xr δ Cc rows roots Cd κ
    vreg hTab hLink hRows hReg
  by_cases hk0 : kd = 0
  · refine (FlockSoundness.Game.prob_le_one _ _ τ).trans ?_
    have hm : ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) = 1 := by rw [hk0]; simp
    rw [hm]
    exact le_add_right (le_add_right (le_add_right (le_add_right le_rfl)))
  have hkd1 : 1 ≤ kd := Nat.pos_of_ne_zero hk0
  set own' := fun (o : Reg × Finset (Fin n) × Discharge.Exec.Transcript) =>
    OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt
  have hDr := Flock.SecurityProofs.EndToEndRegisteredDrawn n Reg I ht own D hD hU pub hpub hscope kd hdiv dj S₀ d₀ y₀
    wr dg N Ω Lb src hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' τ (2 * (t' + tv)) (t' + tv) hTab hLink
  have hReg' := rd.prob_off_le_at G τ (fun o => o.1) (fun o => LiveAcceptsZKR I own o ∧ own' o)
    (fun o => regPathsHJ I D.unroll pub o.2.2 rd.port) (fun o ho => hDr.2 o ho.1 rd ho.2) hReg
  have hcs : ∀ (R R' : Reg) p,
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf R p =
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf R' p :=
    fun _ _ _ => rfl
  have hoffc : ∀ R R' : Reg, rd.Off R → rd.Off R' := by
    intro R R' h
    obtain ⟨p, hp, hne⟩ := h
    exact ⟨p, hp, fun h' => hne ((hcs R R' p).trans h')⟩
  by_cases hoff : rd.Off (reg σ)
  · -- some registered read is off the registrant's leaf
    have hXr := hX n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N
      Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ tv q hTab hLink
    have hsub : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
        (LiveAcceptsZKR I own o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
            (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o))) →
        (LiveAcceptsZK I o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
            (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr)) ∨
        ((LiveAcceptsZKR I own o ∧ own' o) ∧ rd.Off o.1) := by
      rintro o ⟨hA, hrel⟩
      by_cases hown : own' o
      · exact Or.inr ⟨⟨hA, hown⟩, hoffc _ _ hoff⟩
      · refine Or.inl ⟨acceptsZK_of_R hA, ?_⟩
        have hv : vreg o = Cc.eval Xr := funext fun g => if_neg fun h => hown h.1
        rwa [Bridge.witDep_eq_witRows, ← hv]
    refine (prob_mono hsub G τ).trans ((prob_or_le _ _ G τ).trans ((add_le_add hXr hReg').trans ?_))
    exact add_le_add le_self_add le_rfl
  -- every registered read is on the registrant's leaf
  have hon : ∀ p, rd.isRead p → vb.com (wo p) = vb.cs (reg σ) p := by
    intro p hp
    have hl : (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf (reg σ) p =
        rd.leaf p := by
      by_contra h
      exact hoff ⟨p, hp, h⟩
    refine Prod.ext rfl ?_
    show FlockSoundness.Binding.ExplicitLeaf.leaf (FlockSoundness.Discharge.ZkBind.HmRowsZX.scheme
      (rsZHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub) p) (rd.wv p) (rd.ws p) = _
    unfold FlockSoundness.Discharge.ZkBind.HmRowsZX.scheme
    split_ifs with hrow
    · exact hl.symm
    · exact absurd (rd.read_row p hp) hrow
  set canon : Fin Cc.N → vb.Opening → Prop := fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2
  have hcan := hOC n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀
  have hTu := hT n Reg I ht D hD hU pub hpub hpt hscope hrows hdup (Bridge.PortRoots.fixes_of_none hroots) kd hkd hdiv
    dj S₀ d₀ y₀ wr dg N Ω Lb src hTags hCust τ
  have hκin : ∀ g b, κ g = some b → (Cc.op g).isInput = true := fun g b h =>
    Bridge.constsOf_isInput (Discharge.Layout.outsOf D.unroll hU n) h
  set W := wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon (reg σ) Xr ∪
    rowsWrongZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo Xr with hW
  have hsub : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
      (Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
        (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o))).filter (· < n) ⊆ W.map Fin.valEmbedding := by
    classical
    intro o v hv
    obtain ⟨hv, hlt⟩ := Finset.mem_filter.1 hv
    refine Finset.mem_map.2 ⟨⟨v, hlt⟩, ?_, rfl⟩
    by_contra hu
    have hu1 : ⟨v, hlt⟩ ∉ wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon (reg σ) Xr :=
      fun h => hu (Finset.mem_union_left _ h)
    have hu2 : ⟨v, hlt⟩ ∉ rowsWrongZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo Xr :=
      fun h => hu (Finset.mem_union_right _ h)
    obtain ⟨hcX, op, hop⟩ := not_not.1 (mt (mem_wrongRegZc _ _ _ _ _ _).2 hu1)
    have hrw : ∀ g ∈ P.io ⟨v, hlt⟩, rd.isRead (vb.pos g) → Xr g = vb.bit (vb.val (wo (vb.pos g))) g :=
      fun g hg hrd => not_not.1 fun hne =>
        hu2 ((mem_rowsWrongZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo).2 ⟨g, hg, hrd, hne⟩)
    have heval := P.correct_eval (Discharge.Layout.keyProg_readsOwn D.unroll hU n _ ⟨v, hlt⟩) hcX
    refine Bridge.faultyRows P vb.pos (fun o g => vb.bit (vb.val o) g) id vb.com (vb.cs (reg σ)) canon
      Claims.Sampling.sha512Msg Cd roots Xr κ ⟨v, hlt⟩ hκin ⟨op, hop⟩
      (fun g hg hn o ho hc => hTu ⟨v, hlt⟩ g hg hn o ho (hc.2 hn)) ?_ (vreg o) ?_ kd desc q hv
    · intro g hg b hb
      obtain ⟨-, hbit, hcan'⟩ := hop g hg
      exact hbit.symm.trans (hcan'.1 b hb)
    · intro g hg _
      show (if own' o ∧ rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g) = _
      split_ifs with h
      · exact (hrw g hg h.2).symm.trans (heval g hg)
      · rfl
  have hK₁ : min K₀ (n - kd + 1) ≤ n := (min_le_left _ _).trans hK₀
  have hev : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
      (LiveAcceptsZKR I own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o))) →
      LiveAcceptsZK I o ∧ min K₀ (n - kd + 1) ≤ W.card := fun o ho =>
    ⟨acceptsZK_of_R ho.1, (hC P kd K₀ hkd1 hkd _ _ κ desc q roots _ ho.2).trans
      ((Finset.card_le_card (hsub o)).trans (Finset.card_map _).le)⟩
  have hcount := zk_flock_countC_rowsC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon rd.isRead wo
    (Discharge.ZkSession.loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀)
      (placeZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU))
    (Discharge.Layout.keyProg_readsOwn D.unroll hU n _) hcan hRw hM hρ hr
    (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' (min K₀ (n - kd + 1)) σ
  refine (prob_mono hev G τ).trans ((zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr
    (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
        hCust.1
        fun _ => Discharge.ZkHidden.drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ _ _ hcount).trans ?_)
  have hmiss := Discharge.Tables.execOS_subset_miss_le kd hkd Lb src hA3 (min K₀ (n - kd + 1))
  rw [Law.subset_miss hkd hK₁, Bridge.choose_min_count hkd1 hkd] at hmiss
  have hks := ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hTab
  unfold rowsBoundZC
  split_ifs with h3
  · exact (add_assoc _ _ _).symm.le.trans
      (le_add_right (add_le_add_right (add_le_add_right (add_le_add hmiss hks) _) _))
  · exact (h3 ⟨hLink, hRows, hon⟩).elim

end Flock.Guarantees
