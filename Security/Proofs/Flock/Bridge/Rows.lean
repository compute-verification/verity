import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.RowsC
import Proofs.Flock.Soundness.Registered.ReadsX
import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ

/-!
# The sampled audit's knowledge soundness at the registrant's rows, under `verify --zk` (`SampledKSLiveV2Z`)

`Bridge.SampledKSLiveXr` reads the claims' values off the extracted plurality (`XplurZC`): every wire carries its value
in the circuit's evaluation on `XplurZC`'s inputs, so a unit that computes correctly on whatever rows the session
committed passes, whatever rows were registered. `SampledKSLiveV2Z` reads them at the rows the registrant opens: at each
of the session's registered reads (`Registered.ReadsX`, as `EndToEndRegisteredRows` takes them) the bit of the
registrant's row (`rd.wv`), and elsewhere the evaluation on `XplurZC`'s inputs. So a unit that is correct on the
session's rows but not on the registrant's fails the claim. Its bound is `Bridge.SampledKSLiveXr`'s plus the rows term
`rowsPrZC`: the audit extractor's mass at the registered reads where it reads a value other than the registrant's.

`Flock.Bridge.sampledKSLiveV2Z_of_count` proves it from `TransportsLiveC`, `CountTarget` and `OpeningsCanonTarget`: a
faulty population slot is a unit wrong on the registration with canonical openings or off the registrant's rows
(`faultyRows`: a unit in neither carries the evaluation at every committed wire, by `Partition.correct_eval`), so the
count curve at `wrongRegZc ∪ rowsWrongZ` (`ZkLink.zk_flock_countPr_rowsC`) bounds the event by
`miss + ε_ks + δ_link + rowsPrZC`, each term at the finders' collision probabilities, with no hash premise.

Its event is `verify --zk`'s acceptance, which checks no registered read, so its rows term is at the finders only where
the registrant's leaves are the registered ones. `SampledKSLiveV2` (`RowsReg`) states it under
`verify --zk --registered F`, and `SampledKSLivePub` in the additive form; both are proved from this.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **The sampled audit's knowledge soundness, live, at the registrant's rows, under `verify --zk`.**
`Bridge.SampledKSLiveXr`'s setting
(circuit files of `sha512/row/v1`, `sha512/row/v2` or `sha512/row-seg/v1` ports, `Bridge.RowsOk`, each of one name,
with no colliding pair of roots, and a descriptor and query that evaluate to the game's `n` units in order), with a
registrant's opening `rd` of the session's registered reads (`Registered.ReadsX`, as in `EndToEndRegisteredRows`). The
claims' values (`vreg`) are, at each wire whose commit string is a registered read, the bit of the registrant's row
there, and at every other wire its value in the circuit's evaluation on the extracted plurality's inputs (`XplurZC`).
Then for every live prover,
$$\Pr[\text{flock-verify accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witRows}(v_{reg}))] \le
  \binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \varepsilon_{ks} + \delta_{link} + \delta_{rows},$$
`ε_ks`, `δ_link` `Bridge.SampledKSLiveXr`'s, its finders' probabilities of outputting a SHA-512 collision, and
`δ_rows` the rows term `rowsPrZC` at the registrant's openings. No collision-resistance hypothesis; its named assumptions
are A3 and `live-verifier`. -/
def SampledKSLiveV2Z : Prop :=
  ∀ (n : ℕ) (Reg : Type)
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
    -- the registrant's opening of the session's registered reads
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
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
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
    -- the claims' values: the registrant's row's bit at a registered read, and the evaluation on the extracted
    -- plurality's inputs at every other wire
    let vreg : Fin Cc.N → Bool := fun g =>
      if rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots vreg))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ)) +
        rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo k ρ (reg σ) (cont σ)

/-- **`SampledKSLiveV2Z` from the count curve at the registrant's rows**, canonical transport, the count lemma and
canonical openings. -/
theorem _root_.Flock.Bridge.sampledKSLiveV2Z_of_count (hT : Bridge.TransportsLiveC) (hC : Bridge.CountTarget)
    (hOC : Bridge.OpeningsCanonTarget) : SampledKSLiveV2Z := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω
    _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ q L P vb wo σ Xr δ Cc rows roots Cd κ vreg
  by_cases hk0 : kd = 0
  · refine (FlockSoundness.Game.prob_le_one _ _ τ).trans ?_
    have hm : ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) = 1 := by rw [hk0]; simp
    rw [hm]
    exact le_add_right (le_add_right (le_add_right le_rfl))
  have hkd1 : 1 ≤ kd := Nat.pos_of_ne_zero hk0
  set canon : Fin Cc.N → vb.Opening → Prop := fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2
  have hcan := hOC n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀
  have hTu := hT n Reg I ht D hD hU pub hpub hpt hscope hrows hdup (Bridge.PortRoots.fixes_of_none hroots) kd hkd hdiv
    dj S₀ d₀ y₀ wr dg N Ω Lb src hTags hCust τ
  have hκin : ∀ g b, κ g = some b → (Cc.op g).isInput = true := fun g b h =>
    Bridge.constsOf_isInput (Discharge.Layout.outsOf D.unroll hU n) h
  set W := wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon (reg σ) Xr ∪
    rowsWrongZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo Xr with hW
  have hsub : (Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
        (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots vreg)).filter (· < n) ⊆ W.map Fin.valEmbedding := by
    classical
    intro v hv
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
      (fun g hg hn o ho hc => hTu ⟨v, hlt⟩ g hg hn o ho (hc.2 hn)) ?_ vreg ?_ kd desc q hv
    · intro g hg b hb
      obtain ⟨-, hbit, hcan'⟩ := hop g hg
      exact hbit.symm.trans (hcan'.1 b hb)
    · intro g hg _
      show (if rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g) = _
      split_ifs with h
      · exact (hrw g hg h).symm.trans (heval g hg)
      · rfl
  have hK₁ : min K₀ (n - kd + 1) ≤ n := (min_le_left _ _).trans hK₀
  have hev : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
      (LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots vreg)) →
      LiveAcceptsZK I o ∧ min K₀ (n - kd + 1) ≤ W.card := fun o ho =>
    ⟨ho.1, (hC P kd K₀ hkd1 hkd _ _ κ desc q roots _ ho.2).trans
      ((Finset.card_le_card hsub).trans (Finset.card_map _).le)⟩
  have hcount := zk_flock_countPr_rowsC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon rd.isRead wo
    (Discharge.ZkSession.loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀)
      (placeZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU))
    (Discharge.Layout.keyProg_readsOwn D.unroll hU n _) hcan hRw hM hρ hr
    (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) (min K₀ (n - kd + 1)) σ
  refine (prob_mono hev _ τ).trans ((zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr
    (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
        hCust.1
        fun _ => Discharge.ZkHidden.drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ _ _
    hcount).trans ?_)
  have hmiss := Discharge.Tables.execOS_subset_miss_le kd hkd Lb src hA3 (min K₀ (n - kd + 1))
  rw [Law.subset_miss hkd hK₁, Bridge.choose_min_count hkd1 hkd] at hmiss
  have hks := ksAvgZC_le_pr execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ)
  exact (add_assoc _ _ _).symm.le.trans (add_le_add (add_le_add (add_le_add hmiss hks) le_rfl) le_rfl)

end Flock.Guarantees
