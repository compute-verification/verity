import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.Refused

/-!
# The sampled audit's knowledge soundness at the registrant's rows, under `verify --zk --registered F`

`SampledKSLiveV2Z` (`Rows`) bounds the event that `verify --zk` accepts and the claims fail, the claims read at the
registrant's rows (`Registered.ReadsX`, `rd.wv`). `verify --zk` checks no registered read, so its rows term `rowsPrZC`
is at the finders only where the registrant's leaves are the registered ones (`rowsPrZC_le_pr`).

`SampledKSLiveV2` states it under `verify --zk --registered F` (`ZkReg.LiveAcceptsZKR`), with `SampledKSLivePub`'s
claims' values: at each registered read the registrant's row's bit, where the registrant's leaves are the registered
ones (`¬ rd.Off`) or the verifier owns the registered reads at the outcome's draw (`ZkReg.OwnReadsHJ`), and elsewhere
the evaluation on the extracted plurality's inputs. At an accepted outcome every port of each table's circuit is one of
F's registered ports (`ZkReg.complete_of_R`, `verify`'s `checkComplete`). Its bound is `SampledKSLiveV2Z`'s, its rows
term read where `¬ rd.Off`, plus the registered reads' finder's probability of outputting a SHA-512 collision
(`Registered.ReadsX.pick` through the registered check's paths, at one play of the audit). `sampledKSLiveV2_of` proves it
from `Bridge.SampledKSLiveXr` and `SampledKSLiveV2Z`, as `sampledKSLivePub_of` does: the registered reads' leaves are the
public file's at every registration, so `rd.Off` is one proposition. Where it holds, an accepted outcome whose reads the
verifier owns gives a collision at an off read (`checked_of_acceptsHJ`, `Registered.ReadsX.offAt_conflict`), and the
claims at every other outcome are `Bridge.SampledKSLiveXr`'s; where it fails, the claims are `SampledKSLiveV2Z`'s.

`SampledKSLiveV3` reads V2's file hypotheses off acceptance (`Bridge.LiveAcceptsKSR`, `Bridge.RunPopOk`), as
`Refused` describes.
-/

namespace Flock.Bridge

open FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkReg in
/-- **`verify --zk --registered F` accepts, files and transcript**: the live audit's transcript under F's registered
roots `own` (`LiveAcceptsZKR`), and the files' rows, names and roots (`Flock.PortRoots.check`). -/
def LiveAcceptsKSR {n : ℕ} {Reg : Type} (I : FlockSoundness.Discharge.Exec.Inputs) (own : Array Flock.Registered.Port)
    (o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) : Prop :=
  LiveAcceptsZKR I own o ∧ Flock.PortRoots.check (zkTags I) I.circuitFile I.publicFile I.tables = .ok ()

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **The sampled audit's knowledge soundness, live, at the registrant's rows, under `verify --zk --registered F`.**
`SampledKSLiveV2Z`'s setting and named assumptions, with F's registered roots `own`. The claims' values at an outcome
(`vreg`) are `SampledKSLivePub`'s: at each wire whose commit string is a registered read, the bit of the registrant's
row there (`rd.wv`) where the registrant's leaves are the registered ones (`¬ rd.Off`) or the verifier owns the
registered reads at the outcome's draw (`OwnReadsHJ`), and at every other wire its value in the circuit's evaluation on
the extracted plurality's inputs (`XplurZC`). For every live prover,
$$\Pr[\text{flock-verify --registered accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witRows}(v_{reg}))] \le
  \binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \varepsilon_{ks} + \delta_{link} + \delta_{rows} + \Pr[B_R^\tau],$$
`ε_ks`, `δ_link` `SampledKSLiveV2Z`'s, `δ_rows` its rows term `rowsPrZC` at the registered reads where `¬ rd.Off`, and
`Pr[B_R^τ]` the probability that the registered reads' finder (one play of the audit, at budget `q_R`, outputting the
registered check's pair at an off read) outputs a SHA-512 collision. No collision-resistance hypothesis. -/
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
    -- the registrant's opening of the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
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
    -- one play of the audit, in SHA-512 evaluations
    (qR : ℝ)
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
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
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
    -- the claims' values at an outcome: the registrant's row's bit at a registered read, where the registrant's leaves
    -- are the registered ones or the verifier owns the reads at the outcome's draw, and the evaluation on the extracted
    -- plurality's inputs at every other wire
    let vreg : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Fin Cc.N → Bool := fun o g =>
      if (¬ rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.isRead (vb.pos g) then
        vb.bit (vb.val (wo (vb.pos g))) g
      else Cc.eval Xr g
    -- the registered reads' finder: it plays the audit once and outputs the registered check's pair at an off read
    let BR := Bridge.strictComp (StrictCR.oneRun G τ fun o =>
        rd.pick o.1 (Discharge.ZkReg.regPathsHJ I D.unroll pub o.2.2 rd.port)) qR
    prob (fun o => Discharge.ZkReg.LiveAcceptsZKR I own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o)))
        G τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ)) +
        rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (fun c => ¬ rd.Off (reg σ) ∧ rd.isRead c) wo k ρ
          (reg σ) (cont σ) +
        ENNReal.ofReal (BR.Pr Claims.Sampling.sha512Msg)

open Classical FlockSoundness.Discharge.ZkReg in
/-- **`SampledKSLiveV2` from `Bridge.SampledKSLiveXr` and `SampledKSLiveV2Z`.** The registered reads' leaves are the
public file's at every registration, so `rd.Off` is one proposition.

* **Off**: the rows term is `0` (`rowsPrZC_noReads`). An accepted outcome whose reads the verifier owns has a SHA-512
  collision at an off read (`checked_of_acceptsHJ`, `Registered.ReadsX.offAt_conflict`), the registered reads'
  finder's output on that play; at every other outcome the claims are `Bridge.SampledKSLiveXr`'s.
* **Not off**: the claims are `SampledKSLiveV2Z`'s at every outcome. -/
theorem sampledKSLiveV2_of (hX : Bridge.SampledKSLiveXr) (hV : SampledKSLiveV2Z) : SampledKSLiveV2 := by
  intro n Reg own I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω
    _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ qR q L G P vb wo σ Xr δ Cc rows roots Cd κ vreg BR
  set pk := fun o : Reg × Finset (Fin n) × Discharge.Exec.Transcript =>
    rd.pick o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port)
  have hcs : ∀ (R R' : Reg) c,
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf R c =
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf R' c :=
    fun _ _ _ => rfl
  have hoffc : ∀ R R' : Reg, rd.Off R → rd.Off R' := by
    intro R R' h
    obtain ⟨c, hc, hne⟩ := h
    exact ⟨c, hc, fun h' => hne ((hcs R R' c).trans h')⟩
  by_cases hoff : rd.Off (reg σ)
  · -- some registered read is off the registrant's leaf
    have hXr := hX n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N
      Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ q
    have hsub : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
        (LiveAcceptsZKR I own o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
            (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o))) →
        (LiveAcceptsZK I o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
            (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr)) ∨
        ((LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.Off o.1) := by
      rintro o ⟨hA, hrel⟩
      by_cases hown : OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt
      · exact Or.inr ⟨⟨hA, hown⟩, hoffc _ _ hoff⟩
      · refine Or.inl ⟨acceptsZK_of_R hA, ?_⟩
        have hv : vreg o = Cc.eval Xr := funext fun g =>
          if_neg fun h => h.1.elim (fun hn => hn (hoffc _ _ hoff)) hown
        rwa [Bridge.witDep_eq_witRows, ← hv]
    have hreg : prob (fun o : Reg × Finset (Fin n) × Discharge.Exec.Transcript =>
        (LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.Off o.1) G τ ≤ ENNReal.ofReal (BR.Pr Claims.Sampling.sha512Msg) := by
      have hcoll : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
          (LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
            rd.Off o.1 → StrictCR.IsColl H512 (pk o) := by
        rintro o ⟨⟨hA, hO⟩, hoff'⟩
        have hc : rd.Checked o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port) :=
          checked_of_acceptsHJ I ht own dj y₀ (Flock.Desc.parse_flat hD) hU hpub hscope hTags o hA rd hO
        exact (rd.openings o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port) (rd.offAt o.1)).isColl
          (rd.offAt_conflict hc hoff')
      have hle : prob (fun o : Reg × Finset (Fin n) × Discharge.Exec.Transcript =>
          (LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
            rd.Off o.1) G τ ≤ prob (fun o => StrictCR.IsColl H512 (pk o)) G τ :=
        prob_mono hcoll G τ
      have heq : (prob (fun o => StrictCR.IsColl H512 (pk o)) G τ).toReal = BR.Pr Claims.Sampling.sha512Msg := by
        show _ = (Bridge.strictComp (StrictCR.oneRun G τ pk) qR).Pr Claims.Sampling.sha512Msg
        rw [Bridge.strictComp_Pr, StrictCR.oneRun_ev, prob_toReal]
        all_goals rfl
      have hfin' : prob (fun o => StrictCR.IsColl H512 (pk o)) G τ ≠ ⊤ :=
        ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
      have hB : prob (fun o => StrictCR.IsColl H512 (pk o)) G τ = ENNReal.ofReal (BR.Pr Claims.Sampling.sha512Msg) := by
        rw [← heq, ENNReal.ofReal_toReal hfin']
      exact hle.trans hB.le
    have hnr : (fun c => ¬ rd.Off (reg σ) ∧ rd.isRead c) = fun _ => False :=
      funext fun c => propext ⟨fun h => h.1 hoff, False.elim⟩
    rw [hnr, rowsPrZC_noReads, add_zero]
    exact (prob_mono hsub G τ).trans ((prob_or_le _ _ G τ).trans (add_le_add (hXr.trans le_rfl) hreg))
  -- every registered read is on the registrant's leaf
  have hno : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript, ¬ rd.Off o.1 := fun o h => hoff (hoffc _ _ h)
  have hVi := hV n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg
    N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ q
  have hsub : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
      (LiveAcceptsZKR I own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o))) →
      LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots fun g =>
            if rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g) := by
    rintro o ⟨hA, hrel⟩
    refine ⟨acceptsZK_of_R hA, ?_⟩
    have hv : vreg o = fun g => if rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g :=
      funext fun g => by
        show ite _ _ _ = ite _ _ _
        by_cases h : rd.isRead (vb.pos g)
        · exact (if_pos (And.intro (Or.inl (hno o)) h)).trans (if_pos h).symm
        · exact (if_neg (not_and_of_not_right _ h)).trans (if_neg h).symm
    rwa [hv] at hrel
  have hon : (fun c => ¬ rd.Off (reg σ) ∧ rd.isRead c) = rd.isRead :=
    funext fun c => propext ⟨And.right, fun h => ⟨hoff, h⟩⟩
  rw [hon]
  exact (prob_mono hsub G τ).trans (hVi.trans le_self_add)

/-- **`SampledKSLiveV2` with its file preconditions read off acceptance**: `SampledKSLiveV2`'s bound on the event that
`verify --zk --registered F` accepts the transcript and the files (`Bridge.LiveAcceptsKSR`: their ports row/v1, row/v2
or row-seg/v1 rows of distinct names, and no two registered roots, each framed at its port's schema, one root under
different domains, at the public file's population), the sampled-proofs run passes the registration's population
(`Bridge.RunPopOk`: the descriptor `desc` and the query `qr` give the game's `n` units in order), and the claims
relation fails. It has none of V2's hypotheses `RowsOk`, `dupName = none`, `rootsColl = none` and `PopMatches`; the
game's `n` is the public file's population (`pub.n = n`). Its claims' values and its terms are V2's. `SampledKSLiveV2`
implies it (`sampledKSLiveV3_of`). -/
def SampledKSLiveV3 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- F's registered roots: registered before the statement and its coin
    (own : Array Flock.Registered.Port)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the game's units are the public file's population
    (_hn : pub.n = n)
    -- the statement's descriptor, and the registration's query
    (desc : Claims.Msg) (qr : Lean.Json)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    -- the registrant's opening of the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
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
    -- one play of the audit, in SHA-512 evaluations
    (qR : ℝ)
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
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
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
    -- the claims' values at an outcome, V2's
    let vreg : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Fin Cc.N → Bool := fun o g =>
      if (¬ rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.isRead (vb.pos g) then
        vb.bit (vb.val (wo (vb.pos g))) g
      else Cc.eval Xr g
    -- the registered reads' finder, V2's
    let BR := Bridge.strictComp (StrictCR.oneRun G τ fun o =>
        rd.pick o.1 (Discharge.ZkReg.regPathsHJ I D.unroll pub o.2.2 rd.port)) qR
    prob (fun o => (Bridge.LiveAcceptsKSR I own o ∧ Bridge.RunPopOk desc qr n) ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o)))
        G τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ)) +
        rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (fun c => ¬ rd.Off (reg σ) ∧ rd.isRead c) wo k ρ
          (reg σ) (cont σ) +
        ENNReal.ofReal (BR.Pr Claims.Sampling.sha512Msg)

/-- **`SampledKSLiveV2` gives `SampledKSLiveV3`**: at files that pass the verifiers' checks they give V2's hypotheses
(`Bridge.rowsOk_of_scopeOkH`, `Flock.PortRoots.check_ok`, `Bridge.rootsColl_none_of_collides_false` at `pub.n = n`,
`Bridge.popMatches_of_popOk`) and V3's event is inside V2's; at files that fail, V3's event is empty. -/
theorem sampledKSLiveV3_of (h : SampledKSLiveV2) : SampledKSLiveV3 := by
  intro n Reg own I ht D hD hU pub hpub hpt hscope hn desc qr kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω _ _ Lb src hA3 hTags
    hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ qR q L G P vb wo σ Xr δ Cc rows roots Cd κ vreg BR
  by_cases hc : Flock.PortRoots.check (zkTags I) I.circuitFile I.publicFile I.tables = .ok () ∧
      Bridge.RunPopOk desc qr n
  · obtain ⟨hck, p, hp, hpo⟩ := hc
    obtain ⟨_, hdup, hcoll⟩ := Flock.PortRoots.check_ok hck (Flock.Desc.parse_flat hD) hpub
    have hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none := by
      rw [← hn]
      exact Bridge.rootsColl_none_of_collides_false hcoll
    exact (prob_mono (fun o ho => ⟨ho.1.1.1, ho.2⟩) _ τ).trans
      (h n Reg own I ht D hD hU pub hpub hpt hscope (Bridge.rowsOk_of_scopeOkH hscope) hdup hroots desc qr
        (Bridge.popMatches_of_popOk hp hpo) kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM
        ρ hρ hr K₀ hK₀ τ qR q)
  · exact (prob_mono (q := fun _ => False) (fun o ho => hc ⟨ho.1.1.2, ho.1.2⟩) _ τ).trans
      ((prob_false _ τ).le.trans zero_le)

end Flock.Guarantees

/-- **The sampled audit's knowledge soundness, live, at the registrant's rows, under `verify --zk --registered F`**
(`Flock.Guarantees.SampledKSLiveV2`). -/
theorem Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2 :=
  Flock.Guarantees.sampledKSLiveV2_of Flock.Bridge.sampledKSLiveXr Flock.Bridge.sampledKSLiveV2Z

/-- **The sampled audit's knowledge soundness, live, on the event that the verifiers accept the files under
`verify --zk --registered F`** (`Flock.Guarantees.SampledKSLiveV3`): no hypothesis on the files' rows, names, roots or
population order. -/
theorem Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3 :=
  Flock.Guarantees.sampledKSLiveV3_of Flock.SecurityProofs.SampledKSLiveV2
