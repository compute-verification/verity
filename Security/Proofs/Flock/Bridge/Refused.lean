import Proofs.Flock.Bridge.Canon
import Flock.PortRoots

/-!
# `SampledKSLiveV2`'s file preconditions, refused by the verifiers (`SampledKSLiveV3`)

`SampledKSLiveV2` takes four things of the files as hypotheses: every port a row/v1, row/v2 or row-seg/v1 row
(`Bridge.RowsOk`, which the scope check `scopeOkH` already gives: `Bridge.rowsOk_of_scopeOkH`), each of its own name
(`Registered.dupName`), no two ports' registered roots one root under different domains (`PortRoots.rootsColl = none`),
and the registration's population the game's `n` units in order (`Bridge.PopMatches`).
The verifiers of record now refuse every file that fails the last three: `verify --zk` runs `Flock.PortRoots.check`
last in its verdict (`Main.verifyCmd`, with or without `--ks-form`, the check's old flag), and the sampled-proofs run
refuses the registration (`SampledProofs.Run.run`,
`population-order`, `SampledProofs.Slots.popOk`). So the guarantee reads them off acceptance.

* `Bridge.LiveAcceptsKS I o`: `verify --zk` accepts: the live audit's transcript (`LiveAcceptsZK`), and the files pass
  `Flock.PortRoots.check`, whose roots are framed at each port's schema, as `Bridge.rootsOf`'s.
* `Bridge.RunPopOk desc qr n`: the descriptor parses and its population under the query is `n` units in order
  (`Slots.popOk`), which the run checks at registration.
* `rowsOk_of_scopeOkH`, `rootsColl_none_of_collides_false`, `popMatches_of_popOk`: the checks give V2's hypotheses. The
  roots check is computed at the public file's population, so the game's `n` is identified with it (`pub.n = n`).
* `SampledKSLiveV3`: `SampledKSLiveV2` with its event `LiveAcceptsKS ∧ RunPopOk` in place of `LiveAcceptsZK` and
  without the three hypotheses. `sampledKSLiveV3_of` derives it from `SampledKSLiveV2`: files that pass give V2's
  hypotheses, and at files that fail the event is empty.
-/

namespace Flock.Bridge

open Properties.NetworkCertifier.Bytes (bs_inj)

/-- **No colliding pair in the verifier's roots fixes each root's domain**: the roots `Flock.PortRoots.collides`
computes, each port's at its schema, are `rootsOf`'s. -/
theorem fixes_of_collides_false {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ}
    (h : Flock.PortRoots.collides c pub nl = false) : PortRoots.RootFixesDomain c pub nl := by
  intro q q' hlt hq he
  rw [PortRoots.rootsOf_getD c pub nl (hlt.trans hq), PortRoots.rootsOf_getD c pub nl hq] at he
  exact Flock.PortRoots.domOf_eq_of_collides_false h hlt hq (bs_inj he)

/-- **The verifier's roots check is V2's port-roots guard.** -/
theorem rootsColl_none_of_collides_false {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ}
    (h : Flock.PortRoots.collides c pub nl = false) : PortRoots.rootsColl c pub nl = none :=
  PortRoots.rootsColl_none_iff.2 (fixes_of_collides_false h)

/-- **The run's population check is `PopMatches`.** -/
theorem popMatches_of_popOk {desc : Claims.Msg} {qr : Lean.Json} {n : ℕ} {p : SampledProofs.Program}
    (hp : SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p) (h : SampledProofs.Slots.popOk p qr n = true) :
    PopMatches desc qr n := by
  show popCheck desc qr n = true
  unfold popCheck
  rw [hp]
  exact h

open FlockSoundness.Discharge.ZkExec in
/-- **`verify --zk` accepts, files and transcript**: the live audit's transcript (`LiveAcceptsZK`), and the files' rows,
names and roots (`Flock.PortRoots.check`, the last refusal of `verify --zk`'s verdict). -/
def LiveAcceptsKS {n : ℕ} {Reg : Type} (I : FlockSoundness.Discharge.Exec.Inputs)
    (o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) : Prop :=
  LiveAcceptsZK I o ∧ Flock.PortRoots.check (zkTags I) I.circuitFile I.publicFile I.tables = .ok ()

/-- **The sampled-proofs run passes the registration's population** (`SampledProofs.Run.run`, `population-order`):
the descriptor `desc` parses and its population under the query `qr` is `n` units in order (`Slots.popOk`). -/
def RunPopOk (desc : Claims.Msg) (qr : Lean.Json) (n : ℕ) : Prop :=
  ∃ p, SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p ∧ SampledProofs.Slots.popOk p qr n = true

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV2` with its file preconditions read off acceptance**: `SampledKSLiveV2`'s bound on the event that
`verify --zk` accepts the transcript and the files (`Bridge.LiveAcceptsKS`: their ports row/v1, row/v2 or row-seg/v1 rows
of distinct names, and no two registered roots, each framed at its port's schema, one root under different domains, at
the public file's population), the
sampled-proofs run passes the registration's population (`Bridge.RunPopOk`: the descriptor `desc` and the query `qr`
give the game's `n` units in order), and the claims relation fails. It has none of V2's hypotheses `RowsOk`,
`dupName = none`, `rootsColl = none` and `PopMatches`; the game's `n` is the public file's population (`pub.n = n`).
Its knowledge finders' budgets are V2's, `2(t' + v)` and `t' + v`. `SampledKSLiveV2` implies it (`sampledKSLiveV3_of`). -/
def SampledKSLiveV3 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
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
    -- one opening's verification, in SHA-512 evaluations
    (v : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
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
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs, at their cost: two runs of the
    -- prover and two verifications for a rewinding finder, one of each for the self-clash finder
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    -- ecr/sha-512 for the prover's link finders
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      k Rw M t' →
    prob (fun o => (Bridge.LiveAcceptsKS I o ∧ Bridge.RunPopOk desc qr n) ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgStrictZ (planZHJ dj y₀) L k (2 * (t' + v)) (t' + v) (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

/-- **`SampledKSLiveV2` gives `SampledKSLiveV3`**: at files that pass the verifiers' checks they give V2's hypotheses
(`Bridge.rowsOk_of_scopeOkH`, `Flock.PortRoots.check_ok`, `Bridge.rootsColl_none_of_collides_false` at `pub.n = n`,
`Bridge.popMatches_of_popOk`)
and V3's event is inside V2's; at files that fail, V3's event is empty. -/
theorem sampledKSLiveV3_of (h : SampledKSLiveV2) : SampledKSLiveV3 := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hn desc qr kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags
    hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ v q L P vb σ Xr δ Cc rows roots Cd κ hT hL
  by_cases hc : Flock.PortRoots.check (zkTags I) I.circuitFile I.publicFile I.tables = .ok () ∧
      Bridge.RunPopOk desc qr n
  · obtain ⟨hck, p, hp, hpo⟩ := hc
    obtain ⟨_, hdup, hcoll⟩ := Flock.PortRoots.check_ok hck (Flock.Desc.parse_flat hD) hpub
    have hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none := by
      rw [← hn]
      exact Bridge.rootsColl_none_of_collides_false hcoll
    exact (prob_mono (fun o ho => ⟨ho.1.1.1, ho.2⟩) _ τ).trans
      (h n Reg I ht D hD hU pub hpub hpt hscope (Bridge.rowsOk_of_scopeOkH hscope) hdup hroots desc qr
        (Bridge.popMatches_of_popOk hp hpo) kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM
        ρ hρ hr t' ht' K₀ hK₀ τ v q hT hL)
  · exact (prob_mono (q := fun _ => False) (fun o ho => hc ⟨ho.1.1.2, ho.1.2⟩) _ τ).trans
      ((prob_false _ τ).le.trans zero_le)

end Flock.Guarantees
