module

public import Proofs.Flock.Bridge.Refused
public import Flock.KsScope
import Proofs.Flock.Recursive.Defs
import Proofs.Flock.Soundness.ExecCheck
import Proofs.Flock.Soundness.ExecCircuit
import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.Hidden.Facts

/-!
# The sampled audit's knowledge soundness on `verify --zk`'s acceptance (`SampledKSLiveV3`)

`Bridge.SampledKSLiveRoots` takes five things of the statement, the circuit file and the draw as hypotheses beyond the
parses and the public file's form (`pub.tables = none`, the form without shared rows): an untyped statement, the unit a
built net (`Built`), the circuit in `Layout.scopeOkH`, the draw's `k` a multiple of the tables, and the `--zk` tags'
hidden outputs, retained rounds and `hm96-sha512/v1` leaves (`TagsOkZHJ`). `verify --zk` refuses every session outside
them (`Flock.KsScope.check`, after `Flock.PortRoots.check` in its verdict), so the guarantee reads them off acceptance.

* `Bridge.KsScoped I D kd`: the five.
* `Bridge.LiveAcceptsScoped I o`: `verify --zk` accepts: `Bridge.LiveAcceptsKS`, and the files and the record's draw
  pass `Flock.KsScope.check`.
* `Bridge.ksScoped_files`, `Bridge.div_of_drawWhy`: the check gives four of the five from the files (the unit is built
  because `HmRow.parse` built it from rows, `unit_built`), and the fifth, the draw's `k` a multiple of the tables,
  from the record's draw, which custody (`RecordCustodyZKJ`) makes the verifier's.
* `SampledKSLiveV3`: at a statement, files and a draw in `KsScoped`, `SampledKSLiveRoots`' bound on
  `LiveAcceptsScoped`'s event; outside them, `verify --zk` accepts with probability 0. It takes none of the five as a
  hypothesis. `sampledKSLiveV3_of` derives it from `SampledKSLiveV2`, through `Bridge.sampledKSLiveRoots_of`.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Composed
  FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)

/-- **`SampledKSLiveRoots`' statement, file and draw preconditions**: an untyped statement, the unit a built net, the
circuit in `Layout.scopeOkH`, the draw's `k` a multiple of the tables (`nTab`), and the `--zk` tags `TagsOkZHJ`. -/
structure KsScoped (I : Inputs) (D : Flock.Desc) (kd : ℕ) : Prop where
  typed : I.tags.typed = false
  built : Built D.unroll.unit
  scope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true
  div : kd % nTab I = 0
  tags : TagsOkZHJ I

/-- **`verify --zk` accepts, files, draw and transcript**: `LiveAcceptsKS`, and the files and the record's draw pass
`Flock.KsScope.check` (the last refusal of `verify --zk`'s verdict). -/
def LiveAcceptsScoped {n : ℕ} {Reg : Type} (I : Inputs)
    (o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) : Prop :=
  LiveAcceptsKS I o ∧
    Flock.KsScope.check (zkTags I) I.circuitFile I.publicFile I.tables I.count
      (FlockSoundness.Discharge.Exec.recordDraw o.2.2.record) = .ok ()

/-- **The files that pass `Flock.KsScope.check` are in four of `KsScoped`'s five**, and the draw passes `drawWhy`. The
unit is built because `HmRow.parse` built it (`unit_built`, at the untyped tags `tagsWhy` leaves). -/
theorem ksScoped_files {I : Inputs} {D : Flock.Desc} {pub : Flock.Public} {draw : Option Lean.Json}
    (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (hpub : Flock.HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    (h : Flock.KsScope.check (zkTags I) I.circuitFile I.publicFile I.tables I.count draw = .ok ()) :
    I.tags.typed = false ∧ Built D.unroll.unit ∧ FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true ∧
      TagsOkZHJ I ∧ Flock.KsScope.drawWhy I.count draw = none := by
  obtain ⟨ht, hs, -, hd⟩ := Flock.KsScope.check_ok h (Flock.Desc.parse_flat hD) hpub
  obtain ⟨ht1, ht2, ht3, ht4⟩ := Flock.KsScope.tagsWhy_none ht
  have P := parse_facts (FlockSoundness.Discharge.Hidden.parse_untyped ht1 (Flock.Desc.parse_flat hD)).1
  obtain ⟨ls, hls⟩ := P.check
  exact ⟨(zkTags_typed I).symm.trans ht1, unit_built P (check_facts P.typed hls),
    (FlockSoundness.Discharge.FailClosed.scopeOkH_eq _).symm.trans hs,
    ⟨(zkTags_hiddenOutputs I).symm.trans ht2, (zkTags_retainRounds I).symm.trans ht3,
      (zkTags_merkleLeaf I).symm.trans ht4⟩, hd⟩

/-- **The record's draw passing `drawWhy` gives the draw's `k` a multiple of the tables**: custody makes an accepted
record's draw `dj S`, and a `dj S` that parses the verifier's own subset draw of `kd` units. -/
theorem div_of_drawWhy {n mPts : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json}
    {Jf : Finset (Fin n) → Reg → ℕ}
    {wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → FlockSoundness.Discharge.Exec.Transcript}
    {kd : ℕ} (hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJ I dj Jf wr kd) {S : Finset (Fin n)} {R : Reg}
    {lo : LiveOutJ (Jf S R) mPts} (hacc : AcceptsZK I (wr S R lo))
    (hd : Flock.KsScope.drawWhy I.count (FlockSoundness.Discharge.Exec.recordDraw (wr S R lo).record) = none) :
    kd % nTab I = 0 := by
  unfold nTab
  split
  · exact Nat.mod_one kd
  · next hc =>
    obtain ⟨j, d, hj, hjd, hk⟩ := Flock.KsScope.drawWhy_none hd hc
    rw [(hCust.1 S R lo hacc).1] at hj
    obtain rfl := Option.some.inj hj
    have hdS := hCust.2 S d hjd
    exact hk kd (by rw [hdS]; rfl)

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV2` with its statement, file and draw preconditions read off acceptance**: on the event that
`verify --zk` accepts the transcript, the files and the record's draw (`Bridge.LiveAcceptsScoped`:
`Bridge.LiveAcceptsKS`, whose ports are row/v1, row/v2 or row-seg/v1 rows of distinct published domains with no two registered roots
one root under different domains, and `Flock.KsScope.check`), SPOT's run passes the registration's
population (`Bridge.RunPopOk`),
and the claims relation fails, at a statement, files and a draw in `Bridge.KsScoped` the bound is
`Bridge.SampledKSLiveRoots`', and outside them the event has probability 0. It has none of V2's hypotheses `RowsOk`,
`dupId = false`, `rootsColl = none` and `PopMatches`, nor `I.tags.typed = false`, `Built D.unroll.unit`,
`scopeOkH D.unroll = true`, `kd % nTab I = 0` and `TagsOkZHJ I`; the game's `n` is the public file's population
(`pub.n = n`), and the public file has no shared rows (`pub.tables = none`), whose shared-row form is
`SampledKSLiveV5`'s. Its finders' charges are V2's: `2(t' + v)` and `t' + v` for the knowledge finders and `t' + v` per
run for the link finders. `SampledKSLiveV2` implies it (`sampledKSLiveV3_of`). -/
def SampledKSLiveV3 : Prop :=
  ∀ (n : ℕ) (Reg : Type) (I : Inputs)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    -- the game's units are the public file's population
    (_hn : pub.n = n)
    -- the statement's descriptor, and the registration's query
    (desc : Claims.Msg) (qr : Lean.Json)
    (kd : ℕ) (_hkd : kd ≤ n)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- live-verifier
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- one opening's verification, in SHA-512 evaluations
    (v : ℝ) (_hv : 0 ≤ v)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    if hs : Bridge.KsScoped I D kd then
      let L := Law.execOS (n := n) src (.subset kd) n
      let P := (keyProg D.unroll hs.built n).partition (Discharge.Layout.outsOf D.unroll hs.built n)
      let vb := (rsZHJ dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hpt).bindingZ
        (rsZHJ_computes dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hpt)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
        (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (Flock.Desc.parse_flat hD)
            (Discharge.Layout.scopeOkH_spec hs.built hs.scope).2.1 dj y₀ wr _ hs.tags hCust.1
            fun _ => drawOkZKHJ_of_file I hs.typed (Flock.Desc.parse_flat hD) kd n hs.div dj hCust.2 _)) τ
      let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
      let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
      -- the deployed commitment's layout: the game's rows, read off the circuit file
      let Cc := (keyProg D.unroll hs.built n).circuit (Discharge.Layout.outsOf D.unroll hs.built n)
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hs.built n
          (Discharge.Layout.outsOf D.unroll hs.built n) g).map fun r =>
          ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      -- the registration's frame-v3 roots, over the public file's rows
      let roots := Bridge.rootsOf D.unroll pub n
      let Cd := Bridge.commitDep Cc rows D.unroll pub n roots
      -- the circuit's constants, public
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hs.built n g
      -- cr/sha-512 for the knowledge finders of each table the prover's session runs, at their cost: two runs of the
      -- prover and two verifications for a rewinding finder, one of each for the self-clash finder
      (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
      -- ecr/sha-512 for the prover's link finders, at a run of the prover and a verification per run
      FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
        k Rw M (t' + v) →
      prob (fun o => (Bridge.LiveAcceptsScoped I o ∧ Bridge.RunPopOk desc qr n) ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ (.subset kd) desc q roots)
            (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
          (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
        ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
          ksAvgStrictZ (planZHJ dj y₀) L k (2 * (t' + v)) (t' + v) (reg σ) +
          ENNReal.ofReal ((∑ p : vb.Pos,
              (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
            (2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
    else
      prob (fun o => Bridge.LiveAcceptsScoped I o ∧ Bridge.RunPopOk desc qr n)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ = 0

/-- **`SampledKSLiveV2` gives `SampledKSLiveV3`**: at a statement, files and a draw in `KsScoped`, V3's event is inside
`Bridge.SampledKSLiveRoots`' at `KsScoped`'s five, and that follows from V2 (`Bridge.sampledKSLiveRoots_of`); outside
them no outcome the prover reaches is accepted, since an accepted outcome's files pass `Flock.KsScope.check`
(`Bridge.ksScoped_files`) and its record's draw is the verifier's (`Bridge.div_of_drawWhy`). -/
theorem sampledKSLiveV3_of (h : SampledKSLiveV2) : SampledKSLiveV3 := by
  have h := Bridge.sampledKSLiveRoots_of h
  intro n Reg I D hD pub hpub hpt hn desc qr kd hkd dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hCust k Rw M hk hRw hM ρ hρ hr
    t' ht' K₀ hK₀ τ v hv q
  split
  · next hs =>
    intro L P vb σ Xr δ Cc rows roots Cd κ hT hL
    exact (prob_mono (fun o ho => ⟨⟨ho.1.1.1, ho.1.2⟩, ho.2⟩) _ τ).trans
      (h n Reg I hs.typed D hD hs.built pub hpub hpt hs.scope hn desc qr kd hkd hs.div dj S₀ d₀ y₀ wr dg N Ω Lb
        src hA3 hs.tags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ v hv q hT hL)
  · next hs =>
    refine nonpos_iff_eq_zero.1 (((FlockSoundness.Game.prob_mono_outcome (q := fun _ => False) _ τ
      fun o ho he => hs ?_).trans (prob_false _ τ).le))
    unfold auditLive at ho
    obtain ⟨ω, hω⟩ := ho
    obtain ⟨t, ht, rfl⟩ := Game.Outcome.of_map _ _ _ hω
    unfold liveOfZJ at ht
    obtain ⟨lo, -, rfl⟩ := Game.Outcome.of_map _ _ _ ht
    obtain ⟨hty, hU, hsc, hTags, hd⟩ := Bridge.ksScoped_files hD hpub he.1.2
    exact ⟨hty, hU, hsc, Bridge.div_of_drawWhy hCust he.1.1.1 hd, hTags⟩

end Flock.Guarantees
