import Proofs.Flock.Bridge.FaultyRow
import Proofs.Flock.Bridge.ToyPubGates
import Proofs.Flock.Bridge.ToyPubReads
import Proofs.Flock.Bridge.Assumptions.ToyFilesParse
import Proofs.Flock.Bridge.Assumptions.ToySetupPub
import Proofs.Flock.Recursive.Flock

/-!
# `SampledKSLivePub`'s nontrivial lemma, at the toy session's files

`sampledKSLivePub_nontrivial`: `SampledKSLivePubAll`, `SampledKSLivePub` at the verifier that accepts everything, is
false. The instance is the toy `verify --zk` session of `Flock.Test.ToyPubFiles` (one unit of `toy-copy`, whose
returned gate is a row on input bit 0 of port `x`: `Bridge.ToyPub.op_gOut`), drawn by the subset draw of size 1, at the
deployed parameters, with a lying registrant (`Bridge.ToyPub.lieReads`) that opens port `x`'s row to zeros and port
`y`'s to ones. Every transcript's record carries the toy's draw, so every read is one of the verifier's own registered
reads (`OwnReadsHJ`, `Bridge.ToyPub.ownReads_lie`), and the claims' values are the registrant's: the input is `0`, and
the returned gate, which opens under the commitment only at its row of `y`, is `1`. A row of one input `0` returns `0`
(`Bridge.not_relS_row`), so the claim fails at every run, with probability `1 > δ + f(p)`.

Two premises stay, each interim: that the toy's files parse as `Flock.Test.ToyPubFiles` writes
(`Bridge.Assumptions.ToyFilesParse`, which the verifier's own functions check on the files), and the toy's setup
(`Assumptions.ToySetupPub`: the reference setup, transcripts carrying the draw under the live verifier's custody, and
a prover whose finder rarely collides).
-/

namespace Flock.Bridge.ToyPub

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Composed
open Flock.Test.ToyPubFiles (circuit pub root)

theorem toy_dupName : Flock.Registered.dupName circuit = none := by decide +kernel

theorem toy_names : ∀ q : Fin circuit.ports.size,
    (List.range circuit.ports.size).find? (fun j => circuit.ports[j]!.name == circuit.ports[q].name) = some q.val := by
  decide +kernel

/-- **The toy's two ports have different registered roots**, so no pair of them collides. -/
theorem toy_rootsColl (h : (root 0).data ≠ (root 1).data) : PortRoots.rootsColl circuit pub 1 = none := by
  have hp : circuit.ports.size = 2 := rfl
  have e : ∀ q, q < 2 → (rootsOf circuit pub 1).getD q [] = Properties.NetworkCertifier.Bytes.bs (root q) := by
    intro q hq
    rw [PortRoots.rootsOf_getD circuit pub 1 (by rw [hp]; exact hq)]
    rfl
  have hne : (rootsOf circuit pub 1).getD 0 [] ≠ (rootsOf circuit pub 1).getD 1 [] := by
    rw [e 0 (by decide), e 1 (by decide)]
    exact fun he => h (congrArg ByteArray.data (Properties.NetworkCertifier.Bytes.bs_inj he))
  have hpairs : PortRoots.pairs circuit.ports.size = [(0, 1)] := by rw [hp]; rfl
  unfold PortRoots.rootsColl PortRoots.collPair
  rw [hpairs]
  simp
  exact fun he => absurd he (by simpa using hne)

/-- **Probability 1 at every outcome**: an event whose condition fails at every outcome of `liveOfZJ`'s live audit
holds with probability 1, so it exceeds any `x < 1`. -/
theorem exceeds_of_outcomes {n : ℕ} {L : Law n} {Reg : Type} {mPts : ℕ} {dg : ByteArray → ByteArray} {N : ℕ}
    {planZ : Finset (Fin n) → Reg → List (ZkSession.TabZK FlockLevel3.GF128 mPts)}
    {wr : (S : Finset (Fin n)) → (R : Reg) → Tables.LiveOutJ (planZ S R).length mPts → Exec.Transcript}
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) {R : Reg × Finset (Fin n) × Exec.Transcript → Prop}
    (hR : ∀ ω o', ¬ R (τ.1, L.draw ω, wr (L.draw ω) τ.1 o')) {x : ℝ} (hx : x < 1) :
    ENNReal.ofReal x < prob (fun o => True ∧ ¬ R o) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ := by
  have h1 : prob (fun o => True ∧ ¬ R o) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ = 1 :=
    le_antisymm (prob_le_one _ _ _) ((prob_true _ _).symm.le.trans
      (Recursion.prob_auditLive_liveOfZJ_mono dg N wr τ fun ω o' _ => ⟨trivial, hR ω o'⟩))
  rw [h1]
  exact ENNReal.ofReal_lt_one.2 hx

end Flock.Bridge.ToyPub

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open Flock.Test.ToyPubFiles (circuit pub progDesc progQuery)
open Flock.Assumptions (toyPubI toyPubI_typed toyPubI_tags toyPubI_div ToyΩ toyLb toySrc toySrc_uniform)
open Flock.Bridge.ToyPub (gOut xIn)

attribute [local irreducible] Flock.Test.ToyPubFiles.circuit in
open Classical in
/-- **`SampledKSLivePub`'s nontrivial lemma**: `SampledKSLivePub` at the verifier that accepts everything is false, at
the toy session's files (`Bridge.Assumptions.ToyFilesParse`, interim) and its setup (`Assumptions.ToySetupPub`,
interim). There the registrant opens port `x`'s row to zeros and port `y`'s to ones, every read is the verifier's own,
and the toy's returned gate, a row on input bit 0 of `x`, is `1` on an input `0`: the claim fails at every run, so
`Pr[accept ∧ ¬ RelS] = 1`, while `δ + f(J²·K·Pr[B^τ]) < 1` (`Bridge.deltaPubR_lt_one`). -/
theorem sampledKSLivePub_nontrivial (hf : Bridge.Assumptions.ToyFilesParse) (hs : Assumptions.ToySetupPub) :
    ¬ SampledKSLivePubAll := by
  intro hAll
  obtain ⟨hD0, hsc, hpub0, hroot, ⟨Dd, hDd, hDp⟩, ⟨dp, hdp, hidx⟩, ⟨pp, hpp, hpop⟩⟩ := hf
  have hD : Flock.Desc.parse (zkTags toyPubI) toyPubI.circuitFile toyPubI.tables = .ok (.single circuit) := hD0
  have hpub : HmOut.loadPublic (zkTags toyPubI) circuit toyPubI.publicFile = .ok pub := hpub0
  have hU : Built circuit.unit := FlockC3.VStarList.built_of_parse
    (FlockSoundness.Discharge.Hidden.parse_untyped ((zkTags_typed _).trans toyPubI_typed) (Flock.Desc.parse_flat hD)).1
  have hscope : Discharge.Layout.scopeOkH circuit = true := by
    rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
    exact hsc
  have hpt : pub.tables = none := rfl
  obtain ⟨dj, S₀, d₀, y₀, wr, hrec, hCust, dg, N, τ, t', v, qR, hB⟩ := hs hD hU hpub hpt hscope
  have hpm : Bridge.PopMatches progDesc.data.toList progQuery 1 := by
    unfold Bridge.PopMatches Bridge.popCheck
    rw [hpp]
    exact hpop
  obtain ⟨hk, hRw, hM, hρ, hρk⟩ := sampledKSLivePub_honest
  let hc := Flock.Desc.parse_flat hD
  let rd := Bridge.ToyPub.lieReads (Reg := Unit) dj y₀ toyPubI_typed hc hscope hU hpub
  have h := hAll 1 Unit (Bridge.ToyPub.lieOwn (Reg := Unit) dj y₀ toyPubI_typed hc hscope hU hpub) toyPubI
    toyPubI_typed (.single circuit) hD hU pub hpub hpt hscope (Bridge.rowsOk_of_scopeOkH hscope)
    Bridge.ToyPub.toy_dupName (Bridge.ToyPub.toy_rootsColl hroot) progDesc.data.toList progQuery hpm 1 le_rfl
    toyPubI_div dj S₀ d₀ y₀ rd wr dg N ToyΩ toyLb toySrc toySrc_uniform toyPubI_tags hCust Bridge.kDep (2 ^ 192)
    (2 ^ 192) hk hRw hM (1 / 2) hρ hρk t' 1 le_rfl τ v qR ⟨"", 0, []⟩
  have hlt := Bridge.deltaPubR_lt_one (n := 1) (kd := 1) (K₀ := 1) one_pos le_rfl (by norm_num)
    (Bridge.ToyPub.ioSum_le hU) (mul_nonneg (mul_nonneg (sq_nonneg _) (Nat.cast_nonneg _)) (Claims.Pr_nonneg _)) hB
  have hown : ∀ t : Discharge.Exec.Transcript, recordDraw t.record = some Flock.Test.ToyPubFiles.draw →
      Discharge.ZkReg.OwnReadsHJ toyPubI (Bridge.ToyPub.lieOwn (Reg := Unit) dj y₀ toyPubI_typed hc hscope hU hpub)
        circuit pub t rd.isRead rd.port rd.readAt := fun t ht =>
    Bridge.ToyPub.ownReads_lie dj y₀ toyPubI_typed hc hscope hU hpub le_rfl le_rfl ht hDd hDp hdp hidx
      Bridge.ToyPub.toy_names
  obtain ⟨ux, kx, hx⟩ := Bridge.ToyPub.rowOf_xIn hU
  refine absurd h (not_le.2 (Bridge.ToyPub.exceeds_of_outcomes τ (fun ω o' => ?_) hlt))
  have ho := hown _ (hrec ((Law.execOS (n := 1) toySrc (.subset 1) 1).draw ω) τ.1 o')
  refine Bridge.not_relS_row _ 1 rfl _ _ _ _ _ _ (by positivity) (Bridge.ToyPub.gOut_mem hU)
    (Bridge.ToyPub.unit_gOut hU) (Bridge.ToyPub.op_gOut hU) (Bridge.ToyPub.op_xIn hU) (Bridge.ToyPub.consts_xIn hU)
    ?_ ?_
  · beta_reduce
    split
    · exact (Bridge.ToyPub.bit_lie hx []).trans rfl
    · rename_i hn
      exact (hn ⟨Or.inr ho, ux, _, Bridge.vbOf_pos_row hx⟩).elim
  · intro d hd
    have hr := Bridge.commitDep_rows_of_opens _ _ _ _ _ _ _ _ hd
    obtain ⟨⟨ug, qg, kg⟩, hrow⟩ := Option.ne_none_iff_exists'.1 fun h => hr (Option.map_eq_none_iff.2 h)
    have hq := Bridge.ToyPub.rowOf_gOut hU hrow
    have hrow' : FlockSoundness.Discharge.HiddenExec.rowOfH circuit hU 1
        (FlockSoundness.Discharge.Layout.outsOf circuit hU 1) (gOut hU) = some (ug, ⟨qg, kg⟩) := hrow
    beta_reduce
    split
    · exact (Bridge.ToyPub.bit_lie hrow' []).trans (decide_eq_true hq)
    · rename_i hn
      exact (hn ⟨Or.inr ho, ug, qg, Bridge.vbOf_pos_row hrow⟩).elim

end Flock.Guarantees
