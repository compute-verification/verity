import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictStatements
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLigMain
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysDet
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams

/-!
# `ZkRepAccepts`'s Ligerito half: a `--zk` rep's Ligerito verdict accepts the proof's rows

`ZkRepAccepts` (`VerdictStatements.lean`) has three conclusions at every result of the compiled rep: the Ligerito
verdict accepts the opened rows, the inner check holds, and every requested row verifies (`OpensOKZ`). This file is the
first: `ZkLigAccepts` takes exactly `ZkRepAccepts`'s hypotheses and concludes `pend.lig.verdict os.rows = true`.

The compiled masked phase's Ligerito (`ZkLink.ligeritoPadZC`, then `ZkLink.ligTailC`) is the non-ZK session's compiled
Ligerito (`Model.recursiveRoundsC`, `Model.finalCheck`) at levels ≥ 1; at level 0 it batches the padded column
`ZkLink.padColumnO`: the witness lanes folded, `ρ₁·row[2^k₀+2r] + ρ₂·row[2^k₀+2r+1]`, and the pads' correction
`Model.padCorr` at `ȳ`, which is the executable's `Zk.padAt` at `PAD_C`.

`zk_ligerito_accepts` (`VerdictLigMain.lean`) is that refinement for one run of `Zk.ligerito`. Here `zk_rep_lig` carries
it through the rep as `zk_rep_plays` builds the rep's run, at the opening's batched values (`OpenRel`: the executable's
`basis` is the model's `bHat` at the opening's `b`), and a run has one result (`plays_det`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **A `--zk` rep's Ligerito verdict accepts** (live PROTOCOL.md §7, PROTOCOL.md §13). Under `ZkRepAccepts`'s
hypotheses, every result of the compiled rep, played with the rep's messages and the rounds' coins, has a Ligerito
verdict that accepts the openings whose rep-`r` rows are the proof's. -/
def ZkLigAccepts : Prop :=
  ∀ {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript) (cap₀ : Cap Dig)
    (os : ZkLink.OpensZ 𝔽 Dig (List UInt8)),
    Zk.schedule st.m = .ok (sch, tPad) → st.merkle = MerkleScheme.hm96Sha512 → ZkTabSeam T st sch tPad →
    ZkTabLay T st sch → FoldRealizes st.fold T.S →
    extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length →
    (∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j]) →
    Zk.decode (shapeZK st sch extras) bytes = .ok zp → RoundsWF rounds →
    (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size → cap₀ = capDig zp.proof.cap →
    (∀ ℓ j, os.rows r.val ℓ j = openRow zp.proof.ligerito sch ℓ j) →
    (∀ ℓ j, os.salts r.val ℓ j = saltRow zp.proof.ligerito sch ℓ j) → os.inner r.val = innerOpensOf zp.inner →
    ∀ pend : ZkLink.PendZ 𝔽 Dig (List UInt8),
      Plays fOf (ZkLink.repZCAt execArith hs512 T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend →
      pend.lig.verdict os.rows = true

/-- The `--zk` schedule's level 0 has rate 1 and room for the pads: `t_pad ≤ 2^cols₀`. The kernel checks every `m < 40`;
`zk_schedule_seams` bounds `m` by 35. -/
theorem zk_schedule_pad (m : ℕ) (sch : Flock.Schedule) (tPad : ℕ) (h : Zk.schedule m = .ok (sch, tPad)) :
    (sch.levels.getD 0 default).rate = 1 ∧ tPad ≤ 2 ^ (sch.levels.getD 0 default).cols := by
  obtain ⟨-, h35, -⟩ := zk_schedule_seams _ _ _ h
  have key : ∀ m : Fin 40, ∀ x ∈ (Zk.schedule m.val).toOption,
      decide ((x.1.levels.getD 0 default).rate = 1 ∧ x.2 ≤ 2 ^ (x.1.levels.getD 0 default).cols) = true := by
    decide +kernel
  exact of_decide_eq_true (key ⟨m, by omega⟩ (sch, tPad) (by rw [h]; rfl))

/-- **`repProg`'s run accepts the proof's rows** (`zk_rep_plays` with its Ligerito verdict): the run of the compiled rep
that `zk_rep_plays` builds has a Ligerito verdict that accepts openings whose rep-`r` rows are the proof's. -/
theorem zk_rep_lig {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (extras : Array Flock.Claim) (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (o : Opens 𝔽 Dig) (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam T st sch tPad)
    (hlay : ZkTabLay T st sch)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) (ho : ∀ ℓ j, o r.val ℓ j = openRow zp.proof.ligerito sch ℓ j) :
    ∃ pend : ZkLink.PendZ 𝔽 Dig Sl,
      Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend ∧
      pend.lig.verdict o = true := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, hα1, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨hrate, hpad⟩ := zk_schedule_pad _ _ _ hsch
  have hTm : T.S.m = st.m := hseam.hm
  have h7 : 7 ≤ T.S.m := by omega
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨Tb, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok lg =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg', t₄, h₄, h₅⟩ := bind_ok hrun
  clear hrun
  obtain ⟨hlg, ht₄⟩ := pure_ok h₄
  rw [hlg, ht₄] at h₅
  have hwf₀ : RoundsWF (Flock.Transcript.init rounds dom).rounds := hwf
  have hcs3 := replay_cons_size _ _ _ _ h₁
  obtain ⟨hr₁, hk₁, hRP⟩ := zk_replay_plays st _ zp.proof zp.inner.root _ t₁ rp T.S hTm.symm hseam.hk.symm
    (by omega) hwf₀ h₁
  obtain ⟨k₁, zc, row, abM, row', hk₁a, hk₁b, hpz, hpl, hc0, hc1⟩ := hRP T.padsA T.tr
  rw [← hTm] at h₂
  obtain ⟨g₀, g₁, g₂⟩ := two_append_get! ({ rp.claims[0]! with value := Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]! } : Flock.Claim)
    ({ rp.claims[1]! with value := Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]! } : Flock.Claim) extras
  obtain ⟨hr₂, hk₂, op, hop, hrel⟩ := opening_refines T.S.m h7 _
    (⟨fOf (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!), abM.skip, abM.x⟩ ::
      ⟨fOf (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!), execArith.lagS zc.z, zc.r⟩ ::
        Model.extraClaims T.S T.ptLocal mPts pts)
    (by rw [size_two_append]; simp; omega)
    (fun k hk => by
      rcases k with _ | _ | k
      · rw [g₀]; exact hc0 _
      · rw [g₁]; exact hc1 _
      · have hk' : k < (Model.extraClaims T.S T.ptLocal mPts pts).length := by simp at hk; omega
        rw [g₂ k (by omega)]
        exact hext k hk')
    (by rw [size_two_append]; omega) (by rw [size_two_append]; omega) _ t₁ t₂ Tb sws (hr₁ ▸ hwf₀) h₂
  rw [size_two_append] at hop
  obtain ⟨xo, hxo, hxo1⟩ := openingZK_plays h7 _ _ abM.pub ⟨0, execArith.lagS zc.z, zc.r⟩ _ hop
  obtain ⟨xo1, xo2⟩ := xo
  obtain rfl : xo1 = some op := hxo1
  have hwf₂ : RoundsWF t₂.rounds := hr₂ ▸ hr₁ ▸ hwf₀
  obtain ⟨hr₃, hk₃, -⟩ := zk_ligerito_plays st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩
    t₂ t₃ rc hdim hlv hα1 hk0 hwf₂ h₃
  have hlig : ∀ {e : ℕ} (P : ZkSession.Pads0 T.np e), e = 2 → ∀ (Tm : 𝔽),
      ∃ out : ZkSession.LinK 𝕂 T.np × PendL 𝔽 Dig,
        Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P T.sch T.t T.c₀ r.val Tm op.b)
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0) (consumed t₂.rounds t₂.k t₃.k) out ∧
        out.2.verdict o = true := by
    intro e P he Tm
    subst he
    rw [hseam.hsch, hseam.ht, hlay.hc₀]
    exact zk_ligerito_accepts st.merkle sch Tb zp.proof.cap zp.proof.ligerito (Flock.basis sws)
      ⟨tPad, r.val, zp.level0⟩ op.b o hdim hlv hα1 hk0 hrate hpad (fun ℓ _ j _ => congrArg Prod.fst (ho ℓ j))
      (fun orig y hqe hy => by
        have hb := basis_eq hrel orig sch.ext (by rw [hTm, ← hsm]; exact hqe) y hy
        rwa [hTm, ← hsm] at hb)
      hwf₂ h₃ P Tm
  have hwf₃ : RoundsWF t₃.rounds := hr₃ ▸ hwf₂
  obtain ⟨hr₅, hk₅, hIP⟩ := zk_inner_plays _ zp.inner zp.proof.zerocheck.finalC _ t₃ tr
    (by rw [Array.size_append, Array.size_append, hcs3, ligeritoConstraints_size hv]; rfl) hwf₃ h₅
  have hM : ∃ x, Plays fOf (ZkLink.repMaskedZC (D := Dig) execArith T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r.val
        (Model.extraClaims T.S T.ptLocal mPts pts))
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC]))))
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) x ∧
      x.2.verdict o = true := by
    unfold ZkLink.repMaskedZC
    refine pq_map ?_
    refine pq_bind hpz ?_ rfl rfl
    refine pq_map ?_
    refine pq_bind hpl ?_ rfl rfl
    refine pq_map ?_
    unfold ZkLink.restZC
    refine pq_bind hxo ?_ rfl rfl
    obtain ⟨out, hout, hacc⟩ := hlig T.pads hseam.he op.T
    refine pq_bind hout ?_ rfl (List.append_nil _).symm
    exact ⟨_, Plays.bind' (Plays.recv _ (fOf zp.proof.zerocheck.finalC)) (Plays.pure _) rfl rfl, hacc⟩
  have hmsgs : restMsgsZ st sch (2 + extras.size) tPad zp =
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC])))) ++
      [⟨𝔽 × 𝔽, (fOf zp.inner.rho, fOf zp.inner.sigma)⟩, ⟨Dig, toDig zp.inner.tauCommit⟩,
        ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) zp.inner.y⟩] := by
    rw [hTm, hseam.hk, hseam.hnp, hseam.hnP]
    simp [restMsgsZ, innerMsgs]
  have hcoins : rounds.toList.map (·.coins) =
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) ++ consumed t₃.rounds t₃.k tr.k := by
    rw [hr₃, hr₂, hr₁]
    simp only [List.append_assoc]
    rw [← consumed_append _ hk₃ hk₅, ← consumed_append _ hk₂ (hk₃.trans hk₅),
      ← consumed_append _ hk₁b (hk₂.trans (hk₃.trans hk₅)),
      ← consumed_append _ hk₁a (hk₁b.trans (hk₂.trans (hk₃.trans hk₅))), hk]
    exact (consumed_all _).symm
  unfold ZkLink.repZCAt ZkLink.repZC
  rw [hseam.hq]
  refine exQ_recv (toDig zp.inner.root) ?_
  refine pq_bindQ hM (fun x hx => ?_) hmsgs hcoins
  obtain ⟨v, hv⟩ := hIP Hs T.dom T.np T.nP T.pos x.1 T.tr (toDig zp.inner.root)
  exact ⟨⟨x.2, v⟩, Plays.bind' hv (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm, hx⟩

theorem zk_lig_accepts : ZkLigAccepts := by
  intro mPts T pts r st sch tPad extras bytes zp rounds dom tr cap₀ os hsch _ hseam hlay _ hlen hext _ hwf hrun hk _
    ho _ _ pend hp
  obtain ⟨pend', hp', hacc⟩ := zk_rep_lig hs512 T pts r st sch tPad extras zp rounds dom tr os.rows hsch hseam hlay
    hlen hext hwf hrun hk ho
  obtain rfl := plays_det hp hp'
  exact hacc

end FlockSoundness.Discharge.ZkExec
