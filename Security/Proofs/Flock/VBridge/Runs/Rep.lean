import Proofs.Flock.VBridge.Runs.Prog
import Proofs.Flock.VBridge.Runs.Lg

/-!
# A rep's run from its algebra, Ligerito included

`run_of_algebra` closes `prog_of_algebra`'s Ligerito hypothesis with `hlg_of_residuals`: at E's preconditions, with the
proof file decoded and holding V*'s messages and openings, the rep's program accepts and ends at the record's last
round, as `TableFiles.Runs` asks, once the Ligerito rounds' bytes are the frames `hlg_of_residuals` names.
-/

namespace FlockVBridge.Runs

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Compiled
open FlockSoundness.Discharge.Exec (chunks ligFrames)

/-- **A rep's run from its algebra**: under `prog_of_algebra`'s hypotheses without its Ligerito run, with the proof
file `bytes` decoded at the setup's schedule, its Ligerito fields the messages w holds and each level's openings
`Opened`, there are frames `F` such that, when every Ligerito round of the record retains `F`'s bytes, the rep's program
accepts and ends at the record's last round. -/
theorem run_of_algebra (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d) (hacc : AccHolds sh d)
    (st : Setup) (sch : Flock.Schedule) (p : Proof) (bytes : ByteArray) (extras : Array Claim) (t : Transcript)
    (ht : t.k = 0) (hd : d.rounds = t.rounds.map (·.coins)) (hne : NoEmpty d.rounds) (hwf : RoundsWF t.rounds)
    (hm : st.m = sh.m) (hm14 : 14 ≤ st.m) (hkL : st.fold.kLog = sh.kLog)
    (hkS : st.kLog = sh.kLog) (hkm : st.kLog ≤ st.m) (hxs : d.extras = extras)
    (hext : ∀ j < extras.size, extras[j]!.skip.size = 64 ∧ extras[j]!.x.size = st.m - 6)
    (hsch : sh.sched = sch) (hf : Flock.fast100 st.m = some sch)
    -- the proof file: its decode, the messages w holds, the comb port and the levels' openings
    (hdec : decodeProof (shapeOf st sch extras.size) bytes = .ok p)
    (hzp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) p.zerocheck)
    (hlp : LcProofOf (decoded sh d.w d.comb d.cons) p.lincheck) (hpl : sh.holo = false)
    (hrsw : p.ringSwitches = (decoded sh d.w d.comb d.cons).rs)
    (hmessages : p.ligerito.messages =
      (decoded sh d.w d.comb d.cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2))
    (hoodValues : p.ligerito.oodValues = (decoded sh d.w d.comb d.cons).ood)
    (hyr : p.ligerito.finalProof.yr = (decoded sh d.w d.comb d.cons).yr)
    (hinitialCap : (p.ligerito.initialCap == p.cap) = true)
    (hfoldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
        .ok d.comb)
    (hopen : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ∀ cs ∈ info.lg.conss.toList, Opened st.merkle sch p.cap p.ligerito (decoded sh d.w d.comb d.cons) cs)
    -- the zerocheck's frames and counts
    (hn₀ : (t.rounds[t.k]!).n = 6) (hn₁ : (t.rounds[t.k + 1]!).n = st.m - 13) (hn₂ : (t.rounds[t.k + 2]!).n = 1)
    (hnr : ∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1)
    (ha₀ : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
      .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6]))
    (ha₁ : AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)]))
    (ha₂ : AnswersL (t.rounds[t.k + 2]!)
      (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1]))
    (har : ∀ i < st.m - 6, AnswersL (t.rounds[t.k + 3 + i]!)
      (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1]))
    -- the lincheck's frames and counts
    (hlc : LcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck)
    -- ring switching's frames and counts
    (hrn : ∀ k < 2 + extras.size, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).n = 7)
    (hra : ∀ k < 2 + extras.size, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).answers
      (rsFramed (rsPending ByteArray.empty k) p.ringSwitches[k]!))
    (hgn : (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).n = 2 + extras.size)
    (hga : (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).answers
      (ByteArray.empty ++ Transcript.header 4 2 (2 + extras.size))) :
    ∃ (F : ℕ → ByteArray) (P' : ByteArray),
      (∀ j, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size) + 1 ≤ j → j < t.rounds.size →
        (t.rounds[j]!).bytes = some (F j)) →
      repProg st sch p extras t = .ok ((), ⟨t.rounds, t.rounds.size, P'⟩) := by
  obtain ⟨v, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩ := halg
  obtain ⟨-, -, -, -, -, hT, hSws⟩ :=
    rs_of_residuals hrows hS hw hc (hacc v info kappa hrows) hres hzero hm14 hm hkS hkL hkm hxs hext hd hne ht
      (hzp v info kappa hrows).fc hlp.zp hrsw rfl rfl hrn hra hgn hga
  have hsz := claimsF_size sh (valOf sh d.w d.comb d.cons) info extras
  obtain ⟨F, P', hF⟩ := hlg_of_residuals (bytes := bytes) hrows hS hw hc (hacc v info kappa hrows) hres hzero hm14
    hm.symm hkL.symm hsch hf hdec hmessages hoodValues hyr hinitialCap (hopen v info kappa hrows)
    (claims := claimsF sh (valOf sh d.w d.comb d.cons) info extras) (rpp := fun k => (info.rs.sws[k]!).rpp)
    (gamma := info.rs.gamma) (by rw [hsz, hxs]) hT hSws hd hne hwf ht rfl rfl
  refine ⟨F, P', fun hfr => prog_of_algebra sh d ⟨v, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩ hacc st
    sch p extras t ht hd hne hm hm14 hkL hkS hkm hxs hext hzp hlp hpl hrsw hfoldPts hn₀ hn₁ hn₂ hnr ha₀ ha₁ ha₂ har
    hlc hrn
    hra hgn hga ?_⟩
  intro v' info' kappa' h'
  rw [hrows] at h'
  obtain ⟨rfl, rfl, rfl⟩ : v = v' ∧ info = info' ∧ kappa = kappa' := by simpa using h'
  have h := hF (by rw [hsz]; exact hfr)
  rw [hT, hSws, hsz] at h
  exact h

/-- **A rep's run from its algebra, at named frames** (`TableFiles.Runs`' body): `run_of_algebra` with every Ligerito
round of the record retaining its round of `ligFrames` (`hlg_of_frames`), so the rep's program accepts and ends at the
record's last round. -/
theorem run_of_frames (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d) (hacc : AccHolds sh d)
    (st : Setup) (sch : Flock.Schedule) (p : Proof) (bytes : ByteArray) (extras : Array Claim) (t : Transcript)
    (ht : t.k = 0) (hd : d.rounds = t.rounds.map (·.coins)) (hne : NoEmpty d.rounds) (hwf : RoundsWF t.rounds)
    (hm : st.m = sh.m) (hm14 : 14 ≤ st.m) (hkL : st.fold.kLog = sh.kLog)
    (hkS : st.kLog = sh.kLog) (hkm : st.kLog ≤ st.m) (hxs : d.extras = extras)
    (hext : ∀ j < extras.size, extras[j]!.skip.size = 64 ∧ extras[j]!.x.size = st.m - 6)
    (hsch : sh.sched = sch) (hf : Flock.fast100 st.m = some sch)
    -- the proof file: its decode, the messages w holds, the comb port and the levels' openings
    (hdec : decodeProof (shapeOf st sch extras.size) bytes = .ok p)
    (hzp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) p.zerocheck)
    (hlp : LcProofOf (decoded sh d.w d.comb d.cons) p.lincheck) (hpl : sh.holo = false)
    (hrsw : p.ringSwitches = (decoded sh d.w d.comb d.cons).rs)
    (hmessages : p.ligerito.messages =
      (decoded sh d.w d.comb d.cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2))
    (hoodValues : p.ligerito.oodValues = (decoded sh d.w d.comb d.cons).ood)
    (hyr : p.ligerito.finalProof.yr = (decoded sh d.w d.comb d.cons).yr)
    (hinitialCap : (p.ligerito.initialCap == p.cap) = true)
    (hfoldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
        .ok d.comb)
    (hopen : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ∀ cs ∈ info.lg.conss.toList, Opened st.merkle sch p.cap p.ligerito (decoded sh d.w d.comb d.cons) cs)
    -- the zerocheck's frames and counts
    (hn₀ : (t.rounds[t.k]!).n = 6) (hn₁ : (t.rounds[t.k + 1]!).n = st.m - 13) (hn₂ : (t.rounds[t.k + 2]!).n = 1)
    (hnr : ∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1)
    (ha₀ : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
      .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6]))
    (ha₁ : AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)]))
    (ha₂ : AnswersL (t.rounds[t.k + 2]!)
      (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1]))
    (har : ∀ i < st.m - 6, AnswersL (t.rounds[t.k + 3 + i]!)
      (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1]))
    -- the lincheck's frames and counts
    (hlc : LcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck)
    -- ring switching's frames and counts
    (hrn : ∀ k < 2 + extras.size, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).n = 7)
    (hra : ∀ k < 2 + extras.size, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).answers
      (rsFramed (rsPending ByteArray.empty k) p.ringSwitches[k]!))
    (hgn : (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).n = 2 + extras.size)
    (hga : (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).answers
      (ByteArray.empty ++ Transcript.header 4 2 (2 + extras.size)))
    -- the Ligerito rounds' frames
    (hfr : ∀ i < (chunks (ligFrames sch (valOf sh d.w d.comb d.cons .T) p.cap p.ligerito)).1.length,
      Retains (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size) + 1 + i]!)
        (encs ((chunks (ligFrames sch (valOf sh d.w d.comb d.cons .T) p.cap p.ligerito)).1.getD i []))) :
    ∃ t₄ : Transcript, repProg st sch p extras t = .ok ((), t₄) ∧ t₄.k = t.rounds.size := by
  obtain ⟨v, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩ := halg
  obtain ⟨-, -, -, -, -, hT, hSws⟩ :=
    rs_of_residuals hrows hS hw hc (hacc v info kappa hrows) hres hzero hm14 hm hkS hkL hkm hxs hext hd hne ht
      (hzp v info kappa hrows).fc hlp.zp hrsw rfl rfl hrn hra hgn hga
  have hsz := claimsF_size sh (valOf sh d.w d.comb d.cons) info extras
  obtain ⟨P', hF⟩ := hlg_of_frames (bytes := bytes) hrows hS hw hc (hacc v info kappa hrows) hres hzero hm14
    hm.symm hkL.symm hsch hf hdec hmessages hoodValues hyr hinitialCap (hopen v info kappa hrows)
    (claims := claimsF sh (valOf sh d.w d.comb d.cons) info extras) (rpp := fun k => (info.rs.sws[k]!).rpp)
    (gamma := info.rs.gamma) (by rw [hsz, hxs]) hT hSws hd hne hwf ht (k₁ := t.k + 3 + (st.m - 6))
    (k₂ := t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1) rfl rfl (fun i hi => by rw [hsz]; exact hfr i hi)
  refine ⟨⟨t.rounds, t.rounds.size, P'⟩, prog_of_algebra sh d ⟨v, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩
    hacc st sch p extras t ht hd hne hm hm14 hkL hkS hkm hxs hext hzp hlp hpl hrsw hfoldPts hn₀ hn₁ hn₂ hnr ha₀ ha₁ ha₂
    har hlc
    hrn hra hgn hga ?_, rfl⟩
  intro v' info' kappa' h'
  rw [hrows] at h'
  obtain ⟨rfl, rfl, rfl⟩ : v = v' ∧ info = info' ∧ kappa = kappa' := by simpa using h'
  have h := hF
  rw [hT, hSws, hsz] at h
  exact h

end FlockVBridge.Runs
