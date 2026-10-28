import Proofs.Flock.VBridge.Algebra.LigeritoConverse
import Proofs.Flock.VBridge.Compiled
import Proofs.Flock.Soundness.Refine.RepConverse
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# `prog_of_checks`' Ligerito hypothesis from E

`hlg_of_residuals` gives `Refine.prog_of_checks`' `hlg` at a rep: from E's preconditions at the rep's values
(`ligerito_of_residuals`' `hrows hS hw hc hacc hres hzero`), the proof file's decode and the fields that hold V*'s
messages, the stream's coins and the levels' openings, the verifier's `ligerito` (Ligerito.lean:165–277) returns from
round `k₂ + n + 1` at ring switching's `T` and switched claims (`rsT`, `rsSws`), having consumed every recorded round,
on every record whose rounds from there retain the bytes it frames.

* `ligerito_of_residuals_at`: `ligerito_of_residuals` with its start round, the rounds the zerocheck, the lincheck and
  ring switching draw (`zerocheck_coins`, `lincheck_coins`, `ringSwitch_spec`).
* `levelsLoop_count`, `ligeritoRun_conss`: V*'s Ligerito records one consistency context per level, so the proof's
  query and consistency nonces (`decodeProof_sizes`) are as many as the verifier consumes.
-/

namespace FlockVBridge.Runs

open Flock FlockSoundness.Refine Transcript
open Flock.RecAlgebra (Messages LSt Coins Info GfS verifierRows structureOf residuals valOf kapSum decoded Opened
  FinalOK)
open FlockSoundness.Discharge.Exec (chunks ligFrames ligerito_frames)

variable {c : Array (Array F128)}

/-- V*'s level loop records one consistency context per iteration (Structure.lean:415–432). -/
theorem levelsLoop_count [Fact (RecAlgebra.NoEmpty c)] {M : Messages F128} {sch : Flock.Schedule} {R : ℕ}
    (hq : ∀ i ≤ R, (sch.levels.getD i default).q ≠ 0) :
    ∀ (fuel i split : ℕ) (levels : Array (Array F256)) (e : ℕ) (st : LSt F128) (k : ℕ)
      (out : Array (Array F256) × ℕ) (st' : LSt F128) (k' : ℕ), i + fuel ≤ R →
      RecAlgebra.levelsLoop sch M R fuel i split levels e st k (Coins.ofRounds c) = .ok ((out, st'), k') →
      st'.conss.size = st.conss.size + fuel
  | 0, _, _, _, _, st, k, out, st', k', _, h => by
    simp only [RecAlgebra.levelsLoop] at h
    obtain ⟨-, rfl, -⟩ := RecAlgebra.vpure h
    rfl
  | fuel + 1, i, split, levels, e, st, k, out, st', k', hi, h => by
    simp only [RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (RecAlgebra.vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := RecAlgebra.vbind_ok h
    obtain ⟨-, -, -, -, -, hc₁, -⟩ := RecAlgebra.challenges_conv (p := default) 4 0 #[] st st₁ k k₁ chs h₁
    try dsimp only at h
    by_cases hi' : i + 1 < R
    · rw [ite_eq_left hi'] at h
      obtain ⟨u₂, st₂, k₂, h₂, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨u₃, st₃, k₃, h₃, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨u₄, st₄, k₄, h₄, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨-, -, -, hc₂, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
      obtain ⟨-, -, -, hc₃, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
      obtain ⟨-, -, -, -, mm, -, -, -, -, hc₄, -⟩ := RecAlgebra.vqueryFold (hq (i + 1) (by omega)) h₄
      rw [levelsLoop_count hq fuel (i + 1) _ _ _ st₄ k₄ out st' k' (by omega) h, hc₄, Array.size_push, hc₃, hc₂,
        hc₁]
      omega
    · rw [ite_eq_right hi'] at h
      obtain ⟨r, st₂, k₂, h₂, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨rfl, rfl, -, -, rfl⟩ := RecAlgebra.vquery (hq R le_rfl) h₂
      try dsimp only at h
      obtain ⟨-, h⟩ := RecAlgebra.vcoinOne_bind h
      obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := RecAlgebra.vbind_ok h
      obtain ⟨e₃, rfl⟩ := RecAlgebra.vmodify_ok h₃
      rw [levelsLoop_count hq fuel (i + 1) _ _ _ st₃ _ out st' k' (by omega) h, e₃, Array.size_push, hc₁]
      omega

/-- V*'s Ligerito up to its final check records one consistency context per level (Structure.lean:436–450). -/
theorem ligeritoRun_conss [Fact (RecAlgebra.NoEmpty c)] {sch : Flock.Schedule} {M : Messages F128} {T : F128}
    {k₀ k' : ℕ} {initCh : Array F256} {levels : Array (Array F256)} {ext : ℕ} {stV : LSt F128}
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hrun : (RecAlgebra.ligeritoRun sch M).run
        { s := RecAlgebra.kofF T, q := ((RecAlgebra.Val.zero, RecAlgebra.Val.zero),
          (RecAlgebra.Val.zero, RecAlgebra.Val.zero), (RecAlgebra.Val.zero, RecAlgebra.Val.zero)) } k₀
        (Coins.ofRounds c) = .ok (((initCh, levels, ext), stV), k')) :
    stV.conss.size = sch.levels.size - 1 + 1 := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hq : ∀ i ≤ sch.levels.size - 1, (sch.levels.getD i default).q ≠ 0 :=
    fun i hi => (hlv i (by omega)).q_ne_zero
  rw [StateT.run] at hrun
  unfold RecAlgebra.ligeritoRun at hrun
  obtain ⟨-, h⟩ := RecAlgebra.vcoinVec_bind (by omega) hrun
  obtain ⟨y0, -, h⟩ := RecAlgebra.vnextOod_bind h
  obtain ⟨-, h⟩ := RecAlgebra.vcoinOne_bind h
  obtain ⟨⟨⟩, stZ, kZ, hZ, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨eZ, rfl⟩ := RecAlgebra.vmodify_ok hZ
  subst eZ
  obtain ⟨m0, -, h⟩ := RecAlgebra.vnextMsg_bind h
  obtain ⟨⟨⟩, stA, kA, hA, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨eA, rfl⟩ := RecAlgebra.vmodify_ok hA
  obtain ⟨ic, st₁, k₁, h₁, h⟩ := RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, st₂, k₂, h₂, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₄, k₄, h₄, h⟩ := RecAlgebra.vbind_ok h
  obtain ⟨⟨lv', ex'⟩, st₅, k₅, h₅, h⟩ := RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨-, e5, -⟩ := RecAlgebra.vpure h
  subst e5
  obtain ⟨-, -, -, -, -, hc₁, -⟩ := RecAlgebra.challenges_conv (p := default) sch.k0 0 #[] _ _ _ _ _ h₁
  obtain ⟨-, -, -, hc₂, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
  obtain ⟨-, -, -, hc₃, -⟩ := RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
  obtain ⟨-, -, -, -, mq, -, -, -, -, hc₄, -⟩ := RecAlgebra.vqueryFold (hlv 0 (by omega)).q_ne_zero h₄
  have hA0 : stA.conss = #[] := by rw [eA]
  have hcount := levelsLoop_count hq _ _ _ _ _ st₄ _ _ _ _ (by omega) h₅
  rw [hc₄, Array.size_push, hc₃, hc₂, hc₁, hA0, Array.size_empty] at hcount
  clear h₁ h₂ h₃ h₄ h₅ eA
  omega

/-- **Ligerito at a rep, from its start round.** `ligerito_of_residuals` with the round V*'s Ligerito starts at: after
the zerocheck's `3 + (m − 6)` rounds, the lincheck's `3 + (k_log − 6)` and ring switching's `n + 1` for its `n` claims
(Structure.lean:532–539). -/
theorem ligerito_of_residuals_at {sh : Shape} {w comb : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info RecAlgebra.Lin}
    {kappa : Array (Array (Array F128))} {S : Flock.RecAlgebra.GfS} {rs : Array F128} (hm : 14 ≤ sh.m)
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons)
    (hres : residuals S w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0) :
    ∃ (ext : Nat) (st : LSt F128) (k' : Nat),
      (RecAlgebra.ligeritoRun sh.sched (decoded sh w comb cons)).run
          { s := RecAlgebra.kofF (valOf sh w comb cons .T),
            q := ((RecAlgebra.Val.zero, RecAlgebra.Val.zero), (RecAlgebra.Val.zero, RecAlgebra.Val.zero),
              (RecAlgebra.Val.zero, RecAlgebra.Val.zero)) }
          (3 + (sh.m - 6) + 3 + (sh.kLog - 6) + (2 + extras.size) + 1) (Coins.ofRounds rounds) =
        .ok (((info.lg.initCh, info.lg.levels, ext), st), k') ∧
      (Coins.ofRounds rounds).drawnAll k' = true ∧ st.msg = (decoded sh w comb cons).msgs.size ∧
      st.ood = (decoded sh w comb cons).ood.size ∧ info.lg.conss = st.conss ∧
      FinalOK sh.sched info.rs.sws (decoded sh w comb cons).yr info.lg.initCh info.lg.levels ext st := by
  obtain ⟨rs', k, hrun, hres'⟩ := RecAlgebra.residuals_eq_run w comb cons hrows hS hw hc hacc
  rw [hres] at hres'
  cases hres'
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, h1, h2, h3, h4, hrs⟩ := RecAlgebra.run_stages hrun
  obtain ⟨_, _, _, -, -, -, -, -, -, hk1, -⟩ := RecAlgebra.zerocheck_coins hm h1
  obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, -, -, hk2, -⟩ := RecAlgebra.lincheck_coins h2
  obtain ⟨_, -, -, -, hk3, -, -, -⟩ := RecAlgebra.ringSwitch_spec (by simp [RecAlgebra.claimsOf]) h3
  have hzc : (decoded sh w comb cons).zc.size = sh.m - 6 := by
    simp [decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic]
  have hlc : (decoded sh w comb cons).lc.size = sh.kLog - 6 := by
    simp [decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic]
  have hcl : (RecAlgebra.claimsOf sh.m sh.kLog (info.map (RecAlgebra.Lin.eval (valOf sh w comb cons))).zc
      (info.map (RecAlgebra.Lin.eval (valOf sh w comb cons))).lc extras).size = 2 + extras.size := by
    simp [RecAlgebra.claimsOf]
  have hT : (decoded sh w comb cons).tAtom.getD (info.map (RecAlgebra.Lin.eval (valOf sh w comb cons))).rs.T =
      valOf sh w comb cons .T := by
    simp [decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic, RecAlgebra.Lin.eval_var]
  rw [hT] at h4
  have ek : k3 = 3 + (sh.m - 6) + 3 + (sh.kLog - 6) + (2 + extras.size) + 1 := by
    rw [hk3, hcl, hk2, hlc, hk1, hzc]
  rw [ek] at h4
  obtain ⟨ext, st, hl, hd, hmsg, hood, -, hlc', hf, -⟩ :=
    RecAlgebra.ligerito_checks h4 (fun x hx => hzero x (by rw [hrs]; simp [hx]))
  exact ⟨ext, st, k, hl, hd, hmsg, hood, hlc', hf⟩

/-- **`prog_of_checks`' `hlg` from E** (Ligerito.lean:165–277; Verify.lean:49). At a rep whose values meet E's
preconditions (`hrows … hzero`, at the shape `sh` of the setup `st` and its `fast100` schedule), whose proof file
decodes and holds V*'s Ligerito messages, OOD values and final `y_r` and the commitment's cap as its initial cap, whose
levels' openings pass (`Opened`, at the consistency contexts V*'s coins give), at ring switching's `T` and switched
claims as V* computes them (`hT`, `hsws`), on a record whose coins are the rep's: the verifier's Ligerito returns from
round `k₂ + n + 1` with every recorded round consumed, on the record whose rounds from there retain the bytes `F` it
frames. -/
theorem hlg_of_residuals {sh : Shape} {d : Compiled.RepData} {st : Setup} {sch : Flock.Schedule} {p : Proof}
    {bytes : ByteArray} {extras : Array Claim} {t : Transcript} {v : Array F128} {info : Info RecAlgebra.Lin}
    {kappa : Array (Array (Array F128))} {S : Flock.RecAlgebra.GfS} {rs : Array F128}
    -- E's preconditions at the rep
    (hrows : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64)
    (hacc : ∀ r < 2, valOf sh d.w d.comb d.cons (.acc1 r) + valOf sh d.w d.comb d.cons (.acc0 r) =
      kapSum kappa r d.cons)
    (hres : residuals S d.w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0)
    -- the shape
    (hm : 14 ≤ st.m) (hsm : sh.m = st.m) (hsk : sh.kLog = st.fold.kLog) (hsch : sh.sched = sch)
    (hf : Flock.fast100 st.m = some sch)
    -- the proof file
    (hdec : decodeProof (shapeOf st sch extras.size) bytes = .ok p)
    (hmessages : p.ligerito.messages =
      (decoded sh d.w d.comb d.cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2))
    (hoodValues : p.ligerito.oodValues = (decoded sh d.w d.comb d.cons).ood)
    (hyr : p.ligerito.finalProof.yr = (decoded sh d.w d.comb d.cons).yr)
    (hinitialCap : (p.ligerito.initialCap == p.cap) = true)
    (hopen : ∀ cs ∈ info.lg.conss.toList, Opened st.merkle sch p.cap p.ligerito (decoded sh d.w d.comb d.cons) cs)
    -- ring switching's `T` and switched claims
    {claims : Array Claim} {rpp : ℕ → Array F128} {gamma : Array F128} (hn : claims.size = 2 + d.extras.size)
    (hT : rsT claims p.ringSwitches rpp gamma = valOf sh d.w d.comb d.cons .T)
    (hsws : rsSws claims rpp gamma = info.rs.sws)
    -- the record
    (hcoins : d.rounds = t.rounds.map (·.coins)) (hne : RecAlgebra.NoEmpty d.rounds) (hwf : RoundsWF t.rounds)
    (htk : t.k = 0)
    {k₁ k₂ : ℕ} (hk₁ : k₁ = t.k + 3 + (st.m - 6)) (hk₂ : k₂ = k₁ + 2 + (st.fold.kLog - 6) + 1) :
    ∃ (F : ℕ → ByteArray) (P' : ByteArray),
      (∀ j, k₂ + claims.size + 1 ≤ j → j < t.rounds.size → (t.rounds[j]!).bytes = some (F j)) →
        Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
            (Flock.basis (rsSws claims rpp gamma)) ⟨t.rounds, k₂ + claims.size + 1, ByteArray.empty⟩ =
          .ok ((), ⟨t.rounds, t.rounds.size, P'⟩) := by
  subst hsch
  obtain ⟨ext, stV, k', hl, hd, hmsg, hood, hlc, hfin⟩ :=
    ligerito_of_residuals_at (by omega) hrows hS hw hc hacc hres hzero
  have : Fact (RecAlgebra.NoEmpty d.rounds) := ⟨hne⟩
  have hk' : k' = t.rounds.size := by
    have e : (k' == (d.rounds.filter (·.size ≠ 0)).size) = true := hd
    rw [RecAlgebra.filter_noEmpty hne, beq_iff_eq, hcoins, Array.size_map] at e
    exact e
  subst hk'
  have hdim := (fast100_dims st.m sh.sched hf).1
  have hlv : ∀ i < sh.sched.levels.size, LevelSeam (sh.sched.levels.getD i default) := fun i hi => by
    have e : sh.sched.levels.getD i default = sh.sched.levels[i] := by simp [Array.getD, hi]
    rw [e]
    exact fast100_level st.m sh.sched hf _ (Array.getElem_mem hi)
  have hsz := FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ hdec
  have hqn : p.ligerito.queryNonces.size = sh.sched.levels.size - 1 + 1 := hsz.queryNonces
  have hcn : p.ligerito.consistencyNonces.size = sh.sched.levels.size - 1 + 1 := hsz.consistencyNonces
  have hcbn : p.ligerito.claimNonces.size = 3 * (sh.sched.levels.size - 1) + 2 := hsz.claimNonces
  have hcs := ligeritoRun_conss hdim hlv hl
  have hoo : (decoded sh d.w d.comb d.cons).ood.size = 1 + 2 * (sh.sched.levels.size - 1) := by
    simp [decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic]
  have H := RecAlgebra.ligerito_of_run (ms := st.merkle) (cap := p.cap) (p := p.ligerito) hdim hlv hl hfin hmsg hood
    hmessages hoodValues hyr hinitialCap (by omega) (by omega) (by omega) (by rw [← hlc]; exact hopen)
  obtain ⟨P', F, hF⟩ := H.2 ByteArray.empty
  refine ⟨F, P', fun hb => ?_⟩
  have e : k₂ + claims.size + 1 = 3 + (sh.m - 6) + 3 + (sh.kLog - 6) + (2 + d.extras.size) + 1 := by
    subst hk₁ hk₂
    omega
  rw [hT, hsws, e]
  refine hF t.rounds fun j hj hj' => ⟨t.rounds[j], getElem?_pos t.rounds j hj', ?_, ?_, ?_⟩
  · rw [hcoins, getElem!_pos _ j (by simpa using hj'), Array.getElem_map, ← getElem!_pos t.rounds j hj']
    exact (hwf j hj').symm
  · rw [hcoins, getElem!_pos _ j (by simpa using hj'), Array.getElem_map]
  · rw [← getElem!_pos t.rounds j hj']
    exact hb j (by omega) hj'

/-- A record fits the bytes `F` from round `k₀` to its end when its coins are `c`'s, its rounds well formed and each
retains `F`'s bytes. -/
theorem fits_of {F : ℕ → ByteArray} {rs : Array Round} {k₀ : ℕ} (hcoins : ∀ j < rs.size, c[j]! = (rs[j]!).coins)
    (hwf : RoundsWF rs) (hb : ∀ j, k₀ ≤ j → j < rs.size → (rs[j]!).bytes = some (F j)) :
    RecAlgebra.Fits c F rs k₀ rs.size := fun j hj hj' =>
  ⟨rs[j]!, by rw [getElem!_pos rs j hj']; exact getElem?_pos rs j hj', by rw [hcoins j hj', hwf j hj'],
    (hcoins j hj').symm, hb j hj hj'⟩

theorem bytes_of_retains {r : Round} {B : ByteArray} {L : List UInt8} (h : Retains r L) (hB : B.data.toList = L) :
    r.bytes = some B := by
  unfold Retains at h
  cases hb : r.bytes with
  | none => rw [hb] at h; cases h
  | some C =>
    rw [hb] at h
    have hc : C.data.toList = L := Option.some.inj h
    have e : C.data = B.data := Array.ext' (hc.trans hB.symm)
    cases C; cases B; cases e; rfl

/-- **`prog_of_checks`' `hlg` from E, at named frames** (Ligerito.lean:165–277). `hlg_of_residuals` with the bytes
each Ligerito round retains named: round `k₂ + n + 1 + i` retains the `i`-th round of `ligFrames` (the frames the
verifier's Ligerito emits, `Exec.ligerito_frames`) at V*'s `T`, the commitment's cap and the proof's Ligerito fields. -/
theorem hlg_of_frames {sh : Shape} {d : Compiled.RepData} {st : Setup} {sch : Flock.Schedule} {p : Proof}
    {bytes : ByteArray} {extras : Array Claim} {t : Transcript} {v : Array F128} {info : Info RecAlgebra.Lin}
    {kappa : Array (Array (Array F128))} {S : Flock.RecAlgebra.GfS} {rs : Array F128}
    -- E's preconditions at the rep
    (hrows : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64)
    (hacc : ∀ r < 2, valOf sh d.w d.comb d.cons (.acc1 r) + valOf sh d.w d.comb d.cons (.acc0 r) =
      kapSum kappa r d.cons)
    (hres : residuals S d.w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0)
    -- the shape
    (hm : 14 ≤ st.m) (hsm : sh.m = st.m) (hsk : sh.kLog = st.fold.kLog) (hsch : sh.sched = sch)
    (hf : Flock.fast100 st.m = some sch)
    -- the proof file
    (hdec : decodeProof (shapeOf st sch extras.size) bytes = .ok p)
    (hmessages : p.ligerito.messages =
      (decoded sh d.w d.comb d.cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2))
    (hoodValues : p.ligerito.oodValues = (decoded sh d.w d.comb d.cons).ood)
    (hyr : p.ligerito.finalProof.yr = (decoded sh d.w d.comb d.cons).yr)
    (hinitialCap : (p.ligerito.initialCap == p.cap) = true)
    (hopen : ∀ cs ∈ info.lg.conss.toList, Opened st.merkle sch p.cap p.ligerito (decoded sh d.w d.comb d.cons) cs)
    -- ring switching's `T` and switched claims
    {claims : Array Claim} {rpp : ℕ → Array F128} {gamma : Array F128} (hn : claims.size = 2 + d.extras.size)
    (hT : rsT claims p.ringSwitches rpp gamma = valOf sh d.w d.comb d.cons .T)
    (hsws : rsSws claims rpp gamma = info.rs.sws)
    -- the record
    (hcoins : d.rounds = t.rounds.map (·.coins)) (hne : RecAlgebra.NoEmpty d.rounds) (hwf : RoundsWF t.rounds)
    (htk : t.k = 0)
    {k₁ k₂ : ℕ} (hk₁ : k₁ = t.k + 3 + (st.m - 6)) (hk₂ : k₂ = k₁ + 2 + (st.fold.kLog - 6) + 1)
    -- the Ligerito rounds' frames
    (hfr : ∀ i < (chunks (ligFrames sch (valOf sh d.w d.comb d.cons .T) p.cap p.ligerito)).1.length,
      Retains (t.rounds[k₂ + claims.size + 1 + i]!)
        (encs ((chunks (ligFrames sch (valOf sh d.w d.comb d.cons .T) p.cap p.ligerito)).1.getD i []))) :
    ∃ P' : ByteArray,
      Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
          (Flock.basis (rsSws claims rpp gamma)) ⟨t.rounds, k₂ + claims.size + 1, ByteArray.empty⟩ =
        .ok ((), ⟨t.rounds, t.rounds.size, P'⟩) := by
  subst hsch
  obtain ⟨ext, stV, k', hl, hd, hmsg, hood, hlc, hfin⟩ :=
    ligerito_of_residuals_at (by omega) hrows hS hw hc hacc hres hzero
  have : Fact (RecAlgebra.NoEmpty d.rounds) := ⟨hne⟩
  have hk' : k' = t.rounds.size := by
    have e : (k' == (d.rounds.filter (·.size ≠ 0)).size) = true := hd
    rw [RecAlgebra.filter_noEmpty hne, beq_iff_eq, hcoins, Array.size_map] at e
    exact e
  subst hk'
  have hdim := (fast100_dims st.m sh.sched hf).1
  have hlv : ∀ i < sh.sched.levels.size, LevelSeam (sh.sched.levels.getD i default) := fun i hi => by
    have e : sh.sched.levels.getD i default = sh.sched.levels[i] := by simp [Array.getD, hi]
    rw [e]
    exact fast100_level st.m sh.sched hf _ (Array.getElem_mem hi)
  have hsz := FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ hdec
  have hqn : p.ligerito.queryNonces.size = sh.sched.levels.size - 1 + 1 := hsz.queryNonces
  have hcn : p.ligerito.consistencyNonces.size = sh.sched.levels.size - 1 + 1 := hsz.consistencyNonces
  have hcbn : p.ligerito.claimNonces.size = 3 * (sh.sched.levels.size - 1) + 2 := hsz.claimNonces
  have hcs := ligeritoRun_conss hdim hlv hl
  have hoo : (decoded sh d.w d.comb d.cons).ood.size = 1 + 2 * (sh.sched.levels.size - 1) := by
    simp [decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic]
  have H := RecAlgebra.ligerito_of_run (ms := st.merkle) (cap := p.cap) (p := p.ligerito) hdim hlv hl hfin hmsg hood
    hmessages hoodValues hyr hinitialCap (by omega) (by omega) (by omega) (by rw [← hlc]; exact hopen)
  obtain ⟨P', F, hF⟩ := H.2 ByteArray.empty
  refine ⟨P', ?_⟩
  obtain ⟨k₀, hk₀⟩ : ∃ k₀, k₀ = 3 + (sh.m - 6) + 3 + (sh.kLog - 6) + (2 + d.extras.size) + 1 := ⟨_, rfl⟩
  rw [← hk₀] at hF
  have e : k₂ + claims.size + 1 = k₀ := by
    subst hk₁ hk₂
    omega
  rw [hT, hsws, e]
  have hct : ∀ j < t.rounds.size, d.rounds[j]! = (t.rounds[j]!).coins := fun j hj => by
    rw [hcoins, getElem!_pos _ j (by simpa using hj), Array.getElem_map, getElem!_pos t.rounds j hj]
  obtain ⟨rs', hrs'⟩ : ∃ rs' : Array Round, rs' = t.rounds.mapIdx fun j r => { r with bytes := some (F j) } :=
    ⟨_, rfl⟩
  have hsz' : rs'.size = t.rounds.size := by rw [hrs', Array.size_mapIdx]
  have hget : ∀ j < t.rounds.size, (rs'[j]!).n = (t.rounds[j]!).n ∧ (rs'[j]!).coins = (t.rounds[j]!).coins ∧
      (rs'[j]!).bytes = some (F j) := fun j hj => by
    rw [getElem!_pos rs' j (by omega), getElem!_pos t.rounds j hj]
    subst hrs'
    refine ⟨?_, ?_, ?_⟩ <;> simp [Array.getElem_mapIdx]
  have hfit' := fits_of (c := d.rounds) (F := F) (rs := rs') (k₀ := k₀)
    (fun j hj => by rw [hct j (by omega), (hget j (by omega)).2.1])
    (fun j hj => by rw [(hget j (by omega)).2.1, (hget j (by omega)).1]; exact hwf j (by omega))
    (fun j _ hj => (hget j (by omega)).2.2)
  rw [hsz'] at hfit'
  have E := ligerito_frames st.merkle sh.sched _ p.cap p.ligerito _ hdim hlv (t := ⟨rs', k₀, ByteArray.empty⟩)
    (fun i hi => ⟨F i, (hget i (by change i < rs'.size at hi; omega)).2.2⟩) (hF rs' hfit')
  have hlen : t.rounds.size = k₀ + (chunks (ligFrames sh.sched (valOf sh d.w d.comb d.cons .T) p.cap
      p.ligerito)).1.length := E.k
  refine hF t.rounds (fits_of hct hwf fun j hj hj' => ?_)
  have hR := E.ret (j - k₀) (by omega)
  dsimp only at hR
  have hp : (if j - k₀ = 0 then ByteArray.empty.data.toList else ([] : List UInt8)) = [] := by split <;> rfl
  rw [hp, List.nil_append, show k₀ + (j - k₀) = j by omega] at hR
  unfold Retains at hR
  rw [(hget j hj').2.2] at hR
  have ht := hfr (j - k₀) (by omega)
  rw [e, show k₀ + (j - k₀) = j by omega] at ht
  exact bytes_of_retains ht (Option.some.inj hR)

end FlockVBridge.Runs
