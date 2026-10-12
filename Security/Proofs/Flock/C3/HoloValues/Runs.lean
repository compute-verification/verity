import Proofs.Flock.C3.HoloValues.Frames
import Proofs.Flock.VBridge.Runs.Lg
import Proofs.Flock.VBridge.Runs.Rs

/-!
# A holo rep's ring switching and Ligerito, from E

G's `rs_of_residuals` and `hlg_of_frames` at a holo record, whose one round of no coins (`z_partial`'s,
`HoloCoinsShaped`) V* skips (`Coins.ofRounds`): ring switching and Ligerito run on the record one round later than
V*'s index for them.

* `holoRs_of_residuals`: `rs_of_residuals` on the record with its empty round filtered out (V*'s view; `verifierRows`
  is unchanged by the filter, `ofRounds_filter`), read back at the record (`filter_getElem!_after`, `rsRounds_shift`).
* `LigShift`, a stub: V*'s Ligerito run on coins with one more round in front returns the same one round later.
* `holoHlg_of_frames`: `hlg_of_frames` from `LigShift`, with `ligerito_of_run` at V*'s coins behind a round of one
  coin (`#[#[0]]`), which the record fits from Ligerito's first round on (`fits_from`).
-/

namespace FlockC3.HoloValues

open Flock Flock.RecAlgebra FlockSoundness.Refine
open FlockVBridge.Compiled (RepData)
open FlockSoundness.Discharge.Exec (chunks ligFrames ligerito_frames)
open FlockVBridge.Runs (rs_of_residuals ligerito_of_residuals_at ligeritoRun_conss bytes_of_retains)

/-! ## Ring switching -/

/-- Ring switching's rounds, read at another start of the same rounds. -/
theorem rsRounds_shift {R R' : Array Round} {a b : ℕ} {rs : Array (Array F128)} {n : ℕ} {rpp : ℕ → Array F128}
    {gamma : Array F128} (hget : ∀ k ≤ n, R'[a + k]! = R[b + k]!) (hsz : a + n < R'.size → b + n < R.size)
    (h : RsRounds ⟨R', a, ByteArray.empty⟩ rs n rpp gamma) : RsRounds ⟨R, b, ByteArray.empty⟩ rs n rpp gamma := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  refine ⟨hsz h1, fun k hk => ?_, fun k hk => ?_, fun k hk => ?_, ?_, ?_, ?_⟩
  · show (R[b + k]!).n = 7
    rw [← hget k (by omega)]
    exact h2 k hk
  · show (R[b + k]!).coins = rpp k
    rw [← hget k (by omega)]
    exact h3 k hk
  · show (R[b + k]!).answers _
    rw [← hget k (by omega)]
    exact h4 k hk
  · show (R[b + n]!).n = n
    rw [← hget n le_rfl]
    exact h5
  · show (R[b + n]!).coins = gamma
    rw [← hget n le_rfl]
    exact h6
  · show (R[b + n]!).answers _
    rw [← hget n le_rfl]
    exact h7

/-- **Ring switching's checks at a holo rep** (`rs_of_residuals`). On a record whose only round of no coins is
`z_partial`'s: the claims' lincheck `r′` is the coin of the round after it, and ring switching's rounds start two
rounds past it. -/
theorem holoRs_of_residuals {sh : Shape} {d : RepData} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {res : Array F128}
    (hrows : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64)
    (hacc : ∀ r < 2, valOf sh d.w d.comb d.cons (.acc1 r) + valOf sh d.w d.comb d.cons (.acc0 r) =
      kapSum kappa r d.cons)
    (hres : residuals S d.w v = .ok res) (hzero : ∀ x ∈ res, x = 0)
    {st : Setup} {p : Proof} {extras : Array Claim} {t : Transcript}
    (hm : 14 ≤ st.m) (hmS : st.m = sh.m) (hkS : st.kLog = sh.kLog) (hfS : st.fold.kLog = sh.kLog)
    (hkm : st.kLog ≤ st.m) (hxs : d.extras = extras)
    (hext : ∀ j < extras.size, extras[j]!.skip.size = 64 ∧ extras[j]!.x.size = st.m - 6)
    (hco : d.rounds = t.rounds.map (·.coins))
    -- `z_partial`'s round, the record's only one of no coins
    (hZ : 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) < t.rounds.size)
    (hz0 : (t.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6)]!).coins.size = 0)
    (hne : ∀ j < t.rounds.size, j ≠ 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) → (t.rounds[j]!).coins.size ≠ 0)
    (hfc : p.zerocheck.finalC = info.zc.vc.eval (valOf sh d.w d.comb d.cons))
    (hzp : p.lincheck.zPartial = (decoded sh d.w d.comb d.cons).zp)
    (hrsw : p.ringSwitches = (decoded sh d.w d.comb d.cons).rs)
    {k₂ : ℕ} (hk₂ : k₂ = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2)
    (hrn : ∀ k < 2 + extras.size, (t.rounds[k₂ + k]!).n = 7)
    (hra : ∀ k < 2 + extras.size,
      (t.rounds[k₂ + k]!).answers (rsFramed (rsPending ByteArray.empty k) p.ringSwitches[k]!))
    (hgn : (t.rounds[k₂ + (2 + extras.size)]!).n = 2 + extras.size)
    (hga : (t.rounds[k₂ + (2 + extras.size)]!).answers (ByteArray.empty ++ Transcript.header 4 2 (2 + extras.size))) :
    claimsF sh (valOf sh d.w d.comb d.cons) info extras =
        abcClaims st.m st.kLog (zcOf p.zerocheck info.zc.r info.zc.z info.zc.rho)
          (lcOf p.lincheck ((t.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1]!).coins.getD 0 0)
            info.lc.fold.ts) ++ extras ∧
      p.ringSwitches.size = (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size ∧
      (∀ k < (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size,
        p.ringSwitches[k]!.size = 128 ∧ (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!.skip.size = 64 ∧
          (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!.x.size = st.m - 6) ∧
      (∀ k < (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size,
        dot (wOf (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!) p.ringSwitches[k]! =
          (claimsF sh (valOf sh d.w d.comb d.cons) info extras)[k]!.value) ∧
      RsRounds ⟨t.rounds, k₂, ByteArray.empty⟩ p.ringSwitches
        (claimsF sh (valOf sh d.w d.comb d.cons) info extras).size (fun k => (info.rs.sws[k]!).rpp) info.rs.gamma ∧
      rsT (claimsF sh (valOf sh d.w d.comb d.cons) info extras) p.ringSwitches (fun k => (info.rs.sws[k]!).rpp)
          info.rs.gamma = valOf sh d.w d.comb d.cons .T ∧
      rsSws (claimsF sh (valOf sh d.w d.comb d.cons) info extras) (fun k => (info.rs.sws[k]!).rpp) info.rs.gamma =
        info.rs.sws := by
  obtain ⟨R', hR'⟩ : ∃ R', R' = t.rounds.filter (fun x => decide (x.coins.size ≠ 0)) := ⟨_, rfl⟩
  have hp0 : (fun x : Round => decide (x.coins.size ≠ 0)) t.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6)]! =
      false := decide_eq_false (by intro h; exact h hz0)
  have hp1 : ∀ j < t.rounds.size, j ≠ 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) →
      (fun x : Round => decide (x.coins.size ≠ 0)) t.rounds[j]! = true := fun j hj hjz =>
    decide_eq_true (hne j hj hjz)
  have hget : ∀ j, 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) ≤ j → R'[j]! = t.rounds[j + 1]! := by
    rw [hR']
    exact filter_getElem!_after (fun x : Round => decide (x.coins.size ≠ 0)) t.rounds _ hZ hp0 hp1
  have hRsz : R'.size + 1 = t.rounds.size := by
    rw [hR']
    exact filter_size_after (fun x : Round => decide (x.coins.size ≠ 0)) t.rounds _ hZ hp0 hp1
  have hd' : d.rounds.filter (·.size ≠ 0) = R'.map (·.coins) := by
    rw [hco, Array.filter_map, hR']
    rfl
  have hrows' : ∀ ex cb, verifierRows sh (d.rounds.filter (·.size ≠ 0)) ex cb = verifierRows sh d.rounds ex cb := by
    intro ex cb
    unfold verifierRows
    rw [ofRounds_filter]
  obtain ⟨hcl, hrs, hshape, hval, hRR, hT, hSws⟩ := rs_of_residuals
    (d := { d with rounds := d.rounds.filter (·.size ≠ 0) }) (t := ⟨R', 0, ByteArray.empty⟩)
    ((hrows' _ _).trans hrows) hS hw hc hacc hres hzero hm hmS hkS hfS hkm hxs hext hd' (noEmpty_filter d.rounds) rfl
    hfc hzp hrsw (k₁ := 3 + (st.m - 6)) (k₂ := 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1)
    (by show 3 + (st.m - 6) = 0 + 3 + (st.m - 6); omega) rfl
    (fun k hk => by
      show (R'[3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).n = 7
      rw [hget (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k) (by omega),
        show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k + 1 = k₂ + k by omega]
      exact hrn k hk)
    (fun k hk => by
      show (R'[3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).answers _
      rw [hget (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k) (by omega),
        show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k + 1 = k₂ + k by omega]
      exact hra k hk)
    (by
      show (R'[3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).n = 2 + extras.size
      rw [hget (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)) (by omega),
        show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size) + 1 = k₂ + (2 + extras.size) by omega]
      exact hgn)
    (by
      show (R'[3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).answers _
      rw [hget (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)) (by omega),
        show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size) + 1 = k₂ + (2 + extras.size) by omega]
      exact hga)
  dsimp only at hcl hrs hshape hval hRR hT hSws
  rw [hget (3 + (st.m - 6) + 2 + (st.fold.kLog - 6)) le_rfl] at hcl
  refine ⟨hcl, hrs, hshape, hval, rsRounds_shift (fun k _ => ?_) (fun _ => by omega) hRR, hT, hSws⟩
  rw [hget (3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k) (by omega),
    show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k + 1 = k₂ + k by omega]

/-! ## Ligerito -/

/-- **Stub: V*'s Ligerito run behind one more round** (150–200 lines). On coins with one more round of some coins in
front, V*'s Ligerito returns the same from one round later: `Coins.ofRounds` keeps that round, so its round `k + 1` is
the shorter coins' round `k`, and `coinVec`, which alone reads the counter (RecAlgebra.lean:55–61), draws the same. Its
proof is an invariant over `LM`, written like `Conss.CS` (`VBridge/Family/Opened.lean`). -/
def LigShift : Prop :=
  ∀ (sch : Flock.Schedule) (M : Messages F128) (s₀ : LSt F128) (k k' : ℕ) (c : Array (Array F128)) (x : Array F128)
    (out : (Array F256 × Array (Array F256) × ℕ) × LSt F128), x.size ≠ 0 →
    (ligeritoRun sch M).run s₀ k (Coins.ofRounds c) = .ok (out, k') →
      (ligeritoRun sch M).run s₀ (k + 1) (Coins.ofRounds (#[x] ++ c)) = .ok (out, k' + 1)

/-- `fits_of` from round `k₀` on: the coins need be `c`'s only there. -/
theorem fits_from {c : Array (Array F128)} {F : ℕ → ByteArray} {rs : Array Round} {k₀ : ℕ}
    (hcoins : ∀ j, k₀ ≤ j → j < rs.size → c[j]! = (rs[j]!).coins) (hwf : RoundsWF rs)
    (hb : ∀ j, k₀ ≤ j → j < rs.size → (rs[j]!).bytes = some (F j)) : Fits c F rs k₀ rs.size := fun j hj hj' =>
  ⟨rs[j]!, by rw [getElem!_pos rs j hj']; exact getElem?_pos rs j hj', by rw [hcoins j hj hj', hwf j hj'],
    (hcoins j hj hj').symm, hb j hj hj'⟩

theorem cons_getElem!_succ {α : Type} [Inhabited α] (x : α) (xs : Array α) (j : ℕ) :
    (#[x] ++ xs)[j + 1]! = xs[j]! := by
  have e : (#[x] ++ xs).toList = x :: xs.toList := by simp
  rw [array_getElem!_toList (#[x] ++ xs), e, List.getElem!_cons_succ, ← array_getElem!_toList]

/-- **`hlg_of_frames` at a holo rep, from `LigShift`.** On a record whose only round of no coins is `z_partial`'s, the
verifier's Ligerito returns from round `3 + (m − 6) + 2 + (k_log − 6) + 2 + n + 1`, one past V*'s index for it, with
every recorded round consumed, when the rounds from there retain the Ligerito frames. -/
theorem holoHlg_of_frames (hsh : LigShift) {sh : Shape} {d : RepData} {st : Setup} {sch : Flock.Schedule}
    {p : Proof} {bytes : ByteArray} {extras : Array Claim} {t : Transcript} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128}
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
    -- the record, `z_partial`'s round its only one of no coins
    (hcoins : d.rounds = t.rounds.map (·.coins)) (hwf : RoundsWF t.rounds)
    (hZ : 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) < t.rounds.size)
    (hz0 : (t.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6)]!).coins.size = 0)
    (hne : ∀ j < t.rounds.size, j ≠ 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) → (t.rounds[j]!).coins.size ≠ 0)
    -- the Ligerito rounds' frames
    (hfr : ∀ i < (chunks (ligFrames sch (valOf sh d.w d.comb d.cons .T) p.cap p.ligerito)).1.length,
      Retains (t.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2 + claims.size + 1 + i]!)
        (encs ((chunks (ligFrames sch (valOf sh d.w d.comb d.cons .T) p.cap p.ligerito)).1.getD i []))) :
    ∃ P' : ByteArray,
      Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
          (Flock.basis (rsSws claims rpp gamma))
          ⟨t.rounds, 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2 + claims.size + 1, ByteArray.empty⟩ =
        .ok ((), ⟨t.rounds, t.rounds.size, P'⟩) := by
  subst hsch
  obtain ⟨ext, stV, k', hl, hd, hmsg, hood, hlc, hfin⟩ :=
    ligerito_of_residuals_at (by omega) hrows hS hw hc hacc hres hzero
  have hdsz : d.rounds.size = t.rounds.size := by rw [hcoins, Array.size_map]
  have hct : ∀ j < t.rounds.size, d.rounds[j]! = (t.rounds[j]!).coins := fun j hj => by
    rw [hcoins, getElem!_pos _ j (by simpa using hj), Array.getElem_map, getElem!_pos t.rounds j hj]
  -- V*'s coins: the record's past its empty round, behind a round of one coin
  have hZd : 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) < d.rounds.size := by omega
  have hp0 : (fun x : Array F128 => decide (x.size ≠ 0)) d.rounds[3 + (st.m - 6) + 2 + (st.fold.kLog - 6)]! =
      false := decide_eq_false (by intro h; exact h (by rw [hct _ hZ]; exact hz0))
  have hp1 : ∀ j < d.rounds.size, j ≠ 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) →
      (fun x : Array F128 => decide (x.size ≠ 0)) d.rounds[j]! = true := fun j hj hjz =>
    decide_eq_true (by rw [hct j (by omega)]; exact hne j (by omega) hjz)
  have hafter := filter_getElem!_after (fun x : Array F128 => decide (x.size ≠ 0)) d.rounds _ hZd hp0 hp1
  have hfsz := filter_size_after (fun x : Array F128 => decide (x.size ≠ 0)) d.rounds _ hZd hp0 hp1
  obtain ⟨cs, hcsd⟩ : ∃ cs : Array (Array F128), cs = #[#[0]] ++ d.rounds.filter (·.size ≠ 0) := ⟨_, rfl⟩
  have hneC : NoEmpty cs := fun r hr => by
    rw [hcsd, Array.mem_append] at hr
    rcases hr with hr | hr
    · rw [Array.mem_singleton] at hr
      subst hr
      simp
    · exact noEmpty_filter d.rounds r hr
  have : Fact (NoEmpty cs) := ⟨hneC⟩
  have hl' := hl
  rw [← ofRounds_filter] at hl'
  have hlS := hsh _ _ _ _ _ _ #[0] _ (by simp) hl'
  rw [← hcsd] at hlS
  have hk' : k' + 1 = t.rounds.size := by
    have e : (k' == (d.rounds.filter (·.size ≠ 0)).size) = true := hd
    rw [beq_iff_eq] at e
    omega
  rw [hk'] at hlS
  have hdim := (fast100_dims st.m sh.sched hf).1
  have hlv : ∀ i < sh.sched.levels.size, LevelSeam (sh.sched.levels.getD i default) := fun i hi => by
    have e : sh.sched.levels.getD i default = sh.sched.levels[i] := by simp [Array.getD, hi]
    rw [e]
    exact fast100_level st.m sh.sched hf _ (Array.getElem_mem hi)
  have hsz := FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ hdec
  have hqn : p.ligerito.queryNonces.size = sh.sched.levels.size - 1 + 1 := hsz.queryNonces
  have hcn : p.ligerito.consistencyNonces.size = sh.sched.levels.size - 1 + 1 := hsz.consistencyNonces
  have hcbn : p.ligerito.claimNonces.size = 3 * (sh.sched.levels.size - 1) + 2 := hsz.claimNonces
  have hcss := ligeritoRun_conss hdim hlv hlS
  have hoo : (decoded sh d.w d.comb d.cons).ood.size = 1 + 2 * (sh.sched.levels.size - 1) := by
    simp [decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic]
  have H := RecAlgebra.ligerito_of_run (ms := st.merkle) (cap := p.cap) (p := p.ligerito) hdim hlv hlS hfin hmsg hood
    hmessages hoodValues hyr hinitialCap (by omega) (by omega) (by omega) (by rw [← hlc]; exact hopen)
  obtain ⟨P', F, hF⟩ := H.2 ByteArray.empty
  refine ⟨P', ?_⟩
  obtain ⟨k₀, hk₀⟩ : ∃ k₀, k₀ = 3 + (sh.m - 6) + 3 + (sh.kLog - 6) + (2 + d.extras.size) + 1 + 1 := ⟨_, rfl⟩
  rw [← hk₀] at hF
  have e : 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2 + claims.size + 1 = k₀ := by omega
  rw [hT, hsws, e]
  have hctS : ∀ j, k₀ ≤ j → j < t.rounds.size → cs[j]! = (t.rounds[j]!).coins := fun j hj hj' => by
    obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
    rw [hcsd, cons_getElem!_succ, hafter i (by omega), hct (i + 1) hj']
  obtain ⟨rs', hrs'⟩ : ∃ rs' : Array Round, rs' = t.rounds.mapIdx fun j r => { r with bytes := some (F j) } :=
    ⟨_, rfl⟩
  have hsz' : rs'.size = t.rounds.size := by rw [hrs', Array.size_mapIdx]
  have hget : ∀ j < t.rounds.size, (rs'[j]!).n = (t.rounds[j]!).n ∧ (rs'[j]!).coins = (t.rounds[j]!).coins ∧
      (rs'[j]!).bytes = some (F j) := fun j hj => by
    rw [getElem!_pos rs' j (by omega), getElem!_pos t.rounds j hj]
    subst hrs'
    refine ⟨?_, ?_, ?_⟩ <;> simp [Array.getElem_mapIdx]
  have hfit' := fits_from (c := cs) (F := F) (rs := rs') (k₀ := k₀)
    (fun j hj hj' => by rw [hctS j hj (by omega), (hget j (by omega)).2.1])
    (fun j hj => by rw [(hget j (by omega)).2.1, (hget j (by omega)).1]; exact hwf j (by omega))
    (fun j _ hj => (hget j (by omega)).2.2)
  rw [hsz'] at hfit'
  have E := ligerito_frames st.merkle sh.sched _ p.cap p.ligerito _ hdim hlv (t := ⟨rs', k₀, ByteArray.empty⟩)
    (fun i hi => ⟨F i, (hget i (by change i < rs'.size at hi; omega)).2.2⟩) (hF rs' hfit')
  have hlen : t.rounds.size = k₀ + (chunks (ligFrames sh.sched (valOf sh d.w d.comb d.cons .T) p.cap
      p.ligerito)).1.length := E.k
  refine hF t.rounds (fits_from hctS hwf fun j hj hj' => ?_)
  have hR := E.ret (j - k₀) (by omega)
  dsimp only at hR
  have hp : (if j - k₀ = 0 then ByteArray.empty.data.toList else ([] : List UInt8)) = [] := by split <;> rfl
  rw [hp, List.nil_append, show k₀ + (j - k₀) = j by omega] at hR
  unfold Retains at hR
  rw [(hget j hj').2.2] at hR
  have ht := hfr (j - k₀) (by omega)
  rw [e, show k₀ + (j - k₀) = j by omega] at ht
  exact bytes_of_retains ht (Option.some.inj hR)

end FlockC3.HoloValues
