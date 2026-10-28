import Proofs.Flock.Claims.Recursion.VerdictDecision
import Proofs.Flock.Claims.Recursion.VerdictFramedJson
import Proofs.Flock.Claims.Recursion.RepSqCounts
import Proofs.Flock.Claims.Recursion.CallsAgree

/-!
# Row B's `framed` at a decision

`DecAt.runs_assembled` takes `hframed`: rep `r` of table `i`'s rounds in the decoded session hold the rep's frames at
the batched value `T` its `w` reads. It holds at every decision whose rep streams have distinct names (`DecAt.framed`),
which a dense inner statement's do (`DecAt.names_nodup_dense`):

* **`w`'s `T` is the reader's** (`DecAt.valOf_T`): `w` holds the frames' values at the `T` the decided proof fits its
  opened rounds at (`DecAt.ctx_rows_fit`), and `T` is one of them (`T_mem`).
* **The decoded streams are the reader's** (`decoded_streams`): the decoder reads stream `k` of the verdict's record at
  the `k`-th rep stream's name, with the reader's rounds' counts and messages (`arr_streams_jAt`).
* **The reader's rounds are the frames** (`DecAt.framed_ctx`): round `j` of rep `k` is the `j`-th message of the rep's
  block, the framed bytes of its `j`-th chunk (`decR_framesR`, the `T` unique by `fitUnique`), at its schedule's count,
  which is the chunk's closing squeeze's (`sqCount_repFrames`).
-/

namespace FlockC3.Reps

open Flock.RecAlgebra (Key wNames ligRounds)

/-- **Ligerito's batched value is one of w's names**: its first round. -/
theorem T_mem (sh : Flock.Shape) : Key.T ∈ wNames sh := by
  apply mem_wNames_of_lig
  rw [ligRounds_eq]
  have h0 : InRound (lig0 sh.sched).2.2 .T := by
    obtain ⟨a1, a2, a3⟩ := ligF1_range (sh.sched.k0 + 1) 0 #[#[Key.T], #[Key.ood 0]]
    simp only [lig0]
    generalize List.foldl ligF1 (0, #[#[Key.T], #[Key.ood 0]]) (List.range' 0 (sh.sched.k0 + 1)) = A at a1 a2 a3 ⊢
    obtain ⟨b1, b2, b3, b4, b5⟩ := ligF2_range 2 A.fst 1 A.snd
    generalize List.foldl ligF2 (A.fst, 1, A.snd) (List.range' 0 2) = B at b1 b2 b3 b4 b5 ⊢
    exact (b3 _ (a2 _ ⟨#[Key.T], by simp, by simp⟩)).push
  generalize sh.sched.levels.size - 1 = R
  rcases R with _ | n
  · simp only [ligLevels, List.range'_zero, List.forIn_nil]
    exact h0
  · exact (ligLevels_has (R := n + 1) (P := fun _ => False) ligMid_has
      (fun s => ⟨(ligLast_has sh.sched.ext s).1, (ligLast_has sh.sched.ext s).2.1, fun _ h => h.elim⟩) n 0
      (lig0 sh.sched) (by omega)).1 _ h0

end FlockC3.Reps

namespace FlockDeploy

open Flock
open Claims.Bridge.Custody (RoundFields)

theorem getD_map_of_lt {α β : Type _} (f : α → β) {l : List α} {j : ℕ} (hj : j < l.length) (d : β) (d' : α) :
    (l.map f).getD j d = f (l.getD j d') := by
  simp [List.getD_eq_getElem?_getD, hj]

theorem recStreams_size (spec : SessionSpec) (sts : Array Stmt) (ms : List Claims.Msg) (cs : List Coin) :
    (recStreams spec sts ms cs).size = 2 * sts.size := by
  simp [recStreams]

variable {spec : SessionSpec} {sts : Array Stmt} {ms : List Claims.Msg} {cs : List Coin} {k : ℕ}

/-- The reader's rep `k` has one round per message of its block. -/
theorem recStreams_size_k (hk : k < (recStreams spec sts ms cs).size) :
    (recStreams spec sts ms cs)[k].size = ((cutLens (repLens spec sts) ms).getD k []).length := by
  simp [recStreams]

/-- The reader's round `j` of rep `k` counts its schedule's coins. -/
theorem recStreams_n (hk : k < (recStreams spec sts ms cs).size) {j : ℕ}
    (hj : j < (recStreams spec sts ms cs)[k].size) :
    (recStreams spec sts ms cs)[k][j].n = ((RecDeploy.repSchedAt spec sts[k / 2]!).getD j default).1 := by
  simp [recStreams]

/-- The reader's round `j` of rep `k` retains the `j`-th message of the rep's block. -/
theorem recStreams_bytes (hk : k < (recStreams spec sts ms cs).size) {j : ℕ}
    (hj : j < (recStreams spec sts ms cs)[k].size) :
    (recStreams spec sts ms cs)[k][j].bytes =
      some ⟨(((cutLens (repLens spec sts) ms).getD k []).getD j []).toArray⟩ := by
  have hj' : j < ((cutLens (repLens spec sts) ms).getD k []).length := by
    rw [← recStreams_size_k hk]
    exact hj
  simp only [recStreams, List.getElem_toArray, List.getElem_map, List.getElem_range, List.getElem_mapIdx]
  rw [List.getD_eq_getElem?_getD (l := (cutLens (repLens spec sts) ms).getD k []), List.getElem?_eq_getElem hj',
    Option.getD_some]

/-- **The stream at a name is the one at its index**, where the names are distinct. -/
theorem find_name_arr {a : Array StreamRec} {name : String} {k : ℕ} (hk : k < a.size) (hn : a[k].name = name)
    (hnd : (a.toList.map (·.name)).Nodup) : a.find? (·.name == name) = some a[k] := by
  rcases a with ⟨l⟩
  rw [List.find?_toArray]
  exact find_name hk hn hnd

/-- **The decoded streams are the reader's**: where the verdict's record is the reader's with its coins and digests
substituted, the decoder reads stream `k` at the `k`-th rep stream's name, and its rounds at the reader's counts and
messages. -/
theorem decoded_streams {hs : Hashes Flock.V} {H : Claims.Msg → Claims.Msg} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {link : List ByteArray × List ByteArray} {sess : Session}
    (hj : f.j = recJ spec sts link c₁ ms cs f.ps)
    (h : Record.decodeJWith hs spec (jAt H spec sts c₁ mids f) = .ok sess) :
    sess.streams.size = min spec.repStreams.size (recStreams spec sts ms cs).size ∧
    ∀ k (hk : k < sess.streams.size) (h₁ : k < spec.repStreams.size) (h₂ : k < (recStreams spec sts ms cs).size),
      sess.streams[k].name = spec.repStreams[k].2.2 ∧
      sess.streams[k].rounds.size = (recStreams spec sts ms cs)[k].size ∧
      ∀ j (hj : j < sess.streams[k].rounds.size) (hj' : j < (recStreams spec sts ms cs)[k].size),
        sess.streams[k].rounds[j].n = (recStreams spec sts ms cs)[k][j].n ∧
        sess.streams[k].rounds[j].bytes = some ((recStreams spec sts ms cs)[k][j].bytes.getD ByteArray.empty) := by
  rw [Record.decodeJWith_eq_decodeJWithK] at h
  obtain ⟨l, recs, -, hrecs, -, -, hF⟩ := KS.decodeJWithK_ok id hs h
  obtain ⟨recs₀, hr0, hK⟩ := arr_streams_jAt hrecs
  rw [hj, recJ, arr_recordObj_streams] at hr0
  injection hr0 with hr0
  subst hr0
  have hl1 := hF.length_eq
  have hl0 := hK.length_eq
  simp only [Array.length_toList, Array.size_map, Array.size_zip] at hl0 hl1
  refine ⟨by omega, fun k hk h₁ h₂ => ?_⟩
  have hkr : k < recs.size := by omega
  have hKk := (forall₂_getElem.1 hK).2 k
    (by simp only [Array.length_toList, Array.size_map, Array.size_zip]; omega) (by simpa using hkr)
  have hFk := (forall₂_getElem.1 hF).2 k (by simpa using hkr) (by simpa using hk)
  simp only [Array.getElem_toList, Array.getElem_map, Array.getElem_zip] at hKk hFk
  obtain ⟨hname, rs, hrs, hR⟩ := hFk
  have hstr : Record.str recs[k] "stream" = .ok (spec.repStreams[k]).2.2 := by
    unfold Record.str Record.fld
    rw [hKk.stream, streamObj_stream]
    rfl
  rw [hstr] at hname
  injection hname with hname
  obtain ⟨rs', hrs', hR'⟩ := hKk.rounds _ (arr_streamObj_rounds _ _ _ _)
  rw [hrs] at hrs'
  injection hrs' with hrs'
  subst hrs'
  dsimp only at hR'
  have hm1 := hR.length_eq
  have hm0 := hR'.length_eq
  simp only [Array.length_toList, Array.size_map, Array.size_range] at hm0 hm1
  refine ⟨hname.symm, by omega, fun j hj hj' => ?_⟩
  have hjr : j < rs.size := by omega
  have hKj := (forall₂_getElem.1 hR').2 j
    (by simp only [Array.length_toList, Array.size_map, Array.size_range]; exact hj') (by simpa using hjr)
  have hFj := (forall₂_getElem.1 hR).2 j (by simpa using hjr) (by simpa using hj)
  simp only [Array.getElem_toList, Array.getElem_map, Array.getElem_range] at hKj hFj
  rw [getElem!_pos _ j hj'] at hKj
  exact roundFields_roundObj hKj hFj

end FlockDeploy

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf repCtx repLens cutLens chunksOf framesOf FitsRep recJ recStreams SchedOf)
open FlockSoundness.Discharge.Exec (repFrames chunks)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockVBridge.Family (FramedAt sqCount)
open Flock.RecAlgebra (valOf accNames Key)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128)

/-- **A dense decision's rep streams have distinct names**: its statement sets up with no public inputs. -/
theorem names_nodup_dense {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) :
    (D.spec.repStreams.toList.map (·.2.2)).Nodup := by
  obtain ⟨s, spec, st, sch, hset, -, hpi, -⟩ := dense_inner hst hf
  rw [← hx] at hset
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hset] at hv
  cases hv
  obtain ⟨-, hs⟩ := FlockDeploy.setupOf_some hset
  have hds : D.d.spec = spec := congrArg (·.spec) hd
  rw [D.setup_eq.1, hds]
  exact FlockDeploy.KS.repNames_nodup (FlockDeploy.KS.tables_nodup hs hpi)

/-- **The decided proof fits its opened rounds at `T`**, rep by rep. -/
theorem ctx_rows_fit {k : ℕ} (hk : k < D.d.f.ps.size) : FitsRep (D.ctx k) (D.rows k) D.d.f.ps[k]! (D.T k) := by
  have hex : ∃ T, FitsRep (D.ctx k) (D.rows k) D.d.f.ps[k]! T := by
    obtain ⟨r₀, rss, h1, h2, -, hf⟩ := FlockDeploy.decR_fits D.dec
    obtain ⟨T, hT⟩ := hf k hk
    refine ⟨T, ?_⟩
    have e1 : D.r₀ = r₀ := by simp only [DecAt.r₀, h1, Option.getD_some]
    have e2 : D.rss = rss := by simp only [DecAt.rss, h2, Option.getD_some]
    rw [DecAt.ctx, DecAt.rows, e1, e2, getElem!_pos D.d.f.ps k hk]
    exact hT
  unfold DecAt.T
  rw [dif_pos hex]
  exact Classical.choose_spec hex

variable {i : ℕ} (r : Fin 2)

theorem files_st (hi : i < D.sts.size) : (D.files cls i).st = Setup.ofCircuit D.sts[i] := by
  show (tabsOf D.sts)[i]! = _
  rw [getElem!_pos (tabsOf D.sts) i (by simpa [tabsOf] using hi)]
  exact Array.getElem_map ..

/-- **`w`'s `T` is the one the decided proof fits its opened rounds at.** -/
theorem valOf_T (hG : D.RowG cls acc i) (hi : i < D.sts.size) :
    valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T =
      D.T (2 * i + r) := by
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← D.points]
    exact hG.extra
  obtain ⟨e, hne, hps⟩ := (D.ctx_rows_fit (D.rep_lt hi r)).sizes
  have hn := D.ctx_n cls acc i r hG hext
  have he : e = (D.files cls i).extras.size := by
    have : 2 + e = 2 + (D.files cls i).extras.size := hne.symm.trans hn
    omega
  have hst : (D.ctx (2 * i + r)).st = (D.files cls i).st := D.ctx_st cls acc i r hG
  have hsch : (D.ctx (2 * i + r)).sch = (D.files cls i).sch := D.ctx_sch cls acc i r hG
  have hp : ProofSizes (D.sh cls i) ((D.files cls i).proof r) := by
    rw [hst, hsch, he] at hps
    exact hps
  have hw : FlockDecodeTies.WHolds (D.sh cls i) (D.rep cls acc i r).w ((D.files cls i).proof r) (D.T (2 * i + r)) :=
    FlockDecodeTies.wHolds_of_frames (st := (D.ctx (2 * i + r)).st) (sch := (D.ctx (2 * i + r)).sch)
      (n := (D.ctx (2 * i + r)).n) (dom := (D.ctx (2 * i + r)).dom)
      (by rw [D.ctx_st cls acc i r hG] <;> rfl) (by rw [D.ctx_st cls acc i r hG] <;> exact hG.foldKLog)
      hn (D.ctx_sch cls acc i r hG) rfl hp (D.w_prefix (2 * i + r) (acc (2 * i + r)))
  exact FlockDecodeTies.valOf_keyVal hw (FlockC3.Reps.T_mem _) (by simp [accNames]) (by simp) (by simp) (by simp)

/-- **Rep `2 i + r`'s decoded rounds hold its frames at the `T` its proof fits at**, in the reader's context. -/
theorem framed_ctx (hG : D.RowG cls acc i) (hi : i < D.sts.size)
    (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) :
    FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.ctx (2 * i + r)).st (D.ctx (2 * i + r)).sch
      (D.ctx (2 * i + r)).dom (D.ctx (2 * i + r)).n D.d.f.ps[2 * i + r]! (D.T (2 * i + r))) := by
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hti : i < D.spec.tables.size := D.lt_tables hi
  have hk : 2 * i + r < D.d.f.ps.size := D.rep_lt hi r
  have hk2 : 2 * i + r < 2 * D.d.sts.size := by rw [← hsts]; omega
  have h2 : (2 * i + (r : ℕ)) / 2 = i := rep_div_two i r
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← D.points]
    exact hG.extra
  have hst : (D.ctx (2 * i + r)).st = (tabsOf D.sts)[i]! := D.ctx_st cls acc i r hG
  have hn : (D.ctx (2 * i + r)).n = 2 + (D.files cls i).extras.size := D.ctx_n cls acc i r hG hext
  -- the schedule: rep `2 i + r`'s rounds count its frames' squeezes
  have hsched : SchedOf (D.ctx (2 * i + r)) := by
    have hft : fast100 D.sts[i].m = some (D.files cls i).sch := by
      have h0 := hG.fast
      rw [D.files_st cls hi] at h0
      exact h0
    have hex : D.sts[i].extraClaims (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
      have h0 := hext
      rw [show (tabsOf D.sts)[i]! = (D.files cls i).st from rfl, D.files_st cls hi] at h0
      exact h0
    unfold DecAt.ctx
    refine FlockDeploy.schedOf_repCtx (extras := (D.files cls i).extras) (by rw [h2, ← hsts]; exact hi) ?_ ?_
    · simp only [h2, ← hsts]
      rw [hft]
      rfl
    · simp only [h2, ← hsts, ← hsp]
      exact hex
  obtain ⟨hfa, -, rsch, hrr, hsc⟩ := hsched
  rw [hn] at hrr
  -- the record's rep blocks are the frames at `T`
  obtain ⟨r₀, rss, ms', h1, h1', hj, hfr⟩ := FlockDeploy.decR_framesR D.dec
  have e1 : D.r₀ = r₀ := by simp only [DecAt.r₀, h1, Option.getD_some]
  have e2 : D.rss = rss := by simp only [DecAt.rss, h1', Option.getD_some]
  obtain ⟨T', hT', hblk⟩ := hfr _ hk2
  have hT'' : FitsRep (D.ctx (2 * i + r)) (D.rows (2 * i + r)) D.d.f.ps[2 * i + r]! T' := by
    rw [DecAt.ctx, DecAt.rows, e1, e2]
    exact hT'
  have hkl : (D.ctx (2 * i + r)).st.fold.kLog = (D.ctx (2 * i + r)).st.kLog := by
    rw [hst]
    exact hG.foldKLog
  have hTT := (FlockDeploy.fitUnique _ _ _ _ _ _ hkl (D.ctx_rows_fit hk) hT'' D.d.f.ps[2 * i + r]!).2
  have hblock : (cutLens (repLens D.d.spec D.d.sts) ms').getD (2 * i + r) [] =
      framesOf (D.ctx (2 * i + r)) D.d.f.ps[2 * i + r]! (D.T (2 * i + r)) := by
    rw [hblk, ← hTT, ← e1]
    rfl
  -- the decoded streams are the reader's
  have hdec := D.decode
  rw [hsp, hsts] at hdec
  obtain ⟨hL, hS⟩ := FlockDeploy.decoded_streams hj hdec
  have hR1 : 2 * i + r < D.d.spec.repStreams.size := by
    rw [FlockDeploy.repStreams_size, ← hsp, D.tables_size]
    omega
  have hR2 : 2 * i + r < (recStreams D.d.spec D.d.sts ms' cs.tail.dropLast).size := by
    rw [FlockDeploy.recStreams_size]
    exact hk2
  have hkL : 2 * i + r < D.sess.streams.size := by rw [hL]; omega
  have hnd' : (D.d.spec.repStreams.toList.map (·.2.2)).Nodup := by
    rw [← hsp]
    exact hnd
  have hndL : (D.sess.streams.toList.map (·.name)).Nodup := by
    have e : D.sess.streams.toList.map (·.name) =
        (D.d.spec.repStreams.toList.map (·.2.2)).take D.sess.streams.size := by
      refine List.ext_getElem (by simp only [List.length_map, Array.length_toList, List.length_take]; omega)
        fun j h₁ h₂ => ?_
      simp only [List.getElem_map, List.getElem_take, Array.getElem_toList]
      have hj : j < D.sess.streams.size := by simpa using h₁
      exact (hS j hj (by omega) (by omega)).1
    rw [e]
    exact hnd'.sublist (List.take_sublist _ _)
  obtain ⟨hlt, hrs⟩ := Claims.Bridge.Custody.repStreams_getElem D.d.spec (i := i) (r := r)
    (by rw [← hsp]; exact hti) r.isLt
  have hname : D.sess.streams[2 * i + r].name = toString D.spec.tables[i]! ++ toString "/rep" ++ toString r :=
    (hS _ hkL hR1 hR2).1.trans (by rw [hrs, hsp]; try rfl)
  have hs : (D.files cls i).stream r = D.sess.streams[2 * i + r] := by
    have e : (D.files cls i).stream r = (D.sess.streams.find? fun s =>
        s.name == toString D.spec.tables[i]! ++ toString "/rep" ++ toString r).getD default := rfl
    rw [e, FlockDeploy.find_name_arr hkL hname hndL]
    try rfl
  rw [hs]
  -- rep `2 i + r`'s rounds
  obtain ⟨-, hsz, hrd⟩ := hS _ hkL hR1 hR2
  have hlen : ((cutLens (repLens D.d.spec D.d.sts) ms').getD (2 * i + r) []).length =
      (chunksOf (D.ctx (2 * i + r)) D.d.f.ps[2 * i + r]! (D.T (2 * i + r))).length := by
    rw [hblock, framesOf, List.length_map]
  have hjs : ∀ j, j < (chunksOf (D.ctx (2 * i + r)) D.d.f.ps[2 * i + r]! (D.T (2 * i + r))).length →
      j < ((cutLens (repLens D.d.spec D.d.sts) ms').getD (2 * i + r) []).length ∧
      j < (recStreams D.d.spec D.d.sts ms' cs.tail.dropLast)[2 * i + r].size ∧
      j < D.sess.streams[2 * i + r].rounds.size := by
    intro j hj
    have hjb : j < ((cutLens (repLens D.d.spec D.d.sts) ms').getD (2 * i + r) []).length := by
      rw [hlen]
      exact hj
    have hjR : j < (recStreams D.d.spec D.d.sts ms' cs.tail.dropLast)[2 * i + r].size := by
      rw [FlockDeploy.recStreams_size_k]
      exact hjb
    exact ⟨hjb, hjR, by rw [hsz]; exact hjR⟩
  refine framedAt_of_rounds (fun j hj => ?_) (fun j hj => ?_)
  · obtain ⟨hjb, hjR, hjS⟩ := hjs j hj
    rw [getElem!_pos _ j hjS, (hrd j hjS hjR).2, FlockDeploy.recStreams_bytes hR2 hjR, Option.getD_some, hblock,
      framesOf, FlockDeploy.getD_map_of_lt _ (by rw [← hlen]; exact hjb) _ []]
    try rfl
  · obtain ⟨hjb, hjR, hjS⟩ := hjs j hj
    rw [getElem!_pos _ j hjS, (hrd j hjS hjR).1, FlockDeploy.recStreams_n hR2 hjR]
    have hsq := FlockDeploy.sqCount_repFrames (D.ctx (2 * i + r)).dom (D.files cls i).extras.size
      D.d.f.ps[2 * i + r]! (D.T (2 * i + r)) hfa hrr (j := j) (by rw [← hn]; exact hj)
    rw [← hn] at hsq
    rw [hsq, ← hsc]
    try rfl

/-- **Row B's `framed`**: at a decision whose rep streams have distinct names, rep `r` of table `i`'s rounds in the
decoded session hold the rep's frames at `w`'s `T`. -/
theorem framed (hG : D.RowG cls acc i) (hi : i < D.sts.size) (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) :
    FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st (D.files cls i).sch
      ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T)) := by
  obtain ⟨hsp, -⟩ := D.setup_eq
  have h2 : (2 * i + (r : ℕ)) / 2 = i := rep_div_two i r
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← D.points]
    exact hG.extra
  have hst : (D.ctx (2 * i + r)).st = (D.files cls i).st := D.ctx_st cls acc i r hG
  have hsch : (D.ctx (2 * i + r)).sch = (D.files cls i).sch := D.ctx_sch cls acc i r hG
  have hn : (D.ctx (2 * i + r)).n = 2 + (D.files cls i).extras.size := D.ctx_n cls acc i r hG hext
  have hdom : (D.ctx (2 * i + r)).dom =
      ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8 := by
    show (D.d.spec.domain D.d.spec.tables[(2 * i + r) / 2]! ((2 * i + r) % 2)).toUTF8 =
      (D.spec.domain D.spec.tables[i]! r).toUTF8
    rw [h2, hsp, show (2 * i + (r : ℕ)) % 2 = r by omega]
  rw [D.valOf_T cls acc r hG hi, ← hst, ← hsch, ← hdom, ← hn]
  exact D.framed_ctx cls acc r hG hi hnd

/-- **Row B's `framed` at a dense inner statement** (`TargetB`): rep `r` of table `i`'s rounds in the decoded session
hold the rep's frames at `w`'s `T`, at every decision and every `acc`. -/
theorem framed_dense {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (hi : i < D.sts.size) :
    FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st (D.files cls i).sch
      ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T)) :=
  D.framed cls acc r (D.rowG cls acc hst hf hx hi) hi (D.names_nodup_dense hst hf hx)

end DecAt

end RecDeploy.Rows
