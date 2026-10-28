import Proofs.Flock.Claims.KS.CapChunk
import Proofs.Flock.Claims.KS.PlainLevels
import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.Recursion.PlainK
import Proofs.Flock.Claims.Recursion.CallsBridgeE

/-!
# The cap an accepted plain rep absorbs, from the messages before its round

`CapChunk` splits a rep's frames at its `ℓ`-th recursive cap, in round `preLig + capRound k0 (ℓ + 1)` of its stream.
Here that round is a deployed message: at an accepted plain run, rep `r` of table 0 retained in round `j` of its stream
the bytes of its frames' `j`-th chunk (`rep_rounds`, at the oracle answering within the run's log and the table's named
Merkle scheme), and custody makes those bytes `P`'s message `r · R + j + 1`, `R` the table's `repRounds`
(`custody_ok`). So two accepted runs whose coins agree before the plain fork round have one message there
(`depMsg_congr`), and their rep-`r` proofs one flattened `ℓ`-th recursive cap (`capFr_eq_of_chunk`).

* `plain_acc_mids`: an accepted plain run decides on `P`'s messages along its coins.
* `plain_capChunk`: an accepted plain rep's frames, chunk by chunk, are `P`'s round messages.
* `plain_capPrefix`: two accepted plain runs sharing the coins before `forkRound` absorb one cap there.
* `plain_levelTie`: so two different rows they open at one level-`(ℓ + 1)` position tie a collision among their calls.
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Exec

theorem forall₂_getElem?_some {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ {k : ℕ} {a : α}, l₁[k]? = some a →
      ∃ b, l₂[k]? = some b ∧ R a b
  | _, _, .nil, _, _, h => by simp at h
  | _, _, .cons hab _, 0, _, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact ⟨_, rfl, hab⟩
  | _, _, .cons _ hl, k + 1, _, h => by
    simp only [List.getElem?_cons_succ] at h ⊢
    exact forall₂_getElem?_some hl h

/-- The first two arrays' entries in a flattening, by array and position. -/
theorem flat_two_getElem? {α : Type} (a : Array (Array α)) (h2 : 2 ≤ a.size) {r j : ℕ} (hr : r < 2)
    (hj : j < (a[r]'(by omega)).size) :
    (a.toList.flatMap (·.toList))[r * (a[0]'(by omega)).size + j]? = (a[r]'(by omega))[j]? := by
  rcases a with ⟨_ | ⟨a₀, _ | ⟨a₁, l⟩⟩⟩
  · simp at h2
  · simp at h2
  · rcases r with _ | _ | r
    · simp only [List.getElem_toArray, List.getElem_cons_zero] at hj ⊢
      simp [List.getElem?_append_left (show j < a₀.toList.length by simpa using hj)]
    · simp only [List.getElem_toArray, List.getElem_cons_succ, List.getElem_cons_zero] at hj ⊢
      simp [List.getElem?_append_right, List.getElem?_append_left (show j < a₁.toList.length by simpa using hj)]
    · omega

/-- **An accepted plain run decides on `P`'s messages along its coins**: it decides at the statement's setup, on its
last message read as a final message, with the custody checks passing and the shipped verdict accepting within its
verifier log, and its round messages are `P`'s messages `1` to `plainN x` after their tags, each with its coin. -/
theorem plain_acc_midsK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {c : Fin ((V false).k x) → Coin}
    (h : (Claims.run sha512 P (VK keep) x ω c).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P (VK keep) x ω c).seen = some f ∧ checks false s spec x link c₁ mids f = .ok () ∧
      verifyWithMK keep (hashesIn ((Claims.run sha512 P (VK keep) x ω c).vLog.map Prod.fst)) spec (tabsOf sts)
        f.record f.proofs = .ok () ∧
      mids = (List.range (plainN x)).map
        (fun k => ((depMsg sha512 P x ω (Claims.coinSeq c) (k + 1)).tail, Claims.coinSeq c (k + 1))) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accK_decide keep h
  obtain ⟨hv, hc⟩ := decisionK_runs keep (setupOf_some (act_decide hact)).2 hrun
  have hl := act_decide_final hact
  rw [List.getLast?_map, Option.bind_map] at hl
  have hL := plainN_add (act_decide hact)
  refine ⟨s, spec, sts, link, c₁, mids, f, act_decide hact, hl, hc, verdictK_within keep hv, ?_⟩
  have hlen := (accLast_VK keep x _ _ (by simp) (run_acc_next sha512 P (VK keep) x ω c h)).trans
    (congrFun (VK_k keep) x)
  have has := run_depK keep sha512 P x ω c
  rw [List.length_map] at hlen
  rw [hlen] at has
  rw [has, List.length_map, List.length_range, range_map_ends _ (by omega),
    range_map_ends (Claims.coinSeq (k := (VK keep).k x) c) (by omega)] at hact
  obtain ⟨a₁, rest, crest, h1, h2, h3⟩ := act_decide_mids hact
  injection h1 with _ h1
  injection h2 with _ h2
  subst h1 h2
  rw [h3, List.dropLast_concat, List.dropLast_concat, List.map_map, show (V false).k x - 2 = plainN x by omega,
    List.zip_map']
  rfl

/-- `plain_acc_midsK` at the deployed plain verifier (`keep := id`). -/
theorem plain_acc_mids {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {c : Fin ((V false).k x) → Coin}
    (h : (Claims.run sha512 P (V false) x ω c).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P (V false) x ω c).seen = some f ∧ checks false s spec x link c₁ mids f = .ok () ∧
      verdictM false (hashesIn ((Claims.run sha512 P (V false) x ω c).vLog.map Prod.fst)) spec (tabsOf sts) f.record
        f.proofs = .ok () ∧
      mids = (List.range (plainN x)).map
        (fun k => ((depMsg sha512 P x ω (Claims.coinSeq c) (k + 1)).tail, Claims.coinSeq c (k + 1))) :=
  plain_acc_midsK id (Ω := Ω) (P := P) (x := x) (ω := ω) (c := c) (h := h)

/-- **An accepted plain rep's frames are `P`'s round messages, chunk by chunk.** At an accepted plain run, rep `r < 2`
of table 0, its proof file decoded at the table's `fast100` schedule and `|regions| · points` extra claims, has frames
(at some batched value `T`) whose `j`-th chunk is, for every round `j` of the rep's `repRounds`, the bytes of `P`'s
message `r · repRounds + j + 1` after its tag: the verdict accepts the rep (`verifyWithMK_tables`), at the shipped
oracle (`sub`) and the table's named scheme, so its stream's rounds retained its chunks (`rep_rounds`), and the
stream's rounds are the live ones (`decodeWithK_live`), which custody makes the round messages (`custody_ok`). -/
theorem plain_capChunkK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {c : Fin ((V false).k x) → Coin}
    (h : (Claims.run sha512 P (VK keep) x ω c).out = .acc) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {sch : Flock.Schedule} {f : Final} {p : Proof} {r : ℕ} (hs : setupOf false x = some (s, spec, sts))
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hr : r < 2)
    (hf : lastFinal (Claims.run sha512 P (VK keep) x ω c).seen = some f)
    (hp : decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f.proofs[r]! = .ok p) :
    ∃ T : F128, ∀ j < repRounds false spec st,
      encs ((chunks (repFrames (Setup.ofCircuit st) sch (spec.domain spec.tables[0]! r).toUTF8
          (2 + st.regions.size * spec.points) p T)).1.getD j []) =
        (depMsg sha512 P x ω (Claims.coinSeq c) (r * repRounds false spec st + j + 1)).tail := by
  obtain ⟨s', spec', sts', link, c₁, mids, f', hs', hf', hc, hv, hmids⟩ := plain_acc_midsK keep h
  obtain ⟨rfl, rfl, rfl⟩ : s = s' ∧ spec = spec' ∧ sts = sts' := by simpa using hs.symm.trans hs'
  obtain rfl := Option.some.inj (hf.symm.trans hf')
  generalize (Claims.run sha512 P (VK keep) x ω c).vLog.map Prod.fst = L at hv
  obtain ⟨-, lv, hlv, hcus⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hsts0, hst0⟩ := Array.getElem?_eq_some_iff.1 hst
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, schs, hdec, hsz, -, hall0⟩ := verifyWithMK_tables keep _ hv
  obtain ⟨hs0, c₀, c₁, hsch', h₀, h₁⟩ := hall0 0 h0
  obtain ⟨cap, hrep⟩ : ∃ cap, verifyRepWith (hashesIn L) spec (tabsOf sts)[0] schs[0] sess 0 r f.proofs[r]! =
      .ok cap := by
    rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
    · exact ⟨c₀, h₀⟩
    · exact ⟨c₁, h₁⟩
  simp only [ht0] at hsch' hrep
  have e : schs[0] = sch := Option.some.inj (hsch'.symm.trans hsch)
  rw [e] at hrep
  have named : (Setup.ofCircuit st).merkle.Named := by
    show st.merkle.Named
    rcases checks_leaf hc (setupOf_some hs).2 0 hsts0 with e | ⟨-, e⟩ <;> rw [hst0] at e <;> rw [e]
    · exact With.named_of_find? (id := "hm96-sha512/v1") rfl
    · exact With.named_of_find? (id := "flock-leaf/sha512-unsalted") rfl
  have hrep' : verifyRep spec (Setup.ofCircuit st) sch sess 0 r f.proofs[r]! = .ok cap := by
    rw [← With.verifyRepWith_std _ _ named]
    exact (With.verifyRepWith_sim sub (sub_hashes L) spec _ sch sess 0 r _) cap hrep
  have h22 := exec_fast100_ge hsch
  obtain ⟨hdim, -⟩ := fast100_dims _ _ hsch
  have hlvs : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level _ _ hsch _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨stream, p', extras, T, hfind, hext, hp', hR⟩ :=
    rep_rounds spec (Setup.ofCircuit st) sch sess 0 r _ cap (by show 14 ≤ st.m; omega) hdim hlvs hrep'
  obtain ⟨-, hpts, -, hstreams⟩ := decodeWithK_live keep _ hdec hlv
  obtain ⟨hlsz, hall⟩ := rep_plainK keep _ hv hlv
  have hn : extras.size = st.regions.size * spec.points := by
    obtain ⟨sess', -, -, -, hpts', hpn', -⟩ := hall 0 (by omega)
    rw [extraClaims_size (st := st) hext, hpts, ← hpts', hpn']
  rw [hn] at hp' hR
  obtain rfl := Except.ok.inj (hp.symm.trans hp')
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hr
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + r) hk'
  rw [hrs] at hy
  obtain rfl : stream = y := Option.some.inj (hfind.symm.trans hy)
  obtain ⟨-, hsizes⟩ := streams_size_plainK keep _ hv hlv
  have hR0 : ∀ q (hq : q < lv.streams.size), q < 2 → lv.streams[q].size = repRounds false spec st := by
    intro q hq hq2
    rw [hsizes q hq, show q / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0]
  have hq2 : 2 ≤ lv.streams.size := by omega
  simp only [Nat.mul_zero, Nat.zero_add] at hfa
  rw [getElem!_pos lv.streams r (by omega)] at hfa
  have hsize : stream.rounds.size = repRounds false spec st := by
    have := hfa.length_eq
    rw [Array.length_toList, Array.length_toList, hR0 r _ hr] at this
    exact this
  have hfl := (Claims.Bridge.Custody.custody_ok hcus).2.2.2
  have hbytes : ∀ j < repRounds false spec st, (stream.rounds[j]!).bytes =
      some ⟨(depMsg sha512 P x ω (Claims.coinSeq c) (r * repRounds false spec st + j + 1)).tail.toArray⟩ := by
    intro j hj
    have hjr : j < (lv.streams[r]'(by omega)).size := by rw [hR0 r _ hr]; exact hj
    have e1 := flat_two_getElem? lv.streams hq2 hr hjr
    rw [hR0 0 (by omega) (by omega), Array.getElem?_eq_getElem hjr] at e1
    obtain ⟨am, hm, hram⟩ := forall₂_getElem?_some hfl e1
    have hk : r * repRounds false spec st + j < plainN x := by
      have := (List.getElem?_eq_some_iff.1 hm).1
      rw [hmids, List.length_map, List.length_range] at this
      exact this
    rw [hmids, List.getElem?_map, List.getElem?_range hk] at hm
    simp only [Option.map_some, Option.some.injEq] at hm
    subst hm
    have hfj := Claims.Bridge.Custody.forall₂_getElem hfa (show j < stream.rounds.size by omega)
    rw [getElem!_pos stream.rounds j (by omega), hfj.2.2, getElem!_pos _ j hjr]
    exact hram.1
  have hret : AllRetained stream.rounds := fun i hi => ⟨_, hbytes i (hsize ▸ hi)⟩
  obtain ⟨-, hc'⟩ := hR hret
  refine ⟨T, fun j hj => ?_⟩
  have e := hc' j (hsize ▸ hj)
  unfold Retains at e
  rw [hbytes j hj] at e
  simpa using e.symm

/-- `plain_capChunkK` at the deployed plain verifier (`keep := id`). -/
theorem plain_capChunk {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {c : Fin ((V false).k x) → Coin}
    (h : (Claims.run sha512 P (V false) x ω c).out = .acc) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {sch : Flock.Schedule} {f : Final} {p : Proof} {r : ℕ} (hs : setupOf false x = some (s, spec, sts))
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hr : r < 2)
    (hf : lastFinal (Claims.run sha512 P (V false) x ω c).seen = some f)
    (hp : decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f.proofs[r]! = .ok p) :
    ∃ T : F128, ∀ j < repRounds false spec st,
      encs ((chunks (repFrames (Setup.ofCircuit st) sch (spec.domain spec.tables[0]! r).toUTF8
          (2 + st.regions.size * spec.points) p T)).1.getD j []) =
        (depMsg sha512 P x ω (Claims.coinSeq c) (r * repRounds false spec st + j + 1)).tail :=
  plain_capChunkK id (Ω := Ω) (P := P) (x := x) (ω := ω) (c := c) (h := h) (s := s) (spec := spec) (sts := sts) (st := st) (sch := sch) (f := f) (p := p) (r := r) (hs := hs) (hst := hst) (hsch := hsch) (hr := hr) (hf := hf) (hp := hp)

/-- **Two accepted plain runs sharing the coins before the plain fork round absorb one cap there.** At two accepted
plain runs of `P` at one tape whose coins agree before `forkRound spec st k0 r ℓ`, rep `r < 2` of table 0's proof
files, decoded at the table's `fast100` schedule and `|regions| · points` extra claims, have one flattened `ℓ`-th
recursive cap, for each `ℓ + 1` below the schedule's levels: its round's chunk is `P`'s message there in both runs
(`plain_capChunk`), the same message (`depMsg_congr`), and two frame lists with one header list (`repFrames_hds`) split
at their `ℓ`-th caps after as many closing squeezes (`repFrames_capSplit`) whose rounds there have the same bytes split
at one cap (`capFr_eq_of_chunk`). -/
theorem plain_capPrefixK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c₁ c₂ : Fin ((V false).k x) → Coin} (h₁ : (Claims.run sha512 P (VK keep) x ω c₁).out = .acc)
    (h₂ : (Claims.run sha512 P (VK keep) x ω c₂).out = .acc) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ} (hs : setupOf false x = some (s, spec, sts))
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hr : r < 2) (hℓ : ℓ + 1 < sch.levels.size)
    (hc : ∀ i : Fin ((V false).k x), i.val < forkRound spec st sch.k0 r ℓ → c₁ i = c₂ i) {f₁ f₂ : Final}
    {p₁ p₂ : Proof} (hf₁ : lastFinal (Claims.run sha512 P (VK keep) x ω c₁).seen = some f₁)
    (hf₂ : lastFinal (Claims.run sha512 P (VK keep) x ω c₂).seen = some f₂)
    (hp₁ : decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f₁.proofs[r]! = .ok p₁)
    (hp₂ : decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f₂.proofs[r]! = .ok p₂) :
    p₁.ligerito.recursiveCaps[ℓ]!.foldl (· ++ ·) ByteArray.empty =
      p₂.ligerito.recursiveCaps[ℓ]!.foldl (· ++ ·) ByteArray.empty := by
  obtain ⟨T₁, hT₁⟩ := plain_capChunkK keep h₁ hs hst hsch hr hf₁ hp₁
  obtain ⟨T₂, hT₂⟩ := plain_capChunkK keep h₂ hs hst hsch hr hf₂ hp₂
  have h22 := exec_fast100_ge hsch
  obtain ⟨hdim, hsm⟩ := fast100_dims _ _ hsch
  have hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0 := fun i hi =>
    (fast100_level _ _ hsch _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)).q_ne_zero
  have hn0 : sch.n ≠ 0 := by rw [hdim.1, hsm]; omega
  have hm : 14 ≤ (Setup.ofCircuit st).m := by show 14 ≤ st.m; omega
  obtain ⟨a, b, s₁, hF, hs₁, ha⟩ := repFrames_capSplit (Setup.ofCircuit st) sch
    (spec.domain spec.tables[0]! r).toUTF8 (2 + st.regions.size * spec.points) p₁ T₁ hm (by omega) hn0 hq hℓ
  obtain ⟨a', b', s₂, hF', hs₂, ha'⟩ := repFrames_capSplit (Setup.ofCircuit st) sch
    (spec.domain spec.tables[0]! r).toUTF8 (2 + st.regions.size * spec.points) p₂ T₂ hm (by omega) hn0 hq hℓ
  have hcnt : a.countP closes = preLig spec st + capRound sch.k0 (ℓ + 1) := by
    rw [ha]
    unfold preLig
    rw [Nat.mul_comm st.regions.size]
    rfl
  have hlt : a.countP closes < repRounds false spec st := by
    rw [hcnt, repRounds_plain hsch]
    have := capRound_lt (k0 := sch.k0) (show 2 ≤ sch.levels.size by omega) (show ℓ + 1 ≤ sch.levels.size by omega)
    omega
  refine capFr_eq_of_chunk (repFrames_hds (FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ hp₁)
    (FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ hp₂) _ T₁ T₂) hF hF' hs₁ hs₂ (ha.trans ha'.symm) ?_
  rw [hT₁ _ hlt, hT₂ _ hlt]
  congr 1
  refine depMsg_congr sha512 P x ω fun i hi => ?_
  have hi' : i < forkRound spec st sch.k0 r ℓ := by
    unfold forkRound
    rw [hcnt] at hi
    omega
  by_cases hk : i < (V false).k x
  · simp only [Claims.coinSeq, hk, ↓reduceDIte]
    exact hc ⟨i, hk⟩ hi'
  · simp only [Claims.coinSeq, hk, ↓reduceDIte]

/-- `plain_capPrefixK` at the deployed plain verifier (`keep := id`). -/
theorem plain_capPrefix {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c₁ c₂ : Fin ((V false).k x) → Coin} (h₁ : (Claims.run sha512 P (V false) x ω c₁).out = .acc)
    (h₂ : (Claims.run sha512 P (V false) x ω c₂).out = .acc) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ} (hs : setupOf false x = some (s, spec, sts))
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hr : r < 2) (hℓ : ℓ + 1 < sch.levels.size)
    (hc : ∀ i : Fin ((V false).k x), i.val < forkRound spec st sch.k0 r ℓ → c₁ i = c₂ i) {f₁ f₂ : Final}
    {p₁ p₂ : Proof} (hf₁ : lastFinal (Claims.run sha512 P (V false) x ω c₁).seen = some f₁)
    (hf₂ : lastFinal (Claims.run sha512 P (V false) x ω c₂).seen = some f₂)
    (hp₁ : decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f₁.proofs[r]! = .ok p₁)
    (hp₂ : decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f₂.proofs[r]! = .ok p₂) :
    p₁.ligerito.recursiveCaps[ℓ]!.foldl (· ++ ·) ByteArray.empty =
      p₂.ligerito.recursiveCaps[ℓ]!.foldl (· ++ ·) ByteArray.empty :=
  plain_capPrefixK id (Ω := Ω) (P := P) (x := x) (ω := ω) (c₁ := c₁) (c₂ := c₂) (h₁ := h₁) (h₂ := h₂) (s := s) (spec := spec) (sts := sts) (st := st) (sch := sch) (r := r) (ℓ := ℓ) (hs := hs) (hst := hst) (hsch := hsch) (hr := hr) (hℓ := hℓ) (hc := hc) (f₁ := f₁) (f₂ := f₂) (p₁ := p₁) (p₂ := p₂) (hf₁ := hf₁) (hf₂ := hf₂) (hp₁ := hp₁) (hp₂ := hp₂)

/-- **Two accepted plain runs sharing the coins before the plain fork round, opening one later-level position to two
rows, tie a collision among their verifier calls**: each row is an `Opening` of the position in its run's log against
the run's rep-`r` flattened `ℓ`-th recursive cap (`plain_levelOpening` at level `ℓ + 1`), the two caps are one
(`plain_capPrefix`), and two openings of one position against one root to different rows tie a collision
(`tiedColl_of_openings`). -/
theorem plain_levelTieK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c₁ c₂ : Fin ((V false).k x) → Coin} (h₁ : (Claims.run sha512 P (VK keep) x ω c₁).out = .acc)
    (h₂ : (Claims.run sha512 P (VK keep) x ω c₂).out = .acc) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ} (hs : setupOf false x = some (s, spec, sts))
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hr : r < 2) (hℓ : ℓ + 1 < sch.levels.size)
    (hc : ∀ i : Fin ((V false).k x), i.val < forkRound spec st sch.k0 r ℓ → c₁ i = c₂ i) {q : ℕ}
    {a b : List FlockLevel3.GF128}
    (ha : (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c₁).seen r (ℓ + 1))
    (hb : (q, b) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c₂).seen r (ℓ + 1)) (hab : a ≠ b) :
    Claims.TiedColl sha512
      ((Claims.run sha512 P (VK keep) x ω c₁).vLog ++ (Claims.run sha512 P (VK keep) x ω c₂).vLog) := by
  obtain ⟨s₁, spec₁, sts₁, f₁, st₁, sch₁, p₁, row₁, hs₁, hf₁, hst₁, hsch₁, hp₁, hra, -, -, hop₁⟩ :=
    plain_levelOpeningK keep h₁ ha
  obtain ⟨s₂, spec₂, sts₂, f₂, st₂, sch₂, p₂, row₂, hs₂, hf₂, hst₂, hsch₂, hp₂, hrb, -, -, hop₂⟩ :=
    plain_levelOpeningK keep h₂ hb
  obtain ⟨rfl, e₁⟩ := Prod.mk.inj (Option.some.inj (hs₁.symm.trans hs))
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj e₁
  obtain ⟨rfl, e₂⟩ := Prod.mk.inj (Option.some.inj (hs₂.symm.trans hs))
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj e₂
  obtain rfl := Option.some.inj (hst₁.symm.trans hst)
  obtain rfl := Option.some.inj (hst₂.symm.trans hst)
  obtain rfl := Option.some.inj (hsch₁.symm.trans hsch)
  obtain rfl := Option.some.inj (hsch₂.symm.trans hsch)
  have hcap := plain_capPrefixK keep h₁ h₂ hs hst hsch hr hℓ hc hf₁ hf₂ hp₁ hp₂
  rw [Nat.add_sub_cancel] at hop₁ hop₂
  rw [hcap] at hop₁
  exact tiedColl_of_openings hop₁ hop₂ fun he => hab (by rw [hra, hrb, he])

/-- `plain_levelTieK` at the deployed plain verifier (`keep := id`). -/
theorem plain_levelTie {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c₁ c₂ : Fin ((V false).k x) → Coin} (h₁ : (Claims.run sha512 P (V false) x ω c₁).out = .acc)
    (h₂ : (Claims.run sha512 P (V false) x ω c₂).out = .acc) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ} (hs : setupOf false x = some (s, spec, sts))
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hr : r < 2) (hℓ : ℓ + 1 < sch.levels.size)
    (hc : ∀ i : Fin ((V false).k x), i.val < forkRound spec st sch.k0 r ℓ → c₁ i = c₂ i) {q : ℕ}
    {a b : List FlockLevel3.GF128}
    (ha : (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω c₁).seen r (ℓ + 1))
    (hb : (q, b) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω c₂).seen r (ℓ + 1)) (hab : a ≠ b) :
    Claims.TiedColl sha512
      ((Claims.run sha512 P (V false) x ω c₁).vLog ++ (Claims.run sha512 P (V false) x ω c₂).vLog) :=
  plain_levelTieK id (Ω := Ω) (P := P) (x := x) (ω := ω) (c₁ := c₁) (c₂ := c₂) (h₁ := h₁) (h₂ := h₂) (s := s) (spec := spec) (sts := sts) (st := st) (sch := sch) (r := r) (ℓ := ℓ) (hs := hs) (hst := hst) (hsch := hsch) (hr := hr) (hℓ := hℓ) (hc := hc) (q := q) (a := a) (b := b) (ha := ha) (hb := hb) (hab := hab)

end FlockDeploy.KS
