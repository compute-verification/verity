import Proofs.Flock.Claims.Recursion.LastSharedR
import Proofs.Flock.Claims.Recursion.LastBindsR

/-!
# The columns half of the deployed reader's last-message binding (`RecDeploy.RColsBind`)

Two decisions of `nextR rd` on the same rounds and coins share the setup, the link, the first coin and the round
messages (`nextR_shared`), and each verdict reads the proofs' files `filesAt` at the same shapes.

* **Different files collide** (`filesJ_tied`, `filesR_tied`): at a position where the files differ, both pass one rep on
  one session (`Flock.With.verifyRepWith_erase`), and `rep_bind_proof` gives one decoded proof or two colliding openings.
  One decoded proof is one file, decoding being canonical (`decodeProof_read`), so the openings collide among the two
  decisions' calls (`plainE_calls_tied`'s, at `nextR`, with no slack).
* **The same files open the same columns** (`colsR_of_files`): `colsR` reads the live value only through its link points
  and its rounds' coins, which custody pins (`streams_coins_eqJK`).

So `RColsBind` holds at SHA-512 with no premise (`RecDeploy.rColsBind_sha512`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **An accepting step of `nextR` decides.** -/
theorem actR_of_acc {H : Claims.Msg → Claims.Msg} {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} (h : ((nextR rd x as cs).eval H).1 = .acc) :
    ∃ s spec sts link c₁ mids f, actR rd x as cs = .decide s spec sts link c₁ mids f := by
  cases ha : actR rd x as cs with
  | rej => exact absurd h (by simp [nextR, ha, Claims.OC.eval])
  | coin c => exact absurd h (by simp [nextR, ha, Claims.OC.eval])
  | decide s spec sts link c₁ mids f => exact ⟨s, spec, sts, link, c₁, mids, f, rfl⟩

/-! ## Different files collide -/

/-- **Two verdicts at one setup and one custody's round messages, on different proof files, collide among their
calls**: each verdict accepts with SHA-512 answered only within its own log, and at a position where the files differ
both pass one rep on one session, where one decoded proof is one file (`decodeProof_read`), so two openings collide
(`plainE_calls_tied`'s second half, at a parsed value). -/
theorem filesJ_tied {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {fj fj' : FinalJ} {j j' : Json} {F F' : Array ByteArray} {L₁ L₂ : List (Claims.Msg × Claims.Msg)}
    (hV : verifyJWithMK id (hashesIn (L₁.map Prod.fst)) spec (tabsOf sts) j F = .ok ())
    (hV' : verifyJWithMK id (hashesIn (L₂.map Prod.fst)) spec (tabsOf sts) j' F' = .ok ())
    (hck : checksAtJ s spec x link c₁ mids fj j = .ok ()) (hck' : checksAtJ s spec x link c₁ mids fj' j' = .ok ())
    (hsu : s.setup = .ok (spec, sts)) (hpf : F ≠ F') : Claims.TiedColl sha512 (L₁ ++ L₂) := by
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checksAtJ_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checksAtJ_ok hck'
  have hsz := verifyJWithMK_proofs_size _ _ hV
  have hsz' := verifyJWithMK_proofs_size _ _ hV'
  obtain ⟨k, hk, hk', hne⟩ : ∃ (k : ℕ) (hk : k < F.size) (hk' : k < F'.size), F[k] ≠ F'[k] := by
    by_contra hcon
    push Not at hcon
    exact hpf (Array.ext (hsz.trans hsz'.symm) hcon)
  obtain ⟨sess, schs, hdec, htsz, hrep⟩ := verifyJWithMK_rep _ _ hV
  obtain ⟨sess', schs', hdec', -, hrep'⟩ := verifyJWithMK_rep _ _ hV'
  have hq : k < spec.repStreams.size := hsz ▸ hk
  have hi : k / 2 < (tabsOf sts).size := by
    have := repStreams_size spec
    omega
  obtain ⟨hsch, c, hr₁⟩ := hrep k hi
  obtain ⟨hsch', c', hr₂⟩ := hrep' k hi
  rw [hsch] at hsch'
  have hss : schs'[k / 2]! = schs[k / 2]! := (Option.some.inj hsch').symm
  rw [hss] at hr₂
  have hsts : k / 2 < sts.size := by simpa [tabsOf] using hi
  have hms0 : (tabsOf sts)[k / 2].merkle = MerkleScheme.hm96Sha512 ∨
      (tabsOf sts)[k / 2].merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rw [tabsOf_getElem hi]
    rcases checksAtJ_leafL hck hsu (k / 2) hsts with e | ⟨-, e⟩
    · exact Or.inl e
    · exact Or.inr e
  generalize hst : (tabsOf sts)[k / 2] = st at hsch hr₁ hr₂ hms0
  generalize hsc : schs[k / 2]! = sch at hsch hr₁ hr₂
  have hfold := tabsOf_fold sts st (hst ▸ Array.getElem_mem hi)
  have hms := hms0
  have named := named_of hms
  obtain ⟨y, y', hy, hy', hyy, hret⟩ := sess_rounds_eqJK _ _ _ _ hV hlv hV' hlv' hcu hcu' hdec hdec' hq
  simp only [repStreams_getElem spec hq] at hy hy'
  obtain ⟨hrB, hpts, -⟩ := decodeJWithK_live _ _ hdec hlv
  obtain ⟨hrB', hpts', -⟩ := decodeJWithK_live _ _ hdec' hlv'
  obtain ⟨hlB, -, hlP, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  obtain ⟨hlB', -, hlP', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
  have hP : sess.points = sess'.points := by rw [hpts, hpts', hlP, hlP']
  have hB : sess.rootB = sess'.rootB := by
    rw [hrB, hrB']
    exact Array.ext' (hlB.trans hlB'.symm)
  have hr₂s : verifyRepWith (hashesIn (L₂.map Prod.fst)) spec st sch sess (k / 2) (k % 2) F'[k]! = .ok c' := by
    rw [With.verifyRepWith_erase With.Sim.liftMap_V _ spec st sch _ hy hy' hyy hP (by rw [hB])]
    exact hr₂
  have hpure : ∀ {L : List Claims.Msg} {sess₀ : Session} {bs c₀ : ByteArray},
      verifyRepWith (hashesIn L) spec st sch sess₀ (k / 2) (k % 2) bs = .ok c₀ →
        verifyRep spec st sch sess₀ (k / 2) (k % 2) bs = .ok c₀ := by
    intro L sess₀ bs c₀ h
    rw [← With.verifyRepWith_std _ _ named]
    exact (With.verifyRepWith_sim sub (sub_hashes L) spec st sch sess₀ _ _ bs) c₀ h
  have hret' : ∀ stream : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[k / 2]! ++ toString "/rep" ++ toString (k % 2)) =
        some stream → AllRetained stream.rounds := by
    intro stream hs₀
    have e : stream = y := Option.some.inj (hs₀.symm.trans hy)
    rw [e]
    exact hret
  obtain ⟨stream, extras, p, p', hstr, hex, hp, hp', hor⟩ :=
    rep_bind_proof spec st sch sess (k / 2) (k % 2) _ _ c c' hsch hfold.1 hfold.2 hms (hpure hr₁) (hpure hr₂s) hret'
  have hb : F[k]! = F[k] := getElem!_pos F k hk
  have hb' : F'[k]! = F'[k] := getElem!_pos F' k hk'
  rcases hor with rfl | ⟨ℓ, hℓ, j₀, hj₀, hcoll⟩
  · exfalso
    apply hne
    rw [← hb, ← hb']
    exact (decodeProof_read hp).2.symm.trans (decodeProof_read hp').2
  · obtain ⟨s₁, e₁, q₁, hs₁, he₁, hq₁, hlog₁⟩ := rep_level_logs hsch hfold.1 hfold.2 hms hr₁
    obtain ⟨s₂, e₂, q₂, hs₂, he₂, hq₂, hlog₂⟩ := rep_level_logs hsch hfold.1 hfold.2 hms hr₂s
    have es₁ : s₁ = stream := Option.some.inj (hs₁.symm.trans hstr)
    have es₂ : s₂ = stream := Option.some.inj (hs₂.symm.trans hstr)
    have ee₁ : e₁ = extras := Except.ok.inj (he₁.symm.trans hex)
    have ee₂ : e₂ = extras := Except.ok.inj (he₂.symm.trans hex)
    subst s₁ s₂ e₁ e₂
    have eq₁ : q₁ = p := Except.ok.inj (hq₁.symm.trans hp)
    have eq₂ : q₂ = p' := Except.ok.inj (hq₂.symm.trans hp')
    subst q₁ q₂
    refine CollIn.tied hcoll (fun u hu => ?_) (fun u hu => ?_)
    · rw [List.map_append]
      exact List.mem_append_left _ (hlog₁ ℓ hℓ j₀ hj₀ u hu)
    · rw [List.map_append]
      exact List.mem_append_right _ (hlog₂ ℓ hℓ j₀ hj₀ u hu)

/-- **Two accepted decisions of `nextR` at one setup, first coin and round messages, on different proof files, collide
among their calls**, at SHA-512. -/
theorem filesR_tied {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as as' : List Claims.Msg}
    {cs cs' : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f f' : FinalP}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (ha' : actR rd x as' cs' = .decide s spec sts link c₁ mids f')
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) (hacc' : ((nextR rd x as' cs').eval sha512).1 = .acc)
    (hne : filesAt (tabsOf sts) (pointsOf spec c₁) f.ps ≠ filesAt (tabsOf sts) (pointsOf spec c₁) f'.ps) :
    Claims.TiedColl sha512 (((nextR rd x as cs).eval sha512).2 ++ ((nextR rd x as' cs').eval sha512).2) := by
  obtain ⟨-, hck, hsu, -, -⟩ := Claims.Bridge.PlainDraw.decideR_sha512 ha hacc
  obtain ⟨-, hck', -, -, -⟩ := Claims.Bridge.PlainDraw.decideR_sha512 ha' hacc'
  exact filesJ_tied (verdictR_within (decideR_runs ha hacc)) (verdictR_within (decideR_runs ha' hacc')) hck hck'
    hsu hne

/-! ## The same files open the same columns -/

/-- Two families of rounds with the same coins, stream by stream, have the same coins at every round of every
stream. -/
theorem streams_getD_coins {A B : Array (Array Flock.Round)}
    (h : A.toList.map (fun s => s.toList.map Round.coins) = B.toList.map (fun s => s.toList.map Round.coins))
    (k j : ℕ) :
    (A.getD k #[])[j]?.map (fun rd : Flock.Round => rd.coins) =
      (B.getD k #[])[j]?.map (fun rd : Flock.Round => rd.coins) := by
  have hsz : A.size = B.size := by simpa using congrArg List.length h
  unfold Array.getD
  by_cases hk : k < A.size
  · have hk' : k < B.size := hsz ▸ hk
    simp only [hk, hk', ↓reduceDIte]
    have hkA : k < (A.toList.map (fun s => s.toList.map Round.coins)).length := by simpa using hk
    have e := List.getElem_of_eq h hkA
    simp only [List.getElem_map, Array.getElem_toList] at e
    have e2 := congrArg (fun l => l[j]?) e
    simp only [List.getElem?_map, Array.getElem?_toList] at e2
    exact e2
  · have hk' : ¬ k < B.size := hsz ▸ hk
    simp only [hk, hk', ↓reduceDIte]

/-- `nextR`'s run of table `j` where the step decides, the table is set up and its extra claims read. -/
theorem runOfR_decide {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {lv : Live} {j : ℕ} {st : Stmt} {extras : Array Flock.Claim}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hlv : FlockDeploy.liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv) (hst : sts[j]? = some st)
    (hx : (Setup.ofCircuit st).extra lv.points = .ok extras) :
    runOfR rd x as cs j = some
      { m := (Setup.ofCircuit st).m, kLog := (Setup.ofCircuit st).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit st).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit st),
        reps := [(lv.streams.getD (2 * j) #[],
                  (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * j) ByteArray.empty),
                 (lv.streams.getD (2 * j + 1) #[],
                  (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * j + 1) ByteArray.empty)] } := by
  unfold runOfR
  simp [ha, hst, hlv, hx, Except.toOption]

theorem runOfR_none_st {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {j : ℕ}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f) (hst : sts[j]? = none) :
    runOfR rd x as cs j = none := by
  unfold runOfR
  simp [ha, hst]

theorem runOfR_none_x {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {lv : Live} {j : ℕ} {st : Stmt} {e : String}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hlv : FlockDeploy.liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv) (hst : sts[j]? = some st)
    (hx : (Setup.ofCircuit st).extra lv.points = .error e) :
    runOfR rd x as cs j = none := by
  unfold runOfR
  simp [ha, hst, hlv, hx, Except.toOption]

/-- **Two accepted decisions of `nextR` at one setup, first coin and round messages, on the same proof files, open the
same columns**: `colsR` reads the live value through its link points and its rounds' coins only, which custody pins. -/
theorem colsR_of_files {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as as' : List Claims.Msg}
    {cs cs' : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link link' : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f f' : FinalP}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (ha' : actR rd x as' cs' = .decide s spec sts link' c₁ mids f')
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) (hacc' : ((nextR rd x as' cs').eval sha512).1 = .acc)
    (hF : filesAt (tabsOf sts) (pointsOf spec c₁) f.ps = filesAt (tabsOf sts) (pointsOf spec c₁) f'.ps) :
    colsR rd x as cs = colsR rd x as' cs' := by
  obtain ⟨hv, hck, -, -, -⟩ := Claims.Bridge.PlainDraw.decideR_sha512 ha hacc
  obtain ⟨hv', hck', -, -, -⟩ := Claims.Bridge.PlainDraw.decideR_sha512 ha' hacc'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checksAtJ_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checksAtJ_ok hck'
  have hP : lv'.points = lv.points := by
    obtain ⟨-, -, hpts, -⟩ := Claims.Bridge.Custody.custody_ok hcu
    obtain ⟨-, -, hpts', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
    exact hpts'.trans hpts.symm
  have hS := streams_coins_eqJK id id With.std With.std hv hlv hv' hlv' hcu hcu'
  have key : ∀ (sh : Shape) (k : ℕ) (b : ByteArray),
      Claims.Flock.plainRepColsOf sh (lv'.streams.getD k #[]) b =
        Claims.Flock.plainRepColsOf sh (lv.streams.getD k #[]) b :=
    fun sh k b => plainRepColsOf_congr sh b (streams_getD_coins hS.symm k)
  funext j
  unfold colsR
  cases hst : sts[j]? with
  | none => rw [runOfR_none_st ha hst, runOfR_none_st ha' hst]
  | some st =>
    cases hx : (Setup.ofCircuit st).extra lv.points with
    | error e => rw [runOfR_none_x ha hlv hst hx, runOfR_none_x ha' hlv' hst (by rw [hP]; exact hx)]
    | ok extras =>
      rw [runOfR_decide ha hlv hst hx, runOfR_decide ha' hlv' hst (by rw [hP]; exact hx), ← hF]
      simp only [Claims.Flock.plainRunCols, Claims.Flock.ZkRun.plainShape, List.flatMap_cons, List.flatMap_nil,
        key]

end FlockDeploy.KS

namespace RecDeploy

open FlockDeploy (Coin FinalP nextR actR)

/-- **The deployed reader binds its last message's columns, at SHA-512**: two accepted last steps on the same rounds
and coins share the setup, the first coin and the round messages (`nextR_shared`); on the same proof files they open
the same columns (`colsR_of_files`), and on different ones two openings collide among their calls (`filesR_tied`). -/
theorem rColsBind_sha512 (rd : Claims.Msg → Option FinalP) : RColsBind FlockDeploy.sha512 rd := by
  intro x ms l l' cs hk _ hacc hacc' hne
  obtain ⟨s, spec, sts, link, c₁, mids, f, ha⟩ := FlockDeploy.KS.actR_of_acc hacc
  obtain ⟨s', spec', sts', link', c₁', mids', f', ha'⟩ := FlockDeploy.KS.actR_of_acc hacc'
  obtain ⟨e₁, e₂, e₃, e₄, e₅, e₆⟩ :=
    FlockDeploy.KS.nextR_shared (FlockDeploy.KS.ms_ne_nil_of_actR hk ha) ha ha'
  subst s' spec' sts' link' c₁' mids'
  by_cases hF : FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (FlockDeploy.pointsOf spec c₁) f.ps =
      FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (FlockDeploy.pointsOf spec c₁) f'.ps
  · exact absurd (FlockDeploy.KS.colsR_of_files ha ha' hacc hacc' hF) hne
  · obtain ⟨u, v, hcol, hu, hv⟩ := FlockDeploy.KS.filesR_tied ha ha' hacc hacc' hF
    exact ⟨u, hu, v, hv, hcol⟩

end RecDeploy
