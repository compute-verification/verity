import Proofs.Flock.VBridge.Family.Join

/-!
# `TranscriptIs` after the runs

`TableSrc.TranscriptIs` asks that the inner transcript V*'s layers hold be the table files' steps at some link points,
each rep's tape a tape of its spine (`RepTape`). A rep's tape is only known to be one once the rep runs
(`reps_of_runs`: an accepting rep plays its messages on its coins), so `TranscriptIs` is derived after `Runs`:

* `TableSrc.wovenAt`: the table files at link points `P` with each rep's tape `repSteps` of its proof file and stream,
  so `reps_of_runs` needs no hypothesis on the tapes;
* `TableSrc.transcriptIs_of_runs`: from the runs, the round messages (`hmsg`, round by round) and the coins (`hcoin`),
  the transcript's steps are the woven files' (`schedSteps_zip`), at the link points the transcript's first coin
  carries (`ptsOf_head`);
* `coinSteps_eq_coinsOf`: `hcoin` wherever the inner protocol's coin types are V*'s (`CoinTys`; `vCT` is such a
  schedule).

At a family, `HFilesR` is `hfiles` after the runs, and `hFilesR_of` gives it from `MsgsAre` (`hmsg` at every accepted
outer proof) and `HLeaf`. `decOf` is the decoder `wmsgOf` reads: round `i`'s message at stage `i`'s read wires is
the table's round `i` at any accepted outer proof that reads them, so `hmsg` is a theorem (`decOf_spec`) wherever the
table's round `i` is read from stage `i`'s commit strings (`MsgsRead`).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Discharge.PlaysCoin FlockSoundness.Refine
open FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

local notation "𝔽" => FlockLevel3.GF128

/-! ## The woven table files -/

/-- **The table files at link points `P`, each rep's tape woven** from its proof file's messages and its stream's coins
along its spine (`repSteps`). -/
noncomputable def TableSrc.wovenAt (s : TableSrc) (cls : HClass) (P : Fin 2 → Fin cls.mPts → 𝔽) : TableFiles cls :=
  { s.filesAt cls P with
    tape := fun r => repSteps s.st s.S s.sch (extraClaims s.S cls.ptLocal cls.mPts P).length (s.proof r)
      ((s.filesAt cls P).stream r) }

/-- The table's first round: the level-0 cap alone. -/
theorem cut_tableSteps_head {cls : HClass} (G : TableFiles cls) :
    (cut ((tableSteps G).map kindOf) ((tableSteps G).filterMap Sum.getLeft?))[0]? =
      some [⟨Cap Dig, capDig (G.proof 0).cap⟩] := rfl

theorem streamRounds_toList_coins (coins : Array (Array F128)) (fr : ℕ → List UInt8) :
    (streamRounds coins fr).toList.map (·.coins) = coins.toList := by
  rw [← Array.toList_map, streamRounds_coins]

section sched

variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) {cls : HClass}

theorem length_coinSteps_le (cs : List ((k : Fin nR) → CT k)) : ∀ j, (coinSteps CT j cs).length ≤ nR - j := by
  induction cs with
  | nil => intro j; simp [coinSteps]
  | cons ρ cs ih =>
      intro j
      by_cases hj : j < nR
      · rw [coinSteps, dif_pos hj, List.singleton_append, List.length_cons]
        have := ih (j + 1)
        omega
      · rw [coinSteps_ge CT _ j (by omega)]
        simp

theorem length_coinSteps {cs : List ((k : Fin nR) → CT k)} (h : nR ≤ cs.length) : (coinSteps CT 0 cs).length = nR :=
  le_antisymm (by simpa using length_coinSteps_le CT cs 0) (le_length_coinSteps CT cs 0 nR h (by omega))

/-- **The link points the transcript's first coin carries**: when its first round is one message and its first coin
step is the link points `P`. -/
theorem ptsOf_head {ws : List (List (SMsg T))} {cs : List ((k : Fin nR) → CT k)} {P : Fin 2 → Fin cls.mPts → 𝔽}
    (hw : ∃ e, ws.head? = some [e]) (hc : (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin cls.mPts → 𝔽, P⟩) :
    ptsOf T CT (ws.zip cs) = P := by
  obtain ⟨e, he⟩ := hw
  obtain ⟨w0, ws', rfl⟩ : ∃ w0 ws', ws = w0 :: ws' := by
    cases ws with
    | nil => simp at he
    | cons w0 ws' => exact ⟨w0, ws', rfl⟩
  obtain rfl : w0 = [e] := Option.some.inj he
  obtain ⟨c, cs', rfl⟩ : ∃ c cs', cs = c :: cs' := by
    cases cs with
    | nil => simp [coinSteps] at hc
    | cons c cs' => exact ⟨c, cs', rfl⟩
  have h0 : 0 < nR := by
    by_contra h
    rw [coinSteps_ge CT _ 0 (by omega)] at hc
    simp at hc
  have hc' : (⟨CT ⟨0, h0⟩, c ⟨0, h0⟩⟩ : Msg) = ⟨Fin 2 → Fin cls.mPts → 𝔽, P⟩ := by
    rw [coinSteps, dif_pos h0, List.singleton_append, List.head?_cons] at hc
    exact Option.some.inj hc
  apply ptsOf_eq T CT (a := Sum.inl ⟨T e.1, e.2⟩) (rest := schedSteps T CT 1 (ws'.zip cs'))
  rw [List.zip_cons_cons, schedSteps, dif_pos h0, hc']
  rfl

/-- **`TranscriptIs` from the runs.** At link points `P` that the transcript's first coin carries (`hP`), when the
table files run (`hruns`), each of the `r + 1` rounds holds the woven files' messages of that round (`hmsg`), and the
coins are the link points then both reps' rounds (`hcoin`), with at least as many coins as rounds: both reps accept, so
each woven tape is a tape of its spine (`reps_of_runs`), and the transcript's steps are the woven files' steps
(`schedSteps_zip`). -/
theorem TableSrc.transcriptIs_of_runs (s : TableSrc) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (M : Hidden cls) (hS : M.toStatement regs = s.S) (ws : List (List (SMsg T))) (cs : List ((k : Fin nR) → CT k))
    (P : Fin 2 → Fin cls.mPts → 𝔽) (hset : SetupRealizes s.st s.S s.sch cls.ptLocal cls.mPts) (henc : s.Encodes)
    (hP : ptsOf T CT (ws.zip cs) = P) (hruns : (s.files (cls := cls) T CT (ws.zip cs)).Runs)
    (hws : ws.length = nR + 1) (hcs : nR + 1 ≤ cs.length)
    (hmsg : ∀ i (h : i < ws.length), roundMsgs T ws[i] =
      (cut ((tableSteps (s.wovenAt cls P)).map kindOf) ((tableSteps (s.wovenAt cls P)).filterMap Sum.getLeft?))[i]?.getD
        [])
    (hcoin : coinSteps CT 0 cs = ⟨Fin 2 → Fin cls.mPts → 𝔽, P⟩ ::
      ((s.d 0).rounds.toList ++ (s.d 1).rounds.toList).map (coinArr fOf)) :
    s.TranscriptIs T CT cls (ws.zip cs) := by
  have hrG : (s.wovenAt cls P).Runs := by
    have h : (s.filesAt cls (ptsOf T CT (ws.zip cs))).Runs := hruns
    rw [hP] at h
    exact h
  obtain ⟨e, he, hext, -⟩ := s.points_ok hset P
  have hX : s.extrasAt (s.points cls.ptLocal cls.mPts P) = e := s.extrasAt_of he
  have hsz : e.size = 2 * s.S.regions.length := by
    rw [(hext e he).1, FlockSoundness.extraClaims_length]
  obtain ⟨hmS, hkS, hfold, hms, hf, -⟩ := hset
  rw [← hS] at hmS hkS hfold hext
  have hwf : ∀ x ∈ (s.wovenAt cls P).sess.streams.toList, RoundsWF x.rounds := by
    intro x hx
    have hx' : x ∈ [s.stream _ 0, s.stream _ 1] := hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx'
    rcases hx' with rfl | rfl <;> exact s.stream_wf _ _
  have hex : (s.wovenAt cls P).st.extra (s.wovenAt cls P).sess.points = .ok (s.wovenAt cls P).extras := by
    rw [show (s.wovenAt cls P).extras = e from hX]; exact he
  have hdec : ∀ r < 2, decodeProof (shapeOf (s.wovenAt cls P).st (s.wovenAt cls P).sch
      (s.wovenAt cls P).extras.size) ((s.wovenAt cls P).bytes r) = .ok ((s.wovenAt cls P).proof r) := by
    intro r _
    obtain ⟨b, hb⟩ := henc r
    have hb' : decodeProof (shapeOf s.st s.sch (s.extrasAt (s.points cls.ptLocal cls.mPts P)).size) b =
        .ok (s.proof r) := by
      rw [hX, hsz]; exact hb
    exact Classical.epsilon_spec (p := fun b => decodeProof (shapeOf s.st s.sch
      (s.extrasAt (s.points cls.ptLocal cls.mPts P)).size) b = .ok (s.proof r)) ⟨b, hb'⟩
  have hfind : ∀ r < 2, (s.wovenAt cls P).Finds r := by
    intro r hr
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · show Array.find? _ #[s.stream _ 0, s.stream _ 1] = _
      rw [← Array.find?_toList]
      exact List.find?_cons_of_pos (beq_self_eq_true _)
    · show Array.find? _ #[s.stream _ 0, s.stream _ 1] = _
      rw [← Array.find?_toList]
      refine (List.find?_cons_of_neg ?_).trans (List.find?_cons_of_pos (beq_self_eq_true _))
      rw [Bool.not_eq_true]
      exact streamName_ne s.spec
  have hcap : ∀ r < 2, ((s.wovenAt cls P).proof r).cap.foldl (· ++ ·) ByteArray.empty =
      (s.wovenAt cls P).sess.rootB[(s.wovenAt cls P).i]! := by
    intro r _
    show s.cap.foldl (· ++ ·) ByteArray.empty = #[s.cap.foldl (· ++ ·) ByteArray.empty][0]!
    simp
  have h := reps_of_runs (s.wovenAt cls P) regs M hmS hkS hfold hms hf hext hwf hex hrG hfind hdec
    (fun _ _ => rfl) hcap (fun r _ => by rw [hS]; rfl)
  have hT : ∀ r < 2, RepTape (s.wovenAt cls P).st (M.toStatement regs) (s.wovenAt cls P).sch
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts (s.wovenAt cls P).pts).length
      ((s.wovenAt cls P).proof r) ((s.wovenAt cls P).stream r) ((s.wovenAt cls P).tape r) :=
    fun r hr => (h r hr).2.2.2.2
  have hc : coinSteps CT 0 cs = (tableSteps (s.wovenAt cls P)).filterMap Sum.getRight? := by
    rw [tableSteps_coins _ _ _ hT, hcoin]
    show _ = _ :: ((streamRounds (s.d (fin2 0)).rounds _).toList.map (·.coins) ++
      (streamRounds (s.d (fin2 1)).rounds _).toList.map (·.coins)).map (coinArr fOf)
    rw [streamRounds_toList_coins, streamRounds_toList_coins]
    rfl
  have hcut : (cut ((tableSteps (s.wovenAt cls P)).map kindOf)
      ((tableSteps (s.wovenAt cls P)).filterMap Sum.getLeft?)).length = ws.length := by
    rw [length_cut, ← hc, length_coinSteps CT (by omega), hws]
  have hmsg' : ws.map (roundMsgs T) = cut ((tableSteps (s.wovenAt cls P)).map kindOf)
      ((tableSteps (s.wovenAt cls P)).filterMap Sum.getLeft?) := by
    refine List.ext_getElem (by rw [List.length_map, hcut]) fun i h₁ h₂ => ?_
    rw [List.getElem_map, hmsg i (by simpa using h₁), List.getElem?_eq_getElem h₂, Option.getD_some]
  have hsched := schedSteps_zip T CT (tableSteps (s.wovenAt cls P)) ws cs (by omega) hmsg' hc
  refine ⟨P, ((s.wovenAt cls P).tape 0, (s.wovenAt cls P).tape 1), ?_, ?_, hsched⟩
  · have h0 := hT 0 (by omega)
    rw [hS] at h0
    exact h0
  · have h1 := hT 1 (by omega)
    rw [hS] at h1
    exact h1

end sched

/-! ## The coins at V*'s coin types -/

/-- **A coin of `n` field elements, as `coinArr` types it**: one element, else a vector of `n`. -/
def coinTy (n : ℕ) : Type := if n = 1 then 𝔽 else (Fin n → 𝔽)

noncomputable instance (n : ℕ) : Fintype (coinTy n) := by unfold coinTy; split <;> infer_instance

instance (n : ℕ) : Nonempty (coinTy n) := by unfold coinTy; split <;> exact ⟨0⟩

/-- **The inner protocol's coin types are V*'s** at the shape `sh`: coin `0` is the link points (`mPts` coordinates
each, `tableTape`'s first coin), and then each of both reps' rounds in turn is a coin of the size `verifierRows` takes
there (`roundSizes`), so there are `1 + 2 |roundSizes sh|` coins. -/
def CoinTys (mPts : ℕ) (sh : Shape) {nR : ℕ} (CT : Fin nR → Type) : Prop :=
  nR = 1 + 2 * (roundSizes sh).length ∧ (∀ h : 0 < nR, CT ⟨0, h⟩ = (Fin 2 → Fin mPts → 𝔽)) ∧
    ∀ r < 2, ∀ j (hj : j < (roundSizes sh).length) (h : 1 + r * (roundSizes sh).length + j < nR),
      CT ⟨1 + r * (roundSizes sh).length + j, h⟩ = coinTy (roundSizes sh)[j]

/-- **V*'s coin types** at the shape `sh` (`CoinTys`). -/
noncomputable def vCT (mPts : ℕ) (sh : Shape) (k : Fin (1 + 2 * (roundSizes sh).length)) : Type :=
  if k.val = 0 then (Fin 2 → Fin mPts → 𝔽) else coinTy ((roundSizes sh).getD ((k.val - 1) % (roundSizes sh).length) 0)

noncomputable instance (mPts : ℕ) (sh : Shape) (k : Fin (1 + 2 * (roundSizes sh).length)) :
    Fintype (vCT mPts sh k) := by
  unfold vCT; split <;> infer_instance

instance (mPts : ℕ) (sh : Shape) (k : Fin (1 + 2 * (roundSizes sh).length)) : Nonempty (vCT mPts sh k) := by
  unfold vCT; split
  · exact ⟨fun _ _ => 0⟩
  · infer_instance

theorem coinTys_vCT (mPts : ℕ) (sh : Shape) : CoinTys mPts sh (vCT mPts sh) := by
  refine ⟨rfl, fun _ => if_pos rfl, fun r hr j hj h => ?_⟩
  have hk : (1 + r * (roundSizes sh).length + j - 1) % (roundSizes sh).length = j := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · rw [show 1 + 0 * (roundSizes sh).length + j - 1 = j by omega, Nat.mod_eq_of_lt hj]
    · rw [show 1 + 1 * (roundSizes sh).length + j - 1 = j + (roundSizes sh).length by omega, Nat.add_mod_right,
        Nat.mod_eq_of_lt hj]
  show (if 1 + r * (roundSizes sh).length + j = 0 then _ else _) = _
  rw [if_neg (by omega), hk, List.getD_eq_getElem _ _ hj]

/-- **Every coin of a coin type is `coinArr` of an array of its size.** -/
theorem exists_coinArr (n : ℕ) (y : coinTy n) :
    ∃ c : Array F128, c.size = n ∧ (⟨coinTy n, y⟩ : Msg) = coinArr fOf c := by
  by_cases hn : n = 1
  · subst hn
    have e : coinTy 1 = 𝔽 := if_pos rfl
    refine ⟨#[cast e y], rfl, ?_⟩
    have hs : (#[cast e y] : Array F128).size = 1 := rfl
    have h1 : fOf (#[cast e y] : Array F128)[0]! = cast e y := rfl
    rw [coinArr, if_pos hs, h1]
    exact Sigma.ext e (cast_heq e y).symm
  · have e : coinTy n = (Fin n → 𝔽) := if_neg hn
    obtain ⟨c, hc, hv⟩ : ∃ c : Array F128, c.size = n ∧ ∀ i (h : i < n), fOf c[i]! = cast e y ⟨i, h⟩ :=
      have e' : coinTy n = (Fin n → F128) := e
      ⟨Array.ofFn (n := n) fun i => cast e' y i, Array.size_ofFn, fun i h => by
        have hi : i < (Array.ofFn (n := n) fun i => cast e' y i).size := by
          rw [Array.size_ofFn]; exact h
        rw [getElem!_pos _ i hi, Array.getElem_ofFn]; rfl⟩
    subst hc
    refine ⟨c, rfl, ?_⟩
    rw [coinArr, if_neg hn]
    have hf : (fun i : Fin c.size => fOf c[i.val]!) = cast e y := funext fun i => hv i.val i.isLt
    exact Sigma.ext e (by rw [hf]; exact (cast_heq e y).symm)

theorem coinArr_arrOf {n : ℕ} {m : Msg} (h : ∃ c : Array F128, c.size = n ∧ m = coinArr fOf c) :
    coinArr fOf (arrOf n (some m)) = m := by
  have h' : ∃ c : Array F128, c.size = n ∧ some m = some (coinArr fOf c) :=
    let ⟨c, hc, hm⟩ := h; ⟨c, hc, by rw [hm]⟩
  unfold arrOf
  rw [dif_pos h']
  exact (Option.some.inj (Classical.choose_spec h').2).symm

section coins

variable {nR : ℕ} (CT : Fin nR → Type)

theorem getElem?_coinSteps : ∀ (cs : List ((k : Fin nR) → CT k)) (j k n : ℕ) (hn : j + k = n) (h : n < nR)
    (hk : k < cs.length), (coinSteps CT j cs)[k]? = some ⟨CT ⟨n, h⟩, cs[k] ⟨n, h⟩⟩
  | [], _, _, _, _, _, hk => absurd hk (by simp)
  | ρ :: cs, j, 0, n, hn, h, _ => by
      subst hn
      rw [coinSteps, dif_pos (show j < nR from h)]
      rfl
  | ρ :: cs, j, k + 1, n, hn, h, hk => by
      rw [coinSteps, dif_pos (show j < nR by omega), List.singleton_append, List.getElem?_cons_succ]
      exact getElem?_coinSteps cs (j + 1) k n (by omega) h (by simpa using hk)

/-- **`hcoin` at V*'s coin types**: at least as many coins as coin types, the coin steps are the link points the first
carries (`ptsC`), then rep 0's rounds of `coinsOf`, then rep 1's. -/
theorem coinSteps_eq_coinsOf {mPts : ℕ} {sh : Shape} (hCT : CoinTys mPts sh CT) (cs : List ((k : Fin nR) → CT k))
    (hcs : nR ≤ cs.length) :
    coinSteps CT 0 cs = ⟨Fin 2 → Fin mPts → 𝔽, ptsC CT mPts cs⟩ ::
      ((coinsOf CT sh cs 0).toList ++ (coinsOf CT sh cs 1).toList).map (coinArr fOf) := by
  obtain ⟨hn, h0, hk⟩ := hCT
  set ns := roundSizes sh with hns
  have hlen : (coinSteps CT 0 cs).length = nR := length_coinSteps CT hcs
  have hmsg : ∀ r < 2, ∀ j (hj : j < ns.length), ∃ c : Array F128, c.size = ns[j] ∧
      (coinSteps CT 0 cs)[1 + r * ns.length + j]? = some (coinArr fOf c) := by
    intro r hr j hj
    have hlt : 1 + r * ns.length + j < nR := by
      rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl <;> omega
    rw [getElem?_coinSteps CT cs 0 _ _ (Nat.zero_add _) hlt (by omega)]
    obtain ⟨c, hc, he⟩ :=
      exists_coinArr ns[j] (cast (hk r hr j hj hlt) ((cs[1 + r * ns.length + j]'(by omega)) ⟨_, hlt⟩))
    exact ⟨c, hc, by rw [← he]; exact congrArg some (Sigma.ext (hk r hr j hj hlt) (cast_heq _ _).symm)⟩
  have hpos : 0 < nR := by omega
  have hP : (coinSteps CT 0 cs).head? =
      some ⟨Fin 2 → Fin mPts → 𝔽, cast (h0 hpos) ((cs[0]'(by omega)) ⟨0, hpos⟩)⟩ := by
    rw [List.head?_eq_getElem?, getElem?_coinSteps CT cs 0 0 0 rfl hpos (by omega)]
    exact congrArg some (Sigma.ext (h0 hpos) (cast_heq _ _).symm)
  rw [ptsC_eq hP]
  have hro : ∀ r : Fin 2, (coinsOf CT sh cs r).toList.map (coinArr fOf) =
      (List.range ns.length).map fun j => ((coinSteps CT 0 cs)[1 + r.val * ns.length + j]?).getD ⟨Unit, ()⟩ := by
    intro r
    refine List.ext_getElem (by simp [coinsOf, hns]) fun j h₁ h₂ => ?_
    have hj : j < ns.length := by simpa using h₂
    obtain ⟨c, hc, he⟩ := hmsg r.val r.isLt j hj
    have e1 : (coinsOf CT sh cs r).toList[j]'(by simpa using h₁) =
        arrOf ns[j] ((coinSteps CT 0 cs)[1 + r.val * ns.length + j]?) := by
      simp [coinsOf, hns]
    simp only [List.getElem_map, List.getElem_range]
    rw [e1, he, Option.getD_some]
    exact coinArr_arrOf ⟨c, hc, rfl⟩
  rw [List.map_append, hro 0, hro 1]
  refine List.ext_getElem? fun i => ?_
  rcases i with _ | i
  · rw [List.getElem?_cons_zero, ← List.head?_eq_getElem?, hP]
  rw [List.getElem?_cons_succ]
  by_cases hi : i < ns.length
  · rw [List.getElem?_append_left (by simpa using hi)]
    simp only [List.getElem?_map, List.getElem?_range hi, Option.map_some]
    obtain ⟨c, -, he⟩ := hmsg 0 (by omega) i hi
    have e0 : 1 + (0 : Fin 2).val * ns.length + i = i + 1 := by
      have : (0 : Fin 2).val = 0 := rfl
      omega
    have e0' : 1 + 0 * ns.length + i = i + 1 := by omega
    rw [e0]
    rw [e0'] at he
    rw [he, Option.getD_some]
  · by_cases hi' : i < 2 * ns.length
    · rw [List.getElem?_append_right (by simpa using hi)]
      simp only [List.length_map, List.length_range, List.getElem?_map, List.getElem?_range (by omega : i - ns.length <
        ns.length), Option.map_some]
      obtain ⟨c, -, he⟩ := hmsg 1 (by omega) (i - ns.length) (by omega)
      have e1 : 1 + (1 : Fin 2).val * ns.length + (i - ns.length) = i + 1 := by
        have : (1 : Fin 2).val = 1 := rfl
        omega
      have e1' : 1 + 1 * ns.length + (i - ns.length) = i + 1 := by omega
      rw [e1]
      rw [e1'] at he
      rw [he, Option.getD_some]
    · rw [List.getElem?_eq_none (l := coinSteps CT 0 cs) (by rw [hlen]; omega)]
      symm
      apply List.getElem?_eq_none
      simp
      omega

end coins

/-! ## The decoder's messages -/

section msgs

variable {nT : ℕ} (T : Fin nT → Type)

open Classical in
/-- A message as one of the schedule's typed messages, where some `T j` is its type. -/
noncomputable def toS (m : Msg) : List (SMsg T) :=
  if h : ∃ j, T j = m.1 then [⟨Classical.choose h, cast (Classical.choose_spec h).symm m.2⟩] else []

/-- A round's messages as the schedule's typed messages (`toS`). -/
noncomputable def toSs (g : List Msg) : List (SMsg T) := g.flatMap (toS T)

theorem roundMsgs_toSs : ∀ (g : List Msg), (∀ m ∈ g, ∃ j, T j = m.1) → roundMsgs T (toSs T g) = g
  | [], _ => rfl
  | m :: g, h => by
      have hm := h m (List.mem_cons_self ..)
      have ih := roundMsgs_toSs g fun m' hm' => h m' (List.mem_cons_of_mem _ hm')
      have e : toSs T (m :: g) = toS T m ++ toSs T g := rfl
      rw [e, toS, dif_pos hm]
      simp only [roundMsgs, List.map_append, List.map_cons, List.map_nil, List.singleton_append] at ih ⊢
      rw [ih]
      exact congrArg (· :: g) (Sigma.ext (Classical.choose_spec hm) (cast_heq _ _))

theorem roundMsgs_getElem {ws : List (List (SMsg T))} {i : ℕ} (hi : i < ws.length) {g : List Msg}
    (e : (ws[i]?).map (roundMsgs T) = some g) : roundMsgs T ws[i] = g := by
  rw [List.getElem?_eq_getElem hi, Option.map_some] at e
  exact Option.some.inj e

theorem eq_singleton_of_roundMsgs {w : List (SMsg T)} {m : Msg} (h : roundMsgs T w = [m]) : ∃ e, w = [e] := by
  match w, h with
  | [], h => simp [roundMsgs] at h
  | [e], _ => exact ⟨e, rfl⟩
  | _ :: _ :: _, h => simp [roundMsgs] at h

end msgs

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 leaf512 cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

/-- **The woven table files** at a history and V*'s layers: `tableSrc`'s, at the link points of the history's first
coin (`ptsC`), each rep's tape woven from its files (`TableSrc.wovenAt`). -/
noncomputable def woven (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) :
    TableFiles cls :=
  (F.tableSrc spec st cap lig R ω t l).wovenAt cls (ptsC CT cls.mPts (t.map Prod.snd))

/-- **The table's messages, round by round**: the woven files' steps' messages cut at their coins (`PlaysCoin.cut`). -/
noncomputable def tableMsgs (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : List (List Msg) :=
  cut ((tableSteps (F.woven spec st cap lig R ω t l)).map kindOf)
    ((tableSteps (F.woven spec st cap lig R ω t l)).filterMap Sum.getLeft?)

/-- **V*'s sessions accept only at a whole history**: at every accepted outer proof, the history has a coin for each of
the inner protocol's rounds. `VBridge` reads every history, and the inner transcript of a history with fewer coins than
rounds (`zip` stops at the last coin) leaves the openings unsent, so the inner verifier cannot accept it; the recursive
game reads `VBridge` only at its leaves, whose histories are whole (`recGame`'s `rounds … I.r`). -/
def HLeaf : Prop := ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → (ℐ).r ≤ t.length

/-- **`hfiles` after the runs**: at every accepted outer proof whose table files run, they are valid for the transcript
V*'s layers hold at the rounds' coins. `TableFiles.Valid` asks that each rep's tape be a tape of its spine, which holds
once the rep runs (`reps_of_runs`). -/
def HFilesR : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Runs →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Valid regs (F.x R ω) T CT
      ((F.wmsg (layers F.zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))

/-- **The messages V*'s layers hold are the table's** (`hmsg`): at every accepted outer proof there is one per inner
round, and round `i`'s is the table's round `i` (`tableMsgs`). -/
def MsgsAre : Prop := F.AtAccepted dg N k Rw fun R ω t l =>
  (F.wmsg l).length = (ℐ).r ∧
    ∀ i < (ℐ).r, ((F.wmsg l)[i]?).map (roundMsgs T) = some ((F.tableMsgs spec st cap lig R ω t l)[i]?.getD [])

/-- **The inner transcript's link points are the history's first coin's** at an accepted outer proof: its first round
is the level-0 cap alone (`MsgsAre`, `cut_tableSteps_head`) and its first coin the link points (`coinSteps_eq_coinsOf`
at V*'s coin types). -/
theorem ptsOf_at (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw)
    (hmsgs : F.MsgsAre dg N k Rw spec st cap lig)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) :
    ptsOf (cls := cls) T CT ((F.wmsg (layers F.zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)) =
      ptsC CT cls.mPts (t.map Prod.snd) := by
  have hr : nR + 1 ≤ t.length := hleaf R ω t R₂ τ o h
  obtain ⟨-, hm⟩ := hmsgs R ω t R₂ τ o h
  have h0 := hm 0 (Nat.succ_pos nR)
  rw [show (F.tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[0]? = some _ from
    cut_tableSteps_head _, Option.getD_some] at h0
  obtain ⟨w, hw, hw'⟩ := Option.map_eq_some_iff.1 h0
  obtain ⟨e, rfl⟩ := eq_singleton_of_roundMsgs T hw'
  have hr' : nR ≤ (t.map Prod.snd).length := by rw [List.length_map]; omega
  refine ptsOf_head T CT ⟨e, List.head?_eq_getElem?.trans hw⟩ ?_
  exact congrArg List.head? (coinSteps_eq_coinsOf CT hCT _ hr')

/-- **`hfiles` after the runs** for a family whose files are `filesOf`'s and whose `rep₀` is `repOf coinFiles`: the
transcript is the woven files' (`transcriptIs_of_runs`, at the history's first coin's link points, `ptsOf_at`), so the
files are valid (`TableSrc.valid`). -/
theorem hFilesR_of (hF : F.files = F.filesOf spec st cap lig) (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig))
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (henc : F.AtAccepted dg N k Rw fun R ω t l => (F.tableSrc spec st cap lig R ω t l).Encodes)
    (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw) (hmsgs : F.MsgsAre dg N k Rw spec st cap lig) :
    F.HFilesR dg N k Rw := by
  intro R ω t R₂ τ o h hruns
  rw [hF] at hruns ⊢
  have hr : nR + 1 ≤ t.length := hleaf R ω t R₂ τ o h
  have hr' : nR + 1 ≤ (t.map Prod.snd).length := by rw [List.length_map]; exact hr
  obtain ⟨hlen, hm⟩ := hmsgs R ω t R₂ τ o h
  refine TableSrc.valid T CT _ regs (F.x R ω) _ rfl (hset R ω) (henc R ω t R₂ τ o h) ?_
  refine (F.tableSrc spec st cap lig R ω t _).transcriptIs_of_runs T CT regs (F.x R ω) rfl _ (t.map Prod.snd)
    (ptsC CT cls.mPts (t.map Prod.snd)) (hset R ω) (henc R ω t R₂ τ o h) (F.ptsOf_at dg N k Rw spec st cap lig hCT hleaf
      hmsgs h) hruns hlen hr' (fun i hi => ?_) ?_
  · exact roundMsgs_getElem T hi (hm i (by rw [← hlen]; exact hi))
  · show coinSteps CT 0 _ = _ :: ((F.rep R ω t _ 0).rounds.toList ++ (F.rep R ω t _ 1).rounds.toList).map _
    rw [F.rep_rounds spec st cap lig hrep, F.rep_rounds spec st cap lig hrep]
    exact coinSteps_eq_coinsOf CT hCT _ (Nat.le_of_succ_le hr')

/-- **The family at the coins' rep**: `F` with `rep₀` the coins' (`repOf coinFiles`), `rows` and `acc` the layers'
(`rowsOf`, `accOf`), then `alpha` `alphaOf` at that `rep₀`. It reads no message, so the decoder is made from it. -/
noncomputable def coinFamily : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
  let F₁ : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
    { F with rep₀ := F.repOf (F.coinFiles spec st cap lig), rows := F.rowsOf, acc := F.accOf }
  { F₁ with alpha := F₁.alphaOf }

variable (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)

/-- **The table's round-`i` messages are read from stage `i`'s commit strings**: two accepted outer proofs whose layers
agree on the read wires of stage `i`'s commit strings (`maskOf` at `skOf own sel`) have the same round-`i` messages
(`tableMsgs`, at the coins' family). Round `i`'s messages are the values w, the cap and the openings hold in the rows V*
registers for round `i` (`sel i`), which `skOf` makes stage `i`'s commit strings. -/
def MsgsRead : Prop :=
  ∀ i R ω t R₂ τ o R' ω' t' R₂' τ' o', Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    Accepted (I := ℐ) F.zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) =
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R' ω' t' R₂' τ') →
    ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[i]? =
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R' ω' t' (layers F.zs dg N k Rw R' ω' t' R₂' τ'))[i]?

/-- **The inner protocol's message types are V*'s**: at every accepted outer proof, each message of the table's
rounds has its type among the schedule's message types `T`. -/
def MsgTys : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ g ∈ (F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ),
      ∀ m ∈ g, ∃ j, T j = m.1

open Classical in
/-- **The decoder**: round `i`'s message at the read wires `Xs` of stage `i`'s commit strings is the table's round-`i`
messages at any accepted outer proof whose layers read `Xs` there, typed by `T` (`toSs`), and none elsewhere. -/
noncomputable def decOf (i : ℕ) (Xs : (j : Fin F.stmts.size) → (F.Vs j).Bits) : (ℐ).Msg :=
  if h : ∃ g : List Msg, ∃ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o ∧
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) = Xs ∧
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
        (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] = g
  then toSs T (Classical.choose h) else []

/-- **`hmsg` at the decoder**, at an accepted outer proof: round `i`'s message `decOf` reads at the layers' stage-`i`
commit strings is the table's round `i` (`MsgsRead`, `MsgTys`). -/
theorem decOf_spec (hread : F.MsgsRead dg N k Rw spec st cap lig own sel) (htys : F.MsgTys dg N k Rw spec st cap lig)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (i : ℕ) :
    roundMsgs T (F.decOf dg N k Rw spec st cap lig own sel i
      (maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ))) =
    ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
  have hex : ∃ g : List Msg, ∃ R' ω' t' R₂' τ' o', Accepted (I := ℐ) F.zs dg N k Rw R' ω' t' R₂' τ' o' ∧
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R' ω' t' R₂' τ') =
        maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) ∧
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R' ω' t'
        (layers F.zs dg N k Rw R' ω' t' R₂' τ'))[i]?.getD [] = g := ⟨_, R, ω, t, R₂, τ, o, h, rfl, rfl⟩
  have hd : F.decOf dg N k Rw spec st cap lig own sel i
      (maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ)) = toSs T (Classical.choose hex) :=
    dif_pos hex
  rw [hd]
  obtain ⟨R', ω', t', R₂', τ', o', h', hX', hg'⟩ := Classical.choose_spec hex
  have e := hread i R ω t R₂ τ o R' ω' t' R₂' τ' o' h h' hX'.symm
  have hc : Classical.choose hex = ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
    rw [← hg', ← e]
  rw [hc]
  apply roundMsgs_toSs
  intro m hm
  cases hti : ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]? with
  | none => rw [hti] at hm; simp at hm
  | some g =>
      rw [hti, Option.getD_some] at hm
      exact htys R ω t R₂ τ o h g (List.mem_of_getElem? hti) m hm

end VStarFamily

end family

end FlockVBridge.Family
