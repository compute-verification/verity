import Proofs.Flock.Claims.KS.PlainRunR
import Proofs.Flock.Claims.KS.PlainFitCausal
import Proofs.Flock.Claims.KS.PlainLive
import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.KS.PlainStepJ

/-!
# The plain simulator at `VR rd`

`VR rd` reads a round's frames from the last message's proof (`readPs`), but a live prover sends round `j` before the
later coins. The simulator `plainSimR` sends, at round `j`, the `j`-th chunk of any fit consistent with `P`'s rows
through round `j` (`plainFrameR`), which `fit_chunk` fixes wherever the rep is set up, so on an accepted run its round
`j` is the frame `VR rd` reads there.

* `plainSimR_liveOut`: the simulator's live outcome, `P`'s roots, the link points, the rounds of its submissions and
  the files of its last message.
* `plainSimR_schedL`: it submits `repRounds` rounds on rep 0 first.
* `plainSimRunR`: on an accepted run of `VR rd`, its live outcome is the decision's view (`PlainSimRunR`).
* `plainSimAtR`, `plainRdInR`, `rdAllAt_RS`: so `PlainSimAtR` holds at every statement, and with it `PlainRdInR` and
  `RdAllAt` at the deployed reader.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open RecDeploy (outCoin)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-! ## Histories -/

theorem liveH_congr {J mPts : ℕ} (w₁ w₂ : Wire 65536 J mPts) (he : w₁.encRd = w₂.encRd) :
    ∀ (k : ℕ) (b : ℕ → Claims.Msg) (S₁ S₂ : ℕ → SubmitJ J) (r : ℕ → LCoin 65536), (∀ i, (S₁ i).n = (S₂ i).n) →
      liveH w₁ b S₁ r k = liveH w₂ b S₂ r k
  | 0, _, _, _, _, _ => rfl
  | k + 1, b, S₁, S₂, r, hn => by
    simp only [liveH]
    rw [he, hn 0, liveH_congr w₁ w₂ he k (fun i => b (i + 1)) (fun i => S₁ (i + 1)) (fun i => S₂ (i + 1))
      (fun i => r (i + 1)) fun i => hn (i + 1)]

theorem range_map_ends2 {α : Type} (f : ℕ → α) (n : ℕ) :
    (List.range (n + 2)).map f = f 0 :: ((List.range n).map (fun i => f (i + 1)) ++ [f (n + 1)]) := by
  rw [range_map_ends f (by omega : 2 ≤ n + 2), show n + 2 - 2 = n by omega, show n + 2 - 1 = n + 1 by omega]

/-! ## The frames a prefix of rows fixes -/

/-- A message's rows, none where it carries none. -/
def rowsD (m : Claims.Msg) : List ByteArray := (rowsOf m).getD []

/-- A message's rows' nodes (`nodeOf`): round 0's link, as `decR` reads it. -/
def rowsN (m : Claims.Msg) : List ByteArray := (rowsD m).map nodeOf

/-- The rounds before deployed round `i`'s rep: `|plainRepNs|` on rep 1. -/
def plainROff (spec : SessionSpec) (st : Stmt) (i : ℕ) : ℕ := plainRep spec st i * (plainRepNs spec st).length

/-- Deployed round `i`'s round within its rep. -/
def plainRJ (spec : SessionSpec) (st : Stmt) (i : ℕ) : ℕ := i - 1 - plainROff spec st i

/-- The rows of deployed round `i`'s rep through round `i`, at `P`'s messages `dm`. -/
def plainPreR (spec : SessionSpec) (st : Stmt) (dm : ℕ → Claims.Msg) (i : ℕ) : List (List ByteArray) :=
  (List.range (plainRJ spec st i + 1)).map fun t => rowsD (dm (1 + plainROff spec st i + t))

/-- Deployed round `i`'s rep context at the link coin `c₁` and round 0's rows. -/
def plainCtx (spec : SessionSpec) (st : Stmt) (c₁ : Coin) (dm : ℕ → Claims.Msg) (i : ℕ) : RepCtx :=
  repCtx spec #[st] c₁ (rowsN (dm 0)) [] (plainRep spec st i)

open Classical in
/-- **Deployed round `i`'s frame**: the round's chunk of a fit of its rep consistent with the rows through round `i`,
none where none is. -/
noncomputable def plainFrameR (spec : SessionSpec) (st : Stmt) (c₁ : Coin) (dm : ℕ → Claims.Msg) (i : ℕ) :
    Claims.Msg :=
  if h : ∃ q : List (List ByteArray) × Flock.Proof × F128,
      q.1.take (plainRJ spec st i + 1) = plainPreR spec st dm i ∧ FitsRep (plainCtx spec st c₁ dm i) q.1 q.2.1 q.2.2 then
    encs ((chunksOf (plainCtx spec st c₁ dm i) (Classical.choose h).2.1 (Classical.choose h).2.2).getD
      (plainRJ spec st i) [])
  else []

theorem plainROff_le {spec : SessionSpec} {st : Stmt} {i : ℕ} (hi : 1 ≤ i) : 1 + plainROff spec st i ≤ i := by
  unfold plainROff plainRep
  split_ifs <;> omega

/-- **Deployed round `i`'s frame reads `P`'s messages through `i` only.** -/
theorem plainFrameR_congr {spec : SessionSpec} {st : Stmt} {c₁ : Coin} {dm dm' : ℕ → Claims.Msg} {i : ℕ}
    (hi : 1 ≤ i) (h : ∀ t ≤ i, dm t = dm' t) : plainFrameR spec st c₁ dm i = plainFrameR spec st c₁ dm' i := by
  have h0 : plainCtx spec st c₁ dm i = plainCtx spec st c₁ dm' i := by
    unfold plainCtx
    rw [h 0 (Nat.zero_le _)]
  have hp : plainPreR spec st dm i = plainPreR spec st dm' i := by
    unfold plainPreR
    refine List.map_congr_left fun t ht => ?_
    have := List.mem_range.1 ht
    unfold plainRJ at this
    have := plainROff_le (spec := spec) (st := st) hi
    show rowsD (dm (1 + plainROff spec st i + t)) = rowsD (dm' (1 + plainROff spec st i + t))
    rw [h (1 + plainROff spec st i + t) (by omega)]
  unfold plainFrameR
  rw [h0, hp]

/-- **At a fit consistent with the rows so far, the frame is its chunk** (`fit_chunk`). -/
theorem plainFrameR_eq {spec : SessionSpec} {st : Stmt} {c₁ : Coin} {dm : ℕ → Claims.Msg} {i j : ℕ} {c : RepCtx}
    {rows : List (List ByteArray)} {p : Flock.Proof} {T : F128} (hc : plainCtx spec st c₁ dm i = c)
    (hJ : plainRJ spec st i = j) (hsc : SchedOf c) (hfit : FitsRep c rows p T) (hj : j < rows.length)
    (hpre : rows.take (j + 1) = plainPreR spec st dm i) :
    plainFrameR spec st c₁ dm i = encs ((chunksOf c p T).getD j []) := by
  subst hc hJ
  have hex : ∃ q : List (List ByteArray) × Flock.Proof × F128,
      q.1.take (plainRJ spec st i + 1) = plainPreR spec st dm i ∧
        FitsRep (plainCtx spec st c₁ dm i) q.1 q.2.1 q.2.2 := ⟨(rows, p, T), hpre, hfit⟩
  unfold plainFrameR
  rw [dif_pos hex]
  exact congrArg encs (fit_chunk hsc hfit (Classical.choose_spec hex).2 hj
    (hpre.trans (Classical.choose_spec hex).1.symm)).symm

/-- **Round `j` of rep `q`'s frame** is the `j`-th chunk of rep `q`'s fit to its block of rows. -/
theorem plainFrameR_at {spec : SessionSpec} {st : Stmt} {c₁ : Coin} {dm : ℕ → Claims.Msg}
    {rss : List (List ByteArray)} {L q j m : ℕ} {p : Flock.Proof} {T : F128}
    (hn₀ : (plainRepNs spec st).length = L) (hlen : rss.length = L + L)
    (hrows : ∀ i (hi : i < rss.length), rss[i] = rowsD (dm (i + 1))) (hq : q < 2) (hj : j < L)
    (hm : m = q * L + j + 1) (hsc : SchedOf (repCtx spec #[st] c₁ (rowsN (dm 0)) [] q))
    (hfit : FitsRep (repCtx spec #[st] c₁ (rowsN (dm 0)) [] q) ((rss.drop (q * L)).take L) p T) :
    plainFrameR spec st c₁ dm m = encs ((chunksOf (repCtx spec #[st] c₁ (rowsN (dm 0)) [] q) p T).getD j []) := by
  subst hm
  have hqL : q * L + L ≤ L + L := by interval_cases q <;> omega
  have hrep : plainRep spec st (q * L + j + 1) = q := by
    unfold plainRep
    rw [hn₀]
    interval_cases q <;> split_ifs <;> omega
  have hoff : plainROff spec st (q * L + j + 1) = q * L := by
    unfold plainROff
    rw [hrep, hn₀]
  have hJ : plainRJ spec st (q * L + j + 1) = j := by
    unfold plainRJ
    rw [hoff]
    omega
  refine plainFrameR_eq (by unfold plainCtx; rw [hrep]) hJ hsc hfit
    (by rw [List.length_take, List.length_drop]; omega) ?_
  unfold plainPreR
  rw [hJ, hoff]
  refine List.ext_getElem (by simp only [List.length_take, List.length_drop, List.length_map, List.length_range]; omega)
    fun t h₁ h₂ => ?_
  simp only [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range, hrows]
  all_goals (congr 2; omega)

theorem fitsRep_cs {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray} {cs cs' : List Coin}
    {k : ℕ} {rows : List (List ByteArray)} {p : Flock.Proof} {T : F128}
    (h : FitsRep (repCtx spec sts c₁ r₀ cs k) rows p T) : FitsRep (repCtx spec sts c₁ r₀ cs' k) rows p T :=
  ⟨h.rounds, h.sched, h.cap, h.tops, h.nonces, h.sizes, h.levels, h.elems⟩

theorem framesOf_getElem? {c : RepCtx} {p : Flock.Proof} {T : F128} {i : ℕ} (h : i < (chunksOf c p T).length) :
    (framesOf c p T)[i]? = some (encs ((chunksOf c p T).getD i [])) := by
  unfold framesOf
  rw [List.getElem?_map, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h] <;> rfl

/-- **A decided proof at one table**: two reps, each a fit to its block of `L` rows, whose frames are the round
messages. -/
theorem readPs_one {spec : SessionSpec} {st : Stmt} {c₁ : Coin} {r₀ : List ByteArray} {rss : List (List ByteArray)}
    {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg} {L : ℕ}
    (hLen : (RecDeploy.repSchedAt spec st).length = L) (h : readPs spec #[st] c₁ r₀ rss cs fl = some pf) :
    rss.length = L + L ∧ ∃ (p₀ : Flock.Proof) (T₀ : F128) (p₁ : Flock.Proof) (T₁ : F128),
      FitsRep (repCtx spec #[st] c₁ r₀ cs 0) ((rss.drop (0 * L)).take L) p₀ T₀ ∧
      FitsRep (repCtx spec #[st] c₁ r₀ cs 1) ((rss.drop (1 * L)).take L) p₁ T₁ ∧
      pf.2 = framesOf (repCtx spec #[st] c₁ r₀ cs 0) p₀ T₀ ++ framesOf (repCtx spec #[st] c₁ r₀ cs 1) p₁ T₁ := by
  have hlens : repLens spec #[st] = [L, L] := by
    rw [← hLen]
    first | rfl | simp [repLens, List.range_succ]
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * (#[st] : Array Stmt).size ∧ (repLens spec #[st]).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    have h2 : rs.length = 2 := by simpa using hlen
    obtain ⟨⟨pa, ma⟩, ⟨pb, mb⟩, rfl⟩ := List.length_eq_two.1 h2
    have ha : readRep (repCtx spec #[st] c₁ r₀ cs 0) ((cutLens (repLens spec #[st]) rss).getD 0 []) fl.ps[0]! =
        some (pa, ma) := by
      have := hget 0 (by simp) (by simp)
      simpa only [List.getElem_range, List.getElem_cons_zero, List.getElem_cons_succ] using this
    have hb : readRep (repCtx spec #[st] c₁ r₀ cs 1) ((cutLens (repLens spec #[st]) rss).getD 1 []) fl.ps[1]! =
        some (pb, mb) := by
      have := hget 1 (by simp) (by simp)
      simpa only [List.getElem_range, List.getElem_cons_zero, List.getElem_cons_succ] using this
    obtain ⟨T₀, hf₀, hm₀⟩ := readRep_fits ha
    obtain ⟨T₁, hf₁, hm₁⟩ := readRep_fits hb
    have e0 : (cutLens (repLens spec #[st]) rss).getD 0 [] = (rss.drop (0 * L)).take L := by
      rw [hlens]
      simp [cutLens]
    have e1 : (cutLens (repLens spec #[st]) rss).getD 1 [] = (rss.drop (1 * L)).take L := by
      rw [hlens]
      simp [cutLens]
    refine ⟨?_, pa, T₀, pb, T₁, by rw [← e0]; exact hf₀, by rw [← e1]; exact hf₁, ?_⟩
    · have := hc.2
      rw [hlens] at this
      simp at this
      omega
    · simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
      rw [hm₀, hm₁]
  · rw [if_neg hc] at h
    cases h

/-- **The decided frames are the simulator's**: at a decided proof whose rows are `P`'s round messages', frame `k`
is deployed round `k + 1`'s `plainFrameR`, wherever both reps are set up. -/
theorem pf_frames {spec : SessionSpec} {st : Stmt} {c₁ : Coin} {dm : ℕ → Claims.Msg} {rss : List (List ByteArray)}
    {cs : List Coin} {L : ℕ} {p₀ p₁ : Flock.Proof} {T₀ T₁ : F128} {fr : List Claims.Msg}
    (hlen : rss.length = L + L) (hrows : ∀ i (hi : i < rss.length), rss[i] = rowsD (dm (i + 1)))
    (hsc : ∀ q < 2, SchedOf (repCtx spec #[st] c₁ (rowsN (dm 0)) [] q))
    (hf₀ : FitsRep (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 0) ((rss.drop (0 * L)).take L) p₀ T₀)
    (hf₁ : FitsRep (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 1) ((rss.drop (1 * L)).take L) p₁ T₁)
    (hfr : fr = framesOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 0) p₀ T₀ ++
      framesOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 1) p₁ T₁) :
    fr.length = L + L ∧ ((plainRepNs spec st).length = L → ∀ k < L + L,
      fr[k]? = some (plainFrameR spec st c₁ dm (k + 1))) := by
  have hch₀ : (chunksOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 0) p₀ T₀).length = L := by
    rw [hf₀.rounds, List.length_take, List.length_drop]
    omega
  have hch₁ : (chunksOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 1) p₁ T₁).length = L := by
    rw [hf₁.rounds, List.length_take, List.length_drop]
    omega
  have hfr₀ : (framesOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 0) p₀ T₀).length = L := by
    rw [framesOf, List.length_map, hch₀]
  have hfr₁ : (framesOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 1) p₁ T₁).length = L := by
    rw [framesOf, List.length_map, hch₁]
  subst hfr
  refine ⟨by rw [List.length_append, hfr₀, hfr₁], fun hn₀ k hk => ?_⟩
  by_cases hkL : k < L
  · have h1 : k < (framesOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 0) p₀ T₀).length := by rw [hfr₀]; exact hkL
    have h2 : k < (chunksOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 0) p₀ T₀).length := by rw [hch₀]; exact hkL
    rw [List.getElem?_append_left h1, framesOf_getElem? h2]
    exact congrArg some (plainFrameR_at (q := 0) (j := k) (m := k + 1) hn₀ hlen hrows (by omega) hkL (by omega)
      (hsc 0 (by omega)) (fitsRep_cs hf₀)).symm
  · have h1 : (framesOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 0) p₀ T₀).length ≤ k := by rw [hfr₀]; omega
    have h2 : k - L < (chunksOf (repCtx spec #[st] c₁ (rowsN (dm 0)) cs 1) p₁ T₁).length := by rw [hch₁]; omega
    rw [List.getElem?_append_right h1, hfr₀, framesOf_getElem? h2]
    exact congrArg some (plainFrameR_at (q := 1) (j := k - L) (m := k + 1) hn₀ hlen hrows (by omega) (by omega)
      (by omega) (hsc 1 (by omega)) (fitsRep_cs hf₁)).symm

/-! ## The simulator -/

/-- The proof files `decR` decides on, none where it doesn't decide. -/
noncomputable def plainFilesR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg)
    (cs : List Coin) : Array ByteArray :=
  ((decR rd x as cs).map fun d => FlockDeploy.filesAt (tabsOf d.sts) (pointsOf d.spec d.c₁) d.f.ps).getD #[]

theorem plainFilesR_congr (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (f g : ℕ → Coin)
    (hL : 2 ≤ as.length) (h : ∀ i, i + 1 < as.length → f i = g i) :
    plainFilesR rd x as ((List.range as.length).map f) = plainFilesR rd x as ((List.range as.length).map g) := by
  have e : decR rd x as ((List.range as.length).map f) = decR rd x as ((List.range as.length).map g) := by
    refine decR_congr_coins rd x as ?_ ?_
    · rw [range_map_ends f hL, range_map_ends g hL, List.head?_cons, List.head?_cons, h 0 (by omega)]
    · rw [range_map_ends f hL, range_map_ends g hL, List.tail_cons, List.tail_cons, List.dropLast_concat,
        List.dropLast_concat]
      refine List.map_congr_left fun i hi => h (i + 1) ?_
      have := List.mem_range.1 hi
      omega
  unfold plainFilesR
  rw [e]

/-- **The simulator's message at the live history `hl`**: `P`'s at the coins `simCoins` it rebuilds, its first as it
is, each round's as `plainRound` at the round's rep and count with the round's frame (`plainFrameR`), and at the last
the files `VR rd` decides on where it accepts `P` at those coins, else nothing. -/
noncomputable def plainMsgR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) (ω : Ω) (u : plainTape x) (hl : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  if hl.length = 0 then depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl)) 0
  else if hl.length ≤ plainN x then
    plainRound (plainRep spec st hl.length) (plainCnt x spec st hl.length)
      (0 :: plainFrameR spec st (Claims.coinSeq (simCoins x spec st u hl) 0)
        (depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl))) hl.length)
  else if (Claims.run sha512 P (VR rd) x ω (simCoins x spec st u hl)).out = .acc then
    filesCode (plainFilesR rd x ((Claims.run sha512 P (VR rd) x ω (simCoins x spec st u hl)).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (VR rd) x ω (simCoins x spec st u hl)).msgs.map Prod.fst).length).map
        (Claims.coinSeq (simCoins x spec st u hl))))
  else []

/-- **The plain simulator at `VR rd`** of `P` at `x`, at the tape `(ω, u)`. -/
noncomputable def plainSimR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) : Claims.Prover (Ω × plainTape x) :=
  ⟨fun _ hl p => .ret (plainMsgR rd P x spec st p.1 p.2 hl)⟩

theorem plainSimR_first (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) (ω : Ω) (u : plainTape x) :
    (((plainSimR rd P x spec st).next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1 := by
  simp only [plainSimR, Claims.OC.eval, plainMsgR]
  rfl

/-- The simulator's submission `k`: deployed round `k + 1`'s rep, frame and count. -/
noncomputable def plainSR (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (c₁ : Coin) (dm : ℕ → Claims.Msg)
    (k : ℕ) : SubmitJ 1 :=
  ⟨0, ⟨plainRep spec st (k + 1), plainRep_lt spec st (k + 1)⟩, ⟨(plainFrameR spec st c₁ dm (k + 1)).toArray⟩,
    plainCnt x spec st (k + 1)⟩

/-- **The simulator's messages**: deployed round `k + 1`'s on the plain wire before the last, and at the last the
files `fs` where `acc`, else nothing. -/
noncomputable def plainBR (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (c₁ : Coin) (dm : ℕ → Claims.Msg)
    (acc : Bool) (fs : Array ByteArray) (k : ℕ) : Claims.Msg :=
  if k < plainN x then plainRound (plainRep spec st (k + 1)) (plainCnt x spec st (k + 1)) (0 :: plainFrameR spec st c₁ dm (k + 1))
  else if acc then filesCode fs else []

/-- Whether `VR rd` accepts `P` at `plainΦ`'s coins. -/
noncomputable def plainAccR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) (ω : Ω) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536) : Bool :=
  decide ((Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).out = .acc)

/-- The files `VR rd` decides on at `plainΦ`'s coins. -/
noncomputable def plainFsR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) (ω : Ω) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536) :
    Array ByteArray :=
  plainFilesR rd x ((Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
    ((List.range ((Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
      (Claims.coinSeq (plainΦ x spec st (u, r'))))

/-- The simulator's messages at `plainΦ`'s coins. -/
noncomputable def plainBΦ (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) (ω : Ω) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536) :
    ℕ → Claims.Msg :=
  plainBR x spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0) (plainDm P x spec st ω u r')
    (plainAccR rd P x spec st ω u r') (plainFsR rd P x spec st ω u r')

/-- **The simulator sends `plainBΦ` along the live history.** -/
theorem plainSimR_msg (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) (ω : Ω) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x)
    (r' : Fin (plainN x + 2) → LCoin 65536) (hL : plainN x + 2 = (FlockDeploy.V false).k x) (k : ℕ)
    (hk : k ≤ plainN x) :
    (((plainSimR rd P x spec st).next x (plainHl P x spec st ω hw (plainDm P x spec st ω u r')
        (plainBΦ rd P x spec st ω u r') r' k) (ω, u)).eval sha512).1 = plainBΦ rd P x spec st ω u r' k := by
  have hc := simCoins_agree P x spec st ω hw hm1 (plainDm P x spec st ω u r') (plainBΦ rd P x spec st ω u r') u r'
    hL k
  simp only [plainSimR, Claims.OC.eval]
  unfold plainMsgR
  rw [plainHl_length, ite_eq_right (Nat.succ_ne_zero k)]
  generalize plainHl P x spec st ω hw (plainDm P x spec st ω u r') (plainBΦ rd P x spec st ω u r') r' k = H at hc ⊢
  by_cases hkN : k < plainN x
  · rw [ite_eq_left (show k + 1 ≤ plainN x by omega)]
    conv_rhs => rw [plainBΦ, plainBR, ite_eq_left hkN]
    rw [hc 0 (Nat.succ_pos k)]
    have e : plainFrameR spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0)
        (depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u H))) (k + 1) =
        plainFrameR spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0) (plainDm P x spec st ω u r') (k + 1) :=
      plainFrameR_congr (by omega) fun t ht => depMsg_congr sha512 P x ω fun i hi => hc i (by omega)
    rw [e]
  · rw [ite_eq_right (show ¬ k + 1 ≤ plainN x by omega)]
    conv_rhs => rw [plainBΦ, plainBR, ite_eq_right hkN]
    have hkN' : k = plainN x := by omega
    have e : (VR rd).k x = (FlockDeploy.V false).k x := rfl
    have hr : ∀ i : Fin ((VR rd).k x), i.val + 1 < (VR rd).k x →
        simCoins x spec st u H i = plainΦ x spec st (u, r') i := by
      intro i hi
      have hi' : i.val < (FlockDeploy.V false).k x := by omega
      have := hc i.val (by omega)
      simp only [Claims.coinSeq, hi', ↓reduceDIte, Fin.eta] at this
      exact this
    by_cases ha : (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).out = .acc
    · rw [Claims.Recursion.run_eq_of_acc sha512 P (VR rd) x (decideFree_VR rd) ω hr ha, if_pos ha]
      conv_rhs => rw [plainAccR, decide_eq_true ha, ite_eq_left rfl, plainFsR]
      have hlenΦ : ((Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length =
          plainN x + 2 := by
        have := accLast_VR rd x _ _ (by simp) (FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω _ ha)
        omega
      exact congrArg filesCode (plainFilesR_congr rd x _ _ _ (by omega) fun i hi => hc i (by omega))
    · have hna : ¬ (Claims.run sha512 P (VR rd) x ω (simCoins x spec st u H)).out = .acc := fun h' => ha (by
        rw [Claims.Recursion.run_eq_of_acc sha512 P (VR rd) x (decideFree_VR rd) ω (fun i hi => (hr i hi).symm) h']
        exact h')
      rw [if_neg hna]
      conv_rhs => rw [plainAccR, decide_eq_false ha]
      rfl

/-- **The simulator's live outcome**: `P`'s roots, the live link points, the rounds of its submissions and the files
of its last message. -/
theorem plainSimR_liveOut (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (spec : SessionSpec) (st : Stmt) (ω : Ω) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x)
    (r' : Fin (plainN x + 2) → LCoin 65536) (hL : plainN x + 2 = (FlockDeploy.V false).k x) :
    liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512 (plainSimR rd P x spec st)
        x (ω, u) r' =
      ⟨(plainWR spec hw x).root ((P.next x [] ω).eval sha512).1,
        (ptsIf (plainWR spec hw x)).proj (Claims.coinSeq r' 0),
        foldS (plainDg spec)
          (plainSR x spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0) (plainDm P x spec st ω u r'))
          (fun i => Claims.coinSeq r' (i + 1)) (plainN x) (fun _ _ => []),
        (plainWR spec hw x).proofs (plainBΦ rd P x spec st ω u r' (plainN x))⟩ := by
  have hS : ∀ k < plainN x, (plainWR spec hw x).sub (plainBΦ rd P x spec st ω u r' k) =
      some (plainSR x spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0) (plainDm P x spec st ω u r') k) :=
    fun k hk => by
      show plainSub _ = _
      rw [plainBΦ, plainBR, ite_eq_left hk]
      exact plainSub_round _ _ _ (plainRep_lt spec st (k + 1)) (plainCnt_le x spec st (Nat.succ_ne_zero k))
  have hb : ∀ k ≤ plainN x, (((plainSimR rd P x spec st).next x
      ([((((plainSimR rd P x spec st).next x [] (ω, u)).eval sha512).1,
          (plainWR spec hw x).encPts ((ptsIf (plainWR spec hw x)).proj (Claims.coinSeq r' 0)))] ++
        liveH (plainWR spec hw x) (plainBΦ rd P x spec st ω u r')
          (plainSR x spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0) (plainDm P x spec st ω u r'))
          (fun i => Claims.coinSeq r' (i + 1)) k) (ω, u)).eval sha512).1 = plainBΦ rd P x spec st ω u r' k :=
    fun k hk => by
      rw [plainSimR_first rd P x spec st ω u,
        liveH_congr (plainWR spec hw x) (plainWire spec.mPts hw) rfl k _
          (plainSR x spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0) (plainDm P x spec st ω u r'))
          (plainS x spec st (plainDm P x spec st ω u r')) _ fun _ => rfl]
      exact plainSimR_msg rd P x spec st ω hw hm1 u r' hL k hk
  rw [liveOut_known plainN (plainDg spec) (plainWR spec hw) sha512 (plainSimR rd P x spec st) x (ω, u) r' _ _ hS hb,
    plainSimR_first] <;> rfl

/-! ## The schedule -/

section SchedR

variable (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt)
  (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (u : plainTape x)

/-- **The simulator's submission after `k + 1` messages**, `k < plainN x`, is on rep `plainRep (k + 1)`. -/
theorem plainSimR_sub (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ) (hl : hist.length = k + 1)
    (hk : k + 1 ≤ plainN x) :
    ∃ s : SubmitJ 1, (plainWR spec hw x).sub (((plainSimR rd P x spec st).next x hist (ω, u)).eval sha512).1 =
      some s ∧ s.rep.val = plainRep spec st (k + 1) := by
  have e : (plainWR spec hw x).sub (((plainSimR rd P x spec st).next x hist (ω, u)).eval sha512).1 =
      some ⟨0, ⟨plainRep spec st (k + 1), plainRep_lt spec st (k + 1)⟩,
        ⟨(plainFrameR spec st (Claims.coinSeq (simCoins x spec st u hist) 0)
          (depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hist))) (k + 1)).toArray⟩,
        plainCnt x spec st (k + 1)⟩ := by
    show plainSub _ = _
    simp only [plainSimR, Claims.OC.eval]
    unfold plainMsgR
    rw [hl, ite_eq_right (Nat.succ_ne_zero k), ite_eq_left hk]
    exact plainSub_round _ _ _ (plainRep_lt spec st (k + 1)) (plainCnt_le x spec st (Nat.succ_ne_zero k))
  exact ⟨_, e, rfl⟩

/-- **The simulator's bytes prover keeps `SchedK`** after `k + 1` messages, with `|plainRepNs| - k` rounds of rep 0
left. -/
theorem schedK_plainR (rootB : Array ByteArray) (pts : Fin 2 → Fin spec.mPts → 𝔽)
    (hR : (plainRepNs spec st).length ≤ plainN x) :
    ∀ (n : ℕ) (rs : Fin 1 → Fin 2 → List Flock.Round) (hist : List (Claims.Msg × Claims.Msg)) (k : ℕ),
      hist.length = k + 1 → k + n = plainN x →
      SchedK (plainDg spec) rootB pts n rs ((plainRepNs spec st).length - k)
        (Claims.Bridge.toStrat _ (roundsIf (plainWR spec hw x) (plainDg spec) rootB pts n rs)
          (Claims.Bridge.stratOf sha512 (plainSimR rd P x spec st) x (ω, u) _ _ hist none))
  | 0, _, _, k, _, hk => by
    show (plainRepNs spec st).length - k = 0
    omega
  | n + 1, rs, hist, k, hl, hk => by
    obtain ⟨s, ho, hrep⟩ := plainSimR_sub rd x spec st hw P ω u hist k hl (by omega)
    refine schedK_some (plainDg spec) rootB pts (plainWR spec hw x) sha512 (plainSimR rd P x spec st) x (ω, u) n rs _
      hist (((plainSimR rd P x spec st).next x hist (ω, u)).eval sha512).1 _ ((plainWR spec hw x).sub_le _) s ho ?_
      fun c => ?_
    · apply Fin.ext
      rw [hrep]
      unfold plainRep
      by_cases hkR : k < (plainRepNs spec st).length
      · rw [ite_eq_left (show k + 1 - 1 < (plainRepNs spec st).length by omega),
          ite_eq_right (show (plainRepNs spec st).length - k ≠ 0 by omega)]
        rfl
      · rw [ite_eq_right (show ¬ k + 1 - 1 < (plainRepNs spec st).length by omega),
          ite_eq_left (show (plainRepNs spec st).length - k = 0 by omega)]
        rfl
    · rw [show (plainRepNs spec st).length - k - 1 = (plainRepNs spec st).length - (k + 1) by omega]
      exact schedK_plainR rootB pts hR n _ _ (k + 1) (by simp [hl]) (by omega)

/-- **The simulator's live strategy submits `|plainRepNs|` rounds on rep 0 first** at every link point. -/
theorem plainSimR_schedK (pts : Fin 2 → Fin spec.mPts → 𝔽) (hR : (plainRepNs spec st).length ≤ plainN x) :
    SchedL (plainDg spec)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        (plainSimR rd P x spec st) x (ω, u)).1 pts (plainN x) (fun _ _ => []) (plainRepNs spec st).length
      ((liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        (plainSimR rd P x spec st) x (ω, u)).2 pts) :=
  schedL_of_schedK' (plainDg spec) _ pts
    (liveStrat_fst (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
      (plainSimR rd P x spec st) x (ω, u)) (plainN x) _ _ _ _
    (liveStrat_snd_heq (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
      (plainSimR rd P x spec st) x (ω, u) pts)
    (schedK_plainR rd x spec st hw P ω u _ pts hR (plainN x) _ _ 0 rfl (Nat.zero_add _))

end SchedR

/-- **The simulator submits `repRounds` rounds on rep 0 first** at a plain statement of one table whose schedule has
at least two levels. -/
theorem plainSimR_schedL (rd : Claims.Msg → Option FinalP) {x : Claims.Msg} {ins : Ins} {spec : SessionSpec}
    {st : Stmt} (hset : setupOf false x = some (ins, spec, #[st])) {sch : Accounting.Schedule}
    (hsch : Accounting.fast100 st.m = some sch) (h2 : 2 ≤ sch.levels.length) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (u : plainTape x) (pts : Fin 2 → Fin spec.mPts → 𝔽) :
    SchedL (plainDg spec)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        (plainSimR rd P x spec st) x (ω, u)).1 pts (plainN x) (fun _ _ => []) (repRounds false spec st)
      ((liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        (plainSimR rd P x spec st) x (ω, u)).2 pts) := by
  obtain ⟨sF, hF, rfl⟩ := exec_sched hsch
  simp only [FlockSoundness.Refine.schedOf, List.length_map, Array.length_toList] at h2
  have hle := rep_le_rounds (zk := false) (spec := spec) (sts := #[st]) (st := st) (by first | rfl | simp)
  have hk : (FlockDeploy.V false).k x = 2 + rounds false spec #[st] := by simp only [FlockDeploy.V, hset]
  have hlen := plainRepNs_length (spec := spec) hF h2
  rw [← hlen]
  exact plainSimR_schedK rd x spec st hw P ω u pts (by unfold plainN; omega)

/-! ## The simulation -/

/-- **The simulator's submissions to rep `q`**: rep `0`'s rounds first, then rep `1`'s, `n₀` each. -/
theorem plainSR_filter (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (c₁ : Coin) (dm : ℕ → Claims.Msg) {n₀ : ℕ}
    (hN : plainN x = n₀ + n₀) (hn₀ : (plainRepNs spec st).length = n₀) (q : Fin 2) :
    ((List.range (plainN x)).filter fun k =>
        decide ((plainSR x spec st c₁ dm k).tab = 0 ∧ (plainSR x spec st c₁ dm k).rep = q)) =
      (List.range n₀).map fun i => q.val * n₀ + i := by
  have hrep : ∀ k, ((plainSR x spec st c₁ dm k).rep).val = if k < n₀ then 0 else 1 := fun k => by
    show plainRep spec st (k + 1) = _
    unfold plainRep
    rw [hn₀, Nat.add_sub_cancel]
  have htab : ∀ k, (plainSR x spec st c₁ dm k).tab = 0 := fun k => Subsingleton.elim _ _
  rw [hN, List.range_add, List.filter_append, List.filter_map]
  rcases q with ⟨_ | _ | q, hq⟩
  · rw [List.filter_eq_self.2 fun k hk => ?_, List.filter_eq_nil_iff.2 fun k hk => ?_]
    · simp
    · have := hrep (n₀ + k)
      simp only [Function.comp_apply, htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this]
      simp
    · have := hrep k
      simp only [htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this, ite_eq_left (List.mem_range.1 hk)]
  · rw [List.filter_eq_nil_iff.2 fun k hk => ?_, List.filter_eq_self.2 fun k hk => ?_]
    · simp
    · have := hrep (n₀ + k)
      simp only [Function.comp_apply, htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this]
      simp
    · have := hrep k
      simp only [htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this, ite_eq_left (List.mem_range.1 hk)]
      simp
  · omega

/-- **The simulation at `VR rd`** (`PlainSimRunR`): where `P`'s run at `plainΦ`'s coins decides and accepts, at a
plain statement of one table with two link points, the simulator's live outcome is the decision's view: the link's
roots, the link coin's points, the record's reps and the decided files. -/
theorem plainSimRunR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    {ins : Ins} {spec : SessionSpec} {st : Stmt} (hset : setupOf false x = some (ins, spec, #[st]))
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (hp : spec.points = 2) :
    PlainSimRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec hw) (plainSimR rd P x spec st)
      (plainΦ x spec st) := by
  intro u r' hacc s spec' sts' link c₁ mids f hact
  have hso := (actR_setupOf hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at hso
  obtain ⟨e₁, e₂, e₃⟩ := hso
  subst s spec' sts'
  have hL := plainN_add hset
  have hacc' := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω _ hacc
  obtain ⟨hv, hc, -⟩ := decideR_sha512 hact hacc'
  have hd0 := actR_decide hact
  have hdep : (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst =
      (List.range (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).msgs.length).map
        (depMsg sha512 P x ω (Claims.coinSeq (plainΦ x spec st (u, r')))) :=
    stepWith_dep sha512 P (VR rd) x ω _ (answersCoin_VR rd sha512 x) _ [] [] _ rfl rfl rfl
  have hK : ((Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length =
      plainN x + 2 := by
    have h := accLast_VR rd x _ _ (by simp) hacc'
    have e : (VR rd).k x = (FlockDeploy.V false).k x := rfl
    omega
  have hK' : (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).msgs.length = plainN x + 2 := by
    rw [List.length_map] at hK
    exact hK
  have hd := hd0
  rw [hK, hdep, hK', range_map_ends2, range_map_ends2] at hd
  obtain ⟨v, c₁', r₀, rss, fl, link', pf, hv', hc₁, hr₀, hrss, hlk, hpf, hdd⟩ := decR_some hd
  rw [hset] at hv'
  obtain rfl := Option.some.inj hv'
  simp only [List.head?_cons, Option.some.injEq, List.headD_cons] at hc₁ hr₀
  simp only [List.tail_cons, List.dropLast_concat] at hrss hpf
  obtain ⟨-, -, -, hlink, hc1, hmids, -⟩ := DecR.mk.inj hdd
  simp only [List.tail_cons, List.dropLast_concat] at hmids
  subst link'
  subst c₁'
  subst c₁
  obtain ⟨r₀', hr0, hr₀e⟩ := Option.map_eq_some_iff.1 hr₀
  have hr0' : rowsOf (plainDm P x spec st ω u r' 0) = some r₀' := hr0
  have hr₀' : rowsN (plainDm P x spec st ω u r' 0) = r₀ := by
    simp only [rowsN, rowsD, hr0', Option.getD_some, hr₀e]
  clear hr₀e
  subst r₀
  obtain ⟨hrssL, hget⟩ := mapM_some_get hrss
  have hN : rss.length = plainN x := by simpa using hrssL
  have hrows : ∀ i (hi : i < rss.length), rss[i] = rowsD (plainDm P x spec st ω u r' (i + 1)) := fun i hi => by
    have := hget i (by simpa [hN] using hi) hi
    simp only [List.getElem_map, List.getElem_range] at this
    simp only [rowsD, plainDm, this, Option.getD_some]
  obtain ⟨L, hLdef⟩ : ∃ L, (RecDeploy.repSchedAt spec st).length = L := ⟨_, rfl⟩
  obtain ⟨hlen, p₀, T₀, p₁, T₁, hf₀, hf₁, hpf2⟩ := readPs_one hLdef hpf
  have hsc : ∀ q < 2, SchedOf (repCtx spec #[st] (Claims.coinSeq (plainΦ x spec st (u, r')) 0)
      (rowsN (plainDm P x spec st ω u r' 0)) [] q) :=
    fun q hq => schedOf_repCtx_of_checksAtJ id With.std hv hc _ [] (by simpa using hq)
  obtain ⟨hpflen, hframe⟩ := pf_frames (dm := plainDm P x spec st ω u r') hlen hrows hsc hf₀ hf₁ hpf2
  obtain ⟨-, lv, hlv, hcu⟩ := checksAtJ_ok hc
  have hfa := (Claims.Bridge.Custody.custody_ok hcu).2.2.2
  have hD : liveDJ spec (jAt sha512 spec #[st] (Claims.coinSeq (plainΦ x spec st (u, r')) 0) mids f) = lv := by
    simp [liveDJ, hlv]
  obtain ⟨hsz', hns⟩ := streams_ns_plainJK id With.std hv hlv
  have hsz : lv.streams.size = 2 := by simpa using hsz'
  have hstr : ∀ (q : ℕ) (hq : q < lv.streams.size), lv.streams[q].toList.map Round.n = plainRepNs spec st ∧
      ∀ y ∈ lv.streams[q].toList, 1 ≤ y.n ∧ y.n ≤ 65536 := fun q hq => by
    refine ⟨?_, liveOfJ_nK id With.std hv hlv q hq⟩
    rw [hns q hq]
    have : q / 2 = 0 := by omega
    simp [this]
  have hflat := flat_two lv.streams hsz
  have hl0 : (lv.streams[0]'(by omega)).toList.length = (plainRepNs spec st).length := by
    rw [← (hstr 0 (by omega)).1, List.length_map]
  have hl1 : (lv.streams[1]'(by omega)).toList.length = (plainRepNs spec st).length := by
    rw [← (hstr 1 (by omega)).1, List.length_map]
  have hmlen : mids.length = L + L := by
    rw [hmids, List.length_zip, hpflen, List.length_map, List.length_range]
    omega
  have hn₀ : (plainRepNs spec st).length = L := by
    have := hfa.length_eq
    rw [hflat, List.length_append, hl0, hl1, hmlen] at this
    omega
  have hlenN : plainN x = (plainRepNs spec st).length + (plainRepNs spec st).length := by omega
  rw [plainSimR_liveOut rd P x spec st ω hw hm1 u r' hL]
  refine liveOutJ_ext' ?_ ?_ ?_ ?_
  · show (((setupOf false x).bind fun v =>
        ((rowsOf ((P.next x [] ω).eval sha512).1).map (List.map nodeOf)).bind (linkR v.2.2)).getD
        default).1.toArray = link.1.toArray
    have hr1 : (rowsOf ((P.next x [] ω).eval sha512).1).map (List.map nodeOf) =
        some (rowsN (plainDm P x spec st ω u r' 0)) := by
      show (rowsOf (plainDm P x spec st ω u r' 0)).map (List.map nodeOf) = _
      simp only [rowsN, rowsD, hr0', Option.map_some, Option.getD_some]
    have hlk' : linkR #[st] (rowsN (plainDm P x spec st ω u r' 0)) = some link := hlk
    first
      | (simp only [hset, Option.some_bind, hr1, hlk', Option.getD_some]; done)
      | simp [hset, hr1, hlk']
  · rw [ptsIf_proj]
    funext k t
    have hkt : t.val + spec.mPts * k.val < 2 * spec.mPts := by
      have := k.isLt
      have := t.isLt
      rcases k with ⟨_ | _ | _, _⟩ <;> simp <;> omega
    show ptsE spec.mPts (takeE (2 * spec.mPts) (Claims.coinSeq r' 0)) k t =
      Refine.fOf ((pointsOf spec (Claims.coinSeq (plainΦ x spec st (u, r')) 0))[k.val]!)[t.val]!
    have e1 : ptsE spec.mPts (takeE (2 * spec.mPts) (Claims.coinSeq r' 0)) k t =
        takeE (2 * spec.mPts) (Claims.coinSeq r' 0) ⟨t.val + spec.mPts * k.val, hkt⟩ := rfl
    have e2 : k.val * spec.mPts + t.val = t.val + spec.mPts * k.val := by rw [Nat.mul_comm, Nat.add_comm]
    have hk2 : k.val < 2 := k.isLt
    simp only [pointsOf, hp]
    rw [e1, getElem!_pos _ k.val (by simp [hk2]), Array.getElem_map, Array.getElem_range,
      getElem!_pos _ t.val (by simp), Array.getElem_map, Array.getElem_range, e2]
    have he : t.val + spec.mPts * k.val < 65536 := by omega
    simp only [takeE, Coin.at, he, ↓reduceDIte]
    have hc0 : t.val + spec.mPts * k.val < plainCnt x spec st 0 := by
      unfold plainCnt
      simp only [↓reduceIte]
      exact lt_max_of_lt_right hkt
    exact (plainΦ_entry (j := 0) hL u r' (by omega) ⟨_, he⟩ hc0).symm
  · dsimp only [viewOfJ]
    rw [hD]
    funext j q
    obtain rfl : j = 0 := Subsingleton.elim _ _
    have hq : q.val < lv.streams.size := by omega
    have hsq := hstr q.val hq
    have hlq : (lv.streams[q.val]'hq).toList.length = (plainRepNs spec st).length := by
      rw [← hsq.1, List.length_map]
    rw [foldS_eq, plainSR_filter x spec st _ _ hlenN rfl q, List.nil_append, List.map_map,
      show 2 * (0 : Fin 1).val + q.val = q.val by simp, getElem!_pos lv.streams q.val hq]
    refine List.ext_getElem (by simp [hlq]) fun i h₁ h₂ => ?_
    have hi : i < (plainRepNs spec st).length := by simpa using h₁
    simp only [List.getElem_map, List.getElem_range, Function.comp_apply]
    set k := q.val * (plainRepNs spec st).length + i with hk
    have hkN : k < plainN x := by
      have := q.isLt
      rw [hlenN, hk]
      rcases q with ⟨_ | _ | _, _⟩ <;> simp <;> omega
    have hkf : k < (lv.streams.toList.flatMap (·.toList)).length := by
      rw [hflat, List.length_append, hl0, hl1]
      omega
    have hkm : k < mids.length := by rw [hmlen]; omega
    have hR := hfa.get hkf hkm
    simp only [List.get_eq_getElem] at hR
    have hfk : (lv.streams.toList.flatMap (·.toList))[k] = (lv.streams[q.val]'hq).toList[i]'(by omega) := by
      simp only [hflat]
      rcases q with ⟨_ | _ | _, hq2⟩
      · simp only [Nat.zero_mul, Nat.zero_add] at hk
        simp only [hk]
        rw [List.getElem_append_left (by omega)]
      · simp only at hk
        simp only [hk]
        rw [List.getElem_append_right (by omega)]
        have h0s : (lv.streams[0]'(by omega)).size = (plainRepNs spec st).length := by
          rw [← Array.length_toList]; exact hl0
        simp [h0s]
      · omega
    have hpk : pf.2[k]'(by omega) = plainFrameR spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0)
        (plainDm P x spec st ω u r') (k + 1) :=
      Option.some.inj ((List.getElem?_eq_getElem (by omega)).symm.trans (hframe hn₀ k (by omega)))
    have hmk : mids[k] = (plainFrameR spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0)
        (plainDm P x spec st ω u r') (k + 1), Claims.coinSeq (plainΦ x spec st (u, r')) (k + 1)) := by
      simp only [hmids, List.getElem_zip, List.getElem_map, List.getElem_range, hpk]
    rw [hfk, hmk] at hR
    obtain ⟨hby, hco⟩ := hR
    have hn : ((lv.streams[q.val]'hq).toList[i]'(by omega)).n = (plainRepNs spec st)[i] := by
      have := congrArg (fun l => l[i]?) hsq.1
      simp only [List.getElem?_map] at this
      rw [List.getElem?_eq_getElem (by omega), List.getElem?_eq_getElem hi] at this
      simpa using this
    have hbd := hsq.2 ((lv.streams[q.val]'hq).toList[i]'(by omega)) (List.getElem_mem _)
    have hcnt : plainCnt x spec st (k + 1) = (plainRepNs spec st)[i] := by
      refine plainCnt_eq hkN ?_ (hn ▸ hbd.1) (hn ▸ hbd.2)
      unfold plainNsOne
      rw [List.getD_eq_getElem _ _ (by rw [List.length_append]; omega)]
      rcases q with ⟨_ | _ | _, hq2⟩
      · simp only [Nat.zero_mul, Nat.zero_add] at hk
        simp only [hk]
        rw [List.getElem_append_left hi]
      · simp only at hk
        simp only [hk]
        rw [List.getElem_append_right (by omega)]
        simp
      · omega
    generalize hrd : (lv.streams[q.val]'hq).toList[i]'(by omega) = y at hby hco hn hbd
    rcases y with ⟨n, sha, coins, bytes⟩
    simp only at hby hco hn hbd
    subst hby hco
    show Refine.roundOf spec.dg ⟨(plainFrameR spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0)
        (plainDm P x spec st ω u r') (k + 1)).toArray⟩ (plainCnt x spec st (k + 1))
        (takeE (plainCnt x spec st (k + 1)) (Claims.coinSeq r' (k + 1))) = _
    rw [hcnt, ← hn]
    unfold Refine.roundOf Claims.Bridge.Custody.withSha
    simp only [Option.getD_some, Round.mk.injEq, true_and, and_true]
    exact (vals_takeE _ _ hbd.2 fun e he => plainΦ_entry hL u r' (by omega) e
      (by rw [hcnt, ← hn]; exact he)).symm
  · show filesDecode (plainBR x spec st (Claims.coinSeq (plainΦ x spec st (u, r')) 0) (plainDm P x spec st ω u r')
        (plainAccR rd P x spec st ω u r') (plainFsR rd P x spec st ω u r') (plainN x)) =
      FlockDeploy.filesAt (tabsOf #[st]) (pointsOf spec (Claims.coinSeq (plainΦ x spec st (u, r')) 0)) f.ps
    rw [plainBR, ite_eq_right (lt_irrefl _), plainAccR, decide_eq_true hacc, ite_eq_left rfl, filesDecode_code]
    unfold plainFsR plainFilesR
    rw [hd0] <;> rfl

/-- **The simulator sends no files where `VR rd` refuses.** -/
theorem plainSimR_nil (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω)
    {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} (st : Stmt) (hset : setupOf false x = some (ins, spec, sts))
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hna : (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).out ≠ .acc) :
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
      (plainSimR rd P x spec st) x (ω, u) r').proofs = #[] := by
  rw [plainSimR_liveOut rd P x spec st ω hw hm1 u r' (plainN_add hset)]
  show filesDecode (plainBR x spec st _ _ (plainAccR rd P x spec st ω u r') _ (plainN x)) = #[]
  rw [plainBR, ite_eq_right (lt_irrefl _), plainAccR, decide_eq_false hna]
  exact filesDecode_nil

/-! ## At every statement -/

/-- **A simulator at every prover at `VR rd`** (`PlainSimAtR`): `plainSimR`. -/
theorem plainSimAtR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) : PlainSimAtR rd x := by
  intro s hs hin ins spec sts hset st ht Ω P ω
  obtain rfl := sts_of_tables hset ht
  have hm : s.m = st.m := by
    have := parse_m hs hset (by simp : 0 < (#[st] : Array Stmt).size)
    simpa using this
  have h22 : 22 ≤ st.m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : st.m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, -, -⟩ := plain_sched h22 h35
  obtain ⟨-, hm1, hp⟩ := plain_mPts hs hin hset
  refine ⟨plainSimR rd P x spec st, ?_⟩
  unfold PlainSimR
  refine ⟨fun u => ?_, fun u pts => plainSimR_schedL rd hset hsch (by simp) _ P ω u pts,
    plainSimRunR rd P x ω hset _ hm1 hp, fun u r' hna => plainSimR_nil rd P ω st hset _ hm1 u r' hna⟩
  rw [liveStrat_fst, plainSimR_first] <;> rfl

/-- **`PlainRdInR` at every statement.** -/
theorem plainRdInR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) : PlainRdInR rd x :=
  plainRdInR_of_sim rd (plainSimAtR rd x)

/-- **`RdAllAt` at the plain record at the deployed reader.** -/
theorem rdAllAt_RS (rd : Claims.Msg → Option FinalP) :
    RdAllAt (deployPlainR Readable.relation rd) (fun _ => plainRawLevelColsRV rd) :=
  rdAllAt_R_of_sim rd fun x _ => plainSimAtR rd x

end FlockDeploy.KS
