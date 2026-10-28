import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.Recursion.NextRTables
import Proofs.Flock.Claims.Recursion.BoundRun
import Proofs.Flock.Claims.Recursion.DeployHonest
import Proofs.Flock.Claims.Recursion.DeployedR

/-!
# The honest prover's run passes `IVStar` at the deployed reader (`bdR`, `nextR`)

`RecursiveKS` at `nextR` is non-vacuous only if the inner verifier the compiled record's inner view runs, `IVStar` of
`bdR (nextR rd)`, accepts what an honest prover sends. That prover commits each round's rows under salts of its choice
(hm96's `b ‖ c` at the pinned key, `Bound.commitRow`), one commit string a row (`Bound.commitRound`), and its last
message is one that `final` reads as those rows with their salts, then the element-level last message
(`Bound.honestAs`). So:

* **Generic, at every bound record** (`Bound.ivstar_honest`, `Bound.run_honest_acc`): each honest commit string opens to
  its row under its salt (`Bound.opens_commitRow`, at any hash of 64-byte outputs), so `IVStar` accepts exactly where
  `final` reads the last message as the rows and salts, the rounds carry the schedule's counts (`nAt`) and the
  element-level verifier accepts the opened messages (`agrees`).
* **At `bdR (nextR rd)` with `join := joinR`** (`agrees_bdR`, `ivstarR_honest_run`): every opened round before the last
  is `joinR`'s, tag `0`, which the deployed plain verifier answers with its coin (`act_coin_of_tag`), so `agrees` is
  `nextR rd` accepting the opened messages at the last round.
* **At the honest session** (`honestRounds_nAt`, `ivstarR_honest_session`, `ivstarR_honest_files`): the honest rows
  (`honestRounds`: round 0's caps, then each rep's rounds, `NextRHonest`'s `honestMsgs` opened) carry the schedule's
  counts (`RecDeploy.nAtSched`) and the deployed plain verifier's rounds, from `FitsRep` at each rep (`repFits`). So
  the honest run passes `IVStar` wherever `final` reads its last message as the salted honest rows and `nextR rd`
  accepts the honest element-level messages (the honest verdict, HONEST-VERDICT's), in `InnerAccepts`' form too
  (`innerAccepts_honest`), which recursion completeness at `deployRec` takes (`RecDeploy.vstar_complete`).
* **At the deployed `final := finalR`** (`DeployedR.ivstar_honest_filesR`, `DeployedR.innerAccepts_honestR`): the honest
  last message `lastOR` lays out each row then its salt, the element-level last message, then the trailer of the row
  lengths, and `finalR` reads it back as the salted honest rows (`finalR_lastOR`).

What stays a hypothesis here: `nextR rd` accepting the honest messages (HONEST-VERDICT's). It is not a named
assumption: it is an equation about the code at the honest run.
-/

namespace Claims.Recursion

open Claims

theorem xorMask_length (b : Msg) (m : Fin 64 → UInt8) : (xorMask b m).length = b.length :=
  List.length_mapIdx

/-- **Unmasking is an involution** at each salt's mask. -/
theorem xorMask_xorMask (b : Msg) (m : Fin 64 → UInt8) : xorMask (xorMask b m) m = b := by
  refine List.ext_getElem (by simp only [xorMask_length]) fun i h₁ h₂ => ?_
  simp only [xorMask, List.getElem_mapIdx]
  rw [UInt8.xor_assoc, UInt8.xor_self, UInt8.xor_zero]

/-- `allOC` returns `true` where each of its programs does. -/
theorem eval_allOC_true (H : Msg → Msg) : ∀ (ps : List (OC Bool)), (∀ p ∈ ps, (p.eval H).1 = true) →
    ((allOC ps).eval H).1 = true
  | [], _ => rfl
  | p :: ps, h => by
    have hp := h p List.mem_cons_self
    simp only [allOC, eval_bindOC, hp, ↓reduceIte]
    exact eval_allOC_true H ps fun q hq => h q (List.mem_cons_of_mem _ hq)

theorem map_range_getD {α : Type} (l : List α) (d : α) : (List.range l.length).map (fun i => l.getD i d) = l := by
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  simp [List.getD_eq_getElem?_getD, h₂]

section Honest

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **An honest commit string** at the bound key: hm96's `b ‖ c` for `row` under the salt `y`, `b = H(rowIn row) ⊕ Φy`
and `c = H(p_salt ‖ y)`. -/
def Bound.commitRow (Bd : Bound Coin Col T) (H : Msg → Msg) (row : Msg) (y : Fin Bd.saltLen → UInt8) : Msg :=
  xorMask (H (Bd.rowIn row)) (Bd.mask y) ++ H (Bd.saltPrefix ++ List.ofFn y)

/-- **An honest round message**: its rows' commit strings under their salts, in order. -/
def Bound.commitRound (Bd : Bound Coin Col T) (H : Msg → Msg) (os : List (Msg × (Fin Bd.saltLen → UInt8))) : Msg :=
  (os.map fun o => Bd.commitRow H o.1 o.2).flatten

/-- **What the honest prover sends `IVStar`**: each round's commit strings of its rows under their salts (`os`), then a
last message `lastO`. -/
def Bound.honestAs (Bd : Bound Coin Col T) (H : Msg → Msg) (os : List (List (Msg × (Fin Bd.saltLen → UInt8))))
    (lastO : Msg) : List Msg :=
  os.map (Bd.commitRound H) ++ [lastO]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Bound.commitRow_length (Bd : Bound Coin Col T) {H : Msg → Msg} (h64 : Out64 H) (row : Msg)
    (y : Fin Bd.saltLen → UInt8) : (Bd.commitRow H row y).length = 128 := by
  rw [Bound.commitRow, List.length_append, xorMask_length, h64, h64]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Bound.commitRound_length (Bd : Bound Coin Col T) {H : Msg → Msg} (h64 : Out64 H) :
    ∀ os : List (Msg × (Fin Bd.saltLen → UInt8)), (Bd.commitRound H os).length = 128 * os.length
  | [] => rfl
  | o :: os => by
    show (Bd.commitRow H o.1 o.2 ++ Bd.commitRound H os).length = _
    rw [List.length_append, Bd.commitRow_length h64, Bd.commitRound_length h64 os, List.length_cons]
    ring

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **An honest commit string opens to its row under its salt**, at any hash of 64-byte outputs. -/
theorem Bound.opens_commitRow (Bd : Bound Coin Col T) {H : Msg → Msg} (h64 : Out64 H) (row : Msg)
    (y : Fin Bd.saltLen → UInt8) : ((Bd.opens (Bd.commitRow H row y) row y).eval H).1 = true := by
  have hl : (xorMask (H (Bd.rowIn row)) (Bd.mask y)).length = 64 := by rw [xorMask_length, h64]
  have h1 : (Bd.commitRow H row y).take 64 = xorMask (H (Bd.rowIn row)) (Bd.mask y) := by
    rw [Bound.commitRow, List.take_left' hl]
  have h2 : (Bd.commitRow H row y).drop 64 = H (Bd.saltPrefix ++ List.ofFn y) := by
    rw [Bound.commitRow, List.drop_left' hl]
  simp only [Bound.opens, OC.eval, h1, h2, xorMask_xorMask, and_self, decide_true]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- Row `i`'s commit string is the `i`-th 128 bytes of an honest round message. -/
theorem Bound.commitRound_block (Bd : Bound Coin Col T) {H : Msg → Msg} (h64 : Out64 H) :
    ∀ (os : List (Msg × (Fin Bd.saltLen → UInt8))) (i : ℕ) (hi : i < os.length),
      ((Bd.commitRound H os).drop (128 * i)).take 128 = Bd.commitRow H os[i].1 os[i].2
  | [], i, hi => absurd hi (Nat.not_lt_zero _)
  | o :: os, 0, _ => by
    show ((Bd.commitRow H o.1 o.2 ++ Bd.commitRound H os).drop 0).take 128 = _
    rw [List.drop_zero, List.take_left' (Bd.commitRow_length h64 _ _)]
    rfl
  | o :: os, i + 1, hi => by
    show ((Bd.commitRow H o.1 o.2 ++ Bd.commitRound H os).drop (128 * (i + 1))).take 128 = _
    rw [show 128 * (i + 1) = (Bd.commitRow H o.1 o.2).length + 128 * i by
      rw [Bd.commitRow_length h64]; ring, List.drop_append, List.drop_eq_nil_of_le (Nat.le_add_right _ _), List.nil_append,
      Nat.add_sub_cancel_left]
    exact Bd.commitRound_block h64 os i (by simpa using hi)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **An honest round message opens to its rows under their salts.** -/
theorem Bound.opensRound_commitRound (Bd : Bound Coin Col T) {H : Msg → Msg} (h64 : Out64 H)
    (os : List (Msg × (Fin Bd.saltLen → UInt8))) :
    ((Bd.opensRound (Bd.commitRound H os) os).eval H).1 = true := by
  unfold Bound.opensRound
  rw [ite_eq_left (Bd.commitRound_length h64 os)]
  refine eval_allOC_true H _ fun p hp => ?_
  obtain ⟨i, hi, rfl⟩ := List.mem_mapIdx.1 hp
  rw [Bd.commitRound_block h64 os i hi]
  exact Bd.opens_commitRow h64 _ _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Bound.zip_commitRound (Bd : Bound Coin Col T) (H : Msg → Msg) :
    ∀ os : List (List (Msg × (Fin Bd.saltLen → UInt8))),
      ((os.map (Bd.commitRound H)).zip os).map (fun p => Bd.opensRound p.1 p.2) =
        os.map fun o => Bd.opensRound (Bd.commitRound H o) o
  | [] => rfl
  | o :: os => by simp only [List.map_cons, List.zip_cons_cons, Bd.zip_commitRound H os]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Bound.opened_length (Bd : Bound Coin Col T) (os : List (List (Msg × (Fin Bd.saltLen → UInt8)))) (last : Msg) :
    (Bd.opened os last).length = os.length + 1 := by
  simp [Bound.opened]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- Every opened round before the last is `join` of its rows. -/
theorem Bound.opened_getElem (Bd : Bound Coin Col T) (os : List (List (Msg × (Fin Bd.saltLen → UInt8)))) (last : Msg)
    {j : ℕ} (hj : j < os.length) :
    (Bd.opened os last)[j]'(by rw [Bd.opened_length]; omega) = Bd.join (os[j].map Prod.fst) := by
  simp only [Bound.opened]
  rw [List.getElem_append_left (by simpa using hj), List.getElem_map]

/-- **`IVStar` accepts the honest prover's messages**: where `final` reads the last message as the rows and salts `os`,
then `last`, the rounds number `DE`'s, each carries the schedule's count of rows (`nAt`), and `DE`'s verifier accepts
the opened messages at the coins (`agrees`). -/
theorem Bound.ivstar_honest (Bd : Bound Coin Col T) {H : Msg → Msg} (h64 : Out64 H) {x : Msg}
    {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last lastO : Msg} {cs : List Coin}
    (hfin : Bd.final lastO = some (os, last)) (hk : os.length + 1 = Bd.DE.V.k x)
    (hn : ∀ j < os.length, (os.getD j []).length = Bd.nAt x j)
    (hag : ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true) :
    ((Bd.IVStar.next x (Bd.honestAs H os lastO) cs).eval H).1 = .acc := by
  have hl : (Bd.honestAs H os lastO).length = os.length + 1 := by simp [Bound.honestAs]
  have hlast : (Bd.honestAs H os lastO).getLastD [] = lastO := by simp [Bound.honestAs]
  have hdrop : (Bd.honestAs H os lastO).dropLast = os.map (Bd.commitRound H) := by simp [Bound.honestAs]
  have hopen : ((allOC (os.map fun o => Bd.opensRound (Bd.commitRound H o) o)).eval H).1 = true :=
    eval_allOC_true H _ fun p hp => by
      obtain ⟨o, -, rfl⟩ := List.mem_map.1 hp
      exact Bd.opensRound_commitRound h64 o
  have hlt : ¬ (Bd.honestAs H os lastO).length < Bd.DE.V.k x := by omega
  have hc : os.length + 1 = (Bd.honestAs H os lastO).length ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j :=
    ⟨hl.symm, hn⟩
  simp only [Bound.IVStar, hlt, ↓reduceIte, hlast, hfin]
  rw [ite_eq_left hc, hdrop, Bd.zip_commitRound H os, eval_bindOC, hopen]
  simp only [↓reduceIte]
  rw [eval_bindOC, hag]
  rfl

/-- **The honest prover's run passes `IVStar`**: `IVStar`'s run on the honest messages, at coins `cs` numbering its
rounds, accepts, under `Bound.ivstar_honest`'s conditions. -/
theorem Bound.run_honest_acc (Bd : Bound Coin Col T) {H : Msg → Msg} (h64 : Out64 H) {x : Msg}
    {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last lastO : Msg} {cs : List Coin}
    (hfin : Bd.final lastO = some (os, last)) (hk : os.length + 1 = Bd.DE.V.k x)
    (hn : ∀ j < os.length, (os.getD j []).length = Bd.nAt x j) (hcs : cs.length = Bd.DE.V.k x)
    (hag : ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true) :
    (run H (replay (Bd.honestAs H os lastO)) Bd.IVStar x () fun i => cs.getD i.val default).out = .acc := by
  rw [Bd.run_ivstar H x _ _ (by omega)]
  show (Bd.decAt H x (Bd.honestAs H os lastO) fun i => cs.getD i.val default).1 = .acc
  have hl : (Bd.honestAs H os lastO).length = Bd.DE.V.k x := by simp [Bound.honestAs, hk]
  have e1 : (List.range (Bd.DE.V.k x)).map (fun i => (Bd.honestAs H os lastO).getD i []) =
      Bd.honestAs H os lastO := by
    rw [← hl]
    exact map_range_getD _ _
  have e2 : (List.range (Bd.DE.V.k x)).map
      (coinSeq fun i : Fin (Bd.IVStar.k x) => cs.getD i.val default) = cs := by
    conv_rhs => rw [← map_range_getD cs default]
    rw [hcs]
    refine List.map_congr_left fun i hi => ?_
    have hi' : i < Bd.IVStar.k x := List.mem_range.1 hi
    simp only [coinSeq, hi', ↓reduceDIte]
  unfold Bound.decAt
  rw [e1, e2]
  exact Bd.ivstar_honest h64 hfin hk hn hag

end Honest

end Claims.Recursion

namespace FlockDeploy.KS

open Flock FlockComplete
open Claims.Recursion

local notation "𝔽" => FlockLevel3.GF128

/-- **SHA-512's outputs are 64 bytes.** -/
theorem sha512_out64 : Claims.Out64 sha512 := fun m => by
  rw [sha512, Claims.toMsg, Array.length_toList, ByteArray.size_data]
  exact Sha512.hash_size _

theorem getLast?_take_succ {α : Type} {l : List α} {j : ℕ} (hj : j < l.length) :
    (l.take (j + 1)).getLast? = some l[j] := by
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, List.getLast?_concat]

/-- **A round of tag `0` is answered with its coin**: at a statement that sets up, the deployed plain verifier answers
the prefix ending at round `j` with coin `j` where round `j`'s message is of tag `0`. -/
theorem act_coin_of_tag {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin} (hx : (setupOf false x).isSome)
    {j : ℕ} (hj : j < ms.length) (hjc : j < cs.length) {t : Claims.Msg} (h0 : ms[j] = 0 :: t) :
    act false x (ms.take (j + 1)) (cs.take (j + 1)) = .coin cs[j] := by
  obtain ⟨⟨s, spec, sts⟩, hv⟩ := Option.isSome_iff_exists.1 hx
  have h1 : (ms.take (j + 1)).getLast? = some (0 :: t) := by rw [getLast?_take_succ hj, h0]
  have h2 : (cs.take (j + 1)).getLast? = some cs[j] := getLast?_take_succ hjc
  simp only [act, hv, h1, h2]

/-- **The deployed bound record at `nextR rd`, at any columns**: `IVStar` reads no column, so its honest acceptance
holds at `bdR`'s (`colsRV rd`) and at `DeployedR.bdR`'s alike. -/
noncomputable abbrev bdC (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List 𝔽)) (wE : Claims.Msg → ℝ)
    (join : List Claims.Msg → Claims.Msg) (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ) :
    Claims.Recursion.Bound Coin (List 𝔽) Claims.Flock.PlainTrace :=
  RecDeploy.bdDeployed rel (nextR rd) cols wE join rowIn final w

/-- **`agrees` at `bdR` is `nextR rd` accepting at the last round** where every message before the last is of tag
`0`: the deployed plain verifier answers those rounds with their coins, as `IVStar`'s `ans` does. -/
theorem agrees_bdR (H : Claims.Msg → Claims.Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Claims.Msg → ℝ) (join : List Claims.Msg → Claims.Msg)
    (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin} (hx : (setupOf false x).isSome)
    (hlen : cs.length = ms.length) (h0 : ∀ j (hj : j + 1 < ms.length), ∃ t, ms[j] = 0 :: t)
    (hacc : ((nextR rd x ms cs).eval H).1 = .acc) :
    (((bdC rel rd cols wE join rowIn final w).agrees x ms cs).eval H).1 = true := by
  unfold Bound.agrees
  refine eval_allOC_true H _ fun p hp => ?_
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hp
  have hj' := List.mem_range.1 hj
  rw [eval_bindOC]
  by_cases hl : j + 1 < ms.length
  · obtain ⟨t, ht⟩ := h0 j hl
    have hjc : j < cs.length := by omega
    have hc := act_coin_of_tag (ms := ms) (cs := cs) hx (by omega) hjc ht
    have e : (((bdC rel rd cols wE join rowIn final w).DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
        .msg (cs[j]'hjc).bytes := nextR_coin rd H _ _ _ _ hc
    rw [e]
    have ha : (bdC rel rd cols wE join rowIn final w).ans (cs.getD j default) = (cs[j]'hjc).bytes := by
      rw [List.getD_eq_getElem _ _ hjc]
      simp only [bdC, RecDeploy.bdDeployed]
    simp only [Claims.OC.eval, hl, ↓reduceIte, ha, decide_true]
  · have hj1 : j + 1 = ms.length := by omega
    have e : (((bdC rel rd cols wE join rowIn final w).DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
        .acc := by
      rw [hj1, List.take_length, show cs.take ms.length = cs by rw [← hlen, List.take_length]]
      exact hacc
    rw [e]
    simp only [Claims.OC.eval, hl, ↓reduceIte, decide_true]

/-- **The honest run passes `IVStar` at `bdR (nextR rd)`, `join := joinR`**: on the honest commit strings of the rows
and salts `os`, then a last message `final` reads as `os` and `last`, at coins numbering the deployed plain verifier's
rounds, wherever the rounds carry the schedule's counts (`nAtSched`) and `nextR rd` accepts the opened messages. -/
theorem ivstarR_honest_run (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Claims.Msg → ℝ) (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    {x : Claims.Msg} {os : List (List (Claims.Msg × RecDeploy.Salt))} {last lastO : Claims.Msg} {cs : List Coin}
    (hx : (setupOf false x).isSome) (hfin : final lastO = some (os, last)) (hk : os.length + 1 = (V false).k x)
    (hn : ∀ j < os.length, (os.getD j []).length = RecDeploy.nAtSched x j) (hcs : cs.length = (V false).k x)
    (hacc : ((nextR rd x (os.map (fun o => joinR (o.map Prod.fst)) ++ [last]) cs).eval sha512).1 = .acc) :
    (Claims.run sha512 (Claims.replay ((bdC rel rd cols wE joinR rowIn final w).honestAs sha512 os lastO))
      (bdC rel rd cols wE joinR rowIn final w).IVStar x () fun i => cs.getD i.val default).out = .acc := by
  refine (bdC rel rd cols wE joinR rowIn final w).run_honest_acc sha512_out64 hfin hk hn hcs ?_
  refine agrees_bdR sha512 rel rd cols wE joinR rowIn final w
    (ms := (bdC rel rd cols wE joinR rowIn final w).opened os last) hx
    (hcs.trans (hk.symm.trans ((bdC rel rd cols wE joinR rowIn final w).opened_length os last).symm))
    (fun j hj => ?_) hacc
  have hj' : j < os.length := Nat.lt_of_add_lt_add_right
    (lt_of_lt_of_eq hj ((bdC rel rd cols wE joinR rowIn final w).opened_length os last))
  exact ⟨_, ((bdC rel rd cols wE joinR rowIn final w).opened_getElem os last hj').trans rfl⟩

/-! ## The honest session -/

/-- **The honest prover's rows at each round before the last**: round 0's (each table's level-0 cap nodes), then each
rep's rounds' (`repRows`), as messages. `honestMsgs` joins each (`honestMsgs_rounds`). -/
def honestRounds (caps : Array (Array ByteArray)) (reps : List (List (List ByteArray))) : List (List Claims.Msg) :=
  ((capRows caps).map nodeRow).map rowMsg :: reps.flatten.map fun rs => rs.map rowMsg

theorem honestMsgs_rounds (caps : Array (Array ByteArray)) (reps : List (List (List ByteArray))) (last : Claims.Msg) :
    honestMsgs caps reps last = (honestRounds caps reps).map joinR ++ [last] := by
  simp only [honestMsgs, honestRounds, List.map_cons, List.map_map, Function.comp_def]

/-- **Rows under salts**: round `j`'s row `i` under the salt `ys j i`. -/
def salted (ys : ℕ → ℕ → RecDeploy.Salt) (rounds : List (List Claims.Msg)) :
    List (List (Claims.Msg × RecDeploy.Salt)) :=
  rounds.mapIdx fun j rs => rs.mapIdx fun i r => (r, ys j i)

theorem salted_length (ys : ℕ → ℕ → RecDeploy.Salt) (rounds : List (List Claims.Msg)) :
    (salted ys rounds).length = rounds.length :=
  List.length_mapIdx

theorem salted_getD_length (ys : ℕ → ℕ → RecDeploy.Salt) (rounds : List (List Claims.Msg)) (j : ℕ) :
    ((salted ys rounds).getD j []).length = (rounds.getD j []).length := by
  by_cases hj : j < rounds.length
  · rw [List.getD_eq_getElem _ _ (by rw [salted_length]; exact hj), List.getD_eq_getElem _ _ hj]
    simp [salted]
  · rw [List.getD_eq_default _ _ (by rw [salted_length]; omega), List.getD_eq_default _ _ (by omega)]
    rfl

theorem salted_fst (ys : ℕ → ℕ → RecDeploy.Salt) (rounds : List (List Claims.Msg)) :
    (salted ys rounds).map (fun o => o.map Prod.fst) = rounds := by
  refine List.ext_getElem (by simp [salted_length]) fun j h₁ h₂ => ?_
  refine List.ext_getElem (by simp [salted]) fun i h₃ h₄ => ?_
  simp [salted]

theorem salted_join (ys : ℕ → ℕ → RecDeploy.Salt) (rounds : List (List Claims.Msg)) :
    (salted ys rounds).map (fun o => joinR (o.map Prod.fst)) = rounds.map joinR := by
  have h := congrArg (List.map joinR) (salted_fst ys rounds)
  rw [List.map_map] at h
  exact h

/-- **What the honest prover sends `IVStar` at `bdR`**: each round's commit strings of its honest rows under the salts
`ys` (`honestRounds`, `salted`), then the last message `lastO`. -/
noncomputable def honestAsR (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Claims.Msg → ℝ) (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    (caps : Array (Array ByteArray)) (reps : List (List (List ByteArray))) (ys : ℕ → ℕ → RecDeploy.Salt)
    (lastO : Claims.Msg) : List Claims.Msg :=
  (bdC rel rd cols wE joinR rowIn final w).honestAs sha512 (salted ys (honestRounds caps reps)) lastO

/-- Two lists of blocks, block for block of one length, related entry for entry, are related entry for entry once
flattened. -/
theorem flatten_getD_rel {α β : Type} (P : α → β → Prop) (da : α) (db : β) :
    ∀ (L : List (List α)) (M : List (List β)), L.length = M.length →
      (∀ k < L.length, (L.getD k []).length = (M.getD k []).length) →
      (∀ k < L.length, ∀ j < (L.getD k []).length, P ((L.getD k []).getD j da) ((M.getD k []).getD j db)) →
      L.flatten.length = M.flatten.length ∧
        ∀ i < L.flatten.length, P (L.flatten.getD i da) (M.flatten.getD i db)
  | [], [], _, _, _ => ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
  | [], _ :: _, h, _, _ => by simp at h
  | _ :: _, [], h, _, _ => by simp at h
  | a :: L, b :: M, hl, hb, hP => by
    have hab : a.length = b.length := hb 0 (by simp)
    obtain ⟨ih1, ih2⟩ := flatten_getD_rel P da db L M (by simpa using hl)
      (fun k hk => hb (k + 1) (by simp; omega))
      (fun k hk j hj => hP (k + 1) (by simp; omega) j hj)
    refine ⟨by simp only [List.flatten_cons, List.length_append, hab, ih1], fun i hi => ?_⟩
    simp only [List.flatten_cons]
    by_cases hia : i < a.length
    · rw [List.getD_append _ _ _ _ hia, List.getD_append _ _ _ _ (hab ▸ hia)]
      exact hP 0 (by simp) i hia
    · rw [List.getD_append_right _ _ _ _ (by omega), List.getD_append_right _ _ _ _ (by omega), ← hab]
      exact ih2 (i - a.length) (by simp only [List.flatten_cons, List.length_append] at hi; omega)

/-- Rep `k`'s schedule, table `k / 2`'s, rep by rep, is `nAtSched`'s: each table's schedule twice. -/
theorem flatten_repSched (spec : SessionSpec) (sts : Array _root_.Flock.Stmt) :
    ((List.range (2 * sts.size)).map fun k => RecDeploy.repSchedAt spec sts[k / 2]!).flatten =
      sts.toList.flatMap fun st => RecDeploy.repSchedAt spec st ++ RecDeploy.repSchedAt spec st := by
  have key : ∀ (g : ℕ → List RecVStar.SchedRound) (n : ℕ),
      ((List.range (2 * n)).map fun k => g (k / 2)).flatten = ((List.range n).map fun i => g i ++ g i).flatten := by
    intro g n
    induction n with
    | zero => simp
    | succ n ih =>
      rw [show 2 * (n + 1) = 2 * n + 1 + 1 by omega, List.range_succ, List.range_succ, List.range_succ (n := n),
        List.map_append, List.map_append, List.map_append, List.flatten_append, List.flatten_append,
        List.flatten_append, ih]
      simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil,
        show (2 * n + 1) / 2 = n by omega, show 2 * n / 2 = n by omega, List.append_assoc]
  rw [key (fun i => RecDeploy.repSchedAt spec sts[i]!) sts.size, List.flatMap_def]
  congr 1
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  have hi : i < sts.size := by simpa using h₁
  simp [getElem!_pos sts i hi]

/-- **The honest rows carry the schedule's counts and the deployed plain verifier's rounds**: at a statement that sets
up, from each table's level-0 cap (of `level0Nodes` nodes) and each rep's honest proof at its table's cap, round 0
carries the caps' nodes, round `1 + i` the `i`-th recorded round's tops and value row (`nAtSched`), and there are as
many rounds after round 0 as the reps' schedules have (`FitsRep` at each rep, `repFits`). -/
theorem honestRounds_nAt {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : setupOf false x = some (s, spec, sts)) {c₁ : Coin} {cs : List Coin} {caps : Array (Array ByteArray)}
    {ps : Array Flock.Proof} {Ts : ℕ → F128} (hs : caps.size = sts.size)
    (hcs : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) (hps : ps.size = 2 * sts.size)
    (hrep : ∀ k < ps.size, SchedOf (repCtx spec sts c₁ (capRows caps) cs k) ∧
      HonestRep (repCtx spec sts c₁ (capRows caps) cs k) ps[k]! ∧ ps[k]!.cap = caps[k / 2]!) :
    (honestRounds caps ((List.range ps.size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows caps) cs k) ps[k]! (Ts k))).length = 1 + (repLens spec sts).sum ∧
      ∀ j < (honestRounds caps ((List.range ps.size).map fun k =>
          repRows (repCtx spec sts c₁ (capRows caps) cs k) ps[k]! (Ts k))).length,
        ((honestRounds caps ((List.range ps.size).map fun k =>
          repRows (repCtx spec sts c₁ (capRows caps) cs k) ps[k]! (Ts k))).getD j []).length =
          RecDeploy.nAtSched x j := by
  set L := (List.range ps.size).map fun k => repRows (repCtx spec sts c₁ (capRows caps) cs k) ps[k]! (Ts k) with hL
  set M := (List.range (2 * sts.size)).map fun k => RecDeploy.repSchedAt spec sts[k / 2]! with hM
  have hF : ∀ k < ps.size, FitsRep (repCtx spec sts c₁ (capRows caps) cs k)
      (repRows (repCtx spec sts c₁ (capRows caps) cs k) ps[k]! (Ts k)) ps[k]! (Ts k) := fun k hk =>
    repFits _ _ _ (hrep k hk).1 (hrep k hk).2.1 (by rw [cap0_capRows hs hcs (by omega), (hrep k hk).2.2])
  have hLk : ∀ k < L.length, L.getD k [] = repRows (repCtx spec sts c₁ (capRows caps) cs k) ps[k]! (Ts k) :=
    fun k hk => by
      have hk' : k < ps.size := by simpa [hL] using hk
      simp [hL, List.getD_eq_getElem?_getD, hk']
  have hMk : ∀ k < L.length, M.getD k [] = RecDeploy.repSchedAt spec sts[k / 2]! := fun k hk => by
    have hk' : k < 2 * sts.size := by simpa [hL, hps] using hk
    simp [hM, List.getD_eq_getElem?_getD, hk']
  have hlen : L.length = M.length := by simp [hL, hM, hps]
  obtain ⟨hfl, hfP⟩ := flatten_getD_rel
    (fun (r : List ByteArray) (q : RecVStar.SchedRound) => r.length = q.2.1.toList.sum + if q.2.2.2 then 1 else 0)
    [] default L M hlen
    (fun k hk => by
      have hk' : k < ps.size := by simpa [hL] using hk
      rw [hLk k hk, hMk k hk]
      exact ((hF k hk').sched).symm)
    (fun k hk j hj => by
      have hk' : k < ps.size := by simpa [hL] using hk
      rw [hLk k hk] at hj ⊢
      rw [hMk k hk]
      exact ((hF k hk').elems j hj).1)
  have hMflat : M.flatten = sts.toList.flatMap fun st => RecDeploy.repSchedAt spec st ++ RecDeploy.repSchedAt spec st :=
    flatten_repSched spec sts
  have hMsum : M.flatten.length = (repLens spec sts).sum := by
    rw [List.length_flatten, hM, repLens, List.map_map]
    rfl
  have hround0 : (((capRows caps).map nodeRow).map rowMsg).length = (sts.toList.map RecDeploy.level0Nodes).sum := by
    rw [List.length_map, List.length_map, capRows, List.length_flatten, capLens hs hcs]
    rfl
  refine ⟨by simp only [honestRounds, List.length_cons, List.length_map, hfl, hMsum]; omega, fun j hj => ?_⟩
  unfold RecDeploy.nAtSched
  rw [hx]
  rcases j with _ | j
  · simp only [honestRounds, List.getD_cons_zero, ↓reduceIte]
    exact hround0
  · have hj' : j < L.flatten.length := by
      have h := hj
      simp only [honestRounds, List.length_cons, List.length_map] at h
      omega
    have hjM : j < M.flatten.length := hfl ▸ hj'
    simp only [honestRounds, List.getD_cons_succ, Nat.add_one_ne_zero, ↓reduceIte, Nat.add_sub_cancel]
    rw [← hMflat, List.getElem?_eq_getElem hjM]
    have e : (List.map (fun rs => rs.map rowMsg) L.flatten).getD j [] = (L.flatten.getD j []).map rowMsg := by
      rw [List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hj'), List.getD_eq_getElem _ _ hj',
        List.getElem_map]
    rw [e, List.length_map, hfP j hj']
    simp [List.getElem?_eq_getElem hjM]

/-- **The honest session passes `IVStar` at `bdR (nextR rd)`**: at a statement that sets up with `TableOk` tables, from
each table's level-0 cap and each rep's honest proof at its table's cap, the honest prover's commit strings of the
salted honest rows (`honestAsR`), then a last message `final` reads as those rows and the element-level last message
`last`, at coins numbering the deployed plain verifier's rounds, pass `IVStar`, wherever `nextR rd` accepts the honest
element-level messages (`honestMsgs`). -/
theorem ivstarR_honest_session (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Claims.Msg → ℝ) (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} {c₁ : Coin} {cs : List Coin}
    {caps : Array (Array ByteArray)} {ps : Array Flock.Proof} {Ts : ℕ → F128} {ys : ℕ → ℕ → RecDeploy.Salt}
    {last lastO : Claims.Msg} (hx : setupOf false x = some (s, spec, sts))
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i]) (hs : caps.size = sts.size)
    (hcs : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) (hps : ps.size = 2 * sts.size)
    (hrep : ∀ k < ps.size, SchedOf (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ∧
      HonestRep (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! ∧ ps[k]!.cap = caps[k / 2]!)
    (hk : cs.length = (V false).k x)
    (hfin : final lastO = some (salted ys (honestRounds caps ((List.range ps.size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k))), last))
    (hacc : ((nextR rd x (honestMsgs caps ((List.range ps.size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)) last) cs).eval sha512).1 =
        .acc) :
    (Claims.run sha512 (Claims.replay (honestAsR rel rd cols wE rowIn final w caps ((List.range ps.size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows caps) cs.tail.dropLast k) ps[k]! (Ts k)) ys lastO))
      (bdC rel rd cols wE joinR rowIn final w).IVStar x () fun i => cs.getD i.val default).out = .acc := by
  obtain ⟨hl, hn⟩ := honestRounds_nAt (Ts := Ts) hx hs hcs hps hrep
  have hkx : (V false).k x = 2 + rounds false spec sts := by simp only [V, hx]
  refine ivstarR_honest_run rel rd cols wE rowIn final w (by simp [hx]) hfin ?_ (fun j hj => ?_) hk ?_
  · rw [salted_length, hl, repLens_sum htab, hkx]
    omega
  · rw [salted_getD_length]
    exact hn j (by rwa [salted_length] at hj)
  · rw [salted_join, ← honestMsgs_rounds]
    exact hacc

/-- **The tables' facts the files-level results take** (`htab`), at a statement `verify --plain-tables` admits. -/
theorem tabsOk_of_plain {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : setupOf false x = some (s, spec, sts)) (hP : PlainTablesOk s.read sts) (c₁ : Coin) :
    ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048 :=
  fun i hi => ⟨(tablesOk_of_plainTables hx hP i hi).1, (tablesOk_of_plainTables hx hP i hi).2 c₁⟩

/-- **The honest session passes `IVStar` at `bdR (nextR rd)` at its own files**: `ivstarR_honest_session` at the caps
and proofs the pure session's files decode to (`fileCaps`, `filePs`), at `TableOk` tables with at most 2,048 extra
claims at the first coin's points (`tabsOk_of_plain` at a statement `verify --plain-tables` admits; `Run.recTabsOk` at a
plain completeness run), with replies that are the coins' bytes (as `sessionRun_decides_tables`). -/
theorem ivstarR_honest_files (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Claims.Msg → ℝ) (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} {c₁ : Coin} {cs : List Coin}
    {resp : ℕ → Claims.Msg} {wI : Claims.Flock.Wit} {ω : FlockComplete.Tape} {Ts : ℕ → F128}
    {ys : ℕ → ℕ → RecDeploy.Salt} {last lastO : Claims.Msg} (hx : setupOf false x = some (s, spec, sts))
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048) (hc1 : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hk : cs.length = (V false).k x)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS sha512 resp spec sts wI ω = (.ok (rec, pfs), bs))
    (hfin : final lastO = some (salted ys (honestRounds (fileCaps spec sts c₁ pfs)
      ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! (Ts k))), last))
    (hacc : ((nextR rd x (honestMsgs (fileCaps spec sts c₁ pfs)
      ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! (Ts k)) last) cs).eval sha512).1 = .acc) :
    (Claims.run sha512 (Claims.replay (honestAsR rel rd cols wE rowIn final w (fileCaps spec sts c₁ pfs)
      ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! (Ts k)) ys lastO))
      (bdC rel rd cols wE joinR rowIn final w).IVStar x () fun i => cs.getD i.val default).out = .acc := by
  have hc : resp 0 = c₁.bytes := by
    cases cs with
    | nil => simp at hc1
    | cons c cs' =>
      simp only [List.head?_cons, Option.some.injEq] at hc1
      subst hc1
      simpa using hresp 0 (by simp)
  have hfc := filesFinalC sha512_out64 hc (setup_tables hx).2.symm htab hS hresp
    (hlen_of_k hx (fun i hi => (htab i hi).1) hk)
  obtain ⟨hsz, hall⟩ := sessionS_reps sha512_out64 hc (setup_tables hx).2.symm htab hS
    (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast
  have hcaps : ∀ i < (fileCaps spec sts c₁ pfs).size,
      (fileCaps spec sts c₁ pfs)[i]!.size = RecDeploy.level0Nodes sts[i]! := by
    intro i hi
    rw [fileCaps_size] at hi
    have h := (hall (2 * i) (by omega)).2.2.2.2.1
    rw [show 2 * i / 2 = i by omega] at h
    simp only [fileCaps, JsonRT.range_map_getElem! _ hi]
    exact h
  have hrep : ∀ k < (filePs spec sts c₁ pfs).size,
      SchedOf (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) ∧
        HonestRep (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! ∧
          (filePs spec sts c₁ pfs)[k]!.cap = (fileCaps spec sts c₁ pfs)[k / 2]! := by
    intro k hk
    rw [filePs_size] at hk
    have hf := hfc k hk
    rw [hsz] at hk
    obtain ⟨-, h2, h3, h4, -, -, -⟩ := hall k hk
    refine ⟨h2, h3 hf, ?_⟩
    simp only [fileCaps, JsonRT.range_map_getElem! _ (show k / 2 < sts.size by omega)]
    exact h4
  exact ivstarR_honest_session rel rd cols wE rowIn final w hx (fun i hi => (htab i hi).1) (fileCaps_size ..) hcaps
    (by rw [filePs_size, hsz]) hrep hk hfin hacc

end FlockDeploy.KS

namespace RecDeploy

open Flock FlockComplete FlockDeploy FlockDeploy.KS

/-- **Recursion completeness's inner premise at `bdR (nextR rd)`** (`InnerAccepts`, which `vstar_complete` takes): at a
compiled statement whose inner rounds of `v` are the honest prover's (`honestAsR`) for the inner statement, the honest
session's run passes `IVStar`, under `ivstarR_honest_files`' conditions at the inner statement. -/
theorem innerAccepts_honest (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Claims.Msg → ℝ) (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    {xC : Claims.Msg} {v : List (Claims.Msg × Claims.Msg)} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {c₁ : Coin} {cs : List Coin} {resp : ℕ → Claims.Msg} {wI : Claims.Flock.Wit}
    {ω : FlockComplete.Tape} {Ts : ℕ → F128} {ys : ℕ → ℕ → Salt} {last lastO : Claims.Msg}
    (hx : setupOf false (xIOf xC) = some (s, spec, sts))
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048) (hc1 : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hk : cs.length = (FlockDeploy.V false).k (xIOf xC))
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS sha512 resp spec sts wI ω = (.ok (rec, pfs), bs))
    (hfin : final lastO = some (salted ys (honestRounds (fileCaps spec sts c₁ pfs)
      ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! (Ts k))), last))
    (hacc : ((nextR rd (xIOf xC) (honestMsgs (fileCaps spec sts c₁ pfs)
      ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! (Ts k)) last) cs).eval sha512).1 = .acc)
    (hv : (v.take ((bdC rel rd cols wE joinR rowIn final w).IVStar.k (xIOf xC))).map Prod.fst =
      honestAsR rel rd cols wE rowIn final w (fileCaps spec sts c₁ pfs)
        ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
          repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
            (filePs spec sts c₁ pfs)[k]! (Ts k)) ys lastO) :
    InnerAccepts sha512 (bdC rel rd cols wE joinR rowIn final w).IVStar xC cs v := by
  unfold InnerAccepts
  rw [hv]
  exact ivstarR_honest_files rel rd cols wE rowIn final w hx htab hc1 hresp hk hS hfin hacc

end RecDeploy

namespace RecDeploy.DeployedR

open Flock FlockComplete FlockDeploy FlockDeploy.KS
open Claims (Msg)

/-- **One round's openings as the honest prover lays them in V\*'s last message**: each row, then its salt. -/
def encRound (os : List (Msg × Salt)) : Msg := os.flatMap fun o => o.1 ++ List.ofFn o.2

/-- Every earlier round's openings, round by round. -/
def encRounds (oss : List (List (Msg × Salt))) : Msg := oss.flatMap encRound

/-- Each round's opened rows' byte lengths, the trailer's contents. -/
def lensOf (oss : List (List (Msg × Salt))) : List (List ℕ) := oss.map fun os => os.map fun o => o.1.length

/-- **The honest prover's last message to V\***: every earlier round's openings (`encRounds`), then the element-level
last message `last`, then the trailer of the openings' row lengths. -/
def lastOR (oss : List (List (Msg × Salt))) (last : Msg) : Msg := encRounds oss ++ last ++ trailer (lensOf oss)

theorem saltOf_ofFn (y : Salt) : saltOf (List.ofFn y) = y := by
  funext i
  simp only [saltOf]
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

theorem cutRound_encRound (rest : Msg) :
    ∀ os : List (Msg × Salt), cutRound (os.map fun o => o.1.length) (encRound os ++ rest) = some (os, rest)
  | [] => by simp [encRound, cutRound]
  | (r, y) :: os => by
    have hy : (List.ofFn y).length = Hm96.sha512.saltBytes := List.length_ofFn
    have hm : encRound ((r, y) :: os) ++ rest = r ++ (List.ofFn y ++ (encRound os ++ rest)) := by
      simp [encRound, List.append_assoc]
    rw [hm]
    simp only [List.map_cons, cutRound]
    rw [ite_eq_left (by simp only [List.length_append, hy]; omega), List.drop_left, List.drop_left' hy,
      cutRound_encRound rest os, List.take_left, List.take_left' hy, saltOf_ofFn]
    rfl

theorem cutRounds_encRounds (rest : Msg) :
    ∀ oss : List (List (Msg × Salt)), cutRounds (lensOf oss) (encRounds oss ++ rest) = some (oss, rest)
  | [] => by simp [lensOf, encRounds, cutRounds]
  | os :: oss => by
    have hm : encRounds (os :: oss) ++ rest = encRound os ++ (encRounds oss ++ rest) := by
      simp [encRounds, List.append_assoc]
    rw [hm]
    show cutRounds ((os.map fun o => o.1.length) :: lensOf oss) _ = _
    simp only [cutRounds]
    rw [cutRound_encRound]
    simp [cutRounds_encRounds rest oss]

/-- **`finalR` reads the honest last message** as the openings it lays out and the element-level last message. -/
theorem finalR_lastOR (oss : List (List (Msg × Salt))) (last : Msg) :
    finalR (lastOR oss last) = some (oss, last) := by
  rw [lastOR, finalR_trailer, cutRounds_encRounds]

/-- **The honest element-level last message is of tag `1`**, so the deployed plain verifier answers it with no coin
(HONEST-VERDICT's `h0` at `last := encLast`). -/
theorem encLast_head (stmt : Msg) (ps : List Flock.Proof) : (encLast stmt ps).head? ≠ some 0 := by
  simp [encLast]

/-- **The honest reps' rows at the pure session's files**: rep `k`'s `repRows` at its proof `filePs[k]`, the caps
`fileCaps` and the coins between the first and the last (`ivstarR_honest_files`' reps). -/
noncomputable def filesReps (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray)
    (cs : List Coin) (Ts : ℕ → F128) : List (List (List ByteArray)) :=
  (List.range (filePs spec sts c₁ pfs).size).map fun k =>
    repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) (filePs spec sts c₁ pfs)[k]!
      (Ts k)

/-- **The honest rows under their salts at the pure session's files.** -/
noncomputable def filesOs (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray)
    (cs : List Coin) (Ts : ℕ → F128) (ys : ℕ → ℕ → Salt) : List (List (Msg × Salt)) :=
  salted ys (honestRounds (fileCaps spec sts c₁ pfs) (filesReps spec sts c₁ pfs cs Ts))

/-- **The honest session passes `IVStar` at `final := finalR`**: `ivstarR_honest_files` with its last message
`lastOR` of the salted honest rows, which `finalR` reads (`finalR_lastOR`), so only the honest verdict (`nextR rd`
accepting `honestMsgs`) stays a hypothesis. -/
theorem ivstar_honest_filesR (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (rowIn : Msg → Msg) (w : Msg → ℝ) {x : Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin}
    {cs : List Coin} {resp : ℕ → Msg} {wI : Claims.Flock.Wit} {ω : FlockComplete.Tape} {Ts : ℕ → F128}
    {ys : ℕ → ℕ → Salt} {last : Msg} (hx : setupOf false x = some (s, spec, sts))
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    (hc1 : cs.head? = some c₁) (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes)
    (hk : cs.length = (FlockDeploy.V false).k x) {rec : String} {pfs : Array ByteArray} {bs : Array Msg}
    (hS : FlockComplete.sessionS sha512 resp spec sts wI ω = (.ok (rec, pfs), bs))
    (hacc : ((nextR rd x (honestMsgs (fileCaps spec sts c₁ pfs) (filesReps spec sts c₁ pfs cs Ts) last) cs).eval
      sha512).1 = .acc) :
    (Claims.run sha512 (Claims.replay (honestAsR rel rd cols wE rowIn finalR w (fileCaps spec sts c₁ pfs)
      (filesReps spec sts c₁ pfs cs Ts) ys (lastOR (filesOs spec sts c₁ pfs cs Ts ys) last)))
      (bdC rel rd cols wE joinR rowIn finalR w).IVStar x () fun i => cs.getD i.val default).out = .acc :=
  ivstarR_honest_files rel rd cols wE rowIn finalR w hx htab hc1 hresp hk hS (finalR_lastOR _ _) hacc

/-- **Recursion completeness's inner premise at `final := finalR`** (`InnerAccepts`): `innerAccepts_honest` with the
honest last message `lastOR`, which `finalR` reads. -/
theorem innerAccepts_honestR (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (cols : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (rowIn : Msg → Msg) (w : Msg → ℝ) {xC : Msg} {v : List (Msg × Msg)} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {c₁ : Coin} {cs : List Coin} {resp : ℕ → Msg} {wI : Claims.Flock.Wit}
    {ω : FlockComplete.Tape} {Ts : ℕ → F128} {ys : ℕ → ℕ → Salt} {last : Msg}
    (hx : setupOf false (xIOf xC) = some (s, spec, sts))
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048) (hc1 : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hk : cs.length = (FlockDeploy.V false).k (xIOf xC))
    {rec : String} {pfs : Array ByteArray} {bs : Array Msg}
    (hS : FlockComplete.sessionS sha512 resp spec sts wI ω = (.ok (rec, pfs), bs))
    (hacc : ((nextR rd (xIOf xC) (honestMsgs (fileCaps spec sts c₁ pfs) (filesReps spec sts c₁ pfs cs Ts) last)
      cs).eval sha512).1 = .acc)
    (hv : (v.take ((bdC rel rd cols wE joinR rowIn finalR w).IVStar.k (xIOf xC))).map Prod.fst =
      honestAsR rel rd cols wE rowIn finalR w (fileCaps spec sts c₁ pfs) (filesReps spec sts c₁ pfs cs Ts) ys
        (lastOR (filesOs spec sts c₁ pfs cs Ts ys) last)) :
    InnerAccepts sha512 (bdC rel rd cols wE joinR rowIn finalR w).IVStar xC cs v :=
  innerAccepts_honest rel rd cols wE rowIn finalR w hx htab hc1 hresp hk hS (finalR_lastOR _ _) hacc hv

end RecDeploy.DeployedR
