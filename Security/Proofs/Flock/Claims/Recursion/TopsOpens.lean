import Proofs.Flock.Claims.Recursion.CapOpens
import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.BoundAccepts

/-!
# `PartOpensB` at `layoutBTops` and the coin server's record

`partOpensB_layoutBTops` reduces `PartOpensB` at `cmtTr recRead` to the pieces of the stream rounds that commit tops.
Such a round `j` is rep `j / R`'s round `k = j % R` (`SchedMatches`), and its message in the record is its tops' nodes,
then its `c_k` (`Record.msgs`). Its pieces are its tops at their caps, then its row (`topsPiecesAt`):

* **The `e`-th top** sits at part `capsFrom + g`, `g = topsBefore rs k + e`, V\*'s later cap `.cap (g + 1) (2n)` with
  `n` the top's nodes (`capStmt_at`, from `checkTopsListed`). A staged part has a circuit (`staged_circuit`), and its
  port `top`'s slot at instance `rep · n + t` reads `topRow lk (g + 1) (rep · n + t)` (`capRow_bytes`), node `t` of rep
  `rep`'s `g`-th top (`topRow_rep`), which the record holds as round `k`'s `e`-th top (`round_view`, `RepAgrees`).
  Every node is 128 bytes (`innerLink_nodes`: `RecVStar.hexList` refuses any other), so each top's piece starts where
  its nodes do (`pieceLen_topPiece`).
* **The row** is a held row's slot, rep `rep`'s `c_k` at that round (`heldRow_bytes`), the round's message after its
  tops.

So `PartOpensB` holds at `layoutBTops` and `cmtTr recRead` wherever the inner verifier's rounds are the V\* schedule's
(`partOpensB_recRead`), and `ksLin2_deployed_recRead` and `bitsLin2_deployed_recRead` are `ksLin2_deployed_open` and
`bitsLin2_deployed_open` at `recRead` with `PartOpensB` dropped. What they still take: `PartChecksBV`, `DecompB`, the
readers' `ReadersTrue`, and `recRead` reading the coin server's transcript back (`ReadsBackB`).
-/

namespace FlockDeploy

/-- The setup's parsers do not read the public file: a raw statement parses wherever it does with another. -/
theorem ins_publicFile {w : Raw} {pf : ByteArray} {i' : Ins} (h : ({ w with publicFile := pf } : Raw).ins = .ok i') :
    ∃ i, w.ins = .ok i := by
  unfold Raw.ins at h ⊢
  simp only [bind, Except.bind, pure, Except.pure] at h ⊢
  repeat' split at h
  all_goals try (cases h; done)
  repeat' split
  all_goals simp_all

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Lean (Json)

/-! ## The coin server's nodes are 128 bytes -/

theorem linkForIn_inv {α σ ε : Type} (l : List α) (init : σ) (f : α → σ → Except ε (ForInStep σ))
    (P : σ → Prop) (h0 : P init) (hf : ∀ a ∈ l, ∀ s s', P s → f a s = .ok s' → P s'.value) {r : σ}
    (h : forIn l init f = .ok r) : P r := by
  induction l generalizing init with
  | nil => cases h; exact h0
  | cons a l ih =>
    rw [List.forIn_cons] at h
    cases hfa : f a init with
    | error e => rw [hfa] at h; cases h
    | ok s' =>
      rw [hfa] at h
      have hs := hf a List.mem_cons_self init s' h0 hfa
      cases s' with
      | done b => cases h; exact hs
      | yield b => exact ih b hs (fun a' ha' => hf a' (List.mem_cons_of_mem a ha')) h

theorem lmapM_ok_mem {α β ε : Type} {f : α → Except ε β} {l : List α} {r : List β} (h : l.mapM f = .ok r) :
    ∀ b ∈ r, ∃ a ∈ l, f a = .ok b := by
  induction l generalizing r with
  | nil => cases h; simp
  | cons a l ih =>
    rw [List.mapM_cons] at h
    cases hfa : f a with
    | error e => rw [hfa] at h; cases h
    | ok b =>
      rw [hfa] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok bs =>
        rw [hl] at h
        cases h
        intro b' hb'
        rcases List.mem_cons.1 hb' with rfl | hb'
        · exact ⟨a, List.mem_cons_self, hfa⟩
        · obtain ⟨a', ha', hf'⟩ := ih hl b' hb'
          exact ⟨a', List.mem_cons_of_mem a ha', hf'⟩

theorem amapM_ok_mem {α β ε : Type} {f : α → Except ε β} {xs : Array α} {r : Array β} (h : xs.mapM f = .ok r) :
    ∀ b ∈ r, ∃ a ∈ xs, f a = .ok b := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok bs =>
    rw [hl] at h
    cases h
    intro b hb
    obtain ⟨a, ha, hf⟩ := lmapM_ok_mem hl b (by simpa using hb)
    exact ⟨a, by simpa using ha, hf⟩

theorem hexList_size {ctx : String} {xs : Array String} {bs : Array ByteArray}
    (h : RecVStar.hexList ctx xs = .ok bs) : ∀ b ∈ bs, b.size = 128 := by
  intro b hb
  obtain ⟨x, -, hx⟩ := amapM_ok_mem h b hb
  simp only [bind, Except.bind] at hx
  split at hx
  · cases hx
  · split at hx
    · cases hx
    · rename_i hne
      cases hx
      simpa using hne

/-- A loop that only pushes nodes of 128 bytes ends with only such nodes. -/
def Nodes128 (out : Array (ℕ × Array ByteArray)) : Prop := ∀ y ∈ out, ∀ b ∈ y.2, b.size = 128

theorem nodes128_push {out : Array (ℕ × Array ByteArray)} (h : Nodes128 out) {k : ℕ} {bs : Array ByteArray}
    (hb : ∀ b ∈ bs, b.size = 128) : Nodes128 (out.push (k, bs)) := by
  intro y hy
  rcases Array.mem_push.1 hy with hy | rfl
  · exact h y hy
  · exact hb

/-- **Every node of the reps' tops is 128 bytes**: the link reads each through `RecVStar.hexList`, which refuses any
other length. -/
theorem innerLink_nodes {tr : Json} {record : String} {lk : RecVStar.InnerLink}
    (h : RecVStar.innerLink tr record = .ok lk) : ∀ t ∈ lk.tops, ∀ y ∈ t, ∀ b ∈ y.2, b.size = 128 := by
  unfold RecVStar.innerLink at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals try (cases h; done)
  cases h
  intro t ht
  obtain ⟨x, -, hx⟩ := amapM_ok_mem (by assumption) t ht
  split at hx
  · cases hx
  split at hx
  · cases hx
  rename_i v hv
  cases hx
  rw [← Array.forIn_toList] at hv
  refine linkForIn_inv _ _ _ Nodes128 (fun y hy => by simp at hy) ?_ hv
  intro r _ s s' hs hr
  split at hr
  · split at hr
    · cases hr
    rename_i v' hv'
    cases hr
    rw [← Array.forIn_toList] at hv'
    refine linkForIn_inv _ _ _ Nodes128 hs ?_ hv'
    intro a _ s2 s2' hs2 ha
    split at ha
    · cases ha
    split at ha
    · cases ha
    rename_i bs hbs
    cases ha
    exact nodes128_push hs2 (hexList_size hbs)
  · cases hr

theorem getD_getElem? {α : Type} (a : Array (Array α)) (x g : ℕ) : (a.getD x #[])[g]? = a[x]?.bind (·[g]?) := by
  cases h : a[x]? with
  | none =>
    have : a.size ≤ x := by
      rcases Nat.lt_or_ge x a.size with hx | hx
      · rw [Array.getElem?_eq_getElem hx] at h; cases h
      · exact hx
    simp [Array.getD, Nat.not_lt.2 this]
  | some b =>
    rw [Array.getD_eq_getD_getElem?, h]
    rfl

theorem nodes_of_getD {lk : RecVStar.InnerLink} (hN : ∀ t ∈ lk.tops, ∀ y ∈ t, ∀ b ∈ y.2, b.size = 128)
    {rep g kt : ℕ} {top : Array ByteArray} (h : (lk.tops.getD rep #[])[g]? = some (kt, top)) :
    ∀ b ∈ top, b.size = 128 := by
  rw [getD_getElem?] at h
  obtain ⟨t, ht, h⟩ := Option.bind_eq_some_iff.1 h
  exact hN t (Array.mem_of_getElem? ht) (kt, top) (Array.mem_of_getElem? h)

theorem byteArray_length (b : ByteArray) : b.toList.length = b.size := by
  rw [byteArray_toList_eq, Array.length_toList]
  rfl

theorem sum_map_const {α : Type} (l : List α) (c : ℕ) : (l.map fun _ => c).sum = l.length * c := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.sum_cons, ih, List.length_cons, Nat.succ_mul]; omega

theorem flatten_nodes_length {top : Array ByteArray} (h : ∀ b ∈ top, b.size = 128) :
    (top.toList.map (·.toList)).flatten.length = 128 * top.size := by
  rw [List.length_flatten, List.map_map]
  have e : top.toList.map (List.length ∘ (·.toList)) = top.toList.map fun _ => 128 :=
    List.map_congr_left fun b hb => by
      simp only [Function.comp_apply, byteArray_length, h b (Array.mem_toList_iff.1 hb)]
  rw [e, sum_map_const, Array.length_toList, Nat.mul_comm]

/-! ## The record's rounds -/

/-- **Rep `x`'s segment of the record**: the rounds open with rep 0's schedule, then rep 1's, so rep `x`'s rounds
follow `x · R` rounds of none of its stream. -/
theorem rep_segment {r : RecRecord.Record} {t : String} {rs : Array RecVStar.SchedRound}
    {E : List (String × String × ℕ × Array RecVStar.SchedRound)}
    (h : r.rounds.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      ((s!"{t}/rep{(0 : ℕ)}", "", 0, rs) :: (s!"{t}/rep{(1 : ℕ)}", "", 0, rs) :: E).flatMap
        (fun (s, _, _, rs) => rs.toList.mapIdx fun k w => (s.toUTF8.toList, k, w.1, w)))
    {x : ℕ} (hx : x < 2) :
    ∃ P A Q, r.rounds = P ++ A ++ Q ∧ P.length = x * rs.size ∧
      P.filter (·.stream == (s!"{t}/rep{x}").toUTF8.toList) = [] ∧
      A.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
        rs.toList.mapIdx fun k w => ((s!"{t}/rep{x}").toUTF8.toList, k, w.1, w) := by
  simp only [List.flatMap_cons] at h
  obtain ⟨A0, R1, hr, h0, h1'⟩ := List.map_eq_append_iff.1 h
  obtain ⟨A1, AE, rfl, h1, -⟩ := List.map_eq_append_iff.1 h1'
  obtain ⟨l0, a0⟩ := sched_at h0
  rcases (by omega : x = 0 ∨ x = 1) with rfl | rfl
  · exact ⟨[], A0, A1 ++ AE, by rw [hr]; rfl, by simp, rfl, h0⟩
  · refine ⟨A0, A1, AE, by rw [hr, List.append_assoc], by rw [l0]; simp, ?_, h1⟩
    exact List.filter_eq_nil_iff.2 fun a ha => by
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
      rw [(a0 i hi).1]
      simpa using rep_ne t

/-- **A round's message is its tops' nodes, then its `c_k`**: at the record's rounds `P ++ A ++ Q`, inner message
`P.length + k + 1` (round 0 being the link's) is `A`'s round `k`'s. -/
theorem msg_seg {r : RecRecord.Record} {P A Q : List RecRecord.Round} (hr : r.rounds = P ++ A ++ Q) {k : ℕ}
    (hk : k < A.length) :
    (r.msgs.map Prod.fst).getD (P.length + k + 1) [] =
      (A[k].tops.map List.flatten).flatten ++ A[k].commits.flatten := by
  unfold RecRecord.Record.msgs
  rw [List.getD_eq_getElem?_getD, List.map_cons, List.getElem?_cons_succ, List.map_map, List.getElem?_map, hr,
    List.append_assoc, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left, List.getElem?_append_left hk,
    List.getElem?_eq_getElem hk]
  rfl

/-- **A stream's `e`-th top of its round `k` is its `topsBefore rs k + e`-th salted top**: each of the stream's rounds
before `k` commits as many tops as its schedule says. -/
theorem repTops_seg {r : RecRecord.Record} {P A Q : List RecRecord.Round} {s : List UInt8}
    {rs : Array RecVStar.SchedRound} (hr : r.rounds = P ++ A ++ Q) (hP : P.filter (·.stream == s) = [])
    (hA : A.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      rs.toList.mapIdx fun k w => (s, k, w.1, w))
    {k e : ℕ} (hk : k < rs.size) (he : e < ((rs.getD k default).2.1).size) :
    ∃ (hkA : k < A.length) (heA : e < A[k].tops.length),
      (RecSetup.repTops r s)[topsBefore rs k + e]? = some (k, A[k].tops[e]) := by
  obtain ⟨hlen, hat⟩ := sched_at hA
  have hkA : k < A.length := by omega
  obtain ⟨-, hkk, -⟩ := hat k hkA
  have hnt : ∀ i (hi : i < A.length), A[i].tops.length = ((rs.getD i default).2.1).size := by
    intro i hi
    rw [← (hat i hi).2.2]
    simp [RecSetup.schedOfRound]
  have heA : e < A[k].tops.length := by rw [hnt k hkA]; exact he
  refine ⟨hkA, heA, ?_⟩
  have hA1 : A.filter (·.stream == s) = A :=
    List.filter_eq_self.2 fun a ha => by
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
      simp [(hat i hi).1]
  have hsplit : A = A.take k ++ A[k] :: A.drop (k + 1) := by
    rw [← List.drop_eq_getElem_cons hkA, List.take_append_drop]
  have hpre : ((A.take k).flatMap fun x => x.tops.map (x.k, ·)).length = topsBefore rs k := by
    unfold topsBefore
    rw [List.length_flatMap, List.length_flatMap]
    congr 1
    apply List.ext_getElem (by simp [hlen])
    intro i h1 h2
    simp only [List.length_map, List.length_take] at h1
    simp only [List.getElem_map, List.getElem_take, List.length_map, Array.length_toList]
    rw [hnt i (by omega)]
    simp [Array.getD, show i < rs.size by omega]
  have hfl := congrArg (List.flatMap fun x : RecRecord.Round => x.tops.map (x.k, ·)) hsplit
  rw [List.flatMap_append, List.flatMap_cons] at hfl
  unfold RecSetup.repTops
  rw [hr, List.filter_append, List.filter_append, hP, hA1, List.nil_append, List.flatMap_append, hfl,
    List.append_assoc, List.append_assoc, List.getElem?_append_right (by omega), hpre, Nat.add_sub_cancel_left,
    List.getElem?_append_left (by simpa using heA)]
  simp [hkk]

/-- **A stream's `kc`-th commit string list is its `kc`-th commit round's**. -/
theorem repCommits_seg {r : RecRecord.Record} {P A Q : List RecRecord.Round} {s : List UInt8}
    {rs : Array RecVStar.SchedRound} (hr : r.rounds = P ++ A ++ Q) (hP : P.filter (·.stream == s) = [])
    (hA : A.map (fun x => (x.stream, x.k, x.n, RecSetup.schedOfRound x)) =
      rs.toList.mapIdx fun k w => (s, k, w.1, w))
    {kc k : ℕ} (hk : (RecVStar.commitRounds rs)[kc]? = some k) :
    ∃ hkA : k < A.length, (RecSetup.repCommits r s)[kc]? = some (k, A[k].commits) := by
  obtain ⟨hlen, hat⟩ := sched_at hA
  have hA1 : A.filter (·.stream == s) = A :=
    List.filter_eq_self.2 fun a ha => by
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
      simp [(hat i hi).1]
  have hkc : kc < (RecVStar.commitRounds rs).length := by
    rcases Nat.lt_or_ge kc (RecVStar.commitRounds rs).length with h | h
    · exact h
    · rw [List.getElem?_eq_none h] at hk; cases hk
  have hkm : k ∈ RecVStar.commitRounds rs := List.mem_of_getElem? hk
  have hkA : k < A.length := by
    unfold RecVStar.commitRounds at hkm; rw [hlen]; exact List.mem_range.1 (List.mem_filter.1 hkm).1
  refine ⟨hkA, ?_⟩
  unfold RecSetup.repCommits
  rw [hr, List.filter_append, List.filter_append, hP, hA1, List.nil_append, List.filterMap_append,
    commits_of_sched hA, List.getElem?_append_left (by simpa using hkc), List.getElem?_map, hk, Option.map_some]
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkA]

/-- **What a staged part's record holds at rep `rep`'s round `k`**: the round `y` whose message, round
`rep · R + k + 1` as `cmtTr recRead` reads it, is its tops' nodes then its `c_k`; its tops have the schedule's sizes,
each the coin server's top of rep `rep` the link holds at `topsBefore rs k + e`, of 128-byte nodes; and at a commit
round its `c_k` is the commits the setup holds. -/
theorem round_view {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {r : XR} (hr : readX x = some r)
    {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) :
    ∃ (i : RecIns) (lk : RecVStar.InnerLink) (cs : RecVStar.InnerCommits),
      insO xO = some i ∧ i.link = some lk ∧ i.commits = some cs ∧
      ∀ rep < 2, ∀ k < rs.size, ∃ y : RecRecord.Round,
        cmtTr recRead x (rep * rs.size + k + 1) xO = (y.tops.map List.flatten).flatten ++ y.commits.flatten ∧
        y.tops.map List.length = ((rs.getD k default).2.1).toList ∧
        (∀ e (he : e < y.tops.length), ∃ kt top, (lk.tops.getD rep #[])[topsBefore rs k + e]? = some (kt, top) ∧
          y.tops[e] = top.toList.map (·.toList) ∧ ∀ b ∈ top, b.size = 128) ∧
        ∀ (kc kk : ℕ) (c : ByteArray), (RecVStar.commitRounds rs)[kc]? = some k →
          (cs.reps.getD rep #[])[kc]? = some (kk, c) → y.commits = [c.toList] := by
  obtain ⟨i, sh, cs, tt, tr, record, rc, lk, hi, hsh, hcs, hil, hR, hlk, hrd, hshown⟩ := staged_reads_link hS
  obtain ⟨lk', rc', want, hlk', hrd', -, -, -, hw, hrounds, hA0, hA1⟩ := FlockC3.VStarList.checkReads_ok hR
  rw [hlk] at hlk'
  obtain rfl : lk = lk' := Except.ok.inj hlk'
  rw [hrd] at hrd'
  obtain rfl : rc = rc' := Option.some.inj hrd'
  have hsh' : sh = r.shape := by
    have hS' := hS
    obtain ⟨ms, -, -, -, rfl, -, -⟩ := hS'
    obtain ⟨xr, hxr, hinner, -⟩ := insO_staged hi
    rw [hr] at hxr
    cases hxr
    exact Option.some.inj (hsh.symm.trans hinner)
  subst hsh'
  obtain ⟨rs', E, hrs', hE⟩ := scheduleOf_reps hw
  unfold repSched at hrs
  rw [hrs] at hrs'
  cases hrs'
  rw [hE] at hrounds
  have hN := innerLink_nodes hlk
  refine ⟨i, lk, cs, hi, hil, hcs, fun rep hrep k hk => ?_⟩
  have hA : FlockC3.VStarList.RepAgrees rc lk cs rep := by
    rcases (by omega : rep = 0 ∨ rep = 1) with h | h <;> rw [h] <;> assumption
  obtain ⟨P, A, Q, hrr, hPl, hPf, hAA⟩ := rep_segment hrounds hrep
  obtain ⟨hlen, hat⟩ := sched_at hAA
  have hkA : k < A.length := by omega
  have hsk := (hat k hkA).2.2
  have hlens : A[k].tops.map List.length = ((rs.getD k default).2.1).toList := by
    rw [← hsk]
    simp [RecSetup.schedOfRound]
  refine ⟨A[k], ?_, hlens, fun e he => ?_, fun kc kk c hkc hcc => ?_⟩
  · rw [cmtTr_record hshown, ← hPl, msg_seg hrr hkA]
  · have he' : e < ((rs.getD k default).2.1).size := by
      have := congrArg List.length hlens
      simp only [List.length_map, Array.length_toList] at this
      omega
    obtain ⟨_, _, htop⟩ := repTops_seg hrr hPf hAA hk he'
    rw [hA.1, List.getElem?_map, Array.getElem?_toList] at htop
    obtain ⟨⟨kt, top⟩, hkt, he2⟩ := Option.map_eq_some_iff.1 htop
    exact ⟨kt, top, hkt, (Prod.mk.inj he2).2.symm, nodes_of_getD hN hkt⟩
  · obtain ⟨_, hc⟩ := repCommits_seg hrr hPf hAA hkc
    rw [hA.2, List.getElem?_map, Array.getElem?_toList, hcc] at hc
    exact (Prod.mk.inj (Option.some.inj hc)).2.symm

theorem tops_len {tops : List (List (List UInt8))} {arr : Array ℕ} (hl : tops.map List.length = arr.toList)
    {e : ℕ} (he : e < tops.length) : tops[e].length = arr.getD e 0 := by
  have h := congrArg (·[e]?) hl
  simp only [List.getElem?_map, List.getElem?_eq_getElem he, Option.map_some, Array.getElem?_toList] at h
  rw [Array.getD_eq_getD_getElem?, ← h, Option.getD_some]

theorem tops_length {tops : List (List (List UInt8))} {arr : Array ℕ} (hl : tops.map List.length = arr.toList) :
    tops.length = arr.size := by
  simpa using congrArg List.length hl

theorem tops_size {tops : List (List (List UInt8))} {arr : Array ℕ} (hl : tops.map List.length = arr.toList)
    {e : ℕ} (he : e < tops.length) {top : Array ByteArray} (hye : tops[e] = top.toList.map (·.toList)) :
    top.size = arr.getD e 0 := by
  rw [← tops_len hl he, hye, List.length_map, Array.length_toList]

/-! ## The later caps' statements and slots -/

theorem levelCap_at {sh : VStar.Algebra.Shape} {K : Array VStar.Stmt} (h : VStar.levelCapStatements sh = .ok K)
    {g : ℕ} (hg : g < K.size) : ∃ N, K[g] = .cap (g + 1) N := by
  unfold VStar.levelCapStatements at h
  cases hs : sh.sched with
  | error e => rw [hs] at h; cases h
  | ok s =>
    rw [hs] at h
    cases h
    exact ⟨2 * 2 ^ s.levels[g + 1]!.cap, by simp⟩

theorem topsListed_at {i : RecVStar.Inner} {K : Array VStar.Stmt} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.checkTopsListed i K = .ok ()) (hrs : repSched i = some rs)
    {g n : ℕ} (hn : (RecVStar.topSizes rs)[g]? = some n) (hg : g < K.size) : K[g].instances = 2 * n := by
  unfold RecVStar.checkTopsListed at h
  unfold repSched at hrs
  simp only [hrs] at h
  split at h
  · cases h
  · rename_i hc
    have he : K.toList.map (·.instances) = (RecVStar.topSizes rs).map (2 * ·) := by
      simpa [bne_iff_ne] using hc
    have e := congrArg (·[g]?) he
    simp only [List.getElem?_map, Array.getElem?_toList, hn, Option.map_some] at e
    rw [Array.getElem?_eq_getElem hg, Option.map_some] at e
    exact Option.some.inj e

theorem list_at {A C K : Array VStar.Stmt} {N g : ℕ} (hg : g < K.size) :
    (A ++ C ++ (K ++ #[.cap 0 N]))[A.size + C.size + g]? = some K[g] := by
  rw [Array.getElem?_append_right (by simp), Array.size_append, Nat.add_sub_cancel_left,
    Array.getElem?_append_left hg, Array.getElem?_eq_getElem hg]

/-- The schedule's `topsBefore rs k + e`-th top is round `k`'s `e`-th. -/
theorem topSizes_at {rs : Array RecVStar.SchedRound} {k e : ℕ} (hk : k < rs.size)
    (he : e < ((rs.getD k default).2.1).size) :
    (RecVStar.topSizes rs)[topsBefore rs k + e]? = some ((rs.getD k default).2.1[e]) := by
  unfold topsBefore RecVStar.topSizes
  have hd : rs.toList.drop k = rs.getD k default :: rs.toList.drop (k + 1) := by
    rw [List.drop_eq_getElem_cons (show k < rs.toList.length by simpa using hk)]
    simp [Array.getD, hk]
  have hs : rs.toList = rs.toList.take k ++ (rs.getD k default :: rs.toList.drop (k + 1)) := by
    rw [← hd, List.take_append_drop]
  have hf := congrArg (List.flatMap fun w : RecVStar.SchedRound => w.2.1.toList) hs
  rw [List.flatMap_append, List.flatMap_cons] at hf
  rw [hf, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left,
    List.getElem?_append_left (by simpa using he)]
  simp

/-- **The schedule's `g`-th top's cap**: V\*'s statement at `capsFrom + g` is the later cap `.cap (g + 1) (2n)`, `n`
the top's nodes: the list's later caps follow V\*'s and candidate 3's statements (`FlockC3.Reps.list_ok`), the `g`-th
at level `g + 1` (`VStar.levelCapStatements`), holding both reps' nodes (`RecVStar.checkTopsListed`). -/
theorem capStmt_at {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {g n : ℕ} (hn : (RecVStar.topSizes rs)[g]? = some n) :
    (RecVStar.statements r.shape).toOption.bind (·[capsFrom r rs + g]?) = some (.cap (g + 1) (2 * n)) := by
  obtain ⟨vs, hvs, hlen⟩ := readX_statements hr
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, _, _, K, N, hK, hT, rfl⟩ := FlockC3.Reps.list_ok hvs
  have hsz := topsListed_size hT hrs
  obtain ⟨hg', -⟩ := List.getElem?_eq_some_iff.1 hn
  have hg : g < K.size := by rw [hsz]; exact hg'
  obtain ⟨N', hN'⟩ := levelCap_at hK hg
  have hi := topsListed_at hT hrs hn hg
  rw [hN'] at hi
  have hcf : capsFrom r rs + g = A.size + C.size + g := by
    unfold capsFrom
    rw [hlen, ← hsz]
    simp only [Array.size_append, Array.size_singleton]
    omega
  rw [hvs, hcf]
  show (A ++ C ++ (K ++ #[.cap 0 N]))[A.size + C.size + g]? = _
  rw [list_at hg, hN']
  simp only [VStar.Stmt.instances] at hi
  rw [hi]

/-- **A staged part has a circuit**: its setup parsed the raw part with the staged public file
(`RecRaw.ins_base`), so it parses the raw part as it is (`FlockDeploy.ins_publicFile`), at the same tags, circuit file
and tables (`rawIns_publicFile`), where the first table's circuit parsed (`table0`). -/
theorem staged_circuit {st : Stage} {s : ℕ} {x xO : Msg} (h : StagedPart st s x xO) {r : XR}
    (hr : readX x = some r) : ∃ c, r.w.parts[s]?.bind partCircuit = some c := by
  obtain ⟨p, _, _, hp, ht, hpi, -, -, -, -⟩ := stagedTables h
  obtain ⟨ms, i, spec, sts, rfl, hsu, -⟩ := h
  obtain ⟨hins, -⟩ := setupO_some hsu
  have hread : readOfO (staged st x ms s) = some i.base.read := by rw [readOfO, hins, Option.map_some]
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at hins; cases hins
  obtain ⟨r', pt, hr', hpt, hd⟩ := staged_decode hne
  rw [hr] at hr'
  cases hr'
  unfold insO at hins
  rw [hd] at hins
  simp only [Option.bind_eq_bind, Option.bind_some] at hins
  split_ifs at hins
  · simp at hins
  have h' : (partRaw r st x ms s pt).ins.toOption = some i := by simpa using hins
  have hb := RecRaw.ins_base (toOption_eq_some h')
  obtain ⟨ins, hins'⟩ := FlockDeploy.ins_publicFile (w := pt.base) (pf := st.pub x ms s) hb
  obtain ⟨htg, hcf, htb⟩ := rawIns_publicFile hins' hb
  obtain ⟨-, hcf', htb', -⟩ := rawIns_fields hins'
  obtain ⟨hr0, hts⟩ := tablesOf_of_parse hp
  have he : p.toRead = i.base.read := Option.some.inj (hr0.symm.trans hread)
  obtain ⟨st0, -, hc0, -⟩ := Flock.Readable.table0 hts ht hpi
  rw [he, htg, hcf, htb, hcf', htb'] at hc0
  refine ⟨st0.c, ?_⟩
  rw [hpt, Option.bind_some]
  unfold partCircuit
  simp [hins', Except.toOption, hc0]

theorem topRow_succ {lk : RecVStar.InnerLink} {g m k0 k1 : ℕ} {t0 t1 : Array ByteArray} {wh : String}
    {cc : ByteArray} (h0 : (lk.tops.getD 0 #[])[g]? = some (k0, t0)) (h1 : (lk.tops.getD 1 #[])[g]? = some (k1, t1))
    (h : RecVStar.topRow lk (g + 1) m = .ok (wh, cc)) :
    t0[m]? = some cc ∨ (t0[m]? = none ∧ t1[m - t0.size]? = some cc) := by
  rw [getD_getElem?] at h0 h1
  unfold RecVStar.topRow at h
  simp only [Nat.add_one_ne_zero, beq_iff_eq, ite_false, Nat.add_sub_cancel, h0, h1] at h
  split at h
  · cases h; left; assumption
  · cases h; right; constructor <;> assumption
  · cases h

theorem node_getD {a : Array ByteArray} {i : ℕ} {c : ByteArray} (h : a[i]? = some c) : a.getD i default = c := by
  rw [Array.getD_eq_getD_getElem?, h, Option.getD_some]

/-- **Row `rep · n + t` of a later cap is node `t` of rep `rep`'s top**: `topRow` reads rep 0's `n` nodes, then rep
1's. -/
theorem topRow_rep {lk : RecVStar.InnerLink} {g n rep k0 k1 kr : ℕ} {t0 t1 tr : Array ByteArray}
    (h0 : (lk.tops.getD 0 #[])[g]? = some (k0, t0)) (h1 : (lk.tops.getD 1 #[])[g]? = some (k1, t1))
    (hr : (lk.tops.getD rep #[])[g]? = some (kr, tr)) (hrep : rep < 2) (hn0 : t0.size = n) {t : ℕ} (ht : t < n)
    {wh : String} {cc : ByteArray} (h : RecVStar.topRow lk (g + 1) (rep * n + t) = .ok (wh, cc)) :
    cc = tr.getD t default := by
  rcases (by omega : rep = 0 ∨ rep = 1) with rfl | rfl
  · have e := h0.symm.trans hr
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    obtain ⟨-, rfl⟩ := e
    rw [Nat.zero_mul, Nat.zero_add] at h
    rcases topRow_succ h0 h1 h with h' | ⟨h', -⟩
    · exact (node_getD h').symm
    · rw [Array.getElem?_eq_getElem (by omega)] at h'; cases h'
  · have e := h1.symm.trans hr
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    obtain ⟨-, rfl⟩ := e
    rw [Nat.one_mul] at h
    rcases topRow_succ h0 h1 h with h' | ⟨-, h''⟩
    · rw [Array.getElem?_eq_none (by omega)] at h'; cases h'
    · rw [hn0, Nat.add_sub_cancel_left] at h''
      exact (node_getD h'').symm

/-! ## The pieces' bytes -/

theorem hmCommit_length (c : Circuit) (q n : ℕ) : (hmCommit c q n).length = 1024 := by
  simp [hmCommit]

theorem pieceLen_topPiece (r : XR) (s n rep : ℕ) :
    Claims.Recursion.VStarB.pieceLen (topPiece r s n rep) = 128 * n := by
  simp only [Claims.Recursion.VStarB.pieceLen, topPiece, List.map_map, Function.comp_def, hmCommit_length,
    List.length_nil]
  rw [sum_map_const, List.length_range]
  omega

theorem topPiece_at {r : XR} {s : ℕ} {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c) {q : ℕ}
    (hq : capPort c = some q) (n rep : ℕ) :
    topPiece r s n rep = (List.range n).map fun t => (([] : Msg), hmCommit c q (rep * n + t)) := by
  simp only [topPiece, hc, hq, Option.getD_some]

theorem readSegs_topPiece {w : Claims.Flock.Wit} {c : Circuit} {q rep : ℕ} {top : Array ByteArray}
    (hb : ∀ t < top.size,
      Claims.Recursion.VStarB.readForms w (hmCommit c q (rep * top.size + t)) = (top.getD t default).toList) :
    Claims.Recursion.VStarB.readSegs w
        ((List.range top.size).map fun t => (([] : Msg), hmCommit c q (rep * top.size + t))) =
      (top.toList.map (·.toList)).flatten := by
  unfold Claims.Recursion.VStarB.readSegs
  rw [List.flatMap_map]
  refine flatMap_range_nodes top _ fun t ht => ?_
  simp only [List.nil_append]
  exact hb t ht

theorem rowPiece_mem {r : XR} {j : ℕ} {p : ℕ × List (Msg × List HmNets.Lin)} (hp : p ∈ rowPiece r j) :
    ∃ h ∈ heldRows r p.1, h.1 = j ∧ ∃ c, r.w.parts[p.1]?.bind partCircuit = some c ∧
      p.2 = [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)] := by
  unfold rowPiece at hp
  simp only [Option.mem_toList] at hp
  obtain ⟨a, -, hfa⟩ := List.exists_of_findSome?_eq_some hp
  rw [Option.bind_eq_some_iff] at hfa
  obtain ⟨h, hh, hfa⟩ := hfa
  rw [Option.map_eq_some_iff] at hfa
  obtain ⟨c, hc, rfl⟩ := hfa
  exact ⟨h, List.mem_of_find?_eq_some hh, by simpa using List.find?_some hh, c, hc, rfl⟩

theorem toList_length_le {α : Type} (o : Option α) : o.toList.length ≤ 1 := by
  cases o <;> simp

theorem rowPiece_length (r : XR) (j : ℕ) : (rowPiece r j).length ≤ 1 := toList_length_le _

theorem split_pieces {α : Type} {T R pre post : List α} {p : α} (h : T ++ R = pre ++ p :: post) :
    (∃ e, ∃ he : e < T.length, pre = T.take e ∧ p = T[e]) ∨ (∃ R', pre = T ++ R' ∧ R = R' ++ p :: post) := by
  rcases List.append_eq_append_iff.1 h with ⟨a', h1, h2⟩ | ⟨c', h1, h2⟩
  · exact Or.inr ⟨a', h1, h2⟩
  · cases c' with
    | nil =>
      simp only [List.nil_append, List.append_nil] at h1 h2
      exact Or.inr ⟨[], by simp [h1], h2.symm⟩
    | cons c cs =>
      simp only [List.cons_append, List.cons.injEq] at h2
      obtain ⟨rfl, -⟩ := h2
      refine Or.inl ⟨pre.length, by simp [h1], by simp [h1], by simp [h1]⟩

theorem flatten_drop_take {α : Type} (L : List (List α)) (C : List α) {e : ℕ} (he : e < L.length) :
    ((L.flatten ++ C).drop ((L.take e).map List.length).sum).take L[e].length = L[e] := by
  have hs : L = L.take e ++ L[e] :: L.drop (e + 1) := by
    rw [← List.drop_eq_getElem_cons he, List.take_append_drop]
  have hf := congrArg List.flatten hs
  rw [List.flatten_append, List.flatten_cons] at hf
  rw [hf, List.append_assoc, List.append_assoc, ← List.length_flatten, List.drop_left, List.take_left]

theorem flatten_drop {α : Type} (L : List (List α)) (C : List α) :
    (L.flatten ++ C).drop (L.map List.length).sum = C := by
  rw [← List.length_flatten, List.drop_left]

/-- **A tops round's pieces at a staged part read the record's message at their offsets**: stream round `j < 2R` is
rep `j / R`'s round `k = j % R`, whose message is its tops' nodes, then its `c_k` (`round_view`). Each top's piece, at
its later cap (`capStmt_at`), reads its rep's nodes of that top (`capRow_bytes`, `topRow_rep`), `128 · n` bytes after
the earlier tops' (`pieceLen_topPiece`, `innerLink_nodes`); the row's reads its rep's `c_k` (`heldRow_bytes`) after
every top. -/
theorem topsPieces_opens {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR}
    (hr : readX x = some r) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) {j : ℕ}
    (hj : j < 2 * rs.size) {pre post : List (ℕ × List (Msg × List HmNets.Lin))}
    {p : ℕ × List (Msg × List HmNets.Lin)} (hpp : topsPieces x j = pre ++ p :: post) (hps : p.1 = s) :
    ((cmtTr recRead x (j + 1) xO).drop (pre.map fun q => Claims.Recursion.VStarB.pieceLen q.2).sum).take
      (Claims.Recursion.VStarB.pieceLen p.2) = Claims.Recursion.VStarB.readSegs w p.2 := by
  have hpos : 0 < rs.size := by omega
  have htp : topsPieces x j = topsPiecesAt r rs j := by simp [topsPieces, hr, hrs]
  rw [htp] at hpp
  have hk : j % rs.size < rs.size := Nat.mod_lt j hpos
  have hrep : j / rs.size < 2 := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hj)
  obtain ⟨i, lk, cs, hi, hil, hcs, hview⟩ := round_view hS hr hrs
  obtain ⟨y, hmsg, hlens, htops, hcomm⟩ := hview _ hrep _ hk
  rw [Nat.div_add_mod' j rs.size] at hmsg
  rw [hmsg]
  unfold topsPiecesAt at hpp
  by_cases hE : ((rs.getD (j % rs.size) default).2.1).isEmpty = true
  · rw [ite_eq_left hE] at hpp
    simp at hpp
  rw [ite_eq_right hE] at hpp
  have hlenT := tops_length hlens
  have hflat : ∀ e (he : e < y.tops.length), (y.tops[e]).flatten.length = 128 * y.tops[e].length := by
    intro e he
    obtain ⟨kt, top, -, hye, h128⟩ := htops e he
    rw [hye, flatten_nodes_length h128]
    simp
  have hlensT : ((List.range ((rs.getD (j % rs.size) default).2.1).size).map (capPieceAt r rs j)).map
      (fun q => Claims.Recursion.VStarB.pieceLen q.2) = (y.tops.map List.flatten).map List.length := by
    apply List.ext_getElem (by simp [hlenT])
    intro e h1 h2
    have he : e < y.tops.length := by simpa using h2
    simp only [List.getElem_map, List.getElem_range, capPieceAt, pieceLen_topPiece]
    rw [hflat e he, tops_len hlens he]
  rcases split_pieces hpp with ⟨e, he, rfl, rfl⟩ | ⟨R', rfl, hR⟩
  · have heT : e < ((rs.getD (j % rs.size) default).2.1).size := by simpa using he
    have hey : e < y.tops.length := by omega
    have hTe : ((List.range ((rs.getD (j % rs.size) default).2.1).size).map (capPieceAt r rs j))[e] =
        capPieceAt r rs j e := by simp
    rw [hTe] at hps ⊢
    rw [List.map_take, hlensT, ← List.map_take]
    simp only [capPieceAt] at hps ⊢
    subst hps
    have hLe : ∀ S, Claims.Recursion.VStarB.pieceLen
        (topPiece r S (((rs.getD (j % rs.size) default).2.1).getD e 0) (j / rs.size)) =
        ((y.tops.map List.flatten)[e]'(by simpa using hey)).length := fun S => by
      rw [pieceLen_topPiece, List.getElem_map, hflat e hey, tops_len hlens hey]
    rw [hLe, flatten_drop_take _ _ (by simpa using hey), List.getElem_map]
    obtain ⟨kt, top, hkt, hye, -⟩ := htops e hey
    have hts : top.size = ((rs.getD (j % rs.size) default).2.1).getD e 0 := tops_size hlens hey hye
    have hnS : (RecVStar.topSizes rs)[topsBefore rs (j % rs.size) + e]? =
        some (((rs.getD (j % rs.size) default).2.1).getD e 0) := by
      have hgd : ∀ (a : Array ℕ) (i : ℕ) (h : i < a.size), a.getD i 0 = a[i] := fun a i h => by simp [Array.getD, h]
      rw [topSizes_at hk heT, hgd _ _ heT]
    have hsv := capStmt_at hr hrs hnS
    rw [← Nat.add_assoc] at hsv
    obtain ⟨c, hpc⟩ := staged_circuit hS hr
    obtain ⟨q, hq, -, hbytes⟩ := capRow_bytes hS hck hr hpc hsv hi hil
    have hside : ∀ x' < 2, ∃ kt' top', (lk.tops.getD x' #[])[topsBefore rs (j % rs.size) + e]? = some (kt', top') ∧
        top'.size = ((rs.getD (j % rs.size) default).2.1).getD e 0 := by
      intro x' hx'
      obtain ⟨y', -, hlens', htops', -⟩ := hview x' hx' _ hk
      have hey' : e < y'.tops.length := by rw [tops_length hlens']; exact heT
      obtain ⟨kt', top', hkt', hye', -⟩ := htops' e hey'
      exact ⟨kt', top', hkt', tops_size hlens' hey' hye'⟩
    obtain ⟨k0, t0, h0, hs0⟩ := hside 0 (by omega)
    obtain ⟨k1, t1, h1, -⟩ := hside 1 (by omega)
    rw [hye, topPiece_at hpc hq, ← hts]
    refine (readSegs_topPiece fun t ht => ?_).symm
    have hm : j / rs.size * top.size + t < 2 * ((rs.getD (j % rs.size) default).2.1).getD e 0 := by
      rw [← hts]
      have h01 : j / rs.size = 0 ∨ j / rs.size = 1 := by
        rcases Nat.lt_or_ge (j / rs.size) 1 with h | h
        · exact Or.inl (Nat.lt_one_iff.1 h)
        · exact Or.inr (Nat.le_antisymm (Nat.lt_succ_iff.1 hrep) h)
      rcases h01 with h | h <;> rw [h] <;> omega
    obtain ⟨wh, cc, htr, hb⟩ := hbytes _ hm
    rw [hb, topRow_rep h0 h1 hkt hrep (hs0.trans hts.symm) ht htr]
  · have hl := congrArg List.length hR
    simp only [List.length_append, List.length_cons] at hl
    have := rowPiece_length r j
    obtain rfl : R' = [] := List.eq_nil_of_length_eq_zero (by omega)
    have hpm : p ∈ rowPiece r j := by rw [hR]; simp
    obtain ⟨h, hh, hj1, c, hpc, hp2⟩ := rowPiece_mem hpm
    rw [hps] at hh hpc
    obtain ⟨k', rs', kk, cc, hrs', hmr, -, hkk, hbytes⟩ := heldRow_bytes hS hck hr hpc hi hcs hh
    rw [hrs] at hrs'
    cases hrs'
    obtain ⟨kr, hkr, hjr⟩ := Option.map_eq_some_iff.1 hmr
    have hkR := commitRounds_lt hkr
    rw [hj1] at hjr
    have e1 : j / rs.size = h.2.2.2 := by
      rw [← hjr, Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hkR, Nat.zero_add]
    have e2 : j % rs.size = kr := by
      rw [← hjr, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hkR]
    have hyc := hcomm k' kk cc (by rw [e2]; exact hkr) (by rw [e1]; exact hkk)
    have hcl : cc.toList.length = 128 := by
      rw [← hbytes, Claims.Recursion.VStarB.readForms_length, hmCommit_length]
    rw [List.append_nil, hlensT, flatten_drop, hp2, hyc]
    simp only [Claims.Recursion.VStarB.readSegs, Claims.Recursion.VStarB.pieceLen, List.flatMap_cons,
      List.flatMap_nil, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.nil_append, List.append_nil,
      List.flatten_cons, List.flatten_nil, List.length_nil, hmCommit_length, hbytes]
    exact List.take_of_length_le (by omega)

/-! ## `PartOpensB` at `layoutBTops` and `recRead` -/

/-- **`PartOpensB` at `layoutBTops` and the coin server's record**, at every shape whose statements the deployment's
reader reads (`hD`) and where the inner verifier's rounds are the V\* schedule's (`SchedMatches`): every round a part
holds whole is the record's (`partOpensB_layoutBTops`), and each piece of a stream round that commits tops reads the
record's message at its offset (`topsPieces_opens`). -/
theorem partOpensB_recRead (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (b : Brings)
    {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop} {s : ℕ}
    (hD : ∀ x st, D.parse x = some st → parse x = some st) (hM : SchedMatches shape) :
    Claims.Recursion.VStarB.PartOpensB H D (deployZkRec Flock.Readable.relation) (layoutBTops fin) shape
      (showsPart recRead b) (cmtTr recRead) s := by
  refine partOpensB_layoutBTops fin b ?_
  intro x st hx hf xO w hck hS j hj pre p post hpp hps
  obtain ⟨r, rs, hr, hrs, hK⟩ := hM x st (hD x st hx) hf
  have hL : layoutOf.rounds x = 2 + 2 * rs.size := hK
  exact topsPieces_opens hS hck hr hrs (by omega) hpp hps

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)` and the coin server's record**:
`ksLin2_deployed_open` at `recRead`, with `PartOpensB` from `partOpensB_recRead` at `schedMatches_dense`. What stays
open at the dense form is `PartChecksBV` at `showsPart recRead` and `DecompB`; then the readers' `ReadersTrue` and
`recRead` reading the coin server's transcript back (`ReadsBackB`). This bounds no number by itself: those premises are
open. -/
theorem ksLin2_deployed_recRead (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead b) (cmtTr recRead) chk s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  refine ksLin2_deployed_open b rest wE join final w recRead hrb (fun shape hsh => ?_) hRT
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨chk, hPC, hDc⟩ := hP _ hsh
  exact ⟨chk, hPC, fun _ => partOpensB_recRead (finOpen rest) b (fun _ _ h => h) schedMatches_dense, hDc⟩

/-- **`BitsLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)` and the coin server's
record**: `bitsLin2_deployed_open` at `recRead`, with `PartOpensB` from `partOpensB_recRead`. It reads `StrictBirthday`
and `ExpectedCR`, the named SHA-512 collision assumptions, and `ksLin2_deployed_recRead`'s open premises: a bound on a
number under named premises, not a proved security level. -/
theorem bitsLin2_deployed_recRead (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead b) (cmtTr recRead) chk s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  refine bitsLin2_deployed_open b rest wE join final w recRead hrb (fun shape hsh => ?_) hRT
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨chk, hPC, hDc⟩ := hP _ hsh
  exact ⟨chk, hPC, fun _ => partOpensB_recRead (finOpen rest) b (fun _ _ h => h) schedMatches_dense, hDc⟩

end RecDeploy
