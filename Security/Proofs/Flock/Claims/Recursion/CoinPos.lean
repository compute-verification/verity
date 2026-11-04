import Proofs.Flock.Claims.Recursion.CoinTie

/-!
# The coin server's round the coin tie names is the only one of its stream and round number

`CoinTie` gives, for each round `m` of rep `r`'s coins as check 5 loads them, a round `g` of the coin server's record
`R` of that rep's stream at round `sp m` whose coins they are. Check 5's verdict reads the inner session's rounds by
stream and position, so a consumer of the tie that pins `g` by its stream and round number needs `g` to be the only
such round of `R`. This file proves that from what the verifier checks, with no premise:

* `RecSetup.checkReads` holds `R`'s rounds, in order, to the transcript JSON's (`jsonRounds_both`), so two rounds of
  `R` with one stream and round number are two such rounds of the JSON;
* `RecVStar.innerCommits` checks that the JSON's rounds of each stream the inner record names, sorted by round number
  (`transcriptRounds`, a `qsort` of a `filter`, so a permutation of them), number `0 … n-1` (`repCoins_places`), so
  there is at most one at each number (`count_le_one`).

`coinTie_unique` is that in the verifier's code, and `staged_coinTie_unique` and `staged_coinTie_drawn_unique` are
`staged_coinTie` and `staged_coinTie_drawn` with each round `g` unique.
-/

namespace RecDeploy.CoinPos

open Flock Lean
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy.CoinTie

theorem qsort_perm {α : Type} (as : Array α) (lt : α → α → Bool) : (as.qsort lt).Perm as := by
  unfold Array.qsort
  split
  · exact ⟨List.Perm.refl _⟩
  · exact FlockSoundness.Audit.DrawExec.sort_perm lt _ _ _ _ _ _ _ rfl

theorem utf8_inj {s t : String} (h : s.toUTF8.toList = t.toUTF8.toList) : s = t := by
  rw [byteArray_toList_eq, byteArray_toList_eq] at h
  exact String.toByteArray_inj.1 (ByteArray.ext (Array.ext' h))

theorem two_le_countP {α : Type} {p : α → Bool} :
    ∀ (l : List α) (i j : Nat) (hi : i < l.length) (hj : j < l.length), i ≠ j → p l[i] = true → p l[j] = true →
      2 ≤ l.countP p
  | [], _, _, hi, _, _, _, _ => by simp at hi
  | _ :: _, 0, 0, _, _, h, _, _ => absurd rfl h
  | a :: l, 0, j + 1, _, hj, _, hpi, hpj => by
    have hpos : 0 < l.countP p :=
      List.countP_pos_iff.2 ⟨l[j]'(by simpa using hj), List.getElem_mem _, by simpa using hpj⟩
    simp only [List.getElem_cons_zero] at hpi
    rw [List.countP_cons, if_pos hpi]
    omega
  | a :: l, i + 1, 0, hi, _, _, hpi, hpj => by
    have hpos : 0 < l.countP p :=
      List.countP_pos_iff.2 ⟨l[i]'(by simpa using hi), List.getElem_mem _, by simpa using hpi⟩
    simp only [List.getElem_cons_zero] at hpj
    rw [List.countP_cons, if_pos hpj]
    omega
  | a :: l, i + 1, j + 1, hi, hj, h, hpi, hpj => by
    have := two_le_countP l i j (by simpa using hi) (by simpa using hj) (by omega) (by simpa using hpi)
      (by simpa using hpj)
    rw [List.countP_cons]
    split <;> omega

/-- A transcript JSON round's round number is `κ`. -/
def kIs (κ : Nat) (x : Json) : Bool := (x.getObjValAs? Nat "k").toOption == some κ

/-- A transcript JSON round's stream is `s`. -/
def streamIs (s : String) (x : Json) : Bool := (x.getObjValAs? String "stream").toOption == some s

/-- **At most one of the transcript JSON's rounds has a checked stream and a given round number**: `innerCommits`
numbers that stream's rounds `0 … n-1` (`repCoins_places`), and they are its rounds, permuted (`qsort_perm`). -/
theorem count_le_one {name : String} {rounds streams : Array Json} (h : CoinsAgree name rounds streams) (κ : Nat) :
    rounds.toList.countP (fun x => kIs κ x && streamIs name x) ≤ 1 := by
  obtain ⟨cc, hcc, -⟩ := h
  obtain ⟨hk, -⟩ := repCoins_places hcc
  have hp : (transcriptRounds name rounds).toList.Perm (rounds.filter (streamIs name)).toList := by
    unfold transcriptRounds
    exact Array.perm_iff_toList_perm.1 (qsort_perm _ _)
  rw [← List.countP_filter, ← Array.toList_filter, ← hp.countP_eq (kIs κ)]
  have hm : (transcriptRounds name rounds).toList.countP (kIs κ) =
      ((transcriptRounds name rounds).toList.map fun r => (r.getObjValAs? Nat "k").toOption).countP
        (· == some κ) := by
    rw [List.countP_map]
    rfl
  rw [hm, hk, List.countP_map]
  have hc : (List.range (transcriptRounds name rounds).size).countP ((· == some κ) ∘ some) =
      (List.range (transcriptRounds name rounds).size).count κ :=
    List.countP_congr fun a _ => by simp
  rw [hc, List.count_range]
  split <;> omega

/-- **Two rounds of `R` of a checked stream with one round number are one round**, where `R`'s rounds are the
transcript JSON's, round for round (`jsonRounds_both`). -/
theorem stream_unique {name : String} {rounds streams : Array Json} {R : RecRecord.Record}
    (hagree : CoinsAgree name rounds streams)
    (hBoth : Pairs (fun x y => ∃ coins, (RecVStar.coinsOf x).toList.mapM f128OfHex = .ok coins ∧
      y = ((x.getObjValAs? String "stream").toOption.map (·.toUTF8.toList), (x.getObjValAs? Nat "k").toOption,
        coins)) rounds.toList (R.rounds.map fun x => (some x.stream, some x.k, x.coins)))
    {g g' : Nat} (hg : g < R.rounds.length) (hg' : g' < R.rounds.length)
    (hs : R.rounds[g].stream = name.toUTF8.toList) (hs' : R.rounds[g'].stream = name.toUTF8.toList)
    (hk : R.rounds[g'].k = R.rounds[g].k) : g' = g := by
  have hlen := hBoth.length_eq
  simp only [Array.length_toList, List.length_map] at hlen
  have hat : ∀ i (hi : i < R.rounds.length), R.rounds[i].stream = name.toUTF8.toList →
      (kIs R.rounds[i].k (rounds[i]'(by omega)) && streamIs name (rounds[i]'(by omega))) = true := by
    intro i hi hsi
    obtain ⟨coins, -, hy⟩ := hBoth.get i (by simp only [Array.length_toList]; omega)
      (by simp only [List.length_map]; omega)
    simp only [List.getElem_map, Array.getElem_toList, Prod.mk.injEq] at hy
    obtain ⟨h1, h2, -⟩ := hy
    cases hS : (rounds[i].getObjValAs? String "stream").toOption with
    | none => rw [hS] at h1; cases h1
    | some s =>
      rw [hS, Option.map_some, Option.some.injEq, hsi] at h1
      have hns := utf8_inj h1
      simp [kIs, streamIs, hS, ← h2, hns]
  refine Classical.byContradiction fun hne => ?_
  have h2 := two_le_countP (p := fun x => kIs R.rounds[g].k x && streamIs name x) rounds.toList g' g
    (by simp only [Array.length_toList]; omega) (by simp only [Array.length_toList]; omega) hne
    (by simpa [hk] using hat g' hg' hs') (by simpa using hat g hg hs)
  have h1 := count_le_one hagree R.rounds[g].k
  omega

/-- **The coin tie's rounds are unique, in the verifier's code**: under `coinTie_core`'s checks, the record `R` that
`checkReads` reads has at most one round of the decoded record's stream `name` at each round number. -/
theorem coinTie_unique {tr : Json} {record : String} {reg : Option Json} {bound : Option String}
    {shape : Option RecVStar.StackedShape} {cs : RecVStar.InnerCommits} {ish : RecVStar.Inner} {tt : String}
    {spec : SessionSpec} {sess : RunRecord} {name : String} {st : StreamRec}
    (hic : RecVStar.innerCommits tr record reg bound shape = .ok cs)
    (hR : RecSetup.checkReads ish tt tr record cs = .ok ())
    (hdec : Record.decode spec record = .ok sess) (hst : sess.streams.find? (·.name == name) = some st) :
    ∃ R, RecRecord.readFull tt.toUTF8.toList = some R ∧
      ∀ g g' (hg : g < R.rounds.length) (hg' : g' < R.rounds.length),
        R.rounds[g].stream = name.toUTF8.toList → R.rounds[g'].stream = name.toUTF8.toList →
        R.rounds[g'].k = R.rounds[g].k → g' = g := by
  obtain ⟨R, hRead, hJR⟩ := checkReads_rounds hR
  obtain ⟨rounds, rj, streams, hrounds, hparse, hstreams, hall⟩ := innerCommits_reps hic
  obtain ⟨j, sj, hparse', harr, hboth⟩ := decode_streams hdec
  rw [hparse] at hparse'
  cases hparse'
  rw [arr_as harr] at hstreams
  have hss := (Except.ok.inj hstreams).symm
  subst hss
  have hst' : sess.streams.toList.find? (·.name == name) = some st := by
    rw [Array.find?_toList]; exact hst
  obtain ⟨x, hx, hxs⟩ := find?_forall₂ (p := fun x => (x.getObjValAs? String "stream").toOption == some name)
    (fun x st hxs => by simp only [str_as hxs.1, Except.toOption]; simp) hboth hst'
  have hname : st.name = name := by simpa using List.find?_some hst'
  have hmem : name ∈ streams.toList.filterMap (fun x => (x.getObjValAs? String "stream").toOption) :=
    List.mem_filterMap.2 ⟨x, List.mem_of_find?_eq_some hx,
      (by rw [str_as hxs.1, hname]; rfl : (x.getObjValAs? String "stream").toOption = some name)⟩
  exact ⟨R, hRead, fun g g' hg hg' hs hs' hk =>
    stream_unique (hall name hmem) (jsonRounds_both hrounds hJR) hg hg' hs hs' hk⟩


/-! ## At a staged part -/

/-- **`staged_coinTie`, with each round unique**: the round `g` of `R` that holds rep `r`'s round `m`'s coins is the
only round of `R` of its stream at its round number. -/
theorem staged_coinTie_unique {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) :
    ∃ (i : RecIns) (R : Flock.RecRecord.Record), insO xO = some i ∧
      shownOf recRead x xO = some (R.msgs.map fun mc => (mc.1, mc.2.toArray)) ∧
      ∀ f : RecCoins.Files, RecCoins.Source.atC3 i.src i.inner i.commits = some (.files f) →
      ∀ (unit : String) (inn : RecCoins.Inner), RecCoins.Inner.load unit f = .ok inn →
      ∀ (r : ℕ) (rounds : Array (Array F128)), inn.rounds r = .ok rounds →
        ∃ t : String, ∃ sp : ℕ → ℕ, (sp = id ∨ ∃ j, sp = fun m => if m < j then m else m + 1) ∧
          ∀ m (hm : m < rounds.size), ∃ g, ∃ hg : g < R.rounds.length,
            R.rounds[g].stream = (s!"{t}/rep{r}").toUTF8.toList ∧ R.rounds[g].k = sp m ∧
            R.rounds[g].coins = rounds[m].toList ∧
            ∀ g' (hg' : g' < R.rounds.length), R.rounds[g'].stream = R.rounds[g].stream →
              R.rounds[g'].k = R.rounds[g].k → g' = g := by
  obtain ⟨ms, i, _spec, _sts, rfl, hsu, -⟩ := hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨xr, -, hinner, -⟩ := insO_staged hins
  obtain ⟨wr, hd, htr, -, -⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hd] at hd'
  cases hd'
  have hc := RecRaw.ins_commits hw
  rw [hinner] at hc
  obtain ⟨cs, tt, tr, record, hcs, htx, -, hsrc, hR, reg, d, shp, hic⟩ := commitsOf_tie htr hc
  obtain ⟨R, hRead, -⟩ := checkReads_rounds hR
  have hb : (st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  refine ⟨i, R, hins, ?_, fun f hf unit inn hload r rounds hr => ?_⟩
  · simp only [shownOf, hd, htr, Option.bind_some, recRead, hb, hRead, Option.map_some]
  obtain ⟨spec', sess, st', hdec, hfind, rfl⟩ := load_rounds hload hr
  have hrec : f.record = record := by
    rw [atC3_record hf] at hsrc
    exact Option.some.inj hsrc
  rw [hrec] at hdec
  obtain ⟨R', hRead', sp, hsp, hall⟩ := coinTie_core
    (fun a lt x hx => (FlockSoundness.Audit.DrawExec.mem_qsort a lt x).1 hx) hic hR hdec hfind
  obtain ⟨R'', hRead'', huniq⟩ := coinTie_unique hic hR hdec hfind
  rw [hRead] at hRead' hRead''
  cases hRead'
  cases hRead''
  refine ⟨spec'.tables[0]!, sp, hsp, fun m hm => ?_⟩
  obtain ⟨g, hg, h1, h2, h3⟩ := hall m (by simpa using hm)
  exact ⟨g, hg, h1, h2, by simpa using h3, fun g' hg' hs hk => huniq g g' hg hg' h1 (hs.trans h1) hk⟩

/-- **`staged_coinTie_drawn`, with each round unique**: the inner round `g + 1` whose coin drew rep `r`'s round `m`'s
coins is the only round of `R` of its stream at its round number. -/
theorem staged_coinTie_drawn_unique {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {c : ℕ → Coin}
    (hc : showsTr recRead s x xO c) :
    ∃ (i : RecIns) (R : Flock.RecRecord.Record), insO xO = some i ∧
      shownOf recRead x xO = some (R.msgs.map fun mc => (mc.1, mc.2.toArray)) ∧
      ∀ f : RecCoins.Files, RecCoins.Source.atC3 i.src i.inner i.commits = some (.files f) →
      ∀ (unit : String) (inn : RecCoins.Inner), RecCoins.Inner.load unit f = .ok inn →
      ∀ (r : ℕ) (rounds : Array (Array F128)), inn.rounds r = .ok rounds →
        ∃ t : String, ∃ sp : ℕ → ℕ, (sp = id ∨ ∃ j, sp = fun m => if m < j then m else m + 1) ∧
          ∀ m (hm : m < rounds.size), ∃ g, ∃ hg : g < R.rounds.length,
            R.rounds[g].stream = (s!"{t}/rep{r}").toUTF8.toList ∧ R.rounds[g].k = sp m ∧
            Drawn (c (g + 1)) rounds[m] ∧
            ∀ g' (hg' : g' < R.rounds.length), R.rounds[g'].stream = R.rounds[g].stream →
              R.rounds[g'].k = R.rounds[g].k → g' = g := by
  obtain ⟨i, R, hins, hshow, hall⟩ := staged_coinTie_unique hS
  obtain ⟨ps, hps, hdrawn⟩ := hc
  rw [hshow] at hps
  cases hps
  refine ⟨i, R, hins, hshow, fun f hf unit inn hload r rounds hr => ?_⟩
  obtain ⟨t, sp, hsp, hm⟩ := hall f hf unit inn hload r rounds hr
  refine ⟨t, sp, hsp, fun m hmm => ?_⟩
  obtain ⟨g, hg, h1, h2, h3, h4⟩ := hm m hmm
  refine ⟨g, hg, h1, h2, ?_, h4⟩
  have hlen : g + 1 < (R.msgs.map fun mc => (mc.1, mc.2.toArray)).length := by
    simp [Flock.RecRecord.Record.msgs, hg]
  have hd := hdrawn (g + 1) hlen
  have hget : ((R.msgs.map fun mc => (mc.1, mc.2.toArray)).getD (g + 1) ([], #[])).2 = rounds[m] := by
    rw [List.getD_eq_getElem _ _ hlen]
    simp [Flock.RecRecord.Record.msgs, h3]
  rwa [hget] at hd

end RecDeploy.CoinPos
