import Proofs.Flock.Claims.Recursion.VStarBuild
import Proofs.Flock.C3.RegLevels.TopHold

/-!
# When the stager accepts a part (`VStar.In.partIn`)

`partIn` builds a level's port inputs (`levelIn`), an algebra part's (`algebraIn`) or a cap's (`capIn`), and refuses
every other statement. `StmtFits src st` is what each builder checks of the stager's inputs `src`, written out, and
`partIn_ok` is that `partIn` accepts wherever it holds:

* **a level** (`LevelInFits`): its openings number `n`, each of `LANES` lanes and `H` siblings, the level's tops are
  read (`topsAt`), and each opening's top node is among them. That every opening is in the chain (`at?`) is proved:
  `Array.qsort` keeps its elements (`qsort_mem`, through `TopHold.sort_spec`);
* **an algebra part** (`AlgInFits`): each rep has an opening (`repEnds`), and each port is `acc0`, `acc1` or an `m<i>`
  whose message round holds two rows;
* **a cap**: `capIn` accepts (`VStarPorts.capIn_ok`).

At the honest source (`srcOf`) the message rounds are `msgsOf`'s, each of two rows, so an algebra part's `m<i>` fits
wherever `i` is one of the schedule's message rounds (`algFits_srcOf`). The openings are the inner prover's `rest`.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

theorem mem_of_sameOn {α : Type} {as : Array α} {r : Vector α as.size} {lo hi : ℕ}
    (hS : FlockC3.RegLevels.TopHold.SameOn lo hi as.toVector r) (hlo : lo = 0) (hhi : as.size ≤ hi + 1) (x : α) :
    x ∈ r.toArray ↔ x ∈ as := by
  subst hlo
  rw [Vector.mem_toArray_iff, Vector.mem_iff_getElem, Array.mem_iff_getElem]
  constructor
  · rintro ⟨j, hj, rfl⟩
    obtain ⟨j', hj', -, -, e⟩ := (hS.2 r[j]).2 ⟨j, hj, Nat.zero_le _, by omega, rfl⟩
    exact ⟨j', hj', e⟩
  · rintro ⟨j, hj, rfl⟩
    obtain ⟨j', hj', -, -, e⟩ := (hS.2 as[j]).1 ⟨j, hj, Nat.zero_le _, by omega, rfl⟩
    exact ⟨j', hj', e⟩

/-- **`Array.qsort` keeps its elements**, under an order whose negation is transitive. -/
theorem qsort_mem {α : Type} (as : Array α) (lt : α → α → Bool) (hasym : ∀ a b, lt a b = true → lt b a = false)
    (htr : ∀ a b c, lt a b = false → lt b c = false → lt a c = false) (x : α) : x ∈ as.qsort lt ↔ x ∈ as := by
  unfold Array.qsort
  split
  · exact Iff.rfl
  · rename_i h0
    try simp only
    obtain ⟨hS, -⟩ := FlockC3.RegLevels.TopHold.sort_spec lt hasym htr _ as.toVector (min 0 (as.size - 1))
      (max (min 0 (as.size - 1)) (min (as.size - 1) (as.size - 1))) (by omega) (by omega) (by omega)
      (Nat.lt_succ_self _)
    exact mem_of_sameOn hS (Nat.zero_min _) (by omega) x

theorem before_asym (a b : _root_.VStar.In.Query) (h : _root_.VStar.In.before a b = true) :
    _root_.VStar.In.before b a = false := by
  simp [_root_.VStar.In.before] at h ⊢ <;> omega

theorem before_trans (a b c : _root_.VStar.In.Query) (h1 : _root_.VStar.In.before a b = false)
    (h2 : _root_.VStar.In.before b c = false) : _root_.VStar.In.before a c = false := by
  simp [_root_.VStar.In.before] at h1 h2 ⊢ <;> omega

/-- An opening is in the chain's order exactly where it is one of the stager's. -/
theorem mem_order {src : _root_.VStar.In.Src} {o : _root_.VStar.In.Query} :
    o ∈ _root_.VStar.In.order src ↔ o ∈ src.openings :=
  qsort_mem _ _ before_asym before_trans o

theorem at?_some {src : _root_.VStar.In.Src} {o : _root_.VStar.In.Query} (h : o ∈ src.openings) :
    ∃ p, _root_.VStar.In.at? src o = some p := by
  unfold _root_.VStar.In.at?
  cases hf : (_root_.VStar.In.order src).findIdx? (_root_.VStar.In.same o) with
  | some p => exact ⟨p, rfl⟩
  | none =>
    have := Array.findIdx?_eq_none_iff.1 hf o (mem_order.2 h)
    simp [_root_.VStar.In.same] at this

theorem mapM_some_of {α β : Type} {f : α → Option β} {xs : Array α} (h : ∀ a ∈ xs, ∃ b, f a = some b) :
    ∃ bs, xs.mapM f = some bs := by
  have hl : ∀ l : List α, (∀ a ∈ l, ∃ b, f a = some b) → ∃ bs, l.mapM f = some bs := by
    intro l
    induction l with
    | nil => intro _; exact ⟨[], rfl⟩
    | cons a l ih =>
      intro hl
      obtain ⟨b, hb⟩ := hl a List.mem_cons_self
      obtain ⟨bs, hbs⟩ := ih fun a' ha' => hl a' (List.mem_cons_of_mem _ ha')
      exact ⟨b :: bs, by simp [List.mapM_cons, hb, hbs]⟩
  obtain ⟨bs, hbs⟩ := hl xs.toList fun a ha => h a (Array.mem_toList_iff.1 ha)
  exact ⟨bs.toArray, by rw [Array.mapM_eq_mapM_toList, hbs]; rfl⟩

/-- **A rep with an opening has its ends** in the chain (`repEnds`). -/
theorem repEnds_ok {src : _root_.VStar.In.Src} {rep : ℕ} (h : ∃ o ∈ src.openings, o.rep = rep) :
    ∃ a b, _root_.VStar.In.repEnds src rep = .ok (a, b) := by
  obtain ⟨o, ho, hr⟩ := h
  obtain ⟨p, hp, hpo⟩ := Array.mem_iff_getElem.1 (mem_order.2 ho)
  have hin : p ∈ (List.range (_root_.VStar.In.order src).size).filter
      fun p => (_root_.VStar.In.order src)[p]!.rep == rep := by
    rw [List.mem_filter, List.mem_range]
    refine ⟨hp, ?_⟩
    rw [getElem!_pos (_root_.VStar.In.order src) p hp, hpo, hr]
    exact beq_self_eq_true _
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 (List.isSome_min?_of_mem hin)
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (List.isSome_max?_of_mem hin)
  refine ⟨a, b + 1, ?_⟩
  unfold _root_.VStar.In.repEnds
  simp only [ha, hb] <;> rfl

/-- **A loop each of whose steps yields never throws.** -/
theorem forIn_ne_error {α β ε : Type} {f : α → β → Except ε (ForInStep β)} {e : ε} :
    ∀ (l : List α) (b : β), (∀ a ∈ l, ∀ b, ∃ b', f a b = .ok (.yield b')) → forIn l b f = .error e → False
  | [], b, _, h => by simp [List.forIn_nil, pure, Except.pure] at h
  | a :: l, b, hf, h => by
    obtain ⟨b', hb⟩ := hf a List.mem_cons_self b
    rw [List.forIn_cons, hb] at h
    simp only [bind, Except.bind] at h
    exact forIn_ne_error l b' (fun a' ha' => hf a' (List.mem_cons_of_mem _ ha')) h

/-- **What a level's builder checks** (`levelIn`): the level's openings number `n`, each of `LANES` lanes and `H`
siblings; the level's tops are read; each opening's top node is among them. -/
def LevelInFits (src : _root_.VStar.In.Src) (level LANES H n : ℕ) : Prop :=
  (src.openings.filter (·.level == level)).size = n ∧
    (∀ q ∈ src.openings, q.level = level → q.row.size = LANES ∧ q.sibs.size = H) ∧
    ∃ T, _root_.VStar.In.topsAt src level = .ok T ∧
      ∀ q ∈ src.openings, q.level = level → T.first[q.rep]! + q.pos / 2 ^ H < T.rows.size

/-- **What an algebra part's builder checks** (`algebraIn`): each rep has an opening, and each port is `acc0`, `acc1`
or an `m<i>` whose message round holds two rows. -/
def AlgInFits (src : _root_.VStar.In.Src) (ports : List (String × ℕ)) : Prop :=
  (∃ o ∈ src.openings, o.rep = 0) ∧ (∃ o ∈ src.openings, o.rep = 1) ∧
    ∀ nm ∈ ports.map Prod.fst, nm = "acc0" ∨ nm = "acc1" ∨
      ∃ i rs, RecVStar.mIndex nm = some i ∧ src.msgs[i]? = some rs ∧ rs.size = 2

/-- **What the stager checks of its inputs at statement `st`**: a level's, an algebra part's or a cap's; it has no
builder for any other statement. -/
def StmtFits (src : _root_.VStar.In.Src) : _root_.VStar.Stmt → Prop
  | .recOpen level LANES H n => LevelInFits src level LANES H n
  | .inner _ _ p => AlgInFits src p.S.ports
  | .cap level N => ∃ pin, _root_.VStar.In.capIn src level N = .ok pin
  | _ => False

/-- **A level's builder accepts** where its checks hold. -/
theorem levelIn_ok {src : _root_.VStar.In.Src} {prog : String} {level LANES H n : ℕ}
    (hF : LevelInFits src level LANES H n) : ∃ pin, _root_.VStar.In.levelIn src prog level LANES H n = .ok pin := by
  obtain ⟨hn, hsz, T, hT, htp⟩ := hF
  have hq : ∀ q, q ∈ (src.openings.filter (·.level == level)).qsort _root_.VStar.In.before ↔
      q ∈ src.openings ∧ q.level = level := by
    intro q
    simp only [qsort_mem _ _ before_asym before_trans, Array.mem_filter, beq_iff_eq]
  have h1 : (((src.openings.filter (·.level == level)).qsort _root_.VStar.In.before).size == n) = true := by
    rw [FlockC3.RegLevels.TopHold.qsort_size, hn]
    exact beq_self_eq_true n
  have h2 : ((src.openings.filter (·.level == level)).qsort _root_.VStar.In.before).all
      (fun q => q.row.size == LANES && q.sibs.size == H) = true := by
    rw [Array.all_eq_true']
    intro q hqm
    obtain ⟨hm, hl⟩ := (hq q).1 hqm
    obtain ⟨hr, hs⟩ := hsz q hm hl
    simp [hr, hs]
  obtain ⟨ps, h3⟩ : ∃ ps, ((src.openings.filter (·.level == level)).qsort _root_.VStar.In.before).mapM
      (_root_.VStar.In.at? src) = some ps := mapM_some_of fun q hqm => at?_some ((hq q).1 hqm).1
  have h5 : (((src.openings.filter (·.level == level)).qsort _root_.VStar.In.before).map
      fun q => T.first[q.rep]! + q.pos / 2 ^ H).all (· < T.rows.size) = true := by
    rw [Array.all_eq_true']
    intro t ht
    obtain ⟨q, hqm, rfl⟩ := Array.mem_map.1 ht
    obtain ⟨hm, hl⟩ := (hq q).1 hqm
    exact decide_eq_true (htp q hm hl)
  cases hL : _root_.VStar.In.levelIn src prog level LANES H n with
  | ok pin => exact ⟨pin, rfl⟩
  | error e =>
    exfalso
    unfold _root_.VStar.In.levelIn at hL
    simp only [h1, h2, h3, hT, h5, Bool.not_true, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure,
      Except.pure, reduceCtorEq] at hL

theorem algebraIn_ne_error {src : _root_.VStar.In.Src} {prog : String} {ports : List (String × ℕ)} {v : List ℕ}
    {e : String} (hF : AlgInFits src ports) (hA : _root_.VStar.In.algebraIn src prog ports v = .error e) : False := by
  obtain ⟨h0, h1, hp⟩ := hF
  obtain ⟨a0, a1, ha⟩ := repEnds_ok h0
  obtain ⟨b0, b1, hb⟩ := repEnds_ok h1
  unfold _root_.VStar.In.algebraIn at hA
  simp only [bind, Except.bind] at hA
  split at hA
  · rename_i e' he
    rw [ha] at he
    cases he
  · split at hA
    · rename_i e' he
      rw [hb] at he
      cases he
    · split at hA
      · rename_i e' he
        refine forIn_ne_error _ _ ?_ he
        intro a hmem b
        obtain ⟨nm, k⟩ := a
        have hnm := hp nm (List.mem_map_of_mem hmem)
        dsimp only
        by_cases e0 : (nm == "acc0") = true
        · simp only [e0, ↓reduceIte, bind, Except.bind, pure, Except.pure]
          exact ⟨_, rfl⟩
        · by_cases e1 : (nm == "acc1") = true
          · simp only [e0, e1, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure, Except.pure]
            exact ⟨_, rfl⟩
          · rcases hnm with rfl | rfl | ⟨i, rs, hi, hr, hs⟩
            · simp at e0
            · simp at e1
            · have e2 : (rs.size == 2) = true := beq_iff_eq.2 hs
              simp only [e0, e1, e2, Bool.false_eq_true, Bool.not_true, ↓reduceIte, hi, hr, bind, Except.bind, pure,
                Except.pure]
              exact ⟨_, rfl⟩
      · simp [pure, Except.pure] at hA

/-- **An algebra part's builder accepts** where its checks hold. -/
theorem algebraIn_ok {src : _root_.VStar.In.Src} {prog : String} {ports : List (String × ℕ)} {v : List ℕ}
    (hF : AlgInFits src ports) : ∃ pin, _root_.VStar.In.algebraIn src prog ports v = .ok pin := by
  cases hA : _root_.VStar.In.algebraIn src prog ports v with
  | ok pin => exact ⟨pin, rfl⟩
  | error e => exact (algebraIn_ne_error hF hA).elim

/-- **The stager accepts a part** wherever its checks at the part's statement hold (`StmtFits`). -/
theorem partIn_ok {src : _root_.VStar.In.Src} {prog : String} {st : _root_.VStar.Stmt} (h : StmtFits src st) :
    ∃ pin, _root_.VStar.In.partIn src prog st = .ok pin := by
  cases st with
  | recOpen level LANES H n => exact levelIn_ok h
  | inner j name p => exact algebraIn_ok h
  | cap level N => exact h
  | recOpenStacked _ _ _ _ => exact False.elim h
  | sparse _ _ => exact False.elim h
  | opening _ _ _ _ _ _ => exact False.elim h

/-- **The honest source's message rounds** (`msgsOf`): each of the schedule's message rounds holds two rows. -/
theorem msgsOf_getElem? (rs : Array RecVStar.SchedRound) (rr : ℕ → List _root_.VStar.In.Row) {i : ℕ}
    (hi : i < (RecVStar.commitRounds rs).length) : ∃ a, (msgsOf rs rr)[i]? = some a ∧ a.size = 2 := by
  have e : (msgsOf rs rr)[i]? = some (((List.range 2).map fun r =>
      ((msgRound rs r i).bind fun j => (rr (j + 1)).getLast?).getD ⟨#[], ByteArray.empty⟩).toArray) := by
    simp [msgsOf, List.getElem?_range hi]
  exact ⟨_, e, by simp⟩

theorem srcOf_eq {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {r : XR}
    {rs : Array RecVStar.SchedRound} (hr : readX x = some r) (hrs : repSched r.shape = some rs)
    {src : _root_.VStar.In.Src} (hsrc : srcOf I x w ρ hist = some src) :
    src = { I.rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows I w ρ hist), tops := topsOf rs (roundRows I w ρ hist) } := by
  have e : srcOf I x w ρ hist = some { I.rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows I w ρ hist), tops := topsOf rs (roundRows I w ρ hist) } := by
    simp [srcOf, hr, hrs]
  rw [e] at hsrc
  exact (Option.some.inj hsrc).symm

/-- **An algebra part fits at the honest source** where each rep has an opening in the inner prover's `rest` and each
port is `acc0`, `acc1` or an `m<i>` at one of the schedule's message rounds. -/
theorem algFits_srcOf {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)}
    {r : XR} {rs : Array RecVStar.SchedRound} (hr : readX x = some r) (hrs : repSched r.shape = some rs)
    {src : _root_.VStar.In.Src} (hsrc : srcOf I x w ρ hist = some src) {ports : List (String × ℕ)}
    (h0 : ∃ o ∈ (I.rest w ρ.1 hist).openings, o.rep = 0) (h1 : ∃ o ∈ (I.rest w ρ.1 hist).openings, o.rep = 1)
    (hp : ∀ nm ∈ ports.map Prod.fst, nm = "acc0" ∨ nm = "acc1" ∨
      ∃ i, RecVStar.mIndex nm = some i ∧ i < (RecVStar.commitRounds rs).length) :
    AlgInFits src ports := by
  have hs := srcOf_eq hr hrs hsrc
  subst hs
  refine ⟨h0, h1, fun nm hnm => ?_⟩
  rcases hp nm hnm with e | e | ⟨i, hi, hlt⟩
  · exact .inl e
  · exact .inr (.inl e)
  · obtain ⟨a, ha, hsz⟩ := msgsOf_getElem? rs (roundRows I w ρ hist) hlt
    exact .inr (.inr ⟨i, a, hi, ha, hsz⟩)

/-- **The stager accepts every part** at the history `hist` wherever its checks hold at each part's statement. -/
theorem hok_of_fits {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {s : ℕ}
    (h : ∀ st c src, partAt x s = some (st, c) → srcOf I x w ρ hist = some src → StmtFits src st) :
    ∀ st c src, partAt x s = some (st, c) → srcOf I x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) st = .ok pin :=
  fun st c src hp hs => partIn_ok (h st c src hp hs)

end RecDeploy.VStarBuild
