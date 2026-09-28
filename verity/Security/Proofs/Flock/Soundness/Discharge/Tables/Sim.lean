import Proofs.Flock.Soundness.Discharge.Tables.Steps

/-!
# The simulation of a session of `J` tables (the transfer, R11c, at `--session-tables J`)

flock-circuit's `--session-tables J` (`session_tables`, `config_with`) and flock-verify's `Stmt.setupTables` run one
session of `J` tables: one `Commit` with every table's root, one link exchange whose two points every table reads (the
tables share `m_pts`), the `2J` rep streams `(table j, rep r)` in any order the prover picks, each round's coins drawn
after its bytes, and every proof file after the last coin of every stream (the barrier).

* `liveSession dg J mPts N`: that live session, on the challenger with retained round bytes. `liveSession` at `J = 1`
  is `Refine.liveTable` up to the submissions' table index.
* `modelSession out G`: the model's session of `J` tables, `Commit`'s caps, the link points, then `batch` over the
  tables, each its two reps concurrently (`G j pts r`) and its openings. `sessionBL` is one (`Session.lean`).
* `DecodesTable … j`: table `j`'s decoder is right on accepted sessions: on every live outcome the live verdict
  accepts, table `j`'s reps run on the tapes decoded from its streams `(j, 0)`, `(j, 1)` and its proof files, and the
  model accepts table `j`'s output on those runs. One hypothesis per table.
* `simSession`, `live_le_session`: the simulation, and the transfer: against any live prover, the live verdict holds
  with probability at most the model session's acceptance (every table's) against the simulated prover, whenever
  every table's decoder is right.
-/

namespace FlockSoundness.Discharge.Tables

open Flock FlockLevel3 Game Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## The live session -/

/-- A doubly indexed family with the entry at `(j, r)` replaced. -/
def upd {ζ : Sort*} {J : ℕ} (f : Fin J → Fin 2 → ζ) (j : Fin J) (r : Fin 2) (z : ζ) : Fin J → Fin 2 → ζ :=
  fun j' r' => if j' = j ∧ r' = r then z else f j' r'

/-- A round the prover submits in a session of `J` tables (PROTOCOL.md §7.1): its stream (table `tab`, rep `rep`), the
framed bytes, and the coin count. -/
structure SubmitJ (J : ℕ) where
  tab : Fin J
  rep : Fin 2
  bytes : ByteArray
  n : ℕ

/-- Append a round to stream `(j, r)`. -/
def addRoundJ {J : ℕ} (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (rd : Round) :
    Fin J → Fin 2 → List Round :=
  upd rs j r (rs j r ++ [rd])

theorem upd_self {ζ : Sort*} {J : ℕ} (f : Fin J → Fin 2 → ζ) (j : Fin J) (r : Fin 2) (z : ζ) : upd f j r z j r = z :=
  ite_eq_left ⟨rfl, rfl⟩

theorem upd_ne {ζ : Sort*} {J : ℕ} (f : Fin J → Fin 2 → ζ) {j j' : Fin J} {r r' : Fin 2} (z : ζ)
    (h : ¬ (j' = j ∧ r' = r)) : upd f j r z j' r' = f j' r' :=
  ite_eq_right h

theorem addRoundJ_self {J : ℕ} (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (rd : Round) :
    addRoundJ rs j r rd j r = rs j r ++ [rd] :=
  upd_self _ _ _ _

theorem addRoundJ_ne {J : ℕ} (rs : Fin J → Fin 2 → List Round) {j j' : Fin J} {r r' : Fin 2} (rd : Round)
    (h : ¬ (j' = j ∧ r' = r)) : addRoundJ rs j r rd j' r' = rs j' r' :=
  upd_ne _ _ h

/-- **The challenger's rounds over `2J` streams** (PROTOCOL.md §7.1): the prover submits a round on any table's either
rep, or stops, at most `N` times; the server draws the round's `n` coins uniformly once its bytes are fixed and appends
the round to that stream. -/
def liveRoundsJ (dg : ByteArray → ByteArray) (J : ℕ) :
    ℕ → (Fin J → Fin 2 → List Round) → Game (Fin J → Fin 2 → List Round)
  | 0, rs => .ret rs
  | N + 1, rs => .send (Option (SubmitJ J)) fun
    | none => .ret rs
    | some s => .coin (Fin s.n → 𝔽) fun c => liveRoundsJ dg J N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))

/-- A live session's outcome: the roots `Commit` sent, the link points, every stream's rounds, and the proof files. -/
structure LiveOutJ (J mPts : ℕ) where
  rootB : Array ByteArray
  pts : Fin 2 → Fin mPts → 𝔽
  reps : Fin J → Fin 2 → List Round
  proofs : Array ByteArray

/-- The live session's continuation after the rounds: the proof files, then the outcome. -/
def KLJ {J mPts : ℕ} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (rs : Fin J → Fin 2 → List Round) :
    Game (LiveOutJ J mPts) :=
  .send (Array ByteArray) fun proofs => .ret ⟨rootB, pts, rs, proofs⟩

/-- **A live session of `J` tables** (PROTOCOL.md §5, §7, §16.12): `Commit` sends every table's root, the server
draws the two link points every table reads, the prover runs its rounds on the `2J` rep streams, then sends the proof
files, after the last coin. -/
def liveSession (dg : ByteArray → ByteArray) (J mPts N : ℕ) : Game (LiveOutJ J mPts) :=
  .send (Array ByteArray) fun rootB =>
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
  (liveRoundsJ dg J N fun _ _ => []).bind (KLJ rootB pts)

/-! ## The model's session -/

section Model

variable {P OutT Cp Op : Type} {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)

/-- Table `j`'s continuation after its reps: the openings, then its output. -/
def KgJ (j : Fin J) (cap : Cp) (pend : P × P) : Game OutT := .send Op fun o => .ret (out j cap pend o)

/-- Table `j` at its reps' positions `gs j 0`, `gs j 1`. -/
def tabG (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (j : Fin J) : Game OutT :=
  (interleave (gs j 0) (gs j 1)).bind (KgJ out j (caps j))

/-- The tables concurrently, at their reps' positions. -/
def posOf (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) : Game (List OutT) :=
  batch (List.ofFn (tabG out caps gs))

/-- **The model's session of `J` tables**: `Commit`'s caps, one link-point draw, then every table's two reps
concurrently and its openings, the tables concurrently. -/
def modelSession (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) : Game (List OutT) :=
  .send (Fin J → Cp) fun caps => .coin (Fin 2 → Fin mPts → 𝔽) fun pts => posOf out caps fun j => G j pts

/-- The model session accepts when every table's output does (`dflt` past the list). -/
def AllOk (ok : Fin J → OutT → Prop) (dflt : OutT) (outs : List OutT) : Prop :=
  ∀ j : Fin J, ok j (outs.getD j dflt)

end Model

/-! ## Tapes -/

section Tapes

variable {J mPts : ℕ} (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg)

/-- Stream `(j, r)`'s whole tape in a live outcome: its rounds, then its final messages, decoded from the proof files. -/
def repTapeJ (o : LiveOutJ J mPts) (j : Fin J) (r : Fin 2) : List Step :=
  tapeFrom (dec j) (one j) 0 (o.reps j r) ++ (fin j r o.proofs).map .inl

/-- Stream `(j, r)`'s tape after the rounds `rs` has. -/
def restJ (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (o : LiveOutJ J mPts) : List Step :=
  tapeFrom (dec j) (one j) (rs j r).length ((o.reps j r).drop (rs j r).length) ++ (fin j r o.proofs).map .inl

/-- Stream `(j, r)`'s tape from its next round's coins on. -/
def midJ (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (o : LiveOutJ J mPts) : List Step :=
  .inr (coinMsg (one j (rs j r).length) ((o.reps j r).getD (rs j r).length default)) ::
    (tapeFrom (dec j) (one j) ((rs j r).length + 1) ((o.reps j r).drop ((rs j r).length + 1)) ++
      (fin j r o.proofs).map .inl)

theorem restJ_eq_mid (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (b : ByteArray) (rd : Round)
    (hb : rd.bytes = some b) (l : List Round) (o : LiveOutJ J mPts) (ho : o.reps j r = rs j r ++ rd :: l) :
    restJ dec one fin rs j r o = (dec j (rs j r).length b).map .inl ++ midJ dec one fin rs j r o := by
  simp only [restJ, midJ, ho, List.drop_left', List.getD_eq_getElem?_getD, List.getElem?_append_right le_rfl,
    Nat.sub_self, List.getElem?_cons_zero, Option.getD_some, tapeFrom, hb, List.append_assoc, List.cons_append]
  rw [show (rs j r ++ rd :: l).drop ((rs j r).length + 1) = l by rw [List.drop_append]; simp]

theorem midJ_eq_rest (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (rd : Round) (l : List Round)
    (o : LiveOutJ J mPts) (ho : o.reps j r = rs j r ++ rd :: l) :
    midJ dec one fin rs j r o =
      .inr (coinMsg (one j (rs j r).length) rd) :: restJ dec one fin (addRoundJ rs j r rd) j r o := by
  simp only [midJ, restJ, addRoundJ_self, List.length_append, List.length_singleton, ho,
    List.getD_eq_getElem?_getD, List.getElem?_append_right le_rfl, Nat.sub_self, List.getElem?_cons_zero,
    Option.getD_some]

theorem restJ_other (rs : Fin J → Fin 2 → List Round) {j j' : Fin J} {r r' : Fin 2} (h : ¬ (j' = j ∧ r' = r))
    (rd : Round) (o : LiveOutJ J mPts) :
    restJ dec one fin (addRoundJ rs j r rd) j' r' o = restJ dec one fin rs j' r' o := by
  simp only [restJ, addRoundJ_ne rs rd h]

end Tapes

/-! ## Following decoded messages -/

section Follow

variable {α : Type}

open Classical in
/-- The model game after the messages `ms`, when each fits the node it meets. -/
noncomputable def follow : Game α → List Msg → Option (Game α)
  | g, [] => some g
  | g, ⟨M', m'⟩ :: ms => match g with
    | .send M k => if h : M' = M then follow (k (cast h m')) ms else none
    | .ret _ => none
    | @Game.coin _ _ _ _ _ => none

theorem follow_plays : ∀ (g : Game α) (ms : List Msg) {t : List Step} {a : α},
    PlaysT g (ms.map .inl ++ t) a → ∃ g', follow g ms = some g' ∧ PlaysT g' t a
  | g, [], t, a, h => ⟨g, by simp [follow], by simpa using h⟩
  | .ret _, _ :: _, _, _, h => absurd (PlaysT.ret_inv h).1 (by simp)
  | .send M k, ⟨M', m'⟩ :: ms, t, a, h => by
      obtain ⟨m, hm, h'⟩ := PlaysT.send_inv (by simpa using h)
      simp only [Sum.inl.injEq] at hm
      cases hm
      obtain ⟨g', h1, h2⟩ := follow_plays (k m) ms h'
      exact ⟨g', by simp [follow, h1], h2⟩
  | @Game.coin _ _ _ _ _, _ :: _, _, _, h => by
      obtain ⟨x, hx, -⟩ := PlaysT.coin_inv (by simpa using h)
      simp at hx

theorem follow_allInh : ∀ {g : Game α} {ms : List Msg} {g' : Game α}, AllInh g → follow g ms = some g' → AllInh g'
  | _, [], _, hg, h => by simp only [follow, Option.some.injEq] at h; exact h ▸ hg
  | .send M k, ⟨M', m'⟩ :: ms, _, hg, h => by
      by_cases hM : M' = M
      · subst hM
        simp only [follow, dite_true, cast_eq] at h
        exact follow_allInh (hg.send_k _) h
      · simp [follow, hM] at h
  | .ret _, _ :: _, _, _, h => by simp [follow] at h
  | @Game.coin _ _ _ _ _, _ :: _, _, _, h => by simp [follow] at h

open Classical in
/-- The messages `ms` from `g`, as a path of the model's messages. -/
noncomputable def stepsOfFollow : (g : Game α) → (ms : List Msg) → {g' : Game α} → follow g ms = some g' → Steps g g'
  | g, [], _, h => Steps.ofEq (by simp only [follow, Option.some.injEq] at h; exact h)
  | .send M k, ⟨M', m'⟩ :: ms, _, h =>
      if hM : M' = M then .send (cast hM m') (stepsOfFollow _ ms (by simpa [follow, hM] using h))
      else absurd h (by simp [follow, hM])
  | .ret _, _ :: _, _, h => absurd h (by simp [follow])
  | @Game.coin _ _ _ _ _, _ :: _, _, h => absurd h (by simp [follow])

theorem follow_ret {g : Game α} {ms : List Msg} {a : α} (h : PlaysT g (ms.map .inl) a) : follow g ms = some (.ret a) := by
  obtain ⟨g', h1, h2⟩ := follow_plays g ms (t := []) (by simpa using h)
  rw [h1, PlaysT.nil_inv h2]

/-- A run on messages alone, as a path of the model's messages to its end. -/
noncomputable def stepsOfPlays (g : Game α) (ms : List Msg) {a : α} (h : PlaysT g (ms.map .inl) a) : Steps g (.ret a) :=
  stepsOfFollow g ms (follow_ret h)

end Follow

/-! ## The simulation -/

section Sim

variable {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
  (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
  (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)

/-- A rep's output out of the pair of a table's. -/
def sel {P : Type} (p : P × P) (r : Fin 2) : P := if r = 0 then p.1 else p.2

/-- **The decoder is right for table `j`.** On every live outcome the live verdict accepts, table `j`'s reps run on
the tapes decoded from its streams `(j, 0)`, `(j, 1)` and its proof files, and the model's verdict on table `j`'s
output, with the cap `capOf` decodes from the roots and the openings `opn` decodes from the proof files, holds. -/
def DecodesTable (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (j : Fin J) : Prop :=
  ∀ o, pL o → ∃ pend : P × P, PlaysT (G j o.pts 0) (repTapeJ dec one fin o j 0) pend.1 ∧
    PlaysT (G j o.pts 1) (repTapeJ dec one fin o j 1) pend.2 ∧ ok j (out j (capOf o.rootB j) pend (opn o.proofs j))

/-- **The simulation's invariant** at a live subgame `L` and the model reps' positions `gs`: on every accepted live
outcome, the streams' remaining tapes `ts j r o` run the positions to outputs every table of which the model accepts. -/
def InvJ (caps : Fin J → Cp) (L : Game (LiveOutJ J mPts)) (gs : Fin J → Fin 2 → Game P)
    (ts : Fin J → Fin 2 → LiveOutJ J mPts → List Step) : Prop :=
  ∀ o, Out L o → pL o → ∃ pend : Fin J → P × P, (∀ j r, PlaysT (gs j r) (ts j r o) (sel (pend j) r)) ∧
    ∀ j, ok j (out j (caps j) (pend j) (opn o.proofs j))

variable {out ok pL opn}

theorem InvJ.mono {caps : Fin J → Cp} {L L' : Game (LiveOutJ J mPts)} {gs : Fin J → Fin 2 → Game P}
    {ts : Fin J → Fin 2 → LiveOutJ J mPts → List Step} (h : InvJ out ok pL opn caps L gs ts)
    (hL : ∀ o, Out L' o → Out L o) : InvJ out ok pL opn caps L' gs ts :=
  fun o ho hp => h o (hL o ho) hp

variable (out)

theorem allInh_batch {A : Type} : ∀ {l : List (Game A)}, (∀ g ∈ l, AllInh g) → AllInh (batch l)
  | [], _ => .ret _
  | g :: gs, h => (AllInh.interleave (h g (by simp)) (allInh_batch fun g' hg' => h g' (by simp [hg']))).map _

theorem allInh_posOf (caps : Fin J → Cp) {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r)) :
    AllInh (posOf out caps gs) :=
  allInh_batch fun g hg => by
    obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hg
    exact AllInh.bind (fun _ => .send inferInstance fun _ => .ret _) (AllInh.interleave (h j 0) (h j 1))

theorem allInh_upd {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r)) (j : Fin J) (r : Fin 2) {g : Game P}
    (hg : AllInh g) : ∀ j' r', AllInh (upd gs j r g j' r') := by
  intro j' r'
  unfold upd
  split_ifs
  · exact hg
  · exact h j' r'

theorem posOf_upd_self (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (j : Fin J) (r : Fin 2) :
    posOf out caps (upd gs j r (gs j r)) = posOf out caps gs := by
  congr 1
  funext j' r'
  unfold upd
  split_ifs with h
  · rw [h.1, h.2]
  · rfl

theorem posOf_upd₀ (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (j : Fin J) (g : Game P) :
    batch (List.ofFn (Function.update (tabG out caps gs) j ((interleave g (gs j 1)).bind (KgJ out j (caps j))))) =
      posOf out caps (upd gs j 0 g) := by
  unfold posOf
  congr 2
  funext j'
  by_cases h : j' = j
  · subst h
    rw [Function.update_self]
    simp [tabG, upd]
  · rw [Function.update_of_ne h]
    simp [tabG, upd, h]

theorem posOf_upd₁ (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (j : Fin J) (g : Game P) :
    batch (List.ofFn (Function.update (tabG out caps gs) j ((interleave (gs j 0) g).bind (KgJ out j (caps j))))) =
      posOf out caps (upd gs j 1 g) := by
  unfold posOf
  congr 2
  funext j'
  by_cases h : j' = j
  · subst h
    rw [Function.update_self]
    simp [tabG, upd]
  · rw [Function.update_of_ne h]
    simp [tabG, upd, h]

/-- **Rep `r` of table `j` is a context of the session.** -/
def ctxAt (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (j : Fin J) :
    (r : Fin 2) → Ctx fun g => posOf out caps (upd gs j r g)
  | ⟨0, _⟩ => (((Ctx.left (gs j 1)).comp (Ctx.bind (KgJ out j (caps j)))).comp (Ctx.batchFn (tabG out caps gs) j)).congr
      (posOf_upd₀ out caps gs j)
  | ⟨1, _⟩ => (((Ctx.right (gs j 0)).comp (Ctx.bind (KgJ out j (caps j)))).comp (Ctx.batchFn (tabG out caps gs) j)).congr
      (posOf_upd₁ out caps gs j)

theorem sel_pair {P : Type} (p : P × P) : (sel p 0, sel p 1) = p := rfl

/-- Table `j`'s reps run to their ends on their messages, then its openings: a path of the model's messages to table
`j`'s output. -/
noncomputable def stepsTab (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (pend : Fin J → P × P)
    (fs : Fin J → Fin 2 → List Msg) (h : ∀ j r, PlaysT (gs j r) ((fs j r).map .inl) (sel (pend j) r))
    (os : Fin J → Op) (j : Fin J) : Steps (tabG out caps gs j) (.ret (out j (caps j) (pend j) (os j))) :=
  ((Ctx.bind (KgJ out j (caps j))).lift
    ((((Ctx.left (gs j 1)).lift (stepsOfPlays (gs j 0) (fs j 0) (h j 0))).trans
      ((Ctx.right (.ret (sel (pend j) 0))).lift (stepsOfPlays (gs j 1) (fs j 1) (h j 1)))).trans
      (Steps.ofEq (congrArg Game.ret (sel_pair (pend j)))))).trans
    (.send (os j) (.refl _))

/-- The tables' runs to their ends, then their openings: a path of the model's messages to the session's end. -/
noncomputable def stepsEnd (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (pend : Fin J → P × P)
    (fs : Fin J → Fin 2 → List Msg) (h : ∀ j r, PlaysT (gs j r) ((fs j r).map .inl) (sel (pend j) r))
    (os : Fin J → Op) :
    Steps (posOf out caps gs) (.ret (List.ofFn fun j => out j (caps j) (pend j) (os j))) :=
  steps_batch_ofFn _ _ (stepsTab out caps gs pend fs h os)

variable {out}

/-- A live run's streams extend the rounds they had. -/
theorem liveRoundsJ_out : ∀ (n : ℕ) (rs : Fin J → Fin 2 → List Round) {rs' : Fin J → Fin 2 → List Round},
    Out (liveRoundsJ dg J n rs) rs' → ∀ j r, ∃ l, rs' j r = rs j r ++ l
  | 0, _, _, .ret _ => fun _ _ => ⟨[], by simp⟩
  | _ + 1, _, _, .send none (.ret _) => fun _ _ => ⟨[], by simp⟩
  | n + 1, rs, _, .send (some s) (.coin c h) => fun j r => by
      obtain ⟨l, hl⟩ := liveRoundsJ_out n _ h j r
      rw [hl]
      by_cases hjr : j = s.tab ∧ r = s.rep
      · obtain ⟨rfl, rfl⟩ := hjr
        rw [addRoundJ_self]
        exact ⟨roundOf dg s.bytes s.n c :: l, by simp⟩
      · rw [addRoundJ_ne rs _ hjr]
        exact ⟨l, rfl⟩

theorem out_liveJ {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ} {rs : Fin J → Fin 2 → List Round}
    {o : LiveOutJ J mPts} (h : Out ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) o) :
    o.rootB = rootB ∧ o.pts = pts ∧ ∀ j r, ∃ l, o.reps j r = rs j r ++ l := by
  obtain ⟨rs', hrs, ho⟩ := Out.of_bind h
  have hl := liveRoundsJ_out dg n rs hrs
  unfold KLJ at ho
  cases ho with
  | send proofs ho => cases ho; exact ⟨rfl, rfl, hl⟩

/-- The live coin node of a round on stream `(j, r)`. -/
abbrev roundL (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ) (rs : Fin J → Fin 2 → List Round)
    (j : Fin J) (r : Fin 2) (b : ByteArray) (k : ℕ) (c : Fin k → 𝔽) : Game (LiveOutJ J mPts) :=
  (liveRoundsJ dg J n (addRoundJ rs j r (roundOf dg b k c))).bind (KLJ rootB pts)

theorem reps_coin {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ} {rs : Fin J → Fin 2 → List Round}
    {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ} {o : LiveOutJ J mPts}
    (ho : Out (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) o) :
    ∃ c l, o.reps j r = rs j r ++ roundOf dg b k c :: l := by
  cases ho with
  | coin c ho =>
    obtain ⟨-, -, h⟩ := out_liveJ dg ho
    obtain ⟨l, hl⟩ := h j r
    exact ⟨c, l, by rw [hl, addRoundJ_self]; simp⟩

theorem dead_follow {caps : Fin J → Cp} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ}
    {rs : Fin J → Fin 2 → List Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ} {gs : Fin J → Fin 2 → Game P}
    (hf : follow (gs j r) (dec j (rs j r).length b) = none)
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs)) :
    ∀ o, Out (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) o → ¬ pL o := by
  intro o ho hp
  obtain ⟨c, l, hl⟩ := reps_coin dg ho
  obtain ⟨pend, hpl, -⟩ := hI o ho hp
  have h := hpl j r
  rw [restJ_eq_mid dec one fin rs j r b _ rfl l o hl] at h
  obtain ⟨g', h1, -⟩ := follow_plays _ _ h
  rw [hf] at h1
  cases h1

theorem inv_follow {caps : Fin J → Cp} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ}
    {rs : Fin J → Fin 2 → List Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ} {gs : Fin J → Fin 2 → Game P}
    {g' : Game P} (hf : follow (gs j r) (dec j (rs j r).length b) = some g')
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs)) :
    InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r g')
      (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r)) := by
  intro o ho hp
  obtain ⟨c, l, hl⟩ := reps_coin dg ho
  obtain ⟨pend, hpl, hok⟩ := hI o ho hp
  refine ⟨pend, fun j' r' => ?_, hok⟩
  by_cases hjr : j' = j ∧ r' = r
  · obtain ⟨rfl, rfl⟩ := hjr
    rw [upd_self, upd_self]
    have h := hpl j' r'
    rw [restJ_eq_mid dec one fin rs j' r' b _ rfl l o hl] at h
    obtain ⟨g'', h1, h2⟩ := follow_plays _ _ h
    rw [hf] at h1
    cases h1
    exact h2
  · rw [upd_ne _ _ hjr, upd_ne _ _ hjr]
    exact hpl j' r'

/-- What a round's coins are to the model: one field element, or the vector of them. -/
def coinTy (one : Bool) (k : ℕ) : Type := if one = true ∧ k = 1 then 𝔽 else (Fin k → 𝔽)

theorem coinMsg_fst (b : ByteArray) (k : ℕ) (c : Fin k → 𝔽) (one : Bool) :
    (coinMsg one (roundOf dg b k c)).1 = coinTy one k := by
  by_cases hs : one = true ∧ k = 1
  · obtain ⟨rfl, rfl⟩ := hs
    rw [coinMsg_one]
    unfold coinTy
    rw [ite_eq_left ⟨rfl, rfl⟩]
  · rw [coinMsg_vec dg b k c one hs]
    unfold coinTy
    rw [ite_eq_right hs]

theorem dead_mid {caps : Fin J → Cp} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ}
    {rs : Fin J → Fin 2 → List Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ} {gs : Fin J → Fin 2 → Game P}
    {g' : Game P}
    (hg : ∀ (Cm : Type) (iF : Fintype Cm) (iN : Nonempty Cm) (kk : Cm → Game P),
      g' = @Game.coin P Cm iF iN kk → Cm ≠ coinTy (one j (rs j r).length) k)
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r g')
      (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r))) :
    ∀ o, Out (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) o → ¬ pL o := by
  intro o ho hp
  obtain ⟨c, l, hl⟩ := reps_coin dg ho
  obtain ⟨pend, hpl, -⟩ := hI o ho hp
  have h := hpl j r
  rw [upd_self, upd_self, midJ_eq_rest dec one fin rs j r _ l o hl] at h
  cases g' with
  | ret _ => exact absurd (PlaysT.ret_inv h).1 (by simp)
  | send _ _ =>
    obtain ⟨m, hm, -⟩ := PlaysT.send_inv h
    simp at hm
  | coin Cm kk =>
    obtain ⟨x, hx, -⟩ := PlaysT.coin_inv h
    refine hg Cm _ _ kk rfl ?_
    simp only [Sum.inr.injEq] at hx
    exact (congrArg Sigma.fst hx).symm.trans (coinMsg_fst dg b k c _)

theorem inv_coin {caps : Fin J → Cp} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ}
    {rs : Fin J → Fin 2 → List Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ} {gs : Fin J → Fin 2 → Game P}
    {Cm : Type} {iF : Fintype Cm} {iN : Nonempty Cm} {kk : Cm → Game P} (e : Cm ≃ (Fin k → 𝔽))
    (he : (∃ (hs : one j (rs j r).length = true ∧ k = 1) (hC : Cm = 𝔽), e = eOne hC hs.2) ∨
      (¬ (one j (rs j r).length = true ∧ k = 1) ∧ ∃ hC : Cm = (Fin k → 𝔽), e = Equiv.cast hC))
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k))
      (upd gs j r (@Game.coin P Cm iF iN kk)) (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r)))
    (x : Cm) :
    InvJ out ok pL opn caps (roundL dg rootB pts n rs j r b k (e x)) (upd gs j r (kk x))
      (restJ dec one fin (addRoundJ rs j r (roundOf dg b k (e x)))) := by
  intro o ho hp
  obtain ⟨pend, hpl, hok⟩ := hI o (.coin (e x) ho) hp
  obtain ⟨-, -, hl⟩ := out_liveJ dg ho
  refine ⟨pend, fun j' r' => ?_, hok⟩
  by_cases hjr : j' = j ∧ r' = r
  · obtain ⟨rfl, rfl⟩ := hjr
    obtain ⟨l, hl'⟩ := hl j' r'
    rw [addRoundJ_self, List.append_assoc, List.singleton_append] at hl'
    have h := hpl j' r'
    rw [upd_self, upd_self, midJ_eq_rest dec one fin rs j' r' _ l o hl', coinMsg_e dg b k _ x e he] at h
    rw [upd_self]
    exact h.coin_val
  · have h := hpl j' r'
    rw [upd_ne _ _ hjr, upd_ne _ _ hjr] at h
    rw [upd_ne _ _ hjr, restJ_other dec one fin rs hjr]
    exact h

theorem inv_end {caps : Fin J → Cp} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽}
    {rs : Fin J → Fin 2 → List Round} {gs : Fin J → Fin 2 → Game P}
    (hI : InvJ out ok pL opn caps (KLJ rootB pts rs) gs (restJ dec one fin rs)) (proofs : Array ByteArray)
    (hp : pL ⟨rootB, pts, rs, proofs⟩) :
    ∃ pend : Fin J → P × P, (∀ j r, PlaysT (gs j r) ((fin j r proofs).map .inl) (sel (pend j) r)) ∧
      ∀ j, ok j (out j (caps j) (pend j) (opn proofs j)) := by
  obtain ⟨pend, hpl, hok⟩ := hI ⟨rootB, pts, rs, proofs⟩ (.send proofs (.ret _)) hp
  refine ⟨pend, fun j r => ?_, hok⟩
  simpa [restJ, tapeFrom] using hpl j r

theorem allOk_ofFn {v : Fin J → OutT} (h : ∀ j, ok j (v j)) : AllOk ok dflt (List.ofFn v) := fun j => by
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  simpa using h j

theorem inv_init {G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P}
    (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) (N : ℕ) :
    InvJ out ok pL opn (capOf rootB) ((liveRoundsJ dg J N fun _ _ => []).bind (KLJ rootB pts)) (fun j => G j pts)
      (restJ dec one fin fun _ _ => []) := by
  intro o ho hp
  obtain ⟨hr, hpts, -⟩ := out_liveJ dg ho
  choose pend hp0 hp1 hok using fun j => hdec j o hp
  subst hr hpts
  refine ⟨pend, fun j r => ?_, hok⟩
  rcases Fin.exists_fin_two.1 ⟨r, rfl⟩ with rfl | rfl
  · simpa [restJ, repTapeJ, sel] using hp0 j
  · simpa [restJ, repTapeJ, sel] using hp1 j

variable (out ok pL opn)

open Classical in
/-- The live rounds are over: on the proof files, if the live verdict holds, the model runs every table's reps to
their ends on their final messages and sends every table's openings; otherwise no live run is accepted. -/
noncomputable def simEndJ (caps : Fin J → Cp) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Round) (gs : Fin J → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn caps (KLJ rootB pts rs) gs (restJ dec one fin rs)) :
    Sim pL (AllOk ok dflt) (KLJ rootB pts rs) (posOf out caps gs) :=
  .live fun proofs =>
    if hp : pL ⟨rootB, pts, rs, proofs⟩ then
      Sim.ofSteps (stepsEnd out caps gs (Classical.choose (inv_end dec one fin hI proofs hp)) (fun j r => fin j r proofs)
          (Classical.choose_spec (inv_end dec one fin hI proofs hp)).1 (opn proofs))
        (.ret fun _ => allOk_ofFn dflt (Classical.choose_spec (inv_end dec one fin hI proofs hp)).2)
    else Sim.dead (prob_eq_zero_of_out fun o ho hp' => by cases ho; exact hp hp')
      (Classical.choice (allInh_posOf out caps h).strategy)

/-- A model game's simulation, carried along an equality of model games. -/
def Sim.castR {β α : Type} {pL : β → Prop} {pg : α → Prop} {L : Game β} {g g' : Game α} (h : g = g')
    (d : Sim pL pg L g) : Sim pL pg L g' :=
  h ▸ d

open Classical in
/-- **A round's coins** at the model's position `g'` of rep `r` of table `j`, after the round's messages: matched to
the live coins when `g'` is a coin of the type the round decodes to; past any other node no accepted run goes on. -/
noncomputable def coinJ (caps : Fin J → Cp) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ)
    (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (b : ByteArray) (k : ℕ) (gs : Fin J → Fin 2 → Game P)
    (h : ∀ j r, AllInh (gs j r))
    (next : (c : Fin k → 𝔽) → (gs' : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn caps (roundL dg rootB pts n rs j r b k c) gs' (restJ dec one fin (addRoundJ rs j r (roundOf dg b k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs j r b k c) (posOf out caps gs')) :
    (g' : Game P) → AllInh g' →
    InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r g')
      (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r)) →
    Sim pL (AllOk ok dflt) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (posOf out caps (upd gs j r g'))
  | @Game.coin _ Cm iF iN kk, hg, hI =>
      let cs := (ctxAt out caps gs j r).coin kk
      let go (e : Cm ≃ (Fin k → 𝔽))
          (he : (∃ (hs : one j (rs j r).length = true ∧ k = 1) (hC : Cm = 𝔽), e = eOne hC hs.2) ∨
            (¬ (one j (rs j r).length = true ∧ k = 1) ∧ ∃ hC : Cm = (Fin k → 𝔽), e = Equiv.cast hC)) :
          Sim pL (AllOk ok dflt) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k))
            (posOf out caps (upd gs j r (@Game.coin _ Cm iF iN kk))) :=
        Sim.ofSteps cs.steps (Sim.coin e fun x => Sim.castR (cs.eq x).symm
          (next (e x) (upd gs j r (kk x)) (allInh_upd h j r (hg.coin_k x)) (inv_coin dg dec one fin e he hI x)))
      let dead (hC : Cm ≠ coinTy (one j (rs j r).length) k) :
          Sim pL (AllOk ok dflt) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k))
            (posOf out caps (upd gs j r (@Game.coin _ Cm iF iN kk))) :=
        Sim.dead (prob_eq_zero_of_out (dead_mid dg dec one fin (fun Cm' iF' iN' kk' hk => by
            cases hk
            exact hC) hI))
          (Classical.choice (allInh_posOf out caps (allInh_upd h j r hg)).strategy)
      if hs : one j (rs j r).length = true ∧ k = 1 then
        if hC : Cm = 𝔽 then go (eOne hC hs.2) (.inl ⟨hs, hC, rfl⟩)
        else dead (by unfold coinTy; rw [ite_eq_left hs]; exact hC)
      else
        if hC : Cm = (Fin k → 𝔽) then go (Equiv.cast hC) (.inr ⟨hs, hC, rfl⟩)
        else dead (by unfold coinTy; rw [ite_eq_right hs]; exact hC)
  | .ret _, hg, hI => Sim.dead (prob_eq_zero_of_out (dead_mid dg dec one fin (fun _ _ _ _ hk => by cases hk) hI))
      (Classical.choice (allInh_posOf out caps (allInh_upd h j r hg)).strategy)
  | .send _ _, hg, hI => Sim.dead (prob_eq_zero_of_out (dead_mid dg dec one fin (fun _ _ _ _ hk => by cases hk) hI))
      (Classical.choice (allInh_posOf out caps (allInh_upd h j r hg)).strategy)

/-- **A round on stream `(j, r)`**: the model's rep `r` of table `j` takes the messages decoded from the round's
bytes, then draws its coins with the live round's. -/
noncomputable def roundJ (caps : Fin J → Cp) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ)
    (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2) (b : ByteArray) (k : ℕ) (gs : Fin J → Fin 2 → Game P)
    (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs))
    (next : (c : Fin k → 𝔽) → (gs' : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn caps (roundL dg rootB pts n rs j r b k c) gs' (restJ dec one fin (addRoundJ rs j r (roundOf dg b k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs j r b k c) (posOf out caps gs')) :
    Sim pL (AllOk ok dflt) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (posOf out caps gs) :=
  match hf : follow (gs j r) (dec j (rs j r).length b) with
  | none => Sim.dead (prob_eq_zero_of_out (dead_follow dg dec one fin hf hI))
      (Classical.choice (allInh_posOf out caps h).strategy)
  | some g' =>
      Sim.ofSteps ((Steps.ofEq (posOf_upd_self out caps gs j r).symm).trans
          ((ctxAt out caps gs j r).lift (stepsOfFollow _ _ hf)))
        (coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs j r b k gs h next g'
          (follow_allInh (h j r) hf) (inv_follow dg dec one fin hf hI))

/-- **The rounds**: the model follows the live prover round by round, on the stream each round names. -/
noncomputable def simRJ (caps : Fin J → Cp) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    (n : ℕ) → (rs : Fin J → Fin 2 → List Round) → (gs : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs j r)) →
    InvJ out ok pL opn caps ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs) →
    Sim pL (AllOk ok dflt) ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) (posOf out caps gs)
  | 0, rs, gs, h, hI => simEndJ out ok dflt pL opn dec one fin caps rootB pts rs gs h hI
  | n + 1, rs, gs, h, hI => .live fun
    | none => simEndJ out ok dflt pL opn dec one fin caps rootB pts rs gs h (hI.mono fun _ h => .send none h)
    | some ⟨j, r, b, k⟩ => roundJ out ok dflt pL opn dg dec one fin caps rootB pts n rs j r b k gs h
        (hI.mono fun _ h => .send (some (⟨j, r, b, k⟩ : SubmitJ J)) h)
        (fun _ gs' h' hI' => simRJ caps rootB pts n _ gs' h' hI')

/-- **The session's simulation.** The model sends the caps the roots decode to, shares the link points, follows the
rounds, and at the end runs every table's reps on their final messages and sends every table's openings. -/
noncomputable def simSession (N : ℕ) (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (hinh : ∀ j pts r, AllInh (G j pts r)) (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j) :
    Sim pL (AllOk ok dflt) (liveSession dg J mPts N) (modelSession out G) :=
  .live fun rootB => .model (capOf rootB) (.coin (Equiv.refl _) fun pts =>
    simRJ out ok dflt pL opn dg dec one fin (capOf rootB) rootB pts N (fun _ _ => []) (fun j => G j pts)
      (fun j r => hinh j pts r) (inv_init dg dec one fin capOf hdec rootB pts N))

/-- **The transfer, for a session of `J` tables.** Against any live prover, the live verdict holds with probability at
most the model session's acceptance of every table against the simulated prover, whenever every table's decoder is
right on accepted sessions. -/
theorem live_le_session (N : ℕ) (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (hinh : ∀ j pts r, AllInh (G j pts r)) (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j)
    (Pr : Strategy (liveSession dg J mPts N)) :
    prob pL (liveSession dg J mPts N) Pr ≤
      prob (AllOk ok dflt) (modelSession out G)
        ((simSession out ok dflt pL opn dg dec one fin capOf N G hinh hdec).strategy Pr) :=
  Sim.prob_le _ Pr

end Sim

end FlockSoundness.Discharge.Tables
