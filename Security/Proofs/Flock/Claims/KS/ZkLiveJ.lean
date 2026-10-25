import Proofs.Flock.Claims.KS.ZkBoundJ

/-!
# Every masked table's bound from the live sessions' bound at the deployed tables

`ZkKS` at every number of tables. The masked transfer at a verdict on each table's output (`simSessZV`,
`live_le_sessionZV`): a live verdict is no likelier than every table's output passing its own verdict against the
simulated strategy. After `Commit`, table `j` is at position `j` of the batch (`sessAfterZC_split`), so the simulated
strategy after `Commit` at table `j` (`afterJZ`, `simAtJZ`), at the cap the live roots decode to, passes table `j`'s
verdict at least as often as the live verdict holds (`live_le_simAtJZ`, `mean_live_le_simAtJZ`). The masked session's
tables are `T 0, …, T (n - 1)` (`tabsJ`), and the strategy after `Commit` is at `T j` itself.

`ZkLiveAtJ` is `ZkLiveAt` at every table: the escapes at injective fork rounds of every table, and at each tape, table
`j` and table `C` of its word, live strategies with common roots whose verdict bounds the deployed runs accepting with
no column of table `j` off `C`, a verdict on each table's output the decoder is right for, each implying acceptance
and table `j`'s no level-0 opening off `C` unless it self-clashes, and table `j`'s events bounding its `Adv`s and
self-clash. It gives `ZkSessJ` (`zkSessJ_of_liveJ`), so every table's bound (`zk_tableBoundJAt_of_liveJ`).

At the deployed masked record, `T j` is C-Flock ZK's padded table of the statement's table `j` (`zkTabs`, `zkTab` at
table `j`'s layout, link layout and `Zk.schedule`'s schedule), at C-Flock ZK's parameters (`zkTab_m1`) and
committed to table `j`'s model statement (`zkTab_committed`; together, `zk_atTabs`). `ZkLiveJ x` is `ZkLiveAtJ` there
at every layout, and `Flock.KS` at the deployed relation follows from it at every statement some run accepts and
`ExtendJ` (`zk_KS_of_liveJ`), at every number of tables.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable simSession)
open FlockSoundness.Discharge.Composed (outJ repJ sessionZC_eq_modelSession)

local notation "𝔽" => FlockLevel3.GF128

section Sim

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
/-- **The strategy after `Commit` at table `j`** of a strategy of the masked session whose `Commit` sends `caps`, at a
table `T` that is table `j`. -/
noncomputable def afterJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (τ : Strategy (sessionZC A Hs ts)) {caps : Fin ts.length → Model.Cap D} (hc : τ.1 = caps) (j : Fin ts.length)
    {T : TabZK 𝔽 mPts} (hT : ts.get j = T) :
    Strategy (sessAtZ ((gamesZC A Hs ts caps).take j) ((gamesZC A Hs ts caps).drop (j + 1))
      (tabAfterZC A Hs T (caps j))) :=
  cast (by subst hc hT; exact congrArg Strategy (sessAfterZC_split A Hs ts τ.1 j)) τ.2

omit [Fintype K] in
theorem expect_afterJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (τ : Strategy (sessionZC A Hs ts)) {caps : Fin ts.length → Model.Cap D} (hc : τ.1 = caps) (j : Fin ts.length)
    {T : TabZK 𝔽 mPts} (hT : ts.get j = T) (f : List (OutZC 𝔽 D Sl) → ℝ) :
    expect f _ (afterJZ A Hs ts τ hc j hT) = expect f (sessAfterZC (gamesZC A Hs ts τ.1)) τ.2 := by
  subst hc hT
  exact expect_cast (sessAfterZC_split A Hs ts τ.1 j) f τ.2

variable [Inhabited D] [Inhabited Sl]

open Classical in
omit [Fintype K] in
/-- **Every table passing its verdict** has the same probability after `Commit`, read at table `j`. -/
theorem prob_sessionZC_leJ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (τ : Strategy (sessionZC A Hs ts))
    {caps : Fin ts.length → Model.Cap D} (hc : τ.1 = caps) (j : Fin ts.length) {T : TabZK 𝔽 mPts}
    (hT : ts.get j = T) :
    (prob (fun outs => ∀ i : Fin ts.length, ok i (outs.getD i dfltOutZC)) (sessionZC A Hs ts) τ).toReal ≤
      expect (ind fun outs => ∀ i : Fin ts.length, ok i (outs.getD i dfltOutZC)) _ (afterJZ A Hs ts τ hc j hT) := by
  rw [prob_toReal, expect_afterJZ]
  exact le_rfl

/-- **Verdicts that imply acceptance, passed at every table, accept every table up to `j`.** -/
theorem accUpTo_of_okAll {n : ℕ} (ok : Fin n → OutZC 𝔽 D Sl → Prop)
    (hacc : ∀ i o, ok i o → o.accepted = true) {j : ℕ} (hj : j < n) {outs : List (OutZC 𝔽 D Sl)}
    (h : ∀ i : Fin n, ok i (outs.getD i dfltOutZC)) : AccUpTo j outs :=
  fun i hi => hacc _ _ (h ⟨i, by omega⟩)

/-- The simulated strategy of the masked session (`simSessZG`) at a verdict `ok i` on each table `i`'s output, its
dead branches sending empty openings (`nulOpZ`). -/
noncomputable def simSessZV (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) : Strategy (sessionZC A Hs ts) :=
  sessionZC_eq_modelSession A Hs ts ▸
    (Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOpZ (D := D) (Sl := Sl)))
      (simSession (outJ Hs ts) ok dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
        (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec)).strategy Pr

/-- The simulated strategy's `Commit` sends the caps the live prover's roots decode to. -/
theorem simSessZV_fst (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    (simSessZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr).1 = capOf Pr.1 := by
  unfold simSessZV
  exact fst_rec_send (sessionZC_eq_modelSession A Hs ts) _

/-- **The masked transfer at a verdict on each table** (`live_le_sessionZG` table by table): the live verdict is no
likelier than every table's output passing its verdict against the simulated strategy, when every table's decoder is
right for the verdicts. -/
theorem live_le_sessionZV (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (fun outs => ∀ i : Fin ts.length, ok i (outs.getD i dfltOutZC)) (sessionZC A Hs ts)
        (simSessZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessZV
  rw [FlockSoundness.Discharge.Exec.prob_cast]
  exact Claims.Bridge.Sim.prob_le_mapDead _ _ Pr

/-- **The simulated masked strategy after `Commit` at table `j`** of a live prover whose roots are `rootB₀`, at the
caps they decode to, at a table `T` that is table `j`. -/
noncomputable def simAtJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀)
    (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T) :
    Strategy (sessAtZ ((gamesZC A Hs ts (capOf rootB₀)).take j) ((gamesZC A Hs ts (capOf rootB₀)).drop (j + 1))
      (tabAfterZC A Hs T (capOf rootB₀ j))) :=
  afterJZ A Hs ts (simSessZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr) (by rw [simSessZV_fst, hP]) j hT

open Classical in
/-- **A live verdict is no likelier than every table's verdict after `Commit`**, read at table `j`, at the caps the
prover's roots decode to. -/
theorem live_le_simAtJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀)
    (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T) :
    (prob pL (liveSession dg ts.length mPts N) Pr).toReal ≤
      expect (ind fun outs => ∀ i : Fin ts.length, ok i (outs.getD i dfltOutZC)) _
        (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT) :=
  le_trans (ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr))
    (prob_sessionZC_leJ A Hs ts ok _ _ j hT)

open Classical in
/-- **Over a mixture of live strategies with common roots `rootB₀`**: the mean live verdict is at most the mean
probability of every table's verdict after `Commit` at table `j`, over the simulated strategies at the caps the roots
decode to. -/
theorem mean_live_le_simAtJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) {τ : Type} [Fintype τ] (σL : τ → Strategy (liveSession dg ts.length mPts N))
    (hσ : ∀ u, (σL u).1 = rootB₀) (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T) :
    (∑ u, (prob pL (liveSession dg ts.length mPts N) (σL u)).toReal) / Fintype.card τ ≤
      (∑ u, expect (ind fun outs => ∀ i : Fin ts.length, ok i (outs.getD i dfltOutZC)) _
        (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) j hT)) / Fintype.card τ :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    live_le_simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) j hT) (Nat.cast_nonneg _)

end Sim

/-! ## The live sessions' bound at every table -/

/-- **The masked session's tables** `T 0, …, T (n - 1)`. -/
def tabsJ {mPts : ℕ} (n : ℕ) (T : ℕ → TabZK 𝔽 mPts) : List (TabZK 𝔽 mPts) :=
  List.ofFn fun i : Fin n => T i

theorem tabsJ_length {mPts : ℕ} (n : ℕ) (T : ℕ → TabZK 𝔽 mPts) : (tabsJ n T).length = n :=
  List.length_ofFn

theorem tabsJ_get {mPts : ℕ} (n : ℕ) (T : ℕ → TabZK 𝔽 mPts) (j : Fin (tabsJ n T).length) :
    (tabsJ n T).get j = T j :=
  List.get_ofFn (fun i : Fin n => T i) j

/-- Table `j`'s position in the masked session's tables. -/
def finJ {mPts : ℕ} {n : ℕ} (T : ℕ → TabZK 𝔽 mPts) {j : ℕ} (hj : j < n) : Fin (tabsJ n T).length :=
  ⟨j, by rw [tabsJ_length]; exact hj⟩

open Classical in
/-- **The live sessions' bound at the masked tables** `T j` (`ZkLiveAt` at every table): `zkSessJ_of_liveJ`'s
hypothesis on the deployed prover, the masked session's tables `T 0, …` up to the statement's number of tables. Every
prover has escapes at every table `j`'s fork rounds `rd j`, no two one, such that, at each tape, each table `j` and
each table `C` of its word `Ok` allows, live strategies `σL` whose roots agree bound the deployed runs accepting with
no column of table `j` off `C` by their verdict `pL`, the decoder is right for a verdict `ok` on each table's output,
each implying its table accepted and table `j`'s also no level-0 opening off `C` unless it self-clashes, and table `j`'s events bound
the simulated strategies' `Adv`s and self-clash after `Commit` at table `j` (`simAtJZ`). -/
def ZkLiveAtJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) (x : Claims.Msg) {mPts : ℕ}
    (T : ℕ → TabZK 𝔽 mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZ (T j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j))
      (ev : (j : ℕ) → Claims.Flock.IZ (T j).sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
        (Fin ((V true).k x) → Coin) → Prop)
      (ev₁ : ℕ → Ω → (Fin ((V true).k x) → Coin) → Prop)
      (_tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
        ((Claims.run sha512 P (V true) x ω s).vLog ++
          (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).vLog))
      (_tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog),
      ∀ ω (j : ℕ) (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, Ok x j (C p)) →
        ∃ (ok : Fin (tabsJ (tables true x).length T).length → OutZC 𝔽 D Sl → Prop)
          (pL : LiveOutJ (tabsJ (tables true x).length T).length mPts → Prop)
          (opn : Array ByteArray → Fin (tabsJ (tables true x).length T).length → OpensZ 𝔽 D Sl)
          (dg : ByteArray → ByteArray)
          (dec : Fin (tabsJ (tables true x).length T).length → ℕ → ByteArray → List Msg)
          (one : Fin (tabsJ (tables true x).length T).length → ℕ → Bool)
          (fin : Fin (tabsJ (tables true x).length T).length → Fin 2 → Array ByteArray → List Msg)
          (capOf : Array ByteArray → Fin (tabsJ (tables true x).length T).length → Model.Cap D) (N : ℕ)
          (rootB₀ : Array ByteArray)
          (_hok : ∀ o, ok (finJ T hj) o → o.accepted = true ∧
            ¬ (Off₀Z Hs (T j) (capOf rootB₀ (finJ T hj)) (Claims.Flock.oracleOf C) o ∧ ¬ SelfClashZ Hs (T j) o))
          (_hacc : ∀ i o, ok i o → o.accepted = true)
          (hdec : ∀ i, DecodesTable (outJ Hs (tabsJ (tables true x).length T)) ok pL opn dec one fin capOf
            (repJ A Hs (tabsJ (tables true x).length T)) i)
          (τ : Type) (_ : Fintype τ) (_ : Nonempty τ)
          (σL : τ → Strategy (liveSession dg (tabsJ (tables true x).length T).length mPts N))
          (hσ : ∀ u, (σL u).1 = rootB₀),
          (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
              ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
                Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
                  u ≠ C p) ≤
            (∑ u, (prob pL _ (σL u)).toReal) / Fintype.card τ ∧
          (∀ r ℓ, (∑ u, advRSZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj)) r.val (ℓ.val + 1)
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (r, ℓ)) ω s.1 s.2) ∧
          (∀ r : Fin 2, (∑ u, advPadsSZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj)) r.val
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr (.inl r)) ω s.1 s.2) ∧
          (∀ r : Fin 2, (∑ u, advTauSZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj)) r.val
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr (.inr r)) ω s.1 s.2) ∧
          (∑ u, expect (ind fun outs : List (OutZC 𝔽 D Sl) => SelfClashZ Hs (T j) (outs.getD j dfltOutZC)) _
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

/-- **The live sessions' bound gives the masked sessions' after `Commit`** (`ZkSessJ`), the mixture at table `j` the
simulated strategies after `Commit` there (`simAtJZ`) of live strategies whose roots agree. -/
theorem zkSessJ_of_liveJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {A : Model.Arith 𝔽 K} {Hs : Hashes 𝔽 D Sl} {Ok : Claims.Msg → ℕ → List 𝔽 → Prop} {x : Claims.Msg} {mPts : ℕ}
    {T : ℕ → TabZK 𝔽 mPts} (hl : ZkLiveAtJ A Hs Ok x T) : ZkSessJ A Hs Ok x T := by
  intro Ω _ _ P
  obtain ⟨rd, hrd, ev, ev₁, tied, tied₁, hω⟩ := hl Ω P
  refine ⟨rd, hrd, ev, ev₁, tied, tied₁, fun ω j hj C hok => ?_⟩
  obtain ⟨ok, pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hok', hacc', hdec, τ, _, _, σL, hσ, htr, hadv, hP, hTau,
    hsc⟩ := hω ω j hj C hok
  have hlen := take_gamesZC_length A Hs (tabsJ (tables true x).length T) (capOf rootB₀) (finJ T hj)
  refine ⟨τ, inferInstance, inferInstance, _, _, capOf rootB₀ (finJ T hj),
    fun u => simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
      (tabsJ_get _ T _), ?_, hadv, hP, hTau, ?_⟩
  · refine le_trans htr (le_trans (mean_live_le_simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ σL hσ
      (finJ T hj) (tabsJ_get _ T _)) ?_)
    rw [hlen]
    exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
      expect_mono (ind_mono fun outs h =>
        ⟨accUpTo_of_okAll ok hacc' (finJ T hj).isLt h, (hok' _ (h _)).2⟩) _ _) (Nat.cast_nonneg _)
  · rw [hlen]
    exact hsc

/-- **Every masked table's bound at a statement, from the live sessions' bound** at the masked tables `T j`
(`zk_tableBoundJAt_of_sessions` at `zkSessJ_of_liveJ`). -/
theorem zk_tableBoundJAt_of_liveJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
    [Inhabited Sl] (A : Model.Arith 𝔽 K) (hA : A.Correct) (Hs : Hashes 𝔽 D Sl)
    (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) {x : Claims.Msg}
    (hn : Claims.Flock.nTab (deployZk rel) x = (tables true x).length) {spec : SessionSpec} (st : ℕ → Stmt)
    (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j)) (T : ℕ → TabZK 𝔽 spec.mPts)
    (hT : ∀ j < (tables true x).length, 25 ≤ (st j).m ∧ (st j).m ≤ 35 ∧
      (T j).sch = Accounting.m1Schedule (st j).m ∧ (T j).t = Accounting.m1Pad (st j).m ∧ (T j).e = 2 ∧
      (T j).S.shape spec.mPts = acct spec (st j) ∧ (T j).N ≤ 2 ^ 13 ∧ (T j).q = 168 ∧ 7 / 16 ≤ (T j).δ ∧
      (T j).N ≤ Claims.Flock.Numbers.padDomain
        (Zk.PadLayout.new (st j).m (st j).c.kLog (Claims.Flock.zkSched (st j).m).1.k0))
    (hnc : ∀ j < (tables true x).length, ∀ C : Fin (wordLenJ true x j) → List 𝔽, (∀ p, Ok x j (C p)) →
      ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C →
        ¬ (T j).Committed A (Claims.Flock.oracleOf C))
    (hl : ZkLiveAtJ A Hs Ok x T) :
    Claims.Flock.TableBoundJAt sha512 (deployZk rel) (wordLenJ true) Ok x :=
  zk_tableBoundJAt_of_sessions A hA Hs rel Ok hn st hst T hT hnc (zkSessJ_of_liveJ hl)

/-! ## The deployed masked tables -/

/-- **C-Flock ZK's padded tables of a setup** `sts`: at each table `j`, `zkTab` at its layout `L j`, link layout at
`PT_LOCAL` and `Zk.schedule`'s schedule (past the last table, table 0's). -/
noncomputable def zkTabs {sts : Array Stmt} {mPts : ℕ} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (j : ℕ) :
    TabZK 𝔽 mPts :=
  if h : j < sts.size then zkTab (L j h) (hz j h) (hk j h) (hlay j h)
  else zkTab (L 0 h0) (hz 0 h0) (hk 0 h0) (hlay 0 h0)

theorem zkTabs_of_lt {sts : Array Stmt} {mPts : ℕ} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) {j : ℕ}
    (h : j < sts.size) : zkTabs h0 L hz hk hlay j = zkTab (L j h) (hz j h) (hk j h) (hlay j h) :=
  dite_eq_left h

/-- The setup's table `j` (past the last table, table 0). -/
def stJ {sts : Array Stmt} (h0 : 0 < sts.size) (j : ℕ) : Stmt :=
  if h : j < sts.size then sts[j] else sts[0]

/-- **The live sessions' bound at a masked statement of any number of tables**: `ZkLiveAtJ` at the deployed hashes
(`hs512`) and tables of full-width columns, at C-Flock ZK's padded tables of the setup (`zkTabs`), wherever the
verifier sets up the tables `sts`, at their every layout and link layout at `PT_LOCAL`. -/
def ZkLiveJ (x : Claims.Msg) : Prop :=
  ∀ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts),
    ZkLiveAtJ execArith Discharge.ZkExec.hs512 (wideJ true) x (zkTabs h0 L hz hk hlay)

namespace Gen

variable [S : ZkSetup]

/-- **C-Flock ZK's padded tables at a statement in scope**: where the statement parses in scope and sets up at least
one table, each table `j` has C-Flock ZK's `m`, `k_log`, a layout and a link layout (`zk_scopeJ`), and `Zk.schedule`
accepts its `m` (`zk_schedule_ok`), so `zkTabs` there has every property `zk_tableBoundJAt_of_liveJ` asks of the
masked tables: table `j` of the setup is `stJ h0 j`, `zkTab` is at C-Flock ZK's parameters (`zkTab_m1`), and a table
of full-width columns that does not extract table `j` is not committed to it (`zk_not_committedJ`,
`zkTab_committed`). -/
theorem zk_atTabs {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    {B : (ℕ → TabZK 𝔽 spec.mPts) → Prop}
    (hB : ∀ T : ℕ → TabZK 𝔽 spec.mPts,
      Claims.Flock.nTab (deployZk Readable.relation) x = (tables true x).length →
      (∀ j < (tables true x).length, (tables true x)[j]? = some (spec, stJ h0 j)) →
      (∀ j < (tables true x).length, 25 ≤ (stJ h0 j).m ∧ (stJ h0 j).m ≤ 35 ∧
        (T j).sch = Accounting.m1Schedule (stJ h0 j).m ∧ (T j).t = Accounting.m1Pad (stJ h0 j).m ∧ (T j).e = 2 ∧
        (T j).S.shape spec.mPts = acct spec (stJ h0 j) ∧ (T j).N ≤ 2 ^ 13 ∧ (T j).q = 168 ∧ 7 / 16 ≤ (T j).δ ∧
        (T j).N ≤ Claims.Flock.Numbers.padDomain
          (Zk.PadLayout.new (stJ h0 j).m (stJ h0 j).c.kLog (Claims.Flock.zkSched (stJ h0 j).m).1.k0)) →
      (∀ j < (tables true x).length, ∀ C : Fin (wordLenJ true x j) → List 𝔽, (∀ p, wideJ true x j (C p)) →
        ¬ Claims.Flock.GoodTableAt (deployZk Readable.relation) (wordLenJ true) x j C →
          ¬ (T j).Committed execArith (Claims.Flock.oracleOf C)) → B T) :
    ∃ (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
      (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
      (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
      (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts),
      B (zkTabs h0 L hz hk hlay) := by
  have hsc : ∀ j (h : j < sts.size), 25 ≤ sts[j].m ∧ sts[j].m ≤ 35 ∧ sts[j].c.kLog ≤ 27 ∧
      ∃ L : Layout sts[j], (L.model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts :=
    fun j h => zk_scopeJ hs hin hset h
  choose h25 h35 hk L hlay using hsc
  have hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m) :=
    fun j h => zk_schedule_ok (h25 j h) (h35 j h)
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hset]
    simp
  have hn : Claims.Flock.nTab (deployZk Readable.relation) x = (tables true x).length := by
    rw [Claims.Flock.nTab_of_parse (D := deployZk Readable.relation) hs, tables_length hs (by rw [hset]; rfl)]
  refine ⟨L, hz, hk, hlay, hB _ hn (fun j hj => ?_) (fun j hj => ?_) (fun j hj C hw hg hc => ?_)⟩
  · rw [hlen] at hj
    refine tables_getElem? hset ?_
    rw [stJ, dite_eq_left hj]
    exact Array.getElem?_eq_getElem hj
  · rw [hlen] at hj
    rw [zkTabs_of_lt h0 L hz hk hlay hj]
    simp only [stJ, dite_eq_left hj]
    obtain ⟨hsch, hpad, he, hS, hN, hq, hδ, hNp⟩ := zkTab_m1 (L j hj) (hz j hj) (hk j hj) spec (hlay j hj)
    exact ⟨h25 j hj, h35 j hj, hsch, hpad, he, hS, hN, hq, hδ, hNp⟩
  · rw [hlen] at hj
    rw [zkTabs_of_lt h0 L hz hk hlay hj] at hc
    exact zk_not_committedJ hs hin hset hj (L j hj) C hw hg
      ((zkTab_committed (L j hj) (hz j hj) (hk j hj) (hlay j hj) _).1 hc)

end Gen

/-- **C-Flock ZK's padded tables at a statement in scope**: where the statement parses in scope and sets up at least
one table, each table `j` has C-Flock ZK's `m`, `k_log`, a layout and a link layout (`zk_scopeJ`), and `Zk.schedule`
accepts its `m` (`zk_schedule_ok`), so `zkTabs` there has every property `zk_tableBoundJAt_of_liveJ` asks of the
masked tables: table `j` of the setup is `stJ h0 j`, `zkTab` is at C-Flock ZK's parameters (`zkTab_m1`), and a table
of full-width columns that does not extract table `j` is not committed to it (`zk_not_committedJ`,
`zkTab_committed`). -/
theorem zk_atTabs {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    {B : (ℕ → TabZK 𝔽 spec.mPts) → Prop}
    (hB : ∀ T : ℕ → TabZK 𝔽 spec.mPts,
      Claims.Flock.nTab (deployZk Readable.relation) x = (tables true x).length →
      (∀ j < (tables true x).length, (tables true x)[j]? = some (spec, stJ h0 j)) →
      (∀ j < (tables true x).length, 25 ≤ (stJ h0 j).m ∧ (stJ h0 j).m ≤ 35 ∧
        (T j).sch = Accounting.m1Schedule (stJ h0 j).m ∧ (T j).t = Accounting.m1Pad (stJ h0 j).m ∧ (T j).e = 2 ∧
        (T j).S.shape spec.mPts = acct spec (stJ h0 j) ∧ (T j).N ≤ 2 ^ 13 ∧ (T j).q = 168 ∧ 7 / 16 ≤ (T j).δ ∧
        (T j).N ≤ Claims.Flock.Numbers.padDomain
          (Zk.PadLayout.new (stJ h0 j).m (stJ h0 j).c.kLog (Claims.Flock.zkSched (stJ h0 j).m).1.k0)) →
      (∀ j < (tables true x).length, ∀ C : Fin (wordLenJ true x j) → List 𝔽, (∀ p, wideJ true x j (C p)) →
        ¬ Claims.Flock.GoodTableAt (deployZk Readable.relation) (wordLenJ true) x j C →
          ¬ (T j).Committed execArith (Claims.Flock.oracleOf C)) → B T) :
    ∃ (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
      (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
      (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
      (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts),
      B (zkTabs h0 L hz hk hlay) :=
  Gen.zk_atTabs (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (B := B) (hB := hB)

/-- **Every masked table's bound at a statement in scope, from the live sessions' bound**: at tables of full-width
columns, where the statement parses in scope and sets up at least one table, `zk_tableBoundJAt_of_liveJ` at C-Flock
ZK's padded tables (`zk_atTabs`). -/
theorem zk_tableBoundJAt_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (hl : ZkLiveJ x) :
    Claims.Flock.TableBoundJAt sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x := by
  obtain ⟨L, hz, hk, hlay, hb⟩ := zk_atTabs hs hin hset h0
    (B := fun T => ZkLiveAtJ execArith Discharge.ZkExec.hs512 (wideJ true) x T →
      Claims.Flock.TableBoundJAt sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x)
    fun T hn hst hT hnc => zk_tableBoundJAt_of_liveJ execArith execArith_correct Discharge.ZkExec.hs512
      Readable.relation (wideJ true) hn (stJ h0) hst T hT hnc
  exact hb (hl ins spec sts hset h0 L hz hk hlay)

/-- **Every masked table's bound at full-width columns, from the live sessions' bound** at the statements some run
accepts, at every number of tables. -/
theorem zk_tableBoundJ_live (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveJ x) :
    Claims.Flock.TableBoundJ sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJ_of_acc Readable.relation (wideJ true) fun x _ _ _ _ hno hs hin hset h0 =>
    zk_tableBoundJAt_live hs hin hset h0 (hl x hno)

/-- **The deployed masked verifier: KS**, at SHA-512 at the deployed relation, at every number of tables, from the live
sessions' bound at C-Flock ZK's padded tables at the statements some run accepts (`ZkLiveJ`) and `ExtendJ` at
full-width columns. -/
theorem zk_KS_of_liveJ (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveJ x)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_of_tableBoundJ (zk_tableBoundJ_of_extend Readable.relation hext (zk_tableBoundJ_live hl))

end FlockDeploy.KS
