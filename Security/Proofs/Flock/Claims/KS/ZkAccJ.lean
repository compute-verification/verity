import Proofs.Flock.Claims.KS.ZkLiveJ
import Proofs.Flock.Claims.KS.Escapes
import Proofs.Flock.Claims.KS.OnAcc

/-!
# Every masked table's bound with escapes on accepted runs

`Escapes` and `PlainAcc` at the masked record, at every number of tables. An output at the empty masked openings
(`nulOpZ`, which the simulation's dead branches send) never self-clashes (`not_selfClashZ_nul`): its rows are empty, so
none verifies at level 0's width (`width₀_pos`) or a level's `2^lanes`, and its pads pairs are all `(0, 0)`. So at the
masked session, an event no outputs at the empty openings satisfy is no likelier against the simulated strategy than
on accepted live runs whose decoded outputs satisfy it (`sessionZC_le_decoded`, from `Claims.Bridge.prob_le_decoded`),
and table `j`'s self-clash after `Commit` (`simAtJZ`) is no likelier than a decoded self-clash of table `j` on an
accepted live run (`LiveSelfClashZ`, `selfClash_le_liveJZ`, `mean_selfClash_le_liveJZ`).

`TableBoundJAccAt` is `TableBoundJAt` with escapes on accepted runs (`Claims.Escapes.OnAcc`), so it gives
`TableBoundJAt` (`TableBoundJAccAt.tableBoundJAt`), and where no run accepts it holds with none
(`tableBoundJAccAt_of_noAcc`); the old masked escapes joined by round are on accepted runs where every table's are
(`joinEscZ_onAcc`). `ZkLiveAccAtJ` is `ZkLiveAtJ` with every table's events on accepted deployed runs and the
self-clash read on accepted live runs; it gives `ZkLiveAtJ` (`zkLiveAtJ_of_acc`) and every table's bound with escapes on
accepted runs (`zk_tableBoundJAccAt_of_liveJ`, through `zk_le_sessionsJ`). At the deployed masked record that is
`zk_tableBoundJAccAt_live` (at `zk_atTabs`), and at every statement, from `ZkLiveAccJ` where some run accepts,
`zk_tableBoundJAccAt`. `zk_KS_of_liveAccJ` is `zk_KS_of_liveJ` from it.
-/

namespace Claims.Flock

/-! ## Escapes on accepted runs, at every table -/

/-- **The old masked escapes joined by round are on accepted runs** where every table's events are. -/
theorem joinEscZ_onAcc {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZ (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZ (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog)
    (hacc : ∀ j i ω s s', ev j i ω s s' →
      (run H P V x ω s).out = .acc ∧ (run H P V x ω (forkCoins (rd j i) s s')).out = .acc)
    (hacc₁ : ∀ j ω s, ev₁ j ω s → (run H P V x ω s).out = .acc) :
    (joinEscZ H V P x n sch Np rd ev ev₁ tied tied₁).OnAcc := by
  refine ⟨?_, ?_⟩
  · rintro r ω s s' ⟨j, -, i, rfl, h⟩
    exact hacc j i ω s s' h
  · rintro - ω s ⟨j, -, h⟩
    exact hacc₁ j ω s h

section At

variable {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`TableBoundJAt` with escapes on accepted runs.** -/
def TableBoundJAccAt (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop)
    (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.OnAcc ∧
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (j : ℕ), j < nTab D x → ∀ C : Fin (N x j) → Col,
      (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p

theorem TableBoundJAccAt.tableBoundJAt {H : Msg → Msg} {D : Deploy Coin Col T} {N : Msg → ℕ → ℕ}
    {Ok : Msg → ℕ → Col → Prop} {x : Msg} (h : TableBoundJAccAt H D N Ok x) : TableBoundJAt H D N Ok x := by
  intro Ω _ _ P
  obtain ⟨e, -, hf, hb⟩ := h Ω P
  exact ⟨e, hf, hb⟩

/-- **Where the verifier accepts no run**, `TableBoundJAccAt` holds with no escapes, when the weight covers `k` and
`ε_table ≥ 0`. -/
theorem tableBoundJAccAt_of_noAcc (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (Ok : Msg → ℕ → Col → Prop) (x : Msg) (hno : NoAcc H D.V x) (hk : (D.k x : ℝ) ≤ 2 * D.weight x)
    (hε : 0 ≤ D.εtable x) : TableBoundJAccAt H D N Ok x := by
  intro Ω _ _ P
  refine ⟨Escapes.nil H D.V P x, Escapes.onAcc_nil H D.V P x, Escapes.fitsFork_nil H D.V P x _ _ hk,
    fun ω j _ C _ _ => ?_⟩
  have h0 : accPr H D.V P x ω = 0 := by
    unfold accPr
    rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
  rw [h0, Escapes.term_nil]
  have := Pr_nonneg fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
    ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p
  linarith

end At

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable simSession)
open FlockSoundness.Discharge.Composed (outJ repJ sessionZC_eq_modelSession)

local notation "𝔽" => FlockLevel3.GF128

/-! ## Outputs at the empty masked openings -/

/-- **An output at the empty masked openings opens every slot with an empty row.** -/
theorem slotOpensZ_row_nil {D Sl : Type} [Inhabited D] [Inhabited Sl] {out : OutZC 𝔽 D Sl}
    (h : out.opens = nulOpZ) {r ℓ Q : ℕ} {x : ℕ × List 𝔽 × Sl × List D} (hx : x ∈ slotOpensZ out r ℓ Q) :
    x.2.1 = [] := by
  unfold slotOpensZ at hx
  split at hx
  · simp at hx
  · obtain ⟨pj, -, rfl⟩ := List.mem_map.1 hx
    show (out.opens.rows r ℓ pj.2).1 = []
    rw [h]
    rfl

/-- **An empty row never verifies at a positive width.** -/
theorem not_verifiedZ_nil {D Sl : Type} (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (cap : Model.Cap D) {w : ℕ} (hw : 0 < w) (depth : ℕ)
    {x : ℕ × List 𝔽 × Sl × List D} (hx : x.2.1 = []) : ¬ VerifiedZ H E Lf cap w depth x := by
  rintro ⟨hlen, -⟩
  rw [hx, List.length_nil] at hlen
  omega

/-- Level 0's row width under `--zk` is positive. -/
theorem width₀_pos {mPts : ℕ} (T : TabZK 𝔽 mPts) : 0 < width₀ T := by
  unfold width₀
  have := Nat.two_pow_pos T.sch.k0
  omega

/-- **An output at the empty masked openings never self-clashes**: its rows are empty, so none verifies at level 0's
width or at a level's `2^lanes`, and its pads pairs are all `(0, 0)`. -/
theorem not_selfClashZ_nul {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts : ℕ} (Hs : Hashes 𝔽 D Sl)
    (T : TabZK 𝔽 mPts) {out : OutZC 𝔽 D Sl} (h : out.opens = nulOpZ) : ¬ SelfClashZ Hs T out := by
  rintro (⟨x, hx, -, -, -, hv, -⟩ | ⟨r, -, ℓ, -, cap, -, hc⟩ | ⟨r, -, root, jj, -, jj', -, p, -, -, hne⟩)
  · refine not_verifiedZ_nil Hs.H Hs.E Hs.Lf out.cap₀ (width₀_pos T) _ ?_ hv
    unfold opens₀Z at hx
    rcases List.mem_append.1 hx with hx | hx
    · exact slotOpensZ_row_nil h hx
    · exact slotOpensZ_row_nil h hx
  · exact not_clashL_nil Hs.H Hs.E Hs.Lf _ _ _ (fun x hx => slotOpensZ_row_nil h hx) hc
  · rw [h] at hne
    exact hne rfl

/-! ## The masked self-clash on accepted live runs -/

section Sim

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

/-- **The masked session's escapes arise only on accepted live runs**: an event `Q` of the tables' outputs that no
outputs at the empty openings satisfy is, against the simulated strategy (`simSessZV`), no likelier than a live
outcome the live verdict accepts whose decoded outputs are in `Q` (`Claims.Bridge.prob_le_decoded`). -/
theorem sessionZC_le_decoded (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    {Q : List (OutZC 𝔽 D Sl) → Prop}
    (hQ : ∀ (caps : Fin ts.length → Model.Cap D) outs, (∀ j : Fin ts.length, ∃ pend,
      outs.getD j dfltOutZC = outJ Hs ts j (caps j) pend nulOpZ) → ¬ Q outs)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob Q (sessionZC A Hs ts) (simSessZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr) ≤
      prob (fun b => ∃ a, Claims.Bridge.RetOK (outJ Hs ts) pL opn dec one fin capOf (repJ A Hs ts) b a ∧ Q a)
        (liveSession dg ts.length mPts N) Pr := by
  unfold simSessZV
  rw [FlockSoundness.Discharge.Exec.prob_cast]
  exact Claims.Bridge.prob_le_decoded _ _ dfltOutZC pL opn dg dec one fin capOf N _ _ hdec _ hQ Pr

/-- **A decoded masked self-clash on an accepted live run**: the live verdict `pL` holds at the outcome, and table
`j`'s output, at the reps' pendings on their runs of their whole decoded tapes, the caps the roots decode to and the
openings the proof files decode to (`Claims.Bridge.RetOK`), self-clashes at table `T`. -/
def LiveSelfClashZ {J : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (out : Fin J → Model.Cap D → PendZ 𝔽 D Sl × PendZ 𝔽 D Sl → OpensZ 𝔽 D Sl → OutZC 𝔽 D Sl)
    (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → OpensZ 𝔽 D Sl)
    (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Model.Cap D)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (PendZ 𝔽 D Sl)) (j : ℕ) (b : LiveOutJ J mPts) : Prop :=
  ∃ a, Claims.Bridge.RetOK out pL opn dec one fin capOf G b a ∧ SelfClashZ Hs T (a.getD j dfltOutZC)

open Classical in
omit [Fintype K] [Inhabited D] [Inhabited Sl] in
theorem prob_sessionZC_toReal (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (Q : List (OutZC 𝔽 D Sl) → Prop) (τ : Strategy (sessionZC A Hs ts)) :
    (prob Q (sessionZC A Hs ts) τ).toReal = expect (ind Q) (sessAfterZC (gamesZC A Hs ts τ.1)) τ.2 :=
  prob_toReal Q _ τ

open Classical in
/-- **Table `j`'s self-clash after `Commit` arises only on accepted live runs**: against the simulated strategy at the
caps the live prover's roots decode to, table `j`'s self-clash is no likelier than a decoded self-clash of table `j`
on a live run the live verdict accepts (`LiveSelfClashZ`). -/
theorem selfClash_le_liveJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀)
    (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T) :
    expect (ind fun outs => SelfClashZ Hs T (outs.getD j dfltOutZC)) _
        (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT) ≤
      (prob (LiveSelfClashZ Hs T (outJ Hs ts) pL opn dec one fin capOf (repJ A Hs ts) j)
        (liveSession dg ts.length mPts N) Pr).toReal := by
  unfold simAtJZ
  rw [expect_afterJZ, ← prob_sessionZC_toReal]
  refine ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)) ?_
  refine sessionZC_le_decoded A Hs ts ok pL opn dg dec one fin capOf N hdec (fun caps outs h hq => ?_) Pr
  obtain ⟨pend, hp⟩ := h j
  rw [hp] at hq
  exact not_selfClashZ_nul Hs T rfl hq

open Classical in
/-- **Over a mixture of live strategies with common roots `rootB₀`**: the mean self-clash of table `j` after `Commit`,
over the simulated strategies, is at most the mean probability of a decoded self-clash of table `j` on an accepted
live run. -/
theorem mean_selfClash_le_liveJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) {τ : Type} [Fintype τ] (σL : τ → Strategy (liveSession dg ts.length mPts N))
    (hσ : ∀ u, (σL u).1 = rootB₀) (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T) :
    (∑ u, expect (ind fun outs => SelfClashZ Hs T (outs.getD j dfltOutZC)) _
        (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) j hT)) / Fintype.card τ ≤
      (∑ u, (prob (LiveSelfClashZ Hs T (outJ Hs ts) pL opn dec one fin capOf (repJ A Hs ts) j)
        (liveSession dg ts.length mPts N) (σL u)).toReal) / Fintype.card τ :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    selfClash_le_liveJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) j hT)
    (Nat.cast_nonneg _)

end Sim

/-! ## The live sessions' bound at every table, with escapes on accepted runs -/

open Classical in
/-- **The live sessions' bound at the masked tables, with escapes on accepted runs**: `ZkLiveAtJ` with every table's
fork events only where both forked deployed runs accept and its one-run event only where its run accepts, and with
table `j`'s one-run event bounding the live strategies' decoded self-clashes of table `j` on accepted live runs
(`LiveSelfClashZ`). -/
def ZkLiveAccAtJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
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
      (_tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog)
      (_acc : ∀ j i ω s s', ev j i ω s s' → (Claims.run sha512 P (V true) x ω s).out = .acc ∧
        (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).out = .acc)
      (_acc₁ : ∀ j ω s, ev₁ j ω s → (Claims.run sha512 P (V true) x ω s).out = .acc),
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
          (∑ u, (prob (LiveSelfClashZ Hs (T j) (outJ Hs (tabsJ (tables true x).length T)) pL opn dec one fin capOf
              (repJ A Hs (tabsJ (tables true x).length T)) j)
            (liveSession dg (tabsJ (tables true x).length T).length mPts N) (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

/-- **`ZkLiveAccAtJ` gives `ZkLiveAtJ`**: the decoded self-clashes of table `j` on accepted live runs bound the
simulated strategies' self-clash of table `j` after `Commit` (`mean_selfClash_le_liveJZ`). -/
theorem zkLiveAtJ_of_acc {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {A : Model.Arith 𝔽 K} {Hs : Hashes 𝔽 D Sl} {Ok : Claims.Msg → ℕ → List 𝔽 → Prop} {x : Claims.Msg} {mPts : ℕ}
    {T : ℕ → TabZK 𝔽 mPts} (hl : ZkLiveAccAtJ A Hs Ok x T) : ZkLiveAtJ A Hs Ok x T := by
  intro Ω _ _ P
  obtain ⟨rd, hrd, ev, ev₁, tied, tied₁, -, -, hω⟩ := hl Ω P
  refine ⟨rd, hrd, ev, ev₁, tied, tied₁, fun ω j hj C hok => ?_⟩
  obtain ⟨ok, pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hok', hdec, τ, _, _, σL, hσ, htr, hadv, hP, hTau, hsc⟩ :=
    hω ω j hj C hok
  exact ⟨ok, pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hok', hdec, τ, inferInstance, inferInstance, σL, hσ, htr,
    hadv, hP, hTau, le_trans (mean_selfClash_le_liveJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ σL hσ
      (finJ T hj) (tabsJ_get _ T _)) hsc⟩

/-- **Every masked table's bound at a statement with escapes on accepted runs, from the live sessions' bound** at the
masked tables `T j`: the tables' old escapes joined by round (`joinEscZ`), on accepted runs (`joinEscZ_onAcc`), with
`zk_le_sessionsJ` at the simulated strategies after `Commit` at table `j` (`simAtJZ`). -/
theorem zk_tableBoundJAccAt_of_liveJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
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
    (hl : ZkLiveAccAtJ A Hs Ok x T) :
    Claims.Flock.TableBoundJAccAt sha512 (deployZk rel) (wordLenJ true) Ok x := by
  intro Ω _ _ P
  obtain ⟨rd, hrd, ev, ev₁, tied, tied₁, hacc, hacc₁, hω⟩ := hl Ω P
  refine ⟨Claims.Flock.joinEscZ sha512 (V true) P x (tables true x).length (fun j => (T j).sch)
      (fun j => (T j).N) rd ev ev₁ tied tied₁,
    Claims.Flock.joinEscZ_onAcc sha512 (V true) P x (tables true x).length (fun j => (T j).sch)
      (fun j => (T j).N) rd ev ev₁ tied tied₁ hacc hacc₁,
    joinEscZ_fitsFork_zk rel P x st hst (fun j hj => (hT j hj).1) (fun j hj => (hT j hj).2.1) _
      (fun j hj => (hT j hj).2.2.1) _ (fun j hj => (hT j hj).2.2.2.2.2.2.2.2.2) rd ev ev₁ tied tied₁,
    fun ω j hj C hok hg => ?_⟩
  rw [hn] at hj
  obtain ⟨ok, pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hok', hdec, τ, _, _, σL, hσ, htr, hadv, hP, hTau, hsc⟩ :=
    hω ω j hj C hok
  have hlen := take_gamesZC_length A Hs (tabsJ (tables true x).length T) (capOf rootB₀) (finJ T hj)
  refine zk_le_sessionsJ A hA Hs st hst T hT P rd hrd ev ev₁ tied tied₁ ω hj C (hnc j hj C hok hg) _ _
    (capOf rootB₀ (finJ T hj)) (fun u => simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u)
      (finJ T hj) (tabsJ_get _ T _)) ?_ hadv hP hTau ?_
  · refine le_trans htr (le_trans (mean_live_le_simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ σL hσ
      (finJ T hj) (tabsJ_get _ T _)) ?_)
    rw [hlen]
    exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
      expect_mono (ind_mono fun outs h => hok' _ h) _ _) (Nat.cast_nonneg _)
  · rw [hlen]
    exact le_trans (mean_selfClash_le_liveJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ σL hσ
      (finJ T hj) (tabsJ_get _ T _)) hsc

/-! ## The deployed masked tables -/

/-- **The live sessions' bound at a masked statement of any number of tables, with escapes on accepted runs**:
`ZkLiveAccAtJ` where `ZkLiveJ` reads `ZkLiveAtJ`. -/
def ZkLiveAccJ (x : Claims.Msg) : Prop :=
  ∀ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts),
    ZkLiveAccAtJ execArith Discharge.ZkExec.hs512 (wideJ true) x (zkTabs h0 L hz hk hlay)

theorem zkLiveJ_of_acc {x : Claims.Msg} (hl : ZkLiveAccJ x) : ZkLiveJ x :=
  fun ins spec sts hset h0 L hz hk hlay => zkLiveAtJ_of_acc (hl ins spec sts hset h0 L hz hk hlay)

/-- **Every masked table's bound at a statement in scope, with escapes on accepted runs**: at tables of full-width
columns, where the statement parses in scope and sets up at least one table, `zk_tableBoundJAccAt_of_liveJ` at
C-Flock ZK's padded tables (`zk_atTabs`). -/
theorem zk_tableBoundJAccAt_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (hl : ZkLiveAccJ x) :
    Claims.Flock.TableBoundJAccAt sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x := by
  obtain ⟨L, hz, hk, hlay, hb⟩ := zk_atTabs hs hin hset h0
    (B := fun T => ZkLiveAccAtJ execArith Discharge.ZkExec.hs512 (wideJ true) x T →
      Claims.Flock.TableBoundJAccAt sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x)
    fun T hn hst hT hnc => zk_tableBoundJAccAt_of_liveJ execArith execArith_correct Discharge.ZkExec.hs512
      Readable.relation (wideJ true) hn (stJ h0) hst T hT hnc
  exact hb (hl ins spec sts hset h0 L hz hk hlay)

/-- **Every masked table's bound at every statement, with escapes on accepted runs**, at full-width columns, from
`ZkLiveAccJ` at the statements some run accepts: they parse in scope and set up at least one table, and elsewhere no
run accepts and `k ≤ 2·weight` (`zk_k_le`). -/
theorem zk_tableBoundJAccAt (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccJ x) (x : Claims.Msg) :
    Claims.Flock.TableBoundJAccAt sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact Claims.Flock.tableBoundJAccAt_of_noAcc _ _ _ _ x hno (zk_k_le Readable.relation x)
      (εtable_nonneg true x)
  · obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
    obtain ⟨ins, spec, sts, record, proofs, hset, hv, -⟩ := acc_check hno
    exact zk_tableBoundJAccAt_live hs hin hset (served_zk (by simpa using hv)).1 (hl x hno)

/-- **The deployed masked verifier: KS**, at SHA-512 at the deployed relation, at every number of tables, from the live
sessions' bound with escapes on accepted runs at the statements some run accepts (`ZkLiveAccJ`) and `ExtendJ` at
full-width columns. -/
theorem zk_KS_of_liveAccJ (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccJ x)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_of_liveJ (fun x hno => zkLiveJ_of_acc (hl x hno)) hext

end FlockDeploy.KS
