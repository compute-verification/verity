import Proofs.Flock.Claims.KS.ZkBoundW

/-!
# Every masked table's bound from the live sessions, off the table on a witness lane

`ZkLiveAccAtJ` at the verdict `ZkBoundW` reads (`ZkLiveAccAtWJ`): table `j`'s decoder verdict implies acceptance with
no level-0 opening off `C` on a witness lane unless it self-clashes (`OffW₀Z`), and level 0's fork event at each table
(`Claims.Flock.IZW`) bounds the simulated strategies' `Adv₀` after `Commit`. It gives the masked sessions `ZkBoundW`
reads (`zkSessWJ_of_liveW`), so every masked table's bound with the escapes counted per round
(`zk_tableBoundJAccAtR_of_liveW`). At the deployed masked record that is `zk_tableBoundJAccAtR_live`, and at every
statement, from `ZkLiveAccWJ` where some run accepts, `zk_tableBoundJAccAtR` (where none does, with no escapes:
`Claims.Flock.tableBoundJAccAtR_of_noAcc`). At table 0 it is `TableBoundθAccAt` with the escapes counted per round
(`Claims.Flock.TableBoundθAccAtR`, which `TableBoundθAccAt` gives: `Claims.Flock.TableBoundθAccAt.tableBoundθAccAtR`;
`Claims.Flock.TableBoundJAccAtR.tableBoundθAccAtR`), so the masked `plain_tableBoundθAccAt` counted per round follows
from `ZkLiveAccWJ` where some run accepts (`zk_tableBoundθAccAtR`).
-/

namespace Claims.Flock

/-! ## Table 0's bound, the escapes counted per round -/

section At

variable {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`TableBoundθAccAt` with the escapes counted per round** (`Escapes.FitsForkR`), on accepted first runs
(`Escapes.OnAcc₁`). -/
def TableBoundθAccAtR [Nonempty Coin] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.OnAcc₁ ∧
    e.FitsForkR (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) →
      ¬ GoodTable D N x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p

theorem TableBoundθAccAt.tableBoundθAccAtR [Nonempty Coin] {H : Msg → Msg} {D : Deploy Coin Col T} {N : Msg → ℕ}
    {Ok : Msg → Col → Prop} {x : Msg} (h : TableBoundθAccAt H D N Ok x) (hk : 0 < D.V.k x) :
    TableBoundθAccAtR H D N Ok x := by
  intro Ω _ _ P
  obtain ⟨e, ha, hf, hb⟩ := h Ω P
  exact ⟨e, ha.onAcc₁, hf.fitsForkR (Nat.cast_nonneg _) hk, hb⟩

/-- **Where the verifier accepts no run**, `TableBoundJAccAtR` holds with no escapes, when the verifier takes a round,
the weight covers `k` and `ε_table ≥ 0`. -/
theorem tableBoundJAccAtR_of_noAcc (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (Ok : Msg → ℕ → Col → Prop) (x : Msg) (hno : NoAcc H D.V x) (hk : (D.k x : ℝ) ≤ 2 * D.weight x)
    (hε : 0 ≤ D.εtable x) (hVk : 0 < D.V.k x) : TableBoundJAccAtR H D N Ok x :=
  (tableBoundJAccAt_of_noAcc H D N Ok x hno hk hε).tableBoundJAccAtR hVk

/-- **Table 0's bound from every table's, the escapes counted per round**: `TableBoundJAccAt.tableBoundθAccAt`. -/
theorem TableBoundJAccAtR.tableBoundθAccAtR [Nonempty Coin] {H : Msg → Msg} {D : Deploy Coin Col T}
    {N : Msg → ℕ → ℕ} {Ok : Msg → ℕ → Col → Prop} {Ok₀ : Msg → Col → Prop} {x : Msg}
    (h : TableBoundJAccAtR H D N Ok x) (hn : 0 < nTab D x) (hok : ∀ u, Ok₀ x u → Ok x 0 u) :
    TableBoundθAccAtR H D (fun x => N x 0) Ok₀ x := by
  intro Ω _ _ P
  obtain ⟨e, ha, hf, hb⟩ := h Ω P
  exact ⟨e, ha, hf, fun ω C hC hg => hb ω 0 hn C (fun p => hok _ (hC p)) hg⟩

end At

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable simSession)
open FlockSoundness.Discharge.Composed (outJ repJ)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The live sessions' bound at every table, off the table on a witness lane -/

open Classical in
/-- **The live sessions' bound at the masked tables, off the table on a witness lane, with escapes on accepted
runs**: `ZkLiveAccAtJ` with level 0's fork event at each table (`Claims.Flock.IZW`) bounding the simulated strategies'
`Adv₀` after `Commit` at table `j`, every table's decoder verdict implying acceptance, and table `j`'s also no level-0
opening off `C` on a witness lane unless it self-clashes (`OffW₀Z`). Each fork bound is its event's probability and the slack of the
fork at its round (`Claims.Flock.forkSlack`), as `ZkSessWJ` reads. -/
def ZkLiveAccAtWJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) (x : Claims.Msg) {mPts : ℕ}
    (T : ℕ → TabZK 𝔽 mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd δ)
      (ev : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
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
            ¬ (OffW₀Z Hs (T j) (capOf rootB₀ (finJ T hj)) (Claims.Flock.oracleOf C) o ∧ ¬ SelfClashZ Hs (T j) o))
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
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advPadsSZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj)) r.val
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advTauSZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj)) r.val
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
          ((∑ u, adv₀SZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj))
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr ()) ω s.1 s.2) + Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω) ∧
          (∑ u, (prob (LiveSelfClashZ Hs (T j) (outJ Hs (tabsJ (tables true x).length T)) pL opn dec one fin capOf
              (repJ A Hs (tabsJ (tables true x).length T)) j)
            (liveSession dg (tabsJ (tables true x).length T).length mPts N) (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The live sessions' bound at the masked tables, off the table on a witness lane, with escapes on accepted
runs**: `ZkLiveAccAtJ` with level 0's fork event at each table (`Claims.Flock.IZW`) bounding the simulated strategies'
`Adv₀` after `Commit` at table `j`, every table's decoder verdict implying acceptance, and table `j`'s also no level-0
opening off `C` on a witness lane unless it self-clashes (`OffW₀Z`). Each fork bound is its event's probability and the slack of the
fork at its round (`Claims.Flock.forkSlack`), as `ZkSessWJ` reads. -/
def ZkLiveAccAtWJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) (x : Claims.Msg) {mPts : ℕ}
    (T : ℕ → TabZK 𝔽 mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd δ)
      (ev : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
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
            ¬ (OffW₀Z Hs (T j) (capOf rootB₀ (finJ T hj)) (Claims.Flock.oracleOf C) o ∧ ¬ SelfClashZ Hs (T j) o))
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
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advPadsSZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj)) r.val
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
          (∀ r : Fin 2, (∑ u, advTauSZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj)) r.val
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
          ((∑ u, adv₀SZ A Hs (T j) _ _ (capOf rootB₀ (finJ T hj))
              (simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
                (tabsJ_get _ T _))) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr ()) ω s.1 s.2) + Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω) ∧
          (∑ u, (prob (LiveSelfClashZ Hs (T j) (outJ Hs (tabsJ (tables true x).length T)) pL opn dec one fin capOf
              (repJ A Hs (tabsJ (tables true x).length T)) j)
            (liveSession dg (tabsJ (tables true x).length T).length mPts N) (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

/-- **The live sessions' bound off the table on a witness lane gives the masked sessions' after `Commit` there**
(`ZkSessWJ`), the mixture at table `j` the simulated strategies after `Commit` there (`simAtJZ`) of live strategies
whose roots agree, the decoded self-clashes on accepted live runs bounding theirs (`mean_selfClash_le_liveJZ`). -/
theorem zkSessWJ_of_liveW {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {A : Model.Arith 𝔽 K} {Hs : Hashes 𝔽 D Sl} {Ok : Claims.Msg → ℕ → List 𝔽 → Prop} {x : Claims.Msg} {mPts : ℕ}
    {T : ℕ → TabZK 𝔽 mPts} (hl : ZkLiveAccAtWJ A Hs Ok x T) : ZkSessWJ A Hs Ok x T := by
  intro Ω _ _ P
  obtain ⟨rd, hrd, hr0, δ, hδ, hfit, ev, ev₁, tied, tied₁, hacc, hacc₁, hω⟩ := hl Ω P
  refine ⟨rd, hrd, hr0, δ, hδ, hfit, ev, ev₁, tied, tied₁, hacc, hacc₁, fun ω j hj C hok => ?_⟩
  obtain ⟨ok, pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hok', hacc', hdec, τ, _, _, σL, hσ, htr, hadv, hP, hTau,
    h₀, hsc⟩ := hω ω j hj C hok
  have hlen := take_gamesZC_length A Hs (tabsJ (tables true x).length T) (capOf rootB₀) (finJ T hj)
  refine ⟨τ, inferInstance, inferInstance, _, _, capOf rootB₀ (finJ T hj),
    fun u => simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u) (finJ T hj)
      (tabsJ_get _ T _), ?_, hadv, hP, hTau, h₀, ?_⟩
  · refine le_trans htr (le_trans (mean_live_le_simAtJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ σL hσ
      (finJ T hj) (tabsJ_get _ T _)) ?_)
    rw [hlen]
    exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
      expect_mono (ind_mono fun outs h =>
        ⟨accUpTo_of_okAll ok hacc' (finJ T hj).isLt h, (hok' _ (h _)).2⟩) _ _) (Nat.cast_nonneg _)
  · rw [hlen]
    exact le_trans (mean_selfClash_le_liveJZ A Hs _ ok pL opn dg dec one fin capOf N hdec rootB₀ σL hσ
      (finJ T hj) (tabsJ_get _ T _)) hsc

end Gen

/-- **The live sessions' bound off the table on a witness lane gives the masked sessions' after `Commit` there**
(`ZkSessWJ`), the mixture at table `j` the simulated strategies after `Commit` there (`simAtJZ`) of live strategies
whose roots agree, the decoded self-clashes on accepted live runs bounding theirs (`mean_selfClash_le_liveJZ`). -/
theorem zkSessWJ_of_liveW {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {A : Model.Arith 𝔽 K} {Hs : Hashes 𝔽 D Sl} {Ok : Claims.Msg → ℕ → List 𝔽 → Prop} {x : Claims.Msg} {mPts : ℕ}
    {T : ℕ → TabZK 𝔽 mPts} (hl : ZkLiveAccAtWJ A Hs Ok x T) : ZkSessWJ A Hs Ok x T :=
  Gen.zkSessWJ_of_liveW (S := S₀) (K := K) (D := D) (Sl := Sl) (A := A) (Hs := Hs) (Ok := Ok) (x := x) (mPts := mPts)
    (T := T) (hl := hl)

namespace Gen

variable [S : ZkSetup]

/-- **Every masked table's bound at a statement from the live sessions' bound off the table on a witness lane**, the
escapes counted per round (`zk_tableBoundJAccAtR_of_sessionsW` at `zkSessWJ_of_liveW`). -/
theorem zk_tableBoundJAccAtR_of_liveW {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
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
    (hl : ZkLiveAccAtWJ A Hs Ok x T) :
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk rel) (wordLenJ true) Ok x :=
  zk_tableBoundJAccAtR_of_sessionsW A hA Hs rel Ok hn st hst T hT hnc (zkSessWJ_of_liveW hl)

end Gen

/-- **Every masked table's bound at a statement from the live sessions' bound off the table on a witness lane**, the
escapes counted per round (`zk_tableBoundJAccAtR_of_sessionsW` at `zkSessWJ_of_liveW`). -/
theorem zk_tableBoundJAccAtR_of_liveW {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
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
    (hl : ZkLiveAccAtWJ A Hs Ok x T) :
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk rel) (wordLenJ true) Ok x :=
  Gen.zk_tableBoundJAccAtR_of_liveW (S := S₀) (K := K) (D := D) (Sl := Sl) (A := A) (hA := hA) (Hs := Hs) (rel := rel)
    (Ok := Ok) (x := x) (hn := hn) (spec := spec) (st := st) (hst := hst) (T := T) (hT := hT) (hnc := hnc) (hl := hl)

/-! ## The deployed masked tables -/

/-- **The live sessions' bound at a masked statement of any number of tables, off the table on a witness lane**:
`ZkLiveAccAtWJ` where `ZkLiveAccJ` reads `ZkLiveAccAtJ`. -/
def ZkLiveAccWJ (x : Claims.Msg) : Prop :=
  ∀ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts),
    ZkLiveAccAtWJ execArith Discharge.ZkExec.hs512 (wideJ true) x (zkTabs h0 L hz hk hlay)

namespace Gen

variable [S : ZkSetup]

/-- **The live sessions' bound at a masked statement of any number of tables, off the table on a witness lane**:
`ZkLiveAccAtWJ` where `ZkLiveAccJ` reads `ZkLiveAccAtJ`. -/
def ZkLiveAccWJ (x : Claims.Msg) : Prop :=
  ∀ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts),
    ZkLiveAccAtWJ execArith Discharge.ZkExec.hs512 (wideJ true) x (zkTabs h0 L hz hk hlay)

/-- **Every masked table's bound at a statement in scope, off the table on a witness lane**, the escapes counted per
round: `zk_tableBoundJAccAtR_of_liveW` at C-Flock ZK's padded tables (`zk_atTabs`). -/
theorem zk_tableBoundJAccAtR_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (hl : ZkLiveAccWJ x) :
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x := by
  obtain ⟨L, hz, hk, hlay, hb⟩ := zk_atTabs hs hin hset h0
    (B := fun T => ZkLiveAccAtWJ execArith Discharge.ZkExec.hs512 (wideJ true) x T →
      Claims.Flock.TableBoundJAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x)
    fun T hn hst hT hnc => zk_tableBoundJAccAtR_of_liveW execArith execArith_correct Discharge.ZkExec.hs512
      Readable.relation (wideJ true) hn (stJ h0) hst T hT hnc
  exact hb (hl ins spec sts hset h0 L hz hk hlay)

end Gen

/-- **Every masked table's bound at a statement in scope, off the table on a witness lane**, the escapes counted per
round: `zk_tableBoundJAccAtR_of_liveW` at C-Flock ZK's padded tables (`zk_atTabs`). -/
theorem zk_tableBoundJAccAtR_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (hl : ZkLiveAccWJ x) :
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x :=
  Gen.zk_tableBoundJAccAtR_live (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec)
    (sts := sts) (hset := hset) (h0 := h0) (hl := hl)

namespace Gen

variable [S : ZkSetup]

/-- **Every masked table's bound at every statement, off the table on a witness lane**, the escapes counted per
round, at full-width columns, from `ZkLiveAccWJ` at the statements some run accepts; elsewhere no run accepts. -/
theorem zk_tableBoundJAccAtR (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccWJ x) (x : Claims.Msg) :
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact Claims.Flock.tableBoundJAccAtR_of_noAcc _ _ _ _ x hno (zk_k_le Readable.relation x)
      (εtable_nonneg true x) (V_k_pos true x)
  · obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
    obtain ⟨ins, spec, sts, record, proofs, hset, hv, -⟩ := acc_check hno
    exact zk_tableBoundJAccAtR_live hs hin hset (served_zk (by simpa using hv)).1 (hl x hno)

end Gen

/-- **Every masked table's bound at every statement, off the table on a witness lane**, the escapes counted per
round, at full-width columns, from `ZkLiveAccWJ` at the statements some run accepts; elsewhere no run accepts. -/
theorem zk_tableBoundJAccAtR (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccWJ x) (x : Claims.Msg) :
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) x :=
  Gen.zk_tableBoundJAccAtR (S := S₀) (hl := hl) (x := x)

/-- **Table 0's masked bound at every statement, off the table on a witness lane, with escapes on accepted runs
counted per round**, at full-width columns, from `ZkLiveAccWJ` at the statements some run accepts:
`zk_tableBoundθAccAt` at `ZkLiveAccWJ`. -/
theorem zk_tableBoundθAccAtR (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccWJ x) (x : Claims.Msg) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployZk Readable.relation) (wordLen true) (wide true) x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact (Claims.Flock.tableBoundθAccAt_of_noAcc _ _ _ _ x hno (zk_k_le Readable.relation x)
      (εtable_nonneg true x)).tableBoundθAccAtR (V_k_pos true x)
  · obtain ⟨s, hs, -⟩ := acc_inScope Readable.relation hno
    obtain ⟨ins, spec, sts, record, proofs, hset, hv, -⟩ := acc_check hno
    have h0 := (served_zk (by simpa using hv)).1
    have hlen : (tables true x).length = sts.size := by
      unfold tables
      rw [hset]
      simp
    have hn : Claims.Flock.nTab (deployZk Readable.relation) x = (tables true x).length := by
      rw [Claims.Flock.nTab_of_parse (D := deployZk Readable.relation) hs, tables_length hs (by rw [hset]; rfl)]
    rw [← wordLenJ_zero]
    exact (zk_tableBoundJAccAtR hl x).tableBoundθAccAtR (by omega)
      fun u hu => wideJ_zero_of_wide (List.ne_nil_of_length_pos (by omega)) hu

end FlockDeploy.KS
