import Proofs.Flock.Claims.KS.PlainKS
import Proofs.Flock.Claims.KS.Escapes
import Proofs.Flock.Claims.KS.OnAcc

/-!
# The plain table bound with escapes on accepted runs

`PlainLiveAccAt` is `PlainLiveAt` with the escapes' events on accepted deployed runs, and the self-clash it asks the
events to bound read on accepted live runs: the decoded self-clash of a live outcome the verdict accepts
(`LiveSelfClash`), which bounds the simulated strategies' self-clash after `Commit` (`mean_selfClash_le_live`). It gives
`PlainLiveAt` (`plainLiveAt_of_acc`) and the plain table bound with the old escapes on accepted runs
(`plain_tableBoundθAccAt_of_live`, through `plain_le_sessions` and `Flock.oldEsc_onAcc`). At a statement in scope that
is `plain_tableBoundθAccAt_live`, and at every statement, from `PlainLiveAcc` where some run accepts,
`plain_tableBoundθAccAt` (where none does, with no escapes). `plain_KS_of_liveAcc` is `plain_KS_of_live` from it.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

open Classical Discharge.Tables in
/-- **The live sessions' bound at table 0, with escapes on accepted runs**: `PlainLiveAt` with each fork event only
where both forked deployed runs accept and each one-run event only where its run accepts, and with the events bounding
the live strategies' decoded self-clashes on accepted live runs. -/
def PlainLiveAccAt (Ok : Claims.Msg → List 𝔽 → Prop) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h : tables false x = [(spec, st)]) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin ((V false).k x) → Coin) →
        (Fin ((V false).k x) → Coin) → Prop)
      (ev₁ : Ω → (Fin ((V false).k x) → Coin) → Prop)
      (_tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
        ((Claims.run sha512 P (V false) x ω s).vLog ++
          (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog))
      (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog)
      (_acc : ∀ r ℓ ω s s', ev r ℓ ω s s' → (Claims.run sha512 P (V false) x ω s).out = .acc ∧
        (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).out = .acc)
      (_acc₁ : ∀ ω s, ev₁ ω s → (Claims.run sha512 P (V false) x ω s).out = .acc),
      ∀ ω (C : Fin (wordLen false x) → List 𝔽), (∀ p, Ok x (C p)) →
        ∃ (pL : LiveOutJ [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length spec.mPts → Prop)
          (opn : Array ByteArray → Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length →
            Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
          (dg : ByteArray → ByteArray)
          (dec : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → ℕ → ByteArray → List Msg)
          (one : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → ℕ → Bool)
          (fin : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → Fin 2 → Array ByteArray →
            List Msg)
          (capOf : Array ByteArray → Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length →
            Model.Cap Refine.Dig)
          (N : ℕ) (rootB₀ : Array ByteArray)
          (hdec : ∀ j, DecodesTable
            (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
            (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
              ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOf rootB₀ ⟨0, Nat.one_pos⟩) (Claims.Flock.oracleOf C) o ∧
              ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
            pL opn dec one fin capOf
            (fun j => repsOf (D := Refine.Dig) execArith
              ([plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].get j))
            j)
          (τ : Type) (_ : Fintype τ) (_ : Nonempty τ)
          (σL : τ → Strategy
            (liveSession dg [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length spec.mPts N))
          (hσ : ∀ u, (σL u).1 = rootB₀),
          (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
              ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
                Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧
                  u ≠ C p) ≤
            (∑ u, (prob pL _ (σL u)).toReal) / Fintype.card τ ∧
          (∀ r ℓ, (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
              k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val (ℓ.val + 1)
              (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
                (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀
                (σL u) (hσ u))) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2) ∧
          (∑ u, (prob (LiveSelfClash Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pL opn
              dec one fin capOf fun j => repsOf (D := Refine.Dig) execArith
                ([plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].get j))
            (liveSession dg [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length spec.mPts N)
              (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s

open Classical Discharge.Tables in
/-- **`PlainLiveAccAt` gives `PlainLiveAt`**: the decoded self-clashes on accepted live runs bound the simulated
strategies' self-clash after `Commit`. -/
theorem plainLiveAt_of_acc {Ok : Claims.Msg → List 𝔽 → Prop} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m} {ptLocal : ℕ}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts}
    (hl : PlainLiveAccAt Ok h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay) :
    PlainLiveAt Ok h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay := by
  intro Ω _ _ P
  obtain ⟨ev, ev₁, tied, tied₁, -, -, hω⟩ := hl Ω P
  refine ⟨ev, ev₁, tied, tied₁, fun ω C hok => ?_⟩
  obtain ⟨pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hdec, τ, _, _, σL, hσ, htr, hadv, hsc⟩ := hω ω C hok
  exact ⟨pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hdec, τ, inferInstance, inferInstance, σL, hσ, htr, hadv,
    le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀ σL hσ) hsc⟩

open Classical Discharge.Tables in
/-- **The plain table bound at a statement, with escapes on accepted runs, from live sessions**: the old escapes at the
plain fork rounds, on accepted runs, with `plain_le_sessions` at the simulated strategies after `Commit` (`simAt`). -/
theorem plain_tableBoundθAccAt_of_live (rel : Readable.Parsed → Readable.Relation)
    (Ok : Claims.Msg → List 𝔽 → Prop) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h : tables false x = [(spec, st)]) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain rel) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hl : PlainLiveAccAt Ok h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay) :
    Claims.Flock.TableBoundθAccAt sha512 (deployPlain rel) (wordLen false) Ok x := by
  intro Ω _ _ P
  obtain ⟨ev, ev₁, tied, tied₁, hacc, hacc₁, hω⟩ := hl Ω P
  refine ⟨Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
    ev ev₁ tied tied₁, Claims.Flock.oldEsc_onAcc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (plainJ (tables_head_one h) hsch) ev ev₁ tied tied₁ hacc hacc₁,
    oldEsc_fitsFork_plainJ rel P x (tables_head_one h) hsch ev ev₁ tied tied₁, fun ω C hok hg => ?_⟩
  obtain ⟨pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hdec, τ, _, _, σL, hσ, htr, hadv, hsc⟩ := hω ω C hok
  exact plain_le_sessions h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay P ev ev₁ tied tied₁ ω C (hnc C hok hg) τ
    (capOf rootB₀ ⟨0, Nat.one_pos⟩) (fun u => simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀ σL hσ))
    hadv
    (le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀ σL hσ) hsc)

/-- **The live sessions' bound at a plain statement, with escapes on accepted runs**: `PlainLiveAccAt` where
`PlainLive` reads `PlainLiveAt`. -/
def PlainLiveAcc (x : Claims.Msg) : Prop :=
  ∀ (spec : SessionSpec) (st : Stmt) (h : tables false x = [(spec, st)]) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts),
    PlainLiveAccAt (wide false) h L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay

theorem plainLive_of_acc {x : Claims.Msg} (hl : PlainLiveAcc x) : PlainLive x :=
  fun spec st h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay =>
    plainLiveAt_of_acc (hl spec st h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay)

/-- **The plain table bound at a statement in scope, with escapes on accepted runs**: at tables of full-width columns,
where the statement parses in scope and sets up. -/
theorem plain_tableBoundθAccAt_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) (hl : PlainLiveAcc x) :
    Claims.Flock.TableBoundθAccAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  refine plain_tableBoundθAccAt_of_live Readable.relation (wide false) ht L _ k0 l₀ l₁ rest ext hsch hk0 hm13
    PT_LOCAL hlay (fun C hw hg => ?_) (hl spec _ ht L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay)
  have := plain_not_committed hs hin hset h0 L C hw hg
  rwa [e] at this

/-- **The plain table bound at every statement, with escapes on accepted runs**, at full-width columns, from
`PlainLiveAcc` at the statements some run accepts: they parse in scope and set up, and elsewhere no run accepts and
`k ≤ 2·weight` (`plain_k_le`). -/
theorem plain_tableBoundθAccAt (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLiveAcc x)
    (x : Claims.Msg) :
    Claims.Flock.TableBoundθAccAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V false) x
  · exact Claims.Flock.tableBoundθAccAt_of_noAcc _ _ _ _ x hno (plain_k_le Readable.relation x)
      (εtable_nonneg false x)
  · obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
    obtain ⟨ins, spec, sts, -, -, hset, -, -⟩ := acc_check hno
    exact plain_tableBoundθAccAt_live hs hin hset (hl x hno)

/-- **The deployed plain verifier: KS**, at SHA-512 at the deployed relation, from the live sessions' bound with
escapes on accepted runs at the statements some run accepts, and `Extendθ` at full-width columns. -/
theorem plain_KS_of_liveAcc (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLiveAcc x)
    (hext : Claims.Flock.Extendθ (deployPlain Readable.relation) (wordLen false) (wide false)) :
    Claims.Flock.KS sha512 (deployPlain Readable.relation) :=
  plain_KS_of_live (fun x hno => plainLive_of_acc (hl x hno)) hext

end FlockDeploy.KS
