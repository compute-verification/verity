import Proofs.Flock.Claims.KS.PlainBound
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Soundness.Discharge.Integrate.Accepted

/-!
# The deployed plain verifier's KS from the live sessions' bound at table 0

`plain_KS_of_live`: `Flock.KS` at SHA-512 at `deployPlain` at the deployed relation, from two premises:

* `PlainLive x` at every statement some run accepts: at table 0's data, every prover has escapes at the plain fork
  rounds and, at each tape and each table of full-width columns, live strategies whose verdict bounds the deployed runs
  accepting on the table, a decoder right for that verdict, and escapes' events bounding their `Adv` and self-clash
  (`PlainLiveAt`, the hypothesis `plain_tableBoundθAt_of_live` reads);
* `Extendθ` at full-width columns, which carries the bound to tables with unopened positions.

Everything else is proved here at a statement some run accepts, which parses in scope and sets up one table
(`plain_one`): table 0's layout and link layout (`Readable.tableAt`, `stmtOf_linkLayout`), its `fast100` schedule with
two levels and level 0's lanes (`fast100_plainSched`, `Integrate.model_sched`), and that a table that does not extract
is not committed to its model statement (`plain_not_committed`). `plain_tableBoundθ_live` is the bound at full-width
columns, the form a trimmed record without `Extendθ` reads.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

open Classical Discharge.Tables in
/-- **The live sessions' bound at table 0**: `plain_tableBoundθAt_of_live`'s hypothesis on the deployed prover, at
tables of `Ok` columns. Every prover has escapes at the plain fork rounds such that, at each tape and table `C`, live
strategies `σL` whose roots agree bound the deployed runs accepting on `C` by their verdict `pL`, the decoder is right
for `pL` accepting with no opening off `C` unless it self-clashes, and the escapes' events bound the simulated
strategies' `Adv` and self-clash. -/
def PlainLiveAt (Ok : Claims.Msg → List 𝔽 → Prop) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
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
      (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog),
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
          (∑ u, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
              SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                (outs.getD 0 dfltOutL))
            (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
              ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
            (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
              (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀
              (σL u) (hσ u))) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s

/-- **The live sessions' bound at a plain statement**: `PlainLiveAt` at tables of full-width columns, at link layout
`PT_LOCAL`, wherever the setup has the one table `st`, at its every `fast100` schedule of two levels or more. -/
def PlainLive (x : Claims.Msg) : Prop :=
  ∀ (spec : SessionSpec) (st : Stmt) (h : tables false x = [(spec, st)]) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts),
    PlainLiveAt (wide false) h L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay

/-- The tables of a setup with one statement. -/
theorem tables_of_one {zk : Bool} {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf zk x = some (ins, spec, sts)) (h1 : sts.size = 1) :
    tables zk x = [(spec, sts[0]'(by omega))] := by
  unfold tables
  rw [hset]
  rcases sts with ⟨_ | ⟨a, _ | ⟨b, l⟩⟩⟩
  · simp at h1
  · rfl
  · simp at h1

/-- **Table 0 of a plain statement in scope**: its layout, its link layout at `PT_LOCAL` and its link points. -/
theorem plain_table0 {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) (h0 : 0 < sts.size) :
    ∃ L : Layout (sts[0]'h0), (L.model (Refine.regionsOf (sts[0]'h0))).LinkLayout PT_LOCAL spec.mPts := by
  obtain ⟨p, hp, rfl, hP, -, -⟩ := parsed_of_parse hs
  obtain ⟨hpst, hpr, hsr⟩ := parseBytes_setup hp hset
  subst hpst
  obtain ⟨-, -, -, -, -, ht, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at ht hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  obtain ⟨-, L, hwf, hr⟩ := Readable.tableAt (Readable.tablesOf_of_parses hP) ht hpi' h0
  rw [← hpr] at hsr
  unfold Readable.Read.setup at hsr
  rw [hpi'] at hsr
  have hpts := (Discharge.HiddenExec.setupTables_pointsAny hsr).2 0 h0
  refine ⟨L, (Refine.stmtOf_linkLayout _ hwf hr).1, ?_⟩
  show PT_LOCAL + ((p.sts[0]'h0).m - (p.sts[0]'h0).c.kLog) ≤ spec.mPts
  have h1 := hr.mPts
  have h2 := hr.hm
  omega

/-- **The plain schedule at `m`**, for `22 ≤ m ≤ 35`: `fast100`'s, of two levels or more, with level 0's lanes `k0`,
the old model's reading of the deployed `plainSched`. -/
theorem plain_sched {m : ℕ} (h22 : 22 ≤ m) (h35 : m ≤ 35) :
    ∃ k0 l₀ l₁ rest ext, Accounting.fast100 m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧ l₀.lanes = k0 ∧
      Refine.schedOf (Claims.Flock.plainSched m) = ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ := by
  set i : Claims.Flock.PlainIdx := ⟨m - 22, by omega⟩
  have hi : i.m = m := by simp only [i, Claims.Flock.PlainIdx.m]; omega
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := Discharge.Integrate.model_sched (Claims.Flock.plainSched_some i)
  have e : Refine.schedOf (Claims.Flock.plainSched i.m) = ⟨i.m, k0, l₀ :: l₁ :: rest, ext⟩ :=
    Option.some.inj ((Claims.Flock.fast100_plainSched i).symm.trans hsch)
  have hl := Claims.Flock.level₀_plainSched i
  have hk := Claims.Flock.k0_plainSched i
  rw [e] at hl hk
  rw [hi] at hsch e
  refine ⟨k0, l₀, l₁, rest, ext, hsch, ?_, e⟩
  have hl' : l₀ = Refine.levelOf (Claims.Flock.plainLevel0 i.m) := hl
  rw [hl']
  exact hk.symm

/-- **The plain table bound at a statement in scope, from the live sessions' bound**: at tables of full-width
columns, where the statement parses in scope and sets up. -/
theorem plain_tableBoundθAt_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) (hl : PlainLive x) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  refine plain_tableBoundθAt_of_live Readable.relation (wide false) ht L _ k0 l₀ l₁ rest ext hsch hk0 hm13 PT_LOCAL
    hlay (fun C hw hg => ?_) (hl spec _ ht L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay)
  have := plain_not_committed hs hin hset h0 L C hw hg
  rwa [e] at this

/-- **The plain table bound at full-width columns, from the live sessions' bound** at the statements some run
accepts. -/
theorem plain_tableBoundθ_live (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLive x) :
    Claims.Flock.TableBoundθ sha512 (deployPlain Readable.relation) (wordLen false) (wide false) :=
  plain_tableBoundθ_of_acc Readable.relation (wide false)
    fun x _ _ _ _ hno hs hin hset => plain_tableBoundθAt_live hs hin hset (hl x hno)

/-- **The deployed plain verifier: KS**, at SHA-512 at the deployed relation, from the live sessions' bound at the
statements some run accepts and `Extendθ` at full-width columns. -/
theorem plain_KS_of_live (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLive x)
    (hext : Claims.Flock.Extendθ (deployPlain Readable.relation) (wordLen false) (wide false)) :
    Claims.Flock.KS sha512 (deployPlain Readable.relation) :=
  plain_KS_of_table Readable.relation (plain_hone Readable.relation)
    (plain_tableBoundθ_of_extend Readable.relation (plain_colsOk sha512 Readable.relation) hext
      (plain_tableBoundθ_live hl))

end FlockDeploy.KS
