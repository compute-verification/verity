import Proofs.Flock.Claims.KS.PlainThird
import Proofs.Flock.Claims.KS.ForkSlack

/-!
# The plain chain from a simulator

`PlainSimWrPAt` asks, for each prover `P`, for chances `δ` whose slack weights fit level 0's room in the plain weight,
and at each tape for a live prover `P'` with a finite tape and coins `Φ` of `V false` at the tape and the live coins,
of uniform law, with three bounds:

* the first, from `P'` simulating `P`'s accepting runs at `Φ` (`PlainSimRun`) and the record's draw
  (`RecordDrawVerbatim`), by `plain_first_of_sim`;
* the third, from a live self-clash at `P'`'s outcome giving an accepting run of `P` at `Φ` with a collision among the
  verifier's calls (`PlainClashRun`), by `liveMix_le_Pr`;
* the second, the fork bound at `P'`'s strategies, at most the deployed fork with a collision at its round `j`
  (`plainForkEv`) and the slack `Σ_{j<i} Δ_{j,i} Pr[F_i] + Δ_{j,K} Pr[F₁]` (`Flock.forkSlack`), which a coupling of the
  fork's coin law gives (`Flock.ForkCoinCoupling`, `Flock.slack_of_coupling`).

The escapes are then the old ones and one combined fork event per round (`Flock.slackEsc`), which fit the weight
(`slackEsc_fitsFork_plain`), and `TableBoundθAt` holds at the statement (`plain_tableBoundθAt_of_sim`). `Φ` may be
`plainCoins` with the deployed coins as the tape, or `fnAt` with uniform functions of the live coins as the tape.

So KS at the deployed plain verifier's trimmed record holds from `PlainSimW` at the statements some run accepts and
`RecordDrawVerbatim` at them (`plain_KS_of_sim`). The strategies send `P`'s roots wherever `P'`'s first message is
`P`'s and the wire reads roots as the deployed link does (`plainRoot_of_first`). Where the self-clash is at level 0, the
collision in `PlainClashRun` follows from two different columns the run's reps open at one position
(`PlainRawClashRun`, `plainClashRun_of_raw`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The roots of `P`'s first message**, as `plain_first_of_run` reads them. -/
def plainRoot {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : Array ByteArray :=
  ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray

/-- **A live prover whose first message is `P`'s sends `P`'s roots**, where the wire reads roots as the deployed link
does (`liveStrat_fst`): the strategies' roots `PlainSimWrPAt` asks for. -/
theorem plainRoot_of_first {Ω τ : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) {w : Claims.Msg → Wire 65536 1 mPts} {P' : Claims.Prover (Ω × τ)}
    (hw : ∀ a, (w x).root a = ((linkOf a).getD default).1.toArray)
    (h1 : ∀ u, ((P'.next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1) (u : τ) :
    (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω := by
  rw [liveStrat_fst, hw, h1]
  rfl

/-- **An accepting run with a collision among the verifier's calls**: the one-run event the third bound reads. -/
def plainEv₁ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (s : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V false) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog

/-- **An accepting fork with a collision among the verifier's calls in its two runs**, at the fork's coin
`plainJ … r ℓ`: the event the second bound reads. -/
def plainForkEv {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) {m k0 : ℕ}
    {l₀ l₁ : Accounting.Level} {rest : List Accounting.Level} {ext : ℕ}
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) {Ω : Type} (P : Claims.Prover Ω)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (ω : Ω) (s s' : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V false) x ω s).out = .acc ∧
    (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).out = .acc ∧
    Claims.TiedColl sha512 ((Claims.run sha512 P (V false) x ω s).vLog ++
      (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog)

/-- **A live event at `P'`'s outcome gives an accepting run of `P` with a collision**: wherever `pL` holds at the live
outcome of `P'` at the tape `(ω, u)` and the live coins `r'`, `V false` accepts `P` at `Φ (u, r')` with a collision
among its calls (`plainEv₁`). -/
def PlainClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → plainEv₁ P x ω (Φ (u, r'))

/-- **A live event at `P'`'s outcome gives an accepting run of `P` whose reps open two different columns at one
position, or a collision**: `PlainClashRun` with a self-clash at level 0 asked without the hashes (`plainRawCols`).
`SelfClashL` also counts clashes at later levels, against the caps of the reps' pendings; those still need their
collision. -/
def PlainRawClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List 𝔽),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog)

/-- **The clash gives the collision**: two different columns of an accepting run at one position collide in its log
(`plain_selfClash_tied`), so `PlainRawClashRun` gives `PlainClashRun`. -/
theorem plainClashRun_of_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRun P x ω mPts N dg w P' Φ pL) :
    PlainClashRun P x ω mPts N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, ⟨q, a, b, ha, hb, hab⟩ | ht⟩ := h u r' hp
  · exact ⟨hacc, plain_selfClash_tied hacc ha hb hab⟩
  · exact ⟨hacc, ht⟩

open Classical in
/-- **A simulator of each prover at `wrP`'s verdict**: for each prover `P` and tape `ω`, a live prover `P'` with a
finite tape `τ` and coins `Φ` of `V false` at the tape and the live coins, of uniform law, whose strategies send `P`'s
roots, that simulates `P`'s accepting runs at `Φ` (`PlainSimRun`), whose live self-clashes give `P`'s accepting runs
with a collision there (`PlainClashRun`), and whose strategies' fork bound at each plain fork round `j` is no more than
an accepting fork's with a collision there (`plainForkEv`) and the slack at chances `δ` (`Flock.forkSlack`), at `δ`
fixed per prover whose slack weights total at most level 0's `2·2^{d₀}·Q₀` (`Flock.slackC`, `Flock.slackW`). The
deployed coins as the tape and `plainCoins` as `Φ` (`Pr_mergeAt`) are one such choice; uniform functions of the live
coins as the tape (`fnAt`, `Pr_fnAt`) are another. -/
def PlainSimWrPAt {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdict I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (δ : ℕ → ℝ) (_ : ∀ t, 0 ≤ δ t)
      (_ : ∑ i, Claims.Flock.slackC ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (fun r ℓ (i : Fin ((V false).k x)) =>
            Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val) i ≤
        2 * ((2 : ℝ) ^ l₀.logLen * l₀.queries)),
    ∀ ω : Ω, ∃ (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 spec.mPts) (τ : Type)
      (_ : Fintype τ) (_ : Nonempty τ) (P' : Claims.Prover (Ω × τ))
      (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin)
      (_ : ∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E)
      (hσ : ∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω),
      PlainSimRun P x ω spec.mPts N dg w P' Φ ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
        PlainClashRun P x ω spec.mPts N dg w P' Φ
          (LiveSelfClash Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (plainVerdict I d C (plainRoot P x ω)) opn
            (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
            (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
              ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))) ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) → ∀ r ℓ,
        (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
            k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
            (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
              (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdict I d C (plainRoot P x ω)) opn dg
              (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
              (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
              (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))) /
            Fintype.card τ ≤
          (Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) =>
            plainForkEv h hsch P r ℓ ω s.1 s.2) +
            Claims.Flock.forkSlack sha512 (V false) P x δ (plainJ (tables_head_one h) hsch r ℓ).val ω)

open Classical in
/-- **A simulator gives the plain table bound at a statement**, one that sets up at `ins` and the draw `d`, whose
provers' accepting runs carry `d` verbatim, and none of whose full-width tables that does not extract is committed to
table 0's model statement: the escapes are the slack's (`slackEsc`, fitting the weight by `slackEsc_fitsFork_plain`),
the first bound is `plain_first_of_sim`'s, the third `liveMix_le_Pr`'s, and the fork bound with its slack is at most
the slack's events at their weights (`forkSlack_le`), so the mean `ε_c⁻` is at most `ε_table` and their term
(`mean_epsSL_le_slack`, `plain_le_sessionsE`). -/
theorem plain_tableBoundθAt_of_sim {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdict ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j}
    (hset : setupOf false x = some (ins, spec, sts)) (hd : ins.read.draw = some d)
    (hdraw : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω), RecordDrawVerbatim P x ω)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hs : PlainSimWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  intro Ω _ _ P
  obtain ⟨δ, hδ, hfit, hω⟩ := hs Ω P
  refine ⟨Claims.Flock.slackEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
      (fun r ℓ ω s s' => plainForkEv h hsch P r ℓ ω s s') (plainEv₁ P x) (fun _ _ _ _ _ he => he.2.2)
      (fun _ _ he => he.2)
      (fun r ℓ i => Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val)
      (Claims.Flock.slackG sha512 (V false) P x) (Claims.Flock.slackG_tied sha512 (V false) P x),
    slackEsc_fitsFork_plain Readable.relation P x (tables_head_one h) hsch _ _ _ _ _ _ _ hfit, fun ω C hok hg => ?_⟩
  obtain ⟨N, dg, w, τ, _, _, P', Φ, hΦ, hσ, hrun, hsc, hfork⟩ := hω ω
  have htr := plain_first_of_sim P x ω C spec.mPts hset hd (hdraw Ω P ω) N dg w P' Φ ⟨hΦ, hrun⟩
  have hsc' := liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x ω _ (plainEv₁ P x ω) Φ (hΦ _)
    (hsc C hok)
  exact plain_le_sessionsE L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay P _ ω C (hnc C hok hg) τ
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdict ins.read.I d C (plainRoot P x ω)) opn dg
      (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdict ins.read.I d C (plainRoot P x ω)) opn dg
      (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ))
    (mean_epsSL_le_slack execArith Refine.H512 Refine.enc512 Refine.leaf512 h L m k0 l₀ l₁ rest ext hsch hk0 hm
      PT_LOCAL [] [] _ _ sha512 (V false) P (plainJ (tables_head_one h) hsch) _ _ _ _ _
      (fun _ _ _ => Claims.Flock.slackW_nonneg hδ _ _ _) _ _ ω
      (fun r ℓ => (hfork C hok r ℓ).trans (add_le_add le_rfl
        (Claims.Flock.forkSlack_le sha512 (V false) P x hδ _ ω)))
      (le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leaf512
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdict ins.read.I d C (plainRoot P x ω)) opn dg
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ) hsc'))

/-- **A simulator at a plain statement**: `PlainSimWrPAt` at the statements in scope that set up at a draw, at
`plainOpn` where the outputs are public and at `plainOpnH` where they are hidden under `hm96-sha512/v1` leaves, and
`PlainLiveAccAt` where they are hidden under any other leaf the scope admits, as `PlainLiveW` asks. -/
def PlainSimW (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false,
      PlainSimWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdec hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay)) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"),
      PlainSimWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecH hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay)) ∧
    (∀ (_ : ins.read.I.tags.hiddenOutputs = true) (_ : ins.read.I.tags.merkleLeaf ≠ "hm96-sha512/v1"),
      PlainLiveAccAt (wide false) h L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay)

/-- **A simulator gives the plain table bound at a statement some run accepts**: it parses in scope and sets up one
table at a draw (`acc_inScope`, `acc_draw`, `plain_one`), and a full-width table that does not extract is not
committed to table 0's model statement (`plain_not_committed`), so `plain_tableBoundθAt_of_sim` at `plainOpn` or
`plainOpnH`, and `plain_tableBoundθAt_of_live` where the outputs are hidden under another leaf. -/
theorem plain_tableBoundθAt_sim {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V false) x)
    (hdraw : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω), RecordDrawVerbatim P x ω)
    (hs : PlainSimW x) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  obtain ⟨s, hs', hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := acc_draw hno
  have h1 := plain_one hs' hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs' hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs' hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs' hin hset h0 L C hw hg
    rwa [e] at this
  obtain ⟨hw, hhid, hplain⟩ := hs s hs' hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact plain_tableBoundθAt_of_sim hset hd hdraw hnc (hw hh)
  · by_cases hleaf : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · exact plain_tableBoundθAt_of_sim hset hd hdraw hnc (hhid hh hleaf)
    · exact plain_tableBoundθAt_of_live Readable.relation (wide false) ht L _ k0 l₀ l₁ rest ext hsch hk0 hm13 PT_LOCAL
        hlay hnc (plainLiveAt_of_acc (hplain hh hleaf))

/-- **The deployed plain verifier: KS**, at SHA-512 at the trimmed record at the deployed relation, at the statements
some run accepts from a simulator of each prover (`PlainSimW`) and the record's draw verbatim (`RecordDrawVerbatim`).
A prover whose record carries the draw with its keys in another order still passes the checks (`drawEq_ne`), so
`hdraw` holds only where the verifier reads the record's draw as the statement's term. -/
theorem plain_KS_of_sim
    (hdraw : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω), RecordDrawVerbatim P x ω)
    (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainSimW x) :
    Claims.Flock.KS sha512 (deployPlainTrim Readable.relation) :=
  plain_KS_trim Readable.relation (plain_tableBoundθ_of_acc Readable.relation (wide false)
    fun x _ _ _ _ hno _ _ _ => plain_tableBoundθAt_sim hno (hdraw x hno) (hs x hno))

end FlockDeploy.KS
