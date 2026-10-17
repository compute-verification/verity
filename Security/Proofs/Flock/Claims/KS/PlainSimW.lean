import Proofs.Flock.Claims.KS.PlainThird

/-!
# The plain chain from a simulator

`PlainLiveWrPAt` asks, for each prover `P`, for events tied to collisions and one mixture of live strategies with three
bounds. With the events an accepting run or fork with its collision (`plainForkEv`, `plainEv₁`) and the mixture the
strategies of a live prover `P'` with the deployed coins as its tape, two of the three bounds come from deterministic
facts about `P'` (`PlainSimWrPAt`):

* the first, from `P'` simulating `P`'s accepting runs at `plainCoins` (`PlainSimRun`) and the record's draw
  (`RecordDrawVerbatim`), by `plain_first_of_run`;
* the third, from a live self-clash at `P'`'s outcome giving an accepting run of `P` at `plainCoins` with a collision
  among the verifier's calls (`PlainClashRun`), by `plainMix_le_of_run`;
* the second, the fork bound at `P'`'s strategies, is asked as it is.

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
outcome of `P'` at the tape `(ω, u)` and the live coins `r'`, `V false` accepts `P` at `plainCoins (u, r')` with a
collision among its calls (`plainEv₁`). -/
def PlainClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts)
    (P' : Claims.Prover (Ω × (Fin ((V false).k x) → Coin))) (hk : N x + 2 = (V false).k x)
    (n : Fin ((V false).k x) → ℕ) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    plainEv₁ P x ω (plainCoins hk n (u, r'))

/-- **A live event at `P'`'s outcome gives an accepting run of `P` whose reps open two different columns at one
position, or a collision**: `PlainClashRun` with a self-clash at level 0 asked without the hashes (`plainRawCols`).
`SelfClashL` also counts clashes at later levels, against the caps of the reps' pendings; those still need their
collision. -/
def PlainRawClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts)
    (P' : Claims.Prover (Ω × (Fin ((V false).k x) → Coin))) (hk : N x + 2 = (V false).k x)
    (n : Fin ((V false).k x) → ℕ) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (plainCoins hk n (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List 𝔽),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (plainCoins hk n (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (plainCoins hk n (u, r'))).seen ∧ a ≠ b) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω (plainCoins hk n (u, r'))).vLog)

/-- **The clash gives the collision**: two different columns of an accepting run at one position collide in its log
(`plain_selfClash_tied`), so `PlainRawClashRun` gives `PlainClashRun`. -/
theorem plainClashRun_of_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts}
    {P' : Claims.Prover (Ω × (Fin ((V false).k x) → Coin))} {hk : N x + 2 = (V false).k x}
    {n : Fin ((V false).k x) → ℕ} {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRun P x ω mPts N dg w P' hk n pL) :
    PlainClashRun P x ω mPts N dg w P' hk n pL := fun u r' hp => by
  obtain ⟨hacc, ⟨q, a, b, ha, hb, hab⟩ | ht⟩ := h u r' hp
  · exact ⟨hacc, plain_selfClash_tied hacc ha hb hab⟩
  · exact ⟨hacc, ht⟩

open Classical in
/-- **A simulator of each prover at `wrP`'s verdict**: for each prover `P` and tape `ω`, a live prover `P'` with the
deployed coins as its tape, whose strategies send `P`'s roots, that simulates `P`'s accepting runs at `plainCoins`
(`PlainSimRun`), whose live self-clashes give `P`'s accepting runs with a collision there (`PlainClashRun`), and whose
strategies' fork bound is no more than an accepting fork's with a collision (`plainForkEv`). -/
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
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω),
    ∃ (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 spec.mPts)
      (P' : Claims.Prover (Ω × (Fin ((V false).k x) → Coin))) (hk : N x + 2 = (V false).k x)
      (n : Fin ((V false).k x) → ℕ)
      (hσ : ∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω),
      PlainSimRun P x ω spec.mPts N dg w P' (plainCoins hk n) ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
        PlainClashRun P x ω spec.mPts N dg w P' hk n
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
            Fintype.card (Fin ((V false).k x) → Coin) ≤
          Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) =>
            plainForkEv h hsch P r ℓ ω s.1 s.2)

open Classical in
/-- **A simulator gives the live sessions' bound at `wrP`'s verdict**, at a statement that sets up at `ins` and the
draw `d` and whose provers' accepting runs carry `d` verbatim: the first bound by `plain_first_of_run`, the third by
`plainMix_le_of_run`, the second as `PlainSimWrPAt` gives it. -/
theorem plainLiveWrPAt_of_sim {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
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
    (hs : PlainSimWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec) :
    PlainLiveWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec := by
  intro Ω _ _ P
  refine ⟨fun r ℓ ω s s' => plainForkEv h hsch P r ℓ ω s s', plainEv₁ P x, fun _ _ _ _ _ he => he.2.2,
    fun _ _ he => he.2, fun _ _ _ _ _ he => ⟨he.1, he.2.1⟩, fun _ _ he => he.1, fun ω C hC => ?_⟩
  obtain ⟨N, dg, w, P', hk, n, hσ, hrun, hsc, hfork⟩ := hs Ω P ω
  exact ⟨dg, N x, plainRoot P x ω, Fin ((V false).k x) → Coin, inferInstance, inferInstance,
    fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u), hσ,
    plain_first_of_run P x ω C spec.mPts hset hd (hdraw Ω P ω) N dg w P' hk n hrun, hfork C hC,
    plainMix_le_of_run x ω spec.mPts N dg w P' hk n _ (plainEv₁ P x ω) (hsc C hC)⟩

/-- **A simulator at a plain statement**: `PlainSimWrPAt` at the statements in scope that set up at a draw, at
`plainOpn` where the outputs are public and at `plainOpnH` where they are hidden, as `PlainLiveW` asks. -/
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
    (∀ hh : ins.read.I.tags.hiddenOutputs = true,
      PlainSimWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecH hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay))

/-- **`PlainSimW` gives `PlainLiveW`** where every prover's accepting runs carry the statement's draw verbatim. -/
theorem plainLiveW_of_sim {x : Claims.Msg}
    (hdraw : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω), RecordDrawVerbatim P x ω)
    (hs : PlainSimW x) : PlainLiveW x := by
  intro s hs' hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay
  obtain ⟨hw, hhid⟩ := hs s hs' hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay
  exact ⟨fun hh => plainLiveWrPAt_of_sim hset hd hdraw (hw hh), fun hh => plainLiveWrPAt_of_sim hset hd hdraw (hhid hh)⟩

/-- **The deployed plain verifier: KS**, at SHA-512 at the trimmed record at the deployed relation, at the statements
some run accepts from a simulator of each prover (`PlainSimW`) and the record's draw verbatim (`RecordDrawVerbatim`).
A prover whose record carries the draw with its keys in another order still passes the checks (`drawEq_ne`), so
`hdraw` holds only where the verifier reads the record's draw as the statement's term. -/
theorem plain_KS_of_sim
    (hdraw : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω), RecordDrawVerbatim P x ω)
    (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainSimW x) :
    Claims.Flock.KS sha512 (deployPlainTrim Readable.relation) :=
  plain_KS_of_w fun x hno => plainLiveW_of_sim (hdraw x hno) (hs x hno)

end FlockDeploy.KS
