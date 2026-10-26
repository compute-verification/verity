import Proofs.Flock.Claims.KS.DecodeHidden

/-!
# The plain table bound at the deployed plain decoders

`PlainLiveAccAt` asks for decoders right at `plainTab` and a mixture of live strategies whose verdict at them bounds
the deployed runs. At a plain statement in scope with a draw, `wrP`'s decoders are right there, at public outputs
(`plain_hdec`) and at hidden ones (`plain_hdecH`), so what is left is the mixture at them:

* `checks_draw`, `acc_draw`: a statement some run accepts carries a draw, the checks after the verdict refusing a
  session without one.
* `plainVerdict`, `plainOpn`, `plainOpnH`, `plainDec`: `wrP`'s verdict, openings at public and at hidden outputs, and
  decoder, at the draw and no drawn unit.
* `PlainLiveWrPAt`: `PlainLiveAccAt` at `wrP`'s verdict and decoder and at openings whose decoders are right, and
  `plainLiveAccAt_of_wrP`, which gives it.
* `PlainLiveW`: `PlainLiveWrPAt` at `plainOpn` where the outputs are public and at `plainOpnH` where they are hidden
  under `hm96-sha512/v1` leaves, and `PlainLiveAccAt` itself where they are hidden under plain SHA-512 leaves, which the
  scope admits and the deployed decoders are not yet proved right at, at the statements in scope that set up at a
  draw; it gives `PlainLiveAcc` where some run accepts (`plainLiveAcc_of_w`), and KS at the trimmed record
  (`plain_KS_of_w`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The checks after the verdict refuse a session without a draw** (`drawLaw zk tags false none`). -/
theorem checks_draw {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : ∃ d, s.read.draw = some d := by
  unfold checks at h
  obtain ⟨_, -, h⟩ := Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := Refine.ebind_ok h
  obtain ⟨_, h, -⟩ := Refine.ebind_ok h
  cases hd : s.read.draw with
  | none => simp [hd, ProvedScope.drawLaw] at h
  | some d => exact ⟨d, rfl⟩


namespace Gen

variable [S : ZkSetup]

/-- **The checks after the verdict refuse a session without a draw** (`drawLaw zk tags false none`). -/
theorem checks_draw {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : ∃ d, s.read.draw = some d :=
  _root_.FlockDeploy.KS.checks_draw (checks_base h)

end Gen

namespace Gen

variable [S : ZkSetup]

theorem next_acc_draw {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((next zk x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts d, setupOf zk x = some (ins, spec, sts) ∧ ins.read.draw = some d := by
  cases ha : act zk x as cs with
  | rej => simp [next_rej sha512 ha] at h
  | coin c => simp [next_coin sha512 ha] at h
  | decide s spec sts link c₁ mids f =>
    obtain ⟨-, hc⟩ := (next_decide_sha512 ha).1 h
    obtain ⟨d, hd⟩ := checks_draw hc
    exact ⟨s, spec, sts, d, act_decide ha, hd⟩

end Gen

theorem next_acc_draw {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((next zk x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts d, setupOf zk x = some (ins, spec, sts) ∧ ins.read.draw = some d :=
  Gen.next_acc_draw (S := S₀) (zk := zk) (x := x) (as := as) (cs := cs) (h := h)

namespace Gen

variable [S : ZkSetup]

/-- **A statement some run accepts at SHA-512 sets up at a draw.** -/
theorem acc_draw {zk : Bool} {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) :
    ∃ ins spec sts d, setupOf zk x = some (ins, spec, sts) ∧ ins.read.draw = some d := by
  unfold Claims.Flock.NoAcc at hacc
  push Not at hacc
  obtain ⟨Ω, P, ω, r, hr⟩ := hacc
  obtain ⟨as, has⟩ := stepWith_acc _ _ _ _ _ _ _ _ _ hr
  exact next_acc_draw has

end Gen

/-- **A statement some run accepts at SHA-512 sets up at a draw.** -/
theorem acc_draw {zk : Bool} {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) :
    ∃ ins spec sts d, setupOf zk x = some (ins, spec, sts) ∧ ins.read.draw = some d :=
  Gen.acc_draw (S := S₀) (zk := zk) (x := x) (hacc := hacc)

open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain in
/-- **`wrP`'s verdict at the draw `d`**: flock-verify accepts what `wrP` writes for the outcome at `d` and no drawn
unit, the outcome's root is `rootB₀`'s, and the deployed reader looks up no column off `C` in it. -/
def plainVerdict (I : Exec.Inputs) (d : Json) {N mPts : ℕ} (C : Fin N → List 𝔽) (rootB₀ : Array ByteArray)
    (o : Tables.LiveOutJ 1 mPts) : Prop :=
  Exec.Accepts I (wrP I (fun _ : Finset (Fin 0) => d) (fun _ _ => 1) ∅ () o) ∧ o.rootB[0]! = rootB₀[0]! ∧
    TxNoOff I C (wrP I (fun _ : Finset (Fin 0) => d) (fun _ _ => 1) ∅ () o)

/-- **`wrP`'s openings at the table `T` and the draw `d`**: `DecodeZL.opnAt` at no drawn unit. -/
noncomputable def plainOpn {mPts : ℕ} (T : TabSpec mPts) (I : Exec.Inputs) (d : Json) (ps : Array ByteArray)
    (_ : Fin 1) : Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8) :=
  DecodeZL.opnAt (fun (_ : Finset (Fin 0)) (_ : Unit) => T) I (fun _ => d) ∅ () ps

/-- **`wrP`'s openings at hidden outputs**: `HiddenDec.opnAtH` at no drawn unit. -/
noncomputable def plainOpnH {mPts : ℕ} (T : TabSpec mPts) (I : Exec.Inputs) (d : Json) (ps : Array ByteArray)
    (_ : Fin 1) : Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8) :=
  HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) => T) I (fun _ => d) ∅ () ps

/-- **`wrP`'s decoder at the table `T`**: `DecodeZL.decAt` at no drawn unit. -/
noncomputable def plainDec {mPts : ℕ} (T : TabSpec mPts) (_ : Fin 1) : ℕ → ByteArray → List Msg :=
  DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) => T) DecodeZL.wOP DecodeZL.wLIG ∅ ()

open Classical Discharge.Tables in
/-- **The live sessions' bound at table 0, at the deployed plain decoders**: `PlainLiveAccAt` at full-width columns
with `wrP`'s verdict and decoder at the statement's inputs `I` and draw `d` (`plainVerdict`, `plainDec`) and openings
`opn` whose decoders are right at `plainTab` (`hdec`), so it asks for the mixture of live strategies and its bounds
only. -/
def PlainLiveWrPAt {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
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
      ∀ ω (C : Fin (wordLen false x) → List 𝔽), (∀ p, wide false x (C p)) →
        ∃ (dg : ByteArray → ByteArray) (N : ℕ) (rootB₀ : Array ByteArray) (τ : Type) (_ : Fintype τ)
          (_ : Nonempty τ)
          (σL : τ → Strategy
            (liveSession dg [plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].length spec.mPts N))
          (hσ : ∀ u, (σL u).1 = rootB₀),
          (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
              ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
                Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧
                  u ≠ C p) ≤
            (∑ u, (prob (plainVerdict I d C rootB₀) _ (σL u)).toReal) / Fintype.card τ ∧
          (∀ r ℓ, (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
              k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot rootB₀[0]!) r.val (ℓ.val + 1)
              (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
                (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdict I d C rootB₀) opn dg
                (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
                (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) N (hdec C rootB₀) rootB₀ (σL u) (hσ u))) /
                Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2) ∧
          (∑ u, (prob (LiveSelfClash Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
              (plainVerdict I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
              (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig)
                execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
            (liveSession dg [plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].length spec.mPts N)
              (σL u)).toReal) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s

open Classical Discharge.Tables in
/-- **`PlainLiveWrPAt` gives `PlainLiveAccAt`**, at `wrP`'s verdict and decoder and the openings `opn`, whose decoders
`hdec` says are right at `plainTab`. -/
theorem plainLiveAccAt_of_wrP {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} {h : tables false x = [(spec, st)]}
    {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level} {rest : List Accounting.Level} {ext : ℕ}
    {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩} {hk0 : l₀.lanes = k0}
    {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdict I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j}
    (hl : PlainLiveWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d opn hdec) :
    PlainLiveAccAt (wide false) h L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay := by
  intro Ω _ _ P
  obtain ⟨ev, ev₁, tied, tied₁, acc, acc₁, hω⟩ := hl Ω P
  refine ⟨ev, ev₁, tied, tied₁, acc, acc₁, fun ω C hok => ?_⟩
  obtain ⟨dg, N, rootB₀, τ, _, _, σL, hσ, htr, hadv, hsc⟩ := hω ω C hok
  exact ⟨plainVerdict I d C rootB₀, opn, dg, plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay),
    fun _ _ => true, fun _ _ _ => [], fun rs _ => capOfRoot rs[0]!, N, rootB₀, hdec C rootB₀, τ, inferInstance,
    inferInstance, σL, hσ, htr, hadv, hsc⟩

/-- **The live sessions' bound at a plain statement, at the deployed decoders**: at each statement in scope that sets
up at a draw, `PlainLiveWrPAt` at `wrP`'s openings, `plainOpn` where the outputs are public (`plain_hdec`) and
`plainOpnH` where they are hidden under `hm96-sha512/v1` leaves (`plain_hdecH`); where they are hidden under any other
leaf the scope admits (plain SHA-512 leaves), `PlainLiveAccAt` itself. -/
def PlainLiveW (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false,
      PlainLiveWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdec hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay)) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"),
      PlainLiveWrPAt h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecH hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay)) ∧
    (∀ (_ : ins.read.I.tags.hiddenOutputs = true) (_ : ins.read.I.tags.merkleLeaf ≠ "hm96-sha512/v1"),
      PlainLiveAccAt (wide false) h L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay)

/-- **`PlainLiveW` gives `PlainLiveAcc`** where some run accepts: the statement parses in scope (`acc_inScope`) and sets
up at a draw (`acc_draw`). -/
theorem plainLiveAcc_of_w {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V false) x) (hl : PlainLiveW x) :
    PlainLiveAcc x := by
  intro spec st h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay
  obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨ins, spec', sts, d, hset, hd⟩ := acc_draw hno
  obtain rfl : spec' = spec := by
    have hmem : (spec, st) ∈ (tables false x) := by rw [h]; exact List.mem_singleton_self _
    unfold tables at hmem
    rw [hset] at hmem
    obtain ⟨a, -, ha⟩ := List.mem_map.1 hmem
    exact (Prod.mk.inj ha).1
  obtain ⟨hw, hhid, hplain⟩ := hl s hs hin ins spec' sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact plainLiveAccAt_of_wrP (hw hh)
  · by_cases hleaf : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · exact plainLiveAccAt_of_wrP (hhid hh hleaf)
    · exact hplain hh hleaf

/-- **The deployed plain verifier: KS**, at SHA-512 at the trimmed record at the deployed relation, from the live
sessions' bound at the deployed decoders (`PlainLiveW`) at the statements some run accepts. -/
theorem plain_KS_of_w (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLiveW x) :
    Claims.Flock.KS sha512 (deployPlainTrim Readable.relation) :=
  plain_KS_of_liveAcc fun x hno => plainLiveAcc_of_w hno (hl x hno)

end FlockDeploy.KS
