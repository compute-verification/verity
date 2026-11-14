module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Flock.SpotFlock.Honest
public import Proofs.Flock.Claims.Recursion.DeploySessions
public import Proofs.Core.Game.Tape

/-!
# The form's accept is the run's accept

`AcceptsRun` and `StmtAt` (`Projection.lean`) at the two-stage form's reading (`formReading`) and acceptance
(`FormAccepts`):

* **The statement** (`stmtAt`): on a play of the frame the form's check accepts, the run's statement the check reads
  is the boundary registration's record, the one the commitment `c` carries (`stmtOf`). The run of record's record is
  the commitment's as the verifier's parser reads its canonical text (`readBack`), with the same canonical bytes, and
  the instance's case reads the play's commitment (`CaseCommits`), which a play of the game at `c` ends with
  (`Binding.play_reaches`).
* **Same coins, same messages, same outcome** (`out_eq_of_msgAt`): two runs of one verifier on one coin vector whose
  provers send the same messages decide alike, whatever else the provers do. So `AcceptsRun` holds where, at each
  accepted play and each coin vector agreeing with what the check read, some prover sending the check's messages is
  accepted (`ProverAt`, `acceptsRun_of_proverAt`).
* **The accept at the form** (`AcceptsRunOf`): `SpotFlock.proverSeq` at the run of record's records, the session
  statements and the sessions' provers is that prover. Its premises, each a named `Prop`:
  * `PlayAccepts`: at an accepted play, on every round-0 coin and second-stage coins that start with the bytes the run
    read and every sessions' coins that agree with what the check read, `SpotFlock.V` accepts `proverSeq`, and each
    session statement has its template's rounds. `SpotRunAcceptsPlay` with its premises (the held copies
    matching the run, round 2, the sessions) gives it;
  * `SessionMsgs`: the messages `proverSeq` sends in each session's rounds are those the check read;
  * `SourcesFit`: an accepted run read at most one coin's bytes in the first stage and `stage` coins' in the second;
  * `ReadsOnly`: `SpotFlock.V` reads a coin only at a round `readsAt` names;
  * `CoinsCover`: every byte string of one coin's length, and of `stage` coins', is some coins' bytes, so the coins
    `coinOf` and `stageCoinsOf` choose are the run's bytes padded with zeros.

Where the check accepts, the earlier-period signature hole of the leg stands: the device's history in the leg's game
holds one play's signatures, so these statements speak of an audit on a key and ledger made for it.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The statement -/

/-- **The statement a commitment's bytes carry**: the boundary registration's record, its canonical JSON's bytes; none
where the bytes don't decode as a commitment or the record has no canonical form. -/
def stmtOf (c : ByteArray) : Option Msg :=
  match Spot.OneStage.commitmentOf c with
  | .ok cm => recordMsg cm.record
  | .error _ => none

/-- **The instance's case reads the play's commitment**: every case `C` decodes of a play carries the play's
commitment bytes. The instance's composite owes it. -/
def CaseCommits (C : CaseOf) : Prop := ∀ o k, C o = some k → k.commitment = o.commitment

/-- **A play of the frame ends at the commitment `c`.** -/
theorem frame_commitment {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (ω : TapeOf V θ F c σ) (r : Coins A x) :
    (Fr.out ω r).commitment = c :=
  (Verity.Sampled.Binding.play_reaches (Verity.Game.Tape.reaches_out (V.game θ F c) σ (Fr.place ω r))).2.2

/-- **An accepted case's run of record**: the case's commitment decodes, the check handed `Spot.Run.run` an input, that
input's record is the commitment's as the verifier's parser reads its canonical text (`readBack`), and it has an
interior whose record has a canonical form. -/
theorem input_of_accepts {k : Case} (h : AcceptsOn k) :
    ∃ cm i, Spot.OneStage.commitmentOf k.commitment = .ok cm ∧ inputOfCase k = some i ∧
      VStar.Recursive.SampledTwo.readBack cm.record = .ok i.record ∧
      ∃ it m₁, i.interior = some it ∧ recordMsg it.record = some m₁ := by
  obtain ⟨p, cm, e, pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := h
  obtain ⟨-, c₂, ξ₂, r, i, run, hi, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, hj₁, hj₂, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hi
  refine ⟨cm, i, h2, ?_, by rw [hieq]; exact hj₁, ?_⟩
  · unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hi]
    rfl
  · subst hieq
    obtain ⟨s, -, -, hs⟩ := VStar.Recursive.SampledTwo.readBack_ok hj₂
    refine ⟨_, s.toUTF8.data.toList, rfl, ?_⟩
    show recordMsg j₂ = _
    simp only [recordMsg, hs]
    rfl

/-- **The frame's statement is the one the check reads** (`StmtAt` at the form): on every play of the frame the form's
check accepts, the run's statement it reads is the boundary registration's record the commitment `c` carries, where
`x` is that record's bytes and the instance's case reads the play's commitment. -/
theorem stmtAt (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (hC : CaseCommits C) (hx : stmtOf c = some x) :
    StmtAt V A x θ F c σ Fr (formReading A C S dev) (FormAccepts C) := by
  intro ω r hacc
  obtain ⟨k, hk, hA⟩ := hacc
  obtain ⟨cm, i, hcm, hi, hrb, -⟩ := input_of_accepts hA
  have hrec : recordMsg i.record = recordMsg cm.record := by
    obtain ⟨s, hc, -, hc'⟩ := VStar.Recursive.SampledTwo.readBack_ok hrb
    unfold recordMsg
    rw [hc, hc']
  have hkc : k.commitment = c := (hC _ k hk).trans (frame_commitment Fr ω r)
  rw [hkc] at hcm
  show (inputOn C (Fr.out ω r)).bind (fun i => recordMsg i.record) = some x
  rw [← hx]
  unfold inputOn stmtOf
  rw [hk, hcm]
  simp only [Option.bind, hi, hrec]

/-! ## Same coins, same messages, same outcome -/

/-- **A round's message is the prover's next one** on the history so far. -/
theorem stepWith_at {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) (f : ℕ)
    (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript) :
    ((stepWith H P x ω vn (f + 1) hist as t).msgs[t.msgs.length]?).map Prod.fst =
      some ((P.next x hist ω).eval H).1 := by
  rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
  cases o with
  | msg m =>
    rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb]
    obtain ⟨rest, hrest⟩ := (RecDeploy.stepWith_mono H P x ω vn f _ _ _).1
    rw [← hrest]
    simp
  | acc =>
    rw [RecDeploy.stepWith_succ_acc H P x ω vn f hist as t hvb]
    simp
  | rej =>
    rw [RecDeploy.stepWith_succ_rej H P x ω vn f hist as t hvb]
    simp

/-- **Rounds whose provers send the same messages go alike**: from one history and messages, against one verifier,
two provers that send the same message at every round have the same messages and answers and the same decision. -/
theorem stepWith_out_of_msgs {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω') (x : Msg) (ω : Ω)
    (ω' : Ω') (vn : List Msg → OC Out) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Transcript), t.msgs = t'.msgs →
      (∀ i < t.msgs.length + f, ((stepWith H P x ω vn f hist as t).msgs[i]?).map Prod.fst =
        ((stepWith H P' x ω' vn f hist as t').msgs[i]?).map Prod.fst) →
      (stepWith H P x ω vn f hist as t).msgs = (stepWith H P' x ω' vn f hist as t').msgs ∧
        (stepWith H P x ω vn f hist as t).out = (stepWith H P' x ω' vn f hist as t').out
  | 0, _, _, t, t', ht, _ => by simp [stepWith, ht]
  | f + 1, hist, as, t, t', ht, hm => by
    have hpa : ((P.next x hist ω).eval H).1 = ((P'.next x hist ω').eval H).1 := by
      have h0 := hm t.msgs.length (by omega)
      rw [stepWith_at, ht, stepWith_at] at h0
      exact Option.some.inj h0
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    have hvb' : (vn (as ++ [((P'.next x hist ω').eval H).1])).eval H = (o, lo) := by
      rw [← hpa]
      exact hvb
    cases o with
    | msg m =>
      rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb,
        RecDeploy.stepWith_succ_msg H P' x ω' vn f hist as t' hvb'] at hm ⊢
      rw [← hpa] at hm ⊢
      refine stepWith_out_of_msgs H P P' x ω ω' vn f _ _ _ _ (by simp only [ht]) fun i hi => hm i ?_
      simp only [List.length_append, List.length_singleton] at hi
      omega
    | acc =>
      rw [RecDeploy.stepWith_succ_acc H P x ω vn f hist as t hvb,
        RecDeploy.stepWith_succ_acc H P' x ω' vn f hist as t' hvb']
      simp [ht, hpa]
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H P x ω vn f hist as t hvb,
        RecDeploy.stepWith_succ_rej H P' x ω' vn f hist as t' hvb']
      simp [ht, hpa]

/-- **Same coins, same messages, same outcome**: two runs of one verifier at one statement and on one coin vector,
whose provers send the same message at every round, decide alike. -/
theorem out_eq_of_msgAt {Ω Ω' Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω')
    (V : Verifier Coin) (x : Msg) (ω : Ω) (ω' : Ω') (r : Fin (V.k x) → Coin)
    (hm : ∀ i < V.k x, msgAt (run H P V x ω r) i = msgAt (run H P' V x ω' r) i) :
    (run H P V x ω r).out = (run H P' V x ω' r).out :=
  (stepWith_out_of_msgs H P P' x ω ω' _ (V.k x) [] [] ⟨[], [], [], .rej⟩ ⟨[], [], [], .rej⟩ rfl
    fun i hi => hm i (by simpa using hi)).2

/-- **Some prover sending the check's messages is accepted** at every accepted play and every coin vector agreeing
with what the check read. -/
def ProverAt (A : Live₂) (x : Msg) (Rd : Reading A) (Acc : Verity.Exchange → Prop) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Rd.stmt o = some x → ∀ r : Coins A x,
    (∀ i : Fin (A.L.V.k x), Rd.reads i.val → ∀ c, Rd.coin o i.val = some c → Rd.agree o i.val c (r i)) →
    ∃ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω), (Claims.run FlockDeploy.sha512 P A.L.V x ω r).out = .acc ∧
      ∀ i < A.L.V.k x, Rd.msg o i = msgAt (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i

/-- **`AcceptsRun` from one accepted prover**: every prover sending the check's messages decides as the one
`ProverAt` gives (`out_eq_of_msgAt`). -/
theorem acceptsRun_of_proverAt {A : Live₂} {x : Msg} {Rd : Reading A} {Acc : Verity.Exchange → Prop}
    (h : ProverAt A x Rd Acc) : AcceptsRun A x Rd Acc := by
  intro o hA hs Ω P ω r hc hm
  obtain ⟨Ω', P', ω', hacc, hm'⟩ := h o hA hs r hc
  rw [out_eq_of_msgAt FlockDeploy.sha512 P P' A.L.V x ω ω' r fun i hi => (hm i hi).symm.trans (hm' i hi)]
  exact hacc

/-! ## The accept at the form -/

/-- **The session's verifier is SPOT[Flock]'s at the held copies.** -/
theorem L_V (A : Live₂) : A.L.V = SpotFlock.V A.h := rfl

/-- **Every coins' byte string is some coins' bytes**: one coin's length of bytes is a coin's `src`, and `stage` coins'
length is the second stage's coins' `stageSrc`. So `coinOf` and `stageCoinsOf` choose coins whose bytes are the run's
padded with zeros. -/
def CoinsCover (h : SpotFlock.Held) : Prop :=
  (∀ b : ByteArray, b.size = SpotFlock.L → ∃ c, SpotFlock.src c = b) ∧
    ∀ b : ByteArray, b.size = h.stage * SpotFlock.L → ∃ cs : SpotFlock.Stage h, SpotFlock.stageSrc cs = b

/-- **An accepted run read at most the coins it has**: at a play the form's check accepts, the run of record's first
stage read at most one coin's bytes and its second stage at most `stage` coins'. -/
def SourcesFit (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    i.source.size ≤ SpotFlock.L ∧ it.source.size ≤ A.h.stage * SpotFlock.L

/-- **SPOT[Flock] reads a coin only where `readsAt` says**: two runs of `SpotFlock.V h` with one prover, at one
statement, on coins equal at every round `readsAt` names, are one run. -/
def ReadsOnly (h : SpotFlock.Held) : Prop :=
  ∀ (H : Msg → Msg) (Ω : Type) (P : Claims.Prover Ω) (x : Msg) (ω : Ω)
    (r r' : Fin ((SpotFlock.V h).k x) → FlockDeploy.Coin),
    (∀ n : Fin ((SpotFlock.V h).k x), readsAt h n.val → r n = r' n) →
    Claims.run H P (SpotFlock.V h) x ω r = Claims.run H P (SpotFlock.V h) x ω r'

/-- **The coins `proverSeq`'s run takes**, from the run of record's bytes and the sessions' coins: round 0's coin
starts with the first stage's bytes, the second stage's coins with the interior's, and each session's coins agree with
what the check read of them. -/
def SeqCoinsRead (A : Live₂) (S : SessionReading) (o : Verity.Exchange) (i : Spot.Run.Input)
    (it : Spot.Run.Interior) (m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : SpotFlock.Stage A.h)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) → FlockDeploy.Coin) : Prop :=
  (∃ u, SpotFlock.src c₀ = i.source ++ u) ∧ (∃ u, SpotFlock.stageSrc c₁ = it.source ++ u) ∧
    ∀ t < A.h.ms.length, ∀ j (hj : j < (FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) c,
      S.coin o t j = some c → S.agree o t j c (r' t ⟨j, hj⟩)

/-- **`proverSeq`'s run at an accepted play's records**: the run's statement `x`, round 1's message `m₁` and the session
statements' message `m₂`, the sessions' provers `P' o`, and round 0's, the second stage's and the sessions' coins. -/
noncomputable def seqRun (A : Live₂) (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (o : Verity.Exchange)
    (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : SpotFlock.Stage A.h)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) → FlockDeploy.Coin) : Claims.Transcript :=
  Claims.run FlockDeploy.sha512 (SpotFlock.proverSeq A.h [] m₁ m₂ (P' o)) (SpotFlock.V A.h) x (fun _ => ())
    (SpotFlock.seqCoins A.h (x := x) c₀ c₁ r')

/-- **An accepted play's run of record is a play SPOT[Flock] accepts** (`SpotRunAcceptsPlay` with its
premises, at the form's records): at a play the form's check accepts, with the run of record's input `i`, its interior
`it` and the records' bytes `x` and `m₁`, the check read a session statements' message `m₂`, each session statement
has its template's rounds, and on every coins `SeqCoinsRead` describes `SpotFlock.V` accepts `proverSeq` with the
sessions' provers `P' o`. -/
def PlayAccepts (A : Live₂) (C : CaseOf) (S : SessionReading) (P' : Verity.Exchange → ℕ → Claims.Prover Unit) :
    Prop :=
  ∀ o i it x m₁, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → ∃ m₂, S.stmts o = some m₂ ∧
      (∀ t < A.h.ms.length, (FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t) = SpotFlock.k₀ A.h t) ∧
      ∀ c₀ c₁ r', SeqCoinsRead A S o i it m₂ c₀ c₁ r' → (seqRun A P' o x m₁ m₂ c₀ c₁ r').out = .acc

/-- **The sessions' messages are the ones the check read**: at a play the form's check accepts, on the coins
`SeqCoinsRead` describes, `proverSeq`'s message at round `j` of template `t`'s session is the session's message there as
the check read it. -/
def SessionMsgs (A : Live₂) (C : CaseOf) (S : SessionReading) (P' : Verity.Exchange → ℕ → Claims.Prover Unit) :
    Prop :=
  ∀ o i it x m₁ m₂, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → S.stmts o = some m₂ →
    ∀ c₀ c₁ r', SeqCoinsRead A S o i it m₂ c₀ c₁ r' →
      ∀ t < A.h.ms.length, ∀ j < SpotFlock.k₀ A.h t,
        msgAt (seqRun A P' o x m₁ m₂ c₀ c₁ r') (SpotFlock.off A.h t + j) = S.msg o t j

/-- **The form's accept is the run's accept** (`AcceptsRun` at the form; its theorem is `acceptsRun`): at the sessions'
provers `P'`, where every coins' byte string is some coins' (`CoinsCover`), an accepted run read at most the coins it
has (`SourcesFit`), its run of record is a play SPOT[Flock] accepts (`PlayAccepts`) whose sessions' messages are those
the check read (`SessionMsgs`), and SPOT[Flock] reads a coin only where `readsAt` says (`ReadsOnly`), every run whose
coins agree with what the form's check read and whose messages are what it read accepts. -/
def AcceptsRunOf (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) : Prop :=
  ∀ P' : Verity.Exchange → ℕ → Claims.Prover Unit,
    CoinsCover A.h → SourcesFit A C → PlayAccepts A C S P' → SessionMsgs A C S P' → ReadsOnly A.h →
      AcceptsRun A x (formReading A C S dev) (FormAccepts C)

end Verity.SampledTwo
