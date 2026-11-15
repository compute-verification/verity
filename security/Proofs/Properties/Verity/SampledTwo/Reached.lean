module

public import Proofs.Properties.Verity.SampledTwo.ReadsOnly

/-!
# A run whose messages are the check's runs to its end

`AcceptsRun` asks that a run's message be the check's at each of the verifier's `k` messages. A run that stops early
has fewer than `k`, so a piece that fixes the run's messages can speak only of the messages the run has. This file
closes that gap at the form:

* **Messages alike where the first run has one** (`stepWith_reached`, `run_eq_of_reached`): two runs of one verifier
  on one coin vector, where the second prover sends the first's message wherever the first run has one, have the same
  messages and the same decision. A prover sends its message before the verifier decides, so the first place the two
  could differ is one where both have a message.
* **The accept, and every message** (`ReachedAccepts`, `reached_of_proverAt`): at a play the check accepts and a coin
  vector agreeing with what the check read, a run whose messages are the check's wherever it has one accepts and has
  all `k` messages. It is the run of the accepted prover `ProverAt` gives, and an accepted run of SPOT[Flock] has all
  its messages (`length_of_acc`).
* **At the form** (`reachedAccepts`), under `acceptsRun_of'`'s premises, and **at the frame** (`reaches_frame`): on a
  play of the frame the check accepts, with the run's coins agreeing with what the check read, the frame's run has all
  `k` messages where its messages are the check's wherever it has one.

The agreement of the coins is a premise throughout. Without it the statement is false: on other coins the verifier
refuses the session statements' message and the run stops there, its messages so far being the check's.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **Runs alike where the first has a message**: from one history and messages, against one verifier, where the
second prover sends the first's message wherever the first run has one, the two runs have the same messages and the
same decision. -/
theorem stepWith_reached {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω') (x : Msg) (ω : Ω)
    (ω' : Ω') (vn : List Msg → OC Out) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Transcript), t.msgs = t'.msgs →
      (∀ i < t.msgs.length + f, i < (stepWith H P x ω vn f hist as t).msgs.length →
        ((stepWith H P x ω vn f hist as t).msgs[i]?).map Prod.fst =
          ((stepWith H P' x ω' vn f hist as t').msgs[i]?).map Prod.fst) →
      (stepWith H P x ω vn f hist as t).msgs = (stepWith H P' x ω' vn f hist as t').msgs ∧
        (stepWith H P x ω vn f hist as t).out = (stepWith H P' x ω' vn f hist as t').out
  | 0, _, _, t, t', ht, _ => by simp [stepWith, ht]
  | f + 1, hist, as, t, t', ht, hm => by
    have hlt : t.msgs.length < (stepWith H P x ω vn (f + 1) hist as t).msgs.length := by
      by_contra hge
      have hat := stepWith_at H P x ω vn f hist as t
      rw [List.getElem?_eq_none (Nat.le_of_not_lt hge)] at hat
      simp at hat
    have hpa : ((P.next x hist ω).eval H).1 = ((P'.next x hist ω').eval H).1 := by
      have h0 := hm t.msgs.length (by omega) hlt
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
      refine stepWith_reached H P P' x ω ω' vn f _ _ _ _ (by simp only [ht]) fun i hi hl => hm i ?_ hl
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

/-- **Same coins, same messages where the first run has one, same run**: two runs of one verifier at one statement and
on one coin vector, where the second prover sends the first's message wherever the first run has one, have the same
messages and decide alike. -/
theorem run_eq_of_reached {Ω Ω' Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω')
    (V : Verifier Coin) (x : Msg) (ω : Ω) (ω' : Ω') (r : Fin (V.k x) → Coin)
    (hm : ∀ i < V.k x, i < (run H P V x ω r).msgs.length →
      msgAt (run H P V x ω r) i = msgAt (run H P' V x ω' r) i) :
    (run H P V x ω r).msgs = (run H P' V x ω' r).msgs ∧ (run H P V x ω r).out = (run H P' V x ω' r).out :=
  stepWith_reached H P P' x ω ω' _ (V.k x) [] [] ⟨[], [], [], .rej⟩ ⟨[], [], [], .rej⟩ rfl
    fun i hi hl => hm i (by simpa using hi) hl

/-- **The accept, and every message**: at every play `Acc` accepts whose statement is `x`, every run on coins agreeing
with what the check read, whose messages are what the check read wherever the run has one, accepts and has all the
verifier's `k` messages. `AcceptsRun` with the messages asked only where the run has one, and the run's length beside
the accept. -/
def ReachedAccepts (A : Live₂) (x : Msg) (Rd : Reading A) (Acc : Verity.Exchange → Prop) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Rd.stmt o = some x →
    ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x),
      (∀ i : Fin (A.L.V.k x), Rd.reads i.val → ∀ c, Rd.coin o i.val = some c → Rd.agree o i.val c (r i)) →
      (∀ i < (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length,
        Rd.msg o i = msgAt (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) →
      (Claims.run FlockDeploy.sha512 P A.L.V x ω r).out = .acc ∧
        A.L.V.k x ≤ (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length

/-- **`ReachedAccepts` from one accepted prover**: a run whose messages are the check's where it has one is the run of
the prover `ProverAt` gives (`run_eq_of_reached`), which accepts, and an accepted run of SPOT[Flock] has all its
messages (`length_of_acc`). -/
theorem reached_of_proverAt {A : Live₂} {x : Msg} {Rd : Reading A} {Acc : Verity.Exchange → Prop}
    (h : ProverAt A x Rd Acc) : ReachedAccepts A x Rd Acc := by
  intro o hA hs Ω P ω r hc hm
  obtain ⟨Ω', P', ω', hacc, hm'⟩ := h o hA hs r hc
  obtain ⟨hmsgs, hout⟩ := run_eq_of_reached FlockDeploy.sha512 P P' A.L.V x ω ω' r
    fun i hi hl => (hm i hl).symm.trans (hm' i hi)
  have hlen : (Claims.run FlockDeploy.sha512 P' (SpotFlock.V A.h) x ω' r).msgs.length =
      SpotFlock.off A.h A.h.ms.length + 1 := length_of_acc A.h FlockDeploy.sha512 P' x ω' r hacc
  have hlen' : (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length = SpotFlock.off A.h A.h.ms.length + 1 :=
    (congrArg List.length hmsgs).trans hlen
  refine ⟨hout.trans hacc, ?_⟩
  show SpotFlock.off A.h A.h.ms.length + 1 ≤ _
  exact Nat.le_of_eq hlen'.symm

/-- **The accept and every message, at the form**: `ReachedAccepts` at the form's reading and acceptance, under
`acceptsRun_of'`'s premises. -/
theorem reachedAccepts (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hfit : SourcesFit A C) (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) :=
  reached_of_proverAt (proverAt A x C S dev P' (coinsCover A.h) hfit hplay hsess (readsOnly A.h))

/-- **The frame's run has every message**: on a play of the frame the check accepts, where the run's coins agree with
what the check read and the run's messages are what the check read wherever the run has one, the run has all the
verifier's `k` messages. -/
theorem reaches_frame {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) {Rd : Reading A}
    {Acc : Verity.Exchange → Prop} (hst : StmtAt V A x θ F c σ Fr Rd Acc) (h : ReachedAccepts A x Rd Acc)
    (ω : TapeOf V θ F c σ) (r : Coins A x) (hA : Acc (Fr.out ω r))
    (hc : ∀ i : Fin (A.L.V.k x), Rd.reads i.val → ∀ c', Rd.coin (Fr.out ω r) i.val = some c' →
      Rd.agree (Fr.out ω r) i.val c' (r i))
    (hm : ∀ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length,
      Rd.msg (Fr.out ω r) i = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r) i) :
    A.L.V.k x ≤ (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length :=
  (h (Fr.out ω r) hA (hst ω r hA) _ Fr.P ω r hc hm).2

end Verity.SampledTwo
