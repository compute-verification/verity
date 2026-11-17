module

public import Proofs.Properties.Verity.SampledTwo.ReadsOnly
public import Proofs.Flock.SpotFlock.RunsAs.OfCheck
public import Proofs.Flock.SpotFlock.RunsAs.Top
public import Proofs.Flock.Claims.Sampling.Select.Prefix

/-! `PlayAccepts` from the running verifier's two theorems, with what neither decides as named premises;
these speak of the run the check made, not of the period its coins' signature is from. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)
open SpotFlock.RunsAs (AcceptedTwo HeldMatches RoundTwo SessionsAccept PlayCoins AgreesTo)
open SpotFlock.RunsAs.Round2 (NamesDistinct StmtsAre TwoHolds RecordsHeld RegistersNone RoundsHeld ScopeHeld PinHeld)

/-- **The parameters' rule is two-stage**: every case of a play decodes to parameters whose rule is two-stage. The
parameters' parser refuses any other rule. -/
def RuleIsTwo (C : CaseOf) : Prop :=
  ∀ o k p c e, C o = some k → Decoded k p c e → ∃ grain replay n, p.rule = .twoStage grain replay n

def SampledNoneAt (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it → SpotFlock.RunsAs.NoSampledOnly i it

def HeldAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it → HeldMatches i it A.h

def RecordsAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it x m₁, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → RecordsHeld A.h x m₁ ∧ RegistersNone it.record

def RecordsRead (C : CaseOf) : Prop :=
  ∀ o i it x m₁, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → SpotFlock.jsonMsg x = some i.record ∧ SpotFlock.jsonMsg m₁ = some it.record

def StmtsRead (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    ∃ m₂, S.stmts o = some m₂ ∧ StmtsAre A.h m₂ (it.statements.getD #[])

def RoundsHeldAt (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ → RoundsHeld A.h m₂

def ScopeHeldAt (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ → ScopeHeld A.h m₂

def PinHeldAt (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ → PinHeld A.h m₂

def SessionsRun (A : Live₂) (C : CaseOf) (S : SessionReading) (P' : Verity.Exchange → ℕ → Claims.Prover Unit) :
    Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ →
    ∀ r' : ∀ t, Fin ((FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) → FlockDeploy.Coin,
      (∀ t < A.h.ms.length, ∀ j (hj : j < (FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) c,
        S.coin o t j = some c → S.agree o t j c (r' t ⟨j, hj⟩)) →
      ∀ t < A.h.ms.length,
        (Claims.run FlockDeploy.sha512 (P' o t) (FlockDeploy.V true) (SpotFlock.stmtAt A.h m₂ t) () (r' t)).out =
          .acc

def ReadsWithin (h : SpotFlock.Held) : Prop :=
  (∀ ξ draw proved read, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query
      h.reg.first.table h.reg.first.closure ξ = .ok (draw, proved, read) → read ≤ ξ.size) ∧
    ∀ (sample : Array ℕ) ξ draw read,
      Spot.Select.selectSecond sample h.reg.ns h.reg.ks ξ = .ok (draw, read) → read ≤ ξ.size

theorem agreesTo_of_append {ξ b u : ByteArray} {R : ℕ} (hb : b = ξ ++ u) (hR : R ≤ ξ.size) : AgreesTo ξ b R := by
  subst hb
  refine ⟨?_, Properties.Spot.SecurityProofs.extract_append_of_le hR⟩
  rw [ByteArray.size_append]
  omega

theorem playCoins_of_starts {h : SpotFlock.Held} {i : Spot.Run.Input} {it : Spot.Run.Interior}
    {c₀ : FlockDeploy.Coin} {c₁ : SpotFlock.Stage h} (hw : ReadsWithin h)
    (h0 : ∃ u, SpotFlock.src c₀ = i.source ++ u) (h1 : ∃ u, SpotFlock.stageSrc c₁ = it.source ++ u) :
    PlayCoins h i it c₀ c₁ := by
  obtain ⟨u₀, hu₀⟩ := h0
  obtain ⟨u₁, hu₁⟩ := h1
  unfold SpotFlock.RunsAs.PlayCoins
  exact And.intro (fun draw proved read hs => agreesTo_of_append hu₀ (hw.1 _ draw proved read hs))
    (fun draw read hs => agreesTo_of_append hu₁ (hw.2 _ _ draw read hs))

theorem acceptedTwo_of_formAccepts {C : CaseOf} [hsn : Fact (SampledNoneAt C)] {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} (hrule : RuleIsTwo C) (hA : FormAccepts C o)
    (hi : inputOn C o = some i) (hit : i.interior = some it) : ∃ run, AcceptedTwo i it run := by
  have hA₀ := hA
  obtain ⟨k, hk, p, cm, e, pub, prof, hdec, hacc⟩ := hA
  have hon : ∀ {c₂ ξ₂ r i'}, VStar.Recursive.SampledTwo.inputAt p cm e.source e k.blobs = .ok (c₂, ξ₂, r, i') →
      inputOn C o = some i' := by
    intro c₂ ξ₂ r i' hin
    have hik : inputOfCase k = some i' := by
      obtain ⟨h1, h2, h3⟩ := hdec
      unfold inputOfCase
      rw [h1, h2, h3]
      simp only [bind, Except.bind, hin]
      rfl
    unfold inputOn
    rw [hk]
    exact hik
  obtain ⟨-, c₂, ξ₂, r, i', it', run, hin, hat, -⟩ :=
    SpotFlock.RunsAs.acceptedTwo_of_check (hrule o k p cm e hk hdec)
      (fun _ _ _ i'' _ hin'' hit'' => hsn.out o i'' _ hA₀ (hon hin'') hit'') hacc
  have hi' : inputOn C o = some i' := hon hin
  obtain rfl : i = i' := Option.some.inj (hi.symm.trans hi')
  obtain rfl : it = it' := Option.some.inj (hit.symm.trans hat.interior)
  exact ⟨run, hat⟩

theorem playAccepts_of_run (A : Live₂) (C : CaseOf) [Fact (SampledNoneAt C)] (S : SessionReading)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hrule : RuleIsTwo C) (hheld : HeldAt A C)
    (htwo : TwoHolds A.h) (hnd : NamesDistinct A.h) (hrec : RecordsAt A C) (hread : RecordsRead C)
    (hr2 : RoundTwo A.h) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') (hw : ReadsWithin A.h) : PlayAccepts A C S P' := by
  intro o i it x m₁ hA hi hit hx hm₁
  obtain ⟨m₂, hm₂, hsa⟩ := hst o i it hA hi hit
  have hrd : RoundsHeld A.h m₂ := hrounds o m₂ hA hm₂
  refine ⟨m₂, hm₂, hrd, fun c₀ c₁ r' hcoins => ?_⟩
  obtain ⟨hc0, hc1, hcs⟩ := hcoins
  obtain ⟨run, hat⟩ := acceptedTwo_of_formAccepts hrule hA hi hit
  obtain ⟨hrh, hrn⟩ := hrec o i it x m₁ hA hi hit hx hm₁
  obtain ⟨hjx, hjm⟩ := hread o i it x m₁ hA hi hit hx hm₁
  have hS : SessionsAccept A.h FlockDeploy.sha512 (it.statements.getD #[]) m₂ (P' o) r' := by
    unfold SpotFlock.RunsAs.SessionsAccept
    exact ⟨hsa, hrd, hscope o m₂ hA hm₂, hpin o m₂ hA hm₂, hsess o m₂ hA hm₂ r' hcs⟩
  exact SpotFlock.RunsAs.spotRunAcceptsPlay i it run A.h FlockDeploy.sha512 x m₁ m₂ (P' o) r' c₀ c₁ hat
    (hheld o i it hA hi hit) htwo hnd hrh hrn hjx hjm hr2 hS (playCoins_of_starts hw hc0 hc1)

theorem acceptsRun_of_run (A : Live₂) (x : Msg) (C : CaseOf) [Fact (SampledNoneAt C)] (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hfit : SourcesFit A C) (hsm : SessionMsgs A C S P') (hrule : RuleIsTwo C) (hheld : HeldAt A C)
    (htwo : TwoHolds A.h) (hnd : NamesDistinct A.h) (hrec : RecordsAt A C) (hread : RecordsRead C)
    (hr2 : RoundTwo A.h) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') (hw : ReadsWithin A.h) :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' hfit
    (playAccepts_of_run A C S P' hrule hheld htwo hnd hrec hread hr2 hst hrounds hscope hpin hsess hw) hsm

end Verity.SampledTwo
