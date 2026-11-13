module

public import Proofs.Properties.Verity.SampledTwo.SessionAsk
public import Proofs.Properties.Verity.SampledTwo.SessionRecordRounds
public import Proofs.Properties.Verity.SampledTwo.RoundsRise
public import Proofs.Properties.Verity.Sampled.MovesReach
public import Proofs.Flock.Claims.DeployK
public import Proofs.Flock.Claims.DeployRoundsZk
public import Flock.With.Deployed
public import Flock.With.ZkEq

/-!
# No release is put on a session's last slot, from the session's record

`Messages.sessionMsgsAsk_form` (`SessionAsk.lean`) rests on `Messages.LastSlotFree`: the locator puts no release of an
accepted play's log on the last slot of a template's session, the slot the session is decided on. This module takes it
from what the session's record gives its rep streams, but for one named hypothesis.

* **The last slot's stream and round** (`slotStream_rep1`): at a slot past the first rep's, the sessions' layout
  (`SessionCoins.slotStream`) gives the table's rep 1 stream, at the slot less one and less a rep's rounds. So at a
  session of at least two rounds the last slot is a round of the rep 1 stream that is at least a rep's rounds.
* **A release on a slot is of a round the record gives** (`SessionCoins.LogRecorded`, `SessionRecordRounds.lean`).
* **The verdict's count** (`verify_reps`): a record Flock's masked verdict accepts at a session of one table gives
  each of that table's two rep streams as many rounds as the setup's rep has (`FlockDeploy.repRounds`): the verdict
  read each rep's stream to its end (`FlockDeploy.zkVerifyRepWith_rounds`).
* **At the form** (`accepted_reps`): a session the per-member form accepted (`MemberChecks.SessionAccepted`) is one
  the masked verdict accepted at the form's own setup of it, of one table, each of its statements under a scheme of
  the registry; so its record on the blobs gives that table's rep streams the setup's rep's rounds.
* **The named hypothesis** (`SetupsOfTemplates`): on an accepted play, a template's session has at least two rounds,
  and its record is one the masked verdict accepts at a setup of one table, of the template's table name, whose rep
  has `SessionCoins.repsOf` of the template's round count.
* **`LastSlotFree`** (`recordReps_of_setups`, `lastSlotFree_of_recorded`, `lastSlotFree_of_setups`): a release on the
  last slot would be of a round the record gives, at least a rep's rounds, of a stream the record gives exactly a
  rep's rounds. `sessionMsgsAsk_run` is `Messages.sessionMsgsAsk_form` at the run's locator with it.

What this does not prove: `SetupsOfTemplates`. The form's check verified each session's record at its own setup of the
session (`accepted_reps`). That this setup has the template's table name and its count of rep rounds is the tie of a
member's session statement to its template's statement, which no premise here states.
-/

@[expose] public section

namespace Verity.SampledTwo.Messages

open Claims
open Lean (Json)
open Verity.Sampled (repStreamName)
open SessionCoins (slotStream repsOf)

/-! ## The last slot -/

/-- **A slot past the first rep's is a round of the rep 1 stream**: the slot less one, less a rep's rounds. -/
theorem slotStream_rep1 (tab : String) {R j : ℕ} (hj : ¬ j = 0) (hR : ¬ j - 1 < R) :
    slotStream tab R j = (repStreamName tab 1, j - 1 - R) := by
  unfold SessionCoins.slotStream
  rw [ite_eq_right hj, ite_eq_right hR]

/-- **A session's record gives its rep 1 stream no more rounds than a rep has**: on a play accepted (`FormAccepts`)
whose log the check read is the device's (`Good`), a template's session has at least two rounds, and the `t`-th of
the sessions' records the check read (`SessionCoins.sessionRecords`) gives the rep 1 stream of the table `tab t` at
most `SessionCoins.repsOf` of the template's round count. -/
def RecordReps (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ recs rec t, t < A.h.ms.length →
    SessionCoins.sessionRecords C o = some recs → recs[t]? = some rec →
      2 ≤ SpotFlock.k₀ A.h t ∧
        Spot.Challenger.recordRounds rec (repStreamName (tab t) 1) ≤ repsOf (SpotFlock.k₀ A.h t)

/-- **`LastSlotFree`, from the session's record**: a release the locator puts on a session's last slot is of a round
the session's record gives (`SessionCoins.LogRecorded`), which at the last slot is a round of the rep 1 stream at
least a rep's rounds (`slotStream_rep1`); the record gives that stream no more (`RecordReps`). -/
theorem lastSlotFree_of_recorded {A : Live₂} {C : CaseOf} {Good : Verity.Exchange → Prop} {L : CoinMap.Locate}
    {tab : ℕ → String} (hrec : SessionCoins.LogRecorded A C Good L tab) (hreps : RecordReps A C Good tab) :
    LastSlotFree A (FormAccepts C) Good L := by
  intro o hacc hgood a rl b hlog t ht hloc
  have hpos := SpotFlock.k₀_pos A.h t
  obtain ⟨recs, rec, hrecs, hrt, hlt⟩ :=
    hrec o hacc hgood a rl b hlog t (SpotFlock.k₀ A.h t - 1) (by omega) hloc
  obtain ⟨h2, hle⟩ := hreps o hacc hgood recs rec t ht hrecs hrt
  have hR : ¬ SpotFlock.k₀ A.h t - 1 - 1 < repsOf (SpotFlock.k₀ A.h t) := by
    unfold SessionCoins.repsOf
    omega
  rw [slotStream_rep1 (tab t) (show ¬ SpotFlock.k₀ A.h t - 1 = 0 by omega) hR] at hlt
  have hlt' : SpotFlock.k₀ A.h t - 1 - 1 - repsOf (SpotFlock.k₀ A.h t) <
      Spot.Challenger.recordRounds rec (repStreamName (tab t) 1) := hlt
  unfold SessionCoins.repsOf at hlt' hle
  omega

/-! ## The verdict's count -/

/-- **A record the masked verdict accepts gives the first table's rep streams the setup's rep's rounds**: where
`Flock.Zk.verify` accepts the record `text` at the session `spec` of the one table `T`, at the setups of the
statements `sts`, each under a scheme of the registry, and `text` parses as `rec`, `rec` gives `T`'s rep `r` stream
`FlockDeploy.repRounds` of the first statement's rounds. The verdict read the stream to its end
(`FlockDeploy.zkVerifyRepWith_rounds`), over as many link points as the session has. -/
theorem verify_reps {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} {text : String} {proofs : Array ByteArray}
    (hv : Flock.Zk.verify spec (FlockDeploy.tabsOf sts) text proofs = .ok ())
    (hN : ∀ st ∈ sts, st.merkle.Named) {rec : Json} (hp : Flock.JsonParse.parse text = .ok rec) {T : String}
    (hT : spec.tables = #[T]) {r : ℕ} (hr : r < 2) :
    Spot.Challenger.recordRounds rec (repStreamName T r) = FlockDeploy.repRounds true spec sts[0]! := by
  obtain ⟨sess, hd⟩ := Verity.Sampled.zkVerify_decode hv
  obtain ⟨h0, sch, tPad, c, hsch, hrep⟩ := Verity.Sampled.verifyRep_of_accepted hv hd ⟨r, hr⟩
  have hi : 0 < sts.size := by simpa [FlockDeploy.tabsOf] using h0
  have htab : (FlockDeploy.tabsOf sts)[0]'h0 = Flock.Setup.ofCircuit sts[0]! := by
    rw [getElem!_pos sts 0 hi]; exact Array.getElem_map ..
  rw [htab] at hsch hrep
  have hN' : (Flock.Setup.ofCircuit sts[0]!).merkle.Named := by
    rw [getElem!_pos sts 0 hi]; exact hN _ (Array.getElem_mem hi)
  have hrep' : Flock.Zk.verifyRep spec (Flock.Setup.ofCircuit sts[0]!) sch tPad sess 0 r proofs[r]! = .ok c := hrep
  rw [← Flock.With.zk_verifyRepWith_std spec _ hN'] at hrep'
  have hsch' : Flock.Zk.schedule sts[0]!.m = .ok (sch, tPad) := hsch
  obtain ⟨stream, extras, hfind, hex, hsize⟩ := FlockDeploy.zkVerifyRepWith_rounds Flock.With.std hsch hrep'
  have hx : extras.size = sts[0]!.regions.size * sess.points.size := FlockDeploy.extraClaims_size hex
  have hpn : sess.points.size = spec.points := by
    obtain ⟨_, _, _, _, _, -, -, -, -, hpn, -⟩ :=
      FlockDeploy.decodeWith_parts Flock.With.std ((Flock.With.decodeWith_std spec text).trans hd)
    exact hpn
  have e0 : spec.tables[0]! = T := by
    rw [hT]
    rfl
  rw [e0] at hfind
  have hfind' : sess.streams.find? (·.name == repStreamName T r) = some stream := hfind
  have hrr : Verity.Sampled.recordRounds rec (repStreamName T r) = stream.rounds.size :=
    Verity.Sampled.recordRounds_decode hd hp (Verity.Sampled.repStreamName_ne T r) hfind'
  have e : Spot.Challenger.recordRounds rec (repStreamName T r) =
      Verity.Sampled.recordRounds rec (repStreamName T r) := rfl
  rw [e, hrr, hsize]
  unfold FlockDeploy.repRounds
  simp only [↓reduceIte, hsch']
  rw [hx, hpn, Nat.mul_comm]
  rfl

/-- **A member's accepted session's record gives its table's rep streams the form's setup's rep's rounds**: where
the per-member form accepted the session `s` at its member's statement and `recJ` is its record on the blobs, the
form's setup of the session (`setupAt`'s, the one `Flock.Zk.verify` accepted the record at) has one table `T` alone,
each of its statements under a scheme of the registry, the record's text parses as `recJ`, and `recJ` gives each of
`T`'s two rep streams `FlockDeploy.repRounds` of the setup's first statement's rounds (`verify_reps`). -/
theorem accepted_reps {own : Spot.OneStageZk.Own} {circuitB pubB : ByteArray} {bs : Core.Blobs}
    {rd : Spot.OneStageZk.Reads} {pop : Flock.V Unit} {s : Spot.OneStageZk.RunFiles}
    (hacc : MemberChecks.SessionAccepted
      (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop) bs s)
    {recJ : Json} (hrec : Verity.Sampled.recordOn bs s = some recJ) :
    ∃ text dr proofs spec sts rest T,
      Spot.OneStageZk.setupAt (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I
          (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).rd dr =
            .ok (spec, FlockDeploy.tabsOf sts, rest) ∧
        Flock.Zk.verify spec (FlockDeploy.tabsOf sts) text proofs = .ok () ∧ (∀ st ∈ sts, st.merkle.Named) ∧
        Flock.JsonParse.parse text = .ok recJ ∧ spec.tables = #[T] ∧
        ∀ r, r < 2 →
          Spot.Challenger.recordRounds recJ (repStreamName T r) = FlockDeploy.repRounds true spec sts[0]! := by
  obtain ⟨coinKeys, b, text, hg, hu, hv⟩ := hacc
  obtain ⟨proofs, spec, tabs, rest, -, hst, hzk⟩ := Verity.Sampled.verdictOf_ok hv
  obtain ⟨sess, hd⟩ := Verity.Sampled.zkVerify_decode hzk
  have htag : Flock.Tags.find? Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  have hh : (Flock.Witness.zkTags
      (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I).hiddenOutputs = true :=
    (Verity.Sampled.statement_zkTags htag _ _).1
  have hc := SessionCoins.setupAt_coinTree
    (W := (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I) hh rfl hst
  have htab := SessionCoins.setupAt_oneTable
    (W := (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I) hh rfl hst
  obtain ⟨j, cj, nonce, key, hp, -, -⟩ := Verity.Sampled.decode_coinTree hd hc
  have hrec' : Verity.Sampled.recordOn bs s = some ((Flock.JsonParse.parse text).toOption.getD Json.null) := by
    simp [Verity.Sampled.recordOn, hg, hu, Except.toOption]
  have hj : recJ = j := by
    have e := Option.some.inj (hrec.symm.trans hrec')
    rw [hp] at e
    exact e
  rw [← hj] at hp
  obtain ⟨sts, hs, htabs⟩ := Verity.Sampled.setupAt_ok hst
  obtain ⟨hset, -⟩ := Verity.Sampled.sessionAt_ok hs
  have hN : ∀ st ∈ sts, st.merkle.Named := Flock.With.setupTables_named _ _ _ _ _ _ _ _ _ _ _ _ _ _ hset
  have htabs' : tabs = FlockDeploy.tabsOf sts := htabs
  subst htabs'
  exact ⟨text, _, proofs, spec, sts, rest, _, hst, hzk, hN, hp, htab, fun r hr => verify_reps hzk hN hp htab hr⟩

/-! ## The named hypothesis, and the last slot from it -/

/-- **A template's session is set up as its template**: on a play accepted (`FormAccepts`) whose log the check read
is the device's (`Good`), a template's session has at least two rounds, and the `t`-th of the sessions' records the
check read (`SessionCoins.sessionRecords`) is one Flock's masked verdict accepts, each statement under a scheme of the
registry, at a setup of one table alone, of the name `tab t`, whose rep has `SessionCoins.repsOf` of the template's
round count (`FlockDeploy.repRounds`). The form accepted each session at its own setup of it (`accepted_reps`); that
this setup is the template's is what this states. -/
def SetupsOfTemplates (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ recs rec t, t < A.h.ms.length →
    SessionCoins.sessionRecords C o = some recs → recs[t]? = some rec →
      2 ≤ SpotFlock.k₀ A.h t ∧
        ∃ (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) (text : String) (proofs : Array ByteArray),
          Flock.Zk.verify spec (FlockDeploy.tabsOf sts) text proofs = .ok () ∧ (∀ st ∈ sts, st.merkle.Named) ∧
            Flock.JsonParse.parse text = .ok rec ∧ spec.tables = #[tab t] ∧
            FlockDeploy.repRounds true spec sts[0]! = repsOf (SpotFlock.k₀ A.h t)

/-- **`RecordReps`, from `SetupsOfTemplates`**: the record gives the rep 1 stream of the template's table the setup's
rep's rounds (`verify_reps`), which are `SessionCoins.repsOf` of the template's round count. -/
theorem recordReps_of_setups {A : Live₂} {C : CaseOf} {Good : Verity.Exchange → Prop} {tab : ℕ → String}
    (hset : SetupsOfTemplates A C Good tab) : RecordReps A C Good tab := by
  intro o hacc hgood recs rec t ht hrecs hrt
  obtain ⟨h2, spec, sts, text, proofs, hv, hN, hp, hT, hR⟩ := hset o hacc hgood recs rec t ht hrecs hrt
  exact ⟨h2, le_of_eq ((verify_reps hv hN hp hT (show 1 < 2 by omega)).trans hR)⟩

/-- **`LastSlotFree`, from `SessionCoins.LogRecorded` and `SetupsOfTemplates`.** -/
theorem lastSlotFree_of_setups {A : Live₂} {C : CaseOf} {Good : Verity.Exchange → Prop} {L : CoinMap.Locate}
    {tab : ℕ → String} (hrec : SessionCoins.LogRecorded A C Good L tab) (hset : SetupsOfTemplates A C Good tab) :
    LastSlotFree A (FormAccepts C) Good L :=
  lastSlotFree_of_recorded hrec (recordReps_of_setups hset)

/-- **`LastSlotFree` at the run's locator, from `SetupsOfTemplates`**: `SessionCoins.logRecorded_run` is the
release's half. -/
theorem lastSlotFree_run_of_setups (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (Ls : CoinMap.Locate)
    (tab : ℕ → String) (hsz : SessionCoins.HeldSize) (hI : SessionCoins.InteriorLogged C Good)
    (hset : SetupsOfTemplates A C Good tab) :
    LastSlotFree A (FormAccepts C) Good (SessionCoins.runLocate A Ls tab) :=
  lastSlotFree_of_setups (SessionCoins.logRecorded_run A C Good Ls tab hsz hI) hset

/-- **`SessionMsgsAsk` at the form and the run's locator** (`sessionMsgsAsk_form`), with `LastSlotFree` from
`SetupsOfTemplates`. -/
theorem sessionMsgsAsk_run (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (Good : Verity.Exchange → Prop) (Ls : CoinMap.Locate) (tab : ℕ → String) (hplay : PlayAccepts A C S P')
    (hsess : SessionMsgs A C S P') (hrounds : RoundsHeldAt A C S) (hsz : SessionCoins.HeldSize)
    (hI : SessionCoins.InteriorLogged C Good) (hset : SetupsOfTemplates A C Good tab) :
    SessionMsgsAsk A S (FormAccepts C) (fun o => Good o ∧ ReadAt A x (formReading A C S dev) o)
      (SessionCoins.runLocate A Ls tab) :=
  sessionMsgsAsk_form A x C S dev P' Good _ hplay hsess hrounds
    (lastSlotFree_run_of_setups A C Good Ls tab hsz hI hset)

end Verity.SampledTwo.Messages
