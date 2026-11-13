module

public import Properties.Pous.MemoryChallenger.Session
public import Proofs.Properties.Pous.MemoryChallenger.Attest

/-!
# Proofs of what the auditor sees of the memory challenger in a session (`Properties.Pous.MemoryChallenger.Session`)

**The tie.** Soundness's record of a round (`Recorded.round`) is `roundSeen` on what it reads of the round, its size
and its `blockDigest` (`round_eq_roundSeen`), so a change to soundness's function breaks the build here.

**The view.** One step of the device's code is one step of `seenCode` on the exchange's skeleton (`step_skeleton`), and
both start from one state, so on any inputs they send the same messages (`messages_skeleton`). Under `Confined` the
device emits its code's messages, under `Faithful` on its true inputs, and under `WitnessPathPrivate` nobody overhears
its path to the prover, so the auditor sees what `seenCode` sends in the clear on the device's challenges and the
skeleton, at every `dest` (`auditorSeesSkeletonAt`); `auditorSeesDigests` is it at `reportPublic`.

**Honest and nontrivial lemmas.** The premises hold of the device run as it is meant to run (`meantRun`), on any
parameters, challenges and session (`auditorSeesDigests_honest`). Without `WitnessPathPrivate` the conclusion fails: an
observer on the path to the prover who overhears a warm-up's frame shows it to the auditor, though the code sends
nothing in the clear on a warm-up (`auditorSeesDigests_nontrivial`). And the view is not less than the digest: on a
one-round session (`oneRound`), two answers whose `blockDigest`s differ show the auditor different reports
(`auditorSeesDigests_digest`).
-/

@[expose] public section

namespace MemoryChallenger.Session.Properties.SecurityProofs

open Accounting.Assumptions
open MemoryChallenger.Properties (Observed Recorded)
open MemoryChallenger.Attest.Properties.SecurityProofs (meantRun meantRun_ok)

/-- **The tie**: soundness's record of a round is `roundSeen` on what it reads of the round (`seen`). -/
theorem round_eq_roundSeen (c : CoinTree) (l b : Nat) (st : Recorded) (o : Observed) :
    Recorded.round c l b st o = roundSeen c l b st (seen o) := by
  rfl

theorem step_skeleton {Ω : Type} (p : Params Ω) (st : Ω × Recorded) (x : Step ByteArray Observed) :
    (code p).step st x = (seenCode p).step st x.skeleton := by
  cases x with
  | warmup _ => rfl
  | round o =>
    simp only [code, seenCode, codeOn, Step.skeleton, round_eq_roundSeen]

theorem trace_skeleton {Ω : Type} (p : Params Ω) (s : Ω × Recorded) (xs : List (Step ByteArray Observed)) :
    (code p).trace s xs = (seenCode p).trace s (xs.map Step.skeleton) := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih => simp only [Parties.Code.trace, List.map_cons, step_skeleton p s x, ih]

theorem messages_skeleton {Ω : Type} (p : Params Ω) (ω : Ω) (xs : List (Step ByteArray Observed)) :
    (code p).messages ω xs = (seenCode p).messages ω (xs.map Step.skeleton) := by
  unfold Parties.Code.messages
  rw [trace_skeleton]
  rfl

theorem clear_map {Wi : Type} (dest : Ch → Parties.Dest) (wire : Ch × Msg → Wi) (L : List (Ch × Msg)) :
    ((L.map fun m => (dest m.1, wire m)).filter (·.1 = .clear)).map (·.2) = (inClear dest L).map wire := by
  induction L with
  | nil => rfl
  | cons m L ih =>
    by_cases hm : dest m.1 = .clear
    · simpa [inClear, hm] using ih
    · simpa [inClear, hm] using ih

theorem auditorSeesSkeletonAt (dest : Ch → Parties.Dest) : AuditorSeesSkeletonAt dest := by
  intro Ω Wi A p wire r hf hc hw
  obtain ⟨ht, hs⟩ := hf
  have e : r.sent = (code p).messages r.coins r.truth := by
    rw [Parties.Code.messages, ← ht]; exact congrArg Prod.snd hs
  unfold Parties.auditorView
  rw [hc, hw, List.map_nil, List.append_nil, clear_map, e, messages_skeleton]

theorem auditorSeesDigests : AuditorSeesDigests := auditorSeesSkeletonAt reportPublic

/-- **`AuditorSeesDigests`'s honest lemma**: its premises hold of the device run as it is meant to run, at any
parameters, on any challenges and any session's exchanges, answers included. -/
theorem auditorSeesDigests_honest {Ω Wi A : Type} (p : Params Ω) (wire : Ch × Msg → Wi) (ω : Ω)
    (xs : List (Step ByteArray Observed)) :
    ∃ r : Parties.Run Ω (Ω × Recorded) (Step ByteArray Observed) Ch Msg Wi A,
      Parties.Faithful (code p) r ∧ Parties.Confined reportPublic wire r ∧ Parties.WitnessPathPrivate r ∧
        r.coins = ω ∧ r.truth = xs := by
  obtain ⟨h1, -, h2, h3⟩ := meantRun_ok (A := A) (code p) reportPublic wire ω xs
  exact ⟨_, h1, h2, h3, rfl, rfl⟩

/-- A run of the device as it is meant to run on one warm-up, but for an observer on its path to the prover who
overhears the warm-up's frame. -/
def overheardRun {Ω : Type} (p : Params Ω) (ω : Ω) (answer : ByteArray) :
    Parties.Run Ω (Ω × Recorded) (Step ByteArray Observed) Ch Msg (Ch × Msg) Unit :=
  { meantRun (code p) reportPublic id ω [.warmup answer] with overheard := [(Ch.challenge, Msg.warmup)] }

/-- **`AuditorSeesDigests`'s nontrivial lemma**: without `WitnessPathPrivate` its conclusion fails: the device, at any
parameters and challenges, `Faithful` and `Confined` on one warm-up whose frame an observer on the path to the prover
overhears, shows the auditor that frame, though its code sends nothing in the clear on a warm-up. -/
theorem auditorSeesDigests_nontrivial {Ω : Type} (p : Params Ω) (ω : Ω) :
    ¬ ∀ r : Parties.Run Ω (Ω × Recorded) (Step ByteArray Observed) Ch Msg (Ch × Msg) Unit,
      Parties.Faithful (code p) r → Parties.Confined reportPublic id r →
        Parties.auditorView id r =
          (inClear reportPublic ((seenCode p).messages r.coins (r.truth.map Step.skeleton))).map id := by
  intro h
  have hok := meantRun_ok (A := Unit) (code p) reportPublic id ω [.warmup ByteArray.empty]
  have e := h (overheardRun p ω .empty) hok.1 hok.2.2.1
  change [(Ch.challenge, Msg.warmup)] = [] at e
  exact List.cons_ne_nil _ _ e

/-! ## The view is not less than the digest -/

/-- The `CoinTree` of a one-round session over one block: its one challenge is block 0. -/
def oneCoins : CoinTree := ⟨.empty, 1, #[0], .empty, #[.empty], #[], .empty⟩

/-- A session of one round over one-byte answers, whose device signs every message with the empty signature. -/
def oneRound : Params Unit := ⟨fun _ => oneCoins, 0, 1, .empty, "", fun _ => .empty, fun _ _ _ => .empty⟩

/-- The round `oneRound`'s device records of an answer whose `blockDigest` is `d`: round 0, on block 0, both readings
at 0. -/
def oneRecord (d : ByteArray) : Round := ⟨0, 0, 0, 0, d⟩

theorem round_one (b : ByteArray) (hb : b.size = 1) :
    Recorded.round oneCoins 0 1 ⟨#[], none⟩ ⟨0, 0, b⟩ = ⟨#[oneRecord (blockDigest b)], some COMPLETE⟩ := by
  have h : (⟨0, 0, b⟩ : Observed).block.size = 1 := hb
  unfold Recorded.round
  rw [h]
  rfl

theorem message_one (c : SessionCommitment) (e : String) (d : ByteArray) :
    ∃ m, message c e #[oneRecord d] = .ok m := by
  unfold message recordsRoot
  rw [Array.mapM_eq_mapM_toList]
  exact ⟨_, rfl⟩

/-- The report `oneRound`'s device sends once its one round, of digest `d`, has completed the session. -/
def oneReport (d : ByteArray) : Msg :=
  .report ⟨oneRound.commitment (), COMPLETE, #[oneRecord d], "", .empty, .empty⟩ (oneRound.commitment ())
    (#[oneRecord d].map fun r => oneCoins.opening r.j)

theorem messages_one (b : ByteArray) (hb : b.size = 1) :
    (code oneRound).messages () [.round ⟨0, 0, b⟩] =
      [(Ch.challenge, Msg.index 0), (Ch.challenge, Msg.done), (Ch.report, oneReport (blockDigest b))] := by
  obtain ⟨m, hm⟩ := message_one (oneRound.commitment ()) COMPLETE (blockDigest b)
  show (Ch.challenge, Msg.index (oneCoins.indices[0]!)) ::
      oneRound.closing () (Recorded.round oneCoins 0 1 ⟨#[], none⟩ ⟨0, 0, b⟩) ++ [] = _
  rw [round_one b hb, List.append_nil]
  simp only [Params.closing, hm]
  rfl

/-- **The view is not less than the digest**: on the one-round session `oneRound`, run as it is meant to run, two
one-byte answers whose `blockDigest`s differ show the auditor different reports. -/
theorem auditorSeesDigests_digest {A : Type} (b b' : ByteArray) (hb : b.size = 1) (hb' : b'.size = 1)
    (hd : blockDigest b ≠ blockDigest b') :
    Parties.auditorView id (meantRun (A := A) (code oneRound) reportPublic id () [.round ⟨0, 0, b⟩]) ≠
      Parties.auditorView id (meantRun (A := A) (code oneRound) reportPublic id () [.round ⟨0, 0, b'⟩]) := by
  have view : ∀ b : ByteArray, b.size = 1 →
      Parties.auditorView id (meantRun (A := A) (code oneRound) reportPublic id () [.round ⟨0, 0, b⟩]) =
        [(Ch.report, oneReport (blockDigest b))] := by
    intro b hb
    show ((((code oneRound).messages () [.round ⟨0, 0, b⟩]).map fun m => (reportPublic m.1, id m)).filter
        (·.1 = .clear)).map (·.2) ++ ([] : List (Ch × Msg)).map id = _
    rw [messages_one b hb]
    rfl
  rw [view b hb, view b' hb']
  intro e
  simp [oneReport, oneRecord, hd] at e

end MemoryChallenger.Session.Properties.SecurityProofs
