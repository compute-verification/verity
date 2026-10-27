import Proofs.Properties.Verity.SampledRuns

/-!
# The sizes the readers need, from v2's refusals

`SizesAt` (`SampledRuns`) holds wherever v2 accepts, at a setting `A` whose draw is the registered draw's coins and whose
round bound covers v2's cap: v2 refuses a draw that opens other than the positive count of coins the registration's
`window.draw_coins` names (`OneStageZk.draw_of_accept`), and a session past `MAX_ROUNDS` rounds or with a use of `2 ^ 64`
F128s or more (`OneStageZk.rounds_of_accept`). The link points' count, one release per chunk of `m_pts` F128s per rep, is
flock-verify's own refusal (`Flock.Record.decode`'s S9, at the setup's `m_pts`), taken here as `hPts`.
-/

namespace Verity.Canon

open Accounting Core
open FlockSoundness.Discharge

variable {ι α : Type} [Fintype α] {d : Device ι}
variable {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}

/-- **`SizesAt` from v2's refusals**: wherever sampled proofs' verdict accepts, the draw opens one unit per release of
`A`'s draw when `A`'s draw is `64` bytes a coin the registration names (`hLb`), session 0 has at most `A.N` rounds
when `A.N` covers `MAX_ROUNDS` (`hN`), each of fewer than `2 ^ 64` coins, and its link points take one release per chunk
of their coin (`hPts`). -/
theorem sizesAt_of_accept (V : Vocab d α) (Brk : Outcome → Prop) (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc)
    (hLb : ∀ o, V.Verdict .sampled o → ∀ p c' e, SampledCase o.sent p c' e →
      ∀ m, _root_.SampledProofs.OneStageZk.registeredDraw c' = some m → A.Lb 0 = 64 * m)
    (hN : _root_.SampledProofs.OneStageZk.MAX_ROUNDS ≤ A.N)
    (hPts : ∀ o, V.Verdict .sampled o → ¬ Brk o → ∀ p c' e pub prof, SampledCase o.sent p c' e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c' e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
      ∀ (s0 : _root_.SampledProofs.OneStageZk.Session) rest, e.sessions.toList = s0 :: rest →
        ∀ b record us0, (Core.Blobs.ofArray o.sent.blobs).get s0.record s!"session {s0.name}'s record" = .ok b →
          String.fromUTF8? b = some record → _root_.SampledProofs.OneStageZk.sessionUses 0 record = .ok us0 →
          ∀ x ∈ us0.head?,
            x.1.coins = (splitPoints (Exec.mPtsOf D.unroll.g (kd / Composed.nTab A.I))).rest.length + 1) :
    SizesAt V Brk A := by
  intro o hv hb p c' e pub prof hc hacc
  obtain ⟨m, hm, hpos, hop⟩ := _root_.SampledProofs.OneStageZk.draw_of_accept hacc
  refine ⟨fun opened ho => ?_, fun s0 rest hl b record us0 hb0 hrec hus => ?_⟩
  · rw [hop opened ho]
    show m = (Game.splitBits 512 (by decide) (A.Lb 0 * 8)).rest.length + 1
    rw [splitBits_rest512, hLb o hv p c' e hc m hm]
    omega
  · obtain ⟨hlen, h64⟩ :=
      _root_.SampledProofs.OneStageZk.rounds_of_accept hacc s0 rest hl b record us0 hb0 hrec hus
    exact ⟨by omega, h64, hPts o hv hb p c' e pub prof hc hacc s0 rest hl b record us0 hb0 hrec hus⟩

end Verity.Canon
