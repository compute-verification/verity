module

public import Proofs.Properties.Verity.SampledTwo.FormBytes
public import Proofs.Properties.Verity.SampledTwo.StageCut
public import Proofs.Properties.Verity.SampledTwo.SessionCut

/-!
# The first stage's coins before every opening, on two windows

`FirstShownEarly` (`FormBytes.lean`) is `FirstShown` with the coins it reads released before every opening. Here it
is a theorem at the cut's `LogIsDevice`, on every case map whose accepted plays have their two sessions on two
windows (`TwoWindows`).

On two windows the play's log is the first session's releases and then the interior's (`logsOf`). The first stage's
bytes start the coins `drawCoins` reads from the first session (`FirstDrawn`), which are the coins of the first
sample that the first session's releases show (`drawn_shown`), and the first session's read holds no opening: the
form's check refuses a first session's log that holds a coin tree's line there (`SessionCoins.FirstTreeless`,
`SessionCoins.readLog_treeless`).

**What it rests on**, each a named `Prop` taken as a hypothesis: `CaseIsSent`, `CutReads` and `FirstDrawn`
(`StageCut.lean`), `SessionCoins.FirstTreeless` (`SessionCut.lean`), and `TwoWindows`, which is the parameters'.

**Not established here.** On one window both sessions' reads are the whole log, and nothing here holds the first
stage's coins to be released before a round is opened there.
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled (freshCoins readLog)

/-- **The two sessions of every accepted play are on two windows**: where the case names the first session's window
and the interior's, they differ. -/
def TwoWindows (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ p c e key d₁ d₂, caseOf o.sent = .ok (p, c, e) → keyOf o.sent.params = some key →
    firstDraws key p c e = some d₁ → secondDraws p e = some d₂ → d₁.window ≠ d₂.window

/-- **`FirstShownEarly` on two windows**, at the cut's `LogIsDevice`: the play's log is the first session's releases
then the interior's, the first session's releases show the coins the first stage's bytes start (`drawn_shown`), and
none of them is an opening (`SessionCoins.readLog_treeless`). -/
theorem firstShownEarly_two (C : CaseOf) (hC : CaseIsSent C) (hR : CutReads C) (hD : FirstDrawn C)
    (hfree : SessionCoins.FirstTreeless C) (hTwo : TwoWindows C) : FirstShownEarly C LogIsDevice := by
  intro o ha hg i read hi hread _
  obtain ⟨k, p, c, e, _, _, _, _, hk, hd, _, _, _, _⟩ := inputOn_accepts ha hi
  obtain ⟨hcase, -⟩ := hC o k p c e hk hd
  obtain ⟨l, hl⟩ := List.exists_mem_of_ne_nil _ (hR o ha)
  obtain ⟨_, _, _, key, d₁, ξ, _, d₂, _, r₁, r₂, hcase', hkey, hd₁, hξ, -, hd₂, -, hr₁, -, hlogs⟩ :=
    readsLogs_cases hl
  have hpce := Except.ok.inj (hcase.symm.trans hcase')
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  obtain ⟨lines, hlines, hrl⟩ := firstRead_some hd₁ hr₁
  obtain ⟨ls, hls, hlsl⟩ := firstLines_arr hlines
  have hpre := hD o ha i read hi hread _ _ _ key d₁ ξ hcase hkey hd₁ hξ
  have hw := hTwo o ha _ _ _ key d₁ d₂ hcase hkey hd₁ hd₂
  unfold logsOf at hlogs
  rw [if_neg hw] at hlogs
  have hO := hg (r₁ ++ r₂) (by rw [hlogs]; simp)
  rw [List.map_append] at hO
  obtain ⟨a, b, hab, h₁, -⟩ := SessionCoins.forall₂_append_left hO
  have hop := SessionCoins.readLog_treeless hrl
    (hfree o ha _ _ _ key d₁ ξ r₁ d₂ hcase hkey hd₁ hξ hr₁ hd₂ hw)
  refine ⟨d₁.label, d₁.after, a, b, ?_, hab, ?_, ?_⟩
  · rw [hk, Option.bind_some]
    exact firstDraw_of hd hd₁
  · intro r hr j s n hrf
    obtain ⟨x', hx', hRx⟩ := SessionCoins.forall₂_mem_of_right h₁ r hr
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hx'
    have hopen : SessionCoins.openTree x.frame = some j :=
      (congrArg SessionCoins.openTree ((SessionCoins.of_ofV2 hRx).1.trans hrf)).trans rfl
    rw [hop x hx] at hopen
    cases hopen
  · rw [drawn_shown hξ hls hlsl hrl h₁]
    exact hpre

end Verity.SampledTwo
