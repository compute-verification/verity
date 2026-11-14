module

public import Proofs.Properties.Verity.SampledTwo.FormBytes
public import Proofs.Properties.Verity.SampledTwo.StageAlone
public import Proofs.Properties.Verity.SampledTwo.SessionTreeless
public import Proofs.Properties.Verity.SampledTwo.CutAccept

/-!
# The first stage's coins before every opening

`FirstShownEarly` (`FormBytes.lean`) is `FirstShown` with the coins it reads released before every opening. Here it
is a theorem (`firstShownEarly`): at every `Good` that holds the check to have read a log of both sessions and found
it the device's (`ReadIsDevice`), on a case map that is the one sent (`CaseIsSent`).

An accepted play is on two windows: the form refuses an interior audit on the boundary registration's window, so the
two windows its parameters name differ (`windowsOf_ne`, `CutAccept.lean`). There the play's log is the first
session's releases and then the interior's (`logsOf`). The first stage's bytes start the coins `drawCoins` reads
from the first session (`firstDrawn_of`), which are the coins of the first sample that the first session's releases
show (`drawn_shown`), and the first session's read holds no opening: the form's check refuses a first session's log
that holds a coin tree's line there (`SessionCoins.firstTreeless_of_check`, `SessionCoins.readLog_treeless`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled (freshCoins readLog)

/-- **`FirstShownEarly`**, where `Good` holds the check to have read a log and found it the device's: the two
windows the parameters name differ (`windowsOf_ne`), so the play's log is the first session's releases then the
interior's, the first session's releases show the coins the first stage's bytes start (`firstDrawn_of`,
`drawn_shown`), and none of them is an opening (`SessionCoins.firstTreeless_of_check`,
`SessionCoins.readLog_treeless`). -/
theorem firstShownEarly (C : CaseOf) (Good : Verity.Exchange → Prop) (hG : ∀ o, Good o → ReadIsDevice o)
    (hC : CaseIsSent C) : FirstShownEarly C Good := by
  intro o ha hgood i read hi hread _
  have hD : FirstDrawn C := firstDrawn_of C hC
  have hfree : SessionCoins.FirstTreeless C :=
    SessionCoins.firstTreeless_of_check C hC fun _ _ _ _ _ _ _ _ => rfl
  obtain ⟨hne, hg⟩ := hG o hgood
  have hne' : readsLogs o.sent ≠ [] := hne
  obtain ⟨k, p, c, e, _, _, _, _, hk, hd, _, _, _, _⟩ := inputOn_accepts ha hi
  obtain ⟨hcase, -⟩ := hC o k p c e hk hd
  obtain ⟨l, hl⟩ := List.exists_mem_of_ne_nil _ hne'
  obtain ⟨_, _, _, key, d₁, ξ, c₂, d₂, _, r₁, r₂, hcase', hkey, hd₁, hξ, hc₂, hd₂, -, hr₁, -, hlogs⟩ :=
    readsLogs_cases hl
  have hpce := Except.ok.inj (hcase.symm.trans hcase')
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  obtain ⟨lines, hlines, hrl⟩ := firstRead_some hd₁ hr₁
  obtain ⟨ls, hls, hlsl⟩ := firstLines_arr hlines
  have hpre := hD o ha i read hi hread _ _ _ key d₁ ξ hcase hkey hd₁ hξ
  have hw : d₁.window ≠ d₂.window := windowsOf_ne (windowsOf_eq hcase hd₁ hc₂ hd₂)
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
