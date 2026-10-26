import Proofs.Flock.Claims.Recursion.VerdictRowG

/-!
# At a decision of `nextR`, the record is the reader's (`recJ`)

`decR` hands over the record the reader builds (`recJ`, at `recStreams`), not the last message's JSON: the verdict
and the checks after it (`VerdictR.At .verdict`, `.after`) read `jAt` at it. This file states that at every decision.

* **The record** (`decR_recJ`, `actR_recJ`, `DecAt.recJ_eq`): the decision's record is `recJ` at its setup, link,
  first coin, decided proofs and some round messages `ms`, and its rounds are `ms` paired with the coins after the
  link. `DecAt.recJ_eq` states it at the verifier's setup, the one `jAt` is evaluated at.
* **The round messages** (`readPs_frames`, `decR_frames`): `ms` are the decided proofs' framed bytes, rep by rep, each
  at the `T` its opened rounds fit.
-/

namespace FlockDeploy

open Flock

/-- **The proofs `readPs` reads and their round messages**: each proof fits its rep's opened rounds at some `T`, and
the round messages are the proofs' framed bytes at those `T`, rep by rep. -/
theorem readPs_frames {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) :
    ∃ Ts : ℕ → F128, pf.1.size = 2 * sts.size ∧
      (∀ (k : ℕ) (hk : k < pf.1.size),
        FitsRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) pf.1[k] (Ts k)) ∧
      pf.2 = ((List.range pf.1.size).map fun k => framesOf (repCtx spec sts c₁ r₀ cs k) pf.1[k]! (Ts k)).flatten := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    have hk : ∀ k, ∃ T : F128, ∀ hk : k < rs.length,
        FitsRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) rs[k].1 T ∧
          rs[k].2 = framesOf (repCtx spec sts c₁ r₀ cs k) rs[k].1 T := by
      intro k
      by_cases hkl : k < rs.length
      · have hg := hget k (by rw [List.length_range]; omega) hkl
        rw [List.getElem_range] at hg
        obtain ⟨T, hT, he⟩ := readRep_fits hg
        exact ⟨T, fun _ => ⟨hT, he⟩⟩
      · exact ⟨0, fun h => absurd h hkl⟩
    choose Ts hTs using hk
    refine ⟨Ts, by simp [hlen], fun k hk' => ?_, ?_⟩
    · have hk'' : k < rs.length := by simpa using hk'
      simpa using (hTs k hk'').1
    · show (rs.map Prod.snd).flatten = ((List.range (rs.map Prod.fst).toArray.size).map fun k =>
        framesOf (repCtx spec sts c₁ r₀ cs k) (rs.map Prod.fst).toArray[k]! (Ts k)).flatten
      congr 1
      apply List.ext_getElem
      · simp
      · intro n h1 h2
        have hn : n < rs.length := by simpa using h1
        rw [List.getElem_map, (hTs n hn).2, List.getElem_map, List.getElem_range,
          getElem!_pos (rs.map Prod.fst).toArray n (by simpa using hn)]
        simp
  · rw [if_neg hc] at h
    cases h

/-- **At a decision, the record is the reader's**: `recJ` at the decision's setup, link, first coin and proofs, over
round messages `ms` that the decision's rounds pair with the coins after the link. -/
theorem decR_recJ {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) :
    ∃ ms : List Claims.Msg, d.mids = ms.zip cs.tail.dropLast ∧
      d.f.j = recJ d.spec d.sts d.link d.c₁ ms cs.tail.dropLast d.f.ps := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, -, -, -, -, rfl⟩ := decR_some h
  exact ⟨pf.2, rfl, rfl⟩

/-- **At a decision, the round messages are the decided proofs' framed bytes**: each rep's proof fits its opened
rounds at some `T`, and the rounds and the record (`recJ`) are at the proofs' framed bytes at those `T`, rep by rep. -/
theorem decR_frames {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) :
    ∃ (r₀ : List ByteArray) (rss : List (List ByteArray)) (Ts : ℕ → F128),
      (rowsOf (as.headD [])).map (List.map nodeOf) = some r₀ ∧ as.tail.dropLast.mapM rowsOf = some rss ∧
      d.f.ps.size = 2 * d.sts.size ∧
      (∀ (k : ℕ) (hk : k < d.f.ps.size), FitsRep (repCtx d.spec d.sts d.c₁ r₀ cs.tail.dropLast k)
        ((cutLens (repLens d.spec d.sts) rss).getD k []) d.f.ps[k] (Ts k)) ∧
      d.mids = ((List.range d.f.ps.size).map fun k =>
        framesOf (repCtx d.spec d.sts d.c₁ r₀ cs.tail.dropLast k) d.f.ps[k]! (Ts k)).flatten.zip cs.tail.dropLast ∧
      d.f.j = recJ d.spec d.sts d.link d.c₁ ((List.range d.f.ps.size).map fun k =>
        framesOf (repCtx d.spec d.sts d.c₁ r₀ cs.tail.dropLast k) d.f.ps[k]! (Ts k)).flatten
        cs.tail.dropLast d.f.ps := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, -, -, h6, h7, -, h9, rfl⟩ := decR_some h
  obtain ⟨Ts, hsz, hfit, hms⟩ := readPs_frames h9
  exact ⟨r₀, rss, Ts, h6, h7, hsz, hfit, by rw [hms], by rw [hms]⟩

/-- **`decR_recJ` at the step**: where `actR` decides, the record it hands over is the reader's. -/
theorem actR_recJ {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    ∃ ms : List Claims.Msg, mids = ms.zip cs.tail.dropLast ∧
      f.j = recJ spec sts link c₁ ms cs.tail.dropLast f.ps :=
  decR_recJ (actR_decide h)

end FlockDeploy

namespace RecDeploy.Rows.DecAt

open Flock
open FlockDeploy (Coin FinalP recJ)

/-- **`decR_recJ` at the verifier's setup**: the record `jAt` reads at a decision is `recJ` at the setup the
statement's read gives at SHA-512 (`setup_eq`). -/
theorem recJ_eq {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
    (D : DecAt rd x ms cs) :
    ∃ ms' : List Claims.Msg, D.d.mids = ms'.zip cs.tail.dropLast ∧
      D.d.f.j = recJ D.spec D.sts D.d.link D.d.c₁ ms' cs.tail.dropLast D.d.f.ps := by
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  rw [hsp, hsts]
  exact FlockDeploy.decR_recJ D.dec

end RecDeploy.Rows.DecAt
