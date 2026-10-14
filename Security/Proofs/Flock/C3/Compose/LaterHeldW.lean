import Proofs.Flock.C3.Compose.HoloSparseWords

/-!
# `LaterCoins` from the inner session's records, word by word

`StageHeldW`: a stage whose coin steps are one stream's words, one step a word, where a stream round takes as many of
the history's rounds as it holds words (a sparse phase's ζ_c is one round of six coins); `coinsAt_stageHeldW` gives its
`CoinsAt` from custody's coins half and `CoinWordsEncoded` (at `holoMap`, `coinWordsEncoded_holoMap` at
`holoSelWords`). `StageHeld` is the case of one history round a stream round.
`laterCoins_of_held`: `LaterCoins` at a run whose link points are the record's, whose sparse phases are each
`StageHeldW` and whose R stages are each `StageHeld`, each from where the one before ends.
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness.Game (Msg Step)
open FlockC3.Compose (CoinsAt coinAt_congr)
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Assumptions (InnerRecordMap heldCoins InnerRecordCustody)

/-- **Element `n` of a flattened list is word `w` of its `p`-th list**, past the words of the lists before it. -/
theorem flatten_map_idx {α β : Type} (f : β → List α) : ∀ (ks : List β) (n : ℕ),
    n < (ks.map f).flatten.length → ∃ (p : ℕ) (q : β) (w : ℕ), ks[p]? = some q ∧ ∃ hw : w < (f q).length,
      n = ((ks.take p).map fun q => (f q).length).sum + w ∧
      ((ks.take p).map fun q => (f q).length).sum + (f q).length ≤ (ks.map f).flatten.length ∧
      (ks.map f).flatten[n]? = some (f q)[w]
  | [], n, hn => absurd hn (by simp)
  | k :: ks, n, hn => by
    rw [List.map_cons, List.flatten_cons, List.length_append] at hn
    by_cases h : n < (f k).length
    · refine ⟨0, k, n, rfl, h, by simp, ?_, ?_⟩
      · rw [List.take_zero, List.map_nil, List.sum_nil, List.map_cons, List.flatten_cons, List.length_append]
        omega
      · rw [List.map_cons, List.flatten_cons, List.getElem?_append_left h, List.getElem?_eq_getElem h]
    · obtain ⟨p, q, w, hq, hw, he, hb, hx⟩ := flatten_map_idx f ks (n - (f k).length) (by omega)
      refine ⟨p + 1, q, w, by rw [List.getElem?_cons_succ]; exact hq, hw, ?_, ?_, ?_⟩
      · rw [List.take_succ_cons, List.map_cons, List.sum_cons]
        omega
      · rw [List.take_succ_cons, List.map_cons, List.sum_cons, List.map_cons, List.flatten_cons,
          List.length_append]
        omega
      · rw [List.map_cons, List.flatten_cons, List.getElem?_append_right (by omega)]
        exact hx

theorem map_eq_range {α β : Type} (h : α → β) (d : α) (F : List α) :
    F.map h = (List.range F.length).map fun n => h ((F[n]?).getD d) := by
  refine List.ext_getElem (by simp) fun n h₁ _ => ?_
  have hn : n < F.length := by simpa using h₁
  simp only [List.getElem_map, List.getElem_range, List.getElem?_eq_getElem hn, Option.getD_some]

variable {r : ℕ} {CT : Fin r → Type}

/-- **Hypothesis: a stage's coins are one stream's words, drawn in order from the history's round `j` on**: its coin
steps are, one a word, the words of the coins `g q` the verifier holds of a stream `s` at round `ix q`, `q` over `ks`;
the `p`-th's words take the history's rounds from `j` past the words before it, the last of them drawing round `ix q`
of `s` (`selC`), and all of them are below `r`. -/
def StageHeldW {Cm : Type} (t : List (Cm × ((k : Fin r) → CT k)))
    (wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String)
    (held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits) (selC : ℕ → String → ℕ → Prop)
    (j : ℕ) (ts : List Step) : Prop :=
  ∃ (ks : List ℕ) (g : ℕ → Array Flock.F128) (s : String) (ix : ℕ → ℕ),
    ts.filterMap Sum.getRight? =
        (ks.map fun q => (g q).toList).flatten.map (fun x => coinArr FlockSoundness.Refine.fOf #[x]) ∧
      (∀ q c, heldCoins (held (wr t)) s (ix q) = some c → g q = c) ∧
      j + (ks.map fun q => (g q).toList).flatten.length ≤ r ∧
      ∀ p q, ks[p]? = some q → 0 < (g q).size →
        selC (j + ((ks.take p).map fun q => (g q).size).sum + (g q).size - 1) s (ix q)

/-- **A word-level stage's coins are the history's from round `j` on**: its `n`-th coin step is word `w` of the
`p`-th coins it takes (`flatten_map_idx`), held at the round `i` that draws them, so custody's coins half reads it off
the coins up to `i`, where it is the coin of round `i + 1 - (g q).size + w = j + n` (`CoinWordsEncoded`). -/
theorem coinsAt_stageHeldW {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))} {ts : List Step}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : InnerRecordMap Cm ((k : Fin r) → CT k)} {sel : ℕ → String → ℕ → Prop} {j : ℕ}
    (hcus : InnerRecordCustody wr held sparse m) (henc : CoinWordsEncoded sel m.coin)
    (hsel : ∀ i s q, sel i s q → m.selC i s q) (hlen : r + 1 ≤ t.length)
    (h : StageHeldW t wr held sel j ts) : CoinsAt (t.map Prod.snd) j (nCoins ts) ts := by
  obtain ⟨ks, g, s, ix, hts, hd, hr, hdr⟩ := h
  have hts' := hts.trans (map_eq_range (fun x => coinArr FlockSoundness.Refine.fOf #[x]) default
    (ks.map fun q => (g q).toList).flatten)
  refine (coinsAt_of_idx hts' (by rw [List.length_range]; exact hr)
    (by rw [List.length_range, List.length_map]; omega) ?_).2
  intro n n' i hi hc hn hin
  obtain ⟨hlt, he⟩ := List.getElem?_eq_some_iff.1 hn
  rw [List.getElem_range] at he
  rw [← he]
  rw [List.length_range] at hlt
  obtain ⟨p, q, w, hq, hw, hoff, hbd, hx⟩ := flatten_map_idx (fun q => (g q).toList) ks n hlt
  rw [Array.getElem_toList] at hx
  simp only [Array.length_toList] at hw hoff hbd
  have hpos : 0 < (g q).size := by omega
  have hs := hdr p q hq hpos
  have hi₀ : j + ((ks.take p).map fun q => (g q).size).sum + (g q).size - 1 < r := by omega
  have hc₀ : j + ((ks.take p).map fun q => (g q).size).sum + (g q).size - 1 < t.length := by omega
  have hgq := hd q _ (((hcus t).1 _ _ (List.getElem?_eq_getElem hc₀)).2 s (ix q) (hsel _ s (ix q) hs))
  rw [List.map_take] at hgq
  obtain ⟨g', hg', hsz, hwd⟩ := henc _ hi₀ s (ix q) (t.map Prod.snd) (by rw [List.length_map]; exact hc₀) hs
  have hsz' : (g q).size = g' := by rw [hgq]; exact hsz
  show coinArr FlockSoundness.Refine.fOf #[(((ks.map fun q => (g q).toList).flatten)[n]?).getD default] = _
  rw [hx, Option.getD_some, ← getElem!_pos (g q) w hw, hgq]
  exact (hwd w (by omega)).trans (coinAt_congr (t.map Prod.snd) (by omega) _ _ _ _)

end

end FlockC3.Reps

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (nCoins CoinsEncoded CoinWordsEncoded StageHeld StageHeldW PointsEncoded)

/-- **`LaterCoins` from the inner session's records**: the link points are the session record's (`hdec`, `hu`, at
`PointsEncoded`), each sparse phase is one stream's words, drawn in order from the round after the stage before it
(`StageHeldW`, at the map's word-level selector `selW`), and R's opening and R's Ligerito are each one stream's rounds
(`StageHeld`), R's opening after the sparse phases and R's Ligerito after the opening. -/
theorem laterCoins_of_held {r : ℕ} {CT : Fin r → Type} {e : ExecHolo} {Cm : Type}
    {t : List (Cm × ((k : Fin r) → CT k))} {u : C3Run e}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin r) → CT k)} {sel selW : ℕ → String → ℕ → Prop}
    {pts : Array (Array Flock.F128) → Fin 2 → Fin e.cls.mPts → GF128} {spec : Flock.SessionSpec}
    {sess : Flock.Session}
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m) (henc : CoinsEncoded sel m.coin)
    (hsel : ∀ i s q, sel i s q → m.selC i s q) (hencW : CoinWordsEncoded selW m.coin)
    (hselW : ∀ i s q, selW i s q → m.selC i s q)
    (hpe : PointsEncoded pts m.points) (hdec : Flock.Record.decode spec (wr t).2 = .ok sess)
    (hu : pts sess.points = u.pts) (hr : 0 < r) (hlen : r + 1 ≤ t.length)
    (h2 : StageHeldW t wr held selW (1 + nCoins (u.ts 0) + nCoins (u.ts 1)) (u.sp 0))
    (h3 : StageHeldW t wr held selW (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0)) (u.sp 1))
    (h4 : StageHeld t wr held sel (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1))
      u.op)
    (h5 : StageHeld t wr held sel
      (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op) u.lg) :
    LaterCoins (t.map Prod.snd) u :=
  ⟨FlockC3.Reps.linkPoints_of_custody hcus hpe hdec hu hr (by omega),
    FlockC3.Reps.coinsAt_stageHeldW hcus hencW hselW hlen h2, FlockC3.Reps.coinsAt_stageHeldW hcus hencW hselW hlen h3,
    FlockC3.Reps.coinsAt_stageHeld hcus henc hsel hlen h4, FlockC3.Reps.coinsAt_stageHeld hcus henc hsel hlen h5⟩

end

end FlockC3.Compose
