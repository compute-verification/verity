module

public import Proofs.Properties.Verity.SampledTwo.LogStrings
public import Proofs.Properties.Verity.SampledTwo.InputMap
public import Proofs.Properties.Verity.SampledTwo.SecondRead
public import Proofs.Properties.Verity.SampledTwo.RoundsRise
public import Proofs.Properties.Verity.SampledTwo.PrefixFixed
public import Proofs.Properties.Verity.SampledTwo.SessionDev
public import Proofs.Properties.Verity.SampledTwo.DrawRoots
public import Proofs.Properties.Verity.SampledTwo.GoodCase
public import Proofs.Flock.Claims.Sampling.Select.Second
public import Proofs.Flock.Claims.Sampling.Select.OS
public import Proofs.Flock.Bridge.DrawnLaw
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# Round 1 of the two-stage audit's messages

Round 1 of the form's reading reads the interior registration's canonical text. The device's log holds that
registration's digest: the second stage's first coin is released after it. So the digest the play fixes for round 1
is the bytes the release that ends the log before round 1 answers (`Messages.endAnswers`), the string it shows is the
text that digest hashes (`interiorTagged`), and the round's message is that string without its tag.

**The second stage reads a coin.** The check refuses a sampling design with no stratum (`Draw.strataProblem`), and each
stratum's sample reads at least 64 bytes, so an accepted case's second-stage bytes are at least 64
(`VStar.Recursive.SampledTwo.second_source_size`). The log shows those bytes as coins of the second stage's family
(`SecondShown`), so it holds one, and no premise says the design has a stratum.

**The log before round 1 ends at that coin.** No opening of a coin tree comes before the sample's coins
(`sample_before_openings`, from `SessionCoins.readLinkFirst`), and every coin of the second stage's family answers the
interior registration's digest (`CoinMap.StageDraws`). So the first release the locator puts on a round at least 1 is
the family's first coin, on round 1 (`before_one_second`).

* `Properties.Spot.SecurityProofs.second_design_strata`: the second stage's design, as the check parses it, has a
  stratum.
* `Properties.Spot.SecurityProofs.second_sample_size`: its sample reads at least 64 bytes.
* `taggedMsg`, `recordMsg_tagged`: the bytes a record's digest hashes, and the record's message as those bytes after
  the tag.
* `interior_tagged`: an accepted case's interior registration has a tagged text, whose SHA-512 is the digest the second
  stage's coins are released after.
* `showsAt_form_one`, `explains_form_one`: `Messages.ShowsAt` and `Messages.Explains` at round 1 of the form's
  reading.
* `secondShown_of_goodCase`: `SecondShown` at a case, from the case kept on the good plays.

What this does not prove: `Messages.ShowsAt` and `Messages.ExplainsOr` at the statements' round or at a session's round
of the form's reading.
-/

@[expose] public section

section Sampling

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)

/-! ## The second stage's sample reads a coin -/

/-- **A design the check does not refuse has a stratum**: where `Draw.strataProblem` is false, the design's `strata`
are an array that is not empty. -/
theorem strataProblem_strata {law : Json} {n : Int} (h : ¬ Draw.strataProblem law n = true) :
    ∃ ss, getD law "strata" = .arr ss ∧ ss.isEmpty = false := by
  unfold Draw.strataProblem at h
  split at h
  · rename_i ss hss
    refine ⟨ss, hss, ?_⟩
    split at h
    · cases he : ss.isEmpty with
      | false => rfl
      | true =>
        rw [he] at h
        exact absurd rfl h
    · exact absurd rfl h
  · exact absurd rfl h

theorem size_pos_of_isEmpty {α : Type} {a : Array α} (h : a.isEmpty = false) : 0 < a.size := by
  rcases Nat.eq_zero_or_pos a.size with h0 | hpos
  · have h1 : a = #[] := Array.eq_empty_of_size_eq_zero h0
    subst h1
    simp at h
  · exact hpos

theorem forall₂_size_pos {α β : Type} {R : α → β → Prop} {a : Array α} {b : Array β}
    (hF : List.Forall₂ R a.toList b.toList) (hne : a.isEmpty = false) : 0 < b.size := by
  have h1 := hF.length_eq
  rw [Array.length_toList, Array.length_toList] at h1
  have h2 := size_pos_of_isEmpty hne
  omega

/-- **The second stage's design has a stratum**: `Select.secondLaw`'s design, as `Draw.flockLaw` parses it, is the
work design at `k = 1` over at least one stratum, since the check refuses a design with none
(`Draw.strataProblem`). -/
theorem second_design_strata (sample ns ks : Array ℕ) (m : ℕ) (parsed : Flock.Draw.Law)
    (h : Draw.flockLaw (Select.secondLaw sample ns ks) m = .ok parsed) :
    ∃ ws : Array Flock.Draw.WorkStratum, parsed = .work 1 ws ∧ 0 < ws.size := by
  unfold Draw.flockLaw at h
  rw [getD_secondLaw_law] at h
  dsimp only at h
  split at h
  all_goals try (rename_i hs; exact absurd hs (by decide))
  · split at h
    · exact absurd h (throw_bind_ne _ _ _)
    rw [getD_secondLaw_k, getD_secondLaw_strata, nat_toJson] at h
    dsimp only at h
    split at h
    · exact absurd h (throw_bind_ne _ _ _)
    rename_i hsp
    obtain ⟨ss, hss', hne⟩ := strataProblem_strata hsp
    rw [getD_secondLaw_strata] at hss'
    injection hss' with hX
    subst hX
    split at h
    · rename_i strata hss
      have hk : ("work" == "work") = true := by decide
      simp only [hk, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h hss
      subst h
      exact ⟨_, rfl, forall₂_size_pos (mapM_forall₂ _ _ _ hss) hne⟩
    · exact absurd h (throw_ne _ _)

/-- **A work design's first stratum reads at least 64 bytes**: a sample under a work design with a stratum, from
`src` at `p`, ends no earlier than `p + 64`. -/
theorem seqMiss_work_size (k : ℕ) (ws : Array Flock.Draw.WorkStratum) (n : ℕ) (miss : ℕ → Array ℕ → Prop)
    (src : ByteArray) (p : ℕ) (hws : 0 < ws.size) (h : SeqMiss (lawStages (.work k ws) n miss) src p) :
    p + 64 ≤ src.size := by
  obtain ⟨m, hm⟩ : ∃ m, (ws.map (·.toStratum)).size = m + 1 := ⟨ws.size - 1, by rw [Array.size_map]; omega⟩
  have h' : SeqMiss ((List.range' 0 (ws.map (·.toStratum)).size).map fun i =>
      stage (ws.map (·.toStratum))[i]! (miss i)) src p := h
  rw [hm, List.range'_succ, List.map_cons] at h'
  obtain ⟨j, D, ⟨hj, hle, -, -⟩, -, -⟩ := h'
  have hneed : 64 ≤ (stage (ws.map (·.toStratum))[0]! (miss 0)).need := by
    show 64 ≤ (ws.map (·.toStratum))[0]!.need
    unfold Flock.Draw.Stratum.need
    exact Nat.le_add_left _ _
  have := Nat.mul_le_mul hj hneed
  omega

theorem getD_subset_kind (m : ℕ) :
    getD (_root_.VStar.Recursive.SampledTwo.subsetDesign m) "law" = Json.str "subset" := (rfl)

/-- The registration's own design at the last step, a subset of all its proofs, is of kind `subset`. -/
theorem kind_subset (m : ℕ) : kindOf (_root_.VStar.Recursive.SampledTwo.subsetDesign m) = "subset" := by
  simp [kindOf, getD_subset_kind]

/-- **The second stage's sample reads at least 64 bytes**: the sample the coins `src` give under `Select.secondLaw`'s
design, at a verifier whose own design is the subset one, exists only where `src` holds 64 bytes. -/
theorem second_sample_size (v : _root_.Spot.Registration.Verifier) (m : ℕ) (sample ns ks : Array ℕ) (n : ℕ)
    (src : ByteArray) (units : Array ℕ) (read : ℕ)
    (h : _root_.Spot.OneStage.drawnFrom { v with law := _root_.VStar.Recursive.SampledTwo.subsetDesign m }
      (Select.secondLaw sample ns ks) n src = .ok (units, read)) : 64 ≤ src.size := by
  unfold _root_.Spot.OneStage.drawnFrom at h
  obtain ⟨derived, hd, h⟩ := bind_eq_ok.1 h
  obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
  obtain ⟨l, hl, h⟩ := bind_eq_ok.1 h
  obtain rfl := deriveLaw_subset _ _ _ _ _ (kind_subset m) hd
  rw [Option.getD_none] at hl
  obtain ⟨ws, rfl, hws⟩ := second_design_strata sample ns ks n parsed hp
  obtain rfl := (_root_.Flock.Bridge.coveredDesign_ok hl).1
  have hseq := drawUnits_seq _ n src units read h
  have := seqMiss_work_size 1 ws n _ src 0 hws hseq
  omega

end Properties.Spot.SecurityProofs

namespace VStar.Recursive.SampledTwo

open Lean (Json)

/-- **An accepted two-stage case's second stage reads a coin**: where the two-stage check accepts, the run of
record's input holds an interior on the second stage's bytes, and those are at least 64. -/
theorem second_source_size {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧ it.source = ξ₂ ∧
      64 ≤ ξ₂.size := by
  obtain ⟨c₂, ξ₂, r, i, run, law, it, grain, replay, k, ns, ids, ks, hi, -, -, hsc, hit, hsrc, -, -, -, -, -, -,
    hlaw⟩ := accepted_law_run h
  obtain ⟨n, units, read, -, hdf, -⟩ := secondCheck_read hsc
  subst hlaw
  exact ⟨c₂, ξ₂, r, i, it, hi, hit, hsrc,
    Properties.Spot.SecurityProofs.second_sample_size _ _ _ _ _ _ _ _ _ hdf⟩

end VStar.Recursive.SampledTwo

end Sampling

namespace Verity.SampledTwo

open Claims
open Lean (Json)

/-! ## The interior registration's tagged text -/

/-- **A record's tagged text**: the bytes its digest hashes, its own format's tag, a zero byte and its canonical
JSON's bytes (`Flock.WindowRecord.digest`). -/
def taggedMsg (j : Json) : Option Msg :=
  (Flock.canon j).toOption.map fun s =>
    ((Flock.WindowRecord.tagOf j).toUTF8 ++ ByteArray.mk #[0] ++ s.toUTF8).data.toList

/-- Either format's tag is 22 bytes. -/
theorem tag_size (j : Json) : (Flock.WindowRecord.tagOf j).toUTF8.size = 22 := by
  rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]
  unfold Flock.WindowRecord.tagOf
  split
  · decide
  · decide

/-- **A record's message is its tagged text after the tag**: the tag and its zero byte are 23 bytes. -/
theorem recordMsg_tagged (j : Json) : recordMsg j = (taggedMsg j).map fun t => t.drop 23 := by
  unfold recordMsg taggedMsg
  cases Flock.canon j with
  | error err => rfl
  | ok s =>
    show some s.toUTF8.data.toList =
      some (((Flock.WindowRecord.tagOf j).toUTF8 ++ ByteArray.mk #[0] ++ s.toUTF8).data.toList.drop 23)
    have h1 : ((Flock.WindowRecord.tagOf j).toUTF8 ++ ByteArray.mk #[0]).data.toList.length = 23 := by
      have e : ((Flock.WindowRecord.tagOf j).toUTF8 ++ ByteArray.mk #[0]).size = 23 := by
        rw [ByteArray.size_append, tag_size]
        rfl
      rw [Array.length_toList]
      exact e
    have h2 : ((Flock.WindowRecord.tagOf j).toUTF8 ++ ByteArray.mk #[0] ++ s.toUTF8).data.toList =
        ((Flock.WindowRecord.tagOf j).toUTF8 ++ ByteArray.mk #[0]).data.toList ++ s.toUTF8.data.toList := by
      rw [ByteArray.data_append, Array.toList_append]
    rw [h2, ← h1, List.drop_left]

/-- **What a record's digest hashes**: the record has a canonical text, and the digest is `digestOf` its tag and that
text. -/
theorem digest_ok {j : Json} {d : ByteArray} (h : Flock.WindowRecord.digest j = .ok d) :
    ∃ s, Flock.canon j = .ok s ∧ Flock.WindowRecord.digestOf (Flock.WindowRecord.tagOf j) s.toUTF8 = d := by
  unfold Flock.WindowRecord.digest at h
  cases hs : Flock.canon j with
  | error err =>
    rw [hs] at h
    cases h
  | ok s =>
    rw [hs] at h
    exact ⟨s, rfl, Except.ok.inj h⟩

/-- The run of record's interior holds the interior registration's record the form's check computed. -/
theorem inputOf_interior_record (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (ξ ξ₂ : ByteArray) (e : VStar.Recursive.SampledTwo.Evidence) (received₂ : String) (vs : Array (String × Json))
    {it : Spot.Run.Interior} (h : (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it) :
    it.record = e.record := by
  unfold VStar.Recursive.SampledTwo.inputOf at h
  cases h
  rfl

/-- **An accepted case's interior registration, as the log names it**: the run of record's input holds an interior on
at least 64 bytes, its record has a tagged text, and that text's SHA-512 is the digest the second stage's coins are
released after (`secondDraw`). -/
theorem interior_tagged {k : Case} (h : AcceptsOn k) :
    ∃ i it t, inputOfCase k = some i ∧ i.interior = some it ∧ 64 ≤ it.source.size ∧
      taggedMsg it.record = some t ∧
      ∀ label after, secondDraw k = some (label, after) → after.data.toList = Verity.sha512Msg t := by
  obtain ⟨p, cm, e, pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := h
  obtain ⟨c₂, ξ₂, r, i, it, hi, hit, hsrc, hsz⟩ := VStar.Recursive.SampledTwo.second_source_size hacc
  obtain ⟨⟨s, hs, hc₂⟩, -, -, -, vs, rc, j₁, j₂, -, hieq, -, hj₂, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hi
  obtain ⟨-, j₂', hj₂', hdg⟩ := VStar.Recursive.SampledTwo.inputAt_interior hi
  obtain rfl : j₂ = j₂' := Except.ok.inj (hj₂.symm.trans hj₂')
  have hrec : it.record = j₂ := by
    have hit' := hit
    rw [hieq] at hit'
    exact inputOf_interior_record _ _ _ _ _ _ _ hit'
  obtain ⟨s₂, hs₂, hdo⟩ := digest_ok hdg
  refine ⟨i, it, ((Flock.WindowRecord.tagOf j₂).toUTF8 ++ ByteArray.mk #[0] ++ s₂.toUTF8).data.toList, ?_, hit,
    by rw [hsrc]; exact hsz, ?_, ?_⟩
  · unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hi]
    rfl
  · rw [hrec]
    unfold taggedMsg
    rw [hs₂]
    rfl
  · intro label after hsd
    obtain ⟨key, c₂', -, hint, rfl⟩ := secondDraw_parts (c := cm) ⟨h1, h2, h3⟩ hsd
    obtain rfl : c₂ = c₂' := Except.ok.inj ((interiorOf_of hs hc₂).symm.trans hint)
    rw [← hdo]
    rfl

/-- **The interior registration's tagged text on a play**: the tagged text of the record the run of record's interior
holds; none where the play has no such input. -/
def interiorTagged (C : CaseOf) (o : Verity.Exchange) : Option Msg :=
  (inputOn C o).bind fun i => i.interior.bind fun it => taggedMsg it.record

/-- **Round 1's message is the interior registration's tagged text after the tag.** -/
theorem form_msg_one_tagged (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) :
    (formReading A C S dev).msg o 1 = (interiorTagged C o).map fun t => t.drop 23 := by
  rw [form_msg_one A C S dev o]
  unfold interiorTagged
  cases inputOn C o with
  | none => rfl
  | some i =>
    show (i.interior.bind fun it => recordMsg it.record) =
      (i.interior.bind fun it => taggedMsg it.record).map fun t => t.drop 23
    cases i.interior with
    | none => rfl
    | some it => exact recordMsg_tagged it.record

/-! ## The log before round 1 -/

/-- A list that holds an element `P` holds of splits at the first such. -/
theorem first_split {α : Type} (P : α → Prop) (l : List α) (h : ∃ x ∈ l, P x) :
    ∃ pre x post, l = pre ++ x :: post ∧ P x ∧ ∀ y ∈ pre, ¬ P y := by
  induction l with
  | nil =>
    obtain ⟨x, hx, -⟩ := h
    cases hx
  | cons a t ih =>
    by_cases ha : P a
    · exact ⟨[], a, t, rfl, ha, fun y hy => by cases hy⟩
    · obtain ⟨x, hx, hPx⟩ := h
      have hxt : x ∈ t := by
        rcases List.mem_cons.1 hx with rfl | hxt
        · exact absurd hPx ha
        · exact hxt
      obtain ⟨pre, x', post, rfl, hx', hpre⟩ := ih ⟨x, hxt, hPx⟩
      refine ⟨a :: pre, x', post, rfl, hx', fun y hy => ?_⟩
      rcases List.mem_cons.1 hy with rfl | hy'
      · exact ha
      · exact hpre y hy'

theorem isFresh_frame {l : String} {x : Verity.Release} (h : CoinMap.isFresh l x = true) : x.frame = .fresh l := by
  unfold CoinMap.isFresh at h
  split at h
  · rename_i l' hl'
    have hll : l' = l := by simpa using h
    rw [hl', hll]
  · cases h

/-- A log that shows a coin of family `label` released after `after` holds such a release. -/
theorem exists_fresh_of_coins {label : String} {after : ByteArray} {log : List Verity.Release}
    (h : Verity.Sampled.freshCoins label after log ≠ []) :
    ∃ x ∈ log, x.frame = .fresh label ∧ x.answers = after := by
  unfold Verity.Sampled.freshCoins at h
  obtain ⟨cs, hcs⟩ := List.exists_mem_of_ne_nil _ h
  obtain ⟨x, hx, hxc⟩ := List.mem_filterMap.1 hcs
  refine ⟨x, hx, ?_⟩
  cases hf : x.frame with
  | fresh l =>
    rw [hf] at hxc
    by_cases hl : l = label ∧ x.answers = after
    · exact ⟨by rw [hl.1], hl.2⟩
    · simp [hl] at hxc
  | hello sp =>
    rw [hf] at hxc
    simp at hxc
  | opening i s n =>
    rw [hf] at hxc
    simp at hxc

/-- **The second stage's sample comes before every opening**: on an accepted play whose log the check read is the
device's, no release of a coin of the second stage's family after its digest follows an opening of a coin tree
(`SessionCoins.readLinkFirst`, read on the play's log by `SessionCoins.InteriorLogged`). -/
theorem sample_before_openings (C : CaseOf) (Good : Verity.Exchange → Prop) (hRd : SessionCoins.InteriorReads C)
    (hlog : SessionCoins.InteriorLogged C Good) {o : Verity.Exchange} (hA : FormAccepts C o) (hG : Good o)
    {label : String} {after : ByteArray} (hd : (C o).bind secondDraw = some (label, after))
    {a b : List Verity.Release} {r : Verity.Release} (hab : o.log = a ++ r :: b) {i s n : ℕ}
    (hr : r.frame = .opening i s n) : ∀ x ∈ b, ¬ (x.frame = .fresh label ∧ x.answers = after) := by
  intro x hx hxf
  obtain ⟨l, hl⟩ := hRd o hA
  obtain ⟨k, p, e, s', c₂, d, hk, hp, he, hs, hc₂, hsrc, hlink⟩ := SessionCoins.readLinkFirst C o hA l hl
  have hA' := hA
  obtain ⟨k', hk', _, c', _, _, _, ⟨-, hc', -⟩, -⟩ := hA'
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  have hsd : secondDraw k = some (d.label, d.after) := secondDraw_of (c := c') ⟨hp, hc', he⟩ hs hc₂ hsrc
  rw [hk, Option.bind_some, hsd] at hd
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hd)
  obtain ⟨q, l', hql, hq, hF⟩ := hlog o hA hG l hl
  obtain ⟨a', rfl, rfl⟩ := SessionCoins.split_past (hql.symm.trans hab) (fun y hy => (hq y hy).1)
    ((congrArg SessionCoins.openTree hr).trans rfl)
  obtain ⟨la, y, lrest, rfl, -, hy, hrest⟩ :=
    SessionCoins.forall₂_split_right (List.forall₂_map_left_iff.1 hF)
  obtain ⟨b₁, b₂, rfl⟩ := List.append_of_mem hx
  obtain ⟨lb, y', lc, rfl, -, hy', -⟩ := SessionCoins.forall₂_split_right hrest
  have hfy : y.frame = .opening i s n := (SessionCoins.of_ofV2 hy).1.trans hr
  have hfy' : y'.frame = x.frame := (SessionCoins.of_ofV2 hy').1
  obtain ⟨-, hans, -⟩ := hy'
  exact (hlink la y (lb ++ y' :: lc) rfl i s n hfy).1 y' (by simp) ⟨hfy'.trans hxf.1, hans.trans hxf.2⟩

/-- **The log before round 1 ends at the second stage's first coin**: on an accepted play whose log the check read
is the device's, at the run's locator over two families that differ, the first release put on a round at least 1 is
a coin of the second stage's family, and it answers the digest the second stage's coins are released after. The
second stage reads a coin (`interior_tagged`), the log shows it (`SecondShown`), every coin of the family answers the
one digest (`CoinMap.StageDraws`), and no opening comes before it (`sample_before_openings`). -/
theorem before_one_second (A : Live₂) (C : CaseOf) (tab : ℕ → String) (ℓ₁ ℓ₂ : String)
    (Good : Verity.Exchange → Prop) (hne : ℓ₁ ≠ ℓ₂) (hShow : SecondShown C Good)
    (hDraw : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hRd : SessionCoins.InteriorReads C) (hlog : SessionCoins.InteriorLogged C Good) {o : Verity.Exchange}
    (hA : FormAccepts C o) (hG : Good o) :
    ∃ after, (C o).bind secondDraw = some (ℓ₂, after) ∧
      (CoinMap.before (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) 1 [] o.log).2 =
        some (.fresh ℓ₂, after) := by
  have hst : A.h.stage = A.h.more + 1 := rfl
  have hstage : 1 ≤ A.h.stage := by omega
  have hA' := hA
  obtain ⟨k, hk, hacc⟩ := hA'
  obtain ⟨i, it, t, hi, hit, hsz, -, -⟩ := interior_tagged hacc
  have hio : inputOn C o = some i := by
    unfold inputOn
    rw [hk]
    exact hi
  obtain ⟨label, after, hd, hpre⟩ := hShow o hA hG i it hio hit
  obtain ⟨hlab, hone⟩ := hDraw o ⟨hA, hG⟩ label after hd
  rw [hlab] at hd hpre
  refine ⟨after, hd, ?_⟩
  have hcoins : Verity.Sampled.freshCoins ℓ₂ after o.log ≠ [] := by
    intro hnil
    rw [hnil, List.flatten_nil] at hpre
    have hlen := hpre.length_le
    rw [CoinMap.length_bitsOf, List.length_nil] at hlen
    omega
  obtain ⟨x, hx, hxf, -⟩ := exists_fresh_of_coins hcoins
  obtain ⟨pre, rl, post, hsplit, hrf, hpre_no⟩ :=
    first_split (fun y : Verity.Release => y.frame = .fresh ℓ₂) o.log ⟨x, hx, hxf⟩
  have hra : rl.answers = after := hone rl (by rw [hsplit]; simp) hrf
  have hcount : pre.countP (CoinMap.isFresh ℓ₂) = 0 := by
    rw [List.countP_eq_zero]
    intro y hy hfy
    exact hpre_no y hy (isFresh_frame hfy)
  have hj : CoinMap.locAt (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) pre rl.frame rl.answers =
      some 1 := by
    rw [hrf]
    have h0 : CoinMap.stageLocate ℓ₁ ℓ₂ pre (.fresh ℓ₂) rl.answers = some 1 := by
      show (if ℓ₂ = ℓ₁ then some 0
        else if ℓ₂ = ℓ₂ then some (1 + pre.countP (CoinMap.isFresh ℓ₂) / 16384) else none) = some 1
      rw [if_neg (Ne.symm hne), if_pos rfl, hcount]
    show (CoinMap.stageLocate ℓ₁ ℓ₂ pre (.fresh ℓ₂) rl.answers).filter (fun n => decide (n ≤ A.h.stage)) = some 1
    rw [h0, Option.filter_some, if_pos (decide_eq_true hstage)]
  have hq : ∀ a rl₀ b, pre = a ++ rl₀ :: b →
      ¬ ∃ j, CoinMap.locAt (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) a rl₀.frame rl₀.answers =
        some j ∧ 1 ≤ j := by
    intro a rl₀ b hpre_eq ⟨j, hj, hj1⟩
    have hmem : rl₀ ∈ pre := by rw [hpre_eq]; simp
    cases hf : rl₀.frame with
    | hello sp =>
      rw [hf] at hj
      exact absurd hj (by simp [CoinMap.locAt])
    | opening i s n =>
      have hlogeq : o.log = a ++ rl₀ :: (b ++ rl :: post) := by rw [hsplit, hpre_eq]; simp
      exact sample_before_openings C Good hRd hlog hA hG hd hlogeq hf rl (by simp) ⟨hrf, hra⟩
    | fresh l =>
      rw [hf] at hj
      have hl : l ≠ ℓ₂ := fun h => hpre_no rl₀ hmem (by rw [hf, h])
      have hj' : (if l = ℓ₁ then some 0
          else if l = ℓ₂ then some (1 + a.countP (CoinMap.isFresh ℓ₂) / 16384) else none).filter
            (fun n => decide (n ≤ A.h.stage)) = some j := hj
      rw [if_neg hl] at hj'
      by_cases hl1 : l = ℓ₁
      · rw [if_pos hl1, Option.filter_some] at hj'
        split at hj'
        · have := Option.some.inj hj'
          omega
        · cases hj'
      · rw [if_neg hl1] at hj'
        simp at hj'
  have hb := CoinMap.before_first (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (rest := post) hq hj
    (Nat.le_refl 1)
  rw [hsplit, hb]
  show some (rl.frame, rl.answers) = some (Verity.Frame.fresh ℓ₂, after)
  rw [hrf, hra]

/-! ## `ShowsAt` and `Explains` at round 1 -/

/-- **Round 1 of the form's reading shows the interior registration's tagged text**: where the digest a play fixes
for round 1 is the bytes the release that ends the log before round 1 answers (`Messages.endAnswers`), at the run's
locator, and the string it shows is the interior registration's tagged text (`interiorTagged`), an accepted play
whose log the check read is the device's shows a string that has its digest. -/
theorem showsAt_form_one (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (tab : ℕ → String) (ℓ₁ ℓ₂ : String)
    (Good : Verity.Exchange → Prop) (hne : ℓ₁ ≠ ℓ₂) (hShow : SecondShown C Good)
    (hDraw : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hRd : SessionCoins.InteriorReads C) (hlog : SessionCoins.InteriorLogged C Good)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (hd : ∀ o, FormAccepts C o → Good o → Dg o 1 = Messages.endAnswers
      (Messages.logBefore (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) o 1))
    (hr : ∀ o, FormAccepts C o → Good o → raw o 1 = (interiorTagged C o).map fun t => [t]) :
    Messages.ShowsAt (formReading A C S dev) (FormAccepts C) Good Dg raw 1 := by
  intro o hA hG
  have hA' := hA
  obtain ⟨k, hk, hacc⟩ := hA'
  obtain ⟨i, it, t, hi, hit, -, ht, hdig⟩ := interior_tagged hacc
  obtain ⟨after, hsd, hb⟩ := before_one_second A C tab ℓ₁ ℓ₂ Good hne hShow hDraw hRd hlog hA hG
  rw [hk, Option.bind_some] at hsd
  have hafter := hdig ℓ₂ after hsd
  have htag : interiorTagged C o = some t := by
    simp only [interiorTagged, inputOn, hk, Option.bind_some, hi, hit, ht]
  refine ⟨⟨t.drop 23,
    (form_msg_one_tagged A C S dev o).trans (congrArg (Option.map fun t : Msg => t.drop 23) htag)⟩,
    [after.data.toList], [t], ?_, ?_, ?_⟩
  · exact (hd o hA hG).trans
      (congrArg (Option.map fun fb : Verity.Frame × ByteArray => [fb.2.data.toList]) hb)
  · exact (hr o hA hG).trans (congrArg (Option.map fun t : Msg => [t]) htag)
  · exact congrArg (fun m : Msg => [m]) hafter.symm

/-- **Round 1 of the form's reading is explained by the interior registration's tagged text**, at any context: two
accepted plays that show the same tagged text read the same message. -/
theorem explains_form_one {κ : Type} (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ)
    (hr : ∀ o, Acc o → Good o → raw o 1 = (interiorTagged C o).map fun t => [t]) :
    Messages.Explains (formReading A C S dev) Acc Good raw ctx 1 := by
  intro o o' hA hG hA' hG' _ hraw
  rw [hr o hA hG, hr o' hA' hG'] at hraw
  have hinj : Function.Injective fun t : Msg => [t] := fun _ _ h => (List.cons.inj h).1
  have ht : interiorTagged C o = interiorTagged C o' := Option.map_injective hinj hraw
  rw [form_msg_one_tagged A C S dev o, form_msg_one_tagged A C S dev o', ht]

/-! ## The join's facts, at the case itself -/

/-- **`SecondShown` at a case, from the case kept on the good plays**: where the kept case's plays show the second
stage's bytes at `G'`, and `G` gives `G'`, the case's plays `G` holds of show them. -/
theorem secondShown_of_goodCase {C : CaseOf} {G G' : Verity.Exchange → Prop} (hG : ∀ o, G o → G' o)
    (h : SecondShown (goodCase C G) G') : SecondShown C G := by
  intro o hA hg i it hi hit
  have h' := h o (formAccepts_goodCase.2 ⟨hA, hg⟩) (hG o hg) i it ((inputOn_goodCase hg).trans hi) hit
  rw [goodCase_of hg] at h'
  exact h'

end Verity.SampledTwo
