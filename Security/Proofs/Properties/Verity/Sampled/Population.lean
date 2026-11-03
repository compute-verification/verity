import Batteries.Tactic.OpenPrivate
import Proofs.Properties.Verity.SampledRead
import Proofs.Properties.Verity.Sampled.Messages
import Proofs.Properties.Verity.Soundness
import Proofs.Properties.NetworkRecords.Bytes
import Proofs.Flock.Soundness.Refine.Walk
import Proofs.Flock.Soundness.Refine.Setup

/-!
# Sampled proofs' population: the session's is the draw's

At the session `A` v2 sets up on the parameters `θ` and the committed files `F` (`SetsUp`), the population v2's
accepted draw runs over (`Drew`'s `n`, the record's `population`) is the session's (`A.n`), on every play of the
period's game whose sampled verdict accepts (`population_eq`, `population_eq_run` on its tapes), wherever the session's
public file is the one v2's parameters name by its SHA-512 (`p.own.pub`). So the draw's piece of `CouplingTarget`
holds with no hypothesis on the population (`coupled_draw_held`, `coupled_draw_coins_held`).

* **The blob v2 holds is the committed public file** (`checkBlobsAt_public`, `blobs_get_ok`, `find?_first`,
  `namedFile_first`): v2 holds the first blob sent whose SHA-512 is `p.own.pub` to the record; the blobs sent begin with
  the committed files, and the public file (`namedFile`) is the first committed file with the parameters' digest, so
  the first with its SHA-512 is itself. No SHA-512 collision enters.
* **Its header's `instances` is the population** (`publicHeld_ok`, `matchesPublic_population`, `headerOf_ok`,
  `headerLine_of_findByte`, `loadPublic_instances`): v2 reads the header line as `HmRow.loadPublic` does (`Flock.norm`
  of `Json.parse` of the bytes before the first newline), holds its `instances` equal to the record's `population`
  (`jeq`), and `loadPublic`'s `n` is that `instances`, so it is the record's population (`nat_of_jeq`).
-/

open private headerLine headerOf from SampledProofs.OneStageZk

namespace Verity.Sampled

open Lean (Json)
open Verity.Game
open FlockSoundness.Refine
open Properties.Challenger (bitsOf Draws)
open scoped ENNReal

/-! ## The blob v2 holds -/

/-- **What v2's check at the coins `ξ` holds of the public file**: the blob named by the parameters' public digest, held
to the record. -/
theorem checkBlobsAt_public {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {ξ : ByteArray} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {bs : Core.Blobs} {pub : _root_.SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (h : _root_.SampledProofs.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ b, bs.get p.own.pub "public file" = .ok b ∧
      _root_.SampledProofs.OneStageZk.publicHeld c.record b = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.checkBlobsAt at h
  try simp only [] at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨b, hb, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, hu, -⟩ := ebind_ok hr
    exact ⟨b, hb, hu⟩

/-- **A blob found by its digest**: the first of the blobs whose SHA-512 is the digest the hex string spells. -/
theorem blobs_get_ok {bs : Array ByteArray} {dg w : String} {b : ByteArray}
    (h : (Core.Blobs.ofArray bs).get dg w = .ok b) :
    ∃ d, Flock.hexOf d = dg ∧ bs.find? (fun f => Flock.Sha512.hash f == d) = some b := by
  have hd : ∀ {x : ByteArray}, ¬((x.size != 64 || Flock.hexOf x != dg) = true) → Flock.hexOf x = dg := by
    intro x hx
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hx
    exact hx.2
  have hb : ∀ {x k b' : ByteArray},
      (bs.map fun f => ((Flock.Sha512.hash f, f) : ByteArray × ByteArray)).find? (fun e => e.1 == x) =
        some (k, b') → bs.find? (fun f => Flock.Sha512.hash f == x) = some b' := by
    intro x k b' he
    rw [Array.find?_map, Option.map_eq_some_iff] at he
    obtain ⟨a, ha, he⟩ := he
    simp only [Prod.mk.injEq] at he
    obtain ⟨-, rfl⟩ := he
    exact ha
  unfold Core.Blobs.get Core.Blobs.ofArray at h
  try dsimp only at h
  try simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, hd ‹_›, hb ‹_›⟩

/-- **The first element `q` picks** is the first `r` picks, where `r` holds wherever `q` does and `q` holds there. -/
theorem find?_first {β : Type} {l : List β} {q r : β → Bool} {a : β} (hr : l.find? r = some a) (hq : q a = true)
    (h : ∀ x, q x = true → r x = true) : l.find? q = some a := by
  induction l with
  | nil => cases hr
  | cons x l ih =>
    rw [List.find?_cons] at hr ⊢
    cases hx : r x
    · rw [hx] at hr
      have hqx : q x = false := by
        cases hq' : q x
        · rfl
        · exact absurd (h x hq') (by rw [hx]; exact Bool.false_ne_true)
      rw [hqx]
      exact ih hr
    · rw [hx] at hr
      cases hr
      rw [hq]

/-- **The committed file a digest names** (`namedFile`), where one has it: it is one of them with the digest, and the
first that any `q` picks wherever `q` picks only files with the digest and picks it. -/
theorem namedFile_first {h : Option Hash.Digest} {F : Array ByteArray}
    (hF : ∃ f ∈ F, h = some (sha512 f.data.toList)) :
    namedFile h F ∈ F ∧ h = some (sha512 (namedFile h F).data.toList) ∧
      ∀ q : ByteArray → Bool, q (namedFile h F) = true → (∀ f, q f = true → h = some (sha512 f.data.toList)) →
        F.find? q = some (namedFile h F) := by
  obtain ⟨f, hf, hfh⟩ := hF
  unfold namedFile
  have hs : (F.find? fun f => decide (h = some (sha512 f.data.toList))).isSome := by
    rw [Array.find?_isSome]
    exact ⟨f, hf, by simpa using hfh⟩
  obtain ⟨P, hP⟩ := Option.isSome_iff_exists.1 hs
  rw [hP, Option.getD_some]
  have hmem := Array.mem_of_find?_eq_some hP
  have hPh := Array.find?_some hP
  simp only [decide_eq_true_eq] at hPh
  refine ⟨hmem, hPh, fun q hq himp => ?_⟩
  rw [← Array.find?_toList] at hP ⊢
  exact find?_first hP hq fun x hx => by simpa using himp x hx

theorem bytes_toList_toArray (g : ByteArray) : (⟨g.data.toList.toArray⟩ : ByteArray) = g := by
  rw [Array.toArray_toList]

/-- Two files with one SHA-512 have one digest. -/
theorem sha512_of_hash {f g : ByteArray} (h : Flock.Sha512.hash f = Flock.Sha512.hash g) :
    sha512 f.data.toList = sha512 g.data.toList := by
  unfold sha512
  rw [bytes_toList_toArray, bytes_toList_toArray, h]

/-! ## The header's `instances` -/

theorem fail_codes (c : _root_.SampledProofs.Check) (code : String) (w : Json) :
    (c.fail code w).codes.isEmpty = false := by
  unfold _root_.SampledProofs.Check.fail
  split
  · rename_i hc
    rw [Array.isEmpty_eq_false_iff_exists_mem]
    exact ⟨code, Array.mem_of_contains_eq_true hc⟩
  · simp

theorem fail_not_empty (c : _root_.SampledProofs.Check) (code : String) (w : Json) :
    ¬ (c.fail code w).codes.isEmpty = true := by
  rw [fail_codes]
  exact Bool.false_ne_true

/-- **A public file that matches the registration has its population**: `matchesPublic`'s codes are empty only where
the population matches the record's. -/
theorem matchesPublic_population {rec scheme roots domains pop : Json}
    (h : (_root_.SampledProofs.Record.matchesPublic rec scheme roots domains pop).codes.isEmpty = true) :
    _root_.SampledProofs.jeq pop (_root_.SampledProofs.getD rec "population") = true := by
  by_contra hne
  rw [Bool.not_eq_true] at hne
  unfold _root_.SampledProofs.Record.matchesPublic at h
  simp only [Id.run, hne, Bool.not_false, ↓reduceIte] at h
  revert h
  split_ifs <;> exact fail_not_empty _ _ _

/-- **v2 holds the public file's `instances` to the record's `population`**, in the header it reads. -/
theorem publicHeld_ok {rec : Json} {b : ByteArray}
    (h : _root_.SampledProofs.OneStageZk.publicHeld rec b = .ok ()) :
    ∃ H, headerOf b = .ok H ∧
      _root_.SampledProofs.jeq (_root_.SampledProofs.getD H "instances")
        (_root_.SampledProofs.getD rec "population") = true := by
  unfold _root_.SampledProofs.OneStageZk.publicHeld at h
  obtain ⟨H, hH, h⟩ := ebind_ok h
  refine ⟨H, hH, matchesPublic_population (scheme := _root_.SampledProofs.getD H "scheme")
    (roots := _root_.SampledProofs.getD (_root_.SampledProofs.getD H "frame_v3") "roots")
    (domains := _root_.SampledProofs.getD (_root_.SampledProofs.getD H "frame_v3") "domain_ids") ?_⟩
  dsimp only at h
  by_contra hc
  rw [Bool.not_eq_true] at hc
  simp only [hc, Bool.not_false, ↓reduceIte] at h
  exact absurd h (throw_ne_ok _ _)

/-- **The header v2 reads**: `Flock.norm` of the parse of its header line. -/
theorem headerOf_ok {b : ByteArray} {H : Json} (h : headerOf b = .ok H) :
    ∃ line J, String.fromUTF8? (headerLine b) = some line ∧ Json.parse line = .ok J ∧ H = Flock.norm J := by
  unfold headerOf at h
  split at h
  · rename_i line hl
    cases hp : Json.parse line with
    | error e => simp [hp, Except.mapError, Functor.map, Except.map] at h
    | ok J =>
      simp [hp, Except.mapError, Functor.map, Except.map] at h
      exact ⟨line, J, hl, hp, h.symm⟩
  · exact absurd h (throw_ne_ok _ _)

theorem getElem!_eq_get! (b : ByteArray) (i : ℕ) : b[i]! = b.get! i := by
  by_cases hi : i < b.size
  · rw [getElem!_pos b i hi]
    show b.data[i] = b.data[i]!
    rw [getElem!_pos b.data i hi]
  · rw [getElem!_neg b i hi]
    show default = b.data[i]!
    rw [getElem!_neg b.data i hi]

/-- **v2's header line is `loadPublic`'s**: the bytes before the first newline. -/
theorem headerLine_of_findByte {b : ByteArray} {nl : ℕ} (h : Flock.findByte b 10 = some nl) :
    headerLine b = b.extract 0 nl := by
  unfold headerLine
  unfold Flock.findByte at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Id.run, getElem!_eq_get!] at h ⊢
  generalize List.range' _ _ _ = l at h ⊢
  induction l with
  | nil => simp at h
  | cons i l ih =>
    by_cases hi : (b.get! i == 10) = true
    · simp only [List.forIn_cons, hi, ↓reduceIte, pure_bind] at h ⊢
      have h' : some i = some nl := h
      cases h'
      rfl
    · simp only [List.forIn_cons, hi] at h ⊢
      exact ih h

private theorem getNat_ok {j : Json} {n : ℕ} (h : j.getNat? = .ok n) : j = .num ⟨Int.ofNat n, 0⟩ := by
  unfold Json.getNat? at h
  split at h
  · cases h; rfl
  · cases h

private theorem asNat_ok {j : Json} {c : String} {n : ℕ} (h : Flock.J.asNat j c = .ok n) :
    j = .num ⟨Int.ofNat n, 0⟩ := by
  unfold Flock.J.asNat at h
  split at h
  · rename_i _ hg; cases h; exact getNat_ok hg
  · cases h

/-- A key `J.nat` reads is that number. -/
theorem getD_of_nat {H : Json} {k ctx : String} {m : ℕ} (h : Flock.J.nat H k ctx = .ok m) :
    _root_.SampledProofs.getD H k = .num ⟨Int.ofNat m, 0⟩ := by
  unfold Flock.J.nat at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [← asNat_ok h]
  unfold Flock.J.get at hv
  unfold _root_.SampledProofs.getD _root_.SampledProofs.field?
  split at hv
  · rename_i v' hg
    cases hv
    rw [hg]
    rfl
  · cases hv

set_option maxHeartbeats 4000000 in
/-- **`loadPublic`'s population is its header's `instances`**, in the header it parses from the bytes before the first
newline. -/
theorem loadPublic_instances {tags : Flock.Tags} {c : Flock.Circuit} {f : ByteArray} {pop : Flock.Public}
    (h : Flock.HmRow.loadPublic tags c f = .ok pop) :
    ∃ nl text J, Flock.findByte f 10 = some nl ∧ String.fromUTF8? (f.extract 0 nl) = some text ∧
      Json.parse text = .ok J ∧ Flock.J.nat (Flock.norm J) "instances" "public file" = .ok pop.n := by
  unfold Flock.HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hpj := ‹(pure _ : Flock.V Json) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hpj
    subst hpj
    first
    | exact ⟨_, _, _, ‹_›, ‹_›, ‹_›, ‹_›⟩
    | (have hps := ‹(pure _ : Flock.V String) = Except.ok _›
       simp only [pure, Except.pure, Except.ok.injEq] at hps
       subst hps
       exact ⟨_, _, _, ‹_›, ‹_›, ‹_›, ‹_›⟩)

/-- `HmIn.loadPublic` is `HmRow.loadPublic`'s file, with checks. -/
theorem hmIn_loadPublic {tags : Flock.Tags} {c : Flock.Circuit} {pi : Flock.HmIn.Ports} {f : ByteArray}
    {pop : Flock.Public} (h : Flock.HmIn.loadPublic tags c pi f = .ok pop) :
    ∃ c', Flock.HmRow.loadPublic tags c' f = .ok pop := by
  unfold Flock.HmIn.loadPublic at h
  try dsimp only at h
  try simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, by assumption⟩

private theorem toString_inj {n n' : ℕ} (h : toString n = toString n') : n = n' := by
  have e := congrArg String.toList h
  rw [Nat.toString_eq_repr, Nat.toString_eq_repr, Nat.toList_repr, Nat.toList_repr] at e
  rw [← Nat.ofDigitChars_ten_toDigits (n := n), e, Nat.ofDigitChars_ten_toDigits]

/-- **A natural number `jeq` to a JSON natural number is it.** -/
theorem nat_of_jeq {m n : ℕ} {w : Json} (hj : _root_.SampledProofs.jeq (.num ⟨Int.ofNat m, 0⟩) w = true)
    (hn : _root_.SampledProofs.nat? w = some n) : m = n := by
  cases w with
  | num x =>
    obtain ⟨i, e⟩ := x
    by_cases he : e = 0
    · subst he
      simp only [_root_.SampledProofs.nat?, _root_.SampledProofs.int?, beq_self_eq_true, ↓reduceIte] at hn
      split_ifs at hn with hi
      cases hn
      obtain ⟨k, rfl⟩ := Int.eq_ofNat_of_zero_le hi
      simp only [_root_.SampledProofs.jeq, Flock.canon, beq_self_eq_true, ↓reduceIte, pure, Except.pure,
        beq_iff_eq] at hj
      have hk : toString m = toString k := hj
      rw [toString_inj hk]
      rfl
    · simp [_root_.SampledProofs.nat?, _root_.SampledProofs.int?, he] at hn
  | _ => simp [_root_.SampledProofs.nat?, _root_.SampledProofs.int?] at hn

/-! ## The population -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The session's population is the draw's**: at the session `A` v2 sets up on the parameters `θ` and the committed
files `F`, whose public file is the one v2's parameters name by its SHA-512, a draw v2 made in a play on `θ` and `F`
whose sampled verdict accepts runs over `A.n` units. -/
theorem population_eq {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    (hpub : ∃ p, ParamsAt θ p ∧ Flock.hexOf (Flock.Sha512.hash A.I.publicFile) = p.own.pub)
    {k : Verity.Registration.Composite} {o : Verity.Exchange} (hθ : o.params = θ) (hF : o.files = F)
    (hv : V.Verdict k .sampled o) {dr : Draws} {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray} {units : Array ℕ}
    (hd : Drew k o.sent dr L n ξ units) : A.n = n := by
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, hL, -, -⟩ := hd
  obtain ⟨p', hp', hpp⟩ := hpub
  have hp'' : ParamsAt o.sent.params p' := by
    show ParamsAt o.params p'
    rw [hθ]
    exact hp'
  rw [paramsAt_case hv.2.1 hcase hacc hp''] at hpp
  obtain ⟨-, -, -, -, -, pub, prof, hck⟩ := hacc
  obtain ⟨blob, hget, hheld⟩ := checkBlobsAt_public hck
  obtain ⟨dg, hdg, hfind⟩ := blobs_get_ok hget
  rw [← hpp] at hdg
  have hdg' := Properties.NetworkRecords.Bytes.hexOf_inj hdg
  subst hdg'
  have hreg : ∃ f ∈ F, V.publicDigest θ = some (sha512 f.data.toList) := by
    have h2 := hv.1.2
    rw [hθ, hF] at h2
    exact h2
  obtain ⟨-, hPd, hfirst⟩ := namedFile_first hreg
  have hPA : namedFile (V.publicDigest θ) F = A.I.publicFile := hA.publicFile.symm
  rw [hPA] at hPd hfirst
  have hq := hfirst (fun f => Flock.Sha512.hash f == Flock.Sha512.hash A.I.publicFile) (beq_bytes.2 rfl)
    (fun f hf => by
      rw [hPd]
      exact congrArg some (sha512_of_hash (beq_bytes.1 hf)).symm)
  have hblob : blob = A.I.publicFile := by
    change (o.files ++ o.blobs).find? _ = some blob at hfind
    rw [hF, Array.find?_append, hq, Option.some_or] at hfind
    exact (Option.some.inj hfind).symm
  subst hblob
  obtain ⟨H, hH, hjeq⟩ := publicHeld_ok hheld
  obtain ⟨line, J, hline, hJ, rfl⟩ := headerOf_ok hH
  obtain ⟨c', hc'⟩ := hmIn_loadPublic A.hpub
  obtain ⟨nl, text, J', hnl, htext, hJ', hnat⟩ := loadPublic_instances hc'
  rw [headerLine_of_findByte hnl, htext] at hline
  cases hline
  rw [hJ] at hJ'
  cases hJ'
  rw [getD_of_nat hnat] at hjeq
  obtain ⟨-, -, -, -, hpop, -⟩ := hL
  rw [← A.hn]
  exact nat_of_jeq hjeq hpop

/-- Every play of the period's game ends on its parameters and its committed files. -/
theorem run_exchange (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ) :
    (CoinTape.run (V.game θ F c) σ x).params = θ ∧ (CoinTape.run (V.game θ F c) σ x).files = F := by
  by_contra hx
  have h0 : Game.prob (fun o : Exchange => ¬ (o.params = θ ∧ o.files = F)) (V.game θ F c) σ = 0 :=
    prob_play_eq_zero (V.families θ) θ F c _ (fun _ _ h => h ⟨rfl, rfl⟩) σ
  rw [prob_eq_ofReal_pr] at h0
  have hmem : x ∈ Finset.univ.filter fun y : CoinTape (V.game θ F c) σ =>
      ¬ ((CoinTape.run (V.game θ F c) σ y).params = θ ∧ (CoinTape.run (V.game θ F c) σ y).files = F) :=
    Finset.mem_filter.2 ⟨Finset.mem_univ _, hx⟩
  have hpos : 0 < Claims.Pr fun y : CoinTape (V.game θ F c) σ =>
      ¬ ((CoinTape.run (V.game θ F c) σ y).params = θ ∧ (CoinTape.run (V.game θ F c) σ y).files = F) := by
    unfold Claims.Pr
    apply div_pos
    · rw [Nat.cast_pos, Finset.card_pos]
      exact ⟨x, by convert hmem⟩
    · exact_mod_cast Fintype.card_pos
  exact absurd (ENNReal.ofReal_eq_zero.1 h0) (not_le.2 hpos)

/-- `population_eq` on the tapes of the period's game. -/
theorem population_eq_run {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    (hpub : ∃ p, ParamsAt θ p ∧ Flock.hexOf (Flock.Sha512.hash A.I.publicFile) = p.own.pub)
    {k : Verity.Registration.Composite} {c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (x : CoinTape (V.game θ F c) σ) (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x)) {dr : Draws}
    {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray} {units : Array ℕ}
    (hd : Drew k (CoinTape.run (V.game θ F c) σ x).sent dr L n ξ units) : A.n = n :=
  population_eq hA hpub (run_exchange V θ F c σ x).1 (run_exchange V θ F c σ x).2 hv hd

/-- **The draw's piece of `CouplingTarget` at the session v2 sets up**, with its population: `coupled_draw_setsUp`
with `A.n = n` from `population_eq`. -/
theorem coupled_draw_held {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hA : SetsUp V θ F A) (hpub : ∃ p, ParamsAt θ p ∧ Flock.hexOf (Flock.Sha512.hash A.I.publicFile) = p.own.pub)
    (hR : DrawRead k (V.coinsRead .sampled)) (hk : ReadsV2 k) {o : Verity.Exchange}
    (hθ : o.params = θ) (hF : o.files = F) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧ A.n = n ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T,
        (freshCoins dr.label dr.after o.log).flatten <+:
          bitsOf (_root_.Properties.SampledProofs.bytesOf (A.src ω.1 0)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw_setsUp K A θ F c σ hA hR hk hθ hv hl
  have hn := population_eq hA hpub hθ hF hv hd
  exact ⟨dr, L, n, ξ, units, hd, hn, fun ω hp => h ω hn hp⟩

/-- `coupled_draw_coins_setsUp` with `A.n = n` from `population_eq`. -/
theorem coupled_draw_coins_held {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V)
    (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hA : SetsUp V θ F A) (hpub : ∃ p, ParamsAt θ p ∧ Flock.hexOf (Flock.Sha512.hash A.I.publicFile) = p.own.pub)
    (hR : DrawRead k (V.coinsRead .sampled)) (hk : ReadsV2 k) {o : Verity.Exchange}
    (hθ : o.params = θ) (hF : o.files = F) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) {m B : ℕ}
    (κ : A.Ω → Fin m → Fin (8 * B) → Bool)
    (hsrc : ∀ ω, _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m B (κ ω) 0)) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧ A.n = n ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T,
        freshCoins dr.label dr.after o.log <+: (List.ofFn fun j => List.ofFn (κ ω.1 j)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw_coins_setsUp K A θ F c σ hA hR hk hθ hv hl κ hsrc
  have hn := population_eq hA hpub hθ hF hv hd
  exact ⟨dr, L, n, ξ, units, hd, hn, fun ω hp => h ω hn hp⟩

end Verity.Sampled
