module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Core.Game.TwoRun
public import Proofs.Flock.Claims.Prob

/-!
# Sampled proofs' coupling: the messages

`CouplingTarget`'s messages piece: each message the evidence reveals for a round the device opens after its digest, and
the record the commitment is a digest of, is the most likely one at its digest, but on plays whose chance is at most the
square root of a two-continuation finder's success (`Core.Game.TwoRun`), one finder per round and one for the record.

* **The miss on a tape** (`DigestCut.missOn`): a uniform tape of the play (`CoinTape`) misses at a cut where it reaches
  the cut and the play's output reveals there a message other than the most likely one, and that has the miss's
  probability (`DigestCut.pr_missOn`). By the union bound, an event is at most as likely as the event on the tapes that
  miss at none of a family of cuts, plus each cut's miss (`DigestCut.prob_le_noMiss_add_missP`), and at cuts whose
  revealed messages open one digest, plus the square root of each cut's finder's success
  (`DigestCut.prob_le_noMiss_add_sqrt`).
* **The period's game as one game** (`relK`, `releases_bind`, `game_eq_relK`): the prover's request, the device's
  releases and then the close (`finish`), each release's continuation carrying the log after it, so a cut follows the
  frames the prover sends.
* **Where the messages are fixed** (`Reading`): round `j` at the first frame the prover sends that the reading's
  `round j` picks after the device's log before it, its digest the bytes the frame answers (an opening's `after`, the
  SHA-512 of the round's message, `Spot.Challenger.linkAfter` and each rep round's `msg_sha512`); the record
  at the play's start, its digest the reading's. Which opening is round `j` and what the evidence reveals for it are
  the rounds piece's, so everything here is at any reading.
* **The cuts and the finders** (`roundCut`, `recordCut`, `msgFinder`, `recordFinder`): each cut reveals only messages
  whose SHA-512 is its digest, so its messages agree (`DigestCut.Agrees`) by construction (`DigestCut.AgreeCut`). The
  finders are the cuts' two-continuation finders at `2t + 2h` calls on every coin, `Constructions`' `msgFinder`,
  `recordFinder`, `msg_calls` and `record_calls` (`msgFinder_calls`, `recordFinder_calls`).
* **The messages' bound** (`prob_le_messages`): an event of the period's game is at most as likely as the event on the
  tapes that miss at no round and not at the record (`NoMiss`), plus `CouplingTarget`'s messages terms, the square root
  of each round's finder's success and of the record's. So `CouplingTarget` holds once the sampled verdict off the
  misses is bounded by its other terms (`couplingTarget_of_noMiss`).
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

variable {α V : Type}

/-! ## The union bound over uniform coins -/

theorem pr_or_le {Ω : Type} [Fintype Ω] (p q : Ω → Prop) :
    Claims.Pr (fun x => p x ∨ q x) ≤ Claims.Pr p + Claims.Pr q := by
  classical
  rw [Claims.Pr_eq_sum, Claims.Pr_eq_sum p, Claims.Pr_eq_sum q, ← add_div, ← Finset.sum_add_distrib]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun x _ => ?_) (Nat.cast_nonneg _)
  by_cases hp : p x <;> by_cases hq : q x <;> simp [hp, hq]

theorem pr_exists_le {Ω ι : Type} [Fintype Ω] (M : ι → Ω → Prop) (t : Finset ι) :
    Claims.Pr (fun x => ∃ i ∈ t, M i x) ≤ ∑ i ∈ t, Claims.Pr (M i) := by
  classical
  induction t using Finset.induction_on with
  | empty => simp [Claims.Pr_false]
  | @insert a t ha ih =>
    rw [Finset.sum_insert ha]
    refine (Claims.Pr_mono fun x hx => ?_).trans ((pr_or_le (M a) _).trans (add_le_add le_rfl ih))
    obtain ⟨i, hi, hm⟩ := hx
    rcases Finset.mem_insert.1 hi with rfl | hi
    · exact Or.inl hm
    · exact Or.inr ⟨i, hi, hm⟩

/-- **An event is at most as likely as the event off a finite family of events, plus each one's probability.** -/
theorem pr_le_and_forall_not_add {Ω ι : Type} [Fintype Ω] [Fintype ι] (E : Ω → Prop) (M : ι → Ω → Prop) :
    Claims.Pr E ≤ Claims.Pr (fun x => E x ∧ ∀ i, ¬ M i x) + ∑ i, Claims.Pr (M i) := by
  refine (Claims.Pr_mono fun x hx => ?_).trans
    ((pr_or_le _ _).trans (add_le_add le_rfl (pr_exists_le M Finset.univ)))
  by_cases h : ∃ i ∈ Finset.univ, M i x
  · exact Or.inr h
  · exact Or.inl ⟨hx, fun i hi => h ⟨i, Finset.mem_univ i, hi⟩⟩

/-- **A coin, then a uniform dependent function**: an event of the function's value at the coin is averaged over the
coin. -/
theorem pr_pi_eval_fst {C : Type} [Fintype C] [DecidableEq C] {T : C → Type} [∀ c, Fintype (T c)]
    [∀ c, Nonempty (T c)] (P : (c : C) → T c → Prop) :
    Claims.Pr (fun x : C × ((c : C) → T c) => P x.1 (x.2 x.1)) = (∑ c, Claims.Pr (P c)) / Fintype.card C := by
  classical
  have hP : (Fintype.card ((c : C) → T c) : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have key : ∀ c, (∑ φ : (c' : C) → T c', if P c (φ c) then (1 : ℝ) else 0) =
      Claims.Pr (P c) * Fintype.card ((c' : C) → T c') := fun c => by
    have h := sum_pi_eval (T := T) c (fun t => if P c t then (1 : ℝ) else 0)
    rw [Claims.Pr_eq_sum (P c), ← h]
    field_simp
  rw [Claims.Pr_eq_sum, Fintype.sum_prod_type]
  simp only [key]
  rw [← Finset.sum_mul, Fintype.card_prod, Nat.cast_mul, mul_div_mul_right _ _ hP]

/-- **`prob` is the tapes' share**: the probability of an event of the play is the share of uniform tapes whose play
it holds at. -/
theorem prob_eq_ofReal_pr (p : α → Prop) (g : Game α) (s : Strategy g) :
    prob p g s = ENNReal.ofReal (Claims.Pr fun x : CoinTape g s => p (CoinTape.run g s x)) := by
  rw [pr_tape, ← prob_toReal, ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one p g s))]

namespace DigestCut

/-! ## The miss on a tape -/

/-- **A tape misses at the cut**: it reaches the cut, and the play's output reveals there a message other than the
most likely one (`revealMode`). -/
def missOn [Nonempty V] : {g : Game α} → DigestCut V g → (s : Strategy g) → CoinTape g s → Prop
  | _, .skip, _, _ => False
  | g, .here rv, s, x => ∃ w, rv (CoinTape.run g s x) = some w ∧ w ≠ revealMode rv g s
  | _, .send K, s, x => missOn (K s.1) s.2 x
  | _, @DigestCut.coin _ _ _ _ _ _ K, s, x => missOn (K x.1) (s x.1) (x.2 x.1)

/-- **The message the tape's play reveals at the cut it reaches**; `none` where it reaches none, or reveals none
there. -/
def revealOn : {g : Game α} → DigestCut V g → (s : Strategy g) → CoinTape g s → Option V
  | _, .skip, _, _ => none
  | g, .here rv, s, x => rv (CoinTape.run g s x)
  | _, .send K, s, x => revealOn (K s.1) s.2 x
  | _, @DigestCut.coin _ _ _ _ _ _ K, s, x => revealOn (K x.1) (s x.1) (x.2 x.1)

/-- **The most likely message at the cut the tape reaches** (`revealMode`), a function of the coins before the cut;
`none` where it reaches none. -/
noncomputable def modeOn [Nonempty V] : {g : Game α} → DigestCut V g → (s : Strategy g) → CoinTape g s → Option V
  | _, .skip, _, _ => none
  | g, .here rv, s, _ => some (revealMode rv g s)
  | _, .send K, s, x => modeOn (K s.1) s.2 x
  | _, @DigestCut.coin _ _ _ _ _ _ K, s, x => modeOn (K x.1) (s x.1) (x.2 x.1)

/-- **Off the miss, the revealed message is the most likely one**: a tape misses at the cut exactly where its play
reveals a message there that isn't the most likely one. -/
theorem missOn_iff [Nonempty V] : ∀ {g : Game α} (K : DigestCut V g) (s : Strategy g) (x : CoinTape g s),
    K.missOn s x ↔ ∃ w, K.revealOn s x = some w ∧ K.modeOn s x ≠ some w
  | _, .skip, _, _ => by simp [missOn, revealOn]
  | g, .here rv, s, x => by
      simp only [missOn, revealOn, modeOn, ne_eq, Option.some.injEq]
      exact exists_congr fun w => and_congr_right fun _ => ⟨fun h e => h e.symm, fun h e => h e.symm⟩
  | _, .send K, s, x => missOn_iff (K s.1) s.2 x
  | _, @DigestCut.coin _ _ _ _ _ _ K, s, x => missOn_iff (K x.1) (s x.1) (x.2 x.1)

/-- **The tapes that miss at a cut have the miss's probability.** -/
theorem pr_missOn [Nonempty V] : ∀ {g : Game α} (K : DigestCut V g) (s : Strategy g),
    Claims.Pr (K.missOn s) = K.miss s
  | _, .skip, _ => Claims.Pr_false
  | g, .here rv, s => pr_tape (fun o => ∃ w, rv o = some w ∧ w ≠ revealMode rv g s) g s
  | _, .send K, s => pr_missOn (K s.1) s.2
  | _, @DigestCut.coin _ _ C _ _ k K, s => by
      classical
      let _ : ∀ c, Fintype (CoinTape (k c) (s c)) := fun c => CoinTape.fintype (k c) (s c)
      have _ : ∀ c, Nonempty (CoinTape (k c) (s c)) := fun c => CoinTape.nonempty (k c) (s c)
      show Claims.Pr (fun x : C × ((c : C) → CoinTape (k c) (s c)) => (K x.1).missOn (s x.1) (x.2 x.1)) =
        (∑ c, (K c).miss (s c)) / Fintype.card C
      rw [pr_pi_eval_fst (T := fun c => CoinTape (k c) (s c)) fun c => (K c).missOn (s c)]
      simp only [pr_missOn (K _) (s _)]

theorem missP_eq_ofReal [Nonempty V] {g : Game α} (K : DigestCut V g) (s : Strategy g) :
    K.missP s = ENNReal.ofReal (Claims.Pr (K.missOn s)) := by
  rw [pr_missOn, ← missP_toReal, ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (K.missP_le_one s))]

/-- **An event, off the misses of a family of cuts**: an event of the play is at most as likely as the event on the
tapes that miss at none of the cuts, plus each cut's miss. -/
theorem prob_le_noMiss_add_missP [Nonempty V] {ι : Type} [Fintype ι] (E : α → Prop) (g : Game α) (s : Strategy g)
    (Ks : ι → DigestCut V g) :
    prob E g s ≤
      ENNReal.ofReal (Claims.Pr fun x : CoinTape g s => E (CoinTape.run g s x) ∧ ∀ i, ¬ (Ks i).missOn s x) +
        ∑ i, (Ks i).missP s := by
  rw [prob_eq_ofReal_pr]
  simp only [missP_eq_ofReal]
  rw [← ENNReal.ofReal_sum_of_nonneg fun i _ => Claims.Pr_nonneg _,
    ← ENNReal.ofReal_add (Claims.Pr_nonneg _) (Finset.sum_nonneg fun i _ => Claims.Pr_nonneg _)]
  exact ENNReal.ofReal_le_ofReal (pr_le_and_forall_not_add _ _)

/-- **An event, off the misses of a family of cuts whose messages agree**: at most as likely as the event on the tapes
that miss at none of the cuts, plus the square root of each cut's two-continuation finder's success (`finder`, at any
budget). -/
theorem prob_le_noMiss_add_sqrt (H : Claims.Msg → Claims.Msg) {ι : Type} [Fintype ι] (E : α → Prop) (g : Game α)
    (s : Strategy g) (Ks : ι → DigestCut Claims.Msg g) (hK : ∀ i, (Ks i).Agrees H) (q : ι → ℕ) :
    prob E g s ≤
      ENNReal.ofReal (Claims.Pr fun x : CoinTape g s => E (CoinTape.run g s x) ∧ ∀ i, ¬ (Ks i).missOn s x) +
        ∑ i, ENNReal.ofReal (Real.sqrt (((Ks i).finder s (q i)).Pr H)) :=
  (prob_le_noMiss_add_missP E g s Ks).trans
    (add_le_add le_rfl (Finset.sum_le_sum fun i _ => missP_le H (Ks i) (hK i) s (q i)))

/-! ## Cuts whose messages agree -/

/-- **A cut whose revealed messages all have one image under `H`**, with its `Agrees`. -/
structure AgreeCut {D : Type} (H : V → D) (g : Game α) : Type 1 where
  cut : DigestCut V g
  agrees : cut.Agrees H

namespace AgreeCut

variable {D : Type} {H : V → D}

def skip {g : Game α} : AgreeCut H g := ⟨.skip, trivial⟩

/-- The cut here, at a reading that reveals only messages with the image `d`. -/
def here {g : Game α} (rv : α → Option V) (d : D) (h : ∀ o w, rv o = some w → H w = d) : AgreeCut H g :=
  ⟨.here rv, here_agrees H rv d h⟩

def send {M : Type} {k : M → Game α} (K : (m : M) → AgreeCut H (k m)) : AgreeCut H (.send M k) :=
  ⟨.send fun m => (K m).cut, fun m => (K m).agrees⟩

def coin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} (K : (c : C) → AgreeCut H (k c)) :
    AgreeCut H (@Game.coin _ C iF iN k) :=
  ⟨@DigestCut.coin _ _ C iF iN k fun c => (K c).cut, fun c => (K c).agrees⟩

/-- A decision's two cuts, on `ite`'s two branches. -/
def iteCut {c : Prop} {dc : Decidable c} {t e : Game α} (ct : AgreeCut H t) (ce : AgreeCut H e) :
    AgreeCut H (@ite _ c dc t e) :=
  match dc with
  | isTrue _ => ct
  | isFalse _ => ce

/-- The cut on a game equal to its own. -/
def cast {g g' : Game α} (h : g = g') (K : AgreeCut H g) : AgreeCut H g' := h ▸ K

theorem cast_heq {g g' : Game α} (h : g = g') (K : AgreeCut H g) : HEq (K.cast h) K := by
  subst h
  rfl

end AgreeCut

end DigestCut

end Verity.Game

namespace Verity

open Game

/-! ## The period's game as one game -/

/-- An option's two continuations: `n` at `none`, `y x` at `some x`. -/
def optGame {γ β : Type} : Option γ → Game β → (γ → Game β) → Game β
  | none, n, _ => n
  | some x, _, y => y x

/-- **The device's releases, then `f` at their log** (`releases_bind`), each release's continuation carrying the log
after it. -/
def relK {β : Type} : ℕ → List (String × ℕ) → List OpenTree → (List Release → Game β) → Game β
  | 0, _, _, f => f []
  | k + 1, fs, ts, f => .send (Option (Frame × ByteArray)) fun
    | none => f []
    | some (.fresh l, b) => optGame (takeCoin l fs) (f []) fun fs' =>
        .coin (Fin 512 → Bool) fun x => relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)
    | some (.hello spec, ν) =>
      if treeLines ts < maxTreeLines ∧ TreeSpecOk spec then
        .coin (TreeSample spec) fun t => relK k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) fun log =>
          f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log)
      else f []
    | some (.opening i s n, b) =>
      if treeLines ts < maxTreeLines then
        optGame ts[i]? (f []) fun T => optGame (T.next s n) (f []) fun p =>
          haveI : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2) := ⟨⟨T.sample.2 p.1 p.2, rfl⟩⟩
          .coin (TreeOpenable T.nonce T.spec T.sample p.1 p.2) fun d =>
            relK k fs (ts.set i (T.bump p.1)) fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log)
      else f []

theorem releases_bind {β : Type} :
    ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β),
      (releases k fs ts).bind f = relK k fs ts f
  | 0, _, _, _ => rfl
  | k + 1, fs, ts, f => by
    rw [releases, relK, Game.bind_send]
    congr 1
    funext m
    rcases m with _ | ⟨l | spec | ⟨i, s, n⟩, b⟩
    · rfl
    · dsimp only
      cases takeCoin l fs with
      | none => rfl
      | some fs' =>
        simp only [optGame, Game.bind_coin, Game.map, Game.bind_assoc, Game.bind_ret]
        congr 1
        funext x
        exact releases_bind k fs' ts _
    · dsimp only
      split_ifs
      · simp only [Game.bind_coin, Game.map, Game.bind_assoc, Game.bind_ret]
        congr 1
        funext t
        exact releases_bind k _ _ _
      · rfl
    · dsimp only
      split_ifs
      · cases ts[i]? with
        | none => rfl
        | some T =>
          dsimp only [optGame]
          cases T.next s n with
          | none => rfl
          | some p =>
            simp only [Game.bind_coin, Game.map, Game.bind_assoc, Game.bind_ret]
            congr 1
            funext d
            exact releases_bind k _ _ _
      · rfl

theorem play_eq (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    play fs D θ F c = .send ByteArray fun a => relK (frames fs) fs [] (finish D θ F c a) := by
  show Game.send ByteArray _ = _
  congr 1
  funext a
  exact releases_bind _ _ _ _

/-! ## The cuts at the frames the prover sends -/

/-- **The messages `rv` reveals that open the digest `dg`**: those whose SHA-512 is `dg`. -/
def digestReveal {β : Type} (rv : β → Option Claims.Msg) (dg : Claims.Msg) (o : β) : Option Claims.Msg :=
  (rv o).filter fun w => decide (sha512Msg w = dg)

theorem digestReveal_sha512 {β : Type} (rv : β → Option Claims.Msg) (dg : Claims.Msg) (o : β) (w : Claims.Msg)
    (h : digestReveal rv dg o = some w) : sha512Msg w = dg := by
  unfold digestReveal at h
  rcases hr : rv o with _ | u
  · simp [hr] at h
  · rw [hr] at h
    by_cases hu : sha512Msg u = dg
    · simp [Option.filter, hu] at h
      exact h ▸ hu
    · simp [Option.filter, hu] at h

/-- **The cut here, at the digest `dg`**: the messages `rv` reveals that open `dg`. -/
def cutAt {β : Type} {g : Game β} (rv : β → Option Claims.Msg) (dg : Claims.Msg) :
    DigestCut.AgreeCut sha512Msg g :=
  .here (digestReveal rv dg) dg (digestReveal_sha512 rv dg)

/-- An option's two cuts, on `optGame`'s two continuations. -/
def optCut {γ β : Type} {n : Game β} {y : γ → Game β} :
    (o : Option γ) → DigestCut.AgreeCut sha512Msg n → ((x : γ) → DigestCut.AgreeCut sha512Msg (y x)) →
      DigestCut.AgreeCut sha512Msg (optGame o n y)
  | none, cn, _ => cn
  | some x, _, cy => cy x

open Classical in
/-- **The cut at the first frame the prover sends that `P` picks**, `P` reading the device's log `pre` before it and the
frame, and its digest the bytes the frame answers: there the play reveals the messages `rv` reads that open the
digest. A play with no such frame has no cut. -/
noncomputable def frameCut {β : Type} (P : List Release → Frame → Prop) (rv : β → Option Claims.Msg) :
    (k : ℕ) → (fs : List (String × ℕ)) → (ts : List OpenTree) → (f : List Release → Game β) →
      (pre : List Release) → DigestCut.AgreeCut sha512Msg (relK k fs ts f)
  | 0, _, _, _, _ => .skip
  | k + 1, fs, ts, _, pre => .send fun
    | none => .skip
    | some (.fresh l, b) =>
      if P pre (.fresh l) then cutAt rv b.data.toList else
        optCut (takeCoin l fs) .skip fun fs' => .coin fun x =>
          frameCut P rv k fs' ts _ (pre ++ [⟨.fresh l, b, List.ofFn x⟩])
    | some (.hello spec, ν) =>
      if P pre (.hello spec) then cutAt rv ν.data.toList else
        .iteCut (.coin fun t => frameCut P rv k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) _
          (pre ++ [⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩])) .skip
    | some (.opening i s n, b) =>
      if P pre (.opening i s n) then cutAt rv b.data.toList else
        .iteCut (optCut ts[i]? .skip fun T => optCut (T.next s n) .skip fun p => .coin fun d =>
          frameCut P rv k fs (ts.set i (T.bump p.1)) _ (pre ++ [⟨.opening i s n, b, TreeDraw.opened d.1 n⟩])) .skip

end Verity

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The period's game is the prover's request, the device's releases, then the close**, as one game. -/
theorem game_eq_relK (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    V.game θ F c = .send ByteArray fun a =>
      relK (frames (V.families θ)) (V.families θ) [] (finish (V.signer θ) θ F c a) :=
  play_eq _ _ θ F c

/-- **Where a session's messages are fixed, and what the evidence reveals for them**, at `R` rounds. Round `j` is fixed
at the first frame the prover sends that `round j` picks, after the device's log before it; its digest is the bytes
the frame answers. The record is fixed at the play's start, its digest `recordDigest`. -/
structure Reading (R : ℕ) where
  /-- whether a frame, after the device's log before it, is round `j`'s -/
  round : Fin R → List Release → Frame → Prop
  /-- the message the play's output reveals for round `j` -/
  reveal : Fin R → Exchange → Option Claims.Msg
  /-- the record the play's output reveals -/
  record : Exchange → Option Claims.Msg
  /-- the digest of the record the commitment fixes -/
  recordDigest : Claims.Msg

/-- **The readings at each session and each parameters, committed files and commitment.** -/
abbrev Readings := (A : Live) → ByteArray → Array ByteArray → ByteArray → Reading A.rounds

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) {R : ℕ}

/-- **Round `j`'s cut**: after the prover's request, at the first frame round `j`'s reading picks, the messages the
evidence reveals for it that open the bytes the frame answers. -/
noncomputable def roundCut (rd : Reading R) (j : Fin R) : DigestCut.AgreeCut sha512Msg (V.game θ F c) :=
  (DigestCut.AgreeCut.send fun a : ByteArray => frameCut (rd.round j) (rd.reveal j) (frames (V.families θ))
    (V.families θ) [] (finish (V.signer θ) θ F c a) []).cast (game_eq_relK V θ F c).symm

/-- **The record's cut**: at the play's start, the records the evidence reveals that open the record's digest. -/
noncomputable def recordCut (rd : Reading R) : DigestCut.AgreeCut sha512Msg (V.game θ F c) :=
  cutAt rd.record rd.recordDigest

/-- **A tape that misses at no round and not at the record**: its play reveals, at each round's cut and the record's
it reaches, only the most likely message there (`DigestCut.missOn_iff`). -/
def NoMiss (rd : Reading R) (σ : Game.Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ) : Prop :=
  (∀ j, ¬ (roundCut V θ F c rd j).cut.missOn σ x) ∧ ¬ (recordCut V θ F c rd).cut.missOn σ x

/-- **The messages' bound at a reading and a budget**: an event of the period's game is at most as likely as the event
on the tapes that miss at no round and not at the record, plus the square root of each round's two-continuation
finder's success and of the record's. -/
theorem prob_le_noMiss_add (rd : Reading R) (σ : Game.Strategy (V.game θ F c)) (q : ℕ) (E : Exchange → Prop) :
    Game.prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          E (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c rd σ x) +
        (∑ j, ENNReal.ofReal (Real.sqrt (((roundCut V θ F c rd j).cut.finder σ q).Pr sha512Msg)) +
          ENNReal.ofReal (Real.sqrt (((recordCut V θ F c rd).cut.finder σ q).Pr sha512Msg))) := by
  let Ks : Option (Fin R) → DigestCut Claims.Msg (V.game θ F c) := fun
    | none => (recordCut V θ F c rd).cut
    | some j => (roundCut V θ F c rd j).cut
  have hK : ∀ i, (Ks i).Agrees sha512Msg := fun
    | none => (recordCut V θ F c rd).agrees
    | some j => (roundCut V θ F c rd j).agrees
  have h := DigestCut.prob_le_noMiss_add_sqrt sha512Msg E (V.game θ F c) σ Ks hK fun _ => q
  rw [Fintype.sum_option, add_comm (ENNReal.ofReal _) (∑ j, _)] at h
  refine h.trans_eq ?_
  congr 2
  exact Claims.Pr_congr fun x => and_congr_right fun _ =>
    ⟨fun h => ⟨fun j => h (some j), h none⟩, fun h => fun | none => h.2 | some j => h.1 j⟩

/-- **Round `j`'s two-continuation finder**, at the session `A`: round `j`'s cut's, at `2t + 2h` SHA-512 calls on every
coin (`Live.qM`), `t` the strategy's calls. -/
noncomputable def msgFinder (rd : Readings) (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (j : Fin A.rounds) : Claims.Finder :=
  (roundCut V θ F c (rd A θ F c) j).cut.finder σ (2 * V.cost θ F c σ + 2 * A.h)

/-- **The record's two-continuation finder**, at the session `A`: the record's cut's, at `2t + 2h` SHA-512 calls on
every coin. -/
noncomputable def recordFinder (rd : Readings) (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) : Claims.Finder :=
  (recordCut V θ F c (rd A θ F c)).cut.finder σ (2 * V.cost θ F c σ + 2 * A.h)

omit θ F c in
theorem msgFinder_calls (rd : Readings) : ∀ (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (j : Fin A.rounds) (x : (msgFinder V rd A θ F c σ j).S),
    ((msgFinder V rd A θ F c σ j).cost x : ℝ) ≤ A.qM (V.cost θ F c σ) := fun A θ F c σ j _ => by
  show ((2 * V.cost θ F c σ + 2 * A.h : ℕ) : ℝ) ≤ A.qM (V.cost θ F c σ)
  simp [Live.qM]

omit θ F c in
theorem recordFinder_calls (rd : Readings) : ∀ (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (x : (recordFinder V rd A θ F c σ).S),
    ((recordFinder V rd A θ F c σ).cost x : ℝ) ≤ A.qM (V.cost θ F c σ) := fun A θ F c σ _ => by
  show ((2 * V.cost θ F c σ + 2 * A.h : ℕ) : ℝ) ≤ A.qM (V.cost θ F c σ)
  simp [Live.qM]

omit θ F c in
/-- **The messages' bound at constructions whose two-continuation finders are these** (`msgFinder`, `recordFinder`):
an event of the period's game is at most as likely as the event on the tapes that miss at no round and not at the
record, plus `CouplingTarget`'s messages terms. -/
theorem prob_le_messages (rd : Readings) (K : Constructions V) (hM : K.msgFinder = msgFinder V rd)
    (hR : K.recordFinder = recordFinder V rd) (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (E : Exchange → Prop) :
    Game.prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          E (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x) +
        (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder A θ F c σ j).Pr sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.recordFinder A θ F c σ).Pr sha512Msg))) := by
  rw [hM, hR]
  exact prob_le_noMiss_add V θ F c (rd A θ F c) σ _ E

omit θ F c in
/-- **The coupling off the messages' misses**: at constructions whose two-continuation finders are these,
`CouplingTarget` holds once the sampled verdict on the tapes that miss at no round and not at the record is at most as
likely as the live audit's acceptance at the projected strategy, plus the square root of the bound digests' finder's
success, the hiding term, the read finder's collision and the forger's forgery. -/
theorem couplingTarget_of_noMiss (k : Verity.Registration.Composite) (rd : Readings) (K : Constructions V)
    (hM : K.msgFinder = msgFinder V rd) (hR : K.recordFinder = recordFinder V rd) (A : Live) (θ : ByteArray)
    (F : Array ByteArray)
    (h : ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
      ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x) ≤
        FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own o)
            (K.live A θ F c σ).game (K.project A θ F c σ) +
          (ENNReal.ofReal (Real.sqrt ((K.bindFinder A θ F c σ).Pr sha512Msg)) + A.hidingTerm +
            ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
            ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))) :
    CouplingTarget V k K A θ F := fun c σ =>
  (prob_le_messages V rd K hM hR A θ F c σ _).trans <| (add_le_add (h c σ) le_rfl).trans_eq (by ring)

end Verity.Sampled
