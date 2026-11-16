module

public import Proofs.Properties.Verity.SampledTwo.MessagesShown

/-!
# The two-stage audit's messages, fixed by what the device fixed before each round

`Messages.missBound` bounds a round's miss where the device fixed, before the round's coin, the SHA-512 of the very
message `SpotFlock.V` reads there (`MsgOpens`, `DigestBefore`). The two-stage form's rounds are not all of that kind.
A session's link message is SpotFlock's tagged encoding of the record's `root_B` and public words, while the device
opened the link after the SHA-512 of `root_F` and the tables' `root_B`; a rep round's message is its retained bytes
after a tag byte, while the device opened it after the SHA-512 of the bytes alone; a session's last message is its
statement, record and proof files, while the device closed the session after the SHA-512 of its evidence; and the
statements' round has no release of the device at all: its message is fixed by the registered files and the stages'
samples. This module states the messages' bound in the form those rounds need.

* **The miss at any opening** (`pickOf`, `frameOf`, `miss_le_fork`): at a prover that sends, at each history, the most
  likely value of any function `opn` of the tape, the history and the run's coins over the coins the history shows,
  a round's miss is at most the square root of a tape and two coins of one history giving `opn` two values, wherever
  an accepted play gives `opn` the message it reads, at its own history (`hopen`). No digest enters.
* **What the play fixes** (`openingBy`, `frameBy`, `Opens`, `FixedBefore`, `Breaks`, `finderBy`, `missBoundBy`,
  `reachedBy`, `messagesFixedBy`, `callsLeBy`): the play fixes a value `Fx o i` for round `i` (digests, and whatever
  else the coins before the round fix); an accepted play's reading at round `i` opens it (`Opens`, at any relation
  `Opn`); two plays of one tape that each open what they fix, the second at coins that agree with the coins the
  first's history before the round shows, fix one value (`FixedBefore`); and two plays that open one fixed value with
  two messages give a SHA-512 collision (`Breaks`, at the finder's `pair`). The round's miss is then at most the
  square root of the finder's success, and `MessagesFixed` holds at the sum over the rounds. `FixedBefore` asks
  nothing of a play that opens nothing, so the value may be read off what only an accepted play shows, the coins the
  check read among them; `fixedBefore_of_eq` gives it where the value is the same at every such coins.
* **The message's own digest** (`digestOpn`, `digestPair`, `breaks_digest`, `opens_digest`): `Messages.missBound`'s
  case, the fixed value the SHA-512 of the message itself.
* **Strings that open digests** (`stringsFx`, `stringsOpn`, `stringsPair`, `Explains`, `breaks_strings`): the fixed
  value is a list of digests and a context (`ctx`: the coins the device released before the round, the registered
  files), the play shows a list of strings for the round (`raw`: the record's bytes the device's digests hash, the
  blobs the registered digests name), and on accepted plays the message is a function of the strings and the context
  (`Explains`). Two accepted plays with two messages then show two lists of strings under one list of digests, and
  the finder outputs a pair of them that differ (`exists_zip_collide`). `reachedStrings` and `messagesFixedStrings`
  are the bound from `ShowsStrings`, `FixedBefore` and `Explains`. `fixedBefore_strings` gives `FixedBefore` where two
  accepted plays at coins the history shows alike read the same digests and the same context.

**Not established** at the two-stage form: `Opens` (the device's digests are the SHA-512 of the strings the check
read: the interior's root, each opening's `after`), `FixedBefore` and `Explains` (the form's check reads each round's
message off the strings and the earlier coins). `RoundPrefix.lean` proves `FixedBefore` where the value is read off
the device's log before the round.
-/

@[expose] public section

namespace Verity.SampledTwo.Messages

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The miss at any opening -/

/-- **The most likely opening at a history**: the value `opn` most often takes at the history's round over the run's
coins the history shows. -/
noncomputable def pickOf {A : Live₂} {x : Msg} {Ω : Type}
    (opn : ℕ → Ω → List (Msg × Msg) → Coins A x → Option Msg) (ω : Ω) (hist : List (Msg × Msg)) : Msg :=
  modeOn (Shows (A := A) (x := x) hist) (opn hist.length ω hist)

section Fork

variable (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
  (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
  (opn : ℕ → TapeOf V θ F c σ → List (Msg × Msg) → Coins A x → Option Msg) (i : ℕ)

/-- **The frame at the coin map `place` whose prover sends the most likely opening** (`pickOf`). -/
noncomputable def frameOf : Frame V A x θ F c σ := ⟨place, proverOf (pickOf opn)⟩

/-- **That prover's run's history before round `i`**, at the tape `ω` and the run's coins `r`. -/
noncomputable def histOf (ω : TapeOf V θ F c σ) (r : Coins A x) : List (Msg × Msg) :=
  runHist (Claims.run FlockDeploy.sha512 (proverOf (pickOf opn)) A.L.V x ω r) i

/-- A tape and run coins' class at round `i`: the tape, and the run's history before round `i`. -/
noncomputable def clsBy (s : TapeOf V θ F c σ × Coins A x) : TapeOf V θ F c σ × List (Msg × Msg) :=
  (s.1, histOf V θ F c σ opn i s.1 s.2)

/-- The resample of a tape and run coins within their class. -/
noncomputable def resBy (s : TapeOf V θ F c σ × Coins A x) (y : Coins A x) : TapeOf V θ F c σ × Coins A x :=
  (s.1, merge (histOf V θ F c σ opn i s.1 s.2) y)

/-- The opening at round `i`, at a class's tape and history. -/
noncomputable def openBy (K : TapeOf V θ F c σ × List (Msg × Msg)) (s : TapeOf V θ F c σ × Coins A x) :
    Option Msg :=
  opn i K.1 K.2 s.2

/-- The class's most likely opening at round `i`. -/
noncomputable def modeBy (K : TapeOf V θ F c σ × List (Msg × Msg)) : Msg :=
  modeOn (Shows (A := A) (x := x) K.2) (opn i K.1 K.2)

open Classical in
theorem classOf_clsBy (hS : RunShown A x) (s : TapeOf V θ F c σ × Coins A x) :
    classOf (clsBy V θ F c σ opn i) s =
      (Finset.univ.filter fun r' => Shows (histOf V θ F c σ opn i s.1 s.2) r').map
        ⟨fun r' => (s.1, r'), fun _ _ h => (Prod.mk.inj h).2⟩ := by
  ext ⟨ω', r'⟩
  rw [Finset.mem_map]
  unfold classOf
  simp only [Finset.mem_filter]
  constructor
  · rintro ⟨-, h⟩
    have h1 : ω' = s.1 := congrArg Prod.fst h
    have h2 : histOf V θ F c σ opn i ω' r' = histOf V θ F c σ opn i s.1 s.2 := congrArg Prod.snd h
    subst h1
    exact ⟨r', ⟨Finset.mem_univ _, (hist_eq_iff hS (proverOf (pickOf opn)) s.1 s.2 r' i).1 h2⟩, rfl⟩
  · rintro ⟨a, ha, he⟩
    have he' : (s.1, a) = (ω', r') := he
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj he'
    exact ⟨Finset.mem_univ _, Prod.ext rfl ((hist_eq_iff hS (proverOf (pickOf opn)) s.1 s.2 _ i).2 ha.2)⟩

/-- **Round `i`'s miss is at most the square root of a fork**, at the prover that sends the most likely opening:
where an accepted play whose log is the device's gives `opn`, at the play's own history before round `i`, the message
it reads there (`hopen`), the first difference at round `i` is at most as likely as the square root of a tape, the
run's coins and a resample of the coins their history does not show giving `opn` two values at that history. -/
theorem miss_le_fork (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (hS : RunShown A x)
    (hopen : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (outOf V θ F c σ place ω r) →
      Good (outOf V θ F c σ place ω r) → i < A.L.V.k x →
      ∃ m, Rd.msg (outOf V θ F c σ place ω r) i = some m ∧
        opn i ω (playHist A x r (playMsgs Rd (outOf V θ F c σ place ω r)) i) r = some m) :
    Claims.Pr (MissAt (frameOf V θ F c σ place opn) Rd Acc Good i) ≤
      Real.sqrt (Claims.Pr fun s : (TapeOf V θ F c σ × Coins A x) × Coins A x =>
        fork (clsBy V θ F c σ opn i) (openBy V θ F c σ opn i) s.1 (resBy V θ F c σ opn i s.1 s.2)) := by
  classical
  have hunif : ∀ (s : TapeOf V θ F c σ × Coins A x) (φ : TapeOf V θ F c σ × Coins A x → ℝ),
      (∑ y, φ (resBy V θ F c σ opn i s y)) * (classOf (clsBy V θ F c σ opn i) s).card =
        Fintype.card (Coins A x) * ∑ x' ∈ classOf (clsBy V θ F c σ opn i) s, φ x' := by
    intro s φ
    rw [classOf_clsBy V θ F c σ opn i hS, Finset.sum_map, Finset.card_map]
    exact sum_merge (histOf V θ F c σ opn i s.1 s.2) fun r' => φ (s.1, r')
  have hmode : ∀ (s : TapeOf V θ F c σ × Coins A x) m,
      ((classOf (clsBy V θ F c σ opn i) s).filter fun x' =>
          openBy V θ F c σ opn i (clsBy V θ F c σ opn i s) x' = some m).card ≤
        ((classOf (clsBy V θ F c σ opn i) s).filter fun x' =>
          openBy V θ F c σ opn i (clsBy V θ F c σ opn i s) x' =
            some (modeBy V θ F c σ opn i (clsBy V θ F c σ opn i s))).card := by
    intro s m
    rw [classOf_clsBy V θ F c σ opn i hS, Finset.filter_map, Finset.filter_map, Finset.card_map,
      Finset.card_map, Finset.filter_filter, Finset.filter_filter]
    exact modeOn_spec _ _ m
  have hmiss : ∀ s, MissAt (frameOf V θ F c σ place opn) Rd Acc Good i s →
      bad (clsBy V θ F c σ opn i) (openBy V θ F c σ opn i) (modeBy V θ F c σ opn i) s s := by
    rintro ⟨ω, r⟩ hs
    dsimp only [MissAt] at hs
    obtain ⟨hA, hG, hlen, hne, hagree⟩ := hs
    replace hA : Acc (outOf V θ F c σ place ω r) := hA
    replace hG : Good (outOf V θ F c σ place ω r) := hG
    replace hlen : i < (Claims.run FlockDeploy.sha512 (proverOf (pickOf opn)) A.L.V x ω r).msgs.length := hlen
    replace hne : Rd.msg (outOf V θ F c σ place ω r) i ≠
        msgAt (Claims.run FlockDeploy.sha512 (proverOf (pickOf opn)) A.L.V x ω r) i := hne
    replace hagree : ∀ j < i, Rd.msg (outOf V θ F c σ place ω r) j =
        msgAt (Claims.run FlockDeploy.sha512 (proverOf (pickOf opn)) A.L.V x ω r) j := hagree
    have hk : i < A.L.V.k x := lt_of_lt_of_le hlen (Claims.Recursion.run_length_le _ _ _ _ _ _)
    obtain ⟨m, hm, hop⟩ := hopen ω r hA hG hk
    have hKlen : (histOf V θ F c σ opn i ω r).length = i := runHist_length _ hlen.le
    have hmsg : msgAt (Claims.run FlockDeploy.sha512 (proverOf (pickOf opn)) A.L.V x ω r) i =
        some (pickOf opn ω (histOf V θ F c σ opn i ω r)) :=
      run_msgAt (pickOf opn) A.L.V x ω r i hlen
    have hplay : playHist A x r (playMsgs Rd (outOf V θ F c σ place ω r)) i = histOf V θ F c σ opn i ω r := by
      refine playHist_eq A x (proverOf (pickOf opn)) ω r _ i hlen.le fun l hl => ?_
      have h1 := run_msgAt (pickOf opn) A.L.V x ω r l (lt_trans hl hlen)
      have h2 := (hagree l hl).trans h1
      rw [h1]
      show some _ = some ((Rd.msg (outOf V θ F c σ place ω r) l).getD [])
      rw [h2, Option.getD_some]
    have hv : pickOf opn ω (histOf V θ F c σ opn i ω r) =
        modeBy V θ F c σ opn i (clsBy V θ F c σ opn i (ω, r)) := by
      unfold pickOf
      rw [hKlen]
      rfl
    rw [hplay] at hop
    unfold bad
    refine Exists.intro m (And.intro hop fun hmv => hne ?_)
    rw [hm, hmsg, hmv, hv]
  exact miss_le_sqrt (clsBy V θ F c σ opn i) (resBy V θ F c σ opn i) (openBy V θ F c σ opn i)
    (modeBy V θ F c σ opn i) (MissAt (frameOf V θ F c σ place opn) Rd Acc Good i) hunif hmode hmiss

end Fork

/-! ## What the play fixes before a round -/

variable {Φ : Type}

/-- **The message read at round `i` that opens what its play fixes, at the history `hist`**: the message the reading
`Rd` reads at round `i` of the play `out ω r`, where the play's own history before round `i` is `hist` and the play
opens (`Opn`) the value `Fx` reads for round `i` in it. -/
noncomputable def openingBy {A : Live₂} {x : Msg} {Ω : Type} (out : Ω → Coins A x → Verity.Exchange) (Rd : Reading A)
    (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop) (i : ℕ) (ω : Ω)
    (hist : List (Msg × Msg)) (r : Coins A x) : Option Msg := by
  classical
  exact (Rd.msg (out ω r) i).filter fun _ =>
    decide (playHist A x r (playMsgs Rd (out ω r)) i = hist ∧ ∃ φ, Fx (out ω r) i = some φ ∧ Opn φ (out ω r) i)

theorem openingBy_some {A : Live₂} {x : Msg} {Ω : Type} {out : Ω → Coins A x → Verity.Exchange} {Rd : Reading A}
    {Fx : Verity.Exchange → ℕ → Option Φ} {Opn : Φ → Verity.Exchange → ℕ → Prop} {i : ℕ} {ω : Ω}
    {hist : List (Msg × Msg)} {r : Coins A x} {u : Msg} (h : openingBy out Rd Fx Opn i ω hist r = some u) :
    Rd.msg (out ω r) i = some u ∧ playHist A x r (playMsgs Rd (out ω r)) i = hist ∧
      ∃ φ, Fx (out ω r) i = some φ ∧ Opn φ (out ω r) i := by
  classical
  unfold openingBy at h
  rw [Option.filter_eq_some_iff] at h
  exact ⟨h.1, (of_decide_eq_true h.2).1, (of_decide_eq_true h.2).2⟩

theorem openingBy_of {A : Live₂} {x : Msg} {Ω : Type} {out : Ω → Coins A x → Verity.Exchange} {Rd : Reading A}
    {Fx : Verity.Exchange → ℕ → Option Φ} {Opn : Φ → Verity.Exchange → ℕ → Prop} {i : ℕ} {ω : Ω}
    {hist : List (Msg × Msg)} {r : Coins A x} {u : Msg} (h1 : Rd.msg (out ω r) i = some u)
    (hh : playHist A x r (playMsgs Rd (out ω r)) i = hist)
    (h2 : ∃ φ, Fx (out ω r) i = some φ ∧ Opn φ (out ω r) i) :
    openingBy out Rd Fx Opn i ω hist r = some u := by
  classical
  unfold openingBy
  rw [h1, Option.filter_some_eq_some]
  exact decide_eq_true ⟨hh, h2⟩

/-- **The opening the frame's prover reads**, at the coin map `place`, the reading `Rd`, what the play fixes (`Fx`)
and what opens it (`Opn`). -/
noncomputable def opnBy (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop) :
    ℕ → TapeOf V θ F c σ → List (Msg × Msg) → Coins A x → Option Msg :=
  fun i ω hist r => openingBy (outOf V θ F c σ place) Rd Fx Opn i ω hist r

/-- **The frame** at the coin map `place`, whose prover sends at each history the most likely message the play reads
there, over the run's coins the history shows, among the plays of that history that open what they fix for the
round. -/
noncomputable def frameBy (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop) :
    Frame V A x θ F c σ :=
  frameOf V θ F c σ place (opnBy V θ F c σ place Rd Fx Opn)

/-- **`CallsLe` at the frame**, where the verifier's calls are within the strategy's. -/
theorem callsLeBy (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop)
    (h : VerifierCallsLe V A x θ F c σ (frameBy V A x θ F c σ place Rd Fx Opn).P) :
    CallsLe V A x θ F c σ (frameBy V A x θ F c σ place Rd Fx Opn) :=
  callsLe_of V A x θ F c σ _ (pickOf (opnBy V θ F c σ place Rd Fx Opn)) rfl h

/-- **An accepted play's reading opens what the play fixes**: on a play the check accepts whose log is the device's,
the check reads a message at each round below `k`, the play fixes a value for the round, and the play opens it. -/
def Opens (A : Live₂) (x : Msg) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∀ i < A.L.V.k x, (∃ m, Rd.msg o i = some m) ∧ ∃ φ, Fx o i = some φ ∧ Opn φ o i

/-- **What the plays fix for round `i` is fixed by the coins the history before it shows**: two plays of one tape
that each open (`Opn`) the value they fix for round `i`, the second at run coins that agree with the coins the first's
history before round `i` shows, fix the same value. A play that opens nothing is asked nothing. -/
def FixedBefore {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop) (i : ℕ) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r r' : Coins A x) (φ φ' : Φ), Fx (Fr.out ω r) i = some φ → Opn φ (Fr.out ω r) i →
    Fx (Fr.out ω r') i = some φ' → Opn φ' (Fr.out ω r') i →
    Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r' → φ' = φ

/-- **`FixedBefore` where the value is the same at every coins the history shows**: where a play that opens a value
is one the check accepts whose log is the device's, and on such a play the play at any run coins that agree with the
coins its history before round `i` shows fixes the same value for round `i`. -/
theorem fixedBefore_of_eq {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop)
    (i : ℕ) (hOpn : ∀ φ o, Opn φ o i → Acc o ∧ Good o)
    (h : ∀ (ω : TapeOf V θ F c σ) (r r' : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r' → Fx (Fr.out ω r') i = Fx (Fr.out ω r) i) :
    FixedBefore Fr Rd Fx Opn i := by
  intro ω r r' φ φ' hφ ho hφ' _ hs
  have e := h ω r r' (hOpn φ _ ho).1 (hOpn φ _ ho).2 hs
  rw [hφ, hφ'] at e
  exact Option.some.inj e

/-- **Two plays that open one fixed value with two messages give a SHA-512 collision**: the finder's `pair` at the
fixed value and the two plays is two different strings with one SHA-512. -/
def Breaks {A : Live₂} (Rd : Reading A) (Opn : Φ → Verity.Exchange → ℕ → Prop)
    (pair : Φ → Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (i : ℕ) : Prop :=
  ∀ φ o o' m m', Opn φ o i → Opn φ o' i → Rd.msg o i = some m → Rd.msg o' i = some m' → m ≠ m' →
    ∃ p, pair φ o o' i = some p ∧ p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2

/-- **Round `i`'s two-continuation finder**, at `q` SHA-512 calls on every coin: on a tape `ω` and run coins `r` and
`y`, the frame's run's history before round `i` at `r`, the value the play at `r` fixes for the round, and `pair` at
it and the plays at `r` and at the coins the history shows with `y` at every other round. -/
noncomputable def finderBy (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A)
    (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop)
    (pair : Φ → Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (i q : ℕ) : Claims.Finder where
  S := (TapeOf V θ F c σ × Coins A x) × Coins A x
  out s :=
    (Fx (outOf V θ F c σ place s.1.1 s.1.2) i).bind fun φ =>
      pair φ (outOf V θ F c σ place s.1.1 s.1.2)
        (outOf V θ F c σ place s.1.1
          (merge (histOf V θ F c σ (opnBy V θ F c σ place Rd Fx Opn) i s.1.1 s.1.2) s.2)) i
  cost _ := q

/-- **Round `i`'s miss is at most the square root of its finder's success**, at the frame's prover: where an accepted
play's reading opens what the play fixes (`Opens`), two plays that open what they fix at coins the history before the
round shows alike fix one value (`FixedBefore`), and two plays that open one fixed value with two messages give a
collision (`Breaks`). -/
theorem missBoundBy (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (Fx : Verity.Exchange → ℕ → Option Φ)
    (Opn : Φ → Verity.Exchange → ℕ → Prop)
    (pair : Φ → Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (i q : ℕ)
    (hO : Opens A x Rd Acc Good Fx Opn)
    (hF : FixedBefore (frameBy V A x θ F c σ place Rd Fx Opn) Rd Fx Opn i) (hB : Breaks Rd Opn pair i) :
    ENNReal.ofReal (Claims.Pr (MissAt (frameBy V A x θ F c σ place Rd Fx Opn) Rd Acc Good i)) ≤
      ENNReal.ofReal (Real.sqrt ((finderBy V θ F c σ place Rd Fx Opn pair i q).Pr Verity.sha512Msg)) := by
  classical
  have hopen : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (outOf V θ F c σ place ω r) →
      Good (outOf V θ F c σ place ω r) → i < A.L.V.k x →
      ∃ m, Rd.msg (outOf V θ F c σ place ω r) i = some m ∧
        opnBy V θ F c σ place Rd Fx Opn i ω (playHist A x r (playMsgs Rd (outOf V θ F c σ place ω r)) i) r =
          some m := by
    intro ω r hA hG hk
    obtain ⟨⟨m, hm⟩, φ, hφ, hop⟩ := hO _ hA hG i hk
    exact ⟨m, hm, openingBy_of (out := outOf V θ F c σ place) (ω := ω) (r := r) hm rfl ⟨φ, hφ, hop⟩⟩
  have key := miss_le_fork V θ F c σ place (opnBy V θ F c σ place Rd Fx Opn) i Rd Acc Good (runShown A x) hopen
  refine le_trans (ENNReal.ofReal_le_ofReal key) (ENNReal.ofReal_le_ofReal (Real.sqrt_le_sqrt ?_))
  unfold Claims.Finder.Pr
  refine Claims.Pr_mono fun s hs => ?_
  obtain ⟨u, w, hu, hw, huw⟩ := hs
  have hu' : openingBy (outOf V θ F c σ place) Rd Fx Opn i s.1.1
      (histOf V θ F c σ (opnBy V θ F c σ place Rd Fx Opn) i s.1.1 s.1.2) s.1.2 = some u := hu
  have hw' : openingBy (outOf V θ F c σ place) Rd Fx Opn i s.1.1
      (histOf V θ F c σ (opnBy V θ F c σ place Rd Fx Opn) i s.1.1 s.1.2)
      (merge (histOf V θ F c σ (opnBy V θ F c σ place Rd Fx Opn) i s.1.1 s.1.2) s.2) = some w := hw
  obtain ⟨hu1, hh, φ, hφ, hou⟩ := openingBy_some hu'
  obtain ⟨hw1, -, φ', hφ', how⟩ := openingBy_some hw'
  have hsh : Shows (playHist A x s.1.2 (playMsgs Rd (outOf V θ F c σ place s.1.1 s.1.2)) i)
      (merge (histOf V θ F c σ (opnBy V θ F c σ place Rd Fx Opn) i s.1.1 s.1.2) s.2) := by
    rw [hh]
    exact shows_merge _ _
  have hφφ : φ' = φ := hF s.1.1 s.1.2
    (merge (histOf V θ F c σ (opnBy V θ F c σ place Rd Fx Opn) i s.1.1 s.1.2) s.2) φ φ' hφ hou hφ' how hsh
  rw [hφφ] at how
  obtain ⟨p, hp, hne, hsha⟩ := hB φ _ _ u w hou how hu1 hw1 huw
  have hout : (finderBy V θ F c σ place Rd Fx Opn pair i q).out s =
      (Fx (outOf V θ F c σ place s.1.1 s.1.2) i).bind fun φ =>
        pair φ (outOf V θ F c σ place s.1.1 s.1.2)
          (outOf V θ F c σ place s.1.1
            (merge (histOf V θ F c σ (opnBy V θ F c σ place Rd Fx Opn) i s.1.1 s.1.2) s.2)) i := rfl
  refine Exists.intro p.1 (Exists.intro p.2 (And.intro ?_ (And.intro hne hsha)))
  show (finderBy V θ F c σ place Rd Fx Opn pair i q).out s = some (p.1, p.2)
  rw [hout, hφ]
  exact hp

/-- **The messages at the rounds the run reaches, at the frame's prover**: where an accepted play's reading opens what
the play fixes, each round's fixed value is fixed by the coins the history before it shows, and two plays that open
one fixed value with two messages give a collision, `MessagesReached` holds at the sum over the rounds of the square
roots of their finders' successes. -/
theorem reachedBy (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (Fx : Verity.Exchange → ℕ → Option Φ)
    (Opn : Φ → Verity.Exchange → ℕ → Prop)
    (pair : Φ → Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (q : ℕ)
    (hO : Opens A x Rd Acc Good Fx Opn)
    (hF : ∀ i < A.L.V.k x, FixedBefore (frameBy V A x θ F c σ place Rd Fx Opn) Rd Fx Opn i)
    (hB : ∀ i < A.L.V.k x, Breaks Rd Opn pair i) :
    MessagesReached V A x θ F c σ (frameBy V A x θ F c σ place Rd Fx Opn) Rd Acc Good
      (∑ i ∈ Finset.range (A.L.V.k x),
        ENNReal.ofReal (Real.sqrt ((finderBy V θ F c σ place Rd Fx Opn pair i q).Pr Verity.sha512Msg))) :=
  reachedOfRounds V A x θ F c σ _ Rd Acc Good _ fun i hi =>
    missBoundBy V A x θ F c σ place Rd Acc Good Fx Opn pair i q hO (hF i hi) (hB i hi)

/-- **The messages' bound at the frame's prover**: where an accepted play's reading opens what the play fixes, each
round's fixed value is fixed by the coins the history before it shows, two plays that open one fixed value with two
messages give a collision, and the run reaches every round, `MessagesFixed` holds at the sum over the rounds of the
square roots of their finders' successes. -/
theorem messagesFixedBy (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (Fx : Verity.Exchange → ℕ → Option Φ)
    (Opn : Φ → Verity.Exchange → ℕ → Prop)
    (pair : Φ → Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (q : ℕ)
    (hO : Opens A x Rd Acc Good Fx Opn)
    (hF : ∀ i < A.L.V.k x, FixedBefore (frameBy V A x θ F c σ place Rd Fx Opn) Rd Fx Opn i)
    (hB : ∀ i < A.L.V.k x, Breaks Rd Opn pair i)
    (hT : RunsThrough (frameBy V A x θ F c σ place Rd Fx Opn) Rd Acc Good) :
    MessagesFixed V A x θ F c σ (frameBy V A x θ F c σ place Rd Fx Opn) Rd Acc Good
      (∑ i ∈ Finset.range (A.L.V.k x),
        ENNReal.ofReal (Real.sqrt ((finderBy V θ F c σ place Rd Fx Opn pair i q).Pr Verity.sha512Msg))) :=
  messagesFixed_of_reached V A x θ F c σ _ Rd Acc Good _ hT
    (reachedBy V A x θ F c σ place Rd Acc Good Fx Opn pair q hO hF hB)

/-! ## The message's own digest -/

/-- **A play opens a digest with the message it reads**: the digest is the message's SHA-512. -/
def digestOpn {A : Live₂} (Rd : Reading A) (φ : Msg) (o : Verity.Exchange) (i : ℕ) : Prop :=
  ∃ m, Rd.msg o i = some m ∧ φ = Verity.sha512Msg m

/-- The two messages two plays read at round `i`. -/
def digestPair {A : Live₂} (Rd : Reading A) (_ : Msg) (o o' : Verity.Exchange) (i : ℕ) : Option (Msg × Msg) :=
  (Rd.msg o i).bind fun m => (Rd.msg o' i).map fun m' => (m, m')

/-- **Two messages with one SHA-512 are a collision.** -/
theorem breaks_digest {A : Live₂} (Rd : Reading A) (i : ℕ) : Breaks Rd (digestOpn Rd) (digestPair Rd) i := by
  intro φ o o' m m' ho ho' hm hm' hne
  obtain ⟨u, hu, hφ⟩ := ho
  obtain ⟨w, hw, hφ'⟩ := ho'
  have hum : u = m := Option.some.inj (hu.symm.trans hm)
  have hwm : w = m' := Option.some.inj (hw.symm.trans hm')
  have hsha : Verity.sha512Msg m = Verity.sha512Msg m' := by
    rw [← hum, ← hwm]
    exact hφ.symm.trans hφ'
  refine ⟨(m, m'), ?_, hne, hsha⟩
  simp [digestPair, hm, hm']

/-- **`Opens` at the message's own digest** is `MsgOpens`. -/
theorem opens_digest {A : Live₂} {x : Msg} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {Dg : Verity.Exchange → ℕ → Option Msg} (h : MsgOpens A x Rd Acc Good Dg) :
    Opens A x Rd Acc Good Dg (digestOpn Rd) := by
  intro o hA hG i hi
  obtain ⟨m, hm, hd⟩ := h o hA hG i hi
  have h1 : digestOpn Rd (Verity.sha512Msg m) o i := by
    unfold digestOpn
    exact ⟨m, hm, rfl⟩
  exact ⟨⟨m, hm⟩, _, hd, h1⟩

/-! ## Strings that open digests -/

/-- **Two lists with one image under `f` that differ hold, at one place, two different entries with one image.** -/
theorem exists_zip_collide {β γ : Type} (f : β → γ) : ∀ (l l' : List β), l.map f = l'.map f → l ≠ l' →
    ∃ p ∈ l.zip l', p.1 ≠ p.2 ∧ f p.1 = f p.2
  | [], [], _, h => absurd rfl h
  | [], _ :: _, hm, _ => by simp at hm
  | _ :: _, [], hm, _ => by simp at hm
  | a :: l, b :: l', hm, h => by
    rw [List.map_cons, List.map_cons, List.cons.injEq] at hm
    by_cases hab : a = b
    · obtain ⟨p, hp, h1, h2⟩ := exists_zip_collide f l l' hm.2 fun e => h (by rw [hab, e])
      refine ⟨p, ?_, h1, h2⟩
      rw [List.zip_cons_cons]
      exact List.mem_cons.2 (Or.inr hp)
    · refine ⟨(a, b), ?_, hab, hm.1⟩
      rw [List.zip_cons_cons]
      exact List.mem_cons.2 (Or.inl rfl)

variable {κ : Type}

/-- **What a play fixes for round `i`, as digests and a context**: the digests `Dg` reads, with the context `ctx`. -/
def stringsFx (Dg : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) (o : Verity.Exchange)
    (i : ℕ) : Option (List Msg × κ) :=
  (Dg o i).map fun ds => (ds, ctx o i)

/-- **A play opens digests and a context with the strings it shows**: the check accepts the play, its log is the
device's, its context is the one fixed, and the strings it shows for the round have the digests, one each. -/
def stringsOpn (Acc Good : Verity.Exchange → Prop) (raw : Verity.Exchange → ℕ → Option (List Msg))
    (ctx : Verity.Exchange → ℕ → κ) (φ : List Msg × κ) (o : Verity.Exchange) (i : ℕ) : Prop :=
  Acc o ∧ Good o ∧ ctx o i = φ.2 ∧ ∃ bs, raw o i = some bs ∧ bs.map Verity.sha512Msg = φ.1

open Classical in
/-- **Two strings two plays show at one place for round `i`, different, with one SHA-512**, if there are any. -/
noncomputable def stringsPair (raw : Verity.Exchange → ℕ → Option (List Msg)) (_ : List Msg × κ)
    (o o' : Verity.Exchange) (i : ℕ) : Option (Msg × Msg) :=
  if h : ∃ p : Msg × Msg, (p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2) ∧
      ∃ bs bs', raw o i = some bs ∧ raw o' i = some bs' ∧ p ∈ bs.zip bs' then some h.choose else none

/-- **The round's message is a function of the strings and the context**, on accepted plays: two plays the check
accepts whose logs are the device's, with one context and the same strings for round `i`, read the same message at
round `i`. -/
def Explains {A : Live₂} (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) (i : ℕ) : Prop :=
  ∀ o o', Acc o → Good o → Acc o' → Good o' → ctx o i = ctx o' i → raw o i = raw o' i → Rd.msg o i = Rd.msg o' i

/-- **Two accepted plays with two messages show two lists of strings under one list of digests**, where the message
is a function of the strings and the context: the finder's pair is a SHA-512 collision. -/
theorem breaks_strings {A : Live₂} (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) (i : ℕ)
    (hE : Explains Rd Acc Good raw ctx i) :
    Breaks Rd (stringsOpn Acc Good raw ctx) (stringsPair raw) i := by
  classical
  intro φ o o' m m' ho ho' hm hm' hne
  obtain ⟨hA, hG, hc, bs, hbs, hd⟩ := ho
  obtain ⟨hA', hG', hc', bs', hbs', hd'⟩ := ho'
  have hraw : bs ≠ bs' := by
    intro e
    have := hE o o' hA hG hA' hG' (hc.trans hc'.symm) (by rw [hbs, hbs', e])
    rw [hm, hm'] at this
    exact hne (Option.some.inj this)
  obtain ⟨p, hp, hp1, hp2⟩ := exists_zip_collide Verity.sha512Msg bs bs' (hd.trans hd'.symm) hraw
  have hex : ∃ p : Msg × Msg, (p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2) ∧
      ∃ bs bs', raw o i = some bs ∧ raw o' i = some bs' ∧ p ∈ bs.zip bs' :=
    ⟨p, ⟨hp1, hp2⟩, bs, bs', hbs, hbs', hp⟩
  refine ⟨hex.choose, ?_, hex.choose_spec.1.1, hex.choose_spec.1.2⟩
  unfold stringsPair
  split
  next => rfl
  next h => exact absurd hex h

/-- **An accepted play shows strings that have its digests**: on a play the check accepts whose log is the device's,
at each round below `k` the reading reads a message, the play fixes a list of digests, and the strings the play shows
for the round have them, one each. -/
def ShowsStrings (A : Live₂) (x : Msg) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) : Prop :=
  ∀ o, Acc o → Good o → ∀ i < A.L.V.k x, (∃ m, Rd.msg o i = some m) ∧
    ∃ ds bs, Dg o i = some ds ∧ raw o i = some bs ∧ bs.map Verity.sha512Msg = ds

/-- **`Opens` at the strings an accepted play shows**, at any context. -/
theorem opens_strings {A : Live₂} {x : Msg} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {Dg raw : Verity.Exchange → ℕ → Option (List Msg)} (ctx : Verity.Exchange → ℕ → κ)
    (h : ShowsStrings A x Rd Acc Good Dg raw) :
    Opens A x Rd Acc Good (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx) := by
  intro o hA hG i hi
  obtain ⟨hm, ds, bs, hd, hr, hb⟩ := h o hA hG i hi
  have hfx : stringsFx Dg ctx o i = some (ds, ctx o i) := by
    simp [stringsFx, hd]
  have hop : stringsOpn Acc Good raw ctx (ds, ctx o i) o i := by
    unfold stringsOpn
    exact ⟨hA, hG, rfl, bs, hr, hb⟩
  exact ⟨hm, (ds, ctx o i), hfx, hop⟩

/-- **`FixedBefore` at digests and a context**: where two plays of one tape the check accepts whose logs are the
device's, the second at run coins that agree with the coins the first's history before round `i` shows, read the same
digests and the same context for round `i`. -/
theorem fixedBefore_strings {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (ctx : Verity.Exchange → ℕ → κ) (i : ℕ)
    (h : ∀ (ω : TapeOf V θ F c σ) (r r' : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) → Acc (Fr.out ω r') →
      Good (Fr.out ω r') → Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r' →
      Dg (Fr.out ω r') i = Dg (Fr.out ω r) i ∧ ctx (Fr.out ω r') i = ctx (Fr.out ω r) i) :
    FixedBefore Fr Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx) i := by
  intro ω r r' φ φ' hφ ho hφ' ho' hs
  obtain ⟨hd, hc⟩ := h ω r r' ho.1 ho.2.1 ho'.1 ho'.2.1 hs
  unfold stringsFx at hφ hφ'
  obtain ⟨ds, hds, rfl⟩ := Option.map_eq_some_iff.1 hφ
  obtain ⟨ds', hds', rfl⟩ := Option.map_eq_some_iff.1 hφ'
  rw [hd, hds] at hds'
  rw [hc, Option.some.inj hds']

/-- **The messages at the rounds the run reaches, where the plays show strings**: `reachedBy` from `ShowsStrings`,
`FixedBefore` and `Explains`. -/
theorem reachedStrings (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) (q : ℕ)
    (hO : ShowsStrings A x Rd Acc Good Dg raw)
    (hF : ∀ i < A.L.V.k x, FixedBefore
      (frameBy V A x θ F c σ place Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx)) Rd
      (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx) i)
    (hE : ∀ i < A.L.V.k x, Explains Rd Acc Good raw ctx i) :
    MessagesReached V A x θ F c σ
      (frameBy V A x θ F c σ place Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx)) Rd Acc Good
      (∑ i ∈ Finset.range (A.L.V.k x),
        ENNReal.ofReal (Real.sqrt ((finderBy V θ F c σ place Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx)
          (stringsPair raw) i q).Pr Verity.sha512Msg))) :=
  reachedBy V A x θ F c σ place Rd Acc Good (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx) (stringsPair raw) q
    (opens_strings ctx hO) hF (fun i hi => breaks_strings Rd Acc Good raw ctx i (hE i hi))

/-- **The messages' bound where the plays show strings**: where an accepted play shows, for each round, strings that
have the digests the play fixes (`ShowsStrings`), the digests and the context are fixed by the coins the history before
the round shows (`FixedBefore`), the round's message is a function of the strings and the context (`Explains`), and
the run reaches every round, `MessagesFixed` holds at the sum over the rounds of the square roots of the successes of
the finders that output two of the strings two plays show at one place. -/
theorem messagesFixedStrings (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) (q : ℕ)
    (hO : ShowsStrings A x Rd Acc Good Dg raw)
    (hF : ∀ i < A.L.V.k x, FixedBefore
      (frameBy V A x θ F c σ place Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx)) Rd
      (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx) i)
    (hE : ∀ i < A.L.V.k x, Explains Rd Acc Good raw ctx i)
    (hT : RunsThrough (frameBy V A x θ F c σ place Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx)) Rd Acc
      Good) :
    MessagesFixed V A x θ F c σ (frameBy V A x θ F c σ place Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx))
      Rd Acc Good
      (∑ i ∈ Finset.range (A.L.V.k x),
        ENNReal.ofReal (Real.sqrt ((finderBy V θ F c σ place Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx)
          (stringsPair raw) i q).Pr Verity.sha512Msg))) :=
  messagesFixedBy V A x θ F c σ place Rd Acc Good (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx)
    (stringsPair raw) q (opens_strings ctx hO) hF
    (fun i hi => breaks_strings Rd Acc Good raw ctx i (hE i hi)) hT

end Verity.SampledTwo.Messages
