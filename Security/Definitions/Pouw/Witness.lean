import Definitions.Pouw.Dither

/-!
# The witness machine, protocol and period that `WorkSound`'s witnesses are stated at (trusted)

`Proofs.Pouw.PoolWitness` states the honest and nontrivial witnesses of `WorkSound`, `WorkMoment` and `FleetSound`
over these definitions, and `Proofs.Pouw.DitherWitness` proves every premise of
`Definitions.Pouw.Guarantees.Dither.WorkSound` at them, at `k = 2` sampling blocks and the non-uniform prices
`μ = (2, 1, 1)`.

* `tm`, `tw`: the witness machine over three resources (`0` MAC units, `1` hash units, `2` the rest). A program is
  any function with its use and its oracle calls, each call a step that uses a hash unit (G3, `callUse = (0, 1, 0)`).
  The machine's laws (`TAlg`) charge it a MAC unit for each word cell it outputs that varies over its inputs, a hash
  unit for each root cell that varies, and a sampling block's work (dither in, state and root out) both for each root
  cell that varies. A cell that is constant over every input is free: a program can guess a word or a root cell at no
  cost (a free guess). No law reads an answer's output, so answers are free.
* `twNF`, `twRound`: the witness protocol, a no-fold round with two words (`n = 2`, `n' = nb = L = 1`). The noise is
  the word: word `i` is the dither's `d i`, one of 16 values, so a guessed word passes its check with probability
  `p = 1/16`. The check draws one of the two words (`s = 1`). The root is transparent: root cell `i` is word `i`
  reduced into `Fin 17` (`dec`), and a cell opens where its root cell reads back as the word (`enc`). There is no
  oracle (`Orc = Unit`), so the commitment is perfectly binding.
* The period (`twPd`): credited use `hon = (4, 4, 2)`, two honest sampling blocks whose commit phase commits the zero
  instance at no cost, whose work uses `(2, 2, 1)` and makes two oracle calls, and whose answer uses nothing (`honest`,
  timed use `hon`), `T = 8`, and one other job per resource worth its use at `μ`. The cap on a run's calls is `Q = 8`,
  the period's capacity of resource 1, which each call uses; the honest prover makes 4, and its answers none. Each
  sampling block's honest cost is `H = H_mm + H_sha + H_o = 4 + 2 + 1` and the public bests are the honest ones
  (`pub_mm = 4`, `pub_sha = 2`), so `C = 14`. Every instance is registrable for every block (`twReg`): the noise reads
  none.
* The verdict (`twVerdict`) is the sampling blocks' checks at their committed instances and the devices' reading
  (`twDev`): that every block's answer ran within the period, resource by resource. That reading stands in for the
  card's clock (`Definitions.Pouw.Guarantees.DitherInstance`), with every check coin selected (`twSel`).
* The headline's bounds at the witness, prover by prover: `twε` (`ε_A = 1/7 + ln 2 / 2`) and `twκ` (`κ_A = 1/2`),
  with the advice size `twBadv` and no walk or log (`twξ`).
-/

namespace Pouw.Witness

open Pouw.RelativeSecurity Pouw.Dither Finset

section Machine

/-- A root cell: a word reduced into 17 values. 17 is odd, so no product with the roots' type (`289` values) has as
many values as the words' type (`2^64`), which the laws read. -/
abbrev RCell : Type := Fin 17

/-- The witness protocol's words, one per cell `(i, j, b)` with `j b : Fin 1`. -/
abbrev TTr : Type := Fin 2 → Fin 1 → Fin 1 → Word

/-- The witness protocol's roots, one cell per word. -/
abbrev TRoot : Type := Fin 2 → Fin 1 → Fin 1 → RCell

/-- The witness protocol's dither: one of 16 values per word. -/
abbrev TDraw : Type := Fin 2 → Fin 16

open Classical in
/-- The cells of `f`'s output that vary over its inputs. -/
noncomputable def nonconst {X α : Type} (f : X → Fin 2 → Fin 1 → Fin 1 → α) : Finset (Fin 2) :=
  univ.filter fun i => ∃ x y, f x i 0 0 ≠ f y i 0 0

open Classical in
theorem mem_nonconst {X α : Type} {f : X → Fin 2 → Fin 1 → Fin 1 → α} {c : Fin 2} :
    c ∈ nonconst f ↔ ∃ x y, f x c 0 0 ≠ f y c 0 0 := by
  simp [nonconst]

theorem card_le_two (s : Finset (Fin 2)) : (s.card : ℝ) ≤ 2 := by
  have h := card_le_univ s
  rw [Fintype.card_fin] at h
  exact_mod_cast h

/-- **A witness program**: what it computes, its use of the three resources (nonnegative), its oracle calls (each
uses a hash unit), and the machine's three laws, each read where the program's type has the shape it is about:
* `words`: a program that outputs words uses a MAC unit for each word cell that varies over its inputs;
* `roots`: a program that outputs a root uses a hash unit for each root cell that varies;
* `work`: a sampling block's work (oracle, state and dither in; state and root out) uses a MAC unit and a hash unit
  for each root cell that varies: its root's words are computed and hashed. -/
structure TAlg (I O : Type) where
  run : I → O
  use : I → Fin 3 → ℝ
  use_nonneg : ∀ x r, 0 ≤ use x r
  calls : I → ℝ
  calls_nonneg : ∀ x, 0 ≤ calls x
  calls_le : ∀ x, calls x ≤ use x 1
  words : ∀ (h : O = TTr) (x : I), ((nonconst fun y => cast h (run y)).card : ℝ) ≤ use x 0
  roots : ∀ (h : O = TRoot) (x : I), ((nonconst fun y => cast h (run y)).card : ℝ) ≤ use x 1
  work : ∀ (S : Type), I = (Unit × (S × TDraw)) → ∀ (h : O = (S × TRoot)) (x : I),
    ((nonconst fun y => (cast h (run y)).2).card : ℝ) ≤ use x 0 ∧
      ((nonconst fun y => (cast h (run y)).2).card : ℝ) ≤ use x 1

/-- Any function, at the use `u` plus two units of every resource (which pays for any two cells). -/
def TAlg.ofFun {I O : Type} (f : I → O) (u : I → Fin 3 → ℝ) (hu : ∀ x r, 0 ≤ u x r) : TAlg I O where
  run := f
  use x r := 2 + u x r
  use_nonneg x r := by linarith [hu x r]
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le x := by linarith [hu x 1]
  words h x := by linarith [card_le_two (nonconst fun y => cast h (f y)), hu x 0]
  roots h x := by linarith [card_le_two (nonconst fun y => cast h (f y)), hu x 1]
  work _ _ h x := ⟨by linarith [card_le_two (nonconst fun y => (cast h (f y)).2), hu x 0],
    by linarith [card_le_two (nonconst fun y => (cast h (f y)).2), hu x 1]⟩

/-- The witness machine. Its `time` is zero (the headline reads `Accel.use`, not `time`), and composing or routing
programs pays two units of every resource, the most any program's cells can cost. -/
abbrev tm : Machine where
  Alg I O := TAlg I O
  eval f := f.run
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := TAlg.ofFun (g.run ∘ f.run) (fun x r => f.use x r + g.use (f.run x) r)
    fun x r => add_nonneg (f.use_nonneg x r) (g.use_nonneg _ r)
  pair f g := TAlg.ofFun (fun x => (f.run x, g.run x)) (fun x r => f.use x r + g.use x r)
    fun x r => add_nonneg (f.use_nonneg x r) (g.use_nonneg x r)
  fst := TAlg.ofFun Prod.fst (fun _ _ => 0) fun _ _ => le_rfl
  snd := TAlg.ofFun Prod.snd (fun _ _ => 0) fun _ _ => le_rfl
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

/-- The witness accelerator: three resources at rate 1, a run's wall time its total use, nothing exchanged, and an
oracle call a step that uses one hash unit. -/
noncomputable abbrev tw : Accel where
  Mc := tm
  Res := Fin 3
  rate _ := 1
  mem := 0
  use f x r := TAlg.use f x r
  held _ _ := 0
  wall f x := ∑ r, TAlg.use f x r
  ext _ _ := 0
  calls f x := TAlg.calls f x
  callUse r := if r = 1 then 1 else 0
  P1 := True
  P2 := True

theorem card_ttr : Nat.card TTr = 18446744073709551616 := by
  simp [Nat.card_eq_fintype_card, ZMod.card]

theorem card_troot : Nat.card TRoot = 289 := by
  simp [Nat.card_eq_fintype_card]

end Machine

section Round

/-- A word reduced into a root cell: its value where it is below 17, else 0. -/
def dec (w : Word) : RCell := if h : w.val < 17 then ⟨w.val, h⟩ else 0

/-- A root cell read back as a word. -/
def enc (v : RCell) : Word := (v.val : Word)

/-- **The witness protocol's knobs**: two words (`n = 2`, `n' = nb = L = 1`), the noise the word itself (word `i` is
the dither's `d i`), a check of one of the two words (`s = 1`), no oracle, and the transparent root: each word reduced
into `Fin 17`, a cell opening where its root cell reads back as the word. -/
def twNF : NoFold where
  n := 2
  n' := 1
  nb := 1
  L := 1
  Draw := TDraw
  noise _ d := (fun i _ => ((d i : ℕ) : Word), fun _ _ => 1)
  s := 1
  Chk := Fin 2
  cells c _ := (c, 0, 0)
  Orc := Unit
  Root := TRoot
  merkle _ T i j b := dec (T i j b)
  Path := Unit
  opens _ r c v _ := enc (r c.1 c.2.1 c.2.2) = v

/-- The witness protocol. -/
abbrev twRound : RoundProto := noFoldRound twNF

end Round

section Period

/-- The prices: a MAC unit at 2, a hash unit and a unit of the rest at 1. -/
noncomputable def twμ (r : Fin 3) : ℝ := if r = 0 then 2 else 1

/-- The period's credited use: two honest sampling blocks of `(2, 2, 1)`. -/
noncomputable def twHon (r : Fin 3) : ℝ := if r = 2 then 2 else 4

/-- **The witness period**: `T = 8` seconds of the three resources at rate 1, one other job per resource worth its
use at `μ`, and the credited use `hon` worth `C = μ · hon = 14`. -/
noncomputable def twPd : Period tw where
  T := 8
  J := Fin 3
  val := twμ
  needs j r := if j = r then 1 else 0
  hon := twHon
  rv := 14

/-- A sampling block's honest matmul cost at `μ` (two words, a MAC each), its public best's too. -/
noncomputable def twHmm (_ : Fin 2) : ℝ := 4

/-- A sampling block's honest hashing cost at `μ` (two root cells, a hash each), its public best's too. -/
noncomputable def twHsha (_ : Fin 2) : ℝ := 2

/-- The rest of a sampling block's honest cost at `μ`. -/
noncomputable def twHo (_ : Fin 2) : ℝ := 1

/-- The rate a unit of priced cost: `η_mm = 1/4`, `η_sha = 1/2`. -/
noncomputable def twη : ℝ := 1 / 7

/-- The confidence. -/
noncomputable def twδ : ℝ := 1 / 2

/-- The cap on a run's oracle calls: the period's hash units over a call's use of them. -/
noncomputable def twQ : ℝ := 8

/-- The registrable instances of each sampling block: every instance, since the witness's noise reads none. -/
def twReg (_ : Fin 2) : Set twRound.Inst := Set.univ

/-- The instance the honest prover commits. -/
def twInst : twRound.Inst := (fun _ _ => 0, fun _ _ => 0)

/-- The selected check coins: every one. -/
def twSel : twRound.Chk → Prop := fun _ => True

end Period

section Premises

/-- **The witness verdict**: every sampling block's check on the transcript at its committed instance, and the
devices' reading. -/
def twVerdict : Verdict twRound 2 Prop := fun o x t ob =>
  (∀ w, twRound.accept o (t w).1 (x w).1 (x w).2 (t w).2.1 (t w).2.2) ∧ ob

/-- **The devices' reading**: every sampling block's answer ran within the period, resource by resource. It stands in
for the card's clock, which has each answer arrive before the period's end. -/
def twDev : Readings tw twRound 2 Prop := fun A o x => ∀ b r, tw.use (A.answer b) (A.answerIn o x b) r ≤ twPd.cap r

/-- The extractor's advice at sampling block `w`: the coins of the blocks before `w`, which is all the block's state
reads besides the oracle (`Unit` here). -/
abbrev twAdv : Type := Fin 2 → Option (TDraw × Fin 2)

/-- The advice size every prover's extraction reads. -/
def twBadv (_ : Prover tw twRound 2) : ℕ := Fintype.card twAdv

/-- The extraction's walk and log: none. -/
def twξ (_ : Prover tw twRound 2) (_ : Fin 3) : ℝ := 0

end Premises

section Honest

theorem card_nonconst_const {X α : Type} {f : X → Fin 2 → Fin 1 → Fin 1 → α} (hf : ∀ x y, f x = f y) :
    ((nonconst f).card : ℝ) = 0 := by
  have : nonconst f = ∅ := eq_empty_of_forall_notMem fun c hc => by
    obtain ⟨x, y, h⟩ := mem_nonconst.1 hc
    exact h (by rw [hf x y])
  rw [this, card_empty, Nat.cast_zero]

/-- A constant program: it outputs `v` at no use and no calls. No cell of its output varies, so no law charges it. -/
noncomputable def constAlg {I O : Type} (v : O) : TAlg I O where
  run _ := v
  use _ _ := 0
  use_nonneg _ _ := le_rfl
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := le_rfl
  words _ _ := le_of_eq (card_nonconst_const fun _ _ => rfl)
  roots _ _ := le_of_eq (card_nonconst_const fun _ _ => rfl)
  work _ _ _ _ := ⟨le_of_eq (card_nonconst_const fun _ _ => rfl), le_of_eq (card_nonconst_const fun _ _ => rfl)⟩

/-- The honest words at the dither `d`: word `i` is `d i`. -/
def hw (d : TDraw) : TTr := fun i _ _ => ((d i : ℕ) : Word)

/-- The honest sampling block's work: its two words (a MAC unit each), their root (a hash unit each) and a unit of the
rest. Its state is the words. -/
noncomputable def hWork : TAlg (Unit × TTr × TDraw) (TTr × TRoot) where
  run p := (hw p.2.2, fun i j b => dec (hw p.2.2 i j b))
  use _ r := if r = 2 then 1 else 2
  use_nonneg _ r := by split_ifs <;> norm_num
  calls _ := 2
  calls_nonneg _ := by norm_num
  calls_le _ := by norm_num
  words _ _ := (card_le_two _).trans (by simp)
  roots _ _ := (card_le_two _).trans (by simp)
  work _ _ _ _ := ⟨(card_le_two _).trans (by simp), (card_le_two _).trans (by simp)⟩

theorem card_ans : Nat.card (Fin 1 → Word × Unit) = 4294967296 := by
  simp [Nat.card_eq_fintype_card, ZMod.card]

/-- Sampling block `b`'s honest answer: the checked word, from the state after the block's work, at the block's own
check coin among the period's. It outputs neither words nor a root, so no law charges it. -/
noncomputable def hAnswer (b : Fin 2) : TAlg (Unit × TTr × (Fin 2 → TDraw × Fin 2)) (Fin 1 → Word × Unit) where
  run p := fun _ => (p.2.1 (p.2.2 b).2 0 0, ())
  use _ _ := 0
  use_nonneg _ _ := le_rfl
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := le_rfl
  words h _ := absurd (congrArg Nat.card h) (by rw [card_ans, card_ttr]; omega)
  roots h _ := absurd (congrArg Nat.card h) (by rw [card_ans, card_troot]; omega)
  work S _ h _ := absurd (congrArg Nat.card h) (by rw [card_ans, Nat.card_prod, card_troot]; omega)

/-- **The honest prover**: every sampling block's commit phase commits the zero instance, its work computes the words
and commits them, keeping the words as its state, and its answer opens the checked one. -/
noncomputable def honest : Prover tw twRound 2 where
  St := TTr
  start := hw fun _ => 0
  commit _ := constAlg (hw fun _ => 0, twInst)
  work _ := hWork
  absorb _ s _ := s
  answer b := hAnswer b

end Honest

section Bounds

/-- The headline's error at the witness, prover by prover (`WorkSound`'s `ε_A`): `1/7 + ln 2 / 2`. -/
noncomputable def twε (A : Prover tw twRound 2) : ℝ :=
  epsWorkSound (fun w => twHmm w + twHsha w + twHo w) twHmm twHsha twη twδ 0 0 +
    (∑ r, twμ r * twξ A r) / ∑ w, (twHmm w + twHsha w + twHo w)

/-- The headline's probability at the witness, prover by prover (`WorkSound`'s `κ_A`): `δ = 1/2`. -/
noncomputable def twκ (_ : Prover tw twRound 2) : ℝ := twδ + 0 + 0 + 0 + ((2 : ℕ) : ℝ) * (0 + 0)

end Bounds

end Pouw.Witness
