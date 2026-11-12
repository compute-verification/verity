module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.Challenger.CanonRead
import all Spot.Run

/-!
# The first stage's coins are the device's

Where the two-stage form's check accepts a play (`FormAccepts`) and the log the check read is the device's (`Good`), the
coin the form's reading takes for round 0 (`formReading`'s `coin`, the first stage's bytes padded with zeros to one
coin) agrees, on the bytes the first stage read, with every coin the device's release for round 0 agrees with
(`coinsAgree_first`).

**Agreement, not equality.** The projection's `CoinsRead` asks the reading's coin to *equal* the device's release
(`Rd.coin o i = Rd.dev o.log i`). At round 0 it can't: the first stage reads a prefix of the challenger's coin (ten
bytes, say), the reading pads that prefix with zeros, and the device released the whole coin, whose bytes past the
prefix are not zeros. What the projection uses is weaker (its `AcceptsRun` premise): the reading's coin agrees with
the chosen coin wherever the device's release does. `CoinsAgree` says that; `CoinsRead` implies it
(`coinsAgree_of_coinsRead`), and with the coin map's second law it gives `AcceptsRun`'s coin premise.

**What it rests on**, each a named `Prop` taken as a hypothesis:
* `FirstShown`: the play's log shows, after the first stage's digest in the first draw's family, the bytes the first
  stage read. It is the meeting of the run's coin check (`Run.coinsHold`) with the cut's `LogIsDevice`.
* `DevFirst`: the coin map's reading of round 0 starts with those bytes (the coin map's to prove).
* `SourceRead`: the first stage's bytes are no longer than what it read. The run checks only the bytes it read
  (`i.source.extract 0 read`); the form refuses nothing past them today.
* `CoinsOfBytes`: every byte string of at most one coin's bytes is a coin's bytes followed by zeros.

What the run itself establishes, at either value of the form's switch, is proved here (`run_firstRead`): an accepted
run's first stage read `read` of its bytes under `Select.select` and its coins hold on them.

**Not established here.** The run's coin check takes the coin family from the evidence's segment (`firstDraw`), not
from the parameters, and checks a signature over the coins without binding it to the current monitoring period: a
signature from an earlier period passes it. Neither is worked around: `FirstShown` and `DevFirst` read the family the
evidence names.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)

/-! ## The statement -/

/-- **The coins the check reads at group `g` agree with the device's releases**: on a play the check accepts whose log
is the device's, wherever the reading reads a coin `c` of group `g`, `c` agrees with every coin `c'` the device's release
for that round agrees with. -/
def CoinsAgree (A : Live₂) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (g : Group) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Good o → ∀ i, Rd.reads i → groupOf A i = g → ∀ c, Rd.coin o i = some c →
    ∀ c', (∀ d, Rd.dev o.log i = some d → Rd.agree o i d c') → Rd.agree o i c c'

/-- **Equal coins agree**: `CoinsRead` gives `CoinsAgree`. -/
theorem coinsAgree_of_coinsRead {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop} {g : Group}
    (h : CoinsRead A Rd Acc Good g) : CoinsAgree A Rd Acc Good g := by
  intro o ha hg i hr hgi c hc c' hd
  exact hd c ((h o ha hg i hr hgi).symm.trans hc)

/-! ## The first stage as the run runs it -/

/-- **What the first stage read**: as `Run.run` runs it, the work table the rule expands, the rule `Select.resolve`
gives, and `Select.select` on the first stage's bytes, which read `read` of them; and the coins hold on those bytes
(`Run.coinsHold`, at the boundary record and the window root the verifier holds). -/
def FirstRead (i : Spot.Run.Input) (read : ℕ) : Prop :=
  ∃ (workTable : Option Json) (resolved : Spot.Select.Rule) (draw : Json) (rest : Array ℕ),
    i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
      .ok workTable ∧
    Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
    Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
      .ok (draw, rest, read) ∧
    Spot.Run.coinsHold i.coins i.record i.params.verifier.windowRoot (i.source.extract 0 read) = true

/-- **The first stage's draw on a case**: the coin family its coin evidence's segment names, and the boundary
registration's digest, after which the run's coin check holds its coins. -/
def firstDraw (k : Case) : Option (String × ByteArray) := do
  let c ← (Spot.OneStage.commitmentOf k.commitment).toOption
  let e ← (VStar.Recursive.SampledTwo.evidenceOf k.evidence).toOption
  let .str l := Spot.getD (Spot.getD e.coins "segment") "label" | none
  pure (l, c.digest)

/-! ## What the statement rests on -/

/-- **The play's log shows the bytes the first stage read**: on a play the form accepts whose log is the device's, the
bits of the bytes the run's first stage read, where it read any, start the coins the log shows released in the first
draw's family after its digest. -/
def FirstShown (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ i read, inputOn C o = some i → FirstRead i read →
    0 < (i.source.extract 0 read).size →
    ∃ label after, (C o).bind firstDraw = some (label, after) ∧
      Properties.Challenger.bitsOf (i.source.extract 0 read) <+:
        (Verity.Sampled.freshCoins label after o.log).flatten

/-- **The coin map reads round 0 from the first draw's releases**: on a play the form accepts, where bytes `b` of at most
one coin's length, whose bits start the coins the log shows released in the first draw's family after its digest, are
not empty, the coin map's coin for round 0 starts with `b`. -/
def DevFirst (C : CaseOf) (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) : Prop :=
  ∀ o, FormAccepts C o → ∀ label after (b : ByteArray), (C o).bind firstDraw = some (label, after) →
    Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins label after o.log).flatten →
    0 < b.size → b.size ≤ SpotFlock.L →
    ∃ d, dev o.log 0 = some d ∧ (SpotFlock.src d).extract 0 b.size = b

/-- **The first stage's bytes are those it read**: on a play the form accepts, the run's first stage read every byte the
evidence carries for it. -/
def SourceRead (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ i read, inputOn C o = some i → FirstRead i read → i.source.size ≤ read

/-- **Bytes are a coin's prefix**: every byte string of at most one coin's length is a coin's bytes followed by zeros. -/
def CoinsOfBytes : Prop :=
  ∀ b : ByteArray, b.size ≤ SpotFlock.L → ∃ c, SpotFlock.src c = b ++ ⟨Array.replicate (SpotFlock.L - b.size) 0⟩

/-! ## The run -/

private theorem getD_mkObj_none {l : List (String × Json)} {k : String} (hk : (l.map Prod.fst).contains k = false) :
    Spot.getD (Json.mkObj l) k = .null := by
  rw [Json.mkObj, Properties.Challenger.CanonRead.getD_obj,
    Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk]
  all_goals rfl

/-- A refused run names no symbol. -/
private theorem symbol_refused (step : String) (codes : Array String) :
    Spot.getD (Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) "symbol" = .null :=
  getD_mkObj_none (by simp)

private theorem ite_eq_ok {c : Prop} [Decidable c] {a j : Json} {rest : Flock.V Json}
    (h : (if c then Except.ok a else rest) = .ok j) : (c ∧ a = j) ∨ (¬c ∧ rest = .ok j) := by
  by_cases hc : c
  · left; rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at h; exact ⟨hc, Except.ok.inj h⟩
  · right; rw [ite_eq_right_of_eq_false _ _ (eq_false hc)] at h; exact ⟨hc, h⟩

-- `run` unfolds to about 50 decided leaves under one `match` or `if` per step, hence the heartbeat budget.
set_option maxHeartbeats 8000000 in
/-- **A run's answer**, at either value of the form's switch: refused at a step, or past every step, its first stage
on its bytes with coins that hold, its second stage over the first stage's proved units, and the run's interior audit. -/
private theorem run_path {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ∃ (workTable : Option Json) (resolved : Spot.Select.Rule) (draw : Json) (rest : Array ℕ) (read : ℕ),
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      ¬i.closure.isSome = true ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (draw, rest, read) ∧
      ¬(!Spot.Run.coinsHold i.coins i.record i.params.verifier.windowRoot (i.source.extract 0 read)) = true ∧
      ∃ (inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)) (acc₂ : Bool)
        (staged : List (String × Json)) (rest' : List (String × Json)),
        Spot.Run.secondStage i ((Spot.Draw.proved draw).map Int.toNat) = .ok (.ok inner) ∧
        Spot.Run.interiorAudit form i ((Spot.Draw.proved draw).map Int.toNat) inner = .ok (acc₂, staged) ∧
        j = Json.mkObj (rest' ++ staged) := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | exact Or.inr ⟨_, _, _, _, _, by assumption, by assumption, by assumption, by assumption, by assumption,
        _, _, _, _, by assumption, by assumption, (Except.ok.inj h).symm⟩

/-- **An accepted run's first stage read its bytes with coins that hold**, at either value of the form's switch. -/
theorem run_firstRead {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j)
    (hs : Spot.jeq (Spot.getD j "symbol") (Json.str "accept") = true) : ∃ read, FirstRead i read := by
  rcases run_path h with ⟨step, codes, rfl⟩ | ⟨wt, res, draw, rest, read, h1, -, h2, h3, h4, -⟩
  · rw [symbol_refused] at hs
    exact absurd hs (by decide)
  · exact ⟨read, wt, res, draw, rest, h1, h2, h3, by simpa using h4⟩

/-- **The run of record's input on an accepted case** is the one the check ran and accepted. -/
theorem inputOn_accepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input} (ha : FormAccepts C o)
    (hi : inputOn C o = some i) :
    ∃ k p c e c₂ ξ₂ r run, C o = some k ∧ Decoded k p c e ∧
      VStar.Recursive.SampledTwo.inputAt p c e.source e k.blobs = .ok (c₂, ξ₂, r, i) ∧
      Spot.Run.run i true = .ok run ∧ Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true := by
  obtain ⟨k, hk, p, c, e, pub, prof, hd, hc⟩ := ha
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, hrun, hsym, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hc
  have hik : inputOfCase k = some i' := by
    obtain ⟨h1, h2, h3⟩ := hd
    unfold inputOfCase
    rw [h1, h2, h3]
    simp [bind, Except.bind, hin, pure, Except.pure]
  have : i = i' := by
    rw [inputOn, hk, Option.bind_some, hik] at hi
    exact (Option.some.inj hi).symm
  subst this
  exact ⟨k, p, c, e, c₂, ξ₂, r, run, hk, hd, hin, hrun, hsym⟩

/-- **An accepted play's run read its first stage's bytes with coins that hold.** -/
theorem firstRead_of_accepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input} (ha : FormAccepts C o)
    (hi : inputOn C o = some i) : ∃ read, FirstRead i read := by
  obtain ⟨-, -, -, -, -, -, -, run, -, -, -, hrun, hsym⟩ := inputOn_accepts ha hi
  exact run_firstRead hrun hsym

/-! ## The first stage's coin -/

/-- The bytes `b ++ z` start with `b`. -/
theorem extract_append_self (b z : ByteArray) : (b ++ z).extract 0 b.size = b :=
  ByteArray.extract_append_eq_left rfl

/-- Reading at least a byte string's length reads all of it. -/
theorem extract_of_size_le {b : ByteArray} {n : ℕ} (h : b.size ≤ n) : b.extract 0 n = b := by
  ext1
  rw [ByteArray.data_extract]
  exact Array.extract_eq_self_of_le (by simpa using h)

/-- **The reading's round-0 coin starts with the first stage's bytes.** -/
theorem coinOf_src {b : ByteArray} {c : FlockDeploy.Coin} (hB : CoinsOfBytes) (h : coinOf b = some c) :
    b.size ≤ SpotFlock.L ∧ (SpotFlock.src c).extract 0 b.size = b := by
  unfold coinOf at h
  by_cases hL : b.size ≤ SpotFlock.L
  · rw [if_pos hL] at h
    cases h
    refine ⟨hL, ?_⟩
    rw [Classical.epsilon_spec (hB b hL)]
    exact extract_append_self _ _
  · rw [if_neg hL] at h
    cases h

/-- Round 0 is the one round of the first group. -/
theorem groupOf_first {A : Live₂} {n : ℕ} (h : groupOf A n = .first) : n = 0 := by
  unfold groupOf at h
  split_ifs at h <;> first | assumption | cases h

/-- **The first stage's coins are the device's**: on a play the form accepts whose log is the device's, the reading's
round-0 coin agrees, on the first stage's bytes, with every coin the device's release for round 0 agrees with. -/
theorem coinsAgree_first (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop)
    (hShow : FirstShown C Good) (hDev : DevFirst C dev) (hRead : SourceRead C) (hB : CoinsOfBytes) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) Good .first := by
  intro o ha hg n _ hgn c hc c' hd
  obtain rfl := groupOf_first hgn
  simp only [formReading, ↓reduceIte] at hc hd ⊢
  obtain ⟨i, hi, hci⟩ := Option.bind_eq_some_iff.1 hc
  refine ⟨i, hi, ?_⟩
  obtain ⟨hL, hcs⟩ := coinOf_src hB hci
  obtain ⟨read, hread⟩ := firstRead_of_accepts ha hi
  have hsz := hRead o ha i read hi hread
  unfold prefixAgree
  by_cases h0 : i.source.size = 0
  · rw [h0]
    simp
  · have hex : i.source.extract 0 read = i.source := extract_of_size_le hsz
    obtain ⟨label, after, hdraw, hbits⟩ := hShow o ha hg i read hi hread (by rw [hex]; omega)
    rw [hex] at hbits
    obtain ⟨d, hdd, hds⟩ := hDev o ha label after i.source hdraw hbits (by omega) hL
    obtain ⟨i', hi', hpa⟩ := hd d hdd
    rw [hi] at hi'
    cases hi'
    unfold prefixAgree at hpa
    rw [hcs, ← hpa, hds]

end Verity.SampledTwo
