module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.Challenger.CanonRead
public import Proofs.Flock.SpotFlock.Jeq
public import Spot.Run
import all Lean.Data.Json.Basic

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
the chosen coin wherever the device's release does. The projection's `CoinsAgree` says that; `CoinsRead` implies it
(`coinsAgree_of_coinsRead`), and with the coin map's second law it gives `AcceptsRun`'s coin premise.

**What it rests on**, each a named `Prop` taken as a hypothesis:
* `FirstShown`: the play's log shows, after the first stage's digest in the first sample's family, the bytes the first
  stage read. It is the meeting of the run's coin check (`Run.coinsCode`) with the cut's `LogIsDevice`.
* `DevFirst`: the coin map's reading of round 0 starts with those bytes (the coin map's to prove).
* `CoinsOfBytes`: every byte string of at most one coin's bytes is a coin's bytes followed by zeros.

What the run and the form establish, at either value of the form's switch, is proved here: an accepted run's first
stage read `read` of its bytes under `Select.select`, its coins hold on them, and it answers that `read`
(`run_read`); the form refuses where the evidence's first-stage bytes run past it, so on an accepted play the first
stage's bytes are all read (`sourceRead_holds`).

**Not established here.** The run's coin check checks a signature over the coins without binding it to the current
monitoring period: a signature from an earlier period passes it. It is not worked around.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)

/-! ## The statement (`CoinsAgree`, the projection's) -/

/-- **Equal coins agree**: `CoinsRead` gives `CoinsAgree`. -/
theorem coinsAgree_of_coinsRead {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop} {g : Group}
    (h : CoinsRead A Rd Acc Good g) : CoinsAgree A Rd Acc Good g := by
  intro o ha hg i hr hgi c hc c' hd
  exact hd c ((h o ha hg i hr hgi).symm.trans hc)

/-! ## The first stage as the run runs it -/

/-- **What the first stage read**: as `Run.run` runs it, the work table the rule expands, the rule `Select.resolve`
gives, and `Select.select` on the first stage's bytes, which read `read` of them; and the coins hold on those bytes
(`Run.coinsCode` names no refusal, under the run's key, copy of main and family, at the boundary record and the
window root the verifier holds). -/
def FirstRead (i : Spot.Run.Input) (read : ℕ) : Prop :=
  ∃ (workTable : Option Json) (resolved : Spot.Select.Rule) (draw : Json) (rest : Array ℕ),
    i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
      .ok workTable ∧
    Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
    Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
      .ok (draw, rest, read) ∧
    Spot.Run.coinsCode i.key i.main i.coins i.record i.params.verifier.windowRoot i.family
      (i.source.extract 0 read) = none

/-- **The first stage's sample on a case**: the family of the boundary registration's challenger, which the run's coin
check reads (`inputOf`'s `family`), and the boundary registration's digest, after which it holds its coins. -/
def firstDraw (k : Case) : Option (String × ByteArray) :=
  match Spot.OneStage.commitmentOf k.commitment,
      Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson with
  | .ok c, .ok p => (p.boundary.challenger.map (·.2)).map fun l => (l, c.digest)
  | _, _ => none

/-! ## What the statement rests on -/

/-- **The play's log shows the bytes the first stage read**: on a play the form accepts whose log is the device's, the
bits of the bytes the run's first stage read, where it read any, start the coins the log shows released in the first
sample's family after its digest. -/
def FirstShown (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ i read, inputOn C o = some i → FirstRead i read →
    0 < (i.source.extract 0 read).size →
    ∃ label after, (C o).bind firstDraw = some (label, after) ∧
      Properties.Challenger.bitsOf (i.source.extract 0 read) <+:
        (Verity.Sampled.freshCoins label after o.log).flatten

/-- **The coin map reads round 0 from the first sample's releases**: on a play the form accepts, where bytes `b` of at most
one coin's length, whose bits start the coins the log shows released in the first sample's family after its digest, are
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
  rw [Json.mkObj, Properties.Challenger.CanonRead.getD_obj, Std.TreeMap.Raw.get?_eq_getElem?,
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
      (∀ code, Spot.Run.coinsCode i.key i.main i.coins i.record i.params.verifier.windowRoot i.family
        (i.source.extract 0 read) = some code → False) ∧
      ∃ (inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)) (acc₂ : Bool)
        (staged : List (String × Json)) (receipt symbol shown audit : Json),
        Spot.Run.secondStage i ((Spot.Draw.proved draw).map Int.toNat) = .ok (.ok inner) ∧
        Spot.Run.interiorAudit form i ((Spot.Draw.proved draw).map Int.toNat) inner = .ok (acc₂, staged) ∧
        j = Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", Lean.toJson read), ("symbol", symbol),
          ("public", shown), ("audit", audit)] ++ staged) := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | exact Or.inr ⟨_, _, _, _, _, by assumption, by assumption, by assumption, by assumption, by assumption,
        _, _, _, _, _, _, _, by assumption, by assumption, (Except.ok.inj h).symm⟩

/-- **The interior audit stages at most its `interior` entry.** -/
private theorem interiorAudit_staged {form : Bool} {i : Spot.Run.Input} {s : Array ℕ}
    {inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)} {acc : Bool} {staged : List (String × Json)}
    (h : Spot.Run.interiorAudit form i s inner = .ok (acc, staged)) :
    staged = [] ∨ ∃ x, staged = [("interior", x)] := by
  unfold Spot.Run.interiorAudit at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' (first | split at h | focus (cases h; done))
  all_goals first
    | (cases h; exact Or.inl rfl)
    | (cases h; exact Or.inr ⟨_, rfl⟩)

private theorem none_of_forall {o : Option String} (h : ∀ c, o = some c → False) : o = none := by
  cases o with
  | none => rfl
  | some c => exact (h c rfl).elim

private theorem nat?_toJson (n : ℕ) : Spot.nat? (Lean.toJson n) = some n := by
  simp [Spot.nat?, Spot.int?, Lean.toJson, Lean.JsonNumber.fromNat]

/-- **The run's answer at `read`**, where its staged entries are at most the interior's. -/
private theorem getD_answer_read {receipt draw symbol shown audit : Json} {read : ℕ} {staged : List (String × Json)}
    (h : staged = [] ∨ ∃ x, staged = [("interior", x)]) :
    Spot.nat? (Spot.getD (Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", Lean.toJson read),
      ("symbol", symbol), ("public", shown), ("audit", audit)] ++ staged)) "read") = some read := by
  rw [Json.mkObj, Properties.Challenger.CanonRead.getD_obj, Std.TreeMap.Raw.get?_eq_getElem?,
    Std.TreeMap.Raw.getElem?_ofList_of_mem (k := "read") (v := Lean.toJson read) Std.ReflCmp.compare_self ?_ ?_]
  · exact nat?_toJson read
  · rcases h with rfl | ⟨x, rfl⟩ <;> simp [List.pairwise_cons]
  · simp

/-- **An accepted run's first stage read its bytes with coins that hold, and it answers what it read**, at either
value of the form's switch. -/
theorem run_read {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j)
    (hs : Spot.jeq (Spot.getD j "symbol") (Json.str "accept") = true) :
    ∃ read, FirstRead i read ∧ Spot.nat? (Spot.getD j "read") = some read := by
  rcases run_path h with ⟨step, codes, rfl⟩ |
      ⟨wt, res, draw, rest, read, h1, -, h2, h3, h4, _, _, _, _, _, _, _, -, hia, rfl⟩
  · rw [symbol_refused] at hs
    cases SpotFlock.str_of_jeq (SpotFlock.jeq_symm hs)
  · exact ⟨read, ⟨wt, res, draw, rest, h1, h2, h3, none_of_forall h4⟩,
      getD_answer_read (interiorAudit_staged hia)⟩

/-- **An accepted run's first stage read its bytes with coins that hold**, at either value of the form's switch. -/
theorem run_firstRead {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j)
    (hs : Spot.jeq (Spot.getD j "symbol") (Json.str "accept") = true) : ∃ read, FirstRead i read :=
  (run_read h hs).imp fun _ h => h.1

/-- **The first stage reads one length**: `Select.select` is a function of the run's input. -/
theorem firstRead_unique {i : Spot.Run.Input} {r₁ r₂ : ℕ} (h₁ : FirstRead i r₁) (h₂ : FirstRead i r₂) : r₁ = r₂ := by
  obtain ⟨w, s, d, x, h1, h2, h3, -⟩ := h₁
  obtain ⟨w', s', d', x', h1', h2', h3', -⟩ := h₂
  rw [h1] at h1'
  cases h1'
  rw [h2] at h2'
  cases h2'
  rw [h3] at h3'
  cases h3'
  rfl

/-- **The run of record's input on an accepted case** is the one the check ran and accepted. -/
theorem inputOn_accepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input} (ha : FormAccepts C o)
    (hi : inputOn C o = some i) :
    ∃ k p c e c₂ ξ₂ r run, C o = some k ∧ Decoded k p c e ∧
      VStar.Recursive.SampledTwo.inputAt p c e.source e k.blobs = .ok (c₂, ξ₂, r, i) ∧
      Spot.Run.run i true = .ok run ∧ VStar.Recursive.SampledTwo.sourceRead e.source run = .ok () ∧
      Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true := by
  obtain ⟨k, hk, p, c, e, pub, prof, hd, hc⟩ := ha
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, hrun, hsr, -, hsym, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hc
  have hik : inputOfCase k = some i' := by
    obtain ⟨h1, h2, h3⟩ := hd
    unfold inputOfCase
    rw [h1, h2, h3]
    simp [bind, Except.bind, hin, pure, Except.pure]
  have : i = i' := by
    rw [inputOn, hk, Option.bind_some, hik] at hi
    exact (Option.some.inj hi).symm
  subst this
  exact ⟨k, p, c, e, c₂, ξ₂, r, run, hk, hd, hin, hrun, hsr, hsym⟩

/-- **An accepted play's run read its first stage's bytes with coins that hold.** -/
theorem firstRead_of_accepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input} (ha : FormAccepts C o)
    (hi : inputOn C o = some i) : ∃ read, FirstRead i read := by
  obtain ⟨_, _, _, _, _, _, _, run, _, _, _, hrun, -, hsym⟩ := inputOn_accepts ha hi
  exact run_firstRead hrun hsym

/-- The run of record's first-stage bytes are the evidence's. -/
theorem inputOf_source (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ ξ₂ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (received₂ : String) (vs : Array (String × Json)) :
    (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).source = ξ := by
  unfold VStar.Recursive.SampledTwo.inputOf
  rfl

/-- **The first stage read all its bytes**: on a play the form accepts, the run's first stage read every byte of its
input's, since the form refuses where the evidence's bytes run past what the run answers it read (`sourceRead`). -/
theorem sourceRead_holds (C : CaseOf) : SourceRead C := by
  intro o ha i read hi hread
  obtain ⟨_, _, _, e, _, _, _, run, _, _, hin, hrun, hsr, hsym⟩ := inputOn_accepts ha hi
  obtain ⟨read', hread', hr⟩ := run_read hrun hsym
  obtain rfl := firstRead_unique hread hread'
  obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, rfl, -, -, -⟩ :=
    VStar.Recursive.SampledTwo.inputAt_ok hin
  rw [inputOf_source]
  exact VStar.Recursive.SampledTwo.sourceRead_le hsr hr

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
    (hShow : FirstShown C Good) (hDev : DevFirst C dev) (hB : CoinsOfBytes) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) Good .first := by
  intro o ha hg n _ hgn c hc c' hd
  obtain rfl := groupOf_first hgn
  simp only [formReading, ↓reduceIte, eq_self_iff_true, if_true] at hc hd ⊢
  obtain ⟨i, hi, hci⟩ := Option.bind_eq_some_iff.1 hc
  refine ⟨i, hi, ?_⟩
  obtain ⟨hL, hcs⟩ := coinOf_src hB hci
  obtain ⟨read, hread⟩ := firstRead_of_accepts ha hi
  have hsz := sourceRead_holds C o ha i read hi hread
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
