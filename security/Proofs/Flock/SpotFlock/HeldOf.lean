module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
public import Proofs.Flock.SpotFlock.RevealsWired
public import VStarRecursive.SampledTwo
public import Proofs.Flock.SpotFlock.ScopeDesc

/-!
# The held copies, built from what the two-stage form's check holds

`SpotFlock.V h` is stated over held copies `h` (`Held`); the two-stage form's check holds the run of record's input
(`Spot.Run.Input`, with its interior `Spot.Run.Interior`). `heldOf i it u` builds `h` from them, and from what the
auditor registers that the run never reads (`Unread`: the statement's descriptor and query, each template's reference
session statement, the code the frame is built from, and the rounds past round 1 the second stage reads):

* `reg`: the boundary registration `Run.run` hands `Select.select` (the verifier's copies with the sampling design and work table
  the rule expands over the record's population, the resolved rule, the closure map; `firstOf`), the second stage's
  cells and counts as `Run.run` computes them (`cellsOf`), request `ρ`'s unit `p` of template `t` the members' block
  (`filesBlockOf`), and no proof block always checked;
* `vus`: each template's units, `n_t` a request over the boundary layout's requests (`unitsOf`);
* `k`, `names`, `refReads`: the counts, and the templates' names and reference reads in the params' order;
* `links`, `wires`, `rows`: the second stage's, from the outer block's Definition (`SecondStage.replayLinks` and the rest);
* `boundaryNames`, `interiorNames`: each registration's verifier's ports and registered values (`namesOf`);
* `xS₀`: template `t`'s reference statement (`refStmt`), each in Flock's masked scope (`zkOk`), decided;
* `frame`: SPOT's own frame at the wired program, from `Unread.c₀` (`withFrame`), as HONEST2's instance sets it.

`heldOf` is `none` unless the descriptor's checks pass (`descCore`: each member fits, the design is well formed on the
layout's strata, the second stage has a cell and a count of at least `1` for each template, each template's
reference statement reads as a member's, each member's circuit is in Flock's hidden-output scope and the registered
wiring passes its check), the templates' names are distinct, and the interior registration registers no value. A held copy it builds discharges
`NamesDistinct` (`namesDistinct_of`), with the run's population check RUNCOINS's `TwoHolds` (`HeldOfRun.twoHolds_of`, at an accepted run),
and with `RecordsHeld` its `RegistersNone` (`registersNone_of`). The last is
the second two-stage try's design, whose interior registration registers none; the first try's registers `logits`,
and there `heldOf` is `none`.

`heldMatches_of`: a held copy `heldOf` builds agrees with what the run reads (`HeldMatches`), given one clause that
the run's input does not decide, taken as a named hypothesis: `PartsAt`, `HeldMatches.parts`, the interior program's
strata making the held copy's parts at the replay sample the first stage selects on the run's bytes. `PartsAre`, the
same at every sample's size, gives it (`partsAt_of_partsAre`), but no held copy meets `PartsAre` where the interior
program has a stratum (`not_partsAre`: at `q = 0`, `SecondStage.partOf` refuses). `HeldMatches.refs` (`RefsAre`)
holds by construction (`refsAre_of`): the held copy's reference headers are the params'. So do the names the run's
records must hold to (`namesHeld_of`): the boundary and interior registrations' ports and registered values.

## What this does not show

* That `u.xs` are the members' reference statements as the form's blobs hold them: `heldOf` takes them as given.
* That `heldOf` is `some` at an honest run, or that the form refuses where it is `none`: the form does not call it yet.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs SpotFlock.RunsAs.Round2
open Lean (Json)

/-! ## What the run does not read -/

/-- **What the auditor registers that the run never reads**: the statement's descriptor (`desc`) and partition query
(`query`), each template's reference session statement in template order (`xs`), the code SPOT's frame is built from
(`c₀`, for the fields the frame doesn't set), and the rounds past round 1 whose coins the second stage reads
(`more`). -/
structure Unread where
  desc : Msg
  query : Query
  xs : List Msg
  c₀ : Code
  more : ℕ

/-! ## The fields -/

/-- **A registration's names**: its verifier's ports and registered values, in its order. -/
def namesOf (v : Spot.Registration.Verifier) : Names :=
  ⟨(v.ports.map (·.1)).toList, (v.registered.map (·.1)).toList⟩

/-- **The boundary registration the first stage samples on**, as `Run.run` hands it to `Select.select`: the verifier's
copies with the sampling design the rule expands over the boundary record's population and the work table it expands, the resolved
rule (`Select.resolve`) and the closure map; none where any of them is refused. -/
def firstOf (i : Spot.Run.Input) : Option Reg := do
  let v := i.params.verifier
  let n ← Spot.nat? (Spot.getD i.record "population")
  let rule ← i.rule.over n (Spot.Select.strataSizes v.program v.query)
  let law ← (rule.lawOver n).toOption
  let workTable ← (i.rule.tableOver v.program v.query v.workTable).toOption
  let resolved ← (Spot.Select.resolve i.rule v.program v.query).toOption
  pure ⟨{ v with law := law, workTable := workTable }, resolved, i.closure⟩

/-- **The second stage's cells and counts**, as `Run.run` computes them: each template's proof blocks in a replay
block (`Select.replayCells` of the replay block's program under the interior query) and the two-stage rule's count in
each (`Counts.over`); none under a one-stage rule or where either is refused. -/
def cellsOf (i : Spot.Run.Input) (it : Spot.Run.Interior) : Option (Array ℕ × Array ℕ) := do
  let ns ← (Spot.Select.replayCells it.replay it.params.verifier.query).toOption
  match i.rule with
  | .twoStage _ _ k => do
    let ids ← (Spot.Select.templateIds it.params.verifier.query).toOption
    let ks ← (k.over ids ns).toOption
    pure (ns, ks)
  | _ => none

/-- **The requests of a boundary layout**: its proof blocks that are a unit's (an input slot is none). -/
def requests (l : Spot.Slots.SlotOrder) : ℕ := (l.blocks.filter fun b => b.unit.isSome).size

/-- **Each template's units**: `n_t` a request over `r` requests, for `T` templates. -/
def unitsOf (r : ℕ) (ns : Array ℕ) (T : ℕ) : List (Array ℕ) :=
  (List.range T).map fun t => Array.range (r * ns.getD t 0)

/-- **Template `t`'s reference statement**: the `t`-th of `xs`, the first past them. -/
def refStmt (xs : List Msg) (t : ℕ) : Msg := xs.getD t (xs.headD [])

/-- **The members** at reference statements `xs` and units `vus`, as `Held.ms` lays them. -/
def msOf (xs : List Msg) (vus : List (Array ℕ)) : Spot.Bits.Members :=
  (List.range vus.length).map fun t => (stmtOf (refStmt xs t), vus.getD t #[])

/-- **Every reference statement is in Flock's masked scope**, decided over a non-empty list. -/
def xsOk (xs : List Msg) : Bool := !xs.isEmpty && xs.all zkOk

theorem zkOk_refStmt {xs : List Msg} (hx : xsOk xs = true) (t : ℕ) : zkOk (refStmt xs t) = true := by
  unfold xsOk at hx
  rw [Bool.and_eq_true, List.all_eq_true] at hx
  obtain ⟨hne, hall⟩ := hx
  apply hall
  unfold refStmt
  cases xs with
  | nil => simp at hne
  | cons a l =>
    rw [List.getD_eq_getElem?_getD]
    cases hg : (a :: l)[t]? with
    | none => exact List.mem_cons_self
    | some b => exact List.mem_of_getElem? hg

/-- **The held copies before their frame**, at the boundary registration `first`, the cells `ns`, the counts `ks` and
`r` requests, with `frame` the code `u.c₀`. -/
def held₀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (u : Unread) (first : Reg) (ns ks : Array ℕ) (r : ℕ)
    (hz : ∀ t, zkOk (refStmt u.xs t) = true) : Held where
  desc := u.desc
  vus := unitsOf r ns it.circuits.size
  query := u.query
  reg := { first, ns, ks, blockOf := filesBlockOf (msOf u.xs (unitsOf r ns it.circuits.size)) ns,
           always := fun _ => false }
  k := ks.toList
  names := (it.circuits.map (·.1)).toList
  links := Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record
  wires := Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn
  rows := Spot.SecondStage.replayRows i.params.verifier.program.prog it.replay.prog.programFn
  boundaryNames := namesOf i.params.verifier
  interiorNames := namesOf it.params.verifier
  xS₀ := refStmt u.xs
  xS₀_ok := hz
  refReads := fun t => (it.circuits.map (·.2.2.2)).toList.getD t .null
  refHeaders := fun t => (it.circuits.map (·.2.2.1)).toList.getD t .null
  frame := u.c₀
  more := u.more

/-- **SPOT's own frame at the wired program**, from the code `c₀`, as HONEST2's instance sets it. Nothing it reads is
`h.frame`. -/
noncomputable def withFrame (h : Held) (c₀ : Code) : Held :=
  { h with
    frame := frameAt (slotsOf (addressMapWired h (wireMap h) (returned h)))
      (revealCodeWired h (wireMap h) (returned h) (addressMapWired h (wireMap h) (returned h)) c₀) }

/-- **The descriptor's checks `heldOf` makes**: at the replay layout and the first stage's design, every member fits
(`memberOk`), the design is well formed on the layout's strata (`designOk`), the second stage has a count of at least
`1` and a cell for each member, every template's reference statement reads as a member's (`templateOk`), and `two`'s
`membersScoped` and `wiringOk`. It is `descOk` and `two`'s other clauses without the steps per unit (`filesOk`,
`fillOk`), which the run's population check gives (`HeldOfRun.filesOk_held₀`, `HeldOfRun.fillOk_held₀`), and without
`scopeOk`'s pass over every gate, which `memberOk` gives (`two_of_descOk`). -/
def descCore (h : Held) : Bool :=
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some d => h.ms.all memberOk && designOk d l && h.k.length == h.ms.length && h.k.all (1 ≤ ·) &&
      h.reg.ns.size == h.ms.length && (List.range h.ms.length).all (templateOk h) && membersScoped h && wiringOk h
  | _, _ => false

/-- **What `heldOf` decides of the copies it builds**: as many reference statements as templates, the templates'
names distinct, an interior registration that registers no value, and the descriptor's checks (`descCore`). -/
def decided (u : Unread) (it : Spot.Run.Interior) (h : Held) : Bool :=
  u.xs.length == it.circuits.size && decide ((List.range h.ms.length).map fun t => h.names.getD t "").Nodup &&
    h.interiorNames.values.isEmpty && descCore h

/-- `h` where `decided` holds of it. -/
def keep (u : Unread) (it : Spot.Run.Interior) (h : Held) : Option Held := if decided u it h then some h else none

/-- **The held copies of a run**, at the run of record's input `i`, its interior `it` and what the run does not read
`u` (module docstring); none where a field is refused or `decided` fails. -/
noncomputable def heldOf (i : Spot.Run.Input) (it : Spot.Run.Interior) (u : Unread) : Option Held :=
  (firstOf i).bind fun first => (cellsOf i it).bind fun c =>
    (Spot.Slots.layout first.program first.query).toOption.bind fun l =>
      if hz : xsOk u.xs = true then
        keep u it (withFrame (held₀ i it u first c.1 c.2 (requests l) (zkOk_refStmt hz)) u.c₀)
      else none

/-! ## What `heldOf` builds -/

theorem ok_of_toOption {ε α : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
  cases e with
  | error _ => cases h
  | ok b => cases h; rfl

/-- **`heldOf`, unfolded**. -/
theorem heldOf_spec {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) :
    ∃ first ns ks l hz, firstOf i = some first ∧ cellsOf i it = some (ns, ks) ∧
      Spot.Slots.layout first.program first.query = .ok l ∧
      h = withFrame (held₀ i it u first ns ks (requests l) hz) u.c₀ ∧ decided u it h = true := by
  unfold heldOf at hh
  simp only [Option.bind_eq_some_iff] at hh
  obtain ⟨first, hf, ⟨ns, ks⟩, hc, l, hl, hh⟩ := hh
  split at hh
  · rename_i hz
    unfold keep at hh
    split at hh
    · rename_i hd
      cases hh
      exact ⟨first, ns, ks, l, zkOk_refStmt hz, hf, hc, ok_of_toOption hl, rfl, hd⟩
    · cases hh
  · cases hh

theorem ms_length_held₀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (u : Unread) (first : Reg) (ns ks : Array ℕ)
    (r : ℕ) (hz : ∀ t, zkOk (refStmt u.xs t) = true) :
    (held₀ i it u first ns ks r hz).ms.length = it.circuits.size := by
  simp [Held.ms, held₀, unitsOf]

/-- The frame is not read by the members: `withFrame` sets only `frame`. -/
theorem ms_withFrame (h : Held) (c₀ : Code) : (withFrame h c₀).ms = h.ms := by
  unfold withFrame Held.ms
  dsimp only

/-- **The templates' names are distinct** (`NamesDistinct`) at a held copy `heldOf` builds. -/
theorem namesDistinct_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) : NamesDistinct h := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hd⟩ := heldOf_spec hh
  unfold decided at hd
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hd
  exact hd.1.1.2

theorem nil_of_mapM_nil {α β : Type} {f : α → Option β} (l : List α) (h : l.mapM f = some []) : l = [] := by
  cases l with
  | nil => rfl
  | cons a l =>
    simp only [List.mapM_cons, Option.bind_eq_bind] at h
    cases hf : f a with
    | none => simp [hf] at h
    | some b =>
      cases hm : l.mapM f with
      | none => simp [hf, hm] at h
      | some bs => simp [hf, hm] at h

/-- **The first stage's record registers no value** (`RegistersNone`) where it names the interior registration's names
(`RecordsHeld`) and `heldOf` built the held copy: `heldOf` decides that the interior registration registers none, and
a record naming exactly no registered value has no entry under `registered`. -/
theorem registersNone_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held} {x m₁ : Msg}
    {rec : Json} (hh : heldOf i it u = some h) (hrec : RecordsHeld h x m₁) (hm : jsonMsg m₁ = some rec) :
    RegistersNone rec := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hd⟩ := heldOf_spec hh
  unfold decided at hd
  simp only [Bool.and_eq_true] at hd
  have hv : h.interiorNames.values = [] := List.isEmpty_iff.1 hd.1.2
  unfold RecordsHeld recordsOk at hrec
  rw [hm] at hrec
  cases hx : decode x with
  | none => rw [hx] at hrec; simp at hrec
  | some st =>
    rw [hx] at hrec
    simp only [Bool.and_eq_true] at hrec
    have hr := hrec.2
    unfold recordOk at hr
    rw [hv, Bool.and_eq_true] at hr
    have hn := hr.2
    intro es hes e he v _
    unfold namesAt at hn
    rw [hes] at hn
    dsimp only at hn
    unfold namesEq at hn
    split at hn
    · rename_i ns hns
      simp only [Bool.and_eq_true, List.all_eq_true, List.contains_nil] at hn
      have hns0 : ns = [] := List.eq_nil_iff_forall_not_mem.2 fun n hn' => by
        have := hn.1.2 n hn'
        simp at this
      rw [hns0] at hns
      have hnil := nil_of_mapM_nil _ hns
      have he' : e ∈ es.toList := by simpa using he
      rw [hnil] at he'
      simp at he'
    · cases hn

/-! ## The held copies agree with what the run reads -/

/-- **The parts at every replay sample's size `q`**: the interior program's strata make the held copy's parts. -/
def PartsAre (it : Spot.Run.Interior) (h : Held) : Prop :=
  ∀ strata q, Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata →
    Spot.SecondStage.parts q strata = some (partsAt h q)

/-- **`HeldMatches.parts`, the parts at the run's replay sample**: at the replay sample `rest` the first stage selects
on the run's bytes, the interior program's strata make the held copy's parts. -/
def PartsAt (i : Spot.Run.Input) (it : Spot.Run.Interior) (h : Held) : Prop :=
  ∀ draw rest read strata, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query
      h.reg.first.table h.reg.first.closure i.source = .ok (draw, rest, read) →
    Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata →
    Spot.SecondStage.parts rest.size strata = some (partsAt h rest.size)

/-- `PartsAre`, at every replay sample, gives `PartsAt`, at the run's. -/
theorem partsAt_of_partsAre {i : Spot.Run.Input} {it : Spot.Run.Interior} {h : Held} (hP : PartsAre it h) :
    PartsAt i it h :=
  fun _ rest _ strata _ hs => hP strata rest.size hs

/-- **The parts fit the cells**, decided at a replay sample of `q` blocks: the strata make the held copy's parts. One
comparison per template. -/
def partsFit (h : Held) (q : ℕ) (strata : Array (String × Array (ℕ × ℕ))) : Bool :=
  decide (Spot.SecondStage.parts q strata = some (partsAt h q))

theorem parts_of_fit {h : Held} {q : ℕ} {strata : Array (String × Array (ℕ × ℕ))} (hf : partsFit h q strata = true) :
    Spot.SecondStage.parts q strata = some (partsAt h q) :=
  of_decide_eq_true hf

theorem partOf_zero (s : String × Array (ℕ × ℕ)) : Spot.SecondStage.partOf 0 s = none := by
  unfold Spot.SecondStage.partOf
  split
  · simp
  · rfl

/-- **No held copy meets `PartsAre` where the interior program has a stratum**: at a sample of no replay block every
stratum is refused (`partOf_zero`), so the strata make no parts. -/
theorem not_partsAre {it : Spot.Run.Interior} {h : Held} {strata : Array (String × Array (ℕ × ℕ))}
    (hs : Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata)
    (hne : strata ≠ #[]) : ¬ PartsAre it h := by
  intro hP
  have h0 := hP strata 0 hs
  unfold Spot.SecondStage.parts at h0
  rw [Array.mapM_eq_mapM_toList] at h0
  cases hl : strata.toList with
  | nil => exact hne (Array.toList_eq_nil_iff.1 hl)
  | cons a l =>
    rw [hl] at h0
    simp [List.mapM_cons, partOf_zero] at h0

/-- **`HeldMatches.refs`**: each template's reference header (`Held.refHeaders`) is the params'
(`Run.Interior.circuits`); `refsAre_of` proves it at every held copy `heldOf` builds. -/
def RefsAre (it : Spot.Run.Interior) (h : Held) : Prop :=
  refsOf h = it.circuits.map fun (t, _, hd, _) => (t, hd)

theorem refReads_eq (cs : Array (String × String × Json × Json)) :
    ((List.range cs.size).map fun t => ((cs.map (·.1)).toList.getD t "", (cs.map (·.2.2.2)).toList.getD t .null)).toArray =
      cs.map fun (t, _, _, r) => (t, r) := by
  apply Array.ext
  · simp
  · intro j hj₁ hj₂
    have hj : j < cs.size := by simpa using hj₁
    simp [List.getD_eq_getElem?_getD, hj]

theorem refHeaders_eq (cs : Array (String × String × Json × Json)) :
    ((List.range cs.size).map fun t => ((cs.map (·.1)).toList.getD t "", (cs.map (·.2.2.1)).toList.getD t .null)).toArray =
      cs.map fun (t, _, hd, _) => (t, hd) := by
  apply Array.ext
  · simp
  · intro j hj₁ hj₂
    have hj : j < cs.size := by simpa using hj₁
    simp [List.getD_eq_getElem?_getD, hj]

/-- **`RefsAre` holds at a held copy `heldOf` builds**: its reference headers are the params' (`Held.refHeaders`). -/
theorem refsAre_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) : RefsAre it h := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  unfold RefsAre refsOf
  rw [ms_withFrame, ms_length_held₀]
  exact refHeaders_eq it.circuits

/-- **The held names are the two registrations'**, at a held copy `heldOf` builds: the boundary registration's ports
and registered values (`namesOf` at the params' verifier) and the interior registration's (at the interior's), the
names the run's records must hold to (`recordsOk`). -/
theorem namesHeld_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) :
    h.boundaryNames = namesOf i.params.verifier ∧ h.interiorNames = namesOf it.params.verifier := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  unfold withFrame
  exact ⟨rfl, rfl⟩

/-- **A held copy `heldOf` builds agrees with what the run reads** (`HeldMatches`), given the clause `PartsAt`, which
the run's input does not decide. Every other clause holds by `heldOf`'s construction. -/
theorem heldMatches_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hP : PartsAt i it h) : HeldMatches i it h := by
  have hR := refsAre_of hh
  obtain ⟨first, ns, ks, l, hz, hf, hc, -, rfl, -⟩ := heldOf_spec hh
  have hrr : refRegisteredOf (withFrame (held₀ i it u first ns ks (requests l) hz) u.c₀) =
      it.circuits.map fun (t, _, _, r) => (t, r) := by
    unfold refRegisteredOf
    rw [ms_withFrame, ms_length_held₀]
    exact refReads_eq it.circuits
  unfold firstOf at hf
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hf
  obtain ⟨n, hn, rule, hrule, law, hlaw, workTable, hwt, resolved, hres, rfl⟩ := hf
  unfold cellsOf at hc
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hc
  obtain ⟨ns', hns, hc⟩ := hc
  split at hc
  · rename_i grain replay k hk
    simp only [Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq,
      Prod.mk.injEq] at hc
    obtain ⟨ids, hids, ks', hks, rfl, rfl⟩ := hc
    exact
      { first := ⟨n, rule, law, workTable, resolved, hn, hrule, ok_of_toOption hlaw, ok_of_toOption hwt,
          ok_of_toOption hres, rfl⟩
        ns := ok_of_toOption hns
        ks := ⟨grain, replay, k, hk, ids, ok_of_toOption hids, ok_of_toOption hks⟩
        links := rfl
        wires := rfl
        rows := rfl
        parts := hP
        refs := hR
        refReads := hrr }
  · cases hc

/-! ## At the form's run of record -/

/-- **The held copies of the form's run of record** (`SampledTwo.inputOf`): `heldOf` at its input and interior. -/
noncomputable def heldOfForm (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ ξ₂ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (received₂ : String) (vs : Array (String × Json)) (u : Unread) :
    Option Held :=
  match (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior with
  | some it => heldOf (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs) it u
  | none => none

/-- **At the form's run of record**, a held copy `heldOfForm` builds agrees with what the run reads, given `PartsAt`
at its interior. -/
theorem heldMatches_form {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Json)} {u : Unread}
    {h : Held} (hh : heldOfForm p c ξ ξ₂ e received₂ vs u = some h) :
    ∃ it, (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it ∧
      heldOf (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs) it u = some h ∧
      (PartsAt (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs) it h →
        HeldMatches (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs) it h) := by
  unfold heldOfForm at hh
  split at hh
  · rename_i it hit
    exact ⟨it, hit, hh, fun hP => heldMatches_of hh hP⟩
  · cases hh

end SpotFlock.HeldOf
