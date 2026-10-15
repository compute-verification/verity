import Proofs.Flock.Soundness.Discharge.FailClosed.One
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Exec.Tables

/-!
# What an accepted statement's public file is tied to (`flock_verify_public_ties`)

Every statement `flock-verify verify` sets up passed the check of its public file (PROTOCOL.md §16.10):
`HmRow.checkPublic` (`Exec.setupH_public`), with hidden outputs `HmOut.checkPublic` (`Hidden.Spec.checkPublic`). The
check recomputes the frame-v3-sha512 trees natively; the circuit proves only the `hm96-sha512` leaves. `PublicTies`
restates what it established as equalities, and `flock_verify_public_ties` states them of every statement an acceptance
sets up, with or without `--zk` or `--registered F`, public or hidden outputs. No probability, no assumption.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec Refine

/-! ## The ties -/

/-- The schema of input port `port`'s leaves: `hm96-sha512/row-seg/v1` (segmented), `…/row/v2` (v2), else `…/row/v1`. -/
def rowSchema (port : Port) : String :=
  if port.segs.isSome then HmRow.ROW_SCHEMA_SEG else if port.v2 then HmRow.ROW_SCHEMA_V2 else HmRow.ROW_SCHEMA

/-- The population the roots are over: a drawn statement's registered one (`Public.population`), else the file's own
instance count, `b ‖ c` and output words. -/
def popOf (pub : Public) : ℕ × Array ByteArray × Array ℕ := pub.population.getD (pub.n, pub.digests, pub.outputs)

/-- How many of the population's `b ‖ c` are input port `p`'s rows, and where row `r` is: table `p` of a shared-row
file, else one per instance, instance-major. -/
def inRows (c : Circuit) (pub : Public) (p : ℕ) : ℕ × (ℕ → ℕ) :=
  match pub.tables with
  | some t => (t.counts[p]!, fun r => t.starts[p]! + r)
  | none => ((popOf pub).1, fun i => i * c.ports.size + p)

/-- **Input port `p`'s frame-v3-sha512 root** under domain `dom`: the tree over its rows' `hm96-sha512` leaves. -/
def inRoot (c : Circuit) (pub : Public) (p : ℕ) (dom : ByteArray) : ByteArray :=
  FV3.root dom ((Array.range (inRows c pub p).1).map fun r => FV3.leaf dom r (rowSchema c.ports[p]!)
    (Hm96.Default512.treeLeaf ((popOf pub).2.1.getD ((inRows c pub p).2 r) ByteArray.empty)) Sha512.hash) Sha512.hash

/-- Output port `j`'s first word among an instance's output words: the widths of the ports before it. -/
def wordAt (c : Circuit) (j : ℕ) : ℕ := ((c.outPorts.toList.take j).map (·.2)).sum

/-- **An output port's frame-v3-sha512 root** under domain `dom`: the tree over words `at_` to `at_ + w` of each
instance's output words, instance-major, each a u16 big-endian leaf. -/
def outRoot (c : Circuit) (pub : Public) (at_ w : ℕ) (dom : ByteArray) : ByteArray :=
  let words := (Array.range (popOf pub).1).foldl
    (fun acc i => acc ++ (Array.range w).map fun k => (popOf pub).2.2.getD (i * c.outWords + at_ + k) 0) #[]
  FV3.root dom ((Array.range words.size).map fun r =>
    FV3.leaf dom r "u16" (ByteArray.mk #[(words[r]! / 256).toUInt8, (words[r]! % 256).toUInt8]) Sha512.hash) Sha512.hash

/-- Input port `p`'s pinned dummy `b ‖ c`: with hidden outputs the padding instance's row under a zero salt, else the
zero row's. -/
def dummyBc (hidden : Bool) (c : Circuit) (p : ℕ) : ByteArray :=
  if hidden then HmOut.rowBcZeroSalt c.ports[p]! (HmOut.dummyRow c p) else HmRow.zeroRowBc c.ports[p]!

/-- **What the public file of a statement over circuit `c` is tied to**: the header's `frame_v3.frame` is
`frame-v3-sha512` and its `row_key_sha512` the hex SHA-512 of hm96's default key; with a unit draw, `instances` and the
statement's instance count are its proved units'; each input port's `frame_v3.schemas` entry is its leaves' schema, its
`frame_v3.roots` entry the frame-v3-sha512 root of its rows' `b ‖ c` under its `frame_v3.domain_ids` entry, and its
pinned dummy `b ‖ c` is `dummyBc`; with public outputs, each output port's root is that of its words. -/
structure PublicTies (hidden : Bool) (c : Circuit) (pub : Public) : Prop where
  frame : ∃ fv, pub.frameV3 = .ok fv ∧ J.str fv "frame" "public file" = .ok "frame-v3-sha512" ∧
    J.str fv "row_key_sha512" "public file" = .ok (hexOf (Sha512.hash Hm96.Default512.key))
  instances : ∀ d, pub.header.getObjVal? "unit_draw" = .ok d → ∃ u, Draw.UnitDraw.ofJson d = .ok u ∧
    J.nat pub.header "instances" "public file" = .ok u.proved.size ∧ pub.n = u.proved.size
  schemas : ∀ p < c.ports.size, ∃ fv s, pub.frameV3 = .ok fv ∧ J.get fv "schemas" "public file" = .ok s ∧
    J.str s c.ports[p]!.name "public file" = .ok (rowSchema c.ports[p]!)
  inputs : ∀ p < c.ports.size, ∃ dom, pub.fv3Bytes "domain_ids" c.ports[p]!.name = .ok dom ∧
    pub.fv3Bytes "roots" c.ports[p]!.name = .ok (inRoot c pub p dom)
  dummy : ∀ p < c.ports.size, c.dummyDigests.getD p ByteArray.empty = dummyBc hidden c p
  outputs : hidden = false → ∀ (j : ℕ) (hj : j < c.outPorts.size), ∃ dom,
    pub.fv3Bytes "domain_ids" c.outPorts[j].1 = .ok dom ∧
      pub.fv3Bytes "roots" c.outPorts[j].1 = .ok (outRoot c pub (wordAt c j) c.outPorts[j].2 dom)

/-! ## The checks establish them -/

theorem bytes_of_not_bne {a b : ByteArray} (h : ¬ (a != b) = true) : a = b := by
  have hb : (a == b) = true := by simpa [bne] using h
  exact ByteArray.ext (beq_iff_eq.1 (hb : (a.data == b.data) = true))

theorem ok_of_not_bne {α : Type} [BEq α] [LawfulBEq α] {x : V α} {a b : α} (hx : x = .ok a)
    (hc : ¬ (a != b) = true) : x = .ok b := by
  rw [hx, show a = b by simpa using hc]

theorem not_or_true {a b : Bool} (h : ¬ (a || b) = true) : ¬ a = true ∧ ¬ b = true := by
  cases a <;> cases b <;> simp_all

/-- A loop over a list whose every step passes `P a s` and continues at `s + w a`: element `j` passed at the start plus
the widths before it. -/
theorem forIn_sum_each {α : Type} (f : α → ℕ → V (ForInStep ℕ)) (w : α → ℕ) (P : α → ℕ → Prop)
    (hf : ∀ a s y, f a s = .ok y → P a s ∧ y = .yield (s + w a)) :
    ∀ (L : List α) (s v : ℕ), forIn L s f = .ok v →
      ∀ (j : ℕ) (hj : j < L.length), P L[j] (s + ((L.take j).map w).sum)
  | [], _, _, _ => fun j hj => absurd hj (by simp)
  | a :: L, s, v, h => by
      rw [List.forIn_cons] at h
      obtain ⟨y, hy, h⟩ := ebind_ok h
      obtain ⟨hP, rfl⟩ := hf a s y hy
      intro j hj
      cases j with
      | zero => simpa using hP
      | succ j =>
        have := forIn_sum_each f w P hf L (s + w a) v h j (by simpa using hj)
        simpa [Nat.add_assoc] using this

set_option maxHeartbeats 4000000 in
/-- **The check of the public file establishes the ties**: `HmRow.checkPublic` with public outputs, `HmOut.checkPublic`
with hidden ones. Both walk the same input loop; only `HmRow`'s has the output loop. -/
theorem ties_of_check {hidden : Bool} {c : Circuit} {pub : Public}
    (h : hidden = false ∧ HmRow.checkPublic c pub = .ok () ∨ hidden = true ∧ HmOut.checkPublic c pub = .ok ()) :
    PublicTies hidden c pub := by
  rcases h with ⟨hh, h⟩ | ⟨hh, h⟩
  on_goal 1 => unfold HmRow.checkPublic at h
  on_goal 2 => unfold HmOut.checkPublic at h
  all_goals
    generalize hp : pub.population.getD (pub.n, pub.digests, pub.outputs) = P at h
    obtain ⟨n, bcs, outputs⟩ := P
    have hpop : popOf pub = (n, bcs, outputs) := hp
    cases hu : pub.header.getObjVal? "unit_draw" <;> rw [hu] at h <;> dsimp only at h <;>
      simp only [throw_bind', pure_bind] at h <;> repeat' walk_step
  all_goals
    have hcf := not_or_true ‹¬ ((_ : String) != _ || _) = true›
    have hl1 := ‹forIn (m := V) (_ : Std.Legacy.Range) PUnit.unit _ = _›
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hl1
    simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one] at hl1
    have hin := range'_forIn_each c.ports.size _
      (fun p => ((∃ dom, pub.fv3Bytes "domain_ids" c.ports[p]!.name = .ok dom ∧
          pub.fv3Bytes "roots" c.ports[p]!.name = .ok (inRoot c pub p dom)) ∧
        (∃ fv s, pub.frameV3 = .ok fv ∧ J.get fv "schemas" "public file" = .ok s ∧
          J.str s c.ports[p]!.name "public file" = .ok (rowSchema c.ports[p]!))) ∧
        c.dummyDigests.getD p ByteArray.empty = dummyBc hidden c p)
      (fun p s y h => by
        try dsimp only at h
        cases ht : pub.tables <;> rw [ht] at h <;> dsimp only at h <;> (try simp only [throw_bind', pure_bind] at h) <;>
          repeat' walk_step
        all_goals
          simp only [pure, Except.pure, Except.ok.injEq] at h
          refine ⟨⟨⟨⟨_, ‹pub.fv3Bytes "domain_ids" _ = _›, ?_⟩,
            ⟨_, _, ‹pub.frameV3 = _›, ‹J.get _ "schemas" _ = _›, ?_⟩⟩, ?_⟩, _, h.symm⟩
          · have hr := bytes_of_not_bne ‹¬ (FV3.root _ _ _ != _) = true›
            rw [‹pub.fv3Bytes "roots" _ = _›, ← hr]
            simp only [inRoot, inRows, rowSchema, ht, hpop]
          · exact ok_of_not_bne ‹J.str _ c.ports[p]!.name "public file" = _› ‹¬ ((_ : String) != _) = true›
          · exact (bytes_of_not_bne ‹¬ (_ != c.dummyDigests.getD p ByteArray.empty) = true›).symm.trans
              (by simp [dummyBc, hh])) _ _ hl1
    exact {
      frame := ⟨_, ‹pub.frameV3 = _›, ok_of_not_bne ‹J.str _ "frame" _ = _› hcf.1,
        ok_of_not_bne ‹J.str _ "row_key_sha512" _ = _› hcf.2⟩
      instances := fun d hd => by
        first
        | (rw [hu] at hd; cases hd; done)
        | (obtain rfl := Except.ok.inj (hu.symm.trans hd)
           have hcu := not_or_true ‹¬ ((_ : ℕ) != _ || _) = true›
           exact ⟨_, ‹Draw.UnitDraw.ofJson _ = _›, ok_of_not_bne ‹J.nat _ "instances" _ = _› hcu.1,
             by simpa using hcu.2⟩)
      schemas := fun p hp => (hin p hp).1.2
      inputs := fun p hp => (hin p hp).1.1
      dummy := fun p hp => (hin p hp).2
      outputs := fun hf j hj => by
        first
        | exact absurd (hf.symm.trans hh) (by decide)
        | (have hl2 := ‹forIn (m := V) (_ : Array (String × ℕ)) (_ : ℕ) _ = _›
           rw [← Array.forIn_toList] at hl2
           have := forIn_sum_each _ (·.2)
             (fun x s => ∃ dom, pub.fv3Bytes "domain_ids" x.1 = .ok dom ∧
               pub.fv3Bytes "roots" x.1 = .ok (outRoot c pub s x.2 dom))
             (fun x s y h => by
               obtain ⟨oname, w⟩ := x
               try dsimp only at h
               try dsimp only
               try simp only [throw_bind', pure_bind] at h
               repeat' walk_step
               simp only [pure, Except.pure, Except.ok.injEq] at h
               refine ⟨⟨_, ‹pub.fv3Bytes "domain_ids" _ = _›, ?_⟩, h.symm⟩
               have hr := bytes_of_not_bne ‹¬ (FV3.root _ _ _ != _) = true›
               rw [‹pub.fv3Bytes "roots" _ = _›, ← hr]
               simp only [outRoot, hpop]) _ _ _ hl2 j (by simpa using hj)
           simp only [Nat.zero_add, Array.getElem_toList] at this
           exact this) }

/-! ## Every statement an acceptance sets up -/

theorem tagsAt_hiddenOutputs (tags : Tags) (count j : ℕ) :
    (tagsAt tags count j).hiddenOutputs = tags.hiddenOutputs := by
  unfold tagsAt; split_ifs <;> rfl

/-- Every statement `setupOf` sets up is tied to its public file. -/
theorem ties_of_setupOf {I : Inputs} {record : String} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf I record = .ok (spec, sts)) :
    0 < sts.size ∧ ∀ (j : ℕ) (hj : j < sts.size), PublicTies I.tags.hiddenOutputs sts[j].c sts[j].pub := by
  obtain ⟨hpos, hall⟩ := HiddenExec.setupTables_setupOne h
  refine ⟨hpos, fun j hj => ?_⟩
  obtain ⟨d, hd, -⟩ := hall j hj
  cases hh : I.tags.hiddenOutputs
  · rw [HiddenExec.setupOne_public ((tagsAt_hiddenOutputs _ _ _).trans hh)] at hd
    exact ties_of_check (.inl ⟨rfl, (setupH_public hd).1⟩)
  · rw [HiddenExec.setupOne_hidden ((tagsAt_hiddenOutputs _ _ _).trans hh)] at hd
    exact ties_of_check (.inr ⟨rfl, (Hidden.setupHidden_spec' hd).checkPublic⟩)

/-- **The public file's ties, over everything flock-verify accepts.** When `flock-verify verify` accepts (with or without
`--zk` or `--registered F`: the acceptances `flock_verify_sound` quantifies), the statement it set up (`setupOf`, at the
`--zk` inputs under `--zk`) has a table, and each table's public file is tied to it (`PublicTies`): its header's
`frame_v3` pins the frame and row key, and its schemas, domains and roots are those of the population's `b ‖ c` rows and,
with public outputs, output words; `instances` is the unit draw's proved count; the dummy rows are the pinned ones. The
verifier recomputes the trees natively, so this is deterministic. -/
theorem flock_verify_public_ties (zk : Bool) (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript)
    (h : FlockVerifyAccepts zk I t ∨ FlockVerifyAcceptsR zk I own t) :
    ∃ spec sts, setupOf (if zk then zkInputs I else I) t.record = .ok (spec, sts) ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), PublicTies I.tags.hiddenOutputs sts[j].c sts[j].pub := by
  have hh : (if zk then zkInputs I else I).tags.hiddenOutputs = I.tags.hiddenOutputs := by
    cases zk
    · rfl
    · exact zkTags_hiddenOutputs I
  obtain ⟨spec, sts, hs⟩ : ∃ spec sts, setupOf (if zk then zkInputs I else I) t.record = .ok (spec, sts) := by
    cases zk <;> rcases h with h | h
    · obtain ⟨⟨spec, sts, hs, -⟩, -⟩ := (h : Verifies I t); exact ⟨spec, sts, hs⟩
    · obtain ⟨⟨spec, sts, hs, -⟩, -⟩ := (h : Verifies I t); exact ⟨spec, sts, hs⟩
    · obtain ⟨⟨spec, sts, hs, -⟩, -⟩ := (h : VerifiesZK I t); exact ⟨spec, sts, hs⟩
    · obtain ⟨⟨spec, sts, hs, -⟩, -⟩ := (h : VerifiesZKR I own t); exact ⟨spec, sts, hs⟩
  obtain ⟨hpos, hall⟩ := ties_of_setupOf hs
  exact ⟨spec, sts, hs, hpos, fun j hj => hh ▸ hall j hj⟩

end FlockSoundness.Discharge.FailClosed
