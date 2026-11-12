module

public import Proofs.Flock.SpotFlock.EntriesAt
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The published trees a session reads are at most 64 levels deep

`rootedDepthsAt`: in an accepted run, the published tree of each port a checked unit's one-table session reads that no
registered read is named like has at most `2^64` leaves, so it is no deeper than a code at least 64 levels deep
(`RootedDepthsAt`). With it, `frameIs_of` gives `FrameIs` from the reveal's slots alone.

Such a port's tree has a leaf per instance (`PortRoots.countOf`). A loaded file has tables only where its header shares
rows (`loadPublic_counts`), which names every port (`sharedRows_names`), and round 2 lets a header share rows only at
ports its statement reads registered (`SecondStage.refsOk`), whose names are the verifier's registered reads'
(`portsOf_entries`). The instance count is the header's `instances` (`loadPublic_counts`), which normalizing keeps
(`raw_nat`) and round 2 holds to at most `MAX_LEAVES` (`SecondStage.populationOk`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace Depths

/-- `Json.getNat?` reads only a natural number's own encoding. -/
theorem eq_of_getNat {v : Json} {n : ℕ} (h : v.getNat? = .ok n) : v = .num ⟨Int.ofNat n, 0⟩ := by
  unfold Json.getNat? at h
  split at h
  · cases h; rfl
  · cases h

/-- A value equivalent to a number is that number. -/
theorem num_of_jsonEquiv {x : Lean.JsonNumber} {w : Json} (h : FlockDeploy.JsonEquiv (.num x) w) : w = .num x := by
  cases w <;> simp [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at h
  subst h
  rfl

/-- **A natural number the normalized header holds at `k` is the raw header's there.** -/
theorem raw_nat {hdr : Json} (hw : Claims.Json.Wf hdr) {k ctx : String} {n : ℕ}
    (h : _root_.Flock.J.nat (_root_.Flock.norm hdr) k ctx = .ok n) : Spot.nat? (Spot.getD hdr k) = some n := by
  unfold _root_.Flock.J.nat at h
  obtain ⟨v, hv, hn⟩ := FlockSoundness.Refine.ebind_ok h
  have hg : (_root_.Flock.norm hdr).getObjVal? k = .ok v := by
    unfold _root_.Flock.J.get at hv
    split at hv
    · cases hv; assumption
    · cases hv
  have hvn : v.getNat? = .ok n := by
    unfold _root_.Flock.J.asNat at hn
    split at hn
    · cases hn; assumption
    · cases hn
  obtain ⟨u, hu, he, -⟩ :=
    HeaderRoots.getObjVal_equiv (FlockDeploy.wf_norm hdr) hw (FlockDeploy.jsonEquiv_norm hdr hw) hg
  rw [eq_of_getNat hvn] at he
  rw [HeaderRoots.getD_of_getObjVal hu, num_of_jsonEquiv he]
  simp [Spot.nat?, Spot.int?]

/-- **A key under field `o` of the normalized header is there in the raw header**, as an entry of `o`. -/
theorem raw_mem {hdr : Json} (hw : Claims.Json.Wf hdr) {o k : String} {t v : Json}
    (ht : (_root_.Flock.norm hdr).getObjVal? o = .ok t) (hv : t.getObjVal? k = .ok v) :
    ∃ v', (k, v') ∈ Spot.entries (Spot.getD hdr o) := by
  have hn := FlockDeploy.wf_norm hdr
  obtain ⟨t', ht', he, hw'⟩ := HeaderRoots.getObjVal_equiv hn hw (FlockDeploy.jsonEquiv_norm hdr hw) ht
  obtain ⟨u, hu, -, -⟩ := HeaderRoots.getObjVal_equiv (HeaderRoots.wf_of_getObjVal hn ht) hw' he hv
  rw [HeaderRoots.getD_of_getObjVal ht']
  obtain ⟨m, rfl⟩ := FlockComplete.JsonRT.obj_of_getObjVal hu
  cases hw' with
  | obj hm _ =>
    refine ⟨u, ?_⟩
    show (k, u) ∈ m.toArray
    rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray,
      Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm, ← Std.TreeMap.Raw.get?_eq_getElem?]
    exact (FlockComplete.JsonRT.mem_of_getObjVal hm hu).2

/-- An object no element of `ps` is missing from has each one's key. -/
theorem ok_of_not_any {α : Type} {ps : Array α} {nm : α → String} {s : Json} {p : α} (hp : p ∈ ps)
    (h : ¬(ps.any fun p => !(s.getObjVal? (nm p)).isOk) = true) : ∃ v, s.getObjVal? (nm p) = .ok v := by
  cases hg : s.getObjVal? (nm p) with
  | ok v => exact ⟨v, rfl⟩
  | error e => exact absurd (Array.any_eq_true'.2 ⟨p, hp, by rw [hg]; rfl⟩) h

/-- **A header that shares rows names every port under `shared_rows`.** -/
theorem sharedRows_names {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {hd : Json} {cs : Array ℕ}
    (h : _root_.Flock.HmRow.sharedRows tags c hd = .ok (some cs)) :
    ∃ s, hd.getObjVal? "shared_rows" = .ok s ∧ ∀ p ∈ c.ports, ∃ v, s.getObjVal? p.name = .ok v := by
  unfold _root_.Flock.HmRow.sharedRows at h
  try dsimp only at h
  try simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at h
  all_goals
    refine ⟨_, ‹_›, fun p hp => ok_of_not_any hp ?_⟩
    have hc := ‹¬((_ || _) = true)›
    simp only [Bool.or_eq_true, not_or] at hc
    exact hc.2

-- `walk_step` down `HmRow.loadPublic`'s unfolded do-block, a goal per branch
set_option maxHeartbeats 2000000 in
/-- **What `HmRow.loadPublic` reads of a population's size and tables**: its instance count is the normalized header's
`instances`, and a file with tables has a header that shares rows. -/
theorem loadPublic_counts {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {f : ByteArray}
    {pop : _root_.Flock.Public} (h : _root_.Flock.HmRow.loadPublic tags c f = .ok pop) :
    ∃ j, headerOf f = some j ∧ _root_.Flock.J.nat (_root_.Flock.norm j) "instances" "public file" = .ok pop.n ∧
      ∀ t, pop.tables = some t → ∃ cs, _root_.Flock.HmRow.sharedRows tags c (_root_.Flock.norm j) = .ok (some cs) := by
  unfold _root_.Flock.HmRow.loadPublic at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [pure, Except.pure, Except.ok.injEq] at *
    subst_vars
  all_goals
    refine ⟨_, ?_, ‹_›, ?_⟩
    · simp [headerOf, ‹_root_.Flock.findByte f 10 = some _›, ‹String.fromUTF8? _ = some _›,
        ‹_root_.Flock.JsonParse.parse _ = .ok _›, Except.toOption]
    · intro t ht
      first
        | exact ⟨_, ‹_›⟩
        | cases ht

/-- Round 2's population check holds the header's `instances` to at most `MAX_LEAVES`. -/
theorem populationOk_le {q : ℕ} {p : Spot.SecondStage.Part} {hd : Json}
    (hp : Spot.SecondStage.populationOk q p hd = true) :
    ∃ m, Spot.nat? (Spot.getD hd "instances") = some m ∧ m ≤ Spot.SecondStage.MAX_LEAVES := by
  unfold Spot.SecondStage.populationOk at hp
  split at hp
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
    exact ⟨_, ‹_›, hp.1.2⟩
  · cases hp

theorem someMap_ok {α : Type} {x : Except String α} {r : Option α} (h : (some <$> x) = Except.ok r) :
    ∃ a, x = .ok a ∧ r = some a := by
  cases x with
  | error e => cases h
  | ok b => cases h; exact ⟨b, rfl, rfl⟩

/-- **A statement's `registered reads` file gives the verifier its registered reads.** -/
theorem ins_of_registered {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (hw : w.ins = .ok s) {b}
    (hb : w.registered = some b) :
    ∃ j own, FlockDeploy.jsonOf "registered reads" b = .ok j ∧ _root_.Flock.Registered.ofJson j = .ok own ∧
      s.registered = some own := by
  have hm := (ins_reads_public hw).1
  rw [hb, Option.mapM_some] at hm
  obtain ⟨own, hx, hr⟩ := someMap_ok hm
  obtain ⟨j, hj, hoj⟩ := FlockSoundness.Refine.ebind_ok hx
  exact ⟨j, own, hj, hoj, hr⟩

end Depths

variable (h : Held)

/-- **A port of a checked unit's one-table session that no registered read is named like has at most `MAX_LEAVES`
rows**: one per instance, since round 2 lets the header share rows only at ports its statement reads registered, and
round 2 holds the instance count to `MAX_LEAVES`. -/
theorem countOf_le {x m₁ m₂ : Msg} {s} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u : ℕ}
    {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hset : FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts))
    {j : ℕ} (hj : j < sts.size) {q : ℕ} {pop : _root_.Flock.Public} (hq : q < sts[j].c.ports.size)
    (hnr : ∀ own, ins.registered = some own → ∀ reg ∈ own, reg.name ≠ sts[j].c.ports[q]!.name)
    (hpop : _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop) :
    _root_.Flock.PortRoots.countOf pop q ≤ Spot.SecondStage.MAX_LEAVES := by
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨_, _, _, w, S, -, -, -, hw, hS, hc⟩ := readsOk_true h hat.2
  have hwi := ins_of_setupOf hw hset
  obtain ⟨-, hSH, hSr, hSn⟩ := stmtJsonOf_fields h hS
  have hH : headerOf ins.read.I.publicFile = some S.header := by
    rw [(ins_reads_public hwi).2]
    exact hSH
  obtain ⟨-, -, -, hpo, -, -, -, -, -, hrefs, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  obtain ⟨hdr, hhdr, hn, htab⟩ :=
    Depths.loadPublic_counts (FlockSoundness.Discharge.Hidden.loadPublic_nil hpop)
  rw [hH, Option.some.injEq] at hhdr
  subst hhdr
  have hwf := HeaderRoots.wf_of_headerOf hH
  have hnone : pop.tables = none := by
    cases ht : pop.tables with
    | none => rfl
    | some t =>
      exfalso
      obtain ⟨cs, hcs⟩ := htab t ht
      obtain ⟨sr, hsr, hall⟩ := Depths.sharedRows_names hcs
      have hmq : sts[j].c.ports[q]! ∈ sts[j].c.ports := by
        rw [getElem!_pos sts[j].c.ports q hq]
        exact Array.getElem_mem hq
      obtain ⟨v, hv⟩ := hall _ hmq
      obtain ⟨v', hmem⟩ := Depths.raw_mem hwf hsr hv
      simp only [Spot.SecondStage.refsOk] at hrefs
      obtain ⟨⟨nm, e⟩, hme, hnm⟩ := Array.any_eq_true'.1 (Array.all_eq_true_iff_forall_mem.1 hrefs _ hmem)
      have hnm' : nm = sts[j].c.ports[q]!.name := by simpa using hnm
      subst hnm'
      cases hreg : w.registered with
      | none =>
        rw [hSn hreg, show Spot.entries (Spot.getD Json.null "ports") = #[] from rfl] at hme
        exact absurd hme (Array.not_mem_empty _)
      | some b =>
        obtain ⟨jR, own, hjR, hoj, hown⟩ := Depths.ins_of_registered hwi hreg
        have hSj : S.registered = jR := by
          have h4 := hSr b hreg
          rw [hjR] at h4
          exact (Option.some.inj h4).symm
        obtain ⟨kvs, hports, -, hback⟩ := portsOf_entries (portsOf_of_ofJson hoj)
        rw [hSj, getD_of_ok hports] at hme
        obtain ⟨reg, hreg', hrn⟩ := hback _ hme
        exact hnr own hown reg hreg' hrn
  obtain ⟨m, hm, hle⟩ := Depths.populationOk_le hpo
  rw [Depths.raw_nat hwf hn, Option.some.injEq] at hm
  simp only [_root_.Flock.PortRoots.countOf, hnone, hm]
  exact hle

/-- **The published trees a checked unit's one-table session reads are at most 64 levels deep** (`countOf_le`), so
no deeper than a code at least 64 levels deep. -/
theorem rootedDepthsAt {c : Code} (h64 : 64 ≤ depth c.own) : RootedDepthsAt h c := by
  intro x s m₁ m₂ c₀ c₁ _ u ins spec sts hx hok hu _ hset j hj q pop _ hq hnr hpop
  exact (log2ceil_le_64 (countOf_le h hx hok hu hset hj hq hnr hpop)).trans h64

/-- **`FrameIs` at a frame that reads its ports off the roots**, at a code at least 64 levels deep, from the reveal's
slots (`SlotsAt`). -/
theorem frameIs_of {slot : ℕ → Option Slot} {c : Code} {L : Layout} (hf : h.frame = frameAt slot c)
    (hS : SlotsAt h slot c L) (h64 : 64 ≤ depth c.own) : FrameIs h L :=
  frameIs_frameAt h hf hS (entriesAt_of h (depthsAt_of h h64 (rootedDepthsAt h h64)))

end SpotFlock
