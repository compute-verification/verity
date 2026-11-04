import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.Discharge
import Proofs.Flock.Claims.Complete.Replay.RingSwitchAlgebra

/-!
# The extra claims' ring-switch algebra at the deployed relation (`RsExtrasZk`)

`rsExtrasZk`: at `Readable.relation`, from `RelS` alone, every extra claim a masked rep opens is met by its switched
vector of the honest trace, as at the plain record (`claims_switch`): the masked case's one table is the statement the
soundness proof plays, satisfied by the witness's bits (`CaseZk.model`, as `Case.model`), and the record's link points
have `m_pts` coordinates (`RunZk.sess_points`).
-/

namespace FlockComplete.Replay

open Flock FlockDeploy FlockSoundness.Refine

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The statement the masked verifier parsed** at a good case: its tables are the case's, and the witness satisfies
it. -/
theorem _root_.FlockComplete.CaseZk.parsed (c : CaseZk H rel) :
    ∃ p : Readable.Parsed, Readable.parseBytes (readOf true) c.x = some p ∧ p.sts = c.sts ∧
      Claims.Flock.RelS (Readable.toReadable rel p) c.w := by
  obtain ⟨s, hs, hR⟩ := c.rel_w
  change (Readable.parseBytes (readOf true) c.x).map (Readable.toReadable rel) = some s at hs
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  refine ⟨p, hp, ?_, hR⟩
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  have htab := Readable.tablesOf_of_parses (Readable.parses_of_parse hp)
  obtain ⟨spec', hset'⟩ := Readable.setup_of_tablesOf htab
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [ht] at hset'
  have := hset'.symm.trans hset
  simp only [Except.ok.injEq, Prod.mk.injEq] at this
  exact this.2

/-- **The masked case's statement, satisfied**: at a good masked case of the deployed relation, the one table `st` is
well formed, its table's matrices are `stmtOf`'s, and the witness's bits satisfy `stmtOf st hwf (regionsOf st)`. -/
theorem _root_.FlockComplete.CaseZk.model (c : CaseZk H Readable.relation) :
    ∃ st, c.sts = #[st] ∧ ∃ hwf : StmtWF st, RegionsWF st ∧
      (∀ a b : Fin (2 ^ st.c.kLog), FlockLevel3.placedA st a b = matA st a.val b.val ∧
        FlockLevel3.placedB st a b = matB st a.val b.val) ∧
      (stmtOf st hwf (regionsOf st)).Satisfies (Readable.zOf 1 c.w 0 st.m) := by
  obtain ⟨st, hsts, -, -, -⟩ := c.table
  obtain ⟨p, hp, hpe, hR⟩ := c.parsed
  obtain ⟨p', hp', hsc⟩ := (Readable.scope_iff (deployZk Readable.relation) Readable.relation (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  rw [hp] at hp'
  obtain rfl : p = p' := Option.some.inj hp'
  have hP : Readable.Parses p := by
    unfold Readable.parseBytes at hp
    obtain ⟨r, -, hrp⟩ := Option.bind_eq_some_iff.1 hp
    exact Readable.parses_of_parse hrp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hRu := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  rw [hsts] at hpe
  have h0 : 0 < p.sts.size := by simp [hpe]
  obtain ⟨hf, hsat⟩ := (Readable.relS_relation p c.w).1 hR 0 h0
  have e0 : p.sts[0] = st := by simp [hpe]
  have e0' : p.sts[0]! = st := by simp [hpe]
  have e1 : p.sts.size = 1 := by simp [hpe]
  have key := Readable.tableAt_model (Readable.tablesOf_of_parses hP) hRu.2.1 hpi h0
  revert key hsat hf
  rw [e0, e0', e1]
  rintro hf hsat ⟨hwf, hrw, hmat⟩
  refine ⟨st, hsts, hwf, hrw, hmat, ?_⟩
  obtain ⟨h1, h2, h3⟩ := hsat
  have hA : ((Readable.Tab.ofStmt st).toModel hf.2.2).A₀ = (stmtOf st hwf (regionsOf st)).A₀ :=
    funext fun a => funext fun b => (hmat a b).1
  have hB : ((Readable.Tab.ofStmt st).toModel hf.2.2).B₀ = (stmtOf st hwf (regionsOf st)).B₀ :=
    funext fun a => funext fun b => (hmat a b).2
  refine ⟨fun o i => ?_, h2, h3⟩
  have := h1 o i
  rw [hA, hB] at this
  exact this

/-- The masked record's link points have `m_pts` coordinates: `Record.decodeWith` checks it (S9). -/
theorem _root_.FlockComplete.RunZk.sess_points (o : RunZk H rel) {st : Stmt} (hspec : o.spec = st.spec) :
    ∀ p ∈ o.sess.points.toList, p.size = st.mPts := by
  unfold RunZk.sess
  cases h : Record.decodeWith o.hs o.spec o.record with
  | error e =>
    intro p hp
    rw [show (val (Except.error e : Except String RunRecord)).points = #[] from rfl] at hp
    simp at hp
  | ok s =>
    have hp := decodeWith_points o.hs h
    rw [hspec] at hp
    exact hp

end FlockComplete.Replay

namespace FlockComplete

open Flock FlockDeploy FlockSoundness.Refine FlockComplete.Replay

local notation "𝔽" => FlockLevel3.GF128

/-- **The extra claims' ring-switch algebra at the deployed relation**, from `RelS` alone. -/
theorem rsExtrasZk (H : Claims.Msg → Claims.Msg) : RsExtrasZk H Readable.relation := by
  intro o i hi k hk
  obtain ⟨st, hsts, hwf, hrw, -, hsat⟩ := o.model
  obtain ⟨st', hsts', hspec, h25, -⟩ := o.table
  obtain rfl : st = st' := by
    rw [hsts] at hsts'
    simpa using hsts'
  have hsz : o.tabs.size = 1 := by simp [RunZk.tabs, FlockDeploy.tabsOf, hsts]
  obtain rfl : i = 0 := by omega
  have ht : o.tabs[0]! = Setup.ofCircuit st := by simp [RunZk.tabs, FlockDeploy.tabsOf, hsts]
  have htr : (tracesOf o.sts o.w)[0]! = traceOf st o.w.hid.toArray 0 := by simp [tracesOf, hsts]
  have hzb : ∀ a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m),
      Zc.bitF (tracesOf o.sts o.w)[0]!.z a.val = ((Readable.zOf 1 o.w 0 st.m a).val : 𝔽) := by
    intro a
    rw [htr]
    unfold Zc.bitF
    rw [traceOf_z st o.w.hid a.isLt, Lc.bitF_toZ]
    show _ = ((Readable.toZ (o.w.hid.getD (a.val * 1 + 0) false)).val : 𝔽)
    rw [Nat.mul_one, Nat.add_zero]
  have hex : o.extras 0 = val (st.extraClaims o.sess.points) := by
    simp only [RunZk.extras, ht, Setup.ofCircuit]
  rw [hex] at hk ⊢
  have key := claims_switch st hwf hrw (by omega) _ hsat (tracesOf o.sts o.w)[0]! hzb (Zc.coinsAt o.resp 0)
    (Lc.coinsAt o.resp 0 0) o.sess.points (o.sess_points hspec)
  rw [getElem!_pos _ k hk]
  exact key _ (Array.mem_append.2 (Or.inr (Array.getElem_mem hk)))

end FlockComplete
