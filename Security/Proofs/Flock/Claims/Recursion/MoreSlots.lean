import Proofs.Flock.Claims.Recursion.ResLowerDeployed

/-!
# The extra calls' slots at any staged part (`MoreSlotsTrue`'s parts)

`DeployedR.layR` records, beside `layoutBTops`' calls, each output row's (`outsCalls`), check 5's (`valCalls`) and the
`rec-acc` chain's (`accCalls`) hm96 calls. Each pair is a slot's salt call and its row call, at some row forms
(`formCalls`), and both are true wherever the slot outputs the commit string of its row prefix and that row
(`formCalls_true`, `hmCalls_true` at any forms). An input row's slot does at a staged part (`slotCommits_table`);
an output row's does wherever its unit's output group lies in the unit's slot (`slotOut_table`'s `hin`), which holds of
every unit the parser built (`hin_of_check`: D11 in `Net.ofRows`, its group's bits within its words, and
`HmRow.check`'s 16-bit returned outputs), so at every staged part (`slotOut_stagedG`).
-/

namespace RecDeploy.MoreSlots

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms formBit)
open FlockSoundness.Discharge.Hm (bitL)
open RecDeploy.ResLower (outForms slotOut_table staged_hidden)

/-! ## A slot's calls at any row forms -/

/-- **Port `q`'s hm96 calls at instance `n`, with the row at forms `F`**: the salt's, `SHA-512(p_salt ‖ y) = c`, then
the row's, `SHA-512(row_prefix ‖ row) = d`. -/
def formCalls (c : Circuit) (q n : ℕ) (F : List HmNets.Lin) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  [(Hm96.Default512.saltPrefix.data.toList, hmSalt c q n, (hmCommit c q n).drop 512),
    ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [], F, hmDigest c q n)]

theorem hmCalls_eq (c : Circuit) (nm : String) (q n : ℕ) :
    hmCalls c nm q n = formCalls c q n (((rowHidden c nm n).getD []).map HmNets.Lin.var) := rfl

theorem outCalls_eq (c : Circuit) (q n : ℕ) : DeployedR.outCalls c q n = formCalls c q n (DeployedR.outRowForms c q n) :=
  rfl

theorem portCalls_eq (c : Circuit) (nm : String) (q n : ℕ) :
    DeployedR.portCalls c nm q n = formCalls c q n (DeployedR.rowForms c nm q n) := rfl

theorem accOutCalls_eq (c : Circuit) (q n : ℕ) : accOutCalls c q n = formCalls c q n (accOutForms c q n) := rfl

/-- **A slot's calls are true evaluations where it outputs its commit string**, at any row forms `F` (`hmCalls_true`'s
argument): the salt's is the commit string's second half; the row's digest form `b ⊕ Φy` is
`SHA-512(row) ⊕ M·y ⊕ Φy`, and `Φy` is `M·y` (`maskV_filter`). -/
theorem formCalls_true {c : Circuit} {q n : ℕ} {F : List HmNets.Lin} {w : Claims.Flock.Wit}
    (h : readForms w (hmCommit c q n) =
      FlockSoundness.Binding.commitString (rowPfx c q ++ readForms w F) (readForms w (hmSalt c q n))) :
    ∀ p ∈ formCalls c q n F, FlockDeploy.sha512 (p.1 ++ readForms w p.2.1) = readForms w p.2.2 := by
  have hb : ∀ t (ht : t < 1024), w.hid.getD (slotHid c n (hmBase c q n) (hmOutCol c + t)) false =
      bitL (FlockSoundness.Binding.commitString (rowPfx c q ++ readForms w F) (readForms w (hmSalt c q n))) t :=
    fun t ht => by
      have e := bitL_readForms (w := w) (fs := hmCommit c q n) (q := t) (by simpa [hmCommit] using ht)
      rw [h] at e
      rw [e]
      simp [hmCommit, formBit_var]
  have hy : ∀ j (hj : j < 1536), w.hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false =
      bitL (readForms w (hmSalt c q n)) j := fun j hj => by
    rw [bitL_readForms (fs := hmSalt c q n) (by simpa [hmSalt] using hj)]
    simp [hmSalt, formBit_var]
  intro p hp
  simp only [formCalls, List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases hp with rfl | rfl
  · rw [sha512_eq]
    refine (readForms_eq_of_bits (by simp [hmCommit, FlockSoundness.Discharge.Hm.length_digBytes])
      fun t ht => ?_).symm
    have ht' : t < 512 := by simpa [hmCommit] using ht
    rw [List.getElem_drop]
    simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var]
    rw [hb _ (by omega), FlockSoundness.Binding.commitString,
      FlockSoundness.Discharge.Hm.bitL_append_hi (length_xorMask _ _)]
    rfl
  · rw [sha512_eq]
    refine (readForms_eq_of_bits (by simp [hmDigest, FlockSoundness.Discharge.Hm.length_digBytes])
      fun i hi => ?_).symm
    have hi' : i < 512 := by simpa [hmDigest] using hi
    have hm := FlockSoundness.Discharge.Hm.bitL_mask (readForms w (hmSalt c q n)) hi'
    rw [← FlockSoundness.Refine.K512_mask] at hm
    have hx := FlockSoundness.Discharge.Hm.bitL_xorBytes (FlockSoundness.Discharge.Hm.length_digBytes
      (FlockSoundness.Refine.H512 (rowPfx c q ++ readForms w F)))
      (FlockSoundness.Refine.K512.mask_len (readForms w (hmSalt c q n))) hi'
    have hci := hb i (by omega)
    rw [FlockSoundness.Binding.commitString, FlockSoundness.Discharge.Hm.bitL_append_lo (length_xorMask _ _) hi', hx,
      hm, maskV_filter] at hci
    have hmap : ((((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
        fun j => slotHid c n (hmBase c q n) (hmIn c 2 + j)).map fun v => w.hid.getD v false) =
        ((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
          (bitL (readForms w (hmSalt c q n))) := by
      rw [List.map_map]
      exact List.map_congr_left fun j hj => hy j (List.mem_range.1 (List.mem_filter.1 hj).1)
    simp only [hmDigest, List.getElem_map, List.getElem_range, formBit, List.foldr_cons, List.foldr_map]
    simp only [← hmap, List.foldr_map] at hci
    rw [hci, Bool.xor_assoc, Bool.xor_self, Bool.xor_false]
    rfl

/-! ## A unit's output group lies in its slot -/

/-- **`HmRow.check`'s unit groups**: an untyped circuit's unit has an output group, and where the unit returns the
outputs, that group's ports are 16-bit words. -/
theorem outGroup_of_check {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) (ho : c.outNet = c.unitNet) :
    c.unit.outGroups.size ≠ 0 ∧ ∀ b ∈ (c.unit.outGroups.getD 0 default).bits.toList, b = 16 := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨u, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨-, h⟩ := FlockSoundness.ite_throw_ok' h
  obtain ⟨-, h⟩ := FlockSoundness.ite_throw_ok' h
  obtain ⟨-, h⟩ := FlockSoundness.ite_throw_ok' h
  simp only [ht] at h
  obtain ⟨-, h⟩ := FlockSoundness.ite_throw_ok' h
  obtain ⟨hug, -⟩ := FlockSoundness.ite_throw_ok' h
  refine ⟨fun h0 => hug (by simp [Array.isEmpty, h0]), fun b hb => ?_⟩
  by_contra hne
  apply hug
  have hany : ((c.unit.outGroups.getD 0 default).bits.any fun x => x != 16) = true :=
    Array.any_eq_true'.2 ⟨b, Array.mem_def.2 hb, by simpa using hne⟩
  simp only [hany, ho, bne_self_eq_false, Bool.not_false, Bool.true_and, Bool.and_true, Bool.or_true]

/-- **`Net.ofRows`' output groups' ports fit their words.** -/
theorem ofRows_outBits {name : String} {useful constPos : ℕ} {ig og : Array Group} {ra rb : Array (Array ℕ)}
    {n : Net} (h : Net.ofRows name useful constPos ig og ra rb = .ok n) :
    ∀ g ∈ og.toList, g.bits.foldl (· + ·) 0 ≤ g.words * WORD := by
  unfold Net.ofRows at h
  simp only at h
  replace h := FlockSoundness.ite_throw_ok (FlockSoundness.ite_throw_ok h)
  obtain ⟨-, h⟩ := FlockSoundness.ite_throw_ok' h
  obtain ⟨h5, -⟩ := FlockSoundness.ite_throw_ok' h
  simp only [Bool.or_eq_true, Array.any_eq_true, not_or, not_exists, decide_eq_true_eq] at h5
  intro g hg
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_def.2 hg)
  have := h5.2 i hi
  omega

/-- **A built unit's output group lies in its slot** (`slotOut_table`'s `hin`): the group lies in the unit's
`2^unitLog / WORD` words (D11, `Net.ofRows_ok`), and its ports, 16-bit words each (`outGroup_of_check`), fit its words
(`ofRows_outBits`). -/
theorem hin_of_check {c : Circuit} (hB : FlockSoundness.Built c.unit) {tags : Tags} {ls : Lean.Json}
    (ht : c.typed = none) (h : HmRow.check c tags ls = .ok ()) (ho : c.outNet = c.unitNet) :
    ∀ j < (c.unit.outGroups.getD 0 default).bits.size, ∀ t < 16,
      (c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * j + t < 2 ^ c.unit.unitLog := by
  obtain ⟨hne, h16⟩ := outGroup_of_check ht h ho
  obtain ⟨name, useful, constPos, ig, og, ra, rb, hn⟩ := hB
  obtain ⟨-, -, -, -, -, -, hog, -, -, -, -, -, -, -, hD11⟩ := Net.ofRows_ok hn
  have hpos : 0 < c.unit.outGroups.size := Nat.pos_of_ne_zero hne
  have hg : c.unit.outGroups.getD 0 default = c.unit.outGroups[0] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hpos, Option.getD_some]
  have hmem : c.unit.outGroups.getD 0 default ∈ og.toList := by
    rw [hg, ← hog]
    exact Array.getElem_mem_toList hpos
  have hD := hD11 _ hmem
  have hsum := ofRows_outBits hn _ hmem
  have hrep : (c.unit.outGroups.getD 0 default).bits.toList =
      List.replicate (c.unit.outGroups.getD 0 default).bits.size 16 :=
    List.eq_replicate_iff.2 ⟨by simp, h16⟩
  have hfold : (c.unit.outGroups.getD 0 default).bits.foldl (· + ·) 0 =
      (c.unit.outGroups.getD 0 default).bits.size * 16 := by
    rw [← Array.foldl_toList, hrep, FlockVBridge.OfVStar.foldl_add_replicate, Nat.zero_add]
  rw [hfold] at hsum
  have hW : Flock.WORD = 128 := rfl
  rw [hW] at hsum hD ⊢
  have hdiv := Nat.div_mul_le_self (2 ^ c.unit.unitLog) 128
  intro j hj t htt
  generalize 2 ^ c.unit.unitLog = P at hD hdiv ⊢
  generalize P / 128 = K at hD hdiv
  omega

/-! ## An output slot commits at any staged part -/

/-- **A staged part's output slot commits its output bits**, at any of its statement's instances (`slotOut_staged` at
any unit): the part's table has hidden outputs (`staged_hidden`) and returns them from its unit (`scopeOk`), whose
output group lies in its slot (`hin_of_check`), and holds every instance of its statement (`staged_scope`,
`oneFull_staged`), at a block of the table (`div_lt_blocks`), so `slotOut_table`. -/
theorem slotOut_stagedG {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss) {sv : VStar.Stmt}
    (hsv : ss[s]? = some sv) {c : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    {q n : ℕ} (hqs : q < c.ports.size) (hq : ¬ q < HmOut.nIn c) (hn : n < sv.instances) :
    readForms w (hmCommit c q n) = FlockSoundness.Binding.commitString
      (rowPfx c q ++ readForms w (outForms c q n)) (readForms w (hmSalt c q n)) := by
  obtain ⟨p, r', sv', hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨-, i, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have h0' : 0 < sts.size := by subst hp; exact h0
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hle : Flock.RecVStar.population p.sts[0] ≤ p.sts[0].n := by
    rw [e0]
    exact (oneFull_staged hS hsu).2 0 h0'
  obtain ⟨hscope, r2, ss2, sv2, hr2, hss2, hsv2, hpop⟩ := staged_scope hS hx h0
  have e1 : r2 = r := Option.some.inj (hr2.symm.trans hr)
  rw [e1] at hss2
  have e2 : ss2 = ss := Except.ok.inj (hss2.symm.trans hss)
  rw [e2, hsv] at hsv2
  have e3 : sv2 = sv := (Option.some.inj hsv2).symm
  rw [e3] at hpop
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨L, regs, hz⟩ := Flock.Readable.layout_of_check hck hx ht hpi h0
  have hz1 : (L.model regs).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  have hh : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).hiddenOutputs = true := by
    unfold FlockSoundness.Discharge.Exec.tagsAt
    split <;> exact staged_hidden hS hx h0
  have hs := (FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports hs
  have ho : p.sts[0].c.outNet = p.sts[0].c.unitNet := by
    unfold FlockSoundness.Discharge.Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L regs hz1
  have hb := div_lt_blocks hg0 (show n < p.sts[0].n by omega) hnbl
  subst hc0
  obtain ⟨ls, hls⟩ := P.check
  exact slotOut_table ht' hh hd hv1 ho (hin_of_check L.unit_built P.typed hls ho) L regs hz1 hqs hq hb

end RecDeploy.MoreSlots
