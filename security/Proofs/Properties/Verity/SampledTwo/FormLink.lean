module

public import Proofs.Properties.Verity.SampledTwo.FormLast

/-! `LinkFixed` where each template's session statement is fixed; `StmtAtFixed` and
`SessionCoins.VerdictsAt` are hypotheses here. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)
open FlockVBridge.PartS (bind_ok)

theorem forIn_ok_mem {α : Type} {f : α → Flock.V Unit} :
    ∀ {l : List α} {u u' : PUnit.{1}},
      forIn l u (fun x _ => do f x; pure (ForInStep.yield PUnit.unit)) = .ok u' → ∀ q ∈ l, f q = .ok ()
  | [], _, _, _, q, hq => by cases hq
  | a :: l, u, u', h, q, hq => by
    rw [List.forIn_cons] at h
    obtain ⟨r, hr, h⟩ := bind_ok h
    obtain ⟨v, hv, hr⟩ := bind_ok hr
    cases hr
    rcases List.mem_cons.1 hq with rfl | hq
    · exact hv
    · exact forIn_ok_mem h q hq

theorem verify_publics {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {record : String}
    {proofs : Array ByteArray} (h : Flock.Zk.verify spec tabs record proofs = .ok ()) :
    ∃ sess, Flock.Record.decode spec record = .ok sess ∧
      ∀ q ∈ (tabs.zip sess.publics).toList, q.1.publics q.2 = .ok () := by
  unfold Flock.Zk.verify at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := bind_ok h; cases h0
  rw [ite_eq_right c0] at h
  by_cases c1 : spec.coinTree.isNone = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := bind_ok h; cases h0
  rw [ite_eq_right c1] at h
  obtain ⟨sess, hsess, h⟩ := bind_ok h
  obtain ⟨_, hpub, -⟩ := bind_ok h
  rw [← Array.forIn_toList] at hpub
  exact ⟨sess, hsess, forIn_ok_mem (f := fun x : Flock.Setup × ByteArray => x.1.publics x.2) hpub⟩

theorem ofCircuit_publics {st : Flock.Stmt} {p : ByteArray}
    (h : (Flock.Setup.ofCircuit st).publics p = .ok ()) : p = ByteArray.empty := by
  unfold Flock.Setup.ofCircuit at h
  dsimp only at h
  split at h
  · rename_i hc
    exact ByteArray.size_eq_zero_iff.1 (beq_iff_eq.1 hc)
  · cases h

theorem linkShown_of {J : Json} {kvs : Std.TreeMap.Raw String Json} {x s sf : String} {rb rootF : ByteArray}
    (hrb : Spot.getD (Spot.getD J "link") "root_b" = .obj kvs) (hkv : kvs.toArray = #[(x, .str s)])
    (hus : Flock.unhex s = .ok rb) (hrf : Spot.getD (Spot.getD J "link") "root_f" = .str sf)
    (huf : Flock.unhex sf = .ok rootF) : SessionCoins.linkShown J = some (rootF ++ rb) := by
  have hF : SessionCoins.linkRootF J = some rootF := by
    unfold SessionCoins.linkRootF
    rw [hrf]
    show (Flock.unhex sf).toOption = some rootF
    rw [huf]
    rfl
  have hB : SessionCoins.linkRootB J = #[(x, rb)] := by
    unfold SessionCoins.linkRootB
    rw [hrb]
    show kvs.toArray.filterMap _ = _
    rw [hkv]
    show (#[(x, Json.str s)].filterMap fun (kv : String × Json) => match kv.2 with
      | .str s => (Flock.unhex s).toOption.map (kv.1, ·)
      | _ => none) = #[(x, rb)]
    simp only [List.filterMap_toArray, List.filterMap_cons, List.filterMap_nil, hus, Except.toOption,
      Option.map_some]
  unfold SessionCoins.linkShown
  rw [hF, hB, Option.map_some]
  unfold SessionCoins.linkBytes
  rw [Verity.Sampled.qsort_single]
  rfl

theorem link_of_tree {spec : Flock.SessionSpec} {st : Flock.Stmt} {x : Msg} {record : String}
    {proofs : Array ByteArray} {ms : List Msg} {trees : Array Spot.Challenger.LoggedTree} {T : String}
    {nq : Option ℕ} {J : Json} (htab : spec.tables = #[T])
    (hver : Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok ())
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms) (hJ : Flock.JsonParse.parse record = .ok J)
    (htree : Spot.Challenger.sessionTreeReason trees #[(T, nq)] J = none) :
    ∃ rb, SessionCoins.linkShown J = some (spec.sigma ++ rb) ∧
      ms[0]? = some (0 :: FlockDeploy.linkBody [rb] [ByteArray.empty]) := by
  obtain ⟨_, kvs, s, rb, sf, rootF, _, -, hrb, hkv, hus, -, hrf, huf, -, -⟩ :=
    Verity.Sampled.sessionTreeReason_link htree
  have hshown := linkShown_of hrb hkv hus hrf huf
  obtain ⟨lv, mids, hlv, -, rfl⟩ := SessionMsgs.sessionMsgs_some hm
  have hlv' := SpotFlock.ok_of_toOption hlv
  obtain ⟨sess, hd, hpub⟩ := verify_publics hver
  obtain ⟨j, l, _, hp, hl, -, hroot, -, -⟩ := Claims.Bridge.Custody.liveOf_ok hlv'
  obtain ⟨j', l', s0, hp', hl', hs0, hu0, hsig⟩ := Verity.Sampled.decode_rootF hd
  rw [hJ] at hp hp'
  cases hp
  cases hp'
  rw [hl] at hl'
  cases hl'
  rw [Verity.Sampled.fld_getD hl] at hrb hrf
  rw [htab] at hroot
  obtain ⟨b, u, ⟨v, s', hv1, hv2, hv3⟩, hu, hbu⟩ := List.forall₂_cons_left_iff.1 hroot
  rw [List.forall₂_nil_left_iff] at hu
  subst hu
  have hB : lv.rootB = #[b] := Array.toList_inj.1 hbu
  rw [Verity.Sampled.fld_getD hv1] at hrb
  subst hrb
  have hmem := Verity.Sampled.get?_mem (Verity.Sampled.str_get? hv2)
  rw [hkv, Array.mem_singleton, Prod.mk.injEq] at hmem
  obtain ⟨-, hss⟩ := hmem
  injection hss with hss
  subst hss
  rw [hus] at hv3
  injection hv3 with hv3
  subst hv3
  rw [Verity.Sampled.str_getD hs0] at hrf
  injection hrf with hrf
  subst hrf
  rw [hu0] at huf
  cases huf
  rw [hsig] at hshown
  have hpl : sess.publics = lv.publics := FlockDeploy.KS.decode_publics hd hlv'
  obtain ⟨_, _, _, -, -, -, hf⟩ := FlockDeploy.KS.liveOf_pubs hlv'
  rw [htab] at hf
  obtain ⟨pb, u', -, hu', hpu⟩ := List.forall₂_cons_left_iff.1 hf
  rw [List.forall₂_nil_left_iff] at hu'
  subst hu'
  have hP : lv.publics = #[pb] := Array.toList_inj.1 hpu
  have hmem' : (Flock.Setup.ofCircuit st, pb) ∈ ((FlockDeploy.tabsOf #[st]).zip sess.publics).toList := by
    rw [hpl, hP]
    simp [FlockDeploy.tabsOf]
  have hpe : pb = ByteArray.empty := ofCircuit_publics (hpub _ hmem')
  refine ⟨rb, hshown, ?_⟩
  rw [hB, hP, hpe]
  exact List.getElem?_cons_zero

theorem link_at {C : CaseOf} (hsz : SessionCoins.HeldSize) {o : Verity.Exchange} (hacc : FormAccepts C o)
    {k : Case} (hk : C o = some k) {t : ℕ} {x : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec}
    {st : Flock.Stmt} {record : String} {proofs : Array ByteArray} {ms : List Msg}
    (hparts : SessionMsgs.SessionParts k t x s spec #[st] record proofs)
    (hver : Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok ())
    (htab : spec.tables = #[Flock.Tags.circuit.table])
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms) :
    ∃ rb, SessionCoins.linkShownAt C o t = some [(spec.sigma ++ rb).data.toList] ∧
      ms[0]? = some (0 :: FlockDeploy.linkBody [rb] [ByteArray.empty]) := by
  obtain ⟨l, hl⟩ := SessionCoins.interiorReads C hsz o hacc
  obtain ⟨k₁, e₁, _, _, _, _, trees, records, hk₁, he₁, -, -, hMem, hN⟩ :=
    SessionCoins.membersNamed C hsz o hacc l hl
  obtain rfl : k = k₁ := Option.some.inj (hk.symm.trans hk₁)
  have hparts₀ := hparts
  obtain ⟨_, e, fs, hf, hfs, -, -, -, -⟩ := hparts₀
  obtain rfl : e = e₁ := Except.ok.inj ((SessionMsgs.formOf_evidence hf).symm.trans he₁)
  have hS := SessionCoins.sessionRecords_of hk he₁ (SessionCoins.named_records hN)
  obtain ⟨rc, hrc, hparse⟩ := SessionCoins.record_of_parts hk hparts hm hS
  unfold SessionMsgs.filesAt at hfs
  obtain ⟨y, hy, -, -, nq, hyt⟩ := SessionCoins.forall₂_getElem? hN hfs
  have hyr : y.2.2.1 = rc := by
    rw [List.getElem?_map, hy] at hrc
    exact Option.some.inj hrc
  rw [Array.getElem?_toList] at hy
  obtain ⟨hall, -, -⟩ := Verity.Sampled.MemberForm.membersTreeReason_none hMem
  obtain ⟨_, -, -, _, -, -, hSt⟩ :=
    Verity.Sampled.MemberForm.memberTreeReason_none (hall y (Array.mem_of_getElem? hy))
  rw [hyt, hyr] at hSt
  obtain ⟨rb, hshown, h0⟩ := link_of_tree htab hver hm hparse hSt
  exact ⟨rb, SessionCoins.linkShownAt_of hS hrc hshown, h0⟩

theorem linkFixed_of (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) (hM : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j)
    (hsz : SessionCoins.HeldSize) (hv : SessionCoins.VerdictsAt A C Good fun _ => Flock.Tags.circuit.table)
    (hfix : StmtAtFixed A C Good L) : LinkFixed A C S Good L := by
  intro o o' t hA hG hA' hG' ht hp hlog hsh
  have hA₀ := hA
  have hA₀' := hA'
  obtain ⟨k, hk, -⟩ := hA₀
  obtain ⟨k', hk', -⟩ := hA₀'
  obtain ⟨x, s, spec, st, record, proofs, ms, hparts, hver, -, htab, -, hm⟩ := hv o k hA hG hk t ht
  obtain ⟨x', s', spec', st', record', proofs', ms', hparts', hver', -, htab', -, hm'⟩ := hv o' k' hA' hG' hk' t ht
  have hlog₁ : Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) := by
    have hle : A.h.stage + 1 ≤ SpotFlock.off A.h t := by
      have := SpotFlock.stage_le_off A.h t
      omega
    exact CoinMap.before_le L hle [] o.log o'.log hlog
  rcases hfix o o' k k' hA hG hA' hG' hk hk' hp hlog₁ with hstmt | halt
  · have hparts₀ := hparts
    have hparts₀' := hparts'
    obtain ⟨_, _, _, -, -, -, hx, hsu, -⟩ := hparts₀
    obtain ⟨_, _, _, -, -, -, hx', hsu', -⟩ := hparts₀'
    obtain rfl : x = x' := Option.some.inj (hx.symm.trans ((hstmt t ht).trans hx'))
    obtain rfl : spec = spec' := (Prod.mk.inj (Prod.mk.inj (Option.some.inj (hsu.symm.trans hsu'))).2).1
    obtain ⟨rb, hlk, h0⟩ := link_at hsz hA hk hparts hver htab hm
    obtain ⟨rb', hlk', h0'⟩ := link_at hsz hA' hk' hparts' hver' htab' hm'
    rw [hlk, hlk'] at hsh
    obtain rfl : rb = rb' :=
      (ByteArray.append_right_inj _).1 (ByteArray.ext (Array.toList_inj.1 (List.cons.inj (Option.some.inj hsh)).1))
    refine Or.inl ?_
    rw [hM o t 0, hM o' t 0, hk, hk']
    show (SessionMsgs.msgsAtCase k t).bind (fun ms => ms[0]?) = (SessionMsgs.msgsAtCase k' t).bind (fun ms => ms[0]?)
    rw [SessionMsgs.msgsAtCase_of_parts hparts hm, SessionMsgs.msgsAtCase_of_parts hparts' hm']
    exact h0.trans h0'.symm
  · exact Or.inr halt

end Verity.SampledTwo
