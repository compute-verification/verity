module

public import Proofs.Properties.Verity.Sampled.Moves
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import all Lean.Data.Json.Basic
import Mathlib.Data.String.Basic

/-! SPOT's link: the message the challenger opened the link points after. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Flock (SessionSpec RunRecord)
open FlockSoundness.Refine (ebind_ok)
open Claims.Bridge.Custody (decode_ok RootField)

theorem get?_mem {kvs : Std.TreeMap.Raw String Json} {k : String} {v : Json} (h : kvs.get? k = some v) :
    (k, v) ∈ kvs.toArray := by
  obtain ⟨⟨t⟩⟩ := kvs
  show (k, v) ∈ Std.DTreeMap.Internal.Impl.Const.toArray t
  rw [Std.DTreeMap.Internal.Impl.Const.toArray_eq_toArray_map, List.mem_toArray]
  change Std.DTreeMap.Internal.Impl.Const.get? t k = some v at h
  induction t with
  | leaf => cases h
  | inner sz k' v' l r ihl ihr =>
    simp only [Std.DTreeMap.Internal.Impl.Const.get?] at h
    simp only [Std.DTreeMap.Internal.Impl.toListModel_inner, List.map_append, List.map_cons, List.mem_append,
      List.mem_cons]
    split at h
    · exact Or.inl (ihl h)
    · exact Or.inr (Or.inr (ihr h))
    · rename_i hc
      cases h
      exact Or.inr (Or.inl (by rw [compare_eq_iff_eq.1 hc]))

theorem keys?_single {kvs : Std.TreeMap.Raw String Json} {x : String}
    (h : Spot.keys? (.obj kvs) = some #[x]) : ∃ w, kvs.toArray = #[(x, w)] := by
  simp only [Spot.keys?, Option.some.injEq] at h
  have hp := FlockSoundness.Audit.StrataExec.qsort_perm (· < ·) (kvs.toArray.map (·.1))
  rw [show (kvs.toArray.map (·.1)).qsort (· < ·) = Spot.sortStrings (kvs.toArray.map (·.1)) from rfl, h]
    at hp
  have hl : (kvs.toArray.map (·.1)).toList = [x] := List.perm_singleton.1 hp.symm
  rw [Array.toList_map, List.map_eq_singleton_iff] at hl
  obtain ⟨⟨k, w⟩, hkw, rfl⟩ := hl
  exact ⟨w, Array.toList_inj.1 hkw⟩

theorem fld_getD {j v : Json} {k : String} (h : Flock.Record.fld j k = .ok v) : Spot.getD j k = v := by
  unfold Flock.Record.fld at h
  unfold Spot.getD Spot.field?
  split at h
  · rename_i v' hv
    cases h
    rw [hv]
    rfl
  · cases h

theorem str_getD {j : Json} {k s : String} (h : Flock.Record.str j k = .ok s) : Spot.getD j k = .str s := by
  unfold Flock.Record.str at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [fld_getD hv]
  cases v <;> cases h
  rfl

theorem str_get? {kvs : Std.TreeMap.Raw String Json} {k s : String} (h : Flock.Record.str (.obj kvs) k = .ok s) :
    kvs.get? k = some (.str s) := by
  have hg := str_getD h
  unfold Spot.getD Spot.field? at hg
  simp only [Json.getObjVal?] at hg
  split at hg
  · rename_i v hv
    rw [hv]
    exact congrArg some hg
  · cases hg

theorem checkBlobsAt_trees {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (hm : p.members = none) (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ ts, _root_.Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧
      _root_.Spot.OneStageZk.treesHeld p.own ts e bs = .ok () := by
  replace h := (_root_.Spot.OneStageZk.checkBlobsAt_accept_one hm h).1
  unfold _root_.Spot.OneStageZk.checkOneAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨ts, hts, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, -, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, hth, -⟩ := ebind_ok hr
    exact ⟨ts, hts, hth⟩

theorem treesHeld_one {p : _root_.Spot.OneStageZk.Own} {ts : Array _root_.Spot.Challenger.LoggedTree}
    {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {rs : _root_.Spot.OneStageZk.RunFiles} {recB : ByteArray} {record : String} (hs : e.sessions = #[rs])
    (hb : bs.get rs.record s!"session {rs.name}'s record" = .ok recB) (hr : String.fromUTF8? recB = some record)
    (h : _root_.Spot.OneStageZk.treesHeld p ts e bs = .ok ()) :
    _root_.Spot.Challenger.sessionsTreeReason ts
      #[(rs.name, _root_.Spot.OneStageZk.rootBTables p bs record,
        (Flock.JsonParse.parse record).toOption.getD Json.null)] = none := by
  unfold _root_.Spot.OneStageZk.treesHeld at h
  rw [hs] at h
  simp only [← Array.forIn_toList, List.forIn_cons, List.forIn_nil, hb, hr, bind, Except.bind, pure, Except.pure]
    at h
  rw [show (#[].push (rs.name, _root_.Spot.OneStageZk.rootBTables p bs record,
      (Flock.JsonParse.parse record).toOption.getD Json.null) : Array (String × Array (String × Option ℕ) × Json)) =
    #[(rs.name, _root_.Spot.OneStageZk.rootBTables p bs record,
      (Flock.JsonParse.parse record).toOption.getD Json.null)] from rfl] at h
  revert h
  cases _root_.Spot.Challenger.sessionsTreeReason ts
    #[(rs.name, _root_.Spot.OneStageZk.rootBTables p bs record,
      (Flock.JsonParse.parse record).toOption.getD Json.null)] with
  | none => intro _; rfl
  | some w => intro h; cases h

theorem sessionsTreeReason_one {trees : Array _root_.Spot.Challenger.LoggedTree}
    {tables : Array (String × Option ℕ)} {n : String} {J : Json}
    (h : _root_.Spot.Challenger.sessionsTreeReason trees #[(n, tables, J)] = none) :
    _root_.Spot.Challenger.sessionTreeReason trees tables J = none ∧ trees.size = 1 := by
  unfold _root_.Spot.Challenger.sessionsTreeReason at h
  simp only [← Array.forIn_toList, List.forIn_cons, List.forIn_nil] at h
  cases hr : _root_.Spot.Challenger.sessionTreeReason trees tables J with
  | some w => simp [hr, Id.run] at h
  | none =>
    refine ⟨rfl, ?_⟩
    simp [hr, Id.run] at h
    split at h
    · rename_i h1
      exact h1.symm
    · cases h

theorem sessionTreeReason_link {trees : Array _root_.Spot.Challenger.LoggedTree} {x : String}
    {n? : Option ℕ} {J : Json} (h : _root_.Spot.Challenger.sessionTreeReason trees #[(x, n?)] J = none) :
    ∃ t kvs s rb sf rootF o, t ∈ trees ∧
      Spot.getD (Spot.getD J "link") "root_b" = .obj kvs ∧ kvs.toArray = #[(x, .str s)] ∧
      Flock.unhex s = .ok rb ∧ (∀ n, n? = some n → rb.size = n) ∧
      Spot.getD (Spot.getD J "link") "root_f" = .str sf ∧
      Flock.unhex sf = .ok rootF ∧ t.opening _root_.Spot.Challenger.TREE_SESSION 0 = some o ∧
      o.after = Flock.hexOf (Flock.Sha512.hash (rootF ++ rb)) := by
  unfold _root_.Spot.Challenger.sessionTreeReason at h
  simp only [Id.run] at h
  split at h
  · rename_i t htf
    split at h
    · split at h
      · rename_i kvs ps hm
        split at h
        · rename_i hk
          split at h
          · rename_i rootF points streams hm2
            split at h
            · rename_i hsz
              split at h
              · rename_i hlen
                split at h
                · rename_i o hlink
                  split at h
                  · rename_i hok
                    have hrb : Spot.getD (Spot.getD J "link") "root_b" = .obj kvs := by
                      simp only [Prod.mk.injEq] at hm
                      exact hm.2.1
                    rw [hrb] at hk hsz hlen hok
                    simp only [List.map_toArray, List.map_cons, List.map_nil] at hk
                    rw [show Spot.sortStrings #[x] = #[x] from qsort_single _ x, beq_iff_eq] at hk
                    obtain ⟨w, hw⟩ := keys?_single hk
                    simp only [Spot.entries, hw] at hsz hlen hok
                    have h1 := congrArg Prod.fst hm2
                    simp only at h1
                    split at h1
                    · rename_i sf hsf
                      have huf : Flock.unhex sf = .ok rootF := by
                        cases hu : Flock.unhex sf <;>
                          simp only [hu, Except.toOption, reduceCtorEq, Option.some.injEq] at h1
                        rw [h1]
                      cases w with
                      | str s =>
                        cases hus : Flock.unhex s with
                        | error _ => simp [hus, Except.toOption] at hsz
                        | ok rb =>
                          simp [hus, Except.toOption] at hok
                          have hn : ∀ n, n? = some n → rb.size = n := by
                            intro n hn
                            subst hn
                            rw [List.all_toArray, List.all_cons, List.all_nil, Bool.and_true] at hlen
                            simpa [hus, Except.toOption] using hlen
                          have hla : Spot.Challenger.linkAfter rootF #[(x, rb)] =
                              Flock.Sha512.hash (rootF ++ rb) := by
                            unfold Spot.Challenger.linkAfter
                            rw [qsort_single]
                            rfl
                          exact ⟨t, kvs, s, rb, sf, rootF, o, Array.mem_of_find?_eq_some htf, hrb, hw, hus, hn, hsf,
                            huf, hlink, hla ▸ hok.1.2⟩
                      | _ => simp at hsz
                    · cases h1
                  · cases h
                · cases h
              · cases h
            · cases h
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

theorem setupAt_tables {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {draw : Option Json}
    {spec : SessionSpec} {tabs : Array Flock.Setup} {rest : Json × Json} (hc : I.count = 1)
    (h : _root_.Spot.OneStageZk.setupAt I rd draw = .ok (spec, tabs, rest)) : ∃ y, spec.tables = #[y] := by
  unfold _root_.Spot.OneStageZk.setupAt at h
  obtain ⟨⟨spec', sts⟩, hs, h⟩ := ebind_ok h
  have hs' : _root_.Spot.OneStageZk.sessionAt I rd draw = .ok (spec', sts) := by
    cases hx : _root_.Spot.OneStageZk.sessionAt I rd draw <;> rw [hx] at hs <;> cases hs
    rfl
  obtain ⟨_, -, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, -⟩ := h
  unfold _root_.Spot.OneStageZk.sessionAt at hs'
  obtain ⟨_, -, hs'⟩ := ebind_ok hs'
  obtain ⟨_, -, hs'⟩ := ebind_ok hs'
  obtain ⟨r, hb, hs'⟩ := ebind_ok hs'
  obtain ⟨_, -, hs'⟩ := ebind_ok hs'
  obtain ⟨_, -, hs'⟩ := ebind_ok hs'
  obtain ⟨_, -, hs'⟩ := ebind_ok hs'
  cases hs'
  unfold _root_.Spot.OneStageZk.buildSessionOf at hb
  obtain ⟨_, -, hb⟩ := ebind_ok hb
  obtain ⟨x, hx, hb⟩ := ebind_ok hb
  have hset : Flock.Stmt.setupTables (Flock.Witness.zkTags I) I.circuitFile I.publicFile (Flock.Witness.zkCoins I)
      I.count I.tables draw I.partition I.program = .ok x := by
    split at hx
    · rename_i hhid
      rw [Flock.Stmt.setupTables_eq]
      simpa only [Flock.ProvedScope.circuitOf, hhid, ↓reduceIte] using hx
    · exact hx
  obtain ⟨_, -, hb⟩ := ebind_ok hb
  obtain ⟨_, -, hb⟩ := ebind_ok hb
  cases hb
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, -, hsp, -⟩ | ⟨h1, -⟩
  · exact ⟨st.tags.table, by rw [hsp]; rfl⟩
  · omega

theorem checkBlobsAt_link {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (hm : p.members = none) (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ record spec tabs rest proofs sess t o rb st, V2Verified p e bs record spec tabs rest proofs ∧
      Flock.Record.decode spec record = .ok sess ∧ sess.rootB = #[rb] ∧
      tabs[0]? = some st ∧ _root_.Spot.OneStageZk.rootBLen st = .ok rb.size ∧
      _root_.Spot.OneStageZk.treesOf p.base e.base = .ok #[t] ∧
      t.opening _root_.Spot.Challenger.TREE_SESSION 0 = some o ∧
      linkMsg record spec.tables = some (spec.sigma ++ rb).data.toList ∧
      o.after = Flock.hexOf ⟨(sha512Msg (spec.sigma ++ rb).data.toList).toArray⟩ := by
  obtain ⟨record, spec, tabs, rest, proofs, hv⟩ := checkBlobsAt_verified hm h
  obtain ⟨rs, tags0, circuitB, pubB, rd, hs, ht, hc, hpb, hrd, hrec, hst, -, hver⟩ := id hv
  obtain ⟨recB, hb, hr⟩ : ∃ recB, bs.get rs.record s!"session {rs.name}'s record" = .ok recB ∧
      String.fromUTF8? recB = some record := by
    unfold sessionRecord at hrec
    rw [hs] at hrec
    change (bs.get rs.record s!"session {rs.name}'s record").toOption.bind String.fromUTF8? = some record at hrec
    cases hg : bs.get rs.record s!"session {rs.name}'s record" with
    | error _ => rw [hg] at hrec; cases hrec
    | ok recB => rw [hg] at hrec; exact ⟨recB, rfl, hrec⟩
  obtain ⟨ts, hts, hth⟩ := checkBlobsAt_trees hm h
  have htab : _root_.Spot.OneStageZk.sessionTables = #[tags0.table] := by
    simp only [_root_.Spot.OneStageZk.sessionTables, ht, Option.map_some, Option.getD_some]
    rfl
  obtain ⟨-, sess, schs, hd, -, h0, -, -, hsch⟩ := FlockSoundness.Discharge.ZkExec.zk_verify_ok _ _ _ _ hver
  obtain ⟨_, -, hsch0, -⟩ := hsch 0 h0
  obtain ⟨n, hn⟩ : ∃ n, _root_.Spot.OneStageZk.rootBLen tabs[0] = .ok n := by
    unfold _root_.Spot.OneStageZk.rootBLen
    rw [hsch0]
    exact ⟨_, rfl⟩
  have hrt : _root_.Spot.OneStageZk.rootBTables p.own bs record = #[(tags0.table, some n)] := by
    unfold _root_.Spot.OneStageZk.rootBTables _root_.Spot.OneStageZk.rootBTablesAt
    simp only [htab, ht, hc, hpb, hrd, hst, bind, Except.bind, pure, Except.pure, Except.toOption,
      Array.mapIdx_singleton, Option.bind_some, Array.getElem?_eq_getElem h0, hn]
  have h1 := treesHeld_one hs hb hr hth
  rw [hrt] at h1
  obtain ⟨h2, hsz⟩ := sessionsTreeReason_one h1
  obtain ⟨t, kvs, s, rb, sf, rootF, o, htm, hrb, hkv, hus, hlen, hrf, huf, ho, hafter⟩ := sessionTreeReason_link h2
  have hts1 : ts = #[t] := by
    rw [eq_single hsz] at htm ⊢
    rw [Array.mem_singleton.1 htm]
  obtain ⟨j, l, -, hp, hl, -, hroot, -, -⟩ := decode_ok hd
  obtain ⟨j', l', s0, hp', hl', hs0, hu0, hsig⟩ := decode_rootF hd
  rw [hp] at hp'
  cases hp'
  rw [hl] at hl'
  cases hl'
  have hJ : (Flock.JsonParse.parse record).toOption.getD Json.null = j := by rw [hp]; rfl
  rw [hJ, fld_getD hl] at hrb hrf
  obtain ⟨y, hy⟩ := setupAt_tables rfl hst
  rw [hy] at hroot
  obtain ⟨b, u, ⟨v, s', hv1, hv2, hv3⟩, hu, hbu⟩ := List.forall₂_cons_left_iff.1 hroot
  rw [List.forall₂_nil_left_iff] at hu
  subst hu
  have hB : sess.rootB = #[b] := Array.toList_inj.1 hbu
  rw [fld_getD hv1] at hrb
  subst hrb
  have hmem := get?_mem (str_get? hv2)
  rw [hkv, Array.mem_singleton, Prod.mk.injEq] at hmem
  obtain ⟨rfl, hss⟩ := hmem
  injection hss with hss
  subst hss
  rw [hus] at hv3
  injection hv3 with hv3
  subst hv3
  rw [str_getD hs0] at hrf
  injection hrf with hrf
  subst hrf
  rw [hu0] at huf
  cases huf
  have hm : linkMsg record spec.tables = some (sess.rootF ++ rb).data.toList := by
    rw [hy]
    simp only [linkMsg, hp, hl, hs0, hu0, ok_toOption, Option.bind_some, ok_bind, qsort_single,
      mapM_single (rootBAt_of ⟨_, _, hv1, hv2, hus⟩), Option.map_some]
    rfl
  rw [hsig] at hm hafter
  refine ⟨record, spec, tabs, rest, proofs, sess, t, o, rb, tabs[0], hv, hd, hB, Array.getElem?_eq_getElem h0,
    by rw [hlen n rfl]; exact hn, hts1 ▸ hts, ho, hm, ?_⟩
  have he : ∀ b : ByteArray, (⟨b.data.toList.toArray⟩ : ByteArray) = b := fun _ => rfl
  rw [hafter, sha512Msg, he, he]

end Verity.Sampled
