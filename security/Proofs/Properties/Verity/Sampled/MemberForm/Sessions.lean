module

public import Proofs.Properties.Verity.Sampled.MemberForm.Opened
public import Proofs.Properties.Verity.Sampled.MemberForm.Place
public import Proofs.Properties.Verity.Sampled.MovesLayout
public import Proofs.Properties.Verity.Sampled.CountsLayout
public import Proofs.Properties.Verity.Sampled.Window.Closing

/-!
# What the verifier's read of a member's session holds the device's log to

A window's sessions take, of each sampled member's session, what the verifier's read holds the device's log to at
that session's tree (`Verity.Sampled.SessionRead`). This module proves it of the per-member form's read: v2's read
(`readLog`) of release log lines `coinTreesOf` accepts, against which `Spot.Challenger.membersTreeReason` accepts
the sessions' records and evidence.

* **The session's tree** (`session_tree`): the `j`-th session's record names a logged tree, whose last opening is the
  session's closing round, opened after the session's evidence, and which less that round is the session's as
  `sessionTreeReason` holds one.
* **The record v2's read gives a tree's coins from** (`recordTree_strOf`, `find_of_nodup`, `find_record`,
  `openCoins_record`): no tree is two sessions', so the record v2's read finds first of a session's tree is that
  session's, and the coins the read shows of a round are that record's.
* **The record's link** (`record_link`): a record `sessionTreeReason` accepts on one table, decoded at a session of
  that table, has its tree's `session` round 0 opened after the SHA-512 of its link message.
* **An opening of a session's tree** (`openedOf_zero`, `session_opening`): an opening the read shows of the session's
  tree is the opening that tree shows of that round, the first of a tree its round 0.
* **Link first** (`LinkFirst`): each tree's first opening is of its stream named the session's. It is a premise
  here; `MemberForm.Links` proves it of the read of lines `membersTreeReason` accepts.
* **The read less the closing round** (`sessionOpens_read`), **with it** (`sessionRead_read`): `SessionOpens` and
  `SessionRead` at the `j`-th session, on the read, at a session of one table whose record's text decodes at a
  session on that table and its coin tree.
* **On the device's log** (`openedTrees_congr`, `coins_of_read`, `forall₂_split_left`, `sessionOpens_of_read`,
  `sessionRead_of_read`, `sessionRead_logged`, `members_place_logged`): a log whose releases v2's read reads has the
  same trees opened and sampled, the same answers, and the coins the read shows, so what the read is held to, the
  log is.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger
open Claims.Bridge.Custody (decode_ok RootField forall₂_getElem)

open private strOf from Spot.ChallengerTrees

/-! ## The session's tree -/

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Verity.Release} {trees : Array LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)}

/-- **The `j`-th session's tree, by the members' check**: its record names a logged tree `t`, the one `usedOf` lists
at the session's place; `t` less its last opening is the session's as `sessionTreeReason` holds one; and `t`'s last
opening is round 1 of its stream named the session's, opened after the session's evidence's SHA-512 in hex. -/
theorem session_tree (hM : membersTreeReason trees lines records = none) {j : ℕ}
    {x : String × Array (String × Option ℕ) × Json × String} (hx : records[j]? = some x) :
    ∃ t ∈ trees, recordTree x.2.2.1 = t.root ∧ (usedOf records)[j]? = some t.root ∧
      sessionTreeReason #[t.unclosed] x.2.1 x.2.2.1 = none ∧
      ∃ oc, t.opened.back? = some oc ∧ oc.stream = TREE_SESSION ∧ oc.round = 1 ∧ oc.after = x.2.2.2 := by
  obtain ⟨hall, -, -⟩ := membersTreeReason_none hM
  obtain ⟨t, ht, hroot, o, ho, hc, hS⟩ := memberTreeReason_none (hall x (Array.mem_of_getElem? hx))
  obtain ⟨hs, hk, -, ha⟩ := closes_fields hc
  refine ⟨t, ht, ?_, ?_, hS, o, ho, hs, hk, ha⟩
  · unfold recordTree
    rw [hroot]
  · rw [usedOf_getElem? hx, hroot, strOf_str]

/-! ## The record v2's read gives a tree's coins from -/

/-- A record's tree is the root `usedOf` lists for it. -/
theorem recordTree_strOf (J : Json) : recordTree J = strOf (getD (getD J "coin_tree") "root") := by
  unfold recordTree
  generalize getD (getD J "coin_tree") "root" = v
  cases v <;> rfl

/-- In a list whose images under `f` are distinct, the first member with a member's image is that member. -/
theorem find_of_nodup {β : Type} (f : β → String) : ∀ {l : List β} {a : β}, (l.map f).Nodup → a ∈ l →
    l.find? (f · == f a) = some a
  | [], _, _, h => absurd h List.not_mem_nil
  | b :: l, a, hn, h => by
    rw [List.map_cons, List.nodup_cons] at hn
    rw [List.find?_cons]
    by_cases hb : f b = f a
    · rcases List.mem_cons.1 h with rfl | hl
      · simp
      · exact absurd (hb ▸ List.mem_map_of_mem hl) hn.1
    · have hbf : (f b == f a) = false := beq_eq_false_iff_ne.2 hb
      rw [hbf]
      rcases List.mem_cons.1 h with rfl | hl
      · exact absurd rfl hb
      · exact find_of_nodup f hn.2 hl

/-- **The record v2's read finds first of a session's tree is that session's**, where no tree is two sessions'. -/
theorem find_record {j : ℕ} {x : String × Array (String × Option ℕ) × Json × String} (hx : records[j]? = some x)
    (hn : (usedOf records).toList.Nodup) :
    (records.toList.map (·.2.2.1)).find? (recordTree · == recordTree x.2.2.1) = some x.2.2.1 := by
  refine find_of_nodup recordTree ?_ ?_
  · have e : (records.toList.map (·.2.2.1)).map recordTree = (usedOf records).toList := by
      unfold usedOf
      rw [Array.toList_map, List.map_map]
      exact List.map_congr_left fun a _ => recordTree_strOf a.2.2.1
    rw [e]
    exact hn
  · exact List.mem_map.2 ⟨x, Array.mem_toList_iff.2 (Array.mem_of_getElem? hx), rfl⟩

/-- **The coins v2's read shows of a round are the record's**, the record it finds first of the round's tree. -/
theorem openCoins_record {recs : List Json} {ρ : String} {J : Json}
    (hf : recs.find? (recordTree · == ρ) = some J) {name : String} {k : ℕ} {b : ByteArray}
    (hb : recordRound J name k = some b) : Lines.openCoins recs ρ name k = Verity.bitsOfBytes b := by
  simp only [Lines.openCoins, hf, hb, Option.map_some, Option.getD_some]

/-! ## The record's link -/

/-- **A record `sessionTreeReason` accepts on one table has its tree's `session` round 0 opened after the SHA-512 of
its link message**: where `Record.decode` reads the record's text at a session of that table, the link message is
the session's `root_F`, then the table's `root_B` (`sessionTreeReason_link`, `decode_rootF`). -/
theorem record_link {trees : Array LoggedTree} {tb : String} {n? : Option ℕ} {J : Json}
    (h : sessionTreeReason trees #[(tb, n?)] J = none) {spec : Flock.SessionSpec} {text : String}
    {sess : Flock.RunRecord} (hd : Flock.Record.decode spec text = .ok sess)
    (hJ : Flock.JsonParse.parse text = .ok J) (hst : spec.tables = #[tb]) :
    ∃ t ∈ trees, ∃ o rb, t.opening TREE_SESSION 0 = some o ∧
      linkMsg text #[tb] = some (sess.rootF ++ rb).data.toList ∧
      o.after = Flock.hexOf (Flock.Sha512.hash (sess.rootF ++ rb)) := by
  obtain ⟨t, kvs, s, rb, sf, rootF, o, htm, hrb, hkv, hus, hlen, hrf, huf, ho, hafter⟩ := sessionTreeReason_link h
  obtain ⟨j, l, -, hp, hl, -, hroot, -, -⟩ := decode_ok hd
  obtain ⟨j', l', s0, hp', hl', hs0, hu0, hsig⟩ := decode_rootF hd
  rw [hp] at hp'
  cases hp'
  rw [hl] at hl'
  cases hl'
  have hJj : J = j := Except.ok.inj (hJ.symm.trans hp)
  subst hJj
  rw [fld_getD hl] at hrb hrf
  rw [hst] at hroot
  obtain ⟨b, u, ⟨v, s', hv1, hv2, hv3⟩, hu, hbu⟩ := List.forall₂_cons_left_iff.1 hroot
  rw [List.forall₂_nil_left_iff] at hu
  subst hu
  rw [fld_getD hv1] at hrb
  subst hrb
  have hmem := get?_mem (str_get? hv2)
  rw [hkv, Array.mem_singleton, Prod.mk.injEq] at hmem
  obtain ⟨-, hss⟩ := hmem
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
  have hm : linkMsg text #[tb] = some (sess.rootF ++ rb).data.toList := by
    simp only [linkMsg, hp, hl, hs0, hu0, ok_toOption, Option.bind_some, ok_bind, qsort_single,
      mapM_single (rootBAt_of ⟨_, _, hv1, hv2, hus⟩), Option.map_some]
    rfl
  exact ⟨t, htm, o, rb, ho, hm, hafter⟩

/-! ## An opening of a session's tree -/

/-- A log that didn't open a tree opened no round of it. -/
theorem openedOf_zero {L : List Verity.Release} {i : ℕ} (h : i ∉ openedTrees L) (s : ℕ) : openedOf L i s = 0 := by
  rcases Nat.eq_zero_or_pos (openedOf L i s) with h0 | hpos
  · exact h0
  · have hpos' : 0 < L.countP fun r => r.frame.opensAt i s := hpos
    obtain ⟨r, hr, hp⟩ := mem_of_countP_pos hpos'
    obtain ⟨n, hn⟩ := opening_of_opensAt hp
    exact absurd (opening_mem_openedTrees hr hn) h

/-- **An opening the read shows of a tree read is the opening that tree shows of that round**: where `t` is the tree
read first of its root at `i`, an opening `x` the read shows of tree `i`'s stream `si`, named `name` in the `Hello`s
before it, of `n` coins, is the opening `t` shows of the round of `name` that the read's openings of that stream
before `x` count, of `n` coins, after bytes that in hex are what `x` answers, with the coins the records give that
round (`read_opening`, `tree_of_key`). -/
theorem session_opening (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    {t : LoggedTree} (ht : t ∈ trees) {i : ℕ} (hi : (heldTrees trees).findIdx? (·.1 == t.root) = some i)
    {pre : List Verity.Release} {x : Verity.Release} {post : List Verity.Release} (he : log = pre ++ x :: post)
    {si n : ℕ} (hx : x.frame = .opening i si n) {name : String}
    (hname : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[si]?) = some name) :
    ∃ o, t.opening name (openedOf pre i si) = some o ∧ o.n = n ∧ Flock.unhex o.after = .ok x.answers ∧
      x.coins = Lines.openCoins recs t.root name (openedOf pre i si) := by
  obtain ⟨t', ht', name', o, hkey, hname', ho, hn, hun, hco⟩ := read_opening hT hR he hx
  have e : t' = t := tree_of_key hT ht ht' hi hkey
  have e2 : name' = name := Option.some.inj (hname'.symm.trans hname)
  rw [e, e2] at ho hco
  exact ⟨o, ho, hn, hun, hco⟩

/-! ## The read of a session, less its closing round -/

/-- **Each tree's first opening is of its stream named the session's**, in the log `log`. -/
def LinkFirst (log : List Verity.Release) : Prop :=
  ∀ L₁ x L₂ i s n, log = L₁ ++ x :: L₂ → x.frame = .opening i s n → i ∉ openedTrees L₁ →
    ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION

/-- **What the per-member form's read of its `j`-th session holds the read to, less the closing round**
(`Verity.Sampled.SessionOpens`): where `coinTreesOf` accepts the release log's lines, `readLog` reads them at the
sessions' records, `membersTreeReason` accepts the sessions `records` against them, each tree's first opening is of
its stream named the session's (`LinkFirst`), and the `j`-th session, of one table `tb`, has a record whose text
parses as the record the check read and decodes at a session of that table on its coin tree under SHA-512
digests. -/
theorem sessionOpens_read (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) (hrecs : recs = records.toList.map (·.2.2.1))
    (hLF : LinkFirst log) {j : ℕ} {x : String × Array (String × Option ℕ) × Json × String}
    (hx : records[j]? = some x) {tb : String} {n? : Option ℕ} (htab : x.2.1 = #[(tb, n?)]) {text : String}
    (hJ : Flock.JsonParse.parse text = .ok x.2.2.1) {spec : Flock.SessionSpec} {sess : Flock.RunRecord}
    (hd : Flock.Record.decode spec text = .ok sess) (hc : spec.coinTree = some (Flock.Zk.coinSpec #[tb]))
    (hst : spec.tables = #[tb]) (h512 : spec.sha512Digests = true) :
    SessionOpens log #[tb] j text x.2.2.1 sess := by
  obtain ⟨-, hnod, hO⟩ := membersTreeReason_none hM
  obtain ⟨t, ht, hroot, hu, hS, oc, hback, hcs, hck, -⟩ := session_tree hM hx
  obtain ⟨tree, htree⟩ := members_tree hT hR hM
  have hF := coinTreesOf_fold recs hT
  have hjlt : j < records.size := (Array.getElem?_eq_some_iff.1 hx).1
  have hplace := members_place hT hR hM htree hjlt
  have hi := htree j _ hu
  have hnodup := readLog_nodup hT hR
  have hidx : ∀ i, (openedTrees log)[j]? = some i → i = tree j :=
    fun i h => Option.some.inj (h.symm.trans hplace)
  obtain ⟨Lc, c, Lc₂, ic, sc, nc, -, -, hkeyc, -, -, -, -⟩ := closing_read hT hR hO hnod ht
    (Array.mem_of_back? hback) hcs hck
  have hic : ic = tree j := Option.some.inj ((keyAt_idx hkeyc).symm.trans hi)
  rw [hic] at hkeyc
  have hfind : recs.find? (recordTree · == t.root) = some x.2.2.1 := by
    rw [hrecs, ← hroot]
    exact find_record hx hnod
  have hfirst : ∀ L₁ y L₂ i s n, log = L₁ ++ y :: L₂ → y.frame = .opening i s n → i ∉ openedTrees L₁ →
      (openedTrees L₁).length = j →
        ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION ∧
          ∃ o, t.opening TREE_SESSION 0 = some o ∧ o.n = n ∧ Flock.unhex o.after = .ok y.answers ∧
            y.coins = Lines.openCoins recs t.root TREE_SESSION 0 := by
    intro L₁ y L₂ i s n he hy hni hlen
    have hτ : (openedTrees log)[j]? = some i := by
      have h := (openedTrees_new_at hy hni L₂).1
      rw [← he, hlen] at h
      exact h
    have e := hidx i hτ
    subst e
    have hname := hLF L₁ y L₂ _ s n he hy hni
    obtain ⟨o, ho, hon, hun, hco⟩ := session_opening hT hR ht hi he hy hname
    rw [openedOf_zero hni s] at ho hco
    exact ⟨hname, o, ho, hon, hun, hco⟩
  have hlater : ∀ L₁ y L₂ i s n name, log = L₁ ++ y :: L₂ → y.frame = .opening i s n →
      (openedTrees L₁)[j]? = some i → ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == TREE_SESSION) = false →
        openedOf L₁ i s < recordRounds x.2.2.1 name ∧
          ∃ o, t.opening name (openedOf L₁ i s) = some o ∧ o.n = n ∧ Flock.unhex o.after = .ok y.answers ∧
            y.coins = Lines.openCoins recs t.root name (openedOf L₁ i s) := by
    intro L₁ y L₂ i s n name he hy hτ hname hnT
    have hτ' : (openedTrees log)[j]? = some i := by
      rw [he]
      exact openedTrees_getElem?_append hτ (y :: L₂)
    have e := hidx i hτ'
    subst e
    obtain ⟨o, ho, hon, hun, hco⟩ := session_opening hT hR ht hi he hy hname
    have hom : o ∈ t.opened := Array.mem_of_find?_eq_some ho
    have hop := Array.find?_some ho
    simp only [Bool.and_eq_true, beq_iff_eq] at hop
    have hne : o ≠ oc := by
      intro e
      have h1 : name = TREE_SESSION := by rw [← hop.1, e, hcs]
      rw [h1] at hnT
      simp at hnT
    have hrec := tree_recorded hS o (mem_unclosed hom hback hne)
    rw [hop.1, hop.2] at hrec
    exact ⟨hrec, o, ho, hon, hun, hco⟩
  refine ⟨hJ, recordRound_isSome hS, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L₁ y L₂ i s n he hy hni hlen
    obtain ⟨hname, o, ho, -, hun, hco⟩ := hfirst L₁ y L₂ i s n he hy hni hlen
    refine ⟨hname, ?_, ?_⟩
    · have hS' := hS
      rw [htab] at hS'
      obtain ⟨t', ht', o', rb, ho', hm, hafter⟩ := record_link hS' hd hJ hst
      have et := mem_one ht'
      subst et
      have e : o' = o := Option.some.inj ((unclosed_opening ho').symm.trans ho)
      rw [e] at hafter
      rw [hafter] at hun
      have hans : y.answers = Flock.Sha512.hash (sess.rootF ++ rb) :=
        (Except.ok.inj ((FlockComplete.JsonRT.unhex_hexOf _).symm.trans hun)).symm
      have hb : ∀ b : ByteArray, (⟨b.data.toList.toArray⟩ : ByteArray) = b := fun _ => rfl
      refine ⟨_, hm, ?_⟩
      rw [hans, sha512Msg, hb]
    · obtain ⟨b, hb⟩ := LinkTrees.link_some hS
      refine ⟨b, hb, fun _ => ?_⟩
      rw [hco]
      exact openCoins_record hfind hb
  · intro L₁ y L₂ i s n name he hy hτ hname hnT
    obtain ⟨hrec, o, ho, -, hun, hco⟩ := hlater L₁ y L₂ i s n name he hy hτ hname hnT
    refine ⟨hrec, fun st hst' hk => ?_, fun b hb _ => ?_⟩
    · obtain ⟨t', ht', -, streams, hstr, hallr⟩ := rounds_held hS
      have et := mem_one ht'
      subst et
      obtain ⟨j', recs', hp, hrecs', hdec⟩ := decode_sha hd
      rw [hJ] at hp
      have ej : x.2.2.1 = j' := Except.ok.inj hp
      subst ej
      have hsr := arr_getD hrecs'
      rw [hstr] at hsr
      have es : streams = recs' := Lean.Json.arr.inj hsr
      subst es
      have hsn : st.name = name := by simpa using Array.find?_some hst'
      obtain ⟨xs, hxs, hxn, rs, hrs, hrds⟩ := Properties.Spot.SecurityProofs.forall₂_mem_right hdec st
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hst'))
      have hkr : openedOf L₁ i s < rs.size := by
        have e := hrds.length_eq
        simp only [Array.length_toList] at e
        omega
      obtain ⟨sx, hsx, hux⟩ := forall₂_getElem hrds hkr
      have hdk : spec.dk = "sha512" := by
        unfold Flock.SessionSpec.dk
        simp [h512]
      rw [show s!"msg_{spec.dk}" = "msg_sha512" by simp [hdk, toString]] at hsx
      obtain ⟨-, o', ho', hmsg⟩ := hallr xs (Array.mem_toList_iff.1 hxs) name rs
        (by rw [← hsn]; exact str_getD hxn) (arr_getD hrs) (rs[openedOf L₁ i s], openedOf L₁ i s)
        (Array.mem_zipIdx_iff_getElem?.2 (Array.getElem?_eq_getElem hkr))
      have e : o' = o := Option.some.inj ((unclosed_opening ho').symm.trans ho)
      rw [e] at hmsg
      have esx : sx = o.after := Lean.Json.str.inj ((str_getD hsx).symm.trans hmsg)
      rw [esx] at hux
      exact Except.ok.inj (hun.symm.trans hux)
    · rw [hco]
      exact openCoins_record hfind hb
  · intro i s name hτ hname hnT
    have e := hidx i hτ
    subst e
    exact tree_rounds_le hT hR ht hS (KeyAt.of_named hR hF hkeyc hnodup hname)
  · intro name hpos i hτ
    have e := hidx i hτ
    subst e
    obtain ⟨i', si, hk⟩ := tree_keyAt hT ht hS hpos
    have e' : i' = tree j := Option.some.inj ((keyAt_idx hk).symm.trans hi)
    rw [e'] at hk
    exact ⟨si, readLog_named hR hF hk⟩
  · intro i hτ
    have e := hidx i hτ
    subst e
    exact helloSpec_of_text hd hc hJ hT hR (List.forall₂_same.2 fun _ _ => ⟨rfl, rfl⟩) ht hS hkeyc
  · intro L₁ y L₂ i s n he hy hni hlen b hb
    obtain ⟨-, o, ho, hon, -, -⟩ := hfirst L₁ y L₂ i s n he hy hni hlen
    obtain ⟨t', ht', -, hlink, -⟩ := Trees.session_counted hS
    have et := mem_one ht'
    subst et
    obtain ⟨o', ho', hn', -⟩ := hlink TREE_SESSION 0 b (by simp) hb
    have e : o' = o := Option.some.inj ((unclosed_opening ho').symm.trans ho)
    rw [e] at hn'
    exact hon.symm.trans hn'
  · intro L₁ y L₂ i s n name he hy hτ hname hnT st hst' hk
    obtain ⟨hrec, o, ho, hon, -, -⟩ := hlater L₁ y L₂ i s n name he hy hτ hname hnT
    have hcnt := recordCount_decode hd hJ hnT hst' hk
    obtain ⟨cs, hcs'⟩ := Option.isSome_iff_exists.1 (recordRound_isSome hS name _ hrec)
    obtain ⟨t', ht', -, o', ho', hn', -⟩ := Trees.round_counted hS hcs' hcnt
    have et := mem_one ht'
    subst et
    have e : o' = o := Option.some.inj ((unclosed_opening ho').symm.trans ho)
    rw [e] at hn'
    rw [getElem!_pos st.rounds _ hk]
    exact hon.symm.trans hn'

/-! ## The read of a session -/

/-- **What the per-member form's read of its `j`-th session holds the read to** (`Verity.Sampled.SessionRead`):
with `sessionOpens_read`'s premises, at the evidence `w` whose SHA-512 in hex the check held the session's closing
round to. -/
theorem sessionRead_read (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) (hrecs : recs = records.toList.map (·.2.2.1))
    (hLF : LinkFirst log) {j : ℕ} {x : String × Array (String × Option ℕ) × Json × String}
    (hx : records[j]? = some x) {tb : String} {n? : Option ℕ} (htab : x.2.1 = #[(tb, n?)]) {text : String}
    (hJ : Flock.JsonParse.parse text = .ok x.2.2.1) {spec : Flock.SessionSpec} {sess : Flock.RunRecord}
    (hd : Flock.Record.decode spec text = .ok sess) (hc : spec.coinTree = some (Flock.Zk.coinSpec #[tb]))
    (hst : spec.tables = #[tb]) (h512 : spec.sha512Digests = true) {w : Claims.Msg}
    (hev : x.2.2.2 = sha512Hex ⟨w.toArray⟩) : SessionRead log #[tb] j text x.2.2.1 sess w := by
  have hopens := sessionOpens_read hT hR hM hrecs hLF hx htab hJ hd hc hst h512
  obtain ⟨-, hnod, hO⟩ := membersTreeReason_none hM
  obtain ⟨t, ht, -, hu, -, oc, hback, hcs, hck, hca⟩ := session_tree hM hx
  obtain ⟨tree, htree⟩ := members_tree hT hR hM
  have hjlt : j < records.size := (Array.getElem?_eq_some_iff.1 hx).1
  have hplace := members_place hT hR hM htree hjlt
  obtain ⟨Lc, c, L₂, i, s, n, he, hcf, hkey, hname, hcount, hun, hlast⟩ := closing_read hT hR hO hnod ht
    (Array.mem_of_back? hback) hcs hck
  have hi : i = tree j := Option.some.inj ((keyAt_idx hkey).symm.trans (htree j _ hu))
  subst hi
  have hmem : tree j ∈ openedTrees Lc := by
    by_contra hni
    have h0 := openedOf_zero hni s
    omega
  obtain ⟨v, hv⟩ := List.getElem?_of_mem hmem
  have hv' := openedTrees_getElem?_append hv (c :: L₂)
  rw [← he] at hv'
  have hvj : v = j := nodup_getElem?_inj (openedTrees_nodup log) hv' hplace
  rw [hvj] at hv
  rw [hca] at hun
  have hw : sha512Msg w = c.answers.data.toList := by
    rw [answers_of_hex hun hev, sha512Msg]
  rw [he] at hopens ⊢
  exact SessionRead.of_dropped hcf hv hname hcount hw hlast (hopens.drop hcf hmem hname)

/-! ## On the device's log -/

theorem foldl_openStep_congr : ∀ {l l' : List Verity.Release},
    List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l' →
      ∀ acc, l.foldl openStep acc = l'.foldl openStep acc
  | _, _, .nil, _ => rfl
  | x :: _, y :: _, .cons h t, acc => by
    rw [List.foldl_cons, List.foldl_cons]
    have e : openStep acc x = openStep acc y := by
      unfold openStep
      rw [h.1]
    rw [e]
    exact foldl_openStep_congr t _

/-- **Logs of the same frames open the same trees, in the same order.** -/
theorem openedTrees_congr {l l' : List Verity.Release}
    (h : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l') :
    openedTrees l = openedTrees l' :=
  foldl_openStep_congr h []

/-- A release of a list related to another, with the releases before and after it. -/
theorem forall₂_split_left {α β : Type} {R : α → β → Prop} {r : α} {post : List α} :
    ∀ {pre : List α} {l' : List β}, List.Forall₂ R (pre ++ r :: post) l' →
      ∃ pre' r' post', l' = pre' ++ r' :: post' ∧ List.Forall₂ R pre pre' ∧ R r r' ∧ List.Forall₂ R post post'
  | [], _, .cons h t => ⟨[], _, _, rfl, .nil, h, t⟩
  | _ :: _, _, .cons h t =>
    let ⟨pre', r', post', he, hp, hr, hq⟩ := forall₂_split_left t
    ⟨_ :: pre', r', post', by rw [he]; rfl, .cons h hp, hr, hq⟩

/-- **The coins v2's read shows of a release are the release's.** -/
theorem coins_of_read {x y : Verity.Release} (h : (Binding.Read.ofV2 x).Of y) (hne : x.coins ≠ []) :
    y.coins = x.coins := by
  have hm : x.coins ∈ (Binding.Read.ofV2 x).coins := by
    show x.coins ∈ (if x.coins.isEmpty then none else some x.coins)
    split
    · rename_i he
      exact absurd (List.isEmpty_iff.1 he) hne
    · exact Option.mem_def.2 rfl
  exact (h.2.2 x.coins hm).symm

variable {lr Lg : List Verity.Release} {tables : Array String} {u : ℕ} {text : String} {J : Json}
  {sess : Flock.RunRecord}

/-- **What v2's read is held to less the closing round, the device's log is**: where v2's read `lr` reads the
releases of the log `Lg`, `SessionOpens` on `lr` is `SessionOpens` on `Lg`. -/
theorem sessionOpens_of_read (hl : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) Lg)
    (h : SessionOpens lr tables u text J sess) : SessionOpens Lg tables u text J sess := by
  have hR := List.forall₂_map_left_iff.1 hl
  have hfa : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) lr Lg :=
    hR.imp fun _ _ h => ⟨h.1, h.2.1⟩
  have hT := openedTrees_congr hfa
  have hH := helloSpecs_congr hfa
  have hsplit : ∀ {L₁' : List Verity.Release} {y : Verity.Release} {L₂' : List Verity.Release},
      Lg = L₁' ++ y :: L₂' → ∃ L₁ x L₂, lr = L₁ ++ x :: L₂ ∧ openedTrees L₁ = openedTrees L₁' ∧
        helloSpecs L₁ = helloSpecs L₁' ∧ (∀ i s, openedOf L₁ i s = openedOf L₁' i s) ∧
        (Binding.Read.ofV2 x).Of y := by
    intro L₁' y L₂' he
    obtain ⟨L₁, x, L₂, hlr, hpre, hxy, -⟩ :=
      forall₂_split (show List.Forall₂ _ lr (L₁' ++ y :: L₂') by rw [← he]; exact hR)
    have hp : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) L₁ L₁' :=
      hpre.imp fun _ _ h => ⟨h.1, h.2.1⟩
    exact ⟨L₁, x, L₂, hlr, openedTrees_congr hp, helloSpecs_congr hp, fun i s => openedOf_congr hp, hxy⟩
  refine ⟨h.parse, h.coins, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L₁' y L₂' i s n he hy hni hlen
    obtain ⟨L₁, x, L₂, hlr, hT₁, hH₁, -, hxy⟩ := hsplit he
    rw [← hT₁] at hni hlen
    obtain ⟨hname, ⟨w, hw, hwa⟩, b, hb, hbc⟩ := h.link L₁ x L₂ i s n hlr (hxy.1.trans hy) hni hlen
    refine ⟨by rw [← hH₁]; exact hname, ⟨w, hw, by rw [← hxy.2.1]; exact hwa⟩, b, hb, fun hne => ?_⟩
    rw [coins_of_read hxy (by rw [hbc hne]; exact hne)]
    exact hbc hne
  · intro L₁' y L₂' i s n name he hy hτ hname hnT
    obtain ⟨L₁, x, L₂, hlr, hT₁, hH₁, hO₁, hxy⟩ := hsplit he
    rw [← hT₁] at hτ
    rw [← hH₁] at hname
    obtain ⟨hlt, hans, hco⟩ := h.round L₁ x L₂ i s n name hlr (hxy.1.trans hy) hτ hname hnT
    rw [← hO₁ i s]
    refine ⟨hlt, fun st hst hk => ?_, fun b hb hne => ?_⟩
    · rw [← hxy.2.1]
      exact hans st hst hk
    · rw [coins_of_read hxy (by rw [hco b hb hne]; exact hne)]
      exact hco b hb hne
  · intro i s name hτ hname hnT
    rw [← hT] at hτ
    rw [← hH] at hname
    rw [← openedOf_congr hfa]
    exact h.full i s name hτ hname hnT
  · intro name hpos i hτ
    rw [← hT] at hτ
    rw [← hH]
    exact h.key name hpos i hτ
  · intro i hτ
    rw [← hT] at hτ
    rw [← hH]
    exact h.tree i hτ
  · intro L₁' y L₂' i s n he hy hni hlen b hb
    obtain ⟨L₁, x, L₂, hlr, hT₁, -, -, hxy⟩ := hsplit he
    rw [← hT₁] at hni hlen
    exact h.linkCount L₁ x L₂ i s n hlr (hxy.1.trans hy) hni hlen b hb
  · intro L₁' y L₂' i s n name he hy hτ hname hnT st hst hk
    obtain ⟨L₁, x, L₂, hlr, hT₁, hH₁, hO₁, hxy⟩ := hsplit he
    rw [← hT₁] at hτ
    rw [← hH₁] at hname
    rw [← hO₁ i s] at hk ⊢
    exact h.roundCount L₁ x L₂ i s n name hlr (hxy.1.trans hy) hτ hname hnT st hst hk

/-- **What v2's read is held to, the device's log is**: where v2's read `lr` reads the releases of the log `Lg`,
`SessionRead` on `lr` is `SessionRead` on `Lg`. -/
theorem sessionRead_of_read (hl : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) Lg) {w : Claims.Msg}
    (h : SessionRead lr tables u text J sess w) : SessionRead Lg tables u text J sess w := by
  have hR := List.forall₂_map_left_iff.1 hl
  have hfa : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) lr Lg :=
    hR.imp fun _ _ h => ⟨h.1, h.2.1⟩
  refine ⟨sessionOpens_of_read hl h.toSessionOpens, fun i hτ => ?_⟩
  rw [← openedTrees_congr hfa] at hτ
  obtain ⟨Lc, x, L₂, s, n, he, hx, hu, hname, hk, hw, hlast⟩ := h.close i hτ
  obtain ⟨Lc', y, L₂', hLg, hpc, hxy, -⟩ :=
    forall₂_split_left (show List.Forall₂ _ (Lc ++ x :: L₂) Lg by rw [← he]; exact hR)
  have hpc' : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) Lc Lc' :=
    hpc.imp fun _ _ h => ⟨h.1, h.2.1⟩
  refine ⟨Lc', y, L₂', s, n, hLg, hxy.1.symm.trans hx, ?_, ?_, ?_, ?_, ?_⟩
  · rw [← openedTrees_congr hpc']
    exact hu
  · rw [← helloSpecs_congr hpc']
    exact hname
  · rw [← openedOf_congr hpc']
    exact hk
  · rw [← hxy.2.1]
    exact hw
  · intro L₁' y' L₃' s' n' he' hy'
    obtain ⟨L₁, x', L₃, hlr, hp1, hx'y', -⟩ :=
      forall₂_split (show List.Forall₂ _ lr (L₁' ++ y' :: L₃') by rw [← he']; exact hR)
    have h1 := hlast L₁ x' L₃ s' n' hlr (hx'y'.1.trans hy')
    rw [← hp1.length_eq, ← hpc.length_eq]
    exact h1

/-- **What the per-member form's read of its `j`-th session holds the device's log to**
(`Verity.Sampled.SessionRead`): `sessionRead_read`, on a log `Lg` whose releases the read reads. -/
theorem sessionRead_logged (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some lr)
    (hM : membersTreeReason trees lines records = none) (hrecs : recs = records.toList.map (·.2.2.1))
    (hLF : LinkFirst lr) (hl : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) Lg) {j : ℕ}
    {x : String × Array (String × Option ℕ) × Json × String} (hx : records[j]? = some x) {tb : String}
    {n? : Option ℕ} (htab : x.2.1 = #[(tb, n?)]) {text : String} (hJ : Flock.JsonParse.parse text = .ok x.2.2.1)
    {spec : Flock.SessionSpec} {sess : Flock.RunRecord} (hd : Flock.Record.decode spec text = .ok sess)
    (hc : spec.coinTree = some (Flock.Zk.coinSpec #[tb])) (hst : spec.tables = #[tb])
    (h512 : spec.sha512Digests = true) {w : Claims.Msg} (hev : x.2.2.2 = sha512Hex ⟨w.toArray⟩) :
    SessionRead Lg #[tb] j text x.2.2.1 sess w :=
  sessionRead_of_read hl (sessionRead_read hT hR hM hrecs hLF hx htab hJ hd hc hst h512 hev)

/-- **The `j`-th tree the device's log opens is the `j`-th session's, and it opens as many trees as sessions**
(`members_place`, `members_count`, on a log whose releases the read reads). -/
theorem members_place_logged (hT : coinTreesOf lines = .ok trees)
    (hR : readLog label after recs ξ lines = some lr) (hM : membersTreeReason trees lines records = none)
    (hl : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) Lg) :
    (openedTrees Lg).length = records.size ∧ ∀ j, j < records.size → ∃ τ, (openedTrees Lg)[j]? = some τ := by
  have hfa : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) lr Lg :=
    (List.forall₂_map_left_iff.1 hl).imp fun _ _ h => ⟨h.1, h.2.1⟩
  rw [← openedTrees_congr hfa]
  obtain ⟨tree, htree⟩ := members_tree hT hR hM
  exact ⟨members_count hT hR hM, fun j hj => ⟨tree j, members_place hT hR hM htree hj⟩⟩

end Verity.Sampled.MemberForm
