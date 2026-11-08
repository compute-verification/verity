module

public import Proofs.Properties.Verity.Sampled.MovesReach
public import Proofs.Properties.Verity.Sampled.LinkPts
public import Proofs.Properties.Verity.Sampled.Link

/-!
# The roots decoded at the first stop are the record's `root_B`

`MovesOf` asks that the roots the decoders read at the first stop, `D.root` of its modes, be the `root_B` of the session
v2's record decodes to (its clause (ii)). At the readings and the decoders at the session's layout, where the first stop
fixes the link's round, the play's sampled verdict accepts, the log v2's check read is the device's, and the device's
first opening is of the stream `Spot.Challenger.TREE_SESSION` of the tree it opens (the order
`SampleLinkFirst` states), they are (`firstStop_rootB`):

* **The bytes the link answers** (`link_answers`): v2's read of the device's first opening is the read of the release
  log's line opening round 0 of the session stream of the log's one tree (`readLog_round`), and that line is the one
  `coinTreesOf` takes as that tree's round (`opensRound_unique`, `opened_after`), which the challenger opened after the
  SHA-512 of the link's message in v2's record (`checkBlobsAt_link`). So the opening answers that message's SHA-512.
* **The link's round's mode** (`RoundsReach.stop_modes_reveal`): off the misses, the round the first stop fixes, the
  link's, decodes the message the play reveals for it, the link's message in v2's record at the session's tables
  (`v2_tables`), whose SHA-512 the opening answers; no other round is fixed there (`readingAt_unique`).
* **Its roots** (`root_record`): the link's message is `Σ`, then the one table's `root_B`, which the decoders read past
  `Σ`'s 64 bytes.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Verity.Game
open Spot Spot.Challenger
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The line `coinTreesOf` takes as a tree's round -/

/-- **A release log line opening a round, after the bytes the tree shows**: a coin tree's line opening round `o.round`
of stream `o.stream` of the tree whose root is `root` (`OpensRound`), whose `after` is `o.after` and whose `n` is
`o.n`. -/
def AfterLine (l root : String) (o : TreeOpening) : Prop :=
  OpensRound l root o.stream o.round ∧
    ∃ d, event l = .ok d ∧ getD d "after" = .str o.after ∧ nat? (getD d "n") = some o.n

/-- Each round a tree of `trees` shows opened is a line of `log` opening it after the bytes the tree shows. -/
def OpenedAfter (log : List String) (trees : Array LoggedTree) : Prop :=
  ∀ t ∈ trees, ∀ o ∈ t.opened, ∃ l ∈ log, AfterLine l t.root o

/-- **Each round the release log's trees show opened is a line of the log opening it**, naming that tree, with the
round's stream and number and the bytes it was opened after (`coinTreesOf`). -/
theorem opened_after {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    OpenedAfter log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Trees.forIn_inv_run (I := OpenedAfter) ?_ (fun t ht => absurd ht (by simp)) h
  intro pre line ts r hI hr
  have keep : ∀ ts', (∀ t ∈ ts', t ∈ ts ∨ t.opened = #[]) → OpenedAfter (pre ++ [line]) ts' := by
    intro ts' hts t ht o ho
    rcases hts t ht with ht | he
    · obtain ⟨l, hl, hol⟩ := hI t ht o ho
      exact ⟨l, List.mem_append_left _ hl, hol⟩
    · rw [he] at ho
      simp at ho
  split at hr
  case isFalse => cases hr; exact ⟨ts, rfl, keep ts fun t ht => .inl ht⟩
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, Trees.ok_bind] at hr
    split at hr
    · split at hr
      case isFalse => cases hr
      split at hr
      case isTrue => cases hr
      cases hr
      refine ⟨_, rfl, keep _ fun t ht => ?_⟩
      rcases Array.mem_push.1 ht with ht | rfl
      · exact .inl ht
      · exact .inr rfl
    · rename_i hopen
      split at hr
      case isFalse => cases hr
      split at hr
      case h_2 => cases hr
      rename_i root htree
      split at hr
      case h_2 => cases hr
      rename_i i hidx
      split at hr
      case h_2 => cases hr
      rename_i t hget
      split at hr
      case h_2 => cases hr
      rename_i s hstream
      split at hr
      case h_2 => cases hr
      rename_i k n hkn
      obtain ⟨hi, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
      have hti : ts[i] = t := by
        rw [Array.getElem?_eq_getElem hi] at hget
        exact Option.some.inj hget
      rw [hti] at hp
      have hroot : t.root = root := beq_iff_eq.1 hp
      have htmem : t ∈ ts := hti ▸ Array.getElem_mem hi
      simp only [Prod.mk.injEq] at hkn
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          have hhex := hcond.1.2
          cases hr
          refine ⟨_, rfl, ?_⟩
          intro t' ht' o ho
          rcases Trees.mem_modify ht' with ht' | ⟨a, ha, rfl⟩
          · obtain ⟨l, hl, hol⟩ := hI t' ht' o ho
            exact ⟨l, List.mem_append_left _ hl, hol⟩
          obtain rfl : a = t := Option.some.inj (ha.symm.trans hget)
          simp only [Array.mem_push] at ho
          rcases ho with ho | rfl
          · obtain ⟨l, hl, hol⟩ := hI _ htmem o ho
            exact ⟨l, List.mem_append_left _ hl, hol⟩
          refine ⟨line, List.mem_append_right _ (List.mem_singleton_self _),
            ⟨htl, d, hev, by simpa using hopen, by simpa [hroot] using htree, hstream, hkn.1⟩, d, hev, ?_, hkn.2⟩
          cases hc : getD d "after" <;> rw [hc] at hhex <;> first | rfl | cases hhex
        · cases hr

/-- Two places of a list holding elements `p` accepts, with as many such elements before each, are one place. -/
theorem eq_of_countP_eq {β : Type} {p : β → Bool} :
    ∀ {a a' b b' : List β} {x x' : β}, a ++ x :: b = a' ++ x' :: b' → p x = true → p x' = true →
      a.countP p = a'.countP p → x = x'
  | [], [], _, _, _, _, h, _, _, _ => by
    simp only [List.nil_append, List.cons.injEq] at h
    exact h.1
  | [], y :: a', _, _, _, _, h, hx, _, hc => by
    simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
    obtain ⟨rfl, -⟩ := h
    simp [hx] at hc
  | y :: a, [], _, _, _, _, h, _, hx', hc => by
    simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
    obtain ⟨rfl, -⟩ := h
    simp [hx'] at hc
  | y :: a, y' :: a', _, _, _, _, h, hx, hx', hc => by
    simp only [List.cons_append, List.cons.injEq] at h
    obtain ⟨rfl, h⟩ := h
    rw [List.countP_cons, List.countP_cons] at hc
    exact eq_of_countP_eq h hx hx' (Nat.add_right_cancel hc)

/-- **Each round of a tree's stream has one line**: where `coinTreesOf` accepts the release log `lines`, two of its
lines opening round `k` of stream `name` of the coin tree whose root is `ρ` are one, each the line after the `k` lines
opening that stream (`coinTreesOf_round`). -/
theorem opensRound_unique {lines : List String} {trees : Array LoggedTree} (h : coinTreesOf lines = .ok trees)
    {ρ name l l' : String} {k : ℕ} (hl : l ∈ lines) (hl' : l' ∈ lines) (ho : OpensRound l ρ name k)
    (ho' : OpensRound l' ρ name k) : l = l' := by
  obtain ⟨a, b, he⟩ := List.append_of_mem hl
  obtain ⟨a', b', he'⟩ := List.append_of_mem hl'
  exact eq_of_countP_eq (he.symm.trans he') ho.opensLine ho'.opensLine
    ((coinTreesOf_round h he ho).symm.trans (coinTreesOf_round h he' ho'))

/-! ## The log before the link -/

/-- **A log whose first opening is none opens no round.** -/
theorem noOpening_of_firstOpened {L₀ : List Verity.Release} (h : firstOpened L₀ = none) :
    ∀ q ∈ L₀, ∀ i s n, q.frame ≠ .opening i s n := fun q hq i s n hq' => by
  have h' := List.findSome?_eq_none_iff.1 h q hq
  simp [hq'] at h'

/-- **A log that opens no round opens none of a stream.** -/
theorem openedOf_eq_zero {l : List Verity.Release} (h : ∀ q ∈ l, ∀ i s n, q.frame ≠ .opening i s n) (i s : ℕ) :
    openedOf l i s = 0 :=
  List.countP_eq_zero.2 fun q hq hp => by
    obtain ⟨n, hn⟩ := opening_of_opensAt hp
    exact h q hq i s n hn

/-- **The read v2's check shows of the device's log, at a release**: where the reads of `lr` are of the releases
`L₀ ++ r :: L`, `lr` reads `r` at its place, after reads of `L₀`'s frames, with `r`'s frame and the bytes `r` answered. -/
theorem logged_answers {r : Verity.Release} {L : List Verity.Release} :
    ∀ {L₀ lr : List Verity.Release}, List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (L₀ ++ r :: L) →
      ∃ pre y post, lr = pre ++ y :: post ∧ List.Forall₂ (fun a b => a.frame = b.frame) pre L₀ ∧
        y.frame = r.frame ∧ y.answers = r.answers
  | [], [], h => by simp at h
  | [], y :: post, h => by
    simp only [List.nil_append, List.map_cons, List.forall₂_cons] at h
    exact ⟨[], y, post, rfl, .nil, h.1.1, h.1.2.1⟩
  | _ :: _, [], h => by simp at h
  | q :: L₀, y' :: lr, h => by
    simp only [List.cons_append, List.map_cons, List.forall₂_cons] at h
    obtain ⟨pre, y, post, rfl, hfr, hy, ha⟩ := logged_answers h.2
    exact ⟨y' :: pre, y, post, rfl, .cons h.1.1 hfr, hy, ha⟩

/-! ## The bytes the link answers -/

/-- **The session v2 sets up has the session's tables**: wherever v2 sets `A` up, the session v2's check verified is
on the statement's one table. -/
theorem v2_tables {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {p : _root_.Spot.OneStageZk.Params} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} (h : V2Verified p e bs record spec tabs rest proofs) :
    spec.tables = A.tables := by
  obtain ⟨tags0, circuitB, pubB, sts, ht0, -, -, hs, -⟩ := v2Verified_accepts h
  obtain rfl : tags0 = A.I.tags := Option.some.inj (ht0.symm.trans hA.tags)
  unfold Exec.setupOfP at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs (Recursion.Witness.toExec
      (_root_.Spot.OneStageZk.inputsOf A.I.tags p.own circuitB pubB bs))).trans A.hTags.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · obtain ⟨_, hP⟩ := Hidden.setupHidden_specP hst
    show #[st.tags.table] = #[A.I.tags.table]
    rw [hP.tagsEq]
    exact congrArg (fun t => #[t]) (zkTags_table _)
  · exact absurd h1 (by show ¬ 1 < 1; omega)

/-- **The device's first opening answers the SHA-512 of the link's message in v2's record**: where `A` is the session
v2 sets up on the parameters, the sampled verdict accepts, the log v2's check read is the device's, and the device's
first opening `r`, after releases `L₀` that open none, opens the stream `TREE_SESSION` of the tree it opens, the record
v2 reads has a link's message `m` at the session's tables, and `r` answered `m`'s SHA-512. -/
theorem link_answers {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ) (hv : V.Verdict k .sampled x)
    (hl : Binding.Logged Binding.readsV2 x) {L₀ L : List Verity.Release} {r : Verity.Release} {i s n : ℕ}
    (hlog : x.log = L₀ ++ r :: L) (hr : r.frame = .opening i s n)
    (hL₀ : ∀ q ∈ L₀, ∀ i' s' n', q.frame ≠ .opening i' s' n')
    (hO : ((helloSpecs L₀)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION) :
    ∃ m, linkIn A.tables x = some m ∧ sha512Msg m = r.answers.data.toList := by
  subst hθ
  obtain ⟨p', hp', -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  obtain ⟨lr, hlr⟩ := Option.isSome_iff_exists.1 (logRead_isSome hv.2.1 hcase hacc)
  obtain ⟨record, _, _, _, _, hcs, hrec, -, hver⟩ := sent_verified hv.2.1 hcase hacc
  obtain ⟨J, -, hrs⟩ := sent_records hver
  obtain ⟨-, -, -, hξ, hdr, _, _, hck⟩ := hacc
  obtain ⟨key, hch, -⟩ := source_of hdr
  obtain ⟨record', spec, _, _, _, _, t, o, rb, _, hv', -, -, -, -, htr, ho, hlm, haft⟩ := checkBlobsAt_link hck
  obtain rfl : record' = record := by
    obtain ⟨_, _, _, _, _, _, _, _, _, _, h1, -⟩ := hv'
    obtain ⟨_, _, _, _, _, _, _, _, _, _, h2, -⟩ := hver
    exact Option.some.inj (h1.symm.trans h2)
  rw [v2_tables hA hv'] at hlm
  obtain ⟨_, _, htt⟩ := treesOf_ok htr
  obtain ⟨ls, lines, hls0, hls, hct⟩ := trustedTreesOf_ok htt
  have hread : readLog dr.label c.digest [J] ξ lines = some lr := by
    rw [← hlr, logRead_eq hcs hch hξ (linesOf_of hls0 hls), hrs, Option.bind_some]
  have hlog' := hl (lr.map Binding.Read.ofV2) (by
    show lr.map Binding.Read.ofV2 ∈ (logRead x.sent).map (·.map Binding.Read.ofV2)
    rw [hlr]
    exact Option.mem_def.2 rfl)
  rw [hlog] at hlog'
  obtain ⟨pre, y, post, rfl, hfr, hyr, hya⟩ := logged_answers hlog'
  have hpre := noOpening_frames hfr hL₀
  obtain ⟨l, ρ, name, _, hlm', hro, hkey, hname, rfl⟩ := readLog_round hct hread rfl (hyr.trans hr)
  rw [helloSpecs_frames hfr, hO] at hname
  obtain rfl : name = TREE_SESSION := (Option.some.inj hname).symm
  rw [openedOf_eq_zero hpre] at hro
  have hρ : ρ = t.root := by
    obtain ⟨-, sp, hg, -⟩ := hkey
    have hg' : (ρ, sp) ∈ #[t].toList.map fun t => (t.root, t.spec) := List.mem_of_getElem? hg
    obtain ⟨u, hu, he⟩ := List.mem_map.1 hg'
    have hut : u = t := by simpa using hu
    rw [hut] at he
    exact (Prod.mk.inj he).1.symm
  subst hρ
  unfold LoggedTree.opening at ho
  have hp := Array.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  obtain ⟨l', hl', hal, d', hev', haft', -⟩ := opened_after hct t (by simp) o (Array.mem_of_find?_eq_some ho)
  rw [hp.1, hp.2] at hal
  obtain rfl := opensRound_unique hct hlm' hl' hro.1 hal
  obtain ⟨-, -, d, hev, -, aft, haft'', hu⟩ := hro
  obtain rfl : d = d' := Except.ok.inj (hev.symm.trans hev')
  obtain rfl : aft = o.after := Json.str.inj (haft''.symm.trans haft')
  rw [haft, FlockComplete.JsonRT.unhex_hexOf] at hu
  refine ⟨(spec.sigma ++ rb).data.toList, ?_, ?_⟩
  · show (sentRecord x.sent).bind (linkMsg · A.tables) = _
    rw [hrec, Option.bind_some]
    exact hlm
  · rw [← hya, ← Except.ok.inj hu]

/-! ## The roots at the first stop -/

/-- **The roots decoded at the first stop are the record's `root_B`** (`MovesOf`'s clause (ii)): at the readings and
the decoders at the session's layout, on a trace from the first stop at the widened sample `ω`, any rewriting of its
tape consumed, off the misses, where the play's sampled verdict accepts, the log v2's check read is the device's, the
device's first opening is of the stream `TREE_SESSION` of the tree it opens, the first stop's fixed rounds pick a release
after a log `L₀` that opens none (`LoggedAfter`, the link's round) and its cuts are at the bytes that release answers
(`CutsAt`), the roots the decoders read at the first stop are the `root_B` of the session v2's record decodes to at the
session `--zk` sets up at `A`. -/
theorem firstStop_rootB {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A)
    {record₀ : Verity.Exchange → Option Claims.Msg} {recordDigest : Claims.Msg} {rd : Reading A.rounds}
    (hrd : rd = readingAt A.tables A.layout record₀ recordDigest) {reg : Option Claims.Msg → A.Reg}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {σ : Strategy (V.game θ F c)}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {B : RoundHold} {ω : (A.extend M.widening).L.Ω}
    {φ : CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s' →
      CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s'}
    {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {rs : Fin J → Fin 2 → List Flock.Round} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs ((firstStop V θ F c A rd σ M ω).rewrite φ).consume o)
    (hn : NoMiss V θ F c rd σ o.2) (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2))
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (hO : ∀ pre r post, (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: post → ∀ i s n,
      r.frame = .opening i s n → (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION)
    {L₀ : List Verity.Release} {bits : List Bool}
    (hLA : LoggedAfter rd.round (firstStop V θ F c A rd σ M ω) L₀ bits (CoinTape.run (V.game θ F c) σ o.2).log)
    (hc : CutsAt rd.reveal (firstStop V θ F c A rd σ M ω) L₀ (CoinTape.run (V.game θ F c) σ o.2).log)
    (hfo : firstOpened L₀ = none) {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    {sess : Flock.RunRecord} (hrec : sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some record)
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts))
    (hd : Flock.Record.decode spec record = .ok sess) :
    D.root (firstStop V θ F c A rd σ M ω).modes = sess.rootB := by
  subst hrd hD
  obtain ⟨j, i, s, n, b, L, hjK, hall, hj, hlog⟩ := hLA
  have hL₀ := noOpening_of_firstOpened hfo
  obtain ⟨m, hm, hmd⟩ := link_answers hA (play_ends V θ F c σ o.2).1 hv hl hlog rfl hL₀
    (hO _ _ _ hlog i s n rfl hL₀)
  have hlink : A.layout j = .link := by
    have h' : slotRound A.tables (A.layout j) L₀ (.opening i s n) := hj
    cases hLj : A.layout j with
    | link => rfl
    | rep _ _ _ =>
      rw [hLj] at h'
      obtain ⟨_, _, _, -, -, hfo', -⟩ := h'
      rw [hfo] at hfo'
      cases hfo'
  have hw : (readingAt A.tables A.layout record₀ recordDigest).reveal j (CoinTape.run (V.game θ F c) σ o.2) =
      some m := by
    show slotReveal A.tables (A.layout j) _ = some m
    rw [hlink]
    exact hm
  have hmode := RoundsReach.stop_modes_reveal
    (q := Pos.root (roundCuts V θ F c (readingAt A.tables A.layout record₀ recordDigest)) (M.start ω))
    (Pos.inv_root _ _) hr hn hc hlog hjK hw hmd
  obtain ⟨ht, hσ, -⟩ := hA.setup_one hs
  refine root_record reg A.layout (i := j) (by rw [hlink]; rfl) hmode
    (fun j' hj' _ => Pos.modes_eq_none (Bool.eq_false_iff.2 fun h =>
      hj' (readingAt_unique A.layout_injective A.tables_nodup record₀ recordDigest (hall j' h) hj))) hd ht hσ ?_
  rw [ht]
  have e : linkIn A.tables (CoinTape.run (V.game θ F c) σ o.2) = linkMsg record A.tables := by
    show (sentRecord _).bind _ = _
    rw [hrec, Option.bind_some]
  rw [← e]
  exact hm

end Verity.Sampled
