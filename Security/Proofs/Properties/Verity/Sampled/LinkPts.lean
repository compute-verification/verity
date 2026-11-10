module

public import Proofs.Properties.Verity.Sampled.Layout
public import Proofs.Properties.Verity.Sampled.ParseBridge
public import Proofs.Properties.Verity.Sampled.LinkTrees
public import Proofs.Properties.Verity.Sampled.LinkRead

/-!
# The link points the device's log shows are the record's

`MovesOf` asks that the link points written at the link (`linkBits fieldEnc pts`, the coins of the device's log's
first opening) be those of the session v2's record decodes to. Where the sampled verdict accepts, the log v2's check
read is the device's (`Binding.Logged Binding.readsV2`) and the device's first opening is of the stream
`Spot.Challenger.TREE_SESSION` of the tree it opens (the order `SampleLinkFirst` states), they are
(`link_pts`):

* **v2's read of that opening** (`link_read`): v2's read of the release log shows it as the read of the log's first
  line opening a round, which `coinTreesOf` takes only as round 0 of a tree the log shows sampled
  (`LinkTrees.first_round`); the log shows one tree, the record's (`LinkTrees.treesHeld_size`, `Trees.records_held`),
  so the opening's coins are those the record gives the session stream's round 0, its link points.
* **The read of the device's log** (`logged_split`): the read v2's check shows of a release, in order, has its frame,
  and its coins wherever it shows any.
* **The link points' coins are some** (`SetsUp.mPts_pos`, `link_bits_ne`): the session `--zk` sets up at `A` has
  points of `PT_LOCAL + nbl` coordinates, so the record's link coins are 128 bits a coordinate, never none.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Spot.Challenger
open FlockSoundness.Discharge
open FlockSoundness.Refine (ptsOf)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The link points' coins are some -/

/-- **The session `--zk` sets up at `A` has points of some coordinates**, `PT_LOCAL + nbl` of them, wherever v2 sets
`A` up. -/
theorem SetsUp.mPts_pos {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) : 0 < spec.mPts := by
  have hc : A.I.count = 1 := hA.count
  unfold Exec.setupOfP at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · obtain ⟨_, hP⟩ := Hidden.setupHidden_specP hst
    show 0 < st.mPts
    rw [hP.mPts]
    have h24 : Flock.PT_LOCAL = 24 := rfl
    omega
  · exact absurd h1 (by show ¬ 1 < A.I.count; omega)

/-- **The link's coins a decoded record gives are some bits**, at two points of some coordinates. -/
theorem link_bits_ne {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {J : Json}
    {b : ByteArray} (hd : Flock.Record.decode spec record = .ok sess) (hJ : Flock.JsonParse.parse record = .ok J)
    (hp2 : spec.points = 2) (hpos : 0 < spec.mPts) (hb : recordRound J TREE_SESSION 0 = some b) :
    Verity.bitsOfBytes b ≠ [] := by
  obtain ⟨cs, hrr, hf⟩ := link_coins hd hJ hp2
  rw [hrr] at hb
  have hpts := FlockSoundness.Refine.decode_points hd
  have h0 : 0 < sess.points.size := by have := hpts.1; omega
  have hsz : sess.points[0]!.size = spec.mPts :=
    hpts.2 _ (by rw [getElem!_pos sess.points 0 h0]; exact Array.getElem_mem_toList h0)
  intro he
  have hl := congrArg List.length ((bits_of_coins hf hb).symm.trans he)
  rw [length_chunks fieldBits, List.length_nil, Array.length_toList, Array.size_append, hsz] at hl
  omega

/-! ## The read of the device's log -/

/-- **The read v2's check shows of the device's log, at a release**: where the reads of `lr` are of the releases
`L₀ ++ r :: L`, `lr` reads `r` at its place, after reads of `L₀`'s frames, with `r`'s frame, and with `r`'s coins
wherever it shows any. -/
theorem logged_split {r : Verity.Release} {L : List Verity.Release} :
    ∀ {L₀ lr : List Verity.Release}, List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (L₀ ++ r :: L) →
      ∃ pre y post, lr = pre ++ y :: post ∧ List.Forall₂ (fun a b => a.frame = b.frame) pre L₀ ∧
        y.frame = r.frame ∧ (y.coins ≠ [] → y.coins = r.coins)
  | [], [], h => by simp at h
  | [], y :: post, h => by
    simp only [List.nil_append, List.map_cons, List.forall₂_cons] at h
    refine ⟨[], y, post, rfl, .nil, h.1.1, fun hne => ?_⟩
    have hie : y.coins.isEmpty = false := by
      cases hc : y.coins with
      | nil => exact absurd hc hne
      | cons _ _ => rfl
    exact h.1.2.2 _ (by simp [Binding.Read.ofV2, hie])
  | _ :: _, [], h => by simp at h
  | q :: L₀, y' :: lr, h => by
    simp only [List.cons_append, List.map_cons, List.forall₂_cons] at h
    obtain ⟨pre, y, post, rfl, hfr, hy, hc⟩ := logged_split h.2
    exact ⟨y' :: pre, y, post, rfl, .cons h.1.1 hfr, hy, hc⟩

/-- Releases of the same frames show the same coin trees' specs sampled. -/
theorem helloSpecs_frames :
    ∀ {l l' : List Verity.Release}, List.Forall₂ (fun a b => a.frame = b.frame) l l' → helloSpecs l = helloSpecs l'
  | _, _, .nil => rfl
  | _ :: _, _ :: _, .cons h t => by
    have e := helloSpecs_frames t
    simp only [helloSpecs, List.filterMap_cons, h] at e ⊢
    rw [e]

/-- Releases of the frames of releases that open no round open none. -/
theorem noOpening_frames :
    ∀ {l l' : List Verity.Release}, List.Forall₂ (fun a b => a.frame = b.frame) l l' →
      (∀ q ∈ l', ∀ i s n, q.frame ≠ .opening i s n) → ∀ q ∈ l, ∀ i s n, q.frame ≠ .opening i s n
  | _, _, .nil, _, _, hq => absurd hq List.not_mem_nil
  | _, _, .cons h t, hl', q, hq => by
    rcases List.mem_cons.1 hq with rfl | hq
    · intro i s n
      rw [h]
      exact hl' _ (List.mem_cons_self ..) i s n
    · exact noOpening_frames t (fun q hq => hl' q (List.mem_cons_of_mem _ hq)) q hq

/-! ## v2's read of the link's opening -/

/-- **v2's read of the release log's first opening shows the record's link points**: where v2 reads the release log
`lines`, with the one record `J`, as `pre ++ y :: post`, `coinTreesOf` reads the log to one tree, which `J` holds to
(`sessionTreeReason`), `y` opens a round and no release of `pre` does, and `y` opens the stream `TREE_SESSION` of the
tree it opens (by the specs `pre` shows sampled), `y`'s coins are the bits of the coins `J` gives the session stream's
round 0. -/
theorem link_read {label : String} {after : ByteArray} {J : Json} {ξ : ByteArray} {lines : List String}
    {pre post : List Verity.Release} {y : Verity.Release} {trees : Array LoggedTree} {i s n : ℕ}
    (hR : readLog label after [J] ξ lines = some (pre ++ y :: post)) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) {tabs : Array (String × Option ℕ)} (hJ : sessionTreeReason trees tabs J = none)
    (hpre : ∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') (hy : y.frame = .opening i s n)
    (hO : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION) :
    ∃ b, recordRound J TREE_SESSION 0 = some b ∧ y.coins = Verity.bitsOfBytes b := by
  obtain ⟨l₀, l, l₁, dj, ts₁, ts₂, hls, hno, htl, hev, hT, hst, hS⟩ := LinkRead.readLog_first hR hpre hy
  obtain ⟨root, stream, k, i', s', n', hroot, -, hk, hfr, -, hsp, hcoins⟩ := LinkRead.step_open hst hT
  rw [hy] at hfr
  obtain ⟨rfl, rfl, -⟩ := Verity.Frame.opening.inj hfr
  obtain ⟨t, ht, hts⟩ := Option.bind_eq_some_iff.1 hsp
  obtain ⟨sp, hsp', hspec⟩ := LinkRead.forall₂_getElem?' hS ht
  have hstr := LinkRead.specOf_streams' hspec hts
  have hO' : sp.streams[s]? = some TREE_SESSION := by
    rw [hsp'] at hO
    exact hO
  obtain rfl : stream = TREE_SESSION := Option.some.inj (hstr.symm.trans hO')
  obtain ⟨hk0, root', hroot', t₁, ht₁, htr₁⟩ := LinkTrees.first_round hct hls hno htl hev hT
  obtain rfl : k = 0 := Option.some.inj (hk.symm.trans hk0)
  have hrr : root' = root := Json.str.inj (hroot'.symm.trans hroot)
  obtain ⟨b, hb⟩ := LinkTrees.link_some hJ
  obtain ⟨t₂, ht₂, htr₂, -⟩ := Trees.round_held hJ hb
  have hrt : recordTree J = root := by
    rw [htr₂, LinkTrees.eq_of_size_one h1 ht₂ ht₁, htr₁, hrr]
  refine ⟨b, hb, ?_⟩
  rw [hcoins]
  simp [hrt, hb]

/-! ## The link points -/

/-- **The link points the device's log shows are the record's** (`MovesOf`'s link points): where `A` is the session
v2 sets up on the parameters, the sampled verdict accepts, the log v2's check read is the device's, and the device's
first opening `r`, after releases `L₀` that open none, opens the stream `TREE_SESSION` of the tree it opens with coins
`linkBits fieldEnc pts`, the session that v2's record decodes to at the session `--zk` sets up at `A` has the link
points `pts`. -/
theorem link_pts {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ) (hv : V.Verdict k .sampled x)
    (hl : Binding.Logged Binding.readsV2 x) {L₀ L : List Verity.Release} {r : Verity.Release} {i s n : ℕ}
    (hlog : x.log = L₀ ++ r :: L) (hr : r.frame = .opening i s n)
    (hL₀ : ∀ q ∈ L₀, ∀ i' s' n', q.frame ≠ .opening i' s' n')
    (hO : ((helloSpecs L₀)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION) {m : ℕ}
    {pts : Fin 2 → Fin m → 𝔽} (hc : r.coins = linkBits fieldEnc pts) {record : String} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} {sess : Flock.RunRecord} (hrec : sentRecord x.sent = some record)
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts))
    (hd : Flock.Record.decode spec record = .ok sess) : ptsOf m sess.points = pts := by
  subst hθ
  obtain ⟨p', hp', -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  obtain ⟨lr, hlr⟩ := Option.isSome_iff_exists.1 (logRead_isSome hv.2.1 hcase hacc)
  obtain ⟨record', _, _, _, _, hcs, hrec', -, hver⟩ := sent_verified hv.2.1 hcase hacc
  have hrr : record' = record := Option.some.inj (hrec'.symm.trans hrec)
  rw [hrr] at hver
  obtain ⟨J, hJ, hrs⟩ := sent_records hver
  obtain ⟨rs, _, _, _, _, he, -, -, -, -, hsr, -⟩ := hver
  obtain ⟨-, -, -, hξ, hdr, _, _, hck⟩ := hacc
  obtain ⟨key, hch, -⟩ := source_of hdr
  obtain ⟨ts, htr, hth⟩ := trees_of_checkBlobsAt hck
  obtain ⟨_, _, htt⟩ := treesOf_ok htr
  obtain ⟨ls, lines, hls0, hls, hct⟩ := trustedTreesOf_ok htt
  have hlines := linesOf_of hls0 hls
  have hread : readLog dr.label c.digest [J] ξ lines = some lr := by
    rw [← hlr, logRead_eq hcs hch hξ hlines, hrs, Option.bind_some]
  have hlog' := hl (lr.map Binding.Read.ofV2) (by
    show lr.map Binding.Read.ofV2 ∈ (logRead x.sent).map (·.map Binding.Read.ofV2)
    rw [hlr]
    exact Option.mem_def.2 rfl)
  rw [hlog] at hlog'
  obtain ⟨pre, y, post, rfl, hfr, hyr, hcy⟩ := logged_split hlog'
  have h1 : ts.size = 1 := by
    rw [LinkTrees.treesHeld_size hth, he]
    try rfl
  obtain ⟨gb, hgb, hgu⟩ := sessionRecord_blob he hsr
  have hgb' : ((_root_.Core.Blobs.ofArray x.sent.blobs).get rs.record s!"session {rs.name}'s record").toOption =
      some gb := by
    rw [hgb]
    try rfl
  have hJp : (Flock.JsonParse.parse record).toOption.getD .null = J := by
    rw [hJ]
    try rfl
  obtain ⟨tabs, hJt⟩ := Trees.records_held hth (r := rs) (by simp [he]) hgb' hgu hJp
  obtain ⟨bb, hbb, hyc⟩ := link_read hread hct h1 hJt (noOpening_frames hfr hL₀) (hyr.trans hr)
    (by rw [helloSpecs_frames hfr]; exact hO)
  have hp2 := (hA.setup_one hs).2.2.2.1
  have hne : y.coins ≠ [] := by
    rw [hyc]
    exact link_bits_ne hd hJ hp2 (hA.mPts_pos hs) hbb
  exact pts_of_record hd hJ hp2 hbb (hyc.symm.trans ((hcy hne).trans hc))

end Verity.Sampled
