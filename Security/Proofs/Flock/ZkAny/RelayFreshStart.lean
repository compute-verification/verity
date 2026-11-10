module

public import Proofs.Flock.ZkAny.RelayLoopFacts
public import Proofs.Flock.SessionLoop.FinRepRows

/-!
# The next rep's start at a stream's first round

`answered_of_round`'s `hfr` at a frame round's slot: where `seen` has no round of the stream, the model's next-rep state
after the history so far (`Loop.freshOf`) is a masking state. The loop readied the round's frame (`roundPlace`), so the
table's statement, name, level 0 and algebraic phase are there (`roundPlace_first_ok`), and the model's placeholder for
the stream is one wherever the rep's frame starts at the table's statement (`Drive.start`):

* `ofDraws_of_sizes`, `tab_of_drawn`: a level 0 is sampled at `m` from a packed table and samples of the sizes `m`'s
  schedule takes, and one sampled at a statement's `m` gives a table where the statement commits under `hm96Sha512`.
* `placeholder_of`: so the placeholder is one where the rep's frame starts and its zero level 0 gives a table.
* `freshOf_of`: `Loop.freshOf` is a masking state where the placeholder and its algebraic phase are, rep 0's
  algebraic-phase coins being in at rep 1 and the frame's step taking them there (`Frame.step`'s `prior`).
* `repRun_prior_ok`: a frame's run on an earlier rep's coins took them (`step_prior_ok`).
* **`fresh_of_slot`**: at a stream's first round of the deployed loop on a `Live` statement, where the rep's frame
  starts at the table's statement, the model's next-rep state is a masking state. The placeholder's table is of the
  shape of the table the loop's level 0 gives (`Sched.tabOfLevel0_alike`), so its algebraic phase is one as the loop's
  is (`Sched.rounds_alike`); at rep 1 rep 0's frame holds the phase's coins (`roundPlace_prior_size`), which the
  readied frame took (`roundPlace_prior`), as the placeholder's configuration takes them (`step_prior_alike`).
* **`freshStart_deployed`**: so `FreshStart` holds at the deployed loop on a `Live` statement: where the round goes
  out the loop readied its frame (`roundPlace_of_out`), and the rep's frame starts at its table's statement
  (`StartOk`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed)
open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Masker Place)

/-- **Level 0's padded codeword has a row for each tree position.** -/
theorem linkRows_size (lv : _root_.Flock.Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) :
    (_root_.Flock.Firewall.Link.rows lv tPad g extra lanePad).size = 2 ^ lv.d := by
  unfold _root_.Flock.Firewall.Link.rows
  split
  · simp [_root_.Flock.Firewall.Link.rowsFft]
  · simp [_root_.Flock.Firewall.Link.rowsDirect]

/-- **A built level 0 has a row for each tree position.** -/
theorem buildRows_size (m : Nat) (lv : _root_.Flock.Level) (tPad : Nat) (g : Array F128)
    (extra pads : Array (Array F128)) (salts : ByteArray) :
    (_root_.Flock.Firewall.Level0.build m lv tPad g extra pads salts).rows.size = 2 ^ lv.d := by
  exact linkRows_size _ _ _ _ _

/-- **A level 0 is sampled at `m`** from a packed table and samples of the sizes `m`'s `--zk` schedule takes. -/
theorem ofDraws_of_sizes {m : Nat} {s : _root_.Flock.Schedule} {tp : Nat}
    (hz : _root_.Flock.Zk.schedule m = .ok (s, tp)) {g : Array F128} {d : ByteArray}
    (hg : g.size = 2 ^ ((s.levels.getD 0 default).lanes + (s.levels.getD 0 default).cols))
    (hd : d.size = _root_.Flock.Firewall.Level0.drawSize (s.levels.getD 0 default) tp) :
    ∃ l, _root_.Flock.Firewall.Level0.ofDraws m g d = .ok l := by
  unfold _root_.Flock.Firewall.Level0.ofDraws
  simp only [hz, hg, hd, buildRows_size, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, bind, Except.bind,
    pure, Except.pure]
  all_goals exact ⟨_, rfl⟩

/-- **A level 0 sampled at a statement's `m` gives a table** where the statement commits under `hm96Sha512`. -/
theorem tab_of_drawn {stm : Stmt} (hmk : stm.merkle = _root_.Flock.MerkleScheme.hm96Sha512) {g : Array F128}
    {d : ByteArray} {l : _root_.Flock.Firewall.Level0.Level0}
    (hl : _root_.Flock.Firewall.Level0.ofDraws stm.m g d = .ok l) :
    ∃ t, _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l = .ok t := by
  obtain ⟨hm, -, ⟨s, hz, -⟩, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl
  have e1 : (_root_.Flock.Setup.ofCircuit stm).merkle = _root_.Flock.MerkleScheme.hm96Sha512 := hmk
  have e2 : (_root_.Flock.Setup.ofCircuit stm).m = stm.m := rfl
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0
  simp only [e1, e2, hm, hz, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure, Except.pure]
  all_goals exact ⟨_, rfl⟩

/-- Where the left of two related results is a value, the right is one too, related to it. -/
theorem relV_ok_left {α β : Type} {S : α → β → Prop} {x : V α} {y : V β} {a : α} (h : Sched.RelV S x y)
    (hx : x = .ok a) : ∃ b, y = .ok b ∧ S a b := by
  subst hx
  cases y with
  | ok b => exact ⟨b, rfl, h⟩
  | error e => exact False.elim h

/-- **Rep stream `k`'s placeholder is one** where the rep's frame starts at the stream's table's statement and the
statement's zero level 0 gives a table. -/
theorem placeholder_of {spec : SessionSpec} {sts : Array Stmt} {k : Nat} (link : Array F128) {stm : Stmt}
    (hk : sts[k / 2]! = stm) {lay : _root_.Flock.Zk.PadLayout} {mask : _root_.Flock.Firewall.Frame.MaskSlot}
    (hst : _root_.Flock.Firewall.Drive.start stm = .ok (lay, mask)) {sch : _root_.Flock.Schedule} {tp : Nat}
    (hz : _root_.Flock.Zk.schedule stm.m = .ok (sch, tp)) {l : _root_.Flock.Firewall.Level0.Level0}
    (hl : _root_.Flock.Firewall.Level0.ofDraws stm.m
      (Array.replicate (2 ^ ((sch.levels.getD 0 default).lanes + (sch.levels.getD 0 default).cols)) 0)
      (Loop.zeros (_root_.Flock.Firewall.Level0.drawSize (sch.levels.getD 0 default) tp)) = .ok l)
    {t : _root_.Flock.Firewall.Finish.Tab}
    (ht : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l = .ok t) :
    ∃ c' dom xs hi, Loop.placeholder spec sts k link = .ok (c', t, dom, xs, hi) := by
  unfold Loop.placeholder
  simp only [hk, hst, hz, hl, ht, bind, Except.bind, pure, Except.pure]
  all_goals exact ⟨_, _, _, _, rfl⟩

/-- **The loop's frame state at a rep's first round is a masking state** where the rep stream's placeholder and its
algebraic phase are, and at rep 1 rep 0's algebraic phase ended and the placeholder's frame takes its coins. -/
theorem freshOf_of (x : Msg) (cs : List FlockDeploy.Coin) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    {sts : Array Stmt} {q : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : Loop.placeholder spec sts (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 (Loop.wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi)) {alg : Array (ByteArray × Nat)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (h1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 % 2 ≠ 0 → ∃ n0,
      (Loop.roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 - 1) 0 = n0 ∧
      alg.size ≤ n0 ∧ ∃ st', _root_.Flock.Firewall.Frame.step c' {}
        (.prior (((Array.range n0).map fun j => Loop.wordsAt x cs (q - n0 + j + 1)).extract 0 alg.size
            |>.foldl (· ++ ·) #[]).size
          (_root_.Flock.Firewall.Frame.wordBytes
            (((Array.range n0).map fun j => Loop.wordsAt x cs (q - n0 + j + 1)).extract 0 alg.size
              |>.foldl (· ++ ·) #[]))) = (.ok, st')) :
    ∃ m, Loop.freshOf x cs = .ok m := by
  unfold Loop.freshOf
  rw [hx, hq]
  simp only
  rw [hph]
  simp only
  rw [hal]
  simp only
  by_cases h0 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 % 2 = 0
  · rw [ite_eq_left h0]
    exact ⟨_, rfl⟩
  · obtain ⟨n0, hn0, hle, st', hst⟩ := h1 h0
    rw [ite_eq_right h0, hn0, ite_eq_right (Nat.not_lt.mpr hle), hst]
    exact ⟨_, rfl⟩

/-- **The frame's step on an earlier rep's coins** that does not end the rep takes them. -/
theorem step_prior_ok {c : Cfg} {st : _root_.Flock.Firewall.Frame.State} {k : Nat} {b : ByteArray}
    (h : (_root_.Flock.Firewall.Frame.step c st (.prior k b)).1.ends = false) :
    ∃ st', _root_.Flock.Firewall.Frame.step c st (.prior k b) = (.ok, st') := by
  rcases hs : _root_.Flock.Firewall.Frame.step c st (.prior k b) with ⟨a, st'⟩
  have h' : a.ends = false := by
    rw [hs] at h
    exact h
  have ha : a = .ok ∨ a.ends = true := by
    simp only [_root_.Flock.Firewall.Frame.step] at hs
    split at hs
    · exact .inr (by cases hs; rfl)
    · split at hs
      · exact .inl (by cases hs; rfl)
      · exact .inr (by cases hs; rfl)
  rcases ha with rfl | ha
  · exact ⟨st', rfl⟩
  · rw [ha] at h'
    cases h'

/-- **A frame's run on an earlier rep's coins took them**: the run did not end, so its step on them answered `ok`. -/
theorem repRun_prior_ok {k : Nat} {b : ByteArray} {q : Rep} (h : RepRun [.prior k b] [] q) :
    ∃ st', _root_.Flock.Firewall.Frame.step q.cfg {} (.prior k b) = (.ok, st') := by
  have h1 : _root_.Flock.Firewall.Frame.run q.cfg {} [.prior k b] = (q.st, false) := h.1
  rw [run_cons] at h1
  apply step_prior_ok
  cases he : (_root_.Flock.Firewall.Frame.step q.cfg {} (.prior k b)).1.ends with
  | false => rfl
  | true =>
    rw [he] at h1
    simp at h1

set_option maxHeartbeats 1000000 in
/-- **The model's next-rep state at a stream's first round is a masking state**: at the slot of a round of `stream`,
table `i`'s rep `r`, where `LoopFacts` and `LoopJ` hold at the deployed loop on a `Live` statement, `seen` has no round
of `stream`, the rep's frame is at the model's configuration and starts at the table's statement (`hsk`), and
`roundPlace` readies the round, `Loop.freshOf` after the history so far is a masking state. The rep's frame is fresh
(`FreshAt`); `roundPlace` read the table's statement and level 0, which the loop drew (`BuiltAt`), and `withAlg`'s
algebraic phase on it (`withAlg_alg`). The statement commits under `hm96Sha512` (`Loop.tabs_of_live`), so the
placeholder's zero level 0 gives a table (`tab_of_drawn`), of the loop's table's shape (`Sched.tabOfLevel0_alike`),
and the placeholder is one (`placeholder_of`) whose algebraic phase is one as `withAlg`'s (`Sched.rounds_alike`). At
rep 1 rep 0's frame holds as many coins as the phase has rounds (`roundPlace_prior_size`), the history's words for
rep 0's rounds (rep 0's stream seen, its `RepJ`), and the readied frame took them (`roundPlace_prior`,
`repRun_prior_ok`), at the placeholder's configuration too (`step_prior_alike`, `ofLoop_cfgAlike`). -/
theorem fresh_of_slot {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    (hlive : ZKMal.Masked.Live M.x)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = false)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c))
    (hsk : ∀ p, frameAt σ.R sid i r = some p → ∃ lm, _root_.Flock.Firewall.Drive.start sts[i]! = .ok lm)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    ∃ m, M.L.fresh M.x (M.hist σ.c) = .ok m := by
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  obtain ⟨e0, h1, hh⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hc : M.hist σ.c with
    | nil =>
      rw [hc, List.length_nil] at hlen
      omega
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  obtain ⟨q₀, hq⟩ : ∃ q₀, h1.length = q₀ + 1 := by
    refine ⟨opened st - 1, ?_⟩
    rw [hh, List.length_cons] at hlen
    omega
  have hl2 : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := hk0.symm
    unfold nextAt at this
    rwa [hl2] at this
  have h20 : (nextAt spec sts (M.hist σ.c)).2 = 0 := by
    rcases Nat.eq_zero_or_pos (nextAt spec sts (M.hist σ.c)).2 with h | h
    · exact h
    · exact absurd (hsn.symm.trans ((hJ.seenAt i r stream hmem).2 hk0 h)) Bool.false_ne_true
  have hk3 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = 0 := by
    have := h20
    unfold nextAt at this
    rwa [hl2] at this
  have hr2 := rep_lt_two hmem
  have hdiv : (2 * i + r) / 2 = i := by omega
  have hmod : (2 * i + r) % 2 = r := by omega
  obtain ⟨_, p, hfd⟩ := roundPlace_find hrp
  have hfa : frameAt σ.R sid i r = some p := by
    rw [← find_frameAt, hfd]
    rfl
  have hp : FreshRep p := fresh_found (hF.fresh i r stream hmem hsn) hfd
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hfd)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hfd)
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  obtain ⟨rel, name, l0, stm, p', hss, hstm', hn, hl0, hw⟩ := roundPlace_first_ok hfd hrp hfst
  have hsa : sessAt σ.R sid = some rel := hss
  obtain ⟨hspec, hsts⟩ := (hJ.sess hsent).eq hsa
  have hc2 : 2 ≤ σ.c := by
    rw [hF.count hsent]
    omega
  obtain ⟨hli, hle⟩ := Array.getElem?_eq_some_iff.mp hstm'
  have hstmb : sts[i]! = stm := by
    rw [← hsts, getElem!_pos rel.sts i hli, hle]
  have hist : i < sts.size := by
    rw [← hsts]
    exact hli
  have hname : spec.tables[i]! = name := by
    obtain ⟨hni, hne⟩ := Array.getElem?_eq_some_iff.mp hn
    rw [← hspec, getElem!_pos rel.spec.tables i hni, hne]
  obtain ⟨jb, gb, db, _, hjb, -, hdb⟩ := hJ.built hc2 name l0 hl0
  rw [tables_inj (j := i) hD.names hjb (by rw [← hspec]; exact hn), hstmb] at hdb
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  obtain ⟨sch, tp, hsch, ha⟩ := withAlg_alg hw (fresh_alg hp) hst'
  obtain ⟨-, -, hmk⟩ := Loop.tabs_of_live hlive hD.setup hist
  rw [hstmb] at hmk
  obtain ⟨⟨lay, mask⟩, hds⟩ := hsk p hfa
  rw [hstmb] at hds
  obtain ⟨l1, hl1⟩ := ofDraws_of_sizes hsch
    (g := Array.replicate (2 ^ ((sch.levels.getD 0 default).lanes + (sch.levels.getD 0 default).cols)) 0)
    (d := Loop.zeros (_root_.Flock.Firewall.Level0.drawSize (sch.levels.getD 0 default) tp)) (by simp)
    (Loop.zeros_size _)
  obtain ⟨t1, ht1⟩ := tab_of_drawn hmk hl1
  obtain ⟨t0, ht0⟩ := tab_of_drawn hmk hdb
  have hk2 : sts[(2 * i + r) / 2]! = stm := by
    rw [hdiv]
    exact hstmb
  obtain ⟨c', dom', xs', hi', hph⟩ := placeholder_of (spec := spec) (sts := sts) (k := 2 * i + r)
    (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0) hk2 hds hsch hl1 ht1
  have hph' : Loop.placeholder spec sts (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1
      (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0) = .ok (c', t1, dom', xs', hi') := by
    rw [hk1]
    exact hph
  have hTA : Sched.TabAlike t0 t1 := by
    obtain ⟨s', tp', hs', hg'⟩ := Count.ofDraws_gsize hdb
    have hss' : s' = sch := (Prod.mk.inj (Except.ok.inj (hs'.symm.trans hsch))).1
    obtain ⟨-, -, ⟨s0', hz0, hlv0⟩, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hdb
    have hd0 : 64 * 2 ^ l0.lv.d < 2 ^ 64 := by
      rw [hlv0]
      exact (Count.sched_bounds hz0).2 0
    have hgs : gb.size = (Array.replicate
        (2 ^ ((sch.levels.getD 0 default).lanes + (sch.levels.getD 0 default).cols)) (0 : F128)).size := by
      rw [hg', hss']
      simp
    have hA := Sched.tabOfLevel0_alike (_root_.Flock.Setup.ofCircuit stm) hdb hl1 hgs hd0
    rw [ht0, ht1] at hA
    exact hA
  have hcap0 := cap_of_drawn hdb ht0
  have hc : l0.root.size = (_root_.Flock.Firewall.Finish.flat t1.cm0.cap).size := by
    rw [← hcap0]
    exact hTA.2.2.2.2.1
  have hcn : l0.root.size < 2 ^ 64 := by
    rw [← hcap0]
    exact hTA.2.2.2.2.2
  have hdg : (_root_.Flock.Setup.ofCircuit stm).digest.size < 2 ^ 64 := by
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : (_root_.Flock.Setup.ofCircuit stm).m < 2 ^ 64 := by
    have e7 : (_root_.Flock.Setup.ofCircuit stm).m = stm.m := rfl
    rw [e7]
    exact Nat.lt_trans (Count.sched_facts hsch).1 (by decide)
  have hdm : (rel.spec.domain name p.rep).toUTF8 =
      (spec.domain (spec.tables[(2 * i + r) / 2]!) ((2 * i + r) % 2)).toUTF8 := by
    rw [hspec, hpr, hdiv, hmod, hname]
  have hdom := (Loop.sizesOk_at (hJ.sizes hsent) (rep_lt_size hmem)).2.1
  have hRA := Sched.rounds_alike (_root_.Flock.Setup.ofCircuit stm) sch.k0 l0.root
    (_root_.Flock.Firewall.Finish.flat t1.cm0.cap) _ hc hdg hcn hdom hmb
  rw [hdm] at ha
  obtain ⟨alg, hal0, -⟩ := relV_ok_left hRA ha
  obtain ⟨ht1st, hz1⟩ := Count.tabOfLevel0_sched ht1
  have ht1sch : t1.sch = sch := (Prod.mk.inj (Except.ok.inj (hz1.symm.trans hsch))).1
  obtain ⟨-, -, -, -, hdom', -⟩ := Loop.placeholder_ok hph
  have hal : _root_.Flock.Firewall.Skeleton.rounds t1.st t1.sch.k0 (_root_.Flock.Firewall.Finish.flat t1.cm0.cap)
      dom' = .ok alg := by
    rw [ht1st, ht1sch, hdom']
    exact hal0
  have hcs : (h1.map fun e => ZKMal.CoinLayer.rd e.2).length = q₀ + 1 := by rw [List.length_map, hq]
  rw [hD.rel, hh]
  show ∃ m, Loop.freshOf M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) = .ok m
  refine freshOf_of M.x _ hD.setup hcs hph' hal fun h0 => ?_
  have hr1 : r = 1 := by
    rw [hk1] at h0
    omega
  subst hr1
  obtain ⟨_, p0, h0'⟩ := roundPlace_rep0 hfd hrp hfst
  have hsz : p'.alg.size = alg.size := withAlg_size hw (fresh_alg hp) hst' hph hk2 hal
  have hpos : 0 < p'.alg.size := by
    rw [hsz]
    exact Loop.rounds_pos hal
  have hmem0 : (i, 0, s!"{spec.tables[i]!}/rep{(0 : Nat)}") ∈ spec.repStreams.toList := by
    have hs := rep_lt_size hmem
    rw [FlockDeploy.repStreams_size] at hs
    rw [FlockDeploy.repStreams_toList]
    refine List.mem_map.mpr ⟨2 * i, List.mem_range.mpr (by omega), ?_⟩
    show (2 * i / 2, 2 * i % 2, s!"{spec.tables[2 * i / 2]!}/rep{2 * i % 2}") = _
    rw [Nat.mul_div_cancel_left i (by decide : 0 < 2), Nat.mul_mod_right]
  have hpsz := roundPlace_prior_size hfd hrp hfst hss hn hl0 hw h0'
  have hsn0 : seen.contains s!"{spec.tables[i]!}/rep{(0 : Nat)}" = true := by
    cases hc : seen.contains s!"{spec.tables[i]!}/rep{(0 : Nat)}" with
    | true => rfl
    | false =>
      rw [(fresh_found (hF.fresh i 0 _ hmem0 hc) h0').coins, Array.size_empty] at hpsz
      exact absurd hpsz (by omega)
  obtain ⟨p0', -, -, -, hfa0, -, -, -, -, hS0⟩ := hJ.reps i 0 _ hmem0 hsn0
  have hp0 : p0' = p0 := by
    have e : frameAt σ.R sid i 0 = some p0 := by
      rw [← find_frameAt, h0']
      rfl
    exact Option.some.inj (hfa0.symm.trans e)
  have hco := hS0 (by rw [hl2]; omega) (by rw [hl2]; omega)
  rw [hp0, hl2, hh, List.tail_cons, Nat.add_zero] at hco
  have hq2 : (e0 :: h1).length = q₀ + 2 := by rw [List.length_cons, hq]
  have hcp : CfgAlike p.cfg c' := by
    rw [hpc p hfa, hD.rel, hh]
    exact ofLoop_cfgAlike T L₀ real M.x M.ρ (e0 :: h1) hD.setup hq2 hph'
  obtain ⟨hcq, hrun⟩ := roundPlace_prior hfd hrp (fresh_repRun hp) hss hn hl0 hw h0'
  have hcq' : CfgAlike q.cfg c' := by
    rw [hcq]
    exact hcp
  obtain ⟨st₁, hst₁⟩ := repRun_prior_ok hrun
  rw [step_prior_alike hcq', hco, hsz] at hst₁
  refine ⟨(Loop.roundsOf spec sts).getD (2 * i) 0, by rw [hk1, Nat.add_sub_cancel], ?_, st₁, hst₁⟩
  have hn0 : p0.coins.size = (Loop.roundsOf spec sts).getD (2 * i) 0 := by
    first
      | rw [hco, Array.size_map, Array.size_range]
      | (rw [hco]; simp)
  rw [← hsz, ← hn0]
  exact hpsz

/-- **`FreshStart` at the deployed loop on a `Live` statement**: at a frame round's slot whose round goes out, the loop
readied the round's frame (`roundPlace_of_out`), and where the rep's frame starts at its table's statement
(`StartOk`), `fresh_of_slot` gives the model's next-rep state, a masking state. -/
theorem freshStart_deployed {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    (hlive : ZKMal.Masked.Live M.x) {sid : Nat} {stream : String} {n : Nat} :
    FreshStart M sid spec sts stream n := by
  intro seen st pl e σ σ' i r dom v hsl hF hJ hlt _ hr hpc _ _ hsk ho hsn
  obtain ⟨o, hro⟩ := ho
  obtain ⟨q, pc, f, nd, hrp⟩ := roundPlace_of_out
    ((hr.round (request_round_slot hF.reach hF.live hsl)).symm.trans hro)
  exact fresh_of_slot hD hlive hsl hF hJ hr.rep_eq hlt hsn hpc hsk hrp

end Claims.ZKServer.Relay
