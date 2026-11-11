module

public import Proofs.Flock.ZkAny.RelayLoopFirst

/-!
# The frame round's own rep one round on, from a stream of one round

`loopKept_round`'s `hcur` at a round that is its stream's first, where the next round is another stream's:

* **`repJ_next_first`**: the frame the session's next request finds holds the history's words for the stream's one
  round (`roundRun_last_first`, `count_first`), the readied frame being the rep's run from its start and the table the
  one its level 0 gives, as at `repJ_same_first`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Masker Place Req)

open Loop LeanRelease in
/-- **The round's own rep one round on, from its stream's first round, where the next round is another stream's**:
at the loop's own frames (`hrel`), the session's entry being the setup's with the `Commit`'s words as its link
(`hK2`), and every level 0 bound to the session the one the loop drew for its table (`BuiltAt`), the rep streams'
names distinct (`hnd`), where the round is the first of table `i`'s rep `r`'s stream (`hk`) and the next round
another stream's (`hne`), the rep's frame fresh (`FreshAt`): the frame the session's next request finds is the rep's,
on the session's entry, table and level 0 as they stay, holding the history's words for the stream's one round where
the next round is the next stream's first (`roundRun_last_first`, `count_first`), the frame `roundPlace` readied
being the rep's run from its start (`repRun_start` at rep 0, `roundPlace_prior` at rep 1) and the table the one the
rep's level 0 gives (`tab_of_placeholder`, `cap_of_drawn`); the rep's salts are not read, the algebraic phase having a
round past the first's coins. -/
theorem repJ_next_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)}
    (hrel : M.L = ofLoop T L₀ real) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dom : ByteArray} {v : Claims.Msg} {e : Env}
    {σ σ' : LoopAt} {e0 : Claims.Msg × Claims.Msg} {h1 : List (Claims.Msg × Claims.Msg)} {q₀ : Nat}
    (hmem : (i, r, stream) ∈ spec.repStreams.toList) (hh : M.hist σ.c = e0 :: h1) (hq : h1.length = q₀ + 1)
    (hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, 0))
    (hne : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 ≠ 2 * i + r) {m₀ : Masker}
    (hA₀ : M.L.answered M.x M.ρ (M.hist σ.c) = .ok m₀)
    (hK2 : ∀ rel : Release, sessAt σ.R sid = some rel → rel.spec = spec ∧ rel.sts = sts ∧
      rel.link = some (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0))
    (hB : BuiltAt M spec sts σ.L0 sid) (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hfr : FreshAt σ.R sid i r) (hfs : (frameAt σ.R sid i r).isSome)
    (hr : RoundRun sid st pl stream (loopNr M.x h1.length) i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hreq : st.request pl (.round stream (loopNr M.x h1.length)) = none) {q : Rep} {pl' : Place} {f : Bool}
    {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ (loopNr M.x h1.length) =
      .ok (q, pl', f, nd))
    {o : ByteArray} (ho : e.round = .ok o) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr M.x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {m' : Claims.Msg} (hh' : M.hist σ'.c = M.hist σ.c ++ [(m', a.data.toList)]) (hSt : Stays sid σ σ') :
    RepJ M sid spec sts i r σ' := by
  have hlen : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := congrArg Prod.fst hk
    unfold nextAt at this
    rwa [hlen] at this
  have hk2 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = 0 := by
    have := congrArg Prod.snd hk
    unfold nextAt at this
    rwa [hlen] at this
  have hr2 := rep_lt_two hmem
  have hdiv : (2 * i + r) / 2 = i := by omega
  obtain ⟨p, hfa⟩ := Option.isSome_iff_exists.mp hfs
  have hp : FreshRep p := hfr p hfa
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hf)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hf)
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  obtain ⟨rel, name, l0, stm, p', hss, hstm', hn, hl0, hw⟩ := roundPlace_first_ok hf hrp hfst
  have hsa : sessAt σ.R sid = some rel := hss
  obtain ⟨hsp, hsts, hlk⟩ := hK2 rel hsa
  rw [hh, List.tail_cons] at hlk
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  have hRq : ∃ pre : List Req, RepRun pre [] q := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · obtain ⟨p'', pl₀, n₀, hpo, -, hp''⟩ := roundPlace_first_of hf hrp hfst hss hn hl0 hw
      obtain ⟨-, -, hk2'⟩ := placeOf_run hpo
      obtain ⟨-, -, hk1'⟩ := withAlg_run hw
      have hcq : q.coins = #[] := by
        rw [hk2']
        rcases hp'' with rfl | ⟨st₁, rfl⟩
        · rw [hk1', hp.coins]
        · show p'.coins = #[]
          rw [hk1', hp.coins]
      exact ⟨[], repRun_start (by rw [roundPlace_st hf hrp (by decide), hp.st]) hcq⟩
    · obtain ⟨j0, p0, h0⟩ := roundPlace_rep0 hf hrp hfst
      exact ⟨_, (roundPlace_prior hf hrp (fresh_repRun hp) hss hn hl0 hw h0).2⟩
  obtain ⟨pre, hRq⟩ := hRq
  -- the model's side
  have hA₁ : (ofLoop T L₀ real).answered M.x M.ρ (e0 :: h1) = .ok m₀ := by
    rw [← hrel, ← hh]
    exact hA₀
  have hsw : 2 ≤ h1.length → repOf M.x (h1.length + 1) ≠ repOf M.x h1.length := by
    intro h2 he
    have e1 : repOf M.x (h1.length + 1) = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 := by
      rw [show h1.length + 1 = q₀ + 2 by omega]
      exact repOf_add_two hx q₀
    have e2 : repOf M.x h1.length = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ - 1)).1 := by
      rw [show h1.length = q₀ - 1 + 2 by omega]
      exact repOf_add_two hx (q₀ - 1)
    rw [e1, e2] at he
    have h' := Loop.locate_succ (roundsOf spec sts) 0 (q₀ - 1)
      (by rw [show q₀ - 1 + 1 = q₀ by omega]; exact he)
    rw [show q₀ - 1 + 1 = q₀ by omega, hk2] at h'
    omega
  have hm := ofLoop_answered_switch T L₀ real M.x M.ρ e0 h1 hsw hA₁
  have hcs1 : (h1.map fun y => ZKMal.CoinLayer.rd y.2).length = q₀ + 1 := by rw [List.length_map, hq]
  obtain ⟨c', t', dom', xs', hip, -, hph, -, -⟩ := Loop.freshOf_ok M.x _ hx hcs1 hm
  have hph' : placeholder spec sts (2 * i + r) (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0) =
      .ok (c', t', dom', xs', hip) := by
    rw [← hk1]
    exact hph
  have hname : spec.tables[i]! = name := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hn
    rw [← hsp, getElem!_pos rel.spec.tables i hlt, he]
  have hstmb : sts[i]! = stm := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hstm'
    rw [← hsts, getElem!_pos rel.sts i hlt, he]
  obtain ⟨jb, gb, db, _, hjb, -, hdb⟩ := hB name l0 hl0
  rw [tables_inj (j := i) hnd hjb (by rw [← hsp]; exact hn), hstmb] at hdb
  obtain ⟨t, htab⟩ : ∃ t, _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t := by
    have h := tab_of_placeholder (g := gb) (d := db) (l := l0) hph' (by rw [hdiv, hstmb]; exact hdb)
    rwa [hdiv, hstmb] at h
  have hcap : _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root := cap_of_drawn hdb htab
  -- the frame's schedule on no coins, the algebraic phase `withAlg` scheduled
  obtain ⟨xs, hxsd⟩ : ∃ xs, xs = _root_.Flock.Firewall.Drive.extras stm
      (_root_.Flock.Firewall.Drive.linkPoints stm (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0)) :=
    ⟨_, rfl⟩
  have hI' := repInv_withAlg hp hw hst' htab hcap xs #[]
  have hpos : 0 < p'.alg.size := Loop.rounds_pos hI'.alg
  obtain ⟨-, hrw, hcw0, -, -, -⟩ := withAlg_keep hw
  have hsc : schedSoFar t p.rep (rel.spec.domain name r).toUTF8 xs #[] p.coins = .ok p'.alg := by
    have e1 := Loop.schedSoFar_alg t _ xs #[] p' hI'.alg (by rw [hcw0, hp.coins, Array.size_empty]; exact hpos)
    rw [hrw, hcw0, hpr] at e1
    rw [hpr]
    exact e1
  have ho' := (hr.round hreq).symm.trans ho
  obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
  have hlink : rel.link = some (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0) := hlk
  obtain ⟨p₃, hat, -, hcw, -⟩ := roundRun_last_first M.x e0 h1 v hq hr hreq hrp ho' hput ha hb hs hco hRq hfa hp hss
    hn hl0 hw hst' hlink (r0 := ByteArray.empty) (q0 := 0) (hi := #[]) (fun h0 => absurd h0 (by omega)) htab hcap
    hxsd hsc
  refine ⟨p₃, rel, name, l0, hat, ?_, hn, ?_, ?_⟩
  · rw [sessAt_stay (hSt.2 name l0 hl0) (by rw [hsa]; rfl)]
    exact hsa
  · rw [hSt.1 name]
    exact hl0
  · have hl' : (M.hist σ'.c).length - 2 = q₀ + 1 := by
      rw [hh', hh, List.length_append, List.length_cons, hq, List.length_singleton]
      omega
    unfold StreamAt
    rw [hl']
    refine ⟨fun he => absurd he hne, fun he _ => ?_⟩
    have hnext : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 =
        (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 + 1 := by rw [he, hk1]
    have hcnt := count_first hk2 hnext
    rw [he, Nat.add_sub_cancel] at hcnt
    rw [hcw, hcnt, hh', hh, List.cons_append, List.tail_cons]
    simp only [List.map_append, List.map_cons, List.map_nil]

end Claims.ZKServer.Relay
