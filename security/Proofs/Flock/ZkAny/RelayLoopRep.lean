module

public import Proofs.Flock.ZkAny.RelayLoopFacts
public import Proofs.Flock.ZkAny.RelayLoopNext
public import Proofs.Flock.ZkAny.RelayLoopOwn

/-! At every frame round of the deployed loop the round's own rep has `RepJ` one round on (`repJ_round`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.ZkServer.Session (State Placed byte?)
open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.ZkServer.Frames (Sessions Rep Release)
open _root_.Flock.ZkServer.Frame (Cfg Masker Place Req)

theorem roundPlace_l0 {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (h : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    ∃ (s : Release) (name : String) (l0 : _root_.Flock.ZkServer.Level0.Level0),
      R.sessions.find? (·.sid == sid) = some s ∧ s.spec.tables[i]? = some name ∧
        _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0 := by
  unfold _root_.Flock.ZkServer.Frames.roundPlace at h
  rcases hf : R.find sid i r with _ | ⟨j, p⟩
  · simp only [hf] at h <;> cases h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  rcases hs : R.sessions.find? (·.sid == sid) with _ | s
  · simp only [hs] at h <;> cases h
  simp only [hs] at h
  rcases ht : s.sts[i]? with _ | stm
  · simp only [ht] at h <;> cases h
  simp only [ht] at h
  rcases hn : s.spec.tables[i]? with _ | name
  · simp only [hn] at h <;> cases h
  simp only [hn] at h
  rcases hl : _root_.Flock.ZkServer.Level0.ofSession L0 sid name with _ | l0
  · simp only [hl] at h <;> cases h
  refine ⟨s, name, l0, ?_, ?_, ?_⟩ <;> first | rfl | assumption

open Loop LeanRelease in
theorem repJ_round (T : ZKMal.CoinLayer.TableData)
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {sid : Nat} {stream : String} {n : Nat}
    (hD : LoopModel.Deployed M L₀ real s0 spec sts) (hR : Count.RealTabs sts (real M.x M.w M.ρ))
    (hreal : ∀ σ : LoopAt, SessOn spec sts σ.R sid → BuiltAt M spec sts σ.L0 sid →
      ∀ (i r : Nat) (rel : Release) (name : String) (l0 : _root_.Flock.ZkServer.Level0.Level0) (stm : Stmt)
        (t : _root_.Flock.ZkServer.Finish.Tab), r < 2 → sessAt σ.R sid = some rel →
        rel.spec.tables[i]? = some name → _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 →
        rel.sts[i]? = some stm →
        _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
        ∃ hi, real M.x M.w M.ρ (2 * i + r) = .ok (t, hi)) :
    ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat)
      (dom : ByteArray) (v : Claims.Msg), slot st pl = some (.round stream n) → e.setup = .ok (spec, sts) →
      LoopFacts sid spec seen st pl σ → LoopJ M sid spec sts seen st σ → M.Ans σ.c e →
      M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → σ'.c = σ.c + 1 →
      Stays sid σ σ' → RoundFacts M sid spec seen stream n i r dom v σ → ∀ o a, e.round = .ok o →
      e.ans = some a → byte? a 0 = some 2 → (st.answer (.round stream n) a).1.stop = none → e.coins = .ok () →
      RepJ M sid spec sts i r σ' ∧ M.L.fed M.x M.ρ (M.hist σ'.c) = true := by
  intro seen st pl e σ σ' i r dom v hsl _ hF hJ hans _ hr hpc hcc hSt hRA o a hro hae hb hsa hco
  have hlt := hans.1
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  have hreq := request_round_slot hF.reach hF.live hsl
  obtain ⟨hkr, hmem⟩ := round_rep hsl hF hJ hr.rep_eq hlt hD.names
  have hc2 : 2 ≤ σ.c := by
    rw [hF.count hsent]
    omega
  have hl : 2 ≤ (M.hist σ.c).length := by
    rw [LoopModel.hist, length_histOf_take, Nat.min_eq_left (Nat.le_of_lt hlt), hF.count hsent]
    omega
  obtain ⟨m', hh'⟩ := M.hist_ans hans hae hb
  rw [← hcc] at hh'
  obtain ⟨e0, h1, hh⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hc : M.hist σ.c with
    | nil =>
      rw [hc] at hl
      simp at hl
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  obtain ⟨q₀, hq⟩ : ∃ q₀, h1.length = q₀ + 1 := by
    rw [hh, List.length_cons] at hl
    exact ⟨h1.length - 1, by omega⟩
  have hn : loopNr M.x h1.length = n := by
    have h := hRA.coins
    rw [hD.rel, hh, List.length_cons] at h
    exact h
  subst hn
  obtain ⟨m₀, q, f, nd, hA₀, hrp, -, -, hu₀, -⟩ := hRA.place
  have hB := hJ.built hc2
  have hSz := hJ.sizes hsent
  have hK2 : ∀ rel : Release, sessAt σ.R sid = some rel → rel.spec = spec ∧ rel.sts = sts ∧
      rel.link = some (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0) := by
    intro rel hrs
    obtain ⟨hsp, hsts⟩ := (hJ.sess hsent).eq hrs
    obtain ⟨-, name, l0, -, -, hl0⟩ := roundPlace_l0 hrp
    exact ⟨hsp, hsts, hJ.link hc2 name l0 hl0 rel hrs⟩
  have hK3 : ∀ (rel : Release) (name : String) (l0 : _root_.Flock.ZkServer.Level0.Level0) (stm : Stmt)
      (t : _root_.Flock.ZkServer.Finish.Tab), sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
      ∃ hr, real M.x M.w M.ρ (2 * i + r) = .ok (t, hr) :=
    fun rel name l0 stm t => hreal σ (hJ.sess hsent) hB i r rel name l0 stm t (rep_lt_two hmem)
  have hS : ∀ (k : Nat) (link : Array F128) (c' : Cfg) (t : _root_.Flock.ZkServer.Finish.Tab) (dom : ByteArray)
      (xs : Array (Array F128)) (hi : Array ByteArray), k < spec.repStreams.size →
      link.size ≤ _root_.Flock.ZkServer.Frames.COIN_MAX → placeholder spec sts k link = .ok (c', t, dom, xs, hi) →
      dom.size < 2 ^ 64 ∧ xs.size + 2 < 2 ^ 64 := by
    intro k link c' t dom xs hi hk hlk hph
    obtain ⟨-, hd, hx2, -⟩ := Loop.sizesOk_at hSz hk
    obtain ⟨-, -, -, -, hdm, hxe⟩ := Loop.placeholder_ok hph
    subst hdm hxe
    refine ⟨hd, ?_⟩
    rw [Loop.extras_size]
    have := Nat.mul_le_mul_left sts[k / 2]!.regions.size ((Loop.linkPoints_size_le sts[k / 2]! link).trans hlk)
    omega
  have hP : ∀ k, k < spec.repStreams.size → sts[k / 2]!.mPts = spec.mPts ∧ 0 < spec.mPts ∧
      spec.points * spec.mPts ≤ _root_.Flock.ZkServer.Frames.COIN_MAX :=
    fun k hk => (Loop.sizesOk_at hSz hk).1
  have hRow : ∀ (k : Nat) (link : Array F128) (c' : Cfg) (t : _root_.Flock.ZkServer.Finish.Tab) (dom : ByteArray)
      (xs : Array (Array F128)) (hi : Array ByteArray), k < spec.repStreams.size →
      placeholder spec sts k link = .ok (c', t, dom, xs, hi) → c'.rowH.size < 2 ^ 64 := by
    intro k link c' t dom xs hi hk hph
    obtain ⟨lay, mask, hs, hc, -, -⟩ := Loop.placeholder_ok hph
    subst hc
    rw [Loop.ofSamples_rowH_size]
    exact (Loop.sizesOk_at hSz hk).2.2.2 lay mask hs
  rcases Nat.eq_zero_or_pos (nextAt spec sts (M.hist σ.c)).2 with h0 | hpos
  · have hfr := freshAt_cur hsl hF hJ hr.rep_eq hlt hD.names h0
    have hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, 0) := Prod.ext hkr.symm h0
    have hF1 := repJ_same_first T L₀ real hD.rel hD.setup hS hR hP hmem hh hq hk hA₀ hRA.last hK2 hK3 hB
      hD.names hfr hRA.frame hpc hr hreq hrp hu₀ hro hae hb hsa hco hh' hSt
    by_cases hsame : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 = 2 * i + r
    · exact ⟨hF1.2 hsame, hF1.1⟩
    · exact ⟨repJ_next_first T L₀ real hD.rel hD.setup hmem hh hq hk hsame hA₀ hK2 hB hD.names hfr hRA.frame hr
        hreq hrp hro hae hb hsa hco hh' hSt, hF1.1⟩
  · have hJ₀ := repJ_cur hsl hF hJ hr.rep_eq hlt hD.names hpos
    obtain ⟨j₀, hj₀⟩ : ∃ j₀, (nextAt spec sts (M.hist σ.c)).2 = j₀ + 1 :=
      ⟨(nextAt spec sts (M.hist σ.c)).2 - 1, by omega⟩
    have hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, j₀ + 1) := Prod.ext hkr.symm hj₀
    have hF1 : M.L.fed M.x M.ρ (M.hist σ'.c) = true ∧
        ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 = 2 * i + r → RepJ M sid spec sts i r σ') := by
      by_cases hout : ∃ c' t' dom' xs' hi' sched', placeholder spec sts (2 * i + r)
          (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0) = .ok (c', t', dom', xs', hi') ∧
          schedSoFar t' r dom' xs' hi' ((Array.range (j₀ + 1)).map fun k =>
            wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2)
              ((M.hist σ.c).length - 2 - (j₀ + 1) + k + 1)) = .ok sched' ∧ sched'.size ≤ j₀ + 1
      · exact repJ_own T L₀ real hD.rel hD.setup hS hR hRow hmem hh hq hk hA₀ hK2 hK3 hout hr hreq hrp hro
          hae hb hsa hco hh' hSt hJ₀
      · exact repJ_same T L₀ real hD.rel hD.setup hS hR hP hmem hh hq hk hA₀ hRA.last hK2 hK3
          (fun c' t' dom' xs' hi' sched' h₁ h₂ => (Nat.lt_or_ge (j₀ + 1) sched'.size).resolve_right fun h₃ =>
            hout ⟨c', t', dom', xs', hi', sched', h₁, h₂, h₃⟩) hr hreq hrp hro hae hb hsa hco hh' hSt hJ₀
    by_cases hsame : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 = 2 * i + r
    · exact ⟨hF1.2 hsame, hF1.1⟩
    · exact ⟨repJ_next hmem hh hq hk hsame hr hreq hrp hro hae hb hsa hco hh' hSt hJ₀, hF1.1⟩

end Claims.ZKServer.Relay
