import Proofs.Flock.C3.Reps.RepsDrawnAt
import Proofs.Flock.C3.VStarList.Range
import Proofs.Flock.Recursive.HoloSchedCheck

/-!
# RepsDrawn (ii) at the verifier's map, from the covers check's rounds

`coins_of_checkCommitRounds`: the verifier's round-for-round check (`checkCommitRounds`' first loop) holds every stream
but the sparse phases' and R's to the inner shape's schedule (`holoRepRounds` at a holo shape, `repRounds` else), so a
rep's held rounds are the schedule's round for round (`heldRounds_of_check`). `repIdx_of_check` reads RepsDrawn (ii)'s
count off it at a rep unit's values, whose rounds are the held stream's (`UnitProof.coins`): the frames' `z_partial`
round draws no coin (`HoloLcAnswered.zp_n`, `zp_of_rounds`), and every round of the plain schedule draws one
(`repRounds_coins`), so no rep unit passes the check at a plain shape; at a holo shape `z_partial`'s is the schedule's
one round without coins (`holoRounds_coins`), so the frames' `z_partial` round is the schedule's, and the record is
consumed whole (`UnitFrames.whole`), so `t₄.k` is the schedule's count (`repIdx_of_size`). No hypothesis ties the inner
shape's `m` and `k_log` to the statement's: the check pins the frames' rounds to the schedule's. A rep's stream is held
where custody's coins half holds one of its rounds' coins (`stream_of_custody`, `streams_of_custody`), as the map draws
rep `i`'s first coin round (`holoSel_first`).

`repsDrawn_holoMap` is `RepsDrawn` at `holoMap`'s draws from those, with the check at every accepted outer proof
(`hchk`) and rep names past the sparse phases' and R's (`hn`) as hypotheses; `repsDrawn_holoMapAt` is it at the
guarantee's coin types and `m := holoMap`. `hchk` is the covers check's last step (`checkCommitRounds_of_checkCovers`),
so it holds where the covers check passes at the held record at every accepted outer proof (`hchk_of_held`), as
`AcceptsC3` at the history would give it (`hchk_of_acceptsC3`). The checked sessions pass it there (`covers_checked`:
`ChecksC3`'s covers conjunct, which the verifier's run gives through `AcceptsC3`, `covers_of_acceptsC3`), so
`repsDrawn_checked` is `repsDrawn_holoMapAt` at the checked sessions and the guarantee's hold, with `hchk` and the
whole history theirs, `hn` the guarantee's `_htbl`, the schedule read off `holoStreams` (`holoStreams_eq`) and the coin
count's room off the reps' forms at the class's public statement (`hA`).
-/

namespace FlockC3.Reps

open FlockSoundness.Assumptions (InnerRecordMap heldCoins)

section check

open Flock Flock.RecVStar Flock.Guarantees
open Flock.Guarantees.HoloSchedCheck (checkBind_ok checkThrow_ne forIn_each)

/-- **The round-for-round check**: where `checkCommitRounds` passes, every held stream but the sparse phases' and R's
draws the schedule's coins round for round. -/
theorem coins_of_checkCommitRounds {i : RecVStar.Inner} {cs : InnerCommits} {rep : Array SchedRound}
    (hsch : (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims = some rep)
    (h : RecVStar.checkCommitRounds i cs = .ok ()) :
    ∀ st ∈ cs.streams, (st.name.startsWith s!"{SPARSE}/" || st.name.startsWith s!"{STACKED}/") = false →
      st.coins.map (·.size) = rep.map (·.1) := by
  unfold RecVStar.checkCommitRounds at h
  rw [hsch] at h
  try simp only at h
  obtain ⟨_, hloop, -⟩ := checkBind_ok h
  have hall := forIn_each (fun st : InnerStream =>
      (st.name.startsWith s!"{SPARSE}/" || st.name.startsWith s!"{STACKED}/") = false →
        st.coins.map (·.size) = rep.map (·.1)) (fun a r hb => by
      try simp only at hb
      split at hb
      · exact absurd hb (checkThrow_ne _ _)
      · rename_i hne
        refine ⟨fun hn => ?_, (Except.ok.inj hb).symm⟩
        rw [Bool.or_eq_false_iff] at hn
        rw [hn.1, hn.2] at hne
        simpa using hne) cs.streams.toList (Array.forIn_toList.trans hloop)
  exact fun st hst => hall st (Array.mem_toList_iff.2 hst)

/-- The plain schedule's `z_partial` round draws one coin, where the hidden-unit one's draws none. -/
theorem repRounds_zp {m kLog claims : ℕ} {rs : Array SchedRound} (h : holoRepRounds m kLog claims = some rs) :
    ∃ out, repRounds m kLog claims = some out ∧ out[zpRound m kLog]? = some ((1, #[], 0, true) : SchedRound) := by
  unfold holoRepRounds at h
  obtain ⟨out, hout, h⟩ := HoloRepCoins.obind_some h
  refine ⟨out, hout, ?_⟩
  try dsimp only at h
  split at h
  · cases h
  · rename_i hcond
    simpa [zpRound] using hcond

/-- **A rep stream's held rounds are the schedule's round for round** under the check. -/
theorem heldRounds_of_check {i : RecVStar.Inner} {cs : InnerCommits} {rep : Array SchedRound} {s : String}
    {st : InnerStream} (hsch : (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims = some rep)
    (h : RecVStar.checkCommitRounds i cs = .ok ()) (hs : cs.stream? s = some st)
    (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false) :
    (heldRounds (.ok cs) s).map (·.size) = rep.map (·.1) := by
  have hs' : cs.streams.find? (·.name == s) = some st := hs
  have hname : st.name = s := by
    first
    | exact beq_iff_eq.1 (Array.find?_some hs')
    | simpa using Array.find?_some hs'
  show (((cs.stream? s).map (·.coins)).getD #[]).map (·.size) = _
  rw [hs]
  exact coins_of_checkCommitRounds hsch h st (Array.mem_of_find?_eq_some hs') (by rw [hname]; exact hn)

/-- **Custody holds the stream a drawn round reads**: where round `i` of the history draws round `q` of stream `s`
(`m.selC`), the verifier holds the record and `s` in it. -/
theorem stream_of_custody {Cm Coin : Type} {wr : List (Cm × Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {sparse : Option RecSparse.Shape}
    {m : InnerRecordMap Cm Coin} (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m)
    {t : List (Cm × Coin)} {i : ℕ} {e : Cm × Coin} (he : t[i]? = some e) {s : String} {q : ℕ}
    (hq : m.selC i s q) : ∃ cs x, held (wr t) = .ok cs ∧ cs.stream? s = some x := by
  have hc := ((hcus t).1 i e he).2 s q hq
  revert hc
  cases held (wr t) with
  | error _ =>
    first
    | exact fun hc => Option.noConfusion hc
    | exact fun hc => by simp [heldCoins] at hc
  | ok cs =>
    intro hc
    cases hx : cs.stream? s with
    | none =>
      have hc' : (cs.stream? s).bind (·.coins[q]?) = some _ := hc
      rw [hx] at hc'
      cases hc'
    | some x => exact ⟨cs, x, rfl, hx⟩

theorem coinRounds_zero {m kLog claims : ℕ} {rs : Array SchedRound} (hc : 0 < claims)
    (hrs : holoRepRounds m kLog claims = some rs) : (coinRounds rs)[0]? = some 0 := by
  have h22 := HoloRepCoins.holoRepRounds_m hrs
  have hz : 0 < zpRound m kLog := by unfold zpRound; omega
  rw [HoloRepCoins.coinRounds_holoRep_split hc hrs,
    List.getElem?_append_left (by rw [List.length_range']; exact hz), List.getElem?_range' hz]

/-- **`holoMap` draws rep `i`'s first coin round** below `r`, where the history has room for both reps' rounds. -/
theorem holoSel_first {st : Flock.Stmt} {hp : FlockSoundness.Discharge.Recursion.HoloParams} {Cm : Type}
    {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → ℕ → List UInt8} {rs : Array SchedRound}
    {Ls : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: Ls))
    (hcl : 0 < ish.claims) (hrs : holoRepRounds ish.m ish.kLog ish.claims = some rs) {r : ℕ}
    (hr : 1 + 2 * (coinRounds rs).length ≤ r) (i : Fin 2) :
    ∃ j q, j < r ∧ (holoMap st hp Cm ish tbl bc).selC j ((HoloStream.rep i).name tbl) q := by
  have h0 := coinRounds_zero hcl hrs
  have hpos := (List.getElem?_eq_some_iff.1 h0).1
  refine ⟨repStart (coinRounds rs).length i + 0, 0, ?_,
    FlockC3.Compose.selC_of_holoSelOne (holoSelOne_coinRound hS i h0)⟩
  unfold repStart
  split <;> omega

/-- **A rep's consumed rounds are the hidden-unit schedule's coin rounds** where its transcript, from round 0, has its
`z_partial` round at the schedule's and consumed as many rounds as the schedule has. -/
theorem repIdx_of_size {S₀ : FlockSoundness.Model.Statement} {v : UnitVals} {m kLog claims : ℕ}
    {rs : Array SchedRound} (hc : 0 < claims) (hrs : holoRepRounds m kLog claims = some rs) (hk0 : v.t.k = 0)
    (hz : v.t.k + 3 + (v.st.m - 6) + 2 + (S₀.kLog - 6) = zpRound m kLog) (hk : v.t₄.k = rs.size) :
    repIdx S₀ v = coinRounds rs := by
  unfold repIdx
  rw [hz, hk0, Nat.sub_zero, hk]
  exact (HoloRepCoins.coinRounds_holoRep_split hc hrs).symm

/-- **The frames' `z_partial` round is a round of the schedule without coins** where the record's rounds draw the
schedule's coins round for round: the frames' draws none. -/
theorem zp_of_rounds {sh : Flock.Shape} {d : FlockVBridge.Compiled.RepData} {v : UnitVals} {rep : Array SchedRound}
    (e : (v.t.rounds.map (·.coins)).map (·.size) = rep.map (·.1)) (hf : UnitFrames sh d v) :
    ∃ x, rep[v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6)]? = some x ∧ x.1 = 0 := by
  have hlt : v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) < v.t.rounds.size := hf.lc.zp_lt
  have hn0 : (v.t.rounds[v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6)]!).n = 0 := hf.lc.zp_n
  have hwf := hf.wf _ hlt
  rw [getElem!_pos v.t.rounds _ hlt] at hn0 hwf
  have e' := congrArg (fun a : Array ℕ => a[v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6)]?) e
  cases hx : rep[v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6)]? with
  | none => simp [Array.getElem?_map, Array.getElem?_eq_getElem hlt, hx] at e'
  | some x =>
    simp only [Array.getElem?_map, Array.getElem?_eq_getElem hlt, hx, Option.map_some, Option.some.injEq] at e'
    exact ⟨x, rfl, by omega⟩

/-- **RepsDrawn (ii)'s count at a rep unit under the check**: its rounds are its held stream's (`hd`, `hp`), which the
check holds to the inner shape's schedule, round for round. Every round of the plain schedule draws a coin and the
frames' `z_partial` round none, so at a plain shape no rep unit passes the check. At a holo shape the schedule's one
round without coins is its `z_partial` round, so the frames' is at it, and the record is consumed whole, so the unit
consumed the schedule's count. -/
theorem repIdx_of_check {S₀ : FlockSoundness.Model.Statement} {sh : Flock.Shape}
    {d : FlockVBridge.Compiled.RepData} {v : UnitVals} {ish : RecVStar.Inner} {rs : Array SchedRound}
    {cs : InnerCommits} {s : String} {st : InnerStream}
    (hcl : 0 < ish.claims) (hrs : holoRepRounds ish.m ish.kLog ish.claims = some rs)
    (hfk : v.st.fold.kLog = S₀.kLog) (hchk : RecVStar.checkCommitRounds ish cs = .ok ())
    (hs : cs.stream? s = some st) (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false)
    (hd : d.rounds = heldRounds (.ok cs) s) (hp : UnitProof sh d v) (hf : UnitFrames sh d v) :
    repIdx S₀ v = coinRounds rs := by
  have hc : ∀ rep, (if ish.holo then holoRepRounds else repRounds) ish.m ish.kLog ish.claims = some rep →
      (v.t.rounds.map (·.coins)).map (·.size) = rep.map (·.1) := fun _ hsch => by
    rw [← hp.coins, hd]
    exact heldRounds_of_check hsch hchk hs hn
  cases hh : ish.holo with
  | true =>
    have e := hc rs (by simp only [hh, ↓reduceIte]; exact hrs)
    obtain ⟨x, hx, hx0⟩ := zp_of_rounds e hf
    obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.1 hx
    have hzq := (HoloRepCoins.holoRounds_coins hcl ((HoloRepCoins.holoRepRounds_eq _ _ _).symm.trans hrs)).2 _ hlt
    rw [Array.getD_eq_getD_getElem?, hx, Option.getD_some] at hzq
    have hj : v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) = zpRound ish.m ish.kLog :=
      Classical.byContradiction fun hne => absurd (hzq.2 hne) (by omega)
    have hsz := congrArg Array.size e
    simp only [Array.size_map] at hsz
    exact repIdx_of_size hcl hrs hf.k0 (by omega) (hf.whole.trans hsz)
  | false =>
    obtain ⟨out, hout, -⟩ := repRounds_zp hrs
    obtain ⟨x, hx, hx0⟩ := zp_of_rounds (hc out (by simp only [hh, Bool.false_eq_true, ↓reduceIte]; exact hout)) hf
    exact absurd (HoloRepCoins.repRounds_coins hcl hout x (Array.mem_of_getElem? hx)) (by omega)

/-- **The covers check ends in the round-for-round check**: its refusals of the sparse phases' and R's presence and of
R's shape pass, and then `checkCommitRounds` does. -/
theorem checkCommitRounds_of_checkCovers {i : RecVStar.Inner} {cs : InnerCommits}
    (h : RecVStar.checkCovers i cs = .ok ()) : RecVStar.checkCommitRounds i cs = .ok () := by
  unfold RecVStar.checkCovers at h
  simp only at h
  repeat
    split at h
    · cases h
  exact h

end check

section atRep

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
  {sh : Flock.Shape}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String}
  {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
  {s : Fin 2 → String}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The reps' streams are held** at every accepted outer proof, where the map draws a round of each below `r` and
custody holds the history's rounds' coins there. -/
theorem streams_of_custody {m : InnerRecordMap Cm (ℐ).Coin}
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m) (hleaf : HLeafC3 zs dg N k Rw)
    (hsel : ∀ i, ∃ j q, j < r ∧ m.selC j (s i) q) :
    AtRep zs dg N k Rw sh rep fun _ _ t _ i => ∃ cs x, held (wr t) = .ok cs ∧ cs.stream? (s i) = some x := by
  intro R ω t R₂ τ o hA i _ _
  obtain ⟨j, q, hj, hq⟩ := hsel i
  have hlen : r + 1 ≤ t.length := hleaf R ω t R₂ τ o hA
  exact stream_of_custody hcus (List.getElem?_eq_getElem (by omega : j < t.length)) hq

/-- **RepsDrawn (ii)'s count at every accepted outer proof**: `repIdx_of_check` at the values `vals`, whose circuit
fold's `k_log` is the public statement's (`SetupOfVals`), with the rep's rounds its held stream's (`hrd`, `withReads`'
by definition), the check passing at the held record (`hchk`) and the rep's stream held (`hst`). -/
theorem repIdx_atRep {ish : Flock.RecVStar.Inner} {rs : Array Flock.RecVStar.SchedRound} (hcl : 0 < ish.claims)
    (hrs : Flock.RecVStar.holoRepRounds ish.m ish.kLog ish.claims = some rs)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals) (hsv : SetupOfVals zs dg N k Rw src sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hrd : ∀ R ω t Lb i, (rep R ω t Lb i).rounds = heldRounds (held (wr t)) (s i))
    (hn : ∀ i, ((s i).startsWith s!"{Flock.RecVStar.SPARSE}/" ||
      (s i).startsWith s!"{Flock.RecVStar.STACKED}/") = false)
    (hchk : AtRep zs dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ cs, held (wr t) = .ok cs → Flock.RecVStar.checkCommitRounds ish cs = .ok ())
    (hst : AtRep zs dg N k Rw sh rep fun _ _ t _ i => ∃ cs x, held (wr t) = .ok cs ∧ cs.stream? (s i) = some x) :
    AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals R ω t Lb i) =
        Flock.Guarantees.coinRounds rs := by
  intro R ω t R₂ τ o hA i ha hc
  obtain ⟨cs, x, hcs, hx⟩ := hst R ω t R₂ τ o hA i ha hc
  exact repIdx_of_check hcl hrs (hsv R ω t R₂ τ o hA i ha hc).fold.kLog (hchk R ω t R₂ τ o hA i ha hc cs hcs) hx
    (hn i) (by rw [hrd, hcs]) (hpr R ω t R₂ τ o hA i ha hc) (hf R ω t R₂ τ o hA i ha hc)

/-- **`RepsDrawn` at `holoMap`'s draws from the reps' count**: where each rep's consumed rounds are the schedule's coin
rounds at every accepted outer proof (`hcount`, rep 0's by E's preconditions there), the session's streams are both
reps' first and the history has room for both reps' rounds. -/
theorem repsDrawn_of_counts {ish : Flock.RecVStar.Inner} {tbl : String} {bc : Cm → String → ℕ → List UInt8}
    {rs : Array Flock.RecVStar.SchedRound} {Ls : List (Flock.Guarantees.HoloStream × Array Flock.RecVStar.SchedRound)}
    (hS : Flock.Guarantees.holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked =
      some ((.rep 0, rs) :: (.rep 1, rs) :: Ls))
    (hr : 1 + 2 * (Flock.Guarantees.coinRounds rs).length ≤ r)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hcount : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals R ω t Lb i) =
        Flock.Guarantees.coinRounds rs) :
    RepsDrawn zs dg N k Rw sh rep vals (Flock.Guarantees.holoMap st hp Cm ish tbl bc).selC
      (fun i => (Flock.Guarantees.HoloStream.rep i).name tbl) fun _ q => q := by
  intro R ω t R₂ τ o hA i ha hc
  rw [hcount R ω t R₂ τ o hA 0 (halg R ω t R₂ τ o hA 0) (hacc R ω t R₂ τ o hA 0), hcount R ω t R₂ τ o hA i ha hc]
  refine ⟨?_, fun p q h => FlockC3.Compose.selC_of_holoSelOne (holoSelOne_coinRound hS i h)⟩
  unfold repStart
  split <;> omega

/-- **`RepsDrawn` at `holoMap`'s draws** from custody at a map `m` that draws a round of each rep's stream below `r`
(`hsel`; `holoSel_first` at `holoMap`), the reps' rounds their held streams' (`hrd`), the check at every accepted outer
proof (`hchk`) and rep names past the sparse phases' and R's (`hn`). -/
theorem repsDrawn_holoMap {m : InnerRecordMap Cm (ℐ).Coin} {ish : Flock.RecVStar.Inner} {tbl : String}
    {bc : Cm → String → ℕ → List UInt8} {rs : Array Flock.RecVStar.SchedRound}
    {Ls : List (Flock.Guarantees.HoloStream × Array Flock.RecVStar.SchedRound)}
    (hS : Flock.Guarantees.holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked =
      some ((.rep 0, rs) :: (.rep 1, rs) :: Ls))
    (hcl : 0 < ish.claims) (hrs : Flock.RecVStar.holoRepRounds ish.m ish.kLog ish.claims = some rs)
    (hr : 1 + 2 * (Flock.Guarantees.coinRounds rs).length ≤ r)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hleaf : HLeafC3 zs dg N k Rw) (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw src sh rep vals) (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m)
    (hsel : ∀ i, ∃ j q, j < r ∧ m.selC j ((Flock.Guarantees.HoloStream.rep i).name tbl) q)
    (hrd : ∀ R ω t Lb i, (rep R ω t Lb i).rounds =
      heldRounds (held (wr t)) ((Flock.Guarantees.HoloStream.rep i).name tbl))
    (hn : ∀ i : Fin 2, (((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{Flock.RecVStar.SPARSE}/" ||
      ((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{Flock.RecVStar.STACKED}/") = false)
    (hchk : AtRep zs dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ cs, held (wr t) = .ok cs → Flock.RecVStar.checkCommitRounds ish cs = .ok ()) :
    RepsDrawn zs dg N k Rw sh rep vals (Flock.Guarantees.holoMap st hp Cm ish tbl bc).selC
      (fun i => (Flock.Guarantees.HoloStream.rep i).name tbl) fun _ q => q :=
  repsDrawn_of_counts hS hr halg hacc
    (repIdx_atRep (s := fun i => (Flock.Guarantees.HoloStream.rep i).name tbl) hcl hrs hpr hsv hf hrd hn hchk
      (streams_of_custody hcus hleaf hsel))

/-- **`hchk` from the covers check at the held record**: where the covers check passes at the commits the verifier
holds at every accepted outer proof's history (`(held (wr t)).toOption`, `innerHeld`'s at the guarantee's `held`), the
round-for-round check does. -/
theorem hchk_of_held {ish : Flock.RecVStar.Inner}
    (h : AtRep zs dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ cs, (held (wr t)).toOption = some cs → (Flock.RecVStar.checkCovers ish cs).isOk = true) :
    AtRep zs dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ cs, held (wr t) = .ok cs → Flock.RecVStar.checkCommitRounds ish cs = .ok () := by
  intro R ω t R₂ τ o hA i ha hc cs hcs
  obtain ⟨⟨⟩, hk⟩ := FlockC3.VStarList.ok_of_isOk (h R ω t R₂ τ o hA i ha hc cs (congrArg Except.toOption hcs))
  exact checkCommitRounds_of_checkCovers hk

/-- **`hchk` from `AcceptsC3` at the history**: at the guarantee's `held` (`innerCommitsAt`), `AcceptsC3`'s covers
conjunct at every accepted outer proof's history gives the round-for-round check there. -/
theorem hchk_of_acceptsC3 {ish : Flock.RecVStar.Inner} {reg : Lean.Json}
    (h : AtRep zs dg N k Rw sh rep fun _ _ t _ _ => AcceptsC3 ish st (wr t).1 (wr t).2 reg) :
    AtRep zs dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ cs, innerCommitsAt (wr t).1 (wr t).2 st reg = .ok cs → Flock.RecVStar.checkCommitRounds ish cs = .ok () := by
  intro R ω t R₂ τ o hA i ha hc cs hcs
  obtain ⟨⟨⟩, hk⟩ := FlockC3.VStarList.ok_of_isOk ((h R ω t R₂ τ o hA i ha hc).2.2.1 cs hcs)
  exact checkCommitRounds_of_checkCovers hk

/-- **`ChecksC3`'s covers conjunct from the verifier's run**: `AcceptsC3` at the records written at a history, which
the verifier's acceptance gives (`FlockC3.VStarList.acceptsC3_of_verify`), passes the covers check at the verifier's
hold of the inner transcript there (`innerHeld`), as the event's `ChecksVR` reads it. -/
theorem covers_of_acceptsC3 {ish : Flock.RecVStar.Inner} {reg : Lean.Json} {t : List (Cm × (ℐ).Coin)}
    (h : AcceptsC3 ish st (wr t).1 (wr t).2 reg) :
    ∀ c, Flock.Guarantees.innerHeld wr st reg t = some c → (Flock.RecVStar.checkCovers ish c).isOk = true := by
  intro c hc
  unfold Flock.Guarantees.innerHeld at hc
  cases hi : innerCommitsAt (wr t).1 (wr t).2 st reg with
  | error _ => rw [hi] at hc; cases hc
  | ok c' => rw [hi] at hc; cases hc; exact h.2.2.1 _ hi

/-- **The covers check at every accepted outer proof of the checked sessions**: where it fails at the history's hold
`cs t`, no session keeps its own list (`checkedC3`, whose `ChecksC3` reads it), so the first accepts nothing
(`not_acceptsZKR_keepOwn`, as `hLeafC3_checked`). -/
theorem covers_checked {ish : Flock.RecVStar.Inner}
    {cs : List (Cm × (ℐ).Coin) → Option Flock.RecVStar.InnerCommits}
    {coins : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} (hS : 0 < S) :
    AtRep (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h) dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ c, cs t = some c → (Flock.RecVStar.checkCovers ish c).isOk = true := by
  intro R ω t R₂ τ o hA _ _ _ c hc
  by_contra hno
  exact not_acceptsZKR_keepOwn (z := Flock.Guarantees.custodied zr hC h R ω t R₂ ⟨0, hS⟩)
    (P := fun J own tr => (ℐ).r ≤ t.length ∧
      Flock.Guarantees.ChecksC3 ish (cs t) (coins t) (⟨0, hS⟩ : Fin S) (bind ⟨0, hS⟩) rows J own tr)
    (fun _ _ _ hp => by
      obtain ⟨-, _, _, -, -, -, -, -, hcov⟩ := hp
      exact hno (hcov c hc)) _ (hA.1 ⟨0, hS⟩).1

end atRep

section holo

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n}
  {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T (Flock.Guarantees.holoCTs st hp) st hp

/-- **`RepsDrawn` at the verifier's map**, as `stageCoins_holo`'s `hdr` reads it: `repsDrawn_holoMap` at the guarantee's
coin types and `m := holoMap`, which draws each rep's first coin round below the coin count (`holoSel_first`). -/
theorem repsDrawn_holoMapAt {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {ish : Flock.RecVStar.Inner} {tbl : String} {bc : Cm → String → ℕ → List UInt8}
    {rs : Array Flock.RecVStar.SchedRound} {Ls : List (Flock.Guarantees.HoloStream × Array Flock.RecVStar.SchedRound)}
    (hS : Flock.Guarantees.holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked =
      some ((.rep 0, rs) :: (.rep 1, rs) :: Ls))
    (hcl : 0 < ish.claims) (hrs : Flock.RecVStar.holoRepRounds ish.m ish.kLog ish.claims = some rs)
    (hr : 1 + 2 * (Flock.Guarantees.coinRounds rs).length ≤
      1 + (Exec.kindCoinTys (Flock.Guarantees.holoKinds (holoAt st hp))).length)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hleaf : HLeafC3 zs dg N k Rw) (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw src sh rep vals) (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bc))
    (hrd : ∀ R ω t Lb i, (rep R ω t Lb i).rounds =
      heldRounds (held (wr t)) ((Flock.Guarantees.HoloStream.rep i).name tbl))
    (hn : ∀ i : Fin 2, (((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{Flock.RecVStar.SPARSE}/" ||
      ((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{Flock.RecVStar.STACKED}/") = false)
    (hchk : AtRep zs dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ cs, held (wr t) = .ok cs → Flock.RecVStar.checkCommitRounds ish cs = .ok ()) :
    RepsDrawn zs dg N k Rw sh rep vals (Flock.Guarantees.holoMap st hp Cm ish tbl bc).selC
      (fun i => (Flock.Guarantees.HoloStream.rep i).name tbl) fun _ q => q :=
  repsDrawn_holoMap hS hcl hrs hr halg hacc hleaf hpr hsv hf hcus (holoSel_first hS hcl hrs hr) hrd hn hchk

/-- **`RepsDrawn` at the verifier's map at the checked sessions** and the guarantee's hold of the inner transcript
(`innerCommitsAt`): `repsDrawn_holoMapAt` with the whole history (`hLeafC3_checked`) and the round-for-round check
(`hchk_of_held` at `covers_checked`, `ChecksC3`'s covers conjunct) that the checked sessions give, the reps' streams
and rounds `holoStreams`' (`holoStreams_eq`), and the coin count's room for both reps' rounds from the reps' forms
(`heldBefore_rep_forms`, `length_kindCoinTys_holoKinds`) at the class's public statement (`hA`, which
`repFormsAt_holoAt` gives at `ClassOk` and the inner statement check). -/
theorem repsDrawn_checked {ish : Flock.RecVStar.Inner} {reg : Lean.Json}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {coins : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {tbl : String} {bc : Cm → String → ℕ → List UInt8}
    {Ls : List (Flock.Guarantees.HoloStream × Array Flock.RecVStar.SchedRound)} (hS₀ : 0 < S)
    (hS : Flock.Guarantees.holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some Ls)
    (hcl : 0 < ish.claims) (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (halg : AlgebraDecodes (I := ℐ)
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) coins bind rows zr hC h) dg N k Rw sh rep)
    (hacc : AccChains (I := ℐ)
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) coins bind rows zr hC h) dg N k Rw sh rep)
    (hpr : ProofOfVals
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) coins bind rows zr hC h) dg N k Rw sh rep
      vals)
    (hsv : SetupOfVals
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) coins bind rows zr hC h) dg N k Rw src sh
      rep vals)
    (hf : FramesOfVals
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) coins bind rows zr hC h) dg N k Rw sh rep
      vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr (fun w => innerCommitsAt w.1 w.2 st reg) ish.sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bc))
    (hrd : ∀ R ω t Lb i, (rep R ω t Lb i).rounds =
      heldRounds (innerCommitsAt (wr t).1 (wr t).2 st reg) ((Flock.Guarantees.HoloStream.rep i).name tbl))
    (htbl : ∀ i : Fin 2, (((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{Flock.RecVStar.SPARSE}/" ||
      ((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{Flock.RecVStar.STACKED}/") = false) :
    RepsDrawn (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) coins bind rows zr hC h) dg N k Rw
      sh rep vals (Flock.Guarantees.holoMap st hp Cm ish tbl bc).selC
      (fun i => (Flock.Guarantees.HoloStream.rep i).name tbl) fun _ q => q := by
  obtain ⟨rs, Ls', hrs, rfl⟩ := holoStreams_eq hS
  have hc := (heldBefore_rep_size 0 rs).symm.trans (heldBefore_rep_forms 0 hcl hrs)
  have h1 := length_kindCoinTys_holoKinds (holoAt st hp)
  have h2 := congrArg List.length (holoForms_eq (holoAt st hp))
  rw [hA] at h2
  simp only [List.length_append] at h2
  exact repsDrawn_holoMapAt (held := fun w => innerCommitsAt w.1 w.2 st reg) hS hcl
    ((Flock.Guarantees.HoloRepCoins.holoRepRounds_eq _ _ _).trans hrs) (by omega) halg hacc
    (hLeafC3_checked hS₀ dg N k Rw) hpr hsv hf hcus hrd htbl
    (hchk_of_held (held := fun w => innerCommitsAt w.1 w.2 st reg)
      (covers_checked (cs := Flock.Guarantees.innerHeld wr st reg) hS₀))

end holo

end FlockC3.Reps
