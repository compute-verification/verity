import Proofs.Flock.C3.Reps.RepsDrawnAt
import Proofs.Flock.Recursive.HoloSchedCheck

/-!
# RepsDrawn (ii) at the verifier's map, from the covers check's rounds

`coins_of_checkCommitRounds`: the verifier's round-for-round check (`checkCommitRounds`' first loop) holds every stream
but the sparse phases' and R's to the inner shape's schedule (`holoRepRounds` at a holo shape, `repRounds` else), so a
rep's held rounds are the schedule's round for round (`heldRounds_of_check`). `repIdx_of_check` reads RepsDrawn (ii)'s
count off it at a rep unit's values: at a holo shape the record is consumed whole (`UnitFrames.whole`) and the proof's
rounds are the held stream's (`UnitProof.coins`), so `t₄.k` is the schedule's count (`repIdx_of_size`); at a plain shape
the schedule's `z_partial` round draws a coin where the frames' draws none (`repRounds_zp`, `HoloLcAnswered.zp_n`), so
no rep unit passes the check there. A rep's stream is held where custody's coins half holds one of its rounds' coins
(`stream_of_custody`, `streams_of_custody`), as the map draws rep `i`'s first coin round (`holoSel_first`).

`repsDrawn_holoMap` is `RepsDrawn` at `holoMap`'s draws from those, with the check at every accepted outer proof
(`hchk`) and rep names past the sparse phases' and R's (`hn`) as hypotheses.
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

/-- **A rep's consumed rounds are the hidden-unit schedule's coin rounds** where its transcript, from round 0,
consumed as many rounds as the schedule at the rep's `m` and `k_log` has. -/
theorem repIdx_of_size {S₀ : FlockSoundness.Model.Statement} {v : UnitVals} {m kLog claims : ℕ}
    {rs : Array SchedRound} (hc : 0 < claims) (hrs : holoRepRounds m kLog claims = some rs) (hk0 : v.t.k = 0)
    (hm : v.st.m = m) (hkl : S₀.kLog = kLog) (hk : v.t₄.k = rs.size) : repIdx S₀ v = coinRounds rs := by
  have h22 := HoloRepCoins.holoRepRounds_m hrs
  have h6 := S₀.hk6
  have hz : v.t.k + 3 + (v.st.m - 6) + 2 + (S₀.kLog - 6) = zpRound m kLog := by
    unfold zpRound
    omega
  unfold repIdx
  rw [hz, hk0, Nat.sub_zero, hk]
  exact (HoloRepCoins.coinRounds_holoRep_split hc hrs).symm

/-- **RepsDrawn (ii)'s count at a rep unit under the check**: its rounds are its held stream's (`hd`, `hp`), which the
check holds to the inner shape's schedule. At a holo shape the record is consumed whole, so the unit consumed the
schedule's count; at a plain shape the schedule's `z_partial` round draws a coin and the frames' draws none. -/
theorem repIdx_of_check {S₀ : FlockSoundness.Model.Statement} {sh : Flock.Shape}
    {d : FlockVBridge.Compiled.RepData} {v : UnitVals} {ish : RecVStar.Inner} {rs : Array SchedRound}
    {cs : InnerCommits} {s : String} {st : InnerStream}
    (hcl : 0 < ish.claims) (hrs : holoRepRounds ish.m ish.kLog ish.claims = some rs) (hm : v.st.m = ish.m)
    (hfk : v.st.fold.kLog = ish.kLog) (hkl : S₀.kLog = ish.kLog) (hchk : RecVStar.checkCommitRounds ish cs = .ok ())
    (hs : cs.stream? s = some st) (hn : (s.startsWith s!"{SPARSE}/" || s.startsWith s!"{STACKED}/") = false)
    (hd : d.rounds = heldRounds (.ok cs) s) (hp : UnitProof sh d v) (hf : UnitFrames sh d v) :
    repIdx S₀ v = coinRounds rs := by
  have hc : ∀ rep, (if ish.holo then holoRepRounds else repRounds) ish.m ish.kLog ish.claims = some rep →
      (v.t.rounds.map (·.coins)).map (·.size) = rep.map (·.1) := fun _ hsch => by
    rw [← hp.coins, hd]
    exact heldRounds_of_check hsch hchk hs hn
  cases hh : ish.holo with
  | true =>
    have e := congrArg Array.size (hc rs (by simp only [hh, ↓reduceIte]; exact hrs))
    simp only [Array.size_map] at e
    exact repIdx_of_size hcl hrs hf.k0 hm hkl (hf.whole.trans e)
  | false =>
    obtain ⟨out, hout, hz⟩ := repRounds_zp hrs
    have e := hc out (by simp only [hh, Bool.false_eq_true, ↓reduceIte]; exact hout)
    have h22 := HoloRepCoins.holoRepRounds_m hrs
    have h6 := S₀.hk6
    have hk0 := hf.k0
    have hidx : v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) = zpRound ish.m ish.kLog := by
      unfold zpRound
      omega
    have hlt : v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) < v.t.rounds.size := hf.lc.zp_lt
    have hn0 : (v.t.rounds[v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6)]!).n = 0 := hf.lc.zp_n
    have hwf := hf.wf _ hlt
    rw [hidx] at hlt hn0 hwf
    rw [getElem!_pos v.t.rounds (zpRound ish.m ish.kLog) hlt] at hn0 hwf
    have e' := congrArg (fun a : Array ℕ => a[zpRound ish.m ish.kLog]?) e
    simp only [Array.getElem?_map, hz, Array.getElem?_eq_getElem hlt, Option.map_some, Option.some.injEq] at e'
    omega

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

/-- **RepsDrawn (ii)'s count at every accepted outer proof**: `repIdx_of_check` at the values `vals`, whose setup's
`m` and `k_log` are the shape's (`SetupOfVals`), with the rep's rounds its held stream's (`hrd`, `withReads`' by
definition), the check passing at the held record (`hchk`) and the rep's stream held (`hst`). -/
theorem repIdx_atRep {ish : Flock.RecVStar.Inner} {rs : Array Flock.RecVStar.SchedRound} (hcl : 0 < ish.claims)
    (hrs : Flock.RecVStar.holoRepRounds ish.m ish.kLog ish.claims = some rs) (hm : sh.m = ish.m)
    (hkl : sh.kLog = ish.kLog) (hkS : ((holoAt st hp).cls.pubStatement (holoAt st hp).regs).kLog = ish.kLog)
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
  have hu := hsv R ω t R₂ τ o hA i ha hc
  exact repIdx_of_check hcl hrs (hu.m.trans hm) (hu.kL.trans hkl) hkS (hchk R ω t R₂ τ o hA i ha hc cs hcs) hx (hn i)
    (by rw [hrd, hcs]) (hpr R ω t R₂ τ o hA i ha hc) (hf R ω t R₂ τ o hA i ha hc)

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
    (hr : 1 + 2 * (Flock.Guarantees.coinRounds rs).length ≤ r) (hm : sh.m = ish.m) (hkl : sh.kLog = ish.kLog)
    (hkS : ((holoAt st hp).cls.pubStatement (holoAt st hp).regs).kLog = ish.kLog)
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
    (repIdx_atRep (s := fun i => (Flock.Guarantees.HoloStream.rep i).name tbl) hcl hrs hm hkl hkS hpr hsv hf hrd hn
      hchk (streams_of_custody hcus hleaf hsel))

end atRep

end FlockC3.Reps
