module

public import Proofs.Flock.Claims.Complete.Zk.Coins
public import Proofs.Flock.Claims.DeployVK
public import Proofs.Flock.Claims.KS.ForkSlack
public import Proofs.Flock.Claims.KS.PlainBound
public import Proofs.Flock.Claims.KS.PlainFirst
public import Proofs.Flock.Claims.KS.PlainWrP
public import Proofs.Flock.Claims.KS.TableBound
public import Proofs.Flock.Claims.Steps.FlockKSDeployK

/-! # What the plain KS proofs read of `Claims.Recursion.PlainK`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Claims.Recursion

local notation "𝔽" => FlockLevel3.GF128

/-- **`VK keep`'s decision at SHA-512**: it accepts exactly when the verdict read through `keep` accepts the record and
proof files at the statement's setup and the custody checks pass (`next_decide_sha512`'s form). -/
theorem nextK_decide_sha512 (keep : Bool → Bool) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act false x as cs = .decide s spec sts link c₁ mids f) :
    ((nextK keep x as cs).eval sha512).1 = .acc ↔
      verifyWithK keep (Claims.hOf sha512) spec (tabsOf sts) f.record f.proofs = .ok () ∧
        checks false s spec x link c₁ mids f = .ok () := by
  rw [nextK_decide keep sha512 h, act_decide_setupWith h]
  constructor
  · rintro ⟨spec', sts', he, hv, hc⟩
    cases he
    exact ⟨hv, hc⟩
  · rintro ⟨hv, hc⟩
    exact ⟨spec, sts, rfl, hv, hc⟩

end FlockDeploy

/-- **`VK keep` answers each round with its coin's bytes** (`answersCoin_deploy`'s proof, at `nextK keep`). -/
theorem Claims.Bridge.answersCoin_VK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) :
    Claims.Bridge.AnswersCoin H (FlockDeploy.VK keep) x FlockDeploy.Coin.bytes := by
  intro as cs c m h
  change ((FlockDeploy.nextK keep x as (cs ++ [c])).eval H).1 = .msg m at h
  cases ha : FlockDeploy.act false x as (cs ++ [c]) with
  | rej =>
    rw [FlockDeploy.nextK_rej keep H ha] at h
    cases h
  | coin c' =>
    rw [FlockDeploy.nextK_coin keep H ha] at h
    have hc : c = c' := by simpa using Claims.Bridge.act_coin ha
    rw [← Claims.Out.msg.inj h, hc]
  | decide s spec sts link c₁ mids f =>
    simp only [FlockDeploy.nextK, ha] at h
    exact absurd h (Claims.Bridge.outOf_ne_msg _ _ _)

namespace FlockDeploy.KS

open Flock

open Claims (OC OCV)

open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128

/-- **An accepted run of `VK keep` at `sha512`** (`accE_within`'s proof, at `keep`): the statement sets up, the run
decides on its last message, read as a final message on the statement, and on the link of the prover's first message;
the custody checks pass; and the verdict read through `keep` accepts with SHA-512 answered only within the run's
verifier log. -/
theorem accK_within (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} (h : (Claims.run sha512 P (VK keep) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P (VK keep) x ω r).seen = some f ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧ checks false s spec x link c₁ mids f = .ok () ∧
      verifyWithMK keep (hashesIn ((Claims.run sha512 P (VK keep) x ω r).vLog.map Prod.fst)) spec (tabsOf sts)
        f.record f.proofs = .ok () := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accK_decide keep h
  obtain ⟨hv, hc⟩ := decisionK_runs keep (setupOf_some (act_decide hact)).2 hrun
  obtain ⟨rest, hhead⟩ := run_head sha512 P (VK keep) x ω r h
  obtain ⟨rest', has, hlink⟩ := act_decide_link hact
  rw [hhead] at hlink
  refine ⟨s, spec, sts, link, c₁, mids, f, act_decide hact, ?_, hlink, hc, verdictK_within keep hv⟩
  have hl := act_decide_final hact
  rw [List.getLast?_map, Option.bind_map] at hl
  exact hl

/-- **An accepted run of `VK keep` opens a column only as a Merkle opening** of table 0's level 0 in its own log
(`plainE_opening_raw`'s proof, at `keep`): a column `u` a rep opens at `q` is an opened row whose opening under table
0's scheme against `root_B[0]` of the tape's first message's link passed, every SHA-512 input of it in the run's
verifier log. -/
theorem plainK_opening_raw (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (VK keep) x ω r).out = .acc)
    (hu : (q, u) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω r).seen) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P (VK keep) x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := accK_within keep h
  obtain ⟨st, sch, row, hst, hsch, hop, hu⟩ :=
    plain_opening_raw_of_verdictK keep (L := (Claims.run sha512 P (VK keep) x ω r).vLog) hs hf hc hv hu
  exact ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, hop, hu⟩

/-- **A self-clash among an accepted run's columns collides in its own log**, at `VK keep` (`plain_selfClash_tied`'s
proof, at `plainK_opening_raw`). -/
theorem plainK_selfClash_tied (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {q : ℕ} {u v : List 𝔽} (h : (Claims.run sha512 P (VK keep) x ω r).out = .acc)
    (hu : (q, u) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω r).seen)
    (hv : (q, v) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω r).seen) (huv : u ≠ v) :
    Claims.TiedColl sha512 (Claims.run sha512 P (VK keep) x ω r).vLog := by
  obtain ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, ho, hu'⟩ := plainK_opening_raw keep h hu
  obtain ⟨s', spec', sts', st', sch', link', row', hs', hst', hsch', hlink', ho', hv'⟩ := plainK_opening_raw keep h hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_append_self (tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e]))

end FlockDeploy.KS

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec

open FlockSoundness.Discharge.Tables Claims.Bridge.Custody Claims.Bridge.Live

open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A prover of the live session at one table simulates `P` on the coins `Φ`**: wherever `V false` accepts at
`Φ (u, r')`, the coins of `V false` at the tape `u` and the live coins `r'`, the live outcome of `P'` at the tape
`(ω, u)` and `r'` is the run's (`viewOf`). -/
def PlainSimRunK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ} (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r', (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r' = viewOf 1 mPts spec link c₁ f

/-- **A prover of the live session at one table simulates `P`**: the coins `Φ (u, r')` have the sampling design of uniform coins,
and `P'` simulates `P` on them (`PlainSimRun`). -/
def PlainSimK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ} (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  (∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E) ∧
    PlainSimRunK keep P x ω mPts N dg w P' Φ

end FlockDeploy.KS
