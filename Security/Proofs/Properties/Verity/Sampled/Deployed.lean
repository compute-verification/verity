module

public import Proofs.Flock.Claims.Bridge.CustodyZkP
public import Proofs.Flock.Bridge.Draw
public import Proofs.Properties.Verity.SampledCoins
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# A3 and `live-verifier` at the session the deployment runs, with no premise but V5's law

`Verity.Sampled.Live` carries two of V5's hypotheses as fields, A3 at the device's bytes the draw reads (`Live.hA3`)
and `live-verifier` at the writer whose record v2 decodes (`Live.hCust`, `RecordCustodyZKJPLaw` under the session's law,
with the draw file's size, V5's `hsz`). This file proves both at the deployment's session, for the vocabulary instance
to cite:

* **The draw file** (`printedDraw law`): at each draw `S`, the one `flock-verify draw` prints for the verifier's own draw
  of `S` under `law` over the `n` proof blocks (`UnitDraw.toJson` of `ZkLink.drawOf`); at a subset law it is
  `Claims.Bridge.CustodyZkP.djPrinted` (`printedDraw_subset`). It decodes to the draw it printed at every law but a
  Bernoulli one (`drawFileZK_printedDraw`), by the subset, stratified and work laws' round trips
  (`ZkLink.drawFileZK_of_toJson_subset`, `ZkLink.drawFileZK_of_toJson`, and here `ZkLink.drawFileZK_of_toJson_work`).
  V5's `LawOn` admits no Bernoulli law (`lawOn_not_bernoulli`).
* **`live-verifier` at the writer** (`recordCustodyZKJPLaw_deployed`): `RecordCustodyZKJPLaw` at the writer that keeps an
  accepted record (`Claims.Bridge.CustodyZkP.wrZP`), the printed draw file and the law, at any inputs and public inputs
  (`SetsUp`'s `own = none` among them), under V5's `LawOn law ns kd`; and at a member of a window's sample
  (`recordCustodyZKJPLaw_part`), whose draw file at its part `S` is printed under `subset |S|` (`printedPart`).
* **The draw's size** (`drawSize_deployed`): V5's `hsz` at the printed draw file, under the same `LawOn`
  (`Flock.Bridge.lawOn_size`).
* **A3 at the device's bytes** (`uniformRandomBytes_deployed`): the session's sample is the device's one-stage rounds,
  `MAX_COINS` coins of `8 * COIN_BYTES` raw bits (`DeployedCoins`, `Properties.Challenger.oneStageRounds`), and the draw's
  one byte stream is their bytes in order (`deployedSrc`, `coinBytes`), so A3 holds there
  (`uniformRandomBytes_coinBytes`). The coupling's byte source at it is `coinBytes` of the sample itself.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace FlockSoundness.Discharge.ZkLink

open Flock FlockSoundness.Refine Lean

open private stratumOfJson workStratumOfJson from Flock.Draw
open private asNat_toJson fromJson?_str ok_bind getObjVal?_mkObj getObjVal?_mkObj_not arr_mapM_map_ok
  stratumOfJson_toJson from Proofs.Flock.Soundness.Discharge.ZkLink.DrawOkJson

section
variable (t : String) (us : Array (ℕ × ℕ)) (k w f : ℕ)

private theorem workStratum_template :
    (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "template" = .ok (.str t) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem workStratum_units : (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "units" =
    .ok (.arr (us.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem workStratum_work :
    (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "work" = .ok (Lean.toJson w) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem workStratum_floor :
    (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "floor" = .ok (Lean.toJson f) :=
  getObjVal?_mkObj (by simp) (by simp)

private theorem workStratum_k : (Draw.WorkStratum.toJson ⟨⟨t, us, k⟩, w, f⟩).getObjVal? "k" = .ok (Lean.toJson k) :=
  getObjVal?_mkObj (by simp) (by simp)

end

private theorem workStratumOfJson_toJson (s s' : Draw.WorkStratum) (h : workStratumOfJson s.toJson = .ok s') :
    s' = s := by
  obtain ⟨⟨t, us, k⟩, w, f⟩ := s
  unfold workStratumOfJson at h
  simp only [Json.getObjValAs?, Json.getObjValD, workStratum_template, workStratum_units, workStratum_work,
    workStratum_floor, workStratum_k, Except.toOption, Option.getD, fromJson?_str, asNat_toJson, ok_bind] at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    have hs := ‹stratumOfJson _ = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [stratumOfJson_toJson ⟨t, us, k⟩ _ hs]

section
variable (K pop : ℕ) (ws : Array Draw.WorkStratum) (us : Array ℕ)

private abbrev dW : Draw.UnitDraw := { law := .work K ws, population := pop, units := us }

private theorem dW_toJson : (dW K pop ws us).toJson = Json.mkObj [("law", Json.str "work"),
    ("population", Lean.toJson pop), ("k", Lean.toJson K), ("strata", Json.arr (ws.map (·.toJson))),
    ("units", Json.arr (us.map fun x => Lean.toJson x))] :=
  rfl

private theorem work_population : (dW K pop ws us).toJson.getObjVal? "population" = .ok (Lean.toJson pop) := by
  rw [dW_toJson]; exact getObjVal?_mkObj (by simp) (by simp)

private theorem work_law : (dW K pop ws us).toJson.getObjVal? "law" = .ok (.str "work") := by
  rw [dW_toJson]; exact getObjVal?_mkObj (by simp) (by simp)

private theorem work_k : (dW K pop ws us).toJson.getObjVal? "k" = .ok (Lean.toJson K) := by
  rw [dW_toJson]; exact getObjVal?_mkObj (by simp) (by simp)

private theorem work_strata : (dW K pop ws us).toJson.getObjVal? "strata" = .ok (.arr (ws.map (·.toJson))) := by
  rw [dW_toJson]; exact getObjVal?_mkObj (by simp) (by simp)

private theorem work_units :
    (dW K pop ws us).toJson.getObjVal? "units" = .ok (.arr (us.map fun x => Lean.toJson x)) := by
  rw [dW_toJson]; exact getObjVal?_mkObj (by simp) (by simp)

private theorem work_closure : ∃ e, (dW K pop ws us).toJson.getObjVal? "closure" = .error e := by
  rw [dW_toJson]; exact getObjVal?_mkObj_not (by simp)

private theorem ofJson_toJson_work {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson (dW K pop ws us).toJson = .ok D) :
    D = dW K pop ws us := by
  obtain ⟨e, he⟩ := work_closure K pop ws us
  unfold Draw.UnitDraw.ofJson at h
  simp only [J.nat, J.arr, J.str, J.get, J.asStr, J.asArr, work_population, work_law, work_k, work_strata, work_units,
    he, asNat_toJson, ok_bind, Json.getArr?, Json.getStr?, pure, Except.pure] at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    have hu := ‹Array.mapM (fun x => J.asNat x _) (Array.map _ us) = Except.ok _›
    have hs := ‹Array.mapM workStratumOfJson (Array.map _ ws) = Except.ok _›
    rw [arr_mapM_map_ok _ _ (fun a r hr => by rw [asNat_toJson] at hr; exact (Except.ok.inj hr).symm) us _ hu,
      arr_mapM_map_ok _ _ (fun a r hr => workStratumOfJson_toJson a r hr) ws _ hs] at h
    exact (Except.ok.inj h).symm

end

variable {n : ℕ}

/-- **The round trip of the work law's draw file**: when `UnitDraw.ofJson` accepts what `UnitDraw.toJson` prints for the
verifier's own draw of `S` under a work law, it decodes to that draw. -/
theorem ofJson_toJson_drawOf_work (K : ℕ) (ws : Array Draw.WorkStratum) (pop : ℕ) (S : Finset (Fin n))
    {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson (drawOf (.work K ws) pop S).toJson = .ok D) :
    D = drawOf (.work K ws) pop S :=
  ofJson_toJson_work K pop ws _ h

/-- **`DrawFileZK` from the printed work draw**: a record whose draw file at every draw `S` is what `flock-verify draw`
printed for the verifier's own draw of `S` under the work law meets `DrawFileZK` at that law. -/
theorem drawFileZK_of_toJson_work (dj : Finset (Fin n) → Lean.Json) (K : ℕ) (ws : Array Draw.WorkStratum) (pop : ℕ)
    (hdj : ∀ S, dj S = (drawOf (.work K ws) pop S).toJson) : DrawFileZK dj (.work K ws) pop :=
  fun S _ hD => ofJson_toJson_drawOf_work K ws pop S (hdj S ▸ hD)

end FlockSoundness.Discharge.ZkLink

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Discharge

variable {n mPts : ℕ} {Reg : Type}

/-- **The draw file `flock-verify draw` prints** for the verifier's own draw of `S` under the law `law` over the `n`
proof blocks (`UnitDraw.toJson` of `ZkLink.drawOf`). -/
def printedDraw (law : Flock.Draw.Law) (S : Finset (Fin n)) : Lean.Json := (ZkLink.drawOf law n S).toJson

theorem printedDraw_subset (kd : ℕ) :
    printedDraw (n := n) (.subset kd) = Claims.Bridge.CustodyZkP.djPrinted kd :=
  rfl

/-- V5's law is no Bernoulli law. -/
theorem lawOn_not_bernoulli {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (hlaw : Flock.Bridge.LawOn law ns kd)
    (a b : ℕ) : law ≠ .bernoulli a b := by
  rintro rfl
  exact hlaw.elim

/-- **The printed draw file decodes to the draw it printed** (`DrawFileZK`) at every law but a Bernoulli one. -/
theorem drawFileZK_printedDraw (law : Flock.Draw.Law) (hb : ∀ a b, law ≠ .bernoulli a b) :
    ZkLink.DrawFileZK (printedDraw (n := n) law) law n := by
  cases law with
  | subset k => exact ZkLink.drawFileZK_of_toJson_subset _ k n fun _ => rfl
  | bernoulli a b => exact absurd rfl (hb a b)
  | stratified K St => exact ZkLink.drawFileZK_of_toJson _ K St n fun _ => rfl
  | work K ws => exact ZkLink.drawFileZK_of_toJson_work _ K ws n fun _ => rfl

/-- **`live-verifier` at the deployment's writer and draw file, under V5's law** (`Live.hCust`): `RecordCustodyZKJPLaw`
at the writer that keeps an accepted record (`wrZP`), its record half with no premise
(`Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP`), and its draw file half at the printed draw file
(`drawFileZK_printedDraw`), at any inputs `I` and public inputs `own`. -/
theorem recordCustodyZKJPLaw_deployed (I : Exec.Inputs) (own : Option ByteArray) {law : Flock.Draw.Law}
    {ns : List ℕ} {kd : ℕ} (hlaw : Flock.Bridge.LawOn law ns kd) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedDraw law) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedDraw law) Jf) fun _ => law :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, drawFileZK_printedDraw law (lawOn_not_bernoulli hlaw)⟩

/-- **The draw file a member of a window's sample is served** (`serve --draw-file`): at its part `S` of the sample, the
one `flock-verify draw` prints for the verifier's own draw of `S` under `subset |S|` over the member's `n` proof blocks. -/
def printedPart (S : Finset (Fin n)) : Lean.Json := printedDraw (.subset S.card) S

/-- **`live-verifier` at a member's writer and draw file** (`Flock.Guarantees.MemberV5.hCust`): `RecordCustodyZKJPLaw`
at the writer that keeps an accepted record (`wrZP`) and the member's printed draw file, under `subset |S|` at each part
`S`, with no premise: the record half is `Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP`, and the draw half the subset
law's round trip at `|S|`. -/
theorem recordCustodyZKJPLaw_part (I : Exec.Inputs) (own : Option ByteArray) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own printedPart Jf
      (Claims.Bridge.CustodyZkP.wrZP I own printedPart Jf) fun S => .subset S.card :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, fun S =>
    ZkLink.drawFileZK_of_toJson_subset (fun T => printedDraw (.subset S.card) T) S.card n (fun _ => rfl) S⟩

/-- **The printed draw file's size under V5's law** (V5's `hsz`): a printed draw file that `UnitDraw.ofJson` decodes
draws the law's `kd` units (`Flock.Bridge.lawOn_size`). -/
theorem drawSize_deployed {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (hlaw : Flock.Bridge.LawOn law ns kd) :
    ∀ S ud, Flock.Draw.UnitDraw.ofJson (printedDraw (n := n) law S) = .ok ud → ud.units.size = kd :=
  Flock.Bridge.lawOn_size hlaw (drawFileZK_printedDraw law (lawOn_not_bernoulli hlaw))

/-- **The deployed session's sample**: the device's one-stage rounds (`Properties.Challenger.oneStageRounds`),
`MAX_COINS` coins of `8 * COIN_BYTES` raw bits each. -/
abbrev DeployedCoins : Type :=
  Fin _root_.SampledProofs.Challenger.MAX_COINS → Fin (8 * _root_.SampledProofs.Challenger.COIN_BYTES) → Bool

/-- The length of the deployed draw's one byte stream: the coins' bytes. -/
abbrev deployedLb : Fin 1 → ℕ :=
  fun _ => _root_.SampledProofs.Challenger.MAX_COINS * _root_.SampledProofs.Challenger.COIN_BYTES

/-- **The deployed draw's bytes**: the coins' bytes in order, coin `c`'s byte `q` at `c * COIN_BYTES + q`
(`coinBytes`). -/
noncomputable abbrev deployedSrc : DeployedCoins → (i : Fin 1) → Fin (deployedLb i) → Fin 256 :=
  coinBytes _root_.SampledProofs.Challenger.MAX_COINS _root_.SampledProofs.Challenger.COIN_BYTES

/-- **A3 at the deployed draw's bytes** (`Live.hA3`), with no premise (`uniformRandomBytes_coinBytes`). -/
theorem uniformRandomBytes_deployed : Assumptions.UniformRandomBytes deployedLb deployedSrc :=
  uniformRandomBytes_coinBytes _root_.SampledProofs.Challenger.MAX_COINS _root_.SampledProofs.Challenger.COIN_BYTES

end Verity.Sampled
