import Proofs.Properties.Remote.Honest

/-!
# Remote's check past parsing: what it accepts

Remote's check reads bytes through `Lean.Json.parse`, which is `partial`, so no proof evaluates it on a literal input.
`Core.Composition.checkSame_decode` splits it into decoding (`decodeCases`) and a check of the decoded values
(`checkSameOf`), equal on every input (`remote_check_decode`). Here the decoded check is shown to accept:

* `checkSameOf_accept`: a composition's decoded check accepts when the cases fit, every component accepts its decoded
  case, the tied components committed to one record, and the commitment binds the components' digests.
  `checkSameOf_three` is that for any three components of distinct ids that accept on one digest.
* `remote_accepts_decoded`: remote's decoded check accepts on any three cases the policy, PoUW and the one-stage audit
  accept on one digest, and `remote_accepts`: remote's check accepts the bytes of three cases committing to one record
  that the components decode to cases they accept, the converse of `remote_accept`.
* `pouw_accepts`: PoUW's check accepts a literal window, a draw of `k = 4` of 4 units, by `decide +kernel`
  (`Reach.Pouw`).

The policy's and the one-stage audit's own checks parse again past their decoders (the policy decodes its program text
in `names` and `Desc.evaluate`; the one-stage audit reads each session's files), so their acceptance on a literal is a
hypothesis here.
-/

namespace Properties.Remote.SecurityProofs

open Core Core.Composition
open Lean (Json)

private theorem find_fit (cs : List Protocol) (ps : List Named) :
    (cs.zip (ps.zip (ps.map fun n => (n.id, n.params)))).find? (fun x => decide ¬x.2.2.2 = x.2.1.params) = none := by
  induction cs generalizing ps with
  | nil => rfl
  | cons P cs ih =>
    cases ps with
    | nil => rfl
    | cons n ps =>
      simp only [List.map_cons, List.zip_cons_cons, List.find?_cons, not_true_eq_false, decide_false]
      exact ih ps

/-- **A composition's decoded check accepts** when the cases fit (each component named once, in the components' order,
under the parameters the composition names for it), every component's answer on its decoded case accepts, the
components `ss` committed to one record, and the commitment binds the components' digests. It publishes each
component's public outputs and profile under its id. -/
theorem checkSameOf_accept {id : String} {ss : List String} {cs : List Protocol} {ps : List Named} {c : ByteArray}
    {ds : List Decoded} (hp : ps.map (·.id) = cs.map (·.id))
    (hacc : ∀ x ∈ ds, (x.P.answerOf x.d).verdict = .accept) (htie : straysOf ss ds = none)
    (hc : binding id (ds.map fun x => (x.P.id, x.P.answerOf x.d)) = some c) :
    checkSameOf id ss cs ps c (ps.map fun n => (n.id, n.params)) ds =
      .accept ((ds.map fun x => (x.P.id, x.P.answerOf x.d)).map fun (i, a) => (i, a.published.getD .null))
        ((ds.map fun x => (x.P.id, x.P.answerOf x.d)).map fun (i, a) => (i, a.profile.getD .null)) := by
  have hks : (ps.map fun n => (n.id, n.params)).map (·.1) = cs.map (·.id) := by
    rw [List.map_map]; exact hp
  have hrun : runOf cs ps (ps.map fun n => (n.id, n.params)) ds = .ok (ds.map fun x => (x.P.id, x.P.answerOf x.d)) := by
    unfold runOf
    simp only [hp, hks, ne_eq, not_true_eq_false, ite_false]
    rw [find_fit]
  have hrej : ((ds.map fun x => (x.P.id, x.P.answerOf x.d)).any fun a => decide (a.2.verdict = .reject)) = false := by
    rw [List.any_eq_false]
    intro a ha
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp ha
    simp [hacc x hx]
  have hund : ((ds.map fun x => (x.P.id, x.P.answerOf x.d)).any fun a => decide (a.2.verdict = .undecided)) = false := by
    rw [List.any_eq_false]
    intro a ha
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp ha
    simp [hacc x hx]
  unfold checkSameOf
  rw [hrun]
  simp only [htie, judge, hrej, hund, hc, ne_eq, not_true_eq_false, Bool.false_eq_true, ite_false]

/-- **Three components of distinct ids that accept their decoded cases on one digest**: the composition's decoded check
accepts under the commitment binding that digest for each, tied over all three. Over components left abstract. -/
theorem checkSameOf_three {id : String} {P₁ P₂ P₃ : Protocol} {ps : List Named}
    (hp : ps.map (·.id) = [P₁.id, P₂.id, P₃.id])
    (n₁₂ : (P₁.id == P₂.id) = false) (n₁₃ : (P₁.id == P₃.id) = false) (n₂₃ : (P₂.id == P₃.id) = false)
    {p₁ : P₁.Params} {c₁ : P₁.Commitment} {e₁ : P₁.Evidence} {pub₁ : P₁.Public} {prof₁ : P₁.Profile}
    {p₂ : P₂.Params} {c₂ : P₂.Commitment} {e₂ : P₂.Evidence} {pub₂ : P₂.Public} {prof₂ : P₂.Profile}
    {p₃ : P₃.Params} {c₃ : P₃.Commitment} {e₃ : P₃.Evidence} {pub₃ : P₃.Public} {prof₃ : P₃.Profile}
    (h₁ : P₁.check p₁ c₁ e₁ = .accept pub₁ prof₁) (h₂ : P₂.check p₂ c₂ e₂ = .accept pub₂ prof₂)
    (h₃ : P₃.check p₃ c₃ e₃ = .accept pub₃ prof₃)
    (d₂ : P₂.digest c₂ = P₁.digest c₁) (d₃ : P₃.digest c₃ = P₁.digest c₁) :
    checkSameOf id [P₁.id, P₂.id, P₃.id] [P₁, P₂, P₃] ps
        (Composition.bind id [(P₁.id, P₁.digest c₁), (P₂.id, P₁.digest c₁), (P₃.id, P₁.digest c₁)])
        (ps.map fun n => (n.id, n.params))
        [⟨P₁, ⟨.ok p₁, .ok c₁, .ok e₁⟩⟩, ⟨P₂, ⟨.ok p₂, .ok c₂, .ok e₂⟩⟩, ⟨P₃, ⟨.ok p₃, .ok c₃, .ok e₃⟩⟩] =
      .accept [(P₁.id, P₁.publicJson pub₁), (P₂.id, P₂.publicJson pub₂), (P₃.id, P₃.publicJson pub₃)]
        [(P₁.id, P₁.profileJson prof₁), (P₂.id, P₂.profileJson prof₂), (P₃.id, P₃.profileJson prof₃)] := by
  have a₁ := P₁.answerOf_accept h₁
  have a₂ := P₂.answerOf_accept h₂
  have a₃ := P₃.answerOf_accept h₃
  have htie : straysOf [P₁.id, P₂.id, P₃.id]
      [⟨P₁, ⟨.ok p₁, .ok c₁, .ok e₁⟩⟩, ⟨P₂, ⟨.ok p₂, .ok c₂, .ok e₂⟩⟩, ⟨P₃, ⟨.ok p₃, .ok c₃, .ok e₃⟩⟩] = none := by
    have e₁ : committedOf P₁.id
        [⟨P₁, ⟨.ok p₁, .ok c₁, .ok e₁⟩⟩, ⟨P₂, ⟨.ok p₂, .ok c₂, .ok e₂⟩⟩, ⟨P₃, ⟨.ok p₃, .ok c₃, .ok e₃⟩⟩] =
        some (P₁.digest c₁) := by
      simp [committedOf, Except.toOption]
    have e₂ : committedOf P₂.id
        [⟨P₁, ⟨.ok p₁, .ok c₁, .ok e₁⟩⟩, ⟨P₂, ⟨.ok p₂, .ok c₂, .ok e₂⟩⟩, ⟨P₃, ⟨.ok p₃, .ok c₃, .ok e₃⟩⟩] =
        some (P₁.digest c₁) := by
      rw [← d₂]; simp [committedOf, n₁₂, Except.toOption]
    have e₃ : committedOf P₃.id
        [⟨P₁, ⟨.ok p₁, .ok c₁, .ok e₁⟩⟩, ⟨P₂, ⟨.ok p₂, .ok c₂, .ok e₂⟩⟩, ⟨P₃, ⟨.ok p₃, .ok c₃, .ok e₃⟩⟩] =
        some (P₁.digest c₁) := by
      rw [← d₃]; simp [committedOf, n₁₃, n₂₃, Except.toOption]
    unfold straysOf
    simp [e₁, e₂, e₃, bytes_bne_self]
  have hp' : ps.map (·.id) = [P₁, P₂, P₃].map (·.id) := by rw [hp]; rfl
  rw [checkSameOf_accept (cs := [P₁, P₂, P₃]) hp' ?acc htie ?bind]
  · simp only [List.map_cons, List.map_nil, a₁, a₂, a₃, Option.getD_some]
  case acc =>
    intro x hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl <;> simp [a₁, a₂, a₃]
  case bind =>
    simp [binding, a₁, a₂, a₃, d₂, d₃]

/-- Remote's ids, in the order of its components. -/
theorem remote_ids (bs : Blobs) :
    [NciCheck.Protocol.protocol.id, PouwCheck.protocol.id, (SampledProofs.OneStageZk.protocol.onBlobs bs).id] =
      ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"] := by
  rw [id_policy, id_pouw, id_oneStage]

/-- **Remote's check is decoding followed by a check of the decoded values**, on every input
(`Core.Composition.checkSame_decode` at remote's components on the caller's blobs). -/
theorem remote_check_decode (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) :
    (Core.Remote.protocol.on blobs).check ps c ev =
      checkSameOf "verity/remote/v0" ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"]
        [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)]
        ps c (fitOf ev)
        (decodeCases
          [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)]
          ps ev) := by
  rw [remote_check]
  exact checkSame_decode _ _ _ ps c ev

/-- **Remote's decoded check accepts** on three decoded cases that the policy, PoUW and the one-stage audit (on the
caller's blobs) accept on one digest, under the commitment binding that digest for each component. -/
theorem remote_accepts_decoded {bs : Blobs} {ps : List Named}
    (hp : ps.map (·.id) = ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"])
    {pI : NciCheck.Protocol.Params} {cI : NciCheck.Protocol.Commitment} {eI : NciCheck.Protocol.Evidence}
    {pW : PouwCheck.Params} {cW : PouwCheck.Registration} {eW : PouwCheck.Evidence}
    {pS : SampledProofs.OneStageZk.Params} {cS : SampledProofs.OneStage.Commitment}
    {eS : SampledProofs.OneStageZk.Evidence}
    (hI : NciCheck.Protocol.protocol.Accepts pI cI eI) (hW : PouwCheck.protocol.Accepts pW cW eW)
    (hS : (SampledProofs.OneStageZk.protocol.onBlobs bs).Accepts pS cS eS)
    (dW : cW.digest = cI.digest) (dS : cS.digest = cI.digest) :
    ∃ pub prof, checkSameOf "verity/remote/v0"
        ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"]
        [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStageZk.protocol.onBlobs bs] ps
        (Composition.bind "verity/remote/v0" [("verity/inference-only/v0", cI.digest), ("verity/pouw/v1", cI.digest),
          ("verity/sampled-proofs/one-stage/v2", cI.digest)])
        (ps.map fun n => (n.id, n.params))
        [⟨NciCheck.Protocol.protocol, ⟨.ok pI, .ok cI, .ok eI⟩⟩, ⟨PouwCheck.protocol, ⟨.ok pW, .ok cW, .ok eW⟩⟩,
          ⟨SampledProofs.OneStageZk.protocol.onBlobs bs, ⟨.ok pS, .ok cS, .ok eS⟩⟩] = .accept pub prof := by
  obtain ⟨pubI, profI, hI⟩ := hI
  obtain ⟨pubW, profW, hW⟩ := hW
  obtain ⟨pubS, profS, hS⟩ := hS
  have hp' : ps.map (·.id) =
      [NciCheck.Protocol.protocol.id, PouwCheck.protocol.id, (SampledProofs.OneStageZk.protocol.onBlobs bs).id] := by
    rw [remote_ids, hp]
  exact ⟨_, _, checkSameOf_three (id := "verity/remote/v0") hp' ne_policy_pouw (ne_policy_stage bs) (ne_pouw_stage bs)
    hI hW hS dW dS⟩

/-- **Remote's check accepts the bytes of an honest run** (the converse of `remote_accept`): three cases, under the
parameters remote names for each, that commit to one record's bytes and that the policy, PoUW and the one-stage audit
decode to cases they accept. The commitment is the one binding the policy's record's digest for each component. -/
theorem remote_accepts {blobs : Array ByteArray} {nI nW nS : Named} {kI kW kS : Case}
    (hp : [nI.id, nW.id, nS.id] = ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"])
    (hfit : fitOf [kI, kW, kS] = [nI, nW, nS].map fun n => (n.id, n.params))
    (hW' : kW.commitment = kI.commitment) (hS' : kS.commitment = kI.commitment)
    {pI : NciCheck.Protocol.Params} {cI : NciCheck.Protocol.Commitment} {eI : NciCheck.Protocol.Evidence}
    {pW : PouwCheck.Params} {cW : PouwCheck.Registration} {eW : PouwCheck.Evidence}
    {pS : SampledProofs.OneStageZk.Params} {cS : SampledProofs.OneStage.Commitment}
    {eS : SampledProofs.OneStageZk.Evidence}
    (hdI : NciCheck.Protocol.protocol.decode nI.params.toUTF8 kI.commitment kI.evidence = ⟨.ok pI, .ok cI, .ok eI⟩)
    (hdW : PouwCheck.protocol.decode nW.params.toUTF8 kW.commitment kW.evidence = ⟨.ok pW, .ok cW, .ok eW⟩)
    (hdS : (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)).decode nS.params.toUTF8 kS.commitment
      kS.evidence = ⟨.ok pS, .ok cS, .ok eS⟩)
    (hI : NciCheck.Protocol.protocol.Accepts pI cI eI) (hW : PouwCheck.protocol.Accepts pW cW eW)
    (hS : (SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)).Accepts pS cS eS) :
    ∃ pub prof, (Core.Remote.protocol.on blobs).check [nI, nW, nS]
        (Composition.bind "verity/remote/v0" [("verity/inference-only/v0", cI.digest), ("verity/pouw/v1", cI.digest),
          ("verity/sampled-proofs/one-stage/v2", cI.digest)]) [kI, kW, kS] = .accept pub prof := by
  have hcI : NciCheck.Protocol.commitmentOf kI.commitment = .ok cI := congrArg Protocol.Decoded.commitment hdI
  have hcW : PouwCheck.registrationOf kI.commitment = .ok cW := by
    rw [← hW']; exact congrArg Protocol.Decoded.commitment hdW
  have hcS : SampledProofs.OneStage.commitmentOf kI.commitment = .ok cS := by
    rw [← hS']; exact congrArg Protocol.Decoded.commitment hdS
  obtain ⟨dW, dS⟩ := one_digest hcI hcW hcS
  have hds : decodeCases
      [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs)]
      [nI, nW, nS] [kI, kW, kS] =
      [⟨NciCheck.Protocol.protocol, ⟨.ok pI, .ok cI, .ok eI⟩⟩, ⟨PouwCheck.protocol, ⟨.ok pW, .ok cW, .ok eW⟩⟩,
        ⟨SampledProofs.OneStageZk.protocol.onBlobs (Blobs.ofArray blobs), ⟨.ok pS, .ok cS, .ok eS⟩⟩] := by
    simp only [decodeCases, List.zip_cons_cons, List.zip_nil_left, List.map_cons, List.map_nil, hdI, hdW, hdS]
  have hp' : [nI, nW, nS].map (·.id) =
      ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"] := by
    simp only [List.map_cons, List.map_nil]; exact hp
  rw [remote_check_decode, hfit, hds]
  exact remote_accepts_decoded hp' hI hW hS dW dS

end Properties.Remote.SecurityProofs
