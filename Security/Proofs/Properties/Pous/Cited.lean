import Properties.Pous.Cited
import Proofs.Properties.Pous.Run
import Proofs.Flock.Claims.Complete.Json.Hex
import Std.Data.TreeMap.Raw.Lemmas

/-!
# Proofs of the cited run's properties (`Properties.Pous.Cited`): `rootDecodes`, `citedRunAccepted` and `citedAnswerAccepted`

`rootDecodes`: an accepted `C*` decodes to the committed `W` (`acceptsDecodes`), so it has `q`'s shape, and has the
published root (`publishesRoot`), so its leaves and those of a `C*'` of that shape at that root have one root. Over
leaves of 64 bytes, one root means the same leaves unless two inner nodes' inputs collide (`mth_binds`); one leaf means
one leaf input, so one row digest, unless the two inputs collide; and one row digest means one block, the prefix
carrying the block's length, unless the two rows' inputs collide. So `C*'` is `C*` block for block, and decodes as it
does. `citedRunAccepted`: the cited form's acceptance is the citation and `Run.check` (`acceptsCited_deployed`), and
`Accepted` is the setup check's acceptance and `Run.check`. `citedAnswerAccepted`: an answer of the setup protocol that
accepts is its check's acceptance of what the three inputs decode to, publishing `publicJson` of the check's public
outputs (`answer_accept`); the citation reads those outputs back as `pub` (`cite_ok`), and `Public.ofJson` reads back
exactly what `toJson` writes (`ofJson_toJson`), so they are `pub`.
-/

namespace Pous.Cited.SecurityProofs

open Flock PousCheck PousCheck.Properties
open PousCheck.Properties.SecurityProofs (append_cancel mth_binds leaves_entry root_mth acceptsDecodes publishesRoot)
open Pous.EndToEnd (leafInput blockInput)
open Pous.EndToEnd.SecurityProofs (flatten_length flatten_getElem? getElem!_of_some)

/-! ## `C*`'s leaves at any parameters -/

theorem shapedAt_of {q : P2.Params} {c : SetupProtocol.Commitment} {C : Array (Array ByteArray)}
    (h : DecodesToCommitment q c C) : ShapedAt q C := by
  obtain ⟨W, ⟨hCs, -, hseg⟩, -⟩ := h
  refine ⟨hCs, fun s hs => ⟨(hseg s hs).1, fun v hv => ?_⟩⟩
  obtain ⟨-, -, -, -, -, -, hb, -⟩ := (hseg s hs).2 v hv
  exact hb

theorem leaves_lists_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray) :
    ∀ l ∈ C.toList.mapIdx (fun s seg => seg.toList.mapIdx fun v block => Merkle.packageLeaf salt s v block),
      l.length = q.blocksPerSegment := by
  intro l hl
  obtain ⟨s, hs, rfl⟩ := List.mem_iff_getElem.1 hl
  simp only [List.getElem_mapIdx, List.length_mapIdx, Array.length_toList]
  simp only [List.length_mapIdx, Array.length_toList] at hs
  exact (hS.2 s hs).1

theorem leaves_size_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray) :
    (leaves salt C).size = q.blocks := by
  rw [leaves, List.size_toArray, flatten_length q.blocksPerSegment _ (leaves_lists_at hS salt)]
  simp [hS.1, P2.Params.blocks]

theorem leaves_getElem?_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray)
    (s v : Nat) (hv : v < q.blocksPerSegment) :
    (leaves salt C)[s * q.blocksPerSegment + v]? = C[s]?.bind fun seg => seg[v]?.map (Merkle.packageLeaf salt s v) := by
  rw [leaves, List.getElem?_toArray, flatten_getElem? q.blocksPerSegment _ (leaves_lists_at hS salt) s v hv]
  simp only [List.getElem?_mapIdx, Array.getElem?_toList]
  cases C[s]? <;> simp [List.getElem?_mapIdx]

theorem leaves_getElem!_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray)
    {s v : Nat} (hs : s < C.size) (hv : v < C[s].size) :
    (leaves salt C)[s * q.blocksPerSegment + v]! = Merkle.packageLeaf salt s v C[s][v] := by
  have hv' : v < q.blocksPerSegment := (hS.2 s hs).1 ▸ hv
  apply getElem!_of_some
  rw [leaves_getElem?_at hS salt s v hv', Array.getElem?_eq_getElem hs]
  simp [Array.getElem?_eq_getElem hv]

/-! ## The stores' strings -/

theorem mem_store_node {q : P2.Params} {salt : ByteArray} {C : Array (Array ByteArray)} {x : ByteArray}
    (hS : ShapedAt q C) (hx : x ∈ nodeInputs (leaves salt C) 0 q.blocks) : x ∈ storeStrings salt C := by
  unfold storeStrings
  dsimp only
  rw [leaves_size_at hS salt]
  exact List.mem_append_left _ hx

theorem mem_store_block (salt : ByteArray) (C : Array (Array ByteArray)) {s v : Nat} (hs : s < C.size)
    (hv : v < C[s].size) {z : ByteArray}
    (hz : z = leafInput salt s v (Merkle.rowDigest C[s][v]) ∨ z = blockInput C[s][v]) : z ∈ storeStrings salt C := by
  unfold storeStrings
  dsimp only
  refine List.mem_append_right _ (List.mem_flatten.2 ⟨(C[s].toList.mapIdx fun v block =>
      [leafInput salt s v (Merkle.rowDigest block), blockInput block]).flatten, ?_, ?_⟩)
  · exact List.mem_mapIdx.2 ⟨s, by simpa using hs, by simp⟩
  · refine List.mem_flatten.2 ⟨[leafInput salt s v (Merkle.rowDigest C[s][v]), blockInput C[s][v]], ?_, ?_⟩
    · exact List.mem_mapIdx.2 ⟨v, by simpa using hv, by simp⟩
    · rcases hz with rfl | rfl <;> simp

/-! ## The theorems -/

theorem rootDecodes : RootDecodes := by
  intro q c C pub prof hacc C' hS' hroot'
  have hD := acceptsDecodes q c C ⟨pub, prof, hacc⟩
  have hP := publishesRoot q c C pub prof hacc
  have hS := shapedAt_of hD
  by_cases hB : StoreBreak c.salt C C'
  · exact .inr hB
  left
  have e1 := root_mth hP.1
  rw [leaves_size_at hS] at e1
  have e2 := root_mth hroot'
  rw [leaves_size_at hS'] at e2
  rcases mth_binds _ _ (leaves_entry c.salt C) (leaves_entry c.salt C') q.blocks 0
      (by rw [leaves_size_at hS]; omega) (by rw [leaves_size_at hS']; omega) (e1.trans e2.symm) with
    hl | ⟨x, hx, y, hy, hxy⟩
  swap
  · exact absurd ⟨x, mem_store_node hS hx, y, mem_store_node hS' hy, hxy⟩ hB
  have hblock : ∀ s (hs : s < C.size) (hs' : s < C'.size) v (hv : v < C[s].size) (hv' : v < C'[s].size),
      C[s][v] = C'[s][v] := by
    intro s hs hs' v hv hv'
    have hsq : s < q.segments := hS.1 ▸ hs
    have hvq : v < q.blocksPerSegment := (hS.2 s hs).1 ▸ hv
    have hidx : s * q.blocksPerSegment + v < q.blocks := by
      have h1 : s * q.blocksPerSegment + v < (s + 1) * q.blocksPerSegment := by rw [Nat.succ_mul]; omega
      exact Nat.lt_of_lt_of_le h1 (Nat.mul_le_mul_right _ hsq)
    have hlv := hl (s * q.blocksPerSegment + v) (Nat.zero_le _) (by omega)
    rw [leaves_getElem!_at hS c.salt hs hv, leaves_getElem!_at hS' c.salt hs' hv'] at hlv
    have hrow : Merkle.rowDigest C[s][v] = Merkle.rowDigest C'[s][v] := by
      by_cases hli : leafInput c.salt s v (Merkle.rowDigest C[s][v]) = leafInput c.salt s v (Merkle.rowDigest C'[s][v])
      · exact (append_cancel hli rfl).2
      · exact absurd ⟨_, mem_store_block c.salt C hs hv (.inl rfl), _, mem_store_block c.salt C' hs' hv' (.inl rfl),
          hli, hlv⟩ hB
    by_cases hbi : blockInput C[s][v] = blockInput C'[s][v]
    · have hsz : (HmRow.rowPrefixV2 (8 * C[s][v].size)).size = (HmRow.rowPrefixV2 (8 * C'[s][v].size)).size := by
        rw [(hS.2 s hs).2 v hv, (hS'.2 s hs').2 v hv']
      exact (append_cancel hbi hsz).2
    · exact absurd ⟨_, mem_store_block c.salt C hs hv (.inr rfl), _, mem_store_block c.salt C' hs' hv' (.inr rfl),
        hbi, hrow⟩ hB
  have hCC : C' = C := by
    refine Array.ext (hS'.1.trans hS.1.symm) fun s hs' hs => ?_
    refine Array.ext ((hS'.2 s hs').1.trans (hS.2 s hs).1.symm) fun v hv' hv => ?_
    exact (hblock s hs hs' v hv hv').symm
  rw [hCC]
  exact hD

theorem citedRunAccepted : CitedRunAccepted := by
  intro trusted c C pub prof s digests r sid hset hcited
  cases prof
  exact ⟨hset, ((RunProtocol.acceptsCited_deployed trusted s pub digests r sid).1 hcited).2⟩

/-! ## The cited answer -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error _ => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem exactly_ok {α : Type} {f : Lean.Json → V α} {enc : α → Lean.Json} {j : Lean.Json} {b : α}
    (h : MemoryChallenger.exactly f enc j = .ok b) : f j = .ok b := by
  obtain ⟨a, hf, h⟩ := bind_ok h
  obtain ⟨x, -, h⟩ := bind_ok h
  obtain ⟨y, -, h⟩ := bind_ok h
  dsimp only at h
  split at h
  · obtain ⟨u, hu, -⟩ := bind_ok h
    cases hu
  · cases h
    exact hf

theorem getObjVal?_mkObj {l : List (String × Lean.Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Lean.Json} (hm : (k, v) ∈ l) : (Lean.Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Lean.Json.mkObj, Lean.Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

theorem ok_bind {α β : Type} (a : α) (f : α → V β) : (Except.ok a >>= f) = f a := rfl

theorem nat_num (j : Lean.Json) (k c : String) (x : Nat) (h : j.getObjVal? k = .ok (.num x)) :
    J.nat j k c = .ok x := by
  simp only [J.nat, J.get, h]
  rfl

theorem str_str (j : Lean.Json) (k c s : String) (h : j.getObjVal? k = .ok (.str s)) : J.str j k c = .ok s := by
  simp only [J.str, J.get, h]
  rfl

/-- `Public.ofJson` reads back exactly what `Public.toJson` writes. -/
theorem ofJson_toJson (p : SetupProtocol.Public) : SetupProtocol.Public.ofJson p.toJson = .ok p := by
  obtain ⟨b, c, r, s⟩ := p
  have hb := nat_num (SetupProtocol.Public.toJson ⟨b, c, r, s⟩) "blocks" "the setup's public outputs" b
    (getObjVal?_mkObj (by simp) (by simp))
  have hc := str_str (SetupProtocol.Public.toJson ⟨b, c, r, s⟩) "commitment" "the setup's public outputs" (hexOf c)
    (getObjVal?_mkObj (by simp) (by simp))
  have hr := str_str (SetupProtocol.Public.toJson ⟨b, c, r, s⟩) "root" "the setup's public outputs" (hexOf r)
    (getObjVal?_mkObj (by simp) (by simp))
  have hs := str_str (SetupProtocol.Public.toJson ⟨b, c, r, s⟩) "salt" "the setup's public outputs" (hexOf s)
    (getObjVal?_mkObj (by simp) (by simp))
  simp only [SetupProtocol.Public.ofJson, hb, hc, hr, hs, ok_bind, FlockComplete.JsonRT.unhex_hexOf]
  rfl

/-- An answer whose public outputs are `toJson pub'` and which the citation reads as `pub` published `pub'`. -/
theorem published_eq {a : Core.Answer} {pub pub' : SetupProtocol.Public}
    (ha : a.published = some (SetupProtocol.Public.toJson pub')) (h : RunProtocol.publishedOf a = some pub) :
    pub = pub' := by
  unfold RunProtocol.publishedOf at h
  rw [ha] at h
  change (MemoryChallenger.exactly SetupProtocol.Public.ofJson SetupProtocol.Public.toJson
    (SetupProtocol.Public.toJson pub')).toOption = some pub at h
  cases he : MemoryChallenger.exactly SetupProtocol.Public.ofJson SetupProtocol.Public.toJson
      (SetupProtocol.Public.toJson pub') with
  | error e => rw [he] at h; cases h
  | ok x =>
    rw [he] at h
    cases h
    have := (exactly_ok he).symm.trans (ofJson_toJson pub')
    cases this
    rfl

/-- What a citation the run's check accepts says of its answer: an acceptance, at the run's parameters, whose public
outputs read back as `pub`. -/
theorem cite_ok {pt : Run.Point} {pub : SetupProtocol.Public} {s : RunProtocol.SetupRef}
    (h : RunProtocol.cite pt pub s = .ok ()) :
    s.answer.verdict = .accept ∧ s.params = pt.params ∧ RunProtocol.publishedOf s.answer = some pub := by
  unfold RunProtocol.cite at h
  by_cases h1 : s.record.isEmpty = true
  · rw [ite_eq_left h1] at h; cases h
  rw [ite_eq_right h1] at h
  by_cases h2 : s.answer.protocol ≠ SetupProtocol.ID
  · rw [ite_eq_left h2] at h; cases h
  rw [ite_eq_right h2] at h
  by_cases h3 : s.answer.verdict ≠ .accept
  · rw [ite_eq_left h3] at h; cases h
  rw [ite_eq_right h3] at h
  by_cases h4 : s.params ≠ pt.params
  · rw [ite_eq_left h4] at h; cases h
  rw [ite_eq_right h4] at h
  by_cases h5 : s.answer.digest ≠ some (pub.commitment ++ pub.salt) ∨ RunProtocol.publishedOf s.answer ≠ some pub
  · rw [ite_eq_left h5] at h; cases h
  exact ⟨Classical.not_not.1 h3, Classical.not_not.1 h4, Classical.not_not.1 (not_or.1 h5).2⟩

/-- An answer that accepts, of bytes whose parameters decode to `p`: the commitment and evidence decode, the check
accepts them at `p`, and the answer's public outputs are `publicJson` of the check's. -/
theorem answer_accept {P : Core.Protocol} {pb cb eb : ByteArray} {p : P.Params}
    (hp : (Core.canonicalJson pb >>= P.params) = .ok p) (h : (P.answer pb cb eb).verdict = .accept) :
    ∃ c e pub prof, P.commitment cb = .ok c ∧ P.evidence eb = .ok e ∧ P.check p c e = .accept pub prof ∧
      (P.answer pb cb eb).published = some (P.publicJson pub) := by
  unfold Core.Protocol.answer at h ⊢
  rw [hp] at h ⊢
  cases hc : P.commitment cb with
  | error _ => rw [hc] at h; cases h
  | ok c =>
    rw [hc] at h
    cases he : P.evidence eb with
    | error _ => simp only [he] at h; cases h
    | ok e =>
      simp only [he] at h
      cases hch : P.check p c e with
      | accept pub prof => exact ⟨c, e, pub, prof, rfl, rfl, hch, by simp only [hch]⟩
      | reject _ => rw [hch] at h; cases h
      | undecided _ => rw [hch] at h; cases h

theorem citedAnswerAccepted : CitedAnswerAccepted := by
  intro trusted s pub digests r sid pb cb eb hans hpar hcited
  obtain ⟨hcite, hrun⟩ := (RunProtocol.acceptsCited_deployed trusted s pub digests r sid).1 hcited
  obtain ⟨hv, hq, hpub⟩ := cite_ok hcite
  rw [hans] at hv hpub
  obtain ⟨c, C, pub', prof, hc, hC, hchk, hpj⟩ := answer_accept hpar hv
  cases published_eq hpj hpub
  cases prof
  rw [hq] at hchk
  exact ⟨c, C, hc, hC, hchk, hrun⟩

end Pous.Cited.SecurityProofs
