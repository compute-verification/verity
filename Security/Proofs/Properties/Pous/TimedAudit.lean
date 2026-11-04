import Properties.Pous.TimedAudit
import Proofs.Properties.NetworkCertifier.Bytes

/-!
# The timed audit's property, proved

`timedAuditAccepts : TimedAuditAccepts` (`Properties.Pous.TimedAudit`). An accepted verdict came from `verdictOf`
(`verdict_ok`), which read the record's fields, refused nothing it gates on strictly and accepted every round of every
audit (`verdictOf_accepted`, `audit_ok`). Two trees over leaves of 64 bytes with one root have the same leaves unless
two of their inner nodes' inputs collide (`mth_binds`), which ties each answer's leaf to `C*`'s.
-/

namespace PousCheck.Properties.SecurityProofs

open Lean (Json)
open Flock PousCheck.Properties

local instance instLawfulBEqByteArray_properties : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := Properties.NetworkCertifier.Bytes.hash_size m

theorem append_cancel {a b c d : ByteArray} (h : a ++ b = c ++ d) (hs : a.size = c.size) : a = c ∧ b = d := by
  have := congrArg ByteArray.data h
  simp only [ByteArray.data_append] at this
  obtain ⟨h1, h2⟩ := Array.append_inj this hs
  exact ⟨ByteArray.ext h1, ByteArray.ext h2⟩

theorem mth_size (L : Array ByteArray) (hL : ∀ i (h : i < L.size), L[i].size = 64) :
    ∀ n lo, 1 ≤ n → lo + n ≤ L.size → (Merkle.mth L lo n).size = 64 := by
  intro n lo h1 h2
  rw [Merkle.mth]
  split
  · have : n = 1 := by omega
    subst this
    rw [Array.getElem?_eq_getElem (by omega)]
    exact hL lo (by omega)
  · exact hash_size _

theorem nodeInputs_eq (L : Array ByteArray) (lo n : Nat) (h : 2 ≤ n) :
    nodeInputs L lo n = (ByteArray.mk #[1] ++ Merkle.mth L lo (Merkle.split n) ++
        Merkle.mth L (lo + Merkle.split n) (n - Merkle.split n)) ::
      (nodeInputs L lo (Merkle.split n) ++ nodeInputs L (lo + Merkle.split n) (n - Merkle.split n)) := by
  rw [nodeInputs]
  simp only [show ¬ n ≤ 1 by omega, ↓reduceDIte]

theorem mth_node (L : Array ByteArray) (lo n : Nat) (h : 2 ≤ n) :
    Merkle.mth L lo n = Sha512.hash (ByteArray.mk #[1] ++ Merkle.mth L lo (Merkle.split n) ++
        Merkle.mth L (lo + Merkle.split n) (n - Merkle.split n)) := by
  rw [Merkle.mth]
  simp only [show ¬ n ≤ 1 by omega, ↓reduceDIte]
  rfl

/-- Two trees of `n` leaves from `lo` with one root: their leaves agree, or two of their inner nodes' inputs collide. -/
theorem mth_binds (L1 L2 : Array ByteArray) (h1 : ∀ i (h : i < L1.size), L1[i].size = 64)
    (h2 : ∀ i (h : i < L2.size), L2[i].size = 64) :
    ∀ n lo, lo + n ≤ L1.size → lo + n ≤ L2.size → Merkle.mth L1 lo n = Merkle.mth L2 lo n →
      (∀ i, lo ≤ i → i < lo + n → L1[i]! = L2[i]!) ∨
        ∃ x ∈ nodeInputs L1 lo n, ∃ y ∈ nodeInputs L2 lo n, x ≠ y ∧ Sha512.hash x = Sha512.hash y := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hl1 hl2 he
  by_cases hn : n ≤ 1
  · left
    intro i hi1 hi2
    have hi : i = lo := by omega
    subst hi
    unfold Merkle.mth at he
    simp only [hn, ↓reduceDIte] at he
    rw [Array.getElem?_eq_getElem (by omega), Array.getElem?_eq_getElem (by omega)] at he
    simp only [Option.getD_some] at he
    rw [getElem!_pos L1 i (by omega), getElem!_pos L2 i (by omega)]
    exact he
  have hn2 : 2 ≤ n := by omega
  have hk := Merkle.split_lt hn2
  have hk0 := Merkle.split_pos n
  rw [mth_node L1 lo n hn2, mth_node L2 lo n hn2] at he
  rw [nodeInputs_eq L1 lo n hn2, nodeInputs_eq L2 lo n hn2]
  set k := Merkle.split n
  by_cases hx : ByteArray.mk #[1] ++ Merkle.mth L1 lo k ++ Merkle.mth L1 (lo + k) (n - k) =
      ByteArray.mk #[1] ++ Merkle.mth L2 lo k ++ Merkle.mth L2 (lo + k) (n - k)
  · have s1 : (ByteArray.mk #[1] ++ Merkle.mth L1 lo k).size = (ByteArray.mk #[1] ++ Merkle.mth L2 lo k).size := by
      simp only [ByteArray.size_append, mth_size L1 h1 k lo (by omega) (by omega),
        mth_size L2 h2 k lo (by omega) (by omega)]
    obtain ⟨ha, hb⟩ := append_cancel hx s1
    obtain ⟨-, ha⟩ := append_cancel ha rfl
    rcases ih k hk lo (by omega) (by omega) ha with hl | ⟨x, hx1, y, hy1, hxy⟩
    · rcases ih (n - k) (by omega) (lo + k) (by omega) (by omega) hb with hr | ⟨x, hx1, y, hy1, hxy⟩
      · left
        intro i hi1 hi2
        by_cases hik : i < lo + k
        · exact hl i hi1 hik
        · exact hr i (by omega) (by omega)
      · exact .inr ⟨x, by simp [hx1], y, by simp [hy1], hxy⟩
    · exact .inr ⟨x, by simp [hx1], y, by simp [hy1], hxy⟩
  · exact .inr ⟨_, List.mem_cons_self, _, List.mem_cons_self, hx, he⟩


/-! ## The record's readers -/

theorem need_ok {b : Bool} {p : String} {u : Unit} (h : TimedAudit.need b p = .ok u) : b = true := by
  unfold TimedAudit.need at h; split at h <;> simp_all

theorem field_ok {j : Json} {k p : String} {v : Json} (h : TimedAudit.field j k p = .ok v) :
    j.getObjVal? k = .ok v := by
  unfold TimedAudit.field at h
  split at h
  · cases h; assumption
  · contradiction

theorem natOf_ok {j : Json} {p : String} {n : Nat} (h : TimedAudit.natOf j p = .ok n) : j.getNat? = .ok n := by
  unfold TimedAudit.natOf at h
  split at h
  · cases h; assumption
  · contradiction

theorem strOf_ok {j : Json} {p : String} {t : String} (h : TimedAudit.strOf j p = .ok t) : j = .str t := by
  unfold TimedAudit.strOf at h
  split at h
  · rename_i hs; cases h; cases j <;> simp [Json.getStr?] at hs; cases hs; rfl
  · contradiction

theorem hexOf_ok {j : Json} {p : String} {b : ByteArray} (h : TimedAudit.hexOf' j p = .ok b) :
    ∃ t, j = .str t ∧ unhex t = .ok b := by
  unfold TimedAudit.hexOf' at h
  cases hs : TimedAudit.strOf j p with
  | error e => simp [hs, bind, Except.bind] at h
  | ok t =>
    simp only [hs, bind, Except.bind] at h
    split at h
    · rename_i hu; cases h; exact ⟨t, strOf_ok hs, hu⟩
    · contradiction

theorem hexAt_of {doc : Json} {k p : String} {v : Json} {b : ByteArray} (h1 : TimedAudit.field doc k p = .ok v)
    (h2 : TimedAudit.hexOf' v p = .ok b) : HexAt doc k b := by
  obtain ⟨t, rfl, ht⟩ := hexOf_ok h2
  exact ⟨t, field_ok h1, ht⟩

theorem natAt_of {doc : Json} {k p : String} {v : Json} {n : Nat} (h1 : TimedAudit.field doc k p = .ok v)
    (h2 : TimedAudit.natOf v p = .ok n) : NatAt doc k n := ⟨v, field_ok h1, natOf_ok h2⟩

/-! ## Loops -/

theorem list_mapM_ok {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (bs : List β), l.mapM f = .ok bs →
      bs.length = l.length ∧ ∀ a ∈ l, ∃ b ∈ bs, f a = .ok b := by
  intro l
  induction l with
  | nil => intro bs h; simp [List.mapM_nil, pure, Except.pure] at h; subst h; simp
  | cons a as ih =>
    intro bs h
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error e => simp [ha, bind, Except.bind] at h
    | ok b =>
      cases hr : as.mapM f with
      | error e => simp [ha, hr, bind, Except.bind] at h
      | ok bs' =>
        simp [ha, hr, bind, Except.bind, pure, Except.pure] at h
        subst h
        obtain ⟨hl, hm⟩ := ih bs' hr
        refine ⟨by simp [hl], fun x hx => ?_⟩
        rcases List.mem_cons.1 hx with rfl | hx
        · exact ⟨b, List.mem_cons_self, ha⟩
        · obtain ⟨y, hy, hfy⟩ := hm x hx
          exact ⟨y, List.mem_cons_of_mem _ hy, hfy⟩

theorem array_mapM_ok {α β ε : Type} (f : α → Except ε β) (l : Array α) (bs : Array β)
    (h : l.mapM f = .ok bs) :
    bs.size = l.size ∧ ∀ a ∈ l, ∃ b ∈ bs, f a = .ok b := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : l.toList.mapM f with
  | error e => simp [hl, Functor.map, Except.map] at h
  | ok xs =>
    simp [hl, Functor.map, Except.map] at h
    subst h
    obtain ⟨h1, h2⟩ := list_mapM_ok f l.toList xs hl
    refine ⟨by simpa using h1, fun a ha => ?_⟩
    obtain ⟨b, hb, hfb⟩ := h2 a (by simpa using ha)
    exact ⟨b, by simpa using hb, hfb⟩

/-- A loop in `Except` that only pushes, one value a step: each element's step pushed a value it vouches for. -/
theorem forIn_push {α β ε : Type} (P : α → β → Prop) (f : α → Array β → Except ε (ForInStep (Array β)))
    (hf : ∀ a r r', f a r = .ok r' → ∃ x, r' = .yield (r.push x) ∧ P a x) :
    ∀ (l : List α) (init out : Array β), forIn l init f = .ok out →
      (∀ x ∈ init, x ∈ out) ∧ ∀ a ∈ l, ∃ x ∈ out, P a x := by
  intro l
  induction l with
  | nil =>
    intro init out h
    simp only [List.forIn_nil] at h
    cases h
    exact ⟨fun x hx => hx, fun a ha => by simp at ha⟩
  | cons a as ih =>
    intro init out h
    rw [List.forIn_cons] at h
    cases hfa : f a init with
    | error e => simp [hfa, bind, Except.bind] at h
    | ok r' =>
      obtain ⟨x, rfl, hx⟩ := hf a init r' hfa
      simp only [hfa, bind, Except.bind] at h
      obtain ⟨h1, h2⟩ := ih (init.push x) out h
      refine ⟨fun y hy => h1 y (Array.mem_push.2 (.inl hy)), fun b hb => ?_⟩
      rcases List.mem_cons.1 hb with rfl | hb
      · exact ⟨x, h1 x (Array.mem_push.2 (.inr rfl)), hx⟩
      · exact h2 b hb

/-! ## One audit -/

/-- Answer `j` of an audit keyed by `source`, as `audit` judged it `x`. -/
def AnswerFacts (s : P2.Params) (salt root : ByteArray) (leaves : Array ByteArray) (limit : Nat)
    (source : ByteArray) (answers : Array Json) (j : Nat) (x : Bool × Bool) : Prop :=
  ∃ (hj : j < answers.size) (block : ByteArray) (elapsed : Nat),
    NatAt answers[j] "index" (challenged s salt root source j) ∧ NatAt answers[j] "elapsed_ns" elapsed ∧
    HexAt answers[j] "block" block ∧
    x = (block.size == s.w / 8 && Merkle.packageLeaf salt (challenged s salt root source j / s.blocksPerSegment)
      (challenged s salt root source j % s.blocksPerSegment) block == leaves[challenged s salt root source j]!,
      decide (elapsed ≤ limit))

theorem audit_ok {s : P2.Params} {salt root : ByteArray} {leaves : Array ByteArray} {k limit : Nat} {rec : Json}
    {out : Array (Bool × Bool)} (h : TimedAudit.audit s salt root leaves k limit rec = .ok out) :
    ∃ (source : ByteArray) (answers : Array Json), HexAt rec "source" source ∧ 0 < source.size ∧
      rec.getObjVal? "answers" = .ok (.arr answers) ∧ answers.size = k ∧
      ∀ j < k, ∃ x ∈ out, AnswerFacts s salt root leaves limit source answers j x := by
  unfold TimedAudit.audit at h
  simp only [bind, Except.bind] at h
  iterate 6 (split at h <;> try contradiction)
  rename_i _ vsrc hsrc _ source hsource _ _ hpos _ ans hans _ answers hget _ answers' hpure
  cases hpure
  split at h
  · contradiction
  rename_i _ out' hloop
  split at h
  · contradiction
  rename_i _ _ hlong
  cases h
  have hk : answers.size = k := by simpa using need_ok hlong
  have hans' : rec.getObjVal? "answers" = .ok (.arr answers) := by
    rw [field_ok hans]; cases ans <;> simp [Json.getArr?] at hget; cases hget; rfl
  refine ⟨source, answers, hexAt_of hsrc hsource, by simpa using need_ok hpos, hans', hk, fun j hj => ?_⟩
  refine (forIn_push (AnswerFacts s salt root leaves limit source answers) _ ?_ _ _ _ hloop).2 j
    (List.mem_range.2 hj)
  intro a r r' hr
  iterate 8 (split at hr <;> try contradiction)
  rename_i _ _ hshort _ vi hfi _ index hindex _ ve hfe _ elapsed helapsed _ vb hfb _ block hblock _ _ hidx
  cases hr
  have ha : a < answers.size := by simpa using need_ok hshort
  have hi : index = challenged s salt root source a := by simpa [challenged] using need_ok hidx
  subst hi
  rw [getElem!_pos answers a ha] at hfi hfe hfb
  exact ⟨_, rfl, ha, block, elapsed, natAt_of hfi hindex, natAt_of hfe helapsed, hexAt_of hfb hblock, rfl⟩

/-! ## The certificate's gate -/

/-- A store and deadlines the gate refuses nothing of: the certificate's point, `k` and deadlines within it, and no gap
exactly when `W` has the certificate's 8 segments. -/
theorem gate_refused_nil {s : P2.Params} {k delta rtt : Nat} (h : (Certificate.gate s k delta rtt).2.1 = []) :
    (s.w, s.c0, s.blocksPerSegment) = Certificate.POINT ∧ Certificate.CERT_K ≤ k ∧
      delta = Certificate.DELTA_NS ∧ rtt ≤ Certificate.RTT_ALLOWANCE_CAP_NS ∧
      ((Certificate.gate s k delta rtt).2.1 ++ (Certificate.gate s k delta rtt).2.2 = [] ↔ s.segments = 8) := by
  unfold Certificate.gate Certificate.p2 at *
  by_cases hp : (s.w, s.c0, s.blocksPerSegment) = Certificate.POINT
  · simp only [hp, beq_self_eq_true, ite_true] at h ⊢
    simp only [Certificate.CapacityBound.refusals, Certificate.deadlineRefusals,
      Certificate.CapacityBound.effectiveRounds, List.append_eq_nil_iff, ite_eq_right_iff, reduceCtorEq, imp_false,
      not_lt] at h
    obtain ⟨⟨hk, hd⟩, hδ, hr⟩ := h
    have hd' : ¬ (delta + rtt) / Certificate.ROOT_CALL_NS > 0 := fun hc => by simp [hc] at hd
    have hδ' : delta = Certificate.DELTA_NS := by simpa using hδ
    have hlt : Certificate.DELTA_NS + rtt < Certificate.ROOT_CALL_NS := by
      unfold Certificate.ROOT_CALL_NS at hd'; rw [← hδ']; unfold Certificate.ROOT_CALL_NS; omega
    refine ⟨trivial, hk, hδ', hr, ?_⟩
    simp [Certificate.CapacityBound.refusals, Certificate.deadlineRefusals, Certificate.CapacityBound.effectiveRounds,
      Certificate.CERT_BLOCKS, Certificate.POINT, hlt, hδ', Nat.not_lt.2 hk, Nat.not_lt.2 hr]
  · simp [hp] at h

/-! ## The verdict -/

theorem fault_null (t : String) : (Json.mkObj [("fault", .str t)]).getObjValD "accepted" = .null := rfl

theorem refused_null (t : String) : (Json.mkObj [("refused", .str t)]).getObjValD "accepted" = .null := rfl

theorem verdict_accepted (a b c d e f g : Json) :
    (Json.mkObj [("k", a), ("limit_ns", b), ("gaps", c), ("audits", d), ("audits_accepted", e), ("accepted", f),
      ("profile", g)]).getObjValD "accepted" = f := rfl

theorem verdict_ok {ready : Json} {leafBytes : ByteArray} {doc : Json} (h : TimedAuditAccepted ready leafBytes doc) :
    ∃ v, TimedAudit.verdictOf ready leafBytes doc = .ok v ∧ v.getObjValD "accepted" = .bool true := by
  unfold TimedAuditAccepted TimedAudit.verdict at h
  split at h
  · exact ⟨_, ‹_›, h⟩
  · rw [fault_null] at h; cases h

/-- What an accepted `verdictOf` read and judged. -/
theorem verdictOf_accepted {ready : Json} {leafBytes : ByteArray} {doc v : Json}
    (h : TimedAudit.verdictOf ready leafBytes doc = .ok v) (hv : v.getObjValD "accepted" = .bool true) :
    ∃ (s : P2.Params) (salt root : ByteArray) (delta rtt k : Nat) (strict : Bool) (records : Array Json)
      (audits : Array (Array (Bool × Bool))),
      TimedAudit.store ready = .ok (s, salt) ∧ leafBytes.size = s.blocks * Merkle.DIGEST ∧
      HexAt doc "root" root ∧ NatAt doc "deadline_ns" delta ∧ NatAt doc "rtt_allowance_ns" rtt ∧
      NatAt doc "limit_ns" (delta + rtt) ∧ doc.getObjVal? "strict" = .ok (.bool strict) ∧ challengesOf s doc k ∧
      doc.getObjVal? "records" = .ok (.arr records) ∧ Merkle.root (fileLeaves s.blocks leafBytes) = some root ∧
      (strict = true → (Certificate.gate s k delta rtt).2.1 = []) ∧
      records.mapM (TimedAudit.audit s salt root (fileLeaves s.blocks leafBytes) k (delta + rtt)) = .ok audits ∧
      0 < audits.size ∧ ∀ a ∈ audits, ∀ x ∈ a, x = (true, true) := by
  unfold TimedAudit.verdictOf at h
  cases hst : TimedAudit.store ready with
  | error e => simp [hst, bind, Except.bind] at h
  | ok p =>
    obtain ⟨s, salt⟩ := p
    simp only [hst, bind, Except.bind] at h
    iterate 14 (split at h <;> try contradiction)
    all_goals (iterate 3 (split at h <;> try contradiction))
    all_goals (try (cases h; rw [refused_null] at hv; simp at hv))
    all_goals (split at h <;> try contradiction)
    all_goals (try (cases h; rw [refused_null] at hv; simp at hv))
    all_goals (try (split at h; contradiction))
    all_goals
      rename_i _ _ hleaves _ vroot hroot _ root hhex _ delta hdelta _ rtt hrtt _ limit hlimit _ vstrict hstrict
        _ b hb _ strict hpure1 _ vrec hrec _ a harr _ records hpure2 _ vk hk _ k? hk2 _ _ hrootEq _ _ hlimEq hif
        _ audits hmap
      cases h
      rw [verdict_accepted] at hv
      cases hpure1
      cases hpure2
      have hlim : limit = delta + rtt := by simpa using need_ok hlimEq
      subst hlim
      have hstrict' : doc.getObjVal? "strict" = .ok (.bool b) := by
        rw [field_ok hstrict]; cases vstrict <;> simp [Json.getBool?] at hb; cases hb; rfl
      have hrecords : doc.getObjVal? "records" = .ok (.arr a) := by
        rw [field_ok hrec]; cases vrec <;> simp [Json.getArr?] at harr; cases harr; rfl
      simp at hif hv
      have hk' : challengesOf s doc (k?.getD
          ((Option.map (fun x => x.k) (Certificate.p2 s.w s.c0 s.blocksPerSegment)).getD TimedAudit.CHALLENGES)) := by
        unfold challengesOf; rw [hk]
        first
        | (cases hk2; rfl)
        | (cases hn : TimedAudit.natOf vk "record" with
            | error e => simp [hn, Functor.map, Except.map] at hk2
            | ok n => simp [hn, Functor.map, Except.map] at hk2; subst hk2; exact natOf_ok hn)
      refine ⟨s, salt, root, delta, rtt, _, b, a, audits, rfl, by simpa using need_ok hleaves, hexAt_of hroot hhex,
        ?_, ?_, ?_, hstrict', hk', hrecords, ?_, hif, hmap, ?_, ?_⟩
      · split at hdelta
        · contradiction
        · rename_i hf; exact natAt_of hf hdelta
      · split at hrtt
        · contradiction
        · rename_i hf; exact natAt_of hf hrtt
      · split at hlimit
        · contradiction
        · rename_i hf; exact natAt_of hf hlimit
      · have := need_ok hrootEq
        simp only [beq_iff_eq] at this
        exact this.symm
      · exact Array.size_pos_iff.2 hv.1
      · intro x hx y hy
        obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
        obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hy
        obtain ⟨h1, h2⟩ := hv.2 i hi j hj
        exact Prod.ext h1 h2

/-! ## The theorem -/

theorem fileLeaves_size (blocks : Nat) (leafBytes : ByteArray) : (fileLeaves blocks leafBytes).size = blocks := by
  simp [fileLeaves]

theorem fileLeaves_entry {blocks : Nat} {leafBytes : ByteArray} (h : leafBytes.size = blocks * Merkle.DIGEST) :
    ∀ i (hi : i < (fileLeaves blocks leafBytes).size), (fileLeaves blocks leafBytes)[i].size = 64 := by
  intro i hi
  simp only [fileLeaves, Array.size_map, Array.size_range] at hi
  simp only [fileLeaves, Array.getElem_map, Array.getElem_range, ByteArray.size_extract, Merkle.DIGEST] at h ⊢
  have : 64 * (i + 1) ≤ blocks * 64 := by rw [Nat.mul_comm blocks]; exact Nat.mul_le_mul_left 64 hi
  omega

theorem leaves_entry (salt : ByteArray) (C : Array (Array ByteArray)) :
    ∀ i (hi : i < (leaves salt C).size), (leaves salt C)[i].size = 64 := by
  intro i hi
  have hm := Array.getElem_mem hi
  generalize (leaves salt C)[i] = x at hm ⊢
  simp only [leaves, List.mem_toArray, List.mem_flatten, List.mem_mapIdx] at hm
  obtain ⟨_, ⟨_, _, rfl⟩, hx⟩ := hm
  obtain ⟨_, _, rfl⟩ := List.mem_mapIdx.1 hx
  exact hash_size _

theorem root_mth {L : Array ByteArray} {root : ByteArray} (h : Merkle.root L = some root) :
    Merkle.mth L 0 L.size = root := by
  unfold Merkle.root at h
  split at h
  · contradiction
  · exact Option.some.inj h

theorem timedAuditAccepts : TimedAuditAccepts := by
  intro ready leafBytes doc C hacc
  obtain ⟨v, hv1, hv2⟩ := verdict_ok hacc
  obtain ⟨s, salt, root, delta, rtt, k, strict, records, audits, hst, hsize, hroot, hdelta, hrtt, hlimit, hstrict, hk,
    hrecords, hmroot, hgate, hmap, hpos, hall⟩ := verdictOf_accepted hv1 hv2
  by_cases hcoll : ∃ x ∈ runStrings ready leafBytes C, ∃ y ∈ runStrings ready leafBytes C,
      x ≠ y ∧ Sha512.hash x = Sha512.hash y
  · exact .inr hcoll
  left
  have hrun : runStrings ready leafBytes C = nodeInputs (fileLeaves s.blocks leafBytes) 0 s.blocks ++
      nodeInputs (leaves salt C) 0 (leaves salt C).size := by
    unfold runStrings; rw [hst]
  obtain ⟨hsz, hmem⟩ := array_mapM_ok _ records audits hmap
  refine ⟨s, salt, root, delta, rtt, k, strict, records, hst, hsize, hroot, hdelta, hrtt, hlimit, hstrict, hk,
    hrecords, by omega, hmroot, fun hs => gate_refused_nil (hgate hs), fun rec hrec => ?_⟩
  obtain ⟨out, hout, haud⟩ := hmem rec hrec
  obtain ⟨source, answers, hsrc, hsrcpos, hans, hansz, hj⟩ := audit_ok haud
  refine ⟨source, answers, hsrc, hsrcpos, hans, hansz, fun j hjlt => ?_⟩
  obtain ⟨x, hx, _, block, elapsed, hidx, hel, hblk, hxeq⟩ := hj j (hansz ▸ hjlt)
  rw [hall out hout x hx, Prod.mk.injEq] at hxeq
  obtain ⟨h1, h2⟩ := hxeq
  have h1' := h1.symm
  simp only [Bool.and_eq_true, beq_iff_eq] at h1'
  obtain ⟨hbs, hleaf⟩ := h1'
  have hel' : elapsed ≤ delta + rtt := by simpa using h2.symm
  have hi : challenged s salt root source j < s.blocks := by
    by_contra hge
    have hd : (fileLeaves s.blocks leafBytes)[challenged s salt root source j]! = default :=
      getElem!_neg _ _ (by rw [fileLeaves_size]; omega)
    have hs := congrArg ByteArray.size hleaf
    rw [hd, Merkle.packageLeaf, hash_size] at hs
    exact absurd hs (by decide)
  refine ⟨block, elapsed, hidx, hel, hblk, hi, hbs, hel', hleaf, fun hCs hCr => ?_⟩
  rw [hleaf]
  have hr1 := root_mth hmroot
  have hr2 := root_mth hCr
  rw [fileLeaves_size] at hr1
  rw [hCs] at hr2
  rcases mth_binds _ _ (fileLeaves_entry hsize) (leaves_entry salt C) s.blocks 0 (by rw [fileLeaves_size]; omega)
      (by omega) (hr1.trans hr2.symm) with hl | ⟨x, hx, y, hy, hxy⟩
  · exact hl _ (Nat.zero_le _) (by omega)
  · exact absurd ⟨x, by rw [hrun]; exact List.mem_append_left _ hx, y,
      by rw [hrun, hCs]; exact List.mem_append_right _ hy, hxy⟩ hcoll

end PousCheck.Properties.SecurityProofs
