import Definitions.Pouw.Guarantees.Lead
import Proofs.Pouw.Dimension.H100Proofs
import Proofs.Hash.Bytes

/-!
# The lead's honest lemma

The honest strategy writes each committed tile once with one `other` instruction a word, hashes the commitment's
messages with one `other` instruction a credited register, registers the commitment's root and exchanges nothing. For
every coin stream and every commitment format it meets every premise of `G` but `CommitL1w`, and at an admitted call
(`Layout.Admits`) it is accepted at every coin value (`honestAt`): the stream enters only the memory the run starts
from, and the honest run writes every committed tile. So the honest lemma holds at every stream and every commitment
meeting `CommitL1w` (`honestOfCommit`), and at a deployed instance it is `CommitL1w` at the deployed commitment. That
holds where the commitment's messages include frame-v3 leaves, at ranks below `2^64`, over the words' bytes in the
lead's chunks and over A's rows (`commitL1w_of_leaves`): a frame leaf carries its domain's id and its rank
(`leafMsg_id_inj`, `leafMsg_rank_inj`), so leaves at distinct ranks are distinct messages however their values repeat,
and its header puts one more compression on its value's (`blocks_leafMsg`). `honest` is `honestOfCommit`, with a
one-message commitment meeting `CommitL1w` at every call; `Compose/Deployed.lean` has it at the deployed instance
(`honestL1w`).
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-! ## The honest strategy -/

/-- A trace of `other` instructions writes one register each. -/
theorem regCount_other {G : Type} (P : List (Op100 G)) (h : ∀ o ∈ P, ∃ f, o = .other f) : regCount P = P.length := by
  unfold regCount
  rw [List.map_congr_left fun o ho => show Pouw.Lead.Op100.regs o = 1 by obtain ⟨f, rfl⟩ := h o ho; rfl]
  simp

/-- The committed tiles at `ω`, in `tileIdx`'s order. -/
noncomputable def tilesAt (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) :
    List (Fin tileWords → ℤ) :=
  (tileIdx L).toList.map fun x => tileOf L κ ω x.1 x.2.1 x.2.2

/-- The messages the commitment of the words at `ω` hashes: the words' and A's. -/
def commitMsgs (L : Pouw.Lead.Layout) (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) : List Hash.Msg :=
  C.msgs H (word L κ ω) ++ C.aMsgs

/-- The registers hashing `xs` writes, at ShaWrites' credit rounded up. -/
noncomputable def shaRegsOf (isA : Hash.Msg → Bool) (xs : List Hash.Msg) : ℕ := ⌈(xs.map (shaCredit isA)).sum⌉₊

/-- **The honest strategy** for the commitment format `C`: no pre-coin table; at each coin value one `other`
instruction a committed word, one write a committed tile, the commitment's messages hashed with one `other`
instruction a credited register, and its root registered. -/
noncomputable def honestStrategy (L : Pouw.Lead.Layout) (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (isA : Hash.Msg → Bool) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) : Strategy L where
  pre _ _ := 0
  pre_supp _ _ := rfl
  exec ω :=
    { trace := ((tilesAt L κ ω).flatMap fun v => (List.finRange tileWords).map fun q => .other fun _ => (v q : ℚ)) ++
        List.replicate (shaRegsOf isA (commitMsgs L H C κ ω)) (.other 0)
      steps := (tilesAt L κ ω).map fun v => .write (.inl 0) 0 v
      shaRegs := shaRegsOf isA (commitMsgs L H C κ ω)
      hashed := commitMsgs L H C κ ω
      egress := 0
      root := C.root H (word L κ ω) }

theorem honest_trace_other (L : Pouw.Lead.Layout) (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (isA : Hash.Msg → Bool) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) :
    ∀ o ∈ ((honestStrategy L H C isA κ).exec ω).trace, ∃ f, o = .other f := by
  intro o ho
  simp only [honestStrategy, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_replicate] at ho
  rcases ho with ⟨_, _, _, _, rfl⟩ | ⟨_, rfl⟩
  · exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

/-- **The honest strategy meets `G`'s premises but `CommitL1w`**, at every layout, hash, data, coin stream,
commitment format and form classifier: with `Real` the honest strategy alone, the nominal W1 table meets `H32`
(`h32Sat`), the strategy meets `W1`, `TileGrain`, `OneInstructionMinimal`, `CardWithinAccelW1`, `ShaWrites`,
`HonestHashRuled` and `OutsourceWithinEgressLead` (it is its own twin), and at an admitted call it is accepted at every
coin value. -/
theorem honestAt (L : Pouw.Lead.Layout) (H : Hash.Msg → Hash.Digest) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (stream : CoinStream L) (C : Pouw.Lead.Commit L) (isA : Hash.Msg → Bool) (hadm : L.Admits) :
    H32 h100Nominal ∧ W1 h100Nominal (· = honestStrategy L H C isA κ) ∧
      TileGrain H C κ stream (· = honestStrategy L H C isA κ) ∧ OneInstructionMinimal (· = honestStrategy L H C isA κ) ∧
      CardWithinAccelW1 (· = honestStrategy L H C isA κ) (· = honestStrategy L H C isA κ) ∧
      ShaWrites isA (· = honestStrategy L H C isA κ) ∧ HonestHashRuled H C κ (· = honestStrategy L H C isA κ) ∧
      OutsourceWithinEgressLead h100Nominal (betaLead L C) H C κ (· = honestStrategy L H C isA κ) ∧
      honestStrategy L H C isA κ = honestStrategy L H C isA κ ∧
      ∀ ω, Accepted L H C κ (honestStrategy L H C isA κ) ω := by
  refine ⟨h32Sat, ?_, ?_, ?_, fun _ h => h, ?_, ?_, ?_, rfl, ?_⟩
  · rintro σ rfl ω o ho
    obtain ⟨f, rfl⟩ := honest_trace_other L H C isA κ ω o ho
    trivial
  · rintro σ rfl ω - - rb hrb cb hcb t
    have hy : tileOf L κ ω rb cb t ∈ tilesAt L κ ω := List.mem_map.2 ⟨(rb, cb, t), by simp [tileIdx, hrb, hcb], rfl⟩
    exact Or.inl ⟨.inl 0, 0, List.mem_map.2 ⟨_, hy, rfl⟩⟩
  · rintro σ rfl ω
    rw [regCount_other _ (honest_trace_other L H C isA κ ω)]
    simp only [honestStrategy, writes, List.map_map, Function.comp_def, Step.cost, List.length_append,
      List.length_replicate, List.length_flatMap, List.length_map, List.length_finRange, List.map_const',
      List.sum_replicate, smul_eq_mul, mul_one, Nat.mul_comm tileWords]
    rfl
  · rintro σ rfl ω
    exact Nat.le_ceil _
  · rintro σ rfl ω - - y hy
    exact ⟨y, hy, rfl⟩
  · rintro σ rfl
    refine ⟨_, rfl, rfl, fun ω => ⟨rfl, rfl, fun _ h => Or.inl h, ?_⟩⟩
    have : ((honestStrategy L H C isA κ).exec ω).egress = 0 := rfl
    rw [this, Nat.cast_zero, mul_zero, add_zero]
  · intro ω
    refine ⟨hadm, rfl, ?_⟩
    show ((0 : ℕ) : ℚ) ≤ _
    unfold Layout.callBits
    push_cast
    positivity

/-! ## `CommitL1w` from frame-v3 leaves -/

/-- A `uint` encoding determines its number. -/
theorem uint_inj {r r' : ℕ} (h : Hash.uint r = Hash.uint r') : r = r' := by
  have hl := congrArg List.length h
  simp only [Hash.uint, Hash.BindingProofs.be_length] at hl
  have hlt : ∀ x : ℕ, x < 256 ^ max 1 ((Nat.size x + 7) / 8) := fun x =>
    (Nat.lt_size_self x).trans_le <| by
      rw [show (256 : ℕ) = 2 ^ 8 by norm_num, ← pow_mul]
      exact Nat.pow_le_pow_right (by norm_num) (by omega)
  unfold Hash.uint at h
  rw [hl] at h
  exact Hash.BindingProofs.be_inj_lt (hl ▸ hlt r) (hlt r') h

theorem uint_length_le {r : ℕ} (hr : r < 2 ^ 64) : (Hash.uint r).length ≤ 8 := by
  have := Nat.size_le.2 hr
  simp only [Hash.uint, Hash.BindingProofs.be_length]
  omega

theorem one_le_uint_length (r : ℕ) : 1 ≤ (Hash.uint r).length := by
  simp only [Hash.uint, Hash.BindingProofs.be_length]
  omega

/-- **A frame leaf carries its rank**: two frame leaves of one domain at ranks below `2^64` are at one rank, whatever
their schemas and values. -/
theorem leafMsg_rank_inj {D : Hash.Tree.Domain} {r r' : ℕ} {s s' : String} {v v' : Hash.Msg} (hr : r < 2 ^ 64)
    (hr' : r' < 2 ^ 64) (h : Hash.Tree.leafMsg D r s v = Hash.Tree.leafMsg D r' s' v') : r = r' := by
  have h := Hash.BindingProofs.frame_parts h
  simp only [List.flatMap_cons, List.append_assoc, List.append_cancel_left_eq] at h
  obtain ⟨hlen, h⟩ := List.append_inj h (by simp)
  have hlen := Hash.BindingProofs.be_inj_lt ((uint_length_le hr).trans_lt (by norm_num))
    ((uint_length_le hr').trans_lt (by norm_num)) hlen
  exact uint_inj (List.append_inj h hlen).1

/-- **A frame leaf carries its domain's id**: two equal frame leaves are of domains with one id, whatever their ranks,
schemas and values. -/
theorem leafMsg_id_inj {D D' : Hash.Tree.Domain} {r r' : ℕ} {s s' : String} {v v' : Hash.Msg}
    (h : Hash.Tree.leafMsg D r s v = Hash.Tree.leafMsg D' r' s' v') : D.id = D'.id := by
  have h := Hash.BindingProofs.frame_parts h
  simp only [List.flatMap_cons, List.append_assoc, Hash.BindingProofs.bytes_length] at h
  exact Hash.BindingProofs.bytes_injective (List.append_inj (List.append_cancel_left h) rfl).1

/-- Frame leaves of one domain at distinct ranks below `2^64` are distinct messages. -/
theorem nodup_leaves (D : Hash.Tree.Domain) (s : String) (v : ℕ → Hash.Msg) {rs : List ℕ} (hrs : rs.Nodup)
    (hr : ∀ r ∈ rs, r < 2 ^ 64) : (rs.map fun r => Hash.Tree.leafMsg D r s (v r)).Nodup :=
  hrs.map_on fun _ hx _ hy h => leafMsg_rank_inj (hr _ hx) (hr _ hy) h

/-- **A frame leaf's compressions**: its header (at least 147 bytes) puts at least one compression on its value's,
`|v| / 128 + 1 ≤ blocks`. -/
theorem blocks_leafMsg (D : Hash.Tree.Domain) (r : ℕ) (s : String) (v : Hash.Msg) :
    v.length + 128 ≤ 128 * blocks (Hash.Tree.leafMsg D r s v) := by
  have h1 := one_le_uint_length r
  have h2 := one_le_uint_length (D.position r)
  have hF : Hash.Tree.FRAME.length = 33 := by decide
  have hT : (Hash.utf8 "leaf").length = 4 := by decide
  simp only [blocks, Hash.Tree.leafMsg, Hash.Tree.frame, List.length_append, List.length_flatMap, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Hash.BindingProofs.be_length, Hash.BindingProofs.bytes_length, hF, hT]
  omega

/-- The frame leaves at ranks below `k` put at least one compression each on their values'. -/
theorem blocks_leaves (D : Hash.Tree.Domain) (s : String) (v : ℕ → Hash.Msg) :
    ∀ k, ((List.range k).map fun r => (v r).length).sum + 128 * k ≤
      128 * (((List.range k).map fun r => Hash.Tree.leafMsg D r s (v r)).map blocks).sum
  | 0 => by simp
  | k + 1 => by
    have ih := blocks_leaves D s v k
    have hk := blocks_leafMsg D k s (v k)
    simp only [List.range_succ, List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil]
    omega

/-- **`CommitL1w` from distinct messages**: a commitment with the lead's chunks meets `CommitL1w` when, at every coin
value, its messages include distinct words messages (`isA` false) of at least `wordsBlocks` compressions and distinct
A messages (`isA` true) of at least `aBlocks`. -/
theorem commitL1w_of_distinct {L : Pouw.Lead.Layout} (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (isA : Hash.Msg → Bool) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (hch : C.chunks = wordChunks L)
    (h : ∀ ω : Coins L, ∃ ws as : List Hash.Msg, ws.Nodup ∧ as.Nodup ∧
      (∀ x ∈ ws ++ as, x ∈ C.msgs H (word L κ ω) ++ C.aMsgs) ∧ (∀ x ∈ ws, isA x = false) ∧
      (∀ x ∈ as, isA x = true) ∧ wordsBlocks L C ≤ ((ws.map blocks).sum : ℚ) ∧
      aBlocks L ≤ ((as.map blocks).sum : ℚ)) :
    CommitL1w H C isA κ := by
  refine ⟨hch, fun ω => ?_⟩
  obtain ⟨ws, as, hws, has, hsub, hw, ha, hbw, hba⟩ := h ω
  have hnd : (ws ++ as).Nodup := List.nodup_append.2 ⟨hws, has, fun x hx y hy hxy => by
    have h1 := hw x hx
    have h2 := ha y hy
    rw [hxy] at h1
    rw [h1] at h2
    exact Bool.false_ne_true h2⟩
  obtain ⟨l, hp, hl⟩ := hnd.subperm fun x hx => List.mem_dedup.2 (hsub x hx)
  have hle : ((ws ++ as).map (shaCredit isA)).sum ≤
      (((C.msgs H (word L κ ω) ++ C.aMsgs).dedup).map (shaCredit isA)).sum := by
    rw [← (hp.map (shaCredit isA)).sum_eq]
    exact (hl.map (shaCredit isA)).sum_le_sum fun a ha => by
      obtain ⟨b, -, rfl⟩ := List.mem_map.1 ha
      unfold shaCredit shaARegs shaWordsRegs
      split_ifs <;> positivity
  have hW : (ws.map (shaCredit isA)).sum = shaWordsRegs * ((ws.map blocks).sum : ℚ) := by
    rw [List.map_congr_left fun x hx => show shaCredit isA x = shaWordsRegs * (blocks x : ℚ) by
      simp [shaCredit, hw x hx], List.sum_map_mul_left]
    push_cast [List.map_map]
    rfl
  have hA : (as.map (shaCredit isA)).sum = shaARegs * ((as.map blocks).sum : ℚ) := by
    rw [List.map_congr_left fun x hx => show shaCredit isA x = shaARegs * (blocks x : ℚ) by
      simp [shaCredit, ha x hx], List.sum_map_mul_left]
    push_cast [List.map_map]
    rfl
  rw [List.map_append, List.sum_append, hW, hA] at hle
  have hw0 : (0 : ℚ) ≤ shaWordsRegs := by unfold shaWordsRegs; norm_num
  have ha0 : (0 : ℚ) ≤ shaARegs := by unfold shaARegs; norm_num
  nlinarith [mul_le_mul_of_nonneg_left hbw hw0, mul_le_mul_of_nonneg_left hba ha0]

/-- **`CommitL1w` from frame-v3 leaves**: a commitment with the lead's chunks meets `CommitL1w` when, at every coin
value, its messages include the frame leaves of a words domain `Dw` at ranks below `kw` over values `cw ω r` of at
least the words' `4 m n S` bytes in all, with `C.chunks ≤ kw`, and the frame leaves of an A domain `Da` at ranks below
`ka` over values `av r` of at least A's `rows · K` bytes in all, the first classed words and the second A, at ranks
below `2^64`. The values may repeat: the leaves carry their ranks. -/
theorem commitL1w_of_leaves {L : Pouw.Lead.Layout} (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (isA : Hash.Msg → Bool) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (hch : C.chunks = wordChunks L)
    (Dw Da : Hash.Tree.Domain) (sw sa : String) (kw ka : ℕ) (hkw : kw ≤ 2 ^ 64) (hka : ka ≤ 2 ^ 64)
    (hc : C.chunks ≤ kw) (cw : Coins L → ℕ → Hash.Msg) (av : ℕ → Hash.Msg)
    (hbw : ∀ ω, 4 * (L.m * L.n * L.S) ≤ ((List.range kw).map fun r => (cw ω r).length).sum)
    (hba : L.rows * L.K ≤ ((List.range ka).map fun r => (av r).length).sum)
    (hmw : ∀ ω, ∀ r < kw, Hash.Tree.leafMsg Dw r sw (cw ω r) ∈ C.msgs H (word L κ ω) ++ C.aMsgs)
    (hma : ∀ ω, ∀ r < ka, Hash.Tree.leafMsg Da r sa (av r) ∈ C.msgs H (word L κ ω) ++ C.aMsgs)
    (hiw : ∀ ω, ∀ r < kw, isA (Hash.Tree.leafMsg Dw r sw (cw ω r)) = false)
    (hia : ∀ r < ka, isA (Hash.Tree.leafMsg Da r sa (av r)) = true) :
    CommitL1w H C isA κ := by
  have hsum : ∀ (D : Hash.Tree.Domain) (s : String) (v : ℕ → Hash.Msg) (k : ℕ),
      ((((List.range k).map fun r => (v r).length).sum + 128 * k : ℕ) : ℚ) / 128 ≤
        ((((List.range k).map fun r => Hash.Tree.leafMsg D r s (v r)).map blocks).sum : ℚ) := by
    intro D s v k
    rw [div_le_iff₀ (by norm_num)]
    exact_mod_cast (blocks_leaves D s v k).trans_eq (Nat.mul_comm _ _)
  refine commitL1w_of_distinct H C isA κ hch fun ω => ⟨(List.range kw).map fun r => Hash.Tree.leafMsg Dw r sw (cw ω r),
    (List.range ka).map fun r => Hash.Tree.leafMsg Da r sa (av r),
    nodup_leaves Dw sw (cw ω) List.nodup_range fun r hr => (List.mem_range.1 hr).trans_le hkw,
    nodup_leaves Da sa av List.nodup_range fun r hr => (List.mem_range.1 hr).trans_le hka, ?_, ?_, ?_, ?_, ?_⟩
  · intro x hx
    rcases List.mem_append.1 hx with hx | hx <;> obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hx
    · exact hmw ω r (List.mem_range.1 hr)
    · exact hma ω r (List.mem_range.1 hr)
  · intro x hx
    obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hx
    exact hiw ω r (List.mem_range.1 hr)
  · intro x hx
    obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hx
    exact hia r (List.mem_range.1 hr)
  · refine le_trans ?_ (hsum Dw sw (cw ω) kw)
    have h1 : (4 * (L.m * L.n * L.S) : ℚ) ≤ (((List.range kw).map fun r => (cw ω r).length).sum : ℕ) := by
      exact_mod_cast hbw ω
    have h2 : (C.chunks : ℚ) ≤ kw := by exact_mod_cast hc
    rw [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
    unfold wordsBlocks wordsRegs
    linarith
  · refine le_trans ?_ (hsum Da sa av ka)
    have h1 : (L.rows * L.K : ℚ) ≤ (((List.range ka).map fun r => (av r).length).sum : ℕ) := by
      exact_mod_cast hba
    rw [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
    unfold aBlocks
    have : (0 : ℚ) ≤ ka := Nat.cast_nonneg _
    linarith

/-- **The honest lemma at every coin stream and every commitment meeting `CommitL1w`**: at every admitted call, for
every coin stream, commitment format and form classifier meeting `CommitL1w`, some set of strategies meets every
premise of `G` and holds an honest strategy accepted at every coin value. At the protocol's stream and commitment it is
the honest lemma at the deployed instance, given `CommitL1w` there (`honestL1w`). -/
theorem honestOfCommit (T m n K : ℕ) (H : Hash.Msg → Hash.Digest)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K)) (isA : Hash.Msg → Bool)
    (hC : CommitL1w H C isA κ) (hadm : (lead T m n K).Admits) :
    ∃ (Real : Strategy (lead T m n K) → Prop) (σ : Strategy (lead T m n K)),
      CommitL1w H C isA κ ∧ H32 h100Nominal ∧ W1 h100Nominal Real ∧ TileGrain H C κ stream Real ∧
      OneInstructionMinimal Real ∧ CardWithinAccelW1 Real Real ∧ ShaWrites isA Real ∧ HonestHashRuled H C κ Real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead (lead T m n K) C) H C κ Real ∧ Real σ ∧
      ∀ ω, Accepted (lead T m n K) H C κ σ ω :=
  ⟨_, _, hC, honestAt (lead T m n K) H κ stream C isA hadm⟩

/-! ## A commitment meeting `CommitL1w` -/

/-- The words' and A's compressions `CommitL1w` credits, each rounded up. -/
noncomputable def oneMsgBlocks (L : Pouw.Lead.Layout) : ℕ :=
  ⌈4 * wordsRegs L / 128 + (wordChunks L : ℚ)⌉₊ + ⌈aBlocks L⌉₊

/-- One message of `oneMsgBlocks L + 1` compressions. -/
noncomputable def oneMsg (L : Pouw.Lead.Layout) : Hash.Msg := List.replicate (128 * oneMsgBlocks L) 0

/-- A commitment format meeting `CommitL1w` at every layout and data: one words message of `oneMsgBlocks L + 1`
compressions, and no A message. -/
noncomputable def oneMsgCommit (L : Pouw.Lead.Layout) : Pouw.Lead.Commit L where
  msgs _ _ := [oneMsg L]
  root _ _ := 0
  aMsgs := []
  chunks := wordChunks L

theorem blocks_oneMsg (L : Pouw.Lead.Layout) : blocks (oneMsg L) = oneMsgBlocks L + 1 := by
  rw [blocks, oneMsg, List.length_replicate]
  omega

theorem commitL1w_oneMsg (L : Pouw.Lead.Layout) (H : Hash.Msg → Hash.Digest) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    CommitL1w H (oneMsgCommit L) (fun _ => false) κ := by
  refine ⟨rfl, fun ω => ?_⟩
  simp only [oneMsgCommit, List.append_nil, List.dedup_cons_of_notMem List.not_mem_nil, List.dedup_nil,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, shaCredit, blocks_oneMsg, wordsBlocks,
    Bool.false_eq_true, ↓reduceIte]
  have hX := Nat.le_ceil (4 * wordsRegs L / 128 + (wordChunks L : ℚ))
  have hY := Nat.le_ceil (aBlocks L)
  have hY0 : 0 ≤ aBlocks L := by unfold aBlocks; positivity
  unfold oneMsgBlocks
  push_cast
  norm_num only [shaWordsRegs, shaARegs]
  linarith

/-- **The honest lemma** (`Pouw.Guarantees.Lead.Honest`): `honestOfCommit`, and the one-message commitment meets
`CommitL1w` at every call (`commitL1w_oneMsg`). -/
theorem honest : Pouw.Guarantees.Lead.Honest :=
  ⟨honestOfCommit, fun _ _ _ _ H κ => ⟨oneMsgCommit _, fun _ => false, commitL1w_oneMsg _ H κ⟩⟩

end Pouw.SecurityProofs.Lead.Compose
