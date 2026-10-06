import Proofs.Flock.Soundness.Refine.Paths
import Proofs.Flock.Soundness.Refine.Exec
import Proofs.Flock.Verifier.Transcript

/-!
# The transcript's frames (R11b, framing)

The replay transcript (PROTOCOL.md §6) appends one frame per domain, label, absorb or zero-coin squeeze to its pending
bytes, and a squeeze answers the pending bytes and its own header; on the unsalted path the server retained each round's
bytes, and the replay compares them exactly. This file models those bytes.

* `Frame`, `Frame.enc`: a frame and its bytes, as the executable writes them (a 16-byte header naming its kind and
  length, then its payload).
* `encs_inj`: a list of frames is recoverable from its bytes, so a round's retained bytes determine what was absorbed.
* `label_enc`, `absorbF_enc`, `absorbSlice_enc`, `absorbBytes_enc`, `framed_enc`, `init_enc`: each step of the
  transcript appends its frame's bytes.
* `squeeze_bindA`, `sample_bindA`, `sampleVec_bindA`, `sampleK_bindA`, `forIn_answers`: a squeeze, or a loop of them,
  consumes rounds that retained the pending bytes and the squeeze's header.
-/

namespace FlockSoundness.Refine

open Flock

theorem le64_data (x : UInt64) : (le64 x).data.toList = (List.range 8).map fun i => (x >>> (8 * i).toUInt64).toUInt8 := by
  simp [le64, List.range', ByteArray.push, List.range_succ]
  rfl

theorem le64_inj {x y : UInt64} (h : le64 x = le64 y) : x = y := by
  have hd := congrArg (fun b => b.data.toList) h
  simp only [le64_data] at hd
  exact u64_of_bytes x y fun i hi => by
    have := congrArg (·[i]?) hd
    simpa [List.getElem?_map, List.getElem?_range hi] using this

theorem header_data (op kind : UInt8) (len : ℕ) :
    (Transcript.header op kind len).data.toList = [op, kind, 0, 0, 0, 0, 0, 0] ++ (le64 len.toUInt64).data.toList := by
  simp [Transcript.header]

theorem foldl_push_zero (l : List ℕ) (out : ByteArray) :
    (l.foldl (fun r _ => r.push 0) out).data.toList = out.data.toList ++ List.replicate l.length 0 := by
  induction l generalizing out with
  | nil => simp
  | cons a l ih =>
      simp only [List.foldl_cons, ih, List.length_cons]
      rw [List.replicate_succ]
      simp [ByteArray.push]

theorem padded_data (b : ByteArray) :
    (Transcript.padded b).data.toList = b.data.toList ++ List.replicate ((16 - b.size % 16) % 16) 0 := by
  unfold Transcript.padded
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  rw [List.forIn_pure_yield_eq_foldl]
  change (List.foldl (fun b _ => b.push 0) b (List.range' 0 _)).data.toList = _
  rw [foldl_push_zero, List.length_range']
  congr 2
  omega

/-- **A frame** of the replay transcript (PROTOCOL.md §6.1): the bytes one domain, label, absorb or zero-coin squeeze
appends to the pending bytes, or the header a squeeze answers. -/
inductive Frame where
  | dom (b : ByteArray)
  | label (b : ByteArray)
  | one (v : F128)
  | vec (vs : List F128)
  | bytes (b : ByteArray)
  | sq (kind : UInt8) (n : ℕ)

namespace Frame

def op : Frame → UInt8
  | .dom _ => 1 | .label _ => 2 | .one _ => 3 | .vec _ => 3 | .bytes _ => 5 | .sq _ _ => 4

def kind : Frame → UInt8
  | .dom _ => 0 | .label _ => 0 | .one _ => 1 | .vec _ => 2 | .bytes _ => 0 | .sq k _ => k

def len : Frame → ℕ
  | .dom b => b.size | .label b => b.size | .one _ => 1 | .vec vs => vs.length | .bytes b => b.size | .sq _ n => n

/-- The payload after the header. -/
def pay : Frame → List UInt8
  | .dom b => (Transcript.padded b).data.toList
  | .label b => (Transcript.padded b).data.toList
  | .one v => v.toBytes.data.toList
  | .vec vs => colBytes vs
  | .bytes b => (Transcript.padded b).data.toList
  | .sq _ _ => []

/-- A frame's bytes, as the executable writes them: its 16-byte header, then its payload. -/
def enc (f : Frame) : List UInt8 := (Transcript.header f.op f.kind f.len).data.toList ++ f.pay

/-- A frame's length fits its header's eight bytes. -/
def WF (f : Frame) : Prop := f.len < 2 ^ 64

def payLen : Frame → ℕ
  | .dom b => b.size + (16 - b.size % 16) % 16
  | .label b => b.size + (16 - b.size % 16) % 16
  | .one _ => 16
  | .vec vs => 16 * vs.length
  | .bytes b => b.size + (16 - b.size % 16) % 16
  | .sq _ _ => 0

theorem colBytes_length (vs : List F128) : (colBytes vs).length = 16 * vs.length := by
  induction vs with
  | nil => simp [colBytes]
  | cons a vs ih =>
      simp only [colBytes, List.flatMap_cons, List.length_append, toBytes_length] at ih ⊢
      rw [ih, List.length_cons]; ring

theorem pay_length (f : Frame) : f.pay.length = f.payLen := by
  cases f with
  | dom b => simp [pay, payLen, padded_data]
  | label b => simp [pay, payLen, padded_data]
  | one v => exact toBytes_length v
  | vec vs => exact colBytes_length vs
  | bytes b => simp [pay, payLen, padded_data]
  | sq _ _ => rfl

theorem header_length (op kind : UInt8) (n : ℕ) : (Transcript.header op kind n).data.toList.length = 16 := by
  rw [header_data, List.length_append, le64_data]; simp

/-- The header determines the frame's kind and length. -/
theorem header_inj {op kind op' kind' : UInt8} {n n' : ℕ} (hn : n < 2 ^ 64) (hn' : n' < 2 ^ 64)
    (h : (Transcript.header op kind n).data.toList = (Transcript.header op' kind' n').data.toList) :
    op = op' ∧ kind = kind' ∧ n = n' := by
  rw [header_data, header_data] at h
  simp only [List.cons_append, List.cons.injEq] at h
  obtain ⟨rfl, rfl, -, -, -, -, -, -, h⟩ := h
  refine ⟨rfl, rfl, ?_⟩
  have h8 := le64_inj (ByteArray.ext (Array.ext' (by simpa using h)))
  have := congrArg UInt64.toNat h8
  simp only [UInt64.toNat_ofNat', Nat.mod_eq_of_lt hn, Nat.mod_eq_of_lt hn'] at this
  exact this

theorem payLen_eq {f f' : Frame} (hop : f.op = f'.op) (hk : f.kind = f'.kind) (hl : f.len = f'.len) :
    f.payLen = f'.payLen := by
  cases f <;> cases f' <;> simp_all [op, kind, len, payLen]

theorem bytes_of_padded {b b' : ByteArray} (hl : b.size = b'.size)
    (h : (Transcript.padded b).data.toList = (Transcript.padded b').data.toList) : b = b' := by
  rw [padded_data, padded_data, hl] at h
  exact ByteArray.ext (Array.ext' (List.append_inj h (by simpa using hl)).1)

theorem eq_of_parts : ∀ {f f' : Frame}, f.op = f'.op → f.kind = f'.kind → f.len = f'.len → f.pay = f'.pay → f = f'
  | .dom b, .dom b', _, _, hl, hp => by rw [bytes_of_padded hl hp]
  | .label b, .label b', _, _, hl, hp => by rw [bytes_of_padded hl hp]
  | .bytes b, .bytes b', _, _, hl, hp => by rw [bytes_of_padded hl hp]
  | .one v, .one v', _, _, _, hp => by rw [toBytes_inj (ByteArray.ext (Array.ext' hp))]
  | .vec vs, .vec vs', _, _, _, hp => by rw [colBytes_inj hp]
  | .sq k n, .sq k' n', _, hk, hl, _ => by simp only [kind, len] at hk hl; rw [hk, hl]
  | .dom _, .label _, h, _, _, _ | .dom _, .one _, h, _, _, _ | .dom _, .vec _, h, _, _, _
  | .dom _, .bytes _, h, _, _, _ | .dom _, .sq _ _, h, _, _, _
  | .label _, .dom _, h, _, _, _ | .label _, .one _, h, _, _, _ | .label _, .vec _, h, _, _, _
  | .label _, .bytes _, h, _, _, _ | .label _, .sq _ _, h, _, _, _
  | .one _, .dom _, h, _, _, _ | .one _, .label _, h, _, _, _ | .one _, .bytes _, h, _, _, _
  | .one _, .sq _ _, h, _, _, _
  | .vec _, .dom _, h, _, _, _ | .vec _, .label _, h, _, _, _ | .vec _, .bytes _, h, _, _, _
  | .vec _, .sq _ _, h, _, _, _
  | .bytes _, .dom _, h, _, _, _ | .bytes _, .label _, h, _, _, _ | .bytes _, .one _, h, _, _, _
  | .bytes _, .vec _, h, _, _, _ | .bytes _, .sq _ _, h, _, _, _
  | .sq _ _, .dom _, h, _, _, _ | .sq _ _, .label _, h, _, _, _ | .sq _ _, .one _, h, _, _, _
  | .sq _ _, .vec _, h, _, _, _ | .sq _ _, .bytes _, h, _, _, _ => by simp [op] at h
  | .one _, .vec _, _, h, _, _ | .vec _, .one _, _, h, _, _ => by simp [kind] at h

/-- **A frame is its bytes.** -/
theorem eq_of_enc {f f' : Frame} (hw : f.WF) (hw' : f'.WF) (h : f.enc = f'.enc) : f = f' := by
  unfold enc at h
  have hsplit := List.append_inj h (by rw [header_length, header_length])
  obtain ⟨hop, hk, hl⟩ := header_inj hw hw' hsplit.1
  exact eq_of_parts hop hk hl hsplit.2

theorem enc_length (f : Frame) : f.enc.length = 16 + f.payLen := by
  unfold enc; rw [List.length_append, header_length, pay_length]

/-- **Frames delimit themselves:** a frame's header fixes its length. -/
theorem enc_append_inj {f f' : Frame} {r r' : List UInt8} (hw : f.WF) (hw' : f'.WF) (h : f.enc ++ r = f'.enc ++ r') :
    f = f' ∧ r = r' := by
  have h₀ := h
  unfold enc at h₀
  rw [List.append_assoc, List.append_assoc] at h₀
  obtain ⟨hop, hk, hl⟩ := header_inj hw hw' (List.append_inj h₀ (by rw [header_length, header_length])).1
  have hlen : f.enc.length = f'.enc.length := by rw [enc_length, enc_length, payLen_eq hop hk hl]
  obtain ⟨he, hr⟩ := List.append_inj h hlen
  exact ⟨eq_of_enc hw hw' he, hr⟩

end Frame

/-- The bytes of a list of frames. -/
def encs (fs : List Frame) : List UInt8 := fs.flatMap Frame.enc

/-- **Frames are recoverable from their bytes.** -/
theorem encs_inj : ∀ {fs fs' : List Frame}, (∀ f ∈ fs, f.WF) → (∀ f ∈ fs', f.WF) → encs fs = encs fs' → fs = fs'
  | [], [], _, _, _ => rfl
  | [], f' :: _, _, _, h => by
      have := congrArg List.length h
      simp only [encs, List.flatMap_nil, List.flatMap_cons, List.length_append, Frame.enc_length, List.length_nil] at this
      omega
  | f :: _, [], _, _, h => by
      have := congrArg List.length h
      simp only [encs, List.flatMap_nil, List.flatMap_cons, List.length_append, Frame.enc_length, List.length_nil] at this
      omega
  | f :: fs, f' :: fs', hw, hw', h => by
      simp only [encs, List.flatMap_cons] at h
      obtain ⟨rfl, hr⟩ := Frame.enc_append_inj (hw f (by simp)) (hw' f' (by simp)) h
      rw [encs_inj (fun g hg => hw g (by simp [hg])) (fun g hg => hw' g (by simp [hg])) hr]

/-! ## What the transcript's steps append -/

open Transcript

theorem data_append (a b : ByteArray) : (a ++ b).data.toList = a.data.toList ++ b.data.toList := by simp

theorem label_enc (p : ByteArray) (l : String) :
    (p ++ header 2 0 l.toUTF8.size ++ padded l.toUTF8).data.toList = p.data.toList ++ (Frame.label l.toUTF8).enc := by
  simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

theorem absorbF_enc (p : ByteArray) (v : F128) :
    (F128.pushBytes (p ++ header 3 1 1) v).data.toList = p.data.toList ++ (Frame.one v).enc := by
  simp [F128.pushBytes, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

theorem foldl_pushBytes (L : List F128) (out : ByteArray) :
    (L.foldl F128.pushBytes out).data.toList = out.data.toList ++ colBytes L := by
  induction L generalizing out with
  | nil => simp [colBytes]
  | cons a L ih => simp [List.foldl_cons, ih, F128.pushBytes, colBytes]

theorem absorbSlice_enc (p : ByteArray) (vs : Array F128) :
    (vs.foldl F128.pushBytes (p ++ header 3 2 vs.size)).data.toList = p.data.toList ++ (Frame.vec vs.toList).enc := by
  rw [← Array.foldl_toList, foldl_pushBytes]
  simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

theorem absorbBytes_enc (p b : ByteArray) :
    (p ++ header 5 0 b.size ++ padded b).data.toList = p.data.toList ++ (Frame.bytes b).enc := by
  simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

theorem framed_enc (t : Transcript) (kind : UInt8) (n : ℕ) :
    (framed t kind n).data.toList = t.pending.data.toList ++ (Frame.sq kind n).enc := by
  simp [framed, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

theorem init_enc (rounds : Array Round) (domain : ByteArray) :
    (Transcript.init rounds domain).pending.data.toList = (Frame.dom domain).enc := by
  simp [Transcript.init, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

/-- A round whose bytes the server retained answers exactly them. -/
theorem answers_retained {r : Round} {B c : ByteArray} (hb : r.bytes = some B) (h : r.answers c) : c = B := by
  unfold Round.answers at h
  rw [hb] at h
  exact h

/-- Round `r` retained the bytes `L`. -/
def Retains (r : Round) (L : List UInt8) : Prop := r.bytes.map (fun B => B.data.toList) = some L

/-- Every round of `rs` retained its bytes, as on the unsalted path's server. -/
def AllRetained (rs : Array Round) : Prop := ∀ i < rs.size, ∃ B, (rs[i]!).bytes = some B

variable {β : Type}

/-- A squeeze of `n ≠ 0` coins then `f`: round `t.k` issued `n` coins and retained the pending bytes and the squeeze's
header; `f` continues with its coins and nothing pending. -/
theorem squeeze_bindA {kind : UInt8} {n : ℕ} (hn : n ≠ 0) {f : Array F128 → TM β} {t t' : Transcript} {b : β}
    (h : (squeeze kind n >>= f) t = .ok (b, t')) (hret : AllRetained t.rounds) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = n ∧
      Retains (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq kind n).enc) ∧
      f (t.rounds[t.k]!).coins { t with pending := ByteArray.empty, k := t.k + 1 } = .ok (b, t') := by
  obtain ⟨cs, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨r, hr, hrn, hans, hc, rfl⟩ := squeeze_answers kind n t t₁ cs hn h₁
  have hlt : t.k < t.rounds.size := by
    by_contra hge; rw [Array.getElem?_eq_none (by omega)] at hr; cases hr
  have hget : t.rounds[t.k]! = r := by
    rw [getElem!_pos t.rounds t.k hlt]; exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans hr)
  obtain ⟨B, hB⟩ := hret t.k hlt
  rw [hget] at hB
  refine ⟨hlt, hget ▸ hrn, ?_, hget ▸ hc ▸ h₂⟩
  rw [hget, Retains, hB, ← answers_retained hB hans]
  simp [framed_enc]

theorem sample_bindA {f : F128 → TM β} {t t' : Transcript} {b : β} (h : (sample >>= f) t = .ok (b, t'))
    (hret : AllRetained t.rounds) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 1 ∧ Retains (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq 1 1).enc) ∧
      f ((t.rounds[t.k]!).coins.getD 0 0) { t with pending := ByteArray.empty, k := t.k + 1 } = .ok (b, t') := by
  obtain ⟨c, t₁, h₁, h₂⟩ := bind_ok h
  unfold sample at h₁
  obtain ⟨hlt, hn, hR, h₁⟩ := squeeze_bindA (by decide) h₁ hret
  obtain ⟨rfl, rfl⟩ := pure_ok h₁
  exact ⟨hlt, hn, hR, h₂⟩

theorem sampleVec_bindA {n : ℕ} (hn : n ≠ 0) {f : Array F128 → TM β} {t t' : Transcript} {b : β}
    (h : (sampleVec n >>= f) t = .ok (b, t')) (hret : AllRetained t.rounds) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = n ∧ Retains (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq 2 n).enc) ∧
      f (t.rounds[t.k]!).coins { t with pending := ByteArray.empty, k := t.k + 1 } = .ok (b, t') :=
  squeeze_bindA hn h hret

theorem sampleK_bindA {f : F256 → TM β} {t t' : Transcript} {b : β} (h : (sampleK >>= f) t = .ok (b, t'))
    (hret : AllRetained t.rounds) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 2 ∧ Retains (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq 2 2).enc) ∧
      f ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩
        { t with pending := ByteArray.empty, k := t.k + 1 } = .ok (b, t') := by
  obtain ⟨c, t₁, h₁, h₂⟩ := bind_ok h
  unfold sampleK at h₁
  obtain ⟨hlt, hn, hR, h₁⟩ := squeeze_bindA (by decide) h₁ hret
  obtain ⟨rfl, rfl⟩ := pure_ok h₁
  exact ⟨hlt, hn, hR, h₂⟩

/-- **A loop answering one round per step.** If every step of the body, with nothing pending, consumes the next round,
which retained `F i`, and leaves nothing pending, then the loop over `a, …, a + n − 1` consumes `n` rounds, which
retained `F a, …, F (a + n − 1)`. -/
theorem forIn_answers {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (F : ℕ → List UInt8)
    (hbody : ∀ i s t b t', AllRetained t.rounds → t.pending = ByteArray.empty → body i s t = .ok (b, t') →
      ∃ s', b = ForInStep.yield s' ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 ∧ t'.pending = ByteArray.empty ∧
        Retains (t.rounds[t.k]!) (F i)) :
    ∀ (n a : ℕ) (s : σ) (t t' : Transcript) (s' : σ), AllRetained t.rounds → t.pending = ByteArray.empty →
      (forIn (List.range' a n) s body : TM σ) t = .ok (s', t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n ∧ t'.pending = ByteArray.empty ∧
        ∀ i < n, Retains (t.rounds[t.k + i]!) (F (a + i))
  | 0, a, s, t, t', s', _, hp, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact ⟨rfl, rfl, hp, fun i hi => absurd hi (by omega)⟩
  | n + 1, a, s, t, t', s', hret, hp, h => by
      rw [forIn_range'_cons] at h
      obtain ⟨r, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨s₁, rfl, hr₁, hk₁, hp₁, hR₁⟩ := hbody a s t r t₁ hret hp h₁
      obtain ⟨hr, hk, hpe, hR⟩ := forIn_answers body F hbody n (a + 1) s₁ t₁ t' s' (hr₁ ▸ hret) hp₁ h₂
      refine ⟨hr.trans hr₁, by rw [hk, hk₁]; omega, hpe, fun i hi => ?_⟩
      rcases i with _ | i
      · simpa using hR₁
      · have := hR i (by omega)
        rw [hr₁, hk₁, show t.k + 1 + i = t.k + (i + 1) by omega, show a + 1 + i = a + (i + 1) by omega] at this
        exact this

end FlockSoundness.Refine
